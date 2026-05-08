//! Per-layer CPU↔Metal prefill parity regression guard.
//!
//! Companion to the architecture golden tests (`test_arch_golden`) —
//! the goldens check token-level output, this suite checks the
//! per-layer hidden state. Both are needed: a kernel can drift
//! quietly enough to keep the argmax token unchanged for a few steps
//! while compounding into a real bug at longer generations. The
//! per-layer check rejects "good output by luck".
//!
//! Driven entirely through [`larql_inference::residual_diff`] —
//! captures both backends in memory, compares with [`compare_captures`]
//! at the [`ParityThreshold::tight`] preset, asserts via
//! [`ParityReport::assert_clean`]. No tempdirs, no env vars in the
//! test body. The capture module owns that plumbing.
//!
//! ### Caught regressions
//!
//! - **Metal `fused_attention` head_dim>256 bug** — `tg_q[256..512]`
//!   left uninitialised, dropped attention magnitude ~6% per global
//!   layer. Compounded to cos≈0.91 by L59 on Gemma 4 31B; this suite
//!   would surface it at L5 (the first global layer) within the cos
//!   threshold of `tight()`.
//!
//! ### Skip semantics
//!
//! Vindexes can be tens of GB; missing ones print a skip note and
//! return `Ok` so CI stays green. `LARQL_ARCH_STRICT=1` flips skips
//! to hard failures (useful locally to confirm the test actually ran).

#![cfg(any(feature = "metal", feature = "cuda"))]

#[cfg(feature = "metal")]
mod metal_prefill {
    use std::path::PathBuf;

    use larql_inference::residual_diff::{compare_captures, ParityThreshold, ResidualCapture};
    use larql_inference::wrap_chat_prompt;
    use larql_vindex::{
        load_model_weights_q4k, load_vindex_config, load_vindex_tokenizer, QuantFormat,
        SilentLoadCallbacks, VectorIndex,
    };

    struct ParityCase {
        name: &'static str,
        vindex_name: &'static str,
    }

    /// One row per arch we want covered. `gemma-4-26B-A4B-it` is omitted
    /// because its Metal MoE prefill goes through `decode_token` per-position
    /// (`metal/trait_impl.rs:215-229`), bypassing the per-layer dump that
    /// `prefill_q4` populates. Re-add when MoE prefill batches.
    const CASES: &[ParityCase] = &[
        ParityCase {
            name: "gemma3-4b-it",
            vindex_name: "gemma3-4b-q4k-v2",
        },
        ParityCase {
            name: "gemma4-31b-it (dense)",
            vindex_name: "gemma4-31b-q4k",
        },
        ParityCase {
            name: "llama2-7b-hf (base)",
            vindex_name: "llama2-7b-q4k",
        },
        ParityCase {
            name: "mistral-7b-v0.1 (base)",
            vindex_name: "mistral-7b-v0.1-q4k",
        },
    ];

    fn find_vindex(name: &str) -> Option<PathBuf> {
        let filename = format!("{name}.vindex");
        if let Ok(env_path) = std::env::var(format!(
            "LARQL_VINDEX_{}",
            name.to_uppercase().replace('-', "_")
        )) {
            let p = PathBuf::from(env_path);
            if p.is_dir() {
                return Some(p);
            }
        }
        let chris_models = PathBuf::from("/Users/christopherhay/chris-models").join(&filename);
        if chris_models.is_dir() {
            return Some(chris_models);
        }
        let home = std::env::var("HOME").ok()?;
        [
            PathBuf::from(&home)
                .join(".cache/larql/local")
                .join(&filename),
            PathBuf::from("output").join(&filename),
        ]
        .into_iter()
        .find(|p| p.is_dir())
    }

    fn strict_mode() -> bool {
        matches!(
            std::env::var("LARQL_ARCH_STRICT").ok().as_deref(),
            Some("1") | Some("true")
        )
    }

    fn run_case(case: &ParityCase) -> Result<(), String> {
        let Some(vindex_path) = find_vindex(case.vindex_name) else {
            if strict_mode() {
                return Err(format!(
                    "[{}] vindex `{}` not found (LARQL_ARCH_STRICT=1)",
                    case.name, case.vindex_name
                ));
            }
            eprintln!(
                "[{}] skip: vindex `{}` not found in cache",
                case.name, case.vindex_name
            );
            return Ok(());
        };

        let mut cb = SilentLoadCallbacks;
        let cfg =
            load_vindex_config(&vindex_path).map_err(|e| format!("load_vindex_config: {e}"))?;
        if cfg.quant != QuantFormat::Q4K {
            return Err(format!("expected Q4K vindex (got {:?})", cfg.quant));
        }
        let tokenizer = load_vindex_tokenizer(&vindex_path)
            .map_err(|e| format!("load_vindex_tokenizer: {e}"))?;
        let mut q4_index = VectorIndex::load_vindex(&vindex_path, &mut cb)
            .map_err(|e| format!("load vindex: {e}"))?;
        q4_index
            .load_attn_q4k(&vindex_path)
            .map_err(|e| format!("load_attn_q4k: {e}"))?;
        q4_index
            .load_interleaved_q4k(&vindex_path)
            .map_err(|e| format!("load_interleaved_q4k: {e}"))?;
        let _ = q4_index.load_lm_head_q4(&vindex_path);

        // Disjoint weight handles — CPU's per-layer dequant inserts into
        // `weights.tensors`, which would race if both backends shared a
        // single ModelWeights.
        let mut w_metal = load_model_weights_q4k(&vindex_path, &mut cb)
            .map_err(|e| format!("load weights (metal): {e}"))?;
        let mut w_cpu = load_model_weights_q4k(&vindex_path, &mut cb)
            .map_err(|e| format!("load weights (cpu): {e}"))?;

        let prompt = "The capital of France is";
        let wrap = wrap_chat_prompt(&vindex_path, Some(cfg.model.as_str()), prompt);
        let token_ids = larql_inference::encode_prompt(&tokenizer, &*w_metal.arch, &wrap.prompt)
            .map_err(|e| format!("encode_prompt: {e}"))?;

        let metal_backend = larql_compute::metal::MetalBackend::new()
            .ok_or("Metal backend unavailable — rebuild with --features metal")?;

        let metal =
            ResidualCapture::metal_prefill(&mut w_metal, &token_ids, &q4_index, &metal_backend)?;
        let cpu = ResidualCapture::cpu_prefill(&mut w_cpu, &token_ids, &q4_index)?;

        if cpu.num_layers() != metal.num_layers() {
            return Err(format!(
                "[{}] backend produced different layer counts: cpu={}, metal={}",
                case.name,
                cpu.num_layers(),
                metal.num_layers()
            ));
        }

        let report = compare_captures(&cpu, &metal, ParityThreshold::tight());
        report
            .assert_clean()
            .map_err(|e| format!("[{}] {e}", case.name))?;
        eprintln!(
            "[{}] parity OK across {} layers (rel max_abs ≤ {:.1}%)",
            case.name,
            cpu.num_layers(),
            100.0 * ParityThreshold::tight().rel_max_abs
        );
        Ok(())
    }

    #[test]
    fn parity_gemma3_4b_prefill() {
        run_case(&CASES[0]).unwrap_or_else(|e| panic!("{e}"));
    }

    #[test]
    fn parity_gemma4_31b_dense_prefill() {
        run_case(&CASES[1]).unwrap_or_else(|e| panic!("{e}"));
    }

    #[test]
    fn parity_llama2_7b_prefill() {
        run_case(&CASES[2]).unwrap_or_else(|e| panic!("{e}"));
    }

    #[test]
    fn parity_mistral_7b_prefill() {
        run_case(&CASES[3]).unwrap_or_else(|e| panic!("{e}"));
    }
}

#[cfg(feature = "cuda")]
mod cuda_fused_attention {
    use larql_compute::cuda::attn::{
        fused_decode_attention, qkv_rms_proj, FusedDecodeAttentionOpts, QkvProjDims,
    };
    use larql_compute::cuda::CudaBackend;

    fn gpu_or_skip() -> Option<CudaBackend> {
        if std::env::var("LARQL_CUDA_AVAILABLE").ok().as_deref() != Some("1") {
            return None;
        }
        CudaBackend::new().ok()
    }

    fn synth(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s & 0xFF_FFFF) as f32 / 8_388_608.0) - 1.0
            })
            .collect()
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max)
    }

    fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
        let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
        let inv = 1.0 / (mean_sq + eps).sqrt();
        x.iter().zip(weight).map(|(v, w)| v * inv * w).collect()
    }

    fn matvec_rows(w: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut out = vec![0.0; rows];
        for r in 0..rows {
            for c in 0..cols {
                out[r] += w[r * cols + c] * x[c];
            }
        }
        out
    }

    fn norm_rope(
        x: &[f32],
        weight: &[f32],
        eps: f32,
        pos: usize,
        rope_base: f32,
        rotary_dim: usize,
    ) -> Vec<f32> {
        let head_dim = x.len();
        let mut out = rms_norm(x, weight, eps);
        let rdim = if rotary_dim == 0 {
            head_dim
        } else {
            rotary_dim.min(head_dim)
        };
        let hdim = rdim / 2;
        for d in 0..hdim {
            let freq = 1.0 / rope_base.powf((2 * d) as f32 / rdim as f32);
            let angle = pos as f32 * freq;
            let (sin_a, cos_a) = angle.sin_cos();
            let re = out[d];
            let im = out[d + hdim];
            out[d] = re * cos_a - im * sin_a;
            out[d + hdim] = re * sin_a + im * cos_a;
        }
        out
    }

    #[test]
    fn parity_cuda_fused_qkv_and_attention() {
        let Some(backend) = gpu_or_skip() else { return };

        let hidden = 32;
        let q_heads = 2;
        let kv_heads = 1;
        let head_dim = 16;
        let q_dim = q_heads * head_dim;
        let kv_dim = kv_heads * head_dim;
        let eps = 1e-6;
        let x = synth(hidden, 0xCA01);
        let norm: Vec<f32> = synth(hidden, 0xCA02)
            .into_iter()
            .map(|v| 1.0 + 0.05 * v)
            .collect();
        let wq = synth(q_dim * hidden, 0xCA03);
        let wk = synth(kv_dim * hidden, 0xCA04);
        let wv = synth(kv_dim * hidden, 0xCA05);

        let qkv = qkv_rms_proj(
            &backend,
            &x,
            &norm,
            &wq,
            &wk,
            &wv,
            QkvProjDims {
                hidden,
                q_dim,
                kv_dim,
            },
            eps,
            0.0,
        )
        .expect("qkv_rms_proj");

        let x_norm = rms_norm(&x, &norm, eps);
        assert!(max_abs_diff(&qkv.q, &matvec_rows(&wq, &x_norm, q_dim, hidden)) <= 1e-3);
        assert!(max_abs_diff(&qkv.k, &matvec_rows(&wk, &x_norm, kv_dim, hidden)) <= 1e-3);
        assert!(max_abs_diff(&qkv.v, &matvec_rows(&wv, &x_norm, kv_dim, hidden)) <= 1e-3);

        let opts = FusedDecodeAttentionOpts {
            num_q_heads: q_heads,
            num_kv_heads: kv_heads,
            head_dim,
            pos: 2,
            max_seq: 4,
            rotary_dim: 8,
            rope_base: 10_000.0,
            eps,
            qk_norm_offset: 0.0,
            attn_scale: 1.0 / (head_dim as f32).sqrt(),
            softcap: 50.0,
        };
        let q_norm = vec![1.0; head_dim];
        let k_norm = vec![1.0; head_dim];
        let mut k_cache = synth(opts.max_seq * kv_heads * head_dim, 0xCA06);
        let mut v_cache = synth(opts.max_seq * kv_heads * head_dim, 0xCA07);
        for idx in opts.pos * kv_heads * head_dim..k_cache.len() {
            k_cache[idx] = 0.0;
            v_cache[idx] = 0.0;
        }

        let cuda = fused_decode_attention(
            &backend,
            &qkv.q,
            &qkv.k,
            &qkv.v,
            &k_cache,
            &v_cache,
            Some(&q_norm),
            Some(&k_norm),
            opts,
        )
        .expect("fused_decode_attention");

        let mut k_ref = k_cache;
        let mut v_ref = v_cache;
        let k_rot = norm_rope(
            &qkv.k,
            &k_norm,
            eps,
            opts.pos,
            opts.rope_base,
            opts.rotary_dim,
        );
        for d in 0..head_dim {
            k_ref[opts.pos * head_dim + d] = k_rot[d];
            v_ref[opts.pos * head_dim + d] = qkv.v[d];
        }
        let mut out_ref = vec![0.0; q_heads * head_dim];
        for qh in 0..q_heads {
            let q_head = &qkv.q[qh * head_dim..(qh + 1) * head_dim];
            let q_rot = norm_rope(
                q_head,
                &q_norm,
                eps,
                opts.pos,
                opts.rope_base,
                opts.rotary_dim,
            );
            let mut scores = vec![0.0; opts.pos + 1];
            let mut max = f32::NEG_INFINITY;
            for (j, score) in scores.iter_mut().enumerate() {
                let mut dot = 0.0;
                for d in 0..head_dim {
                    dot += q_rot[d] * k_ref[j * head_dim + d];
                }
                let mut s = dot * opts.attn_scale;
                s = opts.softcap * (s / opts.softcap).tanh();
                *score = s;
                max = max.max(s);
            }
            let mut sum = 0.0;
            for score in &mut scores {
                *score = (*score - max).exp();
                sum += *score;
            }
            for d in 0..head_dim {
                for j in 0..=opts.pos {
                    out_ref[qh * head_dim + d] += scores[j] / sum * v_ref[j * head_dim + d];
                }
            }
        }
        assert!(max_abs_diff(&out_ref, &cuda.out) <= 2e-3);
        assert!(max_abs_diff(&k_ref, &cuda.k_cache) <= 1e-3);
        assert!(max_abs_diff(&v_ref, &cuda.v_cache) <= 1e-6);
    }
}
