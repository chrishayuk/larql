//! Diagnose the CPU↔Metal divergence that starts at generation step 1.
//!
//! By this point we've proven prefill is bit-exact between backends
//! (`test_cpu_metal_parity` passes at every layer, including with an
//! extra token appended). So the divergence at step 1 has to be in
//! Metal's KV-cached `decode_token` path: it produces a different
//! final hidden state than a fresh full prefill at the same sequence
//! length would produce.
//!
//! This tool isolates that:
//!
//!   A. CPU full prefill on `prompt_ids + [token_0]` — the reference,
//!      known to match Metal full prefill bit-exactly from the parity
//!      suite.
//!   B. Metal prefill on `prompt_ids` followed by `decode_token`
//!      (KV-cache append + attend + FFN on just the one new token).
//!
//! If A != B, `decode_token`'s output diverges from what a fresh
//! prefill at the same sequence length would compute — bug lives in
//! the KV-cached attention / FFN path (`crates/larql-compute/src/metal/
//! decode/mod.rs`).
//!
//! Usage:
//!   cargo run --release --features metal -p larql-inference \
//!     --example decode_vs_prefill -- <vindex-dir> [prompt]

extern crate blas_src;

use std::path::PathBuf;
use std::time::Instant;

use larql_compute::{ComputeBackend, DecodeBackend};
use larql_inference::layer_graph::generate::generate;
use larql_inference::layer_graph::CachedLayerGraph;
use larql_inference::wrap_chat_prompt;

const DEFAULT_EXAMPLE_KV_CACHE_MAX_SEQ: usize = 4096;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let vindex_path = PathBuf::from(
        args.next()
            .ok_or("usage: decode_vs_prefill <vindex-dir> [prompt]")?,
    );
    let prompt = args
        .next()
        .unwrap_or_else(|| "The capital of France is".to_string());

    if !vindex_path.is_dir() {
        return Err(format!("not a vindex dir: {}", vindex_path.display()).into());
    }

    // ── Load everything ────────────────────────────────────────────────────
    let mut cb = larql_vindex::SilentLoadCallbacks;
    let cfg = larql_vindex::load_vindex_config(&vindex_path)?;
    let tokenizer = larql_vindex::load_vindex_tokenizer(&vindex_path)?;
    let mut index = larql_vindex::VectorIndex::load_vindex(&vindex_path, &mut cb)?;
    index.load_attn_kquant(&vindex_path)?;
    index.load_interleaved_kquant(&vindex_path)?;
    let _ = index.load_lm_head_kquant(&vindex_path);

    // Separate weight handles so CPU's per-layer dequant inserts don't
    // race with Metal's forward on a shared ModelWeights.
    let mut w_metal = larql_vindex::load_model_weights_kquant(&vindex_path, &mut cb)?;
    let mut w_cpu = larql_vindex::load_model_weights_kquant(&vindex_path, &mut cb)?;

    let wrap = wrap_chat_prompt(&vindex_path, Some(cfg.model.as_str()), &prompt);
    let prompt_ids = larql_inference::encode_prompt(&tokenizer, &*w_metal.arch, &wrap.prompt)?;
    let num_layers = w_metal.num_layers;
    let hidden = w_metal.hidden_size;

    println!("━━━ decode_token vs full-prefill reference ─────────────────────────");
    println!("  vindex:     {}", vindex_path.display());
    println!("  model:      {}", cfg.model);
    println!("  family:     {}", cfg.family);
    println!("  prompt:     {prompt:?}");
    println!("  seq_len:    {}  (post-template)", prompt_ids.len());
    println!("  chat:       {}", wrap.note);
    println!();

    // ── Step 0: drive Metal through generate() to populate KV cache
    // and obtain the first-token argmax. We then append that token to
    // the prompt and have two ways to compute the next hidden state. ──
    let metal_backend =
        larql_compute_metal::MetalBackend::new().ok_or("Metal backend unavailable")?;
    let cached = CachedLayerGraph::from_residuals(Vec::new());

    // Warm-up then measured: first generate() call allocates KV buffers;
    // we want the measurement to reflect the fast path.
    let _ = generate(
        &mut w_metal,
        &tokenizer,
        &prompt_ids,
        1,
        &index,
        &metal_backend,
        &cached,
        0..num_layers,
    );
    // Re-run in a way that leaves the KV cache populated for the
    // prefill-only scope (max_tokens=1 → prefill runs, no decode loop).
    let r0 = generate(
        &mut w_metal,
        &tokenizer,
        &prompt_ids,
        1,
        &index,
        &metal_backend,
        &cached,
        0..num_layers,
    );
    let token_0_text = r0
        .tokens
        .first()
        .map(|(t, _)| t.clone())
        .unwrap_or_default();
    println!("  Metal prefill produced first token: {token_0_text:?}");

    // Re-encode (prompt + first-token-string) to get the appended id.
    // Using the rendered chat prompt + the decoded first token ensures
    // the id we re-feed is whatever Metal selected.
    let appended_prompt = format!("{}{}", wrap.prompt, token_0_text);
    let appended_ids =
        larql_inference::encode_prompt(&tokenizer, &*w_metal.arch, &appended_prompt)?;
    let appended_len = appended_ids.len();
    if appended_len <= prompt_ids.len() {
        return Err("failed to append step-0 token to prompt (tokeniser re-merged)".into());
    }
    let token_0_id = *appended_ids.last().unwrap();
    println!("  appended id: {token_0_id}  (new seq_len: {appended_len})");

    // ── A. CPU full prefill on (prompt + token_0) ──
    // This is the "fresh prefill" reference. We already know from the
    // parity suite that CPU full prefill matches Metal full prefill
    // bit-exactly at every layer, so this doubles as a Metal-prefill
    // reference without the tooling overhead of running Metal prefill
    // twice.
    let t0 = Instant::now();
    let cpu_hidden_full =
        larql_inference::vindex::predict_kquant_hidden(&mut w_cpu, &appended_ids, &index, None);
    let cpu_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let cpu_last = cpu_hidden_full
        .row(cpu_hidden_full.nrows().saturating_sub(1))
        .to_owned();
    println!(
        "  A) CPU full prefill({} tok) took {:>7.1} ms",
        appended_ids.len(),
        cpu_ms
    );

    // ── B. Metal prefill(prompt) + single decode_token(token_0). ──
    // `generate()` leaves the backend's KV cache in a usable state for
    // subsequent decode_token calls as long as we don't re-prefill.
    // Reset + re-prefill explicitly so the two paths are equivalent
    // up to the prefill; then run one decode for `token_0_id`.
    let layers = build_layers(&w_metal, &index, num_layers)?;
    let arch = &*w_metal.arch;
    // head_dim / num_q_heads / num_kv_heads / q_dim / kv_dim / rope used
    // to be passed to the pre-refactor `decode_token` / `prefill_kquant`;
    // the new APIs derive them from the `FullPipelineLayer` slice.

    metal_backend.reset_kv_cache();
    {
        let kv_shapes: Vec<(usize, usize)> = (0..num_layers)
            .map(|l| (arch.num_kv_heads_for_layer(l), arch.head_dim_for_layer(l)))
            .collect();
        metal_backend.preallocate_kv_cache_per_layer(&kv_shapes, DEFAULT_EXAMPLE_KV_CACHE_MAX_SEQ);
    }

    // Prefill: same path generate() uses internally.
    let embedded = larql_inference::forward::embed_tokens_pub(&w_metal, &prompt_ids);
    let prefill_x: Vec<f32> = embedded.as_slice().unwrap().to_vec();
    let softcap = arch.attn_logit_softcapping().unwrap_or(0.0);
    let qk_norm_val = arch.attn_q_norm_key(0).is_some();
    let intermediate = index.num_features(0);

    let t1 = Instant::now();
    let prefill_result = metal_backend
        .prefill_kquant(
            &layers,
            &prefill_x,
            hidden,
            intermediate,
            prompt_ids.len(),
            qk_norm_val,
            softcap,
        )
        .ok_or("Metal prefill_kquant returned None")?;
    let metal_prefill_ms = t1.elapsed().as_secs_f64() * 1000.0;

    // Decode one token. Returns the [hidden] output of the final
    // layer — same shape predict_kquant_hidden's last-row gives us.
    let dec_embed = larql_inference::forward::embed_tokens_pub(&w_metal, &[token_0_id]);
    let dec_x: Vec<f32> = dec_embed.row(0).to_vec();

    // Set up per-layer decode dump (gated inside the decode shader by
    // LARQL_DECODE_DUMP_LAYERS). We also need the CPU per-layer dumps
    // at seq_len=19 to compare against — drive CPU through a second
    // predict_kquant_hidden call with its dump env var set to the same dir.
    let decode_dump = tempfile::tempdir()?;
    let cpu_dump = tempfile::tempdir()?;
    std::env::set_var("LARQL_DECODE_DUMP_LAYERS", decode_dump.path());
    std::env::set_var("LARQL_CPU_DUMP_LAYERS", cpu_dump.path());

    // Use the trait method explicitly — the inherent
    // `MetalBackend::decode_token` has a different 11-arg shape that
    // exposes the KVCache directly; the trait form is the one
    // `layer_graph::generate` drives and the one we want to verify.
    let backend_dyn: &dyn ComputeBackend = &metal_backend;
    let t2 = Instant::now();
    let metal_decode = backend_dyn
        .decode_token(&layers, &dec_x, hidden, intermediate)
        .ok_or("Metal decode_token returned None")?;
    let metal_decode_ms = t2.elapsed().as_secs_f64() * 1000.0;

    // Re-run CPU full-prefill with the layer-dump env var set so we can
    // walk the two paths side by side. Cheap relative to the Metal
    // prefill we already paid for.
    let mut w_cpu2 = larql_vindex::load_model_weights_kquant(&vindex_path, &mut cb)?;
    let _ =
        larql_inference::vindex::predict_kquant_hidden(&mut w_cpu2, &appended_ids, &index, None);

    println!(
        "  B) Metal prefill({} tok) + decode(1 tok) took {:>5.1} + {:>5.1} ms",
        prompt_ids.len(),
        metal_prefill_ms,
        metal_decode_ms,
    );
    let _ = prefill_result; // last hidden not needed for the comparison

    // ── Compare A vs B ────────────────────────────────────────────────────
    if cpu_last.len() != metal_decode.len() {
        return Err(format!(
            "shape mismatch: cpu={} metal_decode={}",
            cpu_last.len(),
            metal_decode.len()
        )
        .into());
    }
    let cpu_slice = cpu_last.as_slice().unwrap();
    let (cos, max_abs, cpu_norm, mtl_norm) = compare(cpu_slice, &metal_decode);
    let rel = if mtl_norm > 0.0 {
        max_abs / mtl_norm
    } else {
        0.0
    };

    println!();
    println!("━━━ Hidden state at new position ────────────────────────────────────");
    println!("  cos_sim       {cos:.6}");
    println!(
        "  max|Δ|        {max_abs:.3e}  ({:.3}% of ||mtl||)",
        100.0 * rel
    );
    println!("  ||cpu||       {cpu_norm:.3}");
    println!("  ||mtl_decode|| {mtl_norm:.3}");

    if cos > 0.9999 && rel < 0.01 {
        println!();
        println!("  → decode_token matches full-prefill reference. Bug isn't here.");
    } else {
        println!();
        println!("  → decode_token's final hidden DIVERGES from full prefill.");
        println!("    Bug lives in `crates/larql-compute/src/metal/decode/mod.rs`");
        println!("    or its kernels (kv_attention, rope_at_pos, etc.).");
    }

    // ── Per-layer comparison. decode_token writes one hidden-size
    // vector per layer; CPU full-prefill writes [seq_len, hidden] —
    // we slice out the last-position row for the apples-to-apples
    // comparison. ──
    println!();
    println!("━━━ Per-layer compare: CPU last-row vs decode_token output ─────────");
    println!(
        "  {:>3}  {:>10}  {:>12}  {:>10}  {:>10}",
        "L", "cos_sim", "max_abs_Δ", "||cpu||", "||dec||"
    );
    for l in 0..num_layers {
        let dec_path = decode_dump.path().join(format!("decode_layer_{l:02}.f32"));
        let cpu_path = cpu_dump.path().join(format!("cpu_layer_{l:02}.f32"));
        let dec_v = match std::fs::read(&dec_path) {
            Ok(b) => b
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect::<Vec<f32>>(),
            Err(_) => {
                println!("  L{l:02}  <decode dump missing>");
                continue;
            }
        };
        let cpu_all = match std::fs::read(&cpu_path) {
            Ok(b) => b
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect::<Vec<f32>>(),
            Err(_) => {
                println!("  L{l:02}  <cpu dump missing>");
                continue;
            }
        };
        // CPU dump is [seq_len, hidden] flat; take the last position.
        let sl = cpu_all.len() / hidden;
        let cpu_last_row = &cpu_all[(sl - 1) * hidden..sl * hidden];
        if cpu_last_row.len() != dec_v.len() {
            println!(
                "  L{l:02}  <len mismatch: cpu_row={} dec={}>",
                cpu_last_row.len(),
                dec_v.len()
            );
            continue;
        }
        let (c, m, cn, mn) = compare(cpu_last_row, &dec_v);
        let rel = if mn > 0.0 { m / mn } else { 0.0 };
        let flag = if c < 0.9999 { " ←" } else { "" };
        println!(
            "  L{l:02}  {c:>10.6}  {m:>12.3e}  {cn:>10.3}  {mn:>10.3}  ({:.1}%){flag}",
            100.0 * rel
        );
    }

    Ok(())
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn build_layers<'a>(
    weights: &'a larql_inference::model::ModelWeights,
    index: &'a larql_vindex::VectorIndex,
    num_layers: usize,
) -> Result<Vec<larql_compute::FullPipelineLayer<'a>>, Box<dyn std::error::Error>> {
    let gate_index: &dyn larql_vindex::GateIndex = index;
    let (q4_ffn, ffn_is_q4k) = if let Some(mmap) = gate_index.interleaved_kquant_mmap_ref() {
        (Some(mmap), true)
    } else {
        (gate_index.interleaved_q4_mmap_ref(), false)
    };
    let q4_ffn_mmap = q4_ffn.ok_or("no Q4 FFN mmap available")?;
    let intermediate = gate_index.num_features(0);
    let hidden = weights.hidden_size;
    let q4_ffn_per_matrix = if ffn_is_q4k {
        (intermediate * hidden).div_ceil(256) * 144
    } else {
        intermediate * hidden / 32 * 18
    };
    let ffn_format = if ffn_is_q4k {
        larql_compute::QuantFormat::Q4_K
    } else {
        larql_compute::QuantFormat::Q4_0
    };
    Ok(
        larql_inference::layer_graph::pipeline_layer::build_pipeline_layers(
            weights,
            index,
            0..num_layers,
            q4_ffn_mmap,
            q4_ffn_per_matrix,
            ffn_format,
        ),
    )
}

fn compare(a: &[f32], b: &[f32]) -> (f32, f32, f32, f32) {
    let mut dot = 0.0f64;
    let mut an = 0.0f64;
    let mut bn = 0.0f64;
    let mut max_abs = 0.0f32;
    for i in 0..a.len() {
        let x = a[i] as f64;
        let y = b[i] as f64;
        dot += x * y;
        an += x * x;
        bn += y * y;
        let d = (a[i] - b[i]).abs();
        if d > max_abs {
            max_abs = d;
        }
    }
    let cos = if an > 0.0 && bn > 0.0 {
        (dot / (an.sqrt() * bn.sqrt())) as f32
    } else {
        0.0
    };
    (cos, max_abs, an.sqrt() as f32, bn.sqrt() as f32)
}
