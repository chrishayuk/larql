//! Q4_K pipeline benchmark: compare Q4_K fused QKV vs Q8 fused QKV.
//!
//! Exercises the new fused Q4_K QKV shader through the full_pipeline_q4 path.
//! Usage: cargo run --release --features metal -p larql-compute --example bench_q4k_pipeline

extern crate blas_src;

fn main() {
    #[cfg(not(feature = "metal"))]
    {
        println!("Run with --features metal");
    }

    #[cfg(feature = "metal")]
    {
        use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k, quantize_to_q8};
        use larql_compute::prelude::*;
        use std::time::Instant;

        let metal = larql_compute::metal::MetalBackend::new().expect("Metal required");

        let hidden = 2560usize;
        let inter = 10240usize;
        let num_q_heads = 8usize;
        let num_kv_heads = 4usize;
        let head_dim = 320usize;
        let q_dim = num_q_heads * head_dim; // 2560
        let kv_dim = num_kv_heads * head_dim; // 1280
        let num_layers = 21usize;
        let n = 10;

        println!("=== Q4_K vs Q8 Pipeline Benchmark ===");
        println!("{num_layers} layers, hidden={hidden}, q_dim={q_dim}, kv_dim={kv_dim}\n");

        // Build Q4_K attention weights + Q4_0 FFN weights
        struct LayerData {
            wq_q4k: Vec<u8>,
            wk_q4k: Vec<u8>,
            wv_q4k: Vec<u8>,
            wo_q4k: Vec<u8>,
            wq_q8: Vec<u8>,
            wk_q8: Vec<u8>,
            wv_q8: Vec<u8>,
            wo_q8: Vec<u8>,
            wq_q8s: Vec<f32>,
            wk_q8s: Vec<f32>,
            wv_q8s: Vec<f32>,
            wo_q8s: Vec<f32>,
            gate_q4: Vec<u8>,
            up_q4: Vec<u8>,
            down_q4: Vec<u8>,
            norm: Vec<f32>,
        }

        let mut layers_data: Vec<LayerData> = Vec::new();
        for l in 0..num_layers {
            // Generate synthetic weight matrices
            let wq_f32: Vec<f32> = (0..q_dim * hidden)
                .map(|i| ((i + l * 1000) as f32 * 0.0001).cos())
                .collect();
            let wk_f32: Vec<f32> = (0..kv_dim * hidden)
                .map(|i| ((i + l * 2000) as f32 * 0.0002).sin())
                .collect();
            let wv_f32: Vec<f32> = (0..kv_dim * hidden)
                .map(|i| ((i + l * 3000) as f32 * 0.0003).cos())
                .collect();
            let wo_f32: Vec<f32> = (0..hidden * q_dim)
                .map(|i| ((i + l * 4000) as f32 * 0.0004).sin())
                .collect();
            let g_f32: Vec<f32> = (0..inter * hidden)
                .map(|i| ((i + l * 5000) as f32 * 0.0001).cos())
                .collect();
            let u_f32: Vec<f32> = (0..inter * hidden)
                .map(|i| ((i + l * 6000) as f32 * 0.0002).sin())
                .collect();
            let d_f32: Vec<f32> = (0..hidden * inter)
                .map(|i| ((i + l * 7000) as f32 * 0.0003).cos())
                .collect();

            // Pad to multiples of 256 for Q4_K
            fn pad_for_q4k(data: &[f32]) -> Vec<f32> {
                let padded_len = data.len().div_ceil(256) * 256;
                let mut out = data.to_vec();
                out.resize(padded_len, 0.0);
                out
            }

            // Q4_K quantization (need multiples of 256)
            let wq_q4k = quantize_q4_k(&pad_for_q4k(&wq_f32));
            let wk_q4k = quantize_q4_k(&pad_for_q4k(&wk_f32));
            let wv_q4k = quantize_q4_k(&pad_for_q4k(&wv_f32));
            let wo_q4k = quantize_q4_k(&pad_for_q4k(&wo_f32));

            // Q8 quantization for comparison (need flattened row-major, block of 32)
            let (wq_q8, wq_q8s) = quantize_to_q8(&wq_f32);
            let (wk_q8, wk_q8s) = quantize_to_q8(&wk_f32);
            let (wv_q8, wv_q8s) = quantize_to_q8(&wv_f32);
            let (wo_q8, wo_q8s) = quantize_to_q8(&wo_f32);

            // Q4_0 for FFN (multiples of 32)
            let gate_q4 = quantize_q4_0(&g_f32);
            let up_q4 = quantize_q4_0(&u_f32);
            let down_q4 = quantize_q4_0(&d_f32);

            let norm = vec![1.0f32; hidden];

            layers_data.push(LayerData {
                wq_q4k,
                wk_q4k,
                wv_q4k,
                wo_q4k,
                wq_q8: wq_q8.iter().map(|&x| x as u8).collect(),
                wk_q8: wk_q8.iter().map(|&x| x as u8).collect(),
                wv_q8: wv_q8.iter().map(|&x| x as u8).collect(),
                wo_q8: wo_q8.iter().map(|&x| x as u8).collect(),
                wq_q8s,
                wk_q8s,
                wv_q8s,
                wo_q8s,
                gate_q4,
                up_q4,
                down_q4,
                norm,
            });
        }

        let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.001).sin()).collect();

        // ── Q4_K pipeline ──
        let q4k_layers: Vec<larql_compute::FullPipelineLayer> = layers_data
            .iter()
            .map(|ld| larql_compute::FullPipelineLayer {
                wq: larql_compute::QuantWeight {
                    data: &ld.wq_q4k,
                    scales: None,
                    format: larql_compute::QuantFormat::Q4_K,
                },
                wk: larql_compute::QuantWeight {
                    data: &ld.wk_q4k,
                    scales: None,
                    format: larql_compute::QuantFormat::Q4_K,
                },
                wv: larql_compute::QuantWeight {
                    data: &ld.wv_q4k,
                    scales: None,
                    format: larql_compute::QuantFormat::Q4_K,
                },
                wo: larql_compute::QuantWeight {
                    data: &ld.wo_q4k,
                    scales: None,
                    format: larql_compute::QuantFormat::Q4_K,
                },
                gate: larql_compute::QuantWeight {
                    data: &ld.gate_q4,
                    scales: None,
                    format: larql_compute::QuantFormat::Q4_0,
                },
                up: larql_compute::QuantWeight {
                    data: &ld.up_q4,
                    scales: None,
                    format: larql_compute::QuantFormat::Q4_0,
                },
                down: larql_compute::QuantWeight {
                    data: &ld.down_q4,
                    scales: None,
                    format: larql_compute::QuantFormat::Q4_0,
                },
                input_norm: &ld.norm,
                post_attn_norm: &ld.norm,
                pre_ffn_norm: None,
                post_ffn_norm: None,
                norm_offset: 1.0,
                has_post_norms: false,
                activation: larql_compute::Activation::Silu,
                qk_norm_offset: 0.0,
                eps: 1e-6,
                norm_type: larql_compute::NormType::RmsNorm,
                ffn_type: larql_compute::FfnType::Gated,
                attn_scale: 1.0 / (head_dim as f32).sqrt(),
                head_dim,
                num_q_heads,
                num_kv_heads,
                rope_base: 10000.0,
                rotary_dim: 0,
                sliding_window: 0,
                has_v_norm: false,
                layer_scalar: 0.0,
                input_norm_bias: None,
                post_attn_norm_bias: None,
                q_norm_weight: None,
                k_norm_weight: None,
                ffn_up_bias: None,
                ffn_down_bias: None,
                moe: None,
                ffn_is_remote: false,
                moe_combined_output_norm: false,
                moe_outer_post_norm: None,
            })
            .collect();

        // Warmup
        let _ = metal.full_pipeline_q4(
            &q4k_layers,
            &x,
            hidden,
            inter,
            q_dim,
            kv_dim,
            1,
            num_q_heads,
            num_kv_heads,
            head_dim,
            10000.0,
            false,
            0.0,
        );

        let t0 = Instant::now();
        for _ in 0..n {
            let _ = metal.full_pipeline_q4(
                &q4k_layers,
                &x,
                hidden,
                inter,
                q_dim,
                kv_dim,
                1,
                num_q_heads,
                num_kv_heads,
                head_dim,
                10000.0,
                false,
                0.0,
            );
        }
        let q4k_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;

        // ── Q8 pipeline ──
        let q8_layers: Vec<larql_compute::FullPipelineLayer> = layers_data
            .iter()
            .map(|ld| larql_compute::FullPipelineLayer {
                wq: larql_compute::QuantWeight {
                    data: &ld.wq_q8,
                    scales: Some(&ld.wq_q8s),
                    format: larql_compute::QuantFormat::Q8_0,
                },
                wk: larql_compute::QuantWeight {
                    data: &ld.wk_q8,
                    scales: Some(&ld.wk_q8s),
                    format: larql_compute::QuantFormat::Q8_0,
                },
                wv: larql_compute::QuantWeight {
                    data: &ld.wv_q8,
                    scales: Some(&ld.wv_q8s),
                    format: larql_compute::QuantFormat::Q8_0,
                },
                wo: larql_compute::QuantWeight {
                    data: &ld.wo_q8,
                    scales: Some(&ld.wo_q8s),
                    format: larql_compute::QuantFormat::Q8_0,
                },
                gate: larql_compute::QuantWeight {
                    data: &ld.gate_q4,
                    scales: None,
                    format: larql_compute::QuantFormat::Q4_0,
                },
                up: larql_compute::QuantWeight {
                    data: &ld.up_q4,
                    scales: None,
                    format: larql_compute::QuantFormat::Q4_0,
                },
                down: larql_compute::QuantWeight {
                    data: &ld.down_q4,
                    scales: None,
                    format: larql_compute::QuantFormat::Q4_0,
                },
                input_norm: &ld.norm,
                post_attn_norm: &ld.norm,
                pre_ffn_norm: None,
                post_ffn_norm: None,
                norm_offset: 1.0,
                has_post_norms: false,
                activation: larql_compute::Activation::Silu,
                qk_norm_offset: 0.0,
                eps: 1e-6,
                norm_type: larql_compute::NormType::RmsNorm,
                ffn_type: larql_compute::FfnType::Gated,
                attn_scale: 1.0 / (head_dim as f32).sqrt(),
                head_dim,
                num_q_heads,
                num_kv_heads,
                rope_base: 10000.0,
                rotary_dim: 0,
                sliding_window: 0,
                has_v_norm: false,
                layer_scalar: 0.0,
                input_norm_bias: None,
                post_attn_norm_bias: None,
                q_norm_weight: None,
                k_norm_weight: None,
                ffn_up_bias: None,
                ffn_down_bias: None,
                moe: None,
                ffn_is_remote: false,
                moe_combined_output_norm: false,
                moe_outer_post_norm: None,
            })
            .collect();

        // Warmup
        let _ = metal.full_pipeline_q4(
            &q8_layers,
            &x,
            hidden,
            inter,
            q_dim,
            kv_dim,
            1,
            num_q_heads,
            num_kv_heads,
            head_dim,
            10000.0,
            false,
            0.0,
        );

        let t0 = Instant::now();
        for _ in 0..n {
            let _ = metal.full_pipeline_q4(
                &q8_layers,
                &x,
                hidden,
                inter,
                q_dim,
                kv_dim,
                1,
                num_q_heads,
                num_kv_heads,
                head_dim,
                10000.0,
                false,
                0.0,
            );
        }
        let q8_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;

        // ── FFN-only baseline ──
        let layers_q4_refs: Vec<(&[u8], &[u8], &[u8])> = layers_data
            .iter()
            .map(|ld| {
                (
                    ld.gate_q4.as_slice(),
                    ld.up_q4.as_slice(),
                    ld.down_q4.as_slice(),
                )
            })
            .collect();
        let _ = metal.multi_layer_q4_ffn(&layers_q4_refs, &x, inter, hidden);
        let t0 = Instant::now();
        for _ in 0..n {
            let _ = metal.multi_layer_q4_ffn(&layers_q4_refs, &x, inter, hidden);
        }
        let ffn_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;

        let q4k_attn = q4k_ms - ffn_ms;
        let q8_attn = q8_ms - ffn_ms;
        let q4k_tps = 1000.0 / q4k_ms;
        let q8_tps = 1000.0 / q8_ms;

        println!("--- Full pipeline (attn + FFN, {num_layers} layers, 1 cmd buffer) ---\n");
        println!(
            "  Q4_K attn + Q4_0 FFN:  {q4k_ms:>6.1}ms  ({q4k_tps:.0} tok/s)  attn={q4k_attn:.1}ms"
        );
        println!(
            "  Q8   attn + Q4_0 FFN:  {q8_ms:>6.1}ms  ({q8_tps:.0} tok/s)  attn={q8_attn:.1}ms"
        );
        println!("  FFN-only baseline:     {ffn_ms:>6.1}ms");
        println!("  Q4_K attn speedup:     {:.2}x", q8_attn / q4k_attn);
        println!();

        let q4k_projected = q4k_ms + 1.0 + 1.0; // + KV attend + logits
        let q8_projected = q8_ms + 1.0 + 1.0;
        println!("  Projected decode (+ KV cache + logits):");
        println!(
            "    Q4_K: {q4k_projected:.0}ms → {:.0} tok/s",
            1000.0 / q4k_projected
        );
        println!(
            "    Q8:   {q8_projected:.0}ms → {:.0} tok/s",
            1000.0 / q8_projected
        );
        println!("    Ollama: ~10ms → ~100 tok/s");

        // Data size comparison
        let q4k_qkv_bytes =
            layers_data[0].wq_q4k.len() + layers_data[0].wk_q4k.len() + layers_data[0].wv_q4k.len();
        let q8_qkv_bytes =
            layers_data[0].wq_q8.len() + layers_data[0].wk_q8.len() + layers_data[0].wv_q8.len();
        println!(
            "\n  QKV data per layer: Q4_K={:.1}MB  Q8={:.1}MB  ratio={:.2}x",
            q4k_qkv_bytes as f64 / 1e6,
            q8_qkv_bytes as f64 / 1e6,
            q8_qkv_bytes as f64 / q4k_qkv_bytes as f64
        );

        println!("\n=== Done ===");
    }
}
