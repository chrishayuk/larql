//! Q4_KF decode benchmark: pre-baked scales vs Q4_K vs Q8.
//!
//! Usage: cargo run --release --features metal -p larql-compute --example bench_q4kf_decode

extern crate blas_src;

fn main() {
    #[cfg(not(feature = "metal"))]
    {
        println!("Run with --features metal");
    }

    #[cfg(feature = "metal")]
    {
        use larql_compute::cpu::ops::q4_common::{q4k_to_q4kf, quantize_q4_0, quantize_q4_k};
        use larql_compute::prelude::*;
        use std::time::Instant;

        let metal_raw = larql_compute::metal::MetalBackend::new().expect("Metal required");
        let metal: &dyn ComputeBackend = &metal_raw;

        let hidden = 2560usize;
        let inter = 10240usize;
        let num_q_heads = 8usize;
        let num_kv_heads = 4usize;
        let head_dim = 320usize;
        let q_dim = num_q_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let num_layers = 34usize; // Gemma3 4B actual layer count
        let n = 20;

        println!("=== Q4_KF vs Q4_K vs Q8 Decode Benchmark ===");
        println!("{num_layers} layers, hidden={hidden}\n");

        fn pad256(data: &[f32]) -> Vec<f32> {
            let padded_len = data.len().div_ceil(256) * 256;
            let mut out = data.to_vec();
            out.resize(padded_len, 0.0);
            out
        }

        struct LayerData {
            wq_q4k: Vec<u8>,
            wk_q4k: Vec<u8>,
            wv_q4k: Vec<u8>,
            wo_q4k: Vec<u8>,
            wq_q4kf: Vec<u8>,
            wk_q4kf: Vec<u8>,
            wv_q4kf: Vec<u8>,
            wo_q4kf: Vec<u8>,
            wq_gguf: Vec<u8>,
            wk_gguf: Vec<u8>,
            wv_gguf: Vec<u8>,
            wo_gguf: Vec<u8>,
            gate_q4: Vec<u8>,
            up_q4: Vec<u8>,
            down_q4: Vec<u8>,
            norm: Vec<f32>,
        }

        let mut layers_data: Vec<LayerData> = Vec::new();
        for l in 0..num_layers {
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

            let wq_q4k = quantize_q4_k(&pad256(&wq_f32));
            let wk_q4k = quantize_q4_k(&pad256(&wk_f32));
            let wv_q4k = quantize_q4_k(&pad256(&wv_f32));
            let wo_q4k = quantize_q4_k(&pad256(&wo_f32));

            // Convert Q4_K → Q4_KF (pre-bake scales)
            let q_rows = q_dim;
            let kv_rows = kv_dim;
            let o_rows = hidden;
            let wq_q4kf = q4k_to_q4kf(&wq_q4k, q_rows, hidden);
            let wk_q4kf = q4k_to_q4kf(&wk_q4k, kv_rows, hidden);
            let wv_q4kf = q4k_to_q4kf(&wv_q4k, kv_rows, hidden);
            let wo_q4kf = q4k_to_q4kf(&wo_q4k, o_rows, q_dim);

            // GGUF Q4_K (144-byte blocks, packed scales+mins)
            let wq_gguf = quantize_q4_k(&pad256(&wq_f32));
            let wk_gguf = quantize_q4_k(&pad256(&wk_f32));
            let wv_gguf = quantize_q4_k(&pad256(&wv_f32));
            let wo_gguf = quantize_q4_k(&pad256(&wo_f32));

            layers_data.push(LayerData {
                wq_q4k,
                wk_q4k,
                wv_q4k,
                wo_q4k,
                wq_q4kf,
                wk_q4kf,
                wv_q4kf,
                wo_q4kf,
                wq_gguf,
                wk_gguf,
                wv_gguf,
                wo_gguf,
                gate_q4: quantize_q4_0(&g_f32),
                up_q4: quantize_q4_0(&u_f32),
                down_q4: quantize_q4_0(&d_f32),
                norm: vec![1.0f32; hidden],
            });
        }

        let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.001).sin()).collect();

        // ── Q4_KF decode ──
        let q4kf_layers: Vec<larql_compute::FullPipelineLayer> = layers_data
            .iter()
            .map(|ld| larql_compute::FullPipelineLayer {
                wq: larql_compute::QuantWeight {
                    data: &ld.wq_q4kf,
                    scales: None,
                    format: larql_compute::QuantFormat::Q4_KF,
                },
                wk: larql_compute::QuantWeight {
                    data: &ld.wk_q4kf,
                    scales: None,
                    format: larql_compute::QuantFormat::Q4_KF,
                },
                wv: larql_compute::QuantWeight {
                    data: &ld.wv_q4kf,
                    scales: None,
                    format: larql_compute::QuantFormat::Q4_KF,
                },
                wo: larql_compute::QuantWeight {
                    data: &ld.wo_q4kf,
                    scales: None,
                    format: larql_compute::QuantFormat::Q4_KF,
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

        metal.reset_kv_cache();
        for _ in 0..5 {
            let _ = metal.decode_token(
                &q4kf_layers,
                &x,
                hidden,
                inter,
                q_dim,
                kv_dim,
                num_q_heads,
                num_kv_heads,
                head_dim,
                10000.0,
            );
        }
        let t0 = Instant::now();
        for _ in 0..n {
            let _ = metal.decode_token(
                &q4kf_layers,
                &x,
                hidden,
                inter,
                q_dim,
                kv_dim,
                num_q_heads,
                num_kv_heads,
                head_dim,
                10000.0,
            );
        }
        let q4kf_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;

        // ── Q4_K decode ──
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

        metal.reset_kv_cache();
        for _ in 0..5 {
            let _ = metal.decode_token(
                &q4k_layers,
                &x,
                hidden,
                inter,
                q_dim,
                kv_dim,
                num_q_heads,
                num_kv_heads,
                head_dim,
                10000.0,
            );
        }
        let t0 = Instant::now();
        for _ in 0..n {
            let _ = metal.decode_token(
                &q4k_layers,
                &x,
                hidden,
                inter,
                q_dim,
                kv_dim,
                num_q_heads,
                num_kv_heads,
                head_dim,
                10000.0,
            );
        }
        let q4k_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;

        // ── GGUF Q4_K decode (144-byte blocks, llama.cpp kernel) ──
        let gguf_layers: Vec<larql_compute::FullPipelineLayer> = layers_data
            .iter()
            .map(|ld| larql_compute::FullPipelineLayer {
                wq: larql_compute::QuantWeight {
                    data: &ld.wq_gguf,
                    scales: None,
                    format: larql_compute::QuantFormat::Q4_KF,
                },
                wk: larql_compute::QuantWeight {
                    data: &ld.wk_gguf,
                    scales: None,
                    format: larql_compute::QuantFormat::Q4_KF,
                },
                wv: larql_compute::QuantWeight {
                    data: &ld.wv_gguf,
                    scales: None,
                    format: larql_compute::QuantFormat::Q4_KF,
                },
                wo: larql_compute::QuantWeight {
                    data: &ld.wo_gguf,
                    scales: None,
                    format: larql_compute::QuantFormat::Q4_KF,
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

        metal.reset_kv_cache();
        for _ in 0..5 {
            let _ = metal.decode_token(
                &gguf_layers,
                &x,
                hidden,
                inter,
                q_dim,
                kv_dim,
                num_q_heads,
                num_kv_heads,
                head_dim,
                10000.0,
            );
        }
        let t0 = Instant::now();
        for _ in 0..n {
            let _ = metal.decode_token(
                &gguf_layers,
                &x,
                hidden,
                inter,
                q_dim,
                kv_dim,
                num_q_heads,
                num_kv_heads,
                head_dim,
                10000.0,
            );
        }
        let gguf_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;

        println!("--- decode_token ({num_layers} layers, KV cache) ---\n");
        println!(
            "  GGUF Q4_K (llama):  {gguf_ms:>6.1}ms  ({:.0} tok/s)",
            1000.0 / gguf_ms
        );
        println!(
            "  Q4_KF (pre-baked):  {q4kf_ms:>6.1}ms  ({:.0} tok/s)",
            1000.0 / q4kf_ms
        );
        println!(
            "  Q4_K  (runtime):    {q4k_ms:>6.1}ms  ({:.0} tok/s)",
            1000.0 / q4k_ms
        );
        println!("  Q4_KF speedup:      {:.2}x vs Q4_K", q4k_ms / q4kf_ms);
        println!();
        println!("  Ollama reference:   ~10ms  (~100 tok/s)");
        println!("  Q4_KF gap:          {:.1}x", q4kf_ms / 10.0);
        println!(
            "  Q4_KF data/layer:   {:.1}MB (vs Q4_K {:.1}MB)",
            layers_data[0].wq_q4kf.len() as f64 / 1e6 * 4.0
                + layers_data[0].gate_q4.len() as f64 / 1e6 * 3.0,
            layers_data[0].wq_q4k.len() as f64 / 1e6 * 4.0
                + layers_data[0].gate_q4.len() as f64 / 1e6 * 3.0
        );

        println!("\n=== Done ===");
    }
}
