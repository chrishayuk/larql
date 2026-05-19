//! Architecture Overview — guided tour of larql-compute's major design decisions.
//!
//! Runs a condensed version of each key capability:
//!   1. Backend auto-detection (CPU vs Metal)
//!   2. f32 BLAS matmul with GPU/CPU routing
//!   3. Q4_0 quantization + kernel variants
//!   4. Q4_K/Q6_K Ollama-compatible quantization
//!   5. Fused QKV projection (single dispatch)
//!   6. Safe buffer reads (read_buffer_f32)
//!   7. Full pipeline (21 layers, one command buffer)
//!   8. KV-cached decode (dual Q4_K/Q8 path)
//!
//! Usage:
//!   cargo run --release -p larql-compute --example demo_architecture
//!   cargo run --release --features metal -p larql-compute --example demo_architecture

extern crate blas_src;

fn main() {
    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k, quantize_to_q8};
    use larql_compute::{cpu_backend, default_backend};
    use ndarray::Array2;
    use std::time::Instant;

    println!("╔══════════════════════════════════════════════╗");
    println!("║   larql-compute Architecture Overview        ║");
    println!("╚══════════════════════════════════════════════╝\n");

    // ── 1. Backend Detection ──
    println!("1. Backend Detection");
    let backend = default_backend();
    let cpu = cpu_backend();
    println!("   Default: {} ({})", backend.name(), backend.device_info());
    println!("   CPU:     {}", cpu.name());
    println!(
        "   Q4 support: {}, KV cache: {}\n",
        backend.supports_quant(::larql_compute::QuantFormat::Q4_K),
        backend.has_kv_cache()
    );

    // ── 2. f32 Matmul with Auto-Routing ──
    println!("2. f32 Matmul (BLAS → auto GPU/CPU routing)");
    let a = Array2::from_shape_fn((6, 2560), |_| 0.01f32);
    let b = Array2::from_shape_fn((2560, 2560), |_| 0.01f32);
    let t = Instant::now();
    let _c = backend.matmul_transb(a.view(), b.view());
    println!(
        "   [6, 2560] @ [2560, 2560]^T → {:.2}ms\n",
        t.elapsed().as_secs_f64() * 1000.0
    );

    // ── 3. Q4_0 Quantization ──
    println!("3. Q4_0 Quantization (production FFN kernel)");
    let matrix: Vec<f32> = (0..10240 * 2560)
        .map(|i| (i as f32 * 0.0001).cos())
        .collect();
    let q4 = quantize_q4_0(&matrix);
    let x: Vec<f32> = (0..2560).map(|i| (i as f32 * 0.001).sin()).collect();
    let (q8_x, q8_s) = quantize_to_q8(&x);
    let t = Instant::now();
    let scores = backend.q4_matvec(&q4, &q8_x, &q8_s, 10240, 2560);
    let q4_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!(
        "   Q4_0 [10240, 2560] @ Q8[2560]: {q4_ms:.2}ms  (14.7MB data, {:.0} GB/s)",
        14.7 / q4_ms
    );
    println!(
        "   Output nonzero: {}\n",
        scores.is_some_and(|s| s.iter().any(|v| v.abs() > 0.001))
    );

    // ── 4. Q4_K Ollama-Compatible ──
    println!("4. Q4_K Quantization (Ollama-compatible, 148B per 256 values)");
    let small: Vec<f32> = (0..256 * 2560).map(|i| (i as f32 * 0.001).cos()).collect();
    let q4k = quantize_q4_k(&small);
    let t = Instant::now();
    let q4k_out = backend.q4k_matvec(&q4k, &x, 256, 2560);
    println!(
        "   Q4_K [256, 2560] @ f32[2560]: {:.2}ms",
        t.elapsed().as_secs_f64() * 1000.0
    );
    println!(
        "   Output nonzero: {}\n",
        q4k_out.is_some_and(|s| s.iter().any(|v| v.abs() > 0.001))
    );

    // ── 5. Fused QKV ──
    println!("5. Fused QKV Projection (ADR-003)");
    println!("   Metal decode can fuse Q+K+V projection into one dispatch.");
    println!("   CPU/default builds still expose the same backend trait surface.");
    println!("   Format-specific routing lives behind QuantFormat and QuantWeight.\n");

    // ── 6. Safe Buffer Reads ──
    println!("6. Safe Buffer Reads (ADR-005)");
    println!("   Metal buffer reads go through `read_buffer_f32` helpers.");
    println!("   Callers get owned Vec<f32> data instead of borrowed raw slices.\n");

    // ── 7. Full Pipeline ──
    if backend.supports_quant(::larql_compute::QuantFormat::Q4_K) {
        println!("7. Full Pipeline");
        let norm = vec![1.0f32; 2560];
        let gate = quantize_q4_0(&vec![0.01f32; 10240 * 2560]);
        let up = quantize_q4_0(&vec![0.01f32; 10240 * 2560]);
        let down = quantize_q4_0(&vec![0.01f32; 2560 * 10240]);
        let wq = quantize_q4_k(
            &(0..2560 * 2560)
                .map(|i| (i as f32 * 0.0001).cos())
                .collect::<Vec<_>>(),
        );
        let wk = quantize_q4_k(
            &(0..1280 * 2560)
                .map(|i| (i as f32 * 0.0002).sin())
                .collect::<Vec<_>>(),
        );
        let wv = quantize_q4_k(
            &(0..1280 * 2560)
                .map(|i| (i as f32 * 0.0003).cos())
                .collect::<Vec<_>>(),
        );
        let wo = quantize_q4_k(
            &(0..2560 * 2560)
                .map(|i| (i as f32 * 0.0004).sin())
                .collect::<Vec<_>>(),
        );

        let layer = larql_compute::FullPipelineLayer {
            wq: larql_compute::QuantWeight {
                data: &wq,
                scales: None,
                format: larql_compute::QuantFormat::Q4_K,
            },
            wk: larql_compute::QuantWeight {
                data: &wk,
                scales: None,
                format: larql_compute::QuantFormat::Q4_K,
            },
            wv: larql_compute::QuantWeight {
                data: &wv,
                scales: None,
                format: larql_compute::QuantFormat::Q4_K,
            },
            wo: larql_compute::QuantWeight {
                data: &wo,
                scales: None,
                format: larql_compute::QuantFormat::Q4_K,
            },
            gate: larql_compute::QuantWeight {
                data: &gate,
                scales: None,
                format: larql_compute::QuantFormat::Q4_0,
            },
            up: larql_compute::QuantWeight {
                data: &up,
                scales: None,
                format: larql_compute::QuantFormat::Q4_0,
            },
            down: larql_compute::QuantWeight {
                data: &down,
                scales: None,
                format: larql_compute::QuantFormat::Q4_0,
            },
            input_norm: &norm,
            post_attn_norm: &norm,
            pre_ffn_norm: None,
            post_ffn_norm: None,
            norm_offset: 1.0,
            has_post_norms: false,
            activation: larql_compute::Activation::Silu,
            qk_norm_offset: 0.0,
            eps: larql_compute::RMSNORM_EPSILON_DEFAULT,
            norm_type: larql_compute::NormType::RmsNorm,
            ffn_type: larql_compute::FfnType::Gated,
            attn_scale: 1.0 / (320.0f32).sqrt(),
            head_dim: 320,
            num_q_heads: 8,
            num_kv_heads: 4,
            rope_base: larql_compute::ROPE_BASE_DEFAULT,
            ..larql_compute::FullPipelineLayer::default()
        };
        let layers = vec![layer];

        let t = Instant::now();
        let result = backend.full_pipeline_q4(
            &layers, &x, 2560, 10240, 1,     // seq_len
            false, // use_qk_norm
            0.0,   // softcap
        );
        println!(
            "   1 layer (attn+FFN): {:.2}ms",
            t.elapsed().as_secs_f64() * 1000.0
        );
        if let Some(result) = result {
            println!(
                "   Output: {} elements, nonzero: {}\n",
                result.len(),
                result.iter().any(|v| v.abs() > 1e-6)
            );
        } else {
            println!("   Not available on current backend; Metal implements this path.\n");
        }
    }

    // ── 8. Architecture Summary ──
    println!("8. Architecture Summary");
    println!("   CPU is always available through BLAS-backed kernels.");
    println!("   Metal is optional and macOS-gated behind --features metal.");
    println!("   Quant formats share the backend trait contract across CPU and Metal.");
    println!("   Full decode/prefill paths are tested by crate integration tests.");

    println!("\n   See: crates/larql-compute/PERFORMANCE.md");
    println!("        crates/larql-compute/ROADMAP.md");
    println!("        crates/larql-compute/README.md");
}
