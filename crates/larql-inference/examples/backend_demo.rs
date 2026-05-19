//! Backend demo — shows auto-calibrated hybrid CPU/Metal dispatch.
//!
//! Demonstrates:
//! - Auto-calibration: measures CPU vs Metal on this hardware, finds crossover
//! - FLOP-based routing: small ops → CPU (AMX), large ops → Metal (GPU)
//! - Buffer cache: weight matrices loaded to GPU once, reused across calls
//! - Cold vs warm performance with cached buffers
//!
//! Usage:
//!   cargo run --release -p larql-inference --example backend_demo
//!   cargo run --release -p larql-inference --example backend_demo --features metal

use ndarray::Array2;
use std::time::Instant;

use larql_compute::CpuBackend;
use larql_compute::{default_backend, ComputeBackend, MatMul, MatMulOp};

/// Deterministic f32 matrix.
fn synth_matrix(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
    let mut state = seed;
    let data: Vec<f32> = (0..rows * cols)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        })
        .collect();
    Array2::from_shape_vec((rows, cols), data).unwrap()
}

fn max_diff(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn main() {
    println!("=== MatMul Backend Demo ===\n");

    // ── 1. Backend detection + calibration ──
    let cpu = CpuBackend;
    println!("CPU backend: {}", cpu.name());

    let t0 = Instant::now();
    let default = default_backend();
    let init_ms = t0.elapsed().as_millis();
    println!("Default backend: {}  (init: {init_ms}ms)", default.name());

    // Show calibrated threshold if Metal
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        // The default_backend() already calibrated. Show the result.
        // We need to access the threshold — let's create a second backend to inspect.
        if let Some(metal) = larql_compute_metal::MetalBackend::new() {
            metal.calibrate();
            let threshold = metal.flop_threshold();
            println!(
                "Calibrated FLOP threshold: {} ({:.1}M FLOPs)",
                threshold,
                threshold as f64 / 1e6
            );
        }
    }
    println!();

    // ── 2. Dispatch routing table ──
    let seq = 6;
    let hidden = 2560;
    let head_dim = 256;
    let num_heads = 10;
    let intermediate = 10240;
    let vocab = 262144;

    println!("--- Dispatch routing (auto-calibrated) ---");
    println!();
    let ops_table: Vec<(&str, usize, usize, usize)> = vec![
        ("QK^T (per head)", seq, seq, head_dim),
        ("scores*V (per head)", seq, head_dim, seq),
        ("Q/K/V projection", seq, num_heads * head_dim, hidden),
        ("O projection", seq, hidden, num_heads * head_dim),
        ("FFN gate", seq, intermediate, hidden),
        ("FFN down", seq, hidden, intermediate),
        ("Logits", 1, vocab, hidden),
    ];
    for (name, m, n, k) in &ops_table {
        let f = 2 * m * n * k;
        // We can't directly check the threshold from the trait, but we can
        // measure whether it routes to GPU by timing
        println!("  {name:<25} {f:>12} FLOPs  ({:.1}M)", f as f64 / 1e6);
    }
    println!();

    // ── 3. Buffer cache: cold vs warm ──
    println!("--- Buffer cache: weight matrix reuse ---");
    let w_q = synth_matrix(num_heads * head_dim, hidden, 43);

    // First call: cold cache
    let h = synth_matrix(seq, hidden, 42);
    let t0 = Instant::now();
    let _ = default.matmul_transb(h.view(), w_q.view());
    let cold_us = t0.elapsed().as_micros();

    // Second call: warm cache (same weight matrix pointer)
    let h2 = synth_matrix(seq, hidden, 44);
    let t0 = Instant::now();
    let _ = default.matmul_transb(h2.view(), w_q.view());
    let warm_us = t0.elapsed().as_micros();

    // Third call: still cached
    let h3 = synth_matrix(seq, hidden, 45);
    let t0 = Instant::now();
    let _ = default.matmul_transb(h3.view(), w_q.view());
    let hot_us = t0.elapsed().as_micros();

    // CPU baseline
    let t0 = Instant::now();
    let _ = cpu.matmul_transb(h.view(), w_q.view());
    let cpu_us = t0.elapsed().as_micros();

    println!(
        "  Q proj [{seq},{hidden}] x [{},{hidden}]^T  ({}M FLOPs)",
        num_heads * head_dim,
        2 * seq * (num_heads * head_dim) * hidden / 1_000_000
    );
    println!("  CPU:               {cpu_us:>8} us");
    println!("  Default cold:      {cold_us:>8} us  (buffer created)");
    println!("  Default warm:      {warm_us:>8} us  (cache hit)");
    println!("  Default hot:       {hot_us:>8} us  (cache hit)");
    println!();

    // ── 4. Small ops stay on CPU ──
    println!("--- Small ops: CPU fast path ---");
    let q = synth_matrix(seq, head_dim, 100);
    let k = synth_matrix(seq, head_dim, 101);

    let t0 = Instant::now();
    let s_cpu = cpu.matmul_transb(q.view(), k.view());
    let cpu_us = t0.elapsed().as_micros();

    let t0 = Instant::now();
    let s_default = default.matmul_transb(q.view(), k.view());
    let def_us = t0.elapsed().as_micros();

    let diff = max_diff(&s_cpu, &s_default);
    println!("  QK^T [{seq},{head_dim}] x [{seq},{head_dim}]^T  (18K FLOPs)");
    println!("  CPU:     {cpu_us:>8} us");
    println!("  Default: {def_us:>8} us  (routes to CPU)");
    println!("  Max diff: {diff:.2e}");
    println!();

    // ── 5. FFN gate projection ──
    println!("--- FFN gate projection ---");
    let x = synth_matrix(seq, hidden, 200);
    let w_gate = synth_matrix(intermediate, hidden, 201);

    let t0 = Instant::now();
    let g_cpu = cpu.matmul_transb(x.view(), w_gate.view());
    let cpu_us = t0.elapsed().as_micros();

    // Cold
    let t0 = Instant::now();
    let g_default = default.matmul_transb(x.view(), w_gate.view());
    let def_cold_us = t0.elapsed().as_micros();

    // Warm
    let x2 = synth_matrix(seq, hidden, 202);
    let t0 = Instant::now();
    let _ = default.matmul_transb(x2.view(), w_gate.view());
    let def_warm_us = t0.elapsed().as_micros();

    let diff = max_diff(&g_cpu, &g_default);
    println!("  [{seq},{hidden}] x [{intermediate},{hidden}]^T  (315M FLOPs)");
    println!("  CPU:          {cpu_us:>8} us");
    println!("  Default cold: {def_cold_us:>8} us");
    println!("  Default warm: {def_warm_us:>8} us");
    println!("  Max diff: {diff:.2e}");
    println!();

    // ── 6. Simulated full attention layer ──
    println!("--- Full attention layer (4 projections, cached weights) ---");
    let w_k = synth_matrix(num_heads * head_dim, hidden, 50);
    let w_v = synth_matrix(num_heads * head_dim, hidden, 51);
    let w_o = synth_matrix(hidden, num_heads * head_dim, 52);

    // Warm all weight buffers
    let h_warm = synth_matrix(seq, hidden, 60);
    let _ = default.matmul_transb(h_warm.view(), w_q.view());
    let _ = default.matmul_transb(h_warm.view(), w_k.view());
    let _ = default.matmul_transb(h_warm.view(), w_v.view());
    let attn_warm = synth_matrix(seq, num_heads * head_dim, 61);
    let _ = default.matmul_transb(attn_warm.view(), w_o.view());

    // Benchmark hot path
    let h_input = synth_matrix(seq, hidden, 70);
    let attn_out = synth_matrix(seq, num_heads * head_dim, 71);

    let t0 = Instant::now();
    let _ = cpu.matmul_transb(h_input.view(), w_q.view());
    let _ = cpu.matmul_transb(h_input.view(), w_k.view());
    let _ = cpu.matmul_transb(h_input.view(), w_v.view());
    let _ = cpu.matmul_transb(attn_out.view(), w_o.view());
    let cpu_layer_us = t0.elapsed().as_micros();

    let t0 = Instant::now();
    let _ = default.matmul_transb(h_input.view(), w_q.view());
    let _ = default.matmul_transb(h_input.view(), w_k.view());
    let _ = default.matmul_transb(h_input.view(), w_v.view());
    let _ = default.matmul_transb(attn_out.view(), w_o.view());
    let def_layer_us = t0.elapsed().as_micros();

    println!("  Q + K + V + O projections (all cached)");
    println!("  CPU:     {cpu_layer_us:>8} us");
    println!("  Default: {def_layer_us:>8} us");
    if cpu_layer_us > 0 && def_layer_us > 0 {
        let ratio = cpu_layer_us as f64 / def_layer_us as f64;
        if ratio > 1.0 {
            println!("  -> Default {ratio:.1}x faster");
        } else {
            println!("  -> CPU {:.1}x faster", 1.0 / ratio);
        }
    }
    println!();

    // ── 7. Batched Q/K/V/O in one dispatch ──
    println!("--- Batched attention projections (1 dispatch) ---");
    let ops = vec![
        MatMulOp {
            a: h_input.clone(),
            b: w_q.clone(),
            transpose_b: true,
        },
        MatMulOp {
            a: h_input.clone(),
            b: w_k.clone(),
            transpose_b: true,
        },
        MatMulOp {
            a: h_input.clone(),
            b: w_v.clone(),
            transpose_b: true,
        },
        MatMulOp {
            a: attn_out.clone(),
            b: w_o.clone(),
            transpose_b: true,
        },
    ];

    let t0 = Instant::now();
    let _batched = default.matmul_batch(&ops);
    let batch_us = t0.elapsed().as_micros();

    println!("  4 projections in 1 batch:  {batch_us:>8} us");
    println!("  4 projections serial:      {def_layer_us:>8} us");
    println!("  CPU serial:                {cpu_layer_us:>8} us");
    println!();

    println!("=== Done ===");
}
