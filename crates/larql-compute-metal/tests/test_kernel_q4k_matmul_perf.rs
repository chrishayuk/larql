//! Sanity check that `q4k_matmul` is actually faster than stacked
//! `q4k_matvec` calls on the production prefill shape — that is what
//! makes the kernel worth its complexity. Not a rigorous benchmark
//! (criterion lives in `benches/`); just a wall-clock spot check
//! gated on `LARQL_PERF_SPOT_CHECK=1` so it doesn't slow down `cargo
//! test`.

#![cfg(target_os = "macos")]

extern crate blas_src;

use larql_compute::cpu::ops::q4_common::quantize_q4_k;
use larql_compute::prelude::*;
use larql_compute_metal::MetalBackend;
use std::time::Instant;

fn synth(len: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..len)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

#[test]
fn q4k_matmul_faster_than_stacked_matvec_on_prefill_shape() {
    if std::env::var("LARQL_PERF_SPOT_CHECK").is_err() {
        // Default-skipped: timing is sensitive to system load and
        // not worth the 5-10 s it adds to `cargo test`. Set the env
        // var to opt in.
        return;
    }
    let metal = match MetalBackend::new() {
        Some(m) => m,
        None => return,
    };

    // Gemma 3 4B O projection per layer: N=hidden=2560, K=q_dim=8192.
    // 18-token prompt = realistic prefill seq_len.
    let num_rows = 2560usize;
    let hidden = 8192usize;
    let seq_len = 18usize;

    let weights = synth(num_rows * hidden, 1001);
    let x = synth(seq_len * hidden, 1002);
    let q4k = quantize_q4_k(&weights);

    // Warmup: pin pipeline, prime caches.
    for _ in 0..3 {
        let _ = metal.q4k_matmul(&q4k, &x, num_rows, hidden, seq_len);
    }

    // Time stacked matvec (the current per-position prefill approach).
    let t0 = Instant::now();
    let iters = 5;
    for _ in 0..iters {
        for m in 0..seq_len {
            let row = &x[m * hidden..(m + 1) * hidden];
            let _ = metal.q4k_matvec(&q4k, row, num_rows, hidden);
        }
    }
    let stacked_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    // Time matmul.
    let t1 = Instant::now();
    for _ in 0..iters {
        let _ = metal.q4k_matmul(&q4k, &x, num_rows, hidden, seq_len);
    }
    let matmul_ms = t1.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    let speedup = stacked_ms / matmul_ms;
    eprintln!("q4k_matmul perf vs stacked matvec (N={num_rows}, K={hidden}, M={seq_len}):");
    eprintln!("  stacked matvec: {stacked_ms:.2} ms / call");
    eprintln!("  q4k_matmul:     {matmul_ms:.2} ms / call");
    eprintln!("  speedup:        {speedup:.2}×");

    // The amortisation of dequant across COLS_PER_TG=4 positions
    // should give >= ~1.5× even with imperfect ALU utilisation.
    // Below 1.0× would mean the kernel is actively slower — that's
    // a regression worth surfacing.
    assert!(
        speedup >= 1.0,
        "q4k_matmul ({matmul_ms:.2} ms) slower than stacked matvec ({stacked_ms:.2} ms) — {speedup:.2}×"
    );
}
