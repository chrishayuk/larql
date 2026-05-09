//! Microbench for `cuda::q4k_batched::matvec_batched` at the
//! Gemma 3 4B Q4_K_M shapes that drive decode.
//!
//! Compares `M=1` invoked N times vs `M=N` invoked once at the
//! main projection shapes. Phase 4's Tensor Core re-arm hinges on
//! M≥4 amortising weight bandwidth — this bench answers whether
//! the batched kernel itself wins (regardless of TC dispatch).
//!
//! Run with `--ignored` to opt in:
//!   LARQL_CUDA_AVAILABLE=1 cargo test --release \
//!     -p larql-compute --features cuda \
//!     --test bench_cuda_q4k_batched -- --ignored --nocapture

#![cfg(any(feature = "cuda", feature = "cuda-oxide"))]

use std::time::Instant;

use larql_compute::cuda::q4k_batched::{matvec_batched, M_TILE_MAX};
use larql_compute::cuda::CudaBackend;

const Q4K_BLOCK_BYTES: usize = 144;
const Q4K_BLOCK_ELEMS: usize = 256;

fn gpu_or_skip() -> Option<CudaBackend> {
    if std::env::var("LARQL_CUDA_AVAILABLE").ok().as_deref() != Some("1") {
        return None;
    }
    CudaBackend::new().ok()
}

fn synth_bytes(rows: usize, hidden: usize, seed: u64) -> Vec<u8> {
    let blocks_per_row = hidden / Q4K_BLOCK_ELEMS;
    let n_bytes = rows * blocks_per_row * Q4K_BLOCK_BYTES;
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (0..n_bytes)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) & 0xff) as u8
        })
        .collect()
}

fn synth_x(m: usize, hidden: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0xCBF2_9CE4_8422_2325);
    (0..m * hidden)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s & 0xFFFF) as f32 / 65536.0) - 0.5
        })
        .collect()
}

fn time_one_call(
    backend: &CudaBackend,
    bytes: &[u8],
    x: &[f32],
    rows: usize,
    hidden: usize,
    m: usize,
    iters: usize,
) -> f64 {
    // Warmup
    for _ in 0..3 {
        let _ = matvec_batched(backend, bytes, x, rows, hidden, m).unwrap();
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = matvec_batched(backend, bytes, x, rows, hidden, m).unwrap();
    }
    let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
    elapsed / iters as f64
}

fn bench_shape(backend: &CudaBackend, name: &str, rows: usize, hidden: usize) {
    println!("\n=== {name}  rows={rows}  hidden={hidden} ===");
    let bytes = synth_bytes(rows, hidden, 0xABCD);

    let iters = 200;

    // Baseline: M=1 invoked N times for N in {1, 2, 4, 8}.
    let x1 = synth_x(1, hidden, 1);
    let m1_ms = time_one_call(backend, &bytes, &x1, rows, hidden, 1, iters);
    println!(
        "  M=1×1 single-call  : {m1_ms:.4} ms/call ({:.2} tok/s if 1 tok = 1 call)",
        1000.0 / m1_ms
    );

    // For each M > 1, time both:
    //  - M independent calls with M=1 (the "currently shipped" path)
    //  - 1 call with M_TILE=M (the new batched path)
    for m in &[2usize, 4, 8] {
        if *m > M_TILE_MAX {
            continue;
        }
        let m = *m;
        let x_one = synth_x(1, hidden, 7);
        let x_batched = synth_x(m, hidden, 7);

        // Sequential: M independent M=1 calls.
        for _ in 0..3 {
            for _ in 0..m {
                let _ = matvec_batched(backend, &bytes, &x_one, rows, hidden, 1).unwrap();
            }
        }
        let t0 = Instant::now();
        for _ in 0..iters {
            for _ in 0..m {
                let _ = matvec_batched(backend, &bytes, &x_one, rows, hidden, 1).unwrap();
            }
        }
        let seq_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        // Batched: 1 call with M_TILE=m.
        let bat_ms = time_one_call(backend, &bytes, &x_batched, rows, hidden, m, iters);

        let speedup = seq_ms / bat_ms;
        println!(
            "  M={m}: seq={seq_ms:.4} ms  batched={bat_ms:.4} ms  speedup={speedup:.2}× \
             ({})",
            if speedup > 1.10 {
                "WIN"
            } else if speedup < 0.90 {
                "LOSS"
            } else {
                "wash"
            }
        );
    }
}

#[test]
#[ignore]
fn bench_q4k_batched_gemma_shapes() {
    let Some(backend) = gpu_or_skip() else {
        eprintln!("LARQL_CUDA_AVAILABLE != 1; skipping");
        return;
    };

    // Gemma 3 4B Q4_K_M projection shapes (hidden_dim = 2560 in target,
    // FFN intermediate = 10240, vocab head not benched here).
    bench_shape(&backend, "proj_qkv  (n_q + 2*n_kv heads)", 5120, 2560);
    bench_shape(&backend, "proj_wo                          ", 2560, 2560);
    bench_shape(&backend, "proj_gate_up      (FFN)          ", 10240, 2560);
    bench_shape(&backend, "proj_down         (FFN)          ", 2560, 10240);
}
