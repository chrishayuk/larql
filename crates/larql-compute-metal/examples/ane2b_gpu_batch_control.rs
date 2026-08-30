//! ANE-2B's missing GPU control — is the ANE's 1.04x at N=8 special?
//!
//! ANE-2B measured N vectors through one weight traversal on the Neural
//! Engine: N=1/2/4/8 costing 1.00 / 1.00 / 1.01 / **1.04x**. That is a
//! striking number, but it only becomes a claim about the ANE once the
//! same axis is measured on Metal. This is that control.
//!
//! Mirrors ANE-2B as closely as the backend allows: same `5120 -> 17408`,
//! same deterministic f16 weight and activation generators as ANE-0b and
//! ANE-1, same N ladder, same reported statistics.
//!
//! **No new kernel is written here.** The instruction was to establish
//! the ordinary Metal answer before optimising a special GEMM path, so
//! both arms use kernels the backend already ships.
//!
//! ## Two arms, because LARQL has no f16 GEMM
//!
//! ```text
//! A  f16 gemv, repeated N times   what LARQL ACTUALLY does at f16 today.
//!                                 `f16_gemv` is a gemv: one vector per
//!                                 call, so N vectors is N traversals.
//!
//! B  f32 sgemm_transb, M = N      an existing tiled GEMM. Answers the
//!                                 HARDWARE question — can the GPU amortise
//!                                 a weight traversal across vectors — at
//!                                 the cost of a dtype mismatch (f32, so
//!                                 2x the bytes). Ratio only; absolute
//!                                 latency is NOT comparable to the ANE.
//! ```
//!
//! ## Arm B's result is PRE-REGISTERED, not discovered
//!
//! `sgemm_transb` tiles 32x32. With `M <= 8` every row fits in a single
//! 32-row tile, so `ceil(M/32) == 1` for all N in this ladder: identical
//! threadgroup count, identical K-loop, identical B traffic. **Arm B must
//! therefore come out flat at ~1.00x for structural reasons, and that is
//! a statement about the tile width, not a measured property of the GPU's
//! batching economics.** It is printed before the numbers so it cannot be
//! retro-fitted afterwards.
//!
//! ## The silent-fallback hazard this harness had to defuse
//!
//! `matmul_transb` falls back to CPU BLAS below `flop_threshold`, whose
//! default is 500 MFLOP. At N=1 the work is 178 MFLOP and at N=2 it is
//! 356 MFLOP — both **under** the default — while N=4 and N=8 are over
//! it. Left alone, the sweep would have run its first two points on the
//! CPU and its last two on the GPU and reported the device switch as a
//! batching curve. The threshold is lowered to the floor and verified
//! below.

use larql_compute::backend::matmul::MatMul;
use larql_models::quant::half::{f16_to_f32, f32_to_f16};
use ndarray::Array2;
use std::time::Instant;

const HIDDEN_SIZE: usize = 5120;
const INTERMEDIATE_SIZE: usize = 17408;
const BATCHES: &[usize] = &[1, 2, 4, 8];

const WARMUP_ITERS: usize = 32;
const MEASURED_ITERS: usize = 256;

const BYTES_PER_F16: usize = 2;

/// `sgemm_transb`'s tile, mirrored from the shader for the prediction.
const SGEMM_TILE: usize = 32;

/// ANE-2B's banked ratios, for the side-by-side.
const ANE_RATIOS: &[f64] = &[1.00, 1.00, 1.01, 1.04];

fn weights_f16(n: usize, k: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(n * k * BYTES_PER_F16);
    for i in 0..n * k {
        let v = ((i % 977) as f32 / 977.0) - 0.5;
        bytes.extend_from_slice(&f32_to_f16(v).to_le_bytes());
    }
    bytes
}

fn activation(k: usize) -> Vec<f32> {
    (0..k).map(|i| ((i % 13) as f32) * 0.01 - 0.06).collect()
}

/// The same values as the f16 weights, widened — so arm B computes the
/// same projection, just in a wider dtype.
fn widen(w_f16: &[u8], n: usize, k: usize) -> Array2<f32> {
    let mut v = Vec::with_capacity(n * k);
    for i in 0..n * k {
        let bits = u16::from_le_bytes([w_f16[i * 2], w_f16[i * 2 + 1]]);
        v.push(f16_to_f32(bits));
    }
    Array2::from_shape_vec((n, k), v).expect("shape")
}

struct Stats {
    min: f64,
    p50: f64,
    p90: f64,
    stdev: f64,
}

fn stats(samples: &[f64]) -> Stats {
    let mut s = samples.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
    let n = s.len();
    let mean = s.iter().sum::<f64>() / n as f64;
    let var = s.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
    Stats {
        min: s[0],
        p50: s[n / 2],
        p90: s[(n * 9) / 10],
        stdev: var.sqrt(),
    }
}

fn main() {
    let Some(gpu) = larql_compute_metal::MetalBackend::new() else {
        eprintln!("no Metal device");
        std::process::exit(2);
    };

    let (n, k) = (INTERMEDIATE_SIZE, HIDDEN_SIZE);
    let w16 = weights_f16(n, k);
    let x = activation(k);

    println!("ANE-2B GPU control — {k} -> {n}, warmup {WARMUP_ITERS}, measured {MEASURED_ITERS}");
    println!(
        "PRE-REGISTERED: arm B (sgemm_transb, {SGEMM_TILE}x{SGEMM_TILE} tile) must be flat at"
    );
    println!("~1.00x for all N <= {SGEMM_TILE}, since ceil(M/{SGEMM_TILE}) == 1 throughout.");
    println!("That is the tile width, not the GPU's batching economics.\n");

    // --- defuse the silent CPU fallback -----------------------------
    let before = gpu.flop_threshold();
    gpu.set_flop_threshold(0); // clamps to MIN_FLOP_FLOOR
    let after = gpu.flop_threshold();
    let smallest_flops = 2 * BATCHES[0] * n * k;
    println!("flop_threshold {before} -> {after}; smallest arm-B work {smallest_flops} FLOP");
    if smallest_flops < after {
        eprintln!(
            "refusing: N=1 would fall back to CPU BLAS and the ratio would be a device switch"
        );
        std::process::exit(1);
    }
    println!();

    // --- arm A: what LARQL does at f16 today ------------------------
    println!("arm A — f16 gemv repeated N times (LARQL's actual f16 path)");
    println!(
        "{:>4}{:>10}{:>10}{:>10}{:>12}{:>12}{:>10}",
        "N", "min ms", "p50 ms", "sd ms", "T(N)/T(1)", "per-vector", "ANE ref"
    );
    let mut base_a = 0.0;
    for (i, &b) in BATCHES.iter().enumerate() {
        for _ in 0..WARMUP_ITERS {
            for _ in 0..b {
                gpu.f16_gemv_force(&w16, &x, n, k).expect("gemv");
            }
        }
        let mut samples = Vec::with_capacity(MEASURED_ITERS);
        for _ in 0..MEASURED_ITERS {
            let t = Instant::now();
            for _ in 0..b {
                gpu.f16_gemv_force(&w16, &x, n, k).expect("gemv");
            }
            samples.push(t.elapsed().as_secs_f64() * 1e3);
        }
        let s = stats(&samples);
        if i == 0 {
            base_a = s.min;
        }
        let ratio = s.min / base_a;
        println!(
            "{:>4}{:>10.3}{:>10.3}{:>10.3}{:>12.2}{:>12.2}{:>10.2}",
            b,
            s.min,
            s.p50,
            s.stdev,
            ratio,
            ratio / b as f64,
            ANE_RATIOS[i]
        );
        let _ = s.p90;
    }

    // --- arm B: an existing GEMM ------------------------------------
    println!("\narm B — f32 sgemm_transb, M = N (existing kernel, f32 so 2x bytes; RATIO only)");
    println!(
        "{:>4}{:>10}{:>10}{:>10}{:>12}{:>12}{:>10}",
        "N", "min ms", "p50 ms", "sd ms", "T(N)/T(1)", "per-vector", "ANE ref"
    );
    let w32 = widen(&w16, n, k);
    let mut base_b = 0.0;
    for (i, &b) in BATCHES.iter().enumerate() {
        let a = Array2::from_shape_fn((b, k), |(_, j)| x[j]);
        for _ in 0..WARMUP_ITERS {
            let _ = gpu.matmul_transb(a.view(), w32.view());
        }
        let mut samples = Vec::with_capacity(MEASURED_ITERS);
        for _ in 0..MEASURED_ITERS {
            let t = Instant::now();
            let _ = gpu.matmul_transb(a.view(), w32.view());
            samples.push(t.elapsed().as_secs_f64() * 1e3);
        }
        let s = stats(&samples);
        if i == 0 {
            base_b = s.min;
        }
        let ratio = s.min / base_b;
        println!(
            "{:>4}{:>10.3}{:>10.3}{:>10.3}{:>12.2}{:>12.2}{:>10.2}",
            b,
            s.min,
            s.p50,
            s.stdev,
            ratio,
            ratio / b as f64,
            ANE_RATIOS[i]
        );
    }

    println!("\nArm A is the dtype-faithful comparison and the one LARQL lives with today.");
    println!("Arm B answers whether the GPU CAN amortise, using a kernel that already exists.");
}
