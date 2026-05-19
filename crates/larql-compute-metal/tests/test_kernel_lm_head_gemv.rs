#![cfg(target_os = "macos")]

//! Kernel-level bisect for the CPU/Metal LM-head divergence surfaced
//! by `test_logits_goldens` on tied-embedding models (Gemma 3 4B,
//! Gemma 4 31B).
//!
//! ## What we're testing
//!
//! The LM head goes through `index.lm_head_knn_backend` which has
//! three paths:
//!   1. `backend.q4_matvec` — Q4_0 weights × Q8 quantized query.
//!      Used when `lm_head_q4.bin` exists *or* `lm_head_q4_synth`
//!      was built from f16 embeddings (tied-embed Gemma path).
//!   2. `backend.f16_gemv` — f16 weights × f32 query (some vindexes).
//!   3. `backend.f32_gemv` / BLAS — f32 fallback.
//!
//! End-to-end goldens show CPU and Metal disagree on Gemma's top-5
//! next token, but agree on Llama 2 and Mistral. Per-stage parity
//! tests pass at `cos=1.0` through `down_out`, so the divergence is
//! in the LM-head step. Llama 2 / Mistral go through path 3 (f32
//! BLAS, kernel-equivalent on both backends — see
//! `f32_gemv_matches_ndarray_dot` and the vocab-scale test below);
//! Gemma's tied-embedding path goes through path 1 (Q4_0 + Q8),
//! which is where the divergence has to live.
//!
//! This file pins both paths at vocab scale:
//!
//! - `f32_gemv_cpu_vs_metal_at_vocab_scale` — confirms suspect (3)
//!   is **clean**: the f32 fallback agrees on top-5 + top-1 logit
//!   between CPU and Metal at K=262144 × hidden=2560.
//! - `q4_matvec_cpu_vs_metal_at_vocab_scale` — pins suspect (1):
//!   same Q4_0 weights + Q8 query on both backends. **Currently
//!   fails (2026-04-25)** — Metal `q4_matvec_v4` computes only ~2
//!   rows per TG out of the intended 8 (= 25 % of rows; the rest
//!   stay at 0.0). Confirmed across N from 8 000 to 262 144 by
//!   `q4_matvec_cutoff_sweep` — the ratio is constant. Pipeline's
//!   `maxTotalThreadsPerThreadgroup` is 1024, so the requested 256
//!   threads-per-TG should fit; the silent reduction to 2 simdgroups
//!   firing per TG is **the** root cause of the open Gemma 3/4
//!   CPU/Metal LM-head divergence in `test_logits_goldens`.
//!
//! Both allocate ~2.68 GB f32 + ~1.3 GB Q4_0; gated to keep casual
//! `cargo test` runs cheap.
//!
//! ```bash
//! LARQL_RUN_LM_HEAD_BISECT=1 \
//!   cargo test --release --features metal -p larql-compute \
//!     --test test_kernel_lm_head_gemv -- --nocapture
//! ```

extern crate blas_src;

#[path = "common/mod.rs"]
mod common;
use common::get_metal;

use larql_compute::prelude::*;
use larql_compute::CpuBackend;
use ndarray::Array2;

fn run_enabled() -> bool {
    matches!(
        std::env::var("LARQL_RUN_LM_HEAD_BISECT").ok().as_deref(),
        Some("1") | Some("true")
    )
}

/// Synthesise a deterministic `[n, k]` matrix and a `[k]` query.
/// Values are scaled to land in the magnitude range f32_gemv sees in
/// production (LM-head logits typically run from ~10⁰ to 10³ depending
/// on the model and how tightly normalised its last hidden is).
fn synth_inputs(n: usize, k: usize) -> (Array2<f32>, Vec<f32>) {
    // Compact deterministic generator — no rand crate dependency.
    let mut w = Vec::with_capacity(n * k);
    for i in 0..n * k {
        let f = i as f32;
        w.push(((f * 0.0001).sin() + 0.3 * (f * 0.00037).cos()) * 0.05);
    }
    let w = Array2::from_shape_vec((n, k), w).unwrap();
    let x: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.013).sin() * 0.5).collect();
    (w, x)
}

fn top5(scores: &[f32]) -> [(u32, f32); 5] {
    let mut indexed: Vec<(u32, f32)> = scores
        .iter()
        .copied()
        .enumerate()
        .map(|(i, s)| (i as u32, s))
        .collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    std::array::from_fn(|i| indexed[i])
}

#[test]
fn f32_gemv_cpu_vs_metal_at_vocab_scale() {
    if !run_enabled() {
        eprintln!(
            "skip: LARQL_RUN_LM_HEAD_BISECT=1 not set. \
             This test allocates a ~2.68 GB f32 matrix; gated to keep \
             casual `cargo test` runs cheap."
        );
        return;
    }

    let metal = get_metal();
    metal.set_flop_threshold(1); // force GPU dispatch even for non-tiny

    // Gemma 3 4B tied-embedding LM head shape.
    let n = 262_144usize; // vocab
    let k = 2_560usize; // hidden
    eprintln!(
        "Synthesising W [{n}, {k}] = {:.2} GB and x [{k}]…",
        (n * k * 4) as f64 / 1e9
    );
    let (w, x) = synth_inputs(n, k);

    // CPU has no `f32_gemv` specialisation (returns `None`); production
    // `lm_head_topk` falls back to `matmul_transb` for the CPU path.
    // Mirror that fallback here so we're benching the *exact* code
    // each backend uses in production.
    let cpu_scores: Vec<f32> = match CpuBackend.f32_gemv(w.view(), &x) {
        Some(s) => s,
        None => {
            let q_row = ndarray::Array2::from_shape_vec((1, k), x.clone()).unwrap();
            CpuBackend
                .matmul_transb(q_row.view(), w.view())
                .row(0)
                .to_vec()
        }
    };
    let metal_scores = metal
        .f32_gemv(w.view(), &x)
        .expect("Metal f32_gemv should dispatch above threshold");

    let cpu_top5 = top5(&cpu_scores);
    let metal_top5 = top5(&metal_scores);

    eprintln!("CPU   top-5: {:?}", cpu_top5);
    eprintln!("Metal top-5: {:?}", metal_top5);

    let cpu_top1 = cpu_top5[0];
    let metal_top1 = metal_top5[0];

    // Within-CPU vs within-Metal accumulation order can swap rank
    // within the top-5 by ULP noise — but the **set** must match,
    // and the top-1 logit value should match within 1e-3 absolute on
    // a 0.05-scale matrix. (Total dot-product range here is bounded
    // by Σ |w| * |x| ≈ 0.05 * 0.5 * 2560 ≈ 64.)
    let mut cpu_set: Vec<u32> = cpu_top5.iter().map(|t| t.0).collect();
    let mut metal_set: Vec<u32> = metal_top5.iter().map(|t| t.0).collect();
    cpu_set.sort_unstable();
    metal_set.sort_unstable();
    assert_eq!(
        cpu_set, metal_set,
        "f32_gemv top-5 sets diverge at vocab-scale K=262144 × hidden=2560 \
         (CPU vs Metal). This is the suspect for the open Gemma 3/4 \
         CPU/Metal LM-head divergence in `test_logits_goldens`. \
         If this fails, the Metal `f32_gemv` shader is the cause; if it \
         passes, the divergence is upstream (last-hidden-state differs)."
    );

    let logit_diff = (cpu_top1.1 - metal_top1.1).abs();
    let max_abs = cpu_scores
        .iter()
        .map(|v| v.abs())
        .fold(0.0f32, f32::max)
        .max(1e-6);
    let rel = logit_diff / max_abs;
    assert!(
        rel < 1e-3,
        "top-1 logit diverges: cpu={:.6} metal={:.6} (rel={:.3e})",
        cpu_top1.1,
        metal_top1.1,
        rel,
    );

    eprintln!(
        "✓ f32_gemv vocab-scale CPU vs Metal: top-5 sets match, \
         top-1 logit Δ={:.3e} (rel {:.2e})",
        logit_diff, rel,
    );
}

/// Probe Metal's `q4_matvec_v4` pipeline state for its actual
/// `maxTotalThreadsPerThreadgroup` limit, and assert the dispatch
/// wrapper's requested threads-per-TG fits inside it. If the compiled
/// shader's resource usage ever caps the pipeline below the dispatch
/// request, Metal will silently run fewer threads/TG → fewer
/// simdgroups → fewer rows covered.
///
/// The actual dispatch request lives in `ops::q4_matvec::dispatch`,
/// which (post-fix) imports its constants from the same shader module
/// the pipeline is built from (`q4_matvec_v4`). Pre-fix the wrapper
/// imported from a different shader (`q4_matvec`) and the constants
/// drifted apart silently — that's what we're guarding against.
#[test]
fn q4_matvec_pipeline_max_threads_per_tg() {
    let metal = get_metal();
    // The KernelHandle constructor already runs this check at startup
    // (returns `None` if the pipeline cap is below the requested
    // threads_per_tg). This test mirrors the same assertion at the
    // test surface so a regression in the cap → row-drop chain is
    // visible in a focused per-kernel test, not just at backend init.
    let kernel = &metal.q4.matvec;
    let limit = kernel.state.max_total_threads_per_threadgroup();
    eprintln!(
        "  {} pipeline maxTotalThreadsPerThreadgroup = {limit} \
         (handle requests {})",
        kernel.kernel_name, kernel.threads_per_tg,
    );
    assert!(
        limit >= kernel.threads_per_tg,
        "pipeline cap ({limit}) < KernelHandle threads_per_tg ({}). \
         Metal would silently dispatch only {limit} threads/TG → fewer \
         simdgroups → rows dropped. (rows_per_tg={}). Either lower the \
         handle's threads_per_tg, or simplify the kernel's per-thread \
         register / threadgroup-memory pressure to raise the cap.",
        kernel.threads_per_tg,
        kernel.rows_per_tg,
    );
}

/// Sweep across N to confirm Metal Q4_0 matvec writes every row at
/// every scale we ship. Pre-fix this leaked at constant ratio 25 %
/// (num_rows / 4) because `ops::q4_matvec::dispatch` imported geometry
/// constants from the wrong shader module — `num_tgs = num_rows / 32`
/// while the kernel actually consumed 8 row-addresses per TG.
///
/// Asserts that for every N in the sweep, `count(metal_scores != 0)`
/// equals N (every output row written) and that Metal's top index
/// agrees with CPU's.
#[test]
fn q4_matvec_cutoff_sweep() {
    if !run_enabled() {
        eprintln!("skip: LARQL_RUN_LM_HEAD_BISECT=1 not set");
        return;
    }
    let metal = get_metal();
    metal.set_flop_threshold(1);
    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_to_q8};

    let k = 256usize; // small K so the sweep is fast
    let x: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.013).sin() + 0.5).collect();
    let (q8_x_i8, q8_scales) = quantize_to_q8(&x);

    // Sweep N at and around 8/32-row boundaries: 8000 (1000 TGs of 8),
    // 32K (4000), 65520 (8190), 65536 (8192), 65560 (8195 — first N
    // beyond the pre-fix wrap-around), 70000, 100000, 262144 (vocab).
    for &n in &[
        8000usize, 32000, 65520, 65536, 65560, 65600, 70000, 100000, 200000, 262144,
    ] {
        let w: Vec<f32> = (0..n * k)
            .map(|i| ((i as f32) * 0.0001).sin() + 0.5)
            .collect();
        let q4 = quantize_q4_0(&w);
        let cpu_scores = CpuBackend
            .q4_matvec(&q4, &q8_x_i8, &q8_scales, n, k)
            .unwrap();
        let metal_scores = metal.q4_matvec(&q4, &q8_x_i8, &q8_scales, n, k).unwrap();
        let metal_nonzero = metal_scores.iter().filter(|&&v| v.abs() > 1e-9).count();
        let cpu_nonzero = cpu_scores.iter().filter(|&&v| v.abs() > 1e-9).count();
        let first_zero = metal_scores.iter().position(|&v| v.abs() <= 1e-9);
        eprintln!(
            "  N={n:>6}  TGs(v4)={:>5}  metal_nonzero={metal_nonzero}/{n}  \
             cpu_nonzero={cpu_nonzero}/{n}  first_zero={first_zero:?}",
            n.div_ceil(8),
        );
        assert_eq!(
            cpu_nonzero, n,
            "test invariant: synth inputs are non-zero so CPU output \
             should be all non-zero (got {cpu_nonzero}/{n} at N={n})"
        );
        assert_eq!(
            metal_nonzero,
            n,
            "Metal q4_matvec dropped {} rows at N={n} (first zero at {first_zero:?}). \
             Pre-fix ratio: ~num_rows/4 covered. Post-fix expectation: every row written.",
            n - metal_nonzero,
        );
    }
}

/// Regression for the 75 %-row drop bug fixed 2026-04-25.
///
/// `ops::q4_matvec::dispatch` previously imported geometry constants
/// from `shaders::q4_matvec` (ROWS_PER_TG=32, THREADS_PER_TG=1024) but
/// the pipeline ran the `q4_matvec_v4` kernel — whose row-mapping is
/// hardcoded as `tg_id * 8 + sg_id`. Mismatch → only `num_rows / 4`
/// rows were ever written; the rest stayed at zero (the buffer's
/// initial value).
///
/// This test runs at small N (1024 rows × 256 hidden, < 200 KB Q4) and
/// asserts every output row is non-zero. With the pre-fix bug 75 % of
/// rows would zero-out; post-fix every row is written. Un-gated so
/// it runs in casual `cargo test --features metal` and CI.
#[test]
fn q4_matvec_metal_writes_every_row_small_n() {
    let metal = get_metal();
    metal.set_flop_threshold(1);
    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_to_q8};

    let n = 1024usize;
    let k = 256usize;
    // Bias non-zero so every dot product is non-zero by construction.
    let w: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.001).sin() + 0.5).collect();
    let x: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.013).sin() + 0.5).collect();
    let q4 = quantize_q4_0(&w);
    let (q8_x, q8_scales) = quantize_to_q8(&x);

    let metal_scores = metal.q4_matvec(&q4, &q8_x, &q8_scales, n, k).unwrap();
    let cpu_scores = CpuBackend.q4_matvec(&q4, &q8_x, &q8_scales, n, k).unwrap();

    let metal_zeros: Vec<usize> = metal_scores
        .iter()
        .enumerate()
        .filter(|(_, &v)| v.abs() <= 1e-9)
        .map(|(i, _)| i)
        .collect();
    let cpu_zeros: Vec<usize> = cpu_scores
        .iter()
        .enumerate()
        .filter(|(_, &v)| v.abs() <= 1e-9)
        .map(|(i, _)| i)
        .collect();

    assert!(
        cpu_zeros.is_empty(),
        "test invariant violated: CPU output should be all non-zero, \
         {} rows are zero (synth bias broken)",
        cpu_zeros.len(),
    );
    let preview = &metal_zeros[..metal_zeros.len().min(10)];
    assert!(
        metal_zeros.is_empty(),
        "Metal q4_matvec dropped {} of {n} rows (expected 0). \
         First zero rows: {preview:?}. \
         This is the 75 %-row regression — check that ops/q4_matvec.rs \
         imports geometry constants from the same shader module \
         (q4_matvec_v4) the pipeline is built from in metal/mod.rs.",
        metal_zeros.len(),
    );
}

/// N not divisible by ROWS_PER_TG (8) — the last TG has dead
/// simdgroups whose `row_idx >= N` guard must trip cleanly. Verifies
/// no spurious writes past `num_rows` and no missed rows at the tail.
#[test]
fn q4_matvec_metal_writes_every_row_misaligned_n() {
    let metal = get_metal();
    metal.set_flop_threshold(1);
    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_to_q8};

    // 1027 = 128 full TGs × 8 + 3 spillover rows.
    let n = 1027usize;
    let k = 128usize;
    let w: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.001).sin() + 0.5).collect();
    let x: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.013).sin() + 0.5).collect();
    let q4 = quantize_q4_0(&w);
    let (q8_x, q8_scales) = quantize_to_q8(&x);

    let metal_scores = metal.q4_matvec(&q4, &q8_x, &q8_scales, n, k).unwrap();
    let cpu_scores = CpuBackend.q4_matvec(&q4, &q8_x, &q8_scales, n, k).unwrap();

    assert_eq!(metal_scores.len(), n, "output length must equal num_rows");
    for (i, &v) in metal_scores.iter().enumerate() {
        assert!(
            v.abs() > 1e-9,
            "metal_scores[{i}] = {v} (should be non-zero)"
        );
    }
    // Q4 quantisation is lossy on both sides; agreement to ~1 % of
    // peak value is the kernel-equality bar (matches the rel<1e-2 check
    // in q4_matvec_cpu_vs_metal_at_vocab_scale).
    let max_abs = cpu_scores.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let max_diff = metal_scores
        .iter()
        .zip(&cpu_scores)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < max_abs * 1e-2,
        "metal vs cpu max_diff = {max_diff} (peak = {max_abs}, rel = {:.3e})",
        max_diff / max_abs.max(1e-9),
    );
}

/// Pin the contract between the live `KernelHandle` carried in
/// `MetalBackend.q4.matvec` and the `q4_matvec_v4` shader's
/// hard-coded row map.
///
/// Pre-2026-04-25 the dispatcher imported geometry constants from a
/// *different* shader module than the pipeline was built from — so
/// `num_tgs = num_rows / 32` over-divided and 75 % of rows dropped.
/// Post-fix, geometry travels with the pipeline via `KernelHandle`
/// (see `metal::kernel`), and a misnamed shader-module path simply
/// wouldn't compile.
///
/// Tested with N=64: post-fix `num_tgs = div_ceil(64, 8) = 8` so all
/// 64 rows are written. With the old (32, 1024) constants the v4
/// kernel would only cover rows 0..39 and rows 40..63 would stay at
/// zero. The handle on `metal.q4.matvec` is checked to expose the
/// correct geometry.
#[test]
fn q4_matvec_dispatch_geometry_matches_v4_kernel() {
    use larql_compute_metal::kernels::TiledKernel;
    use larql_compute_metal::shaders::q4_matvec_v4;

    // Compile-time contract: shader module's `Kernel` marker matches
    // the documented constants in the same file.
    assert_eq!(
        <q4_matvec_v4::Kernel as TiledKernel>::ROWS_PER_TG,
        8,
        "q4_matvec_v4 hard-codes `row_idx = tg_id * 8 + sg_id`",
    );
    assert_eq!(
        <q4_matvec_v4::Kernel as TiledKernel>::THREADS_PER_TG,
        256,
        "q4_matvec_v4 covers 8 rows × 32 lanes = 256 threads per TG",
    );
    assert_eq!(
        <q4_matvec_v4::Kernel as TiledKernel>::KERNEL_NAME,
        "q4_matvec_v4",
    );

    // Runtime contract: the live KernelHandle exposes the same values.
    let metal = get_metal();
    let kernel = &metal.q4.matvec;
    assert_eq!(kernel.kernel_name, "q4_matvec_v4");
    assert_eq!(kernel.rows_per_tg, 8);
    assert_eq!(kernel.threads_per_tg, 256);

    // Behavioural contract: at N=64 every row gets written. With the
    // pre-fix (32, 1024) geometry the v4 kernel would cover rows 0..39
    // only, leaving rows 40..63 zero.
    metal.set_flop_threshold(1);
    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_to_q8};
    let n = 64usize;
    let k = 64usize;
    let w: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.01).sin() + 0.5).collect();
    let x: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.013).sin() + 0.5).collect();
    let q4 = quantize_q4_0(&w);
    let (q8_x, q8_scales) = quantize_to_q8(&x);
    let metal_scores = metal.q4_matvec(&q4, &q8_x, &q8_scales, n, k).unwrap();
    for (i, &v) in metal_scores.iter().enumerate() {
        assert!(
            v.abs() > 1e-9,
            "row {i} dropped at N={n}; metal_scores[40..]={:?}",
            &metal_scores[40..],
        );
    }
}

/// Q4_0 + Q8 input matvec at the LM-head shape (vocab × hidden).
///
/// This is the path `lm_head_knn_backend` takes when the vindex has
/// either an `lm_head_q4.bin` file or a tied-embedding `lm_head_q4_synth`
/// built from f16 embeddings. CPU and Metal each implement
/// `q4_matvec(q4_data, q8_x, q8_scales, n, k)` independently — CPU
/// via the `larql-compute/src/csrc/q4_dot.c` ARM NEON kernel, Metal
/// via the `q4_matvec_v4` simdgroup shader. If the two kernels
/// disagree at vocab scale, every Q4_0 LM-head dispatch in
/// production will produce a different top-K on each backend.
#[test]
fn q4_matvec_cpu_vs_metal_at_vocab_scale() {
    if !run_enabled() {
        eprintln!(
            "skip: LARQL_RUN_LM_HEAD_BISECT=1 not set. \
             Allocates a ~2.68 GB f32 matrix + ~1.3 GB Q4_0; gated."
        );
        return;
    }

    let metal = get_metal();
    metal.set_flop_threshold(1);

    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_to_q8};

    let n = 262_144usize;
    let k = 2_560usize;
    eprintln!("Synthesising W [{n}, {k}] f32 → Q4_0 + Q8 query…");
    let (w, x) = synth_inputs(n, k);

    let w_flat: &[f32] = w.as_slice().expect("synth produced contiguous Array2");
    let q4_data = quantize_q4_0(w_flat);
    let (q8_x_i8, q8_scales) = quantize_to_q8(&x);
    eprintln!(
        "  Q4 bytes: {:.2} GB, Q8 input: {} elements, scales: {} blocks",
        q4_data.len() as f64 / 1e9,
        q8_x_i8.len(),
        q8_scales.len(),
    );

    let cpu_scores = CpuBackend
        .q4_matvec(&q4_data, &q8_x_i8, &q8_scales, n, k)
        .expect("CpuBackend.q4_matvec should always return Some");
    let metal_scores = metal
        .q4_matvec(&q4_data, &q8_x_i8, &q8_scales, n, k)
        .expect("MetalBackend.q4_matvec should always return Some");

    let cpu_top5 = top5(&cpu_scores);
    let metal_top5 = top5(&metal_scores);
    eprintln!("CPU   top-5: {:?}", cpu_top5);
    eprintln!("Metal top-5: {:?}", metal_top5);

    let cpu_top1 = cpu_top5[0];
    let metal_top1 = metal_top5[0];

    let mut cpu_set: Vec<u32> = cpu_top5.iter().map(|t| t.0).collect();
    let mut metal_set: Vec<u32> = metal_top5.iter().map(|t| t.0).collect();
    cpu_set.sort_unstable();
    metal_set.sort_unstable();

    if cpu_set != metal_set {
        // Find the boundary — first row where Metal outputs zero.
        let nonzero_count = metal_scores.iter().filter(|&&v| v.abs() > 1e-9).count();
        let first_zero = metal_scores.iter().position(|&v| v.abs() <= 1e-9);
        let last_nonzero = metal_scores.iter().rposition(|&v| v.abs() > 1e-9);
        eprintln!(
            "\n  Metal output diagnostics:\n    \
             nonzero rows: {nonzero_count} / {n}\n    \
             first zero row: {first_zero:?}\n    \
             last nonzero row: {last_nonzero:?}\n    \
             metal_scores[65535]={:.6} metal_scores[65536]={:.6}\n    \
             metal_scores[65537]={:.6} metal_scores[131072]={:.6}\n    \
             metal_scores[200000]={:.6} metal_scores[262143]={:.6}",
            metal_scores[65535],
            metal_scores[65536],
            metal_scores[65537],
            metal_scores[131072],
            metal_scores[200000],
            metal_scores[262143],
        );
        let cpu_score_at = |id: u32| cpu_scores[id as usize];
        let metal_score_at = |id: u32| metal_scores[id as usize];
        eprintln!("\n  Score on CPU at IDs Metal returned:");
        for &(id, _s) in metal_top5.iter() {
            eprintln!(
                "    id {id}: cpu={:.4} metal={:.4}",
                cpu_score_at(id),
                metal_score_at(id)
            );
        }
        eprintln!("  Score on Metal at IDs CPU returned:");
        for &(id, _s) in cpu_top5.iter() {
            eprintln!(
                "    id {id}: cpu={:.4} metal={:.4}",
                cpu_score_at(id),
                metal_score_at(id)
            );
        }
    }

    assert_eq!(
        cpu_set, metal_set,
        "Q4_0 matvec top-5 sets diverge at vocab-scale (N=262144 × K=2560). \
         This is the DIRECT cause of the open Gemma 3/4 CPU/Metal LM-head \
         divergence in `test_logits_goldens`. CPU NEON kernel and Metal \
         simdgroup shader produce different top-5 token IDs for the same \
         Q4_0 weights × Q8 query."
    );

    let logit_diff = (cpu_top1.1 - metal_top1.1).abs();
    let max_abs = cpu_scores
        .iter()
        .map(|v| v.abs())
        .fold(0.0f32, f32::max)
        .max(1e-6);
    let rel = logit_diff / max_abs;
    assert!(
        rel < 1e-2,
        "Q4 top-1 logit diverges: cpu={:.6} metal={:.6} (rel={:.3e})",
        cpu_top1.1,
        metal_top1.1,
        rel,
    );

    eprintln!(
        "✓ Q4 matvec vocab-scale CPU vs Metal: top-5 sets match, \
         top-1 logit Δ={:.3e} (rel {:.2e})",
        logit_diff, rel,
    );
}
