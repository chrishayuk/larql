//! Isolated / batched / single-command-buffer timing harnesses.
//!
//! Split out of `kernel_profile.rs`; [`super`] composes the pieces.

#[allow(unused_imports)]
use super::*;
use std::time::Instant;

pub(super) fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}
pub(super) fn stddev(v: &[f64]) -> f64 {
    let m = mean(v);
    (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
}

pub(super) fn synth_f32(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| (seed + i as f32 * 0.007).sin() * 0.4)
        .collect()
}

pub(super) fn measure_isolated(warmup: usize, iters: usize, f: &mut impl FnMut()) -> (f64, f64) {
    let mut times = Vec::with_capacity(iters);
    for i in 0..warmup + iters {
        let t = Instant::now();
        f();
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        if i >= warmup {
            times.push(ms);
        }
    }
    (mean(&times), stddev(&times))
}

/// Measure batched throughput where each iteration runs `f()` `n_layers`
/// times. **`f()` is responsible for its own cmd-buffer + commit + wait.**
///
/// This MIS-measures throughput when used with closures that create one
/// cmd-buffer per call: each cmd-buffer costs ~10 µs of dispatch overhead
/// that gets billed against the kernel time. Real production runs all
/// `n_layers` dispatches in ONE cmd buffer with a single commit+wait —
/// see [`measure_single_cmdbuf_batched`] for that.
///
/// Kept for callers who genuinely want per-call cmd-buffer overhead in
/// the measurement (rare).
#[allow(dead_code)]
pub(super) fn measure_batched(
    warmup: usize,
    iters: usize,
    n_layers: usize,
    f: &mut impl FnMut(),
) -> f64 {
    let mut times = Vec::with_capacity(iters);
    for i in 0..warmup + iters {
        let t = Instant::now();
        for _ in 0..n_layers {
            f();
        }
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        if i >= warmup {
            times.push(ms / n_layers as f64);
        }
    }
    mean(&times)
}

/// Measure batched throughput with all `n_layers` dispatches in ONE cmd
/// buffer, single commit+wait. This is what production decode actually
/// does (all of a token's per-layer kernels live in one cmd buffer), so
/// the GB/s number reflects real per-kernel cost without dispatch
/// overhead pollution.
///
/// `encode` must NOT call `commit`/`wait_until_completed`/`end_encoding`
/// — this function owns the cmd-buffer lifecycle.
///
/// Discovered 2026-04-28: the older `measure_batched` was being used
/// with closures that did per-call commit+wait, undercounting q6k_matvec
/// throughput by 4× (74 vs real 315 GB/s). See ROADMAP P0 "Decode kernel
/// optimization → Track A" for the bisect.
pub(super) fn measure_single_cmdbuf_batched(
    metal: &crate::MetalBackend,
    warmup: usize,
    iters: usize,
    n_layers: usize,
    encode: &impl Fn(&metal::ComputeCommandEncoderRef),
) -> f64 {
    let mut times: Vec<f64> = Vec::with_capacity(iters);
    for i in 0..warmup + iters {
        let t = Instant::now();
        let cmd = metal.queue().new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        for _ in 0..n_layers {
            encode(enc);
        }
        enc.end_encoding();
        cmd.commit();
        crate::cb_status::wait_or_abort(
            cmd,
            "crates/larql-compute-metal/src/diag/kernel_profile/measure.rs:98",
        );
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        if i >= warmup {
            times.push(ms / n_layers as f64);
        }
    }
    mean(&times)
}
