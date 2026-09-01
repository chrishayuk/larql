//! Timing harnesses and the bandwidth helper.
//!
//! Split out of `shader_bench.rs`; [`super`] composes the pieces.

#[allow(unused_imports)]
use super::*;
use crate::buffers::read_buffer_f32;
use crate::kernels::KernelHandle;
use crate::MetalBackend;
use metal::{Buffer, ComputeCommandEncoderRef};
use std::time::Instant;

#[allow(clippy::too_many_arguments)]
pub(crate) fn measure_tiled(
    metal: &MetalBackend,
    cfg: &Config,
    name: &'static str,
    family: &'static str,
    kh: &KernelHandle,
    shape: String,
    bytes_per_call: u64,
    output: &Buffer,
    output_len: usize,
    sanity: &'static str,
    note: &'static str,
    encode: impl Fn(&ComputeCommandEncoderRef),
) -> BenchResult {
    let (isolated_ms, isolated_sd_ms) = measure_isolated(metal, cfg.warmup, cfg.iters, &encode);
    let batched_ms = measure_batched(metal, cfg.warmup, cfg.iters, cfg.n_layers, &encode);
    let output = read_buffer_f32(output, output_len);
    let output_nonzero = output.iter().filter(|v| v.abs() > 1e-10).count();
    BenchResult {
        name,
        family,
        status: "bench",
        shape,
        rows_per_tg: Some(kh.rows_per_tg),
        threads_per_tg: Some(kh.threads_per_tg),
        bytes_per_call,
        isolated_ms: Some(isolated_ms),
        isolated_sd_ms: Some(isolated_sd_ms),
        batched_ms: Some(batched_ms),
        batched_gbs: Some(gbs(bytes_per_call, batched_ms)),
        output_nonzero: Some(output_nonzero),
        sanity,
        note,
    }
}

pub(crate) fn measure_isolated(
    metal: &MetalBackend,
    warmup: usize,
    iters: usize,
    encode: &impl Fn(&ComputeCommandEncoderRef),
) -> (f64, f64) {
    let mut times = Vec::with_capacity(iters);
    for i in 0..warmup + iters {
        let t = Instant::now();
        let cmd = metal.queue().new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        encode(enc);
        enc.end_encoding();
        cmd.commit();
        crate::cb_status::wait_or_abort(
            cmd,
            "crates/larql-compute-metal/src/diag/shader_bench/measure.rs:64",
        );
        if i >= warmup {
            times.push(t.elapsed().as_secs_f64() * 1000.0);
        }
    }
    (mean(&times), stddev(&times))
}

pub(crate) fn measure_batched(
    metal: &MetalBackend,
    warmup: usize,
    iters: usize,
    n_layers: usize,
    encode: &impl Fn(&ComputeCommandEncoderRef),
) -> f64 {
    let mut times = Vec::with_capacity(iters);
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
            "crates/larql-compute-metal/src/diag/shader_bench/measure.rs:89",
        );
        if i >= warmup {
            times.push(t.elapsed().as_secs_f64() * 1000.0 / n_layers as f64);
        }
    }
    mean(&times)
}

pub(crate) fn gbs(bytes: u64, ms: f64) -> f64 {
    bytes as f64 / 1e6 / ms
}

pub(crate) fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

pub(crate) fn stddev(v: &[f64]) -> f64 {
    let m = mean(v);
    (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
}

pub(crate) fn synth_f32(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let f = i as f32;
            ((seed + f * 0.013).sin() * 0.35) + ((seed * 0.3 + f * 0.007).cos() * 0.15)
        })
        .collect()
}
