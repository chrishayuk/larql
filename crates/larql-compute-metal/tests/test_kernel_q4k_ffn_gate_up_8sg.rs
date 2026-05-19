//! Parity + perf for the 8-simdgroup TG variant of `q4k_ffn_gate_up`.
//!
//! Math is identical to the production 4-simdgroup kernel — only the
//! threadgroup geometry changes (256 threads / 8 simdgroups / 8
//! rows/TG vs the production 128 / 4 / 4). Each lane still processes
//! one output row's contribution (`nr0=1`), so per-thread register
//! footprint is unchanged.
//!
//! Parity must be exact (bit-equal) since the per-row math, lane
//! mapping within each simdgroup, and reduction are all identical.
//! The only difference is how many rows a single TG produces.

#![cfg(target_os = "macos")]

extern crate blas_src;

use larql_compute::cpu::ops::q4_common::quantize_q4_k;
use larql_compute_metal::MetalBackend;
use std::ffi::c_void;
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

fn rms_normed(len: usize, seed: u64) -> Vec<f32> {
    synth(len, seed).into_iter().map(|v| v * 2.0).collect()
}

/// Dispatch using a specific gate+up pipeline. Returns `(gate_out, up_out)`.
#[allow(clippy::too_many_arguments)]
fn dispatch(
    metal: &MetalBackend,
    pipeline: &metal::ComputePipelineState,
    rows_per_tg: u64,
    threads_per_tg: u64,
    gate_q4k: &[u8],
    up_q4k: &[u8],
    x: &[f32],
    n: usize,
    k: usize,
) -> (Vec<f32>, Vec<f32>) {
    let bufs = metal.bufs();
    let wg = bufs.get_bytes(gate_q4k);
    let wu = bufs.get_bytes(up_q4k);
    let xb = bufs.transient_from_f32(x);
    let go = bufs.output((n * 4) as u64);
    let uo = bufs.output((n * 4) as u64);
    let n_val = n as u32;
    let k_val = k as u32;
    let tgs = (n as u64).div_ceil(rows_per_tg);

    let cmd = metal.queue().new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(&wg), 0);
    enc.set_buffer(1, Some(&wu), 0);
    enc.set_buffer(2, Some(&xb), 0);
    enc.set_buffer(3, Some(&go), 0);
    enc.set_buffer(4, Some(&uo), 0);
    enc.set_bytes(5, 4, &n_val as *const u32 as *const c_void);
    enc.set_bytes(6, 4, &k_val as *const u32 as *const c_void);
    // Both gate and up share the same dispatch — the kernel internally
    // partitions tg_id < tgs into gate, tg_id >= tgs into up.
    enc.dispatch_thread_groups(
        metal::MTLSize::new(tgs * 2, 1, 1),
        metal::MTLSize::new(threads_per_tg, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    (
        larql_compute_metal::buffers::read_buffer_f32(&go, n),
        larql_compute_metal::buffers::read_buffer_f32(&uo, n),
    )
}

#[test]
fn q4k_ffn_gate_up_8sg_matches_4sg_bit_equal() {
    let metal = match MetalBackend::new() {
        Some(m) => m,
        None => return,
    };

    // Production-ish shape but small enough to exhibit ragged-N
    // (N=33 means TG count differs between 4sg = ceil(33/4)=9 and
    // 8sg = ceil(33/8)=5). The early-exit guard `if row_idx >= N
    // return` must work in both.
    let n = 33usize;
    let k = 256usize;

    let gate_w = synth(n * k, 91);
    let up_w = synth(n * k, 93);
    let x = rms_normed(k, 95);
    let gate_q4k = quantize_q4_k(&gate_w);
    let up_q4k = quantize_q4_k(&up_w);

    use larql_compute_metal::shaders::{q4k_ffn_gate_up as p4, q4k_ffn_gate_up_8sg as p8};
    let (g4, u4) = dispatch(
        &metal,
        &metal.ffn.q4k_ffn_gate_up_pipeline.state,
        p4::ROWS_PER_TG,
        p4::THREADS_PER_TG,
        &gate_q4k,
        &up_q4k,
        &x,
        n,
        k,
    );
    let (g8, u8) = dispatch(
        &metal,
        &metal.ffn.q4k_ffn_gate_up_8sg_pipeline.state,
        p8::ROWS_PER_TG,
        p8::THREADS_PER_TG,
        &gate_q4k,
        &up_q4k,
        &x,
        n,
        k,
    );

    assert_eq!(g4.len(), g8.len(), "gate output length");
    assert_eq!(u4.len(), u8.len(), "up output length");
    // Bit-equal: math is identical, only the TG dispatch geometry changed.
    for (i, (a, b)) in g4.iter().zip(&g8).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "gate row {i}: 4sg={a} != 8sg={b}");
    }
    for (i, (a, b)) in u4.iter().zip(&u8).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "up row {i}: 4sg={a} != 8sg={b}");
    }
}

#[test]
fn q4k_ffn_gate_up_8sg_perf_vs_4sg() {
    if std::env::var("LARQL_PERF_SPOT_CHECK").is_err() {
        return; // default-skip; opt-in
    }
    let metal = match MetalBackend::new() {
        Some(m) => m,
        None => return,
    };

    // Production shape: Gemma 3 4B gate+up.
    let n = 10240usize;
    let k = 2560usize;

    let gate_w = synth(n * k, 21);
    let up_w = synth(n * k, 23);
    let x = rms_normed(k, 27);
    let gate_q4k = quantize_q4_k(&gate_w);
    let up_q4k = quantize_q4_k(&up_w);

    use larql_compute_metal::shaders::{q4k_ffn_gate_up as p4, q4k_ffn_gate_up_8sg as p8};

    // Warmup both paths.
    for _ in 0..5 {
        let _ = dispatch(
            &metal,
            &metal.ffn.q4k_ffn_gate_up_pipeline.state,
            p4::ROWS_PER_TG,
            p4::THREADS_PER_TG,
            &gate_q4k,
            &up_q4k,
            &x,
            n,
            k,
        );
        let _ = dispatch(
            &metal,
            &metal.ffn.q4k_ffn_gate_up_8sg_pipeline.state,
            p8::ROWS_PER_TG,
            p8::THREADS_PER_TG,
            &gate_q4k,
            &up_q4k,
            &x,
            n,
            k,
        );
    }

    let iters = 20;
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = dispatch(
            &metal,
            &metal.ffn.q4k_ffn_gate_up_pipeline.state,
            p4::ROWS_PER_TG,
            p4::THREADS_PER_TG,
            &gate_q4k,
            &up_q4k,
            &x,
            n,
            k,
        );
    }
    let p4_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    let t1 = Instant::now();
    for _ in 0..iters {
        let _ = dispatch(
            &metal,
            &metal.ffn.q4k_ffn_gate_up_8sg_pipeline.state,
            p8::ROWS_PER_TG,
            p8::THREADS_PER_TG,
            &gate_q4k,
            &up_q4k,
            &x,
            n,
            k,
        );
    }
    let p8_ms = t1.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    // 30 MB per call (gate+up weights = 2 × 14.7 MB; X is tiny).
    let mb = 2.0 * (n * k) as f64 * 0.5625 / 1e6;
    let p4_gbs = mb / p4_ms;
    let p8_gbs = mb / p8_ms;
    let speedup = p4_ms / p8_ms;
    eprintln!(
        "q4k_ffn_gate_up perf @ N={n} K={k}: 4sg {p4_ms:.3}ms ({p4_gbs:.1} GB/s),  8sg {p8_ms:.3}ms ({p8_gbs:.1} GB/s),  speedup {speedup:.2}×"
    );
    // No assertion on direction — record the number, decide adoption
    // separately. Just sanity that both ran.
    assert!(p4_ms > 0.0 && p8_ms > 0.0);
}
