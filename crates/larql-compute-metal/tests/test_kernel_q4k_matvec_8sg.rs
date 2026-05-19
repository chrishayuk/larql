//! Parity test for the 8-simdgroup Q4_K matvec variant. Math is
//! identical to the production 4sg kernel; only TG geometry changes.
//! Output must be bit-equal.

#![cfg(target_os = "macos")]

extern crate blas_src;

use larql_compute::cpu::ops::q4_common::quantize_q4_k;
use larql_compute::prelude::*;
use larql_compute_metal::MetalBackend;
use std::ffi::c_void;

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
fn q4k_matvec_stride32_matches_cpu() {
    let metal = match MetalBackend::new() {
        Some(m) => m,
        None => return,
    };

    let n = 17usize;
    let k = 512usize;

    let w = synth(n * k, 81);
    let x = synth(k, 83);
    let w_q4k = quantize_q4_k(&w);

    let cpu = larql_compute::CpuBackend;
    let cpu_out = cpu
        .q4k_matvec(&w_q4k, &x, n, k)
        .expect("CPU q4k matvec should be available");

    use larql_compute_metal::shaders::q4k_matvec_stride32 as p;
    let metal_out = dispatch(
        &metal,
        &metal.quant.q4k_matvec_stride32_pipeline.state,
        p::ROWS_PER_TG,
        p::THREADS_PER_TG,
        &w_q4k,
        &x,
        n,
        k,
    );

    for (i, (a, b)) in cpu_out.iter().zip(&metal_out).enumerate() {
        let diff = (a - b).abs();
        assert!(
            diff < 0.5,
            "q4k_matvec_stride32 row {i}: cpu={a} metal={b} diff={diff}"
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn dispatch(
    metal: &MetalBackend,
    pipeline: &metal::ComputePipelineState,
    rows_per_tg: u64,
    threads_per_tg: u64,
    w_q4k: &[u8],
    x: &[f32],
    n: usize,
    k: usize,
) -> Vec<f32> {
    let bufs = metal.bufs();
    let wb = bufs.get_bytes(w_q4k);
    let xb = bufs.transient_from_f32(x);
    let ob = bufs.output((n * 4) as u64);
    let n_val = n as u32;
    let k_val = k as u32;
    let tgs = (n as u64).div_ceil(rows_per_tg);

    let cmd = metal.queue().new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(&wb), 0);
    enc.set_buffer(1, Some(&xb), 0);
    enc.set_buffer(2, Some(&ob), 0);
    enc.set_bytes(3, 4, &n_val as *const u32 as *const c_void);
    enc.set_bytes(4, 4, &k_val as *const u32 as *const c_void);
    enc.dispatch_thread_groups(
        metal::MTLSize::new(tgs, 1, 1),
        metal::MTLSize::new(threads_per_tg, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    larql_compute_metal::buffers::read_buffer_f32(&ob, n)
}

#[test]
fn q4k_matvec_8sg_matches_4sg_bit_equal() {
    let metal = match MetalBackend::new() {
        Some(m) => m,
        None => return,
    };

    // Ragged N to exercise the early-exit guard at TG boundary.
    let n = 17usize;
    let k = 256usize;

    let w = synth(n * k, 71);
    let x = synth(k, 73);
    let w_q4k = quantize_q4_k(&w);

    use larql_compute_metal::shaders::{q4k_matvec as p4, q4k_matvec_8sg as p8};
    let r4 = dispatch(
        &metal,
        &metal.quant.q4k_matvec_4sg_pipeline.state,
        p4::ROWS_PER_TG,
        p4::THREADS_PER_TG,
        &w_q4k,
        &x,
        n,
        k,
    );
    let r8 = dispatch(
        &metal,
        &metal.quant.q4k_matvec_8sg_pipeline.state,
        p8::ROWS_PER_TG,
        p8::THREADS_PER_TG,
        &w_q4k,
        &x,
        n,
        k,
    );

    assert_eq!(r4.len(), r8.len());
    for (i, (a, b)) in r4.iter().zip(&r8).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "q4k_matvec row {i}: 4sg={a} != 8sg={b}"
        );
    }
}
