//! Parity + perf for the 8-simdgroup TG variant of `q6k_matvec`.
//!
//! Math is identical to the production 4-simdgroup kernel — only the
//! threadgroup geometry changes (256 threads / 8 simdgroups / 8
//! rows/TG vs the production 128 / 4 / 4). Output must be bit-equal.

#![cfg(target_os = "macos")]

extern crate blas_src;

use larql_compute::cpu::ops::q4_common::quantize_q6_k;
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

#[allow(clippy::too_many_arguments)]
fn dispatch_q6k(
    metal: &MetalBackend,
    pipeline: &metal::ComputePipelineState,
    rows_per_tg: u64,
    threads_per_tg: u64,
    w_q6k: &[u8],
    x: &[f32],
    n: usize,
    k: usize,
) -> Vec<f32> {
    let bufs = metal.bufs();
    let wb = bufs.get_bytes(w_q6k);
    let xb = bufs.transient_from_f32(x);
    let ob = bufs.output((n * 4) as u64);
    let n_val = n as u32;
    let k_val = k as u32;
    let n_tgs = (n as u64).div_ceil(rows_per_tg);

    let cmd = metal.queue().new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(&wb), 0);
    enc.set_buffer(1, Some(&xb), 0);
    enc.set_buffer(2, Some(&ob), 0);
    enc.set_bytes(3, 4, &n_val as *const u32 as *const c_void);
    enc.set_bytes(4, 4, &k_val as *const u32 as *const c_void);
    enc.dispatch_thread_groups(
        metal::MTLSize::new(n_tgs, 1, 1),
        metal::MTLSize::new(threads_per_tg, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    larql_compute_metal::buffers::read_buffer_f32(&ob, n)
}

#[test]
fn q6k_matvec_8sg_matches_4sg_bit_equal() {
    let metal = match MetalBackend::new() {
        Some(m) => m,
        None => return,
    };

    // Ragged N to exercise the early-exit guard.
    let n = 17usize;
    let k = 256usize;

    let w_full = synth(n * k, 71);
    let x = synth(k, 73);
    let w_q6k = quantize_q6_k(&w_full);

    use larql_compute_metal::shaders::{q6k_matvec as p4, q6k_matvec_8sg as p8};
    let r4 = dispatch_q6k(
        &metal,
        &metal.quant.q6k_matvec_4sg_pipeline.state,
        p4::ROWS_PER_TG,
        p4::THREADS_PER_TG,
        &w_q6k,
        &x,
        n,
        k,
    );
    let r8 = dispatch_q6k(
        &metal,
        &metal.quant.q6k_matvec_8sg_pipeline.state,
        p8::ROWS_PER_TG,
        p8::THREADS_PER_TG,
        &w_q6k,
        &x,
        n,
        k,
    );

    assert_eq!(r4.len(), r8.len());
    for (i, (a, b)) in r4.iter().zip(&r8).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "q6k_matvec row {i}: 4sg={a} != 8sg={b} — math should be bit-equal, only TG dispatch geometry changed"
        );
    }
}

#[test]
fn q6k_matvec_8sg_perf_vs_4sg() {
    if std::env::var("LARQL_PERF_SPOT_CHECK").is_err() {
        return;
    }
    let metal = match MetalBackend::new() {
        Some(m) => m,
        None => return,
    };

    // Production shape: Gemma 3 4B FFN down (N=2560, K=10240).
    let n = 2560usize;
    let k = 10240usize;

    let w_full = synth(n * k, 31);
    let x = synth(k, 37);
    let w_q6k = quantize_q6_k(&w_full);

    use larql_compute_metal::shaders::{q6k_matvec as p4, q6k_matvec_8sg as p8};

    for _ in 0..5 {
        let _ = dispatch_q6k(
            &metal,
            &metal.quant.q6k_matvec_4sg_pipeline.state,
            p4::ROWS_PER_TG,
            p4::THREADS_PER_TG,
            &w_q6k,
            &x,
            n,
            k,
        );
        let _ = dispatch_q6k(
            &metal,
            &metal.quant.q6k_matvec_8sg_pipeline.state,
            p8::ROWS_PER_TG,
            p8::THREADS_PER_TG,
            &w_q6k,
            &x,
            n,
            k,
        );
    }

    let iters = 30;
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = dispatch_q6k(
            &metal,
            &metal.quant.q6k_matvec_4sg_pipeline.state,
            p4::ROWS_PER_TG,
            p4::THREADS_PER_TG,
            &w_q6k,
            &x,
            n,
            k,
        );
    }
    let p4_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    let t1 = Instant::now();
    for _ in 0..iters {
        let _ = dispatch_q6k(
            &metal,
            &metal.quant.q6k_matvec_8sg_pipeline.state,
            p8::ROWS_PER_TG,
            p8::THREADS_PER_TG,
            &w_q6k,
            &x,
            n,
            k,
        );
    }
    let p8_ms = t1.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    let mb = (n * (k / 256) * 210) as f64 / 1e6;
    eprintln!(
        "q6k_matvec perf @ N={n} K={k}: 4sg {p4_ms:.3}ms ({:.1} GB/s),  8sg {p8_ms:.3}ms ({:.1} GB/s),  speedup {:.2}×",
        mb / p4_ms,
        mb / p8_ms,
        p4_ms / p8_ms,
    );
    assert!(p4_ms > 0.0 && p8_ms > 0.0);
}
