//! Correctness tests for the dispatch-fusion kernels shipped in 2026-04-25:
//!
//! - `residual_norm_store`: writes both the normed FFN input AND the raw
//!   residual sum in a single cooperative pass, replacing the two-dispatch
//!   `residual_norm + residual_add` pair.
//! - `q4k_q6k_qkv_proj_normed`: fused input-norm + QKV projection for
//!   the Q4_K Q/K + Q6_K V mixed-format path (Gemma 3 4B production).

#![cfg(target_os = "macos")]

extern crate blas_src;

use larql_compute::prelude::*;

#[path = "common/mod.rs"]
mod common;
use common::{get_metal, max_diff};

// ── residual_norm_store ──

/// `residual_norm_store` must write the SAME normed output as `residual_norm`
/// AND the raw sum (a+b) into a second buffer. Any difference means the
/// post-FFN residual add (which reads `sum_out`) or the FFN norm input
/// (which reads `norm_out`) would be wrong.
#[test]
fn residual_norm_store_matches_residual_norm_and_raw_sum() {
    let metal = get_metal();
    let len = 2560usize; // production hidden size
    let eps = 1e-6f32;
    let offset = 1.0f32;

    let a: Vec<f32> = (0..len).map(|i| ((i as f32 * 0.007).sin()) * 0.4).collect();
    let b: Vec<f32> = (0..len).map(|i| ((i as f32 * 0.011).cos()) * 0.3).collect();
    let weight: Vec<f32> = (0..len)
        .map(|i| 0.9 + (i as f32 * 0.001).sin() * 0.1)
        .collect();

    // CPU reference
    let sum: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
    let sum_sq: f32 = sum.iter().map(|v| v * v).sum();
    let rms = 1.0 / (sum_sq / len as f32 + eps).sqrt();
    let cpu_norm: Vec<f32> = sum
        .iter()
        .zip(weight.iter())
        .map(|(s, w)| s * (w + offset) * rms)
        .collect();

    // Metal: residual_norm_store
    let buf_a = metal.bufs().transient_from_f32(&a);
    let buf_b = metal.bufs().transient_from_f32(&b);
    let buf_w = metal.bufs().get_f32(&weight);
    let buf_norm = metal.bufs().output((len * 4) as u64);
    let buf_sum = metal.bufs().output((len * 4) as u64);
    let len_val = len as u32;

    // `residual_norm_store` takes Granite `b_scale` at buffer(8); 1.0
    // recovers the original behaviour these parity tests check.
    let b_scale: f32 = 1.0;
    let cmd = metal.queue().new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&metal.norms.residual_norm_store_pipeline);
    enc.set_buffer(0, Some(&buf_a), 0);
    enc.set_buffer(1, Some(&buf_b), 0);
    enc.set_buffer(2, Some(&buf_w), 0);
    enc.set_buffer(3, Some(&buf_norm), 0);
    enc.set_buffer(4, Some(&buf_sum), 0);
    enc.set_bytes(5, 4, &len_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(6, 4, &eps as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(7, 4, &offset as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(8, 4, &b_scale as *const f32 as *const std::ffi::c_void);
    enc.dispatch_thread_groups(
        metal::MTLSize::new(1, 1, 1),
        metal::MTLSize::new(256_u64.min(len as u64), 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let got_norm = larql_compute_metal::buffers::read_buffer_f32(&buf_norm, len);
    let got_sum = larql_compute_metal::buffers::read_buffer_f32(&buf_sum, len);

    let d_norm = max_diff(&cpu_norm, &got_norm);
    assert!(
        d_norm < 1e-4,
        "residual_norm_store norm_out: max_diff {d_norm:.3e} vs residual_norm reference"
    );

    let d_sum = max_diff(&sum, &got_sum);
    assert!(
        d_sum < 1e-6,
        "residual_norm_store sum_out: max_diff {d_sum:.3e} vs raw a+b"
    );
}

// ── q4k_q6k_qkv_proj_normed ──

/// `q4k_q6k_qkv_proj_normed` must produce the same Q/K/V outputs as
/// a separate `rms_norm` + `q4k_q6k_qkv_proj` pair. Any divergence
/// means the fused-norm fast path is computing the wrong normalization.
#[test]
fn q4k_q6k_qkv_proj_normed_matches_separate_norm_and_proj() {
    let metal = get_metal();

    use larql_compute::cpu::ops::q4_common::{quantize_q4_k, quantize_q6_k};
    use larql_compute_metal::shaders::q4k_q6k_qkv_proj as sh;

    let q_rows = 512usize; // scaled-down Gemma 3 4B (8192→512 to keep test fast)
    let kv_rows = 256usize;
    let hidden = 512usize; // must be multiple of 256

    let wq_f32: Vec<f32> = (0..q_rows * hidden)
        .map(|i| ((i as f32 * 0.001).cos()) * 0.5)
        .collect();
    let wk_f32: Vec<f32> = (0..kv_rows * hidden)
        .map(|i| ((i as f32 * 0.002).sin()) * 0.5)
        .collect();
    let wv_f32: Vec<f32> = (0..kv_rows * hidden)
        .map(|i| ((i as f32 * 0.003).cos()) * 0.4)
        .collect();
    let h_raw: Vec<f32> = (0..hidden)
        .map(|i| ((i as f32 * 0.013).sin() + 0.2) * 0.4)
        .collect();
    let norm_w: Vec<f32> = (0..hidden)
        .map(|i| 0.9 + (i as f32 * 0.001).sin() * 0.1)
        .collect();

    let wq_q4k = quantize_q4_k(&wq_f32);
    let wk_q4k = quantize_q4_k(&wk_f32);
    let wv_q6k = quantize_q6_k(&wv_f32);

    let eps = 1e-6f32;
    let offset = 1.0f32; // Gemma 3 norm_offset

    // Reference: CPU rms_norm then fused QKV via existing tested kernel
    let sum_sq: f32 = h_raw.iter().map(|v| v * v).sum();
    let rms = 1.0 / (sum_sq / hidden as f32 + eps).sqrt();
    let h_normed: Vec<f32> = h_raw
        .iter()
        .zip(norm_w.iter())
        .map(|(h, w)| h * rms * (offset + w))
        .collect();

    // Run existing qkv_proj (non-normed) against pre-normed h
    let ref_q = metal
        .q4k_matvec(&wq_q4k, &h_normed, q_rows, hidden)
        .unwrap();
    let ref_k = metal
        .q4k_matvec(&wk_q4k, &h_normed, kv_rows, hidden)
        .unwrap();
    let ref_v = metal
        .q6k_matvec(&wv_q6k, &h_normed, kv_rows, hidden)
        .unwrap();

    // Fused normed kernel
    let wq_buf = metal.bufs().get_bytes(&wq_q4k);
    let wk_buf = metal.bufs().get_bytes(&wk_q4k);
    let wv_buf = metal.bufs().get_bytes(&wv_q6k);
    let h_buf = metal.bufs().transient_from_f32(&h_raw);
    let nw_buf = metal.bufs().get_f32(&norm_w);
    let q_out = metal.bufs().output((q_rows * 4) as u64);
    let k_out = metal.bufs().output((kv_rows * 4) as u64);
    let v_out = metal.bufs().output((kv_rows * 4) as u64);

    let total_rows = (q_rows + kv_rows + kv_rows) as u64;
    let num_tgs = total_rows.div_ceil(sh::ROWS_PER_TG);
    let q_u = q_rows as u32;
    let kv_u = kv_rows as u32;
    let h_u = hidden as u32;

    let cmd = metal.queue().new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&metal.attention.q4k_q6k_qkv_proj_normed_pipeline.state);
    enc.set_buffer(0, Some(&wq_buf), 0);
    enc.set_buffer(1, Some(&wk_buf), 0);
    enc.set_buffer(2, Some(&wv_buf), 0);
    enc.set_buffer(3, Some(&h_buf), 0);
    enc.set_buffer(4, Some(&nw_buf), 0);
    enc.set_buffer(5, Some(&q_out), 0);
    enc.set_buffer(6, Some(&k_out), 0);
    enc.set_buffer(7, Some(&v_out), 0);
    enc.set_bytes(8, 4, &q_u as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(9, 4, &kv_u as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(10, 4, &kv_u as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(11, 4, &h_u as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(12, 4, &eps as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(13, 4, &offset as *const f32 as *const std::ffi::c_void);
    enc.dispatch_thread_groups(
        metal::MTLSize::new(num_tgs, 1, 1),
        metal::MTLSize::new(sh::THREADS_PER_TG, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let got_q = larql_compute_metal::buffers::read_buffer_f32(&q_out, q_rows);
    let got_k = larql_compute_metal::buffers::read_buffer_f32(&k_out, kv_rows);
    let got_v = larql_compute_metal::buffers::read_buffer_f32(&v_out, kv_rows);

    let threshold = 0.001; // 0.1% relative
    let max_abs_q = ref_q
        .iter()
        .map(|v: &f32| v.abs())
        .fold(0.0f32, f32::max)
        .max(1e-6);
    let dq = max_diff(&ref_q, &got_q);
    assert!(
        dq < max_abs_q * threshold,
        "q4k_q6k_qkv_proj_normed Q: max_diff {dq:.3e} exceeds {:.3e}",
        max_abs_q * threshold
    );
    let max_abs_k = ref_k
        .iter()
        .map(|v: &f32| v.abs())
        .fold(0.0f32, f32::max)
        .max(1e-6);
    let dk = max_diff(&ref_k, &got_k);
    assert!(
        dk < max_abs_k * threshold,
        "q4k_q6k_qkv_proj_normed K: max_diff {dk:.3e} exceeds {:.3e}",
        max_abs_k * threshold
    );
    let max_abs_v = ref_v
        .iter()
        .map(|v: &f32| v.abs())
        .fold(0.0f32, f32::max)
        .max(1e-6);
    let dv = max_diff(&ref_v, &got_v);
    assert!(
        dv < max_abs_v * threshold,
        "q4k_q6k_qkv_proj_normed V: max_diff {dv:.3e} exceeds {:.3e}",
        max_abs_v * threshold
    );
}

/// Production-shape regression for the mixed Q4_K/Q6_K fused-QKV path.
/// Gemma 3 4B uses hidden=2560 (10 super-blocks/row); the small test
/// above uses hidden=512 (2 super-blocks). The roadmap previously
/// flagged this kernel as drifting on the V branch — keep a real-shape
/// parity check so any future regression at the production K is caught
/// immediately, not via a model-output bug report.
#[test]
fn q4k_q6k_qkv_proj_normed_matches_at_production_hidden() {
    let metal = get_metal();

    use larql_compute::cpu::ops::q4_common::{quantize_q4_k, quantize_q6_k};
    use larql_compute_metal::shaders::q4k_q6k_qkv_proj as sh;

    // Gemma 3 4B-like geometry: hidden=2560, GQA num_q_heads=8 num_kv_heads=4.
    // (Real model has 8 / 4 with head_dim=256 → q_dim=2048, kv_dim=1024 — kept
    //  here at smaller q_rows / kv_rows so the test stays fast.)
    let q_rows = 1024usize;
    let kv_rows = 512usize;
    let hidden = 2560usize; // 10 × 256 super-blocks per row

    let wq_f32: Vec<f32> = (0..q_rows * hidden)
        .map(|i| ((i as f32 * 0.0007).cos()) * 0.5)
        .collect();
    let wk_f32: Vec<f32> = (0..kv_rows * hidden)
        .map(|i| ((i as f32 * 0.0011).sin()) * 0.5)
        .collect();
    let wv_f32: Vec<f32> = (0..kv_rows * hidden)
        .map(|i| ((i as f32 * 0.0017).cos()) * 0.4)
        .collect();
    let h_raw: Vec<f32> = (0..hidden)
        .map(|i| ((i as f32 * 0.013).sin() + 0.2) * 0.4)
        .collect();
    let norm_w: Vec<f32> = (0..hidden)
        .map(|i| 0.9 + (i as f32 * 0.001).sin() * 0.1)
        .collect();

    let wq_q4k = quantize_q4_k(&wq_f32);
    let wk_q4k = quantize_q4_k(&wk_f32);
    let wv_q6k = quantize_q6_k(&wv_f32);

    let eps = 1e-6f32;
    let offset = 1.0f32; // Gemma 3 norm_offset

    let sum_sq: f32 = h_raw.iter().map(|v| v * v).sum();
    let rms = 1.0 / (sum_sq / hidden as f32 + eps).sqrt();
    let h_normed: Vec<f32> = h_raw
        .iter()
        .zip(norm_w.iter())
        .map(|(h, w)| h * rms * (offset + w))
        .collect();

    let ref_q = metal
        .q4k_matvec(&wq_q4k, &h_normed, q_rows, hidden)
        .unwrap();
    let ref_k = metal
        .q4k_matvec(&wk_q4k, &h_normed, kv_rows, hidden)
        .unwrap();
    let ref_v = metal
        .q6k_matvec(&wv_q6k, &h_normed, kv_rows, hidden)
        .unwrap();

    let wq_buf = metal.bufs().get_bytes(&wq_q4k);
    let wk_buf = metal.bufs().get_bytes(&wk_q4k);
    let wv_buf = metal.bufs().get_bytes(&wv_q6k);
    let h_buf = metal.bufs().transient_from_f32(&h_raw);
    let nw_buf = metal.bufs().get_f32(&norm_w);
    let q_out = metal.bufs().output((q_rows * 4) as u64);
    let k_out = metal.bufs().output((kv_rows * 4) as u64);
    let v_out = metal.bufs().output((kv_rows * 4) as u64);

    let total_rows = (q_rows + kv_rows + kv_rows) as u64;
    let num_tgs = total_rows.div_ceil(sh::ROWS_PER_TG);
    let q_u = q_rows as u32;
    let kv_u = kv_rows as u32;
    let h_u = hidden as u32;

    let cmd = metal.queue().new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&metal.attention.q4k_q6k_qkv_proj_normed_pipeline.state);
    enc.set_buffer(0, Some(&wq_buf), 0);
    enc.set_buffer(1, Some(&wk_buf), 0);
    enc.set_buffer(2, Some(&wv_buf), 0);
    enc.set_buffer(3, Some(&h_buf), 0);
    enc.set_buffer(4, Some(&nw_buf), 0);
    enc.set_buffer(5, Some(&q_out), 0);
    enc.set_buffer(6, Some(&k_out), 0);
    enc.set_buffer(7, Some(&v_out), 0);
    enc.set_bytes(8, 4, &q_u as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(9, 4, &kv_u as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(10, 4, &kv_u as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(11, 4, &h_u as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(12, 4, &eps as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(13, 4, &offset as *const f32 as *const std::ffi::c_void);
    enc.dispatch_thread_groups(
        metal::MTLSize::new(num_tgs, 1, 1),
        metal::MTLSize::new(sh::THREADS_PER_TG, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let got_q = larql_compute_metal::buffers::read_buffer_f32(&q_out, q_rows);
    let got_k = larql_compute_metal::buffers::read_buffer_f32(&k_out, kv_rows);
    let got_v = larql_compute_metal::buffers::read_buffer_f32(&v_out, kv_rows);

    let threshold = 0.001;
    for (label, gref, got) in [
        ("Q", &ref_q, &got_q),
        ("K", &ref_k, &got_k),
        ("V", &ref_v, &got_v),
    ] {
        let max_abs = gref
            .iter()
            .map(|v: &f32| v.abs())
            .fold(0.0f32, f32::max)
            .max(1e-6);
        let d = max_diff(gref, got);
        assert!(
            d < max_abs * threshold,
            "q4k_q6k_qkv_proj_normed @hidden=2560 {label}: max_diff {d:.3e} exceeds {:.3e}",
            max_abs * threshold
        );
    }
}
