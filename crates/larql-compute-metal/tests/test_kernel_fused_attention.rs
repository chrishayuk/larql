//! Correctness tests for the `fused_attention` Metal shader.
//!
//! Verifies the fused prefill attention kernel (RoPE + causal masked
//! softmax + V-weighted sum) against a CPU reference implementation.
//! Covers standard geometry (3 tokens, 2 heads, head_dim=8) and the
//! wide-head regression case (head_dim=512) that exposed a tg_q
//! population bug in earlier versions.

#![cfg(target_os = "macos")]

extern crate blas_src;

#[path = "common/mod.rs"]
mod common;
use common::max_diff;

// ── fused_attention correctness (3 tokens, 2 heads, verified against CPU) ──

#[test]
fn fused_attention_matches_cpu_reference() {
    let Some(device) = metal::Device::system_default() else {
        return;
    };
    let src = larql_compute_metal::shaders::all_shaders();
    let lib = device
        .new_library_with_source(&src, &metal::CompileOptions::new())
        .unwrap();
    let pipeline = device
        .new_compute_pipeline_state_with_function(
            &lib.get_function("fused_attention", None).unwrap(),
        )
        .unwrap();
    let bufs = larql_compute_metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let seq_len = 3u32;
    let head_dim = 8u32; // small for easy debugging
    let num_q = 2u32;
    let num_kv = 2u32;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let rope_base = 10000.0f32;
    let use_qk_norm = 0u32;
    let softcap = 0.0f32;

    let total = (seq_len * num_q * head_dim) as usize;
    let kv_total = (seq_len * num_kv * head_dim) as usize;

    // Deterministic test data
    let q: Vec<f32> = (0..total)
        .map(|i| (i as f32 * 0.37 + 1.0).sin() * 0.5)
        .collect();
    let k: Vec<f32> = (0..kv_total)
        .map(|i| (i as f32 * 0.23 + 2.0).cos() * 0.5)
        .collect();
    let v: Vec<f32> = (0..kv_total)
        .map(|i| (i as f32 * 0.11 + 3.0).sin() * 0.3)
        .collect();

    // ── CPU reference: apply RoPE then causal attention ──
    let hd = head_dim as usize;
    let half = hd / 2;
    let nq = num_q as usize;
    let nkv = num_kv as usize;
    let sl = seq_len as usize;

    // Apply RoPE to Q and K
    let mut q_rope = q.clone();
    let mut k_rope = k.clone();
    for pos in 0..sl {
        for head in 0..nq {
            for d in 0..half {
                let freq = 1.0 / rope_base.powf(2.0 * d as f32 / hd as f32);
                let angle = pos as f32 * freq;
                let (cos_a, sin_a) = (angle.cos(), angle.sin());
                let idx_re = pos * nq * hd + head * hd + d;
                let idx_im = pos * nq * hd + head * hd + d + half;
                let re = q[idx_re];
                let im = q[idx_im];
                q_rope[idx_re] = re * cos_a - im * sin_a;
                q_rope[idx_im] = re * sin_a + im * cos_a;
            }
        }
        for head in 0..nkv {
            for d in 0..half {
                let freq = 1.0 / rope_base.powf(2.0 * d as f32 / hd as f32);
                let angle = pos as f32 * freq;
                let (cos_a, sin_a) = (angle.cos(), angle.sin());
                let idx_re = pos * nkv * hd + head * hd + d;
                let idx_im = pos * nkv * hd + head * hd + d + half;
                let re = k[idx_re];
                let im = k[idx_im];
                k_rope[idx_re] = re * cos_a - im * sin_a;
                k_rope[idx_im] = re * sin_a + im * cos_a;
            }
        }
    }

    // Causal attention per head per position
    let mut cpu_out = vec![0.0f32; total];
    for head in 0..nq {
        let kv_head = head / (nq / nkv);
        for qi in 0..sl {
            // Compute scores for all k <= qi
            let mut scores = Vec::new();
            for ki in 0..=qi {
                let mut dot = 0.0f32;
                for d in 0..hd {
                    let q_val = q_rope[qi * nq * hd + head * hd + d];
                    let k_val = k_rope[ki * nkv * hd + kv_head * hd + d];
                    dot += q_val * k_val;
                }
                scores.push(dot * scale);
            }
            // Softmax
            let max_s = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scores.iter().map(|s| (s - max_s).exp()).collect();
            let sum_exp: f32 = exps.iter().sum();
            let weights: Vec<f32> = exps.iter().map(|e| e / sum_exp).collect();
            // Weighted V
            for d in 0..hd {
                let mut acc = 0.0f32;
                for ki in 0..=qi {
                    acc += weights[ki] * v[ki * nkv * hd + kv_head * hd + d];
                }
                cpu_out[qi * nq * hd + head * hd + d] = acc;
            }
        }
    }

    // ── Metal ──
    let buf_q = bufs.transient_from_f32(&q);
    let buf_k = bufs.transient_from_f32(&k);
    let buf_v = bufs.transient_from_f32(&v);
    let buf_out = bufs.output((total * 4) as u64);

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&buf_q), 0);
    enc.set_buffer(1, Some(&buf_k), 0);
    enc.set_buffer(2, Some(&buf_v), 0);
    enc.set_buffer(3, Some(&buf_out), 0);
    enc.set_bytes(4, 4, &seq_len as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(5, 4, &head_dim as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(6, 4, &num_q as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(7, 4, &num_kv as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(8, 4, &scale as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(9, 4, &rope_base as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(10, 4, &use_qk_norm as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(11, 4, &softcap as *const f32 as *const std::ffi::c_void);
    let skip_rope_val = 0u32;
    enc.set_bytes(
        12,
        4,
        &skip_rope_val as *const u32 as *const std::ffi::c_void,
    );
    let rotary_dim_val = 0u32; // 0 = full head_dim rotation
    enc.set_bytes(
        13,
        4,
        &rotary_dim_val as *const u32 as *const std::ffi::c_void,
    );
    enc.dispatch_thread_groups(
        metal::MTLSize::new(num_q as u64, seq_len as u64, 1),
        metal::MTLSize::new(256, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let ptr = buf_out.contents() as *const f32;
    let metal_result: Vec<f32> = unsafe { std::slice::from_raw_parts(ptr, total).to_vec() };

    // Compare
    let diff = max_diff(&cpu_out, &metal_result);
    assert!(
        diff < 0.01,
        "fused_attention max diff {diff} (expected < 0.01).\nCPU[0..8]: {:?}\nGPU[0..8]: {:?}",
        &cpu_out[..8.min(total)],
        &metal_result[..8.min(total)]
    );
}

// ── fused_attention at head_dim=512 (Gemma 4 global layers) ──

/// Regression guard for the Metal `fused_attention` shader on wide heads.
///
/// Gemma 4 global attention layers have `head_dim=512`. The fused shader
/// dispatches 256 threads per (head, pos). The earlier implementation
/// loaded `tg_q` under `if (tid < head_dim)`, which silently left
/// `tg_q[256..512]` uninitialised — the subsequent Q·K dot product read
/// garbage for the tail half of every head, producing attention output
/// with ≈6% magnitude loss (cos≈0.965 vs CPU reference). This ruined the
/// per-layer residual from L5 onward on Gemma 4 31B Q4K end-to-end.
///
/// Fix: strided `for (uint d = tid; d < head_dim; d += tg_sz)` for both
/// the tg_q population and the internal QK-norm scale.
///
/// Test strategy: pick head_dim well above 256 (512), skip RoPE (the
/// shader supports `skip_rope=1`) so the CPU reference is a plain
/// causal-masked softmax(QK·scale)·V. If the tg_q tail is ever zeroed
/// again, `attn_out` norm will drop and cos will dip — this test
/// catches it within seconds, no Gemma 4 vindex required.
#[test]
fn fused_attention_head_dim_512() {
    let Some(device) = metal::Device::system_default() else {
        return;
    };
    let src = larql_compute_metal::shaders::all_shaders();
    let lib = device
        .new_library_with_source(&src, &metal::CompileOptions::new())
        .unwrap();
    let pipeline = device
        .new_compute_pipeline_state_with_function(
            &lib.get_function("fused_attention", None).unwrap(),
        )
        .unwrap();
    let bufs = larql_compute_metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    // Gemma 4 31B global layer geometry:
    //   head_dim = 512, num_q = 32, num_kv = 4, seq_len = 4 (short to
    //   keep the hand-computed reference cheap). Using `skip_rope=1` so
    //   the input Q/K are taken as-is (no rotation), isolating the bug
    //   to the tg_q population + Q·K dot + softmax + V-weighted sum.
    let seq_len = 4u32;
    let head_dim = 512u32;
    let num_q = 4u32; // trim vs 32 — still exercises GQA reps and stays fast
    let num_kv = 2u32;
    let scale = 1.0f32; // Gemma 4 uses QK-norm so default scale is 1.0 — matches prod path
    let rope_base = 10000.0f32;
    let use_qk_norm = 0u32;
    let softcap = 0.0f32;
    let skip_rope = 1u32;
    let rotary_dim = 0u32;

    let q_total = (seq_len * num_q * head_dim) as usize;
    let kv_total = (seq_len * num_kv * head_dim) as usize;

    // Non-trivial, position/head-dependent data. Make the tail dims
    // (>= 256) non-zero and non-constant so any bug that zeroes or
    // misreads them produces a detectable difference from the CPU
    // reference — constant tails would mask the bug.
    let q: Vec<f32> = (0..q_total)
        .map(|i| ((i as f32 * 0.017).sin() + 0.5 * ((i >> 7) as f32).cos()) * 0.3)
        .collect();
    let k: Vec<f32> = (0..kv_total)
        .map(|i| ((i as f32 * 0.013).cos() - 0.3 * ((i >> 6) as f32).sin()) * 0.4)
        .collect();
    let v: Vec<f32> = (0..kv_total)
        .map(|i| ((i as f32 * 0.019).sin() + 0.2 * ((i >> 8) as f32).sin()) * 0.25)
        .collect();

    // ── CPU reference: causal GQA softmax with NO RoPE (skip_rope=1). ──
    let hd = head_dim as usize;
    let nq = num_q as usize;
    let nkv = num_kv as usize;
    let sl = seq_len as usize;
    let reps = nq / nkv;

    let mut cpu_out = vec![0.0f32; q_total];
    for head in 0..nq {
        let kv_head = head / reps;
        for qi in 0..sl {
            let mut scores = Vec::with_capacity(qi + 1);
            for ki in 0..=qi {
                let mut dot = 0.0f32;
                for d in 0..hd {
                    let q_val = q[qi * nq * hd + head * hd + d];
                    let k_val = k[ki * nkv * hd + kv_head * hd + d];
                    dot += q_val * k_val;
                }
                scores.push(dot * scale);
            }
            let max_s = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scores.iter().map(|s| (s - max_s).exp()).collect();
            let sum_exp: f32 = exps.iter().sum();
            let weights: Vec<f32> = exps.iter().map(|e| e / sum_exp).collect();
            for d in 0..hd {
                let mut acc = 0.0f32;
                for ki in 0..=qi {
                    acc += weights[ki] * v[ki * nkv * hd + kv_head * hd + d];
                }
                cpu_out[qi * nq * hd + head * hd + d] = acc;
            }
        }
    }

    // ── Metal dispatch. Same launch shape as production
    //   (crates/larql-compute/src/metal/stages/attention.rs) — 256-wide
    //   threadgroup × (num_q, seq_len) grid.
    let buf_q = bufs.transient_from_f32(&q);
    let buf_k = bufs.transient_from_f32(&k);
    let buf_v = bufs.transient_from_f32(&v);
    let buf_out = bufs.output((q_total * 4) as u64);

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&buf_q), 0);
    enc.set_buffer(1, Some(&buf_k), 0);
    enc.set_buffer(2, Some(&buf_v), 0);
    enc.set_buffer(3, Some(&buf_out), 0);
    enc.set_bytes(4, 4, &seq_len as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(5, 4, &head_dim as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(6, 4, &num_q as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(7, 4, &num_kv as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(8, 4, &scale as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(9, 4, &rope_base as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(10, 4, &use_qk_norm as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(11, 4, &softcap as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(12, 4, &skip_rope as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(13, 4, &rotary_dim as *const u32 as *const std::ffi::c_void);
    enc.dispatch_thread_groups(
        metal::MTLSize::new(num_q as u64, seq_len as u64, 1),
        metal::MTLSize::new(256, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let ptr = buf_out.contents() as *const f32;
    let metal_result: Vec<f32> = unsafe { std::slice::from_raw_parts(ptr, q_total).to_vec() };

    // Tight tolerance: this is a direct f32 softmax — no quantisation,
    // no RoPE. Any kernel-level miscompute will produce diffs well above
    // 1e-4. The regressed tg_q bug produced max diff around 5e-2 at this
    // geometry; keeping the bar at 1e-3 gives a ~50× safety margin while
    // still flagging genuine shader breakage.
    let diff = max_diff(&cpu_out, &metal_result);
    assert!(
        diff < 1e-3,
        "fused_attention@head_dim=512 max diff {diff} exceeds 1e-3.\n\
         This usually means the tg_q load (or internal QK-norm scale)\n\
         gated on `tid < head_dim` and left positions 256..512 unset —\n\
         see `crates/larql-compute/src/metal/shaders/fused_attention.rs`.\n\
         CPU[0..8]: {:?}\nGPU[0..8]: {:?}",
        &cpu_out[..8],
        &metal_result[..8],
    );

    // Also pin cosine similarity at the aggregate level — a scalar
    // regression metric that surfaces in per-layer residual drift.
    let mut dot = 0.0f64;
    let mut cn = 0.0f64;
    let mut mn = 0.0f64;
    for i in 0..q_total {
        let a = cpu_out[i] as f64;
        let b = metal_result[i] as f64;
        dot += a * b;
        cn += a * a;
        mn += b * b;
    }
    let cos = dot / (cn.sqrt() * mn.sqrt());
    assert!(
        cos > 0.999999,
        "fused_attention@head_dim=512 cos_sim {cos:.6} below 0.999999 — \
         subtle kernel drift that compounds across layers",
    );
}
