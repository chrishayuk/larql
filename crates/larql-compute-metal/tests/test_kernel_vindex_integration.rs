//! End-to-end regression tests that require a real vindex on disk, plus
//! stage-level composition tests for `stages::residual` and
//! `stages::quant_matvec` encode helpers.
//!
//! The vindex test (`q4kf_proj_matches_cpu_on_real_vindex_bytes`) is
//! gated on the vindex file existing at
//! `../../output/gemma3-4b-q4k-v2.vindex` — it skips cleanly otherwise.
//!
//! Stage tests drive the `encode_post_attn`, `encode_post_ffn`, and
//! `quant_matvec::encode` helpers and compare against CPU references,
//! pinning down composition bugs that individual shader tests miss.

#![cfg(target_os = "macos")]

extern crate blas_src;

use larql_compute::prelude::*;
use ndarray::Array2;

#[path = "common/mod.rs"]
mod common;
use common::{get_metal, max_diff};

fn synth(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
    let mut s = seed;
    Array2::from_shape_fn((rows, cols), |_| {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    })
}

// ── q4kf_proj on REAL vindex Q4_K bytes (end-to-end regression) ──
//
// Background: `q4kf_proj_matches_cpu_reference*` pass (ratio 1.000) with
// weights produced by our `quantize_q4_k`. But on REAL Ollama-GGUF Q4_K
// bytes from a Gemma 3 4B vindex, Metal `q4kf_proj` and CPU
// `dequantize_q4_k + gemv` diverge by ~22% in magnitude (ratio ~0.78).
//
// Root cause (verified 2026-04-18): our `quantize_q4_k` emits a slightly
// different 12-byte scale+min packing than what llama.cpp writes. The
// Metal shader's scale-unpack matches our quantizer; `dequantize_q4_k`
// matches llama.cpp. Since production vindexes contain llama.cpp-layout
// bytes (extracted from Ollama GGUFs), the Metal shader reads them with
// the wrong scale nibbles and returns values ~22% off.
//
// Fix path: either update `quantize_q4_k` to emit llama.cpp-exact
// packing (so shader + data agree again), or update the shader's scale
// unpack to match `dequantize_q4_k`. The shader path (q4kf_qkv_proj.rs)
// is the canonical llama.cpp pattern — easier to leave it alone and fix
// the quantizer.
//
// Test is gated on the vindex file being present; skipped otherwise.
// Failing here is the intended regression gate.
#[test]
fn q4kf_proj_matches_cpu_on_real_vindex_bytes() {
    let vindex = std::path::Path::new("../../output/gemma3-4b-q4k-v2.vindex");
    if !vindex.exists() {
        eprintln!("skip: real vindex {} not present", vindex.display());
        return;
    }
    let manifest_path = vindex.join("attn_weights_q4k_manifest.json");
    let bin_path = vindex.join("attn_weights_q4k.bin");
    let manifest_txt = match std::fs::read_to_string(&manifest_path) {
        Ok(t) => t,
        Err(_) => {
            eprintln!("skip: manifest unreadable");
            return;
        }
    };
    let entries: Vec<serde_json::Value> = serde_json::from_str(&manifest_txt).unwrap();
    let q_entry = entries
        .iter()
        .find(|e| {
            e["key"]
                .as_str()
                .unwrap_or("")
                .contains("layers.0.self_attn.q_proj")
        })
        .expect("layer 0 Q entry in manifest");
    let offset = q_entry["offset"].as_u64().unwrap() as usize;
    let length = q_entry["length"].as_u64().unwrap() as usize;
    let shape: Vec<usize> = q_entry["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let (rows, hidden) = (shape[0], shape[1]);
    let bin = std::fs::read(&bin_path).expect("attn_weights_q4k.bin");
    let q_bytes = &bin[offset..offset + length];

    // CPU reference: dequantize the real bytes, then gemv against a fixed x.
    let dequant = larql_models::quant::ggml::dequantize_q4_k(q_bytes, rows * hidden).unwrap();
    let x: Vec<f32> = (0..hidden).map(|i| ((i as f32) * 0.01).sin()).collect();
    let mut cpu_out = vec![0.0f32; rows];
    for row in 0..rows {
        cpu_out[row] = (0..hidden).map(|k| dequant[row * hidden + k] * x[k]).sum();
    }

    // Metal: dispatch q4kf_proj directly on the real bytes.
    let metal = get_metal();
    use larql_compute_metal::shaders::q4kf_qkv_proj as q4kf;
    let w_buf = metal.bufs().get_bytes(q_bytes);
    let x_buf = metal.bufs().transient_from_f32(&x);
    let out_buf = metal.bufs().output((rows * 4) as u64);

    let cmd = metal.queue().new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&metal.attention.q4kf_proj_pipeline.state);
    enc.set_buffer(0, Some(&w_buf), 0);
    enc.set_buffer(1, Some(&x_buf), 0);
    enc.set_buffer(2, Some(&out_buf), 0);
    let n = rows as u32;
    let k = hidden as u32;
    enc.set_bytes(3, 4, &n as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(4, 4, &k as *const u32 as *const std::ffi::c_void);
    let num_tgs = (rows as u64).div_ceil(q4kf::ROWS_PER_TG);
    enc.dispatch_thread_groups(
        metal::MTLSize::new(num_tgs, 1, 1),
        metal::MTLSize::new(q4kf::THREADS_PER_TG, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let metal_out = larql_compute_metal::buffers::read_buffer_f32(&out_buf, rows);
    let cpu_max = cpu_out.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let met_max = metal_out.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let ratio = cpu_max / met_max.max(1e-9);
    let max_diff_val = cpu_out
        .iter()
        .zip(&metal_out)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!(
        "real-bytes q4kf_proj[{rows}x{hidden}]  cpu_max={cpu_max:.3e}  \
         metal_max={met_max:.3e}  ratio_cpu/metal={ratio:.3}  max_abs_diff={max_diff_val:.3e}"
    );
    assert!(
        (ratio - 1.0).abs() < 0.05,
        "q4kf_proj on REAL vindex data scales differently from CPU dequant+gemv: \
         ratio={ratio:.3} (expected ~1.0). This is the end-to-end regression."
    );
}

// ═══════════════════════════════════════════════════════════════
// Stage-level composition tests.
//
// Each test drives a `stages::*::encode*` helper and compares the
// composed output against a CPU reference computed in the test.
// These pin down composition bugs that individual shader tests miss:
//   - wrong format dispatch inside `quant_matvec::encode`,
//   - off-by-one buffer offsets in `encode_post_attn`,
//   - pre-norm vs post-norm branching in `encode_post_ffn`,
//   - Q8 quant emission when FFN input needs Q8.
// ═══════════════════════════════════════════════════════════════

fn build_pipeline(device: &metal::Device, name: &str) -> metal::ComputePipelineState {
    let src = larql_compute_metal::shaders::all_shaders();
    let lib = device
        .new_library_with_source(&src, &metal::CompileOptions::new())
        .unwrap();
    device
        .new_compute_pipeline_state_with_function(&lib.get_function(name, None).unwrap())
        .unwrap()
}

fn read_f32_buf(buf: &metal::Buffer, n: usize) -> Vec<f32> {
    let ptr = buf.contents() as *const f32;
    unsafe { std::slice::from_raw_parts(ptr, n).to_vec() }
}

/// CPU reference: RMS-norm with llama-style offset on the weight.
fn cpu_rms_norm(x: &[f32], w: &[f32], eps: f32, offset: f32) -> Vec<f32> {
    let n = x.len() as f32;
    let ms: f32 = x.iter().map(|v| v * v).sum::<f32>() / n;
    let inv = 1.0f32 / (ms + eps).sqrt();
    x.iter()
        .zip(w)
        .map(|(v, wv)| v * inv * (offset + wv))
        .collect()
}

/// Stage: `residual::encode_post_attn` in pre-norm mode, no Q8 FFN input.
///
/// Verifies the two-dispatch fusion (residual_add then rms_norm) matches a
/// straight CPU composition. Pre-norm is the Gemma 3 / Llama path.
#[test]
fn stage_post_attn_pre_norm_matches_cpu() {
    let device = metal::Device::system_default().unwrap();
    let rms_norm = build_pipeline(&device, "rms_norm");
    let residual_add = build_pipeline(&device, "residual_add");
    let q8_quant = build_pipeline(&device, "quantize_q8");
    let bufs = larql_compute_metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let hidden = 256usize;
    let seq_len = 3usize;
    let eps = 1e-6f32;
    let offset = 0.0f32;

    let h: Vec<f32> = (0..seq_len * hidden)
        .map(|i| ((i as f32) * 0.013).sin())
        .collect();
    let o: Vec<f32> = (0..seq_len * hidden)
        .map(|i| ((i as f32) * 0.017).cos())
        .collect();
    let w_post_attn: Vec<f32> = (0..hidden).map(|i| 1.0 + 0.01 * (i as f32).sin()).collect();

    // Expected: per-position, h + o → rms_norm(., w_post_attn).
    let mut expected_hpa = vec![0.0f32; seq_len * hidden];
    let mut expected_ffn = vec![0.0f32; seq_len * hidden];
    for p in 0..seq_len {
        let off = p * hidden;
        for i in 0..hidden {
            expected_hpa[off + i] = h[off + i] + o[off + i];
        }
        expected_ffn[off..off + hidden].copy_from_slice(&cpu_rms_norm(
            &expected_hpa[off..off + hidden],
            &w_post_attn,
            eps,
            offset,
        ));
    }

    let h_buf = bufs.transient_from_f32(&h);
    let o_buf = bufs.transient_from_f32(&o);
    let w_buf = bufs.transient_from_f32(&w_post_attn);
    let h_pa = bufs.output((seq_len * hidden * 4) as u64);
    let ffn_out = bufs.output((seq_len * hidden * 4) as u64);
    // Q8 bufs unused on this path, but the helper still takes them.
    let q8 = bufs.output((seq_len * hidden) as u64);
    let q8s = bufs.output((seq_len * hidden.div_ceil(32) * 4) as u64);

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    let mut scratch = |n: u64| bufs.output(n);
    larql_compute_metal::stages::residual::encode_post_attn(
        enc,
        &rms_norm,
        &residual_add,
        &q8_quant,
        &mut scratch,
        &h_buf,
        &o_buf,
        &h_pa,
        &ffn_out,
        &w_buf,
        &w_buf, // post_attn_norm_buf, pre_ffn_weight_buf (same in pre-norm)
        &q8,
        &q8s,
        seq_len,
        hidden,
        eps,
        offset,
        /*has_post_norms*/ false,
        /*ffn_needs_q8*/ false,
        (hidden * 4) as u64,
        hidden as u64,
        (hidden.div_ceil(32) * 4) as u64,
        1.0,
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let metal_hpa = read_f32_buf(&h_pa, seq_len * hidden);
    let metal_ffn = read_f32_buf(&ffn_out, seq_len * hidden);
    let dh = max_diff(&expected_hpa, &metal_hpa);
    let df = max_diff(&expected_ffn, &metal_ffn);
    assert!(dh < 1e-5, "post_attn h_pa diff {dh}");
    assert!(df < 1e-4, "post_attn ffn_norm diff {df}");
}

/// Stage: `residual::encode_post_attn` in post-norm mode.
///
/// Post-norm path (Gemma 2 / some Gemma 3 configs) is:
///   h_post_attn = h + norm(O, post_attn_norm),
///   ffn_norm_out = norm(h_post_attn, pre_ffn_norm).
/// Distinct weight per norm; this exercises the `has_post_norms` branch.
#[test]
fn stage_post_attn_post_norm_matches_cpu() {
    let device = metal::Device::system_default().unwrap();
    let rms_norm = build_pipeline(&device, "rms_norm");
    let residual_add = build_pipeline(&device, "residual_add");
    let q8_quant = build_pipeline(&device, "quantize_q8");
    let bufs = larql_compute_metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let hidden = 128usize;
    let seq_len = 2usize;
    let eps = 1e-6f32;
    let offset = 1.0f32; // Gemma-style offset

    let h: Vec<f32> = (0..seq_len * hidden)
        .map(|i| ((i as f32) * 0.019).sin())
        .collect();
    let o: Vec<f32> = (0..seq_len * hidden)
        .map(|i| ((i as f32) * 0.023).cos())
        .collect();
    let w_post_attn: Vec<f32> = (0..hidden).map(|i| 0.05 * (i as f32).cos()).collect();
    let w_pre_ffn: Vec<f32> = (0..hidden)
        .map(|i| 0.08 * ((i as f32) * 0.3).sin())
        .collect();

    let mut expected_hpa = vec![0.0f32; seq_len * hidden];
    let mut expected_ffn = vec![0.0f32; seq_len * hidden];
    for p in 0..seq_len {
        let off = p * hidden;
        let normed = cpu_rms_norm(&o[off..off + hidden], &w_post_attn, eps, offset);
        for i in 0..hidden {
            expected_hpa[off + i] = h[off + i] + normed[i];
        }
        expected_ffn[off..off + hidden].copy_from_slice(&cpu_rms_norm(
            &expected_hpa[off..off + hidden],
            &w_pre_ffn,
            eps,
            offset,
        ));
    }

    let h_buf = bufs.transient_from_f32(&h);
    let o_buf = bufs.transient_from_f32(&o);
    let w_pa_buf = bufs.transient_from_f32(&w_post_attn);
    let w_pf_buf = bufs.transient_from_f32(&w_pre_ffn);
    let h_pa = bufs.output((seq_len * hidden * 4) as u64);
    let ffn_out = bufs.output((seq_len * hidden * 4) as u64);
    let q8 = bufs.output((seq_len * hidden) as u64);
    let q8s = bufs.output((seq_len * hidden.div_ceil(32) * 4) as u64);

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    let mut scratch = |n: u64| bufs.output(n);
    larql_compute_metal::stages::residual::encode_post_attn(
        enc,
        &rms_norm,
        &residual_add,
        &q8_quant,
        &mut scratch,
        &h_buf,
        &o_buf,
        &h_pa,
        &ffn_out,
        &w_pa_buf,
        &w_pf_buf,
        &q8,
        &q8s,
        seq_len,
        hidden,
        eps,
        offset,
        /*has_post_norms*/ true,
        /*ffn_needs_q8*/ false,
        (hidden * 4) as u64,
        hidden as u64,
        (hidden.div_ceil(32) * 4) as u64,
        1.0,
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let metal_hpa = read_f32_buf(&h_pa, seq_len * hidden);
    let metal_ffn = read_f32_buf(&ffn_out, seq_len * hidden);
    assert!(
        max_diff(&expected_hpa, &metal_hpa) < 1e-4,
        "post_norm h_pa diff"
    );
    assert!(
        max_diff(&expected_ffn, &metal_ffn) < 1e-4,
        "post_norm ffn_norm diff"
    );
}

/// Stage: `residual::encode_post_ffn` plain (pre-norm) residual.
#[test]
fn stage_post_ffn_pre_norm_matches_cpu() {
    let device = metal::Device::system_default().unwrap();
    let rms_norm = build_pipeline(&device, "rms_norm");
    let residual_add = build_pipeline(&device, "residual_add");
    let bufs = larql_compute_metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let hidden = 192usize;
    let seq_len = 3usize;

    let hpa: Vec<f32> = (0..seq_len * hidden)
        .map(|i| ((i as f32) * 0.015).sin())
        .collect();
    let dn: Vec<f32> = (0..seq_len * hidden)
        .map(|i| ((i as f32) * 0.011).cos())
        .collect();

    let expected: Vec<f32> = hpa.iter().zip(&dn).map(|(a, b)| a + b).collect();

    let hpa_buf = bufs.transient_from_f32(&hpa);
    let dn_buf = bufs.transient_from_f32(&dn);
    let out = bufs.output((seq_len * hidden * 4) as u64);

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    let mut scratch = |n: u64| bufs.output(n);
    larql_compute_metal::stages::residual::encode_post_ffn(
        enc,
        &rms_norm,
        &residual_add,
        &mut scratch,
        &dn_buf,
        &hpa_buf,
        &out,
        None,
        seq_len,
        hidden,
        1e-6,
        0.0,
        /*has_post_norms*/ false,
        (hidden * 4) as u64,
        1.0,
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let got = read_f32_buf(&out, seq_len * hidden);
    assert!(max_diff(&expected, &got) < 1e-5, "post_ffn pre-norm diff");
}

/// Stage: `residual::encode_post_ffn` post-norm with a `post_ffn_norm` weight.
#[test]
fn stage_post_ffn_post_norm_matches_cpu() {
    let device = metal::Device::system_default().unwrap();
    let rms_norm = build_pipeline(&device, "rms_norm");
    let residual_add = build_pipeline(&device, "residual_add");
    let bufs = larql_compute_metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let hidden = 128usize;
    let seq_len = 2usize;
    let eps = 1e-6f32;
    let offset = 1.0f32;

    let hpa: Vec<f32> = (0..seq_len * hidden)
        .map(|i| ((i as f32) * 0.021).sin())
        .collect();
    let dn: Vec<f32> = (0..seq_len * hidden)
        .map(|i| ((i as f32) * 0.007).cos())
        .collect();
    let w_post_ffn: Vec<f32> = (0..hidden)
        .map(|i| 0.1 * ((i as f32) * 0.25).sin())
        .collect();

    let mut expected = vec![0.0f32; seq_len * hidden];
    for p in 0..seq_len {
        let off = p * hidden;
        let normed = cpu_rms_norm(&dn[off..off + hidden], &w_post_ffn, eps, offset);
        for i in 0..hidden {
            expected[off + i] = hpa[off + i] + normed[i];
        }
    }

    let hpa_buf = bufs.transient_from_f32(&hpa);
    let dn_buf = bufs.transient_from_f32(&dn);
    let w_buf = bufs.transient_from_f32(&w_post_ffn);
    let out = bufs.output((seq_len * hidden * 4) as u64);

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    let mut scratch = |n: u64| bufs.output(n);
    larql_compute_metal::stages::residual::encode_post_ffn(
        enc,
        &rms_norm,
        &residual_add,
        &mut scratch,
        &dn_buf,
        &hpa_buf,
        &out,
        Some(&w_buf),
        seq_len,
        hidden,
        eps,
        offset,
        /*has_post_norms*/ true,
        (hidden * 4) as u64,
        1.0,
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let got = read_f32_buf(&out, seq_len * hidden);
    assert!(max_diff(&expected, &got) < 1e-4, "post_ffn post-norm diff");
}

/// Stage: `quant_matvec::encode` routes each format to the correct shader.
///
/// Feeds Q4_K, Q6_K, and Q4_0 weights through the same `encode` call and
/// checks each output matches a direct single-format shader dispatch. This
/// is what pins down the `match format` arm selection in the helper.
#[test]
fn stage_quant_matvec_routes_format_to_correct_shader() {
    use larql_compute_metal::kernels::KernelHandle;
    use larql_compute_metal::shaders::{q4_matvec_v4, q4k_matvec, q6k_matvec};

    let device = metal::Device::system_default().unwrap();
    let src = larql_compute_metal::shaders::all_shaders();
    let library = device
        .new_library_with_source(&src, &metal::CompileOptions::new())
        .unwrap();

    let q4kf_proj = build_pipeline(&device, "q4kf_proj");
    let q4k_mv = KernelHandle::from_kernel::<q4k_matvec::Kernel>(&device, &library).unwrap();
    let q6k_mv = KernelHandle::from_kernel::<q6k_matvec::Kernel>(&device, &library).unwrap();
    let q4_matvec = KernelHandle::from_kernel::<q4_matvec_v4::Kernel>(&device, &library).unwrap();
    let bufs = larql_compute_metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    // Q4_K / Q6_K require hidden to be a multiple of 256 (superblock size).
    let rows = 32usize;
    let hidden = 256usize;

    let pipes = larql_compute_metal::stages::quant_matvec::Pipelines {
        q4kf_proj: Some(&q4kf_proj),
        q4k_matvec_fallback: &q4k_mv,
        q6k_matvec: &q6k_mv,
        q4_matvec: &q4_matvec,
        q4k_matmul: None,
    };

    let w_f32: Vec<f32> = (0..rows * hidden)
        .map(|i| ((i as f32) * 0.009).sin())
        .collect();
    let x: Vec<f32> = (0..hidden).map(|i| ((i as f32) * 0.017).cos()).collect();

    // Expected reference: f32 gemv, matches the dequantise-then-dot semantics
    // every quant shader approximates.
    let expected: Vec<f32> = (0..rows)
        .map(|r| (0..hidden).map(|c| w_f32[r * hidden + c] * x[c]).sum())
        .collect();

    let x_buf = bufs.transient_from_f32(&x);
    let out = bufs.output((rows * 4) as u64);

    // Q4_K route.
    let w_q4k = larql_compute::cpu::ops::q4_common::quantize_q4_k(&w_f32);
    let w_q4k_buf = bufs.get_bytes(&w_q4k);
    {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        larql_compute_metal::stages::quant_matvec::encode(
            enc,
            larql_compute::QuantFormat::Q4_K,
            &w_q4k_buf,
            &x_buf,
            0,
            &x_buf,
            0,
            &x_buf,
            0,
            &out,
            0,
            &pipes,
            rows,
            hidden,
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }
    let got_q4k = read_f32_buf(&out, rows);
    let max_abs = expected
        .iter()
        .map(|v| v.abs())
        .fold(0.0f32, f32::max)
        .max(1e-6);
    let rel = max_diff(&expected, &got_q4k) / max_abs;
    assert!(rel < 0.05, "Q4_K route rel err {rel:.4}");

    // Q6_K route (emitted via CPU quantizer).
    let w_q6k = larql_compute::cpu::ops::q4_common::quantize_q6_k(&w_f32);
    let w_q6k_buf = bufs.get_bytes(&w_q6k);
    {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        larql_compute_metal::stages::quant_matvec::encode(
            enc,
            larql_compute::QuantFormat::Q6_K,
            &w_q6k_buf,
            &x_buf,
            0,
            &x_buf,
            0,
            &x_buf,
            0,
            &out,
            0,
            &pipes,
            rows,
            hidden,
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }
    let got_q6k = read_f32_buf(&out, rows);
    let rel = max_diff(&expected, &got_q6k) / max_abs;
    assert!(rel < 0.02, "Q6_K route rel err {rel:.4}");

    // Q4_0 route needs Q8 input.
    let w_q4_0 = larql_compute::cpu::q4::quantize_q4_0(&w_f32);
    let w_q4_0_buf = bufs.get_bytes(&w_q4_0);
    let (q8_x, q8_x_scales) = larql_compute::cpu::q4::quantize_to_q8(&x);
    let q8_x_buf = bufs.transient_from_i8(&q8_x);
    let q8_x_s_buf = bufs.transient_from_f32(&q8_x_scales);
    {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        larql_compute_metal::stages::quant_matvec::encode(
            enc,
            larql_compute::QuantFormat::Q4_0,
            &w_q4_0_buf,
            &x_buf,
            0,
            &q8_x_buf,
            0,
            &q8_x_s_buf,
            0,
            &out,
            0,
            &pipes,
            rows,
            hidden,
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }
    let got_q4_0 = read_f32_buf(&out, rows);
    let rel = max_diff(&expected, &got_q4_0) / max_abs;
    assert!(rel < 0.1, "Q4_0 route rel err {rel:.4}");
}

/// `f32_gemv` shader: `out[N] = W[N,K] · x[K]` matches `ndarray::dot`.
///
/// Motivating case: LM-head logits at autoregressive decode. The shader's
/// value-add over re-using `sgemm_transb` at M=1 is both speed (row-per-
/// simdgroup vs 31/32-wasted-thread tiled gemm) and argmax stability
/// (deterministic per-row reduction order, no shifting of top-K under
/// noisy logits). Test pins both.
#[test]
fn f32_gemv_matches_ndarray_dot() {
    let metal = get_metal();
    // Small shapes fall below the default 500 MFLOP threshold and return
    // None (caller falls back to CPU). We want to exercise the Metal
    // path, so drop the floor.
    metal.set_flop_threshold(1);

    // Dimensions chosen to match the Gemma 3/4 LM-head aspect ratio in
    // miniature: wide N, K a non-power-of-two-multiple-of-32, K % 128 != 0.
    let n = 2048usize;
    let k = 2560usize;
    let w = synth(n, k, 0xa11ce);
    let x: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.013).sin()).collect();

    // CPU reference: ndarray's BLAS gemv.
    let x_arr = ndarray::Array1::from(x.clone());
    let expected = w.dot(&x_arr);

    // Metal path.
    let got = metal
        .f32_gemv(w.view(), &x)
        .expect("gemv should dispatch above threshold");
    assert_eq!(got.len(), n);

    let diff = max_diff(expected.as_slice().unwrap(), &got);
    let max_abs = expected
        .iter()
        .map(|v| v.abs())
        .fold(0.0f32, f32::max)
        .max(1e-6);
    let rel = diff / max_abs;
    assert!(
        rel < 1e-4,
        "f32_gemv rel err {rel:.2e} (abs {diff:.2e}, max_abs {max_abs:.2e})"
    );

    // Argmax stability — the actual property that matters for LM-head top-K.
    let exp_argmax = expected
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    let got_argmax = got
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    assert_eq!(
        exp_argmax, got_argmax,
        "argmax mismatch between CPU and Metal gemv"
    );
}

/// `f16_gemv` shader: f16 weights × f32 query, matches `f32_gemv` within
/// half-precision noise.
///
/// Motivating case: Gemma 4 31B tied-embedding LM head. The current path
/// decodes the 2.8 GB f16 safetensors into a 5.6 GB f32 clone at load;
/// this shader lets the Metal backend consume the f16 bytes directly.
/// Test pins argmax equality with the f32 reference — that's the actual
/// property that matters for top-K.
#[test]
fn f16_gemv_matches_f32_gemv_argmax() {
    use larql_models::quant::half::encode_f16;

    let metal = get_metal();
    metal.set_flop_threshold(1);

    let n = 2048usize;
    let k = 2560usize;
    let w = synth(n, k, 0xf16ce);
    let x: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.013).sin()).collect();

    // f32 reference.
    let x_arr = ndarray::Array1::from(x.clone());
    let expected = w.dot(&x_arr);

    // Encode weights as f16 bytes (IEEE half, little-endian).
    let w_flat: Vec<f32> = w.iter().copied().collect();
    let w_f16 = encode_f16(&w_flat);
    assert_eq!(w_f16.len(), n * k * 2);

    let got = metal
        .f16_gemv(&w_f16, &x, n, k)
        .expect("f16_gemv should dispatch above threshold");
    assert_eq!(got.len(), n);

    // f16 weights introduce relative error ~1e-3 on the output; don't pin
    // values, pin argmax — that's the property the LM head top-K depends on.
    let exp_argmax = expected
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    let got_argmax = got
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    assert_eq!(
        exp_argmax, got_argmax,
        "f16_gemv argmax mismatch vs f32 reference"
    );

    // Sanity: the scores around the argmax should be within f16 relative
    // noise of the f32 reference.
    let tol = expected
        .iter()
        .map(|v| v.abs())
        .fold(0.0f32, f32::max)
        .max(1.0)
        * 5e-3;
    let diff = (expected[exp_argmax] - got[exp_argmax]).abs();
    assert!(
        diff < tol,
        "argmax-value drift {diff:.4} exceeds f16 tolerance {tol:.4}"
    );
}

/// Uniform `q4k_qkv_proj` fused shader matches three `q4k_matvec` dispatches.
///
/// Regression gate for the 148-vs-144 Q4_K super-block stride bug: the
/// first draft of this shader typed weights as `block_q4_K*` (148-byte
/// MSL struct with an obsolete `mins[4]` field), which silently mis-read
/// production GGUF data. Row stride was off by 40 bytes per row,
/// accumulating into buffer-overruns past the first superblock. The
/// output was "approximately correct" enough for argmax to stabilise on
/// trivial prompts, hiding the bug. Now the shader uses manual byte
/// offsets with the correct 144-byte stride.
#[test]
fn q4k_qkv_proj_matches_per_proj_dispatch() {
    let metal = get_metal();
    let q_rows = 2048usize;
    let kv_rows = 1024usize;
    let hidden = 2560usize;

    let wq_f32 = synth(q_rows, hidden, 0xbeef_0001)
        .as_standard_layout()
        .to_owned();
    let wk_f32 = synth(kv_rows, hidden, 0xbeef_0002)
        .as_standard_layout()
        .to_owned();
    let wv_f32 = synth(kv_rows, hidden, 0xbeef_0003)
        .as_standard_layout()
        .to_owned();
    let x: Vec<f32> = (0..hidden).map(|i| ((i as f32) * 0.017).cos()).collect();

    let wq_q4k = larql_compute::cpu::ops::q4_common::quantize_q4_k(wq_f32.as_slice().unwrap());
    let wk_q4k = larql_compute::cpu::ops::q4_common::quantize_q4_k(wk_f32.as_slice().unwrap());
    let wv_q4k = larql_compute::cpu::ops::q4_common::quantize_q4_k(wv_f32.as_slice().unwrap());

    let ref_q = metal
        .q4k_matvec(&wq_q4k, &x, q_rows, hidden)
        .expect("q4k_matvec Q");
    let ref_k = metal
        .q4k_matvec(&wk_q4k, &x, kv_rows, hidden)
        .expect("q4k_matvec K");
    let ref_v = metal
        .q4k_matvec(&wv_q4k, &x, kv_rows, hidden)
        .expect("q4k_matvec V");

    // Fused dispatch through `q4k_qkv_proj`.
    let wq_buf = metal.bufs().get_bytes(&wq_q4k);
    let wk_buf = metal.bufs().get_bytes(&wk_q4k);
    let wv_buf = metal.bufs().get_bytes(&wv_q4k);
    let x_buf = metal.bufs().transient_from_f32(&x);
    let q_out = metal.bufs().output((q_rows * 4) as u64);
    let k_out = metal.bufs().output((kv_rows * 4) as u64);
    let v_out = metal.bufs().output((kv_rows * 4) as u64);

    use larql_compute_metal::shaders::q4k_qkv_proj as sh;
    let total_rows = (q_rows + kv_rows + kv_rows) as u64;
    let num_tgs = total_rows.div_ceil(sh::ROWS_PER_TG);
    let q_u = q_rows as u32;
    let k_u = kv_rows as u32;
    let v_u = kv_rows as u32;
    let hidden_u = hidden as u32;
    let cmd = metal.queue().new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&metal.attention.q4k_qkv_proj_pipeline.state);
    enc.set_buffer(0, Some(&wq_buf), 0);
    enc.set_buffer(1, Some(&wk_buf), 0);
    enc.set_buffer(2, Some(&wv_buf), 0);
    enc.set_buffer(3, Some(&x_buf), 0);
    enc.set_buffer(4, Some(&q_out), 0);
    enc.set_buffer(5, Some(&k_out), 0);
    enc.set_buffer(6, Some(&v_out), 0);
    enc.set_bytes(7, 4, &q_u as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(8, 4, &k_u as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(9, 4, &v_u as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(10, 4, &hidden_u as *const u32 as *const std::ffi::c_void);
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

    let check = |name: &str, r: &[f32], g: &[f32]| {
        let max_abs = r.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-6);
        let d = max_diff(r, g);
        assert!(
            d < max_abs * 1e-3,
            "{name}: max_diff {d:.3e} exceeds 0.1% of max_abs {max_abs:.3e}"
        );
    };
    check("Q", &ref_q, &got_q);
    check("K", &ref_k, &got_k);
    check("V", &ref_v, &got_v);
}

/// `q4k_q6k_qkv_proj` fused shader matches three separate-format dispatches.
///
/// Pins the mixed-quant fused kernel that replaces the 3-dispatch per-proj
/// fallback when a layer ships Q4_K Q/K + Q6_K V (Gemma 3 4B / Gemma 4
/// Ollama convention). If the shader silently regresses to under-read or
/// over-read the Q4_K GGUF 144-byte blocks (as happened once when the
/// first draft used the 148-byte `block_q4_K` MSL struct), this will
/// catch it before real-vindex decode produces garbled tokens.
#[test]
#[allow(clippy::unusual_byte_groupings)]
fn q4k_q6k_qkv_proj_matches_per_proj_dispatch() {
    let metal = get_metal();

    // Shapes modelled on Gemma 3 4B: q_dim = 8 * 256, kv_dim = 4 * 256,
    // hidden = 2560 (K must be a multiple of 256 for Q4_K / Q6_K).
    let q_rows = 2048usize;
    let kv_rows = 1024usize;
    let hidden = 2560usize;

    // Synthesise weight matrices and quantise.
    let wq_f32 = synth(q_rows, hidden, 0xdead_beef_1)
        .as_standard_layout()
        .to_owned();
    let wk_f32 = synth(kv_rows, hidden, 0xdead_beef_2)
        .as_standard_layout()
        .to_owned();
    let wv_f32 = synth(kv_rows, hidden, 0xdead_beef_3)
        .as_standard_layout()
        .to_owned();
    let x: Vec<f32> = (0..hidden).map(|i| ((i as f32) * 0.011).sin()).collect();

    let wq_q4k = larql_compute::cpu::ops::q4_common::quantize_q4_k(wq_f32.as_slice().unwrap());
    let wk_q4k = larql_compute::cpu::ops::q4_common::quantize_q4_k(wk_f32.as_slice().unwrap());
    let wv_q6k = larql_compute::cpu::ops::q4_common::quantize_q6_k(wv_f32.as_slice().unwrap());

    // Reference: dispatch each projection through its native shader.
    let ref_q = metal
        .q4k_matvec(&wq_q4k, &x, q_rows, hidden)
        .expect("q4k_matvec Q");
    let ref_k = metal
        .q4k_matvec(&wk_q4k, &x, kv_rows, hidden)
        .expect("q4k_matvec K");
    let ref_v = metal
        .q6k_matvec(&wv_q6k, &x, kv_rows, hidden)
        .expect("q6k_matvec V");

    // Fused dispatch.
    let wq_buf = metal.bufs().get_bytes(&wq_q4k);
    let wk_buf = metal.bufs().get_bytes(&wk_q4k);
    let wv_buf = metal.bufs().get_bytes(&wv_q6k);
    let x_buf = metal.bufs().transient_from_f32(&x);
    let q_out = metal.bufs().output((q_rows * 4) as u64);
    let k_out = metal.bufs().output((kv_rows * 4) as u64);
    let v_out = metal.bufs().output((kv_rows * 4) as u64);

    use larql_compute_metal::shaders::q4k_q6k_qkv_proj as sh;
    let total_rows = (q_rows + kv_rows + kv_rows) as u64;
    let num_tgs = total_rows.div_ceil(sh::ROWS_PER_TG);
    let q_u = q_rows as u32;
    let k_u = kv_rows as u32;
    let v_u = kv_rows as u32;
    let hidden_u = hidden as u32;
    let cmd = metal.queue().new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&metal.attention.q4k_q6k_qkv_proj_pipeline.state);
    enc.set_buffer(0, Some(&wq_buf), 0);
    enc.set_buffer(1, Some(&wk_buf), 0);
    enc.set_buffer(2, Some(&wv_buf), 0);
    enc.set_buffer(3, Some(&x_buf), 0);
    enc.set_buffer(4, Some(&q_out), 0);
    enc.set_buffer(5, Some(&k_out), 0);
    enc.set_buffer(6, Some(&v_out), 0);
    enc.set_bytes(7, 4, &q_u as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(8, 4, &k_u as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(9, 4, &v_u as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(10, 4, &hidden_u as *const u32 as *const std::ffi::c_void);
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

    // Q4_K quantisation can introduce tiny per-row scale differences
    // depending on which shader dispatch path is taken; absolute tolerance
    // scaled by row magnitude.
    let check = |name: &str, r: &[f32], g: &[f32]| {
        let max_abs = r.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-6);
        let d = max_diff(r, g);
        assert!(
            d < max_abs * 1e-3,
            "{name}: max_diff {d:.3e} exceeds 0.1% of max_abs {max_abs:.3e}"
        );
    };
    check("Q", &ref_q, &got_q);
    check("K", &ref_k, &got_k);
    check("V", &ref_v, &got_v);
}

/// Stage: `residual::encode_post_attn` with FFN that needs Q8 input.
///
/// Verifies the additional q8_quant dispatch runs and produces a Q8
/// representation that round-trips to approximately `ffn_norm_out`.
#[test]
fn stage_post_attn_q8_ffn_emits_roundtrippable_q8() {
    let device = metal::Device::system_default().unwrap();
    let rms_norm = build_pipeline(&device, "rms_norm");
    let residual_add = build_pipeline(&device, "residual_add");
    let q8_quant = build_pipeline(&device, "quantize_q8");
    let bufs = larql_compute_metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let hidden = 256usize;
    let seq_len = 2usize;

    let h: Vec<f32> = (0..seq_len * hidden)
        .map(|i| ((i as f32) * 0.009).sin() * 2.0)
        .collect();
    let o: Vec<f32> = (0..seq_len * hidden)
        .map(|i| ((i as f32) * 0.013).cos() * 1.5)
        .collect();
    let w: Vec<f32> = (0..hidden).map(|i| 1.0 + 0.02 * (i as f32).sin()).collect();

    let h_buf = bufs.transient_from_f32(&h);
    let o_buf = bufs.transient_from_f32(&o);
    let w_buf = bufs.transient_from_f32(&w);
    let h_pa = bufs.output((seq_len * hidden * 4) as u64);
    let ffn_out = bufs.output((seq_len * hidden * 4) as u64);
    let q8 = bufs.output((seq_len * hidden) as u64);
    let q8s = bufs.output((seq_len * hidden.div_ceil(32) * 4) as u64);

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    let mut scratch = |n: u64| bufs.output(n);
    larql_compute_metal::stages::residual::encode_post_attn(
        enc,
        &rms_norm,
        &residual_add,
        &q8_quant,
        &mut scratch,
        &h_buf,
        &o_buf,
        &h_pa,
        &ffn_out,
        &w_buf,
        &w_buf,
        &q8,
        &q8s,
        seq_len,
        hidden,
        1e-6,
        0.0,
        /*has_post_norms*/ false,
        /*ffn_needs_q8*/ true,
        (hidden * 4) as u64,
        hidden as u64,
        (hidden.div_ceil(32) * 4) as u64,
        1.0,
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    // Dequantise Q8 and compare to f32 ffn_norm_out (Q8 error < 1/127 * max).
    // `quantize_q8` writes f32 scales (not f16) — `q8s_stride_bytes` is
    // `blocks_per_row * 4` to reflect that.
    let ffn_f32 = read_f32_buf(&ffn_out, seq_len * hidden);
    let q8_bytes =
        unsafe { std::slice::from_raw_parts(q8.contents() as *const i8, seq_len * hidden) };
    let blocks_per_pos = hidden.div_ceil(32);
    let q8s_f32 = unsafe {
        std::slice::from_raw_parts(q8s.contents() as *const f32, seq_len * blocks_per_pos)
    };
    let mut dequant = vec![0.0f32; seq_len * hidden];
    for p in 0..seq_len {
        for b in 0..blocks_per_pos {
            let scale = q8s_f32[p * blocks_per_pos + b];
            for i in 0..32 {
                let idx = p * hidden + b * 32 + i;
                if idx < (p + 1) * hidden {
                    dequant[idx] = q8_bytes[idx] as f32 * scale;
                }
            }
        }
    }
    let max_abs = ffn_f32.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let d = max_diff(&ffn_f32, &dequant);
    assert!(
        d < max_abs / 100.0 + 1e-4,
        "Q8 roundtrip error {d} exceeds 1% of max_abs {max_abs}"
    );
}
