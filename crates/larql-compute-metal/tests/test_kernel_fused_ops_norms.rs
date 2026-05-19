//! Correctness tests for norm, residual, and quantization Metal shaders:
//! `rms_norm` (with offset, zero offset, large vector SIMD cooperative),
//! `residual_norm` (SIMD cooperative), `residual_add`, `quantize_q8`,
//! and fused ops: `rms_norm_q8`, `residual_norm` (vs CPU), `residual_norm_q8`.
//!
//! All tests compare Metal shader output to a CPU reference implementation.

#![cfg(target_os = "macos")]

extern crate blas_src;

#[path = "common/mod.rs"]
mod common;
use common::max_diff;

// ── rms_norm with offset ──

#[test]
fn rms_norm_matches_cpu() {
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute_metal::shaders::all_shaders();
    let lib = device
        .new_library_with_source(&src, &metal::CompileOptions::new())
        .unwrap();
    let pipeline = device
        .new_compute_pipeline_state_with_function(&lib.get_function("rms_norm", None).unwrap())
        .unwrap();
    let bufs = larql_compute_metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let len = 64usize;
    let x: Vec<f32> = (0..len).map(|i| i as f32 * 0.1 - 3.2).collect();
    let weight: Vec<f32> = (0..len).map(|i| 0.5 + (i as f32 * 0.01)).collect();
    let eps = 1e-6f32;
    let offset = 1.0f32; // Gemma 2/3 style (Gemma 4 uses 0.0)

    // CPU reference
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let rms = 1.0 / (sum_sq / len as f32 + eps).sqrt();
    let cpu_result: Vec<f32> = x
        .iter()
        .zip(weight.iter())
        .map(|(xi, wi)| xi * (wi + offset) * rms)
        .collect();

    // Metal
    let buf_x = bufs.transient_from_f32(&x);
    let buf_w = bufs.transient_from_f32(&weight);
    let buf_out = bufs.output((len * 4) as u64);
    let len_val = len as u32;

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&buf_x), 0);
    enc.set_buffer(1, Some(&buf_w), 0);
    enc.set_buffer(2, Some(&buf_out), 0);
    enc.set_bytes(3, 4, &len_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(4, 4, &eps as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(5, 4, &offset as *const f32 as *const std::ffi::c_void);
    // Single threadgroup dispatch for cooperative SIMD reduction.
    enc.dispatch_thread_groups(
        metal::MTLSize::new(1, 1, 1),
        metal::MTLSize::new(len as u64, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let ptr = buf_out.contents() as *const f32;
    let metal_result: Vec<f32> = unsafe { std::slice::from_raw_parts(ptr, len).to_vec() };

    let diff = max_diff(&cpu_result, &metal_result);
    assert!(diff < 1e-5, "rms_norm max diff {diff}");
}

#[test]
fn rms_norm_zero_offset() {
    // Standard RMS norm (Llama-style, offset=0)
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute_metal::shaders::all_shaders();
    let lib = device
        .new_library_with_source(&src, &metal::CompileOptions::new())
        .unwrap();
    let pipeline = device
        .new_compute_pipeline_state_with_function(&lib.get_function("rms_norm", None).unwrap())
        .unwrap();
    let bufs = larql_compute_metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let len = 32usize;
    let x: Vec<f32> = (0..len).map(|i| i as f32 * 0.2 - 3.0).collect();
    let weight: Vec<f32> = vec![1.0f32; len];
    let eps = 1e-6f32;
    let offset = 0.0f32;

    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let rms = 1.0 / (sum_sq / len as f32 + eps).sqrt();
    let cpu_result: Vec<f32> = x.iter().map(|xi| xi * rms).collect();

    let buf_x = bufs.transient_from_f32(&x);
    let buf_w = bufs.transient_from_f32(&weight);
    let buf_out = bufs.output((len * 4) as u64);
    let len_val = len as u32;

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&buf_x), 0);
    enc.set_buffer(1, Some(&buf_w), 0);
    enc.set_buffer(2, Some(&buf_out), 0);
    enc.set_bytes(3, 4, &len_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(4, 4, &eps as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(5, 4, &offset as *const f32 as *const std::ffi::c_void);
    enc.dispatch_thread_groups(
        metal::MTLSize::new(1, 1, 1),
        metal::MTLSize::new(len as u64, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let ptr = buf_out.contents() as *const f32;
    let metal_result: Vec<f32> = unsafe { std::slice::from_raw_parts(ptr, len).to_vec() };

    let diff = max_diff(&cpu_result, &metal_result);
    assert!(diff < 1e-5, "rms_norm(offset=0) max diff {diff}");
}

// ── cooperative SIMD norm (large vector, multi-simdgroup) ──

#[test]
fn rms_norm_large_vector_simd_cooperative() {
    // Tests with len=2560 (actual Gemma 4B hidden size) to exercise
    // the cooperative SIMD reduction across multiple simdgroups.
    // With TG=256: 8 simdgroups, each sums a 2560/256=10-element stripe.
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute_metal::shaders::all_shaders();
    let lib = device
        .new_library_with_source(&src, &metal::CompileOptions::new())
        .unwrap();
    let pipeline = device
        .new_compute_pipeline_state_with_function(&lib.get_function("rms_norm", None).unwrap())
        .unwrap();
    let bufs = larql_compute_metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let len = 2560usize;
    let x: Vec<f32> = (0..len).map(|i| (i as f32 * 0.0037).sin() * 2.0).collect();
    let weight: Vec<f32> = (0..len).map(|i| 0.8 + (i as f32 * 0.0001)).collect();
    let eps = 1e-6f32;
    let offset = 1.0f32;

    // CPU reference
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let rms = 1.0 / (sum_sq / len as f32 + eps).sqrt();
    let cpu_result: Vec<f32> = x
        .iter()
        .zip(weight.iter())
        .map(|(xi, wi)| xi * (wi + offset) * rms)
        .collect();

    let buf_x = bufs.transient_from_f32(&x);
    let buf_w = bufs.transient_from_f32(&weight);
    let buf_out = bufs.output((len * 4) as u64);
    let len_val = len as u32;

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&buf_x), 0);
    enc.set_buffer(1, Some(&buf_w), 0);
    enc.set_buffer(2, Some(&buf_out), 0);
    enc.set_bytes(3, 4, &len_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(4, 4, &eps as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(5, 4, &offset as *const f32 as *const std::ffi::c_void);
    // Single threadgroup dispatch — cooperative SIMD reduction needs all threads in one TG.
    enc.dispatch_thread_groups(metal::MTLSize::new(1, 1, 1), metal::MTLSize::new(256, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let metal_result = larql_compute_metal::buffers::read_buffer_f32(&buf_out, len);
    let diff = max_diff(&cpu_result, &metal_result);
    assert!(
        diff < 1e-4,
        "rms_norm(len=2560) SIMD cooperative max diff {diff}"
    );
}

#[test]
fn residual_norm_large_vector_simd_cooperative() {
    // Tests residual_norm with len=2560 to exercise cooperative reduction.
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute_metal::shaders::all_shaders();
    let lib = device
        .new_library_with_source(&src, &metal::CompileOptions::new())
        .unwrap();
    let pipeline = device
        .new_compute_pipeline_state_with_function(&lib.get_function("residual_norm", None).unwrap())
        .unwrap();
    let bufs = larql_compute_metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let len = 2560usize;
    let a: Vec<f32> = (0..len).map(|i| (i as f32 * 0.003).cos() * 1.5).collect();
    let b: Vec<f32> = (0..len).map(|i| (i as f32 * 0.007).sin() * 0.5).collect();
    let weight: Vec<f32> = (0..len).map(|i| 0.9 + (i as f32 * 0.00005)).collect();
    let eps = 1e-6f32;
    let offset = 0.0f32;

    // CPU reference: h = a + b, then rms_norm(h)
    let h: Vec<f32> = a.iter().zip(&b).map(|(ai, bi)| ai + bi).collect();
    let sum_sq: f32 = h.iter().map(|v| v * v).sum();
    let rms = 1.0 / (sum_sq / len as f32 + eps).sqrt();
    let cpu_result: Vec<f32> = h
        .iter()
        .zip(weight.iter())
        .map(|(hi, wi)| hi * (wi + offset) * rms)
        .collect();

    let buf_a = bufs.transient_from_f32(&a);
    let buf_b = bufs.transient_from_f32(&b);
    let buf_w = bufs.transient_from_f32(&weight);
    let buf_out = bufs.output((len * 4) as u64);
    let len_val = len as u32;

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&buf_a), 0);
    enc.set_buffer(1, Some(&buf_b), 0);
    enc.set_buffer(2, Some(&buf_w), 0);
    enc.set_buffer(3, Some(&buf_out), 0);
    enc.set_bytes(4, 4, &len_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(5, 4, &eps as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(6, 4, &offset as *const f32 as *const std::ffi::c_void);
    enc.dispatch_thread_groups(metal::MTLSize::new(1, 1, 1), metal::MTLSize::new(256, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let metal_result = larql_compute_metal::buffers::read_buffer_f32(&buf_out, len);
    let diff = max_diff(&cpu_result, &metal_result);
    assert!(
        diff < 1e-4,
        "residual_norm(len=2560) SIMD cooperative max diff {diff}"
    );
}

// ── residual_add ──

#[test]
fn residual_add_matches_cpu() {
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute_metal::shaders::all_shaders();
    let lib = device
        .new_library_with_source(&src, &metal::CompileOptions::new())
        .unwrap();
    let pipeline = device
        .new_compute_pipeline_state_with_function(&lib.get_function("residual_add", None).unwrap())
        .unwrap();
    let bufs = larql_compute_metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let len = 128usize;
    let a: Vec<f32> = (0..len).map(|i| i as f32 * 0.1).collect();
    let b: Vec<f32> = (0..len).map(|i| -(i as f32 * 0.05)).collect();
    let cpu_result: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();

    let buf_a = bufs.transient_from_f32(&a);
    let buf_b = bufs.transient_from_f32(&b);
    let buf_out = bufs.output((len * 4) as u64);
    let len_val = len as u32;

    // residual_add now takes a `b_scale` at buffer(4) so Granite-family
    // models can pass `residual_multiplier` through the kernel. Bind 1.0
    // to recover the original `out = a + b` semantics for this test.
    let b_scale: f32 = 1.0;
    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&buf_a), 0);
    enc.set_buffer(1, Some(&buf_b), 0);
    enc.set_buffer(2, Some(&buf_out), 0);
    enc.set_bytes(3, 4, &len_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(4, 4, &b_scale as *const f32 as *const std::ffi::c_void);
    enc.dispatch_threads(
        metal::MTLSize::new(len as u64, 1, 1),
        metal::MTLSize::new(len as u64, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let ptr = buf_out.contents() as *const f32;
    let metal_result: Vec<f32> = unsafe { std::slice::from_raw_parts(ptr, len).to_vec() };

    let diff = max_diff(&cpu_result, &metal_result);
    assert!(diff < 1e-6, "residual_add max diff {diff}");
}

/// Granite-family `residual_multiplier`: the same kernel with
/// `b_scale != 1.0` must compute `out = a + b_scale * b`. This was the
/// missing piece that caused Granite 4.1's transformer residual stream
/// to overflow (3B residual_multiplier = 0.22; running it as 1.0
/// produced "ikea ikea ikea…" degenerate output).
#[test]
fn residual_add_applies_b_scale() {
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute_metal::shaders::all_shaders();
    let lib = device
        .new_library_with_source(&src, &metal::CompileOptions::new())
        .unwrap();
    let pipeline = device
        .new_compute_pipeline_state_with_function(&lib.get_function("residual_add", None).unwrap())
        .unwrap();
    let bufs = larql_compute_metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let len = 128usize;
    let a: Vec<f32> = (0..len).map(|i| i as f32 * 0.1).collect();
    let b: Vec<f32> = (0..len).map(|i| -(i as f32 * 0.05)).collect();
    // Granite 4.1 3B / 8B residual_multiplier value.
    let b_scale: f32 = 0.22;
    let cpu_result: Vec<f32> = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| x + b_scale * y)
        .collect();

    let buf_a = bufs.transient_from_f32(&a);
    let buf_b = bufs.transient_from_f32(&b);
    let buf_out = bufs.output((len * 4) as u64);
    let len_val = len as u32;

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&buf_a), 0);
    enc.set_buffer(1, Some(&buf_b), 0);
    enc.set_buffer(2, Some(&buf_out), 0);
    enc.set_bytes(3, 4, &len_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(4, 4, &b_scale as *const f32 as *const std::ffi::c_void);
    enc.dispatch_threads(
        metal::MTLSize::new(len as u64, 1, 1),
        metal::MTLSize::new(len as u64, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let ptr = buf_out.contents() as *const f32;
    let metal_result: Vec<f32> = unsafe { std::slice::from_raw_parts(ptr, len).to_vec() };
    let diff = max_diff(&cpu_result, &metal_result);
    assert!(diff < 1e-6, "residual_add b_scale=0.22 max diff {diff}");
}

// ── quantize_q8 shader ──

#[test]
fn quantize_q8_matches_cpu() {
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute_metal::shaders::all_shaders();
    let lib = device
        .new_library_with_source(&src, &metal::CompileOptions::new())
        .unwrap();
    let pipeline = device
        .new_compute_pipeline_state_with_function(&lib.get_function("quantize_q8", None).unwrap())
        .unwrap();
    let bufs = larql_compute_metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let len = 64usize;
    let x: Vec<f32> = (0..len).map(|i| i as f32 * 0.15 - 4.8).collect();

    // CPU reference
    let (cpu_q8, cpu_scales) = larql_compute::cpu::q4::quantize_to_q8(&x);

    // Metal
    let buf_x = bufs.transient_from_f32(&x);
    let buf_q8 = bufs.output(len as u64);
    let buf_scales = bufs.output((len / 32 * 4) as u64);
    let len_val = len as u32;

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&buf_x), 0);
    enc.set_buffer(1, Some(&buf_q8), 0);
    enc.set_buffer(2, Some(&buf_scales), 0);
    let n_blocks = (len / 32) as u32;
    enc.set_bytes(3, 4, &len_val as *const u32 as *const std::ffi::c_void);
    enc.dispatch_threads(
        metal::MTLSize::new(n_blocks as u64, 1, 1),
        metal::MTLSize::new(n_blocks as u64, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let q8_ptr = buf_q8.contents() as *const i8;
    let sc_ptr = buf_scales.contents() as *const f32;
    let metal_q8: Vec<i8> = unsafe { std::slice::from_raw_parts(q8_ptr, len).to_vec() };
    let metal_scales: Vec<f32> = unsafe { std::slice::from_raw_parts(sc_ptr, len / 32).to_vec() };

    // Check scales match
    for i in 0..len / 32 {
        let diff = (cpu_scales[i] - metal_scales[i]).abs();
        assert!(
            diff < 0.01,
            "Q8 scale[{i}] diff: cpu={} metal={}",
            cpu_scales[i],
            metal_scales[i]
        );
    }
    // Check quantized values match (allow ±1 for rounding)
    let mut mismatches = 0;
    for i in 0..len {
        if (cpu_q8[i] as i32 - metal_q8[i] as i32).abs() > 1 {
            mismatches += 1;
        }
    }
    assert!(
        mismatches == 0,
        "Q8 quantize: {mismatches}/{len} values differ by >1"
    );
}

// ── Fused ops: rms_norm_q8, residual_norm, residual_norm_q8 ──

#[test]
fn rms_norm_q8_matches_separate_ops() {
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute_metal::shaders::all_shaders();
    let lib = device
        .new_library_with_source(&src, &metal::CompileOptions::new())
        .unwrap();
    let fused = device
        .new_compute_pipeline_state_with_function(&lib.get_function("rms_norm_q8", None).unwrap())
        .unwrap();
    let bufs = larql_compute_metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let len = 64usize;
    let x: Vec<f32> = (0..len).map(|i| i as f32 * 0.15 - 4.8).collect();
    let weight: Vec<f32> = (0..len).map(|i| 0.5 + i as f32 * 0.01).collect();
    let eps = 1e-6f32;
    let offset = 1.0f32;

    // CPU reference: norm then quantize
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let rms = 1.0 / (sum_sq / len as f32 + eps).sqrt();
    let normed: Vec<f32> = x
        .iter()
        .zip(weight.iter())
        .map(|(xi, wi)| xi * (wi + offset) * rms)
        .collect();
    let (cpu_q8, cpu_scales) = larql_compute::cpu::q4::quantize_to_q8(&normed);

    // Metal fused
    let buf_x = bufs.transient_from_f32(&x);
    let buf_w = bufs.transient_from_f32(&weight);
    let buf_q8 = bufs.output(len as u64);
    let buf_sc = bufs.output((len / 32 * 4) as u64);
    let len_val = len as u32;

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&fused);
    enc.set_buffer(0, Some(&buf_x), 0);
    enc.set_buffer(1, Some(&buf_w), 0);
    enc.set_buffer(2, Some(&buf_q8), 0);
    enc.set_buffer(3, Some(&buf_sc), 0);
    enc.set_bytes(4, 4, &len_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(5, 4, &eps as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(6, 4, &offset as *const f32 as *const std::ffi::c_void);
    enc.dispatch_threads(
        metal::MTLSize::new(len as u64, 1, 1),
        metal::MTLSize::new(len as u64, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let q8_ptr = buf_q8.contents() as *const i8;
    let sc_ptr = buf_sc.contents() as *const f32;
    let metal_q8: Vec<i8> = unsafe { std::slice::from_raw_parts(q8_ptr, len).to_vec() };
    let metal_sc: Vec<f32> = unsafe { std::slice::from_raw_parts(sc_ptr, len / 32).to_vec() };

    // Check scales match
    for i in 0..len / 32 {
        let diff = (cpu_scales[i] - metal_sc[i]).abs();
        assert!(
            diff < 0.1,
            "fused rms_norm_q8 scale[{i}] diff: cpu={} metal={}",
            cpu_scales[i],
            metal_sc[i]
        );
    }
    // Check Q8 values (allow ±2 rounding)
    let mut bad = 0;
    for i in 0..len {
        if (cpu_q8[i] as i32 - metal_q8[i] as i32).abs() > 2 {
            bad += 1;
        }
    }
    assert!(
        bad == 0,
        "fused rms_norm_q8: {bad}/{len} values differ by >2"
    );
}

#[test]
fn residual_norm_matches_separate_ops() {
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute_metal::shaders::all_shaders();
    let lib = device
        .new_library_with_source(&src, &metal::CompileOptions::new())
        .unwrap();
    let fused = device
        .new_compute_pipeline_state_with_function(&lib.get_function("residual_norm", None).unwrap())
        .unwrap();
    let bufs = larql_compute_metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let len = 64usize;
    let a: Vec<f32> = (0..len).map(|i| i as f32 * 0.1 - 3.2).collect();
    let b: Vec<f32> = (0..len).map(|i| i as f32 * 0.05 + 0.3).collect();
    let weight: Vec<f32> = (0..len).map(|i| 0.8 + i as f32 * 0.005).collect();
    let eps = 1e-6f32;
    let offset = 0.0f32;

    // CPU reference: add then norm
    let sum: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
    let sum_sq: f32 = sum.iter().map(|v| v * v).sum();
    let rms = 1.0 / (sum_sq / len as f32 + eps).sqrt();
    let cpu_result: Vec<f32> = sum
        .iter()
        .zip(weight.iter())
        .map(|(s, w)| s * (w + offset) * rms)
        .collect();

    // Metal fused
    let buf_a = bufs.transient_from_f32(&a);
    let buf_b = bufs.transient_from_f32(&b);
    let buf_w = bufs.transient_from_f32(&weight);
    let buf_out = bufs.output((len * 4) as u64);
    let len_val = len as u32;

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&fused);
    enc.set_buffer(0, Some(&buf_a), 0);
    enc.set_buffer(1, Some(&buf_b), 0);
    enc.set_buffer(2, Some(&buf_w), 0);
    enc.set_buffer(3, Some(&buf_out), 0);
    enc.set_bytes(4, 4, &len_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(5, 4, &eps as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(6, 4, &offset as *const f32 as *const std::ffi::c_void);
    enc.dispatch_threads(
        metal::MTLSize::new(len as u64, 1, 1),
        metal::MTLSize::new(len as u64, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let ptr = buf_out.contents() as *const f32;
    let metal_result: Vec<f32> = unsafe { std::slice::from_raw_parts(ptr, len).to_vec() };
    let diff = max_diff(&cpu_result, &metal_result);
    assert!(diff < 1e-4, "residual_norm max diff {diff}");
}

/// `residual_norm_store` is the kernel D-RMS-FUSE Phase 1 dispatches at
/// the post-FFN→next-input boundary on non-Gemma archs. It writes both
/// `sum_out = a + b` (raw, for the residual base used by next layer's
/// post-attn) and `norm_out = rms_norm(a + b, weight)` (the next
/// layer's pre-normed input). This test pins both outputs against the
/// separated CPU reference.
#[test]
fn residual_norm_store_matches_separate_ops() {
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute_metal::shaders::all_shaders();
    let lib = device
        .new_library_with_source(&src, &metal::CompileOptions::new())
        .unwrap();
    let fused = device
        .new_compute_pipeline_state_with_function(
            &lib.get_function("residual_norm_store", None).unwrap(),
        )
        .unwrap();
    let bufs = larql_compute_metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let len = 64usize;
    let a: Vec<f32> = (0..len).map(|i| i as f32 * 0.07 - 1.5).collect();
    let b: Vec<f32> = (0..len).map(|i| i as f32 * 0.04 + 0.2).collect();
    let weight: Vec<f32> = (0..len).map(|i| 0.6 + i as f32 * 0.003).collect();
    let eps = 1e-6f32;
    let offset = 0.0f32;

    // CPU reference: sum_out = a+b, norm_out = rms_norm(a+b, weight)
    let cpu_sum: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
    let sum_sq: f32 = cpu_sum.iter().map(|v| v * v).sum();
    let rms = 1.0 / (sum_sq / len as f32 + eps).sqrt();
    let cpu_norm: Vec<f32> = cpu_sum
        .iter()
        .zip(weight.iter())
        .map(|(s, w)| s * (w + offset) * rms)
        .collect();

    // Metal fused
    let buf_a = bufs.transient_from_f32(&a);
    let buf_b = bufs.transient_from_f32(&b);
    let buf_w = bufs.transient_from_f32(&weight);
    let buf_norm = bufs.output((len * 4) as u64);
    let buf_sum = bufs.output((len * 4) as u64);
    let len_val = len as u32;
    // `residual_norm_store` now takes `b_scale` at buffer(8) (Granite
    // residual_multiplier). 1.0 recovers the original `a + b` semantics
    // for non-Granite parity tests.
    let b_scale: f32 = 1.0;

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&fused);
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
        metal::MTLSize::new(256.min(len as u64), 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let norm_ptr = buf_norm.contents() as *const f32;
    let sum_ptr = buf_sum.contents() as *const f32;
    let metal_norm: Vec<f32> = unsafe { std::slice::from_raw_parts(norm_ptr, len).to_vec() };
    let metal_sum: Vec<f32> = unsafe { std::slice::from_raw_parts(sum_ptr, len).to_vec() };

    let diff_norm = max_diff(&cpu_norm, &metal_norm);
    assert!(
        diff_norm < 1e-4,
        "residual_norm_store norm_out max diff {diff_norm}"
    );
    let diff_sum = max_diff(&cpu_sum, &metal_sum);
    assert!(
        diff_sum < 1e-6,
        "residual_norm_store sum_out max diff {diff_sum} (must be exact)"
    );
}

/// Granite residual_multiplier coverage on the fused
/// `residual_norm_store` kernel: with `b_scale = 0.22` the kernel must
/// compute `sum = a + 0.22 * b` and `norm = rms_norm(sum)`. This is the
/// kernel that fires on the post-attention residual during single-token
/// decode (`encode_attn.rs` ffn_uses_kquant branch) and on the optional
/// D-RMS-FUSE Phase 1 fusion at the post-FFN→next-input boundary.
#[test]
fn residual_norm_store_applies_b_scale() {
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute_metal::shaders::all_shaders();
    let lib = device
        .new_library_with_source(&src, &metal::CompileOptions::new())
        .unwrap();
    let fused = device
        .new_compute_pipeline_state_with_function(
            &lib.get_function("residual_norm_store", None).unwrap(),
        )
        .unwrap();
    let bufs = larql_compute_metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let len = 64usize;
    let a: Vec<f32> = (0..len).map(|i| i as f32 * 0.07 - 1.5).collect();
    let b: Vec<f32> = (0..len).map(|i| i as f32 * 0.04 + 0.2).collect();
    let weight: Vec<f32> = (0..len).map(|i| 0.6 + i as f32 * 0.003).collect();
    let eps = 1e-6f32;
    let offset = 0.0f32;
    // Granite 4.1 3B/8B value.
    let b_scale: f32 = 0.22;

    let cpu_sum: Vec<f32> = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| x + b_scale * y)
        .collect();
    let sum_sq: f32 = cpu_sum.iter().map(|v| v * v).sum();
    let rms = 1.0 / (sum_sq / len as f32 + eps).sqrt();
    let cpu_norm: Vec<f32> = cpu_sum
        .iter()
        .zip(weight.iter())
        .map(|(s, w)| s * (w + offset) * rms)
        .collect();

    let buf_a = bufs.transient_from_f32(&a);
    let buf_b = bufs.transient_from_f32(&b);
    let buf_w = bufs.transient_from_f32(&weight);
    let buf_norm = bufs.output((len * 4) as u64);
    let buf_sum = bufs.output((len * 4) as u64);
    let len_val = len as u32;

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&fused);
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
        metal::MTLSize::new(256.min(len as u64), 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let norm_ptr = buf_norm.contents() as *const f32;
    let sum_ptr = buf_sum.contents() as *const f32;
    let metal_norm: Vec<f32> = unsafe { std::slice::from_raw_parts(norm_ptr, len).to_vec() };
    let metal_sum: Vec<f32> = unsafe { std::slice::from_raw_parts(sum_ptr, len).to_vec() };

    let diff_sum = max_diff(&cpu_sum, &metal_sum);
    assert!(
        diff_sum < 1e-6,
        "residual_norm_store b_scale=0.22 sum_out diff {diff_sum} (must be exact)"
    );
    let diff_norm = max_diff(&cpu_norm, &metal_norm);
    assert!(
        diff_norm < 1e-4,
        "residual_norm_store b_scale=0.22 norm_out diff {diff_norm}"
    );
}

/// D-RMS-FUSE Phase 1 with a non-zero `offset` (Gemma-style HF norm
/// where the saved norm weight already has +1 baked in). Pins the
/// shader's `(weight[i] + offset) * rms` formula.
#[test]
fn residual_norm_store_with_norm_offset() {
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute_metal::shaders::all_shaders();
    let lib = device
        .new_library_with_source(&src, &metal::CompileOptions::new())
        .unwrap();
    let fused = device
        .new_compute_pipeline_state_with_function(
            &lib.get_function("residual_norm_store", None).unwrap(),
        )
        .unwrap();
    let bufs = larql_compute_metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let len = 128usize;
    let a: Vec<f32> = (0..len).map(|i| (i as f32 * 0.11).sin() * 0.4).collect();
    let b: Vec<f32> = (0..len).map(|i| (i as f32 * 0.09).cos() * 0.3).collect();
    let weight: Vec<f32> = (0..len).map(|i| (i as f32 * 0.02).sin() * 0.05).collect();
    let eps = 1e-6f32;
    let offset = 1.0f32; // Gemma-style HF norm offset

    let cpu_sum: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
    let sum_sq: f32 = cpu_sum.iter().map(|v| v * v).sum();
    let rms = 1.0 / (sum_sq / len as f32 + eps).sqrt();
    let cpu_norm: Vec<f32> = cpu_sum
        .iter()
        .zip(weight.iter())
        .map(|(s, w)| s * (w + offset) * rms)
        .collect();

    let buf_a = bufs.transient_from_f32(&a);
    let buf_b = bufs.transient_from_f32(&b);
    let buf_w = bufs.transient_from_f32(&weight);
    let buf_norm = bufs.output((len * 4) as u64);
    let buf_sum = bufs.output((len * 4) as u64);
    let len_val = len as u32;
    // `residual_norm_store` now requires `b_scale` at buffer(8). 1.0 is
    // the no-op for non-Granite parity tests.
    let b_scale: f32 = 1.0;

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&fused);
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
        metal::MTLSize::new(256.min(len as u64), 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let norm_ptr = buf_norm.contents() as *const f32;
    let sum_ptr = buf_sum.contents() as *const f32;
    let metal_norm: Vec<f32> = unsafe { std::slice::from_raw_parts(norm_ptr, len).to_vec() };
    let metal_sum: Vec<f32> = unsafe { std::slice::from_raw_parts(sum_ptr, len).to_vec() };

    let diff_norm = max_diff(&cpu_norm, &metal_norm);
    assert!(
        diff_norm < 1e-4,
        "residual_norm_store with offset norm_out max diff {diff_norm}"
    );
    let diff_sum = max_diff(&cpu_sum, &metal_sum);
    assert!(
        diff_sum < 1e-6,
        "residual_norm_store with offset sum_out max diff {diff_sum}"
    );
}
