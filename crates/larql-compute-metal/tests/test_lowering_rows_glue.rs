//! VERIFY-N glue kernels against their single-position dispatches, bit for
//! bit: `rms_norm_rows` against `rms_norm` at each row's offset, and
//! `rope_rows` against `rope_at_pos_batched` at each row's position. Both
//! run the same per-row arithmetic, so equality is exact — a verify block
//! must predict what `step` would, and these ops are position-local.

#![cfg(target_os = "macos")]

use larql_compute_metal::MetalBackend;

const ROWS: usize = 5;
const HIDDEN: usize = 2560;
const HEADS: usize = 4;
const HEAD_DIM: usize = 256;
const BASE_POSITION: usize = 37;
const EPS: f32 = 1e-6;
const WEIGHT_OFFSET: f32 = 1.0;
const ROPE_THETA: f64 = 10_000.0;

fn values(seed: usize, len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (((i * 31 + seed) % 977) as f32 / 977.0) - 0.5)
        .collect()
}

fn run(gpu: &MetalBackend, encode: impl Fn(&metal::ComputeCommandEncoderRef)) {
    let cmd = gpu.new_lowering_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    encode(enc);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
}

#[test]
fn rms_norm_rows_is_bit_identical_to_per_row_rms_norm() {
    // Fail, never skip: a shader that does not compile makes `new()`
    // return None, and a skipped gate reads as a pass.
    let gpu = MetalBackend::new().expect("Metal backend (shader library must compile)");
    let f = std::mem::size_of::<f32>() as u64;
    let x = gpu.lowering_upload(&values(3, ROWS * HIDDEN)).expect("x");
    let w = gpu.lowering_upload(&values(5, HIDDEN)).expect("w");
    let rows_out = gpu.lowering_scratch(ROWS * HIDDEN);
    let each_out = gpu.lowering_scratch(ROWS * HIDDEN);
    run(&gpu, |enc| {
        gpu.encode_rms_norm_rows(
            enc,
            &x,
            0,
            &w,
            &rows_out,
            0,
            HIDDEN,
            ROWS,
            EPS,
            WEIGHT_OFFSET,
        );
        // The production single-row dispatch the lowering's step uses.
        for r in 0..ROWS {
            let off = (r * HIDDEN) as u64 * f;
            larql_compute_metal::stages::input_norm::encode_f32(
                enc,
                &gpu.norms.rms_norm_pipeline,
                &x,
                off,
                &w,
                &each_out,
                off,
                HIDDEN,
                EPS,
                WEIGHT_OFFSET,
            );
        }
    });
    let a = gpu.lowering_readback(&rows_out, ROWS * HIDDEN).expect("a");
    let b = gpu.lowering_readback(&each_out, ROWS * HIDDEN).expect("b");
    assert!(a.iter().zip(&b).all(|(p, q)| p.to_bits() == q.to_bits()));
}

#[test]
fn rope_rows_is_bit_identical_to_per_position_rope() {
    // Fail, never skip: a shader that does not compile makes `new()`
    // return None, and a skipped gate reads as a pass.
    let gpu = MetalBackend::new().expect("Metal backend (shader library must compile)");
    let f = std::mem::size_of::<f32>() as u64;
    let inv: Vec<f32> = (0..HEAD_DIM / 2)
        .map(|i| ROPE_THETA.powf(-2.0 * i as f64 / HEAD_DIM as f64) as f32)
        .collect();
    let inv = gpu.lowering_upload(&inv).expect("inv_freq");
    let src = values(7, ROWS * HEADS * HEAD_DIM);
    let rows_x = gpu.lowering_upload(&src).expect("x");
    let each_x = gpu.lowering_upload(&src).expect("x");
    run(&gpu, |enc| {
        gpu.encode_rope_rows(
            enc,
            &rows_x,
            0,
            HEADS,
            HEAD_DIM,
            &inv,
            BASE_POSITION,
            1.0,
            ROWS,
        );
        for r in 0..ROWS {
            gpu.encode_rope(
                enc,
                &each_x,
                (r * HEADS * HEAD_DIM) as u64 * f,
                HEADS,
                HEAD_DIM,
                &inv,
                BASE_POSITION + r,
                1.0,
            );
        }
    });
    let a = gpu.lowering_readback(&rows_x, src.len()).expect("a");
    let b = gpu.lowering_readback(&each_x, src.len()).expect("b");
    assert!(a.iter().zip(&b).all(|(p, q)| p.to_bits() == q.to_bits()));
}

/// Gemma 3 4B attention geometry.
const Q_HEADS: usize = 8;
const KV_HEADS: usize = 4;
const ATTN_HEAD_DIM: usize = 256;
/// Row 0's cache length (the block's first position, inclusive).
const BASE_KV_LEN: usize = 40;
/// A window shorter than the block's spans, so `t_start` moves per row.
const SLIDING_WINDOW: u32 = 16;
/// Sequence-parallel slices under test: 4 x 256 = 1024 threads, the
/// kernel's ceiling at this head_dim and the Gemma 3 planner row.
const SEQPAR_SLICES: u64 = 4;

#[allow(clippy::too_many_arguments)]
fn dispatch_attention(
    enc: &metal::ComputeCommandEncoderRef,
    pipeline: &metal::ComputePipelineState,
    bufs: (
        &metal::Buffer,
        &metal::Buffer,
        &metal::Buffer,
        &metal::Buffer,
    ),
    offsets: (u64, u64),
    kv_len: usize,
    window: u32,
    grid_rows: usize,
    threads: u64,
) {
    let (q, k, v, out) = bufs;
    let set_u32 = |i: u64, x: u32| enc.set_bytes(i, 4, &x as *const u32 as *const std::ffi::c_void);
    let set_f32 = |i: u64, x: f32| enc.set_bytes(i, 4, &x as *const f32 as *const std::ffi::c_void);
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(q), offsets.0);
    enc.set_buffer(1, Some(k), 0);
    enc.set_buffer(2, Some(v), 0);
    enc.set_buffer(3, Some(out), offsets.1);
    set_u32(4, kv_len as u32);
    set_u32(5, ATTN_HEAD_DIM as u32);
    set_u32(6, Q_HEADS as u32);
    set_u32(7, KV_HEADS as u32);
    set_f32(8, 1.0 / (ATTN_HEAD_DIM as f32).sqrt());
    set_u32(9, window);
    enc.set_buffer(10, Some(q), 0);
    set_u32(11, 0);
    set_f32(12, 0.0);
    enc.dispatch_thread_groups(
        metal::MTLSize::new(Q_HEADS as u64, grid_rows as u64, 1),
        metal::MTLSize::new(threads, 1, 1),
    );
}

/// One rows dispatch against per-position dispatches of `single`, bit for
/// bit, with and without a sliding window.
fn rows_match_per_position(
    gpu: &MetalBackend,
    rows_pipeline: &metal::ComputePipelineState,
    single: &metal::ComputePipelineState,
    threads: u64,
) {
    let f = std::mem::size_of::<f32>() as u64;
    let q_row = Q_HEADS * ATTN_HEAD_DIM;
    let kv_rows = BASE_KV_LEN + ROWS;
    let q = gpu.lowering_upload(&values(11, ROWS * q_row)).expect("q");
    let k = gpu
        .lowering_upload(&values(13, kv_rows * KV_HEADS * ATTN_HEAD_DIM))
        .expect("k");
    let v = gpu
        .lowering_upload(&values(17, kv_rows * KV_HEADS * ATTN_HEAD_DIM))
        .expect("v");
    for window in [0, SLIDING_WINDOW] {
        let rows_out = gpu.lowering_scratch(ROWS * q_row);
        let each_out = gpu.lowering_scratch(ROWS * q_row);
        run(gpu, |enc| {
            dispatch_attention(
                enc,
                rows_pipeline,
                (&q, &k, &v, &rows_out),
                (0, 0),
                BASE_KV_LEN,
                window,
                ROWS,
                threads,
            );
            for r in 0..ROWS {
                let off = (r * q_row) as u64 * f;
                dispatch_attention(
                    enc,
                    single,
                    (&q, &k, &v, &each_out),
                    (off, off),
                    BASE_KV_LEN + r,
                    window,
                    1,
                    threads,
                );
            }
        });
        let a = gpu.lowering_readback(&rows_out, ROWS * q_row).expect("a");
        let b = gpu.lowering_readback(&each_out, ROWS * q_row).expect("b");
        assert!(
            a.iter().zip(&b).all(|(p, q)| p.to_bits() == q.to_bits()),
            "window {window}: the rows kernel differs from per-position attention"
        );
    }
}

#[test]
fn kv_attention_rows_is_bit_identical_to_per_position_attention() {
    let gpu = MetalBackend::new().expect("Metal backend (shader library must compile)");
    let serial_threads = gpu
        .attention
        .kv_attend_pipeline
        .max_total_threads_per_threadgroup()
        .min(256);
    rows_match_per_position(
        &gpu,
        &gpu.attention.kv_attend_rows_pipeline,
        &gpu.attention.kv_attend_pipeline,
        serial_threads,
    );
}

#[test]
fn kv_attention_seqpar_rows_is_bit_identical_to_per_position_seqpar() {
    let gpu = MetalBackend::new().expect("Metal backend (shader library must compile)");
    rows_match_per_position(
        &gpu,
        &gpu.attention.kv_attend_seqpar_rows_pipeline,
        &gpu.attention.kv_attend_seqpar_pipeline,
        SEQPAR_SLICES * ATTN_HEAD_DIM as u64,
    );
}
