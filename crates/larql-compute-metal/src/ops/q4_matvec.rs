//! Q4×Q8 matrix-vector dispatch.
//!
//! scores[N] = Q4[N, K] @ Q8_x[K]
//!
//! The dispatcher takes a [`KernelHandle`] which carries both the
//! pipeline state and the row-tiling geometry the kernel expects.
//! Geometry travels with the pipeline; bumping the kernel can't
//! desync the dispatcher. (See `metal::kernel` and the q4_matvec_v4
//! 75 %-row-drop ship-log entry.)

use metal::*;
use std::ffi::c_void;

use crate::buffers::BufferCache;
use crate::kernels::KernelHandle;

/// Dispatch a single Q4 matvec on GPU.
///
/// - `kernel`: the q4 matvec [`KernelHandle`] (carries pipeline +
///   row-tiling geometry; geometry can't drift from the kernel)
/// - `q4_data`: packed Q4_0 weights (cached, mmap-backed)
/// - `q8_x`: pre-quantized input vector (transient)
/// - `q8_scales`: per-block Q8 scales (transient)
/// - Returns: f32 scores vector
#[allow(clippy::too_many_arguments)]
pub fn dispatch(
    queue: &CommandQueue,
    bufs: &BufferCache,
    kernel: &KernelHandle,
    q4_data: &[u8],
    q8_x: &[i8],
    q8_scales: &[f32],
    num_rows: usize,
    hidden: usize,
) -> Vec<f32> {
    let buf_q4 = bufs.get_bytes(q4_data);
    let buf_q8 = bufs.transient_from_i8(q8_x);
    let buf_scales = bufs.transient_from_f32(q8_scales);
    let buf_out = bufs.output((num_rows * 4) as u64);

    let n_val = num_rows as u32;
    let k_val = hidden as u32;

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    encode(
        enc,
        kernel,
        &buf_q4,
        &buf_q8,
        &buf_scales,
        &buf_out,
        n_val,
        k_val,
        num_rows,
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    crate::buffers::read_buffer_f32(&buf_out, num_rows)
}

/// Encode a Q4 matvec dispatch into an existing command encoder.
/// Used by batched operations to chain multiple dispatches.
#[allow(clippy::too_many_arguments)]
pub fn encode(
    enc: &ComputeCommandEncoderRef,
    kernel: &KernelHandle,
    buf_q4: &Buffer,
    buf_q8: &Buffer,
    buf_scales: &Buffer,
    buf_out: &Buffer,
    n_val: u32,
    k_val: u32,
    num_rows: usize,
) {
    enc.set_compute_pipeline_state(&kernel.state);
    enc.set_buffer(0, Some(buf_q4), 0);
    enc.set_buffer(1, Some(buf_q8), 0);
    enc.set_buffer(2, Some(buf_scales), 0);
    enc.set_buffer(3, Some(buf_out), 0);
    enc.set_bytes(4, 4, &n_val as *const u32 as *const c_void);
    enc.set_bytes(5, 4, &k_val as *const u32 as *const c_void);

    let num_tgs = (num_rows as u64).div_ceil(kernel.rows_per_tg);
    enc.dispatch_thread_groups(
        MTLSize::new(num_tgs, 1, 1),
        MTLSize::new(kernel.threads_per_tg, 1, 1),
    );
}
