//! Q4 vector-matrix dispatch (scatter-accumulate).
//!
//! out[K] = activation[N] @ Q4[N, K]
//!
//! One thread per output element. GPU-hostile pattern but
//! parallelised across K output elements.

use metal::*;
use std::ffi::c_void;

use crate::buffers::BufferCache;

/// Dispatch Q4 vecmat on GPU.
pub fn dispatch(
    queue: &CommandQueue,
    bufs: &BufferCache,
    pipeline: &ComputePipelineState,
    activation: &[f32],
    q4_data: &[u8],
    intermediate: usize,
    hidden: usize,
) -> Vec<f32> {
    let buf_act = bufs.transient_from_f32(activation);
    let buf_q4 = bufs.get_bytes(q4_data);
    let buf_out = bufs.output((hidden * 4) as u64);

    let n_val = intermediate as u32;
    let k_val = hidden as u32;

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(&buf_act), 0);
    enc.set_buffer(1, Some(&buf_q4), 0);
    enc.set_buffer(2, Some(&buf_out), 0);
    enc.set_bytes(3, 4, &n_val as *const u32 as *const c_void);
    enc.set_bytes(4, 4, &k_val as *const u32 as *const c_void);

    let threads = MTLSize::new(hidden as u64, 1, 1);
    let tg = MTLSize::new(
        crate::kernels::DISPATCH_TG_MAX_THREADS.min(hidden as u64),
        1,
        1,
    );
    enc.dispatch_threads(threads, tg);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    crate::buffers::read_buffer_f32(&buf_out, hidden)
}
