//! QK-norm and V-norm — per-head RMS norm applied inside attention.
//!
//! All three variants use the same `qk_norm` shader (one TG per head,
//! cooperative simdgroup reduction). They differ only in:
//!   - Whose buffer they target (Q vs K vs V)
//!   - Which weight they multiply (learned q_norm / k_norm / all-ones)
//!   - The norm offset (Gemma 2/3 stores `weight - 1` → offset 1.0;
//!     Gemma 4 stores raw → offset 0.0; V-norm is parameter-free →
//!     offset 0.0, weight = 1.0)

use metal::{Buffer, ComputeCommandEncoderRef, ComputePipelineState, MTLSize};
use std::ffi::c_void;

/// Compute the threadgroup width for a `head_dim`-long cooperative reduction.
/// Rounds up to a power of two, capped at 512 (shader limit).
fn tg_width(head_dim: usize) -> u64 {
    let mut tg: u64 = 1;
    while (tg as usize) < head_dim && tg < 512 {
        tg <<= 1;
    }
    tg
}

/// Per-head RMS norm on Q and K (pre-RoPE, Gemma 3 / Gemma 4).
///
/// One shader dispatch per head per position. Writes back to the same Q/K
/// buffers (in-place). Returns `true` on success so the caller can tell
/// `fused_attention` to skip its internal QK-norm (otherwise double-norm).
/// Returns `false` if the pipeline or weights are absent — the caller
/// should then fall back to the shader's internal normalisation.
#[allow(clippy::too_many_arguments)]
pub fn encode_qk_norm(
    enc: &ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    q_buf: &Buffer,
    q_w_buf: &Buffer,
    k_buf: &Buffer,
    k_w_buf: &Buffer,
    seq_len: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    eps: f32,
    qk_norm_offset: f32,
) {
    let hd_val = head_dim as u32;
    let nq_val = num_q_heads as u32;
    let nkv_val = num_kv_heads as u32;
    let tg_w = tg_width(head_dim);

    for pos in 0..seq_len {
        let q_buf_off = (pos * num_q_heads * head_dim * 4) as u64;
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(q_buf), q_buf_off);
        enc.set_buffer(1, Some(q_buf), q_buf_off);
        enc.set_buffer(2, Some(q_w_buf), 0);
        enc.set_bytes(3, 4, &hd_val as *const u32 as *const c_void);
        enc.set_bytes(4, 4, &nq_val as *const u32 as *const c_void);
        enc.set_bytes(5, 4, &eps as *const f32 as *const c_void);
        enc.set_bytes(6, 4, &qk_norm_offset as *const f32 as *const c_void);
        enc.dispatch_thread_groups(
            MTLSize::new(num_q_heads as u64, 1, 1),
            MTLSize::new(tg_w, 1, 1),
        );

        let k_buf_off = (pos * num_kv_heads * head_dim * 4) as u64;
        enc.set_buffer(0, Some(k_buf), k_buf_off);
        enc.set_buffer(1, Some(k_buf), k_buf_off);
        enc.set_buffer(2, Some(k_w_buf), 0);
        enc.set_bytes(4, 4, &nkv_val as *const u32 as *const c_void);
        enc.dispatch_thread_groups(
            MTLSize::new(num_kv_heads as u64, 1, 1),
            MTLSize::new(tg_w, 1, 1),
        );
    }
}

/// Parameter-free per-head RMS norm on V (Gemma 4).
///
/// Weight is implicitly 1.0 (shader still takes a weight buffer — the
/// caller stages an all-ones vector of length `head_dim`). Offset is 0.
#[allow(clippy::too_many_arguments)]
pub fn encode_v_norm(
    enc: &ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    v_buf: &Buffer,
    ones_buf: &Buffer,
    seq_len: usize,
    num_kv_heads: usize,
    head_dim: usize,
    eps: f32,
) {
    let hd_val = head_dim as u32;
    let nkv_val = num_kv_heads as u32;
    let zero_off: f32 = 0.0;
    let tg_w = tg_width(head_dim);

    for pos in 0..seq_len {
        let v_buf_off = (pos * num_kv_heads * head_dim * 4) as u64;
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(v_buf), v_buf_off);
        enc.set_buffer(1, Some(v_buf), v_buf_off);
        enc.set_buffer(2, Some(ones_buf), 0);
        enc.set_bytes(3, 4, &hd_val as *const u32 as *const c_void);
        enc.set_bytes(4, 4, &nkv_val as *const u32 as *const c_void);
        enc.set_bytes(5, 4, &eps as *const f32 as *const c_void);
        enc.set_bytes(6, 4, &zero_off as *const f32 as *const c_void);
        enc.dispatch_thread_groups(
            MTLSize::new(num_kv_heads as u64, 1, 1),
            MTLSize::new(tg_w, 1, 1),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `tg_width` rounds up to a power of two, capped at 512.
    /// Production head_dims (256 / 512 in Gemma 3 / 4) hit the 512 cap on
    /// global layers; sliding layers hit 256. Powers of 2 stay exact.
    #[test]
    fn tg_width_rounds_up_to_power_of_two() {
        assert_eq!(tg_width(1), 1);
        assert_eq!(tg_width(2), 2);
        assert_eq!(tg_width(3), 4);
        assert_eq!(tg_width(64), 64);
        assert_eq!(tg_width(65), 128);
        assert_eq!(tg_width(128), 128);
        assert_eq!(tg_width(256), 256); // Gemma 3 sliding head_dim
        assert_eq!(tg_width(512), 512); // Gemma 4 global head_dim
    }

    /// Cap at 512 for head_dims larger than the shader supports. The
    /// kernel uses cooperative simdgroup reduction with a 512-thread
    /// upper bound; bigger head_dims would have to split across multiple
    /// dispatches (no current model needs it).
    #[test]
    fn tg_width_caps_at_512() {
        assert_eq!(tg_width(513), 512);
        assert_eq!(tg_width(1024), 512);
        assert_eq!(tg_width(8192), 512);
    }

    use crate::MetalBackend;

    fn backend() -> MetalBackend {
        MetalBackend::new().expect("Metal device available on test host")
    }

    /// `encode_qk_norm` runs a real `qk_norm` dispatch over a minimal
    /// 1-position Q/K shape.  After the command buffer completes the
    /// buffers should hold finite normalised values (no NaNs / Infs).
    #[test]
    fn encode_qk_norm_dispatch_completes_and_normalises_in_place() {
        let m = backend();
        let head_dim = 64usize;
        let num_q_heads = 2usize;
        let num_kv_heads = 2usize;
        let seq_len = 1usize;

        let q_src: Vec<f32> = (0..num_q_heads * head_dim)
            .map(|i| (i as f32) * 0.01)
            .collect();
        let k_src: Vec<f32> = (0..num_kv_heads * head_dim)
            .map(|i| ((i + 1) as f32) * 0.02)
            .collect();
        let q_w_src = vec![1.0f32; head_dim];
        let k_w_src = vec![1.0f32; head_dim];

        let q_buf = m.bufs().transient_from_f32(&q_src);
        let k_buf = m.bufs().transient_from_f32(&k_src);
        let q_w_buf = m.bufs().transient_from_f32(&q_w_src);
        let k_w_buf = m.bufs().transient_from_f32(&k_w_src);

        let cmd = m.queue().new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        encode_qk_norm(
            enc,
            &m.norms.qk_norm_pipeline,
            &q_buf,
            &q_w_buf,
            &k_buf,
            &k_w_buf,
            seq_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
            1e-6,
            0.0,
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let q_out = crate::buffers::read_buffer_f32(&q_buf, num_q_heads * head_dim);
        let k_out = crate::buffers::read_buffer_f32(&k_buf, num_kv_heads * head_dim);
        assert!(
            q_out.iter().all(|v| v.is_finite()),
            "Q has non-finite values after qk_norm"
        );
        assert!(
            k_out.iter().all(|v| v.is_finite()),
            "K has non-finite values after qk_norm"
        );
    }

    /// `encode_v_norm` runs the parameter-free per-head RMS dispatch
    /// over V.  Verify dispatch completes with finite output.
    #[test]
    fn encode_v_norm_dispatch_completes_with_finite_output() {
        let m = backend();
        let head_dim = 64usize;
        let num_kv_heads = 2usize;
        let seq_len = 1usize;

        let v_src: Vec<f32> = (0..num_kv_heads * head_dim)
            .map(|i| ((i + 3) as f32) * 0.005)
            .collect();
        let ones = vec![1.0f32; head_dim];
        let v_buf = m.bufs().transient_from_f32(&v_src);
        let ones_buf = m.bufs().transient_from_f32(&ones);

        // `encode_v_norm` reuses the `qk_norm` shader with all-ones
        // weight, NOT the standalone `v_norm` shader — see
        // `metal/ops/full_pipeline/dispatch.rs::encode_v_norm` callsite
        // which passes `qk_norm_pipe`.  Production callers exercise the
        // same path.
        let cmd = m.queue().new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        encode_v_norm(
            enc,
            &m.norms.qk_norm_pipeline,
            &v_buf,
            &ones_buf,
            seq_len,
            num_kv_heads,
            head_dim,
            1e-6,
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let v_out = crate::buffers::read_buffer_f32(&v_buf, num_kv_heads * head_dim);
        assert!(
            v_out.iter().all(|v| v.is_finite()),
            "V has non-finite values after v_norm"
        );
    }
}
