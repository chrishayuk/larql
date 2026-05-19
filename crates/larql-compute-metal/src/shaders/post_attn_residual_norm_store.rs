//! Fused **post-attention norm + residual + FFN-input norm + store** —
//! triple-fusion of the three adjacent dispatches that follow the
//! attention O-projection in the `has_post_norms` decode path
//! (Gemma 3 / Gemma 4):
//!
//! 1. `rms_norm`: `normed_o = RMS(o_out) · post_attn_norm_weight`
//!    (our `encode_rms_norm` dispatch)
//! 2. `residual + ffn_norm`: `ffn_norm_out = RMS(h + normed_o) · pre_ffn_buf`
//! 3. `residual_add`: `h_post_attn = h + normed_o`
//!
//! Original code path used 3 dispatches; an earlier 2026-05-01 fusion
//! collapsed steps 2+3 into `residual_norm_store`. This kernel collapses
//! all three, saving 1 more dispatch per layer × 34 = ~34/tok ≈
//! 0.24 ms/tok on Gemma 3 4B (matches the dispatch-count-reduction
//! path proven by `qk_norm_rope_fused` and `residual_norm_store`).
//!
//! **Math** (per TG, per `len = hidden_size` elements):
//!
//! ```text
//! Phase A (RMS of o):
//!   sum_o_sq = Σ o[i]²
//!   rms_o    = sqrt(sum_o_sq/len + eps)
//!   inv_rms_o = 1/rms_o
//!
//! Phase B (apply post_attn_norm and accumulate residual):
//!   normed_o[i] = o[i] · inv_rms_o · (w_post[i] + offset)
//!   h_sum[i]    = h[i] + normed_o[i]            // → h_post_attn output
//!
//! Phase C (RMS of h_sum, apply ffn norm):
//!   sum_h_sq = Σ h_sum[i]²
//!   rms_h    = sqrt(sum_h_sq/len + eps)
//!   ffn_norm_out[i] = h_sum[i] · (1/rms_h) · (w_ffn[i] + offset)
//! ```
//!
//! `threadgroup_barrier`s separate Phase A from B, and Phase B from C.
//! `h_sum` and `inv_rms_o` are temporaries kept in threadgroup memory
//! (one f32 each, plus a small reduction array).
//!
//! Numerical equivalence to the unfused chain:
//! - Phase A's RMS reduction is bit-equivalent to `rms_norm` (same
//!   `Σ x²` parallel reduction tree).
//! - Phase B's `normed_o[i] = o[i] · inv_rms_o · (w_post[i] + offset)`
//!   is the same expression `rms_norm` writes (`out[i] = (x[i] / rms)
//!   * (offset + w[i])`, after factoring `1/rms` to `inv_rms`).
//! - Phase C is bit-equivalent to `residual_norm_store`'s second
//!   half, with `b` replaced by the just-computed `normed_o` and the
//!   raw-sum output `h_post_attn` written directly from the in-loop
//!   `h_sum[i]`.
//!
//! Same arch_golden + decode_consistency parity contract as the
//! prior fusions.

pub const SHADER: &str = r#"
kernel void post_attn_residual_norm_store(
    device const float* h         [[buffer(0)]],   // pre-attn residual
    device const float* o         [[buffer(1)]],   // raw attn output
    device const float* w_post    [[buffer(2)]],   // post_attn_norm weight
    device const float* w_ffn     [[buffer(3)]],   // pre_ffn_norm weight
    device float*       ffn_norm  [[buffer(4)]],   // FFN input (normed h_sum)
    device float*       h_post    [[buffer(5)]],   // raw h + normed_o (residual)
    constant uint&      len       [[buffer(6)]],
    constant float&     eps       [[buffer(7)]],
    constant float&     offset    [[buffer(8)]],
    uint tid   [[thread_index_in_threadgroup]],
    uint tg_sz [[threads_per_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float tg_p[8];

    // ── Phase A: RMS reduction over o[i] ──
    float partial_o = 0.0f;
    for (uint i = tid; i < len; i += tg_sz) {
        float v = o[i];
        partial_o += v * v;
    }
    {
        float sg_sum = simd_sum(partial_o);
        if (lane == 0) tg_p[sg_id] = sg_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sum_sq_o = tg_p[0];
    uint n_sg = (tg_sz + 31u) / 32u;
    for (uint i = 1u; i < n_sg; i++) sum_sq_o += tg_p[i];
    float inv_rms_o = 1.0f / sqrt(sum_sq_o / float(len) + eps);

    // Use the second half of `tg_p` as a one-slot broadcast for inv_rms_o
    // back to all simdgroups (sg_id==0 has it correctly already, but
    // separate simdgroups all reduced from the same tg_p[] state, so
    // every lane just recomputed the same scalar — no broadcast needed).

    // ── Phase B: write normed_o into ffn_norm scratch (reuse) and
    // compute h_sum[i] = h[i] + normed_o[i], stash in h_post. ──
    // We don't have a separate scratch, so use `ffn_norm` as the
    // intermediate `normed_o` slot — it gets overwritten in Phase C.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < len; i += tg_sz) {
        float normed_o_i = o[i] * inv_rms_o * (w_post[i] + offset);
        float h_sum_i    = h[i] + normed_o_i;
        h_post[i]   = h_sum_i;
        ffn_norm[i] = h_sum_i;          // hold h_sum here for Phase C
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase C: RMS reduction over h_sum (in `ffn_norm` slot) ──
    float partial_h = 0.0f;
    for (uint i = tid; i < len; i += tg_sz) {
        float v = ffn_norm[i];
        partial_h += v * v;
    }
    {
        float sg_sum = simd_sum(partial_h);
        if (lane == 0) tg_p[sg_id] = sg_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sum_sq_h = tg_p[0];
    for (uint i = 1u; i < n_sg; i++) sum_sq_h += tg_p[i];
    float inv_rms_h = 1.0f / sqrt(sum_sq_h / float(len) + eps);

    // Final pass: write ffn_norm[i] = h_sum[i] · inv_rms_h · (w_ffn[i] + offset).
    // h_post[i] is already correct from Phase B.
    for (uint i = tid; i < len; i += tg_sz) {
        float h_sum_i = ffn_norm[i];
        ffn_norm[i] = h_sum_i * inv_rms_h * (w_ffn[i] + offset);
    }
}
"#;

pub struct Kernel;
impl crate::kernels::ShaderKernel for Kernel {
    const KERNEL_NAME: &'static str = "post_attn_residual_norm_store";
}
