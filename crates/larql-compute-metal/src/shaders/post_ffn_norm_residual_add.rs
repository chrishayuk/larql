//! Fused **post-FFN norm + residual add** for the
//! `has_post_norms + post_ffn_norm` decode path (Gemma 3 / Gemma 4).
//!
//! Replaces the consecutive `rms_norm` + `residual_add` dispatches at
//! the end of each layer:
//!
//!   1. `rms_norm`: `normed_ffn = RMS(down_out) · post_ffn_norm_weight`
//!   2. `residual_add`: `new_h = h_post_attn + normed_ffn`
//!
//! into one single-TG kernel doing the RMS reduction once, then writing
//! the post-norm residual sum directly. Saves 1 dispatch/layer × 34 ≈
//! 0.24 ms/tok end-to-end (same fusion mechanic as `qk_norm_rope_fused`,
//! `residual_norm_store`, and `post_attn_residual_norm_store`).
//!
//! **Math** (per TG, per `len = hidden_size` elements):
//!
//! ```text
//! Phase A: sum_sq = Σ down_out[i]²
//!          rms = sqrt(sum_sq/len + eps);  inv_rms = 1/rms
//! Phase B: normed[i] = down_out[i] · inv_rms · (w[i] + offset)
//!          new_h[i]  = h_post_attn[i] + normed[i]
//! ```
//!
//! `threadgroup_barrier(mem_threadgroup)` between A and B (the inv_rms
//! has to be visible to all lanes before the per-element write).
//!
//! Numerical equivalence to the unfused chain is bit-equivalent: same
//! reduction tree (`Σ x²`), same `(x · inv_rms · (w + offset))`
//! expression for the normed output, same `h + normed` for the residual
//! add. Only difference is the `normed_ffn` intermediate is a register
//! (not a device-memory round-trip).

pub const SHADER: &str = r#"
kernel void post_ffn_norm_residual_add(
    device const float* down_out    [[buffer(0)]],   // pre-norm FFN output
    device const float* h_post_attn [[buffer(1)]],   // post-attention residual
    device const float* w           [[buffer(2)]],   // post_ffn_norm weight
    device float*       new_h       [[buffer(3)]],   // out: residual + normed
    constant uint&      len         [[buffer(4)]],
    constant float&     eps         [[buffer(5)]],
    constant float&     offset      [[buffer(6)]],
    uint tid   [[thread_index_in_threadgroup]],
    uint tg_sz [[threads_per_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]])
{
    // ── Phase A: RMS reduction over down_out[i] ──
    float partial = 0.0f;
    for (uint i = tid; i < len; i += tg_sz) {
        float v = down_out[i];
        partial += v * v;
    }
    float sg_sum = simd_sum(partial);
    threadgroup float tg_p[8];
    if (lane == 0) tg_p[sg_id] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum_sq = tg_p[0];
    uint n_sg = (tg_sz + 31u) / 32u;
    for (uint i = 1u; i < n_sg; i++) sum_sq += tg_p[i];
    float inv_rms = 1.0f / sqrt(sum_sq / float(len) + eps);

    // ── Phase B: per-element norm + residual add → new_h ──
    for (uint i = tid; i < len; i += tg_sz) {
        float normed = down_out[i] * inv_rms * (w[i] + offset);
        new_h[i] = h_post_attn[i] + normed;
    }
}
"#;

pub struct Kernel;
impl crate::kernels::ShaderKernel for Kernel {
    const KERNEL_NAME: &'static str = "post_ffn_norm_residual_add";
}
