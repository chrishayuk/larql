//! QK-norm: per-head RMSNorm with learned weight, applied to Q/K projections
//! before RoPE in attention.
//!
//! Formula (matches CPU `larql_inference::residual::rms_norm_heads_eps`):
//!   rms        = sqrt(mean(x_head²) + eps)
//!   out[h, d]  = (x[h, d] / rms) * (offset + weight[d])
//!
//! The weight vector is length `head_dim`, shared across heads. `offset` is
//! 0.0 on Gemma 4 and 1.0 on Gemma 2/3. Needed for Gemma 3/4 decode on the
//! Metal KV-cache attention path, which otherwise feeds un-normalised Q/K
//! into softmax and overflows to NaN.
//!
//! Grid: `(head_dim, num_heads, 1)`. Each thread writes one output element;
//! sum-of-squares is computed locally (head_dim ≤ 512 is cheap enough).

pub const SHADER: &str = r#"
// Dispatch layout:
//   threadgroups: (num_heads, 1, 1)
//   threads per tg: (min(head_dim, 512), 1, 1)
//
// All threads in a threadgroup serve a single head, so the
// `threadgroup_barrier` after the sum-of-squares reduction makes in-place
// (`x == out`) safe — every read of `x[base + i]` finishes before any write
// to `out[base + d]`.
kernel void qk_norm(
    device const float* x         [[buffer(0)]],
    device float*       out       [[buffer(1)]],
    device const float* weight    [[buffer(2)]],
    constant uint&      head_dim  [[buffer(3)]],
    constant uint&      num_heads [[buffer(4)]],
    constant float&     eps       [[buffer(5)]],
    constant float&     offset    [[buffer(6)]],
    uint h_idx [[threadgroup_position_in_grid]],
    uint tid   [[thread_position_in_threadgroup]],
    uint tg_w  [[threads_per_threadgroup]])
{
    if (h_idx >= num_heads) return;
    uint base = h_idx * head_dim;

    // Partial sum over this thread's strided subset of the head.
    float partial = 0.0f;
    for (uint i = tid; i < head_dim; i += tg_w) {
        float v = x[base + i];
        partial += v * v;
    }

    threadgroup float tg_partial[512];
    tg_partial[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction across the threadgroup.
    for (uint stride = tg_w / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            tg_partial[tid] += tg_partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sq_sum = tg_partial[0];
    float rms = sqrt(sq_sum / float(head_dim) + eps);

    // Once every thread has read x into the reduction above, writing to
    // out (= x in the aliased case) is safe.
    for (uint d = tid; d < head_dim; d += tg_w) {
        out[base + d] = (x[base + d] / rms) * (offset + weight[d]);
    }
}

// Fused Q+K norm — applies per-head RMSNorm to both Q and K in one dispatch.
// Grid: (num_q_heads + num_kv_heads, 1, 1). Each TG handles one head.
// Q heads (h_idx < num_q) use Q buffer and q_weight; K heads use K + k_weight.
// Saves one dispatch_thread_groups call per layer × 34 = 34 dispatches/token.
kernel void qk_norm_qk(
    device float*       Q          [[buffer(0)]],   // [num_q * head_dim] in-place
    device float*       K          [[buffer(1)]],   // [num_kv * head_dim] in-place
    device const float* q_weight   [[buffer(2)]],
    device const float* k_weight   [[buffer(3)]],
    constant uint&      head_dim   [[buffer(4)]],
    constant uint&      num_q      [[buffer(5)]],   // q heads count
    constant float&     eps        [[buffer(6)]],
    constant float&     offset     [[buffer(7)]],
    uint h_idx [[threadgroup_position_in_grid]],
    uint tid   [[thread_position_in_threadgroup]],
    uint tg_w  [[threads_per_threadgroup]])
{
    bool is_q = (h_idx < num_q);
    uint local_head = is_q ? h_idx : (h_idx - num_q);
    device float*       buf    = is_q ? Q : K;
    device const float* weight = is_q ? q_weight : k_weight;
    uint base = local_head * head_dim;

    float partial = 0.0f;
    for (uint i = tid; i < head_dim; i += tg_w) {
        float v = buf[base + i];
        partial += v * v;
    }

    threadgroup float tg_partial[512];
    tg_partial[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_w / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) tg_partial[tid] += tg_partial[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float rms = sqrt(tg_partial[0] / float(head_dim) + eps);

    for (uint d = tid; d < head_dim; d += tg_w) {
        buf[base + d] = (buf[base + d] / rms) * (offset + weight[d]);
    }
}
"#;

pub struct Kernel;
impl crate::kernels::ShaderKernel for Kernel {
    const KERNEL_NAME: &'static str = "qk_norm";
}

pub struct QkKernel;
impl crate::kernels::ShaderKernel for QkKernel {
    const KERNEL_NAME: &'static str = "qk_norm_qk";
}
