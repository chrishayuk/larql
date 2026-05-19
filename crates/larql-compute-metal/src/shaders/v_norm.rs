//! Parameter-free V-norm: RMSNorm without learned weights.
//!
//! out = x / sqrt(mean(x²) + eps)
//!
//! Applied to V states before attention in Gemma 4.
//! Unlike regular RMSNorm, there is no weight multiplication —
//! this is purely normalization.

pub const SHADER: &str = r#"
// V-norm: parameter-free RMSNorm on a single vector.
// Grid: (len, 1, 1). Each thread handles one element.
kernel void v_norm(
    device const float* x   [[buffer(0)]],
    device float*       out [[buffer(1)]],
    constant uint&      len [[buffer(2)]],
    constant float&     eps [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= len) return;

    float sum_sq = 0.0f;
    for (uint i = 0; i < len; i++) {
        sum_sq += x[i] * x[i];
    }
    float rms = 1.0f / sqrt(sum_sq / float(len) + eps);
    out[tid] = x[tid] * rms;
}
// Batched V-norm: apply to all KV heads in one dispatch.
// x = [num_heads * head_dim] contiguous.
// Grid: (head_dim, num_heads, 1)
// Threadgroup: (min(head_dim, 256), 1, 1) — one TG per head.
//
// Correctness invariant: when `x` and `out` alias the same buffer
// (which the decode path does for v_norm), each thread's `sum_sq`
// computation must finish reading every `x[base_idx + i]` before any
// thread starts writing. The previous version had every thread
// independently re-compute the full sum_sq, then write its element —
// late-reading threads saw early-writing threads' outputs and produced
// drifted results (visible end-to-end as cos≈0.997 at L0 of Gemma 4
// 31B's KV-cached decode path). Fix: cooperative reduction in
// threadgroup memory with an explicit barrier between read and write
// phases. Mirrors the `qk_norm` shader's structure.
kernel void v_norm_batched(
    device const float* x        [[buffer(0)]],
    device float*       out      [[buffer(1)]],
    constant uint&      head_dim [[buffer(2)]],
    constant float&     eps      [[buffer(3)]],
    constant uint&      num_heads[[buffer(4)]],
    uint  h_idx [[threadgroup_position_in_grid]],
    uint  tid   [[thread_position_in_threadgroup]],
    uint  tg_w  [[threads_per_threadgroup]])
{
    if (h_idx >= num_heads) return;
    uint base_idx = h_idx * head_dim;

    // Phase 1 — partial sum-of-squares from each thread's strided
    // subset of the head. Reads `x` before any thread writes `out`.
    float partial = 0.0f;
    for (uint i = tid; i < head_dim; i += tg_w) {
        float v = x[base_idx + i];
        partial += v * v;
    }

    threadgroup float tg_partial[512];
    tg_partial[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction across the threadgroup.
    for (uint stride = tg_w / 2; stride > 0; stride >>= 1) {
        if (tid < stride) tg_partial[tid] += tg_partial[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sq_sum = tg_partial[0];
    float rms = 1.0f / sqrt(sq_sum / float(head_dim) + eps);

    // Phase 2 — every read of `x` from phase 1 has finished; safe to
    // write `out` (= `x` in the aliased case).
    for (uint d = tid; d < head_dim; d += tg_w) {
        out[base_idx + d] = x[base_idx + d] * rms;
    }
}
"#;

pub struct Kernel;
impl crate::kernels::ShaderKernel for Kernel {
    const KERNEL_NAME: &'static str = "v_norm";
}

pub struct BatchedKernel;
impl crate::kernels::ShaderKernel for BatchedKernel {
    const KERNEL_NAME: &'static str = "v_norm_batched";
}
