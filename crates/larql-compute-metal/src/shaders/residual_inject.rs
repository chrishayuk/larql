//! Cached residual injection: copy a precomputed residual into the pipeline.
//!
//! When skipping cached layers (L0-12), the GPU needs the cached residual
//! as input to the first computed layer. This shader copies it within
//! the command buffer — no CPU-GPU sync needed.
//!
//! Also supports residual add: out = a + b (for skip connections).

pub const SHADER: &str = r#"
// Simple buffer copy — inject cached residual into pipeline.
kernel void residual_copy(
    device const float* src [[buffer(0)]],
    device float*       dst [[buffer(1)]],
    constant uint&      len [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= len) return;
    dst[tid] = src[tid];
}

// Residual add: out = a + b_scale * b (skip connection).
//
// `b_scale` defaults to 1.0 for every model except Granite-family
// architectures, which set `residual_multiplier` (0.22 on Granite 4.1
// 3B/8B, 0.175 on Granite 4.1 30B μP). The Rust wrapper
// `encode_residual_add` always binds buffer(4); callers that don't
// need scaling pass 1.0 and the multiplication is a no-op (the f32
// hardware multiply is single-cycle).
kernel void residual_add(
    device const float* a       [[buffer(0)]],
    device const float* b       [[buffer(1)]],
    device float*       out     [[buffer(2)]],
    constant uint&      len     [[buffer(3)]],
    constant float&     b_scale [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= len) return;
    out[tid] = a[tid] + b_scale * b[tid];
}

// Scale vector: out = input * scalar (per-layer scalar multiplier, Gemma 4).
kernel void scale_vector(
    device const float* input  [[buffer(0)]],
    device float*       out    [[buffer(1)]],
    constant uint&      len    [[buffer(2)]],
    constant float&     scalar [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= len) return;
    out[tid] = input[tid] * scalar;
}

// RMS norm: out = x * (weight + offset) / sqrt(mean(x²) + eps)
// Uses cooperative SIMD reduction — O(N) reads instead of O(N²).
// MUST be dispatched as ONE threadgroup: dispatch_thread_groups(1, tg_size).
kernel void rms_norm(
    device const float* x      [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float*       out    [[buffer(2)]],
    constant uint&      len    [[buffer(3)]],
    constant float&     eps    [[buffer(4)]],
    constant float&     offset [[buffer(5)]],
    uint tid    [[thread_index_in_threadgroup]],
    uint tg_sz  [[threads_per_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]],
    uint sg_id  [[simdgroup_index_in_threadgroup]])
{
    // Cooperative sum_sq: each thread sums a stripe, then SIMD reduce
    float partial = 0.0f;
    for (uint i = tid; i < len; i += tg_sz) {
        partial += x[i] * x[i];
    }
    float sg_sum = simd_sum(partial);
    threadgroup float tg_p[8];
    if (lane == 0) tg_p[sg_id] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum_sq = tg_p[0];
    uint n_sg = (tg_sz + 31) / 32;
    for (uint i = 1; i < n_sg; i++) sum_sq += tg_p[i];

    float rms = 1.0f / sqrt(sum_sq / float(len) + eps);

    // Write all output elements (loop for len > tg_sz)
    for (uint i = tid; i < len; i += tg_sz) {
        out[i] = x[i] * (weight[i] + offset) * rms;
    }
}
"#;

pub struct RmsNormKernel;
impl crate::kernels::ShaderKernel for RmsNormKernel {
    const KERNEL_NAME: &'static str = "rms_norm";
}

pub struct ResidualAddKernel;
impl crate::kernels::ShaderKernel for ResidualAddKernel {
    const KERNEL_NAME: &'static str = "residual_add";
}

pub struct ScaleVectorKernel;
impl crate::kernels::ShaderKernel for ScaleVectorKernel {
    const KERNEL_NAME: &'static str = "scale_vector";
}
