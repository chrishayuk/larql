//! KV-cached attention for token generation (seq=1 decode).
//!
//! Two attention kernels:
//!   - kv_attention: T/window span <= 1024, small threadgroup scores array
//!     (4KB), high occupancy
//!   - kv_attention_long: T/window span <= 4096, larger score array (16KB)
//!     used by Gemma 4 global-attention layers after the cache passes 1024
//!
//! Both use simd_max/simd_sum for reductions and float4 Q·K dot products.

pub const SHADER: &str = r#"
// Fast decode attention — small threadgroup memory for high occupancy.
// 4KB scores = max 1024 tokens. Enough for decode (grows by 1 per step).
kernel void kv_attention(
    device const float* Q       [[buffer(0)]],
    device const float* K_cache [[buffer(1)]],
    device const float* V_cache [[buffer(2)]],
    device float*       out     [[buffer(3)]],
    constant uint&      T       [[buffer(4)]],
    constant uint&      head_dim[[buffer(5)]],
    constant uint&      num_q   [[buffer(6)]],
    constant uint&      num_kv  [[buffer(7)]],
    constant float&     scale   [[buffer(8)]],
    constant uint&      window_size [[buffer(9)]],
    uint tg_id  [[threadgroup_position_in_grid]],
    uint tid    [[thread_index_in_threadgroup]],
    uint tg_sz  [[threads_per_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]],
    uint sg_id  [[simdgroup_index_in_threadgroup]])
{
    uint head = tg_id;
    if (head >= num_q) return;
    uint kv_head = head / (num_q / num_kv);

    device const float* q = Q + head * head_dim;

    uint t_start = (window_size > 0 && T > window_size) ? T - window_size : 0;

    // Small threadgroup scores — 4KB = max 1024 tokens
    threadgroup float tg_scores[1024];

    // Phase 1: Q·K dot products + max
    float local_max = -1e30f;
    for (uint t = t_start + tid; t < T; t += tg_sz) {
        device const float* k = K_cache + t * num_kv * head_dim + kv_head * head_dim;
        float dot = 0.0f;
        for (uint d = 0; d + 3 < head_dim; d += 4) {
            dot += q[d]*k[d] + q[d+1]*k[d+1] + q[d+2]*k[d+2] + q[d+3]*k[d+3];
        }
        for (uint d = (head_dim & ~3u); d < head_dim; d++) dot += q[d] * k[d];
        dot *= scale;
        tg_scores[t - t_start] = dot;
        local_max = max(local_max, dot);
    }

    float sg_max = simd_max(local_max);
    threadgroup float tg_sg_vals[8];
    if (lane == 0) tg_sg_vals[sg_id] = sg_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float global_max = tg_sg_vals[0];
    uint n_sg = (tg_sz + 31) / 32;
    for (uint i = 1; i < n_sg; i++) global_max = max(global_max, tg_sg_vals[i]);

    // Phase 2: softmax
    float local_sum = 0.0f;
    for (uint t = t_start + tid; t < T; t += tg_sz) {
        float w = exp(tg_scores[t - t_start] - global_max);
        tg_scores[t - t_start] = w;
        local_sum += w;
    }

    float sg_sum = simd_sum(local_sum);
    if (lane == 0) tg_sg_vals[sg_id] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float global_sum = tg_sg_vals[0];
    for (uint i = 1; i < n_sg; i++) global_sum += tg_sg_vals[i];
    float inv_sum = 1.0f / global_sum;

    // Normalize
    for (uint t = t_start + tid; t < T; t += tg_sz) {
        tg_scores[t - t_start] *= inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: weighted V sum
    device float* out_head = out + head * head_dim;
    for (uint d = tid; d < head_dim; d += tg_sz) {
        float acc = 0.0f;
        for (uint t = t_start; t < T; t++) {
            acc += tg_scores[t - t_start] * V_cache[t * num_kv * head_dim + kv_head * head_dim + d];
        }
        out_head[d] = acc;
    }
}

kernel void kv_attention_long(
    device const float* Q       [[buffer(0)]],
    device const float* K_cache [[buffer(1)]],
    device const float* V_cache [[buffer(2)]],
    device float*       out     [[buffer(3)]],
    constant uint&      T       [[buffer(4)]],
    constant uint&      head_dim[[buffer(5)]],
    constant uint&      num_q   [[buffer(6)]],
    constant uint&      num_kv  [[buffer(7)]],
    constant float&     scale   [[buffer(8)]],
    constant uint&      window_size [[buffer(9)]],
    uint tg_id  [[threadgroup_position_in_grid]],
    uint tid    [[thread_index_in_threadgroup]],
    uint tg_sz  [[threads_per_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]],
    uint sg_id  [[simdgroup_index_in_threadgroup]])
{
    uint head = tg_id;
    if (head >= num_q) return;
    uint kv_head = head / (num_q / num_kv);

    device const float* q = Q + head * head_dim;

    uint t_start = (window_size > 0 && T > window_size) ? T - window_size : 0;

    // 16KB scores buffer. Matches DEFAULT_KV_CACHE_MAX_SEQ = 4096.
    threadgroup float tg_scores[4096];

    float local_max = -1e30f;
    for (uint t = t_start + tid; t < T; t += tg_sz) {
        device const float* k = K_cache + t * num_kv * head_dim + kv_head * head_dim;
        float dot = 0.0f;
        for (uint d = 0; d + 3 < head_dim; d += 4) {
            dot += q[d]*k[d] + q[d+1]*k[d+1] + q[d+2]*k[d+2] + q[d+3]*k[d+3];
        }
        for (uint d = (head_dim & ~3u); d < head_dim; d++) dot += q[d] * k[d];
        dot *= scale;
        tg_scores[t - t_start] = dot;
        local_max = max(local_max, dot);
    }

    float sg_max = simd_max(local_max);
    threadgroup float tg_sg_vals[8];
    if (lane == 0) tg_sg_vals[sg_id] = sg_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float global_max = tg_sg_vals[0];
    uint n_sg = (tg_sz + 31) / 32;
    for (uint i = 1; i < n_sg; i++) global_max = max(global_max, tg_sg_vals[i]);

    float local_sum = 0.0f;
    for (uint t = t_start + tid; t < T; t += tg_sz) {
        float w = exp(tg_scores[t - t_start] - global_max);
        tg_scores[t - t_start] = w;
        local_sum += w;
    }

    float sg_sum = simd_sum(local_sum);
    if (lane == 0) tg_sg_vals[sg_id] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float global_sum = tg_sg_vals[0];
    for (uint i = 1; i < n_sg; i++) global_sum += tg_sg_vals[i];
    float inv_sum = 1.0f / global_sum;

    for (uint t = t_start + tid; t < T; t += tg_sz) {
        tg_scores[t - t_start] *= inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float* out_head = out + head * head_dim;
    for (uint d = tid; d < head_dim; d += tg_sz) {
        float acc = 0.0f;
        for (uint t = t_start; t < T; t++) {
            acc += tg_scores[t - t_start] * V_cache[t * num_kv * head_dim + kv_head * head_dim + d];
        }
        out_head[d] = acc;
    }
}

kernel void kv_cache_append(
    device const float* new_k    [[buffer(0)]],
    device const float* new_v    [[buffer(1)]],
    device float*       K_cache  [[buffer(2)]],
    device float*       V_cache  [[buffer(3)]],
    constant uint&      pos      [[buffer(4)]],
    constant uint&      num_kv   [[buffer(5)]],
    constant uint&      head_dim [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    uint total = num_kv * head_dim;
    if (tid >= total) return;
    K_cache[pos * total + tid] = new_k[tid];
    V_cache[pos * total + tid] = new_v[tid];
}
"#;

pub struct AttendKernel;
impl crate::kernels::ShaderKernel for AttendKernel {
    const KERNEL_NAME: &'static str = "kv_attention";
}

pub struct AttendLongKernel;
impl crate::kernels::ShaderKernel for AttendLongKernel {
    const KERNEL_NAME: &'static str = "kv_attention_long";
}

pub struct AppendKernel;
impl crate::kernels::ShaderKernel for AppendKernel {
    const KERNEL_NAME: &'static str = "kv_cache_append";
}
