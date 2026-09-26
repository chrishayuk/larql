//! SPLITK-1: decode attention with the span split ACROSS threadgroups
//! ("flash-decoding"), for geometries with too few query heads to fill the
//! device.
//!
//! `kv_attention_seqpar` runs one threadgroup per query head. At Gemma 3
//! 4B's 8 query heads that is 8 threadgroups on a 40-core M3 Max, and the
//! GQA-RE-READ bench (`examples/bench_attention_gqa.rs`) showed the kernel
//! is bound by per-threadgroup latency, not bytes: 4x the threadgroups or
//! 4x the distinct K/V moved the span slope by at most 1.2x, while one
//! threadgroup walks its span at ~17 GB/s. So the span is cut into
//! `n_chunks` contiguous chunks, each its own threadgroup, and merged.
//!
//! Two passes:
//!
//! 1. `kv_attention_splitk[_rows]` — grid `(num_q, n_chunks[, rows])`.
//!    Each threadgroup owns keys `[c0, c1)` of its head's span and writes
//!    the chunk's online-softmax state: local max `m`, `l = Σ exp(s - m)`,
//!    and the UNNORMALISED `o = Σ exp(s - m) · V`. Inside the chunk it is
//!    `kv_attention_seqpar`'s phases unchanged (`slices x head_dim`
//!    threads, fixed-order slice reduction), minus the normalisation.
//! 2. `kv_attention_splitk_merge` — grid `(num_q, rows)`, `head_dim`
//!    threads: `M = max(m_c [, sink])`, `L = Σ l_c e^(m_c - M) [+ e^(sink -
//!    M)]`, `out = Σ o_c e^(m_c - M) / L`, chunks summed in FIXED order
//!    0..n_chunks (determinism: see `kv_attention_seqpar`).
//!
//! Sinks enter only the merge — they are one extra logit per head, not a
//! key in any chunk. Softcap is per score, so it stays in pass 1.
//!
//! Partials are laid out `[rows, num_q, n_chunks, head_dim]` (o) and
//! `[rows, num_q, n_chunks, 2]` (m, l).
//!
//! Caller contract: chunk length `ceil(span / n_chunks) <= 1024`
//! (`tg_scores`); `tg_sz` a multiple of `head_dim`, `<= 1024`.
//!
//! Normalising after the V sum instead of before reassociates the
//! arithmetic, so results are not bitwise equal to seqpar; gated with a
//! tolerance in `tests/test_kernel_kv_attention_splitk.rs`.

pub const SHADER: &str = r#"
inline void kv_attention_splitk_body(
    device const float* Q,
    device const float* K_cache,
    device const float* V_cache,
    device float*       o_part,
    device float*       ml_part,
    uint                T,
    uint                head_dim,
    uint                num_q,
    uint                num_kv,
    float               scale,
    uint                window_size,
    uint                n_chunks,
    float               softcap,
    uint head,
    uint chunk,
    uint tid,
    uint tg_sz,
    uint lane,
    uint sg_id,
    threadgroup float*  tg_scores,
    threadgroup float*  tg_partial,
    threadgroup float*  tg_sg_vals)
{
    if (head >= num_q || chunk >= n_chunks) return;
    uint kv_head = head / (num_q / num_kv);
    device const float* q = Q + head * head_dim;
    uint t_start = (window_size > 0 && T > window_size) ? T - window_size : 0;
    uint span = T - t_start;
    uint chunk_len = (span + n_chunks - 1) / n_chunks;
    uint c0 = min(t_start + chunk * chunk_len, T);
    uint c1 = min(c0 + chunk_len, T);

    // ---- Phase 1: scores + chunk max ----
    float local_max = -1e30f;
    for (uint t = c0 + tid; t < c1; t += tg_sz) {
        device const float* k = K_cache + t * num_kv * head_dim + kv_head * head_dim;
        float dot = 0.0f;
        for (uint d = 0; d + 3 < head_dim; d += 4) {
            dot += q[d]*k[d] + q[d+1]*k[d+1] + q[d+2]*k[d+2] + q[d+3]*k[d+3];
        }
        for (uint d = (head_dim & ~3u); d < head_dim; d++) dot += q[d] * k[d];
        dot *= scale;
        if (softcap > 0.0f) {
            dot = softcap * tanh(clamp(dot / softcap, -15.0f, 15.0f));
        }
        tg_scores[t - c0] = dot;
        local_max = max(local_max, dot);
    }

    float sg_max = simd_max(local_max);
    if (lane == 0) tg_sg_vals[sg_id] = sg_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m = tg_sg_vals[0];
    uint n_sg = (tg_sz + 31) / 32;
    for (uint i = 1; i < n_sg; i++) m = max(m, tg_sg_vals[i]);

    // ---- Phase 2: unnormalised weights + chunk exp-sum ----
    float local_sum = 0.0f;
    for (uint t = c0 + tid; t < c1; t += tg_sz) {
        float w = exp(tg_scores[t - c0] - m);
        tg_scores[t - c0] = w;
        local_sum += w;
    }

    float sg_sum = simd_sum(local_sum);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) tg_sg_vals[sg_id] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float l = tg_sg_vals[0];
    for (uint i = 1; i < n_sg; i++) l += tg_sg_vals[i];

    // ---- Phase 3: sequence-parallel weighted-V over the chunk ----
    uint n_slices = tg_sz / head_dim;
    if (n_slices == 0u) n_slices = 1u;
    uint active = n_slices * head_dim;
    uint d = tid % head_dim;
    uint slice = tid / head_dim;

    if (tid < active) {
        float acc = 0.0f;
        for (uint t = c0 + slice; t < c1; t += n_slices) {
            acc += tg_scores[t - c0]
                 * V_cache[t * num_kv * head_dim + kv_head * head_dim + d];
        }
        tg_partial[slice * head_dim + d] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint cell = head * n_chunks + chunk;
    if (tid < head_dim) {
        // Fixed slice order.
        float sum = tg_partial[tid];
        for (uint s = 1u; s < n_slices; s++) {
            sum += tg_partial[s * head_dim + tid];
        }
        o_part[cell * head_dim + tid] = sum;
    }
    if (tid == 0) {
        ml_part[cell * 2]     = m;
        ml_part[cell * 2 + 1] = l;
    }
}

kernel void kv_attention_splitk(
    device const float* Q        [[buffer(0)]],
    device const float* K_cache  [[buffer(1)]],
    device const float* V_cache  [[buffer(2)]],
    device float*       o_part   [[buffer(3)]],
    constant uint&      T        [[buffer(4)]],
    constant uint&      head_dim [[buffer(5)]],
    constant uint&      num_q    [[buffer(6)]],
    constant uint&      num_kv   [[buffer(7)]],
    constant float&     scale    [[buffer(8)]],
    constant uint&      window_size [[buffer(9)]],
    device float*       ml_part  [[buffer(10)]],
    constant uint&      n_chunks [[buffer(11)]],
    constant float&     softcap  [[buffer(12)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint  tid    [[thread_index_in_threadgroup]],
    uint2 tg_sz2 [[threads_per_threadgroup]],
    uint  lane   [[thread_index_in_simdgroup]],
    uint  sg_id  [[simdgroup_index_in_threadgroup]])
{
    threadgroup float tg_scores[1024];
    threadgroup float tg_partial[1024];
    threadgroup float tg_sg_vals[32];
    kv_attention_splitk_body(Q, K_cache, V_cache, o_part, ml_part, T, head_dim, num_q, num_kv,
                             scale, window_size, n_chunks, softcap, tg_pos.x, tg_pos.y,
                             tid, tg_sz2.x, lane, sg_id, tg_scores, tg_partial, tg_sg_vals);
}

// VERIFY-N: grid (num_q, n_chunks, rows); row r at cache length T + r,
// Q row r of `[rows, num_q, head_dim]`, partial row r of the layout above.
kernel void kv_attention_splitk_rows(
    device const float* Q        [[buffer(0)]],
    device const float* K_cache  [[buffer(1)]],
    device const float* V_cache  [[buffer(2)]],
    device float*       o_part   [[buffer(3)]],
    constant uint&      T        [[buffer(4)]],
    constant uint&      head_dim [[buffer(5)]],
    constant uint&      num_q    [[buffer(6)]],
    constant uint&      num_kv   [[buffer(7)]],
    constant float&     scale    [[buffer(8)]],
    constant uint&      window_size [[buffer(9)]],
    device float*       ml_part  [[buffer(10)]],
    constant uint&      n_chunks [[buffer(11)]],
    constant float&     softcap  [[buffer(12)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint  tid    [[thread_index_in_threadgroup]],
    uint3 tg_sz3 [[threads_per_threadgroup]],
    uint  lane   [[thread_index_in_simdgroup]],
    uint  sg_id  [[simdgroup_index_in_threadgroup]])
{
    threadgroup float tg_scores[1024];
    threadgroup float tg_partial[1024];
    threadgroup float tg_sg_vals[32];
    const ulong q_off = (ulong)tg_pos.z * num_q * head_dim;
    const ulong cells = (ulong)tg_pos.z * num_q * n_chunks;
    kv_attention_splitk_body(Q + q_off, K_cache, V_cache, o_part + cells * head_dim,
                             ml_part + cells * 2, T + tg_pos.z, head_dim, num_q, num_kv,
                             scale, window_size, n_chunks, softcap, tg_pos.x, tg_pos.y,
                             tid, tg_sz3.x, lane, sg_id, tg_scores, tg_partial, tg_sg_vals);
}

// Pass 2: grid (num_q, rows), head_dim threads (or fewer, strided).
kernel void kv_attention_splitk_merge(
    device const float* o_part   [[buffer(0)]],
    device const float* ml_part  [[buffer(1)]],
    device float*       out      [[buffer(2)]],
    constant uint&      head_dim [[buffer(3)]],
    constant uint&      num_q    [[buffer(4)]],
    constant uint&      n_chunks [[buffer(5)]],
    constant float*     sinks    [[buffer(6)]],
    constant uint&      has_sinks [[buffer(7)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint  tid    [[thread_index_in_threadgroup]],
    uint2 tg_sz2 [[threads_per_threadgroup]])
{
    uint head = tg_pos.x;
    if (head >= num_q) return;
    uint base = (tg_pos.y * num_q + head) * n_chunks;

    float M = -1e30f;
    for (uint c = 0; c < n_chunks; c++) M = max(M, ml_part[(base + c) * 2]);
    if (has_sinks != 0u) M = max(M, sinks[head]);
    float L = 0.0f;
    for (uint c = 0; c < n_chunks; c++) {
        L += ml_part[(base + c) * 2 + 1] * exp(ml_part[(base + c) * 2] - M);
    }
    if (has_sinks != 0u) L += exp(sinks[head] - M);
    float inv = 1.0f / L;

    device float* o = out + (tg_pos.y * num_q + head) * head_dim;
    for (uint d = tid; d < head_dim; d += tg_sz2.x) {
        float acc = 0.0f;
        for (uint c = 0; c < n_chunks; c++) {
            acc += o_part[(base + c) * head_dim + d] * exp(ml_part[(base + c) * 2] - M);
        }
        o[d] = acc * inv;
    }
}
"#;

/// Pass 1, one position: grid `(num_q, n_chunks)`.
pub struct SplitKKernel;
impl crate::kernels::ShaderKernel for SplitKKernel {
    const KERNEL_NAME: &'static str = "kv_attention_splitk";
}

/// Pass 1, a verify block: grid `(num_q, n_chunks, rows)`.
pub struct SplitKRowsKernel;
impl crate::kernels::ShaderKernel for SplitKRowsKernel {
    const KERNEL_NAME: &'static str = "kv_attention_splitk_rows";
}

/// Pass 2: grid `(num_q, rows)`.
pub struct SplitKMergeKernel;
impl crate::kernels::ShaderKernel for SplitKMergeKernel {
    const KERNEL_NAME: &'static str = "kv_attention_splitk_merge";
}
