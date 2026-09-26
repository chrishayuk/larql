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
// The decode attention body, shared by `kv_attention` (one position) and
// `kv_attention_rows` (a verify block's positions as grid.y) so the two are
// the same arithmetic by construction. Threadgroup arrays are the caller's:
// MSL allows them only at kernel scope.
inline void kv_attention_body(
    device const float* Q,
    device const float* K_cache,
    device const float* V_cache,
    device float*       out,
    uint                T,
    uint                head_dim,
    uint                num_q,
    uint                num_kv,
    float               scale,
    uint                window_size,
    constant float*     sinks,
    uint                has_sinks,
    float               softcap,
    uint tg_id,
    uint tid,
    uint tg_sz,
    uint lane,
    uint sg_id,
    threadgroup float*  tg_scores,
    threadgroup float*  tg_sg_vals)
{
    uint head = tg_id;
    if (head >= num_q) return;
    uint kv_head = head / (num_q / num_kv);

    device const float* q = Q + head * head_dim;

    uint t_start = (window_size > 0 && T > window_size) ? T - window_size : 0;

    // Small threadgroup scores — 4KB = max 1024 tokens (caller-owned).

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
        // Gemma-2-style logit softcapping. Clamp the tanh argument like
        // the GELU kernels do: Apple's tanh NaNs past |y| ~ 44.
        if (softcap > 0.0f) {
            dot = softcap * tanh(clamp(dot / softcap, -15.0f, 15.0f));
        }
        tg_scores[t - t_start] = dot;
        local_max = max(local_max, dot);
    }

    float sg_max = simd_max(local_max);
    if (lane == 0) tg_sg_vals[sg_id] = sg_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float global_max = tg_sg_vals[0];
    uint n_sg = (tg_sz + 31) / 32;
    for (uint i = 1; i < n_sg; i++) global_max = max(global_max, tg_sg_vals[i]);
    // The sink competes in the softmax, so it must join the max or
    // exp(sink - max) overflows when the sink dominates. Mirrors
    // kv_append_attend_fused — these kernels are its fallback, and a
    // fallback must not change the softmax denominator (audit F7).
    if (has_sinks != 0u) global_max = max(global_max, sinks[head]);

    // Phase 2: softmax
    float local_sum = 0.0f;
    for (uint t = t_start + tid; t < T; t += tg_sz) {
        float w = exp(tg_scores[t - t_start] - global_max);
        tg_scores[t - t_start] = w;
        local_sum += w;
    }

    float sg_sum = simd_sum(local_sum);
    // tg_sg_vals is reused from the max-reduction above. Without this barrier
    // a fast simdgroup can overwrite its slot with sg_sum while a slow
    // simdgroup is still reading that slot for global_max — corrupting the max
    // non-deterministically per dispatch. That drift is invisible to sampling
    // but desyncs the Shannon arithmetic coder (see docs/replay/
    // shannon-transformers-the-same.md). Fence the reuse.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) tg_sg_vals[sg_id] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float global_sum = tg_sg_vals[0];
    for (uint i = 1; i < n_sg; i++) global_sum += tg_sg_vals[i];
    // Denominator only: the sink has no output slot, so the emitted
    // weights deliberately sum to less than one.
    if (has_sinks != 0u) global_sum += exp(sinks[head] - global_max);
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
    constant float*     sinks   [[buffer(10)]],  // per-Q-head sink logits
    constant uint&      has_sinks [[buffer(11)]], // 0 = slot is a placeholder
    constant float&     softcap [[buffer(12)]],  // 0.0 = disabled
    uint tg_id  [[threadgroup_position_in_grid]],
    uint tid    [[thread_index_in_threadgroup]],
    uint tg_sz  [[threads_per_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]],
    uint sg_id  [[simdgroup_index_in_threadgroup]])
{
    threadgroup float tg_scores[1024];
    threadgroup float tg_sg_vals[8];
    kv_attention_body(Q, K_cache, V_cache, out, T, head_dim, num_q, num_kv, scale,
                      window_size, sinks, has_sinks, softcap,
                      tg_id, tid, tg_sz, lane, sg_id, tg_scores, tg_sg_vals);
}

// VERIFY-N: `kv_attention` over a block of `rows` consecutive positions in
// ONE dispatch. Grid (num_q, rows): row r is position base + r, so its
// cache length is T + r (T is row 0's), and its Q and output are row r of
// `[rows, num_q, head_dim]`. Each row runs the shared body unchanged — the
// same arithmetic as a `kv_attention` dispatch at that position.
kernel void kv_attention_rows(
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
    constant float*     sinks   [[buffer(10)]],
    constant uint&      has_sinks [[buffer(11)]],
    constant float&     softcap [[buffer(12)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint tid    [[thread_index_in_threadgroup]],
    uint2 tg_sz2 [[threads_per_threadgroup]],   // uint2: must match the 2-D grid's width
    uint lane   [[thread_index_in_simdgroup]],
    uint sg_id  [[simdgroup_index_in_threadgroup]])
{
    threadgroup float tg_scores[1024];
    threadgroup float tg_sg_vals[8];
    const ulong row_off = (ulong)tg_pos.y * num_q * head_dim;
    kv_attention_body(Q + row_off, K_cache, V_cache, out + row_off, T + tg_pos.y,
                      head_dim, num_q, num_kv, scale, window_size, sinks, has_sinks,
                      softcap, tg_pos.x, tid, tg_sz2.x, lane, sg_id, tg_scores, tg_sg_vals);
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
    constant float*     sinks   [[buffer(10)]],  // per-Q-head sink logits
    constant uint&      has_sinks [[buffer(11)]], // 0 = slot is a placeholder
    constant float&     softcap [[buffer(12)]],  // 0.0 = disabled
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
        // Gemma-2-style logit softcapping. Clamp the tanh argument like
        // the GELU kernels do: Apple's tanh NaNs past |y| ~ 44.
        if (softcap > 0.0f) {
            dot = softcap * tanh(clamp(dot / softcap, -15.0f, 15.0f));
        }
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
    // The sink competes in the softmax, so it must join the max or
    // exp(sink - max) overflows when the sink dominates. Mirrors
    // kv_append_attend_fused — these kernels are its fallback, and a
    // fallback must not change the softmax denominator (audit F7).
    if (has_sinks != 0u) global_max = max(global_max, sinks[head]);

    float local_sum = 0.0f;
    for (uint t = t_start + tid; t < T; t += tg_sz) {
        float w = exp(tg_scores[t - t_start] - global_max);
        tg_scores[t - t_start] = w;
        local_sum += w;
    }

    float sg_sum = simd_sum(local_sum);
    // tg_sg_vals is reused from the max-reduction above. Without this barrier
    // a fast simdgroup can overwrite its slot with sg_sum while a slow
    // simdgroup is still reading that slot for global_max — corrupting the max
    // non-deterministically per dispatch. That drift is invisible to sampling
    // but desyncs the Shannon arithmetic coder (see docs/replay/
    // shannon-transformers-the-same.md). Fence the reuse.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) tg_sg_vals[sg_id] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float global_sum = tg_sg_vals[0];
    for (uint i = 1; i < n_sg; i++) global_sum += tg_sg_vals[i];
    // Denominator only: the sink has no output slot, so the emitted
    // weights deliberately sum to less than one.
    if (has_sinks != 0u) global_sum += exp(sinks[head] - global_max);
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


// ---------------------------------------------------------------------
// KV-B1: sequence-parallel weighted-V accumulation.
//
// `kv_attention`'s phase 3 is the 86%-of-long-span cost, and it is serial
// in exactly the axis that grows:
//
//     for (d = tid; d < head_dim; d += tg_sz)   // only head_dim threads
//         for (t = t_start; t < T; t++)          // whole span, serially
//
// Widening the threadgroup cannot help it — the extra threads leave the
// outer loop immediately (measured: 4x threads bought 1.06-1.09x).
//
// Here the sequence is split across `n_slices = tg_sz / head_dim` slices:
//
//     d     = tid % head_dim
//     slice = tid / head_dim
//     slice s walks t = t_start + s, +n_slices, ...
//
// For any single `t` the 64 `d`-threads of a slice still read the whole
// contiguous head_dim V row, so sequence parallelism is added WITHOUT
// breaking the naturally coalesced head-dimension load.
//
// No online-softmax merge is needed: `tg_scores` is already normalised and
// shared across the threadgroup, so the slices reduce by plain summation.
// (That merge becomes necessary only if phases 1-2 are ever tiled ACROSS
// threadgroups, where each tile would own its own max/sum state.)
//
// The final reduction runs in FIXED slice order 0..n_slices. Determinism is
// not optional here: the `tg_sg_vals` comment above records that
// non-deterministic reduction drift is invisible to sampling but desyncs
// the Shannon arithmetic coder.
//
// Summing partials reassociates the accumulation, so results are NOT
// bitwise equal to `kv_attention` — a legitimate difference, gated with a
// calibrated tolerance rather than a bitwise assert.
//
// Caller contract: `tg_sz` must be a multiple of `head_dim` and at least
// `head_dim`; `tg_sz <= 1024` so `tg_partial` covers `n_slices * head_dim`.
// The sequence-parallel decode body, shared by `kv_attention_seqpar` and
// `kv_attention_seqpar_rows` (a verify block's positions as grid.y), so the
// two are the same arithmetic by construction.
inline void kv_attention_seqpar_body(
    device const float* Q,
    device const float* K_cache,
    device const float* V_cache,
    device float*       out,
    uint                T,
    uint                head_dim,
    uint                num_q,
    uint                num_kv,
    float               scale,
    uint                window_size,
    constant float*     sinks,
    uint                has_sinks,
    float               softcap,
    uint tg_id,
    uint tid,
    uint tg_sz,
    uint lane,
    uint sg_id,
    threadgroup float*  tg_scores,
    threadgroup float*  tg_partial,
    threadgroup float*  tg_sg_vals)
{
    uint head = tg_id;
    if (head >= num_q) return;
    uint kv_head = head / (num_q / num_kv);
    device const float* q = Q + head * head_dim;
    uint t_start = (window_size > 0 && T > window_size) ? T - window_size : 0;

    // n_slices * head_dim <= tg_sz <= 1024 by the caller contract.

    // ---- Phases 1-2: unchanged from kv_attention ----
    float local_max = -1e30f;
    for (uint t = t_start + tid; t < T; t += tg_sz) {
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
        tg_scores[t - t_start] = dot;
        local_max = max(local_max, dot);
    }

    float sg_max = simd_max(local_max);
    if (lane == 0) tg_sg_vals[sg_id] = sg_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float global_max = tg_sg_vals[0];
    uint n_sg = (tg_sz + 31) / 32;
    for (uint i = 1; i < n_sg; i++) global_max = max(global_max, tg_sg_vals[i]);
    if (has_sinks != 0u) global_max = max(global_max, sinks[head]);

    float local_sum = 0.0f;
    for (uint t = t_start + tid; t < T; t += tg_sz) {
        float w = exp(tg_scores[t - t_start] - global_max);
        tg_scores[t - t_start] = w;
        local_sum += w;
    }

    float sg_sum = simd_sum(local_sum);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) tg_sg_vals[sg_id] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float global_sum = tg_sg_vals[0];
    for (uint i = 1; i < n_sg; i++) global_sum += tg_sg_vals[i];
    if (has_sinks != 0u) global_sum += exp(sinks[head] - global_max);
    float inv_sum = 1.0f / global_sum;

    for (uint t = t_start + tid; t < T; t += tg_sz) {
        tg_scores[t - t_start] *= inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Phase 3: sequence-parallel ----
    uint n_slices = tg_sz / head_dim;
    if (n_slices == 0u) n_slices = 1u;
    uint active = n_slices * head_dim;
    uint d = tid % head_dim;
    uint slice = tid / head_dim;

    if (tid < active) {
        float acc = 0.0f;
        for (uint t = t_start + slice; t < T; t += n_slices) {
            acc += tg_scores[t - t_start]
                 * V_cache[t * num_kv * head_dim + kv_head * head_dim + d];
        }
        tg_partial[slice * head_dim + d] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < head_dim) {
        // Fixed slice order — see the determinism note above.
        float sum = tg_partial[tid];
        for (uint s = 1u; s < n_slices; s++) {
            sum += tg_partial[s * head_dim + tid];
        }
        out[head * head_dim + tid] = sum;
    }
}
kernel void kv_attention_seqpar(
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
    constant float*     sinks   [[buffer(10)]],
    constant uint&      has_sinks [[buffer(11)]],
    constant float&     softcap [[buffer(12)]],
    uint tg_id  [[threadgroup_position_in_grid]],
    uint tid    [[thread_index_in_threadgroup]],
    uint tg_sz  [[threads_per_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]],
    uint sg_id  [[simdgroup_index_in_threadgroup]])
{
    threadgroup float tg_scores[1024];
    threadgroup float tg_partial[1024];
    threadgroup float tg_sg_vals[32];
    kv_attention_seqpar_body(Q, K_cache, V_cache, out, T, head_dim, num_q, num_kv, scale,
                             window_size, sinks, has_sinks, softcap,
                             tg_id, tid, tg_sz, lane, sg_id, tg_scores, tg_partial, tg_sg_vals);
}

// VERIFY-N: `kv_attention_seqpar` over a block of `rows` positions in ONE
// dispatch — grid (num_q, rows), row r at position base + r (cache length
// T + r, Q/output row r of `[rows, num_q, head_dim]`), each row the shared
// body unchanged.
kernel void kv_attention_seqpar_rows(
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
    constant float*     sinks   [[buffer(10)]],
    constant uint&      has_sinks [[buffer(11)]],
    constant float&     softcap [[buffer(12)]],
    uint2 tg_pos  [[threadgroup_position_in_grid]],
    uint  tid     [[thread_index_in_threadgroup]],
    uint2 tg_sz2  [[threads_per_threadgroup]],   // uint2: must match the 2-D grid's width
    uint  lane    [[thread_index_in_simdgroup]],
    uint  sg_id   [[simdgroup_index_in_threadgroup]])
{
    threadgroup float tg_scores[1024];
    threadgroup float tg_partial[1024];
    threadgroup float tg_sg_vals[32];
    const ulong row_off = (ulong)tg_pos.y * num_q * head_dim;
    kv_attention_seqpar_body(Q + row_off, K_cache, V_cache, out + row_off, T + tg_pos.y,
                             head_dim, num_q, num_kv, scale, window_size, sinks, has_sinks,
                             softcap, tg_pos.x, tid, tg_sz2.x, lane, sg_id, tg_scores, tg_partial, tg_sg_vals);
}



// ---------------------------------------------------------------------
// KV-B1: sequence-parallel weighted-V accumulation.
//
// `kv_attention`'s phase 3 is the 86%-of-long-span cost, and it is serial
// in exactly the axis that grows:
//
//     for (d = tid; d < head_dim; d += tg_sz)   // only head_dim threads
//         for (t = t_start; t < T; t++)          // whole span, serially
//
// Widening the threadgroup cannot help it — the extra threads leave the
// outer loop immediately (measured: 4x threads bought 1.06-1.09x).
//
// Here the sequence is split across `n_slices = tg_sz / head_dim` slices:
//
//     d     = tid % head_dim
//     slice = tid / head_dim
//     slice s walks t = t_start + s, +n_slices, ...
//
// For any single `t` the 64 `d`-threads of a slice still read the whole
// contiguous head_dim V row, so sequence parallelism is added WITHOUT
// breaking the naturally coalesced head-dimension load.
//
// No online-softmax merge is needed: `tg_scores` is already normalised and
// shared across the threadgroup, so the slices reduce by plain summation.
// (That merge becomes necessary only if phases 1-2 are ever tiled ACROSS
// threadgroups, where each tile would own its own max/sum state.)
//
// The final reduction runs in FIXED slice order 0..n_slices. Determinism is
// not optional here: the `tg_sg_vals` comment above records that
// non-deterministic reduction drift is invisible to sampling but desyncs
// the Shannon arithmetic coder.
//
// Summing partials reassociates the accumulation, so results are NOT
// bitwise equal to `kv_attention` — a legitimate difference, gated with a
// calibrated tolerance rather than a bitwise assert.
//
// Caller contract: `tg_sz` must be a multiple of `head_dim` and at least
// `head_dim`; `tg_sz <= 1024` so `tg_partial` covers `n_slices * head_dim`.
kernel void kv_attention_seqpar_long(
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
    constant float*     sinks   [[buffer(10)]],
    constant uint&      has_sinks [[buffer(11)]],
    constant float&     softcap [[buffer(12)]],
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

    threadgroup float tg_scores[4096];
    // n_slices * head_dim <= tg_sz <= 1024 by the caller contract.
    threadgroup float tg_partial[1024];

    // ---- Phases 1-2: unchanged from kv_attention ----
    float local_max = -1e30f;
    for (uint t = t_start + tid; t < T; t += tg_sz) {
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
        tg_scores[t - t_start] = dot;
        local_max = max(local_max, dot);
    }

    float sg_max = simd_max(local_max);
    threadgroup float tg_sg_vals[32];
    if (lane == 0) tg_sg_vals[sg_id] = sg_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float global_max = tg_sg_vals[0];
    uint n_sg = (tg_sz + 31) / 32;
    for (uint i = 1; i < n_sg; i++) global_max = max(global_max, tg_sg_vals[i]);
    if (has_sinks != 0u) global_max = max(global_max, sinks[head]);

    float local_sum = 0.0f;
    for (uint t = t_start + tid; t < T; t += tg_sz) {
        float w = exp(tg_scores[t - t_start] - global_max);
        tg_scores[t - t_start] = w;
        local_sum += w;
    }

    float sg_sum = simd_sum(local_sum);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) tg_sg_vals[sg_id] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float global_sum = tg_sg_vals[0];
    for (uint i = 1; i < n_sg; i++) global_sum += tg_sg_vals[i];
    if (has_sinks != 0u) global_sum += exp(sinks[head] - global_max);
    float inv_sum = 1.0f / global_sum;

    for (uint t = t_start + tid; t < T; t += tg_sz) {
        tg_scores[t - t_start] *= inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Phase 3: sequence-parallel ----
    uint n_slices = tg_sz / head_dim;
    if (n_slices == 0u) n_slices = 1u;
    uint active = n_slices * head_dim;
    uint d = tid % head_dim;
    uint slice = tid / head_dim;

    if (tid < active) {
        float acc = 0.0f;
        for (uint t = t_start + slice; t < T; t += n_slices) {
            acc += tg_scores[t - t_start]
                 * V_cache[t * num_kv * head_dim + kv_head * head_dim + d];
        }
        tg_partial[slice * head_dim + d] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < head_dim) {
        // Fixed slice order — see the determinism note above.
        float sum = tg_partial[tid];
        for (uint s = 1u; s < n_slices; s++) {
            sum += tg_partial[s * head_dim + tid];
        }
        out[head * head_dim + tid] = sum;
    }
}

// MEASUREMENT INSTRUMENT (KV-B attribution), not a production path.
//
// `kv_attention` phases 1-2 only: Q.K + softmax, stopping before the
// weighted-V accumulation. Exists to answer one question — how much of the
// kernel's span-linear cost is the score pass, and how much is phase 3?
//
// Phase 3 is the loop that cannot be helped by a wider threadgroup: its
// outer loop is over `head_dim`, so only `head_dim` threads ever enter it,
// and each walks the ENTIRE span serially. Measuring 1-2 in isolation
// attributes the remainder to it without guessing. Writes the softmax
// denominator so the compiler cannot eliminate the work.
kernel void kv_attention_phase12_only(
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
    constant float*     sinks   [[buffer(10)]],
    constant uint&      has_sinks [[buffer(11)]],
    constant float&     softcap [[buffer(12)]],
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
    threadgroup float tg_scores[1024];

    float local_max = -1e30f;
    for (uint t = t_start + tid; t < T; t += tg_sz) {
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
    if (has_sinks != 0u) global_max = max(global_max, sinks[head]);

    float local_sum = 0.0f;
    for (uint t = t_start + tid; t < T; t += tg_sz) {
        float w = exp(tg_scores[t - t_start] - global_max);
        tg_scores[t - t_start] = w;
        local_sum += w;
    }
    float sg_sum = simd_sum(local_sum);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) tg_sg_vals[sg_id] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float global_sum = tg_sg_vals[0];
    for (uint i = 1; i < n_sg; i++) global_sum += tg_sg_vals[i];
    if (has_sinks != 0u) global_sum += exp(sinks[head] - global_max);

    // Observable result so phases 1-2 cannot be optimised away.
    if (tid == 0) out[head * head_dim] = global_sum;
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

/// VERIFY-N: `kv_attention` with a block's positions as grid.y.
pub struct AttendRowsKernel;
impl crate::kernels::ShaderKernel for AttendRowsKernel {
    const KERNEL_NAME: &'static str = "kv_attention_rows";
}

/// VERIFY-N: `kv_attention_seqpar` with a block's positions as grid.y.
pub struct AttendSeqParRowsKernel;
impl crate::kernels::ShaderKernel for AttendSeqParRowsKernel {
    const KERNEL_NAME: &'static str = "kv_attention_seqpar_rows";
}

pub struct AttendLongKernel;
impl crate::kernels::ShaderKernel for AttendLongKernel {
    const KERNEL_NAME: &'static str = "kv_attention_long";
}

/// KV-B1 sequence-parallel phase 3, span <= 1024.
pub struct AttendSeqParKernel;
impl crate::kernels::ShaderKernel for AttendSeqParKernel {
    const KERNEL_NAME: &'static str = "kv_attention_seqpar";
}

/// KV-B1 sequence-parallel phase 3, span <= 4096.
pub struct AttendSeqParLongKernel;
impl crate::kernels::ShaderKernel for AttendSeqParLongKernel {
    const KERNEL_NAME: &'static str = "kv_attention_seqpar_long";
}

/// Measurement-only: `kv_attention` phases 1-2 (see the shader comment).
pub struct AttendPhase12OnlyKernel;
impl crate::kernels::ShaderKernel for AttendPhase12OnlyKernel {
    const KERNEL_NAME: &'static str = "kv_attention_phase12_only";
}

pub struct AppendKernel;
impl crate::kernels::ShaderKernel for AppendKernel {
    const KERNEL_NAME: &'static str = "kv_cache_append";
}
