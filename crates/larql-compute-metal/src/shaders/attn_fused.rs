//! Fused **QK-norm + RoPE + KV-cache append + attention** for token decode.
//!
//! Collapses the qk_norm_rope_fused + kv_append_attend_fused two-dispatch
//! pair into ONE kernel per layer. Saves 1 dispatch/layer × 34 ≈ 0.2 ms/tok.
//!
//! **Per-TG layout** (one TG per Q head, num_q TGs total):
//!  1. Compute RMS over raw Q[head] from Q_in → inv_rms_q.
//!  2. Compute RMS over raw K[kv_head] from K_in → inv_rms_k.
//!  3. Write normed Q to threadgroup memory (tg_q).
//!  4. Write normed K to threadgroup memory (tg_k_normed).
//!  5. RoPE pass: for each rotary pair (d, d+hdim), compute (cos_a, sin_a)
//!     **once per pair** and apply to BOTH tg_q (in-place) and tg_k_normed
//!     (writing the rotated K directly to `K_cache[pos][kv_head]`). This
//!     keeps transcendental cost at 1 per-thread per-pair, matching the
//!     standalone `qk_norm_rope_fused` (the first cut of this kernel
//!     duplicated transcendentals and regressed 74→60 tok/s).
//!  6. Tail-copy K beyond rotary band (partial-rope only — for full-rope
//!     archs the loop is empty).
//!  7. Stream V[kv_head] from V_in directly to V_cache[pos][kv_head]
//!     (no norm, no rope).
//!  8. `threadgroup_barrier(mem_flags::mem_device)` to publish K/V cache
//!     writes within the TG.
//!  9. Standard attention over T = pos + 1 positions, reading Q from
//!     threadgroup memory (tg_q) and K/V from the cache.
//!
//! **Why this is safe** (cross-TG memory): with GQA, multiple Q-head TGs
//! share one kv_head and redundantly write the same normed+roped K/V
//! values. Idempotent, race-safe. The TG-internal `mem_device` barrier
//! ensures each TG sees its own writes before reading.
//!
//! **Threadgroup memory budget** (head_dim ≤ 256, T ≤ 1024):
//!  - tg_q[256]         = 1 KB
//!  - tg_k_normed[256]  = 1 KB
//!  - tg_scores[1024]   = 4 KB
//!  - tg_red[8]         = 32 B
//!
//!  Total ~6 KB — well within 32 KB/TG.

pub const SHADER: &str = r#"
kernel void attn_fused(
    device const float* Q_in       [[buffer(0)]],   // raw Q [num_q  * head_dim]
    device const float* K_in       [[buffer(1)]],   // raw K [num_kv * head_dim]
    device const float* V_in       [[buffer(2)]],   // raw V [num_kv * head_dim]
    device float*       K_cache    [[buffer(3)]],
    device float*       V_cache    [[buffer(4)]],
    device float*       out        [[buffer(5)]],
    device const float* q_weight   [[buffer(6)]],   // qk_norm Q weight [head_dim]
    device const float* k_weight   [[buffer(7)]],   // qk_norm K weight [head_dim]
    constant uint&      T          [[buffer(8)]],   // pos + 1 (length AFTER append)
    constant uint&      head_dim   [[buffer(9)]],
    constant uint&      num_q      [[buffer(10)]],
    constant uint&      num_kv     [[buffer(11)]],
    constant float&     scale      [[buffer(12)]],
    constant uint&      window_size[[buffer(13)]],
    constant float&     eps        [[buffer(14)]],
    constant float&     qk_offset  [[buffer(15)]],  // 1.0 on Gemma 2/3, 0.0 on Gemma 4
    constant float&     rope_base  [[buffer(16)]],
    constant uint&      rotary_dim [[buffer(17)]],
    uint tg_id  [[threadgroup_position_in_grid]],
    uint tid    [[thread_index_in_threadgroup]],
    uint tg_sz  [[threads_per_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]],
    uint sg_id  [[simdgroup_index_in_threadgroup]])
{
    uint head = tg_id;
    if (head >= num_q) return;
    uint kv_head = head / (num_q / num_kv);
    uint pos = T - 1u;

    threadgroup float tg_q[256];
    threadgroup float tg_k_normed[256];
    threadgroup float tg_red[8];
    uint n_sg = (tg_sz + 31u) / 32u;

    uint rdim = (rotary_dim == 0u) ? head_dim : min(rotary_dim, head_dim);
    uint hdim = rdim / 2u;

    // ── Phase 1: parallel RMS for Q[head] AND K[kv_head] in one pass ──
    // Each thread accumulates two squares (one for Q, one for K). We use
    // simdgroup reduction and re-use tg_red as a tiny buffer for both.
    float partial_q = 0.0f;
    float partial_k = 0.0f;
    for (uint d = tid; d < head_dim; d += tg_sz) {
        float vq = Q_in[head    * head_dim + d];
        float vk = K_in[kv_head * head_dim + d];
        partial_q += vq * vq;
        partial_k += vk * vk;
    }
    // Reduce Q
    {
        float sg = simd_sum(partial_q);
        if (lane == 0) tg_red[sg_id] = sg;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float ss_q = tg_red[0];
    for (uint i = 1u; i < n_sg; i++) ss_q += tg_red[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Reduce K
    {
        float sg = simd_sum(partial_k);
        if (lane == 0) tg_red[sg_id] = sg;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float ss_k = tg_red[0];
    for (uint i = 1u; i < n_sg; i++) ss_k += tg_red[i];
    float inv_rms_q = 1.0f / sqrt(ss_q / float(head_dim) + eps);
    float inv_rms_k = 1.0f / sqrt(ss_k / float(head_dim) + eps);

    // ── Phase 2: write normed Q,K to TG memory ──
    for (uint d = tid; d < head_dim; d += tg_sz) {
        float vq = Q_in[head    * head_dim + d];
        float vk = K_in[kv_head * head_dim + d];
        tg_q[d]        = (vq * inv_rms_q) * (qk_offset + q_weight[d]);
        tg_k_normed[d] = (vk * inv_rms_k) * (qk_offset + k_weight[d]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 3: shared RoPE — compute (cos, sin) ONCE per pair, apply
    // to both Q (in-place in tg_q) and K (writing rotated values to
    // K_cache directly). Halves transcendental cost vs separate Q/K
    // rope passes.
    uint cache_off = pos * num_kv * head_dim + kv_head * head_dim;
    for (uint d = tid; d < hdim; d += tg_sz) {
        float freq  = 1.0f / pow(rope_base, float(2u * d) / float(rdim));
        float angle = float(pos) * freq;
        float cos_a = cos(angle);
        float sin_a = sin(angle);

        // Q rope: in-place
        float qr = tg_q[d];
        float qi = tg_q[d + hdim];
        tg_q[d]        = qr * cos_a - qi * sin_a;
        tg_q[d + hdim] = qr * sin_a + qi * cos_a;

        // K rope: write rotated K to cache
        float kr = tg_k_normed[d];
        float ki = tg_k_normed[d + hdim];
        K_cache[cache_off + d]        = kr * cos_a - ki * sin_a;
        K_cache[cache_off + d + hdim] = kr * sin_a + ki * cos_a;
    }
    // Tail past rotary band (partial-rope only): copy normed K through.
    for (uint d = tid + rdim; d < head_dim; d += tg_sz) {
        K_cache[cache_off + d] = tg_k_normed[d];
    }

    // ── Phase 4: stream V[kv_head] to V_cache[pos][kv_head] ──
    for (uint d = tid; d < head_dim; d += tg_sz) {
        V_cache[cache_off + d] = V_in[kv_head * head_dim + d];
    }

    threadgroup_barrier(mem_flags::mem_device);

    // ── Phase 5: scores. Reads Q from tg_q, K from K_cache. ──
    uint t_start = (window_size > 0u && T > window_size) ? T - window_size : 0u;
    threadgroup float tg_scores[1024];

    float local_max = -1e30f;
    for (uint t = t_start + tid; t < T; t += tg_sz) {
        device const float* k = K_cache + t * num_kv * head_dim + kv_head * head_dim;
        float dot = 0.0f;
        for (uint d = 0; d + 3 < head_dim; d += 4) {
            dot += tg_q[d]*k[d] + tg_q[d+1]*k[d+1] + tg_q[d+2]*k[d+2] + tg_q[d+3]*k[d+3];
        }
        for (uint d = (head_dim & ~3u); d < head_dim; d++) dot += tg_q[d] * k[d];
        dot *= scale;
        tg_scores[t - t_start] = dot;
        local_max = max(local_max, dot);
    }

    {
        float sg_max = simd_max(local_max);
        if (lane == 0) tg_red[sg_id] = sg_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float global_max = tg_red[0];
    for (uint i = 1u; i < n_sg; i++) global_max = max(global_max, tg_red[i]);

    // ── Phase 6: softmax numerator + sum ──
    float local_sum = 0.0f;
    for (uint t = t_start + tid; t < T; t += tg_sz) {
        float w = exp(tg_scores[t - t_start] - global_max);
        tg_scores[t - t_start] = w;
        local_sum += w;
    }

    {
        float sg_sum = simd_sum(local_sum);
        if (lane == 0) tg_red[sg_id] = sg_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float global_sum = tg_red[0];
    for (uint i = 1u; i < n_sg; i++) global_sum += tg_red[i];
    float inv_sum = 1.0f / global_sum;

    for (uint t = t_start + tid; t < T; t += tg_sz) {
        tg_scores[t - t_start] *= inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 7: V sum, write per-head out ──
    device float* out_head = out + head * head_dim;
    for (uint d = tid; d < head_dim; d += tg_sz) {
        float acc = 0.0f;
        for (uint t = t_start; t < T; t++) {
            acc += tg_scores[t - t_start] * V_cache[t * num_kv * head_dim + kv_head * head_dim + d];
        }
        out_head[d] = acc;
    }
}
"#;

pub struct Kernel;
impl crate::kernels::ShaderKernel for Kernel {
    const KERNEL_NAME: &'static str = "attn_fused";
}
