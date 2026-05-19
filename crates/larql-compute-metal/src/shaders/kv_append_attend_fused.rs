//! Fused **KV-cache append + attention** for token decode.
//!
//! Replaces the consecutive `kv_cache_append` + `kv_attention` dispatches
//! with a single kernel: each TG (per Q head) writes the new K/V row at
//! position `pos` for its kv_head FIRST (cooperatively across the TG's
//! threads), then `threadgroup_barrier(mem_device)` to publish the
//! writes, then proceeds with the standard attention over T = pos + 1
//! positions.
//!
//! **Why this kernel exists**: the `kv_cache_append` dispatch is one
//! standalone call per layer (~7 µs dispatch overhead × 34 layers ≈
//! 0.24 ms/tok). The work itself is tiny (256 floats per kv_head ×
//! 4 kv_heads = 1024 stores) — so the cost is *almost entirely
//! dispatch overhead*. Folding the writes into the front of
//! `kv_attention`'s per-TG init phase eliminates the extra dispatch.
//!
//! **Cross-TG memory ordering**: in GQA, multiple Q-head TGs share one
//! kv_head. Those TGs all redundantly write the same K/V row at
//! position `pos` — idempotent, no race. The TG-internal
//! `threadgroup_barrier(mem_device)` ensures each TG's writes are
//! visible to its own subsequent reads.
//!
//! **Why not also fuse with kv_attention's other phases?** The kernel
//! already does softmax + V sum in one shot; this fusion only attacks
//! the dispatch boundary at the start.

pub const SHADER: &str = r#"
// Decode-mode KV append + attention. Same I/O as kv_attention but takes
// new_k / new_v inputs and writes them to K_cache[pos] / V_cache[pos]
// before the attention loop. Eliminates the kv_cache_append dispatch.
kernel void kv_append_attend_fused(
    device const float* Q       [[buffer(0)]],
    device float*       K_cache [[buffer(1)]],
    device float*       V_cache [[buffer(2)]],
    device float*       out     [[buffer(3)]],
    constant uint&      T       [[buffer(4)]],   // pos + 1 (length AFTER append)
    constant uint&      head_dim[[buffer(5)]],
    constant uint&      num_q   [[buffer(6)]],
    constant uint&      num_kv  [[buffer(7)]],
    constant float&     scale   [[buffer(8)]],
    constant uint&      window_size [[buffer(9)]],
    device const float* new_k   [[buffer(10)]],  // [num_kv * head_dim]
    device const float* new_v   [[buffer(11)]],  // [num_kv * head_dim]
    uint tg_id  [[threadgroup_position_in_grid]],
    uint tid    [[thread_index_in_threadgroup]],
    uint tg_sz  [[threads_per_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]],
    uint sg_id  [[simdgroup_index_in_threadgroup]])
{
    uint head = tg_id;
    if (head >= num_q) return;
    uint kv_head = head / (num_q / num_kv);

    // ── Phase 0: cooperatively write this TG's kv_head's K/V row at
    // position pos = T-1. With GQA each kv_head is shared by
    // (num_q/num_kv) Q heads → the same row gets written by that many
    // TGs. Identical data, idempotent, race-safe.
    uint pos = T - 1u;
    uint cache_row_off = pos * num_kv * head_dim + kv_head * head_dim;
    uint new_off       = kv_head * head_dim;
    for (uint d = tid; d < head_dim; d += tg_sz) {
        K_cache[cache_row_off + d] = new_k[new_off + d];
        V_cache[cache_row_off + d] = new_v[new_off + d];
    }
    threadgroup_barrier(mem_flags::mem_device);

    // ── Phase 1..3: identical to `kv_attention` body, with cache reads
    // now seeing the just-written position pos = T-1.
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
"#;

pub struct Kernel;
impl crate::kernels::ShaderKernel for Kernel {
    const KERNEL_NAME: &'static str = "kv_append_attend_fused";
}
