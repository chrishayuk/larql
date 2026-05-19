//! Fused attention shader: RoPE → QK-norm → GQA → softcap → softmax → V.
//!
//! Handles all model-specific features in one kernel:
//! - RoPE: split-half position-dependent rotation on Q and K
//! - QK-norm: optional per-head L2 normalization (Gemma 3)
//! - GQA: grouped query attention (num_q / num_kv heads share K/V)
//! - Softcap: optional tanh(score/cap)*cap before softmax (Gemma 2)
//! - Causal mask: each position attends only to positions ≤ itself
//!
//! Grid: one threadgroup per (query_head, query_position).
//! At seq=1 (decode), this is one threadgroup per head.

pub const SHADER: &str = r#"
// Fused causal attention with RoPE, optional QK-norm and softcap.
//
// Input: Q[seq, num_q * head_dim], K[seq, num_kv * head_dim], V[seq, num_kv * head_dim]
// Output: out[seq, num_q * head_dim]
//
// One threadgroup per (head, query_position). Threads cooperate on key-dimension dot products.
constant uint MAX_FUSED_ATTENTION_SEQ_LEN = 4096;

kernel void fused_attention(
    device const float* Q       [[buffer(0)]],
    device const float* K       [[buffer(1)]],
    device const float* V       [[buffer(2)]],
    device float*       out     [[buffer(3)]],
    constant uint&      seq_len [[buffer(4)]],
    constant uint&      head_dim[[buffer(5)]],
    constant uint&      num_q   [[buffer(6)]],
    constant uint&      num_kv  [[buffer(7)]],
    constant float&     scale   [[buffer(8)]],
    constant float&     rope_base [[buffer(9)]],
    constant uint&      use_qk_norm [[buffer(10)]],  // 0 or 1
    constant float&     softcap     [[buffer(11)]],  // 0.0 = disabled
    constant uint&      skip_rope   [[buffer(12)]],  // 0 = apply RoPE, 1 = skip (caller pre-applied)
    constant uint&      rotary_dim  [[buffer(13)]],  // 0 = full head_dim, else partial rotation
    uint2 tg_id [[threadgroup_position_in_grid]],    // (head, query_pos)
    uint tid    [[thread_index_in_threadgroup]])
{
    uint head = tg_id.x;
    uint qi = tg_id.y;
    if (head >= num_q || qi >= seq_len) return;

    uint tg_sz = 256;  // threadgroup size
    uint kv_head = head / (num_q / num_kv);
    uint rdim = (rotary_dim == 0) ? head_dim : min(rotary_dim, head_dim);
    uint hdim = rdim / 2;

    // ── Local Q with optional RoPE (partial rotation support) ──
    // Only the first rdim dimensions are rotated; the rest pass through.
    //
    // Strided load: when head_dim > tg_sz (Gemma 4 global layers have
    // head_dim=512 with a 256-thread TG), each thread covers multiple
    // slots so every tg_q[d] is populated. Previously this was gated on
    // `if (tid < head_dim)`, which silently zeroed tg_q[256..512] and
    // gave ~6% magnitude loss in attention output on global layers.
    threadgroup float tg_q[512];   // max head_dim = 512
    for (uint d = tid; d < head_dim; d += tg_sz) {
        uint q_idx = qi * num_q * head_dim + head * head_dim + d;
        float q_val = Q[q_idx];

        if (skip_rope == 0 && d < rdim) {
            // RoPE: split-half rotation within rotary dims
            float freq = 1.0f / pow(rope_base, float(2 * (d % hdim)) / float(rdim));
            float angle = float(qi) * freq;
            float cos_a = cos(angle);
            float sin_a = sin(angle);

            uint pair_d = (d < hdim) ? d + hdim : d - hdim;
            uint pair_idx = qi * num_q * head_dim + head * head_dim + pair_d;
            float pair_val = Q[pair_idx];

            float rotated;
            if (d < hdim) {
                rotated = q_val * cos_a - pair_val * sin_a;
            } else {
                rotated = pair_val * sin_a + q_val * cos_a;
            }
            tg_q[d] = rotated;
        } else {
            tg_q[d] = q_val;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Optional QK-norm: normalize Q vector.
    // Strided write so head_dim > tg_sz works (Gemma 4 global: 512).
    if (use_qk_norm != 0) {
        threadgroup float tg_norm_sum;
        if (tid == 0) {
            float s = 0.0f;
            for (uint d = 0; d < head_dim; d++) s += tg_q[d] * tg_q[d];
            tg_norm_sum = rsqrt(s + 1e-6f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint d = tid; d < head_dim; d += tg_sz) {
            tg_q[d] *= tg_norm_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── Attention scores: Q · K^T for all k ≤ qi ──
    threadgroup float tg_scores[MAX_FUSED_ATTENTION_SEQ_LEN];
    threadgroup float tg_max = 0.0f;
    threadgroup float tg_sum = 0.0f;

    float local_max = -1e30f;
    uint causal_len = qi + 1;

    for (uint k = tid; k < causal_len; k += tg_sz) {
        // Load K[k] for this KV head, optionally apply partial RoPE
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            uint k_idx = k * num_kv * head_dim + kv_head * head_dim + d;
            float k_val = K[k_idx];

            float k_final;
            if (skip_rope == 0 && d < rdim) {
                // RoPE on K (only within rotary dims)
                float freq = 1.0f / pow(rope_base, float(2 * (d % hdim)) / float(rdim));
                float angle = float(k) * freq;
                float cos_a = cos(angle);
                float sin_a = sin(angle);

                uint pair_d = (d < hdim) ? d + hdim : d - hdim;
                uint pair_idx = k * num_kv * head_dim + kv_head * head_dim + pair_d;
                float pair_val = K[pair_idx];

                if (d < hdim) {
                    k_final = k_val * cos_a - pair_val * sin_a;
                } else {
                    k_final = pair_val * sin_a + k_val * cos_a;
                }
            } else {
                k_final = k_val;
            }

            dot += tg_q[d] * k_final;
        }

        dot *= scale;

        // Optional softcap
        if (softcap > 0.0f) {
            dot = tanh(dot / softcap) * softcap;
        }

        tg_scores[k] = dot;
        if (dot > local_max) local_max = dot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce max
    threadgroup float tg_maxes[256];
    tg_maxes[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float m = -1e30f;
        for (uint i = 0; i < min(tg_sz, causal_len); i++) {
            if (tg_maxes[i] > m) m = tg_maxes[i];
        }
        tg_max = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Softmax: exp and sum
    float local_sum = 0.0f;
    for (uint k = tid; k < causal_len; k += tg_sz) {
        float w = exp(tg_scores[k] - tg_max);
        tg_scores[k] = w;
        local_sum += w;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float tg_sums[256];
    tg_sums[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float s = 0.0f;
        for (uint i = 0; i < min(tg_sz, causal_len); i++) s += tg_sums[i];
        tg_sum = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_sum = 1.0f / tg_sum;

    // ── Weighted sum of V ──
    for (uint d = tid; d < head_dim; d += tg_sz) {
        float acc = 0.0f;
        for (uint k = 0; k < causal_len; k++) {
            uint v_idx = k * num_kv * head_dim + kv_head * head_dim + d;
            acc += tg_scores[k] * inv_sum * V[v_idx];
        }
        out[qi * num_q * head_dim + head * head_dim + d] = acc;
    }
}
"#;

pub struct Kernel;
impl crate::kernels::ShaderKernel for Kernel {
    const KERNEL_NAME: &'static str = "fused_attention";
}
