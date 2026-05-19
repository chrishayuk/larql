//! Basic causal attention: QK^T softmax V for small seq_len.
//! One thread per (head_dim, query_position). Designed for seq ≤ 64.

pub const SHADER: &str = r#"
kernel void causal_attention(
    device const float* Q       [[buffer(0)]],
    device const float* K       [[buffer(1)]],
    device const float* V       [[buffer(2)]],
    device float*       out     [[buffer(3)]],
    constant uint&      seq_len [[buffer(4)]],
    constant uint&      head_dim[[buffer(5)]],
    constant float&     scale   [[buffer(6)]],
    uint2 tid [[thread_position_in_grid]])
{
    uint d = tid.x;
    uint q = tid.y;
    if (d >= head_dim || q >= seq_len) return;

    float max_score = -1e30f;
    for (uint k = 0; k <= q; k++) {
        float score = 0.0f;
        for (uint i = 0; i < head_dim; i++)
            score += Q[q * head_dim + i] * K[k * head_dim + i];
        score *= scale;
        if (score > max_score) max_score = score;
    }

    float sum_exp = 0.0f;
    float weighted_v = 0.0f;
    for (uint k = 0; k <= q; k++) {
        float score = 0.0f;
        for (uint i = 0; i < head_dim; i++)
            score += Q[q * head_dim + i] * K[k * head_dim + i];
        score *= scale;
        float w = exp(score - max_score);
        sum_exp += w;
        weighted_v += w * V[k * head_dim + d];
    }

    out[q * head_dim + d] = weighted_v / sum_exp;
}
"#;

pub struct Kernel;
impl crate::kernels::ShaderKernel for Kernel {
    const KERNEL_NAME: &'static str = "causal_attention";
}
