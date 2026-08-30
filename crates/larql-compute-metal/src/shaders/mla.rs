//! MLA's two stages that are not already a matvec.
//!
//! Rung 6a. The trajectory measured the GPU idling ~24 ms a token across
//! seven CPU MLA layers — more than the 19.2 ms of GPU work it was
//! waiting between. So MLA is not a compute port; it is the operator
//! that lets a token stay GPU-owned across a KDA→MLA→KDA boundary, and
//! its own speed is secondary to that.
//!
//! Everything wide in MLA is already a grouped bf16 matvec: `q_proj`,
//! `kv_a_proj` and `o_proj` are one slot each, and `kv_b_proj` — which
//! decompresses EVERY cached position on EVERY step, and is therefore
//! the operator's real cost — is exactly the grouped kernel's
//! `PerSlot` shape with one bank and one offset repeated. Only two
//! stages need new code.
//!
//! ## The cache is the interesting part
//!
//! KDA's state is a fixed `[H][D][D]` matrix. MLA's grows: one
//! `kv_lora_rank + rope` entry per position, cached RAW and decompressed
//! at read time. Nothing decompressed is ever cached, which is the
//! operator's actual cost profile and not an artefact of the reference
//! being naive — so the device path caches the same raw latents, in one
//! buffer that a new position is written into by binding it at an
//! offset. No append kernel, and no host round trip.

/// Positions one attention threadgroup can score. The scores live in
/// threadgroup memory for the softmax, so this is a real limit and the
/// host refuses beyond it rather than overrunning.
pub const MAX_POSITIONS: usize = 2048;
/// Threads per threadgroup for both kernels. `v_head_dim` must not
/// exceed it — the value accumulation gives one thread each output
/// dimension.
pub const THREADS_PER_TG: u64 = 128;

pub fn shader() -> String {
    format!(
        r#"
constant uint MLA_MAX_POSITIONS = {max_positions};

// RMSNorm over the LATENT half of every cached position.
//
// The latent half only: `kv_a_layernorm` never touches the shared rope-K
// tail that shares the same cache entry. One threadgroup per position.
kernel void mla_kv_a_norm_positions(
    device const float* cache   [[buffer(0)]],  // [positions, latent+rope]
    device const float* weight  [[buffer(1)]],  // [latent]
    device float*       out     [[buffer(2)]],  // [positions, latent]
    constant uint&      latent  [[buffer(3)]],
    constant uint&      stride  [[buffer(4)]],  // latent + rope
    constant float&     eps     [[buffer(5)]],
    uint  pos    [[threadgroup_position_in_grid]],
    uint  tid    [[thread_position_in_threadgroup]],
    uint  tcount [[threads_per_threadgroup]])
{{
    threadgroup float partial[{threads}];
    device const float* src = cache + (ulong)pos * stride;
    device float* dst = out + (ulong)pos * latent;

    float acc = 0.0f;
    for (uint i = tid; i < latent; i += tcount) acc += src[i] * src[i];
    partial[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tcount / 2u; s > 0u; s >>= 1u) {{
        if (tid < s) partial[tid] += partial[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    const float inv = rsqrt(partial[0] / float(latent) + eps);
    for (uint i = tid; i < latent; i += tcount) dst[i] = src[i] * inv * weight[i];
}}

// One head's attention over every visible position.
//
// The score is `q_nope . k_nope[p] + q_rope . k_rope[p]`, scaled — the
// rope half comes straight out of the cache entry's tail, unrotated,
// because this checkpoint shares one positionless rope-K across heads.
// Softmax is the max-subtracted form the CPU reference uses.
kernel void mla_attention(
    device const float* q        [[buffer(0)]],  // [heads, q_head_dim]
    device const float* kv_b     [[buffer(1)]],  // [positions, heads*(nope+v_dim)]
    device const float* cache    [[buffer(2)]],  // [positions, latent+rope]
    device float*       weights  [[buffer(3)]],  // [heads, positions] out
    device float*       value    [[buffer(4)]],  // [heads, v_dim] out
    constant uint&      visible  [[buffer(5)]],
    constant uint&      nope     [[buffer(6)]],
    constant uint&      rope     [[buffer(7)]],
    constant uint&      v_dim    [[buffer(8)]],
    constant uint&      latent   [[buffer(9)]],
    constant float&     scaling  [[buffer(10)]],
    /// `heads * (nope + v_dim)` — one cached position's decompressed row.
    constant uint&      kv_row   [[buffer(11)]],
    uint  head   [[threadgroup_position_in_grid]],
    uint  tid    [[thread_position_in_threadgroup]],
    uint  tcount [[threads_per_threadgroup]])
{{
    threadgroup float scores[MLA_MAX_POSITIONS];
    threadgroup float reduce[{threads}];

    const uint q_head_dim = nope + rope;
    const uint kv_head = nope + v_dim;
    const uint stride = latent + rope;
    device const float* q_nope = q + (ulong)head * q_head_dim;
    device const float* q_rope = q_nope + nope;

    // ── scores ──
    for (uint p = tid; p < visible; p += tcount) {{
        device const float* k_nope = kv_b + (ulong)p * kv_row + (ulong)head * kv_head;
        device const float* k_rope = cache + (ulong)p * stride + latent;
        float dot = 0.0f;
        for (uint i = 0; i < nope; ++i) dot += q_nope[i] * k_nope[i];
        for (uint i = 0; i < rope; ++i) dot += q_rope[i] * k_rope[i];
        scores[p] = dot * scaling;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── softmax, max-subtracted exactly as the reference does it ──
    float m = -INFINITY;
    for (uint p = tid; p < visible; p += tcount) m = max(m, scores[p]);
    reduce[tid] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tcount / 2u; s > 0u; s >>= 1u) {{
        if (tid < s) reduce[tid] = max(reduce[tid], reduce[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    const float peak = reduce[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float partial = 0.0f;
    for (uint p = tid; p < visible; p += tcount) {{
        const float e = exp(scores[p] - peak);
        scores[p] = e;
        partial += e;
    }}
    reduce[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tcount / 2u; s > 0u; s >>= 1u) {{
        if (tid < s) reduce[tid] += reduce[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    const float total = reduce[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint p = tid; p < visible; p += tcount) {{
        scores[p] /= total;
        weights[(ulong)head * visible + p] = scores[p];
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── weighted value sum: one thread owns one output dimension, so
    // its accumulation over positions is the reference's own order ──
    for (uint d = tid; d < v_dim; d += tcount) {{
        float acc = 0.0f;
        for (uint p = 0; p < visible; ++p) {{
            device const float* v =
                kv_b + (ulong)p * kv_row + (ulong)head * kv_head + nope;
            acc += scores[p] * v[d];
        }}
        value[(ulong)head * v_dim + d] = acc;
    }}
}}
"#,
        max_positions = MAX_POSITIONS,
        threads = THREADS_PER_TG,
    )
}

pub struct KvANormKernel;
impl crate::kernels::ShaderKernel for KvANormKernel {
    const KERNEL_NAME: &'static str = "mla_kv_a_norm_positions";
}

pub struct AttentionKernel;
impl crate::kernels::ShaderKernel for AttentionKernel {
    const KERNEL_NAME: &'static str = "mla_attention";
}
