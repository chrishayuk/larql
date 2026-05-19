//! Rotary Position Embedding (RoPE) — applies position-dependent rotation to Q/K vectors.
//!
//! Split-half pairing: rotates (x[i], x[i + half_dim]) pairs.
//! Matches HuggingFace default and MLX traditional=False.
//!
//! Both kernels support partial rotation via `rotary_dim`:
//! only the first `rotary_dim` dimensions are rotated, the rest pass through.
//! Gemma 4 global layers use 25% rotation (rotary_dim = head_dim * 0.25).
//!
//! ## RoPE variant coverage gap (model-agnosticity audit, 2026-05-09)
//!
//! llama.cpp ships four RoPE variants (per `libggml-metal.dylib` symbols):
//!
//! - `kernel_rope_norm_*` — split-half (this kernel's pattern). ✓ covered.
//! - `kernel_rope_neox_*` — interleaved RoPE (NeoX style: rotates (x[2i], x[2i+1]) pairs
//!   instead of (x[i], x[i+half_dim])). **TODO**: needed for GPT-NeoX, Pythia, some
//!   Falcon variants. Not present.
//! - `kernel_rope_multi_*` — multi-frequency-band RoPE (used by some long-context
//!   models with frequency interpolation). **TODO**: not present.
//! - `kernel_rope_vision_*` — 2D RoPE for vision models. Not load-bearing for
//!   text-only LMs.
//!
//! Add NeoX variant when a non-split-half RoPE model is brought into scope
//! (current architectures in `larql-models/architectures/` all use split-half).
//! See `docs/llama-cpp-comparison.md` §3 and `docs/shader-inventory.md`.

pub const SHADER: &str = r#"
// Apply RoPE to a single vector [dim] in-place at a given absolute position.
// Used by KV-cached decode: apply to Q and K at the correct sequence position.
// Supports partial rotation: only first `rotary_dim` dims are rotated.
// Grid: (rotary_dim/2, 1, 1).
kernel void rope_at_pos(
    device float* x           [[buffer(0)]],   // [dim] — modified in-place (one head)
    constant uint&  dim       [[buffer(1)]],   // head_dim
    constant float& base      [[buffer(2)]],   // rope_theta
    constant uint&  pos       [[buffer(3)]],   // absolute position in sequence
    constant uint&  rotary_dim[[buffer(4)]],   // dimensions to rotate (≤ dim). 0 = use dim.
    uint tid [[thread_position_in_grid]])
{
    uint rdim = (rotary_dim == 0) ? dim : min(rotary_dim, dim);
    uint hdim = rdim / 2;
    if (tid >= hdim) return;

    float freq = 1.0f / pow(base, float(2 * tid) / float(rdim));
    float angle = float(pos) * freq;
    float cos_a = cos(angle);
    float sin_a = sin(angle);

    float re = x[tid];
    float im = x[tid + hdim];

    x[tid]        = re * cos_a - im * sin_a;
    x[tid + hdim] = re * sin_a + im * cos_a;
}

// Apply RoPE to a [seq_len, dim] matrix in-place.
// Supports partial rotation: only the first `rotary_dim` dimensions are rotated,
// the rest pass through unchanged.
// Each thread handles one (position, dimension_pair).
// Grid: (rotary_dim/2, seq_len, 1).
kernel void rope_apply(
    device float* x           [[buffer(0)]],   // [seq_len, dim] — modified in-place
    constant uint&  dim       [[buffer(1)]],
    constant float& base      [[buffer(2)]],   // rope_theta (e.g., 10000.0 or 1000000.0)
    constant uint&  rotary_dim[[buffer(3)]],   // dimensions to rotate (≤ dim). 0 = use dim.
    uint2 tid [[thread_position_in_grid]])
{
    uint rdim = (rotary_dim == 0) ? dim : min(rotary_dim, dim);
    uint d = tid.x;           // dimension pair index [0, rdim/2)
    uint pos = tid.y;         // sequence position
    uint hdim = rdim / 2;
    if (d >= hdim) return;

    float freq = 1.0f / pow(base, float(2 * d) / float(rdim));
    float angle = float(pos) * freq;
    float cos_a = cos(angle);
    float sin_a = sin(angle);

    uint idx_re = pos * dim + d;
    uint idx_im = pos * dim + d + hdim;

    float re = x[idx_re];
    float im = x[idx_im];

    x[idx_re] = re * cos_a - im * sin_a;
    x[idx_im] = re * sin_a + im * cos_a;
}
// Batched RoPE: apply to all heads in one dispatch.
// x = [num_heads * head_dim] contiguous, each head of `head_dim` elements.
// Grid: (rotary_dim/2, num_heads, 1).
kernel void rope_at_pos_batched(
    device float*       x          [[buffer(0)]],   // [num_heads, head_dim] — in-place
    constant uint&      head_dim   [[buffer(1)]],
    constant float&     base       [[buffer(2)]],
    constant uint&      pos        [[buffer(3)]],
    constant uint&      rotary_dim [[buffer(4)]],
    constant uint&      num_heads  [[buffer(5)]],
    uint2 tid [[thread_position_in_grid]])
{
    uint d  = tid.x;   // pair index within head
    uint h  = tid.y;   // head index
    if (h >= num_heads) return;
    uint rdim = (rotary_dim == 0) ? head_dim : min(rotary_dim, head_dim);
    uint hdim = rdim / 2;
    if (d >= hdim) return;

    float freq  = 1.0f / pow(base, float(2 * d) / float(rdim));
    float angle = float(pos) * freq;
    float cos_a = cos(angle);
    float sin_a = sin(angle);

    uint base_idx = h * head_dim;
    float re = x[base_idx + d];
    float im = x[base_idx + d + hdim];
    x[base_idx + d]        = re * cos_a - im * sin_a;
    x[base_idx + d + hdim] = re * sin_a + im * cos_a;
}

// Fused Q+K batched RoPE — applies RoPE to all Q heads then all K heads
// in one dispatch instead of two. Grid: (rotary_dim/2, num_q+num_kv, 1).
// Saves one `dispatch_threads` call per layer × 34 = 34 saved dispatches/token.
kernel void rope_at_pos_batched_qk(
    device float*       Q          [[buffer(0)]],   // [num_q_heads * head_dim]
    device float*       K          [[buffer(1)]],   // [num_kv_heads * head_dim]
    constant uint&      head_dim   [[buffer(2)]],
    constant float&     rope_base  [[buffer(3)]],
    constant uint&      pos        [[buffer(4)]],
    constant uint&      rotary_dim [[buffer(5)]],
    constant uint&      num_q      [[buffer(6)]],   // q heads count
    uint2 tid [[thread_position_in_grid]])
{
    uint d = tid.x;   // pair index
    uint h = tid.y;   // global head index (0..num_q → Q, num_q.. → K)

    uint rdim = (rotary_dim == 0u) ? head_dim : min(rotary_dim, head_dim);
    uint hdim = rdim / 2u;
    if (d >= hdim) return;

    bool is_q = (h < num_q);
    uint local_h = is_q ? h : (h - num_q);
    device float* x = is_q ? Q : K;
    uint base_idx = local_h * head_dim;

    float freq  = 1.0f / pow(rope_base, float(2u * d) / float(rdim));
    float angle = float(pos) * freq;
    float cos_a = cos(angle);
    float sin_a = sin(angle);

    float re = x[base_idx + d];
    float im = x[base_idx + d + hdim];
    x[base_idx + d]        = re * cos_a - im * sin_a;
    x[base_idx + d + hdim] = re * sin_a + im * cos_a;
}
"#;

pub struct RopeApplyKernel;
impl crate::kernels::ShaderKernel for RopeApplyKernel {
    const KERNEL_NAME: &'static str = "rope_apply";
}

pub struct RopeAtPosKernel;
impl crate::kernels::ShaderKernel for RopeAtPosKernel {
    const KERNEL_NAME: &'static str = "rope_at_pos";
}

pub struct RopeAtPosBatchedKernel;
impl crate::kernels::ShaderKernel for RopeAtPosBatchedKernel {
    const KERNEL_NAME: &'static str = "rope_at_pos_batched";
}

pub struct RopeAtPosBatchedQkKernel;
impl crate::kernels::ShaderKernel for RopeAtPosBatchedQkKernel {
    const KERNEL_NAME: &'static str = "rope_at_pos_batched_qk";
}
