//! Rotary Position Embedding (RoPE) — applies position-dependent rotation to Q/K vectors.
//!
//! Split-half pairing: rotates (x[i], x[i + half_dim]) pairs.
//! Matches HuggingFace default and MLX traditional=False.
//!
//! ## Frequencies come from the host, not from `pow` in-shader
//!
//! The rope-bearing kernels take an `inv_freq` buffer and an `amplitude`
//! scalar, precomputed by `larql_compute::attention::rope::rope_freq_plan` —
//! the same function the CPU path uses. Deriving `1/base^(2d/rdim)` in-shader
//! is what let Metal decode ignore llama3, YaRN *and* Gemma 3's linear
//! position divisor while Metal prefill honoured all three
//! (`docs/k3-funnel.md` §4.10). A kernel that cannot see `rope_base` cannot
//! forget to scale it.
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
    device const float* inv_freq [[buffer(2)]], // [rotary_dim/2], host-computed
    constant uint&  pos       [[buffer(3)]],   // absolute position in sequence
    constant uint&  rotary_dim[[buffer(4)]],   // dimensions to rotate (≤ dim). 0 = use dim.
    constant float& amplitude [[buffer(5)]],   // cos/sin scalar (YaRN); 1.0 otherwise
    uint tid [[thread_position_in_grid]])
{
    uint rdim = (rotary_dim == 0) ? dim : min(rotary_dim, dim);
    uint hdim = rdim / 2;
    if (tid >= hdim) return;

    float angle = float(pos) * inv_freq[tid];
    float cos_a = cos(angle) * amplitude;
    float sin_a = sin(angle) * amplitude;

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
    device const float* inv_freq [[buffer(2)]], // [rotary_dim/2], host-computed
    constant uint&  rotary_dim[[buffer(3)]],   // dimensions to rotate (≤ dim). 0 = use dim.
    constant float& amplitude [[buffer(4)]],   // cos/sin scalar (YaRN); 1.0 otherwise
    uint2 tid [[thread_position_in_grid]])
{
    uint rdim = (rotary_dim == 0) ? dim : min(rotary_dim, dim);
    uint d = tid.x;           // dimension pair index [0, rdim/2)
    uint pos = tid.y;         // sequence position
    uint hdim = rdim / 2;
    if (d >= hdim) return;

    float angle = float(pos) * inv_freq[d];
    float cos_a = cos(angle) * amplitude;
    float sin_a = sin(angle) * amplitude;

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
    device const float* inv_freq   [[buffer(2)]],  // [rotary_dim/2]
    constant uint&      pos        [[buffer(3)]],
    constant uint&      rotary_dim [[buffer(4)]],
    constant uint&      num_heads  [[buffer(5)]],
    constant float&     amplitude  [[buffer(6)]],
    uint2 tid [[thread_position_in_grid]])
{
    uint d  = tid.x;   // pair index within head
    uint h  = tid.y;   // head index
    if (h >= num_heads) return;
    uint rdim = (rotary_dim == 0) ? head_dim : min(rotary_dim, head_dim);
    uint hdim = rdim / 2;
    if (d >= hdim) return;

    float angle = float(pos) * inv_freq[d];
    float cos_a = cos(angle) * amplitude;
    float sin_a = sin(angle) * amplitude;

    uint base_idx = h * head_dim;
    float re = x[base_idx + d];
    float im = x[base_idx + d + hdim];
    x[base_idx + d]        = re * cos_a - im * sin_a;
    x[base_idx + d + hdim] = re * sin_a + im * cos_a;
}

// VERIFY-N: rope_at_pos_batched over `rows` consecutive positions laid
// out `[rows, num_heads, head_dim]`: head h rotates at position
// pos0 + h / num_heads. Same per-pair arithmetic as rope_at_pos_batched,
// but NOT bit-identical to it on every GPU: a uniform `pos` and a per-thread
// `pos` may lower differently under fast math (the macos-14 runner's GPU
// does). Lowered decode therefore dispatches this kernel with rows = 1, so
// decode and a VERIFY-N block share one kernel. Grid: (hdim, rows*num_heads).
kernel void rope_rows(
    device float*       x          [[buffer(0)]],
    constant uint&      head_dim   [[buffer(1)]],
    device const float* inv_freq   [[buffer(2)]],
    constant uint&      pos0       [[buffer(3)]],
    constant uint&      rotary_dim [[buffer(4)]],
    constant uint&      num_heads  [[buffer(5)]],
    constant float&     amplitude  [[buffer(6)]],
    constant uint&      total_heads [[buffer(7)]],
    uint2 tid [[thread_position_in_grid]])
{
    uint d  = tid.x;
    uint h  = tid.y;
    if (h >= total_heads) return;
    uint rdim = (rotary_dim == 0) ? head_dim : min(rotary_dim, head_dim);
    uint hdim = rdim / 2;
    if (d >= hdim) return;

    uint pos = pos0 + h / num_heads;
    float angle = float(pos) * inv_freq[d];
    float cos_a = cos(angle) * amplitude;
    float sin_a = sin(angle) * amplitude;

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
    device const float* inv_freq   [[buffer(3)]],  // [rotary_dim/2]
    constant uint&      pos        [[buffer(4)]],
    constant uint&      rotary_dim [[buffer(5)]],
    constant uint&      num_q      [[buffer(6)]],   // q heads count
    constant float&     amplitude  [[buffer(7)]],
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

    float angle = float(pos) * inv_freq[d];
    float cos_a = cos(angle) * amplitude;
    float sin_a = sin(angle) * amplitude;

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

pub struct RopeRowsKernel;
impl crate::kernels::ShaderKernel for RopeRowsKernel {
    const KERNEL_NAME: &'static str = "rope_rows";
}

pub struct RopeAtPosBatchedKernel;
impl crate::kernels::ShaderKernel for RopeAtPosBatchedKernel {
    const KERNEL_NAME: &'static str = "rope_at_pos_batched";
}

pub struct RopeAtPosBatchedQkKernel;
impl crate::kernels::ShaderKernel for RopeAtPosBatchedQkKernel {
    const KERNEL_NAME: &'static str = "rope_at_pos_batched_qk";
}
