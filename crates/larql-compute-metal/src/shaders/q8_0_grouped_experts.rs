//! Q8_0 grouped-expert matvec — all selected experts in ONE dispatch.
//!
//! The precision ladder's middle rung: canonical ggml Q8_0 (34 bytes per
//! 32-element block — little-endian f16 scale + 32 signed int8), 8.5 bpw
//! against BF16's 16 and Q6_K's 6.5625. It exists because the depth sweep
//! placed the strict whole-layer Q6_K boundary at the FINAL MoE layer
//! only; the open question is how much of the stack an ~8-bit
//! representation admits under the same frozen contract, and that
//! question cannot be asked until the grouped path can execute one.
//!
//! Same ABI as `q6k_grouped_experts` — bank + byte-offset table + shared
//! or per-slot X — so `grouped_handle_for` selects it as a handle swap,
//! never a different lowering. Grid, tiling and output layout are
//! identical: `(row_tiles, n_selected)`, one simdgroup per output row,
//! `out[slot * N + row]`.
//!
//! Unlike Q4/Q6 there is no nibble unpacking: each weight is one byte,
//! decoded as `d * q`. Lanes stride the row's blocks; each lane decodes
//! its block's scale once and accumulates the 32-element sub-dot in f32.

/// Output rows per threadgroup — one simdgroup each, matching the other
/// grouped kernels so occupancy comparisons stay like-for-like.
pub const ROWS_PER_TG: u64 = 4;
pub const THREADS_PER_TG: u64 = 128;

pub const SHADER: &str = r#"
constant uint Q80G_ROWS_PER_TG = 4;
// Canonical ggml Q8_0: f16 scale (2 bytes) + 32 int8 quants.
constant uint Q80G_BLOCK_BYTES = 34;

kernel void q8_0_grouped_experts(
    device const uchar*  W8      [[buffer(0)]],  // all expert payloads
    device const uint*   offsets [[buffer(1)]],  // [n_sel] byte offset per slot
    device const float*  X       [[buffer(2)]],  // shared [K], or [n_sel, K]
    device float*        out     [[buffer(3)]],  // [n_sel, N] per-expert outputs
    constant uint&       N       [[buffer(4)]],
    constant uint&       K       [[buffer(5)]],
    // 0 = every slot reads the same X (gate/up: one hidden state for all
    // experts). K = each slot reads its own X (down: each expert consumes
    // its OWN intermediate activation). Explicit for the same reason as
    // the Q6_K sibling: getting it wrong silently computes the wrong
    // expert's product.
    constant uint&       XSTRIDE [[buffer(6)]],
    uint2 tg_id    [[threadgroup_position_in_grid]],
    uint  lane     [[thread_index_in_simdgroup]],
    uint  sg_id    [[simdgroup_index_in_threadgroup]])
{
    // tg_id.y selects the expert slot; tg_id.x the row tile within it.
    const uint slot    = tg_id.y;
    const uint row_idx = tg_id.x * Q80G_ROWS_PER_TG + sg_id;
    if (row_idx >= N) { return; }

    const uint blocks        = K / 32u;
    const uint bytes_per_row = blocks * Q80G_BLOCK_BYTES;
    device const uchar* row = W8 + offsets[slot] + row_idx * bytes_per_row;
    device const float* Xs  = X + (ulong)slot * XSTRIDE;

    float acc = 0.0f;
    for (uint b = lane; b < blocks; b += 32u) {
        device const uchar* blk = row + b * Q80G_BLOCK_BYTES;
        ushort d_bits = ushort(blk[0]) | (ushort(blk[1]) << 8u);
        float  d = decode_f16_metal(d_bits);
        device const char* q = (device const char*)(blk + 2u);
        const uint xb = b * 32u;
        float sum = 0.0f;
        for (uint j = 0u; j < 32u; ++j) {
            sum += float(q[j]) * Xs[xb + j];
        }
        acc += d * sum;
    }

    acc = simd_sum(acc);
    if (lane == 0u) { out[slot * N + row_idx] = acc; }
}
"#;

/// Marker for the kernel-handle binding. See `metal::kernel::TiledKernel`.
pub struct Kernel;
impl crate::kernels::TiledKernel for Kernel {
    const KERNEL_NAME: &'static str = "q8_0_grouped_experts";
    const ROWS_PER_TG: u64 = ROWS_PER_TG;
    const THREADS_PER_TG: u64 = THREADS_PER_TG;
}
