//! Q4_K grouped-expert matvec — the sibling of `q6k_grouped_experts`.
//!
//! The engine's MoE path (`moe_dispatch.rs`) is Q4_K, and its **down**
//! projection is the one still running `top_k` separate dispatches. Gate/up is
//! already grouped. So this kernel is what the integration actually needs.
//!
//! Addressing only: the reduction body is copied verbatim from `q4k_matvec`,
//! including the deferred `scale*dot - dmin*sumy` formula, so output must match
//! the per-expert path **bit-exactly**. Two things change —
//!
//!   - an offset table selects each slot's payload, so the staged expert
//!     weights can live in one buffer instead of `top_k` separate ones;
//!   - `XSTRIDE` selects the input regime: `0` shares one vector across slots
//!     (gate/up), `K` gives each slot its own row (down, where every expert
//!     consumes its own intermediate activation). Wrong stride yields a
//!     plausible number from another expert's activation rather than an error,
//!     which is why it is an explicit argument.
//!
//! Grid is `(row_tiles, n_selected)`; the y axis is the parallelism the model
//! already supplies and the old loop discarded.

pub const ROWS_PER_TG: u64 = 4;
pub const THREADS_PER_TG: u64 = 128;

pub const SHADER: &str = r#"
constant uint Q4KGE_ROWS_PER_TG = 4;
constant uint Q4KGE_BLOCK_SIZE  = 144;

kernel void q4k_grouped_experts(
    device const uchar*  W4K     [[buffer(0)]],
    device const uint*   offsets [[buffer(1)]],
    device const float*  X       [[buffer(2)]],
    device float*        out     [[buffer(3)]],
    constant uint&       N       [[buffer(4)]],
    constant uint&       K       [[buffer(5)]],
    constant uint&       XSTRIDE [[buffer(6)]],
    uint2 tg_id    [[threadgroup_position_in_grid]],
    uint lane      [[thread_index_in_simdgroup]],
    uint sg_id     [[simdgroup_index_in_threadgroup]])
{
    const uint slot = tg_id.y;
    uint row_idx = tg_id.x * Q4KGE_ROWS_PER_TG + sg_id;
    if (row_idx >= N) return;

    const uint superblocks   = K / 256u;
    const uint bytes_per_row = superblocks * Q4KGE_BLOCK_SIZE;
    device const uchar* row_w = W4K + (ulong)offsets[slot] + (ulong)row_idx * bytes_per_row;
    device const float* Xs = X + (ulong)slot * XSTRIDE;

    // 2-way inter-superblock interleaving.
    // Adjacent lanes in the simdgroup read from different 144-byte superblock
    // regions simultaneously — two DRAM banks served in parallel.
    const uint ix  = lane & 1u;    // 0 or 1
    const uint tid = lane >> 1u;   // 0..15
    const uint j   = tid >> 1u;    // 0..7: which sub-block within superblock
    const uint sh  = tid & 1u;     // 0 or 1: first/last 16 of the 32-elem sub-block

    // Which 32-byte nibble group sub-block j belongs to, and which nibble half.
    const bool  hi    = (j & 1u) != 0u;  // lo nibble (j even) or hi nibble (j odd)
    const uint  group = j >> 1u;          // 0..3

    float acc = 0.0f;

    for (uint sb = ix; sb < superblocks; sb += 2u) {
        device const uchar* block = row_w + sb * Q4KGE_BLOCK_SIZE;
        ushort d_bits    = ushort(block[0]) | (ushort(block[1]) << 8u);
        ushort dmin_bits = ushort(block[2]) | (ushort(block[3]) << 8u);
        float d    = decode_f16_metal(d_bits);
        float dmin = decode_f16_metal(dmin_bits);

        // Unpack the 6-bit scale and 6-bit min for sub-block j.
        device const uchar* sb_bytes = block + 4u;
        uint sc, mn;
        if (j < 4u) {
            sc = uint(sb_bytes[j])      & 0x3Fu;
            mn = uint(sb_bytes[j + 4u]) & 0x3Fu;
        } else {
            sc = (uint(sb_bytes[j + 4u]) & 0x0Fu) | ((uint(sb_bytes[j - 4u]) >> 6u) << 4u);
            mn = (uint(sb_bytes[j + 4u]) >> 4u)    | ((uint(sb_bytes[j])      >> 6u) << 4u);
        }
        float scale = d * float(sc);
        float mmin  = dmin * float(mn);

        // Preload 16 X values into registers BEFORE loading weight bytes.
        // Separating loads from compute lets the GPU pipeline both in parallel.
        // Full unroll keeps xl[] indices compile-time constant → register-resident.
        const uint x_base = sb * 256u + j * 32u + sh * 16u;
        float xl[16];
        _Pragma("clang loop unroll(full)")
        for (uint l = 0u; l < 16u; l++) { xl[l] = Xs[x_base + l]; }

        // Weight nibble bytes for this lane's 16-element slice.
        // group*32 selects the 32-byte nibble group; sh*16 selects the 16-byte half.
        device const uchar* qs = block + 16u + group * 32u + sh * 16u;

        // Precompute sum of X values for the min-correction term.
        // Separating this from the FMA chain lets the compiler schedule
        // the dot loop as a pure FMA sequence without interleaved adds.
        float sumy = 0.0f;
        _Pragma("clang loop unroll(full)")
        for (uint l = 0u; l < 16u; l++) { sumy += xl[l]; }

        // Pure dot product — uninterrupted FMA chain.
        float dot_acc = 0.0f;
        _Pragma("clang loop unroll(full)")
        for (uint l = 0u; l < 16u; l++) {
            uchar byte = qs[l];
            float nib = hi ? float((byte >> 4u) & 0x0Fu) : float(byte & 0x0Fu);
            dot_acc = fma(nib, xl[l], dot_acc);
        }
        // Q4_K deferred formula: scale*dot - dmin*sum_x
        acc += scale * dot_acc - mmin * sumy;
    }

    acc = simd_sum(acc);
    if (lane == 0u) out[slot * N + row_idx] = acc;
}
"#;

/// Marker for the kernel-handle binding. See `metal::kernel::TiledKernel`.
pub struct Kernel;
impl crate::kernels::TiledKernel for Kernel {
    const KERNEL_NAME: &'static str = "q4k_grouped_experts";
    const ROWS_PER_TG: u64 = ROWS_PER_TG;
    const THREADS_PER_TG: u64 = THREADS_PER_TG;
}
