//! Q4_K matrix-vector multiply — GGUF 144-byte block layout.
//!
//! Block layout:
//!   [0..2]    f16 `d`     (super-block scale)
//!   [2..4]    f16 `dmin`  (super-block min scale)
//!   [4..16]   12 bytes of packed 6-bit scales + 6-bit mins (8 of each)
//!   [16..144] 128 bytes of 4-bit nibbles (256 values across 8 sub-blocks)
//!
//! Sub-block structure (32 values each, 8 per super-block):
//!   Sub-block j (j=0..7): nibbles at block+16+group*32 where group=j/2.
//!   Even j → lo nibbles of that 32-byte group; odd j → hi nibbles.
//!
//! **Parallelism — 2-way inter-superblock interleaving (same strategy as q6k_matvec):**
//!
//! `ix = lane & 1` splits 32 lanes into two groups:
//!   ix=0 → processes superblocks 0,2,4,...  ix=1 → superblocks 1,3,5,...
//! Adjacent lanes in the simdgroup read from DIFFERENT 144-byte superblock
//! regions simultaneously, letting the DRAM controller serve two banks in
//! parallel (vs the old sub-block-stride approach where stride-32 lanes hit
//! the same 144-byte block before moving on).
//!
//! `tid = lane >> 1` (0..15) partitions work within each superblock:
//!   j  = tid >> 1 (0..7): which of the 8 sub-blocks
//!   sh = tid & 1  (0/1):  first or last 16 elements of that sub-block
//!
//! X preloading: 16 values loaded into `xl[16]` registers before any weight
//! byte reads, pipelining X fetches behind block/scale reads.
//!
//! ROWS_PER_TG=4 (128 threads): halves the per-TG register footprint vs the
//! previous 256-thread design, allowing more concurrent TGs for latency hiding.

pub const SHADER: &str = r#"
constant uint Q4K_ROWS_PER_TG = 4;
constant uint Q4K_BLOCK_SIZE  = 144;

kernel void q4k_matvec(
    device const uchar*  W4K   [[buffer(0)]],
    device const float*  X     [[buffer(1)]],
    device float*        out   [[buffer(2)]],
    constant uint&       N     [[buffer(3)]],
    constant uint&       K     [[buffer(4)]],
    uint tg_id     [[threadgroup_position_in_grid]],
    uint lane      [[thread_index_in_simdgroup]],
    uint sg_id     [[simdgroup_index_in_threadgroup]])
{
    uint row_idx = tg_id * Q4K_ROWS_PER_TG + sg_id;
    if (row_idx >= N) return;

    const uint superblocks   = K / 256u;
    const uint bytes_per_row = superblocks * Q4K_BLOCK_SIZE;
    device const uchar* row_w = W4K + row_idx * bytes_per_row;

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
        device const uchar* block = row_w + sb * Q4K_BLOCK_SIZE;
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
        for (uint l = 0u; l < 16u; l++) { xl[l] = X[x_base + l]; }

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
    if (lane == 0u) out[row_idx] = acc;
}
"#;

pub const ROWS_PER_TG: u64 = 4;
pub const THREADS_PER_TG: u64 = 128;

/// Marker for the kernel-handle binding. See `metal::kernel::TiledKernel`.
pub struct Kernel;
impl crate::kernels::TiledKernel for Kernel {
    const KERNEL_NAME: &'static str = "q4k_matvec";
    const ROWS_PER_TG: u64 = ROWS_PER_TG;
    const THREADS_PER_TG: u64 = THREADS_PER_TG;
}
