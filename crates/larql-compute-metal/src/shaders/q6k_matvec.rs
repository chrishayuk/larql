//! Q6_K matrix-vector multiply — LARQL linear Q6_K layout.
//!
//! Q6_K super-block layout (256 values = 210 bytes):
//!   [0..127]    128 bytes: ql — lo4 bits, 2 per byte: ql[b] covers elements 2b and 2b+1
//!   [128..191]   64 bytes: qh — hi2 bits, 4 per byte: qh[b] covers elements 4b..4b+3
//!   [192..207]   16 bytes: int8 scales, one per 16-element group
//!   [208..209]    2 bytes: f16 super-block scale d
//!
//! Element i: lo4 = (ql[i/2] >> 4*(i&1)) & 0xF;  hi2 = (qh[i/4] >> 2*(i%4)) & 0x3
//! Weight: d * sc[i/16] * (lo4 | hi2<<4) - 32
//!
//! **Key optimisations vs the previous all-lanes-per-superblock approach:**
//!
//! 1. **Inter-superblock interleaving**: `ix = lane & 1` splits 32 lanes into
//!    two groups. ix=0 processes superblocks 0,2,4,...; ix=1 processes 1,3,5,...
//!    Adjacent lanes read from different 210-byte memory regions simultaneously,
//!    letting the DRAM controller serve two banks in parallel.
//!
//! 2. **X preloading**: 16 X reads (4 per pass × 4 passes) are issued
//!    before ANY weight byte reads, hiding L2 latency behind weight fetches.
//!
//! 3. **Deferred scaling**: accumulate one unscaled sum per 4-element group,
//!    then apply `d * sc[j]` once — 4× fewer scale multiplications vs
//!    the previous per-element approach.
//!
//! 4. **Reduced TG size** (ROWS_PER_TG=4, 128 threads): halves register
//!    pressure vs the previous 256-thread design, allowing 2× more concurrent
//!    TGs on M3 Max for better LPDDR5X latency hiding.
//!
//! Each tid (0..15) within an ix-group handles 4 passes × 4 elements = 16
//! elements per superblock at bases {tid*4, tid*4+64, tid*4+128, tid*4+192}.
//! All 16 tids together cover all 256 elements. ✓

pub const SHADER: &str = r#"
constant uint Q6K_ROWS_PER_TG = 4;
constant uint Q6K_BLOCK_SIZE  = 210;

kernel void q6k_matvec(
    device const uchar*  W6K   [[buffer(0)]],
    device const float*  X     [[buffer(1)]],
    device float*        out   [[buffer(2)]],
    constant uint&       N     [[buffer(3)]],
    constant uint&       K     [[buffer(4)]],
    uint tg_id     [[threadgroup_position_in_grid]],
    uint lane      [[thread_index_in_simdgroup]],
    uint sg_id     [[simdgroup_index_in_threadgroup]])
{
    uint row_idx = tg_id * Q6K_ROWS_PER_TG + sg_id;
    if (row_idx >= N) return;

    const uint superblocks   = K / 256u;
    const uint bytes_per_row = superblocks * Q6K_BLOCK_SIZE;
    device const uchar* row  = W6K + row_idx * bytes_per_row;

    // Lane decomposition: ix splits 32 lanes into two interleaved-superblock
    // groups; tid is the position within each 16-lane group.
    const uint ix  = lane & 1u;   // 0 or 1
    const uint tid = lane >> 1u;  // 0..15

    // Base element index for this tid within a superblock.
    // 4 consecutive elements share one qh byte and one scale entry.
    const uint base    = tid << 2u;      // 0,4,8,...,60
    const uint sc_base = tid >> 2u;      // 0 for tid=0..3, 1 for 4..7, ..., 3 for 12..15

    float acc = 0.0f;

    // ix=0 processes superblocks 0,2,4,...; ix=1 processes 1,3,5,...
    // Adjacent lanes in the simdgroup read from different 210-byte regions.
    for (uint i = ix; i < superblocks; i += 2u) {
        device const uchar* block = row + i * Q6K_BLOCK_SIZE;
        device const uchar* ql   = block;
        device const uchar* qh   = block + 128u;
        device const char*  sc   = (device const char*)(block + 192u);
        ushort d_bits = ushort(block[208]) | (ushort(block[209]) << 8u);
        float  d = decode_f16_metal(d_bits);

        // Preload all 16 X values for the 4 passes before reading any weight
        // bytes. Explicit preload lets the GPU pipeline X fetches in parallel
        // with the upcoming ql/qh/sc reads.
        const uint xb = i * 256u + base;
        float xl[16];
        xl[ 0] = X[xb      ]; xl[ 1] = X[xb +  1u];
        xl[ 2] = X[xb +  2u]; xl[ 3] = X[xb +  3u];
        xl[ 4] = X[xb + 64u]; xl[ 5] = X[xb + 65u];
        xl[ 6] = X[xb + 66u]; xl[ 7] = X[xb + 67u];
        xl[ 8] = X[xb +128u]; xl[ 9] = X[xb +129u];
        xl[10] = X[xb +130u]; xl[11] = X[xb +131u];
        xl[12] = X[xb +192u]; xl[13] = X[xb +193u];
        xl[14] = X[xb +194u]; xl[15] = X[xb +195u];

        // 4 passes, each handling 4 consecutive elements at stride 64.
        // Per pass: 2 ql bytes + 1 qh byte → 4 dequant values.
        // Scale applied once per 4-element group (deferred, 4× cheaper).
        // sc_base + {0,4,8,12} are the 4 group scale indices.

        // Pass 0: elements base+0..3 (scale group sc_base+0)
        {
            const uint b = base;
            uchar la = ql[b >> 1u], lb = ql[(b >> 1u) + 1u], hi = qh[b >> 2u];
            float _sc = d * float(sc[sc_base + 0u]);
            acc += _sc * (
                float((char)((la & 0x0Fu) | ((hi & 0x03u) << 4u)) - 32) * xl[ 0] +
                float((char)(((la >> 4u) & 0x0Fu) | ((hi & 0x0Cu) << 2u)) - 32) * xl[ 1] +
                float((char)((lb & 0x0Fu) | ((hi & 0x30u))) - 32) * xl[ 2] +
                float((char)(((lb >> 4u) & 0x0Fu) | ((hi & 0xC0u) >> 2u)) - 32) * xl[ 3]);
        }

        // Pass 1: elements base+64..67 (scale group sc_base+4)
        {
            const uint b = base + 64u;
            uchar la = ql[b >> 1u], lb = ql[(b >> 1u) + 1u], hi = qh[b >> 2u];
            float _sc = d * float(sc[sc_base + 4u]);
            acc += _sc * (
                float((char)((la & 0x0Fu) | ((hi & 0x03u) << 4u)) - 32) * xl[ 4] +
                float((char)(((la >> 4u) & 0x0Fu) | ((hi & 0x0Cu) << 2u)) - 32) * xl[ 5] +
                float((char)((lb & 0x0Fu) | ((hi & 0x30u))) - 32) * xl[ 6] +
                float((char)(((lb >> 4u) & 0x0Fu) | ((hi & 0xC0u) >> 2u)) - 32) * xl[ 7]);
        }

        // Pass 2: elements base+128..131 (scale group sc_base+8)
        {
            const uint b = base + 128u;
            uchar la = ql[b >> 1u], lb = ql[(b >> 1u) + 1u], hi = qh[b >> 2u];
            float _sc = d * float(sc[sc_base + 8u]);
            acc += _sc * (
                float((char)((la & 0x0Fu) | ((hi & 0x03u) << 4u)) - 32) * xl[ 8] +
                float((char)(((la >> 4u) & 0x0Fu) | ((hi & 0x0Cu) << 2u)) - 32) * xl[ 9] +
                float((char)((lb & 0x0Fu) | ((hi & 0x30u))) - 32) * xl[10] +
                float((char)(((lb >> 4u) & 0x0Fu) | ((hi & 0xC0u) >> 2u)) - 32) * xl[11]);
        }

        // Pass 3: elements base+192..195 (scale group sc_base+12)
        {
            const uint b = base + 192u;
            uchar la = ql[b >> 1u], lb = ql[(b >> 1u) + 1u], hi = qh[b >> 2u];
            float _sc = d * float(sc[sc_base + 12u]);
            acc += _sc * (
                float((char)((la & 0x0Fu) | ((hi & 0x03u) << 4u)) - 32) * xl[12] +
                float((char)(((la >> 4u) & 0x0Fu) | ((hi & 0x0Cu) << 2u)) - 32) * xl[13] +
                float((char)((lb & 0x0Fu) | ((hi & 0x30u))) - 32) * xl[14] +
                float((char)(((lb >> 4u) & 0x0Fu) | ((hi & 0xC0u) >> 2u)) - 32) * xl[15]);
        }
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
    const KERNEL_NAME: &'static str = "q6k_matvec";
    const ROWS_PER_TG: u64 = ROWS_PER_TG;
    const THREADS_PER_TG: u64 = THREADS_PER_TG;
}
