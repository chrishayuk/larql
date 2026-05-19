//! Q6_K matrix-vector multiply — 8-simdgroup-per-TG variant.
//!
//! Identical math to [`q6k_matvec`], only the threadgroup geometry
//! changes:
//!
//! - Production kernel: `ROWS_PER_TG=4`, `THREADS_PER_TG=128` (4 simdgroups)
//! - This variant:    `ROWS_PER_TG=8`, `THREADS_PER_TG=256` (8 simdgroups)
//!
//! `nr0=1` (one output row per simdgroup) is preserved, so per-thread
//! register footprint is unchanged.
//!
//! **Hypothesis under test**: doubling threads per TG increases
//! within-TG latency hiding without forcing per-thread register
//! pressure. q6k_matvec sits at 311 GB/s = 79% of M3 Max LPDDR5X peak
//! (~400 GB/s), so headroom is smaller than for q4k_ffn_gate_up which
//! was at 68%. But the same geometry change just landed +2.1% on
//! gate+up; trying the analogous knob on down is the obvious next
//! sweep.
//!
//! Parity contract: output must be bit-equal to the production kernel
//! (same math, same lane→row mapping, only TG dispatch geometry
//! changed). Tested by `q6k_matvec_8sg_matches_4sg` in the test file.
//!
//! ## Retention rationale (ADR-017)
//!
//! **Status**: opt-in via `LARQL_Q6K_8SG=1`. Default-OFF (4sg).
//!
//! Empirical 2026-04-28: 8sg kernel-isolated 1.96× speedup but
//! end-to-end at parity (slightly worse on quiet GPU: 77.6 → 77.1
//! tok/s, ≈0.08 ms/tok regression). q6k was already at 84% of
//! LPDDR5X peak; the ALU/scheduling slack 8sg exposes is too small
//! to translate end-to-end on M3 Max. Kept opt-in for callers
//! retrying on different hardware (M4 Max, future Apple GPUs with
//! larger shared cache or higher SM utilisation could reverse the
//! null result).
//!
//! **Removal trigger**: if no positive end-to-end A/B result lands
//! across the next two macOS / Apple-Silicon generations, demote.

pub const SHADER: &str = r#"
constant uint Q6K_8SG_ROWS_PER_TG = 8;
constant uint Q6K_8SG_BLOCK_SIZE  = 210;

kernel void q6k_matvec_8sg(
    device const uchar*  W6K   [[buffer(0)]],
    device const float*  X     [[buffer(1)]],
    device float*        out   [[buffer(2)]],
    constant uint&       N     [[buffer(3)]],
    constant uint&       K     [[buffer(4)]],
    uint tg_id     [[threadgroup_position_in_grid]],
    uint lane      [[thread_index_in_simdgroup]],
    uint sg_id     [[simdgroup_index_in_threadgroup]])
{
    uint row_idx = tg_id * Q6K_8SG_ROWS_PER_TG + sg_id;
    if (row_idx >= N) return;

    const uint superblocks   = K / 256u;
    const uint bytes_per_row = superblocks * Q6K_8SG_BLOCK_SIZE;
    device const uchar* row  = W6K + row_idx * bytes_per_row;

    const uint ix  = lane & 1u;
    const uint tid = lane >> 1u;

    const uint base    = tid << 2u;
    const uint sc_base = tid >> 2u;

    float acc = 0.0f;

    for (uint i = ix; i < superblocks; i += 2u) {
        device const uchar* block = row + i * Q6K_8SG_BLOCK_SIZE;
        device const uchar* ql   = block;
        device const uchar* qh   = block + 128u;
        device const char*  sc   = (device const char*)(block + 192u);
        ushort d_bits = ushort(block[208]) | (ushort(block[209]) << 8u);
        float  d = decode_f16_metal(d_bits);

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

pub const ROWS_PER_TG: u64 = 8;
pub const THREADS_PER_TG: u64 = 256;

/// Marker for the kernel-handle binding. See `metal::kernel::TiledKernel`.
pub struct Kernel;
impl crate::kernels::TiledKernel for Kernel {
    const KERNEL_NAME: &'static str = "q6k_matvec_8sg";
    const ROWS_PER_TG: u64 = ROWS_PER_TG;
    const THREADS_PER_TG: u64 = THREADS_PER_TG;
}
