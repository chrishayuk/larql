//! Q4_K matrix-vector multiply — 8-simdgroup-per-TG variant.
//!
//! Identical math to [`q4k_matvec`], only the threadgroup geometry
//! changes:
//!
//! - Production kernel: `ROWS_PER_TG=4`, `THREADS_PER_TG=128` (4 simdgroups)
//! - This variant:    `ROWS_PER_TG=8`, `THREADS_PER_TG=256` (8 simdgroups)
//!
//! `nr0=1` is preserved — same per-thread register footprint.
//!
//! **Why this kernel specifically**: production-batched profiler shows
//! q4k_matvec (Wo, K=8192) running at 220 GB/s = **55% of LPDDR5X
//! peak** — the most under-utilized of all the production matvecs
//! (q6k at 77%, gate+up at 68%, lm_head at 92%). The same 8sg geometry
//! change that landed +2.1% end-to-end on gate+up should produce an
//! even bigger win here, since Wo has the largest bandwidth headroom.
//!
//! Parity contract: bit-equal output to the 4sg kernel.
//!
//! ## Retention rationale (ADR-017)
//!
//! **Status**: production-active by default since 2026-04-28
//! (`q4k_matvec_pipeline` resolves to this variant; `LARQL_Q4K_MATVEC_8SG=0`
//! falls back to the 4sg twin in `q4k_matvec.rs`).
//!
//! Both 4sg and 8sg variants are kept: 8sg as the default (won the
//! end-to-end A/B), 4sg as the explicit fallback for cross-hardware
//! validation and diagnostic A/B.

pub const SHADER: &str = r#"
constant uint Q4K_8SG_ROWS_PER_TG = 8;
constant uint Q4K_8SG_BLOCK_SIZE  = 144;

kernel void q4k_matvec_8sg(
    device const uchar*  W4K   [[buffer(0)]],
    device const float*  X     [[buffer(1)]],
    device float*        out   [[buffer(2)]],
    constant uint&       N     [[buffer(3)]],
    constant uint&       K     [[buffer(4)]],
    uint tg_id     [[threadgroup_position_in_grid]],
    uint lane      [[thread_index_in_simdgroup]],
    uint sg_id     [[simdgroup_index_in_threadgroup]])
{
    uint row_idx = tg_id * Q4K_8SG_ROWS_PER_TG + sg_id;
    if (row_idx >= N) return;

    const uint superblocks   = K / 256u;
    const uint bytes_per_row = superblocks * Q4K_8SG_BLOCK_SIZE;
    device const uchar* row_w = W4K + row_idx * bytes_per_row;

    const uint ix  = lane & 1u;
    const uint tid = lane >> 1u;
    const uint j   = tid >> 1u;
    const uint sh  = tid & 1u;
    const bool  hi    = (j & 1u) != 0u;
    const uint  group = j >> 1u;

    float acc = 0.0f;

    for (uint sb = ix; sb < superblocks; sb += 2u) {
        device const uchar* block = row_w + sb * Q4K_8SG_BLOCK_SIZE;
        ushort d_bits    = ushort(block[0]) | (ushort(block[1]) << 8u);
        ushort dmin_bits = ushort(block[2]) | (ushort(block[3]) << 8u);
        float d    = decode_f16_metal(d_bits);
        float dmin = decode_f16_metal(dmin_bits);

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

        const uint x_base = sb * 256u + j * 32u + sh * 16u;
        float xl[16];
        _Pragma("clang loop unroll(full)")
        for (uint l = 0u; l < 16u; l++) { xl[l] = X[x_base + l]; }

        device const uchar* qs = block + 16u + group * 32u + sh * 16u;

        float sumy = 0.0f;
        _Pragma("clang loop unroll(full)")
        for (uint l = 0u; l < 16u; l++) { sumy += xl[l]; }

        float dot_acc = 0.0f;
        _Pragma("clang loop unroll(full)")
        for (uint l = 0u; l < 16u; l++) {
            uchar byte = qs[l];
            float nib = hi ? float((byte >> 4u) & 0x0Fu) : float(byte & 0x0Fu);
            dot_acc = fma(nib, xl[l], dot_acc);
        }
        acc += scale * dot_acc - mmin * sumy;
    }

    acc = simd_sum(acc);
    if (lane == 0u) out[row_idx] = acc;
}
"#;

pub const ROWS_PER_TG: u64 = 8;
pub const THREADS_PER_TG: u64 = 256;

/// Marker for the kernel-handle binding.
pub struct Kernel;
impl crate::kernels::TiledKernel for Kernel {
    const KERNEL_NAME: &'static str = "q4k_matvec_8sg";
    const ROWS_PER_TG: u64 = ROWS_PER_TG;
    const THREADS_PER_TG: u64 = THREADS_PER_TG;
}
