//! Q4_K fused gate+up — 8-simdgroup-per-TG variant.
//!
//! Identical math to [`q4k_ffn_gate_up`], only the threadgroup geometry
//! changes:
//!
//! - Production kernel: `ROWS_PER_TG=4`, `THREADS_PER_TG=128` (4 simdgroups)
//! - This variant:    `ROWS_PER_TG=8`, `THREADS_PER_TG=256` (8 simdgroups)
//!
//! `nr0=1` (one output row per simdgroup) is preserved, so per-thread
//! register footprint is unchanged — sidesteps the register-pressure
//! regression seen with `nr0>1` in earlier experiments (auto-memory
//! 2026-04-19: "N_DST=4 caused 24× regression, N_DST=2 caused ~10%").
//!
//! **Hypothesis under test**: doubling threads per TG increases
//! within-TG latency hiding (more concurrent simdgroups can hide DRAM
//! latency for each other) without forcing per-thread register
//! pressure. We currently sit at 274 GB/s = 68% of M3 Max LPDDR5X peak
//! (~400 GB/s); ollama's hand-tuned kernels are estimated at 85%+.
//! Bigger TGs should help if the gap is occupancy-bound.
//!
//! **Risk**: more threads per TG also halves the maximum concurrent TG
//! count on the GPU (each TG holds more SRAM/registers). The 2026-04-26
//! attempt at `ROWS_PER_TG=2 / 64 threads/TG` regressed for the inverse
//! reason — fewer TGs means worse latency hiding **across** TGs. The
//! optimal point is empirical; this variant explores the upward
//! direction we haven't tried.
//!
//! Parity contract: output must match the production kernel exactly
//! (same math, same lane→row mapping within each simdgroup, only
//! more simdgroups dispatched per TG). Tested by
//! `q4k_ffn_gate_up_8sg_matches_4sg` in the test file.

pub const SHADER: &str = r#"
constant uint Q4K_GU_8SG_ROWS_PER_TG = 8;
constant uint Q4K_GU_8SG_BLOCK_SIZE  = 144;

kernel void q4k_ffn_gate_up_8sg(
    device const uchar*  Wg    [[buffer(0)]],
    device const uchar*  Wu    [[buffer(1)]],
    device const float*  X     [[buffer(2)]],
    device float*        G_out [[buffer(3)]],
    device float*        U_out [[buffer(4)]],
    constant uint&       N     [[buffer(5)]],
    constant uint&       K     [[buffer(6)]],
    uint tg_id     [[threadgroup_position_in_grid]],
    uint lane      [[thread_index_in_simdgroup]],
    uint sg_id     [[simdgroup_index_in_threadgroup]])
{
    uint tgs_per_mat = (N + Q4K_GU_8SG_ROWS_PER_TG - 1u) / Q4K_GU_8SG_ROWS_PER_TG;
    bool is_up  = (tg_id >= tgs_per_mat);
    uint mat_tg = is_up ? (tg_id - tgs_per_mat) : tg_id;

    uint row_idx = mat_tg * Q4K_GU_8SG_ROWS_PER_TG + sg_id;
    if (row_idx >= N) return;

    device const uchar* W      = is_up ? Wu : Wg;
    device float*       out_buf = is_up ? U_out : G_out;

    const uint superblocks   = K / 256u;
    const uint bytes_per_row = superblocks * Q4K_GU_8SG_BLOCK_SIZE;
    device const uchar* row_w = W + row_idx * bytes_per_row;

    const uint ix  = lane & 1u;
    const uint tid = lane >> 1u;
    const uint j   = tid >> 1u;
    const uint sh  = tid & 1u;
    const bool hi    = (j & 1u) != 0u;
    const uint group = j >> 1u;

    float acc = 0.0f;

    for (uint sb = ix; sb < superblocks; sb += 2u) {
        device const uchar* block = row_w + sb * Q4K_GU_8SG_BLOCK_SIZE;
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
    if (lane == 0u) out_buf[row_idx] = acc;
}
"#;

pub const ROWS_PER_TG: u64 = 8;
pub const THREADS_PER_TG: u64 = 256;

/// Marker for the kernel-handle binding. See `metal::kernel::TiledKernel`.
pub struct Kernel;
impl crate::kernels::TiledKernel for Kernel {
    const KERNEL_NAME: &'static str = "q4k_ffn_gate_up_8sg";
    const ROWS_PER_TG: u64 = ROWS_PER_TG;
    const THREADS_PER_TG: u64 = THREADS_PER_TG;
}
