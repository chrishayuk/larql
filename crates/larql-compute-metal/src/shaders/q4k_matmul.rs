//! Q4_K matrix-matrix multiply (gemm) — `C[m, n] = sum_k W[n, k] * X[m, k]`.
//!
//! Companion to [`q4k_matvec`] for the prefill path. The matvec processes
//! one input position per dispatch; this kernel processes `M` positions in
//! a single dispatch and **amortises the Q4_K dequant cost across M**.
//!
//! Layout:
//!   - W: `[N, K]` Q4_K row-major (one 144-byte super-block per 256 cols)
//!   - X: `[M, K]` f32 row-major (`M` = seq_len for prefill, 1 for decode)
//!   - C: `[M, N]` f32 row-major (output for all M positions for all N rows)
//!
//! Dispatch geometry:
//!   - `tg_id.y` covers `N` in chunks of `ROWS_PER_TG = 4` (one simdgroup
//!     per row, matching `q4k_matvec`)
//!   - `tg_id.x` covers `M` in chunks of `COLS_PER_TG = 4` (per-thread
//!     accumulator array of size 4 — keeps register pressure within
//!     budget; M=1 still works at zero amortisation cost)
//!   - Each lane reads its sub-block half nibbles ONCE per super-block,
//!     then runs `COLS_PER_TG` dot products against `COLS_PER_TG`
//!     consecutive X positions.
//!
//! Amortisation: weight dequant + scale/min unpack happen once per
//! super-block per simdgroup; the X reads + dot loop run COLS_PER_TG
//! times. For seq_len=18 prompt tokens that's 4-5× fewer dequant passes.
//!
//! When M is not a multiple of COLS_PER_TG, the tail TG handles
//! `valid_cols = min(COLS_PER_TG, M - m_base)` positions; out-of-range
//! lanes accumulate into `acc[m]` slots that are simply not written back.
//!
//! Parity contract: `q4k_matmul(W, X, M, N, K)` equals stacking
//! `q4k_matvec(W, X[m..], N, K)` for `m=0..M`. The matmul kernel must NEVER
//! produce a different numerical result — only the same number computed
//! with fewer dequant passes. Validated by
//! `q4k_matmul_matches_stacked_matvec` in `metal/trait_impl/matmul.rs`.

pub const SHADER: &str = r#"
constant uint Q4KMM_ROWS_PER_TG = 4;
constant uint Q4KMM_COLS_PER_TG = 4;
constant uint Q4KMM_BLOCK_SIZE  = 144;

kernel void q4k_matmul(
    device const uchar*  W4K   [[buffer(0)]],
    device const float*  X     [[buffer(1)]],
    device float*        out   [[buffer(2)]],
    constant uint&       N     [[buffer(3)]],   // output rows (W rows)
    constant uint&       K     [[buffer(4)]],   // hidden / inner dim
    constant uint&       M     [[buffer(5)]],   // input positions
    uint2 tg_id    [[threadgroup_position_in_grid]],
    uint  lane     [[thread_index_in_simdgroup]],
    uint  sg_id    [[simdgroup_index_in_threadgroup]])
{
    uint row_idx = tg_id.y * Q4KMM_ROWS_PER_TG + sg_id;
    if (row_idx >= N) return;

    uint m_base = tg_id.x * Q4KMM_COLS_PER_TG;
    if (m_base >= M) return;
    uint cols_in_tg = min(Q4KMM_COLS_PER_TG, M - m_base);

    const uint superblocks   = K / 256u;
    const uint bytes_per_row = superblocks * Q4KMM_BLOCK_SIZE;
    device const uchar* row_w = W4K + row_idx * bytes_per_row;

    // Same lane partitioning as q4k_matvec: 2-way inter-superblock
    // interleave keeps DRAM banks busy across adjacent lanes.
    const uint ix  = lane & 1u;
    const uint tid = lane >> 1u;
    const uint j   = tid >> 1u;
    const uint sh  = tid & 1u;
    const bool  hi    = (j & 1u) != 0u;
    const uint  group = j >> 1u;

    // Per-position partial accumulators. Q4KMM_COLS_PER_TG = 4 → 4 floats
    // per thread → 16 bytes register footprint; fine on M3 Max.
    float acc[Q4KMM_COLS_PER_TG];
    for (uint m = 0u; m < Q4KMM_COLS_PER_TG; m++) acc[m] = 0.0f;

    for (uint sb = ix; sb < superblocks; sb += 2u) {
        device const uchar* block = row_w + sb * Q4KMM_BLOCK_SIZE;
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

        // Dequantise the 16 nibbles for this lane's slice ONCE, then
        // multiply against COLS_PER_TG X positions. This is the
        // amortisation: q4k_matvec recomputes `nib` per dispatch
        // (= per position); we recompute it once per super-block.
        device const uchar* qs = block + 16u + group * 32u + sh * 16u;
        float nibs[16];
        _Pragma("clang loop unroll(full)")
        for (uint l = 0u; l < 16u; l++) {
            uchar byte = qs[l];
            nibs[l] = hi ? float((byte >> 4u) & 0x0Fu) : float(byte & 0x0Fu);
        }

        const uint x_sb_off = sb * 256u + j * 32u + sh * 16u;

        // Process up to COLS_PER_TG positions per super-block. The
        // compile-time COLS_PER_TG=4 unroll lets the compiler issue
        // independent FMA chains in parallel.
        _Pragma("clang loop unroll(full)")
        for (uint m = 0u; m < Q4KMM_COLS_PER_TG; m++) {
            // `acc[m]` slots beyond `cols_in_tg` are never written to
            // `out`, so we don't need to mask the FMA chain — but we
            // do need to read X from a valid position to avoid OOB.
            uint pos = (m < cols_in_tg) ? (m_base + m) : m_base;
            uint x_off = pos * K + x_sb_off;

            float xl[16];
            float sumy = 0.0f;
            _Pragma("clang loop unroll(full)")
            for (uint l = 0u; l < 16u; l++) {
                xl[l] = X[x_off + l];
                sumy += xl[l];
            }

            float dot_acc = 0.0f;
            _Pragma("clang loop unroll(full)")
            for (uint l = 0u; l < 16u; l++) {
                dot_acc = fma(nibs[l], xl[l], dot_acc);
            }
            acc[m] += scale * dot_acc - mmin * sumy;
        }
    }

    // Reduce across lanes for each accumulator slot.
    _Pragma("clang loop unroll(full)")
    for (uint m = 0u; m < Q4KMM_COLS_PER_TG; m++) {
        float reduced = simd_sum(acc[m]);
        if (lane == 0u && m < cols_in_tg) {
            uint pos = m_base + m;
            out[pos * N + row_idx] = reduced;
        }
    }
}
"#;

pub const ROWS_PER_TG: u64 = 4;
pub const COLS_PER_TG: u64 = 4;
pub const THREADS_PER_TG: u64 = 128;

/// Marker for the kernel-handle binding. See `metal::kernel::TiledKernel`.
pub struct Kernel;
impl crate::kernels::TiledKernel for Kernel {
    const KERNEL_NAME: &'static str = "q4k_matmul";
    const ROWS_PER_TG: u64 = ROWS_PER_TG;
    const THREADS_PER_TG: u64 = THREADS_PER_TG;
}
