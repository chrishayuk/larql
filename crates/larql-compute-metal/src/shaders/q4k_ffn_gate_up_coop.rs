//! Fused Q4_K gate+up — cooperative scale-loading variant.
//!
//! ## Retention rationale (post 2026-05-09 model-agnosticity audit)
//!
//! - **Tried**: 2026-05-01 on Gemma 3 4B (K=2560). Result: kernel-isolated
//!   neutral, end-to-end null. Pinned as ADR-015 instance #2.
//! - **Status**: Opt-in via `LARQL_GATE_UP_COOP=1`. Not deleted despite
//!   Gemma A/B loss because:
//!   - **Larger K dimensions could win.** Gemma 3 4B's K=2560 has only 8
//!     sub-blocks per super-block — the cooperative scale-load amortises
//!     across lanes but the lane-0..7 specialisation overhead dominates
//!     when there are few super-blocks. Llama 3 70B (K=8192), Gemma 4 31B
//!     (K=4096), or future larger architectures change this math.
//!   - **Different ALU/bandwidth balance** on M5+/A19 silicon could shift
//!     the trade — production is currently bandwidth-bound at 47% peak;
//!     a wider DRAM bus would expose more compute headroom for the
//!     scale-decode amortisation to matter.
//! - **Re-validation gate**: A/B against `q4k_ffn_gate_up_8sg` on a vindex
//!   with K ≥ 4096 (Gemma 4 31B or Llama 70B). Promote if batched-diag
//!   GB/s improves AND end-to-end shows direction-match (per ADR-015).
//! - **Deletion criterion**: if a clean cross-K bench shows null on every
//!   K ∈ {2560, 4096, 8192} on quiet hardware, the kernel can be deleted
//!   — but no such bench has been run yet.
//!
//! See `docs/shader-inventory.md` for the full retention rationale framework.
//!
//! ## Original kernel description
//!
//! Same Q4_K input format and output as [`q4k_ffn_gate_up`], but the
//! per-super-block sub-block scales/mins (`d * sc[0..7]` and
//! `dmin * mn[0..7]`) are computed once per simdgroup per super-block
//! by lanes 0..7 cooperatively, written to threadgroup memory, and
//! read back by all 32 lanes via the single shared `j` lookup.
//!
//! **Why this kernel exists**: per `metal/diag/kernel_profile.rs`, the
//! production `q4k_ffn_gate_up` runs at 187 GB/s (47% of M3 Max
//! LPDDR5X peak) and is flagged "COMPUTE-BOUND (K=2560 dequant
//! dominates)". Per-lane redundant work in production:
//!
//! - All 32 lanes decode the super-block `d` and `dmin` (32× redundant).
//! - 4 lanes share each `j` and each redundantly unpacks the same
//!   sub-block (sc, mn) from the 12-byte packed header (4× redundant
//!   per `j`, 8 j's per super-block ⇒ 32 unpacks total per super-block
//!   per simdgroup, only 8 of which are unique).
//!
//! Cooperative pattern (this kernel):
//!
//! - Lanes 0..7 each decode the super-block d/dmin (8× redundant —
//!   negligible vs the 32× saved on the per-lane path; avoids a
//!   `simd_broadcast` round-trip that was found to alter inner-FMA
//!   scheduling enough to flip rank-1 in earlier prototypes).
//! - Lanes 0..7 each unpack one sub-block's (sc, mn) (`lane == k`,
//!   `k = 0..7` is the sub-block index).
//! - Lanes 0..7 compute `scale_k = d * sc` and `mmin_k = dmin * mn`,
//!   write to `coeffs[sg_id*16 + k]` (scale) / `coeffs[sg_id*16 + 8 + k]`
//!   (mmin) in threadgroup memory.
//! - `threadgroup_barrier(mem_threadgroup)` flushes those writes.
//! - All 32 lanes read `scale = coeffs[sg_id*16 + j]` and
//!   `mmin = coeffs[sg_id*16 + 8 + j]` where j is the lane's owned
//!   sub-block index (4 lanes per j, 8 j's per simdgroup).
//! - Inner FMA loop runs unchanged on the broadcast values.
//!
//! Net per simdgroup per super-block: 8 d-decodes + 8 sub-block unpacks,
//! down from 32 + 32 = 64 sequence-dependent ALU ops in production.
//! Plus one threadgroup-memory barrier (cheap on Apple Silicon —
//! threadgroup memory is on-tile SRAM).
//!
//! **Parity contract**: numerically equivalent to `q4k_ffn_gate_up` up
//! to FMA-order rounding. The math expressions for `scale`, `mmin`,
//! `dot_acc`, `sumy`, and the final `acc += scale * dot_acc - mmin * sumy`
//! are bit-identical to production; only the *who-computes-what* shifts.
//! Verified by `arch_golden_gemma3_4b_gpu` continuing to emit "**Paris**"
//! and `decode_consistency_gemma3_4b{,_2steps}` continuing to pass.
//!
//! **Geometry**: 4 simdgroups per TG, 4 rows per TG, 128 threads per TG —
//! same as production `q4k_ffn_gate_up` so dispatch grid math is unchanged.

pub const SHADER: &str = r#"
constant uint Q4K_GUC_ROWS_PER_TG = 4;
constant uint Q4K_GUC_BLOCK_SIZE  = 144;
// 16 floats per simdgroup (8 scales + 8 mins), ROWS_PER_TG simdgroups.
constant uint Q4K_GUC_COEFFS_PER_SG = 16u;

kernel void q4k_ffn_gate_up_coop(
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
    uint tgs_per_mat = (N + Q4K_GUC_ROWS_PER_TG - 1u) / Q4K_GUC_ROWS_PER_TG;
    bool is_up  = (tg_id >= tgs_per_mat);
    uint mat_tg = is_up ? (tg_id - tgs_per_mat) : tg_id;

    uint row_idx = mat_tg * Q4K_GUC_ROWS_PER_TG + sg_id;
    if (row_idx >= N) return;

    device const uchar* W       = is_up ? Wu : Wg;
    device float*       out_buf = is_up ? U_out : G_out;

    const uint superblocks   = K / 256u;
    const uint bytes_per_row = superblocks * Q4K_GUC_BLOCK_SIZE;
    device const uchar* row_w = W + row_idx * bytes_per_row;

    // Lane partition (matches production):
    //   ix  = lane & 1   → super-block parity
    //   tid = lane >> 1  → 0..15: which (sub, half) cell
    //   j   = tid >> 1   → 0..7: which sub-block (4 lanes share j)
    //   sh  = tid & 1    → 0/1: first or last 16 of the 32-elem sub-block
    const uint ix  = lane & 1u;
    const uint tid = lane >> 1u;
    const uint j   = tid >> 1u;
    const uint sh  = tid & 1u;
    const bool hi    = (j & 1u) != 0u;
    const uint group = j >> 1u;

    // Per-simdgroup scratch: 8 scales + 8 mins per simdgroup × 4
    // simdgroups = 64 floats = 256 B per TG, well under hardware
    // threadgroup-memory limits.
    threadgroup float coeffs[Q4K_GUC_ROWS_PER_TG * Q4K_GUC_COEFFS_PER_SG];

    float acc = 0.0f;

    for (uint sb = ix; sb < superblocks; sb += 2u) {
        device const uchar* block = row_w + sb * Q4K_GUC_BLOCK_SIZE;

        // ── Cooperative scale/min decode on lanes 0..7 ──
        // Each of those lanes also decodes d/dmin themselves (8×
        // redundant vs production's 32×; negligible cost). Avoids a
        // `simd_broadcast` round-trip that earlier prototypes found
        // re-orders the inner FMA chain enough to flip rank-1 on
        // close-call tokens at the LM head.
        if (lane < 8u) {
            uint k = lane;

            ushort d_bits    = ushort(block[0]) | (ushort(block[1]) << 8u);
            ushort dmin_bits = ushort(block[2]) | (ushort(block[3]) << 8u);
            float d    = decode_f16_metal(d_bits);
            float dmin = decode_f16_metal(dmin_bits);

            device const uchar* sb_bytes = block + 4u;
            uint sc, mn;
            if (k < 4u) {
                sc = uint(sb_bytes[k])      & 0x3Fu;
                mn = uint(sb_bytes[k + 4u]) & 0x3Fu;
            } else {
                sc = (uint(sb_bytes[k + 4u]) & 0x0Fu) | ((uint(sb_bytes[k - 4u]) >> 6u) << 4u);
                mn = (uint(sb_bytes[k + 4u]) >> 4u)    | ((uint(sb_bytes[k])      >> 6u) << 4u);
            }
            uint base = sg_id * Q4K_GUC_COEFFS_PER_SG;
            coeffs[base + k]      = d    * float(sc);
            coeffs[base + 8u + k] = dmin * float(mn);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // All lanes read their owned sub-block's scale/mmin.
        uint base = sg_id * Q4K_GUC_COEFFS_PER_SG;
        float scale = coeffs[base + j];
        float mmin  = coeffs[base + 8u + j];

        // ── Inner work: identical to production `q4k_ffn_gate_up` ──
        // Preload 16 X values into registers BEFORE loading weight bytes.
        const uint x_base = sb * 256u + j * 32u + sh * 16u;
        float xl[16];
        _Pragma("clang loop unroll(full)")
        for (uint l = 0u; l < 16u; l++) { xl[l] = X[x_base + l]; }

        // Weight nibble bytes for this lane's 16-element slice.
        device const uchar* qs = block + 16u + group * 32u + sh * 16u;

        // Precompute Σ X over the 16-element slice for the min-correction.
        float sumy = 0.0f;
        _Pragma("clang loop unroll(full)")
        for (uint l = 0u; l < 16u; l++) { sumy += xl[l]; }

        // Pure FMA chain — uninterrupted by dequant work.
        float dot_acc = 0.0f;
        _Pragma("clang loop unroll(full)")
        for (uint l = 0u; l < 16u; l++) {
            uchar byte = qs[l];
            float nib = hi ? float((byte >> 4u) & 0x0Fu) : float(byte & 0x0Fu);
            dot_acc = fma(nib, xl[l], dot_acc);
        }
        // Q4_K deferred form: scale * Σ(nib*x) - dmin_min * Σ(x).
        acc += scale * dot_acc - mmin * sumy;
    }

    acc = simd_sum(acc);
    if (lane == 0u) out_buf[row_idx] = acc;
}
"#;

pub const ROWS_PER_TG: u64 = 4;
pub const THREADS_PER_TG: u64 = 128;

/// Marker for the kernel-handle binding. See `metal::kernel::TiledKernel`.
pub struct Kernel;
impl crate::kernels::TiledKernel for Kernel {
    const KERNEL_NAME: &'static str = "q4k_ffn_gate_up_coop";
    const ROWS_PER_TG: u64 = ROWS_PER_TG;
    const THREADS_PER_TG: u64 = THREADS_PER_TG;
}
