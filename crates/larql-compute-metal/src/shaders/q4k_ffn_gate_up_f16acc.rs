//! Q4_K fused gate+up with **f16 inner accumulators** — experimental variant.
//!
//! ## Retention rationale (post 2026-05-09 model-agnosticity audit)
//!
//! - **Tried**: 2026-04-28 on Gemma 3 4B (K=2560, M3 Max). Result:
//!   kernel-isolated **1.79× speedup** (0.607 → 0.340 ms iso); end-to-end
//!   **at parity** on quiet GPU. Initial +23% measurement was thermal-
//!   throttle artifact. Pinned as ADR-015 instance #1.
//! - **Status**: Opt-in via `LARQL_F16_ACC=1`. Not deleted despite Gemma
//!   end-to-end null because:
//!   - **M5+/A19 hardware could win**: the production kernel is
//!     bandwidth-bound at ~74% LPDDR5X peak. If future Apple silicon
//!     widens the DRAM bus or shifts the ALU/bandwidth balance, the f16
//!     inner-FMA throughput advantage could materialise end-to-end.
//!   - **Sustained-load thermal scenarios**: the f16 path uses fewer ALU
//!     cycles per FMA, generating less thermal pressure. Long-running
//!     workloads where the GPU is at sustained TDP may see this kernel
//!     stay closer to peak frequency than the f32 variant.
//!   - **Larger-K models could win**: K=2560 (Gemma 3 4B) gives the
//!     bandwidth path the smallest footprint for f16-acc to help; larger
//!     K (Llama 3 70B at K=8192) shifts the dequant amortisation.
//! - **Re-validation gate**: A/B on quiet hardware with `--warmup 8 -n
//!   30 --profile`; promote if batched-diag improves AND end-to-end shows
//!   ≥ 5% tok/s gain with correct output (Paris ✓ on smoke test).
//! - **Deletion criterion**: if a multi-hardware multi-K bench (M3 +
//!   M5/A19 + Llama 70B + Gemma 4 31B) all show null, can be deleted.
//!   No such bench has been run yet.
//!
//! Numerical parity validated against f32 reference: 10-prompt corpus
//! greedy-decode bit-identical on Gemma 3 4B. The f16-acc variant is
//! correct, just not faster on this specific Gemma + hardware combo.
//!
//! See `docs/shader-inventory.md` for the retention framework and
//! `docs/adr/015-isolated-vs-batched-kernel-perf.md` for the iso-vs-batched
//! lesson this kernel pinned.
//!
//! ## Original kernel description
//!
//! Hypothesis: Apple Silicon GPUs run f16 FMA at 2× f32 throughput. The
//! inner per-superblock dot loop (16 FMAs across `nib × xl`) is a clean
//! candidate to drop into half precision provided the partial sum stays
//! in f16 range:
//!   - `nib` is integer 0..15 → exact in f16
//!   - `xl` is RMS-normed residual, typically `|x| < 10`
//!   - max partial: 16 × 15 × 10 = 2400 << f16 max (65504)
//!   - per-element rounding: f16 has 11-bit mantissa = ~3 decimal digits;
//!     accumulation across 16 elements degrades by ~log2(16)/2 = 2 bits.
//!
//! Outer accumulator stays f32 — the per-superblock contributions
//! (`scale × dot - mmin × sumy`) span 10 superblocks at K=2560, and
//! `acc` magnitude can drift in f16 over that range. f32 outer keeps
//! the cross-superblock add error-free.
//!
//! `sumy` (the min-correction sum-of-X term) also stays f32 because
//! `dmin × sumy` is sensitive to X magnitude and small drift in `sumy`
//! gets amplified by `dmin`.
//!
//! Relative to [`q4k_ffn_gate_up`]:
//!   - Inner FMA chain: f16 (was f32)
//!   - X preload: still f32 in memory; cast to half just for FMA
//!   - Final per-superblock contribute: convert dot to f32, then scale
//!
//! Parity contract: numerical drift vs f32 accumulator should be
//! < 1e-3 absolute on `xl` magnitudes < 10. Tested by
//! `q4k_ffn_gate_up_f16acc_matches_f32_within_tolerance` in
//! `tests/test_kernel_q4k_ffn_gate_up_f16acc.rs`. If a future caller's
//! workload pushes |x| above ~50 the f16 path can saturate; gate this
//! at runtime via a `LARQL_F16_ACC=1` opt-in until precision is
//! validated end-to-end on a real prompt.

pub const SHADER: &str = r#"
constant uint Q4K_GU_F16_ROWS_PER_TG = 4;
constant uint Q4K_GU_F16_BLOCK_SIZE  = 144;

kernel void q4k_ffn_gate_up_f16acc(
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
    uint tgs_per_mat = (N + Q4K_GU_F16_ROWS_PER_TG - 1u) / Q4K_GU_F16_ROWS_PER_TG;
    bool is_up  = (tg_id >= tgs_per_mat);
    uint mat_tg = is_up ? (tg_id - tgs_per_mat) : tg_id;

    uint row_idx = mat_tg * Q4K_GU_F16_ROWS_PER_TG + sg_id;
    if (row_idx >= N) return;

    device const uchar* W      = is_up ? Wu : Wg;
    device float*       out_buf = is_up ? U_out : G_out;

    const uint superblocks   = K / 256u;
    const uint bytes_per_row = superblocks * Q4K_GU_F16_BLOCK_SIZE;
    device const uchar* row_w = W + row_idx * bytes_per_row;

    const uint ix  = lane & 1u;
    const uint tid = lane >> 1u;
    const uint j   = tid >> 1u;
    const uint sh  = tid & 1u;
    const bool hi    = (j & 1u) != 0u;
    const uint group = j >> 1u;

    float acc = 0.0f;  // outer accumulator stays f32

    for (uint sb = ix; sb < superblocks; sb += 2u) {
        device const uchar* block = row_w + sb * Q4K_GU_F16_BLOCK_SIZE;
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
        // Load X as f32, immediately cast to half for the FMA chain.
        // Keeping the f32 fetch lets the compiler share the X load with
        // any future f32 paths in the same shader and avoids reading
        // through unaligned half pointers.
        half xl_h[16];
        float sumy = 0.0f;  // sumy stays f32 — dmin × sumy is precision-sensitive
        _Pragma("clang loop unroll(full)")
        for (uint l = 0u; l < 16u; l++) {
            float xv = X[x_base + l];
            xl_h[l] = half(xv);
            sumy += xv;
        }

        device const uchar* qs = block + 16u + group * 32u + sh * 16u;

        // Inner dot in half precision. 16 FMAs of (int 0..15) × (|x| < ~10)
        // stay well under f16 max (65504). 2× FMA throughput vs f32 on M3.
        half dot_acc_h = half(0.0);
        _Pragma("clang loop unroll(full)")
        for (uint l = 0u; l < 16u; l++) {
            uchar byte = qs[l];
            half nib_h = hi ? half((byte >> 4u) & 0x0Fu) : half(byte & 0x0Fu);
            dot_acc_h = fma(nib_h, xl_h[l], dot_acc_h);
        }
        float dot_acc = float(dot_acc_h);

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
    const KERNEL_NAME: &'static str = "q4k_ffn_gate_up_f16acc";
    const ROWS_PER_TG: u64 = ROWS_PER_TG;
    const THREADS_PER_TG: u64 = THREADS_PER_TG;
}
