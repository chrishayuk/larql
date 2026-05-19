//! Q4_K matrix-vector multiply — **stride-32 lane access** variant.
//!
//! Same Q4_K (144-byte super-block) input format as [`q4k_matvec`], but
//! the per-row work is split across 32 simdgroup lanes the way
//! [`f16_gemv`](super::f16_gemv) does: lane `k` accumulates the dot-product
//! contribution of every element `i` where `i % 32 == k`. Final reduction
//! is `simd_sum` across 32 lanes — bit-identical reduction tree to the
//! f16 LM-head path.
//!
//! **Why this kernel exists**: the production [`q4k_matvec`] partitions
//! work *within* the Q4_K block layout (`ix = lane & 1u` splits lanes
//! into odd/even-superblock pairs; `tid = lane >> 1u` tiles 16-element
//! sub-block halves). That layout is cache-friendly for Q4_K but produces
//! a 32-lane parallel reduction whose tree differs from CPU's sequential
//! sum *enough* to flip top-1 on close-call tokens at the LM head — a
//! wrong-answer regression on Gemma 3 4B (`arch_golden_gemma3_4b_gpu`
//! emitting "The Capital of France is" instead of "**Paris**"; see
//! `larql-inference/ROADMAP.md` "Metal lm_head" entry).
//!
//! The f16-on-`embeddings.bin` workaround that ships in v4 fixes the
//! correctness bug at the cost of reading 1.3 GB f16/tok instead of
//! 330 MB Q4_K/tok. This kernel was originally framed as "the path to
//! recovering that loss" by giving Q4_K a stable reduction tree.
//!
//! ## Retention rationale (ADR-017)
//!
//! **Status**: opt-in, not wired into production decode.
//!
//! **Empirical result (2026-05-09, `project_f16_gemv_wiring_todo.md`)**:
//! the broader f16-on-Q4_K wiring story is dead — direct f16 LM head is
//! 1.85 ms vs current Q4_K production at 3.88 ms, but only because the
//! production already ships a fix. Stride-32 has not been A/B-validated
//! end-to-end against the current Q4_K LM-head path; no decode-level
//! tok/s improvement has been measured. The original "+10 tok/s recovery"
//! framing is **stale** — it refers to a regression that was healed by a
//! different path.
//!
//! **Why kept on disk**: the stride-32 reduction tree is bit-identical
//! to f16_gemv's, so this remains the cheapest insurance kernel if a
//! future Q4_K LM-head A/B reopens the close-call top-1 issue on a new
//! arch family. Maintenance cost is one shader file plus a parity test;
//! revival cost is one dispatch site.
//!
//! **Removal trigger**: if a year passes with no LM-head close-call
//! regression on any production model AND no test references this
//! shader outside its own kernel-parity test, demote and delete.
//!
//! **Reduction tree** (key bit):
//!
//! ```text
//! lane k accumulates: Σ over i ∈ {k, k+32, k+64, ...} of dequant(W,i) * X[i]
//!                     (one element per stride-32 modular class)
//! simd_sum(acc) reduces 32 partial sums via the SIMD tree
//! ```
//!
//! Identical to f16_gemv's per-lane work and final reduction.
//!
//! **Memory access**: lane `k`'s elements sit at offsets `k, k+32, ...`
//! within each 256-element super-block. For a fixed sub-block `sub` (0..7)
//! of 32 elements at offsets `sub*32..sub*32+32`, lane `k` reads exactly
//! one element at offset `k`. The 32 lanes therefore read 32 distinct
//! elements per sub-block, covering all 32. Each pair of lanes (`k`, `k+16`)
//! shares one nibble byte (one packs into the lo nibble, the other the hi);
//! each lane reads `bytes_per_row / 32` bytes total — exactly the same
//! aggregate Q4_K bandwidth as the production kernel.
//!
//! `d`, `dmin`, the 12-byte packed scales/mins, and the per-sub-block
//! `scale = d * sc` / `mmin = dmin * mn` are decoded once per super-block
//! per lane (loop-invariant relative to the inner sub-block walk; the
//! compiler should hoist them).
//!
//! **Numerical equivalence**: Per element, the dequantised weight is
//! `scale[sub] * nibble - mmin[sub]`. The lane-local accumulator runs
//! `acc += (scale * nib - mmin) * X[i]` — same per-element form as the
//! CPU reference (`cpu/ops/q4k_matvec.rs::dispatch`). The production
//! kernel uses the deferred form `acc += scale * Σ(nib*x) - mmin * Σ(x)`
//! which is mathematically equivalent but accumulates rounding errors
//! differently. The per-element form, combined with the stride-32
//! reduction tree, gives the closest numerical match to f16_gemv that
//! we can express on Q4_K bytes.
//!
//! **Geometry**: 8 simdgroups per TG, 8 rows per TG, 256 threads per TG.
//! Mirrors `f16_gemv` and `q4k_matvec_8sg` so threadgroup occupancy and
//! dispatch grid math are unchanged.

pub const SHADER: &str = r#"
constant uint Q4K_S32_ROWS_PER_TG = 8;
constant uint Q4K_S32_BLOCK_SIZE  = 144;

kernel void q4k_matvec_stride32(
    device const uchar*  W4K   [[buffer(0)]],
    device const float*  X     [[buffer(1)]],
    device float*        out   [[buffer(2)]],
    constant uint&       N     [[buffer(3)]],
    constant uint&       K     [[buffer(4)]],
    uint tg_id     [[threadgroup_position_in_grid]],
    uint lane      [[thread_index_in_simdgroup]],
    uint sg_id     [[simdgroup_index_in_threadgroup]])
{
    uint row_idx = tg_id * Q4K_S32_ROWS_PER_TG + sg_id;
    if (row_idx >= N) return;

    const uint superblocks   = K / 256u;
    const uint bytes_per_row = superblocks * Q4K_S32_BLOCK_SIZE;
    device const uchar* row_w = W4K + row_idx * bytes_per_row;

    float acc = 0.0f;

    // Lane-local byte addressing within each 32-byte nibble group:
    //   sh    = 0 for lanes 0..15, 1 for lanes 16..31
    //   inner = lane & 15
    // Pre-compute once outside the super-block loop.
    const uint sh    = lane >> 4u;
    const uint inner = lane & 15u;

    for (uint sb = 0u; sb < superblocks; sb++) {
        device const uchar* block = row_w + sb * Q4K_S32_BLOCK_SIZE;

        // Per-super-block scales — decoded once, used 8 times below.
        ushort d_bits    = ushort(block[0]) | (ushort(block[1]) << 8u);
        ushort dmin_bits = ushort(block[2]) | (ushort(block[3]) << 8u);
        float d    = decode_f16_metal(d_bits);
        float dmin = decode_f16_metal(dmin_bits);
        device const uchar* sb_bytes = block + 4u;

        // Walk the 8 sub-blocks. Each lane handles exactly one element
        // per sub-block: lane `k` ← element at offset `k` within the
        // sub-block (k ∈ 0..31). 8 elements per super-block per lane,
        // matching production kernel's 16-elt-per-half × 1-half-per-lane.
        //
        // Per-sub-block sc / mn unpack lives **inside** the loop —
        // hoisting it out and storing 8 scales + 8 mins per super-block
        // costs 32× the unpack work across the simdgroup vs unpacking
        // only the active sub-block's scale/min on the lane that needs
        // it. Compiler should still hoist the constant address math.
        _Pragma("clang loop unroll(full)")
        for (uint sub = 0u; sub < 8u; sub++) {
            uint sc, mn;
            if (sub < 4u) {
                sc = uint(sb_bytes[sub])      & 0x3Fu;
                mn = uint(sb_bytes[sub + 4u]) & 0x3Fu;
            } else {
                sc = (uint(sb_bytes[sub + 4u]) & 0x0Fu) | ((uint(sb_bytes[sub - 4u]) >> 6u) << 4u);
                mn = (uint(sb_bytes[sub + 4u]) >> 4u)    | ((uint(sb_bytes[sub])      >> 6u) << 4u);
            }
            float scale = d    * float(sc);
            float mmin  = dmin * float(mn);

            // Nibble byte location: 4 groups of 32 bytes (group = sub/2).
            // Within each 32-byte group, bytes [0..16] hold lane offsets
            // 0..15 (sh=0), bytes [16..32] hold 16..31 (sh=1). Even
            // sub-blocks (sub%2==0) use the lo nibble of each byte; odd
            // use the hi nibble. `group * 32 + sh * 16 + inner` is the
            // offset from the start of the nibble payload (block + 16).
            uint group = sub >> 1u;
            bool hi    = (sub & 1u) != 0u;
            uchar byte = block[16u + group * 32u + sh * 16u + inner];
            float nib  = hi ? float((byte >> 4u) & 0x0Fu) : float(byte & 0x0Fu);

            uint x_idx = sb * 256u + sub * 32u + lane;
            float w    = fma(scale, nib, -mmin);
            acc        = fma(w, X[x_idx], acc);
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
    const KERNEL_NAME: &'static str = "q4k_matvec_stride32";
    const ROWS_PER_TG: u64 = ROWS_PER_TG;
    const THREADS_PER_TG: u64 = THREADS_PER_TG;
}
