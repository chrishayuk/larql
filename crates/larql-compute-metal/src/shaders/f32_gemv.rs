//! f32 gemv — matrix-vector multiply for the LM head.
//!
//! Computes `out[N] = W[N, K] · x[K]` where `W` is row-major f32.
//!
//! One simdgroup per row. Each of the 32 lanes reads `K/32` strided
//! elements, accumulates a partial dot product, then `simd_sum` reduces
//! into a single output.
//!
//! Sized for the Gemma 3/4 tied LM head: N ~ 262 K, K = 2560–5120. The
//! simdgroup-per-row pattern gets ~4× over the 32×32 tiled sgemm at M=1
//! (which wastes 31/32 of its threads and leaves accumulation precision
//! different enough to shift argmax on noisy logits).

pub const SHADER: &str = r#"
constant uint F32GEMV_SG_PER_TG = 8;   // simdgroups per threadgroup
constant uint F32GEMV_ROWS_PER_TG = F32GEMV_SG_PER_TG; // one row per simdgroup

kernel void f32_gemv(
    device const float* W   [[buffer(0)]],   // [N, K] row-major
    device const float* X   [[buffer(1)]],   // [K]
    device float*       out [[buffer(2)]],   // [N]
    constant uint&      N   [[buffer(3)]],
    constant uint&      K   [[buffer(4)]],
    uint tg_id   [[threadgroup_position_in_grid]],
    uint lane    [[thread_index_in_simdgroup]],
    uint sg_id   [[simdgroup_index_in_threadgroup]])
{
    uint row = tg_id * F32GEMV_ROWS_PER_TG + sg_id;
    if (row >= N) return;

    device const float* w_row = W + row * K;

    float acc = 0.0f;
    // Stride-32 over K; four unrolled per-lane accumulators avoid
    // serialising on a single latency-bound chain.
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    uint k = lane;
    for (; k + 3 * 32 < K; k += 4 * 32) {
        a0 = fma(w_row[k         ], X[k         ], a0);
        a1 = fma(w_row[k + 32    ], X[k + 32    ], a1);
        a2 = fma(w_row[k + 64    ], X[k + 64    ], a2);
        a3 = fma(w_row[k + 96    ], X[k + 96    ], a3);
    }
    acc = (a0 + a1) + (a2 + a3);
    for (; k < K; k += 32) acc = fma(w_row[k], X[k], acc);

    acc = simd_sum(acc);
    if (lane == 0) out[row] = acc;
}
"#;

pub const ROWS_PER_TG: u64 = 8;
pub const THREADS_PER_TG: u64 = 256; // 8 simdgroups × 32 lanes

/// Marker for the kernel-handle binding. See `metal::kernel::TiledKernel`.
pub struct Kernel;
impl crate::kernels::TiledKernel for Kernel {
    const KERNEL_NAME: &'static str = "f32_gemv";
    const ROWS_PER_TG: u64 = ROWS_PER_TG;
    const THREADS_PER_TG: u64 = THREADS_PER_TG;
}

/// Threadgroup width shared by both `f32_argmax_partial` and
/// `f32_topk_partial`. Both shaders assume `tg_sz == PARTIAL_TG_SZ` and
/// size their threadgroup memory to it; the Rust dispatcher must pass the
/// same value. Treat it as a kernel parameter, not a tunable.
pub const PARTIAL_TG_SZ: u64 = 256;

/// Maximum simdgroups per TG, used to size the cross-simdgroup reduction
/// scratch (`tg_v[MAX_SIMDGROUPS_PER_TG]` in argmax,
/// `sg_v[MAX_SIMDGROUPS_PER_TG]` in topk). At `PARTIAL_TG_SZ = 256` and
/// Apple Silicon's 32-lane simdgroup, this is `8`.
pub const MAX_SIMDGROUPS_PER_TG: usize = PARTIAL_TG_SZ as usize / 32;

/// Top-K shader constant. `f32_topk_partial` writes `K_TOPK` (val, idx) pairs
/// per TG. CPU final reduction merges `num_tgs × K_TOPK` candidates into the
/// caller's requested top-k. K=8 covers all production lm_head callers
/// (greedy/sampler use top_k ≤ 5; constrained decode is a different path).
pub const K_TOPK: usize = 8;

/// Metal source for `f32_argmax_partial`. Phase 1 of the two-phase argmax:
/// each TG of `PARTIAL_TG_SZ` threads finds its local max → writes one
/// (val, idx) pair to the partial result arrays. CPU reduces (`num_tgs`
/// candidates). Phase 2 is CPU-side (`num_tgs × 8` bytes ≤ ~8 KB, ~1 µs).
///
/// `MAX_SIMDGROUPS_PER_TG` is templated in via [`argmax_shader_source`] so
/// the threadgroup-memory arrays cannot drift from the dispatcher.
const ARGMAX_SHADER_BODY: &str = r#"
// Phase 1: per-TG argmax. Grid: ceil(N/PARTIAL_TG_SZ) TGs × PARTIAL_TG_SZ threads.
// Writes one (float, uint) pair per TG to out_val / out_idx.
kernel void f32_argmax_partial(
    device const float* scores   [[buffer(0)]],
    device float*       out_val  [[buffer(1)]],
    device uint*        out_idx  [[buffer(2)]],
    constant uint&      N        [[buffer(3)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid   [[thread_position_in_threadgroup]],
    uint tg_sz [[threads_per_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]])
{
    uint i = tg_id * tg_sz + tid;
    float local_val = (i < N) ? scores[i] : -1e38f;
    uint  local_idx = (i < N) ? i : 0u;

    // Simd reduction: find max value in simdgroup, then find index.
    float sg_max = simd_max(local_val);
    // Among lanes holding the max, take the smallest index (stable argmax).
    uint sg_idx = (local_val >= sg_max) ? local_idx : ~0u;
    sg_idx = simd_min(sg_idx);

    // Threadgroup reduction across simdgroups.
    threadgroup float tg_v[MAX_SIMDGROUPS_PER_TG];
    threadgroup uint  tg_i[MAX_SIMDGROUPS_PER_TG];
    if (lane == 0u) { tg_v[sg_id] = sg_max; tg_i[sg_id] = sg_idx; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        uint n_sg = (tg_sz + 31u) / 32u;
        float best_val = tg_v[0]; uint best_idx = tg_i[0];
        for (uint s = 1u; s < n_sg; s++) {
            if (tg_v[s] > best_val || (tg_v[s] == best_val && tg_i[s] < best_idx)) {
                best_val = tg_v[s]; best_idx = tg_i[s];
            }
        }
        out_val[tg_id] = best_val;
        out_idx[tg_id] = best_idx;
    }
}
"#;

/// Build the MSL source for `f32_argmax_partial`, substituting the Rust
/// `MAX_SIMDGROUPS_PER_TG` placeholder so the threadgroup-memory arrays
/// can't drift from the dispatcher's `PARTIAL_TG_SZ`. Called once at
/// backend init via `all_shaders()`. Plain string substitution (rather
/// than MSL `constant uint` declarations) keeps each helper's output
/// self-contained — no order-of-concatenation hazards when several
/// templated shaders end up in the same bundle.
pub fn argmax_shader_source() -> String {
    ARGMAX_SHADER_BODY.replace("MAX_SIMDGROUPS_PER_TG", &MAX_SIMDGROUPS_PER_TG.to_string())
}

pub struct ArgmaxKernel;
impl crate::kernels::ShaderKernel for ArgmaxKernel {
    const KERNEL_NAME: &'static str = "f32_argmax_partial";
}

/// Per-threadgroup top-K kernel source.
///
/// Each TG of `PARTIAL_TG_SZ` threads scans its slice via `K_TOPK` rounds
/// of simd_max → mask the winner → repeat. Per round: 5 simd ops + a
/// barrier. At K=8 that's ~50 ops/TG plus the threadgroup memory
/// accounting, negligible vs the GEMV that produced the scores. Output
/// layout: `out_val[tg_id * K_TOPK + k]` / `out_idx[tg_id * K_TOPK + k]`,
/// sorted by score descending per TG. Stable argmax within ties via
/// lane-min on the original index (matches `f32_argmax_partial`).
///
/// The MSL `constant uint K_TOPK` and the threadgroup-memory array sizes
/// are templated from the Rust constants above via [`topk_shader_source`].
/// Don't paste this string into the all-shaders bundle directly.
const TOPK_SHADER_BODY: &str = r#"
kernel void f32_topk_partial(
    device const float* scores  [[buffer(0)]],
    device float*       out_val [[buffer(1)]],
    device uint*        out_idx [[buffer(2)]],
    constant uint&      N       [[buffer(3)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid   [[thread_position_in_threadgroup]],
    uint tg_sz [[threads_per_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]])
{
    // Each thread loads one element; out-of-range threads load -inf so they
    // never win the argmax. Original index is the per-row global score idx.
    uint i = tg_id * tg_sz + tid;
    threadgroup float tg_v[PARTIAL_TG_SZ];
    threadgroup uint  tg_i[PARTIAL_TG_SZ];
    tg_v[tid] = (i < N) ? scores[i] : -1e38f;
    tg_i[tid] = (i < N) ? i : ~0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float sg_v[MAX_SIMDGROUPS_PER_TG];
    threadgroup uint  sg_i[MAX_SIMDGROUPS_PER_TG];
    threadgroup float winner_v;
    threadgroup uint  winner_i;

    for (uint k = 0u; k < K_TOPK; k++) {
        float v = tg_v[tid];
        // Simd reduction inside the simdgroup of 32 lanes.
        float sg_max = simd_max(v);
        uint  cand   = (v >= sg_max) ? tg_i[tid] : ~0u;
        cand         = simd_min(cand);

        if (lane == 0u) { sg_v[sg_id] = sg_max; sg_i[sg_id] = cand; }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0u) {
            uint n_sg = (tg_sz + 31u) / 32u;
            float best_v = sg_v[0];
            uint  best_i = sg_i[0];
            for (uint s = 1u; s < n_sg; s++) {
                if (sg_v[s] > best_v || (sg_v[s] == best_v && sg_i[s] < best_i)) {
                    best_v = sg_v[s];
                    best_i = sg_i[s];
                }
            }
            out_val[tg_id * K_TOPK + k] = best_v;
            out_idx[tg_id * K_TOPK + k] = best_i;
            winner_v = best_v;
            winner_i = best_i;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Mask the winning thread's value to -inf so it can't win again.
        // Indices are globally unique so exactly one thread matches.
        if (tg_i[tid] == winner_i) {
            tg_v[tid] = -1e38f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
"#;

/// Build the MSL source for `f32_topk_partial`, substituting the Rust
/// `K_TOPK` / `PARTIAL_TG_SZ` / `MAX_SIMDGROUPS_PER_TG` placeholders.
/// Same plain-string approach as `argmax_shader_source` — no MSL
/// `constant` declarations to clash when both shaders share a bundle.
pub fn topk_shader_source() -> String {
    TOPK_SHADER_BODY
        .replace("K_TOPK", &K_TOPK.to_string())
        .replace("PARTIAL_TG_SZ", &PARTIAL_TG_SZ.to_string())
        .replace("MAX_SIMDGROUPS_PER_TG", &MAX_SIMDGROUPS_PER_TG.to_string())
}

pub struct TopKKernel;
impl crate::kernels::ShaderKernel for TopKKernel {
    const KERNEL_NAME: &'static str = "f32_topk_partial";
}
