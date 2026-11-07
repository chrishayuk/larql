//! MoE router projection — `logits[E] = W[E, H] · x[H] + bias[E]`.
//!
//! Rung A of the GPU-dataflow routing ladder: the projection that decides
//! which experts run, computed on the GPU so the route can eventually stop
//! being a host decision. Parity-gated against the CPU oracle
//! `larql_compute::cpu::ops::moe::moe_router_logits` — route-selection
//! semantics live there; this kernel must only reproduce them.
//!
//! Bias joins the logits here, BEFORE any softmax/selection downstream,
//! because it changes which experts win — same contract as the oracle.
//!
//! Geometry: one THREADGROUP per output row, 8 simdgroups cooperating
//! across `H`, cross-simdgroup reduction through threadgroup memory.
//! This deliberately inverts `f32_gemv`'s row-per-simdgroup mapping:
//! that shape is built for vocab-scale N where rows supply the
//! parallelism. A router is the opposite corner — few rows (gpt-oss:
//! 32 experts), long K — and row-per-simdgroup leaves it at 4
//! threadgroups, too few to hide memory latency (measured 16 GB/s /
//! ~22 µs per dispatch; `test_kernel_moe_router_perf`'s occupancy
//! probe). Row-per-threadgroup runs E threadgroups (32 at production
//! shape), which reaches the same ~8 µs small-dispatch floor as a
//! saturated grid.

pub const SHADER: &str = r#"
constant uint MOE_ROUTER_SG_PER_TG = 8;      // simdgroups per threadgroup
constant uint MOE_ROUTER_TG_THREADS = 256;   // 8 simdgroups x 32 lanes

// One logits row. Shared by the single-position and VERIFY-N rows kernels
// so the two cannot drift: a block row is this body at the row's offsets.
static inline void moe_router_logits_row(
    device const float* W, device const float* X, device const float* bias,
    device float* out, uint E, uint H, uint has_bias, uint row,
    uint tid, uint lane, uint sg_id, threadgroup float* sg_partial)
{
    if (row >= E) return;

    device const float* w_row = W + row * H;

    // All 256 threads stride the row; four unrolled accumulators keep
    // the per-thread chain latency-tolerant, as in the gemv family.
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    uint h = tid;
    for (; h + 3 * MOE_ROUTER_TG_THREADS < H; h += 4 * MOE_ROUTER_TG_THREADS) {
        a0 = fma(w_row[h                            ], X[h                            ], a0);
        a1 = fma(w_row[h +     MOE_ROUTER_TG_THREADS], X[h +     MOE_ROUTER_TG_THREADS], a1);
        a2 = fma(w_row[h + 2 * MOE_ROUTER_TG_THREADS], X[h + 2 * MOE_ROUTER_TG_THREADS], a2);
        a3 = fma(w_row[h + 3 * MOE_ROUTER_TG_THREADS], X[h + 3 * MOE_ROUTER_TG_THREADS], a3);
    }
    float acc = (a0 + a1) + (a2 + a3);
    for (; h < H; h += MOE_ROUTER_TG_THREADS) acc = fma(w_row[h], X[h], acc);

    acc = simd_sum(acc);
    if (lane == 0) sg_partial[sg_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        // Ascending-sg_id order: the cross-simdgroup sum is deterministic
        // for a given dispatch geometry.
        float total = 0.0f;
        for (uint s = 0; s < MOE_ROUTER_SG_PER_TG; s++) total += sg_partial[s];
        out[row] = has_bias != 0u ? total + bias[row] : total;
    }
}

kernel void moe_router_logits(
    device const float* W        [[buffer(0)]],   // [E, H] row-major
    device const float* X        [[buffer(1)]],   // [H]
    device const float* bias     [[buffer(2)]],   // [E] (read iff has_bias)
    device float*       out      [[buffer(3)]],   // [E]
    constant uint&      E        [[buffer(4)]],
    constant uint&      H        [[buffer(5)]],
    constant uint&      has_bias [[buffer(6)]],   // 0 = architecture has none
    uint tg_id   [[threadgroup_position_in_grid]],
    uint tid     [[thread_position_in_threadgroup]],
    uint lane    [[thread_index_in_simdgroup]],
    uint sg_id   [[simdgroup_index_in_threadgroup]])
{
    threadgroup float sg_partial[MOE_ROUTER_SG_PER_TG];
    moe_router_logits_row(W, X, bias, out, E, H, has_bias, tg_id, tid, lane, sg_id, sg_partial);
}

// VERIFY-N: logits for `[R, H]` router inputs → `[R, E]`. Grid (E, R):
// threadgroup (row, r) is the single kernel's threadgroup `row` over
// position r's input — bit-identical per position.
kernel void moe_router_logits_rows(
    device const float* W        [[buffer(0)]],   // [E, H] row-major
    device const float* X        [[buffer(1)]],   // [R, H]
    device const float* bias     [[buffer(2)]],   // [E] (read iff has_bias)
    device float*       out      [[buffer(3)]],   // [R, E]
    constant uint&      E        [[buffer(4)]],
    constant uint&      H        [[buffer(5)]],
    constant uint&      has_bias [[buffer(6)]],
    uint2 tg_id  [[threadgroup_position_in_grid]],
    // Index, not position: Metal wants all position attributes the same
    // width, and a 1-D threadgroup's index equals its x position.
    uint tid     [[thread_index_in_threadgroup]],
    uint lane    [[thread_index_in_simdgroup]],
    uint sg_id   [[simdgroup_index_in_threadgroup]])
{
    threadgroup float sg_partial[MOE_ROUTER_SG_PER_TG];
    ulong r = tg_id.y;
    moe_router_logits_row(W, X + r * H, bias, out + r * E, E, H, has_bias, tg_id.x,
                          tid, lane, sg_id, sg_partial);
}
"#;

pub const ROWS_PER_TG: u64 = 1;
pub const THREADS_PER_TG: u64 = 256; // 8 simdgroups × 32 lanes

/// Marker for the kernel-handle binding. See `metal::kernel::TiledKernel`.
pub struct Kernel;

/// VERIFY-N rows twin: grid `(E, R)`, [`THREADS_PER_TG`] threads.
pub struct RowsKernel;
impl crate::kernels::ShaderKernel for RowsKernel {
    const KERNEL_NAME: &'static str = "moe_router_logits_rows";
}
impl crate::kernels::TiledKernel for Kernel {
    const KERNEL_NAME: &'static str = "moe_router_logits";
    const ROWS_PER_TG: u64 = ROWS_PER_TG;
    const THREADS_PER_TG: u64 = THREADS_PER_TG;
}

/// VERIFY-N (rung 5a): lay a block's router/expert input `[R, H]` out one
/// row per expert slot, `[R*K, C]` with slot `s = r*K + j` holding row `r`,
/// so the grouped expert kernels read each slot's input at `s * C` — the
/// same per-slot walk as the single position, with `C` the stored (padded)
/// row width. Columns `H..C` are never written: the caller's buffer keeps
/// them zero, as `MoeScratch::x_buf`'s tail.
pub const ROWS_TO_SLOTS_SHADER: &str = r#"
kernel void moe_rows_to_slots(
    device const float* x   [[buffer(0)]],   // [R, H]
    device float*       xs  [[buffer(1)]],   // [R*K, C]
    constant uint&      H   [[buffer(2)]],
    constant uint&      K   [[buffer(3)]],
    constant uint&      C   [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint h = gid.x;
    uint s = gid.y;
    if (h >= H) return;
    xs[(ulong)s * C + h] = x[(ulong)(s / K) * H + h];
}

// VERIFY-N rung 5b: a block's `S` routed slots grouped by expert, for the
// expert arms that share a distinct expert's reads. `order` is every slot,
// sorted by (expert id, slot). Distinct experts are ranked by expert id; distinct
// expert `u` gets `counts[u]` slots listed at `lists[u * MAXM ..]` in slot
// order; `counts[u] = 0` for `u` past the last distinct expert. ONE
// threadgroup of 256 threads (`E <= 256`, `S <= 256`); deterministic.
kernel void moe_slots_by_expert(
    device const uint* ids    [[buffer(0)]],   // [S]
    device uint*       counts [[buffer(1)]],   // [S]
    device uint*       lists  [[buffer(2)]],   // [S, MAXM]
    constant uint&     S      [[buffer(3)]],
    constant uint&     E      [[buffer(4)]],
    constant uint&     MAXM   [[buffer(5)]],
    device uint*       order  [[buffer(6)]],   // [S]: slots, expert-sorted
    uint tid   [[thread_index_in_threadgroup]],
    uint tg_sz [[threads_per_threadgroup]])
{
    threadgroup uint present[256];
    threadgroup uint rank[256];
    threadgroup uint first[256];
    for (uint t = tid; t < S; t += tg_sz) counts[t] = 0u;
    for (uint e = tid; e < E; e += tg_sz) {
        uint c = 0u;
        for (uint s = 0u; s < S; ++s) c += (ids[s] == e) ? 1u : 0u;
        present[e] = c;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    if (tid == 0u) {
        uint r = 0u;
        uint o = 0u;
        for (uint e = 0u; e < E; ++e) {
            rank[e] = r;
            first[e] = o;
            r += (present[e] > 0u) ? 1u : 0u;
            o += present[e];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    for (uint e = tid; e < E; e += tg_sz) {
        const uint c = present[e];
        if (c == 0u) continue;
        const uint u = rank[e];
        uint m = 0u;
        for (uint s = 0u; s < S; ++s) {
            if (ids[s] == e) {
                if (m < MAXM) lists[u * MAXM + m] = s;
                order[first[e] + m] = s;
                m += 1u;
            }
        }
        counts[u] = min(c, MAXM);
    }
}
"#;

/// Threadgroup width `moe_slots_by_expert` is written for; it also bounds
/// the experts and slots it groups.
pub const SLOTS_BY_EXPERT_THREADS: u64 = 256;

/// Marker for `moe_slots_by_expert` (in [`ROWS_TO_SLOTS_SHADER`]).
pub struct SlotsByExpertKernel;
impl crate::kernels::ShaderKernel for SlotsByExpertKernel {
    const KERNEL_NAME: &'static str = "moe_slots_by_expert";
}

/// Marker for [`ROWS_TO_SLOTS_SHADER`]'s pipeline.
pub struct RowsToSlotsKernel;
impl crate::kernels::ShaderKernel for RowsToSlotsKernel {
    const KERNEL_NAME: &'static str = "moe_rows_to_slots";
}
