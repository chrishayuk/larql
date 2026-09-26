//! Attention execution-geometry planner.
//!
//! Given an attention op's *semantic* geometry — `head_dim`, query heads,
//! KV heads, effective span — and the operator's `LARQL_KV_SEQPAR`
//! request, choose how the Metal backend executes it: the serial phase-3
//! kernel, or the KV-B1 sequence-parallel kernel with `slices` sequence
//! partitions per head (a `slices x head_dim` threadgroup).
//!
//! The policy is **measured per geometry, never named per model.** The
//! first sequence-parallel default shipped as `SEQPAR_DEFAULT_ON_HEAD_DIMS
//! = &[64]` — a span → threadgroup-width table tuned on gpt-oss-20b (64
//! query heads, 8 KV heads, head_dim 64; `docs/kv-attention-scaling.md`,
//! PR #264). Porting the same kernel into the VINDEX3 lowering and running
//! it on Glimmer (32 query heads, 2 KV heads, head_dim 128) showed that
//! table does not transfer: at head_dim 128 the same widths mean half the
//! slices, and with half the query heads there is half the head-level
//! parallelism before any sequence partitioning begins. So the unit of
//! evidence here is a `(head_dim, num_q_heads, num_kv_heads)` row with its
//! own span tiers, and a geometry with no row runs **serial** under an
//! unset request — an unmeasured policy is not a default.
//!
//! `q_heads` is deliberately part of the key even where a derived
//! quantity (GQA ratio, head parallelism) might later prove sufficient:
//! the surface should show a variable is redundant before it is dropped.
//!
//! What this module does not know: threadgroup limits beyond the shared
//! `SEQPAR_MAX_THREADS` bound, and anything about the device beyond what
//! the rows were measured on (M3 Max). Both belong here when a second
//! device family is measured.

use super::kv_seqpar::{slices_for, SeqparRequest, SEQPAR_MAX_THREADS};

/// The semantic geometry of one attention op at one step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttentionGeometryQuery {
    pub head_dim: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    /// Effective span: `min(kv_len, window)` — what the kernel will walk.
    pub span: u32,
}

/// The chosen execution geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionGeometry {
    /// `kv_attention` / `kv_attention_long`: one threadgroup per query
    /// head, phase 3 walked serially by `head_dim` threads.
    Serial,
    /// KV-B1 `kv_attention_seqpar[_long]`: one threadgroup per query head,
    /// `slices x head_dim` threads, phase 3 split across `slices` sequence
    /// partitions and reduced in fixed order.
    SeqPar { slices: usize },
}

impl AttentionGeometry {
    /// Slice count, or 0 for serial — the shape the dispatch helpers take.
    pub fn slices(self) -> usize {
        match self {
            AttentionGeometry::Serial => 0,
            AttentionGeometry::SeqPar { slices } => slices,
        }
    }
}

/// One measured geometry row: span tiers as `(span_floor, slices)`,
/// ascending by floor; the tier whose floor is the largest not exceeding
/// the span applies. `slices == 0` means serial for that tier.
struct MeasuredGeometry {
    head_dim: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    tiers: &'static [(u32, usize)],
}

/// The measured rows. Every entry carries the measurement that licenses
/// it; adding a row means adding a bracketed ladder or surface, not a
/// hunch.
const MEASURED: &[MeasuredGeometry] = &[
    // gpt-oss-20b — the KV-B1 span policy (`kv_seqpar::auto_threads`
    // expressed in slices at head_dim 64: 512/768/1024 threads → 8/12/16
    // slices below span 512, below 1024, and from 1024). Licensed by the
    // bracketed A/B/C ladder: +11.5% / +24.5% / +52.4% throughput at
    // ~36 / ~574 / ~2024 tokens of context (PR #264). Pinned equal to
    // `slices_for(Unset, 64, span)` by `gpt_oss_row_is_the_kv_b1_policy`.
    MeasuredGeometry {
        head_dim: 64,
        num_q_heads: 64,
        num_kv_heads: 8,
        tiers: &[(0, 8), (512, 12), (1024, 16)],
    },
    // Muse-Glimmer-30B through the VINDEX3 lowering (32 query heads, 2 KV
    // heads, head_dim 128; `scripts/glimmer-seqpar-surface.sh`, rested,
    // 2026-08-16, M3 Max, 94 W). ms/token, serial brackets around the
    // slice arms; every arm produced identical token ids and the route
    // witness confirmed the kernel:
    //
    //   ctx    serial   2     4     8    serial   bracket
    //   512      73    70    69    68      77      5.5%   direction only → unlicensed, serial
    //   1024     88    83    83    79      89      1.1%   near-valid: 8 slices +12%
    //   2048    116    96    92    85     113      2.6%   lower bounds +18 / +23 / +33%
    //   4000    144     -   136   134     145      0.7%   VALID: 4 → +6.2%, 8 → +7.8%
    //
    // 8 slices is 8 x 128 = 1024 threads — KV-B1's intra-threadgroup
    // ceiling, reached at this head_dim from ~1K. Past ~2K the whole
    // family's gain collapses (+33% → +8%) although the 39 sliding
    // layers stay at their 2048 window: the serial phase-3 walk is no
    // longer what dominates. Hypothesis for the next rung, to be measured
    // not assumed: 16 query-head threadgroups per KV head each re-read
    // that head's whole K/V, ~2 GB/token at 4K, which no intra-TG slicing
    // touches — a GQA-group kernel (one threadgroup per KV head serving
    // its 16 query heads) is the candidate.
    MeasuredGeometry {
        head_dim: 128,
        num_q_heads: 32,
        num_kv_heads: 2,
        tiers: &[(0, 0), (1024, 8)],
    },
    // Gemma 3 4B through the VINDEX3 lowering (8 query heads, 4 KV heads,
    // head_dim 256; NVFP4 body + head, M3 Max on AC, 2026-09-25). Steady
    // GPU ms/token over 100 decoded tokens after the prompt, serial brackets
    // around the slice arms; every arm produced identical token ids and
    // the route witness confirmed the kernel:
    //
    //   prompt   serial   2      4      serial   bracket
    //   25       10.19   9.66   9.38   10.18    0.1%   VALID: 4 → +8.5%
    //   357      13.89  11.38  10.07   13.94    0.4%   VALID: 4 → +37.9%
    //   891      20.50  14.96  14.80   20.48    0.1%   VALID: 4 → +38.4%
    //
    // 4 slices is 4 x 256 = 1024 threads, KV-B1's ceiling at this
    // head_dim, and wins at every measured depth. Eight query heads is
    // eight threadgroups per layer, so the serial phase-3 walk is the
    // whole cost even at short context — unlike Glimmer, where 32 heads
    // already filled the device below ~1K.
    MeasuredGeometry {
        head_dim: 256,
        num_q_heads: 8,
        num_kv_heads: 4,
        tiers: &[(0, 4)],
    },
];

/// SPLITK-1: the span split across `chunks` threadgroups per head, each
/// `slices x head_dim` threads, then merged (`ops::kv_splitk`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SplitKGeometry {
    pub chunks: usize,
    pub slices: usize,
}

/// A measured split-K row: tiers `(span_floor, chunks, slices)`, ascending
/// by floor, the largest floor not exceeding the span applying; `chunks
/// == 0` means no split-K for that tier (the seqpar row decides).
struct MeasuredSplitK {
    head_dim: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    tiers: &'static [(u32, usize, usize)],
}

const SPLITK_MEASURED: &[MeasuredSplitK] = &[
    // Gemma 3 4B (`examples/bench_attention_splitk.rs`, M3 Max on battery,
    // 2026-09-26; 34 cold caches per command buffer, seqpar(4) baseline
    // bracketed — every row below within 1% drift). Speedup over the
    // production seqpar(4) kernel, both passes included:
    //
    //   span   s2c4   s2c8   s2c16  s1c32   best
    //   128    1.15   1.20   1.12   1.03    s2c8
    //   256    1.27   1.39   1.33   1.24    s2c8
    //   512    1.48   1.78   1.81   1.89    s1c32 ≈ s2c16
    //   1024   1.79   2.40   2.67   2.80    s1c32 ≈ s2c16
    //   2048   1.96   2.83   3.40   3.40    tie
    //   4096   2.04   3.11   3.98   3.86    s2c16
    //
    // The optimum is interior (c64 loses at every span), so the sweep is
    // not bound-limited. Below 128 the 32-span row drifted 8-20% in both
    // runs: direction only → unlicensed, seqpar keeps it. Cause, from
    // GQA-RE-READ (`bench_attention_gqa`): 8 threadgroups on 40 cores is
    // per-threadgroup latency bound, not bytes.
    MeasuredSplitK {
        head_dim: 256,
        num_q_heads: 8,
        num_kv_heads: 4,
        tiers: &[(0, 0, 0), (128, 8, 2), (512, 16, 2)],
    },
];

/// Choose split-K for `q`, or `None` to leave it to
/// [`choose_attention_geometry`].
///
/// Only an UNSET request consults the split-K rows: any explicit
/// `LARQL_KV_SEQPAR` (off, auto, a slice count) is a request for the
/// intra-threadgroup kernels and keeps them — which is also the A/B arm.
/// The chunk count is raised to the kernel's floor
/// (`kv_splitk::min_chunks`) and never exceeds the span or
/// [`kv_splitk::SPLITK_MAX_CHUNKS`](super::kv_splitk::SPLITK_MAX_CHUNKS).
pub fn choose_splitk(request: SeqparRequest, q: &AttentionGeometryQuery) -> Option<SplitKGeometry> {
    use super::kv_splitk::{min_chunks, SPLITK_MAX_CHUNKS};
    if request != SeqparRequest::Unset || q.head_dim == 0 {
        return None;
    }
    let row = SPLITK_MEASURED.iter().find(|m| {
        m.head_dim == q.head_dim
            && m.num_q_heads == q.num_q_heads
            && m.num_kv_heads == q.num_kv_heads
    })?;
    let (_, chunks, slices) = row
        .tiers
        .iter()
        .take_while(|(floor, _, _)| *floor <= q.span)
        .last()
        .copied()?;
    if chunks == 0 {
        return None;
    }
    let chunks = chunks
        .max(min_chunks(q.span))
        .min(q.span.max(1) as usize)
        .min(SPLITK_MAX_CHUNKS);
    let slices = slices.clamp(1, SEQPAR_MAX_THREADS / q.head_dim);
    (chunks > 1).then_some(SplitKGeometry { chunks, slices })
}

fn measured_row(q: &AttentionGeometryQuery) -> Option<&'static MeasuredGeometry> {
    MEASURED.iter().find(|m| {
        m.head_dim == q.head_dim
            && m.num_q_heads == q.num_q_heads
            && m.num_kv_heads == q.num_kv_heads
    })
}

fn tier_slices(row: &MeasuredGeometry, span: u32) -> usize {
    row.tiers
        .iter()
        .take_while(|(floor, _)| *floor <= span)
        .last()
        .map(|(_, slices)| *slices)
        .unwrap_or(0)
}

/// Clamp a slice count to what the kernel can hold, and refuse counts
/// that would not actually partition anything.
fn bounded(slices: usize, head_dim: usize) -> AttentionGeometry {
    if head_dim == 0 || slices <= 1 {
        return AttentionGeometry::Serial;
    }
    let max_by_partial = SEQPAR_MAX_THREADS / head_dim;
    let n = slices.min(max_by_partial);
    if n <= 1 {
        AttentionGeometry::Serial
    } else {
        AttentionGeometry::SeqPar { slices: n }
    }
}

/// Choose the execution geometry for `q` under the operator's request.
///
/// - `Off` → serial on every geometry (an override, not a default).
/// - `Slices(n)` → `n`, clamped to the kernel bound.
/// - `Auto` → the occupancy heuristic (`kv_seqpar::slices_for`), on any
///   geometry: the operator asked for the measured-at-64 span policy and
///   accepts that it is a heuristic elsewhere.
/// - `Unset` → the measured row for exactly this `(head_dim, q_heads,
///   kv_heads)`, or serial when there is none.
pub fn choose_attention_geometry(
    request: SeqparRequest,
    q: &AttentionGeometryQuery,
) -> AttentionGeometry {
    match request {
        SeqparRequest::Off => AttentionGeometry::Serial,
        SeqparRequest::Slices(n) => bounded(n, q.head_dim),
        SeqparRequest::Auto => bounded(
            slices_for(SeqparRequest::Auto, q.head_dim, q.span),
            q.head_dim,
        ),
        SeqparRequest::Unset => match measured_row(q) {
            Some(row) => bounded(tier_slices(row, q.span), q.head_dim),
            None => AttentionGeometry::Serial,
        },
    }
}

#[cfg(test)]
mod tests;
