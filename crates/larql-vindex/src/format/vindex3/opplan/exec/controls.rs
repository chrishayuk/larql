//! The executor's control vocabulary: one deliberate defect per named
//! ordering mistake, across every residual topology it traverses.
//!
//! Its own module because two topologies now own arms of it, and a
//! control enum living inside one of them would read as that topology's
//! property. `Mutation::None` is what production threads; every other
//! value exists so a witness can be made to FAIL, which is the only
//! thing that makes its passing evidence.

/// Deliberate defects, for the negative controls of every residual
/// topology this build traverses.
///
/// Test-only in use, but each perturbs the REAL composition or the real
/// traversal rather than a copy: a control that mutates a duplicate
/// proves only that the duplicate is detectable. Some live in a
/// topology's arithmetic (`hyper_connection::{reduce, update}`,
/// `attention_residual::reduce`); the rest are SEQUENCING defects the
/// traversal applies, because the thing they break — which vector
/// reaches which operator, and in which order — is decided there.
///
/// ONE value threads through one traversal, so this is one type. It
/// lives here rather than in either topology's module because it stopped
/// belonging to hyper-connections the moment a second topology needed
/// controls of its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mutation {
    None,
    /// (a) Run the sublayer single-stream on stream 0 and add its output
    /// back into stream 0: no split, no reduction, no expansion. The
    /// plausible way to "support" the topology without running it.
    BypassComposition,
    /// (b) One Sinkhorn pass instead of the declared count.
    SingleIteration,
    /// (b) Reduce with uniform weights instead of the split's `pre`.
    UniformReduction,
    /// (b) Expand with `comb` transposed — source and destination
    /// streams swapped. The split reported is still the correct one, so
    /// only the expansion disagrees.
    TransposedCombination,
    /// (c) Feed the pre-attention norm stream 0 instead of the reduced
    /// vector.
    PreNormOnStreamZero,
    /// (c) Feed the pre-attention norm the stream mean instead of the
    /// reduced vector.
    PreNormOnStreamMean,
    /// (d) Hand the hybrid FFN stream 0 as its raw residual instead of
    /// the reduced vector — the router and the expert pre-norm read it.
    HybridResidualFromStreamZero,
    /// (d) The same with the stream mean.
    HybridResidualFromStreamMean,
    /// (e, batch only) Apply position 0's split and reduced vector to
    /// every position. Invisible at batch size one; the batch witness
    /// runs three distinguishable positions so it is not.
    SplitFromPositionZero,
    /// (batch only) Exchange positions 0 and 1's bundles between the
    /// reduction and the update, so each position's update carries the
    /// other's state forward. A witness that cannot see per-position
    /// state passes this.
    SwapPositionsBeforeUpdate,

    // ── Attention residuals (K3-ATTNRES-1) ──────────────────────────
    //
    // One per named ordering mistake in the rung's oracle, which
    // measured each against Kimi-K3's own reference before any of this
    // code existed. The measured divergence is quoted on each arm; two
    // of them are EXACTLY zero, and those two are the reason the witness
    // records a site's structure and not only its values.
    /// Append the boundary snapshot BEFORE the attention-site reduce, so
    /// that reduce sees the new set instead of the old one. Oracle
    /// delta 4.62e-01.
    AttnResSiteOverNewSnapshots,
    /// Snapshot the attention site's OUTPUT instead of the entering
    /// prefix state. Oracle delta 9.11e-01, and invisible at layer 0 —
    /// where the attention site is skipped, the two are the same vector.
    AttnResSnapshotIsMixedVector,
    /// Snapshot the post-attention prefix instead of the entering state.
    /// Oracle delta 1.72e+00.
    AttnResSnapshotAfterAttention,
    /// Run the attention site at layer 0, over the single prefix
    /// candidate, instead of skipping it. Oracle delta EXACTLY 0.0:
    /// softmax over one candidate is the identity, so this defect cannot
    /// be caught by any value comparison at any geometry. The witness
    /// catches it by the ABSENCE of a record.
    AttnResLayer0AttentionSiteRuns,
    /// Skip the mlp-site reduce at layer 0 — the honest form of the
    /// "layer 0 mixes one candidate" misreading the oracle falsified.
    /// Oracle delta 4.50e-01.
    AttnResMlpSiteSkippedAtLayer0,
    /// Give the mlp site the attention site's non-empty guard. Oracle
    /// delta EXACTLY 0.0, because the guard never fires: no mlp site in
    /// the schedule ever sees an empty snapshot set. Structural only.
    AttnResMlpSiteGuardedOnNonEmpty,
    /// Mix the RMS-normalised candidates instead of the raw ones. Oracle
    /// delta 1.30e+00.
    AttnResMixOverNormalisedCandidates,
    /// Score the raw candidates instead of the normalised ones. Oracle
    /// delta 5.83e-01.
    AttnResScoreWithoutRmsNorm,
    /// Skip the exit reduction entirely. Oracle delta 2.26e+00.
    AttnResExitSkipped,
    /// Reduce the exit with a layer's pair instead of the shipped output
    /// pair. Oracle delta 5.91e-01 — the control that makes "the SHIPPED
    /// pair" a claim rather than a description.
    AttnResExitUsesALayerPair,
    /// (batch only) Exchange two positions' snapshot histories
    /// immediately before one update, so each position's write carries
    /// the other's state forward. Catches LOSS OF POSITIONAL IDENTITY: a
    /// path that keeps per-position histories but indexes them
    /// inconsistently.
    AttnResSwapPositionHistories,
    /// (batch only) Apply position 0's history to every position at one
    /// reduction. Catches COLLAPSE: a path that built one shared history
    /// and handed it to every row.
    ///
    /// The pair is not redundant. A shared-history implementation passes
    /// the swap in some orderings — swapping two references to the same
    /// state changes nothing — and a mis-indexed one passes the
    /// broadcast, because its histories really are distinct. Neither
    /// control alone separates "vectorised the branch" from "merged the
    /// state", which is the whole invariant of the batch traversal.
    AttnResHistoryFromPositionZero,
    /// (batch only) Write each branch row into the NEXT position's
    /// history, leaving position 0 unwritten and dropping the last row.
    /// Catches MISALIGNMENT of the write, which neither of the other two
    /// reaches.
    ///
    /// The swap is an involution on two rows: a traversal that pairs
    /// deltas with histories by a shifted index still writes every row
    /// into some history, and at positions 2 and beyond the pairing it
    /// produces is the correct one. This control shifts the WHOLE plane,
    /// so the first position receives nothing and every later one
    /// receives its predecessor's branch output — the shape an
    /// off-by-one in a batched write actually takes, and the one a
    /// per-position value check sees at every position at once.
    AttnResWriteOffsetByOne,
}
