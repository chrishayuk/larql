//! BW-B — the compiled compact-dense oracle.
//!
//! Ported verbatim from BW-B (`bw10-movement-ledger`, uncommitted) —
//! duplicated here only because that worktree is concurrently in use
//! by another session; reconcile/de-duplicate before merging.
//!
//! `sparse_gather.rs`'s `gather_q4k_accumulate` re-gathers the route's
//! Q4K rows into contiguous buffers on EVERY call. R4 measured that
//! kernel capturing only 23-40% of its own row-count reduction
//! (`docs/diagnoses/walk-ffn-r4-zeroout.md`), rising with K in a shape
//! that reads as fixed per-call overhead, not a routing-quality
//! problem. LA-6/LA-7 independently found scattered magnitude-selected
//! columns touching 29-89x their logical byte count under block-
//! quantized layouts, then closed the whole dynamic-selection family
//! while leaving one door explicitly open (`docs/vindex2-la.md`,
//! branch `worktree-vindex2-la`):
//!
//! > an offline-compiled, cross-input-STABLE contiguous representation
//! > — neither experiment tested this.
//!
//! `CompactDenseLayer` is that representation. For a FIXED, cross-call-
//! stable feature set, [`CompactDenseLayer::materialize`] gathers the
//! rows ONCE — an offline/setup cost, never inside a timed call — into
//! the exact same buffer shape `gather_kquant_rows` produces. Every
//! subsequent [`WalkFfn::compact_dense_forward`] call then runs
//! `score_and_accumulate`, the SAME fused Q4K kernels
//! `gather_q4k_accumulate` uses, with zero gather-copy cost paid per
//! call.
//!
//! Because both arms share `gather_kquant_rows` (what gets copied) and
//! `score_and_accumulate` (what runs on it), a measured gap between
//! `gather_q4k_accumulate` and `compact_dense_forward` at the same K,
//! same layer, same route can only be the "when" of the gather — never
//! a different kernel, a different byte selection, or a different
//! quantisation path. That isolation is the whole point of BW-B: R4's
//! naive "dequant the gather then call BLAS" variant already lost
//! (0.12x, alloc-dominated — `examples/walk_ffn_gather_gemm.rs`), so
//! the open question was never "can *any* compact form win" but
//! specifically "does paying the gather cost every call explain the
//! loss, once you take that cost out of the critical path."
//!
//! Deliberately NOT wired into [`super::super::walk_config::WalkFfnConfig`]
//! or the routing ladder (`mod.rs`'s priority table) — this is a
//! benchmark-only probe (see `docs/diagnoses/bw10-live-gate.md`, BW-B),
//! not a production rung. Whether it becomes one depends on what it
//! measures.
//!
//! BW-B2 addendum (this worktree, `glm-capability-ladder`): the
//! `patch`-capable variant below (a slot-stable route so a partial mask
//! change can overwrite only the changed rows without reallocating or
//! shifting anything else) is NEW here, not part of BW-B/BW-B1's
//! original scope — see `examples/bwb2_incremental_compact.rs` for why
//! and what it measures.

use super::sparse_gather::{gather_kquant_rows, score_and_accumulate, GatheredRows};
use super::WalkFfn;
use larql_vindex::GateIndex;

/// One layer's compiled compact-dense gate/up/down, for a fixed feature
/// set. Built once via [`Self::materialize`] and reused across every
/// call — the buffers never change after construction.
pub struct CompactDenseLayer {
    rows: GatheredRows,
    /// The feature set this layer was compiled from, in gather order —
    /// kept for `feature_count` and so a caller can confirm which
    /// route a given `CompactDenseLayer` represents.
    feats: Vec<usize>,
}

impl CompactDenseLayer {
    /// Gather `feats`' Q4K rows once, offline. `None` under the same
    /// conditions [`gather_kquant_rows`] declines: no Q4K bytes for
    /// this layer, no feature-major down sidecar, an out-of-range
    /// feature index, or an empty set — the caller has no compact
    /// representation to fall back to in that case, by design (this is
    /// a probe, not a production rung with a safe fallback).
    pub fn materialize(
        index: &dyn GateIndex,
        layer: usize,
        feats: &[usize],
        hidden: usize,
    ) -> Option<Self> {
        let rows = gather_kquant_rows(index, layer, feats, hidden)?;
        Some(Self {
            rows,
            feats: feats.to_vec(),
        })
    }

    /// Rows this layer was compiled from.
    pub fn feature_count(&self) -> usize {
        self.feats.len()
    }

    /// The feature set this layer was compiled from, in gather order.
    pub fn features(&self) -> &[usize] {
        &self.feats
    }

    /// Physical bytes this compact layer occupies — gate + up + down,
    /// all K rows. This is the ENTIRE cost `materialize` pays; nothing
    /// further is gathered or copied by any later forward call.
    pub fn physical_bytes(&self) -> u64 {
        (self.rows.gate.len() + self.rows.up.len() + self.rows.down.len()) as u64
    }

    /// BW-B2: overwrite only the SLOTS whose feature changed, leaving
    /// every other slot's bytes and byte offset untouched.
    ///
    /// `changes` is `(slot, new_feature)` pairs — `slot` indexes
    /// directly into the fixed-width row buffers (`slot * bytes_per_row
    /// .. (slot+1) * bytes_per_row`), NOT a feature ID and NOT a
    /// position that shifts when other slots change. This is only
    /// representationally cheap because `GatheredRows` stores each row
    /// at a byte offset determined purely by its SLOT INDEX (position
    /// in the original `feats` order at materialize time), never by
    /// the feature ID's value or by a sorted/canonical order — see the
    /// module doc on `sparse_gather::GatheredRows` and BW-B2's write-up
    /// for why that specific property is what makes patching cheap
    /// instead of a disguised full rebuild.
    ///
    /// `changes` slots are validated against `self.rows.k`; `hidden` is
    /// the same model hidden size the original `materialize` call used
    /// (needed to look up gate/up `bytes_per_row` for the one-row
    /// gather below — same contract as `materialize`'s `hidden`
    /// parameter, the caller must pass the same value both times).
    ///
    /// Returns `None` under the same conditions `materialize` declines
    /// (out-of-range feature, no Q4K bytes, no down sidecar) or if the
    /// gathered row's format/width disagrees with the buffer it would
    /// be written into — on `None` the whole `CompactDenseLayer` is
    /// left unmodified, so a caller can safely fall back to a full
    /// `materialize` without having partially patched anything.
    ///
    /// Gathers ALL `changes`' new features in ONE call to
    /// `gather_kquant_rows` (one triple of allocations, sized
    /// `changes.len() * bytes_per_row`, not `self.rows.k *
    /// bytes_per_row`) and then scatters each gathered row into its
    /// slot. A first implementation of this method called
    /// `gather_kquant_rows` once PER changed slot instead — functionally
    /// identical bytes, but `changes.len()` separate small heap
    /// allocations plus that many repeated `GateIndex` lookups measured
    /// SLOWER end-to-end than a full rebuild at every non-trivial
    /// changed-slot count (R4/BW-B's "small-allocation-dominated" trap,
    /// reproduced here in miniature) — see BW-B2's write-up. Batching
    /// the gather is what makes `changes.len()`-proportional (not
    /// per-call-fixed-cost-proportional) patching real.
    pub fn patch(
        &mut self,
        index: &dyn GateIndex,
        layer: usize,
        hidden: usize,
        changes: &[(usize, usize)],
    ) -> Option<()> {
        if changes.is_empty() {
            return Some(());
        }
        let (gbpr, ubpr, dbpr) = (
            self.rows.gate_bytes_per_row,
            self.rows.up_bytes_per_row,
            self.rows.down_bytes_per_row,
        );
        for &(slot, _) in changes {
            if slot >= self.rows.k {
                return None; // slot out of range for this layer's buffers
            }
        }
        // One batched gather for every entered feature, in `changes`
        // order — the only allocation this call pays, sized to
        // `changes.len()` rows, never `self.rows.k` rows.
        let new_feats: Vec<usize> = changes.iter().map(|&(_, feat)| feat).collect();
        let batch = gather_kquant_rows(index, layer, &new_feats, hidden)?;
        if batch.gate_bytes_per_row != gbpr
            || batch.up_bytes_per_row != ubpr
            || batch.down_bytes_per_row != dbpr
            || batch.gate_format != self.rows.gate_format
            || batch.up_format != self.rows.up_format
            || batch.down_format != self.rows.down_format
        {
            // Format/row-width drift between the original materialize
            // and this patch call — bail rather than write
            // mismatched-width bytes into a fixed-stride buffer.
            return None;
        }
        // Scatter: row `i` of the batch (feature `new_feats[i]`) goes
        // into slot `changes[i].0` of the persistent buffers — the
        // ONLY bytes this call touches outside the fresh batch
        // allocation itself.
        for (i, &(slot, new_feat)) in changes.iter().enumerate() {
            self.rows.gate[slot * gbpr..(slot + 1) * gbpr]
                .copy_from_slice(&batch.gate[i * gbpr..(i + 1) * gbpr]);
            self.rows.up[slot * ubpr..(slot + 1) * ubpr]
                .copy_from_slice(&batch.up[i * ubpr..(i + 1) * ubpr]);
            self.rows.down[slot * dbpr..(slot + 1) * dbpr]
                .copy_from_slice(&batch.down[i * dbpr..(i + 1) * dbpr]);
            self.feats[slot] = new_feat;
        }
        Some(())
    }
}

impl<'a> WalkFfn<'a> {
    /// Run the fused gate/up/down kernels over an already-materialized
    /// compact layer. No gather, no copy — every byte touched here was
    /// contiguous before this call started; the only work paid inside
    /// the timed window is `score_and_accumulate`'s row-dot / scaled-add
    /// passes, identical to what `gather_q4k_accumulate` runs on
    /// freshly-gathered rows.
    pub fn compact_dense_forward(
        &self,
        compact: &CompactDenseLayer,
        x_slice: &[f32],
        use_gelu: bool,
        hidden: usize,
    ) -> Option<Vec<f32>> {
        score_and_accumulate(
            &compact.rows,
            x_slice,
            use_gelu,
            hidden,
            self.config.effective_activation_floor(),
        )
        .map(|ga| ga.out)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::test_utils::{
        attach_down_features_q4k_to_test_vindex, make_test_q4k_vindex, make_test_q4k_weights,
    };
    use crate::vindex::{WalkFfn, WalkFfnConfig};

    fn x_slice(hidden: usize) -> Vec<f32> {
        (0..hidden).map(|i| (i as f32 + 1.0) * 0.02).collect()
    }

    fn rel_l2(a: &[f32], b: &[f32]) -> f32 {
        let num: f32 = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>()
            .sqrt();
        let den: f32 = b.iter().map(|v| v * v).sum::<f32>().sqrt();
        num / den.max(f32::MIN_POSITIVE)
    }

    /// The safety property `gather_q4k_accumulate` pins for the shared
    /// gather step: no feature-major down sidecar → `materialize`
    /// declines rather than reading the transposed interleaved down as
    /// if it were gatherable.
    #[test]
    fn materialize_declines_without_down_sidecar() {
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let hidden = weights.hidden_size;
        assert!(!index.has_down_features_kquant());
        assert!(CompactDenseLayer::materialize(&index, 0, &[0, 1, 2, 3], hidden).is_none());
    }

    /// An empty feature set has nothing to compile.
    #[test]
    fn materialize_declines_on_empty_route() {
        let weights = make_test_q4k_weights();
        let mut index = make_test_q4k_vindex(&weights);
        attach_down_features_q4k_to_test_vindex(&weights, &mut index);
        let hidden = weights.hidden_size;
        assert!(CompactDenseLayer::materialize(&index, 0, &[], hidden).is_none());
    }

    /// The load-bearing parity property: for the SAME route and input,
    /// the compiled-once forward must reproduce the gather-every-call
    /// kernel exactly — both run `score_and_accumulate` over byte-
    /// identical buffers built by the same `gather_kquant_rows`, so a
    /// divergence here would mean the two arms are silently comparing
    /// different kernels, not different gather timing.
    #[test]
    fn compact_dense_forward_matches_gather_q4k_accumulate_bit_exact() {
        let weights = make_test_q4k_weights();
        let mut index = make_test_q4k_vindex(&weights);
        attach_down_features_q4k_to_test_vindex(&weights, &mut index);
        let hidden = weights.hidden_size;
        let pool: Vec<usize> = (0..weights.intermediate_size).collect();
        let x = x_slice(hidden);

        let cfg = WalkFfnConfig::sparse(weights.num_layers, pool.len())
            .with_pool_per_layer(Arc::new(vec![pool.clone(); weights.num_layers]))
            .with_precomputed_routing(true);
        let ffn = WalkFfn::from_config(&weights, &index, cfg);

        let gathered = ffn
            .gather_q4k_accumulate(0, &pool, &x, false, hidden)
            .expect("gather-every-call kernel runs on this route");

        let compact = CompactDenseLayer::materialize(&index, 0, &pool, hidden)
            .expect("materialize succeeds with the sidecar attached");
        assert_eq!(compact.feature_count(), pool.len());
        let compiled = ffn
            .compact_dense_forward(&compact, &x, false, hidden)
            .expect("compiled-once kernel runs on the same route");

        let err = rel_l2(&compiled, &gathered.out);
        assert!(
            err < 1e-6,
            "compiled-once must reproduce gather-every-call bit-for-bit (same kernel, same \
             bytes, only the gather timing differs) — rel L2 = {err}"
        );
    }

    /// `physical_bytes` accounts for gate + up + down, all K rows —
    /// this is the entire byte cost BW-B's "compile once" arm pays;
    /// nothing further is gathered by any later forward call.
    #[test]
    fn physical_bytes_sums_gate_up_down() {
        let weights = make_test_q4k_weights();
        let mut index = make_test_q4k_vindex(&weights);
        attach_down_features_q4k_to_test_vindex(&weights, &mut index);
        let hidden = weights.hidden_size;
        let pool: Vec<usize> = (0..4).collect();
        let compact = CompactDenseLayer::materialize(&index, 0, &pool, hidden).unwrap();
        assert!(compact.physical_bytes() > 0);
        assert_eq!(compact.features(), pool.as_slice());
    }

    /// BW-B2: a patched layer must reproduce a from-scratch materialize
    /// of the POST-patch feature set bit-for-bit at the changed slots —
    /// patching is only useful if it lands the exact same bytes a full
    /// rebuild would have, just without touching the unchanged slots.
    #[test]
    fn patch_matches_full_rebuild_of_post_patch_set() {
        let weights = make_test_q4k_weights();
        let mut index = make_test_q4k_vindex(&weights);
        attach_down_features_q4k_to_test_vindex(&weights, &mut index);
        let hidden = weights.hidden_size;
        // K=8 route out of `intermediate_size` features so there's room
        // to swap a couple of slots for features outside the original set.
        let pool: Vec<usize> = (0..8).collect();
        let mut compact = CompactDenseLayer::materialize(&index, 0, &pool, hidden)
            .expect("materialize succeeds with the sidecar attached");

        // Swap slot 2 (feature 2 -> feature 9) and slot 5 (feature 5 ->
        // feature 10) — two changed slots out of eight, everything else
        // must stay byte-identical.
        let changes = [(2usize, 9usize), (5usize, 10usize)];
        compact
            .patch(&index, 0, hidden, &changes)
            .expect("patch succeeds — in-range slots, sidecar attached");

        let mut expected_feats = pool.clone();
        expected_feats[2] = 9;
        expected_feats[5] = 10;
        assert_eq!(
            compact.features(),
            expected_feats.as_slice(),
            "patch must update the slot->feature mapping at exactly the patched slots"
        );

        let rebuilt = CompactDenseLayer::materialize(&index, 0, &expected_feats, hidden)
            .expect("full rebuild of the post-patch set succeeds");

        let x = x_slice(hidden);
        let cfg = WalkFfnConfig::sparse(weights.num_layers, pool.len());
        let ffn = WalkFfn::from_config(&weights, &index, cfg);
        let patched_out = ffn
            .compact_dense_forward(&compact, &x, false, hidden)
            .expect("patched layer runs");
        let rebuilt_out = ffn
            .compact_dense_forward(&rebuilt, &x, false, hidden)
            .expect("rebuilt layer runs");

        let err = rel_l2(&patched_out, &rebuilt_out);
        assert!(
            err < 1e-6,
            "patch must reproduce a full rebuild of the same post-patch feature set — \
             rel L2 = {err}"
        );
    }

    /// A no-op patch (empty `changes`) must leave the layer unchanged.
    #[test]
    fn patch_with_no_changes_is_a_no_op() {
        let weights = make_test_q4k_weights();
        let mut index = make_test_q4k_vindex(&weights);
        attach_down_features_q4k_to_test_vindex(&weights, &mut index);
        let hidden = weights.hidden_size;
        let pool: Vec<usize> = (0..4).collect();
        let mut compact = CompactDenseLayer::materialize(&index, 0, &pool, hidden).unwrap();
        let bytes_before = compact.physical_bytes();
        compact.patch(&index, 0, hidden, &[]).unwrap();
        assert_eq!(compact.features(), pool.as_slice());
        assert_eq!(compact.physical_bytes(), bytes_before);
    }

    /// Safety rail: an out-of-range slot must decline and leave the
    /// layer unmodified rather than panicking on the buffer index.
    #[test]
    fn patch_declines_on_out_of_range_slot() {
        let weights = make_test_q4k_weights();
        let mut index = make_test_q4k_vindex(&weights);
        attach_down_features_q4k_to_test_vindex(&weights, &mut index);
        let hidden = weights.hidden_size;
        let pool: Vec<usize> = (0..4).collect();
        let mut compact = CompactDenseLayer::materialize(&index, 0, &pool, hidden).unwrap();
        assert!(compact.patch(&index, 0, hidden, &[(99, 1)]).is_none());
    }

    /// Safety rail: an out-of-range replacement FEATURE (not slot) must
    /// decline the same way `materialize` would for that feature.
    #[test]
    fn patch_declines_on_out_of_range_feature() {
        let weights = make_test_q4k_weights();
        let mut index = make_test_q4k_vindex(&weights);
        attach_down_features_q4k_to_test_vindex(&weights, &mut index);
        let hidden = weights.hidden_size;
        let pool: Vec<usize> = (0..4).collect();
        let mut compact = CompactDenseLayer::materialize(&index, 0, &pool, hidden).unwrap();
        let out_of_range = weights.intermediate_size * 10;
        assert!(compact
            .patch(&index, 0, hidden, &[(0, out_of_range)])
            .is_none());
    }
}
