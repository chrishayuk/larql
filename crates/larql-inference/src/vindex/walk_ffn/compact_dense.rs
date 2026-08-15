//! BW-B — the compiled compact-dense oracle.
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
}
