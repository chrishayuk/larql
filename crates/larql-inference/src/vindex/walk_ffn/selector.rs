//! Joint-criterion feature selection — tests whether the K=200 walk
//! failure is a *selection* problem (gate-score ranking is wrong) or a
//! *coverage* problem (the top-K-by-anything just isn't enough).
//!
//! The current production walk picks top-K by `|gate_score|`. The
//! actual contribution of feature `i` to the residual at this position
//! is `silu(gate_i) · up_i · down_row_i`. Ranking by `|gate|` alone
//! ignores the magnitudes of the `up` and `down` components.
//!
//! This module exposes:
//! - `down_row_norms(layer)` — lazy per-layer cache of `‖down_row‖`
//!   (computed from the dequantised down cache).
//! - `up_row_norms(layer)`   — same, for `‖up_row‖` (Q4K up).
//! - `joint_gate_knn(layer, residual, top_k, kind)` — get full gate
//!   scores via `gate_scores_batch_backend`, weight by the chosen
//!   joint criterion, take top-K. Returns `(feat_idx, raw_gate_score)`
//!   so the FFN math downstream is unchanged.

use std::sync::Arc;

use ndarray::{Array1, Array2};

use super::helpers::selection_weight_cmp_desc;
use super::shortlist::{criterion_inputs, criterion_weight, rerank_cmp};
use super::WalkFfn;
use crate::vindex::walk_config::FeatureSelector;
use larql_vindex::{FFN_DOWN, FFN_UP};

impl<'a> WalkFfn<'a> {
    /// Public view of `down_row_norms` for probes/examples. Same lazy
    /// cache.
    pub fn down_row_norms_pub(&self, layer: usize) -> Option<Arc<Vec<f32>>> {
        self.down_row_norms(layer)
    }

    /// Public view of `up_row_norms`.
    pub fn up_row_norms_pub(&self, layer: usize) -> Option<Arc<Vec<f32>>> {
        self.up_row_norms(layer)
    }

    /// Public view of `compute_full_up_scores`.
    pub fn compute_full_up_scores_pub(
        &self,
        layer: usize,
        residual: &Array1<f32>,
    ) -> Option<Vec<f32>> {
        self.compute_full_up_scores(layer, residual)
    }

    /// Lazy per-layer `‖down_row‖`. Triggers
    /// `kquant_ffn_layer(layer, FFN_DOWN)` on first call, then caches
    /// the norms.
    pub(super) fn down_row_norms(&self, layer: usize) -> Option<Arc<Vec<f32>>> {
        if let Some(Some(arc)) = self.down_norms_cache.borrow().get(layer) {
            return Some(Arc::clone(arc));
        }
        let down_data = self.index.kquant_ffn_layer(layer, FFN_DOWN)?;
        let num_features = self.index.num_features(layer);
        let hidden = self.weights.hidden_size;
        if down_data.len() < num_features * hidden {
            return None;
        }
        let mut norms = Vec::with_capacity(num_features);
        for feat in 0..num_features {
            let row = &down_data[feat * hidden..(feat + 1) * hidden];
            let sumsq: f32 = row.iter().map(|v| v * v).sum();
            norms.push(sumsq.sqrt());
        }
        let arc = Arc::new(norms);
        let mut cache = self.down_norms_cache.borrow_mut();
        if cache.len() <= layer {
            cache.resize_with(layer + 1, || None);
        }
        cache[layer] = Some(Arc::clone(&arc));
        Some(arc)
    }

    /// Compute all per-feature up scores `⟨up_row, residual⟩` at this
    /// layer for the given residual. Prefers native f32 + BLAS; falls
    /// back to Q4K `kquant_matmul_transb`. Returns a Vec of length
    /// `num_features`.
    pub(super) fn compute_full_up_scores(
        &self,
        layer: usize,
        residual: &Array1<f32>,
    ) -> Option<Vec<f32>> {
        let num_features = self.index.num_features(layer);
        let hidden = self.weights.hidden_size;
        if num_features == 0 || residual.len() != hidden {
            return None;
        }

        // Native f32 path — BLAS / GPU dot.
        if let Some(up_view) = self.index.up_layer_matrix(layer) {
            let x_2d = Array2::from_shape_vec((1, hidden), residual.to_vec()).ok()?;
            let result = larql_compute::dot_proj_gpu(&x_2d, &up_view, self.backend);
            if result.shape() == [1, num_features] {
                return Some(result.row(0).to_vec());
            }
            return None;
        }

        // Q4K path — batched Q4 matmul against the layer's up bytes.
        let x_slice = residual.as_slice()?;
        let y = self
            .index
            .kquant_matmul_transb(layer, 1, x_slice, 1, self.backend)?;
        if y.len() == num_features {
            Some(y)
        } else {
            None
        }
    }

    /// Lazy per-layer `‖up_row‖`. Triggers
    /// `kquant_ffn_layer(layer, FFN_UP)` on first call, then caches
    /// the norms.
    pub(super) fn up_row_norms(&self, layer: usize) -> Option<Arc<Vec<f32>>> {
        if let Some(Some(arc)) = self.up_norms_cache.borrow().get(layer) {
            return Some(Arc::clone(arc));
        }
        let up_data = self.index.kquant_ffn_layer(layer, FFN_UP)?;
        let num_features = self.index.num_features(layer);
        let hidden = self.weights.hidden_size;
        if up_data.len() < num_features * hidden {
            return None;
        }
        let mut norms = Vec::with_capacity(num_features);
        for feat in 0..num_features {
            let row = &up_data[feat * hidden..(feat + 1) * hidden];
            let sumsq: f32 = row.iter().map(|v| v * v).sum();
            norms.push(sumsq.sqrt());
        }
        let arc = Arc::new(norms);
        let mut cache = self.up_norms_cache.borrow_mut();
        if cache.len() <= layer {
            cache.resize_with(layer + 1, || None);
        }
        cache[layer] = Some(Arc::clone(&arc));
        Some(arc)
    }

    /// Top-K features by a joint criterion. Computes full gate scores
    /// once via `gate_scores_batch_backend`, multiplies by per-feature
    /// weights derived from `kind` (formulas single-sourced in
    /// `shortlist::criterion_weight`), takes top-K by `|weighted|` in
    /// the deterministic `rerank_cmp` order, and returns
    /// `(feat_idx, raw_gate_score)` so the FFN math downstream is
    /// unchanged.
    ///
    /// Falls back to the production `gate_walk` path if the joint norms
    /// can't be computed (e.g. no Q4K cache yet), so the walk still
    /// produces output. A **NaN** weighted score, by contrast, panics
    /// (`selection_weight_cmp_desc`) — the same contract as
    /// `top_k_by_abs` on the production gate-KNN path; it must never
    /// silently scramble the selection.
    pub(super) fn joint_gate_knn(
        &self,
        layer: usize,
        residual: &Array1<f32>,
        top_k: usize,
        kind: FeatureSelector,
    ) -> Vec<(usize, f32)> {
        let num_features = self.index.num_features(layer);
        if num_features == 0 {
            return Vec::new();
        }
        let hidden = self.weights.hidden_size;

        // Full gate scores in one batched gemv.
        let x = ndarray::Array2::from_shape_vec((1, hidden), residual.to_vec())
            .expect("residual shape (1, hidden)");
        let scores = match self
            .index
            .gate_scores_batch_backend(layer, &x, self.backend)
        {
            Some(s) => s,
            None => {
                // No batched gate-score path — fall back to gate_walk
                // (production behaviour). Random selection also lands
                // here since it doesn't need joint norms either.
                return self.fallback_top_k(layer, residual, top_k, kind);
            }
        };
        let row = scores.row(0);
        if row.len() != num_features {
            return self.fallback_top_k(layer, residual, top_k, kind);
        }

        // Per-feature joint weight — formulas single-sourced in
        // `shortlist::criterion_weight` (shared with the two-stage
        // rerank so the two paths cannot drift). Random has no
        // criterion (`criterion_inputs` returns None) and short-circuits.
        let Some(needs) = criterion_inputs(kind) else {
            use rand::seq::SliceRandom;
            let mut rng = rand::thread_rng();
            let mut idxs: Vec<usize> = (0..num_features).collect();
            idxs.shuffle(&mut rng);
            idxs.truncate(top_k.min(num_features));
            return idxs.into_iter().map(|i| (i, row[i])).collect();
        };
        let down_norms = if needs.down_norm {
            match self.down_row_norms(layer) {
                Some(n) => Some(n),
                None => return self.fallback_top_k(layer, residual, top_k, kind),
            }
        } else {
            None
        };
        let up_norms = if needs.up_norm {
            match self.up_row_norms(layer) {
                Some(n) => Some(n),
                None => return self.fallback_top_k(layer, residual, top_k, kind),
            }
        } else {
            None
        };
        let up_scores = if needs.up_score {
            match self.compute_full_up_scores(layer, residual) {
                Some(s) => Some(s),
                None => return self.fallback_top_k(layer, residual, top_k, kind),
            }
        } else {
            None
        };
        let use_gelu = self.selector_use_gelu();
        let at = |v: &Option<Arc<Vec<f32>>>, i: usize| {
            v.as_ref()
                .map(|v| v.get(i).copied().unwrap_or(0.0))
                .unwrap_or(0.0)
        };
        let mut weighted: Vec<(usize, f32, f32)> = row
            .iter()
            .enumerate()
            .map(|(i, &g)| {
                let u = up_scores
                    .as_ref()
                    .map(|v| v.get(i).copied().unwrap_or(0.0))
                    .unwrap_or(0.0);
                let w =
                    criterion_weight(kind, use_gelu, g, u, at(&up_norms, i), at(&down_norms, i));
                (i, g, w)
            })
            .collect();

        // Partial sort to the top-K by weighted score, then a full sort
        // of those K into the deterministic rerank order (`rerank_cmp`)
        // so the reported rank order matches the two-stage path. NaN
        // weights panic (`selection_weight_cmp_desc`) — same contract
        // as `top_k_by_abs` on the production gate-KNN path.
        let take = top_k.min(num_features);
        weighted.select_nth_unstable_by(take.saturating_sub(1).min(num_features - 1), |a, b| {
            selection_weight_cmp_desc(a.2, b.2)
        });
        weighted.truncate(take);
        weighted.sort_unstable_by(rerank_cmp);
        weighted.into_iter().map(|(i, g, _)| (i, g)).collect()
    }

    /// Pool-restricted top-K by gate-score: compute full gate scores
    /// via `gate_scores_batch_backend`, restrict to the supplied pool
    /// of feature indices, take top-K within the pool by `|gate|`.
    /// Returns `(feat_idx, raw_gate_score)` so downstream FFN math is
    /// unchanged.
    pub(super) fn pool_restricted_gate_knn(
        &self,
        layer: usize,
        residual: &Array1<f32>,
        top_k: usize,
        pool: &[usize],
    ) -> Vec<(usize, f32)> {
        if pool.is_empty() {
            return Vec::new();
        }
        let hidden = self.weights.hidden_size;
        let x = ndarray::Array2::from_shape_vec((1, hidden), residual.to_vec())
            .expect("residual shape (1, hidden)");
        let scores = match self
            .index
            .gate_scores_batch_backend(layer, &x, self.backend)
        {
            Some(s) => s,
            None => {
                // No batched gate path — fall back to production hits.
                return self.fallback_top_k(layer, residual, top_k, FeatureSelector::GateOnly);
            }
        };
        let row = scores.row(0);

        let mut weighted: Vec<(usize, f32, f32)> = pool
            .iter()
            .filter_map(|&i| {
                if i < row.len() {
                    let g = row[i];
                    Some((i, g, g.abs()))
                } else {
                    None
                }
            })
            .collect();
        if weighted.is_empty() {
            return Vec::new();
        }
        // NaN weights panic — same contract as `top_k_by_abs`.
        let take = top_k.min(weighted.len());
        let nth = take.saturating_sub(1).min(weighted.len() - 1);
        weighted.select_nth_unstable_by(nth, |a, b| selection_weight_cmp_desc(a.2, b.2));
        weighted.truncate(take);
        weighted.into_iter().map(|(i, g, _)| (i, g)).collect()
    }

    /// Precomputed-route gate scoring — the **cheap routing** path.
    ///
    /// Unlike `pool_restricted_gate_knn` (which calls
    /// `gate_scores_batch_backend` = a full gate projection over *all*
    /// features, then filters to the pool), this computes the gate score
    /// for **only the pool features** via per-row Q4K dots — O(|pool|),
    /// not O(num_features). It models hash routing (Exp 27): the feature
    /// set is decided upstream (token-deterministic), so selection never
    /// touches the full gate matrix.
    ///
    /// Returns `(feature, gate_score)` for each pool feature, in pool
    /// order (no ranking needed — the route *is* the selection). Falls
    /// back to `None` when the layer exposes no Q4K interleaved gate
    /// bytes with a registered `row_dot` (the only storage this fast path
    /// supports; callers then route to `pool_restricted_gate_knn`).
    pub(super) fn local_pool_gate_knn(
        &self,
        layer: usize,
        x_slice: &[f32],
        pool: &[usize],
    ) -> Option<Vec<(usize, f32)>> {
        if pool.is_empty() {
            return Some(Vec::new());
        }
        let hidden = self.weights.hidden_size;
        // Interleaved Q4K layout is [gate, up, down]; gate is slot 0.
        let slices = self.index.interleaved_kquant_layer_data(layer)?;
        let (gate_bytes, gate_tag) = (slices[0].0, slices[0].1);
        let info = larql_vindex::quant::registry::lookup(gate_tag)?;
        let row_dot = info.row_dot?;
        let bytes_per_row = info.bytes_per_row(hidden)?;
        let n_rows = gate_bytes.len() / bytes_per_row;
        Some(
            pool.iter()
                .filter_map(|&feat| {
                    if feat >= n_rows {
                        return None;
                    }
                    let start = feat * bytes_per_row;
                    let end = start + bytes_per_row;
                    let g = row_dot(&gate_bytes[start..end], x_slice).unwrap_or(0.0);
                    Some((feat, g))
                })
                .collect(),
        )
    }

    /// Fallback when the joint-scoring path can't run — falls back to
    /// the production `gate_walk` → `gate_knn_q4` → `gate_knn` chain so
    /// the walk still produces output.
    fn fallback_top_k(
        &self,
        layer: usize,
        residual: &Array1<f32>,
        top_k: usize,
        kind: FeatureSelector,
    ) -> Vec<(usize, f32)> {
        if !matches!(kind, FeatureSelector::GateOnly) {
            // A joint selector (or Random null hypothesis) silently
            // degrading to production GateOnly invalidates the A/B —
            // the results carry the wrong label. Make every such call
            // observable (2026-07-30 review, M10).
            self.trace_path(layer, "selector:fallback");
            self.selector_fallbacks
                .set(self.selector_fallbacks.get() + 1);
        }
        self.production_gate_chain(layer, residual, top_k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{
        attach_feature_major_f32_to_test_vindex, make_test_q4k_vindex_with_model_gate,
        make_test_q4k_weights, make_test_vindex, make_test_weights,
    };
    use crate::vindex::{WalkFfn, WalkFfnConfig};
    use larql_vindex::{
        FeatureMeta, FfnRowAccess, Fp4FfnAccess, GateLookup, NativeFfnAccess, PatchOverrides,
        QuantizedFfnAccess,
    };

    /// Index whose batched gate scores contain a NaN — models a
    /// corrupted / overflowed gate projection.
    struct NanGateScoresIndex {
        n_features: usize,
    }

    impl GateLookup for NanGateScoresIndex {
        fn gate_knn(
            &self,
            _layer: usize,
            _residual: &Array1<f32>,
            top_k: usize,
        ) -> Vec<(usize, f32)> {
            (0..top_k.min(self.n_features)).map(|i| (i, 1.0)).collect()
        }
        fn feature_meta(&self, _layer: usize, _feature: usize) -> Option<FeatureMeta> {
            None
        }
        fn num_features(&self, _layer: usize) -> usize {
            self.n_features
        }
        fn gate_scores_batch(&self, _layer: usize, _x: &Array2<f32>) -> Option<Array2<f32>> {
            let mut scores = vec![1.0f32; self.n_features];
            scores[1] = f32::NAN;
            Array2::from_shape_vec((1, self.n_features), scores).ok()
        }
    }
    impl PatchOverrides for NanGateScoresIndex {}
    impl NativeFfnAccess for NanGateScoresIndex {}
    impl QuantizedFfnAccess for NanGateScoresIndex {}
    impl Fp4FfnAccess for NanGateScoresIndex {}

    /// The NaN contract (2026-07-30 review, low tier): a NaN gate score
    /// reaching the selector's top-K must PANIC — the previous
    /// `partial_cmp(..).unwrap_or(Equal)` silently scrambled the
    /// selection, splitting the contract from `top_k_by_abs` (which
    /// deliberately panics) on the production gate-KNN path.
    #[test]
    #[should_panic(expected = "NaN in selector weight")]
    fn joint_gate_knn_panics_on_nan_gate_score() {
        let weights = make_test_weights();
        let index = NanGateScoresIndex { n_features: 4 };
        let walk = crate::vindex::WalkFfn::new(&weights, &index, 2);
        let residual = Array1::from_vec(vec![0.02_f32; weights.hidden_size]);
        walk.joint_gate_knn(0, &residual, 2, FeatureSelector::GateOnly);
    }

    // ── Q4K-fixture selection chain ─────────────────────────────────────

    /// Top-K used by the joint-selector tests — small enough that the
    /// expected set is unambiguous on 256 random-valued features.
    const JOINT_TEST_K: usize = 4;

    fn residual_ramp(hidden: usize) -> Array1<f32> {
        Array1::from_vec((0..hidden).map(|i| (i as f32 + 1.0) * 0.02).collect())
    }

    /// The raw batched gate scores for layer 0 — the ground truth every
    /// joint criterion must return scores from.
    fn raw_gate_scores(index: &larql_vindex::VectorIndex, residual: &Array1<f32>) -> Vec<f32> {
        let x = Array2::from_shape_vec((1, residual.len()), residual.to_vec()).unwrap();
        index
            .gate_scores_batch_backend(0, &x, None)
            .expect("Q4K fixture must produce batched gate scores")
            .row(0)
            .to_vec()
    }

    /// Expected top-K feature set under a per-feature weight vector.
    fn expected_top_k(weights_by_feat: &[f32], k: usize) -> std::collections::HashSet<usize> {
        let mut idx: Vec<usize> = (0..weights_by_feat.len()).collect();
        idx.sort_by(|&a, &b| {
            weights_by_feat[b]
                .partial_cmp(&weights_by_feat[a])
                .expect("test weights are NaN-free")
        });
        idx.truncate(k);
        idx.into_iter().collect()
    }

    /// Check a joint-selector result: K hits, unique in-range features,
    /// scores are the RAW gate scores (not the joint weights), and the
    /// selected set is exactly the top-K by the given joint weight.
    fn assert_joint_selection(
        hits: &[(usize, f32)],
        raw: &[f32],
        joint_weight: &[f32],
        label: &str,
    ) {
        assert_eq!(hits.len(), JOINT_TEST_K, "{label}: wrong hit count");
        let feats: std::collections::HashSet<usize> = hits.iter().map(|(f, _)| *f).collect();
        assert_eq!(feats.len(), JOINT_TEST_K, "{label}: duplicate features");
        for &(f, g) in hits {
            assert!(f < raw.len(), "{label}: feature {f} out of range");
            assert_eq!(
                g, raw[f],
                "{label}: selector must return the RAW gate score for {f}"
            );
        }
        assert_eq!(
            feats,
            expected_top_k(joint_weight, JOINT_TEST_K),
            "{label}: selected set is not the top-K by the joint criterion"
        );
    }

    /// `down_row_norms` / `up_row_norms` must equal the row norms of the
    /// dequantised Q4K cache — the selectors rank on these, so a stride
    /// bug here silently reorders every joint selection (the H1 victim
    /// class). Second call must serve the cached Arc.
    #[test]
    fn row_norm_caches_match_dequant_cache_norms() {
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex_with_model_gate(&weights);
        let walk = WalkFfn::from_config(
            &weights,
            &index,
            WalkFfnConfig::sparse(weights.num_layers, JOINT_TEST_K),
        );
        let hidden = weights.hidden_size;
        let n = index.num_features(0);

        for (component, norms, again) in [
            (
                2usize,
                walk.down_row_norms_pub(0),
                walk.down_row_norms_pub(0),
            ),
            (1usize, walk.up_row_norms_pub(0), walk.up_row_norms_pub(0)),
        ] {
            let norms = norms.expect("Q4K fixture must yield row norms");
            let again = again.expect("second call");
            assert!(
                std::sync::Arc::ptr_eq(&norms, &again),
                "component {component}: second call must hit the lazy cache"
            );
            assert_eq!(norms.len(), n);
            let cache = index
                .kquant_ffn_layer(0, component)
                .expect("dequant cache available");
            for feat in 0..n {
                let row = &cache[feat * hidden..(feat + 1) * hidden];
                let expected: f32 = row.iter().map(|v| v * v).sum::<f32>().sqrt();
                assert_eq!(
                    norms[feat], expected,
                    "component {component}, feature {feat}: norm mismatch"
                );
            }
        }
    }

    /// Q4K arm of `compute_full_up_scores`: the batched Q4K matmul must
    /// agree with the per-row fused Q4K dot (`ffn_row_dot`) — two
    /// independent kernels over the same bytes.
    #[test]
    fn compute_full_up_scores_q4k_matches_per_row_dots() {
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex_with_model_gate(&weights);
        let walk = WalkFfn::from_config(
            &weights,
            &index,
            WalkFfnConfig::sparse(weights.num_layers, JOINT_TEST_K),
        );
        let residual = residual_ramp(weights.hidden_size);
        let scores = walk
            .compute_full_up_scores_pub(0, &residual)
            .expect("Q4K fixture must yield up scores");
        assert_eq!(scores.len(), index.num_features(0));
        let x_slice = residual.as_slice().unwrap();
        for (feat, &s) in scores.iter().enumerate() {
            let row_dot = index
                .ffn_row_dot(0, 1, feat, x_slice)
                .expect("per-row Q4K dot");
            assert!(
                (s - row_dot).abs() <= 1e-4 * row_dot.abs().max(1.0),
                "feature {feat}: batched {s} vs per-row {row_dot}"
            );
        }
    }

    /// Native-f32 arm of `compute_full_up_scores`: BLAS batched dot vs a
    /// hand-computed per-row dot over the same up tensor.
    #[test]
    fn compute_full_up_scores_native_matches_manual_dots() {
        let weights = make_test_weights();
        let mut index = make_test_vindex(&weights);
        attach_feature_major_f32_to_test_vindex(&weights, &mut index);
        let walk = WalkFfn::from_config(
            &weights,
            &index,
            WalkFfnConfig::sparse(weights.num_layers, JOINT_TEST_K),
        );
        let residual = residual_ramp(weights.hidden_size);
        let scores = walk
            .compute_full_up_scores_pub(0, &residual)
            .expect("native fixture must yield up scores");
        let up = &weights.tensors[&weights.arch.ffn_up_key(0)];
        assert_eq!(scores.len(), weights.intermediate_size);
        for (feat, &s) in scores.iter().enumerate() {
            let manual: f32 = (0..weights.hidden_size)
                .map(|h| up[[feat, h]] * residual[h])
                .sum();
            assert!(
                (s - manual).abs() <= 1e-5 * manual.abs().max(1.0),
                "feature {feat}: BLAS {s} vs manual {manual}"
            );
        }
    }

    /// Every joint criterion, driven end-to-end on the Q4K fixture:
    /// the selection must be exactly the top-K by that criterion's
    /// weight (computed independently here from the raw scores + norms),
    /// and the returned scores must be the raw gate scores.
    #[test]
    fn joint_gate_knn_ranks_by_each_criterion_on_q4k_fixture() {
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex_with_model_gate(&weights);
        let walk = WalkFfn::from_config(
            &weights,
            &index,
            WalkFfnConfig::sparse(weights.num_layers, JOINT_TEST_K),
        );
        let residual = residual_ramp(weights.hidden_size);
        let raw = raw_gate_scores(&index, &residual);
        let dn = walk.down_row_norms_pub(0).expect("down norms");
        let un = walk.up_row_norms_pub(0).expect("up norms");
        let us = walk
            .compute_full_up_scores_pub(0, &residual)
            .expect("up scores");
        // Gemma-3 fixture → GeluTanh activation, matching the selector's
        // `use_gelu` branch for ActXUpScoreXDownNorm.
        assert!(weights.arch.gate_up_is_gelu_tanh());

        let cases: Vec<(FeatureSelector, Vec<f32>, &str)> = vec![
            (
                FeatureSelector::GateOnly,
                raw.iter().map(|g| g.abs()).collect(),
                "GateOnly",
            ),
            (
                FeatureSelector::GateXDownNorm,
                raw.iter()
                    .enumerate()
                    .map(|(i, g)| g.abs() * dn[i])
                    .collect(),
                "GateXDownNorm",
            ),
            (
                FeatureSelector::GateXUpDownNorm,
                raw.iter()
                    .enumerate()
                    .map(|(i, g)| g.abs() * dn[i] * un[i])
                    .collect(),
                "GateXUpDownNorm",
            ),
            (
                FeatureSelector::GateXUpScore,
                raw.iter()
                    .enumerate()
                    .map(|(i, g)| g.abs() * us[i].abs())
                    .collect(),
                "GateXUpScore",
            ),
            (
                FeatureSelector::ActXUpScoreXDownNorm,
                raw.iter()
                    .enumerate()
                    .map(|(i, &g)| (crate::ffn::gelu_tanh(g) * us[i]).abs() * dn[i])
                    .collect(),
                "ActXUpScoreXDownNorm",
            ),
        ];
        for (kind, joint_weight, label) in cases {
            let hits = walk.joint_gate_knn(0, &residual, JOINT_TEST_K, kind);
            assert_joint_selection(&hits, &raw, &joint_weight, label);
        }
        assert_eq!(
            walk.selector_fallback_count(),
            0,
            "no joint call may have silently degraded to GateOnly (M10)"
        );
    }

    /// `Random` needs no determinism assertion — but it must return
    /// exactly K unique valid features carrying their raw gate scores.
    #[test]
    fn joint_gate_knn_random_returns_k_valid_features_with_raw_scores() {
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex_with_model_gate(&weights);
        let walk = WalkFfn::from_config(
            &weights,
            &index,
            WalkFfnConfig::sparse(weights.num_layers, JOINT_TEST_K),
        );
        let residual = residual_ramp(weights.hidden_size);
        let raw = raw_gate_scores(&index, &residual);
        let hits = walk.joint_gate_knn(0, &residual, JOINT_TEST_K, FeatureSelector::Random);
        assert_eq!(hits.len(), JOINT_TEST_K);
        let feats: std::collections::HashSet<usize> = hits.iter().map(|(f, _)| *f).collect();
        assert_eq!(feats.len(), JOINT_TEST_K, "random draw must be unique");
        for &(f, g) in &hits {
            assert!(f < raw.len());
            assert_eq!(g, raw[f], "random hits still carry the raw gate score");
        }
    }

    /// `pool_restricted_gate_knn`: top-K by |gate| WITHIN the pool only,
    /// out-of-range pool entries filtered, raw scores returned, empty
    /// pool short-circuits.
    #[test]
    fn pool_restricted_gate_knn_respects_the_pool() {
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex_with_model_gate(&weights);
        let walk = WalkFfn::from_config(
            &weights,
            &index,
            WalkFfnConfig::sparse(weights.num_layers, JOINT_TEST_K),
        );
        let residual = residual_ramp(weights.hidden_size);
        let raw = raw_gate_scores(&index, &residual);

        assert!(walk
            .pool_restricted_gate_knn(0, &residual, 2, &[])
            .is_empty());

        let pool = [3usize, 250, 17, 99, usize::MAX]; // MAX must be filtered
        let k = 2;
        let hits = walk.pool_restricted_gate_knn(0, &residual, k, &pool);
        assert_eq!(hits.len(), k);
        // Expected: top-2 by |raw| among the valid pool entries.
        let mut valid: Vec<usize> = pool.iter().copied().filter(|&f| f < raw.len()).collect();
        valid.sort_by(|&a, &b| {
            raw[b]
                .abs()
                .partial_cmp(&raw[a].abs())
                .expect("NaN-free fixture")
        });
        let expected: std::collections::HashSet<usize> = valid[..k].iter().copied().collect();
        let got: std::collections::HashSet<usize> = hits.iter().map(|(f, _)| *f).collect();
        assert_eq!(
            got, expected,
            "must be the top-{k} by |gate| within the pool"
        );
        for &(f, g) in &hits {
            assert!(pool.contains(&f), "hit {f} escaped the pool");
            assert_eq!(g, raw[f]);
        }
    }

    /// Index without batched gate scores: GateOnly falls back to the
    /// production chain WITHOUT counting a selector fallback (the label
    /// is still truthful), while Random — a joint-null selector — must
    /// count and trace the degradation (M10).
    struct NoBatchIndex {
        n_features: usize,
    }

    impl GateLookup for NoBatchIndex {
        fn gate_knn(
            &self,
            _layer: usize,
            _residual: &Array1<f32>,
            top_k: usize,
        ) -> Vec<(usize, f32)> {
            (0..top_k.min(self.n_features))
                .map(|i| (i, 1.0 / (i as f32 + 1.0)))
                .collect()
        }
        fn feature_meta(&self, _layer: usize, _feature: usize) -> Option<FeatureMeta> {
            None
        }
        fn num_features(&self, _layer: usize) -> usize {
            self.n_features
        }
    }
    impl PatchOverrides for NoBatchIndex {}
    impl NativeFfnAccess for NoBatchIndex {}
    impl QuantizedFfnAccess for NoBatchIndex {}
    impl Fp4FfnAccess for NoBatchIndex {}

    #[test]
    fn gate_only_fallback_is_not_counted_as_degradation() {
        let weights = make_test_weights();
        let index = NoBatchIndex { n_features: 8 };
        let walk = WalkFfn::new(&weights, &index, 4).with_dispatch_trace();
        let residual = residual_ramp(weights.hidden_size);

        let hits = walk.joint_gate_knn(0, &residual, 4, FeatureSelector::GateOnly);
        assert_eq!(hits.len(), 4, "production chain still selects");
        assert_eq!(
            walk.selector_fallback_count(),
            0,
            "GateOnly fallback IS the production selector — not a lie"
        );
        assert!(walk.take_dispatch_trace().is_empty());

        let hits = walk.joint_gate_knn(0, &residual, 4, FeatureSelector::Random);
        assert_eq!(hits.len(), 4);
        assert_eq!(
            walk.selector_fallback_count(),
            1,
            "Random degrading to GateOnly invalidates the null — count it"
        );
        let trace = walk.take_dispatch_trace();
        assert_eq!(trace.len(), 1);
        assert_eq!(trace[0].path, "selector:fallback");
    }

    /// Batched scores with the wrong width (≠ num_features) must not be
    /// trusted — the selector falls back to the production chain.
    struct WrongWidthIndex {
        n_features: usize,
    }

    impl GateLookup for WrongWidthIndex {
        fn gate_knn(
            &self,
            _layer: usize,
            _residual: &Array1<f32>,
            top_k: usize,
        ) -> Vec<(usize, f32)> {
            (0..top_k.min(self.n_features)).map(|i| (i, 0.5)).collect()
        }
        fn feature_meta(&self, _layer: usize, _feature: usize) -> Option<FeatureMeta> {
            None
        }
        fn num_features(&self, _layer: usize) -> usize {
            self.n_features
        }
        fn gate_scores_batch(&self, _layer: usize, _x: &Array2<f32>) -> Option<Array2<f32>> {
            // One column short — a truncated/corrupt gate projection.
            Some(Array2::zeros((1, self.n_features - 1)))
        }
    }
    impl PatchOverrides for WrongWidthIndex {}
    impl NativeFfnAccess for WrongWidthIndex {}
    impl QuantizedFfnAccess for WrongWidthIndex {}
    impl Fp4FfnAccess for WrongWidthIndex {}

    #[test]
    fn joint_gate_knn_rejects_wrong_width_batch_scores() {
        let weights = make_test_weights();
        let index = WrongWidthIndex { n_features: 8 };
        let walk = WalkFfn::new(&weights, &index, 4);
        let residual = residual_ramp(weights.hidden_size);
        let hits = walk.joint_gate_knn(0, &residual, 4, FeatureSelector::GateOnly);
        // Fallback chain serves gate_knn's hits, whose scores are 0.5.
        assert_eq!(hits.len(), 4);
        assert!(hits.iter().all(|&(_, g)| g == 0.5));
    }
}
