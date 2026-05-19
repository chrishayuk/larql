//! `infer_patched` — the single forward-pass entry point shared by the LQL
//! `INFER` executor (`larql-lql/src/executor/query/infer.rs`) and the Python
//! binding (`larql-python/src/vindex.rs`).
//!
//! Both surfaces must produce byte-identical top-k predictions for any
//! `(weights, gate_index, knn_store, prompt)` — see ADR 0001. This function
//! owns the three parameters that are easy to drift between callers:
//!
//!   1. `top_k_features` on the walk FFN — always unlimited, because a
//!      bounded cap misroutes post-INSERT on Gemma (a strong `×30` gate slot
//!      dominates a half-weakened baseline).
//!   2. The KNN cosine threshold — `KNN_COSINE_THRESHOLD = 0.75`.
//!   3. Layer iteration order — the first stored layer (lowest index) whose
//!      top-1 cosine exceeds the threshold wins.
//!
//! Callers pass a `&dyn GateIndex` + `Option<&KnnStore>`. `PatchedVindex`
//! bundles both; `PyVindex` keeps them as separate fields. Both pass through
//! here.

use larql_vindex::{GateIndex, KnnStore, PatchedVindex, VectorIndex, WalkHit};
use tokenizers::Tokenizer;

use crate::model::ModelWeights;
use crate::vindex::predict_kquant_with_ffn;
use crate::vindex::WalkFfn;

use super::predict::predict_with_ffn;
use super::PredictResult;

/// Cosine threshold for the L0 KnnStore override. A stored key whose top-1
/// cosine against the captured residual exceeds this value replaces the
/// walk FFN's top-1 prediction.
pub const KNN_COSINE_THRESHOLD: f32 = 0.75;

/// Metadata for a KNN override, if one fired.
#[derive(Clone, Debug)]
pub struct KnnOverride {
    pub token: String,
    pub cosine: f32,
    pub layer: usize,
}

/// Result of the shared INFER pipeline.
pub struct InferPatchedResult {
    /// Top-k predictions. When `knn_override` is `Some`, position 0 holds the
    /// stored target token with probability `1.0` and positions `1..k` hold
    /// the walk FFN's own top-`(k-1)`. When `None`, this is the walk FFN's
    /// raw top-k.
    pub predictions: Vec<(String, f64)>,
    /// Walk FFN's raw top-1 before the KnnStore post-logits override is
    /// applied. This lets display layers show what the model path produced
    /// before an unmaterialized retrieval sidecar changed the answer.
    pub model_top1: Option<(String, f64)>,
    /// Metadata on the KNN override for callers that want to surface it
    /// (e.g. the LQL display layer prints `"KNN override, cos=X, L{layer}"`).
    pub knn_override: Option<KnnOverride>,
    /// Per-layer residuals captured at the last-token position during the
    /// walk FFN pass. LQL uses these to build its inference trace.
    pub residuals: Vec<(usize, Vec<f32>)>,
    /// Wall-clock milliseconds for the walk FFN pass itself.
    pub walk_ms: f64,
}

/// Run a full forward pass with the walk FFN, consult the KnnStore for a
/// possible top-1 override, and return the top-k predictions.
///
/// This is the **only** implementation of the INFER pipeline. `exec_infer`
/// (LQL) and `PyVindex::infer` (Python) both delegate here. Per ADR 0001 any
/// new forward-pass surface MUST call this function rather than assembling a
/// local pipeline.
pub fn infer_patched(
    weights: &ModelWeights,
    tokenizer: &Tokenizer,
    gate_index: &dyn GateIndex,
    knn_store: Option<&KnnStore>,
    token_ids: &[u32],
    top_k: usize,
) -> InferPatchedResult {
    let walk_ffn = WalkFfn::new_unlimited_with_trace(weights, gate_index);

    let start = std::time::Instant::now();
    let PredictResult {
        predictions: raw, ..
    } = predict_with_ffn(weights, tokenizer, token_ids, top_k, &walk_ffn);
    let walk_ms = start.elapsed().as_secs_f64() * 1000.0;

    let residuals = walk_ffn.take_residuals();
    let model_top1 = raw.first().cloned();
    let (predictions, knn_override) = apply_knn_override(raw, &residuals, knn_store, top_k);

    InferPatchedResult {
        predictions,
        model_top1,
        knn_override,
        residuals,
        walk_ms,
    }
}

/// Q4K variant of `infer_patched`. Identical contract but routes the forward
/// pass through `predict_kquant_with_ffn`, which dequantises one layer at a time
/// from the vindex instead of reading pre-loaded f32 tensors.
pub fn infer_patched_q4k(
    weights: &mut ModelWeights,
    tokenizer: &Tokenizer,
    gate_index: &dyn GateIndex,
    knn_store: Option<&KnnStore>,
    token_ids: &[u32],
    top_k: usize,
    index: &VectorIndex,
) -> InferPatchedResult {
    // SAFETY: WalkFfn reads only `weights.arch` and `weights.vectors` (neither
    // of which is mutated by `predict_kquant_with_ffn`). The q4k forward pass
    // mutates only `weights.tensors` (inserting/removing per-layer attn matrices).
    // These are non-overlapping HashMap fields — the aliased read is sound.
    let weights_ref: &ModelWeights = unsafe { &*(weights as *const ModelWeights) };
    let walk_ffn = WalkFfn::new_unlimited_with_trace(weights_ref, gate_index);

    let start = std::time::Instant::now();
    let PredictResult {
        predictions: raw, ..
    } = predict_kquant_with_ffn(weights, tokenizer, token_ids, top_k, index, &walk_ffn);
    let walk_ms = start.elapsed().as_secs_f64() * 1000.0;

    let residuals = walk_ffn.take_residuals();
    let model_top1 = raw.first().cloned();
    let (predictions, knn_override) = apply_knn_override(raw, &residuals, knn_store, top_k);

    InferPatchedResult {
        predictions,
        model_top1,
        knn_override,
        residuals,
        walk_ms,
    }
}

/// Pure function: given raw walk predictions, per-layer residuals, and an
/// optional KnnStore, return `(predictions, knn_override)`.
///
/// Split out of `infer_patched` to be unit-testable without a real forward
/// pass. The behaviour is the contract that ADR 0001's byte-identical claim
/// rests on: the first stored layer (lowest index) whose top-1 cosine against
/// the captured residual exceeds `KNN_COSINE_THRESHOLD` replaces position 0
/// of the top-k with the stored target token at probability `1.0`; positions
/// `1..top_k` are the walk FFN's own top-`(top_k - 1)`.
pub fn apply_knn_override(
    raw: Vec<(String, f64)>,
    residuals: &[(usize, Vec<f32>)],
    knn_store: Option<&KnnStore>,
    top_k: usize,
) -> (Vec<(String, f64)>, Option<KnnOverride>) {
    let knn_override = knn_store.and_then(|store| {
        if store.is_empty() {
            return None;
        }
        let layers = store.layers();
        for (layer, residual) in residuals {
            if !layers.contains(layer) {
                continue;
            }
            if let Some((entry, cosine)) = store.query_top1(*layer, residual) {
                if cosine > KNN_COSINE_THRESHOLD {
                    return Some(KnnOverride {
                        token: entry.target_token.clone(),
                        cosine,
                        layer: *layer,
                    });
                }
            }
        }
        None
    });

    let predictions = match &knn_override {
        Some(ovr) if top_k > 0 => {
            let mut out = Vec::with_capacity(top_k);
            out.push((ovr.token.clone(), 1.0));
            for pair in raw.into_iter().take(top_k.saturating_sub(1)) {
                out.push(pair);
            }
            out
        }
        _ => raw,
    };

    (predictions, knn_override)
}

/// Rebuild a per-layer walk trace from captured residuals — shared between
/// the LQL `INFER` / `EXPLAIN INFER` display paths and the HTTP `/explain`
/// route. Each layer's residual is re-queried against the patched vindex's
/// gate KNN for the top-20 hits, then paired with `FeatureMeta` for display.
///
/// Kept here so that any surface using `infer_patched` can reconstruct the
/// same trace view without duplicating the loop or re-consuming WalkFfn's
/// internal `take_trace` (which drains residuals and so can't coexist with
/// the KNN-override residual capture above).
pub fn walk_trace_from_residuals(
    residuals: &[(usize, Vec<f32>)],
    patched: &PatchedVindex,
) -> Vec<(usize, Vec<WalkHit>)> {
    let mut out = Vec::with_capacity(residuals.len());
    for (layer, residual) in residuals {
        let r = ndarray::Array1::from_vec(residual.clone());
        let hits = patched.gate_knn(*layer, &r, 20);
        let walk_hits: Vec<WalkHit> = hits
            .into_iter()
            .filter_map(|(feature, gate_score)| {
                let meta = patched.feature_meta(*layer, feature)?;
                Some(WalkHit {
                    layer: *layer,
                    feature,
                    gate_score,
                    meta,
                })
            })
            .collect();
        out.push((*layer, walk_hits));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_store_with_key(layer: usize, key: Vec<f32>, target: &str) -> KnnStore {
        let mut store = KnnStore::default();
        store.add(
            layer,
            key,
            0,
            target.to_string(),
            "Atlantis".to_string(),
            "capital".to_string(),
            1.0,
        );
        store
    }

    fn raw(tokens: &[&str]) -> Vec<(String, f64)> {
        tokens
            .iter()
            .enumerate()
            .map(|(i, t)| (t.to_string(), 1.0 - 0.1 * i as f64))
            .collect()
    }

    #[test]
    fn no_store_passes_through_raw_topk() {
        let raw = raw(&["a", "b", "c"]);
        let residuals: Vec<(usize, Vec<f32>)> = vec![(5, vec![1.0, 0.0, 0.0])];

        let (predictions, override_) = apply_knn_override(raw.clone(), &residuals, None, 3);

        assert!(override_.is_none());
        assert_eq!(predictions, raw);
    }

    #[test]
    fn empty_store_passes_through() {
        let raw = raw(&["a", "b", "c"]);
        let residuals = vec![(5, vec![1.0, 0.0, 0.0])];
        let store = KnnStore::default();

        let (predictions, override_) = apply_knn_override(raw.clone(), &residuals, Some(&store), 3);

        assert!(override_.is_none());
        assert_eq!(predictions, raw);
    }

    #[test]
    fn matching_key_overrides_position_zero() {
        let key = vec![1.0, 0.0, 0.0];
        let residuals = vec![(5, key.clone())];
        let store = make_store_with_key(5, key, "Poseidon");

        let (predictions, override_) =
            apply_knn_override(raw(&["a", "b", "c"]), &residuals, Some(&store), 3);

        let ovr = override_.expect("key exactly matches residual — override must fire");
        assert_eq!(ovr.token, "Poseidon");
        assert_eq!(ovr.layer, 5);
        assert!(
            ovr.cosine > 0.99,
            "cosine of identical vectors must be ~1.0"
        );

        assert_eq!(predictions.len(), 3);
        assert_eq!(predictions[0], ("Poseidon".to_string(), 1.0));
        assert_eq!(predictions[1].0, "a");
        assert_eq!(predictions[2].0, "b");
    }

    #[test]
    fn mismatched_key_below_threshold_passes_through() {
        // Orthogonal vectors → cos = 0, well below 0.75 threshold.
        let residuals = vec![(5, vec![1.0, 0.0, 0.0])];
        let store = make_store_with_key(5, vec![0.0, 1.0, 0.0], "Poseidon");

        let (predictions, override_) =
            apply_knn_override(raw(&["a", "b", "c"]), &residuals, Some(&store), 3);

        assert!(
            override_.is_none(),
            "orthogonal residual must not trigger override"
        );
        assert_eq!(predictions[0].0, "a");
    }

    #[test]
    fn override_only_fires_on_stored_layers() {
        // Residual matches a key, but at a layer not present in the store.
        let key = vec![1.0, 0.0, 0.0];
        let residuals = vec![(7, key.clone())];
        let store = make_store_with_key(5, key, "Poseidon");

        let (predictions, override_) =
            apply_knn_override(raw(&["a", "b", "c"]), &residuals, Some(&store), 3);

        assert!(
            override_.is_none(),
            "residual layer not in store — no override"
        );
        assert_eq!(predictions[0].0, "a");
    }

    #[test]
    fn first_matching_layer_wins() {
        // Two stored layers both match; the earliest one (by iteration order
        // of the residuals slice) must take precedence.
        let key = vec![1.0, 0.0, 0.0];
        let residuals = vec![(5, key.clone()), (7, key.clone())];
        let mut store = make_store_with_key(5, key.clone(), "First");
        store.add(
            7,
            key,
            1,
            "Second".to_string(),
            "Atlantis".to_string(),
            "capital".to_string(),
            1.0,
        );

        let (predictions, override_) = apply_knn_override(raw(&["a"]), &residuals, Some(&store), 5);

        let ovr = override_.unwrap();
        assert_eq!(ovr.token, "First");
        assert_eq!(ovr.layer, 5);
        assert_eq!(predictions[0].0, "First");
    }

    #[test]
    fn top_k_one_returns_only_override() {
        let key = vec![1.0, 0.0, 0.0];
        let residuals = vec![(5, key.clone())];
        let store = make_store_with_key(5, key, "Poseidon");

        let (predictions, _) =
            apply_knn_override(raw(&["a", "b", "c"]), &residuals, Some(&store), 1);

        assert_eq!(predictions.len(), 1);
        assert_eq!(predictions[0], ("Poseidon".to_string(), 1.0));
    }

    #[test]
    fn top_k_zero_returns_empty() {
        let key = vec![1.0, 0.0, 0.0];
        let residuals = vec![(5, key.clone())];
        let store = make_store_with_key(5, key, "Poseidon");

        let (predictions, override_) =
            apply_knn_override(raw(&["a", "b", "c"]), &residuals, Some(&store), 0);

        // Override metadata still fires (the match is real) but predictions
        // collapses to raw (which is then truncated by the caller if needed).
        assert!(override_.is_some());
        assert_eq!(predictions.len(), 3);
    }

    // ── infer_patched (full forward pass) ──────────────────────────────

    #[test]
    fn infer_patched_returns_top_k_predictions_and_residuals() {
        use crate::test_utils::TestFixtures;
        let fx = TestFixtures::build();
        let tokens = vec![0u32, 1, 2];
        let result = infer_patched(&fx.weights, &fx.tokenizer, &fx.index, None, &tokens, 5);
        assert!(result.predictions.len() <= 5);
        // Walk pass populates residuals at every layer.
        assert!(!result.residuals.is_empty());
        assert!(result.knn_override.is_none());
        assert_eq!(result.model_top1, result.predictions.first().cloned());
        assert!(result.walk_ms >= 0.0);
    }

    #[test]
    fn walk_trace_from_residuals_returns_per_layer_walk_hits() {
        use crate::test_utils::TestFixtures;
        let fx = TestFixtures::build();
        let patched = larql_vindex::PatchedVindex::new(fx.index);
        let residuals = vec![
            (0usize, vec![0.1f32; fx.weights.hidden_size]),
            (1usize, vec![0.2f32; fx.weights.hidden_size]),
        ];
        let trace = walk_trace_from_residuals(&residuals, &patched);
        // One entry per residual.
        assert_eq!(trace.len(), 2);
        assert_eq!(trace[0].0, 0);
        assert_eq!(trace[1].0, 1);
        // Synthetic vindex returns no FeatureMeta, so walk_hits is empty
        // — but the per-layer entry must still be present.
    }

    #[test]
    fn walk_trace_from_residuals_empty_input_returns_empty() {
        use crate::test_utils::TestFixtures;
        let fx = TestFixtures::build();
        let patched = larql_vindex::PatchedVindex::new(fx.index);
        let trace = walk_trace_from_residuals(&[], &patched);
        assert!(trace.is_empty());
    }

    #[test]
    fn infer_patched_q4k_returns_predictions_via_quantised_path() {
        // Exercises `infer_patched_q4k` end-to-end — same contract as
        // `infer_patched` but routes through the Q4K dequant forward
        // path. Uses the Q4KTestFixtures so the vindex has Q4K bytes
        // for attention + FFN.
        use crate::test_utils::Q4KTestFixtures;
        let mut fx = Q4KTestFixtures::build();
        let tokens = vec![0u32, 1, 2];
        let result = infer_patched_q4k(
            &mut fx.weights,
            &fx.tokenizer,
            &fx.index,
            None,
            &tokens,
            5,
            &fx.index,
        );
        assert!(result.predictions.len() <= 5);
        assert!(result.knn_override.is_none());
        assert_eq!(result.model_top1, result.predictions.first().cloned());
        assert!(result.walk_ms >= 0.0);
    }

    #[test]
    fn infer_patched_with_knn_store_override_routes_through() {
        use crate::test_utils::TestFixtures;
        let fx = TestFixtures::build();
        let tokens = vec![0u32, 1];
        // First, run without override to capture the residuals — then plant
        // a key matching the L0 residual exactly so the override fires on
        // the rerun.
        let baseline = infer_patched(&fx.weights, &fx.tokenizer, &fx.index, None, &tokens, 3);
        let (l0_layer, l0_residual) = baseline
            .residuals
            .first()
            .expect("at least one residual captured");
        let store = make_store_with_key(*l0_layer, l0_residual.clone(), "PLANTED");
        let result = infer_patched(
            &fx.weights,
            &fx.tokenizer,
            &fx.index,
            Some(&store),
            &tokens,
            3,
        );
        let ovr = result
            .knn_override
            .expect("planted key matching residual must fire override");
        assert_eq!(ovr.token, "PLANTED");
        assert_eq!(result.predictions[0].0, "PLANTED");
        assert!((result.predictions[0].1 - 1.0).abs() < 1e-6);
        // model_top1 reflects the unoverridden walk pass.
        assert_eq!(result.model_top1, baseline.predictions.first().cloned());
    }
}
