//! `BoundaryPerLayerEngine` — `KvEngine` implementation with per-layer codec
//! policy.
//!
//! The engine refuses to construct without a matching calibration record
//! (per spec §4.7 + §4.9). v0.1 supports `Bf16` per layer only; other codec
//! choices are rejected at policy construction (per
//! [`super::policy::PolicyError`]).

use larql_compute::ComputeBackend;
use larql_inference::attention::{
    run_attention_block_decode_step_backend, run_attention_with_kv_backend, SharedKV,
};
use larql_inference::ffn::{BackendFfn, FfnBackend};
use larql_inference::forward::{embed_tokens_pub, run_ffn};
use larql_inference::model::ModelWeights;
use larql_inference::{cpu_engine_backend, EngineBackend};
use ndarray::{s, Array2};

use crate::engines::boundary_per_layer::calibration::{
    BoundaryCalibrationRecord, BoundaryCalibrationStore, CalibrationError,
};
use crate::engines::boundary_per_layer::policy::BoundaryLayerPolicy;
use crate::engines::boundary_per_layer::store::{PerLayerEncodedColdLayer, RsStorePerLayer};
use crate::engines::markov_residual::recompute_kv;
use crate::engines::markov_residual_codec::codec::ColdResidualCodec;
use crate::{EngineInfo, KvEngine};

/// Errors during engine construction (preconditions per spec §4.6).
#[derive(Debug, thiserror::Error)]
pub enum EngineConstructionError {
    #[error("policy targets {policy_layers} layers but model has {model_layers}")]
    LayerCountMismatch {
        policy_layers: usize,
        model_layers: usize,
    },
    #[error(transparent)]
    Calibration(#[from] CalibrationError),
}

/// `BoundaryPerLayerEngine` — per-layer codec policy on the cold tier.
pub struct BoundaryPerLayerEngine {
    window_size: Option<usize>,
    policy: BoundaryLayerPolicy,
    record: BoundaryCalibrationRecord,
    store: Option<RsStorePerLayer>,
    backend: Box<dyn EngineBackend>,
}

impl BoundaryPerLayerEngine {
    /// Construct with policy validation against the supplied calibration
    /// store. Returns `Err` when:
    ///
    /// - The policy's layer count does not match `num_model_layers` (§4.6).
    /// - No calibration record exists for the policy's fingerprint (§4.7,
    ///   §8.3).
    pub fn new(
        window_size: Option<usize>,
        policy: BoundaryLayerPolicy,
        num_model_layers: usize,
        calibration: &dyn BoundaryCalibrationStore,
    ) -> Result<Self, EngineConstructionError> {
        Self::with_backend(
            window_size,
            policy,
            num_model_layers,
            calibration,
            cpu_engine_backend(),
        )
    }

    pub fn with_backend(
        window_size: Option<usize>,
        policy: BoundaryLayerPolicy,
        num_model_layers: usize,
        calibration: &dyn BoundaryCalibrationStore,
        backend: Box<dyn EngineBackend>,
    ) -> Result<Self, EngineConstructionError> {
        if policy.num_layers() != num_model_layers {
            return Err(EngineConstructionError::LayerCountMismatch {
                policy_layers: policy.num_layers(),
                model_layers: num_model_layers,
            });
        }
        let record = calibration.get(&policy.fingerprint())?;
        Ok(Self {
            window_size,
            policy,
            record,
            store: None,
            backend,
        })
    }

    pub fn policy(&self) -> &BoundaryLayerPolicy {
        &self.policy
    }

    pub fn calibration_record(&self) -> &BoundaryCalibrationRecord {
        &self.record
    }

    fn run_prefill(&mut self, weights: &ModelWeights, token_ids: &[u32]) -> Option<Array2<f32>> {
        let backend = self.backend.as_ref();
        let num_layers = weights.num_layers;
        let seq_len = token_ids.len();
        let mut h = embed_tokens_pub(weights, token_ids);
        let mut stored: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
        let be = Some(backend as &dyn ComputeBackend);

        for layer in 0..num_layers {
            stored.push(h.clone());
            let (h_post_attn, _k, _v) =
                run_attention_with_kv_backend(weights, &h, layer, be).expect("attention failed");
            let bffn = BackendFfn {
                weights,
                backend: backend as &dyn ComputeBackend,
            };
            let (h_out, _) = run_ffn(weights, &h_post_attn, layer, &bffn, false);
            h = h_out;
        }

        let mut rs = RsStorePerLayer {
            stored,
            cold_encoded: None,
            cold_kv: None,
            cold_abs_start: 0,
            next_position: seq_len,
            max_window: self.window_size,
            policy_codecs: self.policy.entries.clone(),
        };

        let mut overflow_per_layer: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
        for layer in 0..num_layers {
            overflow_per_layer.push(rs.clip_layer_overflow(layer));
        }
        if overflow_per_layer.first().map_or(0, |c| c.shape()[0]) > 0 {
            let mut encoded_layers: Vec<PerLayerEncodedColdLayer> = Vec::with_capacity(num_layers);
            let mut cold_kv: Vec<SharedKV> = Vec::with_capacity(num_layers);
            for (layer, overflow) in overflow_per_layer.iter().enumerate() {
                let codec = self.policy.codec_for(layer);
                let decoded_overflow = roundtrip(overflow, codec);
                let (k, v) = recompute_kv(weights, &decoded_overflow, layer, 0, backend, None)
                    .expect("cold K/V pre-computation failed");
                cold_kv.push((k, v));
                let mut enc = PerLayerEncodedColdLayer::empty(codec, weights.hidden_size);
                enc.append(overflow);
                encoded_layers.push(enc);
            }
            rs.cold_encoded = Some(encoded_layers);
            rs.cold_kv = Some(cold_kv);
            rs.cold_abs_start = 0;
        }

        let last = last_row(&h);
        self.store = Some(rs);
        Some(last)
    }

    fn run_decode(&mut self, weights: &ModelWeights, token_id: u32) -> Option<Array2<f32>> {
        let backend = self.backend.as_ref();
        let rs = self.store.take()?;
        let num_layers = weights.num_layers;
        let abs_position = rs.next_position;
        let mut h_new = embed_tokens_pub(weights, &[token_id]);
        let mut new_stored: Vec<Array2<f32>> = Vec::with_capacity(num_layers);

        for layer in 0..num_layers {
            let h_hot = &rs.stored[layer];
            let s_hot = h_hot.shape()[0];
            let hot_abs_start = abs_position.saturating_sub(s_hot);

            let (k_full, v_full) = if let Some(cold_kv) = &rs.cold_kv {
                let (k_cold, v_cold) = &cold_kv[layer];
                let (k_hot, v_hot) =
                    recompute_kv(weights, h_hot, layer, hot_abs_start, backend, None)?;
                let c = k_cold.shape()[0];
                let kv_dim = k_cold.shape()[1];
                let mut k_combined = Array2::<f32>::zeros((c + s_hot, kv_dim));
                k_combined.slice_mut(s![..c, ..]).assign(k_cold);
                k_combined.slice_mut(s![c.., ..]).assign(&k_hot);
                let mut v_combined = Array2::<f32>::zeros((c + s_hot, kv_dim));
                v_combined.slice_mut(s![..c, ..]).assign(v_cold);
                v_combined.slice_mut(s![c.., ..]).assign(&v_hot);
                (k_combined, v_combined)
            } else {
                let (h_full, full_abs_start) = if let Some(cold_layers) = &rs.cold_encoded {
                    let enc = &cold_layers[layer];
                    if enc.n_positions > 0 {
                        let decoded = enc.decode();
                        let hidden = h_hot.shape()[1];
                        let mut combined =
                            Array2::<f32>::zeros((decoded.shape()[0] + s_hot, hidden));
                        combined
                            .slice_mut(s![..decoded.shape()[0], ..])
                            .assign(&decoded);
                        combined
                            .slice_mut(s![decoded.shape()[0].., ..])
                            .assign(h_hot);
                        (combined, rs.cold_abs_start)
                    } else {
                        (h_hot.clone(), hot_abs_start)
                    }
                } else {
                    (h_hot.clone(), hot_abs_start)
                };
                let (k, v) = recompute_kv(weights, &h_full, layer, full_abs_start, backend, None)?;
                (k, v)
            };

            new_stored.push(h_new.clone());

            let (h_post_attn, _new_kv) = run_attention_block_decode_step_backend(
                weights,
                &h_new,
                layer,
                Some(&(k_full, v_full)),
                abs_position,
                Some(backend as &dyn ComputeBackend),
            )?;

            let bffn = BackendFfn {
                weights,
                backend: backend as &dyn ComputeBackend,
            };
            let (h_out, _) = run_ffn(weights, &h_post_attn, layer, &bffn, false);
            h_new = h_out;
        }

        let mut updated_stored: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
        for (stored, new_row) in rs.stored.iter().zip(new_stored.iter()) {
            let s_old = stored.shape()[0];
            let hidden_dim = stored.shape()[1];
            let mut combined = Array2::<f32>::zeros((s_old + 1, hidden_dim));
            combined.slice_mut(s![..s_old, ..]).assign(stored);
            combined.slice_mut(s![s_old.., ..]).assign(new_row);
            updated_stored.push(combined);
        }

        let mut updated_rs = RsStorePerLayer {
            stored: updated_stored,
            cold_encoded: rs.cold_encoded,
            cold_kv: rs.cold_kv,
            cold_abs_start: rs.cold_abs_start,
            next_position: abs_position + 1,
            max_window: rs.max_window,
            policy_codecs: rs.policy_codecs,
        };

        let mut overflow_per_layer: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
        for layer in 0..num_layers {
            overflow_per_layer.push(updated_rs.clip_layer_overflow(layer));
        }
        if overflow_per_layer.first().map_or(0, |c| c.shape()[0]) > 0 {
            match updated_rs.cold_encoded.as_mut() {
                Some(layers) => {
                    for (layer, overflow) in overflow_per_layer.iter().enumerate() {
                        layers[layer].append(overflow);
                    }
                }
                None => {
                    let hidden = weights.hidden_size;
                    let mut layers: Vec<PerLayerEncodedColdLayer> = Vec::with_capacity(num_layers);
                    for (layer, overflow) in overflow_per_layer.iter().enumerate() {
                        let codec = self.policy.codec_for(layer);
                        let mut enc = PerLayerEncodedColdLayer::empty(codec, hidden);
                        enc.append(overflow);
                        layers.push(enc);
                    }
                    updated_rs.cold_encoded = Some(layers);
                }
            }
            updated_rs.cold_kv = None;
        }

        let last = last_row(&h_new);
        self.store = Some(updated_rs);
        Some(last)
    }
}

impl KvEngine for BoundaryPerLayerEngine {
    fn name(&self) -> &str {
        "boundary-per-layer"
    }

    fn info(&self) -> EngineInfo {
        let config = match self.window_size {
            Some(w) => format!("window={w},layers={}", self.policy.num_layers()),
            None => format!("window=full,layers={}", self.policy.num_layers()),
        };
        let mem = self.store.as_ref().map_or(0, |s| s.memory_bytes());
        EngineInfo {
            name: "boundary-per-layer".into(),
            description: format!(
                "per-layer codec policy on cold tier (kl_bound={:.3} nats, mem={:.1}MB)",
                self.record.kl_bound_nats,
                mem as f64 / 1_048_576.0,
            ),
            backend: self.backend.name().to_string(),
            config,
        }
    }

    fn prefill(
        &mut self,
        weights: &ModelWeights,
        _ffn: &dyn FfnBackend,
        token_ids: &[u32],
    ) -> Option<Array2<f32>> {
        self.run_prefill(weights, token_ids)
    }

    fn decode_step(
        &mut self,
        weights: &ModelWeights,
        _ffn: &dyn FfnBackend,
        token_id: u32,
    ) -> Option<Array2<f32>> {
        self.run_decode(weights, token_id)
    }

    fn memory_bytes(&self) -> usize {
        self.store.as_ref().map_or(0, |s| s.memory_bytes())
    }

    fn window_tokens(&self) -> usize {
        self.store.as_ref().map_or(0, |s| s.window_tokens())
    }

    fn cold_bytes(&self) -> usize {
        self.store.as_ref().map_or(0, |s| s.cold_bytes())
    }

    // ── Phase 2 migration: executor-driven path ──────────────────────────
    //
    // Per-layer codec policy requires per-layer dispatch. Override the
    // dense (non-quant) via_executor methods to drive the layer loop
    // through the executor + honor the caller's FFN backend.

    fn prefill_via_executor(
        &mut self,
        weights: &ModelWeights,
        executor: &dyn larql_inference::layer_executor::LayerExecutor,
        ffn: &dyn FfnBackend,
        token_ids: &[u32],
    ) -> Option<Array2<f32>> {
        use crate::engines::markov_residual::recompute_kv;
        use larql_inference::layer_executor::ExecutorDispatchKind;

        if matches!(executor.dispatch_kind(), ExecutorDispatchKind::Fused) {
            // State policy can't fire under fused dispatch; degrade.
            return self.prefill(weights, ffn, token_ids);
        }

        let backend = executor.backend();
        let num_layers = weights.num_layers;
        let seq_len = token_ids.len();
        let mut h = embed_tokens_pub(weights, token_ids);
        let mut stored: Vec<Array2<f32>> = Vec::with_capacity(num_layers);

        for layer in 0..num_layers {
            stored.push(h.clone());
            let (h_out, _kv) = executor.run_prefill_layer(weights, layer, &h, ffn)?;
            h = h_out;
        }

        let mut rs = RsStorePerLayer {
            stored,
            cold_encoded: None,
            cold_kv: None,
            cold_abs_start: 0,
            next_position: seq_len,
            max_window: self.window_size,
            policy_codecs: self.policy.entries.clone(),
        };

        let mut overflow_per_layer: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
        for layer in 0..num_layers {
            overflow_per_layer.push(rs.clip_layer_overflow(layer));
        }
        if overflow_per_layer.first().map_or(0, |c| c.shape()[0]) > 0 {
            let mut encoded_layers: Vec<PerLayerEncodedColdLayer> = Vec::with_capacity(num_layers);
            let mut cold_kv: Vec<SharedKV> = Vec::with_capacity(num_layers);
            for (layer, overflow) in overflow_per_layer.iter().enumerate() {
                let codec = self.policy.codec_for(layer);
                let decoded_overflow = roundtrip(overflow, codec);
                let (k, v) = recompute_kv(weights, &decoded_overflow, layer, 0, backend, None)
                    .expect("cold K/V pre-computation failed");
                cold_kv.push((k, v));
                let mut enc = PerLayerEncodedColdLayer::empty(codec, weights.hidden_size);
                enc.append(overflow);
                encoded_layers.push(enc);
            }
            rs.cold_encoded = Some(encoded_layers);
            rs.cold_kv = Some(cold_kv);
            rs.cold_abs_start = 0;
        }

        let out = last_row(&h);
        self.store = Some(rs);
        Some(out)
    }

    fn decode_step_via_executor(
        &mut self,
        weights: &ModelWeights,
        executor: &dyn larql_inference::layer_executor::LayerExecutor,
        ffn: &dyn FfnBackend,
        token_id: u32,
    ) -> Option<Array2<f32>> {
        use crate::engines::markov_residual::recompute_kv;
        use larql_inference::layer_executor::ExecutorDispatchKind;

        if matches!(executor.dispatch_kind(), ExecutorDispatchKind::Fused) {
            return self.decode_step(weights, ffn, token_id);
        }

        let backend = executor.backend();
        let rs = self.store.take()?;
        let num_layers = weights.num_layers;
        let abs_position = rs.next_position;
        let mut h_new = embed_tokens_pub(weights, &[token_id]);
        let mut new_stored: Vec<Array2<f32>> = Vec::with_capacity(num_layers);

        for layer in 0..num_layers {
            let h_hot = &rs.stored[layer];
            let s_hot = h_hot.shape()[0];
            let hot_abs_start = abs_position.saturating_sub(s_hot);

            let prior_kv: SharedKV = if let Some(cold_kv) = &rs.cold_kv {
                let (k_cold, v_cold) = &cold_kv[layer];
                let (k_hot, v_hot) =
                    recompute_kv(weights, h_hot, layer, hot_abs_start, backend, None)?;
                let c = k_cold.shape()[0];
                let kv_dim = k_cold.shape()[1];
                let mut k_combined = Array2::<f32>::zeros((c + s_hot, kv_dim));
                k_combined.slice_mut(s![..c, ..]).assign(k_cold);
                k_combined.slice_mut(s![c.., ..]).assign(&k_hot);
                let mut v_combined = Array2::<f32>::zeros((c + s_hot, kv_dim));
                v_combined.slice_mut(s![..c, ..]).assign(v_cold);
                v_combined.slice_mut(s![c.., ..]).assign(&v_hot);
                (k_combined, v_combined)
            } else {
                let (h_full, full_abs_start) = match &rs.cold_encoded {
                    Some(cold_layers) if cold_layers[layer].n_positions > 0 => {
                        let decoded = cold_layers[layer].decode();
                        let hidden = h_hot.shape()[1];
                        let mut combined =
                            Array2::<f32>::zeros((decoded.shape()[0] + s_hot, hidden));
                        combined
                            .slice_mut(s![..decoded.shape()[0], ..])
                            .assign(&decoded);
                        combined
                            .slice_mut(s![decoded.shape()[0].., ..])
                            .assign(h_hot);
                        (combined, rs.cold_abs_start)
                    }
                    _ => (h_hot.clone(), hot_abs_start),
                };
                recompute_kv(weights, &h_full, layer, full_abs_start, backend, None)?
            };

            new_stored.push(h_new.clone());
            let (h_out, _new_kv) =
                executor.run_decode_layer(weights, layer, &h_new, &prior_kv, abs_position, ffn)?;
            h_new = h_out;
        }

        let mut updated_stored: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
        for (stored, new_row) in rs.stored.iter().zip(new_stored.iter()) {
            let s_old = stored.shape()[0];
            let hidden_dim = stored.shape()[1];
            let mut combined = Array2::<f32>::zeros((s_old + 1, hidden_dim));
            combined.slice_mut(s![..s_old, ..]).assign(stored);
            combined.slice_mut(s![s_old.., ..]).assign(new_row);
            updated_stored.push(combined);
        }

        let mut updated_rs = RsStorePerLayer {
            stored: updated_stored,
            cold_encoded: rs.cold_encoded,
            cold_kv: rs.cold_kv,
            cold_abs_start: rs.cold_abs_start,
            next_position: abs_position + 1,
            max_window: rs.max_window,
            policy_codecs: rs.policy_codecs,
        };

        let mut overflow_per_layer: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
        for layer in 0..num_layers {
            overflow_per_layer.push(updated_rs.clip_layer_overflow(layer));
        }
        if overflow_per_layer.first().map_or(0, |c| c.shape()[0]) > 0 {
            match updated_rs.cold_encoded.as_mut() {
                Some(layers) => {
                    for (layer, overflow) in overflow_per_layer.iter().enumerate() {
                        layers[layer].append(overflow);
                    }
                }
                None => {
                    let hidden = weights.hidden_size;
                    let mut layers: Vec<PerLayerEncodedColdLayer> = Vec::with_capacity(num_layers);
                    for (layer, overflow) in overflow_per_layer.iter().enumerate() {
                        let codec = self.policy.codec_for(layer);
                        let mut enc = PerLayerEncodedColdLayer::empty(codec, hidden);
                        enc.append(overflow);
                        layers.push(enc);
                    }
                    updated_rs.cold_encoded = Some(layers);
                }
            }
            updated_rs.cold_kv = None;
        }

        let out = last_row(&h_new);
        self.store = Some(updated_rs);
        Some(out)
    }
}

fn roundtrip(block: &Array2<f32>, codec: ColdResidualCodec) -> Array2<f32> {
    if block.shape()[0] == 0 {
        return block.clone();
    }
    let mut tmp = PerLayerEncodedColdLayer::empty(codec, block.shape()[1]);
    tmp.append(block);
    tmp.decode()
}

fn last_row(h: &Array2<f32>) -> Array2<f32> {
    let last = h.shape()[0] - 1;
    h.slice(s![last..=last, ..]).to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engines::boundary_per_layer::calibration::InMemoryCalibrationStore;
    use larql_inference::ffn::WeightFfn;
    use larql_inference::test_utils::make_test_weights;

    fn store_with_record(policy: &BoundaryLayerPolicy) -> InMemoryCalibrationStore {
        let store = InMemoryCalibrationStore::new();
        store
            .put(BoundaryCalibrationRecord::bf16_uniform_default(
                policy.fingerprint(),
            ))
            .unwrap();
        store
    }

    // ── Construction ──────────────────────────────────────────────────────────

    #[test]
    fn construct_with_matching_calibration_succeeds() {
        let weights = make_test_weights();
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);
        let store = store_with_record(&policy);
        let eng = BoundaryPerLayerEngine::new(None, policy, weights.num_layers, &store);
        assert!(eng.is_ok());
    }

    #[test]
    fn construct_without_calibration_fails() {
        let weights = make_test_weights();
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);
        let store = InMemoryCalibrationStore::new(); // empty
        match BoundaryPerLayerEngine::new(None, policy, weights.num_layers, &store) {
            Err(EngineConstructionError::Calibration(CalibrationError::NoRecord(_))) => {}
            other => panic!("expected NoRecord error, got {:?}", other.err()),
        }
    }

    #[test]
    fn construct_with_layer_count_mismatch_fails() {
        let policy = BoundaryLayerPolicy::bf16_uniform("test", 2);
        let store = store_with_record(&policy);
        match BoundaryPerLayerEngine::new(None, policy, 10, &store) {
            Err(EngineConstructionError::LayerCountMismatch {
                policy_layers: 2,
                model_layers: 10,
            }) => {}
            other => panic!(
                "expected LayerCountMismatch{{policy=2,model=10}}, got {:?}",
                other.err()
            ),
        }
    }

    #[test]
    fn construction_error_display_includes_counts() {
        let e = EngineConstructionError::LayerCountMismatch {
            policy_layers: 3,
            model_layers: 7,
        };
        let s = e.to_string();
        assert!(s.contains('3'));
        assert!(s.contains('7'));
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    #[test]
    fn engine_name_is_boundary_per_layer() {
        let weights = make_test_weights();
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);
        let store = store_with_record(&policy);
        let eng = BoundaryPerLayerEngine::new(None, policy, weights.num_layers, &store).unwrap();
        assert_eq!(eng.name(), "boundary-per-layer");
    }

    #[test]
    fn engine_info_reports_window_and_layers() {
        let weights = make_test_weights();
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);
        let store = store_with_record(&policy);
        let eng =
            BoundaryPerLayerEngine::new(Some(128), policy, weights.num_layers, &store).unwrap();
        let info = eng.info();
        assert!(info.config.contains("window=128"));
        assert!(info
            .config
            .contains(&format!("layers={}", weights.num_layers)));
        assert!(info.description.contains("per-layer codec policy"));
    }

    #[test]
    fn engine_info_reports_unbounded_window() {
        let weights = make_test_weights();
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);
        let store = store_with_record(&policy);
        let eng = BoundaryPerLayerEngine::new(None, policy, weights.num_layers, &store).unwrap();
        let info = eng.info();
        assert!(info.config.contains("window=full"));
    }

    #[test]
    fn policy_accessor_returns_policy() {
        let weights = make_test_weights();
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);
        let store = store_with_record(&policy);
        let eng =
            BoundaryPerLayerEngine::new(None, policy.clone(), weights.num_layers, &store).unwrap();
        assert_eq!(eng.policy().num_layers(), policy.num_layers());
    }

    #[test]
    fn calibration_record_accessor_returns_record() {
        let weights = make_test_weights();
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);
        let store = store_with_record(&policy);
        let eng = BoundaryPerLayerEngine::new(None, policy, weights.num_layers, &store).unwrap();
        assert!(eng.calibration_record().kl_bound_nats < 0.1);
    }

    // ── Prefill / decode ──────────────────────────────────────────────────────

    #[test]
    fn engine_memory_zero_before_prefill() {
        let weights = make_test_weights();
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);
        let store = store_with_record(&policy);
        let eng = BoundaryPerLayerEngine::new(None, policy, weights.num_layers, &store).unwrap();
        assert_eq!(eng.memory_bytes(), 0);
        assert_eq!(eng.window_tokens(), 0);
        assert_eq!(eng.cold_bytes(), 0);
    }

    #[test]
    fn prefill_returns_hidden_and_populates_store() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);
        let store = store_with_record(&policy);
        let mut eng =
            BoundaryPerLayerEngine::new(None, policy, weights.num_layers, &store).unwrap();
        let h = eng.prefill(&weights, &ffn, &[0u32, 1, 2]).expect("prefill");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(eng.memory_bytes() > 0);
    }

    #[test]
    fn decode_step_produces_finite_hidden() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);
        let store = store_with_record(&policy);
        let mut eng =
            BoundaryPerLayerEngine::new(None, policy, weights.num_layers, &store).unwrap();
        eng.prefill(&weights, &ffn, &[0u32, 1]).expect("prefill");
        let h = eng.decode_step(&weights, &ffn, 2).expect("decode");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(h.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn decode_step_without_prefill_returns_none() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);
        let store = store_with_record(&policy);
        let mut eng =
            BoundaryPerLayerEngine::new(None, policy, weights.num_layers, &store).unwrap();
        assert!(eng.decode_step(&weights, &ffn, 0).is_none());
    }

    #[test]
    fn windowed_prefill_creates_cold_tier() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);
        let store = store_with_record(&policy);
        let mut eng = BoundaryPerLayerEngine::with_backend(
            Some(2),
            policy,
            weights.num_layers,
            &store,
            cpu_engine_backend(),
        )
        .unwrap();
        eng.prefill(&weights, &ffn, &[0u32, 1, 2, 3])
            .expect("prefill 4 tokens");
        assert!(eng.window_tokens() <= 2);
        assert!(eng.cold_bytes() > 0);
    }

    #[test]
    fn cold_encoded_path_exercised_after_eviction() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);
        let store = store_with_record(&policy);
        let mut eng =
            BoundaryPerLayerEngine::new(Some(2), policy, weights.num_layers, &store).unwrap();
        eng.prefill(&weights, &ffn, &[0u32, 1, 2, 3])
            .expect("prefill");
        eng.decode_step(&weights, &ffn, 4).expect("first decode"); // clears cold_kv
        let h = eng
            .decode_step(&weights, &ffn, 5)
            .expect("second decode hits cold_encoded path");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(h.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn memory_grows_with_each_decode_step() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);
        let store = store_with_record(&policy);
        let mut eng =
            BoundaryPerLayerEngine::new(None, policy, weights.num_layers, &store).unwrap();
        eng.prefill(&weights, &ffn, &[0u32]).expect("prefill");
        let m0 = eng.memory_bytes();
        eng.decode_step(&weights, &ffn, 1).expect("decode 1");
        let m1 = eng.memory_bytes();
        eng.decode_step(&weights, &ffn, 2).expect("decode 2");
        let m2 = eng.memory_bytes();
        assert!(m1 > m0);
        assert!(m2 > m1);
    }

    #[test]
    fn roundtrip_empty_block_short_circuits() {
        let empty: Array2<f32> = Array2::zeros((0, 8));
        let out = roundtrip(&empty, ColdResidualCodec::Bf16);
        assert_eq!(out.shape(), &[0, 8]);
    }

    #[test]
    fn last_row_extracts_correct_row() {
        let mut h = Array2::<f32>::zeros((3, 4));
        for j in 0..4 {
            h[[2, j]] = (j + 1) as f32;
        }
        let r = last_row(&h);
        assert_eq!(r.shape(), &[1, 4]);
        for j in 0..4 {
            assert_eq!(r[[0, j]], (j + 1) as f32);
        }
    }

    // ── Phase 2 migration: executor-driven path ──────────────────────────

    struct CountingFfn {
        calls: std::sync::atomic::AtomicUsize,
        hidden: usize,
    }
    impl larql_inference::ffn::FfnBackend for CountingFfn {
        fn forward(&self, _layer: usize, x: &ndarray::Array2<f32>) -> ndarray::Array2<f32> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            ndarray::Array2::zeros((x.shape()[0], self.hidden))
        }
        fn forward_with_activation(
            &self,
            layer: usize,
            x: &ndarray::Array2<f32>,
        ) -> (ndarray::Array2<f32>, ndarray::Array2<f32>) {
            let out = self.forward(layer, x);
            (out.clone(), out)
        }
        fn name(&self) -> &str {
            "counting"
        }
    }

    #[test]
    fn prefill_via_executor_runs_and_honors_ffn() {
        use larql_inference::layer_executor::LocalWalkExecutor;
        let weights = make_test_weights();
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);
        let store = store_with_record(&policy);
        let mut engine =
            BoundaryPerLayerEngine::new(None, policy, weights.num_layers, &store).unwrap();
        let backend = larql_compute::cpu_backend();
        let executor = LocalWalkExecutor::new(&*backend);
        let ffn = CountingFfn {
            calls: std::sync::atomic::AtomicUsize::new(0),
            hidden: weights.hidden_size,
        };
        let h = engine
            .prefill_via_executor(&weights, &executor, &ffn, &[0u32, 1, 2])
            .expect("prefill via executor");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert_eq!(
            ffn.calls.load(std::sync::atomic::Ordering::SeqCst),
            weights.num_layers,
            "boundary_per_layer engine should dispatch FFN through the supplied backend"
        );
    }

    #[test]
    fn decode_step_via_executor_extends_store() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::layer_executor::LocalWalkExecutor;
        let weights = make_test_weights();
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);
        let store = store_with_record(&policy);
        let mut engine =
            BoundaryPerLayerEngine::new(None, policy, weights.num_layers, &store).unwrap();
        let backend = larql_compute::cpu_backend();
        let executor = LocalWalkExecutor::new(&*backend);
        let ffn = NullFfn;
        engine
            .prefill_via_executor(&weights, &executor, &ffn, &[0u32, 1])
            .expect("prefill");
        let mem_before = engine.memory_bytes();
        let h = engine
            .decode_step_via_executor(&weights, &executor, &ffn, 2)
            .expect("decode");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(engine.memory_bytes() > mem_before);
    }

    #[test]
    fn executor_path_populates_per_layer_cold_tier() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::layer_executor::LocalWalkExecutor;
        let weights = make_test_weights();
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);
        let store = store_with_record(&policy);
        let mut engine =
            BoundaryPerLayerEngine::new(Some(2), policy, weights.num_layers, &store).unwrap();
        let backend = larql_compute::cpu_backend();
        let executor = LocalWalkExecutor::new(&*backend);
        let ffn = NullFfn;
        engine
            .prefill_via_executor(&weights, &executor, &ffn, &[0u32, 1, 2, 3])
            .expect("prefill with overflow");
        assert!(engine.window_tokens() <= 2);
        assert!(engine.cold_bytes() > 0);
    }

    /// Legacy `decode_step` with cold-tier (lines 172-184 / 186-205 /
    /// 254-272). Drives both the cold_kv combine branch on first decode
    /// and the cold_encoded recompute branch on the second decode (after
    /// overflow clears cold_kv).
    #[test]
    fn legacy_decode_step_traverses_cold_tier_branches() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);
        let store = store_with_record(&policy);
        let mut engine =
            BoundaryPerLayerEngine::new(Some(2), policy, weights.num_layers, &store).unwrap();
        engine
            .prefill(&weights, &ffn, &[0u32, 1, 2, 3])
            .expect("prefill overflow");
        // First decode: cold_kv populated by prefill → hits combine branch.
        let h = engine.decode_step(&weights, &ffn, 4).expect("decode 1");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        // Second decode: prior decode's overflow cleared cold_kv → hits
        // the cold_encoded recompute branch + the cold_encoded None→Some
        // append branch (lines 254-272).
        let h2 = engine.decode_step(&weights, &ffn, 5).expect("decode 2");
        assert_eq!(h2.shape(), &[1, weights.hidden_size]);
    }

    /// Executor-driven decode with cold tier — exercises the same
    /// cold_kv / cold_encoded branches in `decode_step_via_executor`
    /// (lines 431-456 / 494-).
    #[test]
    fn decode_via_executor_traverses_cold_tier_branches() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::layer_executor::LocalWalkExecutor;
        let weights = make_test_weights();
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);
        let store = store_with_record(&policy);
        let mut engine =
            BoundaryPerLayerEngine::new(Some(2), policy, weights.num_layers, &store).unwrap();
        let backend = larql_compute::cpu_backend();
        let executor = LocalWalkExecutor::new(&*backend);
        let ffn = NullFfn;
        engine
            .prefill_via_executor(&weights, &executor, &ffn, &[0u32, 1, 2, 3])
            .expect("prefill overflow");
        // First decode: cold_kv combine branch.
        engine
            .decode_step_via_executor(&weights, &executor, &ffn, 4)
            .expect("decode 1");
        // Second decode: cold_encoded recompute branch.
        let h = engine
            .decode_step_via_executor(&weights, &executor, &ffn, 5)
            .expect("decode 2");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
    }

    /// Fused-executor fallback: lines 350-352 / 414-415 dispatch back
    /// through the legacy `prefill` / `decode_step` path.
    struct FusedStubExecutor {
        backend: larql_compute::CpuBackend,
    }
    impl larql_inference::layer_executor::LayerExecutor for FusedStubExecutor {
        fn backend(&self) -> &dyn larql_compute::ComputeBackend {
            &self.backend
        }
        fn dispatch_kind(&self) -> larql_inference::layer_executor::ExecutorDispatchKind {
            larql_inference::layer_executor::ExecutorDispatchKind::Fused
        }
        fn name(&self) -> &str {
            "fused-stub"
        }
    }

    #[test]
    fn fused_executor_falls_back_to_legacy_path() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);
        let store = store_with_record(&policy);
        let mut engine =
            BoundaryPerLayerEngine::new(None, policy, weights.num_layers, &store).unwrap();
        let exec = FusedStubExecutor {
            backend: larql_compute::CpuBackend,
        };
        let h = engine
            .prefill_via_executor(&weights, &exec, &ffn, &[0u32, 1])
            .expect("fused fallback prefill");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        let h2 = engine
            .decode_step_via_executor(&weights, &exec, &ffn, 2)
            .expect("fused fallback decode");
        assert_eq!(h2.shape(), &[1, weights.hidden_size]);
    }
}
