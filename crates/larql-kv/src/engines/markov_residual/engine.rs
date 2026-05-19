//! MarkovResidualEngine — KvEngine implementation.

use larql_compute::ComputeBackend;
use larql_vindex::VectorIndex;
use ndarray::Array2;

use super::compute::{rs_decode_step, rs_decode_step_profiled, rs_prefill};
use super::store::RsStore;
use super::walk::{ensure_attn_tensors_dequantised, rs_decode_step_walk, rs_prefill_walk};
use crate::profiler::EngineProfiler;
use crate::{DecodeStageSummary, EngineInfo, KvEngine};
use larql_inference::ffn::FfnBackend;
use larql_inference::model::ModelWeights;
use larql_inference::{cpu_engine_backend, EngineBackend};

pub struct MarkovResidualEngine {
    window_size: Option<usize>,
    store: Option<RsStore>,
    backend: Box<dyn EngineBackend>,
    profiling: bool,
    profile: EngineProfiler,
}

impl MarkovResidualEngine {
    pub fn new(window_size: Option<usize>) -> Self {
        Self::with_backend(window_size, cpu_engine_backend())
    }

    pub fn with_backend(window_size: Option<usize>, backend: Box<dyn EngineBackend>) -> Self {
        Self {
            window_size,
            store: None,
            backend,
            profiling: false,
            profile: EngineProfiler::default(),
        }
    }

    pub fn with_profiling(mut self, enabled: bool) -> Self {
        self.profiling = enabled;
        self
    }

    pub fn total_memory_bytes(&self) -> usize {
        self.store.as_ref().map_or(0, |s| s.memory_bytes())
    }
}

impl KvEngine for MarkovResidualEngine {
    fn name(&self) -> &str {
        "markov-rs"
    }

    fn info(&self) -> EngineInfo {
        let config = match self.window_size {
            Some(w) => format!("window={w}"),
            None => "window=full".into(),
        };
        let mem = self.store.as_ref().map_or(0, |s| s.memory_bytes());
        EngineInfo {
            name: "markov-rs".into(),
            description: format!(
                "residual-stream KV replacement — K/V recomputed from stored residuals (mem={:.1}MB)",
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
        let result = rs_prefill(weights, token_ids, self.window_size, self.backend.as_ref());
        let hidden = result.hidden.clone();
        self.store = Some(result.store);
        Some(hidden)
    }

    fn decode_step(
        &mut self,
        weights: &ModelWeights,
        _ffn: &dyn FfnBackend,
        token_id: u32,
    ) -> Option<Array2<f32>> {
        let rs = self.store.take()?;
        let (hidden, new_rs) = if self.profiling {
            rs_decode_step_profiled(
                weights,
                token_id,
                rs,
                self.backend.as_ref(),
                &mut self.profile,
            )?
        } else {
            rs_decode_step(weights, token_id, rs, self.backend.as_ref())?
        };
        self.store = Some(new_rs);
        Some(hidden)
    }

    fn memory_bytes(&self) -> usize {
        self.total_memory_bytes()
    }

    fn window_tokens(&self) -> usize {
        self.store.as_ref().map_or(0, |s| s.window_tokens())
    }

    fn cold_bytes(&self) -> usize {
        self.store.as_ref().map_or(0, |s| s.cold_bytes())
    }

    fn stage_summary(&self) -> Option<DecodeStageSummary> {
        if !self.profiling || self.profile.decode_total.count == 0 {
            return None;
        }
        Some(self.profile.summary("markov-rs", self.backend.name()))
    }

    fn prefill_quant(
        &mut self,
        weights: &mut ModelWeights,
        _ffn: &dyn FfnBackend,
        index: &VectorIndex,
        token_ids: &[u32],
        backend: &dyn ComputeBackend,
    ) -> Option<Array2<f32>> {
        // This engine's state policy IS the residual-stream walk path.
        // The backend-fused fast path is a different engine
        // (`StandardEngine` via `coarse_prefill`); engines that want the
        // fused speed must select it explicitly. No hidden bypass here.
        ensure_attn_tensors_dequantised(weights, index);
        let result = rs_prefill_walk(weights, index, token_ids, self.window_size, backend);
        let hidden = result.hidden.clone();
        self.store = Some(result.store);
        Some(hidden)
    }

    fn decode_step_quant(
        &mut self,
        weights: &mut ModelWeights,
        _ffn: &dyn FfnBackend,
        index: &VectorIndex,
        token_id: u32,
        backend: &dyn ComputeBackend,
    ) -> Option<Array2<f32>> {
        ensure_attn_tensors_dequantised(weights, index);
        let rs = self.store.take()?;
        let (hidden, new_rs) = rs_decode_step_walk(weights, index, token_id, rs, backend)?;
        self.store = Some(new_rs);
        Some(hidden)
    }

    // ── Executor-aware migration (Phase 2 of engine-state-vs-execution spec) ──
    //
    // The methods below override the trait defaults to drive the layer
    // loop through a caller-supplied `LayerExecutor` and honor the
    // caller-supplied `FfnBackend`. Old `prefill_quant` /
    // `decode_step_quant` stay above for backward compat; they construct
    // their own WalkFfn and ignore the FFN parameter. The new methods
    // are what remote-FFN deployments and per-layer codec engines must
    // call to get the engine's contract.

    fn prefill_quant_via_executor(
        &mut self,
        weights: &mut ModelWeights,
        executor: &dyn larql_inference::layer_executor::LayerExecutor,
        ffn: &dyn FfnBackend,
        index: &VectorIndex,
        token_ids: &[u32],
    ) -> Option<Array2<f32>> {
        use crate::engines::markov_residual::recompute_kv;
        use larql_inference::attention::SharedKV;
        use larql_inference::forward::embed_tokens_pub;
        use larql_inference::layer_executor::ExecutorDispatchKind;
        use ndarray::Array2;
        // Engines whose state policy requires per-layer dispatch (this
        // one) must refuse fused executors at construction. Until the
        // `requires_per_layer_dispatch()` trait hook lands (Phase 3),
        // degrade transparently to the legacy fused-or-walk path.
        if matches!(executor.dispatch_kind(), ExecutorDispatchKind::Fused) {
            return self.prefill_quant(weights, ffn, index, token_ids, executor.backend());
        }

        // Q4K attn weights need dequant once before the per-layer
        // executor can drive f32 attention against them.
        ensure_attn_tensors_dequantised(weights, index);

        let backend = executor.backend();
        let num_layers = weights.num_layers;
        let seq_len = token_ids.len();
        let mut h = embed_tokens_pub(weights, token_ids);
        let mut stored: Vec<Array2<f32>> = Vec::with_capacity(num_layers);

        for layer in 0..num_layers {
            stored.push(h.clone());
            // Executor drives attention + FFN; engine doesn't care which
            // backend or whether FFN is local/remote. Engine discards
            // the layer's K/V — residual-stream contract recomputes K/V
            // per decode step from the stored residuals.
            let (h_out, _kv) = executor.run_prefill_layer(weights, layer, &h, ffn)?;
            h = h_out;
        }

        // State management identical to `rs_prefill_walk`: build the
        // store, clip overflow into cold tier, precompute cold K/V via
        // `recompute_kv` (engine policy — the executor doesn't own this).
        let mut rs = RsStore {
            stored,
            cold_residuals: None,
            cold_kv: None,
            cold_abs_start: 0,
            next_position: seq_len,
            max_window: self.window_size,
        };
        let mut cold: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
        for layer in 0..num_layers {
            rs.clip_layer(layer, &mut cold);
        }
        if cold.first().map_or(0, |c| c.shape()[0]) > 0 {
            let cold_kv: Vec<SharedKV> = (0..num_layers)
                .map(|layer| {
                    recompute_kv(weights, &cold[layer], layer, 0, backend, Some(index))
                        .expect("cold K/V pre-computation failed")
                })
                .collect();
            rs.cold_residuals = Some(cold);
            rs.cold_kv = Some(cold_kv);
            rs.cold_abs_start = 0;
        }

        let hidden = {
            use ndarray::s;
            let last = h.shape()[0] - 1;
            h.slice(s![last..=last, ..]).to_owned()
        };
        self.store = Some(rs);
        Some(hidden)
    }

    fn decode_step_quant_via_executor(
        &mut self,
        weights: &mut ModelWeights,
        executor: &dyn larql_inference::layer_executor::LayerExecutor,
        ffn: &dyn FfnBackend,
        index: &VectorIndex,
        token_id: u32,
    ) -> Option<Array2<f32>> {
        use crate::engines::markov_residual::recompute_kv;
        use larql_inference::attention::SharedKV;
        use larql_inference::forward::embed_tokens_pub;
        use larql_inference::layer_executor::ExecutorDispatchKind;
        use ndarray::{s, Array2};

        if matches!(executor.dispatch_kind(), ExecutorDispatchKind::Fused) {
            return self.decode_step_quant(weights, ffn, index, token_id, executor.backend());
        }

        ensure_attn_tensors_dequantised(weights, index);

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

            // Engine assembles the K/V to attend against from its store.
            // The executor doesn't own state — it just receives prior_kv
            // and runs the layer.
            let prior_kv: SharedKV = if let Some(cold_kv) = &rs.cold_kv {
                let (k_cold, v_cold) = &cold_kv[layer];
                let (k_hot, v_hot) =
                    recompute_kv(weights, h_hot, layer, hot_abs_start, backend, Some(index))?;
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
                let (h_full, full_abs_start) = match &rs.cold_residuals {
                    Some(cold) if cold[layer].shape()[0] > 0 => {
                        let h_cold = &cold[layer];
                        let s_cold = h_cold.shape()[0];
                        let hidden = h_hot.shape()[1];
                        let mut combined = Array2::<f32>::zeros((s_cold + s_hot, hidden));
                        combined.slice_mut(s![..s_cold, ..]).assign(h_cold);
                        combined.slice_mut(s![s_cold.., ..]).assign(h_hot);
                        (combined, rs.cold_abs_start)
                    }
                    _ => (h_hot.clone(), hot_abs_start),
                };
                recompute_kv(
                    weights,
                    &h_full,
                    layer,
                    full_abs_start,
                    backend,
                    Some(index),
                )?
            };

            new_stored.push(h_new.clone());
            // Run the layer through the executor.
            let (h_out, _new_kv) =
                executor.run_decode_layer(weights, layer, &h_new, &prior_kv, abs_position, ffn)?;
            h_new = h_out;
        }

        // Append new row to store, clip overflow into cold.
        let mut updated_stored: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
        for (stored, new_row) in rs.stored.iter().zip(new_stored.iter()) {
            let s_old = stored.shape()[0];
            let hidden_dim = stored.shape()[1];
            let mut combined = Array2::<f32>::zeros((s_old + 1, hidden_dim));
            combined.slice_mut(s![..s_old, ..]).assign(stored);
            combined.slice_mut(s![s_old.., ..]).assign(new_row);
            updated_stored.push(combined);
        }

        let mut updated_rs = RsStore {
            stored: updated_stored,
            cold_residuals: rs.cold_residuals,
            cold_kv: rs.cold_kv,
            cold_abs_start: rs.cold_abs_start,
            next_position: abs_position + 1,
            max_window: rs.max_window,
        };

        let mut overflow: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
        for layer in 0..num_layers {
            updated_rs.clip_layer(layer, &mut overflow);
        }
        if overflow.first().map_or(0, |c| c.shape()[0]) > 0 {
            match updated_rs.cold_residuals.as_mut() {
                Some(cold) => {
                    for layer in 0..num_layers {
                        let hidden = cold[layer].shape()[1];
                        let c_old = cold[layer].shape()[0];
                        let c_new = overflow[layer].shape()[0];
                        let mut merged = Array2::<f32>::zeros((c_old + c_new, hidden));
                        merged.slice_mut(s![..c_old, ..]).assign(&cold[layer]);
                        merged.slice_mut(s![c_old.., ..]).assign(&overflow[layer]);
                        cold[layer] = merged;
                    }
                }
                None => {
                    updated_rs.cold_residuals = Some(overflow);
                }
            }
            updated_rs.cold_kv = None;
        }

        let last = h_new.shape()[0] - 1;
        let out = h_new.slice(s![last..=last, ..]).to_owned();
        self.store = Some(updated_rs);
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::KvEngine;
    use larql_inference::ffn::WeightFfn;
    use larql_inference::forward::hidden_to_raw_logits;
    use larql_inference::test_utils::make_test_weights;

    // ── Construction ──────────────────────────────────────────────────────────

    #[test]
    fn engine_name() {
        assert_eq!(MarkovResidualEngine::new(None).name(), "markov-rs");
    }

    #[test]
    fn engine_memory_zero_before_prefill() {
        let eng = MarkovResidualEngine::new(None);
        assert_eq!(eng.memory_bytes(), 0);
        assert_eq!(eng.window_tokens(), 0);
        assert_eq!(eng.cold_bytes(), 0);
    }

    #[test]
    fn engine_info_full_window() {
        let eng = MarkovResidualEngine::new(None);
        let info = eng.info();
        assert!(
            info.config.contains("full"),
            "expected 'full' in config, got '{}'",
            info.config
        );
    }

    #[test]
    fn engine_info_fixed_window() {
        let eng = MarkovResidualEngine::new(Some(16));
        let info = eng.info();
        assert!(
            info.config.contains("16"),
            "expected window size in config, got '{}'",
            info.config
        );
    }

    // ── Prefill → decode cycle ────────────────────────────────────────────────

    #[test]
    fn prefill_stores_residuals_for_all_layers() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut engine = MarkovResidualEngine::new(None);
        let h = engine
            .prefill(&weights, &ffn, &[0u32, 1, 2])
            .expect("prefill");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(
            engine.memory_bytes() > 0,
            "store should be non-empty after prefill"
        );
    }

    #[test]
    fn decode_step_produces_finite_logits() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut engine = MarkovResidualEngine::new(None);
        engine.prefill(&weights, &ffn, &[0u32, 1]).expect("prefill");
        let h = engine.decode_step(&weights, &ffn, 2).expect("decode");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(hidden_to_raw_logits(&weights, &h)
            .iter()
            .all(|v| v.is_finite()));
    }

    #[test]
    fn memory_grows_with_each_decode_step() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut engine = MarkovResidualEngine::new(None);
        engine.prefill(&weights, &ffn, &[0u32]).expect("prefill");
        let mem_after_prefill = engine.memory_bytes();
        engine.decode_step(&weights, &ffn, 1).expect("decode 1");
        let mem_after_1 = engine.memory_bytes();
        engine.decode_step(&weights, &ffn, 2).expect("decode 2");
        let mem_after_2 = engine.memory_bytes();
        assert!(
            mem_after_1 > mem_after_prefill,
            "memory should grow with decode steps"
        );
        assert!(mem_after_2 > mem_after_1);
    }

    #[test]
    fn window_clipping_limits_hot_store() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut engine = MarkovResidualEngine::new(Some(2)); // window=2 tokens
        engine
            .prefill(&weights, &ffn, &[0u32, 1, 2, 3, 4])
            .expect("prefill 5 tokens");
        // After clipping, hot store ≤ window
        assert!(
            engine.window_tokens() <= 2,
            "window_tokens={} should be ≤ 2",
            engine.window_tokens()
        );
        // Cold bytes should now be non-zero (overflow clipped to cold)
        assert!(
            engine.cold_bytes() > 0,
            "cold tier should have bytes after clipping"
        );
    }

    #[test]
    fn multiple_decode_steps_produce_consistent_shapes() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut engine = MarkovResidualEngine::new(None);
        engine.prefill(&weights, &ffn, &[0u32]).expect("prefill");
        for step in 0..3 {
            let h = engine
                .decode_step(&weights, &ffn, step as u32)
                .expect("decode");
            assert_eq!(h.shape(), &[1, weights.hidden_size], "step {step}");
        }
    }

    // ── Profiling ─────────────────────────────────────────────────────────────

    #[test]
    fn with_profiling_enables_profiling_branch() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut engine = MarkovResidualEngine::new(None).with_profiling(true);
        // No decode yet → stage_summary returns None even with profiling on.
        assert!(engine.stage_summary().is_none());

        engine.prefill(&weights, &ffn, &[0u32, 1]).expect("prefill");
        engine.decode_step(&weights, &ffn, 2).expect("decode");

        let summary = engine.stage_summary().expect("profiling summary");
        assert_eq!(summary.engine, "markov-rs");
        assert_eq!(summary.steps, 1);
        assert!(summary.avg_total_decode_us > 0.0);
    }

    #[test]
    fn stage_summary_none_without_profiling() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut engine = MarkovResidualEngine::new(None); // profiling: false
        engine.prefill(&weights, &ffn, &[0u32]).expect("prefill");
        engine.decode_step(&weights, &ffn, 1).expect("decode");
        assert!(
            engine.stage_summary().is_none(),
            "stage_summary must be None when profiling is disabled"
        );
    }

    #[test]
    fn profiling_decode_path_matches_unprofiled_shape() {
        // Two engines: one profiled, one not. Both should yield hidden states
        // of the same shape after the same prefill+decode sequence.
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut profiled = MarkovResidualEngine::new(None).with_profiling(true);
        let mut plain = MarkovResidualEngine::new(None);
        profiled.prefill(&weights, &ffn, &[0u32, 1]).unwrap();
        plain.prefill(&weights, &ffn, &[0u32, 1]).unwrap();
        let h_p = profiled.decode_step(&weights, &ffn, 2).unwrap();
        let h_n = plain.decode_step(&weights, &ffn, 2).unwrap();
        assert_eq!(h_p.shape(), h_n.shape());
    }

    // ── Q4K paths via CPU fallback ────────────────────────────────────────
    //
    // On a CPU backend, `fused_prefill` returns `None`, so the engine
    // falls through to `rs_prefill_walk` against the synthetic VectorIndex.
    // This exercises the prefill_quant / decode_step_quant branches that the
    // Metal-only happy path also takes (apart from the Metal early-return).

    #[test]
    fn prefill_q4k_cpu_fallback_runs_walk_path() {
        use larql_inference::ffn::NullFfn;
        let mut weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        // `NullFfn` satisfies the trait without borrowing `weights`, which is
        // `&mut` here. The engine ignores the FFN parameter on this path.
        let ffn = NullFfn;
        let mut engine = MarkovResidualEngine::new(None);
        let h = engine
            .prefill_quant(&mut weights, &ffn, &index, &[0u32, 1, 2], &*backend)
            .expect("prefill_quant cpu fallback");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(engine.memory_bytes() > 0);
    }

    #[test]
    fn decode_step_q4k_cpu_fallback_extends_store() {
        use larql_inference::ffn::NullFfn;
        let mut weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let ffn = NullFfn;
        let mut engine = MarkovResidualEngine::new(None);
        engine
            .prefill_quant(&mut weights, &ffn, &index, &[0u32, 1], &*backend)
            .expect("prefill_quant");
        let mem_before = engine.memory_bytes();
        let h = engine
            .decode_step_quant(&mut weights, &ffn, &index, 2, &*backend)
            .expect("decode_step_quant cpu fallback");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(
            engine.memory_bytes() > mem_before,
            "store should grow after decode_step_quant"
        );
    }

    // ── Walk-path overflow branches (markov_residual/walk.rs) ────────────
    //
    // The two tests above use `window=None` so the cold tier never
    // populates — leaving the walk.rs cold-K/V precompute (lines 66-76)
    // and cold-residual decode branch (lines 162-186) uncovered. The
    // tests below drive them with a small window + multiple decode steps.

    #[test]
    fn prefill_quant_walk_with_window_populates_cold_kv() {
        use larql_inference::ffn::NullFfn;
        let mut weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let ffn = NullFfn;
        let mut engine = MarkovResidualEngine::new(Some(2));
        engine
            .prefill_quant(&mut weights, &ffn, &index, &[0u32, 1, 2, 3], &*backend)
            .expect("prefill_quant with overflow");
        // window=2 + 4 prompt tokens → cold tier populated → walk.rs
        // lines 67-75 fire.
        assert!(engine.window_tokens() <= 2);
        assert!(engine.cold_bytes() > 0);
    }

    #[test]
    fn decode_step_quant_walk_first_overflow_creates_cold_residuals() {
        // walk.rs lines 305-307: `None => updated_rs.cold_residuals =
        // Some(overflow)`. Fires when prefill didn't overflow (cold = None)
        // but the first decode does (window cap exceeded mid-decode).
        use larql_inference::ffn::NullFfn;
        let mut weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let ffn = NullFfn;
        // window=2, prefill=1 token → no overflow on prefill (cold=None).
        let mut engine = MarkovResidualEngine::new(Some(2));
        engine
            .prefill_quant(&mut weights, &ffn, &index, &[0u32], &*backend)
            .expect("prefill_quant");
        // Decode until hot exceeds window → first-time cold population.
        engine
            .decode_step_quant(&mut weights, &ffn, &index, 1, &*backend)
            .expect("decode 1");
        engine
            .decode_step_quant(&mut weights, &ffn, &index, 2, &*backend)
            .expect("decode 2 — triggers first-overflow None branch");
        // After overflow, cold tier is populated.
        assert!(engine.cold_bytes() > 0);
    }

    #[test]
    fn decode_step_quant_walk_after_overflow_hits_cold_residuals_branch() {
        use larql_inference::ffn::NullFfn;
        let mut weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let ffn = NullFfn;
        let mut engine = MarkovResidualEngine::new(Some(2));
        engine
            .prefill_quant(&mut weights, &ffn, &index, &[0u32, 1, 2, 3], &*backend)
            .expect("prefill_quant");
        // First decode: exercises walk.rs cold_kv branch (lines 132-161).
        engine
            .decode_step_quant(&mut weights, &ffn, &index, 4, &*backend)
            .expect("first decode_step_quant");
        // Second decode: cold_kv was cleared by overflow at the first
        // decode (walk.rs line 309), so this hits the cold_residuals
        // recompute branch (lines 162-187).
        let h = engine
            .decode_step_quant(&mut weights, &ffn, &index, 5, &*backend)
            .expect("second decode_step_quant");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
    }

    // ── Phase 2: executor-driven path ─────────────────────────────────────

    #[test]
    fn prefill_quant_via_executor_runs_through_local_walk() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::layer_executor::LocalWalkExecutor;
        let mut weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let executor = LocalWalkExecutor::new(&*backend);
        let ffn = NullFfn;
        let mut engine = MarkovResidualEngine::new(None);
        let h = engine
            .prefill_quant_via_executor(&mut weights, &executor, &ffn, &index, &[0u32, 1, 2])
            .expect("executor prefill");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(engine.memory_bytes() > 0);
    }

    #[test]
    fn decode_step_quant_via_executor_extends_store() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::layer_executor::LocalWalkExecutor;
        let mut weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let executor = LocalWalkExecutor::new(&*backend);
        let ffn = NullFfn;
        let mut engine = MarkovResidualEngine::new(None);
        engine
            .prefill_quant_via_executor(&mut weights, &executor, &ffn, &index, &[0u32, 1])
            .expect("prefill");
        let mem_before = engine.memory_bytes();
        let h = engine
            .decode_step_quant_via_executor(&mut weights, &executor, &ffn, &index, 2)
            .expect("decode");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(engine.memory_bytes() > mem_before);
    }

    /// Counting FFN that records every `forward` call. Used to prove
    /// the executor path actually dispatches through the caller's
    /// `FfnBackend` instead of constructing a local `WalkFfn` (the
    /// legacy coupling that the migration removes).
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
    fn executor_path_honors_ffn_parameter() {
        // Pass a counting stub. If the engine constructs its own
        // WalkFfn internally (the legacy bug we're fixing) the counter
        // stays at zero.
        use larql_inference::layer_executor::LocalWalkExecutor;
        let mut weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let executor = LocalWalkExecutor::new(&*backend);

        let ffn = CountingFfn {
            calls: std::sync::atomic::AtomicUsize::new(0),
            hidden: weights.hidden_size,
        };
        let mut engine = MarkovResidualEngine::new(None);
        engine
            .prefill_quant_via_executor(&mut weights, &executor, &ffn, &index, &[0u32, 1, 2])
            .expect("prefill via executor");

        let call_count = ffn.calls.load(std::sync::atomic::Ordering::SeqCst);
        // Prefill runs FFN once per layer.
        assert_eq!(
            call_count, weights.num_layers,
            "executor path should dispatch FFN through the supplied backend \
             once per layer; got {call_count} for {} layers — engine is \
             likely constructing its own FFN internally",
            weights.num_layers
        );
    }

    #[test]
    fn prefill_quant_via_executor_with_window_populates_cold_tier() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::layer_executor::LocalWalkExecutor;
        let mut weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let executor = LocalWalkExecutor::new(&*backend);
        let ffn = NullFfn;
        let mut engine = MarkovResidualEngine::new(Some(2));
        engine
            .prefill_quant_via_executor(&mut weights, &executor, &ffn, &index, &[0u32, 1, 2, 3])
            .expect("prefill with overflow");
        assert!(engine.window_tokens() <= 2);
        assert!(engine.cold_bytes() > 0);
    }

    /// Drive `decode_step_quant_via_executor`'s `cold_kv` branch (lines
    /// 315-333): prefill with overflow so the engine pre-computes
    /// cold_kv during prefill, then run a single decode step that
    /// combines cold_kv + hot K/V for attention.
    #[test]
    fn decode_step_quant_via_executor_uses_cold_kv_branch() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::layer_executor::LocalWalkExecutor;
        let mut weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let executor = LocalWalkExecutor::new(&*backend);
        let ffn = NullFfn;
        let mut engine = MarkovResidualEngine::new(Some(2));
        engine
            .prefill_quant_via_executor(&mut weights, &executor, &ffn, &index, &[0u32, 1, 2, 3])
            .expect("prefill overflow → cold_kv populated");
        // First decode reads cold_kv branch (rs.cold_kv = Some(_)).
        let h = engine
            .decode_step_quant_via_executor(&mut weights, &executor, &ffn, &index, 4)
            .expect("decode via cold_kv branch");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
    }

    /// Drive the cold_residuals branch when cold_kv has been cleared
    /// (the second decode after overflow). At line ~399 the engine
    /// clears cold_kv when a new overflow happens, then subsequent
    /// decodes recompute K/V from cold_residuals.
    #[test]
    fn decode_step_quant_via_executor_hits_cold_residuals_branch() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::layer_executor::LocalWalkExecutor;
        let mut weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let executor = LocalWalkExecutor::new(&*backend);
        let ffn = NullFfn;
        let mut engine = MarkovResidualEngine::new(Some(2));
        engine
            .prefill_quant_via_executor(&mut weights, &executor, &ffn, &index, &[0u32, 1, 2, 3])
            .expect("prefill");
        // First decode clears cold_kv via overflow.
        engine
            .decode_step_quant_via_executor(&mut weights, &executor, &ffn, &index, 4)
            .expect("first decode");
        // Second decode: cold_kv is None, exercises the recompute_kv
        // from cold_residuals branch.
        let h = engine
            .decode_step_quant_via_executor(&mut weights, &executor, &ffn, &index, 5)
            .expect("decode via cold_residuals recompute");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
    }

    /// `Fused`-executor fallback in `*_via_executor` (lines 219-221, 294-296):
    /// when the executor advertises fused dispatch, the engine routes back
    /// through the legacy `prefill_quant` / `decode_step_quant` path.
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
    fn fused_executor_falls_back_to_legacy_quant_path() {
        use larql_inference::ffn::NullFfn;
        let mut weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let exec = FusedStubExecutor {
            backend: larql_compute::CpuBackend,
        };
        let ffn = NullFfn;
        let mut engine = MarkovResidualEngine::new(None);
        let h = engine
            .prefill_quant_via_executor(&mut weights, &exec, &ffn, &index, &[0u32, 1])
            .expect("fused fallback prefill");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        let h2 = engine
            .decode_step_quant_via_executor(&mut weights, &exec, &ffn, &index, 2)
            .expect("fused fallback decode");
        assert_eq!(h2.shape(), &[1, weights.hidden_size]);
    }
}
