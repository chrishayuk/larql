//! MarkovResidualEngine — KvEngine implementation.

use larql_compute::{cpu_backend, ComputeBackend};
use larql_vindex::VectorIndex;
use ndarray::Array2;

use super::compute::{rs_decode_step, rs_decode_step_profiled, rs_prefill};
use super::q4k::{ensure_attn_tensors_dequantised, rs_decode_step_walk, rs_prefill_walk};
use super::store::RsStore;
use crate::profiler::{DecodeStageSummary, EngineProfiler};
use crate::{EngineInfo, KvEngine};
use larql_inference::model::ModelWeights;

pub struct MarkovResidualEngine {
    window_size: Option<usize>,
    store: Option<RsStore>,
    backend: Box<dyn ComputeBackend>,
    profiling: bool,
    profile: EngineProfiler,
    metal_prefill_done: bool,
}

impl MarkovResidualEngine {
    pub fn new(window_size: Option<usize>) -> Self {
        Self::with_backend(window_size, cpu_backend())
    }

    pub fn with_backend(window_size: Option<usize>, backend: Box<dyn ComputeBackend>) -> Self {
        Self {
            window_size,
            store: None,
            backend,
            profiling: false,
            profile: EngineProfiler::default(),
            metal_prefill_done: false,
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

    fn prefill(&mut self, weights: &ModelWeights, token_ids: &[u32]) -> Option<Array2<f32>> {
        let result = rs_prefill(weights, token_ids, self.window_size, self.backend.as_ref());
        let hidden = result.hidden.clone();
        self.store = Some(result.store);
        Some(hidden)
    }

    fn decode_step(&mut self, weights: &ModelWeights, token_id: u32) -> Option<Array2<f32>> {
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

    fn prefill_q4k(
        &mut self,
        weights: &mut ModelWeights,
        index: &VectorIndex,
        token_ids: &[u32],
        backend: &dyn ComputeBackend,
    ) -> Option<Array2<f32>> {
        use crate::engines::unlimited_context::engine::q4k_prefill_metal;
        if let Some(h) = q4k_prefill_metal(weights, index, token_ids, backend) {
            self.metal_prefill_done = true;
            self.store = None;
            return Some(h);
        }
        self.metal_prefill_done = false;
        ensure_attn_tensors_dequantised(weights, index);
        let result = rs_prefill_walk(weights, index, token_ids, self.window_size, backend);
        let hidden = result.hidden.clone();
        self.store = Some(result.store);
        Some(hidden)
    }

    fn decode_step_q4k(
        &mut self,
        weights: &mut ModelWeights,
        index: &VectorIndex,
        token_id: u32,
        backend: &dyn ComputeBackend,
    ) -> Option<Array2<f32>> {
        use crate::engines::unlimited_context::engine::q4k_decode_token;
        if self.metal_prefill_done {
            if let Some(h) = q4k_decode_token(weights, index, token_id, backend) {
                return Some(h);
            }
        }
        ensure_attn_tensors_dequantised(weights, index);
        let rs = self.store.take()?;
        let (hidden, new_rs) = rs_decode_step_walk(weights, index, token_id, rs, backend)?;
        self.store = Some(new_rs);
        Some(hidden)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::KvEngine;
    use larql_inference::forward::hidden_to_raw_logits;
    use larql_inference::test_utils::make_test_weights;

    #[cfg(all(target_arch = "wasm32", feature = "browser-tests"))]
    wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

    // ── Construction ──────────────────────────────────────────────────────────

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn engine_name() {
        assert_eq!(MarkovResidualEngine::new(None).name(), "markov-rs");
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn engine_memory_zero_before_prefill() {
        let eng = MarkovResidualEngine::new(None);
        assert_eq!(eng.memory_bytes(), 0);
        assert_eq!(eng.window_tokens(), 0);
        assert_eq!(eng.cold_bytes(), 0);
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn engine_info_full_window() {
        let eng = MarkovResidualEngine::new(None);
        let info = eng.info();
        assert!(
            info.config.contains("full"),
            "expected 'full' in config, got '{}'",
            info.config
        );
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn prefill_stores_residuals_for_all_layers() {
        let weights = make_test_weights();
        let mut engine = MarkovResidualEngine::new(None);
        let h = engine.prefill(&weights, &[0u32, 1, 2]).expect("prefill");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(
            engine.memory_bytes() > 0,
            "store should be non-empty after prefill"
        );
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn decode_step_produces_finite_logits() {
        let weights = make_test_weights();
        let mut engine = MarkovResidualEngine::new(None);
        engine.prefill(&weights, &[0u32, 1]).expect("prefill");
        let h = engine.decode_step(&weights, 2).expect("decode");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(hidden_to_raw_logits(&weights, &h)
            .iter()
            .all(|v| v.is_finite()));
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn memory_grows_with_each_decode_step() {
        let weights = make_test_weights();
        let mut engine = MarkovResidualEngine::new(None);
        engine.prefill(&weights, &[0u32]).expect("prefill");
        let mem_after_prefill = engine.memory_bytes();
        engine.decode_step(&weights, 1).expect("decode 1");
        let mem_after_1 = engine.memory_bytes();
        engine.decode_step(&weights, 2).expect("decode 2");
        let mem_after_2 = engine.memory_bytes();
        assert!(
            mem_after_1 > mem_after_prefill,
            "memory should grow with decode steps"
        );
        assert!(mem_after_2 > mem_after_1);
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn window_clipping_limits_hot_store() {
        let weights = make_test_weights();
        let mut engine = MarkovResidualEngine::new(Some(2)); // window=2 tokens
        engine
            .prefill(&weights, &[0u32, 1, 2, 3, 4])
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

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn multiple_decode_steps_produce_consistent_shapes() {
        let weights = make_test_weights();
        let mut engine = MarkovResidualEngine::new(None);
        engine.prefill(&weights, &[0u32]).expect("prefill");
        for step in 0..3 {
            let h = engine.decode_step(&weights, step as u32).expect("decode");
            assert_eq!(h.shape(), &[1, weights.hidden_size], "step {step}");
        }
    }

    // ── Profiling ─────────────────────────────────────────────────────────────

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn with_profiling_enables_profiling_branch() {
        let weights = make_test_weights();
        let mut engine = MarkovResidualEngine::new(None).with_profiling(true);
        // No decode yet → stage_summary returns None even with profiling on.
        assert!(engine.stage_summary().is_none());

        engine.prefill(&weights, &[0u32, 1]).expect("prefill");
        engine.decode_step(&weights, 2).expect("decode");

        let summary = engine.stage_summary().expect("profiling summary");
        assert_eq!(summary.engine, "markov-rs");
        assert_eq!(summary.steps, 1);
        assert!(summary.avg_total_decode_us > 0.0);
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn stage_summary_none_without_profiling() {
        let weights = make_test_weights();
        let mut engine = MarkovResidualEngine::new(None); // profiling: false
        engine.prefill(&weights, &[0u32]).expect("prefill");
        engine.decode_step(&weights, 1).expect("decode");
        assert!(
            engine.stage_summary().is_none(),
            "stage_summary must be None when profiling is disabled"
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn profiling_decode_path_matches_unprofiled_shape() {
        // Two engines: one profiled, one not. Both should yield hidden states
        // of the same shape after the same prefill+decode sequence.
        let weights = make_test_weights();
        let mut profiled = MarkovResidualEngine::new(None).with_profiling(true);
        let mut plain = MarkovResidualEngine::new(None);
        profiled.prefill(&weights, &[0u32, 1]).unwrap();
        plain.prefill(&weights, &[0u32, 1]).unwrap();
        let h_p = profiled.decode_step(&weights, 2).unwrap();
        let h_n = plain.decode_step(&weights, 2).unwrap();
        assert_eq!(h_p.shape(), h_n.shape());
    }

    // ── Q4K paths via CPU fallback ────────────────────────────────────────
    //
    // On a CPU backend, `q4k_prefill_metal` returns `None`, so the engine
    // falls through to `rs_prefill_walk` against the synthetic VectorIndex.
    // This exercises the prefill_q4k / decode_step_q4k branches that the
    // Metal-only happy path also takes (apart from the Metal early-return).

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn prefill_q4k_cpu_fallback_runs_walk_path() {
        let mut weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let mut engine = MarkovResidualEngine::new(None);
        let h = engine
            .prefill_q4k(&mut weights, &index, &[0u32, 1, 2], &*backend)
            .expect("prefill_q4k cpu fallback");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(engine.memory_bytes() > 0);
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn decode_step_q4k_cpu_fallback_extends_store() {
        let mut weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let mut engine = MarkovResidualEngine::new(None);
        engine
            .prefill_q4k(&mut weights, &index, &[0u32, 1], &*backend)
            .expect("prefill_q4k");
        let mem_before = engine.memory_bytes();
        let h = engine
            .decode_step_q4k(&mut weights, &index, 2, &*backend)
            .expect("decode_step_q4k cpu fallback");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(
            engine.memory_bytes() > mem_before,
            "store should grow after decode_step_q4k"
        );
    }
}
