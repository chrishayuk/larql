//! NoCacheEngine — debug fallback that maintains no K/V cache.
//!
//! Every `decode_step` re-runs a full forward pass over the growing
//! prompt+generated sequence. O(N²) wall-clock; correctness baseline
//! against which other engines' bit-parity claims are measured.
//!
//! Wraps `kv_prefill_run` (discarding the cache each call) so the
//! forward-pass code is shared with `StandardEngine`.

use ndarray::Array2;

use crate::generation::kv_prefill_run;
use crate::{EngineInfo, KvEngine};
use larql_inference::ffn::FfnBackend;
use larql_inference::forward::hooks::NoopHook;
use larql_inference::model::ModelWeights;
use larql_inference::{cpu_engine_backend, EngineBackend};

/// No-cache decode. Stores only the running token sequence and re-runs
/// a full forward pass per step.
pub struct NoCacheEngine {
    tokens: Vec<u32>,
    backend: Box<dyn EngineBackend>,
}

impl NoCacheEngine {
    pub fn new() -> Self {
        Self::with_backend(cpu_engine_backend())
    }

    pub fn with_backend(backend: Box<dyn EngineBackend>) -> Self {
        Self {
            tokens: Vec::new(),
            backend,
        }
    }
}

impl Default for NoCacheEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl KvEngine for NoCacheEngine {
    fn name(&self) -> &str {
        "no-cache"
    }

    fn info(&self) -> EngineInfo {
        EngineInfo {
            name: "no-cache".into(),
            description:
                "no K/V cache — full re-forward per step (O(N²)); correctness fallback only".into(),
            backend: self.backend.name().to_string(),
            config: String::new(),
        }
    }

    fn prefill(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        token_ids: &[u32],
    ) -> Option<Array2<f32>> {
        self.tokens = token_ids.to_vec();
        let (hidden, _cache) = kv_prefill_run(
            weights,
            ffn,
            token_ids,
            None,
            Some(self.backend.as_ref()),
            &mut NoopHook,
        )?;
        Some(hidden)
    }

    fn decode_step(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        token_id: u32,
    ) -> Option<Array2<f32>> {
        self.tokens.push(token_id);
        let (hidden, _cache) = kv_prefill_run(
            weights,
            ffn,
            &self.tokens,
            None,
            Some(self.backend.as_ref()),
            &mut NoopHook,
        )?;
        Some(hidden)
    }

    fn prefill_quant(
        &mut self,
        weights: &mut ModelWeights,
        _ffn: &dyn FfnBackend,
        index: &larql_inference::larql_vindex::VectorIndex,
        token_ids: &[u32],
        backend: &dyn larql_inference::ComputeBackend,
    ) -> Option<Array2<f32>> {
        // Phase-1 pattern: dequant Q4K attn tensors into `weights.tensors`,
        // then run the f32 prefill path. Q4K FFN dispatches through a
        // `WalkFfn` constructed from the vindex (the bench passes
        // `NullFfn` because Q4K FFN is engine-side; using `_ffn` would
        // silently skip the FFN). See `kv-dispatch-quantization.md`.
        larql_inference::vindex::ensure_attn_tensors_dequantised(weights, index);
        let walk_ffn = larql_inference::vindex::WalkFfn::from_config(
            weights,
            index,
            larql_inference::vindex::WalkFfnConfig::dense(weights.num_layers),
        )
        .with_backend(backend);
        self.prefill(weights, &walk_ffn, token_ids)
    }

    fn decode_step_quant(
        &mut self,
        weights: &mut ModelWeights,
        _ffn: &dyn FfnBackend,
        index: &larql_inference::larql_vindex::VectorIndex,
        token_id: u32,
        backend: &dyn larql_inference::ComputeBackend,
    ) -> Option<Array2<f32>> {
        larql_inference::vindex::ensure_attn_tensors_dequantised(weights, index);
        let walk_ffn = larql_inference::vindex::WalkFfn::from_config(
            weights,
            index,
            larql_inference::vindex::WalkFfnConfig::dense(weights.num_layers),
        )
        .with_backend(backend);
        self.decode_step(weights, &walk_ffn, token_id)
    }

    // ── Phase 2 migration: executor-driven path ──────────────────────────
    //
    // NoCacheEngine re-runs the full forward pass per step (no K/V cache).
    // Its `prefill` / `decode_step` already honor the caller's FFN backend
    // properly; the legacy `prefill_quant` only substituted a WalkFfn
    // because callers passed `NullFfn`. The executor-driven path skips
    // the substitution and uses the caller's FFN directly.

    fn prefill_quant_via_executor(
        &mut self,
        weights: &mut ModelWeights,
        _executor: &dyn larql_inference::layer_executor::LayerExecutor,
        ffn: &dyn FfnBackend,
        index: &larql_inference::larql_vindex::VectorIndex,
        token_ids: &[u32],
    ) -> Option<Array2<f32>> {
        // No K/V cache so we don't need to drive the per-layer loop
        // through the executor; the existing prefill (which honors the
        // FFN parameter) is the right path. Just dequant first.
        larql_inference::vindex::ensure_attn_tensors_dequantised(weights, index);
        self.prefill(weights, ffn, token_ids)
    }

    fn decode_step_quant_via_executor(
        &mut self,
        weights: &mut ModelWeights,
        _executor: &dyn larql_inference::layer_executor::LayerExecutor,
        ffn: &dyn FfnBackend,
        index: &larql_inference::larql_vindex::VectorIndex,
        token_id: u32,
    ) -> Option<Array2<f32>> {
        larql_inference::vindex::ensure_attn_tensors_dequantised(weights, index);
        self.decode_step(weights, ffn, token_id)
    }

    fn memory_bytes(&self) -> usize {
        // Only persistent state is the token-id list.
        self.tokens.len() * std::mem::size_of::<u32>()
    }

    fn window_tokens(&self) -> usize {
        self.tokens.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_inference::ffn::WeightFfn;
    use larql_inference::forward::hidden_to_raw_logits;
    use larql_inference::test_utils::make_test_weights;

    #[test]
    fn engine_name() {
        assert_eq!(NoCacheEngine::new().name(), "no-cache");
    }

    #[test]
    fn memory_zero_before_prefill() {
        let eng = NoCacheEngine::new();
        assert_eq!(eng.memory_bytes(), 0);
        assert_eq!(eng.window_tokens(), 0);
    }

    #[test]
    fn prefill_returns_hidden_state() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut engine = NoCacheEngine::new();
        let h = engine
            .prefill(&weights, &ffn, &[0u32, 1, 2])
            .expect("prefill");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert_eq!(engine.window_tokens(), 3);
    }

    #[test]
    fn decode_step_appends_and_returns_hidden() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut engine = NoCacheEngine::new();
        engine.prefill(&weights, &ffn, &[0u32, 1]).expect("prefill");
        let h = engine.decode_step(&weights, &ffn, 2).expect("decode");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert_eq!(engine.window_tokens(), 3);
        assert!(hidden_to_raw_logits(&weights, &h)
            .iter()
            .all(|v| v.is_finite()));
    }

    // ── Step 4 parity gate (NoCache) ─────────────────────────────────────
    //
    // Today's `--kv-cache none` does `predict_with_ffn(weights, tokenizer,
    // &ids_so_far, 1, ffn)` per step. `NoCacheEngine` driven through
    // `generate_with_engine` must produce the same token stream on
    // synthetic weights (the test substrate covers the production
    // non-PLE arch; PLE archs are checked at integration time).

    use crate::generation::generate_with_engine;
    use larql_inference::forward::predict_with_ffn;

    fn run_legacy_no_cache(
        weights: &larql_inference::ModelWeights,
        tokenizer: &larql_inference::tokenizers::Tokenizer,
        ffn: &WeightFfn<'_>,
        prompt: &[u32],
        max: usize,
    ) -> Vec<u32> {
        let mut ids = prompt.to_vec();
        let mut generated = Vec::with_capacity(max);
        for _ in 0..max {
            let result = predict_with_ffn(weights, tokenizer, &ids, 1, ffn);
            let Some(&next_id) = result.token_ids.first() else {
                break;
            };
            ids.push(next_id);
            generated.push(next_id);
        }
        generated
    }

    fn run_engine_no_cache(
        weights: &larql_inference::ModelWeights,
        tokenizer: &larql_inference::tokenizers::Tokenizer,
        ffn: &WeightFfn<'_>,
        prompt: &[u32],
        max: usize,
    ) -> Vec<u32> {
        let mut engine = NoCacheEngine::new();
        generate_with_engine(
            &mut engine as &mut dyn crate::KvEngine,
            weights,
            tokenizer,
            ffn,
            prompt,
            max,
            |_, _| {},
        )
    }

    #[test]
    fn parity_no_cache_matches_legacy_predict_with_ffn() {
        use larql_inference::test_utils::make_test_tokenizer;
        let weights = make_test_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let ffn = WeightFfn { weights: &weights };
        let prompt = &[2u32, 3, 5];
        let max = 4;
        let legacy = run_legacy_no_cache(&weights, &tokenizer, &ffn, prompt, max);
        let engine = run_engine_no_cache(&weights, &tokenizer, &ffn, prompt, max);
        assert_eq!(
            engine, legacy,
            "NoCache engine dispatch must produce identical tokens to legacy predict_with_ffn loop"
        );
    }

    #[test]
    fn memory_grows_linearly_with_tokens() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut engine = NoCacheEngine::new();
        engine.prefill(&weights, &ffn, &[0u32]).expect("prefill");
        let mem1 = engine.memory_bytes();
        engine.decode_step(&weights, &ffn, 1).expect("decode 1");
        let mem2 = engine.memory_bytes();
        engine.decode_step(&weights, &ffn, 2).expect("decode 2");
        let mem3 = engine.memory_bytes();
        assert!(mem2 > mem1);
        assert!(mem3 > mem2);
    }

    // ── Q4K paths via CPU fallback ────────────────────────────────────────
    //
    // prefill_quant + decode_step_quant dequant attn tensors into
    // `weights.tensors`, then route through the f32 prefill/decode.
    // Mirrors the markov_residual::engine CPU-fallback test pattern.

    #[test]
    fn prefill_quant_cpu_fallback_runs_end_to_end() {
        use larql_inference::ffn::NullFfn;
        let mut weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let ffn = NullFfn;
        let mut engine = NoCacheEngine::new();
        let h = engine
            .prefill_quant(&mut weights, &ffn, &index, &[0u32, 1, 2], &*backend)
            .expect("prefill_quant cpu fallback");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
    }

    #[test]
    fn decode_step_quant_cpu_fallback_appends_token() {
        use larql_inference::ffn::NullFfn;
        let mut weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let ffn = NullFfn;
        let mut engine = NoCacheEngine::new();
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
            "no-cache memory should grow with each new token"
        );
    }

    /// `Default::default` is wired through `Self::new()` — covers the
    /// `Default for NoCacheEngine` impl body (lines 40-42).
    #[test]
    fn default_returns_empty_engine() {
        let engine = NoCacheEngine::default();
        assert_eq!(engine.name(), "no-cache");
        assert_eq!(engine.memory_bytes(), 0);
        assert_eq!(engine.window_tokens(), 0);
    }

    // ── Phase 2 executor-driven path ──────────────────────────────────────

    #[test]
    fn prefill_quant_via_executor_dequants_and_runs_prefill() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::layer_executor::LocalWalkExecutor;
        let mut weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let executor = LocalWalkExecutor::new(&*backend);
        let ffn = NullFfn;
        let mut engine = NoCacheEngine::new();
        let h = engine
            .prefill_quant_via_executor(&mut weights, &executor, &ffn, &index, &[0u32, 1])
            .expect("prefill via executor");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert_eq!(engine.window_tokens(), 2);
    }

    #[test]
    fn decode_step_quant_via_executor_appends_token() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::layer_executor::LocalWalkExecutor;
        let mut weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let executor = LocalWalkExecutor::new(&*backend);
        let ffn = NullFfn;
        let mut engine = NoCacheEngine::new();
        engine
            .prefill_quant_via_executor(&mut weights, &executor, &ffn, &index, &[0u32])
            .expect("prefill");
        let mem_before = engine.memory_bytes();
        let h = engine
            .decode_step_quant_via_executor(&mut weights, &executor, &ffn, &index, 1)
            .expect("decode via executor");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(engine.memory_bytes() > mem_before);
    }
}
