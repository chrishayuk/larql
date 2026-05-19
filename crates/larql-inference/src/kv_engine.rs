//! KV-cache engine trait and shared types.
//!
//! Defines the abstract surface that the autoregressive decode loop
//! dispatches against. Concrete engine implementations (MarkovResidual,
//! UnlimitedContext, TurboQuant, Apollo, Standard, NoCache) live in
//! `larql-kv` and `impl larql_inference::KvEngine` against this trait.
//!
//! The trait deliberately lives in `larql-inference` rather than
//! `larql-kv` so the dispatch entry point (which lives here, in the
//! crate that owns the forward pass) can reference the trait without
//! a circular dependency on `larql-kv`. See
//! `docs/specs/kv-engine-unification.md` §10.4.
//!
//! Correctness contract: `prefill` and `decode_step` return the
//! pre-`lm_head` hidden state (shape `[1, hidden_dim]`). The caller
//! applies `final_norm + lm_head` to get logits — see
//! [`forward::hidden_to_raw_logits`](crate::forward::hidden_to_raw_logits).

use crate::ffn::FfnBackend;
use crate::ModelWeights;
use ndarray::Array2;

// ─── EngineInfo ───────────────────────────────────────────────────────────────

/// Runtime diagnostics reported by each engine.
#[derive(Debug, Clone)]
pub struct EngineInfo {
    /// Short engine name (e.g. `"markov-rs"`).
    pub name: String,
    /// Human-readable description of the engine's state management strategy.
    pub description: String,
    /// Hardware backend name from [`larql_compute::ComputeBackend::name`]: `"cpu"`, `"metal"`, etc.
    pub backend: String,
    /// Key config parameters (e.g. `"window=512"`), empty string if unconfigured.
    pub config: String,
}

impl EngineInfo {
    pub fn summary(&self) -> String {
        if self.config.is_empty() {
            format!("{} [{}]  {}", self.name, self.backend, self.description)
        } else {
            format!(
                "{} [{}] ({})  {}",
                self.name, self.backend, self.config, self.description
            )
        }
    }
}

// ─── DecodeStageSummary ───────────────────────────────────────────────────────

/// Per-step averages for a completed engine run. Returned from
/// [`KvEngine::stage_summary`] when profiling was enabled at engine
/// construction.
#[derive(Debug, Clone)]
pub struct DecodeStageSummary {
    pub engine: String,
    pub backend: String,
    pub steps: usize,
    pub avg_embed_us: f64,
    /// K/V recompute from stored residuals (MarkovRS only). Split by tier.
    pub avg_recompute_cold_us: f64,
    pub avg_recompute_hot_us: f64,
    pub avg_attention_us: f64,
    pub avg_ffn_us: f64,
    pub avg_total_decode_us: f64,
}

impl DecodeStageSummary {
    pub fn avg_recompute_total_us(&self) -> f64 {
        self.avg_recompute_cold_us + self.avg_recompute_hot_us
    }

    /// Print a human-readable breakdown table.
    pub fn print(&self) {
        let total = self.avg_total_decode_us;
        let pct = |v: f64| if total > 0.0 { v / total * 100.0 } else { 0.0 };

        println!(
            "\nStage breakdown  ({}, {}, {} decode steps avg):",
            self.engine, self.backend, self.steps
        );
        println!("  {:<25} {:>8}  {:>6}", "Stage", "avg_us", "%");
        println!("  {}", "-".repeat(45));
        println!(
            "  {:<25} {:>8.1}  {:>5.1}%",
            "embed",
            self.avg_embed_us,
            pct(self.avg_embed_us)
        );
        if self.avg_recompute_total_us() > 0.0 {
            println!(
                "  {:<25} {:>8.1}  {:>5.1}%",
                "recompute_kv (cold)",
                self.avg_recompute_cold_us,
                pct(self.avg_recompute_cold_us)
            );
            println!(
                "  {:<25} {:>8.1}  {:>5.1}%",
                "recompute_kv (hot)",
                self.avg_recompute_hot_us,
                pct(self.avg_recompute_hot_us)
            );
        }
        println!(
            "  {:<25} {:>8.1}  {:>5.1}%",
            "attention",
            self.avg_attention_us,
            pct(self.avg_attention_us)
        );
        println!(
            "  {:<25} {:>8.1}  {:>5.1}%",
            "ffn",
            self.avg_ffn_us,
            pct(self.avg_ffn_us)
        );
        println!("  {}", "-".repeat(45));
        println!(
            "  {:<25} {:>8.1}  {:>5.1}%",
            "total (measured)", total, 100.0
        );
        println!();
    }
}

// ─── KvEngine trait ───────────────────────────────────────────────────────────

/// Common interface shared by all KV-cache engines.
pub trait KvEngine: Send {
    fn name(&self) -> &str;

    /// Runtime diagnostics: engine name, backend, config, description.
    fn info(&self) -> EngineInfo;

    /// Run the prefill forward pass over all prompt tokens.
    ///
    /// `ffn` is the FFN backend the engine should dispatch through —
    /// typically [`WeightFfn`](crate::ffn::WeightFfn) /
    /// [`BackendFfn`](crate::ffn::BackendFfn) for local compute, or
    /// [`RemoteWalkBackend`](crate::ffn::RemoteWalkBackend) for grid
    /// routing. Engines that don't consult an FFN router (e.g. ones
    /// that recompute FFN from `weights` directly) may ignore this
    /// parameter.
    ///
    /// Returns the hidden state at the final token position (shape `[1, hidden_dim]`).
    fn prefill(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        token_ids: &[u32],
    ) -> Option<Array2<f32>>;

    /// Run one autoregressive decode step for a single new token.
    /// Returns the hidden state (shape `[1, hidden_dim]`).
    fn decode_step(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        token_id: u32,
    ) -> Option<Array2<f32>>;

    /// Bytes of persistent engine state (excludes model weights).
    fn memory_bytes(&self) -> usize;

    /// Token count in the active hot window (varies by engine type).
    fn window_tokens(&self) -> usize {
        0
    }

    /// Cold-tier bytes (residuals or token IDs past the hot window).
    fn cold_bytes(&self) -> usize {
        0
    }

    /// Per-stage timing summary. Returns `None` if profiling was not enabled.
    fn stage_summary(&self) -> Option<DecodeStageSummary> {
        None
    }

    /// Prefill using Q4K quantised weights from `index` and `backend`.
    ///
    /// When the backend supports the fused Q4 pipeline (Metal), this routes
    /// through `backend.prefill_kquant` for full GPU speed. Falls back to the
    /// f32 path when `backend.supports_quant(::larql_compute::QuantFormat::Q4_K) == false` or `index` has no Q4K data.
    ///
    /// `weights` is `&mut` so the engine can lazily insert dequantised f32
    /// attention tensors into `weights.tensors` on the first call (one-time
    /// cost; subsequent decode steps reuse the cached tensors).
    fn prefill_quant(
        &mut self,
        weights: &mut ModelWeights,
        ffn: &dyn FfnBackend,
        index: &larql_vindex::VectorIndex,
        token_ids: &[u32],
        backend: &dyn larql_compute::ComputeBackend,
    ) -> Option<Array2<f32>> {
        let _ = (index, backend);
        self.prefill(weights, ffn, token_ids) // default: f32 fallback
    }

    /// One autoregressive decode step using Q4K weights.
    ///
    /// Same routing semantics as [`prefill_quant`]: Metal via `decode_token`
    /// when available, f32 fallback otherwise.
    fn decode_step_quant(
        &mut self,
        weights: &mut ModelWeights,
        ffn: &dyn FfnBackend,
        index: &larql_vindex::VectorIndex,
        token_id: u32,
        backend: &dyn larql_compute::ComputeBackend,
    ) -> Option<Array2<f32>> {
        let _ = (index, backend);
        self.decode_step(weights, ffn, token_id) // default: f32 fallback
    }

    /// Prefill via a caller-supplied `LayerExecutor` (dense/f32 path).
    /// See [`docs/specs/engine-state-vs-execution.md`].
    ///
    /// Sibling of [`prefill_quant_via_executor`] for engines that
    /// don't have a quant path (no vindex needed). Default impl falls
    /// through to [`prefill`].
    fn prefill_via_executor(
        &mut self,
        weights: &ModelWeights,
        executor: &dyn crate::layer_executor::LayerExecutor,
        ffn: &dyn FfnBackend,
        token_ids: &[u32],
    ) -> Option<Array2<f32>> {
        let _ = executor;
        self.prefill(weights, ffn, token_ids)
    }

    /// One decode step via a caller-supplied `LayerExecutor` (dense/f32).
    /// Sibling of [`decode_step_quant_via_executor`].
    fn decode_step_via_executor(
        &mut self,
        weights: &ModelWeights,
        executor: &dyn crate::layer_executor::LayerExecutor,
        ffn: &dyn FfnBackend,
        token_id: u32,
    ) -> Option<Array2<f32>> {
        let _ = executor;
        self.decode_step(weights, ffn, token_id)
    }

    /// Prefill via a caller-supplied `LayerExecutor`. See
    /// [`docs/specs/engine-state-vs-execution.md`].
    ///
    /// The default impl falls through to [`prefill_quant`] using
    /// `executor.backend()` — engines that haven't migrated yet keep
    /// working unchanged. Migrated engines override this method to
    /// drive the layer loop through the executor and honor the FFN
    /// parameter properly.
    fn prefill_quant_via_executor(
        &mut self,
        weights: &mut ModelWeights,
        executor: &dyn crate::layer_executor::LayerExecutor,
        ffn: &dyn FfnBackend,
        index: &larql_vindex::VectorIndex,
        token_ids: &[u32],
    ) -> Option<Array2<f32>> {
        self.prefill_quant(weights, ffn, index, token_ids, executor.backend())
    }

    /// One decode step via a caller-supplied `LayerExecutor`. See
    /// [`prefill_quant_via_executor`] for the migration contract.
    fn decode_step_quant_via_executor(
        &mut self,
        weights: &mut ModelWeights,
        executor: &dyn crate::layer_executor::LayerExecutor,
        ffn: &dyn FfnBackend,
        index: &larql_vindex::VectorIndex,
        token_id: u32,
    ) -> Option<Array2<f32>> {
        self.decode_step_quant(weights, ffn, index, token_id, executor.backend())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_info_summary_with_config() {
        let info = EngineInfo {
            name: "markov-rs".into(),
            description: "residual KV".into(),
            backend: "cpu".into(),
            config: "window=512".into(),
        };
        let s = info.summary();
        assert!(s.contains("markov-rs"));
        assert!(s.contains("cpu"));
        assert!(s.contains("window=512"));
    }

    #[test]
    fn engine_info_summary_no_config() {
        let info = EngineInfo {
            name: "test".into(),
            description: "desc".into(),
            backend: "metal".into(),
            config: String::new(),
        };
        let s = info.summary();
        assert!(!s.contains("()"));
    }

    #[test]
    fn decode_stage_summary_recompute_total() {
        let s = DecodeStageSummary {
            engine: "test".into(),
            backend: "cpu".into(),
            steps: 10,
            avg_embed_us: 1.0,
            avg_recompute_cold_us: 2.0,
            avg_recompute_hot_us: 3.0,
            avg_attention_us: 4.0,
            avg_ffn_us: 5.0,
            avg_total_decode_us: 15.0,
        };
        assert_eq!(s.avg_recompute_total_us(), 5.0);
    }

    /// Cover `DecodeStageSummary::print` — both the recompute>0 branch and
    /// the total>0 percentage branch. Output goes to stdout (captured by the
    /// test harness); this is a smoke test for the formatting code path.
    #[test]
    fn decode_stage_summary_print_with_recompute() {
        let s = DecodeStageSummary {
            engine: "markov-rs".into(),
            backend: "cpu".into(),
            steps: 10,
            avg_embed_us: 100.0,
            avg_recompute_cold_us: 500.0,
            avg_recompute_hot_us: 300.0,
            avg_attention_us: 1500.0,
            avg_ffn_us: 800.0,
            avg_total_decode_us: 3200.0,
        };
        s.print();
    }

    /// `print` must also handle the no-recompute, zero-total branch — the
    /// `pct` fallback when `avg_total_decode_us == 0.0` and the
    /// `avg_recompute_total_us() == 0` short-circuit.
    #[test]
    fn decode_stage_summary_print_no_recompute_zero_total() {
        let s = DecodeStageSummary {
            engine: "no-cache".into(),
            backend: "metal".into(),
            steps: 0,
            avg_embed_us: 0.0,
            avg_recompute_cold_us: 0.0,
            avg_recompute_hot_us: 0.0,
            avg_attention_us: 0.0,
            avg_ffn_us: 0.0,
            avg_total_decode_us: 0.0,
        };
        s.print();
    }

    /// Synthetic engine that only implements the required trait methods,
    /// leaving every default (`window_tokens`, `cold_bytes`, `stage_summary`,
    /// `prefill_quant`, `decode_step_quant`) to fire. Exercises the default
    /// bodies that no shipped engine routes through (every concrete engine
    /// overrides them).
    struct DefaultsOnlyEngine {
        prefill_calls: usize,
        decode_calls: usize,
    }

    impl KvEngine for DefaultsOnlyEngine {
        fn name(&self) -> &str {
            "defaults-only"
        }
        fn info(&self) -> EngineInfo {
            EngineInfo {
                name: self.name().into(),
                description: "test fixture".into(),
                backend: "cpu".into(),
                config: String::new(),
            }
        }
        fn prefill(
            &mut self,
            _weights: &ModelWeights,
            _ffn: &dyn FfnBackend,
            _token_ids: &[u32],
        ) -> Option<Array2<f32>> {
            self.prefill_calls += 1;
            Some(Array2::zeros((1, 4)))
        }
        fn decode_step(
            &mut self,
            _weights: &ModelWeights,
            _ffn: &dyn FfnBackend,
            _token_id: u32,
        ) -> Option<Array2<f32>> {
            self.decode_calls += 1;
            Some(Array2::zeros((1, 4)))
        }
        fn memory_bytes(&self) -> usize {
            0
        }
    }

    #[test]
    fn defaults_window_tokens_and_cold_bytes_are_zero() {
        let engine = DefaultsOnlyEngine {
            prefill_calls: 0,
            decode_calls: 0,
        };
        assert_eq!(engine.window_tokens(), 0);
        assert_eq!(engine.cold_bytes(), 0);
        assert!(engine.stage_summary().is_none());
        assert_eq!(engine.name(), "defaults-only");
    }

    /// All four `*_via_executor` default impls dispatch through to their
    /// non-executor sibling, which on `DefaultsOnlyEngine` falls back to
    /// `prefill` / `decode_step`. Covers the function bodies of
    /// `prefill_via_executor` (224-233), `decode_step_via_executor`
    /// (237-246), `prefill_quant_via_executor` (256-265),
    /// `decode_step_quant_via_executor` (269-278).
    #[test]
    fn defaults_via_executor_methods_dispatch_to_non_executor_siblings() {
        struct StubExecutor {
            backend: larql_compute::CpuBackend,
        }
        impl crate::layer_executor::LayerExecutor for StubExecutor {
            fn backend(&self) -> &dyn larql_compute::ComputeBackend {
                &self.backend
            }
            fn dispatch_kind(&self) -> crate::layer_executor::ExecutorDispatchKind {
                crate::layer_executor::ExecutorDispatchKind::PerLayer
            }
            fn name(&self) -> &str {
                "stub"
            }
        }
        let exec = StubExecutor {
            backend: larql_compute::CpuBackend,
        };
        let weights = crate::test_utils::make_test_weights();
        let index = crate::test_utils::make_test_vindex(&weights);
        let ffn = crate::ffn::WeightFfn { weights: &weights };
        let mut engine = DefaultsOnlyEngine {
            prefill_calls: 0,
            decode_calls: 0,
        };

        // prefill_via_executor → prefill
        let out = engine.prefill_via_executor(&weights, &exec, &ffn, &[0, 1]);
        assert!(out.is_some());
        assert_eq!(engine.prefill_calls, 1);

        // decode_step_via_executor → decode_step
        let out = engine.decode_step_via_executor(&weights, &exec, &ffn, 2);
        assert!(out.is_some());
        assert_eq!(engine.decode_calls, 1);

        // prefill_quant_via_executor → prefill_quant → prefill (default fallback)
        let mut weights_q = crate::test_utils::make_test_weights();
        let out = engine.prefill_quant_via_executor(&mut weights_q, &exec, &ffn, &index, &[0, 1]);
        assert!(out.is_some());
        assert_eq!(engine.prefill_calls, 2);

        // decode_step_quant_via_executor → decode_step_quant → decode_step
        let out = engine.decode_step_quant_via_executor(&mut weights_q, &exec, &ffn, &index, 3);
        assert!(out.is_some());
        assert_eq!(engine.decode_calls, 2);
    }

    #[test]
    fn defaults_q4k_methods_fall_back_to_f32() {
        let weights = crate::test_utils::make_test_weights();
        let index = crate::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let ffn = crate::ffn::WeightFfn { weights: &weights };
        let mut engine = DefaultsOnlyEngine {
            prefill_calls: 0,
            decode_calls: 0,
        };

        let mut weights_q4k = crate::test_utils::make_test_weights();
        let out = engine.prefill_quant(&mut weights_q4k, &ffn, &index, &[1, 2, 3], &*backend);
        assert!(out.is_some());
        assert_eq!(
            engine.prefill_calls, 1,
            "default prefill_quant must dispatch to prefill"
        );

        let out = engine.decode_step_quant(&mut weights_q4k, &ffn, &index, 4, &*backend);
        assert!(out.is_some());
        assert_eq!(
            engine.decode_calls, 1,
            "default decode_step_quant must dispatch to decode_step"
        );
    }
}
