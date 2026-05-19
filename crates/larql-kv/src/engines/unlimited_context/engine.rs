//! `UnlimitedContextEngine` — window-based KV cache with boundary-checkpoint replay.
//!
//! Window lifecycle:
//!   1. `process(tokens)` — extends the active window's K,V via
//!      `rs_extend_from_checkpoint`. Auto-closes when the window fills.
//!   2. `close_window()` — saves last-position K,V to `CheckpointStore`,
//!      appends token IDs to `TokenArchive`, resets active window.
//!   3. `replay_window(id)` — reconstructs a window's full K,V by replaying
//!      archived tokens from the prior checkpoint.
//!   4. `stats()` — total bytes, windows, compression ratio vs full KV.
//!
//! Memory at 370K tokens (Gemma 3 4B, W=512):
//!   Checkpoints ≈ 278 KB/window × N_windows
//!   Token archive = 4 bytes/token
//!   Total ≈ 30 MB  vs  25.8 GB for Standard KV  (≈2,000×)

use larql_compute::ComputeBackend;
use larql_inference::{cpu_engine_backend, EngineBackend};
use larql_vindex::VectorIndex;
use ndarray::Array2;
use serde::Serialize;

use super::checkpoint_store::CheckpointStore;
use super::extend::{
    empty_prior, rs_extend_from_checkpoint_backend, rs_extend_from_checkpoint_q4k,
};
use super::token_archive::TokenArchive;
use crate::engines::markov_residual::ensure_attn_tensors_dequantised;
use crate::{EngineInfo, KvEngine};
use larql_inference::attention::SharedKV;
use larql_inference::ffn::FfnBackend;
use larql_inference::model::ModelWeights;

// ─── EngineStats ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct EngineStats {
    pub total_tokens: usize,
    pub archived_windows: usize,
    pub current_window_id: usize,
    pub current_window_tokens: usize,
    pub checkpoint_bytes: usize,
    pub archive_bytes: usize,
    pub total_boundary_bytes: usize,
    pub equivalent_kv_bytes: usize,
    pub compression_ratio: f64,
}

impl EngineStats {
    pub fn summary(&self) -> String {
        format!(
            "{} windows / {} tokens — {:.0}× compression vs full KV",
            self.archived_windows, self.total_tokens, self.compression_ratio,
        )
    }
}

// ─── Engine ──────────────────────────────────────────────────────────────────

pub struct UnlimitedContextEngine {
    pub window_size: usize,
    pub checkpoints: CheckpointStore,
    pub archive: TokenArchive,

    current_window_id: usize,
    current_window_tokens: Vec<u32>,
    current_window_kv: Option<Vec<SharedKV>>,
    abs_offset: usize,
    /// Hidden state at the last processed token; set by `process()`.
    last_hidden: Option<Array2<f32>>,
    backend: Box<dyn EngineBackend>,
}

impl UnlimitedContextEngine {
    pub fn new(window_size: usize) -> Self {
        Self::with_backend(window_size, cpu_engine_backend())
    }

    pub fn with_backend(window_size: usize, backend: Box<dyn EngineBackend>) -> Self {
        Self {
            window_size,
            checkpoints: CheckpointStore::new(),
            archive: TokenArchive::new(),
            current_window_id: 0,
            current_window_tokens: Vec::new(),
            current_window_kv: None,
            abs_offset: 0,
            last_hidden: None,
            backend,
        }
    }

    /// Feed tokens into the engine. Windows auto-close when they fill.
    pub fn process(&mut self, weights: &ModelWeights, tokens: &[u32]) -> Option<()> {
        let mut remaining = tokens;
        while !remaining.is_empty() {
            let free = self.window_size - self.current_window_tokens.len();
            let take = remaining.len().min(free);
            let (chunk, rest) = remaining.split_at(take);
            self.extend_current(weights, chunk)?;
            remaining = rest;
            if self.current_window_tokens.len() >= self.window_size {
                self.close_window();
            }
        }
        Some(())
    }

    /// Close any partial current window. Call before replay if the window hasn't filled.
    pub fn flush(&mut self) {
        if !self.current_window_tokens.is_empty() {
            self.close_window();
        }
    }

    /// Reconstruct a window's full K,V by replaying its archived tokens from
    /// the prior window's boundary checkpoint.
    pub fn replay_window(
        &self,
        weights: &ModelWeights,
        window_id: usize,
    ) -> Option<(Vec<SharedKV>, usize)> {
        let (tokens, abs_offset) = self.archive.retrieve(window_id)?;

        let prior = if window_id > 0 && self.checkpoints.contains(window_id - 1) {
            let (ckpt, _) = self.checkpoints.load(window_id - 1)?;
            ckpt
        } else {
            empty_prior(weights)
        };

        let out = rs_extend_from_checkpoint_backend(
            weights,
            tokens,
            prior,
            abs_offset,
            self.backend.as_ref(),
        )?;
        let abs_end = abs_offset + tokens.len() - 1;
        Some((out.kv_cache, abs_end))
    }

    /// Total storage and context statistics.
    pub fn stats(&self, weights: &ModelWeights) -> EngineStats {
        let arch = &*weights.arch;
        let num_layers = weights.num_layers;
        let kv_dim_sum: usize = (0..num_layers)
            .map(|l| arch.num_kv_heads_for_layer(l) * arch.head_dim_for_layer(l))
            .sum();

        let total_archived = self.archive.total_tokens();
        let current = self.current_window_tokens.len();
        let total_tokens = total_archived + current;

        let equivalent_kv_bytes = total_tokens * kv_dim_sum * 2 * 2;
        let checkpoint_bytes = self.checkpoints.total_bytes();
        let archive_bytes = self.archive.total_bytes();
        let total_boundary_bytes = checkpoint_bytes + archive_bytes;
        let compression_ratio = if total_boundary_bytes == 0 {
            0.0
        } else {
            equivalent_kv_bytes as f64 / total_boundary_bytes as f64
        };

        EngineStats {
            total_tokens,
            archived_windows: self.archive.len(),
            current_window_id: self.current_window_id,
            current_window_tokens: current,
            checkpoint_bytes,
            archive_bytes,
            total_boundary_bytes,
            equivalent_kv_bytes,
            compression_ratio,
        }
    }

    /// CPU Q4K equivalent of `process()` — uses `rs_extend_from_checkpoint_q4k`
    /// (WalkFfn for FFN) instead of the f32-backed `rs_extend_from_checkpoint_backend`.
    fn process_q4k(
        &mut self,
        weights: &ModelWeights,
        index: &VectorIndex,
        tokens: &[u32],
        backend: &dyn ComputeBackend,
    ) -> Option<()> {
        let mut remaining = tokens;
        while !remaining.is_empty() {
            let free = self.window_size - self.current_window_tokens.len();
            let take = remaining.len().min(free);
            let (chunk, rest) = remaining.split_at(take);
            self.extend_current_q4k(weights, index, chunk, backend)?;
            remaining = rest;
            if self.current_window_tokens.len() >= self.window_size {
                self.close_window();
            }
        }
        Some(())
    }

    fn extend_current_q4k(
        &mut self,
        weights: &ModelWeights,
        index: &VectorIndex,
        chunk: &[u32],
        backend: &dyn ComputeBackend,
    ) -> Option<()> {
        if chunk.is_empty() {
            return Some(());
        }

        let prior = if self.current_window_tokens.is_empty() {
            if self.current_window_id > 0 && self.checkpoints.contains(self.current_window_id - 1) {
                let (ckpt, _) = self.checkpoints.load(self.current_window_id - 1)?;
                ckpt
            } else {
                empty_prior(weights)
            }
        } else {
            self.current_window_kv
                .take()
                .unwrap_or_else(|| empty_prior(weights))
        };

        let abs_start = self.abs_offset + self.current_window_tokens.len();
        let out = rs_extend_from_checkpoint_q4k(weights, index, chunk, prior, abs_start, backend)?;

        self.last_hidden = Some(out.last_hidden);
        self.current_window_kv = Some(out.kv_cache);
        self.current_window_tokens.extend_from_slice(chunk);
        Some(())
    }

    fn current_kv_bytes(&self) -> usize {
        self.current_window_kv.as_ref().map_or(0, |kv| {
            kv.iter().map(|(k, v)| (k.len() + v.len()) * 4).sum()
        })
    }

    fn extend_current(&mut self, weights: &ModelWeights, chunk: &[u32]) -> Option<()> {
        if chunk.is_empty() {
            return Some(());
        }

        let prior = if self.current_window_tokens.is_empty() {
            if self.current_window_id > 0 && self.checkpoints.contains(self.current_window_id - 1) {
                let (ckpt, _) = self.checkpoints.load(self.current_window_id - 1)?;
                ckpt
            } else {
                empty_prior(weights)
            }
        } else {
            self.current_window_kv
                .take()
                .unwrap_or_else(|| empty_prior(weights))
        };

        let abs_start = self.abs_offset + self.current_window_tokens.len();
        let out = rs_extend_from_checkpoint_backend(
            weights,
            chunk,
            prior,
            abs_start,
            self.backend.as_ref(),
        )?;

        self.last_hidden = Some(out.last_hidden);
        self.current_window_kv = Some(out.kv_cache);
        self.current_window_tokens.extend_from_slice(chunk);
        Some(())
    }

    fn close_window(&mut self) {
        let kv = match self.current_window_kv.take() {
            Some(kv) => kv,
            None => return,
        };

        let last_kv: Vec<SharedKV> = kv
            .iter()
            .map(|(k, v)| {
                let n = k.shape()[0];
                let last_k = k.slice(ndarray::s![n - 1..n, ..]).to_owned();
                let last_v = v.slice(ndarray::s![n - 1..n, ..]).to_owned();
                (last_k, last_v)
            })
            .collect();

        let window_len = self.current_window_tokens.len();
        let abs_end = self.abs_offset + window_len - 1;

        self.checkpoints
            .save(self.current_window_id, last_kv, abs_end);
        self.archive.archive(
            self.current_window_id,
            std::mem::take(&mut self.current_window_tokens),
            self.abs_offset,
        );
        self.abs_offset += window_len;
        self.current_window_id += 1;
    }
}

impl KvEngine for UnlimitedContextEngine {
    fn name(&self) -> &str {
        "unlimited-context"
    }

    fn info(&self) -> EngineInfo {
        let mem =
            self.checkpoints.total_bytes() + self.archive.total_bytes() + self.current_kv_bytes();
        EngineInfo {
            name: "unlimited-context".into(),
            description: format!(
                "window-boundary KV checkpoints + token replay \
                 (windows={}, tokens={}, mem={:.1}MB)",
                self.archive.len(),
                self.archive.total_tokens() + self.current_window_tokens.len(),
                mem as f64 / 1_048_576.0,
            ),
            backend: self.backend.name().to_string(),
            config: format!("window={}", self.window_size),
        }
    }

    fn prefill(
        &mut self,
        weights: &ModelWeights,
        _ffn: &dyn FfnBackend,
        token_ids: &[u32],
    ) -> Option<Array2<f32>> {
        self.process(weights, token_ids)?;
        self.last_hidden.clone()
    }

    fn decode_step(
        &mut self,
        weights: &ModelWeights,
        _ffn: &dyn FfnBackend,
        token_id: u32,
    ) -> Option<Array2<f32>> {
        self.process(weights, &[token_id])?;
        self.last_hidden.clone()
    }

    fn memory_bytes(&self) -> usize {
        self.checkpoints.total_bytes() + self.archive.total_bytes() + self.current_kv_bytes()
    }

    fn window_tokens(&self) -> usize {
        self.current_window_tokens.len()
    }

    fn cold_bytes(&self) -> usize {
        self.checkpoints.total_bytes() + self.archive.total_bytes()
    }

    /// Q4K prefill — runs the windowed-checkpoint extension regardless of
    /// backend. Engines that want the backend's fused fast path must
    /// select `StandardEngine` explicitly; this engine's whole identity
    /// is window-bounded K/V with checkpoint replay, and bypassing to
    /// fused would skip every checkpoint we'd otherwise emit.
    fn prefill_quant(
        &mut self,
        weights: &mut ModelWeights,
        _ffn: &dyn FfnBackend,
        index: &VectorIndex,
        token_ids: &[u32],
        backend: &dyn ComputeBackend,
    ) -> Option<Array2<f32>> {
        ensure_attn_tensors_dequantised(weights, index);
        self.process_q4k(weights, index, token_ids, backend)?;
        self.last_hidden.clone()
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
        self.process_q4k(weights, index, &[token_id], backend)?;
        self.last_hidden.clone()
    }

    // ── Executor-aware migration (Phase 2 of engine-state-vs-execution spec) ──
    //
    // Drive the per-token layer loop through a caller-supplied `LayerExecutor`
    // and honor the caller-supplied `FfnBackend`. The legacy `*_quant` methods
    // construct their own `WalkFfn` and ignore the FFN parameter; remote-FFN
    // deployments (`larql bench --ffn http://shard:8080`) need this path so
    // the engine actually dispatches through the supplied backend.
    //
    // Window-close semantics (checkpoint + archive at window boundaries) are
    // identical to `process_q4k` / `extend_current_q4k` — the executor only
    // owns per-layer compute; window state is engine state.
    fn prefill_quant_via_executor(
        &mut self,
        weights: &mut ModelWeights,
        executor: &dyn larql_inference::layer_executor::LayerExecutor,
        ffn: &dyn FfnBackend,
        index: &VectorIndex,
        token_ids: &[u32],
    ) -> Option<Array2<f32>> {
        use larql_inference::layer_executor::ExecutorDispatchKind;
        // Spec §3.4: this engine's state policy (windowed checkpoints) is
        // expressible against per-layer dispatch only. Transparent degrade
        // on fused executors until the Phase 3 refusal contract lands.
        if matches!(executor.dispatch_kind(), ExecutorDispatchKind::Fused) {
            return self.prefill_quant(weights, ffn, index, token_ids, executor.backend());
        }
        ensure_attn_tensors_dequantised(weights, index);
        self.process_via_executor(weights, executor, ffn, token_ids)?;
        self.last_hidden.clone()
    }

    fn decode_step_quant_via_executor(
        &mut self,
        weights: &mut ModelWeights,
        executor: &dyn larql_inference::layer_executor::LayerExecutor,
        ffn: &dyn FfnBackend,
        index: &VectorIndex,
        token_id: u32,
    ) -> Option<Array2<f32>> {
        use larql_inference::layer_executor::ExecutorDispatchKind;
        if matches!(executor.dispatch_kind(), ExecutorDispatchKind::Fused) {
            return self.decode_step_quant(weights, ffn, index, token_id, executor.backend());
        }
        ensure_attn_tensors_dequantised(weights, index);
        self.process_via_executor(weights, executor, ffn, &[token_id])?;
        self.last_hidden.clone()
    }
}

// ── Executor-driven window extension ─────────────────────────────────────────

impl UnlimitedContextEngine {
    /// Executor-aware analogue of `process_q4k`: feeds tokens into the
    /// current window, auto-closes on fill, drives per-layer compute
    /// through `executor` instead of constructing a local `WalkFfn`.
    fn process_via_executor(
        &mut self,
        weights: &ModelWeights,
        executor: &dyn larql_inference::layer_executor::LayerExecutor,
        ffn: &dyn FfnBackend,
        tokens: &[u32],
    ) -> Option<()> {
        let mut remaining = tokens;
        while !remaining.is_empty() {
            let free = self.window_size - self.current_window_tokens.len();
            let take = remaining.len().min(free);
            let (chunk, rest) = remaining.split_at(take);
            self.extend_current_via_executor(weights, executor, ffn, chunk)?;
            remaining = rest;
            if self.current_window_tokens.len() >= self.window_size {
                self.close_window();
            }
        }
        Some(())
    }

    fn extend_current_via_executor(
        &mut self,
        weights: &ModelWeights,
        executor: &dyn larql_inference::layer_executor::LayerExecutor,
        ffn: &dyn FfnBackend,
        chunk: &[u32],
    ) -> Option<()> {
        use larql_inference::forward::embed_tokens_pub;
        if chunk.is_empty() {
            return Some(());
        }

        let mut kv_cache: Vec<SharedKV> = if self.current_window_tokens.is_empty() {
            if self.current_window_id > 0 && self.checkpoints.contains(self.current_window_id - 1) {
                let (ckpt, _) = self.checkpoints.load(self.current_window_id - 1)?;
                ckpt
            } else {
                super::extend::empty_prior(weights)
            }
        } else {
            self.current_window_kv
                .take()
                .unwrap_or_else(|| super::extend::empty_prior(weights))
        };

        let num_layers = weights.num_layers;
        if kv_cache.len() != num_layers {
            return None;
        }
        let abs_start = self.abs_offset + self.current_window_tokens.len();
        let mut last_hidden: Option<Array2<f32>> = None;

        for (i, &token_id) in chunk.iter().enumerate() {
            let abs_position = abs_start + i;
            let mut h = embed_tokens_pub(weights, &[token_id]);

            for (layer, kv_slot) in kv_cache.iter_mut().enumerate() {
                let (h_out, new_kv) =
                    executor.run_decode_layer(weights, layer, &h, kv_slot, abs_position, ffn)?;
                h = h_out;
                *kv_slot = new_kv;
            }
            last_hidden = Some(h);
        }

        self.last_hidden = last_hidden;
        self.current_window_kv = Some(kv_cache);
        self.current_window_tokens.extend_from_slice(chunk);
        Some(())
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_engine_is_empty() {
        let eng = UnlimitedContextEngine::new(512);
        assert_eq!(eng.window_size, 512);
        assert_eq!(eng.archive.len(), 0);
        assert_eq!(eng.checkpoints.len(), 0);
        assert_eq!(eng.current_window_id, 0);
        assert_eq!(eng.memory_bytes(), 0);
    }

    #[test]
    fn engine_info_backend_is_cpu() {
        let eng = UnlimitedContextEngine::new(256);
        let info = eng.info();
        assert_eq!(info.name, "unlimited-context");
        assert!(
            info.backend.starts_with("cpu"),
            "expected cpu backend, got {:?}",
            info.backend
        );
        assert_eq!(info.config, "window=256");
        assert!(info.summary().contains("unlimited-context"));
        assert!(info.summary().contains("cpu"));
    }

    #[test]
    fn engine_info_config_contains_window_size() {
        let eng = UnlimitedContextEngine::new(1024);
        assert!(eng.info().config.contains("1024"));
    }

    #[test]
    fn window_tokens_and_cold_bytes_start_zero() {
        let eng = UnlimitedContextEngine::new(512);
        assert_eq!(eng.window_tokens(), 0);
        assert_eq!(eng.cold_bytes(), 0);
    }

    // ── prefill / decode cycle ─────────────────────────────────────────────────

    #[test]
    fn prefill_returns_hidden_state() {
        use larql_inference::ffn::WeightFfn;
        use larql_inference::test_utils::make_test_weights;
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut engine = UnlimitedContextEngine::new(512);
        let h = engine
            .prefill(&weights, &ffn, &[0u32, 1, 2])
            .expect("prefill failed");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(
            h.iter().all(|v| v.is_finite()),
            "hidden state should be finite"
        );
    }

    #[test]
    fn decode_step_returns_hidden_state() {
        use larql_inference::ffn::WeightFfn;
        use larql_inference::test_utils::make_test_weights;
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut engine = UnlimitedContextEngine::new(512);
        engine.prefill(&weights, &ffn, &[0u32]).expect("prefill");
        let h = engine.decode_step(&weights, &ffn, 1).expect("decode_step");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(h.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn window_auto_closes_when_full() {
        use larql_inference::test_utils::make_test_weights;
        let weights = make_test_weights();
        let window_size = 3usize;
        let mut engine = UnlimitedContextEngine::new(window_size);

        // Feed exactly window_size tokens → triggers close
        for tok in 0..window_size as u32 {
            engine.process(&weights, &[tok]).expect("process failed");
        }
        assert_eq!(engine.archive.len(), 1, "one window should be archived");
        assert_eq!(
            engine.current_window_tokens.len(),
            0,
            "current window should be empty"
        );
        assert_eq!(
            engine.checkpoints.len(),
            1,
            "one checkpoint should be saved"
        );
    }

    #[test]
    fn two_full_windows_archives_two() {
        use larql_inference::test_utils::make_test_weights;
        let weights = make_test_weights();
        let mut engine = UnlimitedContextEngine::new(2);

        // 4 tokens = 2 complete windows
        for tok in 0u32..4 {
            engine.process(&weights, &[tok]).expect("process");
        }
        assert_eq!(engine.archive.len(), 2);
        assert_eq!(engine.checkpoints.len(), 2);
    }

    #[test]
    fn partial_window_after_process() {
        use larql_inference::test_utils::make_test_weights;
        let weights = make_test_weights();
        let mut engine = UnlimitedContextEngine::new(4);

        // 3 tokens < window_size=4 → no close
        engine.process(&weights, &[0u32, 1, 2]).expect("process");
        assert_eq!(engine.archive.len(), 0, "no window closed yet");
        assert_eq!(engine.window_tokens(), 3);
    }

    #[test]
    fn flush_closes_partial_window() {
        use larql_inference::test_utils::make_test_weights;
        let weights = make_test_weights();
        let mut engine = UnlimitedContextEngine::new(4);
        engine.process(&weights, &[0u32, 1]).expect("process");
        assert_eq!(engine.archive.len(), 0);
        engine.flush();
        assert_eq!(engine.archive.len(), 1, "flush should close partial window");
    }

    #[test]
    fn cold_bytes_grow_after_window_close() {
        use larql_inference::test_utils::make_test_weights;
        let weights = make_test_weights();
        let mut engine = UnlimitedContextEngine::new(2);
        assert_eq!(engine.cold_bytes(), 0);
        engine.process(&weights, &[0u32, 1]).expect("process"); // closes window
        assert!(
            engine.cold_bytes() > 0,
            "cold tier should grow after window close"
        );
    }

    #[test]
    fn memory_bytes_nonzero_after_prefill() {
        use larql_inference::ffn::WeightFfn;
        use larql_inference::test_utils::make_test_weights;
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut engine = UnlimitedContextEngine::new(512);
        assert_eq!(engine.memory_bytes(), 0);
        engine
            .prefill(&weights, &ffn, &[0u32, 1, 2])
            .expect("prefill");
        assert!(engine.memory_bytes() > 0);
    }

    #[test]
    fn logits_from_unlimited_context_are_finite() {
        use larql_inference::ffn::WeightFfn;
        use larql_inference::forward::hidden_to_raw_logits;
        use larql_inference::test_utils::make_test_weights;
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut engine = UnlimitedContextEngine::new(512);
        let h = engine.prefill(&weights, &ffn, &[0u32, 1]).expect("prefill");
        let logits = hidden_to_raw_logits(&weights, &h);
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "logits should be finite"
        );
    }

    // ── Q4K paths via Q4K fixture ─────────────────────────────────────────
    //
    // `prefill_quant` first tries `fused_prefill` (Metal fast path); on
    // CPU that returns None (no fused decode kernel), so we fall through
    // to the dequant + cached-decode path. The Q4K fixture has the attn
    // Q4K slices the dequant step needs.

    #[test]
    fn prefill_quant_cpu_runs_via_dequant_path() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
        let mut weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let ffn = NullFfn;
        let mut engine = UnlimitedContextEngine::new(512);
        let h = engine
            .prefill_quant(&mut weights, &ffn, &index, &[0u32, 1, 2], &*backend)
            .expect("prefill_quant Q4K cpu fallback");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
    }

    #[test]
    fn decode_step_quant_cpu_extends_state() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
        let mut weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let ffn = NullFfn;
        let mut engine = UnlimitedContextEngine::new(512);
        engine
            .prefill_quant(&mut weights, &ffn, &index, &[0u32, 1], &*backend)
            .expect("prefill_quant");
        let h = engine
            .decode_step_quant(&mut weights, &ffn, &index, 2, &*backend)
            .expect("decode_step_quant Q4K cpu fallback");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
    }

    #[test]
    fn decode_step_quant_without_prefill_returns_none() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
        let mut weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let ffn = NullFfn;
        let mut engine = UnlimitedContextEngine::new(512);
        // No prefill → decode falls through fast-path checks and returns None
        // (or some empty hidden) without panicking.
        let _ = engine.decode_step_quant(&mut weights, &ffn, &index, 0, &*backend);
    }

    // ── Public utility methods (stats, replay_window, summary) ────────────

    #[test]
    fn engine_stats_summary_includes_archived_and_compression() {
        use larql_inference::ffn::WeightFfn;
        use larql_inference::test_utils::make_test_weights;
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut engine = UnlimitedContextEngine::new(512);
        engine
            .prefill(&weights, &ffn, &[0u32, 1, 2])
            .expect("prefill");
        let stats = engine.stats(&weights);
        assert!(stats.total_tokens >= 3);
        // EngineStats::summary builds a one-line string that includes
        // window count and token count.
        let s = stats.summary();
        assert!(s.contains("windows"));
        assert!(s.contains("tokens"));
    }

    #[test]
    fn engine_stats_with_empty_engine_handles_zero_division() {
        let weights = larql_inference::test_utils::make_test_weights();
        let engine = UnlimitedContextEngine::new(512);
        let stats = engine.stats(&weights);
        // No prefill → all counters zero, compression ratio short-circuits
        // to 0.0 (no division by zero).
        assert_eq!(stats.total_tokens, 0);
        assert_eq!(stats.archived_windows, 0);
        assert!(
            stats.compression_ratio == 0.0,
            "compression should be 0 when no boundary bytes archived"
        );
        // Summary still produces a string for the empty case.
        let _ = stats.summary();
    }

    #[test]
    fn replay_window_returns_none_for_missing_window() {
        let weights = larql_inference::test_utils::make_test_weights();
        let engine = UnlimitedContextEngine::new(512);
        // No windows archived → any window_id returns None at the
        // `self.archive.retrieve(window_id)?` line.
        assert!(engine.replay_window(&weights, 0).is_none());
        assert!(engine.replay_window(&weights, 99).is_none());
    }

    #[test]
    fn replay_window_succeeds_after_window_overflow() {
        use larql_inference::ffn::WeightFfn;
        use larql_inference::test_utils::make_test_weights;
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        // window=2; prefill 4 tokens → archives at least 1 window.
        let mut engine = UnlimitedContextEngine::new(2);
        engine
            .prefill(&weights, &ffn, &[0u32, 1, 2, 3])
            .expect("prefill 4 tokens");
        let stats = engine.stats(&weights);
        assert!(
            stats.archived_windows >= 1,
            "expected at least 1 archived window after overflow, got {}",
            stats.archived_windows
        );
        // Replay the first archived window — exercises the
        // `rs_extend_from_checkpoint_backend` path (lines 132-138).
        let replay = engine.replay_window(&weights, 0);
        assert!(replay.is_some(), "replay_window(0) should succeed");
        let (kv, abs_end) = replay.unwrap();
        assert!(!kv.is_empty(), "replayed K/V cache should be non-empty");
        assert!(
            abs_end < 4,
            "abs_end {abs_end} should be within the prefill"
        );
    }

    // ── Phase 2: executor-driven path ─────────────────────────────────────

    #[test]
    fn prefill_quant_via_executor_runs_through_local_walk() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::layer_executor::LocalWalkExecutor;
        use larql_inference::test_utils::make_test_weights;
        let mut weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let executor = LocalWalkExecutor::new(&*backend);
        let ffn = NullFfn;
        let mut engine = UnlimitedContextEngine::new(512);
        let h = engine
            .prefill_quant_via_executor(&mut weights, &executor, &ffn, &index, &[0u32, 1, 2])
            .expect("executor prefill");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(engine.memory_bytes() > 0);
    }

    #[test]
    fn decode_step_quant_via_executor_extends_state() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::layer_executor::LocalWalkExecutor;
        use larql_inference::test_utils::make_test_weights;
        let mut weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let executor = LocalWalkExecutor::new(&*backend);
        let ffn = NullFfn;
        let mut engine = UnlimitedContextEngine::new(512);
        engine
            .prefill_quant_via_executor(&mut weights, &executor, &ffn, &index, &[0u32, 1])
            .expect("prefill");
        let h = engine
            .decode_step_quant_via_executor(&mut weights, &executor, &ffn, &index, 2)
            .expect("decode");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
    }

    /// Counting FFN that records every `forward` call. Proves the executor
    /// path actually dispatches through the caller's `FfnBackend` instead
    /// of constructing a local `WalkFfn` (the legacy coupling the migration
    /// removes).
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
        use larql_inference::layer_executor::LocalWalkExecutor;
        use larql_inference::test_utils::make_test_weights;
        let mut weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let executor = LocalWalkExecutor::new(&*backend);

        let ffn = CountingFfn {
            calls: std::sync::atomic::AtomicUsize::new(0),
            hidden: weights.hidden_size,
        };
        let mut engine = UnlimitedContextEngine::new(512);
        engine
            .prefill_quant_via_executor(&mut weights, &executor, &ffn, &index, &[0u32, 1, 2])
            .expect("prefill via executor");

        let call_count = ffn.calls.load(std::sync::atomic::Ordering::SeqCst);
        // 3 tokens × num_layers — one FFN dispatch per (token, layer)
        // because the engine's per-token loop runs every layer through
        // `run_decode_layer`, which in turn invokes the caller's FFN.
        let expected = 3 * weights.num_layers;
        assert_eq!(
            call_count, expected,
            "executor path should dispatch FFN through the supplied backend \
             once per (token, layer); got {call_count} for {expected} \
             expected — engine is likely constructing its own FFN internally",
        );
    }

    #[test]
    fn prefill_quant_via_executor_with_small_window_archives() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::layer_executor::LocalWalkExecutor;
        use larql_inference::test_utils::make_test_weights;
        let mut weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let executor = LocalWalkExecutor::new(&*backend);
        let ffn = NullFfn;
        // window=2, 4 tokens → triggers two window-close cycles via
        // `process_via_executor`. Exercises the prior-checkpoint-load
        // branch in `extend_current_via_executor`.
        let mut engine = UnlimitedContextEngine::new(2);
        engine
            .prefill_quant_via_executor(&mut weights, &executor, &ffn, &index, &[0u32, 1, 2, 3])
            .expect("prefill 4 tokens through executor");
        let stats = engine.stats(&weights);
        assert!(
            stats.archived_windows >= 1,
            "expected at least 1 archived window, got {}",
            stats.archived_windows
        );
    }
}
