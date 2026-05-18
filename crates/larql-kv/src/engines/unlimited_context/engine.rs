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

use larql_compute::{cpu_backend, ComputeBackend};
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
use larql_inference::layer_graph::pipeline_layer::DEFAULT_GPU_KV_CACHE_MAX_SEQ;
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
    backend: Box<dyn ComputeBackend>,
}

impl UnlimitedContextEngine {
    pub fn new(window_size: usize) -> Self {
        Self::with_backend(window_size, cpu_backend())
    }

    pub fn with_backend(window_size: usize, backend: Box<dyn ComputeBackend>) -> Self {
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
            &prior,
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
        let out = rs_extend_from_checkpoint_q4k(weights, index, chunk, &prior, abs_start, backend)?;

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
            &prior,
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

    fn prefill(&mut self, weights: &ModelWeights, token_ids: &[u32]) -> Option<Array2<f32>> {
        self.process(weights, token_ids)?;
        self.last_hidden.clone()
    }

    fn decode_step(&mut self, weights: &ModelWeights, token_id: u32) -> Option<Array2<f32>> {
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

    /// Q4K prefill — uses Metal `prefill_q4` when available (full GPU pipeline).
    ///
    /// Falls back to the CPU `process()` path when the backend does not support
    /// the fused Q4 pipeline. The Metal path runs at ~75 tok/s on Gemma 3 4B
    /// (same as `larql bench`) because it submits all 34 layers in one command
    /// buffer rather than per-layer CPU dispatch.
    fn prefill_q4k(
        &mut self,
        weights: &mut ModelWeights,
        index: &VectorIndex,
        token_ids: &[u32],
        backend: &dyn ComputeBackend,
    ) -> Option<Array2<f32>> {
        // Try Metal full pipeline. Returns None for CpuBackend — fall through.
        if let Some(h) = q4k_prefill_metal(weights, index, token_ids, backend) {
            self.abs_offset = token_ids.len();
            self.last_hidden = Some(h.clone());
            return Some(h);
        }
        // CPU Q4K path: dequantise attention tensors, use WalkFfn for FFN.
        ensure_attn_tensors_dequantised(weights, index);
        self.process_q4k(weights, index, token_ids, backend)?;
        self.last_hidden.clone()
    }

    fn decode_step_q4k(
        &mut self,
        weights: &mut ModelWeights,
        index: &VectorIndex,
        token_id: u32,
        backend: &dyn ComputeBackend,
    ) -> Option<Array2<f32>> {
        // Try Metal decode_token. Returns None for CpuBackend — fall through.
        if let Some(h) = q4k_decode_token(weights, index, token_id, backend) {
            self.abs_offset += 1;
            self.last_hidden = Some(h.clone());
            return Some(h);
        }
        // CPU Q4K path.
        ensure_attn_tensors_dequantised(weights, index);
        self.process_q4k(weights, index, &[token_id], backend)?;
        self.last_hidden.clone()
    }
}

// ─── Q4K / Metal helper fns ───────────────────────────────────────────────────

/// Run GPU prefill via `backend.prefill_q4` using Q4K pipeline layers built
/// from `index`. Returns the last-token hidden state on success.
pub(crate) fn q4k_prefill_metal(
    weights: &ModelWeights,
    index: &VectorIndex,
    token_ids: &[u32],
    backend: &dyn ComputeBackend,
) -> Option<Array2<f32>> {
    use larql_inference::layer_graph::pipeline_layer::build_pipeline_layers;
    use larql_vindex::GateIndex;

    if !backend.has_q4() {
        return None;
    }

    let gate_index: &dyn GateIndex = index;
    let (q4_ffn_mmap, ffn_is_q4k) = if let Some(m) = gate_index.interleaved_q4k_mmap_ref() {
        (m, true)
    } else {
        let m = gate_index.interleaved_q4_mmap_ref()?;
        (m, false)
    };
    index.attn_q4k_layer_data(0)?;

    let arch = &*weights.arch;
    let hidden = weights.hidden_size;
    let num_layers = weights.num_layers;
    let intermediate = gate_index.num_features(0);
    if intermediate == 0 {
        return None;
    }

    let ffn_format = if ffn_is_q4k {
        larql_compute::QuantFormat::Q4_K
    } else {
        larql_compute::QuantFormat::Q4_0
    };
    let q4_ffn_per_matrix = ffn_format.packed_matrix_bytes(intermediate, hidden)?;

    let layers = build_pipeline_layers(
        weights,
        index,
        0..num_layers,
        q4_ffn_mmap,
        q4_ffn_per_matrix,
        ffn_format,
    );

    let h_embed = larql_inference::forward::embed_tokens_pub(weights, token_ids);
    let x: Vec<f32> = h_embed.as_slice().unwrap_or(&[]).to_vec();

    let seq_len = token_ids.len();
    let softcap = arch.attn_logit_softcapping().unwrap_or(0.0);
    let qk_norm = arch.attn_q_norm_key(0).is_some();

    backend.reset_kv_cache();
    {
        let kv_shapes: Vec<(usize, usize)> = (0..num_layers)
            .map(|l| (arch.num_kv_heads_for_layer(l), arch.head_dim_for_layer(l)))
            .collect();
        backend.preallocate_kv_cache_per_layer(&kv_shapes, DEFAULT_GPU_KV_CACHE_MAX_SEQ);
    }

    let h_vec = backend.prefill_q4(&layers, &x, hidden, intermediate, seq_len, qk_norm, softcap)?;

    // Return pre-final_norm hidden state — the caller (hidden_to_raw_logits) applies it.
    let h_2d = Array2::from_shape_vec((seq_len, hidden), h_vec).ok()?;
    let last = h_2d.shape()[0] - 1;
    Some(h_2d.slice(ndarray::s![last..=last, ..]).to_owned())
}

/// Run one Metal decode step via `backend.decode_token`.
pub(crate) fn q4k_decode_token(
    weights: &ModelWeights,
    index: &VectorIndex,
    token_id: u32,
    backend: &dyn ComputeBackend,
) -> Option<Array2<f32>> {
    use larql_inference::layer_graph::pipeline_layer::build_pipeline_layers;
    use larql_vindex::GateIndex;

    let gate_index: &dyn GateIndex = index;
    let (q4_ffn_mmap, ffn_is_q4k) = if let Some(m) = gate_index.interleaved_q4k_mmap_ref() {
        (m, true)
    } else {
        let m = gate_index.interleaved_q4_mmap_ref()?;
        (m, false)
    };

    let hidden = weights.hidden_size;
    let num_layers = weights.num_layers;
    let intermediate = gate_index.num_features(0);

    let ffn_format = if ffn_is_q4k {
        larql_compute::QuantFormat::Q4_K
    } else {
        larql_compute::QuantFormat::Q4_0
    };
    let q4_ffn_per_matrix = ffn_format.packed_matrix_bytes(intermediate, hidden)?;

    let layers = build_pipeline_layers(
        weights,
        index,
        0..num_layers,
        q4_ffn_mmap,
        q4_ffn_per_matrix,
        ffn_format,
    );

    let h_tok = larql_inference::forward::embed_tokens_pub(weights, &[token_id]);
    let x_dec: Vec<f32> = h_tok.row(0).to_vec();

    let h_vec = backend.decode_token(&layers, &x_dec, hidden, intermediate)?;

    // Return pre-final_norm hidden state — the caller (hidden_to_raw_logits) applies it.
    Array2::from_shape_vec((1, hidden), h_vec).ok()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(all(target_arch = "wasm32", feature = "browser-tests"))]
    wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn new_engine_is_empty() {
        let eng = UnlimitedContextEngine::new(512);
        assert_eq!(eng.window_size, 512);
        assert_eq!(eng.archive.len(), 0);
        assert_eq!(eng.checkpoints.len(), 0);
        assert_eq!(eng.current_window_id, 0);
        assert_eq!(eng.memory_bytes(), 0);
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn engine_info_config_contains_window_size() {
        let eng = UnlimitedContextEngine::new(1024);
        assert!(eng.info().config.contains("1024"));
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn window_tokens_and_cold_bytes_start_zero() {
        let eng = UnlimitedContextEngine::new(512);
        assert_eq!(eng.window_tokens(), 0);
        assert_eq!(eng.cold_bytes(), 0);
    }

    // ── prefill / decode cycle ─────────────────────────────────────────────────

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn prefill_returns_hidden_state() {
        use larql_inference::test_utils::make_test_weights;
        let weights = make_test_weights();
        let mut engine = UnlimitedContextEngine::new(512);
        let h = engine
            .prefill(&weights, &[0u32, 1, 2])
            .expect("prefill failed");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(
            h.iter().all(|v| v.is_finite()),
            "hidden state should be finite"
        );
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn decode_step_returns_hidden_state() {
        use larql_inference::test_utils::make_test_weights;
        let weights = make_test_weights();
        let mut engine = UnlimitedContextEngine::new(512);
        engine.prefill(&weights, &[0u32]).expect("prefill");
        let h = engine.decode_step(&weights, 1).expect("decode_step");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(h.iter().all(|v| v.is_finite()));
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn partial_window_after_process() {
        use larql_inference::test_utils::make_test_weights;
        let weights = make_test_weights();
        let mut engine = UnlimitedContextEngine::new(4);

        // 3 tokens < window_size=4 → no close
        engine.process(&weights, &[0u32, 1, 2]).expect("process");
        assert_eq!(engine.archive.len(), 0, "no window closed yet");
        assert_eq!(engine.window_tokens(), 3);
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn flush_closes_partial_window() {
        use larql_inference::test_utils::make_test_weights;
        let weights = make_test_weights();
        let mut engine = UnlimitedContextEngine::new(4);
        engine.process(&weights, &[0u32, 1]).expect("process");
        assert_eq!(engine.archive.len(), 0);
        engine.flush();
        assert_eq!(engine.archive.len(), 1, "flush should close partial window");
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn memory_bytes_nonzero_after_prefill() {
        use larql_inference::test_utils::make_test_weights;
        let weights = make_test_weights();
        let mut engine = UnlimitedContextEngine::new(512);
        assert_eq!(engine.memory_bytes(), 0);
        engine.prefill(&weights, &[0u32, 1, 2]).expect("prefill");
        assert!(engine.memory_bytes() > 0);
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn logits_from_unlimited_context_are_finite() {
        use larql_inference::forward::hidden_to_raw_logits;
        use larql_inference::test_utils::make_test_weights;
        let weights = make_test_weights();
        let mut engine = UnlimitedContextEngine::new(512);
        let h = engine.prefill(&weights, &[0u32, 1]).expect("prefill");
        let logits = hidden_to_raw_logits(&weights, &h);
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "logits should be finite"
        );
    }
}
