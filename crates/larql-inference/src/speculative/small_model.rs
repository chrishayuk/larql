//! Off-the-shelf small-model `Drafter` implementation.
//!
//! Wraps a separately-loaded LARQL vindex (e.g. Gemma 270M) and runs
//! its forward pass to propose draft tokens. The drafter and target
//! are independent processes — they share only the tokenizer
//! contract (compatible vocab IDs).
//!
//! ## Design
//!
//! - **Token history is the source of truth.** The drafter maintains
//!   its own `Vec<TokenId>` representing the accepted sequence so far.
//!   Each `propose(n)` call runs the small model forward `n` times,
//!   appending each greedy top-1 prediction to a simulated continuation.
//! - **No shared state with target.** The drafter doesn't see the
//!   target's hidden state or KV cache. It infers context purely from
//!   the accepted token IDs (via `accept()`).
//! - **Greedy proposals.** Top-1 with the model's reported probability.
//!   For higher acceptance rates a future variant can sample with
//!   temperature, but greedy is the simplest correct first cut.
//!
//! ## Incremental decode (phase 4c follow-up)
//!
//! The drafter holds its own [`ComputeBackend`] (a separate instance
//! from the target's, so KV caches don't collide). On the first
//! `propose()` after `seed_history`, the drafter prefills the prompt
//! through `prefill_q4`. On subsequent calls (after `accept()`), only
//! the newly-accepted tokens are pushed through `decode_token` to
//! advance the cache. Drafting itself is a small N-step `decode_token`
//! loop with rollback at the end (drafted-but-not-accepted tokens never
//! commit to the cache).
//!
//! Falls back to the original from-scratch `predict_q4k` path when the
//! backend lacks KV cache support (CPU-only builds) or when any cache
//! sync step fails. This keeps the legacy correctness gate available
//! and is the reason `weights`, `tokenizer`, `index` are still owned.

use std::path::{Path, PathBuf};

use larql_compute::{default_backend, ComputeBackend, QuantFormat};
use larql_models::ModelWeights;
use larql_vindex::VectorIndex;
use ndarray::Array2;
use tokenizers::Tokenizer;

use larql_vindex::GateIndex;

use crate::error::InferenceError;
use crate::forward::{apply_norm, embed_tokens_pub};
use crate::layer_graph::generate::lm_head_topk;
use crate::layer_graph::pipeline_layer::{
    attention_geometry_for_arch_layer, build_pipeline_layers, kv_cache_shapes_for_arch,
    DEFAULT_GPU_KV_CACHE_MAX_SEQ,
};
use crate::tokenizer::load_tokenizer;
use crate::vindex::open_inference_vindex;

use super::{DraftToken, Drafter, TokenId};

/// Top-K hits the drafter's lm_head emits per drafted token. Drafter
/// is greedy top-1 in practice (drafts[0]), but the topk-with-softmax
/// distribution gives `p_draft` a meaningful denominator for the
/// verifier's accept ratio. Matches the bench's greedy lm_head_topk
/// usage in `gpu.rs`.
const DRAFTER_LM_HEAD_TOPK: usize = 5;

/// Off-the-shelf draft model loaded from a LARQL vindex directory.
///
/// `path` should point at a directory produced by `larql extract`
/// (containing `tokenizer.json`, `weight_manifest.json`, and the
/// quantized weight blobs). The vindex is loaded eagerly on
/// construction; subsequent `propose()` calls reuse the cached
/// state.
pub struct SmallModelDrafter {
    path: PathBuf,
    weights: ModelWeights,
    tokenizer: Tokenizer,
    index: VectorIndex,
    history: Vec<TokenId>,
    /// Drafter-owned backend, separate instance from the target's.
    /// Each `default_backend()` call returns a fresh CudaBackend (or
    /// MetalBackend / CpuBackend) with its own KV cache slab — so the
    /// drafter writes to its OWN cache, not the target's.
    backend: Box<dyn ComputeBackend>,
    /// Tokens currently committed to `backend`'s KV cache. Invariant:
    /// `cache_len <= history.len()`. The gap is filled by `sync_cache`
    /// at the top of every `propose()`.
    cache_len: usize,
    /// Hidden state at position `cache_len - 1` (the last cached
    /// token). Used to sample the FIRST drafted token without
    /// re-decoding. `None` when the cache is empty or sync failed.
    last_hidden: Option<Vec<f32>>,
    /// Set at construction to record whether the chosen backend
    /// supports KV-cached decode. False forces the legacy
    /// from-scratch fallback (CPU builds, etc.).
    backend_has_cache: bool,
}

impl SmallModelDrafter {
    /// Load a small-model drafter from a LARQL vindex directory.
    /// Uses the same Q4_K vindex loader as `larql bench` (peak heap
    /// ~6 GB for 31B vs ~127 GB for the float path) — see
    /// `larql_vindex::load_model_weights_q4k`.
    pub fn from_vindex(path: impl AsRef<Path>) -> Result<Self, InferenceError> {
        let path = path.as_ref().to_path_buf();
        let mut callbacks = larql_vindex::SilentLoadCallbacks;
        let weights = larql_vindex::load_model_weights_q4k(&path, &mut callbacks)
            .map_err(InferenceError::Vindex)?;
        let tokenizer = load_tokenizer(&path).map_err(|e| {
            InferenceError::Parse(format!("SmallModelDrafter load tokenizer: {e:?}"))
        })?;
        let index = open_inference_vindex(&path)?;
        let backend = default_backend();
        let backend_has_cache = backend.has_kv_cache();
        Ok(Self {
            path,
            weights,
            tokenizer,
            index,
            history: Vec::new(),
            backend,
            cache_len: 0,
            last_hidden: None,
            backend_has_cache,
        })
    }

    /// Path the drafter was loaded from. Useful for diagnostics.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Current token history length. Drafter-side cache_len equivalent.
    pub fn history_len(&self) -> usize {
        self.history.len()
    }

    /// Seed the history with the prompt's tokens. Caller invokes this
    /// once after construction with the same tokens fed to the target's
    /// prefill.
    pub fn seed_history(&mut self, prompt_tokens: &[TokenId]) {
        // Fast path: the v3 dispatch helper calls `seed_history` at
        // the top of every iter with the canonical history (= prompt
        // + accepted span so far). Each iter's history is a strict
        // prefix-extension of the previous (the loop never undoes
        // tokens). Detecting that lets us KEEP the existing KV cache
        // and just APPEND the new tokens — the next propose's
        // `sync_cache` will incrementally decode them. Without this,
        // every iter re-prefills the entire history from scratch and
        // the drafter's KV cache buys nothing.
        if prompt_tokens.len() >= self.history.len()
            && prompt_tokens[..self.history.len()] == self.history[..]
        {
            self.history
                .extend_from_slice(&prompt_tokens[self.history.len()..]);
            return;
        }
        // Divergence (caller restarted with a different prompt) —
        // reset cache and re-prefill from scratch.
        self.history.clear();
        self.history.extend_from_slice(prompt_tokens);
        self.cache_len = 0;
        self.last_hidden = None;
        self.backend.reset_kv_cache();
    }

    /// Compute softmax probability for the picked-top1 within the
    /// drafter's lm_head topk hits. Mirrors what the bench's greedy
    /// path does for `tokens.push((tok_str, prob))` — gives the
    /// verifier a meaningful `p_draft` denominator.
    fn softmax_top1(hits: &[(u32, f32)]) -> Option<(u32, f32)> {
        if hits.is_empty() {
            return None;
        }
        let max_score = hits
            .iter()
            .map(|&(_, s)| s)
            .fold(f32::NEG_INFINITY, f32::max);
        if !max_score.is_finite() {
            return None;
        }
        let mut sum = 0.0_f64;
        let mut top1_exp = 0.0_f64;
        for (i, &(_, s)) in hits.iter().enumerate() {
            let e = ((s - max_score) as f64).exp();
            sum += e;
            if i == 0 {
                top1_exp = e;
            }
        }
        if sum <= 0.0 {
            return None;
        }
        let p = (top1_exp / sum) as f32;
        Some((hits[0].0, p.clamp(f32::MIN_POSITIVE, 1.0)))
    }

    /// Bring `self.backend`'s KV cache up to date with `self.history`,
    /// capturing `self.last_hidden` from the last decoded position.
    /// First-call path uses bulk `prefill_q4`; subsequent calls
    /// incrementally `decode_token` the newly-accepted tokens.
    /// Returns true on success; false on any error (caller falls back
    /// to the legacy `predict_q4k` path).
    fn sync_cache(&mut self) -> bool {
        if !self.backend_has_cache || self.history.is_empty() {
            return false;
        }
        if self.cache_len == self.history.len() {
            return self.last_hidden.is_some();
        }

        let hidden = self.weights.hidden_size;
        let attn = attention_geometry_for_arch_layer(&self.weights, 0);
        let intermediate = self.index.num_features(0);
        if intermediate == 0 {
            return false;
        }
        let gate_index: &dyn GateIndex = &self.index;
        let (q4_ffn_mmap, ffn_is_q4k) = match gate_index.interleaved_q4k_mmap_ref() {
            Some(m) => (m, true),
            None => match gate_index.interleaved_q4_mmap_ref() {
                Some(m) => (m, false),
                None => return false,
            },
        };
        let ffn_format = if ffn_is_q4k {
            QuantFormat::Q4_K
        } else {
            QuantFormat::Q4_0
        };
        let q4_ffn_per_matrix = match ffn_format.packed_matrix_bytes(intermediate, hidden) {
            Some(n) => n,
            None => return false,
        };
        let num_layers = self.weights.num_layers;
        let layers = build_pipeline_layers(
            &self.weights,
            &self.index,
            0..num_layers,
            q4_ffn_mmap,
            q4_ffn_per_matrix,
            ffn_format,
        );

        let new_tokens: Vec<TokenId> = self.history[self.cache_len..].to_vec();
        if self.cache_len == 0 {
            // Initial prefill of the full history.
            self.backend.reset_kv_cache();
            let kv_shapes = kv_cache_shapes_for_arch(&self.weights);
            self.backend
                .preallocate_kv_cache_per_layer(&kv_shapes, DEFAULT_GPU_KV_CACHE_MAX_SEQ);

            let h_embed = embed_tokens_pub(&self.weights, &new_tokens);
            let x_flat: Vec<f32> = match h_embed.as_slice() {
                Some(s) => s.to_vec(),
                None => return false,
            };
            let h_out = match self.backend.prefill_q4(
                &layers,
                &x_flat,
                hidden,
                intermediate,
                attn.q_dim,
                attn.kv_dim,
                new_tokens.len(),
                attn.num_q_heads,
                attn.num_kv_heads,
                attn.head_dim,
                attn.rope_base,
                false,
                0.0,
            ) {
                Some(h) => h,
                None => return false,
            };
            if h_out.len() < new_tokens.len() * hidden {
                return false;
            }
            let last_pos = new_tokens.len() - 1;
            let last_h = h_out[last_pos * hidden..(last_pos + 1) * hidden].to_vec();
            self.cache_len = self.history.len();
            self.last_hidden = Some(last_h);
            true
        } else {
            // Incremental: decode the new tokens one at a time.
            let mut last_h: Option<Vec<f32>> = None;
            for &tok in &new_tokens {
                let h_embed = embed_tokens_pub(&self.weights, &[tok]);
                let x: Vec<f32> = h_embed.row(0).to_vec();
                match self.backend.decode_token(
                    &layers,
                    &x,
                    hidden,
                    intermediate,
                    attn.q_dim,
                    attn.kv_dim,
                    attn.num_q_heads,
                    attn.num_kv_heads,
                    attn.head_dim,
                    attn.rope_base,
                ) {
                    Some(h) => last_h = Some(h),
                    None => return false,
                }
            }
            self.cache_len = self.history.len();
            self.last_hidden = last_h;
            true
        }
    }

    /// Propose `n` drafts using the cached KV state. Drafts are
    /// written to the cache temporarily and rolled back at the end —
    /// the cache returns to `cache_len` (the accepted-history
    /// watermark) on every return. Returns `None` on any failure;
    /// caller falls back to the legacy `predict_q4k` path.
    fn propose_incremental(&mut self, n: usize) -> Option<Vec<DraftToken>> {
        if !self.sync_cache() {
            return None;
        }
        let mut h_for_sample = self.last_hidden.clone()?;

        let hidden = self.weights.hidden_size;
        let attn = attention_geometry_for_arch_layer(&self.weights, 0);
        let intermediate = self.index.num_features(0);
        if intermediate == 0 {
            return None;
        }
        let gate_index: &dyn GateIndex = &self.index;
        let (q4_ffn_mmap, ffn_is_q4k) = match gate_index.interleaved_q4k_mmap_ref() {
            Some(m) => (m, true),
            None => (gate_index.interleaved_q4_mmap_ref()?, false),
        };
        let ffn_format = if ffn_is_q4k {
            QuantFormat::Q4_K
        } else {
            QuantFormat::Q4_0
        };
        let q4_ffn_per_matrix = ffn_format.packed_matrix_bytes(intermediate, hidden)?;
        let num_layers = self.weights.num_layers;
        let layers = build_pipeline_layers(
            &self.weights,
            &self.index,
            0..num_layers,
            q4_ffn_mmap,
            q4_ffn_per_matrix,
            ffn_format,
        );

        let pre_len = self.cache_len;
        let mut drafts = Vec::with_capacity(n);
        let norm_offset = self.weights.arch.norm_weight_offset();
        let final_norm_key = self.weights.arch.final_norm_key();

        for _ in 0..n {
            // Sample top-1 from h_for_sample (greedy + softmax for p_draft).
            let h_arr = Array2::from_shape_vec((1, hidden), h_for_sample.clone()).ok()?;
            let h_final = apply_norm(&self.weights, &h_arr, final_norm_key, norm_offset);
            let h_1d = h_final.row(0).to_owned();
            let hits = lm_head_topk(
                &self.index,
                &self.weights,
                &h_1d,
                DRAFTER_LM_HEAD_TOPK,
                &*self.backend,
            );
            let (id, p) = match Self::softmax_top1(&hits) {
                Some(p) => p,
                None => break,
            };
            drafts.push(DraftToken { id, p_draft: p });

            // Decode the drafted token to extend cache + get next hidden.
            let h_embed = embed_tokens_pub(&self.weights, &[id]);
            let x: Vec<f32> = h_embed.row(0).to_vec();
            let new_h = self.backend.decode_token(
                &layers,
                &x,
                hidden,
                intermediate,
                attn.q_dim,
                attn.kv_dim,
                attn.num_q_heads,
                attn.num_kv_heads,
                attn.head_dim,
                attn.rope_base,
            )?;
            h_for_sample = new_h;
        }

        // Drafts are not committed: rollback cache to the accepted
        // watermark. accept() will catch the cache up next time.
        self.backend.truncate_kv_cache(pre_len);
        Some(drafts)
    }
}

impl Drafter for SmallModelDrafter {
    fn propose(&mut self, _h_target: &[f32], n: usize) -> Vec<DraftToken> {
        if n == 0 {
            return Vec::new();
        }
        // Prefer the KV-cached incremental path. Fall back to the
        // legacy from-scratch `predict_q4k` loop if the backend
        // lacks KV support or if any cache step fails.
        if let Some(drafts) = self.propose_incremental(n) {
            return drafts;
        }
        let mut drafts = Vec::with_capacity(n);
        let mut sim_history = self.history.clone();
        for _ in 0..n {
            let result = crate::vindex::predict_q4k(
                &mut self.weights,
                &self.tokenizer,
                &sim_history,
                1,
                &self.index,
            );
            let id = match result.token_ids.first() {
                Some(&id) => id,
                None => break,
            };
            // Use the model's reported probability for the picked token.
            // Falls back to 1.0 only if the predictions vector is empty
            // (shouldn't happen for top_k=1 with a non-empty vocab).
            let p = result
                .predictions
                .first()
                .map(|(_, p)| *p as f32)
                .unwrap_or(1.0)
                .clamp(f32::MIN_POSITIVE, 1.0);
            drafts.push(DraftToken { id, p_draft: p });
            sim_history.push(id);
        }
        drafts
    }

    fn reset(&mut self) {
        self.history.clear();
        self.cache_len = 0;
        self.last_hidden = None;
        self.backend.reset_kv_cache();
    }

    fn accept(&mut self, accepted: &[TokenId]) {
        self.history.extend_from_slice(accepted);
        // Cache stays at the old `cache_len`; the next propose() will
        // sync forward by `accepted.len()` decode_token calls.
    }

    fn seed_history(&mut self, tokens: &[TokenId]) {
        // Delegate to the inherent method (the impl that knows about
        // prefix-extension + KV cache reset).
        SmallModelDrafter::seed_history(self, tokens);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    fn drafter_or_skip() -> Option<SmallModelDrafter> {
        let path = env::var("LARQL_DRAFT_MODEL_VINDEX").ok()?;
        SmallModelDrafter::from_vindex(&path).ok()
    }

    #[test]
    fn loads_from_env_pointed_vindex_or_skips() {
        // Gated on LARQL_DRAFT_MODEL_VINDEX so CI without a model just
        // no-ops. Local validation: set the env var to any vindex dir
        // (target model works as a smoke-test drafter — α=100% trivially).
        let Some(drafter) = drafter_or_skip() else {
            return;
        };
        assert!(drafter.history_len() == 0);
        assert!(drafter.path().exists());
    }

    #[test]
    fn seed_history_then_propose_returns_n_drafts() {
        let Some(mut drafter) = drafter_or_skip() else {
            return;
        };
        // Seed with a small prompt so propose has context to start from.
        drafter.seed_history(&[2, 100, 200]); // arbitrary token IDs
        let drafts = drafter.propose(&[], 3);
        assert!(
            drafts.len() <= 3,
            "drafter must return at most n drafts; got {}",
            drafts.len()
        );
        for d in &drafts {
            assert!(
                d.p_draft > 0.0 && d.p_draft <= 1.0,
                "p_draft out of range: {}",
                d.p_draft
            );
        }
    }

    #[test]
    fn accept_advances_history() {
        let Some(mut drafter) = drafter_or_skip() else {
            return;
        };
        drafter.seed_history(&[1, 2, 3]);
        assert_eq!(drafter.history_len(), 3);
        drafter.accept(&[4, 5]);
        assert_eq!(drafter.history_len(), 5);
    }

    #[test]
    fn reset_clears_history() {
        let Some(mut drafter) = drafter_or_skip() else {
            return;
        };
        drafter.seed_history(&[1, 2, 3]);
        drafter.reset();
        assert_eq!(drafter.history_len(), 0);
    }

    #[test]
    fn propose_zero_returns_empty() {
        let Some(mut drafter) = drafter_or_skip() else {
            return;
        };
        let drafts = drafter.propose(&[], 0);
        assert!(drafts.is_empty());
    }
}
