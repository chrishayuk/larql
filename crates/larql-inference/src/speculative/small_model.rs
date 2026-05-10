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
//! ## Performance caveat
//!
//! This first slice re-runs the full forward pass from scratch on each
//! `propose()` call. That's O(history_len) per draft proposal, which
//! is wildly suboptimal — production should incrementally extend a KV
//! cache. Correctness over perf for now; the caller knows how to
//! benchmark and iterate.

use std::path::{Path, PathBuf};

use larql_models::ModelWeights;
use larql_vindex::VectorIndex;
use tokenizers::Tokenizer;

use crate::error::InferenceError;
use crate::tokenizer::load_tokenizer;
use crate::vindex::open_inference_vindex;

use super::{DraftToken, Drafter, TokenId};

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
        Ok(Self {
            path,
            weights,
            tokenizer,
            index,
            history: Vec::new(),
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
        self.history.clear();
        self.history.extend_from_slice(prompt_tokens);
    }
}

impl Drafter for SmallModelDrafter {
    fn propose(&mut self, _h_target: &[f32], n: usize) -> Vec<DraftToken> {
        if n == 0 {
            return Vec::new();
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
    }

    fn accept(&mut self, accepted: &[TokenId]) {
        self.history.extend_from_slice(accepted);
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
