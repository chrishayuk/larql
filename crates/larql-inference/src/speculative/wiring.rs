//! Phase 4b call-site wiring helper — the single function the
//! generate loop calls to opt into speculative decoding.
//!
//! ## Caller contract — KV cache management
//!
//! `run_naive_step` does NOT advance the canonical KV cache.
//! `target_forward_naive` runs the target model from scratch on each
//! tree node's ancestor chain (using its own internal per-call KV
//! cache), so the canonical cache held by `DecodeBackend` is unchanged
//! when this function returns.
//!
//! On `Some(tokens)`, the integrator MUST advance the canonical cache
//! by `tokens.len()` positions before the next decode call. The
//! simplest path is to call `backend.decode_token(...)` N times
//! sequentially with each emitted token as input — wasteful (N
//! redundant target forwards) but correct.
//!
//! Phase 4c eliminates this redundancy by having `target_forward`
//! itself write to the canonical cache via the batched
//! `cuda::attn_tree::tree_decode_attention` kernel. Until then, the
//! naive path is **3× slower than baseline** — strictly a parity
//! oracle for phase 4c's batched implementation.

use larql_models::ModelWeights;
use larql_vindex::VectorIndex;

use super::orchestrator::build_linear_tree;
use super::small_model::SmallModelDrafter;
use super::target_forward::target_forward_naive;
use super::verify::{verify_tree, VerifyRng};
use super::{Drafter, SpecConfig, TokenId};

/// One speculative step using the naive sequential `target_forward`.
/// Returns `Some(emitted_tokens)` on a successful step (caller MUST
/// advance KV cache by `tokens.len()`); `None` to signal fall-through
/// to the existing non-speculative path.
///
/// Returns `None` when:
/// - `LARQL_SPECULATIVE_DECODE` is unset / not `1`
/// - SWA window leaves no slack (`cfg.effective_depth(cache_len) == 0`)
/// - Drafter declines (returns empty proposals)
/// - `target_forward_naive` returns the wrong number of distributions
///   (defensive — should not happen but production retries non-spec)
///
/// `history` is the prompt + accepted span so far (the canonical
/// token sequence at the target's current position). `cache_len`
/// matches `history.len()` for sanity but the field is separate so
/// the integrator can clamp via `effective_depth` against the SWA
/// window.
pub fn run_naive_step(
    weights: &mut ModelWeights,
    drafter: &mut SmallModelDrafter,
    history: &[TokenId],
    cache_len: usize,
    cfg: SpecConfig,
    index: &VectorIndex,
    rng: &mut VerifyRng,
) -> Option<Vec<TokenId>> {
    if !super::enabled() {
        return None;
    }
    let depth = cfg.effective_depth(cache_len);
    if depth == 0 {
        return None;
    }
    let drafts = drafter.propose(&[], depth);
    if drafts.is_empty() {
        return None;
    }
    let tree = build_linear_tree(&drafts);
    let p_target = target_forward_naive(weights, history, &tree, index);
    if p_target.len() != tree.len() {
        return None;
    }
    let span = verify_tree(&tree, &p_target, rng);
    let emitted = span.tokens();
    if emitted.is_empty() {
        return None;
    }
    drafter.accept(&emitted);
    Some(emitted)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    fn vindex_path_or_skip() -> Option<std::path::PathBuf> {
        env::var("LARQL_FULL_VOCAB_PROBS_VINDEX")
            .ok()
            .map(std::path::PathBuf::from)
    }

    fn load(path: &std::path::Path) -> ModelWeights {
        let mut callbacks = larql_vindex::SilentLoadCallbacks;
        larql_vindex::load_model_weights_q4k(path, &mut callbacks).expect("load weights")
    }

    fn open_index(path: &std::path::Path) -> VectorIndex {
        crate::open_inference_vindex(path).expect("vindex")
    }

    #[test]
    fn returns_none_when_env_disabled() {
        let Some(path) = vindex_path_or_skip() else {
            return;
        };
        let mut weights = load(&path);
        let index = open_index(&path);
        let mut drafter = SmallModelDrafter::from_vindex(&path).expect("drafter");
        drafter.seed_history(&[2, 100, 200]);
        let mut rng = VerifyRng::new(0);

        unsafe {
            env::remove_var("LARQL_SPECULATIVE_DECODE");
        }
        let result = run_naive_step(
            &mut weights,
            &mut drafter,
            &[2, 100, 200],
            3,
            SpecConfig::default(),
            &index,
            &mut rng,
        );
        assert_eq!(result, None);
    }

    #[test]
    fn returns_some_tokens_when_env_enabled_and_drafter_proposes() {
        let Some(path) = vindex_path_or_skip() else {
            return;
        };
        let mut weights = load(&path);
        let index = open_index(&path);
        let mut drafter = SmallModelDrafter::from_vindex(&path).expect("drafter");
        let history = vec![2u32, 100, 200];
        drafter.seed_history(&history);
        let mut rng = VerifyRng::new(0xCAFE);

        unsafe {
            env::set_var("LARQL_SPECULATIVE_DECODE", "1");
        }
        let cfg = SpecConfig {
            depth: 2,
            branches: 1,
            swa_window: None,
        };
        let result = run_naive_step(
            &mut weights,
            &mut drafter,
            &history,
            3,
            cfg,
            &index,
            &mut rng,
        );
        // With env on and a successful draft, run_naive_step returns
        // at least one token. Exact count depends on acceptance rate
        // (even drafter == target shows non-determinism between two
        // separately-loaded ModelWeights instances at fp32 precision).
        // The contract this test enforces: dispatch succeeds, we get
        // tokens out, and the drafter's history advances.
        let tokens = result.expect("expected tokens from successful step");
        assert!(
            !tokens.is_empty(),
            "successful step must emit at least one token"
        );
        assert!(
            tokens.len() <= 3,
            "must not emit more than depth + 1 tokens"
        );
        assert_eq!(
            drafter.history_len(),
            history.len() + tokens.len(),
            "drafter history must advance by emitted token count"
        );
        unsafe {
            env::remove_var("LARQL_SPECULATIVE_DECODE");
        }
    }

    #[test]
    fn returns_none_when_swa_window_exhausted() {
        let Some(path) = vindex_path_or_skip() else {
            return;
        };
        let mut weights = load(&path);
        let index = open_index(&path);
        let mut drafter = SmallModelDrafter::from_vindex(&path).expect("drafter");
        let history = vec![2u32, 100, 200];
        drafter.seed_history(&history);
        let mut rng = VerifyRng::new(0);

        unsafe {
            env::set_var("LARQL_SPECULATIVE_DECODE", "1");
        }
        let cfg = SpecConfig {
            depth: 4,
            branches: 1,
            swa_window: Some(3),
        };
        let result = run_naive_step(
            &mut weights,
            &mut drafter,
            &history,
            3,
            cfg,
            &index,
            &mut rng,
        );
        assert_eq!(result, None, "exhausted SWA window must fall through");
        unsafe {
            env::remove_var("LARQL_SPECULATIVE_DECODE");
        }
    }
}
