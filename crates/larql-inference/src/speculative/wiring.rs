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

use std::cell::RefCell;
use std::path::Path;

use larql_models::ModelWeights;
use larql_vindex::VectorIndex;
use ndarray::Array2;

use crate::error::InferenceError;

use super::orchestrator::build_linear_tree;
use super::small_model::SmallModelDrafter;
use super::target_forward::{target_forward_naive, target_forward_with_hidden};
use super::verify::{verify_tree, VerifyRng};
use super::{Drafter, SpecConfig, TokenId};

/// Owns a separate `ModelWeights` instance + `VectorIndex` for
/// speculative target re-runs. The bench loads this from the
/// SAME vindex bytes as the canonical target — mmap means zero
/// additional RSS despite logically duplicating the weights.
///
/// Resolves the borrow conflict at gpu.rs:735 documented in PR #18:
/// the canonical decode loop holds `&layers` (borrowing the canonical
/// weights), while the speculative target needs `&mut weights`. Owning
/// a separate instance gives the speculative path its own mutable
/// surface without touching the canonical one.
pub struct SpeculativeTargetExecutor {
    weights: ModelWeights,
    index: VectorIndex,
}

impl SpeculativeTargetExecutor {
    /// Load a separate weights+vindex pair from the same directory the
    /// canonical target is using. Mmap is the underlying storage, so
    /// peak RSS is unaffected.
    pub fn from_vindex(path: impl AsRef<Path>) -> Result<Self, InferenceError> {
        let path = path.as_ref();
        let mut callbacks = larql_vindex::SilentLoadCallbacks;
        let weights = larql_vindex::load_model_weights_q4k(path, &mut callbacks)
            .map_err(InferenceError::Vindex)?;
        let index = crate::open_inference_vindex(path)?;
        Ok(Self { weights, index })
    }

    /// Run the target's full forward pass on `tokens` and return the
    /// `[seq_len, hidden]` hidden state. Used as the closure body for
    /// `target_forward_with_hidden`.
    pub fn compute_hidden(&mut self, tokens: &[TokenId]) -> Array2<f32> {
        crate::vindex::predict_q4k_hidden(&mut self.weights, tokens, &self.index, None)
    }
}

thread_local! {
    /// Per-thread speculative drafter. Set by the caller (e.g.
    /// `larql bench` after `--draft-model` loads) before invoking
    /// `generate_streaming`; read by the per-token loop in
    /// `layer_graph::generate::gpu` to opt into speculative dispatch.
    ///
    /// Thread-local is the surgery-free alternative to changing the
    /// signature of `generate()` and its 17 call sites. Single-thread
    /// bench/CLI use case fits perfectly; if multi-thread serving
    /// adopts speculative later, signature plumbing becomes worth it.
    static THREAD_DRAFTER: RefCell<Option<SmallModelDrafter>> = const { RefCell::new(None) };
    static THREAD_RNG: RefCell<Option<VerifyRng>> = const { RefCell::new(None) };
    static THREAD_CFG: RefCell<SpecConfig> = const {
        RefCell::new(SpecConfig {
            depth: 2,
            branches: 1,
            swa_window: None,
        })
    };
    /// Per-thread separately-loaded target executor. Set by the
    /// caller before generate() begins; read by `try_thread_speculative_step_v2`.
    /// Resolves the borrow conflict by owning its own `&mut ModelWeights`
    /// independent of the canonical loop's weights.
    static THREAD_TARGET_EXEC: RefCell<Option<SpeculativeTargetExecutor>> = const { RefCell::new(None) };
}

/// Install a drafter on the current thread. Pass `None` to clear.
pub fn set_thread_drafter(d: Option<SmallModelDrafter>) {
    THREAD_DRAFTER.with(|cell| {
        *cell.borrow_mut() = d;
    });
}

/// Install a SpecConfig on the current thread (overrides default
/// depth=2 branches=1).
pub fn set_thread_spec_config(cfg: SpecConfig) {
    THREAD_CFG.with(|cell| {
        *cell.borrow_mut() = cfg;
    });
}

/// Install an RNG seed on the current thread. The RNG is consumed
/// (mutated) by each speculative step.
pub fn set_thread_rng(seed: u64) {
    THREAD_RNG.with(|cell| {
        *cell.borrow_mut() = Some(VerifyRng::new(seed));
    });
}

/// Install a separate target executor on the current thread. Pass
/// `None` to clear. Required for `try_thread_speculative_step_v2`
/// to dispatch.
pub fn set_thread_target_executor(exec: Option<SpeculativeTargetExecutor>) {
    THREAD_TARGET_EXEC.with(|cell| {
        *cell.borrow_mut() = exec;
    });
}

/// Borrow-conflict-free dispatch helper. Takes `&ModelWeights`
/// (immutable) for the lm_head + softmax projection inside
/// `target_forward_with_hidden`. Mutability for `predict_q4k_hidden`
/// comes from the thread-local `SpeculativeTargetExecutor`.
///
/// Returns `Some(emitted_tokens)` on a successful step (caller MUST
/// advance KV cache by `tokens.len()`); `None` to fall through to
/// the existing non-speculative path.
///
/// Returns `None` when:
/// - `LARQL_SPECULATIVE_DECODE` is unset / not `1`
/// - thread-local `THREAD_DRAFTER` is None
/// - thread-local `THREAD_TARGET_EXEC` is None
/// - SWA window leaves no slack
/// - drafter declines (empty proposals)
pub fn try_thread_speculative_step_v2(
    weights: &ModelWeights,
    history: &[TokenId],
    cache_len: usize,
) -> Option<Vec<TokenId>> {
    if !super::enabled() {
        return None;
    }
    THREAD_DRAFTER.with(|d_cell| {
        let mut d_ref = d_cell.borrow_mut();
        let drafter = d_ref.as_mut()?;
        THREAD_TARGET_EXEC.with(|t_cell| {
            let mut t_ref = t_cell.borrow_mut();
            let target = t_ref.as_mut()?;
            let cfg = THREAD_CFG.with(|c| *c.borrow());
            let depth = cfg.effective_depth(cache_len);
            if depth == 0 {
                return None;
            }
            // Re-seed the drafter's internal history with the loop's
            // canonical context every iteration. Wasteful (clones the
            // history) but correct — without this, the drafter has no
            // context for its propose() call. Phase 4c can optimize
            // by tracking drafter history incrementally.
            drafter.seed_history(history);
            let drafts = drafter.propose(&[], depth);
            if drafts.is_empty() {
                return None;
            }
            let tree = build_linear_tree(&drafts);
            let p_target = target_forward_with_hidden(weights, history, &tree, |toks| {
                target.compute_hidden(toks)
            });
            if p_target.len() != tree.len() {
                return None;
            }
            let span = THREAD_RNG.with(|r_cell| {
                let mut r_ref = r_cell.borrow_mut();
                let rng = r_ref.get_or_insert_with(|| VerifyRng::new(0xCAFE_BABE_DEAD_F00D));
                verify_tree(&tree, &p_target, rng)
            });
            let emitted = span.tokens();
            if emitted.is_empty() {
                return None;
            }
            drafter.accept(&emitted);
            Some(emitted)
        })
    })
}

/// **Phase 4c task C.2.d** — borrow-conflict-free dispatch using the
/// canonical backend (no separate `SpeculativeTargetExecutor` needed).
/// Composes `super::target_forward::target_forward_via_speculative_decode`
/// (the C.2.b/c work — sequential decode_token + KV rollback, with
/// linear-chain optimization) with `verify_tree` to produce the
/// accepted span.
///
/// **Vs `try_thread_speculative_step_v2`**: v2 uses a separately-loaded
/// `ModelWeights` instance (~6 GB peak heap, ~6s drafter forward per
/// proposal because predict_q4k from-scratch). v3 uses the canonical
/// backend's KV cache via decode_token (~7.5 ms per token) — drops
/// target_forward cost from ~12s to ~15ms (~800× speedup).
///
/// Drafter still uses the SmallModelDrafter path (own KV cache via
/// predict_q4k from scratch) — drafter perf optimization is a
/// separate slice.
///
/// Returns `Some(emitted_tokens)` on a successful step (caller MUST
/// advance the canonical KV cache by `tokens.len()`); `None` to fall
/// through to the existing non-speculative path.
///
/// Returns `None` when:
/// - `LARQL_SPECULATIVE_DECODE` is unset / not `1`
/// - thread-local `THREAD_DRAFTER` is None
/// - `backend.has_kv_cache()` returns false
/// - `backend.kv_cache_len() != history.len()` (caller contract violation)
/// - SWA window leaves no slack
/// - drafter declines (empty proposals)
#[allow(clippy::too_many_arguments)]
pub fn try_thread_speculative_step_v3(
    weights: &ModelWeights,
    history: &[TokenId],
    cache_len: usize,
    backend: &dyn larql_compute::backend::DecodeBackend,
    layers: &[larql_compute::FullPipelineLayer<'_>],
    dims: super::target_forward::TargetForwardDims,
) -> Option<Vec<TokenId>> {
    if !super::enabled() {
        return None;
    }
    THREAD_DRAFTER.with(|d_cell| {
        let mut d_ref = d_cell.borrow_mut();
        let drafter = d_ref.as_mut()?;
        let cfg = THREAD_CFG.with(|c| *c.borrow());
        let depth = cfg.effective_depth(cache_len);
        if depth == 0 {
            return None;
        }
        drafter.seed_history(history);
        let drafts = drafter.propose(&[], depth);
        if drafts.is_empty() {
            return None;
        }
        let tree = build_linear_tree(&drafts);
        let p_target = super::target_forward::target_forward_via_speculative_decode(
            weights, history, &tree, backend, layers, dims,
        )?;
        if p_target.len() != tree.len() {
            return None;
        }
        let span = THREAD_RNG.with(|r_cell| {
            let mut r_ref = r_cell.borrow_mut();
            let rng = r_ref.get_or_insert_with(|| VerifyRng::new(0xCAFE_BABE_DEAD_F00D));
            verify_tree(&tree, &p_target, rng)
        });
        let emitted = span.tokens();
        if emitted.is_empty() {
            return None;
        }
        drafter.accept(&emitted);
        Some(emitted)
    })
}

/// Try one speculative step using the thread-local drafter (if set).
/// Returns `Some(tokens)` on success — caller MUST advance KV cache by
/// `tokens.len()` positions. `None` to fall through to the existing
/// non-speculative path.
///
/// `weights`, `history`, `index` come from the inference loop's
/// existing scope. Drafter, RNG, and SpecConfig come from
/// thread-locals — set by the caller (e.g. `larql bench`) before
/// generate begins.
pub fn try_thread_speculative_step(
    weights: &mut ModelWeights,
    history: &[TokenId],
    cache_len: usize,
    index: &VectorIndex,
) -> Option<Vec<TokenId>> {
    if !super::enabled() {
        return None;
    }
    THREAD_DRAFTER.with(|drafter_cell| {
        let mut drafter_ref = drafter_cell.borrow_mut();
        let drafter = drafter_ref.as_mut()?;
        let cfg = THREAD_CFG.with(|c| *c.borrow());
        THREAD_RNG.with(|rng_cell| {
            let mut rng_ref = rng_cell.borrow_mut();
            let rng = rng_ref.get_or_insert_with(|| VerifyRng::new(0xCAFE_BABE_DEAD_F00D));
            run_naive_step(weights, drafter, history, cache_len, cfg, index, rng)
        })
    })
}

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
