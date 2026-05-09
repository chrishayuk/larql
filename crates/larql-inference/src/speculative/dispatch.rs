//! Phase 4 integration boundary — the single entry point the
//! inference decode loop calls to opt into speculative decoding.
//!
//! Designed to be a **drop-in replacement** for one iteration of
//! the existing per-token decode loop. The function returns
//! `Option<Vec<TokenId>>`:
//!
//! - `Some(tokens)` when speculation succeeded — emit all of these
//!   as the next 1..=N+1 tokens. KV cache should be advanced by
//!   `tokens.len()` positions.
//! - `None` when the caller SHOULD fall through to the existing
//!   non-speculative `decode_token` + sample loop. This happens
//!   when:
//!     - `LARQL_SPECULATIVE_DECODE` is unset / not `1`
//!     - the drafter declines (returns empty proposals)
//!     - the SWA window leaves no slack
//!     - `target_forward` violates its contract
//!
//! ## Caller contract
//!
//! The caller wires `target_forward` to a closure that:
//!   1. Runs the target model's forward pass on the draft tree's
//!      tokens (via `cuda::attn_tree::tree_decode_attention` plus
//!      the existing projection + KV-write kernels).
//!   2. Computes target softmax probabilities at each tree node
//!      (via `cuda::sampling::verify_tree_gpu_parallel` semantics
//!      or a separate softmax pass).
//!   3. Returns one target probability vector per tree node, in
//!      tree-node order.
//!
//! When that closure is wired and `LARQL_SPECULATIVE_DECODE=1`,
//! the integration is complete. Until then this function is a
//! safe no-op (returns `None`).

use super::orchestrator::{SpeculativeStep, StepOutcome};
use super::tree::DraftTree;
use super::verify::VerifyRng;
use super::{Drafter, SpecConfig, TokenId};

/// One speculative decode step. Returns `Some(emitted_tokens)`
/// on a successful step, `None` to signal the caller to do the
/// existing non-speculative path.
///
/// `h_target` is the target model's last hidden state at the
/// current generation position (used by `drafter` to propose
/// candidates).
///
/// `cache_len` is the current KV cache length (for SWA depth
/// clamping).
///
/// `target_forward` is invoked at most once per call, with the
/// constructed draft tree. It SHALL return one target probability
/// vector (vocab-sized) per tree node, in node order. If it
/// returns wrong-sized vectors or the wrong count, the call
/// returns `None` (defensive — production should log + retry
/// non-speculatively).
pub fn maybe_speculative_step<D, F>(
    drafter: &mut D,
    cfg: SpecConfig,
    h_target: &[f32],
    cache_len: usize,
    rng: &mut VerifyRng,
    target_forward: F,
) -> Option<Vec<TokenId>>
where
    D: Drafter,
    F: FnOnce(&DraftTree) -> Vec<Vec<f32>>,
{
    if !super::enabled() {
        return None;
    }
    let mut step = SpeculativeStep::new(drafter, cfg);
    match step.run(h_target, cache_len, rng, target_forward) {
        StepOutcome::Span(span) => Some(span.tokens()),
        StepOutcome::FallThrough => None,
    }
}

#[cfg(test)]
mod tests {
    use super::super::tree::DraftTree;
    use super::super::DraftToken;
    use super::*;
    use std::env;

    struct EmptyDrafter;
    impl Drafter for EmptyDrafter {
        fn propose(&mut self, _h: &[f32], _n: usize) -> Vec<DraftToken> {
            Vec::new()
        }
        fn reset(&mut self) {}
    }

    struct CannedDrafter(Vec<DraftToken>);
    impl Drafter for CannedDrafter {
        fn propose(&mut self, _h: &[f32], n: usize) -> Vec<DraftToken> {
            self.0.iter().take(n).cloned().collect()
        }
        fn reset(&mut self) {}
    }

    fn unit(id: TokenId, vocab: usize) -> Vec<f32> {
        let mut v = vec![0.0; vocab];
        v[id as usize] = 1.0;
        v
    }

    #[test]
    fn returns_none_when_env_disabled() {
        unsafe {
            env::remove_var("LARQL_SPECULATIVE_DECODE");
        }
        let mut drafter = CannedDrafter(vec![DraftToken {
            id: 1,
            p_draft: 1.0,
        }]);
        let mut rng = VerifyRng::new(0);
        let result = maybe_speculative_step(
            &mut drafter,
            SpecConfig::default(),
            &[0.0; 16],
            0,
            &mut rng,
            |_tree: &DraftTree| {
                unreachable!("target_forward must not be called when env is disabled")
            },
        );
        assert_eq!(result, None);
    }

    #[test]
    fn returns_none_on_empty_drafter_even_when_env_enabled() {
        unsafe {
            env::set_var("LARQL_SPECULATIVE_DECODE", "1");
        }
        let mut drafter = EmptyDrafter;
        let mut rng = VerifyRng::new(0);
        let result = maybe_speculative_step(
            &mut drafter,
            SpecConfig::default(),
            &[0.0; 16],
            0,
            &mut rng,
            |_tree: &DraftTree| {
                unreachable!("target_forward must not be called when drafter is empty")
            },
        );
        assert_eq!(result, None);
        unsafe {
            env::remove_var("LARQL_SPECULATIVE_DECODE");
        }
    }

    #[test]
    fn returns_some_tokens_on_successful_step() {
        unsafe {
            env::set_var("LARQL_SPECULATIVE_DECODE", "1");
        }
        let vocab = 4;
        let mut drafter = CannedDrafter(vec![
            DraftToken {
                id: 1,
                p_draft: 1.0,
            },
            DraftToken {
                id: 2,
                p_draft: 1.0,
            },
        ]);
        let cfg = SpecConfig {
            depth: 2,
            branches: 1,
            swa_window: None,
        };
        let mut rng = VerifyRng::new(0xCAFE);
        let result = maybe_speculative_step(
            &mut drafter,
            cfg,
            &[0.0; 16],
            0,
            &mut rng,
            |tree: &DraftTree| tree.tokens().iter().map(|&id| unit(id, vocab)).collect(),
        );
        // All-accept on one-hot target → emits 2 accepted + 1 bonus.
        let tokens = result.expect("expected speculative success");
        assert_eq!(tokens.len(), 3, "accepted + bonus");
        assert_eq!(tokens[..2], vec![1, 2]);
        unsafe {
            env::remove_var("LARQL_SPECULATIVE_DECODE");
        }
    }

    #[test]
    fn returns_none_when_target_forward_violates_contract() {
        // Simulates production-defensive behavior: if the closure
        // returns the wrong number of probability vectors, fall through
        // rather than panic.
        unsafe {
            env::set_var("LARQL_SPECULATIVE_DECODE", "1");
        }
        let mut drafter = CannedDrafter(vec![DraftToken {
            id: 1,
            p_draft: 1.0,
        }]);
        let mut rng = VerifyRng::new(0);
        let result = maybe_speculative_step(
            &mut drafter,
            SpecConfig::default(),
            &[0.0; 16],
            0,
            &mut rng,
            |_tree: &DraftTree| Vec::new(),
        );
        assert_eq!(result, None);
        unsafe {
            env::remove_var("LARQL_SPECULATIVE_DECODE");
        }
    }
}
