//! Speculative decoding (`inference-speculative-decoding`).
//!
//! Phase 1 scaffolding: trait, config, env-flag dispatch, and the
//! CPU reference `verify_and_accept` that later phases will use as
//! the parity oracle for the GPU `verify_tree` kernel.
//!
//! The full design lives in
//! [openspec/changes/cuda-speculative-decoding](
//! ../../../../openspec/changes/cuda-speculative-decoding/proposal.md).
//!
//! Activation is gated by `LARQL_SPECULATIVE_DECODE=1`. Any other
//! value (unset, empty, `0`, anything else) falls through to the
//! existing non-speculative `decode_token` path bit-exactly.

use std::env;

pub mod dispatch;
pub mod eagle;
pub mod orchestrator;
pub mod small_model;
pub mod target_forward;
pub mod tree;
pub mod verify;
pub mod wiring;

pub use dispatch::maybe_speculative_step;
pub use eagle::EagleDraftHead;
pub use orchestrator::{build_linear_tree, SpeculativeStep, StepOutcome};
pub use small_model::SmallModelDrafter;
pub use target_forward::{target_forward_naive, target_forward_with_hidden};
pub use tree::{DraftTree, TreeAttentionMask, TreeNode};
pub use verify::{verify_and_accept, verify_tree, AcceptedSpan, VerifyRng};
pub use wiring::{
    run_naive_step, set_thread_drafter, set_thread_rng, set_thread_spec_config,
    try_thread_speculative_step,
};

pub type TokenId = u32;

/// Per-step draft proposal: a candidate token plus the draft
/// model's probability assigned to it. `verify_and_accept` uses
/// these probabilities for the rejection-sampling rule.
#[derive(Clone, Debug)]
pub struct DraftToken {
    pub id: TokenId,
    pub p_draft: f32,
}

/// Caller-supplied draft model. Phase 1 ships [`EagleDraftHead`];
/// off-the-shelf small-model drafters use [`small_model::SmallModelDrafter`].
///
/// Drafter impls fall into two categories:
///
/// - **Hidden-state drafters** (e.g. EAGLE) — use `h_target` to propose
///   conditionally on the target's hidden state. These typically share
///   the target's tokenizer and live in the same process.
/// - **Off-the-shelf drafters** — load a separate small model and
///   maintain their own KV cache + token history. They ignore
///   `h_target` and rely on `accept()` to keep history in sync with
///   the target's accepted span.
pub trait Drafter {
    /// Propose `n` candidate continuations. Returns at most `n`
    /// tokens; fewer is allowed (e.g. when constrained by sliding
    /// window or when the drafter declines).
    ///
    /// `h_target` is the target model's last hidden state at the
    /// current generation position. Hidden-state drafters use this;
    /// off-the-shelf drafters ignore it.
    fn propose(&mut self, h_target: &[f32], n: usize) -> Vec<DraftToken>;

    /// Reset any per-sequence draft state (e.g. draft KV cache,
    /// token history). Called at the start of each new generation.
    fn reset(&mut self);

    /// Notify the drafter of tokens the target accepted. Off-the-shelf
    /// drafters use this to advance their internal token history so the
    /// next `propose()` call sees the right context. Hidden-state
    /// drafters typically no-op (default impl).
    fn accept(&mut self, _accepted: &[TokenId]) {}
}

/// Speculative-decoder configuration. Defaults match design.md
/// phase 4 starting point: depth=2, branches=2 (5-node tree).
#[derive(Clone, Copy, Debug)]
pub struct SpecConfig {
    pub depth: usize,
    pub branches: usize,
    /// Sliding-window upper bound. Per design.md, dispatched depth
    /// is clamped to `min(depth, swa_window - cache_len)`.
    pub swa_window: Option<usize>,
}

impl Default for SpecConfig {
    fn default() -> Self {
        Self {
            depth: 2,
            branches: 2,
            swa_window: None,
        }
    }
}

impl SpecConfig {
    /// Effective depth given the current cache length, after the
    /// sliding-window clamp. Returns 0 if there is no slack
    /// remaining — caller SHOULD fall back to non-speculative.
    pub fn effective_depth(&self, cache_len: usize) -> usize {
        match self.swa_window {
            Some(w) => self.depth.min(w.saturating_sub(cache_len)),
            None => self.depth,
        }
    }

    /// Total tree node count given `depth` and `branches`. With
    /// branches=K and depth=D this is `(K^(D+1) - 1) / (K - 1)`
    /// for K > 1, or `D + 1` for K = 1. Cap is enforced at 64
    /// (matches `DecodeScratch::max_tree_nodes` in design.md §2.3).
    pub fn tree_nodes(&self) -> usize {
        let nodes = if self.branches <= 1 {
            self.depth + 1
        } else {
            // Geometric sum, plus the implicit root.
            let mut sum = 1usize;
            let mut term = 1usize;
            for _ in 0..self.depth {
                term = term.saturating_mul(self.branches);
                sum = sum.saturating_add(term);
            }
            sum
        };
        nodes.min(64)
    }
}

/// Returns true iff `LARQL_SPECULATIVE_DECODE=1`. Any other value
/// falls through to the legacy path.
pub fn enabled() -> bool {
    env::var("LARQL_SPECULATIVE_DECODE")
        .ok()
        .as_deref()
        .map(|v| v == "1")
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unset_env_uses_legacy_path() {
        // SAFETY: tests in this file are not run in parallel with
        // anything that mutates the same var (no other code in this
        // crate reads LARQL_SPECULATIVE_DECODE).
        unsafe {
            env::remove_var("LARQL_SPECULATIVE_DECODE");
        }
        assert!(
            !enabled(),
            "unset LARQL_SPECULATIVE_DECODE must disable spec path"
        );
    }

    #[test]
    fn env_zero_uses_legacy_path() {
        unsafe {
            env::set_var("LARQL_SPECULATIVE_DECODE", "0");
        }
        assert!(!enabled());
        unsafe {
            env::remove_var("LARQL_SPECULATIVE_DECODE");
        }
    }

    #[test]
    fn env_one_enables_spec_path() {
        unsafe {
            env::set_var("LARQL_SPECULATIVE_DECODE", "1");
        }
        assert!(enabled());
        unsafe {
            env::remove_var("LARQL_SPECULATIVE_DECODE");
        }
    }

    #[test]
    fn env_garbage_uses_legacy_path() {
        unsafe {
            env::set_var("LARQL_SPECULATIVE_DECODE", "true");
        }
        assert!(!enabled());
        unsafe {
            env::remove_var("LARQL_SPECULATIVE_DECODE");
        }
    }

    #[test]
    fn effective_depth_clamps_to_swa_remainder() {
        let cfg = SpecConfig {
            depth: 4,
            branches: 2,
            swa_window: Some(100),
        };
        assert_eq!(cfg.effective_depth(99), 1);
        assert_eq!(cfg.effective_depth(100), 0);
        assert_eq!(cfg.effective_depth(50), 4);
    }

    #[test]
    fn tree_nodes_default_is_seven() {
        // depth=2, branches=2: 1 (root) + 2 + 4 = 7 nodes
        assert_eq!(SpecConfig::default().tree_nodes(), 7);
    }

    #[test]
    fn tree_nodes_caps_at_64() {
        let cfg = SpecConfig {
            depth: 10,
            branches: 4,
            swa_window: None,
        };
        assert_eq!(cfg.tree_nodes(), 64);
    }
}
