//! Speculative decode step orchestrator.
//!
//! Ties the per-step pieces together into the contract phase 3's
//! inference loop calls into:
//!
//! ```text
//! drafter.propose(h, depth)
//!   → build_linear_draft_tree(proposals)
//!   → target_forward(tree)         (phase 3: cuda::attn_tree::tree_decode_attention)
//!   → verify_tree(tree, p_target, rng)
//!   → AcceptedSpan
//! ```
//!
//! The `target_forward` callable is the only piece the orchestrator
//! does not own. Phase 3 wires this to the CUDA tree-attention
//! kernel + softmax; for testing here, a CPU closure provides
//! synthetic per-node target distributions so end-to-end behaviour
//! is verifiable without GPU.

use super::tree::DraftTree;
use super::verify::{verify_tree, AcceptedSpan, VerifyRng};
use super::{DraftToken, Drafter, SpecConfig, TokenId};

/// Outcome of one speculative step. `Span` carries accepted +
/// corrected/bonus tokens; `FallThrough` signals the caller SHALL
/// run a non-speculative decode (drafter declined or window ran
/// out).
#[derive(Clone, Debug, PartialEq)]
pub enum StepOutcome {
    Span(AcceptedSpan),
    FallThrough,
}

impl StepOutcome {
    /// Number of tokens this step contributed. `FallThrough` is 0.
    pub fn emitted_count(&self) -> usize {
        match self {
            StepOutcome::Span(s) => s.emitted_count(),
            StepOutcome::FallThrough => 0,
        }
    }

    pub fn tokens(&self) -> Vec<TokenId> {
        match self {
            StepOutcome::Span(s) => s.tokens(),
            StepOutcome::FallThrough => Vec::new(),
        }
    }
}

/// One-shot speculative step. Holds borrowed references to the
/// drafter and config; constructed per decode step.
pub struct SpeculativeStep<'a, D: Drafter> {
    drafter: &'a mut D,
    cfg: SpecConfig,
}

impl<'a, D: Drafter> SpeculativeStep<'a, D> {
    pub fn new(drafter: &'a mut D, cfg: SpecConfig) -> Self {
        Self { drafter, cfg }
    }

    /// Run propose → verify. `cache_len` is the current target KV
    /// cache length, used for the SWA depth clamp. `target_forward`
    /// is called once with the constructed [`DraftTree`] and SHALL
    /// return one target probability vector per tree node (in node
    /// order). `rng` drives both the rejection-sampling decisions
    /// and any residual / bonus sampling.
    pub fn run<F>(
        &mut self,
        h_target: &[f32],
        cache_len: usize,
        rng: &mut VerifyRng,
        target_forward: F,
    ) -> StepOutcome
    where
        F: FnOnce(&DraftTree) -> Vec<Vec<f32>>,
    {
        let depth = self.cfg.effective_depth(cache_len);
        if depth == 0 {
            return StepOutcome::FallThrough;
        }

        let proposals = self.drafter.propose(h_target, depth);
        if proposals.is_empty() {
            return StepOutcome::FallThrough;
        }

        let tree = build_linear_tree(&proposals);
        let p_target_per_node = target_forward(&tree);

        if p_target_per_node.len() != tree.len() {
            // Defensive: the contract is one distribution per node.
            // Returning FallThrough rather than panicking lets the
            // caller log + retry non-speculatively in production.
            return StepOutcome::FallThrough;
        }

        StepOutcome::Span(verify_tree(&tree, &p_target_per_node, rng))
    }
}

/// Build a linear (depth-N, no branching) tree from a draft list.
/// Phase 3 will swap this for a `build_eagle_tree` that takes the
/// drafter's branching structure into account; for phase 1 / 2 the
/// linear tree is sufficient because batched mmvq exercises q ≥ 2
/// already.
pub fn build_linear_tree(proposals: &[DraftToken]) -> DraftTree {
    debug_assert!(
        !proposals.is_empty(),
        "build_linear_tree requires at least one proposal"
    );
    let mut iter = proposals.iter();
    let root = iter.next().unwrap().clone();
    let mut tree = DraftTree::from_root(root);
    let mut parent = 0;
    for d in iter {
        parent = tree.add_child(parent, d.clone());
    }
    tree
}

#[cfg(test)]
mod tests {
    use super::super::DraftToken;
    use super::*;

    /// Test-only drafter that returns a fixed sequence of proposals.
    struct CannedDrafter {
        next: Vec<DraftToken>,
        reset_count: usize,
    }

    impl CannedDrafter {
        fn empty() -> Self {
            Self {
                next: Vec::new(),
                reset_count: 0,
            }
        }
        fn with(seq: Vec<DraftToken>) -> Self {
            Self {
                next: seq,
                reset_count: 0,
            }
        }
    }

    impl Drafter for CannedDrafter {
        fn propose(&mut self, _h: &[f32], n: usize) -> Vec<DraftToken> {
            self.next.iter().take(n).cloned().collect()
        }
        fn reset(&mut self) {
            self.reset_count += 1;
        }
    }

    fn unit(id: TokenId, vocab: usize) -> Vec<f32> {
        let mut v = vec![0.0; vocab];
        v[id as usize] = 1.0;
        v
    }

    #[test]
    fn empty_drafter_falls_through() {
        let mut drafter = CannedDrafter::empty();
        let mut step = SpeculativeStep::new(&mut drafter, SpecConfig::default());
        let mut rng = VerifyRng::new(0);
        let outcome = step.run(&[0.0; 16], 0, &mut rng, |_tree| {
            unreachable!("target_forward must not be called when drafter is empty")
        });
        assert_eq!(outcome, StepOutcome::FallThrough);
        assert_eq!(outcome.emitted_count(), 0);
        assert!(outcome.tokens().is_empty());
    }

    #[test]
    fn zero_effective_depth_falls_through() {
        // SWA window is exhausted → effective_depth == 0 → fall through
        // before calling drafter or target.
        let mut drafter = CannedDrafter::with(vec![DraftToken {
            id: 1,
            p_draft: 1.0,
        }]);
        let cfg = SpecConfig {
            depth: 4,
            branches: 1,
            swa_window: Some(10),
        };
        let mut step = SpeculativeStep::new(&mut drafter, cfg);
        let mut rng = VerifyRng::new(0);
        let outcome = step.run(&[0.0; 16], 10, &mut rng, |_tree| {
            unreachable!("target_forward must not be called when depth=0")
        });
        assert_eq!(outcome, StepOutcome::FallThrough);
    }

    #[test]
    fn linear_propose_verify_emit_round_trip() {
        // Drafter proposes [1, 2]; target distributions are one-hot
        // at those ids; everything accepts; bonus is sampled from
        // p_target at the deepest node (one-hot at 2 → bonus = 2).
        let vocab = 4;
        let mut drafter = CannedDrafter::with(vec![
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
        let mut step = SpeculativeStep::new(&mut drafter, cfg);
        let mut rng = VerifyRng::new(0xCAFEBABE);

        let outcome = step.run(&[0.0; 16], 0, &mut rng, |tree| {
            assert_eq!(tree.len(), 2);
            tree.tokens().iter().map(|&id| unit(id, vocab)).collect()
        });

        match outcome {
            StepOutcome::Span(s) => {
                assert_eq!(s.accepted, vec![1, 2]);
                assert_eq!(s.bonus, Some(2));
                assert!(s.corrected.is_none());
            }
            _ => panic!("expected Span outcome"),
        }
    }

    #[test]
    fn target_returns_wrong_node_count_falls_through() {
        // Defensive: if target_forward violates its contract, the
        // step falls through rather than panicking — production
        // behaviour is to log and retry non-speculatively.
        let mut drafter = CannedDrafter::with(vec![DraftToken {
            id: 1,
            p_draft: 1.0,
        }]);
        let cfg = SpecConfig::default();
        let mut step = SpeculativeStep::new(&mut drafter, cfg);
        let mut rng = VerifyRng::new(0);
        let outcome = step.run(&[0.0; 16], 0, &mut rng, |_tree| Vec::new()); // empty
        assert_eq!(outcome, StepOutcome::FallThrough);
    }

    #[test]
    fn build_linear_tree_chain_structure() {
        let proposals = vec![
            DraftToken {
                id: 10,
                p_draft: 0.9,
            },
            DraftToken {
                id: 20,
                p_draft: 0.8,
            },
            DraftToken {
                id: 30,
                p_draft: 0.7,
            },
        ];
        let tree = build_linear_tree(&proposals);
        assert_eq!(tree.len(), 3);
        assert_eq!(tree.max_depth(), 2);
        assert_eq!(tree.tokens(), vec![10, 20, 30]);
        // Root has one child, that child has one child.
        let paths = tree.root_to_leaf_paths();
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0], vec![0, 1, 2]);
    }
}
