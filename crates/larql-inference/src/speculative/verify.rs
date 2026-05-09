//! CPU reference for speculative-decode verification.
//!
//! Implements the rejection-sampling rule from
//! Leviathan et al. 2022 ("Fast Inference from Transformers via
//! Speculative Decoding"). This is the **load-bearing correctness
//! oracle** — the GPU `verify_tree` kernel (phase 3) is tested for
//! bit-equal token-ID parity against this implementation.
//!
//! The rule guarantees the output distribution is identical to
//! direct sampling from the target. Proof: see paper §3.

use super::tree::DraftTree;
use super::{DraftToken, TokenId};

/// Result of one speculative step's verification.
#[derive(Clone, Debug, PartialEq)]
pub struct AcceptedSpan {
    /// Draft tokens accepted, in order. Length 0..=draft.len().
    pub accepted: Vec<TokenId>,
    /// On rejection: the token sampled from the residual
    /// `max(0, p_target - p_draft)` distribution. `None` only when
    /// every draft token is accepted.
    pub corrected: Option<TokenId>,
    /// On all-accept: bonus token sampled directly from `p_target`
    /// at the deepest accepted position. `None` on early rejection.
    pub bonus: Option<TokenId>,
}

impl AcceptedSpan {
    /// Total tokens emitted by this step (accepted + corrected +
    /// bonus). Always ≥ 1.
    pub fn emitted_count(&self) -> usize {
        self.accepted.len() + self.corrected.is_some() as usize + self.bonus.is_some() as usize
    }

    /// Flat token sequence in emit order: accepted, then corrected
    /// (if any), then bonus (if any).
    pub fn tokens(&self) -> Vec<TokenId> {
        let mut out = self.accepted.clone();
        if let Some(c) = self.corrected {
            out.push(c);
        }
        if let Some(b) = self.bonus {
            out.push(b);
        }
        out
    }
}

/// Deterministic RNG used for the acceptance/sampling decisions.
/// We use SplitMix64 because (a) it's tiny, (b) trivially portable
/// to a CUDA kernel, (c) deterministic given a seed. Phase 3's
/// GPU verify_tree will use the same scheme so parity tests can
/// share seeds.
#[derive(Clone, Debug)]
pub struct VerifyRng {
    state: u64,
}

impl VerifyRng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Next f32 in [0, 1). Mirrors the kernel-side splitmix.
    pub fn next_f32(&mut self) -> f32 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        // Use the top 24 bits for a [0, 1) f32.
        ((z >> 40) as f32) / ((1u32 << 24) as f32)
    }
}

/// Verify a linear (depth-N, no branching) draft sequence against
/// target probabilities. Implements the exact rejection-sampling
/// rule. Tree verification (phase 3) reuses this row-wise.
///
/// `p_target[k]` is the full target distribution at draft position
/// `k` (length = vocab). `draft[k]` is the candidate sampled by the
/// drafter and the probability it assigned.
pub fn verify_and_accept(
    p_target: &[Vec<f32>],
    draft: &[DraftToken],
    rng: &mut VerifyRng,
) -> AcceptedSpan {
    assert_eq!(
        p_target.len(),
        draft.len(),
        "verify_and_accept: target/draft length mismatch"
    );

    let mut accepted = Vec::with_capacity(draft.len());

    for (k, d) in draft.iter().enumerate() {
        let pt = p_target[k][d.id as usize];
        let pd = d.p_draft.max(f32::MIN_POSITIVE);
        let accept_prob = (pt / pd).min(1.0);

        if rng.next_f32() < accept_prob {
            accepted.push(d.id);
            continue;
        }

        // Rejected at position k. Sample one corrected token from
        // the residual distribution max(0, p_target - p_draft) /
        // Z. Per design.md §3.2, when Z = 0 (residual is empty)
        // fall back to sampling directly from p_target.
        let corrected = sample_residual(&p_target[k], d, rng);
        return AcceptedSpan {
            accepted,
            corrected: Some(corrected),
            bonus: None,
        };
    }

    // All-accept: emit one bonus token sampled directly from
    // p_target at the deepest accepted position.
    let last_target = p_target.last().expect("non-empty draft");
    let bonus = sample_categorical(last_target, rng);
    AcceptedSpan {
        accepted,
        corrected: None,
        bonus: Some(bonus),
    }
}

/// Residual sampling: `max(0, p_target - p_draft) / Z`. Falls back
/// to `p_target` when the residual is empty (Z = 0). The draft has
/// only one in-vocab probability — we treat all other positions as
/// p_draft = 0, so the residual collapses to subtracting at the
/// single draft id.
fn sample_residual(p_target: &[f32], d: &DraftToken, rng: &mut VerifyRng) -> TokenId {
    let mut residual = p_target.to_vec();
    let id = d.id as usize;
    let subtract = d.p_draft.min(p_target[id]);
    residual[id] = (residual[id] - subtract).max(0.0);

    let z: f32 = residual.iter().sum();
    if z <= 0.0 {
        // Residual collapsed; fall back to sampling from p_target.
        // This branch is taken when p_target ≤ p_draft on the
        // single id. Important to handle deterministically — the
        // GPU kernel does the same (design.md §3.2).
        return sample_categorical(p_target, rng);
    }

    let u = rng.next_f32() * z;
    let mut acc = 0.0_f32;
    for (i, p) in residual.iter().enumerate() {
        acc += p;
        if u < acc {
            return i as TokenId;
        }
    }
    (residual.len() - 1) as TokenId
}

/// Inverse-CDF categorical sample. Probabilities need not be
/// normalised; we pass `u * sum(p)` so the loop sees the same
/// scale.
fn sample_categorical(p: &[f32], rng: &mut VerifyRng) -> TokenId {
    let total: f32 = p.iter().sum();
    if total <= 0.0 {
        return 0;
    }
    let u = rng.next_f32() * total;
    let mut acc = 0.0_f32;
    for (i, x) in p.iter().enumerate() {
        acc += x;
        if u < acc {
            return i as TokenId;
        }
    }
    (p.len() - 1) as TokenId
}

/// Tree verification — picks the most-likely-by-p_draft root-to-leaf
/// path and applies linear `verify_and_accept` to it. This is the
/// CPU oracle for phase 3's GPU `verify_tree` kernel; the GPU
/// kernel SHALL emit the same `AcceptedSpan` given the same tree,
/// per-node target distributions, and seeded RNG.
///
/// `p_target_per_node[i]` is the target distribution at tree node
/// `i` (so `p_target_per_node.len() == tree.len()`). `tree`'s root
/// is the first drafted token (depth=0); subsequent levels are
/// continuations.
///
/// Rationale: the most-likely path is what production tree-decoders
/// (Medusa, EAGLE) verify first; sibling-path fallback is a phase 4
/// refinement. Restricting phase 3 to single-path verification
/// keeps the kernel small and the parity oracle simple while still
/// delivering the architectural batch=N>1 win on which Tensor Cores
/// re-arm.
pub fn verify_tree(
    tree: &DraftTree,
    p_target_per_node: &[Vec<f32>],
    rng: &mut VerifyRng,
) -> AcceptedSpan {
    assert_eq!(
        p_target_per_node.len(),
        tree.len(),
        "verify_tree: per-node p_target length must equal tree node count"
    );
    let path = tree.most_likely_path();
    let drafts: Vec<DraftToken> = path
        .iter()
        .map(|&i| tree.nodes()[i].token.clone())
        .collect();
    let p_at_path: Vec<Vec<f32>> = path.iter().map(|&i| p_target_per_node[i].clone()).collect();
    verify_and_accept(&p_at_path, &drafts, rng)
}

#[cfg(test)]
mod tests {
    use super::super::tree::DraftTree;
    use super::*;

    fn unit(id: TokenId, vocab: usize) -> Vec<f32> {
        let mut v = vec![0.0; vocab];
        v[id as usize] = 1.0;
        v
    }

    #[test]
    fn all_accept_emits_bonus() {
        // p_target == p_draft on each position → ratio = 1.0 → always accept.
        let vocab = 4;
        let draft = vec![
            DraftToken {
                id: 1,
                p_draft: 1.0,
            },
            DraftToken {
                id: 2,
                p_draft: 1.0,
            },
        ];
        let p_target = vec![unit(1, vocab), unit(2, vocab)];
        let mut rng = VerifyRng::new(0xDEAD_BEEF);
        let span = verify_and_accept(&p_target, &draft, &mut rng);
        assert_eq!(span.accepted, vec![1, 2]);
        assert!(span.corrected.is_none());
        assert_eq!(span.bonus, Some(2));
        assert_eq!(span.emitted_count(), 3);
    }

    #[test]
    fn certain_rejection_emits_corrected_only() {
        // Draft proposes id=0 with p=1.0, but target's mass is on id=1.
        // Acceptance prob = p_target[0]/p_draft[0] = 0/1 = 0 → reject.
        let vocab = 4;
        let draft = vec![DraftToken {
            id: 0,
            p_draft: 1.0,
        }];
        let p_target = vec![unit(1, vocab)];
        let mut rng = VerifyRng::new(0xCAFE_F00D);
        let span = verify_and_accept(&p_target, &draft, &mut rng);
        assert!(span.accepted.is_empty());
        assert_eq!(span.corrected, Some(1));
        assert!(span.bonus.is_none());
    }

    #[test]
    fn partial_accept_then_reject() {
        // First position: certain accept (id matches target one-hot).
        // Second position: certain reject.
        let vocab = 4;
        let draft = vec![
            DraftToken {
                id: 1,
                p_draft: 1.0,
            },
            DraftToken {
                id: 0,
                p_draft: 1.0,
            },
        ];
        let p_target = vec![unit(1, vocab), unit(2, vocab)];
        let mut rng = VerifyRng::new(42);
        let span = verify_and_accept(&p_target, &draft, &mut rng);
        assert_eq!(span.accepted, vec![1]);
        assert_eq!(span.corrected, Some(2));
        assert!(span.bonus.is_none());
    }

    #[test]
    fn residual_zero_falls_back_to_target() {
        // p_target = [0.5, 0.5], draft picks id=0 with p_draft=0.4.
        // Residual = [max(0, 0.5-0.4), 0.5-0] = [0.1, 0.5] — but
        // wait, the residual subtracts only at the draft id. So:
        // residual = [0.1, 0.5], sum = 0.6 > 0 — not the zero case.
        //
        // To force Z=0: need p_target[d.id] ≤ p_draft AND all other
        // p_target positions = 0. p_target = [1.0, 0.0], draft id=0
        // p_draft=1.0. Then residual = [max(0,0), 0] = [0,0], Z=0.
        // We force a rejection by patching the rng to return 1.0.
        // Easier: construct an explicit case via direct call to
        // sample_residual.
        let p_target = vec![1.0, 0.0];
        let d = DraftToken {
            id: 0,
            p_draft: 1.0,
        };
        let mut rng = VerifyRng::new(7);
        let id = sample_residual(&p_target, &d, &mut rng);
        // Residual is zero → falls back to p_target (which is one-hot at 0).
        assert_eq!(id, 0);
    }

    #[test]
    fn rng_is_deterministic_on_seed() {
        let mut a = VerifyRng::new(0x1234_5678);
        let mut b = VerifyRng::new(0x1234_5678);
        for _ in 0..1024 {
            assert_eq!(a.next_f32().to_bits(), b.next_f32().to_bits());
        }
    }

    #[test]
    fn rng_outputs_in_unit_interval() {
        let mut rng = VerifyRng::new(0xF00D_BABE);
        for _ in 0..10_000 {
            let u = rng.next_f32();
            assert!((0.0..1.0).contains(&u), "out of range: {u}");
        }
    }

    #[test]
    fn distributional_parity_at_high_acceptance() {
        // When p_target == p_draft uniformly, every draft is accepted
        // and the bonus is sampled from p_target. Over many seeds,
        // the bonus distribution should approximate p_target.
        let vocab = 8;
        let p: Vec<f32> = (0..vocab).map(|i| (i + 1) as f32).collect();
        let total: f32 = p.iter().sum();
        let p_norm: Vec<f32> = p.iter().map(|x| x / total).collect();
        let p_target = vec![p_norm.clone(); 1];

        let n_trials = 20_000;
        let mut hist = vec![0u32; vocab];
        for seed in 0..n_trials {
            let draft = vec![DraftToken {
                id: 0,
                p_draft: p_norm[0],
            }];
            let mut rng = VerifyRng::new(seed as u64);
            let span = verify_and_accept(&p_target, &draft, &mut rng);
            // Either accepted-with-bonus or rejected-with-corrected;
            // the last emitted token is the one sampled from a target-
            // shaped distribution. Track it.
            let last = *span.tokens().last().unwrap() as usize;
            hist[last] += 1;
        }
        // Compare empirical to expected within 3% absolute (~3σ for n=20k).
        for i in 0..vocab {
            let observed = hist[i] as f32 / n_trials as f32;
            let expected = p_norm[i];
            let diff = (observed - expected).abs();
            assert!(
                diff < 0.03,
                "vocab {i}: observed {observed:.4} vs expected {expected:.4}, diff {diff:.4}"
            );
        }
    }

    fn linear_tree(ids: &[(TokenId, f32)]) -> DraftTree {
        let mut iter = ids.iter();
        let &(rid, rp) = iter.next().expect("non-empty");
        let mut t = DraftTree::from_root(DraftToken {
            id: rid,
            p_draft: rp,
        });
        let mut parent = 0;
        for &(id, p) in iter {
            parent = t.add_child(parent, DraftToken { id, p_draft: p });
        }
        t
    }

    #[test]
    fn verify_tree_linear_matches_verify_and_accept() {
        // Linear tree (depth=2, branches=1). verify_tree must
        // produce the same span as feeding the same drafts straight
        // into verify_and_accept.
        let vocab = 4;
        let drafts = [(1u32, 1.0_f32), (2u32, 1.0_f32), (3u32, 1.0_f32)];
        let tree = linear_tree(&drafts);
        let p_target = vec![unit(1, vocab), unit(2, vocab), unit(3, vocab)];

        let mut rng_a = VerifyRng::new(0xABCD_1234);
        let span_tree = verify_tree(&tree, &p_target, &mut rng_a);

        let mut rng_b = VerifyRng::new(0xABCD_1234);
        let drafts_v: Vec<DraftToken> = drafts
            .iter()
            .map(|&(id, p)| DraftToken { id, p_draft: p })
            .collect();
        let span_linear = verify_and_accept(&p_target, &drafts_v, &mut rng_b);

        assert_eq!(span_tree, span_linear);
    }

    #[test]
    fn verify_tree_picks_highest_p_draft_path() {
        // Tree:
        //   0 (root, drafted as id=1)
        //   ├── 1 (id=2, p_draft=0.3)
        //   └── 2 (id=3, p_draft=0.7)  <-- most likely
        //         ├── 3 (id=4, p_draft=0.6)  <-- most likely under 2
        //         └── 4 (id=5, p_draft=0.4)
        //   └── ...
        // Path verifier picks: 0 → 2 → 3.
        let mut tree = DraftTree::from_root(DraftToken {
            id: 1,
            p_draft: 1.0,
        });
        let _n1 = tree.add_child(
            0,
            DraftToken {
                id: 2,
                p_draft: 0.3,
            },
        );
        let n2 = tree.add_child(
            0,
            DraftToken {
                id: 3,
                p_draft: 0.7,
            },
        );
        let _n3 = tree.add_child(
            n2,
            DraftToken {
                id: 4,
                p_draft: 0.6,
            },
        );
        let _n4 = tree.add_child(
            n2,
            DraftToken {
                id: 5,
                p_draft: 0.4,
            },
        );

        let path = tree.most_likely_path();
        assert_eq!(path, vec![0, n2, _n3]);

        // p_target one-hot at the path's drafted ids → certain accept all.
        let vocab = 8;
        let mut p_target = vec![unit(0, vocab); tree.len()];
        p_target[0] = unit(1, vocab); // root accepts to id=1
        p_target[n2] = unit(3, vocab);
        p_target[_n3] = unit(4, vocab);

        let mut rng = VerifyRng::new(0xFEEDFACE);
        let span = verify_tree(&tree, &p_target, &mut rng);
        assert_eq!(span.accepted, vec![1, 3, 4]);
        assert!(span.bonus.is_some());
    }

    #[test]
    fn verify_tree_rejects_along_path_emits_corrected() {
        // Most-likely path is [0, 1]. p_target rejects at root.
        let vocab = 4;
        let tree = linear_tree(&[(0, 1.0), (1, 1.0)]);
        // Target wants id=2 at root → reject.
        let p_target = vec![unit(2, vocab), unit(1, vocab)];
        let mut rng = VerifyRng::new(0xCAFE_BEEF);
        let span = verify_tree(&tree, &p_target, &mut rng);
        assert!(span.accepted.is_empty());
        assert_eq!(span.corrected, Some(2));
        assert!(span.bonus.is_none());
    }

    #[test]
    fn verify_tree_panics_on_length_mismatch() {
        let tree = linear_tree(&[(0, 1.0), (1, 1.0), (2, 1.0)]); // 3 nodes
        let p_target = vec![unit(0, 4), unit(1, 4)]; // only 2
        let mut rng = VerifyRng::new(0);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            verify_tree(&tree, &p_target, &mut rng);
        }));
        assert!(result.is_err());
    }

    #[test]
    fn verify_tree_distributional_parity_at_depth2() {
        // Linear tree depth=2 with p_draft at the leaf set equal
        // to the target mass at the drafted id, so acceptance prob
        // = 1.0 at every level. Bonus is then sampled from p_target
        // at the deepest accepted node — should match its multinomial
        // shape over many seeds.
        let vocab = 8;
        let p_at_leaf: Vec<f32> = (1..=vocab as u32).map(|x| x as f32).collect();
        let total: f32 = p_at_leaf.iter().sum();
        let p_at_leaf_norm: Vec<f32> = p_at_leaf.iter().map(|x| x / total).collect();
        let drafted_leaf_id: TokenId = 3;
        let p_draft_leaf = p_at_leaf_norm[drafted_leaf_id as usize];

        let tree = linear_tree(&[(1, 1.0), (2, 1.0), (drafted_leaf_id, p_draft_leaf)]);
        let p_target = vec![unit(1, vocab), unit(2, vocab), p_at_leaf_norm.clone()];

        let n_trials = 20_000;
        let mut hist = vec![0u32; vocab];
        for seed in 0..n_trials {
            let mut rng = VerifyRng::new(seed as u64);
            let span = verify_tree(&tree, &p_target, &mut rng);
            assert_eq!(span.accepted, vec![1, 2, drafted_leaf_id]);
            let bonus = span.bonus.expect("all-accept must produce bonus");
            hist[bonus as usize] += 1;
        }
        for i in 0..vocab {
            let observed = hist[i] as f32 / n_trials as f32;
            let expected = p_at_leaf_norm[i];
            let diff = (observed - expected).abs();
            assert!(
                diff < 0.03,
                "vocab {i}: observed {observed:.4} vs expected {expected:.4}"
            );
        }
    }
}
