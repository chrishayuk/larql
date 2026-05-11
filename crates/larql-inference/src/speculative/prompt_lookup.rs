//! Prompt-lookup decoding (PLD) drafter — phase 4d quick-win, no-training.
//!
//! Idea: for many real workloads (RAG, code completion, instruction-
//! following that quotes the prompt, structured output, repetitive
//! generation) the target model's next tokens are likely to be a
//! continuation that ALREADY appeared somewhere earlier in the
//! prompt + accepted span. Instead of running a small model to
//! propose drafts, we **search the existing token history for an
//! n-gram match** to the current suffix and propose the tokens that
//! followed that match.
//!
//! ## Algorithm
//!
//! - Maintain a rolling token history (prompt + accepted span so far).
//! - On `propose(_, n)`:
//!   1. Take the last `suffix_len` tokens of history as the lookup key.
//!   2. Search earlier history for that key. Pick the most-recent
//!      match (rightmost, so most relevant to current context).
//!   3. The `n` tokens that followed the match become the drafts.
//! - If no match found, return empty (caller falls through to the
//!   non-speculative path for this iter).
//!
//! ## When this works (high α)
//!
//! - **RAG**: the model often quotes passages from the retrieved
//!   context. Each quoted token has its continuation right there in
//!   the prompt → high accept rate.
//! - **Instruction-following / chat**: responses often echo or
//!   restate the user's question fragments.
//! - **Code completion**: variable names, function signatures, and
//!   import paths are repeated across the file.
//! - **Structured output (JSON, YAML, lists)**: keys, brackets, and
//!   delimiters are highly repetitive.
//!
//! ## When this fails (α drops to 0)
//!
//! - Open-ended creative generation with no prompt repetition.
//! - First-of-its-kind tokens (proper nouns introduced fresh).
//! - In those cases, `propose` returns empty and the dispatch falls
//!   through to plain decode for that iter — zero overhead beyond
//!   the lookup itself (~µs for short histories, ms-ish for very
//!   long contexts).
//!
//! ## Tradeoff vs other no-training options
//!
//! - vs **standalone small drafter** (e.g. Gemma 3 270M): PLD has
//!   zero per-token GPU cost during propose (just CPU index lookup).
//!   Drafter has 0 model parameters to load. Trade-off: PLD's
//!   coverage is workload-dependent — empty propose on novel content.
//! - vs **lookahead decoding** (Jacobi iteration): PLD is much
//!   simpler (~150 LoC vs ~500 LoC) and faster to propose. Lookahead
//!   is universal; PLD only wins on prompt-echoing workloads.
//!
//! Inspired by the prompt-lookup-decoding implementation in vLLM
//! and the technique described in
//! https://github.com/apoorvumang/prompt-lookup-decoding.

use super::tree::DraftTree;
use super::{DraftToken, Drafter, TokenId};

/// Default suffix length used by `PromptLookupDrafter::new()`.
/// Empirically, 2-3 is the sweet spot — shorter (1) gives many false
/// matches; longer (5+) starves the lookup on short contexts.
const DEFAULT_SUFFIX_LEN: usize = 2;

/// Maximum number of tokens to look back in the search. Bounds the
/// per-propose lookup cost; set conservatively for typical chat
/// histories. Hard limit to avoid pathological linear scans on
/// very long contexts.
const DEFAULT_LOOKBACK_LIMIT: usize = 4096;

/// Off-the-shelf prompt-lookup drafter. Maintains its own token
/// history; `propose()` searches that history for an n-gram match
/// to the current suffix and returns the continuation.
#[derive(Clone, Debug)]
pub struct PromptLookupDrafter {
    history: Vec<TokenId>,
    /// Length of the suffix (n-gram) to match against history.
    /// 1 = unigram (fast, many false matches), 3 = trigram (fewer
    /// matches but more confident). Default `DEFAULT_SUFFIX_LEN`.
    suffix_len: usize,
    /// Maximum lookback distance from the end of history. Bounds
    /// per-propose CPU cost on long contexts.
    lookback_limit: usize,
}

impl PromptLookupDrafter {
    /// Construct with default suffix length and lookback limit.
    pub fn new() -> Self {
        Self::with_config(DEFAULT_SUFFIX_LEN, DEFAULT_LOOKBACK_LIMIT)
    }

    pub fn with_config(suffix_len: usize, lookback_limit: usize) -> Self {
        Self {
            history: Vec::new(),
            suffix_len: suffix_len.max(1),
            lookback_limit,
        }
    }

    /// Current history length (= prompt + accepted span tokens).
    pub fn history_len(&self) -> usize {
        self.history.len()
    }

    /// Search history for the most-recent (rightmost) match of the
    /// last `suffix_len` tokens. Returns the slice immediately
    /// following the match (up to `n` tokens). `None` if no match.
    fn lookup_continuation(&self, n: usize) -> Option<&[TokenId]> {
        self.lookup_continuations(n, 1).into_iter().next()
    }

    /// Search history for up to `branches` most-recent (rightmost)
    /// distinct matches of the last `suffix_len` tokens. Returns one
    /// continuation slice per match (each up to `n` tokens), ordered
    /// rightmost-first. Duplicate continuations (token-equal to an
    /// earlier match's continuation) are filtered out so each entry
    /// represents a genuinely distinct candidate path. Returns an
    /// empty Vec on no matches.
    fn lookup_continuations(&self, n: usize, branches: usize) -> Vec<&[TokenId]> {
        if n == 0 || branches == 0 {
            return Vec::new();
        }
        let h = &self.history;
        let needle_start = match h.len().checked_sub(self.suffix_len) {
            Some(v) => v,
            None => return Vec::new(),
        };
        let suffix = &h[needle_start..];
        let scan_start = h.len().saturating_sub(self.lookback_limit);
        if needle_start <= scan_start {
            return Vec::new();
        }
        let mut out: Vec<&[TokenId]> = Vec::with_capacity(branches);
        for cand in (scan_start..needle_start).rev() {
            if out.len() >= branches {
                break;
            }
            if cand + self.suffix_len > h.len() {
                continue;
            }
            if h[cand..cand + self.suffix_len] != *suffix {
                continue;
            }
            let cont_start = cand + self.suffix_len;
            if cont_start >= h.len() {
                continue;
            }
            let take = (h.len() - cont_start).min(n);
            if take == 0 {
                continue;
            }
            let cont = &h[cont_start..cont_start + take];
            // Dedupe: an earlier (more recent) match may yield the
            // same continuation, which would be a wasted branch.
            if out.contains(&cont) {
                continue;
            }
            out.push(cont);
        }
        out
    }
}

impl Default for PromptLookupDrafter {
    fn default() -> Self {
        Self::new()
    }
}

impl Drafter for PromptLookupDrafter {
    fn propose(&mut self, _h_target: &[f32], n: usize) -> Vec<DraftToken> {
        if n == 0 {
            return Vec::new();
        }
        let cont = match self.lookup_continuation(n) {
            Some(c) => c,
            None => return Vec::new(),
        };
        // p_draft = 1.0 means the verifier's accept ratio
        // (p_target / p_draft) reduces to just p_target — the verifier
        // accepts each draft with probability equal to the target's
        // prob for that token. Conservative and correct: drafts that
        // happen to match the target's argmax (high p_target) are
        // accepted; drafts that don't (low p_target) are rejected
        // without bias from a fabricated p_draft.
        cont.iter()
            .map(|&id| DraftToken { id, p_draft: 1.0 })
            .collect()
    }

    fn reset(&mut self) {
        self.history.clear();
    }

    fn accept(&mut self, accepted: &[TokenId]) {
        self.history.extend_from_slice(accepted);
    }

    fn seed_history(&mut self, tokens: &[TokenId]) {
        // Same prefix-extension fast path as `SmallModelDrafter` —
        // the v3 dispatch helper calls this every iter.
        if tokens.len() >= self.history.len() && tokens[..self.history.len()] == self.history[..] {
            self.history
                .extend_from_slice(&tokens[self.history.len()..]);
            return;
        }
        self.history.clear();
        self.history.extend_from_slice(tokens);
    }

    fn propose_tree(
        &mut self,
        _h_target: &[f32],
        depth: usize,
        branches: usize,
    ) -> Option<DraftTree> {
        if depth == 0 || branches == 0 {
            return None;
        }
        let conts = self.lookup_continuations(depth, branches);
        if conts.is_empty() {
            return None;
        }

        // Root = rightmost match's first continuation token. Chains
        // that start with a different token cannot merge under this
        // root (DraftTree has a single root), so they're dropped.
        // This keeps the contract simple and matches what verifier
        // expects: the root is the drafted token immediately after
        // the current target hidden state.
        let root_id = conts[0][0];
        let mut tree = DraftTree::from_root(DraftToken {
            id: root_id,
            p_draft: 1.0,
        });

        for cont in conts.iter() {
            if cont[0] != root_id {
                continue;
            }
            // Walk down from root, descending into the child whose
            // token matches `tok`; when no such child exists, branch
            // off by appending a new child. Subsequent tokens hang
            // off that new child as a linear tail.
            let mut cur = 0usize;
            for &tok in &cont[1..] {
                let existing = (0..tree.nodes().len()).find(|&i| {
                    tree.nodes()[i].parent == Some(cur) && tree.nodes()[i].token.id == tok
                });
                cur = match existing {
                    Some(c) => c,
                    None => tree.add_child(
                        cur,
                        DraftToken {
                            id: tok,
                            p_draft: 1.0,
                        },
                    ),
                };
            }
        }
        Some(tree)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_history_returns_no_drafts() {
        let mut d = PromptLookupDrafter::new();
        let drafts = d.propose(&[], 4);
        assert!(drafts.is_empty());
    }

    #[test]
    fn no_match_returns_no_drafts() {
        let mut d = PromptLookupDrafter::with_config(2, 1024);
        d.seed_history(&[1, 2, 3, 4, 5]);
        // Suffix [4, 5] never appeared earlier in history — no match.
        let drafts = d.propose(&[], 3);
        assert!(drafts.is_empty());
    }

    #[test]
    fn finds_continuation_at_simplest_repetition() {
        let mut d = PromptLookupDrafter::with_config(2, 1024);
        // History: 1 2 3 4 1 2
        // Suffix (last 2): [1, 2]
        // Earlier match: positions [0, 1] = [1, 2]
        // Continuation (extends to end of history, per apoorvumang ref):
        // positions [2..6] = [3, 4, 1, 2] — predicts the cycle repeats.
        d.seed_history(&[1, 2, 3, 4, 1, 2]);
        let drafts = d.propose(&[], 4);
        let ids: Vec<u32> = drafts.iter().map(|d| d.id).collect();
        assert_eq!(ids, vec![3, 4, 1, 2]);
    }

    #[test]
    fn picks_most_recent_match_when_multiple() {
        let mut d = PromptLookupDrafter::with_config(2, 1024);
        // History: 9 9 7 8 5 5 7 8 1 2 7 8
        // Suffix [7, 8] appears at positions 2, 6, 10.
        // We want the MOST RECENT (rightmost before suffix) =
        // position 6 → continuation [1, 2, 7, 8] truncated to 4.
        d.seed_history(&[9, 9, 7, 8, 5, 5, 7, 8, 1, 2, 7, 8]);
        let drafts = d.propose(&[], 4);
        let ids: Vec<u32> = drafts.iter().map(|d| d.id).collect();
        assert_eq!(ids, vec![1, 2, 7, 8]);
    }

    #[test]
    fn accept_extends_history_and_subsequent_propose_uses_extended() {
        let mut d = PromptLookupDrafter::with_config(2, 1024);
        d.seed_history(&[1, 2, 3]);
        d.accept(&[1, 2]);
        // History now: 1 2 3 1 2
        // Suffix [1, 2] matches at position 0; continuation extends to
        // end of history → [3, 1, 2].
        let drafts = d.propose(&[], 3);
        let ids: Vec<u32> = drafts.iter().map(|d| d.id).collect();
        assert_eq!(ids, vec![3, 1, 2]);
    }

    #[test]
    fn seed_history_prefix_extension_keeps_history() {
        let mut d = PromptLookupDrafter::with_config(2, 1024);
        d.seed_history(&[1, 2, 3]);
        // Prefix-extend: same prefix + new tail.
        d.seed_history(&[1, 2, 3, 4, 5]);
        assert_eq!(d.history, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn seed_history_divergence_resets() {
        let mut d = PromptLookupDrafter::with_config(2, 1024);
        d.seed_history(&[1, 2, 3]);
        // Different prefix — should reset.
        d.seed_history(&[7, 8, 9]);
        assert_eq!(d.history, vec![7, 8, 9]);
    }

    #[test]
    fn reset_clears_history() {
        let mut d = PromptLookupDrafter::new();
        d.seed_history(&[1, 2, 3]);
        d.reset();
        assert_eq!(d.history_len(), 0);
    }

    #[test]
    fn lookback_limit_bounds_search() {
        // History length 5000, but lookback_limit=10. The match at
        // position 0 is OUT of the lookback window → no match.
        let mut d = PromptLookupDrafter::with_config(2, 10);
        let mut hist = vec![1u32, 2, 99, 99, 99]; // 1 2 at start
        hist.extend(std::iter::repeat_n(0u32, 4990));
        hist.extend([1, 2]); // suffix
        d.seed_history(&hist);
        let drafts = d.propose(&[], 4);
        assert!(
            drafts.is_empty(),
            "match outside lookback window must not be returned"
        );
    }

    // ------------------------------------------------------------
    // PLD-tree (propose_tree) tests
    // ------------------------------------------------------------

    fn collect_children(tree: &DraftTree, parent: usize) -> Vec<(usize, TokenId)> {
        tree.nodes()
            .iter()
            .enumerate()
            .filter(|(_, n)| n.parent == Some(parent))
            .map(|(i, n)| (i, n.token.id))
            .collect()
    }

    #[test]
    fn propose_tree_empty_history_returns_none() {
        let mut d = PromptLookupDrafter::new();
        assert!(d.propose_tree(&[], 4, 2).is_none());
    }

    #[test]
    fn propose_tree_branches_one_is_linear_chain() {
        // branches=1 SHALL produce a tree bit-identical to
        // build_linear_tree on the same PLD continuation.
        let mut d = PromptLookupDrafter::with_config(2, 1024);
        d.seed_history(&[1, 2, 3, 4, 1, 2]);

        let mut d_copy = d.clone();
        let linear_drafts = d.propose(&[], 4);
        let linear_tree = super::super::build_linear_tree(&linear_drafts);

        let tree = d_copy
            .propose_tree(&[], 4, 1)
            .expect("non-empty continuation");

        // Same node count, same flat token sequence, same depths.
        assert_eq!(tree.len(), linear_tree.len());
        assert_eq!(tree.tokens(), linear_tree.tokens());
        for i in 0..tree.len() {
            assert_eq!(tree.nodes()[i].parent, linear_tree.nodes()[i].parent);
            assert_eq!(tree.nodes()[i].depth, linear_tree.nodes()[i].depth);
        }
    }

    #[test]
    fn propose_tree_single_match_degrades_to_linear() {
        // Suffix matches in only one place → tree is a depth-N chain.
        let mut d = PromptLookupDrafter::with_config(2, 1024);
        d.seed_history(&[7, 8, 9, 10, 7, 8]);
        // Match at pos 0 → continuation [9, 10, 7, 8]. Only one match
        // (the suffix itself is excluded), so branches=2 still yields
        // a linear chain.
        let tree = d
            .propose_tree(&[], 4, 2)
            .expect("non-empty single-match tree");
        let paths = tree.root_to_leaf_paths();
        assert_eq!(paths.len(), 1, "single match must yield exactly one path");
        assert_eq!(tree.tokens(), vec![9, 10, 7, 8]);
        assert_eq!(tree.max_depth(), 3);
    }

    #[test]
    fn propose_tree_two_matches_disjoint_after_root() {
        // Two matches that share their FIRST continuation token but
        // diverge after that. PLD-tree merges the shared root and
        // branches at depth 1.
        //
        // History laid out so the suffix [1, 2] appears at positions
        // 0 and 4, with different continuations after one shared token:
        //   pos 0: 1 2 5 7 ...   → cont = [5, 7]
        //   pos 4: 1 2 5 6 1 2   → cont = [5, 6]  (rightmost)
        //   suffix at pos 6 = [1, 2] (excluded from itself)
        let mut d = PromptLookupDrafter::with_config(2, 1024);
        d.seed_history(&[1, 2, 5, 7, 1, 2, 5, 6, 1, 2]);
        let tree = d.propose_tree(&[], 2, 2).expect("two matches available");

        // Root = 5 (shared first token across both matches).
        assert_eq!(tree.root().token.id, 5);
        // Two depth-1 children: 6 (from rightmost match) and 7 (from
        // older match). Order of insertion is rightmost-first.
        let kids = collect_children(&tree, 0);
        let kid_ids: Vec<TokenId> = kids.iter().map(|(_, id)| *id).collect();
        assert_eq!(kid_ids, vec![6, 7], "rightmost match's child added first");
        // Total: 1 root + 2 children = 3 nodes (no shared-prefix
        // beyond the root, no merging beyond depth 1).
        assert_eq!(tree.len(), 3);
    }

    #[test]
    fn propose_tree_two_matches_shared_prefix_merge() {
        // Two matches share TWO continuation tokens, then diverge.
        // History: [1, 2, 9, 8, 7, 1, 2, 9, 8, 6, 1, 2]
        //   suffix at pos 10 = [1, 2]
        //   match at pos 5: cont = [9, 8, 6]  (rightmost)
        //   match at pos 0: cont = [9, 8, 7]
        //   shared prefix: [9, 8]; diverge at depth 2 (6 vs 7).
        let mut d = PromptLookupDrafter::with_config(2, 1024);
        d.seed_history(&[1, 2, 9, 8, 7, 1, 2, 9, 8, 6, 1, 2]);
        let tree = d.propose_tree(&[], 3, 2).expect("two matches");

        assert_eq!(tree.root().token.id, 9, "shared root token");
        // Depth-1: only one child (8), because both chains share it.
        let depth1 = collect_children(&tree, 0);
        assert_eq!(depth1.len(), 1);
        assert_eq!(depth1[0].1, 8);
        // Depth-2: two siblings (6 and 7), both children of node "8".
        let depth1_idx = depth1[0].0;
        let depth2 = collect_children(&tree, depth1_idx);
        let depth2_ids: Vec<TokenId> = depth2.iter().map(|(_, id)| *id).collect();
        assert_eq!(depth2_ids, vec![6, 7]);
        // Total: 1 root + 1 + 2 = 4 nodes (vs 6 without merging).
        assert_eq!(tree.len(), 4);
    }

    #[test]
    fn propose_tree_drops_chains_with_different_root() {
        // Two matches with DIFFERENT first continuation tokens cannot
        // merge under one DraftTree root. The rightmost match wins;
        // the other is dropped (degrades to a single-chain tree).
        // History: [1, 2, 4, 5, 1, 2, 3, 7, 1, 2]
        //   suffix at pos 8 = [1, 2]
        //   match at pos 4: cont = [3, 7]  (rightmost)
        //   match at pos 0: cont = [4, 5]  (different first token)
        let mut d = PromptLookupDrafter::with_config(2, 1024);
        d.seed_history(&[1, 2, 4, 5, 1, 2, 3, 7, 1, 2]);
        let tree = d
            .propose_tree(&[], 2, 4)
            .expect("at least one match exists");

        assert_eq!(tree.root().token.id, 3, "rightmost match's first token");
        // Only the rightmost chain is in the tree.
        let paths = tree.root_to_leaf_paths();
        assert_eq!(paths.len(), 1, "non-matching root drops second chain");
        assert_eq!(tree.tokens(), vec![3, 7]);
    }

    #[test]
    fn propose_tree_dedupes_identical_continuations() {
        // Two matches with bit-identical continuations dedupe — we
        // don't waste a branch slot on a repeat. Result is a single
        // linear chain.
        // History: [1, 2, 5, 6, 9, 9, 1, 2, 5, 6, 1, 2]
        //   suffix at pos 10 = [1, 2]
        //   match at pos 6: cont = [5, 6]   (rightmost)
        //   match at pos 0: cont = [5, 6]   (same → dedup'd)
        let mut d = PromptLookupDrafter::with_config(2, 1024);
        d.seed_history(&[1, 2, 5, 6, 9, 9, 1, 2, 5, 6, 1, 2]);
        let tree = d.propose_tree(&[], 2, 2).expect("matches exist");
        let paths = tree.root_to_leaf_paths();
        assert_eq!(paths.len(), 1);
        assert_eq!(tree.tokens(), vec![5, 6]);
        assert_eq!(tree.len(), 2);
    }

    #[test]
    fn propose_tree_caps_at_branches() {
        // Three distinct matches with shared root but disjoint depth-1
        // tokens. branches=2 means we only take the first two.
        // History: [1, 2, 9, 4, 1, 2, 9, 5, 1, 2, 9, 6, 1, 2]
        //   suffix at pos 12 = [1, 2]
        //   match at pos 8:  cont = [9, 6]   (rightmost)
        //   match at pos 4:  cont = [9, 5]
        //   match at pos 0:  cont = [9, 4]   (dropped at branches=2)
        let mut d = PromptLookupDrafter::with_config(2, 1024);
        d.seed_history(&[1, 2, 9, 4, 1, 2, 9, 5, 1, 2, 9, 6, 1, 2]);
        let tree = d.propose_tree(&[], 2, 2).expect("three matches");
        let depth1 = collect_children(&tree, 0);
        let ids: Vec<TokenId> = depth1.iter().map(|(_, id)| *id).collect();
        assert_eq!(ids, vec![6, 5], "branches=2 keeps rightmost two");
        assert_eq!(tree.len(), 3, "1 root + 2 depth-1 children");
    }

    #[test]
    fn propose_tree_branches_zero_returns_none() {
        // branches=0 is a no-op — caller bug, but degrade gracefully.
        let mut d = PromptLookupDrafter::with_config(2, 1024);
        d.seed_history(&[1, 2, 3, 4, 1, 2]);
        assert!(d.propose_tree(&[], 4, 0).is_none());
    }

    #[test]
    fn propose_tree_branching_parity_256_synthetic_prompts() {
        // `cuda-spec-branching-tree` T4.2: the parity contract for
        // PLD-tree against PLD-linear. The branching tree is purely
        // **additive** on top of the rightmost match — never replaces
        // it. Concretely:
        //
        //   For any history H, `propose_tree(depth, branches=2)`
        //   builds the rightmost match's continuation as the first
        //   chain (nodes [0..L]), then adds nodes from older matches
        //   only via merge-or-branch onto that first chain. The
        //   subtree rooted at node 0 always contains the linear
        //   chain as a depth-first root-to-leaf prefix.
        //
        // We verify two invariants on 256 synthetic prompts:
        //   (a) The first `L` nodes of the branching tree (where
        //       L = linear_tree.len()) have the same tokens as the
        //       linear tree (same insertion order = same indices).
        //   (b) Branching's node count is ≥ linear's node count —
        //       never fewer nodes.
        //
        // We also confirm we're exercising the multi-match code path
        // by counting how many prompts produced a branching shape.
        let mut linear_only = 0usize;
        let mut branching_count = 0usize;
        for seed in 0u64..256 {
            // 32 tokens drawn from a tiny vocab (mod 8) so n-gram
            // matches are common enough to bias toward repetition.
            let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let mut hist = Vec::with_capacity(32);
            for _ in 0..32 {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                hist.push((s & 0x7) as TokenId);
            }

            let mut linear = PromptLookupDrafter::with_config(2, 4096);
            linear.seed_history(&hist);
            let tree_linear = match linear.propose_tree(&[], 4, 1) {
                Some(t) => t,
                None => continue, // No match — both paths trivially equal.
            };

            let mut branching = PromptLookupDrafter::with_config(2, 4096);
            branching.seed_history(&hist);
            let tree_branch = match branching.propose_tree(&[], 4, 2) {
                Some(t) => t,
                None => {
                    panic!("branches=2 returned None where branches=1 found a match (seed={seed})")
                }
            };

            // Invariant (a): first L nodes of branching == linear.
            let linear_tokens = tree_linear.tokens();
            let l = linear_tokens.len();
            let branch_prefix: Vec<TokenId> = tree_branch
                .nodes()
                .iter()
                .take(l)
                .map(|n| n.token.id)
                .collect();
            assert_eq!(
                linear_tokens,
                branch_prefix,
                "seed={seed}: branching tree's first {l} nodes don't match the linear chain. \
                 hist={hist:?} linear={linear_tokens:?} branch_prefix={branch_prefix:?} \
                 branch_all={:?}",
                tree_branch.tokens()
            );

            // Invariant (b): branching never shrinks below linear.
            assert!(
                tree_branch.len() >= tree_linear.len(),
                "seed={seed}: branching tree shrunk below linear ({} < {})",
                tree_branch.len(),
                tree_linear.len()
            );

            if tree_branch.root_to_leaf_paths().len() == 1 && tree_branch.len() == tree_linear.len()
            {
                linear_only += 1;
            } else {
                branching_count += 1;
            }
        }

        // We expect SOME branching trees on synthetic repetitive
        // inputs. Zero means the test isn't exercising the multi-
        // match path.
        assert!(
            branching_count > 0,
            "256 synthetic prompts produced no branching shapes \
             (linear_only={linear_only}) — test isn't exercising propose_tree's multi-match path"
        );
    }

    #[test]
    fn proposes_at_most_n_tokens() {
        let mut d = PromptLookupDrafter::with_config(2, 1024);
        // Long continuation after match.
        d.seed_history(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2]);
        let drafts = d.propose(&[], 3);
        let ids: Vec<u32> = drafts.iter().map(|d| d.id).collect();
        // Match at position 0 → continuation 3,4,5; only need 3.
        assert_eq!(ids.len(), 3);
        assert_eq!(ids, vec![3, 4, 5]);
    }
}
