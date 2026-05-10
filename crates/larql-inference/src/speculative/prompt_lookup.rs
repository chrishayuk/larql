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
        let h = &self.history;
        let needle_start = h.len().checked_sub(self.suffix_len)?;
        let suffix = &h[needle_start..];
        // Earliest start position to consider — bound by
        // lookback_limit so we don't scan a 32k-token context every
        // propose call.
        let scan_start = h.len().saturating_sub(self.lookback_limit);
        // Iterate candidate match starts from rightmost to leftmost
        // within the lookback window. Stop at `needle_start - 1` so
        // we never match the suffix against itself.
        if needle_start <= scan_start {
            return None;
        }
        for cand in (scan_start..needle_start).rev() {
            if cand + self.suffix_len > h.len() {
                continue;
            }
            if h[cand..cand + self.suffix_len] == *suffix {
                let cont_start = cand + self.suffix_len;
                if cont_start >= h.len() {
                    continue;
                }
                let take = (h.len() - cont_start).min(n);
                if take > 0 {
                    return Some(&h[cont_start..cont_start + take]);
                }
            }
        }
        None
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
