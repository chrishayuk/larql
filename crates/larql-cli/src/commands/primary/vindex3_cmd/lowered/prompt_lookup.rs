//! Prompt-lookup proposals for VERIFY-N: guess the next tokens by copying
//! what followed an earlier occurrence, in the PROMPT, of the context's
//! trailing n-gram. Costs no model and no training; it pays exactly where
//! the output repeats its input — edits, rewrites, structured transforms.
//!
//! The rules were chosen by replaying the engine's own greedy ids for
//! Gemma 3 4B under each candidate rule (exact for greedy):
//!
//! - **Prompt only.** On an edit the source is the prompt; the output's
//!   own recent n-grams mislead, and on free prose they are the only
//!   matches there are — which never verify. Prompt-only lookup lifted
//!   N=8 edit acceptance and took free prose from 0.91x to 1.00x.
//! - **Longest suffix first, up to [`MAX_NGRAM`].** A longer match pins
//!   the right place in the source; 16 recovered everything 64 did.
//! - **Earliest occurrence.** The first place a passage appears is the
//!   source being edited; later repeats are more often boilerplate.
//! - **n >= [`MIN_NGRAM`].** A unigram match is almost never accepted.

/// Shortest trailing n-gram that may propose.
pub const MIN_NGRAM: usize = 2;
/// Longest trailing n-gram tried first.
pub const MAX_NGRAM: usize = 16;

/// Up to `max` tokens that followed the earliest occurrence, inside
/// `ctx[..source_len]` (the prompt), of `ctx`'s longest trailing n-gram;
/// the proposal never runs past the source. Empty when nothing matches.
pub fn propose(ctx: &[u32], source_len: usize, max: usize) -> Vec<u32> {
    let source_len = source_len.min(ctx.len());
    if max == 0 {
        return Vec::new();
    }
    for n in (MIN_NGRAM..=MAX_NGRAM.min(ctx.len().saturating_sub(1))).rev() {
        let tail = &ctx[ctx.len() - n..];
        // An occurrence must leave at least one source token after it.
        let Some(last_start) = source_len.checked_sub(n + 1) else {
            continue;
        };
        for start in 0..=last_start {
            if &ctx[start..start + n] == tail {
                let from = start + n;
                return ctx[from..(from + max).min(source_len)].to_vec();
            }
        }
    }
    Vec::new()
}

/// The number of leading `guess` tokens the target's greedy `ids` agree
/// with. `ids[i]` is the target's id after the block's first `i + 1`
/// tokens, and `guess[i]` is block token `i + 1`.
pub fn accepted_prefix(guess: &[u32], ids: &[u32]) -> usize {
    guess.iter().zip(ids).take_while(|(g, id)| g == id).count()
}
