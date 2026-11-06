//! Incremental detokeniser.
//!
//! HuggingFace tokenizers use a `▁` (U+2581) leading-space convention that
//! prefixes word-initial subwords. Decoding `[▁Paris]` alone gives
//! `"Paris"` — the leading space is stripped because the tokenizer assumes
//! the word starts at position 0. Decoding the full sequence
//! `[The, ▁capital, ▁of, ▁France, ▁is, ▁Paris]` joins correctly into
//! `"The capital of France is Paris"`.
//!
//! [`Detokenizer`] preserves spacing for streaming output by holding the
//! cumulative ID buffer and emitting only the freshly-grown suffix on each
//! `push`. Equivalent semantics to llama.cpp's `llama_token_to_piece` and
//! HF Python's `decode_stream`.
//!
//! Multi-byte UTF-8 characters that straddle a token boundary are held
//! back, never printed in part. A byte-level decoder renders the bytes of
//! an incomplete character as U+FFFD, so a decode that ends in U+FFFD
//! ends in a character that is not finished yet: [`Detokenizer::push`]
//! emits only up to it and keeps it pending. When the next token completes
//! the character, the decode replaces the U+FFFD with it and it is emitted
//! whole; when generation stops first, the pending bytes are simply never
//! emitted. (A model that genuinely generates U+FFFD at the very end of a
//! stream loses it from the *stream*; [`Detokenizer::cumulative`] still has it.)

use tokenizers::Tokenizer;

/// How a decoder renders the bytes of a character that is not complete.
const INCOMPLETE_CHAR: char = char::REPLACEMENT_CHARACTER;

/// Length in bytes of the part of `decoded` that can no longer change:
/// everything before a trailing run of [`INCOMPLETE_CHAR`].
fn stable_len(decoded: &str) -> usize {
    decoded.trim_end_matches(INCOMPLETE_CHAR).len()
}

/// Stateful, single-stream incremental detokeniser.
///
/// One instance per generation call. Not `Sync` — clone the underlying
/// tokenizer if multiple streams are decoded in parallel.
pub struct Detokenizer<'a> {
    tokenizer: &'a Tokenizer,
    skip_special: bool,
    ids: Vec<u32>,
    /// Number of bytes already emitted from the cumulative decoded string.
    emitted: usize,
}

impl<'a> Detokenizer<'a> {
    /// Create a new detokeniser. `skip_special` controls the
    /// `skip_special_tokens` flag passed to the underlying decoder; `true`
    /// matches what every existing call site in the inference crate uses.
    pub fn new(tokenizer: &'a Tokenizer) -> Self {
        Self {
            tokenizer,
            skip_special: true,
            ids: Vec::new(),
            emitted: 0,
        }
    }

    /// Toggle `skip_special_tokens`. Default is `true`.
    pub fn skip_special(mut self, skip: bool) -> Self {
        self.skip_special = skip;
        self
    }

    /// Seed with prompt IDs. Decodes them once to set the byte offset, but
    /// returns nothing — the prompt was input, not generated output. After
    /// seeding, the next [`Detokenizer::push`] returns the first generated
    /// token's surface form *with its leading space* if the tokenizer
    /// rendered one.
    pub fn seed(&mut self, prompt_ids: &[u32]) {
        self.ids.extend_from_slice(prompt_ids);
        self.emitted = self
            .tokenizer
            .decode(&self.ids, self.skip_special)
            .map(|s| stable_len(&s))
            .unwrap_or(0);
    }

    /// Append a new token id and return the freshly-decoded suffix.
    ///
    /// Returns an empty string in two cases:
    /// 1. The decode failed (rare — only seen on tokenizer-level errors).
    /// 2. The token carries part of a multi-byte UTF-8 character that
    ///    is not complete yet — it is emitted whole by the push that
    ///    completes it, or never, if generation stops first.
    pub fn push(&mut self, id: u32) -> String {
        self.ids.push(id);
        let full = match self.tokenizer.decode(&self.ids, self.skip_special) {
            Ok(s) => s,
            Err(_) => return String::new(),
        };
        let stable = stable_len(&full);
        if stable <= self.emitted {
            // Nothing new is final: the token decoded to "" (a special
            // token under skip_special_tokens=true) or only extended an
            // incomplete character.
            return String::new();
        }
        // `emitted` is a char boundary whenever the decode of earlier ids
        // is a prefix of this one; snap forward defensively if a
        // normaliser rewrote earlier text.
        let start = (self.emitted..=stable)
            .find(|&i| full.is_char_boundary(i))
            .unwrap_or(stable);
        let delta = full[start..stable].to_string();
        self.emitted = stable;
        delta
    }

    /// Cumulative decoded string of every token pushed so far (including
    /// the seed). Useful for end-of-stream final readout.
    pub fn cumulative(&self) -> String {
        self.tokenizer
            .decode(&self.ids, self.skip_special)
            .unwrap_or_default()
    }

    /// Tokens accumulated so far (seed + pushed).
    pub fn ids(&self) -> &[u32] {
        &self.ids
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a tiny word-level tokenizer over a fixed vocab via the
    /// JSON-loader (avoids `TokenizerBuilder` generic-inference issues).
    /// Token N decodes back to its word; the WordLevel decoder joins with
    /// single spaces between pre-tokenized chunks.
    fn tiny_tokenizer() -> Tokenizer {
        let vocab = [
            ("[UNK]", 0u32),
            ("the", 1),
            ("capital", 2),
            ("of", 3),
            ("france", 4),
            ("is", 5),
            ("paris", 6),
            ("hello", 7),
            ("world", 8),
        ];
        let mut vocab_json = serde_json::Map::new();
        for (k, v) in vocab {
            vocab_json.insert(k.to_string(), serde_json::Value::Number((v as u64).into()));
        }
        let tokenizer_json = serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": { "type": "Whitespace" },
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": vocab_json,
                "unk_token": "[UNK]"
            }
        });
        let bytes = serde_json::to_vec(&tokenizer_json).expect("json");
        Tokenizer::from_bytes(&bytes).expect("tokenizer build")
    }

    #[test]
    fn empty_detokenizer_produces_no_output_until_push() {
        let tok = tiny_tokenizer();
        let detok = Detokenizer::new(&tok);
        assert_eq!(detok.cumulative(), "");
        assert!(detok.ids().is_empty());
    }

    #[test]
    fn push_emits_increasing_suffix() {
        let tok = tiny_tokenizer();
        let mut detok = Detokenizer::new(&tok);
        let a = detok.push(1); // "the"
        let b = detok.push(2); // "capital"
        let c = detok.push(3); // "of"
                               // WordLevel + Whitespace decode joins with single spaces.
        assert_eq!(a, "the");
        assert!(b.contains("capital"));
        assert!(c.contains("of"));
        assert_eq!(detok.cumulative(), "the capital of");
    }

    #[test]
    fn seed_does_not_emit_prompt() {
        let tok = tiny_tokenizer();
        let mut detok = Detokenizer::new(&tok);
        detok.seed(&[1, 2, 3]); // "the capital of"
        assert!(detok.cumulative().starts_with("the capital of"));
        let next = detok.push(4); // "france"
                                  // First emit after seeding must contain only the new token's surface.
        assert!(!next.contains("the"));
        assert!(next.contains("france"));
    }

    #[test]
    fn cumulative_matches_full_decode() {
        let tok = tiny_tokenizer();
        let mut detok = Detokenizer::new(&tok);
        for id in [7u32, 8, 1, 2] {
            detok.push(id);
        }
        let direct = tok.decode(&[7u32, 8, 1, 2], true).unwrap();
        assert_eq!(detok.cumulative(), direct);
    }

    #[test]
    fn ids_tracked() {
        let tok = tiny_tokenizer();
        let mut detok = Detokenizer::new(&tok);
        detok.seed(&[1, 2]);
        detok.push(3);
        assert_eq!(detok.ids(), &[1u32, 2, 3]);
    }

    #[test]
    fn unknown_token_does_not_panic() {
        let tok = tiny_tokenizer();
        let mut detok = Detokenizer::new(&tok);
        // 9999 is out of vocab — decoder should handle gracefully.
        let _ = detok.push(9999);
    }
}
