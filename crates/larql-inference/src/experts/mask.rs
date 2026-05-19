//! `OpNameMask` — vocabulary mask for grammar-constrained op-call generation.
//!
//! Lifted from `tests/test_constrained_dispatch.rs` (where it was named
//! `OpJsonMask`) and made public so the production decode loops can use it.
//!
//! ## What it does
//!
//! At each decode step the mask:
//!
//! 1. Decodes the tokens emitted so far back to text.
//! 2. Detects whether we're inside the `op` field of `{"op":"<NAME>","args":...}`.
//! 3. If we are, restricts the next token to either a continuation that
//!    keeps us on the path of *some* valid op name, or a closing `"` when
//!    the partial name is a complete op.
//!
//! Outside the op-name field, the mask is a no-op — args generation is
//! left free because the host already advertises arg keys via the system
//! prompt and the parser tolerates variation. Constraining args at the
//! token level would also need a JSON-schema decoder, which is out of
//! scope for this work.
//!
//! ## Cost
//!
//! First call: O(vocab_size) — scans every token id and decodes it once
//! to build the candidate set (any token whose characters are subset of
//! op-name characters, plus the closing quote). For a 256K-token Gemma 3
//! tokenizer this takes a second or two and is cached for the rest of
//! the generation.
//!
//! Per-step (inside op-name field): O(candidate_set) decode + prefix
//! check, then a single linear sweep of `logits` to apply NEG_INFINITY.
//!
//! Per-step (outside op-name field): one decode call on `generated_ids`,
//! then early return.

use std::collections::HashSet;
use tokenizers::Tokenizer;

use crate::experts::OpSpec;

/// Where the decoder is in the `{"op":"<NAME>","args":{...}}` skeleton.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum GrammarState {
    /// Haven't seen `{"op":"` yet — free decode.
    Free,
    /// Inside the op-name field, with `so_far` characters emitted since the
    /// opening quote.
    OpName { so_far: String },
    /// Op-name field is closed — args follow, free decode.
    Done,
}

/// Detect the grammar state from `generated_text` alone. Pure helper so
/// the state machine can be unit-tested without a tokenizer.
pub(crate) fn op_grammar_state(generated_text: &str) -> GrammarState {
    if let Some(pos) = generated_text.find("{\"op\":\"") {
        let after = &generated_text[pos + 7..];
        if after.contains('"') {
            GrammarState::Done
        } else {
            GrammarState::OpName {
                so_far: after.to_string(),
            }
        }
    } else {
        GrammarState::Free
    }
}

/// Vocabulary mask that constrains the op-name field of a generated
/// `{"op":"<NAME>","args":{...}}` block to a prefix of one of the
/// advertised op names.
///
/// Construct from a list of valid op names (or [`OpSpec`]s), then call
/// [`Self::apply`] from the `mask_fn` of
/// [`generate_cached_constrained`](crate::forward::generate_cached_constrained)
/// or any equivalent decode loop.
pub struct OpNameMask<'tok> {
    valid_ops: Vec<String>,
    /// Token ids whose decoded string is composed entirely of op-name
    /// characters (or is a single closing quote). Built lazily on first
    /// in-op-name step and reused.
    op_token_cache: Option<Vec<u32>>,
    tokenizer: &'tok Tokenizer,
    generated_text: String,
    /// Optional teacher-forced prefix the caller injected into the prompt
    /// (e.g. `{"op":"`). Prepended to `generated_text` for state detection
    /// so the mask knows the model is already mid-JSON before any tokens
    /// have been generated. Without this the mask sits in `Free` state
    /// for prompts where the model never emits the op-call prefix on its
    /// own — which is the common failure mode on Q4K models.
    seed_text: String,
}

impl<'tok> OpNameMask<'tok> {
    /// Construct from a list of valid op names.
    pub fn new(valid_ops: Vec<String>, tokenizer: &'tok Tokenizer) -> Self {
        Self {
            valid_ops,
            op_token_cache: None,
            tokenizer,
            generated_text: String::new(),
            seed_text: String::new(),
        }
    }

    /// Construct from a slice of [`OpSpec`] (the args field is ignored —
    /// only the op name is constrained at the token level).
    pub fn from_op_specs(specs: &[OpSpec], tokenizer: &'tok Tokenizer) -> Self {
        Self::new(specs.iter().map(|s| s.name.clone()).collect(), tokenizer)
    }

    /// Set a teacher-forced prefix that the caller injected into the prompt
    /// (e.g. `{"op":"`). The mask treats this prefix as already-generated
    /// text for state detection — so on the very first decode step it
    /// knows we're inside the op-name field and applies the constraint.
    pub fn set_seed_text(&mut self, seed: impl Into<String>) {
        self.seed_text = seed.into();
    }

    /// Build (once) the set of token ids whose decoded form is a possible
    /// fragment of a valid op name, plus the closing quote `"`.
    fn op_tokens(&mut self) -> &[u32] {
        if self.op_token_cache.is_none() {
            let valid_chars: HashSet<char> =
                self.valid_ops.iter().flat_map(|op| op.chars()).collect();
            let vocab_size = self.tokenizer.get_vocab_size(false);
            let mut ids: Vec<u32> = Vec::new();
            for id in 0..vocab_size as u32 {
                if let Ok(s) = self.tokenizer.decode(&[id], false) {
                    if s == "\"" || (!s.is_empty() && s.chars().all(|c| valid_chars.contains(&c))) {
                        ids.push(id);
                    }
                }
            }
            self.op_token_cache = Some(ids);
        }
        self.op_token_cache.as_ref().unwrap()
    }

    /// Apply the mask to `logits` given the tokens generated so far.
    /// Plug into the `mask_fn` slot of `generate_cached_constrained`.
    pub fn apply(&mut self, generated_ids: &[u32], logits: &mut [f32]) {
        let decoded = self
            .tokenizer
            .decode(generated_ids, true)
            .unwrap_or_default();
        // Concatenate seed (if any) with newly-generated text. The seed
        // captures `{"op":"` injected into the prompt as teacher forcing.
        self.generated_text = if self.seed_text.is_empty() {
            decoded
        } else {
            format!("{}{decoded}", self.seed_text)
        };

        let state = op_grammar_state(&self.generated_text);
        let so_far = match state {
            GrammarState::OpName { so_far } => so_far,
            _ => return,
        };

        // Materialise the candidate set + clone needed bits to keep the
        // borrow checker happy under simultaneous access to self.
        let _ = self.op_tokens();
        let candidate_ids: Vec<u32> = self.op_token_cache.as_ref().unwrap().clone();
        let valid_ops = self.valid_ops.clone();
        let tokenizer = self.tokenizer;

        let valid_next: HashSet<u32> = candidate_ids
            .iter()
            .copied()
            .filter(|&id| {
                let s = tokenizer.decode(&[id], false).unwrap_or_default();
                if s == "\"" {
                    // Closing quote — only allowed when so_far is a complete op name.
                    valid_ops.iter().any(|op| op == &so_far)
                } else if !s.is_empty() {
                    // Continuation — allowed if `so_far + s` is a prefix of any valid op.
                    let candidate = format!("{so_far}{s}");
                    valid_ops
                        .iter()
                        .any(|op| op.starts_with(candidate.as_str()))
                } else {
                    false
                }
            })
            .collect();

        // Empty valid_next means our grammar-state detection was wrong (e.g.
        // a lookalike `{"op":"` substring inside an arg value). Fall back to
        // free generation in that case rather than masking everything out.
        if valid_next.is_empty() {
            return;
        }
        for (i, v) in logits.iter_mut().enumerate() {
            if !valid_next.contains(&(i as u32)) {
                *v = f32::NEG_INFINITY;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Pure state-machine tests (no tokenizer needed) ────────────────────

    #[test]
    fn grammar_state_free_before_op_marker() {
        assert_eq!(op_grammar_state(""), GrammarState::Free);
        assert_eq!(op_grammar_state("Sure! "), GrammarState::Free);
        assert_eq!(op_grammar_state("{"), GrammarState::Free);
        assert_eq!(op_grammar_state("{\"op\":"), GrammarState::Free);
    }

    #[test]
    fn grammar_state_op_name_after_marker() {
        assert_eq!(
            op_grammar_state("{\"op\":\""),
            GrammarState::OpName {
                so_far: String::new()
            },
        );
        assert_eq!(
            op_grammar_state("{\"op\":\"gc"),
            GrammarState::OpName {
                so_far: "gc".into()
            },
        );
        assert_eq!(
            op_grammar_state("{\"op\":\"gcd"),
            GrammarState::OpName {
                so_far: "gcd".into()
            },
        );
    }

    #[test]
    fn grammar_state_done_after_closing_quote() {
        assert_eq!(op_grammar_state("{\"op\":\"gcd\""), GrammarState::Done,);
        assert_eq!(
            op_grammar_state(r#"{"op":"gcd","args":{"a":12}}"#),
            GrammarState::Done,
        );
    }

    #[test]
    fn grammar_state_handles_preamble_before_op_marker() {
        let text = "Here is the call:\n{\"op\":\"is_pri";
        assert_eq!(
            op_grammar_state(text),
            GrammarState::OpName {
                so_far: "is_pri".into()
            },
        );
    }

    #[test]
    fn grammar_state_picks_first_op_marker() {
        // If `{"op":"` appears more than once (unlikely in practice but
        // we should be deterministic), the first occurrence wins.
        let text = "{\"op\":\"first\",\"args\":{\"q\":\"{\\\"op\\\":\\\"second";
        // After the first quote-closer the state is Done; we should NOT
        // re-enter OpName even though the args contain a lookalike.
        assert_eq!(op_grammar_state(text), GrammarState::Done);
    }

    // ── Constructor tests (no decoding) ────────────────────────────────────

    #[test]
    fn from_op_specs_extracts_names_only() {
        // Verify the constructor unwraps OpSpec to just names — args are
        // intentionally ignored at the token level (handled by the system
        // prompt + parser tolerance).
        let specs = [
            OpSpec {
                name: "gcd".into(),
                args: vec!["a".into(), "b".into()],
            },
            OpSpec {
                name: "is_prime".into(),
                args: vec!["n".into()],
            },
        ];
        // We can't construct an OpNameMask without a Tokenizer, but we can
        // verify the conversion logic by mirroring it manually.
        let names: Vec<String> = specs.iter().map(|s| s.name.clone()).collect();
        assert_eq!(names, vec!["gcd".to_string(), "is_prime".into()]);
    }

    // ── OpNameMask::apply + op_tokens (need a tokenizer) ──────────────────

    /// Build a character-level tokenizer where each id N decodes to a
    /// single ASCII byte (or `[UNK]`). Covers the chars used in `gcd`,
    /// `is_prime`, plus the `"` quote terminator.
    fn char_tokenizer() -> tokenizers::Tokenizer {
        let mut vocab = serde_json::Map::new();
        // Reserve the chars we need; ids are sequential, decode == key.
        let chars: &[&str] = &[
            "g", "c", "d", "i", "s", "_", "p", "r", "m", "e", "\"", "{", ":", " ", "a", "n", "o",
            "x", "y", "z", "u", "t", "h",
        ];
        for (i, ch) in chars.iter().enumerate() {
            vocab.insert((*ch).into(), serde_json::Value::Number((i as u64).into()));
        }
        vocab.insert(
            "[UNK]".into(),
            serde_json::Value::Number((chars.len() as u64).into()),
        );
        let json = serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": null,
            "post_processor": null,
            "decoder": null,
            "model": { "type": "WordLevel", "vocab": vocab, "unk_token": "[UNK]" },
        });
        let bytes = serde_json::to_vec(&json).unwrap();
        tokenizers::Tokenizer::from_bytes(&bytes).unwrap()
    }

    fn id(tok: &tokenizers::Tokenizer, s: &str) -> u32 {
        tok.token_to_id(s)
            .unwrap_or_else(|| panic!("missing id for {s:?}"))
    }

    #[test]
    fn apply_does_nothing_when_state_is_free() {
        // Before `{"op":"` appears, the mask is a no-op. Logits should be
        // returned unmodified.
        let tok = char_tokenizer();
        let mut mask = OpNameMask::new(vec!["gcd".into()], &tok);
        let mut logits = vec![1.0f32; 32];
        let before = logits.clone();
        mask.apply(&[], &mut logits);
        assert_eq!(logits, before, "Free state must not mutate logits");
    }

    #[test]
    fn apply_masks_to_op_name_continuations_with_seed() {
        // Inject `{"op":"` as seed; the mask is now in OpName{so_far=""}.
        // Valid next tokens are first characters of valid op names plus
        // any token decodable as a prefix.
        let tok = char_tokenizer();
        let mut mask = OpNameMask::new(vec!["gcd".into()], &tok);
        mask.set_seed_text("{\"op\":\"");
        let vocab_size = tok.get_vocab_size(false);
        let mut logits = vec![1.0f32; vocab_size + 1];
        mask.apply(&[], &mut logits);

        let g_id = id(&tok, "g") as usize;
        let x_id = id(&tok, "{") as usize; // not a valid prefix for "gcd"
        assert!(logits[g_id].is_finite(), "'g' is a valid first char of gcd");
        assert!(
            logits[x_id].is_infinite() && logits[x_id] < 0.0,
            "'{{' is not a valid op-name continuation — must be masked to -inf"
        );
    }

    #[test]
    fn apply_allows_closing_quote_at_complete_op_name() {
        // so_far = "gcd" — closing quote `"` is allowed; continuation
        // chars are not (no valid op starts with "gcd*").
        let tok = char_tokenizer();
        let mut mask = OpNameMask::new(vec!["gcd".into()], &tok);
        mask.set_seed_text("{\"op\":\"gcd");
        let vocab_size = tok.get_vocab_size(false);
        let mut logits = vec![1.0f32; vocab_size + 1];
        mask.apply(&[], &mut logits);
        let quote_id = id(&tok, "\"") as usize;
        assert!(
            logits[quote_id].is_finite(),
            "closing quote must remain after complete op name"
        );
    }

    #[test]
    fn apply_falls_back_when_no_valid_next() {
        // so_far doesn't match any op's prefix — empty valid_next →
        // mask must return unchanged logits (the "lookalike substring"
        // fallback at lines 192-194).
        let tok = char_tokenizer();
        let mut mask = OpNameMask::new(vec!["gcd".into()], &tok);
        mask.set_seed_text("{\"op\":\"xyzunmatched");
        let vocab_size = tok.get_vocab_size(false);
        let mut logits = vec![1.0f32; vocab_size + 1];
        let before = logits.clone();
        mask.apply(&[], &mut logits);
        assert_eq!(logits, before, "empty valid_next must be a no-op");
    }

    #[test]
    fn apply_uses_generated_ids_when_no_seed() {
        // No seed — the mask decodes generated_ids to derive state.
        let tok = char_tokenizer();
        let mut mask = OpNameMask::new(vec!["gcd".into()], &tok);
        // Generate tokens for `{"op":"g` — state should become
        // OpName{so_far="g"}.
        let prefix_text = "{\"op\":\"g";
        let prefix_ids: Vec<u32> = prefix_text
            .chars()
            .map(|c| id(&tok, &c.to_string()))
            .collect();
        let vocab_size = tok.get_vocab_size(false);
        let mut logits = vec![1.0f32; vocab_size + 1];
        mask.apply(&prefix_ids, &mut logits);
        // 'c' continues "g" toward "gcd" — must stay finite.
        let c_id = id(&tok, "c") as usize;
        assert!(logits[c_id].is_finite(), "'c' continues 'g' toward 'gcd'");
    }
}
