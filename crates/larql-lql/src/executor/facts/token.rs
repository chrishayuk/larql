//! The single-token rule for KNN facts (`docs/fact-id-1.md` D-5).
//!
//! A KNN entry recalls one token id. Its target is admitted only when
//! `" {target}"` (the spaced form every KNN insert encodes) is exactly one
//! token and that token decodes back to the target, so the stored text and
//! the stored id say the same thing.

use larql_inference::tokenizers::Tokenizer;

/// The one token `target` is, or `None` when it is not exactly one token.
pub(crate) fn single_token_id(tokenizer: &Tokenizer, target: &str) -> Option<u32> {
    let encoding = tokenizer.encode(format!(" {target}"), false).ok()?;
    let [id] = encoding.get_ids() else {
        return None;
    };
    let decoded = tokenizer.decode(&[*id], true).ok()?;
    (decoded.trim() == target.trim()).then_some(*id)
}
