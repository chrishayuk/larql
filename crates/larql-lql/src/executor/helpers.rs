//! Shared helpers: formatting, token filtering.

#![allow(clippy::items_after_test_module)]

use std::path::Path;

use larql_inference::ndarray::Array1;

/// Number of leading characters of a target token used for `starts_with`
/// fuzzy matching against tokenizer outputs (e.g. "Pos" → matches " Pos",
/// "Posei", "Poseidon"). Three characters is enough discrimination for
/// the per-relation candidate sets that balance/rebalance probe.
pub(crate) const TARGET_PREFIX_CHARS: usize = 3;

/// Does one decoded vocabulary token stand for the fact's target?
///
/// The balance and rebalance probes ask this of each of the model's top
/// predictions: the token matches when it contains the whole target, or
/// starts with its first [`TARGET_PREFIX_CHARS`] characters.
///
/// Leading whitespace is ignored. A vocabulary that spells a word following a
/// space as its own token keeps that space when the token is decoded on its own:
/// the continuation of "The capital of Atlantis is" decodes as `" Pose"`, not
/// `"Pose"`. Qwen's byte-level BPE does this, and so does Gemma 3's SentencePiece
/// vocabulary, whose decoder replaces `▁` with a space. Without the trim, a
/// prediction of 99.99% for the target read as 0.01%, so a probe that was meant
/// to shrink an over-strong install saw a weak one and amplified it.
///
/// Only leading whitespace is dropped; no other token is newly accepted.
pub(crate) fn target_token_matches(decoded: &str, target: &str) -> bool {
    // ARM A0: the matcher as it was before #701 (decoded token not trimmed).
    let token = decoded;
    token.contains(target) || token.starts_with(target_prefix(target, TARGET_PREFIX_CHARS))
}

/// Take the first `n` characters of `s`, slicing on a UTF-8 character
/// boundary. The previous `&s[..s.len().min(3)]` shape panicked on any
/// multi-byte target ("Köln", emoji, CJK).
pub(crate) fn target_prefix(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((idx, _)) => &s[..idx],
        None => s,
    }
}

/// Average a sequence of embedding rows, applying `embed_scale`. The
/// pure half of [`entity_query_vec`]: extracted so unit tests can
/// exercise the math without a tokenizer fixture.
///
/// Bounds-checks `embed.row(idx)` — out-of-vocab token ids are skipped
/// rather than panicking. Returns `None` when no token ids land in
/// range.
pub(crate) fn average_embed_rows(
    embed: &larql_inference::ndarray::Array2<f32>,
    embed_scale: f32,
    token_ids: &[u32],
) -> Option<Array1<f32>> {
    let hidden = embed.shape()[1];
    let vocab = embed.shape()[0];
    let mut avg = Array1::<f32>::zeros(hidden);
    let mut counted = 0usize;
    for &tok in token_ids {
        let idx = tok as usize;
        if idx >= vocab {
            continue;
        }
        let row = embed.row(idx);
        for (j, v) in row.iter().enumerate() {
            avg[j] += v * embed_scale;
        }
        counted += 1;
    }
    if counted == 0 {
        return None;
    }
    avg.mapv_inplace(|v| v / counted as f32);
    Some(avg)
}

/// Build the averaged-embedding query vector that SELECT / DESCRIBE /
/// INSERT (compose + knn) walks the FFN against. Six call sites used
/// to inline this loop; the helper fixes two issues at once:
///
///   1. Bounds-checks `embed.row(tok)` (via `average_embed_rows`).
///   2. Single canonical "averaging" definition.
pub(crate) fn entity_query_vec(
    tokenizer: &larql_vindex::tokenizers::Tokenizer,
    embed: &larql_inference::ndarray::Array2<f32>,
    embed_scale: f32,
    entity: &str,
) -> Result<Option<Array1<f32>>, crate::error::LqlError> {
    let encoding = tokenizer
        .encode(entity, false)
        .map_err(|e| crate::error::LqlError::exec("tokenize entity", e))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();
    if token_ids.is_empty() {
        return Ok(None);
    }
    Ok(average_embed_rows(embed, embed_scale, &token_ids))
}

/// Get total size of a directory in bytes.
pub(crate) fn dir_size(path: &Path) -> u64 {
    let mut total = 0u64;
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            if let Ok(meta) = entry.metadata() {
                total += meta.len();
            }
        }
    }
    total
}

pub(crate) fn format_number(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.2}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        format!("{n}")
    }
}

pub(crate) fn format_bytes(b: u64) -> String {
    if b >= 1_073_741_824 {
        format!("{:.2} GB", b as f64 / 1_073_741_824.0)
    } else if b >= 1_048_576 {
        format!("{:.1} MB", b as f64 / 1_048_576.0)
    } else if b >= 1024 {
        format!("{:.1} KB", b as f64 / 1024.0)
    } else {
        format!("{b} B")
    }
}

pub(crate) fn format_knn_override_summary(
    ovr: &larql_inference::KnnOverride,
    model_top1: Option<&(String, f64)>,
) -> String {
    let base = format!(
        "source=knn_override/post_logits, cos={:.2}, L{}",
        ovr.cosine, ovr.layer
    );
    match model_top1 {
        Some((tok, prob)) => format!("{base}, model_top1={} ({:.2}%)", tok, prob * 100.0),
        None => base,
    }
}

/// Heuristic: is a token readable enough to show to the user?
/// Filters out encoding garbage, isolated combining marks, etc.
pub(crate) fn is_readable_token(tok: &str) -> bool {
    let tok = tok.trim();
    if tok.is_empty() || tok.len() > 30 {
        return false;
    }
    let readable = tok
        .chars()
        .filter(|c| {
            c.is_ascii_alphanumeric()
                || *c == ' '
                || *c == '-'
                || *c == '\''
                || *c == '.'
                || *c == ','
        })
        .count();
    let total = tok.chars().count();
    readable * 2 >= total && total > 0
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── target_token_matches ─────────────────────────────────────────

    /// The rule before the leading-space fix, kept here as the yardstick the
    /// new rule must not exceed.
    fn matches_before_the_fix(token: &str, target: &str) -> bool {
        token.contains(target) || token.starts_with(target_prefix(target, TARGET_PREFIX_CHARS))
    }

    #[test]
    fn a_leading_space_token_is_recognised_as_the_target() {
        // " Pose" is how a byte-level BPE vocabulary spells the continuation.
        assert!(target_token_matches(" Pose", "Poseidon"));
        assert!(target_token_matches("  Pose", "Poseidon"));
        assert!(target_token_matches("\tPose", "Poseidon"));
        assert!(target_token_matches("\n Poseidon", "Poseidon"));
        assert!(!matches_before_the_fix(" Pose", "Poseidon"));
    }

    #[test]
    fn tokens_without_leading_space_match_as_before() {
        for token in ["Pose", "Pos", "Poseidon", "Poseidon's", "the Poseidon"] {
            assert!(target_token_matches(token, "Poseidon"), "{token:?}");
        }
    }

    #[test]
    fn a_similar_but_wrong_token_is_still_refused() {
        for token in [
            " Zeus", " Paris", " Po", " Pe", " Pa", " pose", " POS", "Zeus", "", " ", "  \n",
        ] {
            assert!(!target_token_matches(token, "Poseidon"), "{token:?}");
        }
    }

    #[test]
    fn leading_space_is_the_only_new_acceptance() {
        let targets = ["Poseidon", "Paris", "Köln", "東京タワー", "Hi", "x"];
        let tokens = [
            "Pose",
            " Pose",
            "Pos",
            " Pos",
            "Po",
            " Po",
            "Paris",
            " Paris",
            "Par",
            " Par",
            "Köln",
            " Köln",
            "Köl",
            " Köl",
            "東京タ",
            " 東京タ",
            "Hi",
            " Hi",
            "H",
            " H",
            "x",
            " x",
            "",
            " ",
            "Zeus",
            " Zeus",
            "the",
            " the",
            " Pose idon",
            "\u{a0}Pose",
        ];
        for target in targets {
            for token in tokens {
                let now = target_token_matches(token, target);
                // Nothing is accepted that the old rule refused on the same
                // token once its leading whitespace is removed.
                assert_eq!(
                    now,
                    matches_before_the_fix(token.trim_start(), target),
                    "{token:?} vs {target:?}"
                );
                // A token with no leading whitespace is judged exactly as before.
                if token.trim_start() == token {
                    assert_eq!(
                        now,
                        matches_before_the_fix(token, target),
                        "{token:?} vs {target:?}"
                    );
                }
            }
        }
    }

    /// A byte-level BPE tokenizer, as Qwen ships: `Ġ` is the byte-level
    /// spelling of a space, and the decoder maps it back.
    fn byte_level_tokenizer() -> larql_vindex::tokenizers::Tokenizer {
        let json = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],
            "normalizer":null,
            "pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true},
            "post_processor":null,
            "decoder":{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true},
            "model":{"type":"WordLevel","vocab":{"ĠPose":0,"Pose":1,"ĠZeus":2,"[UNK]":3},"unk_token":"[UNK]"}}"#;
        larql_vindex::tokenizers::Tokenizer::from_bytes(json.as_bytes()).unwrap()
    }

    /// A SentencePiece tokenizer as Gemma 3 ships it: `▁` marks a leading space and the
    /// decoder replaces it with one, so decoding a continuation token alone keeps the space.
    fn replace_decoder_tokenizer() -> larql_vindex::tokenizers::Tokenizer {
        let json = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],
            "normalizer":null,"pre_tokenizer":null,"post_processor":null,
            "decoder":{"type":"Sequence","decoders":[
                {"type":"Replace","pattern":{"String":"▁"},"content":" "},
                {"type":"ByteFallback"},{"type":"Fuse"}]},
            "model":{"type":"WordLevel","vocab":{"▁Pose":0,"Pose":1,"▁Zeus":2,"[UNK]":3},"unk_token":"[UNK]"}}"#;
        larql_vindex::tokenizers::Tokenizer::from_bytes(json.as_bytes()).unwrap()
    }

    #[test]
    fn a_sentencepiece_replace_decoder_also_keeps_the_space() {
        let tokenizer = replace_decoder_tokenizer();
        let spaced = tokenizer.decode(&[0], false).unwrap();
        assert_eq!(spaced, " Pose", "the premise: this decoder keeps the space");
        assert!(target_token_matches(&spaced, "Poseidon"));
        assert!(target_token_matches(
            &tokenizer.decode(&[1], false).unwrap(),
            "Poseidon"
        ));
        assert!(!target_token_matches(
            &tokenizer.decode(&[2], false).unwrap(),
            "Poseidon"
        ));
    }

    #[test]
    fn a_byte_level_tokenizer_decodes_a_continuation_token_with_its_space() {
        let tokenizer = byte_level_tokenizer();
        let spaced = tokenizer.decode(&[0], false).unwrap();
        let bare = tokenizer.decode(&[1], false).unwrap();
        // The premise of the fix: decoding the token alone keeps the space.
        assert_eq!(spaced, " Pose");
        assert_eq!(bare, "Pose");
        assert!(target_token_matches(&spaced, "Poseidon"));
        assert!(target_token_matches(&bare, "Poseidon"));
        let other = tokenizer.decode(&[2], false).unwrap();
        assert_eq!(other, " Zeus");
        assert!(!target_token_matches(&other, "Poseidon"));
    }

    #[test]
    fn knn_override_summary_names_post_logits_source_and_model_top1() {
        let ovr = larql_inference::KnnOverride {
            token: "Colchester".into(),
            cosine: 0.987,
            layer: 26,
        };

        let summary = format_knn_override_summary(&ovr, Some(&("London".into(), 0.42)));

        assert!(summary.contains("source=knn_override/post_logits"));
        assert!(summary.contains("cos=0.99"));
        assert!(summary.contains("L26"));
        assert!(summary.contains("model_top1=London (42.00%)"));
    }

    #[test]
    fn target_prefix_ascii_three_chars() {
        assert_eq!(target_prefix("Poseidon", 3), "Pos");
    }

    #[test]
    fn target_prefix_handles_multibyte() {
        // "Köln" — `ö` is two UTF-8 bytes. The previous byte-slicing
        // version panicked here.
        assert_eq!(target_prefix("Köln", 3), "Köl");
        // CJK and emoji — each codepoint is multi-byte; slicing must
        // land on a char boundary.
        assert_eq!(target_prefix("東京タワー", 3), "東京タ");
        assert_eq!(target_prefix("🦀🦀🦀🦀", 3), "🦀🦀🦀");
    }

    #[test]
    fn target_prefix_returns_full_string_when_shorter_than_n() {
        assert_eq!(target_prefix("Hi", 5), "Hi");
        assert_eq!(target_prefix("", 3), "");
    }

    #[test]
    fn target_prefix_chars_constant_is_three() {
        // Pinned: balance/rebalance assume 3-char discrimination.
        assert_eq!(TARGET_PREFIX_CHARS, 3);
    }

    #[test]
    fn format_number_buckets() {
        assert_eq!(format_number(5), "5");
        assert_eq!(format_number(1_500), "1.5K");
        assert_eq!(format_number(2_500_000), "2.50M");
    }

    #[test]
    fn format_bytes_buckets() {
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(2 * 1024), "2.0 KB");
        assert_eq!(format_bytes(3 * 1_048_576), "3.0 MB");
        assert_eq!(format_bytes(2 * 1_073_741_824), "2.00 GB");
    }

    // The pure averaging math is the interesting part of
    // `entity_query_vec`; the tokenizer wrapper is exercised by the
    // executor integration tests. We unit-test `average_embed_rows`
    // directly so we don't have to bake a tokenizer fixture.

    #[test]
    fn average_embed_rows_averages_token_embeddings() {
        // Two-token average.
        let mut embed = larql_inference::ndarray::Array2::<f32>::zeros((30, 3));
        embed[[0, 0]] = 1.0;
        embed[[1, 1]] = 1.0;

        let q = average_embed_rows(&embed, 1.0, &[0u32, 1u32]).expect("two tokens");
        assert!((q[0] - 0.5).abs() < 1e-6);
        assert!((q[1] - 0.5).abs() < 1e-6);
        assert!(q[2].abs() < 1e-6);
    }

    #[test]
    fn average_embed_rows_applies_embed_scale() {
        let mut embed = larql_inference::ndarray::Array2::<f32>::zeros((30, 2));
        embed[[0, 0]] = 1.0;
        let q = average_embed_rows(&embed, 4.0, &[0u32]).unwrap();
        // Single-token, scale 4 → row [4, 0].
        assert!((q[0] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn average_embed_rows_skips_out_of_vocab_tokens() {
        // Vocab has 5 rows; token id 25 is past the embed's vocab.
        // Helper must skip rather than panic.
        let mut embed = larql_inference::ndarray::Array2::<f32>::zeros((5, 2));
        embed[[0, 0]] = 2.0;
        let q = average_embed_rows(&embed, 1.0, &[0u32, 25u32]).expect("at least one in-range");
        // Only id 0 contributes.
        assert!((q[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn average_embed_rows_returns_none_when_all_out_of_vocab() {
        let embed = larql_inference::ndarray::Array2::<f32>::zeros((1, 2));
        assert!(average_embed_rows(&embed, 1.0, &[5u32, 10u32]).is_none());
    }

    #[test]
    fn average_embed_rows_returns_none_for_empty_input() {
        let embed = larql_inference::ndarray::Array2::<f32>::zeros((1, 2));
        assert!(average_embed_rows(&embed, 1.0, &[]).is_none());
    }
}

/// Stricter filter for SHOW RELATIONS and DESCRIBE: content words only.
/// Must look like a real word — no code tokens, no encoding fragments.
pub(crate) fn is_content_token(tok: &str) -> bool {
    let tok = tok.trim();
    if !is_readable_token(tok) {
        return false;
    }
    let chars: Vec<char> = tok.chars().collect();
    if chars.len() < 3 || chars.len() > 25 {
        return false;
    }
    // Must be mostly alphabetic
    let alpha = chars.iter().filter(|c| c.is_ascii_alphabetic()).count();
    if alpha < chars.len() * 2 / 3 {
        return false;
    }
    // Reject camelCase code tokens
    for w in chars.windows(2) {
        if w[0].is_ascii_lowercase() && w[1].is_ascii_uppercase() {
            return false;
        }
    }
    // Reject if all non-ASCII (encoding fragment)
    if !chars.iter().any(|c| c.is_ascii_alphabetic()) {
        return false;
    }
    // Filter English stop words and common function words
    let lower = tok.to_lowercase();
    !matches!(
        lower.as_str(),
        "the"
            | "and"
            | "for"
            | "but"
            | "not"
            | "you"
            | "all"
            | "can"
            | "her"
            | "was"
            | "one"
            | "our"
            | "out"
            | "are"
            | "has"
            | "his"
            | "how"
            | "its"
            | "may"
            | "new"
            | "now"
            | "old"
            | "see"
            | "way"
            | "who"
            | "did"
            | "get"
            | "let"
            | "say"
            | "she"
            | "too"
            | "use"
            | "from"
            | "have"
            | "been"
            | "will"
            | "with"
            | "this"
            | "that"
            | "they"
            | "were"
            | "some"
            | "them"
            | "than"
            | "when"
            | "what"
            | "your"
            | "each"
            | "make"
            | "like"
            | "just"
            | "over"
            | "such"
            | "take"
            | "also"
            | "into"
            | "only"
            | "very"
            | "more"
            | "does"
            | "most"
            | "about"
            | "which"
            | "their"
            | "would"
            | "there"
            | "could"
            | "other"
            | "after"
            | "being"
            | "where"
            | "these"
            | "those"
            | "first"
            | "should"
            | "because"
            | "through"
            | "before"
            | "par"
            | "aux"
            | "che"
            | "del"
    )
}
