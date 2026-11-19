//! Fixtures whose tokenizer makes a KNN target one token (FACT-ID-1 D-5).
//!
//! A KNN entry recalls one token id, so INSERT refuses a target that is not
//! one token. The stub vindex shipped an empty BPE (every target encoded to
//! nothing, stored as id 0) and the full fixture a `Whitespace`
//! pre-tokenizer (`"[5]"` splits into `[`, `5`, `]`). Their KNN tests stored
//! a target text with an id that was not its token. These sessions rewrite
//! `tokenizer.json` (a VINDEX2 binding reads it from disk at each INSERT) to
//! a `WhitespaceSplit` word-level vocabulary in which every target the tests
//! use is one token.

use super::*;

/// `words` take ids from 0, `[i]` the ids after them, and `[UNK]` the last
/// id, so every id indexes a row of a `vocab_size`-row embedding table.
pub(super) fn single_token_tokenizer_json(vocab_size: usize, words: &[&str]) -> String {
    assert!(
        words.len() < vocab_size,
        "the vocabulary needs room for [UNK]"
    );
    let unk = vocab_size - 1;
    let mut vocab: Vec<String> = words
        .iter()
        .enumerate()
        .map(|(i, w)| format!(r#""{w}":{i}"#))
        .collect();
    vocab.extend((words.len()..unk).map(|i| format!(r#""[{i}]":{i}"#)));
    vocab.push(format!(r#""[UNK]":{unk}"#));
    format!(
        r#"{{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{{"type":"WhitespaceSplit"}},"post_processor":null,"decoder":null,"model":{{"type":"WordLevel","vocab":{{{}}},"unk_token":"[UNK]"}}}}"#,
        vocab.join(",")
    )
}

/// Rows of the stub vindex's embedding table (`make_test_vindex_dir`).
const STUB_VOCAB: usize = 10;
/// The targets the stub-vindex KNN tests insert.
const STUB_TARGETS: &[&str] = &["Poseidon", "Mu", "Shambhala", "Birnin"];
/// Rows of the full fixture's embedding table (`make_full_test_vindex_dir`
/// extends `make_test_weights`' 32 by one).
const FULL_VOCAB: usize = 33;

/// The stub vindex, with its targets single-token.
pub(super) fn knn_vindex_session(tag: &str) -> (Session, std::path::PathBuf) {
    let (session, dir) = vindex_session(tag);
    std::fs::write(
        dir.join("tokenizer.json"),
        single_token_tokenizer_json(STUB_VOCAB, STUB_TARGETS),
    )
    .unwrap();
    (session, dir)
}

/// The full fixture, with `[N]` one token.
pub(super) fn knn_full_vindex_session(tag: &str) -> (Session, std::path::PathBuf) {
    let (session, dir) = full_vindex_session(tag);
    std::fs::write(
        dir.join("tokenizer.json"),
        single_token_tokenizer_json(FULL_VOCAB, &[]),
    )
    .unwrap();
    (session, dir)
}

/// The tokenizer the q4k-flagged stub needs for its `Poseidon` insert.
pub(super) fn stub_knn_tokenizer_json() -> String {
    single_token_tokenizer_json(STUB_VOCAB, STUB_TARGETS)
}
