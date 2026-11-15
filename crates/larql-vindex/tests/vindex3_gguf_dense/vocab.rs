//! The SentencePiece vocabulary port's refusals and edge cases.

use larql_vindex::format::vindex3::gguf::dense::sentencepiece::{Piece, PieceKind};
use larql_vindex::format::vindex3::gguf::dense::spm_vocab::{
    looks_special, spm_vocab, ScoreSource,
};
use serde_json::json;

fn bpe(vocab: serde_json::Value) -> serde_json::Value {
    json!({"model": {"type": "BPE", "vocab": vocab, "byte_fallback": true, "unk_token": "<unk>"}})
}

#[test]
fn special_looking_text_is_recognised() {
    for t in [
        "<pad>",
        "<mask>",
        "<2mass>",
        "[@BOS@]",
        "<|im_start|>",
        "<\u{ff5c}x\u{ff5c}>",
        "<unused12>",
    ] {
        assert!(looks_special(t), "{t}");
    }
    assert!(!looks_special("<bos>") && !looks_special("hello"));
}

#[test]
fn the_unknown_piece_and_bytes_are_typed_from_tokenizer_json() {
    let v = spm_vocab(
        6,
        None,
        &bpe(json!({"<unk>": 0, "<0x41>": 1, "<0xZZ>": 2, "a": 3})),
        None,
    )
    .unwrap();
    assert_eq!(v.types, vec![2, 6, 1, 1, 5, 5]);
    assert_eq!(v.scores[..4], [0.0, 0.0, -0.0, -1.0]);
    assert_eq!(v.source, ScoreSource::DerivedFromVocabularyOrder);
    assert_eq!(v.padded, 2);
    let keys: Vec<String> = v.entries().into_iter().map(|(k, _)| k).collect();
    assert_eq!(keys[0], "tokenizer.ggml.model");
    assert_eq!(keys[4], "tokenizer.ggml.token_type");
}

#[test]
fn sentencepiece_pieces_keep_their_types() {
    let piece = |text: &str, kind| Piece {
        text: text.into(),
        score: -2.0,
        kind,
    };
    let pieces = [
        piece("<unk>", PieceKind::Unknown),
        piece("<s>", PieceKind::Control),
        piece("u", PieceKind::Unused),
        piece("<0x00>", PieceKind::Byte),
        piece("x", PieceKind::UserDefined),
    ];
    let v = spm_vocab(5, Some(&pieces), &json!({}), None).unwrap();
    assert_eq!(v.types, vec![2, 3, 5, 6, 1]);
    assert!(v.scores.iter().all(|s| *s == -2.0));
}

#[test]
fn malformed_vocabularies_refuse() {
    let msg = |r: Result<_, larql_vindex::VindexError>| r.unwrap_err().to_string();
    assert!(
        msg(spm_vocab(1, None, &bpe(json!({"a": 0, "b": 1})), None)).contains("different model")
    );
    assert!(msg(spm_vocab(
        4,
        None,
        &json!({"model": {"type": "Unigram"}}),
        None
    ))
    .contains("`Unigram`"));
    assert!(msg(spm_vocab(
        4,
        None,
        &json!({"model": {"type": "BPE", "vocab": []}}),
        None
    ))
    .contains("not an object"));
    assert!(msg(spm_vocab(4, None, &bpe(json!({"a": 0, "b": 5})), None)).contains("contiguous"));
    assert!(msg(spm_vocab(4, None, &bpe(json!({"a": 0, "b": 0})), None)).contains("no piece"));
    let bad_id = json!({"added_tokens_decoder": {"x": {"content": "a"}}});
    assert!(msg(spm_vocab(4, None, &bpe(json!({"a": 0})), Some(&bad_id))).contains("not a number"));
    let no_content = json!({"added_tokens_decoder": {"0": {"special": true}}});
    assert!(
        msg(spm_vocab(4, None, &bpe(json!({"a": 0})), Some(&no_content))).contains("no content")
    );
}

#[test]
fn added_tokens_beyond_the_vocabulary_are_skipped() {
    let config = json!({"added_tokens_decoder": {"9": {"content": "<x>", "special": true}}});
    let v = spm_vocab(2, None, &bpe(json!({"a": 0})), Some(&config)).unwrap();
    assert_eq!(v.tokens, vec!["a".to_string(), "[PAD1]".to_string()]);
}
