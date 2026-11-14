//! The shared raw-prompt encoder: which BOS a container declares, and
//! how a raw prompt is encoded with it.

use serde_json::json;

use crate::test_utils::synthetic_tokenizer_json;
use crate::vindex3::{declared_prompt_bos, encode_raw_prompt, PromptEncodeError};

/// Synthetic vocabulary: ids `[0]`…`[VOCAB-1]`.
const VOCAB: usize = 8;
/// The id the containers below declare as BOS.
const BOS: u32 = 2;

fn container_with(files: &[(&str, serde_json::Value)]) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    for (name, body) in files {
        std::fs::write(dir.path().join(name), body.to_string()).unwrap();
    }
    dir
}

fn tokenizer() -> tokenizers::Tokenizer {
    let mut json: serde_json::Value =
        serde_json::from_str(&synthetic_tokenizer_json(VOCAB)).unwrap();
    json["pre_tokenizer"] = json!({"type": "WhitespaceSplit"});
    tokenizers::Tokenizer::from_bytes(json.to_string().as_bytes()).unwrap()
}

#[test]
fn generation_config_declares_the_bos() {
    let dir = container_with(&[("generation_config.json", json!({"bos_token_id": BOS}))]);
    assert_eq!(declared_prompt_bos(dir.path()), Some(BOS));
}

#[test]
fn tokenizer_config_answers_when_generation_config_is_silent() {
    let dir = container_with(&[
        ("generation_config.json", json!({"top_k": 64})),
        ("tokenizer_config.json", json!({"bos_token_id": 7})),
    ]);
    assert_eq!(declared_prompt_bos(dir.path()), Some(7));
}

/// Granite / Qwen3: an id AND `add_bos_token: false` — no BOS.
#[test]
fn add_bos_token_false_outranks_a_declared_id() {
    let dir = container_with(&[
        ("generation_config.json", json!({"bos_token_id": BOS})),
        ("tokenizer_config.json", json!({"add_bos_token": false})),
    ]);
    assert_eq!(declared_prompt_bos(dir.path()), None);
}

#[test]
fn add_bos_token_true_or_non_boolean_keeps_the_id() {
    for flag in [json!(true), json!("false")] {
        let dir = container_with(&[
            ("generation_config.json", json!({"bos_token_id": BOS})),
            ("tokenizer_config.json", json!({"add_bos_token": flag})),
        ]);
        assert_eq!(declared_prompt_bos(dir.path()), Some(BOS), "{flag}");
    }
}

#[test]
fn nothing_declared_or_malformed_is_no_bos() {
    let empty = tempfile::tempdir().unwrap();
    assert_eq!(declared_prompt_bos(empty.path()), None);
    std::fs::write(empty.path().join("generation_config.json"), "{not json").unwrap();
    assert_eq!(declared_prompt_bos(empty.path()), None);
    let out_of_range = container_with(&[(
        "generation_config.json",
        json!({"bos_token_id": u64::from(u32::MAX) + 1}),
    )]);
    assert_eq!(declared_prompt_bos(out_of_range.path()), None);
}

#[test]
fn a_raw_prompt_gets_the_bos_once() {
    let tok = tokenizer();
    assert_eq!(
        encode_raw_prompt(&tok, "[3] [4]", None).unwrap(),
        vec![3, 4]
    );
    assert_eq!(
        encode_raw_prompt(&tok, "[3] [4]", Some(BOS)).unwrap(),
        vec![BOS, 3, 4]
    );
    // Already first: not doubled.
    assert_eq!(
        encode_raw_prompt(&tok, "[2] [4]", Some(BOS)).unwrap(),
        vec![BOS, 4]
    );
}

#[test]
fn an_empty_prompt_is_refused_before_any_bos() {
    let tok = tokenizer();
    let err = encode_raw_prompt(&tok, "", Some(BOS)).unwrap_err();
    assert_eq!(err, PromptEncodeError::Empty);
    assert_eq!(err.to_string(), "prompt tokenises to empty");
    assert!(PromptEncodeError::Tokenize("x".into())
        .to_string()
        .contains("tokenize: x"));
}

/// The shared shape table: the encoder produces each shape's expected
/// ids from the declarations the shape writes — the reference every
/// surface's parity test compares against.
#[test]
fn every_bos_shape_encodes_to_its_expected_ids() {
    use crate::test_utils::{BosShape, RAW_PROMPT};
    for shape in BosShape::ALL {
        let dir = tempfile::tempdir().unwrap();
        shape.write_into(dir.path(), VOCAB);
        let tok = crate::load_tokenizer(dir.path()).unwrap();
        let ids = encode_raw_prompt(&tok, RAW_PROMPT, declared_prompt_bos(dir.path())).unwrap();
        assert_eq!(ids, shape.expected_ids(), "{shape:?}");
    }
}
