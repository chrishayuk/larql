use std::collections::HashSet;
use std::path::Path;

use larql_vindex::format::filenames::{
    GENERATION_CONFIG_JSON, TOKENIZER_CONFIG_JSON, TOKENIZER_JSON,
};
use serde_json::json;

use super::*;
use crate::test_utils::synthetic_tokenizer_json;

/// Ids the fixture tokenizer gives the ChatML terminators and a custom EOS.
const IM_END: u32 = 7;
const ENDOFTEXT: u32 = 8;
const CUSTOM_EOS: u32 = 9;
const VOCAB: usize = 6;

/// A word-level tokenizer over `[0]..[VOCAB)` plus the given extra tokens.
fn write_tokenizer(dir: &Path, extra: &[(&str, u32)]) {
    let mut tok: serde_json::Value =
        serde_json::from_str(&synthetic_tokenizer_json(VOCAB)).expect("synthetic tokenizer");
    for (text, id) in extra {
        tok["model"]["vocab"][*text] = json!(id);
    }
    std::fs::write(dir.join(TOKENIZER_JSON), tok.to_string()).unwrap();
}

fn write(dir: &Path, name: &str, body: serde_json::Value) {
    std::fs::write(dir.join(name), body.to_string()).unwrap();
}

fn set(ids: &[u32]) -> HashSet<u32> {
    ids.iter().copied().collect()
}

#[test]
fn a_container_without_generation_config_stops_at_its_chatml_terminators() {
    // The #752 shape: a VINDEX3 import of Qwen3.5 carries tokenizer.json and
    // tokenizer_config.json but no generation_config.json.
    let dir = tempfile::tempdir().unwrap();
    write_tokenizer(
        dir.path(),
        &[("<|im_end|>", IM_END), ("<|endoftext|>", ENDOFTEXT)],
    );
    write(
        dir.path(),
        TOKENIZER_CONFIG_JSON,
        json!({"eos_token": "<|im_end|>"}),
    );
    assert_eq!(
        resolve_stop_ids(dir.path()).unwrap().ids,
        set(&[IM_END, ENDOFTEXT])
    );
}

#[test]
fn declared_eos_ids_resolve_without_a_tokenizer() {
    let dir = tempfile::tempdir().unwrap();
    write(
        dir.path(),
        GENERATION_CONFIG_JSON,
        json!({"eos_token_id": [1, 2]}),
    );
    assert_eq!(resolve_stop_ids(dir.path()).unwrap().ids, set(&[1, 2]));
}

#[test]
fn declared_ids_and_tokenizer_stops_are_unioned() {
    let dir = tempfile::tempdir().unwrap();
    write(
        dir.path(),
        GENERATION_CONFIG_JSON,
        json!({"eos_token_id": 1}),
    );
    write_tokenizer(dir.path(), &[("<|im_end|>", IM_END)]);
    assert_eq!(resolve_stop_ids(dir.path()).unwrap().ids, set(&[1, IM_END]));
}

#[test]
fn a_declared_stop_string_resolves_through_the_tokenizer() {
    let dir = tempfile::tempdir().unwrap();
    write(
        dir.path(),
        GENERATION_CONFIG_JSON,
        json!({"stop_strings": ["<custom_eos>"]}),
    );
    write_tokenizer(dir.path(), &[("<custom_eos>", CUSTOM_EOS)]);
    assert_eq!(
        resolve_stop_ids(dir.path()).unwrap().ids,
        set(&[CUSTOM_EOS])
    );
}

#[test]
fn the_tokenizer_configs_eos_token_object_form_resolves() {
    let dir = tempfile::tempdir().unwrap();
    write_tokenizer(dir.path(), &[("<custom_eos>", CUSTOM_EOS)]);
    write(
        dir.path(),
        TOKENIZER_CONFIG_JSON,
        json!({"eos_token": {"content": "<custom_eos>", "special": true}}),
    );
    assert_eq!(
        resolve_stop_ids(dir.path()).unwrap().ids,
        set(&[CUSTOM_EOS])
    );
}

#[test]
fn nothing_resolvable_is_refused_not_an_empty_set() {
    let dir = tempfile::tempdir().unwrap();
    // A tokenizer with none of the stop strings as a token.
    write_tokenizer(dir.path(), &[]);
    let err = resolve_stop_ids(dir.path()).unwrap_err();
    assert_eq!(
        err,
        StopIdsError::NoStops {
            container: dir.path().to_path_buf()
        }
    );
    let msg = err.to_string();
    assert!(msg.contains("declares no end-of-sequence token"), "{msg}");
    assert!(msg.contains(GENERATION_CONFIG_JSON), "{msg}");
}

#[test]
fn an_empty_container_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    assert!(matches!(
        resolve_stop_ids(dir.path()),
        Err(StopIdsError::NoStops { .. })
    ));
}

#[test]
fn an_unloadable_tokenizer_is_refused_with_its_path() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join(TOKENIZER_JSON), "not json").unwrap();
    let err = resolve_stop_ids(dir.path()).unwrap_err();
    assert!(matches!(err, StopIdsError::Tokenizer { .. }), "{err:?}");
    let msg = err.to_string();
    assert!(msg.contains("cannot load the tokenizer"), "{msg}");
    assert!(msg.contains(TOKENIZER_JSON), "{msg}");
}

#[test]
fn tokenizer_config_eos_token_shapes() {
    let dir = tempfile::tempdir().unwrap();
    assert_eq!(tokenizer_config_eos_token(dir.path()), None, "no file");
    for (body, want) in [
        (json!({"eos_token": "<e>"}), Some("<e>")),
        (json!({"eos_token": {"content": "<e>"}}), Some("<e>")),
        (json!({"eos_token": {"special": true}}), None),
        (json!({"eos_token": null}), None),
        (json!({}), None),
    ] {
        write(dir.path(), TOKENIZER_CONFIG_JSON, body.clone());
        assert_eq!(
            tokenizer_config_eos_token(dir.path()).as_deref(),
            want,
            "{body}"
        );
    }
    std::fs::write(dir.path().join(TOKENIZER_CONFIG_JSON), "{").unwrap();
    assert_eq!(tokenizer_config_eos_token(dir.path()), None, "unparseable");
}

#[test]
fn a_multi_token_declared_stop_is_reported_never_reinterpreted() {
    // "[1] [2]" is not one token of the word-level fixture (it spans the
    // two tokens "[1]" and "[2]"). It must not become either piece's id;
    // it is reported as not honoured.
    let dir = tempfile::tempdir().unwrap();
    write(
        dir.path(),
        GENERATION_CONFIG_JSON,
        json!({"eos_token_id": 5, "stop_strings": ["[1] [2]"]}),
    );
    write_tokenizer(dir.path(), &[]);
    let stops = resolve_stop_ids(dir.path()).unwrap();
    assert_eq!(stops.ids, set(&[5]));
    assert_eq!(stops.unresolved, vec!["[1] [2]".to_string()]);
}

#[test]
fn a_multi_token_stop_alone_is_still_refused() {
    let dir = tempfile::tempdir().unwrap();
    write(
        dir.path(),
        GENERATION_CONFIG_JSON,
        json!({"stop_strings": ["[1] [2]"]}),
    );
    write_tokenizer(dir.path(), &[]);
    assert!(matches!(
        resolve_stop_ids(dir.path()),
        Err(StopIdsError::NoStops { .. })
    ));
}

#[test]
fn declared_strings_without_a_tokenizer_are_reported() {
    let dir = tempfile::tempdir().unwrap();
    write(
        dir.path(),
        GENERATION_CONFIG_JSON,
        json!({"eos_token_id": 3, "stop_strings": ["<custom_eos>"]}),
    );
    write(
        dir.path(),
        TOKENIZER_CONFIG_JSON,
        json!({"eos_token": "<other>"}),
    );
    let stops = resolve_stop_ids(dir.path()).unwrap();
    assert_eq!(stops.ids, set(&[3]));
    assert_eq!(
        stops.unresolved,
        vec!["<custom_eos>".to_string(), "<other>".to_string()]
    );
}

#[test]
fn absent_builtin_terminators_are_not_reported() {
    // A Qwen tokenizer has no "</s>" or "<eot_id>": those belong to other
    // families and are not this container's declared stops.
    let dir = tempfile::tempdir().unwrap();
    write_tokenizer(dir.path(), &[("<|im_end|>", IM_END)]);
    let stops = resolve_stop_ids(dir.path()).unwrap();
    assert_eq!(stops.ids, set(&[IM_END]));
    assert!(stops.unresolved.is_empty(), "{:?}", stops.unresolved);
}

#[test]
fn an_eos_token_also_declared_as_a_stop_string_is_listed_once() {
    let dir = tempfile::tempdir().unwrap();
    write(
        dir.path(),
        GENERATION_CONFIG_JSON,
        json!({"eos_token_id": 3, "stop_strings": ["<x>"]}),
    );
    write(
        dir.path(),
        TOKENIZER_CONFIG_JSON,
        json!({"eos_token": "<x>"}),
    );
    assert_eq!(
        resolve_stop_ids(dir.path()).unwrap().unresolved,
        vec!["<x>".to_string()]
    );
}
