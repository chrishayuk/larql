//! [`Conversation`]: which text is encoded for each turn, how, and what
//! joins the history — against a word-level tokenizer, no model.

use super::super::conversation::Conversation;
use super::*;
use larql_vindex::tokenizers::Tokenizer;

/// A template whose rendering is a run of fixture tokens: a `[9]` marker,
/// then every turn's content.
const MARKED: &str = "[9] {% for m in messages %}{{ m.content }} {% endfor %}";

fn tokenizer() -> Tokenizer {
    Tokenizer::from_bytes(word_level_tokenizer(DENSE_VOCAB).as_bytes()).unwrap()
}

fn container_with(template: Option<&str>) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    if let Some(t) = template {
        std::fs::write(dir.path().join("chat_template.jinja"), t).unwrap();
    }
    dir
}

#[test]
fn a_templated_turn_is_rendered_and_joins_the_history() {
    let dir = container_with(Some(MARKED));
    let tok = tokenizer();
    let mut status = Vec::new();
    let mut c = Conversation::new(dir.path(), "dense", false);
    let first = c.prompt(&tok, "[1] [2]", &mut status).unwrap();
    assert!(first.templated);
    assert_eq!(first.ids, vec![9, 1, 2]);
    c.record_reply(" [4]");
    assert_eq!(c.history().len(), 2);
    let second = c.prompt(&tok, "[3]", &mut status).unwrap();
    assert_eq!(second.ids, vec![9, 1, 2, 4, 3]);
    assert!(status.is_empty(), "a templated run has nothing to note");
}

#[test]
fn raw_sends_the_text_verbatim_and_keeps_no_history() {
    let dir = container_with(Some(MARKED));
    let tok = tokenizer();
    let mut status = Vec::new();
    let mut c = Conversation::new(dir.path(), "dense", true);
    let p = c.prompt(&tok, "[1] [2]", &mut status).unwrap();
    assert!(!p.templated);
    assert_eq!(p.ids, vec![1, 2]);
    c.record_reply("[4]");
    assert!(c.history().is_empty());
}

#[test]
fn no_template_falls_back_to_raw_and_says_so_once() {
    let dir = container_with(None);
    let tok = tokenizer();
    let mut status = Vec::new();
    let mut c = Conversation::new(dir.path(), "dense", false);
    for _ in 0..2 {
        let p = c.prompt(&tok, "[1]", &mut status).unwrap();
        assert!(!p.templated);
        assert_eq!(p.ids, vec![1]);
        c.record_reply("[2]");
    }
    let status = String::from_utf8(status).unwrap();
    assert_eq!(
        status.matches("carries no chat template").count(),
        1,
        "{status}"
    );
    assert!(c.history().is_empty());
}

#[test]
fn a_broken_template_refuses_and_names_the_escape_hatch() {
    let dir = container_with(Some("{% for"));
    let mut c = Conversation::new(dir.path(), "dense", false);
    let err = c
        .prompt(&tokenizer(), "[1]", &mut Vec::new())
        .unwrap_err()
        .to_string();
    assert!(err.contains("--raw"), "{err}");
}

#[test]
fn an_empty_encoding_is_refused() {
    let dir = container_with(None);
    let mut c = Conversation::new(dir.path(), "dense", true);
    let err = c
        .prompt(&tokenizer(), "", &mut Vec::new())
        .unwrap_err()
        .to_string();
    assert!(err.contains("no tokens"), "{err}");
}

/// The raw path feeds the shared table's ids for every BOS shape —
/// Granite/Qwen3 (no BOS), Gemma 4 (BOS once), a tokenizer that adds it
/// (not doubled) — the table `vindex3 observe`, the server's plain
/// completions and LQL are held to. Before the shared encoder the Gemma
/// 4 shape fed no BOS and its 12B answered "1111…".
#[test]
fn a_raw_prompt_encodes_every_bos_shape_like_every_surface() {
    use larql_inference::test_utils::{BosShape, RAW_PROMPT};
    for shape in BosShape::ALL {
        let dir = container_with(None);
        shape.write_into(dir.path(), DENSE_VOCAB);
        let tok = larql_vindex::load_vindex_tokenizer(dir.path()).unwrap();
        let mut c = Conversation::new(dir.path(), "dense", true);
        let p = c.prompt(&tok, RAW_PROMPT, &mut Vec::new()).unwrap();
        assert_eq!(p.ids, shape.expected_ids(), "{shape:?}");
    }
}

/// A templated prompt carries its own special tokens: no BOS is added
/// on top of what the template rendered, whatever the container declares.
#[test]
fn a_templated_prompt_gets_no_extra_bos() {
    use larql_inference::test_utils::BosShape;
    let dir = container_with(Some(MARKED));
    BosShape::DeclaredRequired.write_into(dir.path(), DENSE_VOCAB);
    let tok = larql_vindex::load_vindex_tokenizer(dir.path()).unwrap();
    let mut c = Conversation::new(dir.path(), "dense", false);
    let p = c.prompt(&tok, "[1]", &mut Vec::new()).unwrap();
    assert!(p.templated);
    assert_eq!(p.ids, vec![9, 1]);
}
