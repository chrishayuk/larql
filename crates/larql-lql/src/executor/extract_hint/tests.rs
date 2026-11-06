use super::*;
use crate::lexer::{Lexer, Token};

fn round_trip(literal_body: &str) -> String {
    let source = format!("\"{literal_body}\"");
    match Lexer::new(&source).tokenise().unwrap().first() {
        Some(Token::StringLit(s)) => s.clone(),
        other => panic!("expected a string literal, got {other:?}"),
    }
}

#[test]
fn a_windows_path_survives_the_lexer() {
    let path = r"C:\Users\RUNNER~1\AppData\Local\Temp\.tmp30E0AU";
    assert_eq!(round_trip(&lql_literal(path)), path);
}

#[test]
fn quotes_survive_the_lexer() {
    let id = r#"odd"name"#;
    assert_eq!(round_trip(&lql_literal(id)), id);
}

#[test]
fn an_hf_id_is_unchanged() {
    assert_eq!(lql_literal("google/gemma-3-4b-it"), "google/gemma-3-4b-it");
}

#[test]
fn the_leaf_is_the_last_component_on_either_separator() {
    assert_eq!(model_leaf("google/gemma-3-4b-it"), "gemma-3-4b-it");
    assert_eq!(model_leaf(r"C:\Temp\.tmp30E0AU"), ".tmp30E0AU");
    assert_eq!(model_leaf("/tmp/ckpt/"), "ckpt");
    assert_eq!(model_leaf("gemma"), "gemma");
}

#[test]
fn a_degenerate_id_names_itself() {
    assert_eq!(model_leaf("/"), "/");
    assert_eq!(model_leaf(""), "");
}

#[test]
fn the_hint_is_paste_able_lql() {
    let path = r"C:\Temp\.tmp30E0AU";
    let hint = extract_first("this operation requires a vindex", path);
    assert!(hint.starts_with("this operation requires a vindex. Extract first:\n"));
    let statement = hint.lines().nth(1).unwrap().trim();
    assert_eq!(
        statement,
        r#"EXTRACT MODEL "C:\\Temp\\.tmp30E0AU" INTO ".tmp30E0AU.vindex""#
    );
    assert!(crate::parser::parse(statement).is_ok(), "{statement}");
}
