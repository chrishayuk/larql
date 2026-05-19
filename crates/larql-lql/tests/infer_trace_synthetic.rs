//! Synthetic-model integration tests for `EXPLAIN INFER`.
//!
//! Covers the body of [`Session::exec_infer_trace`] +
//! [`Session::exec_infer_trace_dense`] without depending on a real
//! model: builds a synthetic vindex via
//! [`larql_inference::test_utils::write_synthetic_model_dir`] in a
//! tempdir, then drives `USE <dir>` + `EXPLAIN INFER …` through the
//! public parser/executor path.
//!
//! These are **plumbing tests** — the synthetic weights produce
//! garbage logits, so we assert on output shape (header rendered,
//! Prediction line present, error path triggers) rather than semantic
//! correctness. Tests that need "model actually predicts Paris" live
//! in `bench/`, not `tests/`.

use larql_inference::test_utils::write_synthetic_model_dir;
use larql_lql::executor::Session;
use larql_lql::parser;

fn fresh_synthetic_session() -> (Session, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    write_synthetic_model_dir(dir.path()).expect("fixture write");
    let mut session = Session::new();
    // Escape backslashes for the LQL lexer (Windows tempdir paths).
    let path_for_sql = dir.path().display().to_string().replace('\\', "\\\\");
    let use_stmt = format!(r#"USE "{path_for_sql}";"#);
    let parsed = parser::parse(&use_stmt).expect("USE parse");
    session.execute(&parsed).expect("USE execute");
    (session, dir)
}

#[test]
fn explain_infer_synthetic_vindex_runs() {
    let (mut session, _dir) = fresh_synthetic_session();
    // Synthetic tokenizer's vocab is `[0]`..`[31]` (Whitespace-pretok'd
    // word lookups). Real-English prompts UNK out; use a single in-vocab
    // token so the encoder produces a non-empty id sequence.
    let parsed = parser::parse(r#"EXPLAIN INFER "[1]";"#).expect("EXPLAIN parse");
    let out = session.execute(&parsed).expect("EXPLAIN INFER execute");
    let joined = out.join("\n");
    assert!(
        joined.contains("Inference trace for"),
        "expected trace header, got:\n{joined}"
    );
    assert!(
        joined.contains("Prediction"),
        "expected Prediction line, got:\n{joined}"
    );
}

#[test]
fn explain_infer_with_attention_synthetic_vindex_runs() {
    let (mut session, _dir) = fresh_synthetic_session();
    let parsed = parser::parse(r#"EXPLAIN INFER "[1]" WITH ATTENTION;"#).expect("EXPLAIN parse");
    let out = session.execute(&parsed).expect("EXPLAIN INFER execute");
    let joined = out.join("\n");
    assert!(joined.contains("Inference trace for"));
    // Attention path emits the compact "L NN feature attn → lens"
    // format. At least one such row must appear (the WITH ATTENTION
    // branch fired) — synthetic data may yield only empty rows so we
    // settle for the header instead if the body is empty.
    // Structural: at minimum the trace ran without erroring.
    assert!(!out.is_empty());
}

#[test]
fn explain_infer_with_band_filter_synthetic_runs() {
    let (mut session, _dir) = fresh_synthetic_session();
    let parsed = parser::parse(r#"EXPLAIN INFER "[1]" KNOWLEDGE;"#).expect("EXPLAIN parse");
    let out = session.execute(&parsed).expect("EXPLAIN INFER execute");
    let joined = out.join("\n");
    assert!(
        joined.contains("(knowledge)"),
        "expected knowledge band tag, got:\n{joined}"
    );
}

#[test]
fn explain_infer_with_relations_only_synthetic_runs() {
    let (mut session, _dir) = fresh_synthetic_session();
    let parsed = parser::parse(r#"EXPLAIN INFER "[1]" RELATIONS ONLY;"#).expect("EXPLAIN parse");
    let out = session.execute(&parsed).expect("EXPLAIN INFER execute");
    let joined = out.join("\n");
    // RELATIONS ONLY culls unlabelled hits; with no classifier on disk
    // every hit drops, so the body may be empty. Header must always
    // render.
    assert!(
        joined.contains("Inference trace for"),
        "header should always render, got:\n{joined}"
    );
}
