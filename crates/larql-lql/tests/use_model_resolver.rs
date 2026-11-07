//! `USE "<name>"` through an embedder's [`ModelResolver`]: an existing path
//! always wins, a missing literal is offered to the hook, the hook's refusal
//! is the error `USE` reports, and with no hook nothing changes.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use larql_inference::test_utils::synthetic_tokenizer_json;
use larql_lql::{parse, run_batch_outcome_in, ModelResolver, Session, EXIT_ERROR, EXIT_OK};
use larql_vindex::format::vindex3::fixtures::{
    encode_fixture_container, miniature_glimmer, G_VOCAB,
};

/// The name the stand-in cache knows the fixture by.
const NAME: &str = "glimmer-mini";
/// A name no resolver in these tests knows.
const UNKNOWN: &str = "no-such-model";
/// The guidance the refusing resolver gives.
const GUIDANCE: &str = "fetch it first: larql run hf://<owner>/no-such-model";

/// The miniature VINDEX3 container, with a tokenizer.
fn container() -> (tempfile::TempDir, tempfile::TempDir) {
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(
        miniature_glimmer,
        checkpoint.path(),
        container.path(),
        "lql-resolver-fixture",
    );
    std::fs::write(
        container.path().join("tokenizer.json"),
        synthetic_tokenizer_json(G_VOCAB),
    )
    .unwrap();
    (checkpoint, container)
}

/// A resolver that knows `NAME` as `dir`, refuses everything else with
/// `GUIDANCE`, and counts its calls.
fn resolver(dir: PathBuf, calls: Arc<AtomicUsize>) -> ModelResolver {
    Box::new(move |literal| {
        calls.fetch_add(1, Ordering::SeqCst);
        if literal == NAME {
            Ok(Some(dir.clone()))
        } else {
            Err(GUIDANCE.to_string())
        }
    })
}

/// Doubled backslashes survive the lexer's escape pass on Windows.
fn lql_literal(path: &Path) -> String {
    path.display().to_string().replace('\\', "\\\\")
}

fn use_stmt(literal: &str) -> String {
    format!("USE \"{literal}\";")
}

fn execute(session: &mut Session, stmt: &str) -> Result<Vec<String>, String> {
    session
        .execute(&parse(stmt).unwrap())
        .map_err(|e| e.to_string())
}

#[test]
fn a_name_binds_through_the_hook() {
    let (_ckpt, dir) = container();
    let calls = Arc::new(AtomicUsize::new(0));
    let mut session = Session::new();
    session.set_model_resolver(Some(resolver(dir.path().to_path_buf(), calls.clone())));
    let lines = execute(&mut session, &use_stmt(NAME)).unwrap();
    assert!(lines.iter().any(|l| l.contains("VINDEX3")), "{lines:?}");
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn an_existing_path_wins_and_the_hook_is_never_asked() {
    let (_ckpt, dir) = container();
    let calls = Arc::new(AtomicUsize::new(0));
    let mut session = Session::new();
    // A hook that would refuse anything but NAME: never consulted.
    session.set_model_resolver(Some(resolver(PathBuf::from("elsewhere"), calls.clone())));
    execute(&mut session, &use_stmt(&lql_literal(dir.path()))).unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[test]
fn a_refusal_is_the_error_use_reports() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut session = Session::new();
    session.set_model_resolver(Some(resolver(PathBuf::from("unused"), calls)));
    let err = execute(&mut session, &use_stmt(UNKNOWN)).unwrap_err();
    assert!(err.contains(GUIDANCE), "{err}");
}

#[test]
fn a_declined_literal_fails_as_a_missing_path() {
    let mut session = Session::new();
    session.set_model_resolver(Some(Box::new(|_| Ok(None))));
    let err = execute(&mut session, &use_stmt(UNKNOWN)).unwrap_err();
    assert!(err.contains("vindex not found"), "{err}");
}

#[test]
fn without_a_hook_a_name_is_a_missing_path_as_before() {
    let mut session = Session::new();
    let err = execute(&mut session, &use_stmt(NAME)).unwrap_err();
    assert!(err.contains(&format!("vindex not found: {NAME}")), "{err}");
}

#[test]
fn removing_the_hook_restores_the_old_behaviour() {
    let (_ckpt, dir) = container();
    let mut session = Session::new();
    session.set_model_resolver(Some(resolver(
        dir.path().to_path_buf(),
        Arc::new(AtomicUsize::new(0)),
    )));
    session.set_model_resolver(None);
    let err = execute(&mut session, &use_stmt(NAME)).unwrap_err();
    assert!(err.contains("vindex not found"), "{err}");
}

#[test]
fn a_batch_runs_in_the_configured_session() {
    let (_ckpt, dir) = container();
    let mut session = Session::new();
    session.set_model_resolver(Some(resolver(
        dir.path().to_path_buf(),
        Arc::new(AtomicUsize::new(0)),
    )));
    let ok = run_batch_outcome_in(&mut session, &format!("{} STATS;", use_stmt(NAME)));
    assert_eq!(ok.exit_code(), EXIT_OK, "{:?}", ok.lines);
    let refused = run_batch_outcome_in(&mut session, &use_stmt(UNKNOWN));
    assert_eq!(refused.exit_code(), EXIT_ERROR);
    assert!(
        refused.lines.iter().any(|l| l.contains(GUIDANCE)),
        "{:?}",
        refused.lines
    );
}

#[test]
fn a_one_shot_statement_has_no_hook_and_keeps_the_old_refusal() {
    let err = larql_lql::run_statement(&use_stmt(NAME))
        .unwrap_err()
        .to_string();
    assert!(err.contains("vindex not found"), "{err}");
}
