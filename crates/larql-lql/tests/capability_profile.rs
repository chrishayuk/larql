//! The capability-profile gate, end to end on a real V3 binding.
//!
//! The invariant under test: **a profile is judged after parsing and
//! before execution** — a refused statement never begins, whatever
//! transport or pipe shape delivers it, and the refusal names the
//! profile and what it serves. The unit tests in `capability.rs` pin
//! the judgement table; these tests pin that `Session::execute` is the
//! single gate.

use std::path::Path;

use larql_inference::test_utils::synthetic_tokenizer_json;
use larql_lql::{parse, CapabilityProfile, LqlError, Session};
use larql_vindex::format::vindex3::fixtures::{
    encode_fixture_container, miniature_glimmer, G_VOCAB,
};

fn lql_path(path: impl AsRef<Path>) -> String {
    path.as_ref().display().to_string().replace('\\', "\\\\")
}

fn v3_container() -> tempfile::TempDir {
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(
        miniature_glimmer,
        checkpoint.path(),
        container.path(),
        "profile-fixture",
    );
    std::fs::write(
        container.path().join("tokenizer.json"),
        synthetic_tokenizer_json(G_VOCAB),
    )
    .unwrap();
    container
}

/// Bind with the FULL profile (USE is a lifecycle statement the public
/// profile refuses), then tighten — the embedding surface's sequence:
/// the server binds its published container, then hands the session to
/// the public.
fn public_session(container: &Path) -> Session {
    let mut session = Session::new();
    let use_stmt = format!("USE \"{}\";", lql_path(container));
    session.execute(&parse(&use_stmt).unwrap()).unwrap();
    session.set_profile(CapabilityProfile::PublicExplorer);
    session
}

fn run(session: &mut Session, stmt: &str) -> Result<Vec<String>, LqlError> {
    session.execute(&parse(stmt).unwrap_or_else(|e| panic!("parse {stmt}: {e}")))
}

#[test]
fn the_default_profile_is_full() {
    assert_eq!(Session::new().profile(), CapabilityProfile::Full);
}

/// The public read surface executes — the profile constrains, it does
/// not degrade what it serves.
#[test]
fn the_public_profile_serves_the_read_surface_on_a_real_binding() {
    let container = v3_container();
    let mut session = public_session(container.path());
    for stmt in [
        "SHOW COMPONENTS;",
        "SHOW REPRESENTATIONS;",
        "SHOW PROVENANCE;",
        "SHOW AUTHORITY;",
        "SHOW LAYERS;",
        "STATS;",
        "EXPLAIN INFER \"[3]\";",
        "INFER \"[3]\" TOP 3;",
        "INFER \"[3]\" GENERATE 2;",
    ] {
        let out = run(&mut session, stmt).unwrap_or_else(|e| panic!("{stmt}: {e}"));
        assert!(!out.is_empty(), "{stmt}: empty output");
    }
}

/// A refused statement dies as `Refused` — before execution, so the
/// overlay records nothing and the error is the profile's, not a
/// backend capability apology.
#[test]
fn the_public_profile_refuses_mutation_before_it_begins() {
    let container = v3_container();
    let mut session = public_session(container.path());
    let err = run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("a", "b", "c");"#,
    )
    .expect_err("INSERT must be refused");
    assert!(matches!(err, LqlError::Refused { .. }), "{err}");
    let msg = err.to_string();
    assert!(msg.contains("INSERT"), "{msg}");
    assert!(msg.contains("PUBLIC_EXPLORER"), "{msg}");

    // Nothing began: no auto-patch session was started by the refused
    // INSERT (the recording starts inside execution, which never ran).
    let err = run(&mut session, "SAVE PATCH;").expect_err("SAVE PATCH is refused too");
    assert!(matches!(err, LqlError::Refused { .. }), "{err}");
}

/// USE is refused under the public profile even on a fresh session —
/// the capability gate outranks the backend check, so the answer is
/// "this profile does not serve USE", never "no backend loaded".
#[test]
fn use_is_refused_ahead_of_the_backend_check() {
    let mut session = Session::new();
    session.set_profile(CapabilityProfile::PublicExplorer);
    let err = run(&mut session, "USE \"anything.vindex\";").expect_err("USE must be refused");
    assert!(matches!(err, LqlError::Refused { .. }), "{err}");
    assert!(!err.to_string().contains("No backend"), "{err}");
}

/// A pipe cannot smuggle a refused leg past the gate: the statement is
/// judged whole, so the permitted leg does not run either.
#[test]
fn a_pipe_with_a_refused_leg_refuses_whole() {
    let container = v3_container();
    let mut session = public_session(container.path());
    let err = run(
        &mut session,
        "STATS |> DELETE FROM EDGES WHERE layer = 0 AND feature = 0;",
    )
    .expect_err("the pipe must refuse");
    assert!(matches!(err, LqlError::Refused { .. }), "{err}");
}

/// The generation bound refuses beyond 32 and names both numbers.
#[test]
fn generate_is_bounded_on_the_public_profile() {
    let container = v3_container();
    let mut session = public_session(container.path());
    let err = run(&mut session, "INFER \"[3]\" GENERATE 4096;").expect_err("beyond the bound");
    let msg = err.to_string();
    assert!(msg.contains("4096") && msg.contains("32"), "{msg}");
}
