//! A legacy `COMPILE INTO MODEL` either exports a checkpoint that holds the
//! session's edits, or fails by name (Gate A4).
//!
//! What used to happen instead:
//!
//! - with MEMIT off, which is the default, an `INSERT` was collected as a
//!   fact and then not used: the export was the source's weights and the
//!   statement answered `Compiled`;
//! - a `DELETE` or `UPDATE` was never read by the export at all;
//! - `FORMAT GGUF` was parsed and dropped, and safetensors were written.

use std::path::{Path, PathBuf};

use larql_inference::test_utils::write_synthetic_model_dir;
use larql_lql::{parse, Session};

const INSERT: &str = r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[1]", "capital", "[2]") AT LAYER 0 MODE COMPOSE;"#;
/// The switch that turns the MEMIT solve on.
const MEMIT_ENABLE_ENV: &str = "LARQL_MEMIT_ENABLE";
/// The slot the compose-mode INSERT above installs into.
const INSTALL_LAYER: usize = 0;
const INSTALL_FEATURE: usize = 0;

fn lql_path(path: impl AsRef<Path>) -> String {
    path.as_ref().display().to_string().replace('\\', "\\\\")
}

fn try_run(session: &mut Session, stmt: &str) -> Result<Vec<String>, String> {
    let parsed = parse(stmt).map_err(|e| format!("parse {stmt}: {e}"))?;
    session.execute(&parsed).map_err(|e| e.to_string())
}

fn run(session: &mut Session, stmt: &str) -> Vec<String> {
    try_run(session, stmt).unwrap_or_else(|e| panic!("{stmt}: {e}"))
}

struct Fixture {
    session: Session,
    out: PathBuf,
    _dir: tempfile::TempDir,
}

fn fixture() -> Fixture {
    let dir = tempfile::tempdir().expect("tempdir");
    let source = dir.path().join("source.vindex");
    std::fs::create_dir_all(&source).expect("source dir");
    write_synthetic_model_dir(&source).expect("synthetic model");
    let mut session = Session::new();
    run(&mut session, &format!("USE \"{}\";", lql_path(&source)));
    Fixture {
        session,
        out: dir.path().join("exported"),
        _dir: dir,
    }
}

impl Fixture {
    fn try_export(&mut self, clause: &str) -> Result<Vec<String>, String> {
        let stmt = format!(
            "COMPILE CURRENT INTO MODEL \"{}\"{clause};",
            lql_path(&self.out)
        );
        try_run(&mut self.session, &stmt)
    }

    /// A refused export names its reason and publishes nothing.
    fn assert_refused(&mut self, clause: &str, reason: &str) {
        let err = self
            .try_export(clause)
            .expect_err("COMPILE INTO MODEL must be refused");
        assert!(err.contains(reason), "expected {reason:?} in: {err}");
        assert!(
            !self.out.exists(),
            "a refused export must not publish a checkpoint"
        );
    }
}

#[test]
fn an_insert_the_export_would_not_contain_is_refused_by_name() {
    // MEMIT is off unless LARQL_MEMIT_ENABLE is set, and it is the only
    // mechanism that writes an INSERT into exported weights.
    let mut fx = fixture();
    run(&mut fx.session, INSERT);
    // Off for this thread whatever the process environment says.
    let _memit = larql_compute::options::ScopedEnvOverride::set(MEMIT_ENABLE_ENV, None);

    fx.assert_refused("", MEMIT_ENABLE_ENV);
}

#[test]
fn a_delete_the_export_cannot_represent_is_refused_by_name() {
    let mut fx = fixture();
    run(&mut fx.session, INSERT);
    let out = run(
        &mut fx.session,
        &format!(
            "DELETE FROM EDGES WHERE layer = {INSTALL_LAYER} AND feature = {INSTALL_FEATURE};"
        ),
    );
    assert!(out.join("\n").contains("Deleted 1 features"), "{out:?}");

    fx.assert_refused("", "DELETE");
}

#[test]
fn a_format_the_export_does_not_write_is_refused_by_name() {
    let mut fx = fixture();

    fx.assert_refused(" FORMAT GGUF", "GGUF");
}

#[test]
fn an_update_the_export_cannot_represent_is_refused_by_name() {
    let mut fx = fixture();
    run(&mut fx.session, INSERT);
    let out = run(
        &mut fx.session,
        &format!(
            "UPDATE EDGES SET confidence = 0.5 WHERE layer = {INSTALL_LAYER} AND feature = {INSTALL_FEATURE};"
        ),
    );
    assert!(out.join("\n").contains("Updated 1 features"), "{out:?}");

    fx.assert_refused("", "UPDATE");
}

#[test]
fn a_knn_mode_insert_is_refused_by_name() {
    // The default INSERT mode stores a key in the KNN store, which a
    // checkpoint does not carry.
    const KNN_INSERT: &str =
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[1]", "capital", "[2]");"#;
    let mut fx = fixture();
    run(&mut fx.session, KNN_INSERT);

    fx.assert_refused("", "KNN");
}

/// The marker every edit refusal carries.
const EDIT_REFUSAL: &str = "cannot export";

#[test]
fn an_insert_the_memit_solve_writes_is_not_refused_as_an_edit() {
    // With the solve switched on a compose-mode INSERT has a path into the
    // weights, so the edit check lets it through. (The export then fails
    // for a reason that has nothing to do with edits; see the docs.)
    let mut fx = fixture();
    run(&mut fx.session, INSERT);
    let _memit = larql_compute::options::ScopedEnvOverride::set(MEMIT_ENABLE_ENV, Some("1"));

    if let Err(err) = fx.try_export("") {
        assert!(!err.contains(EDIT_REFUSAL), "{err}");
    }
}

#[test]
fn an_unedited_session_is_not_refused_as_an_edit_or_for_safetensors() {
    let mut fx = fixture();

    if let Err(err) = fx.try_export(" FORMAT SAFETENSORS") {
        assert!(!err.contains(EDIT_REFUSAL), "{err}");
        assert!(!err.contains("FORMAT"), "{err}");
    }
}
