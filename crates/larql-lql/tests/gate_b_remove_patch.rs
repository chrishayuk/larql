//! Gate B, B2 (`docs/gate-b.md`), through the statement surface on VINDEX2:
//! after `APPLY PATCH` then `REMOVE PATCH` of a compose install, INFER and
//! `COMPILE CURRENT INTO VINDEX` give exactly what a session that never
//! applied the patch gives.
//!
//! Before the fix, the removed install's up and down vectors stayed on the
//! base (EDIT-9), so the compiled weights carried them.

use std::path::{Path, PathBuf};

use larql_inference::test_utils::write_synthetic_model_dir;
use larql_lql::{parse, Session};

const INSERT: &str = r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[1]", "capital", "[2]") AT LAYER 0 MODE COMPOSE;"#;
const PROMPT: &str = "[1] capital";

fn lql_path(path: impl AsRef<Path>) -> String {
    path.as_ref().display().to_string().replace('\\', "\\\\")
}

fn run(session: &mut Session, stmt: &str) -> Vec<String> {
    let parsed = parse(stmt).unwrap_or_else(|e| panic!("parse {stmt}: {e}"));
    session
        .execute(&parsed)
        .unwrap_or_else(|e| panic!("{stmt}: {e}"))
}

fn bound(source: &Path) -> Session {
    let mut session = Session::new();
    run(&mut session, &format!("USE \"{}\";", lql_path(source)));
    session
}

/// INFER without its timing lines.
fn infer(session: &mut Session) -> Vec<String> {
    run(session, &format!("INFER \"{PROMPT}\" TOP 5;"))
        .into_iter()
        .filter(|l| !l.trim_end().ends_with("ms"))
        .collect()
}

fn compile(session: &mut Session, out: &Path) {
    run(
        session,
        &format!("COMPILE CURRENT INTO VINDEX \"{}\";", lql_path(out)),
    );
}

/// Every file of a compiled vindex, by name, with its bytes.
fn files(dir: &Path) -> Vec<(String, Vec<u8>)> {
    let mut out: Vec<(String, Vec<u8>)> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.is_file())
        .map(|p| {
            (
                p.file_name().unwrap().to_string_lossy().into_owned(),
                std::fs::read(&p).unwrap(),
            )
        })
        .collect();
    out.sort();
    out
}

struct Fixture {
    source: PathBuf,
    patch: PathBuf,
    dir: tempfile::TempDir,
}

/// A synthetic VINDEX2 model and a saved patch holding one compose install.
fn fixture() -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("source.vindex");
    std::fs::create_dir_all(&source).unwrap();
    write_synthetic_model_dir(&source).unwrap();
    let patch = dir.path().join("install.vlp");
    let mut author = bound(&source);
    run(
        &mut author,
        &format!("BEGIN PATCH \"{}\";", lql_path(&patch)),
    );
    run(&mut author, INSERT);
    run(&mut author, "SAVE PATCH;");
    Fixture { source, patch, dir }
}

#[test]
fn removing_a_compose_patch_gives_back_the_unpatched_infer_and_compile() {
    let fx = fixture();

    let mut pristine = bound(&fx.source);
    let want_infer = infer(&mut pristine);
    let want_out = fx.dir.path().join("pristine.vindex");
    compile(&mut pristine, &want_out);

    let mut session = bound(&fx.source);
    run(
        &mut session,
        &format!("APPLY PATCH \"{}\";", lql_path(&fx.patch)),
    );
    let patched_out = fx.dir.path().join("patched.vindex");
    compile(&mut session, &patched_out);
    assert_ne!(
        files(&patched_out),
        files(&want_out),
        "the install must change the compiled vindex"
    );
    run(
        &mut session,
        &format!("REMOVE PATCH \"{}\";", lql_path(&fx.patch)),
    );

    assert_eq!(infer(&mut session), want_infer);
    let got_out = fx.dir.path().join("removed.vindex");
    compile(&mut session, &got_out);
    let (got, want) = (files(&got_out), files(&want_out));
    assert_eq!(
        got.iter().map(|(n, _)| n).collect::<Vec<_>>(),
        want.iter().map(|(n, _)| n).collect::<Vec<_>>()
    );
    // Every file byte for byte, `index.json` included: its checksum map is
    // written in key order (Gate B, B6, D4).
    for ((name, got), (_, want)) in got.iter().zip(&want) {
        assert!(got == want, "{name} differs after REMOVE PATCH");
    }
}
