//! FACT-ID-1 acceptance gates (`docs/fact-id-1.md` §9, frozen at r2).
//!
//! Every gate runs through the statement surface on the miniature VINDEX3
//! fixture, whose `[N]` word-level tokenizer makes `" [N]"` one token (id N),
//! so every KNN target here is single-token by construction and INFER runs
//! the real KNN override. The VINDEX2 parity gate (ID-8) builds a VINDEX2
//! vindex with the real extractor.

mod compose_guard;
mod identity;
mod lifecycle;
mod mutation;
mod parity;

use std::path::Path;

use larql_lql::{parse, AssertedFact, LqlError, Session};
use larql_vindex::format::vindex3::fixtures::{
    encode_fixture_container, miniature_glimmer, G_VOCAB,
};

/// The canonical prompt of the fact `("a", "b")`.
pub const CANONICAL_PROMPT: &str = "The b of a is";
/// The fact every gate starts from.
pub const INSERT_AB: &str =
    r#"INSERT INTO EDGES (entity, relation, target) VALUES ("a", "b", "[7]");"#;

/// The `[N]` ↔ id N tokenizer. A KNN target encodes `" {target}"`; a
/// `Strip` normalizer makes that the single token N without giving any id a
/// second spelling (a spaced alias decodes either way from run to run).
pub fn knn_tokenizer_json(vocab: usize) -> String {
    let entries: Vec<String> = (0..vocab).map(|i| format!("\"[{i}]\":{i}")).collect();
    format!(
        "{{\"version\":\"1.0\",\"truncation\":null,\"padding\":null,\"added_tokens\":[],\
         \"normalizer\":{{\"type\":\"Strip\",\"strip_left\":true,\"strip_right\":true}},\
         \"pre_tokenizer\":null,\"post_processor\":null,\"decoder\":null,\
         \"model\":{{\"type\":\"WordLevel\",\"vocab\":{{{}}},\"unk_token\":\"[0]\"}}}}",
        entries.join(",")
    )
}

pub fn v3_container() -> tempfile::TempDir {
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(
        miniature_glimmer,
        checkpoint.path(),
        container.path(),
        "fact-id-fixture",
    );
    std::fs::write(
        container.path().join("tokenizer.json"),
        knn_tokenizer_json(G_VOCAB),
    )
    .unwrap();
    container
}

/// Windows temp paths contain backslashes, which the LQL lexer's escape pass
/// would consume.
pub fn lql_path(path: impl AsRef<Path>) -> String {
    path.as_ref().display().to_string().replace('\\', "\\\\")
}

pub fn try_run(session: &mut Session, stmt: &str) -> Result<Vec<String>, LqlError> {
    let parsed = parse(stmt).unwrap_or_else(|e| panic!("parse {stmt}: {e}"));
    session.execute(&parsed)
}

pub fn run(session: &mut Session, stmt: &str) -> String {
    try_run(session, stmt)
        .unwrap_or_else(|e| panic!("execute {stmt}: {e}"))
        .join("\n")
}

pub fn refused(session: &mut Session, stmt: &str) -> String {
    match try_run(session, stmt) {
        Ok(out) => panic!("{stmt} was not refused: {out:?}"),
        Err(e) => e.to_string(),
    }
}

pub fn bound(container: &Path) -> Session {
    let mut session = Session::new();
    run(&mut session, &format!("USE \"{}\";", lql_path(container)));
    session
}

pub fn facts(session: &Session) -> Vec<AssertedFact> {
    session.asserted_facts().expect("a vindex binding")
}

/// The one asserted fact of a session that holds exactly one.
pub fn only_fact(session: &Session) -> AssertedFact {
    let mut all = facts(session);
    assert_eq!(all.len(), 1, "expected one asserted fact: {all:?}");
    all.remove(0)
}

/// INFER on the canonical prompt, with timing lines removed.
pub fn infer(session: &mut Session, prompt: &str) -> String {
    run(session, &format!("INFER \"{prompt}\" TOP 3;"))
        .lines()
        .filter(|l| !l.trim_end().ends_with("ms"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// The target INFER ranks first on `prompt`.
pub fn infer_top1(session: &mut Session, prompt: &str) -> String {
    let out = infer(session, prompt);
    out.lines()
        .find(|l| l.trim_start().starts_with("1."))
        .unwrap_or_else(|| panic!("no row 1 in {out}"))
        .to_string()
}

/// Everything a statement could change, observed through the statement
/// surface and the typed catalogue: the asserted facts, every discovered row
/// of every layer, the KNN entries DESCRIBE shows, the patch list and the
/// open recording, and INFER on the canonical prompt.
pub fn observe(session: &mut Session) -> String {
    let facts = format!("{:?}", facts(session));
    let edges = run(session, "SELECT * FROM EDGES LIMIT 100000;");
    let describe = run(session, r#"DESCRIBE "a";"#);
    let patches = run(session, "SHOW PATCHES;");
    let infer = infer(session, CANONICAL_PROMPT);
    format!("{facts}\n--\n{edges}\n--\n{describe}\n--\n{patches}\n--\n{infer}")
}

/// Write `json` as a patch file in `dir` and return its LQL path.
pub fn write_patch(dir: &Path, name: &str, json: &serde_json::Value) -> String {
    let path = dir.join(name);
    std::fs::write(&path, serde_json::to_vec_pretty(json).unwrap()).unwrap();
    lql_path(path)
}

/// Record `statements` into a saved patch `name` in `dir` and return its
/// parsed JSON. The session is left with the edits live.
pub fn saved_patch(
    session: &mut Session,
    dir: &Path,
    name: &str,
    statements: &[&str],
) -> serde_json::Value {
    let path = dir.join(name);
    run(session, &format!("BEGIN PATCH \"{}\";", lql_path(&path)));
    for stmt in statements {
        run(session, stmt);
    }
    run(session, "SAVE PATCH;");
    serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap()
}
