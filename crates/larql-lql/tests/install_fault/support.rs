//! A COMPOSE-capable container, a session on it, and ways to look at an install and what a failed one left.

use std::path::{Path, PathBuf};

use larql_lql::{parse, Session};
use larql_vindex::format::vindex3::fixtures::{encode_fixture_container, miniature_glimmer};

pub const INSERT: &str = r#"INSERT INTO EDGES (entity, relation, target) VALUES ("a", "b", "[5]") AT LAYER 1 MODE COMPOSE;"#;
pub const INSERT_SECOND: &str = r#"INSERT INTO EDGES (entity, relation, target) VALUES ("c", "b", "[6]") AT LAYER 1 MODE COMPOSE;"#;

/// A word-level tokenizer with a whitespace pre-tokenizer, so distinct prompts tokenize
/// differently (the plain synthetic tokenizer maps every prose prompt to one `[UNK]`).
fn word_tokenizer_json() -> String {
    let vocab = r#""[UNK]":0,"The":1,"of":2,"is":3,"a":4,"b":5,"c":6,"[5]":7,"[6]":8"#;
    format!(
        "{{\"version\":\"1.0\",\"truncation\":null,\"padding\":null,\"added_tokens\":[],\
         \"normalizer\":null,\"pre_tokenizer\":{{\"type\":\"Whitespace\"}},\
         \"post_processor\":null,\"decoder\":null,\
         \"model\":{{\"type\":\"WordLevel\",\"vocab\":{{{vocab}}},\"unk_token\":\"[UNK]\"}}}}"
    )
}

pub struct Fixture {
    _checkpoint: tempfile::TempDir,
    pub dir: tempfile::TempDir,
}

impl Fixture {
    pub fn new() -> Self {
        let checkpoint = tempfile::tempdir().unwrap();
        let dir = tempfile::tempdir().unwrap();
        encode_fixture_container(
            miniature_glimmer,
            checkpoint.path(),
            dir.path(),
            "trace-fixture",
        );
        std::fs::write(dir.path().join("tokenizer.json"), word_tokenizer_json()).unwrap();
        Self {
            _checkpoint: checkpoint,
            dir,
        }
    }

    pub fn session(&self) -> Session {
        let mut s = Session::new();
        run(&mut s, &format!("USE \"{}\";", q(self.dir.path())));
        s
    }

    pub fn path(&self, name: &str) -> PathBuf {
        self.dir.path().join(name)
    }
}

pub fn q(path: &Path) -> String {
    path.display().to_string().replace('\\', "\\\\")
}

pub fn run(s: &mut Session, stmt: &str) -> Vec<String> {
    let parsed = parse(stmt).unwrap_or_else(|e| panic!("{stmt}: {e}"));
    s.execute(&parsed).unwrap_or_else(|e| panic!("{stmt}: {e}"))
}

/// Like [`run`], but the statement's error is returned as text.
pub fn try_run(s: &mut Session, stmt: &str) -> Result<Vec<String>, String> {
    let parsed = parse(stmt).map_err(|e| format!("parse: {e}"))?;
    s.execute(&parsed).map_err(|e| e.to_string())
}

/// Installs the given statements, saves the recording, and returns the saved bytes: the patch
/// carries the final post-refine, post-balance vectors, so equal bytes mean an equal install.
pub fn install_and_save(
    fixture: &Fixture,
    session: &mut Session,
    name: &str,
    inserts: &[&str],
) -> Vec<u8> {
    let patch = fixture.path(name);
    run(session, &format!("BEGIN PATCH \"{}\";", q(&patch)));
    for stmt in inserts {
        run(session, stmt);
    }
    run(session, "SAVE PATCH;");
    std::fs::read(&patch).unwrap()
}

/// What the session answers to a prompt, as text, without the elapsed-time line (the only part
/// of the output that differs between two identical runs).
pub fn infer(session: &mut Session, prompt: &str) -> Vec<String> {
    run(session, &format!("INFER \"{prompt}\" TOP 5;"))
        .into_iter()
        .filter(|line| !line.trim_end().ends_with("ms"))
        .collect()
}
