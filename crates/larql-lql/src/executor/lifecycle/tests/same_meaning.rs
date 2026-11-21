//! Gate B, B6 §3.3: the check a compacted container must pass before it is
//! published, that it means exactly what its source means.

use std::path::{Path, PathBuf};

use larql_vindex::format::vindex3::fixtures::{encode_fixture_container, miniature_glimmer};

use super::super::compact_into::require_same_meaning;
use crate::{parse, Session};

const INSERT: &str = r#"INSERT INTO EDGES (entity, relation, target) VALUES ("a", "b", "[5]") AT LAYER 1 MODE COMPOSE;"#;

fn tokenizer() -> String {
    let vocab = r#""[UNK]":0,"The":1,"of":2,"is":3,"a":4,"b":5,"c":6,"[5]":7,"[6]":8"#;
    format!(
        "{{\"version\":\"1.0\",\"truncation\":null,\"padding\":null,\"added_tokens\":[],\
         \"normalizer\":null,\"pre_tokenizer\":{{\"type\":\"Whitespace\"}},\
         \"post_processor\":null,\"decoder\":null,\
         \"model\":{{\"type\":\"WordLevel\",\"vocab\":{{{vocab}}},\"unk_token\":\"[UNK]\"}}}}"
    )
}

fn container(dir: &Path) -> PathBuf {
    let checkpoint = dir.join("checkpoint");
    let container = dir.join("container");
    std::fs::create_dir_all(&checkpoint).unwrap();
    std::fs::create_dir_all(&container).unwrap();
    encode_fixture_container(miniature_glimmer, &checkpoint, &container, "same-meaning");
    std::fs::write(container.join("tokenizer.json"), tokenizer()).unwrap();
    container
}

/// `path` spelled for an LQL string literal (backslashes escaped).
fn lql_path(path: &Path) -> String {
    path.display().to_string().replace('\\', "\\\\")
}

fn run(session: &mut Session, stmt: &str) {
    session.execute(&parse(stmt).unwrap()).unwrap();
}

#[test]
fn a_container_means_what_it_means() {
    let dir = tempfile::tempdir().unwrap();
    let source = container(dir.path());
    require_same_meaning(&source, &source).unwrap();
}

#[test]
fn a_container_with_an_edit_baked_in_does_not_mean_what_its_source_means() {
    let dir = tempfile::tempdir().unwrap();
    let source = container(dir.path());
    let edited = dir.path().join("edited");
    let mut session = Session::new();
    run(&mut session, &format!("USE \"{}\";", lql_path(&source)));
    run(&mut session, INSERT);
    run(
        &mut session,
        &format!("COMPILE CURRENT INTO VINDEX \"{}\";", lql_path(&edited)),
    );

    let err = require_same_meaning(&source, &edited)
        .unwrap_err()
        .to_string();
    assert!(err.contains("does not mean what the source means"), "{err}");
}
