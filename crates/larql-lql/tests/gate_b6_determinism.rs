//! Gate B, B6 (`docs/gate-b-b6.md` r2, R1 and D4): two independent sessions
//! that reach the same state write byte-identical output trees, for every
//! COMPILE target and for COMPACT INTO VINDEX.
//!
//! The comparison is over the whole tree: the same relative paths, the same
//! file kinds, and the same bytes in every file. Nothing is masked; the
//! permitted-nondeterminism list is empty (D4).
//!
//! `COMPILE INTO MODEL` is not covered: on `741ecdab` it fails for every
//! input before it writes (its weight writer expects an `index.json` in the
//! fresh output directory), so there is no output to compare. Its handling
//! awaits an owner decision under B6.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use larql_lql::{parse, Session};
use larql_vindex::format::vindex3::fixtures::{
    dense_f32_model, encode_fixture_container, miniature_glimmer, DENSE_VOCAB,
};

const V2_INSERT: &str = r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[3]", "[7]", "[5]") AT LAYER 0 MODE COMPOSE;"#;
const V3_INSERT: &str = r#"INSERT INTO EDGES (entity, relation, target) VALUES ("a", "b", "[5]") AT LAYER 1 MODE COMPOSE;"#;
/// KNN facts at several layers: the store is saved layer by layer, so the
/// layer order is part of the bytes of `knn_store.bin`.
const V2_KNN: &[&str] = &[
    r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[3]", "[7]", "[5]") AT LAYER 0 MODE KNN;"#,
    r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[4]", "[7]", "[6]") AT LAYER 1 MODE KNN;"#,
    r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[2]", "[7]", "[8]") AT LAYER 2 MODE KNN;"#,
    r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[1]", "[7]", "[9]") AT LAYER 3 MODE KNN;"#,
];
const V3_KNN: &[&str] = &[
    r#"INSERT INTO EDGES (entity, relation, target) VALUES ("a", "b", "c") AT LAYER 0 MODE KNN;"#,
    r#"INSERT INTO EDGES (entity, relation, target) VALUES ("c", "b", "a") AT LAYER 1 MODE KNN;"#,
    r#"INSERT INTO EDGES (entity, relation, target) VALUES ("a", "c", "b") AT LAYER 2 MODE KNN;"#,
    r#"INSERT INTO EDGES (entity, relation, target) VALUES ("c", "a", "b") AT LAYER 3 MODE KNN;"#,
];
/// How many times a KNN determinism test compiles. One pair of compiles
/// can agree by chance when a map happens to iterate in the same order;
/// several make that vanishingly unlikely.
const KNN_ROUNDS: usize = 4;
/// Top tokens per feature the extractor writes into `down_meta`.
const V2_DOWN_TOP_K: usize = 8;

fn lql_path(path: impl AsRef<Path>) -> String {
    path.as_ref().display().to_string().replace('\\', "\\\\")
}

fn run(session: &mut Session, stmt: &str) -> Vec<String> {
    let parsed = parse(stmt).unwrap_or_else(|e| panic!("parse {stmt}: {e}"));
    session
        .execute(&parsed)
        .unwrap_or_else(|e| panic!("{stmt}: {e}"))
}

fn bound(path: &Path) -> Session {
    let mut session = Session::new();
    run(&mut session, &format!("USE \"{}\";", lql_path(path)));
    session
}

/// One entry of an output tree: its kind and, for a file, its bytes.
#[derive(Debug, PartialEq, Eq)]
enum Entry {
    Dir,
    File(Vec<u8>),
    Symlink(PathBuf),
}

/// Every entry under `root`, by path relative to it.
fn tree(root: &Path) -> BTreeMap<String, Entry> {
    fn walk(root: &Path, dir: &Path, out: &mut BTreeMap<String, Entry>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            let rel = path
                .strip_prefix(root)
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/");
            let meta = std::fs::symlink_metadata(&path).unwrap();
            if meta.file_type().is_symlink() {
                out.insert(rel, Entry::Symlink(std::fs::read_link(&path).unwrap()));
            } else if meta.is_dir() {
                out.insert(rel, Entry::Dir);
                walk(root, &path, out);
            } else {
                out.insert(rel, Entry::File(std::fs::read(&path).unwrap()));
            }
        }
    }
    let mut out = BTreeMap::new();
    walk(root, root, &mut out);
    out
}

/// The two trees list the same entries with the same bytes. A difference
/// names every entry that differs, so a failure reports each
/// nondeterministic file at once.
fn assert_same_tree(a: &Path, b: &Path) {
    let (ta, tb) = (tree(a), tree(b));
    assert!(!ta.is_empty(), "{} is empty", a.display());
    let names_a: Vec<&String> = ta.keys().collect();
    let names_b: Vec<&String> = tb.keys().collect();
    assert_eq!(names_a, names_b, "the two outputs list different entries");
    let differing: Vec<&String> = ta
        .iter()
        .filter(|(name, entry)| tb.get(*name) != Some(entry))
        .map(|(name, _)| name)
        .collect();
    assert!(
        differing.is_empty(),
        "identical inputs gave different bytes in: {differing:?}"
    );
}

// ── Fixtures ────────────────────────────────────────────────────────────

/// The `[N]` ↔ id N tokenizer (as in `tests/fact_id_1`).
fn n_tokenizer(vocab: usize) -> String {
    let entries: Vec<String> = (0..vocab).map(|i| format!("\"[{i}]\":{i}")).collect();
    format!(
        "{{\"version\":\"1.0\",\"truncation\":null,\"padding\":null,\"added_tokens\":[],\
         \"normalizer\":{{\"type\":\"Strip\",\"strip_left\":true,\"strip_right\":true}},\
         \"pre_tokenizer\":{{\"type\":\"WhitespaceSplit\"}},\"post_processor\":null,\"decoder\":null,\
         \"model\":{{\"type\":\"WordLevel\",\"vocab\":{{{}}},\"unk_token\":\"[0]\"}}}}",
        entries.join(",")
    )
}

fn v3_tokenizer() -> String {
    let vocab = r#""[UNK]":0,"The":1,"of":2,"is":3,"a":4,"b":5,"c":6,"[5]":7,"[6]":8"#;
    format!(
        "{{\"version\":\"1.0\",\"truncation\":null,\"padding\":null,\"added_tokens\":[],\
         \"normalizer\":null,\"pre_tokenizer\":{{\"type\":\"Whitespace\"}},\
         \"post_processor\":null,\"decoder\":null,\
         \"model\":{{\"type\":\"WordLevel\",\"vocab\":{{{vocab}}},\"unk_token\":\"[UNK]\"}}}}"
    )
}

/// A VINDEX2 vindex extracted from the dense fixture model, with weights.
fn v2_vindex(dir: &Path) -> PathBuf {
    let checkpoint = dir.join("checkpoint");
    let vindex = dir.join("source.vindex");
    std::fs::create_dir_all(&checkpoint).unwrap();
    std::fs::create_dir_all(&vindex).unwrap();
    dense_f32_model(&checkpoint);
    let weights = larql_inference::load_model_dir(&checkpoint).unwrap();
    let json = n_tokenizer(DENSE_VOCAB);
    let tokenizer = larql_vindex::tokenizers::Tokenizer::from_bytes(json.as_bytes()).unwrap();
    larql_vindex::build_vindex(
        &weights,
        &tokenizer,
        "gate-b6",
        &vindex,
        V2_DOWN_TOP_K,
        larql_vindex::ExtractLevel::All,
        larql_vindex::StorageDtype::F32,
        &mut larql_vindex::SilentBuildCallbacks,
    )
    .unwrap();
    std::fs::write(vindex.join("tokenizer.json"), json).unwrap();
    vindex
}

fn v3_container(dir: &Path) -> PathBuf {
    let checkpoint = dir.join("checkpoint");
    let vindex = dir.join("container");
    std::fs::create_dir_all(&checkpoint).unwrap();
    std::fs::create_dir_all(&vindex).unwrap();
    encode_fixture_container(miniature_glimmer, &checkpoint, &vindex, "gate-b6");
    std::fs::write(vindex.join("tokenizer.json"), v3_tokenizer()).unwrap();
    vindex
}

/// Two sessions on `source`, each running `stmts` then `verb` into its own
/// destination; the two destinations must be identical trees.
fn assert_two_sessions_agree(source: &Path, stmts: &[&str], verb: &str, dir: &Path) {
    let outs = [dir.join("first.out"), dir.join("second.out")];
    for out in &outs {
        let mut session = bound(source);
        for stmt in stmts {
            run(&mut session, stmt);
        }
        run(&mut session, &format!("{verb} \"{}\";", lql_path(out)));
    }
    assert_same_tree(&outs[0], &outs[1]);
}

/// [`KNN_ROUNDS`] sessions on `source`, each running `stmts` then `verb`;
/// every output must equal the first.
fn assert_sessions_agree(source: &Path, stmts: &[&str], verb: &str, dir: &Path) {
    let outs: Vec<PathBuf> = (0..KNN_ROUNDS)
        .map(|i| dir.join(format!("round{i}.out")))
        .collect();
    for out in &outs {
        let mut session = bound(source);
        for stmt in stmts {
            run(&mut session, stmt);
        }
        run(&mut session, &format!("{verb} \"{}\";", lql_path(out)));
    }
    for out in &outs[1..] {
        assert_same_tree(&outs[0], out);
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[test]
fn vindex2_compile_into_vindex_is_byte_deterministic() {
    let dir = tempfile::tempdir().unwrap();
    let source = v2_vindex(dir.path());
    assert_two_sessions_agree(
        &source,
        &[V2_INSERT],
        "COMPILE CURRENT INTO VINDEX",
        dir.path(),
    );
}

#[test]
fn vindex2_compile_into_vindex_of_an_unedited_session_is_byte_deterministic() {
    let dir = tempfile::tempdir().unwrap();
    let source = v2_vindex(dir.path());
    assert_two_sessions_agree(&source, &[], "COMPILE CURRENT INTO VINDEX", dir.path());
}

#[test]
fn vindex3_compile_into_vindex_is_byte_deterministic() {
    let dir = tempfile::tempdir().unwrap();
    let source = v3_container(dir.path());
    assert_two_sessions_agree(
        &source,
        &[V3_INSERT],
        "COMPILE CURRENT INTO VINDEX",
        dir.path(),
    );
}

#[test]
fn vindex3_compact_into_vindex_is_byte_deterministic() {
    let dir = tempfile::tempdir().unwrap();
    let source = v3_container(dir.path());
    assert_two_sessions_agree(&source, &[], "COMPACT INTO VINDEX", dir.path());
}

#[test]
fn vindex2_compile_of_knn_facts_at_several_layers_is_byte_deterministic() {
    let dir = tempfile::tempdir().unwrap();
    let source = v2_vindex(dir.path());
    assert_sessions_agree(&source, V2_KNN, "COMPILE CURRENT INTO VINDEX", dir.path());
}

#[test]
fn vindex3_compile_of_knn_facts_at_several_layers_is_byte_deterministic() {
    let dir = tempfile::tempdir().unwrap();
    let source = v3_container(dir.path());
    assert_sessions_agree(&source, V3_KNN, "COMPILE CURRENT INTO VINDEX", dir.path());
}
