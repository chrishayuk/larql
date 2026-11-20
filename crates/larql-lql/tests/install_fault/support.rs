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

/// The `[N]` ↔ id N tokenizer for the VINDEX2 fixture.
fn n_tokenizer_json(vocab: usize) -> String {
    let entries: Vec<String> = (0..vocab).map(|i| format!("\"[{i}]\":{i}")).collect();
    format!(
        "{{\"version\":\"1.0\",\"truncation\":null,\"padding\":null,\"added_tokens\":[],\
         \"normalizer\":{{\"type\":\"Strip\",\"strip_left\":true,\"strip_right\":true}},\
         \"pre_tokenizer\":{{\"type\":\"WhitespaceSplit\"}},\"post_processor\":null,\"decoder\":null,\
         \"model\":{{\"type\":\"WordLevel\",\"vocab\":{{{}}},\"unk_token\":\"[0]\"}}}}",
        entries.join(",")
    )
}

/// Top tokens per feature the VINDEX2 extractor writes into `down_meta`.
const V2_DOWN_TOP_K: usize = 8;

pub const V2_INSERT: &str = r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[3]", "[7]", "[5]") AT LAYER 0 MODE COMPOSE;"#;
pub const V2_INSERT_SECOND: &str = r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[4]", "[7]", "[6]") AT LAYER 0 MODE COMPOSE;"#;

/// A VINDEX2 vindex extracted from the dense fixture model, whose installs
/// have distinct gates, so the refine and balance passes do real work.
pub struct V2Fixture {
    _checkpoint: tempfile::TempDir,
    pub dir: tempfile::TempDir,
}

impl V2Fixture {
    pub fn new() -> Self {
        use larql_vindex::format::vindex3::fixtures::{dense_f32_model, DENSE_VOCAB};
        let checkpoint = tempfile::tempdir().unwrap();
        let dir = tempfile::tempdir().unwrap();
        dense_f32_model(checkpoint.path());
        let weights = larql_inference::load_model_dir(checkpoint.path()).unwrap();
        let json = n_tokenizer_json(DENSE_VOCAB);
        let tokenizer = larql_vindex::tokenizers::Tokenizer::from_bytes(json.as_bytes()).unwrap();
        larql_vindex::build_vindex(
            &weights,
            &tokenizer,
            "install-fault-v2",
            dir.path(),
            V2_DOWN_TOP_K,
            larql_vindex::ExtractLevel::All,
            larql_vindex::StorageDtype::F32,
            &mut larql_vindex::SilentBuildCallbacks,
        )
        .unwrap();
        std::fs::write(dir.path().join("tokenizer.json"), json).unwrap();
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

/// Everything an install can change, as text: every state fingerprint
/// component, the VINDEX2 override vectors and KNN count as bits (the
/// fingerprint's overlay components cover VINDEX3 only), and the asserted
/// facts.
pub fn state(session: &mut Session) -> String {
    let fingerprint = format!("{:?}", session.state_fingerprint().components());
    let v2 = session
        .patched_overlay_mut()
        .map(|pv| {
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
            let mut gates: Vec<_> = pv
                .overrides_gate_iter()
                .map(|(l, f, g)| ((l, f), bits(g)))
                .collect();
            let mut ups: Vec<_> = pv
                .up_overrides()
                .iter()
                .map(|(k, v)| (*k, bits(v)))
                .collect();
            let mut downs: Vec<_> = pv
                .down_overrides()
                .iter()
                .map(|(k, v)| (*k, bits(v)))
                .collect();
            gates.sort();
            ups.sort();
            downs.sort();
            format!("{gates:?}\n{ups:?}\n{downs:?}\nknn {}", pv.knn_store.len())
        })
        .unwrap_or_default();
    let facts = format!("{:?}", session.asserted_facts().unwrap());
    format!("{fingerprint}\n--\n{v2}\n--\n{facts}")
}
