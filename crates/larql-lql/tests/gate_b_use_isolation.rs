//! Gate B, B1 (`docs/gate-b.md`): USE isolation. A re-USE of the same path,
//! or a USE of another vindex, leaves the session in the state of a fresh
//! session bound there, on both backends. That is checked on the session
//! state fingerprint (every install cache, the recording, the counters, the
//! resolver cache), on the override vectors bit for bit, and on what later
//! statements do: an INSERT, or the same patch set applied, gives exactly
//! what it gives in a fresh session.
//!
//! The other direction: a USE that fails leaves the previous session as it
//! was, its open recording included. And an open recording a successful USE
//! discards is named in USE's output, never dropped silently.
//!
//! Before the fix, the install caches survived USE, and the compose refine
//! of the next INSERT Gram-Schmidted against the previous binding's
//! residuals: the INSERT wrote different vectors than in a fresh session.

use std::path::{Path, PathBuf};

use larql_lql::{parse, Session};
use larql_vindex::format::vindex3::fixtures::{
    dense_f32_model, encode_fixture_container, miniature_glimmer, DENSE_VOCAB,
};

const V3_A: &str = r#"INSERT INTO EDGES (entity, relation, target) VALUES ("a", "b", "[5]") AT LAYER 1 MODE COMPOSE;"#;
const V3_C: &str = r#"INSERT INTO EDGES (entity, relation, target) VALUES ("c", "b", "[6]") AT LAYER 1 MODE COMPOSE;"#;
const V3_PROMPT: &str = "The b of a is";
const V2_A: &str = r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[3]", "[7]", "[5]") AT LAYER 0 MODE COMPOSE;"#;
const V2_C: &str = r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[4]", "[7]", "[6]") AT LAYER 0 MODE COMPOSE;"#;
const V2_PROMPT: &str = "The [7] of [3] is";
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

fn use_stmt(path: &Path) -> String {
    format!("USE \"{}\";", lql_path(path))
}

fn bound(path: &Path) -> Session {
    let mut session = Session::new();
    run(&mut session, &use_stmt(path));
    session
}

/// The `[N]`-and-words tokenizer of the VINDEX3 audit probe.
fn v3_tokenizer() -> String {
    let vocab = r#""[UNK]":0,"The":1,"of":2,"is":3,"a":4,"b":5,"c":6,"[5]":7,"[6]":8"#;
    format!(
        "{{\"version\":\"1.0\",\"truncation\":null,\"padding\":null,\"added_tokens\":[],\
         \"normalizer\":null,\"pre_tokenizer\":{{\"type\":\"Whitespace\"}},\
         \"post_processor\":null,\"decoder\":null,\
         \"model\":{{\"type\":\"WordLevel\",\"vocab\":{{{vocab}}},\"unk_token\":\"[UNK]\"}}}}"
    )
}

/// The `[N]` ↔ id N tokenizer.
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

struct Fixture {
    vindex: PathBuf,
    a: &'static str,
    c: &'static str,
    prompt: &'static str,
    _dir: tempfile::TempDir,
}

fn v3() -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    let checkpoint = dir.path().join("checkpoint");
    let vindex = dir.path().join("container");
    std::fs::create_dir_all(&checkpoint).unwrap();
    std::fs::create_dir_all(&vindex).unwrap();
    encode_fixture_container(miniature_glimmer, &checkpoint, &vindex, "gate-b-use");
    std::fs::write(vindex.join("tokenizer.json"), v3_tokenizer()).unwrap();
    Fixture {
        vindex,
        a: V3_A,
        c: V3_C,
        prompt: V3_PROMPT,
        _dir: dir,
    }
}

/// A VINDEX2 vindex extracted from the dense fixture model: its installs
/// have distinct gates, so the refine pass reads the residual caches.
fn v2() -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    let checkpoint = dir.path().join("checkpoint");
    let vindex = dir.path().join("source.vindex");
    std::fs::create_dir_all(&checkpoint).unwrap();
    std::fs::create_dir_all(&vindex).unwrap();
    dense_f32_model(&checkpoint);
    let weights = larql_inference::load_model_dir(&checkpoint).unwrap();
    let json = n_tokenizer(DENSE_VOCAB);
    let tokenizer = larql_vindex::tokenizers::Tokenizer::from_bytes(json.as_bytes()).unwrap();
    larql_vindex::build_vindex(
        &weights,
        &tokenizer,
        "gate-b-use",
        &vindex,
        V2_DOWN_TOP_K,
        larql_vindex::ExtractLevel::All,
        larql_vindex::StorageDtype::F32,
        &mut larql_vindex::SilentBuildCallbacks,
    )
    .unwrap();
    std::fs::write(vindex.join("tokenizer.json"), json).unwrap();
    Fixture {
        vindex,
        a: V2_A,
        c: V2_C,
        prompt: V2_PROMPT,
        _dir: dir,
    }
}

/// The whole session state: every component of the state fingerprint (the
/// VINDEX3 overlay, the install caches, the recording, the counters, the
/// resolver cache), the VINDEX2 override vectors and KNN entries as bits, the
/// asserted facts, the patch list and INFER.
fn state(session: &mut Session, prompt: &str) -> String {
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
    let infer: Vec<String> = run(session, &format!("INFER \"{prompt}\" TOP 5;"))
        .into_iter()
        .filter(|l| !l.trim_end().ends_with("ms"))
        .collect();
    format!("{fingerprint}\n--\n{v2}\n--\n{facts}\n--\n{infer:?}")
}

/// Two installs in a recording, then a re-USE of the same path: the session
/// equals a fresh one bound there, and so does what an INSERT then does.
fn assert_reuse_is_fresh(fx: &Fixture) {
    let dir = fx.vindex.parent().unwrap();
    let mut session = bound(&fx.vindex);
    let saved = dir.join("saved.vlp");
    run(
        &mut session,
        &format!("BEGIN PATCH \"{}\";", lql_path(&saved)),
    );
    run(&mut session, fx.a);
    run(&mut session, fx.c);
    run(&mut session, "SAVE PATCH;");
    run(&mut session, &use_stmt(&fx.vindex));

    let mut fresh = bound(&fx.vindex);
    assert_eq!(state(&mut session, fx.prompt), state(&mut fresh, fx.prompt));

    run(&mut session, fx.a);
    run(&mut fresh, fx.a);
    assert_eq!(
        state(&mut session, fx.prompt),
        state(&mut fresh, fx.prompt),
        "an INSERT after re-USE must write what it writes in a fresh session"
    );
}

#[test]
fn vindex3_reuse_of_the_same_path_is_a_fresh_session() {
    assert_reuse_is_fresh(&v3());
}

#[test]
fn vindex2_reuse_of_the_same_path_is_a_fresh_session() {
    assert_reuse_is_fresh(&v2());
}

/// The same with a patch set: re-USE then APPLY equals a fresh session that
/// applies the same patches.
fn assert_reuse_then_apply_is_fresh(fx: &Fixture) {
    let dir = fx.vindex.parent().unwrap();
    let patch = dir.join("c.vlp");
    let mut author = bound(&fx.vindex);
    run(
        &mut author,
        &format!("BEGIN PATCH \"{}\";", lql_path(&patch)),
    );
    run(&mut author, fx.c);
    run(&mut author, "SAVE PATCH;");

    let mut session = bound(&fx.vindex);
    run(&mut session, fx.a);
    run(&mut session, &use_stmt(&fx.vindex));
    run(
        &mut session,
        &format!("APPLY PATCH \"{}\";", lql_path(&patch)),
    );
    run(&mut session, fx.a);

    let mut fresh = bound(&fx.vindex);
    run(
        &mut fresh,
        &format!("APPLY PATCH \"{}\";", lql_path(&patch)),
    );
    run(&mut fresh, fx.a);
    assert_eq!(state(&mut session, fx.prompt), state(&mut fresh, fx.prompt));
}

#[test]
fn vindex3_reuse_then_apply_is_a_fresh_session() {
    assert_reuse_then_apply_is_fresh(&v3());
}

#[test]
fn vindex2_reuse_then_apply_is_a_fresh_session() {
    assert_reuse_then_apply_is_fresh(&v2());
}

/// A USE of another vindex, across backends: installs on VINDEX2, then USE
/// of a VINDEX3 container, equals a fresh session bound to the container,
/// including what an INSERT there writes. And back.
#[test]
fn use_of_another_vindex_is_a_fresh_session_there() {
    let (on_v2, on_v3) = (v2(), v3());
    let mut session = bound(&on_v2.vindex);
    run(&mut session, on_v2.a);
    run(&mut session, on_v2.c);
    run(&mut session, &use_stmt(&on_v3.vindex));
    run(&mut session, on_v3.a);
    let mut fresh = bound(&on_v3.vindex);
    run(&mut fresh, on_v3.a);
    assert_eq!(
        state(&mut session, on_v3.prompt),
        state(&mut fresh, on_v3.prompt)
    );

    run(&mut session, on_v3.c);
    run(&mut session, &use_stmt(&on_v2.vindex));
    run(&mut session, on_v2.a);
    let mut fresh = bound(&on_v2.vindex);
    run(&mut fresh, on_v2.a);
    assert_eq!(
        state(&mut session, on_v2.prompt),
        state(&mut fresh, on_v2.prompt)
    );
}

/// A USE that fails changes nothing: the previous binding, its edits, its
/// install caches and its open recording are all as they were, and SAVE
/// PATCH still writes the recording.
fn assert_failed_use_changes_nothing(fx: &Fixture) {
    let dir = fx.vindex.parent().unwrap();
    let saved = dir.join("kept.vlp");
    let mut session = bound(&fx.vindex);
    run(
        &mut session,
        &format!("BEGIN PATCH \"{}\";", lql_path(&saved)),
    );
    run(&mut session, fx.a);
    let before = state(&mut session, fx.prompt);

    for target in [dir.join("missing.vindex"), dir.join("not-a-vindex")] {
        if target.ends_with("not-a-vindex") {
            std::fs::create_dir_all(&target).unwrap();
        }
        let stmt = use_stmt(&target);
        let err = session.execute(&parse(&stmt).unwrap());
        assert!(err.is_err(), "{stmt} must fail");
        assert_eq!(state(&mut session, fx.prompt), before, "{stmt}");
    }

    run(&mut session, "SAVE PATCH;");
    let patch: serde_json::Value = serde_json::from_slice(&std::fs::read(&saved).unwrap()).unwrap();
    assert!(
        !patch["operations"].as_array().unwrap().is_empty(),
        "the recording survived the failed USE: {patch}"
    );
}

#[test]
fn vindex3_a_failed_use_changes_nothing() {
    assert_failed_use_changes_nothing(&v3());
}

#[test]
fn vindex2_a_failed_use_changes_nothing() {
    assert_failed_use_changes_nothing(&v2());
}

/// A successful USE over an open recording names what it discards.
#[test]
fn use_names_the_open_recording_it_discards() {
    let fx = v3();
    let dir = fx.vindex.parent().unwrap();
    let mut session = bound(&fx.vindex);

    let quiet = run(&mut session, &use_stmt(&fx.vindex));
    assert!(
        !quiet.iter().any(|l| l.contains("Discarded")),
        "no recording, nothing to name: {quiet:?}"
    );

    run(&mut session, fx.a);
    let out = run(&mut session, &use_stmt(&fx.vindex)).join("\n");
    assert!(out.contains("Discarded the anonymous recording"), "{out}");
    assert!(out.contains("1 unsaved operation(s)"), "{out}");
    assert!(out.contains("SAVE PATCH"), "{out}");

    let named = dir.join("named.vlp");
    run(
        &mut session,
        &format!("BEGIN PATCH \"{}\";", lql_path(&named)),
    );
    let out = run(&mut session, &use_stmt(&fx.vindex)).join("\n");
    assert!(
        out.contains(&format!("the recording for {}", named.display())),
        "{out}"
    );
    assert!(out.contains("0 unsaved operation(s)"), "{out}");
}
