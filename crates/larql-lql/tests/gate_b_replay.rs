//! Gate B, B3 (`docs/gate-b.md`): a fresh session that applies the saved
//! patch(es) reaches the live session's state byte for byte, whether the
//! compose installs that share a layer went into one patch or two, on both
//! backends.
//!
//! A compose INSERT re-fits the vectors of earlier installs at its layer
//! (the batch refine) and the balance passes rescale them. Before the fix,
//! only the new install's op carried final vectors, so replay left every
//! earlier install with the vectors it had when it was recorded.

use std::path::{Path, PathBuf};

use larql_lql::{parse, FactStatus, Session};
use larql_vindex::format::vindex3::fixtures::{
    dense_f32_model, encode_fixture_container, miniature_glimmer, DENSE_VOCAB,
};

/// Two compose installs at one layer on the VINDEX3 fixture. Entity and
/// target are real fixture tokens, so the captures carry signal.
const V3_A: &str = r#"INSERT INTO EDGES (entity, relation, target) VALUES ("a", "b", "[5]") AT LAYER 1 MODE COMPOSE;"#;
const V3_C: &str = r#"INSERT INTO EDGES (entity, relation, target) VALUES ("c", "b", "[6]") AT LAYER 1 MODE COMPOSE;"#;
const V3_PROMPT: &str = "The b of a is";
/// The same on a VINDEX2 vindex extracted from the dense fixture model. (The
/// synthetic model of `larql-inference` gives both installs the same gate,
/// so its refine changes nothing and could not show the defect.)
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

fn bound(path: &Path) -> Session {
    let mut session = Session::new();
    run(&mut session, &format!("USE \"{}\";", lql_path(path)));
    session
}

fn infer(session: &mut Session, prompt: &str) -> Vec<String> {
    run(session, &format!("INFER \"{prompt}\" TOP 5;"))
        .into_iter()
        .filter(|l| !l.trim_end().ends_with("ms"))
        .collect()
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

struct Fixture {
    vindex: PathBuf,
    _dir: tempfile::TempDir,
}

fn v3() -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    let checkpoint = dir.path().join("checkpoint");
    let vindex = dir.path().join("container");
    std::fs::create_dir_all(&checkpoint).unwrap();
    std::fs::create_dir_all(&vindex).unwrap();
    encode_fixture_container(miniature_glimmer, &checkpoint, &vindex, "gate-b-replay");
    std::fs::write(vindex.join("tokenizer.json"), v3_tokenizer()).unwrap();
    Fixture { vindex, _dir: dir }
}

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
        "gate-b-replay",
        &vindex,
        V2_DOWN_TOP_K,
        larql_vindex::ExtractLevel::All,
        larql_vindex::StorageDtype::F32,
        &mut larql_vindex::SilentBuildCallbacks,
    )
    .unwrap();
    std::fs::write(vindex.join("tokenizer.json"), json).unwrap();
    Fixture { vindex, _dir: dir }
}

/// Every override vector on either overlay, as bits, by slot. VINDEX3 is
/// read through the session's state fingerprint (a hash over the bits of
/// every gate, up and down vector); VINDEX2 directly.
fn vectors(session: &mut Session) -> String {
    if let Some(pv) = session.patched_overlay_mut() {
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
        return format!("{gates:?}\n{ups:?}\n{downs:?}");
    }
    session.state_fingerprint().components()["overlay.vectors"].clone()
}

/// The ops of a saved patch file.
fn ops(path: &Path) -> Vec<serde_json::Value> {
    let patch: serde_json::Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    patch["operations"].as_array().unwrap().clone()
}

/// Record `inserts` into one saved patch, or one patch each, then replay in
/// a fresh session and compare vectors and INFER with the live session.
fn assert_replays_exactly(fx: &Fixture, inserts: &[&str], prompt: &str, one_patch: bool) {
    let dir = fx.vindex.parent().unwrap();
    let mut live = bound(&fx.vindex);
    let mut files = Vec::new();
    if one_patch {
        let path = dir.join("all.vlp");
        run(&mut live, &format!("BEGIN PATCH \"{}\";", lql_path(&path)));
        for stmt in inserts {
            run(&mut live, stmt);
        }
        run(&mut live, "SAVE PATCH;");
        files.push(path);
    } else {
        for (i, stmt) in inserts.iter().enumerate() {
            let path = dir.join(format!("p{i}.vlp"));
            run(&mut live, &format!("BEGIN PATCH \"{}\";", lql_path(&path)));
            run(&mut live, stmt);
            run(&mut live, "SAVE PATCH;");
            files.push(path);
        }
    }

    let mut replay = bound(&fx.vindex);
    for path in &files {
        run(&mut replay, &format!("APPLY PATCH \"{}\";", lql_path(path)));
    }
    assert_eq!(vectors(&mut replay), vectors(&mut live));
    assert_eq!(infer(&mut replay, prompt), infer(&mut live, prompt));
    // Without a re-fit the comparison above would hold trivially.
    let refits = files
        .iter()
        .flat_map(|f| ops(f))
        .filter(|op| op["op"] == "update")
        .count();
    assert!(
        refits > 0,
        "the second install must re-fit the first, or this test proves nothing"
    );

    // A re-fit is not a physical edit: both facts stay as asserted.
    for session in [&live, &replay] {
        let facts = session.asserted_facts().unwrap();
        assert_eq!(facts.len(), inserts.len(), "{facts:?}");
        for fact in facts {
            assert_eq!(fact.status, FactStatus::Compose, "{fact:?}");
            assert_eq!(fact.version, 1, "{fact:?}");
        }
    }
}

#[test]
fn vindex3_two_installs_in_one_patch_replay_exactly() {
    assert_replays_exactly(&v3(), &[V3_A, V3_C], V3_PROMPT, true);
}

#[test]
fn vindex3_two_installs_in_two_patches_replay_exactly() {
    assert_replays_exactly(&v3(), &[V3_A, V3_C], V3_PROMPT, false);
}

#[test]
fn vindex2_two_installs_in_one_patch_replay_exactly() {
    assert_replays_exactly(&v2(), &[V2_A, V2_C], V2_PROMPT, true);
}

#[test]
fn vindex2_two_installs_in_two_patches_replay_exactly() {
    assert_replays_exactly(&v2(), &[V2_A, V2_C], V2_PROMPT, false);
}

/// REBALANCE rescales the down vectors of installs already in saved
/// patches; its re-fits go into the next recording, so the stack replays
/// exactly.
fn assert_rebalance_replays_exactly(fx: &Fixture, inserts: &[&str], prompt: &str) {
    let dir = fx.vindex.parent().unwrap();
    let mut live = bound(&fx.vindex);
    let mut files = Vec::new();
    for (i, stmt) in inserts.iter().enumerate() {
        let path = dir.join(format!("p{i}.vlp"));
        run(&mut live, &format!("BEGIN PATCH \"{}\";", lql_path(&path)));
        run(&mut live, stmt);
        run(&mut live, "SAVE PATCH;");
        files.push(path);
    }
    let rebalance = dir.join("rebalance.vlp");
    run(
        &mut live,
        &format!("BEGIN PATCH \"{}\";", lql_path(&rebalance)),
    );
    let out = run(&mut live, "REBALANCE MAX 5 FLOOR 0.99 CEILING 0.999;");
    run(&mut live, "SAVE PATCH;");
    files.push(rebalance.clone());

    let mut replay = bound(&fx.vindex);
    for path in &files {
        run(&mut replay, &format!("APPLY PATCH \"{}\";", lql_path(path)));
    }
    assert_eq!(vectors(&mut replay), vectors(&mut live), "{out:?}");
    assert_eq!(infer(&mut replay, prompt), infer(&mut live, prompt));
    assert!(
        !ops(&rebalance).is_empty(),
        "REBALANCE must change the installs, or this test proves nothing: {out:?}"
    );
}

#[test]
fn vindex3_rebalance_over_saved_installs_replays_exactly() {
    assert_rebalance_replays_exactly(&v3(), &[V3_A, V3_C], V3_PROMPT);
}

#[test]
fn vindex2_rebalance_over_saved_installs_replays_exactly() {
    assert_rebalance_replays_exactly(&v2(), &[V2_A, V2_C], V2_PROMPT);
}

/// The semantic UPDATE of a fact: what a re-fit must never read as.
fn semantic_update(insert: &str) -> String {
    let values = insert.split("VALUES (").nth(1).unwrap();
    let parts: Vec<&str> = values.split('"').collect();
    let (entity, relation) = (parts[1], parts[3]);
    format!(
        r#"UPDATE EDGES SET target = "[6]" WHERE entity = "{entity}" AND relation = "{relation}";"#
    )
}

/// A re-fit in a saved patch is not a fact UPDATE. Every re-fit op is a
/// vector-only slot `update` (no `down_meta`), never `update_knn_fact`.
/// Applied in a fresh session, the facts keep status `compose` and version 1,
/// and a semantic UPDATE is still refused (D-6). A user's physical UPDATE of
/// the same slot does change the fact: it degrades it. The fold tells the
/// two apart.
fn assert_refits_are_not_fact_updates(fx: &Fixture, inserts: &[&str]) {
    let dir = fx.vindex.parent().unwrap();
    let mut live = bound(&fx.vindex);
    let mut files = Vec::new();
    for (i, stmt) in inserts.iter().enumerate() {
        let path = dir.join(format!("p{i}.vlp"));
        run(&mut live, &format!("BEGIN PATCH \"{}\";", lql_path(&path)));
        run(&mut live, stmt);
        run(&mut live, "SAVE PATCH;");
        files.push(path);
    }
    let rebalance = dir.join("rebalance.vlp");
    run(
        &mut live,
        &format!("BEGIN PATCH \"{}\";", lql_path(&rebalance)),
    );
    run(&mut live, "REBALANCE MAX 5 FLOOR 0.99 CEILING 0.999;");
    run(&mut live, "SAVE PATCH;");
    files.push(rebalance);

    let all: Vec<serde_json::Value> = files.iter().flat_map(|f| ops(f)).collect();
    let refits: Vec<&serde_json::Value> = all.iter().filter(|op| op["op"] == "update").collect();
    assert!(
        !refits.is_empty(),
        "the patches must carry re-fits: {all:?}"
    );
    for op in &refits {
        assert!(op["down_meta"].is_null(), "a re-fit names no target: {op}");
        assert!(
            ["gate_vector_b64", "up_vector_b64", "down_vector_b64"]
                .iter()
                .any(|k| op[*k].is_string()),
            "a re-fit carries vectors: {op}"
        );
    }
    assert!(
        all.iter().all(|op| op["op"] != "update_knn_fact"),
        "no re-fit is recorded as a fact UPDATE: {all:?}"
    );

    let mut replay = bound(&fx.vindex);
    for path in &files {
        run(&mut replay, &format!("APPLY PATCH \"{}\";", lql_path(path)));
    }
    let facts = replay.asserted_facts().unwrap();
    assert_eq!(facts.len(), inserts.len(), "{facts:?}");
    for fact in &facts {
        assert_eq!(fact.status, FactStatus::Compose, "{fact:?}");
        assert_eq!(fact.version, 1, "{fact:?}");
    }
    let stmt = semantic_update(inserts[0]);
    let why = replay
        .execute(&parse(&stmt).unwrap())
        .expect_err("a compose fact refuses semantic UPDATE")
        .to_string();
    assert!(why.contains("compose install"), "{why}");

    // The contrast: a physical UPDATE of the first fact's slot degrades it.
    let first = facts[0].clone();
    let &(layer, feature) = first.slots.first().expect("a compose fact owns a slot");
    run(
        &mut replay,
        &format!(
            "UPDATE EDGES SET confidence = 0.5 WHERE layer = {layer} AND feature = {feature};"
        ),
    );
    let after = replay.asserted_facts().unwrap();
    let edited = after.iter().find(|f| f.id == first.id).unwrap();
    assert_eq!(edited.status, FactStatus::Degraded, "{edited:?}");
}

#[test]
fn vindex3_refits_in_saved_patches_are_not_fact_updates() {
    assert_refits_are_not_fact_updates(&v3(), &[V3_A, V3_C]);
}

#[test]
fn vindex2_refits_in_saved_patches_are_not_fact_updates() {
    assert_refits_are_not_fact_updates(&v2(), &[V2_A, V2_C]);
}
