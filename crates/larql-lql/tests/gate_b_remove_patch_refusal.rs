//! Gate B, B2 (`docs/gate-b.md`): REMOVE PATCH refuses before it changes
//! anything while the session holds a knowledge edit that no applied patch
//! holds.
//!
//! Removal rebuilds the state from the bound vindex and the remaining
//! patches. Before the refusal, an edit made in the session (saved or not)
//! was silently dropped, and replaying it without the removed patch is not
//! defined: apply A, edit B, apply C, remove A cannot be assumed to leave B
//! with the effect it had.
//!
//! The refusal text is pinned in full: it names the patch, lists the blocking
//! edits in journal order (a fact id only for an op that carries a fact key),
//! says that edits held by applied patches do not block, and gives a recovery
//! that runs. An unknown patch name is "not found" whatever the session holds.

use std::path::{Path, PathBuf};

use larql_inference::test_utils::write_synthetic_model_dir;
use larql_lql::{parse, Session};
use larql_vindex::format::vindex3::fixtures::{encode_fixture_container, miniature_glimmer};

const V2_INSERT: &str = r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[1]", "capital", "[2]") AT LAYER 0 MODE COMPOSE;"#;
const V2_PROMPT: &str = "[1] capital";
const V3_INSERT: &str = r#"INSERT INTO EDGES (entity, relation, target) VALUES ("a", "b", "[5]") AT LAYER 1 MODE COMPOSE;"#;
const V3_PROMPT: &str = "The b of a is";
const V3_MODEL: &str = "gate-b-refusal";
/// The session edit B: a tombstone, a knowledge edit on either backend.
const SESSION_EDIT: &str = "DELETE FROM EDGES WHERE layer = 0 AND feature = 0;";
/// How the refusal lists [`SESSION_EDIT`]: a physical op, by its address.
const SESSION_EDIT_LINE: &str = "  delete at layer 0 feature 0";
/// A keyed compose install made in the session, for the fact-id lines.
const V3_SESSION_INSERT: &str = r#"INSERT INTO EDGES (entity, relation, target) VALUES ("c", "b", "[6]") AT LAYER 1 MODE COMPOSE;"#;

/// The full refusal text for removing `path` over `edits`, with `save` the
/// clause that saves the open recording ("" when none is open).
fn refusal(path: &str, edits: &[&str], save: &str) -> String {
    let mut lines = vec![
        format!("REMOVE PATCH refused: {path:?}. Nothing was changed."),
        format!(
            "Removing a patch rebuilds the state from the vindex and the remaining patches. \
             This session holds {} knowledge edit(s) that no applied patch holds, and they \
             would be dropped:",
            edits.len()
        ),
    ];
    lines.extend(edits.iter().map(|l| l.to_string()));
    lines.push("Edits held by applied patches do not block removal.".into());
    lines.push(format!(
        "To remove the patch and keep these edits: {save}bind the vindex again with USE; then \
         APPLY the patches to keep, including the file(s) this session's edits were saved to."
    ));
    lines.join("\n")
}

const SAVE_ANONYMOUS: &str =
    "save the open recording with BEGIN PATCH \"file.vlp\" then SAVE PATCH; ";

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
    prompt: &'static str,
    model: &'static str,
    _dir: tempfile::TempDir,
}

fn v2() -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    let vindex = dir.path().join("source.vindex");
    std::fs::create_dir_all(&vindex).unwrap();
    write_synthetic_model_dir(&vindex).unwrap();
    Fixture {
        vindex,
        prompt: V2_PROMPT,
        model: "synthetic/tinymodel",
        _dir: dir,
    }
}

fn v3() -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    let checkpoint = dir.path().join("checkpoint");
    let vindex = dir.path().join("container");
    std::fs::create_dir_all(&checkpoint).unwrap();
    std::fs::create_dir_all(&vindex).unwrap();
    encode_fixture_container(miniature_glimmer, &checkpoint, &vindex, V3_MODEL);
    std::fs::write(vindex.join("tokenizer.json"), v3_tokenizer()).unwrap();
    Fixture {
        vindex,
        prompt: V3_PROMPT,
        model: V3_MODEL,
        _dir: dir,
    }
}

/// Everything the refusal must leave as it was: the override vectors as bits
/// (VINDEX2 directly, VINDEX3 through the state fingerprint), the asserted
/// facts, every row, the patch list and recording, and INFER. Without
/// `caches`, VINDEX3's session install caches are left out: USE does not
/// reset them yet (Gate B, B1), and they are not effective state.
fn observe_with(session: &mut Session, prompt: &str, caches: bool) -> String {
    let vectors = if let Some(pv) = session.patched_overlay_mut() {
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
    } else {
        let fp = session.state_fingerprint();
        let parts: Vec<_> = fp
            .components()
            .iter()
            .filter(|(k, _)| caches || k.starts_with("overlay."))
            .collect();
        format!("{parts:?}")
    };
    let facts = format!("{:?}", session.asserted_facts().unwrap());
    let rows = run(session, "SELECT * FROM EDGES LIMIT 100000;");
    let patches = run(session, "SHOW PATCHES;");
    let infer: Vec<String> = run(session, &format!("INFER \"{prompt}\" TOP 5;"))
        .into_iter()
        .filter(|l| !l.trim_end().ends_with("ms"))
        .collect();
    format!("{vectors}\n--\n{facts}\n--\n{rows:?}\n--\n{patches:?}\n--\n{infer:?}")
}

fn observe(session: &mut Session, prompt: &str) -> String {
    observe_with(session, prompt, true)
}

/// Apply A, make session edit B, apply C, then REMOVE PATCH A: refused, and
/// the session is byte for byte as it was. With `save`, B was recorded and
/// saved first, which still leaves it live and owned by no applied patch.
fn assert_remove_refuses(fx: &Fixture, insert: &str, save: bool) {
    let dir = fx.vindex.parent().unwrap();
    let a = dir.join("a.vlp");
    let mut author = bound(&fx.vindex);
    run(&mut author, &format!("BEGIN PATCH \"{}\";", lql_path(&a)));
    run(&mut author, insert);
    run(&mut author, "SAVE PATCH;");
    let c = dir.join("c.vlp");
    let empty = serde_json::json!({"version": 1, "base_model": fx.model, "created_at": "", "operations": []});
    std::fs::write(&c, empty.to_string()).unwrap();

    let mut session = bound(&fx.vindex);
    run(&mut session, &format!("APPLY PATCH \"{}\";", lql_path(&a)));
    if save {
        let b = dir.join("b.vlp");
        run(&mut session, &format!("BEGIN PATCH \"{}\";", lql_path(&b)));
        run(&mut session, SESSION_EDIT);
        run(&mut session, "SAVE PATCH;");
    } else {
        run(&mut session, SESSION_EDIT);
    }
    run(&mut session, &format!("APPLY PATCH \"{}\";", lql_path(&c)));
    let before = observe(&mut session, fx.prompt);

    let stmt = format!("REMOVE PATCH \"{}\";", lql_path(&a));
    let why = session
        .execute(&parse(&stmt).unwrap())
        .expect_err("REMOVE PATCH must refuse while a session edit is live")
        .to_string();
    // Unsaved, the edit sits in the anonymous recording; saved, no recording
    // is open and the edit is in b.vlp.
    let save_clause = if save { "" } else { SAVE_ANONYMOUS };
    assert!(
        why.ends_with(&refusal(&lql_path(&a), &[SESSION_EDIT_LINE], save_clause)),
        "{why}"
    );
    assert_eq!(observe(&mut session, fx.prompt), before);

    // The same refusal again: the text depends only on the state.
    let again = session.execute(&parse(&stmt).unwrap()).unwrap_err();
    assert_eq!(again.to_string(), why);

    // An unknown name is "not found", whatever the session holds.
    let typo = session
        .execute(&parse("REMOVE PATCH \"typo.vlp\";").unwrap())
        .unwrap_err()
        .to_string();
    assert!(typo.ends_with("patch not found: typo.vlp"), "{typo}");
    assert_eq!(observe(&mut session, fx.prompt), before);

    // The recovery the text gives runs, step by step, and ends where a fresh
    // session applying the kept patches does.
    let b = dir.join("b.vlp");
    if !save {
        run(&mut session, &format!("BEGIN PATCH \"{}\";", lql_path(&b)));
        run(&mut session, "SAVE PATCH;");
    }
    run(&mut session, &format!("USE \"{}\";", lql_path(&fx.vindex)));
    for patch in [&a, &c, &b] {
        run(
            &mut session,
            &format!("APPLY PATCH \"{}\";", lql_path(patch)),
        );
    }
    run(&mut session, &stmt);
    let mut want = bound(&fx.vindex);
    for patch in [&c, &b] {
        run(&mut want, &format!("APPLY PATCH \"{}\";", lql_path(patch)));
    }
    assert_eq!(
        observe_with(&mut session, fx.prompt, false),
        observe_with(&mut want, fx.prompt, false)
    );
}

#[test]
fn vindex2_remove_patch_refuses_over_an_unsaved_session_edit() {
    assert_remove_refuses(&v2(), V2_INSERT, false);
}

#[test]
fn vindex2_remove_patch_refuses_over_a_saved_session_edit() {
    assert_remove_refuses(&v2(), V2_INSERT, true);
}

#[test]
fn vindex3_remove_patch_refuses_over_an_unsaved_session_edit() {
    assert_remove_refuses(&v3(), V3_INSERT, false);
}

#[test]
fn vindex3_remove_patch_refuses_over_a_saved_session_edit() {
    assert_remove_refuses(&v3(), V3_INSERT, true);
}

/// A keyed edit is listed with its fact id, every op in journal order; a
/// named open recording is saved by name.
#[test]
fn the_refusal_lists_keyed_and_physical_edits_in_journal_order() {
    let fx = v3();
    let dir = fx.vindex.parent().unwrap();
    let a = dir.join("a.vlp");
    let mut author = bound(&fx.vindex);
    run(&mut author, &format!("BEGIN PATCH \"{}\";", lql_path(&a)));
    run(&mut author, V3_INSERT);
    run(&mut author, "SAVE PATCH;");

    let mut session = bound(&fx.vindex);
    run(&mut session, &format!("APPLY PATCH \"{}\";", lql_path(&a)));
    let named = dir.join("named.vlp");
    run(
        &mut session,
        &format!("BEGIN PATCH \"{}\";", lql_path(&named)),
    );
    run(&mut session, SESSION_EDIT);
    run(&mut session, V3_SESSION_INSERT);
    let slots: Vec<(usize, usize)> = session
        .asserted_facts()
        .unwrap()
        .iter()
        .find(|f| f.entity == "c")
        .expect("the session's fact")
        .slots
        .clone();
    let id = larql_vindex::FactKey::new("c", "b").fact_id();
    let mut edits = vec![SESSION_EDIT_LINE.to_string()];
    edits.extend(
        slots
            .iter()
            .map(|(l, f)| format!("  insert, fact {id}, at layer {l} feature {f}")),
    );
    let edits: Vec<&str> = edits.iter().map(String::as_str).collect();
    let before = observe(&mut session, fx.prompt);

    let why = session
        .execute(&parse(&format!("REMOVE PATCH \"{}\";", lql_path(&a))).unwrap())
        .unwrap_err()
        .to_string();
    let save = format!(
        "save the open recording to {} with SAVE PATCH; ",
        lql_path(&named)
    );
    assert!(
        why.ends_with(&refusal(&lql_path(&a), &edits, &save)),
        "{why}"
    );
    assert_eq!(observe(&mut session, fx.prompt), before);
}

/// The way out the refusal names: bind again, apply the patches wanted. The
/// removal then runs, with no session edit to drop.
#[test]
fn rebinding_and_applying_the_saved_edit_lets_the_removal_run() {
    let fx = v3();
    let dir = fx.vindex.parent().unwrap();
    let a = dir.join("a.vlp");
    let b = dir.join("b.vlp");
    let mut session = bound(&fx.vindex);
    run(&mut session, &format!("BEGIN PATCH \"{}\";", lql_path(&a)));
    run(&mut session, V3_INSERT);
    run(&mut session, "SAVE PATCH;");
    run(&mut session, &format!("BEGIN PATCH \"{}\";", lql_path(&b)));
    run(&mut session, SESSION_EDIT);
    run(&mut session, "SAVE PATCH;");

    let mut want = bound(&fx.vindex);
    run(&mut want, &format!("APPLY PATCH \"{}\";", lql_path(&b)));

    run(&mut session, &format!("USE \"{}\";", lql_path(&fx.vindex)));
    run(&mut session, &format!("APPLY PATCH \"{}\";", lql_path(&a)));
    run(&mut session, &format!("APPLY PATCH \"{}\";", lql_path(&b)));
    run(&mut session, &format!("REMOVE PATCH \"{}\";", lql_path(&a)));
    // SHOW PATCHES names the files, so compare everything but the list.
    let strip = |s: String| {
        s.split("\n--\n")
            .filter(|p| !p.contains(".vlp"))
            .collect::<Vec<_>>()
            .join("\n--\n")
    };
    assert_eq!(
        strip(observe_with(&mut session, fx.prompt, false)),
        strip(observe_with(&mut want, fx.prompt, false))
    );
}
