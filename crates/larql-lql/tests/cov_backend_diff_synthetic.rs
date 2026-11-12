//! Backend-binding and DIFF-lifecycle behaviour across the three kinds of
//! session a user can hold, all built from synthetic fixtures:
//!
//! - unbound (`Session::new()`),
//! - a VINDEX2 vindex (`write_synthetic_model_dir` + `USE`),
//! - a VINDEX3 container (`encode_fixture_container` + `USE`),
//! - live weights (`USE MODEL` on the `miniature_glimmer` HF checkpoint).
//!
//! Covers `executor/backend.rs` (`local_artifact`, the `require_patched`
//! live-weights refusal) and `executor/lifecycle/diff.rs` (`INTO PATCH`
//! with a `LAYER` filter, the live-weights `base_model`, `CURRENT` on an
//! unbound session or live weights, and — on two hand-built VINDEX2
//! vindexes that differ in known features — the modified / removed /
//! added rows, `LIMIT`, and the patch those differences export to).

use std::path::Path;

use larql_inference::test_utils::{synthetic_tokenizer_json, write_synthetic_model_dir};
use larql_lql::{parse, Session};
use larql_models::TopKEntry;
use larql_vindex::format::generation::ContainerGeneration;
use larql_vindex::format::vindex3::fixtures::{
    encode_fixture_container, miniature_glimmer, G_VOCAB,
};
use larql_vindex::ndarray::Array2;
use larql_vindex::{FeatureMeta, PatchOp, VectorIndex, VindexConfig, VindexPatch};

/// The message `LqlError::NoBackend` renders for an unbound session.
const NO_BACKEND_MESSAGE: &str = "No backend loaded. Run USE \"path.vindex\" first.";
/// The remedy every "needs a vindex" refusal on live weights offers.
const EXTRACT_FIRST_HINT: &str = "Extract first:";
/// The layer the filtered `DIFF ... INTO PATCH` keeps; every other layer
/// of the synthetic fixture is skipped by the filter.
const KEPT_LAYER: u32 = 0;
/// The name `encode_fixture_container` stamps into the V3 fixture.
const V3_FIXTURE_NAME: &str = "backend-fixture";
/// The patch file name the INTO PATCH tests write.
const PATCH_FILE: &str = "diff.vlp";

/// The model name `write_synthetic_model_dir` declares.
const SYNTHETIC_FIXTURE_MODEL: &str = "synthetic/tinymodel";
/// The model name both hand-built VINDEX2 vindexes declare.
const EDITED_FIXTURE_MODEL: &str = "test/backend-diff";
/// Shape of the hand-built VINDEX2 vindexes.
const EDIT_HIDDEN: usize = 4;
const EDIT_FEATURES: usize = 3;
const EDIT_LAYERS: usize = 2;
const EDIT_VOCAB: usize = 10;
const EDIT_DOWN_TOP_K: usize = 5;
/// The default `DIFF` row limit when no `LIMIT` is given.
const DEFAULT_DIFF_LIMIT: usize = 20;
/// The one-row limit that truncates the three-difference diff.
const ONE_ROW_LIMIT: usize = 1;

/// The edits between the base and edited vindexes, as `(layer, feature)`.
const MODIFIED_SLOT: (usize, usize) = (0, 0);
const REMOVED_SLOT: (usize, usize) = (0, 1);
const ADDED_SLOT: (usize, usize) = (1, 1);
/// Tokens on either side of those edits.
const BASE_MODIFIED_TOKEN: &str = "Paris";
const BASE_MODIFIED_TOKEN_ID: u32 = 100;
const BASE_MODIFIED_SCORE: f32 = 0.95;
const EDITED_MODIFIED_TOKEN: &str = "Madrid";
const EDITED_MODIFIED_TOKEN_ID: u32 = 110;
const EDITED_MODIFIED_SCORE: f32 = 0.92;
const REMOVED_TOKEN: &str = "French";
const REMOVED_TOKEN_ID: u32 = 101;
const REMOVED_SCORE: f32 = 0.88;
const ADDED_TOKEN: &str = "Rome";
const ADDED_TOKEN_ID: u32 = 211;
const ADDED_SCORE: f32 = 0.85;
/// Slots identical on both sides: `(token, id, c_score)` per layer.
const UNCHANGED_L0F2: (&str, u32, f32) = ("Europe", 102, 0.75);
const UNCHANGED_L1F0: (&str, u32, f32) = ("Berlin", 200, 0.90);
const UNCHANGED_L1F2: (&str, u32, f32) = ("Spain", 202, 0.70);
/// The status column values a DIFF row ends with.
const DIFF_STATUSES: [&str; 3] = ["modified", "removed", "added"];
/// The total number of differences between the two vindexes.
const EDIT_COUNT: usize = 3;

/// Windows temp paths contain backslashes, which the LQL lexer's escape
/// pass would consume; doubling them leaves the path untouched.
fn lql_path(path: impl AsRef<Path>) -> String {
    path.as_ref().display().to_string().replace('\\', "\\\\")
}

fn run(session: &mut Session, stmt: &str) -> Vec<String> {
    let parsed = parse(stmt).unwrap_or_else(|e| panic!("parse {stmt}: {e}"));
    session
        .execute(&parsed)
        .unwrap_or_else(|e| panic!("execute {stmt}: {e}"))
}

fn run_err(session: &mut Session, stmt: &str) -> String {
    let parsed = parse(stmt).unwrap_or_else(|e| panic!("parse {stmt}: {e}"));
    match session.execute(&parsed) {
        Ok(out) => panic!("{stmt} should fail, got {out:?}"),
        Err(e) => e.to_string(),
    }
}

/// A VINDEX2 synthetic vindex on disk.
fn v2_vindex() -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("tempdir");
    write_synthetic_model_dir(dir.path()).expect("fixture write");
    dir
}

fn meta(token: &str, token_id: u32, c_score: f32) -> Option<FeatureMeta> {
    Some(FeatureMeta {
        top_token: token.to_string(),
        top_token_id: token_id,
        c_score,
        top_k: vec![TopKEntry {
            token: token.to_string(),
            token_id,
            logit: c_score,
        }],
    })
}

fn unchanged((token, token_id, c_score): (&str, u32, f32)) -> Option<FeatureMeta> {
    meta(token, token_id, c_score)
}

/// A WordLevel tokenizer naming every token id the fixtures use, so the
/// vindex's stored token ids load back as their strings.
fn fixture_tokenizer_json() -> String {
    let vocab = [
        ("[UNK]", 0),
        (BASE_MODIFIED_TOKEN, BASE_MODIFIED_TOKEN_ID),
        (REMOVED_TOKEN, REMOVED_TOKEN_ID),
        (UNCHANGED_L0F2.0, UNCHANGED_L0F2.1),
        (UNCHANGED_L1F0.0, UNCHANGED_L1F0.1),
        (UNCHANGED_L1F2.0, UNCHANGED_L1F2.1),
        (EDITED_MODIFIED_TOKEN, EDITED_MODIFIED_TOKEN_ID),
        (ADDED_TOKEN, ADDED_TOKEN_ID),
    ]
    .iter()
    .map(|(token, id)| format!("\"{token}\":{id}"))
    .collect::<Vec<_>>()
    .join(",");
    format!(
        "{{\"version\":\"1.0\",\"truncation\":null,\"padding\":null,\"added_tokens\":[],\
         \"normalizer\":null,\"pre_tokenizer\":null,\"post_processor\":null,\"decoder\":null,\
         \"model\":{{\"type\":\"WordLevel\",\"vocab\":{{{vocab}}},\"unk_token\":\"[UNK]\"}}}}"
    )
}

/// Which side of the known edits a hand-built vindex carries.
#[derive(Clone, Copy)]
enum Side {
    Base,
    Edited,
}

/// A hand-built VINDEX2 vindex (2 layers × 3 features). The `Edited`
/// side differs from `Base` in exactly three slots: `MODIFIED_SLOT`
/// changes token, `REMOVED_SLOT` loses its metadata, `ADDED_SLOT` gains
/// metadata. Every other slot is identical.
fn edited_v2_vindex(side: Side) -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("tempdir");
    let gates = (0..EDIT_LAYERS)
        .map(|layer| {
            let mut gate = Array2::<f32>::zeros((EDIT_FEATURES, EDIT_HIDDEN));
            for feature in 0..EDIT_FEATURES {
                gate[[feature, (feature + layer) % EDIT_HIDDEN]] = 1.0;
            }
            Some(gate)
        })
        .collect();
    let (layer0, layer1) = match side {
        Side::Base => (
            vec![
                meta(
                    BASE_MODIFIED_TOKEN,
                    BASE_MODIFIED_TOKEN_ID,
                    BASE_MODIFIED_SCORE,
                ),
                meta(REMOVED_TOKEN, REMOVED_TOKEN_ID, REMOVED_SCORE),
                unchanged(UNCHANGED_L0F2),
            ],
            vec![unchanged(UNCHANGED_L1F0), None, unchanged(UNCHANGED_L1F2)],
        ),
        Side::Edited => (
            vec![
                meta(
                    EDITED_MODIFIED_TOKEN,
                    EDITED_MODIFIED_TOKEN_ID,
                    EDITED_MODIFIED_SCORE,
                ),
                None,
                unchanged(UNCHANGED_L0F2),
            ],
            vec![
                unchanged(UNCHANGED_L1F0),
                meta(ADDED_TOKEN, ADDED_TOKEN_ID, ADDED_SCORE),
                unchanged(UNCHANGED_L1F2),
            ],
        ),
    };
    let index = VectorIndex::new(
        gates,
        vec![Some(layer0), Some(layer1)],
        EDIT_LAYERS,
        EDIT_HIDDEN,
    );
    let mut config = VindexConfig {
        version: 2,
        model: EDITED_FIXTURE_MODEL.into(),
        family: "llama".into(),
        num_layers: EDIT_LAYERS,
        hidden_size: EDIT_HIDDEN,
        intermediate_size: EDIT_FEATURES,
        vocab_size: EDIT_VOCAB,
        embed_scale: 1.0,
        down_top_k: EDIT_DOWN_TOP_K,
        ..Default::default()
    };
    index
        .save_vindex(dir.path(), &mut config)
        .expect("save hand-built vindex");
    let embeddings = vec![0u8; EDIT_VOCAB * EDIT_HIDDEN * std::mem::size_of::<f32>()];
    std::fs::write(dir.path().join("embeddings.bin"), embeddings).expect("embeddings");
    std::fs::write(dir.path().join("tokenizer.json"), fixture_tokenizer_json()).expect("tokenizer");
    dir
}

/// The report row `exec_diff` renders for one slot.
fn diff_row(slot: (usize, usize), token_a: &str, token_b: &str, status: &str) -> String {
    format!(
        "L{:<7} F{:<7} {:<20} {:<20} {:>10}",
        slot.0, slot.1, token_a, token_b, status
    )
}

/// A VINDEX3 container encoded from the miniature Glimmer checkpoint.
fn v3_container() -> tempfile::TempDir {
    let checkpoint = tempfile::tempdir().expect("tempdir");
    let container = tempfile::tempdir().expect("tempdir");
    encode_fixture_container(
        miniature_glimmer,
        checkpoint.path(),
        container.path(),
        V3_FIXTURE_NAME,
    );
    container
}

/// A session bound to live weights: `USE MODEL` on an HF-shaped
/// checkpoint (config.json + safetensors + tokenizer.json).
fn weight_session() -> (Session, tempfile::TempDir, String) {
    let checkpoint = tempfile::tempdir().expect("tempdir");
    miniature_glimmer(checkpoint.path());
    std::fs::write(
        checkpoint.path().join("tokenizer.json"),
        synthetic_tokenizer_json(G_VOCAB),
    )
    .expect("tokenizer write");
    let model_id = lql_path(checkpoint.path());
    let mut session = Session::new();
    let out = run(&mut session, &format!("USE MODEL \"{model_id}\";")).join("\n");
    assert!(out.contains("live weights"), "{out}");
    (session, checkpoint, model_id)
}

// ── Session::local_artifact ─────────────────────────────────────────────

#[test]
fn local_artifact_is_none_for_an_unbound_session() {
    assert!(Session::new().local_artifact().is_none());
}

#[test]
fn local_artifact_reports_the_v2_vindex_path_and_generation() {
    let dir = v2_vindex();
    let mut session = Session::new();
    run(&mut session, &format!("USE \"{}\";", lql_path(dir.path())));
    let (path, generation) = session
        .local_artifact()
        .expect("a USE'd vindex is a local artifact");
    assert_eq!(path, dir.path());
    assert_eq!(generation, ContainerGeneration::V2);
}

#[test]
fn local_artifact_reports_the_v3_container_path_and_generation() {
    let container = v3_container();
    let mut session = Session::new();
    run(
        &mut session,
        &format!("USE \"{}\";", lql_path(container.path())),
    );
    let (path, generation) = session
        .local_artifact()
        .expect("a USE'd V3 container is a local artifact");
    assert_eq!(path, container.path());
    assert_eq!(generation, ContainerGeneration::V3);
}

#[test]
fn local_artifact_is_none_for_live_weights() {
    let (session, _checkpoint, _model_id) = weight_session();
    assert!(
        session.local_artifact().is_none(),
        "live weights are not a local vindex artifact"
    );
}

// ── require_patched: SHOW PATCHES on live weights ───────────────────────

#[test]
fn show_patches_on_live_weights_tells_the_user_to_extract() {
    let (mut session, checkpoint, model_id) = weight_session();
    let err = run_err(&mut session, "SHOW PATCHES;");
    assert!(err.contains("this operation requires a vindex"), "{err}");
    assert!(err.contains(EXTRACT_FIRST_HINT), "{err}");
    // The suggested EXTRACT names the model and a vindex derived from its
    // last path component.
    let leaf = checkpoint
        .path()
        .file_name()
        .expect("tempdir has a leaf")
        .to_string_lossy()
        .into_owned();
    assert!(
        err.contains(&format!("EXTRACT MODEL \"{model_id}\"")),
        "{err}"
    );
    assert!(err.contains(&format!("INTO \"{leaf}.vindex\"")), "{err}");
}

#[test]
fn show_patches_on_an_unbound_session_has_no_backend() {
    let err = run_err(&mut Session::new(), "SHOW PATCHES;");
    assert_eq!(err, NO_BACKEND_MESSAGE);
}

// ── DIFF ────────────────────────────────────────────────────────────────

#[test]
fn diff_current_on_an_unbound_session_has_no_backend() {
    let b = v2_vindex();
    let err = run_err(
        &mut Session::new(),
        &format!("DIFF CURRENT \"{}\";", lql_path(b.path())),
    );
    assert_eq!(err, NO_BACKEND_MESSAGE);
}

#[test]
fn diff_into_patch_with_a_layer_filter_saves_an_empty_patch_for_identical_vindexes() {
    let a = v2_vindex();
    let b = v2_vindex();
    let out_dir = tempfile::tempdir().expect("tempdir");
    let patch_path = out_dir.path().join(PATCH_FILE);
    let mut session = Session::new();
    run(&mut session, &format!("USE \"{}\";", lql_path(a.path())));

    let out = run(
        &mut session,
        &format!(
            "DIFF \"{}\" \"{}\" LAYER {KEPT_LAYER} INTO PATCH \"{}\";",
            lql_path(a.path()),
            lql_path(b.path()),
            lql_path(&patch_path),
        ),
    )
    .join("\n");

    assert!(out.contains("(no differences found)"), "{out}");
    assert!(
        out.contains("(0 ops: 0 inserts, 0 updates, 0 deletes)"),
        "{out}"
    );
    let patch = VindexPatch::load(&patch_path).expect("the patch was written");
    assert!(patch.operations.is_empty());
    let description = patch.description.expect("DIFF describes the patch");
    assert!(description.starts_with("Diff: "), "{description}");
}

#[test]
fn diff_into_patch_on_live_weights_names_side_a_model_not_the_sessions() {
    let (mut session, _checkpoint, model_id) = weight_session();
    let a = v2_vindex();
    let b = v2_vindex();
    let out_dir = tempfile::tempdir().expect("tempdir");
    let patch_path = out_dir.path().join(PATCH_FILE);

    let out = run(
        &mut session,
        &format!(
            "DIFF \"{}\" \"{}\" INTO PATCH \"{}\";",
            lql_path(a.path()),
            lql_path(b.path()),
            lql_path(&patch_path),
        ),
    )
    .join("\n");

    assert!(out.contains("Extracted:"), "{out}");
    let patch = VindexPatch::load(&patch_path).expect("the patch was written");
    // The two vindexes are what the patch is between. The live model the
    // session happens to be bound to has nothing to do with it.
    assert_ne!(patch.base_model, model_id.replace("\\\\", "\\"));
    assert_eq!(patch.base_model, SYNTHETIC_FIXTURE_MODEL);
}

#[test]
fn diff_current_on_live_weights_tells_the_user_to_extract() {
    let (mut session, _checkpoint, _model_id) = weight_session();
    let b = v2_vindex();
    let err = run_err(
        &mut session,
        &format!("DIFF CURRENT \"{}\";", lql_path(b.path())),
    );
    assert!(
        err.contains("CURRENT refers to a live model, not a vindex"),
        "{err}"
    );
    assert!(err.contains(EXTRACT_FIRST_HINT), "{err}");
}

#[test]
fn diff_reports_each_modified_removed_and_added_feature() {
    let a = edited_v2_vindex(Side::Base);
    let b = edited_v2_vindex(Side::Edited);
    let out = run(
        &mut Session::new(),
        &format!(
            "DIFF \"{}\" \"{}\";",
            lql_path(a.path()),
            lql_path(b.path())
        ),
    );
    for row in [
        diff_row(
            MODIFIED_SLOT,
            BASE_MODIFIED_TOKEN,
            EDITED_MODIFIED_TOKEN,
            "modified",
        ),
        diff_row(REMOVED_SLOT, REMOVED_TOKEN, "-", "removed"),
        diff_row(ADDED_SLOT, "-", ADDED_TOKEN, "added"),
    ] {
        assert!(out.contains(&row), "missing {row:?} in {out:#?}");
    }
    let joined = out.join("\n");
    assert!(
        joined.contains(&format!(
            "{EDIT_COUNT} differences shown (limit {DEFAULT_DIFF_LIMIT})"
        )),
        "{joined}"
    );
}

#[test]
fn diff_limit_stops_after_the_first_difference() {
    let a = edited_v2_vindex(Side::Base);
    let b = edited_v2_vindex(Side::Edited);
    let out = run(
        &mut Session::new(),
        &format!(
            "DIFF \"{}\" \"{}\" LIMIT {ONE_ROW_LIMIT};",
            lql_path(a.path()),
            lql_path(b.path())
        ),
    );
    let rows: Vec<&String> = out
        .iter()
        .filter(|l| DIFF_STATUSES.iter().any(|s| l.ends_with(s)))
        .collect();
    // Only the first difference in (layer, feature) order is printed.
    assert_eq!(
        rows,
        [&diff_row(
            MODIFIED_SLOT,
            BASE_MODIFIED_TOKEN,
            EDITED_MODIFIED_TOKEN,
            "modified"
        )],
        "{out:#?}"
    );
    let joined = out.join("\n");
    assert!(
        joined.contains(&format!(
            "{ONE_ROW_LIMIT} differences shown (limit {ONE_ROW_LIMIT})"
        )),
        "{joined}"
    );
}

#[test]
fn diff_into_patch_exports_every_difference_with_the_current_model_as_base() {
    let a = edited_v2_vindex(Side::Base);
    let b = edited_v2_vindex(Side::Edited);
    let out_dir = tempfile::tempdir().expect("tempdir");
    let patch_path = out_dir.path().join(PATCH_FILE);
    let mut session = Session::new();
    run(&mut session, &format!("USE \"{}\";", lql_path(a.path())));

    // LIMIT bounds the printed rows, never the exported patch.
    let out = run(
        &mut session,
        &format!(
            "DIFF CURRENT \"{}\" LIMIT {ONE_ROW_LIMIT} INTO PATCH \"{}\";",
            lql_path(b.path()),
            lql_path(&patch_path)
        ),
    )
    .join("\n");
    assert!(
        out.contains("(3 ops: 1 inserts, 1 updates, 1 deletes)"),
        "{out}"
    );

    let patch = VindexPatch::load(&patch_path).expect("the patch was written");
    assert_eq!(patch.base_model, EDITED_FIXTURE_MODEL);
    assert_eq!(patch.operations.len(), EDIT_COUNT);
    for op in &patch.operations {
        match op {
            PatchOp::Update {
                layer,
                feature,
                down_meta: Some(m),
                ..
            } => {
                assert_eq!((*layer, *feature), MODIFIED_SLOT);
                assert_eq!(m.top_token, EDITED_MODIFIED_TOKEN);
                assert_eq!(m.top_token_id, EDITED_MODIFIED_TOKEN_ID);
            }
            PatchOp::Delete {
                layer,
                feature,
                reason,
            } => {
                assert_eq!((*layer, *feature), REMOVED_SLOT);
                assert_eq!(reason.as_deref(), Some("removed in target"));
            }
            PatchOp::Insert {
                layer,
                feature,
                target,
                confidence,
                ..
            } => {
                assert_eq!((*layer, *feature), ADDED_SLOT);
                assert_eq!(target, ADDED_TOKEN);
                assert_eq!(*confidence, Some(ADDED_SCORE));
            }
            other => panic!("unexpected op {other:?}"),
        }
    }
}

#[test]
fn diff_into_patch_without_a_backend_names_side_a_model_and_applies() {
    let a = edited_v2_vindex(Side::Base);
    let b = edited_v2_vindex(Side::Edited);
    let out_dir = tempfile::tempdir().expect("tempdir");
    let patch_path = out_dir.path().join(PATCH_FILE);
    run(
        &mut Session::new(),
        &format!(
            "DIFF \"{}\" \"{}\" INTO PATCH \"{}\";",
            lql_path(a.path()),
            lql_path(b.path()),
            lql_path(&patch_path)
        ),
    );
    let patch = VindexPatch::load(&patch_path).expect("the patch was written");
    // The patch turns A into B, so it names A's model. It used to name the
    // session's, which with nothing bound was "unknown": a patch APPLY PATCH
    // refuses.
    assert_eq!(patch.base_model, EDITED_FIXTURE_MODEL);

    // And it is a patch: applied to A, the session shows B's records.
    let mut session = Session::new();
    run(&mut session, &format!("USE \"{}\";", lql_path(a.path())));
    run(
        &mut session,
        &format!("APPLY PATCH \"{}\";", lql_path(&patch_path)),
    );
    let against_b = run(
        &mut session,
        &format!("DIFF CURRENT \"{}\";", lql_path(b.path())),
    )
    .join("\n");
    assert!(against_b.contains("(no differences found)"), "{against_b}");
}

#[test]
fn diff_base_against_current_shows_the_sessions_edits() {
    // CURRENT is the session, with its patch overlay. It used to be read
    // from the bound path on disk, so this DIFF reported no differences
    // whatever the session had edited.
    let base = edited_v2_vindex(Side::Base);
    let mut session = Session::new();
    run(&mut session, &format!("USE \"{}\";", lql_path(base.path())));
    let (removed_layer, removed_feature) = REMOVED_SLOT;
    let deleted = run(
        &mut session,
        &format!(
            "DELETE FROM EDGES WHERE layer = {removed_layer} AND feature = {removed_feature};"
        ),
    )
    .join("\n");
    assert!(deleted.contains("Deleted 1 features"), "{deleted}");

    let out = run(
        &mut session,
        &format!("DIFF \"{}\" CURRENT;", lql_path(base.path())),
    )
    .join("\n");

    assert!(!out.contains("(no differences found)"), "{out}");
    assert!(out.contains("removed"), "{out}");
    assert!(out.contains(REMOVED_TOKEN), "{out}");
}

#[test]
fn diff_into_patch_across_two_models_is_refused_and_writes_nothing() {
    let a = v2_vindex();
    let b = edited_v2_vindex(Side::Edited);
    let out_dir = tempfile::tempdir().expect("tempdir");
    let patch_path = out_dir.path().join(PATCH_FILE);

    let err = run_err(
        &mut Session::new(),
        &format!(
            "DIFF \"{}\" \"{}\" INTO PATCH \"{}\";",
            lql_path(a.path()),
            lql_path(b.path()),
            lql_path(&patch_path)
        ),
    );

    assert!(err.contains("DIFF INTO PATCH refused"), "{err}");
    assert!(err.contains(EDITED_FIXTURE_MODEL), "{err}");
    assert!(!patch_path.exists(), "a refused DIFF must write no patch");
}

#[test]
fn diff_into_patch_reports_a_save_failure() {
    let a = edited_v2_vindex(Side::Base);
    let b = edited_v2_vindex(Side::Edited);
    let out_dir = tempfile::tempdir().expect("tempdir");
    let unwritable = out_dir.path().join("missing-dir").join(PATCH_FILE);
    let err = run_err(
        &mut Session::new(),
        &format!(
            "DIFF \"{}\" \"{}\" INTO PATCH \"{}\";",
            lql_path(a.path()),
            lql_path(b.path()),
            lql_path(&unwritable)
        ),
    );
    assert!(err.contains("failed to save patch"), "{err}");
}

#[test]
fn diff_names_the_side_that_is_not_a_loadable_vindex() {
    let good = edited_v2_vindex(Side::Base);
    let not_a_vindex = tempfile::tempdir().expect("tempdir");
    let bad = not_a_vindex.path().display().to_string();

    let err_a = run_err(
        &mut Session::new(),
        &format!(
            "DIFF \"{}\" \"{}\";",
            lql_path(not_a_vindex.path()),
            lql_path(good.path())
        ),
    );
    assert!(err_a.contains(&format!("failed to load {bad}")), "{err_a}");

    let err_b = run_err(
        &mut Session::new(),
        &format!(
            "DIFF \"{}\" \"{}\";",
            lql_path(good.path()),
            lql_path(not_a_vindex.path())
        ),
    );
    assert!(err_b.contains(&format!("failed to load {bad}")), "{err_b}");
}
