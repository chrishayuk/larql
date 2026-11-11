//! `UPDATE EDGES SET` applies every assignment or refuses the statement
//! (Gate A7).
//!
//! What used to happen instead: an assignment to an unknown field, or with
//! a value of the wrong type, was dropped, and the statement still
//! answered `Updated N features`. Beside a valid assignment the valid one
//! was applied and the other silently was not.

use std::path::{Path, PathBuf};

use larql_lql::{parse, Session};
use larql_models::TopKEntry;
use larql_vindex::format::vindex3::fixtures::{
    encode_fixture_container, miniature_glimmer, G_VOCAB,
};
use larql_vindex::ndarray::Array2;
use larql_vindex::{
    ExtractLevel, FeatureMeta, QuantFormat, StorageDtype, VectorIndex, VindexConfig, VindexPatch,
};

const HIDDEN: usize = 4;
const VOCAB: usize = 8;
const LAYER: usize = 0;
const FEATURE: usize = 0;
const TOKEN: &str = "Paris";
const TOKEN_ID: u32 = 1;
const SCORE: f32 = 0.5;
/// A score the fixture does not carry. Exactly representable in f32.
const NEW_SCORE: f32 = 0.25;
const NEW_TARGET: &str = "Lyon";
const TOP_K: usize = 5;
const PATCH_FILE: &str = "update.vlp";
/// What every refusal says about the session.
const NOTHING_UPDATED: &str = "Nothing was updated";
const ONE_SLOT: &str = "WHERE layer = 0 AND feature = 0";

fn lql_path(path: impl AsRef<Path>) -> String {
    path.as_ref().display().to_string().replace('\\', "\\\\")
}

fn try_run(session: &mut Session, stmt: &str) -> Result<Vec<String>, String> {
    let parsed = parse(stmt).map_err(|e| format!("parse {stmt}: {e}"))?;
    session.execute(&parsed).map_err(|e| e.to_string())
}

fn run(session: &mut Session, stmt: &str) -> Vec<String> {
    try_run(session, stmt).unwrap_or_else(|e| panic!("{stmt}: {e}"))
}

/// Write a one-layer browse-level vindex with one record, at slot 0.
fn write_vindex(dir: &Path) {
    std::fs::create_dir_all(dir).expect("vindex dir");
    let mut gate = Array2::<f32>::zeros((2, HIDDEN));
    gate[[0, 0]] = 1.0;
    gate[[1, 1]] = 1.0;
    let record = FeatureMeta {
        top_token: TOKEN.to_string(),
        top_token_id: TOKEN_ID,
        c_score: SCORE,
        top_k: vec![TopKEntry {
            token: TOKEN.to_string(),
            token_id: TOKEN_ID,
            logit: SCORE,
        }],
    };
    let index = VectorIndex::new(
        vec![Some(gate)],
        vec![Some(vec![Some(record), None])],
        1,
        HIDDEN,
    );
    let mut config = VindexConfig {
        version: 2,
        model: "test/update-set".into(),
        family: "llama".into(),
        source: None,
        checksums: None,
        num_layers: 1,
        hidden_size: HIDDEN,
        intermediate_size: 2,
        vocab_size: VOCAB,
        embed_scale: 1.0,
        extract_level: ExtractLevel::Browse,
        dtype: StorageDtype::F32,
        quant: QuantFormat::None,
        layer_bands: None,
        layers: Vec::new(),
        down_top_k: TOP_K,
        has_model_weights: false,
        model_config: None,
        fp4: None,
        ffn_layout: None,
        bitnet_layout: None,
    };
    index.save_vindex(dir, &mut config).expect("save vindex");
    std::fs::write(
        dir.join("embeddings.bin"),
        vec![0u8; VOCAB * HIDDEN * std::mem::size_of::<f32>()],
    )
    .expect("embeddings");
    std::fs::write(
        dir.join("tokenizer.json"),
        format!(
            "{{\"version\":\"1.0\",\"truncation\":null,\"padding\":null,\"added_tokens\":[],\
             \"normalizer\":null,\"pre_tokenizer\":null,\"post_processor\":null,\"decoder\":null,\
             \"model\":{{\"type\":\"WordLevel\",\"vocab\":{{\"unk\":0,\"{TOKEN}\":{TOKEN_ID}}},\
             \"unk_token\":\"unk\"}}}}"
        ),
    )
    .expect("tokenizer");
}

struct Fixture {
    session: Session,
    patch: PathBuf,
    _dir: tempfile::TempDir,
}

fn fixture() -> Fixture {
    let dir = tempfile::tempdir().expect("tempdir");
    let vindex = dir.path().join("bound.vindex");
    write_vindex(&vindex);
    let mut session = Session::new();
    run(&mut session, &format!("USE \"{}\";", lql_path(&vindex)));
    Fixture {
        session,
        patch: dir.path().join(PATCH_FILE),
        _dir: dir,
    }
}

impl Fixture {
    fn record(&mut self) -> (String, f32) {
        let meta = self
            .session
            .patched_overlay_mut()
            .and_then(|patched| patched.feature_meta(LAYER, FEATURE))
            .expect("the fixture record");
        (meta.top_token, meta.c_score)
    }

    /// Run an UPDATE that must be refused, and check the refusal names
    /// `reason`, changed no record and recorded no edit.
    fn assert_refused(&mut self, set: &str, reason: &str) {
        self.assert_refused_where(set, ONE_SLOT, reason);
    }

    fn assert_refused_where(&mut self, set: &str, clause: &str, reason: &str) {
        let stmt = format!("UPDATE EDGES SET {set} {clause};");
        let err = try_run(&mut self.session, &stmt).expect_err("UPDATE must be refused");
        assert!(err.contains(reason), "expected {reason:?} in: {err}");
        assert!(err.contains(NOTHING_UPDATED), "{err}");
        assert_eq!(self.record(), (TOKEN.to_string(), SCORE));
        assert!(
            self.saved_patch().operations.is_empty(),
            "a refused UPDATE must record nothing"
        );
    }

    /// Name the recording, save it and read the file back.
    fn saved_patch(&mut self) -> VindexPatch {
        let _ = std::fs::remove_file(&self.patch);
        run(
            &mut self.session,
            &format!("BEGIN PATCH \"{}\";", lql_path(&self.patch)),
        );
        run(&mut self.session, "SAVE PATCH;");
        VindexPatch::load(&self.patch).expect("saved patch")
    }
}

// ── What a valid SET does (the controls) ────────────────────────────────

#[test]
fn a_target_and_a_score_are_both_written() {
    let mut fx = fixture();

    let out = run(
        &mut fx.session,
        &format!(
            "UPDATE EDGES SET target = \"{NEW_TARGET}\", confidence = {NEW_SCORE} {ONE_SLOT};"
        ),
    )
    .join("\n");

    assert!(out.contains("Updated 1 features"), "{out}");
    assert_eq!(fx.record(), (NEW_TARGET.to_string(), NEW_SCORE));
}

#[test]
fn the_other_spellings_and_an_integer_score_are_accepted() {
    let mut fx = fixture();

    run(
        &mut fx.session,
        &format!("UPDATE EDGES SET top_token = \"{NEW_TARGET}\", c_score = 1 {ONE_SLOT};"),
    );

    assert_eq!(fx.record(), (NEW_TARGET.to_string(), 1.0));
}

// ── Assignments that cannot be applied ──────────────────────────────────

#[test]
fn an_unknown_field_is_refused_by_name() {
    let mut fx = fixture();

    fx.assert_refused("relation = \"capital\"", "`relation`");
}

#[test]
fn an_unknown_field_beside_a_valid_one_stops_the_valid_one() {
    let mut fx = fixture();

    fx.assert_refused(&format!("confidence = {NEW_SCORE}, bogus = 1"), "`bogus`");
}

#[test]
fn a_target_that_is_not_a_string_is_refused() {
    let mut fx = fixture();

    fx.assert_refused("target = 5", "takes a string");
}

#[test]
fn a_score_that_is_not_a_number_is_refused() {
    let mut fx = fixture();

    fx.assert_refused("confidence = \"high\"", "takes a number");
}

#[test]
fn an_empty_target_is_refused() {
    let mut fx = fixture();

    fx.assert_refused("target = \"\"", "empty string");
}

#[test]
fn a_field_set_twice_is_refused_under_either_spelling() {
    let mut fx = fixture();

    fx.assert_refused(
        &format!("confidence = {NEW_SCORE}, c_score = 1"),
        "more than once",
    );
    fx.assert_refused(
        &format!("target = \"{NEW_TARGET}\", top_token = \"Nice\""),
        "more than once",
    );
}

#[test]
fn a_score_too_large_to_hold_is_refused() {
    let mut fx = fixture();
    let huge = format!("1{}.0", "0".repeat(60));

    fx.assert_refused(&format!("confidence = {huge}"), "finite");
}

#[test]
fn an_invalid_set_is_refused_even_when_no_feature_matches() {
    // The statement is wrong whatever it would have matched. It used to
    // answer "(no matching features found)".
    let mut fx = fixture();

    fx.assert_refused_where("bogus = 1", "WHERE layer = 0 AND feature = 1", "`bogus`");
}

// ── The same rule on a VINDEX3 container ────────────────────────────────

#[test]
fn a_vindex3_update_refuses_an_unknown_field() {
    let checkpoint = tempfile::tempdir().expect("checkpoint dir");
    let container = tempfile::tempdir().expect("container dir");
    encode_fixture_container(
        miniature_glimmer,
        checkpoint.path(),
        container.path(),
        "update-set-fixture",
    );
    let vocab: Vec<String> = (0..G_VOCAB).map(|i| format!("\"[{i}]\":{i}")).collect();
    std::fs::write(
        container.path().join("tokenizer.json"),
        format!(
            "{{\"version\":\"1.0\",\"truncation\":null,\"padding\":null,\"added_tokens\":[],\
             \"normalizer\":null,\"pre_tokenizer\":null,\"post_processor\":null,\"decoder\":null,\
             \"model\":{{\"type\":\"WordLevel\",\"vocab\":{{{}}},\"unk_token\":\"[0]\"}}}}",
            vocab.join(",")
        ),
    )
    .expect("tokenizer");
    let mut session = Session::new();
    run(
        &mut session,
        &format!("USE \"{}\";", lql_path(container.path())),
    );

    let err = try_run(
        &mut session,
        &format!("UPDATE EDGES SET bogus = 1 {ONE_SLOT};"),
    )
    .expect_err("an unknown field must be refused");

    assert!(
        err.contains("`bogus`") && err.contains(NOTHING_UPDATED),
        "{err}"
    );
}
