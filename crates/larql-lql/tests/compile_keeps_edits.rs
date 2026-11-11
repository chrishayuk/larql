//! A legacy `COMPILE INTO VINDEX` either produces an artifact that holds
//! every edit and every required file, or fails by name (Gate A4).
//!
//! What used to happen instead:
//!
//! - the compiled `down_meta.bin` was the source's, so a `DELETE`d edge
//!   came back, an `UPDATE` was lost and an `INSERT`ed edge had no record;
//! - a slot deleted after an `INSERT` still had its down and up vectors
//!   baked into the weights;
//! - the carried files were a fixed list without `lm_head.bin`, and a
//!   missing weight file was skipped, so the artifact loaded and ran with
//!   the wrong output head;
//! - a quantised source compiled into an artifact without its weights.

use std::path::{Path, PathBuf};

use larql_inference::test_utils::{write_synthetic_model_dir, write_synthetic_q4k_model_dir};
use larql_lql::{parse, Session};
use larql_models::TopKEntry;
use larql_vindex::format::filenames::{
    DOWN_META_BIN, DOWN_WEIGHTS_BIN, LM_HEAD_BIN, UP_WEIGHTS_BIN, WEIGHT_MANIFEST_JSON,
};
use larql_vindex::ndarray::Array2;
use larql_vindex::{
    ExtractLevel, FeatureMeta, QuantFormat, SilentLoadCallbacks, StorageDtype, VectorIndex,
    VindexConfig,
};

// ── The browse-level fixture: one layer, three slots, two edges ─────────

const HIDDEN: usize = 4;
const VOCAB: usize = 8;
/// `(top token, token id, score)` per slot; `None` is a slot with no edge.
const SLOTS: [Option<(&str, u32, f32)>; 3] =
    [Some(("Paris", 1, 0.95)), Some(("French", 2, 0.88)), None];
const LAYER: usize = 0;
const PARIS: usize = 0;
const FRENCH: usize = 1;
const FRENCH_TOKEN_ID: u32 = 2;
const FRENCH_SCORE: f32 = 0.88;
/// A score no fixture edge carries. Exactly representable in f32.
const EDITED_SCORE: f32 = 0.5;
const UNKNOWN_TOKEN: &str = "unk";
const UNKNOWN_TOKEN_ID: u32 = 0;
const SOURCE_TOP_K: usize = 5;

// ── The weights fixture ─────────────────────────────────────────────────

/// The slot a compose-mode INSERT at layer 0 installs into on the
/// synthetic model: its first free slot.
const INSTALL_LAYER: usize = 0;
const INSTALL_FEATURE: usize = 0;
const INSERT: &str = r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[1]", "capital", "[2]") AT LAYER 0 MODE COMPOSE;"#;

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

fn tokenizer_json() -> String {
    let mut vocab = vec![format!("\"{UNKNOWN_TOKEN}\":{UNKNOWN_TOKEN_ID}")];
    for (token, id, _) in SLOTS.iter().flatten() {
        vocab.push(format!("\"{token}\":{id}"));
    }
    format!(
        "{{\"version\":\"1.0\",\"truncation\":null,\"padding\":null,\"added_tokens\":[],\
         \"normalizer\":null,\"pre_tokenizer\":null,\"post_processor\":null,\"decoder\":null,\
         \"model\":{{\"type\":\"WordLevel\",\"vocab\":{{{}}},\"unk_token\":\"{UNKNOWN_TOKEN}\"}}}}",
        vocab.join(",")
    )
}

/// Write a browse-level vindex whose feature metadata is [`SLOTS`].
fn write_edge_vindex(dir: &Path) {
    let mut gate = Array2::<f32>::zeros((SLOTS.len(), HIDDEN));
    let mut metas = Vec::new();
    for (feature, slot) in SLOTS.iter().enumerate() {
        gate[[feature, feature % HIDDEN]] = 1.0;
        metas.push(slot.map(|(token, token_id, score)| FeatureMeta {
            top_token: token.to_string(),
            top_token_id: token_id,
            c_score: score,
            top_k: vec![TopKEntry {
                token: token.to_string(),
                token_id,
                logit: score,
            }],
        }));
    }
    let index = VectorIndex::new(vec![Some(gate)], vec![Some(metas)], 1, HIDDEN);
    let mut config = VindexConfig {
        version: 2,
        model: "test/compile-keeps-edits".into(),
        family: "llama".into(),
        source: None,
        checksums: None,
        num_layers: 1,
        hidden_size: HIDDEN,
        intermediate_size: SLOTS.len(),
        vocab_size: VOCAB,
        embed_scale: 1.0,
        extract_level: ExtractLevel::Browse,
        dtype: StorageDtype::F32,
        quant: QuantFormat::None,
        layer_bands: None,
        layers: Vec::new(),
        down_top_k: SOURCE_TOP_K,
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
    std::fs::write(dir.join("tokenizer.json"), tokenizer_json()).expect("tokenizer");
}

/// A source vindex, a session bound to it, and a fresh output path.
struct Fixture {
    session: Session,
    source: PathBuf,
    out: PathBuf,
    _dir: tempfile::TempDir,
}

fn fixture(write_source: impl FnOnce(&Path)) -> Fixture {
    let dir = tempfile::tempdir().expect("tempdir");
    let source = dir.path().join("source.vindex");
    std::fs::create_dir_all(&source).expect("source dir");
    write_source(&source);
    let mut session = Session::new();
    run(&mut session, &format!("USE \"{}\";", lql_path(&source)));
    Fixture {
        session,
        out: dir.path().join("compiled.vindex"),
        source,
        _dir: dir,
    }
}

fn edge_fixture() -> Fixture {
    fixture(write_edge_vindex)
}

fn weights_fixture() -> Fixture {
    fixture(|dir| write_synthetic_model_dir(dir).expect("synthetic model"))
}

impl Fixture {
    fn try_compile(&mut self) -> Result<Vec<String>, String> {
        let stmt = format!("COMPILE CURRENT INTO VINDEX \"{}\";", lql_path(&self.out));
        try_run(&mut self.session, &stmt)
    }

    fn compile(&mut self) -> Vec<String> {
        self.try_compile()
            .unwrap_or_else(|e| panic!("COMPILE INTO VINDEX: {e}"))
    }

    /// The compiled vindex as a cold process would load it.
    fn compiled(&self) -> VectorIndex {
        VectorIndex::load_vindex(&self.out, &mut SilentLoadCallbacks).expect("load compiled")
    }

    fn source_bytes(&self, name: &str) -> Vec<u8> {
        std::fs::read(self.source.join(name)).unwrap_or_else(|e| panic!("source {name}: {e}"))
    }

    fn compiled_bytes(&self, name: &str) -> Vec<u8> {
        std::fs::read(self.out.join(name)).unwrap_or_else(|e| panic!("compiled {name}: {e}"))
    }

    /// What the patched session shows for a slot.
    fn session_meta(&mut self, layer: usize, feature: usize) -> Option<FeatureMeta> {
        self.session
            .patched_overlay_mut()
            .and_then(|patched| patched.feature_meta(layer, feature))
    }

    /// A refused compile names its reason and publishes nothing.
    fn assert_refused(&mut self, reason: &str) {
        let err = self
            .try_compile()
            .expect_err("COMPILE INTO VINDEX must be refused");
        assert!(err.contains(reason), "expected {reason:?} in: {err}");
        assert!(
            !self.out.exists(),
            "a refused compile must not publish an artifact"
        );
    }
}

/// The file names the source's weight manifest lists.
fn manifest_files(source: &Path) -> Vec<String> {
    let text = std::fs::read_to_string(source.join(WEIGHT_MANIFEST_JSON)).expect("manifest");
    let entries: Vec<serde_json::Value> = serde_json::from_str(&text).expect("manifest json");
    let mut files: Vec<String> = entries
        .iter()
        .filter_map(|entry| entry["file"].as_str())
        .filter(|name| !name.is_empty())
        .map(str::to_string)
        .collect();
    files.sort();
    files.dedup();
    files
}

// ── Metadata edits ──────────────────────────────────────────────────────

#[test]
fn a_deleted_edge_is_absent_from_the_compiled_vindex() {
    let mut fx = edge_fixture();
    let source_meta = fx.source_bytes(DOWN_META_BIN);
    let out = run(
        &mut fx.session,
        &format!("DELETE FROM EDGES WHERE layer = {LAYER} AND feature = {PARIS};"),
    );
    assert!(out.join("\n").contains("Deleted 1 features"), "{out:?}");

    fx.compile();

    let compiled = fx.compiled();
    assert!(
        compiled.feature_meta(LAYER, PARIS).is_none(),
        "the deleted edge came back"
    );
    let kept = compiled
        .feature_meta(LAYER, FRENCH)
        .expect("the other edge is kept");
    assert_eq!(kept.top_token_id, FRENCH_TOKEN_ID);
    assert_eq!(kept.c_score, FRENCH_SCORE);
    assert_eq!(
        fx.source_bytes(DOWN_META_BIN),
        source_meta,
        "the compile must not write through to the source"
    );
}

#[test]
fn an_updated_score_is_in_the_compiled_vindex() {
    let mut fx = edge_fixture();
    let out = run(
        &mut fx.session,
        &format!(
            "UPDATE EDGES SET confidence = {EDITED_SCORE} WHERE layer = {LAYER} AND feature = {FRENCH};"
        ),
    );
    assert!(out.join("\n").contains("Updated 1 features"), "{out:?}");

    fx.compile();

    let updated = fx
        .compiled()
        .feature_meta(LAYER, FRENCH)
        .expect("the updated edge");
    assert_eq!(updated.c_score, EDITED_SCORE);
    assert_eq!(updated.top_token_id, FRENCH_TOKEN_ID);
}

#[test]
fn an_unedited_compile_keeps_the_source_metadata_bytes() {
    let mut fx = edge_fixture();

    fx.compile();

    assert_eq!(
        fx.compiled_bytes(DOWN_META_BIN),
        fx.source_bytes(DOWN_META_BIN)
    );
}

#[test]
fn a_target_text_the_file_cannot_hold_is_refused_by_name() {
    // UPDATE SET target changes the text and keeps the token id. The
    // metadata file stores the id only, so the edit cannot be compiled.
    let mut fx = edge_fixture();
    run(
        &mut fx.session,
        &format!(
            "UPDATE EDGES SET target = \"changed\" WHERE layer = {LAYER} AND feature = {PARIS};"
        ),
    );

    fx.assert_refused(DOWN_META_BIN);
    fx.assert_refused(&format!("L{LAYER}/F{PARIS}"));
}

// ── INSERT, and DELETE after INSERT ─────────────────────────────────────

#[test]
fn an_inserted_edge_has_a_record_in_the_compiled_vindex() {
    let mut fx = weights_fixture();
    run(&mut fx.session, INSERT);

    fx.compile();

    let session_meta = fx
        .session_meta(INSTALL_LAYER, INSTALL_FEATURE)
        .expect("the session shows the inserted edge");
    let compiled = fx
        .compiled()
        .feature_meta(INSTALL_LAYER, INSTALL_FEATURE)
        .expect("the inserted edge has no metadata record");
    assert_eq!(compiled.top_token_id, session_meta.top_token_id);
    assert_eq!(compiled.c_score, session_meta.c_score);
}

/// The down vector the compiled vindex holds for one slot, as a cold
/// process would load it.
fn down_column(dir: &Path, layer: usize, feature: usize) -> Vec<f32> {
    let weights = larql_vindex::load_model_weights(dir, &mut SilentLoadCallbacks).expect("weights");
    let key = weights.arch.ffn_down_key(layer);
    let down = weights.tensors.get(&key).expect("down tensor");
    down.column(feature).to_vec()
}

#[test]
fn an_insert_deleted_before_the_compile_contributes_nothing() {
    // In the session a tombstoned slot has an activation of zero. The
    // artifact has no tombstone, so the slot's down vector is zero: the
    // insert's vector is not baked, and neither is the source's.
    let mut fx = weights_fixture();
    run(&mut fx.session, INSERT);
    let out = run(
        &mut fx.session,
        &format!(
            "DELETE FROM EDGES WHERE layer = {INSTALL_LAYER} AND feature = {INSTALL_FEATURE};"
        ),
    );
    assert!(
        out.join("\n").contains("Deleted 1 features"),
        "the DELETE must hit the installed slot: {out:?}"
    );

    let report = fx.compile().join("\n");

    assert!(report.contains("Deleted slots zeroed: 1"), "{report}");
    let source = down_column(&fx.source, INSTALL_LAYER, INSTALL_FEATURE);
    assert!(
        source.iter().any(|v| *v != 0.0),
        "the fixture's own vector must be non-zero for this to show anything"
    );
    let compiled = down_column(&fx.out, INSTALL_LAYER, INSTALL_FEATURE);
    assert!(compiled.iter().all(|v| *v == 0.0), "{compiled:?}");
    assert_eq!(
        fx.compiled_bytes(UP_WEIGHTS_BIN),
        fx.source_bytes(UP_WEIGHTS_BIN),
        "the deleted insert's up vector was baked"
    );
    assert!(fx
        .compiled()
        .feature_meta(INSTALL_LAYER, INSTALL_FEATURE)
        .is_none());
    let gate = fx
        .compiled()
        .gate_vector(INSTALL_LAYER, INSTALL_FEATURE)
        .expect("gate row");
    assert!(gate.iter().all(|v| *v == 0.0), "{gate:?}");
}

#[test]
fn an_insert_that_is_kept_changes_the_up_weights_too() {
    // The control for the up-weights equality above.
    let mut fx = weights_fixture();
    run(&mut fx.session, INSERT);

    fx.compile();

    assert_ne!(
        fx.compiled_bytes(UP_WEIGHTS_BIN),
        fx.source_bytes(UP_WEIGHTS_BIN)
    );
}

#[test]
fn an_insert_that_is_kept_does_change_the_down_weights() {
    // The control for the test above: without the DELETE the bake must
    // still happen, or "equal to the source" would prove nothing.
    let mut fx = weights_fixture();
    run(&mut fx.session, INSERT);

    fx.compile();

    assert_ne!(
        fx.compiled_bytes(DOWN_WEIGHTS_BIN),
        fx.source_bytes(DOWN_WEIGHTS_BIN)
    );
}

// ── Required files ──────────────────────────────────────────────────────

#[test]
fn the_compiled_vindex_has_the_separate_output_head() {
    let mut fx = weights_fixture();
    let head = fx.source_bytes(LM_HEAD_BIN);

    fx.compile();

    assert_eq!(fx.compiled_bytes(LM_HEAD_BIN), head);
}

#[test]
fn the_compiled_vindex_has_every_file_the_manifest_names() {
    let mut fx = weights_fixture();
    let files = manifest_files(&fx.source);
    assert!(
        files.iter().any(|name| name == LM_HEAD_BIN),
        "the fixture must exercise a file outside the old fixed list: {files:?}"
    );

    fx.compile();

    for name in &files {
        assert!(fx.out.join(name).is_file(), "compiled vindex lacks {name}");
    }
    assert_eq!(
        fx.compiled_bytes(WEIGHT_MANIFEST_JSON),
        fx.source_bytes(WEIGHT_MANIFEST_JSON)
    );
}

#[test]
fn a_manifest_file_missing_from_the_source_is_refused_by_name() {
    let mut fx = weights_fixture();
    std::fs::remove_file(fx.source.join(LM_HEAD_BIN)).expect("remove the output head");

    fx.assert_refused(LM_HEAD_BIN);
}

#[test]
fn a_missing_down_weights_file_is_refused_by_name() {
    let mut fx = weights_fixture();
    std::fs::remove_file(fx.source.join(DOWN_WEIGHTS_BIN)).expect("remove down weights");

    fx.assert_refused(DOWN_WEIGHTS_BIN);
}

#[test]
fn a_source_that_declares_weights_without_a_manifest_is_refused_by_name() {
    let mut fx = weights_fixture();
    std::fs::remove_file(fx.source.join(WEIGHT_MANIFEST_JSON)).expect("remove the manifest");

    fx.assert_refused(WEIGHT_MANIFEST_JSON);
}

#[test]
fn a_manifest_that_names_a_path_outside_the_vindex_is_refused() {
    const OUTSIDE: &str = "../outside.bin";
    let mut fx = weights_fixture();
    let path = fx.source.join(WEIGHT_MANIFEST_JSON);
    let mut entries: Vec<serde_json::Value> =
        serde_json::from_str(&std::fs::read_to_string(&path).expect("manifest")).expect("json");
    entries[0]["file"] = serde_json::Value::String(OUTSIDE.into());
    std::fs::write(&path, serde_json::to_string(&entries).expect("json")).expect("write");

    fx.assert_refused(OUTSIDE);
}

// ── Layouts the bake cannot write ───────────────────────────────────────

#[test]
fn a_quantised_source_is_refused_by_name() {
    let mut fx = fixture(|dir| write_synthetic_q4k_model_dir(dir).expect("q4k model"));

    fx.assert_refused("quantised weights");
}

// ── Clauses the compile does not act on ─────────────────────────────────

#[test]
fn a_format_clause_the_vindex_compile_ignores_is_refused_by_name() {
    let mut fx = edge_fixture();
    let stmt = format!(
        "COMPILE CURRENT INTO VINDEX \"{}\" FORMAT GGUF;",
        lql_path(&fx.out)
    );

    let err = try_run(&mut fx.session, &stmt).expect_err("FORMAT must be refused");

    assert!(err.contains("FORMAT"), "{err}");
    assert!(!fx.out.exists());
}

// ── Files loaders open by name, with no manifest entry ──────────────────

/// A file the old fixed list did not know.
const UNLISTED_FILE: &str = "router_weights.bin";
const UNLISTED_BYTES: &[u8] = b"router";
const UNLISTED_DIR: &str = "layers";

#[test]
fn a_source_file_no_list_names_is_carried() {
    let mut fx = edge_fixture();
    std::fs::write(fx.source.join(UNLISTED_FILE), UNLISTED_BYTES).expect("write");

    fx.compile();

    assert_eq!(fx.compiled_bytes(UNLISTED_FILE), UNLISTED_BYTES);
}

#[test]
fn a_source_directory_the_compile_does_not_carry_is_refused_by_name() {
    let mut fx = edge_fixture();
    std::fs::create_dir(fx.source.join(UNLISTED_DIR)).expect("mkdir");

    fx.assert_refused(UNLISTED_DIR);
}

#[test]
fn a_hidden_source_file_is_left_behind() {
    const HIDDEN: &str = ".DS_Store";
    let mut fx = edge_fixture();
    std::fs::write(fx.source.join(HIDDEN), b"x").expect("write");

    fx.compile();

    assert!(!fx.out.join(HIDDEN).exists());
}

// ── The other layouts the bake cannot write ─────────────────────────────

/// Rewrite one field of the source's `index.json` and re-bind the session.
fn with_index_field(field: &str, value: serde_json::Value) -> Fixture {
    fixture(|dir| {
        write_edge_vindex(dir);
        let path = dir.join("index.json");
        let mut index: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).expect("index")).expect("json");
        index[field] = value;
        std::fs::write(&path, serde_json::to_string(&index).expect("json")).expect("write");
    })
}

#[test]
fn a_per_layer_ffn_source_is_refused_by_name() {
    let mut fx = with_index_field("ffn_layout", serde_json::json!("per_layer"));

    fx.assert_refused("per-layer FFN weight files");
}
