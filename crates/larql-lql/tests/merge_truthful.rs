//! `MERGE` writes what it reports, records what it writes, and refuses a
//! source it cannot merge whole (Gate A6).
//!
//! What used to happen instead:
//!
//! - `MERGE src INTO other` wrote to the bound vindex's overlay and
//!   answered `Merged src → other`: nothing was ever written to `other`;
//! - the merged records were not in the patch recording, so `SAVE PATCH`
//!   saved a patch without them;
//! - a source built from another model, or with slots the bound vindex
//!   does not have, was merged slot for slot.

use std::path::{Path, PathBuf};

use larql_lql::{parse, Session};
use larql_models::TopKEntry;
use larql_vindex::format::vindex3::fixtures::{
    encode_fixture_container, miniature_glimmer, G_VOCAB,
};
use larql_vindex::ndarray::Array2;
use larql_vindex::{
    ExtractLevel, FeatureMeta, PatchOp, QuantFormat, StorageDtype, VectorIndex, VindexConfig,
    VindexPatch,
};

const HIDDEN: usize = 4;
const VOCAB: usize = 8;
const LAYER: usize = 0;
const MODEL: &str = "test/merge";
const OTHER_MODEL: &str = "test/another-model";
const TOP_K: usize = 5;
const PATCH_FILE: &str = "merge.vlp";

/// One slot's record: `(token, token id, score)`, or `None` for no record.
type Slot = Option<(&'static str, u32, f32)>;

/// The bound vindex: two records and a free slot.
const BOUND: [Slot; 3] = [Some(("Paris", 1, 0.5)), Some(("French", 2, 0.5)), None];
/// A source over the same slots: a stronger record at slot 0, a weaker one
/// at slot 1, and a record for the bound vindex's free slot.
const SOURCE: [Slot; 3] = [
    Some(("Lyon", 3, 0.75)),
    Some(("German", 4, 0.25)),
    Some(("Europe", 5, 0.5)),
];
/// A source with one slot more than the bound vindex has.
const WIDER: [Slot; 4] = [
    Some(("Lyon", 3, 0.75)),
    None,
    None,
    Some(("Europe", 5, 0.5)),
];
const STRONGER: usize = 0;
const WEAKER: usize = 1;
const FREE: usize = 2;

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
    let vocab: Vec<String> = ["unk", "Paris", "French", "Lyon", "German", "Europe"]
        .iter()
        .enumerate()
        .map(|(id, token)| format!("\"{token}\":{id}"))
        .collect();
    format!(
        "{{\"version\":\"1.0\",\"truncation\":null,\"padding\":null,\"added_tokens\":[],\
         \"normalizer\":null,\"pre_tokenizer\":null,\"post_processor\":null,\"decoder\":null,\
         \"model\":{{\"type\":\"WordLevel\",\"vocab\":{{{}}},\"unk_token\":\"unk\"}}}}",
        vocab.join(",")
    )
}

/// Write a one-layer browse-level vindex holding `slots`, built from `model`.
fn write_vindex(dir: &Path, model: &str, slots: &[Slot]) {
    std::fs::create_dir_all(dir).expect("vindex dir");
    let mut gate = Array2::<f32>::zeros((slots.len(), HIDDEN));
    let mut metas = Vec::new();
    for (feature, slot) in slots.iter().enumerate() {
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
        model: model.into(),
        family: "llama".into(),
        source: None,
        checksums: None,
        num_layers: 1,
        hidden_size: HIDDEN,
        intermediate_size: slots.len(),
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
    std::fs::write(dir.join("tokenizer.json"), tokenizer_json()).expect("tokenizer");
}

/// A session bound to [`BOUND`], and a directory for the other vindexes.
struct Fixture {
    session: Session,
    bound: PathBuf,
    dir: tempfile::TempDir,
}

fn fixture() -> Fixture {
    let dir = tempfile::tempdir().expect("tempdir");
    let bound = dir.path().join("bound.vindex");
    write_vindex(&bound, MODEL, &BOUND);
    let mut session = Session::new();
    run(&mut session, &format!("USE \"{}\";", lql_path(&bound)));
    Fixture {
        session,
        bound,
        dir,
    }
}

impl Fixture {
    /// Write another vindex beside the bound one and return its path.
    fn vindex(&self, name: &str, model: &str, slots: &[Slot]) -> PathBuf {
        let path = self.dir.path().join(name);
        write_vindex(&path, model, slots);
        path
    }

    fn token_at(&mut self, feature: usize) -> Option<String> {
        self.session
            .patched_overlay_mut()
            .and_then(|patched| patched.feature_meta(LAYER, feature))
            .map(|meta| meta.top_token)
    }

    /// The session's records, slot by slot.
    fn tokens(&mut self) -> Vec<Option<String>> {
        (0..BOUND.len()).map(|f| self.token_at(f)).collect()
    }

    /// Whether the session has recorded no edit: naming and saving the
    /// recording writes a patch with no operation.
    fn recorded_nothing(&mut self) -> bool {
        self.saved_patch().operations.is_empty()
    }

    /// Name the recording, save it and read the file back.
    fn saved_patch(&mut self) -> VindexPatch {
        let path = self.dir.path().join(PATCH_FILE);
        run(
            &mut self.session,
            &format!("BEGIN PATCH \"{}\";", lql_path(&path)),
        );
        run(&mut self.session, "SAVE PATCH;");
        VindexPatch::load(&path).expect("saved patch")
    }
}

fn bound_tokens() -> Vec<Option<String>> {
    BOUND
        .iter()
        .map(|slot| slot.map(|(token, _, _)| token.to_string()))
        .collect()
}

#[test]
fn a_merge_writes_the_source_records_into_the_session() {
    let mut fx = fixture();
    let source = fx.vindex("source.vindex", MODEL, &SOURCE);

    let out = run(
        &mut fx.session,
        &format!("MERGE \"{}\";", lql_path(&source)),
    )
    .join("\n");

    assert!(out.contains("3 features merged, 0 skipped"), "{out}");
    assert!(out.contains(&fx.bound.display().to_string()), "{out}");
    assert_eq!(
        fx.tokens(),
        [
            Some("Lyon".into()),
            Some("German".into()),
            Some("Europe".into())
        ]
    );
}

#[test]
fn a_merge_into_another_vindex_is_refused_and_merges_nothing() {
    let mut fx = fixture();
    let source = fx.vindex("source.vindex", MODEL, &SOURCE);
    let other = fx.vindex("other.vindex", MODEL, &BOUND);

    let err = try_run(
        &mut fx.session,
        &format!(
            "MERGE \"{}\" INTO \"{}\";",
            lql_path(&source),
            lql_path(&other)
        ),
    )
    .expect_err("MERGE INTO a vindex that is not bound must be refused");

    assert!(err.contains("INTO"), "{err}");
    assert!(err.contains("Nothing was merged"), "{err}");
    assert_eq!(fx.tokens(), bound_tokens());
    assert!(fx.recorded_nothing());
}

#[test]
fn a_merge_into_the_bound_vindex_by_name_is_the_same_merge() {
    let mut fx = fixture();
    let source = fx.vindex("source.vindex", MODEL, &SOURCE);
    let bound = lql_path(&fx.bound);

    run(
        &mut fx.session,
        &format!("MERGE \"{}\" INTO \"{bound}\";", lql_path(&source)),
    );

    assert_eq!(fx.token_at(STRONGER), Some("Lyon".into()));
}

#[test]
fn a_merge_is_in_the_saved_patch() {
    let mut fx = fixture();
    let source = fx.vindex("source.vindex", MODEL, &SOURCE);
    run(
        &mut fx.session,
        &format!(
            "MERGE \"{}\" ON CONFLICT HIGHEST_CONFIDENCE;",
            lql_path(&source)
        ),
    );
    assert_eq!(
        fx.tokens(),
        [
            Some("Lyon".into()),
            Some("French".into()),
            Some("Europe".into())
        ],
        "the stronger record and the free slot are written; the weaker is not"
    );

    let patch = fx.saved_patch();

    let mut recorded: Vec<(usize, String)> = patch
        .operations
        .iter()
        .map(|op| match op {
            PatchOp::Update {
                layer,
                feature,
                down_meta: Some(meta),
                ..
            } => {
                assert_eq!(*layer, LAYER);
                (*feature, meta.top_token.clone())
            }
            other => panic!("a merge records updates, got {other:?}"),
        })
        .collect();
    recorded.sort();
    assert_eq!(
        recorded,
        [(STRONGER, "Lyon".to_string()), (FREE, "Europe".to_string())]
    );
}

#[test]
fn the_saved_patch_replays_the_merge_in_a_fresh_session() {
    let mut fx = fixture();
    let source = fx.vindex("source.vindex", MODEL, &SOURCE);
    run(
        &mut fx.session,
        &format!(
            "MERGE \"{}\" ON CONFLICT HIGHEST_CONFIDENCE;",
            lql_path(&source)
        ),
    );
    let merged = fx.tokens();
    fx.saved_patch();

    let mut fresh = Session::new();
    run(&mut fresh, &format!("USE \"{}\";", lql_path(&fx.bound)));
    run(
        &mut fresh,
        &format!(
            "APPLY PATCH \"{}\";",
            lql_path(fx.dir.path().join(PATCH_FILE))
        ),
    );
    let replayed: Vec<Option<String>> = (0..BOUND.len())
        .map(|feature| {
            fresh
                .patched_overlay_mut()
                .and_then(|patched| patched.feature_meta(LAYER, feature))
                .map(|meta| meta.top_token)
        })
        .collect();

    assert_eq!(replayed, merged);
    assert_eq!(merged[WEAKER], Some("French".into()));
}

#[test]
fn a_merge_that_writes_nothing_starts_no_recording() {
    let mut fx = fixture();
    let source = fx.vindex("source.vindex", MODEL, &BOUND[..2]);

    let out = run(
        &mut fx.session,
        &format!("MERGE \"{}\" ON CONFLICT KEEP_TARGET;", lql_path(&source)),
    )
    .join("\n");

    assert!(out.contains("0 features merged, 2 skipped"), "{out}");
    assert!(!out.contains("Auto-patch started"), "{out}");
    assert!(fx.recorded_nothing());
}

#[test]
fn a_source_built_from_another_model_is_refused_and_merges_nothing() {
    let mut fx = fixture();
    let source = fx.vindex("source.vindex", OTHER_MODEL, &SOURCE);

    let err = try_run(
        &mut fx.session,
        &format!("MERGE \"{}\";", lql_path(&source)),
    )
    .expect_err("a source from another model must be refused");

    assert!(err.contains(OTHER_MODEL) && err.contains(MODEL), "{err}");
    assert_eq!(fx.tokens(), bound_tokens());
    assert!(fx.recorded_nothing());
}

#[test]
fn a_source_that_declares_no_model_is_refused() {
    let mut fx = fixture();
    let source = fx.vindex("source.vindex", "", &SOURCE);

    let err = try_run(
        &mut fx.session,
        &format!("MERGE \"{}\";", lql_path(&source)),
    )
    .expect_err("a source with no model name must be refused");

    assert!(err.contains("without both names"), "{err}");
    assert_eq!(fx.tokens(), bound_tokens());
}

#[test]
fn a_source_whose_records_cannot_be_read_is_refused() {
    // Without a gate table the source reads as holding no records. That
    // must not be answered as a merge of nothing.
    let mut fx = fixture();
    let source = fx.vindex("source.vindex", MODEL, &SOURCE);
    std::fs::remove_file(source.join("gate_vectors.bin")).expect("remove the gate table");

    let err = try_run(
        &mut fx.session,
        &format!("MERGE \"{}\";", lql_path(&source)),
    )
    .expect_err("an unreadable source must be refused");

    assert!(
        err.contains("Nothing was merged") || err.contains("failed to load"),
        "{err}"
    );
    assert_eq!(fx.tokens(), bound_tokens());
}

#[test]
fn a_source_with_a_slot_the_bound_vindex_lacks_is_refused_whole() {
    // The record at slot 0 is in bounds. It must not be merged either.
    let mut fx = fixture();
    let source = fx.vindex("wider.vindex", MODEL, &WIDER);

    let err = try_run(
        &mut fx.session,
        &format!("MERGE \"{}\";", lql_path(&source)),
    )
    .expect_err("a source wider than the bound vindex must be refused");

    assert!(err.contains("out of bounds"), "{err}");
    assert!(err.contains(&format!("F{}", WIDER.len() - 1)), "{err}");
    assert_eq!(fx.tokens(), bound_tokens());
    assert!(fx.recorded_nothing());
}

// ── The same statement on a VINDEX3 container ───────────────────────────

/// The name the fixture container is encoded under.
const V3_MODEL: &str = "merge-fixture";
/// A feature index no fixture layer reaches.
const V3_OUT_OF_BOUNDS: usize = 1_000_000;

fn v3_session() -> (Session, tempfile::TempDir, tempfile::TempDir) {
    let checkpoint = tempfile::tempdir().expect("checkpoint dir");
    let container = tempfile::tempdir().expect("container dir");
    encode_fixture_container(
        miniature_glimmer,
        checkpoint.path(),
        container.path(),
        V3_MODEL,
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
    let sources = tempfile::tempdir().expect("sources dir");
    (session, container, sources)
}

#[test]
fn a_merge_into_a_vindex3_container_is_written_and_recorded() {
    let (mut session, _container, sources) = v3_session();
    let source = sources.path().join("source.vindex");
    write_vindex(&source, V3_MODEL, &SOURCE[..1]);

    let out = run(&mut session, &format!("MERGE \"{}\";", lql_path(&source))).join("\n");

    assert!(out.contains("1 features merged, 0 skipped"), "{out}");
    let patch_path = sources.path().join(PATCH_FILE);
    run(
        &mut session,
        &format!("BEGIN PATCH \"{}\";", lql_path(&patch_path)),
    );
    run(&mut session, "SAVE PATCH;");
    let patch = VindexPatch::load(&patch_path).expect("saved patch");
    assert_eq!(patch.operations.len(), 1, "{:?}", patch.operations);
}

#[test]
fn a_vindex3_merge_refuses_another_model_and_a_slot_it_lacks() {
    let (mut session, _container, sources) = v3_session();
    let other = sources.path().join("other.vindex");
    write_vindex(&other, OTHER_MODEL, &SOURCE[..1]);
    let mut wide: Vec<Slot> = vec![None; V3_OUT_OF_BOUNDS + 1];
    wide[V3_OUT_OF_BOUNDS] = SOURCE[0];
    let wider = sources.path().join("wider.vindex");
    write_vindex(&wider, V3_MODEL, &wide);

    let model = try_run(&mut session, &format!("MERGE \"{}\";", lql_path(&other)))
        .expect_err("another model");
    let bounds = try_run(&mut session, &format!("MERGE \"{}\";", lql_path(&wider)))
        .expect_err("a slot the container lacks");

    assert!(
        model.contains(OTHER_MODEL) && model.contains(V3_MODEL),
        "{model}"
    );
    assert!(bounds.contains("out of bounds"), "{bounds}");
    let shown = run(&mut session, "SHOW PATCHES;").join("\n");
    assert!(
        !shown.contains("Auto-patch") && !shown.contains("1 ops"),
        "{shown}"
    );
}
