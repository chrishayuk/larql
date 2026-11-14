//! A Vindexfile `DELETE` removes the fact it names, and nothing else.
//!
//! The directive carries an entity, a relation and a target. The build
//! used to search the base vindex for the target alone, as a
//! case-insensitive substring of any feature's top token or top-k tokens,
//! in every layer: `DELETE ("France", "capital", "Paris")` deleted every
//! feature that so much as contained "paris". And because it searched the
//! base only, it never found a fact an `INSERT` in the same file had just
//! installed.

use std::path::Path;

use larql_models::TopKEntry;
use larql_vindex::ndarray::Array2;
use larql_vindex::{
    build_from_vindexfile, FeatureMeta, PatchOp, VectorIndex, VindexConfig, VindexPatch,
    Vindexfile, VindexfileDirective,
};

const LAYERS: usize = 2;
const HIDDEN: usize = 4;
const VOCAB: usize = 8;
const BASE_DIR: &str = "base.vindex";
const MODEL: &str = "test/vindexfile-delete";
/// The layer a Vindexfile INSERT lands in: the middle of the stack.
const INSERT_LAYER: usize = LAYERS / 2;
/// One slot's `(token, token id)`, or `None` for a free slot.
type Slot = Option<(&'static str, u32)>;
/// Every layer holds the same records: two that contain "paris", one that
/// does not, and a free slot for an INSERT to claim.
const SLOTS: [Slot; 4] = [
    Some(("Paris", 1)),
    Some(("Parisian", 2)),
    Some(("Berlin", 3)),
    None,
];
const FREE: usize = 3;
const TOKENS: [&str; 6] = ["unk", "Paris", "Parisian", "Berlin", "London", "Lyon"];
const PATCH_FILE: &str = "facts.vlp";

fn tokenizer_json() -> String {
    let vocab: Vec<String> = TOKENS
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

fn write_base(dir: &Path) {
    std::fs::create_dir_all(dir).expect("base dir");
    let gates = (0..LAYERS)
        .map(|_| Some(Array2::<f32>::from_elem((SLOTS.len(), HIDDEN), 0.25)))
        .collect();
    let metas = (0..LAYERS)
        .map(|_| {
            Some(
                SLOTS
                    .iter()
                    .map(|slot| {
                        slot.map(|(token, token_id)| FeatureMeta {
                            top_token: token.to_string(),
                            top_token_id: token_id,
                            c_score: 0.5,
                            top_k: vec![TopKEntry {
                                token: token.to_string(),
                                token_id,
                                logit: 0.5,
                            }],
                        })
                    })
                    .collect(),
            )
        })
        .collect();
    let index = VectorIndex::new(gates, metas, LAYERS, HIDDEN);
    let layer_infos = index.save_gate_vectors(dir).expect("gates");
    index.save_down_meta(dir).expect("down meta");
    std::fs::write(dir.join("tokenizer.json"), tokenizer_json()).expect("tokenizer");
    let config = VindexConfig {
        version: 2,
        model: MODEL.into(),
        family: "synthetic".into(),
        num_layers: LAYERS,
        hidden_size: HIDDEN,
        intermediate_size: SLOTS.len(),
        vocab_size: VOCAB,
        embed_scale: 1.0,
        layers: layer_infos,
        down_top_k: 1,
        ..Default::default()
    };
    VectorIndex::save_config(&config, dir).expect("config");
}

fn insert(entity: &str, relation: &str, target: &str) -> VindexfileDirective {
    VindexfileDirective::Insert {
        entity: entity.into(),
        relation: relation.into(),
        target: target.into(),
    }
}

fn delete(entity: &str, relation: &str, target: &str) -> VindexfileDirective {
    VindexfileDirective::Delete {
        entity: entity.into(),
        relation: relation.into(),
        target: target.into(),
    }
}

/// Build `FROM base` followed by `directives`. Returns the top token of
/// every slot, layer by layer, or the build's error.
fn build(directives: Vec<VindexfileDirective>) -> Result<Vec<Vec<Option<String>>>, String> {
    build_in(&tempfile::tempdir().expect("tempdir"), directives)
}

fn build_in(
    dir: &tempfile::TempDir,
    mut directives: Vec<VindexfileDirective>,
) -> Result<Vec<Vec<Option<String>>>, String> {
    write_base(&dir.path().join(BASE_DIR));
    directives.insert(0, VindexfileDirective::From(format!("./{BASE_DIR}")));
    let vf = Vindexfile {
        directives,
        stages: vec![],
    };
    let built = build_from_vindexfile(&vf, None, dir.path()).map_err(|e| e.to_string())?;
    Ok((0..LAYERS)
        .map(|layer| {
            (0..SLOTS.len())
                .map(|feature| {
                    built
                        .index
                        .feature_meta(layer, feature)
                        .map(|m| m.top_token)
                })
                .collect()
        })
        .collect())
}

/// The base's records, as [`build`] reports them.
fn base_tokens() -> Vec<Vec<Option<String>>> {
    (0..LAYERS)
        .map(|_| {
            SLOTS
                .iter()
                .map(|s| s.map(|(t, _)| t.to_string()))
                .collect()
        })
        .collect()
}

#[test]
fn the_base_fixture_loads_with_its_tokens() {
    // The control: the records the tests below must not lose are there.
    assert_eq!(build(vec![]).expect("build"), base_tokens());
}

#[test]
fn a_delete_does_not_remove_base_features_that_share_the_target() {
    let err = build(vec![delete("France", "capital", "Paris")])
        .expect_err("no directive installed this fact");

    assert!(err.contains("France") && err.contains("capital"), "{err}");
    assert!(err.contains("PATCH"), "the refusal says what to use: {err}");
}

#[test]
fn a_delete_removes_the_fact_an_insert_in_the_same_file_installed() {
    let tokens = build(vec![
        insert("Acme", "headquarters", "London"),
        delete("Acme", "headquarters", "London"),
    ])
    .expect("build");

    assert_eq!(
        tokens,
        base_tokens(),
        "the insert is gone and the base is whole"
    );
}

#[test]
fn an_insert_that_is_not_deleted_is_in_the_build() {
    // The control for the test above.
    let tokens = build(vec![insert("Acme", "headquarters", "London")]).expect("build");

    assert_eq!(tokens[INSERT_LAYER][FREE].as_deref(), Some("London"));
}

#[test]
fn a_delete_with_another_entity_or_relation_matches_nothing() {
    for (entity, relation) in [("Acme", "ceo"), ("Other", "headquarters")] {
        let err = build(vec![
            insert("Acme", "headquarters", "London"),
            delete(entity, relation, "London"),
        ])
        .expect_err("same target, different fact");

        assert!(err.contains(entity) && err.contains(relation), "{err}");
    }
}

/// How many slots of the build hold `token`.
fn count(tokens: &[Vec<Option<String>>], token: &str) -> usize {
    tokens
        .iter()
        .flatten()
        .filter(|slot| slot.as_deref() == Some(token))
        .count()
}

#[test]
fn a_delete_removes_one_of_two_inserted_facts_with_the_same_target() {
    let both = vec![
        insert("Acme", "headquarters", "London"),
        insert("Globex", "headquarters", "London"),
    ];
    let mut one_deleted = both.clone();
    one_deleted.push(delete("Acme", "headquarters", "London"));

    let kept = build(both).expect("two inserts");
    let after = build(one_deleted).expect("two inserts and a delete");

    assert_eq!(count(&kept, "London"), 2);
    assert_eq!(
        count(&after, "London"),
        1,
        "the other fact with that target stays"
    );
}

#[test]
fn a_delete_removes_a_fact_a_patch_installed_and_not_one_the_patch_removed() {
    let dir = tempfile::tempdir().expect("tempdir");
    let installed = |entity: &str, feature: usize| PatchOp::Insert {
        layer: 0,
        feature,
        relation: Some("capital".into()),
        entity: entity.into(),
        target: "Lyon".into(),
        confidence: Some(0.5),
        gate_vector_b64: None,
        up_vector_b64: None,
        down_vector_b64: None,
        down_meta: None,
    };
    let patch = VindexPatch {
        version: 1,
        base_model: MODEL.into(),
        base_checksum: None,
        created_at: "2026-11-12".into(),
        description: None,
        author: None,
        tags: Vec::new(),
        operations: vec![
            installed("France", FREE),
            // Installed and then removed by the same patch: gone already.
            installed("Spain", 0),
            PatchOp::Delete {
                layer: 0,
                feature: 0,
                reason: None,
            },
        ],
    };
    patch
        .save(&dir.path().join(PATCH_FILE))
        .expect("save patch");
    let patch_directive = || VindexfileDirective::Patch(format!("./{PATCH_FILE}"));

    let tokens = build_in(
        &dir,
        vec![patch_directive(), delete("France", "capital", "Lyon")],
    )
    .expect("the patch installed this fact");
    assert_eq!(tokens[0][FREE], None, "the patch's insert is deleted");

    let err = build_in(
        &dir,
        vec![patch_directive(), delete("Spain", "capital", "Lyon")],
    )
    .expect_err("the patch already removed that fact");
    assert!(err.contains("Spain"), "{err}");
}
