//! Gates for the V3 mutation overlay: V2's patch semantics — all-or-
//! nothing apply, replay-on-remove — over the KNN subset the overlay
//! represents today.

use crate::patch::format::{encode_gate_vector, PatchOp, VindexPatch};

use super::super::KnowledgeOverlay;

fn knn_op(entity: &str, target: &str, layer: usize) -> PatchOp {
    PatchOp::InsertKnn {
        layer,
        entity: entity.into(),
        relation: "rel".into(),
        target: target.into(),
        target_id: 5,
        confidence: Some(0.8),
        key_vector_b64: encode_gate_vector(&[1.0, 0.0, 0.0, 0.0]),
    }
}

fn patch_of(description: &str, operations: Vec<PatchOp>) -> VindexPatch {
    VindexPatch {
        version: 1,
        base_model: "overlay-fixture".into(),
        base_checksum: None,
        created_at: String::new(),
        description: Some(description.into()),
        author: None,
        tags: vec![],
        operations,
    }
}

#[test]
fn apply_populates_the_store_and_records_the_patch() {
    let mut overlay = KnowledgeOverlay::new();
    overlay
        .try_apply_patch(patch_of("facts", vec![knn_op("atlantis", "[5]", 1)]))
        .unwrap();
    assert_eq!(overlay.knn_store.len(), 1);
    assert_eq!(overlay.patches.len(), 1);
    let entries = overlay.knn_store.entries_for_entity("atlantis");
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].0, 1, "install layer travels with the op");
    assert_eq!(entries[0].1.target_token, "[5]");
}

#[test]
fn delete_knn_removes_by_entity() {
    let mut overlay = KnowledgeOverlay::new();
    overlay
        .try_apply_patch(patch_of(
            "add-then-drop",
            vec![
                knn_op("atlantis", "[5]", 1),
                knn_op("lemuria", "[6]", 1),
                PatchOp::DeleteKnn {
                    entity: "atlantis".into(),
                },
            ],
        ))
        .unwrap();
    assert!(overlay.knn_store.entries_for_entity("atlantis").is_empty());
    assert_eq!(overlay.knn_store.entries_for_entity("lemuria").len(), 1);
}

/// Since the compose rung, vector-bearing ops apply into the overlay:
/// a compose Insert lands gate/up/down + meta (V2's resolution), an
/// Update's vectors land too — and a corrupt vector still refuses the
/// whole patch.
#[test]
fn vector_bearing_ops_apply_into_the_compose_overlay() {
    let mut overlay = KnowledgeOverlay::new();
    overlay
        .try_apply_patch(patch_of(
            "compose",
            vec![PatchOp::Insert {
                layer: 0,
                feature: 3,
                relation: Some("rel".into()),
                entity: "atlantis".into(),
                target: "[5]".into(),
                confidence: Some(1.0),
                gate_vector_b64: Some(encode_gate_vector(&[1.0, 0.0])),
                up_vector_b64: Some(encode_gate_vector(&[0.5, 0.5])),
                down_vector_b64: Some(encode_gate_vector(&[0.0, 1.0])),
                down_meta: None,
            }],
        ))
        .unwrap();
    assert!(overlay.has_vector_state());
    assert_eq!(overlay.gate_overrides_at(0), vec![(3, &[1.0f32, 0.0][..])]);
    let meta = overlay.resolve_feature_meta(0, 3, None).unwrap();
    assert_eq!(meta.top_token, "[5]", "insert synthesises the meta");

    // An Update's vectors replace the slot's state.
    overlay
        .try_apply_patch(patch_of(
            "retune",
            vec![PatchOp::Update {
                layer: 0,
                feature: 3,
                gate_vector_b64: Some(encode_gate_vector(&[2.0, 0.0])),
                up_vector_b64: None,
                down_vector_b64: None,
                down_meta: None,
            }],
        ))
        .unwrap();
    assert_eq!(overlay.gate_overrides_at(0), vec![(3, &[2.0f32, 0.0][..])]);

    // Corrupt vectors still refuse the whole patch.
    let before_patches = overlay.patches.len();
    let err = overlay
        .try_apply_patch(patch_of(
            "corrupt-compose",
            vec![PatchOp::Insert {
                layer: 1,
                feature: 0,
                relation: None,
                entity: "mu".into(),
                target: "[6]".into(),
                confidence: None,
                gate_vector_b64: Some("!!!".into()),
                up_vector_b64: None,
                down_vector_b64: None,
                down_meta: None,
            }],
        ))
        .expect_err("corrupt vectors must refuse");
    assert!(err.to_string().contains("gate_vector_b64"), "{err}");
    assert_eq!(overlay.patches.len(), before_patches);

    // remove_patch clears vector state on rebuild.
    overlay.remove_patch(0);
    overlay.remove_patch(0);
    assert!(!overlay.has_vector_state());
}

#[test]
fn a_corrupt_key_vector_refuses_the_whole_patch() {
    let mut overlay = KnowledgeOverlay::new();
    let corrupt = VindexPatch {
        operations: vec![PatchOp::InsertKnn {
            layer: 0,
            entity: "atlantis".into(),
            relation: "rel".into(),
            target: "[5]".into(),
            target_id: 5,
            confidence: None,
            key_vector_b64: "!!!not-base64!!!".into(),
        }],
        ..patch_of("corrupt", vec![])
    };
    let err = overlay
        .try_apply_patch(corrupt)
        .expect_err("corrupt vectors must refuse");
    assert!(err.to_string().contains("key_vector_b64"), "{err}");
    assert!(overlay.knn_store.is_empty());
    assert!(overlay.patches.is_empty());
}

/// Removal rebuilds by replaying the remaining list — V2's contract,
/// session-added entries outside any patch included in the reset.
#[test]
fn remove_patch_replays_the_remaining_list() {
    let mut overlay = KnowledgeOverlay::new();
    overlay
        .try_apply_patch(patch_of("first", vec![knn_op("atlantis", "[5]", 1)]))
        .unwrap();
    overlay
        .try_apply_patch(patch_of("second", vec![knn_op("lemuria", "[6]", 1)]))
        .unwrap();
    // A session entry added outside any patch is lost on rebuild —
    // exactly as `PatchedVindex::rebuild_overrides` loses anonymous
    // session overrides.
    overlay.knn_store.add(
        0,
        vec![0.0, 1.0, 0.0, 0.0],
        7,
        "[7]".into(),
        "mu".into(),
        "rel".into(),
        1.0,
    );

    overlay.remove_patch(0);
    assert_eq!(overlay.patches.len(), 1);
    assert_eq!(overlay.patches[0].description.as_deref(), Some("second"));
    assert!(overlay.knn_store.entries_for_entity("atlantis").is_empty());
    assert_eq!(overlay.knn_store.entries_for_entity("lemuria").len(), 1);
    assert!(
        overlay.knn_store.entries_for_entity("mu").is_empty(),
        "session entries outside a patch reset on rebuild, as on V2"
    );
}

#[test]
fn remove_patch_out_of_range_is_a_no_op() {
    let mut overlay = KnowledgeOverlay::new();
    overlay
        .try_apply_patch(patch_of("only", vec![knn_op("atlantis", "[5]", 1)]))
        .unwrap();
    overlay.remove_patch(9);
    assert_eq!(overlay.patches.len(), 1);
    assert_eq!(overlay.knn_store.len(), 1);
}

fn meta(token: &str, c_score: f32) -> crate::index::types::FeatureMeta {
    crate::index::types::FeatureMeta {
        top_token: token.into(),
        top_token_id: 9,
        c_score,
        top_k: vec![],
    }
}

/// The V2 read rule at the source: an override wins, a tombstone hides,
/// otherwise the base answers — and UPDATE after DELETE resurrects.
#[test]
fn tombstone_and_resurrection_follow_the_v2_contract() {
    let mut overlay = KnowledgeOverlay::new();
    let base = Some(meta("[3]", 1.0));

    // Untouched slot: base passes through.
    assert_eq!(
        overlay
            .resolve_feature_meta(0, 0, base.clone())
            .map(|m| m.top_token),
        Some("[3]".to_string()),
        "untouched slots read the base"
    );
    assert!(!overlay.has_feature_state());

    // DELETE: the slot reads absent even though the base has it.
    overlay.delete_feature(0, 0);
    assert!(overlay.resolve_feature_meta(0, 0, base.clone()).is_none());
    assert!(overlay.is_tombstoned(0, 0));
    assert_eq!(overlay.tombstones_at(0), 1);
    assert_eq!(overlay.tombstones_at(1), 0);
    assert!(overlay.has_feature_state());

    // UPDATE: resurrects with the new annotation.
    overlay.update_feature_meta(0, 0, meta("[9]", 0.5));
    let resolved = overlay.resolve_feature_meta(0, 0, base).unwrap();
    assert_eq!(resolved.top_token, "[9]");
    assert!(!overlay.is_tombstoned(0, 0));
}

/// The layer-vector merge used by `feature_metas`-shaped reads.
#[test]
fn apply_meta_overrides_edits_the_layer_in_place() {
    let mut overlay = KnowledgeOverlay::new();
    overlay.delete_feature(0, 0);
    overlay.update_feature_meta(0, 2, meta("[9]", 0.5));

    let mut metas = vec![Some(meta("[1]", 1.0)), Some(meta("[2]", 1.0)), None];
    overlay.apply_meta_overrides(0, &mut metas);
    assert!(metas[0].is_none(), "tombstoned slot hidden");
    assert_eq!(
        metas[1].as_ref().map(|m| m.top_token.as_str()),
        Some("[2]"),
        "untouched slot intact"
    );
    assert_eq!(
        metas[2].as_ref().map(|m| m.top_token.as_str()),
        Some("[9]"),
        "override lands even where the base had nothing"
    );

    // A different layer is untouched.
    let mut other = vec![Some(meta("[1]", 1.0))];
    overlay.apply_meta_overrides(1, &mut other);
    assert!(other[0].is_some());
}

/// Vector-free Delete/Update patch ops replay with V2's resolution:
/// the Update's meta is constructed exactly as `overlay_apply` does
/// (single-entry top_k), and an Update WITHOUT meta after a Delete
/// drops the pinned `None` so reads fall through to the base.
#[test]
fn feature_patch_ops_replay_with_v2_resolution() {
    let mut overlay = KnowledgeOverlay::new();
    overlay
        .try_apply_patch(patch_of(
            "edits",
            vec![
                PatchOp::Delete {
                    layer: 0,
                    feature: 0,
                    reason: None,
                },
                PatchOp::Update {
                    layer: 1,
                    feature: 1,
                    gate_vector_b64: None,
                    up_vector_b64: None,
                    down_vector_b64: None,
                    down_meta: Some(crate::patch::format::PatchDownMeta {
                        top_token: "[9]".into(),
                        top_token_id: 9,
                        c_score: 0.5,
                    }),
                },
            ],
        ))
        .unwrap();
    assert!(overlay
        .resolve_feature_meta(0, 0, Some(meta("[3]", 1.0)))
        .is_none());
    let updated = overlay.resolve_feature_meta(1, 1, None).unwrap();
    assert_eq!(updated.top_token, "[9]");
    assert_eq!(updated.top_k.len(), 1, "V2 builds a single-entry top_k");
    assert_eq!(updated.top_k[0].token_id, 9);

    // Delete then meta-less Update: the pin drops, base answers again.
    let mut overlay = KnowledgeOverlay::new();
    overlay
        .try_apply_patch(patch_of(
            "delete-then-touch",
            vec![
                PatchOp::Delete {
                    layer: 0,
                    feature: 0,
                    reason: None,
                },
                PatchOp::Update {
                    layer: 0,
                    feature: 0,
                    gate_vector_b64: None,
                    up_vector_b64: None,
                    down_vector_b64: None,
                    down_meta: None,
                },
            ],
        ))
        .unwrap();
    let base = Some(meta("[3]", 1.0));
    assert_eq!(
        overlay
            .resolve_feature_meta(0, 0, base)
            .map(|m| m.top_token),
        Some("[3]".to_string()),
        "the meta-less Update must drop the pinned None (V2 rule)"
    );
    assert!(!overlay.is_tombstoned(0, 0));
}

/// remove_patch resets feature-slot state along with the KNN store.
#[test]
fn remove_patch_clears_feature_state_too() {
    let mut overlay = KnowledgeOverlay::new();
    overlay
        .try_apply_patch(patch_of(
            "only",
            vec![PatchOp::Delete {
                layer: 0,
                feature: 0,
                reason: None,
            }],
        ))
        .unwrap();
    overlay.update_feature_meta(1, 1, meta("[9]", 0.5));
    overlay.remove_patch(0);
    assert!(!overlay.has_feature_state(), "rebuild resets slot state");
    let base = Some(meta("[3]", 1.0));
    assert_eq!(
        overlay
            .resolve_feature_meta(0, 0, base)
            .map(|m| m.top_token),
        Some("[3]".to_string())
    );
}

/// Compose mutators + the free-slot rule + operand derivation, against
/// a real miniature plan/view.
#[test]
fn compose_state_derives_operand_edits_from_the_plan() {
    use crate::format::vindex3::fixtures::{miniature_glimmer, G_FFN, G_HIDDEN, G_VOCAB};
    use crate::format::vindex3::inspect::inspect_container;
    use crate::format::vindex3::opplan::plan_component_ops;

    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    crate::format::vindex3::fixtures::encode_fixture_container(
        miniature_glimmer,
        checkpoint.path(),
        container.path(),
        "overlay-compose",
    );
    let tok_json = super::larql_inference_free_tokenizer(G_VOCAB);
    let tokenizer = crate::tokenizers::Tokenizer::from_bytes(tok_json.as_bytes()).unwrap();
    let inspection = inspect_container(container.path(), false).unwrap();
    let outcome = plan_component_ops(&inspection, container.path(), "target").unwrap();
    let plan = outcome.plan.unwrap();
    let store = crate::format::vindex3::opplan::exec::operands::OperandStore::open(
        container.path(),
        &inspection,
    )
    .unwrap();
    let view = super::super::KnowledgeView::from_plan(&plan, &store, &tokenizer).unwrap();

    let mut overlay = KnowledgeOverlay::new();
    assert!(!overlay.has_vector_state());

    // Free-slot rule: every miniature slot is annotated, so the first
    // pick is the weakest c_score; claiming it moves the next pick;
    // a tombstoned slot is free again.
    assert_eq!(
        overlay.find_free_feature(&view, 99),
        None,
        "a layer with no features offers no slot"
    );
    let first = overlay.find_free_feature(&view, 0).expect("a slot");
    overlay.insert_feature(0, first, vec![9.0; G_HIDDEN], meta("[5]", 0.9));
    let second = overlay.find_free_feature(&view, 0).expect("another slot");
    assert_ne!(first, second, "a claimed slot is not offered again");
    overlay.delete_feature(0, second);
    assert_eq!(
        overlay.find_free_feature(&view, 0),
        Some(second),
        "a tombstoned slot is free (its pinned None is not a claim)"
    );

    // delete_feature dropped nothing of `first`'s state…
    assert!(overlay.has_vector_state());
    overlay.set_up_vector(0, first, vec![1.0; G_HIDDEN]);
    overlay.set_down_vector(0, first, vec![2.0; G_HIDDEN]);
    assert_eq!(overlay.gate_overrides_at(0).len(), 1);
    assert_eq!(overlay.gate_overrides_at(1).len(), 0);
    // …but deleting the composed slot itself drops its gate row.
    overlay.delete_feature(0, first);
    assert!(overlay.gate_overrides_at(0).is_empty());
    overlay.insert_feature(0, first, vec![9.0; G_HIDDEN], meta("[5]", 0.9));

    // Operand derivation: gate/up rows and the down column, addressed
    // by the plan's own FFN operands.
    let derived = overlay.operand_overrides(&plan).unwrap();
    let Some(crate::format::vindex3::opplan::LayerFfn::Dense(ffn)) = &plan.layers[0].ffn else {
        panic!("miniature layer 0 is dense");
    };
    assert!(derived.is_overridden(ffn.gate.as_ref().unwrap()));
    assert!(derived.is_overridden(&ffn.up));
    assert!(derived.is_overridden(&ffn.down));

    // The derived edits resolve: the effective gate row is the
    // overlay's, the down column is the overlay's.
    let source =
        crate::format::vindex3::opplan::exec::operands::OperandSource::overlaid(&store, &derived);
    let gate_ref = ffn.gate.as_ref().unwrap();
    let effective = source.load(gate_ref).unwrap();
    assert_eq!(
        &effective[first * G_HIDDEN..(first + 1) * G_HIDDEN],
        &vec![9.0; G_HIDDEN][..]
    );
    let down = source.load(&ffn.down).unwrap();
    for r in 0..G_HIDDEN {
        assert_eq!(down[r * G_FFN + first], 2.0, "down column at row {r}");
    }

    // A slot address beyond the plan fails closed.
    overlay.set_up_vector(9, 0, vec![1.0; G_HIDDEN]);
    let err = overlay.operand_overrides(&plan).unwrap_err();
    assert!(err.to_string().contains("beyond the plan"), "{err}");
}

/// The per-slot override accessors and the V2 `num_overrides` count
/// (distinct slots with meta OR vector state; the KNN store excluded).
#[test]
fn override_accessors_count_distinct_slots() {
    let mut overlay = KnowledgeOverlay::new();
    assert!(overlay.gate_override_at(0, 1).is_none());
    assert!(overlay.up_override_at(0, 1).is_none());
    assert!(overlay.down_override_at(0, 1).is_none());
    assert_eq!(overlay.num_overrides(), 0);

    overlay.set_gate_vector(0, 1, vec![1.0, 2.0]);
    overlay.set_up_vector(0, 1, vec![3.0, 4.0]);
    overlay.set_down_vector(0, 2, vec![5.0, 6.0]);
    overlay.update_feature_meta(1, 0, meta("[9]", 0.5));

    assert_eq!(overlay.gate_override_at(0, 1), Some(&[1.0f32, 2.0][..]));
    assert_eq!(overlay.up_override_at(0, 1), Some(&[3.0f32, 4.0][..]));
    assert_eq!(overlay.down_override_at(0, 2), Some(&[5.0f32, 6.0][..]));
    assert!(overlay.down_override_at(0, 1).is_none());
    assert_eq!(
        overlay.num_overrides(),
        3,
        "(0,1) counts once across gate+up; (0,2) and (1,0) once each"
    );
}

/// COMPILE's refusal list: tombstones and meta-only relabels have no
/// physical form in a baked container (annotations are derived), so
/// they block; a vector-carrying slot does not.
#[test]
fn bake_blockers_name_exactly_the_unbakeable_state() {
    let mut overlay = KnowledgeOverlay::new();
    assert!(overlay.bake_blockers().is_empty());

    overlay.delete_feature(0, 3);
    overlay.update_feature_meta(0, 1, meta("[9]", 0.5));
    overlay.insert_feature(0, 2, vec![1.0, 0.0], meta("[5]", 0.9));

    assert_eq!(
        overlay.bake_blockers(),
        vec![
            "meta-only override at (0,1)".to_string(),
            "tombstone at (0,3)".to_string(),
        ],
        "vector-carrying (0,2) bakes; the other two cannot"
    );
}

/// Update ops carrying up/down vectors and Insert ops carrying a
/// `down_meta` replay with V2's resolution (vectors land in the
/// overlay; a carried meta becomes the override verbatim).
#[test]
fn update_and_insert_replay_their_vector_and_meta_payloads() {
    let mut overlay = KnowledgeOverlay::new();
    overlay
        .try_apply_patch(patch_of(
            "payloads",
            vec![
                PatchOp::Update {
                    layer: 0,
                    feature: 0,
                    gate_vector_b64: None,
                    up_vector_b64: Some(encode_gate_vector(&[0.5, 0.5])),
                    down_vector_b64: Some(encode_gate_vector(&[0.0, 1.0])),
                    down_meta: None,
                },
                PatchOp::Insert {
                    layer: 0,
                    feature: 1,
                    relation: Some("rel".into()),
                    entity: "mu".into(),
                    target: "[7]".into(),
                    confidence: Some(0.4),
                    gate_vector_b64: Some(encode_gate_vector(&[2.0, 0.0])),
                    up_vector_b64: None,
                    down_vector_b64: None,
                    down_meta: Some(crate::patch::format::PatchDownMeta {
                        top_token: "[8]".into(),
                        top_token_id: 8,
                        c_score: 0.7,
                    }),
                },
            ],
        ))
        .unwrap();

    assert_eq!(overlay.up_override_at(0, 0), Some(&[0.5f32, 0.5][..]));
    assert_eq!(overlay.down_override_at(0, 0), Some(&[0.0f32, 1.0][..]));

    let inserted = overlay.resolve_feature_meta(0, 1, None).unwrap();
    assert_eq!(
        inserted.top_token, "[8]",
        "a carried down_meta wins over the synthesised target meta"
    );
    assert_eq!(inserted.top_k.len(), 1, "V2 builds a single-entry top_k");
    assert_eq!(inserted.top_k[0].token_id, 8);
    assert_eq!(overlay.gate_override_at(0, 1), Some(&[2.0f32, 0.0][..]));
}
