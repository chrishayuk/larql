//! The patch-algebra gates

use super::*;

/// insert → delete = absent; removing the delete patch resurrects the
/// inserted fact; removing the insert too returns to base. The delete
/// is a *fact* ("this entity has no KNN entries"), so removal replays
/// visibility rather than un-deleting storage.
#[test]
fn knn_delete_patch_removal_resurrects_the_inserted_fact() {
    let patch_dir = tempfile::tempdir().unwrap();
    let insert_patch = lql_path(&patch_dir.path().join("insert.vlp"));
    let delete_patch_file = patch_dir.path().join("delete.vlp");
    let delete_patch = lql_path(&delete_patch_file);

    {
        let author = v2_vindex();
        let mut session = session_for(author.path());
        run(&mut session, &format!("BEGIN PATCH \"{insert_patch}\";"));
        run(
            &mut session,
            r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[2]", "b", "[5]");"#,
        );
        run(&mut session, "SAVE PATCH;");
    }
    write_patch(
        &delete_patch_file,
        vec![larql_vindex::PatchOp::DeleteKnn {
            entity: "[2]".into(),
        }],
    );

    let v2 = v2_vindex();
    let v3 = v3_container();
    let mut outcomes = Vec::new();
    for target in [v2.path(), v3.path()] {
        let mut session = session_for(target);
        let base = knn_fact_visible(&mut session);
        run(&mut session, &format!("APPLY PATCH \"{insert_patch}\";"));
        let inserted = knn_fact_visible(&mut session);
        run(&mut session, &format!("APPLY PATCH \"{delete_patch}\";"));
        let deleted = knn_fact_visible(&mut session);
        run(&mut session, &format!("REMOVE PATCH \"{delete_patch}\";"));
        let resurrected = knn_fact_visible(&mut session);
        run(&mut session, &format!("REMOVE PATCH \"{insert_patch}\";"));
        let emptied = knn_fact_visible(&mut session);
        outcomes.push((base, inserted, deleted, resurrected, emptied));
    }

    assert_eq!(outcomes[0], outcomes[1], "KNN patch algebra diverges");
    assert_eq!(
        outcomes[0],
        (false, true, false, true, false),
        "fold(base, active patches) must drive visibility: {:?}",
        outcomes[0]
    );
}

/// The logical-fingerprint gate: after applying and then removing
/// every patch, the ENTIRE feature space equals the base exactly —
/// the affected slots are restored AND no unaffected slot was dirtied
/// along the way. This catches mutation that has the desired semantic
/// effect but leaks state elsewhere.
#[test]
fn removing_every_patch_restores_the_exact_base_space() {
    let patch_dir = tempfile::tempdir().unwrap();
    let update_patch = lql_path(&patch_dir.path().join("u.vlp"));
    let delete_patch = lql_path(&patch_dir.path().join("d.vlp"));

    {
        let author = v2_vindex();
        let mut session = session_for(author.path());
        run(&mut session, &format!("BEGIN PATCH \"{update_patch}\";"));
        run(
            &mut session,
            r#"UPDATE EDGES SET target = "[7]" WHERE layer = 1 AND feature = 1;"#,
        );
        run(&mut session, "SAVE PATCH;");
        run(&mut session, &format!("BEGIN PATCH \"{delete_patch}\";"));
        run(
            &mut session,
            "DELETE FROM EDGES WHERE layer = 0 AND feature = 1;",
        );
        run(&mut session, "SAVE PATCH;");
    }

    let v2 = v2_vindex();
    let v3 = v3_container();
    let mut restored_spaces = Vec::new();
    for target in [v2.path(), v3.path()] {
        let mut session = session_for(target);
        let base = feature_space(&mut session, DENSE_LAYERS, 300);

        run(&mut session, &format!("APPLY PATCH \"{update_patch}\";"));
        run(&mut session, &format!("APPLY PATCH \"{delete_patch}\";"));
        let mutated = feature_space(&mut session, DENSE_LAYERS, 300);
        assert_ne!(base, mutated, "affirmative control: the patches must bite");

        run(&mut session, &format!("REMOVE PATCH \"{update_patch}\";"));
        run(&mut session, &format!("REMOVE PATCH \"{delete_patch}\";"));
        let restored = feature_space(&mut session, DENSE_LAYERS, 300);
        assert_eq!(
            base, restored,
            "removal must restore the exact base space — affected slots \
             back, unaffected slots never dirtied"
        );
        restored_spaces.push(restored);
    }
    assert_eq!(
        restored_spaces[0], restored_spaces[1],
        "restored spaces diverge across formats"
    );
}

/// Patches touching DISJOINT objects commute: every application order
/// of {KNN insert, slot update, slot delete} yields the same visible
/// state, on both backends. (Same-object precedence — last applied
/// wins — is gated in `patch_stacking_replays_in_order_on_both_backends`;
/// together they pin fold-order determinism.)
#[test]
fn disjoint_patches_commute_under_every_application_order() {
    let patch_dir = tempfile::tempdir().unwrap();
    let knn = lql_path(&patch_dir.path().join("k.vlp"));
    let upd = lql_path(&patch_dir.path().join("u.vlp"));
    let del = lql_path(&patch_dir.path().join("d.vlp"));

    {
        let author = v2_vindex();
        let mut session = session_for(author.path());
        run(&mut session, &format!("BEGIN PATCH \"{knn}\";"));
        run(
            &mut session,
            r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[2]", "b", "[5]");"#,
        );
        run(&mut session, "SAVE PATCH;");
        run(&mut session, &format!("BEGIN PATCH \"{upd}\";"));
        run(
            &mut session,
            r#"UPDATE EDGES SET target = "[7]" WHERE layer = 1 AND feature = 1;"#,
        );
        run(&mut session, "SAVE PATCH;");
        run(&mut session, &format!("BEGIN PATCH \"{del}\";"));
        run(
            &mut session,
            "DELETE FROM EDGES WHERE layer = 0 AND feature = 1;",
        );
        run(&mut session, "SAVE PATCH;");
    }

    let orders: [[&String; 3]; 6] = [
        [&knn, &upd, &del],
        [&knn, &del, &upd],
        [&upd, &knn, &del],
        [&upd, &del, &knn],
        [&del, &knn, &upd],
        [&del, &upd, &knn],
    ];

    let v2 = v2_vindex();
    let v3 = v3_container();
    for target in [v2.path(), v3.path()] {
        let mut states = Vec::new();
        for order in &orders {
            let mut session = session_for(target);
            for patch in order {
                run(&mut session, &format!("APPLY PATCH \"{patch}\";"));
            }
            let space = feature_space(&mut session, DENSE_LAYERS, 300);
            let fact = knn_fact_visible(&mut session);
            states.push((fact, space));
        }
        assert!(states[0].0, "the KNN fact must be visible in every order");
        for (i, state) in states.iter().enumerate() {
            assert_eq!(
                &states[0], state,
                "order {i} diverged — disjoint patches must commute"
            );
        }
    }
}

/// The compose half of the mutation parity claim, staged so the first
/// divergence names its stage:
///
/// 1. **capture** — the two engines' residual statistic (the normed
///    FFN input, V2's walk-trace tap) agrees to cos ≥ 1 − 1e-5;
/// 2. **identity** — slots, layers, entities, targets, ids: EXACT;
/// 3. **magnitudes** — the reference norms are computed from the same
///    bytes with the same statistic, so vector norms agree within
///    0.1%. For the down column this doubles as an exact
///    balance-decision proxy: one diverged ×1.6/×0.7 step would shift
///    the norm ≥ 40% (observed agreement: 7+ digits — the two arms
///    took identical amplify/shrink/cross-fact sequences);
/// 4. **directions** — cos ≥ 1 − 1e-5 for all three vectors. The
///    refine path earns this tightness because the suppress-basis
///    RANK is stable: this gate's first run caught bitwise-duplicate
///    decoy residuals (degenerate fixture vocab) whose cross-arm
///    noise straddled Gram-Schmidt's 1e-6 near-dependency threshold,
///    flipping basis rank per arm and swinging refined directions to
///    cos ~0.98. With distinct decoy prompts (the real-model regime)
///    the only substrate delta is the stage-1-gated 1e-7 capture
///    noise through shared math.
#[test]
fn v2_and_v3_compose_installs_agree() {
    // ── Stage 1: capture parity, both install layers ──
    {
        let v2 = v2_vindex_worded();
        let v3 = v3_container_worded();
        let mut cb = larql_vindex::SilentLoadCallbacks;
        let weights = larql_vindex::load_model_weights(v2.path(), &mut cb).unwrap();
        let tokenizer = larql_vindex::load_vindex_tokenizer(v2.path()).unwrap();
        let index = larql_vindex::VectorIndex::load_vindex(v2.path(), &mut cb).unwrap();
        let ids: Vec<u32> = tokenizer
            .encode("The b of a is", true)
            .unwrap()
            .get_ids()
            .to_vec();
        let walk = larql_inference::vindex::WalkFfn::new_unlimited_with_trace(&weights, &index);
        let _ = larql_inference::predict_with_ffn(&weights, &tokenizer, &ids, 1, &walk);
        let v2_res = walk.take_residuals();

        use larql_vindex::format::vindex3::opplan::exec::production::ProductionBackend;
        let runtime = larql_inference::vindex3::Vindex3Runtime::open(
            v3.path(),
            "target",
            ProductionBackend::new(),
        )
        .unwrap();
        let mut v3_res: Vec<(usize, Vec<f32>)> = Vec::new();
        let continuation = runtime
            .select_continuation(
                &larql_kv::shipped_continuations(),
                &larql_vindex::format::vindex3::opplan::exec::kv::RowKvState::identity(),
                &larql_vindex::format::vindex3::opplan::exec::continuation_authority::ContinuationConfig::empty(),
            )
            .unwrap();
        runtime
            .execute_streaming(&ids, &continuation, &mut |ev| {
                if let larql_inference::vindex3::PlaneEvent::Layer { index, trace } = ev {
                    v3_res.push((index, trace.ffn_input.last().unwrap().clone()));
                }
                Ok(())
            })
            .unwrap();
        for (layer, r2) in &v2_res {
            let r3 = &v3_res.iter().find(|(l, _)| l == layer).unwrap().1;
            let cos = cosine(r2, r3);
            assert!(
                cos >= 1.0 - 1e-5,
                "stage 1: capture diverges at layer {layer}: cos {cos}"
            );
        }
    }

    let patch_dir = tempfile::tempdir().unwrap();
    let script = |session: &mut Session, patch: &str| {
        run(session, &format!("BEGIN PATCH \"{patch}\";"));
        run(
            session,
            r#"INSERT INTO EDGES (entity, relation, target) VALUES ("a", "b", "[5]") AT LAYER 1 MODE COMPOSE;"#,
        );
        run(
            session,
            r#"INSERT INTO EDGES (entity, relation, target) VALUES ("c", "b", "[6]") AT LAYER 1 MODE COMPOSE;"#,
        );
        run(session, "SAVE PATCH;");
    };

    let v2 = v2_vindex_worded();
    let v2_patch = lql_path(&patch_dir.path().join("v2.vlp"));
    let mut v2_session = session_for(v2.path());
    script(&mut v2_session, &v2_patch);

    let v3 = v3_container_worded();
    let v3_patch = lql_path(&patch_dir.path().join("v3.vlp"));
    let mut v3_session = session_for(v3.path());
    script(&mut v3_session, &v3_patch);

    let load = |p: &str| larql_vindex::VindexPatch::load(std::path::Path::new(p)).unwrap();
    let (p2, p3) = (load(&v2_patch), load(&v3_patch));
    assert_eq!(p2.operations.len(), p3.operations.len(), "op counts");

    for (a, b) in p2.operations.iter().zip(&p3.operations) {
        let (l2, f2, e2, m2, [g2, u2, d2]) = op_parts(a);
        let (l3, f3, e3, m3, [g3, u3, d3]) = op_parts(b);
        assert_eq!((l2, f2, &e2), (l3, f3, &e3), "slot identity");
        assert_eq!(m2, m3, "target id");
        for (name, min_cos, x, y) in [
            ("gate", 1.0f32 - 1e-5, g2, g3),
            ("up", 1.0 - 1e-5, u2, u3),
            ("down", 1.0 - 1e-5, d2, d3),
        ] {
            let (x, y) = (x.as_ref().unwrap(), y.as_ref().unwrap());
            let vx = larql_vindex::patch::core::decode_gate_vector(x).unwrap();
            let vy = larql_vindex::patch::core::decode_gate_vector(y).unwrap();
            let cos = cosine(&vx, &vy);
            let (nx, ny) = (
                vx.iter().map(|v| v * v).sum::<f32>().sqrt(),
                vy.iter().map(|v| v * v).sum::<f32>().sqrt(),
            );
            assert!(
                cos >= min_cos,
                "stage 4: {name} direction diverges for {e2}: cos {cos} (norms {nx} vs {ny})"
            );
            assert!(
                (nx - ny).abs() <= 1e-3 * nx.max(ny).max(1e-12),
                "stage 3: {name} magnitude diverges for {e2}: {nx} vs {ny}"
            );
        }
    }
}

/// The slot, label, target id and gate/up/down of an install's own op, or of
/// the re-fit a later install records for an earlier one (Gate B, B3).
fn op_parts(
    op: &larql_vindex::PatchOp,
) -> (usize, usize, String, Option<u32>, [&Option<String>; 3]) {
    match op {
        larql_vindex::PatchOp::Insert {
            layer,
            feature,
            entity,
            target,
            gate_vector_b64,
            up_vector_b64,
            down_vector_b64,
            down_meta,
            ..
        } => (
            *layer,
            *feature,
            format!("insert {entity} → {target}"),
            down_meta.as_ref().map(|m| m.top_token_id),
            [gate_vector_b64, up_vector_b64, down_vector_b64],
        ),
        larql_vindex::PatchOp::Update {
            layer,
            feature,
            gate_vector_b64,
            up_vector_b64,
            down_vector_b64,
            down_meta: None,
        } => (
            *layer,
            *feature,
            format!("re-fit L{layer}F{feature}"),
            None,
            [gate_vector_b64, up_vector_b64, down_vector_b64],
        ),
        other => panic!("a compose install emitted {other:?}"),
    }
}

/// DIFF across generations refuses with direction, never a confused
/// half-comparison: realise both models in one generation first.
#[test]
fn diff_across_generations_refuses_with_direction() {
    let v2 = v2_vindex();
    let v3 = v3_container();
    let mut session = Session::new();
    let stmt = format!(
        "DIFF \"{}\" \"{}\";",
        lql_path(v2.path()),
        lql_path(v3.path())
    );
    let err = session
        .execute(&parse(&stmt).unwrap())
        .expect_err("mixed-generation diff must refuse");
    assert!(err.to_string().contains("across generations"), "{err}");
}

/// `PHYSICAL` is a VINDEX3 report — V2 sides refuse it with direction.
#[test]
fn physical_diff_refuses_on_v2_sides() {
    let v2 = v2_vindex();
    let mut session = Session::new();
    let stmt = format!(
        "DIFF \"{}\" \"{}\" PHYSICAL;",
        lql_path(v2.path()),
        lql_path(v2.path())
    );
    let err = session
        .execute(&parse(&stmt).unwrap())
        .expect_err("V2 sides must refuse PHYSICAL");
    assert!(err.to_string().contains("VINDEX3 report"), "{err}");
}

/// `COMPACT INTO VINDEX` is the V3 physical statement — a V2 binding
/// is directed to its own tiered compaction.
#[test]
fn compact_into_vindex_refuses_on_v2_with_direction() {
    let v2 = v2_vindex();
    let mut session = session_for(v2.path());
    let err = session
        .execute(&parse(r#"COMPACT INTO VINDEX "out.v3";"#).unwrap())
        .expect_err("V2 must refuse the V3 physical compact");
    assert!(err.to_string().contains("COMPACT MINOR"), "{err}");
}
