use super::*;

/// Control 1: the instrument is stable — one arm, twice, identical.
#[test]
fn the_parity_instrument_is_stable_across_runs() {
    let v3 = v3_container();
    let mut a = session_for(v3.path());
    let mut b = session_for(v3.path());
    assert_eq!(
        feature_space(&mut a, DENSE_LAYERS, 64),
        feature_space(&mut b, DENSE_LAYERS, 64)
    );
    assert_eq!(walk_hits(&mut a, "[3]"), walk_hits(&mut b, "[3]"));
}

/// Control 2: the instrument detects genuinely different models —
/// the dense fixture's feature space is not the miniature's.
#[test]
fn the_parity_instrument_detects_different_models() {
    let dense = v3_container();
    let mini_checkpoint = tempfile::tempdir().unwrap();
    let mini = tempfile::tempdir().unwrap();
    encode_fixture_container(
        miniature_glimmer,
        mini_checkpoint.path(),
        mini.path(),
        "parity-mini",
    );
    std::fs::write(
        mini.path().join("tokenizer.json"),
        unambiguous_tokenizer_json(G_VOCAB),
    )
    .unwrap();

    let mut a = session_for(dense.path());
    let mut b = session_for(mini.path());
    assert_ne!(
        feature_space(&mut a, DENSE_LAYERS, 64),
        feature_space(&mut b, DENSE_LAYERS, 64),
        "instrument cannot tell different models apart"
    );
}

/// THE gate: one checkpoint, two formats, one script — the same
/// logical feature space and the same walk results.
#[test]
fn v2_and_v3_report_the_same_logical_results() {
    let v2 = v2_vindex();
    let v3 = v3_container();
    let mut v2_session = session_for(v2.path());
    let mut v3_session = session_for(v3.path());

    // Feature space: identity and annotation, exact.
    let v2_space = feature_space(&mut v2_session, DENSE_LAYERS, 300);
    let v3_space = feature_space(&mut v3_session, DENSE_LAYERS, 300);
    assert!(!v2_space.is_empty(), "V2 arm reported no features");
    assert_eq!(
        v2_space, v3_space,
        "the two formats disagree about the feature space"
    );

    // WALK: same prompt, same per-layer hit ids in the same order.
    let v2_hits = walk_hits(&mut v2_session, "[3]");
    let v3_hits = walk_hits(&mut v3_session, "[3]");
    assert!(!v2_hits.is_empty(), "V2 arm walked to nothing");
    assert_eq!(v2_hits, v3_hits, "walk results diverge between formats");

    // DESCRIBE runs on both and agrees about edge presence.
    let v2_describe = run(&mut v2_session, r#"DESCRIBE "[3]";"#).join("\n");
    let v3_describe = run(&mut v3_session, r#"DESCRIBE "[3]";"#).join("\n");
    assert_eq!(
        v2_describe.contains("(no edges found)"),
        v3_describe.contains("(no edges found)"),
        "DESCRIBE disagrees about edge presence:\nV2: {v2_describe}\nV3: {v3_describe}"
    );
}

/// The mutation half of the parity claim: the same INSERT lands on the
/// same layer and is observable through the same statements on both
/// formats — and the outcome is affirmative, not vacuously equal.
#[test]
fn v2_and_v3_agree_after_identical_knn_mutations() {
    let v2 = v2_vindex();
    let v3 = v3_container();
    let v2_outcome = knn_mutation_outcome(v2.path());
    let v3_outcome = knn_mutation_outcome(v3.path());
    assert_eq!(
        v2_outcome, v3_outcome,
        "the two formats disagree about the mutation's observable outcome"
    );
    let (_, describe_shows, override_leads) = v2_outcome;
    assert!(describe_shows, "the edit must be visible to DESCRIBE");
    assert!(override_leads, "the stored target must override INFER");
}

/// The feature-mutation half of the parity claim (3B rung 2): the
/// identical UPDATE + DELETE script leaves both formats reporting the
/// same logical feature space — and genuinely changed it (affirmative
/// control against a vacuous pass).
#[test]
fn v2_and_v3_agree_after_identical_feature_mutations() {
    let v2 = v2_vindex();
    let v3 = v3_container();
    let mut v2_session = session_for(v2.path());
    let mut v3_session = session_for(v3.path());

    let pristine = feature_space(&mut v2_session, DENSE_LAYERS, 300);

    for session in [&mut v2_session, &mut v3_session] {
        run(
            session,
            r#"UPDATE EDGES SET target = "[7]", confidence = 0.5 WHERE layer = 1 AND feature = 1;"#,
        );
        run(
            session,
            "DELETE FROM EDGES WHERE layer = 0 AND feature = 2;",
        );
        // V2's statement contract: an UPDATE on the tombstoned slot
        // matches nothing on either backend.
        let out = run(
            session,
            r#"UPDATE EDGES SET target = "[7]" WHERE layer = 0 AND feature = 2;"#,
        )
        .join("\n");
        assert!(out.contains("no matching features"), "{out}");
    }

    let v2_space = feature_space(&mut v2_session, DENSE_LAYERS, 300);
    let v3_space = feature_space(&mut v3_session, DENSE_LAYERS, 300);
    assert_eq!(
        v2_space, v3_space,
        "the two formats disagree about the mutated feature space"
    );
    assert_ne!(v2_space, pristine, "the script must have changed the space");
    assert_eq!(v2_space.get(&(1, 1)).map(String::as_str), Some("[7]"));
    assert!(
        !v2_space.contains_key(&(0, 2)),
        "the delete must be visible"
    );

    // WALK agrees about the mutated space too (the tombstone filter
    // runs inside each backend's own scan path).
    let v2_hits = walk_hits(&mut v2_session, "[3]");
    let v3_hits = walk_hits(&mut v3_session, "[3]");
    assert_eq!(v2_hits, v3_hits, "walks diverge after mutation");
}

/// MERGE of one V2 source lands identically on a V2 target and a V3
/// target: same merged/skipped counts, same resulting feature space
/// (the V3 overlay then holds every slot as an override — reading
/// through it must equal V2's overlay reads).
#[test]
fn merge_of_a_v2_source_lands_on_both_backends() {
    let source = v2_vindex();
    let v2_target = v2_vindex();
    let v3_target = v3_container();

    let mut outcomes = Vec::new();
    for target in [v2_target.path(), v3_target.path()] {
        let mut session = session_for(target);
        let out = run(
            &mut session,
            &format!("MERGE \"{}\";", lql_path(source.path())),
        )
        .join("\n");
        let counts = out
            .lines()
            .find(|l| l.contains("features merged"))
            .unwrap_or_else(|| panic!("no merge report in {out}"))
            .trim()
            .to_string();
        outcomes.push((counts, feature_space(&mut session, DENSE_LAYERS, 300)));
    }

    assert_eq!(outcomes[0].0, outcomes[1].0, "merge counts diverge");
    assert!(
        outcomes[0].0.starts_with(char::is_numeric) && !outcomes[0].0.starts_with('0'),
        "the merge must have written features: {}",
        outcomes[0].0
    );
    assert_eq!(
        outcomes[0].1, outcomes[1].1,
        "post-merge feature spaces diverge"
    );

    // The conflict strategy's losing arm, on the V3 target: KEEP_TARGET
    // skips every already-present slot.
    let mut session = session_for(v3_target.path());
    run(
        &mut session,
        &format!("MERGE \"{}\";", lql_path(source.path())),
    );
    let out = run(
        &mut session,
        &format!(
            "MERGE \"{}\" ON CONFLICT KEEP_TARGET;",
            lql_path(source.path())
        ),
    )
    .join("\n");
    assert!(out.contains("0 features merged"), "{out}");
}

/// MERGE into a tokenizerless V3 binding refuses naming the missing
/// browse capability — after the source loaded, before any write.
#[test]
fn merge_refuses_on_a_tokenizerless_v3_target() {
    let source = v2_vindex();
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(
        dense_f32_model,
        checkpoint.path(),
        container.path(),
        "tokless-merge",
    );
    let mut session = session_for(container.path());
    let stmt = format!("MERGE \"{}\";", lql_path(source.path()));
    let parsed = parse(&stmt).unwrap();
    let err = session
        .execute(&parsed)
        .expect_err("no tokenizer, no browse view, no merge");
    assert!(err.to_string().contains("tokenizer.json"), "{err}");
}

/// The stacking invariant, on both backends: **overlay operations are
/// logical facts; replay order determines visible state**. Two patches
/// write the same slot — the later application wins; removing one
/// replays the remainder; applying them in the opposite order flips
/// the outcome. All four states must agree across formats.
#[test]
fn patch_stacking_replays_in_order_on_both_backends() {
    let patch_dir = tempfile::tempdir().unwrap();
    let patch_a = lql_path(&patch_dir.path().join("a.vlp"));
    let patch_b = lql_path(&patch_dir.path().join("b.vlp"));

    // Author the two patches once, from a V2 session (the .vlp is the
    // portable artifact; both backends must replay it identically).
    {
        let author = v2_vindex();
        let mut session = session_for(author.path());
        run(&mut session, &format!("BEGIN PATCH \"{patch_a}\";"));
        run(
            &mut session,
            r#"UPDATE EDGES SET target = "[7]" WHERE layer = 1 AND feature = 1;"#,
        );
        run(&mut session, "SAVE PATCH;");
        run(&mut session, &format!("BEGIN PATCH \"{patch_b}\";"));
        run(
            &mut session,
            r#"UPDATE EDGES SET target = "[8]" WHERE layer = 1 AND feature = 1;"#,
        );
        run(
            &mut session,
            "DELETE FROM EDGES WHERE layer = 0 AND feature = 1;",
        );
        run(&mut session, "SAVE PATCH;");
    }

    // One arm's observable state: (slot (1,1) token, slot (0,1) alive?).
    let state = |session: &mut Session| -> (Option<String>, bool) {
        let space = feature_space(session, DENSE_LAYERS, 300);
        (space.get(&(1, 1)).cloned(), space.contains_key(&(0, 1)))
    };

    let v2 = v2_vindex();
    let v3 = v3_container();
    let mut outcomes = Vec::new();
    for target in [v2.path(), v3.path()] {
        // A then B: B is the last writer; the delete stands.
        let mut session = session_for(target);
        run(&mut session, &format!("APPLY PATCH \"{patch_a}\";"));
        run(&mut session, &format!("APPLY PATCH \"{patch_b}\";"));
        let ab = state(&mut session);

        // Remove A: replaying B alone must not change what B decided.
        run(&mut session, &format!("REMOVE PATCH \"{patch_a}\";"));
        let b_only = state(&mut session);

        // Fresh session, B then A: now A is the last writer.
        let mut session = session_for(target);
        run(&mut session, &format!("APPLY PATCH \"{patch_b}\";"));
        run(&mut session, &format!("APPLY PATCH \"{patch_a}\";"));
        let ba = state(&mut session);

        outcomes.push((ab, b_only, ba));
    }

    assert_eq!(outcomes[0], outcomes[1], "stacking semantics diverge");
    let (ab, b_only, ba) = outcomes[0].clone();
    assert_eq!(ab.0.as_deref(), Some("[8]"), "last writer wins: {ab:?}");
    assert!(!ab.1, "the delete stands under A+B");
    assert_eq!(b_only.0.as_deref(), Some("[8]"), "B alone keeps B's write");
    assert!(!b_only.1);
    assert_eq!(
        ba.0.as_deref(),
        Some("[7]"),
        "reversed order flips the winner"
    );
    assert!(!ba.1, "the delete is order-independent here");
}

/// The other half of portability: a patch crosses backends because the two
/// declare ONE model, not because its slots and vectors fit. The patch here
/// is the valid one byte for byte except for the model it names, so every
/// dimension agrees, and both backends must still refuse it and stay as they
/// were. Two models of one architecture have identical shapes and unrelated
/// weights; a slot index means nothing across them.
#[test]
fn a_patch_from_another_model_is_refused_on_both_backends_though_every_dimension_fits() {
    let dir = tempfile::tempdir().unwrap();
    let operations = || {
        vec![larql_vindex::PatchOp::Delete {
            layer: 0,
            feature: 1,
            reason: None,
        }]
    };
    let ours = dir.path().join("ours.vlp");
    let theirs = dir.path().join("theirs.vlp");
    write_patch(&ours, operations());
    write_patch_for(&theirs, OTHER_MODEL, operations());

    let v2 = v2_vindex();
    let v3 = v3_container();
    for target in [v2.path(), v3.path()] {
        let mut session = session_for(target);
        let before = feature_space(&mut session, DENSE_LAYERS, 300);
        assert!(before.contains_key(&(0, 1)), "the slot the patch deletes");

        let stmt = format!("APPLY PATCH \"{}\";", lql_path(&theirs));
        let err = session
            .execute(&parse(&stmt).unwrap())
            .expect_err("a patch written against another model is refused")
            .to_string();
        assert!(err.contains(OTHER_MODEL), "{err}");
        assert_eq!(
            feature_space(&mut session, DENSE_LAYERS, 300),
            before,
            "a refused patch changed the overlay"
        );

        // The same operations under this model's name apply: the refusal was
        // the identity, and nothing about the patch's shape.
        run(
            &mut session,
            &format!("APPLY PATCH \"{}\";", lql_path(&ours)),
        );
        assert!(!feature_space(&mut session, DENSE_LAYERS, 300).contains_key(&(0, 1)));
    }
}
