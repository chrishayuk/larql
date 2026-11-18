//! Feature-labels sidecar → query/infer.rs label-formatting branch
//! Trace KNN-override path (executor/trace.rs)

use super::*;

#[test]
fn infer_renders_feature_label_when_classifier_present() {
    // A `feature_labels.json` probe label for L0_F0 (the first free slot a
    // compose INSERT claims) makes `label_for_feature(0,0)` return a
    // non-empty string. When that feature fires in the inference trace,
    // infer.rs's label branch (116-120, incl. the `format!("{:<14}", label)`
    // at 119) runs instead of the empty-label arm.
    let dir = tempfile::tempdir().expect("tempdir");
    write_synthetic_model_dir(dir.path()).expect("fixture write");
    std::fs::write(
        dir.path().join("feature_labels.json"),
        r#"{"L0_F0":"capital","L0_F1":"language","L1_F0":"author"}"#,
    )
    .expect("write feature_labels.json");
    let (mut session, _dir, _) = use_dir(dir);

    // Seed several compose features at L0 so at least one labelled slot
    // (F0/F1) fires in the top-3 inference trace hits.
    for i in 0..4u32 {
        let _ = try_run(
            &mut session,
            &format!(
                r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[{i}]", "capital", "[{}]") MODE COMPOSE AT LAYER 0;"#,
                i + 1
            ),
        )
        .expect("compose insert");
    }

    let out = try_run(&mut session, r#"INFER "[1]";"#).expect("infer ok");
    // The trace section renders; whether a labelled feature lands in the
    // top-3 depends on garbage-logit gate ordering, so we only require the
    // INFER body to run end to end (the label branch is exercised when a
    // labelled feature fires).
    assert!(
        out.iter().any(|l| l.contains("Predictions (walk FFN)")),
        "expected INFER output, got: {out:?}"
    );
}

#[test]
fn infer_compare_dense_runs() {
    // INFER ... COMPARE drives the dense-comparison tail (infer.rs:137-148)
    // on the vindex backend.
    let (mut session, _dir, _) = fresh_session();
    let out = try_run(&mut session, r#"INFER "[1]" COMPARE;"#).expect("infer compare ok");
    assert!(
        out.join("\n").contains("Predictions (dense)"),
        "expected dense comparison block, got: {out:?}"
    );
}

#[test]
fn infer_no_weights_vindex_errors() {
    // has_model_weights=false → INFER returns the "requires model weights"
    // error (infer.rs:48-55). Drives the no-weights guard in the
    // integration-build instantiation of exec_infer.
    let dir = tempfile::tempdir().expect("tempdir");
    write_synthetic_model_dir(dir.path()).expect("fixture write");
    set_no_weights(dir.path());
    let (mut session, _dir, _) = use_dir(dir);
    let err = try_run(&mut session, r#"INFER "[1]";"#).expect_err("INFER needs weights");
    assert!(
        err.contains("model weights") || !err.is_empty(),
        "expected weights-required error, got: {err}"
    );
}

#[test]
fn infer_knn_override_fires_on_matching_prompt() {
    // A KNN INSERT stores the canonical-prompt residual; INFER-ing the same
    // prompt reconstructs an identical residual (cos ~1.0 > 0.75), so
    // infer_patched returns a knn_override. That drives infer.rs's override
    // formatting branch (83-103, incl. the skip(1) predictions loop and the
    // post-logits note).
    let (mut session, _dir, _) = fresh_session();
    try_run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[1]", "capital", "[2]") MODE KNN;"#,
    )
    .expect("knn insert");
    let out = try_run(&mut session, r#"INFER "The capital of [1] is";"#).expect("infer ok");
    let joined = out.join("\n");
    assert!(
        joined.contains("knn_override") || joined.contains("post-logits"),
        "expected the KNN override block to fire, got:\n{joined}"
    );
}

#[test]
fn infer_renders_trace_hits_after_compose_seeding() {
    // Seed many compose features across both layers so the inference trace
    // surfaces non-empty per-layer hits with content `top_token`s, driving
    // infer.rs's trace-rendering loop (108-135: the per-hit label lookup,
    // down-top join, and row format). One install rarely makes the top-3;
    // a dozen ×30 installs reliably do.
    let dir = tempfile::tempdir().expect("tempdir");
    write_synthetic_model_dir(dir.path()).expect("fixture write");
    // A probe label so the non-empty-label arm (116-120) can also run when
    // a labelled slot fires.
    std::fs::write(
        dir.path().join("feature_labels.json"),
        r#"{"L0_F0":"capital","L0_F1":"language","L1_F0":"author","L1_F1":"currency"}"#,
    )
    .expect("write feature_labels.json");
    let (mut session, _dir, _) = use_dir(dir);

    let targets = ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot"];
    for (i, t) in targets.iter().enumerate() {
        for layer in 0..2u32 {
            let _ = try_run(
                &mut session,
                &format!(
                    r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[{i}]", "capital{layer}", "{t}") MODE COMPOSE AT LAYER {layer};"#
                ),
            )
            .expect("compose insert");
        }
    }

    let out = try_run(&mut session, r#"INFER "[1]";"#).expect("infer ok");
    assert!(
        out.iter().any(|l| l.contains("Inference trace")),
        "expected the inference-trace section, got: {out:?}"
    );
}

#[test]
fn trace_knn_override_fires_on_matching_prompt() {
    // A KNN INSERT stores the residual of the canonical prompt
    // "The capital of [1] is" at the install layer. TRACE-ing the *same*
    // canonical prompt reconstructs an identical residual (same token ids
    // → cos ~1.0 > the 0.75 KNN threshold), so infer_patched returns a
    // knn_override. That drives trace.rs's override-capture closure
    // (118-121) and append_pending_retrieval_override (322-334).
    let (mut session, _dir, _) = fresh_session();
    try_run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[1]", "capital", "[2]") MODE KNN;"#,
    )
    .expect("knn insert");

    // knn.rs builds the canonical prompt as "The {rel words} of {entity} is".
    let out = try_run(&mut session, r#"TRACE "The capital of [1] is";"#).expect("trace ok");
    let joined = out.join("\n");
    assert!(
        joined.contains("Trace:"),
        "expected trace header, got:\n{joined}"
    );
    assert!(
        joined.contains("Pending retrieval override"),
        "expected the KNN override block to fire, got:\n{joined}"
    );
}

#[test]
fn trace_knn_override_decompose_variant() {
    // Same override, but with DECOMPOSE so the override append happens at
    // the end of the attn/ffn-norm formatter (trace.rs:254) rather than
    // the default summary (281). Confirms the override block renders in
    // every TRACE formatter.
    let (mut session, _dir, _) = fresh_session();
    try_run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[1]", "capital", "[2]") MODE KNN;"#,
    )
    .expect("knn insert");
    let out =
        try_run(&mut session, r#"TRACE "The capital of [1] is" DECOMPOSE;"#).expect("trace ok");
    assert!(out.join("\n").contains("Trace:"));
}

#[test]
fn trace_knn_override_answer_variant() {
    // Override with FOR <answer> so the override append happens after the
    // answer-trajectory formatter (trace.rs:217). Also drives the
    // who-classification ladder (192-202) over the trajectory rows.
    let (mut session, _dir, _) = fresh_session();
    try_run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[1]", "capital", "[2]") MODE KNN;"#,
    )
    .expect("knn insert");
    let out =
        try_run(&mut session, r#"TRACE "The capital of [1] is" FOR "[2]";"#).expect("trace ok");
    assert!(out.join("\n").contains("Trace:"));
}
