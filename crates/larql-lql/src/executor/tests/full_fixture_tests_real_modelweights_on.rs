//! Full-fixture tests (real ModelWeights on disk)

use super::*;

#[test]
fn full_fixture_loads_via_use() {
    let (session, dir) = full_vindex_session("loads");
    assert!(matches!(session.backend, Backend::Vindex { .. }));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn infer_against_full_fixture_runs() {
    let (mut session, dir) = full_vindex_session("infer_basic");
    // Tokeniser maps any string to `[N]` token IDs in 0..32. Prompt
    // produces a small prefix the FFN walk can run against.
    let stmt = parser::parse(r#"INFER "[1] [2]" TOP 3;"#).unwrap();
    let _ = session.execute(&stmt).expect("INFER");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn infer_with_compare_mode_runs() {
    let (mut session, dir) = full_vindex_session("infer_compare");
    let stmt = parser::parse(r#"INFER "[3]" TOP 2 COMPARE;"#).unwrap();
    let _ = session.execute(&stmt).expect("INFER COMPARE");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn explain_infer_against_full_fixture_runs() {
    let (mut session, dir) = full_vindex_session("explain_infer");
    let stmt = parser::parse(r#"EXPLAIN INFER "[1] [2]";"#).unwrap();
    let _ = session.execute(&stmt).expect("EXPLAIN INFER");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn explain_infer_verbose_with_attention_runs() {
    let (mut session, dir) = full_vindex_session("explain_infer_attn");
    let stmt = parser::parse(r#"EXPLAIN INFER "[5]" VERBOSE WITH ATTENTION;"#).unwrap();
    let _ = session
        .execute(&stmt)
        .expect("EXPLAIN INFER VERBOSE WITH ATTENTION");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn trace_against_full_fixture_runs() {
    let (mut session, dir) = full_vindex_session("trace_basic");
    let stmt = parser::parse(r#"TRACE "[1] [2]";"#).unwrap();
    let _ = session.execute(&stmt).expect("TRACE");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn trace_with_decompose_runs() {
    let (mut session, dir) = full_vindex_session("trace_decompose");
    let stmt = parser::parse(r#"TRACE "[1]" DECOMPOSE;"#).unwrap();
    let _ = session.execute(&stmt).expect("TRACE DECOMPOSE");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn insert_compose_against_full_fixture_runs() {
    // Compose-mode INSERT runs the full pipeline: plan → capture
    // residuals → install slots → balance → cross-fact regression
    // check. Each phase calls `predict_with_ffn` against the random-
    // init weights; outputs are nonsense but every code path runs.
    //
    // Entity/relation/target use `[N]` patterns to land inside the
    // test tokenizer's vocab. Layer 0 is pinned so the plan picks a
    // single layer that exists in the 2-layer fixture.
    let (mut session, dir) = full_vindex_session("insert_compose");
    let stmt = parser::parse(
        r#"INSERT INTO EDGES (entity, relation, target)
           VALUES ("[1]", "[2]", "[5]") AT LAYER 0 MODE COMPOSE;"#,
    )
    .unwrap();
    let _ = session.execute(&stmt).expect("INSERT compose");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn three_compose_inserts_drive_cross_fact_regression_loop() {
    // After two prior compose installs, the third INSERT runs
    // `cross_fact_regression_check` with `priors_to_check` populated
    // (rev().take(MAX_PRIORS_CHECKED) yields 2 entries). Random
    // weights make the priors regress → shrink-and-retry loop runs
    // until CROSS_ITERS exhausts.
    let (mut session, dir) = full_vindex_session("compose_cross_fact");

    for (e, r, t) in &[
        ("[1]", "[2]", "[5]"),
        ("[3]", "[2]", "[6]"),
        ("[7]", "[2]", "[9]"),
    ] {
        let sql = format!(
            r#"INSERT INTO EDGES (entity, relation, target)
               VALUES ("{e}", "{r}", "{t}") AT LAYER 0 MODE COMPOSE;"#,
        );
        let stmt = parser::parse(&sql).unwrap();
        // Random weights mean some installs may fail; accept either
        // outcome but exercise the cross-fact regression path.
        let _ = session.execute(&stmt);
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn insert_compose_with_explicit_alpha_and_confidence() {
    let (mut session, dir) = full_vindex_session("insert_compose_alpha");
    let stmt = parser::parse(
        r#"INSERT INTO EDGES (entity, relation, target)
           VALUES ("[3]", "[4]", "[7]")
           AT LAYER 1
           CONFIDENCE 0.95
           ALPHA 0.20
           MODE COMPOSE;"#,
    )
    .unwrap();
    let _ = session
        .execute(&stmt)
        .expect("INSERT compose with alpha+confidence");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn rebalance_after_compose_insert_runs_real_loop() {
    // After a successful compose-mode INSERT, `installed_edges` has
    // a fact, so REBALANCE skips the early-exit and enters the
    // fixed-point loop (loads weights/tokenizer, runs probe walks).
    let (mut session, dir) = full_vindex_session("rebalance_after_compose");

    let insert = parser::parse(
        r#"INSERT INTO EDGES (entity, relation, target)
           VALUES ("[1]", "[2]", "[5]") AT LAYER 0 MODE COMPOSE;"#,
    )
    .unwrap();
    let _ = session.execute(&insert).expect("INSERT compose");

    let rebal = parser::parse("REBALANCE MAX 2;").unwrap();
    let _ = session.execute(&rebal).expect("REBALANCE after compose");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn insert_knn_against_full_fixture_runs() {
    let (mut session, dir) = knn_full_vindex_session("insert_knn");
    let stmt = parser::parse(
        r#"INSERT INTO EDGES (entity, relation, target)
           VALUES ("[1]", "[2]", "[5]") MODE KNN;"#,
    )
    .unwrap();
    let _ = session.execute(&stmt).expect("INSERT KNN");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn compact_minor_after_knn_insert_promotes_l0_to_l1() {
    // KNN INSERT writes to L0 (knn_store). COMPACT MINOR then
    // promotes those entries to L1 via compose-mode reinstall.
    let (mut session, dir) = knn_full_vindex_session("compact_minor_promotes");

    let insert = parser::parse(
        r#"INSERT INTO EDGES (entity, relation, target)
           VALUES ("[1]", "[2]", "[5]") AT LAYER 0 MODE KNN;"#,
    )
    .unwrap();
    let _ = session.execute(&insert).expect("INSERT KNN");

    let compact = parser::parse("COMPACT MINOR;").unwrap();
    // COMPACT MINOR may succeed or fail (compose-mode reinstall might
    // hit an internal error against the random-init model). Either way
    // it exercises the L0→L1 loop.
    let _ = session.execute(&compact);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn infer_against_synthetic_vindex_errors_without_model_weights() {
    // Basic synthetic fixture has has_model_weights=false. INFER must
    // surface a clean "requires model weights" error rather than panic.
    let (mut session, dir) = vindex_session("infer_no_weights");
    let stmt = parser::parse(r#"INFER "[1]";"#).unwrap();
    let err = session.execute(&stmt).unwrap_err();
    assert!(err.to_string().contains("model weights"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn explain_infer_against_synthetic_vindex_errors_without_model_weights() {
    let (mut session, dir) = vindex_session("explain_infer_no_weights");
    let stmt = parser::parse(r#"EXPLAIN INFER "[1]";"#).unwrap();
    let err = session.execute(&stmt).unwrap_err();
    assert!(err.to_string().contains("model weights"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn infer_on_weight_backend_returns_dense_predictions() {
    // Backend::Weight short-circuits in `exec_infer` (line 17-43 of
    // infer.rs): runs `predict` directly on the weights with no walk
    // FFN / vindex involvement.
    let mut session = weight_backend_session("test/weight-backend-infer");
    let stmt = parser::parse(r#"INFER "[1] [2]";"#).unwrap();
    let out = session.execute(&stmt).expect("INFER on Backend::Weight");
    let joined = out.join("\n");
    assert!(
        joined.contains("dense"),
        "expected dense-prediction header, got: {joined}",
    );
}

#[test]
fn infer_on_weight_backend_with_compare_skips_tip_line() {
    // The `if !compare { ... Tip: EXTRACT ... }` branch is the only
    // path-difference in the Backend::Weight arm: with COMPARE the
    // tip line is suppressed.
    let mut session = weight_backend_session("test/weight-backend-compare");
    let stmt = parser::parse(r#"INFER "[3]" COMPARE;"#).unwrap();
    let out = session
        .execute(&stmt)
        .expect("INFER COMPARE on Backend::Weight");
    let joined = out.join("\n");
    assert!(
        joined.contains("dense"),
        "expected dense header, got: {joined}",
    );
    // The "Tip: EXTRACT into a vindex..." line should NOT be present
    // when COMPARE is set.
    assert!(
        !joined.contains("EXTRACT into a vindex"),
        "Tip line should be suppressed when COMPARE is set, got: {joined}",
    );
}

#[test]
fn trace_on_weight_backend_runs_dense_path() {
    // TRACE short-circuits to the Backend::Weight arm (line 28-37 of
    // trace.rs) and runs the decomposed forward via WeightFfn.
    let mut session = weight_backend_session("test/weight-backend-trace");
    let stmt = parser::parse(r#"TRACE "[1] [2]";"#).unwrap();
    let _ = session.execute(&stmt).expect("TRACE on Backend::Weight");
}

#[test]
fn trace_on_synthetic_vindex_errors_without_model_weights() {
    // Basic synthetic fixture has has_model_weights=false; TRACE
    // should surface the model-weights error path.
    let (mut session, dir) = vindex_session("trace_no_weights_v2");
    let stmt = parser::parse(r#"TRACE "[1]";"#).unwrap();
    let err = session.execute(&stmt).unwrap_err();
    assert!(err.to_string().contains("model weights"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn explain_infer_on_weight_backend_runs_dense_summary() {
    // `exec_infer_trace` short-circuits at line 25 to
    // `exec_infer_trace_dense` — produces a dense-only summary with
    // no per-feature trace.
    let mut session = weight_backend_session("test/weight-backend-explain");
    let stmt = parser::parse(r#"EXPLAIN INFER "[5]";"#).unwrap();
    let out = session
        .execute(&stmt)
        .expect("EXPLAIN INFER on Backend::Weight");
    let joined = out.join("\n");
    assert!(
        joined.contains("dense"),
        "expected dense trace header, got: {joined}",
    );
    assert!(
        joined.contains("EXTRACT for full trace"),
        "expected the EXTRACT-for-trace tip, got: {joined}",
    );
}

#[test]
fn diff_current_with_weight_backend_errors() {
    // `resolve_vindex_ref` Current arm with a Weight backend should
    // surface the "CURRENT refers to a live model" error suggesting
    // EXTRACT first, exercising lines 215-219 of diff.rs.
    let mut session = weight_backend_session("test/weight-backend-diff");
    // Path b doesn't matter — Current is rejected before path resolution.
    let stmt = parser::parse(r#"DIFF CURRENT "/tmp/no_such_vindex_xyz";"#).unwrap();
    let err = session.execute(&stmt).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("live model") && msg.contains("EXTRACT"),
        "expected weight-backend CURRENT error, got: {msg}",
    );
}

#[test]
fn infer_with_canonical_knn_prompt_triggers_override_branch() {
    // INSERT KNN stores the residual at install_layer for the canonical
    // prompt "The {rel} of {entity} is". Re-running INFER with the same
    // prompt forces the cosine match path: the captured residual matches
    // the stored key 1:1, so apply_knn_override returns Some(...) and the
    // formatter renders the override-row branch in infer.rs.
    let (mut session, dir) = knn_full_vindex_session("infer_canonical_override");

    let insert = parser::parse(
        r#"INSERT INTO EDGES (entity, relation, target)
           VALUES ("[1]", "[2]", "[5]") AT LAYER 0 MODE KNN;"#,
    )
    .unwrap();
    let _ = session.execute(&insert).expect("INSERT KNN");

    // Same canonical prompt the INSERT path used internally — this is
    // the only INFER input guaranteed to hit the cosine threshold on
    // the synthetic fixture.
    let stmt = parser::parse(r#"INFER "The [2] of [1] is";"#).unwrap();
    let out = session
        .execute(&stmt)
        .expect("INFER on canonical KNN prompt");
    let joined = out.join("\n");
    // The override branch surfaces a 100% probability line and the
    // post-logits override note.
    assert!(
        joined.contains("100.00%"),
        "expected KNN override row with 100% prob, got: {joined}",
    );
    assert!(
        joined.contains("KNN override"),
        "expected KNN override note, got: {joined}",
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn explain_infer_with_canonical_knn_prompt_renders_override_summary() {
    // Same setup as infer_with_canonical_knn_prompt_triggers_override_branch
    // but for EXPLAIN INFER, which routes through `exec_infer_trace`
    // and renders the override summary line + "Pending retrieval
    // override" note.
    let (mut session, dir) = knn_full_vindex_session("explain_infer_canonical_override");

    let insert = parser::parse(
        r#"INSERT INTO EDGES (entity, relation, target)
           VALUES ("[1]", "[2]", "[7]") AT LAYER 0 MODE KNN;"#,
    )
    .unwrap();
    let _ = session.execute(&insert).expect("INSERT KNN");

    let stmt = parser::parse(r#"EXPLAIN INFER "The [2] of [1] is";"#).unwrap();
    let out = session
        .execute(&stmt)
        .expect("EXPLAIN INFER canonical KNN prompt");
    let joined = out.join("\n");
    assert!(
        joined.contains("Pending retrieval override"),
        "expected 'Pending retrieval override' note, got: {joined}",
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn infer_after_knn_insert_drives_override_branch() {
    // INSERT KNN writes a residual key into the KnnStore. The next
    // INFER call sees `infer.knn_override = Some(_)` and renders the
    // override-row formatting branch.
    let (mut session, dir) = knn_full_vindex_session("infer_knn_override");

    let insert = parser::parse(
        r#"INSERT INTO EDGES (entity, relation, target)
           VALUES ("[1]", "[2]", "[5]") AT LAYER 0 MODE KNN;"#,
    )
    .unwrap();
    let _ = session.execute(&insert).expect("INSERT KNN");

    // Same prompt as the canonical for "[1]" relation "[2]" — the
    // KNN side-channel might or might not match (depends on cosine);
    // either way we exercise the predict path with knn_override
    // populated in the session.
    let stmt = parser::parse(r#"INFER "[1] [2] [3]";"#).unwrap();
    let _ = session.execute(&stmt).expect("INFER post-KNN");
    let _ = std::fs::remove_dir_all(&dir);
}
