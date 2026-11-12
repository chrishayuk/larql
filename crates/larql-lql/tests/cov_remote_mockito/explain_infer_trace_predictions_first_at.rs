//! EXPLAIN INFER (trace, predictions-first, attention, relations
//! EXPLAIN WALK over remote (routes to remote_walk)
//! SHOW RELATIONS (probe section, raw section, examples, error)
//! SELECT (all WHERE field branches, edge table, total, empty)
//! INSERT (success, default confidence, error)
//! DELETE (success, missing layer/feature precondition, error)
//! UPDATE (with target, without target, error)
//! Local patch management over remote (APPLY/SHOW/REMOVE)
//! execute_remote dispatch fall-through + Pipe arm
//! Session::default() smoke (covers Default impl)

use super::*;

#[test]
fn explain_infer_renders_layer_trace() {
    let (mut server, mut session) = connect();
    server
        .mock("POST", "/v1/explain-infer")
        .with_status(200)
        .with_body(
            serde_json::json!({
                "predictions": [{"token": "Paris", "probability": 0.92}],
                "trace": [{
                    "layer": 26,
                    "features": [{
                        "feature": 1, "gate_score": 14.2, "top_token": "Paris",
                        "relation": "capital",
                        "top_tokens": ["Paris", "France", "Europe"],
                    }],
                }],
                "latency_ms": 30.0,
            })
            .to_string(),
        )
        .create();
    let joined = ok_joined(&mut session, r#"EXPLAIN INFER "test" TOP 3;"#);
    assert!(joined.contains("Inference trace"));
    assert!(joined.contains("Prediction: Paris"));
    assert!(joined.contains("L26"));
    assert!(joined.contains("capital"));
    assert!(joined.contains("Paris, France, Europe"));
    assert!(joined.contains("ms (remote)"));
}

#[test]
fn explain_infer_unlabeled_feature_without_top_tokens() {
    // Non-attention trace with a null-relation feature and no top_tokens
    // array exercises the empty-relation label branch and the
    // top_tokens default-empty branch.
    let (mut server, mut session) = connect();
    server
        .mock("POST", "/v1/explain-infer")
        .with_status(200)
        .with_body(
            serde_json::json!({
                "predictions": [{"token": "Paris", "probability": 0.5}],
                "trace": [{
                    "layer": 9,
                    "features": [{
                        "feature": 4, "gate_score": 3.0, "top_token": "Z",
                        "relation": null,
                    }],
                }],
                "latency_ms": 6.0,
            })
            .to_string(),
        )
        .create();
    let joined = ok_joined(&mut session, r#"EXPLAIN INFER "p";"#);
    assert!(joined.contains("L 9"));
    assert!(joined.contains("F4"));
}

#[test]
fn explain_infer_relations_only_skips_unlabeled_features() {
    let (mut server, mut session) = connect();
    server
        .mock("POST", "/v1/explain-infer")
        .with_status(200)
        .with_body(
            serde_json::json!({
                "predictions": [{"token": "X", "probability": 0.5}],
                "trace": [{
                    "layer": 5,
                    "features": [
                        // unlabeled (relation null) → skipped under RELATIONS ONLY
                        {"feature": 0, "gate_score": 9.0, "top_token": "T", "relation": null},
                        // labeled → rendered
                        {"feature": 1, "gate_score": 8.0, "top_token": "U", "relation": "rel"},
                    ],
                }],
            })
            .to_string(),
        )
        .create();
    let joined = ok_joined(
        &mut session,
        r#"EXPLAIN INFER "p" KNOWLEDGE RELATIONS ONLY;"#,
    );
    // labeled relation is rendered; band label appears in the header.
    assert!(joined.contains("rel"));
    assert!(joined.contains("(knowledge)"));
}

#[test]
fn explain_infer_with_attention_compact_format() {
    let (mut server, mut session) = connect();
    server
        .mock("POST", "/v1/explain-infer")
        .with_status(200)
        .with_body(
            serde_json::json!({
                "predictions": [{"token": "X", "probability": 0.5}],
                "trace": [{
                    "layer": 5,
                    "features": [{"feature": 0, "gate_score": 9.0, "top_token": "T",
                                  "relation": "rel"}],
                    "attention": [{"token": "X", "weight": 0.7}],
                    "lens": {"token": "Y", "probability": 0.3},
                }],
            })
            .to_string(),
        )
        .create();
    let joined = ok_joined(&mut session, r#"EXPLAIN INFER "p" TOP 1 WITH ATTENTION;"#);
    assert!(joined.contains("L"));
    assert!(joined.contains("rel"));
}

#[test]
fn explain_infer_with_attention_and_relations_only_filters_then_uses_lens() {
    // RELATIONS ONLY + WITH ATTENTION + an unlabeled feature exercises
    // the `feature_str = None` path where the row is still emitted
    // because the lens part is non-empty.
    let (mut server, mut session) = connect();
    server
        .mock("POST", "/v1/explain-infer")
        .with_status(200)
        .with_body(
            serde_json::json!({
                "predictions": [{"token": "X", "probability": 0.5}],
                "trace": [{
                    "layer": 7,
                    "features": [{"feature": 0, "gate_score": 1.0, "top_token": "T",
                                  "relation": null}],
                    "lens": {"token": "Z", "probability": 0.4},
                }],
            })
            .to_string(),
        )
        .create();
    let joined = ok_joined(
        &mut session,
        r#"EXPLAIN INFER "p" RELATIONS ONLY WITH ATTENTION;"#,
    );
    assert!(joined.contains("L"));
}

#[test]
fn explain_infer_knn_override_surfaces_pending_note() {
    let (mut server, mut session) = connect();
    server
        .mock("POST", "/v1/explain-infer")
        .with_status(200)
        .with_body(
            serde_json::json!({
                "knn_override": {"token": "Madrid", "cosine": 0.88, "layer": 12},
                "trace": [],
            })
            .to_string(),
        )
        .create();
    let joined = ok_joined(&mut session, r#"EXPLAIN INFER "q";"#);
    assert!(joined.contains("Prediction: Madrid"));
    assert!(joined.contains("Pending retrieval override"));
}

#[test]
fn explain_infer_errors_on_http_500() {
    let (mut server, mut session) = connect();
    server
        .mock("POST", "/v1/explain-infer")
        .with_status(500)
        .with_body("boom")
        .create();
    let err = run(&mut session, r#"EXPLAIN INFER "p";"#).unwrap_err();
    assert!(err.contains("500"), "got: {err}");
}

#[test]
fn explain_walk_routes_to_remote_walk() {
    let (mut server, mut session) = connect();
    server
        .mock("GET", mockito::Matcher::Regex(r"/v1/walk".into()))
        .with_status(200)
        .with_body(serde_json::json!({"hits": [], "latency_ms": 1.0}).to_string())
        .create();
    let joined = ok_joined(&mut session, r#"EXPLAIN WALK "p" LAYERS 0-3;"#);
    assert!(joined.contains("Feature scan"));
}

#[test]
fn show_relations_renders_probe_and_table() {
    let (mut server, mut session) = connect();
    server
        .mock("GET", "/v1/relations")
        .with_status(200)
        .with_body(
            serde_json::json!({
                "probe_relations": [
                    {"name": "capital", "count": 12},
                    {"name": "language", "count": 8}
                ],
                "probe_count": 2,
                "relations": [
                    {"name": "Paris", "count": 3, "max_score": 14.0,
                     "min_layer": 20, "max_layer": 26, "examples": ["a", "b"]}
                ],
            })
            .to_string(),
        )
        .create();
    // VERBOSE → both probe section and raw table; WITH EXAMPLES → e.g.
    let joined = ok_joined(&mut session, "SHOW RELATIONS VERBOSE WITH EXAMPLES;");
    assert!(joined.contains("Probe-confirmed"));
    assert!(joined.contains("capital"));
    assert!(joined.contains("Top output tokens"));
    assert!(joined.contains("e.g. a, b"));
}

#[test]
fn show_relations_raw_mode_skips_probe_section() {
    let (mut server, mut session) = connect();
    server
        .mock("GET", "/v1/relations")
        .with_status(200)
        .with_body(
            serde_json::json!({
                "probe_relations": [{"name": "capital", "count": 1}],
                "probe_count": 1,
                "relations": [
                    {"name": "Paris", "count": 3, "max_score": 14.0,
                     "min_layer": 20, "max_layer": 26}
                ],
            })
            .to_string(),
        )
        .create();
    // RAW → probe section skipped, raw table shown, no examples.
    let joined = ok_joined(&mut session, "SHOW RELATIONS RAW;");
    assert!(!joined.contains("Probe-confirmed"));
    assert!(joined.contains("Top output tokens"));
}

#[test]
fn show_relations_errors_on_http_503() {
    let (mut server, mut session) = connect();
    server
        .mock("GET", "/v1/relations")
        .with_status(503)
        .with_body("down")
        .create();
    let err = run(&mut session, "SHOW RELATIONS;").unwrap_err();
    assert!(err.contains("503"), "got: {err}");
}

#[test]
fn select_with_all_where_fields_renders_table() {
    let (mut server, mut session) = connect();
    server
        .mock("POST", "/v1/select")
        .with_status(200)
        .with_body(
            serde_json::json!({
                "edges": [{"layer": 5, "feature": 1, "target": "Paris",
                           "c_score": 0.95, "relation": "capital"}],
                "total": 1,
            })
            .to_string(),
        )
        .create();
    // entity / relation / layer / confidence WHERE fields each map to a
    // distinct branch in remote_select's condition loop.
    let joined = ok_joined(
        &mut session,
        r#"SELECT * FROM EDGES WHERE entity = "France" AND relation = "capital" AND layer = 5 AND confidence >= 0.5 LIMIT 10;"#,
    );
    assert!(joined.contains("Target"));
    assert!(joined.contains("Paris"));
    assert!(joined.contains("capital"));
    assert!(joined.contains("1 total"));
}

#[test]
fn select_with_integer_score_floor_is_sent_as_min_confidence() {
    // `score >= 1` parses to a Value::Integer — a number like any other
    // to the predicate grammar.
    let (mut server, mut session) = connect();
    server
        .mock("POST", "/v1/select")
        .with_status(200)
        .with_body(serde_json::json!({"edges": [], "total": 0}).to_string())
        .create();
    let joined = ok_joined(&mut session, "SELECT * FROM EDGES WHERE score >= 1;");
    assert!(joined.contains("(no matching edges)"));
}

#[test]
fn select_refuses_conditions_the_server_cannot_express_and_sends_nothing() {
    // The remote query carries entity, relation, layer and a confidence
    // floor. Each clause below used to be sent as something else:
    // `layer > 20` as `layer = 20`, `confidence < 0.1` as a floor of 0.1,
    // `relation != "x"` as `relation = "x"`, and `feature` not at all.
    let (mut server, mut session) = connect();
    let never = server.mock("POST", "/v1/select").expect(0).create();
    for sql in [
        "SELECT * FROM EDGES WHERE layer > 20;",
        "SELECT * FROM EDGES WHERE confidence < 0.1;",
        "SELECT * FROM EDGES WHERE confidence > 0.5;",
        r#"SELECT * FROM EDGES WHERE relation != "capital";"#,
        "SELECT * FROM EDGES WHERE feature = 3;",
        r#"SELECT * FROM EDGES WHERE colour = "red";"#,
    ] {
        let parsed = larql_lql::parse(sql).expect("parse");
        let err = session
            .execute(&parsed)
            .expect_err("not expressible remotely")
            .to_string();
        assert!(err.contains("unsupported WHERE clause"), "{sql}: {err}");
    }
    never.assert();
}

#[test]
fn select_empty_emits_no_match_line() {
    let (mut server, mut session) = connect();
    server
        .mock("POST", "/v1/select")
        .with_status(200)
        .with_body(serde_json::json!({"edges": [], "total": 0}).to_string())
        .create();
    let joined = ok_joined(&mut session, "SELECT * FROM EDGES;");
    assert!(joined.contains("(no matching edges)"));
}

#[test]
fn select_errors_on_http_500() {
    let (mut server, mut session) = connect();
    server
        .mock("POST", "/v1/select")
        .with_status(500)
        .with_body("boom")
        .create();
    let err = run(&mut session, "SELECT * FROM EDGES;").unwrap_err();
    assert!(err.contains("500"), "got: {err}");
}

#[test]
fn insert_renders_summary() {
    let (mut server, mut session) = connect();
    server
        .mock("POST", "/v1/insert")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({"inserted": 3, "mode": "compose", "latency_ms": 17.0}).to_string(),
        )
        .create();
    let joined = ok_joined(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("France", "capital", "Paris") AT LAYER 26 CONFIDENCE 0.9;"#,
    );
    assert!(joined.contains("France"));
    assert!(joined.contains("Paris"));
    assert!(joined.contains("compose"));
    assert!(joined.contains("3 layers"));
    assert!(joined.contains("ms (remote)"));
}

#[test]
fn insert_uses_default_confidence_when_omitted() {
    let (mut server, mut session) = connect();
    server
        .mock("POST", "/v1/insert")
        .with_status(200)
        .with_body(serde_json::json!({"inserted": 1, "mode": "knn", "latency_ms": 5.0}).to_string())
        .create();
    let joined = ok_joined(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("X", "r", "Y");"#,
    );
    assert!(joined.contains("knn"));
}

#[test]
fn insert_errors_on_http_500() {
    let (mut server, mut session) = connect();
    server
        .mock("POST", "/v1/insert")
        .with_status(500)
        .with_body("boom")
        .create();
    let err = run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("X", "r", "Y");"#,
    )
    .unwrap_err();
    assert!(err.contains("500"), "got: {err}");
}

#[test]
fn delete_posts_patch_and_renders_summary() {
    let (mut server, mut session) = connect();
    server
        .mock("POST", "/v1/patches/apply")
        .with_status(200)
        .with_body(serde_json::json!({"applied": 1}).to_string())
        .create();
    let joined = ok_joined(
        &mut session,
        "DELETE FROM EDGES WHERE layer = 26 AND feature = 7;",
    );
    assert!(joined.contains("L26"));
    assert!(joined.contains("F7"));
    assert!(joined.contains("remote server"));
}

#[test]
fn delete_errors_without_feature_filter() {
    let (_server, mut session) = connect();
    let err = run(&mut session, "DELETE FROM EDGES WHERE layer = 26;").unwrap_err();
    assert!(err.to_lowercase().contains("feature"), "got: {err}");
}

#[test]
fn delete_errors_on_http_500() {
    let (mut server, mut session) = connect();
    server
        .mock("POST", "/v1/patches/apply")
        .with_status(500)
        .with_body("boom")
        .create();
    let err = run(
        &mut session,
        "DELETE FROM EDGES WHERE layer = 0 AND feature = 0;",
    )
    .unwrap_err();
    assert!(err.contains("500"), "got: {err}");
}

#[test]
fn update_with_target_renders_target_clause() {
    let (mut server, mut session) = connect();
    server
        .mock("POST", "/v1/patches/apply")
        .with_status(200)
        .with_body(serde_json::json!({"applied": 1}).to_string())
        .create();
    let joined = ok_joined(
        &mut session,
        r#"UPDATE EDGES SET target = "Madrid", confidence = 0.85 WHERE layer = 26 AND feature = 0;"#,
    );
    assert!(joined.contains("target=Madrid"));
    assert!(joined.contains("L26"));
}

#[test]
fn update_without_target_omits_target_clause() {
    let (mut server, mut session) = connect();
    server
        .mock("POST", "/v1/patches/apply")
        .with_status(200)
        .with_body(serde_json::json!({"applied": 1}).to_string())
        .create();
    let joined = ok_joined(
        &mut session,
        "UPDATE EDGES SET confidence = 0.5 WHERE layer = 0 AND feature = 0;",
    );
    assert!(!joined.contains("target="));
    assert!(joined.contains("L0 F0"));
}

#[test]
fn update_refuses_a_field_it_cannot_set_before_sending_anything() {
    // No mock is registered: a request would fail the test another way.
    let (_server, mut session) = connect();
    let err = run(
        &mut session,
        r#"UPDATE EDGES SET bogus = 1 WHERE layer = 0 AND feature = 0;"#,
    )
    .unwrap_err();
    assert!(err.contains("`bogus`"), "got: {err}");
    assert!(err.contains("Nothing was updated"), "got: {err}");
}

#[test]
fn update_errors_without_layer_filter() {
    let (_server, mut session) = connect();
    let err = run(
        &mut session,
        r#"UPDATE EDGES SET target = "Z" WHERE feature = 0;"#,
    )
    .unwrap_err();
    assert!(err.to_lowercase().contains("layer"), "got: {err}");
}

#[test]
fn apply_show_remove_local_patch_lifecycle() {
    let (_server, mut session) = connect();

    // SHOW PATCHES with none applied.
    let empty = ok_joined(&mut session, "SHOW PATCHES;");
    assert!(empty.contains("(no local patches)"));

    // APPLY a real patch file.
    let p = write_insert_patch("Hyrule", Some("capital"), "Hateno", 26);
    let applied = ok_joined(&mut session, &format!(r#"APPLY PATCH "{}";"#, sql_path(&p)));
    assert!(applied.contains("Applied locally"));
    assert!(applied.contains("client-side"));

    // SHOW PATCHES now lists the entry.
    let listed = ok_joined(&mut session, "SHOW PATCHES;");
    assert!(listed.contains("Local patches"));

    // REMOVE by description name.
    let removed = ok_joined(&mut session, r#"REMOVE PATCH "insert Hyrule->Hateno";"#);
    assert!(removed.contains("Removed local patch"));
    let _ = std::fs::remove_file(p);
}

#[test]
fn apply_local_patch_errors_on_missing_file() {
    let (_server, mut session) = connect();
    let err = run(
        &mut session,
        r#"APPLY PATCH "/tmp/no_such_remote_patch_xyz.vlp";"#,
    )
    .unwrap_err();
    assert!(err.contains("patch not found"), "got: {err}");
}

#[test]
fn remove_local_patch_errors_on_unknown_name() {
    let (_server, mut session) = connect();
    let err = run(&mut session, r#"REMOVE PATCH "does-not-exist";"#).unwrap_err();
    assert!(err.contains("not found"), "got: {err}");
}

#[test]
fn unsupported_statement_on_remote_errors_with_help() {
    // TRACE is not in the remote-supported verb set → the `_ =>` arm of
    // execute_remote fires with the help message.
    let (_server, mut session) = connect();
    let err = run(&mut session, r#"TRACE "prompt";"#).unwrap_err();
    assert!(
        err.contains("not supported on a remote backend"),
        "got: {err}"
    );
    assert!(err.contains("TRACE requires a local vindex"), "got: {err}");
}

#[test]
fn re_use_remote_while_remote_redispatches_through_exec_use() {
    // A `USE REMOTE` issued while already on a Remote backend hits the
    // `Statement::Use` arm of execute_remote (which delegates to
    // exec_use → exec_use_remote), reconnecting to a second server.
    let (_first, mut session) = connect();

    let mut server2 = mockito::Server::new();
    server2
        .mock("GET", ENDPOINT_STATS)
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(stats_body())
        .expect_at_least(1)
        .create();
    let joined = ok_joined(&mut session, &format!(r#"USE REMOTE "{}";"#, server2.url()));
    assert!(joined.contains("Connected"));
    assert!(joined.contains("test-model"));
}

#[test]
fn pipe_statement_over_remote_chains_both_sides() {
    // A piped pair routes through the Pipe arm of execute_remote, which
    // recurses into execute() for each side.
    let (mut server, mut session) = connect();
    server
        .mock("GET", mockito::Matcher::Regex(r"/v1/walk".into()))
        .with_status(200)
        .with_body(serde_json::json!({"hits": [], "latency_ms": 1.0}).to_string())
        .create();
    // STATS |> WALK — both supported remote verbs.
    let joined = ok_joined(&mut session, r#"STATS |> WALK "p";"#);
    assert!(joined.contains("test-model"));
    assert!(joined.contains("Feature scan"));
}

#[test]
fn session_default_has_no_backend() {
    // Session::default() delegates to Session::new() → Backend::None.
    // A statement requiring a backend errors rather than forwarding to
    // a remote server (the remote dispatch is only taken when Remote).
    let mut session = Session::default();
    let err = run(&mut session, "STATS;").unwrap_err();
    // No remote URL was set, so this cannot be the "Remote:" banner.
    assert!(!err.contains("Remote:"), "got: {err}");
}
