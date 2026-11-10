use super::*;
use crate::ast::{CompareOp, Value};

fn cond(field: &str, value: Value) -> Condition {
    Condition {
        field: field.into(),
        op: CompareOp::Eq,
        value,
    }
}

#[test]
fn require_layer_feature_accepts_explicit_pair() {
    let conds = vec![
        cond("layer", Value::Integer(7)),
        cond("feature", Value::Integer(42)),
    ];
    let (l, f) = require_layer_feature(&conds, "DELETE").unwrap();
    assert_eq!((l, f), (7, 42));
}

#[test]
fn require_layer_feature_errors_when_layer_missing() {
    // Regression: prior shape silently coerced missing fields to 0,
    // turning `WHERE foo = 1` into a destructive `DELETE L0 F0`.
    let conds = vec![cond("feature", Value::Integer(1))];
    let err = require_layer_feature(&conds, "DELETE").unwrap_err();
    assert!(err.to_string().contains("layer"));
    assert!(err.to_string().contains("DELETE"));
}

#[test]
fn require_layer_feature_errors_when_feature_missing() {
    let conds = vec![cond("layer", Value::Integer(0))];
    let err = require_layer_feature(&conds, "UPDATE").unwrap_err();
    assert!(err.to_string().contains("feature"));
    assert!(err.to_string().contains("UPDATE"));
}

#[test]
fn require_layer_feature_rejects_non_integer_value() {
    let conds = vec![
        cond("layer", Value::String("oops".into())),
        cond("feature", Value::Integer(1)),
    ];
    let err = require_layer_feature(&conds, "DELETE").unwrap_err();
    assert!(err.to_string().contains("layer"));
}

#[test]
fn require_layer_feature_rejects_negative_value() {
    let conds = vec![
        cond("layer", Value::Integer(-1)),
        cond("feature", Value::Integer(0)),
    ];
    let err = require_layer_feature(&conds, "DELETE").unwrap_err();
    assert!(err.to_string().contains("layer"));
}

#[test]
fn endpoint_constants_share_prefix() {
    // Pinned: every endpoint hangs off `/v1/`. A future v2 server
    // would need a deliberate sweep here.
    for ep in [
        ENDPOINT_STATS,
        ENDPOINT_DESCRIBE,
        ENDPOINT_WALK,
        ENDPOINT_INFER,
        ENDPOINT_EXPLAIN_INFER,
        ENDPOINT_RELATIONS,
        ENDPOINT_SELECT,
        ENDPOINT_INSERT,
        ENDPOINT_PATCHES_APPLY,
    ] {
        assert!(
            ep.starts_with("/v1/"),
            "expected /v1/-prefixed endpoint, got {ep:?}"
        );
    }
}

#[test]
#[allow(clippy::assertions_on_constants)]
fn default_timeout_is_reasonable() {
    // < 5s would race common deep-walk INFER; > 10min would let
    // the REPL freeze on a hung server.
    assert!(DEFAULT_REMOTE_TIMEOUT_SECS >= 5);
    assert!(DEFAULT_REMOTE_TIMEOUT_SECS <= 600);
}

// ── End-to-end mockito tests ────────────────────────────
//
// Each test stands up a `mockito::Server`, mocks one or more
// `/v1/*` endpoints with canned JSON, then drives the matching
// forwarder via the public verbs. Tests validate that the request
// hits the right path, parses the response shape, and surfaces a
// user-readable summary.

use crate::executor::Session;

fn connect(server_url: &str) -> Session {
    let mut session = Session::new();
    session
        .exec_use_remote(server_url)
        .expect("exec_use_remote with mocked /v1/stats");
    session
}

fn stats_body() -> String {
    serde_json::json!({
        "model": "test-model",
        "family": "llama",
        "layers": 32,
        "features": 4096,
        "hidden_size": 1024,
        "dtype": "f32",
        "extract_level": "all",
        "loaded": {"browse": true, "inference": false},
    })
    .to_string()
}

#[test]
fn use_remote_succeeds_when_server_reachable() {
    let mut server = mockito::Server::new();
    let _m = server
        .mock("GET", ENDPOINT_STATS)
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(stats_body())
        .create();

    let session = connect(&server.url());
    assert!(session.is_remote());
}

#[test]
fn use_remote_errors_on_5xx() {
    let mut server = mockito::Server::new();
    let _m = server
        .mock("GET", ENDPOINT_STATS)
        .with_status(503)
        .with_body("upstream down")
        .create();

    let mut session = Session::new();
    let err = session.exec_use_remote(&server.url()).unwrap_err();
    assert!(err.to_string().contains("503"));
}

#[test]
fn use_remote_errors_on_invalid_json() {
    let mut server = mockito::Server::new();
    let _m = server
        .mock("GET", ENDPOINT_STATS)
        .with_status(200)
        .with_body("not actually json")
        .create();

    let mut session = Session::new();
    let err = session.exec_use_remote(&server.url()).unwrap_err();
    assert!(err.to_string().contains("invalid response"));
}

#[test]
fn use_remote_errors_when_server_unreachable() {
    let mut session = Session::new();
    // Port 1 is reserved + nothing's listening there.
    let err = session.exec_use_remote("http://127.0.0.1:1").unwrap_err();
    assert!(err.to_string().contains("failed to connect"));
}

#[test]
fn remote_stats_renders_summary() {
    let mut server = mockito::Server::new();
    let _stats = server
        .mock("GET", ENDPOINT_STATS)
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(stats_body())
        .expect_at_least(1)
        .create();

    let session = connect(&server.url());
    let out = session.remote_stats().unwrap();
    let joined = out.join("\n");
    assert!(joined.contains("test-model"));
    assert!(joined.contains("Layers: 32"));
    assert!(joined.contains("Features: 4096"));
}

#[test]
fn remote_walk_renders_hits() {
    let mut server = mockito::Server::new();
    let _stats = server
        .mock("GET", ENDPOINT_STATS)
        .with_status(200)
        .with_body(stats_body())
        .create();
    let _walk = server
        .mock("GET", mockito::Matcher::Regex(r"/v1/walk".into()))
        .with_status(200)
        .with_body(
            serde_json::json!({
                "hits": [
                    {"layer": 5, "feature": 3, "gate_score": 12.5, "target": "Paris"},
                    {"layer": 7, "feature": 1, "gate_score": 9.0, "target": "France"}
                ],
                "latency_ms": 42.0,
            })
            .to_string(),
        )
        .create();

    let session = connect(&server.url());
    let out = session.remote_walk("test prompt", Some(3), None).unwrap();
    let joined = out.join("\n");
    assert!(joined.contains("Paris"));
    assert!(joined.contains("France"));
    assert!(joined.contains("42"));
}

#[test]
fn remote_describe_renders_edges() {
    let mut server = mockito::Server::new();
    let _stats = server
        .mock("GET", ENDPOINT_STATS)
        .with_status(200)
        .with_body(stats_body())
        .create();
    let _describe = server
        .mock("GET", mockito::Matcher::Regex(r"/v1/describe".into()))
        .with_status(200)
        .with_body(
            serde_json::json!({
                "edges": [
                    {
                        "target": "Paris",
                        "gate_score": 14.2,
                        "layer": 26,
                        "relation": "capital",
                        "source": "probe",
                        "also": ["French", "Europe"],
                    }
                ],
                "latency_ms": 15.0,
            })
            .to_string(),
        )
        .create();

    let session = connect(&server.url());
    let out = session
        .remote_describe("France", None, crate::ast::DescribeMode::Verbose)
        .unwrap();
    let joined = out.join("\n");
    assert!(joined.starts_with("France"));
    assert!(joined.contains("Paris"));
    assert!(joined.contains("capital"));
}

#[test]
fn remote_describe_no_edges_emits_friendly_line() {
    let mut server = mockito::Server::new();
    let _stats = server
        .mock("GET", ENDPOINT_STATS)
        .with_status(200)
        .with_body(stats_body())
        .create();
    let _describe = server
        .mock("GET", mockito::Matcher::Regex(r"/v1/describe".into()))
        .with_status(200)
        .with_body(serde_json::json!({"edges": [], "latency_ms": 5.0}).to_string())
        .create();

    let session = connect(&server.url());
    let out = session
        .remote_describe("X", None, crate::ast::DescribeMode::Brief)
        .unwrap();
    let joined = out.join("\n");
    assert!(joined.contains("(no edges found)"));
}

#[test]
fn remote_infer_renders_predictions() {
    let mut server = mockito::Server::new();
    let _stats = server
        .mock("GET", ENDPOINT_STATS)
        .with_status(200)
        .with_body(stats_body())
        .create();
    let _infer = server
        .mock("POST", ENDPOINT_INFER)
        .with_status(200)
        .with_body(
            serde_json::json!({
                "predictions": [
                    {"token": "Paris", "probability": 0.7},
                    {"token": "Lyon", "probability": 0.1}
                ],
                "latency_ms": 12.0,
            })
            .to_string(),
        )
        .create();

    let session = connect(&server.url());
    let out = session
        .remote_infer("The capital of France is", Some(2), false)
        .unwrap();
    let joined = out.join("\n");
    assert!(joined.contains("Paris"));
    assert!(joined.contains("70.00%"));
}

#[test]
fn remote_infer_compare_mode_renders_walk_and_dense() {
    let mut server = mockito::Server::new();
    let _stats = server
        .mock("GET", ENDPOINT_STATS)
        .with_status(200)
        .with_body(stats_body())
        .create();
    let _infer = server
        .mock("POST", ENDPOINT_INFER)
        .with_status(200)
        .with_body(
            serde_json::json!({
                "walk": [{"token": "Paris", "probability": 0.7}],
                "walk_ms": 10.0,
                "dense": [{"token": "Paris", "probability": 0.65}],
                "dense_ms": 80.0,
                "latency_ms": 92.0,
            })
            .to_string(),
        )
        .create();

    let session = connect(&server.url());
    let out = session.remote_infer("test", Some(1), true).unwrap();
    let joined = out.join("\n");
    assert!(joined.contains("Predictions (walk)"));
    assert!(joined.contains("Predictions (dense)"));
}

#[test]
fn remote_select_renders_edge_table() {
    let mut server = mockito::Server::new();
    let _stats = server
        .mock("GET", ENDPOINT_STATS)
        .with_status(200)
        .with_body(stats_body())
        .create();
    let _select = server
        .mock("POST", ENDPOINT_SELECT)
        .with_status(200)
        .with_body(
            serde_json::json!({
                "edges": [
                    {"layer": 5, "feature": 1, "target": "Paris", "c_score": 0.95, "relation": "capital"}
                ],
                "total": 1,
            })
            .to_string(),
        )
        .create();

    let session = connect(&server.url());
    let out = session.remote_select(&[], Some(10)).unwrap();
    let joined = out.join("\n");
    assert!(joined.contains("Paris"));
    assert!(joined.contains("capital"));
    assert!(joined.contains("1 total"));
}

#[test]
fn remote_select_empty_emits_no_match_line() {
    let mut server = mockito::Server::new();
    let _stats = server
        .mock("GET", ENDPOINT_STATS)
        .with_status(200)
        .with_body(stats_body())
        .create();
    let _select = server
        .mock("POST", ENDPOINT_SELECT)
        .with_status(200)
        .with_body(serde_json::json!({"edges": [], "total": 0}).to_string())
        .create();

    let session = connect(&server.url());
    let out = session.remote_select(&[], None).unwrap();
    let joined = out.join("\n");
    assert!(joined.contains("(no matching edges)"));
}

#[test]
fn remote_show_relations_renders_probe_section() {
    let mut server = mockito::Server::new();
    let _stats = server
        .mock("GET", ENDPOINT_STATS)
        .with_status(200)
        .with_body(stats_body())
        .create();
    let _rel = server
        .mock("GET", ENDPOINT_RELATIONS)
        .with_status(200)
        .with_body(
            serde_json::json!({
                "probe_relations": [
                    {"name": "capital", "count": 12},
                    {"name": "language", "count": 8}
                ],
                "probe_count": 2,
                "relations": [],
            })
            .to_string(),
        )
        .create();

    let session = connect(&server.url());
    let out = session
        .remote_show_relations(crate::ast::DescribeMode::Verbose, false)
        .unwrap();
    let joined = out.join("\n");
    assert!(joined.contains("Probe-confirmed"));
    assert!(joined.contains("capital"));
}

#[test]
fn remote_insert_returns_summary() {
    let mut server = mockito::Server::new();
    let _stats = server
        .mock("GET", ENDPOINT_STATS)
        .with_status(200)
        .with_body(stats_body())
        .create();
    let _insert = server
        .mock("POST", ENDPOINT_INSERT)
        .with_status(200)
        .with_body(
            serde_json::json!({
                "inserted": 3,
                "mode": "compose",
                "latency_ms": 250.0,
            })
            .to_string(),
        )
        .create();

    let session = connect(&server.url());
    let out = session
        .remote_insert("Atlantis", "capital", "Poseidon", Some(26), Some(0.95))
        .unwrap();
    let joined = out.join("\n");
    assert!(joined.contains("Atlantis"));
    assert!(joined.contains("Poseidon"));
    assert!(joined.contains("compose"));
}

#[test]
fn remote_delete_uses_explicit_layer_feature() {
    let mut server = mockito::Server::new();
    let _stats = server
        .mock("GET", ENDPOINT_STATS)
        .with_status(200)
        .with_body(stats_body())
        .create();
    let _apply = server
        .mock("POST", ENDPOINT_PATCHES_APPLY)
        .with_status(200)
        .with_body(serde_json::json!({"ok": true}).to_string())
        .create();

    let session = connect(&server.url());
    let conds = vec![
        Condition {
            field: "layer".into(),
            op: crate::ast::CompareOp::Eq,
            value: Value::Integer(5),
        },
        Condition {
            field: "feature".into(),
            op: crate::ast::CompareOp::Eq,
            value: Value::Integer(7),
        },
    ];
    let out = session.remote_delete(&conds).unwrap();
    let joined = out.join("\n");
    assert!(joined.contains("L5 F7"));
}

#[test]
fn remote_delete_errors_when_layer_missing() {
    // No mockito setup needed — request never goes out because
    // the precondition check rejects the call locally.
    let mut server = mockito::Server::new();
    let _stats = server
        .mock("GET", ENDPOINT_STATS)
        .with_status(200)
        .with_body(stats_body())
        .create();

    let session = connect(&server.url());
    let conds = vec![Condition {
        field: "feature".into(),
        op: crate::ast::CompareOp::Eq,
        value: Value::Integer(0),
    }];
    let err = session.remote_delete(&conds).unwrap_err();
    assert!(err.to_string().contains("layer"));
}

#[test]
fn remote_show_patches_empty_when_none_applied() {
    let mut server = mockito::Server::new();
    let _stats = server
        .mock("GET", ENDPOINT_STATS)
        .with_status(200)
        .with_body(stats_body())
        .create();

    let session = connect(&server.url());
    let out = session.remote_show_patches().unwrap();
    let joined = out.join("\n");
    assert!(joined.contains("(no local patches)"));
}

#[test]
fn remote_remove_local_patch_errors_on_unknown_name() {
    let mut server = mockito::Server::new();
    let _stats = server
        .mock("GET", ENDPOINT_STATS)
        .with_status(200)
        .with_body(stats_body())
        .create();

    let mut session = connect(&server.url());
    let err = session.remote_remove_local_patch("nope").unwrap_err();
    assert!(err.to_string().contains("not found"));
}

#[test]
fn remote_apply_local_patch_errors_when_file_missing() {
    let mut server = mockito::Server::new();
    let _stats = server
        .mock("GET", ENDPOINT_STATS)
        .with_status(200)
        .with_body(stats_body())
        .create();

    let mut session = connect(&server.url());
    let err = session
        .remote_apply_local_patch("/tmp/definitely_not_a_real_patch_file.vlp")
        .unwrap_err();
    assert!(err.to_string().contains("patch not found"));
}

#[test]
fn http_helpers_error_when_not_remote() {
    // A session with no `USE` should fail every remote forwarder
    // before any HTTP request is sent.
    let session = Session::new();
    assert!(session.remote_stats().is_err());
    let conds: Vec<Condition> = vec![];
    assert!(session.remote_select(&conds, None).is_err());
}

// ── remote_explain_infer ────────────────────────────────────

#[test]
fn remote_explain_infer_renders_layer_trace() {
    let mut server = mockito::Server::new();
    let _stats = server
        .mock("GET", ENDPOINT_STATS)
        .with_status(200)
        .with_body(stats_body())
        .create();
    let _explain = server
        .mock("POST", ENDPOINT_EXPLAIN_INFER)
        .with_status(200)
        .with_body(
            serde_json::json!({
                "predictions": [
                    {"token": "Paris", "probability": 0.92}
                ],
                "trace": [
                    {
                        "layer": 26,
                        "features": [
                            {
                                "feature": 1,
                                "gate_score": 14.2,
                                "top_token": "Paris",
                                "relation": "capital",
                                "down_top": ["Paris", "France", "Europe"],
                            }
                        ]
                    }
                ],
            })
            .to_string(),
        )
        .create();

    let session = connect(&server.url());
    let out = session
        .remote_explain_infer("test", Some(3), None, false, false)
        .expect("remote_explain_infer");
    let joined = out.join("\n");
    assert!(joined.contains("Inference trace"));
    assert!(joined.contains("Paris"));
    assert!(joined.contains("L26"));
}

#[test]
fn remote_explain_infer_with_attention_renders_compact_format() {
    let mut server = mockito::Server::new();
    let _stats = server
        .mock("GET", ENDPOINT_STATS)
        .with_status(200)
        .with_body(stats_body())
        .create();
    let _explain = server
        .mock("POST", ENDPOINT_EXPLAIN_INFER)
        .with_status(200)
        .with_body(
            serde_json::json!({
                "predictions": [{"token": "X", "probability": 0.5}],
                "trace": [
                    {
                        "layer": 5,
                        "features": [{
                            "feature": 0,
                            "gate_score": 9.0,
                            "top_token": "T",
                            "relation": "rel",
                            "down_top": ["T"],
                        }],
                        "attention": [
                            {"token": "X", "weight": 0.7}
                        ],
                        "lens": {"token": "Y", "probability": 0.3},
                    }
                ],
            })
            .to_string(),
        )
        .create();

    let session = connect(&server.url());
    let out = session
        .remote_explain_infer("p", Some(1), None, false, true)
        .expect("remote_explain_infer WITH ATTENTION");
    let joined = out.join("\n");
    assert!(joined.contains("L"));
}

#[test]
fn remote_explain_infer_with_knn_override_surfaces_pending_note() {
    let mut server = mockito::Server::new();
    let _stats = server
        .mock("GET", ENDPOINT_STATS)
        .with_status(200)
        .with_body(stats_body())
        .create();
    let _explain = server
        .mock("POST", ENDPOINT_EXPLAIN_INFER)
        .with_status(200)
        .with_body(
            serde_json::json!({
                "knn_override": {
                    "token": "Madrid",
                    "cosine": 0.88,
                    "layer": 12,
                },
                "trace": [],
            })
            .to_string(),
        )
        .create();

    let session = connect(&server.url());
    let out = session
        .remote_explain_infer("q", None, None, false, false)
        .expect("remote_explain_infer KNN override");
    let joined = out.join("\n");
    assert!(joined.contains("Pending retrieval override"));
    assert!(joined.contains("Madrid"));
}

#[test]
fn remote_infer_renders_knn_override_note() {
    let mut server = mockito::Server::new();
    let _stats = server
        .mock("GET", ENDPOINT_STATS)
        .with_status(200)
        .with_body(stats_body())
        .create();
    let _infer = server
        .mock("POST", ENDPOINT_INFER)
        .with_status(200)
        .with_body(
            serde_json::json!({
                "predictions": [],
                "knn_override": {
                    "token": "Atlantis",
                    "cosine": 0.91,
                    "layer": 5,
                },
                "latency_ms": 12.0,
            })
            .to_string(),
        )
        .create();

    let session = connect(&server.url());
    let out = session.remote_infer("any", None, false).unwrap();
    let joined = out.join("\n");
    assert!(joined.contains("Atlantis"));
    assert!(joined.contains("note: KNN override"));
}

#[test]
fn remote_describe_brief_mode_renders_compact_output() {
    let mut server = mockito::Server::new();
    let _stats = server
        .mock("GET", ENDPOINT_STATS)
        .with_status(200)
        .with_body(stats_body())
        .create();
    let _describe = server
        .mock("GET", mockito::Matcher::Regex(r"/v1/describe".into()))
        .with_status(200)
        .with_body(
            serde_json::json!({
                "edges": [{"target": "Berlin", "gate_score": 8.0, "layer": 26, "relation": "capital", "source": "probe", "also": []}],
                "latency_ms": 4.0,
            })
            .to_string(),
        )
        .create();

    let session = connect(&server.url());
    let out = session
        .remote_describe("Germany", None, crate::ast::DescribeMode::Brief)
        .expect("remote_describe brief");
    let joined = out.join("\n");
    assert!(joined.contains("Berlin"));
}

#[test]
fn remote_walk_with_layer_range_passes_through() {
    let mut server = mockito::Server::new();
    let _stats = server
        .mock("GET", ENDPOINT_STATS)
        .with_status(200)
        .with_body(stats_body())
        .create();
    let _walk = server
        .mock("GET", mockito::Matcher::Regex(r"/v1/walk".into()))
        .with_status(200)
        .with_body(
            serde_json::json!({
                "hits": [],
                "latency_ms": 1.0,
            })
            .to_string(),
        )
        .create();

    let session = connect(&server.url());
    // LAYERS m-n exercises a different query-string serialisation
    // branch than the no-range variant.
    let range = crate::ast::Range { start: 0, end: 5 };
    let _ = session.remote_walk("p", Some(5), Some(&range)).unwrap();
}

#[test]
fn require_layer_feature_refuses_conditions_the_remote_op_would_drop() {
    // A remote DELETE / UPDATE pins one slot. An entity, relation or score
    // beside the pin would be validated here and then never sent.
    for extra in [
        cond("entity", Value::String("France".into())),
        cond("relation", Value::String("capital".into())),
        cond("score", Value::Integer(1)),
    ] {
        let conds = vec![
            cond("layer", Value::Integer(0)),
            cond("feature", Value::Integer(0)),
            extra,
        ];
        let err = require_layer_feature(&conds, "DELETE")
            .unwrap_err()
            .to_string();
        assert!(err.contains("not evaluated remotely"), "{err}");
    }
}
