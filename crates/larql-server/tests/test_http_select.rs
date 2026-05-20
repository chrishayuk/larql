#![cfg(not(target_arch = "wasm32"))]
//! HTTP integration tests: select (all variants), relations (single + multi),
//! session-scoped describe/walk/select.

mod common;
use common::*;

use axum::http::StatusCode;
use std::collections::HashMap;

// ══════════════════════════════════════════════════════════════
// POST /v1/select
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_select_no_filter_returns_all_features() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = post_json(app, "/v1/select", serde_json::json!({})).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["total"], 3);
    let edges = body["edges"].as_array().unwrap();
    assert_eq!(edges.len(), 3);
    assert!(body["latency_ms"].as_f64().is_some());
}

#[tokio::test]
async fn http_select_layer_filter_returns_correct_features() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = post_json(app, "/v1/select", serde_json::json!({"layer": 0})).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["total"], 3); // 3 features at layer 0
    let edges = body["edges"].as_array().unwrap();
    for edge in edges {
        assert_eq!(edge["layer"], 0);
    }
}

#[tokio::test]
async fn http_select_entity_filter() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = post_json(app, "/v1/select", serde_json::json!({"entity": "Par"})).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    let edges = body["edges"].as_array().unwrap();
    // Only "Paris" matches "Par" (case-insensitive substring).
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0]["target"].as_str().unwrap().trim(), "Paris");
}

#[tokio::test]
async fn http_select_min_confidence_filter() {
    let app = single_model_router(state(vec![model("test")]));
    // Only Paris (0.95) and French (0.88) pass min_confidence=0.85.
    let resp = post_json(
        app,
        "/v1/select",
        serde_json::json!({"min_confidence": 0.85}),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    let edges = body["edges"].as_array().unwrap();
    assert_eq!(edges.len(), 2);
    for edge in edges {
        assert!(edge["c_score"].as_f64().unwrap() >= 0.85);
    }
}

#[tokio::test]
async fn http_select_limit_truncates_results() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = post_json(app, "/v1/select", serde_json::json!({"limit": 2})).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    let edges = body["edges"].as_array().unwrap();
    assert_eq!(edges.len(), 2);
    assert_eq!(body["total"], 3); // total still 3, but truncated to 2
}

#[tokio::test]
async fn http_select_order_asc_returns_lowest_confidence_first() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = post_json(
        app,
        "/v1/select",
        serde_json::json!({"order_by": "confidence", "order": "asc"}),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    let edges = body["edges"].as_array().unwrap();
    let scores: Vec<f64> = edges
        .iter()
        .map(|e| e["c_score"].as_f64().unwrap())
        .collect();
    // Should be ascending.
    for i in 1..scores.len() {
        assert!(
            scores[i] >= scores[i - 1],
            "expected ascending: {:?}",
            scores
        );
    }
}

#[tokio::test]
async fn http_select_order_desc_returns_highest_confidence_first() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = post_json(
        app,
        "/v1/select",
        serde_json::json!({"order_by": "confidence", "order": "desc"}),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    let edges = body["edges"].as_array().unwrap();
    let scores: Vec<f64> = edges
        .iter()
        .map(|e| e["c_score"].as_f64().unwrap())
        .collect();
    for i in 1..scores.len() {
        assert!(
            scores[i] <= scores[i - 1],
            "expected descending: {:?}",
            scores
        );
    }
}

#[tokio::test]
async fn http_select_relation_filter_returns_labelled_features() {
    let mut labels = HashMap::new();
    labels.insert((0usize, 0usize), "capital".to_string());
    labels.insert((0usize, 1usize), "language".to_string());
    let m = ModelBuilder::new("test").with_labels(labels).build();
    let app = single_model_router(state(vec![m]));
    let resp = post_json(
        app,
        "/v1/select",
        serde_json::json!({"relation": "capital"}),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    let edges = body["edges"].as_array().unwrap();
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0]["relation"], "capital");
    assert_eq!(edges[0]["target"].as_str().unwrap().trim(), "Paris");
}

#[tokio::test]
async fn http_select_order_by_layer_asc() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = post_json(
        app,
        "/v1/select",
        serde_json::json!({"order_by": "layer", "order": "asc"}),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    // All features are at layer 0 in our 1-layer test index; ordering should succeed.
    assert!(body["edges"].is_array());
}

// ══════════════════════════════════════════════════════════════
// Multi-model select
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_multi_select_all_features() {
    let app = multi_model_router(state(vec![model("m1"), model("m2")]));
    let resp = post_json(app, "/v1/m1/select", serde_json::json!({})).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["total"], 3);
}

// ══════════════════════════════════════════════════════════════
// GET /v1/relations
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_relations_returns_json_structure() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = get(app, "/v1/relations").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert!(body["relations"].is_array());
    assert!(body["probe_relations"].is_array());
    assert!(body["total"].as_u64().is_some());
    assert!(body["probe_count"].as_u64().is_some());
    assert!(body["latency_ms"].as_f64().is_some());
}

#[tokio::test]
async fn http_relations_probe_count_reflects_labels() {
    let mut labels = HashMap::new();
    labels.insert((0usize, 0usize), "capital".to_string());
    labels.insert((0usize, 1usize), "language".to_string());
    let m = ModelBuilder::new("test").with_labels(labels).build();
    let app = single_model_router(state(vec![m]));
    let resp = get(app, "/v1/relations").await;
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["probe_count"], 2);
    let probe_rels = body["probe_relations"].as_array().unwrap();
    let names: Vec<&str> = probe_rels
        .iter()
        .map(|r| r["name"].as_str().unwrap())
        .collect();
    assert!(names.contains(&"capital"));
    assert!(names.contains(&"language"));
}

// ══════════════════════════════════════════════════════════════
// Session-scoped describe/walk/select (multi-model)
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_multi_describe_returns_entity() {
    let app = multi_model_router(state(vec![model("mymodel")]));
    let resp = get(app, "/v1/mymodel/describe?entity=France").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["entity"], "France");
}
