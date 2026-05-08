//! HTTP integration tests using the functional tokenizer.
//!
//! These tests cover routes that need real tokenization to return
//! non-empty results: walk, describe (with edges), and insert.
//! The empty BPE tokenizer in the default model() helper produces no
//! token IDs, causing walk to return 400 and describe to return empty edges.
//! model_functional() uses a WordLevel tokenizer with a small vocabulary,
//! so "France" → token 0, which maps to the [1,0,0,0] embedding row and
//! matches gate feature 0 ("Paris").

mod common;
use common::*;

use axum::http::StatusCode;
use larql_server::state::LoadedModel;
use larql_vindex::{ndarray::Array2, PatchedVindex};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

/// Build a model_functional variant with probe labels on (layer=0, feature=0) → "capital".
/// This allows walk and describe to cover the probe label branch.
fn model_functional_with_labels(id: &str) -> Arc<LoadedModel> {
    let mut labels = HashMap::new();
    labels.insert((0usize, 0usize), "capital".to_string());
    Arc::new(LoadedModel {
        id: id.to_string(),
        path: PathBuf::from("/nonexistent"),
        config: test_config(),
        patched: tokio::sync::RwLock::new(PatchedVindex::new(test_index())),
        embeddings: {
            let mut e = Array2::<f32>::zeros((8, 4));
            e[[0, 0]] = 1.0;
            e[[1, 1]] = 1.0;
            e[[2, 2]] = 1.0;
            e[[3, 3]] = 1.0;
            e
        },
        embed_scale: 1.0,
        tokenizer: functional_tokenizer(),
        infer_disabled: true,
        ffn_only: false,
        embed_only: false,
        embed_store: None,
        release_mmap_after_request: false,
        weights: std::sync::OnceLock::new(),
        probe_labels: labels,
        ffn_l2_cache: larql_server::ffn_l2_cache::FfnL2Cache::new(1),
        expert_filter: None,
        unit_filter: None,
        moe_remote: None,
        #[cfg(feature = "metal-experts")]
        metal_backend: std::sync::OnceLock::new(),
        #[cfg(feature = "metal-experts")]
        moe_scratches: std::sync::Mutex::new(std::collections::HashMap::new()),
        #[cfg(feature = "metal-experts")]
        metal_ffn_layer_bufs: std::sync::OnceLock::new(),
    })
}

// ══════════════════════════════════════════════════════════════
// GET /v1/walk — functional tokenizer
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_walk_functional_returns_hits() {
    let app = single_model_router(state(vec![model_functional("test")]));
    let resp = get(app, "/v1/walk?prompt=France").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert!(body["hits"].is_array(), "response must have a 'hits' array");
}

#[tokio::test]
async fn http_walk_functional_hits_contain_paris() {
    let app = single_model_router(state(vec![model_functional("test")]));
    let resp = get(app, "/v1/walk?prompt=France").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    let hits = body["hits"].as_array().unwrap();
    assert!(!hits.is_empty(), "expected at least one hit for 'France'");
    // The top hit should be "Paris" (feature 0, gate [1,0,0,0] matches embed row 0)
    let targets: Vec<&str> = hits.iter().filter_map(|h| h["target"].as_str()).collect();
    assert!(
        targets.contains(&"Paris"),
        "expected 'Paris' in walk hits, got: {:?}",
        targets
    );
}

#[tokio::test]
async fn http_walk_functional_with_layer_range() {
    let app = single_model_router(state(vec![model_functional("test")]));
    let resp = get(app, "/v1/walk?prompt=France&layers=0-0").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert!(body["hits"].is_array());
}

#[tokio::test]
async fn http_walk_functional_with_layer_list() {
    let app = single_model_router(state(vec![model_functional("test")]));
    let resp = get(app, "/v1/walk?prompt=France&layers=0").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert!(body["hits"].is_array());
}

#[tokio::test]
async fn http_walk_functional_with_oob_layer() {
    // Layer 99 doesn't exist (only layer 0 loaded) — hits should be empty
    let app = single_model_router(state(vec![model_functional("test")]));
    let resp = get(app, "/v1/walk?prompt=France&layers=99").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    let hits = body["hits"].as_array().unwrap();
    assert!(
        hits.is_empty(),
        "out-of-range layer should return empty hits"
    );
}

#[tokio::test]
async fn http_walk_functional_multi_model() {
    let app = multi_model_router(state(vec![model_functional("a"), model_functional("b")]));
    let resp = get(app, "/v1/a/walk?prompt=France").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert!(body["hits"].is_array());
}

#[tokio::test]
async fn http_walk_multi_model_not_found() {
    let app = multi_model_router(state(vec![model_functional("a")]));
    let resp = get(app, "/v1/nosuchmodel/walk?prompt=France").await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

// ══════════════════════════════════════════════════════════════
// GET /v1/describe — functional tokenizer (min_score=0 bypasses 5.0 default)
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_describe_functional_returns_edges() {
    let app = single_model_router(state(vec![model_functional("test")]));
    let resp = get(app, "/v1/describe?entity=France&min_score=0").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    let edges = body["edges"].as_array().unwrap();
    assert!(
        !edges.is_empty(),
        "expected non-empty edges for 'France' with min_score=0"
    );
}

#[tokio::test]
async fn http_describe_functional_paris_edge() {
    let app = single_model_router(state(vec![model_functional("test")]));
    let resp = get(app, "/v1/describe?entity=France&min_score=0").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    let edges = body["edges"].as_array().unwrap();
    let targets: Vec<&str> = edges.iter().filter_map(|e| e["target"].as_str()).collect();
    assert!(
        targets.contains(&"Paris"),
        "expected 'Paris' in describe edges, got: {:?}",
        targets
    );
}

#[tokio::test]
async fn http_describe_functional_band_syntax() {
    let app = single_model_router(state(vec![model_functional("test")]));
    let resp = get(app, "/v1/describe?entity=France&band=syntax&min_score=0").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert!(body["edges"].is_array());
}

#[tokio::test]
async fn http_describe_functional_band_output() {
    let app = single_model_router(state(vec![model_functional("test")]));
    let resp = get(app, "/v1/describe?entity=France&band=output&min_score=0").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert!(body["edges"].is_array());
}

#[tokio::test]
async fn http_describe_functional_band_all() {
    let app = single_model_router(state(vec![model_functional("test")]));
    let resp = get(app, "/v1/describe?entity=France&band=all&min_score=0").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert!(body["edges"].is_array());
}

#[tokio::test]
async fn http_describe_functional_verbose() {
    let app = single_model_router(state(vec![model_functional("test")]));
    let resp = get(app, "/v1/describe?entity=France&verbose=true&min_score=0").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    let edges = body["edges"].as_array().unwrap();
    // With verbose=true each edge should have a "count" field
    if !edges.is_empty() {
        assert!(
            edges[0]["count"].as_u64().is_some(),
            "verbose mode should include 'count' field in each edge"
        );
    }
}

#[tokio::test]
async fn http_describe_functional_min_score_filter() {
    // min_score=100 is far above any gate score (max 0.95 in test_index)
    let app = single_model_router(state(vec![model_functional("test")]));
    let resp = get(app, "/v1/describe?entity=France&min_score=100").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    let edges = body["edges"].as_array().unwrap();
    assert!(
        edges.is_empty(),
        "min_score=100 should filter all edges (max score is 0.95)"
    );
}

#[tokio::test]
async fn http_describe_functional_self_ref_filtered() {
    // The describe handler filters out edges where the target == the entity
    // "Paris" as entity: gate feature 0 is "Paris", which should be filtered out
    let app = single_model_router(state(vec![model_functional("test")]));
    let resp = get(app, "/v1/describe?entity=Paris&min_score=0").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    let edges = body["edges"].as_array().unwrap();
    let targets: Vec<&str> = edges.iter().filter_map(|e| e["target"].as_str()).collect();
    assert!(
        !targets.iter().any(|t| t.to_lowercase() == "paris"),
        "self-reference 'Paris' should be filtered from describe results"
    );
}

#[tokio::test]
async fn http_describe_functional_multi_model() {
    let app = multi_model_router(state(vec![model_functional("a"), model_functional("b")]));
    let resp = get(app, "/v1/a/describe?entity=France&min_score=0").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["entity"], "France");
    assert!(body["edges"].is_array());
}

// ══════════════════════════════════════════════════════════════
// POST /v1/insert — functional tokenizer
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_insert_functional_with_tokenizer() {
    // Insert still works (embedding fallback) with the functional tokenizer
    let app = single_model_router(state(vec![model_functional("test")]));
    let resp = post_json(
        app,
        "/v1/insert",
        serde_json::json!({
            "entity": "France",
            "relation": "capital",
            "target": "Paris"
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["entity"], "France");
    assert_eq!(body["target"], "Paris");
    assert!(body["inserted"].as_u64().is_some());
}

// ══════════════════════════════════════════════════════════════
// GET /v1/walk — prompt field in response
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_walk_functional_response_has_prompt_field() {
    let app = single_model_router(state(vec![model_functional("test")]));
    let resp = get(app, "/v1/walk?prompt=France").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["prompt"], "France");
    assert!(body["latency_ms"].as_f64().is_some());
}

// ══════════════════════════════════════════════════════════════
// GET /v1/walk — probe labels branch (walk.rs line 78)
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_walk_with_probe_label_includes_relation_field() {
    // model_functional_with_labels puts "capital" label on (layer=0, feature=0).
    // Walk for "France" → token 0 → embedding [1,0,0,0] → matches feature 0 (Paris).
    // The probe label branch should set hits[0]["relation"] = "capital".
    let app = single_model_router(state(vec![model_functional_with_labels("test")]));
    let resp = get(app, "/v1/walk?prompt=France").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    let hits = body["hits"].as_array().unwrap();
    assert!(!hits.is_empty(), "expected at least one hit");
    // The top hit should have relation = "capital" from probe labels.
    let relations: Vec<Option<&str>> = hits.iter().map(|h| h["relation"].as_str()).collect();
    assert!(
        relations.contains(&Some("capital")),
        "expected 'relation' = 'capital' in a walk hit (probe label branch), got hits: {:?}",
        hits
    );
}

// ══════════════════════════════════════════════════════════════
// GET /v1/describe — probe labels branch (describe.rs lines 163-164)
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_describe_with_probe_label_includes_relation_and_source() {
    // Same: probe label on (0,0) → "capital". Describe for France should produce
    // an edge for Paris with relation="capital" and source="probe".
    let app = single_model_router(state(vec![model_functional_with_labels("test")]));
    let resp = get(app, "/v1/describe?entity=France&min_score=0").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    let edges = body["edges"].as_array().unwrap();
    let edge_with_label = edges.iter().find(|e| e["relation"].as_str().is_some());
    assert!(
        edge_with_label.is_some(),
        "expected at least one edge with 'relation' field (probe label branch)"
    );
    if let Some(edge) = edge_with_label {
        assert_eq!(edge["relation"], "capital");
        assert_eq!(edge["source"], "probe");
    }
}

// ══════════════════════════════════════════════════════════════
// GET /v1/describe — multi-token entity (describe.rs lines 61-66)
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_describe_multi_token_entity_averages_embeddings() {
    // "France capital" tokenizes to [0, 2] → average of embed rows 0 and 2.
    // Row 0 = [1,0,0,0], Row 2 = [0,0,1,0] → avg = [0.5,0,0.5,0].
    // This exercises the multi-token averaging branch in describe_entity.
    let app = single_model_router(state(vec![model_functional("test")]));
    // URL-encode "France capital" as "France%20capital" to send as entity param.
    let resp = get(
        app,
        "/v1/describe?entity=France%20capital&min_score=0&band=all",
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["entity"], "France capital");
    assert!(body["edges"].is_array());
    // With the averaged query the walk should still return some hits.
}

// ══════════════════════════════════════════════════════════════
// POST /v1/walk-ffn — features-only mode (walk_ffn.rs)
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_walk_ffn_features_single_layer_returns_200() {
    // features-only mode (full_output=false, default) — no model weights needed.
    let app = single_model_router(state(vec![model_functional("test")]));
    let resp = post_json(
        app,
        "/v1/walk-ffn",
        serde_json::json!({
            "layer": 0,
            "residual": [1.0, 0.0, 0.0, 0.0]
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    // features-only single layer: response has "layer", "features", "scores"
    assert!(body["features"].is_array(), "expected 'features' array");
    assert!(body["scores"].is_array(), "expected 'scores' array");
    assert_eq!(body["layer"], 0);
}

#[tokio::test]
async fn http_walk_ffn_features_single_layer_top_hit_is_feature_0() {
    // "France" embedding [1,0,0,0] should score highest against gate feature 0 ("Paris")
    let app = single_model_router(state(vec![model_functional("test")]));
    let resp = post_json(
        app,
        "/v1/walk-ffn",
        serde_json::json!({
            "layer": 0,
            "residual": [1.0, 0.0, 0.0, 0.0],
            "top_k": 3
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    let features = body["features"].as_array().unwrap();
    assert!(!features.is_empty());
    assert_eq!(features[0], 0, "feature 0 should be top hit for [1,0,0,0]");
}

#[tokio::test]
async fn http_walk_ffn_features_layers_array_single_returns_layer_format() {
    // When layers=[0] (exactly one), the handler returns single-layer format
    // (top-level "features"/"scores" keys, no "results" wrapper).
    let app = single_model_router(state(vec![model_functional("test")]));
    let resp = post_json(
        app,
        "/v1/walk-ffn",
        serde_json::json!({
            "layers": [0],
            "residual": [1.0, 0.0, 0.0, 0.0]
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["layer"], 0);
    assert!(body["features"].is_array());
    assert!(body["scores"].is_array());
}

#[tokio::test]
async fn http_walk_ffn_missing_layer_returns_400() {
    // Neither layer nor layers → bad request
    let app = single_model_router(state(vec![model_functional("test")]));
    let resp = post_json(
        app,
        "/v1/walk-ffn",
        serde_json::json!({
            "residual": [1.0, 0.0, 0.0, 0.0]
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn http_walk_ffn_wrong_residual_size_returns_400() {
    // hidden=4 but residual has 3 elements → bad request
    let app = single_model_router(state(vec![model_functional("test")]));
    let resp = post_json(
        app,
        "/v1/walk-ffn",
        serde_json::json!({
            "layer": 0,
            "residual": [1.0, 0.0, 0.0]  // 3 elements, hidden=4
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn http_walk_ffn_multi_model_not_found() {
    let app = multi_model_router(state(vec![model_functional("a")]));
    let resp = post_json(
        app,
        "/v1/nosuchmodel/walk-ffn",
        serde_json::json!({
            "layer": 0,
            "residual": [1.0, 0.0, 0.0, 0.0]
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn http_walk_ffn_binary_without_full_output_returns_400() {
    // Binary wire format requires full_output=true
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt as _;
    // Binary content-type for the walk-ffn wire format.
    let binary_ct = "application/x-larql-ffn";
    // Build a minimal binary request body: layer=0, seq_len=1, flags=0 (full_output=false), top_k=8, residual=[1,0,0,0]
    let mut body = Vec::new();
    body.extend_from_slice(&0u32.to_le_bytes()); // layer
    body.extend_from_slice(&1u32.to_le_bytes()); // seq_len
    body.extend_from_slice(&0u32.to_le_bytes()); // flags (full_output=0)
    body.extend_from_slice(&8u32.to_le_bytes()); // top_k
    body.extend_from_slice(&1.0f32.to_le_bytes()); // residual[0]
    body.extend_from_slice(&0.0f32.to_le_bytes()); // residual[1]
    body.extend_from_slice(&0.0f32.to_le_bytes()); // residual[2]
    body.extend_from_slice(&0.0f32.to_le_bytes()); // residual[3]

    let resp = single_model_router(state(vec![model_functional("test")]))
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/walk-ffn")
                .header("content-type", binary_ct)
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn http_walk_ffn_latency_ms_in_response() {
    let app = single_model_router(state(vec![model_functional("test")]));
    let resp = post_json(
        app,
        "/v1/walk-ffn",
        serde_json::json!({
            "layer": 0,
            "residual": [1.0, 0.0, 0.0, 0.0]
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert!(body["latency_ms"].as_f64().is_some());
}

// ══════════════════════════════════════════════════════════════
// GET /v1/relations — multi-model handler (relations.rs lines 186-197)
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_relations_multi_model_returns_200() {
    let app = multi_model_router(state(vec![model_functional("a"), model_functional("b")]));
    let resp = get(app, "/v1/a/relations").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert!(body["relations"].is_array());
    assert!(body["probe_relations"].is_array());
}

#[tokio::test]
async fn http_relations_multi_model_not_found() {
    let app = multi_model_router(state(vec![model_functional("a")]));
    let resp = get(app, "/v1/nosuchmodel/relations").await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

// ══════════════════════════════════════════════════════════════
// GET /v1/describe — describe cache hit with etag (describe.rs)
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_describe_functional_cache_hit_same_etag() {
    // Two requests to same entity → same etag (cache hit).
    let st = state_with_cache(vec![model_functional("test")], 100);
    let app1 = single_model_router(st.clone());
    let r1 = get(app1, "/v1/describe?entity=France&min_score=0").await;
    assert_eq!(r1.status(), StatusCode::OK);
    let etag1 = r1.headers()["etag"].to_str().unwrap().to_string();

    let app2 = single_model_router(st.clone());
    let r2 = get(app2, "/v1/describe?entity=France&min_score=0").await;
    assert_eq!(r2.status(), StatusCode::OK);
    let etag2 = r2.headers()["etag"].to_str().unwrap().to_string();

    assert_eq!(etag1, etag2, "cache hit should produce same etag");
}

// ══════════════════════════════════════════════════════════════
// POST /v1/insert — multi-model handler (insert.rs lines 242-249)
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_insert_multi_model_returns_200() {
    let app = multi_model_router(state(vec![model_functional("a"), model_functional("b")]));
    let resp = post_json(
        app,
        "/v1/a/insert",
        serde_json::json!({
            "entity": "France",
            "relation": "capital",
            "target": "Paris"
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["entity"], "France");
    assert_eq!(body["target"], "Paris");
}

// ══════════════════════════════════════════════════════════════
// GET /v1/patches — multi-model handler (patches.rs lines 212-219)
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_patches_list_multi_model_returns_200() {
    let app = multi_model_router(state(vec![model_functional("a"), model_functional("b")]));
    let resp = get(app, "/v1/a/patches").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert!(body["patches"].is_array());
}

#[tokio::test]
async fn http_patches_list_multi_model_not_found() {
    let app = multi_model_router(state(vec![model_functional("a")]));
    let resp = get(app, "/v1/nosuchmodel/patches").await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

// ══════════════════════════════════════════════════════════════
// DELETE /v1/patches — multi-model handler (patches.rs lines 267-274)
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_patches_delete_multi_model_not_found() {
    // Deleting a non-existent patch from multi-model → 404.
    let app = multi_model_router(state(vec![model_functional("a")]));
    let resp = delete(app, "/v1/a/patches/nonexistent").await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn http_patches_delete_multi_model_applies_and_removes() {
    // Apply a patch to model "a", then remove it via multi-model path.
    let st = state(vec![model_functional("a"), model_functional("b")]);
    let app1 = multi_model_router(st.clone());
    let apply_resp = post_json(app1, "/v1/a/patches/apply", inline_delete_patch("mp-patch")).await;
    assert_eq!(apply_resp.status(), StatusCode::OK);

    let app2 = multi_model_router(st.clone());
    let del_resp = delete(app2, "/v1/a/patches/mp-patch").await;
    assert_eq!(del_resp.status(), StatusCode::OK);
    let body = body_json(del_resp.into_body()).await;
    assert_eq!(body["removed"], "mp-patch");
}

// ══════════════════════════════════════════════════════════════
// POST /v1/patches/apply — enrich_patch_ops with functional tokenizer
// (covers patches.rs lines 64-112: enrich_patch_ops function)
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_patches_apply_insert_op_enrich_with_functional_tokenizer() {
    // Send an INSERT patch operation without a gate_vector_b64.
    // The enrich_patch_ops function will synthesize one from the entity embedding.
    // This exercises the branch in enrich_patch_ops that tokenizes the entity.
    // Use JSON to avoid needing to know exact PatchOp field layout.
    let patch_json = serde_json::json!({
        "patch": {
            "version": 1,
            "base_model": "test",
            "base_checksum": null,
            "created_at": "2026-04-26",
            "description": "enrich-test",
            "author": null,
            "tags": [],
            "operations": [
                {
                    "op": "insert",
                    "layer": 0,
                    "feature": 0,
                    "entity": "France",
                    "relation": "capital",
                    "target": "Paris",
                    "gate_vector_b64": null
                }
            ]
        }
    });

    let app = single_model_router(state(vec![model_functional("test")]));
    let resp = post_json(app, "/v1/patches/apply", patch_json).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert!(body["applied"].as_str().is_some());
    assert!(body["active_patches"].as_u64().is_some());
}

// ══════════════════════════════════════════════════════════════
// DELETE /v1/patches — session-scoped remove (patches.rs lines 228-237)
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_patches_session_remove_returns_session_field() {
    let st = state(vec![model_functional("test")]);
    let m = st.models[0].clone();
    // Pre-create the session to avoid blocking_read in async context.
    st.sessions.get_or_create("rm-session", &m).await;

    // Apply a session-scoped patch.
    let app1 = single_model_router(st.clone());
    post_json_h(
        app1,
        "/v1/patches/apply",
        inline_delete_patch("rm-patch"),
        ("x-session-id", "rm-session"),
    )
    .await;

    // Remove it via session using get_h helper which sets a header.
    // But delete_h doesn't exist, so build request manually.
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt as _;
    let del_resp = single_model_router(st.clone())
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri("/v1/patches/rm-patch")
                .header("x-session-id", "rm-session")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(del_resp.status(), StatusCode::OK);
    let body = body_json(del_resp.into_body()).await;
    assert_eq!(body["session"], "rm-session");
    assert_eq!(body["removed"], "rm-patch");
}
