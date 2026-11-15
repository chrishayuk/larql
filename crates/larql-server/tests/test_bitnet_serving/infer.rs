//! `POST /v1/infer` on BitNet containers: the native-ternary dense and
//! walk arms, and the dense-only mode contract (coerce an omitted mode,
//! refuse an explicit `walk` / `compare`).

use axum::http::StatusCode;

use super::{bitnet_app, broken_bitnet_app, post, token_prompt, BitnetBuild};

const INFER: &str = "/v1/infer";
const TOP: usize = 3;

fn infer_body(mode: Option<&str>) -> serde_json::Value {
    let mut body = serde_json::json!({"prompt": token_prompt(&[1, 2, 3]), "top": TOP});
    if let Some(m) = mode {
        body["mode"] = serde_json::json!(m);
    }
    body
}

async fn infer(kind: BitnetBuild, mode: Option<&str>) -> (StatusCode, serde_json::Value) {
    let (app, _fixture) = bitnet_app(kind);
    let (status, body) = post(&app, INFER, infer_body(mode)).await;
    let v = serde_json::from_str(&body).unwrap_or_else(|_| serde_json::json!({"raw": body}));
    (status, v)
}

fn assert_predictions(v: &serde_json::Value, key: &str) {
    let preds = v[key]
        .as_array()
        .unwrap_or_else(|| panic!("{key} array: {v}"));
    assert_eq!(preds.len(), TOP, "{v}");
    for p in preds {
        assert!(p["token"].is_string(), "{p}");
        let prob = p["probability"].as_f64().expect("probability");
        assert!((0.0..=1.0).contains(&prob), "{p}");
    }
}

#[tokio::test]
async fn dense_only_omitted_mode_is_served_by_the_ternary_dense_path() {
    let (status, v) = infer(BitnetBuild::DenseOnly, None).await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_eq!(v["mode"], "bitnet", "{v}");
    assert_predictions(&v, "predictions");
    assert!(v["latency_ms"].is_number(), "{v}");
}

#[tokio::test]
async fn dense_only_explicit_dense_mode_is_served() {
    let (status, v) = infer(BitnetBuild::DenseOnly, Some("dense")).await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_eq!(v["mode"], "bitnet", "{v}");
    assert_predictions(&v, "predictions");
}

#[tokio::test]
async fn dense_only_explicit_walk_is_refused_with_the_reason() {
    let (status, v) = infer(BitnetBuild::DenseOnly, Some("walk")).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{v}");
    let msg = v["error"].as_str().unwrap_or_default();
    assert!(msg.contains("--dense-only") && msg.contains("walk"), "{v}");
}

#[tokio::test]
async fn dense_only_explicit_compare_is_refused_with_the_reason() {
    let (status, v) = infer(BitnetBuild::DenseOnly, Some("compare")).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{v}");
    let msg = v["error"].as_str().unwrap_or_default();
    assert!(
        msg.contains("--dense-only") && msg.contains("compare"),
        "{v}"
    );
}

#[tokio::test]
async fn gated_omitted_mode_runs_the_ternary_walk() {
    let (status, v) = infer(BitnetBuild::WithGates, None).await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_eq!(v["mode"], "walk", "{v}");
    assert_predictions(&v, "predictions");
}

#[tokio::test]
async fn gated_compare_returns_both_ternary_arms() {
    let (status, v) = infer(BitnetBuild::WithGates, Some("compare")).await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_predictions(&v, "walk");
    assert_predictions(&v, "dense");
    assert!(v["walk_ms"].is_number() && v["dense_ms"].is_number(), "{v}");
    assert!(
        v.get("mode").is_none(),
        "compare carries no single mode: {v}"
    );
}

#[tokio::test]
async fn empty_prompt_is_a_bad_request() {
    let (app, _fixture) = bitnet_app(BitnetBuild::DenseOnly);
    let (status, body) = post(&app, INFER, serde_json::json!({"prompt": ""})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(body.contains("empty prompt"), "{body}");
}

#[tokio::test]
async fn failed_ternary_load_is_service_unavailable() {
    let (app, _fixture) = broken_bitnet_app(BitnetBuild::DenseOnly);
    let (status, body) = post(&app, INFER, infer_body(None)).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{body}");
    assert!(body.contains("bitnet"), "{body}");
}
