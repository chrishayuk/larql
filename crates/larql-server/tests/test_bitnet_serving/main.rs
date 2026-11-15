//! BitNet 1.58 `--keep-quant` serving against a synthetic I2_S container
//! (`common::synthetic_bitnet_vindex`): the native-ternary arms of
//! `/v1/infer`, the chat SSE stream and the `/v1/completions` stream, the
//! dense-only mode contract, and the ternary-aware memcheck estimate.

#[path = "../common/mod.rs"]
mod common;

mod chat_stream;
mod completions_stream;
mod infer;
mod memcheck;

use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use common::synthetic_bitnet_vindex::{self as bitnet, BitnetBuild, SyntheticBitnetVindex};
use tower::ServiceExt;

/// A router serving one synthetic BitNet container, plus the fixture
/// that owns its directory (keep it alive until the body is drained).
pub fn bitnet_app(kind: BitnetBuild) -> (axum::Router, SyntheticBitnetVindex) {
    let (model, fixture) = bitnet::loaded_model(kind);
    let state = common::state(vec![model]);
    (larql_server::routes::single_model_router(state), fixture)
}

/// Like [`bitnet_app`], with the ternary artifacts broken so the lazy
/// BitNet load fails on the first request.
pub fn broken_bitnet_app(kind: BitnetBuild) -> (axum::Router, SyntheticBitnetVindex) {
    let (app, fixture) = bitnet_app(kind);
    bitnet::break_ternary_artifacts(&fixture);
    (app, fixture)
}

/// POST `body` to `uri` and drain the response; `(status, body text)`.
pub async fn post(app: &axum::Router, uri: &str, body: serde_json::Value) -> (StatusCode, String) {
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(uri)
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

/// The JSON payloads of an SSE body's `data:` lines, `[DONE]` excluded.
pub fn sse_events(body: &str) -> Vec<serde_json::Value> {
    body.lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|d| *d != "[DONE]")
        .map(|d| serde_json::from_str(d).unwrap_or_else(|e| panic!("bad SSE json {d:?}: {e}")))
        .collect()
}

/// The `finish_reason` of the last SSE chunk that carries one.
pub fn finish_reason(events: &[serde_json::Value]) -> Option<String> {
    events.iter().rev().find_map(|e| {
        e["choices"][0]["finish_reason"]
            .as_str()
            .map(str::to_string)
    })
}

/// A prompt of in-vocabulary fixture tokens.
pub fn token_prompt(ids: &[usize]) -> String {
    ids.iter()
        .map(|&i| bitnet::token_text(i))
        .collect::<Vec<_>>()
        .join(" ")
}
