//! HTTP-level coverage for `GET /v1/shard/{model_id}/{range}` — the donor
//! endpoint used by the Mode B shard handoff.

mod common;

use std::path::PathBuf;
use std::sync::Arc;

use axum::body::{to_bytes, Body};
use axum::http::{Request, StatusCode};
use larql_server::ffn_l2_cache::FfnL2Cache;
use larql_server::state::LoadedModel;
use larql_vindex::{ndarray::Array2, PatchedVindex};
use tempfile::TempDir;
use tower::ServiceExt;

use common::{empty_tokenizer, single_model_router, state, test_config, test_index};

/// Build a `LoadedModel` whose `path` points at a real on-disk directory so
/// the shard route can tar-stream its contents.
fn model_with_path(id: &str, path: PathBuf) -> Arc<LoadedModel> {
    Arc::new(LoadedModel {
        id: id.to_string(),
        path,
        config: test_config(),
        patched: std::sync::Arc::new(tokio::sync::RwLock::new(PatchedVindex::new(test_index()))),
        embeddings: {
            let mut e = Array2::<f32>::zeros((8, 4));
            e[[0, 0]] = 1.0;
            e
        },
        embed_scale: 1.0,
        tokenizer: empty_tokenizer(),
        infer_disabled: true,
        ffn_only: false,
        embed_only: false,
        embed_store: None,
        release_mmap_after_request: false,
        weights: std::sync::OnceLock::new(),
        probe_labels: std::collections::HashMap::new(),
        ffn_l2_cache: FfnL2Cache::new(1),
        layer_latency_tracker: std::sync::Arc::new(
            larql_server::metrics::LayerLatencyTracker::new(),
        ),
        requests_in_flight: std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0)),
        requests_total: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
        expert_filter: None,
        unit_filter: None,
        moe_remote: None,
        #[cfg(all(feature = "metal-experts", target_os = "macos"))]
        metal_backend: std::sync::OnceLock::new(),
        #[cfg(all(feature = "metal-experts", target_os = "macos"))]
        moe_scratches: std::sync::Mutex::new(std::collections::HashMap::new()),
        #[cfg(all(feature = "metal-experts", target_os = "macos"))]
        metal_ffn_layer_bufs: std::sync::OnceLock::new(),
    })
}

async fn get(app: axum::Router, path: &str) -> axum::http::Response<Body> {
    app.oneshot(
        Request::builder()
            .method("GET")
            .uri(path)
            .body(Body::empty())
            .unwrap(),
    )
    .await
    .unwrap()
}

#[tokio::test]
async fn shard_endpoint_returns_400_for_malformed_range() {
    let tmp = TempDir::new().unwrap();
    let model = model_with_path("m", tmp.path().to_path_buf());
    let state = state(vec![model]);
    let app = single_model_router(state);

    let resp = get(app, "/v1/shard/m/not-a-range").await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let text = String::from_utf8_lossy(&body);
    assert!(text.contains("invalid layer range"), "got: {text}");
}

#[tokio::test]
async fn shard_endpoint_returns_400_when_start_after_end() {
    let tmp = TempDir::new().unwrap();
    let model = model_with_path("m", tmp.path().to_path_buf());
    let state = state(vec![model]);
    let app = single_model_router(state);

    let resp = get(app, "/v1/shard/m/9-3").await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let text = String::from_utf8_lossy(&body);
    assert!(text.contains("start (9) must be <= end (3)"), "got: {text}");
}

#[tokio::test]
async fn shard_endpoint_returns_404_for_unknown_model() {
    let tmp = TempDir::new().unwrap();
    // Two models loaded so state.model(Some("missing")) is unambiguous —
    // single-model mode never reaches the None branch because there's only
    // one candidate; we need real strict id matching.
    let a = model_with_path("model-a", tmp.path().to_path_buf());
    let b = model_with_path("model-b", tmp.path().to_path_buf());
    let state = state(vec![a, b]);
    let app = single_model_router(state);

    let resp = get(app, "/v1/shard/not-a-real-model/0-4").await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let text = String::from_utf8_lossy(&body);
    assert!(text.contains("not loaded"), "got: {text}");
}

#[tokio::test]
async fn shard_endpoint_returns_500_when_path_is_not_a_directory() {
    let tmp = TempDir::new().unwrap();
    let file_path = tmp.path().join("not-a-dir");
    std::fs::write(&file_path, b"oops").unwrap();
    let model = model_with_path("m", file_path);
    let state = state(vec![model]);
    let app = single_model_router(state);

    let resp = get(app, "/v1/shard/m/0-4").await;
    assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
    let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let text = String::from_utf8_lossy(&body);
    assert!(
        text.contains("not a directory"),
        "expected error message, got: {text}"
    );
}

#[tokio::test]
async fn shard_endpoint_streams_tar_of_directory() {
    let tmp = TempDir::new().unwrap();
    // Populate a vindex-like directory.
    std::fs::write(tmp.path().join("index.json"), b"{\"hello\":\"world\"}").unwrap();
    std::fs::create_dir_all(tmp.path().join("layer-0")).unwrap();
    std::fs::write(tmp.path().join("layer-0/data.bin"), [1u8, 2, 3, 4]).unwrap();

    let model = model_with_path("m", tmp.path().to_path_buf());
    let app_state = state(vec![model]);
    let app = single_model_router(app_state);

    let resp = get(app, "/v1/shard/m/0-4").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let ct = resp
        .headers()
        .get(axum::http::header::CONTENT_TYPE)
        .map(|v| v.to_str().unwrap().to_string());
    assert_eq!(ct.as_deref(), Some("application/x-tar"));

    let body = to_bytes(resp.into_body(), 16 * 1024 * 1024).await.unwrap();
    assert!(!body.is_empty(), "tar body must not be empty");

    // Verify the body is a valid tar containing the files we wrote.
    let unpack_dir = TempDir::new().unwrap();
    let cursor = std::io::Cursor::new(body.as_ref());
    let mut archive = tar::Archive::new(cursor);
    archive.unpack(unpack_dir.path()).unwrap();

    let index = std::fs::read(unpack_dir.path().join("index.json")).unwrap();
    assert_eq!(index, b"{\"hello\":\"world\"}");
    let data = std::fs::read(unpack_dir.path().join("layer-0/data.bin")).unwrap();
    assert_eq!(data, &[1u8, 2, 3, 4]);
}
