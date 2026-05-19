//! Coverage push for `openapi.rs` (was 0%, target ≥ 90%).
//!
//! `openapi.rs` is mostly schema declarations; the runtime surface
//! is just `swagger_router()` + `ApiDoc::openapi()`. Two tests
//! exercise both and confirm the served JSON validates.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

#[tokio::test]
async fn swagger_router_serves_openapi_json() {
    let app = larql_server::openapi::swagger_router();
    let resp = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/openapi.json")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        v["openapi"].as_str().map(|s| s.starts_with("3.")),
        Some(true)
    );
    assert!(
        v["paths"].is_object(),
        "OpenAPI doc must declare paths; got {v:?}"
    );
}

#[tokio::test]
async fn swagger_router_serves_swagger_ui_index() {
    let app = larql_server::openapi::swagger_router();
    let resp = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/swagger-ui/")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    // utoipa-swagger-ui serves the index page at /swagger-ui or
    // redirects from / to /index.html depending on version. Either
    // is fine — what matters is the router served something.
    assert!(
        resp.status().is_success() || resp.status().is_redirection(),
        "swagger-ui should respond 2xx/3xx; got {:?}",
        resp.status()
    );
}

#[test]
fn api_doc_compiles_and_lists_paths() {
    use utoipa::OpenApi;
    let doc = larql_server::openapi::ApiDoc::openapi();
    let json = serde_json::to_value(&doc).unwrap();
    assert!(
        json["paths"].is_object(),
        "ApiDoc must produce a paths object"
    );
    // At least one of our handlers should be in the spec.
    let paths = json["paths"].as_object().unwrap();
    assert!(
        !paths.is_empty(),
        "ApiDoc must have at least one path declared"
    );
}
