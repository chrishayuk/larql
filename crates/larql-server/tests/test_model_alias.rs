//! Model aliases (`--model-alias`): a served model answers to the name it
//! was served under as well as the id its container declares.
//!
//! Matching is exact and case-sensitive for both; the id always wins over
//! an alias of the same spelling; `/v1/models` keeps `id` as the declared
//! id and lists the alias beside it.

mod common;

use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use larql_server::state::{ModelSet, ServedModel};
use tower::ServiceExt;

const DECLARED_ID: &str = "Qwen3-0.6B";
const SERVED_NAME: &str = "qwen3-0.6b";

fn aliased_state() -> std::sync::Arc<larql_server::state::AppState> {
    let st = common::state(vec![common::model(DECLARED_ID)]);
    st.model_set
        .write()
        .unwrap()
        .alias_sole_model(SERVED_NAME)
        .unwrap();
    st
}

fn set_of(ids: &[&str]) -> ModelSet {
    ModelSet {
        models: ids.iter().map(|id| common::model(id)).collect(),
        v3_models: Vec::new(),
        aliases: Default::default(),
    }
}

// ── Resolution ───────────────────────────────────────────────────────────────

#[test]
fn the_alias_and_the_id_both_resolve_to_the_model() {
    let st = aliased_state();
    for name in [SERVED_NAME, DECLARED_ID] {
        match st.served(Some(name)) {
            Some(ServedModel::V2(m)) => assert_eq!(m.id, DECLARED_ID),
            _ => panic!("'{name}' must resolve"),
        }
        assert_eq!(st.model(Some(name)).unwrap().id, DECLARED_ID);
    }
    // No `model` field still means the only model.
    assert!(st.served(None).is_some());
}

#[test]
fn matching_is_case_sensitive() {
    let st = aliased_state();
    assert!(st.served(Some("QWEN3-0.6B")).is_none());
    assert!(st.served(Some("qwen3-0.6B")).is_none());
}

#[test]
fn an_id_wins_over_an_alias_of_the_same_spelling() {
    let mut set = set_of(&["a", "b"]);
    set.aliases.insert("b".into(), "a".into());
    assert_eq!(set.canonical_id("b"), "b");
    assert_eq!(set.canonical_id("missing"), "missing");
}

// ── Binding rules ────────────────────────────────────────────────────────────

#[test]
fn an_alias_may_not_shadow_an_id_or_repoint_an_alias() {
    let mut set = set_of(&["a", "b"]);
    assert!(set
        .add_alias("b", "a")
        .unwrap_err()
        .contains("another model's id"));
    set.add_alias("x", "a").unwrap();
    set.add_alias("x", "a").unwrap(); // idempotent
    assert!(set
        .add_alias("x", "b")
        .unwrap_err()
        .contains("already names 'a'"));
    assert!(set
        .add_alias("y", "nope")
        .unwrap_err()
        .contains("no such model"));
    set.add_alias("a", "a").unwrap(); // its own id: a no-op
    assert_eq!(set.aliases_of("a"), ["x"]);
    assert!(set.aliases_of("b").is_empty());
}

#[test]
fn a_sole_model_alias_needs_exactly_one_model() {
    let err = set_of(&[]).alias_sole_model("x").unwrap_err();
    assert!(err.contains("0 are bound"), "{err}");
    let err = set_of(&["a", "b"]).alias_sole_model("x").unwrap_err();
    assert!(err.contains("2 are bound"), "{err}");
}

// ── HTTP ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn models_listing_keeps_the_id_and_lists_the_alias() {
    let app = larql_server::routes::single_model_router(aliased_state());
    let body = common::body_json(common::get(app, "/v1/models").await.into_body()).await;
    assert_eq!(body["data"][0]["id"], DECLARED_ID);
    assert_eq!(body["data"][0]["aliases"], serde_json::json!([SERVED_NAME]));
}

#[tokio::test]
async fn an_unaliased_listing_has_no_aliases_field() {
    let app = larql_server::routes::single_model_router(common::state(vec![common::model("m")]));
    let body = common::body_json(common::get(app, "/v1/models").await.into_body()).await;
    assert!(body["data"][0].get("aliases").is_none(), "{body}");
}

#[tokio::test]
async fn retrieve_by_alias_returns_the_declared_id() {
    let app = larql_server::routes::single_model_router(aliased_state());
    let resp = common::get(app.clone(), &format!("/v1/models/{SERVED_NAME}")).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = common::body_json(resp.into_body()).await;
    assert_eq!(body["id"], DECLARED_ID);
    assert_eq!(body["aliases"], serde_json::json!([SERVED_NAME]));

    let resp = common::get(app, "/v1/models/QWEN3-0.6B").await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

/// The reported newcomer path: a chat request naming the served name.
#[tokio::test]
async fn chat_completions_accepts_the_served_name() {
    let (model, _fixture) = common::model_with_q4k_weights(DECLARED_ID);
    let st = common::state(vec![model]);
    st.model_set
        .write()
        .unwrap()
        .alias_sole_model(SERVED_NAME)
        .unwrap();
    let app = larql_server::routes::single_model_router(st);
    for (name, want) in [
        (SERVED_NAME, StatusCode::OK),
        (DECLARED_ID, StatusCode::OK),
        ("not-served", StatusCode::NOT_FOUND),
    ] {
        let body = serde_json::json!({
            "model": name,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
        });
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
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
        assert_eq!(status, want, "{name}: {}", String::from_utf8_lossy(&bytes));
    }
}
