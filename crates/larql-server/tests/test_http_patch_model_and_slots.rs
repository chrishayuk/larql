//! `POST /v1/patches/apply` applies a patch only to the model it was written
//! against, and an INSERT that asks the server for a slot gets one that is
//! free on the overlay it lands on.
//!
//! The route used to apply any patch to any model, including one naming no
//! model (remote LQL `DELETE` and `UPDATE` sent an empty `base_model`). And an
//! INSERT with no gate vector and `feature: 0` was given a slot by hashing the
//! entity into `1..=10240`, whatever the layer's width and whether or not the
//! slot was already taken.

mod common;
use common::*;

use axum::http::StatusCode;
use larql_vindex::PatchedVindex;

const SESSION_HEADER: &str = "x-session-id";
const SESSION: &str = "slots-session";
const LAYER: usize = 0;
/// `feature` on an INSERT with no gate vector: the server chooses the slot.
const SERVER_CHOOSES: usize = 0;
const ANOTHER_MODEL: &str = "another/model-7";
/// The targets the inserts below install, read back to find their slots.
/// The fixture's own features are Paris, French and Europe.
const TARGETS: [&str; 2] = ["Rome", "Berlin"];

fn insert_asking_for_a_slot(entity: &str, target: &str) -> serde_json::Value {
    serde_json::json!({
        "op": "insert",
        "layer": LAYER,
        "feature": SERVER_CHOOSES,
        "entity": entity,
        "relation": "capital",
        "target": target,
        "gate_vector_b64": null
    })
}

fn patch(name: &str, base_model: &str, operations: Vec<serde_json::Value>) -> serde_json::Value {
    serde_json::json!({
        "patch": {
            "version": 1,
            "base_model": base_model,
            "base_checksum": null,
            "created_at": "2026-11-12",
            "description": name,
            "author": null,
            "tags": [],
            "operations": operations
        }
    })
}

async fn global_patch_count(st: &std::sync::Arc<larql_server::state::AppState>) -> usize {
    let resp = get(single_model_router(st.clone()), "/v1/patches").await;
    assert_eq!(resp.status(), StatusCode::OK);
    body_json(resp.into_body()).await["patches"]
        .as_array()
        .expect("patches")
        .len()
}

/// The slot holding each of `TARGETS`, in order.
fn slots_of_targets(patched: &PatchedVindex) -> Vec<Option<usize>> {
    let width = patched.base().num_features(LAYER);
    TARGETS
        .iter()
        .map(|target| {
            (0..width).find(|&slot| {
                patched
                    .feature_meta(LAYER, slot)
                    .is_some_and(|meta| meta.top_token == *target)
            })
        })
        .collect()
}

fn assert_two_distinct_slots_in_the_layer(patched: &PatchedVindex) {
    let width = patched.base().num_features(LAYER);
    let slots = slots_of_targets(patched);
    let [Some(first), Some(second)] = slots[..] else {
        panic!("both inserts must be installed, found {slots:?}");
    };
    assert_ne!(first, second, "two inserts were given one slot");
    assert!(first < width && second < width, "{slots:?} past {width}");
}

// ── The patch's model ────────────────────────────────────────────────────

#[tokio::test]
async fn a_patch_written_for_another_model_is_refused_and_not_applied() {
    let st = state(vec![model("test")]);
    let resp = post_json(
        single_model_router(st.clone()),
        "/v1/patches/apply",
        patch(
            "elsewhere",
            ANOTHER_MODEL,
            vec![serde_json::json!(
                {"op": "delete", "layer": LAYER, "feature": 2}
            )],
        ),
    )
    .await;

    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = body_json(resp.into_body()).await.to_string();
    assert!(body.contains(ANOTHER_MODEL), "{body}");
    assert_eq!(global_patch_count(&st).await, 0);
}

#[tokio::test]
async fn a_patch_naming_no_model_is_refused_and_not_applied() {
    let st = state(vec![model("test")]);
    let resp = post_json(
        single_model_router(st.clone()),
        "/v1/patches/apply",
        patch(
            "nameless",
            "",
            vec![serde_json::json!(
                {"op": "delete", "layer": LAYER, "feature": 2}
            )],
        ),
    )
    .await;

    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = body_json(resp.into_body()).await.to_string();
    assert!(body.contains("no base_model"), "{body}");
    assert_eq!(global_patch_count(&st).await, 0);
}

#[tokio::test]
async fn a_session_patch_for_another_model_leaves_no_session_behind() {
    let st = state(vec![model("test")]);
    let resp = post_json_h(
        single_model_router(st.clone()),
        "/v1/patches/apply",
        patch(
            "elsewhere",
            ANOTHER_MODEL,
            vec![serde_json::json!(
                {"op": "delete", "layer": LAYER, "feature": 2}
            )],
        ),
        (SESSION_HEADER, SESSION),
    )
    .await;

    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    assert_eq!(st.sessions.session_count().await, 0);
}

#[tokio::test]
async fn a_patch_for_the_served_model_is_applied() {
    let st = state(vec![model("test")]);
    let resp = post_json(
        single_model_router(st.clone()),
        "/v1/patches/apply",
        inline_delete_patch("ours"),
    )
    .await;

    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(global_patch_count(&st).await, 1);
}

// ── Slots the server chooses ─────────────────────────────────────────────

#[tokio::test]
async fn inserts_asking_for_slots_get_distinct_slots_inside_the_layer() {
    let st = state(vec![model_functional("test")]);
    let resp = post_json(
        single_model_router(st.clone()),
        "/v1/patches/apply",
        patch(
            "two-inserts",
            FIXTURE_MODEL,
            vec![
                insert_asking_for_a_slot("France", TARGETS[0]),
                insert_asking_for_a_slot("Germany", TARGETS[1]),
            ],
        ),
    )
    .await;

    let status = resp.status();
    let body = body_json(resp.into_body()).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let m = st.first_model().expect("model");
    assert_two_distinct_slots_in_the_layer(&*m.patched.read().await);
}

#[tokio::test]
async fn a_sessions_second_insert_does_not_take_the_slot_of_its_first() {
    let st = state(vec![model_functional("test")]);
    for (n, (entity, target)) in [("France", TARGETS[0]), ("Germany", TARGETS[1])]
        .into_iter()
        .enumerate()
    {
        let resp = post_json_h(
            single_model_router(st.clone()),
            "/v1/patches/apply",
            patch(
                &format!("insert-{n}"),
                FIXTURE_MODEL,
                vec![insert_asking_for_a_slot(entity, target)],
            ),
            (SESSION_HEADER, SESSION),
        )
        .await;
        let status = resp.status();
        let body = body_json(resp.into_body()).await;
        assert_eq!(status, StatusCode::OK, "{body}");
    }

    let m = st.first_model().expect("model");
    let session_view = st
        .sessions
        .get_or_create(SESSION, &m)
        .await
        .expect("session overlay");
    assert_two_distinct_slots_in_the_layer(&session_view);

    // The session's slots are its own: the shared model is untouched.
    let global = m.patched.read().await;
    assert_eq!(slots_of_targets(&global), vec![None, None]);
}

#[tokio::test]
async fn a_slot_stated_by_one_op_is_not_chosen_for_another() {
    let st = state(vec![model_functional("test")]);
    let stated = serde_json::json!({
        "op": "insert",
        "layer": LAYER,
        "feature": 2,
        "entity": "Germany",
        "relation": "capital",
        "target": TARGETS[1],
        "gate_vector_b64": null
    });
    let resp = post_json(
        single_model_router(st.clone()),
        "/v1/patches/apply",
        patch(
            "stated-and-chosen",
            FIXTURE_MODEL,
            vec![insert_asking_for_a_slot("France", TARGETS[0]), stated],
        ),
    )
    .await;

    let status = resp.status();
    let body = body_json(resp.into_body()).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let m = st.first_model().expect("model");
    let slots = slots_of_targets(&*m.patched.read().await);
    assert_eq!(slots[1], Some(2), "the stated slot is kept: {slots:?}");
    assert_two_distinct_slots_in_the_layer(&*m.patched.read().await);
}

#[tokio::test]
async fn more_inserts_than_the_layer_has_slots_are_refused_whole() {
    let st = state(vec![model_functional("test")]);
    let m = st.first_model().expect("model");
    let width = m.patched.read().await.base().num_features(LAYER);
    let one_too_many = (0..=width)
        .map(|n| insert_asking_for_a_slot(&format!("entity-{n}"), &format!("target-{n}")))
        .collect();

    let resp = post_json(
        single_model_router(st.clone()),
        "/v1/patches/apply",
        patch("one-too-many", FIXTURE_MODEL, one_too_many),
    )
    .await;

    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = body_json(resp.into_body()).await.to_string();
    assert!(body.contains("no free feature slot"), "{body}");
    assert_eq!(global_patch_count(&st).await, 0);
}
