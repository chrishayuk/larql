//! `POST`/`DELETE /v1/runtime/model` — dynamic single-model lifecycle.
//!
//! Implements the invariant `state::lifecycle` exists to protect
//! (`docs/runtime-lifecycle-design.md` §3-4, §7): this endpoint only
//! ever works on a `SingleModel`-topology server, and only ever keeps
//! the bound count at 0 or 1. `AppState::validate_lifecycle_mutation`
//! is checked first, before anything else — a `MultiModel`-topology
//! boot refuses both verbs outright.
//!
//! Deliberately boring transitions:
//!
//! ```text
//! POST   idle -> loading -> ready
//!        ready + same path  -> idempotent success
//!        ready + other path -> reject (no atomic replacement —
//!                               DELETE the old one, then POST the new)
//!
//! DELETE ready -> unloading/draining -> idle
//!        idle -> idempotent success
//! ```
//!
//! The unload sequence, in order:
//! 1. Remove the `Arc` from `ModelSet` — this alone stops any *new*
//!    request from resolving to the model (`AppState::model`/`served`
//!    read the set fresh every time); an *existing* in-flight request
//!    keeps working off its own cloned `Arc`, memory-safe by
//!    construction (`docs/runtime-lifecycle-design.md` §1).
//! 2. Poll the removed model's in-flight counter until it reaches
//!    zero, or a timeout elapses.
//! 3. On success: invalidate every session and KV-cache entry tied to
//!    the model's id (`docs/runtime-lifecycle-design.md` §1's
//!    id-reuse trap) — on *every* successful unload, not only a
//!    reload, so nothing survives the lifetime of the binding that
//!    produced it.
//! 4. On drain timeout: fail closed. Put the `Arc` back exactly where
//!    it came from, report the model still bound, and refuse — never
//!    claim "unloaded" while a generation might still hold the
//!    weights. This codebase has no cooperative generation
//!    cancellation (see the design doc's non-goals), so "the drain
//!    timed out" is reported as a conflict, not silently overridden.

use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use axum::extract::State;
use axum::Json;
use serde::Deserialize;
use utoipa::ToSchema;

use crate::bootstrap::{load_artifact, LoadVindexOptions, LoadedArtifact};
use crate::error::ServerError;
use crate::state::{AppState, LifecycleState, LoadDecision, ServedModel, UnloadDecision};

use super::runtime::runtime_snapshot;

/// How long `DELETE` waits for a bound model's in-flight generations
/// to reach zero before refusing and leaving it bound. The same order
/// of magnitude as `--infer-timeout`'s own default (60s) — this
/// codebase already accepts that a single generation may legitimately
/// run that long, so unload has to tolerate it too, not the much
/// shorter GT6 walk-ffn drain window (30s, tuned for single-layer
/// compute, not end-to-end generation). Not yet CLI-configurable —
/// follow-up if a fixed default proves wrong in practice.
const UNLOAD_DRAIN_TIMEOUT: Duration = Duration::from_secs(60);

/// How often the drain loop re-checks the in-flight counter.
const UNLOAD_POLL_INTERVAL: Duration = Duration::from_millis(50);

#[derive(Deserialize, ToSchema)]
pub struct LoadModelRequest {
    /// Filesystem path to a VINDEX2 or VINDEX3 container — the same
    /// path `--vindex-path` already accepts. No registry/name
    /// resolution yet (`docs/runtime-lifecycle-design.md`'s explicit
    /// non-goals for this rung).
    pub path: String,
    /// VINDEX3 backend; defaults to CPU. Refused for V2 when set to Metal.
    #[serde(default)]
    pub backend: crate::vindex3::V3Backend,
}

fn model_id_of(model: &ServedModel) -> &str {
    match model {
        ServedModel::V2(m) => &m.id,
        ServedModel::V3(m) => &m.id,
    }
}

fn path_of(model: &ServedModel) -> &Path {
    match model {
        ServedModel::V2(m) => &m.path,
        ServedModel::V3(m) => &m.path,
    }
}

/// A cheap, non-blocking read of how many generations are currently
/// in flight on `model` — `LoadedModel.requests_in_flight` (a public
/// field, shared with the GT6 walk-ffn drain) for V2,
/// `V3Model::requests_in_flight()` (added alongside `V3GenerationGuard`
/// specifically so unload would have something to poll) for V3.
fn in_flight_count(model: &ServedModel) -> u32 {
    match model {
        ServedModel::V2(m) => m.requests_in_flight.load(Ordering::Relaxed),
        ServedModel::V3(m) => m.requests_in_flight(),
    }
}

/// Poll `model`'s in-flight counter until it reaches zero or `timeout`
/// elapses. Returns whether it drained. A parameter rather than the
/// module constant so tests can exercise the timeout path in
/// milliseconds instead of really waiting a minute.
async fn drain(model: &ServedModel, timeout: Duration) -> bool {
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        if in_flight_count(model) == 0 {
            return true;
        }
        if tokio::time::Instant::now() >= deadline {
            return false;
        }
        tokio::time::sleep(UNLOAD_POLL_INTERVAL).await;
    }
}

#[utoipa::path(
    post,
    path = "/v1/runtime/model",
    tag = "admin",
    request_body = LoadModelRequest,
    responses(
        (status = 200, description = "Loaded (or already bound to the same path) — runtime snapshot", body = crate::openapi::schemas::RuntimeResponse),
        (status = 400, body = crate::error::ErrorBody, description = "The path does not produce a loadable model"),
        (status = 409, body = crate::error::ErrorBody, description = "A load/unload is already in progress, a different model is already bound, or this server's topology does not support dynamic lifecycle"),
    ),
)]
pub async fn handle_load_model(
    State(state): State<Arc<AppState>>,
    Json(req): Json<LoadModelRequest>,
) -> Result<Json<serde_json::Value>, ServerError> {
    state.bump_requests();
    load_model(&state, req, UNLOAD_DRAIN_TIMEOUT).await
}

#[utoipa::path(
    delete,
    path = "/v1/runtime/model",
    tag = "admin",
    responses(
        (status = 200, description = "Unloaded (or already idle) — runtime snapshot", body = crate::openapi::schemas::RuntimeResponse),
        (status = 409, body = crate::error::ErrorBody, description = "A load/unload is already in progress, generation is still active past the drain timeout, or this server's topology does not support dynamic lifecycle"),
    ),
)]
pub async fn handle_unload_model(
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>, ServerError> {
    state.bump_requests();
    unload_model(&state, UNLOAD_DRAIN_TIMEOUT).await
}

/// The actual `POST` logic, with the drain timeout as a parameter so
/// tests aren't bound by [`UNLOAD_DRAIN_TIMEOUT`]'s real duration.
/// (A load has no drain of its own, but takes the same parameter as
/// `unload_model` for symmetry and because a future atomic-replace
/// endpoint would need to drain the old model as part of a "load".)
async fn load_model(
    state: &AppState,
    req: LoadModelRequest,
    _drain_timeout: Duration,
) -> Result<Json<serde_json::Value>, ServerError> {
    state
        .validate_lifecycle_mutation(1)
        .map_err(|e| ServerError::Conflict(e.to_string()))?;

    let requested_path = PathBuf::from(&req.path);
    {
        let mut lifecycle = state.lifecycle.lock().unwrap_or_else(|p| p.into_inner());
        match crate::state::decide_load(&lifecycle, &requested_path) {
            LoadDecision::Refuse(msg) => return Err(ServerError::Conflict(msg)),
            LoadDecision::AlreadyBound => {
                let current = state.served(None);
                let same_backend = match current {
                    Some(ServedModel::V3(ref model)) => model.backend == req.backend,
                    _ => req.backend == crate::vindex3::V3Backend::Cpu,
                };
                if !same_backend {
                    return Err(ServerError::Conflict("model is already bound with a different backend; unload it before changing backend".into()));
                }
                return Ok(Json(runtime_snapshot(state)));
            }
            LoadDecision::Proceed => *lifecycle = LifecycleState::Loading,
        }
    }

    let load_path = req.path.clone();
    let load_result = tokio::task::spawn_blocking(move || {
        load_artifact(
            &load_path,
            LoadVindexOptions {
                v3_backend: req.backend,
                ..Default::default()
            },
        )
    })
    .await;

    match load_result {
        Ok(Ok(LoadedArtifact::V2(m))) => {
            let model_id = m.id.clone();
            let path = m.path.clone();
            state
                .model_set
                .write()
                .unwrap_or_else(|p| p.into_inner())
                .insert_v2(Arc::new(*m));
            *state.lifecycle.lock().unwrap_or_else(|p| p.into_inner()) =
                LifecycleState::Ready { model_id, path };
            Ok(Json(runtime_snapshot(state)))
        }
        Ok(Ok(LoadedArtifact::V3(m))) => {
            let model_id = m.id.clone();
            let path = m.path.clone();
            state
                .model_set
                .write()
                .unwrap_or_else(|p| p.into_inner())
                .insert_v3(Arc::new(*m));
            *state.lifecycle.lock().unwrap_or_else(|p| p.into_inner()) =
                LifecycleState::Ready { model_id, path };
            Ok(Json(runtime_snapshot(state)))
        }
        Ok(Err(e)) => {
            // Fail closed: nothing was ever bound, so back to idle.
            *state.lifecycle.lock().unwrap_or_else(|p| p.into_inner()) = LifecycleState::Idle;
            Err(ServerError::BadRequest(format!(
                "'{}' did not produce a loadable model: {e}",
                req.path
            )))
        }
        Err(join_err) => {
            *state.lifecycle.lock().unwrap_or_else(|p| p.into_inner()) = LifecycleState::Idle;
            Err(ServerError::Internal(format!(
                "load task panicked: {join_err}"
            )))
        }
    }
}

/// The actual `DELETE` logic, with the drain timeout as a parameter —
/// see [`load_model`]'s doc comment for why.
async fn unload_model(
    state: &AppState,
    drain_timeout: Duration,
) -> Result<Json<serde_json::Value>, ServerError> {
    state
        .validate_lifecycle_mutation(0)
        .map_err(|e| ServerError::Conflict(e.to_string()))?;

    let model_id = {
        let mut lifecycle = state.lifecycle.lock().unwrap_or_else(|p| p.into_inner());
        match crate::state::decide_unload(&lifecycle) {
            UnloadDecision::Refuse(msg) => return Err(ServerError::Conflict(msg)),
            UnloadDecision::AlreadyIdle => return Ok(Json(runtime_snapshot(state))),
            UnloadDecision::Proceed { model_id } => {
                *lifecycle = LifecycleState::Unloading {
                    model_id: model_id.clone(),
                };
                model_id
            }
        }
    };

    // Step 1: mark unavailable for new generations by taking it out of
    // the set request resolution reads. An existing in-flight request
    // already cloned its own Arc and keeps working regardless.
    let removed = state
        .model_set
        .write()
        .unwrap_or_else(|p| p.into_inner())
        .remove(&model_id);
    let Some(removed) = removed else {
        // Unreachable in practice — `decide_unload` only proposed this
        // id because `lifecycle` said it was bound — but fail closed
        // rather than panic if the two ever disagree.
        *state.lifecycle.lock().unwrap_or_else(|p| p.into_inner()) = LifecycleState::Idle;
        return Err(ServerError::Internal(format!(
            "model '{model_id}' was marked bound but was not found in the model set"
        )));
    };

    // Step 2: wait for in-flight generations to finish.
    if !drain(&removed, drain_timeout).await {
        // Fail closed: put it back exactly where it came from, report
        // it as still bound, and refuse.
        let path = path_of(&removed).to_path_buf();
        let model_id_owned = model_id_of(&removed).to_string();
        state
            .model_set
            .write()
            .unwrap_or_else(|p| p.into_inner())
            .reinsert(removed);
        *state.lifecycle.lock().unwrap_or_else(|p| p.into_inner()) = LifecycleState::Ready {
            model_id: model_id_owned,
            path,
        };
        return Err(ServerError::Conflict(format!(
            "generation on model '{model_id}' was still active after {}s; the model remains bound",
            drain_timeout.as_secs(),
        )));
    }

    // Step 3: invalidate session/KV state tied to this model identity
    // — every successful unload, not only a reload.
    state.sessions.drop_sessions_bound_to(&model_id).await;
    state.v3_kv.drop_owned_by_model(&model_id);

    // Step 4: report idle.
    *state.lifecycle.lock().unwrap_or_else(|p| p.into_inner()) = LifecycleState::Idle;
    Ok(Json(runtime_snapshot(state)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::ModelSet;

    fn stub_v2(id: &str, path: &str) -> ServedModel {
        // Minimal but real `LoadedModel` — enough to exercise
        // in_flight_count/model_id_of/path_of without a vindex on
        // disk. Mirrors state/model_set.rs's own `stub_model`.
        use larql_vindex::ndarray::Array2;
        use larql_vindex::{ExtractLevel, LayerBands, QuantFormat, VectorIndex, VindexConfig};
        let hidden = 4;
        let index = VectorIndex::new(
            vec![Some(Array2::<f32>::zeros((1, hidden)))],
            vec![None],
            1,
            hidden,
        );
        let patched = larql_vindex::PatchedVindex::new(index);
        let tok_json =
            r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
        let tokenizer = larql_vindex::tokenizers::Tokenizer::from_bytes(tok_json).unwrap();
        ServedModel::V2(Arc::new(crate::state::LoadedModel {
            id: id.to_string(),
            path: PathBuf::from(path),
            config: VindexConfig {
                version: 2,
                model: id.to_string(),
                family: "test".to_string(),
                source: None,
                checksums: None,
                num_layers: 1,
                hidden_size: hidden,
                intermediate_size: hidden,
                vocab_size: 4,
                embed_scale: 1.0,
                extract_level: ExtractLevel::Browse,
                dtype: larql_vindex::StorageDtype::default(),
                quant: QuantFormat::None,
                layer_bands: Some(LayerBands {
                    syntax: (0, 0),
                    knowledge: (0, 0),
                    output: (0, 0),
                }),
                layers: vec![],
                down_top_k: 1,
                has_model_weights: false,
                model_config: None,
                fp4: None,
                ffn_layout: None,
                bitnet_layout: None,
            },
            patched: Arc::new(tokio::sync::RwLock::new(patched)),
            embeddings: Array2::<f32>::zeros((4, hidden)),
            embed_scale: 1.0,
            tokenizer,
            infer_disabled: true,
            ffn_only: false,
            embed_only: false,
            embed_store: None,
            release_mmap_after_request: false,
            weights: std::sync::OnceLock::new(),
            weights_init: std::sync::Mutex::new(()),
            bitnet_model: std::sync::OnceLock::new(),
            bitnet_init: std::sync::Mutex::new(()),
            probe_labels: std::collections::HashMap::new(),
            ffn_l2_cache: crate::ffn_l2_cache::FfnL2Cache::new(1),
            layer_latency_tracker: Arc::new(crate::metrics::LayerLatencyTracker::new()),
            requests_in_flight: Arc::new(std::sync::atomic::AtomicU32::new(0)),
            requests_total: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            expert_filter: None,
            unit_filter: None,
            moe_remote: None,
            #[cfg(all(feature = "metal-experts", target_os = "macos"))]
            metal_backend: std::sync::OnceLock::new(),
            #[cfg(all(feature = "metal-experts", target_os = "macos"))]
            moe_scratches: std::sync::Mutex::new(std::collections::HashMap::new()),
            #[cfg(all(feature = "metal-experts", target_os = "macos"))]
            metal_ffn_layer_bufs: std::sync::OnceLock::new(),
        }))
    }

    #[test]
    fn model_id_and_path_read_the_v2_fields() {
        let m = stub_v2("m1", "/vindexes/m1");
        assert_eq!(model_id_of(&m), "m1");
        assert_eq!(path_of(&m), Path::new("/vindexes/m1"));
    }

    #[test]
    fn in_flight_count_reads_the_v2_atomic() {
        let m = stub_v2("m1", "/a");
        assert_eq!(in_flight_count(&m), 0);
        if let ServedModel::V2(ref inner) = m {
            inner.requests_in_flight.fetch_add(3, Ordering::Relaxed);
        }
        assert_eq!(in_flight_count(&m), 3);
    }

    #[tokio::test]
    async fn drain_returns_immediately_when_already_zero() {
        let m = stub_v2("m1", "/a");
        let started = std::time::Instant::now();
        assert!(drain(&m, Duration::from_secs(5)).await);
        assert!(
            started.elapsed() < Duration::from_millis(200),
            "must not wait out the timeout when already drained"
        );
    }

    #[tokio::test]
    async fn drain_times_out_while_generation_is_still_in_flight() {
        let m = stub_v2("m1", "/a");
        if let ServedModel::V2(ref inner) = m {
            inner.requests_in_flight.fetch_add(1, Ordering::Relaxed);
        }
        assert!(!drain(&m, Duration::from_millis(120)).await);
    }

    #[tokio::test]
    async fn drain_succeeds_once_the_counter_drops_to_zero_mid_wait() {
        let m = Arc::new(stub_v2("m1", "/a"));
        if let ServedModel::V2(ref inner) = *m {
            inner.requests_in_flight.fetch_add(1, Ordering::Relaxed);
        }
        let bg = Arc::clone(&m);
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(60)).await;
            if let ServedModel::V2(ref inner) = *bg {
                inner.requests_in_flight.fetch_sub(1, Ordering::Relaxed);
            }
        });
        assert!(drain(&m, Duration::from_secs(2)).await);
    }

    #[test]
    fn model_set_remove_then_reinsert_round_trips_for_the_fail_closed_path() {
        // Exercises the exact sequence unload_model's timeout branch
        // relies on: remove, (drain fails), reinsert — without needing
        // a real AppState or HTTP layer.
        let mut set = ModelSet::default();
        set.insert_v2(match stub_v2("m1", "/a") {
            ServedModel::V2(m) => m,
            ServedModel::V3(_) => unreachable!(),
        });
        let removed = set.remove("m1").expect("just inserted");
        assert!(set.models.is_empty(), "removed while draining");
        set.reinsert(removed);
        assert_eq!(set.models.len(), 1, "put back after a failed drain");
        assert_eq!(set.models[0].id, "m1");
    }

    // ── load_model / unload_model — the full state machine, exercised
    // directly (bypassing the HTTP layer and the real
    // UNLOAD_DRAIN_TIMEOUT) since these are private to this module.
    // HTTP-level wiring and a genuine on-disk load are covered by
    // `tests/test_runtime_lifecycle.rs`.

    fn v2_of(model: ServedModel) -> Arc<crate::state::LoadedModel> {
        match model {
            ServedModel::V2(m) => m,
            ServedModel::V3(_) => unreachable!("stub_v2 only ever produces a V2 model"),
        }
    }

    fn state_with(
        topology: crate::state::RouterTopology,
        lifecycle: LifecycleState,
        models: Vec<Arc<crate::state::LoadedModel>>,
    ) -> AppState {
        AppState {
            model_set: std::sync::RwLock::new(ModelSet {
                models,
                v3_models: Vec::new(),
            }),
            router_topology: topology,
            lifecycle: std::sync::Mutex::new(lifecycle),
            started_at: std::time::Instant::now(),
            requests_served: std::sync::atomic::AtomicU64::new(0),
            api_key: None,
            sessions: crate::session::SessionManager::new(3600),
            responses: crate::response_store::ResponseStore::new(),
            v3_kv: crate::response_kv::ResponseKvCache::new(
                crate::response_kv::DEFAULT_MAX_ENTRIES,
                crate::response_kv::DEFAULT_TTL_SECS,
            ),
            describe_cache: crate::cache::DescribeCache::new(0),
            infer_timeout: Duration::from_secs(60),
            runtime: Arc::new(crate::runtime_stats::RuntimeRecorder::new()),
        }
    }

    #[tokio::test]
    async fn load_model_refuses_under_multimodel_topology() {
        let state = state_with(
            crate::state::RouterTopology::MultiModel,
            LifecycleState::Idle,
            vec![],
        );
        let result = load_model(
            &state,
            LoadModelRequest {
                path: "/a".into(),
                backend: Default::default(),
            },
            Duration::from_secs(1),
        )
        .await;
        assert!(matches!(result, Err(ServerError::Conflict(_))));
    }

    #[tokio::test]
    async fn load_model_refuses_while_a_load_is_already_in_progress() {
        let state = state_with(
            crate::state::RouterTopology::SingleModel,
            LifecycleState::Loading,
            vec![],
        );
        let result = load_model(
            &state,
            LoadModelRequest {
                path: "/a".into(),
                backend: Default::default(),
            },
            Duration::from_secs(1),
        )
        .await;
        assert!(matches!(result, Err(ServerError::Conflict(_))));
        assert_eq!(
            *state.lifecycle.lock().unwrap(),
            LifecycleState::Loading,
            "a refused mutation must not touch the in-progress state"
        );
    }

    #[tokio::test]
    async fn load_model_reports_bad_request_for_an_unloadable_path_and_reverts_to_idle() {
        let state = state_with(
            crate::state::RouterTopology::SingleModel,
            LifecycleState::Idle,
            vec![],
        );
        let result = load_model(
            &state,
            LoadModelRequest {
                backend: Default::default(),
                path: "/definitely/does/not/exist-runtime-lifecycle-test".into(),
            },
            Duration::from_secs(1),
        )
        .await;
        assert!(matches!(result, Err(ServerError::BadRequest(_))));
        assert_eq!(
            *state.lifecycle.lock().unwrap(),
            LifecycleState::Idle,
            "a failed load must fail closed back to idle, not strand Loading"
        );
    }

    #[tokio::test]
    async fn unload_model_is_idempotent_when_already_idle() {
        let state = state_with(
            crate::state::RouterTopology::SingleModel,
            LifecycleState::Idle,
            vec![],
        );
        let result = unload_model(&state, Duration::from_secs(1)).await;
        assert!(result.is_ok(), "{result:?}");
    }

    #[tokio::test]
    async fn unload_model_refuses_under_multimodel_topology() {
        let state = state_with(
            crate::state::RouterTopology::MultiModel,
            LifecycleState::Idle,
            vec![],
        );
        let result = unload_model(&state, Duration::from_secs(1)).await;
        assert!(matches!(result, Err(ServerError::Conflict(_))));
    }

    #[tokio::test]
    async fn unload_model_drains_removes_and_purges_bound_session_state() {
        let model = v2_of(stub_v2("m1", "/a"));
        let path = model.path.clone();
        let state = state_with(
            crate::state::RouterTopology::SingleModel,
            LifecycleState::Ready {
                model_id: "m1".to_string(),
                path,
            },
            vec![Arc::clone(&model)],
        );
        state.sessions.get_or_create("sess1", &model).await;
        assert_eq!(state.sessions.session_count().await, 1);

        let result = unload_model(&state, Duration::from_secs(1)).await;
        assert!(result.is_ok(), "{result:?}");
        assert_eq!(*state.lifecycle.lock().unwrap(), LifecycleState::Idle);
        assert!(state.model_set.read().unwrap().models.is_empty());
        assert_eq!(
            state.sessions.session_count().await,
            0,
            "a session bound to the unloaded model's id must not survive unload \
             (docs/runtime-lifecycle-design.md §1's id-reuse trap)"
        );
    }

    #[tokio::test]
    async fn unload_model_fails_closed_and_rebinds_when_drain_times_out() {
        let model = v2_of(stub_v2("m1", "/a"));
        model.requests_in_flight.fetch_add(1, Ordering::Relaxed);
        let path = model.path.clone();
        let ready = LifecycleState::Ready {
            model_id: "m1".to_string(),
            path,
        };
        let state = state_with(
            crate::state::RouterTopology::SingleModel,
            ready.clone(),
            vec![model],
        );

        let result = unload_model(&state, Duration::from_millis(100)).await;
        assert!(matches!(result, Err(ServerError::Conflict(_))));
        assert_eq!(
            *state.lifecycle.lock().unwrap(),
            ready,
            "a timed-out drain must leave the model reported as still bound"
        );
        assert_eq!(
            state.model_set.read().unwrap().models.len(),
            1,
            "the model must be put back, not left removed, on a failed drain"
        );
    }
}
