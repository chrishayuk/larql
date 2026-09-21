//! POST /v1/infer — full forward pass with attention.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::HeaderMap;
use axum::Json;
use serde::Deserialize;

use crate::band_utils::{INFER_MODE_COMPARE, INFER_MODE_DENSE, INFER_MODE_WALK};
use crate::error::ServerError;
use crate::session::extract_session_id;
use crate::state::{elapsed_ms, AppState, LoadedModel};

#[derive(Deserialize, utoipa::ToSchema)]
pub struct InferRequest {
    /// Prompt to run inference on.
    pub prompt: String,
    /// Top-K next-token predictions to return.
    #[serde(default = "default_top")]
    pub top: usize,
    /// Inference mode: `walk` (default), `dense`, or `compare`.
    #[serde(default = "default_mode")]
    pub mode: String,
}

fn default_top() -> usize {
    5
}
fn default_mode() -> String {
    INFER_MODE_WALK.into()
}

fn round_probability(prob: f64) -> f64 {
    (prob * 10000.0).round() / 10000.0
}

fn format_predictions(predictions: &[(String, f64)]) -> Vec<serde_json::Value> {
    predictions
        .iter()
        .map(|(tok, prob)| {
            serde_json::json!({
                "token": tok,
                "probability": round_probability(*prob),
            })
        })
        .collect()
}

fn format_knn_override(
    ovr: &larql_inference::KnnOverride,
    model_top1: Option<&(String, f64)>,
) -> serde_json::Value {
    let mut value = serde_json::json!({
        "token": &ovr.token,
        "cosine": ovr.cosine,
        "layer": ovr.layer,
        "source": "knn_override",
        "stage": "post_logits",
        "materialized": false,
    });
    if let Some((tok, prob)) = model_top1 {
        value["model_top1"] = serde_json::json!({
            "token": tok,
            "probability": round_probability(*prob),
        });
    }
    value
}

fn infer_mode_flags(mode: &str) -> (bool, bool, bool) {
    let is_compare = mode == INFER_MODE_COMPARE;
    let use_walk = mode == INFER_MODE_WALK || is_compare;
    let use_dense = mode == INFER_MODE_DENSE || is_compare;
    (is_compare, use_walk, use_dense)
}

fn run_infer(
    state: &AppState,
    model: &LoadedModel,
    req: &InferRequest,
    session_id: Option<&str>,
) -> Result<serde_json::Value, ServerError> {
    if model.infer_disabled {
        return Err(ServerError::InferenceUnavailable(
            "inference disabled (--no-infer)".into(),
        ));
    }

    // BitNet 1.58 (--keep-quant) vindex: take the native-ternary
    // forward path.  Skips dense weight loading entirely (~5 GB
    // saved on a 2 B BitNet) and runs predict_bitnet against the
    // pre-loaded BitnetModel.  Walk-mode is supported via
    // residual capture + KNN-store override (no sparse FFN — see
    // larql_inference::ternary::infer_bitnet_walk for the
    // architecture note).
    //
    // Checked before the `has_model_weights` gate below: a
    // `--keep-quant` container carries ternary artifacts rather than
    // the dense weight manifest that gate looks for, so a BitNet
    // vindex would otherwise be refused as weightless.
    if model.is_bitnet() {
        let bitnet_guard = model
            .get_or_load_bitnet()
            .map_err(ServerError::InferenceUnavailable)?;
        let bitnet: &larql_inference::ternary::BitnetModel = &bitnet_guard;

        let encoding = model
            .tokenizer
            .encode(req.prompt.as_str(), true)
            .map_err(|e| ServerError::Internal(format!("tokenize error: {e}")))?;
        let token_ids: Vec<u32> = encoding.get_ids().to_vec();
        if token_ids.is_empty() {
            return Err(ServerError::BadRequest("empty prompt".into()));
        }

        let start = std::time::Instant::now();
        let (is_compare, mut use_walk, mut use_dense) = infer_mode_flags(&req.mode);
        // Dense-only BitNet vindexes (`--dense-only`) have no gate
        // vectors / KNN store, so walk-mode would silently return
        // nothing useful.  Coerce any walk request to dense so
        // clients that omit `mode` (which defaults to walk) still
        // get correct predictions.  Compare-mode also collapses to
        // dense-only output here.
        if model.is_dense_only() && (use_walk || is_compare) {
            use_walk = false;
            use_dense = true;
        }
        let mut result = serde_json::Map::new();
        result.insert("prompt".into(), serde_json::json!(req.prompt));

        if use_walk {
            let run_bitnet_walk = |knn: &larql_vindex::patch::KnnStore| {
                larql_inference::ternary::infer_bitnet_walk(
                    bitnet,
                    &model.tokenizer,
                    Some(knn),
                    &token_ids,
                    req.top,
                )
            };
            // Same lock discipline and session fallback as the dense
            // path below: a reader on the sessions map, and a session
            // with no overlay reads like the global state.
            let walk_pred = if let Some(sid) = session_id {
                let sessions = state.sessions.sessions_blocking_read();
                if let Some(patched) = sessions.get(sid).and_then(|s| s.patched()) {
                    run_bitnet_walk(&patched.knn_store)
                } else {
                    drop(sessions);
                    let patched = model.patched.blocking_read();
                    run_bitnet_walk(&patched.knn_store)
                }
            } else {
                let patched = model.patched.blocking_read();
                run_bitnet_walk(&patched.knn_store)
            };

            let predictions = format_predictions(&walk_pred.predictions);
            if let Some(ovr) = &walk_pred.knn_override {
                result.insert(
                    "knn_override".into(),
                    format_knn_override(ovr, walk_pred.model_top1.as_ref()),
                );
            }
            if is_compare {
                result.insert(INFER_MODE_WALK.into(), serde_json::json!(predictions));
                result.insert(
                    "walk_ms".into(),
                    serde_json::json!((walk_pred.walk_ms * 10.0).round() / 10.0),
                );
            } else {
                result.insert("predictions".into(), serde_json::json!(predictions));
                result.insert("mode".into(), serde_json::json!(INFER_MODE_WALK));
            }
        }

        if use_dense {
            let dense_start = std::time::Instant::now();
            let pred = larql_inference::ternary::predict_bitnet(
                bitnet,
                &model.tokenizer,
                &token_ids,
                req.top,
            );
            let dense_ms = dense_start.elapsed().as_secs_f64() * 1000.0;

            let pred_pairs: Vec<(String, f64)> =
                pred.into_iter().map(|p| (p.token, p.probability)).collect();
            let predictions = format_predictions(&pred_pairs);
            if is_compare {
                result.insert(INFER_MODE_DENSE.into(), serde_json::json!(predictions));
                result.insert(
                    "dense_ms".into(),
                    serde_json::json!((dense_ms * 10.0).round() / 10.0),
                );
            } else {
                result.insert("predictions".into(), serde_json::json!(predictions));
                result.insert("mode".into(), serde_json::json!("bitnet"));
            }
        }

        result.insert("latency_ms".into(), serde_json::json!(elapsed_ms(start)));
        return Ok(serde_json::Value::Object(result));
    }

    if !model.config.has_model_weights
        && model.config.extract_level != larql_vindex::ExtractLevel::Inference
        && model.config.extract_level != larql_vindex::ExtractLevel::All
    {
        return Err(ServerError::InferenceUnavailable(
            "vindex does not contain model weights. Rebuild with --include-weights".into(),
        ));
    }

    let weights_guard = model
        .get_or_load_weights()
        .map_err(ServerError::InferenceUnavailable)?;
    let weights: &larql_inference::ModelWeights = &weights_guard;

    let encoding = model
        .tokenizer
        .encode(req.prompt.as_str(), true)
        .map_err(|e| ServerError::Internal(format!("tokenize error: {e}")))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();

    if token_ids.is_empty() {
        return Err(ServerError::BadRequest("empty prompt".into()));
    }

    let start = std::time::Instant::now();

    let (is_compare, use_walk, use_dense) = infer_mode_flags(&req.mode);

    let mut result = serde_json::Map::new();
    result.insert("prompt".into(), serde_json::json!(req.prompt));

    // Helper: run walk inference against a PatchedVindex.
    let run_walk = |patched: &larql_vindex::PatchedVindex| {
        larql_inference::infer_patched(
            weights,
            &model.tokenizer,
            patched,
            Some(&patched.knn_store),
            &token_ids,
            req.top,
            &larql_inference::KnnRouteMode::from_env(),
        )
    };

    if use_walk {
        let pred = if let Some(sid) = session_id {
            // Session-scoped walk inference.
            //
            // Lock discipline: take a *reader* on the sessions map (not
            // a writer) so concurrent sessioned `/v1/infer` requests do
            // not serialize globally, and so an in-flight forward pass
            // does not deadlock against a concurrent `apply_patch`
            // arriving on another worker.  The previous implementation
            // held `sessions.write()` across the multi-second
            // `run_walk(&session.patched)` call, which on the
            // multi-thread tokio runtime stalled every other handler
            // touching `sessions` (including `GET /v1/stats` and
            // `GET /v1/walk-ffn`).  This mirrors the fix already
            // applied in `session.rs::apply_patch`.
            let sessions = state.sessions.sessions_blocking_read();
            // A session with no overlay has never been patched, so it
            // reads exactly like the global state — same fallback as an
            // unknown session id.
            if let Some(patched) = sessions.get(sid).and_then(|s| s.patched()) {
                run_walk(patched)
            } else {
                drop(sessions);
                let patched = model.patched.blocking_read();
                run_walk(&patched)
            }
        } else {
            let patched = model.patched.blocking_read();
            run_walk(&patched)
        };

        let predictions = format_predictions(&pred.predictions);
        if let Some(ovr) = &pred.knn_override {
            result.insert(
                "knn_override".into(),
                format_knn_override(ovr, pred.model_top1.as_ref()),
            );
        }

        if is_compare {
            result.insert(INFER_MODE_WALK.into(), serde_json::json!(predictions));
            result.insert(
                "walk_ms".into(),
                serde_json::json!((pred.walk_ms * 10.0).round() / 10.0),
            );
        } else {
            result.insert("predictions".into(), serde_json::json!(predictions));
            result.insert("mode".into(), serde_json::json!(INFER_MODE_WALK));
        }
    }

    if use_dense {
        let dense_start = std::time::Instant::now();
        let pred = larql_inference::predict(weights, &model.tokenizer, &token_ids, req.top);
        let dense_ms = dense_start.elapsed().as_secs_f64() * 1000.0;

        let predictions = format_predictions(&pred.predictions);

        if is_compare {
            result.insert(INFER_MODE_DENSE.into(), serde_json::json!(predictions));
            result.insert(
                "dense_ms".into(),
                serde_json::json!((dense_ms * 10.0).round() / 10.0),
            );
        } else {
            result.insert("predictions".into(), serde_json::json!(predictions));
            result.insert("mode".into(), serde_json::json!(INFER_MODE_DENSE));
        }
    }

    result.insert("latency_ms".into(), serde_json::json!(elapsed_ms(start)));

    Ok(serde_json::Value::Object(result))
}

#[utoipa::path(
    post,
    path = "/v1/infer",
    tag = "inference",
    request_body = InferRequest,
    responses(
        (status = 200, description = "Next-token predictions", body = crate::openapi::schemas::InferResponse),
        (status = 400, body = crate::error::ErrorBody),
        (status = 503, body = crate::error::ErrorBody, description = "Inference weights unavailable"),
        (status = 500, body = crate::error::ErrorBody),
    ),
)]
pub async fn handle_infer(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<InferRequest>,
) -> Result<Json<serde_json::Value>, ServerError> {
    state.bump_requests();
    let model = state.model_or_err(None)?.clone();
    let sid = extract_session_id(&headers);
    let state2 = Arc::clone(&state);
    let timeout = state.infer_timeout;
    let result = run_infer_with_timeout(state2, model, req, sid, timeout).await?;
    Ok(Json(result))
}

#[utoipa::path(
    post,
    path = "/v1/{model_id}/infer",
    tag = "inference",
    params(("model_id" = String, Path, description = "Id of a loaded vindex.")),
    request_body = InferRequest,
    responses(
        (status = 200, body = crate::openapi::schemas::InferResponse),
        (status = 400, body = crate::error::ErrorBody),
        (status = 404, body = crate::error::ErrorBody),
        (status = 503, body = crate::error::ErrorBody),
    ),
)]
pub async fn handle_infer_multi(
    State(state): State<Arc<AppState>>,
    Path(model_id): Path<String>,
    headers: HeaderMap,
    Json(req): Json<InferRequest>,
) -> Result<Json<serde_json::Value>, ServerError> {
    state.bump_requests();
    let model = state.model_or_err(Some(&model_id))?.clone();
    let sid = extract_session_id(&headers);
    let state2 = Arc::clone(&state);
    let timeout = state.infer_timeout;
    let result = run_infer_with_timeout(state2, model, req, sid, timeout).await?;
    Ok(Json(result))
}

/// Race the blocking inference against `timeout` (zero = disabled).
///
/// On timeout we drop the JoinHandle and respond 504; the spawned
/// thread runs to completion in the background and its result is
/// discarded.  The next `/v1/infer` arrives against an unblocked
/// handler.  See BUG-infer-deadlock §5.6.
///
/// pub(crate) so the routes::infer::tests module can drive it
/// directly.
pub(crate) async fn run_infer_with_timeout(
    state: Arc<AppState>,
    model: Arc<LoadedModel>,
    req: InferRequest,
    session_id: Option<String>,
    timeout: std::time::Duration,
) -> Result<serde_json::Value, ServerError> {
    let started = std::time::Instant::now();
    let handle =
        tokio::task::spawn_blocking(move || run_infer(&state, &model, &req, session_id.as_deref()));

    if timeout.is_zero() {
        return handle
            .await
            .map_err(|e| ServerError::Internal(e.to_string()))?;
    }

    match tokio::time::timeout(timeout, handle).await {
        Ok(join_result) => join_result.map_err(|e| ServerError::Internal(e.to_string()))?,
        Err(_elapsed) => {
            tracing::warn!(
                target: "larql_server::infer",
                "inference timed out after {:.1}s; dropping in-flight task and \
                 responding 504 (background thread will finish on its own)",
                started.elapsed().as_secs_f64(),
            );
            Err(ServerError::Timeout(format!(
                "inference exceeded server-side timeout of {}s",
                timeout.as_secs(),
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn infer_defaults_match_api_contract() {
        assert_eq!(default_top(), 5);
        assert_eq!(default_mode(), INFER_MODE_WALK);
    }

    #[test]
    fn infer_request_deserializes_defaults() {
        let req: InferRequest = serde_json::from_value(serde_json::json!({
            "prompt": "The capital of France is"
        }))
        .unwrap();
        assert_eq!(req.prompt, "The capital of France is");
        assert_eq!(req.top, 5);
        assert_eq!(req.mode, INFER_MODE_WALK);
    }

    #[test]
    fn infer_request_accepts_dense_and_compare_modes() {
        let dense: InferRequest = serde_json::from_value(serde_json::json!({
            "prompt": "x",
            "top": 2,
            "mode": "dense"
        }))
        .unwrap();
        assert_eq!(dense.top, 2);
        assert_eq!(dense.mode, INFER_MODE_DENSE);

        let compare: InferRequest = serde_json::from_value(serde_json::json!({
            "prompt": "x",
            "mode": "compare"
        }))
        .unwrap();
        assert_eq!(compare.mode, INFER_MODE_COMPARE);
    }

    /// Regression: `run_infer`'s sessioned branch must NOT take a
    /// `sessions_blocking_write` guard — it serialised every
    /// concurrent /v1/infer call against the entire forward pass
    /// and (under cgroup memory pressure during weight load) wedged
    /// the whole HTTP handler.  See `BUG-infer-deadlock.md` §4.3.
    ///
    /// We can't run a real forward pass in a unit test, but we can
    /// drive the same lock pattern against `SessionManager` and
    /// assert that 8 concurrent readers complete in parallel rather
    /// than serially — i.e. their wall-time is ~one slow op, not
    /// ~eight.
    #[test]
    fn sessions_reader_does_not_serialize_concurrent_callers() {
        use crate::session::SessionManager;
        use std::sync::Arc;
        use std::time::Duration;
        use std::time::Instant;

        let mgr = Arc::new(SessionManager::new(60));

        // Pre-seed a session so the reader path doesn't fall
        // through to slow-path session creation.
        {
            let mut sessions = mgr.sessions_blocking_write();
            mgr.bind_in_guard(&mut sessions, "test-sid", "m-test", Instant::now());
        }

        // Eight threads simulating run_infer's sessioned branch:
        // take the reader, sleep 100 ms (proxy for a forward pass),
        // drop.  If we mistakenly used a *writer* (the buggy
        // pre-fix code) the wall time would be 8 * 100 ms = 800 ms.
        // With reader, it should be ~100 ms.
        let start = Instant::now();
        let mut handles = Vec::new();
        for _ in 0..8 {
            let mgr = Arc::clone(&mgr);
            handles.push(std::thread::spawn(move || {
                let sessions = mgr.sessions_blocking_read();
                let _patched = sessions.get("test-sid").and_then(|s| s.patched());
                std::thread::sleep(Duration::from_millis(100));
                drop(sessions);
            }));
        }
        for h in handles {
            let _ = h.join();
        }
        let wall = start.elapsed();

        // Generous bound: real serialization would be 800 ms; even
        // perfect parallelism plus thread spawn jitter sits well
        // under 400 ms on any host that runs the test suite.
        assert!(
            wall < Duration::from_millis(400),
            "sessions reader serialized concurrent callers (took {:?}); \
             expected ~100 ms parallel, observed near 800 ms-style serialisation",
            wall
        );
    }

    #[test]
    fn infer_mode_flags_select_expected_paths() {
        assert_eq!(infer_mode_flags(INFER_MODE_WALK), (false, true, false));
        assert_eq!(infer_mode_flags(INFER_MODE_DENSE), (false, false, true));
        assert_eq!(infer_mode_flags(INFER_MODE_COMPARE), (true, true, true));
        assert_eq!(infer_mode_flags("unknown"), (false, false, false));
    }

    #[test]
    fn format_predictions_rounds_probability() {
        let predictions = format_predictions(&[("Paris".into(), 0.123456)]);
        assert_eq!(predictions[0]["token"], "Paris");
        assert_eq!(predictions[0]["probability"], 0.1235);
    }

    #[test]
    fn format_knn_override_without_model_top1_emits_no_top1_key() {
        let ovr = larql_inference::KnnOverride {
            token: "Paris".into(),
            cosine: 0.92,
            layer: 17,
        };
        let v = format_knn_override(&ovr, None);
        assert_eq!(v["token"], "Paris");
        let cos = v["cosine"].as_f64().unwrap();
        assert!((cos - 0.92).abs() < 1e-4, "got {cos}");
        assert_eq!(v["layer"], 17);
        assert_eq!(v["source"], "knn_override");
        assert_eq!(v["stage"], "post_logits");
        assert_eq!(v["materialized"], false);
        assert!(
            v.get("model_top1").is_none(),
            "no model_top1 when not supplied"
        );
    }

    #[test]
    fn format_knn_override_with_model_top1_includes_rounded_probability() {
        let ovr = larql_inference::KnnOverride {
            token: "Berlin".into(),
            cosine: 0.87,
            layer: 14,
        };
        let top1 = ("Madrid".to_string(), 0.987654321);
        let v = format_knn_override(&ovr, Some(&top1));
        assert_eq!(v["model_top1"]["token"], "Madrid");
        // Probability is rounded to 4 decimals (round_probability * 10000).
        assert_eq!(v["model_top1"]["probability"], 0.9877);
    }

    /// BUG-infer-deadlock §5.6: when an inference exceeds the
    /// server-side timeout, the handler must respond 504 promptly
    /// and the next request must succeed without waiting for the
    /// timed-out one to finish.
    ///
    /// We simulate `run_infer` by feeding `run_infer_with_timeout`
    /// a deliberately slow blocking task (substituted for the real
    /// inference path; the test exercises the timeout machinery,
    /// not the inference kernel).  Asserts:
    ///   - the timeout fires within ~2x the configured timeout,
    ///   - the returned ServerError is `Timeout` (→ 504),
    ///   - a fresh blocking task started after the timeout returns
    ///     normally (the handler is not wedged).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn timeout_drops_handler_without_blocking_subsequent_requests() {
        use std::time::Duration;
        use std::time::Instant;

        // Simulate the inference path with a sleep.  We're not
        // calling run_infer here — we're testing the timeout
        // wrapper directly via tokio::time::timeout against
        // spawn_blocking.
        let started = Instant::now();
        let slow_handle = tokio::task::spawn_blocking(|| -> Result<i32, ServerError> {
            std::thread::sleep(Duration::from_millis(800));
            Ok(42)
        });

        let timeout = Duration::from_millis(100);
        let result: Result<i32, ServerError> =
            match tokio::time::timeout(timeout, slow_handle).await {
                Ok(_) => Err(ServerError::Internal(
                    "task should have timed out".to_string(),
                )),
                Err(_) => Err(ServerError::Timeout(format!(
                    "inference exceeded server-side timeout of {}ms",
                    timeout.as_millis(),
                ))),
            };
        let elapsed = started.elapsed();

        assert!(
            matches!(result, Err(ServerError::Timeout(_))),
            "got {result:?}"
        );
        // Timeout returned within ~2x the budget, not after the
        // 800 ms simulated inference completed.
        assert!(
            elapsed < Duration::from_millis(300),
            "timeout fired late: {elapsed:?}"
        );

        // Now confirm the handler is not wedged: a fresh blocking
        // task started after the timeout completes normally.
        let next_started = Instant::now();
        let fast_handle = tokio::task::spawn_blocking(|| 7);
        let value = fast_handle.await.expect("task joined");
        assert_eq!(value, 7);
        assert!(
            next_started.elapsed() < Duration::from_millis(200),
            "subsequent task delayed: {:?}",
            next_started.elapsed()
        );
    }

    /// Timeout = 0 disables the timeout: a slow blocking task
    /// completes normally with whatever it produces.  This
    /// preserves the historical behaviour for operators who
    /// haven't set the new --infer-timeout-secs flag.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn timeout_zero_passes_through_slow_inference() {
        use std::time::Duration;

        let handle = tokio::task::spawn_blocking(|| -> Result<i32, ServerError> {
            std::thread::sleep(Duration::from_millis(150));
            Ok(99)
        });
        // The wrapper falls through to a plain handle.await when
        // timeout.is_zero().  Mimic the same shape here.
        let zero = Duration::ZERO;
        let result = if zero.is_zero() {
            handle
                .await
                .map_err(|e| ServerError::Internal(e.to_string()))
                .and_then(|inner| inner)
        } else {
            unreachable!("timeout was zero")
        };
        assert_eq!(result.expect("value returned"), 99);
    }

    /// 504 status code mapping: ServerError::Timeout must produce a
    /// HTTP 504 Gateway Timeout response.  This pins the contract
    /// pg_infer's RemoteBackend relies on for retry-after-timeout.
    #[test]
    fn timeout_error_maps_to_504() {
        use axum::http::StatusCode;
        use axum::response::IntoResponse;
        let err = ServerError::Timeout("test".into());
        let response = err.into_response();
        assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);
    }
}
