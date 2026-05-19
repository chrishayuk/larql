//! HTTP server for the router: `AppState`, the `/v1/walk-ffn` handler, the
//! `/v1/stats` proxy, the `/v1/health` heartbeat, and the axum `Router`
//! factory. Moved out of `main.rs` so integration tests can build a Router
//! against a mock shard backend without spawning the binary.

use std::collections::HashMap;
use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::Response;
use axum::routing::{get, post};
use axum::{Json, Router};
use parking_lot::RwLock;
use serde_json::Value;

use crate::dispatch::{
    build_subrequest_body, group_layers_by_url, hedged_post_json, merge_shard_responses,
    resolve_static_only, unique_candidate_urls, HedgeOutcome,
};
use crate::grid::GridState;
use crate::metrics::{encode_metrics_text, RouterMetrics};
use crate::shards::{find_shard_for_layer, peek_binary, Shard};

/// Content-Type used by the FFN binary protocol. JSON requests use the
/// standard `application/json`.
pub const BINARY_CT: &str = "application/x-larql-ffn";

/// Shared HTTP service state. Holds the static shard map, an optional
/// grid handle, and a single reqwest client (whose connection pool is
/// reused across all outbound shard calls).
pub struct AppState {
    pub static_shards: Vec<Shard>,
    pub grid: Option<Arc<RwLock<GridState>>>,
    pub client: reqwest::Client,
    /// ADR-0017 — shared metrics registry. `None` disables
    /// observation (used by some integration tests that don't need
    /// the dependency); production paths always carry a value.
    pub metrics: Option<Arc<RouterMetrics>>,
    /// ADR-0019 — optional HTTP/3 shard transport. When `Some(...)`,
    /// the MoE expert fan-out path dispatches through h3 instead of
    /// reqwest. The dense path keeps reqwest unchanged because the
    /// HTTP/3 win (per-stream independence) only matters for
    /// parallel per-token fan-outs. Always `None` when the crate is
    /// built without the `http3` feature.
    #[cfg(feature = "http3")]
    pub h3_client: Option<Arc<larql_router_protocol::transport::h3::H3Client>>,
    /// ADR-0021 — hedged-dispatch delay. When `Some(d)`, the
    /// multi-shard fan-out picks a secondary replica per sub-request
    /// and dispatches it `d` after the primary if the primary hasn't
    /// responded yet. `None` disables hedging (pre-ADR-0021
    /// behaviour); operators opt in via `--hedge-after-ms M`.
    pub hedge_after: Option<std::time::Duration>,
}

impl AppState {
    /// Resolve every layer to its owning shard URL. Grid lookups take
    /// priority; any layer not covered by the grid falls back to the
    /// static shard map. Returns `Err(first uncovered layer)`.
    pub async fn resolve_all(
        &self,
        model_id: Option<&str>,
        layers: &[usize],
    ) -> Result<HashMap<usize, String>, usize> {
        if let Some(grid) = &self.grid {
            let guard = grid.read();
            let mut out = HashMap::with_capacity(layers.len());
            let mut static_needed: Vec<usize> = Vec::new();
            for &layer in layers {
                match guard.route(model_id, layer as u32) {
                    Some(url) => {
                        out.insert(layer, url);
                    }
                    None => static_needed.push(layer),
                }
            }
            drop(guard);
            for layer in static_needed {
                match find_shard_for_layer(&self.static_shards, layer) {
                    Some(s) => {
                        out.insert(layer, s.url.clone());
                    }
                    None => return Err(layer),
                }
            }
            return Ok(out);
        }
        resolve_static_only(&self.static_shards, layers)
    }
}

/// Build the axum `Router` for the public HTTP surface. Held separate
/// from the binary's `main()` so integration tests can mount it onto an
/// in-process listener.
pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/walk-ffn", post(handle_walk_ffn))
        .route("/v1/health", get(handle_health))
        .route("/v1/stats", get(handle_stats))
        .route("/metrics", get(handle_metrics))
        .with_state(state)
}

/// ADR-0017 — Prometheus text-format `/metrics` endpoint. Unauth,
/// same model as `/v1/health`. Returns 503 with a short body when
/// the router was built without a metrics registry (test harness).
pub async fn handle_metrics(State(state): State<Arc<AppState>>) -> Response {
    let Some(metrics) = &state.metrics else {
        return Response::builder()
            .status(StatusCode::SERVICE_UNAVAILABLE)
            .header(header::CONTENT_TYPE, "text/plain; charset=utf-8")
            .body("metrics registry not installed".to_string().into())
            .unwrap();
    };
    match encode_metrics_text(metrics) {
        Ok(text) => Response::builder()
            .status(StatusCode::OK)
            .header(
                header::CONTENT_TYPE,
                "text/plain; version=0.0.4; charset=utf-8",
            )
            .body(text.into())
            .unwrap(),
        Err(e) => Response::builder()
            .status(StatusCode::INTERNAL_SERVER_ERROR)
            .header(header::CONTENT_TYPE, "text/plain; charset=utf-8")
            .body(format!("metrics encode failed: {e}").into())
            .unwrap(),
    }
}

// ── Handlers ────────────────────────────────────────────────────────────────

/// Returns `true` when `ct` is the FFN binary protocol marker. Pure;
/// extracted so the binary-vs-JSON branch can be unit-tested without
/// building a full HTTP request.
pub fn is_binary_content_type(ct: &str) -> bool {
    ct.starts_with(BINARY_CT)
}

/// ADR-0018 — request shape after parsing. JSON bodies can be **dense**
/// (just `layer` / `layers`) or **MoE** (`experts` / `layer_experts`).
/// Binary bodies are always dense.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RequestSpec {
    /// Plain layer list — every owning shard runs the full layer's FFN.
    Dense(Vec<usize>),
    /// Per-layer expert list — the gate scorer upstream emits sparse
    /// `(layer, expert_ids)` pairs and the router dispatches to each
    /// expert-shard.
    Moe(Vec<(usize, Vec<u32>)>),
}

impl RequestSpec {
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Dense(layers) => layers.is_empty(),
            Self::Moe(pairs) => pairs.is_empty() || pairs.iter().all(|(_, exs)| exs.is_empty()),
        }
    }
}

/// Pull layer IDs and optional `model_id` out of a request body. For
/// binary bodies the header is peeked; for JSON bodies we look for a
/// `layers` array or a `layer` scalar plus an optional `model_id` field.
///
/// `Err(msg)` is returned to the caller as a 400 reply — the message is
/// already user-facing.
pub fn extract_layers_and_model_id(
    body: &[u8],
    is_binary: bool,
) -> Result<(Vec<usize>, Option<String>), String> {
    match extract_request_spec_and_model_id(body, is_binary)? {
        (RequestSpec::Dense(layers), model_id) => Ok((layers, model_id)),
        (RequestSpec::Moe(_), _) => Err(
            "MoE request (`experts`/`layer_experts`) is not accepted on the dense \
             code path; use `handle_walk_ffn`"
                .to_string(),
        ),
    }
}

/// ADR-0018 — dispatch-shape parser. Handles both dense and MoE JSON
/// shapes; binary bodies are always dense.
///
/// JSON MoE shapes (in priority order — first match wins):
///   - `{"layer_experts": [{"layer": L, "experts": [...]}, ...]}`
///   - `{"layer": L, "experts": [...]}`
///
/// JSON dense shapes (fallback):
///   - `{"layers": [...]}`
///   - `{"layer": L}`
///
/// `model_id` is optional in every shape.
pub fn extract_request_spec_and_model_id(
    body: &[u8],
    is_binary: bool,
) -> Result<(RequestSpec, Option<String>), String> {
    if is_binary {
        // ADR-0018: binary protocol stays dense-only. A future v2 wire
        // format with expert IDs is tracked under ADR-0009.
        let layers =
            peek_binary(body).ok_or_else(|| "binary: truncated or malformed header".to_string())?;
        return Ok((RequestSpec::Dense(layers), None));
    }

    let peek: Value = serde_json::from_slice(body).map_err(|e| format!("invalid JSON: {e}"))?;
    let model_id = peek
        .get("model_id")
        .and_then(|v| v.as_str())
        .map(str::to_owned);

    // MoE — multi-layer form takes priority over the single-layer form
    // because `layer_experts` is unambiguous.
    if let Some(arr) = peek.get("layer_experts").and_then(|v| v.as_array()) {
        let mut pairs: Vec<(usize, Vec<u32>)> = Vec::with_capacity(arr.len());
        for item in arr {
            let layer = item
                .get("layer")
                .and_then(|v| v.as_u64())
                .ok_or_else(|| "layer_experts: each entry needs a 'layer' field".to_string())?
                as usize;
            let experts_arr = item
                .get("experts")
                .and_then(|v| v.as_array())
                .ok_or_else(|| "layer_experts: each entry needs an 'experts' array".to_string())?;
            let experts: Vec<u32> = experts_arr
                .iter()
                .filter_map(|e| e.as_u64().map(|n| n as u32))
                .collect();
            if experts.is_empty() {
                return Err(format!(
                    "layer_experts: empty 'experts' array for layer {layer}"
                ));
            }
            pairs.push((layer, experts));
        }
        return Ok((RequestSpec::Moe(pairs), model_id));
    }

    // MoE — single-layer form.
    if let Some(arr) = peek.get("experts").and_then(|v| v.as_array()) {
        let layer = peek
            .get("layer")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| "moe: 'experts' requires a 'layer' scalar".to_string())?
            as usize;
        let experts: Vec<u32> = arr
            .iter()
            .filter_map(|e| e.as_u64().map(|n| n as u32))
            .collect();
        if experts.is_empty() {
            return Err("moe: 'experts' array is empty".into());
        }
        return Ok((RequestSpec::Moe(vec![(layer, experts)]), model_id));
    }

    // Dense fallback.
    let layers: Vec<usize> = if let Some(arr) = peek.get("layers").and_then(|v| v.as_array()) {
        arr.iter()
            .filter_map(|v| v.as_u64().map(|n| n as usize))
            .collect()
    } else if let Some(n) = peek.get("layer").and_then(|v| v.as_u64()) {
        vec![n as usize]
    } else {
        return Err("must provide 'layer' or 'layers'".into());
    };
    Ok((RequestSpec::Dense(layers), model_id))
}

/// `POST /v1/walk-ffn` entry point. Errors are normalised to JSON
/// regardless of the request content-type so clients always see the same
/// envelope.
pub async fn handle_walk_ffn(
    State(state): State<Arc<AppState>>,
    request: axum::extract::Request,
) -> Response {
    // ADR-0017 — observe duration + status. The timer starts before
    // the inner handler runs and stops on every exit path, including
    // early errors.
    let timer = state.metrics.as_ref().map(|m| {
        m.walk_ffn_duration_seconds
            .with_label_values(&[])
            .start_timer()
    });
    let result = handle_walk_ffn_inner(state.clone(), request).await;
    if let Some(t) = timer {
        t.observe_duration();
    }
    if let Some(m) = &state.metrics {
        let label = match &result {
            Ok(_) => "success",
            Err((status, _)) if status.is_client_error() => "error_4xx",
            Err(_) => "error_5xx",
        };
        m.walk_ffn_requests_total.with_label_values(&[label]).inc();
    }
    match result {
        Ok(r) => r,
        Err((status, msg)) => {
            let body = format!(r#"{{"error":{}}}"#, serde_json::Value::String(msg));
            let mut builder = Response::builder()
                .status(status)
                .header(header::CONTENT_TYPE, "application/json");
            // ADR-0020 — clients can use Retry-After to back off from
            // a saturated router rather than hammering it. 0.5s
            // matches the doc default; any 503 emitted by this
            // handler is currently saturation-driven.
            if status == StatusCode::SERVICE_UNAVAILABLE {
                builder = builder.header(header::RETRY_AFTER, "0.5");
            }
            builder.body(axum::body::Body::from(body)).unwrap()
        }
    }
}

async fn handle_walk_ffn_inner(
    state: Arc<AppState>,
    request: axum::extract::Request,
) -> Result<Response, (StatusCode, String)> {
    let is_binary = request
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(is_binary_content_type)
        .unwrap_or(false);

    let body_bytes = axum::body::to_bytes(request.into_body(), 64 * 1024 * 1024)
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("read body: {e}")))?;

    let (spec, model_id_owned) = extract_request_spec_and_model_id(&body_bytes, is_binary)
        .map_err(|m| (StatusCode::BAD_REQUEST, m))?;

    if spec.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "empty layer list".to_string()));
    }

    // ADR-0018 — MoE dispatch branches off here. Dense path continues
    // through the rest of the function unchanged.
    if let RequestSpec::Moe(pairs) = &spec {
        return handle_moe_dispatch(state, model_id_owned.as_deref(), pairs).await;
    }
    let layers: Vec<usize> = match spec {
        RequestSpec::Dense(l) => l,
        RequestSpec::Moe(_) => unreachable!("handled above"),
    };

    let mid = model_id_owned.as_deref();
    let layer_urls = match state.resolve_all(mid, &layers).await {
        Ok(map) => map,
        Err(missing) => {
            // ADR-0020 — distinguish "no shard owns this layer"
            // (400) from "shards own it but all are saturated"
            // (503). Saturation increments a counter so operators
            // can see the load-shedding signal.
            let saturated = match &state.grid {
                Some(grid) => grid.read().has_owners_for(mid, missing as u32),
                None => false,
            };
            if saturated {
                if let Some(m) = &state.metrics {
                    m.route_saturation_total.inc();
                }
                return Err((
                    StatusCode::SERVICE_UNAVAILABLE,
                    format!(
                        "layer {missing}: every replica is at or above the configured \
                         saturation ceiling — retry shortly"
                    ),
                ));
            }
            return Err((
                StatusCode::BAD_REQUEST,
                format!("layer {missing} has no owning shard in this router"),
            ));
        }
    };

    let unique_urls: std::collections::HashSet<&String> = layer_urls.values().collect();

    if unique_urls.len() == 1 || layers.len() == 1 {
        // All layers on the same shard — proxy raw bytes unchanged.
        let url = layer_urls.values().next().unwrap();
        let ct = if is_binary {
            BINARY_CT
        } else {
            "application/json"
        };
        return proxy_raw(&state.client, url, body_bytes, ct).await;
    }

    // Multi-shard dispatch.
    if is_binary {
        return Err((
            StatusCode::BAD_REQUEST,
            "binary fan-out across multiple shards is not supported; use JSON or split by shard"
                .to_string(),
        ));
    }

    let body_value: Value = serde_json::from_slice(&body_bytes)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("invalid JSON: {e}")))?;

    let by_url = group_layers_by_url(&layer_urls);

    // ADR-0021 — derive a secondary URL per primary group, if hedging
    // is enabled AND the grid actually offers a second replica. Static-
    // shard fall-back groups (no grid replica) get None and dispatch
    // through the non-hedged path.
    let hedge_after = state.hedge_after;
    let secondary_by_primary: HashMap<String, Option<String>> = if hedge_after.is_some() {
        let mut out = HashMap::with_capacity(by_url.len());
        if let Some(grid) = &state.grid {
            let guard = grid.read();
            for (primary, shard_layers) in &by_url {
                // Any layer in the group resolves the same replica set
                // (groups share a primary URL → same owning shard range).
                let probe_layer = *shard_layers.first().unwrap_or(&0) as u32;
                let ranked = guard.route_with_rank(mid, probe_layer, 2);
                // Pick the first ranked URL that isn't the primary —
                // route_with_rank's ordering can change between the
                // resolve_all snapshot and this read if a heartbeat
                // landed in between.
                let secondary = ranked.into_iter().find(|u| u != primary);
                out.insert(primary.clone(), secondary);
            }
        }
        out
    } else {
        HashMap::new()
    };

    let mut handles = Vec::new();
    for (url, shard_layers) in &by_url {
        let sub_body = build_subrequest_body(&body_value, shard_layers);
        let client = state.client.clone();
        let primary = url.clone();
        let secondary = secondary_by_primary.get(url).and_then(|s| s.clone());
        handles.push(tokio::spawn(async move {
            let (result, outcome) = hedged_post_json(
                &client,
                &primary,
                secondary.as_deref(),
                hedge_after,
                "/v1/walk-ffn",
                &sub_body,
            )
            .await;
            (result, outcome)
        }));
    }

    let joined: Vec<(Result<Value, String>, HedgeOutcome)> = futures::future::join_all(handles)
        .await
        .into_iter()
        .map(|jh| jh.unwrap_or_else(|e| (Err(e.to_string()), HedgeOutcome::default())))
        .collect();

    // Surface hedge outcomes to metrics before the early-return on
    // shard-error so even a failed hedge still increments the counter.
    if let Some(m) = &state.metrics {
        for (_, outcome) in &joined {
            if outcome.fired {
                m.route_hedge_fires_total.inc();
            }
            if outcome.won {
                m.route_hedge_wins_total.inc();
            }
        }
    }

    let responses: Vec<Value> = joined
        .into_iter()
        .map(|(result, _)| result)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("shard error: {e}")))?;

    let merged = merge_shard_responses(&responses);
    let json_bytes = serde_json::to_vec(&merged)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/json")
        .body(axum::body::Body::from(json_bytes))
        .unwrap())
}

/// ADR-0018 — MoE dispatch path. For each `(layer, [experts])` entry:
///
///   1. Resolve every `(layer, expert)` pair to its owning shard via
///      `GridState::route_all_experts`. Grid-only — MoE has no static
///      shard fallback.
///   2. Group the pairs by destination URL so each shard gets one
///      sub-request carrying every `(layer, expert)` it owns from this
///      call.
///   3. Build a JSON body per shard in the same `layer_experts` shape
///      the caller sent.
///   4. Fan out in parallel; merge responses with the existing
///      [`merge_shard_responses`] envelope.
///
/// Routing requires a live grid — MoE deployments never use static
/// `--shards`. If `state.grid` is `None` the handler 503s with a
/// helpful message.
async fn handle_moe_dispatch(
    state: Arc<AppState>,
    model_id: Option<&str>,
    pairs: &[(usize, Vec<u32>)],
) -> Result<Response, (StatusCode, String)> {
    let Some(grid) = &state.grid else {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "MoE routing requires a self-assembling grid (--grid-port); \
             this router was started in static --shards-only mode"
                .to_string(),
        ));
    };

    // Flatten the (layer, [experts]) list into individual (layer, expert)
    // pairs that route_all_experts can resolve.
    let flat: Vec<(usize, u32)> = pairs
        .iter()
        .flat_map(|(layer, experts)| experts.iter().map(move |&e| (*layer, e)))
        .collect();

    let layer_expert_urls = {
        let guard = grid.read();
        guard
            .route_all_experts(model_id, &flat)
            .map_err(|(layer, expert)| {
                (
                    StatusCode::SERVICE_UNAVAILABLE,
                    format!("no shard owns (layer {layer}, expert {expert}) in this router"),
                )
            })?
    };

    // Group (layer, expert) pairs by destination URL → per-shard
    // sub-request payload.
    let mut by_url: HashMap<String, HashMap<usize, Vec<u32>>> = HashMap::new();
    for ((layer, expert), url) in &layer_expert_urls {
        by_url
            .entry(url.clone())
            .or_default()
            .entry(*layer)
            .or_default()
            .push(*expert);
    }

    // ADR-0021 — derive a secondary URL per primary group when hedging
    // is enabled. Hedge only fires on the reqwest path; the h3 fan-out
    // already gets per-stream independence (ADR-0019), and a hedged h3
    // helper isn't in scope for this ADR.
    let hedge_after = state.hedge_after;
    let secondary_by_primary: HashMap<String, Option<String>> = if hedge_after.is_some() {
        let mut out = HashMap::with_capacity(by_url.len());
        let guard = grid.read();
        for (primary, layer_to_experts) in &by_url {
            // Any (layer, expert) in this group is owned by `primary`;
            // they share an owning replica set, so the first pair's
            // secondary is the secondary for the whole group.
            let (probe_layer, probe_expert) = layer_to_experts
                .iter()
                .next()
                .and_then(|(l, exs)| exs.first().map(|e| (*l as u32, *e)))
                .unwrap_or((0, 0));
            let ranked = guard.route_expert_with_rank(model_id, probe_layer, probe_expert, 2);
            let secondary = ranked.into_iter().find(|u| u != primary);
            out.insert(primary.clone(), secondary);
        }
        out
    } else {
        HashMap::new()
    };

    let mut handles = Vec::new();
    for (url, layer_to_experts) in by_url {
        let layer_experts_json: Vec<Value> = layer_to_experts
            .into_iter()
            .map(|(layer, mut experts)| {
                experts.sort_unstable();
                serde_json::json!({ "layer": layer, "experts": experts })
            })
            .collect();
        let mut sub_body = serde_json::Map::new();
        if let Some(mid) = model_id {
            sub_body.insert("model_id".into(), Value::String(mid.to_string()));
        }
        sub_body.insert("layer_experts".into(), Value::Array(layer_experts_json));
        let sub_body = Value::Object(sub_body);

        // ADR-0019 — when the operator opted into `--http3-shards`,
        // dispatch the MoE sub-request through h3 instead of
        // reqwest. h3 gives per-stream independence over QUIC, which
        // is the whole point: parallel per-token expert sub-requests
        // to the same shard stop blocking each other on TCP HoL.
        #[cfg(feature = "http3")]
        if let Some(h3) = state.h3_client.clone() {
            let h3_handle: tokio::task::JoinHandle<(Result<Value, String>, HedgeOutcome)> =
                tokio::spawn(async move {
                    let r = dispatch_via_h3(h3, url.clone(), sub_body).await;
                    (r, HedgeOutcome::default())
                });
            handles.push(h3_handle);
            continue;
        }

        let client = state.client.clone();
        let secondary = secondary_by_primary.get(&url).and_then(|s| s.clone());
        let primary = url.clone();
        handles.push(tokio::spawn(async move {
            hedged_post_json(
                &client,
                &primary,
                secondary.as_deref(),
                hedge_after,
                "/v1/walk-ffn",
                &sub_body,
            )
            .await
        }));
    }

    let mut responses = Vec::with_capacity(handles.len());
    for h in handles {
        match h.await {
            Ok((Ok(v), outcome)) => {
                if let Some(m) = &state.metrics {
                    if outcome.fired {
                        m.route_hedge_fires_total.inc();
                    }
                    if outcome.won {
                        m.route_hedge_wins_total.inc();
                    }
                }
                responses.push(v)
            }
            Ok((Err(e), outcome)) => {
                if let Some(m) = &state.metrics {
                    if outcome.fired {
                        m.route_hedge_fires_total.inc();
                    }
                    if outcome.won {
                        m.route_hedge_wins_total.inc();
                    }
                }
                return Err((
                    StatusCode::BAD_GATEWAY,
                    format!("MoE sub-request to shard failed: {e}"),
                ));
            }
            Err(e) => {
                return Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("MoE dispatch task join: {e}"),
                ))
            }
        }
    }

    let merged = merge_shard_responses(&responses);
    let body = serde_json::to_vec(&merged)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("encode: {e}")))?;
    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/json")
        .body(axum::body::Body::from(body))
        .unwrap())
}

/// ADR-0019 — issue one MoE sub-request via the HTTP/3 transport.
///
/// Parses the shard URL (`http://host:port` or `https://host:port`)
/// into the `(SocketAddr, server_name)` pair that
/// [`larql_router_protocol::transport::h3::H3Client::post_json`]
/// expects, serializes the JSON body, and returns the parsed
/// response. Feature-gated under `http3` so the dense build never
/// pays the h3 dispatch cost.
#[cfg(feature = "http3")]
async fn dispatch_via_h3(
    client: Arc<larql_router_protocol::transport::h3::H3Client>,
    shard_url: String,
    body: Value,
) -> Result<Value, String> {
    // `shard_url` looks like `http://10.0.0.11:8080` or `http://shard-a:8080`.
    // Strip scheme, split host:port, resolve to a SocketAddr. h3 ignores
    // the URL scheme; what matters is the UDP socket + SNI name.
    let trimmed = shard_url
        .trim_start_matches("https://")
        .trim_start_matches("http://")
        .trim_end_matches('/');
    let (host, port_str) = trimmed
        .rsplit_once(':')
        .ok_or_else(|| format!("shard URL {shard_url:?} missing :port"))?;
    let port: u16 = port_str
        .parse()
        .map_err(|e| format!("shard URL {shard_url:?} bad port: {e}"))?;

    use std::net::ToSocketAddrs;
    let addr = (host, port)
        .to_socket_addrs()
        .map_err(|e| format!("resolve {host}:{port}: {e}"))?
        .next()
        .ok_or_else(|| format!("no address for {host}:{port}"))?;

    let body_bytes = serde_json::to_vec(&body).map_err(|e| format!("encode body: {e}"))?;
    let resp = client
        .post_json(addr, host, "/v1/walk-ffn", body_bytes.into())
        .await
        .map_err(|e| format!("h3 post: {e}"))?;
    if resp.status >= 400 {
        return Err(format!("shard returned HTTP {}", resp.status));
    }
    serde_json::from_slice(&resp.body).map_err(|e| format!("decode response: {e}"))
}

/// Forward raw bytes to a shard, passing the Content-Type header through.
async fn proxy_raw(
    client: &reqwest::Client,
    base_url: &str,
    body: Bytes,
    ct: &str,
) -> Result<Response, (StatusCode, String)> {
    let url = format!("{base_url}/v1/walk-ffn");
    let resp = client
        .post(&url)
        .header(reqwest::header::CONTENT_TYPE, ct)
        .body(body.to_vec())
        .send()
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("shard {base_url}: {e}")))?;

    let status = resp.status();
    let resp_ct = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/json")
        .to_string();
    let resp_bytes = resp
        .bytes()
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("read shard response: {e}")))?;

    Ok(Response::builder()
        .status(status.as_u16())
        .header(header::CONTENT_TYPE, resp_ct)
        .body(axum::body::Body::from(resp_bytes))
        .unwrap())
}

pub async fn handle_health() -> Json<Value> {
    Json(serde_json::json!({"status": "ok"}))
}

/// Proxy `/v1/stats` to the first reachable shard so that clients
/// connecting via `RemoteWalkBackend` (which reads `hidden_size` from
/// `/v1/stats`) work transparently through the router.
pub async fn handle_stats(State(state): State<Arc<AppState>>) -> Response {
    let grid_urls = if let Some(grid) = &state.grid {
        grid.read().all_shard_urls()
    } else {
        Vec::new()
    };
    let candidates = unique_candidate_urls(grid_urls, &state.static_shards);
    for url in candidates {
        let stats_url = format!("{url}/v1/stats");
        if let Ok(resp) = state.client.get(&stats_url).send().await {
            if resp.status().is_success() {
                if let Ok(bytes) = resp.bytes().await {
                    return Response::builder()
                        .status(StatusCode::OK)
                        .header(header::CONTENT_TYPE, "application/json")
                        .body(axum::body::Body::from(bytes))
                        .unwrap();
                }
            }
        }
    }
    Response::builder()
        .status(StatusCode::SERVICE_UNAVAILABLE)
        .header(header::CONTENT_TYPE, "application/json")
        .body(axum::body::Body::from(r#"{"error":"no shard reachable"}"#))
        .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_binary_content_type_recognises_marker_prefix() {
        assert!(is_binary_content_type(BINARY_CT));
        assert!(is_binary_content_type(
            "application/x-larql-ffn; charset=utf-8"
        ));
        assert!(!is_binary_content_type("application/json"));
        assert!(!is_binary_content_type(""));
    }

    #[test]
    fn extract_layers_from_binary_body() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&42u32.to_le_bytes());
        let (layers, model) = extract_layers_and_model_id(&buf, true).unwrap();
        assert_eq!(layers, vec![42]);
        assert!(model.is_none());
    }

    #[test]
    fn extract_layers_binary_truncated_returns_err() {
        let err = extract_layers_and_model_id(&[], true).unwrap_err();
        assert!(err.contains("truncated"));
    }

    #[test]
    fn extract_layers_from_json_array() {
        let body = br#"{"layers":[0,1,2],"model_id":"gemma"}"#;
        let (layers, model) = extract_layers_and_model_id(body, false).unwrap();
        assert_eq!(layers, vec![0, 1, 2]);
        assert_eq!(model.as_deref(), Some("gemma"));
    }

    #[test]
    fn extract_layers_from_json_scalar() {
        let body = br#"{"layer":7}"#;
        let (layers, model) = extract_layers_and_model_id(body, false).unwrap();
        assert_eq!(layers, vec![7]);
        assert!(model.is_none());
    }

    #[test]
    fn extract_layers_json_missing_fields_errors() {
        let body = br#"{"foo":"bar"}"#;
        let err = extract_layers_and_model_id(body, false).unwrap_err();
        assert!(err.contains("must provide"));
    }

    #[test]
    fn extract_layers_invalid_json_errors() {
        let err = extract_layers_and_model_id(b"not json", false).unwrap_err();
        assert!(err.contains("invalid JSON"));
    }

    #[test]
    fn extract_layers_json_filters_non_numeric_entries() {
        let body = br#"{"layers":[0,"oops",2]}"#;
        let (layers, _) = extract_layers_and_model_id(body, false).unwrap();
        assert_eq!(layers, vec![0, 2]);
    }

    // ── ADR-0018: extract_request_spec_and_model_id ─────────────────────────

    #[test]
    fn extract_spec_dense_single_layer() {
        let body = br#"{"layer":3}"#;
        let (spec, model) = extract_request_spec_and_model_id(body, false).unwrap();
        assert_eq!(spec, RequestSpec::Dense(vec![3]));
        assert!(model.is_none());
    }

    #[test]
    fn extract_spec_dense_multi_layer() {
        let body = br#"{"layers":[0,1,2],"model_id":"m"}"#;
        let (spec, model) = extract_request_spec_and_model_id(body, false).unwrap();
        assert_eq!(spec, RequestSpec::Dense(vec![0, 1, 2]));
        assert_eq!(model.as_deref(), Some("m"));
    }

    #[test]
    fn extract_spec_moe_single_layer() {
        let body = br#"{"layer":5,"experts":[0,3,7]}"#;
        let (spec, model) = extract_request_spec_and_model_id(body, false).unwrap();
        assert_eq!(spec, RequestSpec::Moe(vec![(5, vec![0, 3, 7])]));
        assert!(model.is_none());
    }

    #[test]
    fn extract_spec_moe_multi_layer() {
        let body =
            br#"{"layer_experts":[{"layer":5,"experts":[0,3]},{"layer":6,"experts":[1,5]}]}"#;
        let (spec, _) = extract_request_spec_and_model_id(body, false).unwrap();
        assert_eq!(
            spec,
            RequestSpec::Moe(vec![(5, vec![0, 3]), (6, vec![1, 5])])
        );
    }

    #[test]
    fn extract_spec_moe_layer_experts_takes_priority_over_single_form() {
        // If both shapes are present, the multi-layer form wins.
        let body = br#"{"layer":99,"experts":[0],"layer_experts":[{"layer":5,"experts":[0,3]}]}"#;
        let (spec, _) = extract_request_spec_and_model_id(body, false).unwrap();
        assert_eq!(spec, RequestSpec::Moe(vec![(5, vec![0, 3])]));
    }

    #[test]
    fn extract_spec_moe_experts_without_layer_errors() {
        let body = br#"{"experts":[0,3]}"#;
        let err = extract_request_spec_and_model_id(body, false).unwrap_err();
        assert!(err.contains("requires a 'layer'"));
    }

    #[test]
    fn extract_spec_moe_empty_experts_errors() {
        let body = br#"{"layer":5,"experts":[]}"#;
        let err = extract_request_spec_and_model_id(body, false).unwrap_err();
        assert!(err.contains("empty"));
    }

    #[test]
    fn extract_spec_moe_layer_experts_missing_field_errors() {
        let body = br#"{"layer_experts":[{"layer":5}]}"#;
        let err = extract_request_spec_and_model_id(body, false).unwrap_err();
        assert!(err.contains("'experts' array"));
    }

    #[test]
    fn extract_spec_binary_is_always_dense() {
        // Binary bodies bypass JSON parsing entirely.
        // Encode "single layer 9": just 4 LE bytes of u32 = 9.
        let body = (9u32).to_le_bytes();
        let (spec, _) = extract_request_spec_and_model_id(&body, true).unwrap();
        assert_eq!(spec, RequestSpec::Dense(vec![9]));
    }

    #[test]
    fn request_spec_is_empty_branches() {
        assert!(RequestSpec::Dense(vec![]).is_empty());
        assert!(!RequestSpec::Dense(vec![0]).is_empty());
        assert!(RequestSpec::Moe(vec![]).is_empty());
        // All-empty experts → still treated as empty.
        assert!(RequestSpec::Moe(vec![(5, vec![])]).is_empty());
        assert!(!RequestSpec::Moe(vec![(5, vec![1])]).is_empty());
    }

    #[test]
    fn extract_layers_legacy_helper_rejects_moe_bodies() {
        // The dense-only wrapper around the new parser surfaces a clean
        // error for MoE bodies rather than silently accepting them.
        let body = br#"{"layer":5,"experts":[0]}"#;
        let err = extract_layers_and_model_id(body, false).unwrap_err();
        assert!(err.contains("MoE request"));
    }
}
