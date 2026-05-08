//! larql-router — transparent layer-sharding proxy for larql-server.
//!
//! Two dispatch modes:
//!   --shards  "0-16=http://host-a:8080,17-33=http://host-b:8081"
//!             Static shard map (ADR-0003, backwards-compatible).
//!   --grid-port 50052
//!             Self-assembling grid (ADR-0004). Servers connect via gRPC
//!             and announce their capabilities. No static configuration.
//!
//! Both modes can coexist. Grid takes priority; static shards are fallback.
//!
//! # Wire format
//!
//! The router is wire-transparent for both JSON (`application/json`) and binary
//! (`application/x-larql-ffn`) requests. For single-shard routes the body is
//! forwarded byte-for-byte with no intermediate parsing. Multi-shard fan-out
//! is supported for JSON only; binary multi-shard requests are rejected with
//! HTTP 400 (use the batched JSON format or route per-shard manually).

mod grid;

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::Response;
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::Parser;
use serde_json::Value;
use tokio::sync::RwLock;
use tonic::transport::Server as GrpcServer;
use tracing::{info, warn};

use grid::{GridServiceImpl, GridState};
use larql_router_protocol::GridServiceServer;

// ── Binary wire format constants ───────────────────────────────────────────────

const BINARY_CT: &str = "application/x-larql-ffn";
const BATCH_MARKER: u32 = 0xFFFF_FFFF;

// ── CLI ────────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(
    name = "larql-router",
    version,
    about = "Layer-sharding proxy for larql-server"
)]
struct Cli {
    /// Static shard map: comma-separated "START-END=URL" entries (inclusive bounds).
    /// Example: "0-16=http://host-a:8080,17-33=http://host-b:8081"
    /// Optional when --grid-port is provided.
    #[arg(long)]
    shards: Option<String>,

    /// Enable the self-assembling grid gRPC server on this port.
    /// Servers connect here with --join grpc://router:PORT.
    #[arg(long)]
    grid_port: Option<u16>,

    /// HTTP listen port.
    #[arg(long, default_value = "9090")]
    port: u16,

    /// Bind address.
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    /// Per-hop deadline to backend shards, in seconds.
    #[arg(long, env = "LARQL_ROUTER_HOP_DEADLINE_SECS", default_value = "5")]
    timeout_secs: u64,

    /// Log level.
    #[arg(long, default_value = "info")]
    log_level: String,

    /// Shared secret for the self-assembling grid.
    /// Servers must pass the same key via --grid-key to be accepted.
    /// If not set, the grid port is open to any server (development only).
    #[arg(long, env = "LARQL_GRID_KEY")]
    grid_key: Option<String>,
}

// ── Static shard map ───────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
struct Shard {
    layer_start: usize, // inclusive
    layer_end: usize,   // exclusive
    url: String,
}

impl Shard {
    fn owns(&self, layer: usize) -> bool {
        layer >= self.layer_start && layer < self.layer_end
    }
}

fn parse_shards(spec: &str) -> Result<Vec<Shard>, String> {
    let mut shards = Vec::new();
    for entry in spec.split(',') {
        let entry = entry.trim();
        if entry.is_empty() {
            continue;
        }
        let (range, url) = entry
            .split_once('=')
            .ok_or_else(|| format!("expected 'START-END=URL', got '{entry}'"))?;
        let (start_s, end_s) = range
            .split_once('-')
            .ok_or_else(|| format!("expected 'START-END', got '{range}'"))?;
        let start: usize = start_s
            .trim()
            .parse()
            .map_err(|_| format!("invalid start '{start_s}'"))?;
        let end: usize = end_s
            .trim()
            .parse()
            .map_err(|_| format!("invalid end '{end_s}'"))?;
        if end < start {
            return Err(format!("end ({end}) must be >= start ({start})"));
        }
        shards.push(Shard {
            layer_start: start,
            layer_end: end + 1,
            url: url.trim().to_string(),
        });
    }
    if shards.is_empty() {
        return Err("no shards specified".into());
    }
    Ok(shards)
}

// ── Binary routing ─────────────────────────────────────────────────────────────

/// Extract layer indices from a binary request body without parsing the residual.
///
/// Returns `None` if the header is malformed or truncated.
pub(crate) fn peek_binary(body: &[u8]) -> Option<Vec<usize>> {
    if body.len() < 4 {
        return None;
    }
    let first = u32::from_le_bytes(body[0..4].try_into().ok()?);
    if first == BATCH_MARKER {
        if body.len() < 8 {
            return None;
        }
        let n = u32::from_le_bytes(body[4..8].try_into().ok()?) as usize;
        let needed = 8 + n * 4;
        if body.len() < needed {
            return None;
        }
        let layers = (0..n)
            .map(|i| u32::from_le_bytes(body[8 + i * 4..12 + i * 4].try_into().unwrap()) as usize)
            .collect();
        Some(layers)
    } else {
        Some(vec![first as usize])
    }
}

// ── App state ──────────────────────────────────────────────────────────────────

struct AppState {
    /// Static shards from --shards (may be empty).
    static_shards: Vec<Shard>,
    /// Grid state from --grid-port (None if grid mode not enabled).
    grid: Option<Arc<RwLock<GridState>>>,
    client: reqwest::Client,
}

#[derive(Debug)]
struct RouteMiss {
    layer: usize,
    capability: &'static str,
}

impl AppState {
    /// Resolve all layers in one lock acquisition.
    /// Returns Ok(layer → url) or Err(first missing layer).
    async fn resolve_all(
        &self,
        model_id: Option<&str>,
        layers: &[usize],
        capability: &'static str,
    ) -> Result<HashMap<usize, String>, RouteMiss> {
        if let Some(grid) = &self.grid {
            let guard = grid.read().await;
            let mut out = HashMap::with_capacity(layers.len());
            let mut static_needed: Vec<usize> = Vec::new();
            for &layer in layers {
                match guard.route_for_capability(model_id, layer as u32, capability) {
                    Some(url) => {
                        out.insert(layer, url);
                    }
                    None => static_needed.push(layer),
                }
            }
            drop(guard);
            for layer in static_needed {
                match self
                    .static_shards
                    .iter()
                    .find(|s| capability == "expert" && s.owns(layer))
                {
                    Some(s) => {
                        out.insert(layer, s.url.clone());
                    }
                    None => return Err(RouteMiss { layer, capability }),
                }
            }
            return Ok(out);
        }
        let mut out = HashMap::with_capacity(layers.len());
        for &layer in layers {
            match self
                .static_shards
                .iter()
                .find(|s| capability == "expert" && s.owns(layer))
            {
                Some(s) => {
                    out.insert(layer, s.url.clone());
                }
                None => return Err(RouteMiss { layer, capability }),
            }
        }
        Ok(out)
    }
}

// ── Route handler ──────────────────────────────────────────────────────────────

async fn handle_walk_ffn(
    State(state): State<Arc<AppState>>,
    request: axum::extract::Request,
) -> Response {
    match handle_walk_ffn_inner(state, request).await {
        Ok(r) => r,
        Err((status, msg)) => {
            // Always return errors as JSON regardless of input content-type.
            let body = format!(r#"{{"error":{}}}"#, serde_json::Value::String(msg));
            Response::builder()
                .status(status)
                .header(header::CONTENT_TYPE, "application/json")
                .body(axum::body::Body::from(body))
                .unwrap()
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
        .map(|ct| ct.starts_with(BINARY_CT))
        .unwrap_or(false);

    let body_bytes = axum::body::to_bytes(request.into_body(), 64 * 1024 * 1024)
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("read body: {e}")))?;

    let (layers, model_id_owned): (Vec<usize>, Option<String>) = if is_binary {
        let layers = peek_binary(&body_bytes).ok_or_else(|| {
            (
                StatusCode::BAD_REQUEST,
                "binary: truncated or malformed header".to_string(),
            )
        })?;
        (layers, None)
    } else {
        let peek: Value = serde_json::from_slice(&body_bytes)
            .map_err(|e| (StatusCode::BAD_REQUEST, format!("invalid JSON: {e}")))?;
        let layers: Vec<usize> = if let Some(arr) = peek.get("layers").and_then(|v| v.as_array()) {
            arr.iter()
                .filter_map(|v| v.as_u64().map(|n| n as usize))
                .collect()
        } else if let Some(n) = peek.get("layer").and_then(|v| v.as_u64()) {
            vec![n as usize]
        } else {
            return Err((
                StatusCode::BAD_REQUEST,
                "must provide 'layer' or 'layers'".to_string(),
            ));
        };
        let model_id = peek
            .get("model_id")
            .and_then(|v| v.as_str())
            .map(str::to_owned);
        (layers, model_id)
    };

    if layers.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "empty layer list".to_string()));
    }

    let mid = model_id_owned.as_deref();
    let layer_urls = state
        .resolve_all(mid, &layers, "expert")
        .await
        .map_err(|missing| {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                format!(
                    "no shard advertises capability={} for layer {}",
                    missing.capability, missing.layer
                ),
            )
        })?;

    proxy_walk_ffn_layers(state, body_bytes, is_binary, layer_urls, layers).await
}

async fn handle_attention_proxy(
    State(state): State<Arc<AppState>>,
    request: axum::extract::Request,
) -> Response {
    match handle_attention_proxy_inner(state, request).await {
        Ok(r) => r,
        Err((status, msg)) => {
            let body = format!(r#"{{"error":{}}}"#, serde_json::Value::String(msg));
            Response::builder()
                .status(status)
                .header(header::CONTENT_TYPE, "application/json")
                .body(axum::body::Body::from(body))
                .unwrap()
        }
    }
}

async fn handle_attention_proxy_inner(
    state: Arc<AppState>,
    request: axum::extract::Request,
) -> Result<Response, (StatusCode, String)> {
    let path = request.uri().path().to_string();
    let content_type = request
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/json")
        .to_string();
    let method = request.method().clone();
    let body_bytes = axum::body::to_bytes(request.into_body(), 64 * 1024 * 1024)
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("read body: {e}")))?;

    let (layers, model_id_owned) = if body_bytes.is_empty() {
        (vec![0usize], None)
    } else {
        let peek: Value = serde_json::from_slice(&body_bytes)
            .map_err(|e| (StatusCode::BAD_REQUEST, format!("invalid JSON: {e}")))?;
        let layers = layers_from_json(&peek).unwrap_or_else(|| vec![0usize]);
        let model_id = peek
            .get("model_id")
            .and_then(|v| v.as_str())
            .map(str::to_owned);
        (layers, model_id)
    };

    let layer_urls = state
        .resolve_all(model_id_owned.as_deref(), &layers, "attention")
        .await
        .map_err(|missing| {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                format!(
                    "no shard advertises capability={} for layer {}",
                    missing.capability, missing.layer
                ),
            )
        })?;

    let unique_urls: std::collections::HashSet<&String> = layer_urls.values().collect();
    if unique_urls.len() != 1 {
        return Err((
            StatusCode::BAD_REQUEST,
            "attention fan-out across multiple shards is not supported".to_string(),
        ));
    }
    let base_url = layer_urls.values().next().unwrap();
    proxy_raw_to_path(
        &state.client,
        base_url,
        &path,
        method,
        body_bytes,
        &content_type,
    )
    .await
}

fn layers_from_json(peek: &Value) -> Option<Vec<usize>> {
    if let Some(arr) = peek.get("layers").and_then(|v| v.as_array()) {
        Some(
            arr.iter()
                .filter_map(|v| v.as_u64().map(|n| n as usize))
                .collect(),
        )
    } else if let Some(n) = peek.get("layer").and_then(|v| v.as_u64()) {
        Some(vec![n as usize])
    } else if let Some(arr) = peek.get("layer_range").and_then(|v| v.as_array()) {
        arr.first()
            .and_then(|v| v.as_u64())
            .map(|start| vec![start as usize])
    } else {
        None
    }
}

async fn proxy_walk_ffn_layers(
    state: Arc<AppState>,
    body_bytes: Bytes,
    is_binary: bool,
    layer_urls: HashMap<usize, String>,
    layers: Vec<usize>,
) -> Result<Response, (StatusCode, String)> {
    // Determine unique shards.
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

    // JSON fan-out: group layers by URL, dispatch in parallel, merge.
    let body_value: Value = serde_json::from_slice(&body_bytes)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("invalid JSON: {e}")))?;

    let mut by_url: HashMap<String, Vec<usize>> = HashMap::new();
    for (&layer, url) in &layer_urls {
        by_url.entry(url.clone()).or_default().push(layer);
    }

    let mut handles = Vec::new();
    for (url, shard_layers) in &by_url {
        let mut sub_body = body_value.clone();
        if shard_layers.len() == 1 {
            sub_body["layer"] = Value::from(shard_layers[0]);
            sub_body.as_object_mut().unwrap().remove("layers");
        } else {
            sub_body["layers"] =
                Value::Array(shard_layers.iter().map(|&l| Value::from(l)).collect());
            sub_body.as_object_mut().unwrap().remove("layer");
        }
        let client = state.client.clone();
        let target = format!("{url}/v1/walk-ffn");
        handles.push(tokio::spawn(async move {
            let resp = client
                .post(&target)
                .json(&sub_body)
                .send()
                .await
                .map_err(|e| (reqwest_error_status(&e), e.to_string()))?;
            resp.json::<Value>()
                .await
                .map_err(|e| (reqwest_error_status(&e), e.to_string()))
        }));
    }

    let responses: Vec<Value> = futures::future::join_all(handles)
        .await
        .into_iter()
        .map(|jh| {
            jh.map_err(|e| (StatusCode::BAD_GATEWAY, e.to_string()))
                .and_then(|r| r)
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|(status, e)| (status, format!("shard error: {e}")))?;

    let mut all_results: Vec<Value> = Vec::new();
    let mut max_latency: f64 = 0.0;
    for resp in responses {
        if let Some(arr) = resp.get("results").and_then(|v| v.as_array()) {
            all_results.extend(arr.iter().cloned());
        } else if resp.get("layer").is_some() {
            all_results.push(resp.clone());
        }
        if let Some(ms) = resp.get("latency_ms").and_then(|v| v.as_f64()) {
            if ms > max_latency {
                max_latency = ms;
            }
        }
    }
    all_results.sort_by_key(|r| r.get("layer").and_then(|v| v.as_u64()).unwrap_or(0));

    let merged = serde_json::json!({
        "results": all_results,
        "latency_ms": (max_latency * 10.0).round() / 10.0,
    });
    let json_bytes = serde_json::to_vec(&merged)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/json")
        .body(axum::body::Body::from(json_bytes))
        .unwrap())
}

/// Forward raw bytes to a shard, passing the Content-Type header through.
/// The shard's response status and Content-Type are preserved unchanged.
async fn proxy_raw(
    client: &reqwest::Client,
    base_url: &str,
    body: Bytes,
    ct: &str,
) -> Result<Response, (StatusCode, String)> {
    proxy_raw_to_path(
        client,
        base_url,
        "/v1/walk-ffn",
        axum::http::Method::POST,
        body,
        ct,
    )
    .await
}

async fn proxy_raw_to_path(
    client: &reqwest::Client,
    base_url: &str,
    path: &str,
    method: axum::http::Method,
    body: Bytes,
    ct: &str,
) -> Result<Response, (StatusCode, String)> {
    let url = format!("{base_url}{path}");
    let reqwest_method =
        reqwest::Method::from_bytes(method.as_str().as_bytes()).unwrap_or(reqwest::Method::POST);
    let resp = client
        .request(reqwest_method, &url)
        .header(reqwest::header::CONTENT_TYPE, ct)
        .body(body.to_vec())
        .send()
        .await
        .map_err(|e| (reqwest_error_status(&e), format!("shard {base_url}: {e}")))?;

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

fn reqwest_error_status(e: &reqwest::Error) -> StatusCode {
    if e.is_timeout() {
        StatusCode::GATEWAY_TIMEOUT
    } else {
        StatusCode::BAD_GATEWAY
    }
}

async fn handle_health() -> Json<Value> {
    Json(serde_json::json!({"status": "ok"}))
}

/// Proxy /v1/stats to the first reachable shard so that clients connecting
/// via RemoteWalkBackend (which reads hidden_size from /v1/stats) work
/// transparently through the router.
async fn handle_stats(State(state): State<Arc<AppState>>) -> Response {
    // Collect candidate shard URLs: grid shards first, then static.
    let mut candidates: Vec<String> = Vec::new();
    if let Some(grid) = &state.grid {
        let guard = grid.read().await;
        for url in guard.all_shard_urls() {
            candidates.push(url);
        }
    }
    for shard in &state.static_shards {
        if !candidates.contains(&shard.url) {
            candidates.push(shard.url.clone());
        }
    }
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
    // No shard reachable — return minimal synthetic stats so callers don't fail hard.
    Response::builder()
        .status(StatusCode::SERVICE_UNAVAILABLE)
        .header(header::CONTENT_TYPE, "application/json")
        .body(axum::body::Body::from(r#"{"error":"no shard reachable"}"#))
        .unwrap()
}

// ── Main ───────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Accept both `larql-router <args>` and `larql-router route <args>`.
    let args: Vec<String> = std::env::args().collect();
    let filtered: Vec<String> = if args.len() > 1 && args[1] == "route" {
        std::iter::once(args[0].clone())
            .chain(args[2..].iter().cloned())
            .collect()
    } else {
        args
    };
    let cli = Cli::parse_from(filtered);

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(&cli.log_level)),
        )
        .init();

    info!("larql-router v{}", env!("CARGO_PKG_VERSION"));

    if cli.shards.is_none() && cli.grid_port.is_none() {
        eprintln!("error: must provide --shards or --grid-port (or both)");
        std::process::exit(1);
    }

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(cli.timeout_secs))
        .tcp_keepalive(std::time::Duration::from_secs(30))
        .pool_idle_timeout(std::time::Duration::from_secs(90))
        .pool_max_idle_per_host(16)
        .build()?;

    let static_shards = if let Some(spec) = &cli.shards {
        let shards = parse_shards(spec).map_err(|e| format!("--shards: {e}"))?;
        info!("Static shard map:");
        for shard in &shards {
            let status_url = format!("{}/v1/stats", shard.url);
            let healthy = client
                .get(&status_url)
                .send()
                .await
                .map(|r| r.status().is_success())
                .unwrap_or(false);
            let marker = if healthy { "✓" } else { "✗ UNREACHABLE" };
            info!(
                "  layers {}-{}: {}  {}",
                shard.layer_start,
                shard.layer_end - 1,
                shard.url,
                marker
            );
            if !healthy {
                warn!("  Shard {} is not reachable", shard.url);
            }
        }
        shards
    } else {
        Vec::new()
    };

    let grid_state: Option<Arc<RwLock<GridState>>> = if cli.grid_port.is_some() {
        Some(Arc::new(RwLock::new(GridState::default())))
    } else {
        None
    };

    if let (Some(grid_port), Some(state)) = (cli.grid_port, &grid_state) {
        let svc = GridServiceServer::new(GridServiceImpl::new_with_key(
            state.clone(),
            cli.grid_key.clone(),
        ));
        let grpc_addr: SocketAddr = format!("{}:{}", cli.host, grid_port).parse()?;
        info!("Grid gRPC server listening: {grpc_addr}");
        tokio::spawn(async move {
            if let Err(e) = GrpcServer::builder()
                .add_service(svc)
                .serve(grpc_addr)
                .await
            {
                tracing::error!("gRPC server error: {e}");
            }
        });
    }

    let state = Arc::new(AppState {
        static_shards,
        grid: grid_state,
        client,
    });

    let app = Router::new()
        .route("/v1/walk-ffn", post(handle_walk_ffn))
        .route("/v1/attention/session", post(handle_attention_proxy))
        .route(
            "/v1/attention/session/{id}",
            get(handle_attention_proxy).delete(handle_attention_proxy),
        )
        .route("/v1/attention/prefill", post(handle_attention_proxy))
        .route("/v1/attention/decode", post(handle_attention_proxy))
        .route("/v1/kv-cache/snapshot", post(handle_attention_proxy))
        .route("/v1/kv-cache/restore", post(handle_attention_proxy))
        .route("/v1/kv-cache/free", post(handle_attention_proxy))
        .route("/v1/stats", get(handle_stats))
        .route("/v1/health", get(handle_health))
        .with_state(state);

    let addr = format!("{}:{}", cli.host, cli.port);
    info!("HTTP listening: http://{}", addr);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

// ══════════════════════════════════════════════════════════════════════════════
// Tests
// ══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    // ── peek_binary ───────────────────────────────────────────────────────────

    fn make_binary_single(layer: u32, residual_floats: usize) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&layer.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes()); // seq_len
        buf.extend_from_slice(&1u32.to_le_bytes()); // flags (full_output)
        buf.extend_from_slice(&8092u32.to_le_bytes()); // top_k
        buf.extend(std::iter::repeat_n(0u8, residual_floats * 4));
        buf
    }

    fn make_binary_batch(layers: &[u32], residual_floats: usize) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&BATCH_MARKER.to_le_bytes());
        buf.extend_from_slice(&(layers.len() as u32).to_le_bytes());
        for &l in layers {
            buf.extend_from_slice(&l.to_le_bytes());
        }
        buf.extend_from_slice(&1u32.to_le_bytes()); // seq_len
        buf.extend_from_slice(&1u32.to_le_bytes()); // flags
        buf.extend_from_slice(&8092u32.to_le_bytes()); // top_k
        buf.extend(std::iter::repeat_n(0u8, residual_floats * 4));
        buf
    }

    #[test]
    fn peek_binary_single_layer() {
        let body = make_binary_single(5, 4);
        let layers = peek_binary(&body).unwrap();
        assert_eq!(layers, vec![5]);
    }

    #[test]
    fn peek_binary_batch_layers() {
        let body = make_binary_batch(&[5, 20, 30], 4);
        let layers = peek_binary(&body).unwrap();
        assert_eq!(layers, vec![5, 20, 30]);
    }

    #[test]
    fn peek_binary_empty_body_returns_none() {
        assert!(peek_binary(&[]).is_none());
    }

    #[test]
    fn peek_binary_truncated_single_returns_value() {
        // Only 4 bytes — enough for a single-layer marker.
        let buf = 7u32.to_le_bytes();
        let layers = peek_binary(&buf).unwrap();
        assert_eq!(layers, vec![7]);
    }

    #[test]
    fn peek_binary_batch_truncated_layer_list_returns_none() {
        // Claims 10 layers but only provides 2 u32s after num_layers.
        let mut buf = Vec::new();
        buf.extend_from_slice(&BATCH_MARKER.to_le_bytes());
        buf.extend_from_slice(&10u32.to_le_bytes()); // num_layers = 10
        buf.extend_from_slice(&0u32.to_le_bytes()); // layer 0
        buf.extend_from_slice(&1u32.to_le_bytes()); // layer 1 — only 2 of 10
        assert!(peek_binary(&buf).is_none());
    }

    #[test]
    fn peek_binary_zero_batch_layers() {
        let body = make_binary_batch(&[], 4);
        let layers = peek_binary(&body).unwrap();
        assert!(layers.is_empty());
    }

    // ── parse_shards ──────────────────────────────────────────────────────────

    #[test]
    fn parse_shards_single_entry() {
        let shards = parse_shards("0-16=http://host-a:8080").unwrap();
        assert_eq!(shards.len(), 1);
        assert_eq!(shards[0].layer_start, 0);
        assert_eq!(shards[0].layer_end, 17); // exclusive
        assert_eq!(shards[0].url, "http://host-a:8080");
    }

    #[test]
    fn parse_shards_two_entries() {
        let shards = parse_shards("0-16=http://host-a:8080,17-33=http://host-b:8081").unwrap();
        assert_eq!(shards.len(), 2);
        assert!(shards[0].owns(0));
        assert!(shards[0].owns(16));
        assert!(!shards[0].owns(17));
        assert!(shards[1].owns(17));
        assert!(shards[1].owns(33));
    }

    #[test]
    fn parse_shards_empty_string_errors() {
        assert!(parse_shards("").is_err());
    }

    #[test]
    fn parse_shards_missing_url_errors() {
        assert!(parse_shards("0-16").is_err());
    }

    #[test]
    fn parse_shards_end_less_than_start_errors() {
        assert!(parse_shards("16-0=http://host:8080").is_err());
    }

    #[test]
    fn parse_shards_ignores_trailing_comma() {
        let shards = parse_shards("0-16=http://host:8080,").unwrap();
        assert_eq!(shards.len(), 1);
    }

    #[test]
    fn shard_owns_inclusive_bounds() {
        let shards = parse_shards("0-16=http://host:8080").unwrap();
        assert!(shards[0].owns(0));
        assert!(shards[0].owns(16));
        assert!(!shards[0].owns(17));
    }

    #[test]
    fn cli_default_hop_deadline_is_five_seconds() {
        let cli = Cli::parse_from(["larql-router", "--shards", "0-0=http://host:8080"]);
        assert_eq!(cli.timeout_secs, 5);

        let cli = Cli::parse_from([
            "larql-router",
            "--shards",
            "0-0=http://host:8080",
            "--timeout-secs",
            "9",
        ]);
        assert_eq!(cli.timeout_secs, 9);
    }

    fn grid_entry(
        server_id: &str,
        listen_url: &str,
        model_id: &str,
        layer_start: u32,
        layer_end: u32,
        capabilities: &[&str],
    ) -> grid::ServerEntry {
        grid::ServerEntry {
            server_id: server_id.to_string(),
            listen_url: listen_url.to_string(),
            model_id: model_id.to_string(),
            layer_start,
            layer_end,
            cpu_pct: 0.0,
            ram_used: 1024,
            requests_in_flight: 0,
            last_seen: Instant::now(),
            capabilities: capabilities.iter().map(|s| s.to_string()).collect(),
            cached_prefixes: grid::PrefixBloom::new(),
        }
    }

    fn test_state(grid_state: GridState) -> Arc<AppState> {
        Arc::new(AppState {
            static_shards: Vec::new(),
            grid: Some(Arc::new(RwLock::new(grid_state))),
            client: reqwest::Client::new(),
        })
    }

    #[tokio::test]
    async fn resolve_all_filters_grid_by_expert_capability() {
        let mut grid = GridState::default();
        grid.register(grid_entry(
            "gpu",
            "http://gpu",
            "model-a",
            0,
            31,
            &["attention"],
        ));
        grid.register(grid_entry(
            "ffn",
            "http://ffn",
            "model-a",
            0,
            31,
            &["expert"],
        ));
        let state = test_state(grid);

        let resolved = state
            .resolve_all(Some("model-a"), &[17], "expert")
            .await
            .expect("expert shard resolves");
        assert_eq!(resolved.get(&17).map(String::as_str), Some("http://ffn"));
    }

    #[tokio::test]
    async fn attention_proxy_missing_capability_returns_503_body() {
        let mut grid = GridState::default();
        grid.register(grid_entry(
            "ffn",
            "http://ffn",
            "model-a",
            0,
            31,
            &["expert"],
        ));
        let state = test_state(grid);
        let req = axum::http::Request::builder()
            .method("POST")
            .uri("/v1/attention/decode")
            .header(header::CONTENT_TYPE, "application/json")
            .body(axum::body::Body::from(
                r#"{"model_id":"model-a","layer":17,"session_id":"s"}"#,
            ))
            .unwrap();

        let resp = handle_attention_proxy(State(state), req).await;
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = axum::body::to_bytes(resp.into_body(), 1024)
            .await
            .expect("body bytes");
        let body = std::str::from_utf8(&body).unwrap();
        assert!(body.contains("no shard advertises capability=attention"));
    }

    #[tokio::test]
    async fn proxy_raw_timeout_maps_to_504() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            if let Ok((_socket, _peer)) = listener.accept().await {
                tokio::time::sleep(std::time::Duration::from_millis(200)).await;
            }
        });

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_millis(20))
            .build()
            .unwrap();
        let err = proxy_raw_to_path(
            &client,
            &format!("http://{addr}"),
            "/v1/attention/decode",
            axum::http::Method::POST,
            Bytes::from_static(b"{}"),
            "application/json",
        )
        .await
        .expect_err("timeout should map to router error");

        assert_eq!(err.0, StatusCode::GATEWAY_TIMEOUT);
    }
}
