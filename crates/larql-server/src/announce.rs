//! Grid announce task — keeps a persistent gRPC stream to the router.
//!
//! On startup, if --join is provided, this module spawns a background task
//! that connects to the router, sends an AnnounceMsg, and then sends
//! Heartbeats every 10 seconds. On disconnect it reconnects with backoff.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::Duration;

use std::sync::Arc;

use larql_router_protocol::{
    AnnounceMsg, DroppingMsg, GridServiceClient, HeartbeatMsg, RouterPayload, ServerMessage,
    ServerPayload,
};

use crate::metrics::LayerLatencyTracker;
use tokio_stream::StreamExt;
use tonic::metadata::AsciiMetadataValue;
use tracing::{error, info, warn};

// ── Tunables ───────────────────────────────────────────────────────────────────

const RECONNECT_INITIAL_BACKOFF: Duration = Duration::from_secs(1);
const RECONNECT_MAX_BACKOFF: Duration = Duration::from_secs(60);
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(10);
/// Maximum time to wait for in-flight requests to drain before sending DroppingMsg (GT6).
const DRAIN_TIMEOUT: Duration = Duration::from_secs(30);

// ── Config ─────────────────────────────────────────────────────────────────────

pub struct AnnounceConfig {
    /// gRPC endpoint of the router, e.g. "http://router:50052".
    pub join_url: String,
    /// Model identifier, e.g. "gemma3-4b-q4k".
    pub model_id: String,
    /// First owned layer (inclusive).
    pub layer_start: u32,
    /// Last owned layer (inclusive).
    pub layer_end: u32,
    /// URL clients should use to send requests here, e.g. "http://host:8080".
    pub listen_url: String,
    /// Approximate resident RAM for this shard in bytes.
    pub ram_bytes: u64,
    /// Shared secret that the router expects. None = open grid (dev only).
    pub grid_key: Option<String>,
    /// Stable identity hash of the vindex (model_id + num_layers).
    pub vindex_hash: String,
    /// Per-layer latency tracker — populates HeartbeatMsg.layer_stats.
    pub latency_tracker: Arc<LayerLatencyTracker>,
    /// Active request counter — used for drain (GT6) and heartbeat.requests_in_flight.
    pub requests_in_flight: Arc<std::sync::atomic::AtomicU32>,
    /// Cumulative request counter — diffed across the heartbeat interval
    /// to populate `HeartbeatMsg.req_per_sec`, which the router's
    /// hot-shard rebalancer reads to detect saturated shards.
    pub requests_total: Arc<std::sync::atomic::AtomicU64>,
    /// GT6: when set, after `UnassignMsg` + drain + `DroppingMsg`, the server
    /// re-enters Mode B on the same gRPC stream using this config so the
    /// router can immediately reassign it to a different gap.
    ///
    /// The reassignment downloads the new shard via `shard_loader` into
    /// `store_path`, but the running process does not hot-swap its loaded
    /// vindex — a deployer-side restart is required for the new shard to
    /// actually serve. The protocol round-trip nonetheless completes, which
    /// is what the rebalancer needs to see.
    pub available_after_drain: Option<AvailableConfig>,
    /// ADR-0010: SHA-256 fingerprint (hex) of the router's QUIC server
    /// cert. Required when `join_url` uses the `quic://` scheme; ignored
    /// otherwise. `None` together with a `quic://` URL means "skip cert
    /// verification" — LAN / dev only.
    pub quic_cert_fingerprint: Option<String>,
}

// ── Mode B config ──────────────────────────────────────────────────────────────

/// Config for Mode B announce: server has capacity but no shard loaded.
#[derive(Clone)]
pub struct AvailableConfig {
    pub join_url: String,
    pub listen_url: String,
    pub ram_bytes: u64,
    pub disk_bytes: u64,
    pub store_path: String,
    pub grid_key: Option<String>,
    /// ADR-0010: SHA-256 fingerprint (hex) of the router's QUIC server cert.
    pub quic_cert_fingerprint: Option<String>,
}

// ── Public entry points ────────────────────────────────────────────────────────

/// Spawn a background task that keeps the grid connection alive.
/// Returns immediately; the task runs for the process lifetime.
pub fn run_announce(config: AnnounceConfig) {
    tokio::spawn(async move {
        let mut backoff = RECONNECT_INITIAL_BACKOFF;
        loop {
            info!(
                join_url = %config.join_url,
                model_id = %config.model_id,
                layers = %format!("{}-{}", config.layer_start, config.layer_end),
                "Connecting to router grid..."
            );
            match try_once(&config).await {
                Ok(()) => {
                    info!("Grid stream closed cleanly — reconnecting");
                    backoff = RECONNECT_INITIAL_BACKOFF;
                }
                Err(e) => {
                    warn!(
                        "Grid stream error: {e} — retrying in {}s",
                        backoff.as_secs()
                    );
                    tokio::time::sleep(backoff).await;
                    backoff = (backoff * 2).min(RECONNECT_MAX_BACKOFF);
                }
            }
        }
    });
}

/// Spawn a Mode B background task: advertise available capacity, wait for
/// `AssignMsg`, download the assigned shard, send `ReadyMsg`, then
/// re-enter Mode A serving loop.
///
/// Returns immediately; the task runs for the process lifetime.
pub fn run_announce_available(config: AvailableConfig) {
    tokio::spawn(async move {
        let mut backoff = RECONNECT_INITIAL_BACKOFF;
        loop {
            info!(
                join_url = %config.join_url,
                ram_gb = config.ram_bytes / (1024 * 1024 * 1024),
                "Connecting to router grid (Mode B — available)..."
            );
            match try_once_available(&config).await {
                Ok(()) => {
                    info!("Mode B stream closed cleanly — reconnecting");
                    backoff = RECONNECT_INITIAL_BACKOFF;
                }
                Err(e) => {
                    warn!(
                        "Mode B stream error: {e} — retrying in {}s",
                        backoff.as_secs()
                    );
                    tokio::time::sleep(backoff).await;
                    backoff = (backoff * 2).min(RECONNECT_MAX_BACKOFF);
                }
            }
        }
    });
}

/// Build the `available_after_drain` config that lets a Mode A server
/// re-enter the available pool when the rebalancer asks it to drain.
///
/// Returns `None` when `--available-ram` is unset (the deployer hasn't
/// opted in) or when the RAM string fails to parse. `join_url` is left
/// empty here because bootstrap clones the config per router and fills
/// the field per iteration of its join-list loop.
pub fn build_available_after_drain(
    available_ram_bytes: Option<u64>,
    listen_url: &str,
    vindex_store: Option<&str>,
    grid_key: Option<&str>,
) -> Option<AvailableConfig> {
    let ram_bytes = available_ram_bytes?;
    let store_path = vindex_store
        .map(|s| s.to_string())
        .unwrap_or_else(|| "/tmp/larql-shards".to_string());
    Some(AvailableConfig {
        join_url: String::new(),
        listen_url: listen_url.to_string(),
        ram_bytes,
        disk_bytes: 0,
        store_path,
        grid_key: grid_key.map(str::to_string),
        quic_cert_fingerprint: None,
    })
}

/// Stable hash of the vindex identity (not a security primitive — for version checks).
pub fn vindex_identity_hash(model_id: &str, num_layers: usize) -> String {
    let mut h = DefaultHasher::new();
    model_id.hash(&mut h);
    num_layers.hash(&mut h);
    format!("{:016x}", h.finish())
}

fn grid_bearer_value(
    grid_key: Option<&str>,
) -> Result<Option<AsciiMetadataValue>, Box<dyn std::error::Error + Send + Sync>> {
    grid_key
        .map(|k| format!("Bearer {k}").parse())
        .transpose()
        .map_err(Into::into)
}

fn announce_message(cfg: &AnnounceConfig) -> ServerMessage {
    ServerMessage {
        payload: Some(ServerPayload::Announce(AnnounceMsg {
            model_id: cfg.model_id.clone(),
            layer_start: cfg.layer_start,
            layer_end: cfg.layer_end,
            ram_bytes: cfg.ram_bytes,
            listen_url: cfg.listen_url.clone(),
            vindex_hash: cfg.vindex_hash.clone(),
            // ADR-0018: dense by default. A future MoE-aware server
            // CLI flag will surface these.
            expert_start: 0,
            expert_end: 0,
        })),
    }
}

fn heartbeat_message(
    tracker: &LayerLatencyTracker,
    requests_in_flight: &std::sync::atomic::AtomicU32,
    requests_total: &std::sync::atomic::AtomicU64,
    last_total: &mut u64,
    interval: Duration,
) -> ServerMessage {
    use std::sync::atomic::Ordering;
    let now_total = requests_total.load(Ordering::Relaxed);
    // saturating_sub guards against a counter reset (defensive — the
    // counter is monotonic in practice; this just keeps the rate ≥ 0).
    let delta = now_total.saturating_sub(*last_total);
    *last_total = now_total;
    let secs = interval.as_secs_f32().max(f32::EPSILON);
    let req_per_sec = delta as f32 / secs;
    ServerMessage {
        payload: Some(ServerPayload::Heartbeat(HeartbeatMsg {
            cpu_pct: 0.0,
            ram_used: 0,
            requests_in_flight: requests_in_flight.load(Ordering::Relaxed),
            layer_stats: tracker.snapshot(),
            req_per_sec,
        })),
    }
}

fn dropping_message(model_id: String, layer_start: u32, layer_end: u32) -> ServerMessage {
    ServerMessage {
        payload: Some(ServerPayload::Dropping(DroppingMsg {
            model_id,
            layer_start,
            layer_end,
            reason: "reassigned".into(),
        })),
    }
}

/// Wait until `counter` reaches zero or `timeout` expires.
/// Polls every 100 ms. Used by GT6 drain to ensure no requests are
/// mid-flight before sending DroppingMsg.
async fn drain_requests(counter: &std::sync::atomic::AtomicU32, timeout: Duration) {
    use std::sync::atomic::Ordering;
    let deadline = tokio::time::Instant::now() + timeout;
    let mut interval = tokio::time::interval(std::time::Duration::from_millis(100));
    loop {
        if counter.load(Ordering::Acquire) == 0 {
            return;
        }
        if tokio::time::Instant::now() >= deadline {
            warn!(
                "drain timeout ({:.0}s) elapsed with {} requests still in flight",
                timeout.as_secs_f64(),
                counter.load(Ordering::Relaxed)
            );
            return;
        }
        interval.tick().await;
    }
}

// ── Single connection lifecycle ────────────────────────────────────────────────

/// Build a tonic `Channel` for `join_url`, picking the right transport
/// based on URL scheme. `quic://` uses the QUIC transport from
/// `larql-router-protocol` (when built with `--features quic`); anything
/// else falls through to the default TCP gRPC path.
async fn connect_grid_channel(
    join_url: &str,
    quic_cert_fingerprint: Option<&str>,
) -> Result<tonic::transport::Channel, Box<dyn std::error::Error + Send + Sync>> {
    if join_url.starts_with("quic://") {
        #[cfg(feature = "quic")]
        {
            use larql_router_protocol::transport::quic::{client_endpoint, connect_grpc_channel};

            // Parse "quic://host:port" → (host, SocketAddr). We strip the
            // scheme by hand because tonic's Uri parser rejects schemes it
            // doesn't recognise.
            let rest = &join_url["quic://".len()..];
            let host = rest
                .split(':')
                .next()
                .ok_or("quic:// URL missing host")?
                .to_string();
            let server_addr: std::net::SocketAddr = rest.parse().map_err(|e| {
                format!("quic:// URL must be host:port (resolved IPv4/IPv6), got {rest:?}: {e}")
            })?;
            let _ = rustls::crypto::ring::default_provider().install_default();
            let endpoint = client_endpoint(
                "0.0.0.0:0".parse().unwrap(),
                quic_cert_fingerprint.map(str::to_string),
            )
            .map_err(|e| format!("QUIC client endpoint: {e}"))?;
            let (_conn, channel) = connect_grpc_channel(&endpoint, server_addr, &host)
                .await
                .map_err(|e| format!("QUIC connect: {e}"))?;
            // _conn dropped at function exit would close the QUIC
            // connection, which would tear down our gRPC channel. Leak it
            // for the lifetime of the program — the announce task owns
            // the resulting Channel and only lives as long as the
            // connection stays usable. (This is the same lifetime
            // contract Channel::from_shared(...).connect() has on TCP.)
            std::mem::forget(_conn);
            Ok(channel)
        }
        #[cfg(not(feature = "quic"))]
        {
            let _ = quic_cert_fingerprint;
            Err(format!(
                "quic:// scheme requires building with --features quic (join_url = {join_url:?})"
            )
            .into())
        }
    } else {
        let channel = tonic::transport::Channel::from_shared(join_url.to_string())?
            .connect()
            .await?;
        Ok(channel)
    }
}

/// Run one announce connection to completion. Public so integration tests
/// can drive the real production flow against an in-process router. Production
/// code should use `run_announce` instead, which wraps this with reconnect.
#[doc(hidden)]
pub async fn try_once(
    cfg: &AnnounceConfig,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let channel = connect_grid_channel(&cfg.join_url, cfg.quic_cert_fingerprint.as_deref()).await?;

    // Inject the grid key into every outgoing RPC as "Authorization: Bearer <key>".
    let bearer = grid_bearer_value(cfg.grid_key.as_deref())?;
    let mut client =
        GridServiceClient::with_interceptor(channel, move |mut req: tonic::Request<()>| {
            if let Some(val) = &bearer {
                req.metadata_mut().insert("authorization", val.clone());
            }
            Ok(req)
        });

    // Channel for messages we send to the router.
    let (tx, rx) = tokio::sync::mpsc::channel::<ServerMessage>(32);
    let outbound = tokio_stream::wrappers::ReceiverStream::new(rx);

    let response = client.join(outbound).await?;
    let mut inbound = response.into_inner();

    // Send the announce message immediately.
    tx.send(announce_message(cfg)).await?;

    // Spawn the heartbeat sender.
    let tx_hb = tx.clone();
    let tracker = cfg.latency_tracker.clone();
    let rif = cfg.requests_in_flight.clone();
    let req_total = cfg.requests_total.clone();
    let hb_handle = tokio::spawn(async move {
        let mut interval = tokio::time::interval(HEARTBEAT_INTERVAL);
        // First tick fires immediately; seed the running total so the
        // initial heartbeat reports the rate over the first window
        // rather than counting all requests served before announce.
        let mut last_total = req_total.load(std::sync::atomic::Ordering::Relaxed);
        loop {
            interval.tick().await;
            let msg = heartbeat_message(
                &tracker,
                &rif,
                &req_total,
                &mut last_total,
                HEARTBEAT_INTERVAL,
            );
            if tx_hb.send(msg).await.is_err() {
                break;
            }
        }
    });

    // Process incoming router messages.
    while let Some(msg) = inbound.next().await {
        match msg {
            Err(e) => {
                hb_handle.abort();
                return Err(e.into());
            }
            Ok(rm) => match rm.payload {
                Some(RouterPayload::Ack(ack)) => {
                    info!(
                        server_id = %ack.server_id,
                        model_id = %cfg.model_id,
                        layers = %format!("{}-{}", cfg.layer_start, cfg.layer_end),
                        "Registered with router. Serving."
                    );
                }
                Some(RouterPayload::Reject(r)) => {
                    error!(reason = %r.reason, "Router rejected registration");
                    hb_handle.abort();
                    return Err(format!("router rejected: {}", r.reason).into());
                }
                Some(RouterPayload::Assign(a)) => {
                    // Mode A — this server is already serving a layer
                    // range. Mode B's `run_available_loop` is the path
                    // that processes AssignMsg (see announce.rs:509);
                    // arriving here means the router sent an
                    // assignment to a serving stream, which is a
                    // protocol-level error on the router side. We log
                    // loudly and ignore — the operator should see this
                    // in logs as a router bug, not silent misbehaviour.
                    warn!(
                        model_id = %a.model_id,
                        layers = %format!("{}-{}", a.layer_start, a.layer_end),
                        "Mode A stream received AssignMsg — ignoring (router bug? \
                         AssignMsg should target Mode B / available pool only)"
                    );
                }
                Some(RouterPayload::Unassign(u)) => {
                    info!(
                        model_id = %u.model_id,
                        layers = %format!("{}-{}", u.layer_start, u.layer_end),
                        reason = %u.reason,
                        "Router unassigned shard — draining in-flight requests…"
                    );
                    // GT6 drain: wait up to DRAIN_TIMEOUT for active requests
                    // to finish before sending DroppingMsg.
                    drain_requests(&cfg.requests_in_flight, DRAIN_TIMEOUT).await;
                    // Send dropping notice.
                    let _ = tx
                        .send(dropping_message(
                            u.model_id.clone(),
                            u.layer_start,
                            u.layer_end,
                        ))
                        .await;
                    hb_handle.abort();
                    // GT6 §Phase B2: if the deployer enabled drain-to-available,
                    // keep the stream open and re-enter Mode B so the router
                    // can immediately reassign this server.
                    if let Some(avail) = &cfg.available_after_drain {
                        info!("Drain complete — re-entering Mode B available pool on same stream");
                        return run_available_loop(tx, inbound, avail).await;
                    }
                    return Ok(());
                }
                None => {}
            },
        }
    }

    hb_handle.abort();
    Ok(())
}

// ── Mode B connection lifecycle ────────────────────────────────────────────────

/// One Mode B handshake against the router named by `cfg.join_url`:
/// opens a gRPC connection, sends `AvailableMsg`, then runs
/// [`run_available_loop`] to handle `AssignMsg` (download shard via
/// `shard_loader` + send `ReadyMsg`) until `AckMsg` lands or the
/// stream closes. Returns `Ok(())` after the spare has registered as
/// serving.
///
/// Public so integration tests can drive a full Mode B round-trip
/// against an in-process router fixture without resorting to manual
/// gRPC plumbing.
pub async fn try_once_available(
    cfg: &AvailableConfig,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let channel = connect_grid_channel(&cfg.join_url, cfg.quic_cert_fingerprint.as_deref()).await?;

    let bearer = grid_bearer_value(cfg.grid_key.as_deref())?;
    let mut client =
        GridServiceClient::with_interceptor(channel, move |mut req: tonic::Request<()>| {
            if let Some(val) = &bearer {
                req.metadata_mut().insert("authorization", val.clone());
            }
            Ok(req)
        });

    let (tx, rx) = tokio::sync::mpsc::channel::<ServerMessage>(32);
    let outbound = tokio_stream::wrappers::ReceiverStream::new(rx);
    let response = client.join(outbound).await?;
    let inbound = response.into_inner();

    run_available_loop(tx, inbound, cfg).await
}

/// Shared Mode B loop — used by `try_once_available` (fresh connection) and
/// by `try_once` after drain (re-enters Mode B on an existing connection).
///
/// Sends `AvailableMsg`, then handles incoming `AssignMsg` by downloading the
/// shard via `shard_loader` and acknowledging with `ReadyMsg` (or
/// `RefuseMsg` on download failure).
async fn run_available_loop(
    tx: tokio::sync::mpsc::Sender<ServerMessage>,
    mut inbound: tonic::Streaming<larql_router_protocol::RouterMessage>,
    cfg: &AvailableConfig,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use larql_router_protocol::{AvailableMsg, ReadyMsg, RefuseMsg};

    tx.send(ServerMessage {
        payload: Some(ServerPayload::Available(AvailableMsg {
            ram_bytes: cfg.ram_bytes,
            disk_bytes: cfg.disk_bytes,
            store_path: cfg.store_path.clone(),
        })),
    })
    .await?;

    info!(
        join_url = %cfg.join_url,
        ram_gb = cfg.ram_bytes / (1024 * 1024 * 1024),
        "Mode B: sent AvailableMsg — waiting for assignment…"
    );

    while let Some(msg) = inbound.next().await {
        match msg {
            Err(e) => return Err(e.into()),
            Ok(rm) => match rm.payload {
                Some(RouterPayload::Assign(assign)) => {
                    info!(
                        model_id = %assign.model_id,
                        layers = %format!("{}-{}", assign.layer_start, assign.layer_end),
                        origin_url = %assign.origin_url,
                        "Mode B: received AssignMsg — downloading shard…"
                    );

                    match crate::shard_loader::download_and_load_shard(
                        &assign.origin_url,
                        &cfg.store_path,
                        &assign.shard_hash,
                        &assign.model_id,
                        assign.layer_start,
                        assign.layer_end,
                    )
                    .await
                    {
                        Ok(()) => {
                            let _ = tx
                                .send(ServerMessage {
                                    payload: Some(ServerPayload::Ready(ReadyMsg {
                                        model_id: assign.model_id.clone(),
                                        layer_start: assign.layer_start,
                                        layer_end: assign.layer_end,
                                        listen_url: cfg.listen_url.clone(),
                                        expert_start: 0,
                                        expert_end: 0,
                                    })),
                                })
                                .await;
                            info!(
                                model_id = %assign.model_id,
                                "Mode B: shard ready — sent ReadyMsg"
                            );
                        }
                        Err(e) => {
                            warn!("Mode B: shard download failed: {e} — sending RefuseMsg");
                            let _ = tx
                                .send(ServerMessage {
                                    payload: Some(ServerPayload::Refuse(RefuseMsg {
                                        model_id: assign.model_id,
                                        layer_start: assign.layer_start,
                                        layer_end: assign.layer_end,
                                        reason: "download_failed".into(),
                                    })),
                                })
                                .await;
                        }
                    }
                }
                Some(RouterPayload::Ack(ack)) => {
                    info!(server_id = %ack.server_id, "Mode B: registered as serving");
                    break;
                }
                Some(RouterPayload::Reject(r)) => {
                    return Err(format!("Mode B rejected: {}", r.reason).into());
                }
                _ => {}
            },
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> AnnounceConfig {
        AnnounceConfig {
            join_url: "http://router:50052".into(),
            model_id: "gemma-test".into(),
            layer_start: 3,
            layer_end: 7,
            listen_url: "http://server:8080".into(),
            ram_bytes: 42,
            grid_key: Some("secret".into()),
            vindex_hash: "abc123".into(),
            latency_tracker: Arc::new(LayerLatencyTracker::new()),
            requests_in_flight: Arc::new(std::sync::atomic::AtomicU32::new(0)),
            requests_total: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            available_after_drain: None,
            quic_cert_fingerprint: None,
        }
    }

    #[test]
    fn vindex_identity_hash_is_stable_and_hex() {
        let a = vindex_identity_hash("model-a", 30);
        let b = vindex_identity_hash("model-a", 30);
        let c = vindex_identity_hash("model-a", 31);
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_eq!(a.len(), 16);
        assert!(a.chars().all(|ch| ch.is_ascii_hexdigit()));
    }

    #[test]
    fn grid_bearer_value_formats_authorization() {
        let val = grid_bearer_value(Some("secret")).unwrap().unwrap();
        assert_eq!(val.to_str().unwrap(), "Bearer secret");
        assert!(grid_bearer_value(None).unwrap().is_none());
    }

    #[test]
    fn announce_message_copies_config_fields() {
        let cfg = config();
        let msg = announce_message(&cfg);
        let Some(ServerPayload::Announce(announce)) = msg.payload else {
            panic!("expected announce payload");
        };
        assert_eq!(announce.model_id, "gemma-test");
        assert_eq!(announce.layer_start, 3);
        assert_eq!(announce.layer_end, 7);
        assert_eq!(announce.ram_bytes, 42);
        assert_eq!(announce.listen_url, "http://server:8080");
        assert_eq!(announce.vindex_hash, "abc123");
    }

    #[test]
    fn heartbeat_message_uses_zeroed_metrics() {
        let tracker = LayerLatencyTracker::new();
        let rif = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
        let total = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let mut last = 0u64;
        let msg = heartbeat_message(&tracker, &rif, &total, &mut last, Duration::from_secs(10));
        let Some(ServerPayload::Heartbeat(heartbeat)) = msg.payload else {
            panic!("expected heartbeat payload");
        };
        assert_eq!(heartbeat.cpu_pct, 0.0);
        assert_eq!(heartbeat.ram_used, 0);
        assert_eq!(heartbeat.requests_in_flight, 0);
        assert!(heartbeat.layer_stats.is_empty());
        assert_eq!(heartbeat.req_per_sec, 0.0);
    }

    #[test]
    fn heartbeat_includes_layer_stats_after_recording() {
        let tracker = LayerLatencyTracker::new();
        tracker.record(5, 3.0);
        tracker.record(5, 5.0);
        let rif = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
        let total = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let mut last = 0u64;
        let msg = heartbeat_message(&tracker, &rif, &total, &mut last, Duration::from_secs(10));
        let Some(ServerPayload::Heartbeat(hb)) = msg.payload else {
            panic!("expected heartbeat");
        };
        assert_eq!(hb.layer_stats.len(), 1);
        assert_eq!(hb.layer_stats[0].layer, 5);
        assert!(hb.layer_stats[0].avg_ms > 0.0);
    }

    #[test]
    fn heartbeat_computes_req_per_sec_from_counter_delta() {
        let tracker = LayerLatencyTracker::new();
        let rif = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
        let total = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let mut last = 0u64;
        // 50 requests over a 10 s interval = 5 req/s.
        total.store(50, std::sync::atomic::Ordering::Relaxed);
        let msg = heartbeat_message(&tracker, &rif, &total, &mut last, Duration::from_secs(10));
        let Some(ServerPayload::Heartbeat(hb)) = msg.payload else {
            panic!("expected heartbeat");
        };
        assert!(
            (hb.req_per_sec - 5.0).abs() < 0.001,
            "got {}",
            hb.req_per_sec
        );
        assert_eq!(last, 50, "last sample should advance");

        // Second sample: another 30 requests in the same window → 3 req/s.
        total.store(80, std::sync::atomic::Ordering::Relaxed);
        let msg2 = heartbeat_message(&tracker, &rif, &total, &mut last, Duration::from_secs(10));
        let Some(ServerPayload::Heartbeat(hb2)) = msg2.payload else {
            panic!("expected heartbeat");
        };
        assert!(
            (hb2.req_per_sec - 3.0).abs() < 0.001,
            "got {}",
            hb2.req_per_sec
        );
    }

    #[test]
    fn heartbeat_rate_clamps_to_zero_on_counter_reset() {
        // saturating_sub guards against a counter going backwards. The
        // counter is monotonic in production; this just prevents an
        // underflow spike if a deployer pulls the rug.
        let tracker = LayerLatencyTracker::new();
        let rif = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
        let total = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let mut last = 100u64;
        let msg = heartbeat_message(&tracker, &rif, &total, &mut last, Duration::from_secs(10));
        let Some(ServerPayload::Heartbeat(hb)) = msg.payload else {
            panic!("expected heartbeat");
        };
        assert_eq!(hb.req_per_sec, 0.0);
        assert_eq!(last, 0, "last sample should track the reset counter");
    }

    #[test]
    fn build_available_after_drain_returns_none_without_ram() {
        assert!(build_available_after_drain(None, "http://srv", None, None).is_none());
    }

    #[test]
    fn build_available_after_drain_uses_default_store_path() {
        let cfg =
            build_available_after_drain(Some(8 * 1024 * 1024 * 1024), "http://srv", None, None)
                .expect("ram set should produce a config");
        assert_eq!(cfg.ram_bytes, 8 * 1024 * 1024 * 1024);
        assert_eq!(cfg.listen_url, "http://srv");
        assert_eq!(cfg.store_path, "/tmp/larql-shards");
        assert!(cfg.join_url.is_empty(), "filled per-router by bootstrap");
        assert!(cfg.grid_key.is_none());
    }

    #[test]
    fn build_available_after_drain_passes_through_overrides() {
        let cfg =
            build_available_after_drain(Some(1), "http://srv", Some("/mnt/shards"), Some("secret"))
                .unwrap();
        assert_eq!(cfg.store_path, "/mnt/shards");
        assert_eq!(cfg.grid_key.as_deref(), Some("secret"));
    }

    #[test]
    fn dropping_message_marks_reassigned() {
        let msg = dropping_message("model".into(), 1, 2);
        let Some(ServerPayload::Dropping(dropping)) = msg.payload else {
            panic!("expected dropping payload");
        };
        assert_eq!(dropping.model_id, "model");
        assert_eq!(dropping.layer_start, 1);
        assert_eq!(dropping.layer_end, 2);
        assert_eq!(dropping.reason, "reassigned");
    }
}
