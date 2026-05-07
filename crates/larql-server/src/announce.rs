//! Grid announce task — keeps a persistent gRPC stream to the router.
//!
//! On startup, if --join is provided, this module spawns a background task
//! that connects to the router, sends an AnnounceMsg, and then sends
//! Heartbeats every 10 seconds. On disconnect it reconnects with backoff.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::Duration;

use larql_router_protocol::{
    AnnounceMsg, DroppingMsg, GridServiceClient, HeartbeatMsg, RouterPayload, ServerMessage,
    ServerPayload,
};
use tokio_stream::StreamExt;
use tonic::metadata::AsciiMetadataValue;
use tracing::{error, info, warn};

// ── Tunables ───────────────────────────────────────────────────────────────────

const RECONNECT_INITIAL_BACKOFF: Duration = Duration::from_secs(1);
const RECONNECT_MAX_BACKOFF: Duration = Duration::from_secs(60);
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(10);

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
    /// Capabilities to advertise on the announce. Empty ⇒ router
    /// defaults to `["attention", "expert"]` (legacy shard).
    /// attention-service-routes change.
    pub capabilities: Vec<String>,
    /// Closure invoked on every heartbeat tick to compute the
    /// shard's current `cached_prefixes` bloom payload (32 bytes
    /// raw `PrefixBloom`). `None` ⇒ heartbeat omits the field.
    /// attention-service-routes change.
    #[allow(clippy::type_complexity)]
    pub cached_prefixes_provider: Option<std::sync::Arc<dyn Fn() -> Option<Vec<u8>> + Send + Sync>>,
}

// ── Public entry point ─────────────────────────────────────────────────────────

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
            capabilities: cfg.capabilities.clone(),
        })),
    }
}

fn heartbeat_message(cached_prefixes: Vec<u8>) -> ServerMessage {
    ServerMessage {
        payload: Some(ServerPayload::Heartbeat(HeartbeatMsg {
            cpu_pct: 0.0,
            ram_used: 0,
            requests_in_flight: 0,
            cached_prefixes,
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

// ── Single connection lifecycle ────────────────────────────────────────────────

async fn try_once(cfg: &AnnounceConfig) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let channel = tonic::transport::Channel::from_shared(cfg.join_url.clone())?
        .connect()
        .await?;

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
    let provider = cfg.cached_prefixes_provider.clone();
    let hb_handle = tokio::spawn(async move {
        let mut interval = tokio::time::interval(HEARTBEAT_INTERVAL);
        loop {
            interval.tick().await;
            let bloom = provider.as_ref().and_then(|f| f()).unwrap_or_default();
            if tx_hb.send(heartbeat_message(bloom)).await.is_err() {
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
                Some(RouterPayload::Assign(_)) => {
                    warn!("Received AssignMsg but Mode B not implemented — ignoring");
                }
                Some(RouterPayload::Unassign(u)) => {
                    info!(
                        model_id = %u.model_id,
                        layers = %format!("{}-{}", u.layer_start, u.layer_end),
                        reason = %u.reason,
                        "Router unassigned shard"
                    );
                    // Send dropping notice then let the stream close.
                    let _ = tx
                        .send(dropping_message(
                            u.model_id.clone(),
                            u.layer_start,
                            u.layer_end,
                        ))
                        .await;
                    break;
                }
                None => {}
            },
        }
    }

    hb_handle.abort();
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
            capabilities: Vec::new(),
            cached_prefixes_provider: None,
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
        let msg = heartbeat_message(Vec::new());
        let Some(ServerPayload::Heartbeat(heartbeat)) = msg.payload else {
            panic!("expected heartbeat payload");
        };
        assert_eq!(heartbeat.cpu_pct, 0.0);
        assert_eq!(heartbeat.ram_used, 0);
        assert_eq!(heartbeat.requests_in_flight, 0);
        assert!(heartbeat.cached_prefixes.is_empty());
    }

    #[test]
    fn heartbeat_message_carries_cached_prefixes() {
        let bloom = vec![0xAB; 32];
        let msg = heartbeat_message(bloom.clone());
        let Some(ServerPayload::Heartbeat(heartbeat)) = msg.payload else {
            panic!("expected heartbeat payload");
        };
        assert_eq!(heartbeat.cached_prefixes, bloom);
    }

    #[test]
    fn announce_message_carries_capabilities() {
        let mut cfg = config();
        cfg.capabilities = vec!["attention".into()];
        let msg = announce_message(&cfg);
        let Some(ServerPayload::Announce(a)) = msg.payload else {
            panic!("expected announce payload");
        };
        assert_eq!(a.capabilities, vec!["attention".to_string()]);
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
