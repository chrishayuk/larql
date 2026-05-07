//! Grid state and gRPC service implementation for the self-assembling FFN grid.

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use tokio::sync::{mpsc, RwLock};
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use tonic::{Request, Response, Status, Streaming};

use larql_router_protocol::{
    AckMsg, AnnounceMsg, Gap, GridService, ModelCoverage, RejectMsg, RouterMessage, RouterPayload,
    ServerInfo, ServerMessage, ServerPayload, ShardInfo, StatusRequest, StatusResponse,
};

// ── Per-server record ─────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct ServerEntry {
    pub server_id: String,
    pub listen_url: String,
    pub model_id: String,
    pub layer_start: u32, // inclusive
    pub layer_end: u32,   // inclusive
    pub cpu_pct: f32,
    pub ram_used: u64,
    pub requests_in_flight: u32,
    pub last_seen: Instant,
    /// Set of capabilities this shard advertises. Backwards-compat:
    /// pre-`router-heterogeneous-shards` shards register without
    /// this field; they default to `{"attention", "expert"}` so they
    /// continue to receive every RPC. Real shards send their actual
    /// set via the announce proto extension landing with the
    /// `attention-service-routes` change.
    pub capabilities: Vec<String>,
}

impl ServerEntry {
    /// Default capability set used for shards that register without
    /// an explicit `capabilities` field. Equivalent to "this shard
    /// accepts any RPC".
    pub fn default_capabilities() -> Vec<String> {
        vec!["attention".to_string(), "expert".to_string()]
    }

    /// Whether this shard advertises `cap`. Case-insensitive.
    pub fn supports(&self, cap: &str) -> bool {
        self.capabilities
            .iter()
            .any(|c| c.eq_ignore_ascii_case(cap))
    }
}

// ── Grid state ────────────────────────────────────────────────────────────────

#[derive(Default)]
pub struct GridState {
    servers: HashMap<String, ServerEntry>,
    // Pre-built: (model_id, layer) → server_ids; rebuilt only on topology change.
    route_table: HashMap<(String, u32), Vec<String>>,
    // Pre-built: layer → server_ids for model_id=None (single-model) queries.
    any_model_table: HashMap<u32, Vec<String>>,
}

impl GridState {
    pub fn register(&mut self, entry: ServerEntry) {
        tracing::info!(
            server_id = %entry.server_id,
            listen_url = %entry.listen_url,
            model_id = %entry.model_id,
            layers = %format!("{}-{}", entry.layer_start, entry.layer_end),
            "Grid: server joined"
        );
        self.servers.insert(entry.server_id.clone(), entry);
        self.rebuild_route_table();
        self.log_coverage();
    }

    pub fn deregister(&mut self, server_id: &str) {
        if let Some(entry) = self.servers.remove(server_id) {
            tracing::info!(
                server_id = %server_id,
                model_id = %entry.model_id,
                layers = %format!("{}-{}", entry.layer_start, entry.layer_end),
                "Grid: server left"
            );
            self.rebuild_route_table();
            self.log_coverage();
        }
    }

    pub fn update_heartbeat(
        &mut self,
        server_id: &str,
        cpu_pct: f32,
        ram_used: u64,
        requests_in_flight: u32,
    ) {
        if let Some(entry) = self.servers.get_mut(server_id) {
            entry.cpu_pct = cpu_pct;
            entry.ram_used = ram_used;
            entry.requests_in_flight = requests_in_flight;
            entry.last_seen = Instant::now();
        }
        // Heartbeats don't change topology — no table rebuild needed.
    }

    /// Route one layer. O(1) table lookup + O(replicas) least-loaded scan.
    pub fn route(&self, model_id: Option<&str>, layer: u32) -> Option<String> {
        let ids = match model_id {
            Some(m) => self.route_table.get(&(m.to_owned(), layer)),
            None => self.any_model_table.get(&layer),
        };
        ids.and_then(|server_ids| {
            server_ids
                .iter()
                .filter_map(|id| self.servers.get(id))
                .min_by_key(|s| s.requests_in_flight)
                .map(|s| s.listen_url.clone())
        })
    }

    /// Route one layer to a shard that advertises `capability`.
    /// Lands as part of the `router-heterogeneous-shards` change —
    /// attention RPCs filter to capability="attention", expert RPCs
    /// to capability="expert". Returns `None` when no shard covers
    /// the layer with the requested capability.
    pub fn route_for_capability(
        &self,
        model_id: Option<&str>,
        layer: u32,
        capability: &str,
    ) -> Option<String> {
        let ids = match model_id {
            Some(m) => self.route_table.get(&(m.to_owned(), layer)),
            None => self.any_model_table.get(&layer),
        };
        ids.and_then(|server_ids| {
            server_ids
                .iter()
                .filter_map(|id| self.servers.get(id))
                .filter(|s| s.supports(capability))
                .min_by_key(|s| s.requests_in_flight)
                .map(|s| s.listen_url.clone())
        })
    }

    /// Resolve all layers in one call — one lock acquisition covers the whole batch.
    /// Returns Ok(layer → url) or Err(first layer with no owning shard).
    #[allow(dead_code)]
    pub fn route_all(
        &self,
        model_id: Option<&str>,
        layers: &[usize],
    ) -> Result<HashMap<usize, String>, usize> {
        let mut out = HashMap::with_capacity(layers.len());
        for &layer in layers {
            match self.route(model_id, layer as u32) {
                Some(url) => {
                    out.insert(layer, url);
                }
                None => return Err(layer),
            }
        }
        Ok(out)
    }

    /// Rebuild layer→servers index. Called only on join/leave (cold path).
    fn rebuild_route_table(&mut self) {
        let mut rt: HashMap<(String, u32), Vec<String>> = HashMap::new();
        let mut any: HashMap<u32, Vec<String>> = HashMap::new();
        for entry in self.servers.values() {
            for layer in entry.layer_start..=entry.layer_end {
                rt.entry((entry.model_id.clone(), layer))
                    .or_default()
                    .push(entry.server_id.clone());
                any.entry(layer).or_default().push(entry.server_id.clone());
            }
        }
        self.route_table = rt;
        self.any_model_table = any;
    }

    fn log_coverage(&self) {
        // Group by model_id
        let mut by_model: HashMap<&str, Vec<&ServerEntry>> = HashMap::new();
        for entry in self.servers.values() {
            by_model.entry(&entry.model_id).or_default().push(entry);
        }
        for (model_id, entries) in &by_model {
            let layer_count: u32 = entries
                .iter()
                .map(|e| e.layer_end - e.layer_start + 1)
                .sum();
            tracing::info!(
                model_id = model_id,
                servers = entries.len(),
                total_layers_covered = layer_count,
                "Grid coverage updated"
            );
        }
    }

    /// All distinct `listen_url` values across all registered servers.
    /// Used by the `/v1/stats` proxy to find a shard to forward to.
    pub fn all_shard_urls(&self) -> Vec<String> {
        let mut seen = std::collections::HashSet::new();
        self.servers
            .values()
            .filter_map(|s| {
                if seen.insert(s.listen_url.clone()) {
                    Some(s.listen_url.clone())
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn status_response(&self) -> StatusResponse {
        // Build per-model coverage
        let mut by_model: HashMap<String, Vec<&ServerEntry>> = HashMap::new();
        for entry in self.servers.values() {
            by_model
                .entry(entry.model_id.clone())
                .or_default()
                .push(entry);
        }

        let models: Vec<ModelCoverage> = by_model
            .iter()
            .map(|(model_id, entries)| {
                let mut shards: Vec<ShardInfo> = entries
                    .iter()
                    .map(|e| ShardInfo {
                        layer_start: e.layer_start,
                        layer_end: e.layer_end,
                        server_ids: vec![e.server_id.clone()],
                        replica_count: 1,
                    })
                    .collect();
                shards.sort_by_key(|s| s.layer_start);

                // Find gaps
                let mut gaps: Vec<Gap> = Vec::new();
                let mut prev_end: Option<u32> = None;
                for shard in &shards {
                    if let Some(end) = prev_end {
                        if shard.layer_start > end + 1 {
                            gaps.push(Gap {
                                layer_start: end + 1,
                                layer_end: shard.layer_start - 1,
                            });
                        }
                    }
                    prev_end = Some(shard.layer_end);
                }

                ModelCoverage {
                    model_id: model_id.clone(),
                    num_layers: 0, // not known to router without vindex
                    shards,
                    gaps,
                }
            })
            .collect();

        let servers: Vec<ServerInfo> = self
            .servers
            .values()
            .map(|e| ServerInfo {
                server_id: e.server_id.clone(),
                listen_url: e.listen_url.clone(),
                state: "serving".into(),
                model_id: e.model_id.clone(),
                layer_start: e.layer_start,
                layer_end: e.layer_end,
                cpu_pct: e.cpu_pct,
                ram_used: e.ram_used,
                requests_in_flight: e.requests_in_flight,
                rtt_ms: 0,
            })
            .collect();

        StatusResponse { models, servers }
    }
}

// ── gRPC service impl ─────────────────────────────────────────────────────────

pub struct GridServiceImpl {
    pub state: Arc<RwLock<GridState>>,
    next_id: AtomicU64,
    /// If set, every incoming Join stream must present "Authorization: Bearer <key>".
    grid_key: Option<String>,
}

impl GridServiceImpl {
    #[allow(dead_code)]
    pub fn new(state: Arc<RwLock<GridState>>) -> Self {
        Self {
            state,
            next_id: AtomicU64::new(1),
            grid_key: None,
        }
    }

    pub fn new_with_key(state: Arc<RwLock<GridState>>, key: Option<String>) -> Self {
        Self {
            state,
            next_id: AtomicU64::new(1),
            grid_key: key,
        }
    }

    fn alloc_server_id(&self) -> String {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let n = self.next_id.fetch_add(1, Ordering::Relaxed);
        format!("srv-{ts}-{n}")
    }
}

type JoinStream = Pin<Box<dyn futures_core::Stream<Item = Result<RouterMessage, Status>> + Send>>;

#[tonic::async_trait]
impl GridService for GridServiceImpl {
    type JoinStream = JoinStream;

    async fn join(
        &self,
        request: Request<Streaming<ServerMessage>>,
    ) -> Result<Response<Self::JoinStream>, Status> {
        // Auth check — reject streams that don't carry the correct grid key.
        if let Some(expected) = &self.grid_key {
            let token = request
                .metadata()
                .get("authorization")
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.strip_prefix("Bearer "));
            if token.map(|t| t != expected).unwrap_or(true) {
                return Err(Status::unauthenticated("invalid grid key"));
            }
        }

        let state = self.state.clone();
        let server_id = self.alloc_server_id();
        let (tx, rx) = mpsc::channel::<Result<RouterMessage, Status>>(32);
        let mut inbound = request.into_inner();

        let sid = server_id.clone();
        tokio::spawn(async move {
            let mut registered_model: Option<(String, u32, u32)> = None; // (model_id, start, end)

            while let Some(msg) = inbound.next().await {
                match msg {
                    Err(e) => {
                        tracing::warn!(server_id = %sid, "Stream error: {e}");
                        break;
                    }
                    Ok(ServerMessage { payload: None }) => {}
                    Ok(ServerMessage { payload: Some(p) }) => match p {
                        ServerPayload::Announce(AnnounceMsg {
                            model_id,
                            layer_start,
                            layer_end,
                            ram_bytes,
                            listen_url,
                            ..
                        }) => {
                            let entry = ServerEntry {
                                server_id: sid.clone(),
                                listen_url: listen_url.clone(),
                                model_id: model_id.clone(),
                                layer_start,
                                layer_end,
                                cpu_pct: 0.0,
                                ram_used: ram_bytes,
                                requests_in_flight: 0,
                                last_seen: Instant::now(),
                                // Pre-`router-heterogeneous-shards` proto
                                // doesn't carry a capabilities field on
                                // AnnounceMsg yet; default to "everything"
                                // so existing shards keep working.
                                capabilities: ServerEntry::default_capabilities(),
                            };
                            state.write().await.register(entry);
                            registered_model = Some((model_id, layer_start, layer_end));

                            let ack = RouterMessage {
                                payload: Some(RouterPayload::Ack(AckMsg {
                                    server_id: sid.clone(),
                                })),
                            };
                            if tx.send(Ok(ack)).await.is_err() {
                                break;
                            }
                        }

                        ServerPayload::Heartbeat(hb) => {
                            state.write().await.update_heartbeat(
                                &sid,
                                hb.cpu_pct,
                                hb.ram_used,
                                hb.requests_in_flight,
                            );
                        }

                        ServerPayload::Dropping(d) => {
                            tracing::info!(
                                server_id = %sid,
                                model_id = %d.model_id,
                                layers = %format!("{}-{}", d.layer_start, d.layer_end),
                                reason = %d.reason,
                                "Server dropping shard"
                            );
                            state.write().await.deregister(&sid);
                            registered_model = None;
                        }

                        ServerPayload::Available(_) => {
                            // Phase 2: Mode B assignment
                            tracing::info!(server_id = %sid, "Server is available (Mode B — not yet implemented)");
                            let reject = RouterMessage {
                                payload: Some(RouterPayload::Reject(RejectMsg {
                                    reason: "available mode not yet implemented".into(),
                                })),
                            };
                            let _ = tx.send(Ok(reject)).await;
                        }

                        ServerPayload::Ready(_) | ServerPayload::Refuse(_) => {
                            tracing::debug!(server_id = %sid, "Ignored message (not in assignment flow)");
                        }
                    },
                }
            }

            // Stream closed — clean up
            if registered_model.is_some() {
                state.write().await.deregister(&sid);
            }
            tracing::info!(server_id = %sid, "Connection closed");
        });

        let stream = ReceiverStream::new(rx);
        Ok(Response::new(Box::pin(stream)))
    }

    async fn status(
        &self,
        _request: Request<StatusRequest>,
    ) -> Result<Response<StatusResponse>, Status> {
        let resp = self.state.read().await.status_response();
        Ok(Response::new(resp))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(
        server_id: &str,
        listen_url: &str,
        model_id: &str,
        layer_start: u32,
        layer_end: u32,
    ) -> ServerEntry {
        ServerEntry {
            server_id: server_id.into(),
            listen_url: listen_url.into(),
            model_id: model_id.into(),
            layer_start,
            layer_end,
            cpu_pct: 0.0,
            ram_used: 1024,
            requests_in_flight: 0,
            last_seen: Instant::now(),
            capabilities: ServerEntry::default_capabilities(),
        }
    }

    #[test]
    fn route_uses_inclusive_layer_ranges() {
        let mut state = GridState::default();
        state.register(entry("a", "http://a", "model-a", 0, 2));
        state.register(entry("b", "http://b", "model-a", 3, 5));

        assert_eq!(state.route(Some("model-a"), 0).as_deref(), Some("http://a"));
        assert_eq!(state.route(Some("model-a"), 2).as_deref(), Some("http://a"));
        assert_eq!(state.route(Some("model-a"), 3).as_deref(), Some("http://b"));
        assert_eq!(state.route(Some("model-a"), 5).as_deref(), Some("http://b"));
        assert_eq!(state.route(Some("model-a"), 6), None);
    }

    #[test]
    fn route_without_model_uses_any_model_table() {
        let mut state = GridState::default();
        state.register(entry("a", "http://a", "model-a", 0, 1));

        assert_eq!(state.route(None, 1).as_deref(), Some("http://a"));
        assert_eq!(state.route(None, 2), None);
    }

    #[test]
    fn route_prefers_least_loaded_replica() {
        let mut state = GridState::default();
        let mut busy = entry("busy", "http://busy", "model-a", 0, 4);
        busy.requests_in_flight = 12;
        let mut idle = entry("idle", "http://idle", "model-a", 0, 4);
        idle.requests_in_flight = 1;

        state.register(busy);
        state.register(idle);

        assert_eq!(
            state.route(Some("model-a"), 3).as_deref(),
            Some("http://idle")
        );
    }

    #[test]
    fn deregister_removes_server_from_route_table() {
        let mut state = GridState::default();
        state.register(entry("a", "http://a", "model-a", 0, 2));
        state.register(entry("b", "http://b", "model-a", 3, 5));

        state.deregister("a");

        assert_eq!(state.route(Some("model-a"), 1), None);
        assert_eq!(state.route(Some("model-a"), 4).as_deref(), Some("http://b"));
    }

    #[test]
    fn heartbeat_updates_load_without_rebuilding_topology() {
        let mut state = GridState::default();
        state.register(entry("a", "http://a", "model-a", 0, 4));
        state.register(entry("b", "http://b", "model-a", 0, 4));

        state.update_heartbeat("a", 80.0, 2048, 20);
        state.update_heartbeat("b", 10.0, 1024, 0);

        assert_eq!(state.route(Some("model-a"), 2).as_deref(), Some("http://b"));
        let a = state.servers.get("a").unwrap();
        assert_eq!(a.cpu_pct, 80.0);
        assert_eq!(a.ram_used, 2048);
        assert_eq!(a.requests_in_flight, 20);
    }

    #[test]
    fn route_all_returns_first_uncovered_layer() {
        let mut state = GridState::default();
        state.register(entry("a", "http://a", "model-a", 0, 1));
        state.register(entry("b", "http://b", "model-a", 3, 4));

        assert_eq!(state.route_all(Some("model-a"), &[0, 1, 2, 3]), Err(2));
    }

    #[test]
    fn status_response_reports_shards_and_gaps() {
        let mut state = GridState::default();
        state.register(entry("a", "http://a", "model-a", 0, 1));
        state.register(entry("b", "http://b", "model-a", 3, 4));

        let status = state.status_response();

        assert_eq!(status.servers.len(), 2);
        assert_eq!(status.models.len(), 1);
        let model = &status.models[0];
        assert_eq!(model.model_id, "model-a");
        assert_eq!(model.shards.len(), 2);
        assert_eq!(model.gaps.len(), 1);
        assert_eq!(model.gaps[0].layer_start, 2);
        assert_eq!(model.gaps[0].layer_end, 2);
    }

    // ── router-heterogeneous-shards ──────────────────────────────────

    fn entry_with_caps(
        server_id: &str,
        listen_url: &str,
        model_id: &str,
        layer_start: u32,
        layer_end: u32,
        capabilities: &[&str],
    ) -> ServerEntry {
        let mut e = entry(server_id, listen_url, model_id, layer_start, layer_end);
        e.capabilities = capabilities.iter().map(|s| s.to_string()).collect();
        e
    }

    #[test]
    fn default_capabilities_advertise_both_attention_and_expert() {
        let e = entry("a", "http://a", "model-a", 0, 0);
        assert!(e.supports("attention"));
        assert!(e.supports("expert"));
        assert!(!e.supports("rotorquant"));
    }

    #[test]
    fn route_for_capability_filters_by_capability() {
        let mut state = GridState::default();
        state.register(entry_with_caps(
            "ffn", "http://ffn", "model-a", 0, 1, &["expert"],
        ));
        state.register(entry_with_caps(
            "gpu", "http://gpu", "model-a", 0, 1, &["attention"],
        ));

        assert_eq!(
            state.route_for_capability(Some("model-a"), 0, "attention"),
            Some("http://gpu".to_string())
        );
        assert_eq!(
            state.route_for_capability(Some("model-a"), 0, "expert"),
            Some("http://ffn".to_string())
        );
    }

    #[test]
    fn route_for_capability_returns_none_when_no_match() {
        let mut state = GridState::default();
        state.register(entry_with_caps(
            "ffn", "http://ffn", "model-a", 0, 1, &["expert"],
        ));

        assert_eq!(
            state.route_for_capability(Some("model-a"), 0, "attention"),
            None
        );
    }

    #[test]
    fn route_for_capability_falls_back_to_default_caps_shard() {
        // Pre-`router-heterogeneous-shards` shards have the default
        // capability set ({"attention", "expert"}); they should match
        // either capability filter.
        let mut state = GridState::default();
        state.register(entry("legacy", "http://legacy", "model-a", 0, 1));

        assert_eq!(
            state.route_for_capability(Some("model-a"), 0, "attention"),
            Some("http://legacy".to_string())
        );
        assert_eq!(
            state.route_for_capability(Some("model-a"), 0, "expert"),
            Some("http://legacy".to_string())
        );
    }
}
