//! Grid state and gRPC service implementation for the self-assembling FFN grid.
//!
//! The grid is split across this folder:
//!
//! - `mod.rs` (this file) — the core data types ([`GridState`],
//!   [`ServerEntry`], [`AvailableEntry`]), state mutators
//!   (register / deregister / heartbeat updates), and the gRPC
//!   [`GridServiceImpl`].
//! - [`routing`] — `route()` / `route_all()` hot path, the
//!   three-tier comparator, and the cold-path route-table rebuild.
//! - [`replication`] — under/over-replication detection,
//!   `effective_target_for`, gap-fill and `AssignMsg` dispatch
//!   into the Mode B available pool.
//! - [`hot_shard`] — `req/sec` saturation detection and the
//!   elevation set that lifts a hot range's effective replica target.
//! - [`status`] — read-only observation surface (`coverage_gaps`,
//!   `all_shard_urls`, `status_response`).
//! - [`service`] — gRPC `GridService` impl + admin RPCs
//!   (drain_server, assign_range).

pub mod hot_shard;
pub mod replication;
pub mod routing;
pub mod service;
pub mod status;

#[cfg(test)]
pub(crate) mod testing;

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use tokio::sync::mpsc;

use larql_router_protocol::{LayerLatency, RouterMessage};

// ── Per-server record ─────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct ServerEntry {
    pub server_id: String,
    pub listen_url: String,
    pub model_id: String,
    pub layer_start: u32, // inclusive
    pub layer_end: u32,   // inclusive
    /// `vindex_hash` from `AnnounceMsg`. Used as the `shard_hash` when this
    /// server is selected as a Mode B origin for a different replica.
    pub vindex_hash: String,
    pub cpu_pct: f32,
    pub ram_used: u64,
    pub requests_in_flight: u32,
    pub last_seen: Instant,
    /// Per-layer EMA latency and p99, from HeartbeatMsg.layer_stats (GT3).
    /// Key = layer index. Empty until the first heartbeat with layer data arrives.
    pub layer_latencies: HashMap<u32, (f32, f32)>, // (avg_ms, p99_ms)
    /// Shard-scoped request rate (requests/sec) from the most recent
    /// heartbeat. Drives the hot-shard rebalancer tick.
    pub req_per_sec: f32,
    /// Active-probe wire RTT in ms (router → server). Populated by the
    /// optional probe loop spawned when `--rtt-probe-interval-secs > 0`.
    /// `None` until the first probe completes. Used by `route()` as a
    /// tie-breaker when no GT3 per-layer latency data is available yet.
    pub rtt_ms: Option<f32>,
    /// ADR-0018 — MoE expert ownership. Both `0` means the server is
    /// dense (every layer monolithic). Both nonzero advertises the
    /// contiguous expert ID range this server owns across its layer
    /// range. Use [`Self::is_dense`] / [`Self::owns_expert`] rather
    /// than reading these directly.
    pub expert_start: u32, // inclusive (meaningful only when !is_dense)
    pub expert_end: u32, // inclusive (meaningful only when !is_dense)
}

impl ServerEntry {
    /// `true` when this server has no expert-level ownership — every
    /// covered layer is monolithic. The proto3 default of 0/0 maps to
    /// this case so old dense announces work unchanged.
    pub fn is_dense(&self) -> bool {
        self.expert_start == 0 && self.expert_end == 0
    }

    /// `true` if this server owns `expert_id` within its expert range,
    /// OR if the server is dense (a dense server owns every expert
    /// trivially because there is no expert dimension).
    pub fn owns_expert(&self, expert_id: u32) -> bool {
        self.is_dense() || (self.expert_start <= expert_id && expert_id <= self.expert_end)
    }
}

// ── Mode B: available server entry ───────────────────────────────────────────

/// A server in Mode B idle state — it has capacity but no shard loaded yet.
pub struct AvailableEntry {
    pub server_id: String,
    /// Channel to send `RouterMessage` (including `AssignMsg`) to this server.
    pub sender: mpsc::Sender<Result<RouterMessage, tonic::Status>>,
    pub ram_bytes: u64,
    pub disk_bytes: u64,
    pub store_path: String,
    pub joined_at: std::time::Instant,
}

// ── Grid state ────────────────────────────────────────────────────────────────

pub struct GridState {
    servers: HashMap<String, ServerEntry>,
    // Pre-built: (model_id, layer) → server_ids; rebuilt only on topology change.
    route_table: HashMap<(String, u32), Vec<String>>,
    // Pre-built: layer → server_ids for model_id=None (single-model) queries.
    any_model_table: HashMap<u32, Vec<String>>,
    /// Mode B: servers that advertised capacity and are waiting for assignment.
    /// Key = server_id.
    available_servers: HashMap<String, AvailableEntry>,
    /// Sender channels for currently-serving (Mode A) servers.
    /// Used by the rebalancer to push UnassignMsg without holding a lock.
    /// Key = server_id.
    serving_senders: HashMap<String, mpsc::Sender<Result<RouterMessage, tonic::Status>>>,
    /// Phase 4: number of replicas the router tries to maintain per
    /// `(model_id, layer_start, layer_end)` shard range. Default 1 — every
    /// range needs exactly one server. >1 enables auto-replication: when
    /// fewer than N servers cover a range, the router pulls from the
    /// available pool to bring the count back up.
    target_replicas: u32,
    /// Hot-shard book-keeping: ranges whose req/s currently exceeds the
    /// hot-shard threshold get `effective_target_replicas = target + 1`
    /// until the rate subsides. Rebalancer marks ranges on the hot-shard
    /// tick; under/over-replication checks read this set via
    /// `effective_target_for`.
    /// ADR-0018: keyed on the 5-tuple
    /// `(model_id, layer_start, layer_end, expert_start, expert_end)`.
    /// Two slices that share a layer range but own different experts
    /// can be elevated independently.
    elevated_ranges: HashSet<(String, u32, u32, u32, u32)>,
    /// ADR-0020 — per-replica in-flight saturation ceiling. `None`
    /// (default) disables the filter. When set, `route()` /
    /// `route_expert()` drop replicas where
    /// `requests_in_flight >= saturation_ceiling` before running
    /// the three-tier comparator; if every replica is saturated
    /// they return `None` and the dispatcher 503s.
    saturation_ceiling: Option<u32>,
}

impl Default for GridState {
    fn default() -> Self {
        Self {
            servers: HashMap::new(),
            route_table: HashMap::new(),
            any_model_table: HashMap::new(),
            available_servers: HashMap::new(),
            serving_senders: HashMap::new(),
            target_replicas: 1,
            elevated_ranges: HashSet::new(),
            saturation_ceiling: None,
        }
    }
}

impl GridState {
    /// ADR-0020: set the per-replica in-flight saturation ceiling.
    /// `None` (default) disables filtering.
    pub fn set_saturation_ceiling(&mut self, ceiling: Option<u32>) {
        self.saturation_ceiling = ceiling;
    }

    /// ADR-0020: current saturation ceiling (read-only).
    pub fn saturation_ceiling(&self) -> Option<u32> {
        self.saturation_ceiling
    }

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

    /// Register a server and store its sender for rebalancer-initiated UnassignMsg.
    pub fn register_with_sender(
        &mut self,
        entry: ServerEntry,
        sender: mpsc::Sender<Result<RouterMessage, tonic::Status>>,
    ) {
        self.serving_senders.insert(entry.server_id.clone(), sender);
        self.register(entry);
    }

    pub fn deregister(&mut self, server_id: &str) {
        self.serving_senders.remove(server_id);
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
        layer_stats: Vec<LayerLatency>,
        req_per_sec: f32,
    ) {
        if let Some(entry) = self.servers.get_mut(server_id) {
            entry.cpu_pct = cpu_pct;
            entry.ram_used = ram_used;
            entry.requests_in_flight = requests_in_flight;
            entry.req_per_sec = req_per_sec;
            entry.last_seen = Instant::now();
            for ls in layer_stats {
                entry
                    .layer_latencies
                    .insert(ls.layer, (ls.avg_ms, ls.p99_ms));
            }
        }
        // Heartbeats don't change topology — no table rebuild needed.
    }

    /// Record the latest active-probe RTT for a server. `None`
    /// clears the entry (last probe failed); `Some(ms)` overwrites.
    /// Called by the optional `rtt_probe` task; no-op when the
    /// server has already left the grid between snapshot and write.
    pub fn update_rtt_ms(&mut self, server_id: &str, rtt_ms: Option<f32>) {
        if let Some(entry) = self.servers.get_mut(server_id) {
            entry.rtt_ms = rtt_ms;
        }
    }

    /// Route one layer. O(1) table lookup + O(replicas) least-loaded scan.
    ///
    /// Replica selection cascade:
    ///   1. **GT3 per-layer latency** (`layer_latencies[layer].avg_ms`)
    ///      — combines compute + wire cost; best signal when available.
    ///   2. **Active-probe RTT** (`rtt_ms`) — wire-only cost; useful
    ///      before the first heartbeat carries layer stats, or for
    ///      cross-region tie-breaking. Opt-in via
    ///      `--rtt-probe-interval-secs`.
    ///   3. **Requests in flight** — last-resort load shedding.
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

    /// Accessor for all serving servers (for the rebalancer).
    pub fn servers(&self) -> impl Iterator<Item = (&String, &ServerEntry)> {
        self.servers.iter()
    }

    /// Return IDs of serving servers whose `last_seen` is older than `timeout`.
    /// Stream-close already triggers deregister via the gRPC handler; this
    /// covers the case where a server keeps the stream open but stops sending
    /// heartbeats (deadlock, GC pause, etc.).
    pub fn stale_server_ids(&self, timeout: std::time::Duration) -> Vec<String> {
        let now = Instant::now();
        self.servers
            .iter()
            .filter(|(_, e)| now.saturating_duration_since(e.last_seen) > timeout)
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Returns true if there is at least one available server in the Mode B pool.
    pub fn has_available_servers(&self) -> bool {
        !self.available_servers.is_empty()
    }

    /// Get the sender channel for a serving server by ID (for UnassignMsg delivery).
    pub fn serving_sender(
        &self,
        server_id: &str,
    ) -> Option<mpsc::Sender<Result<RouterMessage, tonic::Status>>> {
        self.serving_senders.get(server_id).cloned()
    }

    /// Register a Mode B available server. Returns the server_id.
    pub fn register_available(
        &mut self,
        server_id: String,
        sender: mpsc::Sender<Result<RouterMessage, tonic::Status>>,
        ram_bytes: u64,
        disk_bytes: u64,
        store_path: String,
    ) {
        tracing::info!(
            server_id = %server_id,
            ram_gb = ram_bytes / (1024 * 1024 * 1024),
            "Grid: Mode B server available"
        );
        self.available_servers.insert(
            server_id.clone(),
            AvailableEntry {
                server_id,
                sender,
                ram_bytes,
                disk_bytes,
                store_path,
                joined_at: std::time::Instant::now(),
            },
        );
    }

    /// Remove a server from the available pool.
    pub fn deregister_available(&mut self, server_id: &str) {
        self.available_servers.remove(server_id);
    }
}

#[cfg(test)]
mod tests {
    use super::testing::entry;
    use super::*;

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

        state.update_heartbeat("a", 80.0, 2048, 20, vec![], 0.0);
        state.update_heartbeat("b", 10.0, 1024, 0, vec![], 0.0);

        assert_eq!(state.route(Some("model-a"), 2).as_deref(), Some("http://b"));
        let a = state.servers.get("a").unwrap();
        assert_eq!(a.cpu_pct, 80.0);
        assert_eq!(a.ram_used, 2048);
        assert_eq!(a.requests_in_flight, 20);
    }

    #[test]
    fn heartbeat_stores_layer_latencies() {
        let mut state = GridState::default();
        state.register(entry("a", "http://a", "model-a", 0, 4));

        let stats = vec![LayerLatency {
            layer: 2,
            avg_ms: 3.5,
            p99_ms: 7.0,
        }];
        state.update_heartbeat("a", 0.0, 0, 0, stats, 0.0);

        let entry = state.servers.get("a").unwrap();
        assert_eq!(entry.layer_latencies.get(&2), Some(&(3.5, 7.0)));
    }

    #[test]
    fn register_available_and_deregister() {
        let mut state = GridState::default();
        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        state.register_available(
            "avail-1".into(),
            tx,
            16 * 1024 * 1024 * 1024,
            100 * 1024 * 1024 * 1024,
            "/mnt/shards".into(),
        );
        assert!(state.available_servers.contains_key("avail-1"));
        state.deregister_available("avail-1");
        assert!(!state.available_servers.contains_key("avail-1"));
    }

    #[test]
    fn stale_server_ids_returns_only_overdue_entries() {
        let mut state = GridState::default();
        let mut fresh = entry("fresh", "http://fresh", "model-a", 0, 1);
        fresh.last_seen = Instant::now();
        let mut stale = entry("stale", "http://stale", "model-a", 0, 1);
        stale.last_seen = Instant::now()
            .checked_sub(std::time::Duration::from_secs(60))
            .unwrap_or_else(Instant::now);
        state.register(fresh);
        state.register(stale);

        let ids = state.stale_server_ids(std::time::Duration::from_secs(25));
        assert_eq!(ids, vec!["stale".to_string()]);

        // With a huge timeout nothing is stale.
        assert!(state
            .stale_server_ids(std::time::Duration::from_secs(3600))
            .is_empty());
    }

    #[test]
    fn heartbeat_stores_req_per_sec() {
        let mut state = GridState::default();
        state.register(entry("a", "http://a", "model-a", 0, 4));
        state.update_heartbeat("a", 0.0, 0, 0, vec![], 12.5);
        assert!((state.servers.get("a").unwrap().req_per_sec - 12.5).abs() < 1e-6);
    }

    #[test]
    fn update_rtt_ms_writes_through() {
        let mut state = GridState::default();
        state.register(entry("a", "http://a", "m", 0, 4));
        state.update_rtt_ms("a", Some(7.5));
        let updated = state.servers().find(|(id, _)| **id == "a").unwrap().1;
        assert_eq!(updated.rtt_ms, Some(7.5));

        // Failed probe → None clears the entry.
        state.update_rtt_ms("a", None);
        let cleared = state.servers().find(|(id, _)| **id == "a").unwrap().1;
        assert!(cleared.rtt_ms.is_none());
    }

    #[test]
    fn update_rtt_ms_is_noop_for_unknown_server() {
        let mut state = GridState::default();
        // No server registered — call should silently do nothing.
        state.update_rtt_ms("ghost", Some(5.0));
    }
}
