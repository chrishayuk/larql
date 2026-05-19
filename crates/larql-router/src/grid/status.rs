//! Read-only observation surface: coverage gaps, shard URL list,
//! and the gRPC [`StatusResponse`] builder.
//!
//! These methods are consumed by `/v1/health`, the `larql-router
//! status` admin subcommand, and the `/v1/stats` proxy fan-out
//! (which uses [`GridState::all_shard_urls`] to find a shard to
//! forward to). Pure read paths — no mutation. The mutating
//! state-management side lives in the parent module; replication
//! and gap-fill live in [`super::replication`].

use std::collections::{HashMap, HashSet};

use larql_router_protocol::{
    Gap, LayerLatency, ModelCoverage, ServerInfo, ShardInfo, StatusResponse,
};

use super::{GridState, ServerEntry};

impl GridState {
    /// Return a list of (model_id, layer_start, layer_end) ranges that have no
    /// server covering them, based on the current route table.
    ///
    /// Gaps are only detectable if the router knows the total layer count for
    /// each model. Since the router doesn't store that, we instead return every
    /// layer range between consecutive covered shards.
    pub fn coverage_gaps(&self) -> Vec<(String, u32, u32)> {
        let mut by_model: HashMap<String, Vec<(u32, u32)>> = HashMap::new();
        for entry in self.servers.values() {
            by_model
                .entry(entry.model_id.clone())
                .or_default()
                .push((entry.layer_start, entry.layer_end));
        }
        let mut gaps = Vec::new();
        for (model_id, mut ranges) in by_model {
            ranges.sort_by_key(|(s, _)| *s);
            let mut prev_end: Option<u32> = None;
            for (start, end) in ranges {
                if let Some(pe) = prev_end {
                    if start > pe + 1 {
                        gaps.push((model_id.clone(), pe + 1, start - 1));
                    }
                }
                prev_end = Some(end);
            }
        }
        gaps
    }

    /// All distinct `listen_url` values across all registered servers.
    /// Used by the `/v1/stats` proxy to find a shard to forward to.
    pub fn all_shard_urls(&self) -> Vec<String> {
        let mut seen = HashSet::new();
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
            .map(|e| {
                let mut layer_stats: Vec<LayerLatency> = e
                    .layer_latencies
                    .iter()
                    .map(|(&layer, &(avg_ms, p99_ms))| LayerLatency {
                        layer,
                        avg_ms,
                        p99_ms,
                    })
                    .collect();
                layer_stats.sort_by_key(|l| l.layer);
                ServerInfo {
                    server_id: e.server_id.clone(),
                    listen_url: e.listen_url.clone(),
                    state: "serving".into(),
                    model_id: e.model_id.clone(),
                    layer_start: e.layer_start,
                    layer_end: e.layer_end,
                    cpu_pct: e.cpu_pct,
                    ram_used: e.ram_used,
                    requests_in_flight: e.requests_in_flight,
                    // `0` is the protocol's "unknown" sentinel — the
                    // probe loop hasn't completed a round-trip yet.
                    // Casting from `f32` ms to `u32` ms is fine for
                    // any real RTT (max u32 ≈ 49 days).
                    rtt_ms: e.rtt_ms.map(|ms| ms.round().max(0.0) as u32).unwrap_or(0),
                    layer_stats,
                }
            })
            .collect();

        StatusResponse { models, servers }
    }
}

#[cfg(test)]
mod tests {
    use super::super::testing::entry;
    use super::*;

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

    #[test]
    fn status_response_includes_layer_stats() {
        let mut state = GridState::default();
        let mut srv = entry("a", "http://a", "model-a", 0, 1);
        srv.layer_latencies.insert(0, (2.1, 4.0));
        state.register(srv);

        let status = state.status_response();
        let server = &status.servers[0];
        assert_eq!(server.layer_stats.len(), 1);
        assert_eq!(server.layer_stats[0].layer, 0);
        assert!((server.layer_stats[0].avg_ms - 2.1).abs() < 0.001);
    }

    #[test]
    fn coverage_gaps_finds_uncovered_range() {
        let mut state = GridState::default();
        state.register(entry("a", "http://a", "model-a", 0, 1));
        state.register(entry("b", "http://b", "model-a", 3, 4));

        let gaps = state.coverage_gaps();
        assert_eq!(gaps.len(), 1);
        assert_eq!(gaps[0], ("model-a".to_string(), 2, 2));
    }

    #[test]
    fn coverage_gaps_empty_when_fully_covered() {
        let mut state = GridState::default();
        state.register(entry("a", "http://a", "model-a", 0, 2));
        state.register(entry("b", "http://b", "model-a", 3, 5));

        // Only gap-between-shards; shards are contiguous here.
        let gaps = state.coverage_gaps();
        assert!(gaps.is_empty());
    }

    #[test]
    fn all_shard_urls_deduplicates() {
        let mut state = GridState::default();
        // Two servers on the same listen_url (e.g. shared host); a third on a
        // different one — must collapse to two unique entries.
        let a = entry("a", "http://host:8080", "model-a", 0, 1);
        let b = entry("b", "http://host:8080", "model-a", 2, 3);
        let c = entry("c", "http://other:8081", "model-a", 4, 5);
        state.register(a);
        state.register(b);
        state.register(c);

        let mut urls = state.all_shard_urls();
        urls.sort();
        assert_eq!(urls, vec!["http://host:8080", "http://other:8081"]);
    }

    #[test]
    fn status_response_round_trips_rtt_ms_to_u32() {
        let mut state = GridState::default();
        state.register(entry("a", "http://a", "m", 0, 4));
        state.update_rtt_ms("a", Some(12.7));
        let s = state.status_response();
        assert_eq!(s.servers.len(), 1);
        // f32 → u32 rounding: 12.7 → 13.
        assert_eq!(s.servers[0].rtt_ms, 13);

        // None reports as 0 (the proto's "unknown" sentinel).
        state.update_rtt_ms("a", None);
        assert_eq!(state.status_response().servers[0].rtt_ms, 0);
    }
}
