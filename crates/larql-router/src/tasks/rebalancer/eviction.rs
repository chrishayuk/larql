//! Stale-heartbeat eviction tick.
//!
//! Defensive against deadlocked servers that keep the TCP stream
//! alive but stop sending heartbeats. After eviction, runs gap-fill
//! in case the disappearance exposed a fillable gap.

use std::sync::Arc;
use std::time::Duration;

use parking_lot::RwLock;

use crate::grid::GridState;
use crate::metrics::RouterMetrics;

pub(super) async fn evict_stale_heartbeats(
    state: &Arc<RwLock<GridState>>,
    timeout: Duration,
    metrics: Option<&RouterMetrics>,
) {
    let stale = state.read().stale_server_ids(timeout);
    if stale.is_empty() {
        return;
    }
    let mut guard = state.write();
    for sid in &stale {
        tracing::warn!(
            server_id = %sid,
            timeout_s = timeout.as_secs(),
            "Rebalancer: evicting stale server (no heartbeat within timeout)"
        );
        guard.deregister(sid);
        if let Some(m) = metrics {
            m.grid_deregisters_total.with_label_values(&["stale"]).inc();
            m.rebalancer_actions_total
                .with_label_values(&["evict"])
                .inc();
        }
    }
    let filled = guard.try_fill_all_gaps();
    if filled > 0 {
        tracing::info!(
            filled,
            "Rebalancer: gap re-fill after stale-heartbeat eviction"
        );
        if let Some(m) = metrics {
            m.rebalancer_actions_total
                .with_label_values(&["replicate"])
                .inc_by(filled as u64);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::testing::entry;
    use crate::grid::ServerEntry;
    use std::collections::HashMap;

    #[tokio::test]
    async fn evict_stale_bumps_deregister_and_evict_counters() {
        use crate::metrics::{encode_metrics_text, RouterMetrics};
        let m = RouterMetrics::new();

        let state = Arc::new(RwLock::new(GridState::default()));
        {
            let mut g = state.write();
            let stale = ServerEntry {
                server_id: "stale".into(),
                listen_url: "http://stale".into(),
                model_id: "m".into(),
                layer_start: 0,
                layer_end: 4,
                vindex_hash: "h".into(),
                cpu_pct: 0.0,
                ram_used: 0,
                requests_in_flight: 0,
                last_seen: std::time::Instant::now()
                    .checked_sub(Duration::from_secs(60))
                    .unwrap(),
                layer_latencies: HashMap::new(),
                req_per_sec: 0.0,
                rtt_ms: None,
                expert_start: 0,
                expert_end: 0,
            };
            g.register(stale);
        }
        evict_stale_heartbeats(&state, Duration::from_secs(25), Some(&m)).await;
        let text = encode_metrics_text(&m).unwrap();
        assert!(text.contains("larql_router_grid_deregisters_total{reason=\"stale\"} 1"));
        assert!(text.contains("larql_router_rebalancer_actions_total{action=\"evict\"} 1"));
    }

    #[tokio::test]
    async fn evict_stale_noop_when_all_fresh() {
        let state = Arc::new(RwLock::new(GridState::default()));
        {
            let mut g = state.write();
            g.register(entry("fresh", "http://fresh", "m", 0, 4));
        }
        evict_stale_heartbeats(&state, Duration::from_secs(25), None).await;
        let g = state.read();
        assert_eq!(g.status_response().servers.len(), 1);
    }

    #[tokio::test]
    async fn evict_stale_removes_overdue_servers() {
        let state = Arc::new(RwLock::new(GridState::default()));
        {
            let mut g = state.write();
            // Stale server: last_seen 60s ago.
            let stale = ServerEntry {
                server_id: "stale".into(),
                listen_url: "http://stale".into(),
                model_id: "m".into(),
                layer_start: 0,
                layer_end: 4,
                vindex_hash: "h".into(),
                cpu_pct: 0.0,
                ram_used: 0,
                requests_in_flight: 0,
                last_seen: std::time::Instant::now()
                    .checked_sub(Duration::from_secs(60))
                    .unwrap(),
                layer_latencies: HashMap::new(),
                req_per_sec: 0.0,
                rtt_ms: None,
                expert_start: 0,
                expert_end: 0,
            };
            g.register(stale);
        }

        evict_stale_heartbeats(&state, Duration::from_secs(25), None).await;

        let g = state.read();
        assert_eq!(
            g.status_response().servers.len(),
            0,
            "stale server must be evicted"
        );
    }
}
