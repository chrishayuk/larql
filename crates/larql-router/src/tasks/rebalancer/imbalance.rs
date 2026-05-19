//! Per-layer latency imbalance detection.
//!
//! Tracks how long each `(model_id, layer)` has been observed as
//! imbalanced (max/min replica latency ratio above the configured
//! threshold). Once the imbalance has persisted past
//! [`RebalancerConfig::sustained_window`], the slowest replica
//! receives an `UnassignMsg(reason="rebalancing")` so the spare from
//! the available pool can take over.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use tracing::{debug, info};

use larql_router_protocol::{RouterMessage, RouterPayload, UnassignMsg};

use crate::grid::GridState;
use crate::metrics::RouterMetrics;

use super::config::RebalancerConfig;

/// Tracks how long a given layer has been in imbalanced state.
#[derive(Default)]
pub(super) struct ImbalanceTracker {
    /// (model_id, layer) → first_seen_imbalanced
    first_seen: HashMap<(String, u32), Instant>,
}

impl ImbalanceTracker {
    /// Record that this layer is currently imbalanced. Returns true if the
    /// imbalance has been sustained long enough to trigger action.
    fn record(&mut self, key: (String, u32), sustained: Duration) -> bool {
        let entry = self.first_seen.entry(key).or_insert_with(Instant::now);
        entry.elapsed() >= sustained
    }

    /// Clear a layer's imbalance record (it is now balanced or was acted on).
    fn clear(&mut self, key: &(String, u32)) {
        self.first_seen.remove(key);
    }
}

pub(super) async fn check_imbalance(
    state: &Arc<RwLock<GridState>>,
    cfg: &RebalancerConfig,
    tracker: &mut ImbalanceTracker,
    metrics: Option<&RouterMetrics>,
) {
    // Collect per-layer latency data across all servers.
    // Group by (model_id, layer) → Vec<(server_id, avg_ms)>.
    let snapshot = {
        let guard = state.read();
        let mut by_layer: HashMap<(String, u32), Vec<(String, f32)>> = HashMap::new();
        for (sid, entry) in guard.servers() {
            for (&layer, &(avg_ms, _p99)) in &entry.layer_latencies {
                by_layer
                    .entry((entry.model_id.clone(), layer))
                    .or_default()
                    .push((sid.clone(), avg_ms));
            }
        }
        let has_available = guard.has_available_servers();
        (by_layer, has_available)
    };

    let (by_layer, has_available) = snapshot;

    // Only rebalance if there is a spare server ready to take over.
    if !has_available {
        debug!("Rebalancer: no available servers — skipping imbalance check");
        return;
    }

    for ((model_id, layer), mut servers) in by_layer {
        if servers.len() < 2 {
            // Can't detect imbalance without at least 2 replicas.
            tracker.clear(&(model_id, layer));
            continue;
        }

        servers.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let min_ms = servers.first().map(|(_, ms)| *ms).unwrap_or(0.0);
        let max_ms = servers.last().map(|(_, ms)| *ms).unwrap_or(0.0);
        let slowest_server_id = servers.last().map(|(id, _)| id.clone());

        if min_ms <= 0.0 {
            continue;
        }

        let ratio = max_ms / min_ms;
        let key = (model_id.clone(), layer);

        if ratio > cfg.imbalance_threshold {
            let sustained = tracker.record(key.clone(), cfg.sustained_window);
            if sustained {
                // Imbalance has persisted long enough — send UnassignMsg.
                if let Some(ref server_id) = slowest_server_id {
                    info!(
                        model_id = %model_id,
                        layer,
                        ratio = %format!("{ratio:.1}×"),
                        server_id = %server_id,
                        "Rebalancer: sustained imbalance detected — sending UnassignMsg"
                    );
                    send_unassign(state, server_id, &model_id, layer).await;
                    if let Some(m) = metrics {
                        m.rebalancer_actions_total
                            .with_label_values(&["unassign_imbalance"])
                            .inc();
                    }
                    tracker.clear(&key);
                }
            } else {
                debug!(
                    model_id = %model_id,
                    layer,
                    ratio = %format!("{ratio:.1}×"),
                    "Rebalancer: imbalance observed (not yet sustained)"
                );
            }
        } else {
            tracker.clear(&key);
        }
    }
}

/// Send `UnassignMsg` to the serving server identified by `server_id`.
/// The sender channel is stored in `GridState::serving_senders`.
async fn send_unassign(
    state: &Arc<RwLock<GridState>>,
    server_id: &str,
    model_id: &str,
    layer: u32,
) {
    let guard = state.read();
    if let Some(tx) = guard.serving_sender(server_id) {
        let msg = RouterMessage {
            payload: Some(RouterPayload::Unassign(UnassignMsg {
                model_id: model_id.to_owned(),
                layer_start: layer,
                layer_end: layer,
                reason: "rebalancing".to_owned(),
            })),
        };
        if let Err(e) = tx.try_send(Ok(msg)) {
            tracing::warn!(server_id, "Rebalancer: failed to send UnassignMsg: {e}");
        }
    } else {
        tracing::warn!(
            server_id,
            "Rebalancer: no sender for server — already disconnected?"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::testing::entry;
    use tokio::sync::mpsc;

    #[test]
    fn imbalance_tracker_records_and_clears() {
        let mut t = ImbalanceTracker::default();
        let key = ("model".to_string(), 5u32);
        // First record: not sustained yet (window = 1 hour).
        assert!(!t.record(key.clone(), Duration::from_secs(3600)));
        // Clear and re-record: still fresh.
        t.clear(&key);
        assert!(!t.record(key.clone(), Duration::from_secs(3600)));
        // With zero window: sustained immediately.
        let key2 = ("model".to_string(), 6u32);
        assert!(t.record(key2, Duration::from_secs(0)));
    }

    #[tokio::test]
    async fn check_imbalance_no_op_when_no_available_servers() {
        let state = Arc::new(RwLock::new(GridState::default()));
        let cfg = RebalancerConfig::default();
        let mut tracker = ImbalanceTracker::default();
        {
            let mut g = state.write();
            let mut a = entry("a", "http://a", "m", 0, 0);
            a.layer_latencies.insert(0, (5.0, 10.0));
            let mut b = entry("b", "http://b", "m", 0, 0);
            b.layer_latencies.insert(0, (50.0, 100.0));
            g.register(a);
            g.register(b);
            // No available pool registered — rebalancer must skip.
        }
        check_imbalance(&state, &cfg, &mut tracker, None).await;
        // Tracker stays empty — early return before recording.
        assert!(tracker.first_seen.is_empty());
    }

    #[tokio::test]
    async fn check_imbalance_records_then_acts_after_sustained_window() {
        let state = Arc::new(RwLock::new(GridState::default()));
        let (slow_tx, mut slow_rx) = mpsc::channel::<Result<RouterMessage, tonic::Status>>(4);
        let (fast_tx, _fast_rx) = mpsc::channel::<Result<RouterMessage, tonic::Status>>(4);
        let (spare_tx, _spare_rx) = mpsc::channel::<Result<RouterMessage, tonic::Status>>(4);
        {
            let mut g = state.write();
            // Two replicas with a 10× latency gap on layer 0.
            let mut slow = entry("slow", "http://slow", "m", 0, 0);
            slow.layer_latencies.insert(0, (50.0, 100.0));
            let mut fast = entry("fast", "http://fast", "m", 0, 0);
            fast.layer_latencies.insert(0, (5.0, 10.0));
            g.register_with_sender(slow, slow_tx);
            g.register_with_sender(fast, fast_tx);
            // Spare so the rebalancer is willing to act.
            g.register_available("spare".into(), spare_tx, 1, 0, "/".into());
        }
        // Zero-window config: the first observation immediately becomes
        // sustained, and the rebalancer should send UnassignMsg.
        let cfg = RebalancerConfig {
            check_interval: Duration::from_secs(30),
            imbalance_threshold: 2.0,
            sustained_window: Duration::from_secs(0),
            stale_heartbeat_timeout: Duration::from_secs(25),
            hot_shard_rps_threshold: None,
            hot_shard_demote_ratio: 0.8,
        };
        let mut tracker = ImbalanceTracker::default();
        check_imbalance(&state, &cfg, &mut tracker, None).await;
        let msg = slow_rx
            .try_recv()
            .expect("slow replica should have been Unassigned")
            .expect("ok payload");
        let Some(RouterPayload::Unassign(u)) = msg.payload else {
            panic!("expected Unassign, got {msg:?}");
        };
        assert_eq!(u.reason, "rebalancing");
        assert_eq!(u.layer_start, 0);
        assert_eq!(u.layer_end, 0);
    }

    #[tokio::test]
    async fn check_imbalance_clears_when_balanced() {
        let state = Arc::new(RwLock::new(GridState::default()));
        let (spare_tx, _spare_rx) = mpsc::channel::<Result<RouterMessage, tonic::Status>>(4);
        {
            let mut g = state.write();
            let mut a = entry("a", "http://a", "m", 0, 0);
            a.layer_latencies.insert(0, (5.0, 10.0));
            let mut b = entry("b", "http://b", "m", 0, 0);
            b.layer_latencies.insert(0, (5.5, 10.5));
            g.register(a);
            g.register(b);
            g.register_available("spare".into(), spare_tx, 1, 0, "/".into());
        }
        let cfg = RebalancerConfig {
            sustained_window: Duration::from_secs(60),
            ..RebalancerConfig::default()
        };
        let mut tracker = ImbalanceTracker::default();
        // Pre-populate as if an earlier observation flagged the layer; the
        // balanced state on this tick should clear it.
        tracker.first_seen.insert(("m".into(), 0), Instant::now());
        check_imbalance(&state, &cfg, &mut tracker, None).await;
        assert!(
            tracker.first_seen.is_empty(),
            "balanced layer should clear tracker entry"
        );
    }

    #[tokio::test]
    async fn send_unassign_warns_on_missing_sender() {
        let state = Arc::new(RwLock::new(GridState::default()));
        // No server registered with that id — should hit the "no sender"
        // warn path without panicking.
        send_unassign(&state, "ghost", "m", 0).await;
    }
}
