//! Under/over-replication ticks.
//!
//! Each runs once per rebalancer interval after the hot-shard scan
//! has updated the elevation set:
//!
//! - [`check_under_replication`] pulls spares from the available pool
//!   to bring under-replicated ranges up to their effective target
//!   (`target_replicas`, plus any hot-shard bump).
//! - [`check_over_replication`] sends `UnassignMsg(reason="over_replicated")`
//!   to the least-loaded replica of any range whose live count
//!   exceeds the effective target.

use std::sync::Arc;

use parking_lot::RwLock;

use larql_router_protocol::{RouterMessage, RouterPayload, UnassignMsg};

use crate::grid::GridState;
use crate::metrics::RouterMetrics;

/// Phase 4: pull spares from the available pool to bring under-replicated
/// ranges up to their effective target (`target_replicas`, plus the
/// hot-shard bump). Triggered periodically — in addition to the
/// event-driven triggers in `grid/service.rs` (Available, Ready,
/// deregister).
///
/// Note: this no longer short-circuits on `target_replicas <= 1` because
/// the hot-shard tick can elevate a range's effective target above 1
/// even when the static target is 1.
pub(super) async fn check_under_replication(
    state: &Arc<RwLock<GridState>>,
    metrics: Option<&RouterMetrics>,
) {
    let assigned = state.write().try_replicate_from_available();
    if assigned > 0 {
        tracing::info!(
            assigned,
            "Rebalancer: replicated under-replicated ranges from available pool"
        );
        if let Some(m) = metrics {
            m.rebalancer_actions_total
                .with_label_values(&["replicate"])
                .inc_by(assigned as u64);
        }
    }
}

/// Phase 4: drop one replica per over-replicated range each tick by sending
/// `UnassignMsg` to the least-loaded server. Defensive — never drops below
/// `target_replicas` (the over-replicated check already ensures the count is
/// strictly greater).
pub(super) async fn check_over_replication(
    state: &Arc<RwLock<GridState>>,
    metrics: Option<&RouterMetrics>,
) {
    // Snapshot slices + chosen victims while holding only a read lock.
    // Tuple shape: (server_id, model_id, layer_start, layer_end, expert_start, expert_end).
    let plan: Vec<(String, String, u32, u32, u32, u32)> = {
        let g = state.read();
        if g.target_replicas() == 0 {
            return;
        }
        g.over_replicated_ranges()
            .into_iter()
            .filter_map(|(model_id, ls, le, es, ee, _surplus)| {
                g.least_loaded_in_range(&model_id, ls, le, es, ee)
                    .map(|e| (e.server_id.clone(), model_id, ls, le, es, ee))
            })
            .collect()
    };
    if plan.is_empty() {
        return;
    }
    for (server_id, model_id, ls, le, _es, _ee) in plan {
        let tx = {
            let g = state.read();
            g.serving_sender(&server_id)
        };
        let Some(tx) = tx else {
            tracing::warn!(server_id, "Over-replication: no sender for victim");
            continue;
        };
        let msg = RouterMessage {
            payload: Some(RouterPayload::Unassign(UnassignMsg {
                model_id: model_id.clone(),
                layer_start: ls,
                layer_end: le,
                reason: "over_replicated".into(),
            })),
        };
        if tx.try_send(Ok(msg)).is_ok() {
            tracing::info!(
                server_id,
                model_id,
                layers = %format!("{ls}-{le}"),
                "Rebalancer: dropping over-replicated replica"
            );
            if let Some(m) = metrics {
                m.rebalancer_actions_total
                    .with_label_values(&["drop"])
                    .inc();
            }
        } else {
            tracing::warn!(
                server_id,
                "Over-replication: UnassignMsg send failed (peer disconnected)"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::testing::entry;
    use crate::grid::ServerEntry;
    use std::collections::HashMap;
    use tokio::sync::mpsc;

    #[tokio::test]
    async fn under_replication_bumps_replicate_counter() {
        use crate::metrics::{encode_metrics_text, RouterMetrics};
        let m = RouterMetrics::new();

        let state = Arc::new(RwLock::new(GridState::default()));
        let (spare_tx, _spare_rx) = mpsc::channel::<Result<RouterMessage, tonic::Status>>(4);
        {
            let mut g = state.write();
            g.set_target_replicas(2);
            g.register(entry("a", "http://a", "m", 0, 4));
            g.register_available("spare".into(), spare_tx, 1, 0, "/".into());
        }
        check_under_replication(&state, Some(&m)).await;

        let text = encode_metrics_text(&m).unwrap();
        assert!(text.contains("larql_router_rebalancer_actions_total{action=\"replicate\"} 1"));
    }

    #[tokio::test]
    async fn over_replication_bumps_drop_counter() {
        use crate::metrics::{encode_metrics_text, RouterMetrics};
        let m = RouterMetrics::new();

        // Two replicas of the same range, both with senders so HashMap
        // iteration order can't pick a senderless victim. Distinct
        // requests_in_flight values make least-loaded deterministic.
        let state = Arc::new(RwLock::new(GridState::default()));
        let (idle_tx, _idle_rx) = mpsc::channel::<Result<RouterMessage, tonic::Status>>(4);
        let (busy_tx, _busy_rx) = mpsc::channel::<Result<RouterMessage, tonic::Status>>(4);
        {
            let mut g = state.write();
            g.set_target_replicas(1);
            let mut idle = entry("idle", "http://idle", "m", 0, 4);
            idle.requests_in_flight = 0;
            g.register_with_sender(idle, idle_tx);
            let mut busy = entry("busy", "http://busy", "m", 0, 4);
            busy.requests_in_flight = 5;
            g.register_with_sender(busy, busy_tx);
        }
        check_over_replication(&state, Some(&m)).await;

        let text = encode_metrics_text(&m).unwrap();
        assert!(text.contains("larql_router_rebalancer_actions_total{action=\"drop\"} 1"));
    }

    #[tokio::test]
    async fn under_replication_dispatches_assignment_when_target_above_one() {
        let state = Arc::new(RwLock::new(GridState::default()));
        let (spare_tx, mut spare_rx) = mpsc::channel::<Result<RouterMessage, tonic::Status>>(4);
        {
            let mut g = state.write();
            g.set_target_replicas(2);
            g.register(entry("a", "http://a", "m", 0, 4));
            g.register_available("spare".into(), spare_tx, 1, 0, "/".into());
        }
        check_under_replication(&state, None).await;
        let msg = spare_rx
            .try_recv()
            .expect("spare should have been used")
            .expect("ok payload");
        assert!(matches!(msg.payload, Some(RouterPayload::Assign(_))));
    }

    #[tokio::test]
    async fn under_replication_noop_when_target_equals_one() {
        let state = Arc::new(RwLock::new(GridState::default()));
        // target_replicas defaults to 1; should not attempt anything even
        // with a spare in the pool.
        let (tx, mut rx) = mpsc::channel(4);
        {
            let mut g = state.write();
            g.register(entry("a", "http://a", "m", 0, 4));
            g.register_available("spare".into(), tx, 1, 0, "/".into());
        }
        check_under_replication(&state, None).await;
        assert!(
            rx.try_recv().is_err(),
            "no assignment should fire at target=1"
        );
    }

    #[tokio::test]
    async fn over_replication_drops_least_loaded_replica() {
        let state = Arc::new(RwLock::new(GridState::default()));
        let (tx_idle, mut rx_idle) = mpsc::channel::<Result<RouterMessage, tonic::Status>>(4);
        let (tx_busy, _rx_busy) = mpsc::channel::<Result<RouterMessage, tonic::Status>>(4);

        {
            let mut g = state.write();
            g.set_target_replicas(2);
            // 3 replicas of model-x 0-4 — surplus 1. idle is least-loaded.
            let busy_1 = ServerEntry {
                server_id: "busy-1".into(),
                listen_url: "http://busy-1".into(),
                model_id: "model-x".into(),
                layer_start: 0,
                layer_end: 4,
                vindex_hash: "h".into(),
                cpu_pct: 0.0,
                ram_used: 0,
                requests_in_flight: 9,
                last_seen: std::time::Instant::now(),
                layer_latencies: HashMap::new(),
                req_per_sec: 0.0,
                rtt_ms: None,
                expert_start: 0,
                expert_end: 0,
            };
            let busy_2 = ServerEntry {
                requests_in_flight: 8,
                server_id: "busy-2".into(),
                listen_url: "http://busy-2".into(),
                ..busy_1.clone()
            };
            let idle = ServerEntry {
                requests_in_flight: 1,
                server_id: "idle".into(),
                listen_url: "http://idle".into(),
                ..busy_1.clone()
            };
            g.register_with_sender(idle, tx_idle);
            g.register_with_sender(busy_1, tx_busy);
            g.register(busy_2);
        }

        check_over_replication(&state, None).await;

        let received = rx_idle
            .try_recv()
            .expect("least-loaded server must receive UnassignMsg");
        let Ok(RouterMessage {
            payload: Some(RouterPayload::Unassign(u)),
        }) = received
        else {
            panic!("expected Unassign, got: {received:?}");
        };
        assert_eq!(u.model_id, "model-x");
        assert_eq!(u.layer_start, 0);
        assert_eq!(u.layer_end, 4);
        assert_eq!(u.reason, "over_replicated");
    }

    #[tokio::test]
    async fn over_replication_with_no_sender_logs_and_skips() {
        let state = Arc::new(RwLock::new(GridState::default()));
        let (busy_tx, _busy_rx) = mpsc::channel::<Result<RouterMessage, tonic::Status>>(4);
        {
            let mut g = state.write();
            g.set_target_replicas(2);
            // Three replicas, but the least-loaded one has no serving_sender
            // (registered via plain `register`). Test exercises the
            // "no sender for victim" branch.
            let mut idle = entry("idle", "http://idle", "m", 0, 4);
            idle.requests_in_flight = 1;
            g.register(idle); // no sender path
            let mut busy_1 = entry("busy-1", "http://busy-1", "m", 0, 4);
            busy_1.requests_in_flight = 5;
            g.register_with_sender(busy_1, busy_tx.clone());
            let mut busy_2 = entry("busy-2", "http://busy-2", "m", 0, 4);
            busy_2.requests_in_flight = 7;
            g.register_with_sender(busy_2, busy_tx);
        }
        check_over_replication(&state, None).await;
        // The "no sender" branch is taken; nothing observable on busy_tx
        // because the rebalancer picks `idle` as least-loaded and skips it.
        // Assert through state: no server was removed.
        let g = state.read();
        assert_eq!(g.status_response().servers.len(), 3);
    }
}
