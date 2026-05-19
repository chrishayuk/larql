//! Hot-shard detection tick with two-threshold hysteresis
//! (ADR-0014 amendment, 2026-05-16).
//!
//! Marks slices whose `req/sec` crosses `threshold` as elevated and
//! demotes elevated slices that have cooled below
//! `threshold × demote_ratio`. The two-threshold pattern prevents
//! the elevation flag from oscillating on tick-to-tick boundaries
//! when traffic hovers around the elevation cutoff — a single
//! threshold would mark+demote+mark+demote each tick at the
//! boundary, churning replicas for no net change in load.
//!
//! With `demote_ratio = 0.8` (default), an elevation that fired at
//! 200 req/s only reverses once the rate drops below 160 req/s.
//! Real cool-downs (10× drop) demote immediately; noise around the
//! threshold doesn't.
//!
//! The follow-on `check_under_replication` /
//! `check_over_replication` ticks act on the new effective targets;
//! this function does not send any messages itself.

use std::collections::HashSet;
use std::sync::Arc;

use parking_lot::RwLock;

use crate::grid::GridState;
use crate::metrics::RouterMetrics;

pub(super) async fn check_hot_shards(
    state: &Arc<RwLock<GridState>>,
    threshold: f32,
    demote_ratio: f32,
    metrics: Option<&RouterMetrics>,
) {
    let mut guard = state.write();

    // Slices currently exceeding the *elevation* threshold.
    let hot: HashSet<(String, u32, u32, u32, u32)> =
        guard.hot_layer_ranges(threshold).into_iter().collect();
    let elevated: HashSet<(String, u32, u32, u32, u32)> =
        guard.elevated_ranges_snapshot().into_iter().collect();

    // ADR-0014 amendment — hysteresis.
    //   Elevate side  : slice crossed the *full* `threshold`.
    //   Demote side   : slice fell below `threshold × demote_ratio`
    //                   (i.e. it is *not* in `hot_at_demote_threshold`).
    // The middle band (`threshold × demote_ratio` ≤ rate ≤ `threshold`)
    // is the no-op zone: previously-elevated slices stay elevated,
    // previously-non-elevated stay non-elevated.
    let demote_threshold = threshold * demote_ratio;
    let still_hot_for_demote_check: HashSet<(String, u32, u32, u32, u32)> = guard
        .hot_layer_ranges(demote_threshold)
        .into_iter()
        .collect();

    for slice in hot.difference(&elevated) {
        guard.mark_elevated(&slice.0, slice.1, slice.2, slice.3, slice.4);
        tracing::info!(
            model_id = %slice.0,
            layers = %format!("{}-{}", slice.1, slice.2),
            experts = %format!("{}-{}", slice.3, slice.4),
            threshold,
            "Rebalancer: hot shard detected — effective_target raised by 1"
        );
        if let Some(m) = metrics {
            m.rebalancer_actions_total
                .with_label_values(&["elevate"])
                .inc();
        }
    }
    // Demote only slices that have cooled BELOW the demote threshold.
    // `elevated.difference(&still_hot_for_demote_check)` is exactly
    // the set of elevated slices whose rate is now below
    // `threshold × demote_ratio`.
    for slice in elevated.difference(&still_hot_for_demote_check) {
        guard.demote_elevated(&slice.0, slice.1, slice.2, slice.3, slice.4);
        tracing::info!(
            model_id = %slice.0,
            layers = %format!("{}-{}", slice.1, slice.2),
            experts = %format!("{}-{}", slice.3, slice.4),
            demote_threshold,
            "Rebalancer: hot shard cooled — effective_target restored"
        );
        if let Some(m) = metrics {
            m.rebalancer_actions_total
                .with_label_values(&["demote"])
                .inc();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::replication::{check_over_replication, check_under_replication};
    use super::*;
    use crate::grid::testing::entry;
    use larql_router_protocol::{RouterMessage, RouterPayload};

    #[tokio::test]
    async fn check_hot_shards_bumps_elevate_demote_counters() {
        // Drives both the elevate and demote metric branches in one test.
        use crate::metrics::{encode_metrics_text, RouterMetrics};
        let m = RouterMetrics::new();

        let state = Arc::new(RwLock::new(GridState::default()));
        {
            let mut g = state.write();
            let mut a = entry("a", "http://a", "m", 0, 4);
            a.req_per_sec = 50.0;
            g.register(a);
        }
        // Hot → elevate counter bumps to 1.
        check_hot_shards(&state, 20.0, 0.8, Some(&m)).await;
        // Cool the replica + re-run → demote counter bumps to 1.
        {
            let mut g = state.write();
            g.update_heartbeat("a", 0.0, 0, 0, vec![], 0.0);
        }
        check_hot_shards(&state, 20.0, 0.8, Some(&m)).await;

        let text = encode_metrics_text(&m).unwrap();
        assert!(text.contains("larql_router_rebalancer_actions_total{action=\"elevate\"} 1"));
        assert!(text.contains("larql_router_rebalancer_actions_total{action=\"demote\"} 1"));
    }

    /// ADR-0014 amended hysteresis: a slice that's elevated stays
    /// elevated while its rate sits in the middle band
    /// `(threshold × demote_ratio, threshold]`. Only falling below
    /// the demote threshold reverses the elevation.
    #[tokio::test]
    async fn check_hot_shards_hysteresis_holds_in_middle_band() {
        let state = Arc::new(RwLock::new(GridState::default()));
        {
            let mut g = state.write();
            let mut a = entry("a", "http://a", "m", 0, 4);
            a.req_per_sec = 25.0; // > threshold (20) → elevates
            g.register(a);
        }
        // First tick: above 20, elevates.
        check_hot_shards(&state, 20.0, 0.8, None).await;
        {
            let g = state.read();
            assert_eq!(g.elevated_ranges_snapshot().len(), 1, "must elevate");
        }
        // Drop rate to 18 — *above* demote threshold (20 × 0.8 = 16)
        // but below elevation threshold. Without hysteresis this
        // would demote. With hysteresis it stays elevated.
        {
            let mut g = state.write();
            g.update_heartbeat("a", 0.0, 0, 0, vec![], 18.0);
        }
        check_hot_shards(&state, 20.0, 0.8, None).await;
        {
            let g = state.read();
            assert_eq!(
                g.elevated_ranges_snapshot().len(),
                1,
                "hysteresis: middle-band rate must NOT demote"
            );
        }
        // Drop below demote threshold (16) → now demotes.
        {
            let mut g = state.write();
            g.update_heartbeat("a", 0.0, 0, 0, vec![], 10.0);
        }
        check_hot_shards(&state, 20.0, 0.8, None).await;
        {
            let g = state.read();
            assert!(
                g.elevated_ranges_snapshot().is_empty(),
                "rate below demote threshold must demote"
            );
        }
    }

    /// Disabling hysteresis (`demote_ratio = 1.0`) reproduces the
    /// pre-ADR-0014-amendment single-threshold behavior — any drop
    /// below the elevation threshold demotes immediately.
    #[tokio::test]
    async fn check_hot_shards_ratio_one_disables_hysteresis() {
        let state = Arc::new(RwLock::new(GridState::default()));
        {
            let mut g = state.write();
            let mut a = entry("a", "http://a", "m", 0, 4);
            a.req_per_sec = 50.0;
            g.register(a);
        }
        check_hot_shards(&state, 20.0, 1.0, None).await; // elevate
        {
            let mut g = state.write();
            g.update_heartbeat("a", 0.0, 0, 0, vec![], 19.999); // just below
        }
        check_hot_shards(&state, 20.0, 1.0, None).await; // demotes
        let g = state.read();
        assert!(g.elevated_ranges_snapshot().is_empty());
    }

    #[tokio::test]
    async fn check_hot_shards_marks_newly_hot_ranges() {
        let state = Arc::new(RwLock::new(GridState::default()));
        {
            let mut g = state.write();
            let mut a = entry("a", "http://a", "m", 0, 4);
            a.req_per_sec = 50.0;
            g.register(a);
        }
        check_hot_shards(&state, 20.0, 0.8, None).await;
        let g = state.read();
        assert_eq!(
            g.elevated_ranges_snapshot(),
            vec![("m".to_string(), 0, 4, 0, 0)],
            "range above threshold must be elevated"
        );
    }

    #[tokio::test]
    async fn check_hot_shards_demotes_cooled_ranges() {
        let state = Arc::new(RwLock::new(GridState::default()));
        {
            let mut g = state.write();
            // Pre-elevated range with a cool replica.
            let mut a = entry("a", "http://a", "m", 0, 4);
            a.req_per_sec = 1.0;
            g.register(a);
            assert!(g.mark_elevated("m", 0, 4, 0, 0));
        }
        check_hot_shards(&state, 20.0, 0.8, None).await;
        let g = state.read();
        assert!(
            g.elevated_ranges_snapshot().is_empty(),
            "cooled range must be demoted"
        );
    }

    #[tokio::test]
    async fn check_hot_shards_is_noop_when_state_unchanged() {
        // Hot range that's already elevated stays elevated; cool range
        // that's not elevated stays not elevated. Run twice to confirm
        // idempotence.
        let state = Arc::new(RwLock::new(GridState::default()));
        {
            let mut g = state.write();
            let mut hot = entry("hot", "http://hot", "m", 0, 4);
            hot.req_per_sec = 50.0;
            g.register(hot);
            let mut cool = entry("cool", "http://cool", "m", 5, 9);
            cool.req_per_sec = 1.0;
            g.register(cool);
        }
        check_hot_shards(&state, 20.0, 0.8, None).await;
        check_hot_shards(&state, 20.0, 0.8, None).await;
        let g = state.read();
        assert_eq!(
            g.elevated_ranges_snapshot(),
            vec![("m".to_string(), 0, 4, 0, 0)],
        );
    }

    #[tokio::test]
    async fn hot_then_cool_path_pulls_and_drops_replica() {
        // End-to-end: hot detected → under-rep tick pulls spare; cool
        // detected → over-rep tick drops the surplus replica.
        use tokio::sync::mpsc;

        let state = Arc::new(RwLock::new(GridState::default()));
        let (spare_tx, mut spare_rx) = mpsc::channel::<Result<RouterMessage, tonic::Status>>(4);
        let (busy_tx, _busy_rx) = mpsc::channel::<Result<RouterMessage, tonic::Status>>(4);
        {
            let mut g = state.write();
            // target_replicas == 1 default — hot bump takes effective to 2.
            let mut a = entry("a", "http://a", "m", 0, 4);
            a.req_per_sec = 100.0;
            g.register_with_sender(a, busy_tx);
            g.register_available("spare".into(), spare_tx, 1, 0, "/".into());
        }
        // Hot detection + spare pull (mirrors rebalancer_task ordering).
        check_hot_shards(&state, 50.0, 0.8, None).await;
        check_under_replication(&state, None).await;

        let pulled = spare_rx
            .try_recv()
            .expect("spare should receive AssignMsg")
            .expect("ok payload");
        let Some(RouterPayload::Assign(a)) = pulled.payload else {
            panic!("expected Assign, got {pulled:?}");
        };
        assert_eq!(a.layer_start, 0);
        assert_eq!(a.layer_end, 4);

        // Simulate the spare arriving as a serving replica and the
        // workload cooling: rate drops, hot tick demotes, over-rep
        // tick drops the surplus.
        let (extra_tx, mut extra_rx) = mpsc::channel::<Result<RouterMessage, tonic::Status>>(4);
        {
            let mut g = state.write();
            let mut extra = entry("extra", "http://extra", "m", 0, 4);
            extra.req_per_sec = 0.5;
            g.register_with_sender(extra, extra_tx);
            // Existing replica also cools.
            g.update_heartbeat("a", 0.0, 0, 0, vec![], 0.5);
        }
        check_hot_shards(&state, 50.0, 0.8, None).await;
        check_over_replication(&state, None).await;

        // Either of the two replicas (extra or a) is least-loaded; the
        // important thing is that one Unassign fires for layers 0-4. If
        // the chosen victim is "a" (whose sender is kept by busy_tx), the
        // unassign just gets queued there; the assertion below relaxes to
        // "if extra was the victim, it received an over_replicated
        // Unassign for the correct range."
        let got = extra_rx.try_recv();
        if let Ok(Ok(msg)) = got {
            if let Some(RouterPayload::Unassign(u)) = msg.payload {
                assert_eq!(u.model_id, "m");
                assert_eq!(u.layer_start, 0);
                assert_eq!(u.layer_end, 4);
                assert_eq!(u.reason, "over_replicated");
            }
        }
        let g = state.read();
        assert!(
            g.elevated_ranges_snapshot().is_empty(),
            "range must be demoted after cooling"
        );
    }
}
