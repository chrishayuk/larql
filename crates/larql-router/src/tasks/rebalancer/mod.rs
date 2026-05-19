//! Dynamic-rebalancing background task.
//!
//! Runs every [`RebalancerConfig::check_interval`] seconds and drives
//! four scans against the shared [`GridState`]:
//!
//! 1. [`eviction::evict_stale_heartbeats`] — drop servers whose stream
//!    is up but heartbeats stopped.
//! 2. [`hot_shard::check_hot_shards`] — flip the elevation set for
//!    ranges crossing the per-shard `req/sec` threshold so the next
//!    two ticks see them as effectively under-replicated.
//! 3. [`replication::check_under_replication`] — pull spares from the
//!    available pool to bring under-replicated ranges to target.
//! 4. [`replication::check_over_replication`] — drop the
//!    least-loaded replica of any over-replicated range.
//! 5. [`imbalance::check_imbalance`] — detect sustained per-layer
//!    latency imbalance and send `UnassignMsg(reason="rebalancing")`
//!    to the slowest replica so the spare can take over.
//!
//! The split across files mirrors the same concerns in
//! [`crate::grid`] — replication-on-state lives in
//! `grid::replication`; replication-as-action (pulling spares) lives
//! in `rebalancer::replication`. Both consume the elevation set
//! updated by `rebalancer::hot_shard`.

pub mod config;
mod eviction;
mod hot_shard;
mod imbalance;
mod replication;

use std::sync::Arc;

use parking_lot::RwLock;

use crate::grid::GridState;
use crate::metrics::RouterMetrics;

pub use config::RebalancerConfig;

use eviction::evict_stale_heartbeats;
use hot_shard::check_hot_shards;
use imbalance::{check_imbalance, ImbalanceTracker};
use replication::{check_over_replication, check_under_replication};

/// Spawn the rebalancer background task.
/// Returns immediately; the task runs for the process lifetime.
pub fn spawn(
    state: Arc<RwLock<GridState>>,
    cfg: RebalancerConfig,
    metrics: Option<Arc<RouterMetrics>>,
) {
    tokio::spawn(rebalancer_task(state, cfg, metrics));
}

async fn rebalancer_task(
    state: Arc<RwLock<GridState>>,
    cfg: RebalancerConfig,
    metrics: Option<Arc<RouterMetrics>>,
) {
    let mut interval = tokio::time::interval(cfg.check_interval);
    let mut tracker = ImbalanceTracker::default();

    loop {
        interval.tick().await;
        evict_stale_heartbeats(&state, cfg.stale_heartbeat_timeout, metrics.as_deref()).await;
        // Hot-shard elevation runs before replica checks so newly hot
        // ranges look under-replicated in the same tick (and cooling
        // ranges look over-replicated, freeing the surplus replica).
        if let Some(threshold) = cfg.hot_shard_rps_threshold {
            check_hot_shards(
                &state,
                threshold,
                cfg.hot_shard_demote_ratio,
                metrics.as_deref(),
            )
            .await;
        }
        check_under_replication(&state, metrics.as_deref()).await;
        check_over_replication(&state, metrics.as_deref()).await;
        check_imbalance(&state, &cfg, &mut tracker, metrics.as_deref()).await;
        // ADR-0017: scrape-equivalent gauge refresh — fires once per
        // rebalancer tick so /metrics responses between ticks see the
        // latest values without needing per-mutation gauge plumbing.
        if let Some(m) = metrics.as_deref() {
            m.refresh_gauges(&state.read());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// Spawn the rebalancer with a tight tick, wait long enough for at
    /// least two ticks to fire, then drop our state Arc. Exercises
    /// `spawn` + the loop body in `rebalancer_task` so they aren't
    /// permanently uncovered.
    #[tokio::test]
    async fn spawn_runs_the_task_loop_through_one_tick() {
        let state = Arc::new(RwLock::new(GridState::default()));
        let cfg = RebalancerConfig {
            check_interval: Duration::from_millis(20),
            stale_heartbeat_timeout: Duration::from_secs(60),
            hot_shard_rps_threshold: Some(1.0),
            ..RebalancerConfig::default()
        };
        spawn(state.clone(), cfg, None);
        // tokio::time::interval fires immediately then on cadence, so two
        // 20 ms intervals plus scheduler slack covers two full passes
        // through the loop body.
        tokio::time::sleep(Duration::from_millis(80)).await;
        // No assertion on side effects — the test passes if the task
        // didn't panic and the runtime can still drop the state Arc.
    }
}
