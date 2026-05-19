//! [`RebalancerConfig`] — knobs for the rebalancer background task.
//!
//! Lives in its own file so that `main.rs` can construct/import it
//! without dragging in the rest of the rebalancer module. The
//! defaults mirror the CLI defaults (`--rebalance-interval 30`,
//! `--rebalance-threshold 2.0`, etc.).

use std::time::Duration;

#[derive(Clone)]
pub struct RebalancerConfig {
    /// How often to run the imbalance check.
    pub check_interval: Duration,
    /// Trigger rebalancing when max(avg_ms) / min(avg_ms) exceeds this ratio
    /// across replicas covering the same layer for at least `sustained_window`.
    pub imbalance_threshold: f32,
    /// Sustained imbalance window before action is taken.
    pub sustained_window: Duration,
    /// Servers that haven't sent a heartbeat within this window are evicted
    /// even if the gRPC stream is still alive. Defensive against deadlocked
    /// servers that keep TCP open but stop sending heartbeats. Default 25 s
    /// = 2.5 × the 10 s heartbeat interval.
    pub stale_heartbeat_timeout: Duration,
    /// Hot-shard request-rate threshold (req/s, max across replicas).
    /// `None` disables the check. When set, a shard whose per-replica
    /// req_per_sec exceeds this value is treated as effectively
    /// under-replicated (target + 1) until the rate subsides.
    pub hot_shard_rps_threshold: Option<f32>,
    /// ADR-0014 hysteresis (amended): once a slice has been
    /// elevated, it stays elevated until its rate falls below
    /// `threshold × demote_ratio`. Prevents oscillation at the
    /// boundary when traffic hovers right around the elevation
    /// threshold. Range `(0.0, 1.0]`; values outside that range get
    /// clamped to `0.8` (the default).
    pub hot_shard_demote_ratio: f32,
}

/// Default hysteresis ratio: demote at 80% of the elevation
/// threshold. Picked to give meaningful headroom (≥20% drop in
/// per-replica load) before the elevation reverses — large enough
/// to absorb measurement noise, small enough that real cool-downs
/// fire promptly.
pub const DEFAULT_HOT_SHARD_DEMOTE_RATIO: f32 = 0.8;

impl Default for RebalancerConfig {
    fn default() -> Self {
        Self {
            check_interval: Duration::from_secs(30),
            imbalance_threshold: 2.0,
            sustained_window: Duration::from_secs(60),
            stale_heartbeat_timeout: Duration::from_secs(25),
            hot_shard_rps_threshold: None,
            hot_shard_demote_ratio: DEFAULT_HOT_SHARD_DEMOTE_RATIO,
        }
    }
}

impl RebalancerConfig {
    pub fn from_cli(interval_secs: u64, threshold: f32) -> Self {
        Self {
            check_interval: Duration::from_secs(interval_secs),
            imbalance_threshold: threshold,
            sustained_window: Duration::from_secs(interval_secs * 2),
            stale_heartbeat_timeout: Duration::from_secs(25),
            hot_shard_rps_threshold: None,
            hot_shard_demote_ratio: DEFAULT_HOT_SHARD_DEMOTE_RATIO,
        }
    }

    /// Builder-style setter for the hot-shard threshold so callers
    /// constructed via `default()` / `from_cli()` can add the threshold
    /// without restating every field.
    pub fn with_hot_shard_threshold(mut self, threshold: Option<f32>) -> Self {
        // Treat ≤0 as "disabled" — saves a magic check in the rebalancer.
        self.hot_shard_rps_threshold = threshold.filter(|t| *t > 0.0);
        self
    }

    /// Builder-style setter for the hysteresis ratio. Values outside
    /// `(0.0, 1.0]` clamp to the default so a misconfigured CLI
    /// can't disable hysteresis or invert the cascade.
    pub fn with_hot_shard_demote_ratio(mut self, ratio: f32) -> Self {
        // Reject NaN, ≤0 (would always demote), >1.0 (would demote
        // above the elevation threshold). On any invalid value, fall
        // back to the default rather than failing — this is config
        // tightening, not a precondition.
        self.hot_shard_demote_ratio = if ratio.is_finite() && ratio > 0.0 && ratio <= 1.0 {
            ratio
        } else {
            DEFAULT_HOT_SHARD_DEMOTE_RATIO
        };
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rebalancer_config_defaults() {
        let cfg = RebalancerConfig::default();
        assert_eq!(cfg.check_interval, Duration::from_secs(30));
        assert_eq!(cfg.imbalance_threshold, 2.0);
        assert_eq!(cfg.stale_heartbeat_timeout, Duration::from_secs(25));
    }

    #[test]
    fn from_cli_derives_sustained_window_from_interval() {
        let cfg = RebalancerConfig::from_cli(15, 2.5);
        assert_eq!(cfg.check_interval, Duration::from_secs(15));
        assert_eq!(cfg.imbalance_threshold, 2.5);
        assert_eq!(cfg.sustained_window, Duration::from_secs(30));
        assert_eq!(cfg.stale_heartbeat_timeout, Duration::from_secs(25));
    }

    /// ADR-0014 amended: `with_hot_shard_demote_ratio` accepts
    /// values in `(0.0, 1.0]` and clamps anything else to the
    /// `DEFAULT_HOT_SHARD_DEMOTE_RATIO`.
    #[test]
    fn with_hot_shard_demote_ratio_accepts_valid_range_and_clamps_invalid() {
        // Valid: in range.
        let cfg = RebalancerConfig::default().with_hot_shard_demote_ratio(0.5);
        assert_eq!(cfg.hot_shard_demote_ratio, 0.5);
        let cfg = RebalancerConfig::default().with_hot_shard_demote_ratio(1.0);
        assert_eq!(cfg.hot_shard_demote_ratio, 1.0);

        // Invalid: outside (0.0, 1.0].
        let cfg = RebalancerConfig::default().with_hot_shard_demote_ratio(0.0);
        assert_eq!(cfg.hot_shard_demote_ratio, DEFAULT_HOT_SHARD_DEMOTE_RATIO);
        let cfg = RebalancerConfig::default().with_hot_shard_demote_ratio(-0.5);
        assert_eq!(cfg.hot_shard_demote_ratio, DEFAULT_HOT_SHARD_DEMOTE_RATIO);
        let cfg = RebalancerConfig::default().with_hot_shard_demote_ratio(1.5);
        assert_eq!(cfg.hot_shard_demote_ratio, DEFAULT_HOT_SHARD_DEMOTE_RATIO);
        let cfg = RebalancerConfig::default().with_hot_shard_demote_ratio(f32::NAN);
        assert_eq!(cfg.hot_shard_demote_ratio, DEFAULT_HOT_SHARD_DEMOTE_RATIO);
    }

    #[test]
    fn default_has_hysteresis_ratio_set_to_constant() {
        let cfg = RebalancerConfig::default();
        assert_eq!(cfg.hot_shard_demote_ratio, DEFAULT_HOT_SHARD_DEMOTE_RATIO);
        assert_eq!(DEFAULT_HOT_SHARD_DEMOTE_RATIO, 0.8);
    }

    #[test]
    fn with_hot_shard_threshold_filters_non_positive() {
        let cfg = RebalancerConfig::default().with_hot_shard_threshold(Some(10.0));
        assert_eq!(cfg.hot_shard_rps_threshold, Some(10.0));

        // 0 and negative values disable the check (treated as None).
        let cfg = RebalancerConfig::default().with_hot_shard_threshold(Some(0.0));
        assert_eq!(cfg.hot_shard_rps_threshold, None);
        let cfg = RebalancerConfig::default().with_hot_shard_threshold(Some(-5.0));
        assert_eq!(cfg.hot_shard_rps_threshold, None);
        let cfg = RebalancerConfig::default().with_hot_shard_threshold(None);
        assert_eq!(cfg.hot_shard_rps_threshold, None);
    }
}
