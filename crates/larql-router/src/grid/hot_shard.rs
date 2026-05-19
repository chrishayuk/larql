//! Hot-shard detection and elevation bookkeeping.
//!
//! The rebalancer drives a periodic tick:
//!
//!   1. Call [`GridState::hot_layer_ranges`] with the configured
//!      `req/sec` threshold to find ranges whose serving replicas
//!      are saturating their per-shard load.
//!   2. Call [`GridState::mark_elevated`] on each hot range; that
//!      bumps [`GridState::effective_target_for`] by one until the
//!      range cools down.
//!   3. Walk the snapshot from [`GridState::elevated_ranges_snapshot`]
//!      and call [`GridState::demote_elevated`] on any range that is
//!      no longer hot — the over-replication tick in
//!      [`super::replication`] then drops the surplus replica on the
//!      next pass.
//!
//! The state itself lives in [`GridState::elevated_ranges`] (a
//! `HashSet<(model_id, layer_start, layer_end)>`). Functions in this
//! file are the only writers; the replication module is the only
//! consumer.

use std::collections::HashMap;

use super::GridState;

impl GridState {
    /// Hot-shard detection: distinct
    /// `(model_id, layer_start, layer_end, expert_start, expert_end)`
    /// slices where at least one serving replica's most recent
    /// `req_per_sec` heartbeat exceeds `threshold`. Returns an empty
    /// list when `threshold <= 0` (the feature is disabled).
    ///
    /// Uses max-rate-across-replicas: if a router does perfect
    /// load-balancing the rates converge, so any replica crossing the
    /// threshold means the slice's per-replica load has saturated and
    /// adding capacity is warranted. Sorted for deterministic iteration.
    pub fn hot_layer_ranges(&self, threshold: f32) -> Vec<(String, u32, u32, u32, u32)> {
        // `threshold > 0.0` returns false for NaN; the explicit not-greater
        // form below disables the check for NaN and non-positives alike
        // without tripping the `<=` NaN trap.
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        let disabled = !(threshold > 0.0);
        if disabled {
            return Vec::new();
        }
        let mut max_rate: HashMap<(String, u32, u32, u32, u32), f32> = HashMap::new();
        for e in self.servers.values() {
            let key = (
                e.model_id.clone(),
                e.layer_start,
                e.layer_end,
                e.expert_start,
                e.expert_end,
            );
            let cur = max_rate.entry(key).or_insert(0.0);
            if e.req_per_sec > *cur {
                *cur = e.req_per_sec;
            }
        }
        let mut out: Vec<(String, u32, u32, u32, u32)> = max_rate
            .into_iter()
            .filter_map(|(k, v)| if v > threshold { Some(k) } else { None })
            .collect();
        out.sort();
        out
    }

    /// Mark the `(model_id, layer_range, expert_range)` slice as
    /// elevated so that `effective_target_for` returns
    /// `target_replicas + 1`. Returns `true` if this call newly
    /// inserted the slice.
    pub fn mark_elevated(
        &mut self,
        model_id: &str,
        layer_start: u32,
        layer_end: u32,
        expert_start: u32,
        expert_end: u32,
    ) -> bool {
        self.elevated_ranges.insert((
            model_id.to_owned(),
            layer_start,
            layer_end,
            expert_start,
            expert_end,
        ))
    }

    /// Clear the elevation flag for the
    /// `(model_id, layer_range, expert_range)` slice. Returns `true`
    /// if the slice was previously elevated. After demotion the
    /// standard over-replication tick drops the surplus replica.
    pub fn demote_elevated(
        &mut self,
        model_id: &str,
        layer_start: u32,
        layer_end: u32,
        expert_start: u32,
        expert_end: u32,
    ) -> bool {
        self.elevated_ranges.remove(&(
            model_id.to_owned(),
            layer_start,
            layer_end,
            expert_start,
            expert_end,
        ))
    }

    /// Snapshot of currently-elevated slices. Used by the hot-shard
    /// tick to decide which previously-elevated slices to demote.
    pub fn elevated_ranges_snapshot(&self) -> Vec<(String, u32, u32, u32, u32)> {
        let mut out: Vec<(String, u32, u32, u32, u32)> =
            self.elevated_ranges.iter().cloned().collect();
        out.sort();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::super::testing::entry;
    use super::*;

    #[test]
    fn hot_layer_ranges_empty_when_threshold_zero_or_negative() {
        let mut state = GridState::default();
        let mut a = entry("a", "http://a", "model-x", 0, 4);
        a.req_per_sec = 100.0;
        state.register(a);
        // Disabled when threshold <= 0.
        assert!(state.hot_layer_ranges(0.0).is_empty());
        assert!(state.hot_layer_ranges(-1.0).is_empty());
    }

    #[test]
    fn hot_layer_ranges_returns_max_across_replicas() {
        // Two replicas of the same range, one hot, one cool — range is
        // hot if max(req_per_sec) crosses the threshold.
        let mut state = GridState::default();
        let mut hot = entry("hot", "http://hot", "model-x", 0, 4);
        hot.req_per_sec = 50.0;
        let mut cool = entry("cool", "http://cool", "model-x", 0, 4);
        cool.req_per_sec = 5.0;
        state.register(hot);
        state.register(cool);

        let ranges = state.hot_layer_ranges(20.0);
        assert_eq!(ranges, vec![("model-x".to_string(), 0, 4, 0, 0)]);

        // Threshold above both replicas: range is not hot.
        assert!(state.hot_layer_ranges(75.0).is_empty());
    }

    #[test]
    fn elevated_ranges_lift_effective_target_for_over_and_under() {
        let mut state = GridState::default();
        state.set_target_replicas(2);
        state.register(entry("a", "http://a", "model-x", 0, 4));
        state.register(entry("b", "http://b", "model-x", 0, 4));
        // At target: not over, not under.
        assert!(state.over_replicated_ranges().is_empty());
        assert!(state.under_replicated_ranges().is_empty());

        // Elevate → effective target = 3. Two replicas now look under by 1.
        assert!(state.mark_elevated("model-x", 0, 4, 0, 0));
        assert_eq!(state.effective_target_for("model-x", 0, 4, 0, 0), 3);
        assert_eq!(
            state.under_replicated_ranges(),
            vec![("model-x".to_string(), 0, 4, 0, 0, 1)]
        );
        assert!(state.over_replicated_ranges().is_empty());

        // Add a third — at effective target, neither over nor under.
        state.register(entry("c", "http://c", "model-x", 0, 4));
        assert!(state.over_replicated_ranges().is_empty());
        assert!(state.under_replicated_ranges().is_empty());

        // Demote → effective target = 2. Three replicas surplus by 1.
        assert!(state.demote_elevated("model-x", 0, 4, 0, 0));
        assert_eq!(
            state.over_replicated_ranges(),
            vec![("model-x".to_string(), 0, 4, 0, 0, 1)]
        );
    }

    #[test]
    fn mark_elevated_is_idempotent_and_demote_reports_prior_state() {
        let mut state = GridState::default();
        assert!(state.mark_elevated("m", 0, 4, 0, 0)); // newly inserted
        assert!(!state.mark_elevated("m", 0, 4, 0, 0)); // already there
        assert!(state.demote_elevated("m", 0, 4, 0, 0)); // was present
        assert!(!state.demote_elevated("m", 0, 4, 0, 0)); // already gone
    }

    #[test]
    fn elevated_ranges_snapshot_sorted_and_isolated() {
        let mut state = GridState::default();
        state.mark_elevated("z", 0, 4, 0, 0);
        state.mark_elevated("a", 5, 9, 0, 0);
        state.mark_elevated("a", 0, 4, 0, 0);
        let snap = state.elevated_ranges_snapshot();
        assert_eq!(
            snap,
            vec![
                ("a".to_string(), 0, 4, 0, 0),
                ("a".to_string(), 5, 9, 0, 0),
                ("z".to_string(), 0, 4, 0, 0),
            ]
        );
    }
}
