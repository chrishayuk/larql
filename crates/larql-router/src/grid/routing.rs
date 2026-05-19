//! Route-lookup hot path and its comparator.
//!
//! Owns the `route()` / `route_all()` entry points, the
//! `rebuild_route_table()` cold-path indexer, and the three-tier
//! `compare_servers_for_route()` comparator. The state-mutation
//! callers (register / deregister / update_heartbeat) live in the
//! parent module — they invoke `rebuild_route_table()` whenever
//! topology changes.

use std::collections::HashMap;

use super::{GridState, ServerEntry};

/// Routing comparator used by [`GridState::route`]. Three-tier:
///
///   1. **GT3 per-layer latency** — when both replicas have a value
///      for `layer`, the one with lower `avg_ms` wins. Replicas with
///      a value beat replicas without (NaN-safe).
///   2. **Active-probe RTT** — when neither side has GT3 data, use
///      `rtt_ms` as a wire-only tie-breaker. Replicas with a probe
///      result beat unprobed ones.
///   3. **Requests in flight** — last resort. Always defined.
///
/// Hoisted out of `route()` so the cascade is directly testable
/// without standing up a full `GridState`. NaN-tolerant — partial
/// orderings collapse to `Equal` rather than panicking.
fn compare_servers_for_route(a: &ServerEntry, b: &ServerEntry, layer: u32) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    let lat_a = a.layer_latencies.get(&layer).map(|(avg, _)| *avg);
    let lat_b = b.layer_latencies.get(&layer).map(|(avg, _)| *avg);
    match (lat_a, lat_b) {
        (Some(la), Some(lb)) => la.partial_cmp(&lb).unwrap_or(Ordering::Equal),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => match (a.rtt_ms, b.rtt_ms) {
            (Some(ra), Some(rb)) => ra.partial_cmp(&rb).unwrap_or(Ordering::Equal),
            (Some(_), None) => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (None, None) => a.requests_in_flight.cmp(&b.requests_in_flight),
        },
    }
}

impl GridState {
    pub fn route(&self, model_id: Option<&str>, layer: u32) -> Option<String> {
        let ids = match model_id {
            Some(m) => self.route_table.get(&(m.to_owned(), layer)),
            None => self.any_model_table.get(&layer),
        };
        ids.and_then(|server_ids| {
            let ceiling = self.saturation_ceiling;
            server_ids
                .iter()
                .filter_map(|id| self.servers.get(id))
                // ADR-0020 — drop replicas at or above the saturation
                // ceiling before the comparator runs.
                .filter(|e| match ceiling {
                    Some(c) => e.requests_in_flight < c,
                    None => true,
                })
                .min_by(|a, b| compare_servers_for_route(a, b, layer))
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

    /// ADR-0020 — does at least one server own the requested layer?
    ///
    /// Unlike [`Self::route`], this skips the three-tier comparator
    /// AND the saturation filter — it only asks "is the topology
    /// configured to cover this layer at all?" Lets the dispatcher
    /// distinguish two failure modes:
    ///
    ///   * `route()` returns `None` AND `has_owners_for` returns
    ///     `false` → **400** (no shard configured)
    ///   * `route()` returns `None` AND `has_owners_for` returns
    ///     `true` → **503** (all owning shards are saturated)
    pub fn has_owners_for(&self, model_id: Option<&str>, layer: u32) -> bool {
        let ids = match model_id {
            Some(m) => self.route_table.get(&(m.to_owned(), layer)),
            None => self.any_model_table.get(&layer),
        };
        ids.map(|v| !v.is_empty()).unwrap_or(false)
    }

    /// ADR-0018 — pick the best replica that owns `(layer, expert_id)`.
    ///
    /// Filters the candidate set from the layer's route_table to
    /// servers where `expert_start..=expert_end` contains `expert_id`,
    /// then runs the three-tier comparator (ADR-0013) over the filter.
    /// Dense servers (`expert_start == expert_end == 0`) match every
    /// expert_id — so a dense model passes through unchanged.
    pub fn route_expert(
        &self,
        model_id: Option<&str>,
        layer: u32,
        expert_id: u32,
    ) -> Option<String> {
        let ids = match model_id {
            Some(m) => self.route_table.get(&(m.to_owned(), layer)),
            None => self.any_model_table.get(&layer),
        };
        ids.and_then(|server_ids| {
            let ceiling = self.saturation_ceiling;
            server_ids
                .iter()
                .filter_map(|id| self.servers.get(id))
                .filter(|e| e.owns_expert(expert_id))
                // ADR-0020 — saturation filter applies on the MoE
                // path too. If every owning expert-shard is at the
                // ceiling, return None so the dispatcher 503s.
                .filter(|e| match ceiling {
                    Some(c) => e.requests_in_flight < c,
                    None => true,
                })
                .min_by(|a, b| compare_servers_for_route(a, b, layer))
                .map(|s| s.listen_url.clone())
        })
    }

    /// Batched form of [`Self::route_expert`]. Returns
    /// `Ok((layer, expert) → url)` or
    /// `Err((layer, expert))` for the first pair with no owning server.
    #[allow(dead_code)]
    pub fn route_all_experts(
        &self,
        model_id: Option<&str>,
        layer_experts: &[(usize, u32)],
    ) -> Result<HashMap<(usize, u32), String>, (usize, u32)> {
        let mut out = HashMap::with_capacity(layer_experts.len());
        for &(layer, expert_id) in layer_experts {
            match self.route_expert(model_id, layer as u32, expert_id) {
                Some(url) => {
                    out.insert((layer, expert_id), url);
                }
                None => return Err((layer, expert_id)),
            }
        }
        Ok(out)
    }

    /// ADR-0021 — up to `max` candidate URLs for `(model_id, layer)`,
    /// ordered by the same three-tier comparator [`Self::route`] uses,
    /// with the ADR-0020 saturation filter applied. Returned slice's
    /// first element equals what `route()` would return; subsequent
    /// elements are the next-best fallbacks (e.g. hedge targets).
    ///
    /// Returns an empty `Vec` when no replica owns the layer, or every
    /// owner is over the saturation ceiling. Bounded by the actual
    /// candidate count — never returns more than the topology allows.
    pub fn route_with_rank(&self, model_id: Option<&str>, layer: u32, max: usize) -> Vec<String> {
        if max == 0 {
            return Vec::new();
        }
        let ids = match model_id {
            Some(m) => self.route_table.get(&(m.to_owned(), layer)),
            None => self.any_model_table.get(&layer),
        };
        let Some(server_ids) = ids else {
            return Vec::new();
        };
        let ceiling = self.saturation_ceiling;
        let mut candidates: Vec<&ServerEntry> = server_ids
            .iter()
            .filter_map(|id| self.servers.get(id))
            .filter(|e| match ceiling {
                Some(c) => e.requests_in_flight < c,
                None => true,
            })
            .collect();
        candidates.sort_by(|a, b| compare_servers_for_route(a, b, layer));
        candidates
            .into_iter()
            .take(max)
            .map(|s| s.listen_url.clone())
            .collect()
    }

    /// ADR-0021 — MoE sibling of [`Self::route_with_rank`]. Returns up
    /// to `max` candidate URLs for `(model_id, layer, expert_id)` in
    /// comparator order, saturation-filtered.
    pub fn route_expert_with_rank(
        &self,
        model_id: Option<&str>,
        layer: u32,
        expert_id: u32,
        max: usize,
    ) -> Vec<String> {
        if max == 0 {
            return Vec::new();
        }
        let ids = match model_id {
            Some(m) => self.route_table.get(&(m.to_owned(), layer)),
            None => self.any_model_table.get(&layer),
        };
        let Some(server_ids) = ids else {
            return Vec::new();
        };
        let ceiling = self.saturation_ceiling;
        let mut candidates: Vec<&ServerEntry> = server_ids
            .iter()
            .filter_map(|id| self.servers.get(id))
            .filter(|e| e.owns_expert(expert_id))
            .filter(|e| match ceiling {
                Some(c) => e.requests_in_flight < c,
                None => true,
            })
            .collect();
        candidates.sort_by(|a, b| compare_servers_for_route(a, b, layer));
        candidates
            .into_iter()
            .take(max)
            .map(|s| s.listen_url.clone())
            .collect()
    }

    /// Rebuild layer→servers index. Called only on join/leave (cold path).
    pub(super) fn rebuild_route_table(&mut self) {
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
}

#[cfg(test)]
mod tests {
    use super::super::testing::entry;
    use super::*;

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
    fn route_all_returns_first_uncovered_layer() {
        let mut state = GridState::default();
        state.register(entry("a", "http://a", "model-a", 0, 1));
        state.register(entry("b", "http://b", "model-a", 3, 4));

        assert_eq!(state.route_all(Some("model-a"), &[0, 1, 2, 3]), Err(2));
    }

    #[test]
    fn route_prefers_lower_layer_latency_over_inflight() {
        // slow has fewer requests_in_flight but higher per-layer latency.
        // fast has more requests but lower layer latency.
        // Router should route to fast.
        let mut state = GridState::default();
        let mut slow = entry("slow", "http://slow", "model-a", 0, 4);
        slow.requests_in_flight = 2;
        slow.layer_latencies.insert(2, (50.0, 80.0)); // 50 ms avg

        let mut fast = entry("fast", "http://fast", "model-a", 0, 4);
        fast.requests_in_flight = 8;
        fast.layer_latencies.insert(2, (5.0, 9.0)); // 5 ms avg

        state.register(slow);
        state.register(fast);

        assert_eq!(
            state.route(Some("model-a"), 2).as_deref(),
            Some("http://fast")
        );
    }

    #[test]
    fn compare_uses_gt3_latency_when_both_replicas_have_it() {
        let mut fast = entry("fast", "http://fast", "m", 0, 4);
        fast.layer_latencies.insert(2, (5.0, 10.0));
        fast.rtt_ms = Some(100.0); // worse RTT but better latency wins
        let mut slow = entry("slow", "http://slow", "m", 0, 4);
        slow.layer_latencies.insert(2, (50.0, 80.0));
        slow.rtt_ms = Some(1.0);
        assert_eq!(
            compare_servers_for_route(&fast, &slow, 2),
            std::cmp::Ordering::Less,
            "GT3 latency must beat RTT when both replicas have layer stats"
        );
    }

    #[test]
    fn compare_falls_through_to_rtt_when_no_gt3() {
        // Neither replica has GT3 data at layer 2; pick the lower RTT.
        let mut close = entry("close", "http://close", "m", 0, 4);
        close.rtt_ms = Some(1.5);
        close.requests_in_flight = 9; // higher load but lower RTT wins
        let mut far = entry("far", "http://far", "m", 0, 4);
        far.rtt_ms = Some(30.0);
        far.requests_in_flight = 1;
        assert_eq!(
            compare_servers_for_route(&close, &far, 2),
            std::cmp::Ordering::Less,
        );
    }

    #[test]
    fn compare_prefers_replica_with_rtt_data_over_unprobed() {
        let mut probed = entry("probed", "http://p", "m", 0, 4);
        probed.rtt_ms = Some(10.0);
        let unprobed = entry("unprobed", "http://u", "m", 0, 4); // rtt_ms = None
        assert_eq!(
            compare_servers_for_route(&probed, &unprobed, 2),
            std::cmp::Ordering::Less,
        );
        assert_eq!(
            compare_servers_for_route(&unprobed, &probed, 2),
            std::cmp::Ordering::Greater,
        );
    }

    #[test]
    fn compare_falls_through_to_requests_in_flight_when_no_latency_no_rtt() {
        let mut idle = entry("idle", "http://idle", "m", 0, 4);
        idle.requests_in_flight = 1;
        let mut busy = entry("busy", "http://busy", "m", 0, 4);
        busy.requests_in_flight = 8;
        // Neither has layer_latencies nor rtt_ms — fallback runs.
        assert_eq!(
            compare_servers_for_route(&idle, &busy, 2),
            std::cmp::Ordering::Less,
        );
    }

    // ─── ADR-0018: expert-level routing ──────────────────────────────────────

    /// `route_expert` against a dense server returns the same URL as
    /// `route`: dense servers own every expert_id trivially.
    #[test]
    fn route_expert_passthrough_for_dense_servers() {
        let mut state = GridState::default();
        state.register(entry("a", "http://a", "model-a", 0, 5));
        // Dense by default — expert_start == expert_end == 0.
        assert_eq!(
            state.route_expert(Some("model-a"), 3, 7).as_deref(),
            Some("http://a"),
            "dense server must satisfy any expert_id"
        );
        assert_eq!(
            state.route_expert(Some("model-a"), 3, 0).as_deref(),
            Some("http://a"),
        );
        assert_eq!(
            state.route_expert(Some("model-a"), 3, 999).as_deref(),
            Some("http://a"),
        );
    }

    /// Two MoE shards split layer 5's experts 0-3 / 4-7. `route_expert`
    /// must pick the right one for each expert_id.
    #[test]
    fn route_expert_filters_by_expert_range() {
        let mut state = GridState::default();
        let mut lo = entry("lo", "http://lo", "moe", 5, 5);
        lo.expert_start = 0;
        lo.expert_end = 3;
        let mut hi = entry("hi", "http://hi", "moe", 5, 5);
        hi.expert_start = 4;
        hi.expert_end = 7;
        state.register(lo);
        state.register(hi);

        assert_eq!(
            state.route_expert(Some("moe"), 5, 0).as_deref(),
            Some("http://lo"),
        );
        assert_eq!(
            state.route_expert(Some("moe"), 5, 3).as_deref(),
            Some("http://lo"),
        );
        assert_eq!(
            state.route_expert(Some("moe"), 5, 4).as_deref(),
            Some("http://hi"),
        );
        assert_eq!(
            state.route_expert(Some("moe"), 5, 7).as_deref(),
            Some("http://hi"),
        );
        // No server owns expert 99 → miss.
        assert!(state.route_expert(Some("moe"), 5, 99).is_none());
    }

    /// `route_all_experts` batches and short-circuits on the first
    /// uncovered `(layer, expert)` pair.
    #[test]
    fn route_all_experts_short_circuits_on_first_uncovered_pair() {
        let mut state = GridState::default();
        let mut a = entry("a", "http://a", "moe", 0, 0);
        a.expert_start = 0;
        a.expert_end = 3;
        state.register(a);

        let ok = state.route_all_experts(Some("moe"), &[(0, 0), (0, 2)]);
        assert!(matches!(ok, Ok(map) if map.len() == 2));

        let miss = state.route_all_experts(Some("moe"), &[(0, 0), (0, 5), (0, 1)]);
        assert_eq!(miss, Err((0, 5)));
    }

    /// MoE + dense replicas on the same layer: dense replicas serve
    /// every expert_id, so adding a dense fallback alongside an MoE
    /// shard means *every* expert is reachable. Useful for migration
    /// scenarios.
    #[test]
    fn route_expert_dense_replica_serves_as_fallback() {
        let mut state = GridState::default();
        let mut moe_shard = entry("moe", "http://moe", "m", 5, 5);
        moe_shard.expert_start = 0;
        moe_shard.expert_end = 3;
        // Dense replica covers everything.
        let mut dense = entry("dense", "http://dense", "m", 5, 5);
        dense.requests_in_flight = 100; // make it least-preferred so the MoE shard wins ties
        state.register(moe_shard);
        state.register(dense);

        // Expert 2: both own it (MoE explicitly + dense by default).
        // Comparator picks lowest in-flight → MoE shard (0 vs 100).
        assert_eq!(
            state.route_expert(Some("m"), 5, 2).as_deref(),
            Some("http://moe"),
        );
        // Expert 99: MoE doesn't own it; dense wins.
        assert_eq!(
            state.route_expert(Some("m"), 5, 99).as_deref(),
            Some("http://dense"),
        );
    }

    // ── ADR-0021: route_with_rank / route_expert_with_rank ───────────────────

    #[test]
    fn route_with_rank_returns_empty_when_no_owner() {
        let state = GridState::default();
        assert!(state.route_with_rank(Some("model-a"), 0, 2).is_empty());
    }

    #[test]
    fn route_with_rank_returns_single_replica_when_only_one_owner() {
        let mut state = GridState::default();
        state.register(entry("a", "http://a", "m", 0, 4));
        let ranked = state.route_with_rank(Some("m"), 2, 2);
        assert_eq!(ranked, vec!["http://a".to_string()]);
    }

    #[test]
    fn route_with_rank_orders_by_in_flight_when_no_latency_data() {
        let mut state = GridState::default();
        state.register(entry("a", "http://a", "m", 0, 4));
        state.register(entry("b", "http://b", "m", 0, 4));
        // Make b less loaded than a.
        state.update_heartbeat("a", 0.0, 0, 10, Vec::new(), 0.0);
        state.update_heartbeat("b", 0.0, 0, 1, Vec::new(), 0.0);

        let ranked = state.route_with_rank(Some("m"), 2, 2);
        assert_eq!(ranked[0], "http://b", "best replica must be first");
        assert_eq!(ranked[1], "http://a", "second-best must follow");
    }

    #[test]
    fn route_with_rank_respects_max_bound() {
        let mut state = GridState::default();
        for (i, url) in [("a", "http://a"), ("b", "http://b"), ("c", "http://c")] {
            state.register(entry(i, url, "m", 0, 4));
        }
        assert_eq!(state.route_with_rank(Some("m"), 2, 1).len(), 1);
        assert_eq!(state.route_with_rank(Some("m"), 2, 2).len(), 2);
        assert_eq!(state.route_with_rank(Some("m"), 2, 99).len(), 3);
        assert!(state.route_with_rank(Some("m"), 2, 0).is_empty());
    }

    #[test]
    fn route_with_rank_drops_saturated_replicas() {
        let mut state = GridState::default();
        state.register(entry("a", "http://a", "m", 0, 4));
        state.register(entry("b", "http://b", "m", 0, 4));
        state.update_heartbeat("a", 0.0, 0, 100, Vec::new(), 0.0);
        state.update_heartbeat("b", 0.0, 0, 5, Vec::new(), 0.0);
        state.set_saturation_ceiling(Some(50));

        let ranked = state.route_with_rank(Some("m"), 2, 2);
        // a is over the ceiling, so it's filtered out.
        assert_eq!(ranked, vec!["http://b".to_string()]);
    }

    #[test]
    fn route_expert_with_rank_filters_to_owning_experts() {
        use super::super::testing::entry_with_experts;
        let mut state = GridState::default();
        state.register(entry_with_experts("e1", "http://e1", "m", 5, 5, 0, 3));
        state.register(entry_with_experts("e2", "http://e2", "m", 5, 5, 0, 3));
        state.register(entry_with_experts("e3", "http://e3", "m", 5, 5, 4, 7));

        let ranked_expert_0 = state.route_expert_with_rank(Some("m"), 5, 0, 4);
        // Only e1 and e2 own expert 0.
        assert_eq!(ranked_expert_0.len(), 2);
        assert!(ranked_expert_0
            .iter()
            .all(|u| u == "http://e1" || u == "http://e2"));
        let ranked_expert_5 = state.route_expert_with_rank(Some("m"), 5, 5, 4);
        assert_eq!(ranked_expert_5, vec!["http://e3".to_string()]);
    }
}
