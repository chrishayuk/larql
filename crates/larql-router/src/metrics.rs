//! Prometheus metrics surface for `larql-router` (ADR-0017).
//!
//! Owns the [`Registry`] and the set of counter / gauge / histogram
//! handles that the rest of the crate increments. Construction is
//! one-shot at process startup; the same `Arc<RouterMetrics>` is
//! shared across `AppState`, the rebalancer task, and the RTT probe
//! task. `/metrics` reads off the registry via
//! [`encode_metrics_text`].
//!
//! Cardinality is bounded — labels are static enums
//! (`{serving,available}`, `{success,non_2xx,error}`, etc.). No
//! `model_id`, `server_id`, or `layer_id` labels are emitted.

use std::sync::Arc;

use prometheus::{
    Encoder, HistogramOpts, HistogramVec, IntCounter, IntCounterVec, IntGauge, IntGaugeVec, Opts,
    Registry, TextEncoder,
};

use crate::grid::GridState;

/// All Prometheus metric handles plus the registry they're attached to.
///
/// Wrap in `Arc` at construction; clone the `Arc` into every owner.
pub struct RouterMetrics {
    pub registry: Registry,

    // ── Gauges (refreshed from `GridState` at the end of each rebalancer tick) ──
    pub grid_servers: IntGaugeVec, // state = {serving, available}
    pub grid_models: IntGauge,
    pub grid_coverage_gaps: IntGauge,
    pub grid_elevated_ranges: IntGauge,
    pub target_replicas: IntGauge,
    /// ADR-0018 — bounded-cardinality MoE health gauge.
    /// `kind ∈ {dense, moe}` — dense servers have
    /// `expert_start==expert_end==0`; MoE servers have a real expert
    /// range. The two labels are exhaustive and bounded; no expert IDs
    /// or per-shard cardinality is emitted.
    pub grid_shard_kind: IntGaugeVec, // kind = {dense, moe}

    // ── Counters (event-driven) ────────────────────────────────────────────────
    pub grid_registers_total: IntCounter,
    pub grid_deregisters_total: IntCounterVec, // reason = {stream_close, dropping, stale, drain}
    pub rebalancer_actions_total: IntCounterVec, // action = {replicate, drop, elevate, demote, evict, unassign_imbalance}
    pub rtt_probes_total: IntCounterVec,         // outcome = {success, non_2xx, error}
    pub walk_ffn_requests_total: IntCounterVec,  // status = {success, error_4xx, error_5xx}
    /// ADR-0020 — count of dispatches that 503'd because every
    /// replica covering the requested layer (or `(layer, expert)`)
    /// was at or above the configured saturation ceiling. Rising
    /// values mean the grid needs more capacity OR the ceiling is
    /// set too low.
    pub route_saturation_total: IntCounter,
    /// ADR-0021 — count of hedged sub-requests where the primary's
    /// reply didn't arrive within `--hedge-after-ms` and the
    /// secondary replica was dispatched. Operator's signal of how
    /// often the tail-clipping path is active.
    pub route_hedge_fires_total: IntCounter,
    /// ADR-0021 — count of times the hedge actually beat the
    /// primary (the slow-primary case it's designed for).
    /// `route_hedge_wins_total / route_hedge_fires_total` is the
    /// useful ratio: ~1.0 means hedging clips real tail; ~0.0 means
    /// it's just doubling wire load.
    pub route_hedge_wins_total: IntCounter,

    // ── Histograms (event-driven) ──────────────────────────────────────────────
    pub walk_ffn_duration_seconds: HistogramVec, // (currently no labels; HistogramVec used for future expansion)
}

impl RouterMetrics {
    /// Build a fresh registry plus every metric handle. Returns an
    /// `Arc` because every owner needs cheap shared access.
    pub fn new() -> Arc<Self> {
        let registry = Registry::new();

        // ── build_info as a static gauge always at 1, useful for joining
        //     scrape data with the build that produced it.
        let build_info = IntGaugeVec::new(
            Opts::new(
                "larql_router_build_info",
                "Constant 1 gauge with version labels — used for joining metrics to a release.",
            ),
            &["version"],
        )
        .expect("build_info opts must be valid");
        build_info
            .with_label_values(&[env!("CARGO_PKG_VERSION")])
            .set(1);
        registry
            .register(Box::new(build_info))
            .expect("build_info must register");

        let grid_servers = IntGaugeVec::new(
            Opts::new(
                "larql_router_grid_servers",
                "Number of servers in the grid, split by state.",
            ),
            &["state"],
        )
        .unwrap();
        registry.register(Box::new(grid_servers.clone())).unwrap();

        let grid_models = IntGauge::new(
            "larql_router_grid_models",
            "Number of distinct model_ids served by the grid.",
        )
        .unwrap();
        registry.register(Box::new(grid_models.clone())).unwrap();

        let grid_coverage_gaps = IntGauge::new(
            "larql_router_grid_coverage_gaps",
            "Number of uncovered layer ranges across all models.",
        )
        .unwrap();
        registry
            .register(Box::new(grid_coverage_gaps.clone()))
            .unwrap();

        let grid_elevated_ranges = IntGauge::new(
            "larql_router_grid_elevated_ranges",
            "Number of (model, layer-range) entries currently flagged hot.",
        )
        .unwrap();
        registry
            .register(Box::new(grid_elevated_ranges.clone()))
            .unwrap();

        let target_replicas = IntGauge::new(
            "larql_router_target_replicas",
            "Configured --target-replicas (per-range replica goal).",
        )
        .unwrap();
        registry
            .register(Box::new(target_replicas.clone()))
            .unwrap();

        let grid_shard_kind = IntGaugeVec::new(
            Opts::new(
                "larql_router_grid_shard_kind",
                "Number of serving shards split by ownership kind (dense / moe).",
            ),
            &["kind"],
        )
        .unwrap();
        registry
            .register(Box::new(grid_shard_kind.clone()))
            .unwrap();

        let grid_registers_total = IntCounter::new(
            "larql_router_grid_registers_total",
            "Total server registrations into the serving set.",
        )
        .unwrap();
        registry
            .register(Box::new(grid_registers_total.clone()))
            .unwrap();

        let grid_deregisters_total = IntCounterVec::new(
            Opts::new(
                "larql_router_grid_deregisters_total",
                "Total server deregistrations from the serving set, split by reason.",
            ),
            &["reason"],
        )
        .unwrap();
        registry
            .register(Box::new(grid_deregisters_total.clone()))
            .unwrap();

        let rebalancer_actions_total = IntCounterVec::new(
            Opts::new(
                "larql_router_rebalancer_actions_total",
                "Rebalancer tick outcomes, split by action.",
            ),
            &["action"],
        )
        .unwrap();
        registry
            .register(Box::new(rebalancer_actions_total.clone()))
            .unwrap();

        let rtt_probes_total = IntCounterVec::new(
            Opts::new(
                "larql_router_rtt_probes_total",
                "Active-probe outcomes, split by result class.",
            ),
            &["outcome"],
        )
        .unwrap();
        registry
            .register(Box::new(rtt_probes_total.clone()))
            .unwrap();

        let walk_ffn_requests_total = IntCounterVec::new(
            Opts::new(
                "larql_router_walk_ffn_requests_total",
                "Total /v1/walk-ffn requests served, split by status class.",
            ),
            &["status"],
        )
        .unwrap();
        registry
            .register(Box::new(walk_ffn_requests_total.clone()))
            .unwrap();

        let route_saturation_total = IntCounter::new(
            "larql_router_route_saturation_total",
            "Count of dispatches that 503'd because every replica was at \
             or above the configured ADR-0020 saturation ceiling.",
        )
        .unwrap();
        registry
            .register(Box::new(route_saturation_total.clone()))
            .unwrap();

        let route_hedge_fires_total = IntCounter::new(
            "larql_router_route_hedge_fires_total",
            "ADR-0021 — count of hedged sub-requests where the primary's \
             reply didn't arrive within --hedge-after-ms and a secondary \
             replica was dispatched.",
        )
        .unwrap();
        registry
            .register(Box::new(route_hedge_fires_total.clone()))
            .unwrap();

        let route_hedge_wins_total = IntCounter::new(
            "larql_router_route_hedge_wins_total",
            "ADR-0021 — count of times the hedge actually beat the primary. \
             route_hedge_wins_total / route_hedge_fires_total ≈ 1 means \
             hedging is clipping real tail latency; ≈ 0 means it's just \
             doubling wire load.",
        )
        .unwrap();
        registry
            .register(Box::new(route_hedge_wins_total.clone()))
            .unwrap();

        let walk_ffn_duration_seconds = HistogramVec::new(
            HistogramOpts::new(
                "larql_router_walk_ffn_duration_seconds",
                "End-to-end /v1/walk-ffn handler duration in seconds.",
            )
            .buckets(vec![
                0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
            ]),
            &[],
        )
        .unwrap();
        registry
            .register(Box::new(walk_ffn_duration_seconds.clone()))
            .unwrap();

        // Pre-touch every known label combination so the metric
        // families appear in `gather()` output with a value of 0 even
        // before any event has fired. Without this, a scrape against
        // a freshly-started router that hasn't seen any traffic would
        // miss the family entirely — and Prometheus alerts that
        // reference a missing metric behave inconsistently across
        // versions.
        for state in ["serving", "available"] {
            grid_servers.with_label_values(&[state]).set(0);
        }
        for kind in ["dense", "moe"] {
            grid_shard_kind.with_label_values(&[kind]).set(0);
        }
        for reason in ["stream_close", "dropping", "stale"] {
            grid_deregisters_total.with_label_values(&[reason]);
        }
        for action in [
            "replicate",
            "drop",
            "elevate",
            "demote",
            "evict",
            "unassign_imbalance",
        ] {
            rebalancer_actions_total.with_label_values(&[action]);
        }
        for outcome in ["success", "non_2xx", "error"] {
            rtt_probes_total.with_label_values(&[outcome]);
        }
        for status in ["success", "error_4xx", "error_5xx"] {
            walk_ffn_requests_total.with_label_values(&[status]);
        }
        // `walk_ffn_duration_seconds` is a HistogramVec with no labels
        // today, but the constructor takes a `&[]` label set; pre-touch
        // it so gather() reports the bucket layout immediately.
        let _ = walk_ffn_duration_seconds.with_label_values(&[]);

        Arc::new(Self {
            registry,
            grid_servers,
            grid_models,
            grid_coverage_gaps,
            grid_elevated_ranges,
            target_replicas,
            grid_shard_kind,
            grid_registers_total,
            grid_deregisters_total,
            rebalancer_actions_total,
            rtt_probes_total,
            walk_ffn_requests_total,
            route_saturation_total,
            route_hedge_fires_total,
            route_hedge_wins_total,
            walk_ffn_duration_seconds,
        })
    }

    /// Snapshot the grid's current counts into the gauges. Called at
    /// the end of each rebalancer tick (every 30 s by default) so
    /// scrapes between ticks see the most recent rebalancer-tick
    /// reading.
    pub fn refresh_gauges(&self, state: &GridState) {
        let serving = state.servers().count() as i64;
        let available = state.has_available_servers() as i64; // 0/1 — see below
                                                              // `has_available_servers` returns bool today, not a count.
                                                              // Until the API exposes a count, the gauge reports 0/1 which
                                                              // is still useful for alerting on "pool empty". A future
                                                              // change to `available_server_count()` will swap this in.
        self.grid_servers
            .with_label_values(&["serving"])
            .set(serving);
        self.grid_servers
            .with_label_values(&["available"])
            .set(available);

        let mut models = std::collections::HashSet::new();
        for (_, e) in state.servers() {
            models.insert(e.model_id.clone());
        }
        self.grid_models.set(models.len() as i64);

        self.grid_coverage_gaps
            .set(state.coverage_gaps().len() as i64);
        self.grid_elevated_ranges
            .set(state.elevated_ranges_snapshot().len() as i64);
        self.target_replicas.set(state.target_replicas() as i64);

        // ADR-0018 — count shards by ownership kind.
        let mut dense = 0i64;
        let mut moe = 0i64;
        for (_, e) in state.servers() {
            if e.is_dense() {
                dense += 1;
            } else {
                moe += 1;
            }
        }
        self.grid_shard_kind
            .with_label_values(&["dense"])
            .set(dense);
        self.grid_shard_kind.with_label_values(&["moe"]).set(moe);
    }
}

/// Encode the current registry as Prometheus text format. Used by
/// the `/metrics` axum handler.
pub fn encode_metrics_text(metrics: &RouterMetrics) -> Result<String, prometheus::Error> {
    let mf = metrics.registry.gather();
    let mut buf = Vec::with_capacity(4096);
    let encoder = TextEncoder::new();
    encoder.encode(&mf, &mut buf)?;
    String::from_utf8(buf).map_err(|e| prometheus::Error::Msg(format!("utf8: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_registers_every_metric_family() {
        let m = RouterMetrics::new();
        let families = m.registry.gather();
        let names: Vec<&str> = families.iter().map(|f| f.get_name()).collect();
        // Every documented metric must be present after construction.
        for required in [
            "larql_router_build_info",
            "larql_router_grid_servers",
            "larql_router_grid_models",
            "larql_router_grid_coverage_gaps",
            "larql_router_grid_elevated_ranges",
            "larql_router_target_replicas",
            "larql_router_grid_shard_kind",
            "larql_router_grid_registers_total",
            "larql_router_grid_deregisters_total",
            "larql_router_rebalancer_actions_total",
            "larql_router_rtt_probes_total",
            "larql_router_walk_ffn_requests_total",
            "larql_router_route_saturation_total",
            "larql_router_route_hedge_fires_total",
            "larql_router_route_hedge_wins_total",
            "larql_router_walk_ffn_duration_seconds",
        ] {
            assert!(names.contains(&required), "missing metric {required}");
        }
    }

    #[test]
    fn build_info_label_carries_crate_version() {
        let m = RouterMetrics::new();
        let text = encode_metrics_text(&m).unwrap();
        let needle = format!(
            "larql_router_build_info{{version=\"{}\"}} 1",
            env!("CARGO_PKG_VERSION")
        );
        assert!(
            text.contains(&needle),
            "build_info missing version label; got:\n{text}"
        );
    }

    #[test]
    fn counter_increments_are_visible_in_encoded_text() {
        let m = RouterMetrics::new();
        m.grid_registers_total.inc();
        m.grid_registers_total.inc();
        m.grid_registers_total.inc();
        let text = encode_metrics_text(&m).unwrap();
        assert!(text.contains("larql_router_grid_registers_total 3"));
    }

    #[test]
    fn labelled_counter_increments_to_correct_label() {
        let m = RouterMetrics::new();
        m.rebalancer_actions_total
            .with_label_values(&["replicate"])
            .inc();
        m.rebalancer_actions_total
            .with_label_values(&["drop"])
            .inc();
        m.rebalancer_actions_total
            .with_label_values(&["drop"])
            .inc();
        let text = encode_metrics_text(&m).unwrap();
        assert!(text.contains("larql_router_rebalancer_actions_total{action=\"replicate\"} 1"));
        assert!(text.contains("larql_router_rebalancer_actions_total{action=\"drop\"} 2"));
    }

    #[test]
    fn refresh_gauges_reads_grid_state() {
        use crate::grid::testing::entry;

        let m = RouterMetrics::new();
        let mut state = GridState::default();
        state.set_target_replicas(2);
        state.register(entry("a", "http://a", "model-x", 0, 4));
        state.register(entry("b", "http://b", "model-x", 5, 9));
        state.mark_elevated("model-x", 0, 4, 0, 0);

        m.refresh_gauges(&state);

        let text = encode_metrics_text(&m).unwrap();
        assert!(text.contains("larql_router_grid_servers{state=\"serving\"} 2"));
        assert!(text.contains("larql_router_grid_models 1"));
        assert!(text.contains("larql_router_grid_coverage_gaps 0"));
        assert!(text.contains("larql_router_grid_elevated_ranges 1"));
        assert!(text.contains("larql_router_target_replicas 2"));
        // Two dense servers, no MoE.
        assert!(text.contains("larql_router_grid_shard_kind{kind=\"dense\"} 2"));
        assert!(text.contains("larql_router_grid_shard_kind{kind=\"moe\"} 0"));
    }

    /// ADR-0018: with a mix of dense and MoE shards, the
    /// `grid_shard_kind` gauge splits them correctly.
    #[test]
    fn refresh_gauges_counts_moe_and_dense_shards_separately() {
        use crate::grid::testing::entry;

        let m = RouterMetrics::new();
        let mut state = GridState::default();
        state.register(entry("dense-a", "http://a", "moe", 0, 0));
        let mut moe_lo = entry("moe-lo", "http://lo", "moe", 1, 1);
        moe_lo.expert_start = 0;
        moe_lo.expert_end = 3;
        state.register(moe_lo);
        let mut moe_hi = entry("moe-hi", "http://hi", "moe", 1, 1);
        moe_hi.expert_start = 4;
        moe_hi.expert_end = 7;
        state.register(moe_hi);

        m.refresh_gauges(&state);

        let text = encode_metrics_text(&m).unwrap();
        assert!(text.contains("larql_router_grid_shard_kind{kind=\"dense\"} 1"));
        assert!(text.contains("larql_router_grid_shard_kind{kind=\"moe\"} 2"));
    }

    #[test]
    fn histogram_observe_lands_in_bucket() {
        let m = RouterMetrics::new();
        // 50 ms — should land in the le="0.1" bucket.
        m.walk_ffn_duration_seconds
            .with_label_values(&[])
            .observe(0.05);
        let text = encode_metrics_text(&m).unwrap();
        assert!(text.contains("larql_router_walk_ffn_duration_seconds_count 1"));
        // Anything < 0.1 must increment that bucket.
        assert!(text.contains("le=\"0.1\""));
    }

    #[test]
    fn refresh_gauges_handles_empty_grid() {
        let m = RouterMetrics::new();
        let state = GridState::default();
        m.refresh_gauges(&state);
        let text = encode_metrics_text(&m).unwrap();
        assert!(text.contains("larql_router_grid_servers{state=\"serving\"} 0"));
        assert!(text.contains("larql_router_grid_models 0"));
        assert!(text.contains("larql_router_target_replicas 1")); // default
    }
}
