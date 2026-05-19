# ADR-0017 — Prometheus `/metrics` Endpoint for `larql-router`

**Status:** Accepted — shipped 2026-05-16.
**Depends on:** ADR-0004 (grid), ADR-0011 (rebalancer), ADR-0013 (RTT probe), ADR-0014 (hot-shard), ADR-0016 (module organization).
**Implementation:** `crates/larql-router/src/metrics.rs`, instrumentation hooks in `grid/`, `tasks/`, `http.rs`.

---

## Context

The router emits structured `tracing` logs for every state-changing
operation (server join/leave, rebalancer action, RTT probe outcome,
HTTP request). Logs are sufficient for *post-mortem* analysis — grep
for an incident, follow the timeline — but inadequate for *live*
operations:

- "Is the grid healthy right now?" requires opening a session and
  running `larql-router status`. There's no dashboard.
- "Did the rebalancer fire 3 times or 30 times in the last hour?"
  requires log aggregation infrastructure (Loki, ELK).
- "What's the p99 of `/v1/walk-ffn`?" requires log parsing or
  external probing.
- Alerts can only fire on log patterns, which is brittle.

The standard production answer is Prometheus-format metrics scraped
periodically by an external collector. Once the router exposes a
`/metrics` endpoint, the operator's existing observability stack
(Grafana, Alertmanager, etc.) handles everything else.

---

## Decision

### Library: `prometheus` crate (v0.13)

Battle-tested, small dependency surface, idiomatic for tonic/axum
stacks. The `Encoder`/`Registry`/`Opts` API is stable. The newer
`prometheus-client` crate is OpenMetrics-compliant and more
type-safe, but `prometheus` is more widely deployed and integrates
with existing Grafana dashboards out of the box.

Alternatives considered + rejected:
- **`prometheus-client`** — newer, less code in the wild. The
  type-state win doesn't outweigh the integration risk for what is
  essentially a 200-line wiring file.
- **`metrics` facade + `metrics-exporter-prometheus`** — decoupled,
  but the facade indirection is unnecessary when we have exactly one
  exporter target.
- **Hand-rolled Prometheus text format** — easy to get wrong on edge
  cases (escaping, exemplars, `# HELP`/`# TYPE` headers).

### Endpoint: `GET /metrics` on the existing axum router

Add a `.route("/metrics", get(handle_metrics))` to `build_router`.
Same listener, same auth (none today — `/metrics` is unauth like
`/v1/health`). The encoder produces the standard
`text/plain; version=0.0.4` Prometheus format.

### Metric set (tier 1 — minimum viable)

| Name | Type | Labels | Source |
|---|---|---|---|
| `larql_router_build_info` | Gauge | `version`, `crate_version` | Static, set at startup; value always `1` |
| `larql_router_grid_servers` | Gauge | `state` ∈ {`serving`, `available`} | Scrape-time from `GridState` |
| `larql_router_grid_models` | Gauge | — | Scrape-time |
| `larql_router_grid_coverage_gaps` | Gauge | — | Scrape-time |
| `larql_router_grid_elevated_ranges` | Gauge | — | Scrape-time |
| `larql_router_target_replicas` | Gauge | — | Scrape-time |
| `larql_router_grid_registers_total` | Counter | — | Event-driven |
| `larql_router_grid_deregisters_total` | Counter | `reason` ∈ {`stream_close`, `dropping`, `stale`} | Event-driven |
| `larql_router_rebalancer_actions_total` | Counter | `action` ∈ {`replicate`, `drop`, `elevate`, `demote`, `evict`, `unassign_imbalance`} | Event-driven |
| `larql_router_rtt_probes_total` | Counter | `outcome` ∈ {`success`, `non_2xx`, `error`} | Event-driven |
| `larql_router_walk_ffn_requests_total` | Counter | `status` ∈ {`success`, `error_4xx`, `error_5xx`} | Event-driven |
| `larql_router_walk_ffn_duration_seconds` | Histogram | — | Event-driven |

### Cardinality discipline

Labels are bounded — no per-server, per-model, or per-layer label is
emitted, even when tempting:

- **No `model_id` label** — operators can run thousands of models on
  one grid. Per-model cardinality would explode Prometheus memory.
  Model-level visibility comes from `larql-router status` instead.
- **No `server_id` label** — same reason. Server churn over time
  generates unbounded label values.
- **No `layer_id` label on per-shard metrics** — 30-62 layers per
  model × N models would be too many.

Bounded label sets ({`serving`/`available`}, action types, etc.) are
fine — cardinality is the number of static values.

### Counters: event-driven, plumbed through `RouterMetrics`

A new `RouterMetrics` struct owns the metric handles and the
`Registry`. It's wrapped in `Arc<RouterMetrics>` and passed
alongside `AppState` / through `tasks::*` configs.

- `GridState::register` / `deregister` bumps the relevant counters.
- `tasks::rebalancer::{hot_shard, replication, eviction, imbalance}::check_*`
  bumps action counters on each fired action.
- `tasks::rtt_probe::probe_one` bumps the probe outcome counter.
- `http::handle_walk_ffn` observes the duration histogram and bumps
  the request counter with the right status label.

### Gauges: scrape-time via a custom `Collector`

Gauges that derive from `GridState` (server counts, gap count,
elevation set size, model count) are computed **at scrape time** by
a custom Prometheus `Collector` impl on `GridStateCollector` rather
than being kept in lock-step with state mutations. Reasons:

1. **Single source of truth.** `GridState` is already authoritative.
   Updating both the state and a gauge for every mutation is
   maintenance burden + bug surface.
2. **Cheap.** Scrape happens every 15-30 s; computing
   `coverage_gaps().len()` once per scrape is negligible against
   the state's per-request cost.
3. **Resilient.** If a mutation path forgets to update a gauge,
   we'd serve stale values. Scrape-time derivation can't drift.

The `Collector::collect` impl takes a read lock on `GridState`,
reads the four numbers, and emits four `MetricFamily` instances.
Async-unsafe but called from a sync collector trait; we use
`tokio::runtime::Handle::current().block_on(...)` to bridge.
**Actually** simpler: keep the gauges as plain numbers updated
from the rebalancer tick (every 30 s) — same effective scrape
cadence, no async-bridging.

We'll go with the second variant: rebalancer-tick-driven gauge
update via a `refresh_gauges(&GridState)` method called at the end
of each rebalancer tick. One write to each gauge per 30 s. The gauges
are always-defined values that survive scrapes between rebalancer
ticks unchanged.

### Bucket layout for `walk_ffn_duration_seconds`

Default buckets `[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0,
2.5, 5.0, 10.0]` (Prometheus default), covering the typical 5 ms –
10 s range for FFN dispatch. p99 falls in the 0.5-2.5 s bucket on
the current Gemma 3 4B grid; alerts can fire on 0.5 s breach.

---

## Alternatives Considered

### Push-based metrics (StatsD, etc.)

Rejected — production target is Prometheus, no operator demand for
push-based. Pull is also cheaper for the router (no client
connection upkeep).

### Per-request `tracing` events instead of metrics

Already there — these are the structured logs. Metrics are
*aggregated* visibility on top; they don't replace logs, they
complement them.

### Exposed metrics endpoint behind auth

Rejected for now — `/metrics` follows the same auth model as
`/v1/health`: unauth, intended for an internal scraper. Operators
needing auth would put both endpoints behind a private network or a
reverse proxy with auth. A future `--metrics-port` separation flag
could split metrics onto a private port; not in this ADR.

---

## Consequences

### Positive

- Operations get live grid visibility without re-running admin
  commands.
- Alerts can fire on quantitative thresholds (gaps > 0, rebalancer
  evictions > N/hour, walk_ffn p99 > 1 s).
- Dashboards become possible — Grafana can render replication
  health, hot-shard elevations over time, RTT-probe miss rate.
- The bench numbers in README become re-checkable from production
  data (walk-ffn p99 from the histogram vs the bench's synthetic
  number).

### Negative

- New dependency (`prometheus` crate, ~10 transitive crates). The
  binary grows by ~200 KB stripped.
- A few new instrumentation call sites need to be maintained
  alongside the logic they observe. The pattern is consistent
  (`metrics.foo.inc()` or `metrics.foo.observe(d)`), so the
  maintenance cost is low.
- One more public surface (`/metrics`) for operators to know about.

### Neutral

- The metric set is intentionally minimal. Future ADRs can extend
  it; bounded-cardinality discipline must be preserved.

---

## Implementation pointers

| File | Role |
|---|---|
| `crates/larql-router/Cargo.toml` | Add `prometheus = "0.13"` dependency |
| `crates/larql-router/src/metrics.rs` | `RouterMetrics` struct + handles + `Registry` + `refresh_gauges` |
| `crates/larql-router/src/lib.rs` | `pub mod metrics;` |
| `crates/larql-router/src/http.rs` | `/metrics` route, `handle_metrics` handler, observe walk-ffn duration + status |
| `crates/larql-router/src/grid/mod.rs::GridState::register` | `metrics.grid_registers.inc()` |
| `crates/larql-router/src/grid/mod.rs::GridState::deregister` | `metrics.grid_deregisters.with_label_values(...).inc()` |
| `crates/larql-router/src/grid/hot_shard.rs::{mark_elevated, demote_elevated}` | `metrics.rebalancer_actions.with_label_values(...).inc()` |
| `crates/larql-router/src/tasks/rebalancer/*.rs` | counter bumps in `check_*` |
| `crates/larql-router/src/tasks/rebalancer/mod.rs::rebalancer_task` | call `metrics.refresh_gauges(&state)` at end of each tick |
| `crates/larql-router/src/tasks/rtt_probe.rs::probe_one` | outcome counter bump |
| `crates/larql-router/src/main.rs` | construct `Arc<RouterMetrics>` at startup, pass to `AppState` and to `rebalancer::spawn` / `rtt_probe::spawn` |

### Test coverage

- Unit tests in `metrics.rs::tests` cover: registry assembly, counter
  increment, gauge refresh, histogram observe, encoding round-trip.
- Integration test in `tests/test_http_handlers.rs` exercises the
  `/metrics` route and verifies a couple of counter values change
  after grid operations.
