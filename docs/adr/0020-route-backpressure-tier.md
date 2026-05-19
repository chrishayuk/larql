# ADR-0020 — Backpressure / Saturation Tier in `route()`

**Status:** Accepted — shipped 2026-05-16.
**Depends on:** ADR-0013 (three-tier routing comparator), ADR-0017
(metrics).
**Amends:** ADR-0013 — adds a saturation filter that runs *before*
the three-tier comparator.

---

## Context

The ADR-0013 routing cascade (GT3 layer-latency → active-probe RTT
→ requests-in-flight) always returns *some* replica from the
candidate set, even when every replica is overloaded. If every
replica of a layer is already running 100 in-flight requests, the
"least loaded" one still gets the next request — adding to the
queue rather than shedding load.

Real symptoms this causes:

- **Tail-latency blowup under load spikes.** A misconfigured
  client that floods the grid pushes every replica into deep queue
  territory; `route()` keeps routing because it has no "no, sorry"
  exit.
- **Cascade failures.** A slow shard backs up requests upstream; if
  the rebalancer hasn't yet pulled a spare or the spare pool is
  empty, the slow shard's queue grows unbounded.
- **Misleading hot-shard signal.** `req_per_sec` keeps climbing
  because requests are being accepted; the rebalancer's hot-shard
  elevation (ADR-0014) reacts, but if no spare exists, the symptom
  doesn't clear.

The fix is straightforward: give `route()` a way to say "all
replicas are saturated — go away" so the dispatch layer can return
**503** instead of piling more load onto an already-failing shard.

---

## Decision

Add a **per-replica in-flight ceiling** to `GridState`. When set, a
replica with `requests_in_flight ≥ ceiling` is treated as
saturated and **filtered out of the candidate set** *before* the
three-tier comparator runs.

```
                     all replicas for (model, layer)
                                 │
                                 ▼
              ┌──────────────────────────────────────┐
   ADR-0020 → │  filter: drop replicas where         │
              │  requests_in_flight ≥ ceiling        │
              └──────────────────────────────────────┘
                                 │
                                 ▼
              ┌──────────────────────────────────────┐
   ADR-0013 → │  three-tier comparator:              │
              │  GT3 lat → RTT probe → in-flight     │
              └──────────────────────────────────────┘
                                 │
                                 ▼
                           pick winner
                  or `None` if filter ate everything
```

When the filter empties the candidate set, `route()` returns
`None`. The dispatch layer interprets `None` as **HTTP 503** with a
`Retry-After: 0.5` header — the client should back off briefly
and retry.

### Configuration

New `GridState` field + setter:

```rust
pub struct GridState {
    /// ...
    /// ADR-0020 saturation ceiling — per-replica in-flight count
    /// above which the replica is filtered out of routing. `None`
    /// disables the filter (the pre-ADR-0020 behavior).
    saturation_ceiling: Option<u32>,
}

impl GridState {
    pub fn set_saturation_ceiling(&mut self, ceiling: Option<u32>);
    pub fn saturation_ceiling(&self) -> Option<u32>;
}
```

New CLI flag:

```
--saturation-ceiling N   per-replica in-flight ceiling. When all
                         replicas of a layer are ≥ N, the router
                         503s instead of piling on. Default
                         disabled (matches pre-ADR-0020 behavior).
```

Picking a value: start at `2 × target_replicas × expected_rps_per_replica × p99_latency_s`.
For a typical grid: `target_replicas=2`, 50 RPS/replica, 200 ms
p99 → ceiling ≈ `2 × 2 × 50 × 0.2 = 40`. Operators bench-tune
from there.

### Metric

ADR-0017 grows one new counter, ADR-0020-specific:

```
larql_router_route_saturation_total — count of route() calls that
                                      returned None because all
                                      replicas hit the ceiling.
```

Unlabeled counter (cardinality bounded at 1). Alerting on this
rising means either the grid needs more replicas or the ceiling
is set too low.

### Default: disabled

`saturation_ceiling = None` is the default — operators must
explicitly opt in. The pre-ADR-0020 grid keeps working without
behavior change. The opt-in choice is conservative: an
incorrectly-low ceiling can 503 healthy traffic; making operators
choose a value is the safer default.

---

## Alternatives Considered

### A separate tier inside the comparator (not a filter)

Make saturation the **4th tier** of the cascade (GT3 → RTT →
in-flight → saturation-as-tiebreaker). The cascade would still
pick a replica, just preferring less-saturated ones.

Rejected because it doesn't shed load. The whole point of this
ADR is the 503 escape hatch when every replica is overloaded; a
tiebreaker tier still routes the request.

### Saturation derived from `req_per_sec` instead of `in_flight`

`requests_in_flight` measures queue depth; `req_per_sec` measures
arrival rate. Either could be a saturation signal.

`in_flight` is the right one for backpressure because it
**directly reflects what's stuck in the server's request queue**.
Arrival rate misses backpressure when the server is processing
requests slowly — high `in_flight` with declining `req_per_sec` is
exactly the failure mode we want to detect.

### Server-reported saturation flag

Add a `saturated: bool` field to `HeartbeatMsg` so the server can
declare "I'm full". More accurate (the server knows its own queue
depth) but couples the rebalancer to server-side health policy.

Rejected for v1 because `requests_in_flight` is already a faithful
proxy and avoids the proto change. A future ADR can add the
server-side flag if real workloads show the in-flight signal
isn't sharp enough.

### Per-layer ceiling

Different layers have different per-replica costs (LM head vs
attention vs FFN). A per-layer ceiling could be more
discriminating.

Rejected for v1 — adds config complexity without a clear win.
Operators can tune the single ceiling against the slowest layer.

---

## Consequences

### Positive

- `route()` has an explicit "all replicas saturated" exit, so the
  dispatch layer can shed load gracefully.
- Pairs naturally with hot-shard elevation (ADR-0014): saturation
  signals that elevation hasn't pulled a spare fast enough, and a
  monitoring alert on `larql_router_route_saturation_total` flags
  the gap.
- The filter runs before the comparator, so the comparator's cost
  is unchanged on the happy path (no extra computation).

### Negative

- New config knob. Operators have to choose a value or leave the
  filter disabled.
- Misconfiguration risk: a too-low ceiling 503s healthy traffic;
  the metric makes this observable, but the consequences land on
  real users.
- The `Retry-After: 0.5` header is a hint, not a guarantee.
  Clients that ignore it (curl, custom integrations) will retry
  immediately and re-saturate.

### Neutral

- Comparator semantics unchanged — the filter is a separate stage.
- Dense + MoE routing both pick up the filter; the saturation
  signal is the same regardless of shape.

---

## Implementation pointers

| File | Role |
|---|---|
| `crates/larql-router/src/grid/mod.rs::GridState` | `saturation_ceiling: Option<u32>` field + setter/getter |
| `crates/larql-router/src/grid/routing.rs::route` | Drop replicas above the ceiling before the `min_by` comparator |
| `crates/larql-router/src/grid/routing.rs::route_expert` | Same filter on the MoE path |
| `crates/larql-router/src/metrics.rs::RouterMetrics::route_saturation_total` | The new counter |
| `crates/larql-router/src/http.rs::handle_walk_ffn_inner` | 503 with `Retry-After: 0.5` when `route_all` returns `None` because of saturation |
| `crates/larql-router/src/main.rs` | `--saturation-ceiling N` flag |

### Test coverage

- Filter unit tests in `grid/routing.rs::tests`: empty candidate
  set, all-saturated → `None`, mixed sat/unsat → pick unsat.
- Integration: `walk_ffn_returns_503_with_retry_after_when_replicas_saturated`
  in `crates/larql-router/tests/test_http_handlers.rs` spawns a
  grid, sets `requests_in_flight >= ceiling` on every owning replica,
  fires a `/v1/walk-ffn` request, and asserts:
  - HTTP 503 (not 400 — `has_owners_for` distinguishes the two)
  - `Retry-After: 0.5` header is present
  - `larql_router_route_saturation_total` counter increments by 1
- Long-running chaos test
  (`crates/larql-router/tests/test_grid_chaos.rs`) hits `route()`,
  `has_owners_for()`, and the replication ledger across 5,000
  randomised register/deregister/heartbeat ticks (two variants, one
  with `target_replicas=2`) and asserts no panic, no stale ledger
  entries, and that the coverage floor follows the live server set.
