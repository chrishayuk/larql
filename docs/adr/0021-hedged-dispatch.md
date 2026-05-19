# ADR-0021 — Hedged Dispatch for `walk-ffn` Fan-Out

**Status:** Accepted — shipped 2026-05-16.

**Related:** ADR-0013 (three-tier routing comparator), ADR-0014
(hot-shard load-rate replication), ADR-0017 (Prometheus metrics),
ADR-0020 (saturation-tier backpressure).

---

## Context

Today's multi-shard fan-out in `handle_walk_ffn_inner` /
`handle_moe_dispatch` picks **one** owning replica per sub-request
via `GridState::route()` (the three-tier comparator from ADR-0013)
and forwards. When that replica is slow — long GC pause, hot CPU,
mid-eviction drain — the whole fan-out waits for it. p99 latency on
the router is shaped by the slowest replica's tail, not the typical
case.

`--target-replicas N` (N ≥ 2) means the topology already carries
spare capacity. We're paying for the second replica anyway; it's
sitting idle for most calls. The classic "hedged request" technique
(Dean & Barroso, *The Tail at Scale*, CACM 2013) trades a small
amount of wire/compute load for a tighter p99 — when the primary
hasn't responded after `M` milliseconds, fire the same sub-request
to a second replica and take whichever wins.

This is the legitimate **router-shaped** interpretation of
"speculative dispatch." The pre-ADR-0021 roadmap entry under
"Latency P1 — speculative next-layer prefetch" assumed the inference
client made one HTTP call per layer with residual threading; an
audit in this session showed the router actually sees one batched
call carrying all layers against a single input residual (each
sub-request is an independent feature lookup, fanned out in
parallel). So "prefetch layer N+1 while N is in flight" doesn't
apply at the router boundary — but **hedge the slow primary** does.

## Decision

Add an opt-in hedging tier to the multi-shard fan-out path:

```
                  /v1/walk-ffn arrives
                          ↓
              resolve_all → layer→primary URL
                          ↓
          group_layers_by_url → one sub-request per primary
                          ↓
       (ADR-0021) per sub-request: also resolve a SECONDARY
       replica via route_with_rank(..., max=2)[1]
                          ↓
    spawn(async {
        let primary = post(primary_url, body);
        select! {
            ok = primary => Ok(ok),
            _   = sleep(hedge_after_ms) => {
                // hedge fires: race primary vs secondary
                bump route_hedge_fires_total;
                select! {
                    p = primary => { return p; }
                    s = secondary => { bump route_hedge_wins_total; return s; }
                }
            }
        }
    })
                          ↓
              join_all → merge_shard_responses
```

Both `handle_walk_ffn_inner` (dense fan-out) and
`handle_moe_dispatch` (MoE expert fan-out) gain this branch. The
single-shard `proxy_raw` path is unchanged — hedging only helps when
there's a parallel sub-request to lose against.

### Configuration

```text
--hedge-after-ms <M>
  delay before the hedge fires, in milliseconds. None / 0 disables
  hedging entirely (pre-ADR-0021 behaviour). Typical M is 1–10 ms
  — long enough that the steady-state path doesn't double-dispatch,
  short enough that the hedge clips the long tail.
```

CLI default is `None` (disabled). Operators must explicitly opt in
because the trade-off (≈2× shard load on the tail) only makes sense
when (a) `target_replicas ≥ 2` so a real secondary exists, and (b)
you actually have an observed p99-latency problem worth spending
extra wire/compute on.

### Selecting the secondary replica

`GridState` grows a new method:

```rust
/// Up to `max` candidate URLs for `(model_id, layer)`, ordered by
/// the same three-tier comparator [`route`] uses, with the
/// saturation filter (ADR-0020) applied. Saturated replicas are
/// dropped, so the hedge never targets a replica we'd 503 on.
pub fn route_with_rank(
    &self,
    model_id: Option<&str>,
    layer: u32,
    max: usize,
) -> Vec<String> {
    ...
}
```

Plus a sibling `route_expert_with_rank` for the MoE path.

The fan-out picks `[0]` as the primary (today's `route()` result)
and `[1]` as the hedge candidate. If the layer only has one owning
replica, `[1]` is absent — no hedge fires; the sub-request waits on
the primary unconditionally. This degrades gracefully against the
`target_replicas = 1` deployment.

### Metric

ADR-0017 grows two new counters:

```text
larql_router_route_hedge_fires_total
  Number of hedged sub-requests where the primary's reply didn't
  arrive within --hedge-after-ms and the secondary was dispatched.

larql_router_route_hedge_wins_total
  Number of times the hedge actually beat the primary (the slow-
  primary case the hedge is designed for). hedge_wins / hedge_fires
  is the operator's signal: ratio ~1.0 means hedging is working;
  ratio ~0.0 means it's just doubling wire load.
```

Both are scalar `IntCounter` — bounded cardinality, consistent with
ADR-0017's "no per-replica labels" rule.

### Default: disabled

`hedge_after_ms = None` is the default. Identical to pre-ADR-0021
behaviour. Operators opt in only when they have a measured p99
problem and have provisioned `target_replicas ≥ 2`.

## Alternatives Considered

### Race both replicas from t=0

Always dispatch primary AND secondary in parallel; take the winner.
Halves the tail latency floor but **doubles** steady-state shard
load — a 2× cost that only pays off if every request is in the
tail. Rejected: the configurable `hedge_after_ms` knob lets
operators dial in the cost/benefit. Setting `hedge_after_ms = 0`
approximates this case if someone wants it.

### Backup replica only on detected hot primary

Send only to primary; if the primary's RTT-probe rises above
threshold, mark it cold and redirect future calls. Decision lives
in ADR-0014's hot-shard elevation. **Different mechanism, not a
replacement** — hot-shard reacts on tens-of-seconds timescales
(`rebalance-interval`), hedging reacts within milliseconds.

### Hedge driven from the inference side

Inference client tracks per-shard p99 itself and decides whether to
hedge. Pushes complexity into every client. Rejected: routing
decisions are the router's job; clients shouldn't need to know the
replica topology.

### Cancellation on hedge win

When the secondary wins, send an explicit `Cancel-Request`-style
message to the primary so it stops working. Saves wasted server-
side compute. Deferred — needs server-side support; today's
`walk-ffn` handler is stateless and there's no cancellation
protocol. Reqwest will close the primary connection when the task
is dropped, which is enough for short-lived requests; the cost is
the primary finishing its work locally and dropping the response on
the floor.

## Consequences

### Positive

- p99 latency drops in topologies with `target_replicas ≥ 2` and a
  slow tail — the typical production deployment after this session's
  hot-shard hysteresis work.
- Operator observable: `hedge_wins / hedge_fires` ratio. If the
  ratio is near 1, hedging is buying real tail clipping; if near 0,
  hedging is just costing wire load and the operator should reduce
  the value or disable it.
- Degrades gracefully: if no secondary exists (`target_replicas =
  1`), the dispatch reduces to the pre-ADR-0021 path. No request
  ever errors *because of* hedging.
- Independent of the saturation filter (ADR-0020) — the secondary
  must clear that filter before being elevated to a hedge target;
  hedging never sends to an over-ceiling replica.

### Negative

- **Wire / compute amplification**: when the hedge fires, both
  replicas do the work. With `target_replicas = 2` and a hedge that
  fires on, say, 10% of requests, shard load goes up by ~10%
  relative to no-hedge baseline. The
  `larql_router_route_hedge_fires_total` counter is the operator's
  knob: tune `--hedge-after-ms` upward if the fire rate is higher
  than expected.
- **Hedge interacts with the saturation filter**: if the secondary
  is already at the saturation ceiling, the hedge is silently
  skipped (no secondary). This is a feature (ADR-0020 is the
  load-shedding boundary) but worth documenting so operators know
  why hedging stopped clipping the tail in a saturated cluster.

### Neutral

- Single-shard requests (one owning shard for every requested
  layer) take the unchanged `proxy_raw` path — no hedging applies.
  Hedging only fires on multi-shard fan-out.

## Implementation pointers

| File | Role |
|---|---|
| `crates/larql-router/src/grid/routing.rs::route_with_rank` | New ranked-replica accessor (top-`max` URLs, saturation-filtered) |
| `crates/larql-router/src/grid/routing.rs::route_expert_with_rank` | MoE sibling — top-`max` URLs for `(layer, expert_id)` |
| `crates/larql-router/src/http.rs::AppState` | New `hedge_after: Option<Duration>` field |
| `crates/larql-router/src/dispatch.rs::hedged_post_json` | New helper: race primary against delayed secondary |
| `crates/larql-router/src/http.rs::handle_walk_ffn_inner` | Dense fan-out branches through `hedged_post_json` when `hedge_after.is_some()` |
| `crates/larql-router/src/http.rs::handle_moe_dispatch` | MoE fan-out same |
| `crates/larql-router/src/metrics.rs::RouterMetrics` | New counters `route_hedge_fires_total`, `route_hedge_wins_total` |
| `crates/larql-router/src/main.rs` | `--hedge-after-ms <M>` flag |

### Test coverage

- Grid unit tests on `route_with_rank` / `route_expert_with_rank`:
  empty grid → empty Vec; single replica → 1-elt Vec; multi-replica
  → ordered by comparator; saturation-filtered correctly.
- Integration: `walk_ffn_hedge_fires_when_primary_is_slow` —
  spawn two shards, primary deliberately sleeps past
  `--hedge-after-ms`, assert `route_hedge_fires_total = 1` AND
  `route_hedge_wins_total = 1` AND response body came from the
  secondary.
- Integration: `walk_ffn_hedge_does_not_fire_on_fast_primary` —
  same topology, primary responds instantly; assert hedge_fires = 0.
- Integration: `walk_ffn_no_hedge_when_only_one_replica` — single
  replica owns the range; hedge config set; assert dispatch
  succeeds AND hedge_fires = 0.
