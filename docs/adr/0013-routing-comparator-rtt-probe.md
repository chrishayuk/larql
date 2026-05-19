# ADR-0013 — Three-Tier Routing Comparator + Active-Probe RTT

**Status:** Accepted — shipped 2026-05-16 (GT3 layer-latency tier was
GT3 baseline 2026-05-07; active-probe RTT tier was P2 promoted to P1
and landed 2026-05-16).
**Depends on:** ADR-0004 (self-assembling grid), ADR-0011 (rebalancer)
**Implementation:** `crates/larql-router/src/grid/routing.rs::compare_servers_for_route`,
`crates/larql-router/src/tasks/rtt_probe.rs`

---

## Context

When `route(model_id, layer)` finds multiple replicas covering the
target layer, it must pick one. The naive pick — least
`requests_in_flight` — is correct but undiscriminating: a server that
is *slow* for layer 17 (cold cache, hot neighbour) but has fewer
queued requests will be picked over a faster peer. The router needs
to surface latency information into routing.

Three signals are available:

1. **GT3 per-layer latency** — `HeartbeatMsg.layer_stats` carries
   `(avg_ms, p99_ms)` per layer observed by the serving server.
   Direct measurement of what the request will pay. Available within
   one heartbeat interval (10 s) of the server processing requests
   for that layer.
2. **Active-probe RTT** — wall-clock round-trip from the router to
   the server's `/v1/health` endpoint. Captures wire latency
   (router → server) + the server's HTTP listener queueing, but
   not per-layer compute differences.
3. **Requests in flight** — counter from `HeartbeatMsg`. Always
   defined; useful as a load-shedding signal but says nothing about
   per-request latency.

A single-tier picker using any one signal alone has gaps:

- **GT3 only** — not populated until the server has processed the
  layer at least once. Cold-start replicas look free even when their
  weights aren't paged in.
- **RTT only** — silent on per-layer compute (LM head vs FFN-only
  shards have very different per-layer cost; RTT is identical).
- **In-flight only** — described above; defeats the purpose of GT3.

---

## Decision

`route()` uses a **three-tier cascade** with strict precedence:

```
1. GT3 per-layer latency       (HeartbeatMsg.layer_stats[layer].avg_ms)
   ↓ only when both replicas lack a GT3 entry for `layer`
2. Active-probe RTT             (ServerEntry.rtt_ms)
   ↓ only when neither side has been probed
3. Requests in flight           (HeartbeatMsg.requests_in_flight)
```

Each tier is also asymmetry-aware: when one replica has data for the
tier and the other doesn't, the one with data wins. So a freshly-
probed replica beats an unprobed one even if both lack GT3 stats.

```rust
fn compare_servers_for_route(a: &ServerEntry, b: &ServerEntry, layer: u32) -> Ordering {
    let lat_a = a.layer_latencies.get(&layer).map(|(avg, _)| *avg);
    let lat_b = b.layer_latencies.get(&layer).map(|(avg, _)| *avg);
    match (lat_a, lat_b) {
        (Some(la), Some(lb)) => la.partial_cmp(&lb).unwrap_or(Ordering::Equal),
        (Some(_), None) => Ordering::Less,        // a has GT3, b doesn't → a wins
        (None, Some(_)) => Ordering::Greater,     // mirror
        (None, None) => match (a.rtt_ms, b.rtt_ms) {
            (Some(ra), Some(rb)) => ra.partial_cmp(&rb).unwrap_or(Ordering::Equal),
            (Some(_), None) => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (None, None) => a.requests_in_flight.cmp(&b.requests_in_flight),
        },
    }
}
```

### NaN safety

`f32` latency comparisons use `partial_cmp` and fall back to `Equal`
on NaN. NaN sentinel values can sneak in from broken heartbeats; the
router never panics on them — it just treats them as equal and falls
through to the next tier via the surrounding `min_by`.

### Hoist out of `route()`

The cascade is a free function, not a closure inside `route()`. Two
reasons:

1. **Testability** — unit tests can drive the comparator with
   hand-built `ServerEntry`s and zero `GridState` setup. The
   `compare_*` test family in `routing.rs::tests` covers all four
   tier transitions plus NaN safety.
2. **Reusability** — future code paths (e.g. a "find slowest
   replica" path for the rebalancer) can reuse the comparator with a
   reversed `min_by` / `max_by`.

---

## Active-Probe Loop (Tier 2 data source)

### Why a separate task, not heartbeat piggyback

`HeartbeatMsg` is server → router only. The router never sends an
application-layer ack on the announce stream, so it cannot compute a
round-trip from the existing stream:

- TCP-level RTT would be available via socket-level options but
  doesn't capture the server's HTTP listener queueing.
- A "heartbeat ack" RouterMessage would require server-side timing
  state and a new proto field; it also wouldn't measure the actual
  production transport (`POST /v1/walk-ffn` over reqwest), only the
  gRPC control stream.

An **explicit HTTP probe** measures the same transport the production
traffic uses, so the recorded RTT reflects realistic queueing on the
server's HTTP listener, not just the TCP layer.

### Opt-in, not default

Probing is **disabled by default** (`--rtt-probe-interval-secs 0`).
Two reasons:

1. **GT3 subsumes RTT in steady state.** Once heartbeats carry
   `layer_stats`, the per-layer `avg_ms` includes both wire and
   compute. RTT is only useful for tier 2 — cold-start (no GT3 yet)
   or cross-region (different wire costs across replicas).
2. **Cost.** Probing every serving server every N seconds adds
   `n_servers / N` outbound requests/sec to the router. On a
   single-host dev deployment where RTT variance is below the GT3
   noise floor, this is pure overhead.

The CLI flag advertises this: `--rtt-probe-interval-secs <N>` with
`0` documented as the disable-sentinel.

### Probe-round shape

Each round (per ADR §implementation):

```
1. Read-lock the grid → snapshot (server_id, listen_url) for all serving servers.
2. Drop the read lock.
3. join_all() parallel reqwest GETs against `{listen_url}/v1/health`
   with a 2 s per-probe timeout.
4. Write-lock the grid once → batch-apply `update_rtt_ms(server_id, Option<f32>)`.
```

Snapshot-then-batch-write keeps the write window tight (single
acquire/release covering all servers) and lets the probes themselves
run lock-free in parallel. The 2 s timeout is independent of
`--timeout-secs` (which is for production traffic, much heavier than
a HEAD probe).

### Failure handling

A probe that returns non-2xx, times out, or hits a transport error
calls `update_rtt_ms(server_id, None)`. This **clears** the rtt_ms
field rather than reporting a stale value:

- Tier 2 of the comparator treats `None` as "unprobed" — falls
  through to tier 3.
- Stale RTT is worse than no RTT: a server that's slow now but was
  fast 30 s ago would still beat a fresh-but-uprobed peer.

The `update_rtt_ms` call is also tolerant of the server having left
the grid between snapshot and write — it's a `get_mut` no-op when the
server_id no longer exists.

---

## Alternatives Considered

### Single-tier picker (RTT only or in-flight only)

Rejected — see Context. RTT alone misses per-layer compute; in-flight
alone misses everything that matters.

### Combine all three with a weighted score

`score = α × layer_avg_ms + β × rtt_ms + γ × requests_in_flight`

Rejected because:
- Weight tuning is per-deployment (the right α/β/γ for a
  CPU-bottlenecked LAN cluster differ from a GPU-bottlenecked
  cross-region cluster).
- A score function loses the asymmetry rule (replica-with-data
  beats replica-without-data). Picking sensible defaults for "what
  does a missing GT3 score as?" is exactly the choice the cascade
  encodes for free.
- Strict precedence is easier to reason about and to unit-test.

A weighted score might still be a future ADR if real production data
shows the cascade picking the wrong server frequently. For now the
cascade is unambiguous.

### Piggyback RTT on heartbeat ack

Considered — would avoid the separate probe loop. Rejected because
(a) heartbeats are one-way, no return ack today; (b) measuring the
gRPC control stream RTT doesn't capture the HTTP listener queueing
that production traffic pays. The probe target (`/v1/health`) is the
same axum listener as `/v1/walk-ffn`, so the measurement reflects
real production transport.

---

## Consequences

### Positive

- `route()` is **constant-time in grid size** for the comparator
  loop (replicas-per-layer, not total servers). Bench measurements:
  103 ns (4 servers) → 120 ns (40 servers) for the production-shape
  bench at `target_replicas = 2`.
- Cold-start replicas don't get picked over warm ones (GT3 tier).
- Cross-region tie-breaking works without GT3 (RTT tier).
- Probe failures clear the value rather than poisoning it (stale RTT
  bad, missing RTT acceptable).

### Negative

- Probe loop is opt-in, so the default deployment doesn't benefit
  from tier 2 — but GT3 covers steady state, so cold-start is the
  only window where this matters. Operators of geographically-spread
  deployments need to know to flip the flag.
- Three tiers is more state to keep in mind when reading
  `route()`'s output during debugging. The free function name
  `compare_servers_for_route` and the doc-block on the cascade
  inside `grid/routing.rs` are the documentation.

---

## Implementation pointers

| File | Role |
|---|---|
| `crates/larql-router/src/grid/routing.rs::compare_servers_for_route` | The cascade itself, free function |
| `crates/larql-router/src/grid/routing.rs::route` | Uses `min_by` with the comparator over `route_table[layer]` entries |
| `crates/larql-router/src/grid/mod.rs::GridState::update_rtt_ms` | Mutator called by the probe loop |
| `crates/larql-router/src/tasks/rtt_probe.rs` | Probe task: config, spawn, probe_round, probe_one |
| `crates/larql-router/src/grid/status.rs::status_response` | f32 → u32 ms rounding for the wire (proto field width) |

### Test coverage

- `compare_*` test family in `grid/routing.rs::tests` — all four
  tier transitions + NaN safety, drives the comparator directly
  without standing up a `GridState`.
- `update_rtt_ms_*` tests in `grid/mod.rs::tests` — write-through,
  unknown-server no-op, status_response round-trip.
- `probe_*` tests in `tasks/rtt_probe.rs::tests` — config builder,
  unreachable-host timeout, empty-grid no-op, success path against a
  spawned axum fixture, non-2xx miss, probe_round write-back.

Per-file coverage as of 2026-05-16:

| File | Line coverage |
|---|---|
| `grid/routing.rs` | 97.84% |
| `tasks/rtt_probe.rs` | 94.86% |
