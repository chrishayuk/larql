# larql-router

Layer-sharding router for distributed `larql-server` deployments.

## What it does

Fans out `POST /v1/walk-ffn` calls across multiple `larql-server`
shards. Two sharding shapes are supported in a single router:

- **Dense (layer-pipeline)** — each shard owns a contiguous range
  of transformer layers (e.g. layers 0-14 on shard-a, 15-29 on
  shard-b). Requests carry `{layer: N}` or `{layers: [...]}` and
  the router resolves each layer to its owning shard.
- **MoE (per-expert-shard)** — each shard owns a contiguous range
  of *experts* within a layer range. Requests carry
  `{layer: N, experts: [E1, E2, ...]}` or
  `{layer_experts: [{layer, experts}, ...]}` and the router fans
  out per-token to the owning expert-shard. Designed for trillion-
  parameter MoE models (DeepSeek-V3/V4, Kimi K2 / K2.6) where a
  host typically loads one layer's slice of experts.

The router is intentionally narrow — it exposes only the endpoints
needed for the fan-out, not a full transparent reverse proxy:

- `POST /v1/walk-ffn` — single-layer or multi-layer fan-out across
  the shard map. Multi-layer requests are dispatched in parallel
  to each owning shard and the results merged.
- `GET /v1/health` — liveness + grid coverage summary.
- `GET /metrics` — Prometheus text-format scrape endpoint (ADR-0017).
  Gauges for grid state (server count, gaps, elevation), counters
  for rebalancer actions / RTT probe outcomes / walk-ffn requests,
  and a histogram for `walk-ffn` end-to-end latency. Bounded
  cardinality — no `model_id` / `server_id` / `layer_id` labels.

Other endpoints (`/v1/stats`, `/v1/walk`, `/v1/models`, etc.) live on
the individual shards — clients can call them directly on a shard's
HTTP port. The router exists to coordinate the fan-out, not to be
a full server.

## Two topologies

### Static `--shards` map

Router knows all shards' URLs at boot. Simplest ops; routes are
fixed for the router's lifetime.

```bash
larql-router \
    --shards 0-14=http://shard-a:9181,15-29=http://shard-b:9182 \
    --port 9090
```

### Self-assembling `--grid-port` + `--join`

Router exposes a gRPC port; shards register themselves with `--join
http://router:50052 --public-url http://shard:port`. The router
tracks coverage live and can accept / drop shards without a
restart.

```bash
# Router with HTTP on 9090 + grid gRPC on 50052
larql-router --grid-port 50052 --grid-key <secret> --port 9090

# Each shard joins (see larql-server docs for the full flag list)
larql-server <vindex> --port 9181 --layers 0-14 \
    --join http://router:50052 --grid-key <secret> \
    --public-url http://shard-a:9181
```

When a shard exits cleanly its announce stream closes; the router
logs `Grid: server left layers=N-M` and updates coverage. Requests
for now-uncovered layers return `HTTP 400 "layer N has no owning
shard in this router"` — clean error, not a hang. When the shard
restarts and re-joins, coverage automatically returns.

Both topologies serve the same HTTP API; clients don't need to know
which the operator picked.

## Self-assembling grid features

### Mode A vs Mode B

A `larql-server` joining the grid presents itself as either:

- **Mode A** (announced shard): the server has already loaded a
  specific layer range and sends `AnnounceMsg` to advertise it. This
  is the path used when the operator pins layer ownership via
  `--layers`.
- **Mode B** (available): the server has free disk + RAM but no shard
  loaded, sends `AvailableMsg`, and waits for the router to assign a
  layer range with `AssignMsg`. The server downloads the matching
  vindex shard via `GET /v1/shard/{model}/{start}-{end}` from a live
  origin, then sends `ReadyMsg` and transitions to Mode A.

Mode B lets a fresh server join a running grid and pick up coverage
without the operator pre-deciding which layers each box owns.

### Replication

`--target-replicas N` tells the router how many copies of each shard
range it should maintain. The rebalancer pulls spares from the
available pool when the count drops below `N`, and drops the
least-loaded replica when it climbs above. Combine with Mode B
servers as the spare pool.

### Dynamic rebalancing

Runs every `--rebalance-interval` seconds (default 30). Each tick:

1. Evicts servers whose heartbeat is older than 25 s (defensive
   against deadlocked TCP-but-no-progress connections).
2. Flips the elevated flag on shards exceeding the
   `--hot-shard-rps` threshold so their effective replica target is
   `target + 1`.
3. Pulls spares from the available pool for any under-replicated
   range.
4. Sends `UnassignMsg` to the least-loaded replica of any
   over-replicated range; the server drains in-flight requests for up
   to 30 s and re-enters Mode B if `--available-ram` was set.
5. Detects sustained per-layer latency imbalance
   (`--rebalance-threshold`, default 2× over a 60 s window) and
   evicts the slow replica.

Set `--rebalance-interval 0` to disable the background tick (you can
still drive moves manually via the admin RPCs).

End-to-end walkthrough: [`docs/hot-shard-demo.md`](./docs/hot-shard-demo.md)
(spins up a 2-serving-shard + 1-spare topology, drives load, and
prints the rebalancer's elevation/cool-down log lines). The
companion script is `scripts/demo-hot-shard.sh`.

For a real multi-host LAN deployment (router + two shards across
three boxes, with `--grid-key`, firewall rules, and an optional
QUIC variant), see [`docs/multi-host-demo.md`](./docs/multi-host-demo.md).

### Admin CLI

The same binary doubles as an admin client:

```bash
larql-router status                                   # full grid + servers JSON
larql-router gaps [--model M]                         # uncovered layer ranges
larql-router drain --server <ID> [--reason "..."]     # send UnassignMsg
larql-router assign --model M --layers A-B \
    [--server <ID>] [--origin-url URL] [--origin-hash H]
```

These call `GridService.DrainServer` / `AssignRange` over the
router's gRPC port. `assign` resolves an origin from any live replica
unless `--origin-url` is set, which is the escape hatch for filling
a range that no surviving server still covers (S3, mirror, etc.).

### QUIC transport (opt-in)

Build with `--features quic` and start the router with `--quic-port`
to listen for `quic://router:PORT` joins alongside the TCP listener.
Servers pin the router cert with `--quic-cert-fingerprint <SHA-256>`
(printed at router startup when the self-signed cert is generated).

QUIC carries HTTP/2 over a single bidirectional stream — the same
tonic-generated client/server code as the TCP path. Buys 0-RTT
reconnect, TLS 1.3, and BBRv2 congestion control. Real HTTP/3
(per-stream independence) is a future ADR.

## Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--shards <SPEC>` | Comma-separated `START-END=URL` (inclusive bounds). Optional when `--grid-port` is set. | — |
| `--grid-port <PORT>` | gRPC server port for self-assembling grid. Servers connect with `--join`. | — |
| `--grid-key <KEY>` | Shared secret enforced on `--join` registrations. Reads `LARQL_GRID_KEY` env. Without it, the grid port is open (development only). | — |
| `--port <PORT>` | HTTP listen port. | 9090 |
| `--host <HOST>` | Bind address. | 0.0.0.0 |
| `--timeout-secs <N>` | Per-request timeout to backend shards. | 120 |
| `--target-replicas <N>` | Phase 4 replication target per shard range. `>1` pulls spares from the available pool to maintain count. | 1 |
| `--rebalance-interval <SECS>` | Rebalancer tick cadence; `0` disables dynamic rebalancing. | 30 |
| `--rebalance-threshold <RATIO>` | Latency-imbalance threshold (slowest replica / fastest) before the rebalancer evicts. | 2.0 |
| `--hot-shard-rps <FRAC>` | Hot-shard load-rate replication: shards whose max `req_per_sec` across replicas exceeds this value are treated as effectively under-replicated until the rate subsides. | — (disabled) |
| `--hot-shard-demote-ratio <FRAC>` | ADR-0014 hysteresis: an elevated shard demotes only when its rate falls below `ratio × --hot-shard-rps`. `1.0` disables hysteresis. Values outside `(0.0, 1.0]` clamp to the default. | 0.8 |
| `--rtt-probe-interval-secs <N>` | Active-probe RTT cadence. When `>0`, the router periodically `GET`s `{listen_url}/v1/health` on every serving server and uses the recorded round-trip as a tie-breaker after GT3 per-layer latency in `route()`. | 0 (disabled) |
| `--saturation-ceiling <N>` | ADR-0020 backpressure tier. Replicas whose `requests_in_flight ≥ N` are filtered out of `route()` before the GT3/RTT/in-flight comparator runs. When every owning replica is saturated, the router 503s with `Retry-After: 0.5` and bumps `larql_router_route_saturation_total` instead of forwarding to the least-bad replica. | — (disabled) |
| `--hedge-after-ms <M>` | ADR-0021 hedged dispatch. On a multi-shard fan-out, if a sub-request's primary replica hasn't responded within `M` ms, dispatch the same sub-request to a secondary replica and take whoever wins. Halves p99 tail latency in topologies with `--target-replicas ≥ 2` at the cost of ~2× shard load on the tail. `larql_router_route_hedge_fires_total` counts fires; `larql_router_route_hedge_wins_total` counts secondaries that beat the primary. Single-shard `proxy_raw` requests are not hedged. | — (disabled) |
| `--log-level <LEVEL>` | Logging level. | info |

Run `larql-router --help` for the full set, including the QUIC
transport (`--quic-port` / `--quic-cert` / `--quic-key`) and admin
subcommands (`larql-router status / gaps / drain / assign`). See
[`ROADMAP.md`](./ROADMAP.md) for the per-feature shipping notes.

## Live perf snapshot (2026-05-16, M3 Max)

End-to-end:

| Path | tok/s |
|---|---|
| Gemma 3 4B local Metal | **86.1** |
| ollama gemma3:4b (same machine) | 98.7 |
| Gemma 4 26B-A4B, 2-shard grid (gRPC streaming + UDS + TCP_NODELAY) | 19.7 |

Per-call transport RTT (loopback): TCP HTTP ~660 µs, UDS HTTP ~510 µs,
gRPC streaming (multiplexed) ~460 µs.

gRPC routing hot path (in-process criterion benches; 2026-05-16, `--quick`):

**Production-shape: contiguous shards with `target_replicas=2-3`** (what `route()` actually sees in deployment — replicas-per-layer is a constant, not the total server count):

| Op | 4 srv (2×2) | 10 srv (5×2) | 20 srv (10×2) | 30 srv (10×3) | 40 srv (20×2) |
|---|---|---|---|---|---|
| `route()` single layer | 102 ns | 115 ns | 106 ns | 124 ns | 120 ns |
| `route_all()` 30 layers | 3.49 µs | 3.66 µs | 3.86 µs | — | — |
| `route_all()` 62 layers | — | — | 8.06 µs | — | 7.89 µs |
| `update_heartbeat()` | flat at ~270 ns regardless of grid size |||||
| single `register()` cost | 12 µs (1×30) | 59 µs (10×30) | — | — | 408 µs (100×30) |

So in real deployments, route lookups are **constant-time** at ~110 ns
across grid sizes — replicas-per-layer is the actual scaling axis,
not server count. A 32-layer decode picks shards in ~3.5 µs total.

**Worst case: every server replicates every layer** (stress test, not a realistic topology):

| Op | 1 server | 10 servers | 100 servers |
|---|---|---|---|
| `route()` single layer | 93 ns | 189 ns | 1.22 µs |
| `route_all()` 30 layers | 3.25 µs | 6.07 µs | 43.7 µs |
| `register()` (one rebuild) | 12 µs | 59 µs | 408 µs |
| `register_cascade` (build N from empty) | 9.6 µs | 325 µs | 21.5 ms |

The `register_cascade` row is N² in `n_servers` because it folds N
sequential registrations (each triggers one rebuild over a growing
set) into a single sample. The **single** `register()` row is the
per-join cost a real grid pays.

**ADR-0018 MoE expert routing** (`route_expert` / `route_all_experts`,
post-ADR-0018 benches):

| Op | Topology | Servers | Pairs | Time |
|---|---|---|---|---|
| `route_expert()` | Mixtral-style (1 layer × 4 expert-shards × 1 rep) | 4 | 1 | **121 ns** |
| `route_expert()` | V3-shape (60 layers × 4 shards × 1 rep) | 240 | 1 | **143 ns** |
| `route_expert()` | V3-shape with `target_replicas=2` | 480 | 1 | **196 ns** |
| `route_all_experts()` | Mixtral 32 layers × top-2 | 256 | 64 | **10.6 µs** |
| `route_all_experts()` | DeepSeek-V3 60 layers × top-6 | 240 | 360 | **61.7 µs** |
| `route_all_experts()` | DeepSeek-V3 60 layers × top-8 (aggressive) | 240 | 480 | **84.1 µs** |
| `route_all_experts()` | Kimi-K2-style 80 layers × top-8 | 320 | 640 | **112 µs** |

So **K2.6-scale per-token expert routing is ~112 µs of routing
work** against an inference compute budget orders of magnitude
larger. Routing is not the bottleneck even at trillion-parameter
MoE scale.

**ADR-0020 saturation filter overhead** (`route()` on a
`(10 shards × 2 replicas)` production-shape topology):

| Mode | Time |
|---|---|
| `ceiling=None` (baseline) | 113 ns |
| `ceiling=Some(16)`, no replica saturated | 108 ns |
| `ceiling=Some(4)`, every replica saturated → `None` | 57 ns |

The saturation filter costs nothing measurable on the happy path
(both modes are within noise of the baseline), and the all-saturated
short-circuit is actually faster than the comparator path because no
`min_by` runs. So `--saturation-ceiling N` is free to enable.

**Concurrent `route()` throughput** — N parallel tokio tasks against
a single `Arc<parking_lot::RwLock<GridState>>` (the same shape
`AppState::resolve_all` uses); topology is 10 shards × 2 replicas,
30 layers, 256 routes per worker. Numbers below are post-swap from
`tokio::sync::RwLock` → `parking_lot::RwLock` (2026-05-16):

| Workers | Throughput | vs 1-worker | vs pre-swap |
|---|---|---|---|
| 1 | 6.4 Melem/s | 1.0× | **+14%** |
| 4 | 11.1 Melem/s | 1.7× | **+28%** |
| 8 | 7.2 Melem/s | 1.1× | **+80%** |
| 16 | 6.1 Melem/s | 0.96× | **+70%** |

Peak throughput is at 4 readers; 8 and 16 stay above the 1-worker
baseline (the pre-swap 8-worker case had **collapsed below** 1
worker, 4.0 Melem/s vs 5.6 Melem/s — that pathology is gone). The
remaining sub-linear scaling past 4 readers is `parking_lot`'s
single-atomic read counter on a 12-core M3 Max, not a lock-cost
explosion. ArcSwap is a possible next step for grids with low write
rates; with ~1k heartbeats/sec on a 100-server grid today, the
copy-on-write cost outweighs the read win, so `parking_lot` is the
current sweet spot.

```bash
make bench-routing     # criterion sweeps; see crates/larql-router/benches/routing.rs
```

QUIC has not been benched against TCP yet on real workloads — `quic`
is opt-in and not in the default build.

## Validation

Grid routing + rebalancing are covered by focused unit + integration tests:

- inclusive layer-range routing, model-specific + default single-model tables
- least-loaded replica selection from heartbeat load
- per-layer latency-aware routing (GT3 `HeartbeatMsg.layer_stats`)
- Mode B `Available → Assign → Ready` + Phase B2 drain-then-reassign
- under/over-replication ticks with effective-target bookkeeping
- hot-shard `req_per_sec` detection + elevation/demotion
- stale-heartbeat eviction
- gap-fill on `DroppingMsg` / disconnect
- admin RPCs (`status` / `gaps` / `drain` / `assign`)
- ADR-0018 MoE expert routing — `route_expert` / `route_all_experts`,
  per-(layer, expert-range) replication keys, hot-shard elevation
  per expert-shard, JSON `experts` / `layer_experts` HTTP shapes,
  fan-out merge, dense regression (all 184 pre-MoE tests pass
  unchanged)
- ADR-0020 backpressure tier — saturation filter in `route()` /
  `route_expert()`, dispatcher 503 vs 400 disambiguation via
  `has_owners_for`, `Retry-After: 0.5` header, and counter increment
  (`walk_ffn_returns_503_with_retry_after_when_replicas_saturated`)
- ADR-0021 hedged dispatch — `route_with_rank` /
  `route_expert_with_rank` accessors, `hedged_post_json` helper,
  `--hedge-after-ms` opt-in, dense + MoE fan-out wiring,
  `route_hedge_fires_total` / `route_hedge_wins_total` counters.
  Three integration tests:
  `walk_ffn_hedge_fires_when_primary_is_slow` (asserts both counters
  +1 and secondary actually served the sub-request),
  `walk_ffn_hedge_does_not_fire_on_fast_primary` (counters stay 0,
  secondary untouched), `walk_ffn_no_hedge_when_only_one_replica`
  (single-owner topology — hedge configured but never fires).
- Long-running chaos test (`tests/test_grid_chaos.rs`) — 5,000
  randomised register/deregister/heartbeat/route ticks per variant
  (one with `target_replicas=1`, one with `=2`); asserts ledger
  consistency, coverage floor, and no `route()` panic on every tick

```bash
cargo test -p larql-router                    # 169 lib + 50 integration = 219 tests
                                               # (incl. MoE expert-routing, hot-shard hysteresis,
                                               #  ADR-0020 saturation backpressure, ADR-0021
                                               #  hedged dispatch, grid chaos, dense regression)
cargo test -p larql-router --features http3    # 220 tests (+1 h3 fan-out integration)
cargo test -p larql-router-protocol --features quic
                                               # 18 tests (15 unit + 3 QUIC integration)
make larql-router-coverage-summary             # see Coverage section for current numbers
make larql-router-protocol-coverage-summary    # 91.36% total, 1/1 files ≥90%
```

Test counts as of 2026-05-16.

## Source layout

```
src/
├── lib.rs                       # module declarations + public surface
├── main.rs                      # CLI entry, admin dispatch, server startup
├── http.rs                      # axum handlers (/v1/walk-ffn, /v1/health, /v1/stats, /metrics)
├── metrics.rs                   # ADR-0017 Prometheus registry + RouterMetrics struct
├── dispatch.rs                  # multi-layer fan-out + response merge
├── shards.rs                    # static `--shards` parser + binary peek
├── admin.rs                     # admin client + status/gaps formatters
├── cli_helpers.rs               # build_shard_client + small helpers
├── grid/                        # self-assembling grid state + gRPC service
│   ├── mod.rs                   # ServerEntry, AvailableEntry, GridState core
│   ├── routing.rs               # route() / route_all() + 3-tier comparator
│   ├── replication.rs           # under/over-rep, gap-fill, AssignMsg dispatch
│   ├── hot_shard.rs             # req/sec saturation + elevation set
│   ├── status.rs                # coverage_gaps, all_shard_urls, status_response
│   ├── service.rs               # gRPC GridService impl + admin RPCs
│   └── testing.rs               # #[cfg(test)] shared `entry()` helper
└── tasks/                       # long-lived background tasks
    ├── mod.rs                   # module declarations
    ├── rebalancer/              # 30 s rebalance tick
    │   ├── mod.rs               # spawn + tick loop
    │   ├── config.rs            # RebalancerConfig
    │   ├── hot_shard.rs         # elevation set updates
    │   ├── replication.rs       # under/over-rep ticks
    │   ├── eviction.rs          # stale-heartbeat eviction
    │   └── imbalance.rs         # per-layer latency tracker
    └── rtt_probe.rs             # opt-in active RTT probe loop
```

The grid + rebalancer folders were split out of two monolithic files
on 2026-05-16. All cross-file access goes through the public
`GridState` methods — no private fields are reached around through
`pub(super)`-style escape hatches; submodules see parent fields by
Rust's normal child-module visibility rule. The shared
`crate::grid::testing::entry()` helper is `pub(crate)` and used by
test modules in both `grid::*` and `tasks::rebalancer::*` to keep the
struct literal in one place.

## Examples

Runnable demos under [`examples/`](./examples/):

| Example | What it shows |
|---|---|
| `embed_grid` | Build a `GridState` programmatically: register servers, query routes, inspect coverage gaps, exercise hot-shard elevation. |
| `static_shards_server` | Minimal HTTP router with a hard-coded shard map — `parse_shards` + `build_router` + `axum::serve`. The smallest possible deployment shape. |
| `admin_client` | Calls `admin_status` / `admin_drain` / `admin_assign` from Rust against a running router — the same code paths the CLI uses, but reusable from your own ops tooling. |
| `fanout_dispatch` | The dispatch building blocks (`resolve_static_only`, `group_layers_by_url`, `build_subrequest_body`, `merge_shard_responses`) on a synthetic multi-layer request — no network. |
| `saturation_backpressure` | ADR-0020 in isolation: drives a `GridState` through ceiling=None, ceiling-with-headroom, one-saturated, all-saturated, no-owner scenarios so you can see when `route()` flips to `None` and which HTTP status the dispatcher will emit (200 / 400 / 503). |

```bash
cargo run -p larql-router --example embed_grid
cargo run -p larql-router --example fanout_dispatch
cargo run -p larql-router --example saturation_backpressure
cargo run -p larql-router --example static_shards_server   # listens on :9090
cargo run -p larql-router --example admin_client           # needs a router on :50052
```

## See also

- `crates/larql-router-protocol/README.md` — gRPC schema + the
  QUIC transport wrapper that backs `--quic-port`.
- `crates/larql-server/README.md` — shard configuration, recommended
  setups, the `--join` / `--public-url` / `--grid-key` flags.
- `crates/larql-server/docs/router-spec.md` — protocol-level spec
  for the gRPC schema, endpoint contracts, and binary wire format.
- [`ROADMAP.md`](./ROADMAP.md) — per-feature shipping notes and
  what's still on P1 / P2.
