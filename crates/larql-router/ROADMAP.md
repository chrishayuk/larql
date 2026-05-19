# Roadmap — larql-router / larql-router-protocol

---

## Current state (2026-05-16)

Self-assembling grid is feature-complete across ADR-0004 Phase 1–5, ADR-0010
(QUIC), ADR-0011 (Mode B + Phase B2 drain-then-reassign + replication),
ADR-0012 Phase 2 (criterion micro-benchmarks, both worst-case + production-shape),
ADR-0013 (three-tier routing comparator + active-probe RTT),
ADR-0014 (hot-shard load-rate replication + two-threshold hysteresis amendment),
ADR-0015 (ShardService.Query KNN endpoint),
ADR-0016 (router module organization),
ADR-0017 (Prometheus `/metrics` endpoint, bounded-cardinality),
ADR-0018 (MoE expert routing — `route_expert` / `route_all_experts`,
per-(layer, expert-range) replication, JSON `experts` / `layer_experts`
HTTP shapes),
ADR-0019 (HTTP/3 shard transport, opt-in via `--http3-shards` and
`--http3-port`),
ADR-0020 (saturation-tier backpressure in `route()` —
`--saturation-ceiling N`, 503 with `Retry-After: 0.5`, distinguished
from 400 via `has_owners_for`, `larql_router_route_saturation_total`
counter), and
ADR-0021 (hedged dispatch — opt-in via `--hedge-after-ms M`,
`route_with_rank` / `route_expert_with_rank` accessors, races a
secondary replica against a slow primary when M ms elapses;
`route_hedge_fires_total` / `route_hedge_wins_total` counters).
Static `--shards` (ADR-0003) remains as a fallback and coexists with
the grid.

The codebase is architecture-agnostic: routing logic reads layer ranges,
`model_id`, and server state from the grid protocol — no model-family
constants are hardcoded.

### What works today

- **Mode A** — `AnnounceMsg` → `AckMsg` registration + heartbeat loop + reconnect.
- **Mode B (Phase B1 + B2)** — `AvailableMsg` → `AssignMsg` → `ReadyMsg`; servers
  re-enter the available pool after an `UnassignMsg`-driven drain on the same
  stream.
- **Replication** — `--target-replicas N`; under-replicated ranges pull spares
  from the available pool, over-replicated ranges drop the least-loaded
  replica via `UnassignMsg`. Origin URLs resolved from any live replica via
  `find_origin_for`.
- **Hot-shard load-rate replication** — `--hot-shard-rps THRESHOLD`; when a
  shard's max `HeartbeatMsg.req_per_sec` across replicas exceeds the
  threshold, the rebalancer treats it as effectively under-replicated
  (`target + 1`) and pulls a spare. The elevated flag clears when the rate
  drops; over-replication then prunes the surplus on the next tick.
- **Stale heartbeat eviction** — rebalancer evicts serving servers whose
  `last_seen` exceeds `stale_heartbeat_timeout` (default 25 s = 2.5 ×
  heartbeat interval).
- **Per-layer latency-aware routing (GT3)** — `route()` prefers the server
  with lowest `layer_latencies[layer].avg_ms`; falls back through
  active-probe RTT, then `requests_in_flight`.
- **Active-probe RTT routing** — opt-in via `--rtt-probe-interval-secs`.
  The probe loop `GET`s `{listen_url}/v1/health` against every serving
  server on the configured cadence and lands the round-trip on
  `ServerEntry.rtt_ms`. Used by `route()` as the middle tier of the
  three-tier comparator.
- **`GridService.Join`** bidirectional gRPC stream over TCP (default) or
  QUIC (`--features quic`).
- **QUIC transport (GT7)** — `--quic-port`, `--quic-cert`/`--quic-key` (or
  auto-generated self-signed cert), SHA-256 fingerprint pinning on the
  client side via `--quic-cert-fingerprint`. HTTP/2 carried over a single
  QUIC bi-stream; 0-RTT reconnect + TLS 1.3.
- **Admin CLI (Phase 5)** — `larql-router status` / `gaps` / `drain --server`
  / `assign --model M --layers A-B [--server S] [--origin-url URL]`. Backed
  by new `DrainServer` + `AssignRange` gRPC RPCs.
- `DroppingMsg` → deregistration + auto gap re-fill + auto re-replication.
- Static `--shards` mode with layer-range routing and per-shard parallel
  fan-out.
- Grid + static fallback via `AppState::resolve_all()`.
- `GET /grid-status` (served by `StatusResponse` with `layer_stats` per
  server).
- Auth: optional shared `--grid-key` Bearer token in gRPC metadata.
- Library crate (`larql_router::{grid, tasks, dispatch, shards, http,
  admin, cli_helpers}`) for tests and external consumers. `tasks` rolls
  up `rebalancer` (6 sub-files) and `rtt_probe`; `grid` rolls up
  `mod`, `routing`, `replication`, `hot_shard`, `status`, `service`,
  and a `#[cfg(test)] testing` helper.
- Examples — `examples/embed_grid.rs`, `examples/fanout_dispatch.rs`,
  `examples/static_shards_server.rs`, `examples/admin_client.rs`,
  `examples/saturation_backpressure.rs` (ADR-0020 — drives a
  `GridState` through five saturation/coverage scenarios and prints
  the routing-layer decision plus the HTTP status the dispatcher
  would emit).
- Criterion benchmarks (GT9 ✅) — `routing.rs` with ten groups:
  `route_single_layer` + `route_all` (worst-case full replication),
  `route_realistic` + `route_all_realistic` (production-shape
  contiguous shards × `target_replicas`), `route_expert_single` +
  `route_all_experts` (ADR-0018 MoE), `heartbeat_update`,
  `single_register` (per-join rebuild cost), `register_cascade` (N
  sequential joins — O(N²) cold-start measurement), and
  `saturation_filter` (ADR-0020 — no-filter / filter-all-unsat /
  filter-all-sat across 5, 10, 20 shards × 2 replicas).

### What is not yet implemented

- **Cross-router federation** — multi-region routing (P2).
  (MoE within-layer expert sharding — previously listed here as P2
  — shipped 2026-05-16 as ADR-0018; see the "Shipped" line above
  for `route_expert` / `route_all_experts` and the self-healing
  section below for per-(layer, expert-range) replication.)

---

## Live perf snapshot (2026-05-16, M3 Max)

| Path | tok/s |
|---|---|
| Gemma 3 4B local Metal (today's code) | **86.1** |
| ollama gemma3:4b (same machine) | 98.7 |
| Gemma 4 26B-A4B, 2-shard grid, gRPC streaming + UDS + TCP_NODELAY | 19.7 |

Per-call transport RTT (loopback):

- TCP HTTP: ~660 µs
- UDS HTTP: ~510 µs
- gRPC streaming (multiplexed): ~460 µs

gRPC routing hot path (in-process, criterion `--quick`; rerun 2026-05-16, M3 Max).

**Production-shape — contiguous shards with `target_replicas`** (what
`route()` actually sees: replicas-per-layer is a small constant, not
the total grid size):

| Topology | servers | replicas/layer | `route()` | `route_all(30)` | `route_all(62)` |
|---|---|---|---|---|---|
| 2 shards × 2 | 4 | 2 | 102 ns | 3.49 µs | — |
| 5 shards × 2 | 10 | 2 | 115 ns | 3.66 µs | — |
| 10 shards × 2 | 20 | 2 | 106 ns | 3.86 µs | 8.06 µs |
| 10 shards × 3 | 30 | 3 | 124 ns | — | — |
| 20 shards × 2 | 40 | 2 | 120 ns | — | 7.89 µs |

`route()` is **essentially flat (~110 ns)** across grid sizes — only
`target_replicas` drives the cost. A full 62-layer forward pass picks
shards in ~8 µs total, which is 0.06% of a 13.78 ms decode.

**Worst case — every server replicates every layer** (stress test,
not a production topology):

| Op | 1 server | 10 servers | 100 servers |
|---|---|---|---|
| `route()` single layer | 93 ns | 189 ns | 1.22 µs |
| `route_all()` 30 layers | 3.25 µs | 6.07 µs | 43.7 µs |
| `update_heartbeat()` | 270 ns | 294 ns | 271 ns |
| **single** `register()` 30 layers | 12.3 µs | 59 µs | 408 µs |
| **single** `register()` 62 layers | 24.5 µs | 121 µs | 810 µs |
| `register_cascade` 30 layers | 9.6 µs | 325 µs | 21.5 ms |
| `register_cascade` 62 layers | 18.7 µs | 649 µs | 44.0 ms |

`register_cascade` measures N sequential joins folded into one
sample, so its scaling is `O(N² × L)` — useful as a cold-start
ceiling but not the per-join cost. The `single_register` rows are
the realistic per-join cost a live grid pays. At 810 µs for a 100/62
grid, register cost is negligible against the 30 s rebalance interval.

QUIC has not been benched against TCP yet on real workloads — `quic` is
opt-in and not in the default-build path.

---

## Coverage

```bash
make larql-router-coverage-summary
make larql-router-protocol-coverage-summary
```

Both crates pass policy (2026-05-16):

| Crate | Total | Files at 90% default | Debt baselines |
|---|---|---|---|
| `larql-router` | 93.17% | 19 of 20 | 1 (`grid/service.rs` 88%) |
| `larql-router-protocol` | 91.36% | 1 of 1 | 0 |

Router per-file (2026-05-16, post ADR-0018 MoE expert routing):

| File | Lines |
|---|---|
| `dispatch.rs` | 100.00% |
| `shards.rs` | 100.00% |
| `grid/hot_shard.rs` | 100.00% |
| `grid/status.rs` | 100.00% |
| `grid/testing.rs` | 100.00% |
| `tasks/rebalancer/config.rs` | 100.00% |
| `admin.rs` | 99.64% |
| `metrics.rs` | 99.63% |
| `grid/routing.rs` | 98.27% |
| `cli_helpers.rs` | 98.53% |
| `tasks/rebalancer/replication.rs` | 97.98% |
| `grid/replication.rs` | 96.46% |
| `grid/mod.rs` | ~97% |
| `tasks/rebalancer/eviction.rs` | ~93% |
| `tasks/rebalancer/mod.rs` | ~95% |
| `http.rs` | 93.61% |
| `tasks/rtt_probe.rs` | 94.86% |
| `tasks/rebalancer/imbalance.rs` | 94.83% |
| `tasks/rebalancer/hot_shard.rs` | 92.00% |
| `grid/service.rs` | 89.87% (debt — gRPC streaming join handler, baseline 88%) |
| `main.rs` | (excluded — binary entry point) |

Two file-system reorganizations landed on 2026-05-16:

1. **`grid.rs` (2113 lines) → `grid/` folder** with one file per
   concern: `mod.rs` (state core), `routing.rs`, `replication.rs`,
   `hot_shard.rs`, `status.rs`, `service.rs` (gRPC impl), and a
   `#[cfg(test)] testing.rs` helper used across the test modules.
2. **`rebalancer.rs` (861 lines) → `tasks/rebalancer/` folder** with
   `mod.rs` (spawn + tick loop), `config.rs`, `hot_shard.rs`,
   `replication.rs`, `eviction.rs`, `imbalance.rs`. The folder lives
   under `tasks/` alongside `rtt_probe.rs`, signalling both as
   long-lived background tasks spawned at router startup.

`grid/service.rs` houses the spawned-task body of the gRPC `join`
stream — once isolated from the 2113-line monolith, the
harder-to-unit-test branches drop it to 88.59%. Four new integration
tests in `tests/test_grid_service.rs`
(`available_with_under_replication_triggers_replicate`,
`serving_disconnect_triggers_post_stream_replicate`,
`payload_none_is_silently_skipped`,
`dropping_under_replicated_shard_triggers_replicate_log`) plus a
`tasks::rebalancer::spawn_runs_the_task_loop_through_one_tick` unit
test lifted post-split totals to 92.81%. The remaining ~11% gap on
`grid/service.rs` is mainly unreachable Mode B gap-fill code
(within-grid origin contradicts the gap definition; only the admin
RPC's `explicit_origin_url` path can exercise it) and tx-send-failure
races.

Router-protocol: `src/transport/quic.rs` at 91.36% (the only
instrumented source — proto re-exports filtered out by
`cargo-llvm-cov` since they live in `target/`).

Vindex coverage (for grid-relevant context — gate_knn lives there):

| Crate | Total | Path used by gate_knn |
|---|---|---|
| `larql-vindex` | 90.86% | `patch/overlay.rs` 88.61% (debt baseline 82%) |

---

## Shipped (P1)

### GT3 — Per-layer latency in HeartbeatMsg ✅ shipped 2026-05-07

**Spec**: ADR-0011 §HeartbeatMsg Extension.

**What shipped:**
- `grid.proto`: `LayerLatency { layer, avg_ms, p99_ms }` message;
  `HeartbeatMsg.layer_stats = 4`; `ServerInfo.layer_stats = 11`.
- `ServerEntry.layer_latencies: HashMap<u32, (f32, f32)>`.
- `update_heartbeat()` accepts `Vec<LayerLatency>` and stores them.
- `route()` prefers server with lowest `layer_latencies[layer].avg_ms` when
  data exists; falls back to `requests_in_flight`.
- `status_response()` populates `ServerInfo.layer_stats` sorted by layer.

---

### GT5 — Mode B: gap-fill assignment ✅ shipped 2026-05-13

**Spec**: ADR-0011 §Phase B1 Protocol.

**What shipped:**
- `GridState` carries `available_servers`, `serving_senders`.
- `GridState::find_origin_for(model_id, start..=end) -> Option<(url, hash)>` —
  picks any currently-serving replica covering the range as origin.
- `GridState::try_assign_gap(...)` resolves origin automatically;
  `try_assign_gap_with_origin(...)` retained for external origins.
- `GridState::try_fill_all_gaps()` scans `coverage_gaps()` and fills each
  from the available pool.
- Gap re-fill auto-fires on `DroppingMsg` and stream-close paths.
- Server side: `larql-server` exposes `GET /v1/shard/{model_id}/{start}-{end}`
  as a tar stream so the spare can mirror the donor's vindex; matching tar
  unpack in `shard_loader.rs`.
- Server announce client transitions from Mode A to Mode B on the same
  gRPC stream after drain (`available_after_drain` config).
- Integration tests: `crates/larql-server/tests/test_grid_mode_b.rs` (full
  vertical handoff + negative path) and `test_grid_drain_reassign.rs`
  (Phase B2 cycle).

---

### GT6 — Dynamic rebalancing ✅ shipped 2026-05-13

**Spec**: ADR-0011 §Phase B2 Protocol.

**What shipped:**
- `rebalancer::check_imbalance` — sustained imbalance trigger
  (`max/min > threshold` over `sustained_window`).
- `rebalancer::check_under_replication` + `check_over_replication` — Phase 4
  replica-count enforcement (sends `UnassignMsg` to least-loaded victim when
  over-replicated; pulls from available pool when under-replicated).
- `rebalancer::evict_stale_heartbeats` — defensive eviction of servers that
  stop heartbeating without closing the stream.
- New `GridState::send_assign_to_named_available()` for the admin
  `assign --server <id>` path.

---

### GT7 — QUIC transport ✅ shipped 2026-05-15

**Spec**: ADR-0010 (full spec).

**What shipped (feature-gated under `quic`):**
- `crates/larql-router-protocol/src/transport/quic.rs`:
  - `QuicStream` — wraps `(SendStream, RecvStream)` as `AsyncRead+Write` + `tonic::transport::server::Connected`.
  - `self_signed_tls(server_name)` — rcgen-based dev cert with SHA-256 fingerprint.
  - `server_endpoint(addr, tls)` / `client_endpoint(bind, expected_fingerprint)`.
  - `FingerprintVerifier` — pins server cert by SHA-256 (no CA chain).
  - `spawn_accept_loop(endpoint)` — accepts QUIC conns + bi-streams, feeds tonic `serve_with_incoming`.
  - `connect_grpc_channel(endpoint, addr, server_name)` — full client wiring.
- Router: `--quic-port`, `--quic-cert`, `--quic-key`, `--quic-server-name`.
  Parallel QUIC listener alongside the TCP gRPC server.
- Server: `--quic-cert-fingerprint`. `announce::try_once` branches on
  `quic://` scheme via `connect_grid_channel`.
- Round-trip integration tests: announce → ack streaming + unary `Status`
  over QUIC (`crates/larql-router-protocol/tests/test_quic_roundtrip.rs`).

**Limitation:** This is QUIC-as-TCP-replacement (HTTP/2 over a single QUIC
bi-stream), not HTTP/3. Buys 0-RTT reconnect + TLS 1.3 + BBRv2 congestion
control; per-stream-independence is moot for `Join` (single bidi stream
per server). **Real HTTP/3 for the shard-fan-out path shipped under
ADR-0019** (2026-05-16, `--http3-shards` / `--http3-port`, h3 0.0.8 +
h3-quinn 0.0.10 + h3-axum 0.2). Router-protocol h3 transport ships
`H3Client::post_json` + `serve_axum`; the MoE expert fan-out path uses
it when `h3_client: Some(_)` is wired into `AppState`. See the
"Shipped" line in the self-healing-grid section below.

---

### GT9 — Criterion routing benchmarks ✅ shipped 2026-05-07

**Spec**: ADR-0012 §Layer 2.

**What shipped:**
- `crates/larql-router/benches/routing.rs`: `bench_route_single_layer`,
  `bench_route_all`, `bench_heartbeat_update`, `bench_rebuild_route_table`
  at 1/10/100 servers × 30/62 layers.
- `src/lib.rs` exposes `pub mod grid` for bench linking.
- Makefile: `make bench-routing` / `make bench-all`.

---

### Phase 5 — Admin CLI ✅ shipped 2026-05-15

**Spec**: ADR-0004 §"Admin API".

**What shipped:**
- New proto RPCs: `DrainServer(DrainRequest) -> AdminAck`,
  `AssignRange(AssignRangeRequest) -> AdminAck`.
- Server-side: `GridServiceImpl::drain_server`,
  `GridServiceImpl::assign_range` (resolves origin from live replica or
  accepts `explicit_origin_url`).
- CLI subcommands: `larql-router status` / `gaps [--model M]` /
  `drain --server ID [--reason R]` / `assign --model M --layers A-B [--server S] [--origin-url URL] [--origin-hash H]`.
- Pure helpers in `larql_router::admin`: `format_status`, `format_gaps`,
  `parse_layers`, plus RPC wrappers `admin_status`, `admin_gaps`,
  `admin_drain`, `admin_assign`.
- Integration tests in `crates/larql-router/tests/test_admin_rpcs.rs`.

---

### Hot-shard load-rate replication ✅ shipped 2026-05-15

**Spec**: ROADMAP P1 sketch (this file).

`target_replicas` enforces a *count*; this adds *rate-aware* replication.
A shard whose per-replica `req_per_sec` exceeds the configured threshold
is treated as under-replicated even at `replicas == target_replicas`,
prompting the rebalancer to pull one extra spare. When the rate subsides
the elevation is cleared and the existing over-replication tick drops
the surplus on the next pass.

**What shipped:**
- `grid.proto`: `HeartbeatMsg.req_per_sec = 5` (shard-scoped rate).
- Server: `LoadedModel.requests_total: Arc<AtomicU64>` bumped by
  `walk_ffn`. Heartbeat sender diffs against the last sample and divides
  by `HEARTBEAT_INTERVAL` to populate `req_per_sec`.
- `GridState`:
  - `ServerEntry.req_per_sec` updated by `update_heartbeat`.
  - `elevated_ranges: HashSet<(model_id, start, end)>`.
  - `hot_layer_ranges(threshold) -> Vec<...>` (max-rate-across-replicas).
  - `mark_elevated` / `demote_elevated` / `elevated_ranges_snapshot`.
  - `effective_target_for(model, start, end)` =
    `target_replicas + (1 if elevated else 0)`.
  - `under_replicated_ranges` / `over_replicated_ranges` consult the
    effective target instead of the raw `target_replicas`.
- `rebalancer::check_hot_shards`: marks newly hot ranges as elevated,
  demotes ranges whose rate has dropped below the threshold. Runs before
  under/over-replication so flips land in the same tick.
- `RebalancerConfig::hot_shard_rps_threshold: Option<f32>` with
  `with_hot_shard_threshold` builder.
- CLI: `--hot-shard-rps <f32>` flag on `larql-router`. Unset = disabled.

Validation path remains the same as before: with `--target-replicas 1
--hot-shard-rps 50` and the `--concurrent N` bench harness, a hot shard
pulls a spare to effectively become `target+1`, then drops back once
the bench finishes.

---

### Stale heartbeat eviction ✅ shipped 2026-05-15

**Spec**: ADR-0004 Phase 3 §"Stale heartbeat eviction".

**What shipped:**
- `GridState::stale_server_ids(timeout)` — pure helper, walks `last_seen`.
- `rebalancer::evict_stale_heartbeats` — async wrapper, deregisters + triggers gap-fill.
- `RebalancerConfig::stale_heartbeat_timeout` (default 25 s).

---

### RTT-based routing ✅ shipped 2026-05-16

**Spec**: ROADMAP P2 sketch (was P2; promoted + shipped same day).

`ServerInfo.rtt_ms` was defined in the proto since GT3 but never
populated. Now it gets a value from an active-probe loop and is used
as a tie-breaker in `route()` when no GT3 per-layer latency data is
available yet.

**What shipped:**
- `ServerEntry.rtt_ms: Option<f32>` — `None` until probed, written
  by `GridState::update_rtt_ms`. `status_response` rounds to `u32`
  ms for the wire (proto field width).
- `route()` cascade extended to three tiers: GT3 per-layer
  `avg_ms` → `rtt_ms` → `requests_in_flight`. Comparator lifted to
  free fn `compare_servers_for_route` so the order is unit-testable
  without a full `GridState`.
- New `larql-router/src/rtt_probe.rs`:
  `RttProbeConfig::from_cli(interval_secs)`, `spawn` that owns the
  task lifetime, `probe_round` (snapshot serving list → parallel
  `GET {listen_url}/v1/health` via `reqwest` → batch write). 2 s
  per-probe timeout; failures clear `rtt_ms` rather than reporting
  stale data.
- CLI: `--rtt-probe-interval-secs <N>` on `larql-router`, default 0
  (disabled). Opt-in because GT3 already subsumes RTT in steady
  state; probe mainly helps cold-start and cross-region tie-breaks.
- 11 new tests: 7 on the comparator + status round-trip, 4 on
  `probe_one`/`probe_round` (including a tiny axum server fixture
  for the 2xx success path and the non-2xx miss path).

Test counts: **127 router lib tests** (was 116); `rtt_probe.rs`
coverage 94.86% lines.

---

### Exp 53 — Rust port of the sharded-vindex shard endpoint ✅ shipped 2026-05-16

**Spec**: `experiments/53_sharded_vindex/{README.md, server.py:67-103}`.

Ported the Python prototype's KNN shard service into Rust. The handler
mirrors `server.py:knn_lookup` exactly (cosine similarity, tau gate, k=1
fast path, positive-cosine-weighted top-k average); the wire moves from
the prototype's bespoke binary TCP frame to tonic/gRPC so shard traffic
shares the same channel as `GridService.Join` when `--features quic`
is enabled.

**What shipped:**
- `larql-router-protocol/proto/shard.proto` — `ShardService.Query`
  unary RPC. `ShardQuery { layer_id, k, query_vec, tau_override }` →
  `ShardResult { hit, mlp_out, best_sim }`. `query_vec` / `mlp_out`
  use raw f32 LE bytes (same wire convention as `ExpertService`)
  so hidden-sized arrays don't pay proto varint overhead.
- `larql-server/src/shard_query.rs` — pure helpers (`l2_normalize`,
  `cosine_similarities`, `weighted_topk_average`, `decode_f32_le`,
  `encode_f32_le`) + a `ShardSource` enum with two backends:
    - `ShardSource::Vindex` — production. Queries the server's
      loaded `PatchedVindex` via `gate_knn` + `ffn_row_into`
      (component = down). "Compiled facts" live as vindex patches
      (`insert_feature` + `set_down_vector`); no separate on-disk
      cache format is needed.
    - `ShardSource::Cache` — test fixture. Tiny in-memory
      `HashMap<u32, LayerEntry>` with `insert_layer` +
      `seed_from_normed`; lets unit + integration tests cover the
      wire path without a full vindex.
  Enum dispatch (no `async-trait`).
- `larql-server/src/bootstrap.rs` — opt-in registration: when
  `--shard-query-tau <TAU>` is passed alongside `--grpc-port`, the
  server adds `ShardServiceServer` to the existing tonic builder
  chain (next to `VindexServiceServer` + `ExpertServiceServer`),
  wired over a *shared* `Arc<RwLock<PatchedVindex>>` cloned from
  `LoadedModel.patched`.
- `larql-server/src/state.rs`: `LoadedModel.patched` is now
  `Arc<RwLock<PatchedVindex>>` (was `RwLock<PatchedVindex>`).
  Deref-coercion preserves every existing `.read().await` /
  `.write().await` call site unchanged; only the 12 construction
  sites needed `Arc::new` wrapping. Patches added at runtime are
  immediately visible to both the inference path and the shard
  service — no snapshot, no copy.
- `larql-server/tests/test_shard_query.rs` — 4 round-trip
  integration tests over a real TCP socket: hit / miss-below-tau /
  unknown-layer / **live patch propagation** (proves the shared-Arc
  refactor — a patch added through one Arc handle surfaces on the
  next `Query` through another handle).

**Caveat:** lifting this effectively promotes "Multi-machine MoE" from
P2 → P1 per `ROADMAP_STATUS`.

Test counts: **34 shard_query tests** (30 unit + 4 integration);
shard_query.rs coverage 96.78%.

---

### Exp 41 — LAN preregistration matrix ✅ shipped 2026-05-15

**Spec**: `experiments/41_residual_transport_grid/{SPEC.md,REPORT.md:508-547}`.

Ported `run.py` orchestration into the Rust CLI as `larql bench
--bench-grid-lan PATH`. The Rust runner reads the same JSON config
schema (`runs[*]` with `id`, `command` template, `env`, optional
`estimate`) and emits a JSONL manifest with the same field shape, so
existing Python tooling reading `runs.jsonl` keeps working.

**What shipped:**
- `crates/larql-cli/src/commands/primary/bench/grid_lan.rs` — pure
  helpers (config types, `command_for` template substitution,
  `parse_bench_output`, `estimate_bytes` / `q8k_bytes`, CoV +
  retry-decision, `safe_name`, `selected_runs`). Unit-tested at 99.3%
  line coverage.
- `crates/larql-cli/src/commands/primary/bench/grid_lan_runtime.rs` —
  subprocess driver: per run, spawns `larql bench …`, archives
  stdout/stderr, captures returncode, writes JSONL. Excluded from
  coverage (matches `*_runtime.rs` convention).
- CLI flags on `larql bench`: `--bench-grid-lan PATH`,
  `--grid-lan-out DIR`, `--grid-lan-only ID` (repeatable),
  `--grid-lan-include-disabled`, `--grid-lan-dry-run`,
  `--grid-lan-cov-threshold` (default 0.15, mirrors Exp 41 spec),
  `--grid-lan-extra-repeats` (default 2).
- Exp 41 §LAN Preregistration retry rule: after the base repeats,
  the orchestrator computes per-row CoV across the
  `mean_ms_per_tok` samples and runs up to `extra_repeats` more times
  when the threshold trips.

Smoke-tested with the experiment's `config.example.json` —
`--grid-lan-dry-run --grid-lan-include-disabled` walks the full
5-run matrix and produces a structurally equivalent JSONL to
`run.py --dry-run`.

---

## Next work — by theme

Items are tagged **P1** (active or next-up), **P2** (well-defined,
implementation sketch exists, 3-6 month horizon), **P3** (recognized
future work, no concrete plan yet). Everything surfaced during the
2026-05-16 doc/spec review is folded in.

P1 is **empty by default** — items move into P1 only when explicitly
chosen as next work. The candidate pool below is the menu.

---

### Theme: Dense model sharding

The router's bread-and-butter use case — pipeline-parallel models
across many hosts.

**Shipped:** ADR-0003 (static `--shards`), ADR-0004 P1–5
(self-assembling grid), ADR-0011 (Mode B + replication), ADR-0014
(hot-shard load-rate replication), ADR-0013 (routing comparator).

**P2 — well-defined, implementable:**

- **Auto-shard planner.** Given a `vindex` + N hosts with declared
  RAM budgets, compute a layer assignment that minimises shard-size
  variance under the per-host memory cap. Today the operator picks
  `--layers` manually per host; auto-plan would mean the router (or a
  one-shot `larql-router plan` command) emits a recommended map.
- **Heterogeneous-aware routing.** Server announce carries a `host_kind`
  hint (e.g. `gpu_metal`, `gpu_cuda`, `cpu`). The 3-tier comparator
  gains a 0th tier that prefers GPU hosts for compute-heavy layers
  (`lm_head`, attention) and CPU hosts for FFN-only shards. Extends
  ADR-0013 with a layer-kind classifier.
- **Mid-flight resharding without packet drop.** Today's
  drain-then-reassign (ADR-0011 Phase B2) is operator-driven via
  `admin assign`. P2 makes it traffic-driven: if a host's
  `ram_used / ram_total` exceeds a threshold AND a spare with more
  RAM exists in the available pool, the rebalancer initiates a
  drain-and-reassign on the smaller host. Requires safe handover —
  spare needs to be `Ready` before the original is `Unassign`ed.

**P3 — speculative:**

- **Tensor parallelism (single layer split across hosts).** Current
  model is layer-pipeline; tensor-parallel would split a single
  attention head across hosts. Major proto surgery (per-head IDs,
  partial-residual aggregation) — only worth it for models that
  don't fit on a single host even at one-layer-per-host granularity.
- **Cross-host KV cache reuse.** Attention layers cache K/V per
  prefix. If a prefix is shared across requests (system prompt,
  common preamble), routing same-prefix requests to the same host
  reuses the cache. Needs sticky session routing keyed on prefix
  hash.

---

### Theme: MoE model sharding and routing

**Shipped (ADR-0018, 2026-05-16):**

- **Proto extension** — `AnnounceMsg` / `ReadyMsg` / `AssignMsg`
  carry `expert_start` / `expert_end`. Dense servers send `0/0`;
  MoE shards advertise a contiguous expert range.
- **`ServerEntry::owns_expert(expert_id)` + `is_dense()`** — every
  helper that filters by expert ID short-circuits when the server is
  dense, so dense routing pays zero extra cost.
- **`route_expert(model, layer, expert_id)` + `route_all_experts`** —
  three-tier comparator (ADR-0013) over the filtered candidate set.
- **HTTP shape** — `/v1/walk-ffn` accepts `{layer, experts: [...]}`
  or `{layer_experts: [{layer, experts}, ...]}` alongside the
  existing dense shapes. MoE dispatch is grid-only; static `--shards`
  servers see a 503.
- **Replication keys widen to 5-tuples** — `under/over_replicated_ranges`,
  `find_origin_for`, `try_assign_gap`, `effective_target_for`,
  `send_assign_to_named_available`, `least_loaded_in_range` all key
  on `(model, layer_start, layer_end, expert_start, expert_end)`.
  Two shards sharing a layer range but owning different experts are
  treated as distinct slices.
- **Hot-shard elevation set widens** — `hot_layer_ranges`,
  `mark_elevated`, `demote_elevated`, `elevated_ranges_snapshot` all
  emit/take 5-tuples. Hot saturation on one expert-shard elevates
  only that shard, not its sibling.
- **`larql_router_grid_shard_kind{kind=dense|moe}`** — bounded-
  cardinality Prometheus gauge for grid-wide MoE health.
- **Coverage** — 19/20 files at 90%+ post-MoE + ADR-0020, total
  93.21% (`grid/service.rs` at 89.87% — within its 88% debt
  baseline; `main.rs` excluded from per-file).
- **Dense regression** — all 202 pre-MoE tests still green plus the
  post-MoE/ADR-0020/chaos additions (163 lib + 47 integration =
  210 tests, 211 with `--features http3`); bench shows dense
  `route()` within ±10% of the pre-MoE baseline (the expert filter
  is a single boolean check on a dense `ServerEntry`).

**Target deployment scale (per ADR-0018 §Target deployments):**
DeepSeek-V3 (671B / 60 layers × 256 experts), Kimi K2 / K2.6
(~1T-class), DeepSeek-V4 (≥1T). One physical host per (single
layer, expert-subset) shard; route table stays tractable because the
route_table is keyed on `(model, layer)` and expert filtering happens
inline.

**P2 — future MoE extensions:**

- **Expert affinity routing.** Same expert ID routes to same host
  repeatedly so the host's KV/MLP cache stays warm. Adds a 4th tier
  to the routing comparator. Deferred from ADR-0018 — needs real
  workload data showing the cache-warmth signal is meaningful.

**P3 — needs more discovery:**

- **Expert specialization with refusal.** Hosts may load a subset of
  experts and `Refuse` requests for experts they don't own. Today's
  `RefuseMsg` is for Mode B assignment refusal; expert-level refusal
  is a new semantic.
- **Binary wire format v2 with expert IDs.** Today's binary protocol
  is dense-only (ADR-0018 §"Binary protocol stays single-dimension").
  ADR-0009 (wire-format evolution) is the spec hook.
- **Admin RPC `AssignRangeRequest` expert fields.** Today's admin
  `assign` is dense-only. Additive proto change, no design surprises.

---

### Theme: Splitting large models (deployment-time concerns)

How an operator actually gets a 26B / 70B / 405B model running on a
heterogeneous cluster.

**Shipped:** Static `--shards` + Mode B available pool, multi-host
deploy walkthrough ([`crates/larql-router/docs/multi-host-demo.md`](docs/multi-host-demo.md)
— 3-box LAN topology covering router + 2 shards over `--grid-key`,
firewall rules, NTP, MTU gotchas, plus a QUIC variant for ADR-0010
and a MoE variant for V3/V4-scale models), vindex shard-download
endpoint ([`crates/larql-server/docs/router-spec.md`](../larql-server/docs/router-spec.md)
§4 — `GET /v1/shard/{model_id}/{start}-{end}` serves the vindex
directory as a streamed tar, client side at
`crates/larql-server/src/shard_loader.rs` is idempotent + SHA-256
verified + atomic-unpack, exercised end-to-end by
`crates/larql-server/tests/test_grid_mode_b.rs::mode_b_full_vertical_handoff`
against a real donor; 2026-05-16 audit closed the docs gap).

**P2 — extends auto-shard planner:**

- **Large-model bootstrap timeline.** Warm-up loading curve, vindex
  preload, attention buffer allocation under shard ownership. Today's
  Mode B path treats "Ready" as a single event; large models would
  benefit from progress reporting (`LoadingMsg { pct }`) so the
  rebalancer doesn't see a 10-minute load as a 10-minute stall.
- **Disk + RAM constraint solver.** Available-pool advertises
  `ram_bytes` and `disk_bytes` but `try_assign_gap` only checks RAM.
  Add disk gating so spares without enough disk for the vindex slice
  are skipped.

**P3 — future:**

- **Multi-vindex models** — different layers loaded from different
  `.vindex` files. Useful for fine-tuning experiments (swap one
  layer's weights to compare). Today each server loads exactly one
  vindex.

---

### Theme: Self-healing grid

Replication + gap-fill + stale eviction cover the happy reliability
paths. The gaps are in **partial-failure** and **adversarial-load**
scenarios.

**Shipped:** Stale-heartbeat eviction (ADR-0011), replication ticks,
gap-fill on Dropping/disconnect (ADR-0004 P2), Phase B2 drain-then-
reassign (ADR-0011), hot-shard load-rate replication (ADR-0014),
two-threshold hot-shard hysteresis (ADR-0014 amendment, demote at
0.8×T), backpressure filter in `route()` /
`route_expert()` (ADR-0020, `--saturation-ceiling N`,
`larql_router_route_saturation_total` counter, 503 with
`Retry-After: 0.5` on saturated dispatch), long-running chaos test
(`tests/test_grid_chaos.rs`, 5,000 random churn ticks × 2 variants,
asserts ledger consistency + coverage floor + no `route()` panic).

**P1 — reliability gaps surfaced in reviews:**

**P2:**

- **Multi-failure recovery scenarios.** Stress-test with N
  simultaneous failures (3+ servers crash at once). Today's
  rebalancer ticks every 30 s by default; in a 3-server-fail event
  the gap-fill and replicate paths fire in the same tick — verify
  ordering doesn't dispatch two AssignMsgs for the same range.
- **Network partition tolerance.** Router-server unreachable but
  server-server reachable. Today the router would deregister servers
  it can't see. A "partition-suspected" mode could hold deregistration
  for K seconds to avoid mass-eviction on a switch flap.
- **Cascade-failure isolation.** A slow shard backs up requests
  upstream; without a hop-budget circuit-breaker the slow shard's
  upstream peers also slow down. Add a fail-fast hop budget to
  `walk-ffn`.

**P3:**

- **Split-brain protection for multi-router deployments.** Two
  routers both think they're authoritative for the same grid; they
  could send conflicting `AssignMsg` to the same available server.
  Resolution needs either consensus (raft over a small router set)
  or sticky-leader (one router authoritative per `model_id`).

---

### Theme: Latency (router on the hot path)

Today's per-call wire RTT (`README.md` snapshot): TCP HTTP ~660 µs,
UDS HTTP ~510 µs, gRPC streaming ~460 µs. Across a 30-layer model
sharded into 2 hosts (15 hops × 2 = 30 layers serial), wire alone is
~14 ms — a meaningful chunk of decode time.

**Shipped:** 3-tier route() (ADR-0013), GT3 layer-latency in
heartbeats, active-probe RTT, 110 ns route() in production-shape
benches, connection pool tuning, real HTTP/3 shard transport with
per-stream independence (ADR-0019, 2026-05-16 — `--http3-shards` /
`--http3-port` opt-in, `H3Client::post_json` + `serve_axum` in
larql-router-protocol, used by the MoE expert fan-out path when
`h3_client: Some(_)`), hedged dispatch (ADR-0021, 2026-05-16 —
opt-in via `--hedge-after-ms M`; the multi-shard fan-out picks a
secondary replica per sub-request and dispatches it M ms after the
primary if the primary hasn't responded; halves p99 tail latency in
topologies with `--target-replicas ≥ 2`).

**P1 — biggest near-term win:**

- **(Pre-ADR-0021 "speculative next-layer prefetch" — falsified.)**
  An audit during the 2026-05-16 session found that the inference
  side sends one batched `/v1/walk-ffn` per token with the full
  layer list against a single input residual; the router fans every
  sub-request out in parallel against that input. There is no
  layer-N → layer-N+1 dependency at the router boundary, so
  "prefetch layer N+1 while N is in flight" doesn't apply here.
  Cross-token speculation, if it lands, is a client-side
  (`larql-inference`) concern. The legitimate router-layer
  interpretation is hedged dispatch — that shipped as ADR-0021
  (see Shipped above).

**P2:**

- **Wire RTT budget audit.** Real measurement of where the 460 µs is
  going (gRPC framing, TLS, socket queueing, axum middleware).
  Likely yields actionable per-stage optimisations.
- **Connection-pool tuning at scale.** Current
  `pool_max_idle_per_host(16)` was chosen for 2-shard deployments.
  At 20+ shards the pool churn dominates. Auto-size based on observed
  shard count?
- **Native UDS for same-host shards.** When router + server are on
  the same host (single-box dev mode), Unix domain sockets shave
  ~150 µs per call vs loopback TCP. Detect same-host via
  `listen_url` and prefer UDS when available.

**Shipped (was P3):** Real HTTP/3 with per-stream independence — see
the **Latency-shipped** entry above and ADR-0019. Both prerequisites
landed in the same session: MoE expert fan-out (ADR-0018) and the h3
transport (ADR-0019, h3 0.0.8 + h3-quinn 0.0.10 + h3-axum 0.2,
`--http3-shards` / `--http3-port`). The fan-out path branches to h3
when `h3_client: Some(_)` is wired into `AppState`. No HoL benchmark
yet — needs real multi-shard MoE traffic to surface (separate P2
item under Throughput).

---

### Theme: Throughput / speed

Latency is per-request; throughput is requests/sec at p99. The
router rarely bottlenecks throughput on its own (route() is
constant-time), but rebalancer and wire decisions shape what the
fleet can sustain.

**Shipped:** Bench harness (ADR-0012 GT9), production-shape +
worst-case bench scenarios.

**Shipped:** Concurrent-route bench
(`benches/routing.rs::bench_route_concurrent`, 2026-05-16) drives
`route()` from 1 / 4 / 8 / 16 parallel tokio tasks against a single
`Arc<RwLock<GridState>>` — the lock shape `AppState::resolve_all`
actually uses. **Lock primitive swap** (2026-05-16):
`tokio::sync::RwLock<GridState>` → `parking_lot::RwLock<GridState>`
across `larql-router` and its tests. Every grid critical section is
short and sync (no `await` held across the lock), so the
synchronous primitive is correct — and the compiler will catch any
held-across-await pattern as `!Send` guards. Bench-driven
verification:

| Workers | tokio (before) | parking_lot (after) | Δ |
|---|---|---|---|
| 1 | 5.6 Melem/s | 6.4 Melem/s | +14% |
| 4 | 8.7 Melem/s | 11.1 Melem/s | +28% |
| 8 | 4.0 Melem/s | 7.2 Melem/s | **+80%** |
| 16 | 3.6 Melem/s | 6.1 Melem/s | **+70%** |

The pathological 8-worker collapse (worse than 1 worker) is fixed;
all worker counts now stay above the 1-worker baseline. Peak is at
4 workers (M3 Max has 8 performance cores; past that we hit
parking_lot's single-atomic read counter and E-core scheduling).
220 tests still pass. ArcSwap remains a P3 if write traffic ever
drops enough to amortise the copy-on-write cost; today's ~1k
heartbeats/sec on a 100-server grid makes parking_lot the sweet
spot.

**P1 — bench-driven, queued for separate work:**

- **Per-shard concurrency cap.** Hot-shard elevation reacts to
  `req_per_sec` but doesn't *cap* a shard. A misconfigured client
  flooding one shard can knock it over. Per-shard semaphore in the
  client-side dispatch path, or a server-side cap reported back.

**P2:**

- **Batched walk-ffn.** Today each layer is a separate HTTP call to
  its owning shard. If three consecutive layers all live on the
  same shard, batching them into one call halves overhead per
  three-layer run. Existing `route_all` already returns the layer-
  to-url map; the dispatch side needs to group same-URL layers
  before issuing requests.
- **Wire format options (GT8 from ADR-0012).** f16 / i8 residuals
  cut wire bytes proportionally. f16 is the obvious win (2× wire
  reduction, ~no quality loss); i8 is a step further with
  quantisation error to characterise.
- **GT10 from ADR-0012 — CI regression gate.** A shell script that
  runs a stored baseline and fails the build if throughput or tail
  latency regresses beyond thresholds. ADR-0012 sketched this;
  not implemented.

**P3:**

- **GPU-aware throughput tuning.** Once heterogeneous routing exists,
  the rebalancer can pack GPU hosts to a higher utilisation target
  than CPU hosts.
- **FP4 wire format (post-V2 generality).** Quarters wire bytes per
  Exp 26 result; needs the V2 generality work in `larql-vindex`
  before it's safe to ship as default.

---

### Theme: Operability (observability, admin, deployment)

The router currently emits logs and exposes a status RPC. Production
operations need more.

**Shipped:** Admin CLI (ADR-0004 P5) — `status` / `gaps` / `drain` /
`assign`. Hot-shard demo doc.

**Shipped:**

- **Prometheus `/metrics` endpoint** ✅ shipped 2026-05-16
  (ADR-0017). Counters for grid registers/deregisters (split by
  reason), rebalancer-tick outcomes (replicate / drop / elevate /
  demote / evict / unassign_imbalance), RTT probe outcomes
  (success / non_2xx / error), walk-ffn requests (success /
  error_4xx / error_5xx). Histogram for walk-ffn end-to-end
  duration. Gauges (refreshed at each rebalancer tick) for server
  count, distinct models, coverage gaps, elevated ranges,
  configured `--target-replicas`. Bounded cardinality — no
  `model_id` / `server_id` / `layer_id` labels. Unauth, same
  trust model as `/v1/health`.

**P2:**

- **JSON output mode for admin commands.** `larql-router status --json`
  for dashboard ingestion. `format_status` already separates rendering
  from data — slot a JSON serializer alongside the text one.
- **Multi-host deploy walkthrough doc.** Mirrors
  `docs/hot-shard-demo.md` but for a 2-box LAN topology, including
  TLS setup, firewall ports, and the `--quic-cert-fingerprint` flow.
- **`larql-router metrics` admin subcommand.** Dumps current
  Prometheus-style metrics to stdout for one-shot capture in
  scripts. Built on the same endpoint as above.

**P3:**

- **Web dashboard.** Axum-served minimal HTML, live grid state +
  rebalancer event stream. Probably worth it once metrics +
  multi-host docs are out.
- **Per-layer tau in ShardService (ADR-0015 open question).** Today
  tau is per-server; per-layer would mean tuning each layer's cache
  hit rate independently. Wait for usage data showing it matters.

---

### Theme: Cross-router federation (P2 originally)

Stays at P2 — well-defined but no implementation planned until Act 2
multi-host demo is complete. Multiple routers cover different
geographic regions; a client request is forwarded to the regional
router that owns the model shard. Requires either:

- a `RouterMsg` variant on `GridService.Join` so routers join each
  other's grids, or
- a separate `FederationService` for router-to-router routing decisions.

Probably blocks on the multi-host deploy walkthrough (Operability P2)
and on real cross-region perf data (Latency P2).

---

## Cross-references — workspace-level (other crates)

These don't live in the router crate but shape what the grid is asked
to serve. Tracked here so they're visible alongside router work.

- **Decode/prefill perf gap.** Per `crates/larql-router/README.md`
  perf snapshot: local Metal decode 86 tok/s vs ollama 98.7 tok/s
  (1.15× behind on decode). Per memory: prefill 4-14× behind ollama
  depending on prompt length. Lives in `larql-inference` /
  `larql-compute`; router only sees the result.
- **Compute crate split** (in flight, parallel session). Metal lifted
  out into `larql-compute-metal` sibling crate. Brief workspace
  resolver hiccup observed mid-session; resolved by 2026-05-16 EOD.
- **Exp 27 — hash routing across all layers (V1).** Top-2048 mask,
  100% argmax recovered at KL=0.030 at L0 on Gemma 3 4B
  (`ROADMAP_STATUS` item #2). L0 result is interp-validated; scaling
  across layers and architectures is the next step. Router's interest
  is in the resulting vindex shape — FFN rows become sparse-
  addressable, which changes shard-size economics.
- **Exp 26 — FP4 generality (V2).** `gemma3-4b-f16.vindex` is **99.83%
  per-feature block R<16-compliant natively, no QAT**; `down` is the
  long tail at 99.65% (`ROADMAP_STATUS` item #3). Extending the audit
  script across architectures is the next step. Router impact: FP4
  shards quarter the wire-bytes-per-tok metric tracked by the bench
  harness.
