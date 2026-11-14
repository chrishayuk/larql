# Changelog — larql-router / larql-router-protocol

All notable changes to `larql-router` and `larql-router-protocol` are
documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/) conventions
with dated entries (`YYYY-MM-DD`) instead of semantic versions during the
pre-1.0 phase. Forward-looking work lives in [`ROADMAP.md`](ROADMAP.md).

Entries were migrated from `ROADMAP.md` on 2026-08-04 and preserve the date and
voice they were originally written in. Each retains its original `### GTn` /
`### Phase n` heading so cross-references from ADRs and specs still resolve.

## [2026-11-12] — Keyless public binds are refused; default bind is loopback

**Breaking default:** `--host` now defaults to `127.0.0.1` (was `0.0.0.0`),
matching `larql-server`. A router that must be reachable from other hosts
needs an explicit `--host`.

With a non-loopback `--host`, the router refuses to start (exit 1, message on
stderr) unless the grid is keyed — `--grid-key` or `LARQL_GRID_KEY`, non-empty
— or the operator passes the new `--insecure-public` flag. The check covers
every listener, since HTTP, grid gRPC and QUIC all bind `--host`. A router with
no `--grid-port` has no credential, so on a network address it needs
`--insecure-public` even if a key is given. Policy lives in
[`exposure.rs`](src/exposure.rs).

## [2026-08-22] — N0-router: OpenAI surface on the grid front door

Clients can now point an unmodified `openai` SDK at the **router** and the
grid answers as a single endpoint:

- **`AnnounceMsg.serves_openai`** (grid.proto field 9, backward-compatible):
  a server sets it when it can serve complete OpenAI requests by itself —
  full layer coverage, inference enabled, no `--layers` / `--experts` /
  `--units` filter and not `--no-infer` / `--ffn-only` / `--embed-only`.
  Mode B gap-fill replicas always register as `false` (layer slices).
- **`GET /v1/models`** aggregates distinct model ids across
  OpenAI-capable servers into the OpenAI list shape (`owned_by: "larql"`).
- **`POST /v1/chat/completions` / `/v1/completions` / `/v1/embeddings`**
  proxy verbatim to the least-loaded capable server matching the request's
  `model` field (absent `model` = any capable server), stream the response
  back unbuffered (SSE passes through chunk-by-chunk), and forward the
  client's `Authorization` header so backend `--api-key` still applies.
- Errors are emitted in the OpenAI envelope: 404 `model_not_found` for an
  unknown model, 503 when no capable server is registered (including the
  gridless / static-`--shards` case — static maps carry no capability
  signal), 502 for an unreachable backend.

Implementation: `src/openai/` (mod 96.5% / responses 95.0% line
coverage); e2e tests in `tests/test_openai_proxy.rs` drive real HTTP
against stub backends.

**Responses API (same day):** `/v1/responses` + `GET`/`DELETE
/v1/responses/{id}` are proxied with *sticky routing*
(`src/openai/responses.rs`): the router learns each response's id by
observing the proxied bytes (the envelope and the streaming
`response.created` event lead with `"id":"resp_..."`) and keeps a
bounded FIFO id → backend map (`AppState.openai_responses`). A `POST`
carrying a known `previous_response_id` goes back to the producing
server regardless of load ordering; by-id retrieval routes via the map,
falls through to the only capable server on a single-server grid
(surviving a router restart), and otherwise answers 404
`not_found_error` — the router cannot know which server holds an
unrecorded id. `DELETE` drops the route once the backend confirms.

## [2026-05-28] — Hardening findings from the whole-codebase review

From the whole-codebase review ([`docs/audits/codebase-review-2026-05-28.md`](../../docs/audits/codebase-review-2026-05-28.md)):

- **P1 — validate announced layer ranges.** The announce path builds an unbounded route table (`src/routing.rs:237`) from gRPC-announced ranges with no validation; clamp the span to sane model depth before `rebuild_route_table`. DoS class.
- **larql-router-protocol** — a `None` fingerprint disables TLS verification on a public API; document the contract or gate it behind explicit opt-in.

Both recorded, not fixed. Still open — see [`ROADMAP.md`](ROADMAP.md) §"Open
defects".

## [2026-05-16] — Perf snapshot and coverage baseline

Point-in-time measurements taken on an M3 Max. Kept as the reference the next
run is compared against; they are **not** a statement about current
performance.

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

### Coverage baseline at this date

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


## [2026-05-16] — RTT-based routing; Exp 53 — Rust port of the sharded-vindex shard endpoint

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


## [2026-05-15] — GT7 — QUIC transport; Phase 5 — Admin CLI; Hot-shard load-rate replication; Stale heartbeat eviction; Exp 41 — LAN preregistration matrix

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

## [2026-05-13] — GT5 — Mode B: gap-fill assignment; GT6 — Dynamic rebalancing

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


## [2026-05-07] — GT3 — Per-layer latency in HeartbeatMsg; GT9 — Criterion routing benchmarks

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


### GT9 — Criterion routing benchmarks ✅ shipped 2026-05-07

**Spec**: ADR-0012 §Layer 2.

**What shipped:**
- `crates/larql-router/benches/routing.rs`: `bench_route_single_layer`,
  `bench_route_all`, `bench_heartbeat_update`, `bench_rebuild_route_table`
  at 1/10/100 servers × 30/62 layers.
- `src/lib.rs` exposes `pub mod grid` for bench linking.
- Makefile: `make bench-routing` / `make bench-all`.

---

