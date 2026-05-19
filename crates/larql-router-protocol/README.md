# larql-router-protocol

Generated tonic/prost types for the three gRPC services that wire
`larql-router` together with `larql-server` shards, plus a thin QUIC
transport wrapper that the router and server share when
`--features quic` is enabled.

This crate is intentionally narrow — it holds the proto contracts and
nothing else. The router orchestration (route table, rebalancer,
admin endpoints) lives in `crates/larql-router`; the server-side
announce client + KNN shard cache live in `crates/larql-server`.

## Crate layout

```
proto/
    grid.proto      ← GridService — self-assembling grid lifecycle
    expert.proto    ← ExpertService — MoE expert dispatch over gRPC
    shard.proto     ← ShardService — sharded vindex KNN cache (Exp 53)
src/
    lib.rs          ← re-exports tonic::include_proto! generated code
    transport/
        mod.rs
        quic.rs     ← QUIC server/client endpoints, fingerprint-pinned TLS
build.rs            ← tonic-build invocation
```

`build.rs` invokes `protoc`. On Linux/macOS it pulls in
`protobuf-src` so the toolchain is vendored; on Windows it expects
`protoc` to be on `PATH` (CI installs it via `arduino/setup-protoc`).

## GridService

Persistent bidirectional stream between each shard and the router.

```
Server                                   Router
  │                                        │
  │── Join (stream ServerMessage) ──────►  │  open bidi stream
  │                                        │
  │── AnnounceMsg  (Mode A: shard loaded)  │  register + AckMsg
  │── AvailableMsg (Mode B: spare ready)   │  enter available pool
  │                                        │
  │── HeartbeatMsg every 10s ───────────►  │  update cpu/ram/rif
  │      .layer_stats[]   (GT3)            │  + per-layer EMA + p99
  │      .req_per_sec     (hot-shard)      │  + shard request rate
  │                                        │
  │◄────────────── AssignMsg ──────────────│  (router pulls Mode B
  │                                        │   spare into a gap)
  │── ReadyMsg ─────────────────────────►  │  Ack; spare → serving
  │                                        │
  │◄────────────── UnassignMsg ────────────│  (rebalance / over-rep)
  │── DroppingMsg ─────────────────────►   │  drain + re-enter Mode B
  │                                        │   if --available-ram set
```

Plus three unary RPCs the router uses or operators call directly:

| RPC | Purpose |
|-----|---------|
| `Status(StatusRequest)` | Full grid coverage + per-server stats. Backs `GET /grid-status` and `larql-router status`. |
| `DrainServer(DrainRequest)` | Send `UnassignMsg` to a named server. Backs `larql-router drain`. |
| `AssignRange(AssignRangeRequest)` | Force-assign a layer range. Backs `larql-router assign`. |

### Message reference

| Message | Direction | Notes |
|---------|-----------|-------|
| `AnnounceMsg` | Server → Router | Pre-loaded shard registers (`model_id`, `layer_start`, `layer_end`, `ram_bytes`, `listen_url`, `vindex_hash`). |
| `AvailableMsg` | Server → Router | Mode B: free RAM + disk + store path. |
| `ReadyMsg` | Server → Router | Sent after a Mode B assignment finishes downloading + loading. |
| `HeartbeatMsg` | Server → Router | Every 10 s. Carries CPU%, RAM used, in-flight req count, `LayerLatency` snapshots, and `req_per_sec`. |
| `DroppingMsg` | Server → Router | Clean exit + reason (`shutdown` / `reassigned` / `oom`). |
| `RefuseMsg` | Server → Router | Mode B reject (e.g. insufficient disk). |
| `AssignMsg` | Router → Server | Mode B assignment: `(model_id, layers, origin_url, shard_hash)`. |
| `UnassignMsg` | Router → Server | Drain trigger + reason (`redundant` / `rebalancing` / `over_replicated`). |
| `AckMsg` / `RejectMsg` | Router → Server | Registration ack with stable `server_id`, or reject. |

### Auth

The `Join` stream accepts an `Authorization: Bearer <key>` header.
When the router is started with `--grid-key` (or `LARQL_GRID_KEY`),
streams missing or mis-matching the bearer get
`UNAUTHENTICATED`. Without a configured key the grid is open — dev
only.

## ExpertService

Used by the router (or any client doing its own forward pass) to
dispatch MoE experts to the owning shard. Two RPCs:

- `ExpertBatch(ExpertBatchRequest) → ExpertBatchResponse` — one
  unary call carries every selected `(layer, expert_id, residual)`
  for the current decode step.
- `ExpertStream(stream ExpertLayerInput) → stream ExpertLayerOutput`
  — one bidi stream per decode step. The client sends one
  `ExpertLayerInput` per MoE layer as `h_post_attn` becomes
  available; the server returns the combined `h2` contribution per
  layer. Eliminates per-layer connection setup vs the HTTP path.

Both encode hidden vectors as raw little-endian `f32` bytes (length
= `hidden × 4`) rather than `repeated float` to dodge proto varint
overhead.

## ShardService

Sharded vindex KNN cache (Exp 53). A `larql-server` can host a
pre-compiled `(input, output)` cache at one or more layers and
answer remote KNN queries — clients running their own forward pass
replace one FFN layer's compute with a single unary RPC. On a hit
(cosine ≥ tau) the server returns the matching MLP output; on a
miss the client falls back to local FFN.

| RPC | Input | Output |
|-----|-------|--------|
| `Query` | `ShardQuery { layer_id, k, query_vec, tau_override }` | `ShardResult { hit, mlp_out, best_sim }` |

`query_vec` and `mlp_out` use the same raw f32 LE byte convention
as `ExpertService`. `tau_override = 0.0` means "use the
server-configured tau"; a positive value forces that threshold for
the call. `best_sim` is reported on hit *and* miss for telemetry /
threshold tuning.

The server side lives in `crates/larql-server/src/shard_query.rs`
and is registered when `larql-server --grpc-port <P>
--shard-query-tau <TAU>` are both set. Two backends share a
`ShardSource` enum:

- `ShardSource::Vindex` — production. Queries the server's loaded
  `PatchedVindex` via `gate_knn` + `ffn_row_into`. "Compiled facts"
  live as vindex patches (`insert_feature` / `set_down_vector`),
  so the cache piggy-backs on the existing vindex format and
  there's no separate on-disk artefact to maintain.
- `ShardSource::Cache` — in-memory fixture used by tests, lets
  callers exercise the wire path without standing up a full vindex.

End-to-end demo + microbench:

```bash
# Wire walkthrough — empty / patched / orthogonal / cache paths
cargo run --release -p larql-server --example shard_query_demo

# In-process ShardSource::lookup hot-path bench
make bench-shard-query
```

Latest in-process bench numbers on M3 Max (criterion median):
`cache_lookup` 2.1 µs at n=16/d=256, 43 µs at n=64/d=1024, 171 µs at
n=256/d=1024; `vindex_lookup` 3.1 µs / 46 µs / 180 µs at the same
shapes after the 2026-05-16 `PatchedVindex::gate_knn` optimization
pass: empty-base short-circuit, O(overrides²) → O(overrides) merge
via `HashMap<feat, hit_idx>`, `argmax` instead of sort at
`top_k = 1`, sort-skip when only deletions modified hits, and a
per-layer lazy contiguous-matrix cache with per-layer invalidation.
Cache↔vindex gap closed from ~20% to ~7% at n=256.

The per-layer invalidation wins on cross-layer mutation workloads.
A/B comparison: querying layer 0 after a layer-1 mutation costs
46.6 µs (per-layer, current) vs 54.0 µs (whole-cache, prior
behavior) at n=64/d=1024, and **181.9 µs vs 218.8 µs at
n=256/d=1024 — a 17% saving on the cross-layer query**. The savings
scale with the queried layer's feature count (cost of the matvec
we'd otherwise re-do on every mutation). See
`bench_vindex_cross_layer_mutation*` in
`crates/larql-server/benches/shard_query.rs`.

The full gRPC wire path adds ~5–10 ms of tonic round-trip on
loopback (see the demo output) — dominant cost is the transport,
not the KNN itself.

## QUIC transport (opt-in)

Enable with `--features quic` on this crate (the router and server
re-export it through their own `quic` features).

```rust
use larql_router_protocol::transport::quic::{
    self_signed_tls, server_endpoint, spawn_accept_loop,
    client_endpoint, connect_grpc_channel,
};

// ── Server ───────────────────────────────────────────────────────
let tls = self_signed_tls("router")?;           // or load PEM from disk
println!("cert fingerprint: {}", tls.fingerprint);
let endpoint = server_endpoint("0.0.0.0:5052".parse()?, &tls)?;
let incoming = spawn_accept_loop(endpoint);
tonic::transport::Server::builder()
    .add_service(GridServiceServer::new(grid_impl))
    .serve_with_incoming(tokio_stream::wrappers::ReceiverStream::new(incoming))
    .await?;

// ── Client ───────────────────────────────────────────────────────
let endpoint = client_endpoint(
    "0.0.0.0:0".parse()?,
    Some(server_fingerprint_hex),       // None = LAN-only skip-verify
)?;
let (_conn, channel) = connect_grpc_channel(
    &endpoint, server_addr, "router",
).await?;
let mut client = GridServiceClient::new(channel);
```

Trust model is **SHA-256 leaf-cert fingerprint pinning** — no CA
chain, no hostname verification. The router prints its fingerprint
at startup; servers pass it via `--quic-cert-fingerprint`. `None`
disables verification entirely (`AcceptAny` verifier) for LAN/dev
work.

This is QUIC-as-TCP-replacement: HTTP/2 carried over a single
bidirectional QUIC stream. Buys 0-RTT reconnect, TLS 1.3, and
BBRv2 congestion control. Per-stream independence for fan-out
(real HTTP/3) is a future ADR.

## Cargo features

| Feature | Pulls in | Used by |
|---------|----------|---------|
| (default) | tonic + prost only — gRPC over TCP | Most callers. |
| `quic`  | quinn, rustls, rcgen, rustls-pemfile, hyper-util, tower, sha2 | Router + server with `--quic-port` / `--quic-cert-fingerprint`. |

## Validation

```bash
cargo test -p larql-router-protocol --features quic
make larql-router-protocol-coverage-summary
```

18 tests (15 unit + 3 QUIC round-trip integration tests covering
both fingerprint-pinned and skip-verify paths). Coverage at 91.36%
line on `transport/quic.rs` (the only instrumented source — proto
re-exports live in `target/` and are filtered out by `cargo-llvm-cov`).

## See also

- [`../larql-router/README.md`](../larql-router/README.md) — operator
  guide for the router itself.
- [`../larql-router/ROADMAP.md`](../larql-router/ROADMAP.md) — what
  shipped when, including the GT-numbered transport / rebalancer
  milestones referenced here.
- `../larql-server/docs/router-spec.md` — long-form protocol spec.
