# Distributed FFN — Layer Sharding, Expert Sharding and VINDEX3

**Status:** V2 dense remote FFN, remote MoE, layer/expert sharding and grid management are implemented. VINDEX3 CPU layer workers landed in [PR #507](https://github.com/chrishayuk/larql/pull/507) on 2026-09-23; V3 CPU dense FFN workers now keep attention and KV local; routed-expert placement remains consolidation work.

**ADRs:** [FFN router](../adr/0003-ffn-router.md), [FFN grid](../adr/0004-ffn-grid.md), [HTTP/3 shard transport](../adr/0019-http3-shard-transport.md)

**Operator references:** [router](../../crates/larql-router/README.md), [server](../../crates/larql-server/README.md)

---

## Overview

The deployment examples below describe the V2 FFN service. Its clients retain
attention and continuation state locally while servers compute FFN or selected
expert contributions. V3's current layer workers instead execute whole layer
ranges, including attention; see [VINDEX3 consolidation](#vindex3-consolidation)
for that distinction and the dense operation boundary.

A single `larql-server` holding a full vindex works for development. In production,
the vindex may exceed the RAM of any single machine. Layer sharding splits the
vindex across N servers, each owning a contiguous layer range. A `larql-router`
sits in front and routes requests transparently — the client uses `--ffn-remote`
unchanged and has no knowledge of the topology.

```
Client  (attention + embed, ~2.4 GB)
  │
  │  --ffn-remote http://router:9090  (unchanged)
  ▼
larql-router
  │  layers 0–16  →  larql-server A
  │  layers 17–33 →  larql-server B
```

---

## Memory Model

Each shard server only loads the layers it owns. The savings come from two places:

**Anon mmap (k-quant synthesised gate):** `synthesize_gate_from_q4k` allocates
an anonymous mmap and dequantizes gate weights into it. With `--layers 0-16` on
a 34-layer model, the allocation is `17/34 = 50%` of the full size. Only owned
layers are decoded; out-of-range layers leave a zero `GateLayerSlice` and are
never touched.

**Demand-paged files (gate_vectors.bin, interleaved_kquant.bin / legacy
interleaved_kquant.bin, etc.):** These are mmap'd as a whole — the virtual address
range covers the full file — but the OS only faults in pages that are read.
Because `is_layer_owned(layer)` guards every accessor before any byte is read,
out-of-range pages never enter physical RAM.

**Result:** shard RSS ≈ `(owned_layers / total_layers) × full_vindex_RSS`.

---

## Layer Sharding — Server

```bash
larql-server <vindex> --ffn-only --layers 0-16 --port 8080
larql-server <vindex> --ffn-only --layers 17-33 --port 8081
```

`--layers START-END` uses inclusive bounds. Internally the range is stored as
`(start, end+1)` (exclusive end). Requests for layers outside the owned range
are rejected immediately with HTTP 400:

```
{"error": "layer 20 not served by this shard (owned: 0–16)"}
```

### Implementation

| Location | What it does |
|---|---|
| `larql-vindex::VectorIndex::load_vindex_with_range` | Accepts `Option<(usize, usize)>` range; restricts anon mmap allocation and dequant to owned layers |
| `VectorIndex::is_layer_owned(layer)` | Returns false for out-of-range layers; called before any accessor touches mmap data |
| `VectorIndex::set_layer_range` | Sets the range after construction |
| `larql-server --layers` | Parses `"START-END"`, calls `load_vindex_with_range` |
| `routes/walk_ffn.rs` | Checks `is_layer_owned` for every requested layer before dispatch; returns 400 on mismatch |

---

## Router

Two dispatch modes:

**Static mode** — configured at startup with `--shards`:

```bash
larql-router \
  --shards "0-16=http://host-a:8080,17-33=http://host-b:8081" \
  --port 9090
```

**Grid mode** — servers self-register via gRPC; no static config needed:

```bash
# Router listens for server registrations on gRPC port 50052
larql-router --grid-port 50052 --grid-key "$KEY" --port 9090

# Servers announce themselves on startup
larql-server model.vindex --ffn-only --layers 0-16 \
  --join "http://router:50052" --grid-key "$KEY" \
  --public-url "http://server-a:8080"
```

Both modes can coexist. Grid takes priority; static shards are the fallback.

The router exposes `POST /v1/walk-ffn` — the same endpoint as `larql-server`.
The client's `RemoteWalkBackend` connects to the router with `--ffn-remote http://router:9090`
and is entirely unaware of the sharding topology.

### Dispatch

**Single-layer request** (`"layer": N`): the router finds the owning shard and
proxies the request body unchanged.

**Batched request** (`"layers": [N, M, ...]`): layers are grouped by owning
shard. Each shard receives a sub-request containing only its layers. All shard
sub-requests are dispatched in parallel. Results are merged and sorted by layer
before returning.

```
Request: layers=[5, 20]

  Shard A (0–16):  {"layer": 5,  "residual": [...]}  ─┐
  Shard B (17–33): {"layer": 20, "residual": [...]}  ─┤ parallel
                                                       ↓
  Merged: {"results": [{"layer":5,...}, {"layer":20,...}], "latency_ms": ...}
```

Shard execution overlaps, so fan-out latency is approximately
`max(shard_latencies)` plus dispatch and merge overhead.

**Unknown layer**: request is rejected at the router with HTTP 400 before any shard
is contacted.

**Health check**: on startup the router calls `GET /v1/stats` on each configured
shard. Unreachable shards are logged as warnings; the router still starts. Requests
to an unreachable shard will return HTTP 502 with the upstream error.

### Implementation

| Location | What it does |
|---|---|
| `crates/larql-router/src/main.rs` | CLI entry point (HTTP handler + `resolve_all` now in `src/http.rs`; shard-spec parsing in `src/shards.rs`) |
| `crates/larql-router/src/grid/` | `GridState` (O(1) route cache, `grid/mod.rs`), `GridServiceImpl` (gRPC, `grid/service.rs`) |
| `crates/larql-router-protocol/` | Shared proto types (`grid.proto`) and tonic stubs |
| `crates/larql-server/src/announce.rs` | Background announce task; reconnect with backoff |
| `parse_shards("0-16=http://...")` | Parses `--shards` spec; inclusive→exclusive end |
| `handle_walk_ffn` | Dispatch: `resolve_all` (single lock) → proxy or parallel fan-out |
| `proxy_to` | Single-shard proxy; propagates HTTP error status |

### Validation

```bash
cargo test -p larql-router
cargo test -p larql-server announce
```

These cover static shard parsing, binary layer peeking, self-assembling grid
route tables, heartbeat load updates, deregistration, status gap reporting, and
the server-side announce/heartbeat/drop protocol envelopes.

---

## Deployment Examples

### Two-shard local (Gemma 3 4B, 34 layers)

```bash
# Terminal A
larql-server output/gemma3-4b-q4k.vindex --ffn-only --layers 0-16 --port 8080

# Terminal B
larql-server output/gemma3-4b-q4k.vindex --ffn-only --layers 17-33 --port 8081

# Terminal C
larql-router --shards "0-16=http://127.0.0.1:8080,17-33=http://127.0.0.1:8081" --port 9090

# Client — unchanged
larql walk --ffn-remote http://127.0.0.1:9090 --predict --prompt "The capital of France is"
```

### Three-shard remote (Gemma 4 31B, 62 layers)

```bash
# Server A — layers 0–20   (~11 GB)
larql-server output/gemma4-31b-q4k.vindex --ffn-only --layers 0-20  --port 8080

# Server B — layers 21–41  (~11 GB)
larql-server output/gemma4-31b-q4k.vindex --ffn-only --layers 21-41 --port 8080

# Server C — layers 42–61  (~11 GB)
larql-server output/gemma4-31b-q4k.vindex --ffn-only --layers 42-61 --port 8080

# Router
larql-router \
  --shards "0-20=http://server-a:8080,21-41=http://server-b:8080,42-61=http://server-c:8080" \
  --port 9090
```

---

## Router Options

See the [router option reference](../../crates/larql-router/README.md).

Key flags:

| Flag | Default | Description |
|---|---|---|
| `--shards` | — | Static `START-END=URL` shard map |
| `--grid-port` | — | Enable self-assembling grid gRPC server |
| `--grid-key` | — | Shared auth secret (`LARQL_GRID_KEY` env var) |
| `--port` | 9090 | HTTP listen port |
| `--timeout-secs` | 120 | Per-request timeout to backend shards |

---

## Binary Wire Format

`RemoteWalkBackend` uses the binary wire format (`Content-Type:
application/x-larql-ffn`) by default, eliminating JSON float
serialization overhead on both the client and server.

### Performance (Gemma 3 4B, hidden_size=3072, seq_len=1)

| Format  | Request size | p50 latency |
|---------|-------------|-------------|
| JSON    | ~15.4 KB    | ~8.1 ms     |
| Binary  | ~10.3 KB    | ~7.6 ms     |

~33% smaller requests, ~0.5 ms/hop faster.

### Batched FFN requests

`RemoteWalkBackend.forward_all_layers(layers, x)` sends all layers in a
single HTTP round trip (binary batch request). All layers in a binary request
must belong to one shard when sent through the router. JSON batches can fan
out across shards in parallel; their shard execution time is approximately
the slowest shard's time, plus dispatch and merge overhead.

This API evaluates the supplied residual at each requested layer. It does not
collapse an autoregressive layer stack into one network round trip: each next
layer needs the preceding layer's updated residual and local attention result.

```rust
let backend = RemoteWalkBackend::connect(RemoteFfnConfig::new("http://router:9090"))?;
let layer_outputs: HashMap<usize, Array2<f32>> =
    backend.forward_all_layers(&(0..34).collect::<Vec<_>>(), &residual)?;
```

### Constraints

- Binary format requires `full_output = true`.
- Multi-shard binary fan-out is not supported at the router. Use JSON
  for cross-shard batches, or route shard-local batches directly to the
  shard.
- `model_id` is not in the binary format; multi-model grids use the
  default routing for that layer.

---

## Remote MoE and expert ownership

[RemoteMoeBackend](../../crates/larql-inference/src/ffn/moe_remote/mod.rs)
implements remote expert execution. For the hybrid Gemma path, the client owns
attention, KV state, the dense/shared computation and router weights. It selects
top-K experts locally, groups them by destination, dispatches shards in parallel,
and assembles the expert contribution before continuing the layer.

The current layer-batch path sends one residual plus selected `(expert_id,
weight)` pairs per shard. Each server returns a weighted partial sum; the client
sums those partials and applies the post-expert normalization. The legacy
per-expert batch path also remains available. This placement of weighting and
normalization is part of the numerical contract.

Servers accept inclusive `--experts START-END` ranges alongside `--layers`.
Direct clients use `--moe-shards`; the grid separately tracks `(layer, expert)`
ownership and dispatches expert-aware requests. Grid MoE routing requires
`--grid-port`; a static layer-only `--shards` map does not supply expert ownership.
See the [server's remote MoE topology and commands](../../crates/larql-server/README.md#remote-moe-shard-topology).

Transport support is path-specific:

| Path | Implemented capabilities |
|---|---|
| Direct remote MoE client | HTTP, Unix domain sockets, gRPC unary/streaming, batched expert requests and live `reshard()` |
| Expert wire formats | Binary layer batches, optional f16, and Q8_K multi-layer request helpers |
| Grid router to expert shards | HTTP fan-out; optional HTTP/3 with the `http3` feature and `--http3-shards` |
| Dense router batches | Binary pass-through to one shard; JSON fan-out across shards |

HTTP/3 support does not imply every client endpoint uses QUIC, and direct gRPC
expert dispatch does not imply the dense router proxies requests over gRPC.

## Grid management already implemented

- **Mode B (available workers):** a server advertises capacity, receives an
  assignment, downloads the assigned shard and announces readiness. See
  [the announce implementation](../../crates/larql-server/src/announce.rs).
- **Admin CLI:** `larql-router status`, `gaps`, `drain` and `assign` are implemented
  in the [router CLI](../../crates/larql-router/src/main.rs).
- **Expert-aware routing:** the grid resolves layer/expert ownership and the
  [HTTP dispatcher](../../crates/larql-router/src/http.rs) groups requests by shard.

An FFN result L2 cache at the **router** remains separate work; the
[server FFN cache](../../crates/larql-server/src/ffn_l2_cache.rs) is a different
placement. Dense multi-shard binary fan-out remains unsupported.

## Measured evidence and latency limits

The [DEC funnel](../dec-funnel.md) records the Gemma 4 26B remote-FFN
single-stream loopback anchor at 27.8–28.6 tok/s after the July improvements.
It also records approximately 1,050 tok/s aggregate at B64 for the
dense/shared-expert batch tier. That is a tier measurement, not end-to-end
single-user generation throughput or a routed-expert result. The same programme
records historical field points of approximately 25 tok/s on LAN and 2–3 tok/s
on Fly.io London. None of these are V3 worker measurements.

For one dependent remote FFN crossing per layer, an illustrative network-only
cost is `layers × RTT`: 30 layers at 0.2, 5 or 20 ms RTT cost approximately
6, 150 or 600 ms per token before compute. Within-layer expert fan-out can run
in parallel, but successive layers still depend on each other. Bandwidth,
serialization, queueing and the slowest selected shard also contribute; the
RTT calculation alone does not establish a performance ceiling for every
deployment.

## VINDEX3 consolidation

The [merged CPU worker guide](https://github.com/chrishayuk/larql/blob/d05d9b787a51be7bf0d3a8b3d9c68f4bddeb2084/docs/vindex3/runtime-followups.md#cpu-layer-workers)
documents #507's executable path. For a two-layer container:

```bash
larql-server model.vindex3 --layers 0-0 --port 9181
larql-server model.vindex3 --layers 1-1 --port 9182
larql run model.vindex3 "Hello" \
  --v3-shards http://localhost:9181,http://localhost:9182
```

The coordinator prepares embedding/final norm/head operands; workers prepare
their declared layer ranges. The versioned binding identifies the declared
artifact and plan, CPU numerical provider revision, range, total layers and
hidden width. The coordinator rejects gaps, overlaps, ordering errors, identity
mismatches and malformed responses. Declared payload hashes identify the
artifact; container verification is still needed to check payload bytes.

Workers receive the complete prefix from position zero and recompute it on
every step, retaining no remote KV state. This path supports CPU single-stream
softmax stacks. Remote Metal, KDA/MLA stacks, grid discovery and remote
continuation caches are outside that implementation. All nodes still open the
same container; the worker path does not distribute shard files.

### Dense FFN operation provider

V3 now supports CPU dense FFN workers. For a two-layer dense container:

```bash
larql-server model.vindex3 --ffn-only --layers 0-0 --port 9181
larql-server model.vindex3 --ffn-only --layers 1-1 --port 9182
larql run model.vindex3 "Hello" \
  --v3-ffn-shards http://localhost:9181,http://localhost:9182
```

The coordinator owns embedding, attention, row KV, pre/post-FFN norms,
residual updates and the head. Workers prepare only dense FFN matrices and
return the contribution for one already-normalized row. The existing batch
and decode interpreters call the same operation provider; prefill dispatches
rows individually. The worker has no KV and receives no prefix history.

`GET /v1/vindex3/ffn` advertises a binding; `POST` checks that binding plus the
layer and finite input width. Bindings include declared artifact/plan identity,
CPU lowering revision, layer range, dimensions and effective operand
representations/realizations. The coordinator checks complete non-overlapping
ownership and compares preparation decisions before loading its local operands.
Use the same build on all nodes: realization descriptors are build-specific.
Artifact identity names declared payload hashes, not verification of bytes.

The private single-model server profile exposes the route under existing
authentication. `--v3-shard-token-env ENV` supplies a bearer token to the CLI.
`/v1/runtime` reports `dense_ffn_shard`; whole-model generation from a worker
refuses. A missing or malformed contribution aborts generation. A failed
`DenseFfnSession` is invalidated, retaining its last committed position; recover
by creating a fresh session and replaying committed input. Its partially advanced
KV is privately owned and cannot be reused through that API.

Scope: single-stream softmax stacks with a dense FFN on every layer, CPU
production lowering, base artifact operands and local row KV. `--ffn-only`
requires an explicit inclusive `--layers` range. Metal, other KV selectors,
layer-worker composition, overlays, MoE, KDA/MLA and discovery are refused or
outside this interface. All nodes still open the container; this does not package
or distribute shard files. No speed or memory-RSS claim is made.

[MoeExpertBackend](../../crates/larql-inference/src/ffn/moe_backend.rs) already
unifies in-process, remote and bound expert routes. However, its caller still
supplies `ModelWeights`; [BoundMoeBackend](../../crates/larql-inference/src/ffn/moe_bound.rs)
binds mapped operands through that bridge. This is useful migration evidence,
but a V3 provider must execute the container's operation plan without rebuilding
the V2 model interface.

Validation covers local and loopback-HTTP parity, including layer outputs,
logits, tokenwise prefill and decode beyond a sliding window; independently
observed operand reads against predicted residency; worker repeatability;
coverage, identity, representation and shape refusals; and mid-step failure
followed by fresh-session recovery. These are synthetic fixture checks, not a
model-backed K3 or heterogeneous-hardware result.

Routed experts then extend the same contract with bank/expert coordinates,
selected IDs and weights, parallel dispatch and an explicit reduction order.
Shared experts and post-reduction normalization must each execute exactly once.
Heterogeneous numerical providers need their own parity gates; an advertised
representation or backend name alone is not evidence of equivalence.

This would let KDA/MLA continuation remain local while remote machines hold
expert populations. K3 support, mixed CPU/Metal/CUDA placement, capability
discovery and fleet scheduling are subsequent gates, not capabilities conferred
by #507. The dense operation boundary is implemented here; full K3 placement remains
a separate integration.
