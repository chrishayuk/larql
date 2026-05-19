# Distributed FFN — Layer Sharding and Router

**Status:** Implemented (layer sharding + static router + self-assembling grid)  
**ADR:** docs/adr/0003-ffn-router.md, docs/adr/0004-ffn-grid.md  
**Full spec:** docs/specs/larql-router-spec.md

---

## Overview

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
interleaved_q4k.bin, etc.):** These are mmap'd as a whole — the virtual address
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

Wall-clock latency for a batched fan-out equals `max(shard_latencies)`, not the sum.

**Unknown layer**: request is rejected at the router with HTTP 400 before any shard
is contacted.

**Health check**: on startup the router calls `GET /v1/stats` on each configured
shard. Unreachable shards are logged as warnings; the router still starts. Requests
to an unreachable shard will return HTTP 502 with the upstream error.

### Implementation

| Location | What it does |
|---|---|
| `crates/larql-router/src/main.rs` | CLI, HTTP handler, static shard dispatch, `resolve_all` |
| `crates/larql-router/src/grid.rs` | `GridState` (O(1) route cache), `GridServiceImpl` (gRPC) |
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

See full option reference in `docs/specs/larql-router-spec.md §3`.

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

### Batched forward pass

`RemoteWalkBackend.forward_all_layers(layers, x)` sends all layers in a
single HTTP round trip (binary batch request). The router fans the batch
out to the owning shards in parallel. Wall-clock time = `max(shard
latencies)`.

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

## What Is Not Yet Implemented

- **Mode B (available)** — server starts empty, router assigns a shard (ADR-0004 Phase 2)
- **Admin CLI** — `larql-router status / drain / assign / gaps` (ADR-0004 Phase 5)
- **gRPC transport to backends** — currently HTTP/JSON; a future version uses raw f32 bytes over gRPC (ADR-0003 Phase 2)
- **MoE expert dispatch** — routing by expert ID (ADR-0003 Phase 3)
- **Router L2 cache** — router is the natural cache position but currently passes every request through (ADR-0003 Phase 4)
