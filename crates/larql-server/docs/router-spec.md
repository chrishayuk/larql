# larql-router — Layer-Sharding FFN Proxy

**Version:** 0.2  
**Status:** Implemented (static sharding + self-assembling grid, Phase 1)  
**Implementation:** `crates/larql-router`, `crates/larql-router-protocol`  
**ADR:** `docs/adr/0003-ffn-router.md`, `docs/adr/0004-ffn-grid.md`  
**See also:** `docs/specs/vindex-server-spec.md §13`, `docs/ffn/distributed.md`

---

## 1. Purpose

`larql-router` is a transparent HTTP proxy between a `RemoteWalkBackend` client and
a set of layer-sharded `larql-server` instances. Two dispatch modes are supported:

**Static mode** (`--shards`): fixed layer→URL map configured at startup.  
**Grid mode** (`--grid-port`): self-assembling — servers connect to the router at
runtime and announce their capabilities. No static configuration needed.

Both modes can coexist. Grid takes priority; static shards are the fallback.

```
Client  (--ffn-remote http://router:9090)
  │
  ▼
larql-router:9090
  │  layer 0–16   →  server-a:8080   (grid or static)
  │  layer 17–33  →  server-b:8081   (grid or static)
```

The client has no knowledge of shard topology. `RemoteWalkBackend` is unchanged.

---

## 2. Quickstart

### Static mode

```bash
larql-server output/gemma3-4b-q4k.vindex --ffn-only --layers 0-16  --port 8080
larql-server output/gemma3-4b-q4k.vindex --ffn-only --layers 17-33 --port 8081

larql-router \
  --shards "0-16=http://127.0.0.1:8080,17-33=http://127.0.0.1:8081" \
  --port 9090
```

### Grid mode (self-assembling)

```bash
# Router — grid gRPC on 50052, HTTP on 9090
larql-router --grid-port 50052 --port 9090

# Servers connect to the router on startup, no router restart needed
larql-server output/gemma3-4b-q4k.vindex \
  --ffn-only --layers 0-16 \
  --join "http://127.0.0.1:50052" \
  --public-url "http://127.0.0.1:8080" \
  --port 8080

larql-server output/gemma3-4b-q4k.vindex \
  --ffn-only --layers 17-33 \
  --join "http://127.0.0.1:50052" \
  --public-url "http://127.0.0.1:8081" \
  --port 8081
```

### Grid mode with auth (production)

```bash
larql-router \
  --grid-port 50052 \
  --grid-key "$(cat /run/secrets/grid_key)" \
  --port 9090

larql-server output/gemma3-4b-q4k.vindex \
  --ffn-only --layers 0-16 \
  --join "http://router:50052" \
  --grid-key "$(cat /run/secrets/grid_key)" \
  --public-url "http://server-a:8080"
```

---

## 3. CLI Reference

### larql-router

| Flag | Default | Description |
|------|---------|-------------|
| `--shards <SPEC>` | — | Static shard map: `"START-END=URL[,...]"`. Optional when `--grid-port` is set. |
| `--grid-port <PORT>` | — | Enable self-assembling grid gRPC server on this port. |
| `--grid-key <SECRET>` | — | Shared secret. Servers must present the same key. Also read from `LARQL_GRID_KEY` env var. If not set, the grid is open (dev only). |
| `--port <PORT>` | 9090 | HTTP listen port. |
| `--host <ADDR>` | 0.0.0.0 | Bind address. |
| `--timeout-secs <N>` | 120 | Per-request timeout to backend shards. |
| `--target-replicas <N>` | 1 | Phase 4 replication target per shard range. `>1` enables auto-replication from the available pool. |
| `--rebalance-interval <SECS>` | 30 | GT6 rebalancer tick cadence. `0` disables dynamic rebalancing entirely. |
| `--rebalance-threshold <RATIO>` | 2.0 | GT6 latency-imbalance trigger; the slowest replica must be this many times slower than the fastest. |
| `--hot-shard-rps <FRAC>` | — | Hot-shard load-rate replication: a shard whose max `req_per_sec` across replicas exceeds this value is treated as effectively under-replicated (`target + 1`) until the rate subsides. Unset disables the check. |
| `--hot-shard-demote-ratio <FRAC>` | 0.8 | ADR-0014 hysteresis. An elevated shard demotes only when its rate falls below `ratio × --hot-shard-rps`. Setting to `1.0` disables hysteresis (single-threshold mode). Values outside `(0.0, 1.0]` clamp to the default. |
| `--rtt-probe-interval-secs <N>` | 0 | Active-probe RTT cadence. When `>0`, the router periodically `GET`s `{listen_url}/v1/health` on every serving server and uses the recorded round-trip as a tie-breaker after GT3 per-layer latency in `route()`. `0` disables probing (default — GT3 already subsumes RTT in steady state). |
| `--log-level <LEVEL>` | info | Tracing log level. |

QUIC flags (requires `--features quic` at build time):

| Flag | Default | Description |
|------|---------|-------------|
| `--quic-port <PORT>` | — | Enable QUIC grid listener on this port alongside the TCP `--grid-port`. Servers join via `quic://router:PORT`. |
| `--quic-cert <PEM>` | — | TLS certificate PEM. If omitted, the router auto-generates a self-signed cert and prints its SHA-256 fingerprint at startup. |
| `--quic-key <PEM>` | — | TLS private key PEM matching `--quic-cert`. |
| `--quic-server-name <NAME>` | `router` | TLS SNI embedded in the auto-generated self-signed cert. Clients must connect with this name. |

At least one of `--shards` or `--grid-port` must be provided.

### larql-server (grid-relevant flags)

| Flag | Description |
|------|-------------|
| `--join <URL[,URL,...]>` | Comma-separated gRPC addresses of routers to join. One announce stream per router per model. |
| `--public-url <URL>` | HTTP URL clients should use to reach this server. Required with `--join`. Defaults to `http://HOST:PORT`. |
| `--grid-key <SECRET>` | Shared secret matching the router's `--grid-key`. Also read from `LARQL_GRID_KEY`. |

---

## 4. Endpoints

### POST /v1/walk-ffn

Proxies the walk-ffn protocol. Accepts the same request body as `larql-server`.

**Single-layer request** — proxied unchanged to the owning shard.

```json
{"layer": 5, "residual": [...]}
→ {"layer": 5, "output": [...], "latency_ms": 10.9}
```

**Batched request** — layers grouped by owning shard, dispatched in parallel,
merged and sorted by layer. Wall-clock latency = `max(shard latencies)`.

```json
{"layers": [5, 20], "residual": [...]}
→ {"results": [{"layer": 5, ...}, {"layer": 20, ...}], "latency_ms": 11.2}
```

**Optional `model_id` field** — selects routing table for multi-model grids.
Omit for single-model deployments.

```json
{"layer": 5, "model_id": "gemma3-4b-q4k", "residual": [...]}
```

**MoE expert routing (ADR-0018) — single layer + experts:**

```json
{"layer": 5, "experts": [0, 3, 7], "residual": [...]}
→ {"results": [{"layer": 5, "expert": 0, ...}, ...], "latency_ms": 11.2}
```

The router fans out per `(layer, expert_id)` pair to the owning
expert-shard, then merges the responses. Requires a grid
(`--grid-port`); static `--shards` mode 503s on MoE bodies.

**MoE expert routing — multi-layer + experts:**

```json
{"layer_experts": [
   {"layer": 5, "experts": [0, 3]},
   {"layer": 6, "experts": [1, 5]}
 ], "residual": [...]}
```

The two shapes are mutually exclusive at parse time (`layer_experts`
takes priority). `layer` / `layers` (dense) and `experts` /
`layer_experts` (MoE) cannot be mixed in one request.

**Error responses:**

| Condition | HTTP | Body |
|-----------|------|------|
| Layer has no owning shard | 400 | `{"error": "layer N has no owning shard in this router"}` |
| `(layer, expert_id)` has no owning MoE shard | 503 | `{"error": "no shard owns (layer N, expert E) in this router"}` |
| MoE body against static-`--shards`-only router | 503 | `{"error": "MoE routing requires a self-assembling grid"}` |
| Neither `layer` nor `layers` nor MoE shape | 400 | `{"error": "must provide 'layer' or 'layers'"}` |
| `experts` array without `layer` | 400 | `{"error": "moe: 'experts' requires a 'layer' scalar"}` |
| Empty `layers` / `experts` / `layer_experts` | 400 | `{"error": "empty layer list"}` |
| Shard unreachable | 502 | `{"error": "shard http://...: ..."}` |
| Shard returns error | forwarded | Shard status + body passed through |

### GET /v1/health

```json
{"status": "ok"}
```

Always returns 200.

### GET /v1/shard/{model_id}/{start}-{end}

Served by every `larql-server` that owns the requested layer range.
Used by Mode B spares to download a vindex slice after they receive
an `AssignMsg` from the router (`AssignMsg.origin_url` points at any
live replica of the range; the spare appends
`/v1/shard/{model_id}/{start}-{end}` and fetches).

- **Response**: `200 OK`, `Content-Type: application/x-tar`,
  streaming body (chunked) containing the on-disk vindex directory
  packaged as a tar archive. Symlinks are followed during archive
  creation so the spare receives a self-contained slice.
- **Layer range semantics**: `start` and `end` are inclusive and
  validated (400 on `start > end` or malformed range). The server
  ships its full vindex directory in the tar; the receiver applies
  its own `--layers` filter on unpack — this keeps the streaming
  path simple and matches today's vindex layout.
- **Errors**: `400` on malformed range; `404` if `model_id` is not
  loaded; `500` on disk-side tar failures.

Client side: `larql-server` calls `shard_loader::download_and_load_shard()`
(in `crates/larql-server/src/shard_loader.rs`) on every `AssignMsg`.
The call is idempotent (skips download if the unpacked path already
exists), verifies SHA-256 against `AssignMsg.expected_hash` when set,
and unpacks atomically into `{store}/{model_id}/layers-{start}-{end}/`
via a temp-dir-then-rename. End-to-end integration test:
`crates/larql-server/tests/test_grid_mode_b.rs::mode_b_full_vertical_handoff`
spawns a real donor and exercises the download → unpack → ReadyMsg
path against the live HTTP endpoint.

---

## 5. Dispatch Logic

```
Receive POST /v1/walk-ffn

1. Parse layer(s) from body; extract optional model_id
2. resolve_all(model_id, layers) — one lock acquisition covers all layers:
     grid route_table O(1) lookup per layer → list of replica server_ids
     least-loaded replica selected (min requests_in_flight)
     fallback to static shards for any grid miss
     → Err(missing layer) → 400
3. Single layer → proxy body unchanged to owning shard URL → return
4. Multiple layers:
     group by shard URL
     build sub-request per group (single layer or layers array)
     tokio::spawn parallel dispatch to each group
     await all; merge results by layer; return max latency
```

The residual is not modified in transit.

---

## 6. Self-Assembling Grid (ADR-0004 Phase 1)

### How it works

Servers connect to the router's gRPC port on startup over a persistent bidirectional
stream (`GridService.Join`). The router assigns a stable `server_id` and returns an
`AckMsg`. The server then sends `HeartbeatMsg` every 10 seconds.

When the stream closes (crash or clean shutdown), the router immediately deregisters
the server and rebuilds the route table.

```
Server startup                   Router
  │                                │
  │── Join stream ──────────────►  │
  │── AnnounceMsg ──────────────►  │  register + rebuild route table
  │◄─ AckMsg ────────────────────  │
  │                                │
  │── HeartbeatMsg (every 10s) ──► │  update cpu/ram/requests_in_flight
  │                                │
  │  (stream closes on shutdown)   │  deregister + rebuild route table
```

### Route cache

`GridState` maintains two pre-built lookup tables, rebuilt only on topology
changes (join/leave). Heartbeats update metrics in place without a rebuild.

- `route_table[(model_id, layer)]` → `Vec<server_id>` — named-model queries
- `any_model_table[layer]` → `Vec<server_id>` — single-model grids (`model_id` omitted)

`route()` is an O(1) table lookup + O(replicas) scan for least-loaded selection.
`route_all()` resolves an entire layer batch under a single lock acquisition.

### Multiple routers

Pass a comma-separated list to `--join` to connect to multiple routers
simultaneously. Each router gets an independent announce stream. State converges
within one heartbeat interval (10s).

```bash
--join "http://router-a:50052,http://router-b:50052"
```

No coordination between routers is needed. Each router independently rebuilds its
route table from live streams. An L4 load balancer in front of the routers
provides HA for the HTTP path.

### Authentication

Set `--grid-key` (or `LARQL_GRID_KEY` env var) on both sides. The router enforces
it on every incoming `Join` RPC via an `Authorization: Bearer` gRPC metadata check.
The server injects the header on every outgoing RPC via a tonic interceptor,
including after reconnects.

If `--grid-key` is not set, the grid is open — appropriate for local development.

### Vindex identity hash

Each server computes `hash(model_id, num_layers)` and sends it in `AnnounceMsg`.
The router logs the hash on every registration. A server claiming to serve a
different model version is immediately visible in logs. This is not a cryptographic
check — the grid key provides authentication; the hash provides version visibility.

---

## 7. Deployment Examples

### Two-shard local (Gemma 3 4B, 34 layers)

```bash
larql-server model.vindex --ffn-only --layers 0-16  --port 8080
larql-server model.vindex --ffn-only --layers 17-33 --port 8081
larql-router --shards "0-16=http://127.0.0.1:8080,17-33=http://127.0.0.1:8081"
```

### Grid with 3× redundancy across 10 servers

```bash
larql-router --grid-port 50052 --grid-key "$KEY" --port 9090

# Range A: layers 0-10, 4 replicas
for port in 8080 8081 8082 8083; do
  larql-server model.vindex --ffn-only --layers 0-10 --port $port \
    --join "http://router:50052" --grid-key "$KEY" \
    --public-url "http://$(hostname):$port" &
done

# Range B: layers 11-22, 3 replicas
# Range C: layers 23-33, 3 replicas
```

### High-availability routers

```bash
# Two routers — servers connect to both
larql-router --grid-port 50052 --grid-key "$KEY" --port 9090  # router-a
larql-router --grid-port 50052 --grid-key "$KEY" --port 9090  # router-b

larql-server model.vindex --ffn-only --layers 0-16 \
  --join "http://router-a:50052,http://router-b:50052" \
  --grid-key "$KEY" \
  --public-url "http://server-a:8080"

# L4 LB in front of router-a and router-b handles HTTP HA
```

### Systemd service

```ini
[Unit]
Description=LARQL FFN Router
After=network.target

[Service]
ExecStart=/usr/local/bin/larql-router \
    --grid-port 50052 \
    --port 9090
Environment=LARQL_GRID_KEY=changeme
Restart=always

[Install]
WantedBy=multi-user.target
```

---

## 8. Binary Wire Format

The router transparently forwards the binary wire format (`Content-Type:
application/x-larql-ffn`) without parsing. Clients and servers that support
binary format can use it end-to-end with no router changes.

### Format summary

```
Single-layer request:
  [4: layer u32 LE][4: seq_len u32][4: flags u32 (bit0=full_output)][4: top_k u32]
  [residual f32[] LE]

Batch request:
  [4: BATCH_MARKER=0xFFFFFFFF][4: num_layers u32][K×4: layer u32[]LE]
  [4: seq_len u32][4: flags u32][4: top_k u32][residual f32[] LE]

Single-layer response:
  [4: layer u32 LE][4: seq_len u32][4: latency f32][output f32[] LE]

Batch response:
  [4: BATCH_MARKER][4: num_results u32][4: latency f32]
  Per result: [4: layer u32][4: seq_len u32][4: num_floats u32][output f32[] LE]
```

### Constraints

- Binary format requires `full_output = true`. Features-only binary requests
  are rejected by the server with HTTP 400.
- **Multi-shard binary fan-out is not supported.** The router cannot split a
  binary batch across shards. Use JSON for cross-shard batches or route
  shard-local batches directly. The single-shard case (all layers on one
  shard) is forwarded raw regardless of format.
- `model_id` is not encoded in the binary format; multi-model binary routing
  uses the grid's default routing for that layer.

### Performance

Measured on Gemma 3 4B (hidden_size=3072, seq_len=1):

| Format  | Request size | Shard latency (median) |
|---------|-------------|------------------------|
| JSON    | ~15.4 KB    | ~8.1 ms                |
| Binary  | ~10.3 KB    | ~7.6 ms                |

~33% smaller requests, ~0.5 ms/hop savings from eliminating JSON float
serialization.

---

## 9. Connection Pool

The reqwest client to backend shards is configured for low-latency reuse:

| Setting | Value |
|---------|-------|
| `tcp_keepalive` | 30 s |
| `pool_idle_timeout` | 90 s |
| `pool_max_idle_per_host` | 16 connections |
| Per-request timeout | `--timeout-secs` (default 120 s) |

Idle connections are kept alive to avoid TCP handshake overhead on each inference
hop.

---

## 10. What Is Not Yet Implemented

Tracked in ADR-0003 / ADR-0004:

- **gRPC transport to backends** — fan-out currently uses HTTP/JSON; a future version
  will use raw f32 bytes over gRPC (ADR-0003 Phase 2)

Already shipped (see `crates/larql-router/ROADMAP.md` for details):
Mode B + Phase B2 drain/reassign, stale heartbeat eviction, admin CLI
(`status` / `gaps` / `drain` / `assign`), dynamic rebalancing including
hot-shard load-rate replication + two-threshold hysteresis (ADR-0014),
QUIC transport (ADR-0010), Prometheus `/metrics` (ADR-0017), MoE
expert dispatch by `(layer, expert-range)` ownership with JSON
`experts` / `layer_experts` request shapes (ADR-0018), HTTP/3 shard
transport opt-in via `--http3-shards` / `--http3-port` (ADR-0019),
backpressure tier in `route()` with `--saturation-ceiling N` →
`503 Retry-After: 0.5` (ADR-0020), and the `GET /v1/shard/...`
endpoint documented above.

---

## 11. Validation

Local correctness checks:

```bash
cargo test -p larql-router
cargo test -p larql-server announce
```

The router test suite covers static shard parsing plus grid route-table behavior:
inclusive layer ownership, default single-model routing, least-loaded replica
selection from heartbeat load, deregistration on shard leave, first uncovered
layer reporting for batched requests, and status response shard/gap reporting.

`larql-server announce` covers the server side of the grid protocol envelope:
stable vindex identity hash, bearer metadata formatting, announce payloads,
heartbeats, and dropping notices.

---

## 12. Crate Structure

```
crates/larql-router-protocol/       shared proto types (router + server)
  proto/grid.proto                  GridService, all message types
  src/lib.rs                        re-exports GridServiceClient/Server

crates/larql-router/
  src/main.rs                       CLI, AppState, HTTP handler, static shards
  src/grid.rs                       GridState (route cache), GridServiceImpl
```

**Dependencies:** `axum`, `tokio`, `reqwest`, `tonic`, `serde_json`, `clap`,
`tracing`, `futures`, `larql-router-protocol`. No `larql-vindex` dependency —
the router carries no model knowledge.

---

## License

Apache-2.0
