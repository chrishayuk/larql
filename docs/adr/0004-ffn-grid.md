# ADR-0004 — Self-Assembling Distributed FFN Grid

**Status:** Accepted — All five phases shipped. Phase 1 (Mode A announce + auth + multi-router) shipped 2026-04-19; Phase 2 (Mode B available pool + AssignMsg + ReadyMsg) shipped 2026-05-13 per ADR-0011; Phase 3 stale-heartbeat eviction shipped 2026-05-15; Phase 4 replication + hot-shard load-rate elevation shipped 2026-05-13/15; Phase 5 admin CLI (`status` / `gaps` / `drain` / `assign`) shipped 2026-05-15. Active-probe RTT routing shipped 2026-05-16 (was P2). See `crates/larql-router/ROADMAP.md` for the per-feature shipping notes.  
**Supersedes:** ADR-0003 §3 (static --shards configuration)  
**Depends on:** ADR-0003 (larql-router base)

---

## Problem

The current router requires static configuration at startup (`--shards`).
Adding or removing a server requires restarting the router with a new flag.
The grid cannot adapt to servers joining or leaving.

---

## Core Idea

Servers are autonomous. They connect to the router and declare what they
can do. The router maintains a coverage matrix and routes requests. No
central provisioning. No static configuration. The grid self-assembles.

```
Servers join  →  Router updates coverage matrix  →  Requests route to grid
Servers leave →  Router marks gap                →  Other servers fill it
```

---

## Two Modes of Operation

### Mode A — Announce

Server has a vindex shard already loaded. It connects to the router and
says: "I have this model, these layers, I am ready."

Router adds it to the coverage matrix immediately. No assignment needed.

```bash
$ larql-server output/gemma4-31b-q4k.vindex \
    --ffn-only \
    --layers 0-20 \
    --join grpc://router:50052
```

```
[server] Connecting to router grpc://router:50052
[server] Announcing: gemma4-31b-q4k layers=0-20 ram=11.2GB
[server] Registered. Serving.
```

### Mode B — Available

Server starts with no shard loaded. It connects to the router and says:
"I have capacity. What do you need?"

Router checks the coverage matrix for gaps, picks the most urgent
uncovered layer range, and assigns it. Server downloads and loads the
shard, signals ready.

```bash
$ larql-server \
    --join grpc://router:50052 \
    --available-ram 16GB \
    --vindex-store /mnt/shards/
```

```
[server] Connecting to router grpc://router:50052
[server] Advertising: ram=16GB store=/mnt/shards/
[router] Assigning: gemma4-31b-q4k layers=21-41 from http://origin:8090
[server] Downloading shard... done (11.1GB)
[server] Loaded. Ready.
[server] Registered. Serving.
```

---

## Proto Definition

```protobuf
syntax = "proto3";
package larql.grid.v1;

// ── Server → Router registration stream ──────────────────────────────────

service GridService {
  // Persistent bidirectional stream.
  // Server connects and keeps the stream open for its lifetime.
  // Router sends assignments and control messages.
  // Server sends heartbeats and status updates.
  rpc Join(stream ServerMessage) returns (stream RouterMessage);

  // Read-only grid status (admin / monitoring)
  rpc Status(StatusRequest) returns (StatusResponse);
}

// ── Server → Router ───────────────────────────────────────────────────────

message ServerMessage {
  oneof payload {
    AnnounceMsg  announce  = 1;  // "I have this shard loaded"
    AvailableMsg available = 2;  // "I have capacity, give me work"
    ReadyMsg     ready     = 3;  // "I finished loading an assigned shard"
    HeartbeatMsg heartbeat = 4;  // "I am still alive"
    DroppingMsg  dropping  = 5;  // "I am about to drop this shard"
  }
}

message AnnounceMsg {
  string model_id     = 1;  // "gemma4-31b-q4k"
  uint32 layer_start  = 2;  // inclusive
  uint32 layer_end    = 3;  // inclusive
  uint64 ram_bytes    = 4;  // resident RAM for this shard
  string listen_url   = 5;  // "http://server-a:8080" — where client should send requests
  string vindex_hash  = 6;  // sha256 of vindex content — router verifies compatible shards
}

message AvailableMsg {
  uint64 ram_bytes    = 1;  // available RAM
  uint64 disk_bytes   = 2;  // available disk in vindex store
  string store_path   = 3;  // local path where router can tell it to write shards
}

message ReadyMsg {
  string model_id     = 1;
  uint32 layer_start  = 2;
  uint32 layer_end    = 3;
  string listen_url   = 4;
}

message HeartbeatMsg {
  float  cpu_pct             = 1;
  uint64 ram_used            = 2;
  uint32 requests_in_flight  = 3;
}

message DroppingMsg {
  string model_id     = 1;
  uint32 layer_start  = 2;
  uint32 layer_end    = 3;
  string reason       = 4;  // "shutdown" | "reassigned" | "oom"
}

message RefuseMsg {
  string model_id     = 1;
  uint32 layer_start  = 2;
  uint32 layer_end    = 3;
  string reason       = 4;  // "insufficient_disk" | "wrong_arch" | "busy"
}

// ── Router → Server ───────────────────────────────────────────────────────

message RouterMessage {
  oneof payload {
    AssignMsg    assign    = 1;  // "load this shard"
    UnassignMsg  unassign  = 2;  // "drop this shard, you are redundant"
    AckMsg       ack       = 3;  // "registration accepted"
    RejectMsg    reject    = 4;  // "registration rejected"
  }
}

message AssignMsg {
  string model_id     = 1;  // "gemma4-31b-q4k"
  uint32 layer_start  = 2;
  uint32 layer_end    = 3;
  string origin_url   = 4;  // where to download the shard from
  string shard_hash   = 5;  // sha256 of expected shard file (integrity check)
}

message UnassignMsg {
  string model_id     = 1;
  uint32 layer_start  = 2;
  uint32 layer_end    = 3;
  string reason       = 4;  // "redundant" | "rebalancing"
}

message AckMsg {
  string server_id    = 1;  // router-assigned stable ID for this connection
}

message RejectMsg {
  string reason       = 1;  // "model not recognised" | "layer range conflict"
}

// ── Status ────────────────────────────────────────────────────────────────

message StatusRequest {}

message StatusResponse {
  repeated ModelCoverage models  = 1;
  repeated ServerInfo    servers = 2;
}

message ModelCoverage {
  string         model_id   = 1;
  uint32         num_layers = 2;
  repeated Shard shards     = 3;
  repeated Gap   gaps       = 4;  // layer ranges with no coverage
}

message Shard {
  uint32          layer_start   = 1;
  uint32          layer_end     = 2;
  repeated string server_ids    = 3;  // may be >1 if replicated
  uint32          replica_count = 4;
}

message Gap {
  uint32 layer_start = 1;
  uint32 layer_end   = 2;
}

message ServerInfo {
  string server_id          = 1;
  string listen_url         = 2;
  string state              = 3;  // "announcing" | "available" | "loading" | "serving" | "draining"
  string model_id           = 4;  // empty if available
  uint32 layer_start        = 5;
  uint32 layer_end          = 6;
  float  cpu_pct            = 7;
  uint64 ram_used           = 8;
  uint32 requests_in_flight = 9;
  uint32 rtt_ms             = 10;
}
```

---

## Coverage Matrix

The router maintains a coverage matrix per model. Rows are layer ranges.
Columns are servers. Each cell is the set of servers covering that range.

```
Model: gemma4-31b-q4k (60 layers)

Layer Range   Servers          Replicas   State
──────────────────────────────────────────────────
0  – 19      [server-a]        1          OK
20 – 39      [server-b]        1          OK
40 – 59      [server-c]        1          OK
```

After server-b joins with a second copy of layers 0-19:

```
Layer Range   Servers                 Replicas   State
───────────────────────────────────────────────────────
0  – 19      [server-a, server-b]     2          OK (replicated)
20 – 39      []                        0          GAP ← urgent
40 – 59      [server-c]               1          OK
```

The router detects the gap and assigns it to the next available server
that joins, or requests that an existing server with spare capacity
loads that range.

---

## Router Dispatch with Grid

When a client request arrives for layer N:

```rust
fn route_layer(&self, model_id: &str, layer: u32) -> Result<&str> {
    let servers = self.coverage.servers_for(model_id, layer);

    match servers.len() {
        0 => Err(GridError::Gap { model_id, layer }),
        1 => Ok(&servers[0].listen_url),
        _ => {
            // Multiple replicas — pick least loaded
            Ok(servers
                .iter()
                .min_by_key(|s| s.requests_in_flight)
                .unwrap()
                .listen_url
                .as_str())
        }
    }
}
```

---

## Gap Detection and Assignment

The router runs a background task that scans the coverage matrix for gaps.
When a gap is found:

1. Check the queue of available servers (Mode B servers waiting for work)
2. If an available server has enough RAM and disk, send it an `AssignMsg`
3. If no available servers, log a warning and continue serving covered layers

```rust
async fn gap_monitor(&self) {
    loop {
        sleep(Duration::from_secs(5)).await;

        for (model_id, matrix) in &self.coverage {
            for gap in matrix.gaps() {
                if let Some(server) = self.available_servers
                    .iter()
                    .find(|s| s.can_fit(gap.shard_size_bytes))
                {
                    self.assign(server, model_id, gap.layer_start, gap.layer_end).await;
                } else {
                    warn!("Gap in {model_id} layers {}-{}: no available server",
                          gap.layer_start, gap.layer_end);
                }
            }
        }
    }
}
```

---

## Rebalancing

Rebalancing serves two purposes: coverage (fill gaps) and load (replicate
hot shards). The router rebalances conservatively — it never removes the
last copy of a shard.

### Gap filling (coverage)

A gap exists when a layer range has zero servers. Gap filling is always
triggered immediately. The router assigns the gap to the first available
server.

### Replica management

A shard is under-replicated if it has fewer replicas than the configured
minimum (default: 1). When an available server joins, the router
preferentially assigns under-replicated shards.

A shard is over-replicated if it has more replicas than the maximum
(default: 3). The router sends `UnassignMsg` to the least loaded replica
to free capacity.

### Load-based replication

The router tracks request rate per layer range via the heartbeat stream.
If a layer range's request rate exceeds a threshold (configurable), the
router treats it as under-replicated regardless of the replica count.

```rust
fn replication_priority(&self, shard: &Shard) -> u32 {
    let replica_deficit = self.config.min_replicas
        .saturating_sub(shard.replica_count);

    let load_pressure = if shard.request_rate > self.config.hot_shard_threshold {
        1
    } else {
        0
    };

    replica_deficit + load_pressure
}
```

---

## Server Lifecycle

```
              ┌─────────┐
   startup    │         │   announce / available sent
  ──────────► │ joining │ ─────────────────────────────────┐
              │         │                                   │
              └─────────┘                                   ▼
                                                     ┌────────────┐
              ┌─────────┐   assignment received      │            │
              │         │ ◄─────────────────────────  │  serving   │
              │ loading │                             │  (announce)│
              │         │   ready sent                │            │
              └────┬────┘ ──────────────────────────► └────────────┘
                   │                                        │
                   ▼                                        │ unassign received
              ┌─────────┐                                   │ or shutdown
              │ serving │                                   ▼
              │(assigned│                            ┌────────────┐
              │  shard) │                            │  draining  │
              └─────────┘                            │            │
                                                     └─────┬──────┘
                                                           │ in-flight complete
                                                           ▼
                                                      disconnects
```

---

## Admin API

```bash
# Grid status
$ larql-router status

Model: gemma4-31b-q4k (60 layers)
  Layers  0–19:  server-a (11.2GB, 4 req/s)
  Layers 20–39:  server-b (11.1GB, 3 req/s)
  Layers 40–59:  server-c (11.3GB, 4 req/s)
  Coverage: 100%  Replicas: 1×  Gaps: none

Servers:
  server-a  http://192.168.1.10:8080  serving  gemma4-31b-q4k[0-19]   cpu=12%  ram=11.2GB
  server-b  http://192.168.1.11:8080  serving  gemma4-31b-q4k[20-39]  cpu=10%  ram=11.1GB
  server-c  http://192.168.1.12:8080  serving  gemma4-31b-q4k[40-59]  cpu=13%  ram=11.3GB
  server-d  http://192.168.1.13:8080  available  ram=16GB  waiting for assignment
```

```bash
# Force reassignment of a layer range
$ larql-router assign \
    --model gemma4-31b-q4k \
    --layers 20-39 \
    --server server-d

# Drain a server (graceful removal)
$ larql-router drain --server server-b

# Show gaps
$ larql-router gaps --model gemma4-31b-q4k
```

---

## Demo Sequence

```bash
# Terminal 1: Start the router (nothing configured — just listening)
$ larql-router start --port 50051 --admin-port 9090

  LARQL Grid Router v0.4.1
  Grid:    empty
  Listening: grpc://0.0.0.0:50051
  Admin:     http://0.0.0.0:9090
  Ready. Waiting for servers to join.
```

```bash
# Terminal 2: First server joins — announces layers 0-19
$ larql-server output/gemma4-31b-q4k.vindex \
    --ffn-only --layers 0-19 \
    --join grpc://localhost:50051

  [server-a] Connected to router
  [server-a] Announcing: gemma4-31b-q4k layers=0-19
  [server-a] Registered. Serving.

# Router output:
  [router] server-a joined: gemma4-31b-q4k layers=0-19  ✓
  [router] Coverage: 33%  Gaps: 20-39, 40-59
```

```bash
# Terminal 3: Second server joins — announces layers 20-39
$ larql-server output/gemma4-31b-q4k.vindex \
    --ffn-only --layers 20-39 \
    --join grpc://localhost:50051

  [server-b] Registered. Serving.

# Router output:
  [router] server-b joined: gemma4-31b-q4k layers=20-39  ✓
  [router] Coverage: 67%  Gaps: 40-59
```

```bash
# Terminal 4: Third server joins — available, no shard loaded
$ larql-server \
    --join grpc://localhost:50051 \
    --available-ram 16GB \
    --vindex-store /mnt/shards/

  [server-c] Connected to router
  [server-c] Advertising: ram=16GB, available
  [router]   Assigning: gemma4-31b-q4k layers=40-59 from http://origin:8090
  [server-c] Downloading shard (11.1GB)...
  [server-c] Loaded. Ready.
  [server-c] Registered. Serving.

# Router output:
  [router] server-c joined: available, assigned gemma4-31b-q4k layers=40-59
  [router] Coverage: 100%  Gaps: none  ✓
```

```bash
# Client — unchanged
$ larql-cli predict \
    --model google/gemma-4-31B-it \
    --vindex output/gemma4-31b-q4k.vindex \
    --ffn grpc://localhost:50051 \
    --prompt "The capital of France is"

  Top-1: " Paris"  (0.801)
```

```bash
# Kill server-b mid-demo
^C  (in terminal 3)

# Router output:
  [router] server-b disconnected: gemma4-31b-q4k layers=20-39
  [router] Coverage: 67%  GAP: 20-39  ← urgent

# Another server joins to fill the gap
$ larql-server \
    --join grpc://localhost:50051 \
    --available-ram 16GB \
    --vindex-store /mnt/shards/

  [server-d] Advertising: ram=16GB, available
  [router]   Assigning: gemma4-31b-q4k layers=20-39 (gap fill)
  [server-d] Loading...
  [server-d] Registered. Serving.

# Router output:
  [router] Coverage: 100%  Gaps: none  ✓

# Client never noticed. Requests during the gap returned 503.
# Requests after recovery route to server-d automatically.
```

---

## Implementation Status

### Phase 1 — Registration stream ✅ DONE

**Crates:**
- `crates/larql-router-protocol` — shared proto + tonic codegen (`grid.proto`,
  `GridService`, all message types). Separate crate so neither server nor router
  depends on the other.
- `crates/larql-router/src/grid/` — `GridState` (mod.rs), `GridServiceImpl` (service.rs), routing (routing.rs), replication (replication.rs), hot-shard (hot_shard.rs), status surface (status.rs)
- `crates/larql-server/src/announce.rs` — background announce task

**What is implemented:**

- **Mode A announce**: servers connect to the router with `--join`, send
  `AnnounceMsg`, receive `AckMsg`. Router stores entry in `GridState`.
- **Persistent bidirectional stream**: server keeps the gRPC stream open for its
  lifetime. Sends `HeartbeatMsg` every 10 seconds. On stream close (crash or
  shutdown) the router immediately deregisters the server.
- **Reconnect with backoff**: announce task retries with exponential backoff
  (1s → 2s → 4s … cap 60s) on any connection error.
- **O(1) route cache**: `GridState` maintains two pre-built tables rebuilt on
  every topology change (join/leave only — heartbeats skip the rebuild):
  - `route_table: HashMap<(model_id, layer), Vec<server_id>>` — for named-model queries
  - `any_model_table: HashMap<layer, Vec<server_id>>` — for single-model grids
  - `route()` is O(1) table lookup + O(replicas) least-loaded scan
  - `route_all()` resolves an entire layer batch in one lock acquisition
- **Least-loaded replica selection**: among servers owning the same layer range,
  the one with the smallest `requests_in_flight` counter is chosen. Counter is
  updated by each heartbeat.
- **Multi-model routing**: `(model_id, layer)` key. `model_id` is optional in
  requests — `None` matches any model for single-model grids. Servers announce
  each loaded model separately.
- **Multiple routers (stateless fan-out)**: `--join` accepts a comma-separated
  list of router gRPC URLs. One announce stream is spawned per router per model.
  Each router holds an independent copy of grid state rebuilt from live streams.
  No coordination needed — state converges within one heartbeat interval (10s).
- **Grid authentication**: `--grid-key SECRET` (or `LARQL_GRID_KEY` env var) on
  both router and server. Router rejects `Join` streams with
  `Status::UNAUTHENTICATED` if the bearer token is wrong or absent. Server
  injects `Authorization: Bearer <key>` via a tonic interceptor on every outgoing
  RPC including reconnects.
- **Vindex identity hash**: `vindex_identity_hash(model_id, num_layers)` computed
  on the server (stable hash of model identity, not a cryptographic primitive).
  Sent in `AnnounceMsg.vindex_hash`. Router logs it on registration — mismatched
  model versions are immediately visible.
- **Static shard fallback**: grid takes priority; if no grid route is found for a
  layer, the handler falls through to the static `--shards` map. Both modes
  coexist.
- **Connection pool tuning**: reqwest client configured with `tcp_keepalive(30s)`,
  `pool_idle_timeout(90s)`, `pool_max_idle_per_host(16)` — avoids per-hop TCP
  handshake overhead.

**Measured latency (Gemma 3 4B, localhost):**
- Per-hop overhead: ~2.4 ms (routing + HTTP round-trip)
- Full 34-layer pass (serial): ~371 ms (34 × 10.9 ms)
- Shard size does not affect latency — `RemoteWalkBackend` calls one layer at a
  time regardless of shard size. Shard size only affects RSS.

**Typical launch:**

```bash
# Router — with auth, grid on port 50052
larql-router \
  --grid-port 50052 \
  --grid-key "$(cat /run/secrets/grid_key)" \
  --port 9090

# Server — announces to two routers (fan-out HA)
larql-server output/gemma3-4b-q4k.vindex \
  --ffn-only --layers 0-16 \
  --join "http://router-a:50052,http://router-b:50052" \
  --grid-key "$(cat /run/secrets/grid_key)" \
  --public-url "http://server-a:8080"
```

---

### Phase 2 — Available mode (pending)

- `AvailableMsg` handling — add server to available pool
- Gap monitor background task — scan matrix every 5s, assign from pool
- `AssignMsg` sent to available server; `RefuseMsg` handling tries next
- `ReadyMsg` handling — server moves from loading to serving

### Phase 3 — Heartbeat and health (partial — heartbeat implemented)

- ✅ `HeartbeatMsg` processing — updates `cpu_pct`, `ram_used`, `requests_in_flight`
- ✅ Dead server detection — stream disconnect triggers immediate deregister + table rebuild
- ✅ `requests_in_flight` used for load-aware replica selection
- ⬜ Stale heartbeat eviction — evict servers that haven't sent a heartbeat in >N seconds

### Phase 4 — Rebalancing (pending)

- Replica count tracking per shard
- Under-replication detection — assign additional servers
- Over-replication detection — send `UnassignMsg` to least loaded replica
- Load-based replication — hot shard threshold config

### Phase 5 — Admin CLI (pending)

- `larql-router status` — grid status table (gRPC `Status` RPC is implemented; CLI is not)
- `larql-router drain --server` — graceful server removal
- `larql-router assign` — force assignment of a layer range
- `larql-router gaps` — gap report per model

---

## Open Questions

1. **Shard origin.** In Mode B (available), the router sends an `origin_url`
   for the server to download from. What hosts this? Options: (a) one of the
   announcing servers exposes a `/v1/shard` download endpoint for its own
   shard, (b) a separate origin store (S3, HTTP static). For the demo,
   option (a) — announcing servers serve their shard for download. Add
   `GET /v1/shard` to `larql-server` in Phase 2.

2. **Partial coverage behaviour.** When a gap exists and a request arrives
   for that layer range: return 503 or degrade (skip the layer)? Current
   spec: 503 with `{"error": "gap in coverage: layers 20-39 have no server"}`.
   Degraded mode changes model outputs. Decision: 503 for now, degraded as
   an opt-in flag.

3. **Model identity.** How does the router know that two servers announcing
   different layer ranges of `gemma4-31b-q4k` are from the same vindex
   extract? `vindex_hash` in `AnnounceMsg` (added to proto above) lets the
   router verify compatibility before merging them into the same coverage
   slot.

4. **Assignment refusal.** A Mode B server may refuse an assignment (not
   enough disk, wrong arch). `RefuseMsg` (added to proto above) lets the
   server decline; the router tries the next available server.

5. **Multiple models.** The coverage matrix is per model. A single grid
   can serve multiple models simultaneously — server-a announces
   `gemma4-31b-q4k`, server-e announces `llama3-70b`. The router routes by
   `(model_id, layer)`. Client specifies model in the request:
   `{"model": "gemma4-31b-q4k", "layer": 5, ...}`.
