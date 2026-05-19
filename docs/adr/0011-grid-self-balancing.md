# ADR-0011 — Grid Self-Balancing: Mode B + Dynamic Rebalancing

**Status:** Accepted — shipped 2026-05-13 (Mode B Phase B1) and 2026-05-15 (Phase B2 drain-then-reassign + replication + hot-shard load-rate elevation + stale-heartbeat eviction). Per-layer latency imbalance detection landed under GT6. Code lives in `crates/larql-router/src/grid/replication.rs` + `crates/larql-router/src/tasks/rebalancer/{hot_shard,replication,eviction,imbalance}.rs`.  
**Depends on:** ADR-0004 (Self-Assembling Grid)  
**Supersedes:** ADR-0004 §"Mode B — Available" (stub)

---

## Context

ADR-0004 defined two modes of grid operation. Mode A (server announces what
it has loaded) is implemented. Mode B (server advertises capacity, router
assigns a shard) was designed but not implemented — the router rejects
`AvailableMsg` with "Mode B — not yet implemented" and `AssignMsg` is never
sent.

Two operational gaps result:

1. **Elastic scaling is manual.** A new machine must have the vindex shard
   pre-loaded and the layer range hard-coded in the `--layers` flag. There
   is no way to say "I have 24 GB RAM, what do you need?" and have the
   router provision accordingly.

2. **Load imbalance is visible but unaddressed.** The router tracks
   `requests_in_flight` per server but has no mechanism to rebalance.
   A server that is CPU-saturated continues to receive requests at the same
   rate as an idle server; an overloaded shard cannot shed load.

---

## Decision

Implement Mode B in two sub-phases, each independently shippable.

### Phase B1 — Gap Fill (new server → router assigns shard)

A server starts with no shard loaded and advertises available capacity:

```bash
larql serve --join grpc://router:50052 \
            --available-ram 24GB \
            --vindex-store /mnt/shards/
```

Router receives `AvailableMsg`, checks coverage matrix for gaps, and sends
`AssignMsg` with the origin URL and expected shard hash. Server downloads,
verifies, loads, and sends `ReadyMsg`. Router registers normally.

### Phase B2 — Dynamic Rebalancing (router redistributes shards)

Router monitors load imbalance via per-layer latency from heartbeats (GT3).
When imbalance exceeds a threshold, router sends `UnassignMsg` to the
overloaded server, waits for `DroppingMsg`, then reassigns via `AssignMsg`
to an available server.

---

## State Additions to GridState

```rust
// new — servers in Mode B idle state, waiting for assignment
available_servers: HashMap<String, AvailableEntry>,

// new — in-flight assignments; key = server_id
pending_assignments: HashMap<String, AssignmentRecord>,

// configurable target; default 1; >1 = replicated shards
target_replicas: u32,
```

```rust
struct AvailableEntry {
    server_id: String,
    sender: mpsc::Sender<RouterMessage>,  // channel to send AssignMsg
    ram_bytes: u64,
    disk_bytes: u64,
    store_path: String,
    joined_at: Instant,
}

struct AssignmentRecord {
    server_id: String,
    model_id: String,
    layer_start: u32,
    layer_end: u32,
    assigned_at: Instant,
    timeout: Duration,  // default 10 minutes for download + load
}
```

---

## Phase B1 Protocol

```
Server                                     Router
  │  AvailableMsg{ram, disk, store_path}    │
  │──────────────────────────────────────►  │
  │                                         │  check_coverage_gaps()
  │                                         │  find_assignable_gap(ram_bytes)
  │                                         │  record pending_assignments[server_id]
  │  AssignMsg{model_id, layer_start,        │
  │            layer_end, origin_url,        │
  │            shard_hash}                  │
  │ ◄──────────────────────────────────────│
  │                                         │
  │  [download shard from origin_url]       │
  │  [verify sha256 == shard_hash]          │
  │  [load VectorIndex from store_path]     │
  │                                         │
  │  ReadyMsg{model_id, layer_start,        │
  │           layer_end, listen_url}        │
  │──────────────────────────────────────►  │
  │                                         │  register() → route table rebuild
  │  AckMsg{server_id}                      │
  │ ◄──────────────────────────────────────│
  │  [serving]                              │
```

If the server cannot accept the assignment (insufficient disk, wrong arch):
```
  │  RefuseMsg{reason="insufficient_disk"}  │
  │──────────────────────────────────────►  │
  │                                         │  remove from pending_assignments
  │                                         │  try next available_server
```

Assignment timeout: if `ReadyMsg` is not received within `assignment_timeout`
(default 10 min, configurable), router removes the pending record and
reassigns to the next available server.

---

## Phase B2 Protocol — Drain + Reassign

```
Router                                     Overloaded Server
  │  UnassignMsg{model_id, layer_start,    │
  │              layer_end,                │
  │              reason="rebalancing"}     │
  │──────────────────────────────────────►  │
  │                                         │  [stop accepting new requests for shard]
  │                                         │  [drain in-flight, max 30s]
  │                                         │  [unload shard weights from memory]
  │  DroppingMsg{reason="reassigned"}       │
  │ ◄──────────────────────────────────────│
  │                                         │
  │  [deregister server]                    │
  │  [mark server as available]             │
  │  [run Phase B1 gap-fill for freed gap]  │
```

The server enters the "available" state after dropping — it can immediately
receive a new `AssignMsg` for a different layer range.

---

## Rebalancing Trigger

The rebalancer runs as a background tokio task every 30 seconds:

```rust
async fn rebalancer_task(state: Arc<RwLock<GridState>>, cfg: RebalancerConfig) {
    loop {
        tokio::time::sleep(cfg.check_interval).await;  // default 30s
        let action = state.read().await.check_imbalance(&cfg);
        if let Some(RebalanceAction::Unassign { server_id, reason }) = action {
            state.write().await.initiate_unassign(server_id, reason);
        }
    }
}
```

`check_imbalance` triggers when both conditions hold for a sustained window
(default 60s):
- `max(avg_layer_latency_ms) / min(avg_layer_latency_ms) > 2.0` across
  servers covering the same layer range (i.e., replicated layers)
- At least one `AvailableEntry` exists that could receive a reassignment

For non-replicated shards (most current deployments), rebalancing is a no-op
— the router cannot shed load from a shard with no available replacement.
The trigger fires only when `target_replicas > 1` or a spare server is
available.

---

## HeartbeatMsg Extension (dependency: GT3)

Phase B2 requires per-layer latency in heartbeats to detect imbalance. This
is specified as GT3 in the implementation plan and requires a `grid.proto`
update:

```protobuf
message LayerLatency {
  uint32 layer   = 1;
  float  avg_ms  = 2;  // exponential moving average, α=0.1
  float  p99_ms  = 3;  // ring-buffer p99 over last 100 requests
}

message HeartbeatMsg {
  float  cpu_pct            = 1;
  uint64 ram_used           = 2;
  uint32 requests_in_flight = 3;
  repeated LayerLatency layer_stats = 4;  // NEW
}
```

The server collects layer latency in `routes/walk_ffn.rs` via a
`LayerLatencyTracker` (EMA + ring buffer, one per layer, lock-free via
`AtomicU64` for EMA and `Mutex<VecDeque>` for p99).

---

## Shard Download (server-side, new `shard_loader.rs`)

```rust
pub async fn download_shard(
    origin_url: &str,
    store_path: &Path,
    expected_hash: &str,
    progress: impl Fn(u64, u64),  // bytes_received, total_bytes
) -> Result<PathBuf>
```

- HTTP range requests (resumable): `Range: bytes=N-` if partial file exists.
- SHA-256 verification after download.
- Atomic rename: download to `store_path/.tmp-{hash}`, rename on success.
- Cancellation: tokio `CancellationToken` propagated from the announce task.

The `origin_url` in `AssignMsg` is the vindex HTTP origin (e.g.
`http://vindex-store:8090/gemma4-31b-q4k/layers-0-14.tar`). The server
exposes its vindex via `GET /v1/shard/{model_id}/{layer_start}-{layer_end}`
(new endpoint, ~50 LOC, byte-stream of the shard directory as a tar).

---

## Router CLI

```
larql-router
  --target-replicas N      default 1; >1 enables replication
  --rebalance-interval S   seconds between rebalancer checks (default 30)
  --rebalance-threshold X  latency ratio to trigger rebalance (default 2.0)
  --assignment-timeout S   seconds before pending assignment expires (default 600)
```

---

## Implementation Files

| File | Change |
|---|---|
| `crates/larql-router-protocol/proto/grid.proto` | ADD `LayerLatency`, extend `HeartbeatMsg` |
| `crates/larql-router/src/grid/` | `mod.rs` holds `available_servers`; `replication.rs` holds `try_assign_gap` / `try_replicate_from_available` / `send_assign_to_named_available`; `hot_shard.rs` holds `hot_layer_ranges` + `mark_elevated`. (Final shipped layout post-2026-05-16 split.) |
| `crates/larql-router/src/tasks/rebalancer/` | Background rebalancer task split into `mod.rs` (loop), `config.rs`, `hot_shard.rs`, `replication.rs`, `eviction.rs`, `imbalance.rs`. |
| `crates/larql-router/src/main.rs` | Spawn rebalancer task; add CLI flags |
| `crates/larql-server/src/announce.rs` | Handle `AssignMsg` → trigger `shard_loader`; handle `UnassignMsg` → drain + `DroppingMsg` |
| `crates/larql-server/src/shard_loader.rs` | NEW — HTTP range download, hash verify, atomic rename |
| `crates/larql-server/src/routes/walk_ffn.rs` | Collect per-layer latency via `LayerLatencyTracker` |
| `crates/larql-server/src/bootstrap.rs` | Accept `--available-ram`, `--vindex-store`; CLI for Mode B |

---

## Trade-offs

- **Mode B increases server complexity.** The server must manage shard
  lifecycle (download, verify, load, unload) in addition to serving.
  Mitigated by isolating this in `shard_loader.rs` with clear error
  boundaries.
- **Download failure handling.** If a shard download fails mid-way, the
  server sends `RefuseMsg` and the router tries the next available server.
  The partial download is cleaned up. The gap remains until another server
  fills it.
- **Drain window.** The 30-second drain window for Phase B2 means a
  rebalancing cycle costs at least 30 seconds of continued service at
  degraded routing quality. For interactive workloads this is acceptable;
  for batch workloads it is irrelevant.
- **No cross-model rebalancing.** A server that announces `model_id=gemma4`
  cannot be reassigned to serve `model_id=llama3`. Rebalancing is within
  the same model only.

---

## Open Questions

1. **Shard origin auth.** The `origin_url` may require credentials. For
   Fly.io deployments, the origin is another `larql serve` instance on the
   same private network (no auth needed). For S3 origins, a signed URL in
   the `AssignMsg` is the right mechanism. Defer to deployment-specific
   configuration.

2. **`target_replicas > 1`.** When the router assigns the same layer range
   to two servers, both get `AssignMsg`. The route table stores both;
   `route()` picks least-loaded. Failure of one does not cause a gap.
   Testing with replicated shards requires at least 3 servers; leave for
   G-SCALE validation.
