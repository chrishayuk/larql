# ADR-0015 — ShardService.Query (Remote KNN Cache for FFN Layers)

**Status:** Accepted — shipped 2026-05-16 as Exp 53 promotion.
**Depends on:** ADR-0004 (self-assembling grid),
ADR-0009 (wire-format evolution), ADR-0010 (QUIC transport).
**Implementation:**
`crates/larql-router-protocol/proto/shard.proto`,
`crates/larql-server/src/shard_query.rs`,
`crates/larql-server/src/bootstrap.rs` (registration on
`--shard-query-tau`).

---

## Context

Experiment 53 (`experiments/53_sharded_vindex/`) explored replacing
the FFN compute for a layer with a remote KNN lookup: the server
holds a pre-compiled `(gate_input, mlp_output)` table at a layer, and
a client doing a forward pass queries the server for the closest
gate vector. On a hit (cosine ≥ tau) the server returns the
pre-computed MLP output and the client skips local FFN compute; on a
miss the client falls back to local FFN.

The Python prototype proved the mechanism but used a bespoke binary
TCP frame. To productionise this as a grid feature, three things
need to be true:

1. The wire surface needs to be stable and toolable (same gRPC stack
   the rest of the workspace already uses).
2. The server side has to share state with the inference path — a
   compiled fact added at runtime through `PatchedVindex` should be
   visible to the next `Query` immediately, with no separate cache
   sync.
3. It must coexist with the existing `VindexService` and
   `ExpertService` on the same listener; no separate port to manage.

---

## Decision

### Wire surface — `ShardService.Query` unary RPC

```proto
service ShardService {
  rpc Query(ShardQuery) returns (ShardResult);
}

message ShardQuery {
  uint32 layer_id     = 1;
  uint32 k            = 2;
  bytes  query_vec    = 3;   // f32 LE bytes, length = hidden × 4
  float  tau_override = 4;   // 0.0 = use server-configured tau
}

message ShardResult {
  bool  hit      = 1;
  bytes mlp_out  = 2;        // f32 LE bytes, hidden × 4 (empty on miss)
  float best_sim = 3;        // reported on hit AND miss for telemetry
}
```

### Wire-byte convention: raw f32 little-endian in `bytes`

Hidden-sized vectors (Gemma 3 4B: 2560 floats = 10 KiB each) would
pay ~30% overhead as proto `repeated float` varints. Using `bytes`
with **raw IEEE-754 f32 LE** matches what `ExpertService` already
does for residual transport (ADR-0006 / ADR-0009) and lets the
server `decode_f32_le` straight into a `Vec<f32>` without proto-level
allocation per element.

### `tau_override = 0.0` means "use server config"

A per-call `tau_override` lets the client A/B different thresholds
without restarting the server. `0.0` is the documented "ignore this
field" sentinel — the server falls back to its CLI-configured tau.
Negative tau is also treated as the sentinel (a cosine similarity
below 0 means orthogonal-or-opposite, never a "real" match).

### Two backends behind a `ShardSource` enum

```rust
pub enum ShardSource {
    Vindex(Arc<RwLock<PatchedVindex>>, f32),    // production
    Cache(Arc<RwLock<ShardCache>>),             // tests / fixtures
}
```

- **`ShardSource::Vindex`** — production path. `Query` walks the
  server's loaded `PatchedVindex` via the existing `gate_knn` →
  `ffn_row_into(layer, FFN_COMPONENT_DOWN, feat, &mut out)`
  accessors. "Compiled facts" live as vindex patches
  (`insert_feature` + `set_down_vector`); no separate on-disk cache
  format is needed.
- **`ShardSource::Cache`** — test fixture. Tiny in-memory
  `HashMap<u32, LayerEntry>` with `insert_layer` + `seed_from_normed`.
  Lets unit + integration tests cover the wire path without
  spinning up a full vindex.

Enum dispatch, no `async-trait`. Both arms return the same
`ShardLookup { hit, mlp, best_sim }` struct, so the RPC handler is
identical for both.

### Live-patch propagation via shared Arc

`LoadedModel.patched: Arc<RwLock<PatchedVindex>>` (was
`RwLock<PatchedVindex>`). The Arc handle held by `ShardSource::Vindex`
is the **same** handle as the one held by the inference forward
path. A patch inserted from one path is immediately visible to the
other:

- 12 `Arc::new` wrapping sites at construction
- Every existing `.read().await` / `.write().await` call site is
  preserved by Rust's Deref coercion through `Arc<RwLock<T>>`
- An integration test (`live_patch_propagation` in
  `crates/larql-server/tests/test_shard_query.rs`) covers this: a
  patch added through one Arc handle is observable on the next
  `Query` through another handle.

### Algorithm: cosine + tau + weighted top-k

The KNN step inside `knn_lookup` (and the `Vindex` arm's equivalent
`ffn_row_into` loop) follows the Python prototype exactly:

1. L2-normalize the query and (already-cached) input rows.
2. Compute cosine similarities by dot product across all rows for
   the requested `layer_id`.
3. **Tau gate**: if `max(sims) < tau`, return `hit = false`,
   `mlp_out = []`. Best-sim is still reported.
4. **k = 1 fast path**: return the single argmax's `mlp` row.
5. **k > 1**: take the top `k` rows by similarity, compute a
   **cosine-weighted average** of their outputs:
   `out = sum(sim_i * mlp_i for i in top_k) / sum(sim_i for i in top_k)`.
   Only **positive** sims contribute weight (negative-cosine rows
   are clamped to 0 weight to avoid pulling the answer away).

### Registration on `--shard-query-tau`

Opt-in via the server CLI: `--shard-query-tau <TAU>` (defaults to
the feature being disabled, no `ShardService` registered). When
present alongside `--grpc-port`, the server adds
`ShardServiceServer` to the existing tonic builder chain next to
`VindexServiceServer` + `ExpertServiceServer`. Coexists on the same
listener; clients reach the service via `ShardServiceClient` against
the server's `--grpc-port`.

---

## Alternatives Considered

### Bespoke binary TCP frame (the Python prototype's wire)

Rejected — would require a new listener, new client library, new
auth story. Reusing tonic/gRPC means the service rides the same QUIC
connection as `GridService.Join` when `--features quic` is enabled,
inherits the workspace's existing TLS/auth, and is callable from
tooling that already speaks gRPC.

### Separate cache file format on disk

Rejected — the patch system in `PatchedVindex` already handles
"compiled (input, output) pairs at a specific layer." Adding a
second on-disk format would mean two write paths and two sync stories
(patch-to-cache, cache-to-patch). The `ShardSource::Vindex` arm
treats `PatchedVindex` as the source of truth and runs the KNN
on-line; no cache materialisation needed.

### Trait object instead of enum dispatch

`Box<dyn ShardSourceTrait>` was considered for symmetry with other
async backends in the workspace. Rejected because:
- Only two concrete backends exist and one (`Cache`) is test-only.
- Async-trait would force `Pin<Box<dyn Future>>` allocations on the
  hot RPC path.
- The enum is 3 lines longer; dispatch overhead is one branch.

### `repeated float` proto encoding for `query_vec` / `mlp_out`

Rejected — measurable ~30% overhead at hidden=2560. The `bytes`
convention matches `ExpertService` (ADR-0006/0009) so the codec
choice is consistent across the wire surface.

---

## Consequences

### Positive

- The KNN cache is queryable over the same wire as everything else
  in the grid. One TLS context, one auth path, one observability
  story.
- Runtime-added patches are immediately query-able with no separate
  sync — operators (or future automation) can `insert_feature` from
  one code path and the next `Query` sees it.
- The test fixture (`Cache` backend) means the wire/serde/algorithm
  paths can be unit-tested without standing up a full
  `PatchedVindex` — coverage on `shard_query.rs` is 96.78%.
- Coexists with `VindexService` / `ExpertService` on the same
  `--grpc-port`; no port multiplexing required by the operator.

### Negative

- The `tau` threshold is currently per-server, not per-layer. A
  single vindex with mixed cache densities across layers can't tune
  tau independently — clients work around this with `tau_override`.
  Future ADR if usage shows per-layer tau is needed.
- `query_vec` decoding allocates a `Vec<f32>` per RPC. For
  hidden=2560 that's a 10 KiB allocation per request. Acceptable at
  the current scale (10s of RPS per shard); if it shows up in a
  flame graph we'd switch to a pooled buffer.
- The KNN scan is O(n_entries × hidden) per query. The cache size at
  which this becomes the bottleneck depends on the deployment; today
  it's small (thousands of entries). At million-scale we'd need an
  ANN index (HNSW / IVF) and that's a separate ADR.

---

## Implementation pointers

| File | Role |
|---|---|
| `crates/larql-router-protocol/proto/shard.proto` | The wire contract |
| `crates/larql-router-protocol/src/lib.rs` (re-exports) | `ShardService`, `ShardQuery`, `ShardResult`, `ShardServiceServer`, `ShardServiceClient` |
| `crates/larql-server/src/shard_query.rs` | `ShardCache`, `ShardSource`, `knn_lookup`, `l2_normalize`, `cosine_similarities`, `weighted_topk_average`, `decode_f32_le`, `encode_f32_le`, `ShardServiceImpl` |
| `crates/larql-server/src/bootstrap.rs` | `--shard-query-tau` CLI plumbing + tonic registration |
| `crates/larql-server/src/state.rs` | `LoadedModel.patched: Arc<RwLock<PatchedVindex>>` (the shared Arc) |
| `crates/larql-server/tests/test_shard_query.rs` | 4 round-trip integration tests over real TCP (hit / below-tau miss / unknown-layer / live-patch propagation) |

### Test coverage

- 30 unit tests in `shard_query.rs::tests` (codec, cosine, top-k,
  cache ops, ShardSource dispatch, l2_normalize edge cases).
- 4 integration tests in `tests/test_shard_query.rs` over a real
  TCP socket.

`shard_query.rs` line coverage 96.78% as of 2026-05-16.
