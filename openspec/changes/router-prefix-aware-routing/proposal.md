## Why

Today `larql-router::GridState::route` and the new
`route_for_capability` (shipped in `router-heterogeneous-shards`)
both pick the **least-loaded** shard among replicas. That's
correct for stateless work, but attention shards carry
session-bound KV-cache state — routing a continuation of a session
to a *different* shard means rebuilding the entire prefix from
scratch, paying full prefill latency.

SMG (LightSeek) reports a **23% TTFT reduction** at 8 replicas / 256
concurrency by routing each request to the shard whose KV-cache
already covers the longest prefix of the input. Their cache-routing
implementation is also **10–12× faster** than their prior
implementation.

Once we ship `attention-service-routes` and start running real
sessions on the GPU shard, this becomes the next-leverage routing
fix.

This is **not** on the critical path for the CUDA + RotorQuant
workstream — it's a topology optimisation we revisit once
`attention-service-routes` lands and we have real session data
flowing through the router.

## What Changes

- ADD `ServerEntry::cached_prefixes: BloomFilter<u64>` field — a
  Bloom filter of prefix hashes the shard currently has cached.
- ADD a heartbeat extension so shards periodically report their
  bloom filter (or a delta) to the router. Out of scope for this
  proposal: the wire format details, which fold into the
  `attention-service-routes` AnnounceMsg extension.
- ADD `GridState::route_for_prefix(model_id, layer, capability,
  prefix_hashes)` that picks the shard with the most matching
  prefix hashes, with `route_for_capability` as the fallback when
  no shard has any cached prefix.
- ADD a small bloom-filter helper crate (or inline impl) — a
  256-bit bloom with 4 hash positions per element gives ~1%
  false-positive rate at 64 cached prefixes, well under the 23%
  TTFT we're hunting.

This is non-breaking. The new method is additive; existing
`route` and `route_for_capability` keep working. Shards that don't
report bloom filters get an empty filter and fall through to the
load-balanced path.

## Capabilities

### New Capabilities

(none — implements scenarios on `router-grid`.)

### Modified Capabilities

- `router-grid`: adds a prefix-aware routing requirement with
  scenarios covering bloom-filter match, false-positive bounds,
  and fallback to load-balanced routing.

## Impact

- **Affected files**: `crates/larql-router/src/grid.rs` (~150
  lines incl. tests); a new `bloom.rs` or use of the existing
  `fastbloom`/`growable-bloom` crate.
- **Affected systems**: router-only, plus the shard heartbeat
  payload (proto extension folds into
  `attention-service-routes`).
- **Provenance**: derived from SMG's cache-aware routing pattern.
  We don't vendor their code; the algorithm is straightforward.
- **Out of scope**: per-token granular prefix routing (we route on
  significant-boundary hashes, e.g., chat-template prefix hash);
  prefix eviction protocol; cross-router synchronisation.
