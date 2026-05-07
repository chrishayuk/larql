## Context

`router-heterogeneous-shards` (shipped) gave us
`route_for_capability(model, layer, cap)` that picks a least-loaded
shard with the right capability bit set. For stateless RPCs (FFN
expert dispatch, embedding lookup) that's optimal.

Attention sessions are stateful: a multi-turn chat keeps reusing
the same KV-cache prefix. If the router sends turn N+1 to a
different shard than turn N, the new shard pays full prefill cost
to rebuild the cache. SMG's measurement: 23% TTFT regression vs
sticky routing.

The natural seam is the same `GridState`: add a per-shard
"what prefixes are cached here" structure and have routing
preferentially pick a shard that already has the request's prefix.

## Goals / Non-Goals

**Goals:**
- Route a continuation of a session to a shard that already has
  the prefix's KV-cache.
- Probabilistic match — false positives mean "we route to a shard
  that doesn't actually have the cache, paying one prefill"; this
  is recoverable. False negatives are also recoverable but cost
  more (we miss a sticky-route opportunity).
- Bloom filter overhead per shard ≤ 1 KB; per-route lookup ≤ 5 µs.

**Non-Goals:**
- Per-token granularity. We hash significant boundaries (the chat
  template prefix, post-system-prompt position, etc.) — not every
  token's running hash.
- Negative caching (knowing what a shard does NOT have).
- Eviction protocol. The shard already knows when it evicts;
  routing tolerates stale Bloom filters.
- Cross-router replication.

## Decisions

### D1 — Bloom filter not Cuckoo / radix tree

Three options:

1. **Bloom filter** (256 bits, 4 hash positions) — ~1 KB per shard,
   ~1% FP rate at 64 cached prefixes. Lookup: 4 hash + 4 bit checks.
2. **Cuckoo filter** — supports deletion, similar size. We don't
   need deletion (filter rebuilds periodically).
3. **Radix tree** — exact match, but per-shard memory grows linearly
   with cached prefixes; 1 KB cap is hard.

Chose Bloom. Standard, fast, false-positive cost is bounded by one
prefill.

### D2 — Hash significant boundaries, not every token

The shard hashes its KV-cache at the *position-after-system-message*
and *position-after-each-user-turn-marker*. Each adds ~one entry
to the bloom filter. A 64-entry bloom filter at 1% FP rate covers
~32 active sessions per shard — comfortable for typical
8-replica deployments.

### D3 — `route_for_prefix` is additive

Existing `route_for_capability` stays. The new method has signature:

```rust
pub fn route_for_prefix(
    &self,
    model_id: Option<&str>,
    layer: u32,
    capability: &str,
    prefix_hashes: &[u64],
) -> Option<String>;
```

It picks the shard with the **most** matching prefix hashes (using
the bloom filter as proxy), breaking ties by request load. If no
shard has any match, it falls through to the
`route_for_capability` selection.

### D4 — Bloom filter ships in the announce/heartbeat proto

Folds into the `attention-service-routes` change which extends
`AnnounceMsg`. For this change we just add the field on
`ServerEntry` with a default of "empty bloom" — testable without
the proto extension being live.

## Risks / Trade-offs

- **Risk: false positive routes to a shard without the prefix.** →
  Cost: one extra prefill. Quantifiable at FP rate × prefill cost
  per request. At 1% FP and ~8s prefill, expected overhead is 80
  ms / request — well under the 23% TTFT we save by getting hits
  right.
- **Risk: stale bloom filter routes to a shard that has evicted
  the prefix.** → Cost: same as a false positive. Heartbeats
  refresh the filter; staleness window ≤ 5 s by default.
- **Risk: bloom filter overhead per route adds up at high
  concurrency.** Lookup is 4 hash + 4 bit checks — sub-µs. Even at
  10k req/s the cumulative cost is < 100 ms total CPU per second.

## Migration Plan

Land. Existing routes (`route`, `route_for_capability`) keep
working. The new `route_for_prefix` is opt-in per call site —
attention RPCs migrate to it once `attention-service-routes`
ships. Other RPC types stay with `route_for_capability`.

Rollback: revert. The capability-routing path is unchanged.

## Open Questions

- **Q1: How does a shard compute the prefix hashes?** Sticky
  policy: the shard hashes its KV-cache contents at every
  significant boundary it sees (post-system, post-each-user-turn)
  and adds those hashes to the bloom. Doesn't depend on the
  router; the shard just publishes what it has.
- **Q2: Should we publish bloom diffs or full bloom on heartbeat?**
  Full bloom = 1 KB × heartbeat freq (5 s) = 1.6 kbps per shard,
  trivial. Recommendation: full bloom.
- **Q3: What about the FFN container's KV cache?** It doesn't have
  one. Prefix-aware routing only matters for `capability=="attention"`.
