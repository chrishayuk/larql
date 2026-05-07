## 1. Bloom filter

- [ ] 1.1 Pick a workspace bloom impl (eval `fastbloom` 0.x or
      hand-rolled). Constraint: `Send + Sync + Clone`, 256-bit
      fixed size, 4 hash positions, deterministic across hosts.
- [ ] 1.2 `pub struct PrefixBloom { bits: [u64; 4] }` with
      `insert(u64)`, `contains(u64) -> bool`, `Default::default`.

## 2. ServerEntry extension

- [ ] 2.1 Add `pub cached_prefixes: PrefixBloom` field, defaulting
      to the empty bloom in both the announce-handler and the
      test helper.

## 3. route_for_prefix

- [ ] 3.1 New method on `GridState` that walks candidate shards,
      counts bloom matches, breaks ties by load.
- [ ] 3.2 Falls back to `route_for_capability` when no shard
      matches any prefix.

## 4. Tests

- [ ] 4.1 `empty_bloom_returns_no_matches`.
- [ ] 4.2 `route_for_prefix_picks_shard_with_cached_prefix`.
- [ ] 4.3 `route_for_prefix_falls_back_when_no_match`.
- [ ] 4.4 `route_for_prefix_breaks_ties_by_load`.
- [ ] 4.5 `bloom_false_positive_rate_within_bound`.

## 5. Validation

- [ ] 5.1 `openspec validate router-prefix-aware-routing --strict` passes.
- [ ] 5.2 `cargo test -p larql-router` passes (existing 11 + new 5).
- [ ] 5.3 `make traceability-check` and `make openspec-validate` pass.
