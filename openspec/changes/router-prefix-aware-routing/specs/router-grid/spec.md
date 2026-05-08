## ADDED Requirements

### Requirement: ServerEntry MUST carry a cached-prefix bloom filter

`ServerEntry` SHALL gain a `cached_prefixes` field of fixed-size
Bloom-filter type (256 bits / 4 hash positions). Default for newly-
registered shards is the empty bloom (all zeros). The field is
populated from the announce / heartbeat payload once the
`attention-service-routes` proto extension lands; pre-extension
shards use the empty default.

#### Scenario: empty bloom is the default
- **WHEN** a `ServerEntry` is constructed without an explicit `cached_prefixes` value
- **THEN** the bloom SHALL report no matches for any input hash
<!-- test: larql_router::grid::tests::empty_bloom_returns_no_matches -->

### Requirement: route_for_prefix MUST prefer the shard with the most matching prefixes

`GridState::route_for_prefix(model_id, layer, capability, prefix_hashes)` SHALL pick the shard whose `cached_prefixes` bloom contains the most of the supplied hashes (with ties broken by `requests_in_flight`). When no shard has any match, the call SHALL fall back to `route_for_capability`'s least-loaded selection.

#### Scenario: shard with cached prefix wins over least-loaded
- **WHEN** two attention shards cover layer 0; shard A has the request's prefix in its bloom and shard B has fewer requests in flight
- **THEN** the route SHALL return shard A's URL
<!-- test: larql_router::grid::tests::route_for_prefix_picks_shard_with_cached_prefix -->

#### Scenario: no match falls back to load-balanced
- **WHEN** no shard's bloom contains any of the supplied prefix hashes
- **THEN** the route SHALL match `route_for_capability` semantics (least-loaded among capability matches)
<!-- test: larql_router::grid::tests::route_for_prefix_falls_back_to_least_loaded_when_no_match -->

#### Scenario: tie on prefix-match count breaks by load
- **WHEN** two shards both match all supplied prefix hashes and have different `requests_in_flight`
- **THEN** the less-loaded shard SHALL win
<!-- test: larql_router::grid::tests::route_for_prefix_breaks_ties_by_load -->

### Requirement: bloom-filter false-positive rate MUST be bounded at design load

The bloom filter SHALL use 256 bits and 4 hash positions per element. At the design load of 16 inserted prefix hashes (typical: chat-template prefix + last several session boundaries), the false-positive rate SHALL be ≤ 1.5%. Operators tracking more sessions per shard MUST either rebuild the bloom at heartbeat boundaries or accept the documented capacity-degradation curve (~16% FP at 64 inserts; ~30% at 128).

#### Scenario: bloom holds 1.5% FP rate at design load (n=16)
- **WHEN** a bloom is loaded with 16 random u64 values and queried with 10000 random other u64s
- **THEN** the fraction of false matches SHALL be ≤ 1.5%
<!-- test: larql_router::grid::tests::bloom_false_positive_rate_within_bound_at_design_load -->

#### Scenario: bloom overload degrades predictably (n=64)
- **WHEN** a bloom is loaded with 64 random u64 values and queried with 10000 random other u64s
- **THEN** the fraction of false matches SHALL be ≤ 21% (5% slack on the theoretical 16%)
<!-- test: larql_router::grid::tests::bloom_overload_degrades_predictably -->
