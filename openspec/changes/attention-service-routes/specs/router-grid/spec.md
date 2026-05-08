## MODIFIED Requirements

### Requirement: register MUST consume the capabilities field

`GridState::register` SHALL store the `capabilities` and
`cached_prefixes` carried by the announce payload onto the
`ServerEntry`. (The fields themselves were added in
`router-heterogeneous-shards` and `router-prefix-aware-routing`;
this change wires them to the proto extension.)

#### Scenario: register stores capabilities from the announce

- **WHEN** an announce message with `capabilities =
  ["attention"]` is decoded into a `ServerEntry` and passed to
  `GridState::register`
- **THEN** `GridState::route_for_capability(_, _, "expert")`
  SHALL not return that shard
<!-- test: larql_router::grid::tests::route_for_capability_filters_by_capability -->

#### Scenario: register stores cached_prefixes from the announce

- **WHEN** an announce message includes a `cached_prefixes` of
  32 bytes representing a bloom containing prefix hash `0xCAFE`
- **THEN** `GridState::route_for_prefix(_, _, _, &[0xCAFE])`
  SHALL prefer that shard over a shard whose bloom does not
  contain `0xCAFE`
<!-- test: larql_router::grid::tests::route_for_prefix_picks_shard_with_cached_prefix -->
<!-- test: larql_router::grid::tests::update_heartbeat_with_prefixes_writes_bloom_onto_entry -->

### Requirement: update_heartbeat MUST refresh cached_prefixes

`GridState::update_heartbeat` SHALL accept a
`cached_prefixes: Option<PrefixBloom>` parameter (None when the
heartbeat omitted the field) and write Some values onto
`ServerEntry::cached_prefixes`.

#### Scenario: heartbeat updates the cached prefix bloom

- **WHEN** a shard initially registers with an empty
  `cached_prefixes` bloom, then sends a heartbeat carrying a
  bloom that contains hash `0xBEEF`
- **THEN** subsequent `route_for_prefix(_, _, _, &[0xBEEF])`
  SHALL pick this shard
<!-- test: larql_router::grid::tests::update_heartbeat_with_prefixes_writes_bloom_onto_entry -->

#### Scenario: heartbeat without cached_prefixes preserves prior value

- **WHEN** a shard's bloom contains `0xBEEF`, then it sends a
  heartbeat with no `cached_prefixes` field
- **THEN** the prior bloom SHALL remain on the entry; subsequent
  routing SHALL still find `0xBEEF`
<!-- test: larql_router::grid::tests::update_heartbeat_without_prefixes_preserves_prior_bloom -->
