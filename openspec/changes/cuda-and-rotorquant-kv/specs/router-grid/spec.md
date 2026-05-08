## ADDED Requirements

### Requirement: Shards advertise capability sets

A shard registering with the router SHALL declare a capability set
containing zero or more of `attention`, `expert`. Existing shards
that pre-date this change continue to behave as if they advertised
both capabilities (backwards-compatible default).

#### Scenario: Attention-only shard declares "attention"
- **WHEN** a `larql-server --role attention` boots and announces itself
- **THEN** the announce payload SHALL contain `capabilities: ["attention"]`
<!-- test: larql_server::bootstrap::tests::role_attention_announces_attention_only -->
<!-- test: larql_server::announce::tests::announce_message_carries_capabilities -->

#### Scenario: Pre-change all-roles shard still works
- **WHEN** a shard from a pre-change deployment registers with no capability set
- **THEN** the router SHALL treat it as advertising both `attention` and `expert`
<!-- test: larql_router::grid::tests::default_capabilities_advertise_both_attention_and_expert -->
<!-- test: larql_router::grid::tests::route_for_capability_falls_back_to_default_caps_shard -->

### Requirement: Router routes by capability + layer range

For each incoming request, the router SHALL pick a shard whose
capability set covers the requested operation (`attention` for
attention RPCs, `expert` for FFN dispatch RPCs) AND whose layer
range covers the requested layer. If no shard matches, the router
SHALL respond with HTTP 503 and a body explaining which capability
or layer range was missing.

#### Scenario: Attention RPC reaches the GPU shard
- **WHEN** a `/v1/attention/decode` is sent to the router for layer 17 in a 32-layer model
- **THEN** the router SHALL forward to a shard that advertises `attention` and covers layer 17
<!-- test: larql_router::grid::tests::route_for_capability_filters_by_capability -->

#### Scenario: Expert RPC reaches the CPU shard
- **WHEN** a `/v1/expert/batch` is sent to the router for layer 17
- **THEN** the router SHALL forward to a shard that advertises `expert` and covers layer 17
<!-- test: larql_router::grid::tests::route_for_capability_filters_by_capability -->

#### Scenario: Missing capability returns 503 with a useful body
- **WHEN** an attention RPC reaches a router whose grid contains only expert shards
- **THEN** the response SHALL be HTTP 503 with a body containing "no shard advertises capability=attention"
<!-- test: larql_router::grid::tests::route_for_capability_returns_none_when_no_match -->
<!-- test: unbacked -->

### Requirement: Heterogeneous deadlock prevention

The router SHALL enforce a strict per-layer ordering: attention
output must reach the FFN shard before the FFN result is awaited,
and the FFN result must be present before the next layer's attention
is dispatched. The router SHALL time out individual hops at a
configurable deadline (default 5 s) to prevent deadlocks when one
shard is unhealthy.

#### Scenario: Attention timeout returns 504 and frees the FFN reservation
- **WHEN** an attention shard goes unresponsive mid-decode
- **THEN** within the configured deadline the router SHALL return 504 and the corresponding FFN-shard reservation SHALL be released
<!-- test: unbacked -->
