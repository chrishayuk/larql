## ADDED Requirements

### Requirement: ServerEntry MUST carry a capabilities set

`larql_router::grid::ServerEntry` SHALL have a `capabilities:
Vec<String>` field. New shards register with their declared set;
shards from pre-this-change builds default to
`ServerEntry::default_capabilities()` returning `["attention",
"expert"]` so they continue to receive every RPC.

#### Scenario: default capabilities cover both attention and expert
- **WHEN** a `ServerEntry` is constructed without an explicit capability set
- **THEN** `supports("attention")` SHALL return `true` and `supports("expert")` SHALL return `true`
<!-- test: larql_router::grid::tests::default_capabilities_advertise_both_attention_and_expert -->

### Requirement: Capability-filtered routing returns the right shard

`GridState::route_for_capability(model_id, layer, capability)` SHALL
return the listen URL of a least-loaded shard whose capability set
includes `capability` and whose layer range covers `layer`.

#### Scenario: attention RPC routes to the GPU shard
- **WHEN** an FFN-only shard (`capabilities: ["expert"]`) and a GPU-only shard (`capabilities: ["attention"]`) both cover layer 0
- **THEN** `route_for_capability(model, 0, "attention")` SHALL return the GPU shard's URL and `route_for_capability(model, 0, "expert")` SHALL return the FFN shard's URL
<!-- test: larql_router::grid::tests::route_for_capability_filters_by_capability -->

#### Scenario: missing capability returns None
- **WHEN** only an FFN-only shard is registered and the router asks for an "attention" capability route
- **THEN** the call SHALL return `None`
<!-- test: larql_router::grid::tests::route_for_capability_returns_none_when_no_match -->

### Requirement: Pre-change shards continue to match either capability

A `ServerEntry` constructed via the legacy default (no explicit capabilities) SHALL match capability-filtered routing for both `"attention"` and `"expert"` so existing deployments don't lose traffic when the router upgrades.

#### Scenario: legacy shard receives both capability types
- **WHEN** a legacy default-caps shard covers layer 0 and capability-filtered routing is invoked for both "attention" and "expert"
- **THEN** both calls SHALL return the legacy shard's URL
<!-- test: larql_router::grid::tests::route_for_capability_falls_back_to_default_caps_shard -->
