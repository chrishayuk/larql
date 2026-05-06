## ADDED Requirements

### Requirement: Static layer-shard parsing and ownership

`larql-router` SHALL accept `--shards "START-END=URL[,...]"` and parse
each entry into a `Shard` with inclusive layer bounds. Reversed
ranges (`end < start`), entries missing the `=URL` half, and entirely
empty specs MUST return errors rather than silently install a broken
table. Trailing commas SHALL be tolerated so list-style configs stay
ergonomic. `Shard::owns(layer)` SHALL be `true` if and only if the
layer is within the inclusive `[start, end]` window.

#### Scenario: Single and multi-entry specs parse with inclusive bounds
- **WHEN** `parse_shards("0-16=http://a")` and `parse_shards("0-16=http://a,17-33=http://b")` are called
- **THEN** both SHALL return `Ok(vec![...])` with `start`, `end_inclusive`, and `url` populated correctly
<!-- test: larql_router::tests::parse_shards_single_entry -->
<!-- test: larql_router::tests::parse_shards_two_entries -->
<!-- test: larql_router::tests::shard_owns_inclusive_bounds -->

#### Scenario: Malformed shard specs return errors
- **WHEN** `parse_shards` is given an empty string, an entry without `=URL`, or a reversed range
- **THEN** an `Err` SHALL be returned and the router SHALL refuse to start with that configuration
<!-- test: larql_router::tests::parse_shards_empty_string_errors -->
<!-- test: larql_router::tests::parse_shards_missing_url_errors -->
<!-- test: larql_router::tests::parse_shards_end_less_than_start_errors -->

#### Scenario: Trailing commas in shard specs are tolerated
- **WHEN** `parse_shards("0-16=http://a,")` is called
- **THEN** the trailing comma SHALL be ignored and the call SHALL return the same one-shard result as without the comma
<!-- test: larql_router::tests::parse_shards_ignores_trailing_comma -->

### Requirement: Binary wire peek for transparent forwarding

`peek_binary(body)` SHALL inspect the first u32 of an
`application/x-larql-ffn` request and return the layer indices the
request targets. A single-layer request begins with the layer u32; a
batch request begins with `BATCH_MARKER (0xFFFFFFFF)` followed by the
layer count and each layer index. Empty bodies and truncated batch
layer lists SHALL return `None` so the router refuses to dispatch.
Single-layer payloads truncated after the layer u32 SHALL still
return that layer (the shard is responsible for further validation).

#### Scenario: Single-layer peek returns the layer index
- **WHEN** `peek_binary` is called on a body whose first u32 is the layer index
- **THEN** the return value SHALL be `Some(vec![layer])`
<!-- test: larql_router::tests::peek_binary_single_layer -->
<!-- test: larql_router::tests::peek_binary_truncated_single_returns_value -->

#### Scenario: Batch peek returns every layer index
- **WHEN** `peek_binary` sees `[0xFFFFFFFF, num_layers, layers..., ...]`
- **THEN** every layer in the batch SHALL be returned
<!-- test: larql_router::tests::peek_binary_batch_layers -->
<!-- test: larql_router::tests::peek_binary_zero_batch_layers -->

#### Scenario: Empty or truncated batch headers return None
- **WHEN** `peek_binary` is called with an empty body or with a batch header whose layer list is truncated mid-list
- **THEN** the function SHALL return `None` so the router refuses to dispatch the malformed batch
<!-- test: larql_router::tests::peek_binary_empty_body_returns_none -->
<!-- test: larql_router::tests::peek_binary_batch_truncated_layer_list_returns_none -->

### Requirement: Self-assembling grid route table

`GridState` SHALL maintain two pre-built lookup tables that the
`GridService` rebuilds on every topology change (register / deregister)
and leaves alone on heartbeat updates: `route_table[(model_id, layer)]`
for named-model queries and `any_model_table[layer]` for single-model
grids where `model_id` is omitted. `route(model_id, layer)` SHALL
return the URL of the least-loaded replica (minimum
`requests_in_flight` from the last heartbeat). `route_all(...)` SHALL
resolve a layer batch under a single lock acquisition and return the
first uncovered layer when the grid cannot satisfy the request.
`status_response()` SHALL surface the live shard list and any layer
gaps so operators can spot under-provisioning.

#### Scenario: Layer ranges are inclusive on both ends
- **WHEN** a server registers as owning `layers [start, end]` and `route(None, layer)` is called for `start`, `end`, and out-of-range values
- **THEN** the bounds layers SHALL resolve to the registered server URL and out-of-range layers SHALL return `None`
<!-- test: larql_router::grid::tests::route_uses_inclusive_layer_ranges -->

#### Scenario: Single-model grid uses any_model_table when model_id is None
- **WHEN** `route(None, layer)` is called against a grid with one server registered for some `model_id`
- **THEN** the router SHALL still resolve via the `any_model_table` and return that server's URL
<!-- test: larql_router::grid::tests::route_without_model_uses_any_model_table -->

#### Scenario: Replica selection prefers the least loaded server
- **WHEN** two replicas serve the same layer with different `requests_in_flight`
- **THEN** `route` SHALL return the URL whose load is lower, biasing traffic towards under-utilised replicas
<!-- test: larql_router::grid::tests::route_prefers_least_loaded_replica -->

#### Scenario: Heartbeats update load without rebuilding topology
- **WHEN** `update_heartbeat(server_id, ...)` is called for an already-registered server
- **THEN** the cached load metric SHALL change but the route tables SHALL not be rebuilt (route table identity preserved)
<!-- test: larql_router::grid::tests::heartbeat_updates_load_without_rebuilding_topology -->

#### Scenario: Deregistration removes the server from the route table
- **WHEN** `deregister(server_id)` is called for a registered server
- **THEN** that server SHALL no longer appear in any route lookup, and rebuilds SHALL recover from concurrent re-registrations
<!-- test: larql_router::grid::tests::deregister_removes_server_from_route_table -->

#### Scenario: route_all reports the first uncovered layer
- **WHEN** `route_all(model_id, [layers...])` is called against a grid that does not cover every requested layer
- **THEN** the call SHALL return an error/marker identifying the first uncovered layer so the HTTP handler can return HTTP 400 with the gap
<!-- test: larql_router::grid::tests::route_all_returns_first_uncovered_layer -->

#### Scenario: Status response surfaces shards and gaps
- **WHEN** `status_response()` is called against a partially-covered grid
- **THEN** the returned struct SHALL list every active shard's URL plus every layer gap so operators can see coverage at a glance
<!-- test: larql_router::grid::tests::status_response_reports_shards_and_gaps -->

### Requirement: HTTP dispatch — single-shard forwarding and multi-shard JSON fan-out

`POST /v1/walk-ffn` on `larql-router` SHALL dispatch by parsing layers
from either the JSON body or the binary header (via `peek_binary`).
For requests that resolve to a single shard the router SHALL forward
the body unchanged so binary requests stay byte-for-byte intact. For
multi-shard JSON batches the router SHALL group layers per owning
shard, dispatch sub-requests in parallel, merge the results sorted by
layer, and return the maximum shard latency as the wall-clock
`latency_ms`. Multi-shard binary fan-out SHALL be rejected (HTTP
400) — clients are required to use JSON for cross-shard batches.
Unknown / unowned layers SHALL return HTTP 400 with the offending
layer named in the error body. `/v1/health` SHALL always return
`{"status":"ok"}`.

#### Scenario: Static map serves single-layer requests inclusive of bounds
- **WHEN** the router receives a request whose layer is one of the inclusive bounds of a static `Shard`
- **THEN** `Shard::owns(layer)` SHALL be `true` and the request SHALL forward to that shard's URL
<!-- test: larql_router::tests::shard_owns_inclusive_bounds -->

#### Scenario: Single-layer binary request forwards intact
- **WHEN** a binary request whose first u32 is a layer index resolves to a single shard via `peek_binary` + `resolve_all`
- **THEN** the router SHALL forward the raw body to that shard without re-encoding
<!-- test: larql_router::tests::peek_binary_single_layer -->
<!-- test: larql_router::tests::peek_binary_truncated_single_returns_value -->

#### Scenario: Batch binary request resolves all layers before dispatch
- **WHEN** a batch binary request is parsed by `peek_binary`
- **THEN** every encoded layer SHALL be returned for `resolve_all` to verify ownership and to detect the cross-shard rejection case
<!-- test: larql_router::tests::peek_binary_batch_layers -->
<!-- test: larql_router::tests::peek_binary_zero_batch_layers -->

#### Scenario: Truncated batch binary request is rejected
- **WHEN** the batch binary header advertises N layers but the body terminates before all N indices are present
- **THEN** `peek_binary` SHALL return `None` and the router SHALL respond 400 rather than fan out a partial request
<!-- test: larql_router::tests::peek_binary_batch_truncated_layer_list_returns_none -->
<!-- test: larql_router::tests::peek_binary_empty_body_returns_none -->

### Requirement: GridService gRPC enrollment protocol

`GridServiceImpl` SHALL accept long-lived bidirectional `Join`
streams. On the first message (an announce) the router SHALL allocate
a stable `server_id`, register the server with its `(model_id,
layer_start, layer_end_inclusive, listen_url, vindex_hash)` tuple, and
send an `AckMsg`. Subsequent `HeartbeatMsg` frames SHALL update the
server's load metrics in place without triggering a route-table
rebuild. When the stream closes (clean shutdown or crash) the server
SHALL be deregistered and the route table SHALL be rebuilt. When
`--grid-key` is configured the implementation MUST reject `Join`
RPCs whose metadata does not present `Authorization: Bearer <key>`.
The reusable proto types are exposed via `larql-router-protocol`
(`GridService`, `GridServiceServer`, `GridServiceClient`,
`ServerMessage`, `RouterMessage`, `AnnounceMsg`, `HeartbeatMsg`,
`AckMsg`, `DroppingMsg`).

#### Scenario: Heartbeat-only updates avoid full rebuild
- **WHEN** a heartbeat frame arrives for a registered server
- **THEN** the load metric SHALL change but the cached `route_table` and `any_model_table` SHALL not be rebuilt (heartbeats are O(1))
<!-- test: larql_router::grid::tests::heartbeat_updates_load_without_rebuilding_topology -->

#### Scenario: Stream close triggers deregister and rebuild
- **WHEN** a server's `Join` stream closes
- **THEN** `deregister` SHALL remove the server and the route tables SHALL be rebuilt so subsequent `route` calls return another replica or `None`
<!-- test: larql_router::grid::tests::deregister_removes_server_from_route_table -->

#### Scenario: Newly-registered shard is reachable via every routing surface
- **WHEN** a shard registers via `Join` and a layer in its range is queried via `route`, `route_all`, and `status_response`
- **THEN** every surface SHALL surface the new shard URL and the same answer SHALL be served whether the request supplies `model_id` or omits it
<!-- test: larql_router::grid::tests::route_uses_inclusive_layer_ranges -->
<!-- test: larql_router::grid::tests::route_without_model_uses_any_model_table -->
<!-- test: larql_router::grid::tests::status_response_reports_shards_and_gaps -->

#### Scenario: Grid identity hash is logged per registration
- **WHEN** a server announces with a `vindex_hash` derived from `vindex_identity_hash(model_id, num_layers)`
- **THEN** the hash SHALL be a deterministic 16-char hex string so the router can log it on every registration and operators can spot version mismatches
<!-- test: larql_server::announce::tests::vindex_identity_hash_is_stable_and_hex -->
<!-- test: larql_server::test_unit_state::vindex_identity_hash_is_deterministic -->
<!-- test: larql_server::test_unit_state::vindex_identity_hash_is_hex_string -->

### Requirement: Test suite coverage

The capability's behavior SHALL be exercised by every test in the test modules listed below. New tests added to those modules SHALL be considered part of this capability's contract by default.

#### Scenario: Workspace tests bound to this capability
- **WHEN** the trace tool resolves the wildcard module references for this capability
- **THEN** every backing test SHALL resolve to a real `#[test]` function in the workspace
<!-- test: larql_router::tests::**::* -->
<!-- test: larql_router::grid::tests::**::* -->
