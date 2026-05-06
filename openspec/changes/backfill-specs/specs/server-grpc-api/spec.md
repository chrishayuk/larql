## ADDED Requirements

### Requirement: gRPC service surface mirrors the HTTP knowledge API

`larql-server` SHALL expose `VindexService` over Tonic with the unary
RPCs `Health`, `GetStats`, `Describe`, `Walk`, `Select`, `Infer`,
`GetRelations`, `WalkFfn`, plus the server-streaming RPC
`StreamDescribe`. Per-RPC handlers MUST mirror the semantics of their
HTTP counterparts (band filters, layer filters, ordering, mode
selection, top-k, etc.) and SHALL share the same in-process state
(`AppState`) so that the request counter is maintained across both
transports.

#### Scenario: Health RPC reports status, uptime, and bumps counter
- **WHEN** the `Health` RPC is invoked against `VindexGrpcService`
- **THEN** the response status SHALL be `OK`, the response SHALL carry an `uptime_seconds` value, and the request counter SHALL be incremented
<!-- test: larql_server::test_grpc::grpc_health_returns_ok_status -->
<!-- test: larql_server::test_grpc::grpc_health_returns_uptime -->
<!-- test: larql_server::test_grpc::grpc_health_bumps_request_counter -->

#### Scenario: GetStats returns model info with layer bands
- **WHEN** `GetStats` is called against a service with a model loaded vs. no model
- **THEN** the loaded case SHALL return populated model info including a `layer_bands` shape and the no-model case SHALL return `Status::not_found`
<!-- test: larql_server::test_grpc::grpc_get_stats_returns_model_info -->
<!-- test: larql_server::test_grpc::grpc_get_stats_has_layer_bands -->
<!-- test: larql_server::test_grpc::grpc_get_stats_no_model_returns_not_found -->

### Requirement: Describe / Walk / Select gRPC parity with HTTP

The `Describe`, `Walk`, and `Select` RPCs SHALL produce the same edges,
hits, and selection results as their HTTP counterparts on the same
loaded model. Empty/missing inputs SHALL yield documented `Status`
codes: empty tokenizer SHALL produce empty edges, empty prompt for
`Walk` SHALL return `Status::invalid_argument`, and any RPC against a
service with no model loaded SHALL return `Status::not_found`.

#### Scenario: Describe returns edges and propagates not-found
- **WHEN** `Describe` is called against an empty-tokenizer model, a functional model, and a service with no model
- **THEN** the empty case SHALL return an empty edges list, the functional case SHALL return edges (with the top edge matching the functional fixture's `Paris` mapping), and the no-model case SHALL return `Status::not_found`
<!-- test: larql_server::test_grpc::grpc_describe_empty_tokenizer_returns_empty_edges -->
<!-- test: larql_server::test_grpc::grpc_describe_functional_returns_edges -->
<!-- test: larql_server::test_grpc::grpc_describe_top_edge_is_paris -->
<!-- test: larql_server::test_grpc::grpc_describe_no_model_returns_not_found -->

#### Scenario: Walk hit ranking and input validation
- **WHEN** `Walk` is invoked with a normal prompt, an empty prompt, and against a service with no model
- **THEN** the normal case SHALL return hits with `Paris` as the top hit, the empty prompt SHALL return `Status::invalid_argument`, and the no-model case SHALL return `Status::not_found`
<!-- test: larql_server::test_grpc::grpc_walk_functional_returns_hits -->
<!-- test: larql_server::test_grpc::grpc_walk_top_hit_is_paris -->
<!-- test: larql_server::test_grpc::grpc_walk_empty_prompt_returns_invalid_arg -->
<!-- test: larql_server::test_grpc::grpc_walk_no_model_returns_not_found -->

#### Scenario: Select honors entity filter and not-found path
- **WHEN** `Select` is called with no filter, with an entity filter, and against a service with no model
- **THEN** the unfiltered case SHALL return all features, the entity-filtered case SHALL only return features for that entity, and the no-model case SHALL return `Status::not_found`
<!-- test: larql_server::test_grpc::grpc_select_all_returns_features -->
<!-- test: larql_server::test_grpc::grpc_select_with_entity_filter -->
<!-- test: larql_server::test_grpc::grpc_select_no_model_returns_not_found -->

### Requirement: Infer, Relations, and WalkFfn RPCs

`Infer` SHALL return `Status::unavailable` when inference is disabled,
`Status::not_found` when no model is loaded. `GetRelations` SHALL
return the per-relation list with totals. `WalkFfn` SHALL support
features-only mode and multi-layer batches, MUST validate the residual
size against the model's `hidden_size` (returning
`Status::invalid_argument` on mismatch), and SHALL return
`Status::not_found` when no model is loaded.

#### Scenario: Infer disabled and not-found paths
- **WHEN** `Infer` is invoked against a service with inference disabled and against a service with no model
- **THEN** the disabled case SHALL return `Status::unavailable` and the no-model case SHALL return `Status::not_found`
<!-- test: larql_server::test_grpc::grpc_infer_disabled_returns_unavailable -->
<!-- test: larql_server::test_grpc::grpc_infer_no_model_returns_not_found -->

#### Scenario: GetRelations returns list and propagates not-found
- **WHEN** `GetRelations` is called against a model with relations and against a service with no model
- **THEN** the loaded case SHALL return a relations list and the no-model case SHALL return `Status::not_found`
<!-- test: larql_server::test_grpc::grpc_get_relations_returns_list -->
<!-- test: larql_server::test_grpc::grpc_get_relations_no_model_returns_not_found -->

#### Scenario: WalkFfn features-only and multi-layer batches
- **WHEN** `WalkFfn` is called features-only with a single layer, with a multi-layer batch, with a wrong-sized residual, and against a service with no model
- **THEN** the single- and multi-layer features-only calls SHALL return per-layer features/scores, the wrong residual SHALL return `Status::invalid_argument`, and the no-model case SHALL return `Status::not_found`
<!-- test: larql_server::test_grpc::grpc_walk_ffn_features_only_returns_results -->
<!-- test: larql_server::test_grpc::grpc_walk_ffn_multi_layer_batch_returns_all -->
<!-- test: larql_server::test_grpc::grpc_walk_ffn_wrong_residual_size_returns_invalid_arg -->
<!-- test: larql_server::test_grpc::grpc_walk_ffn_no_model_returns_not_found -->

### Requirement: Server-streaming StreamDescribe

`StreamDescribe` SHALL be a server-streaming RPC that emits per-layer
`DescribeLayerEvent`s with a final `done` event carrying `total_edges`
and `latency_ms`. The stream MUST collect into the same edge set the
unary `Describe` would produce on the same input. When no model is
loaded the stream MUST resolve to `Status::not_found` before any
events are emitted.

#### Scenario: StreamDescribe emits events and aggregates to describe parity
- **WHEN** `StreamDescribe` is invoked on a functional model and against a service with no model
- **THEN** the functional case SHALL produce a non-empty stream that, when collected, matches the unary `Describe` output (modulo ordering), and the no-model case SHALL return `Status::not_found`
<!-- test: larql_server::test_grpc::grpc_stream_describe_returns_stream -->
<!-- test: larql_server::test_grpc::grpc_stream_describe_collects_events -->
<!-- test: larql_server::test_grpc::grpc_stream_describe_no_model_returns_not_found -->

### Requirement: Test suite coverage

The capability's behavior SHALL be exercised by every test in the test modules listed below. New tests added to those modules SHALL be considered part of this capability's contract by default.

#### Scenario: Workspace tests bound to this capability
- **WHEN** the trace tool resolves the wildcard module references for this capability
- **THEN** every backing test SHALL resolve to a real `#[test]` function in the workspace
<!-- test: larql_server::test_grpc::**::* -->
