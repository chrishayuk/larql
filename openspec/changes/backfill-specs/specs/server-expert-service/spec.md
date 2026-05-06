## ADDED Requirements

### Requirement: Expert dispatch endpoints and topology

`larql-server` SHALL expose the expert-service surface for sharded
hybrid-MoE inference: `POST /v1/expert/{layer}/{expert_id}` for a
single-expert call, the legacy `POST /v1/expert/batch` (one residual
per item), `POST /v1/experts/layer-batch` and its `f16` variant
(one residual + K weighted experts → router-weighted sum), `POST
/v1/experts/multi-layer-batch` and its `q8k` variant (multi-layer
batches with prenorm-aware Q8K weights), and the descriptor endpoint
`GET /v1/expert/topology`. All `expert/...` POST routes MUST honor an
expanded body limit (`EXPERT_BATCH_BODY_LIMIT`) so that
`N_positions × top_K × hidden` payloads fit. Public re-exports from
`routes::expert::*` MUST preserve `handle_expert`, `run_expert`,
`handle_expert_batch`, `handle_experts_layer_batch[_f16]`,
`handle_experts_multi_layer_batch[_q8k]`, `run_experts_cpu_batch`, and
`warmup_hnsw_unit_cache` (and `warmup_metal_expert_cache` /
`run_experts_metal_batch` under `metal-experts`).

#### Scenario: Single-shard expert/batch parity with local execution
- **WHEN** the expert endpoint is exercised against a single-shard server
- **THEN** the response output for a sharded batch SHALL match a locally computed reference within numerical tolerance and SHALL preserve order across shard boundaries
<!-- test: larql_server::test_expert_endpoint::expert_endpoint_single_shard_parity -->

#### Scenario: Two-shard topology produces identical aggregate output
- **WHEN** the same input is dispatched across two shards
- **THEN** the merged output SHALL match the single-shard / local reference, validating that the layer-batch / multi-layer-batch handlers correctly weight per-expert contributions
<!-- test: larql_server::test_expert_endpoint::expert_endpoint_two_shard_parity -->

#### Scenario: Resharding does not change the output
- **WHEN** the expert ownership is moved from one shard layout to another
- **THEN** the output SHALL be identical, demonstrating that handlers depend only on `(layer, expert_id)` rather than on shard identity
<!-- test: larql_server::test_expert_endpoint::expert_endpoint_reshard_same_output -->

#### Scenario: A server hosting no shard rejects expert calls
- **WHEN** the expert endpoint is called against a server that owns no expert range
- **THEN** the response SHALL be an error rather than silent zeros
<!-- test: larql_server::test_expert_endpoint::expert_endpoint_no_shard_error -->

### Requirement: Layer-batch and multi-layer-batch wire format

The layer-batch wire format SHALL be a binary frame encoding `(layer
u32, hidden u32, residual f32×hidden, K u32, expert_ids u32×K, weights
f32×K)` for the f32 path and the equivalent `f16` packed payload for
the f16 path. Server-side `decode_layer_batch_request[_f16]` MUST
round-trip whatever `encode_layer_batch_request[_f16]` emits, MUST
return `None` on truncated bytes (so handlers short-circuit), and the
response encoders `encode_layer_batch_response[_f16]` MUST emit a
fixed 8-byte header even on empty input.

#### Scenario: f32 layer-batch decode round-trips
- **WHEN** `encode_layer_batch_request` produces a full frame and `decode_layer_batch_request` is called on it
- **THEN** the decoded `(layer, residual, expert_ids, weights)` tuple SHALL exactly equal the inputs
<!-- test: larql_server::routes::expert::layer_batch::layer_batch_wire_tests::server_decodes_layer_batch_request_f32 -->

#### Scenario: Truncated layer-batch payloads are rejected
- **WHEN** the layer-batch decoder is called on a series of truncations of a valid frame
- **THEN** every truncated decode SHALL return `None`
<!-- test: larql_server::routes::expert::layer_batch::layer_batch_wire_tests::server_rejects_truncated_layer_batch_request -->

#### Scenario: f16 layer-batch decode preserves layer/ids/weights and approximates residual
- **WHEN** `encode_layer_batch_request_f16` produces a frame and the f16 decoder consumes it
- **THEN** layer, expert ids, and weights SHALL be exact, and residual values SHALL match within an f16-appropriate tolerance (≤0.1% relative)
<!-- test: larql_server::routes::expert::layer_batch::layer_batch_wire_tests::server_decodes_layer_batch_request_f16 -->

#### Scenario: Empty response payloads still produce a header
- **WHEN** `encode_layer_batch_response` and `encode_layer_batch_response_f16` are called with empty outputs
- **THEN** each SHALL produce exactly 8 bytes (hidden u32 + latency f32)
<!-- test: larql_server::routes::expert::layer_batch::layer_batch_wire_tests::server_response_encoders_handle_empty -->

### Requirement: Walk-FFN binary wire format

The `/v1/walk-ffn` binary wire SHALL support both single-layer and
batch frames. `decode_binary_request` MUST: accept single-layer
payloads with `(layer u32, seq_len u32, full_output u8, top_k u32,
residual f32×N)`; accept batch payloads beginning with the `BATCH_MARKER`
followed by `(num_layers u32, layers u32×N, …)`; reject truncated,
empty, and odd-length-residual payloads; and reject batch payloads
that claim more layers than they carry. `encode_binary_output` MUST
emit single-entry frames in the form `[layer u32][seq_len u32][latency
f32][output f32×]` and batch frames prefixed with `BATCH_MARKER` plus
per-entry headers. `encode_json_full_output` MUST emit a single-layer
shape when one entry is present (top-level `layer`, `seq_len`,
`output`, `latency_ms`) and a batch shape with `results[]` otherwise.

#### Scenario: Single and batch decode round-trips
- **WHEN** `decode_binary_request` is invoked on synthesised single-layer and batch frames
- **THEN** the decoded request SHALL contain exactly the layer (or layers list), seq_len, full_output flag, top_k, and residual the encoder produced
<!-- test: larql_server::routes::walk_ffn::tests::decode_single_layer_request -->
<!-- test: larql_server::routes::walk_ffn::tests::decode_batch_request -->
<!-- test: larql_server::routes::walk_ffn::tests::decode_features_only_binary -->

#### Scenario: Malformed walk-ffn binary input is rejected
- **WHEN** the decoder receives a truncated body, an empty body, a batch frame whose declared layer count exceeds the bytes provided, or a residual whose byte length is not a multiple of 4
- **THEN** every case SHALL return an error
<!-- test: larql_server::routes::walk_ffn::tests::decode_binary_truncated_body -->
<!-- test: larql_server::routes::walk_ffn::tests::decode_binary_empty_body -->
<!-- test: larql_server::routes::walk_ffn::tests::decode_binary_batch_truncated_layers -->
<!-- test: larql_server::routes::walk_ffn::tests::decode_binary_odd_residual_length -->

#### Scenario: Walk-FFN binary and JSON encoders preserve shape
- **WHEN** outputs are encoded for single and batch cases via `encode_binary_output` and `encode_json_full_output`
- **THEN** the binary frames SHALL have the documented header layout, batch frames SHALL begin with `BATCH_MARKER`, float bits SHALL be preserved exactly through round-trip, and the JSON shape SHALL distinguish single-layer (top-level keys) from batch (`results[]`) responses
<!-- test: larql_server::routes::walk_ffn::tests::encode_single_entry_output -->
<!-- test: larql_server::routes::walk_ffn::tests::encode_batch_output -->
<!-- test: larql_server::routes::walk_ffn::tests::binary_roundtrip_float_preservation -->
<!-- test: larql_server::routes::walk_ffn::tests::json_single_layer_format -->
<!-- test: larql_server::routes::walk_ffn::tests::json_batch_format -->

### Requirement: Band utilities for FFN batch handling

`larql_server::band_utils` SHALL expose `BAND_ALL`, `BAND_KNOWLEDGE`,
`BAND_OUTPUT`, `BAND_SYNTAX`, the inference-mode constants
(`INFER_MODE_WALK`, `_DENSE`, `_COMPARE`), and the insert-mode
constants (`INSERT_MODE_CONSTELLATION`, `_EMBEDDING`).
`get_layer_bands` MUST return the model's configured bands when
present and a fallback when absent. `filter_layers_by_band` MUST
restrict an input layer list to the chosen band, MUST treat unknown
bands as `BAND_ALL`, MUST handle empty inputs and zero-width bands
without panic, and MUST return an empty list when no input layer
falls in the requested band.

#### Scenario: Band, infer-mode, and insert-mode constants are stable
- **WHEN** the constant values are read
- **THEN** they SHALL match the documented strings (`"all"`, `"knowledge"`, `"output"`, `"syntax"`, `"walk"`, `"dense"`, `"compare"`, `"constellation"`, `"embedding"`)
<!-- test: larql_server::test_unit_band_utils::band_constants_correct_values -->
<!-- test: larql_server::test_unit_band_utils::mode_constants_correct_values -->
<!-- test: larql_server::test_unit_band_utils::insert_mode_constants_correct_values -->

#### Scenario: filter_layers_by_band selects the right subset
- **WHEN** `filter_layers_by_band` is called with `syntax`, `knowledge`, `output`, `all`, an unknown band, and on an empty input
- **THEN** each call SHALL return only the layers in the requested band, unknown bands SHALL behave like `all`, empty input SHALL return empty, and inputs with no layer in the band SHALL return empty
<!-- test: larql_server::test_unit_band_utils::filter_syntax_returns_syntax_layers -->
<!-- test: larql_server::test_unit_band_utils::filter_knowledge_returns_knowledge_layers -->
<!-- test: larql_server::test_unit_band_utils::filter_output_returns_output_layers -->
<!-- test: larql_server::test_unit_band_utils::filter_all_returns_all_layers -->
<!-- test: larql_server::test_unit_band_utils::filter_unknown_band_returns_all_layers -->
<!-- test: larql_server::test_unit_band_utils::filter_empty_input_returns_empty -->
<!-- test: larql_server::test_unit_band_utils::filter_no_match_in_band_returns_empty -->
<!-- test: larql_server::test_unit_band_utils::filter_knowledge_with_zero_width_band -->

#### Scenario: get_layer_bands prefers config bands and falls back gracefully
- **WHEN** `get_layer_bands` is called against a model that supplies layer bands and against one that does not
- **THEN** the configured bands SHALL be returned when present and a fallback SHALL be returned otherwise
<!-- test: larql_server::test_unit_band_utils::get_layer_bands_uses_config_bands_when_present -->
<!-- test: larql_server::test_unit_band_utils::get_layer_bands_falls_back_when_none -->

### Requirement: FFN L2 gate cache (LRU, bounded)

`larql_server::ffn_l2_cache::FfnL2Cache` SHALL be a per-layer
order-independent gate cache that derives keys via the same scheme as
the L1 cache (sorted `feature_ids` hash), MUST cap each layer at
`max_entries`, MUST drop further inserts silently when full, MUST
return `Arc`-shared values so concurrent readers do not clone, MUST
maintain hit/miss/total counters and a rounded hit-rate, MUST be
panic-free for out-of-range layers, MUST keep layers independent, and
MUST emit a `stats()` JSON containing `hits`, `misses`, `total`,
`hit_rate`, `layers`, and `max_entries_per_layer`.

#### Scenario: Key scheme matches L1 and is order-independent
- **WHEN** `FfnL2Cache::key` is computed and compared against the L1 derivation, and when the same ids are provided in different orders
- **THEN** L1 and L2 keys SHALL be equal and the keys SHALL be identical regardless of input order
<!-- test: larql_server::ffn_l2_cache::tests::key_matches_l1_scheme -->
<!-- test: larql_server::ffn_l2_cache::tests::key_is_order_independent -->

#### Scenario: Miss-then-hit accounting is correct
- **WHEN** a miss is followed by an insert and a get, and a sequence of hits and misses are issued
- **THEN** misses SHALL increment on miss, inserts SHALL not affect counters, hits SHALL increment on get-hit, and `hit_rate()` SHALL equal `hits / (hits + misses)` to three decimal places
<!-- test: larql_server::ffn_l2_cache::tests::miss_then_hit -->
<!-- test: larql_server::ffn_l2_cache::tests::hit_rate_computation -->

#### Scenario: Capacity cap drops further inserts silently
- **WHEN** more entries than `max_entries` are inserted into a layer
- **THEN** the first `max_entries` SHALL remain readable and additional inserts SHALL be dropped without error
<!-- test: larql_server::ffn_l2_cache::tests::capacity_cap -->

#### Scenario: Layers are independent and out-of-range is safe
- **WHEN** the same key is inserted into different layers and a get/insert is issued against a layer beyond `num_layers`
- **THEN** layers SHALL not interfere with each other and out-of-range operations SHALL not panic
<!-- test: larql_server::ffn_l2_cache::tests::layers_are_independent -->
<!-- test: larql_server::ffn_l2_cache::tests::out_of_range_layer_is_safe -->

#### Scenario: Values are Arc-shared, concurrent reads do not panic, and stats JSON is well-formed
- **WHEN** the same key is fetched twice and produces two `Arc`s, when 8 threads concurrently read a hit, and when `stats()` is serialised
- **THEN** the two Arcs SHALL be `ptr_eq`, the concurrent reads SHALL all observe the cached value without panic, and the JSON SHALL contain numeric `hits`/`misses`/`total`/`hit_rate` plus `layers` and `max_entries_per_layer` fields
<!-- test: larql_server::ffn_l2_cache::tests::arc_values_are_shared_not_cloned -->
<!-- test: larql_server::ffn_l2_cache::tests::concurrent_reads_do_not_panic -->
<!-- test: larql_server::ffn_l2_cache::tests::stats_json_has_expected_fields -->

### Requirement: Test suite coverage

The capability's behavior SHALL be exercised by every test in the test modules listed below. New tests added to those modules SHALL be considered part of this capability's contract by default.

#### Scenario: Workspace tests bound to this capability
- **WHEN** the trace tool resolves the wildcard module references for this capability
- **THEN** every backing test SHALL resolve to a real `#[test]` function in the workspace
<!-- test: larql_server::test_expert_endpoint::**::* -->
<!-- test: larql_server::routes::walk_ffn::tests::**::* -->
<!-- test: larql_server::ffn_l2_cache::tests::**::* -->
