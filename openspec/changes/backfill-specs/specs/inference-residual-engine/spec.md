## ADDED Requirements

### Requirement: Pluggable KV-cache engine surface

`larql_inference::engines` SHALL expose a single `KvEngine` trait
implemented by every concrete decode engine (`MarkovResidualEngine`,
`ApolloEngine`, `TurboQuantEngine`, `UnlimitedContextEngine`). The module
MUST publish an `EngineKind::from_name` parser that round-trips engine
names with optional `key=value` parameter strings, MUST report
`memory_bytes` / `cold_bytes` / `window_tokens` as zero before any prefill,
and MUST surface a non-empty `EngineInfo` summary for every engine.

#### Scenario: EngineKind name parsing round-trips with and without parameters
- **WHEN** `EngineKind::from_name` is called on every registered engine name and on names with `?key=value` parameter strings
- **THEN** every engine SHALL parse successfully, name round-trips SHALL be lossless, unknown parameter keys SHALL be ignored with defaults applied, and `EngineInfo` summaries SHALL include the parameters
<!-- test: larql_inference::engines::tests::engine_kind_from_name_roundtrip -->
<!-- test: larql_inference::engines::tests::engine_kind_from_name_with_params -->
<!-- test: larql_inference::engines::tests::from_name_unknown_param_ignored_defaults_apply -->
<!-- test: larql_inference::engines::tests::from_name_all_engines_parseable -->
<!-- test: larql_inference::engines::tests::engine_info_summary_with_config -->
<!-- test: larql_inference::engines::tests::engine_info_summary_no_config -->

#### Scenario: Every engine reports zero state before prefill
- **WHEN** each registered engine is constructed but not yet prefilled
- **THEN** `memory_bytes`, `window_tokens`, and `cold_bytes` SHALL be zero, the per-engine `EngineInfo` fields SHALL be non-empty, the engine name SHALL be valid, and the stage summary SHALL be `None`
<!-- test: larql_inference::engines::tests::all_engines_memory_zero_before_prefill -->
<!-- test: larql_inference::engines::tests::all_engines_window_tokens_zero_before_prefill -->
<!-- test: larql_inference::engines::tests::all_engines_cold_bytes_zero_before_prefill -->
<!-- test: larql_inference::engines::tests::all_engines_have_valid_name -->
<!-- test: larql_inference::engines::tests::all_engines_info_has_nonempty_fields -->
<!-- test: larql_inference::engines::tests::all_engines_stage_summary_none_before_decode -->

### Requirement: Markov Residual engine bit-perfect bounded-window decode

The `MarkovResidualEngine` (in `larql_inference::engines::kv_engines::markov_residual`) SHALL replace the per-token KV cache with the residual stream itself as the persistent state. The engine MUST keep a hot residual store bounded by a configurable window `W`, MUST archive overflow rows into a cold tier without retaining K/V tensors, MUST grow `memory_bytes` monotonically with each decode step, and MUST produce next-token distributions that are bit-identical (`KL = 0.0`) to the reference standard-KV decode path on supported architectures.

#### Scenario: Engine identity, info, and memory are zero before prefill
- **WHEN** a `MarkovResidualEngine` is constructed with default and fixed-window configurations
- **THEN** the engine SHALL report a non-empty name, return `EngineInfo` with the configured window, and SHALL report `memory_bytes == 0` before prefill
<!-- test: larql_inference::engines::kv_engines::markov_residual::engine::tests::engine_name -->
<!-- test: larql_inference::engines::kv_engines::markov_residual::engine::tests::engine_memory_zero_before_prefill -->
<!-- test: larql_inference::engines::kv_engines::markov_residual::engine::tests::engine_info_full_window -->
<!-- test: larql_inference::engines::kv_engines::markov_residual::engine::tests::engine_info_fixed_window -->

#### Scenario: Prefill stores residuals for every layer and decode produces finite logits
- **WHEN** the engine prefills a multi-token prompt and runs at least one decode step
- **THEN** every layer's residual store SHALL be populated, decode SHALL yield finite logits of vocab size, and `memory_bytes` SHALL grow strictly with each subsequent decode step
<!-- test: larql_inference::engines::kv_engines::markov_residual::engine::tests::prefill_stores_residuals_for_all_layers -->
<!-- test: larql_inference::engines::kv_engines::markov_residual::engine::tests::decode_step_produces_finite_logits -->
<!-- test: larql_inference::engines::kv_engines::markov_residual::engine::tests::memory_grows_with_each_decode_step -->
<!-- test: larql_inference::engines::kv_engines::markov_residual::engine::tests::multiple_decode_steps_produce_consistent_shapes -->

#### Scenario: Hot window clipping moves overflow rows to the cold tier
- **WHEN** more rows than the configured window have been written to a layer's residual store
- **THEN** `clip_layer` SHALL retain only the most recent `W` rows in hot storage, push the older rows into a cold archive, and leave a no-op when the layer is below the window
<!-- test: larql_inference::engines::kv_engines::markov_residual::engine::tests::window_clipping_limits_hot_store -->
<!-- test: larql_inference::engines::kv_engines::markov_residual::store::tests::clip_layer_no_window_is_noop -->
<!-- test: larql_inference::engines::kv_engines::markov_residual::store::tests::clip_layer_within_window_pushes_empty_cold -->
<!-- test: larql_inference::engines::kv_engines::markov_residual::store::tests::clip_layer_excess_rows_moved_to_cold -->
<!-- test: larql_inference::engines::kv_engines::markov_residual::store::tests::clip_layer_exactly_at_window_no_cold -->

#### Scenario: Memory accounting matches stored rows
- **WHEN** `memory_bytes`, `cold_bytes`, and `window_tokens` are read on a populated and on an empty residual store
- **THEN** the values SHALL match the rows actually held, the cold tier SHALL report zero before any clipping, and an empty store SHALL report zero on every metric
<!-- test: larql_inference::engines::kv_engines::markov_residual::store::tests::memory_bytes_hot_only -->
<!-- test: larql_inference::engines::kv_engines::markov_residual::store::tests::memory_bytes_empty_store_is_zero -->
<!-- test: larql_inference::engines::kv_engines::markov_residual::store::tests::cold_bytes_zero_when_no_cold -->
<!-- test: larql_inference::engines::kv_engines::markov_residual::store::tests::window_tokens_matches_stored_rows -->
<!-- test: larql_inference::engines::kv_engines::markov_residual::store::tests::window_tokens_zero_for_empty_store -->

### Requirement: Apollo engine compressed factual recall

`larql_inference::engines::kv_engines::apollo::ApolloEngine` SHALL provide a
compressed-context engine that retrieves stored boundary windows on
demand, reports `memory_bytes == 0` when no store is attached, and surfaces
whether retrieval traverses the compressed boundary path or the raw
window path. The engine MUST gracefully degrade (returning `Err`) when
asked to retrieve without an attached store.

#### Scenario: Empty engine has no store and no memory
- **WHEN** an `ApolloEngine` is constructed without attaching a store
- **THEN** it SHALL report no store, zero `memory_bytes`, zero windows in `info`, and `retrieve` SHALL return an error
<!-- test: larql_inference::engines::kv_engines::apollo::engine::tests::new_engine_has_no_store -->
<!-- test: larql_inference::engines::kv_engines::apollo::engine::tests::info_no_store_shows_zero_windows -->
<!-- test: larql_inference::engines::kv_engines::apollo::engine::tests::memory_bytes_zero_without_store -->
<!-- test: larql_inference::engines::kv_engines::apollo::engine::tests::retrieve_returns_err_when_no_store -->

#### Scenario: Attached store enables routing index and reports populated info
- **WHEN** a store is attached and the routing index is built
- **THEN** the engine SHALL report the window count in `info`, switch its path to compressed when boundaries are present and uncompressed otherwise, and `memory_bytes` SHALL be non-zero
<!-- test: larql_inference::engines::kv_engines::apollo::engine::tests::with_store_attaches_store -->
<!-- test: larql_inference::engines::kv_engines::apollo::engine::tests::build_routing_index_populates_index -->
<!-- test: larql_inference::engines::kv_engines::apollo::engine::tests::info_with_store_shows_window_count -->
<!-- test: larql_inference::engines::kv_engines::apollo::engine::tests::info_shows_compressed_path_when_boundaries_present -->
<!-- test: larql_inference::engines::kv_engines::apollo::engine::tests::info_shows_uncompressed_path_when_no_boundaries -->
<!-- test: larql_inference::engines::kv_engines::apollo::engine::tests::memory_bytes_nonzero_with_store -->

#### Scenario: Retrieval respects seeds, proximity, and top-k backfill
- **WHEN** `retrieve` is called with empty queries, single-seed queries, queries scoped to candidate windows, and queries that need backfilling to reach `top_k`
- **THEN** empty queries SHALL return empty, seed tokens SHALL be matched, proximity neighbours SHALL be included, scoped queries SHALL stay within their candidate set, and backfill SHALL extend results up to `top_k`
<!-- test: larql_inference::engines::kv_engines::apollo::engine::tests::retrieve_empty_query_returns_empty -->
<!-- test: larql_inference::engines::kv_engines::apollo::engine::tests::retrieve_seed_token_matched -->
<!-- test: larql_inference::engines::kv_engines::apollo::engine::tests::retrieve_proximity_neighbour_included -->
<!-- test: larql_inference::engines::kv_engines::apollo::engine::tests::retrieve_scoped_to_candidate_windows -->
<!-- test: larql_inference::engines::kv_engines::apollo::engine::tests::retrieve_backfills_to_top_k -->

#### Scenario: Apollo store load is fail-fast on a missing directory
- **WHEN** the Apollo store is loaded against a missing directory or compared to its default architecture configuration
- **THEN** loading a missing directory SHALL return an error and the default config SHALL match the canonical Apollo-11 baseline
<!-- test: larql_inference::engines::kv_engines::apollo::store::tests::default_arch_config_matches_apollo11 -->
<!-- test: larql_inference::engines::kv_engines::apollo::store::tests::load_missing_directory_errors -->

### Requirement: TurboQuant engine Lloyd-Max plus Walsh-Hadamard 3 or 4-bit KV

`larql_inference::engines::kv_engines::turbo_quant::TurboQuantEngine` SHALL
quantise per-token KV vectors using a Walsh-Hadamard rotation followed by a
Lloyd-Max scalar quantiser at 3 or 4 bits. The engine MUST round-trip a
unit vector with high cosine similarity, MUST preserve approximate norm,
MUST not panic on zero or identical vectors, and MUST report compression
ratios consistent with the chosen bit-width.

#### Scenario: Lloyd-Max + WHT primitives round-trip and preserve norm
- **WHEN** the WHT primitive is applied twice and the Lloyd-Max quantiser is run on a sample distribution
- **THEN** the WHT SHALL be self-inverse and norm-preserving, and the Lloyd-Max round-trip SHALL converge to a representation whose cosine with the input is close to one
<!-- test: larql_inference::engines::kv_engines::turbo_quant::rotation::tests::test_wht_self_inverse -->
<!-- test: larql_inference::engines::kv_engines::turbo_quant::rotation::tests::test_wht_preserves_norm -->
<!-- test: larql_inference::engines::kv_engines::turbo_quant::lloyd_max::tests::test_quantize_dequantize_roundtrip -->
<!-- test: larql_inference::engines::kv_engines::turbo_quant::lloyd_max::tests::test_lloyd_max_convergence -->

#### Scenario: 3-bit and 4-bit packings round-trip and report correct sizes
- **WHEN** the 4-bit and 3-bit packers encode and decode a sample buffer and the engine reports `bytes_per_vector` for several dimensions
- **THEN** packing SHALL be lossless given quantised input, sizes SHALL match the bit-width formula, and the compression ratio versus fp16 SHALL fall in the engine's documented range
<!-- test: larql_inference::engines::kv_engines::turbo_quant::packing::tests::test_4bit_roundtrip -->
<!-- test: larql_inference::engines::kv_engines::turbo_quant::packing::tests::test_3bit_roundtrip -->
<!-- test: larql_inference::engines::kv_engines::turbo_quant::packing::tests::test_4bit_packed_size -->
<!-- test: larql_inference::engines::kv_engines::turbo_quant::packing::tests::test_3bit_packed_size -->
<!-- test: larql_inference::engines::kv_engines::turbo_quant::engine::tests::bytes_per_vector_4bit_dim256 -->
<!-- test: larql_inference::engines::kv_engines::turbo_quant::engine::tests::bytes_per_vector_3bit_dim256 -->
<!-- test: larql_inference::engines::kv_engines::turbo_quant::engine::tests::bytes_per_vector_4bit_dim128 -->
<!-- test: larql_inference::engines::kv_engines::turbo_quant::engine::tests::compression_ratio_vs_fp16 -->

#### Scenario: Encoded vectors round-trip with high cosine similarity at 3 and 4 bits
- **WHEN** a 4-bit and a 3-bit encoder/decoder is exercised on synthetic vectors of various dimensions
- **THEN** the cosine similarity SHALL be near one for 4-bit, acceptable for 3-bit, the norm SHALL be approximately preserved, zero vectors SHALL not panic, and identical inputs SHALL produce identical encodings
<!-- test: larql_inference::engines::kv_engines::turbo_quant::engine::tests::encode_decode_4bit_cosine_near_one -->
<!-- test: larql_inference::engines::kv_engines::turbo_quant::engine::tests::encode_decode_3bit_cosine_acceptable -->
<!-- test: larql_inference::engines::kv_engines::turbo_quant::engine::tests::encode_decode_dim128_roundtrip -->
<!-- test: larql_inference::engines::kv_engines::turbo_quant::engine::tests::norm_approximately_preserved -->
<!-- test: larql_inference::engines::kv_engines::turbo_quant::engine::tests::zero_vector_roundtrip_no_panic -->
<!-- test: larql_inference::engines::kv_engines::turbo_quant::engine::tests::identical_vectors_same_encoding -->

#### Scenario: Engine surface integrates with the prefill/decode pipeline
- **WHEN** the engine prefills a prompt, runs decode steps, and is inspected for its summary
- **THEN** the compressed cache SHALL grow, the compressed layer memory SHALL be smaller than fp32, the resulting logits SHALL be finite, and the summary SHALL include the bit-width
<!-- test: larql_inference::engines::kv_engines::turbo_quant::engine::tests::engine_name_and_config_4bit -->
<!-- test: larql_inference::engines::kv_engines::turbo_quant::engine::tests::engine_name_and_config_3bit -->
<!-- test: larql_inference::engines::kv_engines::turbo_quant::engine::tests::engine_memory_zero_before_prefill -->
<!-- test: larql_inference::engines::kv_engines::turbo_quant::engine::tests::engine_summary_shows_bits_in_config -->
<!-- test: larql_inference::engines::kv_engines::turbo_quant::engine::tests::compressed_layer_memory_is_smaller_than_fp32 -->
<!-- test: larql_inference::engines::kv_engines::turbo_quant::engine::tests::compressed_layer_roundtrip_cosine -->
<!-- test: larql_inference::engines::kv_engines::turbo_quant::engine::tests::prefill_compresses_kv_for_all_layers -->
<!-- test: larql_inference::engines::kv_engines::turbo_quant::engine::tests::decode_step_grows_compressed_cache -->
<!-- test: larql_inference::engines::kv_engines::turbo_quant::engine::tests::logits_finite_after_prefill_and_decode -->
<!-- test: larql_inference::engines::kv_engines::turbo_quant::engine::tests::three_bit_engine_also_works -->

### Requirement: Unlimited Context engine bit-exact within window with cold archival

The `UnlimitedContextEngine` (in `larql_inference::engines::kv_engines::unlimited_context`) SHALL maintain bit-exact KV state within an active window, archive each fully-consumed window into a cold checkpoint store, expose backend identity in `EngineInfo`, and recover any closed window from its checkpoint store on demand. Window auto-close, partial windows, and explicit flush MUST all preserve archived window count and `cold_bytes` accounting.

#### Scenario: Newly constructed engine is empty and CPU-backed
- **WHEN** an `UnlimitedContextEngine` is constructed with default config
- **THEN** it SHALL report empty window state, `EngineInfo.backend == "cpu"`, the configured window size in `info.config`, and zero `window_tokens` / `cold_bytes`
<!-- test: larql_inference::engines::kv_engines::unlimited_context::engine::tests::new_engine_is_empty -->
<!-- test: larql_inference::engines::kv_engines::unlimited_context::engine::tests::engine_info_backend_is_cpu -->
<!-- test: larql_inference::engines::kv_engines::unlimited_context::engine::tests::engine_info_config_contains_window_size -->
<!-- test: larql_inference::engines::kv_engines::unlimited_context::engine::tests::window_tokens_and_cold_bytes_start_zero -->

#### Scenario: Prefill and decode return finite hidden state and grow memory
- **WHEN** the engine prefills a prompt and steps decode multiple times
- **THEN** prefill SHALL yield a finite hidden state, each decode step SHALL yield a finite hidden state, `memory_bytes` SHALL be non-zero after prefill, and the resulting logits SHALL be finite
<!-- test: larql_inference::engines::kv_engines::unlimited_context::engine::tests::prefill_returns_hidden_state -->
<!-- test: larql_inference::engines::kv_engines::unlimited_context::engine::tests::decode_step_returns_hidden_state -->
<!-- test: larql_inference::engines::kv_engines::unlimited_context::engine::tests::memory_bytes_nonzero_after_prefill -->
<!-- test: larql_inference::engines::kv_engines::unlimited_context::engine::tests::logits_from_unlimited_context_are_finite -->

#### Scenario: Window close, partial window, and flush archive cold bytes correctly
- **WHEN** the engine fills a window past capacity, processes two full windows, runs partially through a third, or is explicitly flushed
- **THEN** the archive SHALL grow exactly once per closed window, partial windows SHALL remain hot until processed, flush SHALL close the partial window, and `cold_bytes` SHALL grow only after a window closes
<!-- test: larql_inference::engines::kv_engines::unlimited_context::engine::tests::window_auto_closes_when_full -->
<!-- test: larql_inference::engines::kv_engines::unlimited_context::engine::tests::two_full_windows_archives_two -->
<!-- test: larql_inference::engines::kv_engines::unlimited_context::engine::tests::partial_window_after_process -->
<!-- test: larql_inference::engines::kv_engines::unlimited_context::engine::tests::flush_closes_partial_window -->
<!-- test: larql_inference::engines::kv_engines::unlimited_context::engine::tests::cold_bytes_grow_after_window_close -->

#### Scenario: Token archive and checkpoint store round-trip closed windows
- **WHEN** the token archive is populated and the checkpoint store saves and reloads windows
- **THEN** archive insert/retrieve and total accounting SHALL round-trip, missing keys SHALL return `None`, eviction SHALL remove a window, an empty store SHALL report `is_empty`, and total bytes SHALL scale with layer count and hidden dimension
<!-- test: larql_inference::engines::kv_engines::unlimited_context::token_archive::tests::archive_and_retrieve_roundtrip -->
<!-- test: larql_inference::engines::kv_engines::unlimited_context::token_archive::tests::total_accounting -->
<!-- test: larql_inference::engines::kv_engines::unlimited_context::token_archive::tests::retrieve_missing_returns_none -->
<!-- test: larql_inference::engines::kv_engines::unlimited_context::token_archive::tests::is_empty_on_new -->
<!-- test: larql_inference::engines::kv_engines::unlimited_context::checkpoint_store::tests::save_and_load_roundtrip -->
<!-- test: larql_inference::engines::kv_engines::unlimited_context::checkpoint_store::tests::evict_removes_window -->
<!-- test: larql_inference::engines::kv_engines::unlimited_context::checkpoint_store::tests::total_bytes_scales_with_layers_and_dim -->
<!-- test: larql_inference::engines::kv_engines::unlimited_context::checkpoint_store::tests::is_empty_on_new_store -->
<!-- test: larql_inference::engines::kv_engines::unlimited_context::checkpoint_store::tests::load_missing_returns_none -->

### Requirement: Test suite coverage

The capability's behavior SHALL be exercised by every test in the test modules listed below. New tests added to those modules SHALL be considered part of this capability's contract by default.

#### Scenario: Workspace tests bound to this capability
- **WHEN** the trace tool resolves the wildcard module references for this capability
- **THEN** every backing test SHALL resolve to a real `#[test]` function in the workspace
<!-- test: larql_inference::test_backend::**::* -->
