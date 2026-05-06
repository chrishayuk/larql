## ADDED Requirements

### Requirement: Sparse Walk FFN forward pass

The `larql_inference::vindex::walk_ffn` module SHALL provide a
`WalkFfn` FFN backend that replaces the dense down projection with a
zero-copy mmap'd read from `down_features.bin` (per ADR-002). The
backend SHALL select active features via a gate-KNN top-k lookup
(default K ≈ 10) and SHALL skip the remaining features per layer so
that, on Gemma 4B's 10,240-feature layers, only a small fraction of
features participate in any given token. When `top_k == 0` (no active
features), the backend MUST fall back to the dense `WeightFfn` path
so that FFN output is never silently zeroed.

#### Scenario: Walk FFN preserves shape across token counts and layers
- **WHEN** `WalkFfn::forward` is invoked on single-token, multi-token, and per-layer inputs
- **THEN** the output SHALL have `[seq, hidden]` shape on every call, every layer SHALL succeed, and the sparse path SHALL produce the same shape as the dense reference
<!-- test: larql_inference::vindex::walk_ffn::mod::tests::walk_ffn_forward_shape_single_token -->
<!-- test: larql_inference::vindex::walk_ffn::mod::tests::walk_ffn_forward_shape_multi_token -->
<!-- test: larql_inference::vindex::walk_ffn::mod::tests::walk_ffn_forward_all_layers -->
<!-- test: larql_inference::vindex::walk_ffn::mod::tests::walk_ffn_sparse_vs_dense_same_shape -->

#### Scenario: with_activation returns both activation and output
- **WHEN** `WalkFfn::forward_with_activation` is invoked
- **THEN** both the gate activation and the FFN output SHALL be returned and SHALL have the documented shapes
<!-- test: larql_inference::vindex::walk_ffn::mod::tests::walk_ffn_with_activation_returns_activation -->

#### Scenario: Zero-active-features fall back to WeightFfn
- **WHEN** the gate KNN returns zero entries for a layer
- **THEN** Walk FFN SHALL delegate to the dense `WeightFfn` so output is correct, never a zero matrix
<!-- test: larql_inference::vindex::walk_ffn::mod::tests::walk_ffn_zero_features_falls_back_to_weight_ffn -->

#### Scenario: WalkFfn integrates with a configured compute backend
- **WHEN** `WalkFfn` is built with an explicit compute backend
- **THEN** `forward` SHALL run end-to-end on that backend without panicking
<!-- test: larql_inference::vindex::walk_ffn::mod::tests::walk_ffn_with_backend -->

### Requirement: WalkConfig parameters

The `larql_inference::vindex::walk_config::WalkConfig` SHALL expose
the parameters that govern sparse FFN evaluation: a per-layer or
global `top_k`, an activation `threshold`, and an optional
`layer_subset` so that callers can drive Walk FFN on a subset of
layers. `WalkConfig::default()` SHALL produce a configuration that is
safe to use without further customisation, and a freshly-constructed
`WalkFfn` SHALL report a per-layer `top_k` that matches the
configuration (or the documented "unlimited" sentinel when no cap
applies).

#### Scenario: Default WalkConfig produces a usable backend
- **WHEN** `WalkConfig::default()` is used to build a `WalkFfn`
- **THEN** the constructor SHALL succeed and the backend SHALL run forward without panicking
<!-- test: larql_inference::vindex::walk_ffn::mod::tests::walk_ffn_new_unlimited -->
<!-- test: larql_inference::vindex::walk_ffn::mod::tests::walk_ffn_sparse_k -->

### Requirement: Sparse compute primitives

The `larql_inference::ffn::sparse_compute` module SHALL expose
zero-copy primitives that the Walk FFN backend composes: feature
gathering, top-k selection, activation thresholding, and overrides
that replace the down contribution from a specific feature. The
sparse forward MUST agree with a dense reference when `top_k`
covers all features, MUST return zeros when no features are
selected, MUST sort selected features by activation, and MUST honour
overrides.

#### Scenario: Sparse forward reproduces dense when top_k covers everything
- **WHEN** `sparse_forward` is invoked with `top_k == intermediate_size`
- **THEN** the output SHALL match the dense FFN baseline
<!-- test: larql_inference::ffn::sparse_compute::tests::sparse_forward_all_features_matches_dense_fallback -->

#### Scenario: Sparse forward returns zeros for empty active set
- **WHEN** `sparse_forward` is invoked with no active features
- **THEN** the output SHALL be all zeros and shape SHALL be preserved
<!-- test: larql_inference::ffn::sparse_compute::tests::sparse_forward_empty_features_returns_zeros -->

#### Scenario: Top-k selection is sorted and respects k
- **WHEN** `sparse_forward` is invoked with a `top_k` smaller than the active set
- **THEN** exactly `top_k` features SHALL contribute and they SHALL be selected by descending activation
<!-- test: larql_inference::ffn::sparse_compute::tests::sparse_forward_top_k_selection_is_sorted -->
<!-- test: larql_inference::ffn::sparse_compute::tests::sparse_forward_top_k_respects_k -->
<!-- test: larql_inference::ffn::sparse_compute::tests::sparse_forward_single_feature_output_shape -->
<!-- test: larql_inference::ffn::sparse_compute::tests::sparse_forward_multi_token_shape -->

#### Scenario: Override replaces down contribution
- **WHEN** an override vector is supplied for a feature
- **THEN** the sparse forward SHALL replace that feature's down contribution with the override
<!-- test: larql_inference::ffn::sparse_compute::tests::overrides_replace_down_contribution -->
<!-- test: larql_inference::ffn::sparse_compute::tests::gather_rows_all_features_produces_correct_shape -->

### Requirement: Sparse FFN backend wrapper

`SparseFfn` SHALL implement the same `FfnBackend` trait as `WeightFfn`
and `WalkFfn`, so that the forward pass can swap backends without
changing the layer code. When configured with `top_k` greater than
the layer's intermediate size, the backend MUST fall back to dense
computation rather than error.

#### Scenario: SparseFfn behaves as a drop-in FfnBackend
- **WHEN** `SparseFfn::forward` and `forward_with_activation` are invoked
- **THEN** outputs SHALL preserve `[seq, hidden]` shape across single-token and multi-token inputs and across all layers, and the backend SHALL report a stable `name`
<!-- test: larql_inference::ffn::sparse::tests::sparse_ffn_name -->
<!-- test: larql_inference::ffn::sparse::tests::sparse_ffn_forward_shape_single_token -->
<!-- test: larql_inference::ffn::sparse::tests::sparse_ffn_forward_shape_multi_token -->
<!-- test: larql_inference::ffn::sparse::tests::sparse_ffn_forward_all_layers -->
<!-- test: larql_inference::ffn::sparse::tests::sparse_ffn_with_activation_returns_correct_shapes -->

#### Scenario: top_k larger than intermediate_size falls back to dense
- **WHEN** `SparseFfn` is configured with `top_k > intermediate_size`
- **THEN** the backend SHALL transparently fall back to a dense forward without erroring
<!-- test: larql_inference::ffn::sparse::tests::sparse_ffn_top_k_gt_intermediate_falls_back_to_dense -->

### Requirement: Walker utilities, gate index, and Q4K integration

The `walker` and graph-backend modules SHALL provide the support
machinery that Walk FFN depends on — top-k selection over feature
columns, threshold counting, weight extraction, gate-index build /
save / load, vector extraction for FFN-down and embeddings, and a
template-universe path that combines the gate index with a real
vindex. The integrated path SHALL reach prefill, match the dense
predict path on hidden state, and dispatch correctly through the Q4K
generate loop. The `down_features.bin` mmap SHALL be the source of
truth for the down projection, preserving the ADR-002 sparse-beats-dense
invariant (517ms vs 535ms on Gemma 4B) by design.

#### Scenario: Walker utility helpers behave as documented
- **WHEN** `partial_top_k`, `partial_top_k_column`, `top_entities`, `count_threshold`, `round4`, and `current_date` are exercised across edge cases
- **THEN** every helper SHALL return the documented result, including empty inputs, k larger than data, k zero, and threshold buckets that fire only on high values
<!-- test: larql_inference::test_walker_utils::test_partial_top_k_basic -->
<!-- test: larql_inference::test_walker_utils::test_partial_top_k_k_larger_than_data -->
<!-- test: larql_inference::test_walker_utils::test_partial_top_k_empty -->
<!-- test: larql_inference::test_walker_utils::test_partial_top_k_k_zero -->
<!-- test: larql_inference::test_walker_utils::test_partial_top_k_column -->
<!-- test: larql_inference::test_walker_utils::test_top_entities -->
<!-- test: larql_inference::test_walker_utils::test_top_entities_empty -->
<!-- test: larql_inference::test_walker_utils::test_count_threshold -->
<!-- test: larql_inference::test_walker_utils::test_round4 -->
<!-- test: larql_inference::test_walker_utils::test_current_date_format -->
<!-- test: larql_inference::walker::utils::tests::partial_top_k_returns_k_items_in_desc_order -->
<!-- test: larql_inference::walker::utils::tests::partial_top_k_zero_k_returns_empty -->
<!-- test: larql_inference::walker::utils::tests::partial_top_k_k_larger_than_data_returns_all_sorted -->
<!-- test: larql_inference::walker::utils::tests::partial_top_k_empty_input_returns_empty -->
<!-- test: larql_inference::walker::utils::tests::partial_top_k_column_extracts_correct_column -->
<!-- test: larql_inference::walker::utils::tests::partial_top_k_column_k_zero_returns_empty -->

#### Scenario: Weight, attention, and vector walkers extract from a real-shape model
- **WHEN** `WeightWalker`, `AttentionWalker`, and `VectorExtractor` run against a synthesised safetensors model
- **THEN** every walker SHALL load successfully, extract per-layer edges and stats, and return FFN-down / embedding vectors of the documented shape
<!-- test: larql_inference::test_walkers::walker_tests::test_weight_walker_loads -->
<!-- test: larql_inference::test_walkers::walker_tests::test_weight_walker_extracts_edges -->
<!-- test: larql_inference::test_walkers::walker_tests::test_weight_walker_all_layers -->
<!-- test: larql_inference::test_walkers::walker_tests::test_weight_walker_layer_stats -->
<!-- test: larql_inference::test_walkers::walker_tests::test_attention_walker_loads -->
<!-- test: larql_inference::test_walkers::walker_tests::test_attention_walker_extracts_edges -->
<!-- test: larql_inference::test_walkers::walker_tests::test_vector_extractor_ffn_down -->
<!-- test: larql_inference::test_walkers::walker_tests::test_vector_extractor_embeddings -->
<!-- test: larql_inference::test_walkers::walker_tests::test_loader_loads_all_tensors -->
<!-- test: larql_inference::test_walkers::walker_tests::test_loader_missing_directory -->

#### Scenario: Gate index supports build, lookup, and round-trip
- **WHEN** the gate index is built for a subset of layers, queried by token, then saved and reloaded
- **THEN** the index SHALL contain non-zero entries, lookup SHALL respect `top_k` and the layer set, unknown layers and out-of-range tokens SHALL return empty, and a save/load round-trip SHALL preserve structure
<!-- test: larql_inference::ffn::graph_backend::tests::build_indexes_requested_layers -->
<!-- test: larql_inference::ffn::graph_backend::tests::total_entries_non_zero -->
<!-- test: larql_inference::ffn::graph_backend::tests::build_empty_layers_is_empty -->
<!-- test: larql_inference::ffn::graph_backend::tests::lookup_from_tokens_returns_at_most_top_k -->
<!-- test: larql_inference::ffn::graph_backend::tests::lookup_from_tokens_unknown_layer_returns_empty -->
<!-- test: larql_inference::ffn::graph_backend::tests::lookup_from_tokens_empty_scores_returns_empty -->
<!-- test: larql_inference::ffn::graph_backend::tests::lookup_from_tokens_out_of_range_token_skipped -->
<!-- test: larql_inference::ffn::graph_backend::tests::precompute_entity_has_features_for_known_token -->
<!-- test: larql_inference::ffn::graph_backend::tests::save_load_roundtrip_preserves_structure -->

#### Scenario: Walk FFN integrates with a real Q4K vindex
- **WHEN** the layer-graph integration harness runs against a real Q4K vindex
- **THEN** prefill SHALL produce a finite KV-bearing residual, prefill_with_kv SHALL match the dense `predict_q4k_hidden` path, the template universe SHALL build, the guided-walk layer graph SHALL traverse end-to-end, and Q4K generate SHALL produce tokens
<!-- test: larql_inference::test_layer_graph_integration::prefill_with_kv_shape_and_finiteness -->
<!-- test: larql_inference::test_layer_graph_integration::prefill_with_kv_matches_predict_q4k_hidden -->
<!-- test: larql_inference::test_layer_graph_integration::template_universe_build_with_real_model -->
<!-- test: larql_inference::test_layer_graph_integration::guided_walk_layer_graph_with_real_universe -->
<!-- test: larql_inference::test_layer_graph_integration::detect_template_with_real_token_prefix -->
<!-- test: larql_inference::test_generate_q4k_cpu::generate_q4k_cpu_produces_tokens_against_real_vindex -->

### Requirement: Test suite coverage

The capability's behavior SHALL be exercised by every test in the test modules listed below. New tests added to those modules SHALL be considered part of this capability's contract by default.

#### Scenario: Workspace tests bound to this capability
- **WHEN** the trace tool resolves the wildcard module references for this capability
- **THEN** every backing test SHALL resolve to a real `#[test]` function in the workspace
<!-- test: larql_inference::test_walkers::**::* -->
<!-- test: larql_inference::test_walker_utils::**::* -->
<!-- test: larql_inference::test_generate_q4k_cpu::**::* -->
<!-- test: larql_inference::walker::utils::tests::**::* -->
<!-- test: larql_inference::vindex::l1_cache::tests::**::* -->
