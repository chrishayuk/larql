## ADDED Requirements

### Requirement: TraceStore chain file format

The `larql_inference::trace::TraceStore` SHALL persist per-token activation
chains in an append-only mmap'd binary file with magic `"TRAC"`, a 64-byte
header, and `(n_layers + 1)` waypoints per token where each waypoint stores
three `f32` vectors of `hidden_size` (residual, attention delta, FFN delta).
The store MUST reject malformed inputs and MUST NOT permit partial writes
on error.

#### Scenario: Write-then-read roundtrip preserves every layer for every token
- **WHEN** a `TraceWriter` appends two complete token chains and a `TraceStore` is opened on the resulting file
- **THEN** every residual, `attn_delta`, and `ffn_delta` vector SHALL match the source bytes exactly for both tokens and all `n_layers + 1` waypoints
<!-- test: larql_inference::test_trace::test_trace_store::write_chain_read_back_exact_match -->
<!-- test: larql_inference::trace::store::tests::create_write_read_roundtrip -->
<!-- test: larql_inference::trace::store::tests::multiple_tokens_roundtrip -->

#### Scenario: Out-of-bounds and corrupt files are rejected
- **WHEN** a token, layer, or component index is read past the stored extent, or a file with a bad magic / truncated payload is opened
- **THEN** indexed reads SHALL return `None` and `TraceStore::open` SHALL return an error rather than panic
<!-- test: larql_inference::test_trace::test_trace_store::out_of_bounds_returns_none -->
<!-- test: larql_inference::trace::store::tests::out_of_bounds_returns_none -->
<!-- test: larql_inference::trace::store::tests::open_bad_magic_returns_error -->
<!-- test: larql_inference::trace::store::tests::open_truncated_trace_returns_error -->

#### Scenario: Append rejects malformed chains atomically
- **WHEN** `append_chain` is called with a chain whose length is wrong, whose layer indices are out of order, or which would partially write a token
- **THEN** the writer SHALL return an error and SHALL NOT advance the file's `n_tokens` counter
<!-- test: larql_inference::test_trace::test_trace_store::wrong_chain_length_rejected -->
<!-- test: larql_inference::trace::store::tests::wrong_chain_length_returns_error -->
<!-- test: larql_inference::trace::store::tests::out_of_order_chain_returns_error -->
<!-- test: larql_inference::trace::store::tests::write_trace_rejects_incomplete_position_without_partial_write -->

#### Scenario: Residual stream additivity holds across the trace
- **WHEN** a chain whose residuals were constructed as `residual[layer] = residual[layer-1] + attn_delta[layer] + ffn_delta[layer]` is written and read back
- **THEN** the additive identity SHALL hold within `1e-6` per dimension at every transformer layer
<!-- test: larql_inference::test_trace::test_additive_property::residual_equals_prev_plus_attn_plus_ffn -->

### Requirement: BoundaryStore window markers

`larql_inference::trace::BoundaryStore` SHALL persist one `hidden_size`
residual per window boundary alongside the boundary's `(start, length)` token
range, MUST expose `boundary_for_token(t)` mapping any covered token back to
its boundary index, and MUST reject residual writes whose length does not
match `hidden_size`.

#### Scenario: Boundaries and token ranges round-trip
- **WHEN** three boundaries are appended with distinct `(start, length, residual)` tuples
- **THEN** `n_boundaries`, `total_tokens`, `hidden_size`, `window_size`, and each indexed residual SHALL match what was written, and out-of-range indices SHALL return `None`
<!-- test: larql_inference::test_trace::test_boundary_store::append_and_read_back -->
<!-- test: larql_inference::trace::boundary::tests::create_append_open_roundtrip -->
<!-- test: larql_inference::trace::boundary::tests::multiple_boundaries_indexed_correctly -->
<!-- test: larql_inference::trace::boundary::tests::out_of_range_residual_returns_none -->

#### Scenario: Token-to-boundary lookup honours boundary spans
- **WHEN** `boundary_for_token` is queried for tokens inside, on the edge of, and past the last covered window
- **THEN** the returned index SHALL identify the correct boundary or `None` for out-of-range tokens, and `token_range(i)` SHALL return the boundary's `(start, end)` bounds
<!-- test: larql_inference::test_trace::test_boundary_store::boundary_for_token_lookup -->
<!-- test: larql_inference::test_trace::test_boundary_store::token_range -->
<!-- test: larql_inference::trace::boundary::tests::boundary_for_token_finds_correct_window -->
<!-- test: larql_inference::trace::boundary::tests::token_range_returns_correct_bounds -->

#### Scenario: Mismatched residual size is rejected
- **WHEN** `append` is called with a residual whose length is not `hidden_size`
- **THEN** the writer SHALL return an error
<!-- test: larql_inference::test_trace::test_boundary_store::size_mismatch_rejected -->
<!-- test: larql_inference::trace::boundary::tests::wrong_residual_size_returns_error -->

### Requirement: ContextStore tiered boundary capture

`larql_inference::trace::ContextStore` SHALL support three tiers
(`Residual`, `FfnDeltas`, `Full`) keyed by an explicit `critical_layers`
list. The store MUST size each boundary as `(1 + k * critical_layers.len())
* hidden_size * 4` bytes where `k` is `0`, `1`, or `2` for the three tiers,
and MUST return `None` when reading a delta tier that the file does not
carry.

#### Scenario: Tier 1 stores residuals only and rejects delta reads
- **WHEN** a `ContextStore` is created with `ContextTier::Residual` and two boundaries are appended
- **THEN** the residuals SHALL round-trip exactly, while `ffn_delta` and `attn_delta` reads SHALL return `None`
<!-- test: larql_inference::test_trace::test_context_store::tier1_residual_only -->
<!-- test: larql_inference::trace::context::tests::vectors_per_boundary_residual_is_one -->

#### Scenario: Tier 2 stores per-critical-layer FFN deltas
- **WHEN** a `ContextTier::FfnDeltas` store is written with one residual plus an `ffn_delta` per critical layer
- **THEN** every per-layer FFN delta SHALL round-trip and `attn_delta` reads SHALL return `None`
<!-- test: larql_inference::test_trace::test_context_store::tier2_residual_plus_ffn_deltas -->
<!-- test: larql_inference::trace::context::tests::vectors_per_boundary_ffn_adds_critical_layers -->

#### Scenario: Tier 3 stores residual plus FFN and attention deltas per critical layer
- **WHEN** a `ContextTier::Full` store is written with residual, FFN deltas, and attention deltas per critical layer
- **THEN** all three vector kinds SHALL round-trip with `bytes_per_boundary` equal to `(1 + 2 * n_critical) * hidden_size * 4`
<!-- test: larql_inference::test_trace::test_context_store::tier3_full_store -->
<!-- test: larql_inference::test_trace::test_context_store::bytes_per_boundary_matches_tier -->
<!-- test: larql_inference::trace::context::tests::vectors_per_boundary_full_adds_two_per_critical -->
<!-- test: larql_inference::trace::context::tests::create_open_basic_roundtrip -->

#### Scenario: Token-to-boundary lookup is consistent with the boundary store
- **WHEN** `boundary_for_token` is queried on a `ContextStore` with two boundaries
- **THEN** lookups inside each boundary SHALL return the boundary's index, and lookups past the last token SHALL return `None`
<!-- test: larql_inference::test_trace::test_context_store::context_boundary_for_token -->

#### Scenario: ContextTier u8 mapping is total
- **WHEN** `ContextTier::from_u8` is called on every valid byte and an invalid byte
- **THEN** the round-trip SHALL preserve known tiers and SHALL fall back to `Residual` on invalid input
<!-- test: larql_inference::trace::context::tests::context_tier_from_u8_roundtrip -->
<!-- test: larql_inference::trace::context::tests::context_tier_from_u8_invalid_defaults_to_residual -->

### Requirement: Decomposed trace records (TraceNode / ResidualTrace)

`larql_inference::trace::ResidualTrace` SHALL hold an in-memory list of
`TraceNode { layer, position, residual, attn_delta, ffn_delta }` entries
where `layer == -1` denotes the embedding waypoint. Lookup helpers
(`node`, `last_node`, `layer_nodes`, `position_trajectory`) MUST honour the
embedding sentinel and MUST return empty/`None` results for missing
coordinates rather than panic.

#### Scenario: TraceNode lookup honours layer and position
- **WHEN** `ResidualTrace::node(layer, position)` is queried for a present and a missing coordinate
- **THEN** present coordinates SHALL return `Some(TraceNode)` matching layer/position and missing coordinates SHALL return `None`
<!-- test: larql_inference::trace::types::tests::node_found_at_correct_layer_and_position -->
<!-- test: larql_inference::trace::types::tests::node_returns_none_for_missing_layer -->
<!-- test: larql_inference::trace::types::tests::node_returns_none_for_missing_position -->
<!-- test: larql_inference::trace::types::tests::embedding_layer_minus_one_accessible -->

#### Scenario: TraceStore reconstructs TraceNode with correct embedding layer
- **WHEN** `TraceStore::node(token, store_index)` is called for store index 0
- **THEN** the returned `TraceNode` SHALL carry `layer == -1` and SHALL match the original residual / attn / ffn vectors of the embedding waypoint
<!-- test: larql_inference::test_trace::test_trace_store::node_method_reconstructs_trace_node -->
<!-- test: larql_inference::trace::store::tests::node_accessor_reconstructs_trace_node -->

#### Scenario: Layer and position projections are sorted and total
- **WHEN** `layer_nodes(layer)` and `position_trajectory(position)` are called
- **THEN** the layer projection SHALL return every node at that layer in position order, the trajectory SHALL be sorted ascending by layer, and missing keys SHALL return empty vectors
<!-- test: larql_inference::trace::types::tests::layer_nodes_returns_all_positions_for_layer -->
<!-- test: larql_inference::trace::types::tests::layer_nodes_returns_empty_for_missing_layer -->
<!-- test: larql_inference::trace::types::tests::position_trajectory_sorted_ascending_by_layer -->
<!-- test: larql_inference::trace::types::tests::position_trajectory_empty_for_missing_position -->
<!-- test: larql_inference::trace::types::tests::last_node_returns_node_at_last_token -->
<!-- test: larql_inference::trace::types::tests::last_node_returns_none_for_missing_layer -->

### Requirement: Residual capture during the forward pass

`larql_inference::trace::capture` SHALL provide a `trace_*` family of
helpers that runs a forward pass with hooks installed and emits a
`ResidualTrace` containing the embedding waypoint plus one node per
transformer layer. The resulting nodes MUST be finite, MUST carry vectors of
length `hidden_size`, and MUST satisfy
`residual[L] == residual[L-1] + attn_delta[L] + ffn_delta[L]` to within
numerical tolerance.

#### Scenario: Tracing populates every requested position
- **WHEN** `trace_all_positions` is run on a multi-token prompt
- **THEN** every position SHALL receive `(n_layers + 1)` nodes covering the embedding and each transformer layer
<!-- test: larql_inference::trace::capture::tests::trace_all_positions_populates_nodes -->
<!-- test: larql_inference::trace::capture::tests::trace_last_position_only -->
<!-- test: larql_inference::trace::capture::tests::trace_specific_positions -->

#### Scenario: Captured nodes are finite and correctly sized
- **WHEN** a trace is captured on a small synthetic model
- **THEN** every `residual`, `attn_delta`, and `ffn_delta` SHALL have length `hidden_size` and SHALL contain only finite floats, with the embedding node carrying `layer == -1`
<!-- test: larql_inference::trace::capture::tests::trace_nodes_are_finite -->
<!-- test: larql_inference::trace::capture::tests::trace_deltas_correct_residual_len -->
<!-- test: larql_inference::trace::capture::tests::trace_embedding_layer_minus_one_present -->

#### Scenario: Captured deltas reconstruct the residual stream and match the raw forward pass
- **WHEN** a trace is captured on a forward pass and the per-layer deltas are summed onto the previous residual
- **THEN** the reconstruction SHALL match the trace's stored residuals, and the final residual SHALL equal the unhooked forward pass's pre-`lm_head` hidden state
<!-- test: larql_inference::trace::capture::tests::trace_edges_reconstruct_residuals -->
<!-- test: larql_inference::trace::capture::tests::trace_final_residual_matches_raw_forward_logits -->
<!-- test: larql_inference::trace::capture::tests::trace_custom_ffn_matches_hooked_forward_final_residual -->

### Requirement: Vocabulary embedding-neighbour lookup and differential residual analysis

`larql_inference::trace::vocab` SHALL expose vector helpers
(`vec_norm`, `project_to_logits`) for projecting a residual onto the
embedding matrix to find vocabulary neighbours.
`larql_inference::residual_diff` SHALL provide
`compare_captures` / `compare_stages` plus per-stage capture helpers that
identify the **first** stage at which two traces diverge above a configurable
cosine/abs threshold, and SHALL surface shape mismatches as hard misses.

#### Scenario: Logit projection produces a vocabulary-sized vector
- **WHEN** `project_to_logits` is called on a non-zero hidden vector against an embedding matrix of size `vocab_size × hidden_size`
- **THEN** the result SHALL have length `vocab_size`, SHALL be non-zero for non-zero input, and `vec_norm` SHALL return `0.0` for the zero vector
<!-- test: larql_inference::trace::vocab::tests::project_to_logits_returns_vocab_size_values -->
<!-- test: larql_inference::trace::vocab::tests::project_to_logits_nonzero_input_gives_nonzero_output -->
<!-- test: larql_inference::trace::vocab::tests::vec_norm_known_value -->
<!-- test: larql_inference::trace::vocab::tests::vec_norm_zero_vector -->

#### Scenario: Identical captures compare clean and divergent ones flag the first bad stage
- **WHEN** `compare_captures` is run on two identical capture sets and on a pair where one stage drifts above threshold
- **THEN** identical inputs SHALL report cosine `1.0` and zero max-abs, divergent inputs SHALL flag the first diverging stage as `first_bad`, and a loose threshold SHALL accept what a tight one rejects
<!-- test: larql_inference::residual_diff::compare::tests::identical_captures_have_cos_one_and_zero_max_abs -->
<!-- test: larql_inference::residual_diff::compare::tests::drift_above_threshold_flagged_as_first_bad -->
<!-- test: larql_inference::residual_diff::compare::tests::loose_threshold_accepts_what_tight_rejects -->
<!-- test: larql_inference::residual_diff::compare::tests::assert_clean_returns_err_with_first_bad_detail -->

#### Scenario: Shape mismatches and missing stages surface as hard misses
- **WHEN** two traces have mismatched vector shapes or one side is missing a stage
- **THEN** the comparison SHALL flag the offending stage as the first bad stage rather than silently degrade
<!-- test: larql_inference::residual_diff::compare::tests::shape_mismatch_surfaces_as_hard_miss -->
<!-- test: larql_inference::residual_diff::stages::tests::compare_stages_clean_when_all_match -->
<!-- test: larql_inference::residual_diff::stages::tests::compare_stages_first_bad_is_first_diverging -->
<!-- test: larql_inference::residual_diff::stages::tests::compare_stages_missing_stage_flags_first_bad -->
<!-- test: larql_inference::residual_diff::stages::tests::compare_stages_supports_asymmetric_names -->

#### Scenario: Last-position projection isolates the active token across stride layouts
- **WHEN** `project_to_last_position` is invoked on multi-token captures that may or may not be aligned to the stride
- **THEN** aligned stages SHALL be sliced to the last position only, and unaligned stages SHALL pass through unchanged
<!-- test: larql_inference::residual_diff::capture::tests::last_position_returns_correct_slice -->
<!-- test: larql_inference::residual_diff::capture::tests::project_to_last_position_drops_other_rows -->
<!-- test: larql_inference::residual_diff::stages::tests::project_to_last_position_slices_per_stride -->
<!-- test: larql_inference::residual_diff::stages::tests::project_to_last_position_keeps_unaligned_stages_unchanged -->

#### Scenario: Capture dump-dir helpers leave the environment clean
- **WHEN** `run_with_dump_dir` is used around a capture and `read_f32_vec` is invoked on captured artefacts
- **THEN** prior `LARQL_DUMP_DIR` environment values SHALL be restored, files with non-multiple-of-four bytes SHALL be rejected, and missing files SHALL return `None`
<!-- test: larql_inference::residual_diff::capture::tests::run_with_dump_dir_restores_prior_env -->
<!-- test: larql_inference::residual_diff::capture::tests::run_with_dump_dir_clears_when_no_prior_value -->
<!-- test: larql_inference::residual_diff::capture::tests::read_f32_vec_decodes_le_floats -->
<!-- test: larql_inference::residual_diff::capture::tests::read_f32_vec_rejects_non_multiple_of_four -->
<!-- test: larql_inference::residual_diff::capture::tests::read_f32_vec_returns_none_on_missing_file -->

### Requirement: Test suite coverage

The capability's behavior SHALL be exercised by every test in the test modules listed below. New tests added to those modules SHALL be considered part of this capability's contract by default.

#### Scenario: Workspace tests bound to this capability
- **WHEN** the trace tool resolves the wildcard module references for this capability
- **THEN** every backing test SHALL resolve to a real `#[test]` function in the workspace
<!-- test: larql_inference::test_trace::**::* -->
