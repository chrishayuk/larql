## ADDED Requirements

### Requirement: Lazy bootstrap of vindex artifacts

`load_single_vindex` SHALL load only the artifacts required for the
configured serving mode and SHALL defer optional artifacts (attention
weights, lm_head, interleaved Q4K, full embedding store) until the
mode demands them. Browse-only mode SHALL load gate vectors,
embeddings, and `down_meta` eagerly; inference weights MUST load
lazily on first INFER request via `LoadedModel::get_or_load_weights`.
The loader SHALL respect `--no-infer`, `--ffn-only`, and
`--embed-only` flags so that unused tensors never reach mmap.

#### Scenario: ffn-service skips attention and embed warmup
- **WHEN** the server is launched with `--ffn-only`
- **THEN** `LoadVindexOptions::ffn_only` SHALL be `true`, attention weights and lm_head SHALL be skipped at load time, and gate-vector warmup SHALL be replaced with lazy per-layer decode on first request
<!-- test: larql_server::bootstrap::tests::load_options_are_copyable -->
<!-- test: larql_server::state::loaded_model_tests::weights_not_loaded_by_default -->

#### Scenario: Q4K vindexes select the q4k weight branch
- **WHEN** a `LoadedModel` is constructed from a vindex whose `config.quant == QuantFormat::Q4K`
- **THEN** the q4k branch (`load_model_weights_q4k` + `q4k_ffn_forward_layer`) SHALL be selected by both `get_or_load_weights` and `run_full_output`, and `weights` SHALL stay un-initialised until the first request
<!-- test: larql_server::state::loaded_model_tests::quant_format_selects_q4k_branch -->
<!-- test: larql_server::state::loaded_model_tests::weights_not_loaded_by_default -->

#### Scenario: release-mmap flag round-trips into LoadedModel
- **WHEN** a server is launched with `--release-mmap-after-request`
- **THEN** `LoadedModel::release_mmap_after_request` SHALL be `true`, persisting unchanged so the walk-ffn handler can issue `madvise(MADV_DONTNEED)` post-request
<!-- test: larql_server::state::loaded_model_tests::release_mmap_flag_round_trips_true -->
<!-- test: larql_server::state::loaded_model_tests::release_mmap_flag_round_trips_false -->

### Requirement: Layer-range and expert-range sharding at load time

The loader SHALL accept `--layers START-END` and `--experts START-END`
inclusive ranges (parsed by `parse_layer_range`) and propagate them
through `LoadVindexOptions::layer_range` and
`LoadVindexOptions::expert_filter`. Reversed or malformed ranges MUST
be rejected before any vindex bytes are touched. A `--units` JSON
manifest SHALL be parseable by `parse_unit_manifest` into the canonical
`HashSet<(layer, expert)>` ownership set; ill-formed manifests MUST
return errors that name the offending file path and offending key.

#### Scenario: Inclusive CLI ranges parse with end+1 semantics
- **WHEN** `parse_layer_range("0-19")` is called
- **THEN** the result SHALL be `Ok((0, 20))`, and reversed (`"3-2"`) or non-numeric forms SHALL return `Err`
<!-- test: larql_server::bootstrap::tests::parse_layer_range_accepts_inclusive_cli_range -->
<!-- test: larql_server::bootstrap::tests::parse_layer_range_rejects_bad_shapes -->

#### Scenario: Unit manifest expands per-layer ranges into ownership set
- **WHEN** `parse_unit_manifest` is given a JSON manifest with `{"0":[[0,2]], "3":[[5,7],[10,10]]}`
- **THEN** the resulting set SHALL contain exactly `(0,0), (0,1), (0,2), (3,5), (3,6), (3,7), (3,10)`
<!-- test: larql_server::bootstrap::tests::parse_unit_manifest_round_trips_per_layer_ranges -->
<!-- test: larql_server::bootstrap::tests::parse_unit_manifest_accepts_empty_object -->

#### Scenario: Malformed unit manifest reports actionable errors
- **WHEN** the manifest contains a non-numeric layer key, a reversed range, or the file does not exist
- **THEN** `parse_unit_manifest` SHALL return an error message that includes the offending key/range or the file path
<!-- test: larql_server::bootstrap::tests::parse_unit_manifest_rejects_non_numeric_layer_key -->
<!-- test: larql_server::bootstrap::tests::parse_unit_manifest_rejects_reversed_range -->
<!-- test: larql_server::bootstrap::tests::parse_unit_manifest_missing_file_reports_path -->

### Requirement: Multi-model directory discovery and naming

When `--dir <DIR>` is supplied, `discover_vindexes` SHALL enumerate
subdirectories that contain an `index.json` file and return them in
sorted order so endpoint paths are deterministic. Each loaded model
SHALL receive an `id` derived by `model_id_from_name` (the last path
segment of `config.model`); the resulting `AppState::is_multi_model`
flag SHALL govern whether the router mounts the single-model or
multi-model URL prefix.

#### Scenario: Discovery returns sorted directories with index.json
- **WHEN** `discover_vindexes` is called on a directory containing two valid vindex subdirs (`a.vindex`, `b.vindex`) and one without `index.json`
- **THEN** the return value SHALL be `[a.vindex, b.vindex]` in lexicographic order, and the directory missing `index.json` SHALL be excluded
<!-- test: larql_server::bootstrap::tests::discover_vindexes_returns_sorted_dirs_with_index_json -->

#### Scenario: Model IDs strip directory prefixes from HF names
- **WHEN** `model_id_from_name` is called on `"google/gemma-3-4b-it"`, `"google/foo/bar"`, or `"trailing/"`
- **THEN** it SHALL return the last non-empty path segment for non-trailing forms and behave deterministically for trailing-slash inputs
<!-- test: larql_server::test_unit_state::test_model_id_from_name_no_slash -->
<!-- test: larql_server::test_unit_state::test_model_id_from_name_single_slash -->
<!-- test: larql_server::test_unit_state::test_model_id_from_name_deep_path -->
<!-- test: larql_server::test_unit_state::test_model_id_from_name_trailing_slash -->

#### Scenario: AppState routes lookups through model id
- **WHEN** `AppState::model(Some("id"))` is called on a multi-model state
- **THEN** the model with the matching id SHALL be returned, unknown ids SHALL return `None`, and `model(None)` on multi-model state SHALL also return `None` (forcing the caller to disambiguate)
<!-- test: larql_server::test_unit_state::test_app_state_model_with_id_finds_correct -->
<!-- test: larql_server::test_unit_state::test_app_state_model_unknown_id_returns_none -->
<!-- test: larql_server::test_unit_state::test_app_state_model_multi_none_returns_none -->
<!-- test: larql_server::test_unit_state::test_app_state_model_single_none_returns_first -->
<!-- test: larql_server::test_unit_state::test_app_state_is_multi_model_single -->
<!-- test: larql_server::test_unit_state::test_app_state_is_multi_model_multi -->

### Requirement: HuggingFace path resolution and probe label loading

`load_single_vindex` SHALL resolve any `hf://` path (detected via
`larql_vindex::is_hf_path`) through `larql_vindex::resolve_hf_vindex`
before any bytes are read so the download/cache step is logged
before warmup. Local filesystem paths MUST bypass HF resolution
entirely. `load_probe_labels` SHALL load probe-confirmed
`(layer, feature) → relation` labels from `feature_labels.json`
when present and SHALL return an empty map for missing, malformed,
or non-object JSON without panicking.

#### Scenario: Probe labels load from feature_labels.json when present
- **WHEN** `load_probe_labels` is called against a vindex containing a `feature_labels.json` mapping `(layer, feature) → relation`
- **THEN** every probe-confirmed entry SHALL appear in the returned map, missing/malformed/non-object JSON SHALL return an empty map without panicking
<!-- test: larql_server::test_unit_state::test_load_probe_labels_from_json_file -->
<!-- test: larql_server::test_unit_state::test_load_probe_labels_missing_file_returns_empty -->
<!-- test: larql_server::test_unit_state::test_load_probe_labels_malformed_json_returns_empty -->
<!-- test: larql_server::test_unit_state::test_load_probe_labels_non_object_json_returns_empty -->

### Requirement: f16 embedding store for embed-service mode

`EmbedStoreF16::open` SHALL mmap `embeddings.bin` only when the file
size matches `vocab_size × hidden_size × 2` bytes; mismatched or
missing files SHALL return an error so the loader can fall back to the
heap f32 copy. `EmbedStoreF16::lookup` SHALL decode rows on demand,
populate the L1 cache up to `l1_cap`, and reject token ids that
exceed `vocab_size`. f16→f32 decoding MUST handle zero, normalised,
subnormal, infinity, and NaN bit patterns correctly.

#### Scenario: Open rejects missing or wrong-size files
- **WHEN** `EmbedStoreF16::open` is called against a directory without `embeddings.bin` or one whose size does not match the f16 expectation
- **THEN** an `Err` SHALL be returned so the caller can fall back to the f32 heap copy without panicking
<!-- test: larql_server::embed_store::tests::open_rejects_missing_file -->
<!-- test: larql_server::embed_store::tests::open_rejects_wrong_size -->

#### Scenario: Lookup decodes f16 rows, applies scale, and bounds-checks
- **WHEN** `EmbedStoreF16::lookup(token_id)` is called with a valid id, then with an id ≥ `vocab_size`
- **THEN** the valid call SHALL return a scaled f32 row that is cached in L1 up to `l1_cap`, and the out-of-range id SHALL return `Err`
<!-- test: larql_server::embed_store::tests::lookup_decodes_scales_and_caches_until_cap -->
<!-- test: larql_server::embed_store::tests::lookup_rejects_out_of_range_token -->

#### Scenario: f16-to-f32 decode handles edge bit patterns
- **WHEN** the inline `f16_to_f32` decoder is applied to zero, one, negative two, subnormal, infinity, and NaN halves
- **THEN** the resulting f32 values SHALL match IEEE-754 binary16 semantics within round-trip tolerance for finite values
<!-- test: larql_server::embed_store::tests::f16_to_f32_zero -->
<!-- test: larql_server::embed_store::tests::f16_to_f32_one -->
<!-- test: larql_server::embed_store::tests::f16_to_f32_neg_two -->
<!-- test: larql_server::embed_store::tests::f16_to_f32_roundtrip_approx -->
<!-- test: larql_server::embed_store::tests::f16_to_f32_subnormal_inf_and_nan -->

### Requirement: Test suite coverage

The capability's behavior SHALL be exercised by every test in the test modules listed below. New tests added to those modules SHALL be considered part of this capability's contract by default.

#### Scenario: Workspace tests bound to this capability
- **WHEN** the trace tool resolves the wildcard module references for this capability
- **THEN** every backing test SHALL resolve to a real `#[test]` function in the workspace
<!-- test: larql_server::test_unit_vindex::**::* -->
<!-- test: larql_server::test_unit_state::**::* -->
<!-- test: larql_server::bootstrap::tests::**::* -->
<!-- test: larql_server::state::**::* -->
<!-- test: larql_server::embed_store::tests::**::* -->
