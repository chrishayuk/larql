## ADDED Requirements

### Requirement: On-disk vindex directory layout

A persisted vindex SHALL be a directory whose top-level entries match the
canonical filenames defined in `larql_vindex::format::filenames`. Every
released vindex MUST contain `index.json` (the manifest) and the gate
projection (`gate_vectors.bin` for f32/f16 or the q4k/fp4 variants), and
MUST NOT mix incompatible projection encodings within a single layer.
Optional projections (up, down, attn, embeddings, lm_head, norms,
down_meta) SHALL only appear when the extract level requested them.

#### Scenario: Canonical filename set is unique
- **WHEN** every constant exposed by `larql_vindex::format::filenames` is collected
- **THEN** the set SHALL contain no duplicates so that two different artefacts cannot collide on disk
<!-- test: larql_vindex::format::filenames::tests::all_filenames_unique -->

#### Scenario: HF upload manifest is a subset of all known files
- **WHEN** the curated HuggingFace upload list is intersected with the full filename set
- **THEN** the upload list SHALL be a strict subset of the known filenames
<!-- test: larql_vindex::format::filenames::tests::hf_upload_files_subset_of_all -->

#### Scenario: Loading a vindex with no `index.json` errors
- **WHEN** `load_vindex` is invoked on a directory missing `index.json`
- **THEN** loading SHALL return a structured error rather than panicking
<!-- test: larql_vindex::format::load::tests::load_vindex_missing_index_json_errors -->

#### Scenario: Loading a missing vindex directory errors
- **WHEN** `load_vindex` is invoked on a path that does not exist
- **THEN** loading SHALL return a structured error
<!-- test: larql_vindex::format::load::tests::load_vindex_missing_dir_errors -->

### Requirement: VindexConfig manifest parsing

`index.json` SHALL deserialize into `VindexConfig` and SHALL retain
backwards compatibility with v1 manifests by supplying defaults for new
fields. v2 manifests SHALL round-trip losslessly, including MoE and
fp4-specific extensions, and the parser MUST refuse malformed JSON
rather than silently dropping fields.

#### Scenario: v1 config loads with new-field defaults
- **WHEN** a legacy v1 `index.json` is parsed
- **THEN** parsing SHALL succeed and missing fields SHALL receive their declared defaults
<!-- test: larql_vindex::test_vindex::v1_config_loads_with_defaults -->

#### Scenario: v2 manifest round-trips losslessly
- **WHEN** a fully-populated v2 `VindexConfig` is serialised to JSON and reparsed
- **THEN** the result SHALL equal the original
<!-- test: larql_vindex::test_vindex::v2_config_full_round_trip -->

#### Scenario: MoE-flavoured manifest survives round-trip
- **WHEN** a `VindexConfig` describing a MoE model is serialised and reparsed
- **THEN** expert counts, top-k, and per-layer expert metadata SHALL be preserved
<!-- test: larql_vindex::test_vindex::v2_config_with_moe -->

#### Scenario: Layer-bands field round-trips
- **WHEN** a `VindexConfig` carrying layer bands is round-tripped through JSON
- **THEN** the bands SHALL deserialize byte-equal
<!-- test: larql_vindex::test_vindex::layer_bands_config_round_trip -->

#### Scenario: Malformed manifest produces a structured error
- **WHEN** `load_vindex_config` reads a file that is not valid JSON
- **THEN** it SHALL return an error rather than panic
<!-- test: larql_vindex::format::load::tests::load_vindex_config_malformed_json_errors -->

#### Scenario: Minimal valid fixture loads
- **WHEN** the minimal vindex test fixture is loaded
- **THEN** `load_vindex` SHALL succeed and produce a `VectorIndex`
<!-- test: larql_vindex::format::load::tests::load_vindex_minimal_fixture_succeeds -->

### Requirement: SHA-256 checksums per file

Every persistable file in a vindex directory SHALL have a SHA-256 sum
recorded in the checksum manifest. `compute_checksums` MUST be
deterministic for identical content and MUST skip files that are
absent. `verify_checksums` MUST return `false` when any covered file is
missing or has been mutated.

#### Scenario: SHA-256 is deterministic across calls
- **WHEN** `sha256_file` is called twice on the same file
- **THEN** both calls SHALL return identical digests
<!-- test: larql_vindex::format::checksums::tests::sha256_file_deterministic -->

#### Scenario: Different content produces different hash
- **WHEN** `sha256_file` is called on two files with different bytes
- **THEN** the digests SHALL differ
<!-- test: larql_vindex::format::checksums::tests::sha256_file_different_content_different_hash -->

#### Scenario: Empty file has a well-defined digest
- **WHEN** `sha256_file` is invoked on a zero-byte file
- **THEN** it SHALL return the canonical SHA-256 of the empty string
<!-- test: larql_vindex::format::checksums::tests::sha256_file_empty_file -->

#### Scenario: Missing file surfaces an error
- **WHEN** `sha256_file` is called on a non-existent path
- **THEN** it SHALL return an error
<!-- test: larql_vindex::format::checksums::tests::sha256_file_missing_returns_error -->

#### Scenario: Compute pass tolerates absent optional files
- **WHEN** `compute_checksums` runs against a directory missing optional projections
- **THEN** it SHALL skip the missing files without erroring
<!-- test: larql_vindex::format::checksums::tests::compute_checksums_skips_missing_files -->

#### Scenario: Verify pass detects content drift
- **WHEN** content of a tracked file changes after checksums were computed
- **THEN** `verify_checksums` SHALL return false
<!-- test: larql_vindex::format::checksums::tests::verify_checksums_fail_when_content_changed -->

#### Scenario: Verify pass passes when content is intact
- **WHEN** files match their recorded checksums
- **THEN** `verify_checksums` SHALL return true
<!-- test: larql_vindex::format::checksums::tests::verify_checksums_pass_for_correct_content -->

#### Scenario: Verify pass treats missing files as failure
- **WHEN** a file recorded in the checksum manifest is absent on disk
- **THEN** `verify_checksums` SHALL return false
<!-- test: larql_vindex::format::checksums::tests::verify_checksums_missing_file_is_false -->

### Requirement: Weight manifest and quantization tags

The per-projection weight manifest SHALL record the on-disk format tag
(`f32`, `f16`, `q4k`, `q6k`, `fp4`, packed MXFP4) and the padded tensor
shape so that the loader can reconstruct strides without re-reading the
binaries. Missing format fields MUST be a parse error rather than a
silent default.

#### Scenario: Manifest round-trip preserves wire shape
- **WHEN** a weight manifest is serialised and reparsed
- **THEN** padded shape, format tag, and component counts SHALL match exactly
<!-- test: larql_vindex::format::weights::manifest::tests::round_trip_matches_writer_wire_shape -->

#### Scenario: Format tag matches on-disk strings
- **WHEN** each `WeightFormat` enum variant is rendered to its string tag
- **THEN** the string SHALL equal the literal recorded by the writer
<!-- test: larql_vindex::format::weights::manifest::tests::format_tag_matches_on_disk_strings -->

#### Scenario: Padded width is exposed for shape recovery
- **WHEN** a manifest with a non-uniform layer width is queried for `padded_width`
- **THEN** it SHALL return the second dimension of the padded shape
<!-- test: larql_vindex::format::weights::manifest::tests::padded_width_extracts_second_dim -->

#### Scenario: Manifest without a format field fails to parse
- **WHEN** a manifest JSON omits the `format` key
- **THEN** parsing SHALL fail rather than default silently
<!-- test: larql_vindex::format::weights::manifest::tests::missing_format_field_fails_parse -->

### Requirement: Q4K writer block-padding semantics

The Q4K weight writer SHALL pad rows to the Q4K block boundary, leaving
exact-multiple inputs unchanged and zero-filling otherwise. When V is
shared with K, the writer SHALL substitute K bytes; when both are
absent, the writer MUST decline rather than emit corrupt data.

#### Scenario: Padding is a no-op for exact multiples
- **WHEN** `pad_to_block` runs on input whose length is a multiple of the block size
- **THEN** the buffer SHALL be returned unchanged
<!-- test: larql_vindex::format::weights::write_q4k::tests::pad_to_block_noop_when_exact_multiple -->

#### Scenario: Padding fills to the next block
- **WHEN** input length is one byte below a block boundary
- **THEN** the buffer SHALL be zero-filled up to the next multiple
<!-- test: larql_vindex::format::weights::write_q4k::tests::pad_to_block_handles_one_below_multiple -->

#### Scenario: Empty input stays empty
- **WHEN** `pad_to_block` is called on an empty slice
- **THEN** the result SHALL remain empty
<!-- test: larql_vindex::format::weights::write_q4k::tests::pad_to_block_empty_input_stays_empty -->

#### Scenario: V resolution falls back to K when V is shared
- **WHEN** V is shared with K and the writer requests V bytes
- **THEN** `resolve_v` SHALL return K's bytes
<!-- test: larql_vindex::format::weights::write_q4k::tests::resolve_v_falls_back_to_k_when_v_shared -->

#### Scenario: Missing V without sharing returns None
- **WHEN** V is absent and not configured to share with K
- **THEN** `resolve_v` SHALL return `None`
<!-- test: larql_vindex::format::weights::write_q4k::tests::resolve_v_none_when_missing_and_not_shared -->

### Requirement: Golden save/load round-trip

A vindex written to disk and reloaded SHALL be byte-deterministic and
behaviourally identical to the in-memory original. The mmap path MUST
be zero-copy, and KNN ordering MUST be stable across save/load cycles.

#### Scenario: Save is bit-deterministic across runs
- **WHEN** the same in-memory vindex is saved twice with the same configuration
- **THEN** both directories SHALL contain byte-identical files
<!-- test: larql_vindex::golden_save_load::save_is_deterministic -->

#### Scenario: KNN results survive save/reload unchanged
- **WHEN** a vindex is saved, reloaded, and queried with the same probe
- **THEN** the top-K result SHALL match the pre-save baseline
<!-- test: larql_vindex::golden_save_load::knn_round_trip_preserves_results -->

#### Scenario: mmap reload remains zero-copy
- **WHEN** a saved vindex is reloaded via mmap
- **THEN** the reload SHALL not allocate a heap copy of the projection bytes
<!-- test: larql_vindex::golden_save_load::mmap_load_is_zero_copy -->

#### Scenario: Down-meta round-trips through binary writer
- **WHEN** `save_down_meta` writes binary down-meta and a reload pass parses it back
- **THEN** the records SHALL match the original feature-by-feature
<!-- test: larql_vindex::test_vindex::binary_down_meta_write_read_round_trip -->

#### Scenario: Gate vectors round-trip through saver
- **WHEN** gate vectors are saved and reloaded
- **THEN** every per-layer matrix SHALL match the original
<!-- test: larql_vindex::test_vindex::save_and_load_gate_vectors_round_trip -->

#### Scenario: Config round-trips through saver
- **WHEN** the manifest is saved and reloaded
- **THEN** the resulting `VindexConfig` SHALL equal the input
<!-- test: larql_vindex::test_vindex::save_config_round_trip -->

### Requirement: Test suite coverage

The capability's behavior SHALL be exercised by every test in the test modules listed below. New tests added to those modules SHALL be considered part of this capability's contract by default.

#### Scenario: Workspace tests bound to this capability
- **WHEN** the trace tool resolves the wildcard module references for this capability
- **THEN** every backing test SHALL resolve to a real `#[test]` function in the workspace
<!-- test: larql_vindex::test_vindex::**::* -->
<!-- test: larql_vindex::format::filenames::tests::**::* -->
<!-- test: larql_vindex::format::checksums::tests::**::* -->
<!-- test: larql_vindex::format::load::tests::**::* -->
<!-- test: larql_vindex::config::index::**::* -->
<!-- test: larql_vindex::config::dtype::tests::**::* -->
