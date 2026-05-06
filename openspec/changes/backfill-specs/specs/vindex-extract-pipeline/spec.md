## ADDED Requirements

### Requirement: Three extract levels gate disk footprint

The extract pipeline SHALL expose `ExtractLevel` with at least three
variants — browse (gate-only, ~3 GB), inference (gate + projections, ~6
GB), and all (full extract, ~10 GB) — and the chosen level MUST gate
which projections are materialised. The default for new builds SHALL
be `browse` so that no caller accidentally triples disk usage.

#### Scenario: ExtractLevel default is browse
- **WHEN** an `ExtractLevel` is constructed via its `Default` impl
- **THEN** the value SHALL equal the browse variant
<!-- test: larql_vindex::test_vindex::extract_level_default_is_browse -->

#### Scenario: ExtractLevel survives JSON round-trip
- **WHEN** every `ExtractLevel` variant is serialised and reparsed
- **THEN** the parsed value SHALL equal the original
<!-- test: larql_vindex::test_vindex::extract_level_serialization -->

### Requirement: Build vindex from synthetic and real model weights

The extractor SHALL build a vindex from a HuggingFace-shaped model in
both `f32` and `f16` precision and SHALL preserve enough signal that a
subsequent load + KNN query returns the inserted features. GGUF
metadata MUST be normalised so that GGUF-sourced configs flow through
the same code path as safetensors.

#### Scenario: Synthetic f32 model extracts to a queryable vindex
- **WHEN** a synthetic f32 model is run through the extract pipeline
- **THEN** the resulting vindex SHALL load and respond to gate KNN queries
<!-- test: larql_vindex::test_vindex::extract_synthetic_model_f32 -->

#### Scenario: Synthetic f16 model extracts to a queryable vindex
- **WHEN** a synthetic f16 model is run through the extract pipeline
- **THEN** the resulting vindex SHALL load and respond to gate KNN queries
<!-- test: larql_vindex::test_vindex::extract_synthetic_model_f16 -->

#### Scenario: Extract followed by load retains weights
- **WHEN** a model is extracted and the resulting vindex is reloaded with weights
- **THEN** the weight tensors SHALL match what the source model exposed
<!-- test: larql_vindex::test_vindex::extract_then_load_weights_round_trip -->

#### Scenario: GGUF metadata maps onto the same config shape
- **WHEN** a GGUF metadata blob is converted via `gguf_config_from_metadata`
- **THEN** the resulting config SHALL be structurally equivalent to a safetensors-derived config
<!-- test: larql_vindex::test_vindex::gguf_config_from_metadata -->

#### Scenario: GGUF tensor keys are normalised
- **WHEN** GGUF tensor keys are passed through the GGUF normaliser
- **THEN** the keys SHALL be rewritten into the architecture's canonical form
<!-- test: larql_vindex::test_vindex::gguf_key_normalization -->

### Requirement: Build vindex from pre-computed gate/down vectors

The pipeline SHALL allow a caller to construct a vindex directly from
pre-computed gate (and down) vectors without supplying any model
weights. This path supports incremental rebuilds, distillation
experiments, and patch refinement that needs a base vindex but not
inference-grade weights.

#### Scenario: Build-from-vectors short-circuits the safetensors loader
- **WHEN** `build_vindex_from_vectors` is invoked with synthetic gate/down arrays and no model handle
- **THEN** the resulting vindex SHALL succeed and contain the supplied vectors
<!-- test: larql_vindex::test_vindex::vindexfile_parse_and_build -->

### Requirement: Streaming extraction with quantization support

The streaming extractor SHALL read tensors incrementally from
safetensors and emit per-feature batches via callbacks, including the
`q4k` quantised path. PLE tensors and per-layer intermediate sizes for
variable-FFN models MUST be preserved in the streaming output.

#### Scenario: Streaming extract from safetensors completes
- **WHEN** `streaming_extract` runs against a synthetic safetensors model
- **THEN** the run SHALL complete and emit the full feature stream
<!-- test: larql_vindex::test_vindex::streaming_extract_from_safetensors -->

#### Scenario: Streaming extract handles q4k quantisation inline
- **WHEN** `streaming_extract` is asked to emit q4k blocks
- **THEN** the streaming output SHALL contain valid q4k blocks for every feature
<!-- test: larql_vindex::test_vindex::streaming_extract_q4k_from_safetensors -->

#### Scenario: Streaming extract carries PLE tensors
- **WHEN** a model with per-layer embeddings is streamed
- **THEN** the resulting vindex SHALL include the PLE tensors
<!-- test: larql_vindex::test_vindex::streaming_extract_q4k_carries_ple_tensors -->

#### Scenario: Variable-FFN intermediate sizes are preserved
- **WHEN** a model with non-uniform FFN intermediate sizes is streamed
- **THEN** per-layer intermediate widths SHALL survive into the manifest
<!-- test: larql_vindex::test_vindex::streaming_extract_preserves_per_layer_intermediate_for_variable_ffn -->

### Requirement: Resumable checkpoints with phase tracking

A long-running extract SHALL persist a `.extract_checkpoint.json` that
records completed phases, source-model identity, and feature offsets.
A subsequent run with a compatible checkpoint SHALL skip already-done
phases; an incompatible checkpoint MUST be discarded rather than
trusted, and `mark` operations MUST be idempotent.

#### Scenario: Missing checkpoint loads as None
- **WHEN** the checkpoint file is absent
- **THEN** the loader SHALL return `None` rather than erroring
<!-- test: larql_vindex::extract::checkpoint::tests::missing_checkpoint_loads_as_none -->

#### Scenario: Checkpoint round-trips completed phases
- **WHEN** a checkpoint with completed phases is saved and reloaded
- **THEN** the completed-phase set SHALL match the original
<!-- test: larql_vindex::extract::checkpoint::tests::round_trip_preserves_completed_phases -->

#### Scenario: Marking a phase complete is idempotent
- **WHEN** the same phase is marked complete twice
- **THEN** the resulting checkpoint SHALL match the single-mark case
<!-- test: larql_vindex::extract::checkpoint::tests::mark_is_idempotent -->

#### Scenario: Clearing the checkpoint removes the file
- **WHEN** `clear` is called on an existing checkpoint
- **THEN** the file SHALL no longer exist on disk
<!-- test: larql_vindex::extract::checkpoint::tests::clear_removes_file -->

#### Scenario: Incompatible checkpoint is rejected
- **WHEN** the checkpoint records a different source model than the current run
- **THEN** the compatibility check SHALL refuse to resume
<!-- test: larql_vindex::extract::checkpoint::tests::compatibility_rejects_different_model -->

#### Scenario: Resume after gate phase reproduces full run
- **WHEN** an extract resumes after the gate phase completed
- **THEN** the resulting vindex SHALL be byte-equal to a non-resumed reference run
<!-- test: larql_vindex::golden_resume::resume_after_gate_complete_matches_full_run -->

#### Scenario: Incompatible checkpoint forces a fresh extract
- **WHEN** a checkpoint mismatching the current model is encountered
- **THEN** the pipeline SHALL discard it and run from scratch
<!-- test: larql_vindex::golden_resume::incompatible_checkpoint_is_discarded -->

### Requirement: Stage labels and metadata aggregation

The probe-label assignment stage SHALL emit unique labels per
extraction stage so downstream tooling can attribute features. The
metadata aggregator SHALL copy only files that exist in the source and
SHALL be a no-op when the source set is empty.

#### Scenario: Stage labels are pairwise unique
- **WHEN** every stage label constant is collected
- **THEN** the set SHALL contain no duplicates
<!-- test: larql_vindex::extract::stage_labels::tests::all_labels_unique -->

#### Scenario: Metadata copy skips absent inputs
- **WHEN** the metadata stage is asked to copy an optional file that is missing
- **THEN** it SHALL skip the file without erroring
<!-- test: larql_vindex::extract::metadata::tests::copies_present_files_only -->

#### Scenario: Metadata copy is a no-op for empty source
- **WHEN** the metadata stage runs with an empty source set
- **THEN** it SHALL return without writing anything
<!-- test: larql_vindex::extract::metadata::tests::empty_source_is_noop -->

### Requirement: Test suite coverage

The capability's behavior SHALL be exercised by every test in the test modules listed below. New tests added to those modules SHALL be considered part of this capability's contract by default.

#### Scenario: Workspace tests bound to this capability
- **WHEN** the trace tool resolves the wildcard module references for this capability
- **THEN** every backing test SHALL resolve to a real `#[test]` function in the workspace
<!-- test: larql_vindex::golden_save_load::**::* -->
<!-- test: larql_vindex::golden_resume::**::* -->
<!-- test: larql_vindex::extract::checkpoint::tests::**::* -->
<!-- test: larql_vindex::extract::stage_labels::tests::**::* -->
<!-- test: larql_vindex::extract::metadata::tests::**::* -->
