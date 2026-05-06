## ADDED Requirements

### Requirement: Safetensors loading and dtype conversion

`larql_models::loading::safetensors` SHALL load model weights from one or
more `*.safetensors` shards under a model directory, mmap each shard,
parse the JSON header, and materialise every supported tensor as a
`WeightArray` whose values are `f32`. F32 tensors SHALL be passed
through bit-for-bit; F16 and BF16 tensors MUST be decoded via
`larql_models::quant::half`. Tensors whose dtype is unsupported (for
example I64 attention masks or U8 token-type IDs) SHALL be recorded in
`ModelWeights::skipped_tensors` rather than aborting the load.

#### Scenario: F32 safetensors values load with bit-exact magnitudes
- **WHEN** `load_model_dir` reads a safetensors shard whose tensors are stored as F32
- **THEN** the resulting `WeightArray` SHALL contain the exact float values written to disk
<!-- test: larql_models::test_loading::load_f32_tensors_correct_values -->

#### Scenario: F16 safetensors tensors are decoded to f32
- **WHEN** a safetensors shard contains F16 tensors
- **THEN** loading SHALL decode each pair of bytes via `decode_f16` and store the results as `f32`
<!-- test: larql_models::test_loading::load_f16_tensors_converts_to_f32 -->

#### Scenario: BF16 safetensors tensors are decoded to f32
- **WHEN** a safetensors shard contains BF16 tensors
- **THEN** loading SHALL decode each pair of bytes via `decode_bf16` and store the results as `f32`
<!-- test: larql_models::test_loading::load_bf16_tensors_converts_to_f32 -->

#### Scenario: Unsupported dtypes flow into skipped_tensors
- **WHEN** a safetensors shard contains a tensor whose dtype is not currently supported
- **THEN** loading SHALL succeed and the tensor SHALL appear in `ModelWeights::skipped_tensors` as `(key, dtype_string)`
<!-- test: larql_models::test_loading::unsupported_dtype_goes_to_skipped_tensors -->

#### Scenario: Validated loader fails fast on invalid configs
- **WHEN** `load_model_dir_validated` is called against a directory whose `config.json` violates an architecture invariant
- **THEN** loading SHALL return an error before any tensor data is decoded
<!-- test: larql_models::test_loading::load_model_dir_validated_rejects_invalid_config -->
<!-- test: larql_models::test_loading::load_model_dir_walk_only_validated_rejects_invalid_config -->

### Requirement: GGUF parsing, dequantization, and key translation

`larql_models::loading::gguf` SHALL parse the GGUF v3 header (magic,
version, tensor count, metadata count), translate metadata into a
synthesised `ModelConfig`, normalise tensor keys into the canonical
HuggingFace form, and dequantise each supported tensor type to f32.
GGUF dimensions stored as `[cols, rows]` MUST be reshaped to the
standard `[rows, cols]` row-major layout before insertion. The loader
MUST reject truncated tensor data with a parse error rather than
panicking on a slice OOB.

#### Scenario: GGUF block keys are translated to HuggingFace keys
- **WHEN** `normalize_gguf_key` is called on a GGUF tensor key such as `blk.0.attn_q.weight`
- **THEN** the result SHALL match the canonical safetensors form (for example `layers.0.self_attn.q_proj.weight`)
<!-- test: larql_models::loading::gguf::test_normalize_gguf_key -->

#### Scenario: GGUF tensor dimensions are swapped to rows-by-cols
- **WHEN** GGUF tensor data is loaded for a 2D tensor stored as `[cols, rows]`
- **THEN** the resulting `WeightArray` SHALL have shape `[rows, cols]` and the values SHALL match the row-major reshape
<!-- test: larql_models::loading::gguf::test_load_tensors_swaps_gguf_2d_dims_to_rows_cols -->

#### Scenario: Truncated GGUF tensor data is rejected, not panicked on
- **WHEN** GGUF tensor metadata declares more bytes than the file actually contains
- **THEN** loading SHALL return a parse error
<!-- test: larql_models::loading::gguf::test_load_tensors_rejects_truncated_tensor_data -->

#### Scenario: GGUF metadata maps to ModelConfig with arch-aware overrides
- **WHEN** Gemma 4 GGUF metadata is converted into a `config.json` document
- **THEN** the architecture and `head_dim` SHALL be set per the Gemma 4 spec
<!-- test: larql_models::loading::gguf::test_gemma4_gguf_to_config_json_maps_arch_and_overrides_head_dim -->

#### Scenario: Missing optional GGUF metadata uses architecture defaults
- **WHEN** a GGUF file omits `{arch}.rope.freq_base`
- **THEN** the synthesised config SHALL omit `rope_base` so the architecture default applies, rather than encoding an explicit zero
<!-- test: larql_models::loading::gguf::test_gguf_to_config_json_omits_absent_rope_base_for_arch_default -->

#### Scenario: GGUF loads end-to-end through load_model_dir
- **WHEN** `load_model_dir` is called on a directory containing a single `*.gguf` file
- **THEN** the loader SHALL detect the GGUF file, parse and dequantise its tensors, and return a populated `ModelWeights`
<!-- test: larql_models::test_loading::load_gguf_via_load_model_dir -->
<!-- test: larql_models::test_loading::load_gguf_single_file -->

#### Scenario: GGUF 1D norm tensors land in vectors
- **WHEN** a GGUF tensor describes a 1D norm vector
- **THEN** it SHALL be inserted into `ModelWeights::vectors` rather than `tensors`
<!-- test: larql_models::test_loading::gguf_vectors_map_includes_1d_norms -->

#### Scenario: Largest GGUF shard wins when multiple are present
- **WHEN** a directory contains multiple `*.gguf` files
- **THEN** the loader SHALL pick the largest one and load tensors from it
<!-- test: larql_models::test_loading::load_gguf_prefers_largest_file_when_multiple -->

### Requirement: Architecture-driven prefix stripping

The safetensors loader SHALL normalise tensor keys by stripping the
architecture-specific prefixes returned from
`ModelArchitecture::key_prefixes_to_strip()`. The loader SHALL try the
prefixes in declaration order and SHALL strip only the first matching
prefix. Keys that do not match any prefix MUST pass through unchanged.

#### Scenario: First matching prefix wins
- **WHEN** `normalize_key` is invoked on a key whose first prefix matches
- **THEN** only that prefix SHALL be stripped and the rest of the key SHALL be unchanged
<!-- test: larql_models::loading::safetensors::normalize_key_strips_first_matching_prefix -->

#### Scenario: Falls through to a shorter prefix when the longer one does not match
- **WHEN** `normalize_key` is invoked with a list whose first prefix does not match but a later, shorter one does
- **THEN** the loader SHALL strip the matching shorter prefix
<!-- test: larql_models::loading::safetensors::normalize_key_falls_through_to_shorter_prefix -->

#### Scenario: Keys with no matching prefix pass through unchanged
- **WHEN** `normalize_key` is invoked on a key that matches none of the prefixes
- **THEN** the original key SHALL be returned untouched
<!-- test: larql_models::loading::safetensors::normalize_key_no_match_passthrough -->

#### Scenario: Empty prefix list short-circuits to passthrough
- **WHEN** `normalize_key` is invoked with an empty prefix slice
- **THEN** the original key SHALL be returned without inspecting any prefixes
<!-- test: larql_models::loading::safetensors::normalize_key_empty_prefixes -->

### Requirement: Walk-only and filtered loading skip FFN tensors at parse time

Walk-only and filtered loaders SHALL exclude FFN tensors at parse and
dequantise time so that no f32 buffer is ever allocated for those
tensors. The walk-only filter MUST cover standard FFN naming
(`gate_proj`, `up_proj`, `down_proj`, `ffn_gate`, `ffn_up`,
`ffn_down`), StarCoder2-style `mlp.c_fc` and `mlp.c_proj`, and packed
MoE blocks (GPT-OSS MXFP4 `packed_gate_up_blocks` and
`packed_down_blocks`). The filtered variants
(`load_model_dir_filtered` and `load_model_dir_filtered_validated`)
MUST honor any caller-supplied skip predicate.

#### Scenario: Walk-only safetensors load excludes dense FFN tensors
- **WHEN** `load_model_dir_walk_only` is called against a model with `gate_proj` / `up_proj` / `down_proj` tensors
- **THEN** none of those FFN tensors SHALL appear in the resulting `ModelWeights::tensors`
<!-- test: larql_models::test_loading::walk_only_excludes_ffn_tensors -->

#### Scenario: Walk-only excludes StarCoder2-style mlp.c_fc / mlp.c_proj
- **WHEN** `load_model_dir_walk_only` is called against a StarCoder2-style model
- **THEN** the `mlp.c_fc.weight` and `mlp.c_proj.weight` tensors SHALL be excluded from `ModelWeights::tensors`
<!-- test: larql_models::test_loading::walk_only_excludes_starcoder2_ffn_tensors -->

#### Scenario: Walk-only excludes GPT-OSS packed MXFP4 expert blocks
- **WHEN** `load_model_dir_walk_only` is called against a GPT-OSS-style model whose experts are stored as packed MXFP4 blocks
- **THEN** neither `packed_gate_up_blocks` nor `packed_down_blocks` SHALL be expanded or retained in memory
<!-- test: larql_models::test_loading::walk_only_excludes_gpt_oss_packed_mxfp4_experts -->

#### Scenario: Walk-only GGUF load excludes FFN tensors before dequant
- **WHEN** `load_model_dir_walk_only` is called against a GGUF model
- **THEN** FFN tensors SHALL be skipped at parse time and never dequantised to f32
<!-- test: larql_models::test_loading::load_gguf_walk_only_excludes_ffn_tensor -->

#### Scenario: Custom skip predicate is honored
- **WHEN** `load_model_dir_filtered` is called with a predicate that targets a specific tensor key
- **THEN** that tensor SHALL be excluded from `ModelWeights` and SHALL NOT be allocated as f32
<!-- test: larql_models::test_loading::filtered_custom_predicate_skips_target -->

### Requirement: `is_ffn_tensor` classifier

`larql_models::loading::safetensors::is_ffn_tensor` SHALL return `true`
for canonical FFN tensor key fragments — `gate_proj`, `up_proj`,
`down_proj`, `ffn_gate`, `ffn_up`, `ffn_down`, the per-expert MoE
fragments (`mlp.experts.`, `block_sparse_moe.experts.`), and the
packed-MXFP4 fragments (`packed_gate_up_blocks`, `packed_down_blocks`)
— and SHALL return `false` for non-FFN tensors. The empty key SHALL
return `false`.

#### Scenario: gate_proj is recognised as FFN
- **WHEN** `is_ffn_tensor("layers.0.mlp.gate_proj.weight")` is queried
- **THEN** it SHALL return `true`
<!-- test: larql_models::loading::safetensors::is_ffn_tensor_gate_proj -->

#### Scenario: GGUF-style ffn_* keys are recognised as FFN
- **WHEN** `is_ffn_tensor` is called on `ffn_gate`, `ffn_up`, or `ffn_down` keys
- **THEN** each SHALL return `true`
<!-- test: larql_models::loading::safetensors::is_ffn_tensor_ffn_variants -->

#### Scenario: MoE per-expert tensors are recognised as FFN
- **WHEN** `is_ffn_tensor` is called on `mlp.experts.*` or `block_sparse_moe.experts.*` keys
- **THEN** it SHALL return `true`
<!-- test: larql_models::loading::safetensors::is_ffn_tensor_moe_experts -->

#### Scenario: Packed MXFP4 expert blocks are recognised as FFN
- **WHEN** `is_ffn_tensor` is called on `packed_gate_up_blocks` or `packed_down_blocks`
- **THEN** it SHALL return `true`
<!-- test: larql_models::loading::safetensors::is_ffn_tensor_packed_keys -->

#### Scenario: Non-FFN tensor keys are rejected
- **WHEN** `is_ffn_tensor` is called on attention or norm tensor keys
- **THEN** it SHALL return `false`
<!-- test: larql_models::loading::safetensors::is_ffn_tensor_rejects_non_ffn -->

#### Scenario: Empty key is rejected
- **WHEN** `is_ffn_tensor("")` is queried
- **THEN** it SHALL return `false`
<!-- test: larql_models::loading::safetensors::is_ffn_tensor_empty_key -->

### Requirement: Selective in-memory weight release via `drop_*` methods

`ModelWeights` SHALL expose memory-release helpers that prune a single
class of tensors and return the freed byte count: `drop_ffn_weights`
(FFN projections, FFN biases, packed expert raw bytes, and mmap-backed
packed expert ranges; orphaned packed mmaps MUST be released too),
`drop_attn_weights` (Q/K/V/O projections plus their associated norms),
`drop_lm_head` (replaces the output projection with an empty `0×0`
array), and `drop_embed` (replaces the embedding matrix with an empty
`0×0` array). Each method MUST leave the `ModelWeights` struct in a
valid state.

#### Scenario: drop_ffn_weights frees mmap-backed packed expert ranges
- **WHEN** a Gemma 4 A4B model is loaded with packed BF16 experts and then `drop_ffn_weights` is called
- **THEN** the packed byte ranges SHALL be removed and packed mmaps with no remaining references SHALL be released
<!-- test: larql_models::test_loading::packed_bf16_experts_are_mmap_backed_not_copied -->

#### Scenario: Packed BF16 experts are kept in mmap, not heap-copied
- **WHEN** a Gemma 4 A4B safetensors load encounters packed BF16 expert blocks
- **THEN** the loader SHALL retain a memory-mapped byte range and SHALL NOT clone the expert data into the heap
<!-- test: larql_models::test_loading::packed_bf16_experts_are_mmap_backed_not_copied -->

#### Scenario: drop_lm_head replaces lm_head with an empty 0x0 array
- **WHEN** `drop_lm_head` is called on a populated `ModelWeights`
- **THEN** it SHALL return the previously-occupied byte count and SHALL leave `lm_head` as a valid empty array
<!-- test: unbacked -->

#### Scenario: drop_embed replaces embed with an empty 0x0 array
- **WHEN** `drop_embed` is called on a populated `ModelWeights`
- **THEN** it SHALL return the previously-occupied byte count and SHALL leave `embed` as a valid empty array
<!-- test: unbacked -->

### Requirement: Embedding and tied lm_head fallbacks

The safetensors loader SHALL set `ModelWeights::embed` from the
canonical embedding key (after prefix stripping) and SHALL fall back to
the architecture-supplied `embed_key()` when the canonical key is
missing. When the model declares `tie_word_embeddings`, the loader
SHALL clone `embed` into `lm_head` rather than failing on a missing
`lm_head.weight`. Loading a directory whose embedding cannot be
resolved SHALL return a `MissingTensor` error.

#### Scenario: Tied lm_head is supplied by cloning embed
- **WHEN** the model config sets `tie_word_embeddings = true` and no `lm_head.weight` exists on disk
- **THEN** loading SHALL succeed and `lm_head` SHALL hold the same data as `embed`
<!-- test: larql_models::test_loading::tied_lm_head_falls_back_to_embed -->

#### Scenario: Missing embedding tensor surfaces as MissingTensor
- **WHEN** loading a model directory whose safetensors shards contain no embedding tensor under any known key
- **THEN** the loader SHALL return a `MissingTensor` error
<!-- test: larql_models::test_loading::missing_embed_returns_missing_tensor_error -->

#### Scenario: 1D norm tensors land in vectors, not tensors
- **WHEN** the loader encounters a 1D norm tensor (for example `layers.0.input_layernorm.weight`)
- **THEN** it SHALL insert it into `ModelWeights::vectors` rather than `ModelWeights::tensors`
<!-- test: larql_models::test_loading::load_1d_norm_tensor_goes_into_vectors -->

### Requirement: Model-path resolution and directory shape

`resolve_model_path` SHALL accept either a local directory, a single
`*.gguf` file, or a HuggingFace cache identifier (`org/name`). For
HuggingFace identifiers the resolver SHALL search
`~/.cache/huggingface/hub/models--{org}--{name}/snapshots/` for a
snapshot containing safetensors weights, falling back to a snapshot
that only contains `config.json` if no safetensors weights are found.
MLX-style models that store weights in a `weights/` subdirectory MUST
be detected. Directories with no usable weights SHALL return an
explicit error rather than producing an empty `ModelWeights`.

#### Scenario: Existing local directory resolves directly
- **WHEN** `resolve_model_path` is given a path that exists as a directory
- **THEN** it SHALL return that directory unchanged
<!-- test: larql_models::loading::safetensors::resolve_model_path_existing_dir -->

#### Scenario: Single GGUF file resolves directly
- **WHEN** `resolve_model_path` is given a path to an existing `*.gguf` file
- **THEN** it SHALL return that file path
<!-- test: larql_models::loading::safetensors::resolve_model_path_existing_gguf_file -->

#### Scenario: Nonexistent path returns an error
- **WHEN** `resolve_model_path` is given a path that does not exist as a directory, GGUF file, or HuggingFace cache entry
- **THEN** the resolver SHALL return an error
<!-- test: larql_models::loading::safetensors::resolve_model_path_nonexistent_returns_error -->

#### Scenario: HuggingFace cache snapshot with safetensors is preferred
- **WHEN** the HuggingFace cache contains a snapshot with safetensors weights
- **THEN** `resolve_model_path` SHALL return that snapshot
<!-- test: larql_models::loading::safetensors::resolve_model_path_hf_cache_with_safetensors -->

#### Scenario: HuggingFace cache fallback to a config-only snapshot
- **WHEN** no snapshot has safetensors weights but at least one snapshot has `config.json`
- **THEN** `resolve_model_path` SHALL return the config-only snapshot rather than failing
<!-- test: larql_models::loading::safetensors::resolve_model_path_hf_cache_fallback_config_json -->

#### Scenario: MLX `weights/` subdirectory is detected
- **WHEN** a model directory stores safetensors shards inside a `weights/` subdirectory
- **THEN** `load_model_dir` SHALL discover and load them
<!-- test: larql_models::test_loading::mlx_weights_subdir_is_found -->

#### Scenario: Directory with no safetensors and no GGUF returns an error
- **WHEN** a target directory contains neither safetensors files nor GGUF files
- **THEN** loading SHALL return a `NoSafetensors` error
<!-- test: larql_models::test_loading::no_safetensors_files_returns_error -->

#### Scenario: Path that is neither directory nor GGUF returns an error
- **WHEN** `load_model_dir` is invoked on a regular file that is not a `*.gguf` file
- **THEN** loading SHALL return a `NotADirectory` error
<!-- test: larql_models::test_loading::non_directory_non_gguf_file_returns_error -->

### Requirement: Vector-record JSON serialisation

`larql_models::vectors` SHALL serialise per-vector metadata (vector
records, top-K entries, and vector-file headers) to JSON with stable
field names that round-trip via serde. The set of supported component
identifiers SHALL be a fixed list of six strings, and the published
constants MUST agree with that list.

#### Scenario: Vector records round-trip through JSON
- **WHEN** a `VectorRecord` is serialised to JSON and deserialised back
- **THEN** every field SHALL be preserved bit-for-bit
<!-- test: larql_models::vectors::vector_record_json_roundtrip -->

#### Scenario: Top-K entries clone and serialise
- **WHEN** a `TopKEntry` is cloned and then serialised to JSON
- **THEN** both the clone and the JSON output SHALL agree with the original entry
<!-- test: larql_models::vectors::top_k_entry_clone_and_serialize -->

#### Scenario: Vector-file headers round-trip through JSON
- **WHEN** a vector-file header is serialised to JSON and deserialised back
- **THEN** all metadata fields SHALL match the original
<!-- test: larql_models::vectors::vector_file_header_json_roundtrip -->

#### Scenario: All six component identifiers are advertised
- **WHEN** the public list of components is iterated
- **THEN** it SHALL contain exactly six entries covering every supported component class
<!-- test: larql_models::vectors::all_components_contains_all_six -->

#### Scenario: Component constants match the canonical strings
- **WHEN** the published `pub const` component identifiers are read
- **THEN** each constant SHALL match its expected canonical string value
<!-- test: larql_models::vectors::component_constants_match_expected_strings -->

### Requirement: Test suite coverage

The capability's behavior SHALL be exercised by every test in the test modules listed below. New tests added to those modules SHALL be considered part of this capability's contract by default.

#### Scenario: Workspace tests bound to this capability
- **WHEN** the trace tool resolves the wildcard module references for this capability
- **THEN** every backing test SHALL resolve to a real `#[test]` function in the workspace
<!-- test: larql_models::test_loading::**::* -->
<!-- test: larql_models::loading::safetensors::tests::**::* -->
<!-- test: larql_models::loading::gguf::tests::**::* -->
<!-- test: larql_models::vectors::tests::**::* -->
