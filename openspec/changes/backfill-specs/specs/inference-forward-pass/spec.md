## ADDED Requirements

### Requirement: Full transformer forward pass

The `larql_inference` crate SHALL provide a complete transformer forward
path that maps input token ids through embedding, every transformer
layer, and the final unembedding to produce per-position vocabulary
logits. The pipeline MUST drive every stage from the
`ModelArchitecture` trait so that no model-family string or hardcoded
constant is read inside the forward path, and MUST produce identical
results across the high-level entry points (`forward_logits`,
`forward_raw_logits`, `forward_from_layer`) when invoked on the same
inputs.

#### Scenario: Raw logits have vocabulary shape and finite values
- **WHEN** `forward_raw_logits` is invoked on a small synthetic model with a multi-token prompt
- **THEN** the returned matrix SHALL have `[seq, vocab]` shape and all entries SHALL be finite
<!-- test: larql_inference::forward::predict::raw::tests::forward_raw_logits_returns_vocab_logits -->
<!-- test: larql_inference::forward::predict::raw::tests::forward_raw_logits_single_token -->
<!-- test: larql_inference::forward::predict::raw::tests::forward_raw_logits_with_prefix_shape -->

#### Scenario: forward_from_layer at layer zero matches full forward
- **WHEN** `forward_from_layer` is invoked with `start_layer == 0` and the same inputs as a full forward pass
- **THEN** the resulting logits SHALL be element-wise equal to the full forward output, and starting from a later layer SHALL skip the earlier layers without panicking
<!-- test: larql_inference::forward::predict::raw::tests::forward_from_layer_zero_equals_full_forward -->
<!-- test: larql_inference::forward::predict::raw::tests::forward_from_layer_skips_early_layers -->
<!-- test: larql_inference::forward::predict::raw::tests::forward_from_layer_output_shape -->

#### Scenario: Top-k sort tolerates NaN logits without panicking
- **WHEN** `predict` sorts a logits vector that contains NaN entries alongside real values
- **THEN** the real maximum SHALL be returned at rank zero, all-NaN inputs SHALL NOT panic, and clean inputs SHALL be returned in plain descending order
<!-- test: larql_inference::forward::predict::tests::topk_sort_nan_last_preserves_real_max -->
<!-- test: larql_inference::forward::predict::tests::topk_sort_all_nan_doesnt_panic -->
<!-- test: larql_inference::forward::predict::tests::topk_sort_no_nan_is_plain_descending -->

#### Scenario: Architecture goldens match across CPU and GPU
- **WHEN** the architecture-golden harness runs Gemma 3 4B, Gemma 4 31B dense, Gemma 4 26B-A4B MoE, Llama 2 7B, and Mistral 7B against captured outputs on both backends
- **THEN** every supported (vindex, backend) pair SHALL match the recorded golden top-token within tolerance, and missing weights SHALL be skipped (not failed) when not in strict mode
<!-- test: larql_inference::test_arch_golden::arch_gemma3_4b_cpu -->
<!-- test: larql_inference::test_arch_golden::arch_gemma4_31b_dense_cpu -->
<!-- test: larql_inference::test_arch_golden::arch_gemma4_26b_a4b_moe_cpu -->
<!-- test: larql_inference::test_arch_golden::arch_llama2_7b_cpu -->
<!-- test: larql_inference::test_arch_golden::arch_mistral_7b_cpu -->

#### Scenario: Per-layer pipeline construction covers every layer
- **WHEN** `build_pipeline_layers` is invoked against a real Q4K vindex
- **THEN** it SHALL produce a `FullPipelineLayer` for every transformer layer with the per-layer architecture parameters (head_dim, RoPE base, attn scale, rotary fraction, norm offset, activation, FFN type) populated from `ModelArchitecture`
<!-- test: larql_inference::test_layer_graph_integration::build_pipeline_layers_produces_all_layers -->
<!-- test: larql_inference::test_layer_graph_integration::resolve_attn_weights_returns_some_with_q4k_loaded -->

### Requirement: Embedding lookup

`embed_tokens` SHALL convert a slice of token ids into a `[seq, hidden]`
matrix by gathering rows of the architecture's input-embedding tensor.
The function MUST be deterministic (same input → same output) and MUST
return distinct rows for distinct token ids when the underlying
embedding rows differ.

#### Scenario: Embedding produces correct shape and is deterministic
- **WHEN** `embed_tokens` is called on the same token id repeatedly
- **THEN** every call SHALL return the same `[1, hidden]` row, and distinct token ids SHALL produce distinct rows when their embeddings differ
<!-- test: larql_inference::forward::embed::tests::embed_tokens_shape -->
<!-- test: larql_inference::forward::embed::tests::embed_tokens_single -->
<!-- test: larql_inference::forward::embed::tests::embed_different_tokens_differ -->
<!-- test: larql_inference::forward::embed::tests::embed_same_token_is_deterministic -->

### Requirement: RMS / Layer normalization with weight offset

`rms_norm` and `layer_norm` SHALL apply normalization with an explicit
`offset` parameter so that families that store `weight - 1` on disk
(Gemma 2/3 with offset 1.0) and families that store the raw weight
(Llama, Gemma 4 with offset 0.0) can both be served without re-detection.
`rms_norm_heads` SHALL normalise each head independently when the
hidden dimension is interpreted as `[num_heads * head_dim]`.

#### Scenario: RMS norm preserves shape and produces finite output
- **WHEN** `rms_norm` is applied to a multi-row input
- **THEN** the output SHALL preserve `[rows, hidden]` shape, every entry SHALL be finite, and a zero input row SHALL NOT produce NaN or infinity
<!-- test: larql_inference::residual::tests::rms_norm_shape_preserved -->
<!-- test: larql_inference::residual::tests::rms_norm_output_is_finite -->
<!-- test: larql_inference::residual::tests::rms_norm_zero_row_is_finite -->

#### Scenario: Weight-offset semantics distinguish Llama and Gemma styles
- **WHEN** `rms_norm` is invoked with `offset = 0.0` (Llama style) and `offset = 1.0` (Gemma style)
- **THEN** an all-zero learned weight with offset 1.0 SHALL match a no-weight normalization, and a unit weight with offset 1.0 SHALL produce the documented output
<!-- test: larql_inference::test_modules::test_residual::rms_norm_with_weight_offset_zero -->
<!-- test: larql_inference::test_modules::test_residual::rms_norm_with_weight_offset_one -->
<!-- test: larql_inference::residual::tests::rms_norm_with_ones_weight_and_offset_one -->

#### Scenario: Per-head RMS norm normalises each head independently
- **WHEN** `rms_norm_heads` is applied to a `[seq, num_heads * head_dim]` matrix
- **THEN** each head SHALL be normalised independently, the output SHALL preserve shape, and the no-weight variant SHALL still produce a finite, head-local norm
<!-- test: larql_inference::test_modules::test_residual::rms_norm_heads_two_heads -->
<!-- test: larql_inference::residual::tests::rms_norm_heads_no_weight_shape -->
<!-- test: larql_inference::residual::tests::rms_norm_heads_normalises_each_head_independently -->
<!-- test: larql_inference::residual::tests::rms_norm_heads_with_weight_scales -->

#### Scenario: Layer norm zero-means and unit-variances each row
- **WHEN** `layer_norm` is applied to a multi-row input
- **THEN** every output row SHALL have zero mean and unit variance within tolerance, and the output SHALL be finite
<!-- test: larql_inference::residual::tests::layer_norm_shape_and_finite -->
<!-- test: larql_inference::residual::tests::layer_norm_zero_mean_unit_var -->

### Requirement: Residual stream operations

The forward path SHALL expose composable residual operations
(`dot_proj`, `add_bias`, `apply_norm`) that downstream code can reuse
without duplicating BLAS calls. `add_bias` MUST safely handle a bias
shorter than the row width by no-oping past the bias length, and a
zero bias MUST be a no-op.

#### Scenario: Projection and bias operations behave as documented
- **WHEN** `dot_proj` is applied with an identity-shaped weight, and `add_bias` is applied with a zero bias and a shorter-than-row bias
- **THEN** `dot_proj` SHALL preserve shape and identity values, `add_bias` SHALL update every row, a zero bias SHALL be a no-op, and a shorter bias SHALL NOT overflow
<!-- test: larql_inference::forward::ops::tests::dot_proj_shape -->
<!-- test: larql_inference::forward::ops::tests::dot_proj_identity_weight -->
<!-- test: larql_inference::forward::ops::tests::dot_proj_values_correct -->
<!-- test: larql_inference::forward::ops::tests::add_bias_all_rows_updated -->
<!-- test: larql_inference::forward::ops::tests::add_bias_shorter_bias_does_not_overflow -->
<!-- test: larql_inference::forward::ops::tests::add_bias_zero_bias_is_noop -->

#### Scenario: apply_norm honours weight offset
- **WHEN** `apply_norm` is invoked with `offset = 1.0` and `offset = 0.0`
- **THEN** the two outputs SHALL differ, both SHALL preserve shape, and both SHALL be finite
<!-- test: larql_inference::forward::ops::tests::apply_norm_output_shape_matches_input -->
<!-- test: larql_inference::forward::ops::tests::apply_norm_output_is_finite -->
<!-- test: larql_inference::forward::ops::tests::apply_norm_with_offset_differs_from_without -->

### Requirement: Per-Layer Embeddings (Gemma 4 E2B)

The forward path SHALL support Per-Layer Embeddings: for architectures
that report `has_ple() == true`, it SHALL precompute per-layer embeddings via
`precompute_per_layer_inputs` and apply them with
`apply_per_layer_embedding` at the start of each layer. For
architectures without PLE the precompute SHALL return an empty input,
and applying a missing PLE input MUST leave the residual unchanged.

#### Scenario: PLE precompute returns empty when not applicable
- **WHEN** `precompute_per_layer_inputs` is called on an architecture without PLE or whose projection weight is missing
- **THEN** it SHALL return an empty input vector and SHALL NOT error
<!-- test: larql_inference::forward::ple::tests::precompute_returns_empty_when_arch_has_no_ple -->
<!-- test: larql_inference::forward::ple::tests::precompute_returns_empty_when_projection_weight_missing -->

#### Scenario: apply_per_layer_embedding is a no-op when PLE absent
- **WHEN** `apply_per_layer_embedding` is invoked with `None` PLE input or with PLE input but a missing gate weight
- **THEN** the residual SHALL be returned unchanged in shape and value
<!-- test: larql_inference::forward::ple::tests::apply_ple_none_input_returns_h_unchanged -->
<!-- test: larql_inference::forward::ple::tests::apply_ple_missing_gate_weight_returns_h_unchanged -->
<!-- test: larql_inference::forward::ple::tests::apply_ple_output_shape_matches_input -->

### Requirement: Chat templates and tokenizer

The crate SHALL render multi-turn chat for the four families it
supports — Gemma, ChatML (Qwen / DeepSeek), Llama 3, and Mistral
INST — and SHALL prefer a HuggingFace `tokenizer_config.json` /
standalone `chat_template.jinja` when present, falling back to a
hand-rolled renderer otherwise. The tokenizer wrapper SHALL load
HuggingFace tokenizers from a model directory and SHALL prepend a BOS
token only when the architecture declares one and it is not already
present.

#### Scenario: Multi-turn templates render with family-specific markers
- **WHEN** Gemma, ChatML, Llama 3, and Mistral renderers are invoked on the same multi-turn message list
- **THEN** each output SHALL include the family-specific markers (Gemma turn open, ChatML im_start tags, Llama 3 header tags, Mistral INST markers) and Mistral SHALL prepend the system message to the first user turn
<!-- test: larql_inference::prompt::tests::gemma_multi_turn_includes_model_open -->
<!-- test: larql_inference::prompt::tests::chatml_multi_turn -->
<!-- test: larql_inference::prompt::tests::llama_multi_turn -->
<!-- test: larql_inference::prompt::tests::mistral_prepends_system_to_first_user -->
<!-- test: larql_inference::prompt::tests::mistral_multi_turn -->

#### Scenario: Chat template source resolution prefers HF artifacts
- **WHEN** both a HuggingFace tokenizer-config template and a hand-rolled fallback exist
- **THEN** the HF template SHALL win, a standalone `chat_template.jinja` SHALL beat a tokenizer-config fallback, and a passthrough SHALL be used when nothing matches
<!-- test: larql_inference::chat::mod::hf_template_wins_over_fallback_when_both_exist -->
<!-- test: larql_inference::chat::mod::standalone_jinja_file_beats_tokenizer_config -->
<!-- test: larql_inference::chat::mod::full_passthrough_when_nothing_matches -->
<!-- test: larql_inference::chat::source::tests::try_hf_template_reads_standalone_jinja_file -->
<!-- test: larql_inference::chat::source::tests::try_hf_template_reads_tokenizer_config_fallback -->

#### Scenario: BOS handling matches architecture declaration
- **WHEN** `maybe_prepend_bos` is invoked on inputs from a Gemma 4-style architecture and a no-BOS architecture
- **THEN** the BOS SHALL be prepended only when the architecture declares one, the function SHALL be idempotent when BOS is already present, and SHALL not error on empty input
<!-- test: larql_inference::tokenizer::tests::maybe_prepend_bos_noop_when_arch_has_no_bos -->
<!-- test: larql_inference::tokenizer::tests::maybe_prepend_bos_fires_on_gemma4_style_missing_bos -->
<!-- test: larql_inference::tokenizer::tests::maybe_prepend_bos_idempotent_when_already_present -->
<!-- test: larql_inference::tokenizer::tests::maybe_prepend_bos_empty_input -->

### Requirement: Logit lens and vocabulary projection

The crate SHALL expose a logit-lens API (`track_token`, `track_race`,
`topk`) that projects intermediate residuals through the unembedding
to produce per-layer top-k token distributions, and SHALL provide a
vocabulary-projection API (`embedding_row`, `unembedding_row`,
`embedding_neighbors`, `project_through_unembed`) so that activation
patches and trace analyses can query token neighborhoods without
re-running a forward pass. Mismatched dimensions MUST return an empty
result rather than panic.

#### Scenario: Logit lens top-k respects probability semantics
- **WHEN** `topk` is queried with a valid residual
- **THEN** results SHALL be in descending probability order, every probability SHALL lie in `[0, 1]`, and `k == 0` or a dimension mismatch SHALL return an empty vector
<!-- test: larql_inference::forward::lens::tests::topk_returns_correct_count -->
<!-- test: larql_inference::forward::lens::tests::topk_descending_by_prob -->
<!-- test: larql_inference::forward::lens::tests::topk_probs_in_unit_interval -->
<!-- test: larql_inference::forward::lens::tests::topk_zero_k_returns_empty -->
<!-- test: larql_inference::forward::lens::tests::topk_dim_mismatch_returns_empty -->

#### Scenario: track_race preserves layer order and fans out probabilities
- **WHEN** `track_race` is invoked across all layers
- **THEN** the layer order SHALL be preserved and the per-layer probability total SHALL be close to the full vocabulary mass
<!-- test: larql_inference::forward::lens::tests::track_race_preserves_layer_order -->
<!-- test: larql_inference::forward::lens::tests::track_race_total_prob_per_layer_sums_close_to_full_vocab -->

#### Scenario: Embedding-neighbor lookup behaves under cosine similarity
- **WHEN** `embedding_neighbors` is queried with a known row, a zero query, or a dim-mismatched query
- **THEN** the source token SHALL appear at rank zero with cosine 1.0, results SHALL be in descending similarity order, and zero or mismatched queries SHALL return empty
<!-- test: larql_inference::forward::vocab_proj::tests::embedding_neighbors_self_is_top_with_unit_cosine -->
<!-- test: larql_inference::forward::vocab_proj::tests::embedding_neighbors_descending -->
<!-- test: larql_inference::forward::vocab_proj::tests::embedding_neighbors_zero_query_returns_empty -->
<!-- test: larql_inference::forward::vocab_proj::tests::embedding_neighbors_dim_mismatch_returns_empty -->

#### Scenario: project_through_unembed agrees with manual dot product
- **WHEN** a residual is projected through the unembedding matrix
- **THEN** the returned top-k SHALL match a manual dot-product baseline, descending order, and dim mismatches SHALL return empty
<!-- test: larql_inference::forward::vocab_proj::tests::project_through_unembed_returns_descending_topk -->
<!-- test: larql_inference::forward::vocab_proj::tests::project_through_unembed_matches_manual_dot -->
<!-- test: larql_inference::forward::vocab_proj::tests::project_through_unembed_dim_mismatch_returns_empty -->

### Requirement: Activation patching, hooks, and trace honesty

The crate SHALL support activation patching (`capture_donor_state`,
`patch_at_layer`) so that residuals captured from one prompt can be
injected into another at a specific layer, and SHALL provide a hooks
surface (`on_pre_layer`, `on_post_attention`, `on_attention_weights`,
`on_ffn_activation`, `on_post_layer`) so that interventions can read
or mutate the residual without forking the forward path. The
production entry point `predict_honest` SHALL NOT silently fall back
to a different path or family — when a request cannot be served
honestly it MUST return an error.

#### Scenario: Donor capture and patching propagate downstream
- **WHEN** `capture_donor_state` records a residual at one layer and `patch_at_layer` injects it into a recipient run
- **THEN** the recipient residual at the patched layer SHALL be replaced, downstream layers SHALL see the patched residual, and out-of-range positions SHALL be dropped
<!-- test: larql_inference::forward::patching::tests::capture_donor_state_records_requested_coords -->
<!-- test: larql_inference::forward::patching::tests::capture_donor_state_drops_out_of_range_positions -->
<!-- test: larql_inference::forward::patching::tests::empty_donor_state_is_noop_patch -->
<!-- test: larql_inference::forward::patching::tests::patch_changes_recipient_residual_downstream -->
<!-- test: larql_inference::forward::patching::tests::patch_at_layer_overwrites_residual_at_that_layer -->

#### Scenario: Hooks observe and mutate residuals during prefill and decode
- **WHEN** a no-op hook, a record hook, and a zero-ablate hook are run through `generate_cached_hooked`
- **THEN** the no-op SHALL match the unhooked baseline, the record hook SHALL fire on every layer for both prefill and decode, and steering hooks SHALL change the output token stream
<!-- test: larql_inference::forward::hooks::tests::noop_hook_compiles_and_does_nothing -->
<!-- test: larql_inference::forward::hooks::tests::record_hook_captures_only_requested_layers -->
<!-- test: larql_inference::forward::hooks::tests::zero_ablate_full_layer -->
<!-- test: larql_inference::forward::hooks::tests::zero_ablate_specific_positions -->
<!-- test: larql_inference::forward::kv_generate::tests::generate_cached_hooked_with_noop_matches_baseline -->
<!-- test: larql_inference::forward::kv_generate::tests::generate_cached_hooked_record_fires_during_prefill_and_decode -->
<!-- test: larql_inference::forward::kv_generate::tests::generate_cached_hooked_steer_changes_output -->

#### Scenario: Decode reproducibility across runs
- **WHEN** decode is run twice on the same prompt with deterministic sampling for Gemma 3 4B, Gemma 4 31B dense, Llama 2 7B, and Mistral 7B
- **THEN** the two runs SHALL produce identical token streams for the requested step counts
<!-- test: larql_inference::test_decode_consistency::decode_consistency_gemma3_4b -->
<!-- test: larql_inference::test_decode_consistency::decode_consistency_gemma3_4b_2steps -->
<!-- test: larql_inference::test_decode_consistency::decode_consistency_gemma4_31b_dense -->
<!-- test: larql_inference::test_decode_consistency::decode_consistency_llama2_7b -->
<!-- test: larql_inference::test_decode_consistency::decode_consistency_mistral_7b -->

#### Scenario: Stage bisect localises decode regressions
- **WHEN** the stage-bisect harness drives Gemma 3 4B, Gemma 4 31B dense, Llama 2 7B, and Mistral 7B through each pipeline stage independently
- **THEN** every stage SHALL agree with the reference at that bisect point or fail with a stage-specific diagnostic
<!-- test: larql_inference::test_decode_stage_bisect::stage_bisect_gemma3_4b -->
<!-- test: larql_inference::test_decode_stage_bisect::stage_bisect_gemma4_31b_dense -->
<!-- test: larql_inference::test_decode_stage_bisect::stage_bisect_llama2_7b -->
<!-- test: larql_inference::test_decode_stage_bisect::stage_bisect_mistral_7b -->

#### Scenario: Logit goldens lock per-vindex top-k tokens
- **WHEN** the logits-golden harness runs against captured per-vindex / per-backend goldens
- **THEN** every supported (vindex, backend) pair SHALL match the recorded top-5 tokens within tolerance, including Q4K-down and Q6K-down quantized variants and Gemma 4 E2B
<!-- test: larql_inference::test_logits_goldens::logits_golden_gemma3_4b_cpu -->
<!-- test: larql_inference::test_logits_goldens::logits_golden_gemma4_31b_dense_cpu -->
<!-- test: larql_inference::test_logits_goldens::logits_golden_llama2_7b_cpu -->
<!-- test: larql_inference::test_logits_goldens::logits_golden_mistral_7b_cpu -->
<!-- test: larql_inference::test_logits_goldens::logits_golden_gemma3_4b_q4k_down_cpu -->
<!-- test: larql_inference::test_logits_goldens::logits_golden_gemma4_31b_q6kdown_cpu -->
<!-- test: larql_inference::test_logits_goldens::logits_golden_gemma4_e2b_cpu -->

### Requirement: Test suite coverage

The capability's behavior SHALL be exercised by every test in the test modules listed below. New tests added to those modules SHALL be considered part of this capability's contract by default.

#### Scenario: Workspace tests bound to this capability
- **WHEN** the trace tool resolves the wildcard module references for this capability
- **THEN** every backing test SHALL resolve to a real `#[test]` function in the workspace
<!-- test: larql_inference::test_llm_dispatch::**::* -->
<!-- test: larql_inference::test_decode_consistency::**::* -->
<!-- test: larql_inference::test_logits_goldens::**::* -->
<!-- test: larql_inference::test_arch_golden::**::* -->
<!-- test: larql_inference::test_decode_stage_bisect::**::* -->
<!-- test: larql_inference::test_modules::**::* -->
<!-- test: larql_inference::residual::tests::**::* -->
<!-- test: larql_inference::prompt::tests::**::* -->
<!-- test: larql_inference::tokenizer::tests::**::* -->
