## ADDED Requirements

### Requirement: LayerGraph trait surface and per-layer dispatch

`larql_inference::layer_graph::LayerGraph` SHALL be the single trait through
which decode dispatches forward passes per layer. The trait MUST admit at
least four implementations — `DenseLayerGraph`, `WalkLayerGraph`,
`CachedLayerGraph`, and `PerLayerGraph` — that compose freely behind the
same surface, MUST expose a stable distinct `name()` per implementation,
and MUST produce finite residual outputs of the correct shape for every
layer.

#### Scenario: Dense and walk paths produce identical output shape on the same input
- **WHEN** `DenseLayerGraph` and `WalkLayerGraph` are run on the same input on a small synthetic model
- **THEN** the layer outputs SHALL share the same shape and SHALL be finite, and every implementation's `name()` SHALL be distinct
<!-- test: larql_inference::layer_graph::tests::dense_and_walk_produce_same_output_shape -->
<!-- test: larql_inference::layer_graph::tests::layer_output_residual_is_finite_for_all_impls -->
<!-- test: larql_inference::layer_graph::tests::layer_graph_names_are_distinct -->

#### Scenario: DenseLayerGraph runs single tokens and full layer stacks
- **WHEN** `DenseLayerGraph` is asked to forward a single token and then the full layer range
- **THEN** the output SHALL have the documented shape, the implementation SHALL report a stable `name()`, and the per-layer accessor SHALL return `Some` for in-range and gracefully `None` for out-of-range layer indices
<!-- test: larql_inference::layer_graph::dense::tests::dense_name -->
<!-- test: larql_inference::layer_graph::dense::tests::dense_forward_shape_single_token -->
<!-- test: larql_inference::layer_graph::dense::tests::dense_forward_all_layers -->
<!-- test: larql_inference::layer_graph::dense::tests::per_layer_get_in_range -->
<!-- test: larql_inference::layer_graph::dense::tests::per_layer_get_out_of_range_does_not_panic -->
<!-- test: larql_inference::layer_graph::dense::tests::per_layer_name -->

#### Scenario: WalkLayerGraph and PipelinedLayerGraph match the dense surface
- **WHEN** `WalkLayerGraph` and `PipelinedLayerGraph` are exercised on the same trait
- **THEN** their `name()` SHALL be unique, single-token and all-layer forwards SHALL produce finite output, neither path SHALL capture activations or attention by default, and out-of-range layer indices SHALL return `None`
<!-- test: larql_inference::layer_graph::walk::tests::walk_name -->
<!-- test: larql_inference::layer_graph::walk::tests::walk_forward_shape_single_token -->
<!-- test: larql_inference::layer_graph::walk::tests::walk_forward_all_layers -->
<!-- test: larql_inference::layer_graph::walk::tests::walk_never_captures_activation_or_attention -->
<!-- test: larql_inference::layer_graph::walk::tests::pipelined_name -->
<!-- test: larql_inference::layer_graph::walk::tests::pipelined_in_range_produces_output -->
<!-- test: larql_inference::layer_graph::walk::tests::pipelined_out_of_range_returns_none -->

### Requirement: CachedLayerGraph and template-based caching

`larql_inference::layer_graph::CachedLayerGraph` SHALL pre-compute residuals
for template-fixed layers and serve them on demand. Cache hits MUST return
the stored residual; cache misses MUST return `None`. The cache build path
MUST allow callers to specify exactly which layers are cached, and the
template detector MUST select the longest matching prefix and refuse a
prefix longer than the input. Activation capture during a dense forward
MUST be opt-in.

#### Scenario: Cached residuals are returned for cached layers and absent otherwise
- **WHEN** `CachedLayerGraph::from_residuals` is built (empty, single, or multiple) and forward is queried for a cached and an uncached layer
- **THEN** cached layers SHALL return the stored residual, uncached layers SHALL return `None`, and `build` SHALL only cache the layers explicitly specified
<!-- test: larql_inference::layer_graph::cached::tests::from_residuals_empty -->
<!-- test: larql_inference::layer_graph::cached::tests::from_residuals_single -->
<!-- test: larql_inference::layer_graph::cached::tests::from_residuals_multiple -->
<!-- test: larql_inference::layer_graph::cached::tests::forward_layer_returns_cached -->
<!-- test: larql_inference::layer_graph::cached::tests::forward_layer_none_for_uncached -->
<!-- test: larql_inference::layer_graph::cached::tests::build_caches_specified_layers -->
<!-- test: larql_inference::layer_graph::cached::tests::cached_layer_graph_name -->

#### Scenario: Activation capture during dense forward is explicit
- **WHEN** a dense forward is run with capture disabled and then with capture enabled
- **THEN** the captured activation field SHALL be empty in the first case and SHALL be populated in the second case
<!-- test: larql_inference::layer_graph::dense::tests::dense_no_capture_has_no_activation -->
<!-- test: larql_inference::layer_graph::dense::tests::dense_capture_activation_populates_field -->

#### Scenario: Template detector selects the longest matching prefix
- **WHEN** the template detector is queried against an empty registry, against unmatched input, and against inputs with multiple overlapping prefixes (some too long for the input)
- **THEN** unmatched cases SHALL return `None`, exact prefix matches SHALL be detected, the longest prefix SHALL win, BOS-offset matching SHALL accept BOS at token 0, and prefixes longer than the input SHALL not match
<!-- test: larql_inference::layer_graph::template::tests::detect_no_templates_returns_none -->
<!-- test: larql_inference::layer_graph::template::tests::detect_no_match_returns_none -->
<!-- test: larql_inference::layer_graph::template::tests::detect_exact_prefix_match -->
<!-- test: larql_inference::layer_graph::template::tests::detect_longest_prefix_wins -->
<!-- test: larql_inference::layer_graph::template::tests::detect_bos_offset_allows_bos_at_token0 -->
<!-- test: larql_inference::layer_graph::template::tests::detect_prefix_too_long_for_input_returns_none -->

#### Scenario: Template universe builds and a guided walk produces finite output
- **WHEN** a template universe is constructed (empty, partially populated, fully populated) and a guided walk is run across all layers
- **THEN** missing layers SHALL return `None` from the universe, populated layers SHALL return their feature lists, the guided walk SHALL return correctly-shaped finite residuals, and `total_features` SHALL equal the sum of per-layer features
<!-- test: larql_inference::layer_graph::template::tests::universe_build_empty_entities_is_empty -->
<!-- test: larql_inference::layer_graph::template::tests::universe_get_missing_layer_returns_none -->
<!-- test: larql_inference::layer_graph::template::tests::universe_get_populated_layer_returns_features -->
<!-- test: larql_inference::layer_graph::template::tests::universe_total_features_sums_layers -->
<!-- test: larql_inference::layer_graph::template::tests::guided_walk_empty_universe_returns_correct_shape -->
<!-- test: larql_inference::layer_graph::template::tests::guided_walk_name -->
<!-- test: larql_inference::layer_graph::template::tests::guided_walk_all_layers_finite -->

#### Scenario: Real model integration matches walk and template-guided paths
- **WHEN** the integration tests run prefill and predict against a real Q4K model and against the guided-walk template universe
- **THEN** prefill output SHALL be finite and shaped, `prefill_with_kv` SHALL match the predict path's hidden state, the pipeline SHALL build all layers, attention weights SHALL resolve, the universe SHALL build, the guided walk SHALL run, and template detection SHALL match real token prefixes
<!-- test: larql_inference::test_layer_graph_integration::prefill_with_kv_shape_and_finiteness -->
<!-- test: larql_inference::test_layer_graph_integration::prefill_with_kv_matches_predict_q4k_hidden -->
<!-- test: larql_inference::test_layer_graph_integration::build_pipeline_layers_produces_all_layers -->
<!-- test: larql_inference::test_layer_graph_integration::resolve_attn_weights_returns_some_with_q4k_loaded -->
<!-- test: larql_inference::test_layer_graph_integration::template_universe_build_with_real_model -->
<!-- test: larql_inference::test_layer_graph_integration::guided_walk_layer_graph_with_real_universe -->
<!-- test: larql_inference::test_layer_graph_integration::detect_template_with_real_token_prefix -->

### Requirement: Predict pipeline and decode-stage bisection

`larql_inference::layer_graph::predict` SHALL provide a stable
`predict_*` family that drives a full forward pass through any registered
`LayerGraph`. The predict path MUST run honestly without panic, MUST work
with a `WalkLayerGraph` and a `DenseLayerGraph`, MUST honour
`CachedLayerGraph` shortcuts, and MUST drive a single-token decode path.
The integration test suite SHALL bisect decode through every architecture
to localise the first stage at which output diverges from a reference.

#### Scenario: Predict runs end-to-end across walkers, dense paths, and cached layers
- **WHEN** `predict_with_ffn`, `predict_honest`, the dense layer graph, the walk layer graph, and the predict pipeline are exercised on a synthetic model
- **THEN** every entry point SHALL run without panic, single-token decode SHALL succeed, all layers SHALL produce finite output, and cached layers SHALL be honoured
<!-- test: larql_inference::layer_graph::predict::tests::predict_with_ffn_returns_predictions -->
<!-- test: larql_inference::layer_graph::predict::tests::predict_with_ffn_single_token -->
<!-- test: larql_inference::layer_graph::predict::tests::predict_honest_runs_without_panic -->
<!-- test: larql_inference::layer_graph::predict::tests::predict_honest_single_token_decode_path -->
<!-- test: larql_inference::layer_graph::predict::tests::predict_honest_with_cached_layers -->
<!-- test: larql_inference::layer_graph::predict::tests::dense_layer_graph_forward_runs -->
<!-- test: larql_inference::layer_graph::predict::tests::dense_layer_graph_all_layers -->
<!-- test: larql_inference::layer_graph::predict::tests::walk_layer_graph_forward_runs -->
<!-- test: larql_inference::layer_graph::predict::tests::predict_pipeline_runs -->

#### Scenario: Decode-stage bisection localises divergence per architecture
- **WHEN** the stage-bisect harness runs against Gemma 3 4B, Gemma 4 31B dense, Llama 2 7B, and Mistral 7B
- **THEN** every supported architecture SHALL bisect cleanly through every decode stage and report the first diverging stage if any
<!-- test: larql_inference::test_decode_stage_bisect::stage_bisect_gemma3_4b -->
<!-- test: larql_inference::test_decode_stage_bisect::stage_bisect_gemma4_31b_dense -->
<!-- test: larql_inference::test_decode_stage_bisect::stage_bisect_llama2_7b -->
<!-- test: larql_inference::test_decode_stage_bisect::stage_bisect_mistral_7b -->

### Requirement: Generation loop sampling and detokenisation

`larql_inference::layer_graph::generate` SHALL implement an autoregressive
generation loop with greedy and temperature sampling, top-k and top-p
truncation, frequency and presence penalties, deterministic seeding under a
fixed RNG, EOS detection (id-based and string-based), incremental
detokenisation, and a chat session API supporting Gemma / ChatML / Llama-3
renderers.

#### Scenario: Greedy and penalty-aware sampling behave deterministically
- **WHEN** greedy and seeded-temperature sampling are exercised with frequency, presence, and top-k repetition penalties on synthetic logits
- **THEN** greedy SHALL pick the argmax, non-finite logits SHALL be skipped, empty logits SHALL return `None`, repeated tokens SHALL be pushed below the argmax, and a fixed seed SHALL be reproducible
<!-- test: larql_inference::layer_graph::generate::sampling::tests::greedy_returns_argmax -->
<!-- test: larql_inference::layer_graph::generate::sampling::tests::greedy_ignores_nonfinite -->
<!-- test: larql_inference::layer_graph::generate::sampling::tests::empty_logits_returns_none -->
<!-- test: larql_inference::layer_graph::generate::sampling::tests::frequency_penalty_pushes_repeated_token_below_argmax -->
<!-- test: larql_inference::layer_graph::generate::sampling::tests::presence_penalty_pushes_any_repeated_token -->
<!-- test: larql_inference::layer_graph::generate::sampling::tests::no_penalty_when_history_is_empty -->
<!-- test: larql_inference::layer_graph::generate::sampling::tests::topk_repetition_penalty_applies_to_hit_scores -->
<!-- test: larql_inference::layer_graph::generate::sampling::tests::temperature_seeded_is_reproducible -->

#### Scenario: Top-k, top-p, and temperature collapse to greedy at limits
- **WHEN** sampling is run with `temperature == 0`, `top_k == 1`, `top_p == 1`, and a low `top_p`
- **THEN** zero temperature SHALL be greedy, `top_k == 1` SHALL be greedy under temperature, `top_p == 1` SHALL keep the full distribution, low `top_p` SHALL collapse to argmax, and `top_k` SHALL truncate the candidate list
<!-- test: larql_inference::layer_graph::generate::sampling::tests::temperature_zero_is_greedy -->
<!-- test: larql_inference::layer_graph::generate::sampling::tests::top_k_one_is_greedy_under_temperature -->
<!-- test: larql_inference::layer_graph::generate::sampling::tests::top_p_one_keeps_full_distribution -->
<!-- test: larql_inference::layer_graph::generate::sampling::tests::top_p_low_collapses_to_argmax -->
<!-- test: larql_inference::layer_graph::generate::sampling::tests::top_k_truncates_choices -->
<!-- test: larql_inference::layer_graph::generate::sampling::tests::sample_from_topk_greedy -->
<!-- test: larql_inference::layer_graph::generate::sampling::tests::sample_from_topk_uses_all_when_no_filters -->
<!-- test: larql_inference::layer_graph::generate::sampling::tests::sample_from_topk_empty -->
<!-- test: larql_inference::layer_graph::generate::sampling::tests::config_is_greedy_predicate -->

#### Scenario: LM-head produces top-k logits in the right shape
- **WHEN** the CPU LM-head top-k routine is invoked on a synthetic hidden state and the backend LM-head is queried for its scores
- **THEN** scores SHALL have vocab shape, top-k SHALL be sorted descending, the returned token ids SHALL be in range, and the result length SHALL equal `k`
<!-- test: larql_inference::layer_graph::generate::tests::backend_lm_head_scores_shape -->
<!-- test: larql_inference::layer_graph::generate::tests::cpu_lm_head_topk_length -->
<!-- test: larql_inference::layer_graph::generate::tests::cpu_lm_head_topk_sorted_descending -->
<!-- test: larql_inference::layer_graph::generate::tests::cpu_lm_head_topk_token_ids_in_range -->

#### Scenario: EOS detection handles ids, strings, and tokenizer round-trips
- **WHEN** the EOS detector is exercised against built-in patterns, generation-config eos ids, stop strings, and short-circuit checks via tokenizer
- **THEN** Gemma `<end_of_turn>`, ChatML, and Llama tokens SHALL be recognised, an empty detector SHALL never stop, surface forms SHALL be trimmed, missing files SHALL fall back to the built-in, ID matches SHALL short-circuit, and duplicate stop strings SHALL not be added twice
<!-- test: larql_inference::layer_graph::generate::eos::tests::builtin_recognises_gemma_end_of_turn -->
<!-- test: larql_inference::layer_graph::generate::eos::tests::builtin_recognises_chatml_and_llama -->
<!-- test: larql_inference::layer_graph::generate::eos::tests::empty_never_stops -->
<!-- test: larql_inference::layer_graph::generate::eos::tests::surface_form_trimmed -->
<!-- test: larql_inference::layer_graph::generate::eos::tests::empty_decoded_does_not_match -->
<!-- test: larql_inference::layer_graph::generate::eos::tests::eos_id_match_independent_of_string -->
<!-- test: larql_inference::layer_graph::generate::eos::tests::from_generation_config_scalar_eos_id -->
<!-- test: larql_inference::layer_graph::generate::eos::tests::from_generation_config_array_eos_id -->
<!-- test: larql_inference::layer_graph::generate::eos::tests::from_generation_config_stop_strings_merged -->
<!-- test: larql_inference::layer_graph::generate::eos::tests::duplicate_stop_string_not_added_twice -->
<!-- test: larql_inference::layer_graph::generate::eos::tests::from_vindex_dir_missing_file_falls_back_to_builtin -->
<!-- test: larql_inference::layer_graph::generate::eos::tests::is_eos_with_tokenizer_catches_end_of_turn_after_skip_special_decode -->
<!-- test: larql_inference::layer_graph::generate::eos::tests::is_eos_with_tokenizer_short_circuits_on_id_match -->
<!-- test: larql_inference::layer_graph::generate::eos::tests::is_eos_with_tokenizer_uses_clean_decode_when_non_empty -->

#### Scenario: Incremental detokenisation emits monotonic suffixes only
- **WHEN** tokens are pushed into the incremental detokeniser one at a time, the prompt seed is set, and the cumulative output is compared to a full decode
- **THEN** an empty detokeniser SHALL produce no output until push, each push SHALL emit a monotonically growing suffix, the seed SHALL not emit the prompt, the cumulative output SHALL match a full decode, generated ids SHALL be tracked, and unknown tokens SHALL not panic
<!-- test: larql_inference::layer_graph::generate::detok::tests::empty_detokenizer_produces_no_output_until_push -->
<!-- test: larql_inference::layer_graph::generate::detok::tests::push_emits_increasing_suffix -->
<!-- test: larql_inference::layer_graph::generate::detok::tests::seed_does_not_emit_prompt -->
<!-- test: larql_inference::layer_graph::generate::detok::tests::cumulative_matches_full_decode -->
<!-- test: larql_inference::layer_graph::generate::detok::tests::ids_tracked -->
<!-- test: larql_inference::layer_graph::generate::detok::tests::unknown_token_does_not_panic -->

#### Scenario: Chat session renders Gemma, ChatML, and Llama-3 templates with correct eviction
- **WHEN** a chat session is built with each renderer, user/assistant turns are appended and closed, and eviction is forced past the cap
- **THEN** the Gemma renderer SHALL use the `model` role for assistant, ChatML SHALL pass roles verbatim, Llama-3 SHALL include the EOT token, eviction SHALL drop oldest whole turns but never drop the last turn, generated text SHALL be tokenised through the session tokeniser, token ids SHALL grow monotonically within a turn, and ChatML round-trips SHALL preserve token ids
<!-- test: larql_inference::layer_graph::generate::chat_session::tests::gemma_renderer_uses_model_role_for_assistant -->
<!-- test: larql_inference::layer_graph::generate::chat_session::tests::chatml_renderer_uses_role_verbatim -->
<!-- test: larql_inference::layer_graph::generate::chat_session::tests::llama3_renderer_includes_eot -->
<!-- test: larql_inference::layer_graph::generate::chat_session::tests::empty_session_is_empty -->
<!-- test: larql_inference::layer_graph::generate::chat_session::tests::append_user_records_one_turn -->
<!-- test: larql_inference::layer_graph::generate::chat_session::tests::open_and_close_assistant_turn -->
<!-- test: larql_inference::layer_graph::generate::chat_session::tests::extend_without_open_auto_opens -->
<!-- test: larql_inference::layer_graph::generate::chat_session::tests::eviction_drops_oldest_whole_turns -->
<!-- test: larql_inference::layer_graph::generate::chat_session::tests::eviction_never_drops_last_turn -->
<!-- test: larql_inference::layer_graph::generate::chat_session::tests::reset_clears_state -->
<!-- test: larql_inference::layer_graph::generate::chat_session::tests::token_ids_grows_monotonically_within_a_turn -->
<!-- test: larql_inference::layer_graph::generate::chat_session::tests::extend_with_generated_text_tokenises_through_session_tokenizer -->
<!-- test: larql_inference::layer_graph::generate::chat_session::tests::chatml_session_round_trips_tokens -->

### Requirement: Expert routing registry parser mask and session

`larql_inference::experts` SHALL provide an expert registry that loads
WASM-based experts, a parser that extracts `{"op": "...", "args": {...}}`
JSON from raw model output (handling preambles, code fences, full-width
punctuation, and Mistral-style missing commas), a grammar mask that walks
generation through op-name and argument states, and a session API that
builds a deterministic system prompt, dispatches calls, and surfaces
specific skip outcomes (no op call, unknown op, expert declined).

#### Scenario: Op-call parser extracts JSON from model output across formats
- **WHEN** the parser is run on simple objects, on outputs with a preamble, code fences, blocks without `op`, missing-comma Mistral output, and Unicode/full-width punctuation
- **THEN** every well-formed call SHALL parse, malformed inputs SHALL return `None`, default args SHALL be an empty object, full-width punctuation SHALL be normalised, brace/quote escaping SHALL not break depth tracking, and already-correct args SHALL not be double-patched
<!-- test: larql_inference::experts::parser::tests::extracts_simple_object -->
<!-- test: larql_inference::experts::parser::tests::extracts_after_preamble -->
<!-- test: larql_inference::experts::parser::tests::extracts_from_code_fence -->
<!-- test: larql_inference::experts::parser::tests::skips_blocks_without_op -->
<!-- test: larql_inference::experts::parser::tests::defaults_args_to_empty_object -->
<!-- test: larql_inference::experts::parser::tests::nested_objects_in_args -->
<!-- test: larql_inference::experts::parser::tests::brace_inside_string_value_does_not_break_depth -->
<!-- test: larql_inference::experts::parser::tests::escaped_quote_inside_string_does_not_break_depth -->
<!-- test: larql_inference::experts::parser::tests::fullwidth_punctuation_normalised -->
<!-- test: larql_inference::experts::parser::tests::mistral_missing_comma_before_args_patched -->
<!-- test: larql_inference::experts::parser::tests::already_correct_args_form_not_double_patched -->
<!-- test: larql_inference::experts::parser::tests::returns_none_when_no_object_present -->
<!-- test: larql_inference::experts::parser::tests::returns_none_when_op_missing -->
<!-- test: larql_inference::experts::parser::tests::returns_none_when_op_not_string -->
<!-- test: larql_inference::experts::parser::tests::returns_none_when_op_empty -->
<!-- test: larql_inference::experts::parser::tests::unbalanced_braces_returns_none -->
<!-- test: larql_inference::experts::parser::tests::unicode_in_args_preserved -->

#### Scenario: Grammar mask transitions through free-text, op-name, and done states
- **WHEN** the grammar state machine processes generated tokens through preamble, the op marker, the op-name segment, the closing quote, and multiple op-marker candidates
- **THEN** the state SHALL be free before the op marker, transition to op-name after the marker, accept the first op-marker, transition to done after the closing quote, and the registry SHALL extract op names only from `OpSpec`s
<!-- test: larql_inference::experts::mask::tests::grammar_state_free_before_op_marker -->
<!-- test: larql_inference::experts::mask::tests::grammar_state_op_name_after_marker -->
<!-- test: larql_inference::experts::mask::tests::grammar_state_done_after_closing_quote -->
<!-- test: larql_inference::experts::mask::tests::grammar_state_handles_preamble_before_op_marker -->
<!-- test: larql_inference::experts::mask::tests::grammar_state_picks_first_op_marker -->
<!-- test: larql_inference::experts::mask::tests::from_op_specs_extracts_names_only -->

#### Scenario: Expert session builds deterministic prompts and dispatches calls
- **WHEN** the session builds prompts under several chat templates and dispatches calls through happy paths and skip paths (no op call, unknown op, expert declined)
- **THEN** the system prompt SHALL be deterministic and sorted, list known ops, render args in braces, include the no-extra-text directive, and dispatch SHALL forward args verbatim and surface each skip outcome distinctly
<!-- test: larql_inference::experts::session::tests::system_prompt_is_deterministic -->
<!-- test: larql_inference::experts::session::tests::system_prompt_lists_known_ops -->
<!-- test: larql_inference::experts::session::tests::system_prompt_ops_are_sorted -->
<!-- test: larql_inference::experts::session::tests::build_prompt_wraps_via_template -->
<!-- test: larql_inference::experts::session::tests::build_prompt_plain_template_passes_through_unwrapped -->
<!-- test: larql_inference::experts::session::tests::dispatch_happy_path_returns_outcome -->
<!-- test: larql_inference::experts::session::tests::dispatch_with_preamble_still_finds_call -->
<!-- test: larql_inference::experts::session::tests::dispatch_no_op_call_returns_no_op_call_skip -->
<!-- test: larql_inference::experts::session::tests::dispatch_unknown_op_returns_unknown_op_skip -->
<!-- test: larql_inference::experts::session::tests::dispatch_expert_declined_returns_expert_declined_skip -->
<!-- test: larql_inference::experts::session::tests::system_prompt_is_deterministic_with_mock -->
<!-- test: larql_inference::experts::session::tests::system_prompt_lists_provided_ops_sorted -->
<!-- test: larql_inference::experts::session::tests::system_prompt_handles_empty_op_list -->
<!-- test: larql_inference::experts::session::tests::system_prompt_renders_args_in_braces -->
<!-- test: larql_inference::experts::session::tests::system_prompt_no_extra_text_directive_present -->
<!-- test: larql_inference::experts::session::tests::build_prompt_with_gemma_template_includes_system_and_user -->
<!-- test: larql_inference::experts::session::tests::build_prompt_with_each_template_variant_round_trips -->
<!-- test: larql_inference::experts::session::tests::dispatch_happy_path_with_mock -->
<!-- test: larql_inference::experts::session::tests::dispatch_no_op_call_with_mock -->
<!-- test: larql_inference::experts::session::tests::dispatch_unknown_op_with_mock -->
<!-- test: larql_inference::experts::session::tests::dispatch_expert_declined_with_mock -->
<!-- test: larql_inference::experts::session::tests::dispatch_forwards_args_verbatim_to_dispatcher -->

#### Scenario: End-to-end LLM-mediated, constrained, and trie-based dispatch pipelines
- **WHEN** the integration test suite drives the model through expert dispatch, constrained dispatch, trie dispatch, and full LLM-mediated dispatch
- **THEN** every pipeline SHALL run end-to-end against a real model when a model is provided, and SHALL skip cleanly when pre-conditions are absent
<!-- test: larql_inference::test_expert_dispatch::expert_dispatch_pipeline -->
<!-- test: larql_inference::test_constrained_dispatch::constrained_dispatch_pipeline -->
<!-- test: larql_inference::test_trie_dispatch::trie_dispatch_pipeline -->
<!-- test: larql_inference::test_llm_dispatch::llm_dispatch_pipeline -->

### Requirement: Expert grid and remote MoE wire format integrity

The expert-grid surface (`larql_inference::layer_graph::grid` plus `larql_inference::ffn::moe_remote`) SHALL batch FFN/MoE compute against remote shards. The grid MUST surface errors when the vector index lacks a Q4K mmap. The remote MoE wire protocol MUST round-trip request/response messages, handle truncation, preserve f16 residual values within tolerance, saturate on overflow, handle subnormals, parse layer ranges, and route softmax outputs that sum to one.

#### Scenario: Grid surfaces missing Q4K vindex as an error
- **WHEN** the grid is constructed against a vindex that has no Q4K mmap
- **THEN** the construction SHALL return an error rather than silently degrade
<!-- test: larql_inference::layer_graph::grid::tests::errors_when_vindex_has_no_q4k_mmap -->

#### Scenario: f16 residual values round-trip with bounded error and saturate cleanly
- **WHEN** f16 round-trip is exercised on residual values, overflow, subnormals, and full layer batch request/response payloads (including truncated payloads)
- **THEN** finite values SHALL preserve sign and magnitude within f16 tolerance, overflow SHALL saturate, subnormals SHALL be represented, and request/response serialisation SHALL round-trip exactly while still tolerating short reads
<!-- test: larql_inference::ffn::moe_remote::tests::f16_round_trip_preserves_residual_values -->
<!-- test: larql_inference::ffn::moe_remote::tests::f16_saturates_overflow -->
<!-- test: larql_inference::ffn::moe_remote::tests::f16_handles_subnormals -->
<!-- test: larql_inference::ffn::moe_remote::tests::f16_layer_batch_request_round_trip -->
<!-- test: larql_inference::ffn::moe_remote::tests::f16_layer_batch_response_round_trip -->
<!-- test: larql_inference::ffn::moe_remote::tests::f16_layer_batch_handles_truncation -->
<!-- test: larql_inference::ffn::moe_remote::multi_layer_wire::tests::request_round_trip -->
<!-- test: larql_inference::ffn::moe_remote::multi_layer_wire::tests::response_round_trip -->
<!-- test: larql_inference::ffn::moe_remote::multi_layer_wire::tests::handles_truncation -->

#### Scenario: Shard config and unit manifest parse and own ranges correctly
- **WHEN** layer ranges, shard configs, and unit manifests are parsed
- **THEN** valid ranges SHALL parse, invalid ranges SHALL be rejected, trailing slashes SHALL be stripped, ownership SHALL be reported via per-unit and per-range checks, manifests SHALL round-trip into shard configs, reversed and non-numeric ranges SHALL be rejected, and missing manifest files SHALL surface their path in the error
<!-- test: larql_inference::ffn::moe_remote::tests::parse_range_valid -->
<!-- test: larql_inference::ffn::moe_remote::tests::parse_range_invalid -->
<!-- test: larql_inference::ffn::moe_remote::tests::shard_config_strips_trailing_slash -->
<!-- test: larql_inference::ffn::moe_remote::tests::shard_owns -->
<!-- test: larql_inference::ffn::moe_remote::tests::shard_with_units_only_owns_via_layer_aware_check -->
<!-- test: larql_inference::ffn::moe_remote::tests::shard_layer_uniform_owns_unit_falls_back_to_range -->
<!-- test: larql_inference::ffn::moe_remote::tests::unit_manifest_round_trips_into_shard_configs -->
<!-- test: larql_inference::ffn::moe_remote::tests::unit_manifest_rejects_reversed_range -->
<!-- test: larql_inference::ffn::moe_remote::tests::unit_manifest_rejects_non_numeric_layer -->
<!-- test: larql_inference::ffn::moe_remote::tests::parse_unit_manifest_reports_path_on_missing_file -->

#### Scenario: MoE routing softmax sums to one and forward handles empty input
- **WHEN** the router is exercised with parameter-free normalisation, scalar router input, and standard softmax routing, and the MoE forward is invoked on empty input
- **THEN** the routing weights SHALL sum to one within tolerance under each routing mode, and `forward_moe` SHALL return zero on empty input
<!-- test: larql_inference::ffn::moe_remote::tests::route_softmax_sums_to_one -->
<!-- test: larql_inference::ffn::moe_remote::tests::route_with_parameter_free_router_norm -->
<!-- test: larql_inference::ffn::moe_remote::tests::route_with_router_input_scalar -->
<!-- test: larql_inference::ffn::moe_remote::tests::forward_moe_empty_input_returns_zero -->

### Requirement: Test suite coverage

The capability's behavior SHALL be exercised by every test in the test modules listed below. New tests added to those modules SHALL be considered part of this capability's contract by default.

#### Scenario: Workspace tests bound to this capability
- **WHEN** the trace tool resolves the wildcard module references for this capability
- **THEN** every backing test SHALL resolve to a real `#[test]` function in the workspace
<!-- test: larql_inference::test_layer_graph_integration::**::* -->
<!-- test: larql_inference::test_expert_dispatch::**::* -->
<!-- test: larql_inference::test_constrained_dispatch::**::* -->
<!-- test: larql_inference::test_trie_dispatch::**::* -->
<!-- test: larql_inference::layer_graph::generate::sampling::tests::**::* -->
