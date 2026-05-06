## ADDED Requirements

### Requirement: HTTP routing surface and path conventions

The `larql-server` HTTP layer SHALL expose two `axum::Router` builders —
`single_model_router` and `multi_model_router` — and the union of their
routes MUST cover the documented v1 surface: `/v1/health`, `/v1/models`,
`/v1/describe`, `/v1/walk`, `/v1/select`, `/v1/relations`, `/v1/stats`,
`/v1/infer`, `/v1/explain-infer`, `/v1/insert`, `/v1/patches`,
`/v1/patches/apply`, `/v1/patches/{name}`, `/v1/walk-ffn`,
`/v1/walk-ffn-q8k`, `/v1/expert/topology`, `/v1/expert/batch`,
`/v1/experts/layer-batch`, `/v1/experts/layer-batch-f16`,
`/v1/experts/multi-layer-batch`, `/v1/experts/multi-layer-batch-q8k`,
`/v1/expert/{layer}/{expert_id}`, `/v1/stream`, `/v1/warmup`,
`/v1/embed`, `/v1/embed/{token_id}`, `/v1/logits`, `/v1/token/encode`,
`/v1/token/decode`, plus the OpenAI-compat `/v1/embeddings`,
`/v1/completions`, and `/v1/chat/completions`. The multi-model router
MUST mirror the per-model surface under `/v1/{model_id}/...` for every
endpoint that targets a specific model.

#### Scenario: GET /v1/health returns 200 with required fields
- **WHEN** a client issues `GET /v1/health` against a single-model router
- **THEN** the response status SHALL be `200 OK`, the body SHALL be JSON containing `status`, `uptime_seconds`, and `requests_served`, and the request counter SHALL be incremented
<!-- test: larql_server::test_http_core::http_health_returns_200 -->
<!-- test: larql_server::test_http_core::http_health_body_has_required_fields -->
<!-- test: larql_server::test_http_core::http_health_bumps_request_counter -->

#### Scenario: GET /v1/models lists configured models
- **WHEN** `GET /v1/models` is issued against single- and multi-model routers
- **THEN** a single-model router SHALL return one entry, the path is the canonical `/v1/models`, and the multi-model router SHALL list every configured model with its id
<!-- test: larql_server::test_http_core::http_models_single_lists_one_model -->
<!-- test: larql_server::test_http_core::http_models_single_path_is_v1 -->
<!-- test: larql_server::test_http_core::http_models_multi_path_includes_model_id -->

#### Scenario: Multi-model routes accept `/v1/{model_id}/...`
- **WHEN** the multi-model router is queried with a known and unknown model id on `/v1/{model_id}/stats`
- **THEN** the known model SHALL return `200 OK` and the unknown model SHALL return `404 Not Found`
<!-- test: larql_server::test_http_core::http_multi_stats_valid_model_returns_200 -->
<!-- test: larql_server::test_http_core::http_multi_stats_unknown_model_returns_404 -->

### Requirement: Bearer-token authentication policy

When an API key is configured, the server SHALL require `Authorization:
Bearer <key>` on every authenticated route, MUST reject missing,
malformed, or wrong tokens with `401 Unauthorized`, and SHALL exempt
`/v1/health` so liveness probes work without credentials. When no key
is configured, all routes SHALL be reachable.

#### Scenario: No API key configured allows all requests
- **WHEN** the server boots without an API key and a client calls any endpoint without `Authorization`
- **THEN** the request SHALL succeed
<!-- test: larql_server::test_http_core::http_auth_no_api_key_configured_allows_all -->

#### Scenario: Correct bearer token is accepted, wrong token rejected
- **WHEN** an API key is configured and the client presents a correct, missing, malformed (non-`Bearer`), or wrong bearer
- **THEN** the correct bearer SHALL return `200`, and missing/malformed/wrong SHALL return `401`
<!-- test: larql_server::test_http_core::http_auth_correct_bearer_returns_200 -->
<!-- test: larql_server::test_http_core::http_auth_wrong_bearer_returns_401 -->
<!-- test: larql_server::test_http_core::http_auth_missing_header_returns_401 -->
<!-- test: larql_server::test_http_core::http_auth_non_bearer_format_rejected -->

#### Scenario: Health endpoint exempt from authentication
- **WHEN** an API key is configured and `GET /v1/health` is issued without an `Authorization` header
- **THEN** the response SHALL be `200 OK`
<!-- test: larql_server::test_http_core::http_auth_health_exempt_without_key -->

### Requirement: Knowledge-graph query endpoints (describe/walk/select/relations)

The server SHALL implement a knowledge-graph DSL across `GET
/v1/describe`, `GET /v1/walk`, `POST /v1/select`, and `GET
/v1/relations`: describe MUST support `band`
(`syntax|knowledge|output|all`) and `verbose`, walk MUST parse
`layers` as a range (`24-33`) or list (`14,26,27`), select MUST honor
`entity`, `relation`, `layer`, `limit`, `min_confidence`, `order_by`,
and `order` (`asc|desc`), and relations MUST report a list with
`count` per relation.

#### Scenario: Describe returns 200 with entity field and edge list
- **WHEN** `GET /v1/describe?entity=...` is issued against a model with a populated tokenizer
- **THEN** the response SHALL be `200 OK` with an `entity` field, an `edges` array, and missing entities SHALL return `400`
<!-- test: larql_server::test_http_describe::http_describe_returns_200_with_entity_field -->
<!-- test: larql_server::test_http_describe::http_describe_empty_vocab_returns_empty_edges -->
<!-- test: larql_server::test_http_describe::http_describe_missing_entity_returns_400 -->

#### Scenario: Describe band parameter selects layers
- **WHEN** `band=syntax`, `band=output`, or `band=all` is supplied
- **THEN** the response SHALL be `200 OK` and edges SHALL be filtered to the matching band
<!-- test: larql_server::test_http_describe::http_describe_band_syntax_returns_200 -->
<!-- test: larql_server::test_http_describe::http_describe_band_output_returns_200 -->
<!-- test: larql_server::test_http_describe::http_describe_band_all_returns_200 -->
<!-- test: larql_server::test_http_full_routes::http_describe_functional_band_syntax -->

#### Scenario: Describe verbose mode and probe labels enrich edges
- **WHEN** describe is called with `verbose=true` against a model loaded with probe labels
- **THEN** the response SHALL include verbose fields and probe-labelled edges SHALL carry `relation` and `source` fields
<!-- test: larql_server::test_http_describe::http_describe_verbose_mode_returns_200 -->
<!-- test: larql_server::test_http_full_routes::http_describe_with_probe_label_includes_relation_and_source -->

#### Scenario: Walk returns hits and respects layer filters
- **WHEN** `GET /v1/walk` is issued with `layers=24-33` or `layers=14,26,27`
- **THEN** the response SHALL contain hits restricted to the requested layers, an out-of-bounds layer SHALL still return `200` with no hits, and the response SHALL include a `prompt` field
<!-- test: larql_server::test_http_full_routes::http_walk_functional_returns_hits -->
<!-- test: larql_server::test_http_full_routes::http_walk_functional_with_layer_range -->
<!-- test: larql_server::test_http_full_routes::http_walk_functional_with_layer_list -->
<!-- test: larql_server::test_http_full_routes::http_walk_functional_with_oob_layer -->
<!-- test: larql_server::test_http_full_routes::http_walk_functional_response_has_prompt_field -->

#### Scenario: Select supports filter, ordering, and limit
- **WHEN** `POST /v1/select` is issued with combinations of layer, entity, min_confidence, relation, limit, order_by, and order
- **THEN** the response SHALL apply each filter, ordering ascending vs. descending SHALL produce inverse orderings of `c_score`, and `limit` SHALL truncate the result set
<!-- test: larql_server::test_http_select::http_select_no_filter_returns_all_features -->
<!-- test: larql_server::test_http_select::http_select_layer_filter_returns_correct_features -->
<!-- test: larql_server::test_http_select::http_select_entity_filter -->
<!-- test: larql_server::test_http_select::http_select_min_confidence_filter -->
<!-- test: larql_server::test_http_select::http_select_relation_filter_returns_labelled_features -->
<!-- test: larql_server::test_http_select::http_select_limit_truncates_results -->
<!-- test: larql_server::test_http_select::http_select_order_asc_returns_lowest_confidence_first -->
<!-- test: larql_server::test_http_select::http_select_order_desc_returns_highest_confidence_first -->
<!-- test: larql_server::test_http_select::http_select_order_by_layer_asc -->

#### Scenario: Relations returns JSON list with probe counts
- **WHEN** `GET /v1/relations` is issued against a model carrying probe labels
- **THEN** the response SHALL be a JSON list and the `count` field SHALL reflect the number of labelled edges per relation
<!-- test: larql_server::test_http_select::http_relations_returns_json_structure -->
<!-- test: larql_server::test_http_select::http_relations_probe_count_reflects_labels -->

### Requirement: ETag caching for describe and walk responses

Read-only describe (and similar idempotent endpoints) SHALL emit an
`ETag` header derived from the model state and request parameters,
SHALL serve cache hits when the request repeats, and MUST return
`304 Not Modified` when a client sends a matching `If-None-Match`.

#### Scenario: Describe carries an ETag and serves cache hits
- **WHEN** describe is called twice with identical parameters
- **THEN** the first response SHALL carry an `ETag` header, the second SHALL be served from the cache with the same ETag, and a request whose `If-None-Match` matches that ETag SHALL receive `304 Not Modified`
<!-- test: larql_server::test_http_describe::http_describe_has_etag_header -->
<!-- test: larql_server::test_http_describe::http_describe_cache_hit_returns_cached_response -->
<!-- test: larql_server::test_http_describe::http_describe_if_none_match_returns_304 -->
<!-- test: larql_server::test_http_full_routes::http_describe_functional_cache_hit_same_etag -->

### Requirement: Inference, explain, insert, and stream endpoints

The server SHALL implement the per-model inference surface across
`POST /v1/infer`, `POST /v1/explain-infer`, `POST /v1/insert`, and
`GET /v1/stream`: infer MUST return `503 Service Unavailable` when
inference weights are not loaded, explain MUST return `503` when
weights are absent, insert MUST accept `embedding`/`constellation`
modes and respect a session header, and the stream endpoint SHALL
bump request counters per call. Missing required fields SHALL return
`422 Unprocessable Entity` for infer and `400 Bad Request` for
ill-formed JSON.

#### Scenario: Infer rejects missing prompt and reports unavailable when disabled
- **WHEN** `POST /v1/infer` is issued without a `prompt` or against a server with inference disabled
- **THEN** the missing-prompt case SHALL return `422` and the disabled case SHALL return `503`
<!-- test: larql_server::test_http_mutations::http_infer_missing_prompt_returns_422 -->
<!-- test: larql_server::test_http_mutations::http_infer_disabled_returns_503 -->
<!-- test: larql_server::test_http_mutations::http_infer_no_weights_check_returns_503 -->
<!-- test: larql_server::test_http_mutations::http_infer_bumps_request_counter -->

#### Scenario: Explain returns 503 without weights
- **WHEN** `POST /v1/explain-infer` is issued against a server without weights loaded
- **THEN** the response SHALL be `503 Service Unavailable` and the request counter SHALL still increment
<!-- test: larql_server::test_http_mutations::http_explain_no_weights_returns_503 -->
<!-- test: larql_server::test_http_mutations::http_explain_bumps_request_counter -->
<!-- test: larql_server::test_http_mutations::http_explain_multi_model_not_found_returns_404 -->

#### Scenario: Insert accepts embedding mode and explicit layer; honors session header
- **WHEN** `POST /v1/insert` is issued in embedding mode, with an explicit layer, or with a session header
- **THEN** each call SHALL return `200 OK`, the session-scoped call SHALL include a `session` field in the response, and an unknown multi-model id SHALL return `404`
<!-- test: larql_server::test_http_mutations::http_insert_returns_200_with_embedding_mode -->
<!-- test: larql_server::test_http_mutations::http_insert_with_explicit_layer_returns_200 -->
<!-- test: larql_server::test_http_mutations::http_insert_with_session_header_returns_session_field -->
<!-- test: larql_server::test_http_mutations::http_insert_multi_model_not_found_returns_404 -->
<!-- test: larql_server::test_http_full_routes::http_insert_functional_with_tokenizer -->

#### Scenario: Walk-FFN features-only and full-output modes
- **WHEN** `POST /v1/walk-ffn` is called in features-only and full-output modes via JSON or binary wire
- **THEN** features-only requests SHALL return per-layer features and scores, missing/wrong residual sizes SHALL return `400`, binary requests without `full_output` SHALL return `400`, and successful responses SHALL include `latency_ms`
<!-- test: larql_server::test_http_full_routes::http_walk_ffn_features_single_layer_returns_200 -->
<!-- test: larql_server::test_http_full_routes::http_walk_ffn_features_single_layer_top_hit_is_feature_0 -->
<!-- test: larql_server::test_http_full_routes::http_walk_ffn_features_layers_array_single_returns_layer_format -->
<!-- test: larql_server::test_http_full_routes::http_walk_ffn_missing_layer_returns_400 -->
<!-- test: larql_server::test_http_full_routes::http_walk_ffn_wrong_residual_size_returns_400 -->
<!-- test: larql_server::test_http_full_routes::http_walk_ffn_binary_without_full_output_returns_400 -->
<!-- test: larql_server::test_http_full_routes::http_walk_ffn_latency_ms_in_response -->

### Requirement: Stats, warmup, embed, logits, and tokenizer endpoints

`GET /v1/stats` SHALL surface model metadata, `mode` (`full`,
`ffn-service`, `embed-service`), `family`, and per-layer band info.
`POST /v1/warmup` SHALL accept `skip_weights`, an empty body, and
explicit layer lists, returning a prefetch count. The embed surface —
`POST /v1/embed`, `GET /v1/embed/{token_id}`, `POST /v1/logits`, `GET
/v1/token/encode`, `GET /v1/token/decode` — SHALL accept JSON and
binary content types, MUST validate token ranges and lengths, and SHALL
return `400` for malformed input and `404` when no model is loaded.

#### Scenario: Stats reflects server mode and exposes layer bands
- **WHEN** `GET /v1/stats` is issued in full, FFN-only, and embed-only modes
- **THEN** `mode` SHALL be `full`, `ffn-service`, or `embed-service` respectively and the response SHALL contain a `layer_bands` shape with knowledge/syntax/output arrays
<!-- test: larql_server::test_http_core::http_stats_returns_model_info -->
<!-- test: larql_server::test_http_core::http_stats_mode_full_by_default -->
<!-- test: larql_server::test_http_core::http_stats_mode_ffn_service_when_ffn_only -->
<!-- test: larql_server::test_http_core::http_stats_mode_embed_service_when_embed_only -->
<!-- test: larql_server::test_http_core::http_stats_layer_bands_shape -->

#### Scenario: Warmup variants and missing-model behaviour
- **WHEN** `POST /v1/warmup` is issued with `skip_weights`, an empty body, an explicit layer list, an out-of-range layer list, or against a server with no model
- **THEN** the first three SHALL return `200 OK` (with prefetch counts where applicable), out-of-range layers SHALL produce zero prefetches, and the no-model case SHALL return `404`
<!-- test: larql_server::test_http_core::http_warmup_no_model_returns_404 -->
<!-- test: larql_server::test_http_mutations::http_warmup_skip_weights_returns_200 -->
<!-- test: larql_server::test_http_mutations::http_warmup_empty_body_returns_200 -->
<!-- test: larql_server::test_http_mutations::http_warmup_with_layer_list_returns_prefetch_count -->
<!-- test: larql_server::test_http_mutations::http_warmup_with_out_of_range_layers_returns_zero_prefetch -->

#### Scenario: Embed and logits validate input shape and content type
- **WHEN** the client posts valid token ids, an empty list, an out-of-range token, malformed JSON, a binary payload (truncated or odd length), and a hidden-size mismatch on logits
- **THEN** valid input SHALL return `200`, every malformed case SHALL return `400`, single-token GET SHALL respect `Accept: application/json`, an out-of-range single token SHALL return `400`, and the binary embed response SHALL be returned as binary
<!-- test: larql_server::test_http_embed::http_embed_valid_token_ids_returns_200 -->
<!-- test: larql_server::test_http_embed::http_embed_empty_token_ids_returns_400 -->
<!-- test: larql_server::test_http_embed::http_embed_out_of_range_token_returns_400 -->
<!-- test: larql_server::test_http_embed::http_embed_single_token_returns_correct_shape -->
<!-- test: larql_server::test_http_embed::http_embed_invalid_json_returns_400 -->
<!-- test: larql_server::test_http_embed::http_embed_no_model_returns_404 -->
<!-- test: larql_server::test_http_embed::http_embed_binary_returns_binary_response -->
<!-- test: larql_server::test_http_embed::http_embed_binary_truncated_returns_400 -->
<!-- test: larql_server::test_http_embed::http_embed_single_get_returns_200 -->
<!-- test: larql_server::test_http_embed::http_embed_single_get_json_accept_returns_json -->
<!-- test: larql_server::test_http_embed::http_embed_single_get_out_of_range_returns_400 -->
<!-- test: larql_server::test_http_embed::http_logits_invalid_json_returns_400 -->
<!-- test: larql_server::test_http_embed::http_logits_binary_odd_length_returns_400 -->
<!-- test: larql_server::test_http_embed::http_logits_hidden_mismatch_returns_400 -->
<!-- test: larql_server::test_http_embed::http_logits_binary_hidden_mismatch_returns_400 -->
<!-- test: larql_server::test_http_embed::http_logits_no_model_returns_404 -->

#### Scenario: Token encode/decode contracts
- **WHEN** `GET /v1/token/encode?text=...` and `GET /v1/token/decode?ids=...` are issued
- **THEN** valid input SHALL return `200`, missing `text` or `ids` SHALL return `400`, an empty `ids` SHALL still return `200`, and an invalid id SHALL return `400`
<!-- test: larql_server::test_http_embed::http_token_encode_returns_200 -->
<!-- test: larql_server::test_http_embed::http_token_encode_missing_text_returns_400 -->
<!-- test: larql_server::test_http_embed::http_token_decode_empty_ids_returns_200 -->
<!-- test: larql_server::test_http_embed::http_token_decode_invalid_id_returns_400 -->
<!-- test: larql_server::test_http_embed::http_token_decode_missing_ids_param_returns_400 -->

### Requirement: OpenAI-compatible endpoints

`/v1/embeddings`, `/v1/completions`, and `/v1/chat/completions` SHALL
implement the OpenAI-compat surface: embeddings MUST accept
string/array/pretokenised input, support `format` (`float|base64`),
and reject empty or unknown formats with `400`. Completions and chat
MUST reject `n>1`, MUST reject streaming combined with `echo` or
batched prompts, MUST emit `text/event-stream` for stream requests,
MUST return `503` when inference is disabled, MUST return `404` for
unknown multi-model ids, MUST accept `tools`, `tool_choice`,
`response_format` (`text`, `json_object`, `json_schema`), and standard
sampling parameters, and SHALL return `400` for invalid roles,
unknown tool choices, malformed JSON-schema, missing tool-call ids on
tool messages, or assistant messages without content/tool calls.

#### Scenario: Embeddings input shapes and formats
- **WHEN** `POST /v1/embeddings` is issued with a string, a string array, pre-tokenised ids, base64 format, an unknown format, or empty input
- **THEN** valid shapes SHALL return `200` with the right `data[]` shape (pooled or indexed; base64 returns strings), and unknown/empty SHALL return `400`
<!-- test: larql_server::test_http_embed::http_openai_embeddings_string_input_returns_200_with_pooled_vector -->
<!-- test: larql_server::test_http_embed::http_openai_embeddings_string_array_returns_indexed_data -->
<!-- test: larql_server::test_http_embed::http_openai_embeddings_pretokenised_single_works -->
<!-- test: larql_server::test_http_embed::http_openai_embeddings_base64_format_returns_string -->
<!-- test: larql_server::test_http_embed::http_openai_embeddings_unknown_format_returns_400 -->
<!-- test: larql_server::test_http_embed::http_openai_embeddings_empty_input_returns_400 -->

#### Scenario: Completions and chat streaming and limits
- **WHEN** completions or chat is requested with `n>1`, streaming combined with `echo` or batched prompts, or missing prompt
- **THEN** all of these SHALL return `400` (or `422` for missing prompt) and a valid streaming request SHALL produce `text/event-stream`
<!-- test: larql_server::test_http_embed::http_openai_completions_n_gt_1_returns_400 -->
<!-- test: larql_server::test_http_embed::http_openai_completions_stream_with_echo_returns_400 -->
<!-- test: larql_server::test_http_embed::http_openai_completions_stream_with_batched_prompts_returns_400 -->
<!-- test: larql_server::test_http_embed::http_openai_completions_stream_returns_event_stream_content_type -->
<!-- test: larql_server::test_http_embed::http_openai_completions_missing_prompt_returns_422 -->
<!-- test: larql_server::test_http_embed::http_openai_chat_n_gt_1_returns_400 -->
<!-- test: larql_server::test_http_embed::http_openai_chat_stream_returns_event_stream_content_type -->

#### Scenario: Chat tools, tool_choice, and response_format
- **WHEN** chat is called with tools, a specific tool choice, an unknown tool choice, `tool_choice=none`, tool messages with/without `tool_call_id`, an assistant tool replay, an assistant message lacking content/tool calls, and various `response_format` types (text, json_object, json_schema with valid/invalid schema)
- **THEN** valid combinations SHALL return `200` (streaming where requested), and unknown choices, missing tool_call_id, missing assistant content, malformed/unknown response_format SHALL return `400`
<!-- test: larql_server::test_http_embed::http_openai_chat_tools_are_accepted -->
<!-- test: larql_server::test_http_embed::http_openai_chat_tools_with_specific_choice_is_accepted -->
<!-- test: larql_server::test_http_embed::http_openai_chat_tools_unknown_choice_returns_400 -->
<!-- test: larql_server::test_http_embed::http_openai_chat_tools_with_stream_returns_event_stream -->
<!-- test: larql_server::test_http_embed::http_openai_chat_tool_choice_none_skips_constraint -->
<!-- test: larql_server::test_http_embed::http_openai_chat_tool_message_without_tool_call_id_returns_400 -->
<!-- test: larql_server::test_http_embed::http_openai_chat_tool_replay_is_accepted -->
<!-- test: larql_server::test_http_embed::http_openai_chat_assistant_with_only_tool_calls_is_accepted -->
<!-- test: larql_server::test_http_embed::http_openai_chat_assistant_with_no_content_or_tools_returns_400 -->
<!-- test: larql_server::test_http_embed::http_openai_chat_response_format_json_schema_is_accepted -->
<!-- test: larql_server::test_http_embed::http_openai_chat_response_format_json_schema_missing_schema_field_returns_400 -->
<!-- test: larql_server::test_http_embed::http_openai_chat_response_format_json_schema_invalid_returns_400 -->
<!-- test: larql_server::test_http_embed::http_openai_chat_response_format_text_is_accepted -->
<!-- test: larql_server::test_http_embed::http_openai_chat_response_format_json_object_is_accepted -->
<!-- test: larql_server::test_http_embed::http_openai_chat_response_format_unknown_type_returns_400 -->
<!-- test: larql_server::test_http_embed::http_openai_chat_invalid_role_returns_400 -->
<!-- test: larql_server::test_http_embed::http_openai_chat_empty_messages_returns_400 -->

#### Scenario: OpenAI multi-model routing and inference-disabled behaviour
- **WHEN** the server runs in multi-model mode and the client routes via the `model` body field (known/unknown), or runs single-model with no `model` field, or is called with inference disabled
- **THEN** known model SHALL succeed, unknown SHALL return `404`, single-model SHALL accept the call without `model`, and inference-disabled completions/chat SHALL return `503`
<!-- test: larql_server::test_http_embed::http_openai_models_multi_lists_all_with_openai_shape -->
<!-- test: larql_server::test_http_embed::http_openai_embeddings_multi_routes_via_model_field -->
<!-- test: larql_server::test_http_embed::http_openai_embeddings_multi_unknown_model_returns_404 -->
<!-- test: larql_server::test_http_embed::http_openai_embeddings_no_model_field_in_single_model_works -->
<!-- test: larql_server::test_http_embed::http_openai_completions_multi_routes_via_model_field -->
<!-- test: larql_server::test_http_embed::http_openai_completions_multi_unknown_model_returns_404 -->
<!-- test: larql_server::test_http_embed::http_openai_completions_infer_disabled_returns_503 -->
<!-- test: larql_server::test_http_embed::http_openai_chat_multi_routes_via_model_field -->
<!-- test: larql_server::test_http_embed::http_openai_chat_multi_unknown_model_returns_404 -->
<!-- test: larql_server::test_http_embed::http_openai_chat_infer_disabled_returns_503 -->

### Requirement: Patches and session-scoped operations

The server SHALL support patch lifecycle and session-scoped patches via
`POST /v1/patches/apply`, `GET /v1/patches`, `DELETE
/v1/patches/{name}`, and their `/v1/{model_id}/...` counterparts:
inline patch JSON, named patch removal, and session-scoped patch
lists MUST be keyed by an `X-LARQL-Session` header. Session list /
apply / remove SHALL include the session id in the response, removing
a non-existent patch SHALL return `404`, and patches MUST also be
applicable in `insert` op form against a tokenized model.

#### Scenario: Patches list / apply / delete lifecycle
- **WHEN** patches are listed (empty), applied inline, listed again, deleted by name, and a non-existent patch is deleted
- **THEN** the empty list SHALL be `[]`, apply SHALL return `200`, the post-apply list SHALL contain the patch, delete by name SHALL return `200`, and delete-nonexistent SHALL return `404`. Apply with neither `url` nor `patch` SHALL return `400`
<!-- test: larql_server::test_http_patches::http_patches_list_empty_returns_empty_array -->
<!-- test: larql_server::test_http_patches::http_patches_apply_no_url_no_patch_returns_400 -->
<!-- test: larql_server::test_http_patches::http_patches_apply_inline_returns_200 -->
<!-- test: larql_server::test_http_patches::http_patches_list_after_apply_shows_patch -->
<!-- test: larql_server::test_http_patches::http_patches_delete_named_returns_200 -->
<!-- test: larql_server::test_http_patches::http_patches_delete_nonexistent_returns_404 -->

#### Scenario: Session-scoped patch operations
- **WHEN** a request supplies an `X-LARQL-Session` header on apply / list / remove
- **THEN** every response SHALL include a `session` field, and the session manager SHALL list applied patches and reject removals from unknown sessions or unknown names
<!-- test: larql_server::test_http_patches::http_patches_session_list_returns_session_field -->
<!-- test: larql_server::test_http_patches::http_patches_session_apply_returns_session_field -->
<!-- test: larql_server::test_http_patches::http_patches_session_list_after_session_apply -->
<!-- test: larql_server::test_http_session::session_manager_list_empty_for_unknown_session -->
<!-- test: larql_server::test_http_session::session_manager_apply_patch_and_list -->
<!-- test: larql_server::test_http_session::session_manager_remove_patch_by_name -->
<!-- test: larql_server::test_http_session::session_manager_remove_nonexistent_patch_returns_err -->
<!-- test: larql_server::test_http_session::session_manager_remove_from_unknown_session_returns_err -->

#### Scenario: Multi-model patch routing
- **WHEN** patches are listed, deleted, or applied against `/v1/{model_id}/patches[...]` for known and unknown ids
- **THEN** known ids SHALL behave like the single-model surface and unknown ids SHALL return `404`
<!-- test: larql_server::test_http_full_routes::http_patches_list_multi_model_returns_200 -->
<!-- test: larql_server::test_http_full_routes::http_patches_list_multi_model_not_found -->
<!-- test: larql_server::test_http_full_routes::http_patches_delete_multi_model_not_found -->
<!-- test: larql_server::test_http_full_routes::http_patches_delete_multi_model_applies_and_removes -->
<!-- test: larql_server::test_http_full_routes::http_patches_apply_insert_op_enrich_with_functional_tokenizer -->
<!-- test: larql_server::test_http_full_routes::http_patches_session_remove_returns_session_field -->

### Requirement: Error envelope and request-counter contract

All error responses (`400`, `404`, `500`, `503`) SHALL carry a JSON
body with an `error` key. Every served request — successful or failed —
SHALL increment the global `requests_served` counter exposed via
`/v1/health`.

#### Scenario: Error bodies always include an `error` key
- **WHEN** the server returns `404`, `400`, `500`, or `503`
- **THEN** each body SHALL be JSON containing an `error` field
<!-- test: larql_server::test_http_core::http_server_error_not_found_body_has_error_key -->
<!-- test: larql_server::test_http_core::http_server_error_bad_request_body_has_error_key -->
<!-- test: larql_server::test_http_core::http_server_error_internal_body_has_error_key -->
<!-- test: larql_server::test_http_core::http_server_error_unavailable_body_has_error_key -->

#### Scenario: requests_served increments per request
- **WHEN** any served route is invoked
- **THEN** the `requests_served` field on `/v1/health` SHALL increase by one and select-style endpoints SHALL also bump the counter
<!-- test: larql_server::test_http_core::http_requests_served_increments_per_request -->
<!-- test: larql_server::test_http_core::http_select_increments_request_counter -->
<!-- test: larql_server::test_http_mutations::http_walk_bumps_request_counter -->

### Requirement: Test suite coverage

The capability's behavior SHALL be exercised by every test in the test modules listed below. New tests added to those modules SHALL be considered part of this capability's contract by default.

#### Scenario: Workspace tests bound to this capability
- **WHEN** the trace tool resolves the wildcard module references for this capability
- **THEN** every backing test SHALL resolve to a real `#[test]` function in the workspace
<!-- test: larql_server::test_http_select::**::* -->
<!-- test: larql_server::test_http_core::**::* -->
<!-- test: larql_server::test_http_describe::**::* -->
<!-- test: larql_server::test_http_full_routes::**::* -->
<!-- test: larql_server::test_http_mutations::**::* -->
<!-- test: larql_server::test_http_embed::**::* -->
<!-- test: larql_server::test_http_patches::**::* -->
<!-- test: larql_server::test_http_session::**::* -->
<!-- test: larql_server::routes::openai::schema::fsm::tests::**::* -->
<!-- test: larql_server::routes::openai::util::tests::**::* -->
<!-- test: larql_server::routes::embed::tests::**::* -->
