## ADDED Requirements

### Requirement: API key authentication middleware

When `AppState::api_key` is `Some`, the `auth_middleware` SHALL
validate every request's `Authorization: Bearer <token>` header
against the configured key and respond with HTTP 401 on mismatch or
absence. Requests targeting `HEALTH_PATH` SHALL be exempt so probes
keep working without credentials. When `api_key` is `None`, the
middleware MUST pass every request through unchanged so unauthenticated
deployments stay functional.

#### Scenario: Bearer prefix unwraps the token before comparison
- **WHEN** an incoming request carries `Authorization: Bearer <key>` matching `AppState::api_key`
- **THEN** the request SHALL be forwarded to the inner handler; mismatched tokens SHALL return 401 and absent headers SHALL also return 401
<!-- test: larql_server::test_unit_state::test_bearer_token_extraction -->
<!-- test: larql_server::test_unit_state::test_bearer_token_mismatch -->
<!-- test: larql_server::test_unit_state::test_no_auth_header -->

#### Scenario: Health endpoint is exempt from auth
- **WHEN** the auth middleware sees a request whose path equals `HEALTH_PATH` regardless of the Authorization header
- **THEN** the request SHALL be forwarded without comparing the token (so liveness probes keep working with or without credentials)
<!-- test: larql_server::test_unit_state::test_health_exempt_from_auth -->

### Requirement: Per-IP rate limiting with optional X-Forwarded-For trust

`RateLimiter::parse` SHALL accept short (`100/min`, `10/sec`,
`5/hour`) and long (`100/minute`, `10/second`) forms; invalid or
zero-count specs MUST return `None`. The token-bucket implementation
SHALL allow a burst up to the configured count, refill at the rate
implied by the spec, and key buckets per `IpAddr` so different
clients are independent. The middleware MUST read the bucket key from
the socket peer address by default, and MUST trust the first
`X-Forwarded-For` entry only when `trust_forwarded_for` is enabled —
matching the operator's contract that XFF is reserved for trusted
reverse-proxy deployments. Stale buckets SHALL be evictable via
`evict_stale`.

#### Scenario: Rate-limit specs parse short and long forms
- **WHEN** `RateLimiter::parse` is given `100/min`, `10/sec`, `5/hour`, or their long-form equivalents
- **THEN** a `Some(RateLimiter)` SHALL be returned with the count and refill rate that matches the spec
<!-- test: larql_server::ratelimit::tests::parse_per_minute -->
<!-- test: larql_server::ratelimit::tests::parse_per_second -->
<!-- test: larql_server::ratelimit::tests::parse_per_hour -->
<!-- test: larql_server::ratelimit::tests::parse_short_forms -->
<!-- test: larql_server::test_unit_state::test_rate_limiter_per_minute_long_form -->
<!-- test: larql_server::test_unit_state::test_rate_limiter_per_second_long_form -->
<!-- test: larql_server::test_unit_state::test_rate_limiter_fractional_count -->

#### Scenario: Invalid or zero-count specs are rejected
- **WHEN** `RateLimiter::parse` is given malformed input (e.g. empty string, `"0/min"`, missing slash, non-numeric count)
- **THEN** `None` SHALL be returned so the server logs an explicit warning rather than installing a broken limiter
<!-- test: larql_server::ratelimit::tests::parse_invalid -->
<!-- test: larql_server::test_unit_state::test_rate_limiter_zero_count_rejects_immediately -->
<!-- test: larql_server::test_unit_state::test_rate_limiter_empty_spec_rejects -->

#### Scenario: Token bucket allows burst then throttles per IP
- **WHEN** the limiter receives `count + 1` requests from the same IP in rapid succession
- **THEN** the first `count` SHALL be allowed and the next SHALL be rejected, while a different IP's bucket SHALL remain unaffected
<!-- test: larql_server::ratelimit::tests::token_bucket_allows_burst -->
<!-- test: larql_server::ratelimit::tests::different_ips_independent -->
<!-- test: larql_server::test_unit_state::test_rate_limit_token_bucket -->

#### Scenario: Stale buckets can be evicted to bound memory
- **WHEN** `evict_stale` is called after the configured staleness window
- **THEN** entries that have not been touched within the window SHALL be removed from the bucket map
<!-- test: larql_server::ratelimit::tests::evict_stale_removes_old_entries -->

### Requirement: Strong ETag generation and conditional response

`compute_etag(body)` SHALL produce a deterministic, double-quoted hex
string for any `serde_json::Value`; identical bodies MUST produce
identical ETags and key-order changes SHALL produce different ETags
(no canonical-key reordering). `matches_etag(if_none_match, etag)`
MUST return `true` for the wildcard `*`, for an exact match (with
optional surrounding whitespace), and `false` for any mismatch or
absent header. Endpoints SHALL use these helpers to short-circuit to
HTTP 304 when the client's `If-None-Match` matches the freshly
computed ETag.

#### Scenario: Same body produces the same quoted ETag
- **WHEN** `compute_etag` is called twice on equal `serde_json::Value` inputs
- **THEN** the returned strings SHALL be identical, double-quoted, and stable across runs
<!-- test: larql_server::etag::tests::etag_is_quoted -->
<!-- test: larql_server::etag::tests::same_body_same_etag -->
<!-- test: larql_server::test_unit_state::test_etag_deterministic -->
<!-- test: larql_server::test_unit_state::test_etag_format -->
<!-- test: larql_server::test_unit_state::test_etag_empty_object_is_valid -->

#### Scenario: Different bodies and key orders produce different ETags
- **WHEN** `compute_etag` is called on bodies that differ in value or in key serialisation order
- **THEN** the returned strings SHALL differ
<!-- test: larql_server::etag::tests::different_body_different_etag -->
<!-- test: larql_server::test_unit_state::test_etag_different_key_order_produces_different_hash -->

#### Scenario: matches_etag honors wildcard, whitespace, and mismatch
- **WHEN** `matches_etag` is called with the wildcard `*`, an exact-quoted token, the same token padded with whitespace, or a non-matching token
- **THEN** it SHALL return `true` for the wildcard, exact, and whitespace-padded cases, and `false` otherwise
<!-- test: larql_server::etag::tests::matches_exact -->
<!-- test: larql_server::etag::tests::matches_wildcard -->
<!-- test: larql_server::etag::tests::no_match_on_none -->
<!-- test: larql_server::etag::tests::no_match_on_different -->
<!-- test: larql_server::test_unit_state::test_if_none_match_comparison -->
<!-- test: larql_server::test_unit_state::test_304_not_modified_condition -->
<!-- test: larql_server::test_unit_state::test_matches_etag_extra_whitespace -->
<!-- test: larql_server::test_unit_state::test_matches_etag_mismatch_returns_false -->

### Requirement: Describe-result LRU cache

`DescribeCache::new(ttl_secs)` SHALL be enabled only when
`ttl_secs > 0`. Cache keys SHALL be assembled by
`DescribeCache::key(model_id, entity, band, limit, min_score)` and
SHALL include every parameter that affects the response so distinct
queries cannot collide. Entries that exceed the TTL SHALL be
considered absent on read, and `put` SHALL overwrite stale or existing
entries.

#### Scenario: Cache disabled when TTL is zero
- **WHEN** `DescribeCache::new(0)` is constructed
- **THEN** `is_enabled()` SHALL return `false`, and the server SHALL skip cache lookups entirely
<!-- test: larql_server::cache::tests::disabled_when_ttl_zero -->
<!-- test: larql_server::test_unit_state::test_cache_disabled_when_ttl_zero -->

#### Scenario: Hit, miss, and overwrite semantics
- **WHEN** `put` stores a value under a key, then `get` is called for the same key, then `put` overwrites it, then `get` is called for an unknown key
- **THEN** the first `get` SHALL return `Some` with the stored value, the overwrite SHALL replace it, and the unknown key SHALL return `None`
<!-- test: larql_server::cache::tests::put_and_get -->
<!-- test: larql_server::cache::tests::miss_on_unknown_key -->
<!-- test: larql_server::test_unit_state::test_cache_hit_and_miss -->
<!-- test: larql_server::test_unit_state::test_cache_overwrite_updates_value -->

#### Scenario: Expired entries are treated as absent
- **WHEN** a value is `put` with TTL T, then `get` is called after a duration > T
- **THEN** the entry SHALL be evicted/treated as `None`
<!-- test: larql_server::cache::tests::expired_entry_returns_none -->
<!-- test: larql_server::cache::tests::enabled_when_ttl_nonzero -->

#### Scenario: Cache key includes every salient parameter
- **WHEN** keys are built for differing model id, entity, band, limit, or min_score
- **THEN** every change SHALL produce a different cache key, including float-precision rounding so near-identical floats do not collapse silently
<!-- test: larql_server::cache::tests::key_format -->
<!-- test: larql_server::cache::tests::different_params_different_keys -->
<!-- test: larql_server::test_unit_state::test_cache_key_format -->
<!-- test: larql_server::test_unit_state::test_cache_key_float_precision_truncated -->

### Requirement: FPN binary wire format detection and shape

`wire::has_content_type(headers, expected)` SHALL return `true` only
when the request's `Content-Type` exactly matches `expected` or
matches it with parameters (e.g. `application/x-larql-ffn;
charset=utf-8`). Missing headers and other types MUST yield `false`.
The walk-ffn binary protocol SHALL use a 32-bit little-endian header
in which a single-layer request begins with the layer index and a
batch request begins with the `BATCH_MARKER = 0xFFFFFFFF` constant
followed by a layer count. Binary requests MUST require
`full_output = true` (flags bit 0 set); features-only binary requests
SHALL be rejected. f32 floats MUST round-trip exactly through the
binary encoding.

#### Scenario: Content-Type matching tolerates parameters
- **WHEN** the request carries `application/x-larql-ffn`, `application/x-larql-ffn; charset=utf-8`, a different type, or no header at all
- **THEN** `has_content_type` SHALL return `true` only for the first two
<!-- test: larql_server::wire::tests::matches_exact_type -->
<!-- test: larql_server::wire::tests::matches_with_parameters -->
<!-- test: larql_server::wire::tests::does_not_match_other_type -->
<!-- test: larql_server::wire::tests::missing_header_does_not_match -->

#### Scenario: Single-layer binary request begins with the layer index
- **WHEN** a single-layer binary walk-ffn request is built (`[layer u32 LE][seq_len u32][flags u32][top_k u32][residual f32[]]`)
- **THEN** the first u32 SHALL equal the layer index, the layout SHALL match the documented header, and the residual byte size SHALL equal `seq_len × hidden_size × 4`
<!-- test: larql_server::test_unit_protocol::test_binary_single_request_first_u32_is_layer -->
<!-- test: larql_server::test_unit_protocol::test_binary_single_request_structure -->
<!-- test: larql_server::test_unit_protocol::test_binary_request_residual_size -->
<!-- test: larql_server::test_unit_protocol::test_binary_content_type_constant -->

#### Scenario: Batch binary request begins with BATCH_MARKER
- **WHEN** a batched binary walk-ffn request is built
- **THEN** the first u32 SHALL equal `0xFFFFFFFF`, the layer count and per-layer indices SHALL follow, and the residual SHALL appear once at the end
<!-- test: larql_server::test_unit_protocol::test_binary_batch_request_first_u32_is_marker -->
<!-- test: larql_server::test_unit_protocol::test_binary_batch_request_structure -->
<!-- test: larql_server::test_unit_protocol::test_binary_batch_marker_constant -->

#### Scenario: Binary responses preserve f32 exactly
- **WHEN** a single-layer or batch binary response is produced and re-parsed
- **THEN** all f32 outputs SHALL match the original bit-for-bit
<!-- test: larql_server::test_unit_protocol::test_binary_single_response_structure -->
<!-- test: larql_server::test_unit_protocol::test_binary_batch_response_structure -->
<!-- test: larql_server::test_unit_protocol::test_binary_float_roundtrip_exact -->

#### Scenario: Features-only binary requests are rejected
- **WHEN** a binary walk-ffn request arrives with the `full_output` flag bit clear
- **THEN** the server SHALL reject it (HTTP 400) — only `full_output = true` is wire-supported in binary mode
<!-- test: larql_server::test_unit_protocol::test_binary_features_only_flag_zero -->

### Requirement: Environment-flag feature toggles

`env_flags.rs` SHALL be the single source of truth for `LARQL_*`
runtime knobs. Each accessor (e.g. `moe_timing_enabled`,
`http_timing_enabled`, `no_warmup`, `use_legacy_cpu`,
`use_metal_experts`, `disable_metal_experts`, `disable_q4k_direct`,
`metal_vs_cpu_debug`, `moe_batch_mode`) MUST be cached via
`OnceLock` so repeated calls don't re-read the environment, and every
exposed name MUST start with the `LARQL_` prefix. Names MUST be
unique across the module so two flags never alias the same string.

#### Scenario: All knob names are LARQL-prefixed and unique
- **WHEN** the env-flags module is introspected for every defined `LARQL_*` constant
- **THEN** each name SHALL begin with `LARQL_`, and no two accessors SHALL share the same string
<!-- test: larql_server::env_flags::tests::names_are_larql_prefixed_and_unique -->

### Requirement: Structured error envelope

`ServerError` SHALL map each variant to a stable HTTP status code
(`NotFound → 404`, `BadRequest → 400`, `InferenceUnavailable → 503`,
`Internal → 500`) and SHALL render the body as a single
`{"error": "<message>"}` JSON object. WebSocket streaming responses
retain `{"type": "error", "message": ...}` shape; REST responses MUST
NOT diverge between routes — JSON parse errors, binary protocol
validation, token-id bounds, and model lookup failures SHALL all use
the same envelope.

#### Scenario: Stream error frames keep the streaming envelope shape
- **WHEN** the WebSocket layer-by-layer describe stream emits an error
- **THEN** the JSON message SHALL match `{"type":"error","message":"..."}` rather than the REST `{"error":"..."}` shape
<!-- test: larql_server::test_unit_protocol::test_stream_error_response_format -->
<!-- test: larql_server::test_unit_protocol::test_stream_unknown_type_rejected -->

### Requirement: Per-request session context with patch isolation

`SessionManager` SHALL keep one `PatchedVindex` per session id, with
sessions auto-expiring after the configured TTL. `extract_session_id`
SHALL read the `X-Session-Id` header to pick the bucket; absent
headers SHALL route to the global vindex so unauthenticated REPL
sessions still work. Patches applied to one session MUST NOT affect
the global state or any other session. Removing a non-existent patch
or referencing an unknown session SHALL return an error rather than
panic.

#### Scenario: Session manager creates and counts sessions on demand
- **WHEN** `SessionManager::get_or_create("s1", model)` is called twice and `get_or_create("s2", ...)` once
- **THEN** the first call SHALL register a fresh empty `PatchedVindex` with zero patches, the second call for `"s1"` SHALL not increment the count, and `session_count()` SHALL report 2 once `"s2"` is created
<!-- test: larql_server::test_unit_state::session_get_or_create_new_session_returns_empty_patched -->
<!-- test: larql_server::test_unit_state::session_count_increments_on_first_create -->
<!-- test: larql_server::test_unit_state::session_get_or_create_same_id_does_not_add_session -->

#### Scenario: Removing a patch from an unknown session reports error
- **WHEN** `SessionManager::remove_patch` is called against a session id that does not exist
- **THEN** the result SHALL be `Err` with a message containing `"not found"`
<!-- test: larql_server::test_unit_state::session_remove_patch_from_unknown_session_returns_err -->

#### Scenario: Session header parsing isolates patches per client
- **WHEN** the `X-Session-Id` header is absent / present, and a patch is applied to one session only
- **THEN** the absence SHALL fall through to the shared global vindex, the per-session patch SHALL not appear in another session's `describe`, and the global vindex SHALL stay unaffected
<!-- test: larql_server::test_unit_vindex::test_session_id_header_parsing -->
<!-- test: larql_server::test_unit_vindex::test_session_patch_isolation -->
<!-- test: larql_server::test_unit_vindex::test_session_global_unaffected -->
<!-- test: larql_server::test_unit_vindex::test_session_scoped_describe -->
<!-- test: larql_server::test_unit_vindex::test_session_scoped_walk -->
<!-- test: larql_server::test_unit_vindex::test_session_scoped_select -->

### Requirement: Grid announce identity hash

`announce::vindex_identity_hash(model_id, num_layers)` SHALL return a
deterministic 16-character lowercase-hex string so the router can log
which vindex version any joining shard is serving. The hash MUST be
sensitive to both `model_id` and `num_layers` and stable across
processes/restarts. The grid bearer header SHALL format as
`Authorization: Bearer <key>` so a tonic interceptor can inject it on
every outgoing RPC. The announce/heartbeat/dropping message builders
MUST round-trip every relevant config field for visibility in router
logs.

#### Scenario: Vindex identity hash is deterministic and hex-only
- **WHEN** `vindex_identity_hash` is called twice with the same `(model_id, num_layers)` tuple
- **THEN** the strings SHALL be equal, exactly 16 ASCII hex characters long, and SHALL differ from a hash with a different model id or layer count
<!-- test: larql_server::test_unit_state::vindex_identity_hash_is_deterministic -->
<!-- test: larql_server::test_unit_state::vindex_identity_hash_differs_on_model_id -->
<!-- test: larql_server::test_unit_state::vindex_identity_hash_differs_on_num_layers -->
<!-- test: larql_server::test_unit_state::vindex_identity_hash_is_hex_string -->
<!-- test: larql_server::announce::tests::vindex_identity_hash_is_stable_and_hex -->

#### Scenario: Bearer metadata and message envelopes carry config fields
- **WHEN** `announce_message`, `heartbeat_message`, and `dropping_message` are constructed from an `AnnounceConfig`
- **THEN** the announce envelope SHALL copy every config field, the heartbeat SHALL zero the metrics, the dropping notice SHALL set `reassigned`, and `grid_bearer_value` SHALL produce `"Bearer <key>"`
<!-- test: larql_server::announce::tests::grid_bearer_value_formats_authorization -->
<!-- test: larql_server::announce::tests::announce_message_copies_config_fields -->
<!-- test: larql_server::announce::tests::heartbeat_message_uses_zeroed_metrics -->
<!-- test: larql_server::announce::tests::dropping_message_marks_reassigned -->

### Requirement: Test suite coverage

The capability's behavior SHALL be exercised by every test in the test modules listed below. New tests added to those modules SHALL be considered part of this capability's contract by default.

#### Scenario: Workspace tests bound to this capability
- **WHEN** the trace tool resolves the wildcard module references for this capability
- **THEN** every backing test SHALL resolve to a real `#[test]` function in the workspace
<!-- test: larql_server::test_unit_protocol::**::* -->
<!-- test: larql_server::test_unit_band_utils::**::* -->
<!-- test: larql_server::ratelimit::tests::**::* -->
<!-- test: larql_server::announce::tests::**::* -->
<!-- test: larql_server::etag::tests::**::* -->
<!-- test: larql_server::cache::tests::**::* -->
<!-- test: larql_server::wire::tests::**::* -->
<!-- test: larql_server::env_flags::tests::**::* -->
