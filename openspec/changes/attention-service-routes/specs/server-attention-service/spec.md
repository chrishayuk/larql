## ADDED Requirements

### Requirement: Server SHALL expose session lifecycle endpoints

The attention server SHALL expose three session-lifecycle endpoints:
`POST /v1/attention/session` (create), `GET /v1/attention/session/{id}`
(read state), and `DELETE /v1/attention/session/{id}` (drop).

Session ids SHALL be 128-bit ULIDs (lexicographically sortable). The
session itself owns a `KvCache` whose `kv_format` matches the request,
plus a `last_used: Instant` for TTL eviction.

#### Scenario: create returns a fresh session id

- **WHEN** a client POSTs `/v1/attention/session` with `model_id` and
  no other fields
- **THEN** the response SHALL contain a non-empty `session_id`,
  `layer_range = [0, num_layers)`, and `kv_format = "fp32"`
<!-- test: larql_server::attention_session::tests::insert_and_get_round_trip -->
<!-- test: larql_server::test_http_attention::create_session_returns_id_and_layer_range -->

#### Scenario: get on unknown session id returns 404

- **WHEN** a client GETs `/v1/attention/session/<random-ulid>`
- **THEN** the server SHALL respond `404 Not Found` with body
  `{"error": "no_such_session"}`
<!-- test: larql_server::routes::attention::tests::get_unknown_session_returns_404 -->
<!-- test: larql_server::test_http_attention::get_unknown_session_returns_404 -->

#### Scenario: delete makes get 404

- **WHEN** a client creates a session, then DELETEs it, then GETs it
- **THEN** the GET SHALL return `404 Not Found`
<!-- test: larql_server::attention_session::tests::delete_makes_get_none -->
<!-- test: larql_server::routes::attention::tests::delete_unknown_session_returns_404 -->
<!-- test: larql_server::test_http_attention::delete_session_then_get_returns_404 -->

#### Scenario: idle sessions are reaped after TTL

- **WHEN** a session has been idle for more than `--session-ttl-secs`
  + the reaper's wake interval (30 s)
- **THEN** a subsequent GET SHALL return `404 Not Found`
<!-- test: larql_server::attention_session::tests::reap_drops_idle_sessions -->
<!-- test: unbacked -->

### Requirement: Prefill SHALL populate the cache and return per-layer residuals

`POST /v1/attention/prefill` SHALL run the attention block for every
layer in the session's range, populate the session's KV cache with K
and V projections, and return per-layer post-attention residuals. The
endpoint SHALL accept JSON (default) or binary
(`application/octet-stream`) bodies; the binary form SHALL match the
`MultiLayerBatch` byte layout
(`[u32 layers][u32 seq_len][u32 hidden_dim][f32 layers × seq_len × hidden_dim]`).

#### Scenario: prefill populates all layers

- **WHEN** a client prefills a session with `seq_len = 8` tokens
- **THEN** for every layer in the session's range,
  `cache.is_layer_populated(layer)` SHALL return true after the call
<!-- test: larql_server::test_attention_validation::session_seq_len_advances_after_prefill -->

#### Scenario: prefill returns layers × seq_len × hidden_dim residuals

- **WHEN** a client prefills `seq_len = 8` tokens on a `num_layers =
  4`, `hidden_dim = 16` model
- **THEN** the response SHALL contain a residual array of shape
  `[4, 8, 16]` (JSON) or 4 × 8 × 16 × 4 = 2048 trailing payload bytes
  after the 12-byte header (binary)
<!-- test: larql_server::test_attention_validation::prefill_response_shape_matches_layers_seq_hidden -->
<!-- test: unbacked -->

#### Scenario: prefill rejects unknown session

- **WHEN** a client prefills against a `session_id` that does not exist
- **THEN** the server SHALL return `404 Not Found` with body
  `{"error": "no_such_session"}`
<!-- test: larql_server::test_http_attention::prefill_unknown_session_returns_404 -->

#### Scenario: binary form round-trips

- **WHEN** a client prefills with `Accept: application/octet-stream`
  and the same body decoded as JSON would have produced residual `R`
- **THEN** the binary payload SHALL equal `bytemuck::cast_slice(R)`
  byte-for-byte
<!-- test: unbacked -->

### Requirement: Decode SHALL advance the cache by one token and return the residual

`POST /v1/attention/decode` SHALL append one query position to the
session's KV cache, run masked-attention against the cumulative cache,
and return the post-attention residual for every layer. The endpoint
SHALL accept JSON (default) or binary
(`application/octet-stream`) bodies; binary form SHALL be
`[u32 layers][u32 hidden_dim][f32 layers × hidden_dim]`.

#### Scenario: decode advances seqlen

- **WHEN** a client prefills 8 tokens, then issues 3 decode calls
- **THEN** `GET /v1/attention/session/{id}` SHALL report
  `current_seq_len = 11`
<!-- test: larql_server::test_attention_validation::session_seq_len_advances_after_prefill -->

#### Scenario: decode residual matches local reference within tolerance

- **WHEN** a client prefills and decodes against the server, and a
  local CPU pipeline runs the same weights and inputs
- **THEN** every per-layer residual SHALL agree to cosine ≥ 0.99 and
  max-element relative error ≤ 1e-3
<!-- test: larql_server::test_attention_validation::prefill_server_residuals_match_local_reference_layer_by_layer -->

#### Scenario: decode before prefill is rejected

- **WHEN** a client creates a session, then immediately calls
  `/v1/attention/decode` with no prior prefill
- **THEN** the server SHALL return `400 Bad Request` with body
  `{"error": "decode_before_prefill"}`
<!-- test: larql_server::test_http_attention::decode_before_prefill_returns_400 -->

### Requirement: KV-cache snapshot SHALL be a versioned binary blob

`POST /v1/kv-cache/snapshot` SHALL return a binary blob that
captures the full cache state. The blob SHALL begin with the magic
`0x4C415141` (`'LAQA'` little-endian) followed by a 16-bit version
field. The version on this change SHALL be `1`. Servers SHALL reject
restore calls with an unknown version (returning 400 with body
`{"error": "snapshot_version_unsupported", "supported_versions":
[...]}`).

The blob SHALL include per-layer offsets, dimensions
(num_layers, hidden_dim, num_heads, head_dim, seq_len), and per-layer
payloads tagged as empty / fp32 / quantized.

#### Scenario: snapshot magic + version are present

- **WHEN** a client calls `/v1/kv-cache/snapshot` on a populated session
- **THEN** the first 4 bytes SHALL be `0x41 0x51 0x41 0x4C` (LE 'LAQA')
  and bytes 4–5 SHALL be `0x01 0x00`
<!-- test: larql_server::kv_snapshot::tests::magic_and_version_are_present_at_known_offsets -->
<!-- test: larql_server::test_http_attention::snapshot_returns_base64_with_correct_magic -->

#### Scenario: snapshot then restore round-trips for FP32

- **WHEN** a client snapshots an FP32 session, deletes it, creates a
  new session with the snapshot
- **THEN** every per-layer K and V buffer SHALL be byte-identical to
  the original, and decode against the new session SHALL produce the
  same residual as decode against the original would have
<!-- test: larql_server::kv_snapshot::tests::round_trip_fp32_is_byte_identical -->
<!-- test: larql_server::test_http_attention::restore_round_trips_through_a_new_session -->
<!-- test: unbacked -->

#### Scenario: snapshot then restore round-trips for compressed formats

- **WHEN** a client snapshots an `iso3` session, deletes it, restores
- **THEN** the dequantised K and V SHALL match the original to cosine
  ≥ 0.95 (the format's published bound; round-trip dequant is the
  same as fresh dequant)
<!-- test: larql_server::kv_snapshot::tests::round_trip_compressed_layer -->
<!-- test: unbacked -->

#### Scenario: unknown version is rejected

- **WHEN** a client POSTs `/v1/kv-cache/restore` with a snapshot whose
  version field is `0xFFFF`
- **THEN** the server SHALL return `400 Bad Request` with body
  `{"error": "snapshot_version_unsupported"}`
<!-- test: larql_server::kv_snapshot::tests::unknown_version_is_rejected -->
<!-- test: unbacked -->

### Requirement: Server SHALL announce its `--role` as a router capability

A server started with `--role attention` SHALL announce
`capabilities = ["attention"]` to the router; `--role expert` SHALL
announce `["expert"]`; `--role both` (default) SHALL announce
`["attention", "expert"]`.

#### Scenario: attention role announces correctly

- **WHEN** a server is started with `--role attention --router <url>`
- **THEN** the announce payload received by the router SHALL contain
  exactly `["attention"]` in `capabilities`
<!-- test: larql_server::bootstrap::tests::role_attention_announces_attention_only -->
<!-- test: larql_server::bootstrap::tests::cli_role_flag_parses_attention -->

#### Scenario: legacy server (no --role) announces both

- **WHEN** a server is started with no `--role` flag
- **THEN** the announce payload SHALL contain
  `["attention", "expert"]`
<!-- test: larql_server::bootstrap::tests::role_both_announces_attention_and_expert -->
<!-- test: larql_server::bootstrap::tests::cli_defaults_role_to_both -->

### Requirement: Heartbeat SHALL include the cached-prefix bloom

A server SHALL include a 32-byte raw `PrefixBloom` in its heartbeat
under the `cached_prefixes` field. The server SHALL rebuild this
bloom on each heartbeat tick by hashing the first 16 token-ids of
each currently-active session and inserting them into a fresh bloom.

#### Scenario: heartbeat includes the cached prefixes

- **WHEN** a server has 5 active sessions and emits a heartbeat
- **THEN** the heartbeat payload SHALL contain a `cached_prefixes`
  field of exactly 32 bytes
<!-- test: unbacked -->

#### Scenario: empty server emits empty bloom

- **WHEN** a server with no active sessions emits a heartbeat
- **THEN** the `cached_prefixes` field SHALL be 32 zero bytes
<!-- test: unbacked -->
