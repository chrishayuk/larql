## ADDED Requirements

### Requirement: Attention session lifecycle endpoints

`larql-server` running with `--role attention` SHALL expose session
lifecycle endpoints:

- `POST /v1/attention/session` — body declares `model`, `kv_format`,
  `max_seq_len`. Returns `{ session_id, kv_handle }`.
- `DELETE /v1/attention/session/{session_id}` — frees session VRAM.
- `GET /v1/attention/session/{session_id}` — returns session state
  (current length, KV format, byte footprint).

The KV format is fixed at session creation and SHALL NOT change for
the lifetime of the session.

#### Scenario: Session create returns an opaque handle
- **WHEN** a client POSTs `{"model":"gemma-3-4b","kv_format":"iso3","max_seq_len":8192}` to `/v1/attention/session`
- **THEN** the response body SHALL contain a `session_id` (UUID) and a `kv_handle` (opaque u128)
<!-- test: unbacked -->

#### Scenario: Session DELETE frees VRAM
- **WHEN** a session is deleted
- **THEN** subsequent attention RPCs against its `kv_handle` SHALL return HTTP 410 Gone
<!-- test: unbacked -->

### Requirement: Attention prefill endpoint

`POST /v1/attention/prefill` SHALL accept a session id, a sequence
of token embeddings (FP16 binary body), and run prefill across all
attention layers, populating the session's KV cache. The response
body SHALL contain the post-attention residual stream for each layer
(or only the final layer, gated by a `return_intermediate=true` query
parameter).

#### Scenario: Prefill of 1024 tokens populates the KV cache
- **WHEN** a prefill request with 1024 token embeddings is submitted to a Gemma 4B session
- **THEN** the response SHALL be 200 OK and the session's reported KV length SHALL be 1024
<!-- test: unbacked -->

### Requirement: Attention decode endpoint

`POST /v1/attention/decode` SHALL accept a session id and exactly one
new token embedding. The response SHALL be the post-attention residual
for the new token only.

#### Scenario: Decode appends one position to the KV cache
- **WHEN** a decode request is submitted on a session with current length L
- **THEN** the response SHALL be 200 OK, the session length SHALL become L+1, and the response body SHALL be exactly one residual vector
<!-- test: unbacked -->

### Requirement: KV-cache snapshot and restore

The service SHALL support snapshotting an in-VRAM KV cache to a
client-storable blob and restoring it later, enabling session
checkpoint / resume across container restarts.

- `POST /v1/kv-cache/snapshot` — body `{ session_id }`; returns binary
  blob with header `{ format, layer_count, byte_count }`.
- `POST /v1/kv-cache/restore` — body is the blob from a previous
  snapshot; returns `{ session_id, kv_handle }` for a fresh session
  rehydrated to the snapshot's state.

The snapshot blob format SHALL be:
```
header_json (UTF-8, LEB128-prefixed length)
[ per-layer: K_bytes V_bytes norms_bytes rotation_indices_bytes ]
```

#### Scenario: Snapshot then restore round-trips bit-exact
- **WHEN** a session is snapshotted, the server is restarted, and the snapshot is restored on a fresh session
- **THEN** the resumed session's first decode response SHALL match the original session's pre-snapshot decode response within 1e-4 absolute element difference
<!-- test: unbacked -->

#### Scenario: Snapshot blob carries the KV format
- **WHEN** an iso3 session is snapshotted
- **THEN** the blob header SHALL contain `"format":"iso3"` and the byte count SHALL match the iso3 layout
<!-- test: unbacked -->

### Requirement: gRPC parity for the HTTP surface

Every HTTP endpoint above SHALL have a gRPC counterpart in the
`larql-router-protocol` proto definitions, with matching semantics.
gRPC streaming MAY be used for prefill of large sequences to avoid a
single oversized request body.

#### Scenario: gRPC service handles the same prefill
- **WHEN** the same prefill is submitted via gRPC and HTTP
- **THEN** the responses SHALL be byte-equivalent
<!-- test: unbacked -->

### Requirement: Topology advertises the attention capability

When started with `--role attention`, `larql-server` SHALL announce
itself to the router with `capabilities: ["attention"]` (and not
`"expert"`, since the FFN bank is not loaded). The router SHALL
direct attention RPCs only to shards that advertise the capability.

#### Scenario: GPU shard refuses expert RPCs
- **WHEN** an `/v1/expert/batch` request is routed to an attention-only shard
- **THEN** the shard SHALL respond with HTTP 503 and a clear "no expert weights loaded" body
<!-- test: unbacked -->
