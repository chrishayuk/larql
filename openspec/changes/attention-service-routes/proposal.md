## Why

The CUDA + RotorQuant workstream has shipped its kernel surface
(`cuda-f32-baseline`, `cuda-q4-matvec`, `cuda-fused-attention`) and
its KV-cache compression
(`rotorquant-attention-integration`, `rotorquant-promote-on-read`).
The router has been taught to filter by capability
(`router-heterogeneous-shards`) and to prefer prefix-cached shards
(`router-prefix-aware-routing`).

What's still missing is the **service surface** that exposes those
GPU primitives over the wire. The two-container topology in
`deploy/docker/` boots an attention container that has nothing to
serve, and the router's `cached_prefixes` bloom has no producer
because heartbeats don't carry the field yet. Without these
endpoints, a client can't actually drive Gemma 4B end-to-end
through the topology — every other piece is in place.

This change introduces the `server-attention-service` capability:
session lifecycle, prefill, decode, and KV-cache snapshot/restore,
on both HTTP (axum) and gRPC (tonic). It also extends the
`router-grid` and `router-protocol` capabilities with the
heartbeat fields that drive the existing `route_for_capability`
and `route_for_prefix` selectors.

## What Changes

### `server-attention-service` (NEW capability)

- `POST /v1/attention/session` — create a session. Body:
  `{model_id, kv_format?: "fp32"|"planar3"|"planar4"|"iso3"|"iso4",
  restore_from_snapshot?: bytes_b64}`. Response: `{session_id, layer_range,
  kv_format}`. Server allocates a `KvCache` honouring the format.
- `GET /v1/attention/session/{id}` — return current cache state
  (per-layer compressed flag, current sequence length, format).
- `DELETE /v1/attention/session/{id}` — drop the cache; free VRAM.
- `POST /v1/attention/prefill` — body:
  `{session_id, token_embeddings: [seq_len][hidden_dim] f32}`.
  Runs the attention block for every layer, populates the cache,
  returns per-layer post-attention residuals. Stateful (mutates
  the named session).
- `POST /v1/attention/decode` — body:
  `{session_id, query_token_embedding: [hidden_dim] f32}`. Runs
  one decode step using the session's KV cache. Returns the
  post-attention residual.
- `POST /v1/kv-cache/snapshot` — body: `{session_id}`. Returns
  binary KV blob (versioned wire format). Used by
  `attention-service-prefill-decode-split` for the prefill→decode
  handoff.
- `POST /v1/kv-cache/restore` — body: `{session_id, snapshot: bytes}`.
  Replaces the session's cache with the snapshot's contents.
- `POST /v1/kv-cache/free` — body: `{session_id, layer?: u32}`.
  Frees one layer (or all) of compressed KV.

### `router-protocol` (MODIFIED)

- `AnnounceMsg.capabilities: repeated string` (proto field, was a
  client-side default until this change). Pre-announce shards
  default to `["attention", "expert"]` for backwards compat.
- New `HeartbeatMsg.cached_prefixes: bytes` (32-byte raw
  PrefixBloom payload — 256 bits / 4 hash positions). Optional;
  pre-extension shards send empty.

### `router-grid` (MODIFIED)

- `GridState::register` now reads `capabilities` and
  `cached_prefixes` from the announce/heartbeat payload; field
  defaults preserved for legacy shards.
- `GridState::update_heartbeat` updates `cached_prefixes`
  alongside CPU/RAM metrics.

## Capabilities

### New Capabilities

- `server-attention-service` — HTTP + gRPC routes for session
  lifecycle, prefill, decode, and KV snapshot/restore.

### Modified Capabilities

- `router-protocol` — extends `AnnounceMsg` with
  `capabilities`; adds `HeartbeatMsg.cached_prefixes`.
- `router-grid` — registers and updates the new fields from the
  proto extension.

## Impact

- **Affected files**:
  - `crates/larql-server/src/routes/attention/mod.rs` (new module)
  - `crates/larql-server/src/routes/attention/session.rs` (new)
  - `crates/larql-server/src/routes/attention/prefill.rs` (new)
  - `crates/larql-server/src/routes/attention/decode.rs` (new)
  - `crates/larql-server/src/routes/kv_cache.rs` (new)
  - `crates/larql-server/src/grpc.rs` (extends with
    `AttentionService` impl)
  - `crates/larql-router-protocol/proto/router.proto` (extends
    AnnounceMsg + HeartbeatMsg)
  - `crates/larql-router/src/grid.rs` (consumes the new fields)
  - `deploy/docker/start.sh` (passes `--role attention` /
    `--role expert` based on container)
- **Affected systems**: server, router. No GPU code changes (the
  CUDA backend is already complete).
- **Provenance**: scope is the parent change
  `cuda-and-rotorquant-kv` Phase 4 (tasks 8.1–8.6); this change
  formalises it into a standalone proposal so the work can land
  in reviewable bites.
- **Out of scope**:
  - The PD-disaggregation knobs (`--mode prefill|decode|both`) —
    those land in `attention-service-prefill-decode-split` after
    this change ships.
  - Multi-tenant authentication on the session endpoints — the
    existing `--api-key` header gate is sufficient for the
    single-tenant deployment.
  - Backpressure / rate limiting — that's a deployment concern,
    not part of the protocol contract.
