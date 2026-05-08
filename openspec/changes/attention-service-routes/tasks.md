# attention-service-routes — tasks

## 1. Server-side scaffolding

- [ ] 1.1 New module `larql_server::routes::attention` with sub-modules
  `session`, `prefill`, `decode`.
- [ ] 1.2 Define `SessionId` (ULID, 128-bit) and `Session` struct
  (`KvCache` + `last_used` + `model_id`).
- [ ] 1.3 `SessionMap` (`DashMap<SessionId, Arc<RwLock<Session>>>`)
  on `AppState`. Default cap: 256 concurrent sessions.
- [ ] 1.4 TTL reaper as a background tokio task launched from
  `bootstrap::serve`; configurable via `--session-ttl-secs`
  (default 600).

## 2. HTTP route handlers

- [ ] 2.1 `POST /v1/attention/session` — create + optional restore.
- [ ] 2.2 `GET /v1/attention/session/{id}` — read state.
- [ ] 2.3 `DELETE /v1/attention/session/{id}` — drop.
- [ ] 2.4 `POST /v1/attention/prefill` — JSON + binary form.
- [ ] 2.5 `POST /v1/attention/decode` — JSON + binary form.
- [ ] 2.6 `POST /v1/kv-cache/snapshot` — returns binary blob.
- [ ] 2.7 `POST /v1/kv-cache/restore` — accepts binary blob.
- [ ] 2.8 `POST /v1/kv-cache/free` — frees a layer or all.

## 3. KV-snapshot wire format

- [ ] 3.1 New crate-private module `larql_server::kv_snapshot`.
  Types: `SnapshotHeader`, `LayerKind`, `SnapshotError`.
- [ ] 3.2 `serialize(cache: &KvCache) -> Vec<u8>` — magic + header
  + per-layer offsets + payload. Reuses `bytemuck::cast_slice`
  for f32 ranges; payload-side encoding for compressed layers
  reuses `larql_rotorquant::format::QuantizedKv` byte layout.
- [ ] 3.3 `deserialize(bytes: &[u8]) -> Result<KvCache, SnapshotError>` —
  rejects unknown magic / version / inconsistent dimensions.
- [ ] 3.4 Round-trip tests for FP32 and each RotorQuant format.
- [ ] 3.5 Document the wire format in
  `docs/attention-service-protocol.md`.

## 4. gRPC parity

- [ ] 4.1 Extend `larql-router-protocol/proto/router.proto`:
  - Service `AttentionService` with rpcs:
    `CreateSession`, `GetSession`, `DeleteSession`,
    `Prefill`, `Decode`, `Snapshot`, `Restore`, `Free`.
  - Use server-streaming for `Prefill` (one event per layer).
- [ ] 4.2 `larql_server::grpc::AttentionServiceImpl` thin shim that
  forwards to the same handlers as HTTP.
- [ ] 4.3 Server announces `AttentionService` in its server
  reflection set so the router CLI introspector can dispatch.

## 5. Router-protocol extensions

- [ ] 5.1 `proto/router.proto`:
  - `AnnounceMsg` gains `repeated string capabilities = N;`
  - `HeartbeatMsg` gains `bytes cached_prefixes = N;` (32 bytes,
    raw `PrefixBloom`).
- [ ] 5.2 Bump proto version constant; document the wire change.
- [ ] 5.3 Backwards compat: missing `capabilities` field ⇒ default
  `["attention", "expert"]`; missing `cached_prefixes` ⇒ empty
  bloom (already the documented behaviour from
  `router-prefix-aware-routing`).

## 6. Router-grid wiring

- [ ] 6.1 `GridState::register` reads `entry.capabilities` /
  `entry.cached_prefixes` (already on the struct from earlier
  changes); plumb these from the proto-decoded request.
- [ ] 6.2 `GridState::update_heartbeat` accepts a `cached_prefixes:
  PrefixBloom` parameter and writes it onto the entry.
- [ ] 6.3 Tests:
  - `heartbeat_updates_cached_prefixes`
  - `legacy_announce_gets_default_capabilities`

## 7. Server-side bloom rebuild

- [ ] 7.1 `larql_server::session::SessionMap::current_prefix_bloom()`
  iterates active sessions, hashes their first-N-token prefix, and
  builds a `PrefixBloom`. Configurable N (default 16 tokens).
- [ ] 7.2 Heartbeat task includes the bloom in its `HeartbeatMsg`.
  Rebuild cadence aligned with the heartbeat cadence (default
  60 s).

## 8. Topology + role wiring

- [ ] 8.1 New CLI flag: `--role attention | expert | both`
  (default `both`). When `attention`-only, the bootstrap skips
  loading FFN weights into RAM. When `expert`-only, the bootstrap
  skips attention weight allocation.
- [ ] 8.2 `--role` controls which capabilities the server announces
  to the router (`attention` ⇒ `["attention"]`; `expert` ⇒
  `["expert"]`; `both` ⇒ both).
- [ ] 8.3 Update `deploy/docker/start.sh` and
  `deploy/docker/docker-compose.yml` to pass `--role attention`
  to the GPU container and `--role expert` to the CPU FFN
  container.
- [ ] 8.4 Update `deploy/docker/README.md` topology diagram with
  the role split.

## 9. Integration tests

- [ ] 9.1 `tests/test_attention_endpoint.rs` — spins up a real
  attention server, creates a session, runs prefill + decode,
  validates against a local CPU reference (cosine ≥ 0.99 on
  small synthetic Gemma-shaped weights).
- [ ] 9.2 Snapshot/restore round-trip: create A, prefill, snapshot,
  delete A, create B with restore, decode in B, compare to
  decode in A. Must be bit-equal in FP32 mode.
- [ ] 9.3 Router-aware routing: register two attention shards;
  populate one's bloom with a prefix hash; verify router picks
  it for matching prefix and falls back when no match.

## 10. Documentation

- [ ] 10.1 `docs/attention-service-protocol.md` — full spec for the
  HTTP + gRPC surface. Code examples in Python (`requests`) and
  curl.
- [ ] 10.2 Update `docs/cuda-rotorquant-status.md` to mark the
  capability shipped and refresh the topology diagram.
- [ ] 10.3 Update `openspec/specs/router-grid/spec.md` (after
  archive) so the canonical capability reflects the heartbeat
  fields.
