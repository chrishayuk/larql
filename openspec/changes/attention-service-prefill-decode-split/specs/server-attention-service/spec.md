## ADDED Requirements

### Requirement: server SHALL support a --mode flag for PD-split

`larql-server` SHALL accept `--mode prefill | decode | both`
(default `both`). The flag interacts with `--role attention` (the
existing role): an `attention` shard's mode further constrains
which attention RPCs it serves.

#### Scenario: --mode prefill rejects decode RPCs
- **WHEN** an `/v1/attention/decode` RPC is sent to a `--mode prefill` shard
- **THEN** the response SHALL be HTTP 503 with a JSON body containing `{"role":"prefill","missing":"decode"}`
<!-- test: unbacked -->

#### Scenario: --mode decode rejects prefill RPCs
- **WHEN** an `/v1/attention/prefill` RPC is sent to a `--mode decode` shard
- **THEN** the response SHALL be HTTP 503 with a JSON body containing `{"role":"decode","missing":"prefill"}`
<!-- test: unbacked -->

#### Scenario: --mode both accepts both
- **WHEN** an attention shard runs without `--mode` (default `both`) and receives both prefill and decode RPCs
- **THEN** both SHALL succeed
<!-- test: unbacked -->

### Requirement: prefill MUST return a KV-snapshot blob

`POST /v1/attention/prefill` SHALL be **stateless** — no session
created server-side, no VRAM retained after response. The response
SHALL include both the per-layer post-attention residuals AND a
KV-snapshot blob in the same format as
`/v1/kv-cache/snapshot`. Clients MAY pass that blob to a
subsequent `POST /v1/attention/session`.

#### Scenario: prefill response includes snapshot
- **WHEN** a 1024-token prefill RPC is sent
- **THEN** the response SHALL contain `residuals: …` plus `kv_snapshot: <base64>` whose header parses as a valid snapshot blob
<!-- test: unbacked -->

#### Scenario: prefill is stateless
- **WHEN** the same client immediately follows a prefill with a decode RPC against the same model
- **THEN** the decode SHALL fail with HTTP 404 (no session) — the prefill did not implicitly create one
<!-- test: unbacked -->

### Requirement: session create MUST accept an optional restore_from_snapshot

`POST /v1/attention/session` body MAY include `restore_from_snapshot: <base64 blob>`. When present, the server SHALL restore the session's KV cache from the snapshot before returning the session id.

#### Scenario: session boots pre-loaded from snapshot
- **WHEN** a session create RPC carries a snapshot from a 1024-token prefill
- **THEN** the resulting session's `current_length` SHALL be 1024 (not 0)
<!-- test: unbacked -->

### Requirement: router MUST recognise attention-prefill and attention-decode sub-capabilities

A shard's announce payload SHALL allow `capabilities` entries
`"attention-prefill"` and `"attention-decode"` (in addition to the
catch-all `"attention"`). The router's `route_for_capability(_, _,
"attention-prefill")` SHALL prefer shards advertising the specific
sub-tag, falling back to the catch-all `"attention"` when no
specific match exists.

#### Scenario: prefill RPC routes to attention-prefill shard
- **WHEN** a shard advertising `["attention-prefill"]` and a shard advertising `["attention-decode"]` both cover layer 0
- **THEN** `route_for_capability(_, 0, "attention-prefill")` SHALL pick the first shard
<!-- test: unbacked -->

#### Scenario: catch-all attention shard receives both
- **WHEN** only a shard advertising `["attention"]` is present
- **THEN** both `attention-prefill` and `attention-decode` lookups SHALL return that shard
<!-- test: unbacked -->
