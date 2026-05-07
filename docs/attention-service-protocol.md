# Attention service — HTTP/JSON + gRPC protocol

> _Implementation tracking: [`openspec/changes/attention-service-routes/`](../openspec/changes/attention-service-routes/)._

The attention service exposes the GPU-resident KV cache to clients
(other shards, routers, end users) over HTTP/JSON. A gRPC
`AttentionService` ships alongside; semantics are identical, the
shapes mirror the JSON.

All routes are gated by the same `--api-key` header that covers the
rest of the server (when the flag is set). The endpoints live on the
**single-model** router; multi-model routing is handled by the
client — the session itself is bound to one `model_id`.

## Lifecycle

```
                      ┌───────────────┐
client    ──── POST  /v1/attention/session    ───→ {session_id}
                      │   (cache allocated)        │
client    ──── POST  /v1/attention/prefill        │   N times (rare to repeat)
                      │   (cache populated)        │
client    ──── POST  /v1/attention/decode    × M  │
                      │   (one new token)          │
client    ──── POST  /v1/kv-cache/snapshot        │   any time, returns blob
client    ──── POST  /v1/kv-cache/restore         │   replace cache from blob
client    ──── POST  /v1/kv-cache/free            │   free one or all layers
client    ──── DELETE /v1/attention/session/{id}  ┘   drop, free VRAM
```

Sessions reap automatically after the configured idle TTL
(`--attention-session-ttl-secs`, default 600 s). Reaper wakes every
30 s, so reclamation can lag by up to that interval.

## Routes

### `POST /v1/attention/session`

Create a session.

**Request**

```jsonc
{
  "model_id": "gemma-3-4b",            // required
  "kv_format": "iso3",                  // optional; default "fp32"
  "restore_from_snapshot": null         // optional bytes (b64); see /restore
}
```

`kv_format` ∈ `{"fp32", "planar3", "planar4", "iso3", "iso4"}`. The
`fp32` value disables RotorQuant compression (per-layer cache stays
in FP32; `quantize_layer` is a no-op).

When `restore_from_snapshot` is set, the snapshot's header drives
the format / sequence length / layer count. The `kv_format` field is
ignored in that case.

**Response (201)**

```jsonc
{
  "session_id":     "01HM7E2CK7Q9N6...",   // 26-char ULID
  "layer_range":    [0, 33],                // [inclusive, exclusive)
  "kv_format":      "iso3"
}
```

**Errors**

- `404 no_such_model` — `model_id` not loaded on this server.
- `400 kv_format_unknown` — unrecognised format string.
- `400 snapshot_*` — see `/restore` errors.
- `503 session_map_full` — beyond `--max-attention-sessions`.

### `GET /v1/attention/session/{id}`

Read session state.

**Response (200)**

```jsonc
{
  "session_id":   "01HM7E...",
  "model_id":     "gemma-3-4b",
  "kv_format":    "iso3",
  "seq_len":      0,
  "prefilled":    false,
  "num_layers":   33
}
```

**Errors**

- `404 no_such_session`.

### `DELETE /v1/attention/session/{id}`

Drop the session, free the KV cache memory.

**Response**

- `204 No Content` on success.
- `404 no_such_session` otherwise.

### `POST /v1/attention/prefill`

Stateful: populates the named session's cache with the prefill K/V
projections and returns per-layer post-attention residuals.

**Request**

```jsonc
{
  "session_id":       "01HM7E...",
  "token_embeddings": [               // [seq_len][hidden_dim] f32
    [0.123, -0.456, ...],
    ...
  ]
}
```

For binary clients: `Content-Type: application/octet-stream` plus
the layout `[u32 seq_len][u32 hidden_dim][f32 seq_len × hidden_dim]`.

**Response (200, JSON)**

```jsonc
{
  "post_attention_residuals": [...],  // [layers][seq_len][hidden_dim] f32
  "kv_filled_through_layer":  33,
  "tokens_processed":         17,
  "latency_ms":               42.3
}
```

For binary: `Accept: application/octet-stream`.
Layout: `[u32 layers][u32 seq_len][u32 hidden_dim][f32 …]`.

> Status: **not yet implemented in this branch** — the route slot
> exists; the prefill engine wires in alongside the model's
> attention-block runner.

### `POST /v1/attention/decode`

Stateful: appends one query position, runs masked attention against
the cumulative cache, returns per-layer residuals.

Wire shape: same JSON / binary forms as prefill, but with
`query_token_embedding: [hidden_dim] f32` (singular).

> Status: **not yet implemented in this branch** — same lift as
> prefill.

### `POST /v1/kv-cache/snapshot`

Returns the session's cache as a versioned binary blob.

**Request**: `{"session_id": "..."}`.
**Response**:

```jsonc
{
  "session_id": "...",
  "snapshot":   "AQEBAQAAAAA…",   // base64 of the binary blob
  "bytes_len":  328704
}
```

The blob format is documented below. Snapshots are **not stable
across versions**; the version byte is the only forward contract.

### `POST /v1/kv-cache/restore`

Replaces the session's cache with the contents of a snapshot.

**Request**

```jsonc
{
  "session_id": "...",
  "snapshot":   "AQEBAQAAAAA…"
}
```

**Response**

```jsonc
{ "session_id": "...", "seq_len": 17, "num_layers": 33 }
```

**Errors**

- `404 no_such_session`.
- `400 snapshot_base64_decode_failed`.
- `400 snapshot_magic_mismatch`.
- `400 snapshot_version_unsupported` — `{ "supported_versions": [1] }` in detail.
- `400 snapshot_invalid` — generic deserialise failure.

### `POST /v1/kv-cache/free`

Free one or all layers of the cache without deleting the session.

**Request**: `{"session_id": "...", "layer": 12}` — or `"layer": null`
to free everything.

**Response**: `{ "session_id": "...", "layers_freed": 1 }`.

**Errors**

- `404 no_such_session`.
- `400 layer_out_of_range`.

## Snapshot wire format (v1)

```text
0x00  u32 le   magic = 0x4C415141 ('LAQA')
0x04  u16 le   version = 1
0x06  u16 le   flags
                  bit 0  any layer is RotorQuant-compressed
                  bit 1  any layer is FP32-populated
                  bit 2..15 reserved
0x08  u32 le   num_layers
0x0C  u32 le   max_window     (0 ⇒ unbounded)
0x10  u32 le   next_position
0x14  u32 le   kv_format       (0 ⇒ none, 1=Planar3, 2=Planar4, 3=Iso3, 4=Iso4)
0x18  u32 le   promote_on_read_count_lo
0x1C  u32 le   promote_on_read_count_hi
0x20         per-layer offsets: [u64 le; num_layers]
       payload, packed in layer order:
         u8 tag   (0 = empty, 1 = fp32, 2 = quantized)
         if fp32:
           u32 le  k_rows
           u32 le  k_cols
           [f32 le; k_rows × k_cols]
           u32 le  v_rows
           u32 le  v_cols
           [f32 le; v_rows × v_cols]
         if quantized:
           u8      kv_format (1..=4)
           — K block —
           u32 le  n_rows
           u32 le  head_dim
           u32 le  codes_len
           [u8;    codes_len]
           u32 le  norms_len  (== n_rows)
           [f32;   norms_len]
           u32 le  rot_len    (== n_rows × n_blocks_per_row)
           [u16;   rot_len]
           — V block (same shape) —
```

The reference encoder/decoder lives in
[`crates/larql-server/src/kv_snapshot.rs`](../crates/larql-server/src/kv_snapshot.rs).
Round-trip tests cover FP32 (byte-identical) and Iso4
(field-by-field equal). Truncated / wrong-magic / wrong-version
inputs are rejected with typed errors, not panics.

## Routing

The router (`larql-router`) consults each shard's
`HeartbeatMsg.cached_prefixes` (256-bit `PrefixBloom`,
[`crates/larql-router-protocol/src/lib.rs`](../crates/larql-router-protocol/src/lib.rs))
to route follow-up calls for a session toward the shard whose KV
cache already has the request's prefix warm. The shard rebuilds its
bloom on every heartbeat from `SessionMap::prefix_hashes(16)` —
sixteen-token prefix per active session.

A heartbeat without a `cached_prefixes` field leaves the prior
bloom intact (legacy shards never populate it; the field is
backwards-compat).

## Examples

### Python (httpx, JSON)

```python
import base64, httpx

c = httpx.Client(base_url="http://attention:8080")
sess = c.post("/v1/attention/session", json={
    "model_id": "gemma-3-4b",
    "kv_format": "iso3",
}).json()
sid = sess["session_id"]

# (prefill / decode go here once the runner ships)

snap = c.post("/v1/kv-cache/snapshot", json={"session_id": sid}).json()
blob = base64.b64decode(snap["snapshot"])

# Restore on a different shard:
c2 = httpx.Client(base_url="http://attention-2:8080")
sess2 = c2.post("/v1/attention/session", json={
    "model_id": "gemma-3-4b",
    "restore_from_snapshot": snap["snapshot"],
}).json()

c.delete(f"/v1/attention/session/{sid}")
```

### `curl`

```bash
curl -sX POST http://localhost:8081/v1/attention/session \
     -H 'content-type: application/json' \
     -d '{"model_id":"gemma-3-4b","kv_format":"iso3"}'
```

## Versioning

The HTTP path itself (`/v1/...`) is the version stamp on the
endpoints. The KV snapshot format has its own 16-bit version field
inside the blob; bumps require a fresh path or an explicit version
parameter on `POST /v1/kv-cache/restore`.
