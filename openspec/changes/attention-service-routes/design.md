# attention-service-routes — design

## Session model

A session is a server-side handle for a `KvCache`. The cache itself
already supports compressed (RotorQuant) and FP32 layers via the
side-table that landed in `rotorquant-attention-integration`; this
change adds the lifecycle and per-call routing.

```text
            ┌──────────────────────────────────────────┐
client ───→ │  POST /v1/attention/session              │
            │    → SessionMap.insert(id, KvCache)      │
            │  POST /v1/attention/prefill              │
            │    → cache.set_layer(...) per layer       │
            │  POST /v1/attention/decode (× n)         │
            │    → cache.get_layer(...) per layer       │
            │  POST /v1/kv-cache/snapshot              │
            │  POST /v1/kv-cache/restore               │
            │  DELETE /v1/attention/session/{id}       │
            └──────────────────────────────────────────┘
```

`SessionMap` is a `dashmap::DashMap<SessionId, Arc<RwLock<Session>>>`:

- `SessionId` is a 128-bit ULID (lexicographically sortable; debug-friendly).
- The outer map is sharded; the inner `RwLock` lets prefill writes
  serialise per-session while decodes from different sessions run
  concurrently.
- Each `Session` carries `KvCache` + `last_used: Instant` for TTL eviction
  + `model_id` for routing sanity.

### TTL

Sessions older than `--session-ttl-secs` (default 600) are reaped on a
background tokio task that wakes every 30 s. Reaping snapshots the
session, encrypts it with the configured at-rest key (deferred), and
deletes the in-memory entry.

The reaper's wake-up cadence is decoupled from TTL so a 30 s slack on
reclamation is acceptable; a session that's seconds-from-eviction is
extended on any next request.

## Wire formats

### `POST /v1/attention/session`

```jsonc
// request
{
  "model_id": "gemma-3-4b",
  "kv_format": "iso3",        // optional; default "fp32"
  "restore_from_snapshot": null   // optional bytes (b64); default null
}

// response
{
  "session_id": "01HM7E2CK7Q9...",
  "layer_range": [0, 33],
  "kv_format": "iso3",
  "compression_ratio": 0.083
}
```

`compression_ratio` is reported but informational — the actual ratio
emerges from the cache contents and is calculated as `quantized_bytes
/ fp32_bytes`. For an empty cache it equals the format's static ratio.

### `POST /v1/attention/prefill`

```jsonc
// request
{
  "session_id": "01HM...",
  "token_embeddings": [          // [seq_len][hidden_dim] f32
    [0.123, -0.456, ...],
    ...
  ]
}

// response
{
  "post_attention_residuals": [  // [layers][seq_len][hidden_dim] f32
    ...
  ],
  "kv_filled_through_layer": 33,
  "tokens_processed": 17,
  "latency_ms": 42.3
}
```

The response body is large by JSON standards (~33 × 17 × 4096 × 4 B =
~9 MB on Gemma 4B). The route auto-negotiates to a binary content type
when the client sends `Accept: application/octet-stream`:

```text
[u32 le: layers] [u32 le: seq_len] [u32 le: hidden_dim] [f32 le: layers × seq_len × hidden_dim]
```

This mirrors the `MultiLayerBatch` binary format already in use by the
expert endpoint; reusing the layout lets `bytemuck::cast_slice` the
whole tail without per-layer parsing.

### `POST /v1/attention/decode`

```jsonc
// request
{
  "session_id": "01HM...",
  "query_token_embedding": [...]    // [hidden_dim] f32
}

// response
{
  "post_attention_residual": [...], // [layers][hidden_dim] f32
  "latency_ms": 3.1
}
```

Binary form: `[u32 le: layers] [u32 le: hidden_dim] [f32 le: layers × hidden_dim]`.

### KV snapshot wire format (binary, versioned)

```text
0x00  u32 le  magic = 0x4C415141  ('LAQA' = LARQL Attention SnapshoT, capital A)
0x04  u16 le  version = 1
0x06  u16 le  flags
                bit 0 = compressed (RotorQuant)
                bit 1 = quantized_kv has at least one Some()
                bit 2 = fp32 has at least one Some()
0x08  u32 le  num_layers
0x0C  u32 le  hidden_dim
0x10  u32 le  num_heads
0x14  u32 le  head_dim
0x18  u32 le  seq_len
0x1C  u32 le  reserved
0x20  per_layer_offsets: [u64 le; num_layers]
       each offset points into the same blob
0x20 + 8 * num_layers ...
       per-layer payload:
         flag: u8 (0 = empty, 1 = fp32, 2 = quantized)
         if fp32:
           [f32 le; 2 × num_heads × seq_len × head_dim]   // K then V
         if quantized:
           kv_format: u8  (1 = planar3, 2 = planar4, 3 = iso3, 4 = iso4)
           K block: [norms: f32 × n_rows] [rot_idx: u16 × n_blocks] [codes: u8 × bits/8 × n_codes]
           V block: same shape as K
```

Snapshot/restore is the foundation that
`attention-service-prefill-decode-split` builds on. The wire format
is **not** stable across versions; `version` is the only forward
contract and clients must reject unknown versions.

## Routing topology

- An attention shard advertises `capabilities = ["attention"]` in its
  announce. The router's `route_for_capability(model_id, layer,
  "attention")` already filters correctly; this change just gives it
  real data.
- A shard heartbeat carries its current `cached_prefixes` bloom; the
  router updates `ServerEntry::cached_prefixes` on every heartbeat.
- The shard maintains its bloom by inserting prefix hashes when a
  session's prefill completes, and **rebuilding the whole bloom on a
  configurable cadence** (default 60 s) to drop falsies. Rebuild is
  cheap: `for sess in self.sessions.iter().take(top_k_active) { bloom.insert(prefix_hash(sess.first_n_tokens())) }`.

The router never directly touches sessions — its job is to forward
the call to the shard whose bloom suggests warm cache, then `503`
back the client (with a hint to retry on a different route) if the
shard responds with `unknown session id`. This stale-route mode is
fine because session creation is cheap.

## Failure modes

- **Unknown session id** → 404 with `{"error": "no_such_session"}`.
- **Session in wrong shard** → 410 Gone. Client retries via router
  with a hint header `X-LARQL-Session-Last-Shard` to bias away.
- **KV-format mismatch** → 400 with `{"error":
  "kv_format_mismatch", "expected": "iso3", "got": "fp32"}`.
- **GPU OOM** → 503 with `{"error": "gpu_oom"}`. Server purges the
  oldest sessions until ≥ 20 % VRAM is free, then surfaces 503 to the
  caller (don't retry transparently — caller may want to fall back).
- **Snapshot version mismatch** → 400 with `{"error":
  "snapshot_version_unsupported", "supported_versions": [1]}`.

## Observability

- `tracing::info!` with `session_id`, `model_id`, `op`, `latency_ms`
  on every endpoint.
- Prometheus gauges (deferred to a follow-up):
  `larql_attention_sessions_active`,
  `larql_attention_kv_bytes`,
  `larql_attention_decode_latency_ms_p99`.

## Security

- The existing `--api-key` header gate covers all attention routes;
  no per-session auth.
- Snapshots are encrypted at rest with the configured at-rest key
  (deferred).

## Test plan

| Layer | Test |
|---|---|
| Wire format | `kv_snapshot::tests::round_trip_fp32`, `round_trip_compressed`, `unknown_version_rejected` |
| Session lifecycle | `attention::session::tests::{create_returns_id, get_after_create_works, delete_makes_get_404, ttl_reaps_idle_session}` |
| Prefill | `attention::prefill::tests::{populates_all_layers, returns_residuals, rejects_unknown_session, binary_form_round_trips}` |
| Decode | `attention::decode::tests::{advances_seqlen, residual_matches_local_reference, rejects_pre_prefill_decode}` |
| Router heartbeat | `larql_router::grid::tests::{heartbeat_updates_cached_prefixes, capabilities_default_for_legacy_shards}` |
| Integration | `tests::test_attention_endpoint::e2e_prefill_then_decode_matches_local`, `tests::test_attention_endpoint::snapshot_restore_round_trips_through_router` |
