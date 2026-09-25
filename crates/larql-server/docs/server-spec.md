# Vindex Server — Remote Knowledge & Inference

**Class: ARCHIVE.** This versioned early server draft is retained as written. The [server README](../README.md) and [interface guide](../../../docs/runtime-surfaces.md) describe the current V2/V3 service boundary; mounted routes/capabilities decide availability.

**Version:** 0.1  
**Date:** 2026-04-01  
**Status:** Draft  
**Implementation:** `larql-server` crate (Rust)  
**Depends on:** `larql-vindex`, `larql-inference`

---

## 1. Purpose

A lightweight Rust HTTP server that loads a vindex and serves knowledge queries and inference over the network. No GPU, no ML framework, no Python. One binary.

```bash
larql serve gemma3-4b.vindex --port 8080
# Serving google/gemma-3-4b-it (3.0 GB loaded, browse-only, 0 GPU)
# Endpoints: http://localhost:8080/v1/describe, /v1/walk, /v1/select, ...
```

```bash
larql> USE REMOTE "http://localhost:8080";
larql> DESCRIBE "France";
# capital → Paris (probe) — 47ms round-trip
```

---

## 2. Architecture

```
┌──────────────────────────────────────────────────────┐
│                    larql serve                         │
│                                                       │
│  ┌─────────────────────────────────────────────────┐ │
│  │  HTTP Layer (axum)                               │ │
│  │  Routes → handlers → JSON responses              │ │
│  └────────────────────┬────────────────────────────┘ │
│                       │                               │
│  ┌────────────────────▼────────────────────────────┐ │
│  │  Vindex Layer                                    │ │
│  │  VectorIndex (readonly) + PatchedVindex (overlay)│ │
│  │  gate_knn, walk, describe, feature_meta          │ │
│  └────────────────────┬────────────────────────────┘ │
│                       │                               │
│  ┌────────────────────▼────────────────────────────┐ │
│  │  Weight Layer (optional, lazy-loaded)            │ │
│  │  attn_weights.bin → attention forward pass       │ │
│  │  Only loaded on first INFER request              │ │
│  └─────────────────────────────────────────────────┘ │
│                                                       │
│  Files on disk:                                       │
│  gate_vectors.bin (3 GB) — always loaded              │
│  embeddings.bin (2.5 GB) — always loaded              │
│  down_meta.bin (2 MB) — always loaded                 │
│  attn_weights.bin (2 GB) — loaded on first INFER      │
│  feature_labels.json — always loaded                  │
└──────────────────────────────────────────────────────┘
```

### Design principles

- **Stateless queries.** Every request is independent. No session state on the server. Same input → same output, every time.
- **Lazy weight loading.** Browse endpoints load only gate + embed + down_meta. Attention weights load on first INFER request. Compile weights never load.
- **Readonly base.** The server never modifies the vindex files. Patches are per-session or per-request.
- **No framework dependencies.** No PyTorch, no MLX, no ONNX runtime. Pure Rust with BLAS for the dot products.

---

## 3. CLI

```bash
# Serve a single vindex
larql serve <vindex_path> [OPTIONS]

# Serve multiple vindexes
larql serve --dir <directory> [OPTIONS]

# Serve from HuggingFace (downloads on startup)
larql serve "hf://chrishayuk/gemma-3-4b-it-vindex" [OPTIONS]
```

| Flag | Description | Default |
|------|-------------|---------|
| `--port <PORT>` | Listen port | 8080 |
| `--host <HOST>` | Bind address | 0.0.0.0 |
| `--dir <DIR>` | Serve all .vindex directories in this folder | — |
| `--no-infer` | Disable INFER endpoint (browse-only, reduces memory) | false |
| `--cors` | Enable CORS for browser access | false |
| `--max-concurrent <N>` | Max concurrent requests | 100 |
| `--api-key <KEY>` | Require Bearer token auth (health exempt) | — |
| `--rate-limit <SPEC>` | Per-IP rate limit (e.g. `100/min`, `10/sec`) | — |
| `--trust-forwarded-for` | Trust first `X-Forwarded-For` IP for rate limiting. Enable only behind a trusted reverse proxy. | false |
| `--cache-ttl <SECS>` | Cache TTL for DESCRIBE results (0 = disabled) | 0 |
| `--grpc-port <PORT>` | Enable gRPC server alongside HTTP | — |
| `--uds-path <PATH>` | Bind a Unix domain socket alongside TCP for same-host MoE shard clients (~50 µs/call faster than TCP loopback). Pre-existing socket files are unlinked. Clients use `unix:///path/to/sock` URLs. | — |
| `--experts <START-END>` | (MoE) Serve only this expert ID range across every layer (inclusive). Used to shard the expert bank across machines. | all |
| `--units <PATH>` | (MoE, fine-grained) JSON manifest specifying per-`(layer, expert)` ownership. Mutually exclusive with `--experts`. | — |
| `--warmup-walk-ffn` | Pre-load inference weights + prefetch every owned-layer Q4K mmap at boot (~1.3 s + 3 GB pre-allocated). Recommended for steady-state grid shards. | false |
| `--log-level <LEVEL>` | Logging level | info |
| `--tls-cert <PATH>` | TLS certificate for HTTPS | — |
| `--tls-key <PATH>` | TLS private key | — |

**Environment variables for tuning the MoE remote-expert path** — see
`README.md → Environment variables` for the full table. The names live
in `src/env_flags.rs` (single source of truth: each `LARQL_*` is a
`pub const` with a cached presence accessor backed by `OnceLock`).
Most relevant:

- `LARQL_MOE_NO_SPLIT=1` — opt out of gRPC streaming overlap (default-on
  for gRPC shards; ~12% loopback gain).
- `LARQL_MOE_WIRE_F16=1` — switch the layer-batch wire to f16 (5.5 KB
  vs 11 KB per call; opt-in for LAN deployments).
- `LARQL_HTTP_TIMING=1` / `LARQL_MOE_TIMING=1` — per-call / per-token
  diagnostic timing on stderr.
- `LARQL_NO_WARMUP=1`, `LARQL_USE_LEGACY_CPU=1`,
  `LARQL_USE_METAL_EXPERTS=1`, `LARQL_DISABLE_METAL_EXPERTS=1`,
  `LARQL_DISABLE_Q4K_DIRECT=1`, `LARQL_METAL_VS_CPU_DEBUG=1`,
  `LARQL_MOE_BATCH_MODE=<par|serial|chunked>` — operational + debug
  knobs, all defined in the same module.

**Examples:**

```bash
# Development
larql serve output/gemma3-4b-v2.vindex --port 8080

# Production: browse-only, HTTPS, CORS for web clients
larql serve output/gemma3-4b-v2.vindex \
    --port 443 --no-infer --cors \
    --tls-cert cert.pem --tls-key key.pem

# Multi-model server
larql serve --dir ./vindexes/ --port 8080
# Serves: /v1/gemma-3-4b-it/describe, /v1/llama-3-8b/describe, ...

# From HuggingFace
larql serve "hf://chrishayuk/gemma-3-4b-it-vindex" --port 8080
```

**Startup output:**

```
larql-server v0.1.0
Loading: output/gemma3-4b-v2.vindex
  Model: google/gemma-3-4b-it
  Features: 348,160 (34 layers × 10,240)
  Labels: 1,967 probe-confirmed
  Browse: 5.8 GB loaded (gate + embed + down_meta)
  Infer: available (attn_weights.bin detected, will lazy-load)
Listening: http://0.0.0.0:8080
```

---

## 4. API Endpoints

### 4.1 Knowledge Endpoints (browse-only, always available)

#### GET /v1/describe

Query all knowledge edges for an entity.

```
GET /v1/describe?entity=France
GET /v1/describe?entity=France&band=knowledge
GET /v1/describe?entity=France&band=all&verbose=true
GET /v1/describe?entity=France&limit=10
```

| Param | Type | Description | Default |
|-------|------|-------------|---------|
| `entity` | string | Entity name (required) | — |
| `band` | string | Layer band: `syntax`, `knowledge`, `output`, `all` | `knowledge` |
| `verbose` | bool | Include TF-IDF labels, also-tokens, layer ranges | false |
| `limit` | int | Max edges to return | 20 |
| `min_score` | float | Minimum gate score threshold | 5.0 |

**Response:**

```json
{
  "entity": "France",
  "model": "google/gemma-3-4b-it",
  "edges": [
    {
      "relation": "capital",
      "target": "Paris",
      "gate_score": 1436.9,
      "layer": 27,
      "source": "probe",
      "also": ["Berlin", "Tokyo"]
    },
    {
      "relation": "language",
      "target": "French",
      "gate_score": 35.2,
      "layer": 24,
      "layer_max": 32,
      "count": 4,
      "source": "probe"
    },
    {
      "relation": "continent",
      "target": "Europe",
      "gate_score": 14.4,
      "layer": 25,
      "source": "probe",
      "also": ["Spain", "Australia"]
    }
  ],
  "latency_ms": 12.3
}
```

#### GET /v1/walk

Feature scan — which features fire for a prompt.

```
GET /v1/walk?prompt=The+capital+of+France+is&top=10
GET /v1/walk?prompt=Einstein&top=5&layers=24-33
```

| Param | Type | Description | Default |
|-------|------|-------------|---------|
| `prompt` | string | Prompt text (required) | — |
| `top` | int | Top-K features per layer | 5 |
| `layers` | string | Layer range (e.g. `24-33` or `14,26,27`) | all |

**Response:**

```json
{
  "prompt": "The capital of France is",
  "hits": [
    {"layer": 24, "feature": 4532, "gate_score": 26.1, "target": "French"},
    {"layer": 25, "feature": 4207, "gate_score": 14.4, "target": "Europe"},
    {"layer": 27, "feature": 9515, "gate_score": 1436.9, "target": "Paris"}
  ],
  "latency_ms": 0.4
}
```

#### POST /v1/select

SQL-style edge query.

```json
POST /v1/select
{
  "relation": "capital",
  "limit": 10,
  "order_by": "gate_score",
  "order": "desc"
}
```

```json
POST /v1/select
{
  "entity": "France",
  "min_confidence": 0.5
}
```

**Response:**

```json
{
  "edges": [
    {"entity": "France", "relation": "capital", "target": "Paris", "gate_score": 1436.9},
    {"entity": "Germany", "relation": "capital", "target": "Berlin", "gate_score": 1289.3},
    {"entity": "Japan", "relation": "capital", "target": "Tokyo", "gate_score": 1156.7}
  ],
  "total": 94,
  "latency_ms": 45.2
}
```

#### GET /v1/relations

List all known relation types.

```
GET /v1/relations
GET /v1/relations?source=probe
```

**Response:**

```json
{
  "relations": [
    {"name": "capital", "count": 94, "source": "probe", "example": "France→Paris"},
    {"name": "language", "count": 51, "source": "probe", "example": "France→French"},
    {"name": "birthplace", "count": 91, "source": "probe", "example": "Mozart→Salzburg"}
  ],
  "total_probe": 1967,
  "total_cluster": 512,
  "latency_ms": 0.1
}
```

#### GET /v1/stats

Model and index statistics.

```
GET /v1/stats
```

**Response:**

```json
{
  "model": "google/gemma-3-4b-it",
  "family": "gemma3",
  "layers": 34,
  "features": 348160,
  "features_per_layer": 10240,
  "hidden_size": 2560,
  "vocab_size": 262208,
  "extract_level": "all",
  "dtype": "f32",
  "probe_confirmed": 1967,
  "cluster_types": 512,
  "layer_bands": {
    "syntax": [0, 13],
    "knowledge": [14, 27],
    "output": [28, 33]
  },
  "loaded": {
    "browse": true,
    "inference": true,
    "compile": false
  },
  "memory_mb": 5800
}
```

### 4.2 Inference Endpoint (requires attention weights)

#### POST /v1/infer

Full forward pass with attention. Returns next-token predictions.

```json
POST /v1/infer
{
  "prompt": "The capital of France is",
  "top": 5,
  "mode": "walk"
}
```

| Field | Type | Description | Default |
|-------|------|-------------|---------|
| `prompt` | string | Prompt text (required) | — |
| `top` | int | Top-K predictions | 5 |
| `mode` | string | `walk` (vindex FFN) or `dense` (original FFN) or `compare` | `walk` |

**Response:**

```json
{
  "prompt": "The capital of France is",
  "predictions": [
    {"token": "Paris", "probability": 0.9791},
    {"token": "the", "probability": 0.0042},
    {"token": "a", "probability": 0.0031}
  ],
  "mode": "walk",
  "latency_ms": 210
}
```

**Compare mode:**

```json
{
  "prompt": "The capital of France is",
  "walk": [
    {"token": "Paris", "probability": 0.9791}
  ],
  "dense": [
    {"token": "Paris", "probability": 0.9801}
  ],
  "latency_ms": 420
}
```

### 4.3 Patch Endpoints

#### POST /v1/patches/apply

Apply a patch to the server's in-memory vindex. Per-session — does not modify base files.

```json
POST /v1/patches/apply
{
  "url": "hf://medical-ai/drug-interactions@2.1.0"
}
```

Or upload a patch directly:

```json
POST /v1/patches/apply
{
  "patch": {
    "version": 1,
    "operations": [
      {"op": "insert", "layer": 26, "feature": 8821, ...}
    ]
  }
}
```

**Response:**

```json
{
  "applied": "drug-interactions@2.1.0",
  "operations": 5000,
  "active_patches": 1
}
```

#### GET /v1/patches

List active patches.

```json
{
  "patches": [
    {"name": "drug-interactions@2.1.0", "operations": 5000, "source": "hf://medical-ai/drug-interactions@2.1.0"}
  ]
}
```

#### DELETE /v1/patches/{name}

Remove a patch.

```
DELETE /v1/patches/drug-interactions@2.1.0
```

### 4.4 Management Endpoints

#### GET /v1/health

```json
{"status": "ok", "uptime_seconds": 3600, "requests_served": 12450}
```

#### GET /v1/models

List loaded models. Response conforms to the
[OpenAI Models API](https://platform.openai.com/docs/api-reference/models/list)
shape, which means existing `openai` SDKs work unmodified. Larql-specific
fields (`path`, `features`, `loaded`) are present as additional members —
OpenAI clients ignore them.

```json
{
  "object": "list",
  "data": [
    {
      "id": "gemma-3-4b-it",
      "object": "model",
      "created": 1746094800,
      "owned_by": "larql",
      "path": "/v1/gemma-3-4b-it",
      "features": 348160,
      "loaded": true
    }
  ]
}
```

### 4.5 Session Endpoints (N1, shipped 2026-08-22)

The operational view and eviction control plane for runtime session
state. **Observability and termination only** — this surface never
becomes a second authority for modifying model state, so there is no
create, no rename, and no patch editing here. Sessions come into
existence by being *used*: an `X-Session-Id` on `/v1/patches/apply`,
`/v1/insert`, or `/v1/responses` binds one.

The representation is metadata: patch identities, continuation
availability, and token counts. Never overlay contents, KV bytes,
tokenizer state, or cache keys.

#### GET /v1/sessions

Live sessions, most recently used first.

```json
{
  "object": "list",
  "data": [
    {
      "object": "session",
      "id": "sess-medical-team",
      "model": "gemma-3-4b-it",
      "created_at": 1755861234,
      "last_used_at": 1755861299,
      "expires_at": 1755864899,
      "state": "active",
      "patches": {"active": 2, "ids": ["oncology-v3", "dosing-fix"]},
      "continuation": {
        "available": true,
        "input_tokens": 412,
        "resumptions": 3,
        "reused_tokens_total": 276
      }
    }
  ]
}
```

`state` is `active` — expiry is applied at *read* time, so an expired
session is absent rather than listed as expired. (Reads take a read
guard and filter; they never evict, because a writer here would queue
behind an in-flight sessioned forward pass and stall both inference and
every other observer.)

`continuation.available` / `input_tokens` describe the KV state
currently resident for this session; `resumptions` /
`reused_tokens_total` are cumulative and survive an emptied cache.

#### GET /v1/sessions/{session_id}

The same object for one session; `404` if it is absent or expired.

#### DELETE /v1/sessions/{session_id}

Idempotent and strong. Frees the patch overlay, frees every KV
continuation the session owns, and retires the session so an in-flight
generation cannot re-insert one afterwards. Unrelated sessions are
untouched.

```json
{
  "object": "session.deleted",
  "id": "sess-medical-team",
  "deleted": true,
  "patches_freed": 2,
  "continuations_freed": 1
}
```

Repeat deletes are not an error — `deleted: false`, nothing freed. (This
differs from `DELETE /v1/responses/{id}`, which 404s: a cleanup endpoint
an operator or a script drives should not fail for having already
succeeded.)

**Frozen concurrency contract.** A delete that lands while a sessioned
request is mid-forward-pass waits for that request's read guard: the
in-flight request completes safely on the state it already acquired,
and the deletion then prevents any subsequent acquisition. For the
continuation path — which retains KV *after* generation ends, outside
any session lock — deletion retires the session's lease, and the KV
cache checks that lease while holding its own lock. So the two
possible orderings both end correctly: an insert that gets in first is
removed by the purge that follows, and an insert that arrives after the
retirement is refused. A deleted session can never be resurrected by a
late write.

**TTL interaction.** Expiry uses the same machinery: the maintenance
sweeper (or any manager write path) evicts idle sessions and retires
their leases, at which point the session vanishes from `GET`, from the
listing, and from `/v1/stats.server.sessions.active` together, and its
continuations become orphans that the KV cache's own sweep collects.

### 4.6 OpenAI-Compatible Endpoints (N0 slice 1, shipped 2026-05-02)

Endpoints conforming to the [OpenAI API](https://platform.openai.com/docs/api-reference)
shape: `/v1/models`, `/v1/embeddings`, `/v1/completions`,
`/v1/chat/completions`, and `/v1/responses` (+ `GET`/`DELETE
/v1/responses/{id}`). Existing `openai` Python/JS SDKs work unmodified
— point `base_url` at the larql server and the SDK calls just work.

#### GET /v1/models — covered in §4.4 above (now OpenAI-shape).

#### POST /v1/embeddings

```
Request:  {model?, input: string | string[] | int[] | int[][],
           encoding_format?: "float" | "base64",
           dimensions?, user?}
Response: {object: "list",
           data: [{object: "embedding", embedding: [f32...], index}],
           model, usage: {prompt_tokens, total_tokens}}
```

- `input` accepts strings (server tokenises) or pre-tokenised arrays.
- Pooling: **mean-pool** over per-token static embeddings. Equivalent
  to `np.mean(embeddings_table[token_ids], axis=0)`. Treat as
  "lookup-pooled" not "semantic" embeddings.
- `encoding_format: "base64"` (slice 4.8) returns each vector as a
  base64-encoded little-endian f32 byte string. ~33% smaller wire than
  the JSON float-array form; many production OpenAI clients default to
  base64.
- `dimensions`, `user` accepted but no effect (logged via tracing).

#### POST /v1/completions

```
Request:  {model?, prompt: string | string[],
           max_tokens?, temperature?, top_p?,
           stream?, logprobs?, echo?, stop?,
           n?, best_of?, seed?, user?}
Response: {id: "cmpl-...", object: "text_completion", created,
           model,
           choices: [{text, index, finish_reason, logprobs: null}],
           usage: {prompt_tokens, completion_tokens, total_tokens}}
```

Live: SSE streaming, KV-cached generation, `temperature` / `top_p` /
`seed` / `stop` / `frequency_penalty` / `presence_penalty` honoured
by the sampler, `logprobs: int` populates per-token entries (top-k
alternatives placeholder pending inference work — F18 follow-up).
Constraints:
- `n>1` → 400.
- `stop` → string or string-array; first match halts generation; the
  matched substring is trimmed from the returned `text`.
- `echo: true` → prepends the prompt to the returned `text`. Disallowed
  in stream mode.
- Batched `prompt: [...]` disallowed in stream mode.
- `best_of` → accepted, treated as 1.

`finish_reason` values: `"stop"` (EOS token, end-of-turn marker, or
matched stop string) or `"length"` (hit `max_tokens`).

#### POST /v1/chat/completions

Multi-turn chat with chat-template rendering.

```
Request:  {model?, messages: [{role: "system"|"user"|"assistant"|"tool",
                                content?, tool_calls?, tool_call_id?, name?}, ...],
           max_tokens?, temperature?, top_p?,
           stream?, n?, stop?,
           tools?, tool_choice?, response_format?,
           logprobs?, top_logprobs?,
           frequency_penalty?, presence_penalty?, seed?, user?}
Response: {id: "chatcmpl-...", object: "chat.completion", created,
           model,
           choices: [{
             index,
             message: {role: "assistant",
                       content: string|null,
                       tool_calls?: [{id, type:"function",
                                      function: {name, arguments}}]},
             finish_reason: "stop"|"length"|"tool_calls",
             logprobs: ChatLogprobs | null
           }],
           usage: {prompt_tokens, completion_tokens, total_tokens}}
```

Chat-template selection (auto-detected):
- `arch.family()` returns `gemma2` / `gemma3` / `gemma4` → Gemma
  (`<start_of_turn>` / `<end_of_turn>`)
- `llama` → Llama 3 header tags
  (`<|start_header_id|>...<|end_header_id|>...<|eot_id|>`)
- `qwen` / `qwen2` / `qwen3` / `deepseek` / `gpt_oss` → ChatML
  (`<|im_start|>{role}\n...<|im_end|>`)
- `mistral` / `mixtral` → Mistral `[INST] ... [/INST]` with system
  prepended to first user
- anything else → Plain `User: ...\nAssistant: ...` markers

Sampling fields (`temperature`, `top_p`, `seed`, `stop`,
`frequency_penalty`, `presence_penalty`) are honoured end-to-end
through `SamplingConfig` + `EosConfig`. Penalties clamp to
`[-2.0, 2.0]` per OpenAI's documented range.

Tool-result replay (slice 4.9): assistant messages may carry
`tool_calls` and `content: null`; clients then send a follow-up
`role: "tool"` message with `tool_call_id` and execution result in
`content`. Both render into the chat template before the next
generation pass.

`logprobs: true` (slice F18) populates `choices[i].logprobs.content[]`
with `{token, logprob, bytes, top_logprobs}` per emitted token.
`top_logprobs` currently returns the picked token only; the full
top-K alternatives are gated on inference work.

#### POST /v1/responses (N0.3, shipped 2026-08-22)

The [Responses API](https://platform.openai.com/docs/api-reference/responses).
Served by both V2 and V3 runtimes.

```
Request:  {model?, input: string | [item...], instructions?,
           previous_response_id?, store?, stream?, tools?, tool_choice?,
           text?: {format: {...}}, max_output_tokens?, temperature?,
           top_p?, metadata?, user?}
Response: {id: "resp_...", object: "response", created_at, model,
           status: "completed" | "incomplete",
           incomplete_details?, output: [item...],
           usage: {input_tokens, output_tokens, total_tokens,
                   input_tokens_details: {cached_tokens}}}
```

- **Conversation chaining.** `previous_response_id` replays the stored
  conversation ahead of this request's input, and the stored model id
  becomes the default `model` so a follow-up lands on the same runtime.
  An unknown id is a 404.
- **Server-side storage.** `store` (default true) retains the envelope
  and conversation in a bounded FIFO for chaining and for `GET
  /v1/responses/{id}`; `store: false` retains nothing and the response
  is not chainable.
- **Streaming.** `stream: true` emits the typed event sequence
  (`response.created` → `response.output_item.added` →
  `response.content_part.added` → `response.output_text.delta`… →
  `response.output_text.done` → `response.content_part.done` →
  `response.output_item.done` → `response.completed`), terminated by
  `data: [DONE]`. A generation failure emits `response.failed`.
- **Structured output and tools.** `text.format` and `tools` route
  through the same constrained-decoding FSM as `/v1/chat/completions`
  (below), on both the V2 and V3 arms.
- **KV continuation.** On a V3 runtime a chain can resume from the
  producing turn's resident KV state instead of re-prefilling — see
  §4.5 and the N1 notes below.
- `background: true` → 400 (larql serves responses synchronously).

#### GET /v1/responses/{response_id}

Returns the stored envelope. A response created with `store: false`, or
evicted by the FIFO cap, is a 404 — the same contract clients already
handle for expired server-side state.

#### DELETE /v1/responses/{response_id}

`{id, object: "response.deleted", deleted: true}`. Unlike
`DELETE /v1/sessions/{id}` this is **not** idempotent: an unknown id is
a 404.

#### N1 — KV continuation for chained responses (V3, shipped 2026-08-22)

`previous_response_id` chains on a V3 runtime resume from a resident KV
state rather than re-prefilling the whole conversation. The producing
turn's `V3KvHandoff { continuation, absorbed_ids }` stays in a bounded, TTL-swept
cache keyed by response id (`--v3-kv-cache-entries`, default 4;
`--v3-kv-ttl-secs`, default 600; swept by the maintenance sweeper).

The load-bearing property is that **resumption is purely an
optimisation**. The generation path reuses a handoff only when its
absorbed ids are a strict prefix of the new prompt's ids, and otherwise
falls back to a full prefill — so the produced tokens are identical
either way, and losing an entry (eviction, TTL, restart, branching)
can never change an answer. This is gated bit-for-bit.

Three further contracts:

- **Take-once.** A chained request *takes* the entry. Branching two
  continuations off one response gives the second a full prefill —
  correct, just not accelerated.
- **Model-keyed.** A take naming a different runtime binding is a miss
  and does **not** consume the entry: a KV state under other weights
  would be garbage even when a shared tokenizer makes the id prefix
  coincide, and the rightful chain must still be able to claim it.
- **Session-owned when the client asks.** A request carrying
  `X-Session-Id` binds a session that owns the retained state, so
  `DELETE /v1/sessions/{id}` frees it (§4.5).
- **Authority-checked before the prefix (CONTINUATION-PLUGIN-1 C4).**
  The retained state is sealed with the continuation authority that
  built it: provider identity and configuration digest, compared
  separately. Before the prompt is compared, that authority must equal
  the one the model's binding selected. Otherwise the resume contract
  refuses (`provider_absent`, `revision_changed` or
  `configuration_changed`, naming both sides), and the refused state is
  dropped, never reinterpreted. The server then recovers with a full
  fresh prefill, but only explicitly: the response carries
  `usage.input_tokens_details.continuation_refused` (`kind`, `reason`,
  `recovery: "fresh_prefill"`), and `/v1/stats` counts it under
  `refusals`. It is never counted as a miss. The field is absent when
  nothing was refused. The produced tokens still equal a fresh run's.

Observability: a hit surfaces as `usage.input_tokens_details.
cached_tokens`; `/v1/stats.server.v3_kv` carries `hits`, `misses`,
`resumptions`, `refusals`, `reused_tokens_total`. `hits - resumptions` is the live
**prefix-stability gap** — resident state found but unusable under
exact token-id identity.

What it is worth, measured on a four-turn chain: **2.0–2.4x** on total
wall time (gpt-oss-20b 45.02 s → 18.77 s; Granite 4.1 3B 15.94 s →
7.69 s). The shape is the point — without the cache, turn time grows
linearly as the conversation is re-prefilled from scratch (4.3 → 8.8 →
13.5 → 18.4 s); with it, turn time is flat (4.3 → 5.0 → 4.6 → 4.9 s),
because only the new turn is prefilled.

Whether resumption engages on a given turn depends on the **last token
the model emitted**: if the conversation is re-rendered and that token
re-tokenises differently once the next turn follows, the exact-ids
prefix breaks at the seam and the turn falls back to a full prefill. A
chat template that terminates the assistant turn with an atomic special
token makes the seam stable; the `Plain` fallback, which ends a turn
with a bare newline, does not guarantee it either way.

#### Constrained decoding (slice 4 / N0.6, shipped 2026-05-02)

`response_format` and `tools` route the request through a
schema-typed JSON FSM that masks the LM head per token. Since
2026-08-22 the same masking serves the **V3** runtimes too, on
`/v1/completions`, `/v1/chat/completions`, and `/v1/responses` — one
`drive_session` loop behind both drivers, with mask exhaustion before
the first emission surfacing as an error and after it as a natural
stop (mirroring the V2 `MaskRejectedAllCandidates` contract). Masking
and KV resumption are orthogonal: a chained request can resume *and*
be constrained.

Driving the FSM through an executable JSON path on V3 exposed two
latent bugs shared with the V2 arm, which the V2 fixtures could never
spell and so had never caught: property keys were unconstrained during
emission (the mask admitted a doomed key and then dead-ended), and a
trailing comma was accepted with no viable key left. Both are fixed —
a closed object now offers an enum of still-viable, not-yet-seen
property names, refuses to open a key when none remain, and requires a
key after a comma.

| Request                                         | Schema enforced                                    |
|-------------------------------------------------|----------------------------------------------------|
| `response_format: {"type":"text"}` (or omitted) | none (plain sampling)                              |
| `response_format: {"type":"json_object"}`       | `Object(any)` — any structurally-valid JSON object |
| `response_format: {"type":"json_schema", "json_schema":{"schema":..., "strict": bool}}` | parsed schema; `strict` flips `additionalProperties` default to false |
| `tools: [{type:"function", function:{name, parameters}}, ...]` | `OneOf` of `{name=Const, arguments=<args>}` per tool |

`tool_choice` resolves as: `"auto"` / `"required"` (default when tools
present) → all branches; `"none"` → no constraint; `{type:"function",
function:{name}}` → single matching branch. Unknown tool name → 400.

JSON Schema parser supports `type` (incl. arrays like
`["string","null"]`), `properties`, `required`, `additionalProperties`,
`items`, `minItems`/`maxItems`, `enum`, `const`, `oneOf`/`anyOf`,
`minLength`/`maxLength`, `minimum`/`maximum`, integer-vs-number.
`$ref`, `pattern`, `format`, `allOf`, `not`, `if/then/else`, `false`
schema → 400 with explicit message (no silent relaxation).

Sampling under mask (slice 4.10): the constrained decoder runs
through `pick_next_token_masked_sampled`, which consumes the same
`SamplingConfig` as unconstrained generation. So `temperature`,
`top_p`, `seed`, `frequency_penalty`, `presence_penalty` all apply
on top of the mask. Defaults are greedy.

Tool-call response shape: `message.content: null`, `tool_calls:
[{id: "call_<hex>", type: "function", function: {name, arguments}}]`,
`finish_reason: "tool_calls"`. `arguments` is JSON-stringified
(matches OpenAI's wire shape; SDKs `json.loads` it).

Tools + `stream=true` (slice 4.11): the constrained decoder runs in
buffered mode and emits a single `chat.completion.chunk` carrying the
full `delta.tool_calls[0]` payload, followed by a final chunk with
`finish_reason: "tool_calls"`. Per-token argument streaming is a
follow-up tightening — most OpenAI clients accumulate `arguments`
incrementally and only act on `finish_reason`, so a single fat chunk
is wire-compatible.

EOS tokens are masked while the FSM is mid-structure and become legal
once `is_complete()`. Per-step overhead is `O(vocab × avg_token_len)`
for the surface-form replay; `build_mask` caches the surface-form
table once per request, plus FSM clone+replay per candidate
(~ns × token chars).

Other constraints:
- `n>1` → 400 (single completion per prompt).

#### VINDEX3 runtimes (VI3-SERVE-1)

**What a V3 binding refuses.** `load_artifact`'s V3 branch takes a path
and nothing else — slicing and service modes have no V3 implementation
— so `--no-infer`, `--ffn-only`, `--embed-only`, `--layers`,
`--experts`, `--units` and `--moe-remote` are **refused at load** rather
than accepted and ignored. They used to be silently dropped, which is
the dangerous shape: a `--layers 0-9` "shard" loaded the whole model
and answered complete requests, and `--no-infer` did not disable
inference. V2 cache knobs (`--hnsw`, gate/q4k cache sizes,
`--release-mmap-after-request`) are accepted and simply have nothing to
tune.

**Operand residency.** A container's operands are lowered into the
backend's execution form **once, at bind time**
(`Vindex3Runtime::prepare` → `PreparedVindex3`), and every request reads
that one image; a request contributes only continuation state. Batch
prefill and the decode session read the *same* prepared operands — they
used to materialise the model separately, so a warm request paid for
the model twice (measured 31.9 s → 0.75 s on a 20 B container once that
stopped). Preparation takes an `ExecutionSlice`, so a future shard
prepares only the operands its slice executes; a slice the plan cannot
satisfy is refused rather than truncated.

**Chat template.** A V3 container has no architecture-registry entry, so
the template is resolved from the **container's own declared family**
(read from the inspection the runtime already performs), falling back
to the model-id heuristic and then to `Plain`. Landing on `Plain` is warned about at bind time, because it is
not cosmetic: `Plain` ends an assistant turn with a bare newline, so the
turn's last token re-tokenises differently once the next turn follows —
which breaks N1's exact-ids-prefix rule at the seam and costs every
chained request its resumption.


`larql serve` binds each artifact through `bootstrap::load_artifact`
(`src/bootstrap/load.rs`), which dispatches on the container's generation
marker: a VINDEX3 container binds as `LoadedArtifact::V3` carrying a
`V3Model` (`src/vindex3.rs`) — a `Vindex3Runtime` opened with the
`ProductionBackend` on the container's `target` component, plus the
container's `tokenizer.json` — while VINDEX2 takes the existing
`load_single_vindex` path. `AppState::served` (`src/state.rs`)
resolves a request's model across both registries; routes match the
returned binding once, at the top, and nothing below it re-detects
the container generation.

Served surface for a V3 model:

- `GET /v1/models` — listed with a larql-specific `generation: 3`
  extra; no `features` count (a V3 container is an executable
  program, not a feature index).
- `POST /v1/completions` — buffered and SSE streaming, served by the
  V3 runtime (`src/routes/openai/v3_completions.rs`). Request
  validation, chunk shape, stop-string trimming, and `finish_reason`
  semantics are shared with the V2 path; only the token source
  differs.
- `POST /v1/chat/completions` — same shape, V3 arm in
  `src/routes/openai/chat/v3.rs`. The V2/V3 decision is made once when
  the model is resolved; everything wire-shaped below it is shared.
- `POST /v1/responses` — buffered and streaming, including tools /
  structured output and `previous_response_id` chaining with KV
  continuation (see N1 above).

`/v1/stats` answers on a V3 binding with the program's own shape read
from the opened plan (component, layer count, hidden and vocab size,
whether it carries an output head) plus the server-level block — which
is the only surface carrying the N1 continuation counters, so it has to
work where V3 runs. It does not fake the V2 vocabulary (features,
bands, extract level, q4k caches) onto a V3 binding.

`/v1/embeddings` and the browse endpoints resolve V2 models only — a
V3 container is an executable program, not a feature index, and has no
static embedding table to pool over.

Generation opens a fresh session per request (`generate_v3` in
`src/vindex3.rs`): caller-owned `CanonicalKvState` →
`prefill_into` → `session_with_kv` → `continue_session`, emitting
each token through the incremental detokenizer.
`generate_v3_resumable` is the same stack with an optional inbound
handoff, and returns the state for the next chain link. Plan operands
stay resident per session, not per server — a shared resident
session/operand pool is follow-up perf work.

#### N0-router (shipped 2026-08-22)

These endpoints are mirrored on `larql-router` so the grid is a single
OpenAI endpoint. A server announces `AnnounceMsg.serves_openai` only
when it can answer a complete request alone (full layer coverage,
inference enabled, no shard filter); the router aggregates
`/v1/models` across capable servers and proxies chat/completions/
embeddings to the least-loaded one matching `model`, streaming SSE
back unbuffered. `/v1/responses` is proxied with sticky id → backend
routing so a `previous_response_id` chain returns to the server
holding the conversation and its resident KV state. Grid-only: a
static `--shards` map carries no capability signal and answers 503.
Contract in `crates/larql-router/CHANGELOG.md`.

### 4.7 Remote MoE Expert Endpoints

For hybrid-MoE models (e.g. Gemma 4 26B-A4B), the inference client runs
attention + dense FFN + the per-layer router locally and dispatches
selected expert work to one or more shard servers. Three wire formats are
exposed; new deployments should default to `layer-batch` (or `-f16` on
bandwidth-constrained links).

#### POST /v1/experts/layer-batch

`Content-Type: application/x-larql-experts-layer`. Single residual + K
`(expert_id, weight)` pairs for one layer; server applies
`pre_experts_norm` once, quantises h_norm to Q8_K once, fans out the K
expert kernels with the shared activation via rayon, returns the
router-weighted sum.

```
Request:  [4: layer u32 LE][4: hidden u32][4: K u32]
          + hidden × f32  (residual, sent ONCE per call)
          + K × [4: expert_id u32, 4: weight f32]

Response: [4: hidden u32 LE][4: latency_ms f32]
          + hidden × f32  (router-weighted sum across K experts)
```

Replaces the legacy `/v1/expert/batch` (which shipped K identical residual
copies on the wire). Saves ~2.6 MB/token of redundant wire data plus K-1
redundant per-call CPU work on the server.

#### POST /v1/experts/layer-batch-f16

`Content-Type: application/x-larql-experts-layer-f16`. Same shape as
`layer-batch` but residual + response use IEEE-754 binary16 — halves wire
bytes (5.5 KB request + 5.5 KB response vs 11 + 11 KB f32). Opt-in via
`LARQL_MOE_WIRE_F16=1` on the client; server always exposes both
endpoints. f16 quant noise is well below the Q8_K activation
quantisation already applied in the SDOT path; end-to-end accuracy
unchanged.

#### POST /v1/expert/batch (legacy)

`Content-Type: application/x-larql-expert`. Pre-2026-05-01 wire: N items
each with `(layer, expert_id, residual)`; ships K identical residuals
when called from `forward_moe`. Still served for back-compat. Returns N
per-expert outputs which the client weights and sums (vs server-side
weighting + summing in `layer-batch`).

#### POST /v1/expert/{layer}/{expert_id}

JSON-only single-expert dispatch. Diagnostic / smoke-test path:

```
POST /v1/expert/15/47
{"residual": [0.12, -0.03, ...]}
→ {"output": [0.4, 0.1, ...], "latency_ms": 0.5}
```

#### Transport options

Each `--moe-shards` entry's URL scheme picks the transport:

- `grpc://host:port` — persistent HTTP/2; enables fire/collect streaming
  overlap with dense FFN GPU compute (default-on; ~12% faster on M3 Max
  loopback). Set `LARQL_MOE_NO_SPLIT=1` to opt out.
- `http://host:port` — TCP/HTTP. Server sets `TCP_NODELAY` on accepted
  connections by default to avoid Nagle tail-packet stalls on real LAN.
- `unix:///abs/path/to/sock` — manual HTTP/1.1 over a Unix domain
  socket; ~50 µs/call faster than TCP loopback. Same wire format as
  the TCP HTTP path. Same-host only (matches the server's
  `--uds-path`).

---

## 5. Multi-Model Serving

When serving a directory, each vindex gets its own namespace:

```bash
larql serve --dir ./vindexes/ --port 8080
# Found: gemma-3-4b-it, llama-3-8b, mistral-7b
```

```
GET /v1/gemma-3-4b-it/describe?entity=France
GET /v1/llama-3-8b/describe?entity=France
GET /v1/mistral-7b/describe?entity=France
```

Single-model serving uses the root path:

```
GET /v1/describe?entity=France
```

---

## 6. Client-Side Patches

The server hosts the immutable base. The client brings patches. Two modes:

### 6.1 Session Patches (server-side)

Client sends a patch, server applies it in-memory for that session. Good for web apps where the client can't run gate KNN locally.

```
POST /v1/patches/apply  {"url": "..."}
GET  /v1/describe?entity=aspirin
# Returns edges from base + patch
```

The patch stays in memory until the server restarts or the patch is removed. Each client session can have different patches — the server maintains per-session PatchedVindex instances.

### 6.2 Client-Side Overlay (client-side)

Client has the patch locally. Sends queries to the server, overlays patch results locally. Good for privacy — patch contents never leave the client.

```
Client:
  1. GET /v1/describe?entity=aspirin → base edges from server
  2. Check local patch for aspirin edges → local overrides
  3. Merge: server edges + local overrides → display
```

The REPL handles this transparently:

```sql
USE REMOTE "http://localhost:8080";
APPLY PATCH "private-facts.vlp";        -- stays local
DESCRIBE "aspirin";
-- Server provides base edges
-- REPL overlays local patch edges
-- User sees combined result
-- Server never sees patch contents
```

---

## 7. Performance

### 7.1 Expected Latency

| Endpoint | Server Processing | Network (LAN) | Network (Internet) | Total |
|----------|------------------|---------------|-------------------|-------|
| /v1/describe | 12ms | 1ms | 50ms | 13-62ms |
| /v1/walk | 0.4ms | 1ms | 50ms | 1-50ms |
| /v1/select | 5ms | 1ms | 50ms | 6-55ms |
| /v1/relations | 0.1ms | 1ms | 50ms | 1-50ms |
| /v1/stats | 0.01ms | 1ms | 50ms | 1-50ms |
| /v1/infer | 200ms | 1ms | 50ms | 200-250ms |

Browse queries are dominated by network latency, not computation. The server processes DESCRIBE in 12ms — the rest is round-trip time.

### 7.2 Memory Usage

| Mode | Memory | What's loaded |
|------|--------|---------------|
| Browse-only | ~6 GB (f32) / ~3 GB (f16) | gate + embed + down_meta + labels |
| Browse + Infer | ~8 GB (f32) / ~4 GB (f16) | + attn_weights |
| Browse + Infer + Patches | ~8 GB + patches | + per-session PatchedVindex |

### 7.3 Throughput

The server is stateless for browse queries — each request is an independent gate KNN. This means:

- **Horizontal scaling:** Run N instances behind a load balancer. Each loads the same vindex (or uses mmap to share pages).
- **CDN caching:** For popular entities, cache DESCRIBE responses at the edge. TTL-based invalidation when labels update.
- **Concurrent requests:** axum handles async I/O. Gate KNN is CPU-bound but completes in 12ms. 100 concurrent requests = 1.2 seconds of CPU per batch.

At 12ms per DESCRIBE, a single server handles ~80 queries/second. With 4 instances: ~320 queries/second. More than enough for most use cases.

---

## 8. Security

### 8.1 Authentication (implemented)

Optional API key authentication. `/v1/health` is exempt (always accessible).

```bash
larql serve gemma3-4b.vindex --port 8080 --api-key "sk-abc123"
```

```
GET /v1/describe?entity=France
Authorization: Bearer sk-abc123
```

Requests without a valid token receive 401 Unauthorized.

### 8.2 Concurrency Limit (implemented)

Max concurrent requests via tower middleware:

```bash
larql serve gemma3-4b.vindex --max-concurrent 100
```

### 8.3 Rate Limiting (implemented)

Per-IP token bucket rate limiting. Supports `N/sec`, `N/min`, `N/hour` formats.
`/v1/health` is exempt. The default bucket key is the socket peer IP; untrusted
client-supplied `X-Forwarded-For` is ignored.

```bash
larql serve gemma3-4b.vindex --rate-limit "100/min"

# Behind a trusted reverse proxy only:
larql serve gemma3-4b.vindex --rate-limit "100/min" --trust-forwarded-for
```

Excess requests receive `429 Too Many Requests`.

### 8.3.1 Error Envelope

There are **two** error envelopes, split by endpoint family:

**LARQL paradigm endpoints** (`/v1/describe`, `/v1/walk`, `/v1/select`,
`/v1/relations`, `/v1/stats`, `/v1/infer`, `/v1/patches/*`,
`/v1/walk-ffn`, `/v1/insert`, `/v1/explain-infer`, `/v1/embed`,
`/v1/logits`, `/v1/token/*`) use a **flat** envelope:

```json
{"error": "message"}
```

**OpenAI-compatible endpoints** (`/v1/embeddings`, `/v1/completions`,
`/v1/chat/completions`) use the **nested** OpenAI-canonical envelope so
the OpenAI Python and JS SDKs parse errors without special-casing:

```json
{
  "error": {
    "message": "the encoding_format 'foo' is not supported",
    "type": "invalid_request_error",
    "param": null,
    "code": null
  }
}
```

`type` values used by this server:

| HTTP status | OpenAI `type`              |
|-------------|----------------------------|
| 400         | `invalid_request_error`    |
| 404         | `not_found_error`          |
| 500         | `server_error`             |
| 503         | `service_unavailable_error`|

Both `param` and `code` are always present (possibly `null`) — some SDKs
hard-key on those fields.

WebSocket messages keep their streaming protocol shape:
`{"type": "error", "message": "..."}`. Mid-stream SSE errors on the
OpenAI endpoints emit `data: {"error": {"message": "...", "type": "server_error"}}`
chunks (see `routes/openai/util.rs::error_chunk`).

### 8.4 DESCRIBE Cache (implemented)

In-memory TTL cache for DESCRIBE results. Keys include model ID, entity, band, limit, min_score.

```bash
larql serve gemma3-4b.vindex --cache-ttl 300  # 5 minute TTL
```

### 8.5 Per-Session Isolation (implemented)

Patches can be scoped to a session via the `X-Session-Id` header. Each
session gets its own PatchedVindex overlay, materialised lazily — a
session that never patches carries none and reads exactly like the
global state. Sessions expire after 1 hour of inactivity by default
(`--session-ttl-secs`); §4.5 is the surface for observing and evicting
them.

```
POST /v1/patches/apply
X-Session-Id: sess-medical-team
{"patch": {...}}
```

Without the header, patches go to the global shared state.

### 8.6 Patch Validation

Patches uploaded via API are validated before application:
- Base model checksum must match
- Operations must reference valid layers and features
- Gate vectors must have correct dimensions

### 8.4 No Write Access

The server never modifies vindex files on disk. All mutations are in-memory via PatchedVindex. The base vindex is readonly. There is no endpoint to modify the base files.

---

## 9. REPL Integration

The REPL connects to remote servers transparently:

```sql
USE REMOTE "http://localhost:8080";
-- Connected: google/gemma-3-4b-it (348K features, 1967 probe-confirmed)

DESCRIBE "France";
-- Sends GET /v1/describe?entity=France
-- Displays same format as local DESCRIBE

WALK "Einstein" TOP 10;
-- Sends GET /v1/walk?prompt=Einstein&top=10

SELECT entity, target FROM EDGES WHERE relation = "capital" LIMIT 5;
-- Sends POST /v1/select

INFER "The capital of France is" TOP 5;
-- Sends POST /v1/infer

APPLY PATCH "local-facts.vlp";
-- Patch stays local, overlaid on remote results

SHOW PATCHES;
-- Shows local patches only
```

The user doesn't know or care whether the vindex is local or remote. Same statements, same output format.

**Implementation status:** `USE REMOTE` is fully implemented. Supported: DESCRIBE, WALK, INFER, SELECT, STATS, SHOW RELATIONS, INSERT, DELETE, UPDATE. Local patches (APPLY PATCH) stay client-side and overlay on remote DESCRIBE results.

---

## 10. Deployment

### 10.1 Docker

```dockerfile
FROM rust:1.82-slim AS builder
WORKDIR /build
COPY . .
RUN cargo build --release -p larql-server

FROM debian:bookworm-slim
COPY --from=builder /build/target/release/larql-server /usr/local/bin/
EXPOSE 8080
ENTRYPOINT ["larql-server"]
```

```bash
docker run -v ./vindexes:/data -p 8080:8080 larql-server /data/gemma3-4b.vindex
```

### 10.2 Fly.io / Railway

```toml
# fly.toml
[build]
  builder = "dockerfile"

[env]
  VINDEX_PATH = "/data/gemma3-4b.vindex"

[mounts]
  source = "vindex_data"
  destination = "/data"

[[services]]
  internal_port = 8080
  protocol = "tcp"
  [services.concurrency]
    hard_limit = 100
```

Browse-only deployment: 3 GB RAM (f16). $5-10/month on Fly.io.

For **distributed MoE serving** (multi-shard Gemma 4 26B-A4B etc.) on
fly.io, see `ROADMAP.md → F-FLY`. Open items: VM size for shards
(`performance-cpu-4x`+ for ~10 GB RSS at warmup), vindex distribution
strategy (full mmap vs per-shard slicing), and validation of f16-wire +
TCP_NODELAY wins on real LAN-class RTT (untested on loopback).

### 10.3 Bare Metal / VPS

```bash
# Install
cargo install larql-server

# Run
larql-server /path/to/model.vindex --port 8080

# Systemd service
[Unit]
Description=LARQL Vindex Server
After=network.target

[Service]
ExecStart=/usr/local/bin/larql-server /data/gemma3-4b.vindex --port 8080
Restart=always
MemoryMax=8G

[Install]
WantedBy=multi-user.target
```

$5-20/month VPS. No GPU. No Python. No CUDA drivers.

---

## 11. Crate Structure

Source layout reflects the 2026-05-01 Q1 cleanup pass — see
`crates/larql-server/README.md → Crate Structure` for the canonical
tree. Highlights for spec readers:

- `main.rs` is a thin entry point (~26 LOC). All boot orchestration
  lives in `bootstrap/::serve(cli)` so the same code path can be
  driven from integration tests without going through clap.
- `env_flags.rs` is the single source of truth for `LARQL_*` knobs;
  every read goes through a cached accessor (`OnceLock`) and the
  README env-var table references the same names.
- `wire.rs::has_content_type(headers, expected)` is the shared
  helper used by every route that accepts both binary and JSON bodies
  (walk-ffn, embed, expert/batch).
- `routes/expert/` is split into seven files — `single.rs`,
  `batch_legacy.rs`, `layer_batch.rs`, `cpu.rs`, `metal.rs`,
  `warmup.rs`, plus a `mod.rs` that re-exports the historical public
  surface (`run_expert`, `run_experts_cpu_batch`, `handle_*`,
  `warmup_*`). `metal.rs` is `#[cfg(all(feature = "metal-experts", target_os = "macos"))]`.
- `http.rs` carries shared protocol constants:
  `BINARY_FFN_CONTENT_TYPE`, `JSON_CONTENT_TYPE`,
  `REQUEST_BODY_LIMIT_BYTES` (64 MB), `REQUEST_BODY_LIMIT_LARGE_BYTES`
  (256 MB; logits payloads), `BEARER_PREFIX`.

**Dependencies:** `larql-vindex`, `larql-inference` (for INFER),
`axum`, `axum-server` (rustls), `tokio`, `tonic` + `prost` (gRPC),
`tower` + `tower-http` (concurrency, CORS, tracing), `clap`.

---

## 12. Advanced Endpoints

### 12.1 WebSocket Streaming (implemented)

Layer-by-layer streaming for DESCRIBE via `WS /v1/stream`:

```
→ {"type": "describe", "entity": "France", "band": "all"}
← {"type": "layer", "layer": 14, "edges": []}
← {"type": "layer", "layer": 15, "edges": [{"target": "French", "gate_score": 35.2}]}
← {"type": "layer", "layer": 27, "edges": [{"relation": "capital", "target": "Paris", "gate_score": 1436.9, "source": "probe"}]}
← {"type": "done", "entity": "France", "total_edges": 6, "latency_ms": 12.3}
```

Edges include probe relation labels when available. Error messages: `{"type": "error", "message": "..."}`.

### 12.2 gRPC (implemented)

All endpoints available over gRPC via tonic + prost. Enable with `--grpc-port`:

```bash
larql serve gemma3-4b.vindex --port 8080 --grpc-port 50051
```

Proto definition: `proto/vindex.proto`. Services mirror the REST API: `Describe`, `Walk`, `Select`, `Infer`, `GetRelations`, `GetStats`, `WalkFfn`, `Health`. Includes `StreamDescribe` for server-streaming layer-by-layer results.

gRPC runs alongside HTTP — both ports active simultaneously. Uses bundled protoc (no system install required).

### 12.3 Edge Caching (implemented)

DESCRIBE responses include ETag and Cache-Control headers for CDN edge caching:

```
Cache-Control: public, max-age=86400
ETag: "a1b2c3d4"
```

Clients can send `If-None-Match` to receive `304 Not Modified` when the response hasn't changed. Combined with `--cache-ttl` for server-side caching.

### 12.4 Decoupled Inference Protocol (implemented)

`POST /v1/walk-ffn` has two modes.

#### Features-only mode (default)

Client POSTs a `[hidden_size]` residual; server runs gate KNN and returns feature indices + scores. The client still needs `up_features.bin` + `down_features.bin` locally to compute the FFN output.

```json
POST /v1/walk-ffn
{"layer": 26, "residual": [0.12, -0.34, ...], "top_k": 8092}
→ {"layer": 26, "features": [9515, 4532, ...], "scores": [1436.9, 26.1, ...], "latency_ms": 0.01}
```

**Batched:**
```json
{"layers": [0, 1, ..., 33], "residual": [...]}
→ {"results": [{"layer": 0, "features": [...], "scores": [...]}, ...], "latency_ms": 0.3}
```

#### Full-output mode (`"full_output": true`)

Client POSTs a `[seq_len × hidden_size]` row-major residual; server runs the architecture-correct WalkFfn path (gate KNN → activation → up gather → down projection) and returns the FFN output for each layer. This is what `--ffn-remote` uses — the server holds all FFN weights, the client holds only attention.

```json
POST /v1/walk-ffn
{"layer": 5, "residual": [...], "seq_len": 1, "full_output": true}
→ {"layer": 5, "output": [...], "seq_len": 1, "latency_ms": 8.1}
```

**Batched** (all layers in one round-trip):
```json
{"layers": [0, 1, ..., 33], "residual": [...], "seq_len": 1, "full_output": true}
→ {"results": [{"layer": 0, "output": [...], "seq_len": 1}, ...], "latency_ms": 8.3}
```

Validates that `residual.len() == seq_len * hidden_size`.

#### Binary wire format (`Content-Type: application/x-larql-ffn`)

Full-output mode also accepts a compact binary encoding that eliminates JSON float serialization overhead (~0.5 ms/hop on Gemma 3 4B). The `RemoteWalkBackend` client uses binary by default.

```
Request single layer:  [layer u32 LE][seq_len u32][flags u32 bit0=1][top_k u32][residual f32[]]
Request batch:         [0xFFFFFFFF][num_layers u32][layer u32[]...][seq_len u32][flags u32][top_k u32][residual f32[]]
Response single layer: [layer u32 LE][seq_len u32][latency_ms f32][output f32[]]
Response batch:        [0xFFFFFFFF][num_results u32][latency_ms f32] + per result: [layer u32][seq_len u32][num_floats u32][output f32[]]
```

Binary requires `full_output = true`. Features-only binary is rejected with HTTP 400. Full format spec: `docs/ffn/distributed.md` and `docs/specs/larql-router-spec.md §8`.

---

## 13. FFN-Service Mode and Layer Sharding

### 13.1 FFN-Service Mode (`--ffn-only`)

Run the server as a dedicated FFN backend. The client holds attention weights locally;
this server holds only the FFN weights and answers `/v1/walk-ffn` requests.

```bash
larql-server output/gemma3-4b-q4k.vindex --ffn-only --port 8080
```

**What changes under `--ffn-only`:**

- `/v1/infer` is disabled (`infer_disabled = true`)
- `/v1/stats` advertises `"mode": "ffn-service"` — `RemoteWalkBackend` on the client
  uses this to confirm it has connected to the right endpoint
- f16→f32 gate warmup is skipped; gate decode is lazy per layer on first request
- Attention + embed + lm_head tensors are filtered from the weight manifest at load
  time (pre-mmap filter — no allocation for ~3.4 GB of f32 tensors on 4B, ~14 GB on 31B)

**Client usage:**

```bash
larql walk --ffn-remote http://server:8080 --predict \
    --prompt "The capital of France is"
```

The client runs attention locally and calls the server for every FFN layer.

---

### 13.2 Layer Sharding (`--layers`)

Restrict the server to a contiguous layer range. Only those layers are loaded into
memory; requests for other layers are rejected immediately.

```bash
larql-server output/gemma3-4b-q4k.vindex --ffn-only --layers 0-16  --port 8080
larql-server output/gemma3-4b-q4k.vindex --ffn-only --layers 17-33 --port 8081
```

`START-END` bounds are **inclusive**. A 34-layer model split into two shards:

| Shard | Layers | Approximate RSS |
|-------|--------|-----------------|
| A | 0–16 (17 layers) | ~50% of full vindex |
| B | 17–33 (17 layers) | ~50% of full vindex |

**Memory model:**

Out-of-range layers are never loaded into physical RAM:

- For Q4K vindexes (`synthesize_gate_from_q4k`): the anonymous mmap is sized only
  for owned layers. Unowned layers are skipped entirely during dequantisation.
- For demand-paged files (`gate_vectors.bin`, `interleaved_kquant.bin`): the OS-level
  mmap covers the full file (cheap — virtual address space only). `is_layer_owned()`
  guards every accessor before any byte is read, so out-of-range pages never fault in.

**Startup log:**

```
larql-server v0.1.0
Loading: output/gemma3-4b-q4k.vindex
  Layers: 0–16 (of 34)
  Model: google/gemma-3-4b-it (34 layers, 348160 features)
  Warmup: skipped (--ffn-only, lazy gate decode on first request)
  Mode: ffn-service (--ffn-only)
Listening: http://0.0.0.0:8080
```

**Out-of-range rejection:**

```json
POST /v1/walk-ffn {"layer": 20, "residual": [...]}
→ 400 {"error": "layer 20 not served by this shard (owned: 0–16)"}
```

**CLI options summary:**

| Flag | Description |
|------|-------------|
| `--ffn-only` | FFN-service mode; disables infer, skips warmup, filters attn weights |
| `--layers START-END` | Inclusive layer range to load and serve |
| `--max-gate-cache-layers N` | LRU cap on decoded f16 gate layers in memory (0 = unlimited) |

---

### 13.4 Expert Sharding (`--experts` / `--units`)

Restrict the server to a contiguous range of expert IDs within each MoE
layer (or fine-grained per-`(layer, expert)` ownership via `--units`).
Requires vindexes using the `per_layer` expert format (§5.12 of
`vindex-format-spec.md`). Implemented and production-tested on Gemma 4
26B-A4B as of 2026-05-01; see §4.5 for the wire formats and §10 for
fly.io / multi-host deployment notes (tracked as `F-FLY` in
`ROADMAP.md`).

```bash
larql-server gemma4-26b-a4b.vindex --experts 0-31  --port 8080
larql-server gemma4-26b-a4b.vindex --experts 32-63  --port 8081
larql-server gemma4-26b-a4b.vindex --experts 64-95  --port 8082
larql-server gemma4-26b-a4b.vindex --experts 96-127 --port 8083
```

`START-END` bounds are **inclusive**. Gemma 4 26B A4B (128 experts/layer) split four ways:

| Shard | Experts | RSS per layer file |
|-------|---------|-------------------|
| A | 0–31 (32 experts) | ~25% of layer file |
| B | 32–63 | ~25% |
| C | 64–95 | ~25% |
| D | 96–127 | ~25% |

**Memory model.**

Each `layer_L.experts` file is mmap'd in full (virtual address only — one `mmap()` syscall per file, no RSS). The OS faults in only pages that are actually read. For a shard owning experts 0–31, experts 32–127 are never read and never resident. `is_expert_owned(layer, expert)` is a bitmap lookup; out-of-range expert requests return HTTP 404 before touching any file data.

**Endpoint behaviour under `--experts`.**

`POST /v1/expert/{layer}/{expert_id}` accepts only expert IDs within the shard's range. All other expert IDs return 404 with:
```json
{"error": "expert 47 not owned by this shard (owns 0-31)"}
```

`GET /v1/stats` reports:
```json
{
  "mode": "expert-shard",
  "experts": "0-31",
  "layers": "all",
  "num_experts_owned": 32
}
```

**CLI flag summary.**

| Flag | Meaning |
|------|---------|
| `--experts START-END` | Expert ID range to load and serve (inclusive) |
| `--experts START-END --layers START-END` | Combined expert + layer range (for fine-grained grid shards) |

**Note:** `--experts` requires `ffn_layout: "per_layer"` in `index.json`. Starting a shard against a vindex without this field returns an error at startup.

---

### 13.3 Deployment with a Router

Layer-sharded servers are not meant to be addressed directly. Use `larql-router`
(see `larql-router-spec.md`) as a transparent proxy in front of them. The client
uses `--ffn-remote http://router:9090` and has no knowledge of the shard topology.

```
Client  →  larql-router:9090  →  shard-a:8080  (layers 0–16)
                               →  shard-b:8081  (layers 17–33)
```

The router health-checks each shard on startup and rejects requests for layers that
have no owning shard with a clear error before contacting any backend.

---

## License

Apache-2.0
