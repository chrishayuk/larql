# Roadmap — larql-server / larql-router

## Current state (as of 2026-05-07)

### 2026-05-07 — Wire format evolution + WebSocket streaming + Criterion benchmarks

GT1–GT4 and GT9 shipped. See `G-TRANSPORT` and `G-BENCH` sections below for full details.

**GT1 (f16 wire default)**: `application/x-larql-ffn-f16` content-type; Accept header
negotiation; `encode_binary_output_f16` in server; `decode_binary_single/batch_f16`
in client. Client sends `Accept: i8, f16, f32`; server honours f16 by default
(`LARQL_F16_WIRE_DISABLE` opt-out). 50% bandwidth reduction on all grid paths.

**GT2 (i8 residuals)**: `application/x-larql-ffn-i8`; per-position symmetric
quantisation (scale = max(|x|)/127, zero_point = 0); `encode_binary_output_i8` +
`decode_binary_single/batch_i8`. Opt-in via `LARQL_I8_WIRE=1`. 75% bandwidth reduction.

**GT3 (per-layer latency)**: `LayerLatency { layer, avg_ms, p99_ms }` added to
`HeartbeatMsg.layer_stats` and `ServerInfo.layer_stats` in grid.proto. Server collects
EMA + p99 ring-buffer per layer in `metrics::LayerLatencyTracker`; heartbeat sends
snapshot every 10s. Router stores per-layer latency in `ServerEntry.layer_latencies`
and prefers lowest `avg_ms` for the requested layer when routing replicas.

**GT4 (WebSocket generate)**: `WS /v1/stream` now supports `{"type":"generate",
"prompt":"...","max_tokens":N}` command. Streams `{"type":"token","text":"...","index":N}`
frames per token; emits `{"type":"done","tokens":N,"latency_ms":M}`. Client can send
`{"type":"cancel"}` to abort. Uses same `generate_streaming` engine as SSE chat completions.
SSE streaming on `POST /v1/chat/completions` (N0.1 slice 3) confirmed already wired.

**GT9 (Criterion benchmarks)**: `larql-inference/benches/wire_codec.rs` (encode/decode
throughput at h2560/h4096/h5120, seq1/32/256). `larql-router/benches/routing.rs`
(route/route_all/update_heartbeat/rebuild at 1/10/100 servers). `larql-router` now
has a `lib.rs` exposing `grid` for tests/benches. Makefile targets: `bench-wire`,
`bench-routing`, `bench-grid`, `bench-all`.

### 2026-05-01 — HTTP CPU-path optimisation session

End-to-end Gemma 4 26B-A4B grid jumped from ~17.7 → ~19.7 tok/s on
M3 Max with one local gRPC shard. New per-call wire format,
streaming-overlap default-on, UDS transport, TCP_NODELAY, f16 wire
opt-in. See `Completed` section below for the full per-change list.

### Inherited state (2026-04-26)

- Code quality pass complete: modularity refactor + magic string cleanup + test restructure (see Completed below).
- Follow-up review fixes complete: rate limiting no longer trusts
  `X-Forwarded-For` by default, route/path strings are centralized,
  server loader options are grouped, embed errors use the standard JSON
  error envelope, and server-local clippy allows were reduced.
- Test coverage: **74.2% line / 81.2% function** at the 2026-04-26
  baseline (478 tests). 2026-05-01 (post Q1 cleanup): **131 lib tests +
  37 integration files (~580 tests total), all green**.
- 2026-05-10 re-measurement (post REV1–REV5 fixes, full `--tests`
  run): **65.68% line / 72.18% function** across ~660 tests.
  Coverage drifted ~8 pts down vs the 2026-04-26 claim — partly
  because the upstream `larql-inference` / `larql-compute` API
  refactor temporarily broke `tests/test_expert_endpoint.rs` (now
  fixed) and the four `routes/expert/*` modules sit at 0% line
  coverage because they need a live grid harness to exercise. Per-
  file 90% floor + debt baselines now codified in
  `crates/larql-server/coverage-policy.json`; run
  `make larql-server-coverage-summary` to re-measure (added in this
  session — see CHANGELOG.md).
- Q1 code-quality cleanup (2026-05-01) shipped 9 of 10 items: 1044-LOC
  `routes/expert.rs` split into 7 focused files; 656-LOC `main.rs` reduced
  to 26 LOC with `bootstrap::serve(cli)` as the orchestration point; new
  `env_flags.rs` (single source of truth for `LARQL_*` knobs) and `wire.rs`
  (shared content-type detection); body-size / JSON-content-type / Cli
  default literals all lifted to typed consts. Q1.10 (stream.rs WebSocket
  state machine) deferred until N0.1 SSE infrastructure lands. See
  Completed → "2026-05-01 (continued) — Q1 code-quality cleanup".
- Server-local clippy was clean at the 2026-04-26 baseline with
  `cargo clippy -p larql-server --tests --no-deps -- -D warnings`,
  re-verified clean post-Q1 on 2026-05-01.
  The dependency-checking form still stops in `larql-vindex`; that is
  tracked outside this server-only pass.
- Examples and synthetic benchmarks checked on 2026-04-26 and re-verified
  2026-05-01 (post Q1 cleanup, re-validated): `server_demo`, `embed_demo`,
  `server_bench --release`, `bench_expert_server` (live MoE bench against
  `gemma4-26b-a4b-q4k.vindex`), `bench_embed_server` (live f16 mmap embed
  against `gemma3-4b-q4k-streaming.vindex`) all pass. Numbers within
  noise of pre-Q1 baselines — see Live perf snapshot below.
- Grid route-table checks are now covered by `cargo test -p larql-router`
  (20 tests, including 7 grid-state tests) plus server announce-envelope tests.
- 2-shard local grid validated end-to-end on Gemma 4 26B-A4B (30 layers,
  inclusive layer ranges 0-14 + 15-29).
- W2 feature-major down retrofittable in-place via
  `larql convert add-feature-major-down --input <vindex>` (1.12 s for
  30 layers, 152 MB output).
- Live W2 surface on `GET /v1/stats.q4k_ffn`:
  `{cache_slots, cache_bytes, feature_major_down}`.
- `--warmup-hnsw` flag eager-builds HNSW across owned layers at boot
  (~325 ms for 15-layer shards on Gemma 26B).
- Grid memory profile (per-shard, single-machine): **9.1 GB RSS**,
  6.7 GB MALLOC_LARGE (gate f32 cache), `down_features_q4k.bin`
  resident at 0 K (capability, not yet exercised on dense path).

## Live perf snapshot (M3 Max, 2-shard grid, 26B-A4B)

### Dense walk-ffn / gate-KNN path

| Operation | Cold | Warm |
|---|---|---|
| `walk-ffn` 1 layer (router) | 12.8 ms | **0.2–0.3 ms** |
| `walk-ffn` 6 layers fanout | — | **1.3 ms** |
| `walk-ffn` 12 layers fanout | 64 ms | 2.6 ms |
| `walk-ffn` 24 layers fanout | 75 ms | 5.0 ms |
| `walk-ffn` 30 layers (full) | 30 ms | **5.9 ms** |
| `walk` (gate KNN, 30L) | — | 8.4 ms |
| 8-way concurrent × 15L fan-out | 112 ms wall | ~1070 layer-evals/sec |

P99 under 8-way contention: 24 ms.

### Remote MoE expert path (Gemma 4 26B-A4B, single in-process shard, layer 15, top-K=8)

`bench_expert_server` against per-layer Q4_K vindex
(`output/gemma4-26b-a4b-q4k.vindex`). Hidden=2816, 128 experts,
moe_intermediate=704, 30 MoE layers.

**bench numbers (2026-05-01, re-validated post Q1 cleanup; same hardware,
same vindex, same kernel path — confirms the refactor is bit-exact):**

| Operation | Result | (vs 2026-05-01 pre-Q1) |
|---|---|---|
| Vindex load | 5.4 s, +6.0 GB RSS | 5.2 s, +6.0 GB RSS |
| Lazy `get_or_load_weights()` | 1.36 s, +2.85 GB RSS | 1.3 s, +2.8 GB |
| Per-expert bytes (one bench layer, all 128) | 285 MB gate_up + 156 MB down (Q4_K) | unchanged |
| `forward_moe` warm (router + layer-batch HTTP + combine) | **0.78 ms** mean / 0.78 p50 / 0.88 p99 | 0.80 / 0.79 / 1.09 |
| `cpu_moe_forward` floor (no HTTP, same weights) | **0.34 ms** mean / 0.35 p50 / 0.43 p99 | 0.37 / 0.37 / 0.49 |
| 30-layer sweep (1 decode-step's worth of MoE blocks) | **23.24 ms** (0.77 ms/layer) | 24.8 ms (0.83 ms/layer) |
| Steady RSS | **10.5 GB** | 10.5 GB |

The 2-3% delta between pre- and post-cleanup runs is hardware noise (M3
Max thermal state varies 1-3% across runs) — the refactor moved code
across files but did not change any kernel.

**End-to-end Gemma 4 26B-A4B grid generation (`larql run --moe-shards`,
M3 Max, single local shard, 100-token poem, 3-run avg)**:

| Mode | tok/s |
|---|---|
| HTTP unary (`http://...` shard) | **17.8** |
| gRPC unary (`grpc://...` + `LARQL_MOE_NO_SPLIT=1`) | 17.7 |
| **gRPC + SPLIT overlap (default for gRPC)** | **19.7** |
| UDS HTTP/1.1 (`unix:///path` shard) | 18.2 |
| UDS + f16 wire (`LARQL_MOE_WIRE_F16=1`) | 20.5 (warm); within noise vs UDS f32 |

**Per-call HTTP overhead (loopback, post TCP_NODELAY)**:

| Stage | TCP HTTP | UDS HTTP | gRPC streaming |
|---|---|---|---|
| Server compute (run_experts_cpu_batch) | ~400 µs | ~400 µs | ~400 µs |
| spawn_blocking transition | ~25 µs | ~25 µs | ~25 µs |
| Transport RTT + axum dispatch | ~100 µs | ~50 µs | ~30 µs (multiplexed) |
| Encode + decode | ~5 µs | ~5 µs | ~5 µs (binary protobuf) |
| **Total per-call** | **~660 µs** | **~510 µs** | **~460 µs** |

For comparison, the historical baseline before any of this session's work
was 4.86 ms `forward_moe` warm and 16.6 GB steady RSS on the BF16
monolith (per-expert refactor + Q4_K migration cut that to 1.91 ms / 9.7
GB at 2026-04-26). The 2026-05-01 session took 1.91 ms → 0.78 ms
(another 2.4×) on the same per-call measurement, 56 ms → 23.24 ms
(2.4×) on the 30-layer sweep, and end-to-end ~17.7 → ~19.7 tok/s
(+12%) on the production grid. Cumulative session-on-session win is
**8.6× from the 2.3 tok/s pre-Q4K baseline** (see
`larql-inference/ROADMAP.md → M-CPU-1..6`).

### Embed-service path (Gemma 3 4B, ADR-0008 f16 mmap)

`bench_embed_server` against `gemma3-4b-q4k-streaming.vindex` (262144 ×
2560 vocab × hidden, ~1.34 GB f16 embeddings.bin):

| Operation | Result |
|---|---|
| mmap open (cold, no faults) | 0 ms, RSS 280 MB |
| L1 cache fill (5000 hottest tokens) | 25.2 ms, RSS 426 MB |
| f16 embed 1 token — L1 hit | **4.3 ns/op** (232 M ops/s) |
| f16 embed 1 token — mmap decode (L1 miss) | 3.22 µs/op (310 K ops/s) |
| f16 embed 32 tokens (prefill, mmap decode) | 59.07 µs/op |
| f16 embed 128 tokens (prefill, mmap decode) | 239.18 µs/op |
| f16 embed 512 tokens (prefill, mmap decode) | 1.10 ms/op |
| Logits projection (262208 × 2560, full vocab, CPU) | 335.6 ms (Metal: ~0.67 ms) |

Memory comparison (`--embed-only`, ADR-0008):

| Layout | RSS |
|---|---|
| f32 heap eager decode | ~2.9 GB |
| **f16 mmap + L1 cache (5000 tokens)** | **~1.6 GB** (48% reduction) |

---

## Great new functionality (next big-ticket items)

The numbered F0..F23 items below are mostly **incremental polish**
(metrics, shutdown drain, RBAC, OpenAPI, etc.) — necessary but not
load-bearing for new use cases. The items in this section are
**new capabilities** that would unlock production deployment shapes
the server can't currently serve. Ranked by how much they expand
the addressable surface, not by implementation effort.

### N0. OpenAI API compatibility (Chat Completions, Completions, Responses, Embeddings)

**Status**: **Slices 1 + 2 shipped 2026-05-02** — `/v1/models`,
`/v1/embeddings`, `/v1/completions`, `/v1/chat/completions` (all
non-streaming) live and OpenAI-shape-conformant on `larql-server`.
Live-validated against `output/gemma3-4b-q4k-streaming.vindex`. Chat
templates auto-detected from `arch.family()` (Gemma / Llama / ChatML
/ Mistral / Plain).

Slice 3 (SSE streaming on completions + chat completions) + slice 4
(tools / JSON mode / `response_format: json_schema`) + slice 5
(Responses API) remain; per-item **Status** lines below.

Supersedes the older F10 ("OpenAI-compat `/v1/chat/completions`")
which scoped only the chat endpoint shallowly. **Highest-leverage
item in this section** — every existing OpenAI client (Python `openai`
SDK, JS `openai`, LangChain, LlamaIndex, Cursor, Continue, Aider, eval
harnesses, dashboards) becomes a larql client the day all slices ship.
With slices 1+2 every chat client today already works; slices 3+4 add
the polish (streaming, tools, structured output).

**Router-side parity (N0-router)**: `larql-router` should also serve
the OpenAI surface so clients can hit the grid as a single endpoint
and the router fans out to shards. `/v1/models` aggregates from
registered shards; `/v1/embeddings`, `/v1/completions`, and
`/v1/chat/completions` proxy to shards owning the relevant compute.
Tracked under "Router-side OpenAI surface" in P1.

**Scope** — five endpoints, mapped onto our existing inference path:

#### N0.1 `POST /v1/chat/completions` (Chat Completions API)

**Status**: Slice 2 shipped 2026-05-02 (non-streaming). Live-validated
against `output/gemma3-4b-q4k-streaming.vindex`. Wire conforms to the
OpenAI shape; chat templates auto-detected from `arch.family()` (Gemma
/ Llama / ChatML / Mistral) with id-string fallback and a Plain
template for unknown / non-instruct models. SSE streaming → slice 3.
`tools` / `tool_choice` / `response_format: json_*` → slice 4 (returns
400 with a clear "see ROADMAP" message). `n>1` → 400.

Original spec preserved below for context on the streaming + tools work
that remains.

```
Request:  {model, messages: [{role, content, tool_calls?, tool_call_id?}],
           temperature?, top_p?, max_tokens?, stream?, tools?, tool_choice?,
           response_format?, seed?, stop?, n?, frequency_penalty?,
           presence_penalty?, logprobs?, top_logprobs?, user?}
Response: {id, object: "chat.completion", created, model,
           choices: [{index, message: {role: "assistant", content,
                       tool_calls?}, finish_reason, logprobs?}],
           usage: {prompt_tokens, completion_tokens, total_tokens}}
SSE chunk: {id, object: "chat.completion.chunk", created, model,
            choices: [{index, delta: {role?, content?, tool_calls?},
                       finish_reason?}]}
SSE terminator: `data: [DONE]\n\n`
```

Translation layer:
- `messages` → render via existing `chat::render_user_prompt` (per
  family chat template) → `encode_prompt` → `generate_streaming`.
- `stream: true` → wrap `generate_streaming`'s `on_token` callback in
  an SSE encoder; emit one chunk per token.
- `tools` → constrained-decoding mask routing the model toward valid
  tool-call JSON. Depends on N0.6 (JSON schema → grammar).
- `response_format: {type: "json_object"}` or
  `response_format: {type: "json_schema", json_schema: {...}}` → same
  constrained-decoding hook.
- `stop` strings → augment the existing `EosConfig` for the duration
  of the call.
- `seed` → pass through to `SamplingConfig` (already supported).

#### N0.2 `POST /v1/completions` (Legacy Completions API)

Older but still widely used (especially by older eval harnesses and
embedding/reranker pipelines that haven't migrated). Simpler shape:

```
Request:  {model, prompt: string | string[], max_tokens?, temperature?,
           top_p?, stream?, logprobs?, echo?, stop?, n?, best_of?,
           seed?, suffix?}
Response: {id, object: "text_completion", created, model,
           choices: [{text, index, finish_reason, logprobs?}],
           usage: {...}}
```

Strictly easier than N0.1 — no chat template, no tool calls, no
multi-message context. Maps directly to `encode_prompt` +
`generate_streaming`. Could ship first as a smoke-test of the
overall translation layer.

#### N0.3 `POST /v1/responses` (Responses API — newer, stateful)

OpenAI's 2025 successor to chat completions. Designed for stateful
multi-turn agents with built-in tool execution + reasoning content.
Pairs naturally with **N1 (stateful chat sessions)** — the
`previous_response_id` field references prior turns whose KV-cache
the server kept resident.

```
Request:  {model, input: string | InputItem[], previous_response_id?,
           instructions?, tools?, tool_choice?, response_format?,
           reasoning?, store?, metadata?, parallel_tool_calls?}

InputItem variants: text input ({type: "message", role, content}),
                    function-call output ({type: "function_call_output",
                    call_id, output}), file references, etc.

Response: {id, object: "response", created_at, status: "completed"|...,
           model, output: [
             {type: "message", role: "assistant", content: [{type: "output_text", text}]},
             {type: "function_call", call_id, name, arguments},
             {type: "reasoning", content},  // for o1 / DeepSeek-R1 style models
           ],
           usage: {input_tokens, output_tokens, reasoning_tokens, total_tokens},
           previous_response_id}
```

Implementation path:
- Without N1: each call is a fresh prefill (server-side response storage
  optional via `store: true` — return `id` for retrieval but don't
  reuse KV-cache).
- With N1: `previous_response_id` → look up the session's KV-cache,
  continue from that state (zero re-prefill on the prior turns).
- Reasoning content (DeepSeek-R1 / Gemma-thinking-style models): emit
  thinking traces as a separate `output[]` entry.

#### N0.4 `POST /v1/embeddings` (Embeddings API)

Existing `/v1/embed` endpoint already does this work; just needs an
OpenAI-shape wrapper.

```
Request:  {model, input: string | string[] | int[] | int[][],
           encoding_format?: "float" | "base64", dimensions?}
Response: {object: "list", data: [{object: "embedding", embedding: [...],
           index}], model, usage: {prompt_tokens, total_tokens}}
```

Two important nuances:
- `input` accepts strings (we tokenise) or pre-tokenised arrays
  (we embed directly via existing `/v1/embed`).
- `encoding_format: "base64"` returns embeddings as
  base64-encoded f32 little-endian bytes — ~33% smaller wire than
  the JSON float array form. Many production clients default to
  base64.

#### N0.5 `GET /v1/models` (already exists, needs OpenAI shape)

Current response shape doesn't match OpenAI's. Reshape:

```
{object: "list", data: [
   {id, object: "model", created, owned_by: "larql", parent?, ...}
]}
```

Trivial — existing route just needs the wrapper.

#### N0.6 Constrained decoding (JSON schema + GBNF grammar)

`response_format: {type: "json_schema"}` and `tools` both require
the decoder to emit only tokens that keep the output grammar-valid.
Today the inference-side decoder has a regex/grammar hook
(`EosConfig` / sampling pipeline already supports "stop strings");
need to extend with a real GBNF parser + JSON Schema → GBNF compiler.

Implementation is well-trodden — port from llama.cpp's `grammar.cpp` /
`grammar-parser.cpp` (well-defined spec; ~1000 LOC). Tracked
separately as F17 in this ROADMAP, but N0 makes it load-bearing.

#### Cross-cutting concerns

- **Streaming framing**: SSE format is `data: {json}\n\n` per chunk,
  terminated by `data: [DONE]\n\n`. axum has `axum::response::sse`
  out of the box.
- **Authentication**: the existing `--api-key` Bearer token mechanism
  works as-is; OpenAI clients send `Authorization: Bearer sk-...`.
- **Model identity**: `model` field in the request maps to a vindex
  ID. For single-model servers, ignore. For multi-model, route via
  the existing model-id mux.
- **Usage tokens**: track `prompt_tokens` (count from
  `encode_prompt`'s output) and `completion_tokens` (count tokens
  generated). Trivial bookkeeping.
- **Error envelope**: OpenAI uses `{error: {message, type, param,
  code}}` — slightly different from our `{error: "..."}`. Add an
  OpenAI-shape error mapper at the route layer.
- **Rate limit headers**: `x-ratelimit-limit-requests`,
  `x-ratelimit-remaining-requests`, etc. — pairs with our existing
  `--rate-limit` machinery.

#### Build order recommendation

1. **N0.5 + N0.4 + N0.2** (Models + Embeddings + Completions) —
   smallest, no streaming, validates the OpenAI shape + auth.
   Makes the server immediately usable for embedding-only and
   text-completion workloads.
2. **N0.1 non-streaming** (Chat Completions, no tools, no
   constrained output yet) — covers ~80% of real chat usage.
3. **N0.1 streaming** (SSE) — every chat UI assumes this.
4. **N0.6** (constrained decoding) — unblocks tools + structured
   output.
5. **N0.1 with tools + JSON mode** — production-grade chat.
6. **N0.3 (Responses API)** — pairs with N1 for stateful continuation.

#### Implementation surface (rough)

- N0.5: ~30 LOC (just a wrapper)
- N0.4: ~150 LOC (translate input format, base64 encoding)
- N0.2: ~250 LOC (legacy completions, simpler)
- N0.1 non-streaming: ~400 LOC
- N0.1 streaming SSE: +200 LOC
- N0.6 GBNF + JSON Schema: ~1200 LOC (port from llama.cpp)
- N0.1 with tools + JSON mode: +300 LOC (depends on N0.6)
- N0.3 Responses API (stateless): ~500 LOC
- N0.3 stateful (with N1): +200 LOC on top

**Total**: ~3200 LOC, shippable in slices. The first 5-day slice
(items 1-3 above) is enough to make larql-server a viable target for
most existing clients.

#### Files

New `routes/openai/` directory — one file per endpoint. Shared
`routes/openai/types.rs` for the request/response schemas (use
`serde` to match the OpenAI shape exactly; let serde-rename do the
heavy lifting for camelCase conversions). Wire into
`routes/mod.rs::single_model_router` alongside the existing routes;
multi-model routing via `model` field in the request body.

#### Why this beats every other N item on leverage

- N1 (sessions) is great but only useful if you have a client to use
  it with. **N0 brings every existing client.**
- N4 (multimodal) is an addressable-market expansion, not a
  client-acquisition unlock.
- N5 (federated knowledge graph) is unique but needs a custom
  client until OpenAI adds federated DESCRIBE to their spec (never).
- N0 is the move that makes everything else discoverable. Ship it
  first.

---

### N1. Stateful chat sessions (KV-cache as a first-class resource)

**Why this is the biggest gap.** Every production LLM API today is
session-aware: client sends the new turn, server remembers prior context
via KV-cache. larql-server's `/v1/infer` is single-shot — every request
re-prefills from scratch. For a 4 K context that's ~100 ms of wasted
compute per turn; for 16 K it's seconds. We're not competitive with
vLLM / TGI / OpenAI for any chat workload.

The pieces exist or are tracked piecemeal — F7 (KV-cache prefix
sharing), F22 (persistent patches as a precedent for session
persistence), the chat session machinery already in
`larql-inference::layer_graph::generate::chat_session` — but no
end-to-end story.

**Proposal**:
- `POST /v1/sessions` → returns `{session_id}` + initial state
- `POST /v1/sessions/{id}/append` → adds user message, generates assistant
  reply, returns SSE stream. KV-cache stays resident.
- `GET /v1/sessions/{id}` → describes current state (msg count, token
  count, model, adapter, last activity).
- `DELETE /v1/sessions/{id}` → frees KV-cache.
- Eviction policy: per-session TTL, total-RSS budget, LRU under
  pressure. Surfaces in `/v1/stats.sessions`.
- Pairs with **N3 (LoRA hot-load)** — sessions can pin a specific adapter.

**Implementation surface**: ~600 LOC. New `routes/sessions.rs`,
new `state::SessionStore`, hook into the existing `generate_streaming`
+ `Detokenizer` machinery. Roughly half the work is the eviction /
budget management — non-trivial but well-scoped.

### N2. Asynchronous batch inference job queue

**Why**: Real-time chat is one model; **bulk inference** (RAG document
processing, embedding pre-compute, reranker scoring, evaluation
harnesses) is another. They have very different SLOs. A batch job
submitter doesn't care about per-token latency; it cares about
throughput, cost, and being able to run while the cluster is otherwise
idle. Today users have to wrap `/v1/infer` in their own retry/queue
glue.

**Proposal**:
- `POST /v1/jobs` → submit `{prompts: [...], model_id, params}` →
  returns `{job_id}`.
- `GET /v1/jobs/{id}` → status + partial results.
- `POST /v1/jobs/{id}/cancel`.
- Optional `webhook_url` in the submit body for completion callback.
- Worker pool: independent rayon thread pool, capped concurrency,
  prioritises real-time `/v1/infer` traffic (job worker yields when a
  real-time request arrives).
- Persistence: jobs survive restarts (write-ahead log to disk).

**Pairs with**: F12 (batched infer in same request), F22 (persistent
state). Together those two are the building blocks; this item is the
asynchronous wrapper.

**Implementation surface**: ~800 LOC. New `routes/jobs.rs`, new
`worker::Pool`, persistence to a `jobs/` directory. The hardest piece
is the priority scheduler — getting it wrong means batch starves
real-time or vice versa.

### N3. LoRA / adapter hot-loading per session

**Why**: Multi-tenant production. Today every tenant either gets the
same base model or has to spin up a separate process. Real production
serving (Anthropic, OpenAI, Together, Replicate) supports per-request
adapter swap. Adapters are 10-100 MB vs the 16 GB base model —
hot-loading hundreds of them is feasible if we have the surface.

**Proposal**:
- `POST /v1/adapters/load` → `{adapter_id, source: "hf://..."|"file://..."|"http://...",
  model_id}` → loads into RAM.
- `GET /v1/adapters` → list loaded adapters with size + last-used.
- `DELETE /v1/adapters/{id}` → evict.
- Inference / sessions take an optional `adapter_id` field — applies
  the LoRA delta to gate/up/down/q/k/v/o matmuls per layer per call.
- Eviction: LRU + total-RSS budget, configurable.

**Pairs with**: N1 (sessions pin adapters). Independent enough to ship
first if N1 is too heavy.

**Implementation surface**: ~500 LOC. The LoRA forward-pass plumbing
already exists at the inference-crate level (per
`larql-inference/ROADMAP.md` § F4 LoRA loading). The server piece is
the lifecycle + RSS management.

### N4. Multimodal API surface (vision tower, mixed image+text infer)

**Why**: Gemma 3/4 ships vision variants; Llama 3.2 too. The vindex
extractor already handles vision tower weights (per
`larql-inference/ROADMAP.md → vision`). We're missing the API
surface — there's no way to send an image to the server today.

**Proposal**:
- `POST /v1/embed/image` → multipart upload → vision tower forward →
  returns `{embedding: [...], hidden_size}`.
- `POST /v1/infer` accepts `images: [base64, ...]` field; server
  routes through the vision tower then concatenates with text tokens
  for the language decoder.
- `POST /v1/sessions/{id}/append` accepts images for multimodal chat.

**Implementation surface**: ~400 LOC server-side once the inference
crate's vision forward path is exposed (currently tracked separately).
Big use-case unlock: docVQA, ChartQA, image classification, image
embedding service.

### N5. Federated knowledge graph over multiple vindexes

**Why**: The DESCRIBE/WALK/SELECT trio makes a vindex a queryable
knowledge graph. Multi-model serving (`--dir`) puts multiple
graphs side-by-side — but each is queried independently. There's no
way to ask "describe France using Gemma's knowledge AND Llama's
knowledge AND my custom vindex". This is a unique capability the
larql architecture enables that nothing else (vLLM, TGI, OpenAI) can
do, and it's invisible.

**Proposal**:
- `GET /v1/federated/describe?entity=X&models=gemma,llama,custom` →
  merges edges across vindexes, sourcing each edge with its origin
  model.
- `POST /v1/federated/select` with cross-model joins ("entities
  Gemma calls capitals AND Llama calls capitals").
- New LQL syntax: `DESCRIBE "France" USING gemma, llama;` already
  hinted in the REPL doc (`USE REMOTE`); the server-side surface is
  the missing half.
- Surfacing model disagreement is a research-grade capability:
  "Gemma says Paris is the capital of France with score 1436;
  Llama says Lyon with score 320. Confidence-weighted merge?"

**Implementation surface**: ~600 LOC. New `routes/federated.rs`,
extends multi-model serving to do cross-model fan-out + merge.

### N6. Live blue-green vindex deployment

**Why**: Production model rollouts. Today swapping a vindex requires
restart (modulo F8 hot-swap, which is admin-only and atomic). True
blue-green wants: load v2 alongside v1, route X% of traffic, observe
metric drift, ramp or rollback.

**Proposal**:
- `POST /v1/admin/deploy` → load `v2.vindex` alongside the active
  `v1.vindex`, returns `{green_id}`.
- `POST /v1/admin/traffic` → set weighted routing
  (`{"v1": 0.9, "v2": 0.1}`).
- `GET /v1/stats.deployment` → per-vindex per-endpoint p50/p99/error
  rate side-by-side. Pairs with F3 metrics.
- `POST /v1/admin/promote/{id}` → atomically swap routing to 100%
  green; old vindex becomes stale-evictable.

**Pairs with**: F8 (admin endpoints), F3 (metrics for traffic
comparison). N6 is the **product** built on top of those primitives.

**Implementation surface**: ~700 LOC. New `routes/admin/deploy.rs`,
extends `AppState` to hold multiple model versions, weighted routing
logic in the request entry points.

---

## REV: Code review (2026-05-10)

Lens: per `THESIS.md`, legibility is a primary feature — a vLLM/SGLang/TGI
engineer should be able to read this and copy ideas out. Findings below are
prioritised by what would damage that primary goal if a stranger landed in
the file today. Severity tags: **P0** = correctness/security ship-blocker,
**P1** = structural/legibility fixes that affect "reads like a citation",
**P2** = defensive polish.

### REV1. Panic on NaN in gRPC sort *(P0)*

**Status**: ✅ **Shipped 2026-05-10.**

Both sort sites in `src/grpc.rs` (the `describe` handler at line ~311 and the
`select` handler at line ~432) used `partial_cmp(...).unwrap()`, which
panics on NaN. Replaced with a shared `cmp_score_desc(a, b)` helper that
returns `Ordering::Equal` on NaN, removing the panic path. New
`#[cfg(test)] mod tests` in `grpc.rs` covers: descending order, NaN→Equal,
sort-with-NaN no-panic + length-preservation, sort-with-infinities
ordering, and all-finite descending sort. Five new unit tests, all green;
`cargo test -p larql-server --test test_grpc` (28 integration tests) still
green.

**Files touched**: `src/grpc.rs` (one new helper, two call sites, one new
test module).

**Note on the strict-weak-ordering trade-off**: NaN-as-Equal violates strict
weak ordering, so `sort_by` cannot guarantee that finite values around
NaN are globally descending — only that the call doesn't panic and that
finite/NaN counts are preserved. Acceptable for a defensive change against
upstream data corruption; downstream `truncate(limit)` still picks
finite values when present, which is the property production cared about.

### REV2. Non-constant-time API key comparison *(P0 security)*

**Status**: ✅ **Shipped 2026-05-10.**

`auth.rs` now hashes both the provided token and the configured key with
SHA-256 and compares the 32-byte digests via `subtle::ConstantTimeEq`.
The hash step removes the length-leak; the constant-time compare removes
the bytewise short-circuit leak. Module-level doc comment names the
threat model so the next reader doesn't accidentally regress to `==`.

**Files touched**:
- `Cargo.toml`: added `subtle = "2.6"` as a direct dep (already in the
  lockfile via rustls — declaring it directly makes the threat-model
  rationale visible at the crate boundary).
- `src/auth.rs`: rewritten with module docs, a private
  `tokens_match(provided, expected) -> bool` helper, and a
  `#[cfg(test)] mod tests` covering equal/unequal/empty/length-different/
  single-byte-difference cases (6 unit tests).

**Verification**:
- 6 new auth unit tests pass.
- All 6 existing `http_auth_*` integration tests in `test_http_core.rs`
  still pass (no behavioural regression).
- `cargo clippy -p larql-server --lib --no-deps -- -D warnings` clean.
- `cargo fmt -p larql-server -- --check` clean.

### REV3. `blocking_read` on tokio RwLock inside async path *(P0)*

**Status**: ✅ **Shipped 2026-05-10.**

`apply_patch` is now structured fast-path / slow-path. The fast path
(session already exists) takes one write guard, mutates, returns. The
slow path drops the sessions write guard, awaits `model.patched.read()`
to get the base, then re-acquires the sessions write guard and uses
`entry().or_insert_with(...)` to absorb the race where another task
inserted the same `session_id` between our drop and re-acquire. No
`blocking_read`/`blocking_write` is reachable from an `async fn` in
`session.rs` anymore (the remaining `sessions_blocking_write` helper is
explicitly intended for `spawn_blocking` callers, documented as such).

A module-level lock-discipline doc-block on `apply_patch` names the
hazard so the next reader doesn't reintroduce it.

**Files touched**:
- `src/session.rs`: `apply_patch` rewritten with fast/slow split + doc
  block.
- `tests/test_http_session.rs`: existing tests' workaround
  (`sm.get_or_create(...)` pre-create dance) removed since the slow path
  is now safe; two new regression tests added:
  - `apply_patch_slow_path_makes_progress_with_held_patched_reader` —
    spawns a task holding `model.patched.read()`, asserts the slow-path
    apply_patch finishes within 5s.
  - `concurrent_apply_patch_same_session_finishes` — 16-way
    `apply_patch("contended", …)` from spawned tasks, asserts all
    finish within 5s and the final patch list has 16 entries.

**Verification**:
- 7/7 session integration tests pass (5 existing, post-cleanup, plus 2
  new).
- 232/232 lib tests pass.
- `cargo clippy -p larql-server --lib --no-deps -- -D warnings` clean.
- `cargo fmt -p larql-server -- --check` clean.

**Note**: `get_or_create` (line ~56) also nests `sessions.write().await`
→ `model.patched.read().await`, but uses async-aware `read().await`
(not `blocking_read`), so it doesn't block a worker. The lock-ordering
hazard (deadlock if anywhere acquires `patched.write` then
`sessions.*`) is not reachable today — the only `patched.write()` call
sites in `routes/patches.rs` don't touch `sessions`. Documented in the
new doc-block on `apply_patch` so future contributors keep the order
consistent.

### REV4. OpenAI error envelope diverges from spec *(P0 — breaks OpenAI SDKs)*

**Status**: ✅ **Edits applied 2026-05-10; verification pending stable workspace build.**

Two-envelope split, per the original recommendation:

- **LARQL paradigm endpoints** keep the flat `{"error": "msg"}` shape via
  the existing `crate::error::ServerError` (unchanged).
- **OpenAI-compat endpoints** (`/v1/embeddings`, `/v1/completions`,
  `/v1/chat/completions`) now use a new `OpenAIError` type that renders
  the canonical nested envelope:
  ```json
  {"error": {"message": "...", "type": "invalid_request_error",
             "param": null, "code": null}}
  ```
  with the OpenAI-canonical `type` strings (`invalid_request_error`,
  `not_found_error`, `service_unavailable_error`, `server_error`).
  `param` and `code` are always emitted (possibly `null`) since some SDKs
  hard-key on those fields.

**Files touched**:
- New `src/routes/openai/error.rs` — `OpenAIError`, `OpenAIErrorBody`,
  `OpenAIErrorPayload`, constructor helpers
  (`invalid_request`/`not_found`/`service_unavailable`/`server_error`),
  `From<ServerError>` impl, `IntoResponse`, 7 unit tests covering all
  status classes, the nested-envelope shape, the From mapping, and the
  always-present `param`/`code` keys.
- `src/routes/openai/mod.rs` — registers the module + re-exports
  `OpenAIError`.
- `src/routes/openai/chat.rs` — entry-point return type
  `Result<Response, ServerError>` → `Result<Response, OpenAIError>`.
  All 8 direct-`return Err(ServerError::X(...))` sites in the entry
  handler converted to `OpenAIError::invalid_request(...)` or
  `service_unavailable(...)`. Internal helpers
  (`run_chat_generation`, `resolve_tools`,
  `schema_for_response_format`) keep `ServerError`; their results
  propagate via `?` through `From<ServerError> for OpenAIError`.
- `src/routes/openai/completions.rs` — same pattern: 5 entry-point
  error sites converted; internal helpers unchanged.
- `src/routes/openai/embeddings.rs` — same pattern: 3 entry-point sites
  converted.
- `src/openapi.rs` — registers `OpenAIErrorBody`/`OpenAIErrorPayload`
  in the `components(schemas(...))` block; the three OpenAI handlers'
  utoipa annotations now reference `OpenAIErrorBody` for 400/500
  responses (LARQL endpoints keep `ErrorBody`).
- `docs/server-spec.md` §8.3.1 rewritten to document the two-envelope
  split with the canonical `type` table.
- `tests/test_http_embed.rs` — added 6 integration tests
  (`http_openai_embeddings_400_uses_nested_envelope`,
  `_empty_uses_nested_envelope`,
  `http_openai_completions_400_uses_nested_envelope`,
  `_503_uses_nested_envelope`,
  `http_openai_chat_completions_400_uses_nested_envelope`,
  `_503_uses_nested_envelope`) using a shared
  `assert_openai_error_envelope` helper that asserts the response body
  has the nested shape with the expected `type` and the always-present
  `param`/`code` keys.

**Verification**:
- `cargo test -p larql-server --lib openai::error` — 6/6 passed.
- `cargo test -p larql-server --test test_http_embed _uses_nested_envelope`
  — 6/6 passed (covers the three handlers × 400/503 cases).
- `cargo test -p larql-server --test test_http_embed http_openai`
  — 55/55 passed (no regression in the existing OpenAI surface).
- `cargo test -p larql-server --lib` — 247/247 passed.
- `cargo clippy -p larql-server --lib --no-deps -- -D warnings` clean.
- `cargo fmt -p larql-server -- --check` clean.

### REV5. Tool-call JSON parsing via `find('{')` / `rfind('}')` returns 500 instead of 400 *(P0 legibility)*

**Status**: ✅ **Shipped 2026-05-10.**

`build_tool_call_message` rewritten as a straight-line
`serde_json::from_str(text.trim())`. The `find('{')` / `rfind('}')`
heuristic is gone — it was supposed to absorb model-output drift
(trailing junk, multiple JSON objects, markdown wrappers) but in
practice silently picked the wrong slice and surfaced the failure as
500 Internal at the call site. The new parser fails fast with a clean
`invalid JSON: …` diagnostic, distinguishes "not an object" from
"missing field", and the call site flips from `ServerError::Internal`
→ `OpenAIError::invalid_request` so the client now sees a
**400 invalid_request_error** with a concrete message (and the raw
output included for debugging) instead of a 500 server_error.

The schema FSM in `routes/openai/schema/` is for *constraining*
generation, not validating it post-hoc. Re-parsing with the FSM would
duplicate the Schema-shaped state machine that already runs during
generation. The straight-line `serde_json` parse is the right level
of validation for the post-emit path.

**Files touched**:
- `src/routes/openai/chat.rs::build_tool_call_message` — rewritten;
  added `json_value_kind` helper for clean error messages.
- `src/routes/openai/chat.rs::handle_chat_completions` (line ~377) —
  parse-failure error switched from `ServerError::Internal` to
  `OpenAIError::invalid_request`.
- `src/routes/openai/chat.rs` — added `#[derive(Debug)]` to
  `ChatChoiceMessage`, `ToolCall`, `ToolCallFunction` to support
  `Result::unwrap_err()` in tests.
- `src/routes/openai/chat.rs` `#[cfg(test)] mod tests` — 9 new unit
  tests:
  - happy path
  - tolerates surrounding whitespace
  - handles nested braces in arguments (locks the property the old
    code was trying to preserve)
  - rejects trailing junk with a clean `invalid JSON:` prefix
    (pre-REV5 this was the 500 trap)
  - rejects empty input
  - rejects non-object top-level (with kind reported)
  - rejects missing `name`
  - rejects missing `arguments`
  - rejects invalid JSON

**Verification**:
- `cargo test -p larql-server --lib build_tool_call` — 9/9 passed.
- `cargo test -p larql-server --lib` — 247/247 passed.
- `cargo clippy -p larql-server --lib --no-deps -- -D warnings` clean.
- `cargo fmt -p larql-server -- --check` clean.

**Note on the streaming path** (chat.rs ~line 586): mid-stream
parse-failures still emit an SSE `data: {"error": ...}` chunk via
`error_chunk(...)` and let the stream terminate gracefully. That path
already returned the OpenAI nested error shape in chunks, so no
behaviour change there — the fix here is only on the buffered
non-streaming path.

---

### REV6. Split `bootstrap.rs` (1279 lines) *(P1 legibility)*

**Files**: `src/bootstrap.rs` → new `src/bootstrap/{parsing,loader,warmup}.rs`.

Natural seams:
- `bootstrap/parsing.rs` (~lines 38–141): `parse_ram_bytes`, `parse_layer_range`, `UnitManifest` and friends.
- `bootstrap/loader.rs` (~lines 143–370): `load_single_vindex`, `discover_vindexes`, vindex selection logic.
- `bootstrap/warmup.rs`: the three warmup blocks (walk-ffn, hnsw-units, metal-experts).
- Remaining `bootstrap.rs`: `Cli`, `serve()` orchestration, listener setup, gRPC + announce spawn — should land at ≤ ~500 lines.

**Acceptance**: `serve()` reads top-to-bottom in one screen. Per-file
coverage stays ≥ 90% (project floor). No public API change.

### REV7. Split `routes/walk_ffn.rs` (1347 lines) *(P1 legibility)*

**Files**: `src/routes/walk_ffn.rs` → new `src/routes/walk_ffn/binary.rs`.

The binary codec (`decode_binary_request`, `encode_binary_output*`,
~lines 1–170 + helpers, ~240 LOC total) is orthogonal to the compute
core and HTTP handler logic. Move it out. Move the in-file `#[cfg(test)]`
block (~230 lines) into a sibling test module. Core compute + handlers
stay together (~600 LOC).

**Acceptance**: `walk_ffn.rs` ≤ ~700 LOC; binary codec testable in
isolation; existing integration tests unchanged.

### REV8. Split / de-duplicate `routes/openai/chat.rs` (1214 lines) *(P1 legibility)*

**Files**: `src/routes/openai/chat.rs`.

Streaming and non-streaming handlers duplicate setup (tokenization, model
lock, schema/sampling config). Extract a `ChatPrep` (or similar) struct so
each handler body is short and the difference between the two paths is
obvious at a glance. Tool-call/tools rendering and tool-call parsing
(~lines 822–998) belong in their own submodule.

**Acceptance**: `chat.rs` ≤ ~700 LOC; streaming and non-streaming entry
points fit on one screen each.

### REV9. gRPC error mapping is uniformly `Status::internal` *(P1)*

**Files**: `src/grpc.rs:99,115,134,157,175,194` (and similar `.map_err`
sites in `grpc_expert.rs`).

Every blocking-task failure coerces to `Status::internal`. Tokenization,
validation, and real internal failures collapse to one code; clients can't
distinguish recoverable from non-recoverable. Map at least
`ServerError::BadRequest`/`NotFound` → `invalid_argument`/`not_found`.

**Acceptance**: a single `From<ServerError> for tonic::Status` impl that
preserves status class. Test asserting a malformed `DescribeRequest`
returns `Code::InvalidArgument`, not `Code::Internal`.

### REV10. Single-vs-multi-model handler duplication *(P1 legibility)*

**Files**: `src/routes/describe.rs:279-314` (and the same shape in
`walk.rs`, `select.rs`, `infer.rs`, `relations.rs`, `patches.rs`).

Single-model and `/v1/{model_id}/...` handlers are copy/paste with one
line difference (`model_or_err(None)` vs `model_or_err(Some(&model_id))`).
A porter has to mentally diff every pair. Consolidate behind one handler
that takes `Option<&str>`, or behind a small macro.

**Acceptance**: each route has one handler body; the router still wires
both paths; tests cover both.

### REV11. Streaming client-disconnect leak (chat / completions) *(P1)*

**Files**: `src/routes/openai/chat.rs:517-520`,
`src/routes/openai/completions.rs:243`.

When the SSE channel closes, the on-token callback flips `early_stop` and
returns, but the `spawn_blocking` generation task continues to natural EOS
— burns a CPU-bound worker on a gone client. The expert layer-batch route
already models the right pattern (semaphore permit held across spawn,
released on cancellation); port it to the OpenAI streaming handlers.

**Acceptance**: a test that opens an SSE stream, drops the receiver, and
asserts the spawn_blocking handle finishes within a small bounded window
of the disconnect rather than running to completion.

---

### REV12. Float validation gaps (NaN/Inf) *(P2)*

**Files**: `src/routes/describe.rs:46` (`min_score`),
`src/routes/walk_ffn.rs` (residual deserialisation),
`src/routes/openai/util.rs:113-145` (`temperature`/`top_p` silent clamp).

Incoming `f32`s are not checked for finitude; OpenAI sampler params are
silently clamped rather than rejected. Reject NaN/Inf with 400; reject
out-of-range with a typed message instead of clamping silently.

### REV13. `max_tokens` upper bound unenforced *(P2)*

**Files**: `src/routes/openai/{chat,completions}.rs` request structs.

Raw `usize` is accepted; allocates token buffers from arbitrary client
input. Add a per-model cap (config-derivable from `LoadedModel`) and
return 400 above it.

### REV14. Cache key truncates float to u32 *(P2)*

**Files**: `src/cache.rs:34`.

`format!("{:x}", min_score as u32)` collides `5.2` and `5.7`. Either
encode the full f32 (e.g. `min_score.to_bits()`) or document the
intentional bucketing.

### REV15. Tests use bare `.unwrap()` on RPC results *(P2)*

**Files**: `tests/test_grpc.rs` (17 instances at lines 34, 43, 51, 68,
92, …).

When the RPC itself errors, `.unwrap()` panics on the wrong line and
hides which assertion was being checked. Replace with
`.expect("describe should succeed")` (or similar).

### REV16. `#[allow(dead_code)]` on library APIs *(P2)*

**Files**: `src/session.rs:55` (`get_or_create`), `:166` (`session_count`);
`src/ratelimit.rs:82` (`evict_stale`); `src/ffn_l2_cache.rs:92` (`stats`);
`src/error.rs:24` (`InferenceUnavailable`).

Either delete or add a one-line doc comment saying they're public for
out-of-tree consumers; right now a reader can't tell intent.

---

### REV-COVERAGE. Test coverage gaps vs 90%/file project floor *(P1, partly addressed)*

**Status**: 🟡 **Tooling + policy shipped 2026-05-10; per-file gap-fill ongoing.**

Done in this session:
- `make larql-server-{test,fmt-check,lint,coverage,coverage-summary,coverage-html,coverage-policy,ci}`
  added to the workspace Makefile (mirror the `larql-compute` /
  `larql-vindex` patterns).
- `crates/larql-server/coverage-policy.json` created. Default per-
  file floor is **90.0%**; 28 debt baselines snapshotted from the
  2026-05-10 measurement; `total_line_min_percent: 65.6` matches the
  current floor. New / split files automatically inherit the 90%
  default.
- Real coverage measured: 65.68% line / 72.18% function (was
  claiming 74.2% / 81.2% at 2026-04-26 — drift since then).
- Mainline files added in this session land at 100% (`routes/openai/error.rs`)
  or close to it (`auth.rs` 98.0%, `session.rs` 96.1%).

Still open (per-file gap-fill, listed roughly by impact):
- `routes/openai/completions.rs` 40.3% → 90% (REV8 split helps)
- `routes/walk_ffn.rs` 49.0% → 90% (REV7 split helps)
- `routes/openai/chat.rs` 53.4% → 90% (REV8 split helps)
- `routes/stream.rs` 53.3% → 90% (Q1.10 split helps)
- `routes/explain.rs` 44.8% → 90%
- `routes/infer.rs` 50.2% → 90%
- `routes/expert/{batch_legacy,multi_layer_batch,single,warmup}.rs`
  all 0% — these need a live-grid harness and are best treated as
  one ticket once the grid test fixture exists.
- `routes/openai/schema/mask.rs` 0% — orthogonal to the splits;
  needs unit tests on the FSM-mask behaviour.
- Behavioural-test gaps still open from the original review:
  malformed-JSON rejection (axum `JsonRejection` → 400),
  body-size-limit 413s, ETag 304 paths beyond the one
  `test_http_describe.rs` happy path, gRPC stream backpressure,
  gRPC client cancel mid-stream, OpenAI SSE `[DONE]` framing,
  OpenAI streaming error chunk shape.

**Acceptance**: each file ≥ 90% line coverage (project floor) — track
via `coverage-policy.json` ratchet; each behavioural gap above has at
least one direct test.

### REV-SPEC. Spec / OpenAPI drift *(P1)*

**Files**: `src/openapi.rs:113-118` vs `proto/vindex.proto:194-196`
(`LoadedCapabilities` differs between HTTP and gRPC); `proto/vindex.proto`
populates both `predictions` and `walk_predictions`/`dense_predictions`
in `InferResponse` regardless of mode (HTTP differentiates). Both are
intentional but undocumented and untested across both transports.
Separately, `docs/server-spec.md` does not mention 422 anywhere and
handlers do not emit it — pick a stance (probably 400 only) and state it.

**Acceptance**: a cross-transport contract test that exercises the same
operation over HTTP and gRPC and asserts equivalent semantics. A
paragraph in `docs/server-spec.md` documenting the shape differences
that remain.

---

### Strengths to preserve (do not regress)

- `ServerError` enum + `IntoResponse` discipline — no generic `Internal`
  catch-all in handler paths.
- Centralised `env_flags` with cached reads and README cross-reference —
  the right pattern for a reference implementation.
- Rate limiter degrades open on poisoned mutex; `X-Forwarded-For` only
  honoured under explicit flag.
- Expert `layer_batch` semaphore-as-backpressure — a clean illustration
  of the right way to bound rayon under HTTP load. Treat as a teaching
  artefact and consider citing in the README.
- Sampling clamping + stop-string handling in `routes/openai/util.rs` is
  well-tested and idiomatic.
- Binary wire format with `Accept`-header negotiation in `walk_ffn` is
  elegant; the f16/i8/f32 negotiation is the kind of concrete pattern the
  THESIS expects to diffuse into other stacks.

---

## P0: Active

### G-TRANSPORT. Wire format evolution + WebSocket streaming + QUIC (ADR-0009, ADR-0010)

All work here is architecture-agnostic: no hardcoded layer counts, hidden
sizes, or model-family assumptions. Sizes and dtypes are read from vindex
config at runtime.

#### GT1 — f16 wire default

**Status**: ✅ **Shipped 2026-05-07.**

Added `FFN_F16_CT = "application/x-larql-ffn-f16"` in `wire.rs`; `encode_binary_output_f16` in `walk_ffn.rs`; `preferred_response_ct` selects f16 when client sends `Accept: application/x-larql-ffn-f16`. Client (`ffn/remote/http.rs`) sends `Accept: i8, f16, f32` on every grid request. `LARQL_F16_WIRE_DISABLE` opt-out. `half = "2"` added to both crates.

**Spec**: ADR-0009 §Decision, §Wire Layout (f16).

Wire format is currently f32-only (4 bytes/value). For a model with
hidden_size=H and seq_len=1, one round-trip costs `H × 4 × 2` bytes
(request + response). f16 halves this with no accuracy loss for all tested
architectures.

- Add `F16_WIRE = "LARQL_F16_WIRE"` to `env_flags.rs` (present = opt-out,
  i.e. `LARQL_F16_WIRE=0` forces f32).
- Add `F16_CT = "application/x-larql-ffn-f16"` to `wire.rs`.
- In `routes/walk_ffn.rs`: inspect `Accept` header; if client sends
  `Accept: application/x-larql-ffn-f16`, encode response as f16.
- In `larql-inference/src/ffn/remote/http.rs`: set
  `Accept: application/x-larql-ffn-f16` by default (opt-out via flag).
- Accuracy gate: `larql bench <vindex> --wire f32,f16 --assert-topk-match 5`
  must pass for each model family before enabling as default.

**Acceptance**: `larql bench <vindex> --ffn URL --wire f32,f16` shows <1%
tok/s difference and identical top-5 tokens. Wire bytes column shows 50% reduction.

#### GT2 — i8 quantised residuals (opt-in)

**Status**: ✅ **Shipped 2026-05-07.**

Added `FFN_I8_CT`; `encode_binary_output_i8` (per-position symmetric scale, zero_point=0) in `walk_ffn.rs`; `decode_binary_single/batch_i8` in `codec.rs`. Client advertises i8 in Accept header; server honours when `LARQL_I8_WIRE=1`. `preferred_response_ct` checks i8 before f16.

**Spec**: ADR-0009 §Wire Layout (i8), §Negotiation Protocol.

Per-position symmetric quantisation: `scale = max(|x|)/127`, `zero_point = 0`.
Wire: `[scale f32 LE][zero_point f32 LE][data i8[] × hidden_size]` per position.

- Add `I8_WIRE = "LARQL_I8_WIRE"` to `env_flags.rs` (opt-in, default off).
- Add `I8_CT = "application/x-larql-ffn-i8"` to `wire.rs`.
- Add `encode_i8_request`, `decode_i8_single/batch` to `ffn/remote/codec.rs`.
- Add `encode_i8_output` to `routes/walk_ffn.rs`.
- Accuracy gate: `--wire f32,i8 --assert-topk-match 1` must pass before
  enabling i8 as opt-out on any model family.

**Acceptance**: 75% bandwidth reduction vs f32; top-1 token identical on
≥95% of decode steps across tested architectures.

#### GT3 — Per-layer latency in HeartbeatMsg

**Status**: ✅ **Shipped 2026-05-07.**

`LayerLatency { layer, avg_ms, p99_ms }` added to grid.proto (`HeartbeatMsg.layer_stats` + `ServerInfo.layer_stats`). New `metrics::LayerLatencyTracker` (EMA α=0.1, p99 ring-buffer per layer, thread-safe Mutex). `LoadedModel.layer_latency_tracker` populated at construction; `walk_ffn.rs` records timing per layer after each FFN forward. `announce.rs` heartbeat sender calls `tracker.snapshot()`. Router `grid.rs` stores `layer_latencies: HashMap<u32, (avg_ms, p99_ms)>` in `ServerEntry`; `route()` prefers lowest `avg_ms` for the requested layer.

**Spec**: ADR-0011 §HeartbeatMsg Extension.

Current heartbeat sends `cpu_pct`, `ram_used`, `requests_in_flight` — all
global. Router uses `requests_in_flight` for load balancing. This is blind to
per-layer compute bottlenecks (e.g. a sparse MoE model where layer 15 is 3×
slower than others due to expert placement).

Proto change (`grid.proto`):
```protobuf
message LayerLatency {
  uint32 layer  = 1;
  float  avg_ms = 2;  // EMA α=0.1
  float  p99_ms = 3;  // ring-buffer p99 over last 100 requests
}
message HeartbeatMsg {
  // existing fields unchanged
  repeated LayerLatency layer_stats = 4;
}
```

Server changes:
- `LayerLatencyTracker` struct in new `src/metrics.rs`: one EMA + `VecDeque`
  per layer, updated in `routes/walk_ffn.rs` after each layer forward.
- `announce.rs`: populate `layer_stats` in the heartbeat sender.

Router change:
- `grid.rs::update_heartbeat`: store `layer_stats` in `ServerEntry`.
- `grid.rs::route`: prefer server with lowest `layer_stats[layer].avg_ms`
  when multiple replicas cover the same layer.

**Acceptance**: `larql serve --join ... --log-level debug` logs per-layer
latency in each heartbeat. Router `/grid-status` response includes
`layer_stats` per server.

#### GT4 — WebSocket token streaming (Q1.10 completion + N0.1 SSE)

**Status**: ✅ **Shipped 2026-05-07.**

`handle_stream_generate` added to `routes/stream.rs`: accepts `{"type":"generate","prompt":"...","max_tokens":N}` WebSocket message, calls `generate_streaming` in a `spawn_blocking` task, streams `{"type":"token","text":"...","index":N}` per token, emits `{"type":"done","tokens":N,"latency_ms":M}` on completion. Client cancel supported via `{"type":"cancel"}` frame. SSE on `/v1/chat/completions` (`stream:true`) was confirmed already fully wired (N0.1 slice 3 complete).

`routes/stream.rs` previously had a working WebSocket handler for `describe` and `infer`
commands but lacked a streaming token generation path. This is the missing
piece for N0.1 slice 3 (SSE on `POST /v1/chat/completions`).

- Complete `handle_stream_infer` in `routes/stream.rs`:
  - Accept `{"type": "generate", "prompt": "..."}` WS message.
  - Call `generate_streaming` (already exists in larql-inference).
  - Emit one `{"type": "token", "text": "..."}` frame per token.
  - Emit `{"type": "done", "tokens": N, "ms": M}` on completion.
  - Handle `{"type": "cancel"}` to abort generation.
- Add binary frame support: client can send
  `{"type": "generate", "format": "binary"}` to receive token IDs as u32 LE
  instead of JSON (lower overhead for embedding clients).
- Wire SSE for N0.1: in `routes/chat.rs`, when `stream: true`, use
  `axum::response::Sse` to wrap the same `generate_streaming` callback.
  Emit OpenAI-format `data: {...}\n\n` chunks; terminate with `data: [DONE]\n\n`.

**Acceptance**: `wscat -c ws://localhost:8080/v1/stream` receives one JSON
frame per token. `curl -N -H "Accept: text/event-stream" \
-d '{"model":"...","messages":[...],"stream":true}' \
http://localhost:8080/v1/chat/completions` streams tokens in SSE format.

#### GT7 — QUIC transport for grid

**Status**: ✅ **Shipped 2026-05-15 (router) + earlier on the server side; ROADMAP entry was stale.**

Feature-gated by `--features quic` on both `larql-server` and
`larql-router`. The transport wrapper lives in
`crates/larql-router-protocol/src/transport/quic.rs` (shared between
both crates so client + server code paths stay in sync).

Server side (`crates/larql-server/`):
- `connect_grid_channel` (`src/announce.rs:282-339`) parses `quic://`
  scheme on `--join` URLs and dispatches to the QUIC client endpoint;
  fingerprint pinning via `--quic-cert-fingerprint <SHA-256>`. Falls
  through to plain TCP gRPC for `http://` URLs.
- `--quic-cert-fingerprint` flag wired through to both `AnnounceConfig`
  and `AvailableConfig` (`src/bootstrap.rs:662, 1125-1128, 1176`).

Router side (`crates/larql-router/`):
- `--quic-port`, `--quic-cert`, `--quic-key`, `--quic-server-name`
  flags accept QUIC `Join` connections via the same QUIC endpoint.
- Self-signed TLS cert auto-generated when `--quic-cert`/`--quic-key`
  aren't passed; server logs the SHA-256 fingerprint for clients to
  pin.

Acceptance test:
`crates/larql-router-protocol/tests/test_quic_roundtrip.rs` — opens a
real QUIC endpoint, runs `Join` over the wrapper, asserts streaming
announce/heartbeat semantics survive the transport swap.

**Limitation (clarified in ADR-0019):** This is QUIC-as-TCP-replacement
(HTTP/2 over a single QUIC bi-stream). True HTTP/3 with per-stream
independence shipped separately under ADR-0019 (router) for the MoE
expert fan-out path, behind `--http3-shards` / `--http3-port`.

---

### G-MODEB. Self-assembling grid Mode B (ADR-0011)

#### GT5 — Gap-fill assignment

**Status**: ✅ **Shipped 2026-05-13 (router) + 2026-05-16 (server end-to-end test).**

Server-side `run_available_loop` in `crates/larql-server/src/announce.rs`
sends `AvailableMsg` → handles `AssignMsg` by calling
`shard_loader::download_and_load_shard` (atomic tar-then-rename, SHA-256
verification when a real content hash is provided) → sends `ReadyMsg`
or `RefuseMsg(reason="download_failed")` → loops until `AckMsg` from
the router. Public `try_once_available` entry point lets integration
tests drive a full handshake end-to-end. Router-side serves
`GET /v1/shard/{model_id}/{start}-{end}` as a streamed tar
(`crates/larql-server/src/routes/shard.rs`; documented in
[`docs/router-spec.md`](docs/router-spec.md) §4).

Wired end-to-end + tested:
- `tests/test_grid_mode_b.rs::mode_b_full_vertical_handoff` — protocol-level
  drive of the gRPC stream + direct `shard_loader` call (covers AssignMsg
  shape, hash propagation, tar unpack).
- `tests/test_grid_mode_b.rs::mode_b_try_once_available_drives_full_handshake`
  — exercises the production `try_once_available` loop end-to-end (Available
  → Assign → download → Ready → Ack) against an in-process router.
- `tests/test_grid_mode_b.rs::no_assign_when_gap_has_no_surviving_origin`
  — router declines to assign when no live replica can be origin.

**Known follow-up — GT5 hash semantics mismatch (P1):**
`vindex_identity_hash` (announce.rs:183) emits a 16-hex model-identity
tag (`u64.hash`-based), but `shard_loader` verifies SHA-256 of the
downloaded tar bytes against `AssignMsg.shard_hash`. Today this
"works" only because deployments pass an empty/placeholder hash so
the verification is skipped (see the `skip_hash` branch at
`shard_loader.rs:62`). Real hash verification — meaning the donor
hashes its on-disk shard at announce time and the spare verifies the
download against that — is a follow-up. ADR-0011 left this implicit;
the right shape is probably a new optional `shard_content_sha256`
field on `AnnounceMsg` distinct from `vindex_hash`.

**Mode A AssignMsg edge case:** `announce.rs:413-428` now logs a
descriptive warning when an already-serving Mode A stream receives an
unexpected AssignMsg (router bug — AssignMsg should target Mode B
available pool only). Previously logged "Mode B not implemented",
which was misleading because Mode B *is* implemented in
`run_available_loop`; the stub was for a different code path.

#### GT6 — Dynamic rebalancing

**Status**: ✅ **Shipped 2026-05-13 (router) + earlier on the server side; ROADMAP entry was stale.**

Server-side `announce.rs:416-442` handles `UnassignMsg` by polling
`requests_in_flight` for up to 30 s (`DRAIN_TIMEOUT`), then sending
`DroppingMsg(reason="reassigned")` and either exiting cleanly or
re-entering Mode B on the same gRPC stream via `run_available_loop`
when `available_after_drain` is configured (ADR-0011 §Phase B2).
Router-side rebalancer task lives at
`crates/larql-router/src/tasks/rebalancer/` (6-module folder shipped
in ADR-0016) with periodic ticks for replication, eviction,
imbalance detection, and hot-shard elevation. Latency-driven
rebalancing reads `LayerLatency.avg_ms` from heartbeats (GT3); under-
replication tick pulls spares from the available pool.

Tested:
- `tests/test_grid_drain_reassign.rs::drain_then_reassign_via_available_after_drain`
  — drives the full UnassignMsg → drain → DroppingMsg → re-enter Mode B path.
- Router-side replication + rebalancer covered in
  `crates/larql-router/tests/test_admin_rpcs.rs` and the chaos test.

---

### G-BENCH. Grid benchmarking (ADR-0012)

#### GT8 — `larql bench` grid/wire/transport extensions

**Status**: ✅ **Shipped 2026-05-15.** All flags except `--transport` (which
waits on GT7 QUIC) are live. The CLI now lives under `crates/larql-cli/src/commands/primary/bench/`
as a folder of single-responsibility modules with per-file 90%+ test coverage
gated by `crates/larql-cli/coverage-policy.json`.

**What shipped:**

- `--bench-grid` — 1..N shard sweep over a `--moe-shards` map; emits
  `shard_efficiency = tok/s / (N × single_shard_tok/s)` per row.
- `--wire f32,f16,i8` — one row per format against `--ffn`; the parity
  guarantee is at the codec level (`larql-inference/WirePreference` chooses
  the best mutually-supported format).
- `--concurrent N` — spawns N parallel client threads per backend; aggregate
  tok/s = sum(client.tok_per_s), p99 = max(client.p99). Production wire path
  is `std::thread::spawn` over the existing sync bench fn — no async refactor.
- `--output json` / `--output-file PATH` — emits the ADR-0012 envelope:
  `{timestamp, model, prompt, tokens, wire, concurrent, results[...]}`.

**Module layout** (`commands/primary/bench/`):
- `args.rs` — clap `BenchArgs`.
- `row.rs` — `BenchRow` + `BenchJsonRow` + `BenchJsonResult` + percentile helpers.
- `helpers.rs` — wire-list parser, concurrent aggregator, shard-efficiency math.
- `output.rs` — table renderer split into pure `Vec<String>` formatters.
- `ollama.rs` — Ollama side-by-side bench (curl wrapper isolated behind a
  `Fetcher` indirection so the orchestration is unit-testable).
- `engine.rs` — KV-engine post-processing helpers.
- `local.rs` — local Metal/CPU post-processing helpers.
- `remote_ffn.rs` — concurrent-row aggregation, FFN summary, label composer.
- `remote_moe.rs` — shard-map parser, MoE summary, label composer.
- `*_runtime.rs` — I/O wrappers (`run_larql`, `run_engine*`, `run_remote_ffn_bench`, `run_remote_moe_bench`). Excluded from the per-file coverage gate.
- `run.rs` — top-level dispatch. Excluded from the per-file coverage gate.

`--transport http,quic` is documented but deferred to GT7 (ADR-0010 QUIC).

**Acceptance**: `larql bench <vindex> --ffn URL --wire f32,f16 --output json --output-file out.json`
writes a JSON envelope containing both wire format results with their
`wire_bytes_per_tok` and `ms_per_tok.{mean,p50,p99}` fields populated.

#### GT9 — Criterion micro-benchmarks

**Status**: ✅ **Shipped 2026-05-07.**

`larql-inference/benches/wire_codec.rs` (encode f32 request, decode f32/f16 response, 30-layer batch) and `larql-router/benches/routing.rs` (route single layer, route_all 30/62 layers, update_heartbeat, rebuild_route_table) — both parameterised over server counts and hidden sizes with no hardcoded model names. `larql-router` gained `src/lib.rs` re-exporting `pub mod grid`. Makefile: `bench-wire`, `bench-routing`, `bench-grid`, `bench-all`.

**Spec**: ADR-0012 §Layer 2.

- `crates/larql-inference/benches/wire_codec.rs`: encode/decode throughput
  (MB/s) for f32/f16/i8 at hidden_size ∈ {2560, 4096, 5120}, seq_len ∈ {1, 32, 256}.
  Parameters read as `criterion::BenchmarkId` — no hardcoded model names.
- `crates/larql-router/benches/routing.rs`: `route()` hot path (ns/op at
  1/10/100 servers), `rebuild_route_table()` cold path, `update_heartbeat()`.

Run with: `make bench-wire` / `make bench-routing`.

#### GT10 — CI regression gate

**Status**: ✅ **Shipped 2026-05-15.** Scripts + comparator + baselines
directory all live; the script writes the first run as the baseline and
compares subsequent runs against it.

**Files:**
- `scripts/bench-grid-regress.sh` — wraps `larql bench ... --wire f32,f16 --output json`,
  compares against `bench/baselines/grid-<model>.json`. Saves the current
  run as baseline when none exists. Env vars: `LARQL_BENCH_VINDEX`,
  `LARQL_BENCH_FFN_URL`, optional `LARQL_TOK_PER_S_THRESHOLD` (default 0.05),
  `LARQL_P99_THRESHOLD` (default 0.10).
- `scripts/bench_compare.py` — pure-stdlib JSON diff. Fails if any `backend`
  in the baseline regresses tok/s by more than the threshold or rises p99
  by more than the threshold.
- `bench/baselines/README.md` — workflow for updating baselines after a
  deliberate perf improvement.

**Acceptance**: `LARQL_BENCH_VINDEX=… LARQL_BENCH_FFN_URL=… ./scripts/bench-grid-regress.sh gemma3-4b-q4k`
exits 0 on a clean run; exits 1 with a per-backend failure list if any
threshold trips.

---

### F-COLLECT. Parallelize shard collection in `forward_moe_stream_collect_with_timing`

**Status**: ✅ **Shipped 2026-05-02.** Both halves of the gRPC dispatch are
now parallel across shards:
- `forward_moe_stream_collect_with_timing` uses `std::thread::scope`,
  one OS thread per stream, joined into a single result vector.
  `ShardStream::result_rx` was wrapped in `std::sync::Mutex` to make
  `ShardStream: Sync` (the type-system requirement for parallel borrow).
- `forward_moe_stream_fire` uses `rayon::par_iter().enumerate().try_for_each(...)`
  with a single-shard fast path. The blocking residual-bytes / post-norm-bytes
  clones now happen across rayon workers instead of serially.

Verified on 2-shard local-loopback: per-layer collect ≈ 21 ms (~ equal to
1-shard collect time), confirming `collect ≈ max(per_shard.wall)` rather
than `sum` — the structural win. Real-network validation pending under
**F-FLY** below; loopback can't show the absolute tok/s improvement
because both shards finish nearly simultaneously and the savings sit
under M3 Max P-core saturation noise.

**Driver**: 2026-05-02 bottleneck analysis on the local Metal MoE path
vs the CPU/grid path (single shard, colocated). Both land at ~19 tok/s
because the grid sequentially blocks on each shard's `collect_with_timing()?`
in `crates/larql-inference/src/ffn/moe_remote.rs:1984`. With one shard,
sequential = max. With 2+ shards over real network, the per-layer
collect time stacks instead of overlapping.

**Concrete impact** (Gemma 4 26B-A4B, 30 MoE layers, top_k=8):

| Topology | Per-shard wall (RTT) | Collect/layer today (sequential) | Collect/layer fixed (parallel) | Saved per token |
|---|---|---|---|---|
| 1 shard local | ~8 ms | ~8 ms | ~8 ms (no change) | 0 |
| 2 shards LAN (~5 ms RTT) | ~5–10 ms | sum ≈ 10–20 ms | max ≈ 5–10 ms | ~5–10 ms × 30 layers = **150–300 ms/tok** |
| 4 shards LAN | ~5–10 ms | sum ≈ 20–40 ms | max ≈ 5–10 ms | ~15–30 ms × 30 layers = **450–900 ms/tok** |
| 4 shards cross-region (~50 ms RTT) | ~50 ms | sum ≈ 200 ms | max ≈ 50 ms | ~150 ms × 30 layers = **4500 ms/tok** |

The `fire` half of `forward_moe_stream_fire` already pushes to all
streams' channels in a non-blocking loop — concurrency exists at the
wire layer; the bug is the blocking serial collect on top.

**Fix**: change the collect loop from

```rust
for stream in streams.iter().take(n_streams) {
    let (partial, server_compute_ms) = stream.collect_with_timing()?;
    // accumulate into out
}
```

to a concurrent join. `tokio::join_all` if the call site is async, or
`std::thread::scope` / `rayon::par_iter().map(...)` if not (each
`collect_with_timing` blocks on a condvar inside `ShardStream`, so
parallelism comes from holding multiple condvars in flight). Picking
between these depends on whether `ShardStream::collect_with_timing` is
`Send + Sync`; check before deciding.

**Acceptance**: `LARQL_MOE_TIMING=1` summary line on a 2-shard run
reports `collect ≈ max(per_shard)`, not `sum(per_shard)`. End-to-end
tok/s on a 2-shard local-loopback run improves measurably.

**Strategic context**: this is the load-bearing primitive for the
"split in grids" axis of LARQL — the future Kimi K2.6 / DeepSeek V4
deployment shapes will need 8+ shards. Without this fix, the grid
scales backwards: more shards = more sequential collect time.

### F-LOCAL-MOE. Local Metal MoE optimisations (CPU staging + batched dispatch)

**Status**: Not started.

**Driver**: same 2026-05-02 bottleneck analysis. On the local Metal
MoE path, **67% of wall is CPU work**, only 33% is GPU active (51 ms
wall = 17 ms GPU + 33 ms CPU + sync). The GPU is barely loaded — the
CPU-side per-layer router + memcpy of 8 expert Q4_K byte slices into
staging buffers + commit/wait sync is dominating.

For the "run large models on consumer hardware" axis, every ms here
matters — the user runs LARQL on a single M3 Max, the grid isn't
available.

**Two levers, both CPU-path-safe**:

1. **Zero-copy expert byte aliasing**: today
   `gpu_moe_dispatch_with_scratch` memcpys ~300 KB per expert × 8 ×
   30 layers = ~72 MB of Q4_K bytes per token into pre-allocated
   staging buffers. The infra already exists —
   `MetalBackend::cached_buffer_for_bytes` does
   `new_buffer_with_bytes_no_copy` for the shard server's pre-staged
   path. Wiring it for the local path eliminates the per-layer
   memcpy entirely; experts alias the model's mmap directly.
   **Estimated win: 5–10 ms/tok.**

2. **Batched expert GPU dispatch**: today each MoE layer issues 24
   GPU dispatches (8 × `q4k_ffn_gate_up` + 8 × `geglu` + 8 ×
   `q4k_matvec` for down). Batching these into ~3 dispatches/layer
   using per-expert offsets into the already-staged buffers reduces
   dispatch overhead from ~720 calls/token to ~90.
   **Estimated win: 3–5 ms/tok.**

Combined: **8–15 ms/tok off the local path → 23–28 tok/s** on Gemma 4
26B-A4B Metal MoE (from 19.4 tok/s today).

**Acceptance**: `LARQL_GPU_TIMING=1` shows `cpu` shrunk by ~10 ms/tok;
`larql bench gemma4-26b-a4b-q4k-v2` shows ≥23 tok/s warm-state on
M3 Max with output unchanged.

### F-FLY. Remote multi-shard deployment on fly.io

**Status**: Not started — next session.

**Goal**: validate the HTTP CPU-path optimisations from the 2026-05-01 session
on a real network (LAN-class RTT ≥ 100 µs), not just M3 Max loopback. Most
of what we shipped is designed to win on real links but is invisible on
loopback (TCP_NODELAY, f16 wire). This is the apples-to-apples test that
tells us whether the in-room engineering translates to a deployable grid.

**Setup target (~2 hosts, then 4-8 if Phase 1 looks good)**:

- 1× client host (Mac dev box or fly.io VM): runs `larql run --moe-shards`
  with attention + dense FFN compute. Holds the 2 GB attention/router/dense
  weight set.
- N× shard hosts (fly.io VMs, ~16 GB RAM each): each runs
  `larql-server --experts START-END --grpc-port 9081 --uds-path ...`
  on a slice of the expert table. 26B-A4B has 128 experts × 30 layers;
  e.g., 4 shards × 32 experts × 30 layers ≈ 4 GB Q4_K + 2 GB working set
  per shard.
- Network: same fly.io region (intra-DC ~0.5 ms RTT) for Phase 1; a second
  region (cross-region ~30-100 ms RTT) for Phase 2 to stress the streaming
  overlap.

**What we expect to learn from this**:

1. Whether the **f16 wire** opt-in actually wins on real links (estimate:
   +3-5% on 1 Gbps, more on slower). On loopback it was within noise; we
   need real RTT to see the wire-bytes saving translate.
2. Whether **gRPC SPLIT default** (now on by default for gRPC) holds its
   ~12% steady-state win when the network leg is bigger than the dense
   FFN GPU leg (instead of comparable). The overlap math says the win
   grows when RTT > dense_FFN_time.
3. End-to-end tok/s ceiling on a real grid — we currently know loopback
   is ~19.7 tok/s; a multi-host grid should be slower per-token but
   throughput-scalable (more shards per host = more concurrent expert work).
4. Whether **predispatch (`batch` dispatch mode)** actually breaks
   generation on every multi-host setup or just on M3 Max loopback. We
   saw garbage output on loopback; might be a different story with real
   network timing.

**Prerequisites already in place** (from this session):

- gRPC streaming default-on for gRPC shards (~12% loopback gain,
  expected to grow on RTT-heavier links)
- TCP_NODELAY on accepted connections (defensive against tail-packet
  stalls on real LAN)
- f16 wire as opt-in (`LARQL_MOE_WIRE_F16=1`)
- Unix domain sockets (`--uds-path`, `unix:///path` URL) for same-host
  shard collocation
- `LARQL_HTTP_TIMING=1` per-call instrumentation (encode / send_total /
  recv_body / decode breakdown)
- `LARQL_MOE_TIMING=1` per-token MoE summary (route / collect / server
  compute / network estimate)
- 9.6× CPU MoE speedup on the shard side (bench: 30-layer sweep
  221 → 22.9 ms; production: 2.3 → ~19.7 tok/s end-to-end on M3 Max
  loopback)

**fly.io specifics worth pinning down before deploy**:

- VM size for shards: 26B-A4B vindex is ~16 GB on disk; needs ~10 GB
  RSS at warmup. `performance-cpu-2x` (~7 GB RAM) won't fit a full
  shard; need `performance-cpu-4x` (~14 GB) at minimum, or shard the
  vindex finer.
- Vindex distribution: cheapest is to ship the full 16 GB to each shard
  and let `--experts START-END` cap working set; alternative is per-shard
  vindex slicing (`larql slice` exists but needs a per-shard variant).
- Persistent volume vs in-memory: with `--warmup-walk-ffn` the boot
  cost is ~6-7 s; if VMs reboot per deploy, that adds up. Consider
  fly.io persistent volumes for the vindex.
- Health check: `/v1/health` is already there.
- Authentication: the existing `--api-key` flag works but a multi-tenant
  fly.io setup probably wants per-shard token rotation (out of scope for
  Phase 1).

### F0. CPU MoE correctness — RESOLVED ✅

**Status**: Closed 2026-05-01.

Smoke-test `larql run output/gemma4-26b-a4b-q4k.vindex "The capital of
France is" --max-tokens 5` (no `--moe-shards`, no `--metal`) returns
**"Paris."** End-to-end CPU path on the per-layer Q4_K hybrid-MoE
vindex now produces the correct answer; the M-CPU kernel work
(NEON SDOT direct-Q4K + scratch reuse + correct hybrid-combine
ordering, see `larql-inference/ROADMAP.md → M-CPU-1..6`) shared the
code path with the server-side fix that landed 2026-04-30, so the
local route inherited the correctness for free.

The historical analysis below is preserved as forensics for future
CPU-vs-Metal divergence debugging — the diff-and-localise pattern
generalised better than the specific bug.

**Historical context (2026-04-27, pre-M-CPU work):**

The per-expert refactor + `experts_packed.bin` removal landed without a
correctness end-to-end check. `larql run` on the 26B-A4B vindex via the CPU
MoE path produces incoherent text ("ever own로 el"), while `larql run --metal`
on the same vindex produces "Paris." The server-side remote-expert endpoint
inherits the same bug because `run_single_expert` and `cpu_moe_forward` share
the same per-expert compute.

**What I tried that did not help:**
- Aligning `cpu_moe_forward`'s router-norm input to `h_norm` (matching Metal's
  `cpu_moe_route(&h_norm, ...)` convention) — different garbage, not "Paris".
- Swapping gate/up row order in the `[2*inter, hidden]` slice — different
  garbage, not "Paris".
- Verified `dequantize_q4_k` is bit-identical to the `larql_models` reference
  via `tests/test_q4k_parity.rs` on synthetic ramp data (3 super-blocks of
  varied content, plus round-trip-within-noise).
- Verified `inter_padded` handling matches Metal's convention (zero-pad
  hidden_state to `inter_padded`, dequant down at `hidden * inter_padded`).

**What's still suspect:**
- Q4_K dequant on the **real per-layer file's bytes** has not been compared
  against Metal's GPU dequant. Synthetic parity ≠ real-data parity.
- The **gate/up convention in HF Gemma 4** could differ from what
  `quantize_moe_entries` assumes about the source BF16 layout.
- BLAS `sgemv` on Apple Accelerate vs Metal's `q4k_matvec` shader could have
  precision drift at 26B scale, though both should be IEEE-754 correct.

**Why the bench numbers were misleading:**
`bench_expert_server` measured `forward_moe` warm at 1.91 ms and the
`cpu_moe_forward` floor at 0.10 ms. Post-fix the floor jumped to 1.81 ms (18×).
The 0.10 ms number was the buggy old code silently returning empty buffers
when the dequant length didn't match the bytes — fast because no work was
happening. This was not flagged because no test compared **output values**,
only latency.

**Diagnosis status (2026-04-27, via `larql parity` + dump-and-diff):**

Layer-by-layer cosine-similarity diff between CPU `predict_q4k` and Metal
`predict_q4k_metal` on the 26B-A4B vindex, using `LARQL_CPU_DUMP_LAYERS` +
`LARQL_DUMP_RESIDUALS`:

| Stage at layer 0 | cos(cpu, metal) |
|---|---|
| h_embed (input to layer 0) | 1.000000 |
| h_post_attn (post-attention) | 1.000000 |
| layer_out (post-FFN+MoE+combine) | **0.626708** ← divergence |

Attention is correct on layer 0; the divergence is in the **FFN + MoE +
combine** between `h_post_attn` and `layer_out`. The CPU MoE block routes
to the same top-K experts as Metal at layer 0 (verified via `MOE_DEBUG=1`:
both pick `[79, 114, 16, 92, 89, 101, 67, 46]` with the same `moe_out_rms`).
Per-expert math is provably correct (parity test). The bug is therefore in
how `run_moe_layer_cpu` composes h1 (dense), h2 (MoE), the outer
post-FFN norm, and `layer_scalar` — and it has drifted from Metal's
`metal/decode/moe_combine.rs::apply_outer_combine`.

`larql parity` v1 shipped (CLI subcommand, `larql-cli/src/commands/diagnostics/parity.rs`)
with `--component moe-expert` + `--component moe-block` and `--backends reference,cpu`.
Run on the 26B-A4B vindex the tool reports:

| Component | reference vs cpu max abs diff | Verdict |
|---|---|---|
| `moe-expert` layer 0 / expert 0 | 4.3 × 10⁻⁶ | within fp32+BLAS noise |
| `moe-block` layer 0 (router → top-K → K experts → sum → post-norm) | 8.4 × 10⁻⁵ | within fp32+BLAS noise |

So the entire MoE expert pathway — Q4_K dequant, gate matmul, up matmul,
activation, down matmul, router, top-K, weighted sum, post-experts norm — is
mathematically correct end-to-end. The bug producing garbage on `larql run`
is **outside** the MoE block. Suspect surface area:

- attention block (Q/K/V proj, RoPE, softmax, O proj) — Metal vs CPU
- hybrid combine: `h1 + h2 → moe_post_outer_norm → + h_post_attn` in
  `larql-inference/src/vindex/q4k_forward.rs::layer_step`
- `apply_layer_scalar` and PLE (`apply_per_layer_embedding`) afterwards
- per-position iteration loop on prefill (`for pos in 0..seq_len`)

**Root cause (further localised 2026-04-27):**

The CPU and Metal paths use **two different forward implementations** for
hybrid-MoE Q4_K vindexes — they have drifted:

- **Metal**: `predict_q4k_metal` builds `FullPipelineLayer` per layer and
  calls `backend.decode_token(&layers, ...)`. Hybrid MoE handled by
  `decode_token_with_moe` → `gpu_moe_dispatch`. This works.
- **CPU**: legacy `q4k_forward.rs::predict_q4k_step` →
  `run_moe_layer_cpu` (hand-rolled) → `cpu_moe_forward` per position +
  hand-rolled hybrid combine (`combined = h1 + h2`,
  `combined_normed = outer_norm(combined)`, `h_out = h_post_attn + combined_normed`).
  Doc comment in that function says it's "verified against HF bf16 via
  residual-cosine diff in the Metal `diag.rs` dumps" — but the file has
  since drifted from Metal and the verification is stale. This produces
  garbage end-to-end on Gemma 4 26B-A4B.

Routing-convention fix (apply router_norm to `h_norm`, not raw `h`,
matching Metal's `cpu_moe_route(&h_norm, ...)`) was applied to
`cpu_moe_forward` and `MoeRouterWeights::route`, with regression tests in
`larql-compute/src/cpu/ops/moe/mod.rs`. Necessary but not sufficient — the
hybrid combine in `run_moe_layer_cpu` is still wrong.

**Next steps for F0 (proper fix):**

The cleanest path is to **delete `run_moe_layer_cpu` and route CPU
predictions through the same `FullPipelineLayer` + `decode_token` pipeline
Metal uses**, swapping `MetalBackend` for `CpuBackend`. That requires
`CpuBackend::decode_token` to support Q4 layers (it currently doesn't —
`predict_q4k_metal` literally `expect()`s "need Metal with Q4 kernels").

Either:
- Implement `CpuBackend::decode_token` for Q4 layers — substantial work
  porting the Metal kernels' algorithm to CPU + BLAS, but unifies the two
  paths and resolves all class-of-bug drifts at once.
- Patch `run_moe_layer_cpu` to match Metal's exact hybrid combine. Faster
  but leaves the dual-path drift surface in place; another knob will go
  out of sync next session.

A `larql parity --component layer` (parity v2) component would catch this
class of bug going forward — diffing the **full hybrid layer output**
between CPU and Metal would have surfaced the combine drift immediately.
That's the right next investment.

**Implication for the remote-MoE story:**
The wire format, `--experts` shard ownership (with the off-by-one fix),
the per-expert byte-table API, and the per-layer Q4_K layout all work
correctly. What does **not** work is the CPU numerical compute on the
server side. Until F0 is closed, "remote MoE on Gemma 4 26B-A4B" is
plumbing-correct but inference-incorrect — clients pointing at a remote
larql-server shard will get garbage output. Workaround: use `--metal` for
all-local generation; remote-MoE is on hold.

---

Functional gaps from the 2026-04-27 server review. Numbering is stable so we
can reference items in commits and reviews.

### F1. Router-side expert-shard fan-out
**Files**: `crates/larql-router/src/main.rs`, `crates/larql-router/src/grid.rs`,
`crates/larql-router-protocol/proto/*.proto`.
The grid router fans out `walk-ffn` by layer ranges only. For MoE, the
remote-expert client (`RemoteMoeBackend` in `larql-inference`) carries the
expert→shard map itself; nothing on the router side. Means clients can't just
point at the router for MoE. Add `POST /v1/expert/{layer}/{id}` and
`POST /v1/expert/batch` to the router, with shard discovery via the existing
gRPC announce stream. Pairs with **F11** (topology endpoint).

### F2. Streaming HTTP infer (SSE)
**Files**: `crates/larql-server/src/routes/infer.rs` (new sibling
`infer_stream.rs`).
`/v1/infer` is single-shot — full output buffered, no incremental tokens. WS
has it (`WS_CMD_INFER`) but most chat UIs talk SSE. Add
`POST /v1/infer/stream` with `text/event-stream`. Same generation loop, yield
each token. Mid-generation cancellation on client disconnect (see **F16**).

### F3. `/metrics` (Prometheus)
**Files**: `crates/larql-server/src/main.rs`, new `crates/larql-server/src/metrics.rs`.
No latency histograms, no per-endpoint counters, no rate-limit drops, no
shard-call durations today. Wire `metrics` + `metrics-exporter-prometheus` (or
hand-rolled). Histograms for: `walk-ffn` per `layer_count`, `forward_moe` per
`top_k`, queue wait, auth failures, rate-limit drops, shard-call latency.

### F4. Graceful shutdown with in-flight drain
**Files**: `crates/larql-server/src/main.rs`.
SIGTERM today probably cuts long-running walks. Standard axum + tokio shutdown
signal: stop accepting, drain N seconds (configurable), hard-kill. Important
for grid rolling restarts.

### F5. Readiness vs liveness split
**Files**: `crates/larql-server/src/routes/health.rs`, `routes/mod.rs`.
`/v1/health` returns `{status, uptime, requests_served}`. Add `GET /v1/ready`
returning 503 until weights are loaded (under `--warmup-walk-ffn` or first
lazy load); include `model_id`, `mode`, `version`, `git_sha`, `format`
(per-layer vs legacy) in the readiness payload. Standard k8s liveness/readiness
split.

---

## P1: Active

### Q1.10 Reduce `routes/stream.rs::handle_stream_infer` (327 LOC) — deferred

The remaining open code-quality item from the 2026-05-01 audit. The other
nine (Q1.1–Q1.9) shipped — see "Completed → 2026-05-01 (continued) — Q1
code-quality cleanup". Q1.10 is deferred until N0.1 (OpenAI Chat
Completions SSE) forces a similar streaming state-machine shape; the
two should share infrastructure. Effort estimate: ~3 hours when picked up.

---

### F6. Replica round-robin + retry on shard failure
**Files**: `crates/larql-router/src/grid.rs`.
Router picks first owning shard; no load-balancing across replicas, no retry
on 5xx. `--shards "0-15=A,0-15=B"` doesn't fan evenly today.

### F7. KV-cache prefix sharing for chat
**Files**: `crates/larql-inference/src/layer_graph/generate/*`,
`crates/larql-server/src/routes/infer.rs`.
Every `/v1/infer` call is fresh prefill. For chat (long shared system prompt +
short user turn) prefix-caching is a 5–10× decode-time win. Needs a
`session_id`-keyed KV cache.

### F8. Vindex hot-swap admin endpoints
**Files**: `crates/larql-server/src/routes/` (new `admin.rs`),
`crates/larql-server/src/state.rs` (mutable model registry).
`POST /v1/admin/vindex/load`, `DELETE /v1/admin/vindex/{id}`,
`POST /v1/admin/vindex/reload`. Admin-key-gated (see **F14**). Otherwise every
model swap is a process restart.

### F9. Binary wire format for `expert/batch`
**Files**: `crates/larql-server/src/routes/expert.rs`,
`crates/larql-inference/src/ffn/moe_remote.rs`.
A K=8 batch on Gemma 4 26B-A4B is ~90 KB JSON per call. The
`application/x-larql-ffn` binary format already exists for `walk-ffn`; mirror
it for `expert/batch`. Expected 3–5× wire reduction.

### F10. OpenAI-compat `/v1/chat/completions` — superseded by N0

This item scoped only the chat completions endpoint shallowly. See
**N0** in the "Great new functionality" section above for the full
plan: chat completions + completions + responses + embeddings +
models, with streaming, tools, structured output, and constrained
decoding. F10 is left here for cross-references; the work happens
under N0.

### F11. Expert topology endpoint
**Files**: new `crates/larql-server/src/routes/topology.rs`.
`GET /v1/expert/topology` returns `{model_id, layers, num_experts, owned: [start,end]}`.
Lets clients build the shard map dynamically instead of having it baked in.
Pairs with **F1** (router fan-out).

### F12. Batched infer
**Files**: `crates/larql-server/src/routes/infer.rs`.
`/v1/infer` takes one prompt today. RAG workloads send N prompts; one batched
call across them amortises router/dispatch overhead. Either accept
`prompts: [...]` or new `/v1/infer/batch`.

### T3. Review follow-up — server hygiene ✅ done 2026-04-26

**Scope**: follow-up from review of `larql-server` focused on magic strings,
modularity, cleanliness, tests, and clippy.

Shipped:
- `X-Forwarded-For` is ignored by default for rate limiting; new
  `--trust-forwarded-for` opt-in is for deployments behind a trusted proxy.
- HTTP protocol constants added for shared health path, API prefix,
  bearer prefix, and binary FFN content type.
- Route path literals in `routes/mod.rs` centralized as named constants so
  single-model and multi-model routing drift is easier to spot.
- `load_single_vindex` now takes a `LoadVindexOptions` struct instead of
  an 11-argument call and repeated `too_many_arguments` clippy allows.
- Embed endpoints now return the standard `{"error": ...}` JSON envelope
  for errors instead of a mix of plain text and JSON.
- Server-local clippy cleanup removed the repeated `too_many_arguments`
  exemptions from the vindex loading path.

Follow-up worth keeping open:
- Consider a route-registration macro/table if route count keeps growing.

### T1. Test coverage — functional tokenizer + uncovered routes ✅ done 2026-04-26

**Outcome**: 49.1% → **58.0% line**, 56.4% → **65.3% function**. 345 → 402 tests.

**Root cause fixed**: added `functional_tokenizer()` (WordLevel, France→0 etc.) to
`tests/common/mod.rs`. The empty BPE tokenizer that previously blocked all
tokenize-dependent routes is now supplemented by a real in-memory tokenizer that
maps test words to embeddings with known KNN hits.

**Files moved:**

| File | Before | After |
|---|---|---|
| `band_utils.rs` | 35% | **100%** |
| `routes/describe.rs` | 48% | **95%** |
| `routes/walk.rs` | 38% | **96%** |
| `ratelimit.rs` | 70% | **98%** |
| `routes/walk_ffn.rs` | 54% | **77%** |
| `routes/patches.rs` | 63% | **91%** |
| `routes/relations.rs` | 83% | **91%** |

**Remaining hard ceiling** (no path forward without real weights or real sockets):

| File | Coverage | Reason |
|---|---|---|
| `grpc.rs` | 0% | Needs full gRPC server+client; defer |
| `routes/stream.rs` | 0% | WebSocket — needs `tokio-tungstenite`; defer |
| `routes/explain.rs` | 11% | Calls `get_or_load_weights()`; rest gated on real model |
| `embed_store.rs` | 25% | Reads real f16 embedding files |
| `main.rs` | 0% | CLI entrypoint; skip |

### T2. Test coverage — remaining reachable paths ✅ done 2026-04-26

**Current**: 74.2% line / 81.2% function. 478 tests.

**Completed this pass:**
- `grpc.rs` 0% → **65%** — 28 direct gRPC handler tests (health, stats, describe, walk, select, relations, walk_ffn, infer, stream_describe)
- Magic strings: `"probe"` → `PROBE_RELATION_SOURCE`; `"ok"` → `HEALTH_STATUS_OK`; infer mode strings in grpc.rs; WebSocket message types in stream.rs (`WS_TYPE_*`, `WS_CMD_*`)
- `embed_store.rs` 25% → **98% line** — tiny f16 mmap fixtures cover open, size validation, lookup, L1 cap, out-of-range, subnormal/inf/nan conversion.
- `announce.rs` 6% → **56% line** — extracted deterministic message builders for announce, heartbeat, dropping, and grid bearer metadata.
- `main.rs` boot/loading/discovery helpers moved into `bootstrap.rs`; `bootstrap.rs` has **92% function** coverage for parse/discovery/serve-alias/options behavior.
- `routes/stream.rs` 0% → **65% line** — WebSocket JSON message builders plus pure describe-message planning cover missing-entity, no-model, and functional edge streaming cases.
- `routes/infer.rs` 32% → **56% line** and `routes/explain.rs` 18% → **46% line** via request/default deserialization tests and response-formatting helpers.
- `routes/embed.rs` 67% → **87% line** — binary embed/logits parsing extracted into helpers; HTTP tests cover binary success, malformed JSON, truncated binary input, hidden-size mismatches, no-model errors, and cacheable single-token JSON/binary responses.
- `routes/walk_ffn.rs` 77% → **80% line** — validation helpers now cover layer selection precedence, missing layers, seq_len handling, overflow, and latency rounding.

**Remaining hard ceiling:**

| File | Current | Gap | What to add |
|---|---|---|---|
| `main.rs` | 0% | 237 lines | Tokio binary entrypoint; boot orchestration is covered through `bootstrap.rs` |
| `bootstrap.rs` | 43% | 134 lines | Real vindex load path still requires filesystem fixtures with full vindex assets |
| `routes/stream.rs` | 65% | 148 lines | Full WebSocket socket loop still needs a client harness such as `tokio-tungstenite` |
| `routes/explain.rs` | 46% | 167 lines | Main path gated on `get_or_load_weights()` and real inference trace |
| `routes/infer.rs` | 56% | 82 lines | Prediction paths need real or injectable inference backend |
| `routes/embed.rs` | 87% | 74 lines | Remaining positive logits path requires loadable weights/lm_head fixture |
| `routes/walk_ffn.rs` | 80% | 125 lines | Remaining full-output path requires loadable weights/FFN fixture |
| `routes/warmup.rs` | 80% | ~15 lines | `warmup_hnsw=true` warn path (HNSW not enabled) |
| `announce.rs` | 56% | ~78 lines | Remaining gap is live gRPC stream lifecycle and retry loop |

### G1. Cold-start profile ✅ done 2026-04-26
**Findings**: walk-ffn cold cost decomposes into two distinct phases:

1. **First walk-ffn ever**: ~1.27 s + ~2.9 GB RSS — lazy
   `get_or_load_weights` builds the f32-decoded gate-vector cache,
   loads `lm_head.bin` + `norms.bin`. One-shot regardless of which
   layer was requested. Confirmed not Metal init: a prior gate-KNN
   walk only adds 2 MB.
2. **First touch of each new layer**: ~17 ms + ~11 MB RSS — kernel
   page-fault for the layer's `interleaved_q4k.bin` slice (gate +
   up + down, ~22 MB on disk). Linear in number of cold layers.

Warm steady state is **0.2–0.3 ms/layer**. The 50× cold:warm ratio
is mostly phase 1; phase 2 is ~50× cheaper.

Conclusion: the win lives in phase 1 — pre-load weights at boot.
Mmap prefetch is a 12 ms one-shot for all 30 layers (negligible).
Both wired in **G2** below.

### G2. `/v1/warmup` endpoint + `--warmup-walk-ffn` flag ✅ done 2026-04-26
**Impact (measured on Gemma 26B)**: first walk-ffn **1247 ms → 12.6 ms (99×)** at the cost of +3.2 GB pre-allocated RSS and ~1.3 s boot delay.

Shipped:
- `POST /v1/warmup` accepting `{layers, skip_weights, warmup_hnsw}`
  (all optional). Returns `{weights_loaded, weights_load_ms,
  layers_prefetched, prefetch_ms, hnsw_built, hnsw_warmup_ms,
  total_ms}`.
- `larql-server --warmup-walk-ffn` boot flag — calls the same code
  path before the listener binds. Goes through
  `warmup_model_async` (`spawn_blocking`) because the boot point
  is already inside the tokio runtime.
- The endpoint runs the work on a blocking pool so the runtime
  stays responsive.

### G3. Dual-host gRPC self-assembling grid ✅ done 2026-04-26
**Live-validated** (single-host two-port simulation, exercises the
same code path as a real LAN-distributed grid):

- Shards launched with `--join http://router:50052 --grid-key <s>
  --public-url http://shard:port` register automatically; router
  logs `Grid: server joined layers=0-14` and updates coverage.
- `total_layers_covered` field on the router is the operator's
  view of grid completeness.
- Killed shard A → router logs `Grid: server left`, coverage drops.
  Layer-5 request returns HTTP 400 `"layer 5 has no owning shard"`
  (clean error, not hang). Layer 22 (live shard B) stays at 0.3 ms.
- Restart killed shard → it auto-rejoins, coverage returns to 30,
  layer 5 routes successfully (cold-page first request: 13.9 ms).
- README "Recommended setup" updated with the `--grid-port` /
  `--join` recipe (separate edit pending).

The gRPC mechanism is production-ready as of this validation.
True cross-host RTT measurement is forward-looking (G3a below).

### G3a. Cross-host RTT measurement *(forward-looking)*
**Status**: open. Requires two physical machines on the same LAN.
The same-host validation establishes correctness; cross-host
measures the additional TCP overhead per fan-out.

## P2: Forward-looking

### G-SCALE. Run T-class models on grid (Kimi K2.6, DeepSeek V4 scale)

**Driver**: LARQL's strategic axis is "run large models on consumer
hardware OR split across grids." T-class MoE models (Kimi K2 ≈ 1T total
params, top-K ≈ 8; DeepSeek V3 ≈ 671B, top-K=2; future K2.6 / V4 likely
similar shape) can't fit on any single consumer machine — the grid
deployment shape is **the only way** to run them locally.

**What changes vs Gemma 4 26B A4B (today's reference)**:

| Dimension | Gemma 4 26B-A4B | Kimi K2 (~1T) | DeepSeek V3 (~671B) |
|---|---|---|---|
| Total params | 26B | ~1T | 671B |
| Layers | 30 | ~60 | 61 |
| Experts/layer | 128 | ~384 | 256 |
| Top-K active | 8 | 8 | 8 |
| Active params/token | ~5B | ~37B | ~37B |
| Q4_K vindex size (estimate) | 16 GB | ~600 GB | ~400 GB |

**Implications for the grid primitives**:

1. **Memory-conscious shard layout**. A T-class model's expert table is
   100× our current. With 16 GB consumer-class RAM per shard, K2 needs
   ~40 shards just to fit. Per-shard memory targeting matters: each
   shard owns a tight `(layer, expert_id)` set of mmap pages and never
   loads the rest. The `--units PATH` JSON manifest already supports
   per-(layer, expert) ownership; **G5 below** (per-shard expert routing
   in router-protocol) lights it up at the router layer.
2. **Parallel shard collect is non-negotiable**. With 40+ shards,
   sequential collect would compound to seconds/token. **F-COLLECT**
   above is the prerequisite.
3. **Streaming expert byte transfer**. T-class expert weights per layer
   may not fit in RAM even on a fat shard if it owns many experts. The
   shard's mmap+page-fault behaviour does the right thing today (only
   active expert pages are paged in), but **G4 mmap residency control**
   below becomes operationally important — long-running shards need
   `madvise(DONTNEED)` after a layer to reclaim RSS.
4. **Router-side fan-out batching**. With 40+ shards and 30+ layers,
   per-layer round-trips dominate. Multi-layer `forward_moe_predispatch`
   (already exists) becomes the default rather than an opt-in; the
   pass-1 approximation cost is negligible compared to 40-shard ×
   30-layer sequential RTT.

**Status**: Forward-looking. **F-COLLECT** + **G5** + **G4** are the
direct prerequisites; once those land we should attempt a multi-shard
deployment of one T-class model end-to-end as a capability check, even
if perf is exploratory rather than production-tuned.

### G4. mmap residency control endpoint
**Impact**: For long-running shards under memory pressure, expose
`POST /v1/mmap/advise {layers, advice: "willneed"|"dontneed"}` so
operators can trim RSS or pre-warm specific layer ranges without
restarting.

### G5. Per-shard expert routing
**Impact**: For DeepSeek-V3+/Kimi K-class models (1k+ experts), shard
by expert ID within a layer rather than by layer range. Needs an
`ExpertRoute` message type in `larql-router-protocol` and
GridState dispatch updates. Mentioned in larql-vindex P2. Subsumed by
**F1** (router-side expert fan-out) at the router layer; G5 covers the
router-protocol changes specifically.

### G6. Live router-shard topology change
**Impact**: Today shards are static (`--shards` flag at router boot).
For ops convenience, expose `POST /v1/router/shards` (admin-gated)
to add/remove a shard without restarting the router. Pair with
`--grid-port` health checks.

### F13. OpenTelemetry tracing exporter
**Files**: `crates/larql-server/src/main.rs`.
Per-request spans across HTTP→shard fan-out. `tracing_subscriber::fmt` is the
only output today. Wire `tracing-opentelemetry` + OTLP exporter, configurable
via `--otel-endpoint`. Pairs with **F3** (metrics).

### F14. Per-key quotas + audit log
**Files**: `crates/larql-server/src/auth.rs`, `crates/larql-server/src/main.rs`.
Single API key today; no per-key quotas, no rotation, no scoped tokens. Add
`--api-keys keys.toml` (name + role + per-key rate). Structured audit on
patches + admin ops to a configurable sink (file / stdout / OTel).

### F15. RBAC (read-only vs admin keys)
**Files**: `crates/larql-server/src/auth.rs`, all mutating routes.
Today any key can patch the loaded model. Add `role` per key
(read / infer / patch / admin). Mutating endpoints (`patches/apply`,
`insert`, future `admin/*`) require the matching role.

### F16. Mid-generation cancellation on HTTP infer
**Files**: `crates/larql-server/src/routes/infer.rs`.
Client disconnect on `/v1/infer` waits for the full max_tokens. Wire
`tokio::select!` against an axum `OnUpgrade`-style cancellation token (or just
poll the connection on each decode step) to abort early.

### F17. Structured-output / grammar-constrained generation
**Files**: `crates/larql-inference/src/layer_graph/generate/*`,
`crates/larql-server/src/routes/infer.rs`.
`{format: "json", schema: ...}` or `{grammar: "gbnf:..."}` on `/v1/infer`.
Constrains decoding by masking the logits to grammar-valid tokens. Standard
ML-server feature; missing today.

### F18. Log-prob / perplexity endpoint
**Files**: new `crates/larql-server/src/routes/logprobs.rs`.
`POST /v1/logprobs {prompt, top_k}` — return per-token log-probabilities.
Needed for ranking, classification, and eval workflows.

### F19. OpenAPI schema route
**Files**: new derive macro setup using `utoipa` (or hand-rolled).
`GET /openapi.json`. Required for SDK codegen, `kubectl explain`-style
tooling, and external API consumers. Today external consumers read the
README.

### F20. Compression negotiation
**Files**: `crates/larql-server/src/main.rs`.
No `Content-Encoding: gzip|zstd` advertised; relies on a reverse proxy. Wire
`tower-http::compression`. Particularly useful for `walk-ffn` JSON responses
on slow links.

### F21. `/v1/stats` per-layer mmap residency
**Files**: `crates/larql-server/src/routes/stats.rs`.
Existing `q4k_ffn` block exposes cache slots/bytes; extend with per-layer
hot/cold (resident vs paged-out) so operators can see what `--release-mmap-after-request`
actually buys them.

### F22. Persistent patches
**Files**: `crates/larql-server/src/session.rs`,
`crates/larql-server/src/routes/patches.rs`.
Patches are session-scoped today; no on-disk overlay. Add a durable
`POST /v1/patches/save` + auto-apply on boot. Pairs with **F8** (hot-swap)
so a patched model survives restart.

### F23. Python HTTP client SDK
**Files**: new `crates/larql-python/src/http_client.rs` (or new crate).
`larql-python` is walk-only against a local vindex; no HTTP client. Add a
`pip install larql` package speaking the server's HTTP API (sync + async),
mirroring the OpenAI Python SDK shape. Pairs with **F10** (OpenAI compat) so
the SDK is a thin wrapper over the OpenAI client.

---

## Completed

### 2026-05-02 — F0 closed + N0 slices 1 + 2 (OpenAI compat: models + embeddings + completions + chat completions)

**F0 closed.** `larql run output/gemma4-26b-a4b-q4k.vindex "The capital
of France is" --max-tokens 5` (no `--moe-shards`, no `--metal`) returns
**"Paris."** Local in-process CPU MoE on the per-layer Q4_K hybrid-MoE
vindex now produces the correct answer; the M-CPU kernel work shared
the code path with the 2026-04-30 server-side fix, so the local route
inherited correctness for free. Marked closed under P0 Active.

**N0 slice 1 + slice 2** — four OpenAI-compatible endpoints landed
end-to-end on `larql-server`, live-validated against
`output/gemma3-4b-q4k-streaming.vindex`:

| Endpoint | Slice | Notes |
|---|---|---|
| `GET /v1/models` | 1 | OpenAI `{object: "list", data: [{id, object: "model", created, owned_by: "larql", ...}]}`. Larql-specific extras (`path`, `features`, `loaded`) preserved. |
| `POST /v1/embeddings` | 1 | All four `input` variants (`string`, `string[]`, `int[]`, `int[][]`). Mean-pooled static-embedding lookup. `encoding_format: "base64"` returns 400 (follow-up). |
| `POST /v1/completions` | 1 | Non-streaming; un-KV-cached generation loop. `stream=true` and `n>1` return 400. |
| `POST /v1/chat/completions` | 2 | Multi-turn chat with chat-template auto-detection (Gemma / Llama / ChatML / Mistral / Plain) from `arch.family()`. Same generation path as `/v1/completions`. `tools` / `tool_choice` / `response_format: json_*` / `stream=true` / `n>1` return 400 with clear messages. |

Implementation surface: ~1600 LOC across three new files
(`src/routes/openai_embeddings.rs`, `src/routes/openai_completions.rs`,
`src/routes/openai_chat.rs`) + reshape of `src/routes/models.rs` + 4
routes wired into both single-model and multi-model routers + 23 unit
tests + 19 integration tests + new live `examples/openai_demo.rs`
walkthrough that boots the server in-process via
`tower::ServiceExt::oneshot` and exercises every endpoint.

Live smoke (`gemma3-4b-q4k-streaming.vindex`, port 18081):
- `/v1/models` → OpenAI shape with `gemma-3-4b-it`, `created`, `owned_by`, larql extras.
- `/v1/embeddings input="France"` → 2560-dim pooled vector + correct usage block.
- `/v1/completions max_tokens=5` → wire-correct response (`cmpl-...`,
  `text_completion`, `usage`).
- `/v1/chat/completions max_tokens=8` with system + user → wire-correct
  response (`chatcmpl-...`, `chat.completion`, `choices[0].message.{role:
  "assistant", content}`, `usage`). Output content quality on the
  un-KV-cached path is poor (degenerate greedy on un-trained
  base-decode-without-template); wire is what's verified here.

**Tests** — full sweep:
- `cargo test -p larql-server --lib`: 154 lib tests
- 14 integration files: 392 integration tests
- Total: ~546 tests, 0 failures
- `cargo clippy -p larql-server --tests --no-deps -- -D warnings`: clean
- `cargo fmt -p larql-server -- --check`: clean

**Open follow-ups** (per-item in N0 sub-headers above):
- **Slice 3 (N0.1 SSE)** — `text/event-stream` for both
  `/v1/completions` and `/v1/chat/completions`. Bundles with Q1.10
  (stream.rs reduction) since both touch the same streaming
  state-machine shape.
- **Slice 4 (N0.6)** — constrained decoding for `tools` / `tool_choice`
  / `response_format: json_schema` via JSON schema → GBNF mask.
- **Slice 5 (N0.3)** — `/v1/responses` Responses API, pairs with N1
  stateful sessions.
- **N0.2-fast (shipped 2026-05-02)** — KV-cached generation path now
  live for both `/v1/completions` and `/v1/chat/completions`.
  `LoadedModel.weights` migrated from `OnceLock<ModelWeights>` to
  `OnceLock<RwLock<ModelWeights>>`; OpenAI handlers acquire a write
  guard via `lock_weights_for_gen()` and call
  `larql_inference::layer_graph::generate{,_streaming}` which auto-
  dispatches f16 vindexes to the fused KV-cached path and Q4_K +
  CPU vindexes to the per-step `predict_q4k` fallback. Output on
  Gemma 3 4B: "The capital of France is" → " Paris.\n\nParis is"
  (was " is is is is" pre-fix). Multi-turn chat template rendering
  moved into `larql_inference::prompt::ChatTemplate::render_messages`,
  shrinking the openai handlers further. `bootstrap.rs` now mirrors
  `larql_inference::open_inference_vindex` by loading
  `attn_weights_q4k.bin` + `interleaved_q4k.bin` for inference-capable
  vindexes (without these the Q4_K decode panics).
- **base64 encoding** for `/v1/embeddings` — small follow-up.
- **N0-router** — OpenAI surface on `larql-router` (grid front);
  tracked under "Router-side OpenAI surface" in P1.

### 2026-05-01 (continued) — Q1 code-quality cleanup (9 of 10 items)

The Q1 audit catalogue from earlier the same day, executed in a follow-on
session. All public APIs preserved; existing test surface unchanged.
Q1.10 (stream.rs WebSocket state machine) deferred until N0.1 (OpenAI
Chat Completions SSE) forces a similar shape.

| Item | Outcome |
|---|---|
| **Q1.1** Split `routes/expert.rs` (1044 LOC, 6 concerns) | New `routes/expert/{mod,single,batch_legacy,layer_batch,cpu,metal,warmup}.rs` directory. mod.rs (90 LOC) re-exports the historical public surface (`run_expert`, `run_experts_cpu_batch`, `run_experts_metal_batch`, `warmup_*`, `handle_*`); each sibling file is ~100-225 LOC with one clear concern. `metal.rs` is `#[cfg(all(feature = "metal-experts", target_os = "macos"))]`-gated so non-Metal builds compile clean. |
| **Q1.2** Centralise env-var flags into `src/env_flags.rs` | New module with one `pub const` per `LARQL_*` name + cached presence accessors backed by `std::sync::OnceLock` (process-wide, not TLS — env vars don't change at runtime). Replaced 12 raw `std::env::var(...)` call sites in `routes/expert/*` and `grpc_expert.rs`; removed two ad-hoc `thread_local! { static HTTP_TIMING ... }` blocks. README env-var table now references the same names that show up in `env_flags::*`. |
| **Q1.3 + Q1.9** Shared `wire::has_content_type` | New `src/wire.rs` with `has_content_type(headers, expected) -> bool` (uses `contains` so parameterised types like `application/json; charset=utf-8` match). Replaced 4 inline header-detection patterns in `routes/walk_ffn.rs`, `routes/embed.rs` (×2), `routes/expert/batch_legacy.rs`. 4 unit tests cover exact-match, parameterised, mismatch, and missing-header cases. |
| **Q1.4** Body-size limit constants | `REQUEST_BODY_LIMIT_BYTES = 64 MB` and `REQUEST_BODY_LIMIT_LARGE_BYTES = 256 MB` in `src/http.rs`. Replaced 3 bare literals; `EXPERT_BATCH_BODY_LIMIT` in `routes/mod.rs` now references the same const. |
| **Q1.5** `JSON_CONTENT_TYPE` const | Added to `src/http.rs` next to `BINARY_FFN_CONTENT_TYPE`. Replaced 3 bare `"application/json"` literals across walk_ffn / embed / expert. |
| **Q1.6** Typed `DEFAULT_*` consts | `DEFAULT_PORT`, `DEFAULT_HOST`, `DEFAULT_HNSW_EF_SEARCH`, `DEFAULT_MAX_CONCURRENT`, `DEFAULT_DESCRIBE_CACHE_TTL_SECS`, `DEFAULT_LOG_LEVEL`, `DEFAULT_SESSION_TTL_SECS`, etc. Moved into `bootstrap.rs` (alongside the new `Cli` struct from Q1.8); `clap` now uses `default_value_t = ...`. `SessionManager::new` references the same `DEFAULT_SESSION_TTL_SECS` instead of re-encoding `3600`. |
| **Q1.7** `announce.rs` reconnect/heartbeat consts | `RECONNECT_INITIAL_BACKOFF` / `RECONNECT_MAX_BACKOFF` / `HEARTBEAT_INTERVAL` lifted to module consts; the previous `Duration::from_secs(1) / 60 / 10` magic numbers are gone. |
| **Q1.8** Reduce `main.rs::main` (656 LOC → 26 LOC) | Moved `Cli` struct + `pub async fn serve(cli: Cli)` into `bootstrap.rs`. `main.rs` is now: parse Cli, install tracing, call `bootstrap::serve(cli).await`. Boot orchestration (vindex loading, warmups, listener+TLS+UDS, gRPC, grid announce) is callable from anywhere that wants to drive the server without going through `clap::Parser::parse_from`. |
| **Q1.10** stream.rs reduction | **Deferred** — see P1: Active. Bundling with N0.1 SSE infrastructure when that lands. |
| Tests | 126 → **131 lib tests** (4 new for `wire::has_content_type`, 1 for `env_flags::names_are_larql_prefixed_and_unique`); 37 integration tests unchanged; ~580 tests across lib + integration, 0 failures. |
| Clippy | `cargo clippy -p larql-server --tests --no-deps -- -D warnings` clean. |
| `cargo fmt -p larql-server -- --check` | Clean. |

LOC delta (per-file):

| File | Before | After |
|---|---|---|
| `main.rs` | 656 | **26** |
| `bootstrap.rs` | 464 | 1073 (Cli + serve moved in) |
| `routes/expert.rs` | 1044 | (deleted) |
| `routes/expert/mod.rs` | — | 90 |
| `routes/expert/single.rs` | — | 155 |
| `routes/expert/batch_legacy.rs` | — | 105 |
| `routes/expert/layer_batch.rs` | — | 226 |
| `routes/expert/cpu.rs` | — | 195 |
| `routes/expert/metal.rs` | — | 204 |
| `routes/expert/warmup.rs` | — | 140 |
| `env_flags.rs` (new) | — | 122 |
| `wire.rs` (new) | — | 64 |

The bulk of the `bootstrap.rs` size growth is the Cli struct (~200 LOC of
clap doc-comments + `#[arg]` attributes) and the `serve` function body
that used to live in `main`. The orchestration is unchanged; only its
location moved.

### 2026-05-01 — HTTP CPU-path optimisations + UDS transport + layer-batch wire

End-to-end ~17.7 → ~19.7 tok/s on Gemma 4 26B-A4B (M3 Max, single local
gRPC shard, 100-token poem). Per-call HTTP overhead dropped from ~660 µs
to ~460 µs on gRPC streaming, ~510 µs on UDS, ~660 µs on TCP HTTP (now
with TCP_NODELAY). All optimisations preserve bit-exact semantics
(verified by output equivalence on the same prompts).

| Item | Outcome |
|---|---|
| **`POST /v1/experts/layer-batch`** new endpoint | One residual + K (expert_id, weight) pairs → one router-weighted-sum response. Replaces the K-residual-copies legacy `/v1/expert/batch` for the common-case `forward_moe`. Saves ~2.6 MB/token of redundant wire data + K-1 redundant `pre_experts_norm` + Q8_K quants on the server. |
| **`POST /v1/experts/layer-batch-f16`** new endpoint | f16 variant — halves wire bytes (5.5 KB request + response). Opt-in via `LARQL_MOE_WIRE_F16=1` for LAN deployments. f16 conversion CPU cost (~9 µs/call) cancels the wire saving on loopback; expected +3-5% gain on 1 Gbps Ethernet. |
| **Unix Domain Socket transport** (`--uds-path`, `unix://` URL) | Hand-rolled HTTP/1.1 over `UnixStream` (no new dep). Saves ~150 µs/call on loopback (~3% end-to-end). Persistent stream behind a `Mutex`, lazy reconnect on disconnect. Same wire format as TCP HTTP, so f16 + layer-batch semantics carry through unchanged. |
| **TCP_NODELAY on accepted connections** | `axum::serve::ListenerExt::tap_io` hook calls `set_nodelay(true)` per accept. Defensive against tail-packet stalls (40-200 ms on Linux/BSD delayed ACK) on real LAN; within noise on loopback. |
| **gRPC SPLIT default-on for gRPC shards** | Streaming fire/collect overlap now default for `grpc://` shards. Reliably ~12% steady-state win on M3 Max loopback (re-measured 19.5 vs 17.7 tok/s, alternating-cooled). The historical "20 → 4 tok/s catastrophic regression" warning predates the Metal MoE accuracy fix and the predispatch refactor; under thermal pressure both unary + SPLIT regress similarly, but stable-state SPLIT wins. Set `LARQL_MOE_NO_SPLIT=1` to opt out. |
| Per-call timing instrumentation | `LARQL_HTTP_TIMING=1` (server: decode / spawn_overhead / compute / encode µs; client: encode / send_total / recv_body / decode µs). `LARQL_MOE_TIMING=1` (per-token: per-layer route+fire / collect / server compute estimate / network estimate). Used for the diagnostic round that found `__powisf2` libcall in the f16 decode hot path (now bit-manipulated). |
| Test suite restored | 7+ test files had `LoadedModel { ... }` literals missing the `unit_filter` field added recently — all 9 LoadedModel literal sites in tests/ + tests/common/ patched. Test count went from 119 lib-only (broken integration tests) to **494 total across lib + 14 integration test files, all green**. |
| README + docs updated | `README.md` rewrite: new headline mentioning MoE grid as first-class use case, full env-var reference table, refreshed CLI Options with `--uds-path`/`--units`, rewritten "Remote MoE shard topology" recipe with current numbers, new `/v1/experts/layer-batch[-f16]` API section, accurate Crate Structure (28 source files vs the 16 the doc previously listed). `docs/server-spec.md`: §4.5 Remote MoE Expert Endpoints added, §13.4 dropped "planned" status, §10.2 fly.io references `F-FLY`. |
| `bench_expert_server` re-validated | Refreshed numbers in the Live perf snapshot section above. `cpu_moe_forward` floor 0.10 → 0.37 ms (the 0.10 was a buggy measurement on empty buffers — see prior compute ROADMAP). `forward_moe` warm 1.91 → 0.80 ms. 30-layer sweep 56 → 24.8 ms. RSS unchanged at ~10.5 GB. |

Tried-but-reverted (kept in source for future hardware where the trade
may flip):
- `tokio::task::block_in_place` instead of `spawn_blocking` — server-side
  faster (no transition cost) but tokio kept spawning replacement OS
  workers when every request blocked, regressing sweep ~0.3 ms.
- f16 wire as default — within noise on loopback (CPU conversion cancels
  wire saving); kept as opt-in for LAN.

### 2026-05-01 (continued) — larql-server review pass

Same calendar day, separate session. Audit + fixes across the entire
larql-server crate to land a clean baseline alongside the perf work.

| Item | Outcome |
|---|---|
| Test suite restored | 7+ stale `LoadedModel` test fixtures + 1 stale `PatchOp` example fixture missing recently-added struct fields. All 9 LoadedModel literal sites + 1 PatchOp site patched. **Test count went 119 lib-only → 501 across lib + 14 integration files; all green.** |
| `bench_expert_server` extended | New `--uds` and `--wire f32\|f16` flags. Spawns server bound to both TCP and UDS so the bench can A/B per-call cost. Confirmed UDS gives ~10% loopback win (0.82 → 0.74 ms `forward_moe` warm); f16 is a clear LOSS on loopback (1.05 ms — CPU conversion dominates) but expected to win on LAN. |
| README rewrite | Added env-var reference table, `/v1/experts/layer-batch[-f16]` API section, "Remote MoE shard topology" recipe with current numbers, accurate Crate Structure (28 source files vs the 16 the doc previously listed), "What's coming" section pointing to N0..N6 + F-FLY. ~880 → ~1110 LOC. |
| `docs/server-spec.md` updated | §3 CLI flags get `--uds-path` / `--units` / `--warmup-walk-ffn` / env-var section. New §4.5 Remote MoE Expert Endpoints (full layer-batch + f16 + transport coverage). §13.4 dropped "planned" status. §10.2 fly.io references `F-FLY`. |
| ROADMAP additions | New "Great new functionality" section (N0..N6) at the top — N0 is OpenAI API compatibility (chat completions + completions + responses + embeddings + models), highest-leverage item. F-FLY at top of P0: Active. F0 status updated (server path correct, local in-process TBD). Q1 (code-quality review) added at P1 with 10 sub-items targeting modularity + magic literals. |
| `cargo clippy -p larql-server --tests --no-deps -- -D warnings` | Was failing on 6 errors (manual `is_multiple_of`, `let_unit_value`, dead env-var unpacks, `path_used` unused initial assignment). All fixed. Server-only clippy now clean. |
| `cargo fmt -p larql-server -- --check` | Clean. |
| Coverage | 69.24% line / 75.64% function via `cargo llvm-cov`. Slight regression from 74.2/81.2 baseline attributable to new code added without proportional tests; mitigated by adding `topology.rs` tests (3) + `routes/expert.rs` `layer_batch_wire_tests` mod (4). |
| Code-quality findings catalogued | New Q1 section in ROADMAP with 10 concrete items (Q1.1 split `routes/expert.rs` 1049 LOC, Q1.2 centralise env flags into `src/env_flags.rs`, etc.) — all with file:line references and effort estimates. Total ~7-8 hours for the full sweep. |
| README + ROADMAP doublecheck | Fixed `gemma3-4b.vindex` references (file doesn't exist; replaced with `gemma3-4b-v2.vindex` which does), removed stale `ADR-009` reference (no such file), harmonised the two perf reference tables (Examples vs Recommended setups now reference each other), updated stale "2026-04-26" date stamp. |

### 2026-04-26 — Per-expert byte table refactor + `experts_packed.bin` removal

`MoeLayerWeights.experts_{gate_up,down}` migrated from `&[u8]` (monolith +
`expert_idx * stride` arithmetic in the compute path) to `Vec<&[u8]>`
(per-expert slice table). The CPU MoE consumer (`cpu_moe_forward` and
`run_single_expert{,_with_norm}`) now indexes by expert id directly, with
format dispatch (BF16 vs Q4_K) at the cache layer.

| Item | Outcome |
|---|---|
| `larql-compute` | `cpu/ops/moe/{cache,expert,forward,mod}.rs` and `pipeline.rs::MoeLayerWeights`. `cached_dequant(bytes, format, expected_floats)` dispatches BF16/Q4_K. `expert_byte_slice` deleted. Tests updated. 94/94 pass. |
| `larql-vindex` | `cpu/ops/q4_common.rs::dequantize_q4_k` lifted to module scope so the compute crate can dequant Q4_K without a `larql-models` dependency. |
| `larql-inference` | `build_moe_weights` builds per-expert tables from either `weights.get_layer_entry_bytes(...)` (per-layer Q4_K) or BF16 stride slicing (legacy). `QuantFormat` re-exported. |
| `larql-server` | `routes/expert.rs::run_expert` resolves per-expert bytes through whichever path the vindex provides; honours `expert_filter` ownership. `tests/test_expert_endpoint.rs` updated to slice synthetic monoliths into per-expert tables. 4/4 parity tests pass. |
| 26B-A4B vindex | `weight_manifest.json` stripped of `packed_bf16` rows for experts (60 → 421 entries). `experts_packed.bin` deleted (43 GB freed; vindex 58 → 16 GB). |
| Bench parity | `bench_expert_server` re-runs end-to-end against the per-layer-only vindex. `forward_moe` warm latency unchanged at 1.91 ms (was 1.93 ms when monolith was still on disk). 30-layer sweep at 56 ms (cold-page sweep on BF16 monolith was 866 ms). |

`bench_expert_server` and the parity tests both detect the format
automatically (`weights.has_per_layer_ffn()`); legacy BF16 vindexes still work
unchanged. Future MoE vindexes only emit per-layer files — the q4k extractor
at `format/weights/write_q4k/mod.rs` already does this.

### 2026-04-30 — gRPC grid: end-to-end accuracy

The grid produced semantically wrong text on Gemma 4 26B-A4B-it ("The capital
of France is **not specified in the text**…") despite each shard correctly
running its expert FFN. Root cause was on the **client** side
(`larql-inference::layer_graph::grid`) — chat-template handling, detokeniser,
EOS detection, and special-token suppression — not the shard server. The
server work here was confirming the contract: shards return correct expert
outputs given the right top-K input. Documenting for future grid changes.

| Item | Notes |
|------|-------|
| Server shards verified correct | A 2-shard split (experts 0-63 on `:9081`, 64-127 on `:9082`) running against the unit manifest serves expert outputs that, when combined client-side with the proper detokenisation + EOS + special-token suppression + default system prompt, produce "**Paris**" as the answer |
| Shard contract: per-(layer, expert) ownership via `--units` | The `parse_unit_manifest` path is what the client's `--moe-units-manifest` resolves against; ownership is the strict source of truth and `forward_moe_seq` rejects layers/experts not owned by any shard |
| Decode throughput (loopback, M3 Max) | 2.3 tok/s end-to-end on the 26B-A4B with two shards in the same process — expected to climb meaningfully when shards run on separate hosts (less GPU contention with the client) |

### 2026-04-30 — Metal expert dispatch: 3.7× speedup found, blocked on kernel bug

`LARQL_MOE_TIMING=1` showed the grid bottleneck is **server compute = 95%** of token wall time (network = 2%, route+fire = 3%). Per layer: 8.36ms server / 0.18ms net. Each shard runs its 4 picked experts (gate + GELU + down) on CPU-rayon BLAS — that's where the time goes. Sub-arc:

| Item | Notes |
|------|-------|
| Bottleneck localised | CPU experts = 250ms/token (95%) on the loopback 2-shard setup. Network = 5ms (2%). The grid-side overhead is negligible — accelerating the shard's expert math is the only meaningful lever |
| `--features metal-experts` measured: **3.7× speedup** | Server with Metal expert dispatch: 264ms → 117ms per token, 2.3 tok/s → **9.4 tok/s** (preselected path → 11.2 tok/s). Significant — server compute drops from 250ms → 115ms |
| **Accuracy bug blocks shipping** | Metal expert kernel (`MetalBackend::run_experts_preselected_metal` and `_prestaged_metal`, both routes) produces numerically wrong outputs for Gemma 4 26B-A4B-it MoE shape (cos≈0.7 vs CPU, \|metal\|≈70% of \|cpu\|). End-to-end output: "**Paris**" via CPU vs "answer is in the context" via Metal. Same kernels are correct for dense FFN at inter=2560/10240/21504 — bug is specific to MoE inter=704 dispatch |
| Workaround: default to CPU even on metal-experts builds | `run_experts_metal_batch` now early-returns `None` unless `LARQL_USE_METAL_EXPERTS=1` is set. Shipping correctness over speed; the Metal path stays opt-in for kernel-debug runs |
| Diagnostic: `LARQL_METAL_VS_CPU_DEBUG=1` | Server-side per-call A/B compare in `run_experts_metal_batch` — runs both Metal and CPU on the same input, prints max\|Δ\|, \|metal\|, \|cpu\|, cos. Ready to use when someone digs into the kernel |
| See also | `larql-compute/ROADMAP.md` "Open: Metal MoE expert kernel — accuracy bug at inter=704" for the kernel-side investigation plan |

### 2026-04-26 — examples, synthetic benchmark, grid checks

| Item | Outcome |
|---|---|
| `server_demo` | Runs locally with synthetic data; fixed invalid probe-label JSON comma output and updated rate-limit text for `--trust-forwarded-for`. |
| `embed_demo` | Runs locally with synthetic embed/logits/token responses and binary-wire examples. |
| `server_bench --release` | Synthetic benchmark completed: `gate_knn` top-5 0.022 ms/op, 8-layer `walk` 0.203 ms/op, single-layer `walk-ffn` 0.032 ms/op, batched 8-layer `walk-ffn` 0.321 ms/op, describe simulation 0.298 ms/op, 512-token embed prefill 0.114 ms/op. |
| `bench_embed_server` | Example builds under `cargo check -p larql-server --examples`; execution requires a real vindex path. |
| Grid unit coverage | Added `GridState` tests for inclusive ranges, default single-model routing, least-loaded replica selection, deregistration, batched gap reporting, and status gaps. `cargo test -p larql-router` now runs 20 tests. |
| Docs | Updated server README examples/benchmarks/testing, router README validation, and router spec validation commands. |

### 2026-04-26 — coverage round-6 (embed + walk-ffn reachable gaps)

| Item | Outcome |
|---|---|
| `routes/embed.rs` modularity | Extracted binary embed/logits parse helpers and binary embed response encoder |
| `routes/embed.rs` coverage | **66.7% → 86.5% line**, **70.7% → 86.3% function** |
| `routes/walk_ffn.rs` coverage | **76.7% → 79.5% line**, **77.3% → 82.0% function** |
| Tests | 458 → **478** tests |
| Coverage | **71.9% → 74.2% line**, **78.9% → 81.2% function** |

### 2026-04-26 — modularity + coverage round-5

| Item | Outcome |
|---|---|
| Boot/loading modularity | Moved parse/discovery/vindex-load helpers out of `main.rs` into `bootstrap.rs`; binary now keeps CLI orchestration while library code is directly testable |
| `routes/stream.rs` | Extracted pure `stream_describe_messages`; describe stream behavior can be tested without a WebSocket client |
| `routes/infer.rs` | Extracted mode selection and prediction formatting helpers |
| `routes/explain.rs` | Extracted band mapping, probability/gate/attention rounding, prediction formatting, and lens formatting helpers |
| Clippy | Server-local clippy clean with `--no-deps`; full dependency-checking command is blocked by existing `larql-vindex` warnings |
| Coverage | **69.2% → 71.9% line**, **77.1% → 78.9% function** (458 tests) |

### 2026-04-26 — coverage round-4 (T2 reachable gaps)

| Item | Outcome |
|---|---|
| `embed_store.rs` | 25% → **98% line** with tiny f16 mmap fixtures and L1 cache behavior tests |
| `announce.rs` | 6% → **56% line** by extracting/test-covering announce, heartbeat, dropping, and bearer helpers |
| `main.rs` | 0% → **23% line** with binary unit tests for parse/discovery/serve-alias helpers |
| `routes/stream.rs` | 0% → **28% line** with pure WebSocket message shape builders |
| `routes/infer.rs`, `routes/explain.rs` | Default/request deserialization coverage added; full paths remain weight-gated |
| Coverage | 63.9% → **69.2% line**, 73.4% → **77.1% function** (430 → 458 tests) |

### 2026-04-26 — coverage round-3 (T2 partial) + magic strings round-2

| Item | Outcome |
|---|---|
| `test_grpc.rs` — 28 new gRPC handler tests | Direct method calls on `VindexGrpcService` — no network socket; health, stats, describe, walk, select, relations, walk_ffn, infer, stream_describe |
| `grpc.rs` coverage | 0% → **65%** (169 lines uncovered, all gated on real model weights or gRPC streaming) |
| Magic strings — `"probe"` | `PROBE_RELATION_SOURCE` constant in `band_utils.rs`; used in describe.rs, grpc.rs, stream.rs |
| Magic strings — `"ok"` | `HEALTH_STATUS_OK` constant; used in grpc.rs health handler |
| Magic strings — gRPC modes | `INFER_MODE_WALK/DENSE/COMPARE` applied to grpc.rs (was using bare strings) |
| Magic strings — WebSocket types | `WS_TYPE_ERROR/LAYER/DONE/PREDICTION/INFER_DONE` and `WS_CMD_DESCRIBE/INFER` in stream.rs |
| Coverage | 57.2% → **63.3% line**, 65.3% → **73.2% function** (402 → 430 tests) |

### 2026-04-26 — coverage round-2 (T1)

| Item | Outcome |
|---|---|
| `functional_tokenizer()` in common | WordLevel tokenizer (France→0, …) added to test infra; unblocks describe/walk/walk-ffn body paths |
| `test_http_full_routes.rs` | 39 new HTTP integration tests exercising full describe/walk/walk-ffn code paths |
| `test_unit_band_utils.rs` | 13 pure unit tests for `band_utils.rs` constants + helpers |
| Infer + ratelimit branches | `infer_disabled=false` model builder; ratelimit middleware axum tests |
| Coverage | 49.1% → **58.0% line**, 56.4% → **65.3% function** (345 → 402 tests) |

### 2026-04-26 — code quality round-1

| Item | Outcome |
|---|---|
| Modularity — deduplicate `session_id()` | 3 identical private fn definitions → 1 `pub fn extract_session_id` in `session.rs` |
| Modularity — `get_layer_bands()` / `filter_layers_by_band()` | 5 / 3 duplicated blocks → `src/band_utils.rs` |
| Modularity — `model_or_err()` | 25 repeated `ok_or_else(NotFound)` sites → `AppState::model_or_err()` |
| Modularity — `elapsed_ms()` | 20 repeated latency-rounding expressions → `src/state::elapsed_ms()` |
| Magic strings — band names | `"syntax"/"knowledge"/"output"/"all"` → `BAND_*` constants in `band_utils.rs` |
| Magic strings — infer modes | `"walk"/"dense"/"compare"` → `INFER_MODE_*` constants |
| Magic strings — insert modes | `"constellation"/"embedding"` → `INSERT_MODE_*` constants |
| Magic strings — patch names | `"unnamed"/"inline-patch"` → `PATCH_UNNAMED`/`PATCH_INLINE_NAME` constants |
| Magic strings — HTTP headers | `"x-session-id"` → `HEADER_SESSION_ID`; `"etag"/"cache-control"/"if-none-match"` → axum `header::*` |
| Test restructure | `test_api.rs` (2600 L) + `test_http.rs` (1400 L) → 10 focused files (100–350 L each) + `tests/common/mod.rs` |
| Coverage baseline | 39.7% → **49.1% line**, 41.6% → **56.4% function** (345 tests, 0 failures) |

### 2026-04-26 — perf round-1 (G1+G2+G3)

| Item | Outcome |
|---|---|
| G1 cold-start profile | Two-phase: 1.27 s lazy weight load + 17 ms/layer mmap page-in. Warm steady state 0.2–0.3 ms/layer. |
| G2 `/v1/warmup` + `--warmup-walk-ffn` | First walk-ffn 1247 ms → 12.6 ms (99×). Boot trades ~1.3 s + 3.2 GB pre-allocation. HTTP endpoint also exposed for live re-warm. |
| G3 self-assembling gRPC grid | Live-validated `--grid-port` + `--join`: auto-join, coverage tracking, graceful failure (clean HTTP 400 on uncovered layer), auto-recovery on rejoin. |

### 2026-04-26 — W2 retrofit + grid validation

| Item | Outcome |
|---|---|
| `--warmup-hnsw` flag | Eager-builds HNSW across owned layers at boot via `warmup_hnsw_all_layers()`. Reports correct owned-layer count under `--layers`. |
| Boot log: W2 status | `Down features Q4K: loaded (W2 — per-feature decode skips q4k_ffn_layer cache)` when `down_features_q4k.bin` is present. |
| `/v1/stats.q4k_ffn` field | `{cache_slots, cache_bytes, feature_major_down}` — operators can verify W2 active + cache empty in steady state. |
| `larql convert add-feature-major-down` | New CLI subcommand. Retrofits an existing Q4K vindex without re-quantising the rest. 30 layers / 152 MB / 1.12 s on Gemma 26B. Idempotent. |
| Live grid validation | 2-shard layer-range split (0-14 + 15-29) on real 26B vindex, full fan-out via router, 8-way concurrent stress, 0.2 ms warm per-layer, 5.9 ms full-30-layer fan-out. |

### Pre-2026-04-26 — foundations (already in place)

- HTTP API: `/v1/walk`, `/v1/walk-ffn`, `/v1/stats`, `/v1/health`,
  `/v1/infer`, `/v1/insert`, `/v1/expert/{layer}/{id}`, etc.
- `--layers START-END` shard slicing (mmap pages outside range stay
  paged out, RSS proportional to shard size).
- `--max-q4k-cache-layers` LRU bound on the legacy Q4K dequant cache.
- `--ffn-only` / `--embed-only` mode flags.
- gRPC self-assembling grid (`--grid-port` / `--join` / `--grid-key`).
- Bench rig daemon-aware (`larql-vindex` benches refuse if a server
  shares the host; override with `LARQL_BENCH_ALLOW_DAEMONS=1`).
