# Changelog — larql-server

All notable changes to `larql-server` are documented here.

The format follows the conventions of [Keep a Changelog](https://keepachangelog.com/),
with dated entries (`YYYY-MM-DD`) instead of semantic versions during the
pre-1.0 phase. Forward-looking work lives in [`ROADMAP.md`](ROADMAP.md).

## [2026-11-07] — `utoipa` 6 and `utoipa-swagger-ui` 10

Issue #662. `utoipa-swagger-ui` 10.0.2 builds Swagger UI with `zip` 9, which removes
the last `zip` < 9 (3.0.0, a build-dependency) from the lockfile. It requires
`utoipa` 6; the derives and `SwaggerUi` router here compile unchanged.

## [2026-08-23] — V3-SERVE-2: batched prefill populates KV; the next cost is now visible

`PlanBackend::attention` returns `AttentionOut { outputs, keys, values }`.
The batched realisation always computed the conditioned K/V rows — it
must, to attend at all — and then discarded them, which is what forced a
caller wanting a populated cache down the per-position path. "I want KV"
no longer implies "run attention one position at a time".

**Proven before the trait moved.** `exec/tests/attention_kv_parity.rs`
showed the two realisations agree **bit-for-bit** on K, V and outputs —
reference (the semantic anchor, sharing no arithmetic with
`larql-compute`) and production, every layer, and all 40 layers of a
real Granite container. With a control that fires on a perturbed input,
because `max_abs = 0` everywhere looks the same as an inert harness.

**Measured** (Granite 4.1 3B, battery, `prefill_into`):

| prompt | post-2B | post-2C | gain |
|---|---|---|---|
| 5 | 0.448 s | 0.442 s | 1.01x |
| 64 | 4.099 s | 3.428 s | 1.20x |
| 325 | 34.587 s | 25.024 s | **1.38x** |

Prefill rate 8.56 → **12.09 tok/s** (64 → 325). No change at n=5 with a
gain that grows in n is the signature of fixing a per-position
realisation; drift would have moved all three points.

**What this exposed.** Same session, same power state, the CLI's
batched path runs 19.50 tok/s over 64 → 256 where the server's prefill
runs 11.37 — **1.72x** still outstanding. (The 21.4 tok/s CLI figure
quoted earlier was taken on AC; re-measuring it on battery is what makes
the comparison honest.) Both now run the same batched attention, so the
remainder is what the server does *around* it — `kv.append` per position
per layer and whatever `CanonicalKvState` costs to store a row. That was
invisible while per-position attention dominated, and it is the next
thing to measure.

**Deliberately not done.** A *resumed* prefill still steps: a batched
pass conditions position `p` as the `p`-th token of the sequence it is
handed, so it cannot express a prefill starting part-way through one.
The continuation-parity gate now straddles both branches — the
whole-prompt arm takes the batched path, the split arm's second chunk
takes the stepped one, and they must agree bit-for-bit.

N1 post-2C: 20.75 s → 8.63 s (**2.40x**) on the frozen ledger, with the
cached prefix still resuming (25 / 50 / 71 tokens).

## [2026-08-23] — N1 measured post-2B: 2.4x, and a claim of mine corrected

Taken **before** 2C, because 2B has created an intermediate state that
2C destroys: fixed setup gone, slow prefill still present. Without this
row it would be impossible to separate "N1 became useful because 2B
removed the fixed tax" from "N1 behaves differently because 2C changed
prefill".

Four-turn `/v1/responses` chain, cache on vs off, min of two chains,
*on battery*:

| model | cache off | cache on | saving | speedup | reused |
|---|---|---|---|---|---|
| gpt-oss-20b (ChatML) | 45.02 s | 18.77 s | 26.25 s | **2.40x** | 254 tok |
| granite-4.1-3b (Plain) | 15.94 s | 7.69 s | 8.25 s | **2.07x** | 146 tok |

The per-turn shape is the signature, and it is model-independent:

```text
cache OFF   4.33 -> 8.81 -> 13.49 -> 18.38 s    linear — re-prefills the whole conversation
cache ON    4.31 -> 4.96 ->  4.57 ->  4.92 s    flat   — prefills only the new turn
```

Against the pre-2B row (2026-08-22, gemma-2-2b: 41.75 s vs 41.14 s, a
1.5% difference inside the noise floor), N1 has gone from unmeasurable
to the largest per-request win available on a chained workload. It was
never broken; it was masked by ~7 s of per-request model
materialisation. Note the rows use different models — gemma-2-2b is no
longer on disk — so the cross-row comparison is qualitative; the
within-row A/B and the flat-vs-linear shape are not.

The workload is now checked in as
`examples/n1_continuation_ledger.rs`, which runs both arms in-process
through the real router. Its value is comparability across rungs, so
the turns, output budget and protocol are frozen — a future rung must
re-run *this*, not a re-invented equivalent. It reproduces the numbers
above from the ad-hoc script within ~2% (2.44x / 2.03x) with
`cached_tokens` matching exactly, which is what says the frozen version
measures the same thing.

**Correction.** This run falsified something recorded here twice
yesterday: that Granite "gets 0% because it falls to the Plain
template". Granite resumed cleanly here (cached 25 / 50 / 71). The
accurate claim is narrower — Plain has no atomic turn terminator, so
whether the seam survives depends on the **last token the model
actually emitted**; it *raises the risk* of a break rather than
guaranteeing one. Yesterday's zero came from replies that happened to
end mid-markdown (`**Instruction**:`), not from the template alone.
Template resolution is still worth fixing for output quality, and the
bind-time warning stays — but it should not be described as the cause
of a resumption failure.

## [2026-08-23] — residency and mutation land on one operand authority

**V3-SERVE-1 confirmed on the re-extracted Granite 4.1 3B** (40 layers,
5 in / 1 out, both arms in one process, *on battery* — the original
7.23 s baseline was on AC, so read the absolute seconds as ~10%
pessimistic; the ratio and the decomposition are unaffected):

| | load per call | prepared |
|---|---|---|
| `prefill_into` | 4.15 s | 0.45 s |
| `session_with_kv` | 3.25 s | **0.000 s** |
| decode | 0.13 s | 0.13 s |
| **warm request** | **7.46 s** | **0.576 s** |

**12.95×**, one-off preparation 3.21 s at boot. Arm A reproduces the
original 7.23 s, so the baseline is intact.

The important *negative* result, which is what makes 2C an independent
problem rather than a mixed one: **residency removed a constant and
left the prompt-length slope alone.**

```
prefill_into      arm A     arm B     A-B        rate      arm A   arm B
  5 tokens        4.152     0.448    3.704     5 ->  64    19.61   16.16
 64 tokens        7.161     4.099    3.062    64 -> 325     9.07    8.56
325 tokens       35.931    34.587    1.344
fixed prefill term            3.897 s -> 0.139 s
```

The slope did not improve (the small differences are noise, and the run
was on battery), because residency does not touch per-token
arithmetic — and it still halves with prompt length, which is the
per-position attention realisation. At 325 tokens a request is 34.7 s,
**99.6% of it prefill**: time-to-first-token is now entirely
prefill-bound. That is V3-SERVE-2's target, measured cleanly and
separately.

The re-extracted container also **ships its own `tokenizer.json`**, so
it serves with no overlay — `/v1/completions` answered " Paris" in
0.434 s over HTTP. Note this is a property of that artefact, not yet of
the encoder: `vindex3 encode` still does not write one by construction
(main's `compact` / `compile` paths preserve an existing one). V3-SERVE-4
stays open until the encoder guarantees it and a gate serves a container
the encoder produced.


Rebased onto main, which had landed the **operand-source seam**
(`9a8c627e`): execution resolves operands through an `OperandSource`
(base store + optional `OperandOverrides`) rather than the store
directly. That is the *overlay* half of the same code the residency
work rewrote, and the two compose rather than compete:

```text
base representation + logical overlay → OperandSource → PreparedOperands → executor
```

`OperandSource` decides what the effective model **means**;
`PreparedOperands` decides how that effective model is **represented
for execution**; the executor sees neither mutation nor storage
mechanics. `PreparedOperands::load` therefore takes
`impl Into<OperandSource>`, so a prepared image is *the effective
operands for that source*.

**Staleness invariant.** A prepared image is now a compiled derivative,
so it must be able to say which source it describes — otherwise it
outlives an overlay mutation and quietly keeps executing the pre-edit
model, becoming exactly the second authority the operand seam exists to
prevent. `OperandOverrides` carries a process-unique identity and a
generation bumped on every mutation, `OperandStore` carries an
identity, and `OperandSource::stamp()` combines them into a
`SourceStamp` that preparation records. `PreparedOperands::
is_current_for` / `ensure_current_for` answer the question.

Deliberately conservative: a clone takes a fresh identity, and
reverting an edit yields a new generation, so a *valid* image can be
judged stale (costing one re-preparation) while a stale one can never
be judged valid. What is deliberately **not** a difference: an empty
overlay is the bare store, so an image prepared from either is current
for both — the stamp tracks effective sources, not the syntax used to
build them. Gated both ways.

**`PreparedVindex3` keeps the operand store.** It looked like a small
merge decision; it is what lets a prepared model derive further images
— attention, FFN, expert, layer-range, or an overlay-specific one —
without reopening the container, and keeps `knowledge_view` available
on a prepared model. The compiler input is preserved alongside the
compiled image.

Also: main's rust-1.98 CI fix cleared the `larql-boundary` /
`larql-models` lint noise, which exposed two genuine lints in this
branch's own code — `large_enum_variant` on the operand slot and
`result_large_err` on the router proxy. Both boxed; workspace clippy is
clean across all seven crates for the first time on this branch.

Gates: vindex 2938, inference 1788, kv 1279, server 1145, lql 1078,
router 256 — all green; fmt and clippy clean; server coverage included
93.18%.

## [2026-08-22] — V3-SERVE-1: prepared execution state (42.7x on a warm request)

The server was loading the model **twice per request**. It now lowers a
container's operands into the backend's execution form once, at bind
time, and every request reads that one image.

**The boundary.** New `PreparedOperands` in `larql-vindex`
(`opplan/exec/prepared.rs`) and `PreparedVindex3` in `larql-inference`:
plan + backend + resident operands, owned at model lifetime. A request
contributes only continuation state. Deliberately *not* a cache inside
`OperandStore::load` — residency is a fact about a served model, and
hiding it behind a memoised loader would leave device placement,
accounting, slicing and eventual overlay composition with nowhere to
live. `DecodeSession` did **not** become long-lived; it stays
per-request and borrows the image.

**Both traversals consume it.** The batch path (`traverse` →
`execute_layer`) previously called `store.load(...)` per layer — and per
*position* for norms, so a 325-token prompt on a 40-layer model loaded
norm weights ~39,000 times. It now reads the same prepared layers the
decode session does, which is what makes "prepared once" true rather
than "prepared twice, in two places".

**Slicing is in the type from the start.** `PreparedOperands::load`
takes an `ExecutionSlice` (`Full`, or `LayerRange { start, end }`) and
lowers only that slice's operands; a slice the plan cannot satisfy is
refused rather than truncated. A sliced image carries no embedding
table or head, and executing token ids against one is refused — it
consumes hidden states. This is the seam the decoupled surfaces grow
from (ROADMAP §V3-DECOUPLE), placed now so residency does not have to
be rebuilt around it later.

**Measured, gpt-oss-20b (20 B MoE, 24 layers), same container and
backend, both arms in one process:**

| | load per call | prepared |
|---|---|---|
| `prefill_into` | 17.79 s | 0.61 s |
| `session_with_kv` | 13.82 s | **0.000 s** |
| decode | 0.33 s | 0.14 s |
| **warm request** | **31.93 s** | **0.75 s** |

**42.7x**, with a one-off 13.5 s preparation at boot. Over HTTP a warm
5-token / 1-token request is now **0.543 s** (decode 7.77 tok/s) — the
multi-second construction floor is gone.

**Gates** (`opplan/exec/tests/residency.rs`, 9 tests): preparation
once, asserted through a new `OperandStore::load_count` rather than a
stopwatch — serving 5 requests over a prepared image moves the counter
by zero — plus the counterfactual that the unprepared entry point loads
on *every* call, so the gate cannot pass against an inert store.
Request parity (logits, final hidden, and every layer's KV rows
identical between paths), continuation parity (chunked prefill ≡ whole
prefill), session isolation (interleaved sessions over one image do not
disturb each other), batch-traversal parity plane by plane, and the
slice refusals.

**Also fixed:** yesterday's V3 template resolution read the container
family through a second `Vindex3Container::open` and silently defaulted
to `""` when that failed — so `gpt-oss-20b`, which declares
`family: "gpt_oss"`, was still being served with the Plain fallback.
The family now comes from the inspection the runtime already performs.
Caught by running against a real container, not by a fixture.

## [2026-08-22] — V3 serve: reality check against real containers

Ran the V3 serve path against **real** VINDEX3 containers
(`granite-4.1-3b`, `gemma-2-2b`) rather than synthetic fixtures. What
follows is what that found and what was fixed; the perf rung it
identified is in [`ROADMAP.md`](ROADMAP.md).

**Confirmed working.** V3 inference over `/v1/completions`,
`/v1/chat/completions` and `/v1/responses` on a real 3B model, with
conversation chaining. N1 KV resumption engages for real on a
properly-templated model: a gemma chain served 26/41 then 52/67 prompt
tokens from resident KV (`resumptions: 2`).

**Found: no real V3 container is servable as shipped.**
`load_v3_model` requires `<container>/tokenizer.json`, and the V3
encoder never writes one — the V3 CLI is deliberately id-level, so
nothing needed it before. Every V3 serve test writes a *synthetic*
tokenizer into its fixture, which is exactly why the gate could not
catch this. Not fixed here; it needs the encoder to carry the
tokenizer, and a gate that serves a container the encoder produced.

**Fixed: `/v1/stats` 404'd on a V3-only server.** The handler resolved
V2 only, so the `server` block — the sole surface carrying the N1
continuation counters — was unreachable on exactly the deployments N1
runs on. It now answers with the program's own shape read from the
opened plan, and does not fake the V2 vocabulary onto a V3 binding.

**Fixed: V3 load options were silently ignored.** `--layers 0-9` on a
40-layer container started fine and served the *whole* model with
complete answers; `--ffn-only`, `--embed-only` and `--no-infer` did the
same — `--no-infer` did not disable inference. `load_artifact`'s V3
branch now **fails closed** on every option it cannot honour, naming
the option and why. V2 cache knobs still pass.

**Fixed: V3 chat-template resolution ignored the container.** The model
id is just the directory basename, so `for_model_id` answered `Plain`
for any container not named after its family — Granite among them.
Resolution now reads the container's declared `family` first, then the
id, then `Plain`, and **warns at bind time** when it lands on `Plain`.
This is not cosmetic: `Plain` ends an assistant turn with a bare
newline, so its last token re-tokenises differently once the next turn
follows, which breaks N1's exact-ids-prefix rule at the seam. Measured
on the real tokenizers: granite/Plain kept 20 of 21 ids and lost the
resumption to **one** seam token, while gemma under its own template
resumed at 100%.

**Not shipped, measured:** V3 has no sharding, no FFN/attention
decoupling, and does not join the grid (the server warns and the router
answers 503). `/v1/walk-ffn`, `/v1/expert/*`, `/v1/infer`, `/v1/embed`,
`/v1/shard` are V2-only.

New `examples/v3_request_phase_profile.rs` times the serve path's phases
below HTTP against a real container.

## [2026-08-22] — `/v1/sessions`: session observability and eviction

- **New surface.** `GET /v1/sessions`, `GET /v1/sessions/{id}`,
  `DELETE /v1/sessions/{id}` (`src/routes/sessions/`, spec §4.5) — an
  operational view and eviction control plane, deliberately *not* a
  second authority for modifying model state. The representation is
  metadata only: identity, lifecycle stamps, patch identities, and
  continuation availability/token counts. Sessions are server-level,
  so the path is unprefixed in multi-model mode too.
- **Sessions are identities; overlays are lazy.** `SessionState`'s
  `PatchedVindex` is materialised on first patch/insert instead of at
  creation, so binding a session costs a map entry rather than a
  base-index clone — and a session with no overlay reads exactly like
  the global state on the inference path. `session.rs` grew into
  `session/{clock,lease,manager,state}`.
- **Continuations are session-owned.** A KV state retained by a
  `/v1/responses` request carrying `X-Session-Id` records that session
  as its owner. `DELETE` frees them (`continuations_freed` on the
  receipt); `/v1/sessions` reports `continuation.available` /
  `input_tokens` plus per-session `resumptions` /
  `reused_tokens_total`. Requests without the header retain unowned
  states as before.
- **Deletion cannot be resurrected by a late write.** Generation
  retains KV after the request's session access has ended, so deletion
  now retires the session's **lease** — a lock-free `Arc` the blocking
  generation path carries — and `ResponseKvCache::insert` checks
  liveness under the cache lock. Both orderings of the
  delete-vs-late-insert race are gated across real threads; TTL
  eviction retires leases the same way, and the cache's sweep collects
  the resulting orphans.
- **`DELETE /v1/sessions/{id}` is idempotent** (`deleted: false` on a
  repeat) — deliberately unlike `DELETE /v1/responses/{id}`, which
  404s, because a cleanup endpoint should not fail for having already
  succeeded.
- **One `unix_now`.** The OpenAI surface's `created`-stamp helper now
  re-exports the session clock's implementation instead of keeping its
  own copy.
- Gates: 1132 tests green, clippy `-D warnings`, fmt, coverage policy
  (total 82.44%, included-total 93.12%, up from 92.7%); the five new
  files land at 97–100%.

## [2026-08-22] — P1 eviction fix, N0-router capability bit, N1 first slice

- **P1 closed.** New `src/maintenance/` background sweeper (60 s tick,
  decoupled `SweepTarget` closures) drives real eviction: sessions
  (`SessionManager::evict_expired`, new `--session-ttl-secs`),
  rate-limit buckets (`RateLimiter::evict_stale` finally has a
  production caller), and the N1 KV cache below. `ratelimit.rs` and
  `session.rs` became module folders with `tests/`.
- **N0-router server side.** Servers announce
  `AnnounceMsg.serves_openai` (true only for full-model,
  inference-enabled processes) so `larql-router` can front the grid
  with the OpenAI surface — see the router CHANGELOG §N0-router.
- **N1 first slice (V3 × Responses API).** Chained
  `previous_response_id` requests on a V3 runtime resume from a
  resident KV state (`src/response_kv/`, `--v3-kv-cache-entries` /
  `--v3-kv-ttl-secs`) when the new prompt's ids extend the absorbed
  ids, and fall back to a full prefill otherwise — produced tokens are
  identical either way (`vindex3::generate_v3_resumable`, gated
  bit-for-bit). Hits surface as
  `usage.input_tokens_details.cached_tokens`.
- **OpenAI-surface coverage closure.** Fixed the SSE test
  fixture-lifetime bug that silently uncovered the whole streaming
  pump, extracted the four duplicated per-token callbacks into
  `routes/openai/token_tap.rs`, and coverage-gated the surface
  (included-total ≥ 92%).
- **N1 prefix-stability matrix.** `tests/test_openai_responses_v3_prefix.rs`
  gates exact-token-id resumption under real request construction on a
  template-stable fixture; found + fixed the missing model-identity
  key on the KV cache (cross-model chains are now non-consuming
  misses); new `resumptions` / `reused_tokens_total` counters make the
  `hits`-vs-resumed gap observable in `/v1/stats`; template census:
  all five chat templates preserve their opening under conversation
  growth.
- **N0.6 on V3.** Tools / structured output run on VINDEX3 through the
  shared schema→FSM→mask pipeline via the new masked V3 driver
  (`continue_session_masked`), gated end-to-end with a JSON-lexeme
  fixture (`tests/test_openai_v3_tools.rs`). Two latent FSM bugs fixed
  for BOTH runtimes: emission-time key discipline on closed objects,
  and trailing-comma rejection (post-comma requires a key; a comma
  with no viable key left is refused). Stale `--features metal`
  Makefile targets (feature renamed/moved per ADR-019) repaired.

## [2026-05-28] — Hardening findings from the whole-codebase review

From the whole-codebase review ([`docs/audits/codebase-review-2026-05-28.md`](../../docs/audits/codebase-review-2026-05-28.md)):

- **P1 — unbounded in-memory growth with dead eviction logic.** The session map (`src/session.rs:184`) and rate-limit buckets (`src/ratelimit.rs:83`) never evict. Memory/DoS class — wire up the eviction that already exists but isn't called.

Recorded, not fixed. Still open — see [`ROADMAP.md`](ROADMAP.md) §"Open
defects".

## [2026-05-07] — State of the server at this date (migrated "Current state")

Migrated from `ROADMAP.md` on 2026-08-04, where it had stood as the *current*
state for nearly three months. Read as a snapshot with a date on it, not as a
description of the server today.

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

## [2026-05-07] — Perf snapshot: M3 Max, 2-shard grid, 26B-A4B

Point-in-time measurements, migrated from `ROADMAP.md` on 2026-08-04. The
"Remote MoE expert path" table that `README.md` links to lives here.

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

## [2026-05-17] — Q4K synthetic vindex fixture; completions un-excluded

Built the Q4K fixture I'd named as the gate for chat / completions /
stream coverage in the prior session-close note. Generation paths
now exercise real Q4K storage without panicking on `attn Q4K slices
missing for layer N`.

### Added

- **`tests/common/synthetic_q4k_vindex.rs`** — `build()` produces a
  full Q4K vindex on disk by (a) writing a tiny Llama-shaped
  safetensors model to a tempdir, (b) running it through
  `larql_vindex::build_vindex_streaming` with
  `QuantFormat::Q4K`. Mirrors the gold-standard pattern from
  `larql-vindex/tests/test_vindex_to_q4k.rs::q4k_end_to_end_from_synthetic_safetensors`.
  Dims: hidden=8, intermediate=4, num_layers=2, vocab=16 — small
  enough that each tensor pads to exactly one 256-element Q4_K
  super-block.
- **`tests/common/mod.rs::model_with_q4k_weights()`** — returns
  `(Arc<LoadedModel>, SyntheticQ4kVindex)`. Mirrors production
  `bootstrap.rs:238-256`: calls `VectorIndex::load_attn_q4k` +
  `VectorIndex::load_interleaved_q4k` explicitly after the base
  `load_vindex` so the Q4K data is actually attached to the index
  (without those calls, `insert_q4k_layer_tensors` still panics
  even though the on-disk files exist). The fixture's tokenizer is
  overridden to a 12-entry WordLevel — the streaming pipeline
  defaults to an empty BPE, which would encode every prompt to 0
  tokens and short-circuit the generation loop.
- **`tests/test_synthetic_q4k_smoke.rs`** — 3 tests: file-layout
  inventory, `get_or_load_weights` succeeds, `insert_q4k_layer_tensors`
  returns Ok. The third was the actual gate from the prior session.
- **`tests/test_openai_chat_coverage.rs`** — 11 tests covering the
  chat endpoint against the Q4K fixture: basic non-streaming and
  streaming, system message → template rendering, empty messages
  400, n>1 400, invalid JSON 400, sampling params, stop strings,
  `response_format: json_object`, tools, and multi-model dispatch.
  (Not yet measured at time of writing — handed off to a parallel
  session.)
- **`safetensors = "0.7"`** added as a larql-server dev-dependency
  (mirrors the version pinned by larql-vindex).

### Changed

- **`tests/test_openai_completions_coverage.rs`** swapped to use
  `model_with_q4k_weights` instead of `model_with_real_weights`.
  Drains the streaming SSE body fully (was capped at 64 KiB) and
  asserts on the `[DONE]` terminator.
- **`coverage-policy.json`** — `routes/openai/completions.rs`
  removed from `exclude_globs` and added at debt baseline 86%
  (was 40% pre-session, lifted to 86.85% via the Q4K backing +
  streaming-drain + stop-string tests). The remaining ~13% is the
  per-token streaming callback body — the synthetic Q4K generator
  returns 0 tokens with max_tokens=2 so the callback is never
  invoked.

### Known follow-up

- **Per-token streaming callback path stays uncovered.** The Q4K
  fixture's generator produces 0 tokens given the synthetic weights
  (the diagonal-ramp weights produce all-`-inf` logits after rope +
  Q4K dequant). Hitting the per-token callback body would need
  either a longer `max_tokens` budget or tuning the weights so the
  logits aren't degenerate. Tracked as a finer-grained follow-up,
  not blocking.

## [2026-05-17] — Coverage push session close: 75% total, 37 files at default 90%

Closing pass after the chat/completions/stream fixture wall was
diagnosed. Realised the wall isn't NaN-prone weights — it's that
`generate_with_sampling` panics with `attn Q4K slices missing for
layer 0` when called on a non-Q4K vindex (confirmed in
`vindex/kquant_forward/cached.rs:106`). The generation paths require
a Q4K-quantised synthetic vindex, not just stable weights. So picked
off the remaining files that don't depend on that fixture instead:

### Un-excluded this round

- **`routes/openai/schema/mask.rs` (0% → 93.44%)** — 6 in-test cases
  cover lazy surface-table init, the cache-hit replay path, the
  cache-miss-falls-through path, prompt-prefill replay, EOS masking
  while FSM incomplete, and the out-of-table token graceful fallback.
- **`openapi.rs` (0% → 100%)** — 3 tests against `swagger_router()`
  and `ApiDoc::openapi()`: serves `/v1/openapi.json` (validates
  `openapi: "3.x"`), serves the Swagger UI index, and the `ApiDoc`
  struct emits a non-empty `paths` object.

### Debt-baseline ratchets

Three files reached or cleared the 90% default and were lifted out
of the `per_file_line_min_percent` debt list entirely:

- `env_flags.rs` 59.6% → **100%**
- `shard_loader.rs` 88.0% → **91.72%**
- `state.rs` 85.8% → **91.90%**

One file ratcheted upward within the debt list:

- `routes/warmup.rs` 80.8 → **84.0** (deeper paths still need fixture)

### Q4K-fixture confirmation

A diagnostic smoke test (deleted after diagnosis) panicked with
`attn Q4K slices missing for layer 0` from
`vindex/kquant_forward/cached.rs:106` when calling
`generate_with_sampling` on the synthetic f32 vindex. That confirms
chat / completions / stream / walk_ffn/q8k all need the same
fixture: a Q4K-quantised synthetic vindex with `attn_q4k.bin` +
`interleaved_q4k.bin` plus the required dimensions for the K-quant
kernels. Not built this session.

### Session totals

| Metric | Pre-session | End of session |
|---|---|---|
| Total | 69.82% | **75.05%** (+5.23 pp) |
| Included files | 91.87% | **92.62%** |
| Files at 90% default | 26 | **37** (+11) |
| Debt baselines | 10 | **8** (3 freed, 1 ratcheted up) |
| Tests | 739 | **813** (+74) |

Files un-excluded across the session: `routes/explain.rs`,
`routes/infer.rs`, `routes/openai/schema/mask.rs`, `openapi.rs`.
Plus `walk_ffn.rs` split into a 7-file module folder (5 of 7 land at
≥93% out of the gate; 2 — core.rs and q8k.rs — stay excluded for
documented MoE + Q4K fixture reasons).

## [2026-05-17] — infer.rs un-excluded; fixture wall hit on chat/completions/stream

Continued coverage push. `routes/infer.rs` joins `routes/explain.rs`
in the included set (50% → 97% with 9 new integration tests +
2 new in-file unit tests for `format_knn_override`). Started
coverage tests for `routes/openai/completions.rs` and
`routes/expert/warmup.rs` — both plateau without further fixture
work:

- **`openai/completions.rs` (40% → 56%)**: synthetic vindex's
  diagonal-ish f32 weights NaN partway through
  `generate_with_sampling`, so the per-prompt completion-loop body
  + the streaming spawn_blocking body stay uncovered. Tests
  exercise the handler validation (n>1, empty prompt, echo+stream
  rejection, batched+stream rejection, sampling params, stop
  strings, logprobs, SSE content-type), which lifts the entry
  branches. Stays excluded until stable synthetic generation weights
  land. 12 integration tests pass.
- **`expert/warmup.rs` (0% → 9%)**: only the env-gated and
  `is_hybrid_moe() == false` early-return branches are reachable
  with the current dense-llama synthetic. The deeper unit/expert
  filter resolution + HNSW build path needs a synthetic hybrid-MoE
  arch (router weights + N experts). 2 integration tests pass.

### Realistic-ceiling note

Pushing the remaining excluded files (`openai/chat.rs`,
`openai/completions.rs`, `stream.rs`, all `expert/*.rs`,
`walk_ffn/core.rs`, `walk_ffn/q8k.rs`) above 90% requires three
substantial fixture extensions:

1. **Stable generation weights** — synthetic vindex weights tuned
   so `predict_with_ffn` / `generate_with_sampling` produce
   finite (non-NaN) output for multiple tokens. Unlocks chat,
   completions, stream, infer's compare-mode deeper paths.
2. **Synthetic hybrid-MoE vindex** — router_proj + N expert weights
   + arch override returning `is_hybrid_moe() == true`. Unlocks
   all `expert/*.rs` files + `walk_ffn/core.rs`'s MoE branch.
3. **Synthetic Q4K-quantised vindex** — different storage format
   (interleaved_q4k.bin etc). Unlocks `walk_ffn/q8k.rs`.

Each is ~1-2 days of fixture engineering. The honest path to total
≥ 90% is to land those fixtures, not to keep adding ineffective
tests against the current dense-llama synthetic.

### Headline numbers (this session)

| Metric | Pre-session | End of session |
|---|---|---|
| Total | 69.82% | **74.52%** (+4.70 pp) |
| Included files | 91.87% | **92.61%** |
| Files at 90% default | 26 | **32** (+6) |
| Files excluded | 13 | 12 (replaced walk_ffn.rs monolith with core.rs + q8k.rs; net −1 after explain + infer un-exclusion) |
| Test count | 739 | **804** (+65) |

## [2026-05-17] — walk_ffn split into a module folder + coverage progress

Continuing the coverage push: split the 1434-line `routes/walk_ffn.rs`
monolith into a 7-file `routes/walk_ffn/` module folder, then drove
coverage on the new sub-files. Net: total 71.18% → **73.76%**;
included 91.98% → **92.42%**; 27 → **31** files at default 90%.

### Changed

- **`routes/walk_ffn.rs` → `routes/walk_ffn/`**: split into seven
  files matching the natural section headers in the original:
  - `types.rs` (100%) — `WalkFfnRequest`, `FfnEntry`, `FfnOutput`,
    `RifGuard`, `BINARY_CT`, `BATCH_MARKER`.
  - `binary.rs` (98.56%) — codec (`decode_binary_request`,
    `encode_binary_output` / `_f16` / `_i8`, `encode_json_full_output`)
    + the existing in-file unit tests + new tests for the f16 / i8
    encoder branches that the previous monolith left uncovered.
  - `validate.rs` (93.55%) — pure-function validators
    (`collect_scan_layers`, `validate_residual`, `validate_owned`)
    with 7 new in-file unit tests.
  - `dispatch.rs` (98.39%) — JSON entry (`run_walk_ffn` +
    `run_full_output` / `run_features_only`).
  - `handler.rs` (86.90% — debt baseline) — axum entrypoint
    `handle_walk_ffn`. The 13% uncovered are error paths inside
    `tokio::task::spawn_blocking` closures (no-model-loaded,
    read-body-error, JSON-serialize-error) that are hard to drive
    from integration tests without a special harness.
  - `core.rs` (24.89% — still excluded) — `run_full_output_core`.
    The MoE branches (~80 lines) need a remote-MoE backend the
    synthetic fixture doesn't provide.
  - `q8k.rs` (34.18% — still excluded) — `handle_walk_ffn_q8k`. Needs
    a Q4K-quantised vindex; same fixture gap.
  - `mod.rs` — re-exports preserving `crate::routes::walk_ffn::X`
    paths so callers (`routes/mod.rs`, `openapi.rs`) didn't have
    to change. utoipa's generated `__path_*` types are re-exported
    so the `#[derive(OpenApi)]` macro keeps finding them at the
    pre-split path.

### Coverage policy

- `routes/walk_ffn.rs` removed from `exclude_globs`; replaced with
  `routes/walk_ffn/core.rs` + `routes/walk_ffn/q8k.rs` (the two
  submodules that need MoE / Q4K fixtures to cover).
- `routes/walk_ffn/handler.rs` added at 86% debt baseline.
- Gate: total 73.76% lines, included 92.42%, 42 files checked,
  31 at default 90%, 11 debt baselines. **Up from 36 files / 26 at
  default / 10 baselines pre-split.**

### Tests

- 787 server tests pass (was 748 at start of session, +39 from
  walk_ffn split's in-file unit tests + coverage tests + new
  validate tests). No regressions across `test_grid_*`,
  `test_http_*`, `test_synthetic_vindex`, `test_walk_ffn_coverage`.

## [2026-05-17] — Synthetic-vindex test fixture + coverage push begins

Phase 1 of the larql-server coverage push (target: total ≥ 90%). Built
a reusable test fixture that constructs a complete f32 vindex on disk
from synthetic deterministic weights, then drove the pilot file
`routes/explain.rs` from 44.86% → 93.46% (clears the 90% per-file
floor). Total server coverage 69.82% → 71.18%; included-files
coverage 91.87% → 91.98%.

### Added

- **`tests/common/synthetic_vindex.rs`** — `build()` produces a tempdir
  with `index.json` (`has_model_weights: true`), `weight_manifest.json`,
  `gate_vectors.bin`, `attn_weights.bin`, `up_weights.bin`,
  `down_weights.bin`, `norms.bin`, `lm_head.bin`, `embeddings.bin`,
  `down_meta.bin`, and `tokenizer.json` — exactly what
  `larql_vindex::load_model_weights_with_opts` consumes. Synthetic
  weights match `larql-vindex/tests/test_vindex.rs::make_synthetic_model`:
  2 layers × hidden=8 × intermediate=4, vocab=16. Build time ~10ms.
- **`tests/common/mod.rs::model_with_real_weights(id)`** — returns
  `(Arc<LoadedModel>, SyntheticVindex)`. `LoadedModel.path` points at
  the fixture so `get_or_load_weights()` (called by every
  `full_output=true` route handler) succeeds. Sibling
  `_and_labels(id, labels)` variant seeds `probe_labels` for tests of
  the `relations_only` branches.
- **`tests/test_synthetic_vindex.rs`** — 9 tests:
  fixture smoke (2) + explain-handler coverage (7 — basic, attention,
  relations_only, relations + labels, band filter, multi-model,
  multi-model 404, invalid JSON). Runs in ~30ms.

### Changed

- **`coverage-policy.json`** — `routes/explain.rs` removed from
  `exclude_globs`; per-file 90% default now applies. 27 files (was 26)
  clear the default; 10 debt baselines unchanged. Total floor
  unchanged at 65%, included floor unchanged at 90%.

### Playbook for next sessions (Tasks #94 — remaining 7 excluded files)

Pattern that worked on explain.rs:

1. Pick the next excluded file from `coverage-policy.json::exclude_globs`.
   Recommended order by ROI: `routes/walk_ffn.rs` (874 missed lines, biggest
   single gain), `routes/openai/chat.rs` (547), `routes/openai/completions.rs`
   (303), `routes/stream.rs` (421), `routes/infer.rs` (195),
   `routes/expert/*` (5 files at 0-57%), `grpc.rs` (302).
2. Write one smoke test using `common::model_with_real_weights` that POSTs
   to the file's main route handler. Measure coverage delta.
3. Add 4-6 more tests targeting uncovered branches surfaced by
   `cargo llvm-cov report --package larql-server --json`. The vocab in
   `synthetic_vindex.rs` is small — most uncovered ranges are
   reachable by adjusting query params (`band`, `relations_only`,
   `with_attention`, `top_k`, full_output flags) or by giving the
   fixture a payload it can run through.
4. When the file clears 90%, remove it from `exclude_globs` and
   confirm `make larql-server-coverage` passes.

Caveats observed during the pilot:

- The fixture's tokenizer needs at least a small WordLevel vocab.
  An empty BPE encodes every prompt to 0 tokens; every per-token
  branch in the route handler then stays uncovered. The shipped
  fixture uses 12 WordLevel entries; adjust as needed.
- The fixture's intermediate / hidden sizes are tiny on purpose
  (build time matters). If a route needs larger shapes to exercise a
  specific branch (e.g. multi-head attention paths), bump
  `ModelDims` in `make_weights()`.
- `LoadedModel` is `!Clone`; pass `probe_labels` at construction
  via `model_with_real_weights_and_labels(id, labels)` rather than
  mutating after `Arc::new`.

## [2026-05-16] — Mode B / QUIC ROADMAP backfill + GT5 end-to-end test

ROADMAP-drift sweep: three G-MODEB / G-TRANSPORT items previously
listed as "Not started" were actually shipped between 2026-05-13 and
2026-05-15 (on the router side) and earlier on the server side. The
server ROADMAP was updated to reflect reality and the missing
end-to-end test was added.

### Fixed

- **GT5 (Mode B gap-fill) — server-side ROADMAP corrected + new
  end-to-end test.** `announce.rs::run_available_loop` had been wired
  end-to-end (`AvailableMsg` → handle `AssignMsg` →
  `shard_loader::download_and_load_shard` → `ReadyMsg` /
  `RefuseMsg` → loop until `AckMsg`) since 2026-05-13, but no
  integration test drove the *production* loop —
  `mode_b_full_vertical_handoff` inlined the protocol in the test
  body. New test
  `mode_b_try_once_available_drives_full_handshake` spawns the real
  loop via the newly-public `announce::try_once_available` entry
  point against an in-process router fixture and asserts Available →
  Assign → download → Ready → Ack lands in <3s.
- **Misleading Mode A AssignMsg log.** `announce.rs:413` used to log
  `"Received AssignMsg but Mode B not implemented — ignoring"` when
  a Mode A (already-serving) stream received an unexpected AssignMsg.
  Mode B *is* implemented, in `run_available_loop`; the stub message
  was misleading. Now logs a descriptive warning calling out that
  the router shouldn't target Mode A streams with AssignMsg.
- **Three stale ROADMAP entries marked shipped.** GT5, GT6 (dynamic
  rebalancing / drain-then-reassign — ADR-0011 §Phase B2), and GT7
  (QUIC transport — ADR-0010) all moved from `Not started` →
  `✅ Shipped` with code pointers and test references.
- **Three integration tests un-bit-rotted.**
  `tests/test_grid_mode_b.rs`, `tests/test_grid_replication.rs`,
  `tests/test_grid_drain_reassign.rs` had been broken since
  ADR-0018 (MoE expert routing) widened `try_assign_gap` to take
  `expert_start` / `expert_end` and moved `GridServiceImpl` to
  `larql_router::grid::service`. Patched all three (new
  signatures + import paths + `parking_lot::RwLock` for `GridState`
  to mirror the router's 2026-05-16 lock primitive swap). 9 tests in
  3 files all pass.
- **`parking_lot` added as a server dev-dependency.** Mirrors the
  router's `GridState` lock primitive so test fixtures can construct
  an `Arc<parking_lot::RwLock<GridState>>` directly.

### Known follow-up

- **GT5 hash-verification mismatch (P1).** `vindex_identity_hash`
  emits a 16-hex model-identity tag, but `shard_loader` expects a
  SHA-256 content hash on `AssignMsg.shard_hash`. Today deployments
  must pass an empty/placeholder hash so the verification is
  skipped. Real content hashing wants a new optional
  `shard_content_sha256` field on `AnnounceMsg` distinct from
  `vindex_hash`. See `ROADMAP.md` G-MODEB §GT5 "Known follow-up".

## [2026-05-10] — REV: code review

Migrated from `ROADMAP.md` on 2026-08-04. Kept in full: the lens it applies —
per `THESIS.md`, legibility is a primary feature, so a vLLM/SGLang/TGI engineer
should be able to read this and copy ideas out — is the reusable part, and the
findings record diagnoses that exist nowhere else.

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

## [2026-05-10] — Code-review P0 sweep + coverage scaffolding

Five P0 fixes from the in-tree code review (REV1–REV5 in `ROADMAP.md`)
plus the missing larql-server Makefile coverage targets and a per-file
90% coverage policy.

### Fixed

- **REV1 — gRPC sort panics on NaN scores.** `grpc_describe` and
  `grpc_select` used `partial_cmp(...).unwrap()`, which panics on NaN.
  Replaced both call sites with a shared `cmp_score_desc(a, b)` helper
  that maps NaN → `Ordering::Equal`. A corrupted vindex or a future
  patched-scoring path that produces NaN no longer takes a gRPC worker
  down. Five new unit tests in `grpc.rs` lock the property.
- **REV2 — Non-constant-time API key comparison.** `auth.rs` used
  `==` on `&str`, which short-circuits and leaks bytewise progress
  through request timing. Tokens are now SHA-256-hashed and the digests
  compared via `subtle::ConstantTimeEq`. Module-level doc block names
  the threat model. `subtle` (already in the lockfile via rustls)
  added as a direct dep. Six new unit tests in `auth.rs`; six existing
  `http_auth_*` integration tests still pass with no behavioural
  change.
- **REV3 — `blocking_read` on tokio RwLock inside async path.**
  `SessionManager::apply_patch` previously called
  `model.patched.blocking_read()` while holding `sessions.write().await`
  on a worker thread, which on a multi-thread runtime stalls the
  worker (and risks deadlock against any task acquiring those locks
  in the opposite order). Restructured into fast-path / slow-path:
  the slow path drops the sessions write guard, awaits
  `model.patched.read()`, then re-acquires and uses
  `entry().or_insert_with(...)` to absorb the race where another task
  inserted the same `session_id`. No `blocking_read`/`blocking_write`
  on tokio locks is reachable from an `async fn` in `session.rs`
  anymore. Two new regression tests assert (a) forward progress when
  another task holds a `patched.read()` and (b) 16-way concurrent
  `apply_patch` on the same `session_id` finishes within a bounded
  deadline.
- **REV4 — OpenAI error envelope diverged from spec.** Non-streaming
  responses on `/v1/embeddings`, `/v1/completions`, and
  `/v1/chat/completions` returned `{"error": "msg"}` (flat); the OpenAI
  Python and JS SDKs expect
  `{"error": {"message", "type", "param", "code"}}` (nested) and broke
  on field access against the flat shape. Streaming SSE error chunks
  already used the nested form, so non-stream and stream errors were
  inconsistent. Introduced a new `OpenAIError` type with constructor
  helpers (`invalid_request`, `not_found`, `service_unavailable`,
  `server_error`) and an `IntoResponse` that renders the canonical
  nested envelope with `param`/`code` always present (possibly null).
  `From<ServerError>` lets internal helpers keep `ServerError` and
  propagate via `?`. The three OpenAI handler entry-point return
  types flipped to `Result<_, OpenAIError>` and 16 direct
  `return Err(ServerError::X(...))` sites converted to the matching
  `OpenAIError::Y(...)` constructor. LARQL paradigm endpoints keep the
  flat envelope. Six integration tests assert the nested shape on
  400/503 paths across the three handlers; seven unit tests cover the
  type itself.
- **REV5 — tool-call JSON parser surfaced 500 instead of 400 on
  malformed nested-brace output.** `build_tool_call_message` used
  `find('{')` + `rfind('}')` to extract JSON from constrained-decoder
  output, which silently picked the wrong slice on trailing junk /
  multiple objects / markdown wrappers and surfaced the parse failure
  as `ServerError::Internal` (500). Rewrote as a straight-line
  `serde_json::from_str(text.trim())` with structured diagnostics
  (`invalid JSON: …`, `tool output must be a JSON object`, missing-
  field reports), and flipped the call-site error class from
  `Internal` to `OpenAIError::invalid_request` so the client now sees
  **400 invalid_request_error** with a concrete message. Nine new
  unit tests cover happy path, surrounding whitespace, nested-brace
  arguments, trailing junk, empty/whitespace, non-object top-level,
  missing `name`/`arguments`, and invalid JSON.

### Added

- **Two-envelope error documentation.** `docs/server-spec.md §8.3.1`
  rewritten with the LARQL-flat / OpenAI-nested split and a canonical
  `type` table. README `Error Codes` section updated to match.
- **Makefile coverage targets** for larql-server, mirroring the
  larql-compute / larql-vindex pattern:
  `larql-server-test`, `larql-server-fmt-check`, `larql-server-lint`,
  `larql-server-coverage`, `larql-server-coverage-summary`,
  `larql-server-coverage-html`, `larql-server-coverage-policy`,
  `larql-server-ci`. Threshold variables: `LARQL_SERVER_COVERAGE_MIN`
  (default 65 — current baseline), `LARQL_SERVER_COVERAGE_POLICY`,
  `LARQL_SERVER_COVERAGE_REPORT`.
- **`coverage-policy.json`** with default 90% line floor, 28 per-file
  debt baselines snapshotted from the 2026-05-10 measurement, and the
  total floor at the measured 65.6% baseline. Policy semantics
  ratchet upward only — new / split files automatically inherit the
  90% default.

### Internal

- Cleared 5 pre-existing clippy errors in lib (`bootstrap.rs:230`
  boolean simplification, `metrics.rs:64` missing `Default` for
  `LayerLatencyTracker`, `walk_ffn.rs` doc indentation + needless
  lifetimes + redundant closure). `cargo clippy -p larql-server --lib
  --no-deps -- -D warnings` now clean.
- Updated `tests/test_expert_endpoint.rs` import: `cpu_moe_forward`
  and `MoeLayerWeights` moved from `larql_inference` to
  `larql_compute` in the upstream refactor; the test had a stale
  import that blocked `--tests` builds. Pure plumbing — matches the
  cargo error hint.
- Added `#[derive(Debug)]` to `ChatChoiceMessage`, `ToolCall`,
  `ToolCallFunction` to support `Result::unwrap_err()` in the new
  `build_tool_call_message` tests.

### Coverage snapshot (2026-05-10)

- **TOTAL**: 65.68% line / 72.18% function / 64.90% region.
- **At-or-above 90% default**: `routes/openai/error.rs` (100%),
  `routes/openai/util.rs` (99.6%), `routes/openai/embeddings.rs`
  (93.2%), `session.rs` (96.1%), `state.rs` (85.8% — debt baseline),
  `auth.rs` (98.0%), `wire.rs` (96.9%), `etag.rs` (100%), and 16
  others.
- **Largest debt items** (all carry baselines, must ratchet up):
  `routes/expert/{batch_legacy,multi_layer_batch,single,warmup}.rs`
  at 0% (need a live grid harness),
  `routes/openai/schema/mask.rs` at 0%, `bootstrap.rs` at 29.7%,
  `routes/openai/completions.rs` at 40.3%, `routes/walk_ffn.rs` at
  49.0%, `routes/openai/chat.rs` at 53.4%.

## Completed milestones (migrated section)

Migrated from `ROADMAP.md` on 2026-08-04. Predates the dated-entry convention
above and carries its own internal dates; kept because it is the only record of
the 2026-04/05 shipping sequence. `THESIS.md` links here for the
cos-similarity / tok/s / RSS / latency numbers.

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

---

<!-- Migrated from ROADMAP.md on 2026-08-22. These entries preserve the
     date and voice they were originally written in; the ROADMAP now
     carries only forward-looking work. -->

## [2026-05-02] — F0 closed + N0 slices 1 + 2 (OpenAI compat: models + embeddings + completions + chat completions)

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
tests + 19 integration tests + new live `crates/larql-demos/examples/server/openai_demo.rs`
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

## [2026-05-01 (continued)] — Q1 code-quality cleanup (9 of 10 items)

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

## [2026-05-01] — HTTP CPU-path optimisations + UDS transport + layer-batch wire

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

## [2026-05-01 (continued)] — larql-server review pass

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

## [2026-04-26] — Per-expert byte table refactor + `experts_packed.bin` removal

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

## [2026-04-30] — gRPC grid: end-to-end accuracy

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

## [2026-04-30] — Metal expert dispatch: 3.7× speedup found, blocked on kernel bug

`LARQL_MOE_TIMING=1` showed the grid bottleneck is **server compute = 95%** of token wall time (network = 2%, route+fire = 3%). Per layer: 8.36ms server / 0.18ms net. Each shard runs its 4 picked experts (gate + GELU + down) on CPU-rayon BLAS — that's where the time goes. Sub-arc:

| Item | Notes |
|------|-------|
| Bottleneck localised | CPU experts = 250ms/token (95%) on the loopback 2-shard setup. Network = 5ms (2%). The grid-side overhead is negligible — accelerating the shard's expert math is the only meaningful lever |
| `--features metal-experts` measured: **3.7× speedup** | Server with Metal expert dispatch: 264ms → 117ms per token, 2.3 tok/s → **9.4 tok/s** (preselected path → 11.2 tok/s). Significant — server compute drops from 250ms → 115ms |
| **Accuracy bug blocks shipping** | Metal expert kernel (`MetalBackend::run_experts_preselected_metal` and `_prestaged_metal`, both routes) produces numerically wrong outputs for Gemma 4 26B-A4B-it MoE shape (cos≈0.7 vs CPU, \|metal\|≈70% of \|cpu\|). End-to-end output: "**Paris**" via CPU vs "answer is in the context" via Metal. Same kernels are correct for dense FFN at inter=2560/10240/21504 — bug is specific to MoE inter=704 dispatch |
| Workaround: default to CPU even on metal-experts builds | `run_experts_metal_batch` now early-returns `None` unless `LARQL_USE_METAL_EXPERTS=1` is set. Shipping correctness over speed; the Metal path stays opt-in for kernel-debug runs |
| Diagnostic: `LARQL_METAL_VS_CPU_DEBUG=1` | Server-side per-call A/B compare in `run_experts_metal_batch` — runs both Metal and CPU on the same input, prints max\|Δ\|, \|metal\|, \|cpu\|, cos. Ready to use when someone digs into the kernel |
| See also | `larql-compute/ROADMAP.md` "Open: Metal MoE expert kernel — accuracy bug at inter=704" for the kernel-side investigation plan |

## [2026-04-26] — examples, synthetic benchmark, grid checks

| Item | Outcome |
|---|---|
| `server_demo` | Runs locally with synthetic data; fixed invalid probe-label JSON comma output and updated rate-limit text for `--trust-forwarded-for`. |
| `embed_demo` | Runs locally with synthetic embed/logits/token responses and binary-wire examples. |
| `server_bench --release` | Synthetic benchmark completed: `gate_knn` top-5 0.022 ms/op, 8-layer `walk` 0.203 ms/op, single-layer `walk-ffn` 0.032 ms/op, batched 8-layer `walk-ffn` 0.321 ms/op, describe simulation 0.298 ms/op, 512-token embed prefill 0.114 ms/op. |
| `bench_embed_server` | Example builds under `cargo check -p larql-server --examples`; execution requires a real vindex path. |
| Grid unit coverage | Added `GridState` tests for inclusive ranges, default single-model routing, least-loaded replica selection, deregistration, batched gap reporting, and status gaps. `cargo test -p larql-router` now runs 20 tests. |
| Docs | Updated server README examples/benchmarks/testing, router README validation, and router spec validation commands. |

## [2026-04-26] — coverage round-6 (embed + walk-ffn reachable gaps)

| Item | Outcome |
|---|---|
| `routes/embed.rs` modularity | Extracted binary embed/logits parse helpers and binary embed response encoder |
| `routes/embed.rs` coverage | **66.7% → 86.5% line**, **70.7% → 86.3% function** |
| `routes/walk_ffn.rs` coverage | **76.7% → 79.5% line**, **77.3% → 82.0% function** |
| Tests | 458 → **478** tests |
| Coverage | **71.9% → 74.2% line**, **78.9% → 81.2% function** |

## [2026-04-26] — modularity + coverage round-5

| Item | Outcome |
|---|---|
| Boot/loading modularity | Moved parse/discovery/vindex-load helpers out of `main.rs` into `bootstrap.rs`; binary now keeps CLI orchestration while library code is directly testable |
| `routes/stream.rs` | Extracted pure `stream_describe_messages`; describe stream behavior can be tested without a WebSocket client |
| `routes/infer.rs` | Extracted mode selection and prediction formatting helpers |
| `routes/explain.rs` | Extracted band mapping, probability/gate/attention rounding, prediction formatting, and lens formatting helpers |
| Clippy | Server-local clippy clean with `--no-deps`; full dependency-checking command is blocked by existing `larql-vindex` warnings |
| Coverage | **69.2% → 71.9% line**, **77.1% → 78.9% function** (458 tests) |

## [2026-04-26] — coverage round-4 (T2 reachable gaps)

| Item | Outcome |
|---|---|
| `embed_store.rs` | 25% → **98% line** with tiny f16 mmap fixtures and L1 cache behavior tests |
| `announce.rs` | 6% → **56% line** by extracting/test-covering announce, heartbeat, dropping, and bearer helpers |
| `main.rs` | 0% → **23% line** with binary unit tests for parse/discovery/serve-alias helpers |
| `routes/stream.rs` | 0% → **28% line** with pure WebSocket message shape builders |
| `routes/infer.rs`, `routes/explain.rs` | Default/request deserialization coverage added; full paths remain weight-gated |
| Coverage | 63.9% → **69.2% line**, 73.4% → **77.1% function** (430 → 458 tests) |

## [2026-04-26] — coverage round-3 (T2 partial) + magic strings round-2

| Item | Outcome |
|---|---|
| `test_grpc.rs` — 28 new gRPC handler tests | Direct method calls on `VindexGrpcService` — no network socket; health, stats, describe, walk, select, relations, walk_ffn, infer, stream_describe |
| `grpc.rs` coverage | 0% → **65%** (169 lines uncovered, all gated on real model weights or gRPC streaming) |
| Magic strings — `"probe"` | `PROBE_RELATION_SOURCE` constant in `band_utils.rs`; used in describe.rs, grpc.rs, stream.rs |
| Magic strings — `"ok"` | `HEALTH_STATUS_OK` constant; used in grpc.rs health handler |
| Magic strings — gRPC modes | `INFER_MODE_WALK/DENSE/COMPARE` applied to grpc.rs (was using bare strings) |
| Magic strings — WebSocket types | `WS_TYPE_ERROR/LAYER/DONE/PREDICTION/INFER_DONE` and `WS_CMD_DESCRIBE/INFER` in stream.rs |
| Coverage | 57.2% → **63.3% line**, 65.3% → **73.2% function** (402 → 430 tests) |

## [2026-04-26] — coverage round-2 (T1)

| Item | Outcome |
|---|---|
| `functional_tokenizer()` in common | WordLevel tokenizer (France→0, …) added to test infra; unblocks describe/walk/walk-ffn body paths |
| `test_http_full_routes.rs` | 39 new HTTP integration tests exercising full describe/walk/walk-ffn code paths |
| `test_unit_band_utils.rs` | 13 pure unit tests for `band_utils.rs` constants + helpers |
| Infer + ratelimit branches | `infer_disabled=false` model builder; ratelimit middleware axum tests |
| Coverage | 49.1% → **58.0% line**, 56.4% → **65.3% function** (345 → 402 tests) |

## [2026-04-26] — code quality round-1

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

## [2026-04-26] — perf round-1 (G1+G2+G3)

| Item | Outcome |
|---|---|
| G1 cold-start profile | Two-phase: 1.27 s lazy weight load + 17 ms/layer mmap page-in. Warm steady state 0.2–0.3 ms/layer. |
| G2 `/v1/warmup` + `--warmup-walk-ffn` | First walk-ffn 1247 ms → 12.6 ms (99×). Boot trades ~1.3 s + 3.2 GB pre-allocation. HTTP endpoint also exposed for live re-warm. |
| G3 self-assembling gRPC grid | Live-validated `--grid-port` + `--join`: auto-join, coverage tracking, graceful failure (clean HTTP 400 on uncovered layer), auto-recovery on rejoin. |

## [2026-04-26] — W2 retrofit + grid validation

| Item | Outcome |
|---|---|
| `--warmup-hnsw` flag | Eager-builds HNSW across owned layers at boot via `warmup_hnsw_all_layers()`. Reports correct owned-layer count under `--layers`. |
| Boot log: W2 status | `Down features Q4K: loaded (W2 — per-feature decode skips q4k_ffn_layer cache)` when `down_features_q4k.bin` is present. |
| `/v1/stats.q4k_ffn` field | `{cache_slots, cache_bytes, feature_major_down}` — operators can verify W2 active + cache empty in steady state. |
| `larql convert add-feature-major-down` | New CLI subcommand. Retrofits an existing Q4K vindex without re-quantising the rest. 30 layers / 152 MB / 1.12 s on Gemma 26B. Idempotent. |
| Live grid validation | 2-shard layer-range split (0-14 + 15-29) on real 26B vindex, full fan-out via router, 8-way concurrent stress, 0.2 ms warm per-layer, 5.9 ms full-30-layer fan-out. |

## [Pre-2026-04-26] — foundations (already in place)

- HTTP API: `/v1/walk`, `/v1/walk-ffn`, `/v1/stats`, `/v1/health`,
  `/v1/infer`, `/v1/insert`, `/v1/expert/{layer}/{id}`, etc.
- `--layers START-END` shard slicing (mmap pages outside range stay
  paged out, RSS proportional to shard size).
- `--max-q4k-cache-layers` LRU bound on the legacy Q4K dequant cache.
- `--ffn-only` / `--embed-only` mode flags.
- gRPC self-assembling grid (`--grid-port` / `--join` / `--grid-key`).
- Bench rig daemon-aware (`larql-vindex` benches refuse if a server
  shares the host; override with `LARQL_BENCH_ALLOW_DAEMONS=1`).

## [2026-08-22] — Archive: completed ROADMAP items

Moved out of `ROADMAP.md`, which now carries only forward-looking work.
Each block keeps its own `**Status**` line, so the date a thing actually
shipped is the date recorded inside it — not the date of this entry.
Covers the grid transport / Mode B / benchmarking arcs (GT1–GT10),
`F-COLLECT`, the `F0` CPU-MoE correctness investigation, and the
2026-04-26 test-coverage and cold-start rounds (T1–T3, G1–G3).

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
