# VINDEX3 Runtime — the VI3 inference stack

Status: as of 2026-08-30 — describes the runtime as landed through the
VI3-INF-0..3, VI3-KV-1 and VI3-SERVE-1 rungs, plus the OpenAI-surface
completion of 2026-08-22 (`/v1/chat/completions` and `/v1/responses` V3
arms). The container format itself is specified in
`crates/larql-vindex/docs/vindex3-format-spec.md` (the 3.0 Candidate) and
`docs/vindex3-format.md` (the living spec); this document is the **State
and Serving contract**: what happens *after* a container exists — how it
is opened, executed, given continuation state, and served.

---

## 1. What the V3 runtime is

A VINDEX3 container is not "reorganised weights to reconstruct a
transformer from" — it is a closed executable program (a
`ComponentOpPlan`) plus its operands. The canonical interpreter in
`larql-vindex` owns model *meaning* (operation order, residual placement,
optional operations, span policy); a `PlanBackend` owns only the
arithmetic. The runtime stack gives that program a home inside
`larql-inference` **without translating it back into a V2-shaped model**:
no `ModelWeights`, no family detection, no `ModelArchitecture`
reconstruction (`crates/larql-inference/src/vindex3/mod.rs`).

The seam is deliberately tiny. Generation needs logits and state
progression, so that is the whole contract:

```text
generate (sampler / EOS / streaming)
       │
  LogitsSession        ← format-neutral: prefill, step, position
       │
 Vindex3Session        ← wraps the canonical DecodeSession
       │
  PlanBackend          ← reference / production / Metal arithmetic
```

What deliberately does **not** exist: a `load_vindex3() -> ModelWeights`
bridge, or a `match version {2 => …, 3 => …}` inside the V2 loader. The
two formats have different authority models and converge only above
`LogitsSession`.

---

## 2. The runtime layer — `Vindex3Runtime`

`Vindex3Runtime<B: PlanBackend>`
(`crates/larql-inference/src/vindex3/runtime.rs`) is one opened container
component: the executable plan, its operand store, and the arithmetic
backend. It owns what sessions borrow, so sessions can be created,
dropped, and re-created (fresh conversations) without re-planning the
container.

- `Vindex3Runtime::open(container, component, backend)` — inspects the
  container, plans the component's operations (`plan_component_ops`),
  and opens the `OperandStore`, solely from the container's own
  contents. A component whose stack does not fully classify into the
  declared operations **refuses to open**, with the closure defects in
  the error: an unclosed program must not be "best-effort" executed.
- `Vindex3Runtime::open_with(container, component, backend, policy)` — the
  same opening under an explicit `OpenPolicy { want, source }`: *what*
  encoding execution asks for (a compiled pack such as NVFP4, or `None`
  for the canonical bytes) and *whether* the runtime may manufacture it
  at load. `open` is `open_with` under the default policy. The opener
  itself is `open_component`, the one opening authority: `larql serve`
  binds through it today, and `larql run` / `larql vindex3 exec` are
  being converged onto it so no caller re-implements inspect, plan and
  store opening. The backend handed in is the realisation, never part
  of what the container means.
- `Vindex3Runtime::session()` — an incremental session at position zero.
- `Vindex3Runtime::session_with_kv(kv)` — a session whose continuation
  state lives in, and outlives the session as, the caller's `KvState`
  provider (VI3-INF-2). The session continues from `kv.position()`.
- `Vindex3Runtime::prefill_into(tokens, kv)` — batch prefill into the
  caller's provider (VI3-INF-3), returning the last position's logits so
  generation can sample the first continuation token before resuming
  decode over the **same** provider via `session_with_kv`. A provider
  already holding state is extended from its logical position, so a long
  prompt can prefill in chunks.
- `Vindex3Runtime::plan()` / `Vindex3Runtime::backend()` — the
  model-meaning authority and the arithmetic it executes with.

Backends implement `PlanBackend`
(`larql_vindex::format::vindex3::opplan::exec::backend`). The same plan
runs through a naive f32 reference, the `larql-compute` production
kernels, or the Metal arms — only the arithmetic differs; the
interpreter, and therefore the model's meaning, is shared.

---

## 3. The session layer — `LogitsSession` and generation

`crates/larql-inference/src/vindex3/session.rs` defines the
format-neutral session contract:

```rust
pub trait LogitsSession {
    fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>, InferenceError>;
    fn step(&mut self, token: u32) -> Result<Vec<f32>, InferenceError>;
    fn position(&self) -> usize;
}
```

Logits and state progression — everything generation needs from a model
runtime, and nothing else. Deliberately not another transformer
abstraction: an implementation may be a VINDEX3 plan interpreter, a V2
layer graph, or anything that can advance one token and price a
vocabulary.

`Vindex3Session<'a, B>` is `LogitsSession` over the canonical VINDEX3
incremental executor (`DecodeSession`). Construction loads every operand
once, in the backend's declared weight format — weights stay resident
for the session's lifetime, which is what lets a pointer-keyed device
buffer cache hold the model on the GPU. Construction refuses a plan with
no output head (a session that cannot produce logits cannot serve
generation), and `prefill` refuses an empty prompt. The trait's
tokenwise `prefill` (VI3-INF-0) is a semantic integration gate, not a
fast path — the batch interpreter (`prefill_into`) is the production
prefill.

Above the seam, `crates/larql-inference/src/vindex3/generate.rs` drives
any `LogitsSession` with the existing sampler and EOS machinery
(`Sampler`, `EosConfig` from the layer-graph generation code) — the
driver never learns what a layer is:

- `generate_session(session, prompt, max_new_tokens, sampling, eos, on_token)`
  — prefill then decode, returning a `SessionGeneration { prompt_len,
  tokens }`. A stop token is never included: generation ends *before*
  emitting it. `on_token` fires once per emitted id, in order — the
  streaming surface at this seam.
- `continue_session(session, logits, …)` — the resume-aware driver
  (VI3-SERVE-1): continue from logits already in hand, typically the
  batch prefill's last-position logits. `generate_session` is prefill
  followed by exactly this function, so the two paths cannot drift.

Token ids go in and come out as ids: a tokenizer is part of the fixture
on the V3 path (only one side of a parity comparison may choose it), so
detokenisation composes outside the driver. EOS is judged on token ids
only; stop *strings* need a tokenizer and live above this layer.

---

## 4. The KV seam — `KvState` and `CanonicalKvState`

Continuation state crosses the runtime as a caller-side provider. The
contract lives in the executor
(`larql_vindex::format::vindex3::opplan::exec::kv`, re-exported from
`larql_inference::vindex3`): the `KvState` trait, `LayerKvGeometry`,
`RowKvState` (the executor's plain row provider), and
`plan_kv_geometry`. The provider is `prepare`d with the plan's explicit
per-layer KV geometry — row width and sliding/full window come from the
executable program, never from `ModelArchitecture` inference — and it
owns the logical continuation position. No batch-state → decode-state
translation exists anywhere: batch prefill populates the same provider
that decode resumes from.

`CanonicalKvState` (`crates/larql-kv/src/vindex3/mod.rs`, re-exported as
`larql_kv::CanonicalKvState`) is the VI3-KV-1 adapter: the ordinary
canonical `larql-kv` `KvCache` satisfying the V3 contract with **no
change in semantics**. Deliberately unambitious — no windowing (that is
VI3-KV-2; the executor's span logic owns position exclusion), no
quantisation, no residency optimisation. The KV-1 gates demand
`RowKvState` and `CanonicalKvState` stay bit-identical through prefill,
resume, and decode, chaining `V3 batch == V3 tokenwise == RowKvState ==
larql-kv canonical cache` onto the existing executor ≡ production-forward
parity.

The cache's matrices are the storage authority; the `KvState` read
contract's row views are materialised *from* the matrices after every
write, so the served bits are the stored bits by construction. The same
conversation state can cross the `larql-kv` boundary intact:
`CanonicalKvState::from_cache` adopts an existing (unwindowed) engine
cache as V3 continuation state, `cache()` / `into_cache()` hand it back
to the existing KV machinery, and `geometry()` exposes what the plan
declared.

---

## 5. Serving — binding fork, `V3Model`, the OpenAI surfaces

The V2/V3 distinction is decided **once**, at model binding.
`bootstrap::load_artifact` (`crates/larql-server/src/bootstrap/`)
detects the artifact's container generation
(`larql_vindex::format::generation::detect_generation`) and binds with
the matching loader, returning a `LoadedArtifact`:

```rust
pub enum LoadedArtifact {
    V2(Box<LoadedModel>),
    V3(Box<crate::vindex3::V3Model>),
}
```

A VINDEX3 container binds as an executable program — it structurally
cannot take the V2 path, whose `load_vindex_config` refuses non-V2
generations. Nothing downstream re-detects the format: the server keeps
V3 models in a separate `v3_models` list inside the coherent
`AppState.model_set: RwLock<ModelSet>` snapshot (`crates/larql-server/src/state/model_set.rs`),
and `AppState::served` (same file) resolves a request's model id to a
`ServedModel::V2` or `ServedModel::V3` — the single request-time
decision point.

`V3Model` (`crates/larql-server/src/vindex3.rs`) is one bound container:
the opened `Vindex3Runtime<ProductionBackend>` plus serving glue (an id
derived from the container directory name, and the container's
`tokenizer.json` — the text API cannot serve ids-only). It holds no
`ModelWeights` and no `VectorIndex`, so the old inference path is
structurally unreachable. `load_v3_model` opens the `target` component,
refusing closure defects. `generate_v3` is the SERVE-1 stack for one
request:

```text
VINDEX3 container → Vindex3Runtime → CanonicalKvState
    → prefill_into() → session_with_kv() → continue_session()
    → existing SSE/JSON shaping
```

`crates/larql-server/src/routes/openai/v3_completions.rs` serves
`/v1/completions` on a V3 runtime. Everything wire-shaped is **shared**
with the V2 path (`build_text_completion_chunk`, `finalize_completion`,
the SSE assembly, the response structs), so the two runtimes cannot
drift apart in what a client sees; only the token source differs. The
buffered path runs under the same 504-and-detach server-side timeout
contract as V2, and streaming emits identical chunk shapes, stop
handling, and `[DONE]` termination.

The other OpenAI surfaces have V3 arms through the same single decision
point (`AppState::served`):

- `/v1/chat/completions` — `routes/openai/chat/v3.rs`, dispatched from
  `chat/handler.rs`. Tools and structured output (N0.6) run through the
  **same** schema → FSM → logits-mask pipeline as V2.
- `/v1/responses` — `routes/openai/responses/engine.rs`
  (`ResponsesEngine::V3`), bound in `responses/handler.rs`.
- `/v1/models` lists V3 containers with their generation
  (`routes/models.rs`), and `/v1/runtime` / stats report them
  (`routes/runtime.rs`, `routes/stats.rs`) — a served container never
  silently disappears from a listing surface (the M3 consumer contract,
  `docs/vindex-generation-policy.md`).

Per-request cost note: every request opens a fresh session, which loads
the plan's operands (`DecodeSession::new` keeps weights resident per
session, not per server). Fine for the semantic gate this rung is; a
shared resident session/operand pool is later, perf-shaped work.

---

## 6. CLI entry points

- `larql vindex3 exec` — execute one component's own program from the
  container alone (`crates/larql-cli/src/commands/primary/vindex3_cmd/`),
  with `--backend` selecting the arithmetic realisation (reference,
  production, and the Metal/quantised/lowered arms) and optional
  per-layer hidden-state dumps for `layer-diff` comparison.
  `--generate N` runs greedy autoregressive decode on a `DecodeSession`,
  with phase-separated timing (weight load, prompt ingestion, first
  token, steady decode). Greedy argmax on purpose: generation doubles as
  a fixture, and a sampler would put randomness between two runs of a
  parity comparison.
- `larql serve` — the server. The positional container path (or every
  `.vindex` directory under `--dir`) passes through
  `bootstrap::load_artifact`; V3 containers register into the
  `v3_models` registry and serve over `/v1/completions`,
  `/v1/chat/completions` and `/v1/responses` as above.
  The other `larql vindex3` verbs belong to the container build/gate
  and representation pipelines — `plan`, `encode`, `inspect`, `verify`,
  `ops` (see `docs/vindex3-format.md`) and `represent`, `sensitivity`,
  `consequence` (representation compilation and its quality
  instruments). The format-native read-only surface is the standalone
  `vindex` binary (candidate spec §18.3).
