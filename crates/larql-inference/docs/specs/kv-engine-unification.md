# KV Engine Unification — Specification

**Status:** All 7 steps landed (2026-05-16). Step 6 ships run/walk only; server wiring deferred to the ComputeBackend redesign (§10.6). Step 7 cleanup is partial-by-design: `KvCacheKind` and `generate_cached_backend` are intentionally retained (see §8.7).
**Audience:** LARQL contributors.
**Scope:** Replace the parallel "live decode cache" and "research KV engine"
code paths with a single `KvEngine`-based dispatch, so `larql run` / `larql
walk` / `larql-server` and `larql bench` execute through the same engine
abstraction. Make the `LARQL_KV_ENGINE` selector promised by
`ROADMAP.md:643` real.

This spec covers the engineering refactor only. The correctness contract
for `MarkovResidualEngine` itself is owned by
[`markov-residual-engine.md`](./markov-residual-engine.md) and is not
restated here; this spec depends on that contract holding.

---

## 1. Motivation

Today the repo has two implementations of the same idea:

| Concern | Live decode path | Research engine path |
|---|---|---|
| Trait | `FfnBackend` + raw functions | `KvEngine` (in `larql-kv`) |
| Entry point | `generate_cached_bounded` (`larql-inference/src/forward/kv_generate.rs:125`) | `EngineKind::build(...).prefill / decode_step` |
| Cache shape | Sliding window K/V tensors, optional unbounded growth | Residual-stream + recompute-K/V, or per-engine alternative |
| Reachable from | `larql run`, `larql walk`, server, router | `larql bench --engine` only |
| Tuning surface | `--kv-cache standard\|markov-bounded\|none` + `--context-window` | `--engine markov-rs[:window=N]\|unlimited-context\|turbo-quant\|apollo` |
| Tested by | snapshot tests on `larql run` | 200 unit tests + `bench` smoke |

The duplication has three concrete costs:

1. **Naming collision.** `KvCacheKind::MarkovBounded` (a sliding-window K/V
   cache) and `MarkovResidualEngine` (a residual-stream engine with
   optional bounded window) share a name but are different code paths.
   Reading the code, a contributor cannot tell which "markov" they're
   looking at.
2. **Fabricated capability in ROADMAP.** `ROADMAP.md:643` claims
   `LARQL_KV_ENGINE` selects between the four engines for long-context
   decode. `grep -r LARQL_KV_ENGINE crates/` returns zero matches.
   The capability does not exist; only `bench` can reach the engines.
3. **Research engines never see real prompts.** TurboQuant, Apollo, and
   UnlimitedContext are exercised only by synthetic bench input. Any
   correctness regression that depends on prompt distribution
   (tokenisation edge cases, long-tail vocabulary, multi-turn chat) is
   invisible until someone re-runs bench at the right moment.

The dead-weight `larql-kv` dep on `larql-inference` (`Cargo.toml:104`
with zero `use larql_kv` in `src/`) is a symptom of the same problem:
the dep was added in anticipation of unification, then stranded.

## 2. Decision

Make the live decode cache a `KvEngine` impl. Route `larql run` /
`larql walk` / server decode through `dyn KvEngine` dispatched by
`EngineKind`. Make `LARQL_KV_ENGINE` and `--engine` real on those
commands. Keep the existing `MarkovResidualEngine` contract intact;
this is a wiring change, not a behaviour change for the default path.

## 3. What's in scope, what's out

### 3.1 In scope

- Widening `KvEngine` so it can carry the FFN router (`&dyn FfnBackend`)
  needed for live decode (`--ffn http://...`, MoE shards).
- Introducing two new `KvEngine` impls in `larql-kv` that wrap the
  existing production behaviour exactly: `StandardEngine` (today's
  K/V cache, optionally sliding-window) and `NoCacheEngine` (today's
  O(N²) re-forward fallback). These are the engines `--kv-cache
  standard|markov-bounded|none` resolve to. They preserve current
  behaviour bit-for-bit; they do not replace it with a different
  mechanism.
- Promoting the current `KvCacheKind` flag into engine selection via
  `EngineKind`. The CLI flag continues to accept its existing names
  (`standard`, `markov-bounded`, `none`) for backward compatibility,
  resolving to the new `Standard` / `NoCache` engine variants.
- Wiring `--engine` and `LARQL_KV_ENGINE` into `larql run` and
  `larql walk`, giving access to the full engine catalog (Standard,
  NoCache, MarkovResidual, UnlimitedContext, TurboQuant — see §5).
- Migrating `walk_cmd.rs:1049-1075` and the server's decode entry to
  dispatch through `dyn KvEngine`.
- Moving `larql-kv` from a dead Cargo.toml dep to a live one in
  `larql-inference`, or relocating the trait so the dep direction is
  clean (see §6).
- Bench help text honesty — advertise all engines `EngineKind::from_name`
  parses, or remove the ones we don't intend to ship.

### 3.2 Out of scope (deliberately)

- Apollo. `EngineKind::Apollo` is a boundary-store + residual-injection
  scheme, not a peer K/V cache. Forcing it through the unified `KvEngine`
  shape would either compromise the trait or hide the impedance mismatch.
  Apollo stays in `larql-kv` and remains opt-in for `bench --engine apollo`,
  but it is **not** added to the run/walk dispatch. See §7.
- Promoting a non-standard engine as the *default* on long context. The
  ROADMAP C7 ambition ("Apollo at 20,000× for long context") is
  downstream of this work and is not decided here.
- New engines. This is a unification of the four that exist.
- Changing the `MarkovResidualEngine` correctness contract or any
  engine's measured behaviour.

## 4. Trait widening

### 4.1 The obstacle

Live decode threads an `&dyn FfnBackend` so `--ffn http://...` can route
FFN layer-by-layer to a remote `larql-server` (and so `--moe-shards`
can route MoE experts). The trait today (`larql-kv/src/lib.rs:60-126`)
takes only `(weights, token_id)`:

```rust
fn prefill(&mut self, weights: &ModelWeights, token_ids: &[u32]) -> Option<Array2<f32>>;
fn decode_step(&mut self, weights: &ModelWeights, token_id: u32) -> Option<Array2<f32>>;
```

The four existing engines all assume FFN is local (computed from
`weights`). They never had a reason to support remote FFN dispatch.

### 4.2 The change

Add an FFN router to the trait. Two shapes considered:

**Option A — pass `&dyn FfnBackend` per call (selected):**

```rust
fn prefill(
    &mut self,
    weights: &ModelWeights,
    ffn: &dyn FfnBackend,
    token_ids: &[u32],
) -> Option<Array2<f32>>;

fn decode_step(
    &mut self,
    weights: &ModelWeights,
    ffn: &dyn FfnBackend,
    token_id: u32,
) -> Option<Array2<f32>>;
```

Same widening applied to `prefill_q4k` / `decode_step_q4k`.

**Option B — engine owns `Box<dyn FfnBackend>`:**

```rust
EngineKind::MarkovResidual { window_size }.build(weights, ffn, backend)
```

Rejected because (i) the FFN backend can change mid-session (chat with
`--ffn` reconnecting), (ii) MoE shards are per-prompt config, (iii) it
forces a lifetime parameter onto `Box<dyn KvEngine>` that infects every
caller.

Option A is the spec.

### 4.3 Default impl for engines that don't care

Engines that compute FFN locally from `weights` should not need to
change. The trait accepts the parameter; default `prefill_q4k` /
`decode_step_q4k` continue to dispatch to `prefill` / `decode_step`
with `ffn` forwarded.

### 4.4 What does *not* go on the trait

- `LayerHook` integration. Hooks (`generate_cached_hooked`,
  `kv_generate.rs:174`) are a research-only path; the production decode
  doesn't fire them. Hook-aware decode remains its own entry point,
  parallel to engine dispatch. Mixing the two would force every engine
  to thread per-layer hook callbacks through their internal forward
  passes; the cost is not justified by current usage.
- `on_token` streaming callback. The caller drives the decode loop and
  calls `engine.decode_step` per token, so streaming stays in the
  caller. Engines do not own the loop.
- Tokenizer. Tokens enter the engine as `&[u32]`; tokenisation is the
  caller's concern.

## 5. Engine catalog after unification

`EngineKind` grows two variants (`Standard`, `NoCache`) that wrap the
current production code paths, and keeps the four research engines. The
default path is bit-identical to today's `--kv-cache standard`.

| EngineKind variant | Maps to current CLI flag | Underlying mechanism | Status after unification |
|---|---|---|---|
| `Standard { window_size: None }` | `--kv-cache standard` (default) | Current `generate_cached_bounded(window: None)` — full K/V tensor cache, unbounded growth | **Default**, bit-parity with current production |
| `Standard { window_size: Some(N) }` | `--kv-cache markov-bounded --context-window N` | Current `generate_cached_bounded(window: Some(N))` — sliding-window K/V tensor cache | Opt-in via flag, bit-parity with current production |
| `NoCache` | `--kv-cache none` | Current full re-forward per step (O(N²)) | Opt-in via flag, bit-parity with current production |
| `MarkovResidual { window_size }` | (new) `--engine markov-rs[:window=N]` | Stores residuals, recomputes K/V at decode. **Different mechanism** from `Standard`; bit-identical output under preconditions per `markov-residual-engine.md` | Opt-in; research engine promoted to live path |
| `UnlimitedContext { window_size }` | (new) `--engine unlimited-context:window=N` | Per-window K/V checkpoints | Opt-in, advertised as experimental |
| `TurboQuant { bits }` | (new) `--engine turbo-quant:bits=N` | Quantised K/V | Opt-in, advertised as experimental |
| `Apollo { ... }` | (unchanged) `larql bench --engine apollo` only | Boundary store + residual injection | **Not wired into run/walk**, see §7 |

`Standard` is a thin `KvEngine` adapter over `generate_cached_bounded`;
no new cache code, just a trait impl. `NoCache` is similarly a thin
adapter over the existing full-forward loop. `MarkovResidual` is the
existing residual-stream engine and is **not** an alias for `Standard`
— it's a different mechanism, exposed as a peer option.

## 6. CLI and env surface

### 6.1 `--engine` flag on run/walk

```
larql run <model> --engine markov-rs:window=1024
larql walk <model> --predict --engine unlimited-context:window=256
```

Accepts the same spec strings that `bench --engine` parses, via the
existing `EngineKind::from_name` (extended to recognise the new
`standard` and `none` names per §5).

The legacy `--kv-cache` flag remains and resolves to:

| `--kv-cache` value | Resolved `EngineKind` |
|---|---|
| `standard` (default) | `Standard { window_size: None }` |
| `markov-bounded` | `Standard { window_size: Some(--context-window) }` |
| `none` | `NoCache` |

If both `--kv-cache` and `--engine` are passed, `--engine` wins; passing
both is a `WARN`-level log, not an error.

### 6.2 `LARQL_KV_ENGINE` env var

Same parser as `--engine`. Precedence: CLI flag > env var > default.
Honoured by `larql run`, `larql walk`, and the server. Not honoured by
`bench` (which has its own explicit `--engine` already).

### 6.3 Bench help text

`bench/args.rs:39-40` and `bench/run.rs:143,179` currently advertise
`markov-rs, unlimited-context`. Updated to the full set
`EngineKind::from_name` parses: `standard, no-cache, markov-rs,
unlimited-context, turbo-quant, apollo`. Experimental ones
(`turbo-quant`, `apollo`) annotated as such; `no-cache` annotated as
O(N²) debug fallback. No silent capability hiding.

## 7. Apollo carve-out

`EngineKind::Apollo { injection_layer, inject_coefficient, top_k }` is
not a K/V cache strategy. From `larql-kv/src/engines/apollo/`:

- Stores boundary residuals captured at window-end positions.
- At decode, injects top-k boundary residuals into the residual stream
  at a specified layer with a scaling coefficient.
- Does **not** maintain a sliding window of K/V or a residual hot tier
  in the `MarkovResidualEngine` sense.

Forcing Apollo behind the same trait as "sliding window K/V" obscures
that it is a different mechanism with different correctness
expectations (Apollo's `markov-residual-engine.md:393` row 5 says
"first-token factual, not bit-exact"). Two consequences:

1. Apollo **stays in `larql-kv`**, remains a valid `EngineKind` variant,
   and remains reachable via `bench --engine apollo`.
2. Apollo is **not wired into `larql run` / `larql walk`** in this
   unification. A future spec — or a sibling trait
   (`ContextRecallEngine`?) — handles Apollo's wiring on its own terms.
3. The `--engine apollo` spec in `run`/`walk` returns an explicit
   `not-yet-wired-for-live-decode` error. It does not silently fall
   back to a default engine.

This carve-out is the load-bearing reason the unification is shippable
on a short timeline. Without it, the trait either grows a second
abstract method group (boundary store) or starts leaking
implementation-specific concepts (`injection_layer`) into the live
dispatch path.

## 8. Migration plan

Each step is independently reversible. Each step has a parity guarantee
so the default user experience is byte-identical until §8.5.

### 8.1 Step 1 — move trait surface into `larql-inference` ✅ landed

Move the **trait surface** out of `larql-kv/src/lib.rs` into
`larql-inference` (new module, e.g. `larql-inference/src/kv_engine.rs`):

- `KvEngine` trait
- `EngineInfo` struct
- `DecodeStageSummary` (currently in `larql-kv/src/profiler.rs`)

`EngineKind` and its `from_name` / `build` methods **stay in
`larql-kv`**. They reference concrete engine impls; moving them
would re-introduce the cycle.

`larql-kv`'s `src/lib.rs` keeps the same public surface via
`pub use larql_inference::{KvEngine, EngineInfo, DecodeStageSummary}`
so external callers see no change.

`larql-inference` does **not** gain a dep on `larql-kv`. `larql-kv`
keeps its existing dep on `larql-inference`. Engine impls now write
`impl larql_inference::KvEngine for ...` (or `impl KvEngine` via a
local `use` of the re-export, same effect).

`larql-inference::test_utils::{make_test_weights, make_test_vindex}`
stays where it is; `larql-kv`'s unit tests continue to consume it via
its existing dep on `larql-inference`.

No new crate. `ModelWeights` is already in `larql-models` and does
not move.

**Parity:** Pure trait relocation behind re-exports. Every crate
compiles. Every test passes. No semantic change anywhere.

### 8.2 Step 2 — trait widening (no behaviour change) ✅ landed

Widen `KvEngine::{prefill, decode_step, prefill_q4k, decode_step_q4k}`
to accept `&dyn FfnBackend`. `FfnBackend` itself is decided at
implementation time:
- If `FfnBackend` is small and self-contained, move it to
  `larql-core` so `larql-kv` can name it in the trait signature.
- If it pulls in too much inference-side machinery, define a minimal
  `FfnDispatch` trait in `larql-core` that `FfnBackend` blanket-impls,
  and `KvEngine` takes `&dyn FfnDispatch`. Pick whichever keeps
  `larql-core` smaller.

Existing four engines ignore the parameter in their bodies (FFN is
recomputed from `weights` as before). Bench call sites pass a local
FFN backend constructed from `weights`. All 200 `larql-kv` tests pass
without modification.

**Parity:** `larql bench --engine *` byte-identical pre/post.

### 8.3 Step 3 — introduce `Standard` and `NoCache` engines ✅ landed

Add two new `KvEngine` impls in `larql-kv`:

- `StandardEngine` — internally calls (or contains a copy of) the body
  of `generate_cached_bounded`. `prefill` runs the prefill loop;
  `decode_step` runs one autoregressive step against the K/V cache it
  owns. Takes `window_size: Option<usize>` at construction (`None` =
  unbounded, `Some(N)` = sliding window).
- `NoCacheEngine` — `prefill` records the prompt token IDs;
  `decode_step` appends and re-runs the full forward pass over the
  growing sequence. O(N²); debug fallback only.

Add `EngineKind::Standard { window_size: Option<usize> }` and
`EngineKind::NoCache` variants. Extend `EngineKind::from_name` to parse
`standard[:window=N]` and `no-cache` / `none`.

Trait widening from §8.2 means both engines accept `&dyn FfnBackend`
in their `decode_step` signatures.

**Parity:** No call site changes yet. `larql run` / `larql walk` still
go through `generate_cached_backend` directly. The engines exist but
are unused outside their own unit tests.

### 8.4 Step 4 — dispatch through `KvEngine`, opt-in ✅ landed (parity gate met)

Behind an internal feature gate (e.g. `LARQL_KV_ENGINE_DISPATCH=1`),
`walk_cmd.rs:1049-1075` dispatches through
`EngineKind::build(backend).prefill_q4k(...) +
decode_step_q4k(...)` in a token loop, replacing the call to
`generate_cached_backend`. The current `KvCacheKind` flag values map
to `EngineKind` via the table in §6.1. Default off; existing path
remains the fallback.

**Parity gate:** Side-by-side run with and without the env var on
`bench/baselines/cpu/` prompts. Must produce identical token streams
on all three `KvCacheKind` values (`standard` → `Standard { None }`,
`markov-bounded` → `Standard { Some(N) }`, `none` → `NoCache`). If
parity is not bit-exact on any of the three, this step does not land.
Step 5 cannot start until all three pass.

### 8.5 Step 5 — flip default ✅ landed (env gate removed; engine dispatch is the only path in `walk_cmd::generate_stream`)

Remove the env var gate; `walk_cmd.rs` dispatches through `KvEngine`
unconditionally. `generate_cached_backend` becomes a thin wrapper that
constructs `EngineKind::Standard { window_size }` and dispatches; the
wrapper either stays (as a convenience) or gets inlined and deleted in
a follow-up cleanup commit.

**Parity:** Existing snapshot tests on `larql run` must still pass
byte-for-byte.

### 8.6 Step 6 — surface ✅ landed (run/walk only; server deferred to ComputeBackend redesign per §10.6)

- Add `--engine` flag to `run_cmd` and `walk_cmd` (parsed by
  `EngineKind::from_name`).
- Add `LARQL_KV_ENGINE` env var (read by the same code path) on
  `run` / `walk` / `larql-server` decode.
- Wire engine selection into `larql-server`'s decode path. Default
  `Standard { window_size: None }`. Apollo rejected with a clear
  error if requested via the server (bench-only).
- Update `ROADMAP.md:643` C7 entry from "shipped (4 engines) but opt-in"
  to "shipped; `LARQL_KV_ENGINE` / `--engine` honoured on run / walk /
  server; default `standard` (current production K/V cache); MarkovRS /
  UnlimitedContext / TurboQuant opt-in; Apollo bench-only".
- Update bench help text (§6.3).

### 8.7 Step 7 — cleanup ✅ landed (partial-by-design, 2026-05-16)

- **`KvCacheKind` retained** (`larql-cli/src/commands/primary/run_cmd.rs:45`).
  Backward-compat on the `--kv-cache` flag earns its keep — `--kv-cache
  standard|markov-bounded|none` continues to parse and resolve to
  `EngineKind` via the table in §6.1.
- **`generate_cached_backend` retained as parity oracle**
  (`larql-inference/src/forward/kv_generate.rs:69`). Step 4's
  bit-parity gate (`larql-kv/src/engines/standard.rs:264`) and the
  legacy arm of `engine_decode.rs` bench measure new engine dispatch
  against it. Deleting the wrapper would erase the reference
  implementation that future engines' bit-parity claims are validated
  against. Kept on purpose; not a TODO.
- **`larql-inference` → `larql-kv` dep is dev-only**
  (`larql-inference/Cargo.toml:104` under `[dev-dependencies]`). The
  spec originally predicted this would become a live dep "consumed by
  the dispatch code path"; what landed is cleaner. `larql-cli` bridges
  `larql-kv` (builds engines via `EngineKind::build`) and
  `larql-inference` (calls `generate_with_engine`), so
  `larql-inference` core code never names `larql-kv`. The only
  remaining consumer in `larql-inference/` is
  `examples/apollo_rd_backend.rs`, which justifies the dev-dep.

### 8.8 Rollback

Steps 1-4 are pure additions and can be reverted by deleting the new
code (step 1's `larql-core` extraction is a pure type relocation, so
even that is reversible if the type moves don't break anything
unexpected). Step 5 (default flip) is the only step that changes
user-visible behaviour for the default path; the parity gate in §8.4
is the guarantee that flipping it is a no-op. If a regression
surfaces post-§8.5, revert §8.5 only; §1-4 stay landed and the engine
surface remains opt-in via the env var.

## 9. Non-goals

- **Speedup.** This unification is a refactor. End-to-end tok/s should
  be unchanged on the default path.
- **Promoting MarkovRS / UnlimitedContext / TurboQuant as the default.**
  Their compression ratios are isolated-kernel measurements; whether
  they win end-to-end on real prompts is exactly what running them
  through `larql run` will tell us. Promotion happens in a separate
  decision after data lands.
- **New engine architectures.** No new variants of `EngineKind`.
- **Hot-window serialisation.** Out of scope; covered by
  `markov-residual-engine.md:274` and not re-litigated here.
- **Apollo wiring into run/walk.** See §7.

## 10. Open questions (resolve before implementation)

### 10.1 Spec location

**Resolved:** stays in `larql-inference/docs/specs/` alongside
`markov-residual-engine.md`. The consumer side is the larger change,
and contributors already look for inference specs there.

### 10.2 Disposition of `KvCacheKind::None`

**Resolved:** keep it, as a `NoCacheEngine` impl of `KvEngine` (option
b in the original draft). The trait expresses "store nothing between
steps" by having `NoCacheEngine` retain only the prompt token IDs and
re-run a full forward pass per `decode_step`. The user-facing
`--kv-cache none` flag and the underlying O(N²) correctness fallback
both stay reachable, and they flow through the same dispatch as every
other engine.

Rationale: the no-cache path is the correctness reference against
which every other engine's bit-parity claim is ultimately measured.
Removing it would make that audit trail harder to run. Keeping it as
a parallel non-engine path (option c) would re-create exactly the
two-code-path problem this unification exists to solve.

Engine catalog and migration plan already reflect this resolution
(§5, §8.3).

### 10.3 Apollo's home

**Resolved (deferred):** Apollo stays in `larql-kv`, stays an
`EngineKind` variant for `bench --engine apollo`, and is **not wired
into run/walk/server** in this unification. The decision between a
sibling trait (`ContextRecallEngine`) and subtrait composition
(separate `ResidualInjector`) is deferred to a follow-up spec
written when Apollo actually needs production wiring. This
unification is shippable without that decision.

### 10.4 Dep direction

**Resolved:** move the `KvEngine` trait (+ `EngineInfo`,
`DecodeStageSummary`) from `larql-kv` into `larql-inference`.
`larql-kv` re-exports them so its public API stays the same. No new
crate.

This is option (a) from the original draft. The earlier proposal to
extract a `larql-core` crate was based on the assumption that
`ModelWeights` lived in `larql-inference` (it does not — it's in
`larql-models`, see §10.4 history notes). With `ModelWeights`
already in the right place, the only remaining cycle concern is the
trait itself — and that resolves cleanly by moving it up one level.

Why no cycle:

- `larql-inference` defines the trait and a `dispatch_decode(engine:
  &mut dyn KvEngine, ...)` entry point. It does **not** gain a dep on
  `larql-kv`.
- `larql-kv` keeps its existing dep on `larql-inference`. Engine
  impls write `impl larql_inference::KvEngine for ...`. They already
  depend on inference internals (`forward::*`, `attention::*`,
  `residual::*`) — adding the trait dep is free.
- `larql-cli` depends on both, bridges them: parses `--engine`,
  calls `larql_kv::EngineKind::build()` → `Box<dyn KvEngine>`,
  hands the result to `larql_inference::dispatch_decode`.

Post-unification dep graph:

```
larql-models, larql-compute, larql-vindex   (unchanged)
   ↑
larql-inference (forward pass + KvEngine trait + dispatch loop)
   ↑
larql-kv (engine impls + EngineKind::build)
   ↑
larql-cli (orchestrates: builds via larql-kv, dispatches via larql-inference)
```

`EngineKind` stays in `larql-kv` — its `build` method constructs
concrete engine impls and pulls in their modules.
`larql-inference`'s dispatch only references the trait, never
`EngineKind`, so it remains independent of the engine catalog.

The move is its own migration step — see §8.1.

### 10.5 Default value of `LARQL_KV_ENGINE`

**Resolved:** when the env var is unset and no flag is passed, the
default is `EngineKind::Standard { window_size: None }` — the current
production K/V cache. Bit-identical to today's `--kv-cache standard`.
No change to user-visible default.

### 10.6 Server / router

**Revised (2026-05-16):** server wiring is **deferred** until the
ComputeBackend redesign lands. The original plan to wire
`LARQL_KV_ENGINE` into the server assumed the server's decode path
shared a code surface with `larql run`/`larql walk`. It doesn't —
`larql-server`'s `handle_stream_generate` dispatches through
`larql_inference::layer_graph::generate_streaming` (the Metal
layer-graph fast path), not `generate_with_engine` (the CPU KV-cache
path). Routing the server through `generate_with_engine` would
silently downgrade GPU decode to CPU, which is a real tok/s
regression — not acceptable as a default behaviour.

The correct fix is to give `KvEngine` a GPU side via a redesigned
`ComputeBackend` trait, then plumb server decode through the unified
surface. That work is its own spec
(`compute-backend-redesign.md`, planned), and step 6's server
wiring lands as part of it. Until then `larql-server` continues
using `generate_streaming` directly and ignores `LARQL_KV_ENGINE`.

The Apollo carve-out for the server still applies when wiring
eventually lands: the "first-token factual, not bit-exact" property
(`markov-residual-engine.md:393`) is unlikely to meet server SLAs.

---

## Appendix: state of the world today

For reviewers, the concrete file pointers this spec is built on:

- Live cache entry point: `larql-inference/src/forward/kv_generate.rs:125` (`generate_cached_bounded`)
- Live cache dispatch from CLI: `larql-cli/src/commands/extraction/walk_cmd.rs:1049-1075`
- Live cache flag enum: `larql-cli/src/commands/primary/run_cmd.rs:33-44` (`KvCacheKind`)
- Engine trait: `larql-kv/src/lib.rs:60-126` (`KvEngine`)
- Engine selector: `larql-kv/src/lib.rs:131-251` (`EngineKind`)
- Engine impls: `larql-kv/src/engines/{markov_residual,unlimited_context,turbo_quant,apollo}/`
- Bench consumer: `larql-cli/src/commands/primary/bench/run.rs:100-185`
- FFN router trait: `larql-inference/src/ffn/mod.rs:31` (`FfnBackend`)
- ROADMAP entry needing correction: `ROADMAP.md:643` (C7)
- Dead-weight dep: `larql-inference/Cargo.toml:104`

The 200 `larql-kv` tests passing today is the safety net for §8.2's
trait widening — any widening that breaks them is wrong.
