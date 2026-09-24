# Roadmap — larql-kv

## Semantic promotion — Phase A/B landed, physical seam open (2026-08-05)

`engines/semantic_promotion/` (policy wrapper over an exact decode engine)
and `model_walk/` (its peer planner) are built and gated: 1 120 tests in
this crate, clippy clean, workspace green. Committed 2026-08-06 in
`c1f24291` (bundled into a compute commit rather than the five prepared
ones). Full pickup notes, evidence ledger and ordered queue:
**`docs/semantic-promotion-pickup.md`**. Contract:
`docs/specs/semantic-promotion-engine.md`.

**What the measurements settled.** A late canonical record independently
reconstructs the answer *payload* in a suffix that never saw the source —
3/3 at 4K and 3/3 at ~49K distance, controls clean (EXP-25CF). It does
**not** reconstruct the termination decision: all 12 canonical arms peak
on the first termination token, in both histories, at both distances. So
the runtime permission is `ExactPayloadLength`, never `FirstTermination`,
which is what `RetirementScope::check_covered_by` enforces.

A `DerivedClaim` is the *stronger* replacement at range — at ~49K it
carries payload and trajectory — and actively wrong at short range, where
the model reads it as a fresh operand and runs the operation twice
(7432 → 7433). The switch is abrupt, between 48K and 64K, and identical
across counterfactual histories, so source-conditioned state is excluded.
The corrupt arm is non-monotone across the same sweep, so consumption is
not a function of geometry alone. Certificates therefore carry
`RegimeEvidence::Point` (never an interpolated band) keyed on both
`record_digest` and `materialisation_digest`.

**Scope correction — read before quoting Phase B.** These results are
proven on the CPU per-layer executor. `kv_dispatch::helpers` never
consults `arch.is_sliding_window_layer`; the global/sliding split is a
promotion-wrapper policy, not attention. Ragged per-layer extents are
mechanically valid; that they correspond to production architectural
visibility is **pending**, and R8d is the gate for it.

**Next implementation seam.** `larql_inference::kv_row_positions` exists
and is gated but **not wired**. `clip_kv` keeps the tail *N physical
rows*, so across a positional hole it retains rows thousands of positions
old inside a small window (`r7b`). Wiring order, atomicity requirement and
the R8-storage / R8-policy split are in the pickup notes §5.

**Next experiment.** Source distance is ruled out (answer-mode at
source→query 3K/8K/16K/32K at fixed 64K regime, versus operand at ~3K in
a 4K context). The controlling variable is in the global-context regime,
so the next run holds all three semantic distances fixed and varies an
unrelated prefix. Queue in pickup notes §6.

---

## KV residency contract — documented and tested, not fixed (2026-08-08)

The fused Metal decode path and the coarse `KvDispatch` path implement
**identical attention semantics and different physical-residency
contracts**. Both clamp what attention reads via `attention_span`; only
the coarse path bounds what is retained, by calling
`compact_kv_to_window` each step. So on `larql run` / `larql bench`,
sliding layers accumulate rows without limit.

Three quantities that get conflated, and must not be: architectural
reachability (`W` = 1024), logical occupancy (`<= 2W`, the
`COMPACTION_SLACK` bound), physical capacity (4096, fixed at
allocation). **Compaction reclaims occupancy, not bytes** — wiring it
into the fused path saves no memory, because the buffer is still
allocated at 4096 rows.

Per-layer capacity *would* save ~464 MB on Gemma 3 4B (1.09 GB → ~624
MB), but it is blocked on prefill rather than on allocation: prefill
bulk-copies `seq_len` rows with a bound that is enforced nowhere local,
so shrinking a sliding layer's capacity turns it into an overrun. The
open question — whether a sliding layer needs every prefill row
materialised at once or only the terminal tail — plus the evidence, the
blast radius of the shape-tuple approach (134 refs / 26 files / 5
crates), and why the eventual seam should be setup-time configuration
rather than a per-layer callback, are all in
[`docs/kv-residency-contract.md`](../../docs/kv-residency-contract.md).
Executable form:
[`kv_residency_contract.rs`](../larql-compute-metal/src/kv_residency_contract.rs).

---

## MAP-5 persistence harness — scoped and verified, not started (2026-08-02)

A parallel mechanistic-interpretability thread (MAP-5/5b/5c, registry
`map`, see `crates/larql-inference/examples/map5*.rs`) found that a
late-layer residual can act as a compact, content-specific executable
waypoint: splicing a donor prompt's residual into a recipient at one
position/layer redirects the recipient's output to the donor's answer,
with separate, measurably different depth thresholds for redirecting
the immediate next token vs. sustaining a later branch under repeated
re-injection. The open question — does this persist as genuine hidden
state, or is it entirely mediated by which token gets emitted — could
not be answered by MAP-5b's harness, because `trace_forward_full`/
`patch_and_trace_with_ffn` (the activation-patching primitives it used)
recompute the ENTIRE hidden state from raw token IDs on every call, with
no persistent state threaded between generation steps. Testing genuine
persistence needs a real incremental decode loop with actual K/V
continuity across steps — which is most of what this crate already is.
**Scoped here so the harness can be picked up once VINDEX3 lands**,
without duplicating this crate's own state/execution machinery.

**What already exists (verified against the current code, not assumed):**

- `StandardEngine::prefill(weights, ffn, token_ids)` and
  `::decode_step(weights, ffn, token_id)` (`engines/standard.rs`) are
  genuine incremental decode: `token_id` is explicit per step, state
  (`handles: Option<Vec<KvHandle>>`, `abs_position`) lives in `&mut
  self` and persists across calls — not a recompute-from-scratch design.
  This directly gives arm **C** (donor-token-only, no residual patch:
  `engine.prefill(recipient)?; engine.decode_step(donor_token)?;`) and
  arm **D** (residual patch, then force the recipient's own token into
  the next step) as thin drivers, no new engine code.
- `KvHandle` (`larql-compute::kv_dispatch::handles`) is real but
  **opaque** — `cached_len()`/`kv_dim()`/`backend_name()`/`as_any()`
  only, no per-row read or write. `StandardEngine` migrated OFF the old
  `larql-kv::cache::KvCache` (with its `get_layer`/`set_layer`/
  `clone_layer_from`/`clone_layer_position_range`) on 2026-05-16 per its
  own module doc — that API is not reachable from the current handle-based
  decode path. Earlier notes citing those methods as usable for this
  harness were wrong; they exist, but on a representation `StandardEngine`
  no longer touches.
- `KvEngine` (`larql-inference::kv_engine`) has 8 implementors —
  `StandardEngine`, `NoCacheEngine`, `WindowedCheckpointEngine`,
  `MarkovResidualEngine`, `MarkovResidualCodecEngine`, `TurboQuantEngine`,
  `BoundaryPerLayerEngine`, `BoundaryKvEngine` (`ApolloEngine` implements
  a separate `RetrievalEngine` trait, not this one) — a real, already-built
  catalogue of different continuation-state policies (canonical K/V vs.
  canonical residual with derivative K/V, etc., per `docs/state-policy.md`
  §5's slotting table), reachable through one dispatch surface.
- **Correction to an earlier framing**: `docs/state-policy.md`'s
  `(canonical_state, derivative_state, correctness_contract)` triple is a
  **draft proposal (v0.1)**, not an implemented framework — the doc's own
  §7 states refactoring `KvEngine` to formalize it is explicitly a
  non-goal for now. It's a good *descriptive* lens for reasoning about
  the engines (and for interpreting this harness's own results), but it
  is not code anyone can call today.
- No hook anywhere lets a caller intercept or mutate raw K/V during
  attention — confirmed by direct search (`before_attention`,
  `before_append`, and any K/V-observing `LayerHook` method: zero
  matches). `LayerHook::on_post_attention`/`on_post_layer`
  (`larql-compute::forward::hooks`) only ever see the **residual**, after
  attention has already run and consumed whatever K/V existed — this is
  the one genuinely missing piece.

**What's actually missing, precisely:** an intervention-capable bridge
between an incremental engine's K/V state and the attention computation
that creates/reads/appends it — nothing else needed for the six-arm
design (see `map-5`'s chuk-experiments record for the arm definitions
and how they map onto existing vs. new machinery). A rough shape, not a
commitment:

```rust
// Proposed — does not exist yet. Lives in larql-compute (low-level
// observation/mutation over the real attention/append path), surfaced
// through a larql-inference hook type, orchestrated by an larql-kv
// experimental driver (NOT folded into KvEngine itself — that trait
// stays production-shaped: continuation state, advance, correctness
// contract; a raw-K/V research hook doesn't belong on its signature).
pub trait KvIntervention {
    fn before_attention(&mut self, layer: usize, position: usize, cache: &mut dyn MutableKvView);
    fn before_append(&mut self, layer: usize, position: usize, key: &mut [f32], value: &mut [f32]);
}
```

`MutableKvView` would need to distinguish current-token K/V (about to be
appended) from historical-cache K/V (already stored, read by this
step's attention) from whole-layer transplant from position-range
transplant from K-only/V-only mutation — conflating these makes any
result hard to interpret. CPU implements it first; Metal/opaque
whole-model-handle backends report unsupported rather than silently
falling back to a CPU copy.

**Do not build a parallel experimental decode loop.** The correct shape
is an instrumented research mode bolted onto `StandardEngine`'s existing
CPU path (selectively inspectable/forkable), with the fast/opaque
production path left untouched — not a second engine competing with this
crate's own state-policy catalogue.

**Explicitly deferred pending VINDEX3** (see `ROADMAP_STATUS.md`'s
2026-08-02 entry and chuk-experiments `map-5`'s `next_action`) — this
section exists so the scoping work already done isn't lost, not as a
signal to start now.

## Spin-barrier pool — CPU MoE decode caught llama.cpp (2026-06-13)

## Current state (2026-08-04)

Nine engines behind one `EngineKind` selector at the time of writing (ten
since semantic promotion; `EngineKind::supported_names()` is the roster), all reachable from
`larql bench --engine` and pinned by `tests/gpu_engine_parity`,
`tests/dispatch_parity` and `tests/engine_ple_parity`.

### Which path a run actually takes

This is the thing to know before reading any engine number, because it
decides what was measured:

| path | when | what runs |
|---|---|---|
| **coarse (fused)** | Q4K vindex present, backend accepts the engine's window | backend's fused pipeline; K/V lives in the backend |
| **coarse + window** | as above, window bounded by the backend | same, with attention and K/V bounded |
| **per-layer → host** | prompt longer than the window, or an arch that declines coarse | generic per-layer loop — **on Metal this runs the whole forward on the CPU** |

A bench row reports which one it took (`[coarse]` / `[per-layer→host]`).

**On the coarse path the engine's own state policy is not engaged** — the
K/V sits in the backend behind a sentinel handle, so `standard`,
`markov-rs`, `markov-rs-codec` and `boundary-per-layer` execute the same
kernels and land within ~0.5% of each other. Ranking those four against
one another on that path measures nothing. `turbo-quant` and the
windowed configs do differ, because their state genuinely materialises.

### Windows

`window=N` promises bounded attention *and* bounded K/V. Both backends
honour it on the fused path via `coarse_prefill_windowed` /
`coarse_decode_step_windowed`, which fail closed — a backend that cannot
bound both answers `None` and the engine falls back to per-layer.

CPU trims the cache before each step. Metal clamps the attention span
every step and compacts at 2x the window, so the memmove is O(1)
amortised; up to 2x the window is resident between compactions, but
attention never reads past the window.

Measured, Gemma 3 4B Q4K, Metal, 80 steps at `window=8`: 11.61 ms /
2.4 MB, against 12.06 ms / 23.6 MB unwindowed.

Per-layer sliding-window attention (the *architecture's* window, e.g.
Gemma 3's 1024) is implemented on the CPU attention path and resolved
through one rule shared with the Metal pipeline spec,
`effective_attention_window_for_layer`.

### Measurement

A bench step is a **whole token** — engine forward + lm_head + next-token
pick — matching the reference rows, with a `fwd=` / `head=` split in the
row note. Memory counts K/V the backend holds on the engine's behalf, not
just what the engine allocated, and declines to print a ratio when
nothing was measured.

Cross-backend: prefill agrees to ~5e-7 relative L2; per-step decode
differs by a stable ~3e-3 that does not compound (two Q4K kernels
rounding differently).

### Known limits

- A prompt **longer** than the window still takes the per-layer path: the
  fused prefill has no per-query-position masking, so accepting would
  attend the whole prompt while advertising a bound.
- On Metal the per-layer path delegates every dispatch method to
  `CpuBackend`, so anything that declines coarse runs on the host —
  worth ~9x on Gemma 3 4B.
- `apollo` needs an attached boundary store; without one `prefill` fails
  closed with `RetrievalMiss`, and it is excluded from the criterion
  bench for that reason (`EngineKind::bench_excluded_names`).

See [CHANGELOG.md](CHANGELOG.md) for how each of these came to be.

## Coverage debt

**Status (2026-05-24 — CLOSED.)** All six files below the 90% per-file
floor have been lifted; `make larql-kv-coverage-policy` passes
against fresh `summary.json` regeneration. Workspace total 95.62%
lines, 61/61 files at ≥90%, 0 debt baselines in
`coverage-policy.json`.

| File | Pre | Post |
|---|---:|---:|
| `engines/markov_residual/compute.rs` | 86.85% | **95.30%** |
| `engines/windowed_checkpoint/dispatch.rs` | 59.09% | **97.24%** |
| `engines/markov_residual/dispatch.rs` | 77.51% | **96.78%** |
| `engines/markov_residual_codec/dispatch.rs` | 80.68% | **97.72%** |
| `engines/turbo_quant/dispatch.rs` | 9.35% | **97.85%** |
| `engines/boundary_per_layer/dispatch.rs` | 7.95% | **93.57%** |

**Implementation summary.** No new shared mock infrastructure was
needed: `CpuBackend` (via `cpu_engine_backend()`) already implements
`coarse_*_with_state` for the synthetic Q4K fixture
(`make_test_q4k_weights` + `make_test_q4k_vindex`), which drives
every dispatch happy-path through real per-layer state capture.
~50 new `#[cfg(test)] mod tests` cases added inline per dispatch
file plus ~10 env-var-gated cases in `compute.rs`. Zero regressions;
`make larql-kv-ci` passes.

**Env-var-gated paths — thread-local override pattern.** The
`LARQL_MARKOV_*` (compute.rs walk-KV diagnostics) and
`LARQL_W10_DISABLE` (dispatch mask cascade) helpers were
near-impossible to test safely under `cargo test --jobs N`: setting
process-global env from one test races every other parallel test
that consults the same var (caught a real flake in
`prefill_with_overflow_creates_encoded_cold_tier`). Resolution: each
env helper now consults a per-thread `RefCell` override map
*before* falling back to `std::env::var`. Tests inject values into
the thread-local; production reads env unchanged. No `serial_test`
crate needed, no `#[serial]` annotations, no env mutation. The
helpers:

- `compute.rs::read_markov_env(key)` + `set_markov_env_override(...)` /
  `clear_markov_env_overrides()` (test-only).
- `engines/mod.rs::w10_enabled()` + `set_w10_disabled_override(...)`
  (test-only).

**Open design questions — resolved by the work above.**

1. *Mock `EngineBackend` location* — moot. `CpuBackend` is the mock;
   nothing new was added.
2. *`serial_test` vs config-injection refactor* — chose neither.
   Thread-local override (per-test isolation without process
   mutation) is the third option and the right one.
3. *GPU-only dispatch branches* — non-issue at current coverage.
   Every dispatch file lands at ≥93% via the CPU happy path; the
   Metal-only `StateDumpMask::Full` blit branches are exercised
   indirectly by `CpuBackend`'s in-process implementation. No
   `cfg`-gating needed.

**Lesson for future env-gated production code:** add the
thread-local override at the same time as the `std::env::var` read,
not as a follow-on. Saves the future test-author from picking
between flaky parallel tests, `serial_test` ceremony, or a
config-injection refactor.

## Open work

> **Below all of these:** the decode attention K/V layout. Every cached engine
> shares one marginal cost per context token because they all land in that
> kernel, so a head-major layout would move all six at once — see
> [`larql-compute/ROADMAP.md`](../larql-compute/ROADMAP.md) "head-major K/V
> layout" and [`docs/decode-cost-model.md`](docs/decode-cost-model.md) §4.

### P0 — codebase-health frontier (audit 2026-06-14)

A whole-codebase review (engine faithfulness audit + clippy/coverage sweep)
surfaced four "finish-the-started-refactor" items. None is greenfield — the
ROADMAP already points at #7 and the `LayerExecutor` migration. Ordered by
risk/leverage; the first is a live correctness bug.

1. **Spin pool under heavy oversubscription — INVESTIGATED, pool is SOUND
   (2026-06-14).** On a heavily-loaded host (the spin-barrier pool spinning while
   the user's work pinned every core), the parallel test suite showed *rare*
   intermittent failures across diverse tests — clean with `LARQL_SPIN_POOL=0`
   (faster too) and single-threaded, which read as a contention correctness bug.
   **It is not.** The pool's synchronization was falsified-as-buggy two ways:
   (a) code analysis — the completion barrier's `completed.fetch_add(Release)` /
   `load(Acquire)`-on-the-final-count and the `epoch.fetch_add(Release)` /
   `load(Acquire)` task publication are a correct release/acquire pair, and the
   static strided ownership + the barrier make the dispatcher wait for every
   worker before advancing (so `data`/`tramp` can't go stale and cross-dispatch
   read-after-write is visible); (b) two new stress guards in `spin_pool.rs` —
   disjoint-write under EXTREME oversubscription (2× burner threads + N
   concurrent dispatchers + 4000 rounds) and **cross-dispatch read-after-write**
   under oversubscription — both stayed correct. Several of the "failures" were
   also misreads: `--nocapture` surfaces `#[should_panic]` and
   internally-`catch_unwind`'d expected panics (e.g. the empty-haystack
   `embed` test) that are NOT failures. **ROOT CAUSE FOUND — it was the env
   race, not the pool.** The decode path reads the q4k flags via `getenv`
   (`larql_compute::options::fast_path_on`) on every token; several TESTS toggled
   those flags with `std::env::set_var`, and concurrent `setenv`/`getenv`
   SIGSEGVs libc (and, short of a crash, returns an *inconsistent* flag mid-test
   → e.g. the in-place form reads int8-on while the owned-concat form reads
   int8-off → a bit-identity test "diverges"). Reproduced deterministically:
   `larql-compute`'s `q4k_direct_decode_step_matches_dequant_path` `set_var`s
   `LARQL_Q4K_ATTN_INT8` and flaked the sibling `q4k_direct_inplace_is_bit_identical`
   test. **Fixed:** all q4k `set_var` test sites in BOTH crates (5 in larql-kv,
   3 in larql-compute) moved to a **thread-local override**
   (`set_fast_path_override` / `FastPathGuard` / `Q4kFlagGuard`); no test mutates
   process env for these flags anymore. Both suites now pass clean 3× in parallel
   (706 compute + 765 kv) under load. The spin pool just amplified the window by
   slowing runs. **Remaining:** the generic `with_env*` helpers (moe/options
   tests) still `set_var` *other* vars — same class, folded into the env-sprawl
   item below. Two spin-pool stress guards (disjoint-write + cross-dispatch
   read-after-write under oversubscription) stay as regression pins.

2. **Env-var sprawl.** ~141 `LARQL_*` literals across 9 crates, **5 partial
   registries** with 3 different patterns, no single source. The
   `set_var`-in-tests pattern is also a **segfault class** — concurrent
   `setenv`/`getenv` SIGSEGVs libc.

   **Phase 1 — decode fast-path flags registry: DONE (2026-06-14).** Folded the
   six decode fast-path flags (`LARQL_Q4K_DIRECT_ATTN`/`_ATTN_INT8`/`_LM_HEAD`/
   `_DIRECT_FFN`/`_ASM`, `LARQL_SPIN_POOL`) — four former per-token `getenv`s +
   two ad-hoc per-stage `OnceLock`s — into ONE typed `larql_compute::options::
   DecodeOptions`, `from_env()` once and cached (`decode_options()`); the
   `*_enabled()` accessors read it (no per-token `getenv`). Tests toggle stages
   via a **thread-local override** (`set_fast_path_override` / `FastPathGuard` /
   larql-kv `Q4kFlagGuard`), which wins over the cache — so no test mutates
   process env for these flags. **All `set_var` sites of these flags migrated**
   workspace-wide (5 larql-kv + 3 larql-compute + 1 larql-inference) → the
   segfault/flake class is gone for the decode path; compute 706 + kv 765 +
   inference 1220 green, stable 3× in parallel, clippy clean.

   **Phase 2a — general override + larql-compute fully migrated: DONE
   (2026-06-14).** Generalised the thread-local override to ALL of
   `larql_compute::options`' env helpers (`env_flag`/`env_opt_out`/`env_opt_in`/
   `env_usize`/`env_value`/`env_nonempty_value`/`env_not_zero_or_default`) via a
   single `ENV_OVERRIDES` map + an `env_effective(name)` choke point; extracted
   the `"0"/"true"/…` vocabulary into pure `is_opt_{out,in}_value` parsers
   (directly unit-tested). Added `set_env_override(name, Option<&str>)` (value
   override; `set_fast_path_override` is now a bool wrapper). Migrated **every
   remaining `set_var` test helper in larql-compute** to it — `options`'
   `with_env_vars`, `moe/forward`'s `with_env`, `moe/expert`'s
   `with_env_in_thread` (sets the override *inside* the spawned thread so the
   TLS-cached `Q4K_DIRECT`/`EXPERT_TIMING` reads see it), `dump_config` (now reads
   via `env_value`/`env_usize`). **larql-compute src now has ZERO `env::set_var`**;
   707 tests stable 3× in parallel, clippy clean. The crate where the SIGSEGV was
   demonstrated is now race-free for env.

   **Phase 2b — our-flag migration extended: largely DONE (2026-06-15).**
   Migrated the our-flag `set_var` test sites in larql-inference (chat,
   layer_graph/{generate/lm_head,grid/config}, vindex/{walk_ffn,kquant_forward/
   hidden}, plus the already-done dequant) and larql-lql (executor + compile
   into_model/into_vindex) to the override (routing raw `std::env::var` reads
   through `options::*` where needed). compute 707 + kv 765 + inference 1220 +
   lql 726 + server 306 green, workspace builds + clippy clean.

   **Phase 2b — the remaining `set_var` is NOT override-addressable** (the key
   finding). ~59 of the ~74 remaining sites are **external/process-global env**:
   larql-vindex HF (`HF_HOME`/`HF_TOKEN`/`HF_HUB_CACHE`/`HOME`, read by the HF
   client) and larql-models loading. The thread-local override **cannot** reach
   them — an external reader uses real `getenv` — so they MUST use `set_var`;
   they're already **serialised via a per-module `ENV_LOCK` Mutex**, which is the
   correct (and only) mechanism for process-global env. Leave them (the residual
   `HOME`-vs-unrelated-`getenv` race is inherent to testing process-global env,
   not fixable by us). The small genuinely-remaining our-flag tail is all **cold
   diagnostic/config**, low-risk: `residual_diff/{stages,capture}` (dump-dir +
   env-save/restore-semantics tests — migrating changes what they test, do with
   care), cli `diagnostics/parity` (cross-backend: CPU dump vars are now
   override-aware via `DumpConfig`, the Metal dump var is read by larql-metal so
   it'd need metal-side routing), server `env_flags` (its own OnceLock-cached
   accessors — route through `options::*` or accept read-once), and metal
   `options` `DecodeFlags` tests (separate platform-gated binary). The one PRODUCTION
   smell — `larql-cli extract_index_cmd.rs` set `LARQL_SUMMARY_FEATURES_PER_EXPERT`
   as an env **side-channel** into the streaming gate path — is **FIXED**: threaded
   as a `summary_features_per_expert: usize` parameter from CLI →
   `build_vindex_streaming` → `StreamingContext` → `down_meta`/`gate_vectors`
   stages (the ~26 call-site API ripple the env hack was avoiding). The
   `SummaryEnvGuard` test scaffold and its `#[serial]` are gone; the summary-tier
   test passes K directly. No `LARQL_SUMMARY_FEATURES_PER_EXPERT` remains anywhere.

   **Phase 2c+ (open, lower-value).** markov cluster: own thread-local override
   (`read_markov_env`), per-layer uncached but cheap-when-unset — fold into a
   cached struct + unify with `ENV_OVERRIDES`. `LARQL_MOE_TIMING` read in 4
   places; collapse the ~7 timing flags → `LARQL_TIMING=…`, dump flags →
   `LARQL_DUMP*` (user-facing → aliases). `SKIP_MOE` vs `LARQL_SKIP_MOE` are
   **two different names** (compute `LARQL_SKIP_MOE`, inference `runtime.rs`
   unprefixed `SKIP_MOE`) — back-compat alias, not a rename. (NB: `LARQL_W10_HONLY`
   is **NOT** dead — live in the W10 mask cascade; an earlier audit mis-flagged
   it.) Optional purity: thread `DecodeOptions` through engine signatures to drop
   the global.

3. **Quantization meshing — finish deferred ROADMAP #7 (`FormatRoute`).**
   `QuantFormat` exists with helpers (`packed_matrix_bytes`, `packed_block_layout`,
   `is_kquant_family`) and a clean dispatch point (`backend.quant_matvec`), but
   hand-rolled fast paths bypass them and re-mesh magic numbers.

   **Step 1 (magic-numbers→helper) — DONE 2026-06-17.** Three production sites that
   re-derived the packed row stride as `(cols/256)*144`/`*210` now ask the format:
   - `attention/decode.rs` `q8k_direct_proj` → `packed_matrix_bytes(1, in_dim)`
     (path already requires `in_dim % 256 == 0`, so identical).
   - `cpu/ops/cpu/ops/q4k_q8k_dot/` `q4k_q8k_matvec_parallel` (the *centralized* matvec
     twin) and `kquant_forward/cached/` `matvec_q4k_or_q6k_q8k` — both were
     **string-keyed** (`format: &str`), so the magic-number and string-table
     problems converged there. Added `QuantFormat::from_registry_tag(&str)` (the
     contained version of #7's named helper) and routed both through
     `from_registry_tag` → `packed_block_layout`/`packed_matrix_bytes`. The
     centralized twin now parses the tag once and keys its kernel dispatch off the
     `QuantFormat` (not a second string match). No call-site signature changes.
     `q4k_q8k_matvec_parallel` keeps the truncating `cols/block_elems` (no `%256`
     guard there) via `packed_block_layout`; `cached.rs` uses `packed_matrix_bytes`
     (it guards `%256`). Numerically identical — full larql-compute suite green
     (33 q8k + 77 decode + 43 kquant + 2 new `from_registry_tag` tests), clippy clean.
   - `larql-inference/src/vindex/kquant_forward/cached/` — the "consolidation
     hazard twin" the compute dispatcher's own doc-comment names. Both its sites
     (`matvec_q4k_or_q6k_q8k` row stride + the `down_sb_bytes` per-super-block
     check) now route through `QuantFormat::from_registry_tag` so the two crates'
     copies stay in sync. 60 kquant + 46 cached inference tests green.

   *Lower-priority Step-1 tail (deferred, low value):* `q4_common.rs` is the packer
   where 144/210 are legitimately *defined* (consumer-side strides at 344/350 could
   still ask the format); the `*18` Q4_0 legacy-block sites (`q4_matvec.rs`,
   `crates/larql-compute/src/cpu/ops/q4_common/`, `gpu.rs`) are the block-32 equivalent.

   **Remaining offenders (Step 2 territory):** `cpu/ops/moe/expert.rs` is silently
   **Q4_K-only** (`matches!(format, Q4_K)` + hardcoded `Q4_K_BLOCK_BYTES` at 274/453
   — these are *named* constants, lesser offense; the real issue is the Q4_K-only
   dispatch); `pipeline_layer.rs`'s twin `attn_str_to_format`/`ffn_str_to_format`
   panicking string tables (now subsumable by `from_registry_tag`). **Step 2:** a
   `QuantFormat::q8k_matvec_into_fn()` kernel table so a new format is ~3 edits, not
   ~49 files — this generalizes the Q4_K-only dispatchers to any k-quant kernel.

4. **Engine pluggability — finish the `LayerExecutor` migration.** A new engine
   needs 4 required methods but **~8 boilerplate overrides** (the
   `*_quant`/`*_resident`/`*_via_executor` cross-product, all of which every
   shipped engine overrides) + **6 hand-synced registration sites** in
   `lib.rs` (`EngineKind` variant / `from_name` / `display_name` /
   `supported_names` / `build_with_profiling` / CLI) — one of them a **duplicate
   `KvCacheKind` parser** in `larql-cli/run_cmd.rs`. Shared scaffolding exists
   (`engines::layer_ffn_or_moe`, `run_attention_block_decode_step_auto`,
   `LocalWalkExecutor`) but each engine still hand-wires its per-layer loop.
   **Proposal:** one `decode_step_walk` + a `KvEngineState` policy trait (append/
   read K/V + state-policy hooks) collapses the 8-method cross-product to thin
   adapters; a `register_engine!` macro (or `inventory`) removes the 6 sites and
   makes `engine_kind_supported_names_covers_every_variant` unnecessary; delete
   the duplicate `KvCacheKind` (route `--kv-cache` through `EngineKind::from_name`,
   which already accepts `standard`/`none`/`markov-bounded`). `AnyEngine`'s
   hand-written sum-type forwarders should be macro/`enum_dispatch`-generated too.

   **Quick wins** (low-risk, do-now candidates): quant Step 1
   (magic-numbers→helper), retire `LARQL_W10_HONLY` + fold `SKIP_MOE`, delete the
   `KvCacheKind` duplicate. The larger refactors (DecodeOptions threading, the
   engine-walk collapse, the kernel-fn table) are scoped follow-ups.

   **INVESTIGATED 2026-06-19 — this item is over-scoped; most sub-tasks are NOT
   clean wins.** Three angles checked against the actual code:
   - **`KvCacheKind` delete — entangled, low value.** `EngineKind::from_name`
     collapses `standard` and `markov-bounded` to the *same* `Standard{window}`,
     so it can't recover the distinction the legacy flag needs to apply
     `--context-window` (which is sliding-only). Routing `--kv-cache` straight
     through `from_name` silently breaks that semantic. And `KvCacheKind` is a
     *frozen* 3-variant legacy flag — new engines go through `--engine`, so it
     never grows per-engine. Not the registration burden it was billed as.
   - **`SKIP_MOE` fold — touches a contract.** The unprefixed `SKIP_MOE` (and
     `MOE_TOP_K`/`MOE_NO_SPLIT`) are *intentional* grid-tooling names on
     `GridRuntimeConfig`, arguably a different flag from compute's
     `LARQL_SKIP_MOE` (grid-bypass vs local-decode-skip). Not a trivial alias.
   - **`AnyEngine` → enum_dispatch — doesn't fit, and it's already fine.**
     `AnyEngine` is a 2-variant *family* enum over *two different traits*
     (`KvEngine` / `RetrievalEngine`), with intentionally **asymmetric**
     forwarders (`prefill` passes `ffn` for `Kv` only; `supports_multimodal` is
     `Kv`-only; `prefill_from_hidden` panics on `Retrieval`). `enum_dispatch`
     needs one shared trait + uniform signatures, so it can't apply, and a macro
     would only obscure the explicit 2-arm matches. It grows per *family* (rare),
     not per *engine* — so it's not the boilerplate driver.

   **Reality:** the only genuine per-engine cost is the `lib.rs` registration
   (`EngineKind` variant + `from_name` + `display_name` + `supported_names` +
   `build_with_profiling`), and even that resists a clean `register_engine!`
   macro because each engine parses *different* params (`window`/`bits`/`layer`/
   `coef`/`top_k`) and constructs a different type. A data-driven table could
   drive `display_name`/`supported_names` (and kill
   `engine_kind_supported_names_covers_every_variant`), but `from_name` +
   `build_with_profiling` stay hand-written. **Recommendation: DE-PRIORITIZE.**
   The `LayerExecutor`/`decode_step_walk` collapse (the 8-method override
   cross-product) is the one part with real leverage left, but it's a scoped
   refactor, not a quick win. Prefer quant Step 2 (timed with BitNet) or the P0
   perf frontier (W1-GPU Metal matvecs) for higher return.

### P1 — MoE-aware KV engines (C1) — new 2026-05-28

The KvEngine layer is **dense-only today**: `do_prefill` / `do_decode_step`
dispatch dense FFN via `ffn.forward(layer, x)` and are KV-cached, but no engine
branches on MoE layers (grep for `forward_moe_full_layer` / `run_moe_layer_cpu`
in `larql-kv` is empty). MoE decode — both `--ffn` whole-layer offload and
`--moe-shards` client-side expert sharding — runs through the standalone
full-recompute `predict_kquant_hidden*` path with **no KV cache**. CPU
`--moe-shards` was measured at **0.1–0.4 tok/s** on Gemma-4-26B-A4B (the
full-recompute fix that closed #146, 2026-05-28).

Goal: make the engine layer MoE-aware so CPU MoE decode is KV-cached and
**engine-selectable** (standard / windowed_checkpoint / markov* / turbo_quant /
apollo all apply their mechanism to MoE models, not just dense).

Subtasks:
1. **Engine per-layer MoE branch.** The shared per-layer forward must, on MoE
   layers, compute `h1` (dense FFN) + `h2` (expert contribution via
   `forward_moe_full_layer`) then apply the hybrid-MoE combine + outer-norm.
   Today only `run_moe_layer_cpu` (larql-inference `vindex/kquant_forward/hidden.rs`)
   does this — lift it so the engine forward can call it.
2. **`RemoteMoeFfn` `FfnBackend` wrapper** (larql-inference). `RemoteMoeBackend`
   is the one remote backend that is *not* an `FfnBackend`.
   `FfnBackend::forward_moe_full_layer(layer, h_post_attn)` gets no weights, but
   the moe-shards combine needs local dense FFN + router + norms — so wrap
   `{ weights, remote }` and implement `forward_moe_full_layer` as the
   `run_moe_layer_cpu` body (dense local + experts remote via `forward_moe_seq`
   + combine). This makes `--moe-shards` ride any engine, unifying it with `--ffn`.
3. **CLI routing.** Route CPU `--moe-shards` (and `--ffn` on MoE models) through
   the selected `--engine` instead of the standalone full-recompute path.
4. **Parity + perf.** Tolerance parity vs the full-recompute path and vs local
   CPU MoE; perf gate (KV-cached should be ≫0.4 tok/s on 26B).

Exit criterion: `larql run --moe-shards … --engine standard` (no `--metal`)
decodes KV-cached at parity with the full-recompute path, and the same works
across the other engines. Decision recorded 2026-05-28: keep the full-recompute
fix as the #146 correctness baseline; this item replaces it for performance.

**Status (2026-05-28) — DONE, default path, parity verified.**
Subtasks 1–3 + CLI wiring shipped: `moe_ffn_block_cpu` factored out of
`run_moe_layer_cpu` (parity-preserving), `kv_dispatch` helpers MoE-aware
(`ffn_or_moe_layer`), `RemoteMoeFfn` in larql-inference, and the CLI routes CPU
`--moe-shards` through a `StandardEngine` via `generate_with_engine`. KV-cached
is now the **default**; `LARQL_MOE_FULL_RECOMPUTE=1` and PLE archs fall back.

Two bugs found + fixed during verification:
1. **Wrong driver** — the CLI first used `generate_cached`, which runs the
   *legacy* `kv_prefill_run` path (no `forward_moe_full_layer` hook → experts
   never dispatched). Switched to `generate_with_engine`, which routes through
   the MoE-aware `kv_*_via_dispatch` path.
2. **Prefill RoPE** — `run_attention_with_kv_backend` (engine prefill) used
   `apply_rope_partial` (position_divisor=1.0, llama3=None, raw base), silently
   dropping Gemma 4's scaled global-layer RoPE. The decode-step path
   (`decode.rs`) and full-recompute (`block.rs` core) already used the
   forward-override-effective base + divisor + llama3 via
   `apply_rope_partial_at_full`; prefill was the lone holdout. Fixed to match.
   (NB: `run_attention_block_gpu` has the same unscaled-RoPE call but is
   test-only — no live callers — left as-is.)

Verified live on Gemma-4-26B-A4B (two expert shards, no `--metal`): output is
**byte-identical** to full-recompute (24-token continuation matched exactly) at
**~10× the speed** (4.2–4.4 tok/s vs 0.4–0.5). All suites green.

**Regression guard added** (2026-05-28): `run_attention_with_kv_backend_matches_full_recompute_on_gemma3`
(larql-compute `attention/gpu.rs`) asserts engine prefill == full-recompute
attention on a 6-layer rope-scaled Gemma 3 fixture
(`make_gemma3_rope_scaled_test_weights`, layer 5 global / divisor 8). Validated:
it FAILS at L5 if the prefill-RoPE fix is reverted.

### Which engines support remote MoE? (audit 2026-05-28)

| Engine | FFN routing (driver = immutable `prefill`) | Remote MoE | Verified (26B) |
|---|---|:--:|---|
| **standard** | per-layer via `ffn` trait (`kv_*_via_dispatch`) | ✅ | "Paris", **4.4 tok/s** |
| **markov_residual_codec** | per-layer `compute.rs` `run_ffn` → `layer_ffn_or_moe` | ✅ | "Paris", **3.4 tok/s** |
| **turbo_quant** | per-layer `engine.rs` `run_ffn` → `layer_ffn_or_moe` | ✅ | "Paris", **3.4 tok/s** |
| **markov_residual** | per-layer `compute.rs` `run_ffn` → `layer_ffn_or_moe` | ✅ | "Paris", **3.1 tok/s** |
| **boundary_per_layer** | per-layer `walk::run_prefill`/`run_decode` (larql-kv) → `layer_ffn_or_moe` | ✅ | "Paris", **3.1 tok/s** |
| **boundary_kv** | wraps `StandardEngine` + compressed-residual boundary frames | ✅ | "Paris", **2.9 tok/s** |
| **windowed_checkpoint** | per-layer `rs_extend_from_checkpoint_backend` → `layer_ffn_or_moe` | ✅ | "Paris", **1.7 tok/s** |
| no_cache | legacy `kv_prefill_run` full re-forward | ✗ (by design) | full re-forward per step; not sensible for remote experts |
| apollo | local re-forward (`forward_from_layer`) | ✗ (by design) | crystal re-forward *multiplies* per-step expert round-trips |

**How it works (2026-05-28).** `generate_with_engine` drives the engine's
*immutable* `KvEngine::prefill`/`decode_step`. For `standard`/`boundary_kv` that's
the `kv_*_via_dispatch` path; for the per-layer/windowed engines it's their own
larql-kv forward loop (`rs_extend_from_checkpoint_backend`, `compute.rs`,
`turbo_quant/engine.rs`, …), which *can* call larql-inference. The shared helper
**`engines::layer_ffn_or_moe`** does the per-layer choice: on hybrid-MoE with a
`moe_ffn` hook, call `forward_moe_full_layer` (experts → shards); else dense
`run_ffn`. Threading `ffn` from `prefill`/`decode_step` → the forward loop lights up
an engine with a ~10-line change. **All in larql-kv — no `EngineBackend` trait
change, no Metal-path risk.** **7 of 9 engines now verified for remote MoE** — the
only exclusions (`no_cache`, `apollo`) are excluded *by design*, not by limitation.
(Note: `boundary_per_layer`'s immutable driver path uses `walk::run_prefill`, a
larql-kv loop — *not* the fused coarse path — so the deeper coarse-path hook I'd
flagged turned out unnecessary; only the disused `prefill_quant`/coarse path would
need it.)

**Perf reality — they all *work*; see the bottleneck diagnosis**
([`docs/diagnoses/remote-moe-bottlenecks.md`](../../docs/diagnoses/remote-moe-bottlenecks.md),
2026-05-29). ⚠️ The per-engine tok/s below were the CLI's old `total/n` banner
(model-load + prefill + decode averaged over n) — **load-dominated for short runs,
not true decode**. True steady-state decode for `standard` is **~6 tok/s** (the
banner now reports TTFT vs decode separately). The path is **compute-bound, not
network-bound** (localhost RTT 0.35 ms × 30 layers ≈ 12 ms vs ~165 ms/token);
**model load ~6.8 s** dominates one-shot latency. The figures still rank the
engines correctly but understate absolute decode and compress the spread:
standard **4.4** > markov_codec/turbo **3.4** > markov /
boundary_per_layer **3.1** > boundary_kv **2.9** > unlimited **1.7** tok/s. The
spread is each engine's per-step CPU mechanism *on top of* the shared per-layer
expert network round-trip; the round-trip compresses the spread (4.4→1.7, ~2.6×, vs
the dense-4B 28→19 CPU spread). `standard` stays fastest; `unlimited` is the slowest
(O(window²) prior-KV clone + per-token re-attention). So "they should all run fast"
lands as **true — all seven within ~2.6× and network-bound** — `standard` the pick.

**Best engine for remote MoE:** `standard` for throughput; `boundary_kv` for
wire-efficient cold-context residual frames; `markov`/`turbo`/`boundary_per_layer`
for compressed KV memory at near-standard speed; `windowed_checkpoint` for
long-context windowed KV (slowest, bounded memory). `no_cache` / `apollo` are not a
fit (re-forward multiplies round-trips).

**Resolved (2026-06-13):** `windowed_checkpoint::replay_window` now takes
`moe_ffn` + `index` and threads them to `rs_extend_from_checkpoint_backend`
(matching the live-window `extend_current` path), so an evicted MoE window
replays with experts instead of silently falling back to dense FFN. It is a
standalone utility (no decode-loop caller — the decode path attends to the
current window + boundary checkpoints, never a full replay), so this was a
*latent* correctness gap; it is now correct for any caller. Dense callers pass
`None`/`None`. CLI guard allows the seven verified engines and rejects
`no-cache` / `apollo` with a clear message.

### ✅ DONE / EXCEEDED — Q4K-direct decode path (remove the f32 tax)

**Status (2026-06-13):** done and the target was blown past. This section's exit
was "~20–25 tok/s, within ~10% of the ~22 tok/s bandwidth ceiling." Reality:
the residency stack (Q4K-direct attn/lm_head/ffn + int8 + asm) + KV
append-in-place + the **spin-barrier pool** took the 26B in-process decode to
**~35 tok/s — past llama.cpp (32.1)** — and the whole stack now ships
**default-on** (see ROADMAP.md baseline table + "Spin-barrier pool" above). The
last lever was *not* the f32→Q4K tax (that was the residency work); it was
**rayon fork-join overhead** (driver outside the pool), closed by the spin pool.
Original framing kept below for history.

**Why now (historical):** the bottleneck diagnosis
([`docs/diagnoses/remote-moe-bottlenecks.md`](../../docs/diagnoses/remote-moe-bottlenecks.md),
2026-05-29) measured the remote-MoE decode split on the 26B: **~60% is client-side
f32 compute** (attention + lm_head + dense FFN, on the dequant-to-f32 BLAS path),
~40% is server expert compute, network negligible. The engine path currently
**dequantizes all attn + dense-FFN weights to f32 up front** (the ~6.8 s model-load
tax) and runs attention/FFN/lm_head on f32 BLAS — *not* the NEON **Q4K-direct
matvec** kernels that already exist (the ones that took Gemma-3-4B CPU
0.36 → 28 tok/s).

**Measured client split** (`LARQL_DECODE_STAGES=1`, 26B, prefill+12 decode):
attention **28%** · dense FFN **13%** · lm_head **12%** (≈53% recoverable client
f32) · remote experts 41% (server) · misc 5%. **Attention is the #1 target.**

**Work (ranked by win):**
1. **Attention (28%)** — Q4K-direct path reading attn bytes from the index via
   `q4_attention_proj` (`attention/gpu.rs`, CPU-tested), replacing the f32
   `run_attention_with_kv_backend`. Parity-critical rework of the attention path;
   verify byte-parity on the 26B before flipping.
2. **dense FFN (13%)** — ⚠️ `WalkFfn` tried + reverted (its dense mode runs the
   sparse-walk machinery → 8.5× slower than f32 BLAS). The right kernel is
   `kquant_ffn_forward_layer_q8k` (NEON, no dequant) via a thin `FfnBackend`
   wrapper. Low ROI (f32 BLAS already competitive, only ~13%) — below attention.
3. **lm_head (12%)** — Q4K vocab projection from the loaded lm_head Q4K bytes.

Doing all three also lets the CLI **drop the up-front "dequantize all layers to
f32"** step (`run_with_moe_shards`) — removing most of the ~6.8 s model-load tax
(nothing left to dequantize). Per-stage timers already in place (`decode_stages`).

**Expected:** ~4× decode (measured 7.9 → **~20-25 tok/s** on the 26B, i.e. up to the
DDR5 bandwidth ceiling) **and** much faster startup (no dequant-all). Pure
engineering — depends on no unproven research. **Highest-leverage move fully in our
control.**

**Exit:** remote-MoE `--engine standard` decode within ~10% of the single-box A4B-Q4
bandwidth ceiling (~22 tok/s on the 26B), byte-identical to the f32 path; CLI no
longer dequantizes all layers up front.

**After this** (to go past the ~22 tok/s wall, both out of pure-engineering scope):
distribute expert bandwidth across more grid shards; the compounding stack
(hash-routing 5× **FALSIFIED V1 2026-05-31** — doesn't compound; FP4 2× **confirmed V2**); and
multi-layer expert **prefetch** to hide the 30 sequential layer round-trips on real
LAN/WAN (free on localhost, fatal at 10 ms RTT). 80 tok/s on the 26B needs all
three; for 4B-class it's already near.

### P0 — engine performance (the post-bypass optimization frontier)

The fused-bypass strip (2026-05-17 night) made every engine's actual
per-step cost visible for the first time. The remaining headroom is
substantial — but the goal is to close it **without** re-introducing
bypass paths. Each per-layer engine has a state-policy contract that
defines what work cannot be skipped; the optimization budget is what
remains.

**Reference numbers** (Gemma 3 4B Q4K, Metal, M3 Max, 20-token
decode):

| Engine | tok/s | Hot state | Per-step cmd_bufs (Metal) | Per-step compute model |
|---|---:|---:|---:|---|
| `standard` (fused) | 104 | 0 MB (backend-owned) | 1 | one fused kernel, all 34 layers, append-1-row K/V |
| `windowed_checkpoint` | 25.6 | 4.8 MB | ~103 | per-layer attn+ffn, append-1-row K/V (same compute as standard, different dispatch) |
| `markov_residual_codec` | 4.3 | 6.0 MB | ~103 | per-layer attn+ffn + **recompute K/V from `window_size` residuals every step** |
| `turbo_quant` (4-bit) | 3.9 | 0.6 MB | ~103 | per-layer attn+ffn + **decompress prior K/V + re-encode updated K/V every step** (CPU codec in inner loop) |
| `markov_residual` | 3.6 | 6.0 MB | ~103 | same as codec; no codec overhead on bench (cold tier never fired in 20-step run) |
| `apollo` | — | scales w/ store | varies | re-forward layers `crystal..N` over growing context every step (no K/V cache) |
| `no_cache` | — | token list only | varies | full re-forward every step (O(N²) by design — not an optimization target) |

#### Per-engine bottleneck diagnosis

**Post-W2 measurements — split by backend** (Gemma 3 4B Q4K, M3 Max,
10-token decode, 2026-05-17 night):

| Engine | CPU tok/s | GPU (Metal) tok/s | Where the gap lives |
|---|---:|---:|---|
| `standard` (coarse_prefill control) | 28.2 | 102.7 | GPU's fused fast path is 3.6× the CPU C kernel. |
| `windowed_checkpoint` | 28.1 | 28.4 | **At parity** — no per-layer overhead either side. |
| `markov_residual_codec` | 26.6 | 27.5 | **At parity** (post-W2). |
| `markov_residual` | 26.5 | 26.8 | **At parity** (post-W2). |
| `turbo_quant` (4-bit) | 19.4 | 19.6 | **At parity** — codec overhead dominates on both. |

**Reading the table — the GPU/CPU split reveals an even sharper
diagnosis** (re-checked 2026-05-17 after reading the helper code):

- **On CPU**, every engine clusters at ~26-28 tok/s. The 28 tok/s
  ceiling is the M3 Max CPU compute limit for Gemma 3 4B Q4K
  rayon-parallel matvec at this prompt length.
- **On GPU**, only `standard` reaches 102.7 tok/s — the only engine
  that actually runs on the GPU. The four "per-layer Metal" engines
  all sit at 20-28 tok/s, same as CPU, **because they are running
  CPU code regardless of the `--backends metal` flag.** Tracing
  through `attention_decode_step_native` and `ffn_decode_step_native`
  (the native-quantised helpers all per-layer engines call): the
  `_backend: &dyn ComputeBackend` parameter is plumbed but never
  consulted — these helpers always dispatch to
  `matvec_q4k_or_q6k_q8k`, which is rayon-parallel CPU Q4K×Q8K
  matvec. The Metal backend isn't involved in their per-layer
  compute at all.

This changes the W1 framing. The previous diagnosis ("103 Metal
submits per token = 5-10ms of dispatch overhead") was wrong because
**there are zero Metal submits per token** for per-layer engines
today — the entire per-layer loop runs on CPU. The actual ~28 tok/s
ceiling is the CPU's rayon-parallel matvec throughput, hit equally
under both `--backends cpu` and `--backends metal`.

**The real W1**: route the per-layer Q/K/V/O and gate/up/down matvecs
through Metal kernels (per layer) so the GPU actually participates
in the per-layer engines' compute. This is a larger change than
"batch the dispatches" because today's per-layer code path doesn't
use Metal at all — there's nothing to batch yet.

W2 landed: caching the hot K/V projection across decode steps
moved both markov_residual engines from ~5 to ~27 tok/s — they now
sit on the same curve as `windowed_checkpoint` (which already cached
K/V incrementally), within 1.5 tok/s of each other. The
`recompute_kv` stage no longer fires; FFN+attention dominate
exactly like every other cached-K/V engine. **The hot K/V state
costs ~10.8MB vs 5.3MB pre-W2** (trade memory for speed; still
~50× smaller than standard's full KV).

Reading the table: percentages are *of the engine's own per-step total*,
not vs standard. The three cached-K/V engines (markov-rs, codec,
windowed-checkpoint) now cluster around 27-28 tok/s, all showing the
same FFN-heavy decode profile. The remaining ~4× gap to standard
is per-layer Metal dispatch overhead — W1's target.

**`windowed_checkpoint` — 28.4 tok/s, 35 ms/tok. Per-layer attn + ffn
dominates; no recompute waste.** Compute model is identical to
standard's (append-1-row K/V per layer). 74% of the step is FFN, 25%
is attention. The 4× gap to standard is **per-layer Metal command-
buffer dispatch** — 103 cmd_bufs per token vs standard's 1. Each
submit has ~50-100µs fixed cost, so even with zero-cost compute
there'd be 5-10ms of pure scheduling per token. This is the cleanest
optimization target — the engine's contract doesn't require per-layer
submits, only per-token boundary checkpointing. **Workstream W1
(batched per-layer command buffer) should close most of the gap →
projected ~80-100 tok/s.**

**`markov_residual` / `markov_residual_codec` — 26.8 / 27.5 tok/s,
~37 ms/tok. W2 LANDED.** The hot K/V cache eliminates the 80% recompute
overhead measured pre-W2; both engines now sit on the same curve as
`windowed_checkpoint` while preserving the residual-stream contract
(drop `hot_kv` and the next step recomputes from `stored` — the
fallback path is still there for the via_executor path that doesn't
yet capture K/V). The W2 design preserves the engine identity: K/V is
still derivable from residuals; we just don't re-derive every step.

The codec engine being marginally **faster** than the base engine
(27.5 vs 26.8) on a 10-step bench is variance — both run identical
hot-path code, and the codec's bf16 encode/decode only fires at
window-boundary evictions (rare relative to step count). At long
contexts the codec's value re-emerges as memory savings on the
cold tier.

**`turbo_quant` (4-bit) — 20.3 tok/s, 48 ms/tok. FFN dominates; codec
is ~25% of the budget, not the bottleneck.** This is a real surprise:
the pre-profile guess was "codec encode/decode is the inner-loop
killer." Measured: codec is ~25% (9.4% decode + 15.5% encode), FFN is
53%, attention is 20%. Turbo_quant is much closer to windowed_checkpoint
(28.4 tok/s) than to markov_residual (~5 tok/s) — the engine works.
The codec is a fixed overhead per layer per step, not a quadratic
blow-up. **Workstream W3 (incremental encode of the new row only)
still applies — it would cut the 15.5% encode share roughly in half —
but the bigger lever is W1 (dispatch batching), since FFN dominates
the per-step budget and is the same per-layer-Metal bottleneck as on
windowed_checkpoint.** W4 (SIMD WHT) is now lower-priority than originally
estimated; codec is fast enough that vectorising it shaves single-digit
percent.

**`apollo` — requires store, not benched.** Compute model is
fundamentally different: re-forward layers `crystal..num_layers` over
the growing context every decode step. Per-step cost grows linearly
with generated length. At step N: 4 layers × forward over
(N+window_tokens). This is *closer* to no_cache than to standard —
apollo never caches K/V across steps. The bottleneck isn't dispatch or
codec; it's the recomputation model. See workstream W5.

**`no_cache` — by design O(N²).** Not an optimization target;
correctness-baseline only.

#### Optimization workstreams (contract-preserving)

| ID | Workstream | Engines | Expected gain | Risk |
|---|---|---|---|---|
| W1-GPU | **Route per-layer Q/K/V/O and FFN matvecs through Metal.** Today's `attention_decode_step_native` and `ffn_decode_step_native` ignore the backend param and run rayon CPU matvec — that's why all four per-layer engines hit ~27 tok/s on both `--backends cpu` AND `--backends metal`. The GPU is not involved at all. Workstream: make these helpers actually dispatch to `MetalBackend`'s per-layer quant matvec kernels (the ones `fused_prefill` already uses internally). **GPU only.** | windowed_checkpoint, markov_residual, markov_residual_codec, turbo_quant | Unknown — first deliverable is the measurement. Ceiling ranges from ~40 tok/s (submit overhead dominates) to ~80 tok/s (matches standard's GPU advantage). | Per-layer Metal submit cost (50-100µs each × ~6 per layer × 34 layers = ~10-20ms/token) is the open question. May need to batch within a layer (Q+K+V in one buffer, attn separately, etc.) to amortize. CPU is at parity already; no W1-CPU. |
| W2 | **Persistent hot K/V cache in markov_residual.** The engine contract says "K/V derived from residuals" — it does **not** say "recomputed every step." Cache hot K/V across steps; append-1-row on new residual; only recompute fully on cold-tier eviction (rare). Cold-tier compression remains the engine's selling point. | markov_residual, markov_residual_codec | ~20-30×; engine becomes "windowed_checkpoint with compressed-residual cold tier" | Need to verify residual store still reflects "what we'd recompute from" — i.e., consistency check that cached K/V matches a fresh recompute under same residuals. Add a debug assertion mode. |
| W3 | **Incremental TurboQuant encode (append-only).** Only encode the new K/V row each step; keep prior compressed bytes untouched. Decompress only the new row's neighbourhood for attention scores (or the whole layer if simpler). | turbo_quant | ~10× at long context | Re-encoding for in-place updates is the slow path. Need to define when (if ever) the full layer needs re-encoding. |
| W4 | **TurboQuant SIMD WHT + Lloyd-Max.** Already on P1; promote to P0 once W3 lands so the per-row codec cost is the only remaining work. NEON on Apple Silicon, AVX2 on x86_64. | turbo_quant | 2-4× on the codec step | Mostly mechanical; landing W3 first means each step touches less data, making SIMD's batch budget go further. |
| W5 | **Apollo K/V cache across decode steps.** Cache the K/V for layers `crystal..num_layers` between steps; append-1-row per step instead of re-forwarding. Reduces per-step cost from O(N) to O(1) in generated length. | apollo | linear → constant per-step | Apollo's vec_inject perturbation fires at `injection_layer`; verify the perturbation interacts correctly with cached K/V (it should — perturbation is residual-additive, not K/V-overwriting). Needs an apollo store fixture in tree to bench. |
| W6 | **Cache attn dequant for the engine's lifetime, not per-call.** `ensure_attn_tensors_dequantised` already has an idempotency check; verify it's actually one-shot under bench. If it isn't, fix the cache. | all per-layer engines | 5-15% | Mechanical; just instrument and verify. |
| W7 | **Q4K-path engine profiler.** Today `--profile` surfaces a per-stage breakdown for markov_residual's dense path only. The Q4K decode (`rs_decode_step_walk`) doesn't populate `EngineProfiler`. Wire it, then wire the other engines so `larql bench --profile --engine markov-rs:window=512` produces an attribution. Without this, every workstream above is unfalsifiable. | all per-layer engines | 0 (instrument) | Needs to thread `&mut EngineProfiler` through `rs_decode_step_walk`, `process_q4k`, `decode_step_q4k_cpu`. |

#### Sequencing

Recommended order (revised 2026-05-17 night after W7 produced
measured numbers — replaces the earlier guess-driven sequence):

1. **W7 — DONE.** Profiler wired across markov_residual,
   markov_residual_codec, windowed_checkpoint, turbo_quant. Each
   engine's `--profile` output produces a per-stage attribution.
   See the measured table above.
2. **W2 — DONE.** Hot K/V cache landed on `markov_residual` and
   `markov_residual_codec`. Both moved from ~5 tok/s to ~27 tok/s
   (5.5-5.7×) and now sit on the same curve as `windowed_checkpoint`.
   Engine contract preserved: K/V still derivable from residuals,
   just not re-derived every step. Hot K/V state grew from 5.3MB
   to 10.8MB; that's the speed/memory trade. Bit-parity tests
   confirm the cached path matches the recompute path within fp
   rounding.
3. **W1-GPU — route per-layer matvecs through Metal kernels.**
   Per the corrected diagnosis above, the per-layer engines are
   *not* using Metal today — `attention_decode_step_native` and
   `ffn_decode_step_native` ignore their `_backend` parameter and
   call rayon-parallel CPU matvec. The workstream is to plumb
   per-layer Q/K/V/O and gate/up/down matvecs through Metal kernels
   (the same kernels `standard` uses internally during
   `fused_prefill`'s per-layer encode loop) so the GPU actually
   participates in per-layer engines' compute. Each layer becomes
   ~6 Metal submits (Q, K, V, attn, O, gate+up, act+down) per
   token — there's a real question whether the submit cost is
   worth it on Apple Silicon vs the CPU's 27 tok/s ceiling. **W1's
   first deliverable is the measurement, not a single decision:**
   write the per-layer Metal path, bench, and ratchet from there.
   The ceiling could be anywhere from "1.5× the CPU ceiling" (if
   submit overhead dominates) to "3× the CPU ceiling" (matching
   standard's GPU advantage, modulo per-layer dispatch). The CPU
   ceiling is already the M3 Max compute limit — no separate
   "W1-CPU" work to do; CPU is the floor.
4. **W3 — incremental TurboQuant encode.** Lower priority than
   originally thought (codec is ~25% of turbo_quant's budget, not
   80%). Still worth doing — would halve the 15.5% encode share.
5. **W4 — SIMD WHT.** Demoted; codec is fast enough that vectorising
   it shaves single-digit percent. Only worth landing if W3 already
   has and codec is the largest remaining slice.
6. **W5 — Apollo K/V caching.** Largest behavioural change; sequence
   last. Needs an apollo store fixture in tree before bench can
   surface the bottleneck.

#### What this is NOT

- **Not re-introducing fused bypass.** Standard remains the only
  fused engine. Per-layer engines stay per-layer; the goal is to
  make per-layer fast, not to skip it.
- **Not removing engine contracts.** Markov-rs's residual store
  must still be re-deriveable; turbo_quant's K/V must still be
  compressed; windowed_checkpoint's checkpoints must still emit at
  window boundaries. Optimizations are within the contract.
- **Not optimising no_cache.** It's a correctness baseline; O(N²)
  is the design.

#### Guardrails: don't let the bypass come back

The fused-bypass pattern hid for months because nothing asserted
"the engine actually ran." Two invariants we should land before
the optimization work starts, so a future shortcut can't regress
silently:

- **State-policy assertion.** Every engine declares at least one
  invariant that holds iff its state-policy code executed. For
  example:
  - `markov_residual`: `engine.memory_bytes() > 0` after prefill on
    a non-empty prompt.
  - `markov_residual_codec`: same; plus `cold_bytes() > 0` after
    overflow.
  - `windowed_checkpoint`: `archive.len() > 0` after at least
    `window_size` tokens.
  - `turbo_quant`: `layers.len() == num_layers` after prefill.
  - `apollo`: `context_tokens.len() > 0` after prefill.

  Add a `KvEngine::executed_state_policy() -> bool` method (or a
  test-only trait) and assert it in `larql bench` after prefill
  when `--engine` is set. The bench should print a warning if any
  engine reports `false`. This is what would have caught the
  bypass on day one.

- **Per-stage profiler coverage on the Q4K path** (W7 above). Without
  attribution we have no signal when a bypass re-emerges; the engine
  would just look mysteriously fast.

### P0 — engine performance — open after W8.2 (2026-05-18)

The W8/W8.2 alloc-churn fix collapsed the largest decode hot path
cost. The remaining levers are smaller and more scattered. Listed
in expected ROI order.

- **W9 — Single-kernel prefill state-dump.** Step 9 (2026-05-18) made
  prefill iterative (one `fused_decode_step_with_state` per prefill
  token, ~50 ms × N tokens). For N=5 this lands at ~250 ms vs
  `standard`'s ~300 ms fused — already faster on this prompt shape.
  W9 would consolidate into a single Metal kernel call that dumps
  per-position per-layer state for all prefill positions at once,
  saving the ~10 ms × N per-iter setup. Expected wall-time saving:
  ~50 ms / prefill. Small at 5-token prompts; larger at 100+ token
  prompts. **Scope: Metal-kernel surgery in
  `larql-compute-metal/src/decode/mod.rs` — likely a new
  `fused_prefill_with_state` symmetric to `fused_prefill` but with
  the W7 blit-encoder fusion baked in across positions.**
- **W10 — Engine-side state stays on GPU.** Today
  `decode_step_via_dispatch` reads per-layer K/V back into CPU
  `Array2<f32>` to update the engine's `hot_kv` store, then
  `coarse_decode_step_with_state` re-uploads the cache via its own
  K/V buffer on the next step. With engine-side state on GPU
  (`Vec<KvBufferHandle>`), the readback + re-upload pair collapses
  to zero CPU work per step on the dispatch path. The CPU-side
  `Vec<Array2<f32>>` would materialise lazily on `close_window` /
  `info()` calls. Expected: closes most of the remaining 8-11% gap
  to `standard`. **Scope: extends the `KvDispatch::PerLayerDecodeState`
  shape to carry opaque handles instead of `Vec<f32>`; needs a
  matching CPU-side shadow type for `CpuBackend` which has no
  on-GPU state.** Pre-req: stable `MetalBackend`-side KV cache
  invariants (which Step 9 already established).
- **W8.2 → `windowed_checkpoint` CPU walk fallback.** The legacy CPU
  walk path (`process_via_executor` at engine.rs:~720) still uses
  the per-step `Array2::zeros((s_old+1, dim))` pattern. Not on the
  hot path for the bench (dispatch path is the default), but a
  consistency cleanup. Scope: ~10 lines, mirrors W8 mechanically.
- **W11 — Lift W8.2 pattern to `apollo`'s constellation cache.** Not
  measured today (apollo is bench-skipped because it needs a store);
  if/when the on-disk store loader (P1) lands, apollo's per-step
  K/V append would benefit from the same pre-allocation.

### P0 — other correctness / performance

- **`LocalFusedExecutor`.** Phase 2 of the
  [engine-state-vs-execution spec](../larql-inference/docs/specs/engine-state-vs-execution.md)
  needs a fused executor for `standard` + `boundary_kv` to migrate
  without losing Metal fast path performance. Open design question
  (spec §9): `KvHandle` opaque cache vs `SharedKV` tuple for fused
  executor's return shape. Probably needs sibling methods on the
  `LayerExecutor` trait (`run_prefill_fused` / `run_decode_step_fused`)
  with default-None for per-layer executors.
- **`BoundaryKvEngine::resume`.** Spec §6.3 describes restoring from a
  frame chain via `MarkovResidualEngine::recompute_kv`. The frame
  emission half is shipped; resume is not. Restore-parity test fixture
  needed (capture frame, verify first-5-tokens agreement under
  `D-@high`).
- **D-METAL-PLE** *(carries from larql-compute roadmap)*: Per-Layer
  Embeddings not implemented in Metal. Engines on Gemma 4 E2B fall through
  the deliberate CPU fallback in `gpu.rs:372-374`, costing ~30× decode.
  Fix is a 1-2 day Metal port of `forward/ple.rs`. Engines themselves are
  PLE-agnostic; the gain accrues through the shared `decode_token` Metal
  path.
- **Engine-level profiler coverage.** *(See W7 above — this is now
  the unblocker for the entire P0 performance workstream.)* Today
  `markov_residual`'s dense path (`rs_decode_step_profiled`)
  populates `EngineProfiler`, but the Q4K decode path
  (`rs_decode_step_walk`) does not, and the other engines never
  populate it at all. Without per-stage attribution on the Q4K
  path, the per-engine optimization workstreams (W1-W6) are
  unfalsifiable. Wire it before starting W1.

### P0 — sibling trait extraction for non-K/V engines (Apollo, Mode 5) — **LANDED 2026-05-24**

**Status:** Closed. See the "Closed (recent)" entry for the migration
summary. Section retained below as the canonical motivation /
decision record.

**Problem.** The `KvEngine` trait surface assumes per-step K/V append,
FFN dispatched through `FfnBackend`, and state reconstructible to
K/V tensors. Apollo violates all three (`engines/apollo/engine.rs`:
`_ffn` unused, `decode_step` re-forwards full `context_tokens` each
call, state is residual delta + boundary residual + token list — no
K/V). Mode 5 / Graph-Grounded will violate the same three when it
lands. The trait's `Option<T>` return type also collapses
semantically distinct outcomes — empty prompt, backend unavailable,
retrieval miss, internal error, decode-before-prefill invariant
violation — into a single `None` the harnesses route incompatibly:
`accuracy_suite/runner.rs` silently drops via `filter_map` (Apollo's
store-miss prompts disappear from the JSON, structurally shorter
result vector than other engines), while `engine_runtime.rs` aborts
with `"engine prefill failed"` on the same `None`. Same trait method,
two semantics, neither implements the spec's documented
`fallback_mode = standard` from
[`docs/state-policy.md`](docs/state-policy.md) §3.

**Resolution.** Extract a `RetrievalEngine` (or `QueryEngine`) sibling
trait that drops the per-step K/V append assumption and the
`FfnBackend` dispatch requirement. Apollo moves to it; Mode 5 lands
on it directly. Tighten both trait return types from `Option<T>` to
`Result<T, EngineError>` with a typed error enum so the two harnesses
agree on a single taxonomy and downstream consumers can route on
error kind. Harness dispatch goes through an `AnyEngine::{Kv,
Retrieval}` enum that branches once at construction.

**Scope (atomic — six touchpoints).** Partial application is worse
than no application; a half-refactored trait surface has three
disagreeing semantics instead of two.

1. New `RetrievalEngine` trait. `Apollo` impl moves from `KvEngine`
   to `RetrievalEngine`. Internal behaviour unchanged.
2. `KvEngine::prefill` / `decode_step` (and `*_quant` / `*_via_executor`
   variants) return type changes from `Option<T>` to
   `Result<T, EngineError>`. **All eight `KvEngine` impls touched** —
   `standard`, `no_cache`, `markov_residual`, `markov_residual_codec`,
   `windowed_checkpoint`, `turbo_quant`, `boundary_kv`,
   `boundary_per_layer` — not just the one that motivated the
   refactor. The translation is mechanical: validated on three
   structurally-distinct samples (`markov_residual` for arch
   preconditions, `windowed_checkpoint` for window boundaries,
   `boundary_per_layer` for calibration stores); every `None`-return
   in those engines maps cleanly to `InternalError(...)`. The
   remaining five are variations on already-validated patterns
   (`standard` / `no_cache` are simpler; `markov_residual_codec` /
   `boundary_kv` extend already-sampled families; `turbo_quant`'s
   destructive-codec failure modes are in-contract per
   state-policy §3 worked example and don't surface as `None`).
3. `AnyEngine::{Kv, Retrieval}` dispatch enum at harness boundary.
   Construction parses to one or the other; execution branches once.
4. Accuracy harness (`accuracy_suite/runner.rs`,
   `larql-cli/src/commands/primary/accuracy_cmd.rs`): per-error-kind
   handling replaces `filter_map`; miss-rate surfaces as a first-class
   `served_rate` column inseparable from `match_rate`.
5. Bench harness (`engine_runtime.rs`): distinguish recoverable from
   internal errors; recoverable misses log a skip note but don't
   abort the whole run.
6. `LayerEngine` / `ZoneEngine` (per
   [`layer-engine.md`](../larql-inference/docs/specs/layer-engine.md),
   [`zone-engine.md`](../larql-inference/docs/specs/zone-engine.md))
   consume `AnyEngine` rather than `Box<dyn KvEngine>`.

**Three findings from the validation pass that constrain the design.**

1. **Interim `ffn_backend` JSON limitation (until this refactor lands).**
   Item 1's schema fix (predecessor PR — see "Predecessors" below)
   reports `ffn_backend` as the value passed at engine construction,
   *not* the FFN backend actually used during the run. For engines
   where the trait method dispatches to multiple internal paths with
   different FFN usage (`markov_residual`'s CPU path uses `_ffn`;
   its `*_via_executor` path uses `ffn` — same engine, same trait,
   different ffn-honoring), the reported value may not reflect which
   backend actually executed. **Downstream consumers should not
   condition on `ffn_backend` for engines where this distinction
   matters until this refactor lands.** The fix falls out naturally
   from the typed `Result` carrying path information; deferring to
   the refactor preserves Item 1's 200-300 line scope.

2. **`InternalError` sub-taxonomy is load-bearing for production
   observability — required design decision, not discretionary.**
   "decode_step called before prefill" (`markov_residual::engine.rs:103`,
   `boundary_per_layer::engine.rs:184`, others) is structurally
   different from "the inner backend returned None for an opaque
   reason." The first indicates a harness-level dispatch bug that
   wants immediate investigation; the second indicates a runtime data
   condition that wants diagnostic logging. Collapsing both into a
   single `InternalError` makes production logs unable to distinguish
   these alerting categories. **Recommend splitting `EngineError`
   into `InvariantViolation { what: String }` and
   `BackendFailure { details: String }` as two top-level variants**
   (not a sub-tag under a single `InternalError`). This is the
   trait-extraction PR reviewer's first design call.

3. **Extensibility note — the four-variant enum is not a permanent
   ceiling.** Currently-invisible failure modes — `windowed_checkpoint`'s
   "request crossed an uncheckpointed window boundary" (collapsed
   into generic `process()` None today), `boundary_per_layer`'s
   "calibration record missing for policy fingerprint" (a
   construction-time `.expect()` panic at `lib.rs:362` today) — are
   real conditions the typed `Result` surface *enables* surfacing
   without further trait changes. The starting enum
   (`EmptyPrompt`, `BackendUnavailable`, `RetrievalMiss { reason }`,
   `InvariantViolation`, `BackendFailure`) is the minimum-honest
   shape, not a commitment that the taxonomy is closed. New variants
   are deliberate schema changes — exhaustive enum, breaking changes
   on extension, no `#[non_exhaustive]`. Defaulting new variants
   into existing arms reproduces the silent-drop problem one layer
   down.

**Blocks.** Item 5 in the conversational priority queue (Mode 5 /
Graph-Grounded engine wiring). Mode 5 lands as a `RetrievalEngine`
impl once this refactor is in; its canonical state (retrieval graph
+ token archive) is already accommodated by
[`docs/state-policy.md`](docs/state-policy.md) §2.1's open-list of
canonical state kinds.

**Predecessors.** Item 1 schema fix (`ScoreOutcome` enum +
`served_rate` column in `accuracy_suite/runner.rs`) ships as interim
diagnosability. Its `ScoreOutcome` variants mirror the eventual
`EngineError` enum so migration is a flat projection when this
refactor lands; the field stops being interim, only its construction
path moves from harness-side `match` to engine-side `Result`.
Item 3 (Apollo into `larql accuracy` coverage) is safe to land once
Item 1 ships, but its rows will only be properly diagnosable after
this refactor.

**Closes.** [`docs/state-policy.md`](docs/state-policy.md) §8 Open
Question 1 ("Where does Apollo's fallback live? Two engines stacked
or one engine with `fallback_mode = standard`?"). The state-policy
patch declaring Q1 resolved lands in the same PR as this refactor —
patching the spec to mark Q1 resolved while the harnesses still
disagree would reproduce the same category of error the spec already
commits with `fallback_mode = standard` (documenting intent as if it
were implementation).

### P1 — capability extensions

- **Complete the FFN policy harness arc.** Item 2 v0
  (`FfnBackendKind` + `FfnLayerPolicy` parser + `ValidatedFfnLayerPolicy`
  + `BoundFfnRouter`) shipped 2026-05-24 along with the
  cross-product accuracy harness (see "Closed (recent)"). Three
  follow-ons remain, all blocked on either the sibling trait extraction
  or the `RemoteWalk` build path landing:
  - **Q4K `--ffn-policy` honoring.** `run_engine_q4k` in
    `larql-cli/.../bench/engine_runtime.rs` accepts the flag but
    logs "not yet honored" and uses the engine's internal Q4K
    routing. Honoring it requires the Q4K dispatch trait to take
    `&ModelWeights` instead of `&mut ModelWeights` so a
    `BoundFfnRouter` (which holds `&weights`) can coexist with the
    engine call. Naturally folds into the sibling trait extraction
    (P0 above) since that overhauls the trait surface anyway.
  - **`RemoteWalk` build path.** `FfnBackendKind::RemoteWalk` parses
    but errors with `RemoteWalkNotYetWired` in `build_router`. Wiring
    needs the `RemoteWalkBackend` connection pool plumbed through
    the build path. Slice estimate: ~200 lines.
  - **Bench `--ffn` URL/policy flag unification.** Bench keeps two
    flags today: `--ffn <URL>` (legacy, selects the remote-FFN
    bench scenario via `run_concurrent_ffn`) and `--ffn-policy <SPEC>`
    (new, selects engine-internal FFN backend). Once `RemoteWalk`
    builds work, `--ffn http://x:8080` can become sugar for
    `--ffn-policy remote-walk:endpoint=http://x:8080` and the two
    flags merge. Until then they stay separate. Documented in
    `engine_runtime.rs:run_engine_q4k` and the `--ffn-policy` doc
    comment in `bench/args.rs`.
- **Wire `--ffn http://...` through the executor surface.** The
  existing `--ffn` flag uses `run_concurrent_ffn` (separate path that
  routes through the `larql-metal` reference, not the engines). Once
  the four remaining engines (P0) are on `*_via_executor`, the bench
  should be able to compose `--engine markov-rs-codec:window=512
  --ffn http://shard:8080` and have the codec engine drive remote FFN
  with bounded local memory. Spec §7 calls this out as a primary use
  case.
- **Auto-rewind variant of `boundary_kv`.** Discussed mid-session as the
  only way to combine Metal's fast-path tok/s with bounded memory: emit
  boundary frame every N chunks, reset Metal's K/V cache, re-prefill
  from the last frame. Bounded memory at ~99% of fast-path tok/s with
  periodic re-prefill spikes. Would need an `evict_after_chunks` config
  on `BoundaryKvEngineConfig` plus a `backend.reset_kv_cache()` call
  after the capture. *Note (post 2026-05-17 bypass strip): this is a
  cleaner alternative to per-layer engines for "bounded memory at
  fused speed" — explicitly composes with standard rather than
  bypassing into it. Should benchmark against the W2-optimised
  markov_residual to see which model wins for long-context decode.*
- **Per-layer codec calibration sweep harness.** `BoundaryPerLayerEngine`
  ships with `BoundaryCalibrationStore` trait + `InMemoryCalibrationStore`,
  but the actual sweep tool that populates records (per-layer fragility
  measurement → policy generation → end-to-end KL validation) is not in
  tree. Per spec Phase 1 of
  [boundary-per-layer-engine.md](../larql-inference/docs/specs/boundary-per-layer-engine.md).
- **Page-aligned KV slabs for `windowed_checkpoint`.** The current
  `CheckpointStore` uses owned `Vec<f32>` per layer per checkpoint; a
  hugepage-backed slab would cut allocation churn and improve thermal
  steadiness during 370K-token replays.
- **Apollo store on disk.** `apollo` currently expects an in-memory
  `ApolloStore`. Add an mmap loader that reads the constellation map +
  boundary residuals from the same vindex-style on-disk layout as
  `down_meta.bin`, so apollo can serve ~10⁵-entry stores without RAM cost.
- **TurboQuant SIMD packing.** The Lloyd-Max codec works at scalar f32
  today; the rotation step is amenable to NEON / AVX2 vectorisation.
  *(Now also W4 in the P0 performance workstream — promote to P0 once
  W3 (incremental encode) lands so the per-row codec cost is what's
  left to vectorise.)*

### Falsified hypotheses / closed investigations (don't re-litigate)

- **`build_pipeline_layers` per-step vtable cost** — falsified
  2026-05-18 via samply flamegraph. Hypothesised as the cause of
  `standard`'s 105.9 → 99.4 regression after the kv_dispatch
  refactor; actual flamegraph showed `__bzero` +
  `zip_mut_with_same_shape` + `madvise` as 58% of CPU on per-layer
  engines (allocation churn, not dispatch overhead). The ~6 vtable
  indirections × 34 layers per step is real but ns-scale, not
  meaningful.
- **`let index = index?;` early-return branch cost** — same
  falsification. Branch is one ns-scale prediction; would not show
  as a measurable hot path.
- **`Option<&dyn KvIndex>` fat-pointer spill** — same falsification.
  Register spill is ns-scale; flamegraph showed memory operations
  not spill-related code paths.
- **`Map<I,F>::fold` 13.2% of CPU** — investigated 2026-05-18, traced
  via two-hop parent attribution to
  `larql_vindex::format::weights::load::embeddings::load_embeddings`
  → `decode_f16` of the 256K × 3072 × 2-byte embedding table. **This
  is load-time cost, not decode-time.** Visible in the profile only
  because samply records the full process lifetime; not actionable
  for the decode hot path. Don't re-investigate Map::fold as a
  decode hot-path lever.
- **`synthesize_lm_head_kquant` 19% of CPU on first profile** — same
  attribution: load-time only. The 50-tok profile had high load:decode
  ratio; at 1000 tokens it drops to 5%. Not a decode-hot-path issue.

### Investigation tooling

- **samply + `/tmp/symbolize.py` + `/tmp/symbolize_callers.py`.** The
  cargo-flamegraph-equivalent stack on this machine. Setup steps:
  1. Add `[profile.release] debug = "line-tables-only"` to root
     `Cargo.toml`. **Remember to revert before shipping** — release
     binaries bloat ~3× with line tables.
  2. `samply record --save-only --unstable-presymbolicate -o
     /tmp/profile.json --no-open -- target/release/larql bench
     gemma3-4b-q4k-v2 --tokens 1000 --engine <spec>`
  3. `python3 /tmp/symbolize.py` for top-N self-times.
  4. `python3 /tmp/symbolize_callers.py "<symbol-fragment>"` for
     two-hop call-stack attribution of generic frames.
  5. For decode-only profiles, use `--tokens 1000` so decode
     dominates over prefill / load.

### P2 — research / sequencing

- **Non-`Bf16` codecs in `markov_residual_codec`.** v0.1 ships `Bf16`
  only as the safely-defaultable cold codec. `Int8Clip3Sigma`,
  `AdaptiveBlockG32`, `PerGroupInt4G128` are present in `larql-boundary`
  but Exp 46 showed mid-layer failure for `Int8Clip3Sigma`. The
  per-architecture calibration sweep (P1) gates their promotion to
  defaults. Until then, `BoundaryPerLayerEngine` with a custom policy
  is the way to use them.
- **`MarkovResidualCodecEngine` cold tier on actual Q4K deployments.**
  Bench results confirm 50% cold tier saving on dense models and on
  Q4K Gemma with `--via-executor`. Production deployment scenario:
  long-context decode (10k+ tokens) on a 64 GB consumer Mac with a
  large model — the codec's bf16 cold tier is the difference between
  fits-in-RAM and OOM. No technical work blocking this; needs a
  recipe / docs.
- **Cross-engine comparator.** Today `larql bench --engine <spec>` runs one
  engine at a time and `benches/engine_decode.rs` exercises Standard vs the
  parity oracle. The synthesis question is: which engine wins for which
  prompt regime (long-context QA vs short-prompt multi-turn vs streaming
  generation)? A criterion harness sweeping prompt length × decode length ×
  batch size against the production `KvEngine` impls would surface this —
  the retired `kv-cache-benchmark::kv_strategies` synthetic comparator
  measured the wrong thing (encode/decode of random vectors, not real
  decode steady-state).
- **Compositional engines.** `apollo + turbo_quant` would put quantised
  K/V inside the boundary windows; `markov_residual + apollo` would let
  the residual recompute path read pre-projected boundary residuals.
  `markov_residual_codec + boundary_kv` would give bounded cold +
  cross-session resume. Neither is wired today; the trait already
  supports composition because engines hold the persistent state, not
  the dispatch — but the executor + state-policy separation (Phase 2
  spec) makes composition cleaner.

## Non-goals

- **Sampling.** Engines return hidden states; sampling lives in
  `larql_inference::layer_graph::generate::Sampler`. Don't add sampling
  helpers here.
- **Tokenisation / chat templates.** Out of scope; the engines operate on
  `&[u32]` token IDs already produced by `larql_inference::tokenizer` /
  `chat`.
- **Generic K/V backends for non-transformer architectures.** The
  `KvEngine` trait references `ModelWeights` directly. Generalising to
  state-space models or RNNs is not on this roadmap; rebuilds are cheap
  and that effort would belong in larql-inference's layer-graph surface.

