# Changelog — larql-kv

All notable changes to `larql-kv` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/) conventions
with dated entries (`YYYY-MM-DD`) instead of semantic versions during the
pre-1.0 phase. Forward-looking work lives in [`ROADMAP.md`](ROADMAP.md).

## Apollo store loading no longer trusts file-declared sizes or file types (2026-11-03)

Issue #679 and the reviews of PR #684. A store of a few hundred bytes, or a sparse
file of 16 KiB on disk, could abort `ApolloStore::load`, and a FIFO could hang it.

**The load path no longer calls `zip::ZipArchive::new`.** It reserves capacity for
the record count an archive declares, limited by the file's logical length and with
an infallible `Vec::with_capacity`, so a sparse `.npz` could abort the process. No
entry point on its seekable reader avoids that and zip 9.2.2 still reserves
infallibly, so `npz.rs` is a stored-only reader. Its directory is one fallible
allocation of fixed-size records, capped by `max_records`. `zip` is now a
dev-dependency here, used to write fixtures and to compare the two readers under
fuzzing.

**Files must be regular.** The file type is checked on the open handle, opened with
`O_NONBLOCK` on Unix, so devices, FIFOs and directories are refused without blocking.
Symlinks to regular files still load.

**Reads have limits.** `StoreLimits` and `ApolloStore::load_with_limits` are new; `load`
uses the documented defaults in `store_limits.rs`: 1 MiB for the manifest, 8 MiB for
each boundary or residual, 512 MiB for each archive, 2 GiB for the whole store and
131,072 records per archive or windows per manifest. Input limits are not allocation
safety, so anything sized from a file's contents uses fallible allocation and reports
`AllocationRefused`.

**This is not protection against memory exhaustion.** Limits compare logical length,
so a sparse file inside them still costs real memory, and on a host that overcommits
that can still end in an OOM kill. The worst case is
`max_store_bytes + 8/7 * max_archive_bytes + max_records * ~200 bytes`, because the
entries' bytes are still held while they decode. A sparse store at the defaults was
measured on aarch64 macOS at 2.57 GiB live and 2.59 GiB maximum RSS. Fixed
small allocations remain infallible: JSON values within the manifest limit, `.npy`
headers, paths and error text.

**Newly refused:**

- Stores over the default limits, including some the format allows. Use
  `load_with_limits`.
- Device, FIFO and directory paths.
- `.npz` files whose central directory isn't immediately before the end record (or a
  ZIP64 end record there), whose ZIP64 end record has an extensible data sector as
  well as a prefix, or whose ZIP64 records disagree with a plain end record.
- Members whose extents overlap or reach into the directory, including records sharing
  a payload.
- Members whose declared and stored sizes differ, whose local name differs from the
  directory, or whose flags say encrypted, strong encryption or patched data.
- Nonzero ZIP64 disk numbers.
- Compressed members in any build, including one that enables `zip/deflate`. They
  were already refused in every build in this workspace.
- More than one `entries.npy`, where numpy and `zip` read the last and this reader used
  to read the first.
- `entries.npz` with no member named exactly `entries.npy`. A prefix such as
  `entries_old.npy` used to match.
- Window members not named canonically, such as `00.npy`, `+1.npy`, `1` or `1.npy.npy`.
  These used to be taken as window ids.

**Changed behaviour:**

- A residual that can't be read, is not a regular file, is over its limit or hits a
  refused allocation is now an error. It used to read as absent. A missing residual
  or one whose contents don't parse still reads as absent.
- `StoreLoadError::Zip` is replaced by `StoreLoadError::Archive`, with a typed
  `ArchiveError`. A failed member CRC is now `ArchiveError::CrcMismatch`, not `Io`.
- `num_windows` over `max_records` is `TooManyRecords` rather than the first missing
  boundary file.
- A compressed member that the load doesn't read, such as a README in an archive,
  no longer fails the load.

## Windowed engines keep the fused path; the bench measures a whole token (2026-08-04)

Started as "check the engines benchmark correctly" and the instrument was the
first finding.

**The bench was not measuring a token.** The engine harness stopped its timer
before `pick_next`, while the reference rows included lm_head — both landing in
the same tok/s column, so every engine read 2-3x faster than production for
free. The CPU run made it exact: the engine's whole measured step (4.12 ms)
equalled the reference's *forward stage alone* (4.109 ms). Both halves are
inside the step now, prefill included (the reference's `prefill_ms` encloses its
first `lm_head_predict`), and each row carries a `fwd=` / `head=` split.

**Memory had two undercounts.** Metal's coarse pipeline keeps K/V behind a
sentinel handle, so engines reported 0 bytes owned; and the CPU whole-model
handle was measured with the per-layer formula — a 28x undercount that printed
as "13x vs std-kv" for an engine doing no compression. Backends now report what
they hold (`backend_resident_kv_bytes`), whole-model handles report every layer
(`KvHandleInner::resident_bytes`), and no ratio is invented when nothing was
measured.

**Rows say which path they took.** `DispatchPath` plus the backend's
`per_layer_is_host_delegated` answer, so `[coarse]` and `[per-layer→host]` are
visible. That matters because on Metal *every* per-layer dispatch method
delegates to `CpuBackend` — a windowed engine ran its whole forward on the host
under a `[metal (GPU)]` label.

**Per-layer SWA on the CPU attention path.** It had none, while the Metal
pipeline spec carried a window, so a Gemma-class model attended full history on
layers the architecture declares sliding. Both now resolve through one rule,
`effective_attention_window_for_layer`; a layer declaring itself sliding without
a width is answered "no window" deliberately rather than by an `unwrap_or(0)`
that meant two different things in two places.

**Windowed engines keep the fused path.** A window promises bounded attention
*and* bounded K/V; the coarse surface had neither, so windowed engines fell to
per-layer — 9.4x on Gemma 3 4B. Added `coarse_prefill_windowed` /
`coarse_decode_step_windowed`, fail-closed so a backend that cannot bound both
declines and nothing regresses. CPU trims the cache before each step; Metal
clamps the attention span every step and compacts at 2x the window so the
memmove is O(1) amortised. Metal, 80 steps at `window=8`: **11.61 ms / 2.4 MB**
against 12.06 ms / 23.6 MB unwindowed, from 115.44 ms before.

That needed a prerequisite: `LayerKVCache::current_len` was answering both "rows
stored" and "stream position". They agree only while nothing is evicted, so
compaction would have rewound RoPE on every later token. Split into
`abs_position` and `current_len`.

**First GPU-path tests in this crate** (`tests/gpu_engine_parity`). Every prior
test, bench and pin built engines with `cpu_engine_backend()`, and the `gpu`
feature gates only a dependency — the test count was identical with and without
it. Criterion covered 7 of 9 engines and timed apollo's `RetrievalMiss` as a
250x win; the roster is pinned now.

**A cross-backend divergence closed, after two wrong diagnoses.** Metal's
batched prefill disagreed with the CPU by 23-43% on the Gemma-3 fixture. Blamed
first on per-layer SWA (wrong — no window resolves on that fixture), then on a
`head_dim` shape assumption (wrong, and asserted without testing — a sweep over
head_dim ∈ {32…512} diverged at every shape including the real model's 256).
The cause was the fixture declaring Gemma-3's QK-norm keys and never populating
the weights, so the two backends disagreed about a declared-but-absent weight.
Real checkpoints always carry them, which is why a real Gemma 3 4B agreed to
4.0e-7 throughout. Fixture fixed; `make_test_q4k_weights_with_dims` added,
because every Q4K fixture was pinned to `head_dim = 64` and shape sensitivity
was untestable by construction.

**Open:** `standard:window=N` still declines the fused path when the *prompt*
exceeds the window — the fused prefill has no per-query-position masking. Metal
holds up to 2x the window resident between compactions (attention is still
bounded at the window).

## Spin-barrier pool — CPU MoE decode caught llama.cpp (2026-06-13)

After residency closed the byte-traffic gap (06-11/12), a `/usr/bin/sample` of
live 26B decode showed the remaining ~1.15× was **rayon fork-join overhead**,
not kernels. The decode driver runs *outside* the global rayon pool, so each of
the ~211 parallel sections/token took the cold path (`in_worker_cold →
LockLatch::wait_and_reset → __psynch_cvwait`) and workers slept between sections
— ~40% of thread-time in wait states.

**Built** [`larql_compute::cpu::spin_pool`](../larql-compute/src/cpu/spin_pool.rs):
a llama.cpp-style persistent spin-barrier pool. Workers spin on an epoch counter
and only `park` after a long idle gap; the dispatcher participates as the n-th
worker; **static strided chunk ownership** makes `completed == num_chunks` a
sound barrier (no shared resettable cursor → no stale re-claim across
back-to-back dispatches — a concurrent-dispatcher test caught that bug); a
dispatch `Mutex` + thread-local reentrancy guard make it safe for
`--concurrent`/multi-threaded tests. `par_chunks_mut` / `par_chunks_mut2`
helpers route a row-chunked parallel-for through the pool, or rayon when
`LARQL_SPIN_POOL=0`. **Default-on** (see "Decode fast path default-on" — the
whole Q4K stack ships on, opt out per stage with `=0`); both paths are
numerically identical, only the threading differs.

**Centralized** the four byte-identical `par_chunks_mut` Q4_K/Q6_K×Q8_K matvec
copies (larql-compute `cached.rs`, larql-inference `cached.rs`, lm_head ×2 in
`dense.rs` — the prior "consolidation hazard") into one
`q4k_q8k_matvec_parallel`, and routed every hot decode section (attention int8
Q/K/V/O, GQA, dense FFN gate/up/down, geglu, expert fold, lm_head q4 + f32)
through it — so when enabled the whole token runs on one hot pool.

- **Parity:** 704 compute + 1220 inference + 756 kv green, flags-off AND
  flags-on (incl. the `predict_kquant` oracles). clippy clean.
- **Profile after:** rayon eliminated from the hot path — `in_worker_cold`
  2682→0, `join_context` 10300→0, `wait_until_cold` 4463→9.
- **Measured** (M3 Max, t=8, warm, tight A/B bracket, flags **inline**):
  26B short-ctx OFF ~26.9 → ON **33–35**; n=256 OFF ~27.4 → ON **~34.9
  (+28%)** — vs llama.cpp recorded **32.1** ⇒ ~9% ahead.
- **Default-on + safe (2026-06-13):** shipped a spin→yield→park backoff (spin
  the proven window during active decode → yield once a wait outlives a token →
  park when idle, ~0 CPU; dispatcher unparks on dispatch) so the pool doesn't
  peg cores between requests — what makes on-by-default safe on a shared box.
  Also fixed a panic-safety bug (a panicking chunk killed a worker → the
  barrier spun forever): `catch_unwind` per chunk + re-raise on the dispatcher.
- **Caveat:** the pool spins during active decode (the win on a dedicated box);
  under a transient mid-decode load spike a run can still regress (an n=512 ON
  run hit 10.7 once) — `LARQL_SPIN_POOL=0` falls back to rayon if needed.

## CPU resident fast-path — all engines pluggable into it (2026-06-13)

The 2026-06-11/12 CPU fast-path arc (Q4K-direct + int8 attention, q4k
lm_head/dense residency, hand-asm kernels, KV append-in-place — see
`bench/baselines/c10_gemma4-26b-a4b_cpu_reconciled.json`) initially landed
only on `StandardEngine`: the `KvEngine::decode_step_resident` trait default
DROPPED the index (`let _ = index`), so every own-walk-loop engine stayed on
f32 attention. **Fixed:**

- New single-source dispatcher
  `larql_compute::attention::run_attention_block_decode_step_auto` — makes
  the same q4k-direct-vs-f32 per-layer choice as
  `CpuBackend::attention_step`, for callers that own `SharedKV` caches.
- `markov-rs`, `markov-rs-codec`, `turbo-quant`, `windowed-checkpoint`,
  `boundary_per_layer` now override `decode_step_resident` and thread the
  vindex down their walk loops to `_auto`. `boundary-kv` forwards both
  resident methods to its inner `StandardEngine` (was silently dropping to
  the f32 path). `no_cache`/`apollo` keep the default by design (debug /
  bench-only full re-forward).
- Regression pin: `engines::resident_identity_tests` — for 7 concrete
  engine specs, `prefill/decode_step_resident` must be BIT-IDENTICAL to
  `prefill/decode_step` with the flags off, and the covered-engine count
  must not shrink.
- **Absolute matrix + slow-engine fixes 2026-06-13** (26B, default-on incl.
  spin pool, M3 Max t=8 warm n=128). First measured: unlimited 31.8 / standard
  30.5 / boundary-kv 27.1 (**0.80×→0.89×**, its resident-forwarding fix) /
  turbo 9.4 / markov 7.8 / codec 7.3 — the recompute/codec engines sat at
  **~0.24–0.31×** because the spin pool sped up the shared attention/FFN/matvec
  but not their per-step machinery. **Then fixed all three, feature intact:**
  - **turbo-quant 9.4 → ~24** — `decompress_matrix`'s per-vector WHT decode was
    *serial on the driver* (~35% of it); fanned across the spin pool. Still
    3-4-bit compressed (decoded every step, now parallel) — no memory tradeoff.
  - **markov-rs 7.8 → 27.9, markov-rs-codec 7.3 → 27.7** — ported the W2 hot-K/V
    cache to the **resident walk** (`rs_decode_step_inner`/`_codec`): read the
    cached `hot_kv` and append the free `new_kv` from the attention step instead
    of `recompute_kv`-ing every position each step. Gated `cache_eligible =
    max_window.is_none() && no-cold` so it never tracks a window-clip
    transition; the residual `stored` stays the canonical, re-derivable state
    (the engine's point), the K/V is a droppable derivative. Parity gate:
    `#[cfg(debug_assertions)]` assert cached K/V ≡ `recompute_kv` (≤1e-2),
    exercised by `resident_identity_tests` (extended to a 10-step decode).
  Final matrix: standard 34.5 / unlimited 32.1 / markov 27.9 / codec 27.7 /
  boundary-kv 27.4 / turbo 21.1 — all **0.6–1.0× of standard** (was 0.24–0.31×
  for the slow three). 756 kv tests green debug+release, clippy clean.
- **Comparative bottleneck review + walk allocation fix 2026-06-14.** Profiled
  each engine's driver vs standard: the **shared** wall is the Q6_K expert
  matvec (all engines inherit it); each engine's *delta* is its feature
  machinery. markov/codec's −19/−20% was NOT the residual-store memcpy (~0.8% of
  the driver) — it was **per-step allocation churn**: the resident walk's
  `Array2::zeros((s_old+1, h))` rebuild + the cached-K/V `to_owned`
  (`__bzero`+`szone_malloc` ≈ 2450 driver samples, idling the worker pool at 48%
  vs standard's 80%). **Fixed:** the cache_eligible walk now `append_row`s
  `stored` in place into the W8.2 doubling-capacity buffer (mirrors dispatch.rs)
  and borrows `hot_kv` into attention via `Cow` instead of copying. Churn
  collapsed 2450→150 samples (~16×); **markov/standard ratio 0.81×→0.975×, codec
  0.80×→~1.0×** (same battery state, back-to-back). Parity: resident_identity
  (markov+codec, 10-step, buffer doubles) bit-exact + debug K/V assert. turbo's
  −39% is **inherent** (must decode compressed K/V to attend; already
  parallelized); boundary-kv/unlimited deltas are small (frame-emit/windowing).
  Remaining markov/codec ~2.5% = walk-attention serial work (shared walk
  frontier — full K/V concat + generic GQA vs standard's in-place handle).
- **In-place hot-K/V on the resident walk 2026-06-14 (closes the concat half).**
  The named ~2.5% above was the walk-attention **owned concat**: the resident
  walk drove `run_attention_block_decode_step_q4k_direct`, which allocates a
  fresh `[ctx+1, kv_dim]` K *and* V every layer every step and copies the whole
  prior cache into it before attending — **O(L²)** cache copy over an L-token
  generation, vs `standard`'s in-place append handle (O(L)). The split
  project→append→attend halves already existed for the dispatch path; the walk
  just didn't use them. **Built** `run_attention_block_decode_step_{q4k_direct,
  auto}_inplace` (larql-compute `attention/decode.rs`): projects the new row,
  appends it into the caller's **doubling-capacity** K/V buffer (grows like
  `stored`), and attends over the `[..len+1]` views — no concat. **Wired**
  markov_residual + markov_residual_codec resident walks: step-1 still
  recompute-seeds `hot_kv`; steps 2+ append in place (the steady state). The
  windowed/cold tiers and the flags-off f32 path keep the owned concat
  unchanged. Gated `LARQL_MARKOV_INPLACE_KV` (default on; `=0` → owned concat,
  the A/B reference + escape hatch). **Parity (bit-exact, 4 gates):** compute-
  level `inplace ≡ q4k_direct` concat across a capacity doubling; engine-level
  in-place-vs-owned-concat A/B with Q4K-direct **on** for markov *and* codec
  (hidden states bit-identical every step); `resident_identity` flags-off still
  green (in-place branch's None-fallback = owned concat); 758 kv + 705 compute +
  1220 inference green debug & opt, clippy clean. (The debug `hot_kv ≡
  recompute_kv` assert is gated to the f32 path — the Q4K route's projections
  differ from `recompute_kv` by >1e-2 even in f32-act; its oracle is the A/B.)
  The two q4k-flag-mutating tests serialise on `Q4K_FLAG_ENV_LOCK` (those flags
  read process env on the driver thread — no thread-local). **Perf is
  structural** (eliminates the O(L²) per-step copy; the win grows with context —
  it's the long-ctx tax behind the C10 1.29× vs short 1.15×). **Measured (26B,
  CPU MoE in-process, M3 Max t=8, n=128 warm, `LARQL_MARKOV_INPLACE_KV` A/B,
  same engine ordering):** markov 32.5→34.5, codec 32.5→34.6 with in-place on —
  and the three untouched controls (standard/unlimited/turbo) drifted *down*
  −3/−8/−6% across the A/B (machine warming), so drift-corrected the change is
  **~+11–12%**. Final warm matrix (in-place on = production default): codec
  **36.5** / standard 36.0 / markov **36.0** / unlimited 33.3 / boundary-kv 36.5
  p50 (mean skewed by frame-emit spikes) / turbo 21.2 (inherent) — **markov/codec
  now AT parity with standard** (was 0.81× at the arc's start), the whole cached
  cluster **~12% ahead of llama.cpp's 32.1**. Caveat: bench box was at ~58%
  charging (not cool-dedicated); ordering + A/B *direction* are robust, absolutes
  drifted ~5–8% run-to-run — a cool-box rerun would firm them. (NB: the first
  engine in a fresh process eats the 30GB page-in — standard read 21.8 cold,
  34–36 warm; warm runs are the fair matrix.)
- **Propagated the in-place lever to the two remaining walk engines + faithfulness
  audit 2026-06-14.** A full cross-engine spec/contract audit (all 9 engines vs
  `state-policy.md`'s `(canonical, derivative, contract)` triple) found every
  engine faithful, and flagged the two siblings still paying the O(L²) owned
  concat the markov/codec in-place change eliminated:
  - **boundary_per_layer (was the one NEEDS-FIX)** — carried NO `hot_kv` at all:
    it `recompute_kv`'d the whole hot tier *and* rebuilt an owned `[ctx+1]` concat
    every layer every step (worse than markov *pre*-W2). Added a `hot_kv`
    derivative + the W2-cache + `run_attention_block_decode_step_auto_inplace`
    steady state, mirroring its twin codec — only active in the `cache_eligible`
    (unbounded, no cold) path, like codec; the windowed/cold path (its primary
    purpose) is untouched. `hot_kv` is excluded from `memory_bytes` (droppable
    derivative, matches markov). Engine-level in-place-vs-owned-concat A/B (q4k on)
    bit-identical; f32-gated debug `hot_kv ≈ recompute_kv` assert.
  - **windowed_checkpoint** — its CPU window walk (`extend.rs`) passed the whole
    window K/V by value → backend re-concats `[n+1]` per layer per step (its own
    doc admitted "O(window²) total"). Added `rs_extend_inplace` (appends into the
    window's doubling-capacity buffer, attends over views), wired into
    `extend_current` only when eligible (index + toggle + q4k); `replay_window` /
    quant / executor / tests keep the owned concat. The engine's existing
    `current_window_kv_len` counter already treated the buffers as over-allocated
    (the dispatch path did), so `close_window`/`current_kv_bytes` needed no change.
    A/B (q4k on) bit-identical; `resident_identity` flags-off still green.
  Both reuse the shared `LARQL_MARKOV_INPLACE_KV` toggle + `Q4K_FLAG_ENV_LOCK`.
  Also: **apollo footgun guard** — `injection_layer < crystal_layer` silently
  no-ops the retrieval-injection (the compressed forward starts at `crystal`);
  added a one-time runtime warning in `prepare_injection` (experimental engine,
  warn-don't-fail). Doc-drift swept: boundary-kv spec now flags `resume` as
  NOT-IMPLEMENTED (emit half only), apollo spec `KvEngine`→`RetrievalEngine`,
  `state-policy.md` `fallback_mode` marked retired (per its own §8 resolution).
  760 kv tests green debug + opt, clippy clean. (Same caveat as above: turbo's
  −39% is inherent; boundary-kv inherits standard's opts via resident forwarding.)

Prefill stays on the f32 BLAS gemm for all engines deliberately (the task
#16 prefill falsification: q4k repeated-matvec loses ~20× to AMX at
prefill shapes).

## Hardening — codebase review 2026-05-28

From the whole-codebase review ([`docs/audits/codebase-review-2026-05-28.md`](../../docs/audits/codebase-review-2026-05-28.md)):

- **P2 — CLI-supplied sizing params can reach prefill panics**; validate at the boundary.
- **P2 — positional QKVO contract** (`attn_data[1]/[2]`, shared with larql-models) is maintained by convention, not type. Silent-drift risk — consider a typed accessor.

## [2026-05-20] — boundary_per_layer: bugfixes + W1-GPU dispatch + modular split

**Engine bottleneck audit** (`PERFORMANCE.md` §"2026-05-20"). Findings
across all engines:

- `apollo` — O(N²) **by design** (`forward_from_layer` rebuilds KV each
  step over the growing context; no cross-step persistence). Not a
  bug; documented as a contract caveat for short-query workloads.
- `boundary_per_layer` — two real O(N²) bugs, both fixed:
  - **Bug A** (hot-tier rebuild): every `decode_step` rebuilt every
    layer's `stored[layer]` via `Array2::zeros((s_old+1, h)) + assign`.
    O(N · num_layers · hidden) per step → O(N²) total in unbounded
    mode. Replaced with `ndarray::Array2::push_row` (amortised O(m)).
  - **Bug B** (cold_kv nuke): every overflow set `cold_kv = None`,
    forcing the next decode to recompute K/V over the entire cold
    tier — O(N²) windowed mode. Replaced with
    `cold_tier::extend_cold_kv_with_overflow` which appends K/V at
    each overflow at the pre-`cold_encoded.append` absolute position.

**W1-GPU dispatch wired** for `boundary_per_layer`. New
`try_prefill_via_dispatch` + `decode_step_via_dispatch` route through
the Metal-fused per-layer state-dump kernel when the backend/vindex
support it. Closes the perf gap to its sister engine
`markov_residual_codec`: **91.8 tok/s** vs codec's 92.6 (−0.9%) on
Gemma 3 4B Q4K, M3 Max — with **44% less hot memory** (19.6 MB vs
35.3 MB). Falls back to dense walk on backends/vindexes lacking
direct-matvec.

**FFN routing fix** — `boundary_per_layer`'s dense `run_prefill` /
`run_decode` previously constructed `BackendFfn` internally, ignoring
the caller-supplied `ffn`. This panicked on `--compact` vindexes
where dense FFN weights aren't present. Now routes the caller's FFN
through (e.g. `WalkFfn` from the bench CLI).

**`EngineKind` variant + parser**. `BoundaryPerLayer { window_size,
num_layers }` with three aliases (`boundary-per-layer`,
`boundary_per_layer`, `boundary-pl`); default `num_layers=34` (Gemma
3 4B), override via `layers=N`. Build dispatch seeds a uniform-bf16
`InMemoryCalibrationStore` automatically. Added to
`examples/engine_ladder.rs`.

**Parity gate** — `examples/boundary_per_layer_parity_gate.rs` runs
`boundary-per-layer` vs `markov-rs-codec` end-to-end on a real Gemma
3 4B Q4K vindex. Token-level agreement check (not bit-identity,
because incremental cold_kv vs recompute-each-step differ in BLAS
accumulation order). Pass criterion: first divergence ≥ step 5.
Result on Gemma 3 4B: **100% token agreement** across 50 tokens in
both unbounded and windowed (window=512) — RoPE positioning in
`extend_cold_kv_with_overflow` and codec round-trip are exactly
right.

**Modular split** of `boundary_per_layer/engine.rs` (1250 → 716 LOC),
mirroring `markov_residual_codec`'s module layout. New sibling files
in `engines/boundary_per_layer/`:

- `walk.rs` (204 LOC) — CPU dense walk path
  (`run_prefill` / `run_decode` as free functions).
- `dispatch.rs` (162 LOC) — W1-GPU dispatch path.
- `executor.rs` (186 LOC) — `LayerExecutor`-driven path.
- `cold_tier.rs` (130 LOC) — `extend_cold_kv_with_overflow` +
  `roundtrip` / `last_row` helpers + their unit tests.

Struct fields moved to `pub(super)` so sibling modules can read them
via free-function inputs.

**Test count**: 591 → 598 lib tests (3 parser variants + 1 cold_kv
invariant + 3 from cold_tier extraction). All passing.

The same split pattern is queued for the other 6 engines
(`markov_residual_codec`, `turbo_quant`, `unlimited_context`,
`apollo`, `boundary_kv`, and `markov_residual` last) — deferred to
follow-up turns since each requires its own care and at least one is
gated on in-flight WIP in `markov_residual/compute.rs`.

## [2026-05-16] — KV engine unification (steps 1-5 of 7)

Unifies the parallel "live decode cache" and "research KV engine" code
paths so `larql run` / `larql walk` dispatch through the same `KvEngine`
trait that `larql bench --engine` uses. Spec at
[`crates/larql-inference/docs/specs/kv-engine-unification.md`](../larql-inference/docs/specs/kv-engine-unification.md).

**Trait surface relocated.** `KvEngine` + `EngineInfo` +
`DecodeStageSummary` now live in `larql-inference::kv_engine`; this
crate re-exports them so `larql_kv::KvEngine` keeps the same public
shape. Engine impls in `larql-kv/src/engines/*` continue to write
`impl KvEngine for ...` against the same trait — just resolved through
the re-export. The trait moved upstream so the dispatch entry point
(`larql_inference::forward::generate_with_engine`) can reference it
without inducing a circular dep on `larql-kv`. See spec §10.4.

**Trait widened for FFN dispatch.** `KvEngine::{prefill, decode_step,
prefill_q4k, decode_step_q4k}` now take `ffn: &dyn FfnBackend` after
`weights`. Existing four engines ignore the parameter (FFN is recomputed
from weights as before); new param is plumbing for future engines that
route FFN remotely (`RemoteWalkBackend`, `RemoteMoeBackend`).
`larql_inference::ffn::NullFfn` added as a trait-satisfying stub that
holds no references — used by Q4K callers where `&mut weights` rules
out a `WeightFfn`.

**Two new engines** in `larql-kv/src/engines/`:

- `StandardEngine` — wraps the production K/V tensor cache. `window=None`
  matches today's `--kv-cache standard`; `Some(N)` matches
  `--kv-cache markov-bounded --context-window N`. Bit-identical output.
- `NoCacheEngine` — wraps the O(N²) re-forward fallback. Matches today's
  `--kv-cache none` on non-PLE architectures.

`EngineKind` gains `Standard { window_size: Option<usize> }` and
`NoCache` variants. `from_name` recognises `standard[:window=N]`,
`markov-bounded[:window=N]` (legacy alias → `Standard`), `no-cache`,
`none` (legacy → `NoCache`), plus existing aliases.

**Default flipped** to engine dispatch. `walk_cmd::generate_stream` no
longer carries the legacy `match` over `KvCacheKind`; it builds an
`EngineKind` from the flag and drives `generate_with_engine`.

**Bit-parity gate** lives in `larql-kv/src/engines/{standard,no_cache}.rs`:
- `Standard { window=None }` vs `generate_cached_backend(window=None)` ✓
- `Standard { window=Some(3) }` vs `generate_cached_backend(window=Some(3))` ✓
- `Standard { window=Some(64) }` short-prompt edge case ✓
- `NoCacheEngine` vs legacy `predict_with_ffn` loop ✓ (non-PLE)

**Engine-trait dispatch overhead** measured at ~1.6 % (within noise) on
the synthetic test substrate. See [`PERFORMANCE.md`](PERFORMANCE.md).

**Coverage:** new files land at 99.1 % (`standard.rs`) and 96.1 %
(`no_cache.rs`); upstream `kv_engine.rs` at 94.3 %. Per-file 90 % floor
met for everything new.

Steps 6 (CLI `--engine` flag, `LARQL_KV_ENGINE` env var, server wiring,
ROADMAP update) and 7 (cleanup) pending.

## [2026-05-10] — Coverage push

Total line coverage **67.44 % → 85.13 %** (+17.69 pp, 217 tests, +66 vs
extraction-day). 15 of 21 source files now at ≥ 90 %; the remaining 6
all carry tightened debt baselines.

| File | Before | After |
|---|---:|---:|
| `profiler.rs` | 0.00 % | 100.00 % |
| `engines/apollo/npy.rs` | 58.20 % | 93.61 % |
| `engines/apollo/engine.rs` | 71.98 % | 96.31 % |
| `engines/apollo/store.rs` | 17.81 % | 89.78 % |
| `engines/markov_residual/engine.rs` | 72.02 % | 93.23 % |
| `engines/markov_residual/q4k.rs` | 0.00 % | 57.14 % |
| `lib.rs` | 84.79 % | 90.03 % |

Notable additions:

- 8 `profiler` tests covering `StageAccumulator`, `EngineProfiler`, and
  `DecodeStageSummary` (including the `print()` smoke test for both the
  recompute-tier-present and total-zero branches).
- 4 `compliance_tests` lifting the default `KvEngine::prefill_q4k` /
  `decode_step_q4k` trait-method fallbacks via a synthetic
  `DefaultMethodsEngine` fixture.
- 5 `markov_residual::engine` tests covering profiling on/off split, the
  `with_profiling` setter, and the Q4K CPU fallback (Metal returns
  `None` → `rs_prefill_walk` / `rs_decode_step_walk`).
- 22 `apollo::npy` tests covering all `NpyError` variants, structured
  vs simple dtype dispatch, header field-parser branches.
- 13 `apollo::store` tests including end-to-end `ApolloStore::load`
  against a synthetic on-disk store built with `tempfile` + handwritten
  `.npy`/`.npz` fixtures.
- 11 `apollo::engine` tests including KvEngine `prefill` / `decode_step`
  for both compressed (boundary residual) and uncompressed paths,
  `query_greedy` smoke test, and `store()` getter.

### Warnings cleanup

Same day: removed 3 unused-import warnings in
`kv-cache-benchmark/src/real_model/{decode_comparison,runner}.rs`,
reverted a `kv_dim.is_multiple_of(hd)` clippy-fix in
`turbo_quant/engine.rs` (1.87.0 stable, MSRV 1.80), and reordered
`apollo/engine.rs` so the `KvEngine` impl precedes the test module
(satisfies clippy's `items-after-test-module`). `cargo clippy -p
larql-kv --all-targets --no-deps` is now clean.

### Cross-platform CI

Added `.github/workflows/larql-kv.yml` modelled on
`.github/workflows/larql-vindex.yml`. Test matrix runs on
`ubuntu-latest`, `windows-latest`, and `macos-14` covering fmt check,
`cargo check --all-targets`, examples, clippy, unit tests, and
bench-compile/test. Coverage job runs on Ubuntu only and gates on
`make larql-kv-coverage-policy` (the per-file 90 % floor + the 6
inherited debt baselines). OpenBLAS gets installed via apt on Linux
and via vcpkg on Windows; macOS uses the Accelerate framework — same
matrix the larql-vindex workflow already exercises.

`cargo fmt -p larql-kv` was run to bring three files (benches,
examples, an apollo test fixture) into conformance with the rest of
the workspace.

The Makefile's `larql-kv-lint` uses `--no-deps` so it doesn't trip on
pre-existing clippy debt in the larql-inference dependency. Other
crates' lint targets don't need this because they don't depend on
larql-inference.

## [2026-05-09] — Initial extraction from larql-inference

Genesis commit. The crate was carved out of
`larql-inference/src/engines/` (~5,540 LOC) where the four KV engines and
the supporting trait/dispatch had grown into a self-contained subsystem
with a real second consumer (`kv-cache-benchmark`) already importing it
through compatibility shims.

### Moved into larql-kv

| Component | Origin | Notes |
|---|---|---|
| `KvEngine` trait, `EngineKind`, `EngineInfo` | `engines/mod.rs` | Now the crate root. |
| `accuracy` module | `engines/accuracy.rs` | `softmax` re-exported from `larql_inference::forward::softmax` instead of being internal. |
| `profiler` module | `engines/profiler.rs` | Verbatim. |
| `engines::apollo` | `engines/kv_engines/apollo/` | Drop the redundant `kv_engines/` middle path. |
| `engines::markov_residual` | `engines/kv_engines/markov_residual/` | |
| `engines::turbo_quant` | `engines/kv_engines/turbo_quant/` | |
| `engines::unlimited_context` | `engines/kv_engines/unlimited_context/` | |

All `crate::{attention,ffn,forward,layer_graph,model,residual,vindex}::*`
paths inside the moved code rewritten to `larql_inference::*`.

### Stayed in larql-inference

- `engines::test_utils` — relocated to `larql_inference::test_utils`. ~20
  internal tests across `attention/`, `forward/`, `ffn/`, `layer_graph/`,
  `trace/`, `vindex/walk_ffn/` use these synthetic-weight fixtures and
  cannot follow into a downstream crate without a circular dep.

### Public-API surface widened in larql-inference

- `DEFAULT_GPU_KV_CACHE_MAX_SEQ` lifted from `pub(crate)` to `pub` in
  `layer_graph::pipeline_layer` so engines can read it from the new home.

### Removed re-exports from `larql_inference::*`

The following used to be at the `larql_inference` crate root or in
`research::*` and now live in `larql-kv`:

- `EngineInfo`, `EngineKind`, `KvEngine`
- `MarkovResidualEngine`, `UnlimitedContextEngine`
- `compare_hidden`, `cosine_similarity`, `js_divergence`, `kl_divergence`,
  `mse`, `softmax`, `HiddenAccuracy`

Downstream consumers should add `larql-kv` to their Cargo.toml and import
from there.

### Consumer updates

- `larql-cli` — `bench_cmd.rs` now imports `EngineKind` and
  `kv_memory_bytes_for_seq` from `larql_kv`. Workspace metal feature gained
  `larql-kv/metal`.
- `kv-cache-benchmark` — compat shims (`apollo/`, `turboquant/`,
  `unlimited_context/`, `real_model/markov_layer.rs`) now re-export from
  `larql_kv` directly. README updated.
- `larql-inference` examples — `apollo_rd_backend.rs` imports from
  `larql_kv::apollo`; `mech_interp_demo.rs` uses
  `larql_inference::test_utils`.

### kv-cache-benchmark cleanup

After the extraction landed, `crates/kv-cache-benchmark/src/apollo/` still
contained five orphan `.rs` files (`engine.rs`, `store.rs`, `routing.rs`,
`entry.rs`, `npy.rs`) — pre-extraction copies that the `mod.rs` re-export
shim didn't reference but had been kept around. Two `#[ignore]`'d
`real-model`-feature demo tests (`tests/test_apollo_query.rs`,
`tests/test_apollo_accuracy.rs`) called four demo helpers that lived only
in the orphan `engine.rs` (`query_greedy_with_tokenizer`,
`query_greedy_compressed`, `query_generate_compressed`,
`query_generate_uncompressed`); the test build was failing on
`--features real-model` as a result.

All seven files were deleted as part of this cleanup. The
`apollo-demo/apollo11_store` end-to-end harness can be reconstructed from
git history if needed; the underlying functionality (routing, entry
retrieval, boundary-residual injection) is exercised by the surviving
larql-kv apollo unit tests plus the `kv-cache-benchmark` criterion bench.

### Coverage at extraction

After running `cargo llvm-cov --package larql-kv` plus `profiler.rs` test
top-up, total line coverage was **69.82 %** (2 838 / 4 065 lines, 143 unit
tests + 8 new profiler tests). 10 inherited files sat below the 90 %
per-file floor and carried baselines in `coverage-policy.json` that may
only ratchet upward. See [`ROADMAP.md`](ROADMAP.md) for the remediation
list. `make larql-kv-coverage-policy` enforces the baselines.

### Rationale

The four engines collectively share a trait and dispatch but diverge on
state management. Keeping them inside `larql-inference` meant every change
to a single engine recompiled the whole inference crate (transformer
forward pass, mech-interp surface, layer graphs). They are also the
target of independent benchmarking — the `kv-cache-benchmark` crate already
treated them as separable. Splitting tightens the API contract between
"transformer forward" (larql-inference) and "KV state strategy" (larql-kv).

The cut was clean: every primitive engines depend on (`ModelWeights`,
`BackendFfn`, `WalkFfn`, `KvCache`, `forward_*`, `rms_norm_heads`, …) was
already public in larql-inference, so this extraction did not require
designing new API.

## Earlier — entries migrated from ROADMAP.md (2026-08-04)

These predate this file's adoption of Keep a Changelog dating and were
kept in the roadmap until the roadmap was split by tense.

## Crate-shape state (2026-05-17)

- Crate extracted from `larql-inference::engines` on 2026-05-09 — see
  [`CHANGELOG.md`](CHANGELOG.md).
- **Seven engines shipped** as of 2026-05-17:
  - Original four: `standard`, `no_cache`, `markov_residual`,
    `windowed_checkpoint`, `turbo_quant`, `apollo`.
  - Three new: `boundary_kv`, `markov_residual_codec`, `boundary_per_layer`.
    Specs in `crates/larql-inference/docs/specs/`:
    [boundary-kv-engine.md](../larql-inference/docs/specs/boundary-kv-engine.md),
    [markov-residual-codec-engine.md](../larql-inference/docs/specs/markov-residual-codec-engine.md),
    [boundary-per-layer-engine.md](../larql-inference/docs/specs/boundary-per-layer-engine.md).
- Consumers wired:
  - `larql-cli bench --engine <spec>` (selector dispatch)
  - `larql-cli bench --via-executor` opts into the new `LayerExecutor`
    surface; falls through to legacy path for unmigrated engines.
  - in-crate `benches/engine_decode.rs` (criterion: dispatch helpers + Standard parity)
- Coverage policy: 90 % line coverage per source file (see
  `coverage-policy.json`); CI gate at `make larql-kv-coverage-policy`.
  Workspace `larql-kv` lib total: **95.62% lines, 95.43% regions, 95.50%
  functions** (2026-05-24 evening, post coverage-debt clearance).
  **All 61 files at ≥90% lines; debt baselines cleared from policy
  file.** The 2026-05-24 push lifted the five `engines/*/dispatch.rs`
  files (range 7.95–80.68% → 93.57–97.85%) and
  `engines/markov_residual/compute.rs` (86.85→95.30%). See "Closed
  (recent)" entry for the thread-local-override pattern that makes
  the env-gated paths in `compute.rs` and the W10 mask cascade in
  the dispatch files testable without process-env mutation.

## Architectural cuts (2026-05-17)

Substantive refactors landed; specs reflect the new boundaries.

### Naming hygiene — renamed for honesty

- **`metal_fused_prefill` / `metal_fused_decode_step`** → `fused_prefill`
  / `fused_decode_step`. The "metal" was a lie — `CpuBackend` implements
  `prefill_q4` and `decode_token` via its C Q4 kernel and also takes the
  fused path on `--cpu`. The aliases in `windowed_checkpoint::engine`
  (`quant_prefill_metal`, `quant_decode_token`) follow.
- **`KvEngine::prefill_q4k` / `decode_step_q4k`** → `prefill_quant` /
  `decode_step_quant`. The `_q4k` suffix baked one format into the trait
  surface; the trait is quant-agnostic (dispatches on `index`'s format).
  Internals that are genuinely Q4K-specific (`prefill_q4k_moe`,
  `cpu_q4k_cache_*`, `run_ffn_decode_step_q4k_direct`) keep their names.
- **`ComputeBackend::has_q4()` → `supports_quant(format: QuantFormat)`.**
  Per-format predicate; `CpuBackend` reports support for `Q4_0`, `Q4_K`,
  `Q4_KF`, `Q6_K`; `MetalBackend` adds `Q8_0`. Backends can advertise new
  format support without trait extension.
- **Storage slots `q4k` → `kquant` for K-family fields.** `attn_q4k`,
  `interleaved_q4k`, `set_attn_q4k`, `load_attn_q4k`, etc. — these hold
  K-family quant bytes (Q4_K, Q4_KF, Q6_K — manifest tag picks). Q4_0
  (`attn_q4`) and Q8 (`attn_q8`) slots stay — genuinely format-specific.

### Engine state vs execution — new abstraction

Spec: [engine-state-vs-execution.md](../larql-inference/docs/specs/engine-state-vs-execution.md).

The engines were re-coupling backend / FFN / format decisions into their
state-management code. The new shape:

- **`LayerExecutor` trait** (in `larql-inference::layer_executor`) —
  per-layer execution surface with `run_prefill_layer` /
  `run_decode_layer` returning `(hidden, SharedKV)`. Dispatch kind
  (`Fused` / `PerLayer`) is explicit.
- **`LocalWalkExecutor`** — wraps `run_attention_with_kv_backend` +
  the caller's `&dyn FfnBackend`. The critical decoupling: the executor
  does **not** construct its own `WalkFfn` — it uses whatever the engine
  was handed.
- **Engine trait extension:** `KvEngine::prefill_via_executor`,
  `decode_step_via_executor`, `prefill_quant_via_executor`,
  `decode_step_quant_via_executor`. Default impls fall through to the
  legacy methods so unmigrated engines work unchanged.

### Engines on the new surface

Every engine now runs its own state-policy code; there is no hidden
fall-through to the backend's fused kernel from per-layer engines.
`standard` (and by delegation `boundary_kv`) is the **only** engine
that exercises the fused fast path — via
`ComputeBackend::coarse_prefill` / `coarse_decode_step`, which on
Metal calls `larql_inference::vindex::fused_prefill`.

| Engine | Default dispatch | `*_via_executor` override | Honors FFN backend | Tok/s (Gemma 3 4B Q4K, Metal) | Hot state |
|---|---|---|---|---:|---:|
| `standard` | `ComputeBackend::coarse_prefill` (fused fast path) | n/a (no per-layer code to migrate) | n/a | 104 | 0 MB (backend owns K/V) |
| `boundary_kv` | Delegates to `standard` + emits boundary frames | n/a | n/a | ≈104 | 0 MB |
| `markov_residual` | Per-layer walk via `rs_prefill_walk` | ✅ | ✅ counter test | 3.6 | 6.0 MB |
| `markov_residual_codec` | Per-layer walk via `rs_prefill_codec_walk` (bf16 cold) | ✅ | ✅ counter test | 4.3 | 6.0 MB |
| `windowed_checkpoint` | Windowed checkpoint extension via `process_q4k` | ✅ | ✅ counter test | 25.6 | 4.8 MB |
| `turbo_quant` | Per-layer WHT + Lloyd-Max compression cycle | ✅ | ✅ counter test | 3.9 | 0.6 MB |
| `boundary_per_layer` | Per-layer walk with per-layer codec policy | ✅ (dense) | ✅ counter test | — | matches markov_residual_codec |
| `apollo` | Whole-forward through `forward_layer_range` (boundary prefix + perturb) | ✅ | ✅ counter test | requires store | scales with store |
| `no_cache` | Full re-forward per step (O(N²) wall-time) | ✅ | ✅ already did on legacy `prefill` | — | token list only |

## Closed (recent)

- **2026-05-24 — Multi-modal engine seam (ADR-0023).** `KvEngine` gains
  `supports_multimodal()` (default false) + `prefill_from_hidden(weights,
  ffn, initial_hidden: &Array2<f32>) -> Result<Array2<f32>, EngineError>`.
  `StandardEngine` is the first (and currently only) MM-capable engine.
  Other engines inherit the default-false convention — they remain
  text-only until each individually implements the new method.
  `AnyEngine` forwards both methods. `generate_with_engine_from_hidden`
  wrapper shares the decode loop with `generate_with_engine`. Dispatch
  helpers `kv_prefill_from_hidden_via_dispatch` (sync + async) hoist the
  embed step out of the prefill loop so both text-only and MM inputs
  follow the same layer-forward path. The eventual end state: every
  engine implements `prefill_from_hidden` and `prefill(token_ids)` becomes
  a thin wrapper. No timeline on the seven-engine migration.

- **2026-05-24 — Sibling trait extraction LANDED.** `KvEngine`
  `Option<T>` returns are gone; the typed `EngineError` enum lives in
  `larql-inference::kv_engine` alongside the new `RetrievalEngine`
  trait + `AnyEngine` dispatch enum. The two-harness silent-drop /
  panic disagreement (`accuracy_suite/runner.rs` vs
  `bench/engine_runtime.rs`) is resolved at the type level.

  **Trait surface:** all 8 `KvEngine` impls (`standard`, `no_cache`,
  `markov_residual`, `markov_residual_codec`, `windowed_checkpoint`,
  `turbo_quant`, `boundary_kv`, `boundary_per_layer`) return
  `Result<Array2<f32>, EngineError>` on `prefill` / `decode_step` /
  `*_quant` / `*_via_executor`. Apollo moves to the new
  `RetrievalEngine` trait (`prefill(weights, token_ids)` /
  `decode_step(weights, token_id)` — no `FfnBackend`, no per-step K/V).

  **EngineError variants** (exhaustive, no `#[non_exhaustive]`,
  thiserror): `EmptyPrompt`, `BackendUnavailable`, `RetrievalMiss
  { reason }`, `InvariantViolation { what }`, `BackendFailure
  { details }`. Per Finding 2, `InvariantViolation` and `BackendFailure`
  are kept as two top-level variants to preserve the alert-routing
  distinction (a dispatch bug vs a kernel/data failure). The accuracy
  harness's `ScoreOutcome` mirror followed suit:
  `SkippedInternalError` → `SkippedInvariantViolation` +
  `SkippedBackendFailure` (load-bearing JSON schema change for
  downstream observability).

  **AnyEngine** (`AnyEngine::Kv(Box<dyn KvEngine>) |
  Retrieval(Box<dyn RetrievalEngine>)`) is the harness boundary type.
  Forwarding methods (`prefill` / `decode_step` / `prefill_quant` /
  `decode_step_quant` / `*_via_executor`) take the superset of args
  from both surfaces and ignore the irrelevant ones on the retrieval
  arm. This intentionally walks back the original "don't lift a common
  shape" plan — the harness scalability won out, since the alternative
  is N×2 match arms per call site as more retrieval engines land.

  **Bench harness merged.** `run_engine` + `run_engine_q4k` collapsed
  into one `run_engine(weights, index: Option<&VectorIndex>, ...)`.
  When `index = Some` the dispatch goes through `prefill_quant`
  (quant-agnostic — the vindex's format flows through the engine);
  when `None` the dense `prefill` path runs. FFN selection: dense
  defaults to `WeightFfn`, quant defaults to `NullFfn` (preserves the
  pre-merge Q4K behaviour). `--ffn-policy` honored on dense, logged
  as not-yet-honored on quant due to the `&mut weights` vs
  `&weights`-borrowing-router conflict (unchanged from pre-merge).

  **Coverage debt:** one re-introduced baseline at
  `markov_residual/engine.rs` (89.5% vs 90% floor). The remaining
  uncovered lines are all `.ok_or_else(|| BackendFailure)?`
  constructions that only fire when an internal helper
  (`rs_decode_step_walk`, `recompute_kv`, `executor.run_*_layer`)
  returns None. Triggering those requires the mock `EngineBackend`
  infrastructure that the 2026-05-24 coverage-clearance explicitly
  deferred; the baseline tracks the debt rather than gold-plating
  ahead of need.

  **Outcomes.** Test count larql-kv lib: 712 → 726 (+14). Workspace
  builds clean. `make larql-kv-ci` passes (fmt + clippy + tests +
  fresh coverage policy with 1 baseline). Apollo's `executor.rs`
  deleted (~150 lines of dead code from the old KvEngine `*_via_executor`
  impls). Closes [`docs/state-policy.md`](docs/state-policy.md) §8
  Open Question 1 ("Where does Apollo's fallback live?"); also closes
  the interim `ffn_backend` JSON limitation flagged in Item 1 of the
  2026-05-24 accuracy harness work.

  **Follow-ups** *(deferred to keep this PR atomic)*:
  - Mode 5 / Graph-Grounded engine lands as a `RetrievalEngine` impl
    (was blocked on this refactor).
  - Q4K `--ffn-policy` honoring (was waiting on the same
    `&mut weights` borrow conflict — still present after the merge
    because the trait surface still takes `&mut weights` for lazy
    dequant).
  - `RemoteWalk` build path (~200 lines, standalone — was the second
    blocked item).
  - `markov_residual/engine.rs` coverage debt + mock `EngineBackend`
    infrastructure (deferred per "Sub-project A" of the previous
    coverage push).

- **2026-05-24 — Coverage debt CLEARED.** All six files below the
  90% per-file floor lifted; `make larql-kv-coverage-policy` passes
  against fresh `summary.json` regeneration. Workspace total 95.62%
  lines, 61/61 files at ≥90%, 0 debt baselines remaining.

  Files lifted (pre → post): `turbo_quant/dispatch` 9.35→97.85%,
  `boundary_per_layer/dispatch` 7.95→93.57%, `windowed_checkpoint/dispatch`
  59.09→97.24%, `markov_residual/dispatch` 77.51→96.78%,
  `markov_residual_codec/dispatch` 80.68→97.72%,
  `markov_residual/compute` 86.85→95.30%.

  Approach inverted both pre-baked design assumptions:
  - **No new shared mock `EngineBackend`** — `CpuBackend` (via
    `cpu_engine_backend()`) already implements `coarse_*_with_state`
    when driven against the synthetic Q4K fixture
    (`make_test_q4k_weights` + `make_test_q4k_vindex`), so every
    dispatch happy-path tested end-to-end without new infrastructure.
  - **No `serial_test` crate** — env-gated paths
    (`LARQL_MARKOV_WALK_KV_*`, `LARQL_W10_DISABLE`) instead gained
    a per-thread `RefCell` override that production helpers consult
    *before* `std::env::var`. Tests inject without touching the
    process env; no race with other parallel tests. New helpers:
    `compute.rs::set_markov_env_override(...)`,
    `engines/mod.rs::set_w10_disabled_override(...)` (both
    `#[cfg(test)]` only).

  Test deltas: larql-kv lib 663 → 712 (+49). Zero regressions
  (5/5 successive `cargo test -p larql-kv --lib` runs green after
  the thread-local override fix; pre-fix the env-var-setting tests
  produced flaky `cold_kv.is_some()` failures in unrelated codec
  tests via process-env race). `make larql-kv-ci` passes end-to-end.

- **2026-05-24 — Accuracy harness honesty + FFN policy cross-product
  LANDED.** Multi-PR arc that turns the accuracy suite from "silent
  drop on engine miss" into a discriminating cross-product harness:

  - **Item 1 — accuracy schema fix** (commit `07684457`).
    `ScoreOutcome` enum (exhaustive, flat-tagged serde, mirrors the
    future `EngineError` taxonomy). `PromptScore` / `ConflictScore`
    gain `outcome` field + `Option<T>` score payload with
    `served()` / `skipped()` constructors enforcing
    correlated-optionality. `StrategySplit` gains `*_served` +
    `*_served_rate` per axis as required-companion fields to
    `*_match_rate`. `compute_strategy_split` filters on served subset
    (counting skips as zero would punish honest reporting). Replaces
    `filter_map` silent-drop in all three drivers. Surfaces Apollo's
    store-miss rows as `SkippedRetrievalMiss` instead of dropping.
    `EngineKind::supported_names()` replaces hard-coded six-engine
    error string at two bench sites.

  - **Item 2 v0 — `FfnBackendKind` parser + `FfnLayerPolicy`
    (in `larql-inference::ffn_policy/`).** New crate-shape:
    `FfnBackendKind` (Dense / Walk{k} / RemoteWalk / Null),
    `RoutingPredicate` (All / Layers / Otherwise), `FfnLayerPolicy`
    with from_spec parser supporting per-layer routing
    (`{walk:k=100}@layers=14-27;{dense}@otherwise`).
    Construction-errors on overlapping ranges; exhaustive enums;
    typed error taxonomy (`PolicyParseError` /
    `PolicyValidationError`). Module lives in `larql-inference` not
    `larql-kv` — FFN policy is the FFN axis, not the KV axis.

  - **`build_router` slice — `ValidatedFfnLayerPolicy` newtype +
    `BoundFfnRouter`.** Type-system enforcement of "validate before
    build" via non-public constructor. `BoundFfnRouter<'a>` owns its
    backend instances (`Vec<Box<dyn FfnBackend + 'a>>`) so callers
    don't manage backend lifetimes alongside the router's. `impl
    FfnBackend for BoundFfnRouter` delegates per-layer via the
    trait's existing `layer: usize` parameter — drop-in for the
    `&dyn FfnBackend` surface every engine already takes. Design
    rationale: `larql-inference/docs/ffn-build-router.md`.

  - **Cross-product harness + typed axis columns.** `accuracy_cmd`
    iterates `kv_engine × ffn_backend` cross-product via
    `FfnLayerPolicy::split_specs` (comma-separated, brace-aware,
    re-parse fallback for kv-comma forms like
    `remote-walk:endpoint=X,wire=Y`). New `EvalLabels<'a>` struct
    bundles `(kv_engine, ffn_backend, strategy)` for clean signatures.
    `PromptScore` / `ConflictScore` / `StrategySplit` gain explicit
    `kv_engine: String` + `ffn_backend: String` columns alongside
    `strategy`. `format_strategy_split` grows a two-axis layout
    (`KV engine` + `FFN backend` columns) when any row has
    `ffn_backend != "dense"`; default no-`--ffn` runs keep the
    historical single-`Strategy`-column layout. Closes the
    interim-`ffn_backend`-as-user-input limitation noted in Item 1's
    ROADMAP entry.

  - **CLI wiring.** `larql accuracy --ffn dense,walk:k=100,'{walk:k=100}@layers=14-27;{dense}@otherwise'`
    now runs the cross-product in one invocation. Vindex loaded
    lazily — only when a Walk binding is present.
    `larql bench --ffn-policy <spec>` honors the policy on the
    non-Q4K (CPU) path; Q4K path accepts the flag but doesn't
    honor it yet (P1 follow-on above).

  - **Apollo into accuracy default engines.** `--engines` default
    now includes `apollo`. The schema fix above means Apollo's
    store-miss rows show `served_rate < 1.0` rather than silent
    drops — diagnostic rather than misleading.

  - **Module splits.** `accuracy_suite/runner.rs` (2050 lines) split
    into `accuracy_suite/runner/` folder (6 files: `types` /
    `scoring` / `drivers` / `aggregate` / `legacy` / `mod`). Same
    pattern that produced the `ffn_policy/` folder split in
    `larql-inference`.

  - **Coverage lift across 5 engine files.** Pre-existing engine
    internals had drifted below 90%. Lifted with synthetic-weights
    + CPU-backend tests: `boundary_per_layer/cold_tier.rs`
    (88→100%), `executor.rs` (85→90.6%), `walk.rs` (84→95%),
    `engine.rs` (83→90%), `markov_residual/store.rs` (86→99.6%).
    `markov_residual/compute.rs` partially lifted (81→86.85%);
    full lift gated on `serial_test` for env-var paths.
    Discovered the gate had been passing against a stale JSON —
    fresh `make larql-kv-coverage-summary` is now required to
    surface debt. See "Coverage debt" section above for the
    remaining 6 files.

  Test deltas across the arc: larql-kv lib 595 → 663 (+68),
  larql-inference lib 1086 → 1102 (+16). Zero regressions. Clippy
  clean. Aggregate ~3,500 lines of code + tests added across
  `larql-kv` and `larql-inference`.

  ROADMAP entry for the sibling trait extraction (P0 above)
  references "Item 1 in the conversational priority queue" — Item 1
  is the schema fix above. Mode 5 work is still gated on that P0
  refactor landing.

- **2026-05-18 — W8.2 (doubling-capacity K/V in `markov_residual` +
  `markov_residual_codec`) LANDED: 2.4× decode speedup at 1000 tokens.**
  Lifted the W8 pre-allocation pattern from `windowed_checkpoint` to the
  two unbounded-window engines. Since `max_window=None` rules out a
  fixed pre-alloc, both stores now use a doubling-capacity strategy
  via three private helpers in each engine:
  - `window_capacity(prompt_len, window_size)` — initial cap is
    `max(window, prompt_len)` if windowed, else
    `max(prompt_len * 2, 64)`.
  - `grow_capacity_2d(src, len, cap)` — allocate `[cap, cols]` once
    at prefill, copy the prefill rows in.
  - `append_row(buf, row, len)` — in-place `slice_mut(s![len..len+1,
    ..]).assign(row)` when `len < cap`; otherwise double capacity,
    copy the live rows, then assign. Amortised O(1) per append vs the
    O(n) per step the previous `Array2::zeros((n+1, dim))` pattern
    paid.

  Store changes (both `RsStore` and `RsStoreCodec`):
  - New `pub hot_len: usize` field — logical row count, separate from
    `stored[l].shape()[0]` (which is now capacity ≥ hot_len).
  - `window_tokens()`, `memory_bytes()`, `clip_layer` /
    `clip_layer_overflow` updated to use `hot_len`.
  - New `finalise_hot_len_after_clip()` — must be called after every
    per-layer clip loop. (Subtle bug fix during impl: setting
    `hot_len = window` *inside* the per-layer loop made layers 2..N
    see `rows == window` and skip their clips, dropping half the
    cold-tier payload. Two existing tests caught this.)

  Bench (Gemma 3 4B Q4K, Metal, M3 Max):
  - **1000-tok**:
    - `markov-rs`: 24.8 → **58.7 tok/s (+137%)**
    - `markov-rs-codec`: 25.7 → **57.2 tok/s (+123%)**
    - `windowed-checkpoint`: 49.5 → **57.4 tok/s (+16%)** (variance
      recovery from previous run + sympathy from the codepath audit)
    - `standard` unchanged at 64.1 (untouched)
  - **50-tok**:
    - `markov-rs`: 77.1 → **88.9 tok/s (+15%)**
    - `markov-rs-codec`: 77.5 → **88.8 tok/s (+15%)**

  All three cached-state engines now cluster within 11% of standard's
  64.1 tok/s ceiling at 1000 tokens. The doubling-capacity scales
  linearly with seq_len: at 50 tok the saved alloc bytes are small
  (~400 KB/step); at 1000 tok they're ~8 MB/step. The 137% win at
  long context is the alloc churn that pre-W8.2 was hiding behind
  prefill cost.

  CPU walk + executor fallback paths (`rs_decode_step_walk`,
  `rs_decode_step_codec_walk`, `process_via_executor`) still allocate
  per step — they're not on the hot path for the bench. Defensive
  consistency: every legacy RsStore/RsStoreCodec constructor sets
  `hot_len` from `stored[0].shape()[0]` so non-dispatch paths see a
  consistent invariant.

- **2026-05-18 — Step 9 (iterative Metal `coarse_prefill_with_state`)
  LANDED: ~10× prefill speedup on every state-dump engine.**
  Pre-Step 9, `MetalBackend::coarse_prefill_with_state` defaulted to
  the trait's `coarse_prefill` (no per-layer state dump); engines saw
  `state.is_complete_for() == false` and fell back to the CPU walk
  (~2.7 s on Gemma 3 4B). The new impl pre-allocates `[seq_len,
  hidden]` and `[seq_len, kv_dim]` per layer (W8-style alloc at
  source for prefill too), resets + preallocates the Metal K/V cache,
  then iterates `fused_decode_step_with_state` per prefill token,
  writing the dump into the pre-allocated row position.

  Bench (Gemma 3 4B Q4K, Metal, M3 Max, "The capital of France is",
  5 prefill tokens):
  - `markov-rs` prefill: 2757 → **254 ms** (10.9×)
  - `markov-rs-codec` prefill: 2564 → **249 ms** (10.3×)
  - `windowed-checkpoint` prefill: 2760 → **256 ms** (10.8×)
  - `turbo-quant` prefill: 2750 → **334 ms** (8.2×)

  Predicted ~45× (5 × 12 ms decode time) didn't materialise because
  each iterative `fused_decode_step_with_state` carries per-token
  state-dump readback overhead. Remaining ~250 ms is 5 × ~50 ms
  per-iter + fixed setup. Further closure needs a single-kernel
  prefill that dumps state for all positions in one shot — separate
  Metal-kernel surgery.

  Decode steady-state also moved (W8 + Step 9 compound):
  - `windowed-checkpoint`: 82.7 → **89.2 tok/s** (fastest cached-state
    engine; within 10% of `standard`'s 99.2 ceiling)
  - `markov-rs`: 75.3 → 77.1 tok/s
  - `markov-rs-codec`: 79.0 → 77.5 tok/s

- **2026-05-18 — W8 (pre-allocated K/V buffer in `windowed_checkpoint`)
  LANDED: 58% of decode-CPU alloc churn removed.**
  samply flamegraph on `windowed_checkpoint:window=1024 --tokens 1000`
  (post-W7) surfaced an unexpected hot path: 21% `__bzero` + 19%
  `ndarray::zip_mut_with_same_shape` + 18% `madvise` = **58.5% of
  main-thread CPU** spent on `Array2::<f32>::zeros((n+1, kv_dim))` +
  `slice_mut().assign(k_old)` + `slice_mut().assign(k_new_row)`
  inside `decode_step_via_dispatch` — 68 allocations per token
  (34 layers × 2), each growing linearly with `n`.

  Fix: pre-allocate `Array2::zeros((window_size, kv_dim))` per layer
  once at prefill (in `try_prefill_via_dispatch`), track a single
  `current_window_kv_len: usize` counter, and append in the hot path
  via `slot.0.slice_mut(s![pos..pos+1, ..]).assign(k_new_row)`. One
  small `kv_dim`-sized copy per layer per side, zero alloc per step.
  Readers (`close_window`, `current_kv_bytes`) updated to use the
  counter instead of `k.shape()[0]`; CPU walk fallback paths set the
  counter defensively from the returned narrow-array shape.

  Bench (Gemma 3 4B Q4K, Metal, M3 Max):
  - 50-tok: `windowed-checkpoint:window=256` 82.7 → **86.6 tok/s
    (+4.7%)** vs `standard`'s 99.4 (gap closed ~50%)
  - 1000-tok: `windowed-checkpoint:window=1024` 17.39 ms vs `standard`'s
    15.74 ms → 1.65 ms gap (vs pre-W8 estimated 5-10 ms slope from
    `Array2::zeros((n+1, …))` growing linearly with `n`)

  Post-W8 flamegraph: the `__bzero` / `zip_mut_with_same_shape` /
  `madvise` triple is **gone from the top-20**. Remaining main-thread
  CPU is dominated by `__psynch_cvwait` (Metal GPU wait,
  irreducible), `synthesize_lm_head_kquant` (prefill — separate
  ~2.5 s regression flagged elsewhere), and generic `Map::fold`.

  The optimisation is engine-local (`larql-kv/src/engines/windowed_checkpoint/engine.rs`)
  with no surface change. Same pattern can be lifted to
  `markov_residual` / `markov_residual_codec` / `turbo_quant` once
  their state-policy shape is clarified — they use the same
  `Array2::zeros((n+1, kv_dim))` pattern but have unbounded windows
  by default, so the pre-allocation needs a growable strategy
  (doubling-capacity Vec-style) rather than fixed window size.
  Tracked as W8.2 candidate.

- **2026-05-18 — W7 (blit-encoder fusion) LANDED: per-layer commit
  overhead removed; +30-48% across cached-state engines.**
  Modified `decode_token_with_moe_split_fn` in
  `larql-compute-metal/src/decode/mod.rs` to pre-allocate per-layer
  staging buffers (k / v / h-in) when `state_dump` is `Some`. The
  layer loop blits `k_out` / `v_out` / `h_buf` into the staging
  buffers inside the same command buffer (`new_blit_command_encoder`
  + `copy_from_buffer`) instead of forcing per-layer commit + wait +
  CPU read. The single final commit at the bottom of the function
  flushes everything; reads happen once after that, draining staging
  into `state_dump`. Metal's command-buffer encode ordering
  guarantees blit reads see the settled compute writes.

  Measured (Gemma 3 4B Q4K, Metal, M3 Max):
  - `standard` (control, no state_dump): 105.9 → 99.4 tok/s (noise)
  - `markov-rs`: 58.0 → **75.3 tok/s (+30%)**
  - `markov-rs-codec`: 58.4 → **79.0 tok/s (+35%)**
  - `windowed-checkpoint` (window=256): 56.0 → **82.7 tok/s (+48%)**
  - `turbo-quant` (4-bit, 10-tok bench): 33.0 → **37.7 tok/s (+14%)**

  Engine-cost decomposition post-W7: ~10 ms Metal kernel compute +
  ~3 ms CPU glue. The remaining gap to `standard`'s 99 tok/s is
  pure CPU-side state-update work (state Vec→Array2 conversion,
  appends). Closure path: in-place state updates / pre-allocated
  buffers (W8 candidate).

  Edge cases worth noting:
  - `standard` doesn't touch state_dump → blit branch is dead code
    → 0× regression confirmed.
  - `turbo_quant`'s codec inner loop is the dominant per-token cost;
    the saved 1.7 ms commit overhead is a smaller fraction.
  - The `windowed_checkpoint` +48% win reflects its lighter post-
    kernel CPU work (just append to `current_window_kv`); engines
    with heavier post-kernel work see smaller relative gains.

- **2026-05-17 night — W1-GPU steps 4 + 6 LANDED: windowed_checkpoint +
  turbo_quant now route through dispatch on Metal.**
  Same pattern as steps 5: each engine gains `try_prefill_via_dispatch`
  / `decode_step_via_dispatch` helpers that read per-layer captured
  state and update engine-specific state policy.
  - **turbo_quant**: state.k_new/v_new per layer feeds the
    WHT+Lloyd-Max codec via `CompressedLayer::compress` (prefill)
    and decompress→append→recompress (decode). Bench: **19.6 →
    33.0 tok/s (+68%)** on Metal. Memory stays at 0.6 MB hot
    (compression intact).
  - **windowed_checkpoint**: state.k_new/v_new appends to
    `current_window_kv` per layer; window auto-close at
    `window_size` tokens fires the legacy `close_window` checkpoint
    emit. Bench: **28 → 56.0 tok/s on Metal (+98%)** at
    `window=256` (Gemma 3 4B, M3 Max, 50-token decode). Hot state
    15.7 MB tracks the engine-side window shadow (see KvHandle
    eviction note below).

  Engine memory note: with W1-GPU active, the backend's internal K/V
  cache grows unboundedly alongside each engine's shadow state. This
  defeats the memory benefit of `windowed_checkpoint` /
  `markov_residual_codec` at long contexts. Follow-up: expose a
  `KvHandle::evict_oldest(n)` API on `KvDispatch` so engines can
  bound the backend cache to match their window.
- **2026-05-17 night — W1-GPU step 2 LANDED: Metal per-layer state
  dump → 2.1× decode speedup on markov-rs + codec.**
  Modified `decode_token_with_moe_split_fn` in
  `larql-compute-metal/src/decode/mod.rs` to accept an optional
  `state_dump: Option<&mut DecodeStateDump>` parameter. When active,
  the layer loop:
  1. At top of layer L: pushes `x` (for L=0) or reads `h_buf` (for
     L>0, settled by the previous layer's commit) into
     `state.h_in_per_layer`.
  2. At bottom of layer L: forces `enc.end_encoding()`, `cmd.commit()`,
     `wait_until_completed()`, reads `k_out` / `v_out` (scratch
     buffers reused across layers) into
     `state.k_new_per_layer` / `v_new_per_layer`, then restarts
     command buffer + encoder for the next layer.

  Trait wiring: new `DecodeBackend::decode_token_with_state_dump`
  method (default falls back to plain `decode_token`); MetalBackend's
  trait impl routes through the new kernel function when `state` is
  `Some`. Inference layer adds `fused_decode_step_with_state` +
  `MetalBackend::coarse_decode_step_with_state` /
  `coarse_prefill_with_state`. Engines (markov_residual, codec)
  inherit the Metal acceleration automatically — no engine-side
  changes from step 5.

  Measured (Gemma 3 4B Q4K, Metal, M3 Max, 10-token decode):
  - `markov-rs`: 27.0 → **57.7 tok/s** (+114%)
  - `markov-rs-codec`: 27.8 → **57.5 tok/s** (+107%)
  - `standard` (fused control): 100.8 tok/s (unchanged)

  Per-token cost: ~17 ms = 10 ms Metal compute + ~1.7 ms commit
  overhead (50 µs × 34 layers) + ~5 ms engine state update / CPU
  glue. The remaining gap to standard's 100 tok/s is the
  per-layer commit cost; a follow-up could use blit-encoder
  switches inside a single command buffer to eliminate the
  commit overhead and lift toward 80-100 tok/s.

  Prefill cost: ~2.8 s on Metal (CPU walk for state seeding +
  Metal `fused_prefill` for backend cache). One-shot; doesn't
  affect decode steady-state. Future optimisation: per-position
  per-layer K/V dump on the Metal prefill side to skip CPU walk.
- **2026-05-17 night — W1-GPU infrastructure (decode trait surface +
  CPU impl + engine wiring; Metal kernel modification deferred).**
  Three layered changes landed end-to-end:
  - **Trait surface (`KvDispatch`):** new `coarse_prefill_with_state` /
    `coarse_decode_step_with_state` methods take
    `Option<&mut PerLayerDecodeState>`. Default impls delegate to the
    non-state variants, so unmigrated backends keep working.
  - **`DecodeBackend` trait + `DecodeStateDump` struct** added in
    `larql-compute` for the substrate-level surface. Same default-
    delegation pattern.
  - **CPU implementation** (`predict_kquant_prefill_with_state` /
    `predict_kquant_decode_step_direct_with_state`): threads per-layer
    state capture through the existing per-layer walk at zero
    re-compute cost. Parity test in
    `kv_dispatch::cpu::coarse_decode_step_with_state_populates_and_matches_plain`
    asserts cached and non-cached outputs match within f32 rounding
    and per-layer shapes (`[1, hidden]`, `[1, kv_dim]`) are correct.
  - **Engine wiring** for `markov_residual` and
    `markov_residual_codec`: `try_prefill_via_dispatch` /
    `decode_step_via_dispatch` route through the new
    `coarse_*_with_state` API when the backend implements it. State
    capture feeds `RsStore::stored` (residuals) and `hot_kv` (W2
    cache) in a single backend call. Legacy walk path stays as the
    fallback when state isn't populated (e.g. on backends that
    haven't migrated yet — currently `MetalBackend`). Gated on
    `supports_direct_matvec_decode` so non-Q4K test fixtures skip
    the dispatch path. 113 markov tests pass.
  - **CPU bench numbers stay parity** post-W1-GPU step 5:
    markov-rs 27.4 tok/s, codec 26.6 tok/s — same as W2 (W1-GPU on
    CPU just changes the code path, not the compute; CPU was already
    at the M3 Max compute ceiling).

  **What's NOT done**: `MetalBackend::coarse_*_with_state` still uses
  the default delegation (state stays empty), so engine falls back
  to walk on Metal — no GPU speedup yet. The real Metal acceleration
  requires modifying
  `larql-compute-metal::decode::decode_token_with_moe_split_fn`
  (200+ lines) to thread per-layer dump buffers + blit-encode steps
  into the existing single command buffer. Two implementation
  shapes have been scoped:
  1. **Blit-encoder switches per layer**: cheapest in steady-state
     (~tens of µs per layer); requires careful encoder lifecycle
     management within the existing kernel function.
  2. **Per-layer commit + CPU readback**: simpler (mirror the
     existing `stage_timing_split` pattern); costs ~50µs/layer ×
     34 = ~1.7ms/token overhead. Projected ceiling: 50-80 tok/s
     (vs CPU's 27 tok/s ceiling, vs `standard`'s 102 tok/s fused).

  Choice between shapes is open. The trait surface, CPU impl, and
  engine wiring are all stable and don't change regardless of which
  Metal-side approach lands.
- **2026-05-17 night — W2: hot K/V cache for `markov_residual` and
  `markov_residual_codec`.** Added `hot_kv: Option<Vec<SharedKV>>`
  to both `RsStore` and `RsStoreCodec`; prefill captures K/V from
  the per-layer forward pass (previously discarded) and stashes it;
  decode appends one row per layer via the existing
  `run_attention_block_decode_step_backend` return tuple. On
  window-overflow `clip_layer` slices `hot_kv` consistently with
  `stored`; for `markov_residual` (lossless cold tier) the evicted
  K/V rows merge directly into `cold_kv` (no `recompute_kv` call
  needed); for `markov_residual_codec` (lossy bf16 cold tier)
  `cold_kv` is invalidated on overflow so the next step recomputes
  against the codec-decoded residual. Bench: `markov_residual`
  4.7 → 26.8 tok/s (5.7×); `markov_residual_codec` 5.0 → 27.5 tok/s
  (5.5×). Both now sit on the `windowed_checkpoint` curve. Engine
  contract preserved — drop `hot_kv` and the next step recomputes
  from `stored` (via_executor path takes this fallback). Hot-state
  memory grew from 5.3 → 10.8 MB; still ~50× smaller than
  `standard`'s full KV cache. Parity test
  (`decode_step_quant_w2_cached_matches_recompute_from_residuals`)
  asserts the cached and recompute paths agree within fp rounding.
- **2026-05-17 night — W7: per-engine profiler wired on the quant
  path.** `EngineProfiler` now populates from `rs_decode_step_walk`
  (markov_residual), `rs_decode_step_codec_walk`
  (markov_residual_codec), `rs_extend_from_checkpoint_quant`
  (windowed_checkpoint), and `decode_step_quant_cpu` (turbo_quant).
  Each engine's `stage_summary()` returns `Some(...)` when
  `with_profiling(true)` is set. `larql bench --profile --engine
  <name>` now produces a per-stage attribution table per engine.
  First measurement run produced the bottleneck-diagnosis table in
  the P0 section above, which inverted two of the pre-profile
  guesses: codec overhead in turbo_quant was ~25% not ~80%, and K/V
  recompute (W2 target) was the dominant cost on markov_residual
  (~80%) not dispatch (W1 target). Sequencing in P0 revised
  accordingly.
- **2026-05-17 night — `_q4k` → `_quant` on remaining internal
  function names.** The trait-surface renames earlier today
  (`prefill_q4k` → `prefill_quant`, `has_q4` →
  `supports_quant(format)`, `q4k` → `kquant` storage) missed the
  per-engine implementation wrappers:
  `windowed_checkpoint::process_q4k`,
  `windowed_checkpoint::extend_current_q4k`,
  `extend::rs_extend_from_checkpoint_q4k`,
  `turbo_quant::decode_step_q4k_cpu` /
  `turbo_quant::prefill_kquant_cpu`. All renamed to `_quant` since
  they dispatch on whatever format the vindex carries, not Q4_K
  specifically.
- **2026-05-17 night — Fused-bypass strip: engines are now engines.**
  Every per-layer engine (`markov_residual`, `markov_residual_codec`,
  `windowed_checkpoint`, `turbo_quant`) had a hidden
  `if let Some(h) = fused_prefill(...) { return Some(h); }` short-
  circuit at the top of `prefill_quant` / `decode_step_quant`. The
  short-circuit meant `--engine markov-rs` on Metal silently ran
  `StandardEngine`'s fused kernel instead — five engines tied at
  ~103 tok/s with `hot=0.0MB`, masking every state-policy difference
  and making per-layer optimization invisible. Cut: removed every
  short-circuit; deleted dead `metal_prefill_done` + `force_walk`
  fields and `with_force_walk` builders; dropped the pub(crate)
  `fused_prefill`/`fused_decode_step` re-exports from
  `windowed_checkpoint::engine` (only `StandardEngine::coarse_prefill`
  uses the underlying `larql_inference::vindex::fused_prefill` now,
  via `ComputeBackend::coarse_prefill`). `StandardEngine` remains the
  default engine and the only home of the fused fast path. Bench now
  reports honest numbers: standard 104 tok/s, markov-rs 3.6, codec
  4.3, windowed-checkpoint 25.6, turbo-quant 3.9 — every per-layer
  engine reports non-zero `hot=` memory because their state
  structures actually materialise. The 25-30× standard-vs-per-layer
  gap is the new optimization frontier; previously it was invisible
  because every engine was running the same kernel under different
  labels.
- **2026-05-17 evening — Phase-2 migration completed for the remaining
  three engines.** `windowed_checkpoint`, `turbo_quant`, and `apollo` all
  override `*_via_executor` methods and honor the caller-supplied
  `FfnBackend`. `CountingFfn` stub tests prove per-(token, layer)
  dispatch through the caller's backend. Same push cleared every
  `coverage-policy.json` debt baseline: all 43 files in src/ at ≥90%
  lines, workspace total 95.55%. `larql bench --ffn http://shard:8080`
  now routes through the remote shard for every per-layer engine
  instead of silently constructing a local `WalkFfn`.
- **2026-05-17 — Phase 2 engine migration to `LayerExecutor`.** Four
  engines (`markov_residual`, `markov_residual_codec`,
  `boundary_per_layer`, `no_cache`) override `*_via_executor` methods.
  They drive per-layer dispatch through `executor.run_*_layer` and
  honor the caller's `FfnBackend`. `CountingFfn` stub tests prove the
  FFN parameter is no longer silently ignored. Bench has
  `--via-executor` flag; demoed on Gemma 3 4B Q4K showing the codec
  engine's 50% cold tier saving (22.9 MB → 11.5 MB).
- **2026-05-17 — `LayerExecutor` trait + `LocalWalkExecutor`.** New
  abstraction in `larql-inference::layer_executor` separating state
  policy (engine concern) from execution strategy (executor concern).
  Spec at
  [engine-state-vs-execution.md](../larql-inference/docs/specs/engine-state-vs-execution.md).
- **2026-05-17 — `q4k` → `kquant` storage rename.** K-family storage
  slots (`attn_q4k`, `interleaved_q4k`, manifests, setters, loaders)
  renamed for consistency with accessor naming (`attn_kquant_layer_data`).
  Q4_0 and Q8 slots unchanged. ~60 sites touched.
- **2026-05-17 — `has_q4()` → `supports_quant(format)`.** Per-format
  predicate on `ComputeBackend`. 79 call sites migrated to
  `supports_quant(QuantFormat::Q4_K)`. Enables future Q6_K / FP4
  fused-pipeline backends without trait extension.
- **2026-05-17 — `KvEngine::prefill_q4k` / `decode_step_q4k` →
  `prefill_quant` / `decode_step_quant`.** Trait surface naming made
  quant-agnostic. 112 sites updated. Internals that are genuinely
  Q4K-specific kept their names.
- **2026-05-17 — `metal_fused_*` → `fused_*` rename.** The "metal"
  prefix was a lie: `CpuBackend` implements `prefill_q4` and
  `decode_token` via its C Q4 kernel. Aliases in
  `windowed_checkpoint::engine` follow.
- **2026-05-17 — `BoundaryKvEngine`, `MarkovResidualCodecEngine`,
  `BoundaryPerLayerEngine` shipped.** All three new engines have
  contracts in `crates/larql-inference/docs/specs/`. Per-file coverage
  ≥94 % lines on every new file. Bench demoed end-to-end on Gemma 3 4B,
  Gemma 4 E2B, 26B-A4B, 31B, Qwen3 0.6B (dense + Q4K).
- **2026-05-09 — Initial extraction.** `engines/` carved out of
  `larql-inference` into the new `larql-kv` crate. ~5,540 LOC moved with
  no semantic changes. All four engines + `KvEngine` + accuracy /
  profiler helpers now ship from this crate.
