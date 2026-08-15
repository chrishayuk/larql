# BW10 — the movement/causality ledger, and the BW-A live gate

**Date:** 2026-08-14 · **Box:** M3 Max, AC power
**Instruments:** `larql_compute::movement_ledger` (new) · `gpt-oss-20b-q4k.vindex`
+ `gpt-oss-20b-experts-mxfp4.v3` (native MXFP4 routed banks)

## Why

Every earlier bandwidth experiment in this programme reported bytes/token
in isolation and argued latency reduction from it by inference. That is
the exact shape of the mistake `docs/k3-funnel.md` §4.11 committed: 11.5ms
of MoE decode called "genuine bandwidth" while the path sat at 62% GPU
occupancy and larql was reading *fewer* bytes than a faster engine. Bytes
are diagnostic, not the optimisation target — a byte reduction only buys
latency to the extent byte movement was on the critical path, and that
extent is exactly what a bytes-only ledger cannot show.

BW10 is the instrument that makes the inference checkable instead of
assumed. It reports raw byte arms (semantic / physical / useful bytes,
DRAM / NVMe / network tiers) alongside raw time arms (GPU-busy / bubble /
host-wait / io-wait), then joins them against a declared roofline so a
byte reduction can only claim the latency it actually accounts for.

## What it measures

- **`ByteMovement`** — semantic (what the plan logically asked for) vs
  physical (what was actually bound and streamed, block-padded) vs useful
  (of the physical bytes, those that reached the result). Tier-attributed
  (DRAM / NVMe / network); reuse and prefetch counters distinguish a
  *measured* zero from *nobody reporting*.
- **`TimeAttribution`** — `wall = gpu_busy + gpu_bubble + host_outside_gpu`
  as an explicit, non-additive decomposition. `host_wait` and `io_wait`
  are reported "of which", never summed into the total — they overlap
  `gpu_busy` by construction, and a decomposition that silently double-
  counts is worse than none.
- **`Regime`** — `resident | capacity-constrained | cold-estate`. The
  verdict line **refuses to render** without one declared, because the
  identical byte delta licenses opposite conclusions across regimes.
- **`coverage`** — per weight-surface instrumented / silent / missing
  state. Only `moe-experts` is wired so far; physical totals are a floor
  until the rest are, but arm-to-arm byte *deltas* on a covered surface
  remain valid — which is what every number below actually relies on.

## The defect the live gate caught — and the invariant that closes it

The first live run against `gpt-oss-20b-q4k.vindex` (`-n 1 --warmup 0` —
a window that requests **zero** decode-loop iterations, `for step in
1..1` never runs) recorded **130 tokens**. Root cause: GPT-OSS-20B's
harmony chat template expands even a short prompt to ~130 prefill
positions, and `larql-inference`'s per-position MoE prefill walk
(`layer_graph/generate/gpu/prefill.rs`) reuses the exact same GPU entry
point (`MetalBackend::decode_token_with_moe_split_fn`) that real
autoregressive decode steps use. Every prefill position was silently
entering what should have been a pure decode steady-state mean — the
ledger was measuring the model's chat-template preamble and reporting it
as decode throughput.

Fixed with a thread-local `Phase` / `PhaseScope`
(`movement_ledger::phase`), mirroring `moe_route_observe::LayerScope`'s
existing precedent: attach at the semantic boundary (the prefill walk and
the decode loop, both inside `layer_graph::generate`) rather than
threading a parameter through every intermediate call signature. Rerun of
the same control after the fix: **0 decode tokens, 0 unattributed,
130/130 prefill positions correctly excluded.**

> **Invariant:** a `TokenRecord` with `phase = Some(Phase::Prefill)` or
> `phase = None` (no `PhaseScope` was active) can never enter
> `SteadyState`'s decode accumulator, its `counted()`/`discarded()`
> warmup bookkeeping, or its decode `mean()`. Prefill tokens accumulate
> in their own bucket (`prefill_mean()`, no warmup discard — a run's
> prefill is one window, not a steady-state series); unattributed tokens
> are counted and reported (`unattributed()`) but averaged into neither
> bucket, refused rather than guessed. This is enforced structurally by
> `SteadyState::push`'s match on `rec.phase` (`crates/larql-compute/src/
> movement_ledger/mod.rs`), not by convention, and pinned by
> `prefill_tokens_are_excluded_from_the_decode_mean` and
> `unattributed_tokens_enter_neither_mean`
> (`movement_ledger/tests/mod_api.rs`).

## BW-A: the sanity gate

> Can the ledger correctly diagnose two deliberately opposite, known
> interventions?

Unit half: synthetic calibration fixtures in `movement_ledger/tests/
timing.rs` (`calibration_s2_is_a_scheduling_win_with_no_byte_delta`,
`calibration_mxfp4_is_a_byte_win_that_underdelivers_on_latency`,
`the_two_calibration_shapes_have_opposite_signatures`) pin the two
expected shapes against the project's banked historical numbers.

Live half: `larql bench gpt-oss-20b-q4k.vindex --warmup 16 -n 256`, prompt
long enough to avoid early EOS (255/256 steps measured both arms):

### S2 — GPU-dataflow routing removes the queue-starvation bubble

`LARQL_GPU_ROUTE` 0 → 1, identical expert weights (no `--routed-from`):

| | physical bytes | wall (mean) | tok/s | bubble | host_wait | gpu_occupancy | η |
|---|---:|---:|---:|---:|---:|---:|---:|
| control (`GPU_ROUTE=0`) | 2.090 GB | 19.94 ms | 50.2 | 6.446 ms | 16.457 ms | 0.634 | 0.492 |
| candidate (`GPU_ROUTE=1`) | 2.090 GB | 14.20 ms | 70.4 | 0.000 ms | 0.000 ms | 0.937 | 0.484 |

Physical bytes **identical** — the ledger cannot credit movement — yet
wall time drops 1.40×. The bubble and host-wait terms collapse to zero
together with the win, and gpu_occupancy rises from 63% to 94%: a clean
scheduling win, correctly attributed to the term that actually moved.

### MXFP4 — a real byte win that underdelivers on latency, and the ledger says why

`LARQL_GPU_ROUTE=1` held constant, `--routed-from` native MXFP4 vs the
served Q4K container:

| | physical bytes | wall (mean) | tok/s | η |
|---|---:|---:|---:|---:|
| control (Q4K) | 2.090 GB | 14.20 ms | 70.4 | 0.484 |
| candidate (native MXFP4) | 1.269 GB | 12.12 ms | 82.5 | 0.364 |

Bytes fall 39.3%, wall falls only 14.7% (1.17×) — byte_cut/wall_cut ≈
2.7×. The roofline utilisation η drops 0.484 → 0.364 in the same window:
the MXFP4 kernel streams *less* efficiently against the declared DRAM
roofline than the Q4K kernel does, which is *why* the byte saving doesn't
convert 1:1 into latency. The ledger doesn't just observe the shortfall —
it names the mechanism.

**Verdict:** BW-A is CLOSED. The instrument produces the two
deliberately-opposite signatures the gate demands, on live hardware, and
in the MXFP4 case explains the gap rather than merely reporting it.

## What this licenses, and what it doesn't

This closes the §4.11 conceptual error for future bandwidth work on this
engine: a byte reduction proposal can now be priced against the measured
movement share of the window (`MovementCost::share_of_wall`,
`gpu_busy_share`) instead of assumed proportional to bytes saved. It does
**not** yet cover attention/dense-FFN/lm-head/KV-cache/embeddings/norms
(`coverage::Surface` — only `moe-experts` is wired), so absolute physical
totals remain a floor; only byte *deltas* confined to the covered surface
are licensed, which both arms above are.

## Not yet done

- **BW-B** — CLOSED 2026-08-14, see `docs/diagnoses/bwb-compact-dense-oracle.md`:
  a compiled compact-dense oracle beats both the R4-style sparse-gather
  kernel and dense across the whole tested range (up to 66.7% retained);
  the problem was gather-timing, not sparsity.
- **BW-C** — CLOSED (first pass) 2026-08-14, see
  `docs/diagnoses/bwc-expert-skip-oracle.md`: on 8 real oracle
  single-expert ablations, 4/8 left a 16-token greedy continuation
  byte-identical; 2/8 diverged immediately. Caught a real dispatch-path
  bug en route (the hook's first version silently observed zero calls
  because GPT-OSS's top-4-of-N/8-thread routing bypasses the `add_expert`
  closure entirely).
- **BW-D** — permutation-aligned expert redundancy (cheap, low prior,
  ETC-0A left it open).
- **BW-E** — residency horizon measured against this ledger (external
  bytes avoided per byte of resident capacity, for the cold-estate K3
  regime).
