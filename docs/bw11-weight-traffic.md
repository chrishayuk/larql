# BW11 — Dynamic Weight-Traffic Programme

**Estate:** larql (VINDEX3, Metal MoE kernels, DEC funnel's union-byte accounting)
**Status:** BW11-1 CLOSED and BANKED 2026-08-20 (real, controlled, modest — 2.40× logical ceiling at K=8,
not large enough to justify the grouped-expert kernel now). Programme priority moves to BW12 — `bw12-static-subexpert-repack.md` is a
planned follow-up specification, not a tracked document in this PR.
**Predecessor context:** [`dec-funnel.md`](dec-funnel.md) (the R0–R14 standing rules this doc
inherits wholesale rather than restating), [BW10 movement/causality ledger] and [BW-C expert-skip
oracle] (both closed 2026-08-14, different questions — see naming note below).

---

## 1. Thesis

On M3 Max, attainable GPU read bandwidth is ~367 GB/s against a 400 GB/s SoC spec (`c0-bandwidth-roofline`,
dec-funnel.md §C0). Once a kernel sits near that slope, no further kernel cleverness buys another 2× — the
only lever left is **fewer physical weight bytes per accepted output token**. That is a denominator problem,
not a numerator problem, and it decomposes into a small number of independently falsifiable questions.

**Revised priority queue (2026-08-20, after BW11-1's result was banked):**

1. **BW11 — multi-token weight reuse — BANKED, not pursued further for now.** Real (survives the R9
   shuffled-marginal control) but modest: 1.28×/1.72×/2.40× at K=2/4/8. R6: a logical ceiling, not a kernel
   result — and this repo's own R4 precedent (§3.5 below) says a realised kernel typically captures only
   23–40% of a theoretical ceiling this size, so the deployable number would likely round down to "not
   worth it" before engineering even starts. Revisit only if BW12/BW13 raise the stakes enough to make a
   compounding case.
2. **BW12 — sub-expert structural sparsity — ACTIVE, promoted ahead of progressive precision.** Original
   framing (oracle-rank blocks by contribution, sweep retained fractions) was shelved before writing any
   code: it closely repeats two already-run, mostly-refuted experiments (`R4 zero-out`, `MoE latent-axis
   sparsity` — both found fine-grained selection real but any physically-realisable block size ≥16 collapses
   fidelity, and R13 flags rank-by-marginal-contribution as unreliable evidence of removability on its own).
   Reframed as **BW12-0: static sub-expert repackability** — the one question both prior threads left open
   and called "the only live version of the thesis": is there a temporally- AND globally-stable block subset
   per expert, stable enough across arbitrary tokens to compile a smaller static matrix with zero dynamic
   gather? See the dedicated doc.
3. **BW13 — progressive precision (coarse+correction planes)** — queued after BW12, ideally applied to
   whatever block structure BW12-0 exposes (if any) rather than independently.
4. **BW14 — shared-basis experts** — queued last, unless BW12 turns up something dominant enough to
   reorder the queue.

Original numbering before the revision, kept for the historical record:

1. **Multi-token weight reuse** (BW11-1, this note) — if K token positions are evaluated together, how much
   do their routed-expert selections overlap? A large overlap is a *ceiling*, not a win: DEC's own roadmap
   already names the missing piece (§DEC-2: "build the expert-grouped scheduler... before quoting
   tier-capacity numbers") — no kernel in this codebase groups multiple tokens through one expert's weight
   read today (every `mxfp4_grouped_experts` / `q4k_grouped_experts` / `q6k_grouped_experts` dispatch is
   K=1 token, confirmed by code inspection 2026-08-20).
2. **Micro-expert / sub-FFN block sparsity** — not started. Gated on (1): if whole-expert reuse is small,
   sub-expert granularity is the next place to look; if (1) is large, realising it is the higher-value target
   first.
3. **Progressive precision (coarse+correction planes)** — not started.
4. **Shared-basis experts** — not started.

Only (1) is active. The other three are recorded here as the programme's stated shape, not as commitments —
each needs its own falsification gate before any code is written for it, per this workspace's engineering
vs. research posture (`feedback_engineering_vs_research_posture`).

## 2. Naming note

An earlier draft of this program used "BW-C1" for the multi-token reuse question. That collides with the
**closed, unrelated** `BW-C expert-skip oracle` programme (C1–C5, closed 2026-08-14 — router-weight /
contribution-norm expert-skipping, not weight reuse). This program uses **BW11** (continuing on from BW10)
to keep the two ledgers unambiguous.

## 3. BW11-1 — same-sequence consecutive-position expert-union ceiling

### 3.1 The question, precisely

DEC-0 already measured an expert-union ceiling: at batch 64, unique-expert bytes were 13.9% of naive bytes
on Gemma-4 26B-A4B (`experts_union_frac`, `dec-funnel.md` §DEC-0 Result note) — a ~7.2× amortisation
*if a scheduler realised it*. **That number answers a different question than this one, on every axis:**

| | DEC-0's `experts_union_frac` | BW11-1 |
|---|---|---|
| Union axis | across **B independent, unrelated prompts** at one fixed step (R10: "a domain mixture") | across **K consecutive positions of ONE sequence** |
| Object modelled | a *serving batch* (many users' requests landing on one step) | a *speculative-verification batch* (one user's K draft tokens, evaluated together) |
| Model / shape | Gemma-4 26B-A4B, 8-of-128 experts (6.25% activation) | gpt-oss-20b, **4-of-32 experts (12.5% activation)** — code-verified (`GptOssArch`), not assumed |

R2 is explicit that an expert-union/amortisation ratio does not transfer across activation fractions
(K3's 1.79% activation reads 3.0× at the same batch where Gemma-4's 6.25% reads 7.2×) — and this changes
the *axis* as well as the activation fraction, so DEC-0's number cannot be reused here even as a prior in
the same units. BW11-1 measures gpt-oss's consecutive-position union fresh.

### 3.2 Method

**No new GPU code.** The capture side reuses existing, already-parity-tested infrastructure end to end:

- `larql run <spine>.vindex --routed-from <bank>.v3 --metal -n N "<prompt>"` — the served V2 CPU-route
  decode path (`decode/moe_interleave.rs`, engaged whenever `LARQL_GPU_ROUTE` is unset — confirmed live,
  `gpu_route_layers=0` in the run witness). This path calls `moe_route_from_router_input` (CPU) inside a
  `moe_route_observe::LayerScope`, per layer per position.
- `LARQL_MOE_ROUTE_TRACE=<path.jsonl>` — makes every routed selection append one JSON line:
  `{"layer":L,"seq":S,"experts":[[e0..e_{k-1}]]}`. `seq` is that layer's call index from 0, so **consecutive
  `seq` values are consecutive positions of one continuous sequence** (prefill positions then decode steps,
  in call order) — exactly the object BW11-1 needs. Verified live 2026-08-20 (see §3.4).
- The GPU-routed V3-lowered path (`vindex3 exec --backend metal-lowered`, the fast ~100+ tok/s production
  path) does **not** fire this hook — its own code comment says so explicitly ("production integration
  decides where observation lives once routing stops being a host decision"). Extending it was considered
  and deliberately deferred: it would need new per-layer buffer retention + a host readback point threaded
  through `encode_moe_layer_gpu_route`, a real but avoidable GPU-side change, when the CPU-route path
  already answers the same question with zero new code. Revisit only if BW11-1's answer is favourable
  enough to justify building the grouped-expert kernel — at that point the kernel itself needs GPU-side
  instrumentation anyway, and this trace becomes redundant.

**Accounting.** `larql dec-bench window-union` (new, `crates/larql-cli/src/commands/primary/dec_bench/window_union.rs` +
`window_union_runtime.rs`) parses one or more trace files, and for each layer and each `K`, slides
K-position windows (stride 1) over that layer's `seq`-ordered rows. Each window is one "cell" fed to the
**same, unmodified** `routed_weight_bytes_per_token` DEC already uses for its cross-prompt union (a cell's
union vs. naive expert-id count is the same arithmetic regardless of which axis the rows come from) — so
the two numbers are computed by literally the same tested function, differing only in what a "row" is drawn
from. Windows pool per layer (matching DEC-0's own pooling convention), then layers pool across prompts
into a mean/median/p10/p90 spread. R10 applies: overlapping windows at stride 1 are not independent
samples, so the spread describes *shape*, not a confidence interval.

`per_expert_bytes` for gpt-oss-20b's MXFP4 container (derived from code constants, not a literal):
`MXFP4_GROUP_ELEMS=32`, `MXFP4_GROUP_BYTES=16` (payload), scale = 1 byte (E8M0) per group, external to the
payload stream. gate_up `[5760,2880]` (fused, interleaved) + down `[2880,2880]`:

```
gate_up: 5760*2880 = 16,588,800 elems / 32 = 518,400 groups
         payload 518,400*16 = 8,294,400 B, scale 518,400 B -> 8,812,800 B
down:    2880*2880 =  8,294,400 elems / 32 = 259,200 groups
         payload 259,200*16 = 4,147,200 B, scale 259,200 B -> 4,406,400 B
per-expert total = 13,219,200 B (~12.61 MiB)
```

### 3.3 What a result licenses, and what it does not (R6, stated once)

This is a **logical union ceiling only**. A favourable `union_frac(K)` licenses *building* a kernel that
groups K tokens' worth of rows landing on the same expert into one weight read (the "expert-grouped
scheduler" DEC-2 already names as missing) — it is not itself a bandwidth number, and must not be quoted
as one until that kernel exists and is priced kernel-level (`feedback_isolated_vs_batched_kernel_profile`,
`feedback_reduction_is_a_kernel_claim`).

### 3.4 Operational note — the capture path is not hardened for long runs

The CPU-route (`--routed-from`, no `LARQL_GPU_ROUTE`) path is a comparison/reference path, not the
production decode loop, and running it for a longer prompt + `-n 24` produced repeated
`kIOGPUCommandBufferCallbackErrorOutOfMemory` errors after the very first captured position (9+ minutes
wall clock, zero usable tokens) — consistent with a per-call buffer leak somewhere in the
`legacy_waits`/`host_resolves` expert-materialisation path that this investigation did not chase down (out
of scope for BW11-1; the fast production path that *would* need hardening here doesn't carry the trace hook
at all — see §3.2). The working footprint, confirmed live: short prompts (a handful of words), `-n 8`. That
still captures every prefill position too, so the effective window population per layer is prefill-length +
8, comfortably enough for `K` up to 8 across multiple non-degenerate windows.

### 3.5 Results (2026-08-20, `larql dec-bench window-union`)

**Corpus:** 5 short, topically diverse prompts (~5–8 words each), `larql run --routed-from --metal -n 8`,
`LARQL_MOE_ROUTE_TRACE` per prompt. Every capture verified clean (`8 token(s)` generated, zero
`Insufficient Memory` errors — see §3.4 for the config that does OOM). 24 routed layers × 72–75 captured
positions (prefill + 8 decode) per prompt = 120 (layer, prompt) cells pooled per K.

| K | measured union_frac (mean) | amortisation | **R9 shuffled-marginal control** | measured vs. control |
|---|---:|---:|---:|---|
| 1 | 1.0000 | 1.00× | 1.0000 | — (equivalence point, no window possible) |
| 2 | 0.7841 | 1.28× | 0.8360 | **6.2% below control** |
| 4 | 0.5830 | 1.72× | 0.6508 | **10.4% below control** |
| 8 | 0.4169 | 2.40× | 0.4657 | **10.5% below control** |

The control is `shuffled_control_union_frac` (500 trials, seed 1): pool a layer's rows across all 5 prompts,
Fisher-Yates shuffle to destroy sequence adjacency while keeping the exact same rows (same per-expert
marginal, same real top-4 joint structure), re-measure on disjoint K-chunks, average. **At every K the
windowed ratio sits below its own marginal-preserving control** — the R9 confound (a skewed per-expert
popularity distribution alone, with zero real temporal structure, would already push union_frac below the
naive uniform-random closed form: hand-computed uniform-random prediction at K=8 is 0.656, well above both
the shuffled control's 0.466 and the windowed measurement's 0.417) does not explain the effect away. GPT-OSS's
router shows genuine, if modest, positive correlation between the experts a token selects and the experts its
immediate predecessor selected, beyond what its skewed marginal alone would produce.

**Reading it:**

- The effect is **real but modest** — not DEC-0's ~7.2× (which is a different axis/shape entirely, not a
  fair comparison per R2: cross-prompt batch union at 6.25% activation vs. this same-sequence union at 12.5%
  activation). A same-sequence verification batch of K=8 draft tokens has a *logical* ceiling of ~2.4× on
  routed-expert weight traffic if a kernel could realise the union perfectly — most of that ceiling (2.0× of
  the 2.4×) is the marginal/popularity-skew effect any K=8 sample would show; genuine sequence structure adds
  roughly another 0.2–0.3× on top.
- This sits within the pasted proposal's own estimated range for "multi-token weight amortisation" (1.5–4×),
  on the lower-middle end, and specifically requires K≈8 — K=2/4 amortise far less (1.28×/1.72×).
- **Per R6, this licenses *investigating* a token-grouped expert-GEMV kernel — it does not itself justify
  building one.** The ceiling (2.40× at K=8) is small enough that it needs weighing against that kernel's
  real cost/complexity before committing engineering time: DEC-2's own note applies verbatim ("a structural
  reduction is a claim about a KERNEL, not a matrix — price it on the kernel that will actually run it").
  A grouped-expert kernel's realistic yield is bounded by 2.40× and will be lower once dispatch/gather
  overhead is priced in (R11's own gpt-oss precedent: R4's sparse-FFN kernel captured only 23–40% of its
  row-count reduction in practice).
- **Caveats (R10):** 5 short prompts, ~75 positions each — a small, English-only, single-domain-mixture
  corpus. The spread (p10/p90 above) is over *overlapping* windows within that corpus, not an independent
  sample (R10) — the control's 500-trial Monte Carlo average is the more defensible number to quote. A wider
  prompt set, and pushing K past 8 (headroom exists — captured positions per layer already reach ~75), are
  the natural next steps before treating 2.40× as gpt-oss's real ceiling rather than this corpus's reading of it.

**Verdict: BW11-1 does not, on its own, justify BW11-2 (building the grouped-expert kernel) at high
priority** — the ceiling is real but small relative to the engineering cost such a kernel would carry, and
smaller than the pasted proposal's optimistic estimate. It is priced now, honestly, rather than assumed.
