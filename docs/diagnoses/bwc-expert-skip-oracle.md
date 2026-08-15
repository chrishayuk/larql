# BW-C — whole-expert compute-skip oracle: does the trajectory survive?

**Date:** 2026-08-14 · **Box:** M3 Max, CPU decode (8 rayon threads), AC power
**Instrument:** `crates/larql-compute/src/cpu/ops/moe/expert_override.rs`
(new) · `crates/larql-inference/examples/bwc_expert_skip_oracle.rs`
(new) · `gpt-oss-20b-q4k.vindex`

## The question

BW-B's whole family (compact-dense, materialize break-even) answered
"reduce the representation of an operation" — a smaller, contiguous
version of a selected weight subset still gets read. BW-C tests the
other lever registered alongside it in `docs/diagnoses/
bw10-live-gate.md`: **delete the operation entirely.** When it works,
there is no gather, no compact materialization, no partial kernel — the
entire weight movement for one expert call disappears.

Method, per the brief: oracle ONE real expert invocation at a time on a
real serving trajectory (gpt-oss-20b, greedy decode) — no predictor, the
target is a real observed routing decision, not a guessed index.
Substitute: zero, which for GPT-OSS's un-normalised MoE combine IS
residual/identity pass-through (see the module doc on
`expert_override.rs` for the precise caveat about GPT-OSS's
renormalised top-k weights — zeroing one selected expert's contribution
leaves the survivors summing to `1 - w_e`, not renormalised to 1; a
real, well-defined perturbation, distinct from "never routed there").
Scored by TRAJECTORY preservation, not local cosine: does the
CONTINUATION's greedy token sequence, for M tokens after the ablation,
match the unperturbed baseline.

## A methodology bug the live run caught (bank this alongside the result)

The first working version of the hook used
`moe_route_observe::LayerScope` (thread-local) for layer attribution and
observed **zero expert calls on every run**, despite `LARQL_MOE_DEBUG=1`
confirming real routing and non-zero MoE output. Root cause:
`cpu_moe_forward`'s `add_expert` closure runs on **rayon worker
threads**, and `LayerScope`'s `CURRENT_LAYER` is thread-local — set on
the driving thread, invisible to a worker. Fixed by mirroring
`within_expert.rs`'s existing pattern exactly: a plain `AtomicUsize`,
set once per layer by the driver, read from whichever thread
`add_expert` executes on.

That fix alone still observed zero calls. A second, more consequential
bug: `cpu_moe_forward` has **three** per-expert dispatch paths, and only
two of them call the `add_expert` closure at all:

1. Non-spin-pool: rayon `fold`/`reduce` over `add_expert` — hooked.
2. Spin-pool, low active-expert count: one spin-pool chunk per expert,
   calling `add_expert` inside — hooked.
3. **Spin-pool, "row-parallel" schedule** (`forward.rs`, the
   `active.len() * EXPERT_PARALLEL_MIN_FILL <= pool_threads` branch):
   calls `run_single_expert_kq_q8k_parallel_into` **directly**, with its
   own inline per-expert loop — NOT through `add_expert`.

GPT-OSS-20B's routing (top-4 experts) on an 8-thread pool
(`4 × EXPERT_PARALLEL_MIN_FILL(2) = 8 ≤ 8`) hits branch 3 on **every**
call — meaning the "obvious" hook location (`add_expert`) is
architecturally correct but empirically never executes for this model
on this hardware. Confirmed by direct measurement, not assumed: a
temporary unconditional `eprintln!` inside `add_expert` produced zero
output across a run that unambiguously computed real, non-zero MoE
output (`moe_out_rms` values in the `LARQL_MOE_DEBUG=1` trace). Fixed by
adding the identical two-line hook (`observe` + `should_skip`) to
branch 3's loop. Both hook sites are pinned as one instrument by
`expert_override.rs`'s own doc, and the live gate — running the real
harness against real hardware — is what caught the second bug; the unit
tests alone (which exercise `should_skip`/`observe` in isolation, not
through `cpu_moe_forward`'s real dispatch) could not have.

## Result

Prompt: "The history of the Roman Empire began when" (8 tokens). Greedy
16-token continuation. 8 oracle targets, spread across 2208 real
observed `(layer, expert)` calls from one baseline decode (no repeats —
the same pair only appears once in the target list even if it fired
many times across positions; `arm_once` ablates its FIRST occurrence).

| layer | expert | fired | match_pos (of 16) | first_diverge |
|---:|---:|:---:|---:|---:|
| 0 | 21 | yes | 4 | 4 |
| 3 | 12 | yes | 5 | 5 |
| 8 | 16 | yes | **16** | none |
| 10 | 5 | yes | **16** | none |
| 14 | 16 | yes | **0** | 0 |
| 15 | 0 | yes | **16** | none |
| 19 | 15 | yes | **0** | 0 |
| 22 | 18 | yes | **16** | none |

**4 of 8 (50%)** tested single-expert-call ablations left the greedy
continuation **byte-identical** across the whole 16-token window — that
expert's entire weight movement (gate + up + down, ~14 MB at this
model's Q4_K/hidden=2880/inter=2880 shape) could have been skipped with
zero effect on this trajectory. **2 of 8 (25%)** diverged from the very
first generated token. **2 of 8 (25%)** matched partway (4–5 tokens)
before diverging.

No obvious depth pattern in this sample: layers 8 and 10 (low-mid) and
15 and 22 (mid-high) all fully preserved; layers 14 and 19 (also
mid-high) diverged immediately. Whatever makes an invocation skippable
is not simply "how deep" on this evidence — consistent with the
original critique's framing (fraction of invocations that can disappear
is the quantity of interest, not a depth rule) but this sample is far
too small (8 points, 1 prompt, 1 trajectory) to characterise the
pattern, only to establish that both extremes are real and common.

## What this does NOT show

- **Not a production estimate.** 8 points, one prompt, one greedy
  trajectory. The headline "50%" is a first-pass signal that the
  phenomenon is real and not rare, not a calibrated skip rate.
- **Not correctness-scored beyond exact-match.** `match_pos`/
  `first_diverge` are argmax-token agreement only — no KL/logit
  comparison (see the module doc's rationale: token-sequence match is
  itself a direct, unambiguous trajectory-preservation signal, and
  avoided a real plumbing gap — `generate_with_engine_resident` doesn't
  expose per-step logits, only sampled tokens).
- **Not the "router picked k-1 experts" claim.** See the renormalised-
  weight caveat above — this measures "delete the contribution
  post-hoc," a distinct, real perturbation from "the router never
  routed there."
- **Not bytes measured by the BW10 ledger.** `approx_bytes` here is a
  theoretical Q4_K figure from `hidden × intermediate × 3 × 4.5 bits`,
  disclosed as approximate — for a load-bearing byte claim, wire this
  surface into `movement_ledger::coverage::Surface` the way BW-A wired
  MoE-expert bytes for the GPU path.

## Not yet done

- Mean-expert-response and previous-invocation substitutes (both need
  new infrastructure — no running-mean or last-value cache exists for
  MoE experts today; confirmed absent during scoping).
- The "bytes avoided vs. generation divergence" graph across many more
  sample points, once a cheaper (KV-forked, not full-re-decode) replay
  path is available — the CPU reference path decodes at ~2.4 tok/s
  single-box; the current harness re-decodes the whole prompt+
  continuation per target, which is fine for ~10 targets and would not
  scale to hundreds without either KV-forking (`larql-kv`'s
  `BoundaryCheckpoint`, not currently wired to this decode loop) or a
  much shorter continuation window.
- Combinations of 2+ simultaneous ablations, per the original brief's
  "search one expert invocation at a time first. Then perhaps
  combinations."
- BW-D (permutation-aligned expert redundancy), BW-E (residency
  horizon measured against the ledger) remain open from the registered
  BW programme.

## BW-C1 — skippability correlates (2026-08-14)

**Question**: does the router's own selection weight predict whether a
real invocation is safe to skip — the first, cheapest, most obvious
covariate? If it does, skip decisions could ride the routing signal
already computed for free. If it doesn't, the redundancy BW-C found is
not something the router is tracking.

**Prerequisite — KV-fork wired and validated before scaling.**
`crates/larql-inference/examples/bwc1_kvfork_sanity.rs` forks
`larql-kv`'s `BoundaryCheckpoint` (built for EXP-25/semantic-promotion,
domain-agnostic, previously unwired to this decode path) onto
`StandardEngine`'s per-layer KV seam, and runs the same R1/R4 gate
discipline as `larql-kv`'s own replay-gate tests:

- **R1 (null case)**: two clean replays from the same checkpoint —
  bit-identical (`[1261, 1261, 1261, 1261, 1261, 1261]` both times).
- **R4 (control)**: real single-expert ablations from the same
  checkpoint must be able to diverge. First attempt at the "3 steps past
  prompt" position failed this — **not a restore bug**: even the
  un-restored baseline was already repeating one token 6 times, a
  genuine greedy repetition attractor (attractors resist small
  perturbations by construction, so every one of 96 candidates there
  showed `fired=true, diverged=false`). Fixed by capturing at position 0
  (immediately post-prefill) instead. Re-run: R4's 6th candidate
  (layer=1, expert=16) broke a repetition loop cleanly
  (`[30, 623, 4928, 25, 392, 5958]` vs the clean `[1261]×6`). Both gates
  pass — KV-fork is sound on this path.

This is the same lesson as BW-C's two dispatch-path bugs, applied
before scaling rather than after: **every new measurement instrument
needs a live positive control through the exact production dispatch
path** — a null-case pass alone (R1) does not prove a harness can
detect a real effect; only a control that is forced to diverge (R4)
does.

**Harness**: `crates/larql-inference/examples/bwc1_skippability_correlates.rs`.
For each of 6 diverse prompts × 8 checkpoint positions (steps 0, 2, 4,
6, 8, 10, 12, 14) × 3 sampled layer depths (early/mid/late, as
1/6·1/2·5/6 of `num_layers`) × every really-observed expert at that
layer (GPT-OSS top-4 routing ⇒ 4): capture a `BoundaryCheckpoint`,
observe real `(layer, expert, router_weight)` triples for one step,
restore, decode a 6-token clean baseline, restore, then per target:
`arm_once`, decode 6 ablated tokens, restore, recompute baseline logits
at the same call shape, and record exact-match position, KL (bits) and
top-1-margin change at the intervention token via
`hidden_to_raw_logits` (already public — no changes needed to
`generate_with_engine_resident`). **576 interventions total** (6 × 8 ×
3 × 4), comfortably inside the requested 500–1500 stratified-census
range, using KV-fork instead of full re-decode per target — this is
what made 576 points tractable where BW-C's un-forked harness only
reached 8.

**Result — router weight is falsified as a predictor:**

| n=576 | safe | delayed | immediate |
|---|---:|---:|---:|
| count | 426 | 107 | 43 |
| % | 74.0% | 18.6% | 7.5% |

- router weight: safe mean = 0.2467 (sd 0.0658, n=426) vs non-safe mean
  = 0.2593 (sd 0.0842, n=150) — barely different, and in the *wrong*
  direction to support "high weight ⇒ unsafe to skip."
- **point-biserial correlation(router_weight, is_safe) = −0.0776** —
  `|r| < 0.1`, essentially no linear relationship. The first, most
  obvious hypothesis does not hold at this scale.
- **rank-within-observed-top-k** (an ordinal proxy for weight, immune to
  cross-checkpoint weight-scale variance) confirms the null from a
  second angle: safe% is flat across rank 0–3 (70.8% / 75.7% / 73.6% /
  75.7%), `r = 0.032`. If weight mattered, safe% should fall
  monotonically with rank (rank 0 = highest weight); it doesn't.

**The one real signal found — layer depth, not routing confidence:**

| layer depth | n | safe | delayed | immediate |
|---|---:|---:|---:|---:|
| early (≈1/6) | 192 | 65.6% | 26.6% | 7.8% |
| mid (≈1/2) | 192 | 74.0% | 17.2% | 8.9% |
| late (≈5/6) | 192 | 82.3% | 12.0% | 5.7% |

`correlation(layer, is_safe) = 0.155` — "weak" on the harness's own
scale, but monotonic and the largest of every covariate tested besides
KL itself. Later experts are more skippable than earlier ones. This is
a genuinely different lever from "the router was unsure" — it's about
*where in the network* an invocation sits, not *how confidently it was
routed*.

**Internal-consistency check, not an independent predictor**: KL-bits
at the intervention token separates cleanly by eventual label (safe
mean 0.0137, delayed mean 0.0319, immediate mean 0.2012;
`correlation(kl, is_safe) = -0.221`, the strongest of anything measured)
— as expected, since it's a continuous early read of the same
divergence the discrete label describes 6 steps later, not a covariate
available before running the ablation. `|top-1-margin change|` shows
the same weak, expected direction (`r = -0.154`): bigger immediate
margin swings track less-safe outcomes.

**Reconciling with BW-C's original 50%**: BW-C's 8-point, 1-prompt,
un-stratified sample was explicitly flagged in its own writeup as "far
too small to characterise the pattern, only to establish both extremes
are real and common" — that caveat held. The well-powered estimate is
**74.0% safe** (n=576, ≈70–78% at a rough 95% binomial CI), not 50%;
the earlier number was sampling noise from a tiny, non-stratified draw,
not a wrong direction.

### What this does NOT show

- **Contribution-norm not yet tested.** The brief asked for
  router-weight *and* contribution-norm predictivity before deciding
  BW-C's fate. Only router-weight (plus the free-to-compute rank/layer/
  KL/margin covariates already in scope at each call site) was measured
  this pass — expert output norm, weighted-contribution norm, and
  contribution/residual-norm ratio need capturing the expert's raw
  output vector before the weighted combine, which `add_expert`'s
  current hook site does not expose (only the scalar `w` is visible
  there). Open increment, not done here.
- **Correlations share checkpoints, not fully independent draws.** 576
  rows come from only 48 checkpoints (6 prompts × 8 positions); rank
  and layer vary *within* a checkpoint's ~12 interventions (more
  independent), but checkpoint-level factors (e.g. proximity to a
  repetition-prone position) could still correlate outcomes within a
  checkpoint. Not corrected for here — a caveat on the precision of the
  weak signals (`r ≈ 0.15–0.22`), not on the strong null (`|r| < 0.1`
  is not going to flip sign from a clustering correction).
- **Still zero-substitute, greedy-only, single model.** Same scope
  limits as BW-C's first pass — mean-response/previous-invocation
  substitutes and non-greedy trajectories remain untested.

### Not yet done

- Contribution-norm covariates (see above) — the other half of the
  brief's explicit ask.
- The bytes-avoided-vs-divergence graph the original critique targeted
  — now tractable at scale via the same KV-fork, not yet built.
- BW-D/E, combinations of simultaneous ablations: still open.

## BW-C1/C2 — contribution norm (2026-08-14)

**Question**: BW-C1 killed router weight. Is skippability a magnitude
effect instead — does the raw SIZE of the deleted contribution predict
whether it's safe to delete, independent of how confidently the router
selected it?

**Instrument extension**: `expert_override::observe` now also captures,
at zero extra allocation, the ablated expert's raw (pre-weight) output
L2 norm (computed once per hook site, right where the raw output vector
already exists — before the `w * v` scaling) and the incoming residual
stream's L2 norm (captured once per POSITION, not per expert call, via
a new `set_current_residual_norm` cross-thread atomic set by the driver
in `hidden.rs`, mirroring `set_current_layer`'s pattern). Observations
became a named `ExpertObservation` struct partway through this work —
clippy's `type_complexity` lint on a growing tuple was the proximate
trigger, but the real reason is that a 5-element positional tuple is
exactly the kind of thing that silently transposes two `f32` fields at
a call site, which this module had already grown into across two
covariate waves.

**Why the ratio, not just the raw norm**: raw output norm is
**confounded with depth almost completely** —
`spearman(raw_output_norm, layer) = 0.9428`. Activation/residual norms
grow substantially with depth in this architecture on their own, so a
raw-norm covariate would mostly be re-measuring "how deep" under a
different name, not testing "how big was THIS contribution relative to
what it's being added to." `contrib_over_residual_norm` (=
`out_norm / residual_norm`) is the normalisation that strips this out —
confirmed working: `spearman(contrib_over_residual_norm, layer) = 0.0827`,
near-zero, unlike the raw norm's 0.94.

**Result — contribution norm is ALSO not a predictor, once measured
correctly** (same 576-intervention census, n=576 throughout):

| covariate vs is_safe | pearson | spearman |
|---|---:|---:|
| raw_output_norm | 0.1327 | 0.1419 |
| weighted_contribution_norm (`w·out_norm`) | 0.1167 | 0.1317 |
| **contrib_over_residual_norm** | **0.0120** | **0.0376** |

The raw norm's weak positive number is not new information — it is
almost entirely the layer-depth signal C1 already reported (r=0.155)
riding along on a covariate that happens to track depth at
spearman=0.94. Once contribution magnitude is measured RELATIVE to the
stream it's being added to (the ratio), the correlation with
skippability collapses to essentially zero under both measures.

**A real methodology correction, caught by the standing "positive
control every new instrument" rule**: the first pass tested the
positive control (does contribution norm predict its own immediate
downstream KL, which it obviously should if it's measuring anything
real) with Pearson and got `r = -0.0483` — wrong sign, an apparent
FAIL. Re-checked with Spearman before trusting that: `r = 0.1564`,
weakly positive as expected — a genuine PASS. The contribution-norm
covariates span roughly three orders of magnitude across layer depth
alone; Pearson on a heavy-tailed magnitude covariate lets a handful of
extreme late-layer points dominate the linear fit and can flip its
sign relative to the true monotonic relationship. This is a "match the
metric to the operation" failure, not a finding — the harness now
reports both for every C2 covariate, and the Spearman column is the
one to trust. The corrected control passes; the is_safe correlations
above (also essentially unchanged between the two measures, since
`contrib_over_residual_norm` is not itself as heavy-tailed as the raw
norms) can be read at face value.

**Standing conclusion after C1 + C2**: neither of the two "obvious"
covariates — how confidently the router selected an expert, or how big
that expert's contribution was — predicts whether deleting it is safe.
The only real (if weak) signal across everything tested remains layer
depth. This is consistent with the surrounding network state, not the
intervention's own properties, determining whether a contribution is
causally necessary — precisely the hypothesis that makes BW-C3
(minimum sufficient expert set per checkpoint, exhaustive subset
ablation of top-4) the next experiment, not a magnitude-based
predictor search.

### Not yet done (C2)

- Contribution direction relative to the residual stream (cosine
  between the expert's raw output vector and the incoming residual) —
  the brief's "if cheap" covariate. Deferred: unlike a scalar norm,
  this needs the actual output VECTOR transported out of the hook
  (real new plumbing — memory and another dispatch-path surface, not a
  free add).
- Relative contribution among the four co-selected experts (e.g. this
  expert's norm as a fraction of the sum of all four) — not yet
  computed; would need cross-expert aggregation at observation time
  that the current per-call hook doesn't do.

## BW-C3 — minimum sufficient expert set (2026-08-14)

**Question**: individually-safe (74%, BW-C1/C2) does NOT imply
jointly-removable. Experts A, B, C could each be individually
dispensable while mutually redundant with each other — not all three
simultaneously droppable. This is the number that actually bounds how
much routed-expert compute could be cut, if some policy could find it.

**Instrument extension — `arm_set`, generalising `arm_once` to
multiple simultaneous targets.** `expert_override`'s single-expert
`TARGET_EXPERT: AtomicUsize` became `TARGET_MASK: AtomicU64` (bit `e`
= expert `e` is targeted), `FIRED: AtomicBool` became `FIRED_MASK:
AtomicU64` (bit `e` = expert `e`'s ablation actually fired). Each
targeted expert fires independently — its own bit cleared by its own
compare-exchange the first time `should_skip` sees it — so `arm_once`
is now exactly `arm_set(layer, &[expert])`, preserving every existing
test's observable behaviour unchanged (confirmed: all 7 pre-existing
tests pass without modification). 4 new tests cover the set-specific
behaviour: independent per-expert firing, partial-fire detection via
`fired_mask`, and the `arm_once`≡`arm_set` equivalence.

**Method**: same KV-fork checkpoint machinery as BW-C1/C2. For each
(checkpoint, target layer): capture the real top-4 `(expert, weight)`
routing, then EXHAUSTIVELY test all 15 non-empty subsets of those 4
experts (`arm_set`), decode 6 tokens, compare against one clean
baseline for that checkpoint. `fired_mask()` checked against the
intended subset on every test (0 mismatches across all 1,080 tests —
every real observed top-4 ablated exactly as intended, every time).
`minimum_sufficient_size = 4 - max(|R| : removing R was safe)` — not
assumed monotonic in `|R|`, so all 15 subsets are tested per point,
never a search along one dimension. Scale: 6 prompts × 4 checkpoint
positions × 3 layer depths = 72 (checkpoint, layer) points, 1,080
subset tests, ~25 min wall-clock via the same KV-fork that made
BW-C1/C2 tractable.

**Space + guarantee (R12)**: `minimum_sufficient_size` earns
minimum-CARDINALITY — not greedy, not inclusion-minimal — because it
comes from exhaustive enumeration, never a stopping rule. It is
minimum ONLY within: (a) the 4 experts THIS step's real router
selected (never tested against swapping in a different expert); (b)
safety = exact 6-token greedy match, not a looser quality bar; (c) one
single decode-step snapshot (the same layer at a different position
has a different real top-4 and a different result). Says nothing about
longer horizons (BW-C4) or a repeated policy (BW-C5).

**A real, controlled result — this is a bigger finding than BW-C1/C2**:

| minimum sufficient size | n=72 | % |
|---|---:|---:|
| 0 (whole top-4 group jointly unnecessary) | 48 | 66.7% |
| 1 | 12 | 16.7% |
| 2 | 5 | 6.9% |
| 3 | 4 | 5.6% |
| 4 (nothing tested preserved the trajectory) | 3 | 4.2% |

**66.7% of checkpoints had a REMOVED set of all 4 top-routed experts
that was still safe** — the entire routed-MLP contribution at that
layer, for that one token, was replaceable by pure residual pass-
through (GPT-OSS's un-normalised combine makes this architecturally
well-defined: `h_out = h_post_attn`) with the 6-token greedy
continuation staying byte-identical. This is well above what
independence from BW-C1/C2's 74% individual-safe rate alone would
suggest, and far too large a number to accept without a control.

**Live positive control: repetition-attractor check.** `bwc1_kvfork_
sanity.rs` already found a real case where a greedy trajectory landing
on a repetition attractor read as "safe" under EVERY tested ablation —
not because the computation was redundant, but because the un-
perturbed baseline itself was stuck (`[1261]×6`). Before trusting this
headline, re-ran with each checkpoint's baseline tagged by distinct-
token count (≤2 distinct out of 6 = likely attractor). **Result: 0 of
72 checkpoints flagged** — every baseline had ≥3 distinct tokens
(mostly 6/6, a few 5/6, four checkpoints from one prompt at 3/6 with
no distinguishable min_suff skew). The 66.7% figure is not an
attractor artifact.

**Depth pattern reinforces and sharpens BW-C1/C2's finding**:

| depth | n | 0 | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|---:|
| early | 24 | 13 (54.2%) | 5 | 4 | 0 | 2 |
| mid | 24 | 15 (62.5%) | 7 | 1 | 1 | 0 |
| late | 24 | 20 (83.3%) | 0 | 0 | 3 | 4.2%(1) |

Late layers are the clearest: mostly fully-redundant (83.3%) with
almost nothing at min_suff=1 or 2 (0% each) — a bimodal split between
"the whole group is unnecessary" and "most/all of it is load-bearing",
unlike early/mid's more graduated distribution. Consistent with (not
proof of) the "residual-stream overdetermination" hypothesis: by late
layers the decision may already be sufficiently constrained that
routed contributions become wholesale redundant rather than partially
so.

### What this does NOT show

- **Not a production skip-rate.** One layer ablated per test, one real
  routing trace, 6-token exact-match window, greedy decode only. See
  the harness's own "space + guarantee" printout for the full,
  precise scope.
- **Not simultaneous multi-layer removal.** Every test ablates ONE
  layer's top-4 at a time — never tests removing multiple layers'
  routed contributions in the same forward pass.
- **Not evidence the FFN sublayer is globally unimportant** at those
  positions — only that greedy top-1 token choice over a short window
  didn't change; the underlying probability distribution may still
  have shifted (KL was not measured for these joint tests, unlike
  BW-C1/C2's single-expert census).

### Not yet done

- **BW-C5 — repeated policy.** Apply a conservative skip rule
  continuously during generation (not one-shot) and measure
  quality/trajectory degradation — the transition from causal
  experiment to inference technique.
- KL/margin instrumentation for the joint-subset tests (BW-C1/C2 had
  it for single-expert tests; BW-C3 currently only has exact-match).

## BW-C4 — horizon survival, and a real confound found and corrected (2026-08-14)

**Question**: does BW-C3's "safe at 6 tokens" mean the whole-group
deletion's perturbation was truly absorbed, or has it merely not
surfaced yet?

**Instrument**: `bwc4_horizon_survival.rs` re-derives BW-C3's 72
(checkpoint, layer) points with identical prompts/positions/depths,
but tests only ONE ablation per point — removing all 4 real
top-routed experts at once (`min_suff=0` is exactly "removing all 4
was safe at 6 tokens", since there is only one size-4 removed-set —
no need to re-run the full 15-subset search). Re-derivation reproduced
BW-C3's count EXACTLY (48/72, 0 skips) — a free, strong consistency
check that both censuses measure the same phenomenon. Points decoded
to 64 tokens for both baseline and ablated, with logits retained at
markers 6/16/32/64 to track KL drift even where tokens still match.

**Naive headline — and why it's wrong to quote alone**: survival
33/48 = 68.8% at horizon 64 (100% → 79.2% → 77.1% → 68.8% at
6/16/32/64). The depth split looked clean: late layers flat at 85%
from 16 through 64, early/mid still eroding (61.5%, 53.3%). **This
naive number is confounded by prompt predictability and should not be
the headline.**

**The confound, found by reading the raw per-point table (not the
aggregate) before trusting it**: two of the six prompts' rows looked
suspiciously perfect. Stratifying by prompt:

| prompt | n | survive to h64 |
|---|---:|---:|
| 0 — Roman Empire (open-ended narrative) | 10 | 50.0% |
| 1 — quantum computing (open-ended, technical) | 8 | 12.5% |
| 2 — fibonacci code (open-ended, code) | 5 | 40.0% |
| 3 — recipe ingredient list (templated) | 10 | **100.0%** |
| 4 — formal letter opener (templated) | 8 | **100.0%** |
| 5 — "red, blue, and ___" (templated) | 7 | **100.0%** |

Half the sample (25/48) came from three prompts whose natural
continuation is nearly forced by CONTEXT ALONE (a recipe listing
standard ingredients, a form-letter's stock opening, a three-color
completion) — distinct from the repetition-ATTRACTOR confound BW-C3
already controlled for (a literal stuck-token loop; distinct-token
count ≥3 everywhere here, so that check doesn't catch this). A
near-forced continuation is trivially robust to almost ANY ablation,
independent of whether the deleted computation was architecturally
redundant — this is evidence about the OUTPUT DISTRIBUTION's entropy,
not about the computation.

**Corrected picture — open-ended prompts only (n=23, prompts 0/1/2)**:

| horizon | survival |
|---|---:|
| 6 | 100.0% |
| 16 | 56.5% |
| 32 | 52.2% |
| **64** | **34.8%** |

Templated prompts (n=25): 100% at every horizon, all the way to 64 —
essentially confirming they carry no information about computational
redundancy and should be excluded from any horizon claim.

**Depth split on open-ended prompts only — this is the real result**:

| depth (layer) | n | h6 | h16 | h32 | h64 |
|---|---:|---:|---:|---:|---:|
| early (4) | 6 | 100% | 50% | 50% | **17%** |
| mid (12) | 7 | 100% | 43% | 29% | **0%** |
| late (20) | 10 | 100% | 70% | 70% | **70%** |

**Mid-layer "safe at 6" cases on open-ended prompts collapse to 0%
survival by 64 tokens — every single one was delayed divergence, not
absorption.** Early layers mostly collapse too (17%). **Late layers
hold flat at 70% from horizon 16 through 64** — the SAME plateau
pattern seen in the naive (confounded) split, now confirmed on the
subset that actually carries signal. This directly answers the
question this experiment was built to ask: late-layer whole-group
deletion looks like genuine absorption, not merely slower
propagation, because it stops decaying once past the first ~16
tokens rather than continuing to erode toward BW-C1/C2's early/mid
decay pattern.

**KL-bits among still-matching survivors** (naive, unstratified):
mean 0.0051→0.0026→0.0044→0.0009 at 6/16/32/64 — flat-to-declining,
not a rising trend. Weak additional evidence for absorption over
delayed drift (a genuinely building hidden perturbation about to cross
an argmax boundary would be expected to show rising KL among
survivors), but not re-stratified by prompt type yet — treat as
suggestive only.

### What this does NOT show

- **n=10 for the open-ended/late-depth cell is small.** The 70%
  flat-late-layer number is the most important single figure in this
  section and rests on only 10 points. A dedicated late-layer-only,
  open-ended-prompt-only follow-up census would tighten this
  considerably before it's load-bearing for any bigger claim.
- **"Open-ended vs templated" was a post-hoc 3-vs-3 split**, not a
  pre-registered stratification variable — a reasonable read of an
  unexpected pattern in the raw table, not a designed experiment. The
  next census should stratify by a measured prompt-entropy proxy
  (e.g. mean baseline top-1 margin across the continuation) rather
  than an eyeballed open-ended/templated label.
- **KL-drift wasn't re-computed on the open-ended-only subset** — the
  flat/declining trend reported is on the full (confounded) 48-point
  set.

### Not yet done

- Margin-stratified KL-drift analysis.
- BW-C5 — repeated policy applied continuously during generation, now
  informed by "expect roughly two-thirds durability on contested
  late-layer cases, not near-100%."

## BW-C5 — oracle repeated-policy ceiling (2026-08-14)

**Question**: BW-C1–C4.5 all tested ONE deletion at a time against
the UNTOUCHED canonical trajectory. That answers "is this opportunity
real" but not "do opportunities COMPOSE" — a skip's safety was always
judged relative to a state that had never been perturbed. This asks
the actual inference question: `canonical state → one deletion →
observe` (BW-C3/C4) becomes `modified state → next decision → maybe
delete again → modified state → ...` (BW-C5).

**Method — strict, greedy, single late layer**: `bwc5_oracle_
repeated_policy.rs`, 8 open-ended prompts, 32-token real generations.
At EVERY token of the real generation: capture the CURRENT actual
state (which may already include earlier skips in this same run),
observe the real top-4 routing fresh at that point, and test via a
6-token lookahead whether ablating the whole group HERE leaves that
window byte-identical to not ablating. If yes, COMMIT the skip and
advance the real trajectory from the ablated state; if no, advance
normally. At the end, compare the accumulated real trajectory against
a SEPARATE, never-ablated canonical decode of the same length from the
same prompt — the number that actually answers whether locally-
invisible composes into globally-invisible. `arm_set`'s `fired_mask`
checked against the intended target on every single opportunity: 0
mismatches across 256 tests, 0 refused for a malformed top-4 — clean
instrument behaviour throughout, no dispatch-path surprises this time.

**Result — composition mostly holds, well beyond expectation**:

| | value |
|---|---:|
| aggregate skip rate | **227/256 = 88.7%** |
| prompts with FULL 32-token fidelity vs canonical | **6/8 = 75.0%** |
| approx bytes avoided (8 prompts, 32 tokens each) | 12.7 GB |

| prompt | opportunities | skipped | skip% | first divergence |
|---|---:|---:|---:|---:|
| 0 — Roman Empire | 32 | 31 | 96.9% | none |
| 1 — quantum computing | 32 | 29 | 90.6% | **7** |
| 2 — fibonacci code | 32 | 16 | 50.0% | none |
| 3 — detective | 32 | 31 | 96.9% | **7** |
| 4 — scientists debated | 32 | 32 | 100.0% | none |
| 5 — economic policy | 32 | 28 | 87.5% | none |
| 6 — climate research | 32 | 30 | 93.8% | none |
| 7 — novel's protagonist | 32 | 30 | 93.8% | none |

At the single late layer tested, a strict/greedy/unrestricted policy
skips the whole top-4 group on the large majority of tokens (88.7%
pooled, ranging 50–100% per prompt) and preserves the EXACT canonical
trajectory over the full 32-token window on 6 of 8 prompts. This is
well above the "even 15–25% would be a huge result" bar this
experiment was framed against — locally-invisible skips are, for this
layer, mostly composing almost for free, not accumulating into visible
drift.

**Two divergences, same position — flagged, not yet explained**: both
non-fidelity prompts (1 and 3) diverged at EXACTLY position 7. With
only 2 instances this could be coincidence — per the standing
[[feedback_coincidental_invariant]] pattern, do not read a mechanism
into an n=2 coincidence. Worth checking on a larger run whether this
recurs at a real rate above chance, but not investigated further here.

### What this does NOT show

- **Single late layer only.** This composes ONE layer's repeated
  removal — it says nothing about simultaneously skipping MULTIPLE
  late layers' groups in the same forward pass, which is the actual
  lever for a large aggregate compute reduction. The natural next
  increment, not attempted here.
- **Fidelity checked only to 32 tokens.** BW-C4.5 found individually-
  tested "safe at 6" cases keep eroding out to 64 tokens (66.7%
  survival at the most contested cell, not 100%). Some of this run's
  "full fidelity" prompts might still diverge if extended past 32 —
  the 75% full-fidelity figure is a 32-token-window statement, not an
  unbounded one.
- **Greedy/local, not globally optimal.** "Locally safe" is judged
  against a 6-token lookahead from the CURRENT (possibly already-
  modified) state — a different sequence of skip/no-skip decisions
  might do better OR worse. This is an upper bound under ONE specific
  policy, not a search over policies.
- **Strict exact-match only** — no KL/quality-thresholded variant
  tested yet, by design (an unambiguous ceiling, not muddied by a
  quality metric, per the brief).
- **n=8 prompts, exploratory scale** — not yet a properly-powered
  census in the sense BW-C1–C4.5 used for their headline numbers.

### Not yet done

- Multi-layer simultaneous skipping (the actual aggregate-reduction
  lever).
- Extend fidelity-checking horizon past 32 tokens for the composed
  policy.
- Investigate the position-7 double-divergence at scale.
- A percentage-capped policy ladder (C5-A/B/C from the brief) —
  useful once the unrestricted ceiling's failure mode is better
  understood.
- Put a confirmed policy through the production Metal/serve path and
  let BW-A score bytes AND latency — the step that turns this from a
  research result into an engineering one.

## BW-C4.5 — late-layer census at scale, resolving the n=10 plateau (2026-08-14)

**Question**: BW-C4's late-depth, open-ended cell (70% flat from
horizon 16 through 64) rested on only n=10 — real signal, or a small-
sample artifact? Scaled up: LATE LAYER ONLY (drop early/mid — no
longer the question), 20 deliberately open-ended prompts (avoiding
recipe/letter/list-completion patterns), 4 positions each = 80
checkpoints, 63 safe-at-6 (78.8%, consistent with BW-C3/C4's late-
depth acceptance rate). Also replaced BW-C4's eyeballed open-ended/
templated label with a continuous covariate: mean baseline top-1
logit margin across the clean continuation (low = contested, high =
obvious/predictable — the templated-prompt signature, measured
directly instead of inferred from prompt topic).

**The plateau does NOT hold at scale — the naive pooled curve
continues eroding**: 100% → 95.2% → 87.3% → **82.5%** at horizons
6/16/32/64 (n=63). This is not flat; it keeps declining through 64.
BW-C4's n=10 "flat at 70%" was itself margin-selection noise from a
tiny sample, not a real absorption plateau.

**But the margin covariate resolves this cleanly, and confirms the
confound generalises beyond literally-templated prompts**:
`correlation(baseline_mean_margin, survives_to_64) = 0.3476`
(moderate) — even within a hand-picked "open-ended" set, some prompts
have fairly predictable immediate continuations, and margin captures
it. Tertile split makes the dose-response exact:

| margin tertile | n | h6 | h16 | h32 | h64 |
|---|---:|---:|---:|---:|---:|
| bottom (<1.79, most contested) | 21 | 100% | 90.5% | 71.4% | **66.7%** |
| middle (1.79–3.53) | 21 | 100% | 95.2% | 90.5% | 81.0% |
| top (≥3.53, most predictable) | 21 | 100% | 100% | 100% | **100.0%** |

The top tertile is PERFECTLY flat at every horizon — behaving
identically to BW-C4's literally-templated prompts even though none
of these 20 prompts use recipe/letter/list patterns. The bottom
tertile shows real, continuing decay, not a plateau.

**Reconciling with BW-C4's n=10 (70% flat)**: the bottom tertile here
(n=21, the closest apples-to-apples comparison — genuinely contested
continuations, late layer only) gives **66.7% at horizon 64**, close
to BW-C4's 70% and confirming that number wasn't spurious — but the
"flat, no further decay past 16" READ was the artifact; at proper
scale there IS continued attrition from 32→64 (71.4%→66.7%), just
much slower and from a much higher base than early/mid layers (which
were at 0% and 17% in BW-C4's corrected split).

**Sharpened conclusion**: late-layer whole-group deletion shows a
real, substantial, and DEPTH-SPECIFIC redundancy signal — roughly
two-thirds of genuinely contested (low-margin) cases survive to 64
tokens, dramatically higher than early/mid layers' near-total collapse
— but it is NOT the flat "fully absorbed, zero further erosion" story
the tiny sample suggested. The honest headline is "large and durable,
not perfectly durable" — a real effect worth building on, not a
plateau to take for granted.

### What this does NOT show

- KL-drift among survivors was NOT re-stratified by margin tertile —
  only the pooled (all-margin) numbers are reported, which given the
  strong margin-survival correlation likely mixes two different
  regimes. A margin-stratified KL breakdown is a natural next
  refinement, not done here.
- The margin proxy is ONE candidate covariate (mean top-1 margin over
  the whole 64-token window) — it was not validated against
  alternatives (e.g. margin over just the first 6-16 tokens, or
  entropy rather than margin) before being trusted; it correlates
  strongly with the templated/open-ended split found in BW-C4
  (perfect flat-100% top tertile) which is reassuring but not a formal
  validation.

### Not yet done

- Margin-stratified KL-drift analysis.
- BW-C5 — repeated policy applied continuously during generation, now
  informed by "expect roughly two-thirds durability on contested
  late-layer cases, not near-100%."

---

## BW-C → production: the execution-policy seam

BW-C5 closed with "then the production Metal/serve path with BW-A scoring
bytes AND latency — where this becomes an engineering result, not just a
research one." The first half of that has landed: see
**[ADR-0027 — Execution-Policy Seam](../adr/0027-execution-policy-seam.md)**.

What it is: `ExecutionStrategy::{Canonical, Skip}`, decided immediately
before the expensive expert kernels on BOTH production Metal MoE arms
(the `LARQL_GPU_ROUTE=1` descriptor path and the CPU-routed zero-copy
path), with the decision recorded into the BW10 ledger as
`requested / executed / skipped / semantic_avoided / physical_avoided`.
Default is `Canonical` behind one relaxed atomic load, so nothing changes
unless a policy is installed.

What it is NOT: a predictor. BW-C1/C2 falsified the two obvious
candidates, so the shipped policies are a static `(layer × step)` mask
and a replay of a recorded oracle trace. The seam exists so that when a
predictor is earned, there is somewhere for it to go.

**`expert_override` is unchanged.** It stays the research instrument
described above — one-shot, per-`(layer, expert)`, on the CPU
intervention path — so every BW-C1..C5 number here remains reproducible.
The seam is a different tool with a different unit (the whole routed
group at a layer) on a different path. Do not conflate them, and do not
build a third thing.

### What the seam makes newly askable

- **The composed BW-C5/C5.1 policy on the serve path.** `TraceReplay`
  takes exactly the `(layer, step)` decisions those harnesses record, so
  the offline oracle ceiling and the production path can be compared in
  bytes rather than only in token parity. Precondition the caller owns:
  a replayed address only means the same thing under a deterministic
  greedy decode from a fixed prompt.
- **Bytes-avoided against bytes-moved, per token, in one currency.**
  `physical_touched + physical_avoided` reconstructs the canonical arm's
  traffic on the covered surface, because both arms price the operation
  with the same shape arithmetic.
- **Whether the byte saving converts.** It is not assumed to: the ledger
  prints a PROJECTED time at the arm's own observed streaming rate and
  labels it "NOT a measured saving". BW-A's MXFP4 arm is the standing
  reason — 39.3% of the bytes bought 14.7% of the wall
  (`byte_cut/wall_cut ≈ 2.7x`). A latency claim needs a steady-state A/B
  with the policy installed and uninstalled, warmup 16 / n 256, which is
  the next measurement, not something this seam delivers on its own.

### Running it — the powered A/B is now one command per arm

`LARQL_EXEC_POLICY` arms the seam from the environment for `larql run`
and `larql bench`; see ADR-0027's "Arming it" section for the grammar and
the two-arm bench recipe. Two arms are worth distinguishing:

- **Unconditional** (`skip-layers:<late-layer>`) — deletes 100% of that
  layer's expert traffic, not the oracle's 88.7%. Available now, no trace
  needed. Upper bound on the byte saving, lower bound on fidelity, and
  the right FIRST measurement because it isolates "does removing expert
  bytes move tok/s?" from "was it safe to remove them?".
- **Oracle-gated** (`trace:<file>`) — replays BW-C5's own decisions.
  `bwc5_oracle_repeated_policy --emit-trace <dir>` writes one trace per
  prompt (a trace addresses `(layer, decode step)` within ONE
  generation). Replay against the SAME prompt; the provenance header
  carries layer/lookahead/generation-length/prompt/skip-count/fidelity so
  the result cannot be quoted without them.

Decode step 0 is the first GENERATED token in both the harness and the
serve path — prefill positions carry their own phase index and are never
skipped — which is what makes the two step indices the same address. The
safety verdict does NOT transfer: it was established on the CPU resident
decode path, and the serve path's routing can differ in fp provenance.
The skip DECISIONS transfer exactly.

### BW-C5.1 status

The multi-layer ladder harness
(`larql-inference/examples/bwc5_1_multilayer_ladder.rs`) is built and
unit-tested but the full matrix has NOT been run. Its fidelity horizon is
now selectable — `--generation-length 32` (BW-C5's window, the default,
directly comparable) or `64` (BW-C4.5's horizon, where individually-safe
cases were still eroding). At anything below 64 the harness prints its
own caveat that the full-fidelity counts are an UPPER BOUND, so a 32-token
result cannot be quoted as a durable composition result by accident.
