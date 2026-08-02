# VINDEX2-LA — the lookup-vs-approximation semantics programme

**Programme:** `vindex2-la` — an eight-experiment programme (LA-0…LA-8) asking whether FFN
computation splits into a lookup-like regime (sparse addressable memory) and an
approximation-like regime (distributed computation), and whether that split is
detectable *before* reading the full FFN bank.
**Not the same thing as:** `docs/vindex2-experiments.md` / `crates/larql-vindex/docs/vindex2-format-spec.md`
(worktree `vindex2`, branch `worktree-vindex2`) — that programme is the VINDEX2 **container
format**: per-region quant tags, bank kinds, manifests, representation variants. It is about
bytes and quantisation fidelity. This programme is about **computation semantics** — whether
there is a real distinction to represent at all. They run in parallel by design; the source
proposal is explicit that the storage ABI does not wait on this, and this programme does not
gate it. See §5.
**Scope:** discovery only. No model, kernel, or format change. LA-6/LA-7 are the only
experiments that could turn this into a VINDEX2 (format) consequence — **both have now run
(2026-08-02) and both CLOSED NEGATIVE for the dynamic form** (see §2l). VINDEX2 does not gain
lookup/approximation-conditioned execution semantics from this programme.
**Status:** v0.3 — LA-0…LA-1d piloted on Model A (TinyModel v11); GLA-0/GLA-1 ported and piloted
on Model B (Gemma 3 4B), 2026-08-01; **LA-6/LA-7 run on Model B at feature-level granularity and
CLOSED 2026-08-02**, combined with the independent real-kernel result [[project_r4_zeroout_sparse_ffn]]
(R4) — see §2l. TinyModel discovery phase closed (see §2e) — Model A's narrow competence envelope
makes family and confidence structurally inseparable; Model B is now the primary discovery model.
Worktree `.claude/worktrees/vindex2-la`, branch `worktree-vindex2-la`, branched from main.
**Registry:** chuk-experiments programme `vindex2-la` — `la-tinymodel-instrument-validation`
(completed, falsified as discovery/validated as method), `gla-gemma-functional-sparsity`
(running — arithmetic fragility confirmed beyond confidence, compositional multi-layer execution
still open), and `EXP-20260801-232758-00633` (LA-6/LA-7, **completed/closed**).
**Date:** 2026-08-02

---

## 0. The question, and the failure it's trying to explain

Prior whole-model FFN-sparsity work on this codebase failed twice, in two different senses:

- **Row-population sparsity** (which FFN *rows*/neurons fire) — R4 zero-out, refuted: the
  kernel captures only 23–40% of the row reduction it measures, with routing free
  ([[project_r4_zeroout_sparse_ffn]]).
- **Shared-input sparsity** (which *input channels* an FFN reads) — MoE latent-axis probe:
  real and heavily concentrated per-channel (free at r=0.875 relative to the project's own
  bits gate) but the *block* form dies at block≥16, and no stable per-layer channel subset
  exists to compile ([[project_moe_latent_axis_sparsity]]).

Both results average over every token-layer event the model computes. The working hypothesis
here is that this is the wrong denominator: **lookup-like computation may be sparsifiable and
approximation-like computation may not be, and prior work measured their average.** If so, a
*conditional* sparsity policy — sparse retrieval when the FFN is behaving like addressable
memory, full/wide execution when it isn't, fail-closed to full execution when uncertain — could
recover savings that a universal policy cannot. That is a real alternative explanation for two
independent negative results, which is exactly the kind of organising claim that needs its own
falsification test rather than accumulating as a third confirmation
([[feedback_organizing_vs_empirical_claims]]) — which is what LA-0…LA-8 is.

The full experiment ladder (LA-0 trace freeze → LA-1 concentration census → LA-2 stable-value
transplant → LA-3 necessity/sufficiency → LA-4 context dependence → LA-5 path repeatability →
LA-6 pre-fetch predictability → LA-7 selective sparsity execution → LA-8 held-out model) and the
five-property lookup-likeness score (address selectivity × semantic stability × causal
autonomy × contribution concentration × (1 − coalition dependence)) are recorded verbatim in
`docs/vindex2-la-proposal.md` (promoted from session transcript to a committed file 2026-08-02 —
it was previously a single point of failure, living only in chat history); this document tracks
what has actually been run against them, not the full text of the proposal itself.

---

## 1. Test models and their real calibration (read before trusting any family label)

| | Purpose | Status |
|---|---|---|
| **A — TinyModel v11 / Cell80** | instrument validation (closed, not discovery) | LA-0…LA-1d done, §2/§2e |
| **B — Gemma 3 4B dense** | primary discovery model, full 8-family battery | GLA-0+GLA-1 piloted, §2f |
| **C — Gemma 4 MoE** | expert-level translation of the distinction | not started |
| **D — held-out (GPT-OSS or other)** | generalisation check | not started, LA-8 only |

**Model A's real capability envelope, established empirically, not assumed.** TinyModel v11 is
trained *only* on TinyStories (16M + 8M tokens; `tiny-model/model/v11/config.json`). Its
tokenizer vocab is broad — WordNet, Wikidata, 77 tree-sitter grammars — but that is a tokenizer
property, built separately, and says nothing about what the 20L/512d/2048-ffn *weights* learned.
The LA-0 pilot confirms this directly: `calibration_probe` items ("The capital of France is",
"Bonjour means", arithmetic phrasing) rank the correct answer 303rd–14,553th, never top-5. So
**Model A cannot support the source proposal's full 8-family battery** (translation, broad
factual, cross-grammar syntax are all out). That battery is Model B's job. Model A's narrowed
family set and why each one is trustworthy or not is in
`tiny-model/experiments/la0_lookup_vs_approx/PREREG.md` and `RESULTS.md`.

**Cell80 is not an in-model circuit.** It is an external Z80 micro-VM whose result gets injected
as a residual delta at a fixed FFN layer/position (`tier1_cell80_compute/PREREG.md`,
`INJECTION_LAYER = 19`), frozen and hash-pinned. It is the proposal's own choice of positive
control for LA-3 ("Your Cell80 prosthesis result becomes the positive control: it should score
strongly lookup-/circuit-like") — a known discrete, causally autonomous computation to check the
lookup-likeness score against. It is not evidence of organic in-weight arithmetic circuits, and
LA-0/LA-1 below did not use the injected arm — only the frozen prompt panel, run organically, as
a no-signal negative control (§2).

---

## 2. LA-0 + LA-1 pilot, Model A — what's banked

Code: `tiny-model/experiments/la0_lookup_vs_approx/` (`dataset.py`, `capture_traces.py`,
`census_la1.py`, `PREREG.md`, `RESULTS.md`). 39 items, 6 families, last-token-position only (a
deliberate pilot narrowing — every item here is framed as a single next-token prediction).

**LA-0 (trace freeze) surfaced two findings before LA-1 even ran:**

- `lexical_cliche` idioms are strongly learned (8/10 positives rank the expected word #1,
  p up to 0.999) but **brittle to paraphrase** — one of three paraphrase items collapsed the
  expected word from rank 1 (p=0.999) to rank 49. That is an LA-4 (context dependence) result
  that fell out of dataset validation, not a separate run.
- `context_retrieval` is **contaminated by a generic corpus prior**: the unrelated-context
  control (`ctx_unrelated`, where the name "Max" was never established) still predicts "Max"
  top-1 at p=0.388 — nearly matching the actually-grounded item's p=0.545. The family label
  needs a revision (vary the entity name across many filler contexts) before it can be trusted
  as a clean in-context-binding test.

**LA-1 (contribution-width census) result: local magnitude concentration does not separate the
families, including the negative controls.** Per-feature contribution ranking
(`c_i = activation_i · down_col_i`, exact for a dense FFN — reconstruction error at K=ffn_dim
is 4e-6, confirming the decomposition itself is correct) gives K90/ffn_dim in a narrow
0.38–0.44 band for every family:

| family | median K90/ffn_dim |
|---|---:|
| story_continuation | 0.376 |
| lexical_cliche | 0.391 |
| context_retrieval | 0.409 |
| novel_composition | 0.412 |
| arithmetic_organic (no-signal control) | 0.425 |
| calibration_probe (no-signal, diagnostic) | 0.436 |

`arithmetic_organic` — a family the model has *zero* measured competence on (expected-token
rank 330–1885) — needs *more* features to reconstruct its own FFN output than `lexical_cliche`
does, not fewer, and the gap between best and worst family is 0.06 of ffn_dim. Checked directly
for the aggregation-masking bug that produced a false negative in
[[project_moe_latent_axis_sparsity]] (layer-aggregated stats reading uniform even when every
per-layer subset is stable): printed K90 **per layer**, not medianed away. Same story at every
layer past L1 — this is not an aggregation artifact.

**What this does and doesn't establish.** Local-magnitude contribution concentration is not
(yet) the lookup-likeness signal, at Model A's scale, ranked by immediate magnitude, at the
last token position. It does not distinguish: (a) "concentration is a generic property of a
trained silu-gated FFN with no task signal" from (b) "TinyModel v11 is too small/architecturally
flat (dense, no expert-selection axis) for the distinction to resolve, and it would show up at
Model B scale." The proposal's own §"Discovery result" separates local-magnitude ranking from
downstream-logit-effect ranking as different curves — per
[[project_moe_latent_axis_sparsity]]'s audit finding that magnitude/Hamming agreement does not
track the unequal *functional* cost of errors, a family gap invisible to local-K90 could still
show up in downstream-effect-K90.

## 2b. LA-1b — downstream-effect census (now run)

Code: `downstream_effect.py`. Patched the same magnitude-ranked `partial_K` into a **fresh**
forward pass at one layer's `ffn.down` output (last-token-position patching only, so causal
attention means the rest of the sequence is undisturbed and layers downstream of the patch
propagate genuinely, not from a cache). Sanity check: full-K patch reproduces baseline logits,
max |logit diff| = 2e-6. Measured in predictive units per
[[feedback_predictive_units_evaluation]]: KL(baseline ‖ patched) in bits, top-1 agreement.

- **Functional recovery converges far faster than geometric recovery.** By K=256 (12.5% of
  ffn_dim), every family reaches 92–100% top-1 agreement and ≤0.001 bits KL — against LA-1's
  local-magnitude K90 of 770–894 (38–44% of ffn_dim) for the same families. Most of what
  magnitude-ranked contributions buy in exact vector reconstruction is functionally redundant.
  This is the concrete case where judging in geometric units understated the real ceiling —
  exactly the failure mode [[feedback_predictive_units_evaluation]] warns about.
- **Layer 0 is uniquely load-bearing for every family** — dropping its entire FFN contribution
  collapses top-1 agreement to 0–27% for all six families, far outside any other layer's range.
  Swamps a naive all-layer median; re-aggregated excluding layer 0, K=0 top-1 agreement:
  `lexical_cliche` 100%, `story_continuation` 97%, `calibration_probe` 97%,
  `novel_composition` 92%, `context_retrieval` 86%, `arithmetic_organic` 71%.
- **That ordering is real but confounded with baseline prediction confidence** — Pearson
  r=0.496 (n=39) between an item's baseline top-1 probability and its post-ablation top-1
  agreement. Partial, not total (e.g. one p=0.164 item still holds 100% agreement), so the
  finding survives as a confound to control for, not grounds to discard it.

**Net read: neither a clean bimodal split nor a clean null.** Closer to a context/confidence-
dependent outcome than the proposal's stratification prior predicted, and only established on
Model A's narrow TinyStories-only competence envelope.

## 2c. LA-1c — confidence-vs-family, and a headline result that didn't survive its own check

Code: `la1c_kstar.py`. Computed a proper oracle `K*` per item (13-point grid, stable
multi-criterion threshold: KL≤0.05 bits AND top-1 preserved AND margin degradation ≤0.15,
holding at that K *and every larger K*), to directly test "does family/state predict required
budget beyond confidence" rather than leave it as a confound.

**First pass looked like a real answer** — R² went from 0.095 (confidence alone) to 0.291
(confidence + family). **It did not survive review.** Adjusted R² (penalising the 5 extra
family-dummy parameters against n=39) shrinks the gain to +0.088. Worse: K* is heavily
zero-inflated (32/39 items have median K*=0), and the nonzero tail is dominated by two outliers
(`arith_gcd`, `arith_fact`, both K*=64, both arithmetic). **Excluding just those two items,
adjusted R² delta flips to −0.009 — family's apparent explanatory power vanishes.** The
original result was two outlier points, not a general pattern.

**Corrected reading:** this pilot does not establish that family predicts functional budget
beyond confidence. It establishes something more modest: single-layer FFN necessity is low
almost everywhere on TinyModel v11, and the rare exceptions cluster in the one family already
known to carry no learned signal. LA-1b's r=0.496 (confidence alone) remains the more defensible
read. A real test needs low-confidence items spread across every family, not concentrated in
one — not built here.

## 2d. LA-1d — layer-0 decomposition: content-specific, not generic conditioning

Code: `la1d_layer0.py`. At layer 0 only, patched six substitutes for the true `ffn.down` output:
`zero`, `mean` (dataset-average, fixed), `rand_matched` (random direction, norm-matched to the
item's true output), `scale_25`/`scale_75` (the item's *own* direction at reduced magnitude),
`transplant` (another item's true output).

**Decisive contrast:** `rand_matched` — full norm, wrong direction — recovers only 9–40%
top-1 agreement. `scale_25` — a quarter the norm, *right* direction — already recovers
0–82%, climbing to 43–100% at `scale_75`. `mean` performs about as badly as `zero` (0–20%)
everywhere. **This rules out layer 0 as generic architectural conditioning** (an embedding-scale
or norm correction would have made `mean`/`rand_matched` competitive with `scale_*`, and neither
is). Layer 0's necessity is about item-specific *direction* — real content-specific
computation. (`arithmetic_organic`'s `transplant` outlier at 71% is consistent with everything
else known about that family: baseline predictions are already low-confidence/generic, so
almost any reasonably-scaled vector lands near a weak, non-specific default.)

**Net position after LA-1c + LA-1d:** downstream functional sparsity is real and large (LA-1b,
unaffected by LA-1c's walk-back), but this pilot has not shown family/task identity predicts it
beyond confidence — confidence itself is doing the work LA-1c tried to attribute elsewhere.
Layer 0 is a separate, clean, positive finding: genuine content-specific necessity, not an
architecture artifact. Full detail in `tiny-model/experiments/la0_lookup_vs_approx/RESULTS.md`.

## 2e. TinyModel discovery phase — closed

TinyModel has done its job as a methodological test harness, but it is the wrong model to decide
whether lookup-like and approximation-like computation are meaningfully separable: its
lexical-cliche items are high-confidence *because* they are memorised TinyStories surface
patterns, its context-retrieval family turned out to be a shallow/contaminated binding test
(§2, LA-0), and its arithmetic is an unsupported low-confidence failure mode rather than a
genuine computation. Family and confidence are structurally inseparable on this checkpoint — not
a dataset-size problem, a competence-envelope problem. Three things TinyModel established and
that carry forward as validated instruments, not conclusions: judge sparsity downstream, not by
local vector reconstruction (LA-1 vs LA-1b); single-layer interventions must not be read as
composable execution sparsity — untested, not shown; layer 0 (on this architecture) performs
real content-specific computation, not generic conditioning (LA-1d). Not expanding the TinyModel
dataset further. Model B (Gemma 3 4B) is now the primary discovery model.

## 2f. GLA-0 + GLA-1 — ported to Model B (Gemma 3 4B), piloted

Code: `chris-experiments/gla0_lookup_vs_approx/` (`dataset.py`, `gla0_census.py`, `RESULTS.md`).
22 items across `language_of`/`capital_of`/`birthplace` (factual), `addition_no_carry`,
`translation`, `generic_continuation` — the same route categories as
`state-construction` Exp 67-71's already-closed component-rank90 finding (factual rank90 3-5,
addition-no-carry 9, translation 20-22; `SYNTHESIS.md` §10), fresh hand-written prompts, not
their case set.

**Porting note:** Gemma has no PyTorch-style hook API in MLX — every Gemma script in
`arithmetic_mechanism/` reimplements the forward pass manually, layer by layer; followed that
pattern. **GLA-0's own sanity check (full-K patch must reproduce baseline exactly) caught a real
bug before any result was trusted**: building the cumulative top-K patch by summing
magnitude-sorted contributions introduces a ~2e-5 float32 reordering difference that, once the
patch vector is cast to bf16 (Gemma runs natively bf16 throughout, unlike TinyModel's float32),
can flip a rounding bucket and compound through 34 nonlinear layers into a 0.25 final-logit
diff. Fixed by summing in natural column order (boolean mask) instead of sort-then-slice; sanity
check now passes at exactly 0.0. **Bf16-specific hazard, not a TinyModel-transferable one** — the
same sort-then-sum code was fine in TinyModel's float32.

Three findings from the corrected pilot, none needing a walk-back:

1. **No Gemma equivalent of TinyModel's layer 0.** Per-layer K=0 top-1 agreement ranges
   81.8-100% across all 34 layers — no catastrophic single layer. Real architectural contrast
   with TinyModel, checked the same way (per-layer, not pooled).
2. **Translation does not show the fragility the geometric rank90 result would predict** — it's
   *maximally* robust to single-layer FFN ablation (100% at K=0), same tier as the factual
   families. This is a genuine **dissociation** between within-instance functional necessity
   (this instrument) and cross-instance construction dimensionality (Exp 67-71's rank90) — a
   route can span a high-rank subspace across many query instances while still tolerating any
   one layer's removal within a given instance. Open question, not yet reconciled (§3).
3. **Arithmetic (`addition_no_carry`) is the fragile family, and — unlike TinyModel's LA-1c
   claim — this survives a confound check.** R²(confidence only) = 0.422 → R²(confidence +
   is_addition) = 0.739 (adjusted delta +0.319), and it **holds after removing the two most
   extreme addition items** (adjusted delta still +0.252) — not an outlier artifact this time.

Full detail, including the numerical bug's root-cause chain and the leave-out checks:
`chris-experiments/gla0_lookup_vs_approx/RESULTS.md`.

## 2g. Reconciliation against Exp 67-71's exact entities — confirms and sharpens

**Scoping correction:** Exp 71's `late_rank_90` is a *route-level* statistic (one number per
route, from the pooled cross-instance cloud) — there is no per-case rank90 to correlate against
in the saved data, so a per-case correlation matrix isn't computable from what exists. What is
checkable: whether §2f's family pattern replicates on Exp 71's actual entities rather than fresh
hand-written ones. Extracted the exact entity/target pairs from
`state-construction/71_component_map_basin/results/component_map_basin_main.json`'s `cases`
field (Bulgaria→Bulgarian, Angola→Luanda, Emmanuel Macron→Amiens, the exact translation words
and addition operands) into `dataset_exp71.py`, reran the full census.

**Both headline findings replicate, and arithmetic sharpens.** Translation stays maximally
robust (98% at K=0, vs 100% on fresh prompts) — the dissociation from rank90=22 is not a
prompt-selection artifact. Arithmetic gets *more* fragile on Exp 71's own operands (44% at K=0,
down from 67%; still only 74% at K=4096 where every other family is at 100%). Confound check
repeated and **strengthens**: adjusted R² delta from adding `is_addition` beyond confidence is
+0.449 (was +0.319), and still +0.502 after excluding the two most extreme addition items.
Cleanest single contrast: `add_1121` at confidence 0.450 — higher than every translation item in
the set — still only reaches 61.8% agreement, while every translation item at equal-or-lower
confidence reaches 88–100%.

Full detail: `chris-experiments/gla0_lookup_vs_approx/RESULTS.md`.

## 2h. GLA-2 — composition: real, bounded, and depth-position-dominated

Code: `gla2_composition.py`. Two arms, both genuine one-shot forward passes with all target
layers simultaneously zeroed (last-token-position only, same intervention point as GLA-1's K=0):
**Arm D** ranks all 34 layers by already-measured single-layer dispensability (global average
K=0 KL), then zeros the `c` most-dispensable layers together, `c` swept 1→34 — caveat: this
ranking is **in-sample**, derived from and tested on the same dataset, not corrected here.
**Arm B** zeros a contiguous depth prefix `0..c-1`, same sweep — tests whether count or position
is what matters. Sanity check (empty zero-set reproduces a plain forward): 0.000000 on both
datasets.

**Finding 1 — real, sizeable compositional safe zone, then a sharp cliff, not a gradual decline.**
Factual/translation routes hold ~100% top-1 agreement with `c` up to 8-12 layers simultaneously
dropped (up to a third of the 34-layer stack, dispensability-ranked) — then collapse
catastrophically by `c=16`-`20` (KL 5-19 bits, agreement to 0%). Replicates on both datasets
(fresh pilot's plateau is even larger, holding to `c=16`-`20`). **Single-layer redundancy is
real and harvestable up to a floor** — not merely padding any one neighbouring exact layer was
privately repairing.

**Finding 2 — arithmetic never gets a safe zone at all.** Already degraded at `c=1` (67% both
datasets) and declines gradually, no plateau — unlike every other family. Extends, not just
repeats, the single-layer finding (§2f/§2g): arithmetic has zero compositional margin.

**Finding 3 — depth position dominates raw count.** At the same `c`, Arm B (contiguous prefix)
collapses far earlier than Arm D (ranked): `generic_continuation` hits 0% by `c=3` under Arm B
where Arm D still holds it at 100% through `c=8`; every family under Arm B has collapsed by
`c=6`, where Arm D holds most families near-perfect through `c=12`. **Which layers, not how
many, is the dominant variable** — a flat per-layer necessity score would badly mis-price a
contiguous-prefix drop.

**Net position going in to §2i:** single-layer redundancy composes up to a real budget,
arithmetic has none of it, and layer identity/position matters more than raw sparsity fraction —
but the specific ~c=8-12 number was an in-sample estimate, corrected below.

## 2i. GLA-2b — held-out transfer: real but narrower than GLA-2's in-sample estimate

Code: `gla2b_heldout.py`. Closed §2h's caveat using a split that already existed for free:
**calibration = `dataset_exp71`** (ranking derived here), **held-out = `dataset`** (the
independently hand-written fresh pilot, never seen by any non-cheating ranking). Five policies
tested on held-out items: `oracle_per_instance` (cheating upper bound, each item's own K=0
ranking), `global_calibration` (§2h's Arm D ranking, reused unchanged), `family_calibration`
(per-family ranking from calibration items), `random_scattered` (fixed-seed random order —
isolates "avoid early layers" from "know which specific layers"), `early_prefix` (§2h's Arm B).
Sanity: 0.000000.

**Finding 1 — real transfer benefit at small budgets.** At `c=6`, `global_calibration` clearly
beats `random_scattered` (91% vs 73%), nearly matching oracle (91%).

**Finding 2 — that benefit erodes and nearly vanishes at exactly the budget §2h found most
exciting.** By `c=12`, `family_calibration` (45%) is barely above random (41%); by `c=16`,
`family_calibration` and `random_scattered` are tied at 27%, and `global_calibration` (18%) is
*worse* than random. **The ~c=8-12 safe zone from §2h was optimistic — under a fixed,
non-cheating, precomputed ranking, what transfers is closer to c≤6-8.** Oracle stays clearly
ahead of everything else past `c=6`, confirming the underlying redundancy is real — it's
specifically the *static, precomputed* ranking that fails to transfer at scale, not the
redundancy itself.

**Finding 3 — the failure isn't uniform and family-specific calibration doesn't reliably fix
it.** At `c=8`, `capital_of` collapses to 0% under both global and random policies while oracle
holds 100%. At `c=12`, `family_calibration` — despite being calibrated specifically on
`language_of` — scores 0% on held-out `language_of`, while the *non-family-specific* global
ranking scores 100% on the same family at the same budget. This is the "unstable ranking"
outcome flagged as a real possibility before this ran: the redundancy is real, but which specific
layers are safe is not robustly captured by a static average-KL ranking, in-sample or
family-conditioned.

**Consequence for what's deployable:** a production policy on this instrument needs either a
materially larger calibration set (30 items may simply be too few for a stable per-layer
average), a genuinely state-conditioned adaptive decision rather than a precomputed rank, or a
conservative ceiling around c≈6-8 rather than the c≈12 in-sample number. Full detail:
`chris-experiments/gla0_lookup_vs_approx/RESULTS.md`.

## 2j. GLA-3 — adaptive risk-controlled skipping: a real bug caught pre-emptively, then a genuine null

Code: `gla3_adaptive.py`. Sequential, state-conditioned decisions: at each layer, estimate skip
risk from the state actually produced by prior decisions (never the baseline trace), skip only
when predicted risk is low. Five cheap pre-FFN features (residual norm, residual delta,
logit-lens margin/entropy, calibration-typicality — no family label), three linear predictors
fit on the already-captured K0 KL costs, plus a `layer_only` control and an `oracle_sequential`
policy that cheats (evaluates the true cost via an extra lookahead forward pass rather than
predicting it). Sanity: 0.000000.

**First fit (OLS on raw KL): silently broken, caught by a pre-check before the expensive run.**
K0 KL is heavily right-skewed (median 0.024, max 11.7); OLS collapsed to a near-constant
prediction, producing all-or-nothing skip behavior and a wrong-signed margin coefficient. Fixed
by fitting in log-space on standardized features, plus an explicit pre-check (R², prediction
range, fraction skipped per threshold) that must be inspected before Phase 2 runs.

**Corrected fit: R² ≈ 0.006–0.086 across all three feature sets — essentially no signal**, and
the embedded calibration-size learning curve confirms this isn't a sample-size problem (R² flat
at 0.07-0.09 across 25%→100% of calibration data). **This is the "neither improves" outcome** —
these cheap linear features do not predict per-layer dispensability.

**`oracle_sequential`, unaffected by the bug, remains excellent**: 12.5 mean layers skipped,
**100% top-1 agreement**, max KL 0.34 bits at the aggressive threshold — dramatically better than
any static policy at a comparable count (`global_calibration` c=12: 64% agreement, 3.72 bits max
KL). Confirms again that large, state-dependent redundancy is real; this specific cheap
approximation of it just didn't work.

**Net position:** a clean three-way split now stands. (1) The redundancy is real and large — four
independent confirmations (GLA-2's arms, GLA-2b's oracle_per_instance, this oracle_sequential).
(2) A static rank captures only the first slice and degrades past c≈6-8 (GLA-2b). (3) A cheap
linear state-conditioned predictor captures essentially none of the remaining gap. The open
problem is the gap between (1) and (3): richer/nonlinear features, or accepting that per-layer
risk needs more computation to estimate than these signals provide. Full detail:
`chris-experiments/gla0_lookup_vs_approx/RESULTS.md`.

## 2k. GLA-4 — one-step fidelity does not survive generation, for any policy including oracle

Code: `gla4_generation.py`. Reduced scope, stated explicitly: 6 items (one per family), 26
generated tokens, oracle lookahead bounded to a 12-layer candidate pool. No KV cache — every step
recomputes the full sequence, mathematically equivalent to a real cache because skip decisions
are recorded per-position and replayed identically at every later step. Sanity: 0.0.

**Headline: 0% of trajectories stayed fully identical to the exact reference, for any of the
three sparse policies (`static_c6`, `random_c6`, `oracle_capped`), on any item.**
`oracle_capped` diverges later (mean 8.7 of 26 tokens vs static's 4.8, random's 2.7) and stays
2.6-3x closer to exact (mean KL 0.484b vs 1.27-1.50b) — meaningfully better, not immune.

**The mechanism matters more than the headline number.** `tr_dog`'s `oracle_capped` trajectory
spikes to 19.5 and 29.2 bits KL at steps 15-16 — *while the oracle's marginal decision at both of
those exact steps was to skip zero new layers.* The divergence traces to **earlier** sparse
decisions (step 14 skipped all 6 candidate layers) permanently altering the effective history —
a one-step-ahead lookahead has no mechanism to see this coming, since it only ever evaluates the
marginal cost of the *current* position's candidates. The trajectory recovers to near-zero KL two
steps later (0.022 bits) — a transient but real tail excursion a mean-KL summary would hide.

**A partial silver lining: the skip budget genuinely shrinks after divergence**, for 5 of 6 items
(e.g. `birth_einstein` 5.8→2.3 mean layers skipped, pre- vs post-divergence) — the one-step
lookahead has no explicit memory of having diverged, but a drifted state naturally produces
higher marginal costs under the same threshold, so the policy becomes measurably more cautious
without being told to.

**Consequence:** speculative execution with rollback (checkpoint, tentatively skip, verify via a
cheap post-hoc integrity signal, commit or replay exact) is now substantially better motivated
than a purely predictive pre-check — Finding 3's failure mode (delayed, compounding divergence
invisible to one-step lookahead) is exactly what a post-hoc check could catch and a forward-only
prediction cannot. Full detail: `chris-experiments/gla0_lookup_vs_approx/RESULTS.md`.

## 2l. LA-6 + LA-7 — CLOSED, combined with R4's independent real-kernel result

Code: `chris-experiments/gla0_lookup_vs_approx/gla6_*.py`. Full proposal text recovered from
session transcript and committed: `docs/vindex2-la-proposal.md` (previously a single point of
failure — only paraphrased here, never the actual criteria). Run at the **feature-level**
granularity the proposal actually specifies (K individual neurons within a layer's 10240-wide FFN
intermediate activation) — the first time this granularity was tested on Model B; GLA-0…GLA-4
only ever tested whole-layer skip (K=0 as a special case of a coarser unit).

**Chain of findings:** a 2-signal deterministic pre-fetch classifier (screened for individual
discriminative power, then jointly threshold-fit — ANDing independently-marginal-optimal
thresholds first collapsed to 0% predicted-positive) transfers to held-out at 74%
accuracy/87% precision — real signal, unlike GLA-3's R²≈0.09 null. But naive simultaneous
application collapses unevenly by family; a corrected compositional oracle (real greedy
sequential validation, not unioned isolated labels) revised the family story — birthplace, the
worst naive collapse, has the deepest real compositional ceiling. A block-touch diagnostic found
scattered magnitude-selected columns touch **29–89× more physical storage than their logical
count** under block-quantized layouts; block-CONSTRAINED selection (cancellation-aware scoring)
recovers honest 1:1 physical accounting at a bounded fidelity cost. In the combined depth×block
sweep, the adaptive classifier is sometimes markedly *worse* than blind random layer selection
under composition — **a trivial static calibration-derived layer ranking beats it at nearly
every operating point.** Best cell: `global_static @ block=64, P=256, N=6` — 91% top-1, 5.7%
byte savings, near-perfect for 5 of 6 families (arithmetic breaks it). Neither the scientific
(≥15%) nor serving (≥25%) promotion gate is reachable at fidelity-preserving depth budgets.

**Combined closure with [[project_r4_zeroout_sparse_ffn]]:** R4 (§0 above) independently tested
the same hypothesis on real Q4K kernels and the real decode loop, same model — isolated sparse
kernel 1.29× faster, but real decode ~0.15× baseline (page/scheduling overhead dominates), and
the kernel captures only 23–40% of its theoretical row reduction. LA-6 (fidelity/locality side)
and R4 (kernel/decode-loop side) converge from opposite ends. **Dynamic feature-, row-, and
block-selective FFN execution is CLOSED as a K3 throughput lever.** Not closed: an
offline-compiled, cross-input-*stable* contiguous representation (R4's "compiled compact-dense"
survivor) — neither experiment tested this; LA-6's own block selection remained
activation-conditioned within each layer (mean contiguous run length ≈ 1.0 throughout), so it is
not evidence for or against that lane.

**Ship decision:** do not promote dynamic FFN substructure sparsity into the K3 20 tok/s roadmap
or as an organising VINDEX2 primitive. Retain block-addressable storage/sparse execution
(`larql-vindex`'s `interleaved_kquant.rs`, `larql-inference`'s `WalkFfn`/`sparse_gather`) as
research instrumentation only. Full write-up: chuk-experiments `EXP-20260801-232758-00633`.

---

## 3. Run order

### Now

| | | Depends on |
|---|---|---|
| 1 | ~~Reconcile GLA-1 against Exp 67-71's entities~~ — done, §2g | §2f |
| 2 | ~~Compositional multi-layer execution~~ — done, §2h | §2g |
| 3 | ~~Held-out calibration split~~ — done, §2i | §2h |
| 4 | ~~Sequential adaptive budgeting~~ — done, §2j; oracle confirms the redundancy is real and large, but the cheap linear predictor tried captures none of it (R²≈0.09) | §2i |
| 5 | ~~Multi-token generation coverage~~ — done, §2k; **no policy survives generation intact, including oracle** — this bounds every prior single-step result | §2f, §2j |
| 6 | J-lens (Anthropic's Jacobian-lens / global-workspace probe, published 2026-07-06) as a richer, downstream-influence-weighted feature source for the risk predictor — MLX port validated (Phase 0a/0b passed), Gemma-scale fit running in background (checkpointed/resumable), interim n=4 finding negative-so-far (only the numerically trustworthy source layer shows signal, and it's concurrent not predictive). See `jlens_port/` | §2j |
| 7 | Speculative execution with rollback — better motivated after §2k's Finding 3 (delayed, compounding divergence invisible to one-step lookahead) than a purely predictive pre-check | §2k |
| 8 | Full-scale GLA-4 replication (22 items, 64 tokens, uncapped oracle, multi-step-ahead lookahead) — this pilot used a reduced scope; the qualitative finding is unlikely to reverse but the quantitative divergence-onset numbers aren't final | §2k |
| 9 | ~~LA-6 + LA-7 — pre-fetch predictability + selective sparsity execution~~ — done, §2l; **CLOSED negative for the dynamic form**, combined with R4's independent real-kernel result | §2j |

### After Model B

10. LA-3 causal-portability tests (transplant across paraphrases/relations, corrupt-answer
    controls) — Cell80 remains the TinyModel-side positive control (§1); Gemma-side portability
    uses `arithmetic_mechanism/a2b_causal.py`'s subspace-ablation pattern.
11. Translate to a real MoE (Gemma MoE or GPT-OSS) — expert/block-level. The feature-level
    question is now SETTLED on Model B (§2l, closed negative for the dynamic form) — a MoE
    translation of this specific thesis inherits that closure unless the MoE's expert-routing
    axis is materially different from within-FFN feature selection, which would need its own
    justification, not an assumed reset.
12. K3 — final bandwidth/execution test, only after (11).

---

## 4. Standing constraints carried over from adjacent programmes

- **Mean-ablate, never zero-ablate**, across stacked layers — zeroing 10+ layers breaks
  `final_norm`+`lm_head` ([[feedback_stacked_zero_ablation]]).
- **A reduction is a kernel claim, not a matrix claim** — confirmed a third time by LA-6/LA-7's
  own closure (§2l): the feature-level classifier found a real pre-fetch-predictable regime in
  fidelity terms, but pricing it against the real kernel (R4, independent) showed the sparse
  execution path regresses decode throughput regardless ([[feedback_reduction_is_a_kernel_claim]];
  now fired three times — DEC-8.8, R4, LA-6/LA-7).
- **Judge in predictive units** — KL/NLL/bits-per-token/first-divergence/top-K, not cosine
  ([[feedback_predictive_units_evaluation]]). LA-1's local-magnitude K90 is a geometry metric,
  not a predictive one — a reason to weight §3 item 1 (downstream-effect K90) over refining the
  local metric further.
- **Don't put lookup/approximation semantics into VINDEX2's (the format programme's) ABI.**
  LA-6 and LA-7 have now both run (§2l) — and both closed NEGATIVE for the dynamic form. This
  constraint is therefore now settled, not pending: dynamic feature/row/block-selective execution
  does not get a graph node type, a trace-conditioned edge, or a selective-execution hint in the
  format ABI. The only door left open is an offline-compiled, cross-input-stable contiguous
  representation, which neither LA-6 nor LA-7 tested and which would need its own new evidence
  before any format consequence.

---

## 5. Relationship to the VINDEX2 format programme

`docs/vindex2-experiments.md` (worktree `vindex2`) is a 9-experiment, 5-gate programme about the
serving *container*: per-region quant tags, bank kinds, MoE manifests, representation variants,
conformance fixtures A–D, up through Inkling-Small. Grepped for overlap: its only uses of
"approximation" are about lossy-vs-exact quantisation fidelity (E5's "profile authority and
approximation policy") — a different sense of the word from this programme's lookup-vs-
approximation *computation* semantics. No content overlap found. The two programmes share a
name prefix because they share an eventual consumer (VINDEX2's query/graph ontology, per the
source proposal's own closing section) — not because they answer the same question. Keep them in
separate worktrees and don't let a result in one silently authorise a change in the other's
pre-registration.
