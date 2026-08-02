# VINDEX2-LA — originating proposal (LA-0…LA-8), verbatim extraction

**Status:** promoted from session transcript to a committed file, 2026-08-02.
Until this file existed, the full LA-2…LA-8 methodology lived only in the
chat history of the session that launched the `vindex2-la` programme
(`234b1d1c-c1f4-4a6f-980f-74935c8179a4`) — `docs/vindex2-la.md` explicitly
says it "tracks what has actually been run against them, not the full text
of the proposal itself," and no other committed file has this detail. That
made the actual pass/fail criteria for LA-6/LA-7 (the two experiments that
gate any VINDEX2 format consequence) a single point of failure against
session/transcript rotation. This file fixes that.

Recovered by a full-repo + git-history + cross-session-transcript search
(2026-08-02) before starting LA-6/LA-7 implementation work — confirmed no
newer/updated version exists anywhere else.

---

## Operational definition — the five-property lookup-likeness score

Composed as:

```
L = address_selectivity × semantic_stability × causal_autonomy
    × contribution_concentration × (1 − coalition_dependence)
```

Explicit instruction: **do not threshold this immediately** — first inspect
distribution shape, per-layer/per-task-family/model-to-model stability, and
check which of five named outcomes actually obtains:

- **A** — clean bimodal split (lookup vs. approximation cleanly separable)
- **B** — continuous spectrum (no clean split, but a real ordering)
- **C** — context-dependent only (the unit that's lookup-like is
  feature×context, not feature alone)
- **D** — no *pre-fetch* predictability (a lookup-like regime exists
  post-hoc but can't be identified before paying the I/O cost)
- **E** — no stable distinction at all

### 1. Address selectivity
Narrow/coherent activation across inputs: activation frequency, activation
entropy across task families, paraphrase consistency, negative-control
rejection, top-input concentration.

### 2. Contribution concentration
K50/K90/K99 needed to recover 90% of: FFN delta norm / downstream logit
effect / next-layer state / final answer behaviour, computed from
`c_i = activation_i × down_vector_i`.

### 3. Semantic stability
Does the same feature/injection cause a stable downstream semantic effect
across compatible contexts? Measured via residual-direction cosine,
top-logit-effect overlap, decoded-concept stability, paraphrase consistency,
layer-to-output persistence. Explicit warning: "the down vector is
physically fixed for every feature, so fixed direction alone proves
nothing" — stability has to be measured in effect, not in the static weight.

### 4. Causal autonomy
Single-feature / small-coalition ablation and injection, damaged-behaviour
recovery, corrupt-value control, unrelated-task control.

### 5. Coalition dependence
Joint-ablation effect vs. sum of individual ablation effects; injection
alone vs. injection with its usual coalition. Explicit warning against
measuring "interaction" by merely summing down vectors (trivially
additive) — the test has to be behavioral, not linear-algebraic.

---

## LA-2 — Stable-value transplant

For candidate features/coalitions:
1. Capture activation on source prompt A.
2. Inject the same resulting contribution into compatible prompt B.
3. Inject into incompatible prompt C.
4. Inject a matched-norm random direction.
5. Inject a corrupted semantic counterpart.

Lookup-like candidates should show: paraphrase transfer, relation-preserving
transfer, semantic specificity, corrupt-value following, minimal unrelated
damage. Approximation features should **not** behave as portable stored
values. Explicitly framed as "much stronger than examining top tokens from
a down vector."

## LA-3 — Necessity, sufficiency and compactness

Curves over: baseline, single ablation, ranked cumulative ablation, single
injection, ranked cumulative injection, sham injection, corrupt injection →
"behaviour lost versus features removed" and "behaviour recovered versus
features restored." A lookup-like circuit should exhibit a **sharp
transition** with a small set. An approximation should degrade and recover
**gradually** across a larger set. **Cell80 is the explicit positive
control**: "it should score strongly lookup-/circuit-like under this test."

## LA-4 — Context dependence

Same candidate feature tracked across: paraphrases, different
subjects/same relation, different relations/similar wording, unrelated
contexts, adversarial contexts, long-context variants. Measures: activation
stability, contribution stability, semantic-effect stability, required
coalition stability. Purpose: decide whether the classifiable unit is
"feature / feature × task / feature × context / feature coalition." Prior
stated in the proposal itself: "a static per-feature label will be too
crude. The useful unit may be a **feature event** or **feature-plus-
coalition pattern**."

## LA-5 — Layer-path repeatability

Tracks `feature/expert at layer L → residual direction → feature/expert
activated at layer L+1 → eventual output effect`, then tests path
recurrence across paraphrases. Measures: exact feature-path overlap,
expert-path overlap, subspace overlap, causal path persistence after
upstream ablation, path restoration after injection. Determines whether
VINDEX2/LQL graph nodes should be individual features, expert groups,
feature coalitions, subspaces, or trace-conditioned events.

## LA-6 — Pre-fetch predictability (the decisive sparsity experiment)

> Using only information available **before reading the full FFN bank**,
> predict whether the current layer event is lookup-like enough for
> aggressive sparsity.
>
> Candidate signals:
> - residual-state similarity to known lookup states;
> - router entropy;
> - top-K routing margin;
> - gate-vector nearest-neighbour margin;
> - predicted contribution concentration;
> - task-independent residual geometry;
> - previous-layer path state.
>
> Do not train or fine-tune the model. Start with preregistered
> deterministic rules such as:
> ```
> high nearest-key margin
> AND low router entropy
> AND stable path match
> → sparse lookup mode
> ```
>
> Compare:
> 1. **Baseline:** full execution.
> 2. **Universal sparse:** same K everywhere.
> 3. **Oracle selective:** uses the post-hoc measured lookup score.
> 4. **Predictive selective:** uses only pre-fetch signals.
> 5. **Fail-safe selective:** predictive policy, but falls back when
>    confidence is low.
>
> The difference between oracle selective and predictive selective tells
> us whether the distinction can actually solve serving sparsity.

No explicit numeric pass/fail threshold for LA-6 itself — success is
relational (does predictive ≈ oracle?). The router-entropy / routing-margin
signals are MoE-specific language; on a dense model (Gemma 3 4B) these need
a dense analogue (e.g. the gated-FFN's gate-activation margin) rather than
a literal router.

**Granularity note (resolved 2026-08-02):** the K in "same K everywhere" /
the aggressive-K sweep is **feature-level** (individual neurons within an
FFN's intermediate activation) — the same unit as LA-1's original TinyModel
census (`c_i = activation_i × down_vector_i`), K = 1/2/4/8/16. This is a
different, finer granularity than the whole-layer-skip unit every Gemma
script since GLA-0 has used (include/exclude an entire FFN block at one
position) — that whole-layer work does not by itself satisfy LA-6/LA-7's
spec, though its baseline/universal-sparse/oracle-selective data points
remain useful context at the coarser granularity.

**STATUS: CLOSED 2026-08-02.** Run on Gemma-3-4B (`gla6_*.py`,
chuk-experiments `EXP-20260801-232758-00633`, `vindex2-la` programme).
Verdict, combined with the independent real-kernel result
[[project_r4_zeroout_sparse_ffn]] (R4, same day): dynamic feature/row/
block-selective FFN execution is refuted as a throughput lever — scattered
selection has no physical locality (29-89x physical/logical byte ratio
under block-quantized storage), block-constrained selection recovers real
byte accounting but tops out ~5-8% savings at fidelity-preserving budgets,
and the adaptive pre-fetch classifier is *worse* than a trivial static
layer ranking under composition. R4 independently confirmed the kernel/
decode-loop side: the sparse kernel captures only 23-40% of its
theoretical row reduction and the real decode loop runs ~0.15x baseline
despite the isolated kernel being 1.29x faster. See LA-6's chuk-experiments
write-up for the full combined synthesis. **Not closed:** an
offline-compiled, cross-input-stable contiguous representation — neither
LA-6 nor R4 tested this, it remains a distinct open lane.

## LA-7 — Selective sparsity execution

> For predicted lookup-like events, sweep aggressive K:
> ```
> features: 1, 2, 4, 8, 16
> experts: 1, 2, 4, routed top-K subset
> ```
>
> For approximation-like or uncertain events, retain baseline execution.
>
> Measure:
> - bytes read per token;
> - useful bytes;
> - page faults;
> - kernel work;
> - end-to-end tok/s;
> - teacher-forced BPB;
> - KL distribution;
> - token agreement;
> - task accuracy;
> - p99 catastrophic failures;
> - long-generation stability.
>
> ### Promotion gate
>
> A candidate selective policy should require:
> - no more than 0.5% BPB regression;
> - no pathological p99 KL tail;
> - no systematic task-family collapse;
> - no routing instability cascade;
> - reproducible whole-system byte reduction;
> - meaningful end-to-end speedup, not only kernel savings.
>
> I would set an initial success bar of:
> ```
> ≥25% routed-weight byte reduction
> with
> ≤0.5% BPB regression
> ```
>
> Anything below that may be scientifically interesting but does not yet
> solve K3 serving.

This is the one experiment in the ladder with an explicit numeric bar.
Note the standing constraint this interacts with:
[[feedback_reduction_is_a_kernel_claim]] — "bytes read per token,"
"page faults," "kernel work," and "end-to-end tok/s" are real serving-
kernel measurements, not something a dense forward-pass-masking harness
(the MLX approach every GLA script uses) can produce on its own. Whether
LA-7 needs real kernel integration or can proceed on counted/simulated
bytes as an interim proxy is a scoping decision to make explicitly when
LA-7 starts, not something to assume silently.

LA-7 explicitly depends on LA-6 (`predicted lookup-like events` — LA-7
sweeps K on events LA-6's classifier flags): it cannot run meaningfully
before LA-6 produces a working predictive policy.

**STATUS: CLOSED 2026-08-02, alongside LA-6.** The promotion gate
(≥25% routed-weight byte reduction, ≤0.5% BPB regression) was tested
against a depth×block×budget sweep and never reached at fidelity-preserving
settings (best: ~7.8% savings at 91% one-step top-1); the independent real-
kernel result (R4) additionally shows the mechanism regresses real decode
throughput regardless. See LA-6's status note above and its chuk-experiments
record for the full combined verdict — this closes LA-7 as specified for
the dynamic form, not the still-open offline-compiled-contiguous lane.

## LA-8 — Held-out model

> Freeze the metrics and deterministic policy on the discovery models.
> Then run unchanged on a held-out model.

Allowed changes: tensor-name adapter, layer geometry, expert count,
existing activation-function handling. **Forbidden:** changing lookup
thresholds, changing the score definition, adding a model-specific feature
class, hand-labelling layers. "If it generalises, the distinction is a
candidate LQL/VINDEX concept. If it fails, keep it as model-specific
analysis rather than putting it into the ABI."

---

## VINDEX2 (format) consequence — the actual gate

> Only if LA-6 and LA-7 succeed should VINDEX2 gain representation for:
> lookup-likeness evidence; trace-conditioned graph edges; feature
> coalitions; selective execution hints.

This matches `docs/vindex2-la.md` §4's standing constraint verbatim. Until
both succeed, nothing in this programme licenses a graph node type, a
trace-conditioned edge, or a selective-execution hint in the format ABI.
