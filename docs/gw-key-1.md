# GW-KEY-1 — causal source roles through L24H1

**Status:** compact joint candidate read path supported, 2026-09-21  
**Fixed operator:** target component, zero-based L24H1  
**Claim boundary:** source-role causality; not query construction, correctness,
partial-execution cost, an access primitive, or WALK

GW-KEY-1 asks one question:

> Does a compact, frozen set of source roles causally supply the information
> through L24H1 that produces the GW-HEAD-1 semantic-alignment effect?

The frozen preregistration identity is
`sha256:87e80308a9cc684f875a0b16684f3cf2965e3e22b34a50338273c4e1935732d8`.
The 426-row role map is
`sha256:1efcc11b5f6afbf3a1b8e69344a1386c8d304499fa5f01b31cc95b0f3c5296be`.
Neither artifact uses attention weights, activations, or outcomes.

## Result

Train-only exhaustive execution selected one compact joint intervention:

```text
natural final-position Q
        +
all source K replaced by train/template/role references
        +
natural subject_entity V only
        ↓
frozen L24H1 and real downstream execution
```

The joint union cardinality is one source role. It retained 99.2% of the raw
candidate-JS effect and 100.1% of the z-scored effect on train. The independently
selected V-only arm also retained only `subject_entity` V, with natural K.

The frozen joint arm passed necessity and sufficiency for both semantic metrics
on validation and test:

| Split | Metric | Necessity, 95% CI | Sufficiency, 95% CI |
|---|---:|---:|---:|
| validation | raw candidate JS | 0.320 [0.268, 0.366] | 0.970 [0.957, 0.980] |
| validation | z-scored candidate JS | 0.0490 [0.0404, 0.0573] | 0.942 [0.920, 0.964] |
| test | raw candidate JS | 0.314 [0.273, 0.351] | 0.981 [0.960, 1.005] |
| test | z-scored candidate JS | 0.0348 [0.0278, 0.0414] | 0.972 [0.948, 1.002] |

Carrier qualification agrees: joint sufficiency is 0.933 on validation and
0.940 on test, with positive necessity on both. Terminal-logit transport also
passes the same necessity/sufficiency thresholds. All natural and identity
parity checks were bit-exact, and all 1,197 held-out interventions fired once.

The supported claim is:

> A frozen compact source role—subject/entity V—is jointly necessary and
> sufficient through L24H1 for the held-out global relation-stratified,
> control-adjusted semantic-alignment effect under train-derived template/role
> reference routing.

This earns **candidate read path**, not efficient WALK.

The global gate is relation-stratified but does not imply four separate
relation passes. Capital, currency, and language show strong descriptive
subject-V recovery. Hypernym is weak: its validation raw denominator is only
0.00318, and both test denominators are negative, so relation-level sufficiency
refuses there; test raw necessity is also slightly negative. This is consistent
with the earlier descriptive H4/hypernym split and must be replicated on a
larger corpus before any shared four-relation mechanism claim.

## Load-bearing K qualification

K-only subset selection refused on train. Its raw sufficiency denominator was
small and positive (0.000412), while its z-scored denominator was negative
(-0.00190): all-reference K slightly exceeded natural K on that train metric.
No K subset was frozen or exposed to held-out outcomes.

Therefore the result does **not** identify a compact natural K source subset.
The passing joint arm replaces every source K with train-derived means grouped
by template and role, so those reference K vectors still provide structured
routing competition. Natural final-position Q is also unchanged. The result
localizes the compact causal payload to subject/entity V; GW-READ-1 must still
determine whether the query and routing scaffold can be produced cheaply.

## Fixed causal graph

```text
natural final-position Q
        ↓
frozen L24H1
        ↓
source K roles ── score competition and softmax routing
source V roles ── delivered content under those weights
        ↓
effective prepared Q8 W_O
        ↓
real post-norm → residual → FFN → later layers
        ↓
inherited GW-HEAD-1 alignment estimand
```

Q stays natural in every arm. The other seven L24 heads stay natural. K
replacement occurs before scoring, so every K arm reruns score scaling,
softcapping where declared, and the complete softmax denominator. V replacement
occurs before weighted-V aggregation. Attention weights are observations, never
causal proxies.

## Outcome-blind role vocabulary

The vocabulary has six ordered roles:

1. `bos_system`
2. `subject_entity`
3. `relation_query`
4. `answer_cue`
5. `punctuation_separator`
6. `instruction_template`

Assignment is a fixed table over the 12 frozen prompt templates. Roles are
exhaustive and non-overlapping. Punctuation and residual instruction/template
material may be empty; the subject, relation, and answer-cue roles are always
populated. Multi-token subjects remain one abstract role. Exact token positions
are recorded beneath roles and cannot become search candidates.

Reference K/V directions are train-only means grouped by template and role.
Train search uses leave-one-semantic-edge-out banks; validation/test use the
single bank frozen from all train rows. Each reference source vector is scaled
to the corresponding natural vector's L2 norm. Degenerate cells refuse.

## Three separately frozen searches

- **K:** selected roles retain natural K; other K roles use references; all V
  stays natural.
- **V:** all K stays natural; selected V roles retain natural V; other V roles
  use references.
- **Joint:** K and V have independently selected natural-role subsets and
  reference complements.

K and V each have 64 subsets. Joint search evaluates all 4,096 K/V mask pairs
by direct H1 execution on train. Singleton scores are never summed. Eligibility
requires at least 80% train recovery independently for raw and z-scored
candidate JS. The deterministic selection rules minimize role cardinality
before retained effect, replaced-vector norm, and lexical mask order. A joint
route is compact only when the union of its K/V roles contains at most three of
the six roles.

## Held-out adjudication

For each of K, V, and joint interventions:

\[
N(S)=E(I_{full})-E(I_{full\setminus S})
\]

\[
R(S)=\frac{E(I_S)-E(I_{ref})}{E(I_{full})-E(I_{ref})}.
\]

The candidate-raw and independently z-scored GW-HEAD-1 metrics remain
conjunctive. Validation and test must separately have necessity lower 95%
bounds above zero and sufficiency lower 95% bounds at least 0.50. Intervals use
10,000 relation-stratified semantic-edge bootstrap samples with seed
`1398114331`. Carrier cosine is qualification only. Terminal logits are a
mandatory separate transport report.

Only a passing joint result earns **candidate read path**. It does not establish
that producing the natural L24 query is cheap; that question belongs to
GW-READ-1.

## Frozen artifacts

- `bench/gw0/gemma3-4b-it-phase1/gwkey1-preregistration.json`
- `bench/gw0/gemma3-4b-it-phase1/gwkey1-source-roles.jsonl`
- `scripts/gwkey1_preregister.py`

## Execution artifacts

- natural source capture:
  `output/gwkey1-gemma3-4b-it-phase1-capture5/source-capture-manifest.json`
- train replay and metrics:
  `output/gwkey1-gemma3-4b-it-phase1-search/`
- frozen selection:
  `output/gwkey1-gemma3-4b-it-phase1-search/selection-v2.json`
- held-out arms and adjudication:
  `output/gwkey1-gemma3-4b-it-phase1-heldout2/`

Frozen selection identity:
`sha256:eed14eb42f5432960677d47bd328a6b3b74777f542011f7b77b58469cfe69f6f`.
Adjudication identity:
`sha256:a554877b15b69467a6c4f5ecca9427669a1536ff5b9e88056000cf2bb78dd60a`.

The unreferenced `selection.json` and first `heldout/` directory are superseded
pre-handoff artifacts: their V-family selection record serialized the wrong K
mask for `full_minus_selected`, although the runner executed the correct arm.
`selection-v2.json` fixed the record from the same train surface, and
`heldout2/` reran and rebound every held-out artifact. The numeric tensors are
bit-identical across the two held-out executions; only the final identities
above are authoritative.

Build and validate without model execution:

```bash
python3 scripts/gwkey1_preregister.py build \
  --input-rows bench/gw0/gemma3-4b-it-phase1/input.jsonl \
  --head-selection output/gwhead1-gemma3-4b-it-phase1/selection.json \
  --head-adjudication output/gwhead1-gemma3-4b-it-phase1-heldout-full/adjudication.json \
  --roles bench/gw0/gemma3-4b-it-phase1/gwkey1-source-roles.jsonl \
  --output bench/gw0/gemma3-4b-it-phase1/gwkey1-preregistration.json

python3 scripts/gwkey1_preregister.py validate \
  bench/gw0/gemma3-4b-it-phase1/gwkey1-preregistration.json
```
