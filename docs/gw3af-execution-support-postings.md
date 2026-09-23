# GW-3A-F — gate postings as an FFN execution-support index

**Status:** complete; progression failed, 2026-09-21  
**Cohort:** the 231 reconstructed FFN transition-candidate sites in sealed GW-0B  
**Claim boundary:** observational support recovery, not semantic or causal recall

GW-0B separates three objects that earlier WALK work could conflate:

```text
semantic neighbourhood != WALK candidate set != execution support
```

The working thesis is:

> Gate geometry provides a useful semantic index, but executable transitions,
> rather than nearest FFN features, are the stronger candidate for the model's
> native graph edges.

GW-3A-F asks a narrower physical-index question: can non-learned gate postings
address the FFN features that observationally contributed to a recorded
transition candidate? A positive result earns postings a role as an addressing
mechanism. It does not identify an FFN feature as the logical edge.

## Frozen authorities and cohort

The machine-readable contract is
[`bench/gw0/gemma3-4b-it-phase1/gw3af-preregistration.json`](../bench/gw0/gemma3-4b-it-phase1/gw3af-preregistration.json).
It binds:

- sealed GW-0 bundle `sha256:ac9871abdaef1206298788fae15df9121542fb1f64496cccc07bc12b5751b6bf`;
- sealed census file `sha256:9db9f925d314554d03f6bae0e019694525b1840ddc7cfca22953e3b9885bbd3a`;
- final GW-0B reconciliation report
  `sha256:10c7af77077914c37b569d6a1c29ab06a8bec676c1e9686fe406c70873e956bc`;
- the report-bound production-effective attribution files; and
- the frozen Gemma 3 4B input, tokenizer, container, and edge-family splits.

The primary unit is a reconstructed `(edge_id, layer, ffn)` observation site.
There are 231 sites. Each has one checked reconstruction and exactly 100 ranked
feature contributions from an eligible universe of 10,240 addresses. The 195
attention-only rows are ineligible exclusions. They are not execution misses.
All causal statuses are `untested`; supported-causal recall therefore has no
denominator in this experiment.

The source-key vocabulary is the deduplicated tokenization of every frozen
semantic subject with the container tokenizer, `add_special_tokens=false`.
Held-out subjects may appear because index construction consumes no edge,
relation, target, split, attribution, or outcome labels. Each query uses only
the tokens of its own subject. The vocabulary, tokenizer identity, and exact
derivation rule are frozen before lookup generation.

## Addressing arms

For each source token and layer, rank features by
`abs(feature_gate dot scaled_token_embedding)`, breaking equal scores by feature
index. A row unions the postings for its distinct subject tokens. The width is
the number retained **per source token per eligible layer**.

The width ladder is exactly:

```text
1, 2, 4, 8, 16, 32, 64, 128, 256
```

No interpolated width, post-hoc width, or tuned relation-specific width may
enter the progression decision.

The progressive arms are:

1. `unmasked`: all container FFN layers are eligible;
2. `gw1-mask`: the independently frozen relation-only mask, if a valid GW-1
   result exists before lookup generation.

Absence of a frozen GW-1 mask omits the second arm; it does not permit deriving
a mask from GW-3A-F outcomes. The `oracle-layer` arm admits only the observed
FFN site layer and is descriptive. It cannot satisfy the progression gate.

## Metrics and denominators

For site `i`, let `S_i` be its 100 recorded contribution addresses, `C_i(w)`
the unique candidates returned at width `w`, and `U_i` the addresses in the
arm's eligible layers.

```text
address_recall_i(w) = |S_i intersect C_i(w)| / |S_i|
candidate_coverage_i(w) = |C_i(w)| / |U_i|
```

`|U_i|` is computed from the actual feature count of each eligible layer for
that row. It is never the model-global feature count when a mask admits a
smaller surface. Candidate count and coverage are computed after union and
deduplication.

Let `m_ij` be the recorded `contribution_l2` for address `j` in `S_i`:

```text
top100_mass_recall_i(w) =
    sum(m_ij for j in S_i intersect C_i(w)) / sum(m_ij for j in S_i)
```

This is **top-100 contribution-L2 mass recall**. The attribution does not store
the remaining 10,140 contribution norms, so the result must never be described
as total-write mass recall.

The report includes per-site values, micro address recall, address-weighted
candidate coverage, macro site means, and breakdowns by relation, prompt
family, split, and edge family. The progression statistic is micro top-100
address recall versus address-weighted eligible-universe coverage. Mass recall
is a co-primary interpretation surface and cannot replace address recall.

The interpretation is frozen as:

| Address recall | Top-100 mass recall | Interpretation |
|---|---|---|
| high | high | stable support set recovered |
| low | high | dominant computation recovered; support identity is unstable |
| high | low | many addresses recovered; important contributions are missed |
| low | low | postings fail as an execution-support index |

“High” in this table means the predeclared 95% region. Reports retain the full
curves and do not convert the table into a new fitted classifier.

## Controls

Every postings point has three controls.

**Exact-WALK matched candidate count.** Recompute the dense exact-WALK ranking
from the frozen mean subject embedding. Within the identical eligible layers,
retain exactly `|C_i(w)|` unique addresses. This tests whether postings add
information beyond spending the same candidate budget on the reference gate
ranking. Saved top-20 hits alone are insufficient for widths that need more
addresses; this control must use the bound container and exact query rule.

**Layer and token-frequency matched null.** For semantic-target recovery
descriptions only, keep the exact-WALK address set and its layer profile fixed,
then replace each target token with deterministic non-target decoys having the
same empirical frequency in the sealed prompt tokens plus target continuation
tokens. The corpus, trial count, and seed are in the preregistration. This does
not enter execution-support recall.

**Layer-matched random addresses.** Draw the same number of addresses as the
oracle-layer postings arm from the observed FFN layer. This is a descriptive
address baseline with a frozen seed and cannot satisfy progression.

**Oracle layer.** Restrict postings to the observed FFN attribution layer and
repeat the width ladder. This isolates addressability once location is known.
It is labelled `descriptive_non_progressive` everywhere.

Candidate features, dot products, gate bytes, other bytes, index bytes, and
wall time remain separate accounting surfaces. Offline index construction is
reported separately from lookup cost.

## Progression rule

GW-3A-F passes only if a non-oracle arm reaches:

```text
micro top-100 execution-support address recall >= 0.95
address-weighted eligible-universe candidate coverage <= 0.10
```

at one of the nine frozen widths. The complete curve and the matched-candidate
control remain part of the result. A high mass-recall point with address recall
below 0.95 is reported according to the frozen interpretation table and does
not pass this gate.

## Semantic descriptions and the eight aligned rows

The existing 81/142 observation is reported only as:

> 81/142 unique edges had at least one promoted semantic target recovered
> somewhere in the searched exact-WALK candidate set (up to 680 features per
> edge, with eight promoted tokens per feature).

It is not called semantic recall before the matched null is available.

The eight GW-0B rows where semantic target, exact WALK, and execution support
all agree form a descriptive audit cohort. They cannot define thresholds or
enter progression separately. The audit records:

- relation, layer/depth, prompt family, final target rank and log probability;
- contributor concentration (Herfindahl index over normalized top-100
  `contribution_l2`);
- top-20 WALK score entropy, using normalized absolute gate scores;
- largest-contribution dominance (largest top-100 `contribution_l2` divided by
  their sum); and
- the number of distinct subject-token posting lists that independently hit at
  least one recorded execution-support address.

## Architectural interpretation

The result is interpreted under two separate abstractions:

```text
TransitionIdentity
    source semantic state
    relation / operation
    destination semantic state
    layer / site context

TransitionSupport
    FFN: feature/address contributions
    attention: head + source-token contributions
    MoE: expert contributions
    recurrent/state: operator-specific state contributors
```

WALK, postings, head attribution, and future operator indices locate or explain
support. They do not independently define the transition identity.

## Result

The sealed postings artifact contains all 93 unique subjects at all nine widths
and is bound to preregistration
`sha256:bf4f773521d3bba6d52498e943caaf75531cb048325618310b798858bcc4df1e`:

```text
postings.jsonl      sha256:e4e702a9e42eea5d9b0eb8d4f5813298ce3c30935e3a6a432edd47cf3fae21cf
report-sealed.json  sha256:0539b515953d93a2707261578069b712a32193e7a3ca7a6d15972ef3dd58e0e5
```

No frozen width reaches the progression region. The primary top-100 curve is:

| Width | Address recall | Top-100 L2-mass recall | Address-weighted coverage |
|---:|---:|---:|---:|
| 1 | 0.44% | 2.37% | 0.013% |
| 2 | 0.73% | 3.18% | 0.025% |
| 4 | 1.13% | 4.10% | 0.051% |
| 8 | 1.85% | 5.60% | 0.101% |
| 16 | 2.67% | 6.94% | 0.202% |
| 32 | 3.91% | 8.82% | 0.404% |
| 64 | 5.30% | 10.67% | 0.807% |
| 128 | 7.28% | 13.17% | 1.610% |
| 256 | 9.79% | 16.63% | 3.210% |

At width 256, the matched exact-WALK control obtains the same 9.79% address
recall and 16.70% mass recall at the same coverage. Layer-matched random obtains
3.23% and 2.84%. Gate geometry therefore enriches execution support above
random, especially toward large contributors, but the postings construction
does not improve on a candidate-count-matched dense gate ranking and remains an
order of magnitude below the address-recall gate. The frozen interpretation is
low address recall plus low mass recall: postings fail as an execution-support
index at these declared widths.

The oracle-layer arm has the same recall and fractional coverage because the
width rule and feature count are uniform per layer. It removes candidates from
the other 33 layers, reducing absolute lookup work, but does not make the target
layer's support more addressable. It remains non-progressive.

Relation breakdown at width 256 is descriptive: top-100 address recall is
14.22% for currency, 11.72% for capital, 8.95% for language and 3.36% for
hypernym. Prompt-family values are much tighter (9.54–9.97%), and test recall
(10.64%) does not fall below train (9.76%). No relation, family or split defines
a new gate.

The semantic calibration supports a separate conclusion. Nine Euro edges have
no non-target token with the exact same sealed-corpus frequency and are excluded
by the frozen null. Among the remaining 133 unique edges, the target occurs in
the searched exact-WALK promotions for 73, versus a null mean of 19.95, null
maximum 31 over 1,000 trials, and one-sided empirical `p = 0.000999`. This
confirms that the 81/142 occurrence result contains real semantic structure; it
does not turn that occurrence statistic into calibrated recall or execution
support.

The eight fully agreeing rows are not uniformly single-feature computations.
Their top-100 contribution Herfindahl indices range from 0.013 to 0.042 and
their largest-contributor shares from 3.5% to 18.0%. Four are independently hit
by a source-token width-1 posting and seven by width 2; the two-token hypernym
row is hit by both token postings from width 32 onward. This is descriptive and
selection-conditioned, but it points to unusually strong addressability rather
than a general collapse onto one dominant FFN feature.

## Commands

Validate the frozen contract before generating candidates:

```bash
python3 scripts/gw3af_preregister.py \
  bench/gw0/gemma3-4b-it-phase1/gw3af-preregistration.json

target/release/examples/observatory_record --gw3af-postings \
  /Users/christopherhay/chris-models/gemma3-4b-it.vindex3 \
  output/gw0-gemma3-4b-it-phase1/sealed/manifest.json \
  bench/gw0/gemma3-4b-it-phase1/gw3af-preregistration.json \
  output/gw3af-gemma3-4b-it-phase1/postings.jsonl

python3 scripts/gw3af_execution_postings.py \
  --preregistration bench/gw0/gemma3-4b-it-phase1/gw3af-preregistration.json \
  --manifest output/gw0-gemma3-4b-it-phase1/sealed/manifest.json \
  --input-manifest bench/gw0/gemma3-4b-it-phase1/manifest.json \
  --attributions output/gw0b-gemma3-4b-it-phase1/attributions-final \
  --promotions output/gw0b-gemma3-4b-it-phase1/feature-promotions.jsonl \
  --postings output/gw3af-gemma3-4b-it-phase1/postings.jsonl \
  --output output/gw3af-gemma3-4b-it-phase1/report-sealed.json
```
