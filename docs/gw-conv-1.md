# GW-CONV-1 — convergence-operator localization

**Status:** adjudicated; frozen conjunctive gate passed, 2026-09-21  
**Cohort:** 142 semantic edges, 426 executions; train/validation/test = 85/29/28 edges  
**Claim boundary:** observational write localization, not causal head attribution or a discrete graph edge

GW-CONV-1 asks which observed writes make three prompt-family trajectories for
the same fact more alike than frozen relation-matched, different-fact controls.
It follows the negative GW-SUP-1 transition-superposition result: no semantic
beam or collapse premise enters this experiment.

The frozen preregistration identity is
`sha256:03a01d91bedd69a3ca825c5857a37f796daed088dd557582bbb049cd23fcf8ea`.
The adjudication identity is
`sha256:398781476f37833a1cefd4a99d649e4e197fcbfd1e91771314a81f77c793b491`.

## Result

The train-only global optimum and all four independently selected relation
optima are the same write:

```text
site 48 = layer 24 attention
```

No validation or test outcome selected or moved that site. At the fixed site,
all three preregistered adjusted-gain intervals are above zero on both held-out
splits:

| Surface | Validation adjusted gain, 95% CI | Test adjusted gain, 95% CI |
|---|---:|---:|
| Carrier cosine | 0.00262 [0.00233, 0.00291] | 0.00227 [0.00195, 0.00259] |
| Raw candidate JS | 0.328 [0.274, 0.376] | 0.312 [0.273, 0.348] |
| Z-scored candidate JS | 0.0519 [0.0432, 0.0606] | 0.0345 [0.0278, 0.0411] |

The global site has positive mean adjusted raw candidate gain in all four
relations on validation and test. Relation-conditioned routing also passes,
although it does not produce four distinct routes: capital, currency,
language, and hypernym each select L24 attention from their training rows.
Consequently the frozen global-operator, relation-routing, cross-relation, and
full-conjunction conditions all pass.

The raw training profile is sharply localized. L24 attention contains 66.0% of
the positive adjusted raw-candidate mass; its positive-mass effective site
count is 2.24. This is a descriptive training surface, not an additional gate.
The fixed site has median signed distance zero from held-out shared emergence,
but the distribution is broad: it lies at or before emergence for 59% of
validation and 68% of test edges. Hypernym's much earlier emergence is a major
reason not to equate this global site with every relation's answer-emergence
mechanism.

## Essential qualification: adjusted is not absolute

The preregistered primary quantity is

```text
adjusted gain = same-fact gain - matched-control gain
```

The components prevent a misleading “residual collapse” interpretation.

On test, same-fact raw candidate JS genuinely falls from 0.225 before the L24
attention write to 0.183 after it, a gain of 0.0416. The z-scored view also
converges, by 0.00576. Matched controls instead diverge strongly: raw JS rises
from 0.263 to 0.533 and z-scored JS from 0.0535 to 0.0822. Validation has the
same pattern.

Carrier geometry is different. Same-fact carrier cosine on test moves slightly
down, from 0.991581 to 0.991514 (gain -0.000067), while matched controls fall
much more, from 0.990755 to 0.988416 (gain -0.002339). The positive adjusted
carrier result therefore means **relative preservation/separation**, not
absolute carrier-vector convergence.

The supported interpretation is accordingly narrow:

> L24 attention is a reproducible site where prompt-family states become more
> alike in the frozen semantic readout while different-fact controls separate;
> the full carrier vectors do not collapse onto one another there.

This is stronger than confidence sharpening because the z-score control agrees,
but weaker than an attractor claim over the residual state.

## Post-adjudication examples

Only after sealing the aggregate result, one test edge per relation was selected
as the edge closest to that relation's median adjusted raw gain. The examples
are stored separately and do not enter site selection or adjudication.

- Brazil/language converges cleanly toward Portuguese across all three prompt
  families (raw JS 0.0274 to 0.00350).
- Ghana/currency moves all prompts toward Taka, not the annotated Cedi target,
  while still becoming modestly more mutually similar (0.186 to 0.164). This
  repeats GW-SUP-1's warning that convergence need not be target correctness.
- Afghanistan/capital makes Kabul top-ranked in all three prompts, but the full
  restricted distributions become marginally less similar (0.0666 to 0.0678);
  its positive adjusted result is control-relative.
- The representative hypernym case remains highly prompt-specific despite a
  small JS decrease (0.560 to 0.557).

A post-hoc relation summary likewise shows that test same-fact raw convergence
is not uniform: mean gains are -0.0117 capital, 0.1599 currency, 0.0107
language, and 0.0080 hypernym. The preregistered cross-relation gate concerned
adjusted gain, so these numbers qualify rather than alter its pass.

## Consequence for WALK and HEAD

GW-CONV-1 does not restore nearest-feature WALK and does not unlock semantic
beam preservation. It nominates a stable operator/site context:

```text
prompt-conditioned carriers
          ↓
    L24 attention write
          ↓
semantic-readout alignment + different-fact separation
```

Because the selected site is attention in every frozen training selection,
HEAD decomposition is now the justified next localization rung. It should ask
which heads and source-token reads account for the L24 write, using the
post-attention-norm-aware reconstruction contract. GW-CONV-1 itself identifies
no head and supplies no causal evidence.

The result also refines the operator-aware transition-support programme: start
with the nominated L24 attention event, then determine whether its head/source
support is stable across prompts and relations. If it is, WALK may route to
convergence machinery rather than to a feature or a semantic beam.

## Evidence and commands

The before-write readout used the production-selected prepared Q8 output head.
Its selected 126 rows were bit-identical to the corresponding full-head rows on
the parity witness. No prompt was re-executed; all 426 before carriers came from
the sealed GW-0 corpus, and the existing immutable GW-SUP-1 artifact supplied
the after-write readouts.

```bash
python3 scripts/gwconv1_preregister.py validate \
  bench/gw0/gemma3-4b-it-phase1/gwconv1-preregistration.json

target/release/examples/observatory_record --gwconv1-readout \
  /Users/christopherhay/chris-models/gemma3-4b-it.vindex3 \
  bench/gw0/gemma3-4b-it-phase1/gwconv1-preregistration.json \
  bench/gw0/gemma3-4b-it-phase1/gwsup1-candidates.json \
  output/gw0-gemma3-4b-it-phase1/sealed/manifest.json \
  output/gwconv1-gemma3-4b-it-phase1

python3 scripts/gwconv1_adjudicate.py \
  --preregistration bench/gw0/gemma3-4b-it-phase1/gwconv1-preregistration.json \
  --candidates bench/gw0/gemma3-4b-it-phase1/gwsup1-candidates.json \
  --before-readout-manifest output/gwconv1-gemma3-4b-it-phase1/before-readout-manifest.json \
  --after-readout-manifest output/gwsup1-gemma3-4b-it-phase1/readout-manifest.json \
  --sealed-manifest output/gw0-gemma3-4b-it-phase1/sealed/manifest.json \
  --output output/gwconv1-gemma3-4b-it-phase1
```

Primary artifacts are
`output/gwconv1-gemma3-4b-it-phase1/adjudication.json`,
`site-profiles.json`, `edge-metrics.jsonl`, and the explicitly post-hoc
`posthoc-examples.json`.
