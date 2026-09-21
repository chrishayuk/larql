# GW-TS-1 programme dependency freeze

**Status:** frozen before population capture, 2026-09-21

**Bound measurement protocol:**
`sha256:d873d238d99406dc8a549681aa746e750a9d4b802e24ff21a8eb26d152a97d55`

**Machine contract:**
[`bench/gw-ts-1/programme-dependencies.json`](../bench/gw-ts-1/programme-dependencies.json)

This addendum freezes programme orchestration without changing the GW-TS-1
ontology, extraction rule, controls, gate or claim boundary. The observational
and causal tracks share frozen coordinates but neither selects evidence for the
other.

## Dependency graph

```text
HEAD-OBS-1
    ↓
ATTR-1D
    ├── Run experiment subjects
    │       ↓
    │   population TransitionSupport capture
    │       ↓
    │   GW-TS-1
    │   reproducibility + held-out next-support prediction
    │
    └── frozen observational coordinates
            ↓
INTERVENE-1 ──→ ATTR-1C
                 necessity / sufficiency
```

ATTR-1D is a prerequisite of both branches because it defines descriptive
head/source support on the canonical observation record. Run experiment subjects
are a prerequisite for population capture and GW-TS-1 assessment. INTERVENE-1 is
a prerequisite for ATTR-1C only.

ATTR-1C is **not** a prerequisite for population capture, path extraction,
recurrence or prediction. Intervention outcomes, causal labels and event rankings
derived from interventions are forbidden inputs to every GW-TS-1 discovery and
assessment surface.

## Join rule

ATTR-1C may test an event only by joining to the coordinate identity already
defined by ATTR-1D and the frozen GW-TS-1 observational rule. It may append causal
evidence under that coordinate. It cannot add, remove, merge, reorder or relabel
an observational support event, and it cannot update a path prototype, threshold,
alignment or candidate vocabulary.

Likewise, GW-TS-1 recurrence or prediction cannot select the coordinates that
ATTR-1C intervenes on for the primary causal claim. Any later prediction-guided
intervention study is a new, separately frozen successor experiment.

## Promotion rule

The tracks make independent claims:

```text
GW-TS-1   the path is reproducible and predicts what comes next
ATTR-1C   the independently defined support events causally matter
```

An executability rung may open only after independent positive verdicts from both
tracks. It must then freeze its own skip/route/prefetch/substitute operation,
correctness floor, calibration rule and measured-work gate. Neither positive
alone permits replacing normal computation.

## Change rule

This dependency artifact is separate from the hash-bound GW-TS-1 measurement
protocol. Later orchestration changes receive a new dependency identity and an
explicit amendment; they do not rewrite protocol `d873d238…` or feed causal
evidence backward into its observational definitions.
