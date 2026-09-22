# GW-READ-1 results — cheap Q/K, unresolved V and carrier context

**Verdict:** `mechanistically_compact_but_not_economical_read_path`

**Protocol identity:**
`sha256:92947e60d1e831183346b2336053e1d10990c7bfb0c65f3fe31f4858b0cfc0ee`

**Selection identity:**
`sha256:fac2eeb935621f0e76833ee12879ac828ac5537cb2113ca8e29cc46f8e27afe3`

**Adjudication identity:**
`sha256:a9154e18942ed407dc1da6b672e5976331c3f07f1b55baf1728cfc24760f75bb`

GW-READ-1 tested economical construction over the frozen GW-KEY-1 causal
interface. It did not search for a new head, source role or semantic surface.
Train alone fit and selected Q, K and V constructions. Validation and test saw
only the sealed component identities, followed by a fixed composition.

## Frozen train selection

| Component | Selected construction | Train raw retention | Train z retention |
|---|---|---:|---:|
| Q | cached template×relation mean | 0.997 | 1.006 |
| K | cached global role×ordinal scaffold | 0.974 | 0.922 |
| V | L0 rank-32 predictor | 0.930 | 0.849 |

The implied candidate-only shared frontier was L0: Q and K were cached, while V
required one original-prompt layer plus a rank-32 map. That was a train-only
hypothesis, not a performance result.

## Held-out component fidelity

The table reports factual-relation proximal effect retention. Each component
also required positive terminal transport. Q and K passed the frozen component
gate on both splits; V did not.

| Component | Validation raw | Validation z | Test raw | Test z | Gate |
|---|---:|---:|---:|---:|---|
| Q | 0.998 | 0.998 | 1.003 | 1.002 | pass |
| K | 0.866 | 0.843 | 0.923 | 0.855 | pass |
| V | 0.690 | 0.563 | 0.782 | 0.785 | **fail** |

Q is the cleanest systems result. A train-derived template×relation query mean
preserved essentially the entire held-out effect and transported through later
layers. Natural L24 query construction was therefore not necessary for this
alignment effect.

K also became static. A global role×ordinal scaffold—without target-row K or
target-natural norm reads—retained the preregistered fraction on validation and
test. This supports a reusable routing scaffold, although role assignment and
lookup costs remain real inputs.

V did not generalize at the frozen threshold. Its L0 rank-32 predictor retained
only 56% of validation z-scored effect and 78% on test. Later layers softened
the damage: terminal V retention was 0.936/0.818 on validation and 0.984/0.822
on test for raw/z, with positive effects throughout. Terminal transport cannot
rescue failure of the inherited proximal semantic-alignment estimand.

## Composition

The unchanged selected Q/K/V constructions were composed without tuning.

| Split | Raw retention | Z retention | Quality gate |
|---|---:|---:|---|
| Validation | 0.631 | 0.482 | **fail** |
| Test | 0.737 | 0.681 | **fail** |

Composition transported positively to terminal logits, but component errors
interacted at the proximal surface. The fixed 2×2×2 factorial was diagnostic
only; no alternate combination was selected after seeing held-out data.

READ-1E also failed the independent systems gate. Causal replay still consumed
the natural L24 carrier-before state and all seven non-H1 head outputs. Those
inputs require full pre-L24 execution and were explicitly charged rather than
being hidden behind the small Q/K/V tensors. Consequently no latency or avoided
work claim is admissible, even if composition quality had passed.

## Supported interpretation

> **The L24H1 query and routing scaffold are cheaply replaceable for the
> measured alignment effect, but the entity payload is not yet cheaply
> constructible at the frozen fidelity threshold, and H1 still depends on an
> expensive natural carrier context.**

GW-READ-1 therefore narrows the next problem in two independent directions:

1. improve payload construction without held-out retuning—between L0 and the
   natural L24 entity V, or through a reusable materialized entity payload;
2. establish what carrier seed and non-H1 context are actually necessary for
   H1 transport, rather than assuming the isolated Q/K/V calculation can enter
   later execution for free.

This is not efficient WALK. It is stronger than a generic negative result:
lookup-key construction and routing caching passed, while payload
materialization, component separability and carrier-context independence are
now the localized failures.

## Authoritative artifacts

- protocol: `bench/gw-read-1/gwread1-protocol.json`
  (file SHA-256 `1c6824395535ff7392717dcc02b8cbde8ce50554ea015df079f1a6adb13d3b26`)
- outcome-free capture:
  `output/gwread1-gemma3-4b-it-phase1-capture/candidate-input-capture-manifest.json`
  (`b9970412bbfad86ba702b43df701b75f88c877b6930de35a14866184ae887df7`)
- fitted candidate artifact:
  `output/gwread1-gemma3-4b-it-phase1-fit3/candidate-artifact-manifest.json`
  (`8ef0729988dedce936feba45bba0007f124fdf9288378e0d40fd638532d42919`)
- train replay:
  `output/gwread1-gemma3-4b-it-phase1-train3/train-candidate-replay-manifest.json`
  (`3ec45e7791a39c6bbb2154c412340c9da7b4f00bab89736740f9e844daed0b9b`)
- train metrics and selection:
  `output/gwread1-gemma3-4b-it-phase1-train3/train-metrics-v2.json`,
  `output/gwread1-gemma3-4b-it-phase1-train3/selection-v2.json`
- held-out replay:
  `output/gwread1-gemma3-4b-it-phase1-heldout/heldout-replay-manifest.json`
  (`9d2c46d5471e442ccf45a894fc7c13bc489a5477ea7ea44e5d6a3c027c826fb5`)
- adjudication:
  `output/gwread1-gemma3-4b-it-phase1-heldout/adjudication.json`
  (`4ef0d9962172293c609efa05b2d19ac8d4671c55523dc25e3723113b2bc92231`)

The earlier `fit`, `fit2`, `train`, `train2`, unversioned metrics and unversioned
selection artifacts are superseded pre-held-out diagnostics. The authoritative
chain is `fit3` → `train3` → `train-metrics-v2`/`selection-v2` → `heldout`.
