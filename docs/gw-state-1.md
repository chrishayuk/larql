# GW-STATE-1 — dependency removal around the frozen L24H1 interface

**Status:** FROZEN PROTOCOL — sealed before STATE-1 outcomes; replay not yet executed  
**Predecessors:** GW-HEAD-1, GW-KEY-1, GW-READ-1, GW-V2  
**Claim boundary:** surrounding-state necessity for the established H1 causal
effect; not payload reconstruction, correctness, partial-execution speed, or
efficient WALK

GW-STATE-1 asks one question:

> What surrounding state is actually necessary for the L24H1 intervention
> effect to remain causal?

The experiment holds the established H1 interface fixed and removes context.
It does not fit another V predictor. The two context families are the carrier
entering L24 attention and the seven non-H1 L24 head values. Their complete
factorial surface determines whether the causal unit is H1 alone, carrier plus
H1, a small local head circuit, or broad natural state.

## Why this follows GW-V2

The causal programme has narrowed in four steps:

| Rung | Supported result |
|---|---|
| GW-HEAD-1 | A single intervention interface, L24H1, carries most of the measured held-out effect. |
| GW-KEY-1 | Under the frozen routing scaffold, the retained natural payload localizes strongly to subject/entity V. |
| GW-READ-1 | Cached Q and K pass, while cheap V construction and economical composition fail; natural carrier and seven non-H1 heads remain charged inputs. |
| GW-V2 | No cell in the preregistered 35-cell depth/rank surface clears the held-out low-rank linear-reconstruction gate. |

The next uncertainty is therefore the size of the causal execution context,
not the location of another linear payload decoder.

## Frozen object and inherited population

The intended population is the sealed 666-execution GW-V2 factual cohort:
capital, currency, and language over train, validation, and test. Its row order,
semantic-edge clusters, prompt families, matched controls, candidate vocabulary,
raw and independently z-scored readouts, and terminal transport surface are
inherited byte-for-byte. Hypernym remains outside this rung.

Before execution, a machine-readable preregistration must bind the original
GW-V2 protocol, population, execution binding, exact-control result, model
container, plan, effective prepared weights, and the GW-HEAD-1/GW-KEY-1
interface identities. No GW-STATE-1 outcome may be read before that seal.

The intervention site remains the final prompt position at zero-based L24H1.
The treatment value is the row's exact natural H1 weighted-V head value before
its effective `W_O` slice. The H1 baseline is the corresponding value from a
frozen same-relation/different-subject donor. H1 is never predicted, refit,
rank-truncated, or chosen from GW-V2 cells.

## Dependency-removal values

Every target row already has an outcome-blind `same_relation_different_subject`
matched control in the frozen GW-V2 input rows. Across the sealed 666-row
population, that map is a one-to-one derangement: it stays within split,
relation, and prompt family, changes the subject on every row, and reuses no
donor. The executable protocol must inherit that exact map rather than
construct or optimize a new one. The separate GW-V2-AMEND-1 fit-sham donor map
is not this map: it covers only train token positions and has two fixed subjects.

For each target row, STATE-1 captures:

- the natural carrier entering L24 attention;
- the donor carrier at the same declared site;
- all eight natural pre-`W_O` head values; and
- all eight donor pre-`W_O` head values.

The already sealed GW-V2 capture contains the needed target-natural tensors
for all 666 rows; applying the inherited matched-control row map supplies the
donor tensors. STATE-1 must verify those artifact hashes and repeat the local
composition/parity controls under its own runner before using them. The model
index declares segment hashes; execution verifies those segment bytes as well
as the bound metadata and plan before any STATE-1 outcome.

Donor substitution is the primary removal operation because every replacement
is a real execution state with the same relation and prompt family. It removes
target-natural state without treating an all-zero vector as ordinary model
state. Zero-input arms are excluded because STATE-1 tests dependence on
naturally realized carrier/head context under matched substitution, not ablation
robustness. They enter neither selection nor adjudication.

The captured natural path, identity replacements, head sum through the
effective prepared `W_O`, declared post-attention norm, residual write, L24 FFN,
and terminal logits must reproduce their bound authorities within the existing
GW-HEAD-1 tolerances. Natural/no-op and exact/identity selected logits must be
bit-identical. Any parity, geometry, donor, firing-count, or non-finite failure
invalidates the run.

## Complete train surface

Let `C` select the entering carrier:

- `C=donor`: use the frozen donor carrier;
- `C=natural`: restore the target-natural carrier.

Let `S` be a subset of the seven non-H1 heads
`{H0,H2,H3,H4,H5,H6,H7}`. Heads in `S` use their target-natural values; every
other non-H1 head uses its frozen donor value. This yields the complete
`2 × 2^7 = 256` context surface.

Each context has a paired H1 contrast:

```text
baseline:  donor H1 + context(C, S)
treatment: target-natural H1 + the identical context(C, S)
```

Both arms run through the real effective `W_O`, post-attention norm, residual
write, L24 FFN, and all later layers. The causal difference therefore isolates
H1 within a fixed surrounding context. No singleton-score addition or isolated
vector projection may substitute for the composed operator path.

The runner's context ID is `128 × C + mask`, with `C=0` for donor carrier and
`C=1` for natural carrier. The seven mask bits, low to high, name
`H0,H2,H3,H4,H5,H6,H7`; a set bit restores the target-natural head. Rows retain
the frozen GW-V2 execution order. Within each row and context, the donor-H1 arm
precedes the exact target-natural-H1 arm. Train executes context IDs `0..255`;
held out executes only the sorted union of the four sentinels and the two
non-null train selections.

For terminal replay, the runner reuses the target row's natural prefix and KV
state, then installs the locally composed carrier at the L24 attention write.
The canonical continuation executes L24 FFN and all later layers. The L24
attention computation immediately before that replacement cannot affect the
current token downstream except through the replaced write; its KV entry is
not consulted by a later layer at this token. The runner checks the all-natural
composite against the full natural exit and GW-V2's bound natural readouts.
Its hashed `proximal.f32` and `terminal.f32` outputs have axes
`[stage rows, context IDs, donor/target H1, 142 candidate tokens]`; the manifest
records the exact original row indices, contexts, intervention count, parity,
and executable hash. A stage cannot overwrite a prior tensor or seal.

For raw and z-scored alignment separately, define the control-adjusted H1
effect in context `X=(C,S)` as:

```text
D_X = E(target H1 | X) - E(donor H1 | X)
```

and normalize it by the exact all-natural context:

```text
R_X = D_X / D_all-natural
```

A non-positive all-natural proximal denominator refuses train selection. On a
held-out split, retention is refused if the denominator is non-positive at the
point estimate, any bootstrap draw, or its 2.5th-percentile lower bound.
The same contrast is measured proximally and at terminal logits.
Because the same pre-intervention readout occurs in both H1 arms, its terms
cancel in `D_X`. The implementation may calculate the equivalent difference
of their post-intervention same-fact and matched-control JS scores. It must
use the inherited matched-control edges for both arms, with no rematching.

## Train-only freezing

Train evaluates all 256 contexts. One minimum context is frozen independently
for each carrier branch. A context is train-eligible only if its raw and
z-scored proximal retention estimates are both at least `0.80`, both causal
differences are positive, and both terminal effects are positive.

Within each carrier branch, the deterministic tie-break is:

1. fewer target-natural non-H1 heads;
2. higher minimum raw/z-scored retention;
3. lower summed natural effective-`W_O` contribution norm;
4. lexicographically ascending head IDs.

If a branch has no eligible context, it freezes an explicit `none` result.
The contribution-norm tie-break sums each retained head's train-row mean
effective-`W_O` L2 norm; only target-natural, never donor, contributions enter
that tie-break.
Validation and test cannot change either branch's subset, donor map, reference
arm, threshold, metric, or tie-break. The empty subset and all-seven subset in
both carrier branches are retained as fixed sentinel arms regardless of train
selection.

## Held-out gate and verdict classes

Validation and test are adjudicated separately with 10,000 subject-clustered
bootstrap samples. Each subject carries all three relations and all three
prompt families, preserving the fixed relation balance within every draw. The
bootstrap seed is `27022033`, reset independently for validation and test.
The frozen preregistration specifies one simultaneous one-sided
studentized max-t lower-band family within each split. That family includes
every distinct frozen branch selection and all four fixed sentinel contexts,
across both raw/z-scored metrics and proximal-retention/terminal-effect claims.
Selections frozen as `none` contribute no context. Every context used to assign
a verdict must therefore enter the same family; the exact all-natural
denominator is required to have a positive point estimate and positive
pointwise lower bound on both metrics. Non-positive bootstrap denominators
invalidate retention for that split. The all-natural proximal retention is
identically one in every draw; it is reported as a deterministic identity and
excluded from studentization because its bootstrap standard error is zero.

A frozen context clears one held-out split only when:

- raw and z-scored proximal point retention are each at least `0.80`;
- their simultaneous one-sided 95% lower bounds are each at least `0.50`; and
- raw and z-scored terminal-effect simultaneous lower bounds are each greater
  than zero.

It clears held out only by clearing validation and test independently. The
adjudication then assigns the first applicable class:

| Class | Required held-out result | Supported interpretation |
|---|---|---|
| `edge_only` | donor carrier, empty non-H1 subset clears | H1 survives after removing target-natural carrier and every target-natural neighbouring head. |
| `carrier_plus_edge` | natural carrier, empty non-H1 subset clears; `edge_only` does not | Retaining the natural carrier suffices for this gate without target-natural neighbouring heads. |
| `local_head_circuit` | a frozen subset of at most three non-H1 heads clears; neither simpler class does | A small retained local L24 context suffices, qualified by its carrier branch. |
| `broad_context` | a frozen selected subset of four or more non-H1 heads clears, or only the all-natural sentinel clears | The tested simpler frozen contexts did not clear; the minimum held-out dependency size is not identified. |
| `no_stable_context` | the exact all-natural H1 contrast or mandatory controls fail | STATE-1 is execution-invalid or the inherited effect does not replicate. |

The result reports both carrier branches and every frozen sentinel even when an
earlier class applies. The class is a dependency statement. It does not show
that retained components can be constructed cheaply, cached, or executed
without the prefix that originally produced them.
Failure to clear a gate is not itself a significance test of necessity. Any
claim that a particular retained component is necessary requires a separate
paired contrast between contexts, with its own frozen uncertainty rule.

## Hard boundaries

STATE-1 refuses:

- another V, carrier, or head predictor search;
- adaptation from validation or test;
- changing the H1 treatment across contexts;
- per-context donor rewiring;
- replacing a failed donor arm with zero;
- scoring contexts independently and adding their effects;
- hiding natural scalar, vector, KV, or prefix dependencies from accounting;
- latency claims without an exclusive peer-acknowledged measurement window;
- correctness, architecture-general, or efficient-WALK claims.

GW-V2 latency bookkeeping is not a prerequisite. STATE-1 should be sealed only
after its donor rule, bootstrap family, exact artifact authorities, and
execution implementation are reviewable together.

## Outcome-free start checkpoint

On 22 September 2026, `scripts/gwstate1_preflight.py --verify-segments`
verified the inherited execution binding, 12 named predecessor authority files,
all seven GW-V2 capture tensors, all four GW-V2 replay tensors, and all five
declared model segments against their hashes. It proved that the matched-control
map is a 666-row one-to-one derangement; the SHA-256 of its canonical row-index
array is `76ba2128552420f89aba6f6d447f10175302f8af8fe57e75362f3de91b9a0b67`.
`scripts/gwstate1_analysis.py` implements the train-only tie-break and the
sentinel-inclusive held-out max-t family on abstract subject-effect arrays.
The `--gwstate1` runner implements the exact donor/target H1 arm order,
complete train context order, frozen held-out context order, and parity
controls. `scripts/gwstate1_preregister.py` constructs and validates the
machine protocol; `scripts/gwstate1_select.py` and
`scripts/gwstate1_adjudicate.py` implement the later stage seals. No STATE-1
context outcome has been produced. The runner and preregistration underwent
joint pre-outcome review and were sealed against the already-built executable
on 23 September 2026. Any subsequent change to a bound authority requires an
explicit amendment rather than a silent edit.
