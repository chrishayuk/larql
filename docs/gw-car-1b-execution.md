# GW-CAR-1B — capital-cohort executable protocol

This document completes the [source-design draft](gw-car-1b.md). The user
accepted fresh-subject capital confirmation on 28 September 2026. It narrows
the scientific claim to this relation; it does not extend the result to the
old three-relation population. No candidate outcomes were available when this
executable protocol was prepared.

## Population and information access

Use all 67 subjects and all 201 prompts in the hash-bound
[capital proposal](../bench/gw-car-1/gemma3-4b-it-phase1/gwcar1b-capital-population.proposal.json):
34 subjects / 102 prompts for validation and 33 / 99 for test. Each subject
has the same three capital templates used by the train bank. Exclude all
127 country identities studied in factual GW-0 or any GW-V2 split. The entire
44-subject / 396-row GW-V2 training bank is retained for source fitting,
including all three old relations. No previous held-out carriers enter fitting.

The pinned source's annotations are the target contract, including any
historical ambiguity; no manual country exclusions or outcome-driven cleanup
are allowed. This is a complete remaining source-defined population, not a
power-selected sample. Failure to establish an effect remains a possible
outcome. No post-hoc power or universal-semantic claim is authorized.

The readout is the sorted union of the proposal's 57 first-answer token IDs.
It is frozen before capture. Standalone subject-name tokens without special
tokens supply the subject embedding feature; each effective embedding is
scaled by the model embedding operation, accumulated in f64, averaged in f64,
then stored as f32. This extraction runs no attention/FFN block.

Fresh row IDs are the existing canonical `short_id("car1b", [subject_id,
prompt_family])`. In each split × template cell, sort row IDs and assign the
cyclic next row as matched donor. This is a bijective, split-isolated
derangement. The same three prompts for one subject form the same-fact group;
their three matched donors form the different-fact group. The donor triple
need not itself represent one subject. No target labels select a neighbor.

## Capture, controls and continuation

The separate `gwcar1b_record` binary executes canonical prepared-production
CPU operands under the historically bound Gemma 3 4B plan and model segment
hashes. The protocol seals the actual binary before capture. Adding this
example to Cargo does not change historical protocol, replay, source, or binary
bytes; historical manifests remain records of their earlier Cargo revision.

Capture 396 train rows followed by 201 fresh rows. Record the entering L24
carrier, its natural attention-write output, eight L24 head outputs, entering
L12/L16 carriers, subject embedding feature, and terminal readout. L12/L16
features come from the previous FFN's residual write with declared layer scale,
at the final prompt position. These are not V projections. Capturing in a full
run is observation, not a measurement of an early-exit implementation.

Every natural L24 head mixture must reconstruct the observed natural attention
write bit for bit. All 396 new training carriers must reproduce the immutable
historical carriers bit for bit; any mismatch invalidates capture. Capture
contains model outputs but no candidate results; it still occurs only after seal.

Independently replay fresh STATE-1-style contexts 0, 128 and 255, each with donor
and target-exact H1. Context 0 uses donor carrier/other heads; context 128 uses
natural carrier/donor other heads; context 255 uses natural carrier/natural
other heads. All use target-natural KV and ordinary L24 FFN onward. The
context-255 target-H1 arm must reproduce the captured natural attention-write
vector and terminal readout bit for bit on every fresh row.

Replay all 20 candidates from the source design, both H1 arms, on all 201 fresh
rows. Candidate 0 must reproduce independent context 128; candidate 1 must
reproduce independent context 0, proximal and terminal, bit for bit. Exactly
8,040 candidate firings and 1,206 reference firings are required. The 396 train
rows are used for fitting and historical identity, not for outcome-driven
selection. No CAR-1B training intervention sweep is needed or performed.

These experiments still consume exact target H1, donor other heads, and
natural target KV. Even a passing embedding-only carrier source does not prove
end-to-end avoided prefix work. Timings are operational only.

## Fitting and artifacts

All constructors, supports, ridge penalties, means, shuffles, and retrieval
rules remain those of the source-design draft. Only train carrier/features
are passed to `fit`; query constructors receive metadata and their one declared
feature vector. Oracle candidate controls have their own API. Same-subject bank
retrieval refuses. This first wave tests fresh subject composition/retrieval,
not cache hits on repeated prompts.

Each candidate stores its actual numeric model in an uncompressed NPZ plus a
JSON metadata file containing the group/row lookup it needs. Report numeric
array bytes, serialized model bytes and metadata bytes separately. The
201 × 20 × 2,560 f32 candidate replay tensor is a materialization artifact,
not a claim that every source must store that tensor. Per-row receipts identify
feature source, retained prefix depth, retrieved IDs, output/feature bytes,
ridge multiply-add pairs or distance-coordinate counts, and unmeasured work.
Input extraction, offline training, and process memory remain separate costs.

Store norms, top-32 support overlap counts, and MSE as descriptive diagnostics. Their computation cannot
select a source, support, regularizer, donor, or gate. Model and artifact hashes
bind every subsequent stage; existing files/directories cannot be overwritten.

## Frozen statistic and decision

For each raw/z readout, convert candidate logits to softmax probabilities;
z-score each row over its 57 logits with population standard deviation. A zero
z-score standard deviation refuses. For a subject's three templates compute
the mean of their three pairwise Jensen-Shannon divergences. Compute the same
quantity for its three matched-donor rows. The effect is

```
(same_fact_JS_donor_H1 - same_fact_JS_target_H1)
 - (different_fact_JS_donor_H1 - different_fact_JS_target_H1).
```

This is the prior control-adjusted relational contrast, now with exactly one
capital edge per subject, not three averaged relations. Analyze validation and
test independently with sorted subjects, 10,000 clustered draws, NumPy PCG64
seed 27022034 reset per split, ddof=1, and linear quantiles.

The proximal denominator is fresh context 255's effect. Both raw/z denominator
points, every bootstrap denominator, and the pointwise 2.5% lower quantile must
be positive; otherwise report `invalid_exact_denominator` without candidate
claims. Use one studentized one-sided 95% max-t family with 104 coordinates:

- 20 candidates × raw/z proximal retention (40);
- 20 candidates × raw/z terminal effect (40);
- 11−8, 12−8, 13−8, 11−17, 12−18 and 13−19, each as raw/z proximal/terminal
  effect differences (24).

Floor the common critical value at zero. For a coordinate whose bootstrap SE
is at most 1e-15, use standardized statistic zero and lower bound
`min(point, 0)`: it cannot support a positive claim. This handles exact
template/shuffle coincidences conservatively without deleting candidates or
changing the declared family after seeing results.

Each candidate clears a split iff both proximal point retentions are ≥0.80,
both simultaneous proximal lower bounds are ≥0.50, and both simultaneous
terminal-effect lower bounds are >0. Candidate 0 must clear both splits;
otherwise report `invalid_exact_context128_control`. All valid source claims
require passing both splits. IDs 6–15 alone are eligible sources. IDs 11–13
earn an additional informative-delta label only when both their template and
their corresponding shuffle comparisons have positive simultaneous lower
bounds on all four readouts and both splits. Absolute adequacy alone does not
establish entity information or natural-state fidelity.

Report the entire surface. No best-cell selection, grid extension, threshold
change, or rerun after a scientific failure is authorized by this protocol.
Integrity failures preserve their files and require a documented correction;
they are never silently overwritten. A negative result covers only the ten
registered constructors at this oracle-H1/natural-KV interface.

## Execution order

1. Run synthetic source, population, artifact and analysis tests; compile/test
   the isolated Rust example and inspect the prose/executable mapping.
2. Generate the executable draft; verify all historical model segment hashes;
   seal to the release binary before any CAR-1B capture/fitting/replay.
3. Capture, verify train identity and feature provenance; fit/train-only source
   models and materialize all fresh candidates without reading causal results.
4. Run independent reference contexts, then the complete candidate replay.
5. Verify identities, artifact hashes and full coverage; adjudicate both splits
   once and write a separate results document. Historical and sealed files keep
   their original bytes.

The checked protocol exposes exact input rows, donor indices, readout IDs,
model/code hashes, and executable identity. The code paths are
[`gwcar1b_runtime.py`](../scripts/gwcar1b_runtime.py),
[`gwcar1b_analysis.py`](../scripts/gwcar1b_analysis.py), and
[`gwcar1b_record.rs`](../crates/larql-demos/examples/inference/gwcar1b_record.rs).
