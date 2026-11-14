# LINK-2 revision 3 (frozen)

Status: **approved and frozen on 2026-11-14. This is the contract in force
for LINK-2 qualification. No qualifying run has been made.** Any change to a
parameter or rule here is a new revision, not an edit. Diagnostic runs are
declared before they start and never count. Qualification additionally
requires that the implementation conforms to this document and that the
machine is explicitly quiet.

The approval notes that the simulated coverage below is evidence for a
defensible pre-registered procedure, not a guarantee about real host jitter.

[Revision 2](LINK-2.md) stays as it was frozen, and stays failed as a
qualification design: its precision rule cannot realistically be met (see
[the finding](link-results/link-2/README.md)). Its one diagnostic run was
declared a diagnostic before it started. It did not count under revision 2
and cannot count under revision 3 or any later revision. The same holds for
every LINK-1 run.

Revision 3 changes only how uncertainty is judged. Everything else is
revision 2, unchanged.

## What changes

| | Revision 2 | Revision 3 |
|---|---|---|
| Autocorrelation at lag 5 and 10 | Pass/fail gate at 0.2 | **Removed as a gate.** Reported at lags 1 to 30 |
| Pairs per point | 100 | **200** |
| Bootstrap block | 5 pairs | **20 pairs**, circular |
| Precision rule | half-width ≤ half the tolerance, and the gate | **twice the half-width ≤ half the tolerance** (half-width ≤ a quarter of the tolerance) |
| Null point precision | half-width ≤ 12.5 µs | **half-width ≤ 6.25 µs** (same doubling) |

## What does not change

- The instrument, the frozen relay binary, the frame sizes, 10-exchange
  blocks and the baseline, shaped, shaped, baseline order.
- Accuracy: `|estimate − expected| ≤ max(5% of expected, 50 µs)`.
- The null point's accuracy bound of 25 µs.
- The estimate (median pair difference), 2,000 resamples and a fixed,
  recorded seed.
- Two complete passing runs on a peer-quiet machine; no exclusions; declared
  diagnostics never count; one failed qualifying run stops the work.
- The scope of a pass: the relay's added cost under this probe, not the
  timing of the inference endpoints. Every gate of the inference sweep
  stays in force.

## Why these values

Evidence is in
[`link-results/link-2/revision-3-groundwork`](link-results/link-2/revision-3-groundwork/),
with the scripts that produced it.

**1. On the existing diagnostic, the interval hardly depends on the block
length.** Recomputed for blocks of 1, 5, 10, 20, 25 and 50 pairs, every
point's half-width stayed between 0.7 and 20.1 µs. The tightest margin is the
null point: at most 2.1 µs against a limit of 12.5 µs. This is a statement
about that one diagnostic under peer load. It is used here only to choose a
procedure and does not qualify anything.

**2. No block length makes the nominal 95% interval honest at 100 pairs.**
Simulated series with known correlation (first-order autoregressive, `phi`
the lag-1 correlation), interval for the median, how often it contains the
truth:

| `phi` | Block 1 | Block 5 | Block 10 | Block 20 | Block 25 |
|---|---|---|---|---|---|
| 0 (independent) | 95% | 93% | 94% | 85% | 83% |
| 0.40 | 85% | 93% | 92% | 86% | 83% |
| 0.76 (largest lag-1 seen) | 59% | 85% | 83% | 85% | 84% |
| 0.90 | 42% | 61% | 70% | 75% | 75% |

A longer block helps correlated data and hurts independent data, because
100 pairs hold only five blocks of 20. A block of 20 at 100 pairs is
therefore not justified by this evidence, although it was the candidate
suggested for evaluation.

**3. Widening the interval is what restores coverage, and more pairs let a
long block work.** Coverage when the half-width is doubled:

| `phi` | 100 pairs, block 10 | 100 pairs, block 20 | 200 pairs, block 10 | 200 pairs, block 20 |
|---|---|---|---|---|
| 0 | 100% | 98% | 100% | 100% |
| 0.40 | 99% | 98% | 100% | 99% |
| 0.76 | 98% | 98% | 100% | 100% |
| 0.90 | 91% | 92% | 96% | 98% |

200 pairs with a block of 20 and a doubled half-width holds at least the
nominal 95% across the whole range simulated, including correlation stronger
than anything observed. It costs about twice the measuring time per run,
roughly five minutes.

## Limits of this justification

- The simulation is first-order autoregressive with Gaussian noise. Real
  pair differences need not be either. The diagnostic's own autocorrelation
  did not decay like a first-order process at every point.
- 400 simulated trials per cell: each percentage is uncertain by about two
  to three points.
- The largest correlation was observed under peer load. Its size on a quiet
  machine is not known.
- Doubling the half-width is a conservative convention chosen from this
  simulation, not a derived correction. The interval remains a pragmatic
  bound, and autocorrelation is reported so a reader can see how far a
  point's data is from the simulated range.

## An alternative that keeps 100 pairs

If 100 pairs must stay: block 10, half-width doubled. It holds 98% or better
up to the largest correlation observed and falls to 91% at `phi` = 0.9. It
is weaker than the proposal above and is listed so the choice is explicit.

## Implementation

`link_pair_model.py` carries these parameters and rules;
`test_link_pair_model.py` pins them. The probe and the relay are untouched.
Every receipt records the protocol revision, and a run measured with another
revision's pair count is refused, not judged.
