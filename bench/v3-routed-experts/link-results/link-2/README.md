# LINK-2 receipts

**LINK-2 is qualified under revision 3: 2 of the 2 required qualifying runs passed**, on 2026-11-14, both on the frozen relay binary `7b1097f8…`, with the other sessions on the machine confirmed quiet beforehand and no compiler or model process running at the start or end of either run.

What this certifies is the cost the relay adds under the paired probe. It does not certify the timing of the inference endpoints, and no network sweep has been run.
The contract in force is [revision 3](../../LINK-2-revision-3.md). Runs made
under revision 2 are listed for the record and can never count.

| Receipt | Kind | Counts | Result |
|---|---|---|---|
| [`20261114-00-plumbing-diagnostic`](20261114-00-plumbing-diagnostic/calibration.json) | Revision 2 diagnostic, declared before it started, under peer load (load average about 4.4) | No | Run fails: 6 of 10 points pass |
| [`20261114-01-revision-3-validation-diagnostic`](20261114-01-revision-3-validation-diagnostic/calibration.json) | Revision 3 diagnostic, declared before it started, to validate the implementation; peers not asked to be quiet (load average about 1.8) | No | Run passes: 10 of 10 points |
| [`20261114-02-qualifying-1`](20261114-02-qualifying-1/calibration.json) | Revision 3 **qualifying** run 1; peers confirmed quiet | **Yes** | **Run passes: 10 of 10 points.** Largest error 44.3 µs (20 ms point, tolerance 1,000); null 0.0 µs; largest judged half-width 23% of its limit |
| [`20261114-03-qualifying-2`](20261114-03-qualifying-2/calibration.json) | Revision 3 **qualifying** run 2; peers confirmed quiet | **Yes** | **Run passes: 10 of 10 points.** Largest error 38.7 µs; null −0.9 µs; largest judged half-width 30% of its limit |

## Finding before qualification: the autocorrelation gate

Recorded here because it bears on whether a qualifying attempt can succeed
under revision 2 as frozen. It changes no rule.

**The diagnostic's four failures are all the autocorrelation gate.** Every
point, including the null point, was inside its accuracy tolerance with an
interval half-width between 1.6 and 13.5 µs, against tolerances of 25 to
1,231 µs. The points that failed did so because the absolute autocorrelation
of their pair differences at lag 5 or lag 10 exceeded 0.2.

Two separate things are going on.

1. **The gate fails independent data by chance.** With 100 pairs, 0.2 is two
   standard errors, so a truly independent series exceeds it at lag 5 or
   lag 10 about 7% of the time (simulated: 7.4%;
   `test_link_pair_model.py` records the rate). A run has ten points and
   qualification needs two runs. For a perfect instrument with independent
   pairs the chance that all twenty point-checks clear the gate is about 22%.
2. **The pair differences are in fact correlated, beyond five pairs.** In the
   diagnostic, lag-1 autocorrelation ranged from 0.08 to 0.76, and three
   shaped points had lag-5 values of 0.31 to 0.47. So the gate is also
   reporting something real: a block of five pairs does not span the
   correlation on this host, at least under peer load.

**What this does and does not show.** It does not show the relay is
inaccurate; the diagnostic's errors were between −2 and +84 µs. It does show
that revision 2's precision rule is unlikely to be satisfied on this host
even when accuracy is clear by a wide margin, and that part of that is a
property of the rule and part is a property of the data.

**Not done.** No rule, threshold, block length or seed was changed, and no
qualifying run was attempted. Options for a revision 3, each needing its own
approval before any qualifying run:

- lengthen the bootstrap block to span the observed correlation and gate at
  that longer lag, with a bound that controls the false-fail rate across all
  the checks in a qualification;
- keep the block, and replace the pass/fail gate with a reported diagnostic
  plus a deliberately conservative interval;
- collect more pairs per point so both the interval and the autocorrelation
  estimate are tighter.

The diagnostic was not run on a quiet machine, so the size of the
correlation on a quiet machine is not known.

## Notes on the qualifying runs

- Host load average was about 3 to 6.5 during the runs, from desktop
  applications (browser, window server). No compiler or model process was
  present. "Quiet" here means the other sessions were idle, not that the
  machine was otherwise unused.
- Lag-1 autocorrelation of the pair differences reached 0.77 and 0.71 at
  some points. Under revision 3 this is reported and not gated; the doubled
  block-bootstrap half-widths stayed at or below 30% of their limits.
- The 1 ms point read about +16 µs in both qualifying runs and in the
  validation diagnostic. It is inside the 50 µs tolerance, but it is a
  consistent offset, not noise, and its cause is not identified.
