# LINK-2: paired calibration of the link relay (pre-registration, revision 2, frozen)

Status: **revision 2, approved and frozen on 2026-11-14, then superseded the
same day by [revision 3](LINK-2-revision-3.md) because its precision rule
could not realistically be met. Kept unchanged as the record of revision 2.
No qualifying run was made under it.** Any change to a parameter or rule below is a new
revision, not an edit. Development runs are declared before they start and
never count.

One reading note was attached to the approval and changes no rule: the
autocorrelation gate is a pragmatic diagnostic of whether correlation
outlasts the bootstrap block. Passing it does not establish that the pairs
are independent or that the interval is valid.

A finding made while validating the instrument, before any qualifying run,
is recorded in [`link-results/link-2/README.md`](link-results/link-2/README.md).
It concerns the autocorrelation gate and needs a decision before
qualification is attempted.

LINK-2 is a new calibration protocol for the frozen relay of
[LINK-1](LINK-1.md). It does not amend LINK-1. LINK-1's seven failed
calibrations and its failed qualifying attempt stay recorded as failures
([index](link-results/calibration/index.json)); nothing measured under LINK-1
counts toward LINK-2, and no LINK-1 run is re-scored.

## Why a new protocol

LINK-1 compared a shaped measurement with a baseline taken by a different
probe process over a different connection. The
[jitter diagnostic](link-results/calibration/20261114-11-jitter-trace/summary.json)
found a per-batch slowdown of about 1.4 to 1.8 times that:

- is present with no relay in the path,
- is the same for a new relay process and a long-lived one,
- scales all three legs of an exchange together, including the leg that never
  touches the relay,
- is not explained by core class (cycles per CPU second within 8%),
- and is smaller, but not absent, when one connection stays open.

Its cause inside the host is not identified. On a baseline near 85 µs that
factor is about 40 µs, the size of LINK-1's 50 µs tolerance floor.

## Question

Does measuring the shaped path and the baseline path as interleaved pairs,
from one long-lived process over two long-lived connections, determine the
relay's added cost per exchange within the unchanged LINK-1 tolerance at
every sweep point?

This is not assumed. One process can still be descheduled, and two
connections through two relay processes can differ from each other for
reasons unrelated to the configured link. The protocol has a control for
each.

## Instrument

- **One measuring process**, running for the whole run, holding the client
  and the echo peer and polling its sockets (as `v3_link_trace` does).
- **Two relay processes** of the frozen `v3_link_shaper` build (binary
  `7b1097f8…`, source as committed in `6347291d`), started once per point:
  a baseline relay at zero delay and unlimited rate, and a shaped relay at the
  point's delay and rate. The relay is not modified.
- **One connection through each relay**, opened once per point and reused.
- **Frames** are LINK-1's: 11,576 request bytes plus an 8-byte probe header,
  and 46,136 response bytes (the one-worker GPT-OSS exchange).
- Inference workers, wire frames and the numerical path are not involved.

## Measurement

Fixed before any qualifying run:

| Parameter | Value |
|---|---|
| Warm-up | 20 exchanges on each connection, discarded |
| Block | 10 exchanges on one connection |
| Pair | four blocks in the order baseline, shaped, shaped, baseline |
| Pair difference | median of the 20 shaped exchanges minus median of the 20 baseline exchanges |
| Pairs per point | 100 |
| Point estimate | median of the 100 pair differences |
| Uncertainty | 95% percentile interval of that median over 2,000 circular block-bootstrap resamples of the pair sequence, block length 5 pairs, fixed seed recorded in the receipt |

The baseline-shaped-shaped-baseline order cancels drift that is linear over a
pair. Every exchange is kept in the receipt, not only the summaries.

### Serial correlation

The 100 pair differences of a point come from one process, two sockets and
one host over a few seconds, so they may be correlated in time. Resampling
them as if independent would make the interval too narrow. Three rules, fixed
here:

1. **The interval always uses a block bootstrap.** Pairs are resampled in
   their collection order as circular blocks of 5 consecutive pairs
   (the cube root of 100, rounded up). There is no data-dependent choice
   between an independent and a block procedure: the block procedure is used
   for every point, and it remains valid when the pairs happen to be
   independent.
2. **Correlation is measured and reported.** For each point the receipt
   records the autocorrelation of the pair differences at lags 1 to 10, and
   the interval an independent-pair bootstrap would have given, beside the
   block interval that is actually used.
3. **A block length that is too short fails the point.** If the absolute
   autocorrelation at lag 5 or at lag 10 exceeds 0.2 (two standard errors for
   100 independent values), correlation reaches beyond the block and the
   interval is not trusted. The point then fails the precision rule, whatever
   its interval width. The block length is not adjusted in response.

Points: LINK-1's nine (RTT 0.2, 1, 5, 20 ms at unlimited rate; 1 Gbit/s and
100 Mbit/s at zero delay; (1 ms, 100 Mbit/s), (5 ms, 1 Gbit/s),
(20 ms, 100 Mbit/s)), plus the **null point** below. Expected cost per
exchange is LINK-1's: the configured round trip plus frame bits divided by the
rate.

## Controls

1. **Null point.** The "shaped" relay is configured exactly as the baseline
   relay. The expected difference is zero. This measures whatever two relay
   processes and two connections differ by on their own; any such asymmetry
   adds directly to every other point.
2. **Pairing check.** For each point the receipt reports the spread of the
   100 pair differences beside the spread obtained when the same shaped and
   baseline blocks are re-paired at random (fixed seed). This shows whether
   pairing removed common-mode variation or merely averaged it.
3. **Relay lateness.** Both relays' self-reported lateness is recorded per
   point, as in LINK-1.
4. **Host state.** Load average and the absence of compiler and model
   processes are recorded at the start and end of each run, and the other
   sessions on the machine must have confirmed quiet beforehand.

## Decision rules

A **point passes** only if both hold:

- **Accuracy:** `|estimate − expected| ≤ max(5% of expected, 50 µs)`.
  This is LINK-1's tolerance, unchanged.
- **Precision:** the half-width of the 95% block-bootstrap interval is at
  most half that tolerance, **and** the lag-5 and lag-10 autocorrelations are
  within the bound of the serial-correlation rule. A point that lands inside
  the tolerance with a wide or untrusted interval does not pass.

The **null point passes** only if `|estimate| ≤ 25 µs` with the interval
half-width at most 12.5 µs and the same autocorrelation bound. It is held to half the tolerance floor because
its error is shared by every other point.

A **run passes** only if all nine points and the null point pass.

**Qualification** requires two complete passing runs, both on the frozen
relay binary, both under control 4.

Fixed in advance:

- No exchange, pair or point is excluded after the fact. A run is void only
  if the instrument itself fails to complete (a crash or a closed
  connection), and a void run is recorded as void with its cause.
- Development and diagnostic runs made while building the instrument are
  labelled as such before they start and never count, whatever they show.
- If either qualifying run fails, work stops and the failure is reported with
  its receipts. There is no third attempt, no relay change and no threshold
  change without a new decision.
- The pairing check is reported, not gated. The statement "pairing removed
  common-mode host variation" may be made for a point only if its paired
  spread is at most half its re-paired spread.

## Scope of a pass: the relay, not the runtime

**LINK-2 qualifies one thing: the cost the frozen relay adds to an exchange,
as seen by this paired polling probe.** It is a statement about the relay's
configured delay and serialization rate. It is not a statement about the
timing of the real inference endpoints.

The coordinator and the expert workers are different programs from the probe.
They use blocking I/O, run in separate processes, wake from sleep on every
exchange, and do model work between exchanges. Nothing in LINK-2 measures how
those endpoints behave behind the relay, and a pass must not be described as
calibrating or certifying the execution environment.

For that reason a LINK-2 pass replaces only LINK-1's calibration receipts.
Every gate of the inference sweep stays in force, unchanged and independent:

- token parity of every trial against the saved control,
- unshaped relay controls before and after each point, agreeing within 1%,
- relay lateness per trial within the timing gate,
- void points kept as diagnostics and never fitted or pooled,
- the latency model fitted on single-axis points and judged on held-out
  joint points,
- one topology first, reported for validity before any second topology.

The sweep's own controls are what establish how the real runtime responds to
the link. If the fitted coefficients differ from 1, that is a finding about
the runtime behind a correctly configured relay, not a reason to revisit
LINK-2.

A pass also does **not** show:

- that LINK-1's calibration would have passed,
- what a physical network of the same RTT would do.

A fail would mean this host cannot certify the relay to 50 µs per exchange by
this design either. The remaining options would then be a different host or a
sweep contract that reports its smallest points as below resolution, each a
separate decision.

## To be built if approved

One example binary (working name `v3_link_pair`) and a harness mode that
drives it and writes receipts. No change to the relay, to LINK-1's harness
contract, or to any runtime crate.
