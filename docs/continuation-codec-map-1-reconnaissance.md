# CONTINUATION-CODEC-MAP-1 — reconnaissance

**Status: reconnaissance, not a freeze.** It localises where CODEC-2's
failing tail comes from, on calibration text only. Nothing here is a
threshold or a verdict. Any claim made from it needs its own freeze,
adjudicated on a new held-out bank (bank 3). Bank 2 stays sealed.

Evidence: [continuation-codec-map-1-recon-evidence.json](represent/forecasts/continuation-codec-map-1-recon-evidence.json)
holds every stage report, both run provenances and the sha256 of every
checkpoint.

## The question

CONTINUATION-CODEC-2 closed **NOT ACCEPTABLE**: 4-bit codec/v1 stacked on the
Q4_K execution (CH) failed the incremental-tail rule, while the mean and
confident-band rules held or nearly held
([notes](represent/forecasts/continuation-codec-2-notes.json), `closure`).
This reconnaissance asks **where that tail originates**: in K or V, which
attention type, which depth, and which row age. Is the damage concentrated
enough for a mixed continuation representation, mostly 4-bit with a small
protected set?

## Method

- **Text:** bank 1 (Pride and Prejudice, CODEC-1's bank), as calibration
  only. One rung: the first 2,048 positions, with the last 1,024 scored
  teacher-forced.
- **Arms:** R (full precision, row/v1), C (Q4_K weights, exact K/V) and CH
  (codec/v1 4-bit on the Q4_K store). Every map below runs on the same Q4_K
  store as C and CH.
- **Instrument:** `MappedCodec`, a test-only simulated partial codec. It
  uses codec/v1's retention, encodes each appended K/V head once with
  codec/v1's own TurboQuant at 4 bits, and at each layer read takes each
  row's decoded or exact copy according to a map: layers (all / sliding /
  full / depth quarter / list), K and/or V, row age, and unions of these.
- **SIM-0 (the stop rule):** with every row compressed, the simulator must
  reproduce CH **bit for bit**, with the same retained range after every
  append and the same layer and range for every read (a 210,188-event
  trace hash). It held on fixtures and on gemma3-4b-it in both stores.
- **Response variable:** per position, d = KL(R‖map) − KL(R‖C), the
  damage the map adds to the quantised execution. Each map is reported
  as:
  - mean d and p99 d;
  - **the fraction of CH's tail it reproduces**, Σ d_map / Σ d_CH over
    CH's worst 5% (52 positions) and worst 1% (11 positions);
  - the share of K/V bytes it leaves exact.
- **Adaptive, capped:** each stage's maps were chosen from the previous
  stage's report. There were 10 maps after stage 0, against an agreed cap
  of 8–10 after stage 1.

## Results (gemma3-4b-it, bank 1, 1,024 positions)

Reference points: R↔C top-1 0.816, mean KL 0.299. CH adds mean d +0.071
and p99 d +2.05, with top-1 0.778.

### Localisation: what a map *reproduces* when only it is compressed

| map | compressed | mean d | p99 d | CH worst 5% | CH worst 1% |
|---|---|---|---|---|---|
| CH (all K+V) | 100% of K/V | +0.071 | +2.05 | 100% | 100% |
| **K only** | 50.0% | **+0.060** | +1.66 | **59%** | **53%** |
| **V only** | 50.0% | **−0.008** | +0.40 | **−1%** | **−1%** |
| K, sliding layers | 37.2% | +0.038 | +1.22 | 35% | 48% |
| K, full-attention layers | 12.8% | +0.024 | +0.76 | 16% | 15% |
| K, depth Q1 | 12.8% | +0.016 | +0.60 | 12% | 28% |
| K, depth Q2 | 11.5% | +0.026 | +0.88 | 23% | 23% |
| K, depth Q3 | 14.1% | +0.012 | +0.67 | 18% | 12% |
| K, depth Q4 | 11.5% | +0.001 | +0.30 | 2% | 3% |
| K, newest 64 rows exact | 47.3% | +0.030 | +0.95 | 27% | 15% |
| K, newest 256 rows exact | 39.1% | +0.016 | +0.64 | 13% | −2% |
| **K, only newest 256 compressed** (inverse) | **10.9%** | **+0.043** | +1.33 | **37%** | **62%** |

### Intervention: what *protecting* a set removes from full CH

The candidate came from stage 4: keep the **newest 256 K rows exact in
every layer**, and compress all V and all older K
(`all:v:all|all:k:older256`).

| | exact bytes | mean d | p99 d | top-1 | CH worst 5% removed | CH worst 1% removed |
|---|---|---|---|---|---|---|
| CH | 0% | +0.071 | +2.05 | 0.778 | — | — |
| **protect newest 256 K** | **10.9%** | **+0.016** | **+0.66** | **0.803** | **94%** | **≈100%** |
| additive prediction (from the inverse map) | 10.9% | | | | 37% | 62% |

Mean damage removed: 77% measured, against 60% predicted.

## Reading

1. **K, not V.** V-only compression added no measurable damage on this
   run: mean d −0.008, and top-1 identical to C's, 0.8164. K-only reproduced
   about 85% of CH's mean damage and about 55–60% of its tail. This is
   supported only for this model, text and length. "V may tolerate
   aggressive compression" is a hypothesis to freeze, not a finding.
2. **Within K, recency dominates; depth and attention type matter less.**
   - Compressing only the newest 256 K rows (10.9% of bytes) reproduces 71%
     of K's mean damage and 62% of CH's worst-1% tail. Old K costs about a
     tenth as much per byte. The inverse control rules out "fewer
     compressed rows, less damage".
   - Depth Q4 is nearly free. Q2 is the densest depth region, at about 2×
     its byte share.
   - Attention type splits roughly in proportion to bytes; full-attention K
     is about 1.8× denser per byte on the mean.
3. **The damage is not additive in the tail; it interacts.**
   - Mean damage adds almost exactly across complementary maps: sliding +
     full gives 0.062 against K's 0.060, Q1–Q4 sum to 0.055, and old + new
     to 0.058.
   - The tail does not. Complementary parts always reproduce *less* tail
     than the whole: 50% against 59% for both attention type and age.
   - Decisively, **protecting the newest 256 K rows removed 94% of CH's
     worst-5% tail where additivity predicted 37%.** Recent-K error is the
     hinge of an interaction: compressing V and old K does little damage
     *unless* recent K is also compressed.
4. **The tail has a stable core.**
   - Of CH's 52 worst-5% positions, 17 are in the worst 5% of at least 6
     of the 11 maps. These are the most damaged positions (mean d_CH 1.65)
     with ordinary confidence (R margin 0.53).
   - The 7 positions specific to at most one map have lower margins (0.35),
     higher entropy and smaller damage (0.70).
   - The fragile core is the large-damage tail that recent-K protection
     removes.

**Against the target set before the run** ("70–90% of the tail from under
20% of the bytes"): protecting the newest 256 K rows reached **94% of the
worst-5% tail and essentially all of the worst-1% at 10.9% of K/V bytes
exact.** A mixed representation, mostly 4-bit with recent K exact, is
strongly motivated on this evidence. It is not established.

## Limits

- **One model, one text, one length.** gemma3-4b-it, bank 1, a single
  2,048 rung; 1,024 positions, so the worst-1% set is 11 positions.
  CODEC-2's 4,096 and 8,192 behaviour, and rungs past the 1,024-position
  sliding window's interaction with age, are not covered.
- **The window size is a pre-chosen guess.** 256 was one of two
  pre-chosen ages (64 and 256), not optimised; 256 rows is a quarter of
  the sliding window here.
- **Not a production path.** The simulator holds f32 copies by design.
  It says nothing about memory or decode cost of a real mixed provider.
- **Two instrument commits.** The localisation maps ran from instrument
  `a4807465` (binary `f31deb06…`). The protection map ran later from
  `958d69c6` (union maps; binary `b614be95…`) in a fresh store, with stage
  0 and SIM-0 rerun. The two stores' CH scores are identical per position,
  with the same trace hash.
- **Contended wall times.** A peer session's CPU job overlapped stages
  0–2. Wall times are recorded, not interpreted.

## What a freeze could ask (options, not choices)

- **The question:** does a mixed continuation representation (4-bit
  codec/v1 except the newest w K rows per layer, kept exact) pass
  CODEC-2's frozen rules on a NEW held-out bank? Or: what is the smallest
  w that does?
- **Arms:** C, CH, CH-protected(w) and full-precision-R, as in CODEC-2,
  with the same B1–B4 machinery.
- **Instrument:** a real provider (exact recent window + encoded older
  rows), not the simulator. Its residency must be proven as CODEC-1's
  was: its resident bytes have to be accounted for, not assumed.
- **Window size:** w chosen before the bank-3 run, from a calibration
  sweep on bank 1 or another calibration text, never bank 2 or bank 3.
- **Decode cost:** stays out until quality holds.
