# GW-CAR-1 — carrier compactness and construction

**Status:** DESIGN DRAFT — no CAR-1 candidate replay or adjudication.

**Predecessor:** [GW-STATE-1](gw-state-1-results.md), closed as `carrier_plus_edge` with [Erratum 1](gw-state-1-erratum-1.md).
**Claim boundary:** CAR-1A measures captured-carrier compactness. CAR-1B will test construction from a separately declared source and requires its own prospective protocol.

GW-STATE-1 context `128` provides the fixed causal interface. Its target-natural
entering L24 carrier, target-natural H1 treatment, and seven matched-donor
non-H1 heads cleared the registered held-out gate. CAR-1A varies only that
entering carrier. H1, donor identity, the seven other heads, target prompt,
natural prefix/KV state, continuation, candidate-token vocabulary, readouts,
and matched controls remain exactly as in STATE-1.

## CAR-1A: captured-carrier compression

The source of every candidate is the already captured 2,560-dimensional
target-natural carrier before L24 attention. The candidate is decoded to f32
and substituted at that local composition input. Both donor-H1 and exact
target-H1 arms receive the same candidate carrier. The head mixture runs
through the effective prepared `W_O`, declared post-attention norm and
residual write; the target's natural prefix/KV and L24 FFN onward provide the
terminal continuation. The candidate cannot change any head, KV entry, or
later state independently. All candidates therefore remain natural-prefix
dependent; none can establish cheaper execution by itself.

Candidate IDs and encodings are fixed before CAR-1 outcomes:

| IDs | Family | Levels | Per-row payload |
|---|---|---|---|
| 0 | exact captured carrier | f32 | 10,240 bytes |
| 1 | matched-donor carrier control | f32 | 10,240 bytes |
| 2 | IEEE f16 round trip | 16 bits/coordinate | 5,120 bytes |
| 3–5 | signed symmetric per-row uniform quantization | 8, 4, 2 bits/coordinate | packed codes plus one f32 scale |
| 6–11 | per-row top-absolute-coordinate sparsity | 32, 64, 128, 256, 512, 1,024 coordinates | each retained coordinate is a u16 index plus f32 value |
| 12–18 | train-only centered PCA basis | 8, 16, 32, 64, 128, 256, 384 coefficients | f32 coefficients plus shared f32 mean and basis |

Uniform quantization uses `scale = max(abs(x)) / (2^(b-1)-1)`, zero scale
only for an all-zero row, NumPy `rint` ties-to-even, and signed codes clamped
to the representable symmetric range. The scale is stored as f32; division
and rounding use f64 intermediates. The code pack order is increasing
coordinate index, least-significant code first within each byte. Sparse
coordinates are ranked by descending absolute f32 value with ascending
coordinate index breaking ties, then serialized in ascending index order.
Unretained coordinates decode to zero. PCA fits only the 396 frozen train
rows, after a train mean is subtracted. It uses the leading right singular
vectors from NumPy `linalg.svd(full_matrices=False)` of the centered matrix;
each vector's largest-absolute coordinate is
made positive to fix its sign. A held-out row is projected with the frozen
mean and basis. The mean is accumulated in f64 then stored as f32; SVD uses
f64 centered values and its leading vectors are stored as f32. Projection
and reconstruction use f64 intermediates, with f32 coefficients and decoded
carrier. The fit artifact fixes the exact basis bytes. No validation or test
carrier influences the fit, grid, or decoder.

Every candidate reports actual encoded bytes per row and shared model bytes,
the latter amortized separately over 666 rows. Exact/donor controls are not
eligible compactness results. The ledger also records decode arithmetic and
that obtaining the source still requires the natural prefix. Wall time is
outside this causal experiment unless a separate exclusive measurement window
and a valid baseline/candidate/baseline bracket are established.
Decode arithmetic is a formula-derived count of scalar operations, not a
measured FLOP or latency claim.

The runner checks candidate 0 against the already sealed STATE-1 context-128
proximal and terminal tensors, both H1 arms, bit for bit on all rows before
using candidate results. Candidate 1 must likewise reproduce context 0. Any
failure of provenance, geometry, finite values, firing count, or those
identities invalidates the run. Train and held-out replay outputs have axes
`[stage rows in frozen order, candidate IDs 0..18, donor/target H1, 142
candidate tokens]` on both proximal and terminal surfaces.

## CAR-1A analysis and gate

For each candidate, the causal contrast is the same subject-clustered,
control-adjusted target-H1 minus donor-H1 effect used in STATE-1. The
denominator is the exact all-natural context-255 effect from the sealed
STATE-1 replay on the same split and metric. The exact denominator must be
positive at point estimate, in every bootstrap draw, and at its pointwise
2.5th-percentile lower bound. Candidate 0 is an identity reference, but its
retention is not identically one because the denominator is context 255.

Validation and test are adjudicated independently with 10,000
subject-clustered draws, sorted subject IDs, NumPy PCG64 seeded at `27022033`
and reset per split, `ddof=1`, and NumPy's linear quantile convention. One
studentized one-sided max-t family within each split contains every candidate
ID 0..18 across raw/z-scored proximal retention and raw/z-scored terminal
effect. A candidate clears one split only when both proximal point retentions
are at least 0.80, both simultaneous lower bounds are at least 0.50, and both
terminal-effect simultaneous lower bounds are greater than zero. It clears
held out only by clearing both splits. Exact candidate 0 must clear both
splits or CAR-1A is invalid, regardless of other candidates.

The result is a full dimension/bit/byte versus effect curve, with actual
per-row and shared storage reported separately. The smallest clearing
payload can be described using total bytes at the declared 666-row
amortization, with candidate ID breaking exact ties. This ranking is a
post-adjudication summary over a simultaneous family, not permission to tune
the grid on held-out rows. Non-clearing candidates are reported. Compression
does not imply a cheap carrier producer or a necessary-component claim.

The staged execution order is: review the human spec, machine protocol,
codec, runner and analysis together; seal the protocol to the already-built
`gwcar1a_record` executable; fit PCA on train carriers; generate all train
candidate payloads and replay their paired H1 arms; generate held-out payloads
using that same fit; replay validation and test; then adjudicate once. The
draft protocol, candidate fitter, and runner must refuse fit or model replay
before the executable seal. No CAR-1A model outcome is authorized by a design
draft.

## CAR-1B: separately prospective source construction

CAR-1B will freeze a new source inventory, construction algorithm, cost ledger,
selection rule, and held-out population before producing its own outcomes.
Candidate sources may include earlier state, lookup/index entries, cached
state, or learned transforms. Every source must identify whether it requires
the original prefix, what it reads or computes, and how H1 and continuation
are supplied. CAR-1A may nominate a compact target representation; it does
not preregister CAR-1B or license reuse of CAR-1A held-out rows as fresh
confirmation.
