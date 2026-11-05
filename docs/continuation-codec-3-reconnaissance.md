# CONTINUATION-CODEC-3 — reconnaissance

**Status: reconnaissance, not a freeze.** It builds the real provider that
[CODEC-MAP-1](continuation-codec-map-1-reconnaissance.md) simulated, proves
it is that simulation, measures what it actually holds, and picks a window
on calibration text only. Any quality claim needs its own freeze,
adjudicated on a new held-out bank (bank 3). Bank 2 stays sealed. Decode
cost is out of scope until quality holds.

Evidence: [continuation-codec-3-recon-evidence.json](represent/forecasts/continuation-codec-3-recon-evidence.json)
holds the report, stage 0, every window's residency and trace, both run
provenances, and the sha256 of every checkpoint.

## The provider: `codec-recent/v1`

[`crates/larql-kv/src/vindex3/codec_recent.rs`](../crates/larql-kv/src/vindex3/codec_recent.rs),
shipped in the continuation registry. KV-only.

- **Retention** is codec/v1's (and so window/v1's): each layer holds the
  plan-required range and frees the rest.
- **V:** every row is encoded once, at append, with codec/v1's TurboQuant.
- **K:** the newest `w` rows of each layer are held as exact f32 copies, one
  allocation each. A K row is encoded once, when an append pushes it out of
  the window.
- **Configuration:** `bits` (3 or 4) and `exact_recent_k` (w ≥ 1), both
  required, no defaults. A window of 0 is codec/v1 and is refused.

The codec is deterministic, so a K row encoded when it ages out holds the
bytes codec/v1 would have written at append. Age is relative to the
layer's newest held row at the read, as in the simulator. The exact row is
**copied, not adopted**: the backend's row is freed inside `append`, so
everything the provider holds was born in its own appends, where the
residency tally sees it.

## The three things the provider had to establish

| | fixtures | gemma3-4b-it, bank 1 |
|---|---|---|
| **Is the simulated map** | logits bit-identical to `all:v:all\|all:k:older<w>` and the same retention/read trace, for w ∈ {1, 2, 3, 5, 64} | w = 256 scores **identical, every field, every one of 1,024 positions**, to CODEC-MAP-1's `p_recent256` |
| **Window semantics** | at every read, K of age < w is the appended row bit for bit; older K and all V are codec/v1's read of the same appends; w = 0 is codec/v1 | w = 0 logits bit-identical to CH; every window's trace identical to CH's (210,188 events) |
| **Exact residency** | live append-born bytes = what w declares; a hidden f32 copy and a window wider than declared are both caught | holds at every window: live = declared, 0 strays, 0 window mismatches |

The declared residency is derived from the retained range and w, never
from the provider's own list lengths. Two provider mutants (window + 1, a
1-ulp "exact" row) turn the controls red. The run module was executed end
to end on fixtures first, including its stop on a moved reference.

## Results (gemma3-4b-it, bank 1, 2,048 rung, 1,024 scored)

Same rung, store pair and response variable as CODEC-MAP-1: per position,
d = KL(R‖arm) − KL(R‖C). Removal is 1 − the share of CH's tail damage the
arm reproduces, over CH's worst 5% (52 positions) and worst 1% (11).
C itself: top-1 0.816.

| arm | top-1 | mean d | p99 d | CH worst 5% removed | CH worst 1% removed | resident | × CH | exact K, share of bytes |
|---|---|---|---|---|---|---|---|---|
| CH (codec/v1) | 0.778 | +0.071 | +2.05 | — | — | 43.8 MB | 1.00 | 0% |
| w = 64 | 0.785 | +0.035 | +1.05 | 71% | 87% | 52.7 MB | 1.20 | 17% |
| w = 128 | 0.800 | +0.018 | +0.67 | 84% | 100% | 60.5 MB | 1.38 | 29% |
| **w = 256** | **0.803** | **+0.016** | **+0.66** | **94%** | **106%** | **76.3 MB** | **1.74** | **47%** |
| w = 512 | 0.808 | +0.011 | +0.62 | 93% | 103% | 107.7 MB | 2.46 | 66% |

Resident bytes are live append-born allocations at the end of the journey:
encoded V, encoded aged K, exact K, and the row lists. The decode scratch
(16.8 MB, within its bound) is the same for every arm and excluded. Removal
above 100% means the arm is better than C on CH's worst positions.

## Reading

1. **The real provider is the simulated map.** At w = 256 it reproduces
   CODEC-MAP-1's protection result exactly. The 94% / ≈100% tail removal is
   now a property of a shippable provider, not of a test instrument.
2. **The tail saturates between 128 and 256.**
   - The narrow tail is fully removed from w = 128.
   - The wide tail reaches 84% at 128 and 94% at 256; 512 adds nothing.
   - Mean damage keeps falling slowly (0.018 → 0.016 → 0.011).
3. **Bytes are the cost, and they come from the f32 window.**
   - An exact f32 K row costs 7.8× an encoded one (4,096 against 528 bytes).
   - At w = 256 the window is 10.9% of K/V rows but 47% of resident bytes:
     1.74× CH. That is still about a quarter of an all-f32 cache of the same
     rows (≈ 327 MB).
   - The window's precision was not varied here. A narrower exact
     representation is a separate question, not a finding.

## The window choice

The rule was declared in code before the run: *the smallest swept w whose
removal of CH's worst-5% AND worst-1% is each at least 0.9 × the sweep's
best.* The best removals were 0.94 and 1.06, so the thresholds are 0.846
and 0.954. **It chooses w = 256.**

The choice is close. w = 128 clears the narrow tail (1.00) and misses the
wide one by less than a point (0.838 against 0.846), on a 52-position set.
It holds 21% fewer bytes than w = 256. The rule stands as declared; the
near-miss is recorded so the freeze can decide whether to carry 128 as a
second arm.

## Limits

- **One model, one text, one length.** gemma3-4b-it, bank 1, one 2,048
  rung; the worst-1% set is 11 positions. Windows past the 1,024-position
  sliding history are not exercised (512 is the largest swept).
- **Calibration only.** The window was chosen on the same text the tail was
  localised on. Bank 3 is what tests it.
- **Residency is end-of-journey.** Peak during the journey and decode cost
  are not measured.
- **Two attempts.** The first run (`60e75d76`) passed stage 0, then stopped
  at w = 256 before SIM-W was evaluated, on an instrument fault: CODEC-MAP-1
  checkpoints a map's spec as `<name>=<clauses>`, and the run compared the
  bare clauses. Fixed in `21110ae6`, with the fixture e2e now writing the
  real format. The rerun is in a fresh store; the stopped one was not
  resumed or adjudicated. Both stage 0s have identical CH traces.
- **Revision identity.** The run is stamped with `21110ae6`. To remove
  commit-message trailers, the branch's commits were rebuilt with
  identical trees: `21110ae6` is `69c476e4` on the branch (tree
  `24ff4b5f`), `60e75d76` is `8e732a8f`. The evidence file maps all
  three.
- **Contended wall times.** A peer session's CPU job ran throughout. Every
  gate is bit-exact or deterministic, so it affects time only.

## What the freeze could ask (options, not choices)

- **The question:** does `codec-recent/v1` at w = 256 (4-bit) pass
  CODEC-2's frozen rules (B1–B4, unchanged) on a NEW held-out bank 3?
- **Arms:** C, CH, CH-recent(256), R, as in CODEC-2. Optionally
  CH-recent(128), as the rule's near-miss.
- **Residency:** carried as a reported quantity with its proof (live =
  declared), not a pass condition, unless the freeze names a budget.
- **Out:** decode cost; the exact window's precision.
