# LARQL CPU execution — state of play and roadmap

Written 2026-08-25, at the end of the CPU-5 / CPU-6 work, so that
someone returning cold (including the author) can pick it up without
re-deriving anything.

---

## 1. Where things stand, in one table

```
programme   subject                             state
CPU-1..4Y   the kernel ladder to Q4 x Q8        DONE, merged (PR #308)
CPU-5       does Q4 x Q8 preserve the model?    CLOSED — FAIL
CPU-6       paired revalidation of the candidate  FROZEN, PARKED, NOT RUN
CPU-7       make the CPU execute differently    NOT STARTED
```

Immutable identities:

```
df36ca9f   candidate implementation freeze   crates/ frozen here
a87f441d   CPU-6 protocol / banks / adjudicator freeze
4a47e74a   CPU-6 parked, unrun
branch     cpu5-q4q8-quality   (all of the above)
branch     cpu7-parallel-execution  (from 4a47e74a)
```

---

## 2. The headline result, stated precisely

There is a CPU integer projection path that is **32% faster than what
ships**, and it is **not validated**.

```
shipped     Q8 x F32                348 ms/token   2.87 tok/s
candidate   Q8[64] x asym-Q8[16]    264 ms/token   3.79 tok/s
                                    81% of attainable CPU bandwidth
```

Quality history of that candidate:

```
quality-bank-1   PASS   KL 0.000283 vs gate 0.000316   (discovery bank)
quality-bank-2   FAIL   KL 0.000407 vs gate 0.000316   (spent)
quality-bank-3A/3B      FROZEN, NOT RUN
```

**Why Bank 2 failed is a defect in the GATE, not necessarily the
representation.** G1 was intended as "candidate degradation <= 2x
shipped degradation" and was implemented as "candidate KL on bank N <=
2x shipped KL measured on BANK 1" — an absolute number carried across
prompt sets. Whether the candidate is relatively worse on Bank 2 is
unknowable and stays unknowable; Bank 2 is spent and its missing anchor
must never be measured now.

CPU-6 repairs the gate and re-tests the UNCHANGED candidate.

---

## 3. Durable findings (independent of whether the candidate ever ships)

### Representation

- **Blanket Q4 x Q8 is REJECTED on Qwen3.8.** KL 0.05384, top-1 90.98%,
  **38 flips at BF16 margin >= 0.10**.
- **Logit KL scales as the quantisation step SQUARED.** Q4/Q8 weight
  cost ratio 332.7x against a step ratio squared of 329x — agreement to
  1%. This turns exception-set search into arithmetic.
- **No class-level exception set rescues uniform int4.** Measured: R0
  blanket 0.05013, R3 (FFN restored, ~70% of bytes) 0.02770 — restoring
  70% of the bytes removed only 45% of the damage. The cheapest possible
  split (head only, 9% of bytes) is still ~27x the gate and saves 4.5%
  of traffic.
- **Sensitivity is architecture-specific.** Qwen3.8's damage concentrates
  in attention+GatedDelta at 1.83x the per-byte rate — the OPPOSITE knee
  from PR #299's Granite late-FFN finding. A recipe carried between
  models would have been wrong.
- **Per-tensor int8 activation is not a usable representation.** Residual
  peak/rms is 28-36 at depth, leaving ~2 effective bits. Blocking fixes
  it: rel_rms 0.476 -> 0.047 with exact weights.
- **Activation scale geometry is nearly free in BYTES and not in
  COMPUTE.** A weight scale is paid per block per row; the activation is
  one vector (320 B at block 64, 1.3 KB at block 16, against 14.4 GB of
  weights). But per-block work scales with block COUNT, which is where
  the compute went.
- **Asymmetric coding is worth ~1.25x at block 16** and, once the kernel
  is right, costs ~1 ms.

### Kernel / physical execution

```
K1  scalar precomputed weight sums    SLOWER  868 vs 757 ms   FALSIFIED
K2  batch integer reductions          +20%    bit-identical
K3  stay vector-domain through floats  1.73x  sym16 484 -> 279
K4  vector precomputed sums            ~0%    index consumed, time unmoved
K5  delete transient fold buffers      1.95x  asym16 516 -> 264
```

- **Three of five hypotheses were wrong, all for one reason**: they
  removed ARITHMETIC while the binding constraint was data movement.
- **The cheap discriminator is apparent GB/s against a known wall.** At
  58 GB/s against 127, no amount of ALU deletion can help. That reading
  was available before K1 and would have saved two rungs.
- **Preserve operand VARIANCE DOMAINS.** `activation_scale` and
  `activation_midpoint` are row-invariant; the weight scale is not.
  Materialising their product per row repeated row-invariant work 5,120
  times and roughly doubled the inner loop.

### Instruments

- **A byte ledger that counts stored model bytes prices a kernel 3x
  wrong.** It cannot see transient per-row materialisation.
- **A cost model is indexed by kernel geometry, not just
  representation.** Bytes/rate predicted 237 ms where the truth was
  757 ms, while predicting the block-64 arm to within 5%.
- **Attainable CPU bandwidth is 127 GB/s and saturates at TWO threads**
  (1t 76 -> 2t 122 -> flat to 16t). GPU reaches 367 GB/s on the same
  unified memory. Not movable by kernels — P-cluster fabric ports.

---

## 4. Resuming CPU-6 (about 4.6 hours on a quiet machine)

Nothing is re-authored and nothing is re-frozen.

```
git checkout a87f441d          # or any HEAD whose crates/ == df36ca9f
python3 bench/prompts/cpu6_freeze.py verify <container>
python3 bench/prompts/cpu6_run.py <container> <out3a> --bank 3a --sha df36ca9f...
python3 bench/prompts/cpu6_run.py <container> <out3b> --bank 3b --sha df36ca9f...
python3 bench/prompts/cpu6_adjudicate.py <out3a> --bank 3a
python3 bench/prompts/cpu6_adjudicate.py <out3b> --bank 3b
```

Six arms: BF16 reference, shipped anchor and candidate, per bank, with
the anchor measured INSIDE the bank it adjudicates. Run all six before
adjudicating either bank.

```
3A PASS + 3B PASS   VALIDATED — productionise
either FAIL         RETIRE. No Bank 4.
```

Full protocol: `bench/prompts/CPU6-VALIDATION.md`.

**What to look at besides PASS/FAIL.** Whether the local anchor explains
the cross-bank movement:

```
A  shipped and candidate both move, proportionally   -> CPU-5 was a broken anchor
B  shipped stable, candidate varies                  -> distribution-specific sensitivity
C  both vary, candidate more                         -> pairing helps but robustness is real
D  both banks comfortably under 2x                   -> strong validation
```

B or C would be the more important finding, and would apply to every
future VINDEX3 representation rather than just this candidate.

---

## 5. If CPU-6 passes: productionisation order

The candidate's authority is currently entirely environment variables
(`LARQL_CPU_ARITHMETIC`, `LARQL_CPU_ACT_BLOCK`, `LARQL_CPU_ACT_CODE`),
resolved once per process. Correct for an experiment, wrong for a
shipped plan.

```
1  PhysicalProjectionPlan owns the arithmetic choice
2  ActivationRep becomes first-class: Q8 { block: 16, code: Asymmetric }
3  the planner selects the kernel from weight/activation/accumulator reps
4  env vars demoted to explicit debug override only
5  persist the chosen physical representation on disk (no startup rebuild)
```

The execution plan should be able to state `Q8[64] x AsymQ8[16] -> I32
-> F32` without anyone knowing which shell variable produces it.

Also owed: a proper steady-state performance gate under
`cpu::environment`. The 264 ms is a 6-token `--generate` measurement, not
a benchmark-grade number.

---

## 6. CPU-7 — make the CPU execute differently

The conventional path is nearly exhausted:

```
27.02 GB weights / token / 127 GB/s attainable  =  213 ms floor
candidate is at 264 ms  =  81% of attainable
remaining kernel headroom ~24%, then it is over  ->  ~4.7 tok/s ceiling
```

Past that requires changing one of the other terms in

```
tokens/sec ~ (useful tokens per weight read x effective bandwidth) / bytes touched
```

### CPU-7A — is 127 GB/s a DRAM wall or a single-traversal wall?

```
A: gate_proj 6 workers, then up_proj 6 workers      (sequential)
B: gate_proj 3 workers + up_proj 3 workers          (concurrent)
   same activation, same bytes, same worker budget
```

Bands: `~121` nothing available; `130-140` modest; `140-160` significant;
`>160` the wall was single-GEMV concurrency.

**Prior: likely to find nothing.** The roofline programme already
measured saturation at two threads, flat to sixteen, with a cache/DRAM
control. Cheap enough to settle definitively for this access pattern.

### CPU-7B — weight-stationary, the high-value one

```
one Q8 projection x 1, 2, 4, 8 activation vectors
load each weight tile ONCE, apply to N positions, discard
```

Prediction, from SDOT throughput against the memory wall:

```
compute floor at 2 SDOT/cycle/core   40 ms  ->  5.3x headroom  ->  N~5 near-free
compute floor at 4 SDOT/cycle/core   20 ms  -> 10.6x headroom  ->  N~11
```

So time should stay roughly flat to N~4 and then bend. **If it bends at
N=2, the whole speculative-decoding direction is worth much less** — which
is why this cheap synthetic probe gates the expensive programme.

Instrument transient bytes in the probe. The activation side scales with
N (N sets of codes/scales/mids, N accumulators per row), and per-row
transient state is exactly where the ledger has already under-counted.

### If 7B works: the programme it unlocks

```
draft K tokens cheaply  ->  verify all K in ONE weight traversal
```

Turns batch-1 GEMV into a narrow GEMM. At ~350 ms per 4-position
verification and 2.5 tokens accepted on average, that is ~7 tok/s. The
drafter could itself be a cheap VINDEX3 physical plan — fewer layers,
lower precision — which makes the planner answer not just "how do I
execute this model" but "how much model do I need for this purpose".

### Ranked opportunity list

```
idea                                   potential   difficulty
multi-token speculative verification      2-3x+        high
lower effective bits + correction         1.5-2x       high
  (Q4 base + sparse/low-rank residual; progressive bit planes;
   NVFP4's e2m1 alphabet is integer-realisable, so SDOT applies)
concurrent independent projections        1.2-1.5x?    low     <- 7A, probably nil
temporal / delta projection               unknown      research
  (is x_{t+1} - x_t compressible enough to skip columns of W?)
weight-stationary microbatching           large        medium  <- 7B
fused physical projection packs           5-20%        medium
register tiling / ILP                     5-15%        medium
ordinary kernel polish                    a few %      low
```

---

## 7. Method rules earned the hard way

- **Freeze gates before measuring, and never move them after.** CPU-5
  failed by 28.6% and the gate did not move.
- **The anchor must live in the bank it adjudicates.** That single
  omission is the whole of CPU-5's failure.
- **A control must be shown to fail on known-different input**, or it is
  decoration. Every gate here was tested by planting a violation.
- **Kill background work by captured PID, never a pattern.** `pkill -f`
  missed a renamed binary twice; the survivor corrupted a dump and
  nearly deleted a live directory.
- **Verify output integrity, not process exit.** A truncated dump that
  is a whole number of records short raises nothing and scores fewer
  positions.
- **A plan built from a contaminated number inherits the
  contamination.** A 15 h estimate came from throughput measured while a
  duplicate process was running; the truth was 4.6 h.
