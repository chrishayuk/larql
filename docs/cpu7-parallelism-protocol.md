# CPU-7A / 7B — memory-level parallelism and weight-stationary execution

Pre-registered 2026-08-25, before any measurement. Gates in this file are
FROZEN: they were written without a number in hand and they do not move
afterwards. See `docs/cpu-execution-roadmap.md` §6 for why this programme
exists, and `bench/prompts/quality-bank-1/CPU5-Q4Q8-QUALITY.md` for the
method this inherits.

---

## The question

The conventional single-token CPU path is nearly exhausted:

```
27.02 GB weights / token / 127 GB/s attainable  =  213 ms floor
CPU-5 candidate                                 =  264 ms  (81% of attainable)
remaining kernel headroom ~24%                  ->  ~4.7 tok/s ceiling
```

Everything past that has to move a different term of

```
tokens/sec  ~  (tokens per weight read x effective bandwidth) / bytes touched
```

CPU-7A asks whether **effective bandwidth** is really 127 GB/s, or whether
127 is the ceiling of ONE traversal and independent traversals can overlap.
CPU-7B asks whether **tokens per weight read** can exceed one.

Both are cheap synthetic probes deliberately run BEFORE the expensive
programme (speculative decoding) that depends on them.

---

## Instrument controls — these run FIRST and gate everything

A probe that cannot reproduce a known number is not measuring what it
claims to. None of the arms below are readable until all three pass.

```
C1  single-projection reference
    6 workers, Q8[64] x asym-Q8[16], DRAM-resident working set.
    MUST land within 10% of the banked 118.0 GB/s for this kernel.
    Outside that, the probe is measuring itself and every arm is void.

C2  cache/DRAM discriminator
    The same sweep at an L2-resident working set MUST substantially
    outrun the DRAM sweep. If it does not, the loop is issue-bound and
    the DRAM number is a FLOOR, not a ceiling — the probe says so in its
    own output rather than letting the rate get quoted as attainable.
    (Inherited from `membw_probe`, which was built after C12 measured
    identical GB/s at both sizes and had measured its own issue rate.)

C3  numerical equivalence, 7B only
    The N-vector weight-stationary kernel MUST produce results
    BIT-IDENTICAL to N separate single-vector calls. Same SDOT sequence
    per (row, vector), same fold order, so identity is achievable and
    anything less means the arms compute different things and the timing
    comparison is meaningless.
```

C3 is planted-violation tested: the check is shown to FAIL when one
activation vector is perturbed in its last element, before it is trusted
to pass. A control that has never fired is decoration.

---

## CPU-7A — is 127 GB/s a DRAM wall or a single-traversal wall?

Two independent projections that share one activation — the MLP's
`gate_proj` and `up_proj`, the most common such pair in the model.

```
arm S  sequential   gate on 6 workers, then up on 6 workers
arm C  concurrent   gate on 3 workers + up on 3 workers, all 6 live
                    at once behind one barrier
```

Same bytes, same worker budget, same activation, same kernel. The only
difference is whether the six workers are traversing one matrix or two.

**The trap this arm is built to avoid.** The shipped executor collapses a
projection reached from inside a rayon worker to a single thread
(`caller_owns_the_machine`), which is correct for the shipped path and
fatal here: the obvious implementation of arm C — spawn two tasks, each
calling `project` — would silently compare 6 workers against 2 and report
a catastrophic loss that is entirely an artifact. So arm C builds ONE flat
list of six row-ranges across two matrices and never nests. The probe
reports the live thread count per arm so this is visible rather than
assumed.

### FROZEN bands — aggregate GB/s, arm C

```
<= 125        NOTHING AVAILABLE. 127 is a DRAM wall; concurrency is not
              a lever and the executor should keep its current policy.
126 - 140     modest. Worth a scheduling change only if it is free.
141 - 160     significant. The executor should schedule independent
              physical operators concurrently.
> 160         the wall was single-GEMV concurrency, not DRAM. Reopens
              the whole executor scheduling question.
```

**Prior: expect nothing.** The roofline programme already measured
saturation at TWO threads and flat to sixteen, with a cache/DRAM control
(`membw_probe`). That says the memory system is saturated long before six
workers, and two saturating streams cannot sum to more than one. This arm
is run because it is cheap and because that argument is about a pure read
stream, not about this access pattern — not because the outcome is in
doubt.

---

## CPU-7B — weight-stationary, the arm that matters

One Q8 projection against N activation vectors, each weight tile loaded
ONCE into registers and applied to all N before it is discarded.

```
for weight_tile:            <- streamed from DRAM exactly once
    load tile
    for n in 0..N:          <- N SDOT passes over a resident tile
        accumulate into out[n]
```

N in {1, 2, 4, 8}. N=1 is the baseline AND control C1.

### FROZEN gate — total cost relative to N=1

Not GB/s. The whole point is amortisation, so the gate is the multiplier:

```
              total sweep cost      per-vector cost      verdict
N=1                     1.00x                 1.00x     baseline
N=2            <= 1.35x                <= 0.68x         amortising
N=4            <= 1.90x                <= 0.48x         STRONG
N=8            <= 3.20x                <= 0.40x         STRONG

N=2 > 1.70x                                             DEAD — the kernel
                                                        is not weight-bound
                                                        and speculation is
                                                        worth much less
```

Read the N=2 row first. A bend there means each extra vector costs nearly
a full traversal, which is the outcome that retires the speculative-decode
direction before anything expensive is built.

### Pre-registered PREDICTION, with its falsifier

SDOT throughput against the memory wall gives 5-11x compute headroom:

```
compute floor at 2 SDOT/cycle/core   ~40 ms   ->  5.3x  ->  N~5 near-free
compute floor at 4 SDOT/cycle/core   ~20 ms   -> 10.6x  ->  N~11
```

**Predicted:** roughly flat to N=4, bending by N=8.
**Falsifier:** a bend at N=2. If N=2 costs more than 1.70x, the prediction
is wrong and the stated mechanism (memory-bound on weight loads) is wrong
with it — report that, do not rescue it with a tuned N.

### The ledger this arm owes

The byte ledger must count TRANSIENT traffic, not just stored model
bytes. A ledger that counted stored bytes priced a kernel 3x wrong once
already, and K5 turned out to be a load/store OP THROUGHPUT finding, not
a DRAM finding — the buffers it deleted were 2560 B and lived in L1.

So this probe reports three quantities per arm, separately, and never
sums them into one rate:

```
weight bytes       DRAM class. Streamed once per sweep regardless of N.
                   The 127 GB/s wall is comparable to THIS and only this.
activation bytes   L1 class. Re-read per row, and scales with N.
                   Reported so a flat weight-GB/s cannot hide N-fold
                   growth in op traffic.
per-row op count   SDOTs and vector load/stores per row, per N.
                   K5's mechanism was visible here and nowhere else.
```

---

## Scope limits, stated up front

- This is a SYNTHETIC probe on production-shaped data (5120x5120,
  Q8[64] x asym-Q8[16]), not the real decode. The banked synthetic->real
  correction on this programme is x1.047; nothing here licenses a tok/s
  claim.
- A PASS on 7B licenses exactly one claim: *the weight-stationary kernel
  amortises on this shape at this operating point*. It does not say a
  hybrid architecture can supply N independent positions — that is 7C,
  and Qwen3.8's GatedDelta recurrence is the reason it is a separate rung.
- Neither arm says anything about acceptance rate, which is the other
  factor in every speculative-decoding estimate and is not measured here.
- Run on AC power on a quiet machine. A hot or contended box invents
  1.5-3x phantom differences, and both arms here are ratio measurements
  where that lands directly on the verdict.

---

## What a result does next

```
7A > 160         reopen executor scheduling; independent operators
                 become a concurrency group.
7A <= 125        close the concurrency lever permanently. Bank it.

7B N=2 > 1.70x   RETIRE the weight-stationary / speculative direction.
                 CPU-8 (fewer bits) becomes the only open axis.
7B N=4 <= 1.90x  earn CPU-7C: the same primitive in a layer-shaped
                 harness, across positions, through the real recurrence.
```

---

# AMENDMENT 1 — before the adjudicating run, after the provisional one

Three changes. The original text above is untouched; this is what stands.

## C4 is added: the sweep is run in BOTH orders

The provisional run was made on battery, and the operator has accepted
battery for the adjudicating run. That is defensible for CPU-7B in a way
it would not be for an absolute rate, because 7B's gate is a ratio to an
`N = 1` cell measured in the SAME run under the same power state — the
internal-anchor property whose absence is what made the 7A gate malformed.

But it leaves exactly one live threat. The cells run in the order
`N = 1, 2, 4, 8`, so `N = 8` always measures last, after ~30 s of sustained
load, which is when thermal or power drift would bite hardest. An ordering
artifact would inflate the `N = 8` multiplier and look like a real bend.

```
C4  order-reversal falsifier
    The DRAM sweep is run forward (1, 2, 4, 8) and reversed (8, 4, 2, 1).
    Every multiplier must agree between the two to within 0.15, and the
    verdict is read from the WORSE of the two.
    Outside 0.15 the ordering artifact is live and no 7B band is readable.
```

Interleaving is deliberately NOT the mitigation. A clean A/B/A/B has
already produced a false +2.4% on this machine; a reversal test can FAIL,
which an interleave cannot.

## The 7A gate is recorded as malformed, and not rehabilitated

The band is stated in absolute GB/s against an anchor measured elsewhere.
A clean run does not repair that, and a clean number inside the band must
not be read as passing it.

```
original gate      frozen but MALFORMED — reported for historical
                   completeness only, never adjudicated
interpretation     concurrent / same-run single-traversal ratio
```

At the provisional 0.95x, the finding is negative and needs no rescue:
independent projection concurrency does not unlock additional memory
bandwidth on this part.

## Scope limit sharpened, because the provisional result invites the error

`N = 2` costing nothing means **two projection vectors are nearly free
relative to one weight traversal of that projection.** It does NOT mean a
two-position verification pass is free. A real two-position layer also
carries the recurrence, normalisation, activation quantisation, attention
and GatedDelta state transitions, residual and output handling — none of
which this probe contains, and not all of which amortise.

How much of the projection-level amortisation survives a complete layer is
the entire question CPU-7C exists to answer, and it is not answered here.

---

## RUN LOG

### 2026-08-25, provisional — VOID for verdict purposes

First execution of `cpu7_probe`. Recorded because it happened, not because
it counts: the machine was on **battery** at 93% with an unrelated Python
job at ~42% CPU and load average 1.68. Both arms are ratio measurements,
which is the exact shape a thermal or contention artifact corrupts without
looking wrong. **No band in this file may be adjudicated against these
numbers.**

All five controls PASSED, and those are load-independent:

```
C1   127.4 GB/s against 118.0 banked                    within 10%
C2   199.0 cache / 127.4 dram = 1.56x                   >= 1.5x
C3a  worst relative error 3.773e-5                      < 1e-4
C3b  64 rows x 8 vectors bit-identical to N=1
C3c  planted violation moved its vector and only its vector
```

Provisional arms, for shape only:

```
7B dram    N=1 15.74 ms   N=2 0.98x   N=4 1.29x   N=8 2.44x
7B cache   N=1 10.79 ms   N=2 1.15x   N=4 2.03x   N=8 3.77x
7A         one 133.7 GB/s   sequential 121.8   concurrent 127.5
```

### Two instrument defects this run exposed, to be fixed before the real one

**1. The 7A band is stated in ABSOLUTE GB/s, and should be relative to the
probe's own single-traversal reference.** The frozen bands read 127.5 as
"modest" because 127.5 > 125. But this probe's own `one matrix` arm
measures the single traversal at 133.7 GB/s on the same box in the same
run, so the concurrent arm is 0.95x a single traversal — nothing at all.
The band inherited an absolute number measured elsewhere, which is
precisely how CPU-5's G1 failed: *an anchor must live in the bank it
adjudicates.*

The gate does NOT move — it is frozen and it stays frozen. Both readings
get reported, and the internally-anchored one is the honest one. Any
future band on this programme is stated as a ratio to a reference measured
in the same run.

**2. `sequential` versus `concurrent` conflates two changes.** The
sequential arm carries a phase barrier between the two matrices (needed, or
it is not sequential); the concurrent arm has none. So `concurrent` beating
`sequential` by 4.7% may be barrier removal rather than memory-level
parallelism. The clean comparison is `one matrix` against `concurrent`,
which is already in the probe and shows concurrency LOSING.

### 2026-08-25, adjudicating run — CPU-7B PASSES, CPU-7A CLOSED NEGATIVE

Battery, 92% -> 91%, load 1.51, machine 90.9% idle, no unrelated job (the
Python process in the provisional run had exited on its own; nothing was
killed). Binary `92b4c897dbd71840`, same frozen protocol.

All six controls PASS.

```
C1   121.3 GB/s against 118.0 banked                    +2.8%, within 10%
C2   199.2 cache / 121.3 dram = 1.64x                   >= 1.5x
C3a  worst relative error 3.773e-5                      < 1e-4
C3b  64 rows x 8 vectors bit-identical to N=1
C3c  planted violation moved its vector and only its vector
C4   worst forward/reversed drift 0.07                  <= 0.15
```

**CPU-7B, read from the worse of the two orders per C4:**

```
        total cost    per-vector    frozen band       verdict
N=2          1.02x         0.51x    <= 1.35           amortising
N=4          1.27x         0.32x    <= 1.90           STRONG
N=8          2.41x         0.30x    <= 3.20           STRONG
```

The pre-registered prediction — flat to `N ~ 4`, bending by `N = 8` — is
what happened, and the named falsifier (a bend at `N = 2`, i.e. > 1.70x)
did not fire; `N = 2` is free to within 2%.

**The mechanism is legible, not merely the outcome.** The DRAM and
cache-resident arms converge exactly where compute becomes the limit:

```
        dram GB/s   cache GB/s
N=1         121.3        199.2     dram memory-bound, cache is not
N=2         122.2        172.3
N=4          98.8         98.6     converged — both compute-bound
N=8          51.9         52.8     converged
```

Below the knee the DRAM arm is hiding SDOT work under memory latency;
above it, both arms pay the same arithmetic. So the finding is not "the
kernel got faster" — it is that **batch-1 decode was leaving a large
fraction of the machine's arithmetic capacity idle waiting on memory, and
extra activation vectors fill it without proportional weight traffic.**

Also confirmed here and not assumed: the hoisted `w . 1` term is NOT what
moved. It is 0.1 G SDOTs at every `N`, against 0.1 -> 0.9 G that scale
with `N`, so the amortisation is traffic, not the arithmetic shortcut the
kernel doc warned could be mistaken for it.

**CPU-7A: CLOSED, negative.**

```
one matrix        126.1 GB/s
two, sequential   112.4
two, concurrent   121.0     =  0.96x a single traversal
```

Concurrency does not unlock bandwidth on this part. The malformed absolute
band happens to agree this time (121.0 <= 125), but per AMENDMENT 1 it is
not what the finding rests on — the same-run ratio is. The `sequential`
arm is slower than `one` because it carries the phase barrier that makes
it honestly sequential; that is why `one` is the reference and not `seq`.

**On running this on battery.** The provisional and adjudicating runs
agree to <= 0.04 on every multiplier (0.98/0.99, 1.29/1.23, 2.44/2.34),
across a power state, a background job appearing and leaving, and both
sweep orders. The ratio is empirically insensitive to the conditions that
were feared, which is a stronger statement than the argument that it
should be. Absolute rates DID move — C1 read 127.4 provisionally against
121.3 here — so the caution was correctly aimed at absolute numbers and
correctly relaxed for the ratio.

### What this licenses, exactly

> Two projection vectors are nearly free relative to one weight traversal
> of that projection, on this shape, at this operating point.

It does not say a two-position verification pass is free. That is CPU-7C.
