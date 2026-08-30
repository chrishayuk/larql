# ANE-3 — concurrency. CLOSED at outcome 2 by ANE-3b.

> **ANE-3b repaired the defect and the verdict is now stable. See the
> final section; the sections before it record the broken instrument and
> why, because the repair only makes sense against them.**

---

# ANE-3 (original run) — partly settled, and the top of the taxonomy is not.

Banked 2026-08-25, battery regime (matching ANE-0b), git `16bedaf2`.
Same `5120 -> 17408` f16 projection throughout. Two sessions, run in
**opposite condition orders**, 4 s steady-state window per condition,
barrier-released, samples filtered to the interval where both engines
were genuinely running.

---

## The two orders disagree, and that is the headline

```
                       forward (G, A, GA)     reverse (GA, A, G)
GPU alone      GB/s              233.5                  257.5     <- 9.8% apart
GPU concurrent GB/s              215.4                  206.8
GPU kept                          0.92                   0.80
ANE alone      GB/s              103.0                  106.5
ANE concurrent GB/s              107.8                  104.8
ANE kept                          1.05                   0.98
aggregate      GB/s              323.1                  311.5
verdict emitted            4 ADDITIVE            2 BUBBLE RECOVERY
```

**The verdict flipped on condition ordering alone.** Every concurrent
measurement agrees between the orders to ~4%, and the ANE arms agree to
~3%. The single unstable quantity is **GPU alone**, which reads 10%
higher when it runs last.

The mechanism is almost certainly power state: in forward order the
GPU-alone arm runs first, on an idle SoC with clocks ramping; in reverse
it runs after eight seconds of sustained load. The effect being measured
— "how much does the GPU lose when sharing?" — is the same size as that
drift, so the instrument cannot currently separate them.

The first session also produced **ANE kept = 1.05**: the ANE apparently
*faster* while sharing the machine. That is not physical for a contention
experiment and is the same artifact seen from the other side.

Order-reversal was the right falsifier. A clean interleave could not have
failed here; this did.

## What IS supported

Stable across both orders, to within ~4%:

- **The ANE pays essentially nothing to run alongside the GPU.** Latency
  ratio 1.02 in both orders; throughput kept 0.98–1.05.
- **Aggregate projection throughput exceeds the GPU alone in both
  orders.** Taking the most pessimistic pairing available — the highest
  GPU-alone reading (257.5) against the lowest concurrent aggregate
  (311.5) — the machine still delivers **1.21x** what the GPU delivers
  by itself. Forward order says 1.38x.
- **The GPU pays somewhere between 8% and 20%**, and the width of that
  range is the drift, not the phenomenon.

That is enough to call **outcome 2, bubble recovery, SUPPORTED**:
concurrent execution raises delivered work, and the GPU gives up less
than the ANE adds.

## What is NOT supported

- **Outcomes 3 and 4 are not distinguishable from 2 with this
  instrument.** The first session's "outcome 4" label came from the
  analyser's thresholds applied to a GPU-alone reading that the second
  session showed to be 10% low. No claim of genuine additive bandwidth
  is made.
- **The fabric ceiling was never tested.** Sustained isolated sums are
  336–364 GB/s and the concurrent aggregate is 311–323 GB/s, all under
  the ~400 GB/s prior. Nothing here approaches P2's boundary, so P2 is
  neither confirmed nor challenged.

## A correction to this track's own pre-registration

ANE-1's adjudication set up ANE-3 with:

```
GPU alone   288.9 GB/s
ANE alone   111.9 GB/s
sum         400.8 GB/s   "almost exactly the fabric ceiling"
```

Those were **min-latency equivalent rates** — peak instantaneous, from
the fastest single call. Sustained throughput over a 4 s window is much
lower: 233–257 GB/s for the GPU and 103–107 for the ANE. The neat
coincidence with ~400 GB/s was an artifact of comparing a peak statistic
against a sustained ceiling. Corrected here so it stops shaping
expectations.

## ANE-3b — what would settle it

The fix is small and specific: **equalise the power state across
conditions.** Run a fixed pre-load ramp (both engines, discarded) before
every condition, so no arm is measured on a cold SoC. Then repeat in both
orders and require the two to agree before reading a verdict.

Until that runs, the ordering of the remaining ladder should assume
outcome 2:

- `down_proj` k-split becomes worth testing — bubble recovery means
  moving more projection traffic to the ANE has some value, though less
  than additive bandwidth would imply.
- ANE-5 stays scoped to bubble-filling, not dense partitioning.
- **ANE-4 is untouched.** It never depended on this rung, and the one
  thing this rung shows most robustly — that the ANE runs at full speed
  while the GPU works — is exactly the property a drafter needs.

---

# ANE-3b — the repair, and the closing verdict

Banked 2026-08-25, same regime and shape. **Every condition now launches
both engines**, holds them at full load for 1.5 s, and releases them from
a common barrier; the engine not taking part runs with role `ramp` and
exits at `go`. All three conditions therefore begin from the same SoC
power state.

## The gate: GPU-alone must agree across orders

```
                    ANE-3 (no ramp)        ANE-3b (ramp)
forward  GPU alone        233.5                 254.4
reverse  GPU alone        257.5                 256.5
spread                     9.8%                  0.82%     <- PASS
```

Comfortably inside the ~6% cross-session floor, and an order of
magnitude better than before. ANE-alone is now identical across orders
(106.8 / 106.8).

## The result, now stable

```
                    forward     reverse
GPU alone   GB/s      254.4       256.5
GPU conc    GB/s      212.3       210.2
GPU kept               0.83        0.82
GPU latency x          1.05        1.15
ANE alone   GB/s      106.8       106.8
ANE conc    GB/s       94.2       107.8
ANE kept               0.88        1.01
ANE latency x          1.01        1.01
aggregate   GB/s      306.5       318.0
aggregate / GPU alone  1.20        1.24
verdict            OUTCOME 2   OUTCOME 2
```

**Both orders emit outcome 2. ANE-3 is closed at BUBBLE RECOVERY.**

The architectural statement, in one line:

> Running the ANE alongside the GPU delivers **1.20–1.24x** the
> projection throughput of the GPU alone, and costs the GPU **17–18%**.

## What remains uncertain, and it is not the verdict

**ANE concurrent throughput is the one wobbly quantity**: 94.2 vs 107.8
GB/s across orders, 13.5% apart, while everything else settled to within
a few percent. But the ANE's *latency* ratio is 1.01 in both orders —
per-operation speed is untouched. What varies is how many operations the
worker managed to issue.

The likely cause is CPU-side, not fabric-side: the ANE worker is a Python
process and the GPU worker is a Rust process spinning a core hard on a
submit/wait loop. Contention for CPU can throttle the ANE's *issue rate*
without touching ANE execution. If so, part of the measured "cost of
sharing" is an artifact of this harness's process structure rather than a
property of the silicon — and a production LARQL that drove both engines
from one process would not pay it.

That does not change the verdict (outcome 2 holds in both orders, and the
aggregate gain is present in both), but it means **1.20–1.24x is probably
a floor, not a ceiling**.

## Consequences

- **ANE-3 is closed. Do not keep squeezing it** looking for additive
  bandwidth; the fabric was never approached (306–318 GB/s concurrent
  against a ~400 GB/s prior) and P2 remains untested.
- **ANE-5 stays scoped to bubble-filling**, not dense partitioning.
- **The `down_proj` k-split is worth testing but modestly** — moving more
  target traffic to the ANE has real value at outcome 2, less than
  additive bandwidth would have implied.
- **ANE-4 is go, and this rung strengthens it.** The property a drafter
  needs is that the ANE runs at full per-op speed while the GPU works;
  the latency ratio of 1.01 in both orders says exactly that.

## Method note worth keeping

Both workers are long-lived, warm before the barrier, and stamp every
sample with an absolute epoch time; the analyser keeps only samples that
start and finish inside the intersection of the two active windows
(4833/4834 and 2418/2419 in session 1). That machinery worked. It was
the *power state between conditions*, not the synchronisation, that
turned out to be the weak point — a reminder that a well-built harness
can still be measuring a drifting machine.
