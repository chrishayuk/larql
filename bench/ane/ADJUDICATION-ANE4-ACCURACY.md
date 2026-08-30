# ANE-4 accuracy causality — controlled, and the mechanism was wrong

Banked 2026-08-25, battery regime, coremltools 9.0.
Harness: `bench/ane/ane4_accuracy.py`.

ANE-4 step 1 attributed an error gap to "where `down_proj` ran". **That
attribution was not controlled** — variants A and B differed in device
*and* in every weight matrix, activation width and reduction length.
This is the control, plus the ladder that the control made worth running.

---

## 1. The causal control: one artifact, two devices

The FFN block at intermediate 16384, compiled **once**, loaded twice with
different `computeUnits`. Same graph, same weights, same input, same
reference. Only the device differs.

```
                ANE dev   ANE rel_rms    CPU dev   CPU rel_rms     ratio
ffn_16384           ANE     3.302e-02        CPU     2.986e-04    110.6x
```

**The device is the cause. Confirmed.** And the effect is *larger* than
the uncontrolled A/B comparison suggested — 110.6x, not 26x. The earlier
figure was diluted by comparing ANE-at-16384 against CPU-at-17408.

The ANE arm reproduces step 1's 3.302e-02 exactly, as it must: same
artifact, same weights.

## 2. The ladder: the mechanism was NOT what step 1 claimed

Isolated `linear`, n = 5120, same tensors within each pair, both devices.

```
              ANE rel_rms   CPU rel_rms    ratio
k =  5120       1.550e-02     1.441e-03    10.8x
k =  8192       1.514e-02     2.362e-03     6.4x
k = 12288       1.522e-02     2.868e-03     5.3x
k = 16384       1.500e-02     3.330e-03     4.5x
```

```
across a 3.20x increase in k:
   ANE error grew  0.97x     (i.e. FLAT)
   CPU error grew  2.31x     (sqrt(k) would be 1.79x, linear 3.20x)
```

**ANE error is constant in reduction depth. CPU error grows with it.**

Step 1 said the gap came from "fp16 accumulation across a 16384-long
reduction on the ANE". **That is refuted.** Accumulation error grows with
the number of terms; the ANE's does not move at all across a 3.2x range.
It is the *CPU* whose error behaves like an accumulation process.

A relative error that is flat in k is the signature of a **fixed
precision floor**, not a summation effect: roughly 1.5e-2 RMS, about
2^-6, i.e. on the order of six or seven effective mantissa bits —
materially coarser than the fp16 the graph nominally requests.

Note what the ratio column does: it narrows from 10.8x to 4.5x purely
because the CPU is climbing toward a stationary ANE floor. Quoting a
single "ANE is Nx worse" number is therefore meaningless without the k it
was measured at.

## 3. This falsifies the optimistic k-split hypothesis, before it was run

The appealing possibility was that splitting `down_proj` into two
8704-term reductions might **recover** accuracy relative to one
16384-term reduction — making the split "residency up, accuracy also
up".

That rested on the error being a function of reduction depth. It is not,
so the *mechanism* by which a split would recover accuracy is gone.

**But that is a prior, not a prediction.** The flat single-projection
floor does not mathematically determine a split projection's final
`rel_rms`. Splitting changes the numerical structure: two partial errors
that are even partly uncorrelated may cancel, while cancellation between
the two *true* partial sums would amplify relative error in the result.
So:

> **Pre-registered for the k-split: no recovery is the mechanism-based
> prior. Accuracy remains an empirical outcome, and needs a CPU-split arm
> to separate "splitting changes the arithmetic" from "the ANE changes
> the precision".**

Running the ladder first was still worth it: it removed the *reason* to
expect recovery, which stops a null result from looking like a surprise
and stops a positive one from being over-read.

The split is still worth doing for the placement and latency questions.
It should no longer be expected to help accuracy.

## 4. What this gives VINDEX3

The ANE backend description gains a precision term, and it is simpler
than a curve:

```
ANE linear (f16 graph):
    admissible   reduction axis <= 16384
    preferred    reduction axis >= ~4992   (f16) / ~4096 (int8 weights)
    accuracy     OBSERVED synthetic-tensor floor, these f16 linear probes:
                 relative RMS ~1.5e-2, FLAT in reduction depth
                 -> a floor, not an accumulation term

CPU linear (f16 graph):
    accuracy     relative RMS grows with reduction depth
                 1.4e-3 at k=5120 -> 3.3e-3 at k=16384
```

A planner can use a flat floor directly: any op whose output tolerance is
tighter than ~1.5e-2 must not go to the ANE, regardless of its shape.

## 5. Consequences

- **Target-side ANE execution is now clearly gated on quality.** ~1.5%
  relative RMS per projection is not a rounding detail for a 27B model.
  Any ANE-5 plan needs a predictive-units gate (KL / NLL), of the kind
  CPU-5 used — not a cosine or an rel_rms eyeball.
- **The drafter is unaffected in principle and must still be measured.**
  A drafter's errors are filtered by the target's verification, so a
  1.5e-2 floor may cost acceptance rate rather than correctness. That is
  an acceptance-rate experiment, not an accuracy one.
- **ANE-4 step 1's placement and latency findings stand unchanged.** Only
  its mechanism sentence was wrong.

## What this does NOT license

- No claim about *why* the floor sits at ~1.5e-2. The number is
  consistent with reduced effective mantissa, but the mechanism was not
  probed and Core ML's internal representation was not inspected.
- Synthetic pseudorandom weights and activations. A real weight/activation
  distribution — which is not zero-mean-unit-ish and has outliers — could
  sit somewhere else on this floor.
- f16 graphs only. int8 weights were not run through this control.
- One session, battery regime.
