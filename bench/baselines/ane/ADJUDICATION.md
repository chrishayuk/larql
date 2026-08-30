# ANE-0b — adjudication

Banked 2026-08-25. **Immutable once ANE-3 begins.**

Subject: Qwen3.8-27B FFN gate/up, `5120 -> 17408`, f16 weights × f32
activation, `f16_gemv`, 178,257,920 weight bytes per call.
Sessions `s1-battery` and `s2-battery`, warmup 64, measured 1024,
git `16bedaf22cf365702bfa320182da474175149fd4`.

**Power regime: BATTERY, by explicit override.** ANE-3 must be measured
in the same regime or this baseline must be re-taken.

## The control

```
                     s1        s2      spread
raw GB/s   (min)    288.7     289.1     0.15%     <- ANE-3's denominator
raw GB/s   (p50)    251.4     260.4     3.52%
gate/up ms (min)   0.6175    0.6166
gate/up ms (p50)   0.7090    0.6845
```

Both sessions are far inside the ±6% cross-session e2e floor, and the
min statistic is essentially identical across them. **The banked
denominator is 288.7 / 289.1 GB/s raw effective, min-of-1024.**

## The diagnostic (NOT the denominator)

```
                     s1        s2      spread
floor-adj GB/s (min) 356.1     358.4     0.64%
floor-adj GB/s (p50) 334.1     349.2     4.44%
dispatch floor ms    0.1169    0.1192    1.94%   (min)
dispatch floor ms    0.1754    0.1740    0.79%   (p50)
```

Floor as a share of the control's isolated wall-clock: **18.9% at min,
24.7% at p50.**

## Why the diagnostic is not banked as a finding

An earlier battery smoke run put the floor-adjusted rate at ~381 GB/s.
The two banked sessions put it at ~357. Over the same interval the raw
number moved only 293 → 289.

```
                smoke -> banked
raw               293 -> 289      1.4% move
floor-adjusted    381 -> 357      6.3% move
```

The adjusted number inherits all of the floor's variance, and the floor
is the noisier quantity. Wall-clock is the stable observable; the
adjustment is an interpretation. Scoring ANE-3 against the adjusted
number would import that instability into the result.

## Geometry control

`ffn_down` (17408 -> 5120, 640 threadgroups) came in at 285.1 / 283.3
GB/s against gate/up's 288.7 / 289.1 at 2176 threadgroups — within
0.6–1.3%. **No `ROWS_PER_TG` geometry surprise**: the two FFN
orientations move the same bytes at the same rate despite a 3.4×
difference in dispatch width.

## Probe self-controls

Both passed in both sessions:

- Projection recomputed on the CPU for one row per shape, `rel_err`
  1.0e-6 / 1.6e-7 / 1.3e-6. The kernel is doing the arithmetic.
- Implied rate below the 600 GB/s plausibility ceiling.

## What this licenses

1. **A same-shape, same-dtype, same-kernel-family denominator for
   ANE-3**: 288.7–289.1 GB/s raw, on battery, on this machine.
2. **A measured size for the bubble-recovery hypothesis (ANE-3 outcome
   2)**: 18.9–24.7% of an isolated invocation is not weight streaming.

## What it does NOT license

- **It does not prove the fabric is idle during that 19–25%.** The
  `dispatch_floor` line prices a *minimal round trip*; it does not show
  that the control has an equally long window in which the memory system
  is free. The GPU may be prefetching, and a minimal single-threadgroup
  dispatch is not the same dispatch as a 2176-threadgroup one. Whether
  that window is actually usable by another engine is precisely what
  ANE-3 measures — it must not be assumed here.
- **It is not a claim that the GPU is at the fabric ceiling.** ~357 GB/s
  adjusted sits near the roofline programme's 367, but by a subtraction
  whose stability is worse than the thing it is being used to argue
  about.
- **It says nothing about AC behaviour.** Battery by override.
