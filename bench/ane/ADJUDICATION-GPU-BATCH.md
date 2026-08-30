# ANE-2B's GPU control — is the ANE's 1.04x at N=8 special?

Banked 2026-08-25, battery regime, git `16bedaf2`. Two sessions.
Probe: `crates/larql-compute-metal/examples/ane2b_gpu_batch_control.rs`.
Same `5120 -> 17408`, same deterministic generators as ANE-0b/ANE-1/2B.

## Answer: no — but not for the reason that phrasing suggests

```
        ANE-2B      GPU arm A          GPU arm B
   N    T(N)/T(1)   f16 gemv x N       f32 sgemm_transb
   1      1.00      1.00  1.00         1.00  1.00
   2      1.00      2.00  2.01         1.00  0.99
   4      1.01      3.99  4.07         1.00  1.00
   8      1.04      8.07  8.16         0.98  0.98
```

- **The GPU *can* amortise a weight traversal across vectors.** Arm B, an
  existing tiled GEMM, is flat to N=8.
- **LARQL's f16 path cannot.** Arm A is strictly linear — 8 vectors cost
  8.1 traversals — because `f16_gemv` is a gemv. This is what LARQL does
  today at f16.

So the ANE's flat curve is not exotic hardware behaviour. It is flat for
the same reason arm B is: batched work amortises a weight read. What is
different is **availability**: the ANE gets it by default through Core ML
with no new kernel, whereas on Metal it needs an f16 GEMM that **does not
exist in this codebase**.

That reframes the question from a hardware verdict to an engineering
choice: *write an f16 GEMM for Metal, or use the engine that already
batches.*

## Arm B's result was pre-registered, not discovered

`sgemm_transb` tiles 32x32, so `M <= 8` always occupies exactly one
32-row tile: identical threadgroup count, identical K-loop, identical B
traffic for every N in the ladder. Arm B **had** to be flat. The
prediction is printed by the harness before the numbers so it cannot be
retro-fitted, and the measurement confirms it (1.00 / 1.00 / 1.00 / 0.98).

Read arm B as *"the GPU has at least 32-wide headroom available through
an existing kernel"*, not as a measured batching economy.

## A measurement hazard this control had to defuse first

`matmul_transb` silently falls back to CPU BLAS below `flop_threshold`,
whose default is **500 MFLOP**:

```
N = 1   178 MFLOP   under  -> CPU BLAS
N = 2   356 MFLOP   under  -> CPU BLAS
N = 4   713 MFLOP   over   -> GPU
N = 8  1426 MFLOP   over   -> GPU
```

Left alone, the sweep would have run its first two points on the CPU and
its last two on the GPU, and reported that **device switch** as a
batching curve. The harness lowers the threshold to the 100 K floor and
refuses to run if the smallest arm would still fall back.

## Per-vector, at N=8

```
ANE          0.208 ms/vector    f16, 178 MB
GPU arm A    0.627 ms/vector    f16, 178 MB   LARQL's path today
GPU arm B    0.476 ms/vector    f32, 356 MB   existing GEMM
```

The ANE is **3.0x better per vector than LARQL's current Metal f16 path**
at N=8. That comparison is fair — same dtype, same bytes, same shape.

**It is not fair against a good GPU GEMM, and none was measured.** Arm B
runs at ~91 GB/s, far off this machine's roofline: it is a naive 32x32
tiled f32 kernel, kernel-limited rather than hardware-limited. Its bytes
must not be normalised away either, since it is not bandwidth-bound. A
proper f16 simdgroup GEMM could plausibly beat the ANE, and nothing here
says otherwise.

## What this licenses

- **ANE batching is excellent but not anomalous.** The GPU amortises too,
  through a kernel that exists.
- **Against LARQL as it stands today**, the ANE is the only engine in the
  codebase that executes a batched weight-stationary f16 projection at
  all.
- **The ANE-as-verifier idea is not settled either way.** It rests on
  whether an f16 Metal GEMM gets written, which is a cost question, not
  a physics question.

## What it does NOT license

- No claim that the ANE beats a well-written GPU GEMM. Unmeasured.
- No absolute comparison from arm B — ratio only, f32, kernel-limited.
- Nothing about int8 (2C) or concurrency (ANE-3).

## Consequence for the ordering

ANE-4's drafter thesis is **unaffected** — it rests on ANE-2A's
wide-and-shallow placement rule and on the drafter running somewhere the
GPU is not, neither of which this touches.

What this does change is the **ANE-as-verifier** variant: it now has a
named competitor (an f16 Metal GEMM) and should not be pursued until
ANE-3 says whether the ANE can run without taking bandwidth from the GPU
anyway.
