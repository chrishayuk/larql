# NVFP4-Q8-1: does a Q8 activation close V3's CPU NVFP4 gap?

Pre-registered 2026-09-23, after reconnaissance and before any implementation of the arm. The hypothesis,
forecasts, workload, verdict definitions and fidelity rule below are FROZEN. Nothing here claims a result.

Programme: VINDEX3 CPU execution. It follows Q8K-ACT-1 (merged #512), which moved stored Q4_K onto a Q8_K
activation, and #529, which decodes NVFP4 in NEON registers.

---

## Reconnaissance: the same bytes, two arithmetics

Measured before this freeze on Gemma 3 4B. The binary was main `e8307501` plus #530 (the per-plan rate
report) plus #529 (NEON NVFP4 decode), merged locally as `851e4214`. The run used a 24-token prompt and
`--generate 32`, four arms interleaved, three repeats, 60 s cooldowns, on AC power. The raw runs are in
`bench/nvfp4-q8-1/recon/`.

| arm | ms/token | tok/s | main projection plan | bytes/token | rate |
|---|---:|---:|---|---:|---:|
| `production`, bf16 cap | 79 | 12.6 | `FusedBf16` | 7.40 GB | 114 GB/s |
| `production`, default policy | 66–67 | 15.0 | `FusedQ8` (Q8 × **f32**) | 3.55 GB | 80–82 GB/s |
| `production-nvfp4` on the NVFP4 pack | **105–106** | **9.5** | `FusedNvfp4` (NVFP4 × f32) | **1.80 GB** | **~20 GB/s** |
| `production-q4k-q8k` on the Q4_K pack | **33** | **30.2** | `FusedKQuantQ8k` (Q4_K × Q8_K) | **1.80 GB** | **111–112 GB/s** |

The two 4-bit packs stream the same 1.80 GB of projection weights per token.
- **Projection time:** NVFP4 × f32 takes ~90 ms, against ~16 ms for Q4_K × Q8_K, a ~5.6× gap.
- **Whole token:** ~105 ms against ~33 ms, ~3.2×.
- **The remainder:** in both arms it is about 15–17 ms/token, mostly the output head at `FusedQ8`, about
  10 ms.

The bandwidth needed is demonstrated on this machine, on this model, on the production path. The NVFP4 gap
is the arithmetic, not the bytes and not the hardware.

## Hypothesis

V3's CPU NVFP4 gap to Q4_K is mainly caused by the activation form of the kernel. `FusedNvfp4` decodes
each FP4 code to float and multiply-adds against an f32 activation. Executing the same stored NVFP4 bytes
against a Q8 activation with integer dot products will cut the NVFP4 projection time, and the decode
latency, substantially.

## The arm

- **Name:** `production-nvfp4-q8`, a new `ExecBackend`. It wants the same NVFP4 pack as `production-nvfp4`.
- **Realization:** a new physical projection plan, `FusedNvfp4Q8`, for stored NVFP4 operands.
  - **The weight side is exact.** E2M1 magnitudes {0, ½, 1, 1½, 2, 3, 4, 6} doubled are the integers
    {0, 1, 2, 3, 4, 6, 8, 12}. A 16-entry table maps each 4-bit code (sign included) to an int8, with no
    rounding. The group's E4M3 scale and the tensor scale, halved to undo the doubling, are applied to each
    group's int32 sum.
  - **The activation** is quantised once per call to Q8 with a per-block scale. Each block is a whole number
    of 16-element NVFP4 groups.
  - **The inner product** is int8 × int8 → int32, on SDOT where the target has it. The weights are never
    widened to f32 or to an intermediate buffer.
  - The block size, table and blocking are implementation choices. They are recorded, not frozen.
- **Provider identity:** registered under its own `LoweringIdentity`, as Q8K-ACT-1's arm is. The same pins
  compute different numbers under it, so an image prepared for one provider must never execute under the
  other.
- **Declared lossy:** the realization is marked as approximating the activation.

**No reinterpretation.** `production-nvfp4` keeps its f32-activation semantics unchanged. The output head
(`FusedQ8`) and every non-NVFP4 operand keep their current realization. No Metal, scheduling, representation,
interpreter or accounting change is part of this rung.

## Workload W

The reconnaissance workload, unchanged: Gemma 3 4B, `gemma3-4b-it.nvfp4.vindex3` (and
`gemma3-4b-it.q4k.vindex3` for the control), the same 24 prompt ids, `--generate 32`. The metric is the
steady, last-half ms/token. Three interleaved repeats of `production-nvfp4`, `production-nvfp4-q8` and
`production-q4k-q8k`, with 60 s cooldowns, on AC, with the load average recorded per run. The verdict uses
the median of the three.

## Performance verdict (P)

On the median steady ms/token of `production-nvfp4-q8`:

| verdict | condition |
|---|---|
| **HIT** | ≤ 40 ms/token (≥ 2.6× over the 105 ms reconnaissance) |
| **STRETCH** (reported beside HIT) | ≤ 35 ms/token, the Q4_K × Q8_K regime |
| **PARTIAL** | 40 < x ≤ 70 ms/token |
| **MISS** | > 70 ms/token |

The same run's `production-nvfp4` must reproduce the reconnaissance within ±10% (95–116 ms). Otherwise the run
is **void** and repeated; it is not judged.

## Mechanism check (M)

From the same runs' per-plan report:

- **M holds** if `FusedNvfp4Q8` streams ≥ 80 GB/s (≈ 22.5 ms for 1.80 GB). **Stretch:** ≥ 100 GB/s, parity
  with `FusedKQuantQ8k`'s 111 GB/s.
- **Integrity, required for any M reading:**
  - `FusedNvfp4Q8` carries the same 1.80 GB and 238 calls that `FusedNvfp4` carries in the control arm;
  - `FusedNvfp4` reports 0 calls in the candidate arm;
  - `runtime compile: 0`.

  A candidate that widened, or fell back, is not measured.
- A HIT without M is reported as a HIT whose mechanism is unexplained. It is not adopted.

## Fidelity verdict (F), separate from P

The change isolates activation quantisation: the same stored weights, with the only difference being the
activation's representation and the integer accumulation. So F compares the two NVFP4 arms directly, through
`larql vindex3 measure` (MEASURE-PLAN-1), over Q-BANK-1's 69 sequences:

- **reference** `production-nvfp4` on the NVFP4 pack (stored);
- **candidate** `production-nvfp4-q8` on the same pack (stored);
- **yardstick:** Q8K-ACT-1's accepted activation change, `production-q4k` against `production-q4k-q8k` on
  the Q4_K pack, measured by the same procedure on the same bank, in the same session.

The procedure must return **admissible**, with complete facts. F **PASSES** if both hold:
1. mean KL(reference ‖ candidate) ≤ 1.5 × the yardstick's mean KL;
2. top-1 agreement ≥ 99.5% on positions where the reference's margin is ≥ 0.5 (the procedure's confident band).

Otherwise F **FAILS**. F is judged independently of P, and a P HIT with an F FAIL is not adopted. If the
yardstick itself is inadmissible, F is reported as **uninformative**, not as a pass.

## Witnesses before the verdict runs (implementation acceptance)

- **Exactness of the weight side:** for random NVFP4 matrices, the int8 table decode times scale equals
  `larql_models::quant::nvfp4::dequantize` exactly, element for element.
- **Kernel parity:** `FusedNvfp4Q8` against the f32 reference computed on the dequantised weights and the
  quantised activation agrees to f32 reassociation. That isolates the kernel from the quantisation.
- **Quantisation bound:** against `FusedNvfp4` (f32 activation), the relative error is within the Q8 block
  bound, recorded on a fixed seed.
- **Plan enumeration and identity:** the new plan appears in the lowerings and plan-enumeration tests, and the
  arm refuses an image prepared under another provider.
- Each witness is committed RED against a skeleton, then turned GREEN, then mutation-proven.

## Order

1. This freeze, with the reconnaissance evidence.
2. The arm: skeleton and witnesses (RED), then the kernel (GREEN), then mutation checks.
3. F: `vindex3 measure`, reference against candidate and the yardstick.
4. P and M: the workload on a quiet machine.
5. Adjudication, committed.

## Out of scope

- Metal.
- The output head's representation.
- A Q8 activation for plans other than NVFP4.
- SCOPED-ACCOUNTING-1. This rung's M reading is single-execution and single-threaded, which the 2026-09-06
  ruling allows.
- CPU lowering. By the reconnaissance, it is second-order until projections fall.

---

## Adjudication (2026-09-24)

Judged against the frozen text above. Nothing in it was edited. Binary: source `a258c33d`, release build, with no
`LARQL_*` environment.

### F: PASS

Evidence: `bench/nvfp4-q8-1/f/`. Both procedures are **admissible**, with complete facts. Each covers Q-BANK-1's 69
samples, 1810 positions, in bank `74eded7e`, and reports "changed: the arm only".

| comparison | mean KL (nats) | top-1, margin ≥ 0.5 |
|---|---:|---:|
| candidate: `production-nvfp4-q8` vs `production-nvfp4` (same stored pack) | **6.305e-4** | **100.00%** (n=1012) |
| yardstick: `production-q4k-q8k` vs `production-q4k` | 6.550e-3 | 100.00% (n=1040) |

- **Criterion 1 holds:** 6.305e-4 ≤ 1.5 × 6.550e-3 = 9.825e-3. The candidate is about 10× below the accepted Q8_K change.
- **Criterion 2 holds:** 100.00% ≥ 99.5%.
- **Where the flips are:** they sit where the reference itself is undecided. At margin 0–0.1, top-1 agreement is
  93.8% (n=227); across all positions it is 99.17%.
- **Execution attested:** `attest.txt` records the realization each arm ran:
  - `FusedNvfp4` in `production-nvfp4`, `FusedNvfp4Q8` in `production-nvfp4-q8`;
  - `FusedKQuant` in `production-q4k`, `FusedKQuantQ8k` in `production-q4k-q8k`;
  - 238 operands each, 0 runtime compiles.

### P: PARTIAL

Evidence: `bench/nvfp4-q8-1/pm/`. Three interleaved repeats, 60 s cooldowns, on AC power. The 1-min load at each run
start was 1.7–2.2, with the 5- and 15-min averages still falling after a peer's compile.

| arm | steady ms/token (runs) | median | tok/s |
|---|---|---:|---:|
| `production-nvfp4` | 104.98, 105.70, 105.15 | 105.15 | 9.51 |
| `production-nvfp4-q8` | 47.93, 47.33, 47.34 | **47.34** | 21.12 |
| `production-q4k-q8k` | 33.01, 32.85, 32.86 | 32.86 | 30.43 |

- **The run counts:** the control arm reproduces the reconnaissance (105.15 is inside 95–116 ms).
- **Verdict:** 47.34 ms/token falls in 40 < x ≤ 70, so the verdict is **PARTIAL**. That is a 2.22× end-to-end
  speedup over `production-nvfp4`. It is not the HIT (≤ 40) and not the STRETCH (≤ 35).

### M: does not hold (integrity satisfied)

| plan | calls | GB | projection ms (median) | GB/s (runs) |
|---|---:|---:|---:|---|
| `FusedNvfp4` (control arm) | 238 | 1.80 | 90.2 | 20.2, 20.0, 19.8 |
| `FusedNvfp4Q8` (candidate arm) | 238 | 1.80 | 32.8 | 55.5, 54.6, 55.1 |
| `FusedKQuantQ8k` (yardstick) | 238 | 1.80 | 16.1 | 112.1, 112.2, 112.6 |

- **Integrity holds, so M can be read:** the same 1.80 GB and 238 calls as the control, 0 `FusedNvfp4` calls in the
  candidate arm, 0 runtime compiles.
- **Reading:** 55.1 GB/s < 80 GB/s, so M does not hold.
- **The projection itself:** its time falls 2.75× (90.2 → 32.8 ms).

### Hypothesis

**Supported in direction, and not to the frozen level.** On the same stored NVFP4 bytes, the activation form accounts
for most of the gap:
- **Projection time:** 90 → 33 ms.
- **Decode latency:** 105 → 47 ms/token.
- **Fidelity:** about 10× inside the accepted Q8_K yardstick.

The arm does not reach the Q4_K × Q8_K regime. `FusedNvfp4Q8` streams at half `FusedKQuantQ8k`'s rate on the same
bytes.

**Why M misses** was recorded before any P/M run, in `bench/nvfp4-q8-1/diagnostic-split-threshold.md`, which is not
an adjudication input. Gemma 3 4B's NVFP4 attention projections (1.5–2.9 MB) fall under the executor's 4 MB split
threshold and run single-threaded, about 0.30 GB/token:
- the worker slabs in the reports show it: 748 slabs over 238 calls for both NVFP4 arms, against 238 for the
  library-threaded Q4_K kernel;
- this rung froze scheduling out of scope, so the threshold is not changed here.

### Disposition

- **Kept as a named arm, not a default:** `production-nvfp4-q8` stays opt-in under its own provider identity, as
  `production-q4k-q8k` does. `production-nvfp4` keeps its f32-activation meaning. A PARTIAL with M not holding is not
  adopted.
- **The next question is its own freeze:** the executor's split policy is written in representation-independent
  resident bytes, and compressing the weights moved whole matrix classes under it. The intervention is the policy,
  not this kernel.
