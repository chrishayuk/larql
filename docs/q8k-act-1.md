# Q8K-ACT-1: does a Q8_K activation close V3's CPU Q4_K gap?

Pre-registered 2026-09-22, before any implementation of the arm. The hypothesis,
forecast, workload, tolerance rules and verdict definitions below are FROZEN. The C0
section is filled in by running the frozen C0 procedure, and its thresholds follow
from the frozen rules. Nothing here claims a result.

Programme: VINDEX3 CPU execution. Follows the first `larql bench` VINDEX3 comparison
(2026-09-22) and a falsified predecessor diagnosis, recorded below.

---

## Background: the falsified predecessor

The first V3 row on Gemma 3 4B (`production-q4k`, 68.2 ms/token against V2
`larql-cpu` at 28.9 ms/token) was read as "the Q4_K projection is single-threaded",
because the ledger reported one worker slab per call. That reading was wrong.
`FusedKQuant` declares `CpuParallelism::LibraryOwned`, so the executor never
partitions it and slabs equal calls by design. `q4k_matvec_into` row-parallelises
itself through `spin_pool`. A thread sweep settled it:

| Threads | Steady ms/token | Projection throughput |
|---|---|---|
| 1 | 372 | 7 GB/s |
| 2 | 196 | 13 GB/s |
| 4 | 107 | 24 GB/s |
| 8 | 66 | 39 GB/s |

The scaling is near-linear at a tenth of the machine's bandwidth, so the kernel is
compute-bound. The bytes model's "26–33 ms/token" was a bandwidth forecast and does
not apply to a compute-bound kernel. That forecast stays falsified. It is not
reused here as evidence.

## Hypothesis

V3's CPU Q4_K gap to V2 is mainly caused by the activation form of the kernel. V3
runs `q4k_matvec_into`: f32 activation, dequantise-and-FMA per weight. V2's CPU
decode quantises the activation once to Q8_K and runs the integer-dot
`q4k_q8k_matvec` family. Executing the same stored Q4_K blocks through the Q8_K
route will cut V3 decode latency substantially.

## The arm

- **Name:** `production-q4k-q8k`, a new `ExecBackend`. It wants the same Q4_K pack
  as `production-q4k`.
- **Realization:** a new CPU projection plan for stored K-quant members whose
  format has a `q8k_matvec` kernel (Q4_K and Q6_K). The activation is quantised once
  per call with `quantize_x_to_q8k_into` and multiplied with
  `q4k_q8k_matvec_parallel`. That is the entry point V2's decode uses, including its
  `LARQL_Q4K_ASM` switch (default off; the value is recorded with every run).
  Q8_0 members and every non-K-quant operand keep their current realization.
- **Parallelism:** `LibraryOwned`, as `FusedKQuant`.
- **Declared lossy:** the realization is marked as approximating the activation.

**No reinterpretation.** `production-q4k` keeps its f32-activation semantics
unchanged. No scheduling, representation-selection, interpreter or Metal change is
part of this rung.

## Workload W (benchmark corpus B)

| Item | Value |
|---|---|
| Container | `~/chris-models/gemma3-4b-it.q4k.vindex3` (Q4_K pack compiled 2026-09-22 by `larql vindex3 represent --encoding Q4_K`; `index.json` sha256 `fa91746f8fa37c17bba4dc54675d4aa7c9ddd218ddac7ae5f7b27ac0a11a5a35`) |
| Prompt B | `Write a detailed essay about the history of Paris, from its founding to the present day.` |
| Prompt B ids (27) | `2,105,2364,107,6974,496,9813,20494,1003,506,4083,529,9079,236764,699,1061,37813,531,506,1861,1719,236761,106,107,105,4368,107` |
| Decode | `--warmup 4 -n 64` |
| Threads | 8 (bench auto default; `RAYON_NUM_THREADS` unset) |
| Command | `larql bench <container> --prompt "<B>" --backends <arm> -n 64 --warmup 4` |
| Sampling | 3 separate processes per arm, alternating `production-q4k`, `production-q4k-q8k` ×3; `uptime` recorded before each |
| Source | the implementation commit, named in the result |

## Performance verdict (P)

- **Measure:** the bench row's mean ms/token (the mean over the 64 post-warmup
  steps), taking the median of the 3 processes.
- **Frozen forecast:** 25–35 ms/token.
- **Verdict:** HIT if the median is in [25, 35]. Otherwise MISS, stated as LOW
  (< 25) or HIGH (> 35). A LOW is still a miss of the forecast and is reported that
  way.
- **Control:** `production-q4k` in the same session. If the control's median moves
  by more than 15% from 68.2 ms/token, the session is recorded as machine-disturbed
  and rerun before any verdict.

## Mechanism check (M)

- Run the 1/2/4/8 thread sweep with `larql vindex3 exec <container> --backend
  production-q4k-q8k --tokens <B ids> --generate 16`, reading the projection ledger's
  line for the new plan.
- **Forecast:** at 8 threads, projection throughput is ≥ 1.5× the baseline's
  39 GB/s, i.e. ≥ 58 GB/s.
- Whether scaling from 1 to 8 threads turns sub-linear (a move toward a bandwidth
  limit) is reported. It is not a gate.
- **The hypothesis is SUPPORTED** only if P is HIT and M's forecast holds. P HIT
  with M failing means the speed came from somewhere other than the proposed
  mechanism. That is recorded as unexplained, not as support.

## Fidelity verdict (F), separate from P

Q8_K activation quantisation is lossy by design. Bit-identity with `production-q4k`
is not required. The tolerances come from C0, run on a **held-out** characterisation
prompt A, never on B. Each tolerance is set by a rule frozen here, before C0 ran.

### C0 characterisation (before implementation, arm-blind)

- **Prompt A:** `Explain how photosynthesis works in plants, step by step.`
- **Prompt A ids (20):**
  `2,105,2364,107,155122,1217,93036,4146,528,6485,236764,2918,684,2918,236761,106,107,105,4368,107`
- **C0-a (projection).** `cargo run --release -p larql-vindex --example q8k_act_c0 --
  <container> --prompt <A ids> --generate 24 --capture 8`. On every Q4_K/Q6_K
  projection of the last 8 steady steps, it runs the existing Q8_K kernel against
  the existing f32-activation kernel on the same stored blocks and the same captured
  activation. The per-call metric is relative L2, `‖y_q8k − y_f32‖ / ‖y_f32‖`.
- **C0-e (end to end).** Teacher-force S_A (prompt A ids followed by the 24 ids C0-a
  generates) through `production` on `gemma3-4b-it.vindex3` (BF16) and through
  `production-q4k` on the Q4_K container, using `larql vindex3 exec --logit-dump`.
  `scripts/q8k_act_1_logits.py` reports per-position KL(BF16 ‖ Q4_K) and top-1
  agreement. This is the weight-quantisation budget, the approximation this
  pipeline already accepts.

### Tolerance rules (frozen)

- **τ1** = 2 × (C0-a maximum relative L2 over all calls), rounded up to one
  significant figure.
- **τ2** = 0.25 × (C0-e mean per-position KL(BF16 ‖ Q4_K)). Activation quantisation
  may add at most a quarter of the divergence weight quantisation already
  introduces.

### Gates, evaluated on corpus B

- **F0 (wiring).** A unit test: on a fixture operand, the arm's output is
  bit-identical to `quantize_x_to_q8k_into` followed by `q4k_q8k_matvec_parallel` on
  the same input. This proves the arm runs the kernel it declares.
- **F1 (projection).** C0-a's procedure on prompt B (`--generate 24 --capture 8`).
  The maximum per-call relative L2 must be ≤ τ1.
- **F2 (logits).** Teacher-force S_B (prompt B ids followed by `production-q4k`'s 64
  greedy ids) through `production-q4k` and `production-q4k-q8k`. The mean
  per-position KL(`production-q4k` ‖ arm) must be ≤ τ2.
- **F3 (decisions).** On S_B, the arm's top-1 agreement with `production-q4k` must be
  ≥ `production-q4k`'s top-1 agreement with BF16 `production` on S_B. Activation
  quantisation may not flip more positions than weight quantisation does.

**The F verdict is PASS only if F0–F3 all pass.** F is reported beside P, never
folded into it. For example, "P HIT, F FAIL" is its own result.

**Descriptive, not gated:** the first position where free-running greedy output
diverges from `production-q4k`, and how the arm's text compares with V2's.

## V2 comparison: descriptive only, and why

Greedy-token identity with V2 was proposed as a gate and rejected before the freeze.
The gate already fails before any intervention, for two independent reasons:

1. **V2 feeds a doubled BOS.** The V2 bench's prompt B is 28 ids beginning `2, 2`.
   The Gemma chat template emits `<bos>`, and `encode_prompt` prepends another. V3
   feeds 27 ids. This is a V2 defect, recorded here and not fixed in this rung.
2. **V2 and V3 hold different weights.** V2's extract is Q4_K with Q6_K for V/down.
   V3's pack comes from `represent --encoding Q4_K`. On prompt B, V2's greedy text
   opens with "From Celtic Outpost to Global Icon" and V3's with "The Ever-Evolving
   Heart".

A gate that cannot pass at baseline invites reinterpretation. The comparable V2
number (`larql-cpu` 28.9 ms/token, `standard` 34.8 ms/token) is context for P, not a
target.

## Order

1. Commit this document and the C0 instruments (`q8k_act_c0`, `Captured`
   accessors, `scripts/q8k_act_1_logits.py`).
2. Run C0, fill in the C0 results and thresholds below, and commit.
3. Implement the arm and F0, and commit.
4. Run P, M and F1–F3 on W, adjudicate against this document, and commit.

## C0 results

Run 2026-09-23 against freeze commit `47807926`, from a release build of that
commit, with `LARQL_Q4K_ASM` unset. Another session was running builds and tests on
the machine at the time. C0 measures numerical error, not timing, so that load does
not affect it.

### C0-a (projection), prompt A

The 24 generated ids (S_A = prompt A ids followed by these):
`19058,236764,1531,236789,236751,2541,1679,93036,1271,506,7646,1657,6485,1161,531,2490,26808,1131,2780,236888,5715,236789,236751,496`

| Member | Calls | rel L2 median | p99 | max | max\|Δ\|/rms median | max |
|---|---|---|---|---|---|---|
| Q4_K | 1904 | 1.551e-2 | 4.260e-2 | 7.236e-2 | 7.136e-2 | 2.599e-1 |
| Q6_K | 0 | — | — | — | — | — |

The pack is Q4_K throughout, so no Q6_K calls exist. The Q6_K route is part of the
arm, but this corpus does not exercise it.

### C0-e (end to end), S_A, 44 positions

KL(BF16 `production` ‖ `production-q4k`): **mean 2.2405e-1**, max 8.280 (position
12), median 3.5e-3. Top-1 agreement 0.9091 (4 of 44 positions disagree).

The mean is dominated by one prompt position: position 12 alone contributes 0.188
of the 0.224. The frozen rule uses the mean, and it is applied as written. F2 is
also a mean, so the arm is judged by the same statistic. The median is reported
here for context only.

### Thresholds (by the frozen rules)

- **τ1** = 2 × 7.236e-2 = 0.1447, rounded up to one significant figure = **0.2**.
- **τ2** = 0.25 × 2.2405e-1 = **5.60e-2**.

### Instrument control (run with C0, before the arm)

A median relative L2 of 1.5% from 8-bit activations with per-256 scales is larger
than a uniform-rounding estimate would suggest, so C0-a was checked for
confounding. The example also runs the f32 kernel on the Q8_K-**dequantised**
activation and compares that with the Q8_K kernel:

| Member | control rel L2 median | max |
|---|---|---|
| Q4_K | 2.254e-7 | 1.283e-6 |

The two kernels agree to f32 rounding once they see the same activation. C0-a's
error is therefore the activation's quantisation error alone, consistent with
high-crest-factor projection inputs, where a few outlier channels set the block
scale. The control changes no threshold.

## Results: fidelity verdict (F)

Run 2026-09-23 against implementation commit `baa81944`, from a release build of
that commit, with `LARQL_Q4K_ASM` unset. The machine was under unrelated load at
the time: another session's GW-STATE-1 training run and a cargo test in another
worktree. F measures numbers, not time, so the load does not affect it.

S_B is the 27 prompt B ids followed by `production-q4k`'s 64 greedy ids (91
positions):
`1408,669,19865,236772,236788,101672,19657,236787,562,10358,529,9079,108,50429,236764,506,999,17698,529,46768,2130,506,999,17698,529,11543,2130,532,496,4185,3988,573,1610,236764,8013,236764,532,6540,236764,41041,496,4083,618,53091,532,3996,618,506,52996,3707,529,1061,15729,236761,9567,3925,5889,236858,236745,886,529,11059,11381,236764`

In F2's run, the arm's ledger shows every Q4_K projection executed as
`FusedKQuantQ8k` (21,658 calls). The head stays `FusedQ8`, as before.

| Gate | Measured | Threshold | Verdict |
|---|---|---|---|
| F0 wiring | output bit-identical to `quantize_x_to_q8k` + `q4k_q8k_matvec_parallel` (Q4_K, Q6_K); a mutation that swaps in the f32 kernel turns it red | exact | **PASS** |
| F1 projection (prompt B, 1904 Q4_K calls) | max rel L2 **7.362e-2** (median 1.518e-2, p99 3.521e-2); instrument control max 1.174e-6 | ≤ τ1 = 0.2 | **PASS** |
| F2 logits (S_B) | mean KL(`production-q4k` ‖ arm) **6.9655e-3**, max 1.009e-1 | ≤ τ2 = 5.60e-2 | **PASS** |
| F3 decisions (S_B) | arm vs `production-q4k` top-1 agreement **0.9890** (1 of 91 positions flips) | ≥ `production-q4k` vs BF16 **0.9341** (6 of 91) | **PASS** |

**F verdict: PASS.** On this corpus, Q8_K activation quantisation adds about a
tenth of the logit divergence that Q4_K weight quantisation already introduces
(mean KL 7.0e-3 vs 7.3e-2), and flips one decision where weight quantisation flips
six.

## Results: performance verdict (P) and mechanism (M)

Run 2026-09-23 from 00:52, on a release build of `baa81944` (the code at HEAD
equals that commit), with `LARQL_Q4K_ASM` unset and 8 bench threads. By then the
earlier load had cleared: the 1-minute load average was between 3.4 and 4.3
throughout, with no build, test or training process running. An earlier session at
load 29 was disturbed and discarded (see below).

### P: workload W, 3 processes per arm, alternating

| Run | load (1 min) | `production-q4k` (control) | `production-q4k-q8k` (arm) |
|---|---|---|---|
| 1 | 3.36 / 3.56 | 72.81 ms/token | 33.27 ms/token |
| 2 | 3.35 / 3.78 | 70.40 ms/token | 33.35 ms/token |
| 3 | 3.99 / 4.05 | 69.22 ms/token | 33.37 ms/token |
| **median** | | **70.40** (+3.2% vs 68.2, within the 15% bound: session valid) | **33.35** |

Prefill (27 tokens) is about 1.48 s for the control and about 0.49 s for the arm.
Generation fingerprints are identical across all three processes of each arm
(`3f36dfba…` control, `3c4cbac7…` arm).

**P verdict: HIT.** 33.35 ms/token is inside the frozen 25–35 ms/token. That is a
2.11× decode speedup over `production-q4k` on the same stored Q4_K pack.

### M: thread sweep, `vindex3 exec --backend production-q4k-q8k --generate 16`

| Threads | Steady ms/token | Projection throughput | f32 arm (predecessor sweep) |
|---|---|---|---|
| 1 | 87 | 29 GB/s | 372 ms/token, 7 GB/s |
| 2 | 53 | 48 GB/s | 196 ms/token, 13 GB/s |
| 4 | 35 | 71 GB/s | 107 ms/token, 24 GB/s |
| 8 | 32 | **78 GB/s** | 66 ms/token, 39 GB/s |

**M forecast held:** 78 GB/s ≥ 58 GB/s, which is 2.0× the baseline's 39 GB/s.
Scaling is now clearly sub-linear, and it flattens between 4 and 8 threads. The f32
arm scaled almost linearly across the same range. The Q8_K arm has moved from a
compute-bound regime toward a shared limit, most likely memory bandwidth or
per-token fixed costs. This rung does not test which.

### Descriptive (not gated)

- Free-running greedy output first diverges from `production-q4k` at generated
  token 1: the title word, a near-tie consistent with F3's single flipped position.
  From there the arm's text opens "## From Celtic Outpost to Global Icon: A History
  of Paris". V2's text opens the same way. That is a coincidence of a near-tie, not
  a parity claim: V2 holds different weights and feeds a doubled BOS.
- V2 context from the 2026-09-22 session: `larql-cpu` 28.9 ms/token, `standard`
  34.8 ms/token. The V3 arm now sits in that band. Those were different sessions on
  different weights, so this is not a head-to-head.

## Adjudication

| Verdict | Result |
|---|---|
| **P** (performance) | **HIT**: 33.35 ms/token in [25, 35] |
| **M** (mechanism) | **HOLDS**: 78 GB/s ≥ 58 GB/s |
| **F** (fidelity) | **PASS**: F0–F3 |
| **Hypothesis** | **SUPPORTED**: P HIT and M holds, with fidelity inside the frozen contract |

V3's CPU Q4_K gap to V2 was mostly the activation form of the kernel. It was not
an executor-wide cost and not a threading defect. With V2's Q8_K integer-dot route,
the same stored pack decodes at 33.4 ms/token instead of 70.4. Activation
quantisation costs about a tenth of the divergence weight quantisation already
introduces.

**What this does not establish:**
- It says nothing about Metal.
- It says nothing about other models or corpora. Only one prompt was scored for F,
  and one workload for P.
- It does not settle whether `production-q4k-q8k` should become a default. That is
  a separate decision, and it would need a wider fidelity corpus first.
- The Q6_K route is implemented and passes F0, but this Q4_K-only pack never
  exercised it end to end.

### The disturbed session (not a reading)

At 00:45 the 1-minute load average was 29 (a GW-STATE-1 training run and a cargo
test in another worktree). A smoke comparison had the control at 124 ms/token
(+82%) and the arm at 185 ms/token. Under the frozen rule that session is not a
reading. It is recorded because the arm degraded worse than the control there,
which suggests the spin pool behind `q4k_q8k_matvec_parallel` is more
contention-sensitive than the f32 path. That is a separate question, and this rung
does not answer it.
