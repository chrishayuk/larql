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

*(not yet run)*
