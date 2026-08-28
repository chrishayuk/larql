# Continuation State (STATE programme)

**Thesis.** Continuation state is an explicit **liveness model** for neural
execution state. Some state is live, some is recomputable, some is
unreachable, some might eventually be approximable — and the engine should
know which is which, per layer, with a named correctness contract. The
first genuinely useful consequence is exact garbage collection of provably
dead KV; the flashier consequence ("replace a chunk with a boundary
vector") was tested adversarially and killed. This file records both.

The code seams live in `crates/larql-vindex/src/format/vindex3/opplan/exec/`:

- `continuation.rs` — geometry: **what must persist** (KV / recurrent /
  stateless, one entry per layer).
- `representation.rs` (STATE-1) — **how the persisting state is physically
  held**, and what getting it back promises: `ContinuationPlan`,
  `CorrectnessContract` (only `Exact` exists; a bounded contract waits for
  its first non-exact consumer, and its metric is the Residual ABI's —
  horizon-T predictive KL with a spacing constraint).
- `retire.rs` + the `retired_spans` seam (STATE-2) — a provider may declare
  spans of the consumed past **retired**; both traversals validate and
  forward the declaration into every `AttentionStepCall`; the reference
  backend honours it, production and device refuse it by name.

## STATE-2: the boundary-context hypothesis, falsified

**Hypothesis under test** (from the chuk-mlx context system, 2026-03/04):
a context chunk's last rows — its *boundary* — carry enough of the chunk
to stand in for its interior KV.

**Protocol** (`examples/state2_boundary_retirement.rs`; frozen artifacts
in `bench/state2-boundary-retirement/`): Granite 4.2-3b (full attention,
40 layers, no window confound), reference backend end to end. One prompt
with a distinctive 224-position chunk C1 (interior fact `RX-4471`, 146
positions before the boundary); one shared prefill; matched-footprint
arms retain k ∈ {1,4,16} rows of C1 — its **last** k (boundary), its
**first** k, or a seeded **random** k — plus a retain-nothing floor.
Teacher-forced KL in bits (f64 log-softmax) against the canonical arm
over 24 steps. Capability pre-screen passed: canonical answers
`RX-4471.` exactly. Three conditions:

| condition | zero | boundary-16 | random-16 | first-16 |
|---|---|---|---|---|
| continuation — KL mean bits/step | 2.09 | 2.06 | 1.76 | 1.55 |
| query-back (retire after prefill) — step-1 KL | 7.58 | 7.63 | 7.60 | 7.78 |
| query-back-early (question consumed under retirement) — step-1 KL | 14.9 | 17.2 | 19.0 | 17.0 |

**Verdict (pre-registered kill criterion): killed.**

1. **The boundary carries nothing.** boundary-k ≈ random-k ≈ zero at every
   footprint in every condition. The one mild outlier is first-k — an
   early-position sink/anchor effect, the opposite of the hypothesis.
2. **A fact is addressable only through its own rows.** Every retiring arm
   loses the interior fact at the first answer token — including when the
   question was prefilled with full attention to C1. The answer is not
   absorbed into the question's rows; decode addresses the chunk's
   interior KV directly. Retirement-after-consumption is not safe either.
3. **Continuation is not cheap** (~2.1 bits/step): continuing a document
   *is* reading it back.

Do **not** build a `Boundary` representation. What STATE-2 falsified is
*passive* boundary state; an *engineered* summary written at the boundary
and then retiring the source behind it is a different treatment
(STATE-4, one disciplined attempt, after the exact work is banked — the
harness runs it via its `retire_at` argument).

## What survived, and what is next

**The retirement abstraction is the recovered value.** Behind a sliding
window, positions are *provably unreachable* — and the STATE-2 gates
show retiring them is **bit-for-bit free** (`exec/tests/retire.rs`:
`retiring_behind_every_sliding_window_is_free`). For a Gemma/Glimmer
stack that means: sliding layers only ever need the live window; only the
global layers retain full history.

**STATE-3 — window-shadow retirement, exact: BUILT.**
`exec/window.rs` (`WindowKvState`) physically frees every row in a
sliding layer's window shadow as it appends, keeping the store
position-aligned (freed rows become empty husks, so a wrong read is a
loud panic and the executor contract is untouched — every backend
qualifies unchanged). Gates (`exec/tests/window.rs`): 61 decode steps
bit-identical to the full store on the reference AND production
backends; residency as arithmetic — the sliding layer holds exactly its
window, the full layer exactly the context, payload bytes agree to the
byte at two well-separated lengths; the batch prefill frees shadows as
it goes; the KV-only refusal names the provider. What is bounded is the
payload (`4·kv_dim` B/row vs ~24 B of husk metadata); a flat
representation that also bounds the metadata is a later storage rung.

Next: integrate the policy into the continuation plan (the geometry
should carry the span so the provider stops trusting `window` alone),
then the real-model long-context residency/throughput bench — Gemma 3
4B via capability-scoped encode (29 sliding + 5 global) or Glimmer.
