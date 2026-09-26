# V3-FFN-SLICE-1: can a VINDEX3 routed-expert operation move to a stateless worker?

**Class: FREEZE.** Nothing here has been implemented or run. This page fixes
the question, the arms, the gates and their order before any execution code
is written. It changes only by a new, dated section. A gate that turns out to
be unsatisfiable is a failed gate and leads to a new freeze. It is not
reinterpreted.

## Question

Can the routed-expert contribution of a VINDEX3 execution be computed by
independently bound, stateless workers, each holding only a declared
`(layer, expert)` slice, **without changing the execution at all**?

This is a question about VINDEX3. It is not a port of the legacy distributed
stack ([`docs/ffn/distributed.md`](ffn/distributed.md)). `RemoteMoeBackend`,
`MoeExpertBackend` and the `/v1/expert/*` routes may supply transport and
protocol ideas. They are not an oracle, and no legacy execution code runs in
any arm.

## Background

- PR #507 added `ExecutionSlice::LayerRange` workers
  ([scope](vindex3/runtime-followups.md#cpu-layer-workers)). A worker runs
  whole layers, attention included, so it recomputes the full prefix on every
  step to avoid remote KV state. `distributed::ensure_supported` refuses
  non-softmax attention, hyper-connections and attention residual, which
  rules out K3.
- A routed-expert computation has no temporal state. Given a layer, a row
  and a selected expert, the output depends only on the weights. An expert
  worker needs no KV, no continuation and no replay.
- In the V3 executor, routing, expert execution and combination are **one**
  backend call: `PlanBackend::routed_ffn(RoutedFfnCall)`
  (`opplan/exec/backend.rs`). The CPU production implementation
  (`opplan/exec/production.rs`) selects experts, then for each selected
  `(expert, weight)` in selection order computes the expert output (bias
  included) and accumulates `acc += weight * v`. Floating-point addition is
  not associative, so exact parity is possible only if the split keeps this
  order.

## Design constraints (frozen)

1. **The seam is inside `routed_ffn`, not around it.** The coordinator
   routes (router input, logits, selection, weights) and combines. Workers
   compute **unweighted per-expert outputs, down bias included**. The
   coordinator accumulates them in the local path's selection order, with
   the same expression. Workers never receive routing weights and never sum
   across experts. This is the wire contract for every backend, not only
   for CPU: an expert output **includes** that expert's down bias and
   **excludes** its routing weight. A future Metal or CUDA worker whose
   kernel would fuse the weighting in must still return the unweighted
   output, or declare a different contract in its own freeze.
2. **Two new slice kinds**, alongside the existing variants:
   - a coordinator slice: every operand of the plan **except** the routed
     expert banks (router, norms, attention, dense/shared FFN, embedding,
     head);
   - `RoutedExperts { layers, experts }` for a worker: only the expert-bank
     rows of the owned experts on the owned layers. No router, no attention,
     no embedding, no head. A packed bank is bound over the owned row range,
     never over the whole region.
3. **Binding extends #507's.** It keeps artifact/plan identity, backend,
   lowering identity, layer range, total layers and hidden width, and adds:
   - `experts`: owned expert range per owned layer (half-open, like layers);
   - `representation`: per owned bank operand, the representation the
     worker **bound**: the resolved `RepresentationFacts` label, the codec
     identity, the catalogue `RepresentationIdentity` when one was
     selected, the `WeightFormat` the backend was given, and the operand's
     declared payload hash.
   Like #507, identity names **declared** hashes. Byte verification remains
   `vindex verify`, and this rung claims nothing beyond it.
4. **Coverage is judged before the first token.** For every routed layer,
   the union of worker expert ranges must equal `0..experts` exactly once.
   A selected expert with no owner cannot arise at dispatch time. If it
   does, that is a coordinator bug and a refusal, never a skip.
5. **Failure is fatal.** A missing, malformed, mis-bound or late response
   ends the step with an error. It never becomes a zero contribution, a
   local fallback or a retry that changes the result. As in #507, a failed
   step leaves the logical input history unchanged.
6. **Scope:** CPU production lowering, `LayerFfn::Routed` only, and
   single-stream softmax stacks (the coordinator still passes
   `ensure_supported` in this rung). Out of scope: Metal, `LayerFfn::Hybrid`
   (Gemma 4), shared experts, latent/bottleneck routed ops (Kimi), grid
   discovery, retries, compression on the wire, and any performance claim.

## Workload

**Model:** `~/chris-models/gpt-oss-20b.vindex3`. It already executes on the
V3 CPU production path, has softmax attention with a sliding window, and
has a packed MXFP4 expert bank with gate/up and down biases, so the
per-expert bias path is exercised. Its expected shape is 24 layers,
32 experts and top-4; C0 confirms this from the plan and does not assume it.

**Prompts:** three fixed prompts, greedy, 32 generated tokens each. At least
one must exceed the sliding window (prompt plus generation longer than the
window) so the split crosses the boundary that #507's fixtures cross.

**Topologies** (all loopback, one coordinator):

| ID | Workers | Ownership |
|---|---|---|
| T0 | 0 | ordinary V3 whole-model execution: the authority |
| T1 | 1 | all layers, all experts |
| T2 | 2 | all layers; experts split in half |
| T3 | 3 | uneven: layers split in two; one layer range's experts split again |

T3 exists so that a worker never owns a rectangle that happens to match
layer boundaries.

## C0: reconnaissance before implementation

Run before any execution code is written, and record the results here as a
dated section.

- **C0-a** `ensure_supported` accepts `gpt-oss-20b.vindex3`, and the plan's
  routed layers are `LayerFfn::Routed` with `ExpertSlices::Fused`. If
  either fails, the model is replaced **before** implementation, and the
  replacement is recorded with its reason.
- **C0-b Determinism control.** Two T0 runs in separate processes on the
  same build are bitwise identical in logits for every prompt and position.
  If they are not, bitwise parity is not a meaningful gate, and this freeze
  stops until a new tolerance is frozen from that measured noise.
- **C0-c Representation reachability.** Name a concrete case where
  artifact, plan **and** lowering identity match but the bound
  representation differs, for example an NVFP4 request that binds at source
  precision in one process and not in another. If no such case exists, the
  `representation` field is redundant with lowering identity. Gate R3 is
  then withdrawn **before** implementation, and the field stays only as a
  diagnostic. A gate that cannot go red for its stated reason is not a gate.
- **C0-d** Record the selection and accumulation order for the CPU
  production path at the commit the freeze is taken against, as the
  reference the coordinator reduction must reproduce.

## Gates

Every gate compares against T0, never against legacy execution.

### Parity (P)

Tolerance: **bitwise** on the CPU production lowering, on one machine and
build, conditional on C0-b. A bitwise miss is a FAIL. The tolerance is not
widened afterwards.

- **P1** Per routed layer and position: the coordinator's combined
  routed-expert output equals T0's `routed_ffn` output.
- **P2** Per layer: the layer output (residual after the FFN block) equals
  T0.
- **P3** Final logits at every position equal T0.
- **P4** The greedy trajectory (32 tokens, three prompts) is identical to
  T0 for T1, T2 and T3.

### Refusal (R)

Each refusal is tested by changing **only** the property named, with every
other binding field held equal. The error must name that property. A
refusal raised by an earlier, unrelated check does not count.

- **R1** Incomplete expert coverage on any layer refuses before the first
  token.
- **R2** Overlapping expert ownership refuses before the first token.
- **R3** A different bound representation (the C0-c case) with the same
  artifact and lowering refuses. Withdrawn if C0-c finds no reachable case.
- **R4** A different artifact/plan refuses.
- **R5** A different backend or lowering identity refuses.
- **R6** A layer or expert out of the worker's slice is refused **by the
  worker** as well as by the coordinator.
- **R7** A worker refuses whole-model generation, as #507's layer workers
  do.

### Failure (F)

- **F1** A worker that errors, times out, returns the wrong number of rows,
  the wrong row width or a different binding fails the step. No token is
  emitted, and the logical history equals its value before the step.
- **F2** After F1, restarting the worker and repeating the step reproduces
  T0 bitwise.

### Statelessness and containment (S)

- **S1** The same request sent twice returns bitwise identical responses.
- **S2** Requests from two coordinators running different prompts,
  interleaved on one worker, each return what they return alone.
- **S3** A worker's preparation ledger (the operand plan a slice loads,
  computed before any byte is read) lists only its owned expert rows. It
  lists no router, attention, embedding or head operand, and the byte
  total equals the owned rows' declared size.
- **S4** No legacy execution participates. The new coordinator and worker
  modules import nothing from `larql_inference::ffn` or
  `larql_inference::vindex`, and this is checked mechanically. A recording
  `PlanBackend` shows that every expert output in the parity arm came
  through the worker's V3 backend.

### Descriptive (not gated)

Per routed layer and step, record dispatch start, each worker's completion,
the maximum completion across workers, and reduce time. Loopback numbers
are not a performance result. They exist so the instrument is in place,
with the per-layer critical path (`dispatch + max(worker) + reduce`) as a
field, before a later rung freezes a latency claim.

## Verdict

**PASS** only if P1–P4, R1–R7 (less R3 if withdrawn), F1–F2 and S1–S4 all
pass on T1, T2 and T3. Any failure is reported as a failure, with the arm,
prompt, layer and position of the first divergence.

**What a PASS would establish:** a V3 routed-expert operation can be placed
on stateless workers bound to a declared `(layer, expert)` slice without
changing execution, on this model, lowering and machine.

**What it would not establish:** anything about latency, throughput, WAN
behaviour, Metal, hybrid or latent MoE, K3, or cross-machine numerical
agreement. Cross-machine agreement needs its own determinism control, since
two machines' kernels are not the same build in the sense C0-b tests.

## Order

1. Commit this freeze.
2. Run C0 and commit its results, including any model replacement or R3
   withdrawal.
3. Implement: the two slice kinds, the `routed_ffn` seam, binding, worker
   route and coordinator.
4. Run the witnesses (P, R, F, S) and record raw results before any verdict.
5. Adjudicate against this page, then commit.

## Next rungs (each a new freeze)

- **V3-FFN-SLICE-2: transport.** Real LAN timing. The claim is about the
  per-layer critical path (`Σ_layers dispatch + max(selected worker
  completion) + reduce`), not aggregate throughput, since one slow worker
  per layer governs decode latency.
- **K3 transfer.** The coordinator keeps KDA/MLA, attention residual and
  routing; workers hold only experts, so `ensure_supported`'s coordinator
  restriction has to be revisited. Known obstacles: the latent/bottleneck
  routed op (the router reads the block input, experts run at latent width),
  the shared expert, and the Metal grouped-experts path.

## C0 results (2026-09-24)

Run against the freeze commit `4794c0d5`, release build of `larql` and
`larql-server` from that commit. The machine was shared with a peer session
running a fidelity bank, so wall-clock numbers below are not measurements.
Prompts and token ids: [`bench/v3-ffn-slice-1/c0/prompts.json`](../bench/v3-ffn-slice-1/c0/prompts.json)
(p1 5 tokens, p2 13, p3 151, which is past the 128-token window). The logit
dumps are not committed; their hashes are below.

### C0-a: admissible. PASS

`larql-server gpt-oss-20b.vindex3 --layers …` prepared the full stack and
served this binding. `binding()` returns only after `ensure_supported`
accepts the plan and operands:

```json
{"schema":1,"artifact":"163e5d70feddc7127fd3ac76a5b817a3341b3e19940e4f0343804a4eb9324891",
 "backend":"cpu","lowering":"cpu-production/v1","start":0,"end":24,"layers":24,"hidden":2880}
```

The graph declares one routed FFN: 32 experts, top-4,
`gpt_oss_topk_then_softmax`, `normalised_over_selected`, router bias,
`packed_mxfp4` with `interleaved` gate/up, no shared experts, not hybrid.
Attention is softmax with sinks, and 12 layers use a 128-token window.
The model is kept.

**Incidental defect found in #507 (outside this rung).** `--layers 0-23`
refused with `execution slice 0..25 is outside component`.
`parse_layer_range` already returns an exclusive end (`"0-19"` gives
`(0, 20)`), and `load_v3_model_slice` treats it as inclusive and adds 1
again. The witness above therefore used `--layers 0-22`, which prepares
`0..24`. The #507 test builds `layer_range: Some((layer, layer))` directly
and never goes through the parser, so it pins the wrong convention. This is
reported for a separate fix. It does not change C0-a: the binding records
the range that was actually prepared.

### C0-b: whole-model V3 is bitwise deterministic. PASS

`larql vindex3 exec --backend production --logit-dump`, teacher-forced, two
separate processes per prompt:

| Prompt | Positions | run 1 = run 2 | SHA-256 of `[positions, 201088]` f32 |
|---|---|---|---|
| p1 | 5 | bitwise | `f4c4e82cc4575caf3d88da2048d3d73aab066f36842627a0b88e76021abc67d5` |
| p2 | 13 | bitwise | `f47da0409d2637df63caf726ffdc7fb5ef969d99e35011606194ccb0e012ef87` |
| p3 | 151 | bitwise | `55c6db0e26466cb9175cc57123ca34c995f600951d5f540e4a8ead76fa22b5ee` |

Control beyond the freeze: p2 with `LARQL_CPU_WORKERS=3` and
`VECLIB_MAXIMUM_THREADS=1` has the same hash. The worker count comes from
`parse_workers`, which accepts `3`, so the setting took effect; the thread
count itself was not observed. Per-position time varied from 0.27 s to
5.1 s across runs under different load, with identical bytes. Sanity: p1's
final argmax is " Paris" and every value is finite.

Bitwise parity therefore stands as the P tolerance.

### C0-c: no reachable representation mismatch. R3 WITHDRAWN

For the expert bank on a CPU layer worker, nothing that leaves artifact,
plan and lowering identity equal changes the bound representation:

- The worker is forced to `--v3-backend cpu`
  (`load_v3_model_slice` refuses anything else), so the backend is fixed.
- The production CPU route runs a packed bank through `as_f32()`: the bank
  is widened to f32 at load.
- The CPU format policy's process-wide knobs (`LARQL_CPU_MAX_FORMAT`,
  `LARQL_CPU_Q4_CLASSES` and the arithmetic arm) exclude the bank by
  class: `Q4Classes::admits(RoutedExpertBank)` is `false`, because "the
  bank is widened to f32 on the way in".
- `LARQL_KQUANT_EXEC` applies to stored K-quant packs, and this bank is
  MXFP4. Weight staging changes residency, not values.

R3 is withdrawn under the rule in C0-c. The `representation` field stays in
the binding as a diagnostic.

**Observation for a later freeze (not tested here).** `LoweringIdentity`
excludes provider configuration by design ("what it changes is already
pinned in the realization form"), but #507's binding carries the lowering
identity and not the realization. The same CPU knobs do change
**non-expert** operands, such as a streaming projection manufactured as
Q8 or kept as bf16. So two #507 layer workers, or this rung's coordinator,
could bind identically and compute differently. It becomes reachable for R3
as soon as a bank is not widened, for example a Separate bf16 bank under a
future class policy. That is a new question, not an expansion of this rung.

### C0-d: the order the coordinator must reproduce

At `4794c0d5`:

- `backend.routed_ffn` has one caller (`opplan/exec/experts.rs`), shared by
  the batch executor and the decode session.
- CPU production `routed_ffn` (`opplan/exec/production.rs`): route with
  `select_experts`, then for each `(expert, weight)` **in the returned
  order**, compute the expert output with `add_expert_bias` applied to both
  the fused gate/up and the down projections, then `*acc += weight * v`
  over a zero-initialised `hidden` vector. This is the `ExpertSlices::Fused`
  arm used for this bank.
- `router::select` (`larql-compute/src/ffn/expert_weight/router.rs`)
  returns at most `k` experts ordered by **descending logit**, with ties
  going to the lower expert index (a stable sort, matching `torch.topk`).
  Weights are computed after ranking. `branch_scale` multiplies the
  weights when it is not 1.

The coordinator calls `select_experts` itself rather than reimplementing
it, and accumulates with the same expression in the same order.

### C0 verdict

C0-a PASS (model kept). C0-b PASS (bitwise tolerance stands). C0-c: R3
withdrawn. C0-d recorded. The freeze proceeds to implementation with gates
P1–P4, R1–R2 and R4–R7, F1–F2 and S1–S4.

## Closure (2026-09-26)

This freeze was not executed as written. The capability it asked about
landed on main by a different route: PR #547 added V3 CPU dense FFN workers
and packed-MXFP4 routed-expert workers that keep normalization, routing,
selection, attention, KV and the weighted reduction on the coordinator
([routed provider guide](ffn/v3-routed-experts.md)). That work carries its
own gates and real-model checks on GPT-OSS 20B and does not cite this page.

So gates P, R, F and S above were never adjudicated against this freeze,
and nothing here is a verdict on #547. What stands is the C0 record: the
model is admissible (C0-a), whole-model V3 is bitwise deterministic, so a
bitwise parity tolerance is achievable (C0-b), R3 is withdrawn (C0-c), and
C0-d records the reduction order a coordinator must reproduce. The next
rungs listed above (transport timing, K3 transfer) remain open and would
each start from a new freeze against the #547 providers.
