# V3-OBS-1 — can the canonical decode traversal expose every carrier write without changing execution?

Pre-registered 2026-09-19, before any implementation or measurement. Properties and
forecasts below are FROZEN. Nothing in this document claims a result.

Programme: OBSERVE (LARQL as observable, later perturbable, execution). This is rung 1
of the executor-side substrate. The Observatory UI is being built in a separate
session against the contract this rung freezes; it is out of scope here.

Relationship to the peer drafts written the same day: `observatory.md` owns the
product; `vindex3-observation-contract.md` owns event meaning, envelope identity,
capture profiles, loss accounting, transport and receipts. This document sits **under**
that contract and freezes only the executor tap the contract's Standard profile needs.
It answers the first two of the contract's open engineering decisions (§14): the first
parity and replay witness is Granite 4.2 3B on the CPU decode path, and the canonical tap
that exposes carrier and applied writes is `leave_site` in the decode traversal. The
contract's per-event `run_id`, `sequence` and `timestamp_ns` are assigned by the runner
adapter from the session identity (C2 below); the executor itself emits neither.

---

## The question

The VINDEX3 decode traversal already writes to the residual carrier at two sites per
transformer layer and one site per mixer-only layer. The observer contract
(`crates/larql-vindex/src/format/vindex3/opplan/exec/observe.rs`) already fires
structural events at those boundaries and already exposes the values that exist only
on the two non-trivial carrier topologies. What it does **not** expose is the write
itself on the ordinary single-stream carrier: the delta that was added, and the
carrier after the add.

V3-OBS-1 adds that observation. The question is whether it can be added as **one more
tap on the one executor** — the rule the observe module already states — with the
observed and unobserved paths staying bit-identical, on every carrier form the
executor traverses, and at a cost that is measured rather than asserted.

---

## What already exists (read before implementing)

| Fact | Where |
|---|---|
| Structural events `Embedded / AttentionDone / FfnDone / Logits{vocab}`; observed == unobserved parity gate; rule "finer taps = more events on the one executor, never a second traversal" | `opplan/exec/observe.rs` |
| Borrowed tensor taps already exist: `operand_input` at the normalised attention input, the normalised FFN input and the FFN output; `hyper_connection_site` (split, reduced, branch_output, bundle_out); `attention_residual_site` / `_boundary` (probs, mixed_vector, prefix_before/after) | `opplan/exec/observe.rs` |
| Every carrier write on every topology passes through one function. For `Carrier::Single` the write is `backend.residual_add(h, &delta)`; `delta` and the post-add `h` are both in scope there | `opplan/exec/decode.rs` `leave_site` |
| A Gemma 4 `layer_scalar` (`OperandRole::LayerScalar`) multiplies the **whole carrier** after the FFN-site add, on single-stream carriers only. Preparation refuses it on bundle/history carriers | `opplan/exec/decode.rs` after the FFN `leave_site`; `graph/roles.rs` |
| The batch interpreter records `post_attention`, `ffn_input` (normed) and `post_layer` per layer per position; `post_layer` is captured **after** the layer scale | `opplan/exec/mod.rs` `LayerTrace`, `ExecutionTrace` |
| `FfnDone` is the layer boundary and fires on **every** layer, including mixer-only layers; the FFN carrier write is conditional on the layer carrying an FFN program | `opplan/exec/decode.rs` |
| The existing parity test asserts bit-identical logits between `step` and `step_observed`, and that the event stream mirrors the plan structure, on the golden synthetic plan | `opplan/exec/tests/observe.rs` |
| The Metal whole-stack path (`HybridStack::forward`) submits one command buffer per device run and fires **no** observer | `opplan/exec/stack_metal.rs` |
| `Mutation` is the negative-control vocabulary threaded as `Mutation::None` in production; `step_mutated` is `#[cfg(test)]` | `opplan/exec/controls.rs` |

V3-OBS-1 therefore does not invent carrier state. The batch path already represents it.
The decode path gets the same state in event form, and the two become an independent
parity witness of each other.

---

## Contract properties (frozen as properties, not as names)

**C1 — owned event, borrowed record.** `StepEvent` derives `Clone` and `PartialEq` and
the recording observer and parity tests depend on that. A carrier write therefore has an
owned structural event **and** a separate borrowed tensor record delivered through its
own observer method, following the existing `operand_input` / site-record pattern. A
borrowed slice never lives inside the event enum.

**C2 — identity belongs to the session.** The executor allocates no run id. An
observer is constructed with an execution identity composed from authorities that
already exist: the container, the backend's `LoweringIdentity`, the session's pinned
realizations, any overlay or patch authority, and the decode configuration. Events
carry only coordinates: position, layer, site, and a per-step sequence number.

**C3 — topology is not collapsed.** The carrier has three forms (`Single`, `Bundle`,
`History`). The write record's shape may differ by form. For `Bundle` and `History` the
values are already delivered by the existing site records; V3-OBS-1 adds the uniform
structural write event across all three and the value record for `Single`. The executor
never reduces a bundle or a history to one vector for the observer's convenience; a
consumer may.

**C4 — declared writes, not assumed writes.** A layer emits exactly the carrier writes
its bound program performs. A mixer-only layer emits one. `FfnDone` continues to fire
as the layer boundary on every layer; it is not a write.

**C6 — the entering carrier is observable.** P2 chains writes: the `before` of the
FFN-site write is the `after` of the attention-site write, and the `before` of layer
L+1's attention site is layer L's (scaled) FFN `after`. The chain needs a first link,
so the `[hidden]` vector entering layer 0 (the embedding after its scale and norm, the
batch path's `embedded` plane) is delivered by its own borrowed tap. Observation only;
covered by P1.

**C5 — the layer scale is part of the write's fate.** On a component that declares a
`LayerScalar`, the value the next layer reads is `scale · (before + delta)`, not
`before + delta`. The record for the FFN-site write on such a component carries the
post-add, pre-scale `after` **and** the scale, so a consumer can reconstruct the layer
output exactly and a witness can compare it to the batch `post_layer`, which is
post-scale.

---

## Frozen acceptance properties

**P1 — observational parity.** Same subject, prompt, binding, backend and decode
configuration: `step` with `NoopObserver` and `step_observed` with the carrier observer
produce bit-identical logits at every position. Extends the existing observe test to the
new tap and to real subjects.

**P2 — exact single-stream reconstruction.** For every `Single` write:
`before + delta == after` bit-for-bit under the backend's `residual_add`, where
`before` is the carrier the witness held entering the site. On a `LayerScalar`
component the layer output additionally equals `scale_row(after, scale)`. The witness
must go RED when either vector is perturbed by one ulp in one element (negative
control), or it proves nothing.

**P3 — program-faithful emission.** Per executed layer, the observed write count equals
the count derived from the bound plan before the run: two on a layer with an FFN
program, one on a mixer-only layer, never a fabricated FFN write, never a duplicate,
never a normalised "attention then FFN" pattern imposed on an architecture that does not
have it. The structural `FfnDone` count is separately forecast and must match the
executor's existing behaviour (every layer).

**P4 — every carrier form witnessed, on the plan family that exists for it.** See the
witness map. A form witnessed only on a synthetic plan is recorded as such; the real
subject upgrade is a named follow-on, not an implied pass.

**P5 — capture cost measured.** Token wall with the stats observer minus token wall with
`NoopObserver`, per the protocol below. Reported, not thresholded. "Cheap" is not a
word this rung may use before the number exists.

---

## Witness map (corrected against the containers on disk, 2026-09-19)

| Carrier form | Family fact | Subject | Status |
|---|---|---|---|
| Single, attention + gated FFN, 40 layers | Granite 4.2 | `~/chris-models/granite-4.2-3b.vindex3` | **primary witness** (Chris's named test bed, 2026-09-19); P1, P2, P3, P5, batch/decode |
| Single, softmax attention + gated FFN | OLMo2 | `~/chris-models/olmo2-1b.vindex3` | secondary single-stream witness; P1, P2, P3 |
| Single, mixer-only layers (no FFN program at all) | Mamba2-attn hybrid | `~/chris-models/mamba2attn-2.7b.vindex3` | P3 mixer-only witness |
| Single, recurrent (KDA) and MLA attention sites, MoE FFN sites | Kimi-Linear | `~/chris-models/Kimi-Linear-48B-A3B-Instruct.s6.vindex3` | P3 on non-softmax sites; **must run through `DecodeSession`, not the `lowered` whole-stack Metal path, which fires no observer** |
| Bundle (mHC, 4-wide residual stream) | GLM-5.3-Flash (`mhc`, `hc_*`) | HF cache only; no V3 container encoded | interim witness = synthetic HC plan already used by `exec/tests/wave19_hc_decode.rs`; real-subject upgrade named as follow-on |
| History (attention-residual prefix, `attn_res_block_size`) | Kimi K3 (2.8T) | not local | witness = synthetic attention-residual plan already used by `exec/tests/attn_res_2a_decode.rs`; no local real subject exists |
| Single, Gemma 3 block (pre+post norms, sliding/global attention, softcapped head) | Gemma 3 | **Gemma-3-4B-IT is Chris's working model and the Gemma test bed.** `~/chris-models/gemma3-4b-it.vindex3` and `gemma3-12b-it.vindex3` were encoded on 2026-09-19 as text-only components with the vision tower carried by name | Gemma witness, run AFTER Granite: the same real-container test, which also answers whether this build's executor closes and runs the `gemma3_text` plan at all. Reference-logit parity against HF is #436's gate, not this rung's |
| Single with `LayerScalar` (post-add whole-carrier scale) | Gemma 4 | the synthetic Gemma 4 miniature already in `exec/tests/gemma4.rs` (V3-F0 witness 3) | C5 and the P2 scale form witnessed on the miniature; a real Gemma 4 container is not a test bed Chris asked for and is not scheduled |

Kimi-Linear is **not** a bundle carrier. Its system graph declares KDA, MLA and a
256-expert MoE on a single residual stream and declares neither `mhc` nor
`attn_res_block_size`. An earlier note in this programme naming Kimi as the bundle
witness was wrong and is superseded by this table.

---

## Forecasts (written before implementation)

**F1 — semantics.** Bit-identical logits, noop vs observed, on every witness.

**F2 — reconstruction.** `before + delta == after` on every `Single` write in every
witness; the one-ulp negative control goes RED.

**F3 — write counts per token, derived from the bound plans on disk today.**

| Subject | Layers | Per-layer program | Carrier writes / token | `FfnDone` / token |
|---|---|---|---|---|
| OLMo2-1B | 16 | 16 × softmax attention + gated FFN | **32** (16 attention-site, 16 FFN-site) | 16 |
| mamba2attn-2.7b | 64 | 58 × mamba2 + 6 × conv_qkv attention, no FFN program on any layer | **64** (64 mixer/attention-site, 0 FFN-site) | 64 |
| Kimi-Linear-48B s6 | 27 | 20 × KDA + 7 × MLA, MoE FFN on every layer | **54** (27 attention-site, 27 FFN-site) | 27 |
| granite-4.2-3b | 40 | 40 × attention + gated FFN; declared `residual_scale` 1.0 (a **delta** scale, applied inside the delta before the add — P2 unaffected) | **80** (40 attention-site, 40 FFN-site) | 40 |
| gemma3-4b-it (encoded 2026-09-19, text component; vision carried by name) | 34 | 34 × softmax attention (29 sliding + 5 full) + gated FFN; pre+post norms; tied softcapped head; no layer scalar | **68** (34 + 34) | 34 |
| gemma3-12b-it (encoded 2026-09-19, same layout) | 48 | 48 × softmax attention (40 sliding + 8 full) + gated FFN | **96** (48 + 48) | 48 |
| qwen3-4b | 36 | 36 × softmax attention + gated FFN | **72** (36 + 36) | 36 |
| qwen3-0.6b | 28 | 28 × softmax attention + gated FFN | **56** (28 + 28) | 28 |

Plus one `Embedded` and one `Logits` event per step on each. These counts were read
from each container's `system_graph` on 2026-09-19 and must be re-derived from the
**bound plan** by the implementation before results are inspected; a disagreement
between the two derivations is a finding, not a typo. Qwen3-4B and Qwen3-0.6B are
listed because the Observatory session may pick either as a live subject; each is a
single-stream witness of the same class as Granite and OLMo2.

**F4 — carrier coverage.** All three forms observe without a second traversal. A form
that cannot is a failed property; it is not permission to reduce that form to
single-stream.

**F5 — cost.** No number is forecast. The frozen claim is only: the stats observer is
materially cheaper than full tensor materialisation and than a per-layer logit lens.
That claim is what P5 tests.

---

## P5 protocol

- Subject: Granite 4.2 3B on the CPU decode path. Release build only — a debug build
  reprices CPU stages by roughly thirty times and would measure the wrong thing.
- Control: `NoopObserver`. Treatment: a stats observer consuming the carrier tap and
  computing exactly: carrier norm, delta norm, a fixed low-dimensional projection
  (basis carried with provider, identity, hash, dimensionality), and selected-token
  logits from **pre-extracted output-head rows** for a token set declared before the
  run.
- Excluded from the treatment: full-vocabulary top-k, any logit-lens pass, per-head
  capture, tensor persistence, Metal host readback.
- Same process, same session, same prompt and decode length, interleaved control and
  treatment arms, steady-state protocol from `docs/` (warm-up discarded; enough
  repetitions to characterise variance; thermal state noted).
- Report: both walls, the difference, the spread, and the build and machine identity.
  A negative or negligible difference does not fail P5. An unmeasured one does.

Full top-k at every layer is a logit lens — one full output-head projection per layer
per token — and is not an observation statistic. It is computed on demand for a
selected site, or in replay.

---

## Batch/decode witness

For Granite 4.2 3B, and OLMo2-1B as a second subject, run the same tokens through the batch interpreter and through
`DecodeSession` with the carrier observer. Compare, per layer and position, the decode
attention-site `after` to `LayerTrace.post_attention`, and the decode layer output
(post-scale where a scale exists) to `LayerTrace.post_layer`. Report `bit_identical`,
`max_abs` and `rel_rms` in the form the existing `attention_kv_parity` test reports.
Forecast on the reference backend: bit-identical, because the decode loop's ordering
mirrors the batch traversal by construction. This witness is additional to P1, not a
substitute for it.

---

## Stream and record (constraints this rung must not violate)

Two consumers will sit on this tap in later rungs and their policies differ: a live
stream is lossy and non-blocking and must never stall inference; a recorder is lossless
and must say so explicitly when it cannot be. V3-OBS-1 does not build either, but the
record it delivers must be sufficient for both, and nothing in it may assume a
particular consumer. Timing, when it arrives, is session-scoped and monotonic; the
process-global timing ledger is not a run identity.

---

## Separations this rung keeps

- **Intervention is not observation.** Future `Zero / Replace / PatchFrom / Scale`
  operations get their own type on the same execution seam and their own law
  (`intervene(NoOp, run) == run`). They are never added to the `Mutation` control
  enum, whose purpose is to make witnesses fail.
- **Metal instrumented capture is a separate rung.** Per-layer observation on the
  whole-stack device path requires readback per layer; value parity can hold there,
  timing parity cannot, and a receipt will have to say so (`timing_intrusive`,
  `device_readbacks`). V3-OBS-1 makes no claim about that path.
- **Per-head observation is a separate rung.** The observe module states that the
  attention core output never surfaces before `o_proj`. Head contributions are not to
  be inferred algebraically and presented as observations.

---

## Out of scope

Observatory UI, transport, run history, Compare, interventions, per-head observation,
source-token attribution, full-vocabulary logit lens, Gemma Scope and SAE providers,
Python V3 bindings, experiment runner, Metal per-layer instrumentation, closing #436.

---

## Housekeeping owed before V3 trace terminology becomes public

Label the legacy residual trace as belonging to the pre-VINDEX3 walk path in
`docs/residual-trace.md`, the TraceStore section of `docs/larql-python.md`, and
`crates/larql-inference/docs/trace-format.md`. The two trace concepts may coexist for
now; their scopes may not be implicit.

---

## Amendments (recorded before any implementation or measurement)

- **2026-09-19, subjects.** Chris named Gemma and Granite 3B-class models as the test
  beds, and then Gemma 3 4B specifically as the model he uses. Granite 4.2 3B replaces
  OLMo2-1B as the primary witness; OLMo2 stays as a second single-stream subject.
  Gemma-3-4B-IT is the Gemma test bed and is blocked on #436; closing #436 is the next
  executor rung after this one. The Gemma 4 miniature fixture already exercises
  `layer_scalar` on every CPU backend with a decode session, so the layer-scale branch
  is witnessed on that plan family without a real Gemma 4 container.
- **2026-09-19, entering-carrier tap (C6).** Added so P2's chain has its first link.
- **2026-09-19, event placement.** The structural write event fires inside the write
  function for all three carrier forms, after the form's value record, so a
  structure-only consumer counts writes and a value consumer sees the values first.
- **2026-09-19, P2's negative control, corrected by the witness itself.** P2 as first
  written demanded RED on a one-ulp perturbation of *either* vector. One ulp of `after`
  is always caught (the comparison is on bits). One ulp of `delta` is **not** always
  caught: when the delta is small against the carrier, the add's rounding absorbs it and
  `before + delta'` lands on the same f32 as `before + delta`. That is a fact about IEEE
  addition, not a weakness of the tap, and the control was wrong to ask for it. The
  delta control now perturbs by four ulps at the magnitude of the sum, the smallest
  change the witness could be asked to see, and is caught at exactly the perturbed
  (layer, site) at every position.
- **2026-09-19, the C5 control derives its expectation.** The Gemma 4 miniature's last
  layer scalar is exactly 1.0, so a chain that forgets the scale is only wrong after
  layers whose scalar is not 1.0. The control's expected failure set is derived from the
  scales the records carried, never assumed to be "every layer".

## Results (recorded 2026-09-19, after the forecasts above were frozen)

Implementation: `opplan/exec/observe.rs` (event, form, record, two taps),
`opplan/exec/decode.rs` (`leave_site` fires the record then the event on all three carrier
forms; `entering_carrier` after `Embedded`), `opplan/exec/observe_stats.rs` (the stats
observer and basis identity), witnesses in `opplan/exec/tests/carrier_write.rs`,
`carrier_write_real.rs`, `observe_stats.rs`. Gates: fmt, clippy on larql-vindex /
larql-inference / larql-lql, workspace check, full larql-vindex suite (4761 passed),
larql-inference vindex3 tests (37 passed), all green.

**Synthetic plan families (unit witnesses, 20 tests, all PASS):** P1 on the reference
and production CPU backends; P2 with both negative controls caught at exactly the
perturbed (layer, site) at every position; P3 on the plain stack, the mixer-only Mamba2
stack (one write per layer, `FfnDone` still on every layer, zero FFN-site writes), the
hyper-connected stack (every write a bundle, one site record per write) and the
attention-residual stack (every write a history, layer 0's attention site writes with no
record); C5 on the Gemma 4 miniature with the forgetful-chain control; batch/decode
bit-identical on the plain stack on both backends and through the layer scale on the
Gemma 4 miniature.

**Granite 4.2 3B, real container, production CPU backend, release build.** The
unsuffixed container on disk is a schema-5 encode and this build refuses it (schema 6);
the witness ran on `~/chris-models/granite-4.2-3b.s6.vindex3`. Prompt: token ids 1..=8.

| Property | Result |
|---|---|
| P1 | PASS — bit-identical logits at all 8 positions |
| P2 | PASS — 640 of 640 writes reconstruct bit-for-bit through the chain |
| P3 | PASS — 640 writes observed = 80 per token × 8, the frozen forecast; `FfnDone` 40 per token |
| Batch/decode | PASS — every carrier state at every layer and position bit-identical to the batch planes |
| P5 (15 interleaved pairs, one warm-up pair discarded, basis `seed-24301-3x2560`, 3 dims, no probe) | see below |

P5 per token, stats observer minus `NoopObserver`:

| Estimator | noop | stats | difference | ratio |
|---|---|---|---|---|
| median | 62.67 ms | 62.82 ms | +0.15 ms | 1.0024 |
| min | 58.64 ms | 59.91 ms | +1.27 ms | 1.0217 |
| mean | 99.24 ms | 121.90 ms | +22.66 ms | 1.2283 |

The three estimators disagree because the machine was not in one state during the run:
the warm-up pair and the first pairs ran at ~170 ms per token, the later pairs at ~60 ms,
so both means carry a state shift the interleaving cannot remove. The median and the
floor agree that the observer's own arithmetic (80 writes × two norms and three dot
products over 2560 lanes) costs on the order of a tenth of a millisecond to a
millisecond per token on this subject. The mean is reported because P5 says report, not
because it measures the observer. A quiet-window re-run under the steady-state bench
protocol is owed before any cost figure is quoted outside this document.

**mamba2attn-2.7b, real container, production CPU backend, release build (the real
mixer-only witness).** 64 layers, none with an FFN program. P1 PASS (bit-identical logits
at 8 positions); P2 PASS (512 of 512 writes reconstruct bit-for-bit); P3 PASS (512 writes
= 64 per token × 8, the frozen forecast, with zero FFN-site writes and `FfnDone` still on
every layer); batch/decode PASS. The program's write topology, not a transformer
template, is what the tap emits.

**OLMo2-1B, real container, same backend and build.** 16 layers with an FFN program. P1
PASS; P2 PASS (256 of 256); P3 PASS (256 = 32 per token × 8, the frozen forecast);
batch/decode PASS.

Real-subject witness count for the single-stream form: three (Granite 4.2 3B, OLMo2-1B,
mamba2attn-2.7b), all bit-identical on every property. The bundle and history forms
remain witnessed on their synthetic plan families only, as the witness map says.

**Granite 4.2 3B, CROSS-BACKEND (production CPU vs reference CPU), same container,
same prompt, operands prepared once per backend.** The driver now runs each requested
backend (`LARQL_V3_BACKENDS`, default `production,reference`) over one prepared image,
passes P1–P3 and batch/decode on each, then compares the two chains.

| Quantity | Result |
|---|---|
| Structural event stream (1296 events), write count (640), sites, positions, layer scales | **identical** — asserted |
| Entering carrier (embedding, all 8 positions) | bit-identical |
| Carrier writes bit-identical | 0 of 640; first difference at layer 0, position 0 |
| Relative RMS by depth | L0 4.6e-3 · L5 8.3e-3 · L10 1.2e-2 · L20 1.4e-2 · L30 1.3e-2 · L39 2.4e-2 |
| Final logits | max abs 0.69, relative RMS 3.4e-2 |

The values differ because the two backends executed **different declared realizations
of the same BF16 bytes**, read off the pinned `RealizationRecord.selection` of each
arm, not inferred from the numbers:

| Arm | Pinned realizations (282 operands) | Activation arm (process-global) |
|---|---|---|
| production | 121 × `Requantise(FusedQ8)`, 80 × `Direct(FusedBf16)`, 80 × `Decode(BlasF32)`, 1 × `DecodedGather` | `FloatActivation` |
| reference | 281 × `Decode(ScalarF32)`, 1 × `DecodedGather` | `FloatActivation` |

The production CPU size policy (`cpu/physical.rs::choose_for`) requantises every BF16
projection above its compact threshold to Q8 under the default f32-activation arm; the
reference widens the same bytes to f32 exactly. A layer-0 relative RMS of 4.6e-3 is the
Q8 requantisation class, and it compounds with depth as expected. This is a declared
realization difference, visible in the executor's own records, not accumulation-order
drift and not a defect in either backend or in the tap.

**The same comparison with the Q8 manufacture switched off**
(`LARQL_CPU_MAX_FORMAT=bf16`, the only cap the policy honours; a first attempt with
`f32` was ignored by the policy and the run's own realization lines showed it had not
taken effect). Production then pinned 201 × `Direct(FusedBf16)`, 80 × `Decode(BlasF32)`,
1 × `DecodedGather`; reference unchanged. P1–P3 and batch/decode PASS on both arms.

| Quantity | Result |
|---|---|
| Structure | identical, as before |
| Carrier writes bit-identical | 0 of 640 (BLAS and fused kernels order their sums differently from the scalar reference) |
| Relative RMS by depth | L0 1.6e-6 · L5 3.2e-6 · L10 4.2e-6 · L20 5.0e-6 · L30 4.5e-6 · L39 7.9e-6 |
| Worst layer | max abs 4.2e-2, relative RMS 7.9e-6 |
| Final logits | max abs 1.1e-4, relative RMS 6.5e-6 |

Two facts fall out, both from the executor's records rather than from the numbers alone.
The fused bf16 kernel computes against f32 activations: its disagreement with the scalar
f32 reference is accumulation order, three orders of magnitude below the 1e-3 tripwire.
And the entire default-policy disagreement (4.6e-3 at layer 0, 2.4e-2 at layer 39) was
the Q8 requantisation of the 121 large projections, a realization the production
backend chose and pinned, not a property of the tap or of either backend's arithmetic.

Two consequences for the ledger and for any consumer. First, the single-stream form is
CROSS-BACKEND WITNESSED for **structure**, and for **values at f32 accumulation
precision** (relative RMS at or below 8e-6 over 40 layers) when both arms execute exact
forms of the same bytes; under the default production policy the values differ by the
declared Q8 requantisation and are witnessed only at that precision. Bit-identical
values across backends are not claimed and are not expected: the reference is scalar,
production is BLAS and fused kernels. Second, a projected coordinate is only comparable across runs
when the run record carries the pinned realizations and the process arithmetic arm
alongside the basis identity: the arm is process-global (`arithmetic_arm()` is a
`OnceLock`) and appears in no record, which is the execution-scoped-accounting gap this
programme already knows about. The driver prints both so a run's numbers are
attributable.

**OLMo2-1B, CROSS-BACKEND: NOT RUN — the reference backend refuses the model.** Its
container is stored as F32 (production pins 113 × `Direct(BlasF32)`, reference would
pin 113 × `Decode(ScalarF32)`), so it would have isolated pure accumulation-order drift
between BLAS and scalar f32 with no requantisation in the way. The reference arm
refuses at the first decode step with `full-projection QK norm has no judged reference
execution yet`: OLMo2's QK norm is over the full projection, and the reference backend
declares it has not been judged for that operation. A declared refusal, surfaced by the
driver, not a defect in the tap. The production arm passed P1–P3 and batch/decode again
on the way.

**Gemma 3 4B IT, real container, both CPU backends (run after PR #483 merged, on
`origin/main` + this rung, in an isolated worktree).** 34 layers, every one with an FFN
program. Prompt: token ids 1..=8.

| Property | production | reference |
|---|---|---|
| P1 | PASS, bit-identical logits at 8 positions | PASS |
| P2 | 544 of 544 exact | 544 of 544 exact |
| P3 | 544 = 68 per token × 8, **the frozen forecast** | same |
| Batch/decode | PASS, bit-identical | PASS, bit-identical |

The reference backend executes Gemma 3 (it refused OLMo2), so the cross-backend witness
ran too. Default production policy pinned 103 × `Requantise(FusedQ8)`, 68 ×
`Direct(FusedBf16)`, 68 × `Decode(BlasF32)`, 1 gather; reference 239 × `Decode(ScalarF32)`.

| Cross-backend | default policy (Q8 manufactured) | Q8 capped (`LARQL_CPU_MAX_FORMAT=bf16`) |
|---|---|---|
| Structure | identical: 1104 events, 544 writes | identical |
| Relative RMS by depth | L0 5.2e-3 · L5 1.0e-2 · L10 1.3e-1 · L15 1.2e-1 · L25 6.1e-2 · L33 1.3e-1 | L0 1.6e-6 · L5 1.7e-6 · L10 3.6e-5 · L15 4.6e-5 · L25 1.9e-5 · L33 3.7e-5 |
| Worst layer | max abs 5.4e3, rel RMS 1.5e-1 | max abs 1.6, rel RMS 7.3e-5 |
| Final logits | max abs 2.58, rel RMS 8.3e-2 | max abs 1.4e-3, rel RMS 6.0e-5 |

Read as on Granite: the default-policy gap is the declared Q8 requantisation, and it is
about five times larger here at the carrier level (worst 1.5e-1 against Granite's
2.4e-2; logits 8.3e-2 against 3.4e-2), which is consistent with Gemma 3's known
outlier-channel magnitudes (carrier max abs in the thousands) but is recorded as an
observation, not as a quality claim: no reference logits against HF were compared here.
With realizations aligned the residual is 7.3e-5, an order of magnitude above Granite's
8e-6 and still well below the 1e-3 tripwire; whether Gemma's magnitude regime accounts
for that extra order is not established by this run.

The contract did not change to admit Gemma. This is evidence promotion only.

**Kimi-Linear-48B-A3B-Instruct, real container `.s7` (re-encoded 2026-09-20 so the
surface carries `kda_gate_form: softplus`), production CPU backend, held at bf16
(`LARQL_CPU_MAX_FORMAT=bf16`) so the ontology question is not confounded by the Q8
policy.** 27 layers: 20 KDA and 7 MLA attention sites, a 256-expert MoE FFN on every
layer, a single-stream carrier. P1 PASS (bit-identical logits at 8 positions); P2 PASS
(432 of 432 writes reconstruct bit-for-bit); P3 PASS (432 = 54 per token × 8, the frozen
forecast, `FfnDone` 27 per token); batch/decode PASS (bit-identical carrier states at
every layer and position). Pinned forms: 105 × `Direct(FusedBf16)`, 85 ×
`Decode(BlasF32)`, 19968 × `MappedStored { Bf16, Demand }` (the expert bank), 1 gather.
Residency, recorded as evidence and not as acceptance: 87.75 GiB mapped, 0 resident and
3.31 GiB allocated after preparation (8.5 s); 10.56 GiB resident after the witnesses,
which is the experts the routed tokens touched; peak RSS 16.9 GB on a 128 GiB machine.
The single P5 pair (noop 407 ms per token, stats +8.6 ms) is one trial and not a cost
claim. **The observation contract needed no change to describe a recurrent, latent-
attention, routed-expert stack.** The Q8 default-policy arm was not run.

**Kimi container identities (SHA-256 of `index.json` / `system_graph.json`), recorded
before `.lift2` was deleted on 2026-09-20 at Chris's direction; `.s6` is kept as the
latent-norm refusal fixture.**

| Container | index.json | system_graph.json | `kda_gate_form` | `mla.kv_a_norm_eps` | Fate |
|---|---|---|---|---|---|
| `.lift2` | `ddd9cad3e4603d12…` | `1b4c0fc41da8e83d…` | absent | 1e-06 | refused at first step (gate form absent); DELETED |
| `.s6` | `ddd9cad3e4603d12…` | `a33b81a92da58237…` | absent | absent | refused at preparation (no latent-norm epsilon); KEPT as fixture |
| `.s7` | `ddd9cad3e4603d12…` | `1fe02abad2f3bcdc…` | {"form": "softplus"} | 1e-06 | witnessed; the container of record |

**Gemma 3 12B IT, real container, production CPU backend.** 48 layers, every one with an
FFN program; production pinned 241 × `Requantise(FusedQ8)`, 96 × `Direct(FusedBf16)`,
1 gather (no f32 BLAS form at this width). P1 PASS (bit-identical logits at 8 positions);
P2 PASS (768 of 768 exact); P3 PASS (768 = 96 per token × 8, the frozen forecast);
batch/decode PASS. The reference arm was not run on the 12B. This is the model behind
the sealed sg_invariants witness, so its tap is now real-subject witnessed ahead of that
programme's rung 0.

## Run provenance (added after the cross-backend finding)

The finding above made a record necessary that no operand carried: which forms the
image pinned and which arithmetic arm the process resolved. `opplan/exec/provenance.rs`
now provides it from the executor's own facts:

- `ExecutionProvenance::of(&prepared)` — the lowering provider(s), every pinned
  realization class (representation, codec, physical form, operand count, sorted), the
  process arithmetic arm and the K-quant execution mode, with a SHA-256 `fingerprint()`
  over the canonical serialisation. Two runs whose fingerprints match executed the same
  **recorded** realizations under the same recorded arm. That is what the record
  establishes; it is not a proof that no unrecorded nondeterminism exists, so a residual
  difference between two matching runs is measured, never assumed to be summation order.
- `RunProvenance::new(execution, Some(&stats_observer))` — the execution half plus the
  basis identity, the probe's token set and the norm and probe method names, serialisable
  as JSON for a runner to embed verbatim in its envelope or receipt.

Everything in it is an observed fact of the image or the process, never the environment
value that asked. The real-container driver now prints each arm's provenance and decides
the value tripwire by fingerprint equality. Re-exported from `larql_inference::vindex3`.
The runner adapter (the peer contract's envelope) must persist it alongside `run_id`,
`sequence` and `timestamp_ns`; that wiring is V3-STREAM-1's, not this rung's.

## Evidence states (read this before quoting anything above)

Four states, in increasing strength. A claim is quoted at its state and no higher.

- **FORECAST** — a count or property derived from a system graph or a plan; nothing ran.
- **STRUCTURALLY WITNESSED** — held on a synthetic plan family in the unit tests.
- **REAL-SUBJECT WITNESSED** — held on a real container, one backend, this build.
- **CROSS-BACKEND WITNESSED** — held on a real subject on more than one execution backend.

| Claim | State as of 2026-09-19 |
|---|---|
| Single-stream carrier writes: P1, P2, P3, batch/decode | REAL-SUBJECT WITNESSED on Granite 4.2 3B, OLMo2-1B and mamba2attn-2.7b (production CPU backend), and on Granite on the reference CPU backend too. |
| Single-stream observation is backend-invariant | CROSS-BACKEND WITNESSED **for structure** on real Granite (production vs reference: identical events, writes, sites, positions). For **values**: witnessed at f32 accumulation precision (rel RMS ≤ 8e-6 over 40 layers, logits 6.5e-6) with production capped at bf16 so both arms execute exact forms; under the default production policy the difference is the declared Q8 requantisation (rel RMS 4.6e-3 at L0 → 2.4e-2 at L39) and is witnessed only at that precision. Bit-identical values NOT CLAIMED (scalar vs BLAS/fused summation order). |
| Mixer-only layers write once and still close with `FfnDone` | REAL-SUBJECT WITNESSED (mamba2attn-2.7b, 64 layers, zero FFN-site writes) |
| Bundle carrier writes, one site record per write | STRUCTURALLY WITNESSED (synthetic hyper-connection plan) |
| History carrier writes, layer 0 attention writes without a record | STRUCTURALLY WITNESSED (synthetic attention-residual plan) |
| Layer scale rides on the FFN write; chain must apply it; batch `post_layer` is post-scale | STRUCTURALLY WITNESSED (Gemma 4 miniature) |
| Kimi-Linear-48B: 54 writes per token, single stream, KDA/MLA/MoE sites; P1–P3; batch/decode | REAL-SUBJECT WITNESSED on the production CPU backend (bf16 cap) on the `.s7` container, 432 = 54 × 8. The `.s6` and `.lift2` refusals stand as container facts (no MLA latent-norm epsilon; surface predates the softplus judgement). Reference arm and Q8 arm NOT RUN. |
| Gemma 3 4B: 68 writes per token; P1–P3; batch/decode | REAL-SUBJECT WITNESSED on both CPU backends (production and reference). Cross-backend: structure identical; values at aligned-realization precision (7.3e-5 worst) with Q8 capped, at Q8 precision (1.5e-1 worst) under the default policy. |
| Gemma 3 12B: 96 writes per token; P1–P3; batch/decode | REAL-SUBJECT WITNESSED on the production CPU backend (768 = 96 × 8). Reference arm NOT RUN. |
| Qwen3-4B / Qwen3-0.6B: 72 / 56 writes per token | FORECAST |
| Run provenance record (pinned forms, arm, basis) exists and fingerprints canonically | STRUCTURALLY WITNESSED (unit tests on the golden plan: same backend → same fingerprint, two backends → different, every field moves it); printed on every real-container run. Persistence in a run envelope NOT CLAIMED — V3-STREAM-1. |
| P5 observer cost on Granite, median +0.15 ms per token | MEASURED ONCE, NOT CHARACTERISED — quiet-window re-run owed; not quotable |
| Metal whole-stack path observation | NOT CLAIMED — separate rung |
| Per-head observation | NOT CLAIMED — separate rung |

## Verdict rule

V3-OBS-1 is complete only when P1, P2, P3 and P4 PASS and P5 is MEASURED. Any semantic
failure leaves the rung incomplete. Completion unlocks V3-OBS-2 (stats and projection
event schema, the Observatory's actual input), then stream, record, replay and compare.
