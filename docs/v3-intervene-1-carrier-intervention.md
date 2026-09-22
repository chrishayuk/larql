# V3-INTERVENE-1 — can the canonical decode step change one carrier at one declared address, and prove it changed nothing else?

Pre-registered 2026-09-20, before any implementation. Properties and forecasts below are
FROZEN. Nothing here claims a result.

Programme: OBSERVE. A sibling of V3-LENS-1 and V3-HEAD-OBS-1 above V3-OBS-1 (#484) and
V3-STREAM-1 (#485). The first causal manipulation on the VINDEX3 execution authority. The
migration ledger names it as the home of the legacy walk's `ZeroAblateHook`, `SteerHook`
and activation patching; none of those exists on VINDEX3 today.

Specification proceeds now. Implementation against the substrate waits for the `observe`
verb (#486) to land: no rung under moving substrate.

---

## The question

V3-OBS-1 reads every carrier write and proves, per run, that reading changed nothing
(P1: bit-identical logits, observed against unobserved). An intervention is the opposite
promise: it changes exactly one thing, at one declared address, and everything the
executor does after that point is the model's own response to the change.

The legacy walk had three shapes of this — zero a residual row, add `α·v` to it, replace
it with a vector from another run — but each was a hook on the old forward with no
identity, no provenance for the injected vector, and no way to tell an intervened trace
from a baseline after the fact. VINDEX3 refuses to inherit that. Here an intervention is
a declared object with an address, a kind and a vector provenance; its receipt names it;
a run with a no-op intervention is the unobserved run bit for bit; and a vector patched in
from another run carries that run's identity, never a bare array.

V3-INTERVENE-1 asks whether that can be threaded through the one decode step without a
second execution path, and whether the first causal arm the programme already needs —
HEAD-1's CARRIED patch — can run on the executor and be compared with its sealed rows.

---

## What already exists (read before implementing)

| Fact | Where |
|---|---|
| The one decode step: `run(entry, observer, mutation)`; every public and test entry is this | `opplan/exec/decode.rs`, `DecodeSession::run` |
| The write: `leave_site` computes `delta`, calls `backend.residual_add(h, &delta)` for `Carrier::Single`, and fires `carrier_write` with `before`, `delta`, `after` borrowed; `enter_site` is the same carrier entering the next site | `opplan/exec/decode.rs`, `leave_site` / `enter_site` |
| The `Mutation` seam: a `#[cfg(test)]` DEFECT vocabulary (one deliberate ordering mistake per named variant; `Mutation::None` in production; per-family enums for gated-delta, KDA, MLA) — the place a second, intervention enum threads, never a set of new variants | `opplan/exec/controls.rs`; `decode.rs` threads it |
| A test-only entry that hands a caller-supplied state to the first executed layer (`Entry::Bundle`, hyper-connected components only) — prior art for injecting state, not an intervention | `opplan/exec/decode.rs`, `step_from_bundle` |
| The parity gate this rung must keep for observers and extend for interventions | `opplan/exec/tests/carrier_write.rs` (V3-OBS-1 P1) |
| Run identity from the declared authorities; the receipt; the record read fails closed on a hash or count mismatch | `larql-inference/src/vindex3/record.rs` (`RunIdentity`, `Receipt`, `RunRecord::read_jsonl`) |
| Run provenance: lowering, realizations, arithmetic arm, basis identity | `opplan/exec/provenance.rs` |
| A prompt in, a record out, prepared once | `larql-cli … vindex3_cmd/observe.rs` (#486) |
| The first causal arm, sealed outside this repo: CARRIED = `h15^A + (h14^B − h14^A)` at the last prompt position, held through an 8-token continuation, 336 same-entity/different-relation pairs on the HEAD plans' subject (8 heads × 256 = Gemma 3 4B geometry), MLX f32 | `~/chris-source/chris-experiments/semantic_graph/SG_B14_HEAD1_PLAN.md`, the Intervention table and gate G8 |

What does NOT exist: any V3 intervention; any way to source a vector from another run
(the record carries stats and projections, never carriers); any receipt field that says
a run was intervened.

---

## Contract properties (frozen as properties, not as names)

**I1 — a second enum, not new variants.** Interventions are their own type on the
decode step, threaded beside `Mutation`, never as variants of it. `Mutation` stays the
defect vocabulary. Production threads `Intervention::None` exactly where it threads
`Mutation::None`.

**I2 — one address form.** An intervention names `(layer, SublayerSite, positions)`
and acts on the `Single` carrier as it leaves that site — the same vector `carrier_write`
reports as `after`, before any layer scale. A `Bundle` or `History` carrier at the
address is a declared refusal in this rung (I8), not a silent projection onto stream 0.

**I3 — three kinds, exact.** `Zero` sets the carrier to zero. `Add(v)` sets it to
`after + v`. `Replace(v)` sets it to `v`. `v` is `[hidden]` wide, finite, and the
executor performs the arithmetic in the backend's f32 residual path so an `Add` of a
zero vector is a no-op bit for bit.

**I4 — a vector has provenance.** Every `v` carries where it came from: a caller
literal (its hash), or a captured carrier from a named run at a named address
`(run_id, layer, site, position)` with its hash. The receipt records the provenance
tuple and the hash. The vector itself is never written to the record.

**I5 — capture is an observer, in memory, declared.** Sourcing a carrier from another
run is a `StepObserver` that copies the carrier at declared addresses into memory,
keyed by address, and hands them to the intervention with their provenance. Nothing is
persisted; the receipt design of V3-STREAM-1 is untouched. A declared, priced
full-capture file is a different, later rung and is out of scope here.

**I6 — the intervention is applied once and its consequences are the model's.** The
patched carrier feeds the next site exactly as an unpatched one would: the next layer's
projections, the K and V rows appended for that position, every later position's
attention over them. Nothing downstream is patched again. This is the legacy walk's
"held through the continuation", made a property of the executor rather than a habit of
a script.

**I7 — identity and receipt.** A run with a non-`None` intervention has a different
run identity from the same prompt without it: the intervention declaration is part of
the identity's inputs. The receipt carries the declaration, the kind, the address, the
vector provenance and hash, and `interventions_applied` (a count). A record from an
intervened run cannot be read as a baseline.

**I8 — refusals are declared.** An address off the plan, a position never reached, a
non-`Single` carrier at the address, a vector of the wrong width or non-finite, or a
provenance whose hash does not match the bytes handed over, each refuses before the
first token executes. An intervention that is declared but never fires (the position
was not reached) is a refusal at the end of the run, on the receipt, not a silent
no-op.

**I9 — observers still see the truth.** With an intervention armed, `carrier_write` at
the intervened site reports the patched `after` (and a `delta` that includes the patch),
and the lens (V3-LENS-1) at that site reads the patched carrier. Observation reports
what executed.

---

## Frozen acceptance properties

**IP1 — the no-op law.** `intervene(None, run)` and `intervene(Add(0), run)` are
bit-identical to the unobserved run at every position, on the reference and production
CPU backends, on the golden plan and on every real subject run. This is the analogue of
V3-OBS-1 P1 and is the gate every later intervention rung inherits.

**IP2 — exactness at the address.** With `Zero`, `Add(v)` or `Replace(v)` armed at
`(layer, site, p)`, the carrier the `carrier_write` tap reports as `after` at that
address is, bit for bit, zero / `after_unpatched + v` / `v`. Every other write at
position `p` before the address, and every write at positions `< p`, is bit-identical to
the unintervened run.

**IP3 — causality.** Logits at positions `< p` are bit-identical to the unintervened
run. Logits at `p` differ unless the intervention was a no-op. (No claim is made about
later positions beyond I6's mechanism; a later position may or may not differ.)

**IP4 — provenance round trip.** A carrier captured from run B at `(L, site, p)` under
I5, patched into run A with `Replace`, yields at that address exactly run B's `after`
bit for bit, and the receipt names run B's identity and the hash.

**IP5 — refusals fire.** Each I8 condition is witnessed by a test that constructs it and
asserts the refusal text and that no token executed (or, for the never-fired case, that
the receipt names it).

**IP6 — on the record.** An intervened run's record carries the declaration and
provenance in its identity and receipt; replay is equal; the record reads back and the
receipt's `interventions_applied` equals the number of firings.

**IP7 — the first causal arm.** Gemma 3 4B IT, production CPU, `LARQL_CPU_MAX_FORMAT=bf16`
(the only cap the policy honours), ONE HEAD-1 donor pair chosen before the run and named
in the amendments: record run A (recipient) and run B (donor) with I5 capture at the
carrier entering block 15 (the FFN write of layer 14) and at the carrier entering block
14; construct CARRIED as `Add(h14^B − h14^A)` at that address on run A; generate 8
tokens greedily. Record the continuation tokens and compare them with the sealed BACK-2
Stage B row for that pair. **No agreement is forecast** — the sealed rows are MLX f32,
the executor's arm is bf16-capped with a Q8-realised head; the comparison is recorded as
a comparison, and disagreement is a finding about the arm, not a failure of this rung.

---

## Forecasts

- **IF1**: IP1 holds bit for bit on both backends and all witnesses; the intervention
  seam adds no cost measurable above run-to-run noise when `None` is threaded.
- **IF2**: the one decode step gains one parameter (the intervention) and `leave_site`
  gains one branch; no other execution code changes. If the batch prefill path needs the
  seam for IP7's prompt positions, that is recorded as a finding and the freeze is
  amended before implementation continues.
- **IF3**: the STREAM-1 schema gains one event kind (`Intervened`) and the receipt gains
  the declaration fields; `RunIdentity` gains the declaration hash. Nothing else.
- **IF4**: the CLI's `observe` verb gains one argument, a declaration file, and a
  `--capture` address list for I5; there is no separate `intervene` verb in this rung.
- **IF5**: IP7's continuation agrees with the sealed row on the first token and may
  diverge later; if it agrees on all eight, that is recorded, not claimed as
  replication.

---

## Implementation (recorded 2026-09-20, after the properties above were frozen; #486 merged first)

`opplan/exec/intervene.rs`: `InterventionKind`, `VectorProvenance` (`Literal` | `Captured`),
`Address`, `Intervention` (the hash is checked at construction), `InterventionPlan` (refuses
two interventions sharing a position at one site; `admit` refuses a `Bundle` or `History`
carrier, a layer off the executed range, an FFN site on a layer without an FFN program, and a
vector of the wrong width; `declaration_sha256`; `unreached`), `Firing`,
`InterventionStepOutput`, and `CarrierCapture` (I5). `decode.rs`: the one step takes the plan
beside `Mutation`; `leave_site`'s `Single` arm applies the intervention after the branch's
write lands, in the backend's residual path, reports `delta = after − before`, and fires
`StepEvent::Intervened` before the write's record; `DecodeSession::step_intervened` admits
before the token executes. `larql-inference`: `RunIdentity.intervention_sha256`,
`EventKind::Intervened`, receipt fields `interventions_declared`, `interventions_applied`,
`intervention_refusal` (named only on a complete record), `RunRecorder::with_interventions`,
`Vindex3Session::step_intervened`. CLI: `vindex3 observe --intervene <decl.json>` (a literal
vector, a reference into a capture file with that run's provenance, or the difference of two
references), `--capture <layer:site:position,…> --capture-out <file>` (a declared capture the
run never reached refuses).

| Property | Result |
|---|---|
| IP1 no-op law | PASS — `None` and `Add(0)` bit-identical to the unobserved run on the reference and production backends, golden plan |
| IP2 exactness | PASS — `Zero`, `Add(v)`, `Replace(v)` exact at the address on both backends; every write before the address in execution order bit-identical; `Zero`'s delta is exactly `−before`; the attention site addressable too; several positions fire each |
| IP3 causality | PASS — logits at positions before the address bit-identical; the address position moves |
| IP4 provenance | PASS — a carrier captured from run B replaces bit for bit in run A; provenance names run B, the address and the hash |
| IP5 refusals | PASS — empty positions, non-finite, empty vector, hash mismatch, overlapping addresses, missing capture, `zero` from capture (construction); layer off range, wrong width, FFN on a no-FFN layer, `Bundle` (hyper-connected substrate), `History` (attention-residual substrate) (admission, before any token); unreached addresses named |
| IP6 on the record | PASS — identity hash present and different from a baseline; the event at its position before the write's stats and structural event, once; receipt counts; refusal names the unreached position on a complete record and not on a prefix; JSON-lines round trip equal |
| IP7 first causal arm | RECORDED below — the executor's two baselines and its CARRIED continuation equal the sealed MLX f32 rows token for token, on the one pre-registered pair; a sign-reversed control does not |

Gates: full `larql-vindex` library suite 4787 pass, 0 fail, 8 ignored; `fmt` and `clippy -D
warnings` green on `larql-vindex`, `larql-inference`, `larql-cli`; doc-integrity green;
`intervene.rs` 97.5% line coverage, `decode.rs` 92.8%.

Amendments to the forecasts: **IF3** — the schema gained one event kind, one identity field
and three receipt fields (the forecast said "the declaration fields"). **IF4** — the verb
gained three flags, not one argument and a list: the capture needs an output file, and a
declaration needs to reference one. **IF2** held: one parameter on the step, one branch in
`leave_site`, the batch path untouched. One decision the freeze left open is now fixed: at an
intervened site the record's `delta` is `after − before`, so `before + delta` reaches the
patched `after` to rounding (exactly for `Zero` and `Add`, within one ulp for `Replace`), and
the `Intervened` event says why the chain identity is not the branch's own there.

**IP7 pre-registration (written before the run).** Pair: recipient (Denmark, currency),
prompt `The currency of Denmark is`; donor (Denmark, capital), prompt `The capital of Denmark
is` — the templates `fleet/E26_recurrent_depth_xarch/e26_common.py` uses, encoded with BOS as
the HF tokenizer does. Sealed rows from `captures/b14h1_full_meta.json` (condition CARRIED,
kind rel, "layer 15" = the vector entering block 15): CARRIED continuation ids `[68063,
236761, 108, 162623, 236789, 236751, 15130, 563]` = ` Copenhagen.\n\nDenmark's currency is`;
recipient baseline `[506, 46553, 155054, 568, 12536, 236855, 769, 108]` = ` the Danish Krone
(DKK).\n\n`; donor baseline `[68063, 236761, 1030, 563, 496, 28239, 532, 17110]`. V3 arm:
`gemma3-4b-it.vindex3`, production CPU, `LARQL_CPU_MAX_FORMAT=bf16`; capture both runs at
layer 13's FFN site at the last prompt position (the carrier entering block 14, `h14`); declare
`add` at layer 14's FFN site (the carrier entering block 15) of the difference `h14^B − h14^A`;
generate 8 greedy tokens. Recorded: the executor's two baselines against the sealed baselines,
and the CARRIED continuation against the sealed row. No agreement is forecast.

**IP7 result (recorded 2026-09-20, after the pre-registration above).** Release CLI from this
branch, `vindex3 observe` on `gemma3-4b-it.vindex3`, production CPU, `LARQL_CPU_MAX_FORMAT=bf16`,
prompts encoded to six ids with BOS (last prompt position 5), 8 greedy tokens, three runs plus
one control. Records `A.jsonl`, `B.jsonl`, `carried.jsonl`, `anti.jsonl` (2898–2899 events over
14 positions each, 68 carrier writes per position, complete); capture files `A.json`, `B.json`
at layer 13's FFN site, position 5; declaration hash `9f1a03f6…`, 1 declared, 1 fired, no
refusal.

| Arm | Executor continuation (ids) | Sealed MLX f32 row | Agreement |
|---|---|---|---|
| A, recipient baseline `The currency of Denmark is` | `506,46553,155054,568,12536,236855,769,108` ` the Danish Krone (DKK).\n\n` | same | 8/8 |
| B, donor baseline `The capital of Denmark is` | `68063,236761,1030,563,496,28239,532,17110` ` Copenhagen. It is a vibrant and historic` | same | 8/8 |
| CARRIED: `add(h14^B − h14^A)` at layer 14 FFN, position 5, on A | `68063,236761,108,162623,236789,236751,15130,563` ` Copenhagen.\n\nDenmark's currency is` | same | 8/8 |
| control: `add(h14^A − h14^B)`, same address, on A | `506,46553,155054,568,12536,236855,769,108` (A's own baseline) | — | differs from CARRIED, as it must |

Read at honest strength: one pair, one model, one arm, a bf16-capped Q8-realised executor
against an f32 oracle, and the pre-registration forecast nothing. What it shows is that the
V3-native intervention reproduces the sealed CARRIED row exactly on this pair, that the
continuation is not a copy of the donor's (it diverges from the donor baseline at the third
token exactly where the sealed row does), and that the comparison could have failed (the
sign-reversed control returns the recipient's own continuation). It is recorded as a
comparison on one pair; the 336-pair replication belongs to EXPERIMENT-1 as a registered
procedure, not to this rung.

## Out of scope

Per-head arms (`H_h`, `LOO_h`: ATTR-1C, which needs V3-HEAD-OBS-1's split), scaling
interventions, route forcing on MoE, interventions on `Bundle` or `History` carriers,
KV-row surgery (ownership deliberately unassigned until this rung answers whether an
intervention is an operation over any addressable V3 state or over execution carriers),
the Metal path, the batch prefill path unless IF2 forces it, any full-capture file, a
Python surface.

---

## Verdict rule

V3-INTERVENE-1 is complete only when IP1–IP6 PASS and IP7 is recorded with its
declared arm. It unlocks EXPERIMENT-1 (intervene-two-arm as a registered procedure) and
ATTR-1C. If IP1 cannot be made to hold on a backend, that backend refuses interventions
by declaration and the rung records the refusal; it does not weaken the law.

---

## Rebase migration onto main (recorded 2026-09-21)

The implementation above was built against a pre-HEAD-OBS-1 `main`. `intervene-1-rebase`
re-threads it beside V3-HEAD-OBS-1's per-head observation seam (`attention_step_observed`,
`wants_attention_heads`) and V3-LENS-1, both of which landed after the original freeze. The
port is mechanical — `intervene.rs`, the `Intervened` event, `SiteContext`'s intervention
field, the firing bookkeeping, the record/receipt fields, and the CLI's `--intervene` /
`--capture` surface all carry over unchanged in shape — and IP1–IP6 were re-run in full on
the rebased seam, not merely recompiled:

- All 13 golden-plan tests (IP1–IP6 plus the declaration-identity/unreached tests) PASS on
  both the reference and production CPU backends.
- Full suites green on the rebased tree: `larql-vindex` 4804 pass / 0 fail / 8 ignored,
  `larql-inference` 1545 pass / 0 fail / 5 ignored, `larql-cli` 919 pass / 0 fail / 1 ignored
  (including the CLI-level capture → provenance-replace → CARRIED-difference → refusal
  end-to-end test). `fmt --check` and `clippy -D warnings` clean on all three crates.
- Coverage on the rebased files: `intervene.rs` 97.54% line / 100% function / 98.95% region;
  `decode.rs` 92.64% line / 83.33% function / 95.00% region — matching the original
  freeze's numbers (97.5% / 92.8%) to within rounding.

**The IP7 witness was reproduced exactly, not merely re-run.** Release CLI built from
`intervene-1-rebase`, `gemma3-4b-it.vindex3`, production CPU, `LARQL_CPU_MAX_FORMAT=bf16`,
same pair, same addresses, same declaration:

| Arm | Rebased executor continuation | Original IP7 row | Agreement |
|---|---|---|---|
| A, recipient baseline | `506,46553,155054,568,12536,236855,769,108` | same | 8/8 |
| B, donor baseline | `68063,236761,1030,563,496,28239,532,17110` | same | 8/8 |
| CARRIED | `68063,236761,108,162623,236789,236751,15130,563` | same | 8/8 |
| control (sign-reversed) | `506,46553,155054,568,12536,236855,769,108` = A's own baseline | same behaviour | matches original finding |

Declaration hash `9f1a03f63f7bc235bed7394d6bba4ff640cc1e8416599b54890411addab21d35` (prefix
matches the original freeze's recorded `9f1a03f6…`); 1 declared, 1 fired, no refusal, on both
baselines and the CARRIED run. The engineering migration is therefore CLOSED: the seam holds
its no-op law, its exactness, and its one recorded causal result, unchanged, on top of
HEAD-OBS-1. INTERVENE-1 is unfrozen for the next rung (EXPERIMENT-1 / ATTR-1C) rather than
needing to be re-earned.
