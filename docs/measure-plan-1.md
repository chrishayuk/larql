# MEASURE-PLAN-1: a model-agnostic teacher-forced measurement over the plan

Frozen 2026-09-23, before implementation. The procedure, its refusals, its outputs, the acceptance
witnesses and the PR order below are FROZEN. Nothing here claims a measurement result.

Programme: REPRESENT, slice 1 of three. Slice 2 wires this procedure into the optimiser loop (executor
registry, ingestion, snapshot producer). Slice 3 earns gates for non-Kimi models. Neither is in scope here.

---

## Why

A representation decision needs a fidelity measurement. The one LARQL has cannot make most of them.

`represent::measure::run` (procedure `teacher-forced-two-arm/v1`) is Kimi-specific in every input:

| input | today | consequence |
|---|---|---|
| source | `KimiSourceModel`, bound by Hugging Face tensor names | only Kimi opens |
| candidate | `CandidateOverlay` over the routed-expert bank | an NVFP4 pack cannot be a candidate |
| head | hard-coded BF16 from the source (`kimi_source.rs:584-594`) | a compiled head cannot be measured |
| corpus | pre-embedded rows (`seq_{i}.f32`), schema `kimi-teacher-forced/v1` | no other model has a bank |
| gates | four `kimi-logit-*` ids | no gate applies elsewhere |
| invocation | env vars read by one test | no command reaches it |

The trigger was concrete. On gpt-oss-20b, compiling the output head to NVFP4 took the lowered decode
from 10.37 to 8.51 ms/token (a 3-repeat protocol run, 2026-09-23). Whether that head is acceptable could
not be asked of LARQL's own instrument. A hand-built comparison (HEAD-FID-1) was UNINFORMATIVE by its own
frozen rule. Meanwhile `bench/prompts/quality-bank-1/run_bank.py` measures any model through
`vindex3 exec`, but it has no validity proofs, no gate and different units (bits over the full vocabulary,
against the procedure's nats over the top 2048), and nothing reconciles the two paths.

## The claim of this rung

> One procedure measures any candidate realization of a VINDEX3 container against a reference realization,
> for any model the plan executes. It carries the same five claims as `teacher-forced-two-arm/v1`
> (a measurement plus four validity proofs), and it refuses by type when a proof fails.

The measurement runs over the plan (`ComponentOpPlan`), never over a model's Hugging Face names. The
Kimi procedure is untouched; it stays the authority for its gates until slice 2 or a Kimi migration
replaces it.

## Procedure `teacher-forced-two-arm/plan-v1`

### Request (every field required, refused when missing or unknown)

- **reference:** a container path and an execution arm (a `vindex3 exec --backend` name).
- **candidate:** a container path and an arm. Either may equal the reference's. What differs is the
  changed variable.
- **bank:** a token-id bank directory (schema below).
- **sequences:** how many bank samples to measure, taken in bank order from `seq-000`.
- **label** and **output directory:** where the report and receipt are written. There is no default
  path.
- **representation source:** `stored` for the candidate by default, so a pack arm cannot quietly
  quantise at load.

### Arms

An arm is a trait, `TeacherForcedArm`: given a sample's token ids, it returns every position's full-vocabulary
logits and its realization record (what it bound, from where). The procedure never knows which arm it holds.

- **Interpreter arm** (in `larql-vindex`): `PreparedOperands` plus any `PlanBackend`, stepped one position at
  a time through a `DecodeSession`, as `exec --logit-dump` does. With the reference backend it runs on
  every platform, so CI exercises the whole procedure.
- **Lowered arm** (in `larql-cli`, where `LoweredSession` lives): the session's teacher-forced step, as
  `exec --logit-dump` does on a lowered backend.

### Bank schema `teacher-forced-token-bank/v1`

- **`manifest.json`:**
  - schema;
  - `bank_id`;
  - the source prompt file and its sha256;
  - the tokenizer's sha256;
  - the chat-template policy (`raw` or `templated`), per sample;
  - `payload_authority: "teacher-forced-token-bank-payload/v1"`;
  - one entry per sample: id `seq-NNN`, category, token count, and the payload's sha256.
- **Payload:** one `seq-NNN.u32` file per sample, little-endian token ids.
- **Exporter:** a CLI subcommand tokenises a prompt bank with a container's own tokenizer. The first bank
  is Q-BANK-1's `prompts.json`, which is frozen. Editing it makes a new bank, never a new version of this one.
- **Seal:** the procedure reads a payload only after its sha256 matches the manifest.

### Metrics, per position

All metrics use the full vocabulary, in nats, computed in f64 from f32 logits:
- KL(reference ‖ candidate);
- top-1 agreement, and top-5 set overlap;
- ΔNLL of the actual next token, where the bank has one;
- the reference's top-1 margin (p1 − p2) and entropy, so a flip at a near-tie is distinguishable from a
  flip at a confident position.

Aggregates: mean, p50 and p99 of KL (p99 via the existing `KlP99` statistic), top-1 agreement, and each
of these split by bank category and by reference-margin band.

Units change on purpose. The procedure reports nats over the full vocabulary; the Kimi procedure's
top-2048 truncation stays its own. `run_bank.py` becomes a client of the new procedure in PR 4 and stops
computing its own numbers.

### Validity proofs (typed refusals, `Inadmissible` when numbers exist but are not evidence)

1. **Null arm.** The reference arm is run twice over the first `min(4, sequences)` samples. The two sets
   of logits must be bit-identical, or `NullArmNotZero`. This is the most dangerous condition: a
   non-deterministic device yields a plausible KL that is entirely artifact.
2. **Candidate scope.** The changed variable must exist, or `CandidateCompilesNothing`.
   - **Two containers:** they must differ.
   - **One container, two arms:** the arms' realization tables must differ.
3. **Non-candidate identity.** With two containers, every segment outside the candidate's declared
   compiled set must be sha256-identical across them, or `ProtectedOperandChanged` (this names the segment).
   The candidate's declared set comes from its `candidate.json` and index, never from the diff.
4. **Physical attribution.** Each arm's realization record must match its declaration, or
   `UnexpectedPhysicalRead`:
   - the candidate reads its compiled objects from the pack;
   - under `stored`, nothing is quantised at load;
   - the reference reads no pack the request did not name.
5. **Seal and read.** Every bank payload read must match its seal. Every pack segment an arm binds must
   match its recorded digest. Otherwise `SealMismatch`.
6. **Corpus.** The bank's tokenizer digest must equal each container's, or `CorpusNotForThisModel`.
   Position counts must agree across arms, or `PositionCountMismatch`.

Execution failures (no Metal device, an unreadable artifact, a refused step) are
`ExecutionFailure`, never `Inadmissible`. A `PlanVerifiedFacts` record states what each proof checked,
with a `complete()` that is explicit, not a count.

### Output

Written to the requested directory, the report before any summary is printed:
- `report.json`: request, both arms' identities (container digests, arm names, realization tables), the
  bank id and manifest sha256, the aggregates, and the verified facts.
- `positions.jsonl`: one record per position with every per-position metric.
- `receipt.json`: whether the run is admissible, and each refusal if not.

### No gate in this rung

The procedure characterises; it does not rule. The report states the `KlP99` sample-size adequacy from
UNCERTAINTY-2 (the p99 is biased low below 32 sequences, and needs 64–128 for ±25%), so an underpowered
reading says so. A decision, such as the NVFP4 head, is made by a separately pre-registered rule over
this procedure's output.

## Acceptance witnesses (each committed RED before the code that turns it green)

- **W1, the null arm catches nondeterminism.** An arm that perturbs one logit on its second run is refused
  `NullArmNotZero`. Mutation: removing the check makes the witness fail.
- **W2, identity is not a measurement.** A candidate identical to the reference (same container, same arm)
  is refused `CandidateCompilesNothing`. No report claims KL 0 as a finding.
- **W3, a foreign reference agrees.** On the dense fixture, the procedure's per-position KL, top-1 and
  margin match an independent numpy computation over the same `exec --logit-dump` files to within 1e-9
  nats per position. The numpy implementation shares no code with the procedure.
- **W4, protected segments are protected.** Tampering with one byte of a non-candidate segment of the
  candidate container is refused `ProtectedOperandChanged`, naming that segment.
- **W5, seals are read.** Tampering with one bank payload is refused `SealMismatch`.
- **W6, the corpus belongs to the model.** A bank exported with a different tokenizer is refused
  `CorpusNotForThisModel`.
- **W7, attribution is checked.** Under `stored`, a candidate arm that would quantise at load is refused
  `UnexpectedPhysicalRead`, not measured.
- **W8, Kimi is untouched.** `q2a_teacher_forced` and every existing `measure` test pass unchanged.
- **W9, a real run.** On gpt-oss-20b (Metal, lowered arms), reference R (`metal-lowered-f16` on the
  source) against C (`metal-lowered` on the conservative pack) completes as admissible, over at least 64
  Q-BANK-1 sequences, with every verified fact complete. The report is recorded; nothing is judged.

## Out of scope

- **Gates for non-Kimi models:** slice 3, earned from a consequence ladder, never transplanted.
- **The optimiser loop:** slice 2 (executor registration, ingestion, snapshot producer).
- **Migrating the Kimi procedure onto the plan:** tracked in BACKEND-AUTHORITY-1.
- **Multimodal inputs:** `exec` takes token ids only.
- **Deciding the NVFP4 head:** it gets its own pre-registration, run through this procedure after W9.

## PR order

1. **The bank:** schema, exporter subcommand, seal check. Witnesses W5 and W6.
2. **The procedure core in `larql-vindex`:** request, arm trait, interpreter arm, metrics, refusals, facts,
   output. Witnesses W1–W4, W7 and W8 on the dense fixture, runnable in CI.
3. **The lowered arm in `larql-cli`,** and the `larql vindex3 measure` verb.
4. **`run_bank.py` becomes a client;** then W9 on gpt-oss-20b.

Order of evidence, per the programme's rule: this freeze is committed first, then each PR's witnesses RED,
then the implementation that turns them GREEN.
