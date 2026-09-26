# MEASURE-PLAN-2: plan-v1 readings in the optimiser loop

Frozen 2026-09-26, before implementation. The types, the refusals, the acceptance witnesses and the PR
order below are FROZEN. Nothing here claims a measurement result or registers a gate.

Programme: REPRESENT, slice 2 of three (`docs/measure-plan-1.md`). Slice 1 built the procedure
`teacher-forced-two-arm/plan-v1`. This slice lets its readings be requested, executed, ingested, stored
and judged by the optimiser loop (`docs/auto-rep-1.md`, AUTO-REP-1b). Slice 3 registers gates for
non-Kimi models. Registering a gate id or a threshold is out of scope here.

---

## Why a new kind, not a filled `QualityBank`

The loop is typed to the Kimi procedure's reading end to end: `Observed`, `MeasurementArtifact`,
`MeasurementRegistry`, `SearchConfig.gate`, `gate_by_id` and `ConstraintVector::of`. The two readings
are different instruments:

| | Kimi `QualityBank` (`teacher-forced-two-arm/v1`) | plan-v1 `Aggregate` |
|---|---|---|
| KL | nats over the baseline's top 2048 ids | nats over the full vocabulary |
| quantiles | p50, p95, p99 | mean, p50, p99, max |
| counts | top-1 flips, top-10 changes | top-1 agreement share, top-5 overlap |
| other | routing, displacement, covered mass | ΔNLL, max-abs logit delta, per-category and per-margin splits |

Filling a `QualityBank` would force defaults for fields plan-v1 never measures. A `0` there reads as "no
change" when it means "not measured", and a Kimi gate would then judge full-vocabulary nats as though they
were top-2048 nats. Nothing today binds a gate to the instrument that took a reading. This slice makes
that binding structural.

## Kinds

- **`Observation`** is `Kimi(QualityBank)` or `Plan(PlanObservation)`.
  - It is serialised untagged. A Kimi reading serialises exactly as a `QualityBank` does today, so every
    persisted snapshot and artifact reads back byte-identically.
  - `PlanObservation` carries `procedure: "teacher-forced-two-arm/plan-v1"`, `sequences`, `positions`,
    and the `all`, `by_category` and `by_margin_band` aggregates from `Summary`, and nothing else. It
    refuses unknown fields.
- **`Gate`** is `Kimi(QualityGate)` or `Plan(PlanGate)`, also untagged, with the Kimi form unchanged.
  - `PlanGate` names its criteria over plan statistics only:
    - `positions_min` and `kl_p99_max` (full-vocabulary nats), both required;
    - `kl_mean_max`, `top1_disagreement_max` (1 − agreement) and `delta_nll_mean_max`, each optional.
  - This slice defines that vocabulary. It registers no plan gate id or threshold: `gate_by_id` resolves
    none outside tests.
- **`ConstraintVector::judge(&Gate, &Observation)`** refuses a kind mismatch (`GateMismatch`) instead of
  producing margins.
  - The Kimi arm is today's `ConstraintVector::of`, unchanged.
  - The plan arm produces margins with new `Criterion` and `Statistic` variants, so a plan margin can never
    be read as a Kimi one.
- **Kimi-only analyses** take the Kimi reading or refuse by name, and never coerce a plan reading. These
  are diagnostics, resampling, stream replay, the observation stream and bank statistics.

## Ingestion and execution

- **Procedure registry.** `MeasurementProcedure` gains plan-v1. `by_name` resolves both procedures.
- **Bank evidence per procedure.**
  - The Kimi verifier is unchanged.
  - A token-bank verifier checks, for a `teacher-forced-token-bank/v1` bank:
    - the schema and payload authority;
    - `bank_id` recomputed from the manifest;
    - the `seq-NNN` order;
    - every payload's seal.
  - Positions are derived from the manifest's per-sample token counts, never from a fixed per-sample
    length.
- **Facts per procedure.** An artifact carries `Kimi(VerifiedFacts)` or `Plan(PlanVerifiedFacts)`.
  - Completeness is each procedure's own predicate, with `PlanVerifiedFacts::complete(sequences)` for
    plan-v1.
  - The artifact's facts, observation and procedure must agree in kind, or it is refused.
- **Gate agreement.** A reading whose kind differs from the snapshot gate's kind is refused at ingestion,
  so a valid record never holds a reading its gate cannot judge.
- **Execution.**
  - `ExecutionRefusal` carries a `PlanRefusal`, typed, not stringified.
  - A `PlanTeacherForcedExecutor` builds both arms from the request and runs `plan::run`. The arms come
    from the locator's container, the protocol's bank and a declared backend for each arm.
  - A library arm builder in `larql-vindex` covers the interpreter backends (`reference`, `production`).
  - Lowered backends are registered by `larql-cli`, where `LoweredSession` lives, as a second executor
    with the same procedure.

## Snapshot producer

A production builder turns a real container into a `SearchSnapshot`. Today only test fixtures build one.
It takes:

- the source container;
- a `RepresentSpec` (encoding and roles, no protections);
- a token bank;
- a gate id resolved through `gate_by_id`.

It derives the rest:

- the model from `read_source_identity`;
- the surface from the container's own plan roles;
- the group vocabulary from AUTO-REP-1b;
- accounting from `read_source_storage`;
- the protocol and instrument from the bank and the procedure.

The instrument is `metric: KL nats`, `truncation: None`, and `procedure: plan-v1`.

## Acceptance witnesses (each committed RED before the code that turns it green)

- **W1, Kimi records are byte-identical.** Every existing snapshot and artifact fixture round-trips
  byte-for-byte through the new kinds. Every existing `represent` test passes unchanged.
- **W2, a gate judges only its own instrument.** A Kimi gate over a plan reading, or a plan gate over a
  Kimi reading, is `GateMismatch`, never margins. Ingestion refuses a plan reading into a Kimi-gated
  snapshot.
- **W3, absent means absent.** A plan reading deserialises only as `Plan`. Stripping its procedure field
  refuses it rather than reading it as Kimi. No plan margin is produced for a criterion plan-v1 does not
  measure.
- **W4, the token bank is sealed.** Tampering with one payload is refused before execution, and a
  recomputed `bank_id` that disagrees is refused. Positions come from the token counts.
- **W5, plan facts are complete or refused.** A plan artifact missing any verified fact is refused as an
  incomplete run. Kimi facts on a plan reading are refused.
- **W6, the loop runs on plan-v1.** On the dense fixture, AUTO-REP-1b runs end to end through the real
  `PlanTeacherForcedExecutor` with interpreter arms: compile, execute `plan::run`, ingest, adjudicate. It
  uses a test-only plan gate, so no production gate id exists.
- **W7, the producer builds a record the loop accepts.** For the dense fixture container and bank, the
  producer's snapshot is accepted by `auto_rep::run` without hand edits. Its surface roles equal the
  container's plan roles.

## Amendment A1 (2026-09-26, before any implementation landed)

Recorded before the code it changes, on review of the frozen text. It adds two things; nothing above is
withdrawn.

### A1.1 Characterisation-only records

`SearchConfig.gate` becomes `Option<Gate>`. A record without a gate:

- **Can** ingest, store and replay readings, and carry accounting and cost.
- **Cannot** adjudicate anything. `adjudicate` returns `None`, the frontier holds no adjudications, and
  nothing is admitted or promoted.
- **Is refused by AUTO-REP.** An `ExactNoGood` needs a refused reading, and no reading is refused without
  a gate.

A present gate serialises exactly as before, so every persisted Kimi record is unchanged. This lets
slice 2 produce and fill a real plan-v1 record without inventing a slice-3 threshold. The test-only plan
gate remains, only for proving the loop end to end (W6).

### A1.2 Structured metric semantics and the gate binding

`InstrumentSemantics` gains an optional structured `semantics: MetricSemantics { quantity, unit, support,
direction }`:

- `quantity`: `KlDivergence`;
- `unit`: `Nats`;
- `support`: `FullVocabulary` or `TopN(n)`;
- `direction`: `ReferenceToCandidate`.

Its rules:

- **Digest.** The field enters the instrument digest only when present, so every existing Kimi instrument
  id, and every measurement key built from one, is unchanged.
- **Consistency.** When present, `support` must agree with `truncation`: `TopN(n)` with `Some(n)`, and
  `FullVocabulary` with `None`. Otherwise it is refused.
- **Plan instrument.** The plan-v1 instrument declares `KlDivergence`, `Nats`, `FullVocabulary`,
  `ReferenceToCandidate`.

**The gate binding.** A `PlanGate` also declares the `MetricSemantics` its limits are written in. Before
evaluation, ingestion and request preparation refuse a gate whose kind differs from the reading's, or
whose declared semantics differ from the instrument's. Kimi gates predate the field and bind by kind
alone. Kimi instruments carry no semantics until a Kimi migration declares them.

### Added witnesses

- **W8, characterisation-only.** A gate-less plan record ingests a plan reading, reports it, and replays
  it. It admits nothing, and `auto_rep::run` refuses it.
- **W9, semantics bind.** A plan gate declaring `TopN(2048)` refuses a reading from the full-vocabulary
  instrument. An instrument whose `semantics.support` disagrees with its `truncation` is refused. Every
  existing Kimi instrument id is unchanged.

These land in PR 1 (kinds and binding) and PR 2 (ingestion), with W8 and W9 committed RED first.

## Out of scope

- **Any plan gate id or threshold** (slice 3), and whether any criterion earns authority.
- **Migrating the Kimi procedure** onto the plan (BACKEND-AUTHORITY-1).
- **The first real AUTO-REP campaign** (AUTO-REP-1c), which needs slice 3's gate.

## PR order

1. **Kinds:** `Observation`, `Gate`, `PlanObservation`, `PlanGate`, `judge`, and the typed registry,
   artifact and snapshot. Witnesses W1–W3.
2. **Procedure in the loop:** the registry entry, the token-bank verifier, per-procedure facts,
   `PlanRefusal` in execution, the interpreter executor and the arm builder. Witnesses W4–W6.
3. **Producer and CLI:** the snapshot producer, the lowered executor registration, and a `vindex3` verb
   that runs AUTO-REP over a produced record. Witness W7.

## Amendment A2 (2026-09-26, before PR 3's implementation)

Recorded before the code it governs. It narrows PR 3's scope and splits it; nothing above is withdrawn.

### A2.1 The producer makes characterisation-only records

A search record carries four judgement fields besides the gate:

- `gate`;
- `tail_support` (the tail observations a percentile needs before it can price anything);
- `diagnostic_policy`;
- `semantics.promotion_rule`.

The only builder in the tree, a test fixture, fills them with Kimi values: `kimi-logit-balanced-v1`,
`route_cal_1`, `bs2-kimi-v1`, and `kimi-balanced-v1-authority-only/v1`. Every one is a judgement about an
instrument's readings, and none has been earned for plan-v1.

So `represent::produce` builds only characterisation-only plan records:

- `gate: None`.
- A diagnostic policy with no observations.
- A promotion rule id stating that nothing is promoted.
- A tail-support policy whose provenance states it is not earned. Nothing reads it while the record has
  no gate, since `adjudicate` returns `None` first.

Slice 3's pre-registration installs all four together. Borrowing any Kimi value, or inventing a plan one
here, is refused.

**Surface.** The producer's surface uses the same role authority `compile_representation` uses: the
container's plan roles for primary-text objects, and name classification otherwise. Both go through one
shared helper, never a copy. That makes the roles the solver groups by the roles the compiler compiles.

### A2.2 PR 3a: producer and verbs

- `represent::produce::{produce, ProduceInputs, plan_surface, plan_instrument}`.
- `larql vindex3 auto-rep init`: writes a produced record as JSON, which `optimizer-mcp --snapshot` can
  load.
- `larql vindex3 auto-rep run`: loads a record and runs AUTO-REP-1b with the interpreter plan-v1 executor
  and a plugin-aware compiler, then writes back the advanced record and the campaign record. It refuses a
  characterisation-only record, as `auto_rep::run` does, until slice 3 arms one.

**W7** is as frozen: the producer's record, with only the test-only gate installed, is accepted by
`auto_rep::run`, and its surface roles equal the container's plan roles. It adds:

- the producer writes no Kimi value into any judgement field;
- `init` round-trips through `optimizer-mcp`'s loader;
- `run` on a produced, gate-less record refuses before compiling anything.

### A2.3 PR 3b: the lowered executor

A plan-v1 `ExperimentExecutor` over `LoweredArm`, registered by `larql-cli`, with arms named by lowering
identity (Finding 1). It needs a Metal device. Its tests must fail, not skip, when the device or shader
library is missing, and it lands as its own PR.

## Implementation record

- **PR 1** (#579): the kinds, the binding and the optional gate. W1–W3, W9 and W8's adjudication half; eight mutations, each caught.
- **PR 2**: `RunFacts`, `ExecutionRefusal::Plan`, `MeasurementProcedure::PlanV1`, `ingest/token_bank_evidence.rs`, per-procedure completeness and plan invariants, `PlanTeacherForcedExecutor`, and `measure::plan::arm::interpreter_arm_for`.
  - **W4** is covered by unit tests of the token-bank verifier.
  - **W5, W6 and W8's ingestion half** run on the dense fixture with a real token bank and real plan-v1 runs. W6's first proposal, uniform NVFP4, measured a full-vocabulary KL p99 of 0.0031 nats against the test gate's 0.001. It was refused and cut, and two further real proposals were compiled and measured before the budget of 3 ran out.
  - **Mutations:** eight, each caught. Positions from a fixed length, seals unread, a declared per-sample length accepted, plan completeness skipped, facts and reading of different procedures accepted, the candidate arm reading the source, plan positions unchecked, and sample order unchecked.
- **Finding 1, arm names.** The executor first named its arms by role (`reference`, `candidate`). plan-v1 records `arm_changed` from arm names, so two identical execution arms looked like a changed variable, and a candidate arm reading the SOURCE container passed as a measurement. Arms are now named by their lowering identity. The mutation that points the candidate arm at the source now fails three witnesses.
- **Finding 2, `top5_overlap_mean`.** It is the mean COUNT of shared top-5 ids, in [0, 5], not a share. The first real run measured 4.30. The invariant bounds it by `TOP_K_OVERLAP`, and a witness refuses 5.5 and holds a real value above 1.
