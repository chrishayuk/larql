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
