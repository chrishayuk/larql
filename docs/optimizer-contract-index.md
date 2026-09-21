# REPRESENT / optimizer — contract index

**Status: frozen at OPT-CONTRACT-1.**
Qualified against main `2944ace6` (ACTUATION-1, #463).

A second index, not an extension of the first.
[`represent-v1-contract-index.md`](represent-v1-contract-index.md) defines
itself as the frozen REPRESENT/accounting v1 contract set of nine, and the
optimizer's invariants are a different programme with a different freeze.
Appending to it would make "v1" mean two things.

Same discipline: for each contract below, **here is what must go red if you
violate it.**

> **"Frozen" does not mean immutable code.** It means any semantic change to
> one of these contracts requires an explicit new transition, with its own
> evidence and controls, rather than silently redefining it.

Every row names a test that exists at the path recorded.
`scripts/check_contract_index.py` reads this file alongside the v1 index,
extracts every `name` — `path` pair, and fails when a test cannot be found
where the index says it is. It checks EXISTENCE and LOCATION; `cargo test`
decides whether the contracts still hold. It fails closed: a citation row it
cannot parse is an error, not a row it quietly skips.

Paths are relative to the repository root. `…/actuate/` abbreviates
`crates/larql-vindex/src/format/vindex3/represent/actuate/`, and `…/state/`
abbreviates `crates/larql-vindex/src/format/vindex3/represent/state/`.

## What these five protect

The first four protect scientific identity along the forward path. The fifth
protects the failure path that surrounds all four.

```text
selection  →  protocol  →  preparation  →  execution        1, 2, 3, 4

when any of those boundaries refuses, the reason may not lose
the evidence needed to act on it                                     5
```

---

## 1. Selected identity is reconstructable

**Invariant.** A selected `MeasurementKey` becomes exactly one truthful
`MeasurementRequest`. Resolving the request's candidate map over the record's
own surface, under the record's own layout policy, yields the physical state
the key names — and the whole four-part key is re-derivable from the request
alone. A map that presents a different state cannot be handed out as a request
for that key.

**Authority.** `MeasurementRequest::of`, `MeasurementRequest::derived_key`.

| | |
|---|---|
| positive | `a_request_re_derives_the_whole_experiment_from_itself` — `…/actuate/tests/request.rs` |
| positive | `a_record_that_prices_and_declares_its_protocol_prepares_a_real_experiment` — `…/actuate/tests/prepare.rs` |
| negative | `a_request_for_a_state_the_map_does_not_present_is_refused_naming_both` — `…/actuate/tests/request.rs` |
| negative | `an_applied_set_the_vocabulary_does_not_have_builds_no_map` — same file |
| qualified by | PR #463 |

---

## 2. Protocol identity has authoritative declarations

**Invariant.** A digest alone is not enough. The record carries the
`EvidenceBank` and `InstrumentSemantics` its standing intent's digests stand
for, and they self-check against those digests — against the intent, and
separately against the key actually selected. A record that declares one
protocol and searches under another is refused.

**Authority.** `MeasurementProtocol::describes`,
`MeasurementProtocol::describes_key`.

| | |
|---|---|
| positive | `the_record_s_protocol_describes_the_experiment_it_searches_under` — `…/state/protocol_tests.rs` |
| positive | `describing_the_intent_does_not_describe_every_key` — same file |
| negative | `a_protocol_declaring_another_corpus_is_refused_naming_both_digests` — same file |
| negative | `a_protocol_declaring_another_meaning_is_refused_distinctly` — same file |
| negative | `a_record_that_names_its_protocol_only_by_digest_refuses_and_says_so` — `…/actuate/tests/prepare.rs` |
| qualified by | PR #463 |

---

## 3. Preparation cannot author an experiment

**Invariant.** The bridge restates a selection; it may not change what is
measured. Every control reaches the request from the record — never from an
adapter's default — preparation follows the route the policy ranked first, is
idempotent, and writes nothing. A gate the build cannot resolve, or has
redefined under the same name, is refused rather than silently applied.

**Authority.** `PreparedExperiment::of`, `RequestRefusal::GateRedefined`.

| | |
|---|---|
| positive | `the_request_follows_the_route_the_policy_ranked_first` — `…/actuate/tests/prepare.rs` |
| positive | `preparing_twice_gives_the_same_request_and_changes_nothing` — same file |
| positive | `the_controls_come_from_the_record_and_never_from_the_adapters_defaults` — `…/actuate/tests/request.rs` |
| negative | `a_record_whose_gate_this_build_has_redefined_is_refused` — same file |
| negative | `a_record_naming_a_gate_this_build_does_not_implement_is_refused` — same file |
| qualified by | PR #463 |

---

## 4. Execution cannot substitute an experiment

**Invariant.** An executor is selected by the procedure the record declares,
in any registration order, and two executors may not claim one procedure. The
observation it returns must be an observation of the experiment it was handed;
one about another experiment is refused rather than recorded.

**Authority.** `ExecutorRegistry::execute`,
`ExecutionRefusal::ObservedAnotherExperiment`.

| | |
|---|---|
| positive | `two_distinct_procedures_dispatch_by_name_in_either_registration_order` — `…/actuate/tests/executor.rs` |
| negative | `an_executor_that_answers_about_another_experiment_is_refused` — same file |
| negative | `two_executors_claiming_one_procedure_are_refused` — same file |
| qualified by | PR #463 |

---

## 5. A refusal preserves its actionable authority when rendered

**Invariant.**

> A typed refusal is not complete merely because it contains the facts needed
> to diagnose or repair the failure. Its rendered form must preserve: what
> failed; the expected authority or identity; the conflicting observed
> authority or identity where one exists; and any actionable alternative the
> refusal promises to provide.

**Scope.** OPTIMIZER refusals — the refusal types on the actuation and
protocol and ingestion paths named below. This is not a claim that every `Display` in LARQL is
normative: an internal diagnostic must not be pulled into a scientific-
interface guarantee it was never designed to carry. A refusal enters this
contract when it is part of the optimizer's answer to an operator or an agent.

**Why it is a contract and not error-message hygiene.** ACT1-N8 is the
evidence. The per-file coverage gate on #463 found that a third of stage 5b's
refusals held exactly the right facts and had never had their rendered form
read by anything. The variant containing useful facts is not the same claim as
the caller being able to act on the refusal, and only the second is what a
refusal is for.

**Authority.** The `must_say` matches in `…/actuate/tests/refusals.rs`, which
are exhaustive — a new refusal arm cannot be added without deciding what its
message must say.

| | |
|---|---|
| positive | `every_request_refusal_names_what_it_promises` — `…/actuate/tests/refusals.rs` |
| positive | `every_locator_refusal_names_what_it_promises` — same file |
| positive | `every_execution_refusal_names_what_it_promises` — same file |
| positive | `wrapping_a_refusal_keeps_the_inner_ones_message` — same file |
| qualified by | PR #463 |

OPT-6 extends the same refusal obligation to ingestion, with an exhaustive
`must_say` match. Its admission and write-closure witnesses also pin the boundary
between an independently valid artifact and evidence for an authorised experiment.

| | |
|---|---|
| positive | `every_ingestion_refusal_preserves_its_actionable_authority` — `crates/larql-vindex/src/format/vindex3/represent/ingest/tests.rs` |
| negative | `valid_y_is_established_then_refused_for_requested_x_without_any_scientific_change` — same file |
| negative | `freshly_sealed_incomplete_run_with_correct_positions_and_gate_refuses` — same file |
| negative | `each_missing_run_validity_obligation_refuses_transactionally` — same file |
| negative | `conflicting_duplicate_preserves_both_observations_and_all_facts` — same file |
| closure | `every_production_recording_call_has_an_exact_registered_owner` — `crates/larql-vindex/tests/ingestion_closure.rs` |
| control | `scanner_detects_new_owners_aliases_and_macro_calls_but_excludes_test_only_code` — same file |

---

## What this index does not yet cover

Stated so absence is a fact rather than an oversight.

- **Quality-driven improvement at scale.** Transition C's paired witness below
  demonstrates changed selection after accepting an experiment, through the
  existing `AlreadyObserved` rule. It does not claim quality-driven promotion
  or a K3 optimisation result.
- **Numerical execution truth.** Candidate and bank readers establish persisted
  input authority. Ingestion does not independently reproduce the numerical
  observation or certify an executor's runtime pointer/attribution reports.
- **A second real procedure.** Contract 4's dispatch is witnessed with stub
  procedures. ACT1-N5 stays open: no semantic relationship between an
  instrument's declared procedure and an execution procedure is claimed.

The Transition C composition claim includes both accepted and rejected evidence
in one witness. Its expected successor is fixed before dispatch or ingestion.

| | |
|---|---|
| paired witness | `accepted_evidence_changes_future_selection_and_rejected_evidence_cannot` — `crates/larql-vindex/src/format/vindex3/represent/ingest/loop_tests.rs` |
