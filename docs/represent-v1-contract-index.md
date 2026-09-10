# REPRESENT / accounting — v1 contract index

**Status: frozen, pending ACCOUNTING-D1 (#457).**
Qualified against main `202ffb7b` (ATTESTATION-1, #447) plus #457.

This is an index, not an essay. It exists so that "frozen" means something
checkable: for each of the nine contracts below, **here is what must go red
if you violate it.**

> **"Frozen" does not mean immutable code.** It means any semantic change to
> one of these nine contracts requires an explicit new transition, with its
> own evidence and controls, rather than silently redefining v1.

Every row names a test that exists at the path recorded. If a rename or move
makes one of these names wrong, that is itself a contract event: update this
index in the same change, or the index is lying.

That is enforced rather than asked for. `scripts/check_contract_index.py`
reads this file, extracts every `name` — `path` pair, and fails when a test
cannot be found where the index says it is. It checks EXISTENCE and
LOCATION; `cargo test` decides whether the contracts still hold. The checker
fails closed: a citation row it cannot parse is an error, not a row it
quietly skips — an earlier version passed a renamed test by skipping it,
which is the failure mode a checker exists to prevent.

Paths are relative to the repository root. `…/exec/` abbreviates
`crates/larql-vindex/src/format/vindex3/opplan/exec/`, and `…/codec/`
abbreviates `crates/larql-vindex/src/format/vindex3/represent/codec/`.

---

## 1. Representation identity and revision

**Invariant.** A codec names itself, and the ABI identity it declares is
admitted back to it by the registry. A future revision, an alien family, or
disagreeing geometry is refused by name rather than decoded optimistically.

**Authority.** `CodecRegistry::admit`, `CodecIdentity`.

| | |
|---|---|
| positive | `every_codec_names_itself_and_its_abi_is_admitted_back_to_it` — `…/codec/tests/contract.rs` |
| positive | `the_identity_gate_the_index_calls_is_the_registry_s` — `…/codec/tests/registry.rs` |
| negative | `admit_refuses_a_future_revision_an_alien_family_and_disagreeing_geometry` — `…/codec/tests/registry.rs` |
| negative | `an_unregistered_label_is_refused_naming_every_registered_one` — `…/codec/tests/registry.rs` |
| qualified by | PR #419, `80911f23` |

---

## 2. Named streams and referenced operands

**Invariant.** A codec declares its streams by name and one values stream
first; what it depends on is *addressed* through the container's reference
table, never spelled by convention. A missing stream says what was bound; a
short one says how short.

**Authority.** `RepresentationCodec::streams`, `AuxiliaryReferences`.

| | |
|---|---|
| positive | `every_codec_declares_one_values_stream_and_declares_it_first` — `…/codec/tests/contract.rs` |
| positive | `one_target_serves_many_owners_and_each_owner_names_it_itself` — `…/auxiliary_references/tests/table.rs` |
| negative | `a_missing_stream_lists_what_was_bound_and_a_short_one_says_how_short` — `…/codec/tests/streams.rs` |
| negative | `each_way_a_dependency_can_be_wrong_is_refused_by_name` — `…/auxiliary_references/tests/closure.rs` |
| negative | `a_cycle_is_refused_with_the_path_that_closes_it` — `…/auxiliary_references/tests/closure.rs` |
| qualified by | PR #419 (`80911f23`), PR #441 (`85b46061`) |

---

## 3. Extents and access capabilities

**Invariant.** Extents run from the base up and a request past them is
refused. A codec admits its own alignment and refuses access granularity it
does not provide. Stored bytes are the certificate's bits at scale.

**Authority.** `RepresentationCodec::extents`, `CodecCapabilities`,
`RepresentationExtent`.

| | |
|---|---|
| positive | `every_codec_declares_its_extents_from_the_base_up_and_refuses_past_them` — `…/codec/tests/contract.rs` |
| positive | `stored_bytes_are_the_certificate_s_bits_at_scale` — `…/codec/tests/contract.rs` |
| positive | `an_external_provider_s_extents_reach_selection_and_a_budget_pins_one` — `crates/larql-vindex/tests/external_progressive_provider.rs` |
| negative | `every_codec_admits_its_own_alignment_and_refuses_access_it_lacks` — `…/codec/tests/contract.rs` |
| negative | `a_refusal_names_what_is_provided_and_what_is_required` — `…/codec/tests/capability.rs` |
| qualified by | PR #419 (`80911f23`), PR #440 (`624aa59f`) |

---

## 4. Realization-scoped residency

**Invariant.** Residency is a property of the **realization**, not of the
stored bytes: a decode profile is an f32 image, a direct profile is the
stored bits, and only requantisation changes values. A declaration that
disagrees with what the loader bound is a refusal, not a rounding.

**Authority.** `ResidencyProfile`, `declared_resident_for`, `reconcile`.

| | |
|---|---|
| positive | `the_decode_profile_is_an_f32_image_and_direct_profiles_are_the_stored_bits` — `…/codec/tests/residency.rs` |
| positive | `a_prepared_dense_plan_reconciles_every_pin_and_the_census_is_the_same_bytes` — `…/exec/tests/accounting.rs` |
| mutant | `every_resident_form_reconciles_with_its_declaration_and_a_mutated_geometry_breaks_it` — `…/exec/tests/accounting.rs` (the control is in the test: a mutated geometry must break reconciliation) |
| negative | `reconciliation_refuses_strays_omissions_and_disagreeing_forms` — `…/exec/tests/accounting.rs` |
| qualified by | PR #419, `80911f23` |

---

## 5. Typed-source fidelity certificates

**Invariant.** A certificate measures decoded values against the **typed
logical source tensor presented at the representation boundary** — not
against the stored representation (which would make every terminal extent
trivially exact), and not including kernel arithmetic (which is realization
qualification, a separate plane).

A lossless **carrier** may therefore state `0.0`; a codec that **fits** its
source states nothing until an attestation supplies a measured radius.
Radii compose only in like terms.

**Authority.** `FidelityCertificate` (see its type documentation, which is
the canonical statement of the referent).

| | |
|---|---|
| positive | `only_a_lossless_carrier_declares_a_radius_and_a_progressive_one_declares_one_per_extent` — `…/codec/tests/contract.rs` |
| positive | `every_shipped_certificate_is_stated_in_the_builds_own_terms` — `…/codec/tests/fidelity.rs` |
| negative | `a_radius_is_finite_and_not_negative` — `…/codec/tests/fidelity.rs` |
| negative | `a_metric_this_build_never_heard_of_is_expressible_and_refused_by_name` — `…/codec/tests/fidelity.rs` |
| negative | `composition_adds_like_terms_and_refuses_unlike_ones` — `…/codec/tests/fidelity.rs` |
| qualified by | PR #419 (`80911f23`), referent settled in PR #447 (`202ffb7b`) |

---

## 6. Artifact attestations and trust policy

**Invariant.** An attestation binds a measurement to an operand, extent,
codec family and revision, shape, content digest, dependency baselines and
recipe. **Presence is not trust**: `Bound` means the tuple holds and the
authority is recognised; only `Verified` — the payload hashed — hands over a
certificate. Absent, stale and unrecognised are three different answers, and
none of them is zero.

**Authority.** `AttestationTable::status_of`, `RecognisedMethods`,
`JudgedAttestation::content_mismatch`.

| | |
|---|---|
| positive | `an_attestation_that_still_describes_the_operand_is_bound` — `…/representation_attestations/tests_tuple.rs` |
| positive | `the_same_attestation_is_usable_or_not_purely_by_recognition` — `…/representation_attestations/tests_recognition.rs` |
| positive | `a_tampered_payload_is_refused_and_the_same_one_untampered_is_not` — `…/representation_attestations/tests_digest.rs` |
| negative | `unrecognised_is_not_absent_and_not_stale` — `…/representation_attestations/tests_recognition.rs` |
| negative | `a_bound_attestation_hands_over_nothing_until_its_bytes_are_checked` — `…/representation_attestations/tests_digest.rs` |
| negative | `checking_bytes_cannot_promote_a_status_that_already_failed` — `…/representation_attestations/tests_digest.rs` |
| negative | `a_stale_status_still_names_the_measurement_and_never_implies_zero` — `…/representation_attestations/tests_tuple.rs` |
| qualified by | PR #447, `202ffb7b` |

---

## 7. Auxiliary-certificate composition and floor enforcement

**Invariant.** The bound a floor judges is the **derived** one: the attested
claim where a measurement was verified, the codec's declaration otherwise,
composed with the certificates of the dependency extents actually selected.
Selection is phased — metadata admission, then payload verification, then
derivation — and **no plan is built from `Bound` evidence and corrected
afterwards.**

The floors ask two different questions and neither answers the other:

* `TerminalExtent` — structural, the complete stored representation, **no
  source-fidelity claim**. The default.
* `CertifiedExact` — a verified compatible certificate with radius `0.0`.
* `Within(r)` — a verified composed certificate at or under `r`.

Evidence floors are enforced on the **initially selected extent**;
reselection is enforced at the point of choice by `shallowest_saving`, which
filters candidates by the same floor.

| | |
|---|---|
| positive | `the_floor_judges_the_composed_bound_and_not_the_codecs_declared_one` — `…/exec/tests/composed_floor/mod.rs` |
| positive | `a_verified_measurement_replaces_the_declared_bound_and_reaches_the_floor` — `…/exec/tests/composed_floor/mod.rs` |
| positive | `certified_exact_refuses_a_terminal_extent_that_certifies_nothing` — `…/exec/tests/composed_floor/floor_vocabulary.rs` |
| positive | `certified_exact_admits_only_a_zero_radius` — `…/exec/tests/composed_floor/floor_vocabulary.rs` (any positive radius is not exact, however small) |
| positive | `within_requires_a_certificate_at_every_depth` — `…/exec/tests/composed_floor/floor_vocabulary.rs` |
| negative | `a_foreign_metric_satisfies_neither_evidence_floor` — `…/exec/tests/composed_floor/floor_vocabulary.rs` |
| default | `the_default_floor_is_the_structural_one` — `…/exec/tests/composed_floor/floor_vocabulary.rs` |
| over-promise stated as fact | `the_declared_bound_alone_would_have_admitted_both` — `…/exec/tests/composed_floor/mod.rs` |
| negative | `an_unrecognised_measurement_leaves_the_declared_bound_standing` — `…/exec/tests/composed_floor/mod.rs` |
| negative | `certified_exact_refuses_an_unbounded_quantised_plan` — `…/exec/tests/composed_floor/quantised_refusal.rs` |
| negative | `within_refuses_a_quantised_plan_at_any_bound` — `…/exec/tests/composed_floor/quantised_refusal.rs` |
| boundary | `terminal_extent_plans_a_quantised_model_without_claiming_anything_about_it` — `…/exec/tests/composed_floor/quantised_refusal.rs` |
| mutant | remove the composition loop in `VerifiedEvidence::derive` (`…/exec/attested_fidelity.rs`) → every fidelity arm above fails |
| mutant | remove the `enforce_floor` call in `select_realizations_within` (`…/exec/prepared.rs`) → both quantised refusals fail while the structural arm stays green |
| qualified by | PR #441 (`85b46061`), PR #447 (`202ffb7b`) |

---

## 8. Provider-identity carriage and invalidation

**Invariant.** A prepared image records the provider identity each pin
resolved to, on **both planes**: the codec that says what the stored bytes
are, and the lowering provider that qualified the realization. Losing
either, or substituting one at a revision that means something different,
**invalidates the preparation by name** rather than falling back — and each
plane refuses in its own vocabulary, so a refusal says which authority
moved. An out-of-tree provider reaches selection through registration
alone: no core file names it.

**Authority.** `PreparedOperands::ensure_providers_in`, `CodecIdentity`;
`PreparedOperands::ensure_lowerings_in`, `LoweringIdentity`
(LOWERING-PLUGIN-1, L4).

| | |
|---|---|
| positive | `an_external_representation_resolves_an_external_dependency_and_decodes` — `crates/larql-vindex/tests/external_auxiliary_provider.rs` |
| positive | `the_external_representation_and_its_dependency_reach_the_plan` — `crates/larql-vindex/tests/external_attested_provider/main.rs` |
| negative | `losing_or_substituting_either_provider_invalidates_preparation` — `crates/larql-vindex/tests/external_attested_provider/main.rs` |
| negative | `losing_the_dependencys_provider_invalidates_the_image_by_name` — `crates/larql-vindex/tests/external_auxiliary_provider.rs` |
| negative | `a_provider_that_disappears_invalidates_the_preparation_rather_than_falling_back` — `…/exec/tests/accounting.rs` |
| admission | `a_pack_under_the_external_identity_is_admitted_through_the_registry_the_store_opens_with` — `crates/larql-vindex/tests/external_codec_provider.rs` (refused by the built-in registry naming every family it knows; admitted by the registry the store is opened with; re-pointing re-runs the admission) |
| genericity | `no_external_identity_or_role_appears_in_production_larql` — `crates/larql-vindex/tests/external_attested_provider/genericity.rs` |
| scan control | `the_scan_would_have_found_an_identity_that_was_there` — same file (a shipped label *is* found by the same walk, so an empty result means "nothing there", not "nothing looked at") |
| lowering positive | `every_pin_names_both_authorities_and_the_two_move_independently` — `…/exec/tests/lowering_pin.rs` (the lowering identity changes while the codec identity does not, on the same fixture) |
| lowering negative | `only_the_provider_that_pinned_the_image_keeps_it_valid` — same file (gone, present at another revision, and a registry full of other providers: all invalid, none a fallback) |
| plane separation | `each_plane_refuses_on_its_own_authority` — same file (codec valid / lowering moved and the reverse; neither refusal speaks the other's vocabulary) |
| carriage | `a_serialized_and_reloaded_pin_retains_both_authorities` — same file (`PinnedAuthorities` written down and read back yields the same verdicts) |
| execution seam | `a_pin_is_executed_by_the_provider_that_decided_it_or_not_at_all` — same file (the check available where no registry is: the executing provider is the one that pinned) |
| configuration control | `one_identity_over_two_configurations_prepares_differently_and_executes_one_image_identically` — same file (one identity, two format tables: two preparations, but either instance means the same thing by a given one — why configuration is not a second authority) |
| production path | `a_served_model_refuses_when_the_provider_that_pinned_it_is_gone` — `crates/larql-inference/src/vindex3/tests/mod.rs` (the served model, re-pointed at an authority without its provider, refuses at session and at prefill) |
| qualified by | PR #441 (`85b46061`), PR #447 (`202ffb7b`) |

---

## 9. Exact preparation-read accounting

**Invariant.** Preparation accounting reports bytes **actually read**,
including attestation verification, counted according to **physical reads** —
not metadata lengths, and not call counts.

The frozen relationship, not merely the existence of a figure:

```text
read_to_prepare
  =
representation_materialisation
+ auxiliary_resolution
+ attestation_verification
```

And the physical semantics:

* counts **successful bytes actually read**, not metadata lengths or calls;
* `map_region` contributes **zero** — binding is not reading;
* verification is attributed **per operand**, and aggregates on the same
  once-per-operand rule as the stored footprint;
* **two independent verification reads count twice**, because that is what
  production currently does: `load_raw` opens, seeks and copies per
  candidate and nothing shares that materialisation. Counting once would
  claim an optimisation the code has not made;
* verification and a later decode are **separate materialisations** and both
  count;
* **the aggregate is derived from the breakdown, never separately
  accumulated. A second accumulator is a second answer.**

**Authority.** `PrepareReads`, `ResourceLedger::aggregate`,
`OperandStore::bytes_read`.

| | |
|---|---|
| the equation | `assert_total_holds` — `…/exec/tests/composed_floor/prepare_accounting.rs` (asserted by every arm below) |
| positive | `a_verified_operand_contributes_exactly_the_bytes_that_were_hashed` — same file (ledger **equals** the store's observed physical reads) |
| positive | `each_cause_lands_in_its_own_bucket` — same file |
| zero-read | `a_container_that_attests_nothing_verifies_nothing` — same file |
| zero-read | `an_unrecognised_authority_costs_no_payload_read` — same file |
| zero-read | `a_stale_tuple_is_refused_before_any_payload_read` — same file |
| counted twice | `two_attestations_over_one_operand_are_read_and_counted_twice` — same file |
| materialisations | `verification_and_a_later_decode_are_not_the_same_materialisation` — same file (`#[serial]`: it decodes, and decoding stages through a process-global arena other tests assert on) |
| isolation | `verification_changes_preparation_reads_and_no_other_resource` — same file (stored footprint, residency, staging peak, per-token touch and device all unmoved) |
| mutant | drop the `attestation_verification` term in `ResourceLedger::aggregate` (`…/exec/accounting.rs`) → four arms fail; the three zero-read arms stay green, which is the correct signature |
| qualified by | PR #457 |

---

## Consumer closure — evidence, not a tenth contract

The nine contracts say what a codec declares. These witnesses say that
nothing in the executor is reachable *around* those declarations — the
property "every codec uses the registry" stated as tests. They are listed
so a kernel or a loader path added without a declaration goes red here.

| | |
|---|---|
| closure | `every_direct_kernel_over_stored_bytes_is_declared_by_a_registered_codec` — `…/codec/tests/closure.rs` |
| control | `the_fp8_kernel_is_declared_by_exactly_the_fp8_codec` — `…/codec/tests/closure.rs` |
| production dependant | `selection_pins_the_fp8_kernel_and_retains_its_grid` — `…/exec/tests/fp8_carriage.rs` (the first `Retained` dependency lifetime in the tree) |
| production dependant | `the_canonical_decode_is_the_reference_dequantisation_bit_for_bit` — `…/exec/tests/fp8_carriage.rs` |
| declared, not spelled | `the_encoder_declares_the_scale_grid_as_the_codes_dependency` — `…/exec/tests/fp8_carriage.rs` |

---

## Not frozen in v1

> **Lowering-target / backend realization policy is NOT frozen in v1.**

The existence of this freeze does not settle Metal or direct-kernel
selection, lowering targets, GGUF or other export paths, remote execution
services, or any future optimised realization. None of those has been
qualified here and none is bound by this index.

The two planes answer different questions and are kept apart deliberately:

* **Representation plugins** answer *"what is stored, and what guarantees
  does it carry?"* — that is the plane this document freezes.
* **Lowering plugins** answer *"how is a planned graph realized for CPU,
  Metal, MLX, GGUF export, or a remote FFN service?"* — untouched, and to be
  opened only after this freeze lands.

Related debts that are named and unpaid, and which this freeze does not
resolve: shared dependency materialisation (D1's neighbour — decoding a
dependency once per representation instance rather than per owner), and
realization-error certificates, which become necessary once accelerated
kernels make fidelity claims of their own.
