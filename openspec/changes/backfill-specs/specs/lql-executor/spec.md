## ADDED Requirements

### Requirement: Session lifecycle and statement dispatch

`larql_lql::executor::Session` SHALL hold a `Backend` (one of
`None`, `Vindex`, `Weight`, or `Remote`), an optional active
`PatchRecording`, an `auto_patch` flag, accumulated installed-edge
state, an LSM `epoch` counter, and `mutations_since_minor /
_major` counters. `Session::execute` MUST dispatch every supported
`Statement` variant to its per-verb implementation, MUST
auto-start an anonymous patch session on the first
INSERT/DELETE/UPDATE when none is active, and MUST advance the
epoch on every mutation. When the active backend is remote the
session MUST forward to `execute_remote`, which only allows the
documented remote-supported verbs and rejects unsupported ones with
a clear "TRACE requires a local vindex" hint.

#### Scenario: Statements without a backend produce a NoBackend error
- **WHEN** STATS / WALK / DESCRIBE / SELECT / EXPLAIN / SHOW RELATIONS / SHOW LAYERS / SHOW FEATURES / SHOW ENTITIES / TRACE / COMPACT MINOR / COMPACT MAJOR / SHOW COMPACT STATUS / REMOVE PATCH / INFER are executed against a `Backend::None` session
- **THEN** each call MUST return `LqlError::NoBackend`-class error rather than panic
<!-- test: larql_lql::executor::tests::no_backend_stats -->
<!-- test: larql_lql::executor::tests::no_backend_walk -->
<!-- test: larql_lql::executor::tests::no_backend_describe -->
<!-- test: larql_lql::executor::tests::no_backend_select -->
<!-- test: larql_lql::executor::tests::no_backend_explain -->
<!-- test: larql_lql::executor::tests::no_backend_show_relations -->
<!-- test: larql_lql::executor::tests::no_backend_show_layers -->
<!-- test: larql_lql::executor::tests::no_backend_show_features -->
<!-- test: larql_lql::executor::tests::no_backend_show_entities -->
<!-- test: larql_lql::executor::tests::no_backend_trace -->
<!-- test: larql_lql::executor::tests::no_backend_compact_minor -->
<!-- test: larql_lql::executor::tests::no_backend_compact_major -->
<!-- test: larql_lql::executor::tests::no_backend_show_compact_status -->
<!-- test: larql_lql::executor::tests::no_backend_remove_patch -->
<!-- test: larql_lql::executor::tests::infer_no_backend -->
<!-- test: larql_lql::executor::tests::insert_no_backend -->
<!-- test: larql_lql::executor::tests::delete_no_backend -->
<!-- test: larql_lql::executor::tests::update_no_backend -->
<!-- test: larql_lql::executor::tests::compile_no_backend -->

#### Scenario: SHOW MODELS works without a backend
- **WHEN** `SHOW MODELS;` is executed on a `Backend::None` session
- **THEN** the call MUST succeed and emit a non-empty model list
<!-- test: larql_lql::executor::tests::show_models_no_crash -->

#### Scenario: Pipe statement composes outputs and propagates errors
- **WHEN** `Statement::Pipe { left, right }` is executed
- **THEN** the executor MUST run `left`, append `right`'s output, and propagate any error from either side
<!-- test: larql_lql::executor::tests::pipe_error_propagates -->
<!-- test: larql_lql::executor::tests::pipe_propagates_no_backend_error -->
<!-- test: larql_lql::executor::tests::pipe_concatenates_both_sides_output -->

### Requirement: USE switches between vindex, model, and remote backends

`exec_use` SHALL load a vindex from disk (`USE "path.vindex"`),
load a HuggingFace model (`USE MODEL "id" [AUTO_EXTRACT]`), or set
the remote target (`USE REMOTE "url"`). Failures MUST surface as
`LqlError::Execution` with the offending path/id and MUST NOT
mutate session state. Successful loads MUST replace any prior
`Backend` and refresh the patch overlay so subsequent statements
see the new index.

#### Scenario: USE on a missing vindex/model errors cleanly
- **WHEN** `USE "/nonexistent.vindex"` and `USE MODEL "nonexistent/id"` are executed
- **THEN** each MUST return an `LqlError` and leave the session backend at `None`
<!-- test: larql_lql::executor::tests::use_nonexistent_vindex -->
<!-- test: larql_lql::executor::tests::use_model_fails_on_nonexistent -->
<!-- test: larql_lql::executor::tests::use_model_auto_extract_parses -->

#### Scenario: USE on a synthetic vindex installs a Vindex backend
- **WHEN** a synthetic vindex directory is loaded via `USE "<path>"`
- **THEN** subsequent `Session::patched_overlay_mut()` MUST return `Some` and queries that require a vindex MUST succeed
<!-- test: larql_lql::executor::tests::use_synthetic_vindex_loads -->
<!-- test: larql_lql::executor::tests::patched_overlay_mut_returns_some_for_vindex_backend -->
<!-- test: larql_lql::executor::tests::patched_overlay_mut_returns_none_for_no_backend -->

#### Scenario: Weight-only backend rejects vindex-required verbs
- **WHEN** WALK / DESCRIBE / SELECT / EXPLAIN / INSERT / SHOW RELATIONS / COMPILE CURRENT are executed on a session whose backend is weights-only (`Backend::Weight`)
- **THEN** the executor MUST return an error indicating that a vindex is required, while STATS and SHOW MODELS still succeed
<!-- test: larql_lql::executor::tests::weight_backend_stats -->
<!-- test: larql_lql::executor::tests::weight_backend_walk_requires_vindex -->
<!-- test: larql_lql::executor::tests::weight_backend_describe_requires_vindex -->
<!-- test: larql_lql::executor::tests::weight_backend_select_requires_vindex -->
<!-- test: larql_lql::executor::tests::weight_backend_explain_walk_requires_vindex -->
<!-- test: larql_lql::executor::tests::weight_backend_insert_requires_vindex -->
<!-- test: larql_lql::executor::tests::weight_backend_show_relations_requires_vindex -->
<!-- test: larql_lql::executor::tests::weight_backend_compile_current_requires_vindex -->
<!-- test: larql_lql::executor::tests::weight_backend_show_models_works -->

### Requirement: EXTRACT, COMPILE, DIFF lifecycle execution

`exec_extract` SHALL fail cleanly when the requested model is
unavailable. `exec_compile` SHALL support both `INTO MODEL`
(requires baked weights) and `INTO VINDEX` (overlay-bake) targets,
honour the `ON CONFLICT` policy when multiple patches touch the
same `(layer, feature)` slot, refresh recorded patch ops to
preserve the latest overlay vectors, and accept either an active
backend or a supplied path source. `exec_diff` SHALL fail when
either operand vindex does not exist.

#### Scenario: EXTRACT against a missing model errors
- **WHEN** `EXTRACT MODEL "nonexistent/id" INTO "out.vindex"` is executed
- **THEN** the executor MUST return an `LqlError` and emit no vindex on disk
<!-- test: larql_lql::executor::tests::extract_fails_on_nonexistent_model -->

#### Scenario: COMPILE INTO VINDEX bakes overlays without active backend
- **WHEN** `COMPILE "src.vindex" INTO VINDEX "out.vindex"` is executed with a fresh path source and no patches
- **THEN** the call MUST succeed using the supplied source, and emit a clean output vindex
<!-- test: larql_lql::executor::tests::compile_into_vindex_no_patches_succeeds -->
<!-- test: larql_lql::executor::tests::compile_path_into_vindex_uses_supplied_source_without_active_backend -->
<!-- test: larql_lql::executor::tests::compile_into_vindex_with_down_overrides_bakes_them -->

#### Scenario: COMPILE INTO MODEL requires model weights and reports source requirements
- **WHEN** `COMPILE "src.vindex" INTO MODEL "out/" FORMAT safetensors` is executed without baked weights
- **THEN** the executor MUST return an error that names the missing requirement
<!-- test: larql_lql::executor::tests::compile_into_model_requires_model_weights -->
<!-- test: larql_lql::executor::tests::compile_path_into_model_reports_supplied_source_requirements -->

#### Scenario: COMPILE INTO VINDEX honours ON CONFLICT policies
- **WHEN** patches collide on the same slot during `COMPILE … ON CONFLICT FAIL` and `… ON CONFLICT LAST_WINS`
- **THEN** the FAIL form MUST detect the collision and abort, while LAST_WINS MUST succeed and keep the last patch's overlay
<!-- test: larql_lql::executor::tests::compile_on_conflict_fail_detects_collision -->
<!-- test: larql_lql::executor::tests::compile_on_conflict_last_wins_succeeds -->

#### Scenario: COMPILE refreshes recorded patch ops to latest overlay
- **WHEN** an in-memory overlay is mutated and re-compiled
- **THEN** the recorded patch ops MUST persist the latest overlay vectors instead of stale snapshots
<!-- test: larql_lql::executor::tests::refresh_recorded_patch_ops_for_slots_persists_latest_overlay_vectors -->

#### Scenario: COMPILE bake helpers cover f16/f32 dtypes and shape validation
- **WHEN** `patch_down_weights` is called with `f32`, `f16`, multiple layers, mismatched shapes, unrecognised dtype, or a missing source
- **THEN** valid inputs MUST write the correct columns and invalid inputs MUST return an error
<!-- test: larql_lql::executor::lifecycle::compile::bake::tests::patch_down_weights_f32_writes_correct_columns -->
<!-- test: larql_lql::executor::lifecycle::compile::bake::tests::patch_down_weights_f16_writes_correct_columns -->
<!-- test: larql_lql::executor::lifecycle::compile::bake::tests::patch_down_weights_multiple_layers_and_features -->
<!-- test: larql_lql::executor::lifecycle::compile::bake::tests::patch_down_weights_rejects_wrong_shape -->
<!-- test: larql_lql::executor::lifecycle::compile::bake::tests::patch_down_weights_rejects_unrecognised_dtype_size -->
<!-- test: larql_lql::executor::lifecycle::compile::bake::tests::patch_down_weights_missing_source_errors -->

#### Scenario: COMPILE INTO VINDEX collision detection isolates per-patch slots
- **WHEN** `compute_collisions` is queried over patches
- **THEN** it MUST return empty when each slot is unique, surface collisions across patches, and ignore repeats within a single patch
<!-- test: larql_lql::executor::lifecycle::compile::into_vindex::tests::collisions_empty_when_each_slot_unique -->
<!-- test: larql_lql::executor::lifecycle::compile::into_vindex::tests::collisions_detect_same_slot_in_two_patches -->
<!-- test: larql_lql::executor::lifecycle::compile::into_vindex::tests::collisions_ignore_repeats_within_one_patch -->

#### Scenario: DIFF on a missing vindex errors
- **WHEN** `DIFF "nonexistent.vindex" CURRENT` is executed
- **THEN** the executor MUST return an `LqlError` instead of panicking
<!-- test: larql_lql::executor::tests::diff_nonexistent_vindex -->

### Requirement: STATS, SHOW, INFER reachable execution

The STATS, SHOW, and INFER paths SHALL produce stable text output for valid sessions, format numbers and bytes with appropriate units, and reject INFER on sessions without prerequisites. `exec_stats`, the `exec_show_*` family, and the helper utilities MUST cover the small / KB / MB / GB / thousand / million formatter bands so the user-facing units stay consistent.

#### Scenario: Stats and SHOW ENTITIES scan a synthetic vindex
- **WHEN** STATS or `SHOW ENTITIES` is executed on a vindex session
- **THEN** the call MUST succeed and produce non-empty output
<!-- test: larql_lql::executor::tests::weight_backend_stats -->
<!-- test: larql_lql::executor::tests::show_entities_scans_synthetic_vindex -->

#### Scenario: Number and byte formatters cover unit bands
- **WHEN** the formatters are exercised across small, thousand, million, KB, MB, and GB inputs
- **THEN** each result MUST use the appropriate suffix
<!-- test: larql_lql::executor::tests::format_number_small -->
<!-- test: larql_lql::executor::tests::format_number_thousands -->
<!-- test: larql_lql::executor::tests::format_number_millions -->
<!-- test: larql_lql::executor::tests::format_bytes_small -->
<!-- test: larql_lql::executor::tests::format_bytes_kb -->
<!-- test: larql_lql::executor::tests::format_bytes_mb -->
<!-- test: larql_lql::executor::tests::format_bytes_gb -->

#### Scenario: KNN-override summary helpers name source and top-1
- **WHEN** `remote_knn_override_summary` is rendered
- **THEN** the output MUST include the post-logits source and the model's top-1 token
<!-- test: larql_lql::executor::helpers::tests::knn_override_summary_names_post_logits_source_and_model_top1 -->

### Requirement: Token readability filtering for WALK/INFER

The executor SHALL filter raw tokens through a readability check
before they reach a user-facing WALK/INFER output: stop-words,
short tokens, and code-shaped tokens MUST be rejected, while
content-bearing tokens MUST pass through.

#### Scenario: Readable / unreadable / content tokens are gated
- **WHEN** the readability filter is exercised on representative tokens
- **THEN** readable content tokens MUST pass; stop-words, short, and code-shaped tokens MUST be rejected
<!-- test: larql_lql::executor::tests::readable_tokens -->
<!-- test: larql_lql::executor::tests::unreadable_tokens -->
<!-- test: larql_lql::executor::tests::content_tokens_pass -->
<!-- test: larql_lql::executor::tests::stop_words_rejected -->
<!-- test: larql_lql::executor::tests::short_tokens_rejected -->
<!-- test: larql_lql::executor::tests::code_tokens_rejected -->

### Requirement: INSERT (KNN and Compose), DELETE, UPDATE, MERGE execution

`exec_insert` SHALL default to `InsertMode::Knn` and store the
fact in the patched overlay's `KnnStore`, MUST honour `AT LAYER`
hints, MUST persist a fact across COMPILE+USE round-trips, MUST
fall back to embedding-key install for q4k vindices without
weights, and MUST emit `Inserted ... KNN store` confirmation
messages including the entry count. `exec_delete` and `exec_update`
SHALL refuse relation-label-based filters when no labels exist and
MUST not mutate state on rejection. `exec_merge` MUST error
cleanly on a non-existent source. Per-fact tracking SHALL count
inserts only and MUST de-duplicate facts repeated across patches.

#### Scenario: INSERT (default KNN mode) populates the KnnStore
- **WHEN** `INSERT INTO EDGES (entity, relation, target) VALUES (…)` is executed
- **THEN** the output MUST contain `Inserted` and `KNN store` confirmation, and a follow-up `entries_for_entity` lookup MUST find the fact
<!-- test: larql_lql::executor::tests::knn_store_insert_populates_store -->
<!-- test: larql_lql::executor::tests::knn_store_describe_shows_inserted_edges -->
<!-- test: larql_lql::executor::tests::patched_overlay_mut_round_trip_via_insert_feature -->

#### Scenario: INSERT works for multiple facts and AT LAYER hints
- **WHEN** several KNN INSERTs are executed and a final INSERT carries `AT LAYER 0`
- **THEN** the running entry count MUST track the inserts and the AT LAYER hint MUST land at the requested layer
<!-- test: larql_lql::executor::tests::knn_store_insert_multiple_facts -->
<!-- test: larql_lql::executor::tests::knn_store_insert_at_layer_hint -->

#### Scenario: INSERT survives COMPILE INTO VINDEX + USE round-trip
- **WHEN** an INSERT is followed by `COMPILE CURRENT INTO VINDEX "out"` and a fresh `USE "out"`
- **THEN** the KnnStore MUST contain the original fact and the patched overlay MUST report it via `entries_for_entity`
<!-- test: larql_lql::executor::tests::knn_store_compile_saves_and_loads -->

#### Scenario: INSERT on q4k vindex without weights uses embedding-key fallback
- **WHEN** an INSERT is executed against a q4k vindex with no model weights loaded
- **THEN** the embedding-key fallback path MUST run and emit a fact instead of returning a quantization error
<!-- test: larql_lql::executor::tests::knn_insert_q4k_flagged_no_weights_uses_embedding_fallback -->

#### Scenario: KNN PatchOp serialises round-trip via JSON
- **WHEN** `PatchOp::InsertKnn` and `PatchOp::DeleteKnn` are serialised and deserialised
- **THEN** the resulting JSON tags MUST be `insert_knn`/`delete_knn` and the round-tripped variants MUST preserve all fields
<!-- test: larql_lql::executor::tests::knn_store_patch_op_serialization -->
<!-- test: larql_lql::executor::tests::knn_store_delete_knn_patch_op -->

#### Scenario: INSERT compose-mode helpers (unit/median/refine) behave deterministically
- **WHEN** the compose helpers are exercised
- **THEN** unit-vector normalisation MUST round to length one (with passthrough for the zero vector and idempotent for unit input), the median selector MUST sort in place and use defaults for empty slices, and `should_refine` MUST gate based on input cardinality and decoy availability
<!-- test: larql_lql::executor::mutation::insert::compose::tests::unit_vector_normalises_to_length_one -->
<!-- test: larql_lql::executor::mutation::insert::compose::tests::unit_vector_passthrough_on_zero -->
<!-- test: larql_lql::executor::mutation::insert::compose::tests::unit_vector_handles_already_unit -->
<!-- test: larql_lql::executor::mutation::insert::compose::tests::median_or_picks_middle -->
<!-- test: larql_lql::executor::mutation::insert::compose::tests::median_or_uses_default_when_empty -->
<!-- test: larql_lql::executor::mutation::insert::compose::tests::median_or_handles_single_element -->
<!-- test: larql_lql::executor::mutation::insert::compose::tests::median_or_sorts_input_in_place -->
<!-- test: larql_lql::executor::mutation::insert::compose::tests::install_math_produces_competing_activation -->
<!-- test: larql_lql::executor::mutation::insert::compose::tests::should_refine_empty_inputs_never_runs -->
<!-- test: larql_lql::executor::mutation::insert::compose::tests::should_refine_single_input_needs_a_decoy -->
<!-- test: larql_lql::executor::mutation::insert::compose::tests::should_refine_two_plus_inputs_runs_without_decoys -->
<!-- test: larql_lql::executor::mutation::insert::compose::tests::should_refine_combined_sets_always_run -->

#### Scenario: INSERT capture decoys are unique on 3-word prefix
- **WHEN** the insert capture pipeline assembles canonical decoys
- **THEN** every decoy prompt MUST have a distinct first-three-word prefix
<!-- test: larql_lql::executor::mutation::insert::capture::tests::canonical_decoys_have_unique_3word_prefixes -->

#### Scenario: DELETE matches and reports
- **WHEN** `DELETE FROM EDGES WHERE layer = … AND feature = …` is executed against a populated vindex
- **THEN** the call MUST succeed and report removed entries; a no-match form MUST emit a "no matches" message
<!-- test: larql_lql::executor::tests::delete_by_layer_and_feature_succeeds -->
<!-- test: larql_lql::executor::tests::delete_no_matches_returns_message -->
<!-- test: larql_lql::executor::tests::knn_store_delete_removes_entries -->

#### Scenario: DELETE / UPDATE relation filters error before mutating
- **WHEN** `DELETE` / `UPDATE` use a relation-label filter and the loaded vindex has no labels
- **THEN** the executor MUST error before mutating state
<!-- test: larql_lql::executor::tests::delete_relation_filter_without_labels_errors_before_mutating -->
<!-- test: larql_lql::executor::tests::update_relation_filter_without_labels_errors_before_mutating -->

#### Scenario: UPDATE feature target succeeds
- **WHEN** `UPDATE EDGES SET target = "…" WHERE entity = "…" AND relation = "…"` is executed against a populated vindex
- **THEN** the call MUST succeed and persist the new target through the patch overlay
<!-- test: larql_lql::executor::tests::update_feature_target_succeeds -->

#### Scenario: MERGE on a missing source errors cleanly
- **WHEN** `MERGE "nonexistent.vindex"` is executed
- **THEN** the call MUST return an `LqlError` without crashing
<!-- test: larql_lql::executor::tests::merge_nonexistent_source -->
<!-- test: larql_lql::executor::tests::merge_nonexistent_source_errors_cleanly -->

#### Scenario: MEMIT fact tracking counts inserts and deduplicates across patches
- **WHEN** the per-fact accountant is fed inserts (with relation-template variants) and the same fact across patches
- **THEN** the count MUST reflect inserts only and duplicates across patches MUST be collapsed
<!-- test: larql_lql::executor::tests::memit_facts_count_inserts_only -->
<!-- test: larql_lql::executor::tests::memit_facts_deduplicate_across_patches -->
<!-- test: larql_lql::executor::tests::memit_fact_struct -->
<!-- test: larql_lql::executor::tests::relation_template_simple -->
<!-- test: larql_lql::executor::tests::relation_template_multi_word -->
<!-- test: larql_lql::executor::tests::relation_template_hyphenated_produces_double_of -->

#### Scenario: MEMIT store is created lazily and persisted
- **WHEN** `memit_store_mut` is called on a session without a backend, on a fresh vindex, or after install cycles
- **THEN** it MUST be unavailable on no-backend, return an empty store on a fresh vindex, and persist added cycles otherwise
<!-- test: larql_lql::executor::tests::memit_store_mut_unavailable_without_backend -->
<!-- test: larql_lql::executor::tests::memit_store_mut_returns_empty_store_on_fresh_vindex -->
<!-- test: larql_lql::executor::tests::memit_store_persists_added_cycles -->

### Requirement: Patch session lifecycle and overlay management

`Session::ensure_patch_session` SHALL auto-start an anonymous
recording on the first mutation if none is active. `BEGIN PATCH`
MUST upgrade an in-flight anonymous session by adopting its
operations under the new path. `SAVE PATCH` MUST require an active
non-anonymous session, write the patch to disk, and return
insert/update/delete counts. `APPLY PATCH` MUST require a
`Backend::Vindex`, `SHOW PATCHES` MUST list patches plus the
session's pending operations, and `REMOVE PATCH` MUST error when
the requested patch is unknown.

#### Scenario: BEGIN PATCH and SAVE PATCH succeed end-to-end
- **WHEN** `BEGIN PATCH "x.vlp"` followed by an INSERT and `SAVE PATCH` are executed
- **THEN** a session MUST start, the insert MUST attach to the recording, and the file MUST be written to disk
<!-- test: larql_lql::executor::tests::explicit_begin_patch_starts_session -->
<!-- test: larql_lql::executor::tests::save_patch_writes_file_to_disk -->

#### Scenario: First mutation auto-starts an anonymous patch session
- **WHEN** an INSERT/DELETE/UPDATE is executed without a prior BEGIN PATCH
- **THEN** an anonymous patch session MUST start and the executor MUST emit the auto-patch banner
<!-- test: larql_lql::executor::tests::auto_patch_session_starts_on_first_mutation -->

#### Scenario: SHOW PATCHES handles no-patches and unknown REMOVE PATCH
- **WHEN** `SHOW PATCHES` is run on a vindex with no patches, or `REMOVE PATCH "unknown"` is run
- **THEN** SHOW PATCHES MUST emit a "no patches" message and REMOVE PATCH MUST return an error
<!-- test: larql_lql::executor::tests::show_patches_with_no_patches_returns_message -->
<!-- test: larql_lql::executor::tests::remove_patch_unknown_errors_cleanly -->

### Requirement: TRACE, REBALANCE, COMPACT execution

`exec_trace` SHALL run only when the loaded vindex provides the
weights TRACE needs and SHALL emit a clear weights-required hint
when the vindex is browse-only or quantised below the trace
threshold. `exec_rebalance` SHALL be a no-op when there are no
compose installs to rebalance and on a `Backend::None` session.
`exec_compact_minor` and `exec_compact_major` SHALL produce
informational messages when L0 is empty, and `SHOW COMPACT STATUS`
SHALL report empty-tier counts on a fresh vindex.

#### Scenario: TRACE on q4k vindex returns a clear error
- **WHEN** TRACE is executed on a q4k vindex
- **THEN** the executor MUST return a clear quantisation-related error
<!-- test: larql_lql::executor::tests::trace_on_q4k_vindex_returns_clear_error -->

#### Scenario: TRACE on a browse-only vindex hints at WITH WEIGHTS
- **WHEN** TRACE is executed on a vindex extracted with `Browse` level only
- **THEN** the error message MUST instruct the user to re-EXTRACT with the inference / all level
<!-- test: larql_lql::executor::tests::trace_on_browse_only_vindex_errors_with_weights_hint -->

#### Scenario: REBALANCE without backend or installs is a no-op
- **WHEN** REBALANCE is executed on a fresh session or one with no compose installs
- **THEN** the call MUST succeed without modifying state and MUST emit a "nothing to rebalance" message
<!-- test: larql_lql::executor::tests::rebalance_without_backend_is_noop -->
<!-- test: larql_lql::executor::tests::rebalance_without_compose_installs_is_noop -->

#### Scenario: COMPACT MINOR / SHOW COMPACT STATUS report empty tiers
- **WHEN** `COMPACT MINOR` is run on an empty L0 and `SHOW COMPACT STATUS` is run on a fresh vindex
- **THEN** each MUST emit informational text about empty tiers
<!-- test: larql_lql::executor::tests::compact_minor_on_empty_l0_returns_message -->
<!-- test: larql_lql::executor::tests::show_compact_status_reports_empty_tiers -->

### Requirement: Test suite coverage

The capability's behavior SHALL be exercised by every test in the test modules listed below. New tests added to those modules SHALL be considered part of this capability's contract by default.

#### Scenario: Workspace tests bound to this capability
- **WHEN** the trace tool resolves the wildcard module references for this capability
- **THEN** every backing test SHALL resolve to a real `#[test]` function in the workspace
<!-- test: larql_lql::executor::tests::**::* -->
<!-- test: larql_lql::executor::helpers::tests::**::* -->
<!-- test: larql_lql::executor::lifecycle::compile::bake::tests::**::* -->
<!-- test: larql_lql::executor::lifecycle::compile::into_vindex::tests::**::* -->
<!-- test: larql_lql::executor::mutation::insert::capture::tests::**::* -->
<!-- test: larql_lql::executor::mutation::insert::compose::**::* -->
