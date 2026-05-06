## ADDED Requirements

### Requirement: PatchedVindex overlay semantics

`PatchedVindex` SHALL wrap an immutable base `VectorIndex` plus a
mutation overlay so that callers see a transparently mutated view
without writing through to the base on disk. Inserts MUST be visible
to gate KNN, deletes MUST hide the underlying feature, and overlays
MUST stack so a later patch overrides an earlier one. The base SHALL
NEVER be mutated through the overlay.

#### Scenario: Patched vindex hides deleted features
- **WHEN** a delete patch is applied for a feature in the base vindex
- **THEN** the patched view SHALL no longer return that feature
<!-- test: larql_vindex::test_vindex::patched_vindex_delete_hides_feature -->

#### Scenario: Patched vindex inserts a new feature
- **WHEN** an insert patch adds a new feature
- **THEN** the patched view SHALL return that feature on lookup
<!-- test: larql_vindex::test_vindex::patched_vindex_insert_feature -->

#### Scenario: Patched vindex deletion through public API
- **WHEN** the public delete API is invoked on a patched vindex
- **THEN** subsequent lookups SHALL hide the deleted feature
<!-- test: larql_vindex::test_vindex::patched_vindex_delete_feature -->

#### Scenario: Patches stack with overrides taking precedence
- **WHEN** multiple patches mutate the same feature
- **THEN** the most recently applied patch SHALL win
<!-- test: larql_vindex::test_vindex::patched_vindex_later_patch_overrides_earlier -->

#### Scenario: Multiple patches accumulate
- **WHEN** several non-conflicting patches are applied in sequence
- **THEN** all of their effects SHALL be visible
<!-- test: larql_vindex::test_vindex::patch_multiple_patches_stack -->

#### Scenario: Empty patch leaves the vindex untouched
- **WHEN** a patch with no operations is applied
- **THEN** the patched view SHALL match the base
<!-- test: larql_vindex::test_vindex::patch_empty_operations -->

#### Scenario: Removing a patch rebuilds overrides correctly
- **WHEN** a previously applied patch is removed
- **THEN** the overlay SHALL be recomputed without the removed entries
<!-- test: larql_vindex::patch::overlay_apply::tests::remove_patch_rebuilds_overrides -->

#### Scenario: Removing an out-of-bounds patch is a no-op
- **WHEN** `remove_patch` is called with an out-of-range index
- **THEN** the call SHALL leave the overlay unchanged
<!-- test: larql_vindex::patch::overlay_apply::tests::remove_patch_out_of_bounds_is_noop -->

#### Scenario: Insert then delete cleans up the gate override
- **WHEN** a feature is inserted via patch and then immediately deleted
- **THEN** the gate override slot SHALL be cleared
<!-- test: larql_vindex::patch::overlay_apply::tests::insert_then_delete_removes_gate_override -->

#### Scenario: Insert with explicit gate vector populates the override slot
- **WHEN** an insert patch supplies a gate vector
- **THEN** the gate override SHALL be populated with that vector
<!-- test: larql_vindex::patch::overlay_apply::tests::apply_insert_with_gate_vector_populates_overrides_gate -->

#### Scenario: Update sets metadata only
- **WHEN** an update patch carries no vector payload
- **THEN** only feature metadata SHALL change
<!-- test: larql_vindex::patch::overlay_apply::tests::apply_update_sets_meta_only -->

### Requirement: VLP patch file format

The `.vlp` JSON patch file SHALL preserve `INSERT`, `UPDATE`, and
`DELETE` operations, MUST encode gate / up / down vectors as
base64-aligned floats, and SHALL load legacy patches that lack
up/down vectors. Vector decode MUST reject unaligned input rather than
truncate.

#### Scenario: Patch save and load round-trips
- **WHEN** a patch is saved to `.vlp` and reloaded
- **THEN** every operation SHALL match the original
<!-- test: larql_vindex::test_vindex::patch_save_and_load_round_trip -->

#### Scenario: Save/load preserves gate, up, and down vectors
- **WHEN** a patch carrying gate/up/down vectors is saved and reloaded
- **THEN** all three vectors SHALL match exactly
<!-- test: larql_vindex::patch::format::tests::save_load_round_trip_preserves_gate_up_down_vectors -->

#### Scenario: Loading a legacy patch without up/down succeeds
- **WHEN** a legacy `.vlp` file lacking up/down vectors is loaded
- **THEN** loading SHALL succeed and the missing vectors SHALL default to absent
<!-- test: larql_vindex::patch::format::tests::load_legacy_patch_without_up_down_vectors -->

#### Scenario: Loading a missing patch file errors
- **WHEN** `load_patch` is called on a non-existent path
- **THEN** it SHALL return a structured error
<!-- test: larql_vindex::patch::format::tests::load_missing_file_returns_error -->

#### Scenario: Base64 decoder rejects unaligned input
- **WHEN** an encoded vector blob is not aligned to four bytes
- **THEN** decoding SHALL return an error
<!-- test: larql_vindex::patch::format::tests::decode_rejects_unaligned_bytes -->

#### Scenario: Base64 vector encode/decode round-trips
- **WHEN** a multi-element float vector is base64 encoded and decoded
- **THEN** the result SHALL equal the original
<!-- test: larql_vindex::patch::format::tests::encode_decode_round_trip_multi_float -->

#### Scenario: Patch op keys distinguish insert/update/delete
- **WHEN** an `INSERT`, `UPDATE`, or `DELETE` op is queried for its identifier key
- **THEN** the keys SHALL be unique per operation type
<!-- test: larql_vindex::patch::format::tests::patch_op_key_insert -->

### Requirement: Save patch persists overlay without writing the base

`SAVE PATCH` SHALL serialise the current overlay to a `.vlp` file and
MUST NOT modify the underlying base vindex on disk. Subsequent reloads
of the base SHALL produce the same bytes as before the patch was
applied. Bake-down semantics MUST stay intact across the save / reload
cycle so a baked vindex's overrides are preserved.

#### Scenario: Bake-down preserves baked vectors
- **WHEN** a patched vindex is baked down
- **THEN** the baked overrides SHALL persist into the new vindex
<!-- test: larql_vindex::test_vindex::patched_vindex_bake_down -->

#### Scenario: Bake-down preserves baked vectors after additional patches
- **WHEN** additional patches are applied before bake-down
- **THEN** the resulting baked vindex SHALL incorporate all overrides
<!-- test: larql_vindex::test_vindex::patched_vindex_bake_down_preserves -->

#### Scenario: Patches removed from overlay are dropped
- **WHEN** `remove_patch` is invoked
- **THEN** the resulting overlay SHALL no longer carry that patch's effects
<!-- test: larql_vindex::test_vindex::patched_vindex_remove_patch -->

#### Scenario: Full lifecycle build → query → mutate → save → reload
- **WHEN** a vindex is built, queried, mutated, saved, and reloaded
- **THEN** every stage SHALL preserve its expected state
<!-- test: larql_vindex::test_vindex::full_lifecycle_build_query_mutate_save_reload -->

#### Scenario: Extract followed by mutate then reload preserves the mutation
- **WHEN** an extracted vindex is mutated and reloaded
- **THEN** the mutation SHALL still be present after reload
<!-- test: larql_vindex::test_vindex::extract_mutate_reload_verifies_mutation -->

#### Scenario: Patches survive bake/reload via extract path
- **WHEN** an extracted vindex with patches is baked down and reloaded
- **THEN** the patched state SHALL persist
<!-- test: larql_vindex::test_vindex::extract_with_patches_bake_down -->

### Requirement: L0 residual-key KNN store

A patched vindex SHALL maintain an L0 residual-key KNN store so that
inserted facts can be retrieved by associative query independently of
the base layer. The store MUST handle entity-based removal,
case-insensitive entity keys, and SHALL save/load losslessly.

#### Scenario: Add and length
- **WHEN** entries are added to the KNN store
- **THEN** the length SHALL increase accordingly
<!-- test: larql_vindex::patch::knn_store::tests::test_add_and_len -->

#### Scenario: Top-1 returns exact match when present
- **WHEN** a probe equals one stored key
- **THEN** the top-1 result SHALL be that entry
<!-- test: larql_vindex::patch::knn_store::tests::test_query_top1_exact_match -->

#### Scenario: Removing by entity is case-insensitive
- **WHEN** entries are removed by entity name with different casing
- **THEN** removal SHALL match regardless of case
<!-- test: larql_vindex::patch::knn_store::tests::test_remove_by_entity_case_insensitive -->

#### Scenario: KNN store save/load round-trips
- **WHEN** the KNN store is saved and reloaded
- **THEN** every entry SHALL match the original
<!-- test: larql_vindex::patch::knn_store::tests::test_save_load_roundtrip -->

#### Scenario: Insert-knn op routes to the KNN store
- **WHEN** an insert-knn patch is applied
- **THEN** the entry SHALL be added to the L0 residual KNN store
<!-- test: larql_vindex::patch::overlay_apply::tests::apply_insert_knn_adds_to_knn_store -->

#### Scenario: Delete-knn op removes from the store
- **WHEN** a delete-knn patch is applied
- **THEN** the matching entry SHALL be removed
<!-- test: larql_vindex::patch::overlay_apply::tests::apply_delete_knn_removes_from_knn_store -->

### Requirement: Refine pass for inserted gate vectors

The patch refine pass SHALL adjust an inserted gate vector so that it
maintains its norm against orthogonal peers, loses norm against
parallel peers, and removes overlap with decoy residuals. Cross-layer
facts MUST NOT interfere, and the refine pass SHALL pass arrays
through unchanged when there is nothing to refine.

#### Scenario: Empty input returns empty output
- **WHEN** the refine pass receives an empty input
- **THEN** it SHALL return an empty result without erroring
<!-- test: larql_vindex::patch::refine::tests::empty_input_returns_empty_result -->

#### Scenario: Orthogonal inputs retain their full norm
- **WHEN** the inserted vector is orthogonal to its peers
- **THEN** the refined vector SHALL retain its norm
<!-- test: larql_vindex::patch::refine::tests::orthogonal_inputs_retain_full_norm -->

#### Scenario: Parallel inputs lose norm
- **WHEN** the inserted vector is parallel to a peer
- **THEN** the refined vector SHALL lose norm
<!-- test: larql_vindex::patch::refine::tests::parallel_inputs_lose_norm -->

#### Scenario: Overlapping peers lose norm under refine
- **WHEN** peers overlap with the inserted vector
- **THEN** the refined vector SHALL lose norm proportional to the overlap
<!-- test: larql_vindex::patch::refine::tests::overlapping_peers_lose_norm_under_refine -->

#### Scenario: Decoy residuals are removed
- **WHEN** the refine pass is given decoy residuals to project away
- **THEN** the refined vector SHALL no longer overlap with the decoys
<!-- test: larql_vindex::patch::refine::tests::decoy_residuals_remove_decoy_overlap -->

#### Scenario: Cross-layer facts do not interfere
- **WHEN** facts from another layer are present
- **THEN** the refine pass SHALL ignore them
<!-- test: larql_vindex::patch::refine::tests::cross_layer_facts_dont_interfere -->

#### Scenario: Array-form input is passed through
- **WHEN** the refine pass receives an array-form input that needs no work
- **THEN** the input SHALL pass through unchanged
<!-- test: larql_vindex::patch::refine::tests::passthrough_for_array_input_form -->

### Requirement: Test suite coverage

The capability's behavior SHALL be exercised by every test in the test modules listed below. New tests added to those modules SHALL be considered part of this capability's contract by default.

#### Scenario: Workspace tests bound to this capability
- **WHEN** the trace tool resolves the wildcard module references for this capability
- **THEN** every backing test SHALL resolve to a real `#[test]` function in the workspace
<!-- test: larql_vindex::patch::overlay::**::* -->
<!-- test: larql_vindex::patch::refine::tests::**::* -->
<!-- test: larql_vindex::patch::format::tests::**::* -->
<!-- test: larql_vindex::patch::knn_store::tests::**::* -->
