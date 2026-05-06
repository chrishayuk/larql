## ADDED Requirements

### Requirement: VectorIndex construction and per-layer access

`VectorIndex` SHALL expose the loaded layer set, the number of features
per layer, and per-feature metadata lookup. The constructor MUST
preserve the gate and down-meta override slots empty by default and
MUST report the configured layer dimensions on a freshly built index.
Cloning a `VectorIndex` SHALL share Arc-mmap handles rather than
deep-copying.

#### Scenario: New index reports its declared dimensions
- **WHEN** a fresh `VectorIndex` is constructed from a known config
- **THEN** dimension accessors SHALL match the config
<!-- test: larql_vindex::test_vindex::new_index_has_correct_dimensions -->

#### Scenario: Loaded-layers view reflects the layer range
- **WHEN** a vindex is constructed for a sub-range of layers
- **THEN** `loaded_layers` SHALL list exactly that range
<!-- test: larql_vindex::test_vindex::loaded_layers -->

#### Scenario: Layer feature counts match per-layer config
- **WHEN** `num_features_per_layer` is queried on a multi-layer vindex
- **THEN** each layer SHALL return the count declared in its layer info
<!-- test: larql_vindex::test_vindex::num_features_per_layer -->

#### Scenario: Total feature count aggregates per-layer counts
- **WHEN** `total_counts` is read on a multi-layer vindex
- **THEN** the total SHALL equal the sum of per-layer counts
<!-- test: larql_vindex::test_vindex::total_counts -->

#### Scenario: Feature meta lookup returns inserted entry
- **WHEN** a feature is inserted and then looked up by index
- **THEN** the metadata SHALL match the insertion
<!-- test: larql_vindex::test_vindex::feature_meta_lookup -->

#### Scenario: Missing feature meta is None
- **WHEN** a non-existent feature index is queried
- **THEN** the lookup SHALL return `None`
<!-- test: larql_vindex::test_vindex::feature_meta_none_for_missing -->

#### Scenario: Down-meta accessor returns the per-feature slice
- **WHEN** `down_meta_at` is called on a populated layer
- **THEN** it SHALL return the slice for that feature
<!-- test: larql_vindex::test_vindex::down_meta_at_returns_slice -->

#### Scenario: Cloning preserves Arc-mmap fields
- **WHEN** a vindex backed by mmap is cloned
- **THEN** the clone SHALL share the underlying mmap handle via Arc
<!-- test: larql_vindex::index::core::tests::clone_shares_arc_mmap_handles -->

#### Scenario: New mmap-backed index defaults remaining fields
- **WHEN** `new_mmap` constructs a vindex
- **THEN** mmap fields SHALL be set and the rest SHALL fall back to defaults
<!-- test: larql_vindex::index::core::tests::new_mmap_sets_mmap_fields_and_defaults_rest -->

### Requirement: Dense gate KNN with batching

Gate-vector KNN SHALL return the top-K matches for a probe in dot-product
similarity, MUST handle empty layers without panicking, and SHALL
preserve descending-score ordering. The KNN dispatch path MUST select
the quantization-aware backend automatically when q4k storage is in
play.

#### Scenario: Gate KNN finds the best match
- **WHEN** the probe is the same as one stored gate vector
- **THEN** that vector SHALL appear at rank 1
<!-- test: larql_vindex::test_vindex::gate_knn_finds_best_match -->

#### Scenario: Top-K ordering is monotonic descending
- **WHEN** `gate_knn` returns multiple results
- **THEN** scores SHALL be sorted in descending order
<!-- test: larql_vindex::test_vindex::gate_knn_top_k_ordering -->

#### Scenario: Gate KNN returns empty for missing layer
- **WHEN** a layer that has no gate vectors is queried
- **THEN** the result SHALL be empty rather than erroring
<!-- test: larql_vindex::test_vindex::gate_knn_empty_for_missing_layer -->

#### Scenario: Gate KNN over q4k storage produces results
- **WHEN** a vindex backed by q4k storage is queried
- **THEN** `gate_knn_q4` SHALL return non-empty top-K matches
<!-- test: larql_vindex::test_vindex::gate_knn_q4_produces_results -->

#### Scenario: Q4 method invocation matches direct path
- **WHEN** the q4 KNN method is invoked through the public API
- **THEN** it SHALL produce results equivalent to the direct dispatch path
<!-- test: larql_vindex::test_vindex::gate_knn_q4_method_works -->

#### Scenario: Gate-walk and gate-KNN agree
- **WHEN** the same probe is evaluated by `gate_walk` and `gate_knn`
- **THEN** the resulting hits SHALL agree
<!-- test: larql_vindex::test_vindex::gate_walk_matches_gate_knn -->

### Requirement: HNSW approximate KNN

The vindex SHALL support an HNSW index that returns approximate top-K
matches with O(log N) traversal, scores expressed as dot products, and
results sorted in descending score order. Edge cases — empty index,
single-vector index — MUST not panic, and recall SHALL meet the
documented targets at K=10 and K=100.

#### Scenario: HNSW build and search on a small set
- **WHEN** an HNSW index is built over a small corpus and queried
- **THEN** the search SHALL succeed and return up to K candidates
<!-- test: larql_vindex::test_hnsw::build_and_search_small -->

#### Scenario: Recall at 10 meets target
- **WHEN** the HNSW index is queried for top-10
- **THEN** the recall versus brute-force SHALL meet the documented threshold
<!-- test: larql_vindex::test_hnsw::recall_at_10 -->

#### Scenario: Recall at 100 meets target on a larger corpus
- **WHEN** the HNSW index is queried for top-100 against a larger corpus
- **THEN** recall versus brute-force SHALL meet the documented threshold
<!-- test: larql_vindex::test_hnsw::recall_at_100_large -->

#### Scenario: Empty index returns no results
- **WHEN** an HNSW index over zero vectors is queried
- **THEN** the result SHALL be empty
<!-- test: larql_vindex::test_hnsw::empty_index -->

#### Scenario: Single-vector index returns that vector
- **WHEN** an HNSW index containing exactly one vector is queried
- **THEN** the returned hit SHALL be that vector
<!-- test: larql_vindex::test_hnsw::single_vector -->

#### Scenario: HNSW scores are dot products
- **WHEN** HNSW returns a hit
- **THEN** the score SHALL equal the dot product between probe and stored vector
<!-- test: larql_vindex::test_hnsw::scores_are_dot_products -->

#### Scenario: HNSW results are sorted descending
- **WHEN** HNSW returns multiple hits
- **THEN** scores SHALL be in descending order
<!-- test: larql_vindex::test_hnsw::results_sorted_descending -->

#### Scenario: HNSW gate-KNN smoke path returns results
- **WHEN** HNSW is wired into the gate KNN dispatcher
- **THEN** end-to-end queries SHALL succeed
<!-- test: larql_vindex::test_hnsw::gate_knn_hnsw_smoke -->

#### Scenario: HNSW survives reload
- **WHEN** a vindex is saved and reloaded with HNSW enabled
- **THEN** HNSW results SHALL overlap with brute-force results
<!-- test: larql_vindex::golden_save_load::hnsw_after_reload_overlaps_brute -->

### Requirement: mmap-first storage with residency tracking

Per-layer storage SHALL be backed by mmap by default, MUST support an
LRU cache with a configurable cap, and MUST expose residency state
(cold, mmap, pinned). Pinning, eviction, and budget enforcement MUST
work without surprising the caller.

#### Scenario: Unlimited cache grows without eviction
- **WHEN** the LRU cap is `None`
- **THEN** entries SHALL accumulate without eviction
<!-- test: larql_vindex::index::storage::gate_store::tests::unlimited_cache_grows_without_eviction -->

#### Scenario: Two-entry cap evicts LRU on third insert
- **WHEN** the cap is set to two and a third entry is loaded
- **THEN** the least-recently-used entry SHALL be evicted
<!-- test: larql_vindex::index::storage::gate_store::tests::cap_two_evicts_lru_on_third_access -->

#### Scenario: Cache hits promote the layer to most-recent
- **WHEN** a cached layer is touched again
- **THEN** it SHALL be moved to the most-recently-used slot
<!-- test: larql_vindex::index::storage::gate_store::tests::cache_hit_promotes_layer_to_newest -->

#### Scenario: Shrinking the cap evicts down to the bound
- **WHEN** the cap is reduced below the current entry count
- **THEN** entries SHALL be evicted until the cap is met
<!-- test: larql_vindex::index::storage::gate_store::tests::shrinking_cap_evicts_down_to_new_bound -->

#### Scenario: Setting cap zero is a no-op for existing entries
- **WHEN** the cap is set to zero with entries already loaded
- **THEN** the existing entries SHALL be retained
<!-- test: larql_vindex::index::storage::gate_store::tests::set_cap_zero_is_noop_on_existing_entries -->

#### Scenario: Pinning succeeds within budget
- **WHEN** `pin_layer` is invoked and the residency budget is sufficient
- **THEN** pinning SHALL succeed and the layer SHALL transition to pinned
<!-- test: larql_vindex::index::storage::residency::tests::pin_layer_succeeds_within_budget -->

#### Scenario: Pinning fails when over budget
- **WHEN** `pin_layer` would exceed the configured budget
- **THEN** the call SHALL fail rather than silently exceed the cap
<!-- test: larql_vindex::index::storage::residency::tests::pin_layer_fails_when_over_budget -->

#### Scenario: Eviction frees memory for non-pinned layers
- **WHEN** `evict_layer` is invoked on a pinned layer
- **THEN** the layer SHALL transition out of pinned state and free its memory
<!-- test: larql_vindex::index::storage::residency::tests::evict_layer_frees_memory -->

#### Scenario: Auto-pin fills budget by access count
- **WHEN** `auto_pin` is invoked
- **THEN** the most-accessed layers SHALL be pinned first up to the budget
<!-- test: larql_vindex::index::storage::residency::tests::auto_pin_fills_budget_most_accessed_first -->

### Requirement: Quantization-aware FFN dispatch

The FFN row-dot, row-scaled-add, and row-into helpers SHALL prefer FP4
when available, fall back to native f32/f16, and finally to Q4K. When
no backend covers the request, the helpers SHALL signal failure rather
than silently no-op.

#### Scenario: FP4 wins when present
- **WHEN** all three backends are available for a layer
- **THEN** FFN row dot SHALL use FP4
<!-- test: larql_vindex::index::ffn_dispatch_tests::ffn_row_dot_priority_fp4_wins_over_native_and_q4k -->

#### Scenario: Falls through FP4 None to native
- **WHEN** FP4 is missing for a layer but native weights exist
- **THEN** the dispatcher SHALL fall through to native
<!-- test: larql_vindex::index::ffn_dispatch_tests::ffn_row_dot_falls_through_fp4_none_to_native -->

#### Scenario: Falls through to Q4K when no native
- **WHEN** only Q4K is available
- **THEN** the dispatcher SHALL invoke the Q4K backend
<!-- test: larql_vindex::index::ffn_dispatch_tests::ffn_row_dot_falls_through_to_q4k_when_no_native -->

#### Scenario: Returns None when no backend covers
- **WHEN** no backend covers the request
- **THEN** FFN row dot SHALL return `None`
<!-- test: larql_vindex::index::ffn_dispatch_tests::ffn_row_dot_returns_none_when_no_backend_covers -->

#### Scenario: Row-scaled-add gate/up uses direct Q4K
- **WHEN** the gate or up component is requested over Q4K storage
- **THEN** the dispatcher SHALL invoke the direct Q4K path
<!-- test: larql_vindex::index::ffn_dispatch_tests::ffn_row_scaled_add_gate_up_uses_direct_q4k -->

#### Scenario: Row-scaled-add returns false when no backend exists
- **WHEN** no backend covers a row-scaled-add call
- **THEN** the helper SHALL return false
<!-- test: larql_vindex::index::ffn_dispatch_tests::ffn_row_scaled_add_returns_false_when_no_backend -->

### Requirement: Test suite coverage

The capability's behavior SHALL be exercised by every test in the test modules listed below. New tests added to those modules SHALL be considered part of this capability's contract by default.

#### Scenario: Workspace tests bound to this capability
- **WHEN** the trace tool resolves the wildcard module references for this capability
- **THEN** every backing test SHALL resolve to a real `#[test]` function in the workspace
<!-- test: larql_vindex::test_hnsw::**::* -->
<!-- test: larql_vindex::index::ffn_dispatch_tests::**::* -->
<!-- test: larql_vindex::index::storage::gate_store::**::* -->
<!-- test: larql_vindex::index::storage::residency::tests::**::* -->
