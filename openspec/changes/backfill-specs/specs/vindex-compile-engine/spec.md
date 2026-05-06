## ADDED Requirements

### Requirement: StorageEngine wraps PatchedVindex with tier lifecycle

`larql_vindex::engine::core::StorageEngine` SHALL wrap a
`PatchedVindex` and surface L0 (KNN store), L1 (overrides), and
L2 (MEMIT) state through a single read-only `compact_status()`
snapshot. The engine SHALL expose `patched()` /
`patched_mut()` / `into_patched()` accessors, MUST own a
`MemitStore`, and SHALL gate MEMIT participation behind a
hidden-dim threshold so small models cannot accidentally enable
MEMIT.

#### Scenario: A new engine starts at epoch zero
- **WHEN** `StorageEngine::new` is constructed from a `PatchedVindex`
- **THEN** the engine's epoch SHALL be zero
<!-- test: larql_vindex::engine::core::tests::new_engine_epoch_zero -->

#### Scenario: compact_status reflects current L0/L1/L2 occupancy
- **WHEN** `compact_status()` is called on an empty engine
- **THEN** the snapshot SHALL report zero L0 entries, zero L1 edges, zero L2 facts, and the engine's current epoch
<!-- test: larql_vindex::engine::core::tests::compact_status_empty -->

#### Scenario: Mutations update the status fields
- **WHEN** L0/L1/L2 mutations are applied through the patched vindex
- **THEN** the matching counts in `compact_status()` SHALL increase to reflect the new state
<!-- test: larql_vindex::engine::core::tests::mutations_tracked -->

#### Scenario: MEMIT is guarded by hidden-dim threshold
- **WHEN** `supports_memit()` is queried on an engine whose backing model has a hidden size below the published MEMIT minimum
- **THEN** it SHALL return false so callers cannot wire L2 onto a model that is too small to absorb it
<!-- test: larql_vindex::engine::core::tests::memit_guard_small_model -->

### Requirement: Monotonic epoch counter on every mutation

`larql_vindex::engine::epoch::Epoch` SHALL be a monotonically
increasing `u64` counter that starts at zero and advances by
exactly one each time a mutation is recorded. The counter MUST
NOT wrap, MUST NOT decrement, and MUST be observable from
outside the engine via `epoch()` so callers can implement
optimistic concurrency on top of it.

#### Scenario: Epoch starts at zero and only advances forward
- **WHEN** `Epoch::default()` is constructed and `advance()` is invoked an arbitrary number of times
- **THEN** the value SHALL begin at zero and SHALL increase by one per `advance` call
<!-- test: larql_vindex::engine::epoch::tests::epoch_starts_at_zero -->
<!-- test: larql_vindex::engine::epoch::tests::epoch_advances -->

#### Scenario: Engine advance_epoch increments the engine's view
- **WHEN** `StorageEngine::advance_epoch()` is invoked
- **THEN** `engine.epoch()` SHALL increase by one and the per-tier mutation counters SHALL also bump
<!-- test: larql_vindex::engine::core::tests::advance_epoch_increments -->

### Requirement: CompactStatus diagnostic snapshot

`larql_vindex::engine::status::CompactStatus` SHALL be a value
type that records the current epoch, L0/L1/L2 occupancy, base
layer count, base feature count per layer, MEMIT cycle count, and
total facts. Its `Display` implementation SHALL render the L2
section only when MEMIT is populated so empty engines render as
a compact one-line status.

#### Scenario: Display includes MEMIT only when populated
- **WHEN** a `CompactStatus` is rendered with and without MEMIT facts
- **THEN** the populated form SHALL include the MEMIT cycle/fact counts and the empty form SHALL omit them
<!-- test: larql_vindex::engine::status::tests::display_with_memit -->
<!-- test: larql_vindex::engine::status::tests::display_without_memit -->

### Requirement: MEMIT weight decomposition store

`larql_vindex::engine::memit_store::MemitStore` SHALL hold the
per-cycle factual residuals produced by the MEMIT ridge-regression
solver, expose lookup-by-fact and per-cycle iteration, accept
new cycles via `add_cycle`, and SHALL NOT silently overwrite a
fact that already exists in an earlier cycle. The solver SHALL
populate diagnostics (residual norm, iteration count) on every
solve call and the resulting tensors SHALL feed back into the
store so subsequent reads observe them.

#### Scenario: Empty store reports no facts and no cycles
- **WHEN** `MemitStore::new()` is queried for cycle / fact counts
- **THEN** both SHALL be zero and lookup of any fact SHALL return None
<!-- test: larql_vindex::engine::memit_store::tests::empty_store -->

#### Scenario: A single cycle inserts and looks up facts
- **WHEN** a cycle is added containing one fact and the fact is looked up by key
- **THEN** the lookup SHALL return the cycle's residual and the cycle count SHALL be one
<!-- test: larql_vindex::engine::memit_store::tests::add_cycle_and_lookup -->

#### Scenario: Multiple cycles preserve insertion order
- **WHEN** several cycles are added in order
- **THEN** iteration SHALL yield them in insertion order with each cycle's facts intact
<!-- test: larql_vindex::engine::memit_store::tests::multi_cycle -->

#### Scenario: Solver round-trip on orthonormal input
- **WHEN** the MEMIT solver is run on an orthonormal target with a known residual
- **THEN** the solver SHALL recover the input within ridge-regression tolerance
<!-- test: larql_vindex::engine::memit_store::tests::memit_solve_orthonormal_round_trip -->

#### Scenario: Solver populates diagnostics and feeds the store
- **WHEN** the MEMIT solver is invoked
- **THEN** it SHALL populate residual-norm and iteration-count diagnostics, and the resulting cycle SHALL be visible to subsequent `MemitStore` reads
<!-- test: larql_vindex::engine::memit_store::tests::memit_solve_populates_diagnostics -->
<!-- test: larql_vindex::engine::memit_store::tests::memit_solve_feeds_store -->

### Requirement: COMPILE CURRENT INTO VINDEX bake-out path

The COMPILE pipeline SHALL persist a `PatchedVindex` to a
destination directory by hardlinking unchanged base weight files
(via `std::fs::hard_link` on APFS / ext4) and rewriting only
`down_weights.bin` column-wise so previously baked vindexes
share inodes with their parent. The baked vindex MUST NOT carry
the override sidecars used by the patched form, MUST be
deterministic byte-for-byte across runs given the same input,
and MUST round-trip through mmap-based reload without a copy.
After reload, the vindex MUST preserve KNN and HNSW results so
queries do not regress on saved artefacts.

#### Scenario: Save is deterministic across runs
- **WHEN** a synthetic vindex is saved twice to two different directories
- **THEN** the SHA-256 of every produced file SHALL match between the two saves
<!-- test: larql_vindex::golden_save_load::save_is_deterministic -->

#### Scenario: KNN results survive a save/load round-trip
- **WHEN** a vindex is saved and reloaded
- **THEN** KNN queries against the reloaded vindex SHALL return the same neighbour set as queries against the in-memory original
<!-- test: larql_vindex::golden_save_load::knn_round_trip_preserves_results -->

#### Scenario: mmap-based load is zero-copy
- **WHEN** a vindex is loaded from disk via the mmap path
- **THEN** the load SHALL not allocate per-tensor copies and the resulting buffers SHALL alias the on-disk pages
<!-- test: larql_vindex::golden_save_load::mmap_load_is_zero_copy -->

#### Scenario: HNSW after reload overlaps brute-force baseline
- **WHEN** HNSW queries are run against a reloaded vindex and compared against brute-force ground truth
- **THEN** the HNSW result set SHALL overlap the brute-force result set within the published recall threshold
<!-- test: larql_vindex::golden_save_load::hnsw_after_reload_overlaps_brute -->

### Requirement: Test suite coverage

The capability's behavior SHALL be exercised by every test in the test modules listed below. New tests added to those modules SHALL be considered part of this capability's contract by default.

#### Scenario: Workspace tests bound to this capability
- **WHEN** the trace tool resolves the wildcard module references for this capability
- **THEN** every backing test SHALL resolve to a real `#[test]` function in the workspace
<!-- test: larql_vindex::engine::core::tests::**::* -->
<!-- test: larql_vindex::engine::epoch::tests::**::* -->
<!-- test: larql_vindex::engine::status::tests::**::* -->
<!-- test: larql_vindex::engine::memit_store::tests::**::* -->
