# Gate B / Gate S closure audit: findings ledger

**Audited:** main `3b5dd3d2` (2026-11-18). This is a read-only audit; nothing
in the engine changed.
**Definitions:** [`gate-b.md`](gate-b.md) r1, frozen. This ledger records
findings; it does not define the gates.
**Reproducers:** [`probes/gate-b/`](../probes/gate-b/). This is a standalone
crate, not a workspace member and not built by CI. Each test asserts the
invariant, so on the audited main it **fails**. A fix PR moves the probe into
the owning crate's `tests/` and deletes it here.

```sh
# from the repository root; the copied lock pins the workspace's dependency versions
cp Cargo.lock probes/gate-b/Cargo.lock
CARGO_TARGET_DIR=probes/gate-b/target \
  cargo test --manifest-path probes/gate-b/Cargo.toml --no-fail-fast
```

Result on `3b5dd3d2` (2026-11-19), every probe as recorded below:

| Test file | Fails (the finding) | Passes |
|---|---|---|
| `engine_overlay` | `edit9_remove_patch_restores_up_down`, `v2_remove_patch_keeps_base_knn`, `knn_duplicate_key_first_insert_wins` | none |
| `engine_lql` | `b1_reuse_same_path_then_insert_equals_fresh_session`, `b3_two_compose_inserts_same_layer_replay_exactly`, `b3_split_across_two_patches_replay_exactly`, `b4_knn_reinsert_same_key_takes_effect`, `b5a_refused_insert_leaves_no_auto_patch` | `control_single_insert_replays_exactly_and_is_deterministic` (control) |
| `server_overlay` | `s_insert_survives_or_vanishes_whole_after_an_unrelated_remove`, `s_update_set_confidence_takes_effect`, `s_update_does_not_resurrect_a_deleted_slot` | `s_slot_the_chooser_can_pick_a_model_feature` (characterisation) |

Grades follow `gate-b.md` §2: PASS needs code plus a test that would fail;
"exec" means shown by running code, "read" means shown by reading the code. File
references are to `3b5dd3d2`.

---

## Gate B: local engine

### B1: USE isolation | FAIL (exec)

- **State that survives USE:** `installed_edges`, `raw_install_residuals`,
  `decoy_residual_cache` and the epoch/mutation counters on `Session`
  (`crates/larql-lql/src/executor/mod.rs`) are never reset. `relation_resolver`
  is cached per path, so a re-USE of the same path keeps it.
- **What USE does rebuild:** the overlays, the KNN store, the tokenizer, the
  memit store and the patch recording (`executor/lifecycle/use_cmd.rs`).
- **Mechanism:** the compose refine pass Gram-Schmidts each new install against
  the stale residuals, then writes gate/up rows for slots that are not in the
  new binding. On VINDEX2, `set_up_vector` writes an up override onto a base
  slot that no patch records.
- **An existing test pins the leak:**
  `tests/vindex3_mutation.rs::rebalance_converges_and_survives_an_emptied_overlay`
  treats the survival as intended.
- **Decision 2026-11-19 (owner): USE refuses to discard unsaved operations.**
  When the open recording holds operations, USE refuses before doing
  anything, as does EXTRACT, which binds what it builds. The binding and the
  recording stay as they were. SAVE PATCH keeps the operations. An empty
  recording does not block. An explicit discard statement may follow
  separately; it is not part of B1.
- **Probe:** `engine_lql::b1_reuse_same_path_then_insert_equals_fresh_session`.
  Two compose INSERTs, then a re-USE of the same path; the session fingerprint
  still differs from a pristine one. INSERT a: the overlay vectors differ from
  the same INSERT in a fresh session.
- **Also (read):** an open recording is discarded silently on USE, and
  residuals from another model would be reused.

### B2: DELETE completeness, exact REMOVE PATCH | FAIL (exec), VINDEX2 only

- **EDIT-9 still holds on VINDEX2.** `rebuild_overrides`
  (`crates/larql-vindex/src/patch/overlay_apply.rs`) clears meta, gate,
  tombstones and KNN. It does not clear the base `up_overrides`/`down_overrides`
  that patches wrote. Those stay live: `override_slots_at` includes them, and
  COMPILE bakes them.
  - Probe: `engine_overlay::edit9_remove_patch_restores_up_down`. After
    removing an Insert carrying gate/up/down, `up=[2.0;4]` and `down=[3.0;4]`
    remain.
- **VINDEX2 REMOVE PATCH wipes the vindex's own `knn_store.bin`.**
  `rebuild_overrides` resets the store to its default and has no base copy to
  restore.
  - Probe: `engine_overlay::v2_remove_patch_keeps_base_knn`. With one base
    entry, applying and then removing an unrelated patch leaves 0 entries.
- **VINDEX3 is correct and tested:**
  - `crates/larql-vindex/tests/weight_patch/lifecycle.rs::removing_a_patch_replays_over_the_containers_own_knowledge`
  - `crates/larql-lql/tests/vindex3_mutation.rs::compose_insert_alters_execution_and_reverts_bit_for_bit`
  - `knn_delete_patch_removal_resurrects_the_inserted_fact`
- **Why the existing test missed it:**
  `removing_every_patch_restores_the_exact_base_space` covers only metadata
  edits.
- **DELETE by slot** tombstones meta and gate. Up/down overrides stay, but the
  tombstone neutralises them for inference and for COMPILE (test
  `compile_keeps_edits::an_insert_deleted_before_the_compile_contributes_nothing`).
- **No LQL DELETE reaches a KNN fact on main.** FACT-ID-1 (#880) adds
  `delete_knn_fact`.
- **Addendum 2026-11-19 (found while fixing B2):** REMOVE PATCH drops session
  edits that no applied patch holds, on both backends. Examples are an INSERT
  made in this session and edits kept live after SAVE PATCH. VINDEX3 drops all
  of them. VINDEX2 drops their metadata and gate rows but, before the B2 fix,
  kept their up/down vectors. This is open under B2 pending an owner decision:
  replay them from a session journal, or refuse.
- **Decision 2026-11-19 (owner): refuse.** REMOVE PATCH refuses, before
  anything changes, while the session holds a knowledge edit no applied patch
  holds. Replaying such edits without the removed patch is not defined
  (apply A, edit B, apply C, remove A). A replay or rebase of session edits
  would be a separate, explicitly verified operation. Implemented in #884
  (`crates/larql-lql/tests/gate_b_remove_patch_refusal.rs`).

### B3: exact replay | FAIL (exec on VINDEX3; VINDEX2 by reading the code at audit time)

- **Mechanism:** the compose refine pass rewrites the gate/up of earlier slots
  at the same layer. Only the op of the current install is refreshed
  (`crates/larql-lql/src/executor/mutation/insert/mod.rs`).
  `refresh_recorded_patch_ops_for_slots` runs only from REBALANCE.
- **Probes:** `engine_lql::b3_two_compose_inserts_same_layer_replay_exactly`
  (one patch) and `engine_lql::b3_split_across_two_patches_replay_exactly`
  (two patches). After a fresh APPLY, the overlay-vector fingerprints differ
  from the live session (`210fa6…` live, `0641610…` replay). INFER top-5 matches
  to two decimals, which is exactly why semantic checks alone are not enough.
- **Control:** `engine_lql::control_single_insert_replays_exactly_and_is_deterministic`
  passes. A single insert replays bit-exact, and the run is deterministic.
- **Addendum 2026-11-19:** the audit probes ran on VINDEX3 only. The B3 fix
  PR shows the VINDEX2 case by execution as well, on a vindex extracted from
  the dense fixture model. The synthetic model of `larql-inference` gives both
  installs the same gate, so its refine changes nothing. REBALANCE has the same
  gap across saved patches: it refreshed only ops in the open recording.
- **Also (read):** COMPACT MINOR removes KNN entries with no recorded op
  (`crates/larql-lql/src/executor/compact.rs`), so a replay holds both the KNN entry and
  the compose slot.

### B4: duplicate assertions | FAIL on main (exec); closed by #880

- `KnnStore::add` appends without deduplicating, and ties break by insertion
  order. Re-inserting the same compose fact installs another slot.
- **Probes:**
  - `engine_lql::b4_knn_reinsert_same_key_takes_effect`. Both INSERTs report
    success and DESCRIBE lists both, but INFER returns `[5]`.
  - `engine_overlay::knn_duplicate_key_first_insert_wins` shows the same at
    store level.
- **#880 closes it by refusal:** INSERT refuses a live key, APPLY PATCH refuses
  a patch that re-asserts one (I-2), and legacy duplicates load as `ambiguous`.
  The probes assert "replace", so after #880 they are replaced by #880's gates
  ID-12 and ID-13, not moved.

### B5: INSERT atomicity | FAIL (exec)

- **Compose install is not atomic.** This is already pinned as a known defect
  in `crates/larql-lql/tests/install_fault/atomicity.rs`:
  - `known_defect_a_failed_install_leaves_the_session_changed_after_its_first_write`
  - `known_defect_the_overlay_can_hold_an_edit_the_saved_patch_lacks`
  - `#[ignore] a_failed_install_changes_nothing`

  Decoys, slots and residuals are committed before the fallible
  `balance_installed`/`cross_fact_regression_check`. The fault points exist on
  VINDEX3 only; VINDEX2 has the same shape (read).
- **A refused INSERT opens an automatic recording.** The dispatcher calls
  `ensure_patch_session` before `exec_insert` (`executor/mod.rs`); DELETE and
  UPDATE avoid this.
  - Probe: `engine_lql::b5a_refused_insert_leaves_no_auto_patch`.

### B6: COMPACT / COMPILE | FAIL for COMPACT MINOR (read); NOT_EVIDENCED for MAJOR, INTO and COMPILE

- **COMPACT MINOR** swallows each per-fact `exec_insert` error and returns Ok
  with "N failed". Each failure can leave a half-written install. Successful
  promotions remove KNN entries without recording an op, and the epoch
  advances regardless.
- **COMPACT MAJOR** writes `memit_store.json` into the **base vindex
  directory**. A persist failure there is only a warning.
- **COMPILE** borrows the session immutably on every path, so a failure cannot
  mutate it. But no test fingerprints the session after a failed COMPILE.
- **Addendum 2026-11-19 (found while fixing B2):** COMPILE's output is not
  byte-deterministic. `index.json` writes its checksum map in HashMap order, so
  compiling the same state twice can give different bytes.

### B7: verifier closure | FAIL against the FACT-WSP-1 requirements

The verifier is the Python package in `verifier/`. The engine is pinned to the
v4 fixtures by
`crates/larql-lql/tests/weight_patch_lql/v4_fixtures.rs::the_engine_still_writes_the_containers_the_verifier_is_tested_on`.

| Op or composition | Covered |
|---|---|
| Delta (UPDATE WEIGHTS), several ops on one tensor and across tensors | yes (v2 fixtures; **not pinned to the engine**, provenance `47d56f6b`) |
| One REMOVE DIRECTION; two on one tensor; delta + removal on different tensors | yes (v3 fixtures; not pinned to the engine) |
| ADD LOW RANK: f64-rne1, f32, sham; mixed with REMOVE DIRECTION and a delta | yes (v4, pinned, with a tamper suite) |
| ADD LOW RANK + REMOVE DIRECTION on different tensors | yes, but only with `--patch`; without it the numerical half is not evaluated |
| **Two or more ADD LOW RANK on one tensor (disjoint columns)** | **no**, although `declare_patches` allows it |
| **A zero-valued op beside a nonzero op on one tensor** | **no**; whether it is allowed is undetermined |
| **ADD LOW RANK f64-rne2, REMOVE DIRECTION f32-seq-rne1 inside a container** | **no** (arithmetic only, in `test_vlp_numerics.py`) |
| **VINDEX3 knowledge bake (`COMPILE CURRENT INTO VINDEX`)** | **no**; no ledger is written |
| **VINDEX2 weight-affecting paths** (compose INSERT, REBALANCE, COMPACT MAJOR/MEMIT, MERGE, VINDEX2 COMPILE) | **no**; OUT_OF_SCOPE for FACT-WSP-1 |
| InsertKnn / DeleteKnn | no |
| Delta + structured op on one tensor; weight ops + knowledge edits in one compile | refused (tested) |

### Rows that are PASS on main

| Row | Evidence |
|---|---|
| UPDATE: validate and resolve before any write | `where_predicate::update_refuses_every_unevaluable_clause_and_changes_nothing`, `a_refused_or_matchless_mutation_does_not_open_the_auto_patch`, `update_set_validation.rs` |
| DELETE refusal paths | `where_predicate::delete_refuses_every_unevaluable_clause_and_changes_nothing`, `a_pinned_slot_outside_the_vindex_is_refused_not_recorded` |
| APPLY PATCH whole-patch validation | `apply_patch_integrity` v2/v3 `one_invalid_operation_stops_the_valid_ones_before_it`; larql-vindex `tests/patch_validate.rs` |
| MERGE atomicity | `merge_truthful::a_source_with_a_slot_the_bound_vindex_lacks_is_refused_whole` and the three refusal tests beside it. **Weak spot:** the VINDEX3 refusal test checks SHOW PATCHES, not the overlay records. |

---

## Gate S: server

| Row | Grade | Finding |
|---|---|---|
| EDIT-5: remote DELETE/UPDATE are per session | PASS | #801; `tests/cov_remote_mockito/session_scope.rs` |
| ID-2: server checks `base_model` | PASS | #802; `test_http_patch_model_and_slots.rs` |
| ID-4: slot allocation under the lock | PASS when sequential; **NOT_EVIDENCED** under concurrency (S5) | #802. No concurrent slot test. Characterisation probe `server_overlay::s_slot_the_chooser_can_pick_a_model_feature`: "free" can be the weakest real model feature (documented in server-spec §8.6). |
| Session reads | PASS for the HTTP browse routes (#804); **FAIL (read)** for `/v1/stream`, gRPC, walk-ffn and the OpenAI routes; **NOT_EVIDENCED** for INFER and EXPLAIN INFER (S4) | `test_http_session_reads.rs` |
| S-INSERT: remote INSERT | **FAIL** (exec at overlay level) | `/v1/insert` writes the overlay with no patch, so SHOW PATCHES and REMOVE PATCH cannot reach it. Layers with no free slot are skipped one by one. `inserted:0` is still HTTP 200 and prints "Inserted". ALPHA and MODE are dropped. Probe `server_overlay::s_insert_survives_or_vanishes_whole_after_an_unrelated_remove`: after an unrelated REMOVE PATCH, the insert's meta and gate are gone but its down vector stays on the base slot. |
| S-UPDATE: remote UPDATE | **FAIL** (exec at overlay level) | `SET confidence` alone builds an op with no metadata; it changes nothing but prints "Updated" (probe `s_update_set_confidence_takes_effect`). It brings deleted slots back, which local UPDATE does not (probe `s_update_does_not_resurrect_a_deleted_slot`). SET target forces `c_score` 0.9 and `top_token_id` 0. |
| Remote APPLY PATCH / DELETE atomicity | PASS | `test_http_patch_integrity.rs` (3 tests); `session_scope.rs` |
| Multi-model session writes | **FAIL** (read) | Reads check the bound model; writes do not. `POST /v1/B/patches/apply` with a session bound to A passes B's `base_model` check and lands in A's overlay. |
| A request with no session header writes to the shared model | by design | No auth on this path. This needs a Gate S decision under S1. |

## Separate: Q-1, ORDER BY | FAIL (read)

- An ORDER BY on an unknown field is ignored (the `_ => {}` arm in
  `crates/larql-lql/src/executor/query/select/edges.rs`).
- ORDER BY on FEATURES/ENTITIES is dropped.
- ORDER BY and NEAREST TO on a remote SELECT are dropped (also S6).
