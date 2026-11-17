# FACT-ID-1 — logical fact identity for asserted edges

**Status:** r2 (2026-11-17) — **FROZEN.** Owner conditional GO on r1 with
D-5, D-6 and D-8 refined and three invariants added (§3.5). The acceptance
gates (§9) are frozen at this revision. A change to them is a new revision,
recorded here, before code relies on it.
**Base:** `origin/main` 110a3b3e. **Branch:** `fact-id-1`.
**Successors (not in this branch):** FACT-PROMOTE-1 (verified KNN → compose
promotion on VINDEX3), FACT-WSP-1 (behavioural UPDATE through weight-edit
ops), FACT-GRAPH-1 (edit-site discovery from the graph-walk work).

### Revision log

- **r1** — draft for review.
- **r2** — D-5 limited to KNN; compose DELETE as well as UPDATE refused
  (§4, D-6: no ledger op can restore a compose slot to its base, so no
  semantic removal is provably safe); version = logical transition, and a
  KNN UPDATE is one op (`update_knn_fact`), not delete + insert; cross-patch
  conflict rule on APPLY PATCH; invariants for durable identity, patch
  conflicts and structured results (§3.5); the session op journal (§3.3),
  because `SAVE PATCH` leaves its edits live in no applied patch (C-10);
  C-10 to C-12 added.

## 1. Purpose

A fact a user INSERTs must be one object that SELECT lists, that UPDATE and
DELETE address by the same `(entity, relation)` key, and that INFER
reflects. Today none of that holds end to end (§2). FACT-ID-1 establishes
what an **asserted** fact is and how its lifecycle works. It does not
change what model introspection means: features found by the gate walk stay
**discovered** rows, readable by meaning and mutable only by physical slot.

FACT-ID-1 does not depend on finding any neuron. A later parametric
realization (FACT-WSP-1) plugs in behind the interface defined here.

## 2. Current behaviour this contract changes or guards

Each item is a behaviour of `origin/main` 110a3b3e, reported before
anything changes it.

| # | Current behaviour | Where | FACT-ID-1 |
|---|---|---|---|
| C-1 | `UPDATE`/`DELETE … WHERE entity AND relation` is refused: SELECT answers that pair with a ranked gate walk, too broad to mutate. | `executor/mutation/mod.rs` `resolve_mutation_targets` | The pair addresses **asserted facts only**. Discovered rows stay unreachable by it. |
| C-2 | `SELECT FROM EDGES` scans feature metadata only; a KNN-inserted fact is never listed (only `DESCRIBE` and `INFER` see it). | `query/select/edges.rs` | SELECT lists asserted facts in their own block. |
| C-3 | Re-INSERTing an existing `(entity, relation)` appends a second KNN entry (`KnnStore::add` never deduplicates). | `larql-vindex/src/patch/knn_store.rs` | Refused; UPDATE changes a fact. |
| C-4 | The only KNN delete op, `delete_knn`, removes **every relation** of an entity. No statement removes one KNN fact. | `patch/format.rs` | New ops `delete_knn_fact`, `update_knn_fact` (§6.1). `delete_knn` kept for replay. |
| C-5 | A KNN insert stores the full target text but only the **first** token id of `" {target}"`; INFER's override reports the text. | `insert/knn.rs`; `larql-inference/src/forward/infer_patched.rs` | KNN INSERT/UPDATE require a single-token target (D-5). Legacy records flagged `multi_token`. |
| C-6 | `UPDATE EDGES SET target` changes `top_token` text, keeps `top_token_id`; COMPILE refuses it. | `mutation/update_set.rs` | Unchanged for physical rows. Asserted KNN UPDATE changes id and text together. |
| C-7 | SELECT ignores its field list. | `exec_select` | Unchanged; projection out of scope. Asserted rows are structured results (§3.5 I-3). |
| C-8 | Compose installs are not independent: the refine pass rewrites earlier slots' gates; `REBALANCE` rewrites every installed fact's down vector and rewrites earlier recorded ops in place; `installed_edges` is session-only. Tombstoning a slot zeroes the base slot's own contribution too, so it is not a restore. | `insert/compose_v3/`, `mutation/rebalance.rs`, `tests/compile_keeps_edits.rs` | Semantic UPDATE **and** DELETE of compose facts refused (D-6). |
| C-9 | A compose fact's identity does not survive COMPILE (the container keeps vectors and `down_meta`, not entity/relation). A KNN fact's does (`knn_store.bin`). | `compile/into_vindex*.rs` | Guarded: COMPILE says which asserted compose facts lose identity; they read back as discovered. |
| C-10 | `SAVE PATCH` writes the recording and ends it, but the edits stay live in the overlay and are **not** an applied patch. They are in no ledger component the session can list. | `executor/patch_statements.rs` `save_patch` | A session op journal on the overlay (§3.3) makes them part of the ledger. No change to SAVE's behaviour. |
| C-11 | On VINDEX2, `REMOVE PATCH` rebuilds the KNN store from empty, dropping the container's own `knn_store.bin` facts; VINDEX3 restores them from `base_knn`. | `patch/overlay_apply.rs` `rebuild_overrides` | Not changed (VINDEX2 is compatibility). The catalogue sees the loss and marks those facts `inconsistent` (§3.4); semantic mutation of them refuses. |
| C-12 | `COMPACT MINOR` (VINDEX2) removes KNN entries straight from the store, with no recorded op. | `executor/compact.rs` | Not changed. The catalogue detects the divergence and marks the facts `inconsistent`. |

## 3. Model

### 3.1 Asserted vs discovered

- **Asserted fact**: one LQL put there. It comes from an `INSERT` in the
  effective ledger (§3.3), or from a KNN entry in the bound container's
  `knn_store.bin` (a compiled insert).
- **Discovered row**: a feature-metadata row the existing resolver finds.
  It is evidence about a feature, not a claim that one fact lives there.
  FACT-ID-1 gives it no identity.
- Compose `Insert` ops without a relation (written before relations were
  recorded) have no key and are not asserted facts.

### 3.2 Fact key and fact id

- **Fact key** = `(norm(entity), norm(relation))`. `norm` lowercases, maps
  `-` and `_` to space, collapses whitespace runs to one space and trims.
- **Fact id** = `f_` + the first 16 hex digits of
  `sha256(norm(entity) ‖ 0x1F ‖ norm(relation))`. It depends on the key
  alone: the same whatever the op's position, patch order or
  consolidation, and the same across realizations.
- **Collision:** an INSERT, or an APPLY PATCH, that would give two different
  keys one id is refused, naming both keys. Ids are never lengthened or
  renumbered.
- The displayed spelling is the one the latest op on the key used.

### 3.3 The ledger is authoritative

There is no fact store. The catalogue is a view folded, in order, from:

1. the container's own KNN entries (`base_knn`): origin `container`;
2. the applied patches, in order: origin is the patch name;
3. the **session journal**: every knowledge op the session applied to the
   overlay directly (INSERT, UPDATE, DELETE, MERGE, REBALANCE), in order:
   origin `session`.

The journal is a field of the overlay (`KnowledgeOverlay`, `PatchedVindex`),
so its lifetime is the overlay's. It is cleared whenever the overlay
resets session state (`remove_patch`) and is never written to disk. The
open recording is a *subset* of it, the part SAVE PATCH will write. Both
are appended at the same point, through one helper. The journal is an op
log, not a fact catalogue: it holds `PatchOp`s exactly as recorded.

The fold yields, per key: the realization records, the target and its id,
the confidence, the **version** and the provenance. Dropping the result and
folding again gives the same catalogue (I-1).

**Versions count logical transitions.** These each count as one transition:
`insert_knn`, `update_knn_fact`, `delete_knn_fact`, a `delete_knn`
that removes the key, a container entry (version 1), and one compose INSERT.
A compose INSERT records one `Insert` op per slot; consecutive ops of one
source with the same key are one transition. A deleted key keeps its
version, so a later INSERT continues from it.

### 3.4 Realization and status

| realization | backed by |
|---|---|
| `knn` | the `KnnEntry`s with the key |
| `compose` | the slots of keyed `PatchOp::Insert` ops |

| status | meaning | INFER behaviour claimed? | semantic UPDATE | semantic DELETE |
|---|---|---|---|---|
| `active` | one KNN entry, target single-token | yes | yes | yes |
| `multi_token` | one KNN entry whose target text needs more than one token | no | refused | yes |
| `ambiguous` | more than one live KNN entry for the key (legacy ledgers) | no | refused | yes (removes all) |
| `inconsistent` | the fold disagrees with the live KNN store for this key (C-11, C-12, or an unknown cause) | no | refused | refused |
| `compose` | a keyed compose install | no (see FACT-PROMOTE-1) | refused | refused |
| `degraded` | a compose fact one of whose slots was later deleted or overwritten by a physical statement | no | refused | refused |

`inconsistent` is computed every time the catalogue is built: the fold's
KNN records per key are compared with the live store's entries per key
(layer, target id, target text, confidence). A live key the fold does not
know is listed as `inconsistent` with provenance `unknown`.

### 3.5 Invariants

- **I-1 Durable identity.** A fact is reported with `durability`:
  `container` (in `knn_store.bin`), `patch` (in an applied patch file) or
  `session` (only in the journal: lost on exit unless saved). Replay and
  reload of a saved patch, and COMPILE then reload of KNN facts, reproduce
  the same ids, targets and statuses. Versions are reproduced on replay; a
  compiled container restarts every fact at version 1, its container
  entry. Compose facts do not survive COMPILE (C-9). COMPILE says so, and
  never reports them as durable afterwards. The catalogue is a disposable
  cache.
- **I-2 Patch conflicts refuse.** APPLY PATCH folds the incoming patch's
  keyed ops over the current catalogue before anything is applied. The
  whole patch is refused, with no partial apply, when it would:
  - insert (`insert_knn` or a keyed compose `Insert`) a key that is live at
    that point (`active`, `multi_token`, `ambiguous`, `inconsistent`,
    `compose` or `degraded`);
  - `update_knn_fact` or `delete_knn_fact` a key that is not live as `knn`;
  - collide on an id.

  A patch that deletes a key and then inserts it is accepted. Duplicate
  inserts *within* one legacy patch keep loading (status `ambiguous`).
  Rejecting them would break replay of existing files.
- **I-3 Structured results.** The catalogue is a typed value
  (`AssertedFact`, with `FactRealization`, `FactStatus`,
  `FactDurability`), returned by `Session::asserted_facts()`. SELECT formats
  it. No later code parses text to recover ids, realizations or statuses.

## 4. Decisions (owner-approved r1 → r2)

- **D-1 Ledger authoritative** (§3.3). No second persistent store.
- **D-2 One SELECT view** with origin; semantic mutation reaches asserted
  rows only.
- **D-3 KNN UPDATE is the behavioural milestone.** Key and layer kept;
  target re-tokenized; id and text replaced together, by one op; INFER
  changes; rollback and replay exact.
- **D-5 Single-token rule for KNN.** New KNN INSERT and every asserted KNN
  UPDATE require `" {target}"` to encode to exactly one id whose decoding,
  trimmed, equals the trimmed target. Compose INSERT is **not** restricted.
  Legacy multi-token records stay readable as `multi_token`.
- **D-6 Compose facts are read-only by identity.** Semantic UPDATE and
  DELETE of a compose fact are refused. There is no ledger op that returns
  a compose slot to its base: `PatchOp::Delete` tombstones it, and that
  zeroes the base slot's own contribution too (C-8). Balancing and refine
  also make earlier facts' recorded vectors depend on later installs. The
  refusal names `REMOVE PATCH` of the patch that installed the fact as the
  exact rollback, or a physical DELETE by layer and feature for whoever
  accepts its effect.
- **D-7 FR3 synonym resolution stays discovered-only.** Asserted rows
  match the relation by its normalized form, in SELECT and in mutation.
- **D-8 Semantic targeting is exactly `entity = …` AND `relation = …`**
  and nothing else in the WHERE. Every other shape keeps today's meaning and
  lists no asserted block.

## 5. Statement semantics

### 5.1 INSERT

Before capture, with read-only borrows:
1. Normalize the key.
2. Refuse if the key is live (any status). Use UPDATE, or DELETE first.
3. Refuse an id collision.
4. KNN mode: refuse a target that fails D-5.

Then the existing install runs unchanged. Its ops go to the journal and
the recording through the one helper. Output adds `  fact f_… (vN)`.

### 5.2 SELECT FROM EDGES

For the D-8 shape, when an asserted fact has the key:

```
Asserted facts:
  fact                entity  relation  target  realization  status  version  durability
  f_3a9c…             France  capital   Paris   knn          active  2        session
Discovered edges:
<today's output for the same clause, unchanged>
```

When none does, or for any other shape, output is byte-identical to today.

### 5.3 UPDATE EDGES … WHERE entity = … AND relation = …

- `SET target` and/or `confidence` (the existing `UpdateSet` validation).
- No live asserted fact with the key: refused. It never falls back to
  discovered rows.
- `active` knn: one `update_knn_fact { entity, relation, target,
  target_id, confidence }`. Key vector and layer untouched. The new target
  must pass D-5. `confidence` unchanged when not set.
- Any other status: refused, naming it (§3.4).
- Output: `Updated fact f_… (v2 → v3): target Paris → Lyon (knn, L<n>)`.

### 5.4 DELETE FROM EDGES … WHERE entity = … AND relation = …

- `active`, `multi_token`, `ambiguous` knn: one `delete_knn_fact`. Every
  entry with the key goes.
- `inconsistent`, `compose`, `degraded`: refused (D-6, §3.4).
- No live asserted fact: refused.

### 5.5 INFER, DESCRIBE

No change in mechanism.

### 5.6 COMPILE

When the catalogue holds compose facts, COMPILE (both backends) adds one
line naming how many asserted compose facts lose their identity in the
output, and that they will read back as discovered rows.

## 6. Ledger ops, replay, rollback

### 6.1 New ops

```json
{ "op": "delete_knn_fact", "entity": "France", "relation": "capital" }
{ "op": "update_knn_fact", "entity": "France", "relation": "capital",
  "target": "Lyon", "target_id": 12345, "confidence": 0.9 }
```

They match entries by fact key (§3.2), through new `KnnStore` methods.
`remove_by_entity_relation` (used by COMPACT MINOR) is left unchanged.
They are **new op tags**. An older reader fails to parse an unknown tag and
refuses the whole patch. Adding a field to `delete_knn` would instead let
an older reader ignore the field and delete every relation of the entity.
Patch counts: `update_knn_fact` → updates, `delete_knn_fact` → deletes.

### 6.2 Replay and rollback

- Saving and reapplying a patch reproduces the catalogue and the KNN store.
  `insert_knn` carries the key vector. The update op keeps the stored one.
- `REMOVE PATCH` rebuilds from the base plus the remaining patches, as
  today. On VINDEX3 that restores a container fact a removed patch had
  deleted or updated. On VINDEX2 it does not (C-11, flagged
  `inconsistent`).
- `insert_knn` keys are decode-checked before apply on both backends
  (`validate_patch_vectors`, `validate_v3_patch`); FACT-ID-1 confirms this
  with a test.

## 7. Backends

- **VINDEX3 primary**: all gates on `fixtures::miniature_glimmer` with the
  `[N]` word-level tokenizer (`tests/vindex3_mutation.rs`). INFER runs the
  real KNN override there.
- **VINDEX2 compatibility**: same ops, journal and catalogue. The parity
  gate compares asserted rows from the same script. The new ops are added
  to V2's `apply_patch_unchecked` as to V3's `apply_unchecked`.
- **Remote**: forwarders unchanged.

## 8. Atomicity

Every refusal (shape, key resolution, D-5, collision, status, D-6, I-2
conflict) is decided with read-only borrows before the overlay, the
journal or the recording is touched. A refused statement or patch leaves
all of these unchanged: the overlay's KNN entries, slot overrides and
tombstones, the journal, the recording's op list and the epoch.

## 9. Acceptance gates (frozen at r2)

Tests: `crates/larql-lql/tests/fact_id_1/`, on the VINDEX3 fixture unless
noted. "Same fact" = the same `f_…` id. Each gate is satisfied by the
named tests passing and by nothing else.

| Gate | Requirement | Test |
|---|---|---|
| ID-1 | One identity across INSERT, SELECT, UPDATE, DELETE | INSERT prints id X v1; `asserted_facts()` and SELECT show X; UPDATE prints X v1 → v2; DELETE removes X; SELECT then shows no asserted block. |
| ID-2 | One SELECT with origin | A key with both an asserted fact and discovered rows prints both blocks. With no asserted match, output is byte-identical to the same SELECT before any INSERT. |
| ID-3 | Semantic mutation never touches discovered rows | Every slot's metadata, overrides and tombstones are snapshotted before and after semantic UPDATE/DELETE (success and refusal); equal. A key with only discovered rows is refused. |
| ID-4 | KNN UPDATE changes INFER | INFER on the canonical prompt: `[7]`; UPDATE to `[9]`; INFER: `[9]`, and the stored entry's `target_id` is 9. |
| ID-5 | Single-token rule (KNN only) | A two-token KNN target is refused on INSERT and on UPDATE. A legacy multi-token entry (from a written patch) lists `multi_token` and refuses UPDATE. A compose INSERT of the same target is not refused by D-5. |
| ID-6 | Reversible | Patch A inserts, patch B updates: removing B restores A's target in INFER and SELECT. Same for a delete. VINDEX3: a container fact deleted in a patch is restored by REMOVE PATCH. |
| ID-7 | Save → replay → reload | The catalogue (ids, targets, statuses, versions) and INFER after a script equal those of a fresh session APPLYing its saved patch. COMPILE then reopen: KNN facts present, `durability = container`, same ids. |
| ID-8 | VINDEX2 parity | On a VINDEX2 fixture, the same ops give the same ids, statuses and versions. A VINDEX2 REMOVE PATCH that drops container facts reports them `inconsistent` (C-11). |
| ID-9 | Compose guard | A keyed compose fact lists as `compose`; semantic UPDATE and DELETE are refused naming REMOVE PATCH, with state unchanged. COMPILE with one prints the identity-loss line. |
| ID-10 | No regression | Existing `larql-lql` and `larql-vindex` suites pass. Any test changed by D-5 or by the duplicate-INSERT refusal is listed in the PR. |
| ID-11 | Atomic failure | For each refusal path in §8, the state of §8 is unchanged. |
| ID-12 | Duplicate key | A second INSERT of a live key is refused. A legacy patch with two entries for a key loads; the fact is `ambiguous`, UPDATE refuses, DELETE removes both. |
| ID-13 | Patch conflicts (I-2) | APPLY of a patch inserting a live key is refused whole; one updating or deleting an absent key is refused; one that deletes then inserts is accepted. |
| ID-14 | Version semantics | One UPDATE is one version step. One compose INSERT spanning several slots is one version step. A container fact is version 1. |
| ID-15 | Old readers refuse new ops | A patch with an unknown op tag fails to parse. `delete_knn_fact` and `update_knn_fact` serialize under their own tags. |
| ID-16 | Journal across SAVE | INSERT, BEGIN/SAVE PATCH: the fact stays listed (`durability = session` before APPLY, since SAVE does not apply) and is live in INFER. REMOVE of an unrelated patch clears the journal and the fact goes, matching the store. |

## 10. Out of scope

FACT-PROMOTE-1; FACT-WSP-1; FACT-GRAPH-1; multi-token targets; compose
UPDATE/DELETE by identity; compose identity across COMPILE; SELECT
projection; addressing by fact id; FR3 for asserted rows; VINDEX2's
container-KNN loss on REMOVE PATCH (C-11) and COMPACT MINOR's unrecorded
removal (C-12), both only detected; remote-specific behaviour.
