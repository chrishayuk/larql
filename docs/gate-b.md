# Gate B (engine) and Gate S (server): gate definitions

**Status:** r1, definitions FROZEN 2026-11-19 by owner decision. The fixes are
separate PRs and must not change these definitions. Any change to a definition
is a new revision of this document, made on its own and before the fix that
needs it.
**Audited against:** main `3b5dd3d2`. The findings are in
[`gate-b-audit-2026-11-18.md`](gate-b-audit-2026-11-18.md) and the reproducers
are in [`probes/gate-b/`](../probes/gate-b/).
**Supersedes:** the "Gate B" tags in
[`review-2026-11-10-findings-ledger.md`](review-2026-11-10-findings-ledger.md).
That ledger's EDIT-5, ID-2 and ID-4 were fixed by #801, #802 and #804 and are
graded PASS in the audit.

---

## 1. Why the gates exist

A VLP patch is meant to be a complete, portable description of a change to a
model. The audit found the opposite on main:
- REMOVE PATCH can leave a different effective weight state from the one the
  remaining patches describe (B2).
- A saved patch can replay to different vectors from the session that recorded
  it (B3).

No weight-editing result can be trusted while that holds. That includes
FACT-WSP-1, which will put weight edits behind semantic UPDATE.

**The governing invariant:** the recorded patches and the effective model state
always agree.

## 2. Grades

Every obligation gets exactly one grade.

| Grade | Meaning |
|---|---|
| `PASS` | Code that implements the invariant **and** a version-controlled test that fails if the invariant breaks. Reading the code alone never earns a PASS. |
| `FAIL` | A concrete input → wrong state. The finding says whether it was shown by execution or by reading the code. |
| `NOT_EVIDENCED` | The code may be right, but no test proves it, or the auditor could not tell. |
| `OUT_OF_SCOPE` | Excluded from a named consumer's admission, such as FACT-WSP-1. The entry must say which scope excludes it and what refuses it at runtime. `OUT_OF_SCOPE` never turns a `FAIL` into a pass: the gate itself stays open until the obligation is `PASS`. |

**Evidence rule.** Where the representation supports it, equivalence is
checked **byte for byte**: overlay vectors, override maps, KNN entries, saved
patch bytes and compiled tensors. Semantic checks such as INFER top-k and
DESCRIBE come in addition, never instead. Identical answers alone do not detect
drift in the weights underneath.

**Regression rule.** Each fix PR turns the failing audit probe into a permanent
test inside the crate's own `tests/` folder. The probe in `probes/gate-b/` is
deleted only by the PR that adds that test.

## 3. Gate B: local engine

Scope: local execution on both backends, VINDEX2 (`PatchedVindex`) and
VINDEX3 (`KnowledgeOverlay`). Covered: the patch lifecycle, reconstruction of
the effective state, rollback, and independent verification.

| # | Obligation | Acceptance condition |
|---|---|---|
| **B1** | USE isolation | Re-USE of the same path, or a USE of another vindex, leaves the session in the same state as a fresh session bound there. Every cache that derives from the binding is reset or rebuilt: `installed_edges`, `raw_install_residuals`, `decoy_residual_cache`, `relation_resolver`, the counters, and anything added later. No later statement writes vectors that no patch records. An open recording is never discarded silently. |
| **B2** | DELETE completeness and exact REMOVE PATCH | Removing a patch restores the exact prior effective state. This covers meta, gate, up and down overrides on base slots and on inserted slots, tombstones, KNN entries including the vindex's own `knn_store.bin`, and INFER and COMPILE results. DELETE removes the whole structure it targets. |
| **B3** | Exact replay | A fresh session that applies the saved patch(es) reaches the live session's effective state byte for byte. This holds whether the facts went into one patch or several, and whether compose operations share a layer or not. If a later operation changes an earlier operation's effective contribution, the recorded representation of the earlier one is updated as well. |
| **B4** | Duplicate entity–relation assertions | A second assertion of a live key either replaces the first exactly or is refused. It never succeeds without effect. |
| **B5** | INSERT atomicity | Every failed local INSERT leaves the overlay, the patch recording, the session journal, slot allocation and every observable state unchanged. A refused INSERT does not open an automatic patch recording. |
| **B6** | COMPACT and COMPILE truthfulness | A failure leaves no partial change. A reported success is complete: "N failed" is not success. Base artifacts, including the bound vindex directory, are never modified unless the statement says so. Every state change is recorded. |
| **B7** | Verifier closure | Independent verification (`verifier/`) covers every operation and every combination of operations admitted to the FACT-WSP-1 VINDEX3 path. See §6. |

These rows were already PASS on main and must stay PASS: UPDATE refusals,
DELETE refusals, APPLY PATCH whole-patch validation, and MERGE atomicity.

**Fix order (owner, 2026-11-19), one PR each:** B2 (EDIT-9), B3, B1, B5, B6,
then B7. B4 is closed by FACT-ID-1 (#880) when it merges.

## 4. Gate S: server

This is a separate gate for **ownership** reasons. It is not a safety
exemption. Scope: remote mutation, and isolation across sessions, models and
transports.

| # | Invariant |
|---|---|
| **S1** | Every mutation reaches exactly the model and the session it was authorised for. A session bound to model A never applies a patch that was checked against model B. |
| **S2** | No partial operation reports success. A partial or zero insert is an error. |
| **S3** | Remote INSERT and UPDATE leave an observable patch record (SHOW PATCHES) and can be rolled back. A remote UPDATE means the same thing as a local UPDATE. |
| **S4** | Every read path that knows about sessions uses the session's effective state: the HTTP browse routes, INFER, EXPLAIN INFER, WebSocket `/v1/stream`, gRPC, walk-ffn and the OpenAI-compatible routes. |
| **S5** | Concurrent operations preserve the slot-allocation and mutation invariants, and a test that runs operations concurrently shows it. |
| **S6** | Unsupported or unknown operations and options are refused, never silently ignored. Examples: ALPHA and MODE on a remote INSERT; ORDER BY and NEAREST TO on a remote SELECT. |

**No remote semantic-to-weight UPDATE ships until Gate S passes.**

## 5. Separate query-correctness fix (Q-1)

An ORDER BY on an unknown field is silently ignored (`select/edges.rs`), and
ORDER BY on SELECT FROM FEATURES/ENTITIES is dropped as well. This is tracked
as its own fix, outside both gates, **provided** ORDER BY cannot affect which
rows a mutation targets. If any mutation statement ever orders or limits the
rows it targets, Q-1 moves into Gate B.

## 6. FACT-WSP-1 scope and admission

FACT-WSP-1 covers **local VINDEX3 only**. Its contract may be drafted now.
Weight mutation behind semantic UPDATE is not implemented until every Gate B
obligation that applies to VINDEX3 is `PASS` and §6.2 holds.

### 6.1 Scope

| Operation | FACT-WSP-1 |
|---|---|
| Local VINDEX3 weight UPDATE | In scope once Gate B and the verifier pass |
| Local VINDEX3 weight DELETE / rollback | Only with a verified inverse or exact restoration |
| VINDEX2 weight editing | Out of scope; refused explicitly at runtime |
| Remote weight mutation | Blocked until Gate S passes; refused at runtime, so the feature cannot be reached through any remote route that has not passed |
| KNN assertion UPDATE (FACT-ID-1) | A separate operation with its own provenance and verification; unchanged |
| Compose identity mutation | Still refused under FACT-ID-1 D-6 |

A KNN-assertion UPDATE and a weight UPDATE remain distinguishable operations,
with different provenance and different verification requirements. A semantic
UPDATE becomes a weight edit only when the caller explicitly selects a
supported mutation target and the transformation can be verified
independently.

**VINDEX3-only does not exempt the engine defects.** The VINDEX3 instances of
B1, B3, B5 and B6 must be PASS. EDIT-9 (B2) and the loss of `knn_store.bin`
are VINDEX2-only on main, because the VINDEX3 REMOVE PATCH is tested correct.
They are graded `OUT_OF_SCOPE` for FACT-WSP-1 admission and stay `FAIL` in
Gate B until they are fixed.

### 6.2 What the independent verifier must establish

1. The requested semantic target matches the intended tensor, layer and
   operation.
2. The emitted patch is structurally valid.
3. Its effective numerical result matches an independent reconstruction,
   within frozen numerical tolerances.
4. Several edits on one tensor compose according to the declared semantics.
   This includes several ADD LOW RANK ops on one tensor, which are allowed
   today but untested.
5. Failed, removed and replayed edits preserve the invariants of Gate B.
6. The VINDEX3 knowledge bake is verified independently if FACT-WSP-1 uses
   it. Otherwise it stays outside the admitted operation set and is refused.

## 7. Choices I made where the decision did not say

These choices were made while writing the definitions down; amend them by
revision if they are wrong:
- B2 counts the VINDEX2 `knn_store.bin` wipe on REMOVE PATCH as part of B2,
  not a separate obligation.
- B1 lists `relation_resolver` and the counters among the caches to reset,
  because the audit found they survive USE.
- B5 counts the automatic patch recording opened by a refused INSERT as an
  INSERT atomicity failure.
- Gate B's own grading covers both local backends. FACT-WSP-1 admission reads
  only the VINDEX3 rows (§6).
