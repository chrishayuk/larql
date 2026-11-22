# Gate B, B6: COMPACT and COMPILE acceptance contract

**Status:** **FROZEN r3** (2026-11-20). The owner ruled on D1 to D5 (§7).
Changing this contract needs an owner-approved revision.
**Implementation hold:** no B6 branch opens until #884, #885, #888 and #891
have cleared required CI. B6-COMPACT also waits for #891 to merge.
**Base:** `origin/main` `cce18787` (2026-11-20).

### Revision log

- **r1** — draft for review. Listed decisions D1 to D4.
- **r2** — the owner's rulings, frozen:
  - D1 = P, with promotion atomic at statement level, a set of required
    tests, and a fact-catalogue comparison.
  - D2 = refuse COMPACT MAJOR before any work.
  - D3 = durable publication. The sync sequence is defined. Failures before
    publication and durability failures after it are reported differently.
  - D4 = an empty allowlist, checked over the whole output tree.
- **r3** (2026-11-20) — owner ruling D5 on COMPILE INTO MODEL scope (§7,
  §8). Nothing else in r2 changes.
**Parent documents:** [`gate-b.md`](gate-b.md) r1 (frozen) defines the gate.
[`gate-b-audit-2026-11-18.md`](gate-b-audit-2026-11-18.md) records the B6
finding.
**Implementation:** two PRs, one for COMPACT and one for COMPILE (§6). Both
are accepted against this one contract.

## 1. Scope

| Verb | Backend | What it changes | Entry point on main |
|---|---|---|---|
| `COMPACT MINOR` | VINDEX2 | session: KNN (L0) entries become compose (L1) installs | `executor/compact.rs` `exec_compact_minor` |
| `COMPACT MAJOR` | VINDEX2 | session MEMIT store, plus `memit_store.json` in the **bound vindex directory** | `executor/compact.rs` `exec_compact_major` |
| `COMPACT INTO VINDEX` | VINDEX3 | a new container; refuses while the session holds any overlay state | `executor/lifecycle/compact_into.rs` |
| `COMPILE CURRENT INTO VINDEX` | VINDEX2 | a new vindex directory | `lifecycle/compile/into_vindex.rs` |
| `COMPILE CURRENT INTO VINDEX` | VINDEX3 | a new container (rewritten segments plus hard links to unchanged ones) | `lifecycle/compile/into_vindex_v3.rs` |
| `COMPILE INTO MODEL` | VINDEX2 | a new model directory | `lifecycle/compile/into_model.rs` |

Paths are relative to `crates/larql-lql/src/`.

A correction to the earlier report: COMPACT is not all VINDEX2. `COMPACT
MINOR` and `COMPACT MAJOR` are VINDEX2 only. `COMPACT INTO VINDEX` is the
VINDEX3 form.

**Out of scope:**
- B7, the verifier.
- How good MEMIT is as a solver. B6 judges only what COMPACT MAJOR writes and
  when.
- FACT-PROMOTE-1, which is VINDEX3 promotion and has its own contract.
- Any change to the `.vlp` op set. B6 adds no patch-format op. §3.1 uses only
  existing ops.
- A ledger-backed MEMIT op. That needs a contract of its own before COMPACT
  MAJOR can run again.

**Relation to FACT-WSP-1.** FACT-WSP-1 is VINDEX3 only. `COMPACT MINOR`,
`COMPACT MAJOR` and VINDEX2 COMPILE are not on its execution path. Gate B
still needs all of them closed before LARQL's mutation system is called safe
in general. FACT-WSP-1's admission depends only on the VINDEX3 rows: `COMPACT
INTO VINDEX` and VINDEX3 COMPILE. B6 reports the two groups separately so
that neither is taken as standing in for the other.

## 2. Definitions

- **Observable state** is these four things, each compared exactly:
  1. The session: `state_fingerprint()`. On VINDEX2 this also includes the
     override vector bits and the KNN entries, as in the B5 tests.
  2. The source container: the path, size and SHA-256 of every file under the
     bound directory, symlinks and hard links included.
  3. The destination path: absent, or every file's path, size and SHA-256.
  4. The output's parent directory, listed, so that a leftover staging
     directory shows.
- **Permitted nondeterministic metadata** is a closed list of manifest fields
  that may differ between two runs on identical input. The list is **empty**
  (D4). A field is added only by an owner-approved revision of this document,
  made before acceptance. The revision names the field and why it cannot be
  made deterministic. No test may skip, mask or normalise a field to make it
  pass. Where a timestamp is written, it is copied from the source or derived
  from the input.
- **Fact catalogue** is FACT-ID-1's asserted-fact catalogue
  (`Session::asserted_facts`): for each fact, its id, key, target, status and
  representation.
- **Independent replay** means a fresh `Session` that runs `USE` on the same
  base, then `APPLY` on each saved patch file in order, then the same
  statement.

## 3. Requirements by verb

R1 to R5 are the owner's five requirements:

- **R1** deterministic output;
- **R2** failure atomicity;
- **R3** no partial success;
- **R4** staged, validated publication;
- **R5** exact reconstruction.

### 3.1 COMPACT MINOR (VINDEX2)

**On main (by reading):**
- It calls `exec_insert` directly. That skips FACT-ID-1's `check_fact_insert`
  and the dispatcher's recording handling.
- Each promotion installs a compose `Insert` for a key that is still live as
  KNN, then calls `remove_by_entity_relation` and records no op.
- A per-entry error is caught, counted ("N failed") and the statement returns
  Ok.
- `advance_epoch` runs whatever happened.

As a result, the recording, if one is open, holds a keyed `Insert` for a key
that FACT-ID-1 I-2 says is live. A patch saved from it is refused at APPLY.
The KNN removal is in no record. FACT-ID-1 C-12 already marks the effect: the
catalogue flags the fact `inconsistent`.

**Ruling D1 = P: promote and record.** For each fact K, the recording holds
`delete_knn_fact(K)` followed by the keyed compose `Insert` op(s) for K, plus
any B3 re-fits. FACT-ID-1 I-2 accepts a delete followed by an insert of the
same key.

- **One atomic promotion.** The delete and the install form one promotion,
  and the whole statement is one transaction. No observable state ever shows
  K gone from KNN but not yet installed as compose. That covers the session
  state, the recording and the catalogue, both after the statement and after
  any failure. Atomicity comes from B5's statement-level snapshot, widened to
  the KNN store. No new op is added.
- **Engine order equals recorded order.** The engine removes the KNN entry
  and then installs. The live state is therefore produced by the same
  sequence a replay runs, and the install's capture never sees the old KNN
  override.
- **Identity and provenance.** K keeps its fact id. The only declared change
  is its representation, from `knn` to `compose`.
- **COMPILE loses compose identity.** FACT-ID-1 C-9 still holds: a compose
  fact's identity does not survive COMPILE. The COMPACT MINOR output names the
  promoted facts and says so.

**Further requirements:**
- **R3.** One statement is one transaction. If any entry cannot be promoted,
  the whole statement refuses. The refusal names the entry and the reason. No
  "N failed" success text remains.
- **R2.** A refusal or failure leaves the observable state as it was:
  overlay, KNN store, journal, recording, the B1 binding state and the epoch.
  It reuses B5's snapshot and widens it to cover the KNN store. If COMPACT
  opened a recording, the recording is closed again (the B5 rule).
- **R1.** Entries are promoted in a defined order: by layer, then entity,
  then relation, by fact key. The order is not that of map iteration. Two
  runs on identical sessions record identical ops.
- Before anything is changed, the statement refuses if any entry's catalogue
  status is not `active` (`multi_token`, `unverified`, `ambiguous`,
  `inconsistent`). Promoting those would carry a known defect into the
  compose representation.
- An empty L0 stays a successful no-op that records nothing.
- There is no undo statement. Rollback is by not applying the patch, or by
  REMOVE PATCH under B2's rules.

**Catalogue comparison.** This is a second check, separate from comparing
vectors. Compare the fact catalogue before and after a successful COMPACT
MINOR. The two must be identical except that each promoted fact's
representation changes from `knn` to `compose`. No fact may:
- disappear;
- gain a second id;
- change key or target;
- become `ambiguous`, `inconsistent` or `degraded`.

The same comparison holds between the live session and an independent replay
of its saved patch.

**Required tests** (permanent, fail before and pass after where the defect
exists on main):
1. **Identity:** the fact id, key and target are the same before and after
   promotion, by the catalogue comparison.
2. **Replay:** live and replayed states are bit-identical: session
   fingerprint, override bits and KNN store.
3. **A failure between delete and install** restores the whole previous
   state. This uses fault points after the KNN removal and after the k-th
   promotion (§4).
4. **APPLY:** the saved patch applies to a fresh session bound to the same
   base without an I-2 refusal.
5. **REMOVE PATCH** of the promotion patch restores the original KNN
   representation, by catalogue and by KNN store. The fixture's KNN fact sits
   in the container or in an earlier applied patch, so B2 does not refuse the
   removal.
6. **Second attempt:** promoting the same fact again never duplicates it.
   Once promoted, K is not in L0, so a second COMPACT MINOR records nothing
   for K. If K is somehow live in both representations (an `inconsistent`
   or `ambiguous` status), COMPACT MINOR refuses before any change.

### 3.2 COMPACT MAJOR (VINDEX2)

**On main (by reading):**
- The cycle is added to the session's MEMIT store before anything is
  persisted.
- `save_memit_store(<bound vindex dir>)` then writes `memit_store.json` into
  the source container.
- A write failure becomes a `warning:` line under a "complete" message.
- The epoch advances.
- No `.vlp` op records the cycle, so R5 cannot hold: no replay reproduces it.

**Ruling D2 = refuse.**

- COMPACT MAJOR refuses before any work: before loading weights, before the
  solve, and before any filesystem write, epoch advance, MEMIT store change or
  recording change.
- The refusal text says that a MEMIT cycle has no ledger representation, so
  its result could not be replayed or reconstructed (R5). It also says that
  restoring the verb needs a ledger-backed MEMIT op under its own contract.
- No session-only exception is allowed.
- The test pins the full refusal text. It checks that all four parts of §2's
  observable state are unchanged, the bound directory's file hashes included.
  The existing tests of a successful MAJOR become refusal tests.
- A `memit_store.json` already in a vindex is still read on `USE`. B6 does
  not change rehydration.

### 3.3 COMPACT INTO VINDEX (VINDEX3)

On main it refuses while there is overlay state, and writes through
`format::fresh_output::write_fresh_directory`.

Required:
- **R1:** compacting the same container twice into two destinations gives
  byte-identical trees.
- **R2:** the session fingerprint and the source tree are unchanged on success
  and on every injected failure (§4).
- **R4:** publication follows §3.6. The staged output is validated before the
  rename. It must reopen as a container, and the logical DIFF against the
  source must be empty. A validation failure removes the staging and leaves
  no output.
- **R5:** the reopened output answers the existing DIFF and INFER oracles in
  `tests/vindex3_compact.rs` the same as the source.

### 3.4 COMPILE (VINDEX2 INTO VINDEX, VINDEX3 INTO VINDEX, INTO MODEL)

On main all three write through `write_fresh_directory`, which:
- refuses a destination that is the source, nests with it, is a symlink, or
  is non-empty;
- writes into a hidden sibling staging directory;
- promotes it with one `rename`.

Its own documentation states two limits: it is "atomic, not durable" (no
`fsync`), and it does not defend against a concurrent adversary.

Required:
- **R1:** compiling the same state twice, into two destinations, gives
  byte-identical trees. A pass needs two separate sessions, not one session
  compiled twice. This covers every file, `index.json` included (§3.5).
- **R2:** after a failure at any fault point in §4 before the rename (§3.6):
  - the destination is as it was;
  - no staging directory is left, except when the injected fault is the
    cleanup itself (then the error names the leftover path);
  - the source tree is unchanged;
  - the session fingerprint is unchanged.

  COMPILE borrows the session immutably today. The test pins that, so a later
  change that mutates the session fails it.
- **R4:** publication follows §3.6. A validation step runs before the rename.
  The staged output must
  reopen with the backend's loader, and every checksum in its manifest must
  match its bytes. On VINDEX3 it must also show that each hard-linked segment
  is byte-equal to its source. The validation is new; on main nothing reopens
  the staged output.
- **R5:** the output of a live session compile and the output of an
  independent replay compile are byte-identical. On VINDEX3 the scope is what
  B7 does not already cover. B7 owns the numerical check on the weight
  tensors.
- **VINDEX3 hard links.** The bake hard-links unchanged segments, so source
  and output share inodes. B6 requires a test showing that nothing LARQL
  writes later edits a shared inode in place. Every writer of a container
  file must replace it by rename. Otherwise the hard links are replaced by
  copies. An in-place write would change the source as a side effect, which
  the owner's MAJOR point forbids.
- **Publication** follows §3.6. Its guarantees must be stated in user-facing
  docs, not only in module docs.

### 3.6 Durable publication (ruling D3)

These rules apply to COMPACT INTO and to every COMPILE target, on supported
local filesystems. The supported set is named in the user-facing docs; macOS
APFS and Linux ext4/xfs at minimum.

**Sequence:**
1. Write every staged file.
2. `fsync` each file, flushing its contents and the metadata needed to read
   it.
3. `fsync` the staging directory and every directory inside it.
4. Validate the staged output (the R4 checks above).
5. Atomically rename staging to the destination. The staging directory is a
   sibling of the destination, so the rename never crosses filesystems.
   - On Unix the rename replaces an empty destination atomically and fails on
     a non-empty one.
   - Elsewhere the empty destination is first removed non-recursively. A crash
     between the removal and the rename leaves no output, never a partial one.
6. `fsync` the destination's parent directory.

**Return values.** Success means all six steps completed. Failures fall into
two classes, and the error says which:
- **Before publication (steps 1 to 5 fail).** The destination is as it was,
  the staging directory is removed, and the source and session are unchanged.
  This is R2 in full.
- **After publication (step 6 fails).** The rename has happened, so the
  destination may already be visible. B6 does **not** claim a rollback here.
  The error says that the output was published at `<path>` but its
  durability is not confirmed. Nothing is deleted. Recovery is to re-run the
  parent sync, or remove the output by hand. The test injects this fault and
  asserts that exact outcome, not "destination absent".

**Interrupted runs.** An interrupted run leaves a hidden staging sibling,
recognisable by `STAGING_INFIX`. It is never promoted, and nothing deletes it
automatically. Recovery is to delete it by hand.

**Limits that remain documented**, unless an implementation closes them:
- `write_fresh_directory` does not defend against concurrent writers.
- The destination check and the rename are separate steps. A process that
  swaps a parent directory for a symlink in between can redirect the rename.
  The one race that is closed is a destination that gains contents: the
  promotion then fails rather than deleting them.

### 3.5 Canonical manifests

- `IndexConfig.checksums` is a `HashMap`, so `index.json`'s key order varies
  between runs. It becomes an ordered map (`BTreeMap`). The same applies to
  any other map type reachable from a serialized manifest (`index.json`, the
  VINDEX3 manifest and segment tables, and `memit_store.json` if D2 keeps it).
- **Test, independent of insertion order.** Build logically identical
  manifests by inserting the same entries in several different orders. They
  must serialize to identical bytes.
- **Whole-tree digests.** R1 is judged over the whole output tree, not
  `index.json` alone. Every corresponding file in two outputs must have the
  same SHA-256, and the two trees must list the same files. Add a compile-time or unit check that
  lists the map types in each manifest struct, so that a new `HashMap` field
  fails it. This is the owner's "fix the field, miss the next collection"
  guard.

## 4. Fault injection

A test-only fault hook, like B5's `InstallFaultPoint`, at these points:

COMPACT MAJOR refuses (D2), so it has only the refusal test of §3.2.

| Point | COMPACT MINOR | COMPACT INTO / COMPILE | Expected outcome |
|---|---|---|---|
| before any work | yes | yes | unchanged |
| after a KNN removal, before its install | yes | n/a | unchanged |
| after the k-th promotion (k = 1, last) | yes | n/a | unchanged |
| after staging is created | n/a | yes | unchanged |
| after a partial write into staging | n/a | yes | unchanged |
| a file sync fails (step 2) | n/a | yes | unchanged |
| the staging directory sync fails (step 3) | n/a | yes | unchanged |
| validation fails (step 4) | n/a | yes | unchanged |
| the rename fails (step 5) | n/a | yes | unchanged |
| cleanup after a failure fails | n/a | yes | unchanged except the named staging leftover |
| the parent sync fails (step 6) | n/a | yes | published, durability-unconfirmed error (§3.6) |

"Unchanged" means the test asserts §2's observable state against the
snapshot taken before the statement. It then runs a retry with the fault removed and
asserts that the result is identical to that of a session that never failed
(the B5 retry property).

## 5. Tests and evidence

- Every fix comes with a permanent test in the owning crate's `tests/`
  directory. Each test is shown to fail with the fix reverted (fail before)
  and to pass with it applied (pass after). The PR body records the
  fail-before output.
- Determinism is asserted on bytes (SHA-256 of every file), not on
  load-and-compare.
- The B6 suites run in the full-suite check with each `cargo` exit code
  recorded separately. A filtered or piped output is not evidence.
- Coverage is at least 90% per changed file.
- B6 tests run on VINDEX3 first, VINDEX2 second. They are reported as the two
  groups of §1.

## 6. PR split

| PR | Contents | Depends on |
|---|---|---|
| B6-COMPILE | §3.3 to §3.6, and §4 for COMPILE and COMPACT INTO: deterministic manifests, validated staging, durable publication, exact replay, VINDEX3 hard-link safety | none (larql-vindex manifest and `fresh_output`, plus lql compile) |
| B6-COMPACT | §3.1, §3.2: atomic recorded promotion, truthful refusals, fact identity, refusal of COMPACT MAJOR | the B5 snapshot (#891, merged first) |

The two surfaces share only `fresh_output` (changed by B6-COMPILE, not used
by B6-COMPACT) and the session snapshot (B5). Neither opens until the current
Gate B PRs have cleared CI. B6 closes when both have merged and the B6 row of the audit ledger
is PASS on each group of §1.

## 7. Owner rulings (2026-11-20)

| Decision | Ruling | Where |
|---|---|---|
| D1, COMPACT MINOR | **P**: promote and record as `delete_knn_fact(K)` then the keyed `Insert(K)`. The pair is one atomic promotion at statement level. No new op. | §3.1 |
| D2, COMPACT MAJOR | **Refuse** before any work. No session-only exception. | §3.2 |
| D3, durability | **Require fsync**, by the sequence in §3.6. Failures before and after publication are reported differently. | §3.6 |
| D4, nondeterminism | **Empty allowlist**, over the whole output tree. An amendment is needed before acceptance; no runtime exclusions. | §2, §3.5 |
| D5, COMPILE INTO MODEL | A successful export is **out of B6 scope**; its failure must be atomic and reported. See §8. | §3.4, §8 |

## 8. Acceptance record

### 8.1 COMPILE INTO MODEL (ruling D5)

On `741ecdab` the export fails for every input before it writes: its weight
writer expects an `index.json` in the fresh output directory. Making it work
is feature work, not mutation safety, so it stays outside B6.

| Requirement | COMPILE INTO MODEL |
|---|---|
| Successful export | Out of B6 scope |
| Failure leaves the source unchanged | Required |
| Failure leaves the session unchanged | Required |
| No partial publication at the destination | Required |
| Deterministic successful output (R1) | Not applicable until the export works |
| Replay of a successful output (R5) | Not applicable until the export works |
| The export defect itself | Open, tracked separately |

The exclusion applies to INTO MODEL only. COMPILE INTO VINDEX and COMPACT
INTO VINDEX work, and must meet every B6 requirement that applies to them.

A regression test runs the failing export on both backends and checks the
source, the session, the destination and the destination's parent. When the
export is fixed that test fails, and R1 and R5 then apply in full.

### 8.2 Coverage exception

`crates/larql-lql/src/executor/lifecycle/compile/into_model.rs` is below the
90% per-file floor (77.7% on the B6-COMPILE head). The exception is narrow:
- it covers only the lines after the export's existing failure point, which
  no input reaches;
- the lines B6 changed in that file (publication through the validated
  staging path) are covered;
- it does not make the file compliant, and it ends when the export is fixed.

No other file in B6 has a coverage exception.

### 8.3 COMPACT INTO and an unreadable ledger

Before B6, COMPACT INTO copied a weight-patch ledger as bytes and published
whatever it found. Since B6 the staged container's structural check refuses
one that cannot be read, before publication. A missing ledger and an
unreadable one are each covered by their own negative test.
