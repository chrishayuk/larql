# Gate B, B6: COMPACT and COMPILE acceptance contract

**Status:** DRAFT r1, for owner review. Nothing here is frozen yet. §7 lists
the decisions the owner has to make before the freeze.
**Base:** `origin/main` `cce18787` (2026-11-20).
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
- Any change to the `.vlp` op set beyond what §3.1 option P uses. It uses
  only existing ops.

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
  that may differ between two runs on identical input. The list **starts
  empty**. A field is added only by an owner-approved revision of this
  document, which names the field and why it cannot be fixed. Wall-clock
  timestamps are not added by default. Where one is written, the first choice
  is to copy it from the source or derive it from the input.
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

**Required whatever §7 D1 decides:**
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

**Choice (owner decision D1):**
- **Option P: promote, recorded.** For each fact K the recording holds
  `delete_knn_fact(K)` followed by the keyed compose `Insert` op(s) for K,
  plus any B3 re-fits. FACT-ID-1 I-2 accepts a delete followed by an insert
  of the same key.
  - The engine runs in recorded order: remove the KNN entry, then install. The
    live state is therefore produced by the same sequence a replay runs, and
    the install's capture never sees the old KNN override.
  - The fact keeps its id. Its catalogue origin changes from `knn` to
    `compose`, and the adjacent ops in one recording record that transition.
  - There is no undo statement. To roll back, do not apply the patch, or
    REMOVE PATCH. B2's refusal rules apply as they do to any other edit.
  - FACT-ID-1 C-9 still holds: a compose fact's identity does not survive
    COMPILE. Promotion therefore trades the durable identity of the KNN
    representation for a compose install. The COMPACT output must say so,
    naming the facts.
- **Option R: refuse.** COMPACT MINOR refuses whenever L0 is non-empty and
  points to FACT-PROMOTE-1. That retires the verb on VINDEX2 until a
  promotion contract exists.

### 3.2 COMPACT MAJOR (VINDEX2)

**On main (by reading):**
- The cycle is added to the session's MEMIT store before anything is
  persisted.
- `save_memit_store(<bound vindex dir>)` then writes `memit_store.json` into
  the source container.
- A write failure becomes a `warning:` line under a "complete" message.
- The epoch advances.
- No `.vlp` op records the cycle, so R5 cannot hold: no replay reproduces it.

**Required whatever §7 D2 decides:**
- **Never write into the bound container.** No file under the bound
  directory is created, changed or removed. Requirement 2 of §2 is checked
  before and after, on success and on failure.
- **A failed write is a failure.** No "complete" text accompanies it, and the
  session MEMIT store is left as it was (R2, R3).

**Choice (owner decision D2):**
- **Option M-refuse (recommended).** COMPACT MAJOR refuses before any work.
  The text says MEMIT cycles have no ledger representation, so the result
  could not be reconstructed (R5). MEMIT is outside FACT-WSP-1 and has no
  verifier coverage. Refusing is the one behaviour that meets R1 to R5 today.
  The existing tests of a successful MAJOR become refusal tests.
- **Option M-session.** The cycle stays in the session only and is written
  only into a COMPILE output, through §3.4's staging. That removes the source
  write, but R5 still fails until a cycle is a ledger op. B6 would have to
  record that as an explicit, owner-accepted exception.

Either way, a `memit_store.json` already in a vindex is still read on `USE`.
B6 does not change rehydration.

### 3.3 COMPACT INTO VINDEX (VINDEX3)

On main it refuses while there is overlay state, and writes through
`format::fresh_output::write_fresh_directory`.

Required:
- **R1:** compacting the same container twice into two destinations gives
  byte-identical trees.
- **R2:** the session fingerprint and the source tree are unchanged on success
  and on every injected failure (§4).
- **R4:** the staged output is validated before promotion. It must reopen as
  a container, and the logical DIFF against the source must be empty. A
  validation failure removes the staging and leaves no output.
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
- **R2:** after a failure at any fault point in §4:
  - the destination is as it was;
  - no staging directory is left, except when the injected fault is the
    cleanup itself (then the error names the leftover path);
  - the source tree is unchanged;
  - the session fingerprint is unchanged.

  COMPILE borrows the session immutably today. The test pins that, so a later
  change that mutates the session fails it.
- **R4:** a validation step runs before promotion. The staged output must
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
- **Publication guarantees** must be stated in user-facing docs, not only in
  module docs:
  - Atomic on one filesystem. The staging directory is a sibling of the
    destination, so the rename never crosses filesystems.
  - On Unix, the rename replaces an empty destination atomically and fails on
    a non-empty one.
  - Elsewhere, the empty destination is removed and then renamed into. A
    crash between the two leaves no output, never a partial one.
  - An interrupted run leaves a hidden staging sibling, recognisable by
    `STAGING_INFIX`. It is never promoted, and nothing deletes it
    automatically. Recovery is to delete it by hand.
  - Durability against power loss is D3.

### 3.5 Canonical manifests

- `IndexConfig.checksums` is a `HashMap`, so `index.json`'s key order varies
  between runs. It becomes an ordered map (`BTreeMap`). The same applies to
  any other map type reachable from a serialized manifest (`index.json`, the
  VINDEX3 manifest and segment tables, and `memit_store.json` if D2 keeps it).
- **Test, independent of insertion order.** Build logically identical
  manifests by inserting the same entries in several different orders. They
  must serialize to identical bytes. Add a compile-time or unit check that
  lists the map types in each manifest struct, so that a new `HashMap` field
  fails it. This is the owner's "fix the field, miss the next collection"
  guard.

## 4. Fault injection

A test-only fault hook, like B5's `InstallFaultPoint`, at these points:

| Point | COMPACT MINOR | COMPACT MAJOR (if D2 = M-session) | COMPACT INTO / COMPILE |
|---|---|---|---|
| before any work | yes | yes | yes |
| after the k-th promotion (k = 1, last) | yes | n/a | n/a |
| after staging is created | n/a | yes | yes |
| after a partial write into staging | n/a | yes | yes |
| validation fails | n/a | yes | yes |
| promotion (`rename`) fails | n/a | yes | yes |
| cleanup after failure fails | n/a | yes | yes |

At each point the test asserts §2's observable state against the snapshot
taken before the statement. It then runs a retry with the fault removed and
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
| B6-COMPILE | §3.4, §3.5, §4 for COMPILE and COMPACT INTO | none (larql-vindex manifest plus lql compile) |
| B6-COMPACT | §3.1, §3.2 | the B5 snapshot (#891, merged first) |

The two surfaces share only `fresh_output` (read, not changed, by
B6-COMPACT) and the session snapshot (B5). Either PR may land first once #891
is merged. B6 closes when both have merged and the B6 row of the audit ledger
is PASS on each group of §1.

## 7. Owner decisions before the freeze

- **D1, COMPACT MINOR:** option P (promote, recorded as `delete_knn_fact`
  then the keyed `Insert`) or option R (refuse, defer to FACT-PROMOTE-1).
  - Recommendation: **P**. It keeps the verb, uses only existing ops, keeps
    the fact id, and is replayable under I-2.
  - Choose R if promotion on VINDEX2 is not worth the C-9 identity loss at
    COMPILE.
- **D2, COMPACT MAJOR:** M-refuse or M-session.
  - Recommendation: **M-refuse**. It is the one option that meets R5.
- **D3, durability:** whether B6 requires `fsync` of the staged files and
  directory before the rename, and of the parent after it, or keeps "atomic,
  not durable" as a documented limit.
  - Recommendation: **require it**. The cost is one sync per file at the end
    of a compile.
- **D4, permitted nondeterministic metadata:** confirm the list starts empty.
  The first B6-COMPILE test reports any field that differs. Each such field
  comes back to the owner as an amendment, not a silent exception.
