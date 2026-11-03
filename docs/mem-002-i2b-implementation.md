# MEM-002 I2b: what was implemented, and what it does not establish

**Class: IMPLEMENTATION RECORD — NOT A FREEZE.** It records how I2b differs
from the accepted proposal ([`mem-002-i2-proposal.md`](mem-002-i2-proposal.md)
§4), the maintainer's decisions of 2026-11-02 and the limits every result
carries. Nothing here qualifies a protocol, and no calibration or
qualification ran for it. The source inventory is on #674.

## 1. The claim, and its limit

Three things stay separate:

| | Question | After I2b |
|---|---|---|
| 1 | What was the plan built from? | **Not established.** Read by path before the store exists |
| 2 | Which opened files did execution read? | Monitored, through retained descriptors |
| 3 | Did those files hold the expected bytes throughout? | Only as far as metadata detects change, and as far as an independently supplied manifest's verification goes |

**I2b alone does not establish complete verification-to-execution identity.
That remains a freeze gate,** unless the eventual protocol explicitly narrows
its identity claim and independently justifies doing so.

## 2. Monitored and not bound

**Monitored:** each selected segment file, mapped or copied. The store opens
it once and keeps the descriptor. Its header, every copied read
(`load_raw`, `load_raw_range`) and every mapping go through that descriptor,
and its `FileIdentity` is taken from that descriptor, never from a path.

**Not bound:** `index.json`, `system_graph.json`, plan-time segment headers,
the auxiliary reference table, representation attestations and other reads
made by path before the store exists. `index_json_sha256` comes from a separate
open from the one that was parsed.

Selected headers are re-read through the retained descriptors. **That does not
show the planner read identical bytes,** and every record carrying identity
facts says so in its `scope` block. A stream that ended before any identity was
judged has no such block, and no facts to qualify.

**Change detection is not proof of content.**
- A descriptor binds an inode, not its bytes: an in-place write shows through a
  live mapping and through copied reads.
- Metadata can be restored, and a file replaced and restored between checks may
  go unseen.
- Extended attributes and hard links move the ctime with no content change,
  which is a false invalidation.
- Writable mappers are forbidden by protocol and not enforced here.
- No payload is hashed by a run. That stays outside the timed experiment, in the
  separate manifest producer.

**Distributed workers are unsupported for verified campaign evidence.** Their
reads use a retained descriptor, but no evidence path records their identity.
`map_compiled` is out of scope: the residency curve does not reach it, though
`measure`'s teacher-forced overlay does.

## 3. What a run does

| Situation | Campaign attempt | Diagnostic run |
|---|---|---|
| Identity capture failure, or no segment opened | Records the reason and stops before the next timed region | Records and continues |
| Change since open, found before binding or at admission | Same | Records and continues |
| Manifest `mismatch` or `absent`, or differing index digest | Same | Records and continues |
| Change found after the run | Keeps the whole attempt, marks it `invalid`, exits non-zero | Records and continues |
| No manifest supplied | May execute, class `unverified` | Class `diagnostic` |
| Threshold rejection | Records it and continues with latency sealed: **not** invalid evidence | Not applicable |
| Manifest supplied but unreadable | Refused, on the record | Refused, on the record if a stream was requested |
| `--representation-source transient` | Refused: its precision map comes from an unbound pack header | Allowed |

Identity is judged **before binding, at admission and after the run**, and each
judgement is on the record with its per-segment facts.

**The record before binding carries no clock reading.** The next record is the
binding observation, so a time there would date the bind that `--sealed-latency`
seals. File creation times can still approximate the bind, as they could before
I2b: that stays a matter for the protocol's access rules.

An attempt stopped before its passes finishes its sealed file and puts the digest
on `attempt_end`, since it has no `after_run` header.

`evidence_class` is the run's own reading of identity: `identity_consistent`,
`unverified`, `invalid` or `diagnostic`. `identity_consistent` means the monitored
facts agree with the manifest; it does not mean verified. On `attempt_end`,
`status` separately says whether the attempt completed, so an attempt with
`status: error` is unusable whatever its class. A campaign ended before any
judgement is `invalid`. **The evaluator must recompute it
and independently reject invalid and unverified attempts,** and a successful run
cannot promote an `unverified` attempt.

The store's read counters are recorded as `post_admission_reads`. They are
diagnostic only, and show nothing about readers outside the store.

## 4. Departures from the proposal

| Proposal | Implemented | Why |
|---|---|---|
| Identity "per mapped segment" (§4.2) | Every selected segment, with `mapped` and `copied` | Only per-expert banks are mapped; other weights are copied |
| "Identity is recorded beside the decision… the evaluator voids" (§3.1) | Also fails closed at run time | Maintainer's decision |
| Rename-over "keeps the old identity" (§4.6) | Test pins: inode kept, ctime moved, `nlink` 0, path names another inode | Observed on APFS; the stated behaviour was wrong |
| No link count | `nlink` added, not part of the manifest key | Maintainer approved; the manifest format has no such field |
| Schema unaddressed | `larql.residency-observation/4` | A field's meaning changed |
| Linux unaddressed | Volume UUID null with `platform` named; a null cannot match a manifest | Maintainer's decision |
| `map_compiled` changed too (§4.1) | Unchanged | Not reached by the curve |
| Non-unix gives identity null and `unverified` (§4.6) | A capture failure, which stops a campaign | Maintainer's fail-closed decision |
| Unreadable manifest unaddressed | Refuses diagnostic runs too | An operator who supplied evidence isn't silently given none. **Differs from "diagnostic continues"; flagged for the maintainer** |
| Seek then read | Positional reads | One shared descriptor would otherwise race, silently |
| By-path fallback for copied reads (§4.2a) | None left | Every store read uses the retained descriptor |

The manifest is read and compared. Its producer remains separate and
unauthorised, so tests use manifests written by hand. A manifest that is not a
`pre` graph verification, lacks a top-level or `verifier` field the format
requires (their contents, and `identity.dev`, are not checked), or whose computed
and declared hashes do not both equal the container index's hash, verifies
nothing.

Comparison key: volume UUID, inode, size, mtime and ctime. Device id and link
count are excluded. The volume UUID is compared exactly, as uppercase, so a
producer writing another case fails closed. The manifest may list segments this
run did not select; they are ignored, including one the store recorded as absent.

The manifest's read and hash and an fsync of the stream now happen before the bind
timer starts. Neither reads a payload.

## 5. A new binding baseline

**A bind time from a binary before I2b and one from after it are different
implementations' results.** Removing repeated opens and the second header parse
can change binding latency with no change to what the model computes. It is not
evidence for or against any memory policy. Records carry
`binding_implementation: single_descriptor_i2b`.

Measured once on 2026-11-03, on a loaded machine with a warm page cache, with
five alternating `--sealed-latency` runs per binary (`354c3b93` and this
change):

| Container | Pre-I2b bind, median (ms) | Post-I2b bind, median (ms) |
|---|---|---|
| Per-expert miniature fixture | 2.3 | 0.6 |
| `oute-mamba2attn-250m` | 307.9 | 301.7 |
| `qwen3-0.6b` | 750.1 | 738.6 |

**This only records that the baseline moved.** It isn't a tolerance, a benefit
claim or a cold measurement, and it has no statistical test. Page-cache and
read-ahead effects of one descriptor against many were not measured.

## 6. Parity

The same two release binaries, on the same three containers:

| Container | Generated ids | Region set (object, segment, offset, length) | Use, post-I2b |
|---|---|---|---|
| Per-expert miniature fixture | Identical | Identical, 48 records | 1 mapped, 4 copied, of 5 |
| `oute-mamba2attn-250m` | Identical | Identical, none mapped | 3 copied, of 3 |
| `qwen3-0.6b` | Identical | Identical, none mapped | 4 copied, of 4 |

Tests also pin copied, ranged and mapped bytes against an independent read of
the file, and a real execution that computes the opened files' values after every
segment is replaced under the store, with a control showing the replacement
really differs. `.s7` was not run.

## 7. Not established

- Whether keeping one descriptor changes read-ahead or page-cache state against
  many opens. Unmeasured, and read-ahead may be per open file on some systems, so
  I2b's cold behaviour must not be assumed unchanged.
- That the descriptor's identity and the store's header come from one open cannot
  be proven by a test without a race hook. Reading the code shows it; two mutants
  in that area survive.
- Plugins, lowerings and `--representation` are allowed in a campaign and run in
  process; the record's facts do not cover them. Flagged for the maintainer.
- Capture failure through the real curve on macOS or Linux: no fixture can make
  `fstat` fail on demand. It is pinned on constructed facts there, and through
  the real path where the platform has no descriptor identity.
- That the `fgetattrlist` result continues to hold on other OS versions: the man
  page says otherwise, so a test pins it and a fallback exists.
