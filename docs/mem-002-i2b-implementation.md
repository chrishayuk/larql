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
| `--plugin`, `--lowering`, `--representation` | Refused before any plugin loads | Allowed and recorded |
| A `--backend` other than `reference` or `production*` | Refused: its execution identity has not been shown end to end | Unchanged |
| A wanted encoding with no pack, or otherwise not executed as asked | Refused as invalid, before binding or at admission | Records the problem and continues |
| Execution identity incomplete | Refused as invalid, before the passes | Records the problem and continues |

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
`pre` graph verification, lacks a field this reader requires, or whose computed
and declared hashes do not both equal the container index's hash, verifies
nothing. `container.path`, `container.volume_uuid`, `verifier.command` and
`verified_wall` must be present with the right type and are not otherwise checked:
a manifest with a meaningless path or command can still reach `identity_consistent`.
The contents of `clock` and `verifier.binary` are not checked, and `identity.dev`,
though in the proposal's format, is not required.

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

## 6a. Execution identity

Verified file identity is not execution identity: the same files can run under a
different engine, lowering or encoding. The independent check before merge showed a
plugin reversing logits with an indistinguishable record, and `production-nvfp4` running
canonical bytes under one.

Each record separates the request from the effect:

- **`requested`, from the command line:** backend, wanted encoding, plugins with their
  digests by path, lowering, representation. Never evidence of what ran.
- **Before binding, from the store:** each object's selected encoding and whether it came
  from a pack.
- **At admission, from the prepared image and the process:** `engine`, `provider`,
  `pinned_lowering`, `objects`, `precision_program`, `realizations` (representation, codec,
  form, operand count), `process` (`ProcessArithmetic`), `execution_digest` and `complete` with
  its `problems`.
- **After the run, from the executor's ledger:** `projection_plans_ran` (calls and bytes,
  no timings), the `wanted_plans` the pins named, and `complete`, which includes what
  admission found.

A campaign attempt stops as invalid when execution identity is incomplete: no pins, pins
naming another provider, an overlaid source or unreadable process arithmetic. It also stops
when a wanted encoding:

- has no pack of it, or its pack isn't on disk;
- has no operand running it in place, so a pack that only serves a gathered embedding table
  doesn't count;
- has a pin that decodes or requantises it, which is how `LARQL_KQUANT_EXEC=widen` shows;
- comes from a container with no precision program for that encoding, which also refuses
  a whole pack in an older container with no program, or a layered one whose program names
  another encoding;
- leaves an object without a pack that the container's precision program compiles;
- has a pinned kernel the ledger never counted, found after the run: the attempt's record is
  kept whole and marked invalid.

A diagnostic run records the same problems and carries on.

`identity_consistent` therefore also requires a complete execution identity. It still
isn't a claim of numerical equivalence or correctness. A different digest means a different
recorded identity, not necessarily different numbers: `LARQL_CPU_WORKERS=2` moved the digest
and gave byte-identical logits. Equal digests don't show either run is right.

**Limits:**

- **Coverage is judged by object, against the program.** A tensor carried verbatim inside a
  pack, or held at source by a program exception such as `--protect-layers`, is recorded and
  not refused. So a pack can execute little of itself and be complete: a Q4_K pack with most
  layers protected ran 7 of 198 pins in place, and with every decoder layer protected and the
  head compiled, 1 of 198, generating what canonical bytes generate. `realizations` shows how
  much was pinned. A protocol that needs more must say so.
- **A precision program names roles, not `--object`.** A role whose tensors are all in an
  object with no pack is refused even if a compile was narrowed on purpose. That fails closed.
  An object whose tensors could not hold the encoding at all reads the same way.
- **A stored MXFP4 label is exempt**, because real containers name `expert-weight` in their
  program and leave that bank alone. The test reads the label, not the operation.
- **A form with no named kernel can't be checked against the ledger.** Mapped and sliced banks
  have none: all 48 expert pins of a real `gpt-oss` run appear in `realizations` and produce no
  ledger row.
- **The ledger check's firing path has not run for real.** No prompt tried missed a pinned kernel.
  `ran_problems` is pinned on constructed facts, and the line from it to the after-run class on
  reading: a campaign refuses the plugins that could force it.
- **A selected pack for an object outside the component** is still judged missing if it isn't on
  disk. That fails closed.
- **A store that can't bind ends with status `error`** and the class it held by then, not `invalid`:
  two full packs of different encodings refuse at load this way. Read the status with the class.
- **`process.kquant_execution` reads the process setting.** It can say `Widen` beside `Direct` pins
  under a provider that ignores it. Read the pins.
- **Process settings are recorded, not refused**, unless they change what a wanted encoding's
  pins do. `execution_digest` moves with them.
- **`execution_digest` doesn't bind the model.** File identity is the other half of the record.
- **`index.json` is parsed when the store opens and hashed by path later.** A swap in that window
  would have the digest authenticate a file the store didn't parse.
- **A diagnostic run's plugin is recorded by path and digest, after it loaded.** A replaced file
  gives a digest of something else. Its code runs in process and could change anything,
  including the record, so its effect isn't covered.
- **The console's `served entirely from stored bytes` isn't evidence.** It counts manufactured
  MXFP4 and NVFP4 only, and real packs pin the head as `Requantise(FusedQ8)`. The record reads the
  pins instead.

## 7. Not established

- Whether keeping one descriptor changes read-ahead or page-cache state against
  many opens. Unmeasured, and read-ahead may be per open file on some systems, so
  I2b's cold behaviour must not be assumed unchanged.
- That the descriptor's identity and the store's header come from one open cannot
  be proven by a test without a race hook. Reading the code shows it; two mutants
  in that area survive.
- Capture failure through the real curve on macOS or Linux: no fixture can make
  `fstat` fail on demand. It is pinned on constructed facts there, and through
  the real path where the platform has no descriptor identity.
- That the `fgetattrlist` result continues to hold on other OS versions: the man
  page says otherwise, so a test pins it and a fallback exists.
