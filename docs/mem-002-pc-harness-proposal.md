# MEM-002 PC1/PC2 intervention harness — proposal

**Class: DRAFT PROPOSAL — NOT AUTHORISED FOR IMPLEMENTATION.** Dated
2026-11-02 at main `bbf27efe`. Nothing is built or run here. Revised after an
independent review the same day.

It answers gap G2 of the reconciliation
([`mem-002-protocol-reconciliation.md`](mem-002-protocol-reconciliation.md),
#675, cited **[RC §n]**): PC1 and PC2 need another process to read a known
byte range and record what it did, and no instrument does that. The
maintainer approved a proposal for a minimal harness on 2026-11-02 (cited
**[M]**). The positive controls are defined in the protocol candidate
([`mem-002-cold-state-qualification.md`](mem-002-cold-state-qualification.md),
#670, **[Q §4, §4.1]**), which stays unfrozen, and in the I1 + I3 proposal
([`mem-002-i1-i3-proposal.md`](mem-002-i1-i3-proposal.md), #671 §5).

## 1. What it is

One committed script, `scripts/mem002_intervene.py` (new). It performs **one**
declared intervention per invocation and writes **one** JSON record of it.
It decides nothing, admits nothing and pairs nothing. Those are the
evaluator's [RC §3 G3].

| Kind | What it does | Used by |
|---|---|---|
| `read` | reads a declared byte range of one segment file through the page cache | PC1; PC2 treatment |
| `sham` | the same invocation with a zero-length range: opens and identifies the file, reads nothing | PC2 sham |
| `reset` | runs one declared reset command and records its outcome | PC2 setup (G1) |

## 2. The `read` and `sham` interventions

### 2.1 Inputs, all required, none defaulted

- `--container <path>`, and `--segment <declared path>`, the segment path as
  `index.json` declares it, relative to the container root.
- `--offset <bytes>`, `--length <bytes>` (0 for `sham`) and `--block-size
  <bytes>`.
- `--out <record path>`. It is created with `O_EXCL` **as the first
  action**, so an existing file is refused and records are immutable [M].
  Every later outcome, including a refusal or an unexpected exception, is
  written to it: an invocation that got past creating `--out` always leaves
  a record.
- **The kind must match the length:** `sham` requires `--length 0`, and
  `read` requires a length above zero. A mismatch is a `refused` record.
- `--pair-id`, `--arm` (`sham` | `treatment` | `pc1`) and `--label`. These
  are copied into the record and never interpreted.

### 2.2 What it does

1. Create `--out` (above). Hash the container's `index.json`, and refuse
   (a `refused` record, with the reason) if the declared segment is not in
   it.
2. Open the segment file read-only.
3. `fstat` the open descriptor, recording dev, inode, size, mtime_ns and
   ctime_ns, plus the volume UUID on macOS, read as in I2b.
4. Refuse (a `refused` record) if the range exceeds the file size.
5. Record the start time, read the range by `os.pread` in `block-size`
   chunks, and record the end time.
   - Plain buffered reads go through the page cache. That is the purpose:
     a treatment must warm its target.
   - No `F_NOCACHE` and no `mmap`. (`posix_fadvise` does not exist on
     macOS.)
6. `fstat` again, record it, close.

### 2.3 The record (`larql.pc-intervention/1`)

```
{"schema":"larql.pc-intervention/1","kind":"read"|"sham",
 "pair_id":…,"arm":…,"label":…,
 "container":{"path":…,"index_json_sha256":…},
 "segment":"<declared>","segment_path":"<absolute, diagnostic>",
 "file_identity":{"before":{dev,inode,size,mtime_ns,ctime_ns,volume_uuid},
                  "after":{…}},
 "requested":{"offset":…,"length":…,"block_size":…},
 "returned_bytes":…,
 "status":"ok"|"short_read"|"error"|"refused",
 "error":{"kind":…,"errno":…,"message":…}|null,
 "clock":{"basis":"CLOCK_MONOTONIC_RAW","boot_session":…,
          "start_system_mono_ns":…,"end_system_mono_ns":…},   // field names shared with I2a
 "wall":{"start":…,"end":…},
 "pid":…,"harness":{"path":…,"sha256":…,"commit":…|null}}
```

- **Requested bytes are not device reads** [M].
  - `requested.length` is what was asked for, and `returned_bytes` what the
    reads returned.
  - Neither is physical device traffic. Some or all of the range may already
    have been cached, and readahead may fetch more.
  - The record has no field for device bytes, and its documentation says
    so.
- **Status:**
  - `ok` only when `returned_bytes == requested.length`;
  - a short read is `short_read`;
  - any exception is `error`, with the errno when there is one;
  - a precondition failure is `refused`, with its reason.
  - **A failed or short read voids the control; it never fails it**
    (#671 §5). The harness records the status; the evaluator applies the
    rule.
- **Completion time** is the end of the last read call. That is the "read
  completion" of the separately recorded window, read completion → B
  binding [RC §5 N-c]. For a sham, it is the end of the zero-length
  invocation.
- **The clock** is the same basis and field names I2a records
  (`time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)`; the review confirmed
  Python and Rust read the same clock id here, with the boot session
  from `sysctl kern.bootsessionuuid`). So intervention and observation
  records can be ordered on one basis within one boot. Ordering is not
  causality.
- **Identity before and after** detects a change during the read. A
  mismatch is recorded and never corrected.
- **`harness.sha256`** is the hash of the script file that ran, so a record
  names its own code.

## 3. The `reset` intervention (G1)

The maintainer approved a separately witnessed reset as **unqualified
setup for PC2, not a cold method**. `purge` is a candidate, not an accepted
solution [M].

- **Input: `--reset <name>`, from a fixed allow-list in the script.** Today
  the list has one entry: `purge`, which runs `["sudo", "-n",
  "/usr/sbin/purge"]`. It is the candidate the maintainer named, not an
  accepted method.
  - **No arbitrary command is accepted.** Adding a candidate is a reviewed
    change to the script.
  - `sudo -n` never prompts. Without cached or configured privilege it fails
    at once, and that is recorded.
  - Each entry has a fixed timeout. A timeout is recorded as such.
  - The harness runs the entry once, and records its argv, exit status,
    stderr (truncated, with its length), and start and end times on the same
    clock.
  - It needs no container.
- **The harness does not decide whether the reset worked.**
  - That is established only by the **eligibility observation**: the
    `after_run` record of a fresh `--residency-curve` process with
    `--residency-regions` [Q §4.1].
  - The evaluator checks that observation is complete, and judges target
    eligibility from it.
- **Every attempt is recorded**, including failed ones and ones whose
  eligibility observation shows the targets still resident [M].
  - The PC2 runbook (stage 1) must keep all reset records.
  - If the reset does not reliably make targets non-resident, **PC2 fails
    feasibility**; the criterion is not changed [M].
- **Root.** `purge` needs root. The harness never prompts for it. The
  operator provides the privilege out of band, and the record shows the
  argv that ran.

## 4. What the harness does not do

- It does not select targets or ranges.
  - Ranges are declared before any pair runs. Their rule is fixed at stage 1
    and their values at stage 2 [RC §2.3].
  - They are never chosen from latency or treatment outcomes [M].
  - The harness only reads what it is told.
- It does not run `--residency-curve`, or sequence a pair. The PC2 runbook
  sequences: eligibility observation, sham A, sham intervention, sham B,
  treatment A, treatment intervention, treatment B [Q §4.1].
- It does not compute Δ, eligibility or pass criteria. That is the
  evaluator.
- It does not write to any model file. It opens read-only, and the review
  checks this.

## 5. Tests (`scripts/test_mem002_intervene.py` (new))

On temporary files:
- `read` returns exactly the requested bytes, `status ok`;
- a range past EOF gives a `refused` record, with no read;
- `sham` with a non-zero length, or `read` with zero, gives `refused`;
- an unexpected exception after `--out` is created still leaves an `error`
  record;
- a forced short read (a file truncated between `fstat` and read, or a
  mocked `pread`) gives `short_read`;
- an I/O error gives `error` with errno;
- `sham` reads 0 bytes, `status ok`, with identity recorded;
- an existing `--out` is refused;
- a segment not declared in `index.json` is refused;
- the record validates against the schema and carries no device-bytes
  field;
- clock and identity fields are present;
- a file rewritten between the two `fstat` calls shows differing identity;
- `reset` records a failing command's exit status, and never reports
  success for it;
- the harness opens no file for writing other than `--out` (asserted by a
  mock on `open`), and opens the segment read-only;
- `--reset` refuses a name not on the allow-list, uses `sudo -n`, and
  records a timeout.

A test cannot show that pages became resident; that is PC1 and PC2's job
on the real subject.

## 6. Review checklist

1. Python (stdlib only, alongside the existing `scripts/` preregistration
   tools), or a Rust example beside the CLI?
2. Is `os.pread` in `block-size` chunks the right read, or should the block
   size be a stage-2 number with a single read?
3. Should the allow-listed `reset` live in this harness, or in a separate
   script?
4. Is `harness.sha256` plus an optional commit enough provenance for the
   script?

## 7. Authorisation

This document authorises nothing. Implementing the harness needs separate
approval after this proposal is reviewed. Running it, the PC2 pilot, any
reset, calibration, the freeze and qualification remain unauthorised.
