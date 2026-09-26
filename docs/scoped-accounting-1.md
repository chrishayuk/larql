# SCOPED-ACCOUNTING-1: projection accounting owned by the execution

Frozen 2026-09-23, before implementation. The claim, the design, the witnesses and the PR order are FROZEN.
Nothing here claims a measurement.

Programme: VINDEX3 execution authority. It completes MEASURE-PLAN-1's PR 4b and fixes BACKEND-AUTHORITY-1's
F1 (the stationary SITE read on a worker thread). Both have the same cause: one process-global instrument.

---

## Why

A standing ruling (2026-09-06, on the PROGRESSIVE-1 merge):

> Before any further qualified performance claims, replace that global ledger with execution-scoped accounting
> and rerun the affected measurements. Thread-local storage alone is only sufficient if a witness proves all
> stage scopes remain on the coordinating thread.

The projection ledger is still global (`exec/cpu/ledger.rs:457`, `static LEDGER`). Its own doc gives the
reason: `ProductionBackend` is a zero-sized value that call sites construct freely, so per-instance counters
would each see a fraction of a decode. Three consequences are now concrete.

1. **Two arms in one process cannot be told apart.** MEASURE-PLAN-1 runs a reference and a candidate
   interleaved in one process. Its client therefore refuses K-quant candidates. It cannot attest whether a
   stored K-quant ran in place (`FusedKQuant`) or widened (`BlasF32`), because the only record is the global
   ledger, which mixes both arms.
2. **The stationary decision reads the site on the wrong thread** (BACKEND-AUTHORITY-1 F1). The interpreter
   sets the site with `in_site()` on the coordinating thread. `stationary.rs:132` reads `current_site()`,
   a thread-local, on whatever thread the kernel runs on. On a rayon worker it reads `Unclassified`, so the
   ledger records the grouped path while the looped path ran. The CPU-7C arm E/D readings are invalid until
   rerun.
3. **Concurrent tests contaminate each other** (the "process-global test instruments race" class). Any test
   that resets or reads the ledger sees every other test's projections.

## The claim of this rung

> Projection accounting belongs to one execution: it is created by the execution, reachable by every
> projection that execution makes, and read only through it. The site a projection belongs to travels with
> the call as a value. No projection path reads a process-global counter or a thread-local site.

## Design

- **`ExecutionAccount`:** the projection tallies of one execution (per plan: calls, bytes, groups; per site:
  shares; projection nanoseconds), the same quantities `ProjectionLedger` holds, owned by the execution. A
  `DecodeSession` and a one-shot `execute_plan` each create one. An arm of the MEASURE-PLAN-1 procedure owns
  its session's account.
- **`ProjectCall` carries the site and the account:** `site: Site` and `account: &ExecutionAccount`. The
  interpreter already knows the site at every point it now calls `in_site()` (`exec/mod.rs:1362`, `:1490`,
  `:1505`; `decode.rs:850`, `:1027`; `conv_qkv.rs:125`; `gated_delta.rs:577`; `mamba2.rs:170`), so it passes
  the value instead of setting a thread-local.
- **The executor records into the call's account,** from whatever thread runs the kernel. The account is
  `Sync` (relaxed atomics, as today), so a rayon worker records into the right execution. The stationary
  decision reads `call.site`, not `current_site()`.
- **`SITE`, `in_site` and `current_site` are removed.** `static LEDGER` and `ledger()` are removed too, not
  kept beside the account: two authorities for one fact is the failure being fixed. `THREAD_CALLS` goes
  with them, if the audit in PR 1 finds no reader that needs it.
- **Readers move to the account they mean:** `vindex3 exec`'s `projection plans:` line, `generate`, `decode`,
  `larql bench`, `examples/cpu7c_arms.rs`, the `larql-inference` record test, and the exec tests that read
  the ledger. Each reads the account of the execution it ran.
- **The provider handle forwards nothing new.** The account travels with the call, not with the backend, so
  `SharedProvider`, the registry and every `PlanBackend` implementation need no new method, and no forwarding
  can be silently defaulted.

## Witnesses (each committed RED before the change that turns it green)

- **W1: two executions, two accounts.** Two executions in one process, interleaved on one thread, each read
  exactly their own projections: calls and bytes equal a solo run's. RED today: the ledger reports the sum.
- **W2: concurrency.** Two executions on two threads at once, each through the rayon-parallel projector,
  report their solo totals. RED today.
- **W3: the site travels with the call.** A projection issued at `Site::Ffn` and executed on a rayon worker
  is recorded, and decided stationary, as `Ffn`. RED today: the worker reads `Unclassified`. This is
  BACKEND-AUTHORITY-1 F1's witness.
- **W4: no global remains.** A source scan (the ingestion-closure pattern) finds no `static` projection
  ledger, and no `thread_local!` site, in `exec/`. Mutation: reintroducing either fails it.
- **W5: arithmetic is unchanged.** Logits on the default production path are bit-identical before and after,
  on the dense fixture, across every `production*` arm. Accounting must not move a number.
- **W6: K-quant attestation in the procedure.** Each MEASURE-PLAN-1 arm reports its own projection plans.
  For a candidate that binds a stored K-quant, the procedure requires the plan the arm declares. `direct`
  requires `FusedKQuant` to carry the pack's projections; `widen` requires `BlasF32`. Otherwise
  `UnexpectedPhysicalRead`. `run_bank.py` stops refusing K-quant candidates. This carries
  `run_bank_legacy.py:assert_stored_kquant_ran_as_declared` into the procedure, with the same conditions.

## Reruns owed (after the code, before any claim that used them)

- **CPU-7C arms E and D,** with the site read from the call. Their earlier readings stay recorded as invalid,
  not overwritten.
- Any published projection split (per plan or per site) whose run was not single-execution and
  single-threaded is listed as needing a rerun. V4's split was taken single-threaded and stands, per the
  2026-09-06 ruling.

## Out of scope

- **The stage ledger** (`exec/stages.rs`, `static StageLedger`) has the same defect and falls under the same
  ruling. It is the next rung (SCOPED-ACCOUNTING-2), so this one stays reviewable. Its readers are listed
  there.
- The Metal submission clock (#518) is already per backend instance.
- BACKEND-AUTHORITY-1's other findings (F2–F6).

## PR order

1. **The account type and the call's site:** `ExecutionAccount`, `ProjectCall { site, account }`, the
   interpreter passing both, and the executor and stationary decision reading them. Witnesses W1–W3 go red
   first, then green. W5 guards arithmetic.
2. **Readers migrated, and every global removed:** CLI, bench, examples, inference and tests. W4.
3. **The procedure's arms report plans:** W6, and `run_bank.py`'s K-quant refusal lifted.
4. **CPU-7C E/D rerun:** recorded beside the invalid readings.
