# CI-E2-D — tiered Metal evidence (DESIGN, not activated)

**Nothing in this document is wired to a trigger.** The manifest and the
classifier ship; `larql-compute-metal.yml` is untouched. E2-D is held
until E2-A's A1 sample closes, because D changes *which* pull requests
enter the Metal population and A measures *how long* the gate takes for
the ones that do. Landing them together would give a freshly-corrected
instrument two simultaneous attribution problems — the bill E1's C2
already paid.

## The question D answers

A asked: how long should the Metal gate take? Answer so far, 24m52s and
23m00s against a 56.61-minute baseline.

D asks a different one: **how often should we pay that at all?**

Over pull requests #415–#449, 18 triggered the Metal gate and **12 of
them (67%) touched no Metal file** — three were `Cargo.lock` alone.

## Why not better globs

`crates/larql-compute/**` holds two different kinds of thing:

- the trait boundary Metal **links** against — a change breaks the
  compile, and a compile catches it;
- the CPU kernels Metal's tests **compare numbers** against — 72
  references to `cpu::ops::q4_common` alone. A change here changes what
  the Metal kernels are required to reproduce, and only running them
  finds out.

No amount of directory ancestry separates those. So the distinction is
**declared**, in `crates/larql-compute-metal/parity-obligations.json`.

### The correction that motivated this, corrected again

`larql-compute/src/cpu/kquant_gemv.rs` (#420) was cited twice in the
design discussion as the case proving globs cannot work. On inspection it
is not: #420 **added** that file (+113/−0), it is referenced only from
`cpu/mod.rs`, and its consumer is `larql-vindex`'s opplan executor. The
Metal crate neither imports it nor reaches it.

It still classifies **tier A** here — because `cpu/**` is declared parity
under the asymmetry below, and that entry is marked `needs_review` with
this exact question attached. One review decision moves one pull request
in the sample. That is the honest size of the uncertainty, stated rather
than hidden.

## The asymmetry that decides every uncertain call

> A surface wrongly placed in **tier B is unsafe** — Metal compiles
> against changed numerics and never re-qualifies.
> A surface wrongly placed in **tier A is merely expensive.**

So uncertain goes to A, and every entry that is there by default rather
than by a positive finding carries `confidence: needs_review` and the
question that would settle it. Five of eleven entries currently do.

## The tiers

| tier | when | evidence |
|---|---|---|
| **A** | the Metal crate changed, or a declared parity obligation changed | full behavioural qualification (~24 min) |
| **B** | a crate Metal depends on changed, outside any parity obligation | `cargo check --all-targets` + clippy + the ordinary-profile witness (~2 min) |
| **C** | neither | no macOS work |

Tier B is not "trust it". It is the evidence proportional to the claim:
an upstream API change can make the Metal crate **uncompilable**, and a
compile establishes that it did not. It cannot establish that a number
changed — which is why anything that could change a number is tier A by
declaration.

## Architecture

A cheap Linux **triage** job computes the tier and the macOS jobs are
conditional on it. No generated glob list to drift, and no cargo
invocation inside a paths filter:

```
triage (ubuntu, ~30s)  ->  full   (macos-14, if tier == A)
                       ->  compat (macos-14, if tier == B)
```

Dependencies come from `larql-compute-metal/Cargo.toml`, read at triage
time — declared, not inferred. A crate that stops being a dependency
stops being a tier-B trigger with no glob to edit.

**`msrv-metal` folds into `compat`.** It is a compiler-compatibility
claim, and in E1's AFTER window it cost **63.3 macOS queue-minutes to buy
4.3 execution-minutes** — cheap to run, expensive to schedule, which is
precisely the pathology E1's c2 was written to remove and which c2 itself
created. Merging it into the tier-B job removes a whole macOS scheduling
unit.

## The completeness gate

The manifest's weakness is omission: a new parity dependency added
without a declaration would silently classify as B. So
`scripts/metal_tier.py check` extracts every `larql_compute::` /
`larql_models::` path the Metal crate names and fails if any is
undeclared.

The extraction is **approximate** — brace imports (`cpu::{ops, q4}`) and
type imports (`cpu::Foo`) truncate. That is why it is a *check on the
declaration* and never the declaration itself, and the error direction is
the safe one: truncation reports a shallower path, which is harder to
satisfy.

**It does not close every hole.** A fixture that changes its DATA rather
than its signature changes what Metal's tests assert without breaking any
compile and without adding a reference. That is a parity obligation
wearing an interface's clothes, it is recorded as the most likely single
hole in the manifest, and no static check finds it.

## What the sample predicts

Replayed against the manifest (`E2-D-classification.json`):

```
35 pull requests with file data
18 trigger the Metal gate today   (6 metal-touching, 12 upstream-only)

under E2-D:   A = 7    B = 11
```

**11 of 18 triggers (61%) drop from full behavioural qualification to
compile-plus-witness.** Seven stay tier A: #449, #446, #444, #442, #438,
#434 (all touch Metal or its workflow) and #420 (the `needs_review`
default above).

On the measured tier costs — 24 min and ~2 min — that is 18 × 24 = 432
macOS execution-minutes today against 7 × 24 + 11 × 2 = 190. It is also
11 fewer macOS scheduling units, which E1 *hypothesised* is the larger
effect. Be precise about the provenance: E1's **baseline** measurement
found the asymmetry — `quality`'s msrv-macOS leg was 33 jobs, 544
queue-minutes and 22 execution-minutes, cheap to run and expensive to
schedule. E1's **C2**, which would have tested whether removing such jobs
actually wins, came back `INVALID COMPARISON` and is unadjudicated. So
scheduling-unit relief is a live hypothesis inherited from a
pre-intervention measurement, not an established finding, and D4 is the
first prediction that can score it on an intensive metric.

Treat both as arithmetic on a replayed sample, not as a result. They are
what `E2-D-contract.json` will be scored against, not evidence for it.

## The review, resolved (manifest version 2)

All five `needs_review` entries are decided, and two of them were
**wrong**, in the unsafe direction.

| question | answer | evidence |
|---|---|---|
| `cpu/kquant_gemv.rs` | **exempt, tier B** | 0 references from the Metal crate; its only consumer outside `larql-compute` is `larql-vindex` |
| `cpu/nvfp4_gemv.rs` | parity | 42 references, including `shaders/nvfp4_matvec.rs` |
| `cpu/spin_pool.rs` | **silent parity** | 0 references, and `attention/decode/gqa_step.rs:125` calls `spin_pool::global().for_each_chunk()` — reduction ORDER inside a parity surface |
| `cpu::q4` | already covered | a re-export of `ops::q4_common`, `ops::q4_matvec`, `ops::q4_vecmat` |
| `backend/{capability,factory}.rs` | exempt, tier B | `pub enum Capability` with a false-returning default impl; `pub enum BackendKind` plus registry, `as_str`, Display |
| `pipeline`, `pipeline_layer` | parity | `test_pipeline_and_moe.rs` builds `FullPipelineLayer` and calls `prefill_kquant`, exercising the `dispatch_full_pipeline` + `moe_fn` chain |
| `options.rs` | **silent parity** | exposes `decode_options()` and `spin_pool_enabled()`; its own docs say "decode fast path — default ON", "force the f32/rayon path" |
| `test_fixtures.rs` (both crates) | **silent parity** | the input DATA the assertions are written against |

### A third class, not a reassignment

`options.rs` and the fixtures were filed as `interface_only` in version 1
on the argument that a compile catches a change. A compile catches a
**signature** change, which is not the hazard. Both belong to a class the
version-1 taxonomy had no room for:

> **`silent_parity_obligations`** — changes WHICH values appear without
> computing any. Path selection, reduction order, fixture data. Tier A on
> the same terms as parity, but **no static check can find them**: they
> add no reference and break no signature.

`metal_tier.py check` therefore *lists* them on every run and says it
cannot verify them. `spin_pool.rs` prints as `NOT referenced — either way
it stays tier A`, which is the class stating its own nature.

They exist as a separate class for one reason: a future reviewer looking
at `test_fixtures.rs` will reason "nothing imports it, it is test
scaffolding, it is structural" and demote it. The class name and its
`do_not_simplify` note exist to stop that.

### The asymmetry, applied structurally

Parity subtrees stay **whole** and exemptions are **named files**. A file
added to `cpu/` or `backend/` tomorrow defaults to tier A rather than
falling through to tier B. Two selftest controls hold that.

### A limit the review did not remove

**#420 is still tier A**, and not because of `kquant_gemv.rs` — that is
now exempt, along with its test module. It is tier A because it added
five lines to `crates/larql-compute/src/cpu/mod.rs`, and that module root
carries both inert `pub mod` declarations and the `pub mod q4 { pub use
super::ops::... }` re-export block that decides which implementation a
Metal reference resolves to. Classification is per file; a module root is
one file.

So the review moved **zero** pull requests in the sample — A=7, B=11,
unchanged. An earlier draft of this document predicted it would move one.
That prediction was wrong, and the reason is worth more than the
prediction: exempting a leaf does not help when the diff also touches the
root that re-exports it. Do not paper over this by exempting a module
root; going below file granularity is a different and much harder design.

## Before activation

1. **A1 closes** (n ≥ 8 successful Metal runs under E2-A).
2. ~~The five `needs_review` entries~~ — **done**, manifest version 2.
3. ~~The instrument gains a job-scoped metric~~ — **done**.
   `job_exec_max.<workflow>.<job>` counts successful jobs only, returns
   `NO DATA` rather than `PARTIAL` when the job is absent from either
   sample, and is validated against a baseline snapshot for any frozen
   forecast. A selftest control demonstrates the mixture it prevents: a
   24-minute and a 2-minute job in one workflow read as 24 and 2, where
   the workflow-level p50 reads 13.
4. `E2-D-contract.json` freezes with numeric baselines taken from the
   post-E2-A steady state, which does not exist yet.

## Later, not now: E2-E

GitHub evaluates a `pull_request` paths filter against the **whole PR
diff**, not the latest push — observed on #450, where a docs-only commit
retriggered the full Metal gate because the pull request as a whole edits
the workflow. So once a pull request contains a tier-A change, every
later push re-pays tier A, including pushes that only edit prose.

The fix would be to **reuse a successful qualification when the exact set
of inputs capable of invalidating it has not changed** — which needs a
relevant-input digest covering base and merge state, not "the latest
commit touched no Metal files". That is the same ancestry inference this
document rejects, one layer up.

**Not part of D.** D changes which pull requests enter the population; E
would change how often an entered one re-pays. Two mechanisms, two
experiments.
