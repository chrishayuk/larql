# Gate B, B7: independent verification of VINDEX3 weight edits

**Status:** **DRAFT r1** (2026-11-21), for the owner to rule on and freeze.
No verifier or fixture code is written until this matrix is frozen.
**Base:** `origin/main` `3406037d` (2026-11-21), after #898 (`dffe0038`).
**Parent documents:** [`gate-b.md`](gate-b.md) r1, frozen: B7 row and §6.2.
[`gate-b-audit-2026-11-18.md`](gate-b-audit-2026-11-18.md) records the B7
finding. The verifier contracts are
[`weight-patches-v4.md`](verification/weight-patches-v4.md),
[`weight-patches-v2-v3.md`](verification/weight-patches-v2-v3.md),
[`q8-0-pack-derivation.md`](verification/q8-0-pack-derivation.md) and
[`independence.md`](verification/independence.md).

### Revision log

- **r1** — draft. Inventory (§2), invariant (§3), coverage matrix (§4),
  controls (§5), sequence and boundary (§6), decisions for the owner (§7).

## 1. What B7 proves

For every admitted sequence of VINDEX3 stored-weight edits `P1 … Pn` on a base
container `B`:

```text
W_engine  =  VerifyApply(W_base, P1 … Pn)
```

- `W_engine` is what the engine publishes: the compiled container's canonical
  BF16 tensors, and the Q8_0 pack it re-derives from them.
- `VerifyApply` is the Python verifier in `verifier/`. It implements the
  frozen numerics itself and imports nothing from LARQL. This is enforced by
  `verifier/tests/test_boundaries.py` and the required `boundary` check.

**Tolerance.** The frozen numerics (`conformance/vlp-numerics/v1` for
`remove_direction` and `v2` for `add_low_rank_columns`, in all three profiles)
fix every result bit. The tolerance is therefore zero:
- every BF16 element must be bit-identical, signed zeros included;
- every Q8_0 pack byte must be identical;
- discrete metadata must match exactly: op set, op order as recorded, tensor
  names, columns, digests and counts.

No other tolerance is admitted, and B7 does not introduce one.

**Not proved by B7:**
- That a kernel computes correctly with the bytes. That is a separate claim,
  per `q8-0-pack-derivation.md` §1.
- That an edit changes model behaviour as intended. That is FACT-WSP-1.
- Where operands came from. The digest covers `origin`'s text, not its truth.

## 2. Inventory (main `3406037d`)

### 2.1 Admitted VINDEX3 stored-weight operations

| `.vlp` op | LQL statement | Engine | Numerics |
|---|---|---|---|
| `weight_delta` (`WeightDelta`) | `UPDATE WEIGHTS SET DELTA = … WHERE TENSOR … ROW … COL …` | `weight_patch/op.rs` | one RNE of exact `old + delta` to BF16 |
| `structured_weight` (`StructuredWeight`) | `REMOVE DIRECTION … FROM … COLUMNS … SCALE … NUMERICS …` | `weight_patch/structured_op.rs` | vlp-numerics v1, profiles `f64-seq-rne1/v1`, `f64-seq-rne2/v1`, `f32-seq-rne1/v1` |
| `low_rank_columns_weight` (`LowRankColumnsWeight`) | `ADD LOW RANK … TO … COLUMNS … NUMERICS …` | `weight_patch/low_rank_op.rs` | vlp-numerics v2, the same three profiles |

Paths are under `crates/larql-vindex/src/format/vindex3/`.

**How an op reaches a container.** An op is either staged directly by its
statement or replayed from a `.vlp` by `APPLY PATCH`. It is withdrawn by
`REMOVE PATCH` or `DISCARD WEIGHT EDITS`. `COMPILE CURRENT INTO VINDEX` then
writes a container and its ledger: `v2` for deltas only, `v3` for removals
without low-rank, and `v4` when any tensor has an `add_low_rank_columns` op.

**The live view** (`executor/vindex3/live.rs`) materialises the same bytes
through COMPILE's own path. INFER and TRACE execute it. Each reply prints the
sha256 of each live tensor.

**Composition rule** (`weight_patch/any_op.rs`, v4 §4):
- Ops conflict by their **declared footprint**:
  - delta against delta conflicts on the same element;
  - structured against structured conflicts on the same column;
  - delta and structured ops on one tensor are refused as mixed forms.
- Admitted ops on one tensor have disjoint footprints, and each one reads only
  its own columns of the **base**.
- **No admitted sequence is order-dependent in its bytes.** An order that would
  matter is an overlap, and an overlap is refused.

A zero-valued low-rank op (a sham) is admitted, alone or beside nonzero ops.
A nonzero op that rounds away entirely is refused (`op_target.rs` `resolve_operation_target`).

### 2.2 Existing independent coverage

| Area | Verifier contract | Fixtures | Pinned to today's engine? |
|---|---|---|---|
| Deltas, removals, delta + removal on different tensors | `weight_patches_v2v3.py` | `conformance/weight_patches_v2v3/` (4 cases) | **no**. Generated at `47d56f6b`, no engine test reproduces them. |
| Low-rank, low-rank + removal on one tensor, + delta on another, sham, f32 profile, split | `weight_patches_v4.py` (99 tests, including the v4 §7 tamper set) | `conformance/weight_patches_v4/` (5 cases) | **yes**, `tests/weight_patch_lql/v4_fixtures.rs` (byte for byte) |
| Re-derived Q8_0 pack of an edited container | `q8_0_pack.py` | 4 of the 5 v4 cases (`split` is absent); no v2/v3 case | via the v4 pin |
| The live view equals the compiled bytes | none independent | — | engine-side only: `weight_patch_lql/live.rs` `a_patched_session_matches_a_fresh_binding_of_the_container_it_compiles` |

**What the verifier cannot judge today.** It checks the container against
*the ledger's* ops, not against *the ops the caller asked for*. The one
exception is `remove_direction` under `--patch`. So a compile that dropped,
added or retargeted a whole op would leave a ledger and container that agree
with each other, and the verifier would pass them. This is §6.2 item 1 of
`gate-b.md`, and it is the largest gap. Row V-1 below addresses it.

## 3. The invariant, case by case

| Case | The verifier must establish |
|---|---|
| One compile | the container's canonical tensors equal the base plus the ops; the ops equal the ops the patches supplied; the pack equals the derivation of the canonical tensors; everything else equals the base |
| A sequence with withdrawals | the same, where the op set is the ops **still in effect**: applied ops minus removed and discarded ones |
| A full rollback | the compile after withdrawing every op equals the base, tensor by tensor, with no weight ledger |
| A replay | two sessions given the same patches in the same order write byte-identical containers, `candidate.json` excepted |
| A reorder | the canonical tensors and pack are identical; the ledger differs only in the order of `operations` and `origins` |

## 4. Coverage matrix (to freeze)

Grades:
- `COVERED`: an independent verifier test on pinned engine output exists on
  main.
- `GAP`: B7 must add it.
- `REFUSED`: the engine must refuse. An engine refusal test is required, and
  where it applies, a verifier test that refuses a hand-built container of
  that shape.
- `NOT_VERIFIED`: outside the admitted set. It is labelled, and FACT-WSP-1 may
  not use it.

### 4.1 Individual operations (B7-B)

| # | Row | Grade |
|---|---|---|
| I-1 | delta, several elements, two tensors | COVERED, but **unpinned** (v2v3 `deltas`): GAP until pinned |
| I-2 | `remove_direction` `f64-seq-rne1/v1` | COVERED, unpinned (v2v3 `removal`): GAP until pinned |
| I-3 | `remove_direction` `f64-seq-rne2/v1` | COVERED, unpinned (v2v3 `removal_two`): GAP until pinned |
| I-4 | `remove_direction` `f32-seq-rne1/v1` in a container | **GAP** (arithmetic vectors only) |
| I-5 | `add_low_rank_columns` `f64-seq-rne1/v1`, rank 2, scattered columns, `-0` factors | COVERED (`low_rank_k2`) |
| I-6 | `add_low_rank_columns` `f64-seq-rne2/v1` in a container | **GAP** |
| I-7 | `add_low_rank_columns` `f32-seq-rne1/v1` | COVERED (`f32_profile`) |
| I-8 | sham, alone | COVERED (`sham`) |

### 4.2 Repeated edits on one tensor (B7-B, first priority)

| # | Row | Grade |
|---|---|---|
| R-1 | three `ADD LOW RANK` on one tensor, interleaved disjoint columns, ranks 1, 2 and 3 | **GAP** |
| R-2 | R-1 with each op under a different profile | **GAP** (the engine admits mixed profiles on one tensor) |
| R-3 | R-1 where the ops arrive from three separate `.vlp`s through `APPLY PATCH` | **GAP** |
| R-4 | a sham beside two nonzero low-rank ops on one tensor | **GAP** (admitted; the audit had it as undetermined) |
| R-5 | two `remove_direction` on one tensor, disjoint columns, two profiles | COVERED, unpinned (`removal_two`): GAP until pinned |
| R-6 | many deltas on one tensor, disjoint elements, including one that rounds to its own bits | COVERED, unpinned (`deltas`): GAP until pinned |
| R-7 | the most ops on one tensor that `MAX_STRUCTURED_WORK` admits, and one more, which must be refused | **GAP** (bounds have verifier tests, no engine-produced container) |

### 4.3 Mixed operations (B7-B)

| # | Row | Grade |
|---|---|---|
| M-1 | low-rank + removal on one tensor, delta on another | COVERED (`mixed`) |
| M-2 | low-rank on one tensor, removal on another (needs `--patch`) | COVERED (`split`) |
| M-3 | delta tensor + low-rank tensor + removal tensor + an untouched tensor in the same segment, in one compile | **GAP** |
| M-4 | delta and structured op on one tensor | REFUSED: engine tests exist (`low_rank.rs`, `structured.rs`); verifier `a_target_with_both_operations_and_edits` |
| M-5 | overlapping columns (low-rank/low-rank, low-rank/removal) and a shared delta element | REFUSED: engine tests exist; verifier `test_overlapping_columns_between_two_ops_of_one_target`. **GAP:** no verifier test for a duplicated delta element in an engine-shaped ledger |
| M-6 | weight ops + knowledge edits in one compile | REFUSED (tested). The knowledge bake stays outside (§4.6) |

### 4.4 Sequences and controls (B7-B)

| # | Row | Grade |
|---|---|---|
| S-1 | the same ops in reverse order: tensors and pack identical, ledger order reversed, both verify | **GAP** |
| S-2 | apply P1, P2 and P3, `REMOVE PATCH` P2, compile: verifies against the ops of P1 and P3 **only**, and is byte-identical to a fresh session applying P1 and P3 | **GAP** |
| S-3 | the same patches in a fresh session: byte-identical container | partly COVERED, single-op only (`structured.rs`, `low_rank.rs` fresh-session tests). GAP for R-3 and S-2 |
| S-4 | a refused op between two accepted ones: the compile verifies with the two accepted ops only | **GAP** |
| S-5 | full rollback (apply, remove, compile; and stage, `DISCARD WEIGHT EDITS`, compile): tensors byte-identical to the base, no weight ledger | **GAP**. The verifier has no "identical to base" judgement for a container with no ledger |
| S-6 | a chained compile: edit C1 = compile(B), then edit and compile C1 again | **D-3**, undetermined (§7) |
| V-1 | **op-set equality**: given the patch or patches, the ledger's ops (by digest, kind, tensor, columns and order) equal the ops in effect. A dropped, added, retargeted or reordered op is a ledger violation | **GAP**, verifier code (§2.2) |

### 4.5 Tampering (B7-B)

These already exist as v4 verifier tests on the committed fixtures:
- a flipped bit inside the footprint;
- a changed element outside it (including `-0`/`+0`);
- another tensor changed;
- a stale binding;
- `a`/`b` swapped or reordered;
- the wrong `b` layout;
- operand, result, base and count digests tampered;
- an op edited without resealing;
- an unknown schema, op or profile.

| # | Row | Grade |
|---|---|---|
| T-1 | the tamper set above re-run on the R-1/R-2 fixtures, where a tamper of op 2 of 3 must be caught | **GAP** |
| T-2 | target identifier: an op's `tensor` re-pointed at another tensor of the same shape and resealed, so the ledger and container are consistent | **GAP**, caught only by V-1 |
| T-3 | ordering: the ledger's `operations` permuted against the patch | **GAP**, V-1 |
| T-4 | numerical value: one operand element changed, digest and operand digest resealed, container recompiled to match | **GAP**, caught only by V-1 |

### 4.6 Compiled output (B7-C)

| # | Row | Grade |
|---|---|---|
| C-1 | every edited tensor's canonical bytes recomputed whole; every other tensor and file byte-identical (v4 §7 steps 3 and 4) | COVERED for the v4 cases; applies to every new fixture |
| C-2 | the re-derived Q8_0 pack of every B7 fixture is the declared derivation (`q8_0_pack.py`) | **GAP** for `split` and every v2/v3 and new case |
| C-3 | the live view: the sha256 each reply prints per tensor equals the sha256 of the tensor the verifier recomputed | **GAP**. The fixture records the printed digests; a verifier test compares them |
| C-4 | `COMPACT INTO VINDEX` of an edited container still verifies against the original base | **GAP** (engine test `a_weight_patched_container_compacts_with_its_ledger` checks only that the ledger survives) |
| C-5 | a knowledge compile of an edited container (`a_knowledge_compile_of_a_weight_patched_container_keeps_its_ledger`) still verifies against the original base | **GAP** |

### 4.7 Not admitted

| Path | Grade |
|---|---|
| VINDEX2 weight-affecting paths: compose INSERT, REBALANCE, MERGE, VINDEX2 COMPILE (COMPACT MAJOR refuses since #898) | **NOT_VERIFIED**. They do not inherit any VINDEX3 result |
| VINDEX3 knowledge bake (`COMPILE CURRENT INTO VINDEX` of KNN/compose state) | **NOT_VERIFIED**. No ledger. FACT-WSP-1 may not use it (`gate-b.md` §6.2 item 6) |
| Remote weight mutation | blocked until Gate S |
| `InsertKnn` / `DeleteKnn` / FACT-ID-1 ops | not weight ops; FACT-ID-1 governs them |

## 5. Independence controls

1. **No shared routines.** `verifier/tests/test_boundaries.py` already refuses
   any import beyond the standard library, NumPy and `gguf`. B7 adds no
   exception to it.
2. **Fixtures are engine output, not oracles.** Every fixture is produced by
   `make_fixtures.py` from recorded LQL statements. The expected values come
   only from the verifier.
3. **Pinning.** Every fixture case, v2/v3 included, is pinned by an engine
   test that recompiles its recorded statements byte for byte. A case is
   pinned because it is on disk, not because it is named in a list (§6, B7-B0).
4. **Disagreement is evidence.** If the verifier and engine disagree on any
   B7 case:
   - the case is committed as it stands, with both outputs;
   - neither side's semantics changes until the owner rules whether the
     engine, the verifier or the specification is wrong;
   - B7 does not change engine numerics to make a row pass.
5. **Fail-before.** Each new verifier test is shown failing on a tampered
   container, and V-1 on a ledger with a dropped op, before it counts.

## 6. Sequence and the change boundary

The `boundary` check fails a PR that changes both `verifier/` and `crates/`.
Each verifier-path PR needs your review evidence or a `Verifier-review
exception (<head sha>):` comment on its exact head. So:

| Step | PR | Side | Contents |
|---|---|---|---|
| B7-B0 | 1 | engine | `v4_fixtures.rs` pins every case on disk under `weight_patches_v4/` **and** `weight_patches_v2v3/`, without a hard-coded list. If v2/v3 no longer reproduce at `3406037d`, that is recorded per §5.4 and not regenerated silently |
| B7-B1 | 2 | verifier | V-1 op-set equality for all three op kinds (extends `--patch`), and T-2/T-3/T-4 |
| B7-B2 | 3 | verifier | fixtures I-4, I-6, R-1–R-7, M-3, S-1–S-5 with `.vlp`s and recorded statements; the matrix tests; T-1 |
| B7-C | 4 | verifier | C-2 on every fixture; C-3 from recorded live digests; C-4 and C-5 fixtures |
| B7-close | 5 | docs | the acceptance record, row by row, on the ledger |

Fixtures generated by `make_fixtures.py` in steps 2–4 are pinned
automatically by B7-B0's engine test, so no PR needs the `verifier-and-engine`
override. If a fixture's statements cannot be recorded as plain LQL (S-2
needs `.vlp` files committed beside them), the pin test's statement replay is
extended in B7-B0, before any such fixture exists.

## 7. Decisions for the owner

- **D-1 Tolerance.** §1 reads "frozen numerical tolerances" as zero
  tolerance: the frozen profiles define every bit. Confirm, or name a
  tolerance and where it applies.
- **D-2 Reverse-order control.** By §2.1 no admitted sequence is
  order-dependent, since overlaps are refused. S-1 therefore tests order
  *invariance* plus the recorded ledger order. Confirm that is the intended
  control, and that a ledger whose `operations` order differs from the
  patches' order is a ledger violation under V-1.
- **D-3 Chained compiles (S-6).** It is not yet known whether the engine
  admits editing an already weight-patched container whose ledger is bound to
  its own base. Options:
  - (a) admit it and verify each link against its own base;
  - (b) refuse it, with a test;
  - (c) leave it NOT_VERIFIED and out of FACT-WSP-1.

  I recommend (c) for B7, with a probe in B7-B0 that records what the engine
  does.
- **D-4 v2/v3 fixtures that no longer reproduce.** If B7-B0 shows a v2/v3
  case is not what today's engine writes, regenerate it under review, or keep
  the old one as historical and add a fresh pinned one. I recommend the
  latter, so the change is visible.
- **D-5 Verifier review.** Steps 2–4 change verifier paths, and each needs
  your exception or approval on its head. Confirm that a fresh-context model
  review cited in your exception is acceptable, as `independence.md` permits.

**Acceptance.** B7 is PASS for FACT-WSP-1 when every row of §4.1–§4.6 is
COVERED or REFUSED on pinned engine output, §5 holds, and the §4.7 rows carry
their labels in the code and docs. FACT-WSP-1 weight-mutation implementation
stays blocked until then.
