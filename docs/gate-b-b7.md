# Gate B, B7: independent verification of VINDEX3 weight edits

**Status:** **FROZEN r2** (2026-11-21). The owner ruled on D-1 to D-5 (§7)
and strengthened V-1 (§4.4a). Changing this contract needs an owner-approved
revision.
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
- **r2** (2026-11-21) — frozen with the owner's rulings:
  - D-1 zero tolerance;
  - D-2 order invariance plus exact recorded order;
  - D-3 option (c), with FACT-WSP-1 required to refuse the path;
  - D-4 keep historical fixtures;
  - D-5 fresh-context review is eligible evidence, with head-bound approval
    or exception per PR and no blanket exception.

  V-1 is strengthened to bind requested ops to execution through an
  independent expected-operation manifest (§4.4a), and a third verdict half
  is added (§1). The order rule the manifest must predict is recorded from
  the engine (§2.1). T-5 is added for a consistent but wrong pair. The
  acceptance record must state the restricted algebra (§8).

## 1. What B7 proves

For every admitted sequence of VINDEX3 stored-weight edits `P1 … Pn` on a base
container `B`:

```text
Requested ops = Ledger ops   ∧   W_engine = VerifyApply(W_base, Requested ops)
```

The first equality is established against an expected-operation manifest
that the verifier builds from independently captured inputs. It never uses
the ledger the compiler wrote (§4.4a). A verdict therefore has three halves,
each reported on its own:
- **intent:** the manifest's effective ops equal the ledger's ops, in the
  recorded order;
- **ledger integrity and provenance:** v4 §7 steps 1, 2, 4 and 5;
- **numerical satisfaction:** v4 §7 step 3.

A container is verified only when all three are satisfied.

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

**Order the compile is given** (`knowledge/overlay.rs` `weight_ops`; v4 §3
"the order the compile was given them"):
1. the ops of each applied patch still in effect, patches in apply order and
   each patch's ops in file order;
2. then the directly staged ops still in effect, in staging order.

This is not chronological order. A direct op staged before an `APPLY PATCH`
is listed after that patch's ops. The ledger's `origins` follow this order,
and each target's `operations` follow it filtered to that tensor. The manifest
implements this rule itself (§4.4a). If the engine's order is found to differ
from the rule recorded here, that is evidence under §5.4, not a reason to
change either side.

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
| S-6 | a chained compile: edit C1 = compile(B), then edit and compile C1 again | **NOT_VERIFIED** (D-3 = c). B7-B0 probes and records what the engine does. If the engine admits it, the acceptance record says the capability exists outside the verified boundary, and FACT-WSP-1 must refuse it explicitly, not rely on users knowing it is unverified. Cumulative provenance across compiled generations is a later programme |
| V-1 | **intent**: the ledger's ops equal the manifest's effective ops (§4.4a), by digest, kind, tensor, columns and recorded order. A dropped, added, retargeted or reordered op violates the intent half | **GAP**, verifier code (§2.2) |

### 4.4a The expected-operation manifest (owner, r2)

V-1 is circular if the requested ops come from the compiler. So the verifier
builds them itself.

**Inputs.** The manifest is built only from inputs captured independently of
the compile, each recorded by its SHA-256:
- the session script: the ordered LQL statements of the session;
- every file it names: operand files, direction files and `.vlp` patches;
- the base container's `index.json` and the edited representation's payload
  digest, which supply each op's source binding.

The ledger, the compiled container and the engine's replies are never inputs.

**The admitted statement subset:**
- `USE`;
- `UPDATE WEIGHTS`, `REMOVE DIRECTION` and `ADD LOW RANK`;
- `APPLY PATCH`, `REMOVE PATCH` and `DISCARD WEIGHT EDITS`;
- `BEGIN PATCH` and `SAVE PATCH`;
- one final `COMPILE CURRENT INTO VINDEX`.

A script containing anything else cannot be evaluated: no manifest is built
and the intent half is `not_evaluated`.

**How each op is derived:**
- **A direct statement:** the verifier builds the op from the statement text
  and its operand file. That covers the tensor, columns, profile, operands,
  scale or delta, origin and source binding. It computes the digest with its
  own implementation of the §3a text, or the delta op's equivalent in
  `weight-patches-v2-v3.md`.
- **A `.vlp` patch:** each op's fields are read from the patch bytes and its
  digest is recomputed, never trusted.

**Three ordered lists:**

| List | Contents |
|---|---|
| `submitted` | every weight op the script presents, direct or in a patch, in script order |
| `accepted` | the submitted ops the verifier's own admission rules admit, judged at the moment each is submitted |
| `effective` | the accepted ops still in effect after the script's `REMOVE PATCH` and `DISCARD WEIGHT EDITS`, in the §2.1 order |

The admission rules for `accepted` are:
- the footprint conflict rules of §2.1 against the ops in effect at that
  moment;
- mixed forms on one tensor;
- the v4 §5 bounds;
- the source binding;
- intactness;
- for a nonzero structured op, at least one changed element, which the
  verifier decides with its own numerics.

`DISCARD WEIGHT EDITS` withdraws direct ops only. Ops carried by an applied
patch stay.

**Immutability.**
- The manifest is a canonical JSON document. It carries its inputs' digests
  and its own SHA-256, and is committed with each fixture.
- The intent check takes the manifest by path and reports its digest.
- A manifest whose recorded input digests do not match the inputs on disk is
  refused.

**Engine outcomes are compared, not trusted.** The fixture records the
engine's reply to each statement, accepted or refused. A statement the
verifier admits and the engine refused, or the reverse, is a finding under
§5.4. It is reported beside the verdict, and neither list is adjusted to hide
it.

**Intent half.** It is satisfied when:
- the ledger's `origins` equal the digests of `effective`, in order;
- each target's `operations` equal `effective` filtered to that tensor, field
  for field and in order;
- for element targets, the set of `effective` ops that edit each tensor is
  exactly the set the ledger attributes to it.

Any other ledger op, missing op or reordering is an intent violation, even
when the ledger and numerical halves are both satisfied.

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
| T-5 | **a consistent but wrong pair** (owner, r2): the engine itself compiles a script that differs from the manifest's: one op dropped, one op added, one op retargeted to a same-shape tensor, one operand value changed, two patches applied in the other order. Each container is judged against the original manifest. Ledger and numerical halves must be satisfied and the intent half violated. This is the failure the pre-B7 verifier could not detect | **GAP** |
| T-6 | a tampered manifest or input: one byte of an operand file, a patch or the script changed after the manifest was built | **GAP**: refused as an input fault, intent `not_evaluated` |

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
| B7-B0 | 1 | engine | `v4_fixtures.rs` pins every case on disk under `weight_patches_v4/` **and** `weight_patches_v2v3/`, without a hard-coded list. A v2/v3 case that no longer reproduces is recorded per §5.4 and handled per D-4. Also adds the S-6 probe, an engine test that records whether a chained compile is admitted |
| B7-B1 | 2 | verifier | the manifest builder and the intent half (§4.4a) for all three op kinds; T-2 to T-6, with T-5's wrong pairs compiled by the engine through `make_fixtures.py` |
| B7-B2 | 3 | verifier | fixtures I-4, I-6, R-1–R-7, M-3, S-1–S-5 with `.vlp`s and recorded statements; the matrix tests; T-1 |
| B7-C | 4 | verifier | C-2 on every fixture; C-3 from recorded live digests; C-4 and C-5 fixtures |
| B7-close | 5 | docs | the acceptance record, row by row, on the ledger |

Fixtures generated by `make_fixtures.py` in steps 2–4 are pinned
automatically by B7-B0's engine test, so no PR needs the `verifier-and-engine`
override. If a fixture's statements cannot be recorded as plain LQL (S-2
needs `.vlp` files committed beside them), the pin test's statement replay is
extended in B7-B0, before any such fixture exists.

## 7. Owner rulings (r2)

| Decision | Ruling |
|---|---|
| D-1 Tolerance | **Zero tolerance.** BF16 bits, Q8_0 bytes and discrete metadata must match exactly under the frozen numerical contracts |
| D-2 Reverse order | **Accepted.** S-1 tests numerical order invariance and checks the exact recorded op order independently (intent half, §4.4a) |
| D-3 Chained compiles | **Option (c).** NOT_VERIFIED, excluded from FACT-WSP-1, with the engine probe in B7-B0. If the engine admits it, FACT-WSP-1 must refuse it explicitly (S-6) |
| D-4 Historical fixtures | **Accepted.** A v2/v3 case that does not reproduce is kept as historical, and a fresh pinned case is added beside it. Nothing is regenerated silently |
| D-5 Verifier review | **Fresh-context review is eligible evidence.** Each PR that changes the verifier still needs review of its exact head and an explicit, head-bound approval or `Verifier-review exception`. **No blanket exception is granted** |
| V-1 | **Strengthened:** requested ops come from an independent expected-operation manifest (§4.4a), and T-5 tests a consistent but wrong ledger/container pair |

The numerics are unchanged. The explicit unverified boundaries (§4.7, S-6)
are retained.

## 8. Acceptance

B7 is PASS for FACT-WSP-1 when:
- every row of §4.1 to §4.6 is COVERED or REFUSED on pinned engine output,
  with all three verdict halves (§1);
- §5 holds;
- the §4.7 rows and S-6 carry their labels in the code and docs.

FACT-WSP-1 weight-mutation implementation stays blocked until then.

**The acceptance record must say what algebra was verified.** B7 covers a
deliberately restricted algebra of weight transformations:
- disjoint-footprint deltas, `remove_direction` and `add_low_rank_columns`;
- each computed on the base;
- byte-wise order-invariant;
- overlaps refused rather than given an order-dependent meaning.

It is not a verification of arbitrary sequential weight updates, of chained
compiled generations, or of fine-tuning. A later operation with
order-dependent semantics needs its own contract.
