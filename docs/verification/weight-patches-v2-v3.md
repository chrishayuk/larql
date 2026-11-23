# Weight-patch ledgers `weight-patches/v2` and `v3`: `WEIGHT_PATCHES_V2`, `WEIGHT_PATCHES_V3`

**Status: draft for review.** This document is the contract an independent verifier of a compiled element edit is
written against. It changes no format and no numerics: the ledgers are what the engine already writes, the delta rule
is `bf16-rne-of-exact-sum` as [weight-patches-v4.md](weight-patches-v4.md) §3b states it, and a `remove_direction` op
is the frozen `conformance/vlp-numerics/v1`.

Section references of the form "v4 §N" are to [weight-patches-v4.md](weight-patches-v4.md). The verifier's findings
cite that document's sections too, because the judge is the same one.

## 1. Scope

Two contracts, three shapes of compile, and nothing else:

| Contract | Ledger it judges | Shape of compile |
|---|---|---|
| `WEIGHT_PATCHES_V2` | `weight-patches/v2` | literal delta edits only (`UPDATE WEIGHTS`) |
| `WEIGHT_PATCHES_V3` | `weight-patches/v3` | `remove_direction` results only (`REMOVE DIRECTION`) |
| `WEIGHT_PATCHES_V3` | `weight-patches/v3` | `remove_direction` results on some tensors and literal deltas on others |

`WEIGHT_PATCHES_V4` still means exactly `weight-patches/v4`, and judges it as before except for three corrections to
the shared code, listed in §13. No contract reads another's ledger, and
no ledger is converted or "upgraded" to another schema to be judged. `weight-patches/v1` (one target, read by the
engine, never written) is judged by none of them.

Each contract is asked with a base container, a compiled container and, optionally, the `.vlp` patch the container was
compiled from. The answer is `larql-verify-ledger/1` ([contract-protocol.md](contract-protocol.md)), with the two
halves of v4 §7: **numerical satisfaction**, and **ledger integrity and provenance**.

## 2. What the engine writes

This section restates the engine's writer (`build_ledger` and `resolve_edits` in the weight-patch `apply` module, the
ledger types beside it, and the LQL specification §3.5.1 and §3.5.2). It is the format as read from that code, not a
format defined here: see §10.

### 2.1 Which schema

One compile writes one ledger, named by `index.json`'s `weight_patches` field. Its schema follows the ops compiled:

- `weight-patches/v4` when any tensor is edited by an `add_low_rank_columns` op;
- otherwise `weight-patches/v3` when any edit is a result-bits edit, which is when any `remove_direction` op was
  compiled;
- otherwise `weight-patches/v2`: every edit is a delta.

### 2.2 Top level

`schema`, `rounding` (always `bf16-rne-of-exact-sum`), `writer`, `source`, `targets`, `origins` (omitted when empty)
and `result`. `source` and `result` are exactly as in a v4 ledger: the base's model name, the SHA-256 of its
`index.json`, the payload and segment digests of the edited canonical representation before and after, and one entry
per representation re-derived from it.

### 2.3 Targets

Every target of a v2 or v3 ledger is an **element target**: `object`, `representation`, `tensor`, `dtype` (`BF16`),
`shape` (`[rows, cols]`) and `edits`. There is no `operations`, `base_tensor_sha256`, `result_tensor_sha256` or
`changed_elements`: those belong to a v4 operation target. All targets name one canonical BF16 representation, each
tensor is the subject of one target, and no element is listed twice. All targets together list at most
`MAX_PATCH_ELEMENTS` = 2^18 elements.

A tensor is edited by deltas or by `remove_direction` ops, never by both, so the edits of one target are all of one
form.

### 2.4 A delta edit

```json
{"row": 5, "col": 12, "delta": 0.1, "delta_bits": 1036831949, "old_bits": 48355, "new_bits": 15764}
```

`encoding` is absent (the reader also accepts `"encoding": "delta"`). `delta_bits` is the delta, an IEEE binary32 bit
pattern; `delta` is its decimal rendering in the fewest digits that read back to the same binary32, so `0.1` for
`0x3dcccccd`, and it may be written with an exponent (`1e-10`). `old_bits` is the stored BF16 element in the base and
`new_bits` the element the compile wrote:

`new_bits = RNE_bf16(exact(old_bits) + exact(delta_bits))`

The two values are added as real numbers and the sum is rounded **once** to BF16 (8 significant bits, exponents −126
to 127, gradual underflow with quantum 2^−133), to nearest, ties to even. A sum whose rounded magnitude is 2^128 or
more overflows, and that includes the tie at the top of the range. An exact zero sum is `+0`, except that `−0 + −0` is
`−0`; a nonzero sum that rounds to zero keeps its own sign. The stored element and the delta are both finite. A delta
that changes no bits (a delta of zero on a nonzero element, or one far below half a step) is still recorded, with
`new_bits = old_bits`.

### 2.5 A result-bits edit

```json
{"row": 0, "col": 33, "delta": 0.0, "delta_bits": 0, "old_bits": 48398, "new_bits": 48384, "encoding": "result_bits"}
```

`new_bits` is the BF16 element a `remove_direction` op produced. There is no delta: `delta` is `0.0` and `delta_bits`
is `0`. The target lists **exactly the elements whose bits the ops change**, and no others. The op is not in the
ledger. Its digest (v4 §3a) is in `origins`.

### 2.6 `origins`

The digest of every `.vlp` weight op the compile was assembled from, in the order the compile took them: one per
`weight_delta` op (§3.1) and one per `remove_direction` op. A compile driven through LQL always records them. The
engine's library interface can compile with none, and then the field is absent.

## 3. The patch

A `.vlp` file is JSON with an `operations` list. Three kinds of entry are stored-weight ops.

### 3.1 `weight_delta`

```json
{"op": "weight_delta", "rounding": "bf16-rne-of-exact-sum",
 "source": {"model": "…", "index_sha256": "…", "representation": "…", "payload_sha256": "…"},
 "tensor": "…", "edits": [{"row": 5, "col": 12, "delta": 0.1, "delta_bits": 1036831949}], "digest": "…"}
```

`digest` is the lowercase hex SHA-256 of this UTF-8 text, every line ending in `\n`, the edits sorted by `(row, col)`
and one line each:

```text
weight_delta/v1
rounding={rounding}
model={source.model}
index_sha256={source.index_sha256}
representation={source.representation}
payload_sha256={source.payload_sha256}
tensor={tensor}
{row},{col},{delta_bits}
```

Several `weight_delta` ops may edit one tensor (on distinct elements). The ledger merges them into that tensor's one
target.

### 3.2 `remove_direction` and `add_low_rank_columns`

`"op": "structured_weight"` and `"op": "low_rank_columns_weight"`, with the digests of v4 §3a.

### 3.3 Which ops of a supplied patch are read

A patch given to `WEIGHT_PATCHES_V2` or `WEIGHT_PATCHES_V3` is **held to the ledger in full**: it must be exactly the
ops the container was compiled from, or the answer is not `satisfied`. Every entry of `operations` is read.

| Entry | `WEIGHT_PATCHES_V2` | `WEIGHT_PATCHES_V3` |
|---|---|---|
| `weight_delta` | held to the ledger's delta edits (§7 step 8) | the same |
| `remove_direction` | a ledger violation: a delta-only compile has none | held to `origins`, and the tensor is recomputed from it (§7 step 7) |
| `add_low_rank_columns` | a ledger violation: a compile with one writes `weight-patches/v4` | the same |
| any other kind, or an entry that is not an object | the patch cannot be used | the same |

The patch's header (`version`, `base_model`, `base_checksum` and the descriptive fields) is not read: each op binds
itself to the base through its `source` and its digest.

`WEIGHT_PATCHES_V4` is different, and is not changed here: it reads only the `remove_direction` ops of a patch. A
low-rank op is judged from the ledger's own record of it, and a `weight_delta` op in the patch is not compared with
the ledger's delta edits. Under v4 the answer therefore does not attest the patch as a whole.

For a delta the patch adds nothing to the arithmetic: the ledger carries every delta as an edit and the edit is what
is recomputed. Holding the patch's `weight_delta` ops to those edits is cheap and exact (same tensor, same elements,
same `delta_bits`), and it is what makes `origins` checkable for a delta, so it is done.

## 4. What a `satisfied` answer establishes

All of these, for the base and compiled containers given:

1. **Every declared edit was applied.** At each element a delta edit lists, the compiled container holds the BF16
   nearest the exact sum of the base's stored element and the delta, recomputed here. At each element a
   `remove_direction` op of the patch changes, it holds that op's result, recomputed here from the base.
2. **No undeclared element of an edited tensor changed.** Every element of every tensor the ledger names is compared,
   not only those listed: an element no edit lists must hold the base's bits.
3. **No other tensor of the edited representation changed**, byte for byte.
4. **The edited representation has the base's tensor layout**: the same tensor table (names, dtypes, shapes, offsets,
   lengths, in the same order) and the same segment header bytes.
5. **Nothing else in the container differs from the base** except the differences §5 lists, each held to the rule
   §5 gives for it. A difference that is not on that list violates the ledger half.
6. **The ledger is bound to this base**: `source.index_sha256` is the SHA-256 of the base's `index.json`,
   `source.model` its model, `source.canonical` the edited representation's payload and segment digests in the base,
   and `source.derived` its re-derived representations as the base's index records them.
7. **The ledger describes this container**: `result.canonical` and `result.derived` are the compiled container's
   digests, and each re-derived representation names the compiled canonical payload as its source.
8. **The ledger's own record is right**: every `old_bits` is the base's element, every `new_bits` the recomputed
   one, every `delta` reads as its `delta_bits`, and under v3 the result-bits edits are exactly the elements the
   patch's ops change.
9. **The ledger is of the form its schema is written for** (§7 step 3).
10. **When a patch is given, it is the one the container was compiled from**, as far as the ledger can show: every
    weight op in it is intact, bound to the base and listed in `origins`; every origin is an op of the patch; its
    delta ops are exactly the ledger's delta edits; its removals are on the tensors the ledger records by result
    bits; it holds no low-rank op.

`WEIGHT_PATCHES_V3` is `satisfied` only with the patch (§7 step 7). `WEIGHT_PATCHES_V2` can be `satisfied` without
one; items 1 to 9 are then established and item 10 is not.

## 5. Allowed differences

A container is its regular files, by relative path and bytes, and its directories. This is everything that may
differ between the base and a compiled container under these contracts. **Nothing else may**: any other difference
violates the ledger half. The list is not "whatever the compiler changed": each item has a rule, and a wrong value in
an allowed place is caught, except where the last column says the item is not judged.

"The edited representation" is the one canonical BF16 representation the ledger's targets name. "A re-derived
representation" is one whose index entry in the **base** has `compiled_from` equal to it.

| # | What may differ | Why a compile changes it | What is checked of its new value | Judged by |
|---|---|---|---|---|
| D1 | The edited representation's segment file: the elements of the tensors the ledger names | It holds the edited weights | **Recomputed independently.** Every element of every named tensor is compared with the base and the declared edits (§7 steps 6, 7). The header bytes and the tensor table must be the base's. Every other tensor must be the base's bytes | this contract |
| D2 | `index.json`: the edited representation's `payload_sha256` and `segment_sha256` | The segment's bytes changed | **Recomputed**: both are the digests of the compiled segment's bytes, taken by the validating reader. The ledger's `result.canonical` must equal them | this contract |
| D3 | Each re-derived representation's segment file | The compiler re-derives the pack from the edited weights | **Chained by digest only.** The file must parse (framing, lengths, an unambiguous table), its tensor table must be the base's (names, dtypes, shapes, in order), and its bytes must hash to D4. Its header bytes must be the base's. Its content is not recomputed | `Q8_0_PACK_DERIVATION` for the pack's bytes |
| D4 | `index.json`: each re-derived representation's `payload_sha256` and `segment_sha256` | The pack's bytes changed | **Recomputed** from the compiled pack's bytes by the reader. The ledger's `result.derived` entry must equal them, with the entry's `encoding` and `encoder` | this contract |
| D5 | `index.json`: each re-derived representation's `source_representation_digest` | The pack now derives from the edited weights | **Held to another value**: it must equal D2's `payload_sha256`, in the index and in the ledger's `result.derived` | this contract |
| D6 | `index.json`: the top-level `weight_patches` field | The compile names its ledger | **Held to a rule**: a string, a relative path inside the container, naming a file the base does not hold (D7) | this contract |
| D7 | The ledger file D6 names: a new file | The compile writes its record | **Judged in full**: it is the claim, and every value in it is checked against the containers and the recomputation | this contract |
| D8 | `candidate.json` at the top level: its bytes, and whether it is present | The representation compiler writes it for the container it produced, with artifact hashes and a `locator_hint` naming where it was written. A compile that re-derives a pack regenerates it; one that re-derives nothing writes none, so it may be present in one container only | **Not judged at all** | no verification contract. It is not an authority for anything these contracts check, and nothing here should be read as vouching for it |
| D9 | `index.json`: its bytes, where the parsed values are equal | The compile rewrites the file | **Not judged**: the file is read as strict JSON and compared value by value, with types, so key order and whitespace are not seen. Every value outside D2, D4, D5 and D6 must equal the base's | this contract, for the values |

What follows from the list, stated so that no reader has to derive it. Each of these violates the ledger half:

- a file present in only one container, other than D7 and D8: a new top-level file, a new segment file, a removed
  file;
- any byte of a file that is not D1, D3, D7, D8 or `index.json`: `system_graph.json`, a tokenizer or template file,
  a knowledge store, a segment of another representation;
- any `index.json` value outside D2, D4, D5 and D6: a new field, a removed field, a changed field of any
  representation's entry (its `segment` path, `tensor_count`, `payload_bytes`, `encoding`, `codec`, `encoder`,
  `compiled_from`), an entry added or removed, a changed type;
- a different set of re-derived representations;
- a directory present in only one container;
- a ledger whose path is a file of the base, so that "the ledger" replaces a file the base holds.

A segment file that is neither edited nor re-derived is compared as a whole file and never parsed (v4 §7 step 4). If
such a file is missing from the compiled container while its index entry remains, the container is not intact and
nothing is judged (`could_not_evaluate`), which is not a pass.

This list is the one `WEIGHT_PATCHES_V4` applies, with two additions made here and not there: a directory present
in only one container, and a ledger path that is a file of the base. Both are reported as findings against the v4
route in the change that introduced this document.

One difference a real compile could make is **not** on the list: the `encoder` of a re-derived representation's
index entry. The engine's ledger types allow the base's pack to have been written by an earlier revision of the
encoder than the one that re-derives it. Such a compile would be reported as `violated` here and under v4. It is
left that way: admitting it means deciding which encoder revisions are acceptable, which is the pack contract's
subject.

## 6. What it does not establish

- **The re-derived (quantised) representations.** Their digests are chained to the compiled canonical payload; their
  content is not recomputed. A re-derived Q8_0 pack with arbitrary bytes and consistent digests passes these
  contracts. That the pack is the declared derivation of the edited weights is `Q8_0_PACK_DERIVATION`'s claim
  ([q8-0-pack-derivation.md](q8-0-pack-derivation.md)).
- **`candidate.json`.** Every compile regenerates it; it is not judged (v4 §7 step 4).
- **`origins`, when no patch is given.** A `weight_delta` digest in `origins` cannot be recomputed from the ledger, so
  under `WEIGHT_PATCHES_V2` without a patch `origins` is checked only for its form (a list of SHA-256 strings).
  Emptying it or replacing an entry is not detected. With the patch it is.
- **That the edit is the one anyone intended.** The contracts judge the container against its own ledger and, when
  given, a patch. Where the deltas or the operand came from (an operand's `origin` text is covered by its digest and
  read by nothing) is not established.
- **A compile assembled from several patches.** One patch is read. A container compiled from two cannot be shown to
  be either's.
- **A v3 ledger written with no `origins`** (possible through the engine's library interface, never through LQL). No
  patch can be tied to it, so it is never `satisfied`.
- **Who is answering.** The verifier is not authenticated ([independence.md](independence.md)).

## 7. The judgement

Performed with the base container, the compiled container, the compiled container's ledger and, if given, the patch.
Nothing the engine wrote is used as an expected value: each recorded value is checked against what is recomputed or
read from the containers.

1. **Read the inputs.** Both containers must be directories of regular files, read as
   [contract-protocol.md](contract-protocol.md#how-a-container-directory-is-read-pack-residency-and-ledger-contracts)
   states (a symlink is followed only to a regular file inside the container or its Hugging Face snapshot's `blobs/`,
   and refused otherwise), with the edited and re-derived segments passing the validating reader and every other
   segment file hashing to its own index entry (v4 §7 step 4). The compiled `index.json` must name a ledger, and the
   base's must not: a container with a ledger is a compiled edit, the engine refuses to stage an edit on one, and so
   it is the base of no compile (not judged). The ledger is read as strict JSON (no duplicate keys, no non-finite
   constants).
2. **The schema must be the contract's.** `WEIGHT_PATCHES_V2` judges `weight-patches/v2` and nothing else;
   `WEIGHT_PATCHES_V3` judges `weight-patches/v3` and nothing else. Any other schema, including the other contract's
   and v4, is not judged.
3. **The form must be the schema's.**
   - No target is an operation target (both).
   - No edit is a result-bits edit (`WEIGHT_PATCHES_V2`).
   - At least one edit is a result-bits edit (`WEIGHT_PATCHES_V3`).
   - No target mixes delta and result-bits edits (both).
   - A result-bits edit carries no delta: `delta_bits` is `0` and `delta` reads as `+0` (both).
   - `rounding` is `bf16-rne-of-exact-sum` (both). A rule the contract does not know is not a violation: neither
     half is evaluated (§8).
   - `origins`, when present, is a list of SHA-256 hex strings (both).
4. **The source binding** (v4 §7 step 1): `source` against the base, as §4 item 6.
5. **Each target, before any arithmetic**: one canonical BF16 representation for all targets; the tensor is a BF16
   matrix of the base with the recorded object and shape, and is present with that shape in the compiled container;
   `edits` is non-empty; each edit has integer `row`, `col`, `old_bits`, `new_bits`, `delta_bits` in range and a
   numeric `delta`; no element twice; an `encoding` that is absent, `delta` or `result_bits`; at most
   `MAX_PATCH_ELEMENTS` elements over all targets.
6. **A delta target is recomputed from the ledger.** For each edit: `old_bits` must be the base's element;
   `delta_bits` must be finite; `delta` must read as `delta_bits` (v4 §3b); the base element must be finite; the sum
   must not overflow; `new_bits` must be the recomputed bits. The expected tensor is the base with the recomputed
   bits at the listed elements, and **every element** of the compiled tensor is compared with it.
7. **A result-bits target is recomputed from the patch.** Without a patch it cannot be: the container is held to the
   recorded bits under the ledger half, and the numerical half is `could_not_evaluate`. With the patch, each
   `remove_direction` op on that tensor is admitted as v4 §7 step 2 admits a ledger's op (source binding, §3a digest,
   §5 bounds, columns disjoint from the other ops on the tensor) and must be one of `origins`. Each is applied to the
   **base** (v4 §4), the expected tensor is the base with the ops' columns replaced, the ledger's edits must be
   exactly the elements that change, with the base's `old_bits` and the recomputed `new_bits`, and every element of
   the compiled tensor is compared. An op that changes no element is refused. A result-bits tensor with no
   `remove_direction` op in the patch is not recomputed.
8. **The patch is held to the ledger in full** (§3.3), when one is given:
   - a `remove_direction` op on a tensor the ledger does not record by result bits, or one that names no tensor, is a
     violation;
   - an `add_low_rank_columns` op is a violation;
   - each `weight_delta` op must have exactly the fields of §3.1, the known rounding rule, the base's source binding,
     an intact digest that is one of `origins`, and edits whose `delta` reads as its `delta_bits`;
   - on each tensor, the elements of the patch's `weight_delta` ops must be exactly the ledger's delta edits, with
     the same `delta_bits`, none twice; a `weight_delta` op on a tensor the ledger does not edit by deltas is a
     violation;
   - every entry of `origins` must be the digest of a weight op of the patch.
9. **Everything else** (v4 §7 step 4): §4 items 3, 4, 5 and 7, by the list of §5. The two containers hold the
   same directories, and the ledger's path is not a file of the base.
10. **Two verdicts.** Numerical satisfaction is steps 6 and 7. Ledger integrity and provenance is steps 2 to 5, 8 and
    9, and the ledger's own record in steps 6 and 7. The answer is `satisfied` only when both are.

## 8. Which half a refusal belongs to

This follows v4 §7a. "Not evaluated" is `could_not_evaluate`; a half that is not evaluated is never `satisfied`.

| Defect | Numerical half | Ledger half |
|---|---|---|
| A container that cannot be read, a symlink in one that is refused, no ledger named, an input nested too deeply to read | not evaluated | not evaluated |
| A ledger of another schema than the contract's | not evaluated | not evaluated |
| An edit `encoding` or a `rounding` rule the contract does not know | not evaluated where it is needed | not evaluated |
| A ledger that is not JSON, has a duplicate key, or is not an object | not evaluated | violated |
| An operation target; under v2, a result-bits edit; a target mixing the two edit forms | not evaluated for that tensor | violated |
| A v3 ledger with no result-bits edit | judged | violated |
| A result-bits edit that carries a delta; `origins` not a list of SHA-256 strings | judged | violated |
| A malformed edit, an element listed twice, an edit outside the tensor, a target with no edits or that does not describe the tensor, too many elements | not evaluated for that tensor | violated |
| A non-finite `delta_bits`, a non-finite base element, a sum that overflows | not evaluated for that tensor | violated |
| A stale source binding (the ledger is bound to another base) | not evaluated | violated |
| A wrong `old_bits`, a wrong `new_bits`, a `delta` that does not read as its `delta_bits` | judged | violated |
| A compiled element that is not the recomputed one, listed or not, where the record is sound | violated | judged |
| A recorded result and the compiled element changed together | violated | violated |
| A tensor the ledger does not name, another representation, a file or an index field changed; a tensor table that differs; a file or a directory in only one container; a ledger path that is a file of the base | judged | violated |
| A segment file missing from a container whose index still names it | not evaluated | not evaluated |
| A v3 ledger with no patch | not evaluated | judged, the container held to the recorded bits |
| A patch that cannot be read, is not JSON, is a symlink, has no `operations` list, an entry that is not an object or an op of a kind not in §3.3 | not evaluated | judged without the patch |
| A patch `remove_direction` op that is damaged, bound to another base, not in `origins`, or overlapping another | not evaluated for that tensor | violated |
| A patch `remove_direction` op on a tensor not recorded by result bits; a low-rank op; a `weight_delta` op that is damaged, stale, not in `origins`, on another tensor, or whose elements are not the ledger's; an origin the patch does not hold | judged | violated |
| A patch that lacks one of several removals a tensor needs | violated | violated |
| A patch op with a field or a numerics profile the contract does not know | not evaluated for that tensor | not evaluated |

A patch never makes a violation disappear: one that cannot be used leaves the ledger judged as it is without a patch,
and the numerical half is then `could_not_evaluate` even for a v2 ledger that needs none, because evidence was asked
for and could not be used.

One classification is inherited that could be argued the other way: a ledger that is not valid JSON violates the
ledger half, while an `index.json` that is not valid JSON leaves both halves not evaluated. Neither is ever
`satisfied`.

## 9. Wrong pairings

The contract is not chosen by the ledger's schema string alone. Each of these is not `satisfied`:

| Asked | Given | Answer |
|---|---|---|
| `WEIGHT_PATCHES_V2` | a v3 or v4 ledger | `could_not_evaluate` (step 2) |
| `WEIGHT_PATCHES_V3` | a v2 or v4 ledger | `could_not_evaluate` (step 2) |
| `WEIGHT_PATCHES_V4` | a v2 or v3 ledger | `could_not_evaluate` (as before) |
| `WEIGHT_PATCHES_V2` | a v2 ledger and a patch with a `remove_direction` op | `violated` (step 8) |
| `WEIGHT_PATCHES_V2` or `V3` | a patch with an `add_low_rank_columns` op | `violated` (step 8) |
| `WEIGHT_PATCHES_V2` or `V3` | a patch with an op on a tensor the ledger does not name | `violated` (step 8) |
| `WEIGHT_PATCHES_V2` or `V3` | another compile's patch | `violated` (step 8) |
| `WEIGHT_PATCHES_V3` | a v3 ledger and no patch | `could_not_evaluate` (step 7) |
| `WEIGHT_PATCHES_V3` | a v3 ledger and a patch missing or adding a removal | `violated` (steps 7, 8) |
| `WEIGHT_PATCHES_V2` | a ledger relabelled `weight-patches/v2` that holds an operation target or a result-bits edit | `violated` (step 3) |
| `WEIGHT_PATCHES_V3` | a ledger relabelled `weight-patches/v3` that holds an operation target or no result-bits edit | `violated` (step 3) |

## 10. Independence and its limit

**What is independent.** The delta arithmetic is the verifier's own: the exact rational sum, rounded once. The
`remove_direction` arithmetic is the verifier's own implementation of the frozen v1 numerics. No engine code is
called, linked or translated, and the engine's ledger is never the source of an expected value. The tests hold the
delta rule to vectors derived in the tests themselves from exact rationals, and the removal arithmetic to the frozen
vectors of `conformance/vlp-numerics/v1` and to the exact-rational route of `STRUCTURED_REMOVE_DIRECTION`.

**What is not.** Before this document, `weight-patches/v2` and `v3` and the `weight_delta` op were specified nowhere
but in the engine. Section 2 and §3.1 were written by reading the engine's writer, its consistency check and the LQL
specification. So the format this contract verifies is the format the engine implements, as its own source states
it: a mistake in what the engine takes a v2 or v3 ledger to mean would be reproduced here, not caught. The author of
the verifier code had read that source, including the engine's own rounding routine, which the verifier does not
follow (the engine adds in binary64 and repairs ties; the verifier adds rationals).

The contract and the code were written together, by one author, in one change. The tests establish conformance to
this document and sensitivity to deliberate corruption. They are not an independent replication against a contract
frozen beforehand.

The fixtures in `verifier/conformance/weight_patches_v2v3/` were compiled by the engine. They are inputs to judge,
never oracles.

## 11. Where the engine and v4 §3b differ or are silent

1. **v2 and v3 had no specification.** v4 §3b reads an element target "exactly as in v2/v3", which no document
   defined. §2 here is that definition, taken from the engine.
2. **A result-bits edit's `delta`.** The engine writes `delta: 0.0`, `delta_bits: 0`, and its consistency check
   reports a result-bits edit that carries a delta. v4 §3b does not say so, and `WEIGHT_PATCHES_V4` does not check
   it. These contracts do (step 3).
3. **Mixed edit forms in one target.** The engine never writes one (a tensor takes deltas or structured ops, never
   both). v4 §3b is silent, the engine's consistency check does not look, and `WEIGHT_PATCHES_V4` leaves such a
   tensor's numerical half not evaluated without violating the ledger half. These contracts violate the ledger half
   (step 3).
4. **`rounding`.** The engine's consistency check requires the known rule in every ledger. v4 §3b does not mention
   the field, and `WEIGHT_PATCHES_V4` checks it only when it recomputes a delta, so a ledger with no delta edit is
   not held to it. These contracts require it always (step 3).
5. **`origins`.** v4 §3 lists it at the top level and defines nothing about it; the v4 route uses it only to tie a
   patch's `remove_direction` op to the ledger, and accepts a malformed value. The engine writes one digest per
   weight op, `weight_delta` ops included. These contracts check its form always and its content against the patch
   (steps 3 and 8).
6. **The `weight_delta` digest.** v4 §3a defines the digests of the two structured ops only. §3.1 here states the
   delta op's, from the engine.
7. **A delta that changes nothing.** The engine records it, with `new_bits = old_bits`; v4 §3b does not say whether
   such an edit may be listed. It is accepted here, as the engine writes it. A `remove_direction` op that changes
   nothing is refused by the engine and here.
8. **The order of edits.** The engine writes a target's edits in op order, then row-major. Nothing here depends on
   the order.

## 12. Fixtures

`verifier/conformance/weight_patches_v2v3/`, compiled by `make_fixtures.py` from
`verifier/conformance/structured_patch/base`, with each case's statements, operands, saved patch and a provenance
record:

| Case | Ledger | Ops |
|---|---|---|
| `deltas` | v2 | seven deltas over two tensors, among them `0.1`, `-0.3`, `-0.0625` and one that changes no bits |
| `removal` | v3 | one `remove_direction` on three columns of one tensor |
| `removal_two` | v3 | two `remove_direction` ops on disjoint columns of one tensor, in two profiles |
| `mixed` | v3 | three deltas on one tensor and a `remove_direction` on another |

## 13. Corrections to the shared `WEIGHT_PATCHES_V4` code

These contracts reuse the v4 judge. Three inputs that judge mishandled are corrected, and the corrections apply on the
v4 route too. None of them lets a v4 answer become `satisfied`; each is tested on the v4 route
(`tests/test_weight_patches_v4.py`).

1. **A `source.canonical` or `result.canonical` that is not an object.** It raised an uncaught error. It is now a
   ledger violation: every field is reported as not the container's. This was fixed on main independently
   (`canonical_object`, 966455b5), and this change takes main's form. It is tested there
   (`test_a_canonical_section_that_is_not_an_object_is_a_ledger_finding`).
2. **A `rounding` rule the contract does not implement, on a ledger with delta edits.** The edited elements were
   expected to keep the base's bits, so a container holding exactly the recorded `new_bits` was reported `violated`
   with a false finding. They are now expected to hold the recorded `new_bits`, and both halves are
   `could_not_evaluate`, as v4 §7a says of a profile it does not know. A container that differs from the recorded
   bits is still `violated`. **This changes a v4 answer from `violated` to `could_not_evaluate`.**
3. **A target that names no tensor.** It was already a ledger violation; the numerical half, which said `satisfied`,
   now says `could_not_evaluate`. The overall answer was and is `violated`.
