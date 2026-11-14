# Weight-patch ledger `weight-patches/v4`: operation targets

**Status: draft for review (SG-EDIT-1 I2).** This document is the contract an independent verifier of a compiled
structured weight edit is written against. It does not change any numerics: the arithmetic of every op is the frozen
`conformance/vlp-numerics/v1` (`remove_direction`) and `v2` (`add_low_rank_columns`) specifications.

## 1. Why a new schema

`weight-patches/v2` and `v3` record a compiled edit **element by element**: every changed element with its old and new
BF16 bits. That is the right record for a delta edit, which names elements. A structured op names columns and operands
instead, and its footprint can be a whole matrix: a rank-one edit of a Gemma 3 4B `down_proj` writes up to 26,214,400
elements, which a per-element JSON ledger cannot reasonably hold, while the op itself is two vectors and a column list.

`v4` records such a tensor by the **operation** that produced it, the exact operands, and digests that bind the record to
both the base and the compiled bytes. What was requested (the op) and what was written (the tensor) stay distinct, and
both are checkable.

## 2. When it is written

A compile writes `v4` when, and only when, at least one tensor is an **operation target**. A tensor is an operation
target when at least one `add_low_rank_columns` op edits it. Every structured op on that tensor (`add_low_rank_columns`
or `remove_direction`, necessarily on disjoint columns) is then recorded as an operation of that target.

Nothing else changes: a compile with only delta edits writes `v2`, and one with only `remove_direction` ops writes `v3`,
both byte for byte as before. In a `v4` ledger, tensors edited by deltas or only by `remove_direction` stay element
targets (`edits`), exactly as in `v2`/`v3`.

## 3. Shape

The top level is that of `v3` (`schema`, `rounding`, `writer`, `source`, `targets`, `origins`, `result`), with
`schema = "weight-patches/v4"`. An **operation target** is:

```json
{
  "object": "<graph object>",
  "representation": "<canonical representation id>",
  "tensor": "<tensor name>",
  "dtype": "BF16",
  "shape": [rows, cols],
  "operations": [
    {
      "op": "low_rank_columns_weight",
      "...": "every field of the .vlp op, exactly as the patch records it (source, tensor, numerics, rank, a, b, columns, origin?, digest)",
      "operands_sha256": { "a": "<hex>", "b": "<hex>" },
      "changed_elements": 1234
    }
  ],
  "base_tensor_sha256": "<hex>",
  "result_tensor_sha256": "<hex>",
  "changed_elements": 1234
}
```

- **`operations`** lists the ops in the order the compile was given them. Each carries the complete `.vlp` op (tag `op`
  is `low_rank_columns_weight` or `structured_weight`), so the container alone holds everything needed to reconstruct
  the result: no patch file is required.
- **`operands_sha256`**: for each operand, SHA-256 over its elements' bits as little-endian bytes of the profile's
  working width (8 bytes for the f64 profiles, 4 for `f32-seq-rne1/v1`), in stored order. Keys are `a`, `b` for
  `add_low_rank_columns` and `u`, `alpha` for `remove_direction`.
- **`changed_elements`** per op: elements of the base whose bits that op changes (elements it writes back with their
  own bits are not counted). The target's `changed_elements` is their sum. It is zero only for a sham (§4a).
- **`base_tensor_sha256`** / **`result_tensor_sha256`**: SHA-256 over the tensor's stored payload (row-major BF16,
  little-endian), in the source and in the compiled container.
- An operation target has no `edits`; an element target has no `operations`, `base_tensor_sha256`,
  `result_tensor_sha256` or `changed_elements`.

## 3a. The op digest

Each `.vlp` op carries `digest`: the lowercase hex SHA-256 of the UTF-8 text below, built from the op's own fields.
Every line ends with `\n`, including the last. `columns` are decimal integers joined by `,`; operand lists are their
hex strings joined by `,` exactly as stored; `origin` is `-` when absent and `+` followed by its text when present.
`{source.*}` are the fields of the op's `source` object.

`add_low_rank_columns` (`"op": "low_rank_columns_weight"`):

```text
structured_op/add_low_rank_columns/v2
model={source.model}
index_sha256={source.index_sha256}
representation={source.representation}
payload_sha256={source.payload_sha256}
tensor={tensor}
numerics={numerics}
rank={rank}
columns={columns}
a={a}
b={b}
origin={origin}
```

`remove_direction` (`"op": "structured_weight"`):

```text
structured_op/remove_direction/v1
model={source.model}
index_sha256={source.index_sha256}
representation={source.representation}
payload_sha256={source.payload_sha256}
tensor={tensor}
numerics={numerics}
alpha={alpha}
columns={columns}
operand={operand}
origin={origin}
```

The digest detects damage and casual edits; it is not a signature. An op is **intact** when its recomputed digest
equals `digest`.

## 3b. Element targets in a `v4` ledger

An element target (`edits`, no `operations`) is read exactly as in `v2`/`v3`:

- `encoding` absent or `delta`: `new_bits = RNE_bf16(exact(old_bits) + exact(delta_bits as f32))`, one rounding, ties to
  even, IEEE signed zeros, refused if the result is not finite. `delta` must equal the f32 value of `delta_bits`.
- `encoding: result_bits`: `new_bits` is the declared result of a `remove_direction` op that the ledger does **not**
  carry (a tensor edited only by removals stays an element target, §2). Only the op's digest is in `origins`, so an
  independent verifier can hold the container to the recorded bits but cannot recompute them from the ledger alone;
  it reports that tensor's numerical half as not evaluated unless it is given the patch.
- No element is listed twice; every element is inside the tensor; all element targets together list at most
  `MAX_PATCH_ELEMENTS` = 2^18 elements. Operation targets do not count against it.

## 4. Composition

The ops of one target have pairwise disjoint column sets, and each reads and writes only its own columns. The compiled
tensor is therefore the base with, for each op, the columns it writes replaced by that op's result computed on the
**base**. Order does not change the bytes.

## 4a. Shams: zero-valued ops

An `add_low_rank_columns` op is **zero-valued** when no element has a term with both factors nonzero: for every
`k`, column `k` of `a` or row `k` of `b` is all zero (`−0` counts as zero). This is decided from the operands alone,
before any arithmetic, and under the v2 identity rule (v2 SPEC §4) such an op leaves its tensor byte-identical,
given the §5 precondition of v2 (no NaN or infinity inside rows × J, which is still refused).

A zero-valued op is the declared form of a **sham** edit (SG-EDIT-1 §4.3). It compiles: its tensor is an operation
target with `changed_elements = 0` and `result_tensor_sha256 = base_tensor_sha256`. Every other op that changes no
element (a `remove_direction` op, or a nonzero op whose results all round back to the stored bits) is refused, as
before: an edit that is not declared as an identity must change something. A target is a sham only if every one of
its operations is zero-valued; a zero-valued op never changes an element, and recording that it did is a defect.

## 5. Bounds

A compact record must not become an instruction to do unbounded work. A structured op is refused before any
arithmetic when it exceeds any of these:

| bound | value | meaning |
|---|---|---|
| `MAX_STRUCTURED_RANK` | 256 | the rank `K` of an `add_low_rank_columns` op |
| `MAX_STRUCTURED_OPERAND_ELEMENTS` | 2^24 | operand elements of one op (`len(a) + len(b)`, or `len(u) + 1`) |
| `MAX_STRUCTURED_TENSOR_ELEMENTS` | 2^30 | `rows × cols` of a tensor a structured op edits |
| `MAX_STRUCTURED_WORK` | 2^32 | multiply-adds of one compile's structured ops together (`rows × |J| × K` per low-rank op, `2 × rows × |S|` per removal) |

Each bound is inclusive: a value equal to it is admitted.

`MAX_PATCH_ELEMENTS` still bounds every element target. Operation targets do not count against it.

## 6. What `check_ledger` establishes (consistency, not correctness)

For an operation target, the writer's own consistency check confirms: the stored tensor's bytes hash to
`result_tensor_sha256`; every op is intact (its digest), names this tensor, and is bound to the ledger's source (index
digest, representation and canonical payload digest); the ops' column sets are disjoint; every `operands_sha256`
matches the op's operands; the `changed_elements` add up; the bounds of §5 hold; and the result differs from the base,
unless the target is a sham (§4a), in which case the result equals the base and nothing is recorded as changed.
It does **not** recompute the op. That is the verifier's job.

## 7. What an independent verifier must do

Given the base container and the compiled container (and nothing from the compiler but the ledger it wrote), it must,
**with its own implementation of the frozen numerics**:

1. Check the base: its `index.json` digest and the edited representation's payload digest equal the ledger's
   `source`, and its tensor hashes to `base_tensor_sha256`.
2. Check every op: digest intact, bound to that base, operand digests as recorded, bounds of §5, columns disjoint.
3. Reconstruct the expected tensor from the base and the ops (§4), and compare **every element** of the compiled
   tensor, not only the elements the ops change: every element outside the ops' column footprint, and every element
   inside it that the v2 rules leave untouched, must equal the base bit for bit. A matching `result_tensor_sha256`
   alone is not sufficient, and neither is checking the elements recorded or expected as changed: the expected bytes
   of the whole tensor are recomputed, never taken from the ledger or from the compiler.
4. Confirm every other tensor, edited or not by name, of every representation not re-derived by the compile is
   byte-identical to the base, every file outside the edited and re-derived segments is unchanged apart from
   `index.json`, the ledger and `candidate.json`, and the re-derived representations derive from the compiled
   canonical payload the ledger names. In `index.json` only the edited and re-derived representations' digests,
   the re-derived `source_representation_digest` and the `weight_patches` pointer may differ.
   `candidate.json` is the representation compiler's record of the container it produced (artifact and graph
   hashes, segment digests, a `locator_hint` naming where it was written); every compile regenerates it. It holds no
   weights and is not an authority for anything the verifier checks, so it may differ and is not judged.
   A segment that is neither edited nor re-derived is checked as a **whole file**: byte-identical to the base's file
   of the same path satisfies step 4 for every tensor in it, and the verifier need not parse its tensor table. (A
   byte-identical file is a stronger check than tensor-by-tensor equality, and it judges a base whose untouched
   segments a strict reader could not parse, such as one with a duplicate tensor name.) Only the edited and
   re-derived segments must be parsed, and their tensor tables must be unambiguous.
5. Refuse malformed, overlapping, stale (bound to another base) or unsupported ops, and an unknown schema.
6. For a sham (§4a), decide zero-valuedness from the operands itself and confirm the compiled tensor is the base's
   bytes exactly.
7. Report **numerical satisfaction** (step 3) separately from **ledger integrity and provenance** (steps 1, 2, 4,
   5): a container whose bytes are right but whose ledger is inconsistent, or the reverse, is not verified, and the
   report says which half failed.

## 7a. Which half a refusal belongs to

- A malformed, stale, overlapping, out-of-bounds or wrongly targeted op, or a ledger record that contradicts the
  containers (digests, counts, operand digests, an element target that also has operations): the **ledger half is
  violated**, and the affected tensor's numerical half is **not evaluated** (nothing trustworthy to recompute).
- An unknown schema, op kind, op field or profile: **both halves not evaluated**; the verifier cannot read the claim.
- A wrong element, inside or outside the footprint, where the ops themselves are sound: the **numerical half is
  violated**.
- A changed tensor or file outside the edit: the **ledger half is violated** (step 4 belongs to it).

It must be shown to reject, at least: a flipped bit in a written element inside the footprint; a changed element
outside the footprint (including a `-0` to `+0` flip); a changed element in another tensor; a stale source binding; a
swapped or reordered `a`/`b`; a `b` in the wrong layout (`K × cols` instead of `K × |J|`); a tampered
`operands_sha256`; and a tampered `result_tensor_sha256` (v2 SPEC §8, extended to this ledger).
