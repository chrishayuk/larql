# Execution residency: `EXECUTION_RESIDENCY_V1`

A container's stored weights and the weights a run computes with are not always the same bytes. On the CPU the
loader decides, per matrix, what to hold in memory: the stored bytes, an exact widening of them, or a lossy Q8
derivation it manufactures at load. This contract judges that decision and its result, from a record the engine
writes of what it loaded (`larql-residency-record/1`) and the container the record is about:

> For every matrix operand the record names: was it made resident as the declared derivation of the container's
> stored bytes, and in the representation its size, the recorded cache size and the run's policy select?

## 1. Four claims, kept apart

| Claim | Established by |
|---|---|
| The canonical weights are the base with exactly the declared edit | `WEIGHT_PATCHES_V4` |
| What a run made resident is the declared derivation of the canonical weights, and the representation its policy selects | **this contract**, in two verdicts |
| A kernel was handed those resident bytes | not established here: a dispatch record, not yet specified |
| The kernel computed correctly with them | not established here: a numerical reference experiment |

A satisfied verdict here says the engine's account of what it loaded is consistent with the container, rule by
rule and choice by choice. It is the engine's own account: nothing here observes the process.

## 2. The record

`larql-residency-record/1` is one JSON object with exactly these fields:

| Field | Meaning |
|---|---|
| `schema` | `larql-residency-record/1` |
| `container`, `container_index_sha256` | The container and the SHA-256 of its `index.json` when it was opened |
| `selection_inputs` | `l2_bytes` (the cache size the size policy compared against), and `max_format`, `arithmetic`, `kquant_exec`: the `LARQL_CPU_MAX_FORMAT`, `LARQL_CPU_ARITHMETIC` and `LARQL_KQUANT_EXEC` settings, or null |
| `rules` | The engine's description of its rules. Informational: this contract uses its own |
| `operands` | One entry per matrix operand loaded |

Each operand has exactly: `order`, `object`, `tensor`, `shape`, `stored_dtype`, `stored_sha256` (of the stored
bytes the loader read), `representation_source` (`stored`, `auto` or `transient`: what the run asked for),
`requested_format`, `resident_format` (what it got), `manufactured`, `rule`, `resident_sha256` and
`resident_digest_layout`.

`order` is an integer and `shape` a list of integers; an operand without them is malformed. `stored_dtype` and
`stored_sha256` may be null (an operand the run read through an overlay of edits, which has no stored bytes of its
own), and `stored_dtype` may be `Q8_0` (the run read the object's stored pack). Neither is a form this contract
derives from: such an operand is not judged, in either verdict. The values of `order`, `requested_format`,
`container` and `rules` are not judged.

## 3. Derivation rules

Each is defined on a stored BF16 matrix `[rows, cols]`, row-major. A BF16 pattern `h` is the binary32 with bits
`h << 16`.

### 3.1 `q8-load/1` — the loader's Q8 (`resident_format` `Q8`, `manufactured` true)

Row by row, in blocks of 64 consecutive elements; a row's last block is short and no block spans two rows. For a
block:

1. `peak = max |w|`.
2. `scale = peak / 127`, one binary32 division, round to nearest even; `scale = 1` if `peak = 0`. The scale is a
   binary32.
3. `code = round_half_away(w / scale)`, one binary32 division per element, clamped to `[−127, 127]`.

Resident digest (`codes-i8-row-major||scales-f32le-row-then-block`): SHA-256 over the codes, one signed byte each in
row-major order, followed by the scales as little-endian binary32, row by row and block by block.

The rule is defined on finite elements. A tensor with a NaN or an infinity is outside it: the derivation of an
operand under `q8-load/1` from such a tensor is not judged. For finite input the clamp is never reached.

This is not the stored Q8_0 pack's rule (`Q8_0_PACK_DERIVATION`): the block is 64
elements, not 32; the scale is a binary32, not a binary16; the code is a division by the scale, not a product with
its reciprocal; the clamp is ±127.

### 3.2 `widen-f32/1` — exact widening (`resident_format` `F32`, `manufactured` false)

Each element widened to binary32. Nothing is rounded. Resident digest (`f32le-row-major`): SHA-256 over the values
as little-endian binary32, row-major.

### 3.3 `stored-bytes` — the stored bytes, copied (`resident_format` `Bf16`, `manufactured` false)

Resident digest (`stored-bytes`): SHA-256 of the stored bytes; it equals `stored_sha256`.

Any other rule, `stored-bytes-mapped` included (the engine's name for stored bytes mapped in place, for which it
records no resident digest), is one this contract does not derive.

## 4. Selection rule

For a stored BF16 matrix of `E = rows × cols` elements, cache size `L = l2_bytes`, under the default arithmetic arm:

| Condition | Rule |
|---|---|
| `4E < L` | `widen-f32/1` |
| otherwise, `2E ≥ L`, source `auto`, and `max_format` is not `bf16` | `q8-load/1` |
| otherwise | `stored-bytes` (or `stored-bytes-mapped`) |

So source `stored`, or the cap `bf16`, forbids the manufacture. `max_format` is compared after white space around
it is removed, and with its case kept: ` bf16 ` is the cap and `BF16` is not. The rule is specified for sources
`stored` and `auto` with `arithmetic` and `kquant_exec` unset (null or empty). The `transient` source and the other
arithmetic arms are not specified here.

This is the production CPU backend's rule, for a codec that declares the direct BF16 kernel. The record does not
say which backend made it, so a record from another backend is judged by this rule all the same.

The cache size is the record's. Whether it is the host's is not something a record can show.

## 5. What is checked, and the verdicts

**Before either verdict:** the record parses, has exactly the fields of §2, a positive integer `l2_bytes`, at least
one operand, and `container_index_sha256` is this container's. Otherwise both verdicts are `could_not_evaluate`: a
record about another container says nothing about this one.

**Derivation**, per operand: the tensor is a BF16 matrix of the object's canonical representation with the recorded
shape and dtype; `stored_sha256` is the digest of the container's tensor; `resident_sha256`, in the rule's layout, is
§3's result for those bytes, recomputed here; `resident_format` and `manufactured` are the rule's.

**Selection**, per operand: the operand's rule is the one §4 gives.

**Completeness**, both verdicts: no operand is recorded twice, and for every object the record covers, every
matrix of its canonical representation is in the record. Each missing matrix is one violation in each verdict.

An object the record does not name at all is **not** a defect: which objects a run loads is not something the
container shows. Such objects are listed in the answer (`uncovered_objects`) and are not judged, so a `satisfied`
answer is about the objects the record covers and no others.

| Defect | Derivation | Selection |
|---|---|---|
| A resident digest that is not the rule's result (altered codes or scales) | violated | — |
| A stale or foreign `stored_sha256` | violated | — |
| `resident_format`, `manufactured` or the layout not the rule's | violated | — |
| A shape or dtype that is not the stored tensor's; a tensor the container lacks | violated | not evaluated |
| A rule §4 does not give for the size, source, cap and cache size | — | violated |
| A cache size the choices do not follow from | — | violated |
| An unknown `representation_source` | — | violated |
| A missing operand; an operand recorded twice | violated | violated |
| A rule this contract does not derive (`stored-bytes-mapped`) that §4 places | not evaluated | judged |
| A rule neither §3 nor §4 knows | not evaluated | not evaluated |
| Null stored fields, a stored `Q8_0`, a non-finite tensor under `q8-load/1` | not evaluated | not evaluated (judged, for the non-finite tensor) |
| The `transient` source, an arithmetic arm or a K-quant mode | — | not evaluated |
| A malformed operand | violated | not evaluated |
| A malformed record; a record of another container | not evaluated | not evaluated |

`status` is `satisfied` only when both verdicts are; `violated` if either is; `could_not_evaluate` otherwise.
Nothing that was not judged is folded into a pass.

## 6. Not established

Listed in every answer:

- that the record is an honest account of what the process held;
- that any kernel consumed the resident bytes, or computed correctly with them;
- that the recorded cache size is the host's;
- that the record names every object the run loaded.

Two limits of the reader: a container with a MoE manifest, auxiliary references, or a representation in any
encoding but BF16 and Q8_0 is refused whole, so it is always `could_not_evaluate`; and the element count is
`rows × cols` of the stored shape.

## 7. Independence

The rules are implemented in `verifier/larql_contract_verifier/contracts/execution_residency.py` from this
document. No engine code is run or imported. The tests hold the Q8 rule to hand-derived blocks and to an
exact-rational implementation with every rounding explicit, on a committed tensor and on seeded blocks across the
BF16 range, and hold all three rules to digests the engine's own loader produced for every matrix of the committed
fixture's decoder stack (`verifier/conformance/execution_residency/engine_digests.json`: engine output, not an
oracle). Those matrices have 64 or 256 columns, so a short last block is held to the exact implementation only, not
to the engine. The fixture was generated by the engine test named in it, which is in the engine's residency-record
change and not in this one.

**A limit on that independence.** §3.1 and §4 were written down by reading the engine's loader, because no
specification of them existed. The document and the engine can therefore share a misreading of what the rule should
be. They cannot share an implementation mistake.
