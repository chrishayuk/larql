# Q8_0 pack derivation: `Q8_0_PACK_DERIVATION`

A VINDEX3 container holds its weights canonically (BF16) and, usually, a **re-derived representation**: a Q8_0 pack
compiled from those weights. The pack is what a Q8_0 execution reads. This contract answers one question about it:

> Is every re-derived Q8_0 representation of this container, byte for byte, the pack its declared derivation rule
> produces from the container's own canonical BF16 weights?

It exists because the other contracts stop short of the pack. `WEIGHT_PATCHES_V4` establishes that an edited
container's **canonical** weights are what its ledger declares, and chains the pack's digests to them, but does not
re-quantise: a pack with arbitrary bytes and recomputed digests passes it. `Q8_0_T2` bounds a pack's reconstruction
error against a pinned upstream checkpoint, which an edited container no longer matches.

## 1. Three claims, kept apart

| Claim | Contract | What it starts from |
|---|---|---|
| The canonical weights are the base with exactly the declared edits | `WEIGHT_PATCHES_V4` (`ledger`) | the base container and the ledger's ops |
| The pack is the declared derivation of the canonical weights | `Q8_0_PACK_DERIVATION` (`pack`) | the container's own canonical bytes |
| A runtime kernel computes with the pack's bytes as the format defines | not established by either | — |

Each is judged on its own and none implies another. A satisfied pack verdict on a container whose canonical weights
are wrong is a faithful pack of wrong weights. Both verdicts together say the pack is the declared derivation of the
declared edit. Neither says anything about execution: byte equivalence of a pack is not a proof that a kernel reads
it, or reads it correctly.

## 2. The derivation rule: `kquant-larql-native`, revision 2

The rule is named by the pack's index entry: `encoder` is exactly `{"algorithm": "kquant-larql-native",
"revision": 2}` and `codec` is exactly the Q8_0 identity (`ggml-block-34`, 32 elements per block, in-block scale). A
pack that names any other rule is not judged.

A tensor is quantised as a row-major run of elements, 32 to a block, rows contiguous with no padding. Its last
dimension is a multiple of 32, so no block spans two rows. For one block of values `x[0..32]`:

1. **Widen.** Each stored BF16 pattern `h` is the binary32 with bits `h << 16`. Exact.
2. **Largest magnitude.** `amax = max |x[i]|`. Exact.
3. **Scale.** `s = amax / 127`, one binary32 division, round to nearest, ties to even.
4. **Stored scale.** `d = s` converted to binary16, round to nearest, ties to even. Binary16 subnormals are kept, and
   a value too large for binary16 is `+inf` (`0x7c00`). `d` is the block's first two bytes, little-endian.
5. **Reciprocal.** `r = 1 / s` if `s > 0`, one binary32 division; `r = 0` if `s = 0`. The reciprocal is of the
   **binary32** scale `s`, not of the stored `d`.
6. **Codes.** For each element, `p = x[i] × r`, one binary32 multiplication; then `p` rounded to the nearest integer
   with **halves away from zero**; then clamped to `[−128, 127]`. The 32 codes follow `d` as signed bytes.

Corners the steps already decide, stated so that no reader has to guess:

- **A zero block** (`amax = 0`, signs ignored): `d = +0`, every code 0.
- **A reciprocal that overflows** (`s ≤ 2^-128`): `r = +inf`, so a nonzero element saturates by its sign
  (`+127`, `−128`), and a zero element gives `0 × inf`, a NaN, whose code is 0. A binary32-subnormal `s` above
  `2^-128` has a finite reciprocal and is quantised by the ordinary steps.
- **A non-finite element** is outside the rule: the contract does not judge such a tensor.

A tensor the pack does not quantise is **carried**: stored in the canonical dtype with the canonical bytes.

Decoding is not part of this contract: a code `q` in a block with stored scale `d` denotes `q × d`
(`refcheck/q8_0.py`), which is not the value `q × s` the encoder had in hand. The rule fixes the bytes, not their
error.

## 3. What is checked

For every representation that has `compiled_from`, and every representation whose `encoding` is `Q8_0` whether or
not it has one:

1. Its `encoding` is `Q8_0`, it names a source in `compiled_from`, its `codec` is the Q8_0 identity and its `encoder`
   is the rule of §2, with an integer revision. Anything else is `could_not_evaluate`. A Q8_0 representation that
   names no source is what a Q8_0 execution would read, so it is not passed over.
2. `compiled_from` names a canonical BF16 representation of the same graph object in this container.
3. Both segments parse under the validating reader and their bytes hash to the index's digests.
4. `source_representation_digest` is the canonical representation's payload digest.
5. The pack's tensor table is the canonical one's, name for name and shape for shape, in order.
6. **Every Q8_0 tensor:** the canonical tensor is finite, the pack is derived from it by §2, and **every block** is
   compared, stored scale and all 32 codes. Nothing is sampled and no digest stands in for a comparison.
7. **Every carried tensor** has the canonical dtype and the canonical bytes.

`status` is `satisfied` only when every representation, tensor and block was derived and matched. One differing
block is a violation, counted per block; a pack that fails item 2, 4 or 5 is a violation too. A violation found is
reported as `violated` even when another part could not be judged, and the answer then lists the unjudged parts
under `reasons`: the violation count is of what was judged, not of the container. With no violation, any part that
could not be judged makes the whole answer `could_not_evaluate`: an unjudged part is never folded into
`satisfied`. A container with no re-derived representation, and a pack for which no Q8_0 block was derived, are
`could_not_evaluate`, not vacuously satisfied.

## 4. What is not established

The answer lists these under `not established`:

- **That the canonical weights are the intended ones.** §1: another contract's claim.
- **That the right tensors were quantised.** Which tensors are Q8_0 is read from the pack's own table. The policy
  that chose them (the precision map's roles) is not re-derived. The answer reports how many matrices the pack
  leaves unquantised so that a pack that quantises nothing of interest is visible, but it does not judge the choice.
- **That a runtime kernel consumes these bytes, or consumes them correctly.**

Two limits of scope that follow from §3:

- **Other encodings.** A re-derived representation in any encoding but Q8_0 (a Q4_K pack beside the Q8_0 one, say)
  makes the whole container `could_not_evaluate`, not only that representation. An honest multi-pack container is
  therefore not judged by this contract as it stands.
- **Non-finite elements are looked for only in quantised tensors.** A carried tensor is compared byte for byte and
  nothing else: a NaN carried identically is not a finding.

## 5. Independence

The expected bytes are derived in `verifier/larql_contract_verifier/contracts/q8_0_pack.py` from this document. The
engine's packer is not run, imported or translated, and nothing the engine wrote is an expected value. The module
holds the rule twice: vectorised binary32 arithmetic, which judges containers, and exact rational arithmetic with
every rounding explicit, which shares no arithmetic with the first. The tests hold the two to each other on
hand-derived blocks, on a sample of one committed tensor's blocks (every seventh: the exact route is slow) and on
seeded blocks across the BF16 range, and hold the binary32-to-binary16 conversion to the library's. Every block of
every committed container is checked by the vectorised route only, against the bytes the engine wrote.

**A limit on that independence.** The rule of §2 was written down by reading the engine's encoder, because no
specification of it existed. So this document and the engine can share a misreading of what the rule *should* be;
what they cannot share is an implementation mistake, since the derivation here is separate code checked against
exact arithmetic. This is the same standing as the rest of `larql_contract_verifier/` ([independence.md](independence.md)).

That the engine and this document agree is evidence, not a premise: the committed containers are engine output, and
the tests require this derivation to reproduce every block of them.
