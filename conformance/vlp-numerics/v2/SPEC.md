# VLP structured-op numerics, v2: `add_low_rank_columns`

**Status: DRAFT 2026-11-12. Not frozen.** It becomes final only after an independent evaluator, written from this text
alone, reproduces every case in `vectors.json`. Then `FREEZE.json` is written, as for v1.

**Relation to v1.** v1 (`../v1/`, frozen) is **unchanged and is not edited by this version**. v2 adds one op and reuses
v1's values, formats, profiles, arithmetic rules and refusal vocabulary and phases (v1 §1, §2, §4), with only the
differences this document states (the column footprint, the compact `b`, the identity rule of §4, and the wording of a few
refusals in §5). `generate.py` loads
`../v1/generate.py` read-only for its exact-rational arithmetic, so there is one rounder in the repository.

## 1. Why a new op

v1's `add_low_rank` has no footprint: it reads and writes **every** element. An edit confined to a head's slice of `o_proj`,
or to any subset of columns, cannot be expressed. Padding `b` with zeros is not a substitute, for two reasons that the
vectors pin (variant `zero_padded_full`):

- it reads the whole tensor, so a NaN or inf outside the intended columns is refused instead of kept;
- it recomputes every element, and `(−0) + (+0) = +0`, so any `−0` base element anywhere changes its bits.

## 2. Axes and operand layout (stated, not implied)

The tensor is the stored one: row-major `[rows, cols]` exactly as in the checkpoint (for HF/safetensors linear weights,
`[out_features, in_features]`). So:

- **rows** index the **output** axis; **columns** index the **input** axis.
- `columns` = `J`, a strictly increasing list of column indices into the **input** axis. `|J| = n`.
- `a` is `rows × K`, row-major: `a[r, k]` is the output-axis factor of term `k` for row `r`.
- `b` is `K × n`, row-major, **compact**: `b[k, p]` is the input-axis factor of term `k` for column `J[p]`. A `b` laid out
  `K × cols` (v1's layout) is a `shape_invalid` refusal when `n ≠ cols`.

The update is `Σ_k a[r,k]·b[k,p]` added to `W[r, J[p]]`. There is no scalar `alpha`: a producer that has `α·u·vᵀ` folds
`α` into its operands, and records `α`, `u`, `v` as provenance outside the op.

K ≥ 1. K = 1 is the case that matters for a single rank-one edit; K > 1 is supported and covered.

## 2a. The arithmetic contract (restated from v1 §1-§2 so nothing is left to inference)

- **Profiles.** `numerics` is one of `f64-seq-rne1/v1`, `f64-seq-rne2/v1`, `f32-seq-rne1/v1` (v1 §2). Anything else is
  `numerics_invalid`. The **working format W** is f64 for the first two and f32 for the third.
- **Operands** `a`, `b` are W bit patterns (hex: 16 digits for f64, 8 for f32). A NaN or infinity is `non_finite_operand`.
  The base BF16 widens to W exactly.
- **Every arithmetic step is one IEEE-754 operation in W**, round to nearest, ties to even, gradual underflow, no fused
  multiply-add, no flush-to-zero or denormals-are-zero: for each `k`, first the product `mul(a[r,k], b[k,p])`
  is rounded once in W, then the addition `add(w, product)` is rounded once in W. An operation whose result overflows W is `non_finite_intermediate`.
- **Accumulation order** is fixed: the running value starts at `widen(W[r,j])` (never `+0`) and the terms are added in
  increasing `k`, one rounding per addition. No reordering, pairwise or blocked summation, and no BLAS.
- **Well-formed documents.** The op and the base are taken to be well-formed: operands, fixups and base words use the
  widths of v1 §5 (BF16 4 hex digits, f32 8, f64 16, lowercase hex); the integer fields (`shape`, `rank`, `columns`, fixup
  coordinates) are JSON integers (not floats or booleans); and every field is present with its stated type (a missing
  `fixups` is an empty list). A violation of any of these is **outside this contract**: no behaviour is defined for it and the
  vectors contain none; the parser that reads the patch refuses it before this op runs. A fixup's bits are a BF16 word, and NaN or infinity is judged on its BF16 exponent field.
- **Signed zero** follows IEEE-754 RNE: `(−0)+(−0) = −0`, `(−0)+(+0) = +0`, `x+(−x) = +0`, and a product of zero factors has
  the XOR of their signs.
- **The conversion chain** to BF16 is applied once, to the final `w`: `rne1`: W to BF16; `rne2`: W to f32, then (in that order) f32 to BF16;
  `f32-seq-rne1`: f32 to BF16. Each conversion rounds to nearest, ties to even, with gradual underflow; a nonzero value that
  rounds to zero keeps its sign. A conversion whose rounded magnitude reaches `2^(emax+1)` is `overflow`, including a tie at
  the top of the range and the f32 step of `rne2`. Converting a zero keeps its sign.
- **Visiting order.** Rows outermost, then `p` (the columns of `J` in order), then `k`. An element is converted to BF16 as soon
  as its accumulation finishes, and the first failing operation in this order decides between `non_finite_intermediate` and
  `overflow`.
- **"Zero"** means a value equal to zero of either sign; `−0` and `+0` are both zero wherever this document says nonzero or
  zero.

## 3. The op

Operands `a`, `b` are stored as the working format's bit patterns, as in v1. For each row `r` and each `p`, with
`j = J[p]` and the terms `(a[r,k], b[k,p])`, `k = 0 … K−1`:

```
written(r, j)  =  some term has both factors nonzero          (decided on the operand values, not on rounded products,
                                                               and BEFORE any arithmetic: a selection, not a correction)

if not written(r, j):  W'[r, j] = W[r, j]  bit for bit        (no arithmetic; a base -0 stays -0)
else:
    w = widen(W[r, j])                                        start from the widened base, not from +0
    for k = 0 .. K-1:  w = add(w, mul(a[r,k], b[k,p]))        sequential in k, one operation each
    W'[r, j] = chain(w)
```

**In particular, for a rank-1 edit** (K = 1): a row `r` with `a[r] = 0` has every one of its elements inside `J` left
untouched, and a position `p` with `b[p] = 0` has its whole column `J[p]` left untouched, a base `−0` included. Only the
elements where both factors are nonzero are written (`k1-zero-row-of-a-untouched`, `k1-zero-entry-of-b-untouched`).
For K > 1 an element is untouched only if every one of its K terms has a zero factor.

**Selection first.** Whether an element is written is decided from the operands alone. An untouched element is never run
through the arithmetic: nothing computes `W + 0` and then repairs the sign afterwards.

Elements with `j ∉ J` keep their exact base bits and are never read. **Fixups** are as in v1, with the touched set
`{(r, j) : j ∈ J}`; a fixup in any other column is `fixups_invalid`. `fixup_no_op` compares a fixup with the computed bits at its element, which for
an untouched element are its base bits.

## 4. The identity rule, and where it departs from v1

**The rule:** an element whose terms are all zero keeps its base bits. It is what makes `a = 0` (or `b = 0`) a genuine
no-op: **the compiled tensor is byte-identical to the base**, including `−0` elements inside `J` (case
`sham-whole-tensor-unchanged`), **provided the precondition of §5 holds** (no NaN or infinity inside rows × J).

**The conflict with v1, stated.** With `J` = every column, v2 and v1 agree on every element **except** one class: an
element whose terms are all zero and whose base is `−0`. v1 computes `(−0) + (+0) = +0` and writes `+0`; v2 keeps `−0`.
Where all terms are zero and the base is anything else (nonzero finite, `+0`), `w + (±0)` returns `w` unchanged, so the two
agree. v1 is not edited and remains correct for its own op; this is a different op with a stronger identity law.

**Equivalence scope, enumerated.** `vectors.json` re-expresses every v1 `add_low_rank` case (38 cases, `eq-*`, with
`equivalent_to_v1`) with `columns` = every column. Each records `same_as_v1`. 37 are `true`, byte-for-byte or refusal-for-
refusal. Exactly one is `false`: `eq-lr-negzero-plus-poszero` (`differs_from_v1_because`: all-zero terms keep a base `−0`).
The generator asserts that any difference from v1 is only a kept `−0`.

**Written is decided on operands.** `nonzero-operands-underflowing-product-f32`: two nonzero f32 operands whose product
underflows to `+0` still make the element *written*, so a base `−0` becomes `+0`. The identity rule applies to
all-zero **operands**, not to a computed zero. Likewise `exact-cancellation-from-negzero` (`+x` and `−x`) is written and gives
`+0`, and `nonzero-update-underflows-positive` / `-negative` show a nonzero update that rounds to zero from a base `−0`
(`+0` and `−0` respectively, by the sign of the value).

## 5. Refusals and order

The vocabulary and phases are v1 §4. The order for this op: `numerics_invalid`; `shape_invalid` (declared shape);
`fixups_invalid` (order, duplicates); `non_finite_operand` (a fixup's bits); `columns_invalid` (empty, not strictly
increasing, or an entry that is not a non-negative integer); `rank_invalid`; `shape_invalid` (`len(a) ≠ rows·K` or `len(b) ≠ K·n`); `non_finite_operand` (an operand);
then the header phase: `shape_invalid` (the tensor's shape differs from the declared one, or the base holds a word count other than `rows·cols`), `tensor_refused` (`tensor_role` is `embed_tokens`; any other value is ignored), `columns_invalid`
(a column ≥ `cols`), `fixups_invalid` (a fixup outside the tensor or outside `J`); then `non_finite_operand` for a
non-finite base element **inside rows × J**; then compute (`non_finite_intermediate`, `overflow`); then `fixup_no_op`.
Non-finite base elements outside `J` are kept, not read, and do not refuse (`out-keeps-special-values`,
`out-nan-base-outside-j-is-kept`). The codes that belong only to `zero_columns` (`numerics_forbidden`, `fixups_forbidden`) do not arise.

The non-finite-base check covers rows × J whether or not an element is written, so a NaN inside `J` refuses even under a
zero update (`ref-nan-base-inside-j`). **This is a declared precondition.** A zero-valued op (one in which every element of rows × J has only zero-factor terms, such as `a = 0` or `b = 0`) is an identity (byte-identical
to the base) only over tensors with no NaN or infinity inside rows × J, not over arbitrary contents.

## 6. Wrong semantics the vectors discriminate

| variant | the wrong semantics |
|---|---|
| `zero_padded_full` | v1 `add_low_rank` on `b` padded with zeros outside `J` |
| `zero_update_not_kept` | all-zero-term elements are computed, so `(−0) + (+0) = +0` |
| `low_rank_from_zero` | `s = +0` accumulated over `k`, then `w = add(widen(W), s)` (`k-order-start-from-base`) |

Each case lists the variants it was built to catch (`discriminates`) and every variant that changes its result
(`under_wrong_semantics`). v1's variants that apply to `add_low_rank` (FTZ/DAZ, chain, `bf16_output_flush`) are inherited
through the equivalence cases.

## 7. What the vectors cover

BF16 ties to even in both directions and either side of a tie (`tie-*`, `just-above-tie`, `just-below-tie`); double rounding
under each chain (`tie-double-rounding-*`); near-cancellation to a tiny nonzero (`near-cancellation*`) and exact cancellation;
signed zero inside `J` as above; special values and non-finite values outside `J`; K = 1, 2, 3; single column, first, last,
all-but-one, one row; the three profiles; fixups that apply and each fixup refusal; every refusal in §5.
A native CPython-float route recomputes the f64-profile cases (58) and agrees with the exact route.

## 8. What the vectors do not test (the verifier and engine contracts)

- **Footprint and conflicts.** The footprint is `J` over all rows. Two ops on one tensor **within one compilation stage
  (one patch)** are compatible iff their column sets are disjoint (the engine's structured-structured rule); v1's
  at-most-one-op-per-tensor rule is a patch-level check. **This is not a limit on learning.** A later patch is a new stage,
  bound to the compiled output of the one before it, so it may edit the same columns again: its operands are applied to
  that output's weights, and it is verified against that output. Overlap is refused only inside a single stage, where
  the result would depend on an unstated order. Whether the first improvement survives the second patch is a behavioural
  property that neither this contract, nor source binding, nor numerical verification guarantees.
- **Fault injection.** A verifier for this op must be shown to reject each of: a flipped bit in a written element **inside**
  the footprint; a changed element **outside** `J` (including a `−0`→`+0` flip); a changed element in another tensor; a
  stale source binding; a swapped or reordered `a`/`b`; a `b` in the wrong layout. These are verifier tests, as for
  `STRUCTURED_REMOVE_DIRECTION`'s golden (`verifier/tests/test_structured_patch_golden.py`), and are not numerics vectors.
- Operand encoding, source binding, the op digest, and resource limits, as v1.

## 9. Open decisions for review

1. The name `add_low_rank_columns` (SG-EDIT-1's "ADD OUTER" is dropped).
2. Operand-level `written` versus an exact-rational "update is zero" rule. The operand rule was chosen because every
   implementation can apply it without arithmetic; its cost is that exactly cancelling nonzero terms count as written.
3. Whether the non-finite-base check should cover only written elements. It covers rows × J here, the stricter reading.
