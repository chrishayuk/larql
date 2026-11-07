# VLP structured-op numerics, v1: the contract and its golden vectors

This is a **language-neutral numerical contract** for the three structured weight-patch ops,
`remove_direction`, `zero_columns` and `add_low_rank`. Any implementation, in any language and on any machine,
must produce **exactly** the BF16 bytes in `vectors.json` for every case, or refuse with exactly the stated code.
An implementation that cannot guarantee these semantics on its platform, for example under a forced
flush-to-zero mode, must refuse to run. It must never produce different bytes.

**Its source.** The contract restates, and does not change:
- WSP-001 stage-2 amendment 1, A1: the profiles (`docs/wsp-001-stage2-amendment-1.md`);
- the G2 design r3, §1–§4: the ops, the refusals and the phases (`docs/wsp-001-g2-design.md`).

Both were frozen on 2026-11-07 on the research branch. Where this document fixes a detail those texts leave
open, it says so under "Choices made here".

**The files:**
- `vectors.json`: the cases;
- `generate.py`: writes them. It uses the standard library only and imports nothing from LARQL;
- `test_generate.py`: the checks.

`python3 -I generate.py --check` regenerates the vectors and compares them byte for byte with the committed file.

## 1. Values and formats

| format | precision p (bits, incl. implicit) | emin | emax | bits |
|---|---|---|---|---|
| binary64 (`f64`) | 53 | −1022 | 1023 | 64 |
| binary32 (`f32`) | 24 | −126 | 127 | 32 |
| bfloat16 (`bf16`) | 8 | −126 | 127 | 16 |

**`round(x, F)`** maps an exact real `x` to format F:
- **Rounding:** to nearest, **ties to even**.
- **Underflow:** gradual. Below `2^emin`, the quantum is `2^(emin − p + 1)`.
- **Overflow:** a result whose rounded magnitude is `≥ 2^(emax+1)` overflows. That includes a tie at the top
  of the range, which rounds up under ties-to-even.
- **Zero results:**
  - an exact zero takes the sign given by the IEEE-754 rules for the operation;
  - a nonzero `x` that rounds to zero keeps the sign of `x`.

**Every arithmetic step is exactly one IEEE-754 operation in the working format:**
- `mul(a, b) = round(a·b)`;
- `add(a, b) = round(a + b)`;
- `sub(a, b) = add(a, −b)`.

**Signed zero follows IEEE-754 RNE:**
- `(−0) + (−0) = −0`;
- `(−0) + (+0) = +0`;
- `x + (−x) = +0`;
- the sign of a zero product is the XOR of the operands' signs.

**Fused multiply-add is forbidden.** Subnormal inputs and results are kept: no FTZ, no DAZ.

**BF16 bits decode** exactly to a value, as IEEE binary formats do: sign, 8-bit exponent with bias 127, 7-bit
mantissa.

## 2. Profiles (amendment A1)

| profile | working format W | conversion chain to BF16 |
|---|---|---|
| `f64-seq-rne1/v1` | f64 | `round(x, bf16)` |
| `f64-seq-rne2/v1` | f64 | `round(round(x, f32), bf16)` |
| `f32-seq-rne1/v1` | f32 | `round(x, bf16)` (report-only for p50c) |

**The rules:**
- **Operands** (`u`, `alpha`, `a`, `b`) are stored as W's bit patterns, as hex strings in the vectors.
- **The base BF16 widens exactly** to W.
- **An overflow inside W arithmetic** is `non_finite_intermediate`.
- **An overflow at any step of the conversion chain** is `overflow`. Under rne2 that includes the f32 step.

## 3. The ops

Every op edits a row-major tensor `W` of shape `[rows, cols]` in BF16 and produces BF16. Elements not written
keep their **exact base bits**.

**`remove_direction`** (`u`: `rows` values, `alpha`, `columns` S):

```
for each j in S, in increasing order:
    w[r]  = widen(W[r, j])                                      r = 0 .. rows-1
    c     = +0
    for r = 0 .. rows-1:  c = add(c, mul(u[r], w[r]))           sequential in r, one accumulator per j
    for r = 0 .. rows-1:  W'[r, j] = chain( sub(w[r], mul(mul(alpha, u[r]), c)) )
```

**`add_low_rank`** (`rank` K ≥ 1; `a` is rows × K and `b` is K × cols, both row-major):

```
for each (r, j):
    w = widen(W[r, j])                                          start from the widened base, not from +0
    for k = 0 .. K-1:  w = add(w, mul(a[r, k], b[k, j]))        sequential in k
    W'[r, j] = chain(w)
```

**`zero_columns`** (`columns` S): `W'[r, j] = 0x0000` for every r and every j ∈ S, whatever the base sign. It
has no arithmetic, no profile and no fixups.

**Fixups** (`[row, col, bits]`; on `remove_direction` and `add_low_rank` only): after computing, each fixup
replaces one element's bits.

## 4. Refusals and the order of checks

An op either produces every output byte or refuses with one code. It never produces a partial result. Each
refusal case in the vectors has exactly one defect. Implementations still check in this order:

| phase | code | when |
|---|---|---|
| parse | `numerics_invalid` | `numerics` missing or not one of the three profile ids (arithmetic ops) |
| parse | `numerics_forbidden` | `zero_columns` carries `numerics` |
| parse | `shape_invalid` | `shape` is not two positive integers |
| parse | `fixups_invalid` | fixups not strictly increasing by (row, col), or a duplicate element |
| parse | `non_finite_operand` | a fixup's bits are NaN or ±inf |
| parse | `fixups_forbidden` | `zero_columns` carries any fixup |
| parse | `columns_invalid` | `columns` empty, or not strictly increasing |
| parse | `rank_invalid` | `rank` < 1 (`add_low_rank`) |
| parse | `shape_invalid` | `len(u) ≠ rows`; `len(a) ≠ rows·K`; `len(b) ≠ K·cols` |
| parse | `non_finite_operand` | any `u`, `alpha`, `a` or `b` value is NaN or ±inf |
| header | `shape_invalid` | the base tensor's shape differs from the declared `shape` |
| header | `tensor_refused` | the tensor is the token embedding (`tensor_role: "embed_tokens"` in the vectors) |
| header | `columns_invalid` | a column ≥ `cols` |
| header | `fixups_invalid` | a fixup outside the tensor, or outside the op's touched set (`remove_direction`: columns S, all rows; `add_low_rank`: every element) |
| tensor bytes | `non_finite_operand` | a base element the op **reads** is NaN or ±inf (arithmetic ops: every element of the touched set). Non-finite elements outside the touched set are kept bit for bit, and `zero_columns` reads no value, so it overwrites them |
| compute | `non_finite_intermediate` | any W-format operation overflows |
| compute | `overflow` | any step of the conversion chain overflows |
| fixups | `fixup_no_op` | a fixup's bits equal the computed bits at that element under the declared profile |

**Within the compute phase** the first failing operation in §3's iteration order decides the code. Each element
is converted as soon as it is computed, so an element's chain overflow comes before a later element's
intermediate overflow.

## 5. The vector file

```
{ "schema": "vlp-numerics-golden/v1",
  "profiles": {...}, "refusals": [...codes...], "variants": {...},
  "cases": [ {
     "id", "purpose",
     "base":   { "shape": [rows, cols], "bf16": ["3f80", ...] (row-major), "sha256" },
     "op":     { "kind", "numerics"?, "shape", "u"? , "alpha"?, "columns"?, "rank"?, "a"?, "b"?, "fixups": [[r, c, "bits"]] },
     "tensor_role"?: "embed_tokens",
     "expect": { "bf16": [...], "sha256" }  |  { "refuse": "<code>" },
     "discriminates": [variant, ...],
     "under_wrong_semantics": { variant: { "bf16": [...] } | { "refuse": code } },
     "construction"?: { "method", "seed", "attempt" } } ] }
```

**Encodings:**
- Bits are lowercase hex with no prefix: 4 digits for BF16, 8 for f32 operands, 16 for f64 operands.
- `sha256` is over the BF16 values as little-endian 2-byte words, in row-major order.

**What the vectors are not.** They test numerics, not the `.vlp` file encoding: base64 operands, the source
binding, the digest and `input_sha256` / `output_sha256` are tested separately.

**`under_wrong_semantics`** records what each named wrong semantics produces, wherever that differs from
`expect`. A mutation test that injects that defect should produce exactly that result. The variants are:

| variant | the wrong semantics, exactly |
|---|---|
| `fma_in_reduction` | `remove_direction`: `c = round(u[r]·w[r] + c)` with the exact product, one rounding per term |
| `reversed_reduction` | `remove_direction`: r from rows−1 down to 0, otherwise as §3 |
| `pairwise_reduction` | `remove_direction`: first `p[r] = mul(u[r], w[r])` for all r; then `pw([x]) = add(+0, x)`, and `pw(xs) = add(pw(xs[:h]), pw(xs[h:]))` with `h = floor(len/2)` |
| `alpha_times_uc` | `remove_direction`: `mul(alpha, mul(u[r], c))` |
| `f32_accumulation` | `remove_direction` only (fault 5): each product is the exact `u[r]·w[r]` rounded once to f32, the accumulator adds in f32 from +0, and `c` is then widened exactly back to f64 |
| `single_rounding` / `double_rounding` | the other f64 chain |
| `ftz_daz` | in W arithmetic, a subnormal input is replaced by a zero of its own sign before the operation, and a result that is subnormal **after rounding** is replaced by a zero of its sign. The same flush applies to the f32 step of the chain, never to the BF16 step |
| `bf16_output_flush` | a BF16 result that is subnormal after rounding is replaced by a zero of its sign |
| `low_rank_from_zero` | `add_low_rank`: `s = +0; s = add(s, mul(a[r,k], b[k,j]))` for k in order, then `w = add(widen(W), s)` |
| `zero_keeps_sign` | `zero_columns` keeps the base sign bit, writing `0x8000` over negative entries (fault 17) |

**`discriminates`** lists the variants a case was **built** to catch. For each, the result is asserted to
differ in at least one byte or in the outcome. The `construction` block records the seeded search for the
cases found by search: alpha is chosen so that one output lands within an ulp of a BF16 tie, or for rne2 half
an f32 ulp above one.

## 6. Coverage (design §2's list)

| what | cases |
|---|---|
| FMA | `rd-fma`, `rd-rne2-fma` |
| reduction order | `rd-order-reverse` (reverse ≠ forward), `rd-order-pairwise` (pairwise ≠ forward and ≠ reverse), `rd-f32-order`. One output element flips between only two values, so a single case cannot separate all three orders; the two cases between them do |
| association | `rd-assoc`, `rd-f32-assoc` |
| f32 accumulation (fault 5 fallback) | `rd-f32-accum` |
| BF16 ties, both parities | `lr-tie-even-down`, `lr-tie-even-up`, `lr-tie-negative`, `lr-bf16-subnormal-tie`, `lr-smallest-normal` |
| double rounding, each chain | `lr-double-rne1`, `lr-double-rne2`, `lr-double-f32`, `lr-double-negative-rne2` |
| the low-rank start | `lr-from-w` |
| signed zero | `lr-negzero-plus-poszero`, `lr-negzero-plus-negzero`, `rd-x-minus-x`, `lr-tiny-negative`, `lr-tiny-negative-rne2`, `zc-negatives` |
| subnormals: f64, f32 step, BF16 | `rd-f64-subnormal-c`, `lr-f32-step-subnormal`, `lr-f32-profile-subnormal`, `lr-bf16-subnormal`, `lr-bf16-subnormal-tie` |
| subnormal base (exact widening, DAZ on read) | `lr-subnormal-base-rne1`, `lr-subnormal-base-f32` |
| extremes | `lr-max-finite`; overflow at each chain's final result: `ref-overflow-rne1`, `ref-overflow-rne2`, `ref-overflow-f32-step`, `ref-overflow-f32-profile`, `ref-rd-overflow`; intermediates: `ref-intermediate-*`, `ref-rd-intermediate` |
| non-finite operands and base | `ref-nan-u`, `ref-nan-u-f32`, `ref-inf-alpha`, `ref-nan-a`, `ref-inf-b`, `ref-*-fixup`, `ref-nan-base-touched`, `ref-inf-base-low-rank`; kept: `rd-inf-base-untouched`, `zc-nan-base` |
| fixups | `rd-fixup`, `lr-fixup-rne1`, `ref-fixup-no-op-rne2`, `ref-fixup-no-op-rd`, `ref-fixup-*` |
| malformed and refused | every `ref-*` case; every code in §4 appears at least once |
| general | `rd-general-*`, `lr-general-*`: 16 × 6, every profile |

**Implementation rules that vectors cannot test** are in the frozen texts and still bind every implementation:
- the FTZ/DAZ runtime canaries in f64 and f32 on runtime values (amendment A1; design §2);
- no BLAS, `dot`, `einsum`, iterator `sum()`, `mul_add` or Metal fast-math for this arithmetic, and SIMD only
  across `j`, never across `r` (design §2);
- for a numpy verifier: numpy ≥ 2 and every operand and intermediate asserted at the working dtype.

**Not covered here:**
- resource limits (fault 18 is synthetic at G3);
- the at-most-one-op-per-tensor rule (a patch-level check);
- the runtime rules listed just above, which each implementation enforces on its own platform.

## 7. How the expected values are made, and their independence

- `generate.py` computes every rounding exactly, with rational arithmetic (`fractions.Fraction`). It encodes
  bits itself.
- **A second route** recomputes every f64-profile arithmetic case. It uses CPython binary64 floats (separate
  `*` and `+`), BF16 rounding by `frexp` and `round()` (half to even), and `struct` for the f32 step. The two
  routes must agree.
- **The tests** also check:
  - the exact rounder against those native routes on 20,000 random and adversarial values;
  - the f32 profile's cases against numpy float32 (numpy ≥ 2).
- **The limit of this independence:** these routes were written by one author. The contract's independent
  check is a reviewer's own implementation of §1–§4, written from this document alone, that reproduces
  `vectors.json`.

## Choices made here (not fixed by the frozen texts; for review)

1. **The refusal codes** and their check order (§4). The frozen texts say what is refused, not how it is named.
2. **`fixup_no_op` for both arithmetic ops.** The design names a no-op fixup as structural fault 11 (on p10c).
   This applies the same rule to `remove_direction`.
3. **`zero_columns` refuses a `numerics` field.** The design says its `round_chain` is absent; this makes its
   presence a refusal rather than something ignored.
4. **The vectors' tensor role.** The `embed_tokens` refusal is carried as `tensor_role`, because the vectors
   have no container.
5. **A non-finite base element** is refused (`non_finite_operand`) only when the op reads it. Elements outside
   the touched set keep their bits ("bytes untouched", design §1), and `zero_columns` overwrites without reading.
6. **The compute-phase order** (§4): the first failing operation in §3's iteration order decides between
   `non_finite_intermediate` and `overflow`.
