"""The frozen VLP structured-op numerics, implemented here from their specifications and nothing else:

* `conformance/vlp-numerics/v1/SPEC.md`: values, formats, profiles, rounding, refusals; `remove_direction` and v1's
  `add_low_rank`;
* `conformance/vlp-numerics/v2/SPEC.md`: `add_low_rank_columns` (the column footprint, the compact `b`, the operand-level
  `written` selection and its identity rule, and the refusal order).

This is the verifier's own route. It does not use, and was not written from, the engine or the vectors' generator.

**How the arithmetic is exact.** Every arithmetic step of a profile is one IEEE-754 operation in the working format W
(binary64 or binary32), round to nearest, ties to even, gradual underflow, no FMA. NumPy's elementwise `multiply`, `add`
and `subtract` on float64 (float32) arrays are exactly those single IEEE operations, so the profiles run vectorised. The
vectorisation is only ever ACROSS independent elements (across columns `j`, or across rows `r` for `add_low_rank_columns`
where every element has its own accumulator); a reduction (the `remove_direction` dot product, the sum over `k`) is
always a Python loop in the declared order, one ufunc call per step. Nothing calls BLAS, `dot`, `einsum`, `sum` or
`matmul` on values. The dtype of every operand and intermediate is asserted to be W.

**The conversion to BF16 (and the f32 step of rne2) is written here, on bit patterns**: no NumPy cast rounds anything.
A runtime canary refuses to run if the platform flushes subnormals in either working format (v1 §6).
"""

from __future__ import annotations

import numpy as np

PROFILE_RNE1 = "f64-seq-rne1/v1"
PROFILE_RNE2 = "f64-seq-rne2/v1"
PROFILE_F32 = "f32-seq-rne1/v1"
#: profile -> (working dtype, hex digits of an operand, whether the chain goes through f32 first)
PROFILES = {
    PROFILE_RNE1: (np.float64, 16, False),
    PROFILE_RNE2: (np.float64, 16, True),
    PROFILE_F32: (np.float32, 8, False),
}

# Refusal codes, v1 §4 and v2 §5.
NUMERICS_INVALID = "numerics_invalid"
SHAPE_INVALID = "shape_invalid"
FIXUPS_INVALID = "fixups_invalid"
NON_FINITE_OPERAND = "non_finite_operand"
COLUMNS_INVALID = "columns_invalid"
RANK_INVALID = "rank_invalid"
TENSOR_REFUSED = "tensor_refused"
NON_FINITE_INTERMEDIATE = "non_finite_intermediate"
OVERFLOW = "overflow"
FIXUP_NO_OP = "fixup_no_op"
#: Not a v1/v2 code: a document that is not well-formed (v2 §2a puts this outside the numerics contract; the parser
#: that reads the patch or the ledger refuses it before the op runs).
MALFORMED = "malformed"
EMBED_TOKENS_ROLE = "embed_tokens"

BF16_HEX_DIGITS = 4
BF16_EXP_MASK = 0x7F80
BF16_SIGN = 0x8000
F32_EXP_MASK = 0x7F800000
HEX = frozenset("0123456789abcdef")

# (precision p incl. the implicit bit, emin, exponent field width, total bits) of the target formats, v1 §1
BF16_FMT = (8, -126, 8, 16)
F32_FMT = (24, -126, 8, 32)
F64_BIAS = 1023
F64_MANTISSA_BITS = 52
F64_EXP_FIELD = 0x7FF
MAX_SHIFT = 63
#: rows per block: bounds peak memory on a large tensor without changing a single result (rows are independent)
ROW_BLOCK = 256


class Refusal(Exception):
    """The op is refused with one of the codes above. `detail` says where."""

    def __init__(self, code: str, detail: str = ""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


# ── rounding, on bit patterns ───────────────────────────────────────────────


def _round_from_f64(x: np.ndarray, fmt) -> tuple[np.ndarray, np.ndarray]:
    """Round every finite binary64 value of `x` to the target format by RNE with gradual underflow (v1 §1).

    Returns `(bits, overflow)`: the target's bit pattern (sign included; a nonzero value that rounds to zero keeps its
    sign, and a zero keeps its sign), and a mask of the values whose rounded magnitude reaches `2^(emax+1)`, including
    a tie at the top of the range. Where `overflow` is set, `bits` is meaningless.

    A finite binary64 is `M * 2^E` with an integer significand `M < 2^53`. The target's quantum at that magnitude is
    `2^q` with `q = max(floor(log2|x|), emin) - (p - 1)`, so the rounded value is `N * 2^q` with `N` = `M / 2^(q - E)`
    rounded half to even, done in integers. The target's bit pattern is then `((q - qmin) << (p - 1)) + N`, which
    carries into the exponent field by itself when `N` rounds up to `2^p`, and is a subnormal (or zero) when
    `q = qmin`."""
    p, emin, ebits, total = fmt
    x = np.ascontiguousarray(x, dtype=np.float64)
    assert x.dtype == np.float64
    raw = x.view(np.uint64)
    sign = raw >> np.uint64(63)
    efield = ((raw >> np.uint64(F64_MANTISSA_BITS)) & np.uint64(F64_EXP_FIELD)).astype(np.int64)
    if np.any(efield == F64_EXP_FIELD):
        raise AssertionError("a non-finite value reached the rounder")
    mant = raw & np.uint64((1 << F64_MANTISSA_BITS) - 1)
    normal = efield > 0
    # A binary64 subnormal is below 2^-1022, far under half the smallest subnormal of either target: it rounds to a
    # zero of its own sign. So only normal inputs need the integer rounding.
    sig = np.where(normal, mant | np.uint64(1 << F64_MANTISSA_BITS), np.uint64(0))
    flog = efield - F64_BIAS  # floor(log2 |x|) of a normal value
    q = np.maximum(flog, emin) - (p - 1)
    e_unit = efield - (F64_BIAS + F64_MANTISSA_BITS)  # exponent of M's unit bit
    shift = np.minimum(q - e_unit, MAX_SHIFT)
    assert np.all(shift[normal] > 0)  # every target here is narrower than binary64
    shift = np.where(normal, shift, 1).astype(np.uint64)
    kept = sig >> shift
    rem = sig & ((np.uint64(1) << shift) - np.uint64(1))
    half = np.uint64(1) << (shift - np.uint64(1))
    up = (rem > half) | ((rem == half) & ((kept & np.uint64(1)) == np.uint64(1)))
    n = kept + up.astype(np.uint64)
    qmin = emin - (p - 1)
    code = np.where(n == 0, np.uint64(0),
                    (np.maximum(q - qmin, 0).astype(np.uint64) << np.uint64(p - 1)) + n)
    top = np.uint64(((1 << ebits) - 1) << (p - 1))
    overflow = normal & (code >= top)
    code = np.where(overflow, np.uint64(0), code)
    bits = (sign << np.uint64(total - 1)) | code
    return bits, overflow


def f64_to_bf16(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`round(x, bf16)` from binary64: BF16 bits (uint16) and the overflow mask."""
    bits, overflow = _round_from_f64(x, BF16_FMT)
    return bits.astype(np.uint16), overflow


def f64_to_f32_bits(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`round(x, f32)` from binary64: binary32 bits (uint32) and the overflow mask."""
    bits, overflow = _round_from_f64(x, F32_FMT)
    return bits.astype(np.uint32), overflow


def f32_bits_to_f64(bits: np.ndarray) -> np.ndarray:
    """Exact widening of binary32 bits to binary64 (every binary32 is a binary64; subnormals included)."""
    return np.ascontiguousarray(bits, dtype=np.uint32).view(np.float32).astype(np.float64)


def bf16_to(bits: np.ndarray, dtype) -> np.ndarray:
    """Exact widening of BF16 bits to the working format: BF16 is the high half of a binary32."""
    wide = (np.asarray(bits, dtype=np.uint32) << np.uint32(16)).view(np.float32)
    return wide.astype(dtype)


def chain(w: np.ndarray, profile: str) -> tuple[np.ndarray, np.ndarray]:
    """The profile's conversion chain to BF16, applied to final values `w` in W (v1 §2, v2 §2a)."""
    if profile == PROFILE_F32:
        assert w.dtype == np.float32
        return f64_to_bf16(w.astype(np.float64))  # exact widening, then one rounding
    assert w.dtype == np.float64
    if profile == PROFILE_RNE1:
        return f64_to_bf16(w)
    via, over32 = f64_to_f32_bits(w)
    bits, over16 = f64_to_bf16(f32_bits_to_f64(via))
    return bits, over32 | over16


def bf16_is_finite(bits: np.ndarray) -> np.ndarray:
    return (np.asarray(bits, dtype=np.uint16) & np.uint16(BF16_EXP_MASK)) != BF16_EXP_MASK


# ── the platform canary ──────────────────────────────────────────────────────


def check_platform() -> None:
    """Refuse to run if subnormals are flushed (FTZ) or treated as zero (DAZ) in either working format, through the
    same vectorised ufuncs the profiles use, or if NumPy is older than 2 (v1 §6)."""
    if int(np.__version__.split(".")[0]) < 2:
        raise RuntimeError("numpy >= 2 is required")
    for dt in (np.float64, np.float32):
        tiny = np.full(64, np.finfo(dt).smallest_subnormal, dtype=dt)
        one = np.ones(64, dtype=dt)
        half = np.full(64, 0.5, dtype=dt)
        ok = (np.all(tiny * one == tiny) and np.all(tiny + tiny == tiny * np.full(64, 2, dtype=dt))
              and np.all((tiny * np.full(64, 3, dtype=dt)) * half != 0)
              and np.all(np.finfo(dt).smallest_normal * half != 0) and np.all(tiny != 0))
        if not ok:
            raise RuntimeError(f"the platform flushes {np.dtype(dt).name} subnormals; the profiles cannot be run")


# ── parsing (v1 §4 and v2 §5 parse phase) ─────────────────────────────────────


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def parse_hex(values, digits: int, what: str) -> np.ndarray:
    """A list of lowercase hex bit patterns of exactly `digits` digits, as unsigned integers."""
    if not isinstance(values, list):
        raise Refusal(MALFORMED, f"{what} is not a list")
    for h in values:
        if not isinstance(h, str) or len(h) != digits or not HEX.issuperset(h):
            raise Refusal(MALFORMED, f"{what} holds {h!r}, not {digits} lowercase hex digits")
    dt = np.uint64 if digits == 16 else np.uint32
    if not values:
        return np.zeros(0, dtype=dt)
    return np.array([int(h, 16) for h in values], dtype=dt)


def operand_values(bits: np.ndarray, dtype) -> np.ndarray:
    return bits.view(np.float64 if dtype == np.float64 else np.float32)


def operand_digest_bytes(bits: np.ndarray) -> bytes:
    """Little-endian bytes of the operand's bit patterns at the working width, in stored order (v4 §3)."""
    return bits.astype(bits.dtype.newbyteorder("<")).tobytes()


def _profile(op: dict) -> tuple:
    numerics = op.get("numerics")
    if numerics not in PROFILES:
        raise Refusal(NUMERICS_INVALID, f"{numerics!r} is not one of {sorted(PROFILES)}")
    return PROFILES[numerics]


def _shape(shape) -> tuple[int, int]:
    if not isinstance(shape, (list, tuple)) or len(shape) != 2 or not all(_is_int(d) and d > 0 for d in shape):
        raise Refusal(SHAPE_INVALID, f"declared shape {shape!r} is not two positive integers")
    return int(shape[0]), int(shape[1])


def _fixups(raw) -> list[tuple[int, int, int]]:
    raw = [] if raw is None else raw
    if not isinstance(raw, list):
        raise Refusal(MALFORMED, "fixups is not a list")
    out = []
    for f in raw:
        if (not isinstance(f, list) or len(f) != 3 or not _is_int(f[0]) or not _is_int(f[1])
                or not isinstance(f[2], str) or len(f[2]) != BF16_HEX_DIGITS or not HEX.issuperset(f[2])):
            raise Refusal(MALFORMED, f"fixup {f!r} is not [row, col, 4 hex digits]")
        out.append((f[0], f[1], int(f[2], 16)))
    keys = [(r, c) for r, c, _ in out]
    if any(b <= a for a, b in zip(keys, keys[1:])):
        raise Refusal(FIXUPS_INVALID, "fixups are not strictly increasing by (row, col), or repeat an element")
    for r, c, b in out:
        if b & BF16_EXP_MASK == BF16_EXP_MASK:
            raise Refusal(NON_FINITE_OPERAND, f"the fixup at ({r}, {c}) is NaN or infinite")
    return out


def _columns(columns, strict_entries: bool) -> list[int]:
    if not isinstance(columns, list) or not columns:
        raise Refusal(COLUMNS_INVALID, "columns are empty")
    if any(not _is_int(c) for c in columns):
        if strict_entries:
            raise Refusal(COLUMNS_INVALID, "a column is not an integer")
        raise Refusal(MALFORMED, "a column is not an integer")
    if strict_entries and any(c < 0 for c in columns):
        raise Refusal(COLUMNS_INVALID, "a column is negative")
    if any(b <= a for a, b in zip(columns, columns[1:])):
        raise Refusal(COLUMNS_INVALID, "columns are not strictly increasing")
    if not strict_entries and columns[0] < 0:
        raise Refusal(COLUMNS_INVALID, "a column is negative")
    return list(columns)


def _nonfinite_operand(bits: np.ndarray, digits: int) -> bool:
    if digits == 16:
        return bool(np.any((bits & np.uint64(0x7FF0000000000000)) == np.uint64(0x7FF0000000000000)))
    return bool(np.any((bits & np.uint32(F32_EXP_MASK)) == np.uint32(F32_EXP_MASK)))


def _header(base: np.ndarray, rows: int, cols: int, tensor_shape, tensor_role) -> np.ndarray:
    if tensor_shape is not None and tuple(tensor_shape) != (rows, cols):
        raise Refusal(SHAPE_INVALID, f"the tensor is {tuple(tensor_shape)}, the op declares {(rows, cols)}")
    base = np.asarray(base, dtype=np.uint16).reshape(-1)
    if base.size != rows * cols:
        raise Refusal(SHAPE_INVALID, f"the base holds {base.size} words, not {rows}x{cols}")
    if tensor_role == EMBED_TOKENS_ROLE:
        raise Refusal(TENSOR_REFUSED, "the op edits the token embedding")
    return base.reshape(rows, cols)


def _fixups_in(fixups, rows, cols, touched_cols) -> None:
    for r, c, _ in fixups:
        if not (0 <= r < rows and 0 <= c < cols):
            raise Refusal(FIXUPS_INVALID, f"the fixup at ({r}, {c}) is outside the tensor")
        if touched_cols is not None and c not in touched_cols:
            raise Refusal(FIXUPS_INVALID, f"the fixup at ({r}, {c}) is outside the op's columns")


def _apply_fixups(out: np.ndarray, fixups) -> None:
    for r, c, b in fixups:
        if int(out[r, c]) == b:
            raise Refusal(FIXUP_NO_OP, f"the fixup at ({r}, {c}) equals the computed bits")
    for r, c, b in fixups:
        out[r, c] = b


# ── add_low_rank_columns (v2) and v1's add_low_rank ───────────────────────────


class LowRank:
    """A parsed `add_low_rank_columns` op (or v1's `add_low_rank`, with `columns` = every column)."""

    def __init__(self, op: dict, shape, *, v1_full: bool = False):
        self.v1_full = v1_full
        self.numerics = op.get("numerics")
        self.dtype, digits, _ = _profile(op)
        self.profile = self.numerics
        self.rows, self.cols = _shape(shape)
        self.fixups = _fixups(op.get("fixups"))
        if v1_full:
            self.columns = list(range(self.cols))
        else:
            self.columns = _columns(op.get("columns"), strict_entries=True)
        rank = op.get("rank")
        if not _is_int(rank):
            raise Refusal(MALFORMED, "rank is not an integer")
        if rank < 1:
            raise Refusal(RANK_INVALID, f"rank {rank} < 1")
        self.rank = rank
        a, b = op.get("a"), op.get("b")
        if not isinstance(a, list) or not isinstance(b, list):
            raise Refusal(MALFORMED, "a or b is not a list")
        width = self.cols if v1_full else len(self.columns)
        if len(a) != self.rows * rank or len(b) != rank * width:
            raise Refusal(SHAPE_INVALID, f"len(a) = {len(a)} (want {self.rows * rank}), len(b) = {len(b)} "
                                         f"(want {rank * width})")
        self.a_bits = parse_hex(a, digits, "a")
        self.b_bits = parse_hex(b, digits, "b")
        if _nonfinite_operand(self.a_bits, digits) or _nonfinite_operand(self.b_bits, digits):
            raise Refusal(NON_FINITE_OPERAND, "an operand of a or b is NaN or infinite")
        self.a = operand_values(self.a_bits, self.dtype).reshape(self.rows, rank)
        self.b = operand_values(self.b_bits, self.dtype).reshape(rank, width)

    def zero_valued(self) -> bool:
        """v4 §4a: for every k, column k of `a` or row k of `b` is all zero (either sign), from the operands alone."""
        return all(not np.any(self.a[:, k] != 0) or not np.any(self.b[k] != 0) for k in range(self.rank))

    def written(self, r0: int, r1: int) -> np.ndarray:
        """The operand-level selection for rows r0..r1 (v2 §3): some term has both factors nonzero. No arithmetic."""
        if self.v1_full:
            return np.ones((r1 - r0, len(self.columns)), dtype=bool)
        sel = np.zeros((r1 - r0, len(self.columns)), dtype=bool)
        for k in range(self.rank):
            sel |= (self.a[r0:r1, k] != 0)[:, None] & (self.b[k] != 0)[None, :]
        return sel

    def apply(self, base: np.ndarray, tensor_shape=None, tensor_role=None) -> np.ndarray:
        """The BF16 bits of the whole tensor after this op, or `Refusal` (v2 §3, §5)."""
        rows, cols = self.rows, self.cols
        grid = _header(base, rows, cols, tensor_shape, tensor_role)
        if self.columns[-1] >= cols:
            raise Refusal(COLUMNS_INVALID, f"column {self.columns[-1]} >= {cols}")
        _fixups_in(self.fixups, rows, cols, None if self.v1_full else set(self.columns))
        cols_ix = np.asarray(self.columns, dtype=np.int64)
        sub = grid[:, cols_ix]
        if not np.all(bf16_is_finite(sub)):
            r, p = np.argwhere(~bf16_is_finite(sub))[0]
            raise Refusal(NON_FINITE_OPERAND, f"the base element at ({r}, {self.columns[p]}) is NaN or infinite")
        out = grid.copy()
        new_sub = sub.copy()
        dt = self.dtype
        with np.errstate(all="ignore"):
            for r0 in range(0, rows, ROW_BLOCK):
                r1 = min(rows, r0 + ROW_BLOCK)
                sel = self.written(r0, r1)
                if not sel.any():
                    continue
                w = bf16_to(sub[r0:r1], dt)  # start from the widened base, never from +0
                bad = np.zeros(w.shape, dtype=bool)
                for k in range(self.rank):  # sequential in k, one rounding per product and per addition
                    prod = self.a[r0:r1, k][:, None] * self.b[k][None, :]
                    assert prod.dtype == dt and w.dtype == dt
                    bad |= ~np.isfinite(prod)
                    w = w + prod
                    assert w.dtype == dt
                    bad |= ~np.isfinite(w)
                bad &= sel
                safe = np.where(bad | ~sel, dt(0), w)
                bits, over = chain(safe, self.profile)
                over &= sel & ~bad
                failed = bad | over
                if failed.any():  # rows outermost, then p: the first failing element decides the code
                    r, p = np.argwhere(failed)[0]
                    where = f"({r0 + r}, {self.columns[p]})"
                    if bad[r, p]:
                        raise Refusal(NON_FINITE_INTERMEDIATE, f"an operation overflows W at {where}")
                    raise Refusal(OVERFLOW, f"the conversion to BF16 overflows at {where}")
                new_sub[r0:r1] = np.where(sel, bits, sub[r0:r1])
        out[:, cols_ix] = new_sub
        _apply_fixups(out, self.fixups)
        return out

    def operand_digests(self) -> dict[str, bytes]:
        return {"a": operand_digest_bytes(self.a_bits), "b": operand_digest_bytes(self.b_bits)}

    def work(self) -> int:
        return self.rows * len(self.columns) * self.rank


# ── remove_direction (v1) ─────────────────────────────────────────────────────


class RemoveDirection:
    """A parsed `remove_direction` op: `u` (rows values), `alpha`, `columns` S."""

    def __init__(self, op: dict, shape, *, u_key: str = "u"):
        self.dtype, digits, _ = _profile(op)
        self.profile = op.get("numerics")
        self.rows, self.cols = _shape(shape)
        self.fixups = _fixups(op.get("fixups"))
        self.columns = _columns(op.get("columns"), strict_entries=False)
        u = op.get(u_key)
        if not isinstance(u, list):
            raise Refusal(MALFORMED, f"{u_key} is not a list")
        if len(u) != self.rows:
            raise Refusal(SHAPE_INVALID, f"len(u) = {len(u)}, rows = {self.rows}")
        self.u_bits = parse_hex(u, digits, u_key)
        self.alpha_bits = parse_hex([op.get("alpha")], digits, "alpha")
        if _nonfinite_operand(self.u_bits, digits) or _nonfinite_operand(self.alpha_bits, digits):
            raise Refusal(NON_FINITE_OPERAND, "u or alpha is NaN or infinite")
        self.u = operand_values(self.u_bits, self.dtype)
        self.alpha = operand_values(self.alpha_bits, self.dtype)[0]

    def apply(self, base: np.ndarray, tensor_shape=None, tensor_role=None) -> np.ndarray:
        """v1 §3: for each j in S, `c = sum_r u[r]*w[r]` sequentially from +0, then `w - (alpha*u[r])*c`, chained.
        Vectorised across the columns of S only; the reduction over r is a loop in row order."""
        rows, cols = self.rows, self.cols
        grid = _header(base, rows, cols, tensor_shape, tensor_role)
        if self.columns[-1] >= cols:
            raise Refusal(COLUMNS_INVALID, f"column {self.columns[-1]} >= {cols}")
        _fixups_in(self.fixups, rows, cols, set(self.columns))
        cols_ix = np.asarray(self.columns, dtype=np.int64)
        sub = grid[:, cols_ix]
        if not np.all(bf16_is_finite(sub)):
            r, p = np.argwhere(~bf16_is_finite(sub))[0]
            raise Refusal(NON_FINITE_OPERAND, f"the base element at ({r}, {self.columns[p]}) is NaN or infinite")
        dt = self.dtype
        w = bf16_to(sub, dt)
        with np.errstate(all="ignore"):
            c = np.zeros(len(self.columns), dtype=dt)  # +0
            bad_c = np.zeros(len(self.columns), dtype=bool)
            for r in range(rows):
                prod = self.u[r] * w[r]
                c = c + prod
                assert prod.dtype == dt and c.dtype == dt
                bad_c |= ~np.isfinite(prod) | ~np.isfinite(c)
            scaled = self.alpha * self.u  # mul(alpha, u[r]) for each r: one rounding each
            assert scaled.dtype == dt
            bad_s = ~np.isfinite(scaled)
            term = scaled[:, None] * c[None, :]
            res = w - term  # sub(w, x) = add(w, -x): IEEE subtraction is exactly that, signed zeros included
            assert term.dtype == dt and res.dtype == dt
            bad = bad_s[:, None] | ~np.isfinite(term) | ~np.isfinite(res)
            bits, over = chain(np.where(bad, dt(0), res), self.profile)
            over &= ~bad
        for p in range(len(self.columns)):  # j outermost, then r: the first failing operation decides the code
            if bad_c[p]:
                raise Refusal(NON_FINITE_INTERMEDIATE, f"the dot product overflows W at column {self.columns[p]}")
            failing = np.flatnonzero(bad[:, p] | over[:, p])
            if failing.size:
                r = failing[0]
                code = NON_FINITE_INTERMEDIATE if bad[r, p] else OVERFLOW
                raise Refusal(code, f"at ({r}, {self.columns[p]})")
        out = grid.copy()
        out[:, cols_ix] = bits
        _apply_fixups(out, self.fixups)
        return out

    def operand_digests(self) -> dict[str, bytes]:
        return {"u": operand_digest_bytes(self.u_bits), "alpha": operand_digest_bytes(self.alpha_bits)}

    def work(self) -> int:
        return 2 * self.rows * len(self.columns)


def apply_vector_case(op: dict, base_bits: list[str], base_shape, tensor_role=None) -> list[str]:
    """Run one `vectors.json` case's op (v1 or v2 kind) on its base: the bits as 4-digit hex, or `Refusal`."""
    kind = op.get("kind")
    shape = op.get("shape")
    base = np.array([int(h, 16) for h in base_bits], dtype=np.uint16)
    if kind == "add_low_rank_columns":
        parsed = LowRank(op, shape)
    elif kind == "add_low_rank":
        parsed = LowRank(op, shape, v1_full=True)
    elif kind == "remove_direction":
        parsed = RemoveDirection(op, shape)
    else:
        raise ValueError(f"op kind {kind!r} is not implemented here")
    out = parsed.apply(base, tensor_shape=tuple(base_shape), tensor_role=tensor_role)
    return [f"{int(b):04x}" for b in out.reshape(-1)]
