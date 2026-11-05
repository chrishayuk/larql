"""Q8_0 matrix-vector product reference: `Q8_0_GEMV_SEQ` and `Q8_0_GEMV_B1`.

Source: Q8_0-GEMV-SPEC.md revision 1 (sha256 c3b439c0...), admitted by D15 clarification 3. Development
only; nothing here is blind acceptance evidence. Results are measurements. Section 5 of the specification is a
paper derivation that no proof system has checked, and `B1` is experimental.

Everything takes raw bytes: `blocks` (n*(k/32)*34 bytes), `x_bits` (k binary32 bit patterns), `y_bits` (n binary32
bit patterns). No container or file format is involved.

* `seq_bits` is section 3's program, evaluated in binary32 with one numpy ufunc call per `*` and per `+` (so no
  fused multiply-add), strictly in ascending j. It never calls a library sum routine.
* The checks do their arithmetic in exact integers, never in floating point:
  every quantity is a whole number of units of U = 2**-173. A binary16 scale is a multiple of 2**-24, a binary32
  value is a multiple of 2**-149, so q * d * x is a multiple of 2**-173. `y*`, `S` and `B` are therefore exact,
  and the B1 verdict is the exact-rational verdict (compare |y - y*| <= B by integer cross-multiplication).
  There is no faster path that could differ.

Outside the domain (a non-finite scale or x, or `(1 + gamma_k)(S + k*eta) >= Omega` for some row) the verdict for the
whole input is `could_not_evaluate`; it is never `satisfied`. A non-finite `y_i` inside the domain is `violated`.
`n = 0` is `could_not_evaluate` (nothing was measured). Malformed input is refused the same way (`malformed_input`
reasons) by the checks, and raises `ValueError` from `seq_bits`.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from typing import Optional

import numpy as np

from . import q8_0

SATISFIED = "satisfied"
VIOLATED = "violated"
COULD_NOT_EVALUATE = "could_not_evaluate"

K_LIMIT = 1 << 24  # k < 2**24 (spec section 1)
_TWO24 = 1 << 24
# Exact constants in units of U = 2**-173.
_ETA_U = 1 << 23  # eta = 2**-150 = 2**23 * 2**-173
_OMEGA_U = ((1 << 128) - (1 << 103)) << 173  # Omega = 2**128 - 2**103
_MAX_LISTED = 8  # rows named in a reason string


@dataclass(frozen=True)
class GemvResult:
    status: str
    reasons: tuple
    rows_checked: int
    rows_violating: int
    first_violating_row: Optional[int]
    # B1 only: worst |y - y*| / B over the rows checked, as a decimal string ("inf" for a non-finite y), and its row.
    worst_ratio: Optional[str] = None
    worst_ratio_row: Optional[int] = None


class _Malformed(ValueError):
    pass


# ---------------------------------------------------------------- input handling


def _int_arg(v, name):
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)):
        raise _Malformed(f"{name} is not an integer ({type(v).__name__})")
    return int(v)


def _bits_arg(v, name, count):
    """A sequence of `count` binary32 bit patterns -> uint32 array. Floats are refused: a bit pattern is required."""
    if isinstance(v, np.ndarray):
        if v.ndim != 1 or v.dtype.kind not in "ui":
            raise _Malformed(f"{name} must be a 1-D integer array of bit patterns (got dtype {v.dtype}, ndim {v.ndim})")
        if v.size != count:
            raise _Malformed(f"{name} has {v.size} entries, expected {count}")
        if v.size and (int(v.min()) < 0 or int(v.max()) >= 1 << 32):
            raise _Malformed(f"{name} holds a value outside 0..2**32-1")
        return v.astype(np.uint32)
    if isinstance(v, (list, tuple)):
        if len(v) != count:
            raise _Malformed(f"{name} has {len(v)} entries, expected {count}")
        for e in v:
            if isinstance(e, (bool, np.bool_)) or not isinstance(e, (int, np.integer)) or not 0 <= int(e) < 1 << 32:
                raise _Malformed(f"{name} holds an entry that is not an integer in 0..2**32-1")
        return np.array([int(e) for e in v], dtype=np.uint32)
    raise _Malformed(f"{name} must be an integer array or a list of ints, got {type(v).__name__}")


def _validate(blocks, x_bits, y_bits, n, k, need_y=True):
    n = _int_arg(n, "n")
    k = _int_arg(k, "k")
    if n < 0:
        raise _Malformed(f"n = {n} is negative")
    if k <= 0:
        raise _Malformed(f"k = {k} is not positive")
    if k % 32:
        raise _Malformed(f"k = {k} is not a multiple of 32")
    if k >= K_LIMIT:
        raise _Malformed(f"k = {k} is not below 2**24")
    if isinstance(blocks, np.ndarray):
        if blocks.dtype != np.uint8 or blocks.ndim != 1:
            raise _Malformed("blocks must be bytes-like (a 1-D uint8 array is accepted)")
        raw = blocks.tobytes()
    elif isinstance(blocks, (bytes, bytearray, memoryview)):
        raw = bytes(blocks)
    else:
        raise _Malformed(f"blocks must be bytes-like, got {type(blocks).__name__}")
    want = n * (k // 32) * q8_0.BLOCK_BYTES
    if len(raw) != want:
        raise _Malformed(f"blocks has {len(raw)} bytes, expected n*(k/32)*34 = {want}")
    x = _bits_arg(x_bits, "x_bits", k)
    y = _bits_arg(y_bits, "y_bits", n) if need_y else None
    return raw, x, y, n, k


# ---------------------------------------------------------------- section 3's program


def seq_bits(blocks, x_bits, n, k) -> np.ndarray:
    """Section 3: for each row, acc = +0.0; acc = acc + (w_ij * x_j) for j ascending; return the 32-bit pattern.

    Each `*` and `+` is one numpy float32 ufunc call (IEEE binary32, round to nearest even, no FMA, subnormals
    kept). The loop over j is a Python loop; rows are independent and run side by side. Returns a uint32 array of n
    patterns. Defined for any bit patterns (a non-finite input follows IEEE rules; a NaN's payload is whatever the host
    produces); the checks only call it inside the domain. Raises ValueError on malformed input.
    """
    raw, x, _, n, k = _validate(blocks, x_bits, None, n, k, need_y=False)
    if n == 0:
        return np.zeros(0, dtype=np.uint32)
    with np.errstate(all="ignore"):
        w = q8_0.dequantize_q8_0_rows(raw, n, k)  # q * d in binary32: exact for finite d (spec section 2)
        wt = np.ascontiguousarray(w.T)  # (k, n)
        xs = x.view(np.float32)
        acc = np.zeros(n, dtype=np.float32)  # +0.0
        p = np.empty(n, dtype=np.float32)
        for j in range(k):
            np.multiply(wt[j], xs[j], out=p, dtype=np.float32)
            np.add(acc, p, out=acc, dtype=np.float32)
    return acc.view(np.uint32).copy()


# ---------------------------------------------------------------- exact arithmetic


def _f16_int(h: int):
    """Finite binary16 pattern -> signed integer in units of 2**-24, or None if infinite/NaN."""
    e = (h >> 10) & 31
    m = h & 1023
    if e == 31:
        return None
    mag = m if e == 0 else (1024 + m) << (e - 1)
    return -mag if h & 0x8000 else mag


def _f32_int(b: int):
    """Finite binary32 pattern -> signed integer in units of 2**-149, or None if infinite/NaN."""
    e = (b >> 23) & 255
    m = b & 0x7FFFFF
    if e == 255:
        return None
    mag = m if e == 0 else ((1 << 23) + m) << (e - 1)
    return -mag if b & 0x80000000 else mag


def exact_rows(raw: bytes, x: np.ndarray, n: int, k: int):
    """Exact per-row (y*, S) in units of U = 2**-173, plus the lists of rows with a non-finite scale and of non-finite x.

    Returns (ystar, S, bad_scale_rows, bad_x_indices); ystar and S hold None for a row with a non-finite scale.
    """
    nb = k // 32
    xi = [_f32_int(int(b)) for b in x]
    bad_x = [j for j, v in enumerate(xi) if v is None]
    xi = [0 if v is None else v for v in xi]
    xblocks = [xi[b * 32 : (b + 1) * 32] for b in range(nb)]
    xabs = [[abs(v) for v in xb] for xb in xblocks]
    ystar, S, bad_rows = [], [], []
    for i in range(n):
        ys = ss = 0
        bad = False
        for b in range(nb):
            off = (i * nb + b) * q8_0.BLOCK_BYTES
            D = _f16_int(raw[off] | (raw[off + 1] << 8))
            if D is None:
                bad = True
                break
            q = [c - 256 if c > 127 else c for c in raw[off + 2 : off + 34]]
            ys += D * sum(qc * xv for qc, xv in zip(q, xblocks[b]))
            ss += abs(D) * sum(abs(qc) * xv for qc, xv in zip(q, xabs[b]))
        if bad:
            bad_rows.append(i)
            ystar.append(None)
            S.append(None)
        else:
            ystar.append(ys)
            S.append(ss)
    return ystar, S, bad_rows, bad_x


def _in_domain_third(S: int, k: int) -> bool:
    """(1 + gamma_k)(S + k*eta) < Omega, exactly: 2**24 (S + k*eta) < Omega (2**24 - k)."""
    return _TWO24 * (S + k * _ETA_U) < _OMEGA_U * (_TWO24 - k)


def _y_int(yb: int):
    return _f32_int(yb)  # units 2**-149, or None


def _decimal(num: int, den: int) -> str:
    with localcontext() as ctx:
        ctx.prec = 30
        return format(Decimal(num) / Decimal(den), ".12e")


def _list_rows(rows):
    shown = ",".join(str(r) for r in rows[:_MAX_LISTED])
    return f"{shown}{',...' if len(rows) > _MAX_LISTED else ''} ({len(rows)} in all)"


def _refuse(reasons, extra=()):
    return GemvResult(COULD_NOT_EVALUATE, tuple(reasons) + tuple(extra), 0, 0, None)


def _prepare(blocks, x_bits, y_bits, n, k):
    """Validate, run the domain checks. Returns (early_result, context)."""
    try:
        raw, x, y, n, k = _validate(blocks, x_bits, y_bits, n, k)
    except _Malformed as e:
        return _refuse([f"malformed_input: {e}"]), None
    if n == 0:
        return _refuse(["no_rows: n = 0, nothing was measured"]), None
    ystar, S, bad_rows, bad_x = exact_rows(raw, x, n, k)
    reasons = []
    if bad_x:
        reasons.append(f"x_not_finite: columns {_list_rows(bad_x)}")
    if bad_rows:
        reasons.append(f"scale_not_finite: rows {_list_rows(bad_rows)}")
    out = [i for i in range(n) if S[i] is not None and not _in_domain_third(S[i], k)]
    if out:
        reasons.append(f"outside_domain: (1+gamma_k)(S+k*eta) >= Omega in rows {_list_rows(out)}")
    if reasons:
        return _refuse(reasons), None
    return None, (raw, x, y, n, k, ystar, S)


# ---------------------------------------------------------------- the two contracts


def check_seq(blocks, x_bits, y_bits, n, k) -> GemvResult:
    """`Q8_0_GEMV_SEQ`: every y_i has exactly the bit pattern of section 3's program (the sign of zero included)."""
    early, ctx = _prepare(blocks, x_bits, y_bits, n, k)
    if early is not None:
        return early
    raw, x, y, n, k, _, _ = ctx
    res = seq_bits(raw, x, n, k)
    bad = [i for i in range(n) if int(res[i]) != int(y[i])]
    if bad:
        return GemvResult(VIOLATED, (f"seq_bits_mismatch: rows {_list_rows(bad)}",), n, len(bad), bad[0])
    return GemvResult(SATISFIED, (), n, 0, None)


def check_b1(blocks, x_bits, y_bits, n, k) -> GemvResult:
    """`Q8_0_GEMV_B1` (experimental): |y_i - y*_i| <= B_i for every row, exactly, with B = gamma_k S + k eta (1 + gamma_k).

    With gamma_k = k/(2**24 - k) and 1 + gamma_k = 2**24/(2**24 - k), the test multiplied through by (2**24 - k) is
    the integer inequality  |Y - Y*| (2**24 - k) <= k S + k eta 2**24  (all in units of 2**-173).
    """
    early, ctx = _prepare(blocks, x_bits, y_bits, n, k)
    if early is not None:
        return early
    raw, x, y, n, k, ystar, S = ctx
    bad, nonfinite = [], []
    worst_num, worst_den, worst_row, worst_inf = -1, 1, None, False
    for i in range(n):
        yi = _y_int(int(y[i]))
        if yi is None:
            nonfinite.append(i)
            bad.append(i)
            if not worst_inf:
                worst_inf, worst_row = True, i
            continue
        dif = abs(yi * (1 << 24) - ystar[i])  # y is in units 2**-149; times 2**24 gives units 2**-173
        lhs = dif * (_TWO24 - k)
        rhs = k * S[i] + k * _ETA_U * _TWO24  # > 0 always (the eta term)
        if lhs > rhs:
            bad.append(i)
        if not worst_inf and lhs * worst_den > worst_num * rhs:
            worst_num, worst_den, worst_row = lhs, rhs, i
    if worst_inf:
        ratio = "inf"
    else:
        ratio = _decimal(worst_num, worst_den)
    reasons = []
    if nonfinite:
        reasons.append(f"y_not_finite: rows {_list_rows(nonfinite)}")
    over = [i for i in bad if i not in set(nonfinite)]
    if over:
        reasons.append(f"b1_exceeded: rows {_list_rows(over)}")
    if bad:
        return GemvResult(VIOLATED, tuple(reasons), n, len(bad), bad[0], ratio, worst_row)
    return GemvResult(SATISFIED, (), n, 0, None, ratio, worst_row)
