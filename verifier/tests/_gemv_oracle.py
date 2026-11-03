"""An independent exact oracle for Q8_0-GEMV-SPEC.md, for tests only.

It shares no code with refcheck/gemv.py: scales and x are decoded with `struct` (binary16 'e', binary32 'f'), all
arithmetic is `fractions.Fraction`, and binary32 rounding is written out here (round to nearest, ties to even, with
subnormals and overflow, and the sign of zero). It is slow and meant for small cases.
"""

import struct
from fractions import Fraction

TWO24 = 1 << 24
OMEGA = Fraction(2**128 - 2**103)
ETA = Fraction(1, 2**150)
U = Fraction(1, 2**24)


def f16_value(h):
    """Exact value of a finite binary16 pattern as a Fraction (None if inf/NaN); sign of zero is not represented."""
    f = struct.unpack("<e", struct.pack("<H", h))[0]
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return Fraction(f)


def f32_value(b):
    f = struct.unpack("<f", struct.pack("<I", b))[0]
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return Fraction(f)


def f32_bits_of_float(f):
    return struct.unpack("<I", struct.pack("<f", f))[0]


def _round_half_even(fr):
    fl = fr.numerator // fr.denominator
    rem = fr - fl
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and fl % 2 == 1):
        fl += 1
    return fl


def round_to_f32(r: Fraction, sign_if_zero: int):
    """Round an exact real to binary32 bits. A zero result takes `sign_if_zero` (0/1); a non-zero real that
    underflows to zero keeps its own sign. Raises OverflowError if the rounded result is infinite."""
    if r == 0:
        return sign_if_zero << 31
    sign = 1 if r < 0 else 0
    a = abs(r)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    elif Fraction(2) ** (e + 1) <= a:
        e += 1
    if e > 127:
        raise OverflowError
    eff = max(e, -126)
    quantum = Fraction(2) ** (eff - 23)
    m = _round_half_even(a / quantum)
    if m == 0:
        return sign << 31
    if eff == -126 and m < (1 << 23):
        return (sign << 31) | m  # subnormal
    if m == (1 << 24):
        eff += 1
        m = 1 << 23
    if eff > 127:
        raise OverflowError
    return (sign << 31) | ((eff + 127) << 23) | (m - (1 << 23))


def _val_and_sign(bits):
    v = f32_value(bits)
    return v, bits >> 31


def mul_bits(a, b):
    va, sa = _val_and_sign(a)
    vb, sb = _val_and_sign(b)
    return round_to_f32(va * vb, sa ^ sb)


def add_bits(a, b):
    va, sa = _val_and_sign(a)
    vb, sb = _val_and_sign(b)
    s = va + vb
    if s == 0:
        both_zero_same_sign = va == 0 and vb == 0 and sa == sb
        return round_to_f32(s, sa if both_zero_same_sign else 0)
    return round_to_f32(s, 0)


def weight_bits(q: int, d_bits: int):
    """w = q * d in binary32 (q converted to binary32 first, as a C float multiply does)."""
    d32 = f32_bits_of_float(struct.unpack("<e", struct.pack("<H", d_bits))[0])
    return mul_bits(f32_bits_of_float(float(q)), d32)


def parse(raw: bytes, n: int, k: int):
    """-> list of rows, each a list of (q, d_bits) per column."""
    nb = k // 32
    rows = []
    for i in range(n):
        row = []
        for b in range(nb):
            off = (i * nb + b) * 34
            d = raw[off] | (raw[off + 1] << 8)
            for c in raw[off + 2 : off + 34]:
                row.append((c - 256 if c > 127 else c, d))
        rows.append(row)
    return rows


def seq_oracle(raw: bytes, x_bits, n: int, k: int):
    """Section 3 with exact rational arithmetic and an explicit binary32 rounding of each * and +."""
    out = []
    for row in parse(raw, n, k):
        acc = 0  # +0.0
        for (q, d), xb in zip(row, x_bits):
            w = weight_bits(q, d)
            acc = add_bits(acc, mul_bits(w, xb))
        out.append(acc)
    return out


def exact_row(raw: bytes, x_bits, n: int, k: int):
    """-> list of (ystar, S) as Fractions, or None for a row with a non-finite scale; None x -> raises ValueError."""
    xs = [f32_value(b) for b in x_bits]
    if any(v is None for v in xs):
        raise ValueError("non-finite x")
    res = []
    for row in parse(raw, n, k):
        ds = [f16_value(d) for _, d in row]
        if any(v is None for v in ds):
            res.append(None)
            continue
        ys = sum(Fraction(q) * d * x for (q, _), d, x in zip(row, ds, xs))
        S = sum(abs(Fraction(q) * d * x) for (q, _), d, x in zip(row, ds, xs))
        res.append((ys, S))
    return res


def b1_bound(S: Fraction, k: int) -> Fraction:
    g = Fraction(k) * U / (1 - Fraction(k) * U)
    return g * S + k * ETA * (1 + g)


def in_domain(S: Fraction, k: int) -> bool:
    g = Fraction(k) * U / (1 - Fraction(k) * U)
    return (1 + g) * (S + k * ETA) < OMEGA


def b1_oracle(raw, x_bits, y_bits, n, k):
    """-> ('satisfied'|'violated'|'could_not_evaluate', worst_ratio Fraction or None, worst_row, violating_rows)."""
    if any(f32_value(b) is None for b in x_bits):
        return "could_not_evaluate", None, None, []
    ex = exact_row(raw, x_bits, n, k)
    if n == 0 or any(e is None for e in ex) or any(not in_domain(e[1], k) for e in ex):
        return "could_not_evaluate", None, None, []
    viol, worst, worst_row = [], None, None
    for i, (ys, S) in enumerate(ex):
        yv = f32_value(y_bits[i])
        if yv is None:
            viol.append(i)
            worst, worst_row = "inf", i
            continue
        B = b1_bound(S, k)
        r = abs(yv - ys) / B
        if abs(yv - ys) > B:
            viol.append(i)
        if worst != "inf" and (worst is None or r > worst):
            worst, worst_row = r, i
    return ("violated" if viol else "satisfied"), worst, worst_row, viol


# ---------------------------------------------------------------- building inputs


def block(d_bits: int, qs) -> bytes:
    assert len(qs) == 32
    return struct.pack("<H", d_bits) + bytes(q & 0xFF for q in qs)


def build(rows):
    """rows: list of rows; each row a list of (d_bits, qs[32]) blocks -> bytes."""
    return b"".join(block(d, qs) for row in rows for d, qs in row)


def pack_bits(values):
    import numpy as np

    return np.array(values, dtype=np.uint32)
