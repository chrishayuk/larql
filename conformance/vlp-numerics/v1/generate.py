"""Golden vectors for the VLP structured-op numerical profiles, v1 (see SPEC.md).

    python3 -I conformance/vlp-numerics/v1/generate.py           # write vectors.json
    python3 -I conformance/vlp-numerics/v1/generate.py --check   # regenerate in memory, compare, cross-check

This is the "third formulation" of the frozen G2 design (docs/wsp-001-g2-design.md §2): a scalar loop in pure
Python where every rounding is computed exactly with `fractions.Fraction`. It imports nothing but the standard
library, and nothing from LARQL. Every expected value is also recomputed by a second, independent route
(`native_*`: CPython binary64 floats, `round()` half-to-even, and `struct` for f32), and the two must agree.

A case built to discriminate a semantics (FMA, reduction order, association, chain, subnormals, ...) records the
result under each named wrong semantics, and the generator asserts that the target one changes at least one byte.
"""
import hashlib
import json
import math
import os
import random
import struct
import sys
from fractions import Fraction

SCHEMA = "vlp-numerics-golden/v1"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "vectors.json")
SEARCH_SEED = 20261107  # fixed: the vectors are a pure function of this file
MAX_SEARCH = 20000

# (precision in bits including the implicit one, emin, emax, total bits, exponent bits)
FMT = {"f64": (53, -1022, 1023, 64, 11), "f32": (24, -126, 127, 32, 8), "bf16": (8, -126, 127, 16, 8)}

PROFILES = {
    "f64-seq-rne1/v1": {"work": "f64", "chain": ["bf16"]},
    "f64-seq-rne2/v1": {"work": "f64", "chain": ["f32", "bf16"]},
    "f32-seq-rne1/v1": {"work": "f32", "chain": ["bf16"]},
}

REFUSALS = ["numerics_invalid", "numerics_forbidden", "shape_invalid", "rank_invalid", "columns_invalid",
            "fixups_invalid", "fixups_forbidden", "fixup_no_op", "non_finite_operand",
            "non_finite_intermediate", "overflow", "tensor_refused"]


class Refuse(Exception):
    def __init__(self, code):
        assert code in REFUSALS, code
        super().__init__(code)
        self.code = code


# ---------------------------------------------------------------- exact values: (Fraction, negative-zero flag)

def is_neg(v):
    return v[0] < 0 or (v[0] == 0 and v[1])


def neg(v):
    return (-v[0], (not v[1]) if v[0] == 0 else False)


def floor_log2(a):
    """Largest e with 2**e <= a, for a Fraction a > 0."""
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    elif Fraction(2) ** (e + 1) <= a:
        e += 1
    return e


def is_subnormal(v, fmt):
    _, emin, *_ = FMT[fmt]
    return v[0] != 0 and abs(v[0]) < Fraction(2) ** emin


def rnd(x, zneg, fmt, ftz=False):
    """Round the exact x to `fmt`, nearest, ties to even, with gradual underflow (or flush, if `ftz`).
    zneg is the sign to give an exact zero. Returns (Fraction, negzero). Raises OverflowError."""
    if x == 0:
        return (Fraction(0), bool(zneg))
    p, emin, emax, *_ = FMT[fmt]
    sign = x < 0
    a = -x if sign else x
    e = max(floor_log2(a), emin)
    q = Fraction(2) ** (e - p + 1)
    m = a / q
    n = m.numerator // m.denominator
    rem = m - n
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and n % 2 == 1):
        n += 1
    val = n * q
    if val >= Fraction(2) ** (emax + 1):
        raise OverflowError(fmt)
    if ftz and val != 0 and val < Fraction(2) ** emin:
        val = Fraction(0)
    if val == 0:
        return (Fraction(0), sign)
    return (-val if sign else val, False)


def decode(bits, fmt):
    """Bits to an exact value. Raises Refuse on NaN/inf (the caller states what the bits were)."""
    p, emin, emax, nb, eb = FMT[fmt]
    mbits = p - 1
    s = (bits >> (nb - 1)) & 1
    ex = (bits >> mbits) & ((1 << eb) - 1)
    mant = bits & ((1 << mbits) - 1)
    if ex == (1 << eb) - 1:
        raise ValueError("non-finite")
    if ex == 0:
        val = mant * Fraction(2) ** (emin - mbits)
    else:
        val = ((1 << mbits) + mant) * Fraction(2) ** (ex - (1 << (eb - 1)) + 1 - mbits)
    return ((-val if s else val), bool(s) and val == 0)


def encode(v, fmt):
    p, emin, emax, nb, eb = FMT[fmt]
    mbits = p - 1
    s = 1 if is_neg(v) else 0
    a = abs(v[0])
    if a == 0:
        return s << (nb - 1)
    if a < Fraction(2) ** emin:
        m = a / Fraction(2) ** (emin - mbits)
        assert m.denominator == 1
        return (s << (nb - 1)) | m.numerator
    e = floor_log2(a)
    m = a / Fraction(2) ** (e - mbits)
    assert m.denominator == 1 and (1 << mbits) <= m.numerator < (1 << p)
    return (s << (nb - 1)) | ((e + (1 << (eb - 1)) - 1) << mbits) | (m.numerator - (1 << mbits))


def exact(x, fmt):
    """An exactly representable value in `fmt`, as bits (used by the builders only)."""
    v = (Fraction(x), False) if not isinstance(x, tuple) else x
    r = rnd(v[0], v[1], fmt)
    assert r[0] == v[0], f"{x} is not exact in {fmt}"
    return encode(r, fmt)


# ---------------------------------------------------------------- the semantics, with named wrong variants

class Arith:
    """Arithmetic in one working format. `variant` selects a wrong semantics, for discrimination only."""

    def __init__(self, fmt, ftz=False):
        self.fmt, self.ftz = fmt, ftz

    def _r(self, x, zneg):
        try:
            return rnd(x, zneg, self.fmt, self.ftz)
        except OverflowError:
            raise Refuse("non_finite_intermediate")

    def _in(self, v):
        return (Fraction(0), is_neg(v)) if self.ftz and is_subnormal(v, self.fmt) else v  # DAZ

    def mul(self, a, b):
        a, b = self._in(a), self._in(b)
        return self._r(a[0] * b[0], is_neg(a) != is_neg(b))

    def add(self, a, b):
        a, b = self._in(a), self._in(b)
        s = a[0] + b[0]
        zneg = (a[0] == 0 and b[0] == 0 and is_neg(a) and is_neg(b))  # x + (-x) = +0 under RNE
        return self._r(s, zneg)

    def sub(self, a, b):
        return self.add(a, neg(b))

    def fma(self, a, b, c):
        a, b, c = self._in(a), self._in(b), self._in(c)
        prod_neg = is_neg(a) != is_neg(b)
        s = a[0] * b[0] + c[0]
        zneg = s == 0 and a[0] * b[0] == 0 and c[0] == 0 and prod_neg and is_neg(c)
        return self._r(s, zneg)


def convert(v, profile, variant):
    """The profile's conversion chain to BF16 bits. Overflow at any step is `overflow`."""
    chain = PROFILES[profile]["chain"]
    if variant.get("chain"):
        chain = variant["chain"]
    for fmt in chain:
        try:
            v = rnd(v[0], is_neg(v), fmt, ftz=(variant.get("ftz", False) and fmt != "bf16")
                  or (variant.get("bf16_flush", False) and fmt == "bf16"))
        except OverflowError:
            raise Refuse("overflow")
    return encode(v, "bf16")


def operand(bits, fmt):
    try:
        return decode(bits, fmt)
    except ValueError:
        raise Refuse("non_finite_operand")


def reduce_terms(ar, terms, order, fused=None):
    """Sum `terms` (a list of (u, w) pairs) from +0. order: forward | reverse | pairwise. fused: FMA."""
    if order == "pairwise":
        prods = [ar.mul(u, w) for u, w in terms]

        def pw(xs):
            if len(xs) == 1:
                return ar.add((Fraction(0), False), xs[0])
            h = len(xs) // 2
            return ar.add(pw(xs[:h]), pw(xs[h:]))
        return pw(prods)
    seq = terms if order == "forward" else list(reversed(terms))
    acc = (Fraction(0), False)
    for u, w in seq:
        acc = ar.fma(u, w, acc) if fused else ar.add(acc, ar.mul(u, w))
    return acc


def validate_common(case, op):
    """Parse-phase and header-phase checks shared by every op. Returns (rows, cols, base_values)."""
    shape = op.get("shape")
    if not (isinstance(shape, list) and len(shape) == 2 and all(type(x) is int and x > 0 for x in shape)):
        raise Refuse("shape_invalid")
    rows, cols = shape
    fixups = op.get("fixups", [])
    keys = [(f[0], f[1]) for f in fixups]
    if keys != sorted(keys) or len(set(keys)) != len(keys):
        raise Refuse("fixups_invalid")
    for f in fixups:
        try:
            decode(int(f[2], 16), "bf16")
        except ValueError:
            raise Refuse("non_finite_operand")
    return rows, cols, fixups


def check_columns(cols_list, cols):
    if not cols_list or cols_list != sorted(set(cols_list)):
        raise Refuse("columns_invalid")


def header_phase(case, rows, cols, cols_list):
    base = case["base"]
    if base["shape"] != [rows, cols]:
        raise Refuse("shape_invalid")
    if case.get("tensor_role") == "embed_tokens":
        raise Refuse("tensor_refused")
    if cols_list is not None and cols_list[-1] >= cols:
        raise Refuse("columns_invalid")
    w = []
    for h in base["bf16"]:
        try:
            w.append(decode(int(h, 16), "bf16"))
        except ValueError:
            w.append(None)  # NaN or inf: kept if untouched, refused if an arithmetic op reads it
    return w


def check_touched_finite(w_all, idx):
    """Tensor-bytes phase: every base element an arithmetic op reads must be finite."""
    if any(w_all[i] is None for i in idx):
        raise Refuse("non_finite_operand")


def check_fixups(fixups, rows, cols, touched):
    """Header phase: every fixup is inside the tensor and inside the op's touched set."""
    for r, c, h in fixups:
        if not (0 <= r < rows and 0 <= c < cols) or not touched(r, c):
            raise Refuse("fixups_invalid")


def apply_fixups(out, fixups, rows, cols, computed):
    for r, c, h in fixups:
        if int(h, 16) == computed[r * cols + c]:
            raise Refuse("fixup_no_op")
        out[r * cols + c] = int(h, 16)


def run(case, variant=None):
    """The whole op on one case. Returns ("bf16", [bits...]) or ("refuse", code)."""
    variant = variant or {}
    op = case["op"]
    try:
        kind = op["kind"]
        if kind == "zero_columns":
            return ("bf16", run_zero_columns(case, op, variant))
        prof = op.get("numerics")
        if prof not in PROFILES:
            raise Refuse("numerics_invalid")
        return ("bf16", (run_remove_direction if kind == "remove_direction" else run_add_low_rank)(case, op, prof, variant))
    except Refuse as e:
        return ("refuse", e.code)


def run_zero_columns(case, op, variant):
    if "numerics" in op:
        raise Refuse("numerics_forbidden")
    rows, cols, fixups = validate_common(case, op)
    if fixups:
        raise Refuse("fixups_forbidden")
    S = op["columns"]
    check_columns(S, cols)
    header_phase(case, rows, cols, S)
    out = [int(h, 16) for h in case["base"]["bf16"]]
    for r in range(rows):
        for j in S:
            out[r * cols + j] = (out[r * cols + j] & 0x8000) if variant.get("keep_sign") else 0x0000
    return out


def run_remove_direction(case, op, prof, variant):
    work = PROFILES[prof]["work"]
    rows, cols, fixups = validate_common(case, op)
    S = op["columns"]
    check_columns(S, cols)
    if len(op["u"]) != rows:
        raise Refuse("shape_invalid")
    u = [operand(int(h, 16), work) for h in op["u"]]
    alpha = operand(int(op["alpha"], 16), work)
    w_all = header_phase(case, rows, cols, S)
    in_S = set(S)
    check_fixups(fixups, rows, cols, lambda r, c: c in in_S)
    check_touched_finite(w_all, [r * cols + j for j in S for r in range(rows)])
    ar = Arith(variant.get("accum_fmt", work), variant.get("ftz", False))
    ar_main = Arith(work, variant.get("ftz", False))
    out = [int(h, 16) for h in case["base"]["bf16"]]
    for j in S:
        col = [w_all[r * cols + j] for r in range(rows)]
        c = reduce_terms(ar, list(zip(u, col)), variant.get("order", "forward"), variant.get("fma"))
        if variant.get("accum_fmt"):
            c = ar_main._r(c[0], is_neg(c))
        for r in range(rows):
            if variant.get("assoc") == "alpha_times_uc":
                t = ar_main.mul(alpha, ar_main.mul(u[r], c))
            else:
                t = ar_main.mul(ar_main.mul(alpha, u[r]), c)
            out[r * cols + j] = convert(ar_main.sub(col[r], t), prof, variant)
    computed = list(out)
    apply_fixups(out, fixups, rows, cols, computed)
    return out


def run_add_low_rank(case, op, prof, variant):
    work = PROFILES[prof]["work"]
    rows, cols, fixups = validate_common(case, op)
    K = op.get("rank")
    if type(K) is not int or K < 1:
        raise Refuse("rank_invalid")
    if len(op["a"]) != rows * K or len(op["b"]) != K * cols:
        raise Refuse("shape_invalid")
    a = [operand(int(h, 16), work) for h in op["a"]]
    b = [operand(int(h, 16), work) for h in op["b"]]
    w_all = header_phase(case, rows, cols, None)
    check_fixups(fixups, rows, cols, lambda r, c: True)
    check_touched_finite(w_all, range(rows * cols))
    ar = Arith(work, variant.get("ftz", False))
    out = []
    for r in range(rows):
        for j in range(cols):
            if variant.get("start_from_zero"):
                s = (Fraction(0), False)
                for k in range(K):
                    s = ar.add(s, ar.mul(a[r * K + k], b[k * cols + j]))
                w = ar.add(w_all[r * cols + j], s)
            else:
                w = w_all[r * cols + j]
                for k in range(K):
                    w = ar.add(w, ar.mul(a[r * K + k], b[k * cols + j]))
            out.append(convert(w, prof, variant))
    computed = list(out)
    apply_fixups(out, fixups, rows, cols, computed)
    return out


# ---------------------------------------------------------------- the independent native route (f64 profiles)

def native_bf16_from_f64(x):
    """BF16 RNE bits of a finite binary64 x, by frexp and round() (half to even). OverflowError past max."""
    if x == 0.0:
        return 0x8000 if math.copysign(1.0, x) < 0 else 0
    s = 0x8000 if x < 0 else 0
    a = abs(x)
    m, e = math.frexp(a)  # a = m * 2**e, 0.5 <= m < 1
    if e - 1 < -126:
        n = round(math.ldexp(a, 133))  # quantum 2**-133; scaling by a power of two is exact here
        if n == 0:
            return s
        if n >= 128:
            return s | (1 << 7) | (n - 128) if n < 256 else None
        return s | n
    n = round(math.ldexp(m, 8))  # 128 <= n <= 256
    if n == 256:
        n, e = 128, e + 1
    biased = e - 1 + 127
    if biased >= 255:
        raise OverflowError("bf16")
    return s | (biased << 7) | (n - 128)


def native_f32(x):
    try:
        return struct.unpack("<f", struct.pack("<f", x))[0]
    except OverflowError:
        raise


def native_run(case):
    """The f64 profiles, computed with CPython floats (binary64, separate * and +, no FMA)."""
    op = case["op"]
    prof = op.get("numerics")
    f64 = lambda h: struct.unpack("<d", int(h, 16).to_bytes(8, "little"))[0]
    bf = lambda h: struct.unpack("<f", (int(h, 16) << 16).to_bytes(4, "little"))[0]
    rows, cols = op["shape"]
    W = [bf(h) for h in case["base"]["bf16"]]
    out = [int(h, 16) for h in case["base"]["bf16"]]

    def conv(x):
        if not math.isfinite(x):
            raise Refuse("non_finite_intermediate")
        try:
            if prof == "f64-seq-rne2/v1":
                x = native_f32(x)
            return native_bf16_from_f64(x)
        except OverflowError:
            raise Refuse("overflow")

    def fin(x):
        if not math.isfinite(x):
            raise Refuse("non_finite_intermediate")
        return x

    try:
        if op["kind"] == "remove_direction":
            u, alpha = [f64(h) for h in op["u"]], f64(op["alpha"])
            for j in op["columns"]:
                c = 0.0
                for r in range(rows):
                    c = fin(c + fin(u[r] * W[r * cols + j]))
                for r in range(rows):
                    out[r * cols + j] = conv(fin(W[r * cols + j] - fin(fin(alpha * u[r]) * c)))
        else:
            K = op["rank"]
            a, b = [f64(h) for h in op["a"]], [f64(h) for h in op["b"]]
            for r in range(rows):
                for j in range(cols):
                    w = W[r * cols + j]
                    for k in range(K):
                        w = fin(w + fin(a[r * K + k] * b[k * cols + j]))
                    out[r * cols + j] = conv(w)
    except Refuse as e:
        return ("refuse", e.code)
    for r, c, h in op.get("fixups", []):
        out[r * cols + c] = int(h, 16)
    return ("bf16", out)


# ---------------------------------------------------------------- case builders

def hx(bits, fmt):
    return f"{bits:0{FMT[fmt][3] // 4}x}"


def bf16_list(values):
    return [hx(exact(v, "bf16"), "bf16") if not isinstance(v, str) else v for v in values]


def f64_of(x):
    return f"{struct.unpack('<Q', struct.pack('<d', x))[0]:016x}"


def wv(x, fmt):
    """A working-format operand from a Python float (rounded to fmt for f32, exact for f64)."""
    if fmt == "f64":
        return f64_of(x)
    return f"{struct.unpack('<I', struct.pack('<f', x))[0]:08x}"


def case_rd(cid, purpose, prof, base, shape, u, alpha, columns, fixups=(), **extra):
    work = PROFILES[prof]["work"] if prof in PROFILES else "f64"
    return {"id": cid, "purpose": purpose,
            "base": {"shape": list(shape), "bf16": bf16_list(base)},
            "op": {"kind": "remove_direction", "numerics": prof, "shape": list(shape),
                   "u": [x if isinstance(x, str) else wv(x, work) for x in u],
                   "alpha": alpha if isinstance(alpha, str) else wv(alpha, work),
                   "columns": list(columns), "fixups": [list(f) for f in fixups]}, **extra}


def case_lr(cid, purpose, prof, base, shape, rank, a, b, fixups=(), **extra):
    work = PROFILES[prof]["work"] if prof in PROFILES else "f64"
    return {"id": cid, "purpose": purpose,
            "base": {"shape": list(shape), "bf16": bf16_list(base)},
            "op": {"kind": "add_low_rank", "numerics": prof, "shape": list(shape), "rank": rank,
                   "a": [x if isinstance(x, str) else wv(x, work) for x in a],
                   "b": [x if isinstance(x, str) else wv(x, work) for x in b],
                   "fixups": [list(f) for f in fixups]}, **extra}


def case_zc(cid, purpose, base, shape, columns, fixups=(), **extra):
    op = {"kind": "zero_columns", "shape": list(shape), "columns": list(columns), "fixups": [list(f) for f in fixups]}
    op.update(extra.pop("op_extra", {}))
    return {"id": cid, "purpose": purpose, "base": {"shape": list(shape), "bf16": bf16_list(base)}, "op": op, **extra}


VARIANTS = {
    "fma_in_reduction": {"fma": True},
    "reversed_reduction": {"order": "reverse"},
    "pairwise_reduction": {"order": "pairwise"},
    "alpha_times_uc": {"assoc": "alpha_times_uc"},
    "f32_accumulation": {"accum_fmt": "f32"},
    "single_rounding": {"chain": ["bf16"]},
    "double_rounding": {"chain": ["f32", "bf16"]},
    "ftz_daz": {"ftz": True},
    "low_rank_from_zero": {"start_from_zero": True},
    "zero_keeps_sign": {"keep_sign": True},
    "bf16_output_flush": {"bf16_flush": True},
}


def random_bf16(rng, lo=-2.0, hi=2.0):
    while True:
        x = rng.uniform(lo, hi)
        b = struct.unpack("<I", struct.pack("<f", x))[0] >> 16
        if b & 0x7F80 not in (0, 0x7F80):
            return f"{b:04x}"


def bf16_value(h):
    return struct.unpack("<f", (int(h, 16) << 16).to_bytes(4, "little"))[0]


def search_rd(cid, purpose, prof, target, rows, rng, cancel=True, distinct=()):
    """Seeded search for a remove_direction case on which variant `target` changes at least one byte.
    alpha is chosen so one output lands within an ulp of a BF16 rounding midpoint, which makes a tiny change
    in the reduction visible."""
    work = PROFILES[prof]["work"]
    for attempt in range(MAX_SEARCH):
        base = [random_bf16(rng) for _ in range(rows)]
        W = [bf16_value(h) for h in base]
        u = [rng.uniform(-1, 1) for _ in range(rows)]
        if cancel:  # make the last term nearly cancel the others, so c is small and its rounding error large
            partial = sum(u[r] * W[r] for r in range(rows - 1))
            u[-1] = -partial / W[-1] * (1 + rng.uniform(-1e-6, 1e-6))
        if work == "f32":
            u = [native_f32(x) for x in u]
        c = 0.0
        for r in range(rows):
            c = c + u[r] * W[r]
        if c == 0.0:
            continue
        r0 = rng.randrange(rows)
        mid = W[r0] - rng.choice([1, -1]) * 2.0 ** -7 * (1.5 + rng.randrange(4))
        # a BF16 midpoint near `mid`: (n + 1/2) quanta at mid's exponent
        e = math.frexp(mid)[1]
        q = 2.0 ** (e - 8)
        lo = math.floor(mid / q)
        lo -= lo % 2  # the lower neighbour is even, so a tie rounds down and anything above it rounds up
        midpoint = (lo + 0.5) * q
        if PROFILES[prof]["chain"] == ["f32", "bf16"]:
            # aim half an f32 ulp above the BF16 tie: the f32 step then lands on the tie (rounds down) or one
            # f32 ulp above it (rounds up), so a tiny change before the f32 step still changes the output
            midpoint += 2.0 ** (math.frexp(midpoint)[1] - 24) / 2
        alpha = (W[r0] - midpoint) / (u[r0] * c)
        if not math.isfinite(alpha) or alpha == 0.0:
            continue
        if work == "f32":
            alpha = native_f32(alpha)
        case = case_rd(cid, purpose, prof, base, [rows, 1], u, alpha, [0],
                       construction={"method": "seeded search", "seed": SEARCH_SEED, "attempt": attempt})
        good = run(case)
        alt = run(case, VARIANTS[target])
        others = [run(case, VARIANTS[d]) for d in distinct]  # the target must differ from these too
        outs = [good, alt] + others
        if all(o[0] == "bf16" for o in outs) and good[1] != alt[1] and all(alt[1] != o[1] for o in others):
            return case
    raise SystemExit(f"{cid}: no discriminating case in {MAX_SEARCH} attempts")


def build_cases():
    rng = random.Random(SEARCH_SEED)
    R1, R2, F32 = "f64-seq-rne1/v1", "f64-seq-rne2/v1", "f32-seq-rne1/v1"
    cases = []
    T = lambda c, *targets: (c.setdefault("discriminates", list(targets)), cases.append(c))

    # 1-3, 5 of design §2: discriminated by search
    T(search_rd("rd-fma", "FMA in the reduction changes the output", R1, "fma_in_reduction", 2, rng), "fma_in_reduction")
    T(search_rd("rd-order-reverse", "reversed reduction order changes the output", R1, "reversed_reduction", 4, rng),
      "reversed_reduction")
    T(search_rd("rd-order-pairwise", "pairwise reduction differs from both the forward and the reversed order", R1,
                "pairwise_reduction", 4, rng, distinct=("reversed_reduction",)), "pairwise_reduction")
    T(search_rd("rd-assoc", "alpha*(u*c) instead of (alpha*u)*c changes the output", R1, "alpha_times_uc", 3, rng),
      "alpha_times_uc")
    T(search_rd("rd-f32-accum", "accumulating c in f32 changes the output (fault 5's golden fallback)", R1,
                "f32_accumulation", 3, rng), "f32_accumulation")
    T(search_rd("rd-rne2-fma", "FMA, under the double-rounding profile", R2, "fma_in_reduction", 2, rng),
      "fma_in_reduction")
    T(search_rd("rd-f32-order", "reversed order, under the f32 profile", F32, "reversed_reduction", 4, rng),
      "reversed_reduction")
    T(search_rd("rd-f32-assoc", "association, under the f32 profile", F32, "alpha_times_uc", 3, rng),
      "alpha_times_uc")

    # 4: BF16 ties, both parities (exact sums, so the only rounding is the final one)
    T(case_lr("lr-tie-even-down", "1 + 2^-8 is a BF16 tie; the even neighbour is 1.0 (0x3f80)", R1,
              [1.0], [1, 1], 1, [1.0], [2.0 ** -8]))
    T(case_lr("lr-tie-even-up", "1 + 3*2^-8 is a BF16 tie; the even neighbour is 1 + 2^-6 (0x3f82)", R1,
              [1.0], [1, 1], 1, [1.0], [3 * 2.0 ** -8]))
    T(case_lr("lr-tie-negative", "-(1 + 2^-8) ties to -1.0 (0xbf80)", R1,
              [-1.0], [1, 1], 1, [1.0], [-(2.0 ** -8)]))

    # 5: double rounding, each chain
    for prof, name in ((R1, "rne1"), (R2, "rne2"), (F32, "f32")):
        tgt = "double_rounding" if prof == R1 else "single_rounding"
        if prof == F32:
            T(case_lr(f"lr-double-{name}", "1 + 2^-8 + 2^-30: the f32 sum is already 1 + 2^-8, a tie -> 0x3f80", prof,
                      [1.0], [1, 1], 1, [1.0], [2.0 ** -8 + 2.0 ** -30]))
        else:
            T(case_lr(f"lr-double-{name}", "1 + 2^-8 + 2^-30: single RNE gives 0x3f81, f32 then BF16 gives 0x3f80",
                      prof, [1.0], [1, 1], 1, [1.0], [2.0 ** -8 + 2.0 ** -30]), tgt)
    T(case_lr("lr-double-negative-rne2", "-(1 + 2^-8 + 2^-30) under the double chain gives 0xbf80", R2,
              [-1.0], [1, 1], 1, [1.0], [-(2.0 ** -8 + 2.0 ** -30)]), "single_rounding")

    # the low-rank sum starts from the widened w, sequential in k
    T(case_lr("lr-from-w", "w + a0b0 + 3 x 2^-54, sequential from w: each 2^-54 is lost; from zero it is not", R1,
              [1.0], [1, 1], 4, [1.0, 1.0, 1.0, 1.0], [2.0 ** -8, 2.0 ** -54, 2.0 ** -54, 2.0 ** -54]),
      "low_rank_from_zero")

    # 6: signed zero
    T(case_lr("lr-negzero-plus-poszero", "(-0) + (+0) = +0", R1, ["8000"], [1, 1], 1, [1.0], [0.0]))
    T(case_lr("lr-negzero-plus-negzero", "(-0) + (-0) = -0, preserved as 0x8000", R1, ["8000"], [1, 1], 1, [1.0],
              [-0.0]))
    T(case_rd("rd-x-minus-x", "x - x = +0, from a negative base", R1, [-1.5], [1, 1], [1.0], 1.0, [0]))
    T(case_lr("lr-tiny-negative", "-2^-140 rounds to -0 in BF16 (0x8000)", R1, [0.0], [1, 1], 1, [1.0],
              [-(2.0 ** -140)]))
    T(case_lr("lr-tiny-negative-rne2", "-2^-140 is an f32 subnormal, then -0 in BF16", R2, [0.0], [1, 1], 1, [1.0],
              [-(2.0 ** -140)]))
    T(case_zc("zc-negatives", "zero_columns writes 0x0000 over negative, positive and -0 entries; other columns kept",
              [-1.5, 2.0, -0.25, "8000", 3.0, -7.0], [2, 3], [0, 2]), "zero_keeps_sign")

    # 7: subnormals
    T(case_rd("rd-f64-subnormal-c", "c = 2^-1030 is an f64 subnormal; (alpha*u1)*c = 2^-8 reaches the output", R1,
              [1.0, 0.0], [2, 1], [2.0 ** -1030, 2.0 ** 1000], 2.0 ** 22, [0]), "ftz_daz")
    T(case_lr("lr-f32-step-subnormal", "2^-130 + 2^-140 is an f32 subnormal at the chain's f32 step", R2,
              [0.0], [1, 1], 1, [1.0], [2.0 ** -130 + 2.0 ** -140]), "ftz_daz")
    T(case_lr("lr-bf16-subnormal", "3 x 2^-133 is a BF16 subnormal (0x0003)", R1, [0.0], [1, 1], 1, [1.0],
              [3 * 2.0 ** -133]), "bf16_output_flush")
    T(case_lr("lr-bf16-subnormal-tie", "2.5 x 2^-133 ties to 2 x 2^-133 (0x0002)", R1, [0.0], [1, 1], 1, [1.0],
              [2.5 * 2.0 ** -133]), "bf16_output_flush")
    T(case_lr("lr-subnormal-base-rne1", "a BF16-subnormal base (0x0003) widens exactly: + 2^-133 gives 0x0004", R1,
              ["0003"], [1, 1], 1, [1.0], [2.0 ** -133]))
    T(case_lr("lr-subnormal-base-f32", "a BF16-subnormal base is an f32 subnormal under the f32 profile (DAZ)", F32,
              ["0003"], [1, 1], 1, [1.0], [0.0]), "ftz_daz")
    T(case_lr("lr-smallest-normal", "2^-126 - 2^-134 rounds up to the smallest normal (0x0080)", R1, [0.0], [1, 1],
              1, [1.0], [2.0 ** -126 - 2.0 ** -134]))
    T(case_lr("lr-f32-profile-subnormal", "3 x 2^-133, an f32 subnormal operand, under the f32 profile", F32,
              [0.0], [1, 1], 1, [1.0], [3 * 2.0 ** -133]), "ftz_daz")

    # extremes that stay finite
    T(case_lr("lr-max-finite", "the largest BF16 (0x7f7f) plus a quarter ulp (2^118) stays 0x7f7f", R1,
              ["7f7f"], [1, 1], 1, [1.0], [2.0 ** 118]))

    # 8: refusals, one defect each
    nan64, inf64 = "7ff8000000000000", "7ff0000000000000"
    T(case_lr("ref-overflow-rne1", "max BF16 + 2^119 is the tie above the largest BF16; ties-to-even overflows", R1,
              ["7f7f"], [1, 1], 1, [1.0], [2.0 ** 119]))
    T(case_lr("ref-overflow-rne2", "max BF16 + 2^119: exact at the f32 step, overflows at the BF16 step", R2,
              ["7f7f"], [1, 1], 1, [1.0], [2.0 ** 119]))
    T(case_lr("ref-overflow-f32-step", "max BF16 + 2^126 is finite in f64, overflows at the f32 step", R2,
              ["7f7f"], [1, 1], 1, [1.0], [2.0 ** 126]))
    T(case_lr("ref-intermediate-f64", "a*b = 2^1000 * 2^1000 overflows f64", R1, [0.0], [1, 1], 1, [2.0 ** 1000],
              [2.0 ** 1000]))
    T(case_lr("ref-intermediate-f32", "under the f32 profile, 2^100 * 2^100 overflows f32", F32, [0.0], [1, 1], 1,
              [2.0 ** 100], [2.0 ** 100]))
    T(case_rd("ref-nan-u", "a NaN in u", R1, [1.0, 1.0], [2, 1], [nan64, 1.0], 1.0, [0]))
    T(case_rd("ref-inf-alpha", "an infinite alpha", R1, [1.0], [1, 1], [1.0], inf64, [0]))
    T(case_lr("ref-inf-b", "an infinite b", R1, [1.0], [1, 1], 1, [1.0], [inf64]))
    T(case_lr("ref-nan-fixup", "a fixup with NaN bits", R1, [1.0], [1, 1], 1, [1.0], [0.5], fixups=[(0, 0, "7fc0")]))
    T(case_lr("ref-inf-fixup", "a fixup with +inf bits", R1, [1.0], [1, 1], 1, [1.0], [0.5], fixups=[(0, 0, "7f80")]))

    T(case_lr("ref-overflow-f32-profile", "f32 profile: max BF16 + 2^119 is exact in f32, a tie that overflows BF16",
              F32, ["7f7f"], [1, 1], 1, [1.0], [2.0 ** 119]))
    T(case_rd("ref-rd-overflow", "remove_direction: max BF16 x (1 + 2^-8) is finite in f64, overflows BF16", R1,
              ["7f7f"], [1, 1], [1.0], -(2.0 ** -8), [0]))
    T(case_rd("ref-rd-intermediate", "remove_direction: u*w = 2^1000 * 2^100 overflows f64", R1, [2.0 ** 100], [1, 1],
              [2.0 ** 1000], 1.0, [0]))
    T(case_lr("ref-nan-a", "a NaN in a", R1, [1.0], [1, 1], 1, [nan64], [1.0]))
    T(case_rd("ref-nan-u-f32", "a NaN in u, as f32 bits, under the f32 profile", F32, [1.0], [1, 1], ["7fc00000"], 1.0,
              [0]))
    T(case_rd("ref-nan-base-touched", "a NaN base element in a touched column", R1, ["7fc0", 1.0], [2, 1],
              [1.0, 1.0], 1.0, [0]))
    T(case_lr("ref-inf-base-low-rank", "a -inf base element (add_low_rank touches every element)", R1, ["ff80"],
              [1, 1], 1, [1.0], [1.0]))
    T(case_rd("rd-inf-base-untouched", "a +inf base element outside S is kept, bit for bit", R1, ["7f80", 1.0],
              [1, 2], [1.0], 0.5, [1]))
    T(case_zc("zc-nan-base", "zero_columns overwrites a NaN base element with 0x0000; it reads no value", ["7fc0", 2.0],
              [1, 2], [0]))

    # fixups
    T(case_rd("rd-fixup", "a fixup overrides one computed element", R1, [1.0, 2.0, 3.0, 4.0], [2, 2],
              [0.5, 0.25], 1.0, [1], fixups=[(1, 1, "4000")]))
    T(case_lr("ref-fixup-no-op-rne2", "under the double chain the computed value is 0x3f80, so this fixup is a no-op",
              R2, [1.0], [1, 1], 1, [1.0], [2.0 ** -8 + 2.0 ** -30], fixups=[(0, 0, "3f80")]))
    T(case_lr("lr-fixup-rne1", "the same fixup under the single chain is a real correction (0x3f81 -> 0x3f80)", R1,
              [1.0], [1, 1], 1, [1.0], [2.0 ** -8 + 2.0 ** -30], fixups=[(0, 0, "3f80")]))
    T(case_rd("ref-fixup-no-op-rd", "remove_direction: a fixup equal to the computed 0x4060 is a no-op", R1,
              [1.0, 2.0, 3.0, 4.0], [2, 2], [0.5, 0.25], 1.0, [1], fixups=[(1, 1, "4060")]))
    T(case_rd("ref-fixup-outside", "a fixup outside the touched columns", R1, [1.0, 2.0, 3.0, 4.0], [2, 2],
              [0.5, 0.25], 1.0, [1], fixups=[(0, 0, "4000")]))
    T(case_rd("ref-fixup-range", "a fixup beyond the tensor", R1, [1.0, 2.0, 3.0, 4.0], [2, 2], [0.5, 0.25], 1.0, [1],
              fixups=[(2, 1, "4000")]))
    T(case_rd("ref-fixup-unsorted", "fixups not sorted by (row, col)", R1, [1.0, 2.0, 3.0, 4.0], [2, 2], [0.5, 0.25],
              1.0, [0, 1], fixups=[(1, 0, "4000"), (0, 1, "4000")]))
    T(case_rd("ref-fixup-duplicate", "the same element fixed twice", R1, [1.0, 2.0, 3.0, 4.0], [2, 2], [0.5, 0.25],
              1.0, [0, 1], fixups=[(0, 1, "4000"), (0, 1, "4040")]))

    # malformed ops
    T(case_rd("ref-profile-unknown", "an unknown numerics profile", "f64-seq-rne3/v1", [1.0], [1, 1], [1.0], 1.0, [0]))
    c = case_rd("ref-profile-missing", "no numerics field", R1, [1.0], [1, 1], [1.0], 1.0, [0])
    del c["op"]["numerics"]
    T(c)
    T(case_rd("ref-columns-empty", "an empty column set", R1, [1.0, 2.0], [1, 2], [1.0], 1.0, []))
    T(case_rd("ref-columns-unsorted", "columns not strictly increasing", R1, [1.0, 2.0], [1, 2], [1.0], 1.0, [1, 0]))
    T(case_rd("ref-columns-duplicate", "a repeated column", R1, [1.0, 2.0], [1, 2], [1.0], 1.0, [1, 1]))
    T(case_rd("ref-columns-range", "a column beyond the tensor", R1, [1.0, 2.0], [1, 2], [1.0], 1.0, [2]))
    T(case_rd("ref-u-length", "u shorter than the row count", R1, [1.0, 2.0], [2, 1], [1.0], 1.0, [0]))
    c = case_rd("ref-shape-mismatch", "the declared shape differs from the base tensor's", R1, [1.0, 2.0], [2, 1],
                [1.0, 1.0], 1.0, [0])
    c["base"]["shape"] = [1, 2]
    T(c)
    T(case_lr("ref-rank-zero", "rank 0", R1, [1.0], [1, 1], 0, [], []))
    T(case_lr("ref-ab-size", "b is not rank x cols", R1, [1.0, 2.0], [1, 2], 1, [1.0], [1.0]))
    T(case_zc("ref-zc-fixup", "zero_columns carrying a fixup", [1.0, 2.0], [1, 2], [0], fixups=[(0, 0, "3f80")]))
    T(case_zc("ref-zc-numerics", "zero_columns carrying a numerics profile", [1.0, 2.0], [1, 2], [0],
              op_extra={"numerics": R1}))
    T(case_rd("ref-embed-tokens", "an op on the token embedding (it doubles as the tied head)", R1, [1.0], [1, 1],
              [1.0], 1.0, [0], tensor_role="embed_tokens"))

    # general cases: many rows and columns, every profile
    for prof, name in ((R1, "rne1"), (R2, "rne2"), (F32, "f32")):
        rows, cols = 16, 6
        base = [random_bf16(rng) for _ in range(rows * cols)]
        u = [rng.uniform(-0.3, 0.3) for _ in range(rows)]
        T(case_rd(f"rd-general-{name}", "16 x 6, three columns, random operands", prof, base, [rows, cols], u, 6.0,
                  [0, 2, 5], construction={"method": "seeded random", "seed": SEARCH_SEED}))
        a = [rng.uniform(-0.05, 0.05) for _ in range(rows * 2)]
        b = [rng.uniform(-0.05, 0.05) for _ in range(2 * cols)]
        T(case_lr(f"lr-general-{name}", "16 x 6, rank 2, random operands", prof, base, [rows, cols], 2, a, b,
                  construction={"method": "seeded random", "seed": SEARCH_SEED}))
    return cases


def sha_bf16(hexes):
    return hashlib.sha256(b"".join(int(h, 16).to_bytes(2, "little") for h in hexes)).hexdigest()


def finish(cases):
    ids = set()
    for c in cases:
        assert c["id"] not in ids, c["id"]
        ids.add(c["id"])
        kind, val = run(c)
        c["base"]["sha256"] = sha_bf16(c["base"]["bf16"])
        c["expect"] = {"bf16": [hx(x, "bf16") for x in val], "sha256": sha_bf16([hx(x, "bf16") for x in val])} \
            if kind == "bf16" else {"refuse": val}
        alts = {}
        for name, var in VARIANTS.items():
            k2, v2 = run(c, var)
            r = {"bf16": [hx(x, "bf16") for x in v2]} if k2 == "bf16" else {"refuse": v2}
            if r != {k: v for k, v in c["expect"].items() if k != "sha256"}:
                alts[name] = r
        disc = c.get("discriminates", [])
        for d in disc:
            assert d in alts, f"{c['id']}: variant {d} does not change the result"
        c["under_wrong_semantics"] = alts  # every named variant that changes the result, targeted or not
        c.setdefault("discriminates", [])
    return cases


def cross_check(cases):
    """The independent native route must agree with the exact route on every f64-profile arithmetic case."""
    n = 0
    for c in cases:
        op = c["op"]
        if op["kind"] == "zero_columns" or op.get("numerics") not in ("f64-seq-rne1/v1", "f64-seq-rne2/v1"):
            continue
        exp = c["expect"]
        if "refuse" in exp and exp["refuse"] not in ("overflow", "non_finite_intermediate"):
            continue
        if exp.get("refuse") == "fixup_no_op" or c.get("tensor_role"):
            continue
        got = native_run(c)
        want = ("refuse", exp["refuse"]) if "refuse" in exp else ("bf16", [int(h, 16) for h in exp["bf16"]])
        assert got == want, f"{c['id']}: native route {got} != exact route {want}"
        n += 1
    return n


def document():
    cases = finish(build_cases())
    n = cross_check(cases)
    return {"schema": SCHEMA, "spec": "SPEC.md", "generator": "generate.py",
            "profiles": PROFILES, "refusals": REFUSALS,
            "variants": {k: v for k, v in VARIANTS.items()},
            "native_cross_checked_cases": n, "cases": cases}


def dumps(doc):
    return json.dumps(doc, indent=1, sort_keys=False) + "\n"


if __name__ == "__main__":
    text = dumps(document())
    if "--check" in sys.argv[1:]:
        with open(OUT) as f:
            if f.read() != text:
                raise SystemExit("vectors.json differs from what generate.py produces")
        print("vectors.json reproduces;", json.loads(text)["native_cross_checked_cases"], "cases cross-checked")
    else:
        with open(OUT, "w") as f:
            f.write(text)
        print("wrote", OUT, hashlib.sha256(text.encode()).hexdigest())
