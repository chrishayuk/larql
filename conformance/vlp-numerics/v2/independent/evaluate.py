# Independent evaluator for VLP numerics v2 op `add_low_rank_columns`.
# Documents used (and nothing else):
#   conformance/vlp-numerics/v2/SPEC.md   and   conformance/vlp-numerics/v1/SPEC.md
# Python 3 standard library only; exact rational arithmetic (fractions.Fraction).
# Usage: python3 -I evaluate.py <inputs.json> <results.json>
import json
import sys
from fractions import Fraction

# format: (precision p, emin, emax, total bits, exponent bits)
F64 = (53, -1022, 1023, 64, 11)
F32 = (24, -126, 127, 32, 8)
BF16 = (8, -126, 127, 16, 8)
PROFILES = {
    "f64-seq-rne1/v1": (F64, "rne1"),
    "f64-seq-rne2/v1": (F64, "rne2"),
    "f32-seq-rne1/v1": (F32, "rne1"),
}


class Refuse(Exception):
    def __init__(self, code):
        self.code = code


# A value is (q, neg): q an exact Fraction, neg meaningful for sign of zero
# (for nonzero q neg == q < 0).
def decode(bits, fmt):
    p, emin, emax, nbits, ebits = fmt
    mbits = p - 1
    sign = (bits >> (nbits - 1)) & 1
    exp = (bits >> mbits) & ((1 << ebits) - 1)
    man = bits & ((1 << mbits) - 1)
    if exp == (1 << ebits) - 1:
        return None  # NaN or inf
    if exp == 0:
        q = Fraction(man) * Fraction(2) ** (emin - mbits)
    else:
        q = Fraction((1 << mbits) + man) * Fraction(2) ** (exp - (emax) - mbits)
    if sign:
        q = -q
    return (q, bool(sign))


def ilog2_floor(m):
    # floor(log2(m)) for positive Fraction m
    n, d = m.numerator, m.denominator
    e = n.bit_length() - d.bit_length()
    # 2^e <= m < 2^(e+1) up to one correction
    if e >= 0:
        if n < (d << e):
            e -= 1
    else:
        if (n << -e) < d:
            e -= 1
    return e


def rnd(x, fmt):
    """Round exact nonzero Fraction x to fmt. Returns (value Fraction, neg).
    Raises Refuse('overflow') on overflow; zero result keeps sign of x."""
    p, emin, emax, _, _ = fmt
    neg = x < 0
    m = -x if neg else x
    e = ilog2_floor(m)
    if e < emin:
        e = emin
    quantum = Fraction(2) ** (e - p + 1)
    n = m / quantum
    fl = n.numerator // n.denominator
    rem = n - fl
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and (fl & 1)):
        fl += 1
    r = fl * quantum
    if r >= Fraction(2) ** (emax + 1):
        raise Refuse("overflow")
    return ((-r if neg else r), neg)


def mul(a, b, W):
    (qa, sa), (qb, sb) = a, b
    if qa == 0 or qb == 0:
        return (Fraction(0), sa != sb)
    try:
        return rnd(qa * qb, W)
    except Refuse:
        raise Refuse("non_finite_intermediate")


def add(a, b, W):
    (qa, sa), (qb, sb) = a, b
    if qa == 0 and qb == 0:
        return (Fraction(0), sa and sb)
    s = qa + qb
    if s == 0:
        return (Fraction(0), False)
    try:
        return rnd(s, W)
    except Refuse:
        raise Refuse("non_finite_intermediate")


def chain(w, W, kind):
    q, neg = w
    if q == 0:
        return w
    if kind == "rne2":
        w = rnd(q, F32)  # raises Refuse('overflow')
        q, neg = w
        if q == 0:
            return w
    return rnd(q, BF16)


def encode_bf16(v):
    q, neg = v
    p, emin, emax, _, _ = BF16
    sign = 0x8000 if neg else 0
    if q == 0:
        return sign
    m = abs(q)
    e = ilog2_floor(m)
    if e < emin:
        man = m / Fraction(2) ** (emin - 7)
        assert man.denominator == 1 and man < 128
        return sign | int(man)
    man = m / Fraction(2) ** (e - 7) - 128
    assert man.denominator == 1 and 0 <= man < 128
    return sign | ((e + 127) << 7) | int(man)


def is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def parse_hex(s, ndigits):
    if not isinstance(s, str) or len(s) != ndigits:
        raise Refuse("non_finite_operand")  # GUESS: malformed encoding (unspecified)
    try:
        return int(s, 16)
    except ValueError:
        raise Refuse("non_finite_operand")


def evaluate(case):
    base = case["base"]
    op = case["op"]
    # 1. numerics_invalid
    prof = op.get("numerics")
    if prof not in PROFILES:
        raise Refuse("numerics_invalid")
    W, ckind = PROFILES[prof]
    ndig = W[3] // 4
    # 2. shape_invalid (declared)
    shape = op.get("shape")
    if not (isinstance(shape, list) and len(shape) == 2 and all(is_int(s) and s > 0 for s in shape)):
        raise Refuse("shape_invalid")
    rows, cols = shape
    # 3. fixups order / duplicates
    fixups = op.get("fixups", [])
    prev = None
    for f in fixups:
        key = (f[0], f[1])
        if prev is not None and key <= prev:  # equal => duplicate, smaller => unordered
            raise Refuse("fixups_invalid")
        prev = key
    # 4. fixup bits non-finite (BF16 bits)
    fbits = []
    for f in fixups:
        v = int(f[2], 16)
        if decode(v, BF16) is None:
            raise Refuse("non_finite_operand")
        fbits.append(v)
    # 5. columns
    J = op.get("columns")
    if not isinstance(J, list) or len(J) == 0:
        raise Refuse("columns_invalid")
    for i in range(len(J)):
        if not is_int(J[i]) or J[i] < 0 or (i > 0 and J[i] <= J[i - 1]):
            raise Refuse("columns_invalid")
    n = len(J)
    # 6. rank
    K = op.get("rank")
    if not is_int(K) or K < 1:
        raise Refuse("rank_invalid")
    # 7. operand lengths
    A, B = op["a"], op["b"]
    if len(A) != rows * K or len(B) != K * n:
        raise Refuse("shape_invalid")
    # 8. operand finiteness
    a = [decode(parse_hex(s, ndig), W) for s in A]
    b = [decode(parse_hex(s, ndig), W) for s in B]
    if any(v is None for v in a) or any(v is None for v in b):
        raise Refuse("non_finite_operand")
    # header
    if list(base["shape"]) != [rows, cols] or len(base["bf16"]) != rows * cols:
        raise Refuse("shape_invalid")
    if case.get("tensor_role") == "embed_tokens":
        raise Refuse("tensor_refused")
    if J[-1] >= cols:
        raise Refuse("columns_invalid")
    Jset = set(J)
    for f in fixups:
        if not (is_int(f[0]) and is_int(f[1])) or not (0 <= f[0] < rows and 0 <= f[1] < cols) or f[1] not in Jset:
            raise Refuse("fixups_invalid")
    # non-finite base inside rows x J
    out = [int(h, 16) for h in base["bf16"]]
    for r in range(rows):
        for j in J:
            if decode(out[r * cols + j], BF16) is None:
                raise Refuse("non_finite_operand")
    # compute, row-major over (r, p), then k
    for r in range(rows):
        for p, j in enumerate(J):
            written = any(a[r * K + k][0] != 0 and b[k * n + p][0] != 0 for k in range(K))
            if not written:
                continue
            w = decode(out[r * cols + j], BF16)
            for k in range(K):
                w = add(w, mul(a[r * K + k], b[k * n + p], W), W)
            out[r * cols + j] = encode_bf16(chain(w, W, ckind))
    # fixups
    for f, bits in zip(fixups, fbits):
        idx = f[0] * cols + f[1]
        if out[idx] == bits:
            raise Refuse("fixup_no_op")
    for f, bits in zip(fixups, fbits):
        out[f[0] * cols + f[1]] = bits
    return ["%04x" % v for v in out]


def main():
    inp = json.load(open(sys.argv[1]))
    res = {}
    for c in inp["cases"]:
        try:
            res[c["id"]] = {"bf16": evaluate(c)}
        except Refuse as e:
            res[c["id"]] = {"refuse": e.code}
    with open(sys.argv[2], "w") as fh:
        json.dump(res, fh, indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
