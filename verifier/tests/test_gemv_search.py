"""A search for a counterexample to B1 (Q8_0-GEMV-SPEC.md revision 1), using section 3's own program for y.

Four regions where the derivation in section 5 is thinnest:
  R1  products near the subnormal boundary (around 2^-126);
  R2  many terms in the underflow range, including exact ties that all round the same way;
  R3  large cancellation (S far above |y*|), with near-duplicate terms of opposite sign;
  R4  the top of the domain, S just inside (1 + gamma_k)(S + k eta) < Omega.
Each region starts from seeded random inputs and then hill-climbs on the measured ratio |y - y*| / B with single-edit
mutations (an x mantissa bit, an x neighbour, a code, a scale bit, a duplicated x). Not finding a violation does not
establish the bound: it is a finite search. A violation inside the domain would be a counterexample: it is
printed with the exact input bits, the test fails, and per section 6 nothing else is changed.

Size: GEMV_SEARCH_SCALE (default 1) multiplies the number of starting points.
"""

import os
import struct
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _gemv_gen as gen  # noqa: E402
import _gemv_oracle as orc  # noqa: E402
from refcheck import gemv  # noqa: E402

SCALE = int(os.environ.get("GEMV_SEARCH_SCALE", "1"))
STARTS = 24 * SCALE
STEPS = 20
STATS = {}
COUNTEREXAMPLES = []


def stat(region):
    return STATS.setdefault(region, {"cases": 0, "in_domain": 0, "outside": 0, "max_ratio": 0.0, "violated": 0, "edge_inside": 0})


def f32_from_float(v):
    try:
        return struct.unpack("<I", struct.pack("<f", v))[0]
    except OverflowError:
        return None


def evaluate(region, raw, x, n, k):
    """seq_bits -> check_b1; update statistics; return the ratio (float) or None outside the domain."""
    st = stat(region)
    st["cases"] += 1
    y = gemv.seq_bits(raw, x, n, k)
    r = gemv.check_b1(raw, x, y, n, k)
    if r.status == "could_not_evaluate":
        st["outside"] += 1
        return None
    st["in_domain"] += 1
    if r.status == "violated":
        st["violated"] += 1
        # confirm with the independent oracle before reporting it as a counterexample candidate
        oracle = orc.b1_oracle(raw, x, [int(v) for v in y], n, k) if k <= 256 else ("not re-run", None, None, [])
        COUNTEREXAMPLES.append(
            {"region": region, "n": n, "k": k, "blocks_hex": raw.hex(), "x_bits": [hex(v) for v in x],
             "y_bits": [hex(int(v)) for v in y], "check_b1": r, "oracle_status": oracle[0]}
        )
        return float("inf")
    ratio = float(r.worst_ratio)
    st["max_ratio"] = max(st["max_ratio"], ratio)
    return ratio


def mutate(rng, raw, x, k, n):
    raw = bytearray(raw)
    x = list(x)
    op = int(rng.integers(0, 6))
    j = int(rng.integers(0, k))
    if op == 0:
        nx = x[j] ^ (1 << int(rng.integers(0, 23)))
    elif op == 1:
        nx = (x[j] + int(rng.choice([-1, 1]))) & 0xFFFFFFFF
    elif op == 4:
        nx = x[int(rng.integers(0, k))]
    elif op == 5:
        nx = x[j] ^ 0x80000000
    else:
        nx = None
    if nx is not None:
        if (nx >> 23) & 255 != 255:
            x[j] = nx
    elif op == 2:
        i = int(rng.integers(0, n)) * (k // 32) + j // 32
        raw[i * 34 + 2 + (j % 32)] = int(rng.integers(0, 256))
    else:  # op == 3: one bit of a scale
        i = int(rng.integers(0, n * (k // 32)))
        d = raw[i * 34] | (raw[i * 34 + 1] << 8)
        d2 = d ^ (1 << int(rng.integers(0, 10)))
        if (d2 >> 10) & 31 != 31:
            raw[i * 34] = d2 & 255
            raw[i * 34 + 1] = d2 >> 8
    return bytes(raw), x


def climb(region, rng, case):
    raw, x, n, k = case
    best = evaluate(region, raw, x, n, k)
    for _ in range(STEPS):
        r2, x2 = mutate(rng, raw, x, k, n)
        v = evaluate(region, r2, x2, n, k)
        if v is not None and (best is None or v >= best):
            raw, x, best = r2, x2, v
        if v == float("inf"):
            return


# ---------------------------------------------------------------- region generators


def gen_r1(rng):
    """Products straddling 2^-126: pick w, then x so that |w x| ~ 2^t with t in [-140, -110]."""
    k = int(rng.choice([32, 64, 128]))
    n = 2
    raw = gen.make_case(rng, n, k, str(rng.choice(["typical", "random_finite", "min_normal", "subnormal"])), "normal", "random")[0]
    rows = orc.parse(raw, n, k)
    x = []
    for j in range(k):
        q, d = rows[0][j]
        w = abs(float(orc.f16_value(d) or 0) * q) or 1.0
        t = rng.uniform(-140, -110)
        v = float(rng.uniform(1, 2)) * 2.0**t / w
        b = f32_from_float(v)
        x.append((b if b is not None else 0) | (int(rng.integers(0, 2)) << 31))
    return raw, x, n, k


def gen_r2(rng):
    """Many terms in the underflow range: small scales (0.5, 0.25, 2^-24, ...), odd codes, small integer multiples of 2^-149."""
    k = int(rng.choice([256, 1024, 3072]))
    n = 1
    if rng.random() < 0.5:
        # aligned ties: d = 0.5, q = 1, x = +/- m * 2^-149 with odd m, so every product m/2 (in units of 2^-149) is a tie.
        # Ties-to-even moves m/2 by -1/2 for m = 1 mod 4 and +1/2 for m = 3 mod 4; the sign of each term is chosen so that
        # every rounding error has the same sign, making the total error k eta (the most the model allows from products).
        m = rng.choice([1, 3, 5, 7, 9, 11, 15, 17], k)
        sign = np.where(m % 4 == 1, 1, 0).astype(np.uint64) << 31
        raw = b"".join(struct.pack("<H", 0x3800) + bytes([1] * 32) for _ in range(k // 32))
        return raw, [int(a) | int(b) for a, b in zip(m, sign)], n, k
    d = int(rng.choice([0x3800, 0x3400, 0x3C00, 0x0001, 0x0200, 0x1000]))
    raw = bytearray()
    for _ in range(k // 32):
        raw += struct.pack("<H", d) + bytes(int(v) & 0xFF for v in rng.choice([1, 3, 5, -1, -3, 7, 127, -127], 32))
    m = rng.choice([1, 3, 5, 7, 1 << 10, 1 << 20], k)
    sign = rng.integers(0, 2, k) << 31
    x = [int(a) | int(b) for a, b in zip(m, sign)]
    return bytes(raw), x, n, k


def gen_r3(rng):
    """Cancelling pairs: (q, x) and (-q, x +/- a few ulps), columns shuffled; one scale for the whole row."""
    k = int(rng.choice([64, 256, 1024]))
    n = 2
    sc = int(np.float16(rng.standard_normal() * 0.05 + 0.01).view(np.uint16))
    xs, qs = [], []
    lo, hi = int(rng.integers(60, 120)), int(rng.integers(120, 190))
    for _ in range(k // 2):
        e = int(rng.integers(lo, hi))
        xb = (e << 23) | int(rng.integers(0, 1 << 23)) | (int(rng.integers(0, 2)) << 31)
        q = int(rng.integers(-127, 128)) or 5
        xs += [xb, xb + int(rng.choice([-2, -1, 0, 1, 2]))]
        qs += [q, -q]
    xs = [v & 0xFFFFFFFF for v in xs]
    for i, v in enumerate(xs):
        if (v >> 23) & 255 == 255:
            xs[i] = v & ~(1 << 23)
    perm = rng.permutation(k)
    xs = [xs[i] for i in perm]
    qs = [qs[i] for i in perm]
    raw = bytearray()
    for _ in range(n):
        for b in range(k // 32):
            raw += struct.pack("<H", sc) + bytes(q & 0xFF for q in qs[32 * b : 32 * b + 32])
    return bytes(raw), xs, n, k


def gen_r4(rng):
    """All-positive large terms, x scaled by bisection so S is as close as possible inside the domain limit."""
    k = int(rng.choice([32, 64]))
    n = 1
    d = int(rng.choice([0x7BFF, 0x3C00, 0x5800]))
    q = int(rng.choice([127, 100, 1]))
    raw = b"".join(struct.pack("<H", d) + bytes([q] * 32) for _ in range(k // 32))
    base = [1.0 + float(rng.random()) for _ in range(k)]

    def xs_for(f):
        out = []
        for v in base:
            b = f32_from_float(v * f)
            if b is None:
                return None
            out.append(b)
        return out

    def inside(f):
        xx = xs_for(f)
        if xx is None:
            return False
        ystar, S, bad, bx = gemv.exact_rows(raw, np.array(xx, dtype=np.uint32), 1, k)
        return not bad and not bx and gemv._in_domain_third(S[0], k)

    w = abs(float(orc.f16_value(d))) * q
    lo, hi = 2.0**100 / (w * k), 2.0**130 / (w * k)
    if not inside(lo) or inside(hi):
        return None
    for _ in range(60):
        mid = (lo + hi) / 2
        if inside(mid):
            lo = mid
        else:
            hi = mid
    stat("R4 top of the domain")["edge_inside"] += 1
    return raw, xs_for(lo), n, k


REGIONS = [
    ("R1 near the subnormal boundary", gen_r1),
    ("R2 many underflow-range terms", gen_r2),
    ("R3 large cancellation", gen_r3),
    ("R4 top of the domain", gen_r4),
]


class Search(unittest.TestCase):
    def test_deliberate_tight_input_is_inside_and_below_one(self):
        """d = 0.5, q = 1, x = -2^-149 in every column: error exactly k eta, ratio 1 / (1 + 2 gamma_k) < 1."""
        for k in (32, 1024, 3072):
            raw = b"".join(struct.pack("<H", 0x3800) + bytes([1] * 32) for _ in range(k // 32))
            x = [0x80000001] * k
            r = gemv.check_b1(raw, x, gemv.seq_bits(raw, x, 1, k), 1, k)
            self.assertEqual(r.status, "satisfied")
            g = k / (2**24 - k)
            self.assertAlmostEqual(float(r.worst_ratio), 1 / (1 + 2 * g), places=9)

    def test_top_of_domain_edge(self):
        """Just past the edge the answer is could_not_evaluate; at the edge it is a number."""
        rng = np.random.default_rng(404)
        found = 0
        for _ in range(10):
            case = gen_r4(rng)
            if case is None:
                continue
            raw, x, n, k = case
            found += 1
            self.assertIsNotNone(evaluate("R4 top of the domain", raw, x, n, k))
            # x scaled up by 0.2% is outside the domain (S is within 1 part in 2**50 of the limit, a few ulps)
            xo = [f32_from_float(struct.unpack("<f", struct.pack("<I", b))[0] * 1.002) for b in x]
            r = gemv.check_b1(raw, xo, [0], n, k)
            self.assertEqual(r.status, "could_not_evaluate")
        self.assertGreater(found, 5)

    def test_search_for_a_counterexample(self):
        rng = np.random.default_rng(20261103)
        for region, fn in REGIONS:
            for _ in range(STARTS):
                case = fn(rng)
                if case is None:
                    continue
                climb(region, rng, case)
        if COUNTEREXAMPLES:
            self.fail("COUNTEREXAMPLE CANDIDATE (section 6: record it, change nothing): " + repr(COUNTEREXAMPLES[0]))
        total = sum(s["in_domain"] for s in STATS.values())
        self.assertGreater(total, 100)


def tearDownModule():
    print("\n\nB1 counterexample search (section 3's program as y); a finite search, not evidence of the bound:")
    print(f"{'region':<36} {'cases':>8} {'in domain':>10} {'outside':>8} {'violated':>9} {'max |y-y*|/B':>14}")
    tot = 0
    for region, _ in REGIONS:
        s = STATS.get(region)
        if s:
            tot += s["cases"]
            print(f"{region:<36} {s['cases']:>8} {s['in_domain']:>10} {s['outside']:>8} {s['violated']:>9} {s['max_ratio']:>14.6g}")
    print(f"total cases evaluated (each: seq_bits then check_b1): {tot}")


if __name__ == "__main__":
    # `python -I tests/test_gemv_search.py 40` runs the search 40 times larger (same as GEMV_SEARCH_SCALE=40)
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        SCALE = int(sys.argv[1])
        STARTS = 24 * SCALE
        sys.argv = sys.argv[:1]
    unittest.main()
