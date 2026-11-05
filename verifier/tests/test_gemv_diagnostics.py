"""Numerical diagnostics for the Q8_0 GEMV (Q8_0-GEMV-SPEC.md revision 1): section 3's program against exact arithmetic.

For each input family and k in (32, 1024, 3072), a few rows each, seeded, this reports the worst |y - y*| / B of the
program's own output. These are measurements on synthetic data; a ratio below 1 on these inputs says nothing about inputs
that were not tried. The table is printed at the end of the module. The only assertions are that the program's result
is not flagged by B1 (a flag would be a counterexample: record it under section 6 and stop, do not edit anything).
"""

import os
import struct
import sys
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _gemv_gen as gen  # noqa: E402
import _gemv_oracle as orc  # noqa: E402
from refcheck import gemv  # noqa: E402

KS = (32, 1024, 3072)
N_ROWS = 3
TABLE = {}  # (family, k) -> (worst ratio str, row, status)


def _f32_exp(rng, k, lo, hi, positive=False):
    sign = np.zeros(k, dtype=np.uint32) if positive else rng.integers(0, 2, k).astype(np.uint32) << 31
    e = rng.integers(lo, hi, k)
    m = rng.integers(0, 1 << 23, k)
    return [int(s) | (int(a) << 23) | int(b) for s, a, b in zip(sign, e, m)]


def _build(rng, n, k, scale_fn, q_fn):
    nb = k // 32
    raw = bytearray()
    for _ in range(n * nb):
        raw += struct.pack("<H", scale_fn(rng)) + bytes(c & 0xFF for c in q_fn(rng))
    return bytes(raw)


def _typical_scale(rng):
    return int(np.float16(rng.standard_normal() * 0.05).view(np.uint16))


def _pos_scale(rng):
    return int(np.float16(abs(rng.standard_normal()) * 0.05 + 1e-3).view(np.uint16))


def _rand_q(rng):
    return [int(v) for v in rng.integers(-128, 128, 32)]


def _pos_q(rng):
    return [int(v) for v in rng.integers(1, 128, 32)]


def fam_typical(rng, n, k):
    return _build(rng, n, k, _typical_scale, _rand_q), gen.xs(rng, k, "normal")


def fam_all_positive(rng, n, k):
    return _build(rng, n, k, _pos_scale, _pos_q), _f32_exp(rng, k, 120, 134, positive=True)


def fam_cancellation(rng, n, k):
    """Second half of each row repeats the first half's x with the codes negated: y* = 0 exactly."""
    half = k // 2
    xs = gen.xs(rng, half, "normal")
    sc = _typical_scale(rng)
    raw = bytearray()
    for _ in range(n):
        q = [int(v) for v in rng.integers(-127, 128, half)]
        row_q = q + [-v for v in q]
        for b in range(k // 32):
            raw += struct.pack("<H", sc) + bytes(c & 0xFF for c in row_q[32 * b : 32 * b + 32])
    return bytes(raw), xs + xs


def fam_wide_exponent(rng, n, k):
    return _build(rng, n, k, _typical_scale, _rand_q), _f32_exp(rng, k, 127 - 60, 127 + 60)


def fam_subnormal_x(rng, n, k):
    return _build(rng, n, k, _typical_scale, _rand_q), gen.xs(rng, k, "subnormal")


def fam_underflow_products(rng, n, k):
    """Small scales and tiny x: most products land in or below the subnormal range."""
    sc = lambda r: int(r.integers(1, 0x0400))  # binary16 subnormals  # noqa: E731
    return _build(rng, n, k, sc, _rand_q), _f32_exp(rng, k, 0, 40)


def fam_extreme_scales(rng, n, k):
    pick = lambda r: int(r.choice([0x7BFF, 0xFBFF, 0x0400, 0x0001, 0x3C00]))  # noqa: E731
    q = lambda r: [int(v) for v in r.choice([-128, 127], 32)]  # noqa: E731
    return _build(rng, n, k, pick, q), _f32_exp(rng, k, 127 - 20, 127 + 20)


def fam_absorption(rng, n, k):
    """One large term first, then many tiny ones that are absorbed or half-absorbed."""
    xs = _f32_exp(rng, k, 127 - 30, 127 - 20, positive=True)
    xs[0] = 0x3F800000 | (30 << 23)  # 2^30
    sc = lambda r: 0x3C00  # noqa: E731
    q = lambda r: [int(v) for v in r.integers(1, 128, 32)]  # noqa: E731
    return _build(rng, n, k, sc, q), xs


def fam_ties_in_underflow(rng, n, k):
    """d = 0.5, q = 1, x = -2^-149 everywhere: every product is a tie that rounds to even (zero): the error is k eta."""
    raw = bytes(b for _ in range(n * (k // 32)) for b in struct.pack("<H", 0x3800) + bytes([1] * 32))
    return raw, [0x80000001] * k


def fam_near_omega(rng, n, k):
    """All-positive terms scaled so S is close to (not past) the domain limit."""
    raw = _build(rng, n, k, lambda r: 0x7BFF, lambda r: [127] * 32)
    base = _f32_exp(rng, k, 127, 128, positive=True)  # [1, 2)
    lo, hi = 1.0 / 2**24, 1.0
    # w = 127 * 65504; row sum S = w * sum(x). Find the largest power-of-two-free factor f with S in the domain.
    w = 127 * 65504
    s1 = sum(struct.unpack("<f", struct.pack("<I", b))[0] for b in base)
    limit = float(orc.OMEGA) * (1 - k / 2**24)
    f = 0.999 * limit / (w * s1)
    x = [orc.f32_bits_of_float(struct.unpack("<f", struct.pack("<I", b))[0] * f) for b in base]
    return raw, x


FAMILIES = [
    ("typical (N(0,1) x, scales ~0.05, random codes)", fam_typical),
    ("all positive (no cancellation)", fam_all_positive),
    ("exact cancellation (y* = 0)", fam_cancellation),
    ("wide exponents (x in 2^-60..2^60)", fam_wide_exponent),
    ("subnormal x", fam_subnormal_x),
    ("underflowing products (subnormal scales, tiny x)", fam_underflow_products),
    ("extreme scales (65504, min normal, min subnormal; codes -128/127)", fam_extreme_scales),
    ("absorption (2^30 first, then tiny terms)", fam_absorption),
    ("ties in the underflow range (k eta error)", fam_ties_in_underflow),
    ("near Omega (S within 0.1% of the domain limit)", fam_near_omega),
]


class Diagnostics(unittest.TestCase):
    def test_worst_ratio_table(self):
        t0 = time.time()
        for fi, (label, fn) in enumerate(FAMILIES):
            for k in KS:
                rng = np.random.default_rng(1000 * fi + k)
                raw, x = fn(rng, N_ROWS, k)
                y = gemv.seq_bits(raw, x, N_ROWS, k)
                r = gemv.check_b1(raw, x, y, N_ROWS, k)
                self.assertIn(r.status, ("satisfied", "could_not_evaluate"), f"counterexample candidate: {label}, k={k}: {r}")
                self.assertEqual(gemv.check_seq(raw, x, y, N_ROWS, k).status, r.status if r.status != "satisfied" else "satisfied")
                if r.status == "satisfied":
                    # cross-check this very measurement with the independent oracle at k = 32 (the slow path)
                    if k == 32:
                        st, worst, _, _ = orc.b1_oracle(raw, x, [int(v) for v in y], N_ROWS, k)
                        self.assertEqual(st, "satisfied")
                        self.assertAlmostEqual(float(worst), float(r.worst_ratio), delta=1e-9 * max(1.0, float(worst)))
                    TABLE[(label, k)] = (r.worst_ratio, r.worst_ratio_row, "in domain")
                else:
                    TABLE[(label, k)] = ("-", None, "outside domain: " + r.reasons[0].split(":")[0])
        TABLE["_seconds"] = time.time() - t0
        in_domain = sum(1 for key, v in TABLE.items() if key != "_seconds" and v[2] == "in domain")
        self.assertGreaterEqual(in_domain, 20)


def tearDownModule():
    print("\n\nGEMV diagnostics: worst |y - y*| / B of section 3's program, per family and k (3 rows each, seeded)")
    print(f"{'family':<68} {'k=32':>16} {'k=1024':>16} {'k=3072':>16}")
    for label, _ in FAMILIES:
        cells = []
        for k in KS:
            v = TABLE.get((label, k))
            cells.append("n/a" if v is None else (f"{float(v[0]):.5e}" if v[2] == "in domain" else "outside domain"))
        print(f"{label:<68} {cells[0]:>16} {cells[1]:>16} {cells[2]:>16}")
    print(f"(runtime {TABLE.get('_seconds', 0):.1f} s)")


if __name__ == "__main__":
    unittest.main()
