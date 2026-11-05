"""Two exact boundaries of refcheck.gemv that no earlier test pins down. Synthetic data only; tests only.

An audit changed refcheck/gemv.py in a scratch copy and two changes passed every test:

1. `check_b1` compared `lhs > rhs`; `lhs >= rhs` passed. Section 4 says `<=`, so an error exactly equal to B_i is `satisfied`.
2. The domain's third condition read `(1 + gamma_k) * S < Omega`, without `k * eta`; that passed too.

Constructions (worked out on paper first; the Fractions below only confirm them, using tests/_gemv_oracle.py, which shares no
code with refcheck/gemv.py). Units: U = 2^-173; a binary16 scale is an integer in 2^-24, a binary32 value an integer in 2^-149.

(1) Exact equality, k = 32, one row, one block. gamma_32 = 32/(2^24 - 32) = 1/(2^19 - 1). Take d = 2^-24 (pattern 0x0001), code
    q_0 = 1 and the other codes 0, x_0 = T * 2^-149 with T = 32 * (2^19 - 17) = 16776672 (pattern 0x00FFFDE0: a normal
    binary32), the other x = +0. Then y* = S = T * U. In units of U, the equality |Y - Y*| (2^24 - k) = k S + k eta 2^24 reads
    (Y - T) (2^24 - 32) = 32 T + 32 * 2^47, i.e. (Y - T)(2^19 - 1) = T + 2^47. With Y = 2^24 * y_int (y_int in 2^-149) and
    T = 32 (c (2^19 - 1) - 2^23) at c = 17, the solution is y_int = 17: y = 17 * 2^-149 (pattern 0x00000011, subnormal), since
    17 * 2^24 - T = 32 (2^23 + 17) and 32 (2^23 + 17)(2^19 - 1) = T + 2^47. So |y - y*| = B exactly. y_int = 18 is one ulp of
    y past B (violated); y_int = 16 is inside. Negating x gives y* < 0 and the mirror image, y = -17 * 2^-149.

(2) The window, k = 64. 1 + gamma_k = 1 / (1 - k u), so (1 + gamma_k) S < Omega  <=>  S < M := Omega (1 - k u), and
    Omega <= (1 + gamma_k)(S + k eta)  <=>  S >= M - k eta. M = 2^258 (2^25 - 1)(2^18 - 1) U is an integer multiple of U and
    k eta = 2^29 U, so the rows in the window are S = M - 2^29 ... M - 1 (in units of U). Such an S is reachable: block 0 has
    scale 65504 (0x7BFF) and all 32 codes 127, block 1 has scale 2^-24 (0x0001) and all 32 codes 1, all x positive, so
    S = sum_j C_j X_j with C_j = 127 * 65504 * 2^24 (block 0) or 1 (block 1) and X_j the x_j in units of 2^-149. x_j are chosen
    greedily: each takes the largest binary32 value (top 24 bits of the integer) not above the remainder. Block 0 takes the
    remainder down below C_0 (about 2^47); block 1, whose coefficient is 1, finishes it exactly. The builder asserts the
    remainder is 0. Largest needed x is about 2^255, under the binary32 maximum, so every x is finite.
    Rows tested: S = M - 1 and S = M - 2^29 (in the window: could_not_evaluate), S = M and S = M + 1 (outside under either
    reading: could_not_evaluate), S = M - 2^29 - 1 (inside the domain: evaluated, so the refusals above are not for another reason).
"""

import os
import sys
import unittest
from fractions import Fraction

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _gemv_oracle as orc  # noqa: E402
from refcheck import gemv  # noqa: E402

UNIT = Fraction(1, 2**173)


def u32(v):
    return np.array(v, dtype=np.uint32)


def one_block(d_bits, q0, x0_bits):
    qs = [0] * 32
    qs[0] = q0
    x = [0] * 32
    x[0] = x0_bits
    return orc.build([[(d_bits, qs)]]), x


class ExactEquality(unittest.TestCase):
    """check_b1 treats |y - y*| == B as satisfied and one ulp of y past it as violated (section 4's `<=`)."""

    K = 32
    T = 32 * (2**19 - 17)  # 16776672
    X_BITS = T  # a normal binary32 with exponent field 1: its pattern equals its integer in units of 2^-149

    def setUp(self):
        self.assertEqual(self.X_BITS, 0x00FFFDE0)

    def _equality_holds(self, d_bits, x_bits, y_bits):
        raw, x = one_block(d_bits, 1, x_bits)
        (ystar, S), = orc.exact_row(raw, x, 1, self.K)
        return abs(orc.f32_value(y_bits) - ystar) - orc.b1_bound(S, self.K)  # exact; 0 means equal

    def test_construction_is_exact(self):
        # Independent of gemv: the error is exactly B; one ulp of y either way is not.
        self.assertEqual(self._equality_holds(0x0001, self.X_BITS, 17), 0)
        self.assertGreater(self._equality_holds(0x0001, self.X_BITS, 18), 0)
        self.assertLess(self._equality_holds(0x0001, self.X_BITS, 16), 0)
        self.assertEqual(self._equality_holds(0x0001, self.X_BITS | 0x80000000, 0x80000011), 0)

    def test_error_equal_to_bound_is_satisfied(self):
        raw, x = one_block(0x0001, 1, self.X_BITS)
        r = gemv.check_b1(raw, x, u32([17]), 1, self.K)
        self.assertEqual((r.status, r.reasons, r.rows_violating), ("satisfied", (), 0))
        self.assertEqual(Fraction(r.worst_ratio), 1)

    def test_one_ulp_past_the_bound_is_violated(self):
        raw, x = one_block(0x0001, 1, self.X_BITS)
        r = gemv.check_b1(raw, x, u32([18]), 1, self.K)
        self.assertEqual((r.status, r.rows_violating, r.first_violating_row), ("violated", 1, 0))
        self.assertTrue(any(s.startswith("b1_exceeded") for s in r.reasons))
        self.assertGreater(float(r.worst_ratio), 1.0)
        self.assertEqual(gemv.check_b1(raw, x, u32([16]), 1, self.K).status, "satisfied")

    def test_mirror_image_with_negative_values(self):
        raw, x = one_block(0x0001, 1, self.X_BITS | 0x80000000)
        r = gemv.check_b1(raw, x, u32([0x80000011]), 1, self.K)
        self.assertEqual((r.status, Fraction(r.worst_ratio)), ("satisfied", 1))
        self.assertEqual(gemv.check_b1(raw, x, u32([0x80000012]), 1, self.K).status, "violated")

    def test_equality_in_a_second_row_is_not_masked(self):
        # Row 0 is far inside; row 1 is exactly on the bound. The verdict is satisfied; one ulp past in row 1 gives violated, row 1.
        qs0 = [0] * 32
        qs1 = [0] * 32
        qs1[0] = 1
        raw = orc.build([[(0x0001, qs0)], [(0x0001, qs1)]])
        x = [0] * 32
        x[0] = self.X_BITS
        self.assertEqual(gemv.check_b1(raw, x, u32([0, 17]), 2, self.K).status, "satisfied")
        r = gemv.check_b1(raw, x, u32([0, 18]), 2, self.K)
        self.assertEqual((r.status, r.first_violating_row), ("violated", 1))


def f32_floor_units(v):
    """Largest integer <= v (in units of 2^-149) that is a finite binary32 magnitude, and its bit pattern."""
    xmax = ((1 << 24) - 1) << 253
    v = min(v, xmax)
    if v < (1 << 24):
        return v, v  # subnormal, or exponent field 1 (pattern equals the integer)
    s = v.bit_length() - 24
    t = v >> s
    return t << s, ((s + 1) << 23) | (t - (1 << 23))


def build_row(target_units):
    """k = 64, one row: returns (blocks, x_bits) whose exact S equals target_units * U."""
    cs = [127 * 65504 * 2**24] * 32 + [1] * 32
    rem = target_units
    xs = []
    for c in cs:
        val, bits = f32_floor_units(rem // c)
        rem -= c * val
        xs.append(bits)
    assert rem == 0, rem
    raw = orc.build([[(0x7BFF, [127] * 32), (0x0001, [1] * 32)]])
    return raw, xs


class DomainThirdCondition(unittest.TestCase):
    """The third domain condition includes k*eta: rows with (1+g)S < Omega <= (1+g)(S+k eta) are outside the domain."""

    K = 64
    M = 2**258 * (2**25 - 1) * (2**18 - 1)  # Omega (1 - k u) in units of U, by hand
    W = 2**29  # k * eta in units of U

    def test_hand_values_agree_with_the_constants(self):
        g = Fraction(self.K, 2**24 - self.K)
        self.assertEqual(self.M * UNIT, orc.OMEGA / (1 + g))
        self.assertEqual(self.W * UNIT, self.K * orc.ETA)

    def _row(self, s_units):
        raw, x = build_row(s_units)
        self.assertTrue(all(orc.f32_value(b) is not None for b in x))  # every x finite
        (ystar, S), = orc.exact_row(raw, x, 1, self.K)
        self.assertEqual(S, s_units * UNIT)  # S is exactly the target
        g = Fraction(self.K, 2**24 - self.K)
        return raw, x, S, g

    def test_window_rows_are_could_not_evaluate(self):
        for s_units in (self.M - 1, self.M - self.W):
            raw, x, S, g = self._row(s_units)
            # the two inequalities of the window, in exact rationals
            self.assertLess((1 + g) * S, orc.OMEGA)
            self.assertLessEqual(orc.OMEGA, (1 + g) * (S + self.K * orc.ETA))
            self.assertFalse(orc.in_domain(S, self.K))
            for y in (0, 0x7F7FFFFF):
                for check in (gemv.check_b1, gemv.check_seq):
                    r = check(raw, x, u32([y]), 1, self.K)
                    self.assertEqual((r.status, r.rows_checked), ("could_not_evaluate", 0), (s_units, y))
                    self.assertTrue(any(s.startswith("outside_domain") for s in r.reasons), r.reasons)

    def test_rows_outside_under_either_reading_are_could_not_evaluate(self):
        for s_units in (self.M, self.M + 1):
            raw, x, S, g = self._row(s_units)
            self.assertGreaterEqual((1 + g) * S, orc.OMEGA)
            for check in (gemv.check_b1, gemv.check_seq):
                r = check(raw, x, u32([0]), 1, self.K)
                self.assertEqual(r.status, "could_not_evaluate")

    def test_row_just_inside_is_evaluated(self):
        raw, x, S, g = self._row(self.M - self.W - 1)
        self.assertLess((1 + g) * (S + self.K * orc.ETA), orc.OMEGA)
        self.assertTrue(orc.in_domain(S, self.K))
        # y = 0 is nowhere near y* (about 2^127), so B1 is violated, which is an evaluated verdict.
        r = gemv.check_b1(raw, x, u32([0]), 1, self.K)
        self.assertEqual((r.status, r.rows_checked), ("violated", 1))
        self.assertFalse(any(s.startswith("outside_domain") for s in r.reasons))
        r = gemv.check_seq(raw, x, u32([0]), 1, self.K)
        self.assertNotEqual(r.status, "could_not_evaluate")
        self.assertFalse(any(s.startswith("outside_domain") for s in r.reasons))


if __name__ == "__main__":
    unittest.main()
