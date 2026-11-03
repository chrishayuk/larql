"""Positive tests for refcheck.gemv (Q8_0-GEMV-SPEC.md revision 1). Synthetic data only.

Expected values come from three places: results worked out by hand (the HAND tests), `tests/_gemv_oracle.py`
(Fractions with explicit binary32 rounding, sharing no code with refcheck/gemv.py), and the reference's own seq_bits.
"""

import os
import sys
import unittest
from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _gemv_gen as gen  # noqa: E402
import _gemv_oracle as orc  # noqa: E402
from refcheck import gemv  # noqa: E402

ONE_BITS = 0x3F800000


def one_block_row(d_bits, qs_nonzero, x_nonzero):
    """k = 32, n = 1: qs_nonzero / x_nonzero map column -> value for the few non-zero entries."""
    qs = [0] * 32
    for j, q in qs_nonzero.items():
        qs[j] = q
    x = [0] * 32
    for j, b in x_nonzero.items():
        x[j] = b
    return orc.build([[(d_bits, qs)]]), x


def ratio_str(num, den):
    with localcontext() as ctx:
        ctx.prec = 30
        return format(Decimal(num) / Decimal(den), ".12e")


class Hand(unittest.TestCase):
    """Expected bits worked out on paper, not by running any code."""

    def test_two_ties_to_even_keep_one(self):
        # d = 1.0 (0x3C00); q = 1, 1, 1; x = 1.0, 2^-24, 2^-24 (0x3F800000, 0x33800000, 0x33800000).
        # acc: 0 + 1 = 1. 1 + 2^-24 is a tie between 1 and 1 + 2^-23; 1.0 has an even last bit, so 1. Again: 1.
        # Result 1.0 = 0x3F800000. (Pairwise, reverse or binary64 accumulation give 1 + 2^-23 = 0x3F800001.)
        raw, x = one_block_row(0x3C00, {0: 1, 1: 1, 2: 1}, {0: 0x3F800000, 1: 0x33800000, 2: 0x33800000})
        y = np.array([0x3F800000], dtype=np.uint32)
        self.assertEqual(int(gemv.seq_bits(raw, x, 1, 32)[0]), 0x3F800000)
        r = gemv.check_seq(raw, x, y, 1, 32)
        self.assertEqual((r.status, r.rows_checked, r.rows_violating, r.first_violating_row), ("satisfied", 1, 0, None))
        self.assertEqual(gemv.check_seq(raw, x, [0x3F800001], 1, 32).status, "violated")
        # B1 by hand: y* = 1 + 2^-23, |y - y*| = 2^-23, S = 1 + 2^-23, k = 32, gamma_32 = 32/(2^24 - 32) ~ 2^-19.
        # |y - y*| / B ~ 2^-23 / 2^-19 = 1/16 = 0.0625 (the 32 eta term is 2^-145, negligible).
        b = gemv.check_b1(raw, x, y, 1, 32)
        self.assertEqual(b.status, "satisfied")
        self.assertAlmostEqual(float(b.worst_ratio), 0.0625, delta=1e-5)
        self.assertEqual(b.worst_ratio_row, 0)
        # the exact ratio, written out: 2^-23 (2^24 - 32) / (32 S + 32 eta 2^24)  with S = 1 + 2^-23, eta = 2^-150
        S = Fraction(1) + Fraction(1, 2**23)
        eta = Fraction(1, 2**150)
        exact = Fraction(1, 2**23) * (2**24 - 32) / (32 * S + 32 * eta * 2**24)
        self.assertEqual(b.worst_ratio, ratio_str(exact.numerator, exact.denominator))

    def test_simple_exact(self):
        # d = 2.0 (0x4000), q0 = 3, x0 = 0.25 (0x3E800000): 3 * 2 * 0.25 = 1.5 = 0x3FC00000.
        raw, x = one_block_row(0x4000, {0: 3}, {0: 0x3E800000})
        self.assertEqual(int(gemv.seq_bits(raw, x, 1, 32)[0]), 0x3FC00000)

    def test_code_minus_128_and_negative_scale(self):
        # d = -0.5 (0xB800), q = -128, x = 1.0: (-128)(-0.5)(1) = 64.0 = 0x42800000.
        raw, x = one_block_row(0xB800, {0: -128}, {0: ONE_BITS})
        self.assertEqual(int(gemv.seq_bits(raw, x, 1, 32)[0]), 0x42800000)

    def test_code_127_and_largest_scale(self):
        # d = 65504 = 0x7BFF, q = 127, x = 1.0: 127 * 65504 = 8,319,008, an integer below 2^24, so exact in binary32.
        raw, x = one_block_row(0x7BFF, {0: 127}, {0: ONE_BITS})
        out = int(gemv.seq_bits(raw, x, 1, 32)[0])
        self.assertEqual(np.array([out], dtype=np.uint32).view(np.float32)[0], np.float32(127 * 65504))
        self.assertEqual(127 * 65504, 8319008)

    def test_product_tie_rounds_to_even_zero(self):
        # d = 2^-24 (0x0001), q = 1, x = 2^-126 (0x00800000): the product is 2^-150, half the smallest subnormal,
        # a tie between 0 and 2^-149; zero is even -> +0.0 = 0x00000000. (Flushing would give the same bits here, see below.)
        raw, x = one_block_row(0x0001, {0: 1}, {0: 0x00800000})
        self.assertEqual(int(gemv.seq_bits(raw, x, 1, 32)[0]), 0)

    def test_product_tie_rounds_up_to_even_subnormal(self):
        # d = 0.5 (0x3800), q = 1, x = 3 * 2^-149 (0x00000003): 1.5 * 2^-149, a tie between 1 and 2 (times 2^-149);
        # 2 is even -> 2 * 2^-149 = 0x00000002. A flush-to-zero implementation would give 0.
        raw, x = one_block_row(0x3800, {0: 1}, {0: 0x00000003})
        self.assertEqual(int(gemv.seq_bits(raw, x, 1, 32)[0]), 0x00000002)

    def test_sign_of_zero(self):
        # q0 = -1, d = 1, x0 = +0.0: the product is -0.0; acc = +0.0 + -0.0 = +0.0 (spec section 3): 0x00000000.
        raw, x = one_block_row(0x3C00, {0: -1}, {0: 0})
        self.assertEqual(int(gemv.seq_bits(raw, x, 1, 32)[0]), 0x00000000)
        self.assertEqual(gemv.check_seq(raw, x, [0x00000000], 1, 32).status, "satisfied")
        r = gemv.check_seq(raw, x, [0x80000000], 1, 32)
        self.assertEqual((r.status, r.first_violating_row), ("violated", 0))
        # B1 reads values, not bit patterns: -0.0 equals y* = 0 exactly, so B1 is satisfied where SEQ is violated.
        self.assertEqual(gemv.check_b1(raw, x, [0x80000000], 1, 32).status, "satisfied")

    def test_fused_multiply_add_would_differ(self):
        # term 0: q = -127, x = 1.0 -> -127 exactly. term 1: q = 127, x = 1 + 2^-23 (0x3F800001): the exact product
        # 127 + 127 * 2^-23 needs 30 bits; rounded to 24 bits (spacing 2^-17 in [64, 128)) it is 127 + 2^-16
        # (127 * 2^-23 = 1.984 * 2^-17 -> 2 * 2^-17). acc = -127 + (127 + 2^-16) = 2^-16 = 0x37800000.
        # A fused multiply-add would give the exact 127 * 2^-23 (about 1.98 * 2^-17), not 2^-16.
        raw, x = one_block_row(0x3C00, {0: -127, 1: 127}, {0: ONE_BITS, 1: 0x3F800001})
        self.assertEqual(int(gemv.seq_bits(raw, x, 1, 32)[0]), 0x37800000)
        fused = Fraction(127, 2**23)  # exact: -127 + 127 + 127 * 2^-23
        self.assertNotEqual(orc.round_to_f32(fused, 0), 0x37800000)

    def test_sequential_not_pairwise(self):
        # k = 256, d = 1, all q = 1; x0 = 1.0, x_j = 2^-24 for j >= 1. Sequentially every "1 + 2^-24" is a tie that
        # rounds back to 1.0, so the result is 1.0 = 0x3F800000. A pairwise (numpy-style) sum, a reversed sum or a
        # binary64 sum of the products gives something else.
        qs = [1] * 32
        raw = orc.build([[(0x3C00, qs)] * 8])
        x = [ONE_BITS] + [0x33800000] * 255
        self.assertEqual(int(gemv.seq_bits(raw, x, 1, 256)[0]), 0x3F800000)
        prod = np.array(x, dtype=np.uint32).view(np.float32)  # w = 1, so the products are the x values
        self.assertNotEqual(int(np.sum(prod, dtype=np.float32).view(np.uint32)), 0x3F800000)  # numpy's sum is pairwise
        self.assertNotEqual(int(np.float32(np.sum(prod.astype(np.float64))).view(np.uint32)), 0x3F800000)
        rev = np.float32(0)
        for v in prod[::-1]:
            rev = np.float32(rev + v)
        self.assertNotEqual(int(rev.view(np.uint32)), 0x3F800000)


class AgainstOracle(unittest.TestCase):
    """seq_bits, check_seq and check_b1 against the exact-rational oracle on small cases."""

    def cases(self):
        rng = np.random.default_rng(20261103)
        for sk in gen.SCALE_KINDS:
            for xk in gen.X_KINDS:
                qk = gen.Q_KINDS[(len(sk) + len(xk)) % 3]
                yield (sk, xk, qk), gen.make_case(rng, 3, 32, sk, xk, qk)
        for qk in gen.Q_KINDS:
            yield ("typical", "normal", qk, "k64"), gen.make_case(rng, 2, 64, "typical", "normal", qk)

    def test_seq_bits_matches_oracle(self):
        n_in_domain = 0
        for label, (raw, x, n, k) in self.cases():
            ex = orc.exact_row(raw, x, n, k)
            if any(e is None for e in ex) or any(not orc.in_domain(e[1], k) for e in ex):
                continue  # outside the domain the program's result is not what the contracts are about
            n_in_domain += 1
            want = orc.seq_oracle(raw, x, n, k)
            got = [int(v) for v in gemv.seq_bits(raw, x, n, k)]
            self.assertEqual(got, want, label)
            self.assertEqual(gemv.check_seq(raw, x, np.array(want, dtype=np.uint32), n, k).status, "satisfied", label)
        self.assertGreater(n_in_domain, 25)

    def test_b1_matches_oracle_for_exact_and_perturbed_y(self):
        rng = np.random.default_rng(7)
        compared = {"satisfied": 0, "violated": 0, "could_not_evaluate": 0}
        for label, (raw, x, n, k) in self.cases():
            try:
                want = orc.seq_oracle(raw, x, n, k)
            except (OverflowError, ValueError, TypeError):
                want = [0] * n  # outside the domain (non-finite or overflow): the verdict must still agree
            variants = [list(want)]
            for _ in range(2):
                v = list(want)
                i = int(rng.integers(0, n))
                v[i] = (v[i] + int(rng.choice([1, -1, 2, 5, 1000, 1 << 20, 1 << 22]))) & 0x7FFFFFFF | (v[i] & 0x80000000)
                variants.append(v)
            for y in variants:
                got = gemv.check_b1(raw, x, np.array(y, dtype=np.uint32), n, k)
                st, worst, worst_row, viol = orc.b1_oracle(raw, x, y, n, k)
                self.assertEqual(got.status, st, label)
                compared[st] += 1
                if st == "could_not_evaluate":
                    self.assertEqual((got.rows_checked, got.rows_violating, got.first_violating_row), (0, 0, None))
                    continue
                self.assertEqual(got.rows_checked, n)
                self.assertEqual(got.rows_violating, len(viol))
                self.assertEqual(got.first_violating_row, viol[0] if viol else None)
                if worst == "inf":
                    self.assertEqual(got.worst_ratio, "inf")
                else:
                    self.assertEqual(got.worst_ratio, ratio_str(worst.numerator, worst.denominator), label)
                    self.assertEqual(got.worst_ratio_row, worst_row)
        self.assertGreater(compared["satisfied"], 20)
        self.assertGreater(compared["violated"], 3)

    def test_verdict_flips_exactly_at_the_bound(self):
        """y chosen next to y* +/- B: the verdict must equal exact arithmetic on both sides of the edge."""
        rng = np.random.default_rng(99)
        sat = vio = 0
        for trial in range(40):
            raw, x, n, k = gen.make_case(rng, 1, 32, "typical", "wide" if trial % 2 else "normal", "random")
            ex = orc.exact_row(raw, x, n, k)
            if ex[0] is None or not orc.in_domain(ex[0][1], k):
                continue
            ys, S = ex[0]
            B = orc.b1_bound(S, k)
            for target in (ys + B, ys - B):
                try:
                    base = orc.round_to_f32(target, 0)
                except OverflowError:
                    continue
                for delta in range(-3, 4):
                    yb = base + delta
                    if not 0 <= yb < 0xFFFFFFFF or ((yb >> 23) & 255) == 255:
                        continue
                    st, _, _, _ = orc.b1_oracle(raw, x, [yb], n, k)
                    got = gemv.check_b1(raw, x, [yb], n, k).status
                    self.assertEqual(got, st, (trial, delta))
                    sat += st == "satisfied"
                    vio += st == "violated"
        self.assertGreater(sat, 20)
        self.assertGreater(vio, 20)


class Kinds(unittest.TestCase):
    """The scale, code, x and domain cases the work order lists, each checked against the oracle."""

    def run_case(self, raw, x, n, k, expect_status="satisfied"):
        want = orc.seq_oracle(raw, x, n, k)
        got = [int(v) for v in gemv.seq_bits(raw, x, n, k)]
        self.assertEqual(got, want)
        y = np.array(want, dtype=np.uint32)
        self.assertEqual(gemv.check_seq(raw, x, y, n, k).status, "satisfied")
        b = gemv.check_b1(raw, x, y, n, k)
        self.assertEqual(b.status, expect_status)
        self.assertEqual(b.status, orc.b1_oracle(raw, x, want, n, k)[0])
        return got, b

    def test_every_scale_kind(self):
        rng = np.random.default_rng(5)
        for name in ("zero", "negzero", "sub_min", "sub_max", "min_normal", "max_normal", "neg_max", "neg_sub", "one"):
            d = gen.F16[name]
            q = gen.codes(rng, 32, "random")
            q[0], q[1] = -128, 127
            raw = orc.build([[(d, q)], [(d, list(reversed(q)))]])
            # x scaled so the largest scale stays inside the domain
            xk = "normal" if name != "max_normal" and name != "neg_max" else "wide"
            x = gen.xs(rng, 32, xk) if xk == "normal" else [gen.f32bits(v) for v in rng.standard_normal(32)]
            self.run_case(raw, x, 2, 32)

    def test_zero_scales_give_signed_zero_results(self):
        # all-zero weights: every product is a zero; acc starts +0.0 so the result is +0.0 whatever the signs
        for d in (0x0000, 0x8000):
            raw = orc.build([[(d, [5] * 32)]])
            x = [gen.f32bits(-1.5)] * 32
            got, _ = self.run_case(raw, x, 1, 32)
            self.assertEqual(got, [0])

    def test_codes_minus_128_and_127_everywhere(self):
        rng = np.random.default_rng(6)
        raw = orc.build([[(0x3C00, [-128] * 16 + [127] * 16)], [(0x3C00, [127] * 16 + [-128] * 16)]])
        self.run_case(raw, gen.xs(rng, 32, "normal"), 2, 32)

    def test_subnormal_x_and_underflowing_products(self):
        rng = np.random.default_rng(8)
        for sk in ("typical", "subnormal", "sub_min", "min_normal"):
            raw = gen.make_case(rng, 2, 32, "subnormal" if sk == "sub_min" else sk, "subnormal", "random")[0]
            self.run_case(raw, gen.xs(rng, 32, "subnormal"), 2, 32)
        # tiny scale times tiny x: products below 2^-149 round to zero (both signs)
        raw = orc.build([[(0x0001, [1, -1, 2, -2] * 8)]])
        got, b = self.run_case(raw, [0x00000100] * 32, 1, 32)
        self.assertEqual(got, [0])

    def test_wide_exponent_ranges(self):
        rng = np.random.default_rng(9)
        for xk in ("wide", "tiny", "large"):
            raw, x, n, k = gen.make_case(rng, 2, 32, "typical", xk, "random")
            ex = orc.exact_row(raw, x, n, k)
            if any(not orc.in_domain(e[1], k) for e in ex):
                self.assertEqual(gemv.check_b1(raw, x, [0, 0], n, k).status, "could_not_evaluate")
            else:
                self.run_case(raw, x, n, k)

    def test_exact_cancellation(self):
        """y* = 0 exactly. Adjacent +a x, -a x cancel exactly in the program; separated pairs leave rounding residue."""
        rng = np.random.default_rng(10)
        xv = gen.xs(rng, 16, "normal")
        q = [int(v) for v in rng.integers(-127, 128, 16)]
        # adjacent: columns 2i, 2i+1 hold (q_i, -q_i) on the same x
        raw = orc.build([[(0x3C00, [v for qi in q for v in (qi, -qi)])]])
        x = [b for xb in xv for b in (xb, xb)]
        got, b = self.run_case(raw, x, 1, 32)
        self.assertEqual(got, [0])
        self.assertEqual(orc.exact_row(raw, x, 1, 32)[0][0], 0)
        self.assertEqual(float(b.worst_ratio), 0.0)
        # B1 here: |y| <= gamma_32 S + 32 eta (1 + gamma): ~1e-7 S passes, 1e-3 S does not; each verdict equals the oracle's
        S = orc.exact_row(raw, x, 1, 32)[0][1]
        cands = (0x00000001, 0x80000008, 0x00000100, orc.f32_bits_of_float(float(S) * 1e-7),
                 orc.f32_bits_of_float(float(S) * 1e-3), orc.f32_bits_of_float(-float(S) * 1e-3))
        for yb in cands:
            self.assertEqual(gemv.check_b1(raw, x, [yb], 1, 32).status, orc.b1_oracle(raw, x, [yb], 1, 32)[0], hex(yb))
        self.assertEqual(gemv.check_b1(raw, x, [cands[3]], 1, 32).status, "satisfied")
        self.assertEqual(gemv.check_b1(raw, x, [cands[4]], 1, 32).status, "violated")
        self.assertEqual(gemv.check_seq(raw, x, [0x80000000], 1, 32).status, "violated")  # -0.0 is not +0.0
        # separated pairs: y* = 0 but the program's result need not be (rounding between the terms); SEQ equals the oracle
        raw2 = orc.build([[(0x3C00, q + [-v for v in q])]])
        self.run_case(raw2, xv + xv, 1, 32)
        self.assertEqual(orc.exact_row(raw2, xv + xv, 1, 32)[0][0], 0)

    def test_domain_edges(self):
        """Just inside and just outside Omega, with a single term: S = x_0 for q = 1, d = 1."""
        # largest x with (1 + gamma_32)(x + 32 eta) < Omega, by scanning down from the largest finite binary32
        b = 0x7F7FFFFF
        while not orc.in_domain(orc.f32_value(b), 32):
            b -= 1
        inside, outside = b, b + 1
        self.assertFalse(orc.in_domain(orc.f32_value(outside), 32))
        for xb, expect in ((inside, "satisfied"), (outside, "could_not_evaluate")):
            raw, x = one_block_row(0x3C00, {0: 1}, {0: xb})
            if expect == "satisfied":
                got, bres = self.run_case(raw, x, 1, 32)
                self.assertEqual(got, [xb])
            else:
                for fn in (gemv.check_seq, gemv.check_b1):
                    r = fn(raw, x, [xb], 1, 32)
                    self.assertEqual(r.status, "could_not_evaluate")
                    self.assertTrue(any(s.startswith("outside_domain") for s in r.reasons), r.reasons)
                    self.assertEqual(orc.b1_oracle(raw, x, [xb], 1, 32)[0], "could_not_evaluate")

    def test_non_finite_inputs_are_outside_the_domain(self):
        rng = np.random.default_rng(11)
        raw, x, n, k = gen.make_case(rng, 2, 32, "typical", "normal", "random")
        y = np.zeros(2, dtype=np.uint32)
        for bad_x in (0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00001):
            xx = list(x)
            xx[5] = bad_x
            for fn in (gemv.check_seq, gemv.check_b1):
                r = fn(raw, xx, y, n, k)
                self.assertEqual(r.status, "could_not_evaluate")
                self.assertTrue(r.reasons[0].startswith("x_not_finite"))
        for bad_d in (gen.F16["inf"], 0xFC00, gen.F16["nan"]):
            # one row has a non-finite scale (with q = 0 in that block, which would make 0 * inf = NaN); the other is fine
            qs = [0] * 32
            raw2 = orc.build([[(0x3C00, [1] * 32)], [(bad_d, qs)]])
            for fn in (gemv.check_seq, gemv.check_b1):
                r = fn(raw2, x, y, 2, 32)
                self.assertEqual(r.status, "could_not_evaluate")  # the whole input, not just row 1
                self.assertTrue(any(s.startswith("scale_not_finite") for s in r.reasons))
                self.assertEqual(r.rows_checked, 0)

    def test_non_finite_y_inside_the_domain_is_violated(self):
        rng = np.random.default_rng(12)
        raw, x, n, k = gen.make_case(rng, 2, 32, "typical", "normal", "random")
        good = [int(v) for v in gemv.seq_bits(raw, x, n, k)]
        for bad in (0x7F800000, 0xFF800000, 0x7FC00000):
            y = [good[0], bad]
            s = gemv.check_seq(raw, x, y, n, k)
            self.assertEqual((s.status, s.first_violating_row), ("violated", 1))
            b = gemv.check_b1(raw, x, y, n, k)
            self.assertEqual((b.status, b.first_violating_row, b.worst_ratio, b.worst_ratio_row), ("violated", 1, "inf", 1))
            self.assertTrue(b.reasons[0].startswith("y_not_finite"))

    def test_rows_are_independent_and_in_order(self):
        rng = np.random.default_rng(13)
        raw, x, n, k = gen.make_case(rng, 5, 64, "typical", "normal", "random")
        want = orc.seq_oracle(raw, x, n, k)
        self.assertEqual([int(v) for v in gemv.seq_bits(raw, x, n, k)], want)
        y = list(want)
        y[3] ^= 1 << 30
        r = gemv.check_seq(raw, x, y, n, k)
        self.assertEqual((r.rows_checked, r.rows_violating, r.first_violating_row), (5, 1, 3))

    def test_larger_k_matches_between_checks(self):
        # k = 1024 with the oracle (one row; slow path kept to one case)
        rng = np.random.default_rng(14)
        raw, x, n, k = gen.make_case(rng, 1, 1024, "typical", "normal", "random")
        want = orc.seq_oracle(raw, x, n, k)
        self.assertEqual([int(v) for v in gemv.seq_bits(raw, x, n, k)], want)
        b = gemv.check_b1(raw, x, want, n, k)
        st, worst, _, _ = orc.b1_oracle(raw, x, want, n, k)
        self.assertEqual((b.status, b.worst_ratio), (st, ratio_str(worst.numerator, worst.denominator)))


class Refusals(unittest.TestCase):
    def setUp(self):
        self.raw, self.x = one_block_row(0x3C00, {0: 1}, {0: ONE_BITS})
        self.y = [ONE_BITS]

    def refused(self, *args):
        for fn in (gemv.check_seq, gemv.check_b1):
            r = fn(*args)
            self.assertEqual(r.status, "could_not_evaluate", args)
            self.assertTrue(r.reasons and r.reasons[0].startswith("malformed_input"), r.reasons)
            self.assertEqual((r.rows_checked, r.rows_violating, r.first_violating_row), (0, 0, None))

    def test_lengths(self):
        self.refused(self.raw[:-1], self.x, self.y, 1, 32)
        self.refused(self.raw + b"\0", self.x, self.y, 1, 32)
        self.refused(self.raw, self.x[:-1], self.y, 1, 32)
        self.refused(self.raw, self.x + [0], self.y, 1, 32)
        self.refused(self.raw, self.x, self.y + self.y, 1, 32)
        self.refused(self.raw, self.x, [], 1, 32)
        self.refused(self.raw, self.x, self.y, 2, 32)

    def test_k(self):
        self.refused(self.raw, self.x, self.y, 1, 0)
        self.refused(self.raw, self.x, self.y, 1, 33)
        self.refused(self.raw, self.x, self.y, 1, 16)
        self.refused(self.raw, self.x, self.y, 1, -32)
        self.refused(b"", [], [0], 1, 1 << 24)  # k = 2**24 is not below 2**24
        r = gemv.check_seq(b"", [], [0], 1, 1 << 24)
        self.assertIn("2**24", r.reasons[0])
        self.refused(b"", [], [0], 1, 32 * ((1 << 24) // 32 + 1))

    def test_types_and_ranges(self):
        self.refused(self.raw, self.x, self.y, 1.0, 32)
        self.refused(self.raw, self.x, self.y, 1, 32.0)
        self.refused(self.raw, self.x, self.y, True, 32)
        self.refused(self.raw, self.x, self.y, -1, 32)
        self.refused("text", self.x, self.y, 1, 32)
        self.refused(list(self.raw), self.x, self.y, 1, 32)
        self.refused(self.raw, [float(1.0)] * 32, self.y, 1, 32)  # floats are not bit patterns
        self.refused(self.raw, np.ones(32, dtype=np.float32), self.y, 1, 32)
        self.refused(self.raw, self.x, [1.0], 1, 32)
        self.refused(self.raw, self.x, [1 << 32], 1, 32)
        self.refused(self.raw, self.x, [-1], 1, 32)
        self.refused(self.raw, [True] + [0] * 31, self.y, 1, 32)
        self.refused(self.raw, np.zeros((32, 1), dtype=np.uint32), self.y, 1, 32)

    def test_seq_bits_raises(self):
        for args in ((self.raw[:-1], self.x, 1, 32), (self.raw, self.x, 1, 33), (self.raw, self.x, 1, 0), (self.raw, self.x[:-1], 1, 32)):
            with self.assertRaises(ValueError):
                gemv.seq_bits(*args)

    def test_n_zero_is_not_satisfied(self):
        for fn in (gemv.check_seq, gemv.check_b1):
            r = fn(b"", self.x, [], 0, 32)
            self.assertEqual(r.status, "could_not_evaluate")
            self.assertTrue(r.reasons[0].startswith("no_rows"))
            self.assertEqual(r.rows_checked, 0)
        self.assertEqual(len(gemv.seq_bits(b"", self.x, 0, 32)), 0)


class ImplementationFacts(unittest.TestCase):
    def test_numpy_float32_keeps_subnormals_and_does_not_fuse(self):
        a = np.array([2.0**-149], dtype=np.float32)
        self.assertEqual(int((a * np.float32(1)).view(np.uint32)[0]), 1)
        self.assertEqual(int((a + a).view(np.uint32)[0]), 2)

    def test_accepts_numpy_and_list_inputs_equally(self):
        rng = np.random.default_rng(15)
        raw, x, n, k = gen.make_case(rng, 2, 64, "typical", "normal", "random")
        y = gemv.seq_bits(raw, x, n, k)
        for blk in (raw, bytearray(raw), memoryview(raw), np.frombuffer(raw, dtype=np.uint8)):
            for xb in (x, tuple(x), np.array(x, dtype=np.uint32), np.array(x, dtype=np.int64)):
                self.assertEqual(gemv.check_seq(blk, xb, y, n, k).status, "satisfied")
                self.assertEqual(gemv.check_seq(blk, xb, [int(v) for v in y], n, k).status, "satisfied")


if __name__ == "__main__":
    unittest.main()
