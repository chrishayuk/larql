"""Tests for refcheck.e1, on synthetic data only.

Expected values come from exact rational arithmetic (fractions.Fraction) or from numbers worked out by hand, never from
running the code under test on its own output.
"""

import json
import os
import shutil
import struct
import sys
import unittest
from fractions import Fraction

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from refcheck import container as C  # noqa: E402
from refcheck import e1, q8_0, synth  # noqa: E402

TMP = os.path.join(os.path.dirname(HERE), ".tmp", "test_e1")


def bf16(values) -> np.ndarray:
    """float32 array of values that are exactly representable in BF16 (low 16 bits zero), as a BF16 tensor decodes."""
    a = np.array(values, dtype=np.float32)
    assert not (a.view(np.uint32) & 0xFFFF).any(), "not exactly BF16"
    return a


def f16_value(bits: int) -> Fraction:
    """Exact value of a finite binary16 pattern (Fraction)."""
    s = -1 if bits >> 15 else 1
    e, m = (bits >> 10) & 0x1F, bits & 0x3FF
    if e == 0:
        return s * Fraction(m, 1) * Fraction(1, 2**24)
    return s * (1 + Fraction(m, 1024)) * (Fraction(2) ** (e - 15))


def t2_exact(d: Fraction) -> Fraction:
    """The protocol's T2 right-hand side, in exact arithmetic, written term by term as in PROTOCOL.md."""
    d = abs(d)
    return d * (Fraction(1, 2) + Fraction(1, 2**12) + Fraction(1, 2**16) + 127 * Fraction(1, 2**11)) + 127 * Fraction(1, 2**25)


def block(d_bits: int, qs) -> bytes:
    return synth.q8_block(d_bits, qs)


ONE = 0x3C00  # f16 1.0


class T2Bound(unittest.TestCase):
    def test_factor_is_36849_over_65536(self):
        self.assertEqual(Fraction(e1.T2_FACTOR), Fraction(36849, 65536))
        self.assertEqual(Fraction(e1.T2_ADDEND), Fraction(127, 2**25))

    def test_every_finite_scale_matches_exact_arithmetic(self):
        bits = np.arange(65536, dtype=np.uint16)
        blocks = b"".join(block(int(b), []) for b in bits)
        got = e1.t2_bounds(blocks)
        for b in range(65536):
            # D1 OQ1: defined for a finite d >= 0 (+0 and -0 included); undefined (NaN) for negative, +/-inf, NaN.
            if (b & 0x7C00) == 0x7C00 or ((b & 0x8000) and (b & 0x7FFF) != 0):
                self.assertTrue(np.isnan(got[b]), hex(b))
            else:
                self.assertEqual(Fraction(float(got[b])), t2_exact(f16_value(b)), hex(b))

    def test_negative_scale_has_no_bound(self):
        # The old reading used |d| for a negative scale; D1 OQ1 puts such blocks outside T2's domain.
        got = e1.t2_bounds(block(ONE, []) + block(0xBC00, []))
        self.assertFalse(np.isnan(got[0]))
        self.assertTrue(np.isnan(got[1]))

    def test_scale_class_covers_every_f16_pattern_exactly_once(self):
        bits = np.arange(65536, dtype=np.uint16)
        d = q8_0.f16_bits_to_f32_bits(bits).view(np.float32).astype(np.float64)
        cls = e1.scale_class(d)
        for b in range(65536):
            exp_ = b & 0x7C00
            frac, neg = b & 0x3FF, bool(b & 0x8000)
            if exp_ == 0x7C00:
                want = 4 if frac else (3 if neg else 2)
            elif neg and (b & 0x7FFF) != 0:
                want = 1
            else:
                want = 0  # positive finite, +0, -0
            self.assertEqual(int(cls[b]), want, hex(b))
        self.assertTrue(np.array_equal(e1.in_t2_domain(d), cls == 0))

    def test_zero_scale_bound_is_the_addend_only(self):
        got = e1.t2_bounds(block(0x0000, []) + block(0x8000, []))
        self.assertEqual(Fraction(float(got[0])), Fraction(127, 2**25))
        self.assertEqual(got[0], got[1])


class ElementCounting(unittest.TestCase):
    def test_hand_worked_block(self):
        # d = 1.0, codes [3, 0, ...]; x = [3.5, 0.25, 0, ...]. Errors 0.5 and 0.25, both inside 0.5623.
        x = np.zeros(32, dtype=np.float32)
        x[0], x[1] = 3.5, 0.25
        r = e1.e1_q8_tensor(x, block(ONE, [3]))
        self.assertEqual(r["n_exceed_t2"], 0)
        self.assertEqual(r["max_abs_error"], 0.5)
        self.assertEqual(r["zero_scale_blocks"], 0)
        # x = 3.7 is 0.7 from 3: exceeds. x = 3.0 + 0.5 is inside.
        x[0] = 3.7
        r = e1.e1_q8_tensor(x, block(ONE, [3]))
        self.assertEqual(r["n_exceed_t2"], 1)
        self.assertEqual(r["n_blocks_exceed_t2"], 1)
        self.assertAlmostEqual(r["max_abs_error"], 0.7, places=6)

    def test_rel_frobenius_by_hand(self):
        # x = [2, 0...], decoded [1, 0...]: error 1, norm 2 -> 0.5 exactly.
        x = np.zeros(32, dtype=np.float32)
        x[0] = 2.0
        r = e1.e1_q8_tensor(x, block(ONE, [1]))
        self.assertEqual(r["err_frobenius"], 1.0)
        self.assertEqual(r["ref_frobenius"], 2.0)
        self.assertEqual(r["rel_frobenius"], 0.5)

    def test_rel_frobenius_two_blocks_two_rows(self):
        # Two blocks: errors (3,4 in separate elements of block 0 -> 5) and 0; norm sqrt(100+36+...)
        x = np.zeros(64, dtype=np.float32)
        x[0], x[1] = 10.0, 6.0
        blocks = block(ONE, [7, 3]) + block(ONE, [])
        r = e1.e1_q8_tensor(x.reshape(2, 32), blocks)
        # errors: |10-7| = 3, |6-3| = 3 -> sqrt(18); ||x|| = sqrt(136)
        self.assertAlmostEqual(r["rel_frobenius"], np.sqrt(18) / np.sqrt(136), places=14)
        self.assertEqual(r["n_blocks"], 2)

    def test_zero_reference_norm_q8_is_null_with_flag_and_absolute_error(self):
        # D4: ||upstream|| = 0 -> relative error is null (never 0, 0/0 or x/0), flag set, absolute error reported.
        z = np.zeros(32, dtype=np.float32)
        r = e1.e1_q8_tensor(z, block(ONE, []))  # decoded is all zero too: still null, not 0
        self.assertIsNone(r["rel_frobenius"])
        self.assertIs(r["zero_reference_norm"], True)
        self.assertEqual((r["err_frobenius"], r["ref_frobenius"], r["max_abs_error"]), (0.0, 0.0, 0.0))
        r = e1.e1_q8_tensor(z, block(ONE, [3, 4]))  # decoded [3, 4, 0...]: absolute Frobenius error 5
        self.assertIsNone(r["rel_frobenius"])
        self.assertIs(r["zero_reference_norm"], True)
        self.assertEqual(r["err_frobenius"], 5.0)
        self.assertEqual(r["max_abs_error"], 4.0)
        self.assertEqual(json.loads(e1.canonical_json(r))["rel_frobenius"], None)

    def test_nonzero_norm_q8_has_flag_false(self):
        x = np.zeros(32, dtype=np.float32)
        x[0] = 2.0
        r = e1.e1_q8_tensor(x, block(ONE, [2]))  # exact decode: error 0, norm 2
        self.assertIs(r["zero_reference_norm"], False)
        self.assertEqual((r["rel_frobenius"], r["err_frobenius"], r["max_abs_error"]), (0.0, 0.0, 0.0))

    def test_boundary_matches_exact_arithmetic(self):
        # d = 1, q = 0: the element x is inside T2 iff |x| <= 36849/65536 + 127/2**25 (exactly). Probe f32 values
        # around that bound, ones either side of it, and compare the count with a Fraction decision per value.
        bound = t2_exact(Fraction(1))
        centre = np.float32(float(bound))
        xs = [centre]
        for _ in range(40):
            xs.append(np.nextafter(xs[-1], np.float32(1)))
        ys = [centre]
        for _ in range(40):
            ys.append(np.nextafter(ys[-1], np.float32(0)))
        vals = np.array(sorted(set(xs + ys)), dtype=np.float32)
        want = sum(1 for v in vals if abs(Fraction(float(v))) > bound)
        self.assertTrue(0 < want < len(vals))  # the probe straddles the bound
        # one element per block so each is judged separately
        data = b"".join(block(ONE, []) for _ in vals)
        x = np.zeros((len(vals), 32), dtype=np.float32)
        x[:, 0] = vals
        r = e1.e1_q8_tensor(x, data)
        self.assertEqual(r["n_exceed_t2"], want)

    def test_error_is_formed_exactly(self):
        # Random f32 x against random-scale blocks: max error must equal the exact rational maximum.
        rng = np.random.default_rng(3)
        x = (rng.standard_normal((50, 32)) * 0.3).astype(np.float32)
        raw = synth.ref_quantize_q8_0(x)
        dec = q8_0.dequantize_q8_0(raw).reshape(50, 32)
        want = max(abs(Fraction(float(a)) - Fraction(float(b))) for a, b in zip(x.ravel(), dec.ravel()))
        r = e1.e1_q8_tensor(x, raw)
        self.assertEqual(Fraction(r["max_abs_error"]), want)

    def test_zero_scale_block_with_nonzero_weights(self):
        x = np.zeros(32, dtype=np.float32)
        x[5] = 1e-3
        for sbits in (0x0000, 0x8000):
            r = e1.e1_q8_tensor(x, block(sbits, [100]))  # codes are irrelevant: q * 0 = 0
            self.assertEqual(r["zero_scale_blocks"], 1)
            self.assertEqual(r["n_exceed_t2"], 1)  # 1e-3 > 127 * 2**-25
        x[5] = 3e-6  # below 127*2**-25 = 3.78e-6
        r = e1.e1_q8_tensor(x, block(0x0000, []))
        self.assertEqual((r["zero_scale_blocks"], r["n_exceed_t2"]), (1, 0))

    def test_nonfinite_upstream_in_a_domain_block_is_outside_t2_not_exceeding_it(self):
        # D5 AMB-019: counted by class, never within and never exceeding. D6 replaces D5's "non-finite in the
        # floating-point statistics": the full-tensor metrics are null with a reason.
        for value, cls, d6cls in ((np.nan, "nan", "nan"), (np.inf, "inf", "+inf"), (-np.inf, "-inf", "-inf")):
            x = np.zeros(32, dtype=np.float32)
            x[2] = value
            r = e1.e1_q8_tensor(x, block(ONE, []))
            want = {"nan": 0, "inf": 0, "-inf": 0}
            want[cls] = 1
            self.assertEqual(r["nonfinite_upstream_in_t2_domain"], want, cls)
            self.assertEqual(r["n_nonfinite_upstream_in_t2_domain"], 1)
            self.assertEqual((r["n_within_t2"], r["n_exceed_t2"], r["n_blocks_exceed_t2"]), (31, 0, 0), cls)
            self.assertEqual(r["n_elements_in_t2_domain"], 32)
            self.assertEqual(r["n_nonfinite_elements"], 1)
            self.assertEqual(r["n_elements_outside_t2_domain"], 0)  # the block itself is in the domain
            self.assertEqual(r["error_over_bound_histogram"]["nonfinite_upstream_in_t2_domain"], 1)
            self.assertEqual(sum(r["error_over_bound_histogram"].values()), 32)
            self.assertEqual(r["error_over_bound_histogram"][">1"], 0)
            self.assertEqual(r["worst_ratio"], 0.0)  # the 31 finite zeros decode exactly
            self.assertEqual(r["numerical_metric_status"], "non_comparable_nonfinite", cls)
            for f in ("max_abs_error", "absolute_frobenius_error", "err_frobenius", "ref_frobenius", "rel_frobenius"):
                self.assertIsNone(r[f], (cls, f))
                self.assertEqual(r["null_metric_reasons"][f], "nonfinite_upstream")
            d6 = {"nan": 0, "+inf": 0, "-inf": 0}
            d6[d6cls] = 1
            self.assertEqual(r["upstream_nonfinite"], d6)
            self.assertNotIn("max_abs_error_finite_elements", r)  # only a finite_subset_* name for this tensor
            self.assertEqual((r["finite_subset_n_elements"], r["finite_subset_max_abs_error"]), (31, 0.0))
            # canonical report: null metrics are JSON null, the integer counts are numbers
            text = e1.canonical_json(r)
            self.assertIn('"nonfinite_upstream_in_t2_domain":' + e1.json.dumps(want, sort_keys=True, separators=(",", ":")), text)
            self.assertIn('"max_abs_error":null', text)
            self.assertIn('"absolute_frobenius_error":null', text)

    def test_nonfinite_upstream_exact_counts_in_a_mixed_tensor(self):
        # block 0 (scale 1.0): 3 NaN, 2 +inf, 1 -inf, 26 finite zeros; block 1 (scale 1.0): 1 finite element 100.0
        # (exceeds: code 0 decodes to 0, error 100 > bound); block 2 (+inf scale): 32 NaN upstream but outside the domain.
        x = np.zeros(96, dtype=np.float32)
        x[[0, 5, 9]] = np.nan
        x[[1, 2]] = np.inf
        x[3] = -np.inf
        x[32] = 100.0
        x[64:96] = np.nan
        r = e1.e1_q8_tensor(x, block(ONE, []) + block(ONE, []) + block(0x7C00, [1] * 32))
        self.assertEqual(r["nonfinite_upstream_in_t2_domain"], {"nan": 3, "inf": 2, "-inf": 1})
        self.assertEqual(r["n_nonfinite_upstream_in_t2_domain"], 6)
        self.assertEqual(r["n_elements_in_t2_domain"], 64)
        self.assertEqual((r["n_within_t2"], r["n_exceed_t2"]), (26 + 31, 1))
        self.assertEqual(r["n_within_t2"] + r["n_exceed_t2"] + r["n_nonfinite_upstream_in_t2_domain"],
                         r["n_elements_in_t2_domain"])
        self.assertEqual(r["n_elements_outside_t2_domain"], 32)  # the +inf-scale block, not counted by class above
        self.assertEqual(r["n_blocks_exceed_t2"], 1)
        h = r["error_over_bound_histogram"]
        self.assertEqual((h["nonfinite_upstream_in_t2_domain"], h["outside_t2_domain"], h[">1"]), (6, 32, 1))
        self.assertEqual(sum(h.values()), 96)

    def test_block_of_only_nonfinite_upstream_has_no_ratio(self):
        x = np.full(32, np.nan, dtype=np.float32)
        r = e1.e1_q8_tensor(x, block(ONE, []))
        self.assertEqual(r["nonfinite_upstream_in_t2_domain"], {"nan": 32, "inf": 0, "-inf": 0})
        self.assertEqual((r["n_within_t2"], r["n_exceed_t2"]), (0, 0))
        self.assertIsNone(r["worst_ratio"])
        self.assertEqual(r["block_error_over_bound"]["q0.5"], None)
        self.assertEqual(r["n_blocks_in_t2_domain"], 1)

    def test_finite_tensor_reports_zero_nonfinite_upstream(self):
        r = e1.e1_q8_tensor(np.zeros(32, dtype=np.float32), block(ONE, []))
        self.assertEqual(r["nonfinite_upstream_in_t2_domain"], {"nan": 0, "inf": 0, "-inf": 0})
        self.assertEqual(r["n_nonfinite_upstream_in_t2_domain"], 0)

    def test_nonfinite_error_statistics_with_nan_and_inf_together(self):
        # Before D6 these were inf; D6 makes the full-tensor metrics and the reference norm null.
        x = np.zeros(32, dtype=np.float32)
        x[2] = np.inf
        x[3] = np.nan
        r = e1.e1_q8_tensor(x, block(ONE, []))
        self.assertIsNone(r["absolute_frobenius_error"])
        self.assertIsNone(r["ref_frobenius"])
        self.assertEqual(r["upstream_nonfinite"], {"nan": 1, "+inf": 1, "-inf": 0})


class T2DomainClasses(unittest.TestCase):
    """D1 OQ1: three classes of block; the out-of-domain ones are inventoried by scale class and counted nowhere in T2."""

    CASES = (("negative", 0xBC00), ("pos_inf", 0x7C00), ("neg_inf", 0xFC00), ("nan", 0x7E00))

    def test_each_class_is_counted_and_never_within_or_exceeding(self):
        for name, bits in self.CASES:
            x = np.full(32, 0.5, dtype=np.float32)
            r = e1.e1_q8_tensor(x, block(bits, [1] * 32))
            self.assertEqual(r["n_blocks_outside_t2_domain"], 1, name)
            self.assertEqual(r["n_elements_outside_t2_domain"], 32, name)
            for other, _ in self.CASES:
                self.assertEqual(r["outside_t2_domain"][other], {"blocks": int(other == name), "elements": 32 * int(other == name)})
            self.assertEqual((r["n_within_t2"], r["n_exceed_t2"], r["n_blocks_exceed_t2"]), (0, 0, 0), name)
            self.assertEqual((r["n_blocks_in_t2_domain"], r["n_elements_in_t2_domain"]), (0, 0), name)
            self.assertEqual(r["zero_scale_blocks"], 0)
            self.assertEqual(r["n_blocks"], 1)  # stays in the inventory
            self.assertIsNone(r["worst_ratio"])
            self.assertEqual(r["error_over_bound_histogram"]["outside_t2_domain"], 32)
            self.assertEqual(sum(r["error_over_bound_histogram"].values()), 32)

    def test_infinite_and_nan_scales_report_nonfinite_statistics(self):
        x = np.full(32, 0.5, dtype=np.float32)
        r = e1.e1_q8_tensor(x, block(0x7C00, [1] * 32))  # +inf scale, codes 1: decode +inf
        self.assertEqual(r["max_abs_error"], np.inf)
        self.assertEqual(r["n_nonfinite_elements"], 32)
        self.assertEqual(r["max_abs_error_finite_elements"], 0.0)
        self.assertEqual(r["block_max_abs_error"]["q1"], np.inf)
        r = e1.e1_q8_tensor(x, block(0x7C00, [0] * 32))  # code 0 x inf = NaN
        self.assertTrue(np.isnan(r["max_abs_error"]))

    def test_negative_scale_statistics_use_the_decoded_values(self):
        # d = -1, codes 1: decoded -1, upstream +1: error 2 everywhere. Included in the other statistics.
        x = np.ones(32, dtype=np.float32)
        r = e1.e1_q8_tensor(x, block(0xBC00, [1] * 32))
        self.assertEqual(r["max_abs_error"], 2.0)
        self.assertAlmostEqual(r["rel_frobenius"], 2.0)
        self.assertEqual(r["outside_t2_domain"]["negative"], {"blocks": 1, "elements": 32})
        self.assertEqual(r["n_exceed_t2"], 0)  # |d| is not used to put it inside the domain

    def test_zero_scales_stay_inside_with_their_own_count(self):
        x = np.zeros(64, dtype=np.float32)
        x[40] = 1e-3
        r = e1.e1_q8_tensor(x, block(0x0000, []) + block(0x8000, []))
        self.assertEqual(r["zero_scale_blocks"], 2)
        self.assertEqual((r["n_blocks_in_t2_domain"], r["n_blocks_outside_t2_domain"]), (2, 0))
        self.assertEqual((r["n_within_t2"], r["n_exceed_t2"]), (63, 1))

    def test_mixed_tensor_counts_add_up(self):
        x = np.zeros(32 * 6, dtype=np.float32)
        x[32 * 1 + 3] = 1.0  # block 1 has a +0 scale (decodes to 0): in domain, and the only exceeding element
        blocks = (block(ONE, []) + block(0x0000, [])  # in, in (+0)
                  + block(0xBC00, []) + block(0x7C00, []) + block(0xFC00, []) + block(0x7E00, []))
        r = e1.e1_q8_tensor(x, blocks)
        self.assertEqual(r["n_blocks"], 6)
        self.assertEqual(r["n_blocks_in_t2_domain"] + r["n_blocks_outside_t2_domain"], 6)
        self.assertEqual(r["n_within_t2"] + r["n_exceed_t2"], r["n_elements_in_t2_domain"])
        self.assertEqual(r["n_elements_in_t2_domain"] + r["n_elements_outside_t2_domain"], r["n_elements"])
        self.assertEqual(sum(v["blocks"] for v in r["outside_t2_domain"].values()), 4)
        self.assertEqual(r["nonfinite_scale_blocks"], 3)
        self.assertEqual(r["n_exceed_t2"], 1)  # only the 1.0 in the +0-scale block
        self.assertEqual(r["zero_scale_blocks"], 1)

    def test_report_is_deterministic_and_integer_counts_are_ints(self):
        x = np.zeros(64, dtype=np.float32)
        blocks = block(0x7C00, [0] * 32) + block(ONE, [])
        a, b = e1.e1_q8_tensor(x, blocks), e1.e1_q8_tensor(x, blocks)
        self.assertEqual(e1.canonical_json(a), e1.canonical_json(b))
        self.assertIn('"nan"', e1.canonical_json(a))  # non-finite float written as a string, strict JSON
        for k in ("n_exceed_t2", "n_within_t2", "n_blocks_outside_t2_domain", "zero_scale_blocks", "n_nonfinite_elements"):
            self.assertIs(type(a[k]), int, k)
        for v in a["outside_t2_domain"].values():
            self.assertIs(type(v["blocks"]), int)
            self.assertIs(type(v["elements"]), int)

    def test_shape_and_length_errors(self):
        with self.assertRaises(ValueError):
            e1.e1_q8_tensor(np.zeros(33, dtype=np.float32), block(ONE, []))
        with self.assertRaises(ValueError):
            e1.e1_q8_tensor(np.zeros(64, dtype=np.float32), block(ONE, []))


class EncoderConformsToT2(unittest.TestCase):
    """The reference encoder is ggml's; T2 is derived from it, so it must never exceed T2. Not a proof (protocol)."""

    def run_sigma(self, sigma, n=3000, seed=0):
        rng = np.random.default_rng(seed)
        x = (rng.standard_normal((n, 32)) * sigma).astype(np.float32)
        return e1.e1_q8_tensor(x, synth.ref_quantize_q8_0(x))

    def test_no_violations_over_scales_including_the_f16_subnormal_range(self):
        for sigma in (1.0, 0.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8):
            r = self.run_sigma(sigma, seed=int(-np.log10(sigma) * 10))
            self.assertEqual(r["n_exceed_t2"], 0, f"sigma={sigma} worst={r['worst_ratio']}")
            self.assertLessEqual(r["worst_ratio"], 1.0)

    def test_forced_ties(self):
        # Codes land exactly halfway: x = (k + 0.5) * d with d a power of two, amax pinned so d = 1/... exact.
        d = np.float32(2.0**-7)
        x = np.zeros((200, 32), dtype=np.float32)
        x[:, 0] = np.float32(127) * d  # amax -> d32 = d exactly
        x[:, 1:] = (np.arange(1, 32, dtype=np.float32) + np.float32(0.5)) * d
        r = e1.e1_q8_tensor(x, synth.ref_quantize_q8_0(x))
        self.assertEqual(r["n_exceed_t2"], 0)
        want = Fraction(1, 2) * Fraction(float(d)) / t2_exact(Fraction(float(d)))  # every tie sits half a step from its code
        self.assertAlmostEqual(r["worst_ratio"], float(want), places=12)

    def test_amax_zero_block(self):
        r = e1.e1_q8_tensor(np.zeros((4, 32), dtype=np.float32), synth.ref_quantize_q8_0(np.zeros((4, 32), np.float32)))
        self.assertEqual((r["n_exceed_t2"], r["zero_scale_blocks"], r["max_abs_error"]), (0, 4, 0.0))

    def test_a_truncating_f16_conversion_is_caught(self):
        # A deliberately wrong encoder: the scale is truncated to f16 (not rounded) and flushed to zero below the f16
        # normal range. This is the family of fault T2 exists to expose; the point here is that the counter sees it.
        rng = np.random.default_rng(11)
        x = (rng.standard_normal((4000, 32)) * 0.02).astype(np.float32)
        amax = np.abs(x).max(axis=1)
        d32 = (amax / np.float32(127)).astype(np.float32)
        trunc = (d32.view(np.uint32) & np.uint32(0xFFFFE000)).view(np.float32)  # drop 13 mantissa bits
        d16 = trunc.astype(np.float16).view(np.uint16)
        q = np.rint(x / d32[:, None]).astype(np.int8)
        raw = b"".join(block(int(b), qs) for b, qs in zip(d16, q.tolist()))
        r = e1.e1_q8_tensor(x, raw)
        self.assertGreater(r["n_exceed_t2"], 0)
        self.assertGreater(r["worst_ratio"], 1.0)


class DistributionReport(unittest.TestCase):
    def test_histogram_partitions_the_elements(self):
        rng = np.random.default_rng(5)
        x = (rng.standard_normal((100, 32)) * 0.05).astype(np.float32)
        r = e1.e1_q8_tensor(x, synth.ref_quantize_q8_0(x))
        self.assertEqual(sum(r["error_over_bound_histogram"].values()), r["n_elements"])
        self.assertEqual(r["error_over_bound_histogram"][">1"], r["n_exceed_t2"])

    def test_histogram_bin_edges(self):
        # d = 1, q = 0 everywhere, so the error of an element is |x|; bound 0.5623. By hand: 0.25 -> ratio 0.4446
        # (bin [0.25,0.5)), 0.5 -> 0.8892 (bin [0.75,0.9)), the rest 0 (bin [0,0.1)).
        x = np.zeros((3, 32), dtype=np.float32)
        x[1, 0], x[2, 0] = 0.25, 0.5
        h = e1.e1_q8_tensor(x, block(ONE, []) * 3)["error_over_bound_histogram"]
        self.assertEqual(h["[0,0.1)"], 3 * 32 - 2)
        self.assertEqual((h["[0.25,0.5)"], h["[0.75,0.9)"]), (1, 1))
        self.assertEqual(sum(h.values()), 96)

    def test_quantile_summary_keys_and_order(self):
        rng = np.random.default_rng(6)
        x = (rng.standard_normal((200, 32)) * 0.05).astype(np.float32)
        r = e1.e1_q8_tensor(x, synth.ref_quantize_q8_0(x))
        for k in ("block_max_abs_error", "block_error_over_bound"):
            v = [r[k][f"q{q:g}"] for q in e1.QUANTILES]
            self.assertEqual(v, sorted(v))
        self.assertEqual(r["block_max_abs_error"]["q1"], r["max_abs_error"])
        self.assertEqual(r["block_error_over_bound"]["q1"], r["worst_ratio"])


class UnquantisedAndCoverage(unittest.TestCase):
    def test_bit_exact_compare_distinguishes_signed_zero(self):
        a = np.array([0.0, 1.0, -2.0], dtype=np.float32)
        self.assertTrue(e1.e1_exact_tensor(a, a.copy())["bit_identical"])
        b = a.copy()
        b[0] = -0.0
        r = e1.e1_exact_tensor(a, b)
        self.assertEqual((r["bit_identical"], r["n_bit_mismatches"]), (False, 1))

    def test_bit_identical_bf16_tensor_reports_zero_error(self):
        a = bf16([1.5, -2.0, 0.25, 3.0])
        r = e1.e1_exact_tensor(a, a.copy())
        self.assertTrue(r["bit_identical"])
        self.assertEqual(r["n_bit_mismatches"], 0)
        self.assertEqual((r["max_abs_error"], r["err_frobenius"], r["rel_frobenius"]), (0.0, 0.0, 0.0))
        self.assertIs(r["zero_reference_norm"], False)
        self.assertEqual(r["ref_frobenius"], float(np.sqrt(1.5**2 + 4 + 0.0625 + 9)))

    def test_differing_bf16_tensor_computes_both_from_values(self):
        a = bf16([3.0, 4.0, 0.0, 0.0])
        b = bf16([3.0, 4.0, 0.5, -1.0])  # two differing elements
        r = e1.e1_exact_tensor(a, b)
        self.assertEqual(r["n_bit_mismatches"], 2)
        self.assertFalse(r["bit_identical"])
        self.assertEqual(r["max_abs_error"], 1.0)  # not the mismatch count 2
        self.assertAlmostEqual(r["err_frobenius"], np.sqrt(1.25), places=15)
        self.assertAlmostEqual(r["rel_frobenius"], np.sqrt(1.25) / 5.0, places=15)
        self.assertIs(r["zero_reference_norm"], False)

    def test_bf16_differing_by_one_bf16_ulp_uses_the_value_difference(self):
        a = bf16([1.0, 1.0])
        b = a.copy()
        b[0] = np.float32(1.0078125)  # next bf16 above 1.0
        r = e1.e1_exact_tensor(a, b)
        self.assertEqual(r["max_abs_error"], 0.0078125)
        self.assertEqual(r["rel_frobenius"], 0.0078125 / np.sqrt(2.0))

    def test_signed_zero_difference_has_zero_error_but_a_mismatch(self):
        a = bf16([1.0, 0.0])
        b = bf16([1.0, -0.0])
        r = e1.e1_exact_tensor(a, b)
        self.assertEqual((r["n_bit_mismatches"], r["max_abs_error"], r["rel_frobenius"]), (1, 0.0, 0.0))

    def test_zero_reference_norm_bf16_is_null_with_flag(self):
        z = bf16([0.0, -0.0, 0.0])
        r = e1.e1_exact_tensor(z, z.copy())  # bit-identical, zero norm: null, not 0
        self.assertTrue(r["bit_identical"])
        self.assertIsNone(r["rel_frobenius"])
        self.assertIs(r["zero_reference_norm"], True)
        self.assertEqual((r["err_frobenius"], r["max_abs_error"]), (0.0, 0.0))
        r = e1.e1_exact_tensor(z, bf16([0.0, 3.0, 4.0]))  # differing: null, absolute Frobenius error 5
        self.assertIsNone(r["rel_frobenius"])
        self.assertIs(r["zero_reference_norm"], True)
        self.assertEqual((r["err_frobenius"], r["max_abs_error"]), (5.0, 4.0))
        self.assertIsNone(json.loads(e1.canonical_json(r))["rel_frobenius"])

    def test_bit_identical_nonfinite_tensor_is_identical_but_not_comparable(self):
        # D6 supersedes the D4-era reading (error 0, rel 0): representation identity passes, the metrics are null.
        a = np.array([1.0, np.inf], dtype=np.float32)
        r = e1.e1_exact_tensor(a, a.copy())
        self.assertEqual((r["bit_identical"], r["n_bit_mismatches"]), (True, 0))
        self.assertEqual(r["numerical_metric_status"], "non_comparable_nonfinite")
        for f in ("max_abs_error", "absolute_frobenius_error", "err_frobenius", "ref_frobenius", "rel_frobenius"):
            self.assertIsNone(r[f], f)
            self.assertEqual(r["null_metric_reasons"][f], "nonfinite_upstream")
        self.assertEqual(r["upstream_nonfinite"], {"nan": 0, "+inf": 1, "-inf": 0})

    def test_nan_payloads_are_compared_as_bits(self):
        a = np.array([0x7FC00000, 0x7FC00001], dtype=np.uint32).view(np.float32)
        self.assertEqual(e1.e1_exact_tensor(a, a[::-1].copy())["n_bit_mismatches"], 2)

    def test_coverage(self):
        up = {"a": (2, 3), "b": (4,), "c": (1,)}
        self.assertTrue(e1.coverage(dict(up), up)["ok"])
        r = e1.coverage({"a": (3, 2), "b": (4,), "z": (1,)}, up)
        self.assertFalse(r["ok"])
        self.assertEqual(r["missing_from_container"], ["c"])
        self.assertEqual(r["extra_in_container"], ["z"])
        self.assertEqual(r["shape_mismatch"], {"a": {"container": [3, 2], "upstream": [2, 3]}})


FULL = ("max_abs_error", "absolute_frobenius_error", "err_frobenius", "ref_frobenius", "rel_frobenius")


def f32_from_bits(*bits) -> np.ndarray:
    return np.array(bits, dtype=np.uint32).view(np.float32)


class D6NonFiniteUpstream(unittest.TestCase):
    """D6: a tensor whose upstream values include NaN, +inf or -inf, any representation."""

    def assert_non_comparable(self, r):
        self.assertEqual(r["numerical_metric_status"], "non_comparable_nonfinite")
        for f in FULL:
            self.assertIsNone(r[f], f)
        self.assertEqual(r["null_metric_reasons"], {f: "nonfinite_upstream" for f in FULL})
        self.assertIs(r["zero_reference_norm"], False)

    def test_q8_bit_identical_nonfinite_tensor(self):
        # block 0: +inf scale, codes 1 -> +inf (32 elements); block 1: -inf scale, codes 1 -> -inf; block 2: scale 1.0,
        # codes 0..31 -> 0..31. Upstream equal bit for bit.
        x = np.concatenate([np.full(32, np.inf), np.full(32, -np.inf), np.arange(32)]).astype(np.float32)
        r = e1.e1_q8_tensor(x, block(0x7C00, [1] * 32) + block(0xFC00, [1] * 32) + block(ONE, list(range(32))))
        self.assertEqual((r["bit_identical"], r["n_bit_mismatches"]), (True, 0))
        self.assert_non_comparable(r)
        # Whole-tensor upstream counts (D6) include the out-of-domain blocks; D5's in-domain counts do not.
        self.assertEqual(r["upstream_nonfinite"], {"nan": 0, "+inf": 32, "-inf": 32})
        self.assertEqual(r["n_upstream_nonfinite"], 64)
        self.assertEqual(r["nonfinite_upstream_in_t2_domain"], {"nan": 0, "inf": 0, "-inf": 0})
        self.assertEqual((r["n_within_t2"], r["n_exceed_t2"], r["n_blocks_outside_t2_domain"]), (32, 0, 2))
        self.assertEqual((r["finite_subset_n_elements"], r["finite_subset_max_abs_error"]), (32, 0.0))

    def test_q8_infinity_sign_and_nan_payload_are_bits(self):
        # Upstream -inf where the decode is +inf: a mismatch. A signalling-NaN upstream pattern can never equal a
        # decoded NaN (arithmetic NaNs are quiet), whatever the host's payload rule: a mismatch.
        x = np.full(32, np.inf, dtype=np.float32)
        x[0] = -np.inf
        r = e1.e1_q8_tensor(x, block(0x7C00, [1] * 32))
        self.assertEqual((r["bit_identical"], r["n_bit_mismatches"]), (False, 1))
        self.assertEqual(r["upstream_nonfinite"], {"nan": 0, "+inf": 31, "-inf": 1})
        self.assert_non_comparable(r)
        x = np.full(32, np.inf, dtype=np.float32)
        x[5] = f32_from_bits(0x7F800001)[0]
        r = e1.e1_q8_tensor(x, block(0x7C00, [1] * 32))
        self.assertEqual((r["bit_identical"], r["n_bit_mismatches"]), (False, 1))
        self.assertEqual(r["upstream_nonfinite"], {"nan": 1, "+inf": 31, "-inf": 0})

    def test_q8_mixed_finite_and_nonfinite_tensor(self):
        # Scale 1.0, codes 0. Upstream: 3 NaN, 2 +inf, 1 -inf, 0.25 at index 10 and -0.5 at index 40, zeros elsewhere.
        x = np.zeros(64, dtype=np.float32)
        x[[0, 1, 2]] = np.nan
        x[[3, 4]] = np.inf
        x[5] = -np.inf
        x[10], x[40] = 0.25, -0.5
        r = e1.e1_q8_tensor(x, block(ONE, []) + block(ONE, []))
        self.assert_non_comparable(r)
        self.assertEqual(r["upstream_nonfinite"], {"nan": 3, "+inf": 2, "-inf": 1})
        self.assertEqual(r["n_upstream_nonfinite"], 6)
        # D5 element-level classification and counts are kept.
        self.assertEqual(r["nonfinite_upstream_in_t2_domain"], {"nan": 3, "inf": 2, "-inf": 1})
        self.assertEqual((r["n_within_t2"], r["n_exceed_t2"]), (58, 0))
        # Mismatches: 6 non-finite, 0.25, -0.5 and the bit patterns of +0 vs +0 elsewhere agree.
        self.assertEqual((r["bit_identical"], r["n_bit_mismatches"]), (False, 8))
        # Finite-subset diagnostic, by hand: 58 elements, largest error 0.5.
        self.assertEqual((r["finite_subset_n_elements"], r["finite_subset_max_abs_error"]), (58, 0.5))
        self.assertNotIn("max_abs_error_finite_elements", r)

    def test_q8_all_nan_upstream_has_an_empty_finite_subset(self):
        r = e1.e1_q8_tensor(np.full(32, np.nan, dtype=np.float32), block(ONE, []))
        self.assert_non_comparable(r)
        self.assertEqual((r["finite_subset_n_elements"], r["finite_subset_max_abs_error"]), (0, None))

    def test_exact_tensor_nan_payloads_and_mixed_values(self):
        # Upstream NaN 0x7FC00000 against decoded 0x7FC00001: one mismatch; -inf against -inf: equal; 2.0 vs 2.5.
        up = f32_from_bits(0x7FC00000, 0xFF800000, 0x40000000, 0x3F800000)
        dec = f32_from_bits(0x7FC00001, 0xFF800000, 0x40200000, 0x3F800000)
        r = e1.e1_exact_tensor(up, dec)
        self.assertEqual((r["bit_identical"], r["n_bit_mismatches"]), (False, 2))
        self.assert_non_comparable(r)
        self.assertEqual(r["upstream_nonfinite"], {"nan": 1, "+inf": 0, "-inf": 1})
        self.assertEqual((r["finite_subset_n_elements"], r["finite_subset_max_abs_error"]), (2, 0.5))

    def test_exact_tensor_bit_identical_with_all_three_classes(self):
        up = f32_from_bits(0x7FC00123, 0x7F800000, 0xFF800000, 0x00000000)
        r = e1.e1_exact_tensor(up, up.copy())
        self.assertEqual((r["bit_identical"], r["n_bit_mismatches"]), (True, 0))
        self.assert_non_comparable(r)
        self.assertEqual(r["upstream_nonfinite"], {"nan": 1, "+inf": 1, "-inf": 1})
        text = json.loads(e1.canonical_json(r))
        self.assertEqual(text["numerical_metric_status"], "non_comparable_nonfinite")
        for f in FULL:
            self.assertIsNone(text[f])

    def test_finite_tensors_are_comparable_and_unchanged(self):
        x = np.array([1.0, 2.0] + [0.0] * 30, dtype=np.float32)
        r = e1.e1_q8_tensor(x, block(ONE, [1, 2]))
        self.assertEqual(r["numerical_metric_status"], "comparable")
        self.assertEqual(r["upstream_nonfinite"], {"nan": 0, "+inf": 0, "-inf": 0})
        self.assertEqual((r["max_abs_error"], r["absolute_frobenius_error"], r["err_frobenius"]), (0.0, 0.0, 0.0))
        self.assertEqual(r["rel_frobenius"], 0.0)
        self.assertEqual(r["null_metric_reasons"], {})
        self.assertEqual((r["bit_identical"], r["n_bit_mismatches"]), (True, 0))
        self.assertEqual(r["max_abs_error_finite_elements"], 0.0)
        self.assertNotIn("finite_subset_max_abs_error", r)
        z = e1.e1_exact_tensor(bf16([0.0, 0.0]), bf16([0.0, 3.0]))
        self.assertEqual(z["numerical_metric_status"], "comparable")
        self.assertEqual(z["null_metric_reasons"], {"rel_frobenius": "zero_reference_norm"})
        self.assertEqual(z["absolute_frobenius_error"], 3.0)

    def test_nonfinite_decode_of_finite_upstream_stays_nonfinite(self):
        # D1 OQ1 is not reopened: +inf scale against finite upstream is comparable, with an inf error.
        r = e1.e1_q8_tensor(np.full(32, 0.5, dtype=np.float32), block(0x7C00, [1] * 32))
        self.assertEqual(r["numerical_metric_status"], "comparable")
        self.assertEqual((r["max_abs_error"], r["absolute_frobenius_error"]), (np.inf, np.inf))
        self.assertEqual(r["n_upstream_nonfinite"], 0)
        self.assertEqual((r["bit_identical"], r["n_bit_mismatches"]), (False, 32))

    def test_q8_bit_identity_of_finite_values(self):
        x = np.zeros(32, dtype=np.float32)
        x[0] = -0.0  # code 0 x 1.0 decodes to +0: a bit mismatch with error 0
        x[1] = 3.0   # code 3: equal
        r = e1.e1_q8_tensor(x, block(ONE, [0, 3]))
        self.assertEqual((r["bit_identical"], r["n_bit_mismatches"], r["max_abs_error"]), (False, 1, 0.0))
        # A float64 upstream value that float32 cannot hold is compared as binary64: a mismatch.
        x64 = np.zeros(32, dtype=np.float64)
        x64[1] = 0.1
        r = e1.e1_q8_tensor(x64, block(ONE, []))
        self.assertEqual(r["n_bit_mismatches"], 1)

    def test_whole_container_with_one_nonfinite_upstream_tensor(self):
        shutil.rmtree(TMP, ignore_errors=True)
        os.makedirs(TMP)
        try:
            model = synth.tiny_model(np.random.default_rng(0), layers=2)
            w = synth.write_container(os.path.join(TMP, "d6"), model)
            c = C.open_container(os.path.join(TMP, "d6"))
            shapes = {k: v.shape for k, v in w.source.items()}
            clean = e1.e1_report(c, lambda k: w.source[k], shapes)
            target = w.q8_keys[0]
            other = next(k for k in w.source if k not in w.q8_keys)

            def upstream(k):
                a = np.array(w.source[k], dtype=np.float32, copy=True)
                if k == target:
                    a.reshape(-1)[[0, 1]] = (np.nan, -np.inf)
                if k == other:
                    a.reshape(-1)[0] = np.inf
                return a

            rep = e1.e1_report(c, upstream, shapes)
            for k, t in rep["tensors"].items():
                if k not in (target, other):
                    self.assertEqual(t, clean["tensors"][k], k)
                    self.assertEqual(t["numerical_metric_status"], "comparable")
            self.assertEqual(rep["tensors"][target]["numerical_metric_status"], "non_comparable_nonfinite")
            self.assertEqual(rep["tensors"][other]["numerical_metric_status"], "non_comparable_nonfinite")
            t = rep["total"]
            self.assertEqual(t["upstream_nonfinite"], {"nan": 1, "+inf": 1, "-inf": 1})
            self.assertEqual((t["n_upstream_nonfinite"], t["n_tensors_non_comparable_nonfinite"]), (3, 2))
            self.assertIsNone(t["max_abs_error"])
            self.assertEqual(t["null_metric_reasons"], {"max_abs_error": "nonfinite_upstream"})
            self.assertEqual(clean["total"]["n_tensors_non_comparable_nonfinite"], 0)
            self.assertEqual(clean["total"]["null_metric_reasons"], {})
            e1.canonical_json(rep)  # serialisable
        finally:
            shutil.rmtree(TMP, ignore_errors=True)


class WholeContainer(unittest.TestCase):
    def setUp(self):
        shutil.rmtree(TMP, ignore_errors=True)
        os.makedirs(TMP)
        self.model = synth.tiny_model(np.random.default_rng(0), layers=2)

    def tearDown(self):
        shutil.rmtree(TMP, ignore_errors=True)

    def report(self, name="clean", **kw):
        root = os.path.join(TMP, name)
        w = synth.write_container(root, self.model, **kw)
        c = C.open_container(root)
        shapes = {k: v.shape for k, v in w.source.items()}
        return e1.e1_report(c, lambda k: w.source[k], shapes), w

    def test_clean_container(self):
        rep, w = self.report()
        t = rep["total"]
        self.assertTrue(rep["coverage"]["ok"])
        self.assertEqual(len(rep["tensors"]), len(w.source))
        self.assertEqual(t["n_q8_tensors"], len(w.q8_keys))
        self.assertEqual(t["n_unquantised_tensors"], len(w.source) - len(w.q8_keys))
        self.assertEqual(t["n_exceed_t2"], 0)
        self.assertTrue(t["unquantised_all_bit_identical"])
        # D4: every tensor, unquantised included, carries the error statistics and the zero-norm flag.
        for key, e in rep["tensors"].items():
            for f in ("max_abs_error", "err_frobenius", "ref_frobenius", "rel_frobenius", "zero_reference_norm"):
                self.assertIn(f, e, key)
            if e["dtype"] != "Q8_0":
                self.assertTrue(e["bit_identical"])
                self.assertEqual(e["max_abs_error"], 0.0)
                self.assertEqual(e["rel_frobenius"], 0.0 if not e["zero_reference_norm"] else None)
        self.assertTrue(t["gguf_all_bit_identical"])
        self.assertEqual(t["n_elements"], sum(w.source[k].size for k in w.q8_keys))

    def test_payload_edit_resealed_is_invisible_to_the_reader_but_not_to_e1(self):
        f = next(f for f in synth.FAULTS if f.name == "control-payload-change-resealed")
        clean, _ = self.report("clean")
        rep, _ = self.report("edit", hooks=f.hooks)
        # The edit flips the low bit of one code, i.e. moves one element by one step d. E1 sees it as a changed error
        # in exactly one tensor. T2 need not flag it: a step of d lands inside 0.5623 d only when the original error
        # was already near the edge, so the T2 count is a bound check, not an identity check.
        changed = [k for k, t in rep["tensors"].items() if t != clean["tensors"][k]]
        self.assertEqual(len(changed), 1)
        self.assertNotEqual(rep["tensors"][changed[0]]["err_frobenius"], clean["tensors"][changed[0]]["err_frobenius"])

    def test_out_of_domain_scale_changes_only_its_tensor_and_exactly_the_predicted_counts(self):
        # OQ4-style check on our own report: control vs a container whose one block has a negative scale.
        clean, w = self.report("clean2")
        a = "model.layers.0.self_attn.k_proj.weight"
        raw = bytearray(w.q8_bytes[a])
        raw[0:2] = struct.pack("<H", 0xBC00)  # first block: scale -1.0, a legal bit pattern
        rep, _ = self.report("negscale", q8_override={a: bytes(raw)})
        for k, t in rep["tensors"].items():
            if k != a:
                self.assertEqual(t, clean["tensors"][k], k)
        c, m = clean["tensors"][a], rep["tensors"][a]
        self.assertEqual(m["outside_t2_domain"]["negative"], {"blocks": 1, "elements": 32})
        self.assertEqual(m["n_blocks_in_t2_domain"], c["n_blocks_in_t2_domain"] - 1)
        self.assertEqual(m["n_within_t2"] + m["n_exceed_t2"], c["n_within_t2"] + c["n_exceed_t2"] - 32)
        self.assertEqual(rep["total"]["outside_t2_domain"]["negative"], {"blocks": 1, "elements": 32})
        self.assertEqual(e1.canonical_json(rep), e1.canonical_json(self.report("negscale", q8_override={a: bytes(raw)})[0]))

    def test_transposed_shape_control_is_caught_by_coverage(self):
        f = next(f for f in synth.FAULTS if f.name == "control-q8-shape-transposed")
        rep, _ = self.report("transposed", hooks=f.hooks)
        self.assertFalse(rep["coverage"]["ok"])
        self.assertEqual(len(rep["coverage"]["shape_mismatch"]), 1)

    def test_wrong_tensor_bound_to_a_name_is_caught(self):
        # Swap the Q8_0 bytes of two same-shaped projections.
        _, w = self.report("a")
        a, b = "model.layers.0.self_attn.k_proj.weight", "model.layers.0.self_attn.v_proj.weight"
        rep, _ = self.report("swap", q8_override={a: w.q8_bytes[b], b: w.q8_bytes[a]})
        self.assertGreater(rep["total"]["n_exceed_t2"], 0)
        self.assertGreater(rep["tensors"][a]["n_exceed_t2"], 0)


if __name__ == "__main__":
    unittest.main()
