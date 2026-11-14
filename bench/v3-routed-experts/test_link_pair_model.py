"""Analysis-only tests for LINK-2's decision rules; nothing here starts a process."""
import random
import statistics
import unittest

import link_model as lm
import link_pair_model as pm

HEADER, REQUEST, RESPONSE = 8, 11_576, 46_136
BASELINE_NS = 35_000


def pairs_with(differences, baseline=BASELINE_NS):
    """Pairs whose pair difference is exactly each given value."""
    return [{"baseline_ns": [baseline] * (2 * pm.BLOCK), "shaped_ns": [baseline + d] * (2 * pm.BLOCK)}
            for d in differences]


def noisy(expected, spread, seed, count=pm.PAIRS):
    generator = random.Random(seed)
    return [expected + generator.uniform(-spread, spread) for _ in range(count)]


POINT = {"kind": "axis", "rtt_ms": 1.0, "rate_bits_per_s": lm.UNLIMITED}
EXPECTED = 1_000_000.0
NULL_SEED = 1


class Frozen(unittest.TestCase):
    def test_points_are_link1s_nine_then_the_null_point(self):
        points = pm.frozen_points()
        self.assertEqual(len(points), 10)
        self.assertEqual([p["kind"] for p in points][-1], "null")
        self.assertNotIn("direct", [p["kind"] for p in points])
        self.assertEqual(points[:-1], [p for p in lm.frozen_points() if p["kind"] != "direct"])

    def test_expected_cost_is_round_trip_plus_frame_bits_over_rate(self):
        delay = {"kind": "axis", "rtt_ms": 0.2, "rate_bits_per_s": lm.UNLIMITED}
        rate = {"kind": "axis", "rtt_ms": 0.0, "rate_bits_per_s": 1e9}
        self.assertEqual(pm.expected_ns(delay, REQUEST, RESPONSE, HEADER), 200_000)
        self.assertAlmostEqual(pm.expected_ns(rate, REQUEST, RESPONSE, HEADER),
                               (HEADER + REQUEST + RESPONSE) * 8, places=6)
        self.assertEqual(pm.expected_ns(pm.NULL, REQUEST, RESPONSE, HEADER), 0)


class Differences(unittest.TestCase):
    def test_a_pair_difference_is_median_shaped_minus_median_baseline(self):
        pair = {"baseline_ns": [10] * 19 + [10_000], "shaped_ns": [110] * 19 + [5]}
        self.assertEqual(pm.pair_differences([pair]), [100])

    def test_malformed_pairs_refuse(self):
        with self.assertRaises(ValueError):
            pm.pair_differences([])
        with self.assertRaises(ValueError):
            pm.pair_differences([{"baseline_ns": [1] * pm.BLOCK, "shaped_ns": [1] * (2 * pm.BLOCK)}])

    def test_common_mode_slowdown_cancels_inside_a_pair(self):
        # Each pair runs at its own host speed; the added cost is constant.
        generator = random.Random(3)
        pairs = []
        for _ in range(pm.PAIRS):
            host = generator.choice([30_000, 45_000, 60_000])
            pairs.append({"baseline_ns": [host] * (2 * pm.BLOCK),
                          "shaped_ns": [host + 200_000] * (2 * pm.BLOCK)})
        self.assertEqual(set(pm.pair_differences(pairs)), {200_000})
        judged = pm.judge({"kind": "axis", "rtt_ms": 0.2, "rate_bits_per_s": lm.UNLIMITED}, pairs, 200_000)
        self.assertTrue(judged["passes"])
        self.assertTrue(judged["pairing_removed_common_mode"])
        self.assertEqual(judged["paired_spread_ns"], 0)
        self.assertGreater(judged["repaired_spread_ns"], 0)


class Correlation(unittest.TestCase):
    def test_autocorrelation_of_an_alternating_and_a_trending_series(self):
        alternating = [1.0, -1.0] * 50
        self.assertAlmostEqual(pm.autocorrelation(alternating, 1), -0.99, places=2)
        self.assertAlmostEqual(pm.autocorrelation(alternating, 2), 0.98, places=2)
        self.assertGreater(pm.autocorrelation(list(range(100)), 5), 0.8)
        self.assertEqual(pm.autocorrelation([4.0] * 100, 3), 0.0)

    def test_block_bootstrap_is_wider_than_independent_for_a_persistent_series(self):
        # Long runs of the same level: neighbours are strongly correlated.
        generator = random.Random(11)
        series = []
        while len(series) < pm.PAIRS:
            series.extend([generator.uniform(-40_000, 40_000)] * 10)
        series = series[:pm.PAIRS]
        block = pm.block_bootstrap_interval(series, 10, pm.BOOTSTRAP_RESAMPLES, pm.BOOTSTRAP_SEED)
        independent = pm.block_bootstrap_interval(series, 1, pm.BOOTSTRAP_RESAMPLES, pm.BOOTSTRAP_SEED)
        self.assertGreater(block[1] - block[0], independent[1] - independent[0])

    def test_bootstrap_is_reproducible_brackets_the_median_and_bounds_its_block(self):
        series = noisy(0, 10_000, seed=5)
        first = pm.block_bootstrap_interval(series, pm.BOOTSTRAP_BLOCK, 500, 9)
        self.assertEqual(first, pm.block_bootstrap_interval(series, pm.BOOTSTRAP_BLOCK, 500, 9))
        self.assertLessEqual(first[0], statistics.median(series))
        self.assertGreaterEqual(first[1], statistics.median(series))
        for bad in (0, len(series) + 1):
            with self.assertRaises(ValueError):
                pm.block_bootstrap_interval(series, bad, 10, 1)


class Rules(unittest.TestCase):
    def test_an_accurate_tight_uncorrelated_point_passes(self):
        judged = pm.judge(POINT, pairs_with(noisy(EXPECTED, 8_000, seed=1)), EXPECTED)
        self.assertTrue(judged["accurate"] and judged["precise"] and judged["passes"])
        self.assertEqual(judged["tolerance_ns"], 50_000)
        self.assertEqual(sorted(judged["autocorrelation"], key=int), [str(lag) for lag in range(1, 31)])
        self.assertNotIn("autocorrelation_within_bound", judged)
        self.assertEqual(judged["judged_half_width_ns"], 2 * judged["interval_half_width_ns"])

    def test_tolerance_is_the_larger_of_five_percent_and_fifty_microseconds(self):
        large = 20_000_000.0
        judged = pm.judge({"kind": "axis", "rtt_ms": 20.0, "rate_bits_per_s": lm.UNLIMITED},
                          pairs_with(noisy(large + 900_000, 5_000, seed=2)), large)
        self.assertEqual(judged["tolerance_ns"], 1_000_000)
        self.assertTrue(judged["accurate"])
        off = pm.judge(POINT, pairs_with(noisy(EXPECTED + 60_000, 5_000, seed=2)), EXPECTED)
        self.assertFalse(off["accurate"])
        self.assertFalse(off["passes"])

    def test_inside_tolerance_with_a_wide_interval_does_not_pass(self):
        judged = pm.judge(POINT, pairs_with(noisy(EXPECTED, 400_000, seed=4)), EXPECTED)
        self.assertTrue(judged["accurate"])
        self.assertGreater(judged["judged_half_width_ns"], 25_000)
        self.assertFalse(judged["precise"])
        self.assertFalse(judged["passes"])

    def test_correlation_is_reported_and_never_decides_a_point(self):
        # A slow wave of small amplitude: strongly correlated at every lag,
        # accurate and tight. Revision 2 failed this on its autocorrelation
        # gate; revision 3 reports the correlation and judges the interval.
        import math
        wave = [EXPECTED + 2_000 * math.sin(2 * math.pi * i / 50) for i in range(pm.PAIRS)]
        judged = pm.judge(POINT, pairs_with(wave), EXPECTED)
        self.assertGreater(abs(judged["autocorrelation"]["5"]), 0.5)
        self.assertTrue(judged["accurate"] and judged["precise"] and judged["passes"])

    def test_the_half_width_is_doubled_before_it_is_judged(self):
        # Chosen so the raw half-width is under the limit and its double is over.
        for spread in range(60_000, 400_000, 10_000):
            judged = pm.judge(POINT, pairs_with(noisy(EXPECTED, spread, seed=4)), EXPECTED)
            limit = pm.PRECISION_SHARE * judged["tolerance_ns"]
            if judged["interval_half_width_ns"] <= limit < judged["judged_half_width_ns"]:
                self.assertTrue(judged["accurate"])
                self.assertFalse(judged["precise"])
                return
        self.fail("no spread separated the raw and doubled half-widths")

    def test_the_null_point_is_held_to_twenty_five_microseconds(self):
        self.assertTrue(pm.judge(pm.NULL, pairs_with(noisy(10_000, 3_000, seed=NULL_SEED)), 0.0)["passes"])
        asymmetric = pm.judge(pm.NULL, pairs_with(noisy(30_000, 3_000, seed=NULL_SEED)), 0.0)
        self.assertEqual(asymmetric["tolerance_ns"], pm.NULL_BOUND_NS)
        self.assertFalse(asymmetric["accurate"])
        loose = pm.judge(pm.NULL, pairs_with(noisy(0, 400_000, seed=NULL_SEED)), 0.0)
        self.assertGreater(loose["judged_half_width_ns"], pm.PRECISION_SHARE * pm.NULL_BOUND_NS)
        self.assertFalse(loose["precise"])

    def test_the_frozen_parameters_are_revision_three(self):
        self.assertEqual((pm.PAIRS, pm.BLOCK, pm.WARMUP), (200, 10, 20))
        self.assertEqual((pm.BOOTSTRAP_BLOCK, pm.BOOTSTRAP_RESAMPLES, pm.HALF_WIDTH_FACTOR), (20, 2000, 2.0))
        self.assertEqual((pm.TOLERANCE_RELATIVE, pm.TOLERANCE_ABSOLUTE_NS, pm.NULL_BOUND_NS), (0.05, 50_000, 25_000))
        self.assertEqual(pm.REVISION, "LINK-2 revision 3")
        self.assertFalse(hasattr(pm, "AUTOCORRELATION_BOUND"))

    def test_pairing_is_not_claimed_when_it_did_not_help(self):
        # Independent noise on the difference: re-pairing changes nothing.
        judged = pm.judge(POINT, pairs_with(noisy(EXPECTED, 8_000, seed=8)), EXPECTED)
        self.assertFalse(judged["pairing_removed_common_mode"])

    def test_a_point_needs_exactly_the_frozen_number_of_pairs(self):
        with self.assertRaisesRegex(ValueError, "exactly"):
            pm.judge(POINT, pairs_with(noisy(EXPECTED, 1_000, seed=1, count=pm.PAIRS - 1)), EXPECTED)

    def test_a_run_passes_only_if_every_frozen_point_passes_in_order(self):
        points = pm.frozen_points()
        judged = [{"point": p, "passes": True} for p in points]
        self.assertTrue(pm.run_verdict(judged))
        judged[3]["passes"] = False
        self.assertFalse(pm.run_verdict(judged))
        with self.assertRaisesRegex(ValueError, "frozen points"):
            pm.run_verdict(judged[:-1])
        with self.assertRaisesRegex(ValueError, "frozen points"):
            pm.run_verdict(judged[::-1])


class Receipts(unittest.TestCase):
    RELAY = "a" * 64

    def receipt(self, **changes):
        points = [{"point": p, "passes": True} for p in pm.frozen_points()]
        receipt = {"schema": pm.SCHEMA, "protocol": pm.REVISION, "kind": "qualifying",
                   "counts_toward_qualification": True, "frozen": pm.frozen_parameters(),
                   "binary_sha256": {"v3_link_shaper": self.RELAY}, "run_passes": True, "points": points}
        receipt.update(changes)
        return receipt

    def test_a_passing_qualifying_receipt_of_this_revision_and_relay_is_accepted(self):
        pm.check_qualifying_receipt(self.receipt(), self.RELAY)

    def test_everything_else_is_refused_by_name(self):
        failed_point = [{"point": p, "passes": i != 4} for i, p in enumerate(pm.frozen_points())]
        cases = [
            ("diagnostic", self.receipt(kind="diagnostic", counts_toward_qualification=False)),
            ("diagnostic", self.receipt(counts_toward_qualification=False)),
            ("revision", self.receipt(protocol="LINK-2 revision 2")),
            ("revision", self.receipt(schema="larql.v3.link-calibration.v2")),
            ("pairs", self.receipt(frozen=dict(pm.frozen_parameters(), pairs=100))),
            ("bootstrap_block", self.receipt(frozen=dict(pm.frozen_parameters(), bootstrap_block=5))),
            ("tolerance_absolute_ns", self.receipt(frozen=dict(pm.frozen_parameters(), tolerance_absolute_ns=80_000))),
            ("different relay", self.receipt(binary_sha256={"v3_link_shaper": "b" * 64})),
            ("did not pass", self.receipt(run_passes=False)),
            ("did not pass", self.receipt(points=failed_point)),
        ]
        for needle, receipt in cases:
            with self.assertRaisesRegex(ValueError, needle):
                pm.check_qualifying_receipt(receipt, self.RELAY)
        with self.assertRaisesRegex(ValueError, "frozen points"):
            pm.check_qualifying_receipt(self.receipt(points=failed_point[:-1]), self.RELAY)


if __name__ == "__main__":
    unittest.main()
