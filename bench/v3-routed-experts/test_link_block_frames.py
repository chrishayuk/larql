"""Analysis-only tests for the LINK-2 block-frame addendum; nothing here starts a process."""
import copy
import unittest

import link_block_frames_model as bm
import link_model as lm
import link_pair_model as pm

HEADER = 8
RELAY_SHA = "7b1097f8" + "0" * 56
BASELINE_NS = 35_000
# GPT-OSS 20B, one worker: the frame definition of docs/ffn/v3-expert-rows.md.
HIDDEN, EXPERTS_PER_ROW, FRAME_HEADER, PAIR_PREFIX = 2880, 4, 44, 8


def pairs_at(difference):
    return [{"baseline_ns": [BASELINE_NS] * (2 * pm.BLOCK),
             "shaped_ns": [BASELINE_NS + difference] * (2 * pm.BLOCK)} for _ in range(pm.PAIRS)]


def judged_run(error_ns=0.0):
    return [pm.judge(point, pairs_at(bm.expected_ns(point, HEADER) + error_ns), bm.expected_ns(point, HEADER))
            for point in bm.frozen_points()]


def receipt(judged, started="2026-11-14T00:00:00Z", **changes):
    base = {"schema": bm.SCHEMA, "protocol": bm.ADDENDUM, "rules": pm.REVISION, "kind": "qualifying",
            "counts_toward_qualification": True, "frozen": pm.frozen_parameters(),
            "geometry": {str(w): list(s) for w, s in bm.geometry().items()},
            "binary_sha256": {"v3_link_shaper": RELAY_SHA}, "host_before": {"utc": started},
            "points": judged, "run_passes": True}
    base.update(changes)
    return base


class Geometry(unittest.TestCase):
    def test_sizes_follow_the_multi_row_frame_definition(self):
        for width, (request, response) in bm.geometry().items():
            pairs = width * EXPERTS_PER_ROW
            self.assertEqual(request, FRAME_HEADER + width * (PAIR_PREFIX + 4 * HIDDEN) + 4 * pairs)
            self.assertEqual(response, FRAME_HEADER + pairs * (PAIR_PREFIX + 4 * HIDDEN))

    def test_points_are_null_then_each_round_trip_for_each_width(self):
        points = bm.frozen_points()
        self.assertEqual(len(points), len(bm.WIDTHS) * (1 + len(bm.RTT_MS)))
        self.assertEqual([p["width"] for p in points[::4]], list(bm.WIDTHS))
        for start in range(0, len(points), 4):
            group = points[start:start + 4]
            self.assertEqual([p["kind"] for p in group], ["null", "axis", "axis", "axis"])
            self.assertEqual([p["rtt_ms"] for p in group], [0.0, *bm.RTT_MS])
            self.assertTrue(all(p["rate_bits_per_s"] == lm.UNLIMITED for p in group))

    def test_expected_cost_is_the_round_trip_whatever_the_width(self):
        for point in bm.frozen_points():
            self.assertEqual(bm.expected_ns(point, HEADER), point["rtt_ms"] * 1e6)


class Verdict(unittest.TestCase):
    def test_an_exact_run_passes_and_qualifies_twice(self):
        judged = judged_run()
        self.assertTrue(bm.run_verdict(judged))
        bm.check_qualification([receipt(judged), receipt(judged, started="2026-11-14T01:00:00Z")], RELAY_SHA)

    def test_one_point_beyond_tolerance_fails_the_run(self):
        judged = judged_run()
        point = bm.frozen_points()[-1]
        expected = bm.expected_ns(point, HEADER)
        beyond = max(pm.TOLERANCE_RELATIVE * expected, pm.TOLERANCE_ABSOLUTE_NS) + 1
        judged[-1] = pm.judge(point, pairs_at(expected + beyond), expected)
        self.assertFalse(bm.run_verdict(judged))

    def test_a_null_point_is_held_to_the_null_bound(self):
        point = bm.frozen_points()[0]
        self.assertTrue(pm.judge(point, pairs_at(pm.NULL_BOUND_NS), 0.0)["passes"])
        self.assertFalse(pm.judge(point, pairs_at(pm.NULL_BOUND_NS + 1), 0.0)["passes"])

    def test_points_missing_or_out_of_order_are_refused_not_judged(self):
        judged = judged_run()
        for broken in (judged[:-1], judged[::-1], judged[4:] + judged[:4]):
            with self.assertRaises(ValueError):
                bm.run_verdict(broken)


class Qualification(unittest.TestCase):
    def refused(self, *receipts):
        with self.assertRaises(ValueError):
            bm.check_qualification(list(receipts), RELAY_SHA)

    def test_everything_but_two_distinct_passing_qualifying_runs_is_refused(self):
        judged = judged_run()
        good = receipt(judged)
        later = receipt(judged, started="2026-11-14T01:00:00Z")
        self.refused(good)
        self.refused(good, copy.deepcopy(good))
        self.refused(good, receipt(judged, started="x", kind="diagnostic", counts_toward_qualification=False))
        self.refused(good, receipt(judged, started="x", rules="LINK-2 revision 2"))
        self.refused(good, receipt(judged, started="x", protocol=pm.REVISION))
        self.refused(good, receipt(judged, started="x", binary_sha256={"v3_link_shaper": "other"}))
        self.refused(good, receipt(judged, started="x", run_passes=False))
        self.refused(good, receipt(judged_run(error_ns=10 * pm.TOLERANCE_ABSOLUTE_NS), started="x"))
        other_sizes = dict(good["geometry"], **{"8": [1, 1]})
        self.refused(good, receipt(judged, started="x", geometry=other_sizes))
        fewer_pairs = dict(pm.frozen_parameters(), pairs=pm.PAIRS // 2)
        self.refused(good, receipt(judged, started="x", frozen=fewer_pairs))
        bm.check_qualification([good, later], RELAY_SHA)

    def test_a_revision_3_receipt_is_not_a_block_frame_receipt(self):
        with self.assertRaises(ValueError):
            bm.check_qualifying_receipt({"schema": pm.SCHEMA, "protocol": pm.REVISION}, RELAY_SHA)


if __name__ == "__main__":
    unittest.main()
