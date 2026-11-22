"""Synthetic tests for GW-COMPRESS-1 band admission (no model, no capture files)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gwcompress_band_admit as badm  # noqa: E402

L, WIN, RID = 4, [1, 2], "capital|P2|XX"


class FakeCaps:
    def __init__(self):
        self.entries = {RID: {"vector_sha256": {f"{l}|{p}": f"nat{l}{p}" for l in range(L) for p in WIN}}}


def fixture(lo=1, hi=2):
    vec = {f"{l}|{p}": f"v{l}{p}" for l in range(lo, hi + 1) for p in WIN}
    plan = {"layers": L, "holdout": sorted(badm.SPENT_HOLDOUT),
            "runs": {RID: {"country": "XX", "tokens": [0, 1, 2, 3], "window": WIN}},
            "targets": [{"id": "t", "run": RID, "arms": [{"name": "S-band", "from_layer": lo + 1, "to_layer": hi,
                                                          "target_span": WIN, "vectors": vec}]}]}
    index = {v: {"sha256": "sha" + v, "recipe_sha256": "r" + v} for v in vec.values()}
    sc = {f"{l}|{p}": (f"nat{l}{p}" if l < lo else ("sha" + vec[f"{l}|{p}"] if l <= hi else f"new{l}{p}"))
          for l in range(L) for p in WIN}
    n = (hi - lo + 1) * len(WIN)
    rec = {"id": "t", "natural": {"span_carrier_sha256": {f"{l}|{p}": f"nat{l}{p}" for l in range(L) for p in WIN}},
           "arms": {"S-band": {"band": [lo, hi], "replacements": n, "firings": n,
                               "vectors": {k: "sha" + v for k, v in vec.items()},
                               "recipes": {"r" + v: 1 for v in vec.values()}, "span_carrier_sha256": sc,
                               "propagation_fixpoint": {"bit_exact": True, "replayed_addresses": (L - hi - 1) * len(WIN)}}}}
    return plan, index, rec


class AdmitRecords(unittest.TestCase):
    def check(self, mutate=None, **kw):
        plan, index, rec = fixture(**kw)
        if mutate:
            mutate(plan, rec)
        return badm.admit_records(plan, index, [rec], FakeCaps())

    def test_clean(self):
        self.assertEqual(self.check(), [])
        self.assertEqual(self.check(lo=0, hi=3), [])

    def test_below_band_not_natural(self):
        f = self.check(lambda p, r: r["arms"]["S-band"]["span_carrier_sha256"].__setitem__("0|1", "x"))
        self.assertTrue(any("below the band" in x for x in f))

    def test_in_band_not_injected(self):
        f = self.check(lambda p, r: r["arms"]["S-band"]["span_carrier_sha256"].__setitem__("2|2", "x"))
        self.assertTrue(any("in the band" in x for x in f))

    def test_missing_fixpoint(self):
        f = self.check(lambda p, r: r["arms"]["S-band"].__setitem__("propagation_fixpoint", None))
        self.assertTrue(any("fixpoint" in x for x in f))

    def test_counts(self):
        f = self.check(lambda p, r: r["arms"]["S-band"].__setitem__("firings", 5))
        self.assertTrue(any("counts" in x for x in f))

    def test_natural_differs_from_capture(self):
        f = self.check(lambda p, r: r["natural"]["span_carrier_sha256"].__setitem__("3|2", "x"))
        self.assertTrue(any("differs from the capture" in x for x in f))


class AdmitPlan(unittest.TestCase):
    def test_spent_holdout_omitted_and_invalid_band(self):
        plan, index, _ = fixture()
        plan["holdout"] = plan["holdout"][1:]
        plan["targets"][0]["arms"][0]["to_layer"] = 0
        f = badm.admit_plan(plan, index, None, ["XX"])
        self.assertIn("plan holdout list omits spent holdout countries", f)
        self.assertTrue(any("invalid band" in x for x in f))

    def test_spent_country_run(self):
        plan, index, _ = fixture()
        plan["runs"][RID]["country"] = "TO"
        plan["targets"][0]["arms"] = []
        self.assertTrue(any("holdout country" in x for x in badm.admit_plan(plan, index, None, ["XX"])))


if __name__ == "__main__":
    unittest.main()
