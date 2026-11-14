"""Analysis-only tests for the DEC-V3-1 link model; nothing here starts a process."""
import unittest

import link_model as lm


def position(exchanges_per_layer):
    """One measured position: per layer, a list of (request, response) body bytes."""
    calls = []
    for layer, exchanges in enumerate(exchanges_per_layer):
        for shard, (request, response) in enumerate(exchanges):
            calls.append({"kind": "expert_shard", "layer": layer, "shard": shard,
                          "transport": [{"request_bytes": request, "response_bytes": response}]})
    return {"provider_calls": calls}


def trial(point, total_ns, shape, on_schedule=True):
    return {"point": point, "total_ns": total_ns, "ffn_ns": total_ns / 2, "shape": shape,
            "relay": {"relay_on_schedule": on_schedule}}


SHAPE = {"crossings": 24, "critical_bits": 8_000_000, "body_bytes": 1_500_000}
FLOOR_NS = 100_000_000


def exact(point):
    """A position time on a perfectly additive link."""
    rtt_ns, rate_ns = lm.regressors(point, SHAPE)
    return FLOOR_NS + rtt_ns + rate_ns


class Shape(unittest.TestCase):
    def test_parallel_shards_count_one_crossing_and_only_the_largest_exchange(self):
        shape = lm.link_shape([position([[(100, 400), (100, 900)], [(100, 300)]])], layers=2)
        self.assertEqual(shape["crossings"], 2)
        self.assertEqual(shape["critical_bits"], (1000 + 400) * 8)
        self.assertEqual(shape["body_bytes"], 500 + 1000 + 400)

    def test_a_layer_without_a_remote_exchange_refuses(self):
        with self.assertRaisesRegex(ValueError, "layer 1"):
            lm.link_shape([position([[(1, 1)], []])], layers=2)

    def test_regressors_are_nanoseconds_and_unlimited_rate_costs_nothing(self):
        rtt_ns, rate_ns = lm.regressors({"rtt_ms": 5.0, "rate_bits_per_s": 1e9}, SHAPE)
        self.assertEqual(rtt_ns, 24 * 5_000_000)
        self.assertEqual(rate_ns, 8_000_000)
        self.assertEqual(lm.regressors({"rtt_ms": 0.0, "rate_bits_per_s": "inf"}, SHAPE), (0.0, 0.0))

    def test_delay_is_split_evenly_between_directions(self):
        self.assertEqual(lm.one_way_delay_us({"rtt_ms": 0.2}), 100)


class Schedule(unittest.TestCase):
    def test_every_point_sits_between_floors_and_odd_repeats_reverse(self):
        points = lm.frozen_points()
        order = lm.schedule(points, repeats=2)
        kinds = [p["kind"] for p in order]
        self.assertEqual(kinds[0], "floor")
        self.assertEqual(kinds[-1], "floor")
        runs = "".join("F" if k == "floor" else "p" for k in kinds).split("F")
        self.assertTrue(all(len(run) <= lm.POINTS_PER_BRACKET for run in runs))
        first = [p for p in order if p["repeat"] == 0 and p["kind"] != "floor"]
        second = [p for p in order if p["repeat"] == 1 and p["kind"] != "floor"]
        strip = lambda ps: [(p["kind"], p["rtt_ms"], p["rate_bits_per_s"]) for p in ps]
        self.assertEqual(strip(first), strip(points))
        self.assertEqual(strip(second), strip(points)[::-1])

    def test_frozen_points_identify_both_terms_and_hold_joint_points_out(self):
        points = lm.frozen_points()
        axis = [p for p in points if p["kind"] == "axis"]
        self.assertTrue(any(p["rtt_ms"] > 0 for p in axis))
        self.assertTrue(any(p["rate_bits_per_s"] != "inf" for p in axis))
        self.assertTrue(all(p["rtt_ms"] == 0 or p["rate_bits_per_s"] == "inf" for p in axis))
        self.assertTrue(all(p["rtt_ms"] > 0 and p["rate_bits_per_s"] != "inf"
                            for p in points if p["kind"] == "joint"))

    def test_empty_schedule_refuses(self):
        with self.assertRaises(ValueError):
            lm.schedule([], repeats=1)


class RelayGate(unittest.TestCase):
    def test_lateness_is_judged_per_position_against_the_timing_gate(self):
        stats = [{"bytes": 10, "chunks": 4, "late_sum_ns": 400, "late_max_ns": 300},
                 {"bytes": 30, "chunks": 4, "late_sum_ns": 1600, "late_max_ns": 900}]
        gate = lm.relay_gate(stats, positions=2, floor_total_ns=100_000)
        self.assertEqual(gate["relay_late_per_position_ns"], 1000)
        self.assertEqual(gate["relay_late_mean_ns"], 250)
        self.assertEqual(gate["relay_late_max_ns"], 900)
        self.assertEqual(gate["relay_bytes"], 40)
        self.assertTrue(gate["relay_on_schedule"])
        self.assertFalse(lm.relay_gate(stats, 2, floor_total_ns=99_999)["relay_on_schedule"])

    def test_a_silent_relay_is_an_error_not_a_pass(self):
        with self.assertRaises(ValueError):
            lm.relay_gate([], 1, 1)
        with self.assertRaises(ValueError):
            lm.relay_gate([{"bytes": 0, "chunks": 0, "late_sum_ns": 0, "late_max_ns": 0}], 1, 1)


class Fit(unittest.TestCase):
    def run_schedule(self, measure):
        return [trial(p, FLOOR_NS if p["kind"] == "floor" else measure(p), SHAPE)
                for p in lm.schedule(lm.frozen_points(), repeats=1)]

    def test_an_additive_link_recovers_unit_coefficients_and_predicts_joint_points(self):
        report = lm.analyse(self.run_schedule(exact))
        model = report["model"]
        self.assertAlmostEqual(model["rtt_coefficient"], 1.0, places=9)
        self.assertAlmostEqual(model["rate_coefficient"], 1.0, places=9)
        self.assertAlmostEqual(model["intercept_ns"], 0.0, delta=1e-3)
        self.assertEqual(len(report["held_out"]), len(lm.JOINT_POINTS))
        for held in report["held_out"]:
            self.assertAlmostEqual(held["error_fraction_of_position"], 0.0, places=9)

    def test_an_interaction_term_shows_up_as_held_out_error_not_in_the_axis_fit(self):
        def interacting(point):
            rtt_ns, rate_ns = lm.regressors(point, SHAPE)
            return exact(point) + (FLOOR_NS if rtt_ns and rate_ns else 0)
        report = lm.analyse(self.run_schedule(interacting))
        self.assertAlmostEqual(report["model"]["rtt_coefficient"], 1.0, places=9)
        self.assertTrue(all(h["error_fraction_of_position"] > 0.1 for h in report["held_out"]))

    def test_direct_points_are_reported_but_never_fitted(self):
        report = lm.analyse(self.run_schedule(lambda p: exact(p) - (5 if p["kind"] == "direct" else 0)))
        direct = [o for o in report["observations"] if o["point"]["kind"] == "direct"]
        self.assertEqual([o["delta_ns"] for o in direct], [-5])
        self.assertEqual(report["model"]["observations"],
                         len(lm.RTT_AXIS_MS) + len(lm.RATE_AXIS_BITS_PER_S))

    def test_disagreeing_floors_void_the_points_between_them(self):
        trials = self.run_schedule(exact)
        last_floor = max(i for i, t in enumerate(trials) if t["point"]["kind"] == "floor")
        trials[last_floor]["total_ns"] = FLOOR_NS * (1 + 3 * lm.GATE)
        report = lm.analyse(trials)
        self.assertGreater(report["void"], 0)
        voided = [o for o in report["observations"] if not o["valid"]]
        self.assertTrue(all(o["floor_drift"] > lm.GATE for o in voided))
        self.assertTrue(all(o["index"] > last_floor - 1 - lm.POINTS_PER_BRACKET for o in voided))

    def test_a_late_relay_voids_its_point(self):
        trials = self.run_schedule(exact)
        target = next(t for t in trials if t["point"]["kind"] == "joint")
        target["relay"]["relay_on_schedule"] = False
        report = lm.analyse(trials)
        self.assertEqual(report["void"], 1)
        self.assertEqual(len(report["held_out"]), len(lm.JOINT_POINTS) - 1)

    def test_too_few_valid_axis_points_refuse_a_model_instead_of_inventing_one(self):
        trials = [t for t in self.run_schedule(exact)
                  if t["point"]["kind"] in ("floor", "joint")]
        report = lm.analyse(trials)
        self.assertIsNone(report["model"])
        self.assertIn("model_refused", report)

    def test_a_point_without_a_closing_floor_refuses(self):
        trials = self.run_schedule(exact)[:-1]
        with self.assertRaisesRegex(ValueError, "floor before and after"):
            lm.analyse(trials)

    def test_solve_refuses_a_singular_system(self):
        with self.assertRaisesRegex(ValueError, "identify"):
            lm.solve([[1.0, 2.0], [2.0, 4.0]], [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
