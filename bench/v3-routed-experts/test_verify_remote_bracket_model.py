"""VERIFY-REMOTE-1 revision 2: the bracket study's schedule, measures, selection rule and driver loop."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import unittest

import verify_remote_bracket_model as bs
import verify_remote_model as vr
from test_verify_remote_model import BASE_NS, fake_relay, fake_trial, reference_ids

PINNED_BRACKET_ORDERS_SHA256 = "817e21c467e249a7f57f09547c65d186e4ceed96ab0bf975552526d851a7a0d5"


def obs(entry, per_position_ns, defects=()):
    """An observation of `entry` whose time per timed id is `per_position_ns`."""
    return dict(entry, summary={"per_position_ns": per_position_ns}, ledger_defects=list(defects),
                frame_defects=[], relay=None, suspensions=[])


def study(drift_by_candidate, filler_spread=0.0, defective_unit=None):
    """Observations for the whole schedule. `drift_by_candidate[length](unit)` is the
    closing bracket's relative change; the filler's four times spread by `filler_spread`."""
    out = []
    filler_index = Counter()
    for entry in bs.schedule():
        unit = entry["unit"]
        defects = ["a frame defect"] if unit == defective_unit and entry["role"] == bs.ROLE_FILLER else []
        if entry["role"] == vr.ROLE_BRACKET:
            base = 200e6 * entry["candidate"] / 16
            change = drift_by_candidate[entry["candidate"]](unit) if entry["bracket"] == "after" else 0.0
            out.append(obs(entry, base * (1 + change), defects))
        else:
            i = filler_index[unit]
            filler_index[unit] += 1
            out.append(obs(entry, 300e6 * (1 + filler_spread * i / 3), defects))
    return out


def constant(value):
    return lambda unit: value


class Schedule(unittest.TestCase):
    def test_eight_stand_in_units_of_three_brackets_each_side_around_the_filler(self):
        order = bs.schedule()
        self.assertEqual(len(order), bs.STAND_INS * (2 * len(bs.CANDIDATES) + len(bs.FILLER_ARMS)))
        for unit in range(bs.STAND_INS):
            members = [e for e in order if e["unit"] == unit]
            roles = [(e["role"], e["bracket"]) for e in members]
            self.assertEqual(roles, [("bracket", "before")] * 3 + [("filler", None)] * 4
                             + [("bracket", "after")] * 3)
            before = [e["candidate"] for e in members if e["bracket"] == "before"]
            after = [e["candidate"] for e in members if e["bracket"] == "after"]
            self.assertEqual(before, after)
            self.assertEqual(sorted(before), list(bs.CANDIDATES))
            fillers = [e for e in members if e["role"] == bs.ROLE_FILLER]
            self.assertEqual([e["arm"] for e in fillers], list(vr.ARMS))
            self.assertTrue(all(e["workload"] == "explain" and e["rtt_ms"] == 20.0
                                and e["generate"] == vr.GENERATE for e in fillers))
        brackets = [e for e in order if e["role"] == vr.ROLE_BRACKET]
        self.assertTrue(all(e["arm"] == vr.SEQUENTIAL and e["workload"] == "prose_free"
                            and e["rtt_ms"] == 0.0 and e["generate"] == e["candidate"] for e in brackets))
        self.assertEqual(len({e["label"] for e in order}), len(order))

    def test_bracket_orders_are_seeded_and_pinned(self):
        orders = bs.bracket_orders()
        self.assertEqual(orders, bs.bracket_orders())
        digest = hashlib.sha256(json.dumps(orders).encode()).hexdigest()
        self.assertEqual(digest, PINNED_BRACKET_ORDERS_SHA256)
        # Each candidate opens first in at least one unit: no fixed position.
        self.assertEqual({o[0] for o in orders}, set(bs.CANDIDATES))

    def test_the_study_pauses_only_before_a_stand_in_unit(self):
        points = [e for e in bs.schedule() if bs.is_pause_point(e)]
        self.assertEqual([e["unit"] for e in points], list(range(bs.STAND_INS)))
        self.assertTrue(all(e["label"] == bs.first_labels()[e["unit"]] for e in points))


class Selection(unittest.TestCase):
    def select(self, **drifts):
        by = {length: drifts.get(f"c{length}", constant(0.0)) for length in bs.CANDIDATES}
        return bs.select(bs.unit_readings(study(by)))

    def test_seven_of_eight_within_qualifies_and_six_does_not(self):
        seven = lambda unit: 0.02 if unit == 3 else 0.005
        six = lambda unit: 0.02 if unit in (3, 5) else 0.005
        selection = self.select(c16=six, c32=seven, c64=constant(0.0))
        reading = {c["candidate"]: c for c in selection["candidates"]}
        self.assertFalse(reading[16]["qualifies"])
        self.assertEqual(reading[16]["within_gate"], 6)
        self.assertTrue(reading[32]["qualifies"])
        self.assertEqual(reading[32]["within_gate"], 7)
        self.assertEqual(selection["selected"], 32)

    def test_exactly_the_gate_is_within_it_and_just_over_is_not(self):
        at = self.select(c16=constant(0.01))
        self.assertEqual(at["selected"], 16)
        over = self.select(c16=constant(0.0101), c32=constant(0.0101), c64=constant(0.0101))
        self.assertIsNone(over["selected"])

    def test_the_shortest_qualifying_length_is_selected_even_if_a_longer_one_is_steadier(self):
        selection = self.select(c16=constant(0.009), c32=constant(0.001), c64=constant(0.0))
        self.assertEqual(selection["selected"], 16)

    def test_no_qualifying_length_selects_nothing_and_reports_every_candidate(self):
        selection = self.select(c16=constant(0.05), c32=constant(0.03), c64=constant(0.02))
        self.assertIsNone(selection["selected"])
        self.assertTrue(selection["review_needed"])
        self.assertIn("no length is selected", selection["outcome"])
        self.assertEqual([c["candidate"] for c in selection["candidates"]], list(bs.CANDIDATES))
        self.assertTrue(all(len(c["drifts"]) == bs.STAND_INS for c in selection["candidates"]))

    def test_a_unit_lost_to_a_defect_is_not_a_drift_within_the_gate(self):
        by = {16: lambda unit: 0.02 if unit == 0 else 0.0, 32: constant(0.0), 64: constant(0.0)}
        readings = bs.unit_readings(study(by, defective_unit=5))
        self.assertFalse(readings[5]["counted"])
        self.assertIn("a frame defect", readings[5]["reasons"][0])
        selection = bs.select(readings)
        reading = {c["candidate"]: c for c in selection["candidates"]}
        self.assertEqual(reading[16]["within_gate"], 6)
        self.assertFalse(reading[16]["qualifies"])
        self.assertEqual(reading[32]["within_gate"], 7)
        self.assertEqual(selection["selected"], 32)

    def test_missed_disturbances_are_counted_beside_the_rule_not_in_it(self):
        by = {length: constant(0.0) for length in bs.CANDIDATES}
        readings = bs.unit_readings(study(by, filler_spread=0.05))
        selection = bs.select(readings)
        self.assertEqual(selection["selected"], 16)
        self.assertEqual(selection["selected_missed_disturbances"], list(range(bs.STAND_INS)))
        self.assertTrue(selection["review_needed"])
        quiet = bs.select(bs.unit_readings(study(by, filler_spread=0.005)))
        self.assertEqual(quiet["selected_missed_disturbances"], [])
        self.assertFalse(quiet["review_needed"])

    def test_a_disturbance_the_bracket_sees_is_not_missed(self):
        by = {16: constant(0.03), 32: constant(0.0), 64: constant(0.0)}
        reading = {c["candidate"]: c for c in bs.select(bs.unit_readings(study(by, filler_spread=0.05)))["candidates"]}
        self.assertEqual(reading[16]["missed_disturbances"], [])
        self.assertEqual(len(reading[32]["missed_disturbances"]), bs.STAND_INS)

    def test_filler_spread_is_range_over_mean_and_needs_all_four(self):
        fillers = [{"summary": {"per_position_ns": t}} for t in (100, 102, 101, 99)]
        self.assertAlmostEqual(bs.filler_spread(fillers), 3 / 100.5)
        self.assertIsNone(bs.filler_spread(fillers[:3]))


class Execute(unittest.TestCase):
    """The study's run loop end to end, with every process replaced by a fake."""

    def run_execute(self, fail_at=None):
        import argparse
        import contextlib
        import tempfile
        import verify_remote as driver
        import verify_remote_bracket as study_driver

        folder = Path(tempfile.mkdtemp())
        model = folder / "model"
        model.mkdir()
        for name in ("index.json", "system_graph.json"):
            (model / name).write_text("{}")
        out = folder / "out"
        out.mkdir()
        handshake = folder / "handshake.txt"
        handshake.write_text("quiet agreed")
        calls = {"n": 0}

        def timed_trial(root, model_, out_, env, entry, urls):
            if fail_at is not None and calls["n"] == fail_at:
                raise RuntimeError("trial process exited 101")
            calls["n"] += 1
            trial = fake_trial(entry["arm"], ids=reference_ids(entry["generate"]), generate=entry["generate"],
                               per_id_ns=BASE_NS)
            start = 60.0 * calls["n"]
            return trial, fake_relay(prefill_late_ns=BASE_NS), {
                "started_utc": "x", "ended_utc": "y", "started_monotonic": start, "ended_monotonic": start + 59}

        class Quiet:
            stalls = []
            samples = 0

            def __init__(self, *a):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        saved = (driver.timed_trial, driver.sha, study_driver.workers, study_driver.sha, driver.d2.StallWatch,
                 driver.ht.Sampler, driver.ht.spotlight_starts, driver.b1.provenance, driver.lp.host_state)
        try:
            driver.timed_trial = timed_trial
            driver.sha = lambda path: "0" * 64
            study_driver.workers = lambda *a, **k: contextlib.nullcontext(["http://w"])
            study_driver.sha = lambda path: "0" * 64
            driver.d2.StallWatch = Quiet
            driver.ht.Sampler = Quiet
            driver.ht.spotlight_starts = lambda start, end: {"per_minute": {}}
            driver.b1.provenance = lambda root, sources: {}
            driver.lp.host_state = lambda: {"load_average": [0, 0, 0], "busy_processes": []}
            args = argparse.Namespace(peer_handshake=handshake, diagnostic="unit test", continue_unsettled=False,
                                      port=0)
            report = study_driver.execute(args, Path(driver.__file__).resolve().parents[2], model, out)
        finally:
            (driver.timed_trial, driver.sha, study_driver.workers, study_driver.sha, driver.d2.StallWatch,
             driver.ht.Sampler, driver.ht.spotlight_starts, driver.b1.provenance, driver.lp.host_state) = saved
        return report, {p.name for p in out.iterdir()}, json.loads((out / "manifest.json").read_text())

    def test_a_complete_study_selects_by_the_rule_and_writes_everything(self):
        report, files, manifest = self.run_execute()
        self.assertTrue(report["complete"], report.get("stopped"))
        self.assertEqual(report["observed"], len(bs.schedule()))
        self.assertEqual(report["selection"]["selected"], 16)
        self.assertTrue(all(u["counted"] for u in report["units"]))
        self.assertFalse(manifest["counts"])
        self.assertFalse(manifest["measures_arm_speed"])
        self.assertEqual(manifest["peer_handshake"], "quiet agreed")
        self.assertTrue({"manifest.json", "analysis.json", "trials.json", "references.json",
                         "warm.json", "verification.json"} <= files)

    def test_a_stopped_study_selects_nothing(self):
        report, files, _ = self.run_execute(fail_at=2 + 2 + 5)
        self.assertFalse(report["complete"])
        self.assertNotIn("selection", report)
        self.assertIn("no length is selected", report["note"])
        self.assertIn("failure.txt", files)


if __name__ == "__main__":
    unittest.main()
