"""VERIFY-REMOTE-1 revision 2: the real-model stop check's requirements and driver loop."""
import json
from pathlib import Path
import unittest

import verify_remote_model as vr
import verify_remote_stop_model as sm
from test_verify_remote_model import PROMPT, fake_relay, fake_trial, reference_ids

STOP = vr.STOP_IDS[0]


def stopping_ids(n=10):
    return reference_ids(n) + [STOP]


def arm_trials(ids=None, **changes):
    ids = stopping_ids() if ids is None else ids
    out = []
    for arm, width in vr.ARMS.items():
        if width is not None:
            trial = fake_trial(arm, ids=list(ids), generate=vr.GENERATE)
            trial.update(changes)
            out.append((arm, trial))
    return out


class Requirements(unittest.TestCase):
    def setUp(self):
        self.reference = fake_trial(vr.SEQUENTIAL, ids=stopping_ids(), generate=vr.GENERATE)

    def test_the_schedule_is_a_reference_then_each_verify_arm_at_zero_ms(self):
        reference, arms = sm.entries()
        self.assertEqual((reference["arm"], reference["workload"], reference["rtt_ms"]),
                         (vr.SEQUENTIAL, "stop_check", 0.0))
        self.assertEqual([a["arm"] for a in arms], ["verify-2", "verify-4", "verify-8"])
        self.assertTrue(all(a["generate"] == vr.GENERATE and a["corrupt"] == "none" for a in arms))

    def test_a_reference_that_stops_within_the_limit_is_a_natural_stop(self):
        self.assertEqual(sm.reference_stop(self.reference), sm.NATURAL)
        limit = fake_trial(vr.SEQUENTIAL, ids=reference_ids(vr.GENERATE), generate=vr.GENERATE)
        self.assertEqual(sm.reference_stop(limit), sm.NO_NATURAL_STOP)

    def test_exact_arms_that_end_at_the_stop_pass(self):
        result = sm.reading(self.reference, arm_trials())
        self.assertTrue(result["passes"], result)
        self.assertFalse(result["synthetic_stop"])
        for arm in result["arms"]:
            self.assertEqual(arm["failures"], [])
            self.assertEqual(arm["final_position"], PROMPT + len(stopping_ids()) - 1)
            self.assertIn(arm["stop_delivery"]["source"], ("target-correction", "single-step"))

    def test_the_stop_delivery_names_its_block_and_whether_the_proposal_was_whole(self):
        trial = fake_trial("verify-8", ids=stopping_ids(), generate=vr.GENERATE, accept=2)
        delivery = sm.stop_delivery(trial)
        if delivery["source"] == "target-correction":
            block = trial["ledger"]["blocks"][delivery["block"]]
            self.assertEqual(delivery["whole_proposal_accepted"], block["accepted"] == block["proposed"])

    def test_different_ids_fail(self):
        ids = stopping_ids()
        ids[4] += 1
        result = sm.reading(self.reference, arm_trials(ids=ids))
        self.assertFalse(result["passes"])
        self.assertTrue(any("ids differ" in f for f in result["arms"][0]["failures"]))

    def test_an_arm_that_runs_to_the_limit_fails(self):
        result = sm.reading(self.reference, arm_trials(ids=reference_ids(vr.GENERATE)))
        self.assertFalse(result["passes"])
        self.assertTrue(any("did not end at a stop id" in f for f in result["arms"][0]["failures"]))

    def test_an_id_after_the_stop_fails(self):
        trials = arm_trials()
        trial = trials[0][1]
        trial["ids"] = trial["ids"] + [5]
        trial["id_counts"] = dict(trial["id_counts"], emitted=len(trial["ids"]))
        failures = sm.requirement_failures(trial, self.reference)
        self.assertTrue(any("gate 2" in f for f in failures), failures)

    def test_a_wrong_continuation_position_fails(self):
        result = sm.reading(self.reference, arm_trials(final_position=PROMPT + 99))
        self.assertTrue(any("final position" in f for f in result["arms"][0]["failures"]))

    def test_an_exchange_after_the_last_unit_fails(self):
        trials = arm_trials()
        trial = trials[2][1]
        trial["exchanges"] = trial["exchanges"] + [dict(trial["exchanges"][-1])]
        failures = sm.requirement_failures(trial, self.reference)
        self.assertTrue(any("belong to no unit" in f for f in failures), failures)

    def test_a_missing_arm_cannot_pass(self):
        result = sm.reading(self.reference, arm_trials()[:2])
        self.assertFalse(result["passes"])

    def test_no_natural_stop_cannot_pass(self):
        limit = fake_trial(vr.SEQUENTIAL, ids=reference_ids(vr.GENERATE), generate=vr.GENERATE)
        result = sm.reading(limit, arm_trials(ids=reference_ids(vr.GENERATE)))
        self.assertEqual(result["reference"]["stop"], sm.NO_NATURAL_STOP)
        self.assertFalse(result["passes"])

    def test_the_stop_check_workload_is_outside_the_performance_workloads(self):
        import verify_remote as driver
        self.assertNotIn(sm.WORKLOAD, vr.WORKLOADS)
        path = driver.workload_path(sm.WORKLOAD)
        self.assertEqual(path.parent.name, "functional")
        workload = json.loads(path.read_text())
        self.assertEqual(workload["tokens"], len(workload["prompt"]))
        self.assertIn(STOP, workload["prompt"])  # the earlier turn's stop, so lookup cuts there


class Execute(unittest.TestCase):
    def run_execute(self, reference_ids_=None):
        import argparse
        import contextlib
        import tempfile
        import verify_remote as driver
        import verify_remote_stop as stop_driver

        folder = Path(tempfile.mkdtemp())
        model = folder / "model"
        model.mkdir()
        for name in ("index.json", "system_graph.json"):
            (model / name).write_text("{}")
        out = folder / "out"
        out.mkdir()
        ids = stopping_ids() if reference_ids_ is None else reference_ids_

        def timed_trial(root, model_, out_, env, entry, urls):
            return fake_trial(entry["arm"], ids=list(ids), generate=entry["generate"]), fake_relay(), {
                "started_utc": "x", "ended_utc": "y", "started_monotonic": 0.0, "ended_monotonic": 1.0}

        saved = (driver.timed_trial, driver.sha, stop_driver.workers, stop_driver.sha, driver.b1.provenance,
                 driver.lp.host_state)
        try:
            driver.timed_trial = timed_trial
            driver.sha = lambda path: "0" * 64
            stop_driver.workers = lambda *a, **k: contextlib.nullcontext(["http://w"])
            stop_driver.sha = lambda path: "0" * 64
            driver.b1.provenance = lambda root, sources: {}
            driver.lp.host_state = lambda: {"load_average": [0, 0, 0], "busy_processes": []}
            args = argparse.Namespace(diagnostic="unit test", port=0)
            return stop_driver.execute(args, Path(driver.__file__).resolve().parents[2], model, out)
        finally:
            (driver.timed_trial, driver.sha, stop_driver.workers, stop_driver.sha, driver.b1.provenance,
             driver.lp.host_state) = saved

    def test_a_natural_stop_reproduced_by_every_arm_passes(self):
        report = self.run_execute()
        self.assertTrue(report["complete"], report.get("stopped"))
        self.assertTrue(report["passes"])
        self.assertEqual(report["reference_stop"], sm.NATURAL)
        self.assertTrue(all(d == [] for d in report["relay_marks"].values()))

    def test_a_reference_without_a_natural_stop_stops_the_check(self):
        report = self.run_execute(reference_ids_=reference_ids(vr.GENERATE))
        self.assertFalse(report["complete"])
        self.assertFalse(report["passes"])
        self.assertIn("did not stop naturally", report["stopped"])


if __name__ == "__main__":
    unittest.main()
