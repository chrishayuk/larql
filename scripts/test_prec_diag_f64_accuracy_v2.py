"""Tests for prec_diag/f64_accuracy_v2.py, the `hf-f64-accuracy/2` test.

The gate runs on synthetic arrays, including constructed controls that must
fail. The end to end case runs E1 and E2 through the experiment runner's
synthetic world (fake LARQL, real `hf_oracle.py` on a synthetic checkpoint).
No real model is used.

Run: python3 -I -m unittest discover -s scripts -p 'test_prec_diag_*.py'
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prec_diag import evidence as ev  # noqa: E402
from prec_diag import f64_accuracy as fa  # noqa: E402
from prec_diag import f64_accuracy_v2 as f2  # noqa: E402
import test_prec_diag_experiment as tex  # noqa: E402

N = 1040
BASE = 1e-4


def world(x=BASE, a32=BASE):
    """accuracy[b][i], a32_accuracy[i], agreement[b][i], all constant."""
    acc = {b: {i: np.full(N, x) for i in f2.INPUTS} for b in f2.BACKENDS}
    a32 = {i: np.full(N, a32) for i in f2.INPUTS}
    agr = {b: {i: np.zeros(N) for i in f2.INPUTS} for b in f2.BACKENDS}
    return acc, a32, agr


class Constants(unittest.TestCase):
    def test_v2_reuses_the_frozen_r1_constants_and_invents_none(self):
        self.assertEqual((f2.NAME, f2.VERSION, f2.TEST), ("hf-f64-accuracy", 2, "hf-f64-accuracy/2"))
        self.assertEqual((f2.GATE, f2.EPSILON, f2.TAIL_RATIO), (1e-3, 1e-5, 4.0))
        self.assertEqual((f2.analysis.R1_COMPARABLE_MEDIAN, f2.analysis.R1_COMPARABLE_P99,
                          f2.analysis.R1_MATERIAL_COUNT), (2.0, 4.0, 3))
        self.assertEqual((f2.BACKENDS, f2.INPUTS), (fa.BACKENDS, fa.INPUTS))


class Gate(unittest.TestCase):
    def test_as_accurate_as_a32_everywhere_passes(self):
        v = f2.verdict(*world())
        self.assertEqual(v["verdict"], f2.PASS)
        self.assertEqual(v["backends"]["C"]["r1"]["verdict"], "comparable")

    def test_the_1b_shape_passes_v2_and_still_reports_the_v1_failure(self):
        # Gemma 3 1B, I_nr1 pos 888: A32 9.88e-4, C 1.082e-3, D 1.22e-4.
        acc, a32, agr = world()
        a32["I_nr1"][888] = 9.88e-4
        acc["C"]["I_nr1"][888] = 1.082e-3
        acc["D"]["I_nr1"][888] = 1.22e-4
        v = f2.verdict(acc, a32, agr)
        self.assertEqual(v["verdict"], f2.PASS)
        self.assertEqual(len(v["backends"]["C"]["r1"]["material_discrepancies"]), 1)
        self.assertEqual(v["v1_reported_not_gated"]["verdict"], fa.FAIL)
        self.assertEqual(v["v1_reported_not_gated"]["failing"], ["C.I_nr1"])

    def test_one_backend_failing_fails_the_test(self):
        acc, a32, agr = world()
        acc["D"]["I_rep"][:] = 5 * BASE
        v = f2.verdict(acc, a32, agr)
        self.assertEqual((v["verdict"], v["failing"]), (f2.FAIL, ["D"]))
        self.assertEqual(v["backends"]["C"]["verdict"], f2.PASS)

    def test_median_and_p99_limits_are_inclusive(self):
        self.assertEqual(f2.verdict(*world(x=2 * BASE))["verdict"], f2.PASS)
        acc, a32, agr = world()
        acc["C"]["I_nr2"][:] = 2 * BASE
        acc["C"]["I_nr2"][: N // 50] = 4 * BASE   # top 2%: p99 = 4 exactly, median 2
        v = f2.verdict(acc, a32, agr)
        self.assertEqual(v["backends"]["C"]["r1"]["inputs"]["I_nr2"]["ratio"]["p99"], 4.0)
        self.assertEqual(v["verdict"], f2.PASS)

    def test_two_material_positions_pass_three_fail_counted_across_inputs(self):
        # Material: d_X > 1e-3 while A32 <= 1e-3. r = 1.5e-3 / 9e-4 = 1.67, under the tail guard.
        acc, a32, agr = world()
        for i, p in (("I_rep", 10), ("I_nr1", 20), ("I_nr2", 30)):
            a32[i][p] = 9e-4
        acc["C"]["I_rep"][10] = 1.5e-3
        acc["C"]["I_nr1"][20] = 1.5e-3
        self.assertEqual(f2.verdict(acc, a32, agr)["verdict"], f2.PASS)
        acc["C"]["I_nr2"][30] = 1.5e-3
        v = f2.verdict(acc, a32, agr)
        self.assertEqual(v["verdict"], f2.FAIL)
        self.assertEqual(v["backends"]["C"]["r1"]["verdict"], "materially less accurate")

    def test_one_material_position_far_above_a32_fails_by_the_tail_guard(self):
        # R1 allows up to 2 material positions whatever their ratio; the tail
        # guard (approved D2) additionally requires r <= 4 at every d > 1e-3.
        acc, a32, agr = world()
        acc["C"]["I_rep"][10] = 1.5e-3           # A32 1e-4: r = 15
        c = f2.verdict(acc, a32, agr)["backends"]["C"]
        self.assertEqual((c["r1"]["verdict"], c["verdict"]), ("comparable", f2.FAIL))


class Controls(unittest.TestCase):
    """Constructed perturbations that breach the rule. Each must fail."""

    def test_known_bad_uniform_degradation_fails(self):
        v = f2.verdict(*world(x=5 * BASE))
        self.assertEqual(v["verdict"], f2.FAIL)
        self.assertEqual(v["backends"]["C"]["r1"]["verdict"], "materially less accurate")
        # v1 alone would pass it: 5e-4 is under 1e-3 everywhere.
        self.assertEqual(v["v1_reported_not_gated"]["verdict"], fa.PASS)

    def test_known_bad_tail_degradation_fails_on_p99(self):
        acc, a32, agr = world()
        acc["D"]["I_rep"][: N // 20] = 6 * BASE    # 5% of positions at 6x A32
        v = f2.verdict(acc, a32, agr)
        self.assertEqual(v["verdict"], f2.FAIL)
        self.assertEqual(v["backends"]["D"]["r1"]["verdict"], "inconclusive")

    def test_a_large_excursion_where_a32_is_also_inaccurate_fails_only_by_the_tail_guard(self):
        # A32 is itself above the gate at pos 500 (3e-3); C is far worse there
        # (2e-2, r = 6.7). R1 reads comparable: the position is not material
        # (A32 > 1e-3) and one position moves neither median nor p99.
        acc, a32, agr = world()
        a32["I_rep"][500] = 3e-3
        acc["C"]["I_rep"][500] = 2e-2
        v = f2.verdict(acc, a32, agr)
        c = v["backends"]["C"]
        self.assertEqual(c["r1"]["verdict"], "comparable")
        self.assertEqual(c["verdict"], f2.FAIL)
        self.assertEqual([(t["input"], t["position"]) for t in c["tail_guard"]], [("I_rep", 500)])
        self.assertEqual(v["verdict"], f2.FAIL)

    def test_the_tail_guard_is_inclusive_at_4_and_ignores_small_d(self):
        acc, a32, agr = world()
        a32["I_rep"][500] = 3e-3
        acc["C"]["I_rep"][500] = 4 * 3e-3          # r = 4 exactly: allowed
        a32["I_nr1"][7] = 1e-5
        acc["C"]["I_nr1"][7] = 9e-4                # r = 90 but d under the gate
        v = f2.verdict(acc, a32, agr)
        self.assertEqual(v["backends"]["C"]["tail_guard"], [])
        self.assertEqual(v["verdict"], f2.PASS)


class Joint(unittest.TestCase):
    def test_the_joint_distribution_counts_every_position_once(self):
        acc, a32, agr = world()
        j = f2.verdict(acc, a32, agr)["backends"]["C"]["joint"]["I_rep"]
        self.assertEqual(sum(map(sum, j["counts"])), N)
        self.assertEqual(len(j["counts"]), len(f2.D_EDGES) - 1)
        self.assertEqual(len(j["counts"][0]), len(f2.R_EDGES) - 1)

    def test_positions_at_the_epsilon_floor_are_reported(self):
        acc, a32, agr = world()
        a32["I_nr2"][[3, 4]] = 0.0
        acc["C"]["I_nr2"][3] = 5e-5                # r = 5 against the floor
        acc["C"]["I_nr2"][4] = 1e-6
        j = f2.verdict(acc, a32, agr)["backends"]["C"]["joint"]["I_nr2"]["eps_floor"]
        self.assertEqual(j["positions"], 2)
        self.assertAlmostEqual(j["d_x_max"], 5e-5)
        self.assertAlmostEqual(j["r_max"], 5.0)


class EvidenceClass(unittest.TestCase):
    def manifest(self, test=f2.TEST, bound=None):
        files = {f2.SCRIPT_LABEL: {"sha256": bound if bound is not None else f2.own_sha256()}}
        return {"values": {"admission_test": test} if test else {}, "files": files}

    def test_a_freeze_naming_v2_and_binding_this_script_is_prospective(self):
        cls, _, problems = f2.evidence_class(self.manifest(), "f", Path("."), None, None)
        self.assertEqual((cls, problems), (f2.PROSPECTIVE, []))

    def test_a_freeze_made_before_v2_without_an_addendum_is_invalid(self):
        cls, _, problems = f2.evidence_class(self.manifest(test=None), "f", Path("."), None, None)
        self.assertIsNone(cls)
        self.assertTrue(any("retrospective addendum" in p for p in problems))

    def test_a_freeze_binding_another_script_is_not_prospective(self):
        cls, _, problems = f2.evidence_class(self.manifest(bound="0" * 64), "f", Path("."), None, None)
        self.assertIsNone(cls)
        self.assertTrue(any(f2.SCRIPT_LABEL in p for p in problems))


class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name).resolve()
        cls.w = tex.World(cls.tmp / "w")
        cls.w.run("E1")
        cls.w.run("E2")
        (cls.w.runs / f2.SUMS).write_text("sealed\n")
        cls.v1 = cls.tmp / "v1.json"
        fa.run(cls.w.freeze, cls.w.sha, cls.w.runs, cls.v1)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def addendum(self, name):
        out = self.tmp / name
        sha = f2.write_addendum(out, f2.make_addendum(self.w.sha, self.w.runs, self.v1))
        return out, sha

    def test_runs_frozen_before_v2_need_an_addendum(self):
        r = f2.run(self.w.freeze, self.w.sha, self.w.runs, self.tmp / "no-addendum.json")
        self.assertEqual((r["verdict"], r["evidence_class"]), (f2.INVALID, None))

    def test_a_retrospective_report_binds_the_addendum_and_carries_v1(self):
        add, sha = self.addendum("add.json")
        with self.assertRaisesRegex(ev.EvidenceError, "written once"):
            f2.write_addendum(add, {})
        out = self.tmp / "report.json"
        r = f2.run(self.w.freeze, self.w.sha, self.w.runs, out, add, sha)
        self.assertIn(r["verdict"], (f2.PASS, f2.FAIL), r.get("problems"))
        self.assertEqual((r["test"], r["evidence_class"]), ("hf-f64-accuracy/2", f2.RETROSPECTIVE))
        self.assertEqual(r["v1_sealed_report"]["sha256"], ev.sha256_file(self.v1))
        self.assertIn("v1_reported_not_gated", r)
        self.assertEqual(json.loads(out.read_text())["verdict"], r["verdict"])
        with self.assertRaisesRegex(ev.EvidenceError, "written once"):
            f2.run(self.w.freeze, self.w.sha, self.w.runs, out, add, sha)

    def test_an_addendum_with_another_digest_is_refused(self):
        add, _ = self.addendum("add2.json")
        r = f2.run(self.w.freeze, self.w.sha, self.w.runs, self.tmp / "bad-digest.json", add, "0" * 64)
        self.assertEqual(r["verdict"], f2.INVALID)

    def test_resealed_runs_invalidate_the_addendum(self):
        add, sha = self.addendum("add3.json")
        sums = self.w.runs / f2.SUMS
        sums.write_text("changed\n")
        try:
            r = f2.run(self.w.freeze, self.w.sha, self.w.runs, self.tmp / "resealed.json", add, sha)
        finally:
            sums.write_text("sealed\n")
        self.assertEqual(r["verdict"], f2.INVALID)
        self.assertIn("addendum runs_sha256sums differs", r["problems"])

    def test_an_addendum_for_another_freeze_is_refused(self):
        with self.assertRaisesRegex(ev.EvidenceError, "another freeze"):
            f2.make_addendum("0" * 64, self.w.runs, self.v1)



class E1E2OnlyFreeze(unittest.TestCase):
    """A prospective E1/E2 freeze with no capture positions (Qwen3-0.6B's shape):
    the report reads E1 and E2 only and proceeds to evaluation."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name).resolve()
        cls.w = tex.World(cls.tmp / "w", extra_files={f2.SCRIPT_LABEL: Path(f2.__file__).resolve(),
                                                      "script.f64_accuracy": Path(fa.__file__).resolve()},
                          extra_values={"capture_positions": {}, "admission_test": f2.TEST})
        cls.w.run("E1")
        cls.w.run("E2")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_the_report_scope_is_e1_and_e2(self):
        self.assertEqual(f2.REPORT_EXPERIMENTS, ("E1", "E2"))

    def test_an_e1_e2_freeze_with_empty_capture_positions_reaches_a_verdict(self):
        r = f2.run(self.w.freeze, self.w.sha, self.w.runs, self.tmp / "report.json")
        self.assertIn(r["verdict"], (f2.PASS, f2.FAIL), r.get("problems"))
        self.assertEqual(r["evidence_class"], f2.PROSPECTIVE)
        self.assertIn("backends", r)

    def test_missing_e1_or_e2_evidence_is_invalid_not_omitted(self):
        for run_id, key in (("E2.D.I_nr2", "('D', 'I_nr2')"), ("E1.A64.I_rep", "('A64', 'I_rep')")):
            exp = run_id.split(".")[0]
            d = self.w.runs / exp / run_id
            aside = self.w.runs / f"{run_id}.aside"
            d.rename(aside)
            try:
                r = f2.run(self.w.freeze, self.w.sha, self.w.runs, self.tmp / f"missing-{run_id}.json")
            finally:
                aside.rename(d)
            self.assertEqual(r["verdict"], f2.INVALID)
            self.assertTrue(any(p.startswith(key) for p in r["problems"]), r["problems"])

    def test_a_freeze_binding_the_tooling_running_now_needs_no_repair(self):
        (self.w.runs / f2.SUMS).write_text("sealed\n")
        attempt = self.tmp / "attempt.err"
        attempt.write_text("traceback\n")
        with self.assertRaisesRegex(ev.EvidenceError, "nothing to repair"):
            f2.make_repair_addendum(self.w.freeze, self.w.sha, self.w.runs, [attempt], 1, "a" * 40, "scope")


class ReportingRepair(unittest.TestCase):
    """A prospective freeze whose bound report script could not read the runs:
    one repair addendum, beside the freeze, binds the corrected tooling without
    rewriting the freeze."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name).resolve()
        cls.frozen_v2 = cls.tmp / "f64_accuracy_v2.frozen.py"
        cls.frozen_v2.write_text(Path(f2.__file__).read_text() + "\n# the tooling the freeze bound\n")
        cls.w = cls.world("w")
        cls.w.run("E1")
        cls.w.run("E2")
        cls.attempt = cls.tmp / "REPORT-ATTEMPT-1.md"
        cls.attempt.write_text("KeyError: 'I_rep'\n")
        cls.add_path, cls.report_path = f2.repair_paths(cls.w.freeze)
        cls.sha = f2.write_repair_addendum(cls.w.freeze, cls.make(cls.w))
        cls.n = 0

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    @classmethod
    def world(cls, name, test=f2.TEST, files=None):
        values = {"capture_positions": {}, **({"admission_test": test} if test else {})}
        w = tex.World(cls.tmp / name, extra_values=values,
                      extra_files={f2.SCRIPT_LABEL: cls.frozen_v2,
                                   "script.f64_accuracy": Path(fa.__file__).resolve(), **(files or {})})
        w.runs.mkdir(parents=True, exist_ok=True)
        (w.runs / f2.SUMS).write_text("sealed\n")
        return w

    @classmethod
    def make(cls, w, **over):
        args = {"failed_attempts": [cls.attempt], "pr": 893, "merge_commit": "b" * 40,
                "statement": "run discovery only", **over}
        return f2.make_repair_addendum(w.freeze, w.sha, w.runs, **args)

    def elsewhere(self, addendum):
        """A copy of an addendum at a path of its own (tampering cases)."""
        type(self).n += 1
        out = self.tmp / f"elsewhere-{self.n}.json"
        return out, f2.write_addendum(out, addendum)

    def report(self, add, sha, w=None):
        type(self).n += 1
        w = w or self.w
        return f2.run(w.freeze, w.sha, w.runs, self.tmp / f"report-{self.n}.json", add, sha)

    def test_without_an_addendum_the_report_is_invalid(self):
        r = f2.run(self.w.freeze, self.w.sha, self.w.runs, self.tmp / "bare.json")
        self.assertEqual((r["verdict"], r["evidence_class"]), (f2.INVALID, None))
        self.assertTrue(any(f2.SCRIPT_LABEL in p for p in r["problems"]))

    def test_a_repaired_report_is_classed_as_such_and_binds_the_repair(self):
        a = json.loads(self.add_path.read_text())
        self.assertEqual(set(a["frozen_tooling"]), set(f2.TOOLING_LABELS))
        self.assertEqual(a["frozen_tooling"][f2.SCRIPT_LABEL], ev.sha256_file(self.frozen_v2))
        self.assertEqual(a["corrected_tooling"], f2.tooling_sha256())
        self.assertEqual([k for k in f2.TOOLING_LABELS if a["frozen_tooling"][k] != a["corrected_tooling"][k]],
                         [f2.SCRIPT_LABEL])
        self.assertEqual(a["failed_attempts"][0]["path"], str(self.attempt))
        with self.assertRaisesRegex(ev.EvidenceError, "written once"):
            f2.write_repair_addendum(self.w.freeze, a)
        r = f2.run(self.w.freeze, self.w.sha, self.w.runs, self.report_path, self.add_path, self.sha)
        self.assertIn(r["verdict"], (f2.PASS, f2.FAIL), r.get("problems"))
        self.assertEqual(r["evidence_class"], f2.REPAIRED)
        self.assertEqual(r["addendum_sha256"], self.sha)
        self.assertEqual(r["reporting_repair"]["repair"]["pr"], 893)
        self.assertEqual(r["reporting_repair"]["meaning"], f2.REPAIRED_MEANING)
        self.assertEqual(r["reporting_repair"]["failed_attempts"][0]["sha256"], ev.sha256_file(self.attempt))
        self.assertEqual(self.report_path.stat().st_mode & 0o777, 0o444)
        with self.assertRaisesRegex(ev.EvidenceError, "written once"):
            f2.run(self.w.freeze, self.w.sha, self.w.runs, self.report_path, self.add_path, self.sha)

    def test_a_copied_freeze_elsewhere_cannot_repair_the_runs(self):
        other = self.tmp / "copied"
        other.mkdir()
        copy = other / "FREEZE.json"
        copy.write_bytes(self.w.freeze.read_bytes())
        with self.assertRaisesRegex(ev.EvidenceError, "not beside FREEZE.json"):
            f2.make_repair_addendum(copy, self.w.sha, self.w.runs, [self.attempt], 893, "b" * 40, "scope")
        add = f2.repair_paths(copy)[0]
        sha = f2.write_addendum(add, json.loads(self.add_path.read_text()))
        with self.assertRaisesRegex(ev.EvidenceError, "one repair per freeze"):
            f2.run(copy, self.w.sha, self.w.runs, f2.repair_paths(copy)[1], add, sha)
        self.assertFalse(f2.repair_paths(copy)[1].exists())

    def test_a_non_repairable_module_changed_after_the_addendum_invalidates_it(self):
        real = f2.tooling_sha256
        f2.tooling_sha256 = lambda: {**real(), "script.analysis": "0" * 64}
        try:
            r = self.report(self.add_path, self.sha)
        finally:
            f2.tooling_sha256 = real
        self.assertEqual(r["verdict"], f2.INVALID)
        self.assertTrue(any(p.startswith("script.analysis differs from the freeze") for p in r["problems"]),
                        r["problems"])

    def test_one_repair_per_freeze_at_fixed_names(self):
        with self.assertRaisesRegex(ev.EvidenceError, "one repair per freeze"):
            f2.run(self.w.freeze, self.w.sha, self.w.runs, self.tmp / "other-report.json", self.add_path, self.sha)
        add, sha = self.elsewhere(self.make(self.w))   # a second, valid addendum elsewhere
        with self.assertRaisesRegex(ev.EvidenceError, "one repair per freeze"):
            f2.run(self.w.freeze, self.w.sha, self.w.runs, self.tmp / "second-report.json", add, sha)
        self.assertFalse((self.tmp / "second-report.json").exists())

    def test_a_changed_failed_attempt_record_invalidates_the_repair(self):
        before = self.attempt.read_text()
        self.attempt.write_text("rewritten\n")
        try:
            r = self.report(self.add_path, self.sha)
        finally:
            self.attempt.write_text(before)
        self.assertEqual(r["verdict"], f2.INVALID)
        self.assertTrue(any("failed attempt" in p for p in r["problems"]), r["problems"])

    def test_resealed_runs_invalidate_the_repair(self):
        sums = self.w.runs / f2.SUMS
        sums.write_text("changed\n")
        try:
            r = self.report(self.add_path, self.sha)
        finally:
            sums.write_text("sealed\n")
        self.assertEqual(r["verdict"], f2.INVALID)
        self.assertIn("addendum runs_sha256sums differs", r["problems"])

    def test_a_repair_with_another_digest_is_refused(self):
        self.assertEqual(self.report(self.add_path, "0" * 64)["verdict"], f2.INVALID)

    def test_tooling_changed_after_the_addendum_invalidates_it(self):
        real = f2.tooling_sha256
        f2.tooling_sha256 = lambda: {**real(), "script.experiment": "0" * 64}
        try:
            r = self.report(self.add_path, self.sha)
        finally:
            f2.tooling_sha256 = real
        self.assertEqual(r["verdict"], f2.INVALID)
        self.assertIn("addendum corrected_tooling differs", r["problems"])

    def test_an_addendum_for_another_freeze_is_invalid(self):
        a = {**json.loads(self.add_path.read_text()), "freeze_sha256": "0" * 64}
        r = self.report(*self.elsewhere(a))
        self.assertEqual(r["verdict"], f2.INVALID)
        self.assertIn("addendum freeze_sha256 differs", r["problems"])

    def test_a_repair_may_change_only_discovery_and_the_report(self):
        changed = self.tmp / "analysis.changed.py"
        changed.write_text(Path(f2.analysis.__file__).read_text() + "\n# not the running analysis\n")
        w = self.world("analysis-differs", files={"script.analysis": changed})
        with self.assertRaisesRegex(ev.EvidenceError, "script.analysis differs from the freeze"):
            self.make(w)

    def test_a_freeze_that_never_bound_the_v2_script_cannot_be_repaired(self):
        w = self.world("unbound")
        m = json.loads(w.freeze.read_text())
        del m["files"][f2.SCRIPT_LABEL]
        sha = tex.fz.write(w.freeze.with_name("FREEZE.unbound.json"), m)
        with self.assertRaisesRegex(ev.EvidenceError, f"does not bind {f2.SCRIPT_LABEL}"):
            f2.make_repair_addendum(w.freeze.with_name("FREEZE.unbound.json"), sha, w.runs, [self.attempt], 1,
                                    "a" * 40, "scope")

    def test_a_repair_must_name_its_attempts_pr_commit_and_scope(self):
        for over in ({"failed_attempts": []}, {"failed_attempts": [self.tmp / "absent"]}, {"pr": 0},
                     {"merge_commit": "abc"}, {"merge_commit": "B" * 40}, {"statement": " "}):
            with self.assertRaises(ev.EvidenceError, msg=over):
                self.make(self.w, **over)

    def test_a_repair_is_only_for_a_freeze_naming_v2(self):
        w = self.world("pre-v2", test=None)
        with self.assertRaisesRegex(ev.EvidenceError, "prospective freeze"):
            self.make(w)
        a = {**json.loads(self.add_path.read_text()), "freeze_sha256": w.sha}
        r = self.report(*self.elsewhere(a), w=w)
        self.assertEqual(r["verdict"], f2.INVALID)
        self.assertIn(f"freeze does not name {f2.TEST}", r["problems"])

    def test_a_malformed_addendum_is_invalid_not_a_crash(self):
        base = json.loads(self.add_path.read_text())
        for broken in ({k: v for k, v in base.items() if k != "repair"},
                       {k: v for k, v in base.items() if k != "meaning"},
                       {**base, "failed_attempts": ["not a record"]},
                       {**base, "failed_attempts": [{"path": 1, "sha256": "x"}]},
                       ["not", "an", "object"],
                       {**base, "repair": {**base["repair"], "merge_commit": "short"}}):
            r = self.report(*self.elsewhere(broken))
            self.assertEqual((r["verdict"], r["evidence_class"]), (f2.INVALID, None))
        type(self).n += 1
        bad = self.tmp / f"not-json-{self.n}.json"
        bad.write_text("{not json")
        r = self.report(bad, ev.sha256_file(bad))
        self.assertEqual((r["verdict"], r["evidence_class"]), (f2.INVALID, None))


class ControlEndToEnd(unittest.TestCase):
    """N1 through the synthetic world: plumbing, binding and the expectation
    check. The fake's logits are not a model, so the verdict itself is not
    asserted, only that it is a verdict and compared with the expectation."""

    EXPECT = {"N1": {"arm": "Dauto", "expected": f2.FAIL, "reason": "load-time Q8 residency"}}

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name).resolve()
        script = Path(f2.__file__).resolve()
        cls.w = tex.World(cls.tmp / "w", extra_files={f2.SCRIPT_LABEL: script},
                          extra_values={"control_expectation": cls.EXPECT})
        cls.w.run("E1")
        cls.records = cls.w.run("N1")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_n1_runs_production_from_the_auto_source(self):
        self.assertEqual(sorted(self.records), sorted(f"N1.Dauto.{i}" for i in f2.INPUTS))
        self.assertTrue(all(r["status"] == "complete" for r in self.records.values()), self.records)

    def test_a_control_report_never_admits_and_compares_with_the_expectation(self):
        out = self.tmp / "control.json"
        r = f2.run_control(self.w.freeze, self.w.sha, self.w.runs, out, "N1")
        self.assertIn(r["observed"], (f2.PASS, f2.FAIL), r.get("problems"))
        self.assertEqual((r["evidence_class"], r["admits"]), (f2.CONTROL, False))
        self.assertEqual(r["matches_expectation"], r["observed"] == f2.FAIL)
        self.assertEqual(set(r["arms"]), {"Dauto"})
        self.assertEqual(r["expectation"], self.EXPECT["N1"])
        with self.assertRaisesRegex(ev.EvidenceError, "written once"):
            f2.run_control(self.w.freeze, self.w.sha, self.w.runs, out, "N1")

    def test_a_control_without_a_preregistered_expectation_is_invalid(self):
        w = tex.World(self.tmp / "w2", extra_files={f2.SCRIPT_LABEL: Path(f2.__file__).resolve()})
        r = f2.run_control(w.freeze, w.sha, w.runs, self.tmp / "no-expectation.json", "N1")
        self.assertEqual((r["observed"], r["matches_expectation"]), (f2.INVALID, None))
        self.assertTrue(any("expected outcome" in p for p in r["problems"]))


if __name__ == "__main__":
    unittest.main()
