"""Tests for prec_diag/experiment.py, the E1–E3 runner (unit 4).

- LARQL runs use the fake (`prec_diag/fake_larql.py`).
- The transformers runs are real `hf_oracle.py` processes on a synthetic
  Gemma 3 checkpoint.
- The fake is shaped like that checkpoint (vocab, hidden, heads, layers),
  so analysis can pair the two sides.

No real checkpoint, container or model is used.

Run: python3 -I -m unittest discover -s scripts -p 'test_prec_diag_*.py'
"""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prec_diag import evidence as ev  # noqa: E402
from prec_diag import experiment as ex  # noqa: E402
from prec_diag import freeze as fz  # noqa: E402
from prec_diag import hf_oracle as ho  # noqa: E402
import test_prec_diag_hf_oracle as hft  # noqa: E402

N = 1040
SCRIPTS = Path(__file__).resolve().parent
FAKE = SCRIPTS / "prec_diag" / "fake_larql.py"
OWN_SCRIPTS = {"script.hf_oracle": "prec_diag/hf_oracle.py", "script.v0": "prec_diag/v0.py",
               "script.analysis": "prec_diag/analysis.py", "script.evidence": "prec_diag/evidence.py",
               "script.freeze": "prec_diag/freeze.py", "script.experiment": "prec_diag/experiment.py",
               "script.oracle_loading": "oracle_loading.py", "script.equivalence_metrics": "equivalence_metrics.py"}
GEOMETRY = {"vocab": hft.VOCAB, "hidden": hft.HIDDEN, "q_heads": hft.HEADS, "kv_heads": hft.KV,
            "head_dim": hft.HEAD_DIM, "layers": hft.LAYERS}
POSITIONS = sorted(set([0, 1, 20, 255, 510, 511, 512, 513, 514, 767, 1022, 1023, 1024, 1025, 1026, 1039]
                       + [80 + 80 * k for k in range(12)] + [44 + 80 * k for k in range(13)]
                       + [17 + 80 * k for k in range(13)]))


class World:
    """One freeze over synthetic artifacts, under `root`."""

    def __init__(self, root: Path, doc_text="plan rev 3\n", fake_behaviour=None):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.ckpt = hft.make_checkpoint(root / "ckpt", max_position_embeddings=2 * N)
        self.container = root / "c.vindex3"
        (self.container / "segments").mkdir(parents=True)
        (self.container / "segments" / "s.bin").write_bytes(b"segment")
        (self.container / "index.json").write_text(json.dumps({"representations": {"target.x@BF16": {
            "segment": "segments/s.bin", "segment_sha256": ev.sha256_bytes(b"segment")}}}))
        attention = [{"operator": "softmax", "geometry": {"head_dim": hft.HEAD_DIM, "num_kv_heads": hft.KV}}]
        (self.container / "system_graph.json").write_text(json.dumps({"components": [{
            "id": "target", "role": "primary_text", "num_layers": hft.LAYERS, "hidden_size": hft.HIDDEN,
            "attention": attention * hft.LAYERS,
            "execution": {"attention": {"num_q_heads": hft.HEADS}, "head": {"vocab_size": hft.VOCAB}}}]}))
        self.larql = root / "bin" / "larql"
        self.larql.parent.mkdir()
        self.larql.write_text(f"#!{sys.executable}\n" + FAKE.read_text())
        self.larql.chmod(0o755)
        Path(str(self.larql) + ".behaviour.json").write_text(json.dumps({**GEOMETRY, **(fake_behaviour or {})}))
        files = {label: str(SCRIPTS / rel) for label, rel in OWN_SCRIPTS.items()}
        docs = root / "docs"
        docs.mkdir()
        for label in fz.REQUIRED_FILES:
            if label not in files:
                (docs / label).write_text(doc_text if label == "plan" else f"{label}\n")
                files[label] = str(docs / label)
        rng = np.random.default_rng(3)
        for name in ex.INPUTS:
            p = root / f"{name}.ids.json"
            p.write_text(json.dumps([2] + [int(x) for x in rng.integers(5, hft.VOCAB - 5, N - 1)]))
            files[f"ids.{name}"] = str(p)
        files["larql_binary"] = str(self.larql)
        boundary = root / "boundary.json"
        boundary.write_text(json.dumps(hft.BOUNDARY))
        files["loading_boundary"] = str(boundary)
        cfg = json.loads((self.ckpt / "config.json").read_text())
        for lt, params in cfg["rope_parameters"].items():
            d = root / "tables" / lt
            d.mkdir(parents=True)
            i = np.arange(hft.HEAD_DIM // 2, dtype=np.float64)
            ang = np.arange(N, dtype=np.float64)[:, None] * params["rope_theta"] ** (-2.0 * i / hft.HEAD_DIM)
            np.cos(ang).astype("<f4").tofile(d / "larql.cos.f32")
            np.sin(ang).astype("<f4").tofile(d / "larql.sin.f32")
            files[f"rope_table.{lt}.cos"] = str(d / "larql.cos.f32")
            files[f"rope_table.{lt}.sin"] = str(d / "larql.sin.f32")
        values = {"larql_commit": "5c375d43", "checkpoint_revision": "synthetic", "packages": ho.packages(),
                  "threads": dict(ex.HF_THREADS), "python": sys.executable,
                  "capture_positions": {"I_rep": POSITIONS, "I_nr1": POSITIONS},
                  "resource_limits": {"wall_s_per_run": 300, "wall_s_total": {"E1": 7200, "E2": 7200, "E3": 7200}, "rss_bytes": 16 * 10**9,
                                      "capture_bytes_per_run": 2 << 30, "min_free_bytes": 1},
                  "larql_environment": {"HOME": str(root), "PATH": "/usr/bin:/bin", "RAYON_NUM_THREADS": "16"}}
        spec = {"files": files, "container": str(self.container),
                "checkpoint": {"directory": str(self.ckpt), "revision": "synthetic"}, "values": values}
        self.freeze = root / "FREEZE.json"
        self.sha = fz.write(self.freeze, fz.build(spec))
        self.runs = root / "runs"
        self.manifest = ex.load_freeze(self.freeze, self.sha)

    def run(self, experiment):
        return {r["run_id"]: r for r in ex.run_experiment(self.freeze, self.sha, self.runs, experiment)}


class Tmp(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name).resolve()


class Plan(unittest.TestCase):
    def test_the_three_experiments(self):
        self.assertEqual(len(ex.plan("E1")), 15)
        self.assertEqual(len(ex.plan("E2")), 18)
        self.assertEqual(len(ex.plan("E3")), 8)
        ids = [r["run_id"] for e in ("E1", "E2", "E3") for r in ex.plan(e)]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual({r["input"] for r in ex.plan("E3")}, {"I_rep", "I_nr1"})
        self.assertEqual({r["rope_angles"] for r in ex.plan("E2") if "32a" in r["arm"]}, {"hf-f32"})
        with self.assertRaises(ValueError):
            ex.plan("E9")


class Identities(Tmp):
    def test_every_identity_is_derived_from_the_freeze(self):
        w = World(self.tmp / "w")
        m = w.manifest
        run = next(r for r in ex.plan("E2") if r["arm"] == "C32a" and r["input"] == "I_nr1")
        e = ex.expected_identity(m, w.sha, run, "E2")
        self.assertEqual(e["freeze_sha256"], w.sha)
        self.assertEqual(e["binary"]["sha256"], ev.sha256_file(w.larql))
        self.assertEqual(e["container_index_sha256"], ev.sha256_file(w.container / "index.json"))
        self.assertEqual(e["rope_angles"], "hf-f32")
        self.assertEqual(e["environment"], m["values"]["larql_environment"])
        hf = ex.expected_identity(m, w.sha, next(r for r in ex.plan("E3") if r["arm"] == "A64-capture"), "E3")
        self.assertEqual(hf["bind"]["freeze_sha256"], w.sha)
        self.assertEqual(hf["scripts"]["hf_oracle.py"], ev.sha256_file(SCRIPTS / "prec_diag" / "hf_oracle.py"))
        self.assertEqual(hf["positions"], POSITIONS)
        self.assertEqual(hf["checkpoint"]["safetensors"], m["checkpoint"]["safetensors"])

    def test_a_freeze_that_is_not_the_reviewed_one_is_refused(self):
        w = World(self.tmp / "w")
        with self.assertRaisesRegex(ev.EvidenceError, "reviewed digest"):
            ex.load_freeze(w.freeze, "0" * 64)

    def test_a_freeze_missing_what_the_runner_needs_is_refused(self):
        w = World(self.tmp / "w")
        m = json.loads(w.freeze.read_text())
        del m["values"]["capture_positions"]
        other = self.tmp / "other.json"
        sha = fz.write(other, m)
        with self.assertRaisesRegex(ev.EvidenceError, "capture_positions"):
            ex.load_freeze(other, sha)


class LarqlRuns(Tmp):
    def test_e2_runs_complete_and_are_admitted_under_their_derived_identities(self):
        w = World(self.tmp / "w")
        records = w.run("E2")
        self.assertEqual({r["status"] for r in records.values()}, {ev.COMPLETE}, records)
        for run in ex.plan("E2"):
            adm = ev.admit(w.runs / "E2" / run["run_id"], ex.expected_identity(w.manifest, w.sha, run, "E2"))
            self.assertTrue(adm.admitted, (run["run_id"], adm.reasons))
        batched = w.runs / "E2" / "E2.C-batched.I_rep" / "logits.f32"
        cached = w.runs / "E2" / "E2.C.I_rep" / "logits.f32"
        self.assertEqual(batched.read_bytes(), cached.read_bytes())

    def test_a_run_from_another_freeze_is_refused_though_valid_and_self_consistent(self):
        a = World(self.tmp / "a")
        b = World(self.tmp / "b", doc_text="plan rev 3, one word changed\n")
        self.assertNotEqual(a.sha, b.sha)
        a.run("E2")
        b.run("E2")
        run = next(r for r in ex.plan("E2") if r["arm"] == "C" and r["input"] == "I_rep")
        src, dst = a.runs / "E2" / run["run_id"], b.runs / "E2" / run["run_id"]
        self.assertTrue(ev.admit(src, ex.expected_identity(a.manifest, a.sha, run, "E2")).admitted)
        shutil.rmtree(dst)
        shutil.copytree(src, dst)
        adm = ev.admit(dst, ex.expected_identity(b.manifest, b.sha, run, "E2"))
        self.assertFalse(adm.admitted)
        self.assertTrue(any("freeze_sha256" in r for r in adm.reasons), adm.reasons)

    def test_a_run_placed_under_another_run_id_is_refused(self):
        w = World(self.tmp / "w")
        w.run("E2")
        plan = {r["run_id"]: r for r in ex.plan("E2")}
        c, d = w.runs / "E2" / "E2.C.I_rep", w.runs / "E2" / "E2.D.I_rep"
        tmp = w.runs / "E2" / "swap"
        c.rename(tmp)
        d.rename(c)
        tmp.rename(d)
        adm = ev.admit(c, ex.expected_identity(w.manifest, w.sha, plan["E2.C.I_rep"], "E2"))
        self.assertFalse(adm.admitted)
        self.assertTrue(any("run_id" in r or "arm" in r for r in adm.reasons), adm.reasons)

    def test_an_intervention_run_must_report_hf_angles(self):
        w = World(self.tmp / "w", fake_behaviour={"rope_angles": "larql"})
        records = w.run("E2")
        self.assertEqual(records["E2.C32a.I_rep"]["status"], ev.FAULT)
        self.assertIn("rope_angles", json.dumps(records["E2.C32a.I_rep"]["detail"]))
        self.assertEqual(records["E2.C.I_rep"]["status"], ev.COMPLETE)

    def test_a_short_batched_dump_is_a_fault(self):
        w = World(self.tmp / "w", fake_behaviour={"short_logits": True})
        records = w.run("E2")
        rec = records["E2.D-batched.I_nr2"]
        self.assertEqual(rec["status"], ev.FAULT)
        self.assertIn("bytes, expected", json.dumps(rec["detail"]))  # refused by the size check, not by luck


class Preflight(Tmp):
    def test_an_experiment_is_made_once(self):
        w = World(self.tmp / "w")
        (w.runs / "E2").mkdir(parents=True)
        records = ex.run_experiment(w.freeze, w.sha, w.runs, "E2")
        self.assertEqual(records[0]["status"], ev.ABORTED_PREFLIGHT)
        self.assertTrue(any("made once" in p for p in records[0]["detail"]))

    def test_drift_in_any_frozen_file_aborts_and_attempts_are_kept(self):
        w = World(self.tmp / "w")
        Path(w.manifest["files"]["plan"]["path"]).write_text("edited after the freeze\n")
        first = ex.run_experiment(w.freeze, w.sha, w.runs, "E2")
        self.assertEqual(first[0]["status"], ev.ABORTED_PREFLIGHT)
        self.assertTrue(any("plan" in p for p in first[0]["detail"]))
        self.assertFalse((w.runs / "E2").exists())
        Path(w.manifest["files"]["plan"]["path"]).write_text("plan rev 3\n")
        second = ex.run_experiment(w.freeze, w.sha, w.runs, "E2")
        self.assertEqual(len(second), 18)
        self.assertTrue((w.runs / "E2.preflight-1").is_dir() and (w.runs / "E2.preflight-2").is_dir())

    def test_a_running_script_that_is_not_the_frozen_one_aborts(self):
        w = World(self.tmp / "w")
        real = ev.sha256_file

        def lie(path):
            return "0" * 64 if Path(path).name == "experiment.py" else real(path)

        with mock.patch.object(ex.ev, "sha256_file", side_effect=lie):
            problems = ex.preflight(w.freeze, w.sha, w.runs, "E2")
        self.assertTrue(any("experiment.py" in p for p in problems), problems)


class ReadingsBoundary(Tmp):
    """Readings are refused, before anything is written, if the frozen world
    drifted after the runs were made."""

    def drifted(self, w):
        with self.assertRaisesRegex(ev.EvidenceError, "drifted"):
            ex.readings(w.freeze, w.sha, w.runs)
        self.assertFalse((w.runs / "PREC-DIAG-1.results.jsonl").exists())
        self.assertEqual(list(w.runs.glob("analysis-*")), [])

    def test_a_frozen_file_changed_after_the_runs_refuses_readings(self):
        w = World(self.tmp / "w")
        w.run("E2")
        Path(w.manifest["files"]["plan"]["path"]).write_text("edited after the runs\n")
        self.drifted(w)

    def test_a_running_analysis_or_dependency_that_is_not_the_frozen_one_refuses_readings(self):
        w = World(self.tmp / "w")
        w.run("E2")
        real = ev.sha256_file
        for name in ("analysis.py", "equivalence_metrics.py", "evidence.py"):
            def lie(path, name=name):
                return "0" * 64 if Path(path).name == name else real(path)

            with self.subTest(name), mock.patch.object(ex.ev, "sha256_file", side_effect=lie):
                self.drifted(w)


class HfRuns(Tmp):
    def launch_with(self, w, exit_code):
        """Run one oracle through a python that runs the real oracle and then
        exits with `exit_code`, whatever the oracle did."""
        wrapper = self.tmp / f"python-exit-{exit_code}"
        wrapper.write_text(f'#!/bin/sh\n"{sys.executable}" "$@"\nexit {exit_code}\n')
        wrapper.chmod(0o755)
        m = json.loads(json.dumps(w.manifest))
        m["values"]["python"] = str(wrapper)
        run = ex.plan("E1")[0]
        return run, ex.run_hf(m, w.sha, run, w.runs / "E1" / run["run_id"], "E1")

    def test_a_complete_record_from_an_oracle_that_exits_nonzero_is_not_a_success(self):
        w = World(self.tmp / "w")
        run, rec = self.launch_with(w, 3)
        self.assertEqual(rec["status"], ev.FAULT, rec)
        self.assertEqual(rec["detail"]["oracle_record_status"], ev.COMPLETE)
        kept = w.runs / "E1" / rec["detail"]["oracle_record_kept_at"]
        self.assertEqual(json.loads((kept / ev.RESULT_FILE).read_text())["status"], ev.COMPLETE)
        adm = ev.admit(w.runs / "E1" / run["run_id"], ex.expected_identity(w.manifest, w.sha, run, "E1"))
        self.assertFalse(adm.admitted)

    def test_the_same_oracle_exiting_cleanly_is_a_success(self):
        w = World(self.tmp / "w")
        run, rec = self.launch_with(w, 0)
        self.assertEqual(rec["status"], ev.COMPLETE, rec.get("detail"))
        self.assertTrue(ev.admit(w.runs / "E1" / run["run_id"],
                                 ex.expected_identity(w.manifest, w.sha, run, "E1")).admitted)

    def test_an_oracle_past_its_wall_limit_is_recorded_resource_limit(self):
        w = World(self.tmp / "w")
        m = json.loads(json.dumps(w.manifest))
        m["values"]["resource_limits"]["wall_s_per_run"] = 0.5
        run = ex.plan("E1")[0]
        rec = ex.run_hf(m, w.sha, run, w.runs / "E1" / run["run_id"], "E1")
        self.assertEqual(rec["status"], ev.RESOURCE_LIMIT, rec)
        on_disk = json.loads((w.runs / "E1" / run["run_id"] / ev.RESULT_FILE).read_text())
        self.assertEqual(on_disk["status"], ev.RESOURCE_LIMIT)


class TotalLimit(Tmp):
    """C11: each experiment has its own wall-clock total."""

    def rewrite(self, w, totals):
        m = json.loads(w.freeze.read_text())
        m["values"]["resource_limits"]["wall_s_total"] = totals
        other = self.tmp / "other.json"
        return other, fz.write(other, m)

    def test_each_experiment_reads_its_own_total(self):
        m = World(self.tmp / "w").manifest
        m["values"]["resource_limits"]["wall_s_total"] = {"E1": 10800, "E2": 14400, "E3": 7200}
        self.assertEqual([ex.total_limit(m, e) for e in ("E1", "E2", "E3")], [10800, 14400, 7200])

    def test_a_single_total_for_every_experiment_is_refused(self):
        m = World(self.tmp / "w").manifest
        m["values"]["resource_limits"]["wall_s_total"] = 3600
        with self.assertRaisesRegex(ex.EvidenceLimitError, "map each experiment"):
            ex.total_limit(m, "E2")

    def test_a_missing_or_nonpositive_total_aborts_preflight_and_nothing_runs(self):
        w = World(self.tmp / "w")
        for totals in ({"E1": 1, "E3": 1}, {"E2": 0}, {"E2": True}):
            other, sha = self.rewrite(w, totals)
            problems = ex.preflight(other, sha, self.tmp / f"runs-{len(totals)}", "E2")
            self.assertTrue(any("no positive limit for E2" in p for p in problems), problems)
            other.unlink()

    def test_past_the_total_every_remaining_run_is_resource_limit(self):
        w = World(self.tmp / "w")
        other, sha = self.rewrite(w, {"E1": 7200, "E2": 1e-9, "E3": 7200})
        records = {r["run_id"]: r for r in ex.run_experiment(other, sha, self.tmp / "runs", "E2")}
        self.assertEqual(sorted(records), sorted(r["run_id"] for r in ex.plan("E2")))
        statuses = [r["status"] for r in records.values()]
        self.assertLessEqual(statuses.count(ev.COMPLETE), 1)
        self.assertGreaterEqual(statuses.count(ev.RESOURCE_LIMIT), 17)


class EndToEnd(unittest.TestCase):
    """E1, E2 and E3 under one freeze, then the readings. The numbers are
    meaningless (the fake is not the model); this pins the plumbing."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.w = World(Path(cls._tmp.name).resolve() / "w")
        cls.records = {}
        for e in ("E1", "E2", "E3"):
            cls.records.update(cls.w.run(e))
        cls.analysis = ex.readings(cls.w.freeze, cls.w.sha, cls.w.runs)
        cls.log = {e["reading"]: e for e in ev.ResultsLog(cls.w.runs / "PREC-DIAG-1.results.jsonl").verify()}

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_every_run_completes(self):
        bad = {k: (r["status"], r.get("detail")) for k, r in self.records.items() if r["status"] != ev.COMPLETE}
        self.assertEqual(bad, {})
        self.assertEqual(len(self.records), 41)

    def test_the_analysis_is_a_complete_run_over_every_record(self):
        self.assertEqual(self.analysis["status"], ev.COMPLETE, self.analysis.get("detail"))
        self.assertEqual(len(self.analysis["identity"]["consumed_records"]), 12 + 18 + 8)  # E1 without its repeats

    def test_the_readings_are_admitted_through_the_derived_identities(self):
        for name in ("R1 accuracy C", "R1 accuracy D", "R1 yardstick A32s", "R2 repetition amplification",
                     "R3 RoPE construction", "Outcome", "R4 divergence C vs A32 I_rep",
                     "R4 divergence D vs A64 I_nr1", "R6 cached vs batched C I_rep",
                     "Validity A32 repeatability I_rep"):
            self.assertEqual(self.log[name]["status"], ev.READING_COMPLETE, (name, self.log[name].get("consumed")))
        self.assertTrue(self.log["Validity A32 repeatability I_nr2"]["values"]["identical"])
        self.assertTrue(self.log["R6 cached vs batched D I_nr1"]["values"]["E1_identical"])

    def test_the_analysis_record_binds_the_freeze(self):
        self.assertEqual(self.analysis["identity"]["freeze_sha256"], self.w.sha)
        adm_dir = self.w.runs / "analysis-1"
        self.assertTrue(ev.admit(adm_dir, {"freeze_sha256": self.w.sha}).admitted)
        refused = ev.admit(adm_dir, {"freeze_sha256": "0" * 64})
        self.assertFalse(refused.admitted)
        self.assertIn("identity.freeze_sha256", refused.reasons[0])

    def test_readings_are_recorded_once(self):
        with self.assertRaisesRegex(ev.EvidenceError, "recorded once"):
            ex.readings(self.w.freeze, self.w.sha, self.w.runs)


if __name__ == "__main__":
    unittest.main()
