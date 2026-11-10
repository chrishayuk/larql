"""Tests for prec_diag/v0.py against a fake `larql` (prec_diag/fake_larql.py).

No model runs here. The fake writes what `vindex3 exec --logit-dump
[--capture-sites]` writes, wrong in whichever way its sidecar says.

Run: python3 -I -m unittest discover -s scripts -p 'test_prec_diag_*.py'
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prec_diag import evidence as ev  # noqa: E402
from prec_diag import v0  # noqa: E402

FAKE = Path(__file__).resolve().parent / "prec_diag" / "fake_larql.py"
N, VOCAB, LAYERS = 6, 16, 2
POSITIONS = [0, 2, 5]
IDS = [2, 818, 8858, 33358, 528, 506]
HIST_COMMIT_TAG = "historical build"


def limits(**over):
    base = {"wall_s_per_run": 30, "wall_s_total": 600, "rss_bytes": 16 * 10**9,
            "capture_bytes_per_run": 2 << 30, "min_free_bytes": 1}
    base.update(over)
    return base


class V0Case(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)  # setUp is re-entered by tests that try several cases
        self.tmp = Path(tmp.name).resolve()
        self.container = self.tmp / "model.vindex3"
        (self.container / "segments").mkdir(parents=True)
        reps = {}
        for name in ("target.embedding@BF16", "target.decoder_stack@BF16"):
            seg = f"segments/{name.split('.')[1].split('@')[0]}.bin"
            (self.container / seg).write_bytes(name.encode() * 3)
            reps[name] = {"segment": seg, "segment_sha256": ev.sha256_file(self.container / seg)}
        (self.container / "index.json").write_text(json.dumps({"representations": reps}))
        # The fake's geometry (prec_diag/fake_larql.py): hidden 8, 2 q heads, 1 kv head, head_dim 4.
        attention = [{"operator": "softmax", "geometry": {"head_dim": 4, "num_kv_heads": 1}}] * LAYERS
        (self.container / "system_graph.json").write_text(json.dumps({"components": [
            {"id": "target", "role": "primary_text", "num_layers": LAYERS, "hidden_size": 8,
             "attention": attention, "execution": {"attention": {"num_q_heads": 2}}}]}))
        self.registered = {k: {"segment_sha256": v["segment_sha256"]} for k, v in reps.items()}
        self.ids = self.tmp / "ids.json"
        self.ids.write_text(json.dumps(IDS))
        self.doc = self.tmp / "V0.md"
        self.doc.write_text("V0 spec\n")
        self.hist = self.fake("larql-66ac840e", HIST_COMMIT_TAG)
        self.exp = self.fake("larql-5c375d43", "experimental build")
        self.root = self.tmp / "V0"
        self.freeze_path = self.tmp / "V0.freeze.json"

    def fake(self, name, tag):
        path = self.tmp / "bin" / name
        path.parent.mkdir(exist_ok=True)
        path.write_text(f"#!{sys.executable}\n# {tag}\n" + FAKE.read_text())
        path.chmod(0o755)
        return path

    def behave(self, binary, **b):
        Path(str(binary) + ".behaviour.json").write_text(json.dumps(b))

    def build(self, **over):
        kw = dict(doc=self.doc, container=self.container, ids_path=self.ids,
                  expected_ids_sha256=v0.ids_json_sha256(IDS), historical=self.hist,
                  historical_sha256=ev.sha256_file(self.hist), experimental=self.exp,
                  experimental_commit="5c375d43", positions=POSITIONS, layers=LAYERS, vocab=VOCAB,
                  home=str(self.tmp), limits=limits(), registered_representations=self.registered)
        kw.update(over)
        return v0.build_freeze(**kw)

    def freeze(self, **over):
        v0.write_freeze(self.freeze_path, self.build(**over))
        return self.freeze_path

    def run_all(self, **over):
        return {r["run_id"]: r for r in v0.run_all(self.freeze(**over), self.root)}

    def statuses(self, records):
        return {rid: r["status"] for rid, r in records.items()}

    def readings(self):
        return {e["reading"]: e for e in v0.readings(self.freeze_path, self.root)}


class Freeze(V0Case):
    def test_the_freeze_is_deterministic_and_written_once(self):
        self.assertEqual(ev.canonical_json(self.build()), ev.canonical_json(self.build()))
        self.freeze()
        with self.assertRaises(ev.EvidenceError):
            v0.write_freeze(self.freeze_path, self.build())

    def test_the_freeze_records_both_id_digests_and_the_caveat(self):
        f = self.build()
        self.assertEqual(f["input"]["ids_json_sha256"], v0.ids_json_sha256(IDS))
        self.assertEqual(f["input"]["ids_u32le_sha256"], v0.ids_u32le_sha256(IDS))
        self.assertIn("not a reproduction of the 2.1.0 run", f["caveat"])
        self.assertEqual([r["run_id"] for r in f["runs"]],
                         ["V0-C-66ac840e", "V0-C-5c375d43", "V0-C-5c375d43-cap",
                          "V0-D-66ac840e", "V0-D-5c375d43", "V0-D-5c375d43-cap"])
        self.assertEqual(f["environment"], {"HOME": str(self.tmp), "PATH": "/usr/bin:/bin",
                                            "RAYON_NUM_THREADS": "16"})

    def test_the_freeze_refuses_inputs_that_differ_from_the_registered_ones(self):
        with self.assertRaisesRegex(ValueError, "ids digest"):
            self.build(expected_ids_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "registered"):
            self.build(historical_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "registered record"):
            self.build(registered_representations={**self.registered,
                                                   "target.embedding@BF16": {"segment_sha256": "0" * 64}})
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            self.build(positions=[2, 0])
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            self.build(positions=[0, N])


class HappyPath(V0Case):
    def test_six_complete_runs_and_byte_level_readings(self):
        records = self.run_all()
        self.assertEqual(set(self.statuses(records).values()), {ev.COMPLETE}, records)
        rec = records["V0-C-5c375d43-cap"]
        self.assertIn("sites/manifest.json", rec["outputs"])
        self.assertEqual(rec["nonfinite_logits"], 0)
        self.assertEqual(rec["identity"]["environment"]["RAYON_NUM_THREADS"], "16")
        self.assertGreater(rec["peak_rss_bytes"], 0)
        r = self.readings()
        for name in ("V0-R1 bridge C", "V0-R1 bridge D", "V0-R2 capture neutrality C",
                     "V0-R2 capture neutrality D"):
            self.assertEqual(r[name]["status"], ev.READING_COMPLETE, name)
            self.assertTrue(r[name]["values"]["identical"], name)
            self.assertIn("not a reproduction", r[name]["caveat"])
        self.assertEqual(r["V0-R3 capture completeness"]["values"]["admitted"],
                         ["V0-C-5c375d43-cap", "V0-D-5c375d43-cap"])
        res = r["V0-R4/R5 resources and sanity"]["values"]
        self.assertEqual({v["status"] for v in res.values()}, {ev.COMPLETE})
        ev.ResultsLog(self.root / "V0.results.jsonl").verify()

    def test_the_binary_sees_only_the_frozen_environment(self):
        with mock.patch.dict(os.environ, {"LARQL_LEAK": "1", "RAYON_NUM_THREADS": "3"}):
            records = self.run_all()
        self.assertEqual(records["V0-D-5c375d43-cap"]["status"], ev.COMPLETE,
                         records["V0-D-5c375d43-cap"].get("detail"))

    def test_readings_are_recorded_once(self):
        self.run_all()
        self.readings()
        with self.assertRaises(ev.EvidenceError):
            v0.readings(self.freeze_path, self.root)


class ReadingOutcomes(V0Case):
    def test_a_build_difference_is_reported_with_where_it_starts(self):
        self.behave(self.exp, differ_at=[3, 7])
        self.run_all()
        r = self.readings()
        for arm in "CD":
            vals = r[f"V0-R1 bridge {arm}"]["values"]
            self.assertFalse(vals["identical"])
            self.assertEqual(vals["first_difference"], {"position": 3, "vocab_index": 7})
            self.assertEqual((vals["differing_positions"], vals["differing_values"]), (1, 1))
        self.assertTrue(r["V0-R2 capture neutrality C"]["values"]["identical"])

    def test_a_capture_that_perturbs_the_logits_fails_neutrality(self):
        self.behave(self.exp, capture_perturbs=True)
        records = self.run_all()
        self.assertEqual(records["V0-C-5c375d43-cap"]["status"], ev.COMPLETE)
        r = self.readings()
        for arm in "CD":
            vals = r[f"V0-R2 capture neutrality {arm}"]["values"]
            self.assertFalse(vals["identical"])
            self.assertEqual(vals["first_difference"], {"position": N - 1, "vocab_index": VOCAB - 1})
        self.assertTrue(r["V0-R1 bridge C"]["values"]["identical"])

    def test_a_run_tampered_after_completion_is_not_admitted(self):
        self.run_all()
        dump = self.root / "runs" / "V0-C-5c375d43" / v0.LOGITS
        dump.chmod(0o644)
        data = bytearray(dump.read_bytes())
        data[0] ^= 1
        dump.write_bytes(bytes(data))
        r = self.readings()
        self.assertEqual(r["V0-R1 bridge C"]["status"], ev.NOT_ADMITTED)
        self.assertIsNone(r["V0-R1 bridge C"]["values"])
        self.assertEqual(r["V0-R1 bridge D"]["status"], ev.READING_COMPLETE)

    def test_a_wrong_execution_identity_is_not_admitted(self):
        self.run_all()
        path = self.root / "runs" / "V0-D-66ac840e" / ev.RESULT_FILE
        rec = json.loads(path.read_text())
        rec["identity"]["binary"]["sha256"] = ev.sha256_file(self.exp)
        path.write_text(json.dumps(rec))
        r = self.readings()
        self.assertEqual(r["V0-R1 bridge D"]["status"], ev.NOT_ADMITTED)
        reasons = r["V0-R1 bridge D"]["consumed"][0]["reasons"]
        self.assertIn("identity.binary.sha256", reasons[0])

    def test_runs_on_different_os_builds_are_not_admitted(self):
        builds = iter(["Darwin 27.0.0 A", "Darwin 27.0.0 A", "Darwin 27.0.0 B"] + ["Darwin 27.0.0 A"] * 3)
        with mock.patch.object(v0, "os_build", side_effect=lambda: next(builds)):
            self.run_all()
        r = self.readings()
        self.assertTrue(all(e["status"] == ev.NOT_ADMITTED for k, e in r.items() if k.startswith("V0-R1")))


class RunFaults(V0Case):
    def expect_fault(self, binary, run_id, needle, **behaviour):
        self.behave(binary, **behaviour)
        records = self.run_all()
        rec = records[run_id]
        self.assertEqual(rec["status"], ev.FAULT, rec)
        self.assertIn(needle, json.dumps(rec["detail"]))
        self.assertFalse(ev.admit(self.root / "runs" / run_id).admitted)
        return records

    def test_a_nonzero_exit_is_a_fault(self):
        self.expect_fault(self.hist, "V0-C-66ac840e", '"exit": 3', exit=3)

    def test_a_short_logit_dump_is_a_fault(self):
        self.expect_fault(self.hist, "V0-D-66ac840e", "expected", short_logits=True)

    def test_a_missing_meta_is_a_fault(self):
        self.expect_fault(self.exp, "V0-C-5c375d43", "logits.meta.json missing", no_meta=True)

    def test_the_experimental_build_must_report_larql_angles(self):
        self.expect_fault(self.exp, "V0-C-5c375d43", "rope_angles", rope_angles="hf-f32")
        self.setUp()
        self.expect_fault(self.exp, "V0-D-5c375d43", "rope_angles", rope_angles=None)

    def test_an_incomplete_capture_is_a_fault(self):
        self.expect_fault(self.exp, "V0-C-5c375d43-cap", "capture status 'fault'", capture_status="fault")

    def test_a_manifest_that_does_not_match_its_result_is_a_fault(self):
        self.expect_fault(self.exp, "V0-C-5c375d43-cap", "manifest sha256 differs", manifest_tamper=True)

    def test_a_malformed_manifest_is_a_fault_not_a_stuck_run(self):
        self.expect_fault(self.exp, "V0-C-5c375d43-cap", "verification", malformed_manifest=True)

    def test_a_plane_whose_bytes_differ_from_the_manifest_is_a_fault(self):
        self.expect_fault(self.exp, "V0-D-5c375d43-cap", "sha256 differs from the manifest", plane_tamper=True)

    def test_a_missing_or_extra_plane_is_a_fault(self):
        self.expect_fault(self.exp, "V0-C-5c375d43-cap", "missing [('residual', 1)]", drop_plane=True)
        self.setUp()
        self.expect_fault(self.exp, "V0-C-5c375d43-cap", "extra [('residual', 2)]", extra_plane=True)

    def test_a_wrong_layer_count_is_a_fault(self):
        self.expect_fault(self.exp, "V0-C-5c375d43-cap", "missing", layers=1)

    def test_a_self_consistent_capture_of_another_geometry_is_a_fault(self):
        # Manifest and planes agree with each other; they disagree with the frozen model.
        self.expect_fault(self.exp, "V0-C-5c375d43-cap", "frozen model", hidden=12)
        self.setUp()
        self.expect_fault(self.exp, "V0-C-5c375d43-cap", "layer geometry", head_dim=8)

    def test_a_wrong_plane_width_is_a_fault(self):
        self.expect_fault(self.exp, "V0-C-5c375d43-cap", "width", wrong_width=True)

    def test_a_nonfinite_plane_is_a_fault(self):
        self.expect_fault(self.exp, "V0-C-5c375d43-cap", "non-finite", nonfinite_plane=True)

    def test_a_plane_named_outside_the_capture_is_a_fault(self):
        self.expect_fault(self.exp, "V0-C-5c375d43-cap", "not inside the capture directory", plane_escape=True)

    def test_a_capture_under_another_identity_is_a_fault(self):
        for override, needle in (({"binary_sha256": "0" * 64}, "binary_sha256"),
                                 ({"tokens_sha256": "0" * 64}, "tokens_sha256"),
                                 ({"backend": "Production"}, "backend"),
                                 ({"representation_source": "stored"}, "representation_source"),
                                 ({"rope_angles": "hf-f32"}, "rope_angles"),
                                 ({"container_index_sha256": "0" * 64}, "container_index_sha256")):
            self.setUp()
            self.expect_fault(self.exp, "V0-C-5c375d43-cap", needle, identity_override=override)

    def test_a_capture_reporting_another_environment_is_a_fault(self):
        self.expect_fault(self.exp, "V0-C-5c375d43-cap", "environment", extra_env={"LARQL_X": "1"})

    def test_a_fault_in_one_run_does_not_stop_the_others(self):
        records = self.expect_fault(self.hist, "V0-C-66ac840e", '"exit": 1', exit=1)
        self.assertEqual(records["V0-D-66ac840e"]["status"], ev.FAULT)
        self.assertEqual(records["V0-C-5c375d43-cap"]["status"], ev.COMPLETE)
        r = self.readings()
        self.assertEqual(r["V0-R1 bridge C"]["status"], ev.NOT_ADMITTED)
        self.assertEqual(r["V0-R2 capture neutrality C"]["status"], ev.READING_COMPLETE)


class Limits(V0Case):
    def test_the_wall_clock_limit_kills_the_run_and_records_resource_limit(self):
        self.behave(self.hist, sleep=30)
        records = self.run_all(limits=limits(wall_s_per_run=1))
        rec = records["V0-C-66ac840e"]
        self.assertEqual(rec["status"], ev.RESOURCE_LIMIT)
        self.assertEqual(rec["detail"]["limit"], "wall_s_per_run")
        self.assertLess(rec["wall_s"], 10)

    def test_the_memory_limit_records_resource_limit(self):
        records = self.run_all(limits=limits(rss_bytes=1))
        self.assertEqual(set(self.statuses(records).values()), {ev.RESOURCE_LIMIT})

    def test_the_total_limit_records_the_remaining_runs(self):
        clock = iter(range(0, 10**6, 1000))
        with mock.patch.object(v0.time, "monotonic", side_effect=lambda: next(clock)):
            records = self.run_all(limits=limits(wall_s_total=1500, wall_s_per_run=10**6))
        later = [r for r in records.values() if r.get("detail") == {"limit": "wall_s_total"}]
        self.assertTrue(later)
        self.assertTrue(all(r["status"] == ev.RESOURCE_LIMIT for r in later))


class Preflight(V0Case):
    def expect_abort(self, needle):
        records = v0.run_all(self.freeze_path, self.root)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["status"], ev.ABORTED_PREFLIGHT)
        self.assertTrue(any(needle in p for p in records[0]["detail"]), records[0]["detail"])
        runs = self.root / "runs"
        self.assertTrue(not runs.exists() or not any(runs.iterdir()), "nothing may run")

    def test_a_changed_binary_aborts_before_anything_runs(self):
        self.freeze()
        self.exp.write_text(self.exp.read_text() + "# rebuilt\n")
        self.expect_abort("experimental binary sha256")

    def test_a_changed_segment_aborts(self):
        self.freeze()
        (self.container / "segments" / "embedding.bin").write_bytes(b"other")
        self.expect_abort("segment target.embedding@BF16")

    def test_changed_ids_abort(self):
        self.freeze()
        self.ids.write_text(json.dumps(IDS[::-1]))
        self.expect_abort("input ids digest")

    def test_existing_runs_abort(self):
        self.freeze()
        (self.root / "runs").mkdir(parents=True)
        self.expect_abort("exists")

    def test_a_changed_runner_aborts(self):
        f = self.build()
        f["runner"]["v0.py"] = "0" * 64
        v0.write_freeze(self.freeze_path, f)
        self.expect_abort("runner modules")

    def test_an_aborted_preflight_may_be_reattempted_and_both_are_kept(self):
        self.freeze()
        self.exp.write_text(self.exp.read_text() + "# rebuilt\n")
        v0.run_all(self.freeze_path, self.root)
        self.exp.write_text(self.exp.read_text().replace("# rebuilt\n", ""))
        records = v0.run_all(self.freeze_path, self.root)
        self.assertEqual(len(records), 6)
        statuses = [json.loads((self.root / f"preflight-{i}" / ev.RESULT_FILE).read_text())["status"]
                    for i in (1, 2)]
        self.assertEqual(statuses, [ev.ABORTED_PREFLIGHT, ev.COMPLETE])

    def test_a_changed_protocol_document_aborts(self):
        self.freeze()
        data = bytearray(self.doc.read_bytes())
        data[0] ^= 1
        self.doc.write_bytes(bytes(data))
        self.expect_abort("protocol document sha256")

    def test_a_missing_protocol_document_aborts(self):
        self.freeze()
        self.doc.unlink()
        self.expect_abort("protocol document missing")

    def test_the_freeze_takes_its_geometry_from_the_frozen_graph(self):
        f = self.build()
        self.assertEqual(f["capture"]["geometry"]["hidden"], 8)
        self.assertEqual(len(f["capture"]["geometry"]["layers"]), LAYERS)
        with self.assertRaisesRegex(ValueError, "frozen graph"):
            self.build(layers=LAYERS + 1)

    def test_a_freeze_record_of_another_kind_is_refused(self):
        self.freeze_path.write_text(json.dumps({"schema": "other"}))
        with self.assertRaises(ev.EvidenceError):
            v0.run_all(self.freeze_path, self.root)


class Interruption(V0Case):
    def interrupt_run(self, run_id):
        real = v0.spawn

        def spawn(argv, env, run_dir, limit):
            if run_dir.name == run_id:
                raise KeyboardInterrupt
            return real(argv, env, run_dir, limit)

        self.freeze()
        with mock.patch.object(v0, "spawn", side_effect=spawn):
            with self.assertRaises(KeyboardInterrupt):
                v0.run_all(self.freeze_path, self.root)

    def test_an_interrupted_run_stays_in_progress_and_is_not_admitted(self):
        self.interrupt_run("V0-C-5c375d43")
        rec = json.loads((self.root / "runs" / "V0-C-5c375d43" / ev.RESULT_FILE).read_text())
        self.assertEqual(rec["status"], ev.IN_PROGRESS)
        r = self.readings()
        self.assertEqual(r["V0-R1 bridge C"]["status"], ev.NOT_ADMITTED)
        self.assertEqual(r["V0-R4/R5 resources and sanity"]["values"]["V0-D-66ac840e"]["status"], "missing")

    def test_one_replacement_with_a_note_keeps_the_original(self):
        self.interrupt_run("V0-C-5c375d43")
        with self.assertRaises(ev.EvidenceError):
            v0.replace_interrupted(self.freeze_path, self.root, "V0-C-5c375d43", "  ")
        rec = v0.replace_interrupted(self.freeze_path, self.root, "V0-C-5c375d43", "machine slept")
        self.assertEqual(rec["status"], ev.COMPLETE)
        kept = self.root / "runs" / "V0-C-5c375d43.interrupted-1"
        self.assertEqual(json.loads((kept / ev.RESULT_FILE).read_text())["status"], ev.IN_PROGRESS)
        self.assertEqual((kept / "REPLACEMENT_NOTE.txt").read_text(), "machine slept")
        with self.assertRaises(ev.EvidenceError):
            v0.replace_interrupted(self.freeze_path, self.root, "V0-C-5c375d43", "again")

    def test_a_finished_run_is_never_replaced(self):
        self.run_all()
        for rid in ("V0-C-66ac840e",):
            with self.assertRaises(ev.EvidenceError):
                v0.replace_interrupted(self.freeze_path, self.root, rid, "want a rerun")

    def test_a_second_interruption_needs_an_amendment(self):
        self.interrupt_run("V0-C-5c375d43")
        with mock.patch.object(v0, "spawn", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                v0.replace_interrupted(self.freeze_path, self.root, "V0-C-5c375d43", "slept")
        with self.assertRaisesRegex(ev.EvidenceError, "amendment"):
            v0.replace_interrupted(self.freeze_path, self.root, "V0-C-5c375d43", "slept again")


if __name__ == "__main__":
    unittest.main()
