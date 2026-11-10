"""Tests for prec_diag/evidence.py: lifecycle, admission, the results log.

Run: python3 -I -m unittest discover -s scripts -p 'test_prec_diag_*.py'
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prec_diag import evidence as ev  # noqa: E402


class Tmp(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def completed(self, name="run", identity=None, files=None):
        rr = ev.RunRecord.begin(self.root / name, name, "test", identity or {"binary": {"sha256": "a"}})
        for rel, data in (files or {"out.bin": b"payload"}).items():
            (rr.run_dir / rel).parent.mkdir(parents=True, exist_ok=True)
            (rr.run_dir / rel).write_bytes(data)
        rr.complete(list((files or {"out.bin": b""}).keys()))
        return rr


class Lifecycle(Tmp):
    def test_begin_records_in_progress_and_refuses_an_existing_directory(self):
        rr = ev.RunRecord.begin(self.root / "r", "r", "test", {})
        self.assertEqual(json.loads(rr.path.read_text())["status"], ev.IN_PROGRESS)
        with self.assertRaises(ev.EvidenceError):
            ev.RunRecord.begin(self.root / "r", "r", "test", {})

    def test_a_record_reaches_exactly_one_final_status(self):
        rr = self.completed()
        with self.assertRaises(ev.EvidenceError):
            rr.fail(ev.FAULT, "again")
        with self.assertRaises(ev.EvidenceError):
            rr.complete(["out.bin"])

    def test_complete_is_not_a_failure_status_and_unknown_statuses_are_refused(self):
        rr = ev.RunRecord.begin(self.root / "r", "r", "test", {})
        with self.assertRaises(ev.EvidenceError):
            rr.fail(ev.COMPLETE, "no")
        with self.assertRaises(ev.EvidenceError):
            rr.fail("passed", "no")
        self.assertEqual(json.loads(rr.path.read_text())["status"], ev.IN_PROGRESS)

    def test_a_missing_output_records_fault_not_complete(self):
        rr = ev.RunRecord.begin(self.root / "r", "r", "test", {})
        with self.assertRaises(ev.EvidenceError):
            rr.complete(["absent.bin"])
        rec = json.loads(rr.path.read_text())
        self.assertEqual(rec["status"], ev.FAULT)
        self.assertEqual(rec["detail"], {"missing_output": "absent.bin"})

    def test_an_output_outside_the_run_directory_records_fault(self):
        (self.root / "outside.bin").write_bytes(b"x")
        rr = ev.RunRecord.begin(self.root / "r", "r", "test", {})
        with self.assertRaises(ev.EvidenceError):
            rr.complete(["../outside.bin"])
        self.assertEqual(json.loads(rr.path.read_text())["status"], ev.FAULT)

    def test_an_interrupted_write_leaves_the_previous_record_intact(self):
        rr = ev.RunRecord.begin(self.root / "r", "r", "test", {})
        before = rr.path.read_bytes()
        with mock.patch.object(ev.os, "replace", side_effect=OSError("disk gone")):
            with self.assertRaises(OSError):
                rr.fail(ev.FAULT, "x")
        self.assertEqual(rr.path.read_bytes(), before)
        self.assertEqual(json.loads(rr.path.read_text())["status"], ev.IN_PROGRESS)


class Admission(Tmp):
    def test_a_complete_record_with_intact_outputs_is_admitted(self):
        rr = self.completed(files={"out.bin": b"payload", "sub/x.bin": b"x"})
        adm = ev.admit(rr.run_dir, {"binary": {"sha256": "a"}}, ["out.bin"])
        self.assertTrue(adm.admitted, adm.reasons)
        self.assertEqual(adm.record_sha256, ev.record_digest(rr.run_dir))

    def test_only_complete_admits(self):
        for status in (ev.RESOURCE_LIMIT, ev.FAULT, ev.REFUSED, ev.ABORTED_PREFLIGHT):
            rr = ev.RunRecord.begin(self.root / status, status, "test", {})
            rr.fail(status, "x")
            self.assertFalse(ev.admit(rr.run_dir).admitted, status)
        rr = ev.RunRecord.begin(self.root / "running", "running", "test", {})
        adm = ev.admit(rr.run_dir)
        self.assertFalse(adm.admitted)
        self.assertIn("in_progress", adm.reasons[0])

    def test_a_missing_record_is_refused(self):
        (self.root / "empty").mkdir()
        self.assertEqual(ev.admit(self.root / "empty").reasons, [f"{ev.RESULT_FILE} missing"])
        self.assertFalse(ev.admit(self.root / "nowhere").admitted)

    def test_changed_bytes_size_or_absence_are_refused(self):
        rr = self.completed(files={"out.bin": b"payload"})
        (rr.run_dir / "out.bin").write_bytes(b"paylaod")
        self.assertIn("sha256 differs", ev.admit(rr.run_dir).reasons[0])
        (rr.run_dir / "out.bin").write_bytes(b"payload!")
        self.assertIn("size", ev.admit(rr.run_dir).reasons[0])
        (rr.run_dir / "out.bin").unlink()
        self.assertIn("missing on disk", ev.admit(rr.run_dir).reasons[0])

    def test_a_required_output_the_record_does_not_list_is_refused(self):
        rr = self.completed()
        adm = ev.admit(rr.run_dir, required_outputs=["logits.f32"])
        self.assertFalse(adm.admitted)
        self.assertIn("required output logits.f32", adm.reasons[0])

    def test_a_wrong_or_missing_identity_is_refused(self):
        rr = self.completed(identity={"binary": {"sha256": "a", "commit": "c"}, "arm": "C"})
        self.assertTrue(ev.admit(rr.run_dir, {"binary": {"sha256": "a"}}).admitted)
        wrong = ev.admit(rr.run_dir, {"binary": {"sha256": "b"}, "arm": "C"})
        self.assertEqual(wrong.reasons, ["identity.binary.sha256 = 'a', expected 'b'"])
        missing = ev.admit(rr.run_dir, {"environment": {"RAYON_NUM_THREADS": "16"}})
        self.assertEqual(missing.reasons, ["identity.environment missing"])

    def test_a_record_naming_a_file_outside_the_run_is_refused(self):
        (self.root / "secret").write_bytes(b"s")
        rr = self.completed()
        rec = json.loads(rr.path.read_text())
        rec["outputs"]["../secret"] = {"sha256": ev.sha256_bytes(b"s"), "bytes": 1}
        rec["outputs"]["/etc/hosts"] = {"sha256": "x", "bytes": 1}
        rr.path.write_text(json.dumps(rec))
        reasons = ev.admit(rr.run_dir).reasons
        self.assertEqual(sum("not inside the run directory" in r for r in reasons), 2)

    def test_a_symlink_leaving_the_run_is_refused(self):
        (self.root / "secret").write_bytes(b"s")
        rr = self.completed()
        os.symlink(self.root / "secret", rr.run_dir / "link")
        rec = json.loads(rr.path.read_text())
        rec["outputs"]["link"] = {"sha256": ev.sha256_bytes(b"s"), "bytes": 1}
        rr.path.write_text(json.dumps(rec))
        self.assertIn("not inside", ev.admit(rr.run_dir).reasons[0])

    def test_garbage_fails_closed_and_never_raises(self):
        for i, raw in enumerate((b"{not json", b"[]", b'{"schema": "larql.prec-diag-1.run-result/1", "status": "complete",'
                    b' "outputs": {"a": 5}}', b"\xff\xfe")):
            d = self.root / f"g{i}"
            d.mkdir()
            (d / ev.RESULT_FILE).write_bytes(raw)
            adm = ev.admit(d, {"x": 1})
            self.assertFalse(adm.admitted, raw)
            self.assertTrue(adm.reasons)

    def test_a_wrong_schema_is_refused(self):
        rr = self.completed()
        rec = json.loads(rr.path.read_text())
        rec["schema"] = "something-else/1"
        rr.path.write_text(json.dumps(rec))
        self.assertIn("schema", ev.admit(rr.run_dir).reasons[0])


class Contained(Tmp):
    def test_only_plain_relative_names_inside_the_base_pass(self):
        (self.root / "a").mkdir()
        self.assertEqual(ev.contained(self.root, "a/b.f32"), self.root / "a" / "b.f32")
        for bad in ("", "/etc/hosts", "../x", "a/../../x", "./a", "a//b", "a/", "a\\b", "a\0b", None, 3):
            self.assertIsNone(ev.contained(self.root, bad), bad)

    def test_a_missing_base_contains_nothing(self):
        self.assertIsNone(ev.contained(self.root / "absent", "x"))


class ResultsLog(Tmp):
    def test_appends_chain_and_verify(self):
        log = ev.ResultsLog(self.root / "r.jsonl")
        a = log.append({"reading": "one", "status": "complete"})
        b = log.append({"reading": "two", "status": "complete"})
        self.assertEqual((a["seq"], b["seq"]), (0, 1))
        self.assertEqual(a["prev_line_sha256"], ev.GENESIS)
        self.assertEqual([e["reading"] for e in log.verify()], ["one", "two"])

    def test_an_entry_cannot_override_the_chain_fields(self):
        log = ev.ResultsLog(self.root / "r.jsonl")
        e = log.append({"reading": "x", "seq": 99, "prev_line_sha256": "f" * 64, "schema": "mine"})
        self.assertEqual((e["seq"], e["prev_line_sha256"], e["schema"]), (0, ev.GENESIS, ev.LOG_SCHEMA))
        log.verify()

    def test_an_edit_or_a_deletion_breaks_the_chain(self):
        log = ev.ResultsLog(self.root / "r.jsonl")
        for i in range(3):
            log.append({"reading": str(i)})
        lines = log.path.read_bytes().split(b"\n")
        log.path.write_bytes(b"\n".join([lines[0], lines[1].replace(b'"1"', b'"X"'), lines[2], b""]))
        with self.assertRaises(ev.EvidenceError):
            log.verify()
        log.path.write_bytes(b"\n".join([lines[0], lines[2], b""]))
        with self.assertRaises(ev.EvidenceError):
            log.verify()

    def test_a_torn_last_line_is_never_extended(self):
        log = ev.ResultsLog(self.root / "r.jsonl")
        log.append({"reading": "0"})
        with open(log.path, "ab") as f:
            f.write(b'{"reading": "1", "sch')
        with self.assertRaises(Exception):
            log.append({"reading": "2"})
        self.assertNotIn(b'"2"', log.path.read_bytes())


class Readings(Tmp):
    def test_a_reading_over_an_unadmitted_input_is_not_admitted_and_carries_no_values(self):
        good = ev.admit(self.completed("good").run_dir)
        bad = ev.admit(self.root / "missing")
        entry = ev.reading("R", ev.READING_COMPLETE, [good, bad], {"identical": True})
        self.assertEqual(entry["status"], ev.NOT_ADMITTED)
        self.assertIsNone(entry["values"])
        ok = ev.reading("R", ev.READING_COMPLETE, [good], {"identical": True})
        self.assertEqual((ok["status"], ok["values"]), (ev.READING_COMPLETE, {"identical": True}))

    def test_canonical_json_is_deterministic_and_refuses_nan(self):
        self.assertEqual(ev.canonical_json({"b": 1, "a": [1, 2]}), b'{"a":[1,2],"b":1}')
        with self.assertRaises(ValueError):
            ev.canonical_json({"x": float("nan")})

    def test_numeric_environment_keeps_only_numeric_prefixes(self):
        env = {"HOME": "/h", "RAYON_NUM_THREADS": "16", "LARQL_X": "1", "OMP_NUM_THREADS": "8", "PATH": "/b"}
        self.assertEqual(ev.numeric_environment(env),
                         {"LARQL_X": "1", "OMP_NUM_THREADS": "8", "RAYON_NUM_THREADS": "16"})


if __name__ == "__main__":
    unittest.main()
