"""`larql-verifier operator`: what ties a witness to an operator and a container, and how the answer is formed."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from larql_contract_verifier import cli  # noqa: E402
from larql_contract_verifier.contracts import q8_0_gemv as G  # noqa: E402
from refcheck import container as C, gemv, synth  # noqa: E402

SEQ, B1 = "Q8_0_GEMV_SEQ", "Q8_0_GEMV_B1"


def stream(directory, name, data):
    with open(os.path.join(directory, name), "wb") as f:
        f.write(data)
    return {"file": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


class Base(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.d)
        self.root = os.path.join(self.d, "c")
        synth.write_container(self.root, synth.tiny_model(np.random.default_rng(0)))
        c = C.open_container(self.root)
        self.t = next(t for t in c.tensors() if t.dtype == "Q8_0" and len(t.shape) == 2 and t.shape[1] % 32 == 0)
        self.op, (self.n, self.k) = self.t.name, self.t.shape
        self.blocks = c.read_bytes(self.t)
        self.x = (np.arange(self.k, dtype=np.float32) / 7.0 - 3.0).view(np.uint32).copy()
        self.y = gemv.seq_bits(self.blocks, self.x, self.n, self.k)
        self.wdir = os.path.join(self.d, "w")
        os.mkdir(self.wdir)
        self.path = os.path.join(self.wdir, "witness.json")

    def manifest(self, y=None, blocks=None, **top):
        y = self.y if y is None else y
        rec = {"call": 0, "tensor": self.op, "encoding": "Q8_0", "rows": self.n, "columns": self.k,
               "blocks": stream(self.wdir, "0.blocks", self.blocks if blocks is None else blocks),
               "x": stream(self.wdir, "0.x", self.x.astype("<u4").tobytes()),
               "y": stream(self.wdir, "0.y", y.astype("<u4").tobytes())}
        m = {"schema": G.WITNESS_SCHEMA, "kernel": G.KERNEL, "activation": "f32", "execution_arm": "Direct",
             "build_commit": "abc", "binary_sha256": "def", "container": "/elsewhere", "representation": "Q8_0",
             "calls_observed": 1, "operands_noted": 1, "records": [rec]}
        m.update(top)
        return m

    def write(self, m):
        with open(self.path, "w") as f:
            json.dump(m, f)

    def ask(self, contract=SEQ, operator=None, container=None, path=None):
        return cli.evaluate_operator(contract, operator or self.op, path or self.path, container or self.root)

    def refused(self, expected, **kw):
        r = self.ask(**kw)
        self.assertEqual(r["status"], "could_not_evaluate", r)
        self.assertRegex(" ".join(r["reasons"]), expected)
        self.assertEqual(r["violations"], 0)
        return r


class Verdicts(Base):
    def test_a_correct_record_satisfies_both_and_says_what_it_is(self):
        self.write(self.manifest())
        for contract in (SEQ, B1):
            r = self.ask(contract)
            self.assertEqual((r["status"], r["violations"], r["offending"]), ("satisfied", 0, []), contract)
            self.assertEqual(r["schema"], "larql-verify-operator/1")
            self.assertEqual(r["evidence_kind"], "numerical_measurement")
            self.assertEqual(r["witness"]["basis"], "engine_reported")
            self.assertEqual(r["witness"]["reported"]["build_commit"], "abc")
            self.assertRegex(r["report_sha256"], r"^[0-9a-f]{64}$")
            self.assertRegex(r["container"]["sha256"], r"^[0-9a-f]{64}$")

    def test_b1_says_it_is_a_measured_bound_and_not_a_proof(self):
        self.write(self.manifest())
        self.assertIn("not a machine-checked proof", self.ask(B1)["contract_statement"])
        self.assertIn("experimental", G.STATEMENTS[B1])
        self.assertIn("not accuracy", G.STATEMENTS[SEQ])

    def test_only_b1_reports_a_ratio(self):
        self.write(self.manifest())
        names = lambda r: [m["name"] for m in r["measurements"]]  # noqa: E731
        self.assertIn("worst error / bound", names(self.ask(B1)))
        self.assertNotIn("worst error / bound", names(self.ask(SEQ)))

    def test_the_lowest_bit_of_one_output_violates_seq_and_passes_b1(self):
        y = self.y.copy()
        row = next(i for i in range(self.n) if y[i] & 0x7F800000)
        y[row] ^= 1
        self.write(self.manifest(y=y))
        r = self.ask(SEQ)
        self.assertEqual((r["status"], r["violations"]), ("violated", 1))
        self.assertEqual(r["offending"], [{"name": self.op, "violations": 1, "detail": f"first row {row}"}])
        self.assertEqual(self.ask(B1)["status"], "satisfied")

    def test_a_flipped_sign_violates_both(self):
        y = self.y.copy()
        row = int(np.argmax(np.abs(y.view(np.float32))))
        y[row] ^= 0x80000000
        self.write(self.manifest(y=y))
        self.assertEqual(self.ask(SEQ)["status"], "violated")
        self.assertEqual(self.ask(B1)["status"], "violated")

    def test_the_object_judged_and_the_worst_row_are_in_the_answer(self):
        self.write(self.manifest())
        m = {x["name"]: x["value"] for x in self.ask(B1)["measurements"]}
        self.assertEqual(m["object"], self.t.object_id)
        self.assertIn("worst row", m)

    def test_a_violation_says_what_the_reference_found(self):
        y = self.y.copy()
        y[0] ^= 1
        self.write(self.manifest(y=y))
        m = {x["name"]: x["value"] for x in self.ask(SEQ)["measurements"]}
        self.assertTrue(m["found"])

    def test_the_report_digest_covers_the_container(self):
        self.write(self.manifest())
        a = self.ask(SEQ)["report_sha256"]
        with open(os.path.join(self.root, "added"), "w") as f:
            f.write("x")
        self.assertNotEqual(a, self.ask(SEQ)["report_sha256"])

    def test_the_report_digest_follows_the_result(self):
        self.write(self.manifest())
        a, b = self.ask(SEQ)["report_sha256"], self.ask(SEQ)["report_sha256"]
        self.assertEqual(a, b)
        self.assertNotEqual(a, self.ask(B1)["report_sha256"])
        y = self.y.copy()
        y[0] ^= 1
        self.write(self.manifest(y=y))
        self.assertNotEqual(a, self.ask(SEQ)["report_sha256"])


class Refusals(Base):
    def test_an_unknown_contract_including_t2(self):
        self.write(self.manifest())
        self.refused("unknown contract", contract="NOPE")
        self.refused("unknown contract", contract="Q8_0_T2")

    def test_a_missing_or_malformed_witness(self):
        self.refused("can't be read")
        with open(self.path, "w") as f:
            f.write("{not json")
        self.refused("valid JSON")
        with open(self.path, "w") as f:
            f.write("[]")
        self.refused("JSON object")

    def test_each_required_manifest_field(self):
        for key, bad in (("schema", "other/1"), ("kernel", "gpu"), ("activation", "q8k"), ("execution_arm", "Widen"),
                         ("representation", "Q4_K")):
            self.write(self.manifest(**{key: bad}))
            self.refused(key)

    def test_an_operator_it_does_not_hold_or_holds_twice(self):
        self.write(self.manifest())
        self.refused("0 records", operator="not.a.tensor")
        m = self.manifest()
        m["records"].append(m["records"][0])
        self.write(m)
        self.refused("2 records")

    def test_a_record_with_the_wrong_shape_or_encoding(self):
        for key, bad, why in (("rows", self.n + 1, "bytes"), ("columns", self.k + 32, "bytes"), ("columns", 31, "shape"),
                              ("rows", 0, "shape"), ("rows", True, "shape"), ("rows", "3", "shape"),
                              ("encoding", "Q6_K", "encoding")):
            m = self.manifest()
            m["records"][0][key] = bad
            self.write(m)
            self.refused(why)

    def test_a_changed_stream_is_refused_by_its_digest(self):
        for which in ("blocks", "x", "y"):
            m = self.manifest()
            self.write(m)
            p = os.path.join(self.wdir, m["records"][0][which]["file"])
            data = bytearray(open(p, "rb").read())
            data[0] ^= 1
            with open(p, "wb") as f:
                f.write(data)
            self.refused("digest", contract=B1)

    def test_a_short_stream_and_a_declared_length(self):
        m = self.manifest()
        self.write(m)
        p = os.path.join(self.wdir, m["records"][0]["y"]["file"])
        open(p, "wb").write(open(p, "rb").read()[:-1])
        self.refused("on disk")
        m = self.manifest()
        m["records"][0]["y"]["bytes"] -= 4
        self.write(m)
        self.refused("declares")

    def test_a_stream_outside_the_witness_directory(self):
        secret = os.path.join(self.d, "outside")
        for name in ("../outside", secret, "..", "", None, 5):
            m = self.manifest()
            m["records"][0]["y"]["file"] = name
            self.write(m)
            self.refused("not a file name")

    def test_bytes_that_are_not_the_containers(self):
        bad = bytearray(self.blocks)
        bad[5] ^= 1
        self.write(self.manifest(blocks=bytes(bad), y=gemv.seq_bits(bytes(bad), self.x, self.n, self.k)))
        self.refused("aren't the container's")

    def test_a_container_that_does_not_hold_this_tensor(self):
        m = self.manifest()
        m["records"][0]["tensor"] = "not.a.tensor"
        self.write(m)
        self.refused("0 tensors", operator="not.a.tensor")

    def test_a_witness_naming_the_object_is_held_to_it(self):
        m = self.manifest()
        m["records"][0]["object"] = self.t.object_id
        self.write(m)
        self.assertEqual(self.ask()["status"], "satisfied")
        m["records"][0]["object"] = "another.object"
        self.write(m)
        self.refused("0 tensors")

    def raw(self, m, *swaps):
        text = json.dumps(m)
        for old, new in swaps:
            self.assertIn(old, text)
            text = text.replace(old, new, 1)
        with open(self.path, "w") as f:
            f.write(text)

    def test_a_duplicate_key_is_refused_wherever_it_is(self):
        m = self.manifest()
        self.raw(m, ('"kernel": ', '"kernel": "gpu.other", "kernel": '))
        self.refused("valid JSON")
        self.raw(m, ('"records": ', '"records": [], "records": '))
        self.refused("valid JSON")
        self.raw(m, ('"tensor": ', '"tensor": "decoy", "tensor": '))
        self.refused("valid JSON")

    def test_a_non_finite_constant_is_refused(self):
        self.raw(self.manifest(), ('"calls_observed": 1', '"calls_observed": NaN'))
        self.refused("valid JSON")

    def test_a_field_of_the_wrong_kind_is_refused(self):
        for key, bad in (("calls_observed", 0), ("calls_observed", 1.0), ("calls_observed", None), ("records", [1])):
            self.write(self.manifest(**{key: bad}))
            self.refused(key if key != "records" else "list of records")
        for bad in (None, -1, 1.5, "0", True):
            m = self.manifest()
            m["records"][0]["call"] = bad
            self.write(m)
            self.refused("call")
        m = self.manifest()
        del m["records"][0]["call"]
        self.write(m)
        self.refused("call")
        m = self.manifest()
        m["records"][0]["y"]["bytes"] = float(m["records"][0]["y"]["bytes"])
        self.write(m)
        self.refused("declares")
        m = self.manifest()
        m["records"][0]["object"] = 5
        self.write(m)
        self.refused("object")

    def test_stream_names_that_can_crash_a_reader(self):
        for name in ("a\x00b", "\ud800"):
            m = self.manifest()
            m["records"][0]["y"]["file"] = name
            self.write(m)
            self.refused("y")

    def test_a_very_deep_witness(self):
        with open(self.path, "w") as f:
            f.write("[" * 200000)
        self.refused("JSON")

    def test_a_stream_or_witness_that_is_not_a_regular_file(self):
        m = self.manifest()
        self.write(m)
        p = os.path.join(self.wdir, m["records"][0]["y"]["file"])
        real = os.path.join(self.d, "elsewhere")
        shutil.copy(p, real)
        os.remove(p)
        os.symlink(real, p)
        self.refused("can't be read")
        os.remove(p)
        os.mkfifo(p)
        self.refused("regular file")
        os.remove(p)
        os.symlink("/dev/zero", p)
        self.refused("can't be read")
        target = os.path.join(self.d, "manifest")
        shutil.copy(self.path, target)
        os.remove(self.path)
        os.symlink(target, self.path)
        self.refused("can't be read")

    def test_a_container_holding_a_symlink(self):
        self.write(self.manifest())
        os.symlink("/etc", os.path.join(self.root, "link"))
        self.refused("symlink")

    def test_a_container_the_reader_warns_about(self):
        self.write(self.manifest())
        index = os.path.join(self.root, "index.json")
        data = json.load(open(index))
        rep = data["representations"][self.t.rep_id]
        self.assertIn("compiled_from", rep)
        del rep["compiled_from"]
        json.dump(data, open(index, "w"))
        self.refused("warns|compiled_from")

    def test_an_unreadable_container_file_is_not_a_crash(self):
        self.write(self.manifest())
        os.symlink("/no/such/file", os.path.join(self.root, "dangling"))
        self.refused("symlink")
        os.remove(os.path.join(self.root, "dangling"))
        index = os.path.join(self.root, "index.json")
        data = json.load(open(index))
        data["system_graph"] = "index.json/x"
        json.dump(data, open(index, "w"))
        self.refused("can't be tied")

    def test_a_container_changing_during_the_evaluation_gets_no_verdict(self):
        self.write(self.manifest())
        real = G.evaluate

        def change_it(*args):
            answer = real(*args)
            with open(os.path.join(self.root, "added"), "w") as f:
                f.write("x")
            return answer

        with mock.patch.object(G, "evaluate", change_it):
            r = self.refused("changed during")
        self.assertEqual((r["measurements"], r["offending"], r["report_sha256"]), ([], [], None))

    def test_a_witness_recording_the_containers_index_is_held_to_it(self):
        with open(os.path.join(self.root, "index.json"), "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        self.write(self.manifest(container_index_sha256=digest))
        self.assertEqual(self.ask()["status"], "satisfied")
        self.write(self.manifest(container_index_sha256="0" * 64))
        self.refused("different index.json")

    def test_a_shape_the_container_disagrees_with(self):
        n, k = self.k, self.n * 32  # the right byte count for neither, so the streams are rebuilt for it
        blocks, x = bytes(n * (k // 32) * 34), np.zeros(k, dtype=np.uint32)
        rec = {"call": 0, "tensor": self.op, "encoding": "Q8_0", "rows": n, "columns": k,
               "blocks": stream(self.wdir, "0.blocks", blocks), "x": stream(self.wdir, "0.x", x.tobytes()),
               "y": stream(self.wdir, "0.y", np.zeros(n, dtype="<u4").tobytes())}
        self.write(self.manifest() | {"records": [rec]})
        self.refused("shape")

    def test_a_tensor_the_container_does_not_hold_as_q8_0(self):
        c = C.open_container(self.root)
        other = next((t for t in c.tensors() if t.dtype != "Q8_0"), None)
        if other is None:
            self.skipTest("this synthetic container has no such tensor")
        m = self.manifest()
        m["records"][0]["tensor"] = other.name
        self.write(m)
        self.refused("not Q8_0", operator=other.name)

    def test_a_corrupted_container(self):
        self.write(self.manifest())
        segment = C.open_container(self.root).segments[self.t.rep_id].path
        data = bytearray(open(segment, "rb").read())
        data[-1] ^= 1
        with open(segment, "wb") as f:
            f.write(data)
        self.refused("can't be tied")

    def test_not_a_directory(self):
        self.write(self.manifest())
        self.refused("not a directory", container=os.path.join(self.d, "nope"))

    def test_outside_the_contracts_domain(self):
        x = self.x.copy()
        x[0] = 0x7F800000  # infinity
        self.x = x
        self.write(self.manifest())
        self.refused("x_not_finite")


class Executable(Base):
    def run_cli(self, *args):
        return subprocess.run([sys.executable, "-m", "larql_contract_verifier.cli", "operator", *args],
                              capture_output=True, text=True, cwd=ROOT)

    def test_stdout_is_one_json_object_and_every_answer_exits_0(self):
        self.write(self.manifest())
        p = self.run_cli("--contract", SEQ, "--operator", self.op, "--witness", self.path, "--container", self.root)
        self.assertEqual(p.returncode, 0, p.stderr)
        o = json.loads(p.stdout)
        self.assertEqual((o["status"], o["operator"], o["witness"]["path"]), ("satisfied", self.op, self.path))
        p = self.run_cli("--contract", "NOPE", "--operator", self.op, "--witness", self.path, "--container", self.root)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(json.loads(p.stdout)["status"], "could_not_evaluate")


if __name__ == "__main__":
    unittest.main()
