"""The verifier's own code: the source reader, the pins and the protocol answer."""

import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from larql_contract_verifier import checkpoint as S, cli  # noqa: E402
from refcheck import synth  # noqa: E402


def write_safetensors(path, tensors):
    header, payload, off = {"__metadata__": {"format": "pt"}}, b"", 0
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + len(data)]}
        payload += data
        off += len(data)
    text = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(text)) + text + payload)


class Temp(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.d)


class Pins(unittest.TestCase):
    def test_pins_are_well_formed_and_cover_the_weights(self):
        self.assertEqual(len(S.PINS), 9)
        self.assertIn("model.safetensors", S.PINS)
        self.assertEqual(len({d for _, d in S.PINS.values()}), 9)
        for name, (size, digest) in S.PINS.items():
            self.assertGreater(size, 0, name)
            self.assertRegex(digest, r"^[0-9a-f]{64}$", name)

    @unittest.skipUnless(os.environ.get("LCV_SOURCE"), "set LCV_SOURCE to a real checkpoint directory")
    def test_the_pins_match_the_real_checkpoint(self):
        self.assertEqual(S.verify_source(os.environ["LCV_SOURCE"]), 9)


class Source(Temp):
    def pins(self, name, data, size=None, digest=None):
        with open(os.path.join(self.d, name), "wb") as f:
            f.write(data)
        return {name: (len(data) if size is None else size, digest or hashlib.sha256(data).hexdigest())}

    def test_matching_files_pass(self):
        self.assertEqual(S.verify_source(self.d, self.pins("a", b"abc")), 1)

    def test_a_missing_file_a_wrong_size_and_a_wrong_hash_are_refused(self):
        with self.assertRaisesRegex(S.SourceError, "lacks"):
            S.verify_source(self.d, {"absent": (1, "0" * 64)})
        with self.assertRaisesRegex(S.SourceError, "pinned 9"):
            S.verify_source(self.d, self.pins("a", b"abc", size=9))
        with self.assertRaisesRegex(S.SourceError, "sha256"):
            S.verify_source(self.d, self.pins("a", b"abc", digest="0" * 64))

    def test_bf16_widens_exactly_with_its_shape(self):
        bits = np.array([0x3F80, 0xBF80, 0x0000, 0x8000, 0x0001, 0x7F80], dtype="<u2")
        write_safetensors(os.path.join(self.d, "model.safetensors"), {"t": ("BF16", [2, 3], bits.tobytes())})
        out = S.load_weights(self.d)["t"]
        self.assertEqual(out.shape, (2, 3))
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(out.reshape(-1).view(np.uint32).tolist(), [int(b) << 16 for b in bits])

    def test_other_dtypes_and_truncation_are_refused(self):
        p = os.path.join(self.d, "model.safetensors")
        write_safetensors(p, {"t": ("F32", [1], b"\0\0\0\0")})
        with self.assertRaisesRegex(S.SourceError, "BF16 only"):
            S.load_weights(self.d)
        write_safetensors(p, {"t": ("BF16", [2], b"\0\0\0\0")})
        with open(p, "r+b") as f:
            f.truncate(os.path.getsize(p) - 1)
        with self.assertRaisesRegex(S.SourceError, "short read"):
            S.load_weights(self.d)


class Answers(Temp):
    def setUp(self):
        super().setUp()
        self.model = synth.tiny_model(np.random.default_rng(0))
        self.root = os.path.join(self.d, "c")
        self.w = synth.write_container(self.root, self.model)

    def evaluate(self, weights=None, contract="Q8_0_T2", root=None):
        with mock.patch.object(S, "load_weights", return_value=weights or self.w.source):
            return cli.evaluate(contract, root or self.root, self.d, pins={})

    def test_a_clean_container_is_satisfied(self):
        r = self.evaluate()
        self.assertEqual((r["status"], r["violations"], r["offending"]), ("satisfied", 0, []))
        self.assertEqual(r["evidence_kind"], "numerical_measurement")
        self.assertRegex(r["report_sha256"], r"^[0-9a-f]{64}$")

    def test_a_violation_counts_and_lists_what_broke_it(self):
        weights = {k: v.copy() for k, v in self.w.source.items()}
        key = next(k for k, v in weights.items() if v.ndim == 2 and "proj" in k)
        weights[key][0, 0] += 1000.0
        r = self.evaluate(weights)
        self.assertEqual(r["status"], "violated")
        self.assertGreater(r["violations"], 0)
        self.assertEqual(sum(o["violations"] for o in r["offending"]), r["violations"])
        self.assertEqual(r["offending"][0]["name"], key)

    def test_the_report_digest_is_deterministic_and_sensitive(self):
        a, b = self.evaluate(), self.evaluate()
        self.assertEqual(a["report_sha256"], b["report_sha256"])
        weights = {k: v.copy() for k, v in self.w.source.items()}
        key = next(k for k, v in weights.items() if "proj" in k)
        weights[key][0, 0] += 1e-4
        self.assertNotEqual(a["report_sha256"], self.evaluate(weights)["report_sha256"])

    def test_a_container_with_no_q8_0_is_never_satisfied(self):
        root = os.path.join(self.d, "none")
        w = synth.write_container(root, self.model, policy=lambda k, s: False)
        with mock.patch.object(S, "load_weights", return_value=w.source):
            r = cli.evaluate("Q8_0_T2", root, self.d, pins={})
        self.assertEqual(r["status"], "could_not_evaluate")
        self.assertIn("vacuously", r["reasons"][0])

    def test_a_missing_source_tensor_is_never_satisfied(self):
        weights = dict(self.w.source)
        weights.pop(next(iter(weights)))
        self.assertEqual(self.evaluate(weights)["status"], "could_not_evaluate")

    def test_a_refused_container_an_unknown_contract_and_an_unpinned_source(self):
        with open(os.path.join(self.root, "index.json"), "w") as f:
            f.write("{")
        self.assertIn("refused the container", self.evaluate()["reasons"][0])
        self.assertIn("unknown contract", self.evaluate(contract="NOPE")["reasons"][0])
        r = cli.evaluate("Q8_0_T2", self.root, self.d)  # the real pins
        self.assertEqual(r["status"], "could_not_evaluate")
        self.assertIn("not the checkpoint", r["reasons"][0])

    def test_a_result_is_bound_to_the_code_that_measured_it(self):
        v = self.evaluate()["verifier"]
        self.assertRegex(v["code_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(v["code_sha256"], cli.code_sha256())
        self.assertRegex(v["refcheck_commit"], r"^[0-9a-f]{40}$")
        self.assertTrue(v["version"])

    def test_paths_are_echoed_as_given(self):
        r = self.evaluate()
        self.assertEqual(r["container"]["path"], self.root)
        self.assertEqual(r["source"]["path"], self.d)


class CommandLine(Temp):
    def test_stdout_is_one_json_object_and_could_not_evaluate_exits_0(self):
        p = subprocess.run(
            [sys.executable, "-m", "larql_contract_verifier.cli", "contract", "--contract", "NOPE",
             "--container", self.d, "--source", self.d],
            capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(p.returncode, 0, p.stderr)
        o = json.loads(p.stdout)
        self.assertEqual(o["schema"], "larql-verify-contract/1")
        self.assertEqual(o["status"], "could_not_evaluate")


class Provenance(unittest.TestCase):
    def test_copied_files_match_the_provenance_table(self):
        with open(os.path.join(ROOT, "provenance", "PROVENANCE.md")) as f:
            rows = re.findall(r"^\| `([^`]+)` \| `([0-9a-f]{40})` \|", f.read(), re.M)
        self.assertGreaterEqual(len(rows), 11)
        for path, blob in rows:
            with open(os.path.join(ROOT, path), "rb") as f:
                data = f.read()
            got = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
            self.assertEqual(got, blob, path)


class Expected(unittest.TestCase):
    def test_the_expected_results_are_consistent_with_the_protocol(self):
        with open(os.path.join(ROOT, "expected", "qwen3-0.6b.json")) as f:
            e = json.load(f)
        self.assertEqual(e["source"]["revision"], S.REVISION)
        for c in e["containers"].values():
            self.assertRegex(c["larql_commit"], r"^[0-9a-f]{40}$")
            self.assertEqual(c["status"] == "satisfied", c["violations"] == 0)
        o = e["operators"]
        self.assertRegex(o["larql_commit"], r"^[0-9a-f]{40}$")
        self.assertEqual(o["container"], "corrected")
        self.assertTrue(o["checked"])
        for fault in o["faults"].values():
            self.assertEqual(fault["Q8_0_GEMV_SEQ"], "violated")


if __name__ == "__main__":
    unittest.main()
