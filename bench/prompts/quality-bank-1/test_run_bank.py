"""run_bank.py as a client of `larql vindex3 measure`: what it refuses, and
the command it composes. No larql binary is needed; nothing is measured."""

import contextlib
import io
import json
import os
import tempfile
import unittest

import run_bank


def write_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(value, f)


class ClientRefusals(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def assertRefused(self, fragment, fn, *args):
        with self.assertRaises(SystemExit) as caught:
            fn(*args)
        self.assertIn("REFUSED", str(caught.exception))
        self.assertIn(fragment, str(caught.exception))

    def test_legacy_flags_are_refused_by_name(self):
        self.assertRefused("run_bank_legacy.py", run_bank.main,
                           ["compare", "c", "o", "--keep"])

    def test_a_legacy_output_directory_is_refused(self):
        write_json(os.path.join(self.root, "reference.json"), {"arm": "reference", "entries": []})
        self.assertRefused("legacy runner", run_bank.read_reference, self.root)

    def test_a_kquant_candidate_is_refused_until_it_can_be_attested(self):
        write_json(os.path.join(self.root, "reference.json"),
                   {"schema": run_bank.CLIENT_SCHEMA, "reference": "/r", "backend": "metal",
                    "token_bank": "bank"})
        candidate = os.path.join(self.root, "cand")
        write_json(os.path.join(candidate, "index.json"), {"precision_map": {"encoding": "Q4_K"}})
        self.assertRefused("K-quant", run_bank.cmd_compare,
                           candidate, self.root, "production-q4k", "stored", "x")

    def test_a_foreign_tokenizer_is_refused(self):
        container = os.path.join(self.root, "c")
        write_json(os.path.join(container, "tokenizer.json"), {"model": "a"})
        other = os.path.join(self.root, "other.json")
        write_json(other, {"model": "b"})
        self.assertRefused("own tokenizer", run_bank.cmd_reference,
                           container, other, os.path.join(self.root, "out"), "metal", 128)


class ComposedCommand(unittest.TestCase):
    def test_measure_covers_every_bank_sample_and_records_provenance(self):
        with tempfile.TemporaryDirectory() as root:
            write_json(os.path.join(root, "bank", "manifest.json"),
                       {"samples": [{"id": "seq-000"}, {"id": "seq-001"}, {"id": "seq-002"}]})
            meta = {"schema": run_bank.CLIENT_SCHEMA, "reference": "/ref", "backend": "metal",
                    "token_bank": "bank"}
            cmd = run_bank.measure_command(meta, "/cand", root, "metal-lowered", "stored", "h")
            self.assertEqual(cmd[1:3], ["vindex3", "measure"])
            self.assertEqual(cmd[cmd.index("--sequences") + 1], "3")
            self.assertEqual(cmd[cmd.index("--reference-backend") + 1], "metal")
            self.assertEqual(cmd[cmd.index("--candidate-source") + 1], "stored")
            self.assertEqual(cmd[cmd.index("--output") + 1], os.path.join(root, "measure-h"))
            provenance = [cmd[i + 1] for i, a in enumerate(cmd) if a == "--provenance"]
            self.assertTrue(any(p.startswith("source_commit=") for p in provenance))

    def test_report_prints_the_procedures_own_numbers(self):
        agg = {"positions": 10, "kl_mean": 0.01, "kl_p99": 0.1, "top1_agreement": 0.9,
               "max_abs_delta_p99": 1.5}
        report = {"label": "h", "units": "nats, full vocabulary",
                  "reference": {"arm": {"arm": "metal", "container": "/r"}},
                  "candidate": {"arm": {"arm": "metal-lowered", "container": "/c"}},
                  "facts": {"changed_representations": ["target.output_head@NVFP4"]},
                  "summary": {"all": agg, "by_category": [["code", agg]],
                              "by_margin_band": [[[0.5, 1.0], agg]]}}
        with tempfile.TemporaryDirectory() as root:
            write_json(os.path.join(root, "measure-h", "report.json"), report)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                run_bank.cmd_report(root, "h")
        text = out.getvalue()
        self.assertIn("target.output_head@NVFP4", text)
        self.assertIn("margin 0.5-1.0", text)
        self.assertIn("90.00%", text)


if __name__ == "__main__":
    unittest.main()
