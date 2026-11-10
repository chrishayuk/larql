"""Unit tests for the incomplete-run recorder's log parser."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import record_incomplete_run as r  # noqa: E402

LOG = [
    "gemma: window 512; oracle load Gemma3ForCausalLM gate pass []",
    "                chat n=21     reference: cached 4.36e-04 prefill 1.12e-04 [pass]  "
    "production-stored: cached 3.76e-04 prefill 2.19e-04 [pass]",
    "  decode production: cached 5.85e-01 prefill 5.85e-01 [fail]",
]


class ParseTests(unittest.TestCase):
    def test_case_and_decode_lines_are_parsed(self):
        out = r.parse_completed(LOG)
        self.assertEqual(out["cases"][0]["name"], "chat")
        self.assertEqual(out["cases"][0]["length"], 21)
        self.assertEqual(set(out["cases"][0]["arms"]), {"reference", "production-stored"})
        self.assertEqual(out["cases"][0]["arms"]["reference"]["cached_max_abs_logprob"], 4.36e-04)
        self.assertEqual(out["decode"]["production"]["gate"], "fail")

    def test_other_lines_are_ignored(self):
        self.assertEqual(r.parse_completed(["noise"]), {"cases": [], "decode": {}})


if __name__ == "__main__":
    unittest.main()
