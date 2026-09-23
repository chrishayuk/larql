"""W3's committed numpy expectations reproduce from the committed logits."""

import importlib.util
import json
import unittest
from pathlib import Path

try:
    import numpy  # noqa: F401
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False


@unittest.skipUnless(HAVE_NUMPY, "numpy is not installed")
class W3ReferenceReproduces(unittest.TestCase):
    def test_expected_json_matches_a_fresh_computation(self):
        path = Path(__file__).with_name("measure_plan_w3_reference.py")
        spec = importlib.util.spec_from_file_location("w3", path)
        w3 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(w3)
        committed = json.loads((w3.DIR / "expected.json").read_text())
        self.assertEqual(committed, w3.compute())


if __name__ == "__main__":
    unittest.main()
