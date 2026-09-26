import importlib.util
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("gw4a_write_families.py")
SPEC = importlib.util.spec_from_file_location("gw4a_write_families", MODULE_PATH)
assert SPEC and SPEC.loader
gw4a = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gw4a)


class Gw4aWriteFamilyTests(unittest.TestCase):
    def test_auc_handles_ties(self):
        scores = np.array([0.1, 0.2, 0.2, 0.9])
        labels = np.array([False, True, False, True])
        self.assertAlmostEqual(gw4a.roc_auc(scores, labels), 0.875)

    def test_balanced_accuracy_counts_refusals_as_misses(self):
        result = gw4a.balanced_accuracy(
            ["a", "a", "b", "b"], ["a", None, "a", "b"], ["a", "b"]
        )
        self.assertEqual(result["per_relation_recall"], {"a": 0.5, "b": 0.5})
        self.assertEqual(result["value"], 0.5)


if __name__ == "__main__":
    unittest.main()
