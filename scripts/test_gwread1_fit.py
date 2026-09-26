from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from gwread1_fit import semantic_edge_identity  # noqa: E402


class GwRead1FitTests(unittest.TestCase):
    def test_semantic_edge_identity_does_not_read_target(self) -> None:
        left = {"semantic_edge": {"subject": "Sweden", "relation": "capital", "target": "Stockholm"}}
        right = {"semantic_edge": {"subject": "Sweden", "relation": "capital", "target": "forbidden"}}
        self.assertEqual(semantic_edge_identity(left), semantic_edge_identity(right))

    def test_semantic_edge_identity_separates_relation(self) -> None:
        capital = {"semantic_edge": {"subject": "X", "relation": "capital"}}
        currency = {"semantic_edge": {"subject": "X", "relation": "currency"}}
        self.assertNotEqual(semantic_edge_identity(capital), semantic_edge_identity(currency))


if __name__ == "__main__":
    unittest.main()
