import itertools
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent))
from gwhead1_select import EXPECTED_EDGES, select_scope


def surface(preferred: tuple[int, ...] = (2, 5)) -> dict:
    rows = []
    for size in range(9):
        for heads in itertools.combinations(range(8), size):
            retention = 0.9 if heads == preferred else min(0.79, 0.2 * len(heads))
            if heads == tuple(range(8)):
                retention = 1.0
            rows.append(
                {
                    "heads": list(heads),
                    "E": {
                        "candidate_raw": retention,
                        "candidate_zscore": retention * 2.0,
                    },
                    "total_natural_contribution_norm": float(sum(heads)),
                }
            )
    return {
        "semantic_edges": EXPECTED_EDGES["global"],
        "E_full": {"candidate_raw": 1.0, "candidate_zscore": 2.0},
        "E_ref": {"candidate_raw": 0.0, "candidate_zscore": 0.0},
        "subsets": rows,
    }


class GwHead1SelectionTests(unittest.TestCase):
    def test_minimum_cardinality_precedes_full_set(self) -> None:
        result = select_scope("global", surface(), 0.8)
        self.assertEqual(result["selected"]["heads"], [2, 5])
        self.assertEqual(result["evaluated_subsets"], 256)

    def test_higher_minimum_retention_breaks_cardinality_tie(self) -> None:
        data = surface((2, 5))
        for row in data["subsets"]:
            if row["heads"] == [1, 7]:
                row["E"] = {"candidate_raw": 0.95, "candidate_zscore": 1.9}
        result = select_scope("global", data, 0.8)
        self.assertEqual(result["selected"]["heads"], [1, 7])

    def test_incomplete_power_set_refuses(self) -> None:
        data = surface()
        data["subsets"].pop()
        with self.assertRaisesRegex(ValueError, "complete 256-subset surface"):
            select_scope("global", data, 0.8)

    def test_unstable_reference_denominator_refuses(self) -> None:
        data = surface()
        data["E_ref"]["candidate_raw"] = 1.0
        with self.assertRaisesRegex(ValueError, "unstable sufficiency denominator"):
            select_scope("global", data, 0.8)


if __name__ == "__main__":
    unittest.main()
