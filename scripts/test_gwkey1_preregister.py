import json
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent))
from gwkey1_preregister import ROLES, TEMPLATES, role_row


class GwKey1PreregistrationTests(unittest.TestCase):
    def test_every_frozen_row_has_an_exhaustive_disjoint_role_map(self) -> None:
        path = Path("bench/gw0/gemma3-4b-it-phase1/input.jsonl")
        if not path.exists():
            self.skipTest("frozen GW cohort is unavailable")
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        mapped = [role_row(row) for row in rows]
        self.assertEqual(len(mapped), 426)
        self.assertEqual({row["template_id"] for row in mapped}, set(TEMPLATES))
        for row in mapped:
            positions = [p for role in ROLES for p in row["roles"][role]]
            self.assertEqual(sorted(positions), list(range(row["token_count"])))
            self.assertEqual(len(positions), len(set(positions)))

    def test_role_assignment_does_not_consume_model_outputs(self) -> None:
        source = {
            "edge_id": "edge",
            "split": "train",
            "prompt": {
                "template_id": "capital-canonical",
                "token_ids": [2, 818, 5279, 529, 42, 563],
            },
            "semantic_edge": {
                "relation": "capital",
                "prompt_semantic_family": "canonical",
            },
        }
        mapped = role_row(source)
        self.assertEqual(mapped["roles"]["subject_entity"], [4])
        self.assertEqual(mapped["roles"]["relation_query"], [2])
        self.assertEqual(mapped["roles"]["answer_cue"], [5])


if __name__ == "__main__":
    unittest.main()
