import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("gwsup1_preregister.py")
SPEC = importlib.util.spec_from_file_location("gwsup1_preregister", MODULE_PATH)
gwsup1 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(gwsup1)


def row(edge_id: str, subject: str, target: str, token: int) -> dict:
    return {
        "edge_id": edge_id,
        "split": "train",
        "semantic_edge": {
            "subject": subject,
            "relation": "capital",
            "target": target,
            "target_token_ids": [token],
            "prompt_semantic_family": "canonical",
        },
    }


class GwSup1PreregistrationTests(unittest.TestCase):
    def test_canonical_hash_excludes_only_the_identity_field(self):
        document = {"schema": "x", "identity": None, "value": 1}
        original = gwsup1.canonical_hash(document, "identity")
        document["identity"] = original
        self.assertEqual(original, gwsup1.canonical_hash(document, "identity"))
        document["value"] = 2
        self.assertNotEqual(original, gwsup1.canonical_hash(document, "identity"))

    def test_control_requires_a_different_subject_and_destination(self):
        source = row("source", "France", "Paris", 10)
        same_target = row("same-target", "Other", "Paris", 10)
        valid = row("valid", "Germany", "Berlin", 20)
        self.assertEqual(
            gwsup1.control_for(source, [source, same_target, valid]),
            "valid",
        )


if __name__ == "__main__":
    unittest.main()
