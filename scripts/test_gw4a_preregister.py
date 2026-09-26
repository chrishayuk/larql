import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("gw4a_preregister.py")
SPEC = importlib.util.spec_from_file_location("gw4a_preregister", MODULE_PATH)
assert SPEC and SPEC.loader
gw4a = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gw4a)


class Gw4aPreregistrationTests(unittest.TestCase):
    def test_identity_ignores_only_its_own_field(self):
        document = {"schema": gw4a.SCHEMA, "preregistration_sha256": "one", "x": [1, 2]}
        first = gw4a.canonical_hash(document)
        document["preregistration_sha256"] = "two"
        self.assertEqual(first, gw4a.canonical_hash(document))
        document["x"].append(3)
        self.assertNotEqual(first, gw4a.canonical_hash(document))


if __name__ == "__main__":
    unittest.main()
