"""The decision the boundary gate makes. Its use of GitHub is exercised on real pull requests."""

import importlib.util
import os
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
spec = importlib.util.spec_from_file_location("gate", os.path.join(REPO, "scripts", "check_verifier_boundary.py"))
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)

WHY = "Verifier-and-engine justification: the container format changed on both sides."


class Sides(unittest.TestCase):
    def test_what_counts_as_the_measurement(self):
        for path in (
            "verifier/refcheck/e1.py",
            "verifier/larql_contract_verifier/contracts/q8_0_t2.py",
            "verifier/expected/qwen3-0.6b.json",
            "verifier/provenance/PROVENANCE.md",
            "verifier/requirements.lock",
            "verifier/pyproject.toml",
            "scripts/check_verifier_boundary.py",
            ".github/workflows/verifier-boundary.yml",
        ):
            self.assertEqual(gate.sides([path]), ([path], []), path)

    def test_what_does_not(self):
        for path in ("verifier/README.md", "verifier/tests/test_e1.py", "docs/verification/independence.md", "scripts/x.py"):
            self.assertEqual(gate.sides([path]), ([], []), path)

    def test_the_engine(self):
        self.assertEqual(gate.sides(["crates/larql-lql/src/lib.rs"]), ([], ["crates/larql-lql/src/lib.rs"]))

    def test_only_a_prefix_at_the_start_counts(self):
        self.assertEqual(gate.sides(["docs/verifier/refcheck/x", "docs/crates/x"]), ([], []))


class Override(unittest.TestCase):
    def test_a_maintainer_with_a_reason_is_let_through(self):
        for role in ("admin", "maintain"):
            self.assertEqual(gate.override_problems(role, [(role, WHY)]), [])

    def test_anyone_else_applying_the_label_is_refused(self):
        for role in ("write", "triage", "read", None):
            self.assertIn("applied by a maintainer", gate.override_problems(role, [("admin", WHY)])[0])

    def test_a_reason_is_required(self):
        self.assertIn("justification", gate.override_problems("admin", [])[0])

    def test_a_reason_from_anyone_else_does_not_count(self):
        for role in ("write", "triage", "read", None):
            self.assertIn("justification", gate.override_problems("admin", [(role, WHY)])[0])

    def test_an_empty_reason_does_not_count(self):
        for text in ("Verifier-and-engine justification:", "Verifier-and-engine justification:   \nnext line", "", None):
            self.assertIn("justification", gate.override_problems("admin", [("admin", text)])[0], repr(text))

    def test_the_reason_can_be_any_line_of_a_comment(self):
        self.assertEqual(gate.override_problems("admin", [("admin", f"Reviewed.\n\n{WHY}\n")]), [])

    def test_both_problems_are_reported(self):
        self.assertEqual(len(gate.override_problems("write", [])), 2)


if __name__ == "__main__":
    unittest.main()
