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


class ReviewEvidence(unittest.TestCase):
    HEAD = "abcdef0123456789abcdef0123456789abcdef01"

    def evidence(self, approvals=(), statements=()):
        return gate.has_review_evidence("author", self.HEAD, approvals, statements)

    def exception(self, sha=None, reason="the only maintainer, reviewed by a fresh-context model"):
        return f"Verifier-review exception ({sha or self.HEAD[:12]}): {reason}"

    def test_what_needs_it(self):
        for path in (
            "verifier/README.md",
            "verifier/tests/test_e1.py",
            "docs/verification/independence.md",
            "scripts/check_verifier_boundary.py",
            ".github/workflows/verifier.yml",
            ".github/workflows/verifier-boundary.yml",
            ".github/CODEOWNERS",
        ):
            self.assertEqual(gate.owned([path]), [path], path)

    def test_what_does_not(self):
        for path in ("crates/x/src/lib.rs", "docs/spec.md", "scripts/x.py", ".github/workflows/quality.yml", "xverifier/a"):
            self.assertEqual(gate.owned([path]), [], path)

    def test_nothing_on_the_record_is_refused(self):
        self.assertFalse(self.evidence())

    def test_another_maintainers_approval_of_the_head_counts(self):
        for role in ("admin", "maintain"):
            self.assertTrue(self.evidence(approvals=[("other", role, self.HEAD)]))

    def test_an_approval_of_an_older_commit_does_not_count(self):
        self.assertFalse(self.evidence(approvals=[("other", "admin", "0" * 40)]))

    def test_an_authors_own_approval_does_not_count(self):
        self.assertFalse(self.evidence(approvals=[("author", "admin", self.HEAD)]))

    def test_an_approval_from_anyone_else_does_not_count(self):
        for role in ("write", "triage", "read", None):
            self.assertFalse(self.evidence(approvals=[("other", role, self.HEAD)]), role)

    def test_a_maintainers_exception_naming_the_head_counts(self):
        self.assertTrue(self.evidence(statements=[("admin", self.exception())]))
        self.assertTrue(self.evidence(statements=[("admin", "Reviewed.\n\n" + self.exception(self.HEAD) + "\n")]))

    def test_an_exception_for_an_older_commit_does_not_count(self):
        self.assertFalse(self.evidence(statements=[("admin", self.exception("1234567"))]))

    def test_an_exception_from_anyone_else_does_not_count(self):
        for role in ("write", "triage", "read", None):
            self.assertFalse(self.evidence(statements=[(role, self.exception())]), role)

    def test_an_exception_needs_a_reason_and_a_real_sha(self):
        for text in (self.exception(reason=""), f"Verifier-review exception: because", self.exception("abc"), None, ""):
            self.assertFalse(self.evidence(statements=[("admin", text)]), repr(text))


if __name__ == "__main__":
    unittest.main()
