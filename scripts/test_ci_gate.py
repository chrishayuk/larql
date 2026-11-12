"""The gate's verdict on a listing of workflow runs."""

import unittest
import zlib

import ci_gate
from ci_gate import FAIL, PASS, WAIT, judge

SELF = 999


def run(name, conclusion="success", *, status="completed", run_id=None, workflow_id=None,
        created_at="2026-11-10T00:00:00Z", path=None):
    workflow_id = workflow_id if workflow_id is not None else zlib.crc32(name.encode())
    return {
        "id": run_id if run_id is not None else workflow_id * 10,
        "name": name,
        "path": path if path is not None else f".github/workflows/{name}.yml",
        "workflow_id": workflow_id,
        "status": status,
        "conclusion": conclusion if status == "completed" else None,
        "created_at": created_at,
        "html_url": f"https://example.invalid/{name}",
    }


class Verdict(unittest.TestCase):
    def test_every_workflow_green_passes(self):
        verdict, _ = judge([run("quality"), run("larql-cli"), run("larql-lql")], SELF)
        self.assertEqual(verdict, PASS)

    def test_one_failed_workflow_fails_and_is_named(self):
        verdict, lines = judge([run("quality"), run("larql-cli", "failure")], SELF)
        self.assertEqual(verdict, FAIL)
        self.assertTrue(any("FAILED" in line and "larql-cli" in line for line in lines))

    def test_every_conclusion_outside_the_good_set_fails(self):
        for conclusion in ("failure", "cancelled", "timed_out", "action_required",
                           "startup_failure", "stale", "skipped", "neutral", None,
                           "a-conclusion-added-later"):
            with self.subTest(conclusion=conclusion):
                verdict, _ = judge([run("quality"), run("larql-cli", conclusion)], SELF)
                self.assertEqual(verdict, FAIL)

    def test_a_skipped_always_present_workflow_is_not_a_pass(self):
        # Present in the listing, and still no evidence that it ran anything.
        verdict, _ = judge([run("quality", "skipped"), run("larql-cli")], SELF)
        self.assertEqual(verdict, FAIL)

    def test_an_unfinished_workflow_waits(self):
        for status in ("queued", "in_progress", "requested", "waiting", "pending"):
            with self.subTest(status=status):
                verdict, _ = judge([run("quality"), run("larql-cli", status=status)], SELF)
                self.assertEqual(verdict, WAIT)

    def test_a_failure_beats_an_unfinished_workflow(self):
        verdict, _ = judge(
            [run("quality"), run("a", "failure"), run("b", status="in_progress")], SELF
        )
        self.assertEqual(verdict, FAIL)


class EmptyIsNotGreen(unittest.TestCase):
    def test_no_runs_at_all_waits(self):
        verdict, lines = judge([], SELF)
        self.assertEqual(verdict, WAIT)
        self.assertTrue(any("missing" in line and "quality" in line for line in lines))

    def test_only_the_gate_itself_waits(self):
        verdict, _ = judge([run("ci-gate", status="in_progress", run_id=SELF)], SELF)
        self.assertEqual(verdict, WAIT)

    def test_green_runs_without_an_always_present_workflow_wait(self):
        verdict, _ = judge([run("larql-cli"), run("larql-lql")], SELF)
        self.assertEqual(verdict, WAIT)

    def test_the_always_present_list_names_a_workflow_that_exists(self):
        # A name nothing runs under would make the gate wait until its deadline.
        import pathlib
        import re

        workflows = pathlib.Path(__file__).resolve().parent.parent / ".github" / "workflows"
        names = {
            match.group(1).strip().strip("'\"")
            for path in workflows.glob("*.yml")
            for match in [re.search(r"^name:\s*(.+)$", path.read_text(), re.M)]
            if match
        }
        for name in ci_gate.ALWAYS_PRESENT + ci_gate.ADVISORY:
            self.assertIn(name, names)


class Advisory(unittest.TestCase):
    def test_a_failed_or_cancelled_advisory_workflow_does_not_fail_the_gate(self):
        for conclusion in ("failure", "cancelled", "timed_out"):
            with self.subTest(conclusion=conclusion):
                verdict, lines = judge([run("quality"), run("mutants", conclusion)], SELF)
                self.assertEqual(verdict, PASS)
                self.assertTrue(any("advisory" in line and conclusion in line for line in lines))

    def test_an_unfinished_advisory_workflow_is_not_waited_for(self):
        verdict, _ = judge([run("quality"), run("mutants", status="in_progress")], SELF)
        self.assertEqual(verdict, PASS)

    def test_an_advisory_workflow_cannot_stand_in_for_an_always_present_one(self):
        verdict, _ = judge([run("mutants")], SELF, always_present=("mutants",))
        self.assertEqual(verdict, PASS)
        verdict, _ = judge([run("mutants")], SELF)
        self.assertEqual(verdict, WAIT)

    def test_the_same_failure_under_another_name_still_fails(self):
        verdict, _ = judge([run("quality"), run("larql-cli", "cancelled")], SELF)
        self.assertEqual(verdict, FAIL)


AI_SCAN_PATH = "dynamic/agents/github-advanced-security"


def ai_scan(pr, conclusion="failure", **kwargs):
    """The "Code scanning AI findings" run GitHub starts on a pull request."""
    return run(f"Code scanning AI findings on PR #{pr}", conclusion, path=AI_SCAN_PATH,
               **kwargs)


class GithubStartedAdvisoryRuns(unittest.TestCase):
    def test_a_failed_ai_scan_does_not_fail_the_gate(self):
        # PR 798's scan concluded `failure` with `402 You have no quota`.
        verdict, lines = judge([run("quality"), ai_scan(798)], SELF)
        self.assertEqual(verdict, PASS)
        self.assertTrue(any("advisory" in line and "PR #798" in line for line in lines))

    def test_an_unfinished_ai_scan_is_not_waited_for(self):
        verdict, _ = judge([run("quality"), ai_scan(798, status="in_progress")], SELF)
        self.assertEqual(verdict, PASS)

    def test_a_workflow_file_with_the_same_name_is_still_judged(self):
        # Only GitHub's own run is advisory; a file that copies its name is not.
        impostor = run("Code scanning AI findings on PR #798", "failure")
        verdict, _ = judge([run("quality"), impostor], SELF)
        self.assertEqual(verdict, FAIL)

    def test_another_github_started_run_is_still_judged(self):
        codeql = run("CodeQL", "failure", path="dynamic/github-code-scanning/codeql")
        verdict, _ = judge([run("quality"), codeql], SELF)
        self.assertEqual(verdict, FAIL)

    def test_an_ai_scan_cannot_stand_in_for_an_always_present_workflow(self):
        verdict, _ = judge([ai_scan(798, "success")], SELF)
        self.assertEqual(verdict, WAIT)

    def test_a_listing_without_paths_is_judged_by_name_alone(self):
        # An older listing had no `path`; nothing in it becomes advisory by accident.
        bare = run("larql-cli", "failure")
        del bare["path"]
        verdict, _ = judge([run("quality"), bare], SELF)
        self.assertEqual(verdict, FAIL)


class NewestRunOfEachWorkflow(unittest.TestCase):
    def test_a_superseded_cancelled_run_does_not_count(self):
        runs = [
            run("quality"),
            run("commit-messages", "cancelled", workflow_id=7, run_id=70,
                created_at="2026-11-10T00:00:00Z"),
            run("commit-messages", "success", workflow_id=7, run_id=71,
                created_at="2026-11-10T00:05:00Z"),
        ]
        self.assertEqual(judge(runs, SELF)[0], PASS)

    def test_a_newer_failure_replaces_an_older_success(self):
        runs = [
            run("quality"),
            run("larql-cli", "success", workflow_id=7, run_id=70,
                created_at="2026-11-10T00:00:00Z"),
            run("larql-cli", "failure", workflow_id=7, run_id=71,
                created_at="2026-11-10T00:05:00Z"),
        ]
        self.assertEqual(judge(runs, SELF)[0], FAIL)

    def test_a_newer_unfinished_run_replaces_an_older_success(self):
        runs = [
            run("quality"),
            run("larql-cli", "success", workflow_id=7, run_id=70,
                created_at="2026-11-10T00:00:00Z"),
            run("larql-cli", status="queued", workflow_id=7, run_id=71,
                created_at="2026-11-10T00:05:00Z"),
        ]
        self.assertEqual(judge(runs, SELF)[0], WAIT)

    def test_runs_created_in_the_same_second_are_ordered_by_id(self):
        runs = [
            run("quality"),
            run("larql-cli", "failure", workflow_id=7, run_id=71),
            run("larql-cli", "success", workflow_id=7, run_id=70),
        ]
        self.assertEqual(judge(runs, SELF)[0], FAIL)

    def test_two_workflows_sharing_a_name_are_both_judged(self):
        runs = [
            run("quality"),
            run("test", "success", workflow_id=1),
            run("test", "failure", workflow_id=2),
        ]
        self.assertEqual(judge(runs, SELF)[0], FAIL)

    def test_the_gates_own_failed_earlier_attempt_is_ignored(self):
        runs = [run("quality"), run("ci-gate", "failure", run_id=SELF)]
        self.assertEqual(judge(runs, SELF)[0], PASS)


if __name__ == "__main__":
    unittest.main()
