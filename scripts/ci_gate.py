#!/usr/bin/env python3
"""One required check that stands for every workflow a pull request started.

The per-crate workflows carry `paths:` filters, so none of them can be a required check:
a pull request that doesn't touch a crate never starts that crate's workflow, and a
required check that never reports blocks the merge for good. Without a required check,
though, a pull request merges with its tests red.

This gate reads the workflow runs GitHub started for the pull request's head commit and
passes only when every one of them finished well. It runs no code from the pull request;
it only calls the Actions API.

What it judges:

- The newest run of each workflow. A workflow that ran twice on one commit (editing a
  pull request body starts `commit-messages` again and cancels the first run) is judged
  by its newest run, so the superseded run's `cancelled` doesn't count.
- Only `success` passes. A run that was skipped, or that declined to judge, did not
  show that anything works.
- Every workflow in ALWAYS_PRESENT must have a run. An empty list of runs is not a pass:
  it means the runs haven't been created yet, or the query is wrong.
- A workflow in ADVISORY is reported and never judged or waited for. Its own file says
  it is a signal and not a gate.
- Nothing is pending on two polls in a row, so a workflow GitHub creates a few seconds
  late is still seen.

A failed run that is re-run and passes doesn't re-run this gate. Re-run the gate too.
"""

import json
import os
import subprocess
import sys
import time

# The one conclusion that lets a merge through. `skipped` (every job conditioned out)
# and `neutral` (a check that chose not to judge) are not evidence that anything ran, so
# they fail like any other conclusion. No workflow here ends that way today; one that is
# meant to must be named in ADVISORY, on the record.
GOOD = {"success"}

# Workflows with no `paths:` filter, so every pull request to main has a run of each.
# Their absence means the listing is incomplete, not that there is nothing to wait for.
ALWAYS_PRESENT = ("quality",)

# Workflows that declare themselves a signal and never a gate. `mutants` runs out of
# clock on large diffs and concludes `cancelled`; see the header of mutants.yml.
ADVISORY = ("mutants",)

SETTLE_SECONDS = 45
POLL_SECONDS = 30
QUIET_POLLS = 2
# Longer than the slowest judged workflow's own timeout (larql-compute-metal, 90
# minutes) plus a macOS runner queue wait.
DEADLINE_SECONDS = 170 * 60

PASS, FAIL, WAIT = "pass", "fail", "wait"


def newest_per_workflow(runs, self_run_id):
    """The newest run of each workflow, without this gate's own run."""
    newest = {}
    for run in runs:
        if run["id"] == self_run_id:
            continue
        held = newest.get(run["workflow_id"])
        if held is None or (run["created_at"], run["id"]) > (held["created_at"], held["id"]):
            newest[run["workflow_id"]] = run
    return sorted(newest.values(), key=lambda run: run["name"])


def judge(runs, self_run_id, always_present=ALWAYS_PRESENT, advisory=ADVISORY):
    """(verdict, lines) for one listing of the head commit's workflow runs."""
    judged = newest_per_workflow(runs, self_run_id)
    lines = []
    bad, pending = [], []
    for run in judged:
        if run["name"] in advisory:
            outcome = run["conclusion"] if run["status"] == "completed" else run["status"]
            lines.append(f"  advisory {run['name']} ({outcome}), not judged")
        elif run["status"] != "completed":
            pending.append(run)
            lines.append(f"  pending  {run['name']} ({run['status']})")
        elif run["conclusion"] in GOOD:
            lines.append(f"  ok       {run['name']} ({run['conclusion']})")
        else:
            bad.append(run)
            lines.append(f"  FAILED   {run['name']} ({run['conclusion']}) {run['html_url']}")

    names = {run["name"] for run in judged}
    missing = [name for name in always_present if name not in names]
    for name in missing:
        lines.append(f"  missing  {name} (no run for this commit yet)")

    if bad:
        return FAIL, lines
    if pending or missing:
        return WAIT, lines
    return PASS, lines


def list_runs(repo, sha):
    """Every workflow run GitHub holds for one head commit."""
    out = subprocess.run(
        [
            "gh", "api", "--paginate",
            f"repos/{repo}/actions/runs?head_sha={sha}&per_page=100",
            "--jq", ".workflow_runs[] | {id, name, workflow_id, status, conclusion, created_at, html_url}",
        ],
        check=True, capture_output=True, text=True,
    ).stdout
    return [json.loads(line) for line in out.splitlines() if line.strip()]


def main():
    repo = os.environ["REPO"]
    sha = os.environ["HEAD_SHA"]
    self_run_id = int(os.environ["GITHUB_RUN_ID"])

    started = time.monotonic()
    time.sleep(SETTLE_SECONDS)
    quiet = 0
    while True:
        verdict, lines = judge(list_runs(repo, sha), self_run_id)
        print(f"{sha[:8]} after {int(time.monotonic() - started)}s: {verdict}")
        print("\n".join(lines), flush=True)

        if verdict == FAIL:
            print("::error::a workflow for this commit did not pass")
            return 1
        quiet = quiet + 1 if verdict == PASS else 0
        if quiet >= QUIET_POLLS:
            return 0
        if time.monotonic() - started > DEADLINE_SECONDS:
            print("::error::workflows for this commit were still unfinished at the deadline")
            return 1
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    sys.exit(main())
