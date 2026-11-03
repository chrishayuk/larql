#!/usr/bin/env python3
"""The gate between the engine and the verifier that measures it.

One pull request must not change both, or it could make a wrong result look right. A
change that really needs both is let through only by a maintainer, on the record: a
maintainer applies the label, and a maintainer writes down why.

GitHub lets anyone with triage access apply a label, so the label alone proves nothing.
This asks GitHub who applied it. It runs from the base branch and never runs the pull
request's code. See docs/verification/independence.md.
"""

import json
import os
import re
import subprocess
import sys

MEASUREMENT = re.compile(
    r"^verifier/(refcheck|larql_contract_verifier|expected|provenance)/"
    r"|^verifier/(pyproject\.toml|requirements\.lock)$"
    r"|^scripts/check_verifier_boundary\.py$"
    r"|^\.github/workflows/verifier-boundary\.yml$"
)
ENGINE = re.compile(r"^crates/")
LABEL = "verifier-and-engine"
JUSTIFICATION = re.compile(r"^\s*Verifier-and-engine justification:[ \t]*\S", re.M)
MAINTAINER_ROLES = {"admin", "maintain"}
GITHUB_FILE_LIST_LIMIT = 3000


def sides(paths):
    """The changed paths on each side of the boundary."""
    return (
        sorted(p for p in paths if MEASUREMENT.search(p)),
        sorted(p for p in paths if ENGINE.search(p)),
    )


def override_problems(labeller_role, statements):
    """Why the override doesn't hold, given the labeller's role and (role, text) statements."""
    problems = []
    if labeller_role not in MAINTAINER_ROLES:
        problems.append(f"the {LABEL} label must be applied by a maintainer or admin, not role {labeller_role!r}")
    if not any(role in MAINTAINER_ROLES and JUSTIFICATION.search(text or "") for role, text in statements):
        problems.append('a maintainer must comment or review with a line starting "Verifier-and-engine justification:"')
    return problems


def api(path, jq):
    """One JSON value per output line, through every page. Any failure stops the gate."""
    out = subprocess.run(
        ["gh", "api", "--paginate", path, "--jq", f"{jq} | @json"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [json.loads(line) for line in out.splitlines()]


def role_of(repo, login):
    """A login's role in this repository, or None if GitHub won't say."""
    try:
        return api(f"repos/{repo}/collaborators/{login}/permission", ".role_name")[0]
    except (subprocess.CalledProcessError, IndexError):
        return None


def check(repo, number):
    """The problems with this pull request, read live rather than from the triggering event."""
    pull = f"repos/{repo}/pulls/{number}"
    issue = f"repos/{repo}/issues/{number}"

    changed = api(pull, ".changed_files")[0]
    if changed > GITHUB_FILE_LIST_LIMIT:
        return [f"{changed} changed files is more than GitHub will list, so this can't be checked"]

    paths = api(f"{pull}/files", ".[] | .filename, (.previous_filename // empty)")
    measurement, engine = sides(paths)
    if not (measurement and engine):
        return []

    print("Measurement:", *measurement, "", "Engine:", *engine, "", sep="\n")
    if LABEL not in api(f"{issue}/labels", ".[].name"):
        return [f"this changes both the verifier's measurement and the engine: split it, or see the {LABEL} override"]

    applied = api(f"{issue}/events", f'.[] | select(.event == "labeled" and .label.name == "{LABEL}") | .actor.login')
    labeller_role = role_of(repo, applied[-1]) if applied else None

    written = api(f"{issue}/comments", ".[] | [.user.login, .body]")
    written += api(f"{pull}/reviews", '.[] | select(.state != "DISMISSED") | [.user.login, .body]')
    roles = {login: role_of(repo, login) for login in {login for login, _ in written}}
    return override_problems(labeller_role, [(roles[login], text) for login, text in written])


def main():
    problems = check(os.environ["REPO"], os.environ["PR"])
    if problems:
        for problem in problems:
            print(f"::error::{problem}. See docs/verification/independence.md.")
        return 1
    print("Boundary holds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
