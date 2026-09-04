#!/usr/bin/env python3
"""Every coverage-producing make target must own its measurement state.

The invariant: a coverage gate must not consume measurement artifacts
originating from a different source tree. On 2026-09-04 a gate reported
75.94% for a file untouched by the branch (91.50% after a clean) and named
a file absent from the checked-out tree. The exact trigger is unreproduced
— two candidate mechanisms were tested and excluded — so the guarantee is
structural rather than a fix for one known path.

This checks the shape that keeps it, because the property lives in 24
separate recipes and a 25th will be added one day:

    1. COVERAGE_CLEAN is defined and uses --workspace
    2. every `larql-*-coverage[-summary]` target runs it BEFORE measuring
    3. it is never used as a PREREQUISITE
    4. .NOTPARALLEL is present

(3) is not a style rule. Make builds a prerequisite once per invocation, so
`make a-coverage b-coverage` would clean once and hand `b` whatever `a`
produced — the defect this file exists to prevent, wearing the costume of
the fix. The lifecycle is per measurement, never per invocation.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

TARGET = re.compile(r"^(larql-[a-z0-9-]+-coverage(?:-summary)?):")
CLEAN_CALL = "$(COVERAGE_CLEAN)"


def main() -> int:
    makefile = Path(sys.argv[1] if len(sys.argv) > 1 else "Makefile")
    lines = makefile.read_text().split("\n")
    failures: list[str] = []

    definition = [l for l in lines if l.startswith("COVERAGE_CLEAN")]
    if not definition:
        failures.append("COVERAGE_CLEAN is not defined")
    elif "--workspace" not in definition[0]:
        failures.append(
            "COVERAGE_CLEAN must use --workspace: --profraw-only leaves stale "
            "test binaries whose coverage maps still merge"
        )

    if not any(l.startswith(".NOTPARALLEL:") for l in lines):
        failures.append(
            ".NOTPARALLEL is absent: coverage gates share one measurement "
            "store and must not run concurrently"
        )

    checked = 0
    for i, line in enumerate(lines):
        match = TARGET.match(line)
        if not match:
            continue
        name = match.group(1)
        checked += 1

        body: list[str] = []
        j = i + 1
        while j < len(lines) and (lines[j].startswith("\t") or not lines[j].strip()):
            if lines[j].strip():
                body.append(lines[j].strip())
            j += 1

        clean_at = next((k for k, b in enumerate(body) if b == CLEAN_CALL), None)
        measure_at = next(
            (
                k
                for k, b in enumerate(body)
                if b.startswith("cargo llvm-cov") and " clean" not in b
            ),
            None,
        )
        if clean_at is None:
            failures.append(f"{name}: does not run {CLEAN_CALL}")
        elif measure_at is not None and clean_at > measure_at:
            failures.append(f"{name}: measures before it cleans")

        # The rejected design: a clean that is a dependency, not a step.
        if CLEAN_CALL in line or "coverage-clean" in line.split(":", 1)[1]:
            failures.append(
                f"{name}: cleans via a PREREQUISITE. Make runs it once per "
                "invocation, so a second coverage target in the same "
                "invocation inherits this one's artifacts. Make it a recipe "
                "line instead."
            )

    if not checked:
        failures.append("no coverage targets found — has the naming changed?")

    if failures:
        print("Coverage gate isolation failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print(
        f"Coverage gate isolation passed: {checked} targets, each establishing "
        "its own clean measurement state."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
