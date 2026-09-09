#!/usr/bin/env python3
"""Verify every test named by a contract index still exists.

A freeze that names tests is only worth something if the names are true.
This reads each contract index below, extracts every `name` / `path` pair
from its tables, and fails when a test cannot be found at the path
recorded — which is exactly what a rename or a move would cause.

It checks EXISTENCE and LOCATION, not outcome: `cargo test` decides
whether the contracts hold. This decides whether the index is honest
about where they are pinned.

There is more than one index because there is more than one freeze. The
REPRESENT/accounting v1 set and the optimizer set are different programmes
with different qualifying commits, so each carries its OWN abbreviations —
`…/state/` in one index must not silently resolve against the other's
vocabulary.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPRESENT = "crates/larql-vindex/src/format/vindex3/represent/"
# Each index and the abbreviations that index defines for itself.
INDEXES: list[tuple[Path, dict[str, str]]] = [
    (
        Path("docs/represent-v1-contract-index.md"),
        {
            "…/exec/": "crates/larql-vindex/src/format/vindex3/opplan/exec/",
            "…/codec/": REPRESENT + "codec/",
            "…/auxiliary_references/": (
                "crates/larql-vindex/src/format/vindex3/auxiliary_references/"
            ),
            "…/representation_attestations/": (
                "crates/larql-vindex/src/format/vindex3/representation_attestations/"
            ),
        },
    ),
    (
        Path("docs/optimizer-contract-index.md"),
        {
            "…/actuate/": REPRESENT + "actuate/",
            "…/state/": REPRESENT + "state/",
        },
    ),
]
# `name` — `path`, where the path ends in `.rs`. The name pattern is
# deliberately permissive: a citation whose name does not match must be
# reported as MISSING, never quietly skipped.
ROW = re.compile(r"`([A-Za-z0-9_]+)`\s+—\s+`([^`]+\.rs)`")
SAME_FILE = re.compile(r"`([A-Za-z0-9_]+)`\s+—\s+same file")
# Any table row that cites something. If one of these yields no name, the
# row is malformed and the checker must FAIL rather than silently check
# one fewer test — a checker that fails open is not a checker.
CITATION = re.compile(r"^\|.*(`[^`]+\.rs`|same file)")
# Rows that describe a source mutation rather than naming a test.
MUTANT = re.compile(r"^\|\s*mutant\s*\|")


def expand(path: str, abbreviations: dict[str, str]) -> str:
    for short, full in abbreviations.items():
        if path.startswith(short):
            return full + path[len(short) :]
    return path


def check(root: Path, index: Path, abbreviations: dict[str, str]) -> tuple[int, list[str]]:
    """Return how many tests this index named, and everything wrong with it."""
    missing: list[str] = []
    checked = 0
    # Reset per index: a `same file` row may only inherit a path named by
    # the SAME document.
    last_path: str | None = None

    for number, line in enumerate((root / index).read_text().splitlines(), start=1):
        pairs: list[tuple[str, str]] = []
        cites = bool(CITATION.search(line)) and not MUTANT.search(line)
        for name, path in ROW.findall(line):
            last_path = expand(path, abbreviations)
            pairs.append((name, last_path))
        if not pairs:
            for name in SAME_FILE.findall(line):
                if last_path is None:
                    missing.append(f"{index}:{number}: `{name}` says 'same file' with no file yet")
                    continue
                pairs.append((name, last_path))

        if cites and not pairs:
            missing.append(
                f"{index}:{number}: cites a test but no `name` — `path` pair could be read "
                f"from it, so nothing was checked: {line.strip()[:90]}"
            )
        for name, path in pairs:
            checked += 1
            target = root / path
            if not target.is_file():
                missing.append(f"{index}:{number}: {path} does not exist (for `{name}`)")
                continue
            if not re.search(rf"^\s*fn {re.escape(name)}\b", target.read_text(), re.M):
                missing.append(f"{index}:{number}: {path} has no `fn {name}`")

    return checked, missing


def main() -> int:
    root = Path(__file__).resolve().parent.parent

    total = 0
    problems: list[str] = []
    for index, abbreviations in INDEXES:
        if not (root / index).is_file():
            print(f"{index} not found", file=sys.stderr)
            return 2
        checked, missing = check(root, index, abbreviations)
        if checked == 0:
            print(
                f"{index} named no tests — the parser is looking at the wrong shape",
                file=sys.stderr,
            )
            return 2
        total += checked
        problems.extend(missing)

    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        print(
            f"\n{len(problems)} of {total} named tests could not be found. "
            "Either the move was unintended, or the index must be updated in the "
            "same change — a freeze that names tests that no longer exist is worse "
            "than no freeze.",
            file=sys.stderr,
        )
        return 1
    names = ", ".join(index.name for index, _ in INDEXES)
    print(f"all {total} tests named by {names} exist where they say")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
