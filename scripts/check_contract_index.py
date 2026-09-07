#!/usr/bin/env python3
"""Verify every test named in the v1 contract index still exists.

A freeze that names tests is only worth something if the names are true.
This reads `docs/represent-v1-contract-index.md`, extracts every
`name` / `path` pair from its tables, and fails when a test cannot be
found at the path recorded — which is exactly what a rename or a move
would cause.

It checks EXISTENCE and LOCATION, not outcome: `cargo test` decides
whether the contracts hold. This decides whether the index is honest
about where they are pinned.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

INDEX = Path("docs/represent-v1-contract-index.md")
# `…/exec/` and `…/codec/` are abbreviations the document defines.
ABBREVIATIONS = {
    "…/exec/": "crates/larql-vindex/src/format/vindex3/opplan/exec/",
    "…/codec/": "crates/larql-vindex/src/format/vindex3/represent/codec/",
    "…/auxiliary_references/": "crates/larql-vindex/src/format/vindex3/auxiliary_references/",
    "…/representation_attestations/": (
        "crates/larql-vindex/src/format/vindex3/representation_attestations/"
    ),
}
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


def expand(path: str) -> str:
    for short, full in ABBREVIATIONS.items():
        if path.startswith(short):
            return full + path[len(short) :]
    return path


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    index = root / INDEX
    if not index.is_file():
        print(f"{INDEX} not found", file=sys.stderr)
        return 2

    missing: list[str] = []
    checked = 0
    last_path: str | None = None

    for number, line in enumerate(index.read_text().splitlines(), start=1):
        pairs: list[tuple[str, str]] = []
        cites = bool(CITATION.search(line)) and not MUTANT.search(line)
        for name, path in ROW.findall(line):
            last_path = expand(path)
            pairs.append((name, last_path))
        if not pairs:
            for name in SAME_FILE.findall(line):
                if last_path is None:
                    missing.append(f"line {number}: `{name}` says 'same file' with no file yet")
                    continue
                pairs.append((name, last_path))

        if cites and not pairs:
            missing.append(
                f"line {number}: cites a test but no `name` — `path` pair could be read "
                f"from it, so nothing was checked: {line.strip()[:90]}"
            )
        for name, path in pairs:
            checked += 1
            target = root / path
            if not target.is_file():
                missing.append(f"line {number}: {path} does not exist (for `{name}`)")
                continue
            if not re.search(rf"^\s*fn {re.escape(name)}\b", target.read_text(), re.M):
                missing.append(f"line {number}: {path} has no `fn {name}`")

    if checked == 0:
        print("the index named no tests — the parser is looking at the wrong shape", file=sys.stderr)
        return 2
    for problem in missing:
        print(problem, file=sys.stderr)
    if missing:
        print(
            f"\n{len(missing)} of {checked} named tests could not be found. "
            "Either the move was unintended, or this index must be updated in the "
            "same change — a freeze that names tests that no longer exist is worse "
            "than no freeze.",
            file=sys.stderr,
        )
        return 1
    print(f"all {checked} tests named by the v1 contract index exist where it says")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
