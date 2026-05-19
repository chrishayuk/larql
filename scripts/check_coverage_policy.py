#!/usr/bin/env python3
"""Check cargo-llvm-cov JSON output against a per-file coverage policy."""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from pathlib import Path
from typing import Any


# Tolerance against `cargo-llvm-cov` measurement noise. The 0.01% original
# value was tight enough that the same code reported below-floor on
# different `rustc` / `llvm-cov` builds even with no edits to the file
# under test (grpc.rs 64.64 vs floor 64.70, walk_ffn.rs 48.97 vs 49.00
# on the 2026-05-11 main run). 0.1% absorbs that without letting a real
# regression of a few percent through.
EPSILON = 0.1


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def repo_relative(path: str, repo_root: Path) -> str:
    source_path = Path(path)
    try:
        return source_path.resolve().relative_to(repo_root).as_posix()
    except ValueError:
        return source_path.as_posix()


def matches_any(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def line_summary(file_entry: dict[str, Any]) -> tuple[int, float]:
    lines = file_entry["summary"]["lines"]
    return int(lines["count"]), float(lines["percent"])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate cargo-llvm-cov JSON summary against a coverage policy."
    )
    parser.add_argument("report", type=Path, help="cargo-llvm-cov JSON summary")
    parser.add_argument("policy", type=Path, help="coverage policy JSON")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="repository root used to normalize absolute paths",
    )
    args = parser.parse_args()

    report = load_json(args.report)
    policy = load_json(args.policy)
    repo_root = args.repo_root.resolve()

    include_globs = policy.get("include_globs", [])
    exclude_globs = policy.get("exclude_globs", [])
    default_min = float(policy["default_line_min_percent"])
    total_min = float(policy.get("total_line_min_percent", 0.0))
    # Optional: floor computed over included files only (excludes I/O
    # wrappers + main.rs + similar from the denominator). When this is
    # set, it's checked in addition to `total_line_min_percent`. Useful
    # for binary crates where a chunk of code is intentionally untested.
    included_total_min = float(policy.get("included_total_line_min_percent", 0.0))
    per_file_min = {
        str(path): float(minimum)
        for path, minimum in policy.get("per_file_line_min_percent", {}).items()
    }

    data = report["data"][0]
    total_percent = float(data["totals"]["lines"]["percent"])
    failures: list[str] = []

    if total_percent + EPSILON < total_min:
        failures.append(
            f"TOTAL lines {total_percent:.2f}% below minimum {total_min:.2f}%"
        )

    checked = 0
    debt = 0
    seen: set[str] = set()
    included_covered = 0
    included_count = 0
    for file_entry in data["files"]:
        rel_path = repo_relative(file_entry["filename"], repo_root)
        if include_globs and not matches_any(rel_path, include_globs):
            continue
        if matches_any(rel_path, exclude_globs):
            continue

        line_count, line_percent = line_summary(file_entry)
        if line_count == 0:
            continue

        checked += 1
        seen.add(rel_path)
        minimum = per_file_min.get(rel_path, default_min)
        if minimum < default_min:
            debt += 1
        if line_percent + EPSILON < minimum:
            failures.append(
                f"{rel_path}: lines {line_percent:.2f}% below minimum {minimum:.2f}%"
            )

        # Sum for the included-only total. cargo-llvm-cov reports
        # `count` (total instrumented lines) and `covered` (hits) per
        # file; if `covered` isn't present, derive it from percent.
        lines_block = file_entry["summary"]["lines"]
        if "covered" in lines_block:
            covered = int(lines_block["covered"])
        else:
            covered = int(round(line_count * line_percent / 100.0))
        included_count += line_count
        included_covered += covered

    included_pct = (
        100.0 * included_covered / included_count if included_count > 0 else 0.0
    )
    if included_total_min > 0 and included_pct + EPSILON < included_total_min:
        failures.append(
            f"INCLUDED total lines {included_pct:.2f}% below minimum {included_total_min:.2f}%"
        )

    stale = sorted(set(per_file_min) - seen)
    for rel_path in stale:
        failures.append(f"{rel_path}: policy entry did not match any covered file")

    if failures:
        print("Coverage policy failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    at_default = checked - debt
    included_note = (
        f", included {included_pct:.2f}%" if included_total_min > 0 else ""
    )
    print(
        "Coverage policy passed: "
        f"total {total_percent:.2f}% lines{included_note}, "
        f"{checked} files checked, "
        f"{at_default} files at {default_min:.1f}% default, "
        f"{debt} debt baselines."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
