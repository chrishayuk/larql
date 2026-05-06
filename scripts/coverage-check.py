#!/usr/bin/env python3
"""
coverage-check.py — enforce per-crate coverage thresholds.

Reads:
  - coverage-thresholds.toml at repo root
  - target/llvm-cov/coverage.json (cargo-llvm-cov --json output)

Exits 1 if any non-skipped crate is below its line or branch threshold.
Otherwise prints a summary table and exits 0.

Modes:
  coverage-check.py                  # enforce
  coverage-check.py --suggest        # print thresholds matching current measurement
  coverage-check.py --json out.json  # use a different llvm-cov JSON path
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import tomllib  # py311+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

REPO_ROOT = Path(__file__).resolve().parent.parent
THRESHOLDS = REPO_ROOT / "coverage-thresholds.toml"
DEFAULT_LLVM_JSON = REPO_ROOT / "target" / "llvm-cov" / "coverage.json"


def load_thresholds(path: Path) -> dict:
    if not path.exists():
        sys.stderr.write(f"missing {path}\n")
        sys.exit(2)
    with path.open("rb") as f:
        return tomllib.load(f)


def crate_threshold(thresholds: dict, crate: str) -> dict:
    crate_section = thresholds.get("crate", {}).get(crate)
    if crate_section is not None:
        return crate_section
    return thresholds.get("default", {"line": 70, "branch": 60})


def crate_for_path(file_path: str) -> str | None:
    """Map a source file path → owning crate name (Cargo `[package].name`)."""
    p = Path(file_path)
    parts = list(p.parts)
    if "crates" in parts:
        i = parts.index("crates")
        if i + 1 < len(parts):
            return parts[i + 1]
    return None


def coverage_per_crate(llvm_json: dict) -> dict:
    """Aggregate llvm-cov per-file results into per-crate totals."""
    totals: dict[str, dict] = {}
    data = llvm_json.get("data") or [llvm_json]
    for entry in data:
        for f in entry.get("files", []):
            file_path = f.get("filename") or ""
            crate = crate_for_path(file_path)
            if not crate:
                continue
            summary = f.get("summary", {})
            t = totals.setdefault(
                crate,
                {
                    "line_covered": 0,
                    "line_count": 0,
                    "branch_covered": 0,
                    "branch_count": 0,
                    "region_covered": 0,
                    "region_count": 0,
                    "files": 0,
                },
            )
            t["files"] += 1
            ls = summary.get("lines", {})
            t["line_covered"] += ls.get("covered", 0)
            t["line_count"] += ls.get("count", 0)
            bs = summary.get("branches", {}) or summary.get("regions", {})
            # llvm-cov --json reports 'branches' on newer versions; fall back to regions.
            t["branch_covered"] += bs.get("covered", 0)
            t["branch_count"] += bs.get("count", 0)
            rs = summary.get("regions", {})
            t["region_covered"] += rs.get("covered", 0)
            t["region_count"] += rs.get("count", 0)
    # Compute percentages
    for crate, t in totals.items():
        t["line_pct"] = pct(t["line_covered"], t["line_count"])
        t["branch_pct"] = pct(t["branch_covered"], t["branch_count"])
    return totals


def pct(num: int, denom: int) -> float:
    if denom == 0:
        return 100.0
    return 100.0 * num / denom


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--json", default=str(DEFAULT_LLVM_JSON), help="cargo-llvm-cov JSON path")
    p.add_argument("--suggest", action="store_true", help="emit suggested thresholds at current coverage")
    args = p.parse_args(argv)

    thresholds = load_thresholds(THRESHOLDS)
    llvm_path = Path(args.json)
    if not llvm_path.exists():
        sys.stderr.write(
            f"{llvm_path} not found.\n"
            f"Run `make coverage` first to produce target/llvm-cov/coverage.json.\n"
        )
        return 2

    with llvm_path.open() as f:
        llvm_json = json.load(f)

    per_crate = coverage_per_crate(llvm_json)

    if args.suggest:
        sys.stdout.write("# Suggested thresholds — measured coverage rounded down to nearest 5% minus 1%.\n")
        for crate in sorted(per_crate):
            t = per_crate[crate]
            ls = max(0, int(t["line_pct"] // 5 * 5) - 1)
            bs = max(0, int(t["branch_pct"] // 5 * 5) - 1)
            sys.stdout.write(f"[crate.{crate}]\nline = {ls}\nbranch = {bs}\n\n")
        return 0

    fmt = "{:30s}  {:>8s}  {:>8s}  {:>10s}  {:>10s}  {:>6s}"
    sys.stdout.write(fmt.format("crate", "line%", "branch%", "thr_line", "thr_branch", "ok") + "\n")
    sys.stdout.write("-" * 78 + "\n")
    failed = False
    for crate in sorted(per_crate):
        rule = crate_threshold(thresholds, crate)
        t = per_crate[crate]
        skipped = bool(rule.get("skip"))
        thr_line = int(rule.get("line", 0))
        thr_branch = int(rule.get("branch", 0))
        ok = (
            skipped
            or (t["line_pct"] + 1e-9 >= thr_line and t["branch_pct"] + 1e-9 >= thr_branch)
        )
        status = "skip" if skipped else ("OK" if ok else "FAIL")
        sys.stdout.write(
            fmt.format(
                crate,
                f"{t['line_pct']:.1f}",
                f"{t['branch_pct']:.1f}",
                str(thr_line),
                str(thr_branch),
                status,
            )
            + "\n"
        )
        if not ok:
            failed = True

    if failed:
        sys.stderr.write("\ncoverage-check: FAILED — one or more crates below their threshold.\n")
        sys.stderr.write("Update tests, or raise/lower the threshold in coverage-thresholds.toml with reviewer ack.\n")
        return 1

    sys.stdout.write("\ncoverage-check: OK\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
