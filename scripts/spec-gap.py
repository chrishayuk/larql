#!/usr/bin/env python3
"""
spec-gap.py — emit OpenSpec gap reports.

Two reports:

  --untested-code     openspec/changes/<change>/gaps-untested-code.md
                      Lists every public Rust symbol whose owning file has
                      0% line coverage in target/llvm-cov/coverage.json.
                      Sorted by largest gap first per crate.

                      If `target/llvm-cov/coverage.json` is missing, falls
                      back to a structural heuristic: every pub fn/pub
                      struct/pub enum in a `.rs` file with no `#[test]`
                      attribute anywhere is flagged as "no inline tests"
                      with a note that the LLVM coverage gate hasn't run.

  --unbacked          Forwarded to spec-trace.py --unbacked.

Optional:
  --change-dir PATH   override which change to write into (default: the
                      first non-archive directory under openspec/changes/,
                      preferring `backfill-specs`).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CRATES_DIR = REPO_ROOT / "crates"
LLVM_JSON = REPO_ROOT / "target" / "llvm-cov" / "coverage.json"


def find_change_dir(override: str | None) -> Path:
    if override:
        p = Path(override).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p
    base = REPO_ROOT / "openspec" / "changes"
    if not base.exists():
        sys.stderr.write("openspec/changes does not exist.\n")
        sys.exit(2)
    candidates = [
        d for d in base.iterdir() if d.is_dir() and d.name != "archive"
    ]
    for c in candidates:
        if c.name == "backfill-specs":
            return c
    if candidates:
        return candidates[0]
    sys.stderr.write("no active change directory under openspec/changes/.\n")
    sys.exit(2)


# ---------- structural heuristic ---------------------------------------------


PUB_SYMBOL_RE = re.compile(
    r"^\s*pub(?:\([^)]*\))?\s+"
    r"(?:async\s+)?"
    r"(?:unsafe\s+)?"
    r"(fn|struct|enum|trait|type|const|static|union)\s+"
    r"([A-Za-z_][A-Za-z0-9_]*)"
)
TEST_ATTR_RE = re.compile(r"#\[(?:tokio::test|test|rstest|serial_test::serial)\b")


def crate_for_path(path: Path) -> str | None:
    parts = list(path.relative_to(REPO_ROOT).parts)
    if "crates" in parts:
        i = parts.index("crates")
        if i + 1 < len(parts):
            return parts[i + 1]
    return None


def scan_pub_symbols(rs: Path) -> list[tuple[str, str, int]]:
    """Return [(kind, name, line)] for pub symbols defined in rs."""
    out: list[tuple[str, str, int]] = []
    try:
        text = rs.read_text(errors="replace")
    except Exception:
        return out
    for line_no, line in enumerate(text.splitlines(), 1):
        m = PUB_SYMBOL_RE.match(line)
        if m:
            out.append((m.group(1), m.group(2), line_no))
    return out


def file_has_tests(rs: Path) -> bool:
    try:
        text = rs.read_text(errors="replace")
    except Exception:
        return False
    return bool(TEST_ATTR_RE.search(text))


def has_test_companion(rs: Path) -> bool:
    """Cheap heuristic: file under tests/ next to the crate, or any test fn
    in the workspace mentions the file's module name."""
    return False  # used only as a tie-breaker; structural pass is good enough


def structural_untested_code() -> dict[str, list[dict]]:
    """Group symbols-without-inline-tests by crate."""
    out: dict[str, list[dict]] = defaultdict(list)
    if not CRATES_DIR.exists():
        return out
    for rs in CRATES_DIR.rglob("*.rs"):
        rp = rs.relative_to(REPO_ROOT).as_posix()
        if "/target/" in rp or "/.git/" in rp:
            continue
        if "/tests/" in rp:
            continue  # the test files themselves
        crate = crate_for_path(rs)
        if not crate:
            continue
        if file_has_tests(rs):
            continue  # has at least one inline test
        symbols = scan_pub_symbols(rs)
        if not symbols:
            continue
        out[crate].append(
            {
                "file": rp,
                "symbols": [{"kind": k, "name": n, "line": ln} for (k, n, ln) in symbols],
            }
        )
    return out


# ---------- llvm-cov-driven mode ---------------------------------------------


def llvm_uncovered_files() -> dict[str, list[dict]]:
    """Return {crate: [{file, line_pct, uncovered_lines, total}]} sorted worst-first."""
    out: dict[str, list[dict]] = defaultdict(list)
    if not LLVM_JSON.exists():
        return out
    with LLVM_JSON.open() as f:
        llvm = json.load(f)
    data = llvm.get("data") or [llvm]
    for entry in data:
        for f in entry.get("files", []):
            file_path = f.get("filename") or ""
            crate = crate_for_path(Path(file_path))
            if not crate:
                continue
            summary = f.get("summary", {}).get("lines", {})
            count = summary.get("count", 0)
            covered = summary.get("covered", 0)
            if count == 0:
                continue
            pct = 100.0 * covered / count
            out[crate].append(
                {
                    "file": file_path,
                    "line_pct": pct,
                    "uncovered": count - covered,
                    "total": count,
                }
            )
    for crate in out:
        out[crate].sort(key=lambda r: (r["line_pct"], -r["uncovered"]))
    return out


# ---------- markdown rendering -----------------------------------------------


def render_untested_md(
    structural: dict[str, list[dict]],
    llvm: dict[str, list[dict]],
) -> str:
    lines: list[str] = []
    lines.append("# Untested Code Gap Report")
    lines.append("")
    if llvm:
        lines.append(
            "Sourced from `cargo-llvm-cov --json` "
            f"(`target/llvm-cov/coverage.json`). Each crate's worst-covered "
            "files are listed first."
        )
        lines.append("")
        for crate in sorted(llvm):
            files = llvm[crate]
            zero = [f for f in files if f["line_pct"] == 0.0]
            partial = [f for f in files if 0.0 < f["line_pct"] < 50.0]
            if not zero and not partial:
                continue
            lines.append(f"## {crate}")
            lines.append("")
            lines.append(f"- Files with 0% coverage: **{len(zero)}**")
            lines.append(
                f"- Files with <50% coverage: **{len(partial)}** "
                f"(top 20 listed below)"
            )
            lines.append("")
            lines.append("| File | line% | uncovered/total |")
            lines.append("|---|---:|---:|")
            for r in (zero + partial)[:20]:
                rel = r["file"].replace(str(REPO_ROOT) + "/", "")
                lines.append(
                    f"| `{rel}` | {r['line_pct']:.1f} | {r['uncovered']}/{r['total']} |"
                )
            lines.append("")
    else:
        lines.append(
            "**`target/llvm-cov/coverage.json` not found.** "
            "Falling back to a structural heuristic: every public symbol "
            "in a `.rs` file with no `#[test]` attribute is flagged as a "
            "candidate untested-code gap. Run `make coverage` to regenerate "
            "this report from real coverage measurements."
        )
        lines.append("")
        total_files = sum(len(v) for v in structural.values())
        total_syms = sum(
            sum(len(f["symbols"]) for f in v) for v in structural.values()
        )
        lines.append(
            f"**Totals:** {total_files} file(s), {total_syms} pub symbol(s) "
            f"with no inline `#[test]`."
        )
        lines.append("")
        for crate in sorted(structural):
            files = structural[crate]
            if not files:
                continue
            lines.append(f"## {crate}")
            lines.append("")
            lines.append(
                f"- Files with no inline tests: **{len(files)}**"
            )
            lines.append(
                f"- Public symbols in those files: "
                f"**{sum(len(f['symbols']) for f in files)}**"
            )
            lines.append("")
            lines.append("| File | pub symbols |")
            lines.append("|---|---:|")
            for r in sorted(files, key=lambda r: -len(r["symbols"]))[:30]:
                lines.append(
                    f"| `{r['file']}` | {len(r['symbols'])} |"
                )
            if len(files) > 30:
                lines.append(
                    f"\n_(truncated; {len(files) - 30} more files have no inline tests)_"
                )
            lines.append("")

    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--untested-code", action="store_true")
    p.add_argument("--unbacked", action="store_true")
    p.add_argument("--change-dir", default=None)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    if not args.untested_code and not args.unbacked:
        args.untested_code = True
        args.unbacked = True

    change_dir = find_change_dir(args.change_dir)

    if args.untested_code:
        llvm = llvm_uncovered_files()
        struct = structural_untested_code() if not llvm else {}
        md = render_untested_md(struct, llvm)
        out = change_dir / "gaps-untested-code.md"
        out.write_text(md)
        if not args.quiet:
            sys.stdout.write(f"wrote {out.relative_to(REPO_ROOT)}\n")

    if args.unbacked:
        # Delegate to spec-trace.py
        rc = subprocess.call(
            [sys.executable, str(REPO_ROOT / "scripts" / "spec-trace.py"), "--unbacked", "--quiet"]
        )
        if rc != 0:
            return rc

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
