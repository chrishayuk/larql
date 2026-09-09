#!/usr/bin/env python3
"""Check the documentation references that `check_doc_links.py` cannot see.

That script validates markdown links, and reports all of them resolving —
truthfully. But a 2026-08-22 audit of 311 documents found ~90 broken
references, and **not one was a markdown link**. They were inline-code
paths, `file.rs:LINE` coordinates, `--example` names, `make` targets and
env-var names. The link gate is structurally blind to every one, which is
why that drift accumulated silently for months.

Rules, each chosen because the audit found real breakage of that shape:

  coord     `path.rs:120`      file must exist, and have >= 120 lines
  path      `crates/foo/bar.rs` inline paths under known roots must exist
  example   `--example NAME`   must be a real example, and any `-p CRATE`
                               beside it must be the crate that owns it
  bench     `--bench NAME`     must be a real bench target
  make      `make target`      must be a rule in the Makefile
  env       `LARQL_FOO`        must appear as a literal in crates/ or scripts/

Two design decisions keep the output worth reading rather than ignored:

  * **Resolve basenames before judging.** Docs legitimately write
    `hidden.rs:38` rather than the full path. Matching naively flags ~60%
    of coordinates as broken — pure noise. This resolves by unique
    basename or path-suffix and *skips* ambiguous ones, which is the
    difference between a gate people run and one they disable.
  * **Only match `make x` in backticks or at line start.** Unanchored, it
    fires on English ("make the", "make it", "make sure").

Limits, stated so nobody mistakes a green run for a correct corpus: this
catches the BROKEN class only. It cannot see the STALE class — "not
started" on shipped work, superseded numbers, wrong counts — which the
audit found to be both larger (~45 vs ~15) and more damaging, because a
broken path fails loudly while a stale status marker silently reroutes a
week of work.

Usage:
    python3 scripts/check_doc_references.py            # whole repo, report
    python3 scripts/check_doc_references.py --strict   # exit 1 on findings
    python3 scripts/check_doc_references.py docs/ AGENTS.md
"""

from __future__ import annotations

import collections
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_DIRS = ("/target/", "/.git/", "/.claude/", "/.venv/", "/node_modules/", "/coverage/")
# Historical records describe what was true when written; a renamed symbol
# or moved example in a CHANGELOG is not a defect to fix.
HISTORICAL = re.compile(r"CHANGELOG|/adr/|/audits/|/replay/|/baselines/")
SOURCE_EXT = (".rs", ".py", ".md", ".toml", ".json", ".sh", ".yml", ".msl")
# References into upstream projects and sibling repos are not ours to resolve.
# Without this the checker reports ~10 findings nobody can act on, which is
# how a gate earns its way onto the ignore list.
EXTERNAL = re.compile(
    r"^(transformers/|modeling_|streaming_|inferencer\.py|torch/|mlx/|llama\.cpp|chris-experiments/|"
    r"larql_probes/|~/|/Users/)"
)
PATH_ROOTS = ("crates/", "docs/", "bench/", "scripts/", "knowledge/", ".github/")
# A reference the prose has already LABELLED is not a trap, and this gate
# exists to catch traps. Two shapes qualify: a roadmap naming the file it
# intends to create, and a doc that says outright the thing is missing.
# Both leave the reader correctly informed, which is the whole point —
# whereas an unannotated dangling path sends someone looking for code that
# is not there.
#
# Without this the gate flags legitimate content as breakage, and a gate
# that cries wolf is one people switch off. Deliberately narrow: it wants
# an explicit marker on the same line, not a vague future tense.
PLANNED = re.compile(
    r"\(new\)|new `|\bgenerate\b|\bplanned\b|\bproposed\b|\bTODO\b|"
    r"\bnot started\b|will land|lands the|to be (?:written|created|added)|"
    r"does not exist|not yet|never committed|was never|no longer exists",
    re.I,
)


def skipped(dirpath: str, root: str) -> bool:
    """Is this directory one of SKIP_DIRS, RELATIVE to the repository root?

    Relative is the whole point. Matching SKIP_DIRS against the absolute path
    means a checkout that merely LIVES under one of these names matches on its
    own prefix, every directory beneath it matches too, and the walk yields
    nothing -- while the caller reports success over an empty corpus. Every
    worktree in this repository lives under `.claude/worktrees/`, so that was
    the state of every local run until DOC-GATE-1.
    """
    rel = os.path.relpath(dirpath, root)
    probe = "/" if rel == "." else "/" + rel.replace(os.sep, "/") + "/"
    return any(s in probe for s in SKIP_DIRS)


def index_files(root: str = ROOT) -> tuple[dict, dict]:
    by_rel: dict[str, str] = {}
    by_base: dict[str, list[str]] = collections.defaultdict(list)
    for dirpath, dirnames, filenames in os.walk(root):
        if skipped(dirpath, root):
            dirnames[:] = []
            continue
        for name in filenames:
            rel = os.path.relpath(os.path.join(dirpath, name), root)
            by_rel[rel] = os.path.join(dirpath, name)
            by_base[name].append(rel)
    return by_rel, by_base


def resolve(ref: str, by_rel: dict, by_base: dict) -> list[str]:
    """Candidates for a possibly-abbreviated path. Empty = not found."""
    if ref in by_rel:
        return [ref]
    hits = [r for r in by_rel if r.endswith("/" + ref)]
    if hits:
        return hits
    return by_base.get(os.path.basename(ref), []) if "/" not in ref else []


def line_count(rel: str) -> int:
    try:
        with open(os.path.join(ROOT, rel), "rb") as fh:
            return sum(1 for _ in fh)
    except OSError:
        return 0


def cargo_targets(kind: str) -> dict[str, set[str]]:
    """Map target name -> owning crates, from Cargo.toml plus conventions."""
    owners: dict[str, set[str]] = collections.defaultdict(set)
    for dirpath, dirnames, filenames in os.walk(os.path.join(ROOT, "crates")):
        if skipped(dirpath, ROOT):
            dirnames[:] = []
            continue
        if "Cargo.toml" not in filenames:
            continue
        crate = os.path.basename(dirpath)
        text = open(os.path.join(dirpath, "Cargo.toml")).read()
        for m in re.finditer(rf'\[\[{kind}\]\]\s*\nname\s*=\s*"([^"]+)"', text):
            owners[m.group(1)].add(crate)
        conv = os.path.join(dirpath, kind + "s")
        if os.path.isdir(conv):
            for sub, _d, fs in os.walk(conv):
                for f in fs:
                    if f.endswith(".rs"):
                        owners[f[:-3]].add(crate)
    return owners


def _fixture(root: Path, *, with_docs: bool = True, with_broken: bool = False) -> None:
    """A miniature repository: one real reference, and traps in skipped dirs."""
    (root / "docs").mkdir(parents=True, exist_ok=True)
    src = root / "crates" / "larql-x" / "src"
    src.mkdir(parents=True, exist_ok=True)
    (src / "lib.rs").write_text("pub fn f() {}\n")
    # Each of these carries a reference that WOULD be reported broken if the
    # directory were examined -- so control B needs no bookkeeping: if skipping
    # regresses, the run fails.
    for name in ("target", ".git", "node_modules", ".venv", "coverage", ".claude"):
        d = root / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "trap.md").write_text("cites `crates/larql-x/src/nowhere.rs`\n")
    if with_docs:
        (root / "docs" / "real.md").write_text("cites `crates/larql-x/src/lib.rs`\n")
    if with_broken:
        (root / "docs" / "bad.md").write_text("cites `crates/larql-x/src/gone.rs`\n")


def _run_in(root: Path) -> tuple[int, str]:
    """Run THIS script, as its own entry point, against a fixture root."""
    scripts = root / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    shutil.copy(os.path.abspath(__file__), scripts / "check_doc_references.py")
    done = subprocess.run(
        [sys.executable, str(scripts / "check_doc_references.py"), "--strict"],
        capture_output=True, text=True, check=False,
    )
    return done.returncode, done.stdout + done.stderr


def _counts(output: str) -> tuple[int, int] | None:
    found = re.search(r"checked (\d+) references across (\d+) documents", output)
    return (int(found.group(1)), int(found.group(2))) if found else None


def selftest() -> int:
    """The DOC-GATE-1 controls. Each proves a verdict is REACHABLE."""
    results: list[tuple[bool, str]] = []
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)

        # 1 + 3. The same fixture at an ordinary path and beneath
        # `.claude/worktrees/`. Identical corpus, or the skip semantics are
        # still absolute. Non-zero, or both are vacuous and "identical" is
        # satisfied by 0 == 0.
        ordinary = base / "ordinary"
        worktree = base / ".claude" / "worktrees" / "wt"
        _fixture(ordinary)
        _fixture(worktree)
        rc_o, out_o = _run_in(ordinary)
        rc_w, out_w = _run_in(worktree)
        got_o, got_w = _counts(out_o), _counts(out_w)
        results.append((rc_o == 0, f"ordinary checkout passes (rc={rc_o})"))
        results.append((rc_w == 0, f"`.claude` worktree passes (rc={rc_w})"))
        results.append(
            (got_o is not None and got_o == got_w,
             f"same corpus from both paths: ordinary={got_o} worktree={got_w}")
        )
        results.append(
            (got_w is not None and got_w[1] > 0,
             f"the worktree run examined documents, not zero: {got_w}")
        )

        # 2. A corpus with no documents is a failure, not a green.
        empty = base / "empty"
        _fixture(empty, with_docs=False)
        rc_e, _ = _run_in(empty)
        results.append((rc_e != 0, f"zero documents examined FAILS (rc={rc_e})"))

        # 5. The traps in skipped directories were not examined -- if they
        # had been, `nowhere.rs` would have been reported and rc would be 1.
        results.append(
            (rc_w == 0, "skipped directories stayed skipped from a worktree")
        )

        # And detection still works, so none of the above passes by being blind.
        broken = base / "broken"
        _fixture(broken, with_broken=True)
        rc_b, out_b = _run_in(broken)
        results.append(
            (rc_b != 0 and "gone.rs" in out_b,
             f"a genuinely broken reference still FAILS (rc={rc_b})")
        )

    for ok, label in results:
        print(f"  {'ok  ' if ok else 'FAIL'}  {label}")
    bad = [label for ok, label in results if not ok]
    if bad:
        print(f"\n{len(bad)} of {len(results)} DOC-GATE-1 controls failed", file=sys.stderr)
        return 1
    print(f"\nall {len(results)} DOC-GATE-1 controls hold")
    return 0


def main() -> int:
    if "selftest" in sys.argv[1:]:
        return selftest()
    strict = "--strict" in sys.argv
    scope = [a for a in sys.argv[1:] if not a.startswith("-")]

    by_rel, by_base = index_files()
    examples = cargo_targets("example")
    benches = cargo_targets("bench")
    try:
        make_rules = set(
            re.findall(r"^([A-Za-z0-9_.-]+):", open(os.path.join(ROOT, "Makefile")).read(), re.M)
        )
    except OSError:
        make_rules = set()
    try:
        env_blob = subprocess.run(
            ["grep", "-rho", "LARQL_[A-Z0-9_]*", "crates", "scripts"],
            cwd=ROOT, capture_output=True, text=True, check=False,
        ).stdout
        known_env = set(env_blob.split())
    except OSError:
        known_env = set()

    docs = []
    for rel in sorted(by_rel):
        if not rel.endswith(".md") or HISTORICAL.search(rel):
            continue
        if scope and not any(rel == s or rel.startswith(s.rstrip("/") + "/") for s in scope):
            continue
        docs.append(rel)

    # DOC-GATE-1 condition 2. A gate that examined nothing has not passed;
    # it has failed to run. Reporting "no broken references" over an empty
    # corpus is the exact shape of the bug this guard exists to make
    # impossible to reintroduce, whatever future path or filter causes it.
    if not docs:
        where = f" matching {scope}" if scope else ""
        print(
            f"examined 0 documents{where} -- the corpus is empty, so this is "
            "not a pass. Either the scope matches nothing, or the walk is "
            "being pruned before it starts.",
            file=sys.stderr,
        )
        return 2

    findings: list[tuple[str, int, str, str]] = []
    counts: collections.Counter = collections.Counter()

    for rel in docs:
        try:
            lines = open(os.path.join(ROOT, rel), encoding="utf-8").read().splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        for num, line in enumerate(lines, 1):
            # 1. `path.rs:LINE`
            for m in re.finditer(
                r"`([A-Za-z0-9_./-]+\.(?:rs|py|md|toml|json|sh|yml)):(\d+)(?:[-–]\d+)?`", line
            ):
                counts["coord"] += 1
                ref, want = m.group(1), int(m.group(2))
                if EXTERNAL.search(ref):
                    counts["coord"] -= 1
                    continue
                cand = resolve(ref, by_rel, by_base)
                if not cand:
                    findings.append((rel, num, "coord: no such file", m.group(0)))
                elif len(cand) == 1 and line_count(cand[0]) < want:
                    findings.append(
                        (rel, num, f"coord: {cand[0]} has {line_count(cand[0])} lines", m.group(0))
                    )
            # 2. inline paths under known roots
            for m in re.finditer(r"`((?:" + "|".join(PATH_ROOTS) + r")[A-Za-z0-9_./-]+)`", line):
                ref = m.group(1).rstrip("/")
                counts["path"] += 1
                if ref in by_rel or os.path.isdir(os.path.join(ROOT, ref)):
                    continue
                if not ref.endswith(SOURCE_EXT):
                    continue  # prose-ish path, or a directory that moved
                if not resolve(ref, by_rel, by_base) and not PLANNED.search(line):
                    findings.append((rel, num, "path: does not exist", m.group(0)))
            # 3. --example NAME, with its -p CRATE if present
            m = re.search(r"--example\s+([\w-]+)", line)
            if m:
                counts["example"] += 1
                name = m.group(1)
                owners = examples.get(name)
                pm = re.search(r"-p\s+([\w-]+)", line)
                if not owners:
                    if pm:  # bare `--example` may be an external repo's
                        findings.append((rel, num, "example: does not exist", name))
                elif pm and pm.group(1) not in owners:
                    findings.append(
                        (rel, num, f"example: lives in {sorted(owners)}", f"-p {pm.group(1)}")
                    )
            # 4. --bench NAME
            m = re.search(r"--bench\s+([\w-]+)", line)
            if m:
                counts["bench"] += 1
                name = m.group(1)
                owners = benches.get(name)
                pm = re.search(r"-p\s+([\w-]+)", line)
                if not owners:
                    findings.append((rel, num, "bench: does not exist", name))
                elif pm and pm.group(1) not in owners:
                    findings.append(
                        (rel, num, f"bench: lives in {sorted(owners)}", f"-p {pm.group(1)}")
                    )
            # 5. make targets — anchored, or prose fires constantly
            for m in re.finditer(r"(?:^|`)make\s+([a-z][a-z0-9-]{2,})", line):
                counts["make"] += 1
                # `make larql-<crate>-ci` is a documented PATTERN, not a
                # target; the regex stops at the `<` and would report the
                # truncated stem. A placeholder is correct content.
                if line[m.end():m.end() + 1] == "<":
                    counts["make"] -= 1
                    continue
                if make_rules and m.group(1) not in make_rules:
                    findings.append((rel, num, "make: no such target", m.group(1)))
            # 6. env vars
            for m in re.finditer(r"`(LARQL_[A-Z0-9_]+)`", line):
                counts["env"] += 1
                if known_env and m.group(1) not in known_env:
                    findings.append((rel, num, "env: not read anywhere", m.group(1)))

    checked = sum(counts.values())
    if checked == 0 and not scope:
        print(
            f"examined {len(docs)} documents and extracted 0 references -- a "
            "whole-repository run cannot legitimately find none, so the "
            "extractors are not matching. Not a pass.",
            file=sys.stderr,
        )
        return 2
    print(f"checked {checked} references across {len(docs)} documents")
    for rule, n in sorted(counts.items()):
        print(f"  {rule:<8} {n}")
    if not findings:
        print("\nno broken references")
        return 0
    print(f"\n{len(findings)} finding(s):\n")
    for rel, num, why, what in findings:
        print(f"  {rel}:{num}  {why}  —  {what}")
    print(
        "\nNote: this checks the BROKEN class only. Stale status markers and "
        "superseded numbers are invisible to it and are the larger problem."
    )
    return 1 if strict else 0


if __name__ == "__main__":
    sys.exit(main())
