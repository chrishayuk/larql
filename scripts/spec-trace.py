#!/usr/bin/env python3
"""
spec-trace.py — OpenSpec capability ↔ Rust test traceability tool.

Walks every `spec.md` under `openspec/changes/*/specs/` and `openspec/specs/`,
parses ### Requirement and #### Scenario blocks plus their inline
`<!-- test: ... -->` annotations, then walks `crates/*/{src,tests}/**/*.rs`
to discover every `#[test]` / `#[tokio::test]` / `#[rstest]` /
`#[serial_test::serial]` function, and cross-references the two.

Modes:

    spec-trace.py                # generate openspec/coverage/traceability.{md,json}
    spec-trace.py --check        # exit 1 if regenerated output differs from committed
    spec-trace.py --unbacked     # write gaps-unbacked-scenarios.md (active change dir)
    spec-trace.py --orphans      # list tests not referenced by any scenario
    spec-trace.py --json-only    # emit only traceability.json
    spec-trace.py --md-only      # emit only traceability.md

The annotation format inside a scenario is:

    <!-- test: <crate_with_underscores>::<module_path>::<test_fn> -->
    <!-- test: py:<file>::<pytest_fn> -->          # Python tests
    <!-- test: doc:<file>:<line> -->               # rustdoc doctests
    <!-- test: unbacked -->                        # explicitly no test yet

A scenario without ANY `<!-- test: ... -->` lines counts as `unbacked`.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
SPECS_ROOTS = [
    REPO_ROOT / "openspec" / "specs",
    REPO_ROOT / "openspec" / "changes",
]
CRATES_DIR = REPO_ROOT / "crates"
COVERAGE_DIR = REPO_ROOT / "openspec" / "coverage"
TRACE_MD = COVERAGE_DIR / "traceability.md"
TRACE_JSON = COVERAGE_DIR / "traceability.json"

REQ_RE = re.compile(r"^###\s+Requirement:\s*(.+?)\s*$")
SCN_RE = re.compile(r"^####\s+Scenario:\s*(.+?)\s*$")
TEST_RE = re.compile(r"<!--\s*test:\s*(.+?)\s*-->")
RUST_TEST_ATTR_RE = re.compile(
    r"#\[(?:tokio::test|test|rstest|serial_test::serial)(?:\([^)]*\))?\s*\]"
)
RUST_FN_RE = re.compile(r"\bfn\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(")


@dataclass
class Scenario:
    name: str
    tests: list[str] = field(default_factory=list)
    spec_file: str = ""
    line: int = 0

    @property
    def unbacked(self) -> bool:
        return not self.tests or self.tests == ["unbacked"]


@dataclass
class Requirement:
    name: str
    scenarios: list[Scenario] = field(default_factory=list)
    spec_file: str = ""


@dataclass
class Capability:
    name: str
    spec_file: str
    requirements: list[Requirement] = field(default_factory=list)


@dataclass
class TestFn:
    crate: str
    module: str  # logical Rust module path within the crate (e.g. `format::filenames::tests`)
    name: str
    file: str
    line: int

    @property
    def fqn(self) -> str:
        # Logical fully-qualified name as scenarios reference it.
        # Example: larql_models::loading::safetensors::is_ffn_tensor_gate_proj
        if self.module:
            return f"{self.crate}::{self.module}::{self.name}"
        return f"{self.crate}::{self.name}"


# ---------- spec parsing -------------------------------------------------------


def parse_spec_file(path: Path) -> Capability:
    rel = path.relative_to(REPO_ROOT).as_posix()
    cap_name = path.parent.name
    cap = Capability(name=cap_name, spec_file=rel)

    current_req: Requirement | None = None
    current_scn: Scenario | None = None

    text = path.read_text()
    for line_no, line in enumerate(text.splitlines(), 1):
        m = REQ_RE.match(line)
        if m:
            if current_scn and current_req:
                current_req.scenarios.append(current_scn)
            if current_req:
                cap.requirements.append(current_req)
            current_req = Requirement(name=m.group(1), spec_file=rel)
            current_scn = None
            continue

        m = SCN_RE.match(line)
        if m:
            if current_scn and current_req:
                current_req.scenarios.append(current_scn)
            current_scn = Scenario(name=m.group(1), spec_file=rel, line=line_no)
            continue

        m = TEST_RE.search(line)
        if m and current_scn is not None:
            current_scn.tests.append(m.group(1).strip())

    if current_scn and current_req:
        current_req.scenarios.append(current_scn)
    if current_req:
        cap.requirements.append(current_req)

    return cap


def discover_capabilities() -> list[Capability]:
    """Collect capabilities, MERGING requirements across every spec.md
    that names the same capability. This matters when multiple in-flight
    changes touch the same capability — last-wins masking would hide
    backfilled scenarios behind a delta. We accumulate instead, dropping
    duplicate requirement names so a change's MODIFIED rewrite still
    overrides the original."""
    seen: dict[str, Capability] = {}
    for root in SPECS_ROOTS:
        if not root.exists():
            continue
        for spec in sorted(root.rglob("spec.md")):
            if "/specs/" not in spec.as_posix():
                continue
            cap = parse_spec_file(spec)
            existing = seen.get(cap.name)
            if existing is None:
                seen[cap.name] = cap
                continue
            # Merge: append requirements that aren't already present by
            # name. spec_file stays as the canonical (first-seen) source.
            existing_req_names = {r.name for r in existing.requirements}
            for req in cap.requirements:
                if req.name not in existing_req_names:
                    existing.requirements.append(req)
                    existing_req_names.add(req.name)
    return sorted(seen.values(), key=lambda c: c.name)


# ---------- test discovery ----------------------------------------------------


def crate_name_from_dir(crate_dir: Path) -> str:
    cargo = crate_dir / "Cargo.toml"
    if not cargo.exists():
        return crate_dir.name.replace("-", "_")
    for line in cargo.read_text().splitlines():
        m = re.match(r"\s*name\s*=\s*\"([^\"]+)\"", line)
        if m:
            return m.group(1).replace("-", "_")
    return crate_dir.name.replace("-", "_")


def module_path_for_src(crate_dir: Path, src_file: Path) -> str:
    rel = src_file.relative_to(crate_dir / "src").with_suffix("")
    parts = list(rel.parts)
    if parts and parts[-1] == "mod":
        parts.pop()
    if parts and parts[0] in ("lib", "main"):
        parts.pop(0)
    return "::".join(parts)


def module_path_for_test(crate_dir: Path, test_file: Path) -> str:
    rel = test_file.relative_to(crate_dir / "tests").with_suffix("")
    parts = list(rel.parts)
    return "::".join(parts)


def discover_rust_tests() -> list[TestFn]:
    tests: list[TestFn] = []
    if not CRATES_DIR.exists():
        return tests

    for crate_dir in sorted(CRATES_DIR.iterdir()):
        if not crate_dir.is_dir():
            continue
        crate = crate_name_from_dir(crate_dir)

        # Inline tests under src/
        src = crate_dir / "src"
        if src.exists():
            for rs in src.rglob("*.rs"):
                in_tests_mod = "tests" in rs.parts or "test_helpers" in rs.parts
                module = module_path_for_src(crate_dir, rs)
                # Inline tests are conventionally under `mod tests { ... }`,
                # but cfg(test) blocks can use any inner module name. We
                # heuristically emit two candidate logical paths so scenario
                # annotations can use either form.
                discovered = scan_rust_file_for_tests(rs)
                for name, line_no, nested_mod in discovered:
                    # Build the full logical Rust module path:
                    #   file_module + nested_mod chain
                    file_mod = module
                    if nested_mod:
                        full_mod = (
                            f"{file_mod}::{nested_mod}" if file_mod else nested_mod
                        )
                    else:
                        full_mod = file_mod
                    tests.append(
                        TestFn(
                            crate=crate,
                            module=full_mod,
                            name=name,
                            file=rs.relative_to(REPO_ROOT).as_posix(),
                            line=line_no,
                        )
                    )
                    # Also emit a form with `tests` mod stripped, so a scenario
                    # may reference `<crate>::<file>::<name>` whether or not
                    # the test was nested under `mod tests`.
                    if full_mod and full_mod != file_mod:
                        # Strip a final `::tests` if present.
                        stripped = re.sub(r"::tests$", "", full_mod)
                        if stripped != full_mod:
                            tests.append(
                                TestFn(
                                    crate=crate,
                                    module=stripped,
                                    name=name,
                                    file=rs.relative_to(REPO_ROOT).as_posix(),
                                    line=line_no,
                                )
                            )

        # Integration tests under tests/
        tests_dir = crate_dir / "tests"
        if tests_dir.exists():
            for rs in tests_dir.rglob("*.rs"):
                module = module_path_for_test(crate_dir, rs)
                discovered = scan_rust_file_for_tests(rs)
                for name, line_no, nested_mod in discovered:
                    if nested_mod:
                        full_mod = f"{module}::{nested_mod}" if module else nested_mod
                    else:
                        full_mod = module
                    tests.append(
                        TestFn(
                            crate=crate,
                            module=full_mod,
                            name=name,
                            file=rs.relative_to(REPO_ROOT).as_posix(),
                            line=line_no,
                        )
                    )
                    # Strip optional ::tests suffix so scenarios using either
                    # form resolve. (Mirrors the inline-test behavior.)
                    if full_mod:
                        stripped = re.sub(r"::tests$", "", full_mod)
                        if stripped != full_mod:
                            tests.append(
                                TestFn(
                                    crate=crate,
                                    module=stripped,
                                    name=name,
                                    file=rs.relative_to(REPO_ROOT).as_posix(),
                                    line=line_no,
                                )
                            )

    return tests


MOD_DECL_RE = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?mod\s+([a-zA-Z_][a-zA-Z0-9_]*)\b")


def _tokenize_skip_strings_comments(text: str) -> str:
    """Return `text` with string/char literals and comments replaced by
    same-length whitespace, so brace counts on the result reflect only
    structural braces. Handles:
      - // line comments
      - /* block comments */ (nested)
      - "regular strings"
      - 'c' char literals (and single-quote lifetimes left alone)
      - r"raw strings"
      - r#"raw with hashes"#  (any number of #)
      - rb"...", b"...", br#"..."# byte / byte-raw strings
    """
    out = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        # Line comment
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            # Replace with spaces until newline
            while i < n and text[i] != "\n":
                out.append(" ")
                i += 1
            continue
        # Block comment (Rust supports nesting)
        if c == "/" and i + 1 < n and text[i + 1] == "*":
            depth = 1
            out.append(" ")
            out.append(" ")
            i += 2
            while i < n and depth > 0:
                if text[i] == "/" and i + 1 < n and text[i + 1] == "*":
                    depth += 1
                    out.append(" ")
                    out.append(" ")
                    i += 2
                elif text[i] == "*" and i + 1 < n and text[i + 1] == "/":
                    depth -= 1
                    out.append(" ")
                    out.append(" ")
                    i += 2
                else:
                    out.append(" " if text[i] != "\n" else "\n")
                    i += 1
            continue
        # Raw string: optional b prefix, then r, then >=0 #, then "
        # b"..." normal byte string is also possible.
        # Detect prefix.
        prefix_len = 0
        j = i
        if j < n and text[j] == "b":
            j += 1
            prefix_len += 1
        if j < n and text[j] == "r":
            j += 1
            prefix_len += 1
            hashes = 0
            while j < n and text[j] == "#":
                j += 1
                hashes += 1
            if j < n and text[j] == '"':
                # Found raw string opener — skip until closing "###...
                terminator = '"' + "#" * hashes
                start = i
                # Output spaces for prefix + opening hashes + opening quote
                out.append(" " * (j - i + 1))
                i = j + 1
                # Find terminator
                end = text.find(terminator, i)
                if end == -1:
                    # Unterminated — bail out, blank rest of file
                    while i < n:
                        out.append(" " if text[i] != "\n" else "\n")
                        i += 1
                    break
                while i < end:
                    out.append(" " if text[i] != "\n" else "\n")
                    i += 1
                # Skip terminator itself
                out.append(" " * len(terminator))
                i += len(terminator)
                continue
            # else: not a raw string; rewind to just after the optional 'b'.
            # We don't need special handling of b"..." separately; the
            # generic string handler below covers it.
        # Normal string
        if c == '"':
            out.append(" ")
            i += 1
            while i < n:
                if text[i] == "\\" and i + 1 < n:
                    # Escape pair: replace `\` with space, but preserve the
                    # following char's newline status so multi-line escape
                    # continuations don't collapse line numbers.
                    out.append(" ")
                    out.append(" " if text[i + 1] != "\n" else "\n")
                    i += 2
                    continue
                if text[i] == '"':
                    out.append(" ")
                    i += 1
                    break
                out.append(" " if text[i] != "\n" else "\n")
                i += 1
            continue
        # Char literal: 'x' or '\n' — but careful: 'a is a lifetime, no closing '.
        if c == "'":
            # Look ahead: if there's a closing ' within 5 chars (allowing escape),
            # treat as char literal; otherwise leave as-is (lifetime).
            k = i + 1
            if k < n and text[k] == "\\":
                k += 2
                if k < n and text[k] == "'":
                    out.append("   ")
                    i = k + 1
                    continue
            elif k + 1 < n and text[k + 1] == "'":
                out.append("   ")
                i = k + 2
                continue
            # Lifetime — leave alone
            out.append(c)
            i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def scan_rust_file_for_tests(path: Path) -> list[tuple[str, int, str]]:
    """Return [(test_fn, line_no, nested_mod_path)] for every #[test] in the file.

    Tracks `mod NAME { ... }` nesting using a brace-depth heuristic.
    Strings/chars/comments (including Rust raw strings `r#"..."#`) are
    blanked out before counting so structural braces are counted correctly.
    """
    out: list[tuple[str, int, str]] = []
    try:
        text = path.read_text(errors="replace")
    except Exception:
        return out

    cleaned_full = _tokenize_skip_strings_comments(text)

    raw_lines = text.splitlines()
    cleaned_lines = cleaned_full.splitlines()
    # Pad shorter list (rare edge case at EOF)
    while len(cleaned_lines) < len(raw_lines):
        cleaned_lines.append("")

    mod_stack: list[tuple[str, int]] = []
    depth = 0
    pending_test = False
    pending_test_line = 0

    for idx, raw_line in enumerate(raw_lines):
        line_no = idx + 1
        clean = cleaned_lines[idx]
        m_mod = MOD_DECL_RE.match(raw_line)

        if m_mod and "{" in clean:
            mod_stack.append((m_mod.group(1), depth))

        if RUST_TEST_ATTR_RE.search(raw_line):
            pending_test = True
            pending_test_line = line_no

        if pending_test:
            mfn = RUST_FN_RE.search(raw_line)
            if mfn and "fn " in raw_line:
                path_parts = [m for m, _ in mod_stack]
                mod_path = "::".join(path_parts)
                out.append((mfn.group(1), line_no, mod_path))
                pending_test = False
                pending_test_line = 0
            elif line_no - pending_test_line > 8:
                pending_test = False

        opens = clean.count("{")
        closes = clean.count("}")
        depth += opens - closes
        while mod_stack and mod_stack[-1][1] >= depth:
            mod_stack.pop()

    return out


def discover_python_tests() -> list[TestFn]:
    """Discover pytest function names in crates/larql-python/tests/*.py."""
    out: list[TestFn] = []
    py_tests_dir = CRATES_DIR / "larql-python" / "tests"
    if not py_tests_dir.exists():
        return out
    py_test_re = re.compile(r"^def\s+(test_[a-zA-Z0-9_]+)\s*\(")
    py_class_re = re.compile(r"^class\s+(Test[A-Za-z0-9_]+)\s*[:(]")
    for py in sorted(py_tests_dir.rglob("*.py")):
        rel = py.relative_to(REPO_ROOT).as_posix()
        in_class: str | None = None
        for line_no, line in enumerate(py.read_text().splitlines(), 1):
            mc = py_class_re.match(line.lstrip())
            if mc and not line.startswith(" "):
                in_class = mc.group(1)
                continue
            mf = py_test_re.match(line.lstrip())
            if mf:
                fn = mf.group(1)
                if in_class and line.startswith("    "):
                    fqn = f"py:{rel.split('crates/larql-python/')[-1]}::{in_class}::{fn}"
                else:
                    fqn = f"py:{rel.split('crates/larql-python/')[-1]}::{fn}"
                out.append(
                    TestFn(
                        crate="larql_python",
                        module=in_class or "",
                        name=fqn,
                        file=rel,
                        line=line_no,
                    )
                )
    return out


# ---------- traceability matrix ----------------------------------------------


def build_index(tests: list[TestFn]) -> tuple[dict[str, list[TestFn]], dict[tuple[str, str], list[TestFn]]]:
    """Map reference forms → TestFn(s). Returns:
       - exact:    fqn → [TestFn]
       - by_name:  (crate, fn_name) → [TestFn]
    """
    exact: dict[str, list[TestFn]] = defaultdict(list)
    by_name: dict[tuple[str, str], list[TestFn]] = defaultdict(list)
    for t in tests:
        if t.name.startswith("py:"):
            exact[t.name].append(t)
            by_name[("larql_python", t.name)].append(t)
        else:
            exact[t.fqn].append(t)
            by_name[(t.crate, t.name)].append(t)
    return exact, by_name


def resolve_test(
    ref: str,
    exact: dict[str, list[TestFn]],
    by_name: dict[tuple[str, str], list[TestFn]],
    all_tests: list[TestFn] | None = None,
) -> list[TestFn]:
    if ref == "unbacked":
        return []
    if ref.startswith("doc:"):
        return []  # accepted, but not validated against test discovery
    if ref.startswith("py:"):
        # Allow wildcard at the end: py:tests/test_bindings.py::*
        if ref.endswith("::*"):
            prefix = ref[:-3]
            return [
                t
                for t in (all_tests or [])
                if t.name.startswith(prefix + "::")
            ]
        if "::**::*" in ref:
            head = ref.split("::**::*")[0]
            return [
                t
                for t in (all_tests or [])
                if t.name.startswith(head + "::")
            ]
        return exact.get(ref, [])

    # Wildcard match: <prefix>::*  → every TestFn whose fqn lives directly
    # under <prefix>::, with no further nesting (single-level expansion).
    # Deep wildcard: <prefix>::**::*  → every TestFn whose fqn starts with <prefix>::
    if all_tests is not None:
        if ref.endswith("::**::*"):
            head = ref[: -len("::**::*")]
            return [t for t in all_tests if t.fqn.startswith(head + "::")]
        if ref.endswith("::*"):
            head = ref[: -len("::*")]
            prefix = head + "::"
            matches = []
            for t in all_tests:
                if not t.fqn.startswith(prefix):
                    continue
                # Single-level: there must be no further `::` after head + "::"
                # except the test fn name itself. So the part after prefix
                # must be exactly the test name (no `::` in it).
                tail = t.fqn[len(prefix):]
                if "::" not in tail:
                    matches.append(t)
            return matches

    if ref in exact:
        return exact[ref]
    # Fuzzy fall-back: same crate + fn name, ignoring intermediate module path.
    parts = ref.split("::")
    if len(parts) >= 2:
        crate = parts[0]
        fn = parts[-1]
        candidates = by_name.get((crate, fn), [])
        if len(candidates) == 1:
            return candidates  # unique name within crate → accept
        # If multiple, we still accept but flag — the trace JSON records the
        # exact match too. We accept matches whose module path prefix-overlaps.
        ref_path = parts[1:-1]
        good: list[TestFn] = []
        for c in candidates:
            cand_path = c.module.split("::") if c.module else []
            # Keep candidate if cand_path ends with ref_path's tail (best-effort).
            if not ref_path or any(
                cand_path[-len(ref_path):] == ref_path
                for _ in (1,)
            ):
                good.append(c)
        if good:
            return good
        # Otherwise return all candidates rather than nothing — the spec
        # lookup is best-effort during fuzzy fall-back.
        return candidates
    return []


def trace(capabilities: list[Capability], tests: list[TestFn]) -> dict:
    exact, by_name = build_index(tests)
    referenced: set[str] = set()
    out_caps = []
    summary = {
        "capabilities": 0,
        "requirements": 0,
        "scenarios": 0,
        "scenarios_backed": 0,
        "scenarios_unbacked": 0,
        "test_refs": 0,
        "test_refs_unresolved": 0,
    }

    for cap in capabilities:
        out_reqs = []
        for req in cap.requirements:
            out_scns = []
            for scn in req.scenarios:
                resolved: list[str] = []
                unresolved: list[str] = []
                explicitly_unbacked = scn.tests == ["unbacked"]
                for ref in scn.tests:
                    summary["test_refs"] += 1
                    if ref == "unbacked":
                        continue
                    if ref.startswith("doc:"):
                        resolved.append(ref)
                        continue
                    matches = resolve_test(ref, exact, by_name, tests)
                    if matches:
                        resolved.append(ref)
                        referenced.add(ref)
                        # Also mark the matched tests' FQN(s) as referenced
                        # so they don't show up as orphans.
                        for m_t in matches:
                            referenced.add(
                                m_t.name if m_t.name.startswith("py:") else m_t.fqn
                            )
                    else:
                        unresolved.append(ref)
                        summary["test_refs_unresolved"] += 1
                unbacked = scn.unbacked or (
                    not resolved and not explicitly_unbacked and not scn.tests
                )
                summary["scenarios"] += 1
                if unbacked or explicitly_unbacked:
                    summary["scenarios_unbacked"] += 1
                else:
                    summary["scenarios_backed"] += 1
                out_scns.append(
                    {
                        "name": scn.name,
                        "tests_resolved": resolved,
                        "tests_unresolved": unresolved,
                        "unbacked": unbacked or explicitly_unbacked,
                        "spec_file": scn.spec_file,
                        "line": scn.line,
                    }
                )
            out_reqs.append({"name": req.name, "scenarios": out_scns})
            summary["requirements"] += 1
        out_caps.append(
            {
                "name": cap.name,
                "spec_file": cap.spec_file,
                "requirements": out_reqs,
            }
        )
        summary["capabilities"] += 1

    # orphan tests = tests in the workspace not referenced by any scenario
    referenced_keys = set(referenced)
    orphan_tests = []
    for t in tests:
        key = t.name if t.name.startswith("py:") else t.fqn
        if key not in referenced_keys:
            orphan_tests.append(
                {
                    "fqn": key,
                    "file": t.file,
                    "line": t.line,
                    "crate": t.crate,
                }
            )

    return {
        "summary": summary,
        "capabilities": out_caps,
        "orphan_tests": orphan_tests,
    }


def render_md(matrix: dict) -> str:
    s = matrix["summary"]
    lines: list[str] = []
    lines.append("# OpenSpec Traceability Matrix")
    lines.append("")
    lines.append("Generated by `scripts/spec-trace.py`. Do not edit by hand.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Capabilities: **{s['capabilities']}**")
    lines.append(f"- Requirements: **{s['requirements']}**")
    lines.append(
        f"- Scenarios: **{s['scenarios']}** "
        f"(backed: {s['scenarios_backed']}, "
        f"unbacked: {s['scenarios_unbacked']})"
    )
    lines.append(
        f"- Test references: **{s['test_refs']}** "
        f"(unresolved: {s['test_refs_unresolved']})"
    )
    if s["scenarios"]:
        coverage_pct = 100.0 * s["scenarios_backed"] / s["scenarios"]
        lines.append(f"- Spec coverage: **{coverage_pct:.1f}%**")
    lines.append("")

    for cap in matrix["capabilities"]:
        lines.append(f"## {cap['name']}")
        lines.append("")
        lines.append(f"_Source: `{cap['spec_file']}`_")
        lines.append("")
        lines.append("| Requirement | Scenario | Backed by |")
        lines.append("|---|---|---|")
        for req in cap["requirements"]:
            for scn in req["scenarios"]:
                if scn["unbacked"]:
                    backed = "_unbacked_"
                else:
                    parts = []
                    for t in scn["tests_resolved"]:
                        parts.append(f"`{t}`")
                    if scn["tests_unresolved"]:
                        parts.append(
                            "**unresolved:** "
                            + ", ".join(f"`{t}`" for t in scn["tests_unresolved"])
                        )
                    backed = "<br>".join(parts) or "_—_"
                lines.append(
                    f"| {req['name']} | {scn['name']} | {backed} |"
                )
        lines.append("")

    if matrix["orphan_tests"]:
        lines.append("## Orphan tests (not referenced by any scenario)")
        lines.append("")
        lines.append(f"Total: **{len(matrix['orphan_tests'])}**")
        lines.append("")
        # Limit listing to keep the file scannable.
        sample = matrix["orphan_tests"][:200]
        lines.append("| FQN | File:Line |")
        lines.append("|---|---|")
        for o in sample:
            lines.append(f"| `{o['fqn']}` | `{o['file']}:{o['line']}` |")
        if len(matrix["orphan_tests"]) > 200:
            lines.append(
                f"\n_(truncated; {len(matrix['orphan_tests']) - 200} more in `traceability.json`)_"
            )

    return "\n".join(lines) + "\n"


def render_unbacked_md(matrix: dict) -> str:
    """Generate the gaps-unbacked-scenarios.md report."""
    lines: list[str] = []
    lines.append("# Unbacked Scenarios")
    lines.append("")
    lines.append(
        "Scenarios with no resolved `<!-- test: ... -->` annotation. "
        "Either explicitly marked `unbacked`, missing the annotation, or "
        "referencing a test that the discovery tool can't find."
    )
    lines.append("")

    total = 0
    for cap in matrix["capabilities"]:
        cap_lines: list[str] = []
        for req in cap["requirements"]:
            for scn in req["scenarios"]:
                if scn["unbacked"] or scn["tests_unresolved"]:
                    total += 1
                    cap_lines.append(
                        f"- **{req['name']} / {scn['name']}** "
                        f"(`{cap['spec_file']}:{scn['line']}`)"
                    )
                    if scn["tests_unresolved"]:
                        cap_lines.append(
                            f"  - Unresolved test annotations: "
                            + ", ".join(f"`{t}`" for t in scn["tests_unresolved"])
                        )
        if cap_lines:
            lines.append(f"## {cap['name']}")
            lines.append("")
            lines.extend(cap_lines)
            lines.append("")

    if total == 0:
        lines.append("**No unbacked scenarios.** All scenarios resolve to a real test.")
        lines.append("")
    else:
        lines.insert(2, f"**Total: {total} scenario(s).**")
        lines.insert(3, "")

    return "\n".join(lines) + "\n"


# ---------- main --------------------------------------------------------------


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true", help="exit 1 if regenerated output differs from committed")
    p.add_argument("--unbacked", action="store_true", help="also write gaps-unbacked-scenarios.md")
    p.add_argument("--orphans", action="store_true", help="print orphan tests to stdout")
    p.add_argument("--json-only", action="store_true", help="emit only traceability.json")
    p.add_argument("--md-only", action="store_true", help="emit only traceability.md")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    capabilities = discover_capabilities()
    tests = discover_rust_tests() + discover_python_tests()
    matrix = trace(capabilities, tests)

    COVERAGE_DIR.mkdir(parents=True, exist_ok=True)

    md = render_md(matrix)
    js = json.dumps(matrix, indent=2, sort_keys=False) + "\n"

    if args.check:
        # Compare against committed files; do not write.
        diff = []
        if not args.json_only:
            committed_md = TRACE_MD.read_text() if TRACE_MD.exists() else ""
            if committed_md != md:
                diff.append(f"{TRACE_MD.relative_to(REPO_ROOT)}")
        if not args.md_only:
            committed_js = TRACE_JSON.read_text() if TRACE_JSON.exists() else ""
            if committed_js != js:
                diff.append(f"{TRACE_JSON.relative_to(REPO_ROOT)}")
        if diff:
            sys.stderr.write(
                "spec-trace --check: committed traceability files diverge from regenerated output:\n"
            )
            for d in diff:
                sys.stderr.write(f"  - {d}\n")
            sys.stderr.write("Run `make traceability` and commit the result.\n")
            return 1
        if not args.quiet:
            sys.stdout.write("spec-trace --check: OK\n")
        return 0

    if not args.json_only:
        TRACE_MD.write_text(md)
    if not args.md_only:
        TRACE_JSON.write_text(js)

    if args.unbacked:
        # Find the active change dir (or default to the first under changes/)
        change_dirs = sorted(
            d for d in (REPO_ROOT / "openspec" / "changes").iterdir() if d.is_dir()
        )
        # Prefer 'backfill-specs' if it exists (the bootstrap change), otherwise
        # write into the first change dir.
        active = None
        for d in change_dirs:
            if d.name == "backfill-specs":
                active = d
                break
        if active is None and change_dirs:
            active = change_dirs[0]
        if active is not None:
            (active / "gaps-unbacked-scenarios.md").write_text(
                render_unbacked_md(matrix)
            )

    if args.orphans:
        for o in matrix["orphan_tests"]:
            sys.stdout.write(f"{o['fqn']}\t{o['file']}:{o['line']}\n")

    if not args.quiet:
        s = matrix["summary"]
        sys.stdout.write(
            f"spec-trace: {s['capabilities']} capabilities, "
            f"{s['requirements']} requirements, "
            f"{s['scenarios']} scenarios "
            f"(backed: {s['scenarios_backed']}, unbacked: {s['scenarios_unbacked']}, "
            f"unresolved refs: {s['test_refs_unresolved']}); "
            f"{len(matrix['orphan_tests'])} orphan tests.\n"
        )
        sys.stdout.write(f"Wrote {TRACE_MD.relative_to(REPO_ROOT)} and {TRACE_JSON.relative_to(REPO_ROOT)}.\n")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
