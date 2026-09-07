#!/usr/bin/env python3
"""Whether a cached conformance plan may be read as CURRENT evidence.

A saved plan is a VERDICT, and a verdict is only comparable with another
verdict from a planner that judges the same way.

    scripts/conformance_evidence.py selftest
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# ── Evidence integrity: cached plans belong to a planner ───────────────
#
# A saved plan is a VERDICT, and a verdict is only comparable with another
# verdict from a planner that judges the same way. `PLANNER_SEMANTICS_VERSION`
# is what says so — the Rust constant that changes whenever a rule change
# makes a checkpoint admissible or blocked where it was not before.
#
# The hazard this guards is specific and was found in the wild: a cache at
# semantics 16 sitting beside a semantics-22 planner. Every row parsed,
# every field the readers want was present, and NOTHING in this script
# looked at the stamp — so four verdict-versions of drift were
# indistinguishable from current evidence. That is worse than a corrupt
# cache, which announces itself.
#
# The rule has one job and deliberately does not grow past it: prevent
# verdict-version drift from masquerading as comparable evidence. It is
# NOT a cache-schema or format-invalidation framework; if those are ever
# needed they get their own authority rather than overloading planner
# semantics.
#
# Scope: the COMPARATIVE readers only (`report`, `envelopes`, `clusters`,
# `leverage`) — the commands that turn cached rows into claims. `run`
# writes rows and must stay able to overwrite a stale directory, and
# nothing here deletes anything: a stale cache is still historical
# evidence, it just is not CURRENT evidence.
PLANNER_SEMANTICS_SOURCE = (
    REPO_ROOT / "crates" / "larql-vindex" / "src" / "format" / "vindex3" / "plan" / "report.rs"
)
PLANNER_SEMANTICS_CONST = "PLANNER_SEMANTICS_VERSION"
# Where each saved row records the planner that judged it.
SEMANTICS_PATH = ("plan", "planner", "semantics_version")


class StaleEvidence(Exception):
    """A cache that cannot be consumed as CURRENT comparative evidence."""


def current_semantics_version() -> int:
    """The planner's own `PLANNER_SEMANTICS_VERSION`, from the Rust source.

    The Rust constant is the authority and this reads it directly. The
    two alternatives are both wrong: `vindex` exposes no flag for it, and
    taking the value from a freshly planned row would be self-consistent
    in exactly the way that defeats the check — a stale cache compared
    against a stale row agrees with itself.
    """
    text = PLANNER_SEMANTICS_SOURCE.read_text()
    match = re.search(
        rf"pub const {PLANNER_SEMANTICS_CONST}\s*:\s*u32\s*=\s*(\d+)\s*;", text
    )
    if not match:
        raise StaleEvidence(
            f"cannot read {PLANNER_SEMANTICS_CONST} from {PLANNER_SEMANTICS_SOURCE} — "
            "the comparative readers have no authority to check cached evidence against, "
            "and refusing is the only honest answer"
        )
    return int(match.group(1))


def recorded_semantics_version(record: dict):
    """The semantics version a saved row records, or `None`.

    `None` covers two different states and the caller distinguishes them:
    a row that carries no plan at all (unreachable or errored — it asserts
    nothing about semantics and cannot drift), and a plan that records no
    version (written before the stamp existed, so its verdicts are of an
    unknown vintage).
    """
    node = record
    for key in SEMANTICS_PATH:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node if isinstance(node, int) else None


def require_current_evidence(records: list[tuple[pathlib.Path, dict]], out_dir: pathlib.Path):
    """Refuse a cache whose verdicts were not judged by this planner.

    Names the cached version, the current one and the path, because the
    fix is always the same and the reader should not have to work out
    which of three caches on the machine was consulted.
    """
    current = current_semantics_version()
    seen: dict[object, list[str]] = {}
    for path, record in records:
        # A row with no plan carries no verdict — it cannot be stale
        # evidence, and refusing on it would make an unreachable repo
        # look like a versioning problem.
        if not record.get("ok") or "plan" not in record:
            continue
        seen.setdefault(recorded_semantics_version(record), []).append(path.name)
    stale = {version: rows for version, rows in seen.items() if version != current}
    if not stale:
        return current

    def describe(version) -> str:
        name = "no recorded semantics version" if version is None else f"semantics {version}"
        rows = sorted(stale[version])
        shown = ", ".join(rows[:3]) + (f" (+{len(rows) - 3} more)" if len(rows) > 3 else "")
        return f"  {len(rows):3d} row(s) at {name}: {shown}"

    # A cache holding more than one version refuses WHOLE rather than
    # being partially analysed. Counting the versions actually seen, not
    # just the stale ones: a directory of current rows beside one stale
    # row is the dangerous shape, because a table built from the current
    # subset would silently answer a different question — over a
    # different set of checkpoints — from the one asked.
    mixed = "" if len(seen) == 1 else (
        f"\n  the cache is MIXED across {len(seen)} planner versions, so no subset of it is "
        "a comparable set either"
    )
    raise StaleEvidence(
        f"refusing to read {out_dir} as current comparative evidence: this planner is at "
        f"{PLANNER_SEMANTICS_CONST} {current} and the cache is not.\n"
        + "\n".join(describe(v) for v in sorted(stale, key=lambda v: (v is None, v)))
        + mixed
        + "\n  A verdict is comparable only with a verdict judged the same way. Re-sweep into "
        "a fresh directory (`arch_sweep.py --out <dir> run`); the cache is kept, it is simply "
        "not current."
    )


# ── selftest — the evidence guard, on caches with known answers ───────
#
# A checker nobody runs is not a check (`ci_throughput.py`'s own lesson),
# so this runs in CI. No network and no planner: every cache below is
# built here from a row shaped like the real thing.


def _row(version, *, ok: bool = True) -> dict:
    """A saved row that the comparative readers would classify happily.

    Deliberately COMPLETE: repo, ok, an artifact with a finding, a
    capability closure, and a planner block. The point of the negative
    control is that a row can be perfect in every way a reader inspects
    and still not be current evidence — so a fixture missing fields would
    test the wrong thing.
    """
    planner = {"package": "larql-vindex", "package_version": "0.2.0"}
    if version is not None:
        planner["semantics_version"] = version
    return {
        "repo": "acme/model",
        "ok": ok,
        "seconds": 1.0,
        "plan": {
            "admissible": False,
            "planner": planner,
            "artifacts": [
                {
                    "model_type": "llama",
                    "findings": [
                        {
                            "id": 1,
                            "subject": "text_config.rope_theta",
                            "class": "execution_semantic",
                            "category": "unrepresented",
                            "component": "text",
                            "cluster": "unclustered",
                            "detail": "declared and uncarried",
                        }
                    ],
                }
            ],
            "capabilities": [
                {
                    "capability": "text_generation",
                    "admissible": False,
                    "available": True,
                    "supported": True,
                    "blocking": 1,
                    "blocker_ids": [1],
                }
            ],
        },
    }


def _cache(tmp: pathlib.Path, rows: dict) -> pathlib.Path:
    """Write `{filename: record}` as a plans directory."""
    out = tmp / "plans"
    out.mkdir(parents=True, exist_ok=True)
    for name, record in rows.items():
        (out / f"{name}.json").write_text(json.dumps(record, indent=1))
    return out


def cmd_selftest(_args: argparse.Namespace) -> int:
    import tempfile

    failures: list[str] = []

    def ok(name: str, got, want) -> None:
        if got != want:
            failures.append(f"{name}: got {got!r}, want {want!r}")
        else:
            print(f"  ok  {name}")

    def refusal(rows: dict):
        """The refusal a cache of `rows` produces, or `None` if it loads."""
        with tempfile.TemporaryDirectory() as tmp:
            out = _cache(pathlib.Path(tmp), rows)
            try:
                require_current_evidence(
                    [(p, json.loads(p.read_text())) for p in sorted(out.glob("*.json"))],
                    out,
                )
            except StaleEvidence as e:
                return str(e)
            return None

    current = current_semantics_version()
    ok("the authority is the Rust constant, and it reads",
       isinstance(current, int) and current > 0, True)
    ok(f"{PLANNER_SEMANTICS_CONST} is declared exactly once in the source",
       PLANNER_SEMANTICS_SOURCE.read_text().count(f"pub const {PLANNER_SEMANTICS_CONST}"), 1)

    # ── The positive control ──────────────────────────────────────────
    ok("a cache at the current version loads",
       refusal({"a": _row(current), "b": _row(current)}), None)

    # ── THE control this rung exists for ──────────────────────────────
    #
    # The row below is complete, parses, and every field the comparative
    # readers touch is present and well-formed. It differs from the
    # positive control in ONE integer. It must refuse, and it must refuse
    # for that reason — which is why the message is asserted, not just
    # the fact of the exception.
    stale = refusal({"a": _row(current - 1)})
    ok("a syntactically perfect row one version behind REFUSES", stale is not None, True)
    ok("the refusal names the cached version", f"semantics {current - 1}" in (stale or ""), True)
    ok("the refusal names the current version", str(current) in (stale or ""), True)
    ok("the refusal names the cache path", "plans" in (stale or ""), True)
    # And the same row at the current version is accepted, so the refusal
    # above is attributable to the version and to nothing else about it.
    ok("the same row at the current version does not refuse",
       refusal({"a": _row(current)}), None)

    # ── Mixed caches refuse whole ─────────────────────────────────────
    mixed = refusal({"a": _row(current), "b": _row(current - 1)})
    ok("a MIXED cache refuses rather than analysing its current subset",
       mixed is not None, True)
    ok("and says it is mixed", "MIXED" in (mixed or ""), True)
    # The dangerous shape is current rows BESIDE stale ones, so mixedness
    # is counted over the versions actually present and not over the
    # stale ones alone — the first version of this message said nothing
    # about the case above, because it saw only one stale group.
    two_stale = refusal({"a": _row(current - 1), "b": _row(current - 2)})
    ok("two distinct stale versions are mixed too", "MIXED" in (two_stale or ""), True)
    ok("a cache uniformly one version behind is NOT reported as mixed",
       "MIXED" in (refusal({"a": _row(current - 1), "b": _row(current - 1)}) or ""), False)

    # ── A plan with no stamp is of unknown vintage ────────────────────
    unstamped = refusal({"a": _row(None)})
    ok("a plan recording no semantics version refuses", unstamped is not None, True)
    ok("and is named as unstamped rather than as some number",
       "no recorded semantics version" in (unstamped or ""), True)

    # ── Rows that carry no verdict cannot be stale ────────────────────
    #
    # An unreachable repo asserts nothing about semantics. Refusing on it
    # would report a gated checkpoint as a versioning problem.
    unreachable = {"repo": "acme/gated", "ok": False, "error": "HTTP 401"}
    ok("an unreachable row does not make a current cache stale",
       refusal({"a": _row(current), "b": unreachable}), None)
    ok("a cache of nothing but unreachable rows does not refuse",
       refusal({"b": unreachable}), None)

    # ── Scope: `run` is untouched ─────────────────────────────────────
    #
    # The guard belongs to the readers that turn rows into claims. `run`
    # writes rows and must stay able to overwrite a stale directory —
    # otherwise the only way to repair a stale cache would be to delete
    # it by hand, and this rung deletes nothing.
    # Imported HERE and not at module scope: `arch_sweep` imports this
    # module, so a top-level import back would be a cycle. The wiring is
    # still this module's property to assert — the guard is worthless if
    # the reader does not call it, or calls it too late.
    import inspect

    import arch_sweep

    load_results, cmd_run = arch_sweep.load_results, arch_sweep.cmd_run

    ok("the guard is reached from the comparative reader",
       "require_current_evidence" in inspect.getsource(load_results), True)
    # And BEFORE the reader shells out to the built binary: the refusal
    # must be reachable without a compiled planner.
    source = inspect.getsource(load_results)
    ok("the guard runs before the reader needs a built binary",
       source.index("require_current_evidence") < source.index("recognised_model_types()"), True)
    ok("and NOT from `run`",
       "require_current_evidence" in inspect.getsource(cmd_run), False)

    print()
    if failures:
        for f in failures:
            print(f"  FAIL {f}")
        print(f"{len(failures)} failing check(s)")
        return 1
    print("selftest: all checks passed")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser(
        "selftest", help="the evidence guard, against caches with known answers"
    ).set_defaults(func=cmd_selftest)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
