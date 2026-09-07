#!/usr/bin/env python3
"""Which Metal evidence does this change actually require?

CI-E2-D. The Metal gate costs ~24 minutes of a scarce macOS runner and
fires whenever `crates/larql-compute/**` or `crates/larql-models/**`
moves. Over pull requests #415-#449 that was 18 triggers, and 12 of them
touched no Metal file at all.

The wrong fix is a cleverer set of path globs. `crates/larql-compute/**`
holds both the trait boundary Metal LINKS against and the CPU kernels
Metal's tests compare NUMBERS against -- 72 references to
`cpu::ops::q4_common` alone -- and no amount of directory ancestry
separates those two. So the distinction is DECLARED, in
crates/larql-compute-metal/parity-obligations.json, and this script
checks the declaration stays complete.

Three tiers:

    A  full behavioural qualification   the Metal crate itself changed,
                                        or a declared parity obligation
    B  compatibility only               a crate Metal depends on changed
                                        in a way that can break the
                                        COMPILE but not the numbers
    C  nothing                          Metal cannot be affected

The asymmetry that sets every uncertain call: a surface wrongly placed
in tier B is UNSAFE (Metal never re-qualifies against changed numerics);
wrongly placed in tier A it is merely expensive. Uncertain goes to A.

    metal_tier.py classify --files-from <path>   # one path per line
    metal_tier.py check                          # manifest completeness
    metal_tier.py selftest
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
METAL_CRATE = "crates/larql-compute-metal"
MANIFEST = Path(METAL_CRATE) / "parity-obligations.json"

TIER_A, TIER_B, TIER_C = "A", "B", "C"

# Tier A on their own: the Metal crate, and the workflow that defines the
# gate (a change to the gate must be qualified by the gate).
TIER_A_ALWAYS = (
    f"{METAL_CRATE}/**",
    ".github/workflows/larql-compute-metal.yml",
)

# Tier B floor: the workspace crates Metal actually depends on, read from
# its Cargo.toml rather than assumed, plus the files that can change the
# build itself.
BUILD_INPUTS = ("Cargo.toml", "Cargo.lock", "Makefile", "rust-toolchain.toml")

# Where a reference to `larql_compute::foo::bar` is looked for.
CRATE_DIRS = {"larql_compute": "crates/larql-compute", "larql_models": "crates/larql-models"}
REFERENCE_RE = re.compile(r"larql_(compute|models)((?:::[a-z_0-9]+)+)")


def load_manifest(root: Path = REPO_ROOT) -> dict:
    return json.loads((root / MANIFEST).read_text())


def manifest_paths(manifest: dict, key: str) -> list[str]:
    return [p for entry in manifest.get(key, []) for p in entry["paths"]]


def matches(path: str, pattern: str) -> bool:
    """Glob match where `**` means 'this subtree'."""
    if pattern.endswith("/**"):
        return path == pattern[:-3] or path.startswith(pattern[:-2])
    return fnmatch.fnmatch(path, pattern)


def metal_workspace_deps(root: Path = REPO_ROOT) -> list[str]:
    """Workspace crates Metal depends on, read from its manifest.

    Declared in Cargo.toml, not inferred from the directory tree: that is
    the whole point. A crate that stops being a dependency stops being a
    tier-B trigger without anyone editing a glob.
    """
    import tomllib

    cargo = tomllib.loads((root / METAL_CRATE / "Cargo.toml").read_text())
    out: set[str] = set()
    for section in ("dependencies", "dev-dependencies", "build-dependencies"):
        for spec in (cargo.get(section) or {}).values():
            if isinstance(spec, dict) and "path" in spec:
                rel = os.path.normpath(os.path.join(METAL_CRATE, spec["path"]))
                out.add(f"{rel}/**")
    return sorted(out)


def classify(changed: list[str], manifest: dict, deps: list[str]) -> tuple[str, list[str]]:
    """Highest tier any changed file demands, with the reasons."""
    reasons: list[str] = []
    tier = TIER_C
    parity = manifest_paths(manifest, "parity_obligations")
    interface = manifest_paths(manifest, "interface_only")

    for path in changed:
        # interface_only is an EXEMPTION from parity, so it is tested
        # first — otherwise a file inside a declared parity subtree could
        # never be exempted.
        if any(matches(path, p) for p in interface):
            if tier == TIER_C:
                tier = TIER_B
            reasons.append(f"{path}: interface_only -> B")
            continue
        if any(matches(path, p) for p in TIER_A_ALWAYS):
            tier = TIER_A
            reasons.append(f"{path}: Metal crate or its workflow -> A")
            continue
        if any(matches(path, p) for p in parity):
            tier = TIER_A
            reasons.append(f"{path}: declared parity obligation -> A")
            continue
        if any(matches(path, p) for p in deps) or path in BUILD_INPUTS:
            if tier == TIER_C:
                tier = TIER_B
            reasons.append(f"{path}: dependency or build input -> B")
    return tier, reasons


def referenced_paths(root: Path = REPO_ROOT) -> dict[str, int]:
    """Upstream paths the Metal crate names, resolved to files/subtrees.

    APPROXIMATE, and deliberately used only as a completeness CHECK on the
    declaration rather than as the declaration itself: brace imports
    (`cpu::{ops, q4}`) and type imports (`cpu::Foo`) truncate here, so this
    under-resolves. Under-resolution makes the check conservative — it
    reports a shallower path, which is harder to satisfy, not easier.
    """
    hits: dict[str, int] = {}
    for sub in ("src", "tests", "benches"):
        base = root / METAL_CRATE / sub
        if not base.is_dir():
            continue
        for dirpath, _dirs, files in os.walk(base):
            for name in files:
                if not name.endswith(".rs"):
                    continue
                text = (Path(dirpath) / name).read_text(errors="replace")
                for match in REFERENCE_RE.finditer(text):
                    crate_dir = CRATE_DIRS[f"larql_{match.group(1)}"]
                    parts = [p for p in match.group(2).split("::") if p]
                    resolved = _resolve(root, crate_dir, parts)
                    if resolved:
                        hits[resolved] = hits.get(resolved, 0) + 1
    return hits


def _resolve(root: Path, crate_dir: str, parts: list[str]) -> str | None:
    rel = f"{crate_dir}/src"
    best: str | None = None
    for seg in parts:
        as_dir, as_file = f"{rel}/{seg}", f"{rel}/{seg}.rs"
        if (root / as_dir).is_dir():
            best, rel = f"{as_dir}/**", as_dir
        elif (root / as_file).is_file():
            return as_file
        else:
            break
    return best


def check(root: Path = REPO_ROOT) -> tuple[list[str], dict[str, int]]:
    """Referenced-but-undeclared paths. Empty list means complete."""
    manifest = load_manifest(root)
    declared = manifest_paths(manifest, "parity_obligations") + manifest_paths(manifest, "interface_only")
    undeclared = []
    hits = referenced_paths(root)
    for path in sorted(hits):
        probe = path[:-3] if path.endswith("/**") else path
        if not any(matches(probe, d) for d in declared):
            undeclared.append(path)
    return undeclared, hits


# ---------------------------------------------------------------- commands

def cmd_classify(args: argparse.Namespace) -> int:
    changed = [l.strip() for l in Path(args.files_from).read_text().splitlines() if l.strip()]
    manifest = load_manifest()
    tier, reasons = classify(changed, manifest, metal_workspace_deps())
    if args.quiet:
        print(tier)
        return 0
    print(f"tier: {tier}   ({len(changed)} changed files)")
    for r in reasons[: args.max_reasons]:
        print(f"  {r}")
    if len(reasons) > args.max_reasons:
        print(f"  ... and {len(reasons) - args.max_reasons} more")
    return 0


def cmd_check(_args: argparse.Namespace) -> int:
    undeclared, hits = check()
    print(f"{len(hits)} upstream paths referenced by {METAL_CRATE}")
    if undeclared:
        print("\nREFERENCED BUT NOT DECLARED — every one is a surface that could change")
        print("what Metal must reproduce with nothing in CI noticing:")
        for path in undeclared:
            print(f"  {path}  ({hits[path]} references)")
        return 1
    print("manifest is complete: every referenced path is declared")
    return 0


def cmd_selftest(_args: argparse.Namespace) -> int:
    failures: list[str] = []

    def ok(name: str, got, want) -> None:
        if got != want:
            failures.append(f"{name}: got {got!r}, want {want!r}")
        else:
            print(f"  ok  {name} == {want!r}")

    m = {
        "parity_obligations": [{"id": "p", "paths": ["crates/larql-compute/src/cpu/ops/**"]}],
        "interface_only": [{"id": "i", "paths": ["crates/larql-compute/src/options.rs"]}],
    }
    deps = ["crates/larql-compute/**", "crates/larql-models/**"]

    ok("metal source is A", classify([f"{METAL_CRATE}/src/lib.rs"], m, deps)[0], TIER_A)
    ok("the gate's own workflow is A",
       classify([".github/workflows/larql-compute-metal.yml"], m, deps)[0], TIER_A)
    ok("declared parity path is A",
       classify(["crates/larql-compute/src/cpu/ops/q4k_matvec.rs"], m, deps)[0], TIER_A)
    ok("undeclared dependency path is B",
       classify(["crates/larql-compute/src/cpu/spin_pool.rs"], m, deps)[0], TIER_B)
    ok("interface_only inside a dependency is B",
       classify(["crates/larql-compute/src/options.rs"], m, deps)[0], TIER_B)
    ok("Cargo.lock alone is B", classify(["Cargo.lock"], m, deps)[0], TIER_B)
    ok("docs are C", classify(["docs/ci-throughput/README.md"], m, deps)[0], TIER_C)
    ok("an unrelated crate is C", classify(["crates/larql-lql/src/lib.rs"], m, deps)[0], TIER_C)

    # The tier is the MAXIMUM demand across the diff, never the last file seen.
    ok("A wins over C regardless of order",
       classify(["docs/x.md", f"{METAL_CRATE}/src/lib.rs"], m, deps)[0], TIER_A)
    ok("A wins over C in the other order",
       classify([f"{METAL_CRATE}/src/lib.rs", "docs/x.md"], m, deps)[0], TIER_A)
    ok("A wins over B", classify(["Cargo.lock", "crates/larql-compute/src/cpu/ops/q.rs"], m, deps)[0], TIER_A)
    ok("empty diff is C", classify([], m, deps)[0], TIER_C)

    # An exemption must be able to fire INSIDE a declared parity subtree,
    # or interface_only would be unreachable wherever it mattered.
    nested = {
        "parity_obligations": [{"id": "p", "paths": ["crates/larql-compute/src/**"]}],
        "interface_only": [{"id": "i", "paths": ["crates/larql-compute/src/options.rs"]}],
    }
    ok("interface_only exempts a file inside a parity subtree",
       classify(["crates/larql-compute/src/options.rs"], nested, deps)[0], TIER_B)
    ok("its sibling is still A",
       classify(["crates/larql-compute/src/other.rs"], nested, deps)[0], TIER_A)

    ok("subtree glob matches the directory itself",
       matches("crates/larql-compute/src/cpu", "crates/larql-compute/src/cpu/**"), True)
    ok("subtree glob does not match a prefix sibling",
       matches("crates/larql-compute/src/cpu_extra.rs", "crates/larql-compute/src/cpu/**"), False)

    deps_real = metal_workspace_deps()
    ok("dependencies are read from Cargo.toml",
       set(deps_real), {"crates/larql-compute/**", "crates/larql-models/**"})

    undeclared, hits = check()
    ok("the shipped manifest references something at all", len(hits) > 0, True)
    ok("the shipped manifest is complete", undeclared, [])

    print()
    if failures:
        for f in failures:
            print(f"  FAIL {f}")
        print(f"{len(failures)} failing check(s)")
        return 1
    print("selftest: all checks passed")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_cl = sub.add_parser("classify", help="tier for a set of changed files")
    p_cl.add_argument("--files-from", required=True, help="file with one changed path per line")
    p_cl.add_argument("--quiet", action="store_true", help="print only the tier letter")
    p_cl.add_argument("--max-reasons", type=int, default=10)
    p_cl.set_defaults(func=cmd_classify)

    sub.add_parser("check", help="manifest completeness").set_defaults(func=cmd_check)
    sub.add_parser("selftest", help="classification rules against known answers").set_defaults(func=cmd_selftest)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
