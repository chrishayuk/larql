#!/usr/bin/env python3
"""Architecture conformance sweep — how much of the ecosystem already
collapses onto VINDEX3 semantics.

Runs `vindex plan hf://<repo>` over a matrix of checkpoints. `plan` reads
configuration and safetensors headers only, so a 2.8T checkpoint costs
about as much as a 0.6B one and the sweep is bounded by round trips, not
by weights.

Each row lands in one of four outcomes:

  GREEN  admissible — every declaration has a home in the schema
  AMBER  representable, not executable — a component is positively
         identified and this build has no implementation for it
  RED    semantic gap — the checkpoint declares something VINDEX3 cannot
         express. This is the gold: it says where the ontology must grow
  BUG    should work but doesn't — an alias, a prefix, a default

Outcomes are read from the plan, never asserted here: this script
classifies and tabulates, and every claim it prints is traceable to a
saved plan JSON under `--out`.

    scripts/arch_sweep.py run              # sweep, resumable
    scripts/arch_sweep.py report           # table + semantic leverage map
    scripts/arch_sweep.py envelopes        # coverage by semantic shape
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import pathlib
import re
import subprocess
import sys
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_MATRIX = REPO_ROOT / "docs" / "arch-conformance" / "matrix.json"
DEFAULT_OUT = pathlib.Path.home() / "chris-models" / "_conformance"
VINDEX_BIN = REPO_ROOT / "target" / "release" / "vindex"
LARQL_BIN = REPO_ROOT / "target" / "release" / "larql"

# Network-bound, and the hub rate-limits. Four keeps the sweep moving
# without turning a 429 into a false "unreachable" verdict.
DEFAULT_WORKERS = 4
PLAN_TIMEOUT_S = 900

# A finding class that means "we know exactly what this is and have no
# implementation" — representable identity, absent execution operator.
UNSUPPORTED_COMPONENT = "unsupported_component"

# `SemanticClass::is_critical` (plan/report.rs), mirrored. A finding blocks
# when it is not representable AND its class is critical — so `training_only`
# and `alias` findings are NOT blockers. Reading "every non-representable
# finding" as a blocker inflates the gap: it counts `router_aux_loss_coef`,
# a training-time constant, as an architecture VINDEX3 cannot express.
CRITICAL_CLASSES = {
    "execution_semantic",
    "tensor_semantic",
    "interface_semantic",
    UNSUPPORTED_COMPONENT,
    "unknown",
}

# The capability the sweep scores against. The claim under test is about
# TEXT architectures, so the gate must be the text-generation closure and
# not whole-model completeness — a checkpoint whose only blockers are in a
# vision tower is not evidence that VINDEX3 cannot represent its language
# model. The plan computes this closure itself; it is read, not re-derived.
SCORED_CAPABILITY = "text_generation"

# How an unreachable repo is told apart from a semantic gap. A repo the
# sweep cannot read is not evidence about VINDEX3, and must never be
# counted as one.
UNREACHABLE_PATTERNS = [
    (re.compile(r"HTTP 401|HTTP 403"), "gated"),
    # A repo that exists but ships only Mistral-native `params.json` is not
    # absent: it is a config DIALECT this source does not read. Saying
    # "absent" would file a real gap under a network accident.
    (re.compile(r"has no config\.json"), "no-config.json"),
    (re.compile(r"HTTP 404"), "absent"),
    (re.compile(r"HTTP 429"), "rate-limited"),
]

OUTCOMES = ["GREEN", "AMBER", "RED", "BUG", "UNREACHABLE"]


# ── Semantic clustering ────────────────────────────────────────────────
#
# RETIRED. This module used to carry a regex table mapping finding
# subjects to semantic ideas, and it was scratch analysis pretending to be
# a taxonomy: a fourth authority on a question the compiler already
# answered three ways. The plan now carries `cluster` on every finding and
# `blocker_ids` on every capability closure, so everything below reads
# what VINDEX3 decided rather than guessing from finding text.
#
# The concrete gain is that leverage stopped being an estimate. With only
# a blocking COUNT per closure, "retiring idea X clears N rows" had to be
# computed over whole-model findings and hedged as an upper bound. With
# the blocking SET named, the three quantities separate cleanly — and they
# are three different numbers, which is the lesson the inert wave taught:
#
#   reach             checkpoints that contain this idea at all
#   blockers removed  how much semantic debt retiring it deletes
#   rows cleared      how many verdicts actually move
#
# Reach ranks the work. Only rows-cleared predicts a verdict. The inert
# clusters reached 34 checkpoints and cleared one.

UNCLUSTERED = "unclustered"

# There is deliberately NO list of "non-text" clusters to exclude from the
# leverage tables. The text closure names `modality_vision` findings on 18
# checkpoints — a container whose root the closure cannot assign blocks on
# its image keys — and the script reads what the closure decided. An earlier
# exclusion set was spelled two ways (`modality_vision` at the top of the
# file, `modality.vision` above the cover) and so matched nothing, which is
# the same lesson as the retired regex table: two derivations of one fact.


def slug(repo: str) -> str:
    return repo.replace("/", "__")


def load_matrix(path: pathlib.Path) -> list[dict]:
    return json.loads(path.read_text())["checkpoints"]


def load_envelopes(path: pathlib.Path) -> dict[str, dict]:
    """The envelope vocabulary the matrix declares beside its rows."""
    return json.loads(path.read_text())["envelopes"]


def recognised_model_types() -> list[tuple[str, str]]:
    """The match patterns this build recognises, from its own manifest.

    Patterns, not labels. Half the registry matches by prefix, so a
    checkpoint declaring `gemma3_text` or `granitemoehybrid` matches an
    entry whose label is `gemma3` / `granite`. Comparing declared strings
    against labels reports four supported families as unsupported.
    """
    out = subprocess.run(
        [str(LARQL_BIN), "capabilities"], capture_output=True, text=True, check=True
    )
    return [
        (pattern["kind"], pattern["value"])
        for arch in json.loads(out.stdout)["architectures"]
        for pattern in arch["matches"]
    ]


def engine_recognises(model_type: str, patterns: list[tuple[str, str]]) -> bool:
    """`larql_models::detect::find_architecture`, mirrored.

    False means `detect_from_json` falls through to `GenericArch` — which
    serves the checkpoint with Llama-style defaults rather than refusing.
    """
    return any(
        model_type == value if kind == "exact" else model_type.startswith(value)
        for kind, value in patterns
    )


def run_plan(repo: str, out_dir: pathlib.Path, refresh: bool) -> dict:
    """Plan one checkpoint, caching the raw result under `out_dir`.

    The stored record is the evidence; everything downstream is derived
    from it, so a re-run of `report` can never disagree with the sweep
    that produced it.
    """
    dest = out_dir / f"{slug(repo)}.json"
    if dest.exists() and not refresh:
        return json.loads(dest.read_text())

    started = time.time()
    proc = subprocess.run(
        [str(VINDEX_BIN), "plan", f"hf://{repo}", "--json"],
        capture_output=True,
        text=True,
        timeout=PLAN_TIMEOUT_S,
    )
    elapsed = round(time.time() - started, 1)
    if proc.returncode == 0:
        record = {"repo": repo, "ok": True, "seconds": elapsed, "plan": json.loads(proc.stdout)}
    else:
        record = {
            "repo": repo,
            "ok": False,
            "seconds": elapsed,
            "error": (proc.stderr or proc.stdout).strip(),
        }
    dest.write_text(json.dumps(record, indent=1))
    return record


def classify(record: dict, recognised: list[tuple[str, str]]) -> dict:
    """Turn one plan record into an outcome, with its reason.

    The outcome is derived from the plan's own verdict and finding
    classes. Nothing here decides representability on its own.
    """
    repo = record["repo"]
    if not record["ok"]:
        message = record.get("error", "")
        for pattern, reason in UNREACHABLE_PATTERNS:
            if pattern.search(message):
                return {"repo": repo, "outcome": "UNREACHABLE", "reason": reason, "blockers": []}
        return {
            "repo": repo,
            "outcome": "BUG",
            "reason": message.splitlines()[0][:160] if message else "plan failed",
            "blockers": [],
        }

    plan = record["plan"]
    artifacts = plan.get("artifacts", [])
    model_types = [a.get("model_type", "?") for a in artifacts]
    findings = {f["id"]: f for a in artifacts for f in a.get("findings", [])}
    blockers = [
        {
            "subject": f["subject"],
            "class": f["class"],
            "category": f["category"],
            "component": f.get("component", ""),
            "cluster": f.get("cluster", UNCLUSTERED),
            "detail": f["detail"],
        }
        for f in findings.values()
        if f["category"] != "representable" and f["class"] in CRITICAL_CLASSES
    ]
    executable = any(engine_recognises(t, recognised) for t in model_types)

    # The plan's own verdict for the scored capability. Absent only on a
    # plan too old to carry capabilities, in which case fall back to the
    # whole-model verdict and say so rather than inventing one.
    text = next(
        (c for c in plan.get("capabilities", []) if c["capability"] == SCORED_CAPABILITY),
        None,
    )
    text_blocking = text["blocking"] if text else None
    text_admissible = text["admissible"] if text else plan.get("admissible")
    # The exact set this checkpoint is blocked on, by idea. Not derived
    # from the whole-model findings any more: these are the findings the
    # plan's own text closure named.
    text_clusters = sorted(
        {
            findings[i].get("cluster", UNCLUSTERED)
            for i in (text or {}).get("blocker_ids", [])
            if i in findings
        }
    )

    # AMBER's own definition is "component identified, no implementation",
    # so it cannot be reported for a checkpoint whose `model_type` matches
    # no registered family: that component is not identified, and calling
    # it AMBER would say this build knows what the model is when it does
    # not. Such a row stays RED until its identity resolves, and only then
    # can an unsupported component be the thing standing in its way.
    identity_unresolved = any(b["cluster"] == "architecture_identity" for b in blockers)
    if text_admissible:
        outcome = "GREEN"
        reason = "text closure: every declaration has a home"
    elif not identity_unresolved and any(
        b["class"] == UNSUPPORTED_COMPONENT for b in blockers
    ):
        outcome = "AMBER"
        reason = "component identified, no implementation"
    else:
        outcome = "RED"
        n = text_blocking if text_blocking is not None else len(blockers)
        reason = f"{n} blocking on text closure"

    return {
        "repo": repo,
        "outcome": outcome,
        "reason": reason,
        "model_types": model_types,
        "executable": executable,
        "whole_model_admissible": bool(plan.get("admissible")),
        "text_blocking": text_blocking,
        "blockers": blockers,
        "text_clusters": text_clusters,
        "seconds": record.get("seconds"),
    }


def cmd_run(args: argparse.Namespace) -> int:
    matrix = load_matrix(args.matrix)
    if args.priority:
        matrix = [m for m in matrix if m["priority"] in args.priority]
    out_dir = args.out / "plans"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"sweeping {len(matrix)} checkpoints -> {out_dir}", flush=True)
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_plan, m["repo"], out_dir, args.refresh): m for m in matrix
        }
        for future in concurrent.futures.as_completed(futures):
            entry = futures[future]
            done += 1
            try:
                record = future.result()
                mark = "ok " if record["ok"] else "ERR"
            except Exception as exc:  # a crashed plan is a result, not a stop
                mark = "ERR"
                record = {"repo": entry["repo"], "ok": False, "error": f"{type(exc).__name__}: {exc}"}
                (out_dir / f"{slug(entry['repo'])}.json").write_text(json.dumps(record, indent=1))
            print(f"  [{done}/{len(matrix)}] {mark} {entry['repo']}", flush=True)
    return 0


def load_results(args: argparse.Namespace) -> tuple[list[dict], dict]:
    matrix = {m["repo"]: m for m in load_matrix(args.matrix)}
    recognised = recognised_model_types()
    rows = []
    for repo, meta in matrix.items():
        path = args.out / "plans" / f"{slug(repo)}.json"
        if not path.exists():
            continue
        row = classify(json.loads(path.read_text()), recognised)
        row.update(meta)
        rows.append(row)
    return rows, recognised


def cmd_report(args: argparse.Namespace) -> int:
    rows, recognised = load_results(args)
    if not rows:
        print("no plans found — run `arch_sweep.py run` first", file=sys.stderr)
        return 1

    by_outcome = {o: [r for r in rows if r["outcome"] == o] for o in OUTCOMES}
    scored = [r for r in rows if r["outcome"] != "UNREACHABLE"]

    print(f"\n{'=' * 78}\nVINDEX3 ARCHITECTURE CONFORMANCE\n{'=' * 78}")
    print(f"checkpoints planned   {len(rows)}")
    print(f"  scored              {len(scored)}")
    print(f"  unreachable         {len(by_outcome['UNREACHABLE'])} (not evidence about VINDEX3)")
    print(f"lineages              {len(set(r['lineage'] for r in scored))}")
    print(f"family-generations    {len(set(r['generation'] for r in scored))}")
    for outcome in ["GREEN", "AMBER", "RED", "BUG"]:
        n = len(by_outcome[outcome])
        pct = f"{100 * n / len(scored):.0f}%" if scored else "-"
        print(f"  {outcome:<6}{n:>4}  {pct}")

    whole = sum(1 for r in scored if r.get("whole_model_admissible"))
    print(f"whole-model admissible {whole}  (text closure is the scored gate)")

    print(f"\n{'-' * 78}\nBY FAMILY-GENERATION\n{'-' * 78}")
    print(f"{'generation':<22}{'lineage':<18}{'text':<8}{'whole':<7}{'exec':<6}{'model_type':<18}reason")
    for row in sorted(scored, key=lambda r: (r["lineage"], r["generation"], r["repo"])):
        mt = ",".join(row.get("model_types", []))[:17]
        ex = "yes" if row.get("executable") else "no"
        wh = "yes" if row.get("whole_model_admissible") else "no"
        print(
            f"{row['generation'][:21]:<22}{row['lineage'][:17]:<18}"
            f"{row['outcome']:<8}{wh:<7}{ex:<6}{mt:<18}{row['reason'][:26]}"
        )

    if by_outcome["UNREACHABLE"]:
        print(f"\n{'-' * 78}\nUNREACHABLE\n{'-' * 78}")
        reason_width = max(len(reason) for _, reason in UNREACHABLE_PATTERNS) + 2
        for row in sorted(by_outcome["UNREACHABLE"], key=lambda r: r["repo"]):
            print(f"  {row['reason']:<{reason_width}}{row['repo']}")

    print(f"\n{'-' * 78}\nSEMANTIC LEVERAGE — one fix, how many checkpoints\n{'-' * 78}")
    clusters: dict[tuple[str, str, str], set[str]] = {}
    for row in scored:
        for blocker in row["blockers"]:
            key = (blocker["subject"], blocker["class"], blocker["component"])
            clusters.setdefault(key, set()).add(row["repo"])
    print(f"{'blocking subject':<44}{'class':<22}{'component':<10}ckpts")
    for (subject, cls, component), repos in sorted(
        clusters.items(), key=lambda kv: -len(kv[1])
    )[: args.top]:
        print(f"{subject[:43]:<44}{cls[:21]:<22}{component[:9]:<10}{len(repos)}")
    covered = set(r for v in clusters.values() for r in v)
    print(f"\n{len(clusters)} distinct blocking subjects over {len(covered)} checkpoints")

    if args.json:
        args.json.write_text(json.dumps(rows, indent=1))
        print(f"\nwrote {args.json}")
    return 0



def cmd_envelopes(args: argparse.Namespace) -> int:
    """Coverage by semantic shape — the claim worth publishing.

    Every scored row sits in exactly one envelope, declared on the row in
    the matrix. The verdict columns are the sweep's; this only groups.
    """
    rows, _ = load_results(args)
    envelopes = load_envelopes(args.matrix)
    scored = [r for r in rows if r["outcome"] != "UNREACHABLE"]

    groups: dict[str, list[dict]] = {}
    for row in scored:
        groups.setdefault(row["envelope"], []).append(row)

    print(f"\n{'=' * 78}\nCOVERAGE BY SEMANTIC ENVELOPE — {len(scored)} scored rows\n{'=' * 78}")
    header = ("| envelope | lineages | rows | gens | GREEN | AMBER | RED |", "|---|---|---:|---:|---:|---:|---:|")
    print("\n".join(header))
    for slug in envelopes:
        members = groups.get(slug)
        if not members:
            continue
        lineages = sorted({r["lineage"] for r in members})
        gens = {r["generation"] for r in members}
        counts = {o: sum(1 for r in members if r["outcome"] == o) for o in ("GREEN", "AMBER", "RED")}
        print(
            f"| {envelopes[slug]['name']} | {', '.join(lineages)} | {len(members)} | {len(gens)} "
            f"| {counts['GREEN']} | {counts['AMBER']} | {counts['RED']} |"
        )
    unfiled = sorted(slug for slug in groups if slug not in envelopes)
    if unfiled:
        print(f"\nrows filed under an undeclared envelope: {unfiled}")
        return 1
    return 0


def cmd_clusters(args: argparse.Namespace) -> int:
    """Subjects -> semantic ideas, and how much each idea would unlock."""
    rows, _ = load_results(args)
    scored = [r for r in rows if r["outcome"] != "UNREACHABLE"]

    clusters: dict[str, dict] = {}
    subjects: set[str] = set()
    for row in scored:
        for blocker in row["blockers"]:
            subjects.add(blocker["subject"])
            entry = clusters.setdefault(
                blocker["cluster"],
                {"repos": set(), "generations": set(), "subjects": set(), "classes": set()},
            )
            entry["repos"].add(row["repo"])
            entry["generations"].add(row["generation"])
            entry["subjects"].add(blocker["subject"])
            entry["classes"].add(blocker["class"])

    scope = "whole model — the scored verdict uses the plan's own text closure"
    blocked = set(r["repo"] for r in scored if r["blockers"])
    print(f"\n{'=' * 78}\nSEMANTIC CENSUS — {scope}\n{'=' * 78}")
    print(f"checkpoints with a blocker  {len(blocked)}")
    print(f"distinct blocking subjects  {len(subjects)}")
    print(f"semantic clusters           {len(clusters)}")
    if clusters:
        print(f"compression                 {len(subjects) / len(clusters):.2f}x "
              f"(subjects per idea)")

    print(f"\n{'cluster':<32}{'ckpts':<7}{'gens':<6}{'subjects':<10}classes")
    for name, entry in sorted(clusters.items(), key=lambda kv: -len(kv[1]["repos"])):
        print(f"{name:<32}{len(entry['repos']):<7}{len(entry['generations']):<6}"
              f"{len(entry['subjects']):<10}{','.join(sorted(entry['classes']))[:30]}")

    if args.show:
        entry = clusters.get(args.show)
        if not entry:
            print(f"\nno cluster named {args.show}")
            return 1
        print(f"\n{'-' * 78}\n{args.show}\n{'-' * 78}")
        print("subjects:", ", ".join(sorted(entry["subjects"])))
        print("\ngenerations:", ", ".join(sorted(entry["generations"])))
    return 0



def cmd_leverage(args: argparse.Namespace) -> int:
    """Which ideas to retire, and what each one actually buys.

    Three quantities, kept apart because conflating them cost a wrong
    prediction: the inert clusters reached 34 checkpoints and cleared one.

      reach     checkpoints whose text closure is blocked on this idea
      removed   blocking findings retiring it deletes
      cleared   checkpoints whose text closure would then be EMPTY

    All three come from the plan's own text-generation closure — the
    findings it named, not findings matched by hand — so `cleared` is
    exact rather than an upper bound.
    """
    rows, _ = load_results(args)
    scored = [r for r in rows if r["outcome"] != "UNREACHABLE"]
    blocked = {
        r["repo"]: set(r["text_clusters"])
        for r in scored
        if r.get("text_clusters")
    }

    reach: dict[str, int] = {}
    removed: dict[str, int] = {}
    for row in scored:
        for cluster in row.get("text_clusters", []):
            reach[cluster] = reach.get(cluster, 0) + 1
        for blocker in row["blockers"]:
            c = blocker["cluster"]
            removed[c] = removed.get(c, 0) + 1

    print(f"\n{'=' * 78}\nENGINEERING LEVERAGE — exact, from the text closure\n{'=' * 78}")
    print(f"{len(blocked)} checkpoints have a non-empty text closure\n")
    print(f"{'idea':<34}{'reach':>7}{'removed':>9}{'clears alone':>14}")
    solo = {
        c: sum(1 for v in blocked.values() if v == {c})
        for c in reach
    }
    for cluster in sorted(reach, key=lambda c: (-reach[c], c)):
        print(f"{cluster:<34}{reach[cluster]:>7}{removed.get(cluster, 0):>9}"
              f"{solo[cluster]:>14}")

    print(f"\n{'-' * 78}\nGREEDY COVER — retire in this order\n{'-' * 78}")
    print(f"{'#':<3}{'idea':<34}{'clears':>8}{'cumulative':>12}")
    retired: set[str] = set()
    cleared: set[str] = set()
    for step in range(1, args.steps + 1):
        best, best_repos = None, set()
        for candidate in set(reach) - retired:
            trial = retired | {candidate}
            repos = {r for r, v in blocked.items() if v <= trial} - cleared
            if len(repos) > len(best_repos):
                best, best_repos = candidate, repos
        if not best or not best_repos:
            break
        retired.add(best)
        cleared |= best_repos
        print(f"{step:<3}{best:<34}{len(best_repos):>8}{len(cleared):>12}")

    print(f"\n{len(retired)} ideas -> {len(cleared)} of {len(blocked)} checkpoints clear "
          f"({100 * len(cleared) / len(blocked):.0f}%)" if blocked else "")
    if args.show_remaining:
        remaining = {r: v for r, v in blocked.items() if r not in cleared}
        print(f"\nstill blocked ({len(remaining)}):")
        for repo, v in sorted(remaining.items()):
            print(f"  {repo:<48}{','.join(sorted(v - retired))[:60]}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--matrix", type=pathlib.Path, default=DEFAULT_MATRIX)
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="plan every checkpoint in the matrix")
    run.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    run.add_argument("--refresh", action="store_true", help="re-plan rows already saved")
    run.add_argument("--priority", nargs="*", help="restrict to these priorities, e.g. P0")
    run.set_defaults(func=cmd_run)

    report = sub.add_parser("report", help="tabulate the saved plans")
    report.add_argument("--json", type=pathlib.Path, help="also write the rows here")
    report.add_argument("--top", type=int, default=30, help="leverage rows to show")
    report.set_defaults(func=cmd_report)

    envelopes = sub.add_parser("envelopes", help="coverage by semantic shape")
    envelopes.set_defaults(func=cmd_envelopes)

    clusters = sub.add_parser("clusters", help="subjects -> semantic ideas")
    clusters.add_argument("--show", help="list the subjects in one cluster")
    clusters.set_defaults(func=cmd_clusters)

    leverage = sub.add_parser("leverage", help="greedy cover over semantic ideas")
    leverage.add_argument("--steps", type=int, default=12)
    leverage.add_argument("--show-remaining", action="store_true")
    leverage.set_defaults(func=cmd_leverage)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
