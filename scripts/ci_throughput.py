#!/usr/bin/env python3
"""CI throughput instrument — what a pull request actually costs in
wall clock and in runner minutes.

Written to score one specific intervention (E1: PR-run cancellation,
macOS PR occupancy, VINDEX bench placement) as a before/after, so that
the claim afterwards is a number and not "CI feels faster".

The unit of observation is an ATTEMPT: one head SHA of one pull request,
with every workflow run GitHub created for it. t0 for an attempt is the
earliest job-creation timestamp across those runs, which is the closest
thing the API offers to "when the push landed".

Every metric separates QUEUE from EXECUTION, because the two respond to
different changes and mixing them hides which one moved:

  queue      job.started_at   - job.created_at    (waiting for a runner)
  execution  job.completed_at - job.started_at    (doing the work)

Metrics, and the intervention each one scores:

  first-blocking-green   t0 -> earliest blocking run to succeed
                         perceived feedback speed
  merge-ready            t0 -> last blocking run to succeed
                         actual merge readiness
  macos-queue            max macOS queue wait in the attempt
                         scores the macOS-occupancy change
  superseded-execution   executed minutes spent on runs for a SHA that
                         a later push had already superseded
                         scores run cancellation
  windows-vindex         execution time of larql-vindex's Windows leg
                         scores bench placement
  queued/executed by OS  did pressure move, or actually go away

"Blocking" is defined here as every PR-triggered workflow EXCEPT those
named in --informational (default: mutants, which declares itself
advisory and runs continue-on-error). The repository ruleset carries no
required status checks, so this is a stated convention, not something
read off the branch protection.

    scripts/ci_throughput.py collect --since 2026-08-20 --out before.json
    scripts/ci_throughput.py report  --before before.json --after after.json
    scripts/ci_throughput.py selftest

`collect` snapshots the API; `report` never calls the network. Freeze the
BEFORE snapshot before the change lands — the runs are immutable, but
retention is not forever.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
FORECAST_DIR = REPO_ROOT / "docs" / "ci-throughput"

PER_PAGE = 100
MAX_PAGES = 50
DEFAULT_INFORMATIONAL = ("mutants",)
DEFAULT_PERCENTILES = (50, 95)
SECONDS_PER_MINUTE = 60.0

# Substring -> canonical OS name. Read off a job's `labels` (the literal
# `runs-on` values), never off the runner's hostname, which is assigned.
OS_LABEL_MARKERS = (
    ("ubuntu", "linux"),
    ("linux", "linux"),
    ("windows", "windows"),
    ("macos", "macos"),
    ("macOS", "macos"),
)
OS_ORDER = ("linux", "windows", "macos", "unknown")

# Metric KIND, declared once here rather than inferred at scoring time.
#
# An EXTENSIVE metric's value scales with how many attempts are in the
# sample: totals of runner-minutes, counts of jobs. An INTENSIVE one does
# not: percentiles, per-attempt means.
#
# This distinction is not decoration. E1 froze two extensive metrics
# (`queued.macos`, `executed.macos`) and collected 12 attempts against a
# 33-attempt baseline. A 12/33 ratio moves any per-attempt-flat total by
# -63.6% BEFORE any effect exists, so E1's P3 could not have returned
# HELD and its P2 could not have returned FALSIFIED. The adjudicator
# reported `INVALID COMPARISON` for the mechanism, which was the designed
# refusal — but it fired on sample size, not on the composition change P3
# was written to catch, and the mechanism came back unadjudicated rather
# than refuted.
#
# So: every extensive metric gets an intensive companion (`*_per_attempt`),
# and `score_prediction` REFUSES an extensive metric whose two samples
# differ in attempt count. A forecast that wants to score runner-minutes
# across unlike samples must name the per-attempt metric.
METRIC_EXTENSIVE = "extensive"
METRIC_INTENSIVE = "intensive"

# Prefixes/names that flatten_metrics() emits in extensive form. The
# per-attempt companions are derived from these, so the two can never
# drift apart.
EXTENSIVE_OS_BUCKETS = ("queued", "executed")
EXTENSIVE_SCALAR_METRICS = ("superseded_exec_total",)
PER_ATTEMPT_SUFFIX = "_per_attempt"


# Metric families whose name carries a workflow, e.g.
# `workflow_exec_max.larql-compute-metal.p50`. Their names only exist once
# that workflow appears in a snapshot, so a forecast that mentions one has
# to be validated by SHAPE plus a separate check that the workflow is real.
WORKFLOW_SCOPED_FAMILIES = ("workflow_exec_max", "workflow_share")
PROBE_WORKFLOW = "<workflow>"


def metric_shape(name: str) -> str:
    """A workflow-scoped metric name with the workflow replaced by a probe.

    `workflow_exec_max.larql-compute-metal.p50` -> `workflow_exec_max.<workflow>.p50`
    `workflow_share.bench-regress`              -> `workflow_share.<workflow>`

    Anything else is returned unchanged. rsplit, not split, because a
    workflow name may contain a dot; the STAT never does.
    """
    for family in WORKFLOW_SCOPED_FAMILIES:
        prefix = f"{family}."
        if not name.startswith(prefix):
            continue
        remainder = name[len(prefix):]
        if family == "workflow_share":
            return f"{family}.{PROBE_WORKFLOW}"
        workflow, _, stat = remainder.rpartition(".")
        return f"{family}.{PROBE_WORKFLOW}.{stat}" if workflow else name
    return name


# Metric families that are NOT durations. Printing a share as "0.5 min"
# is the unit error this whole programme exists to avoid making.
DIMENSIONLESS_FAMILIES = ("workflow_share",)


def metric_unit(name: str) -> str:
    return "" if name.split(".", 1)[0] in DIMENSIONLESS_FAMILIES else " min"


def metric_workflow(name: str) -> str | None:
    """The workflow a workflow-scoped metric names, or None."""
    for family in WORKFLOW_SCOPED_FAMILIES:
        prefix = f"{family}."
        if not name.startswith(prefix):
            continue
        remainder = name[len(prefix):]
        if family == "workflow_share":
            return remainder
        workflow, _, _stat = remainder.rpartition(".")
        return workflow or None
    return None


# The intensive companion of each extensive SCALAR. The OS buckets derive
# theirs by suffix; this table covers the ones that do not, so a refusal can
# never point at a metric the instrument does not produce. `selftest`
# asserts every extensive metric has a companion and that the companion is
# itself produced and intensive.
INTENSIVE_COMPANIONS = {"superseded_exec_total": "superseded_exec_mean"}


def intensive_companion(name: str) -> str | None:
    """The metric to score instead, when `name` is extensive."""
    if name in INTENSIVE_COMPANIONS:
        return INTENSIVE_COMPANIONS[name]
    stem, dot, rest = name.partition(".")
    if stem in EXTENSIVE_OS_BUCKETS and dot:
        return f"{stem}{PER_ATTEMPT_SUFFIX}.{rest}"
    return None


def metric_kind(name: str) -> str:
    """extensive | intensive, for any name flatten_metrics() produces."""
    stem = name.split(".", 1)[0]
    if stem.endswith(PER_ATTEMPT_SUFFIX):
        return METRIC_INTENSIVE
    if stem in EXTENSIVE_OS_BUCKETS or name in EXTENSIVE_SCALAR_METRICS:
        return METRIC_EXTENSIVE
    return METRIC_INTENSIVE

TERMINAL_SUCCESS = "success"
TERMINAL_CANCELLED = "cancelled"


# --------------------------------------------------------------------------
# time helpers
# --------------------------------------------------------------------------

def parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def seconds_between(start: datetime | None, end: datetime | None) -> float | None:
    if start is None or end is None:
        return None
    delta = (end - start).total_seconds()
    # Clock skew between GitHub's queue and runner clocks shows up as small
    # negatives. Clamp rather than discard: a negative queue wait is a zero
    # wait, and dropping the row would bias the percentile.
    return max(0.0, delta)


def fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    total = int(round(seconds))
    return f"{total // 60}m{total % 60:02d}s"


def fmt_minutes(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    return f"{seconds / SECONDS_PER_MINUTE:.1f}m"


def percentile(values: Sequence[float], pct: float) -> float | None:
    """Linear-interpolated percentile. Defined for n == 1, unlike
    statistics.quantiles, because early samples are small."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


# --------------------------------------------------------------------------
# GitHub API
# --------------------------------------------------------------------------

def gh_api(path: str, params: dict[str, str] | None = None) -> Any:
    # `--method GET` is not redundant: `gh api` defaults to POST as soon as
    # any `-f` parameter is present, which turns a read into a 404.
    cmd = ["gh", "api", "--method", "GET", "-H", "Accept: application/vnd.github+json", path]
    for key, value in (params or {}).items():
        cmd += ["-f", f"{key}={value}"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"gh api {path} failed: {proc.stderr.strip()}")
    return json.loads(proc.stdout)


def gh_paginated(path: str, key: str, params: dict[str, str] | None = None) -> list[dict]:
    out: list[dict] = []
    for page in range(1, MAX_PAGES + 1):
        payload = gh_api(path, {**(params or {}), "per_page": str(PER_PAGE), "page": str(page)})
        batch = payload.get(key, []) if isinstance(payload, dict) else payload
        if not batch:
            break
        out.extend(batch)
        if len(batch) < PER_PAGE:
            break
    return out


def repo_slug() -> str:
    return gh_api("repos/{owner}/{repo}")["full_name"]


def classify_os(labels: Iterable[str]) -> str:
    joined = " ".join(labels).lower()
    for marker, name in OS_LABEL_MARKERS:
        if marker.lower() in joined:
            return name
    return "unknown"


# --------------------------------------------------------------------------
# collect
# --------------------------------------------------------------------------

def collect(args: argparse.Namespace) -> int:
    slug = repo_slug()
    params = {"event": "pull_request"}
    if args.since:
        params["created"] = f">={args.since}"
    if args.branch:
        params["branch"] = args.branch

    print(f"collecting pull_request runs for {slug} ...", file=sys.stderr)
    runs = gh_paginated(f"repos/{slug}/actions/runs", "workflow_runs", params)
    if args.until:
        cutoff = parse_ts(f"{args.until}T23:59:59Z")
        runs = [r for r in runs if (parse_ts(r["created_at"]) or cutoff) <= cutoff]
    runs.sort(key=lambda r: r["created_at"], reverse=True)

    # Bound by PR, not by run count, so a snapshot is always a whole number
    # of pull requests and the two periods stay comparable.
    by_branch: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        by_branch[run.get("head_branch") or "?"].append(run)
    branches = sorted(by_branch, key=lambda b: max(r["created_at"] for r in by_branch[b]), reverse=True)
    if args.max_prs:
        branches = branches[: args.max_prs]
    selected = [r for b in branches for r in by_branch[b]]

    print(f"  {len(selected)} runs across {len(branches)} branches; fetching jobs", file=sys.stderr)
    records = []
    for index, run in enumerate(selected, 1):
        jobs = gh_paginated(f"repos/{slug}/actions/runs/{run['id']}/jobs", "jobs")
        records.append(
            {
                "id": run["id"],
                "workflow": run.get("name"),
                "branch": run.get("head_branch"),
                "sha": run.get("head_sha"),
                "attempt": run.get("run_attempt"),
                "status": run.get("status"),
                "conclusion": run.get("conclusion"),
                "created_at": run.get("created_at"),
                "run_started_at": run.get("run_started_at"),
                "pr": (run.get("pull_requests") or [{}])[0].get("number"),
                "jobs": [
                    {
                        "name": j.get("name"),
                        "status": j.get("status"),
                        "conclusion": j.get("conclusion"),
                        "created_at": j.get("created_at"),
                        "started_at": j.get("started_at"),
                        "completed_at": j.get("completed_at"),
                        "labels": j.get("labels") or [],
                        "os": classify_os(j.get("labels") or []),
                    }
                    for j in jobs
                ],
            }
        )
        if index % 20 == 0:
            print(f"  {index}/{len(selected)}", file=sys.stderr)

    snapshot = {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "repo": slug,
        "query": {"since": args.since, "until": args.until, "max_prs": args.max_prs, "branch": args.branch},
        "runs": records,
    }
    out = Path(args.out)
    out.write_text(json.dumps(snapshot, indent=2) + "\n")
    print(f"wrote {out} — {len(records)} runs, {len(branches)} branches", file=sys.stderr)
    return 0


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------

@dataclass
class AttemptMetrics:
    branch: str
    sha: str
    workflows: list[str] = field(default_factory=list)
    first_blocking_green: float | None = None
    merge_ready: float | None = None
    all_blocking_succeeded: bool = False
    complete: bool = True
    macos_queue_max: float | None = None
    windows_vindex_exec: float | None = None
    # Longest single job execution per workflow, within this attempt. Wall
    # clock for the workflow, not its runner-minute cost, because the claims
    # it scores are about how long a gate makes a pull request wait.
    workflow_exec_max: dict[str, float] = field(default_factory=dict)
    superseded_exec: float = 0.0
    queued_by_os: dict[str, float] = field(default_factory=dict)
    executed_by_os: dict[str, float] = field(default_factory=dict)


def run_bounds(run: dict) -> tuple[datetime | None, datetime | None]:
    created = [parse_ts(j["created_at"]) for j in run["jobs"]]
    done = [parse_ts(j["completed_at"]) for j in run["jobs"]]
    created = [c for c in created if c]
    done = [d for d in done if d]
    return (min(created) if created else None, max(done) if done else None)


def analyse(
    snapshot: dict,
    informational: Sequence[str],
    workflow_filter: Sequence[str] | None,
    require_complete: bool,
) -> list[AttemptMetrics]:
    runs = snapshot["runs"]
    if workflow_filter:
        wanted = set(workflow_filter)
        runs = [r for r in runs if r["workflow"] in wanted]

    by_attempt: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for run in runs:
        by_attempt[(run["branch"] or "?", run["sha"] or "?")].append(run)

    # t0 per attempt, and the ordering of attempts within a branch — needed
    # to decide which runs were superseded by a later push.
    t0: dict[tuple[str, str], datetime] = {}
    for key, group in by_attempt.items():
        starts = [run_bounds(r)[0] for r in group]
        starts = [s for s in starts if s]
        if starts:
            t0[key] = min(starts)

    by_branch: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key in by_attempt:
        if key in t0:
            by_branch[key[0]].append(key)
    for branch in by_branch:
        by_branch[branch].sort(key=lambda k: t0[k])

    results: list[AttemptMetrics] = []
    for key, group in sorted(by_attempt.items(), key=lambda kv: t0.get(kv[0], datetime.max.replace(tzinfo=timezone.utc))):
        branch, sha = key
        if key not in t0:
            continue
        start = t0[key]
        metrics = AttemptMetrics(branch=branch, sha=sha, workflows=sorted({r["workflow"] or "?" for r in group}))

        # When did the NEXT push on this branch happen? Anything still
        # executing after that moment is work on a superseded commit.
        siblings = by_branch[branch]
        position = siblings.index(key)
        next_start = t0[siblings[position + 1]] if position + 1 < len(siblings) else None

        blocking_done: list[float] = []
        blocking_all_ok = True
        saw_blocking = False

        for run in group:
            is_informational = (run["workflow"] or "") in informational
            _, finished = run_bounds(run)
            if run["status"] != "completed":
                metrics.complete = False

            for job in run["jobs"]:
                created = parse_ts(job["created_at"])
                started = parse_ts(job["started_at"])
                completed = parse_ts(job["completed_at"])
                queued = seconds_between(created, started)
                executed = seconds_between(started, completed)
                os_name = job["os"]

                if queued is not None:
                    metrics.queued_by_os[os_name] = metrics.queued_by_os.get(os_name, 0.0) + queued
                    if os_name == "macos":
                        metrics.macos_queue_max = max(metrics.macos_queue_max or 0.0, queued)
                if executed is not None:
                    metrics.executed_by_os[os_name] = metrics.executed_by_os.get(os_name, 0.0) + executed
                    if next_start is not None and completed is not None and completed > next_start:
                        # Only the portion spent after the superseding push
                        # is waste; the rest was legitimate work at the time.
                        overlap = seconds_between(max(started or next_start, next_start), completed)
                        metrics.superseded_exec += overlap or 0.0
                    # SUCCEEDED runs only. A run cancelled by the
                    # supersede rule stops mid-suite, so counting it would
                    # let more cancellation masquerade as a faster gate —
                    # the metric would be measuring c1, not the gate.
                    if run["conclusion"] == TERMINAL_SUCCESS:
                        workflow = run["workflow"] or "?"
                        metrics.workflow_exec_max[workflow] = max(
                            metrics.workflow_exec_max.get(workflow, 0.0), executed)
                    # E1's forecast names this OS-scoped special case, so it
                    # stays; workflow_exec_max is the general form and the one
                    # new forecasts should use.
                    if run["workflow"] == "larql-vindex" and os_name == "windows":
                        metrics.windows_vindex_exec = max(metrics.windows_vindex_exec or 0.0, executed)

            if is_informational:
                continue
            saw_blocking = True
            if run["conclusion"] == TERMINAL_SUCCESS and finished is not None:
                blocking_done.append((finished - start).total_seconds())
            else:
                blocking_all_ok = False

        if saw_blocking and blocking_done:
            metrics.first_blocking_green = min(blocking_done)
        metrics.all_blocking_succeeded = saw_blocking and blocking_all_ok
        if metrics.all_blocking_succeeded and blocking_done:
            metrics.merge_ready = max(blocking_done)

        if require_complete and not metrics.complete:
            continue
        results.append(metrics)

    return results


def aggregate(rows: Sequence[AttemptMetrics], percentiles: Sequence[int]) -> dict[str, Any]:
    def series(pick) -> list[float]:
        return [v for v in (pick(r) for r in rows) if v is not None]

    out: dict[str, Any] = {"attempts": len(rows), "merge_ready_attempts": sum(1 for r in rows if r.merge_ready is not None)}
    for label, pick in (
        ("first_blocking_green", lambda r: r.first_blocking_green),
        ("merge_ready", lambda r: r.merge_ready),
        ("macos_queue_max", lambda r: r.macos_queue_max),
        ("windows_vindex_exec", lambda r: r.windows_vindex_exec),
    ):
        values = series(pick)
        out[label] = {f"p{p}": percentile(values, p) for p in percentiles}
        out[label]["n"] = len(values)

    # Composition, as a scoreable number rather than only a printed line.
    # A per-attempt runner-minute figure is intensive in the SAMPLE SIZE but
    # not in the MIX: if a later window triggers the expensive macOS gate on
    # half as many pull requests, macOS minutes fall without anything having
    # been made faster. E1's P3 tried to catch that with an extensive total
    # and could not. This is the metric that can.
    out["workflow_share"] = {}
    if rows:
        present: dict[str, int] = defaultdict(int)
        for row in rows:
            for workflow in row.workflows:
                present[workflow] += 1
        out["workflow_share"] = {w: c / len(rows) for w, c in present.items()}

    workflows = sorted({w for r in rows for w in r.workflow_exec_max})
    out["workflow_exec_max"] = {}
    for workflow in workflows:
        values = [r.workflow_exec_max[workflow] for r in rows if workflow in r.workflow_exec_max]
        entry = {f"p{p}": percentile(values, p) for p in percentiles}
        entry["n"] = len(values)
        out["workflow_exec_max"][workflow] = entry

    out["superseded_exec_total"] = sum(r.superseded_exec for r in rows)
    out["superseded_exec_mean"] = statistics.fmean([r.superseded_exec for r in rows]) if rows else None
    for bucket, attr in (("queued_by_os", "queued_by_os"), ("executed_by_os", "executed_by_os")):
        totals: dict[str, float] = defaultdict(float)
        for row in rows:
            for os_name, value in getattr(row, attr).items():
                totals[os_name] += value
        out[bucket] = dict(totals)
        # The intensive companion. Derived from the same totals so the two
        # cannot disagree, and defined only when there is a sample to
        # divide by.
        out[f"{bucket}_per_attempt"] = (
            {os_name: value / len(rows) for os_name, value in totals.items()} if rows else {}
        )
    return out


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def delta_str(before: float | None, after: float | None) -> str:
    if before is None or after is None or before == 0:
        return "-"
    return f"{(after - before) / before * 100:+.0f}%"


def print_table(before: dict | None, after: dict, percentiles: Sequence[int]) -> None:
    header = f"{'metric':<26}{'before':>12}{'after':>12}{'delta':>10}"
    if before is None:
        header = f"{'metric':<26}{'value':>12}"
    print(header)
    print("-" * len(header))

    def row(label: str, b: float | None, a: float | None, fmt=fmt_duration) -> None:
        if before is None:
            print(f"{label:<26}{fmt(a):>12}")
        else:
            print(f"{label:<26}{fmt(b):>12}{fmt(a):>12}{delta_str(b, a):>10}")

    for key, label in (
        ("first_blocking_green", "first green"),
        ("merge_ready", "merge-ready"),
        ("macos_queue_max", "macOS queue (max/attempt)"),
        ("windows_vindex_exec", "vindex windows exec"),
    ):
        for p in percentiles:
            row(f"{label} p{p}", (before or {}).get(key, {}).get(f"p{p}"), after[key].get(f"p{p}"))

    row("superseded exec (mean)", (before or {}).get("superseded_exec_mean"), after.get("superseded_exec_mean"))
    row("superseded exec (total)", (before or {}).get("superseded_exec_total"), after.get("superseded_exec_total"), fmt_minutes)

    print()
    # Totals first, then the same quantities per attempt. Only the second
    # block is comparable when the two periods hold different numbers of
    # pull requests, which is why both are printed side by side and never
    # one without the other.
    for suffix, title_suffix in (("", " (TOTAL — comparable only at equal n)"),
                                 (PER_ATTEMPT_SUFFIX, " per attempt")):
        for bucket, title in (("queued_by_os", "queued runner-minutes"),
                              ("executed_by_os", "executed runner-minutes")):
            key = f"{bucket}{suffix}"
            print(f"{title}{title_suffix}:")
            names = sorted(set((before or {}).get(key, {})) | set(after.get(key, {})),
                           key=lambda n: OS_ORDER.index(n) if n in OS_ORDER else 99)
            for name in names:
                row(f"  {name}", (before or {}).get(key, {}).get(name), after.get(key, {}).get(name), fmt_minutes)
            print()


def report(args: argparse.Namespace) -> int:
    percentiles = [int(p) for p in args.percentiles.split(",")]
    informational = tuple(w.strip() for w in args.informational.split(",") if w.strip())
    workflows = [w.strip() for w in args.workflow.split(",")] if args.workflow else None

    after_snap = json.loads(Path(args.after).read_text())
    after_rows = analyse(after_snap, informational, workflows, args.require_complete)
    after_agg = aggregate(after_rows, percentiles)

    before_agg = None
    before_rows: list[AttemptMetrics] = []
    if args.before:
        before_snap = json.loads(Path(args.before).read_text())
        before_rows = analyse(before_snap, informational, workflows, args.require_complete)
        before_agg = aggregate(before_rows, percentiles)

    print(f"informational (non-blocking): {', '.join(informational) or 'none'}")
    if workflows:
        print(f"workflow subset: {', '.join(workflows)}")
    print(f"attempts: before={len(before_rows)} after={len(after_rows)}")
    print()
    print_table(before_agg, after_agg, percentiles)

    # Composition, so a delta can never be read without seeing whether the
    # two periods were made of comparable pull requests.
    print("workflow composition (attempts containing each workflow):")
    for label, rows in (("before", before_rows), ("after", after_rows)):
        if not rows:
            continue
        counts: dict[str, int] = defaultdict(int)
        for r in rows:
            for w in r.workflows:
                counts[w] += 1
        summary = ", ".join(f"{w}:{c}" for w, c in sorted(counts.items(), key=lambda kv: -kv[1]))
        print(f"  {label}: {summary}")

    if args.rows:
        print()
        print(f"{'branch':<34}{'sha':<10}{'first':>9}{'ready':>9}{'macosQ':>9}{'super':>9}")
        for r in (before_rows + after_rows):
            print(
                f"{r.branch[:33]:<34}{(r.sha or '')[:8]:<10}"
                f"{fmt_duration(r.first_blocking_green):>9}{fmt_duration(r.merge_ready):>9}"
                f"{fmt_duration(r.macos_queue_max):>9}{fmt_duration(r.superseded_exec):>9}"
            )

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({"before": before_agg, "after": after_agg}, indent=2) + "\n")
    return 0


# --------------------------------------------------------------------------
# adjudication — score a frozen forecast, per mechanism
# --------------------------------------------------------------------------

# Everything the adjudicator compares is in MINUTES. The aggregates carry
# seconds internally; converting once, here, keeps the frozen thresholds in
# one unit and out of reach of a scoring-time reinterpretation.
# The mechanism name reserved for "the tranche as a whole".
SYSTEM_MECHANISM = "system"

VERDICT_HELD = "HELD"
VERDICT_PARTIAL = "PARTIAL"
VERDICT_FALSIFIED = "FALSIFIED"
VERDICT_INVALID = "INVALID COMPARISON"
VERDICT_NO_DATA = "NO DATA"
# Distinct from NO DATA: the metric exists on both sides and is perfectly
# well defined, but comparing it across samples of different size answers a
# question about sample size rather than about the intervention.
VERDICT_UNSCOREABLE = "UNSCOREABLE"


@dataclass(frozen=True)
class Sample:
    """A flat metric table plus the sample size that produced it.

    The attempt count travels WITH the metrics because the parity check is
    not optional: a scorer that has to remember to look it up separately is
    a scorer that will one day forget.
    """

    metrics: dict[str, float | None]
    attempts: int

    def get(self, name: str) -> float | None:
        return self.metrics.get(name)

OPERATORS = {
    "lt": lambda a, b: a < b,
    "lte": lambda a, b: a <= b,
    "gt": lambda a, b: a > b,
    "gte": lambda a, b: a >= b,
}


def flatten_metrics(agg: dict) -> dict[str, float | None]:
    """Aggregate -> flat metric table, all values in minutes."""
    if not agg:
        return {}
    out: dict[str, float | None] = {}
    for key in ("first_blocking_green", "merge_ready", "macos_queue_max", "windows_vindex_exec"):
        for stat, value in (agg.get(key) or {}).items():
            if stat == "n":
                continue
            out[f"{key}.{stat}"] = None if value is None else value / SECONDS_PER_MINUTE
    # A share is a fraction, not a duration: it is NOT divided by
    # SECONDS_PER_MINUTE, so a threshold on it reads as a share.
    for workflow, share in (agg.get("workflow_share") or {}).items():
        out[f"workflow_share.{workflow}"] = share
    for workflow, stats in (agg.get("workflow_exec_max") or {}).items():
        for stat, value in stats.items():
            if stat == "n":
                continue
            out[f"workflow_exec_max.{workflow}.{stat}"] = None if value is None else value / SECONDS_PER_MINUTE
    for key in ("superseded_exec_total", "superseded_exec_mean"):
        value = agg.get(key)
        out[key] = None if value is None else value / SECONDS_PER_MINUTE
    for bucket in EXTENSIVE_OS_BUCKETS:
        for os_name, value in (agg.get(f"{bucket}_by_os") or {}).items():
            out[f"{bucket}.{os_name}"] = value / SECONDS_PER_MINUTE
        for os_name, value in (agg.get(f"{bucket}_by_os_per_attempt") or {}).items():
            out[f"{bucket}{PER_ATTEMPT_SUFFIX}.{os_name}"] = value / SECONDS_PER_MINUTE
    # merge_ready.p50 is the common shorthand; keep the full path too.
    return out


def evaluate_conditions(conditions: dict, after: float | None, before: float | None) -> bool | None:
    """True/False, or None when the data cannot answer the question."""
    if not conditions:
        return None
    for field, tests in conditions.items():
        if field == "after":
            value = after
        elif field in ("delta_pct", "abs_delta_pct"):
            if before in (None, 0) or after is None:
                return None
            value = (after - before) / before * 100.0
            # A two-sided claim needs a two-sided condition. Conditions are
            # ANDed, so `delta_pct` cannot express "more than 10% away in
            # EITHER direction" — writing `lt: -10` silently drops the
            # upper half, and a +20% move comes back neither HELD nor
            # FALSIFIED. E2-D's D3 was written that way and is the reason
            # this field exists.
            if field == "abs_delta_pct":
                value = abs(value)
        else:
            raise ValueError(f"unknown condition field: {field}")
        if value is None:
            return None
        for op, threshold in tests.items():
            if op not in OPERATORS:
                raise ValueError(f"unknown operator: {op}")
            if not OPERATORS[op](value, threshold):
                return False
    return True


def sample_size_bias_pct(before: Sample, after: Sample) -> float | None:
    """The movement an EXTENSIVE metric shows from sample size alone.

    Exactly (n_after / n_before - 1) for a metric that is flat per attempt.
    Reported so a reader can see how much of a delta was bought and paid
    for before the intervention was considered.
    """
    if not before.attempts:
        return None
    return (after.attempts - before.attempts) / before.attempts * 100.0


def score_prediction(pred: dict, before: Sample, after: Sample) -> dict:
    metric = pred["metric"]
    kind = metric_kind(metric)
    b, a = before.get(metric), after.get(metric)
    held = evaluate_conditions(pred.get("held_if", {}), a, b)
    falsified = evaluate_conditions(pred.get("falsified_if", {}), a, b)
    bias = sample_size_bias_pct(before, after)
    unequal = before.attempts != after.attempts
    if a is None:
        verdict = VERDICT_NO_DATA
    elif kind == METRIC_EXTENSIVE and unequal:
        # Refuse rather than report. See METRIC_EXTENSIVE above: this is the
        # E1 C2 defect, and the only correct answer is to name it.
        verdict = VERDICT_UNSCOREABLE
    elif falsified:
        verdict = VERDICT_FALSIFIED
    elif held:
        verdict = VERDICT_HELD
    else:
        verdict = VERDICT_PARTIAL
    delta = None if (b in (None, 0) or a is None) else (a - b) / b * 100.0
    scored = {"id": pred["id"], "mechanism": pred["mechanism"], "kind": pred.get("kind", "effect"),
              "metric": metric, "metric_kind": kind, "before": b, "after": a, "delta_pct": delta,
              "verdict": verdict, "claim": pred.get("claim", "")}
    if verdict == VERDICT_UNSCOREABLE:
        companion = intensive_companion(metric)
        remedy = f"Score {companion} instead." if companion else (
            "This metric has no intensive companion; compare equal-sized samples.")
        scored["unscoreable_because"] = (
            f"{metric} is {METRIC_EXTENSIVE}; n_before={before.attempts} n_after={after.attempts} "
            f"contributes {bias:+.1f}% before any effect. {remedy}"
        )
        scored["intensive_companion"] = companion
    return scored


def mechanism_verdict(scored: Sequence[dict]) -> str:
    # An unscoreable prediction is not a failed one. If the mechanism's
    # validity gate cannot be evaluated, nothing downstream of it can be
    # either, and saying so beats inventing a verdict.
    if any(s["verdict"] == VERDICT_UNSCOREABLE for s in scored):
        return VERDICT_UNSCOREABLE
    validity = [s for s in scored if s["kind"] == "validity"]
    if any(s["verdict"] in (VERDICT_FALSIFIED, VERDICT_PARTIAL) for s in validity):
        return VERDICT_INVALID
    effects = [s for s in scored if s["kind"] == "effect"]
    if not effects or all(s["verdict"] == VERDICT_NO_DATA for s in effects):
        return VERDICT_NO_DATA
    verdicts = {s["verdict"] for s in effects if s["verdict"] != VERDICT_NO_DATA}
    if verdicts == {VERDICT_HELD}:
        return VERDICT_HELD
    if verdicts == {VERDICT_FALSIFIED}:
        return VERDICT_FALSIFIED
    return VERDICT_PARTIAL


def adjudicate(args: argparse.Namespace) -> int:
    forecast = json.loads(Path(args.forecast).read_text())
    percentiles = [int(p) for p in args.percentiles.split(",")]
    informational = tuple(w.strip() for w in args.informational.split(",") if w.strip())

    before_rows = analyse(json.loads(Path(args.before).read_text()), informational, None, args.require_complete)
    after_rows = analyse(json.loads(Path(args.after).read_text()), informational, None, args.require_complete)
    before = Sample(flatten_metrics(aggregate(before_rows, percentiles)), len(before_rows))
    after = Sample(flatten_metrics(aggregate(after_rows, percentiles)), len(after_rows))

    scored = [score_prediction(p, before, after) for p in forecast["predictions"]]
    by_mech: dict[str, list[dict]] = defaultdict(list)
    for s in scored:
        by_mech[s["mechanism"]].append(s)

    names = forecast.get("mechanisms", {})
    # Forecast order, with the whole-tranche mechanism last: a system
    # verdict reads as a summary, and a summary printed first invites
    # skipping the per-mechanism causal information underneath it.
    declared = list(forecast.get("mechanisms", {}))
    ranked = [m for m in declared if m in by_mech and m != SYSTEM_MECHANISM]
    ranked += [m for m in by_mech if m not in declared and m != SYSTEM_MECHANISM]
    if SYSTEM_MECHANISM in by_mech:
        ranked.append(SYSTEM_MECHANISM)
    order = ranked

    experiment = forecast.get("experiment", "?")
    print(f"{experiment} adjudication — before n={before.attempts} attempts, after n={after.attempts} attempts")
    bias = sample_size_bias_pct(before, after)
    if bias is not None and before.attempts != after.attempts:
        print(f"  sample sizes differ: any EXTENSIVE metric carries {bias:+.1f}% from that alone,")
        print(f"  so extensive predictions are refused rather than scored")
    print()
    results = {}
    for mech in order:
        entries = by_mech[mech]
        verdict = mechanism_verdict(entries)
        results[mech] = verdict
        label = "SYSTEM" if mech == SYSTEM_MECHANISM else mech.upper()
        print(f"{label}  {names.get(mech, '')}")
        for s in entries:
            delta = "-" if s["delta_pct"] is None else f"{s['delta_pct']:+.0f}%"
            places = 1 if metric_unit(s["metric"]) else 3
            b = "-" if s["before"] is None else f"{s['before']:.{places}f}"
            a = "-" if s["after"] is None else f"{s['after']:.{places}f}"
            kind = " (validity)" if s["kind"] == "validity" else ""
            unit = metric_unit(s["metric"])
            print(f"  {s['id']}{kind}  {s['metric']}: {b} -> {a}{unit}  ({delta})   {s['verdict']}")
            if s.get("unscoreable_because"):
                print(f"       {s['unscoreable_because']}")
        print(f"  VERDICT: {verdict}")
        if verdict == VERDICT_INVALID:
            print("  the validity gate failed: this mechanism's effect predictions are NOT interpreted")
        if verdict == VERDICT_UNSCOREABLE:
            print("  this mechanism's forecast names a metric these two samples cannot compare")
        print()

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"experiment": experiment, "mechanisms": results, "predictions": scored,
             "before_attempts": before.attempts, "after_attempts": after.attempts,
             "sample_size_bias_pct": bias}, indent=2) + "\n")
    return 0


# --------------------------------------------------------------------------
# selftest — the metric definitions, on synthetic runs with known answers
# --------------------------------------------------------------------------

def _job(created: str, started: str, completed: str, labels: list[str], conclusion: str = TERMINAL_SUCCESS) -> dict:
    return {
        "name": "j", "status": "completed", "conclusion": conclusion,
        "created_at": created, "started_at": started, "completed_at": completed,
        "labels": labels, "os": classify_os(labels),
    }


def _run(workflow: str, sha: str, jobs: list[dict], conclusion: str = TERMINAL_SUCCESS, branch: str = "pr-1") -> dict:
    return {
        "id": 1, "workflow": workflow, "branch": branch, "sha": sha, "attempt": 1,
        "status": "completed", "conclusion": conclusion,
        "created_at": jobs[0]["created_at"], "run_started_at": jobs[0]["started_at"],
        "pr": 1, "jobs": jobs,
    }


def selftest(_args: argparse.Namespace) -> int:
    failures: list[str] = []

    def check(name: str, got: Any, want: Any) -> None:
        if got != want:
            failures.append(f"{name}: got {got!r}, want {want!r}")
        else:
            print(f"  ok  {name} == {want!r}")

    T = "2026-09-06T10:{:02d}:00Z".format

    # 1. queue vs execution are separated, and macOS queue is picked out.
    snap = {"runs": [_run("quality", "aaa", [
        _job(T(0), T(20), T(21), ["macos-14"]),      # 20m queue, 1m exec
        _job(T(0), T(1), T(11), ["ubuntu-latest"]),  # 1m queue, 10m exec
    ])]}
    rows = analyse(snap, DEFAULT_INFORMATIONAL, None, True)
    check("macos queue max", rows[0].macos_queue_max, 20 * 60.0)
    check("queued linux", rows[0].queued_by_os["linux"], 60.0)
    check("executed macos", rows[0].executed_by_os["macos"], 60.0)

    # 1b. the per-attempt companion is the total divided by the attempt
    #     count — checked on two attempts so the divisor is not 1.
    two = {"runs": [
        _run("quality", "aaa", [_job(T(0), T(20), T(21), ["macos-14"])], branch="pr-1"),
        _run("quality", "bbb", [_job(T(30), T(40), T(50), ["macos-14"])], branch="pr-2"),
    ]}
    agg2 = aggregate(analyse(two, DEFAULT_INFORMATIONAL, None, True), DEFAULT_PERCENTILES)
    check("two attempts aggregated", agg2["attempts"], 2)
    check("queued macos total", agg2["queued_by_os"]["macos"], 30 * 60.0)
    check("queued macos per attempt", agg2["queued_by_os_per_attempt"]["macos"], 15 * 60.0)
    check("per-attempt is empty with no attempts",
          aggregate([], DEFAULT_PERCENTILES)["queued_by_os_per_attempt"], {})

    # 2. merge-ready spans to the LAST blocking success; the informational
    #    workflow is excluded even when it finishes last and fails.
    snap = {"runs": [
        _run("larql-core", "aaa", [_job(T(0), T(1), T(6), ["ubuntu-latest"])]),
        _run("quality", "aaa", [_job(T(0), T(1), T(15), ["ubuntu-latest"])]),
        _run("mutants", "aaa", [_job(T(0), T(1), T(45), ["ubuntu-latest"], "failure")], conclusion="failure"),
    ]}
    rows = analyse(snap, DEFAULT_INFORMATIONAL, None, True)
    check("first blocking green", rows[0].first_blocking_green, 6 * 60.0)
    check("merge-ready", rows[0].merge_ready, 15 * 60.0)
    check("all blocking ok", rows[0].all_blocking_succeeded, True)

    # 2b. control: a FAILING blocking run must leave merge-ready undefined.
    snap["runs"][1] = _run("quality", "aaa", [_job(T(0), T(1), T(15), ["ubuntu-latest"], "failure")], conclusion="failure")
    rows = analyse(snap, DEFAULT_INFORMATIONAL, None, True)
    check("merge-ready when a gate fails", rows[0].merge_ready, None)

    # 3. superseded execution counts only the part after the next push.
    #    sha aaa runs 00:00->00:30; sha bbb is pushed at 00:10 => 20m wasted.
    snap = {"runs": [
        _run("larql-core", "aaa", [_job(T(0), T(0), T(30), ["ubuntu-latest"])]),
        _run("larql-core", "bbb", [_job(T(10), T(10), T(25), ["ubuntu-latest"])]),
    ]}
    rows = analyse(snap, DEFAULT_INFORMATIONAL, None, True)
    by_sha = {r.sha: r for r in rows}
    check("superseded exec (old sha)", by_sha["aaa"].superseded_exec, 20 * 60.0)
    check("superseded exec (newest sha)", by_sha["bbb"].superseded_exec, 0.0)

    # 3b. control: cancel the old run at the push and the waste goes to zero.
    snap["runs"][0] = _run("larql-core", "aaa", [_job(T(0), T(0), T(10), ["ubuntu-latest"], TERMINAL_CANCELLED)], conclusion=TERMINAL_CANCELLED)
    rows = analyse(snap, DEFAULT_INFORMATIONAL, None, True)
    check("superseded exec after cancellation", {r.sha: r for r in rows}["aaa"].superseded_exec, 0.0)

    # 4. adjudication: each verdict must be reachable, or the adjudicator is
    #    not an instrument. Synthetic metric tables, same shape the real
    #    forecast scores against.
    p_effect = {"id": "PX", "mechanism": "c2", "kind": "effect", "metric": "queued.macos",
                "held_if": {"delta_pct": {"lte": -30}}, "falsified_if": {"delta_pct": {"gt": -20}}}
    p_valid = {"id": "PV", "mechanism": "c2", "kind": "validity", "metric": "executed.macos",
               "held_if": {"delta_pct": {"gte": -10}}, "falsified_if": {"delta_pct": {"lt": -20}}}
    N = 20      # equal sample sizes, so the parity guard stays out of the way
    base = Sample({"queued.macos": 4526.6, "executed.macos": 3034.9}, N)

    good = Sample({"queued.macos": 2263.3, "executed.macos": 2882.2}, N)   # -50% queue, -5% exec
    check("effect HELD", score_prediction(p_effect, base, good)["verdict"], VERDICT_HELD)
    check("validity HELD", score_prediction(p_valid, base, good)["verdict"], VERDICT_HELD)
    check("c2 verdict when both hold",
          mechanism_verdict([score_prediction(p_effect, base, good), score_prediction(p_valid, base, good)]),
          VERDICT_HELD)

    flat = Sample({"queued.macos": 4300.0, "executed.macos": 2950.0}, N)   # -5% queue
    check("effect FALSIFIED", score_prediction(p_effect, base, flat)["verdict"], VERDICT_FALSIFIED)

    # At EQUAL n, a queue drop alongside an execution collapse is a real
    # composition change, and the validity gate must veto the effect.
    shifted = Sample({"queued.macos": 1000.0, "executed.macos": 1500.0}, N)  # -78% queue, -51% exec
    check("effect looks HELD under a composition shift",
          score_prediction(p_effect, base, shifted)["verdict"], VERDICT_HELD)
    check("validity FALSIFIED under a composition shift",
          score_prediction(p_valid, base, shifted)["verdict"], VERDICT_FALSIFIED)
    check("c2 verdict is vetoed by validity",
          mechanism_verdict([score_prediction(p_effect, base, shifted), score_prediction(p_valid, base, shifted)]),
          VERDICT_INVALID)

    check("missing metric yields NO DATA", score_prediction(p_effect, base, Sample({}, N))["verdict"], VERDICT_NO_DATA)

    # 4f. two-sided validity. `delta_pct` ANDs its operators, so a claim
    #     of the form "within +/- X%" cannot be falsified on both sides by
    #     one delta_pct clause; abs_delta_pct exists for exactly that, and
    #     the control checks BOTH signs and the passing middle.
    p_two_sided = {"id": "PT", "mechanism": "m", "kind": "validity", "metric": "merge_ready.p50",
                   "held_if": {"delta_pct": {"gte": -10, "lte": 10}},
                   "falsified_if": {"abs_delta_pct": {"gt": 10}}}
    b2 = Sample({"merge_ready.p50": 100.0}, N)
    check("two-sided validity HOLDS in the middle",
          score_prediction(p_two_sided, b2, Sample({"merge_ready.p50": 105.0}, N))["verdict"], VERDICT_HELD)
    check("two-sided validity FALSIFIED below",
          score_prediction(p_two_sided, b2, Sample({"merge_ready.p50": 85.0}, N))["verdict"], VERDICT_FALSIFIED)
    check("two-sided validity FALSIFIED above",
          score_prediction(p_two_sided, b2, Sample({"merge_ready.p50": 120.0}, N))["verdict"], VERDICT_FALSIFIED)

    #     ...and the bug it was written for: a ONE-sided falsifier lets a
    #     +20% move escape as PARTIAL, which is what D3 originally did.
    p_one_sided = dict(p_two_sided, id="PO", falsified_if={"delta_pct": {"lt": -10}})
    check("one-sided falsifier misses the other direction",
          score_prediction(p_one_sided, b2, Sample({"merge_ready.p50": 120.0}, N))["verdict"], VERDICT_PARTIAL)

    # 4a. the E1 C2 defect, as a regression test. The SAME numbers that score
    #     HELD/FALSIFIED above must be refused once the samples differ in
    #     size, because then they are a statement about n and not about the
    #     intervention. Both directions are checked: the guard must fire
    #     here and must NOT fire at equal n (above), or it is not a gate.
    thin = Sample({"queued.macos": 1000.0, "executed.macos": 1500.0}, N // 2)
    check("extensive effect is UNSCOREABLE at unequal n",
          score_prediction(p_effect, base, thin)["verdict"], VERDICT_UNSCOREABLE)
    check("extensive validity is UNSCOREABLE at unequal n",
          score_prediction(p_valid, base, thin)["verdict"], VERDICT_UNSCOREABLE)
    check("mechanism is UNSCOREABLE, not INVALID, when the metric is the problem",
          mechanism_verdict([score_prediction(p_effect, base, thin), score_prediction(p_valid, base, thin)]),
          VERDICT_UNSCOREABLE)
    check("the refusal names the per-attempt metric to use instead",
          "queued_per_attempt.macos" in score_prediction(p_effect, base, thin)["unscoreable_because"], True)
    check("the refusal names superseded_exec_MEAN, which exists",
          intensive_companion("superseded_exec_total"), "superseded_exec_mean")

    # E1's real numbers: 12 attempts against 33 is -63.6% before any effect.
    check("E1's sample-size bias is reported",
          round(sample_size_bias_pct(Sample({}, 33), Sample({}, 12)), 1), -63.6)

    # 4c. an INTENSIVE metric is scoreable across unlike samples — that is
    #     the whole point of having the companion.
    p_intensive = {"id": "PI", "mechanism": "c2", "kind": "effect",
                   "metric": "queued_per_attempt.macos",
                   "held_if": {"delta_pct": {"lte": -30}}, "falsified_if": {"delta_pct": {"gt": -20}}}
    b_int = Sample({"queued_per_attempt.macos": 137.2}, 33)
    a_int = Sample({"queued_per_attempt.macos": 80.1}, 12)
    check("intensive metric scores across unlike samples",
          score_prediction(p_intensive, b_int, a_int)["verdict"], VERDICT_HELD)

    # 4d. declaration fate: every metric the instrument produces has a kind,
    #     and the per-attempt companions exist for every extensive bucket.
    produced = set(flatten_metrics(aggregate([
        AttemptMetrics(branch="b", sha="s", workflows=[PROBE_WORKFLOW],
                       queued_by_os={"macos": 1.0}, executed_by_os={"macos": 1.0},
                       workflow_exec_max={PROBE_WORKFLOW: 60.0})
    ], DEFAULT_PERCENTILES)))
    check("every extensive metric has a per-attempt companion",
          all(f"{b}{PER_ATTEMPT_SUFFIX}.macos" in produced
              for b in EXTENSIVE_OS_BUCKETS), True)
    check("metric_kind classifies every produced metric",
          {metric_kind(m) for m in produced} <= {METRIC_EXTENSIVE, METRIC_INTENSIVE}, True)
    #     Declaration fate: every extensive metric the instrument produces
    #     must have a companion, that companion must itself be produced, and
    #     it must be intensive. Without this a refusal can name a metric
    #     that does not exist — as it did for superseded_exec_total.
    extensive_produced = [m for m in produced if metric_kind(m) == METRIC_EXTENSIVE]
    check("there are extensive metrics to check", len(extensive_produced) > 0, True)
    check("every extensive metric has a companion",
          all(intensive_companion(m) for m in extensive_produced), True)
    check("every companion is itself produced",
          all(intensive_companion(m) in produced for m in extensive_produced), True)
    check("every companion is intensive",
          {metric_kind(intensive_companion(m)) for m in extensive_produced}, {METRIC_INTENSIVE})
    check("queued.macos is extensive", metric_kind("queued.macos"), METRIC_EXTENSIVE)
    check("queued_per_attempt.macos is intensive",
          metric_kind("queued_per_attempt.macos"), METRIC_INTENSIVE)
    check("merge_ready.p50 is intensive", metric_kind("merge_ready.p50"), METRIC_INTENSIVE)
    check("superseded_exec_total is extensive",
          metric_kind("superseded_exec_total"), METRIC_EXTENSIVE)
    check("superseded_exec_mean is intensive",
          metric_kind("superseded_exec_mean"), METRIC_INTENSIVE)
    check("workflow_exec_max is intensive",
          metric_kind("workflow_exec_max.larql-compute-metal.p50"), METRIC_INTENSIVE)
    check("workflow_share is intensive",
          metric_kind("workflow_share.larql-compute-metal"), METRIC_INTENSIVE)
    check("metric_shape abstracts the workflow out of an exec metric",
          metric_shape("workflow_exec_max.larql-compute-metal.p50"),
          f"workflow_exec_max.{PROBE_WORKFLOW}.p50")
    check("metric_shape abstracts the workflow out of a share",
          metric_shape("workflow_share.bench-regress"), f"workflow_share.{PROBE_WORKFLOW}")
    check("metric_shape survives a dot in a workflow name",
          metric_shape("workflow_exec_max.a.b.p95"), f"workflow_exec_max.{PROBE_WORKFLOW}.p95")
    check("metric_workflow recovers the name", metric_workflow("workflow_exec_max.a.b.p95"), "a.b")
    check("metric_shape leaves an unscoped metric alone",
          metric_shape("merge_ready.p50"), "merge_ready.p50")
    check("metric_workflow is None for an unscoped metric",
          metric_workflow("merge_ready.p50"), None)

    # 4e. workflow_exec_max picks the LONGEST job in the workflow, per
    #     attempt, and is a percentile across attempts — so it survives a
    #     change in how many pull requests the window holds.
    wf_snap = {"runs": [
        _run("larql-compute-metal", "aaa", [
            _job(T(0), T(1), T(11), ["macos-14"]),
            _job(T(0), T(1), T(31), ["macos-14"]),
        ], branch="pr-1"),
        _run("larql-compute-metal", "bbb", [_job(T(0), T(1), T(21), ["macos-14"])], branch="pr-2"),
    ]}
    wf_agg = aggregate(analyse(wf_snap, DEFAULT_INFORMATIONAL, None, True), DEFAULT_PERCENTILES)
    check("workflow_exec_max takes the longest job",
          wf_agg["workflow_exec_max"]["larql-compute-metal"]["p50"], 25 * 60.0)
    check("workflow_exec_max is flattened in minutes",
          flatten_metrics(wf_agg)["workflow_exec_max.larql-compute-metal.p50"], 25.0)

    #     Share counts attempts the workflow RAN on, success or not, because
    #     it describes what triggered, not what passed.
    share_snap = {"runs": wf_snap["runs"] + [
        _run("quality", "ccc", [_job(T(0), T(1), T(6), ["ubuntu-latest"])], branch="pr-3")]}
    share_agg = aggregate(analyse(share_snap, DEFAULT_INFORMATIONAL, None, True), DEFAULT_PERCENTILES)
    check("workflow_share counts attempts, not runs",
          round(share_agg["workflow_share"]["larql-compute-metal"], 4), round(2 / 3, 4))
    check("workflow_share is a fraction, not minutes",
          round(flatten_metrics(share_agg)["workflow_share.larql-compute-metal"], 4), round(2 / 3, 4))

    #     Control: a CANCELLED run must not enter the metric, or cancelling
    #     more runs would read as a faster gate.
    wf_snap["runs"][1] = _run("larql-compute-metal", "bbb",
                              [_job(T(0), T(1), T(4), ["macos-14"], TERMINAL_CANCELLED)],
                              conclusion=TERMINAL_CANCELLED, branch="pr-2")
    wf_agg2 = aggregate(analyse(wf_snap, DEFAULT_INFORMATIONAL, None, True), DEFAULT_PERCENTILES)
    check("a cancelled run is excluded from workflow_exec_max",
          wf_agg2["workflow_exec_max"]["larql-compute-metal"]["n"], 1)
    check("the surviving successful attempt sets the value",
          wf_agg2["workflow_exec_max"]["larql-compute-metal"]["p50"], 30 * 60.0)

    # 4b. every shipped forecast must be scoreable: every condition parses
    #     and every metric name is one flatten_metrics actually produces.
    #     Forecasts frozen from E2 onward must additionally name only
    #     INTENSIVE metrics, so the E1 C2 defect cannot recur by copy-paste.
    for forecast_path, intensive_only in (
        (FORECAST_DIR / "E1-forecast.json", False),
        (FORECAST_DIR / "E2-contract.json", True),
    ):
        if not forecast_path.exists():
            continue
        forecast = json.loads(forecast_path.read_text())
        label = forecast.get("experiment", forecast_path.stem)
        snapshot_workflows: set[str] = set()
        for snapshot_key in ("primary_snapshot", "metal_gate_snapshot", "baseline_snapshot"):
            snapshot_path = forecast.get("baseline", {}).get(snapshot_key) or forecast.get(
                "frozen_against", {}).get(snapshot_key)
            if snapshot_path and (REPO_ROOT / snapshot_path).exists():
                snap = json.loads((REPO_ROOT / snapshot_path).read_text())
                snapshot_workflows |= {r.get("workflow") for r in snap.get("runs", [])}
        for pred in forecast["predictions"]:
            # SHAPE, because a workflow-scoped name exists only once that
            # workflow is in a snapshot...
            check(f"{label} {pred['id']} metric is produced",
                  metric_shape(pred["metric"]) in produced, True)
            # ...and then the workflow itself must be one the baseline
            # actually contains, or the prediction is unscoreable by name.
            workflow = metric_workflow(pred["metric"])
            if workflow is not None and snapshot_workflows:
                check(f"{label} {pred['id']} names a workflow the baseline contains",
                      workflow in snapshot_workflows, True)
            if intensive_only:
                check(f"{label} {pred['id']} names an intensive metric",
                      metric_kind(pred["metric"]), METRIC_INTENSIVE)
            for conditions in (pred.get("held_if", {}), pred.get("falsified_if", {})):
                evaluate_conditions(conditions, 1.0, 2.0)   # raises on an unknown field/operator
        check(f"{label} conditions all parse", True, True)

    # 5. percentile is defined at n == 1 and interpolates at n == 2.
    check("percentile n=1", percentile([7.0], 95), 7.0)
    check("percentile n=2 p50", percentile([0.0, 10.0], 50), 5.0)

    print()
    if failures:
        for f in failures:
            print(f"  FAIL {f}")
        print(f"{len(failures)} failing check(s)")
        return 1
    print("selftest: all checks passed")
    return 0


# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_collect = sub.add_parser("collect", help="snapshot pull_request runs and their jobs")
    p_collect.add_argument("--out", required=True, help="snapshot path to write")
    p_collect.add_argument("--since", help="YYYY-MM-DD lower bound on run creation")
    p_collect.add_argument("--until", help="YYYY-MM-DD upper bound on run creation")
    p_collect.add_argument("--branch", help="restrict to one head branch")
    p_collect.add_argument("--max-prs", type=int, default=20, help="most recent N head branches (default 20)")
    p_collect.set_defaults(func=collect)

    p_report = sub.add_parser("report", help="score a snapshot, optionally against a baseline")
    p_report.add_argument("--after", required=True, help="snapshot to score")
    p_report.add_argument("--before", help="baseline snapshot")
    p_report.add_argument("--informational", default=",".join(DEFAULT_INFORMATIONAL),
                          help="comma-separated non-blocking workflow names")
    p_report.add_argument("--workflow", help="comma-separated workflow subset")
    p_report.add_argument("--percentiles", default=",".join(str(p) for p in DEFAULT_PERCENTILES))
    p_report.add_argument("--require-complete", action="store_true",
                          help="drop attempts with a run still in flight")
    p_report.add_argument("--rows", action="store_true", help="print one row per attempt")
    p_report.add_argument("--json-out", help="also write the aggregates as JSON")
    p_report.set_defaults(func=report)

    p_adj = sub.add_parser("adjudicate", help="score a frozen forecast, per mechanism")
    p_adj.add_argument("--forecast", required=True, help="frozen forecast JSON")
    p_adj.add_argument("--before", required=True, help="baseline snapshot")
    p_adj.add_argument("--after", required=True, help="snapshot to score")
    p_adj.add_argument("--informational", default=",".join(DEFAULT_INFORMATIONAL))
    p_adj.add_argument("--percentiles", default=",".join(str(p) for p in DEFAULT_PERCENTILES))
    p_adj.add_argument("--require-complete", action="store_true")
    p_adj.add_argument("--json-out", help="also write the verdicts as JSON")
    p_adj.set_defaults(func=adjudicate)

    p_self = sub.add_parser("selftest", help="check the metric definitions against known answers")
    p_self.set_defaults(func=selftest)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
