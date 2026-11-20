"""`hf-f64-accuracy/2`: is LARQL an f32 implementation as accurate as transformers' own?

`hf-f64-accuracy/1` (f64_accuracy.py) is unchanged and is still reported in
full. v1 gates every position at an absolute 1e-3 against the audited f64
oracle (A64). transformers' own f32 run (A32) fails that rule: on Gemma 3 270M
I_rep it is 4.25e-3 from A64. So v1 measures how sensitive the model and input
are to f32 rounding as well as how faithful LARQL is.

v2 gates each LARQL backend X in {C, D} against A64 *relative to A32's own
error*. It uses PREC-DIAG-1's R1 reading (`analysis.r1`, constants unchanged,
frozen for 270M on 2026-11-10) plus one guard:

- R1 must read `comparable`: median and p99 of r_X = d(X, A64) / max(d(A32, A64), eps)
  at most 2 and 4 on every input, and fewer than 3 material positions across
  all inputs. A material position has d(X, A64) > 1e-3 while d(A32, A64) <= 1e-3.
- Tail guard: no position with d(X, A64) > 1e-3 and r_X > 4. R1 alone would accept
  one large excursion at a position where A32 is also above 1e-3.

The test passes only when both C and D pass. The v1 verdict, A32 agreement
and the joint distribution of absolute error against ratio (including the
positions at the eps floor) are reported and never gated.

Evidence class. The rule was adopted after Gemma 3 1B failed v1, so the
report names how its runs relate to the rule:
- `prospective`: the freeze was written after v2 was fixed. It names this test
  (`values.admission_test`) and binds this script (`files.script.f64_accuracy_v2`).
- `retrospective`: the runs were frozen before v2. A write-once addendum binds the
  freeze, the sealed runs (`SHA256SUMS`), this script, `analysis.py` and the v1
  report, whose verdict is carried into the v2 report.
- `prospective-reporting-repair`: prospective measurement with a post-run
  reporting repair. The runs were made under a prospective freeze, but the
  report tooling that freeze binds could not read them, and was corrected
  after the runs and before any verdict. A write-once repair addendum binds the
  freeze, the sealed runs, the failed report attempts, the frozen and the
  corrected tooling hashes, and the reviewed PR and merge commit. The freeze is
  never rewritten and is never said to have bound the corrected tooling.
Anything else is `invalid`.

Scope. The test reads E1 (A32, A64) and E2 (C, D) only (`REPORT_EXPERIMENTS`).
E3 captures are outside it and are never looked at, so a freeze for an E1/E2
experiment need not name capture positions.

Controls (`control`): a control experiment (N1: `Dauto`, production CPU from
the `auto` source) is judged by the same per-backend rule against the same
A32/A64. The freeze must bind this script and pre-register the expected
outcome (`values.control_expectation`). The report says whether the outcome
matched the expectation. A control never admits anything, and its outcome
never changes a constant.

usage:
  python3 -I f64_accuracy_v2.py addendum --freeze-sha256 SHA --root RUNS --v1-report V1.json --out ADDENDUM.json
  python3 -I f64_accuracy_v2.py repair-addendum --freeze FREEZE.json --freeze-sha256 SHA --root RUNS \
         --failed-attempt FILE [--failed-attempt FILE ...] --pr N --merge-commit SHA --statement TEXT \
         --out ADDENDUM.json
  python3 -I f64_accuracy_v2.py run --freeze FREEZE.json --freeze-sha256 SHA --root RUNS --out REPORT.json
         [--addendum ADDENDUM.json --addendum-sha256 SHA]
  python3 -I f64_accuracy_v2.py control N1 --freeze FREEZE.json --freeze-sha256 SHA --root RUNS --out REPORT.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prec_diag import analysis  # noqa: E402
from prec_diag import evidence as ev  # noqa: E402
from prec_diag import experiment as ex  # noqa: E402
from prec_diag import f64_accuracy as v1  # noqa: E402
from prec_diag import freeze as fz  # noqa: E402

NAME = v1.NAME
VERSION = 2
TEST = f"{NAME}/{VERSION}"
BACKENDS, INPUTS = v1.BACKENDS, v1.INPUTS
GATE = analysis.GATE                        # "material" absolute d, = v1's gate
EPSILON = analysis.EPSILON                  # ratio denominator floor
TAIL_RATIO = analysis.R1_COMPARABLE_P99     # tail guard reuses R1's p99 limit
COMPARABLE = "comparable"
PASS, FAIL, INVALID = v1.PASS, v1.FAIL, v1.INVALID
PROSPECTIVE, RETROSPECTIVE, CONTROL = "prospective", "retrospective", "control"
REPAIRED = "prospective-reporting-repair"
REPAIRED_MEANING = ("prospective measurement with post-run reporting repair: the runs were made under a "
                    "prospective freeze naming this test; the report tooling was corrected after the runs and "
                    "before any verdict; this addendum, not the freeze, binds the tooling that read them")
#: The experiments this test reads: A32 and A64 from E1, C and D from E2.
REPORT_EXPERIMENTS = ("E1", "E2")
#: Control arm -> the gated backend whose family (tie tolerance) it shares.
CONTROL_FAMILY = {"Dauto": "D"}
ADDENDUM_SCHEMA = "larql.hf-f64-accuracy-2.addendum/1"
REPAIR_SCHEMA = "larql.hf-f64-accuracy-2.reporting-repair/1"
#: The report tooling a repair addendum binds, by freeze label.
TOOLING_LABELS = ("script.f64_accuracy_v2", "script.experiment", "script.analysis")
SCRIPT_LABEL = "script.f64_accuracy_v2"
SUMS = "SHA256SUMS"
#: Bin edges of the reported joint distribution (absolute d, ratio r).
D_EDGES = (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, math.inf)
R_EDGES = (0.0, 0.5, 1.0, 2.0, 4.0, math.inf)


def own_sha256() -> str:
    return ev.sha256_file(Path(__file__).resolve())


def analysis_sha256() -> str:
    return ev.sha256_file(Path(analysis.__file__).resolve())


def tooling_sha256() -> dict:
    """The report tooling running now, by freeze label."""
    return {"script.f64_accuracy_v2": own_sha256(), "script.experiment": ev.sha256_file(Path(ex.__file__).resolve()),
            "script.analysis": analysis_sha256()}


def frozen_tooling_sha256(manifest: dict) -> dict:
    """The report tooling the freeze binds, by freeze label (None if unbound)."""
    return {label: manifest.get("files", {}).get(label, {}).get("sha256") for label in TOOLING_LABELS}


# ------------------------------------------------------------------ gate

def ratios(dx: np.ndarray, da: np.ndarray) -> np.ndarray:
    return analysis.ratio(dx, da)[0]


def tail_positions(dx: dict, da: dict) -> list[dict]:
    """Positions with d(X, A64) > GATE and r_X > TAIL_RATIO, over all inputs."""
    out = []
    for i in dx:
        x, a = np.asarray(dx[i], dtype=np.float64), np.asarray(da[i], dtype=np.float64)
        r = ratios(x, a)
        for p in np.flatnonzero((x > GATE) & (r > TAIL_RATIO)):
            out.append({"input": i, "position": int(p), "d_x": float(x[p]), "d_a32": float(a[p]), "r": float(r[p])})
    return out


def joint(dx: np.ndarray, da: np.ndarray) -> dict:
    """Counts of positions by (absolute d bin, ratio bin), and the eps-floor positions."""
    x, a = np.asarray(dx, dtype=np.float64), np.asarray(da, dtype=np.float64)
    r = ratios(x, a)
    counts, _, _ = np.histogram2d(x, r, bins=(np.asarray(D_EDGES), np.asarray(R_EDGES)))
    floor = a < EPSILON
    return {"d_edges": [str(e) for e in D_EDGES], "r_edges": [str(e) for e in R_EDGES],
            "counts": counts.astype(int).tolist(),
            "eps_floor": {"positions": int(floor.sum()),
                          "d_x_max": float(x[floor].max()) if floor.any() else None,
                          "r_max": float(r[floor].max()) if floor.any() else None}}


def backend_verdict(dx: dict, da: dict) -> dict:
    """`dx[i]`, `da[i]`: per-position d(X, A64), d(A32, A64) on input i."""
    r1 = analysis.r1(dx, da, True)
    tail = tail_positions(dx, da)
    passed = r1["verdict"] == COMPARABLE and not tail
    return {"verdict": PASS if passed else FAIL, "r1": r1, "tail_guard": tail,
            "joint": {i: joint(dx[i], da[i]) for i in dx}}


def verdict(accuracy: dict, a32_accuracy: dict, agreement: dict) -> dict:
    """`accuracy[b][i]` = d(b, A64); `a32_accuracy[i]` = d(A32, A64);
    `agreement[b][i]` = d(b, A32). Only v2 is gated; v1 is reported beside it."""
    per = {b: backend_verdict(accuracy[b], a32_accuracy) for b in BACKENDS}
    failing = [b for b in BACKENDS if per[b]["verdict"] != PASS]
    return {"verdict": FAIL if failing else PASS, "failing": failing, "backends": per,
            "v1_reported_not_gated": v1.verdict(accuracy, agreement)}


# --------------------------------------------------------- evidence class

def make_addendum(freeze_sha256: str, root: Path, v1_report: Path) -> dict:
    """The retrospective binding for runs frozen before v2."""
    rep = json.loads(Path(v1_report).read_text())
    if rep.get("test") != f"{v1.NAME}/{v1.VERSION}":
        raise ev.EvidenceError(f"{v1_report}: not a {v1.NAME}/{v1.VERSION} report")
    if rep.get("freeze_sha256") != freeze_sha256:
        raise ev.EvidenceError(f"{v1_report}: names another freeze")
    sums = Path(root) / SUMS
    if not sums.is_file():
        raise ev.EvidenceError(f"{sums}: missing; seal the runs first")
    return {"schema": ADDENDUM_SCHEMA, "test": TEST, "evidence_class": RETROSPECTIVE,
            "freeze_sha256": freeze_sha256, "runs_sha256sums": ev.sha256_file(sums),
            "script_sha256": own_sha256(), "analysis_sha256": analysis_sha256(),
            "v1_report": {"path": str(v1_report), "sha256": ev.sha256_file(Path(v1_report)),
                          "verdict": rep["verdict"], "failing": rep.get("failing", [])}}


def write_addendum(out: Path, addendum: dict) -> str:
    out = Path(out)
    if out.exists():
        raise ev.EvidenceError(f"{out} exists: an addendum is written once")
    data = ev.canonical_json(addendum) + b"\n"
    ev.write_atomic(out, data)
    out.chmod(0o444)
    return ev.sha256_bytes(data)


def evidence_class(manifest: dict, freeze_sha256: str, root: Path,
                   addendum_path: Path | None, addendum_sha256: str | None) -> tuple[str | None, dict, list[str]]:
    """(class, details, problems). Prospective comes from the freeze itself;
    retrospective needs a verified addendum."""
    if addendum_path is None:
        problems = []
        if manifest["values"].get("admission_test") != TEST:
            problems.append(f"freeze does not name {TEST}: runs frozen before v2 need a retrospective addendum")
        bound = manifest.get("files", {}).get(SCRIPT_LABEL, {}).get("sha256")
        if bound != own_sha256():
            problems.append(f"freeze does not bind this script ({SCRIPT_LABEL})")
        return (None, {}, problems) if problems else (PROSPECTIVE, {}, [])
    raw = Path(addendum_path).read_bytes()
    if addendum_sha256 is None or ev.sha256_bytes(raw) != addendum_sha256:
        return None, {}, [f"{addendum_path}: sha256 is not the reviewed digest"]
    a = json.loads(raw)
    if a.get("schema") == REPAIR_SCHEMA:
        return repair_class(manifest, freeze_sha256, root, a, addendum_sha256)
    checks = {
        "schema": (a.get("schema"), ADDENDUM_SCHEMA),
        "evidence_class": (a.get("evidence_class"), RETROSPECTIVE),
        "freeze_sha256": (a.get("freeze_sha256"), freeze_sha256),
        "runs_sha256sums": (a.get("runs_sha256sums"),
                            ev.sha256_file(Path(root) / SUMS) if (Path(root) / SUMS).is_file() else None),
        "script_sha256": (a.get("script_sha256"), own_sha256()),
        "analysis_sha256": (a.get("analysis_sha256"), analysis_sha256()),
    }
    problems = [f"addendum {k} differs" for k, (got, want) in checks.items() if got != want]
    v1r = a.get("v1_report", {})
    p = Path(v1r.get("path", ""))
    if not p.is_file() or ev.sha256_file(p) != v1r.get("sha256"):
        problems.append("addendum v1_report missing or changed")
    return (None, {}, problems) if problems else (RETROSPECTIVE, {"addendum_sha256": addendum_sha256,
                                                                "v1_report": v1r}, [])


def runs_sums_sha256(root: Path) -> str | None:
    sums = Path(root) / SUMS
    return ev.sha256_file(sums) if sums.is_file() else None


def make_repair_addendum(manifest_path: Path, freeze_sha256: str, root: Path, failed_attempts: list[Path],
                         pr: int, merge_commit: str, statement: str) -> dict:
    """The binding for a prospective experiment whose frozen report tooling
    failed before any verdict and was then corrected (see the module doc)."""
    manifest = ex.load_freeze(manifest_path, freeze_sha256)
    if manifest["values"].get("admission_test") != TEST:
        raise ev.EvidenceError(f"the freeze does not name {TEST}: a reporting repair is for a prospective freeze")
    frozen, corrected = frozen_tooling_sha256(manifest), tooling_sha256()
    if frozen == corrected:
        raise ev.EvidenceError("the freeze binds the tooling running now: there is nothing to repair")
    if runs_sums_sha256(root) is None:
        raise ev.EvidenceError(f"{Path(root) / SUMS}: missing; seal the runs first")
    if not failed_attempts:
        raise ev.EvidenceError("a reporting repair names the failed report attempt(s) it follows")
    missing = [str(p) for p in failed_attempts if not Path(p).is_file()]
    if missing:
        raise ev.EvidenceError(f"failed attempt records missing: {missing}")
    if not (isinstance(pr, int) and pr > 0) or len(merge_commit) != 40 or \
            any(c not in "0123456789abcdef" for c in merge_commit) or not statement.strip():
        raise ev.EvidenceError("a reporting repair names its reviewed PR, its full merge commit and its scope")
    return {"schema": REPAIR_SCHEMA, "test": TEST, "evidence_class": REPAIRED, "meaning": REPAIRED_MEANING,
            "freeze_sha256": freeze_sha256, "runs_sha256sums": runs_sums_sha256(root),
            "frozen_tooling": frozen, "corrected_tooling": corrected,
            "failed_attempts": [{"path": str(p), "sha256": ev.sha256_file(Path(p))} for p in failed_attempts],
            "repair": {"pr": pr, "merge_commit": merge_commit, "statement": statement}}


def repair_class(manifest: dict, freeze_sha256: str, root: Path, a: dict,
                 addendum_sha256: str) -> tuple[str | None, dict, list[str]]:
    checks = {
        "test": (a.get("test"), TEST),
        "evidence_class": (a.get("evidence_class"), REPAIRED),
        "freeze_sha256": (a.get("freeze_sha256"), freeze_sha256),
        "runs_sha256sums": (a.get("runs_sha256sums"), runs_sums_sha256(root)),
        "frozen_tooling": (a.get("frozen_tooling"), frozen_tooling_sha256(manifest)),
        "corrected_tooling": (a.get("corrected_tooling"), tooling_sha256()),
    }
    problems = [f"addendum {k} differs" for k, (got, want) in checks.items() if got != want]
    if manifest["values"].get("admission_test") != TEST:
        problems.append(f"freeze does not name {TEST}")
    attempts = a.get("failed_attempts") or []
    if not attempts:
        problems.append("addendum names no failed report attempt")
    for f in attempts:
        p = Path(f.get("path", ""))
        if not p.is_file() or ev.sha256_file(p) != f.get("sha256"):
            problems.append(f"addendum failed attempt {p} missing or changed")
    if problems:
        return None, {}, problems
    return REPAIRED, {"addendum_sha256": addendum_sha256,
                      "reporting_repair": {k: a[k] for k in ("meaning", "repair", "failed_attempts",
                                                              "frozen_tooling", "corrected_tooling")}}, []


# ------------------------------------------------------------- evaluation

def evaluate(runs: dict, expected: dict, vocab: int) -> dict:
    keys = [(b, i) for b in BACKENDS for i in INPUTS] + [(r, i) for r in ("A32", "A64") for i in INPUTS]
    problems = []
    for k in keys:
        if k not in runs:
            problems.append(f"{k}: not run")
            continue
        adm = ev.admit(runs[k], expected.get(k), None)
        if not adm.admitted:
            problems.append(f"{k}: not admitted: {adm.reasons}")
    for i in INPUTS:
        if ("A64", i) in runs:
            problems += v1.audit_problems(runs[("A64", i)])
    if problems:
        return {"verdict": INVALID, "problems": problems}
    acc, agr, a32acc = {b: {} for b in BACKENDS}, {b: {} for b in BACKENDS}, {}
    for i in INPUTS:
        a64 = analysis.load_logits(runs[("A64", i)], vocab)
        a32 = analysis.load_logits(runs[("A32", i)], vocab)
        a32acc[i] = analysis.position_metrics(a64, a32, analysis.FAMILY["A32"])["d"]
        for b in BACKENDS:
            x = analysis.load_logits(runs[(b, i)], vocab)
            acc[b][i] = analysis.position_metrics(a64, x, analysis.FAMILY[b])["d"]
            agr[b][i] = analysis.position_metrics(a32, x, analysis.FAMILY[b])["d"]
    return verdict(acc, a32acc, agr)


def control_problems(manifest: dict, experiment: str) -> list[str]:
    problems = []
    exp = manifest["values"].get("control_expectation", {}).get(experiment, {})
    if exp.get("expected") not in (PASS, FAIL):
        problems.append(f"freeze pre-registers no expected outcome for control {experiment}")
    if manifest.get("files", {}).get(SCRIPT_LABEL, {}).get("sha256") != own_sha256():
        problems.append(f"freeze does not bind this script ({SCRIPT_LABEL})")
    return problems


def evaluate_control(manifest: dict, freeze_sha256: str, root: Path, experiment: str) -> dict:
    runs, expected = ex.runs_and_identities(manifest, freeze_sha256, root, REPORT_EXPERIMENTS)
    cruns, cexpected = ex.control_runs(manifest, freeze_sha256, root, experiment)
    arms = sorted({a for a, _ in cruns})
    keys = [(r, i) for r in ("A32", "A64") for i in INPUTS]
    problems = []
    for k in keys:
        adm = ev.admit(runs[k], expected.get(k), None)
        if not adm.admitted:
            problems.append(f"{k}: not admitted: {adm.reasons}")
    for k, d in cruns.items():
        adm = ev.admit(d, cexpected.get(k), None)
        if not adm.admitted:
            problems.append(f"{k}: not admitted: {adm.reasons}")
    for i in INPUTS:
        problems += v1.audit_problems(runs[("A64", i)])
    if problems:
        return {"verdict": INVALID, "problems": problems}
    vocab = ex.vocab_of(manifest)
    dx, da = {a: {} for a in arms}, {}
    for i in INPUTS:
        a64 = analysis.load_logits(runs[("A64", i)], vocab)
        da[i] = analysis.position_metrics(a64, analysis.load_logits(runs[("A32", i)], vocab), analysis.FAMILY["A32"])["d"]
        for a in arms:
            fam = analysis.FAMILY[CONTROL_FAMILY[a]]
            dx[a][i] = analysis.position_metrics(a64, analysis.load_logits(cruns[(a, i)], vocab), fam)["d"]
    per = {a: backend_verdict(dx[a], da) for a in arms}
    for a in arms:
        per[a]["d_x_a64_max"] = {i: float(np.max(dx[a][i])) for i in INPUTS}
    failing = [a for a in arms if per[a]["verdict"] != PASS]
    return {"verdict": FAIL if failing else PASS, "failing": failing, "arms": per}


def run_control(manifest_path: Path, freeze_sha256: str, root: Path, out: Path, experiment: str) -> dict:
    out = Path(out)
    if out.exists():
        raise ev.EvidenceError(f"{out} exists: a report is written once")
    manifest = ex.load_freeze(manifest_path, freeze_sha256)
    problems = fz.preflight(manifest_path, freeze_sha256) + control_problems(manifest, experiment)
    result = {"verdict": INVALID, "problems": problems} if problems else \
        evaluate_control(manifest, freeze_sha256, root, experiment)
    exp = manifest["values"].get("control_expectation", {}).get(experiment, {})
    observed = result["verdict"]
    report = {"test": TEST, "evidence_class": CONTROL, "admits": False, "control": experiment,
              "expectation": exp, "observed": observed,
              "matches_expectation": None if observed == INVALID else observed == exp.get("expected"),
              "scope": v1.scope(manifest), "freeze_sha256": freeze_sha256, "root": str(root),
              "script_sha256": own_sha256(), "analysis_sha256": analysis_sha256(), **result}
    ev.write_atomic(out, ev.canonical_json(report) + b"\n")
    return report


def run(manifest_path: Path, freeze_sha256: str, root: Path, out: Path,
        addendum: Path | None = None, addendum_sha256: str | None = None) -> dict:
    out = Path(out)
    if out.exists():
        raise ev.EvidenceError(f"{out} exists: a report is written once")
    manifest = ex.load_freeze(manifest_path, freeze_sha256)
    cls, details, problems = evidence_class(manifest, freeze_sha256, root, addendum, addendum_sha256)
    problems = fz.preflight(manifest_path, freeze_sha256) + problems
    if problems:
        result = {"verdict": INVALID, "problems": problems}
    else:
        runs, expected = ex.runs_and_identities(manifest, freeze_sha256, root, REPORT_EXPERIMENTS)
        result = evaluate(runs, expected, ex.vocab_of(manifest))
        if cls == RETROSPECTIVE and result.get("verdict") != INVALID:
            result["v1_sealed_report"] = details["v1_report"]
    report = {"test": TEST, "evidence_class": cls, "reference": "A64 (audited f64)", "baseline": "A32 (transformers f32)",
              "gate": {"rule": "R1 comparable (PREC-DIAG-1) and tail guard, per backend; C and D both pass",
                       "r1_median_max": analysis.R1_COMPARABLE_MEDIAN, "r1_p99_max": analysis.R1_COMPARABLE_P99,
                       "material_d": GATE, "material_positions_below": analysis.R1_MATERIAL_COUNT,
                       "tail_ratio_max": TAIL_RATIO, "epsilon": EPSILON},
              "reported_not_gated": ["v1 verdict", "agreement_vs_A32", "joint distribution of d and r"],
              "scope": v1.scope(manifest), "freeze_sha256": freeze_sha256, "root": str(root),
              "script_sha256": own_sha256(), "analysis_sha256": analysis_sha256(), **details, **result}
    ev.write_atomic(out, ev.canonical_json(report) + b"\n")
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a_ = sub.add_parser("addendum", help="bind runs frozen before v2 (retrospective)")
    a_.add_argument("--freeze-sha256", required=True)
    a_.add_argument("--root", required=True, type=Path)
    a_.add_argument("--v1-report", required=True, type=Path)
    a_.add_argument("--out", required=True, type=Path)
    p_ = sub.add_parser("repair-addendum", help="bind a post-run reporting repair (prospective freeze)")
    p_.add_argument("--freeze", required=True, type=Path)
    p_.add_argument("--freeze-sha256", required=True)
    p_.add_argument("--root", required=True, type=Path)
    p_.add_argument("--failed-attempt", required=True, type=Path, action="append")
    p_.add_argument("--pr", required=True, type=int)
    p_.add_argument("--merge-commit", required=True)
    p_.add_argument("--statement", required=True)
    p_.add_argument("--out", required=True, type=Path)
    r_ = sub.add_parser("run")
    r_.add_argument("--freeze", required=True, type=Path)
    r_.add_argument("--freeze-sha256", required=True)
    r_.add_argument("--root", required=True, type=Path)
    r_.add_argument("--out", required=True, type=Path)
    r_.add_argument("--addendum", type=Path)
    r_.add_argument("--addendum-sha256")
    c_ = sub.add_parser("control", help="judge a control experiment (never admits)")
    c_.add_argument("experiment", choices=ex.CONTROL_EXPERIMENTS)
    c_.add_argument("--freeze", required=True, type=Path)
    c_.add_argument("--freeze-sha256", required=True)
    c_.add_argument("--root", required=True, type=Path)
    c_.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "control":
        r = run_control(a.freeze, a.freeze_sha256, a.root, a.out, a.experiment)
        print(r["test"], CONTROL, a.experiment, "observed", r["observed"], "expected",
              r["expectation"].get("expected"), "matches", r["matches_expectation"], *r.get("problems", [])[:5])
        return 0 if r["matches_expectation"] is not None else 2
    if a.cmd == "repair-addendum":
        print("repair addendum sha256", write_addendum(a.out, make_repair_addendum(
            a.freeze, a.freeze_sha256, a.root, a.failed_attempt, a.pr, a.merge_commit, a.statement)))
        return 0
    if a.cmd == "addendum":
        print("addendum sha256", write_addendum(a.out, make_addendum(a.freeze_sha256, a.root, a.v1_report)))
        return 0
    report = run(a.freeze, a.freeze_sha256, a.root, a.out, a.addendum, a.addendum_sha256)
    print(report["test"], report["evidence_class"], report["verdict"], *report.get("failing", []),
          *report.get("problems", [])[:5])
    return {PASS: 0, FAIL: 1}.get(report["verdict"], 2)


if __name__ == "__main__":
    sys.exit(main())
