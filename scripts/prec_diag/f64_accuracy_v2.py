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
  corrected hashes of every module the report imports, and the reviewed PR and
  merge commit. Only `experiment.py` and this script may differ from the
  freeze; every other module (analysis, evidence, v0, ...) must be the frozen
  bytes. There is one repair per freeze: the addendum and the repaired report
  live at fixed names beside FREEZE.json, each written once. The freeze is never
  rewritten and is never said to have bound the corrected tooling.
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
         --failed-attempt FILE [--failed-attempt FILE ...] --pr N --merge-commit SHA --statement TEXT
         (writes REPAIR_ADDENDUM beside FREEZE.json; the repaired report must be REPAIR_REPORT there)
  python3 -I f64_accuracy_v2.py run --freeze FREEZE.json --freeze-sha256 SHA --root RUNS --out REPORT.json
         [--addendum ADDENDUM.json --addendum-sha256 SHA]
  python3 -I f64_accuracy_v2.py control N1 --freeze FREEZE.json --freeze-sha256 SHA --root RUNS --out REPORT.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
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
from prec_diag import families as fam  # noqa: E402
from prec_diag import v0  # noqa: E402

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
#: Every module the report imports, by freeze label: a repair addendum binds them all.
TOOLING_LABELS = ("script.analysis", "script.evidence", "script.experiment", "script.f64_accuracy",
                  "script.f64_accuracy_v2", "script.families", "script.freeze", "script.v0")
#: The only modules a reporting repair may change (run discovery and this report).
REPAIRABLE = ("script.experiment", "script.f64_accuracy_v2")
#: One repair per freeze, at fixed names beside FREEZE.json.
REPAIR_ADDENDUM = "hf-f64-accuracy-2.reporting-repair.json"
REPAIR_REPORT = "hf-f64-accuracy-2.reporting-repair.report.json"
HEX40 = re.compile(r"[0-9a-f]{40}")
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
    """Every module the report imports, as running now, by freeze label."""
    files = {"script.analysis": analysis.__file__, "script.evidence": ev.__file__, "script.experiment": ex.__file__,
             "script.f64_accuracy": v1.__file__, "script.f64_accuracy_v2": __file__, "script.families": fam.__file__,
             "script.freeze": fz.__file__, "script.v0": v0.__file__}
    return {label: ev.sha256_file(Path(files[label]).resolve()) for label in TOOLING_LABELS}


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
    try:
        a = json.loads(raw)
    except ValueError:
        return None, {}, [f"{addendum_path}: not JSON"]
    if not isinstance(a, dict):
        return None, {}, [f"{addendum_path}: not a JSON object"]
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


def tooling_problems(frozen: dict, corrected: dict) -> list[str]:
    """Why `corrected` is not a reporting repair of `frozen` (empty if it is)."""
    problems = [f"the freeze does not bind {label}" for label in TOOLING_LABELS if not frozen.get(label)]
    problems += [f"{label} differs from the freeze: a reporting repair may change only {list(REPAIRABLE)}"
                 for label in TOOLING_LABELS if label not in REPAIRABLE and corrected.get(label) != frozen.get(label)]
    if frozen == corrected:
        problems.append("the freeze binds the tooling running now: there is nothing to repair")
    return problems


def repair_fields_problems(failed_attempts, repair) -> list[str]:
    problems = []
    if not isinstance(failed_attempts, list) or not failed_attempts:
        problems.append("a reporting repair names the failed report attempt(s) it follows")
    else:
        for f in failed_attempts:
            p = Path(f["path"]) if isinstance(f, dict) and isinstance(f.get("path"), str) else None
            if p is None or not p.is_absolute() or not p.is_file() or ev.sha256_file(p) != f.get("sha256"):
                problems.append(f"failed attempt {f!r} missing, relative or changed")
    if not isinstance(repair, dict) or not (isinstance(repair.get("pr"), int) and repair["pr"] > 0) or \
            not HEX40.fullmatch(str(repair.get("merge_commit", ""))) or not str(repair.get("statement", "")).strip():
        problems.append("a reporting repair names its reviewed PR, its full merge commit and its scope")
    return problems


def repair_paths(manifest_path: Path) -> tuple[Path, Path]:
    d = Path(manifest_path).resolve().parent
    return d / REPAIR_ADDENDUM, d / REPAIR_REPORT


def repair_location_problems(manifest_path: Path, root: Path) -> list[str]:
    """A repair is anchored where the runs are: the run root must sit beside
    FREEZE.json, so a copied freeze elsewhere has no runs to repair."""
    if Path(root).resolve().parent != Path(manifest_path).resolve().parent:
        return [f"the run root {root} is not beside FREEZE.json: a repair lives with its runs"]
    return []


def make_repair_addendum(manifest_path: Path, freeze_sha256: str, root: Path, failed_attempts: list[Path],
                         pr: int, merge_commit: str, statement: str) -> dict:
    """The binding for a prospective experiment whose frozen report tooling
    failed before any verdict and was then corrected (see the module doc)."""
    manifest = ex.load_freeze(manifest_path, freeze_sha256)
    if manifest["values"].get("admission_test") != TEST:
        raise ev.EvidenceError(f"the freeze does not name {TEST}: a reporting repair is for a prospective freeze")
    frozen, corrected = frozen_tooling_sha256(manifest), tooling_sha256()
    if runs_sums_sha256(root) is None:
        raise ev.EvidenceError(f"{Path(root) / SUMS}: missing; seal the runs first")
    attempts = [{"path": str(Path(p).resolve()), "sha256": ev.sha256_file(Path(p)) if Path(p).is_file() else None}
                for p in failed_attempts]
    repair = {"pr": pr, "merge_commit": merge_commit, "statement": statement}
    problems = tooling_problems(frozen, corrected) + repair_fields_problems(attempts, repair) + \
        repair_location_problems(manifest_path, root)
    if problems:
        raise ev.EvidenceError(f"not a reporting repair: {problems}")
    return {"schema": REPAIR_SCHEMA, "test": TEST, "evidence_class": REPAIRED, "meaning": REPAIRED_MEANING,
            "freeze_sha256": freeze_sha256, "runs_sha256sums": runs_sums_sha256(root),
            "frozen_tooling": frozen, "corrected_tooling": corrected,
            "failed_attempts": attempts, "repair": repair}


def write_repair_addendum(manifest_path: Path, addendum: dict) -> str:
    """Write the one repair addendum this freeze may have, beside it."""
    return write_addendum(repair_paths(manifest_path)[0], addendum)


def repair_class(manifest: dict, freeze_sha256: str, root: Path, a: dict,
                 addendum_sha256: str) -> tuple[str | None, dict, list[str]]:
    checks = {
        "test": (a.get("test"), TEST),
        "evidence_class": (a.get("evidence_class"), REPAIRED),
        "meaning": (a.get("meaning"), REPAIRED_MEANING),
        "freeze_sha256": (a.get("freeze_sha256"), freeze_sha256),
        "runs_sha256sums": (a.get("runs_sha256sums"), runs_sums_sha256(root)),
        "frozen_tooling": (a.get("frozen_tooling"), frozen_tooling_sha256(manifest)),
        "corrected_tooling": (a.get("corrected_tooling"), tooling_sha256()),
    }
    problems = [f"addendum {k} differs" for k, (got, want) in checks.items() if got != want]
    if manifest["values"].get("admission_test") != TEST:
        problems.append(f"freeze does not name {TEST}")
    problems += tooling_problems(frozen_tooling_sha256(manifest), tooling_sha256())
    problems += repair_fields_problems(a.get("failed_attempts"), a.get("repair"))
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
    if cls == REPAIRED and ((Path(addendum).resolve(), out.resolve()) != repair_paths(manifest_path)
                            or repair_location_problems(manifest_path, root)):
        raise ev.EvidenceError(f"a reporting repair is read from {REPAIR_ADDENDUM} and reported to "
                               f"{REPAIR_REPORT}, beside FREEZE.json and its run root: one repair per freeze")
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
    if cls == REPAIRED:
        out.chmod(0o444)
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
        print(repair_paths(a.freeze)[0], "sha256", write_repair_addendum(a.freeze, make_repair_addendum(
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
