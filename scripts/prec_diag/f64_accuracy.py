"""`hf-f64-accuracy/1`: LARQL's accuracy against the audited transformers f64 oracle.

A separate test from `strict-f32` (harness 2.1.0), which stays as it is. The two
answer different questions:

- `strict-f32` (agreement): does LARQL reproduce transformers' f32 outputs?
- `hf-f64-accuracy/1` (accuracy): how close is LARQL to the f64 reference (A64)?

The gate: for both LARQL backends (C = reference, D = Metal) on all three
1,040-token inputs (I_rep, I_nr1, I_nr2), the full-vocabulary max |Δ logprob|
against A64 is at most 1e-3, the same value as `strict-f32`. The agreement
distance to transformers f32 (A32) is reported beside it and never gated, so a
change that trades one for the other is visible.

A64 is admissible only with its audit passed. The oracle already records a
failed audit as `fault`, which is never admitted. This test also re-reads
`audit.json`: the f64 operations ran (call totals equal expected) and every
checkpoint parameter was preserved.

Scope: Gemma 3 270M at the frozen checkpoint, these inputs, the C and D paths,
and the environment bound by `FREEZE.json`. A pass shows LARQL is within 1e-3
of an f64 implementation, which is itself finite precision. It does not show
that LARQL has no numerical bugs elsewhere.

usage: python3 -I f64_accuracy.py --freeze FREEZE.json --freeze-sha256 SHA --root RUNS --out REPORT.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prec_diag import analysis  # noqa: E402
from prec_diag import evidence as ev  # noqa: E402
from prec_diag import experiment as ex  # noqa: E402
from prec_diag import freeze as fz  # noqa: E402

NAME = "hf-f64-accuracy"
VERSION = 1
GATE = 1e-3
BACKENDS = ("C", "D")
INPUTS = ("I_rep", "I_nr1", "I_nr2")
SCOPE = ("Gemma 3 270M (google/gemma-3-270m-it at the frozen revision); inputs I_rep, I_nr1, I_nr2; "
         "LARQL C (reference) and D (Metal) cached paths; the environment bound by FREEZE.json. "
         "The reference is the audited transformers f64 oracle, itself finite precision.")
PASS, FAIL, INVALID = "pass", "fail", "invalid"


def audit_problems(a64_dir: Path) -> list[str]:
    """The A64 audit, re-read: f64 operations ran and the weights were preserved."""
    p = Path(a64_dir) / "audit.json"
    if not p.is_file():
        return [f"{p}: missing"]
    a = json.loads(p.read_text())
    problems = []
    if a.get("pass") is not True or a.get("problem_count", 1) != 0:
        problems.append(f"{p}: audit did not pass")
    if a.get("call_totals") != a.get("expected_call_totals") or not a.get("call_totals"):
        problems.append(f"{p}: f64 call totals differ from expected")
    params = a.get("parameters", {})
    if params.get("pass") is not True or params.get("parameters_checked") != params.get("parameters"):
        problems.append(f"{p}: checkpoint parameters not all preserved")
    return problems


def verdict(accuracy: dict, agreement: dict) -> dict:
    """`accuracy[b][i]` / `agreement[b][i]`: per-position max |Δ logprob| of
    backend b on input i against A64 / A32. Only accuracy is gated."""
    cells, failing = {}, []
    for b in BACKENDS:
        for i in INPUTS:
            acc = np.asarray(accuracy[b][i], dtype=np.float64)
            over = np.flatnonzero(acc > GATE)
            cells[f"{b}.{i}"] = {"accuracy_vs_A64": analysis.summary(acc), "positions_over_gate": int(over.size),
                                 "agreement_vs_A32": analysis.summary(agreement[b][i])}
            if over.size:
                failing.append(f"{b}.{i}")
    return {"verdict": FAIL if failing else PASS, "failing": failing, "cells": cells}


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
            problems += audit_problems(runs[("A64", i)])
    if problems:
        return {"verdict": INVALID, "problems": problems}
    acc, agr = {b: {} for b in BACKENDS}, {b: {} for b in BACKENDS}
    for i in INPUTS:
        a64 = analysis.load_logits(runs[("A64", i)], vocab)
        a32 = analysis.load_logits(runs[("A32", i)], vocab)
        for b in BACKENDS:
            x = analysis.load_logits(runs[(b, i)], vocab)
            acc[b][i] = analysis.position_metrics(a64, x, analysis.FAMILY[b])["d"]
            agr[b][i] = analysis.position_metrics(a32, x, analysis.FAMILY[b])["d"]
    return verdict(acc, agr)


def run(manifest_path: Path, freeze_sha256: str, root: Path, out: Path) -> dict:
    out = Path(out)
    if out.exists():
        raise ev.EvidenceError(f"{out} exists: a report is written once")
    manifest = ex.load_freeze(manifest_path, freeze_sha256)
    problems = fz.preflight(manifest_path, freeze_sha256)
    if problems:
        result = {"verdict": INVALID, "problems": problems}
    else:
        runs, expected = ex.runs_and_identities(manifest, freeze_sha256, root)
        result = evaluate(runs, expected, ex.vocab_of(manifest))
    report = {"test": f"{NAME}/{VERSION}", "gate_max_abs_logprob": GATE, "reference": "A64 (audited f64)",
              "gated": "accuracy_vs_A64", "reported_not_gated": "agreement_vs_A32", "scope": SCOPE,
              "freeze_sha256": freeze_sha256, "root": str(root), **result}
    ev.write_atomic(out, ev.canonical_json(report) + b"\n")
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--freeze", required=True, type=Path)
    ap.add_argument("--freeze-sha256", required=True)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    report = run(a.freeze, a.freeze_sha256, a.root, a.out)
    print(report["test"], report["verdict"], *report.get("failing", []), *report.get("problems", [])[:5])
    return {PASS: 0, FAIL: 1}.get(report["verdict"], 2)


if __name__ == "__main__":
    sys.exit(main())
