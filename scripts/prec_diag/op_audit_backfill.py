"""Diagnostic backfill of A64 audit part 5 (`op_audit.py`) over the families
qualified before it existed: Gemma 3 text and Qwen3 (#897).

DIAGNOSTIC ONLY. This changes and re-evaluates no sealed verdict:
- it reads no sealed run and writes nothing under any experiment directory;
- its forwards are short (at most `MAX_TOKENS` tokens) and throwaway;
- it computes no accuracy metric: the logits are discarded unread.

Each subject is one A64 forward made exactly as `hf_oracle.forward` makes it
(same load, the same patches, the module-boundary hooks, and
`op_audit=True`). The subjects are:
- the tiny random checkpoints of the oracle's own tests
  (`test_prec_diag_hf_oracle*.py`), with their own ids;
- the real checkpoints named on the command line (`--checkpoint NAME=DIR`),
  on `TEXT` cut to at most `MAX_TOKENS` tokens.

The report lists every non-float64 op found, by call site, beside the
existing parts' verdicts on the same forward, every dtype conversion, and
every kernel probe.

A clean result is only worth something if the instrument can see a fault
on the same subject. So each subject also gets a control forward: the same
load and hooks with every A64 patch removed (transformers' modeling file
as shipped, on float64 weights). Its non-float64 sites are reported beside
the module-boundary audit's problem count on that same forward.

usage:
  python3 -I op_audit_backfill.py --out-json REPORT.json --out-md REPORT.md \\
      [--checkpoint NAME=DIR ...]
"""

from __future__ import annotations

import argparse
import datetime
import json
import platform
import sys
import tempfile
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prec_diag import hf_oracle as ho  # noqa: E402
from prec_diag import op_audit as oa  # noqa: E402

import torch  # noqa: E402

#: The longest backfill forward (owner's bound for this diagnostic).
MAX_TOKENS = 16
#: Throwaway text for the real checkpoints. Nothing is measured on its output.
TEXT = "The quick brown fox jumps over the lazy dog near the river bank."
BOUNDARY = {"allowlist": {}, "in_closure_prefixes": {"model": "target", "lm_head": "target"}}
SCHEMA = "larql.prec-diag-1.a64-op-audit-backfill/1"
STATEMENT = ("Diagnostic only. No sealed verdict is changed or re-evaluated: no sealed run is read, no accuracy "
             "metric is computed, and the forwards are throwaway (at most 16 tokens; logits discarded unread).")


def synthetic_subjects(root: Path) -> list[dict]:
    """The oracle tests' tiny random checkpoints and ids."""
    scripts = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(scripts))
    import test_prec_diag_hf_oracle as gemma  # noqa: E402
    import test_prec_diag_hf_oracle_qwen3 as qwen3  # noqa: E402
    return [{"name": "synthetic gemma3_text (test_prec_diag_hf_oracle)", "kind": "synthetic",
             "checkpoint": gemma.make_checkpoint(root / "gemma3"), "ids": list(gemma.IDS)},
            {"name": "synthetic qwen3 (test_prec_diag_hf_oracle_qwen3)", "kind": "synthetic",
             "checkpoint": qwen3.make_checkpoint(root / "qwen3"), "ids": list(qwen3.IDS)}]


def real_subject(name: str, directory: Path) -> dict:
    from tokenizers import Tokenizer
    ids = Tokenizer.from_file(str(directory / "tokenizer.json")).encode(TEXT).ids[:MAX_TOKENS]
    return {"name": name, "kind": "real", "checkpoint": directory, "ids": ids,
            "revision": directory.name}


def control(model, ids: list[int], family: ho.Family) -> dict:
    """The forward with every A64 patch removed: what part 5 and the
    module-boundary audit (part 2) each see of transformers' own f32 sites."""
    audit, ops = ho.A64Audit(model, family), oa.OpDtypeAudit()
    every_patch = frozenset(family.expected_calls(1))
    with ho.SiteCapture(model, None, family), ho.a64_patches(model, audit, every_patch), \
            ho.module_dtype_hooks(model, audit), ops, torch.no_grad():
        model(torch.tensor([ids]), use_cache=False)
    o = ops.finish()
    by_site: dict = {}
    for v in o["violations"]:
        entry = by_site.setdefault(v["site"], {"site": v["site"], "ops": 0,
                                               "code": v["call_stack"][0]["code"] if v["call_stack"] else ""})
        entry["ops"] += v["count"]
    return {"patches_removed": sorted(every_patch), "part_5_pass": o["pass"], "non_f64_ops": o["non_f64_ops"],
            "floating_ops": o["floating_ops"], "non_f64_sites": list(by_site.values()),
            "module_boundary_problems": sum(p.startswith("module ") for p in audit.problems)}


def audit_subject(subject: dict) -> dict:
    """One A64 forward with part 5; the existing parts' verdicts beside it."""
    ckpt = Path(subject["checkpoint"])
    family = ho.Family.of(ckpt)
    modeling = ho.modeling_sha256(family)
    if modeling != family.facts["modeling_sha256"]:
        raise ho.Refused(f"{ckpt}: modeling_{family.facts['modeling']}.py is not the frozen file")
    model, _ = ho.load(ckpt, "A64", BOUNDARY)
    result = ho.forward(model, subject["ids"], "A64", None, None, ckpt, family, op_audit=True)
    del result["logits"]  # never read
    audit = result["records"]["audit"]
    ops = audit.pop("operations")
    existing = {"parts_2_to_4_pass": audit["pass"],
                "module_boundary_problems": sum(p.startswith("module ") for p in audit["problems"]),
                "parameters_pass": audit["parameters"]["pass"], "problem_count": audit["problem_count"],
                "call_totals": audit["call_totals"]}
    return {"name": subject["name"], "kind": subject["kind"], "family": family.name,
            "modeling_sha256": modeling, "revision": subject.get("revision"),
            "checkpoint": str(ckpt).replace(str(Path.home()), "~") if subject["kind"] == "real" else "<temporary>",
            "tokens": len(subject["ids"]), "ids": subject["ids"], "existing_audit": existing, "operations": ops,
            "control_unpatched": control(model, subject["ids"], family)}


def report(subjects: list[dict]) -> dict:
    return {"schema": SCHEMA, "diagnostic": True, "statement": STATEMENT, "issue": "#897",
            "audit_schema": oa.SCHEMA, "date": datetime.date.today().isoformat(),
            "packages": ho.packages(), "threads": ho.thread_state(), "machine": platform.machine(),
            "os": platform.platform(), "subjects": [audit_subject(s) for s in subjects]}


# ----------------------------------------------------------------- markdown

def _cell(text) -> str:
    return str(text).replace("|", "\\|")


def findings(rep: dict) -> list[str]:
    """The backfill's findings, stated from the data."""
    out = ["## Findings", ""]
    for family in sorted({s["family"] for s in rep["subjects"]}):
        subs = [s for s in rep["subjects"] if s["family"] == family]
        clean = all(s["operations"]["pass"] for s in subs)
        faults = sum(s["operations"]["non_f64_ops"] for s in subs)
        casts = sorted({(c["from"], c["to"], c["site"]) for s in subs for c in s["operations"]["casts"]})
        out.append(f"- **{family}** ({len(subs)} subjects): part 5 {'passes' if clean else 'FAILS'} on the A64 "
                   f"forward; {faults} non-float64 op events; every op with floating tensors is classified; "
                   f"{sum(len(s['operations']['probes']) for s in subs)} kernel probes, "
                   f"{sum(not p['pass'] for s in subs for p in s['operations']['probes'])} failed. Dtype conversions: "
                   + ("; ".join(f"{f} -> {t} at `{site}`" for f, t, site in casts) or "none") + ".")
    gap = [s for s in rep["subjects"] if not s["control_unpatched"]["part_5_pass"]
           and s["control_unpatched"]["module_boundary_problems"] == 0]
    out.append(f"- **The gap #897 names, on these families**: with A64's patches removed, part 5 fails on "
               f"{len(gap)} of {len(rep['subjects'])} subjects while the module-boundary audit (part 2) reports 0 "
               "problems on the same forward. The shipped float32 sites (RMSNorm, rotary, eager softmax) cast and "
               "cast back inside one call. A64's existing patches cover exactly these sites, and part 3's call "
               "counts and witnesses already caught a missing patch; part 5 also catches a float32 site that "
               "no patch was written for.")
    gemma = [s for s in rep["subjects"] if s["family"] == "gemma3_text"]
    if gemma and not any("ScaledWordEmbedding" in site["site"] for s in gemma
                         for site in s["control_unpatched"]["non_f64_sites"]):
        out.append("- **A limit of any dtype audit, shown here**: the control also removed Gemma 3's "
                   "scaled-embedding patch, and part 5 recorded no embedding site. transformers builds "
                   "`embed_scale` as a float32 tensor and `.to(float64)` widens it, so the forward multiplies by "
                   "a float64 tensor holding sqrt(hidden) rounded to float32. Its dtype is float64; only its "
                   "value is float32-rounded. Part 3's embedding witness is what catches it, so part 5 adds to "
                   "parts 1-4 and replaces none of them.")
    out.append("- Sealed verdicts are unchanged. This backfill re-evaluates none of them.")
    return out


def markdown(rep: dict) -> str:
    lines = ["# PREC-DIAG A64 operation-level dtype audit: backfill (diagnostic)", "",
             f"**{rep['statement']}**", "",
             f"Issue {rep['issue']}. Audit schema `{rep['audit_schema']}`. Run {rep['date']} on "
             f"{rep['machine']}, torch {rep['packages']['torch']}, transformers {rep['packages']['transformers']}, "
             f"{rep['threads']['torch_num_threads']} threads. The full record is the JSON beside this file.", "",
             "Each subject is one A64 forward made as `hf_oracle.forward` makes it, with the existing patches "
             "and the module-boundary hooks, plus audit part 5 (`op_audit.py`).", "",
             *findings(rep), "",
             "## Summary", "",
             "| subject | family | tokens | ops traced | floating ops | non-f64 ops | violation sites | "
             "unclassified | probes (passed) | casts (non-f64) | part 5 | parts 2-4 / module-boundary problems |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in rep["subjects"]:
        o, e = s["operations"], s["existing_audit"]
        passed = sum(p["pass"] for p in o["probes"])
        non_f64_casts = sum(c["count"] for c in o["casts"] if c["non_f64"])
        lines.append(f"| {_cell(s['name'])} | {s['family']} | {s['tokens']} | {o['ops_traced']} | "
                     f"{o['floating_ops']} | {o['non_f64_ops']} | {o['violation_sites']} | "
                     f"{len(o['unclassified'])} | {o['probes_run']} ({passed}) | "
                     f"{sum(c['count'] for c in o['casts'])} ({non_f64_casts}) | "
                     f"{'pass' if o['pass'] else 'FAIL'} | "
                     f"{'pass' if e['parts_2_to_4_pass'] and e['parameters_pass'] else 'fail'} / "
                     f"{e['module_boundary_problems']} |")
    lines += ["", "## Non-float64 operations, by site", ""]
    any_violation = False
    for s in rep["subjects"]:
        for v in s["operations"]["violations"]:
            any_violation = True
            code = v["call_stack"][0]["code"] if v["call_stack"] else ""
            lines.append(f"- {_cell(s['name'])}: `{v['site']}` {v['op']} {v['role']} {v['dtype']} x{v['count']}: "
                         f"`{_cell(code)}`")
        for name, count in s["operations"]["unclassified"].items():
            any_violation = True
            lines.append(f"- {_cell(s['name'])}: {name} has no precision evidence (x{count})")
    if not any_violation:
        lines.append("None: in every subject, every op read, returned and wrote float64 only, and every op with "
                     "floating tensors was classified.")
    lines += ["", "## Dtype conversions, by site", "",
              "Every conversion the forward made, faulting or not.", "",
              "| subject | site | op | from | to | count | non-f64 |", "|---|---|---|---|---|---|---|"]
    for s in rep["subjects"]:
        for c in s["operations"]["casts"]:
            lines.append(f"| {_cell(s['name'])} | `{c['site']}` | {c['op']} | {c['from']} | {c['to']} | "
                         f"{c['count']} | {'yes' if c['non_f64'] else 'no'} |")
    lines += ["", "## Kernel probes", "",
              "Accumulating ops are probed at every distinct shape and stride the forward used; `error` must be 0 "
              "(exact) or within 1e-12, and the float32 counterfactual must miss.", "",
              "| subject | op | reduction length | comparison | error | f32 counterfactual | pass |",
              "|---|---|---|---|---|---|---|"]
    for s in rep["subjects"]:
        for p in s["operations"]["probes"]:
            lines.append(f"| {_cell(s['name'])} | {p['op']} | {p.get('reduction_length', '')} | "
                         f"{p['comparison']} | {p.get('error', '')} | {p.get('f32_counterfactual_error', '')} | "
                         f"{'pass' if p['pass'] else 'FAIL'} |")
    lines += ["", "## Control: the same forward with every A64 patch removed", "",
              "Not an A64 forward: transformers' modeling file as shipped, on the same float64 weights. It shows "
              "that part 5 sees the families' own float32 sites on these very subjects, and that the "
              "module-boundary audit (part 2) does not.", "",
              "| subject | patches removed | non-f64 ops / floating ops | part 5 | module-boundary problems |",
              "|---|---|---|---|---|"]
    for s in rep["subjects"]:
        c = s["control_unpatched"]
        lines.append(f"| {_cell(s['name'])} | {', '.join(c['patches_removed'])} | "
                     f"{c['non_f64_ops']} / {c['floating_ops']} | {'pass' if c['part_5_pass'] else 'FAIL'} | "
                     f"{c['module_boundary_problems']} |")
    lines += ["", "Non-float64 sites in the control:", ""]
    for s in rep["subjects"]:
        for site in s["control_unpatched"]["non_f64_sites"]:
            lines.append(f"- {_cell(s['name'])}: `{site['site']}` ({site['ops']} op events): "
                         f"`{_cell(site['code'])}`")
    vis = rep["subjects"][0]["operations"]["visibility"] if rep["subjects"] else {}
    lines += ["", "## What the dispatch level sees, and what it cannot", "", "Sees:", ""]
    lines += [f"- {v}" for v in vis.get("sees", [])]
    lines += ["", "Does not see:", ""]
    lines += [f"- {v}" for v in vis.get("does_not_see", [])]
    lines += ["", "## Subjects", ""]
    for s in rep["subjects"]:
        lines.append(f"- {_cell(s['name'])}: family `{s['family']}`, modeling sha256 `{s['modeling_sha256']}`"
                     + (f", revision `{s['revision']}`" if s["revision"] else "") + f", ids `{s['ids']}`")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-json", required=True, type=Path)
    ap.add_argument("--out-md", required=True, type=Path)
    ap.add_argument("--checkpoint", action="append", default=[], help="NAME=DIR of a real checkpoint")
    a = ap.parse_args(argv)
    ho.set_threads()
    with tempfile.TemporaryDirectory() as tmp:
        subjects = synthetic_subjects(Path(tmp))
        subjects += [real_subject(n, Path(d)) for n, d in (c.split("=", 1) for c in a.checkpoint)]
        rep = report(subjects)
    a.out_json.write_text(json.dumps(rep, indent=1, sort_keys=True, default=str) + "\n")
    a.out_md.write_text(markdown(rep))
    failed = [s["name"] for s in rep["subjects"] if not s["operations"]["pass"]]
    print("part 5 failed for:" if failed else "part 5 passed for every subject", *failed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
