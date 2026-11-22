"""GW-ELEMENT-1 natural-baseline feasibility (natural inference only; owner GO 2026-11-21).

    gwelem_baseline.py prepare OUT_DIR   # model-free: plan from the frozen feasibility audit
    gwelem_baseline.py bind OUT_DIR      # identities, parser qualification, --check, protected records
    gwelem_baseline.py execute OUT_DIR   # verify binding, run (power logged), integrity, parse, report

Protocol: docs/gw-element-1-baseline-protocol.md.
- Runner: `gwtemplate_baseline`, unchanged; qualified in baseline-qual-01, binary 5816e19d…
- No intervention, no fitting, no C2. Prompts, exclusions and split come from
  the frozen feasibility audit (result 6314) and are never changed on outputs.
- Four anchor cells from the Test 1 baseline (baseline-01) must reproduce
  their records bit-exactly (same model, runner and tokenizer).
"""
from __future__ import annotations

import collections
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import gwelem_feasibility as fz  # noqa: E402
import gwelem_parse as ps  # noqa: E402
import gwtemplate_baseline_plan as bp  # noqa: E402
import gwtemplate_feasibility as feas  # noqa: E402
import gwtemplate_test3_holdout_run as hr  # noqa: E402

GF = ROOT / "output/gw-template-factorial"
FEAS = ROOT / "output/gw-element-1/feasibility-01/feasibility.json"
FEAS_SHA = "sha256:068ecc941c1aec641f4017d890f1ce5ccc70fe643cd1822ca85349f031d8924e"   # artifact 2748
BASE1 = GF / "baseline-01"
EXE = "target/release/examples/gwtemplate_baseline"
EXPECT = {"runner": "sha256:5816e19dca022ec148865c0b8b32135e3fb9341e3cf20ac3d09ce2b40acd6dba",
          "runner_source": "sha256:bd76a13491a39f1243639cee8b3ffa491fc128635ae8d5801d3584b981d9902c",
          "tokenizer": "sha256:" + feas.TOKENIZER_SHA}
PHRASINGS = ("A2", "A3")
RELS = ("symbol", "number")
WORD = {"symbol": "symbol", "number": "number"}
EXCLUDED = {"multi_token_symbol": ["Yb", "Hf", "Tl", "Np", "Lr", "Sg"],
            "A3_subject_echo": ["Al", "Sc", "Y", "I", "Pr", "Eu", "Os", "U", "Fm"]}
ANCHORS = ("capital|P2|FR", "capital|P3|JP", "currency_code_iso4217|P2|BR", "capital|P1|KE")
MIN = {"fit": 30, "development": 10, "confirmation": 20}
NUMBER_FORMAT_MIN = 0.80
CONTINUATION, TOP_K = 8, 64
FILES = {
    "protocol": "docs/gw-element-1-baseline-protocol.md",
    "feasibility_doc": "docs/gw-element-1-feasibility.md",
    "orchestrator": "scripts/gwelem_baseline.py",
    "parser": "scripts/gwelem_parse.py",
    "parser_tests": "scripts/test_gwelem_parse.py",
    "feasibility_script": "scripts/gwelem_feasibility.py",
    "feasibility": "output/gw-element-1/feasibility-01/feasibility.json",
    "plan_builder_helpers": "scripts/gwtemplate_baseline_plan.py",
    "runner": EXE,
    "runner_source": "crates/larql-demos/examples/inference/gwtemplate_baseline.rs",
    "runner_qualification": "output/gw-template-factorial/baseline-qual-01/qualification.json",
    "tokenizer": str(feas.TOKENIZER),
    "model_index": str(feas.TOKENIZER.parent / "index.json"),
    "model_system_graph": str(feas.TOKENIZER.parent / "system_graph.json"),
    "anchor_plan": "output/gw-template-factorial/baseline-01/plan.json",
    "anchor_out": "output/gw-template-factorial/baseline-01/run/baseline-out.jsonl",
}
PROTECTED = [GF / p for p in ("test3-stage2-02/binding.json", "test3-stage4-02/binding.json", "test3-stage4-02/HOLDOUT-SPENT.txt",
                              "compress1-cp3a-01/binding.json", "compress1-cp3b-01/binding.json",
                              "compress1-cp3b-01/integrity.json", "compress1-cp3c-01/binding.json",
                              "compress1-cp3c-01/integrity.json", "compress1-cp3c-01/report.json")] + [FEAS]


def sha(p) -> str:
    return "sha256:" + hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fail(m):
    raise SystemExit(f"FAIL CLOSED: {m}")


def path(p):
    return Path(p) if Path(p).is_absolute() else ROOT / p


def load_jsonl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def eligible_tokens(tok, feas_rep) -> list[tuple]:
    """Recompute the token-level exclusions and require them to equal the frozen list."""
    multi = sorted(s for z, s, n in fz.ELEMENTS if len(tok.encode(" " + s, add_special_tokens=False).ids) > 1)
    echo = sorted(feas_rep["phrasings"]["A3"]["subject_echo"]["symbol"])
    if multi != sorted(EXCLUDED["multi_token_symbol"]) or echo != sorted(EXCLUDED["A3_subject_echo"]):
        fail(f"token exclusions differ from the frozen audit: {multi} {echo}")
    out = {s for v in EXCLUDED.values() for s in v}
    return [e for e in fz.ELEMENTS if e[1] not in out]


def build_plan(tok) -> dict:
    if sha(FEAS) != FEAS_SHA:
        fail("feasibility record differs from the registered artifact 2748")
    rep = json.loads(FEAS.read_text())
    split = rep["split"]["assignment"]
    els = eligible_tokens(tok, rep)
    space = tok.encode(" ", add_special_tokens=False).ids[0]
    digits = [tok.encode(d, add_special_tokens=False).ids[0] for d in "0123456789"]
    syms = {tok.encode(" " + s, add_special_tokens=False).ids[0] for z, s, n in fz.ELEMENTS}
    cset = sorted(syms | {space} | set(digits))
    cells = []
    for z, s, name in els:
        for ph in PHRASINGS:
            for rel in RELS:
                t = fz.CANDIDATES[ph][rel]
                subj = name if "{x}" in t else name[0].upper() + name[1:]
                c = bp.cell(tok, f"{rel}|{ph}|{z}", t.replace("{x}", "{X}"), WORD[rel], subj, "element",
                            {"element": z, "symbol": s, "name": name, "relation_name": rel, "phrasing": ph,
                             "split": split[str(z)]})
                cells.append(c)
    a_plan = json.loads((BASE1 / "plan.json").read_text())
    a_cells = {c["id"]: c for c in a_plan["cells"]}
    anchors = [a_cells[i] for i in ANCHORS]
    sets = {"element": cset} | {c["candidate_set"]: a_plan["candidate_sets"][c["candidate_set"]] for c in anchors}
    return {"schema": bp.SCHEMA, "container": bp.CONTAINER, "tokenizer_sha256": "sha256:" + feas.TOKENIZER_SHA,
            "continuation_tokens": CONTINUATION, "top_k": TOP_K,
            "purpose": "GW-ELEMENT-1 natural-baseline feasibility; natural inference only",
            "candidate_sets": sets, "cells": cells + anchors, "anchors": list(ANCHORS),
            "token_exclusions": EXCLUDED, "split_salt": rep["split"]["salt"]}


def run(args):
    return subprocess.run([str(a) for a in args], capture_output=True, text=True, cwd=ROOT)


def prepare(out: Path):
    if out.exists():
        fail(f"{out} exists")
    out.mkdir(parents=True)
    tok = bp.tokenizer()
    plan = build_plan(tok)
    (out / "plan.json").write_text(json.dumps(plan, indent=1))
    els = {c["element"]: c["split"] for c in plan["cells"] if "element" in c}
    print(json.dumps({"cells": len(plan["cells"]), "element_cells": sum(1 for c in plan["cells"] if "element" in c),
                      "anchors": len(ANCHORS), "token_eligible": len(els),
                      "by_split": dict(collections.Counter(els.values()))}))


def bind(out: Path):
    if (out / "binding.json").exists():
        fail("already bound")
    files = {k: {"path": p, "sha256": sha(path(p))} for k, p in FILES.items()}
    for k, v in EXPECT.items():
        if files[k]["sha256"] != v:
            fail(f"{k} identity {files[k]['sha256']} != qualified {v}")
    if json.loads(path(FILES["runner_qualification"]).read_text()).get("status") not in ("qualified", "QUALIFIED"):
        fail("baseline runner qualification is not 'qualified'")
    files["plan"] = {"path": str((out / "plan.json").relative_to(ROOT)), "sha256": sha(out / "plan.json")}
    tok = bp.tokenizer()
    if json.dumps(build_plan(tok), indent=1) != (out / "plan.json").read_text():
        fail("plan is not reproducible from the frozen inputs")
    r = run([sys.executable, "-m", "unittest", "scripts/test_gwelem_parse.py"])
    if r.returncode != 0:
        fail(f"parser qualification: {r.stderr[-400:]}")
    r2 = run([ROOT / EXE, "--check", out / "plan.json"])
    if r2.returncode != 0:
        fail(f"--check: {r2.stderr[-300:]}")
    b = {"schema": "larql.gwelem.baseline-binding.v1", "status": "bound before inference; natural inference only",
         "authorisation": "owner GO 2026-11-21: GW-ELEMENT-1 natural-baseline feasibility only",
         "files": files, "parser_qualification": r.stderr.strip().splitlines()[-1], "runner_check": (r2.stdout + r2.stderr).strip()[-300:],
         "constants": {"min_naturally_eligible": MIN, "number_format_min": NUMBER_FORMAT_MIN,
                       "continuation_tokens": CONTINUATION, "top_k": TOP_K, "anchors": list(ANCHORS),
                       "special_terminators": ["<eos>", "<end_of_turn>"]},
         "power_policy": "log_only", "protected": {str(p.relative_to(ROOT)): sha(p) for p in PROTECTED}}
    (out / "binding.json").write_text(json.dumps(b, indent=1))
    print("bound", sha(out / "binding.json"))


def report(plan: dict, recs: dict, tok) -> dict:
    dec = lambda i: tok.decode([i])  # noqa: E731
    space = tok.encode(" ", add_special_tokens=False).ids[0]
    special = frozenset({tok.token_to_id("<eos>"), tok.token_to_id("<end_of_turn>")})
    per, acc, reasons, first = {}, collections.defaultdict(lambda: [0, 0]), collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter)
    for c in plan["cells"]:
        if "element" not in c:
            continue
        r = recs[c["id"]]
        cont = r["continuation"]
        if c["relation_name"] == "symbol":
            want = tok.encode(" " + c["symbol"], add_special_tokens=False).ids[0]
            v = ps.classify_symbol(cont, want, dec, special)
        else:
            v = ps.classify_number(cont, c["element"], space, dec, special)
        key = f"{c['relation_name']}|{c['phrasing']}"
        acc[key][1] += 1
        acc[key][0] += v["correct"]
        reasons[key][v["reason"]] += 1
        first[key][dec(cont[0])] += 1
        per[c["id"]] = v | {"continuation_text": "".join(dec(i) for i in cont), "first_token": dec(cont[0])}
    elements = sorted({c["element"]: c for c in plan["cells"] if "element" in c}.values(), key=lambda c: c["element"])
    elig, why = {}, {}
    for c in elements:
        z = c["element"]
        bad = [f"{rel}|{ph}: {per[f'{rel}|{ph}|{z}']['reason']}" for ph in PHRASINGS for rel in RELS
               if not per[f"{rel}|{ph}|{z}"]["correct"]]
        elig[z] = not bad
        if bad:
            why[f"{z} {c['symbol']} {c['name']} ({c['split']})"] = bad
    split = {c["element"]: c["split"] for c in elements}
    counts = {s: {"token_eligible": sum(1 for z in split if split[z] == s), "naturally_eligible": sum(1 for z in split if split[z] == s and elig[z])}
              for s in MIN}
    fmt = {ph: first[f"number|{ph}"][" "] / acc[f"number|{ph}"][1] for ph in PHRASINGS}
    req = {f"min_{s}": counts[s]["naturally_eligible"] >= MIN[s] for s in MIN}
    req |= {f"number_first_token_space_{ph}": fmt[ph] >= NUMBER_FORMAT_MIN for ph in PHRASINGS}
    return {"schema": "larql.gwelem.baseline-report.v1",
            "accuracy": {k: {"correct": v[0], "of": v[1], "share": v[0] / v[1]} for k, v in sorted(acc.items())},
            "reasons": {k: dict(v) for k, v in sorted(reasons.items())},
            "first_tokens": {k: v.most_common(10) for k, v in sorted(first.items())},
            "number_first_token_space_share": fmt, "counts": counts, "requirements": req,
            "verdict": "feasible: frozen requirements met" if all(req.values()) else "UNSUITABLE: frozen requirements failed",
            "naturally_eligible": sorted(z for z in elig if elig[z]), "natural_exclusions": why,
            "token_exclusions": EXCLUDED, "per_cell": per}


def execute(out: Path):
    b = json.loads((out / "binding.json").read_text())
    for k, f in b["files"].items():
        if sha(path(f["path"])) != f["sha256"]:
            fail(f"bound file changed: {k}")
    if (out / "integrity.json").exists():
        fail("already executed")
    guard = hr.PowerGuard(enforce=False)
    t0 = time.time()
    o = out / "out.jsonl"
    with open(out / "runner-stderr.txt", "w") as err:
        rc = guard.run([str(ROOT / EXE), str(out / "plan.json"), str(o)], stderr=err, label="element-baseline")
    integ = {"power_policy": "log_only", "power_events": guard.events, "wall_seconds": round(time.time() - t0, 1), "rc": rc}
    if rc != 0:
        integ["status"] = "runner failure"
        (out / "integrity.json").write_text(json.dumps(integ, indent=1))
        fail(f"runner rc={rc}")
    plan = json.loads((out / "plan.json").read_text())
    man = json.loads((out / "out.jsonl.manifest.json").read_text())
    rows = load_jsonl(o)
    recs = {r["id"]: r for r in rows}
    base = {r["id"]: r for r in load_jsonl(path(FILES["anchor_out"]))}
    anchors = {i: recs.get(i) == base[i] for i in ANCHORS}
    gates = {"manifest_complete": man["status"] == "complete",
             "manifest_executable": man["executable_sha256"] == b["files"]["runner"]["sha256"],
             "manifest_plan": man["plan_sha256"] == b["files"]["plan"]["sha256"],
             "records_cover_plan": sorted(recs) == sorted(c["id"] for c in plan["cells"]) and len(rows) == len(plan["cells"]),
             "anchors_bit_exact": all(anchors.values()),
             "protected_unchanged": {str(p.relative_to(ROOT)): sha(p) for p in PROTECTED} == b["protected"]}
    integ.update(gates=gates, anchors=anchors, status="admitted" if all(gates.values()) else "NOT admitted")
    (out / "integrity.json").write_text(json.dumps(integ, indent=1))
    print(json.dumps(integ, indent=1))
    if integ["status"] != "admitted":
        fail("integrity not admitted; no report")
    rep = report(plan, recs, bp.tokenizer())
    (out / "report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: rep[k] for k in ("accuracy", "reasons", "first_tokens", "number_first_token_space_share",
                                          "counts", "requirements", "verdict")}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    cmd, d = sys.argv[1], Path(sys.argv[2]).resolve()
    {"prepare": prepare, "bind": bind, "execute": execute}[cmd](d)
