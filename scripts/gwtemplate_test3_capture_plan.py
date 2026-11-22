"""GW-TEMPLATE-FACTORIAL test 3, stage 1b: the capture and verify plans (model-free).

Built only from frozen test 2 evidence, read-only:

- `test2-02/plan.json`: token runs, windows, arms;
- `test2-02/run/seal.jsonl`: per-run carrier, K/V and step-logit hashes;
- `test2-02/run/runner-out.jsonl`: the recorded arm logit hashes;
- the test 3 holdout ID list (`test1-population-01.json`), used only as a
  refusal list.

**Capture plan:** the 236 development natural runs. For each run, the window
is the seal's declared positions. The plan cross-checks it against every
test 2 window arm that uses the run as a target or donor. The run declares
the seal's hashes.

**Verify plan:**
- reinject every run;
- reproduce a declared salted sample of 12 D arms and 4 E arms from test 2,
  with salt `gw-template-factorial/test3-capture-check/v1|`;
- dump K rows for every sampled arm.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
T2 = ROOT / "output/gw-template-factorial/test2-02"
POP = ROOT / "output/gw-template-factorial/test1-population-01.json"
PLAN_SCHEMA = "larql.gwtemplate.test3-capture-plan.v1"
VERIFY_SCHEMA = "larql.gwtemplate.test3-capture-verify.v1"
SALT = "gw-template-factorial/test3-capture-check/v1|"
N_D, N_E = 12, 4


def sha(p: Path) -> str:
    return "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def holdout() -> list[str]:
    return sorted(json.loads(POP.read_text())["proposed_test3_entity_holdout"]["countries"])


def capture_plan() -> dict:
    plan = json.loads((T2 / "plan.json").read_text())
    seal = {r["id"]: r for r in load_jsonl(T2 / "run/seal.jsonl")}
    if set(seal) != set(plan["runs"]):
        raise SystemExit("the seal and the test 2 plan name different runs")
    hold = set(holdout())
    windows = {}
    for t in plan["targets"]:
        for a in t["arms"]:
            if "-window-" in a["name"] or a["name"] == "identity":
                for run, span in ((t["run"], a["target_span"]), (a["donor_run"], a["source_span"])):
                    if windows.setdefault(run, span) != span:
                        raise SystemExit(f"run {run}: inconsistent window across arms")
    runs = {}
    for rid in sorted(seal):
        s, toks = seal[rid], plan["runs"][rid]["tokens"]
        lo, hi = s["positions"]
        win = list(range(lo, hi + 1))
        if windows.get(rid) != win:
            raise SystemExit(f"run {rid}: seal window {win} differs from the test 2 arms {windows.get(rid)}")
        country = rid.split("|")[-1]
        if country in hold:
            raise SystemExit(f"run {rid}: holdout country in the test 2 plan")
        runs[rid] = {"country": country, "tokens": toks, "window": win, "window_ids": [toks[p] for p in win],
                     "expected_carrier_sha256": s["carriers"], "expected_kv_sha256": s["kv_sha256"],
                     "expected_step_logit_sha256": s["step_logit_sha256"]}
    return {"schema": PLAN_SCHEMA, "container": plan["container"], "tokenizer_sha256": plan["tokenizer_sha256"],
            "layers": plan["layers"], "holdout": sorted(hold), "runs": runs,
            "sources": {"test2_plan_sha256": sha(T2 / "plan.json"), "test2_seal_sha256": sha(T2 / "run/seal.jsonl"),
                        "population_sha256": sha(POP)}}


def sample_arms(plan: dict, rows: list[dict]) -> list[dict]:
    rec = {r["id"]: r for r in rows}
    pool = {"D": [], "E": []}
    for t in plan["targets"]:
        for a in t["arms"]:
            k = a["name"][0]
            if k in pool:
                key = hashlib.sha256((SALT + t["id"] + "|" + a["name"]).encode()).hexdigest()
                pool[k].append((key, t, a))
    out = []
    for k, n in (("D", N_D), ("E", N_E)):
        for _, t, a in sorted(pool[k], key=lambda x: x[0])[:n]:
            out.append({"id": f"{t['id']}|{a['name']}", "target_run": t["run"], "donor_run": a["donor_run"],
                        "source_span": a["source_span"], "target_span": a["target_span"],
                        "from_layer": a["from_layer"], "kv_dump": True,
                        "expect_last_logit_sha256": rec[t["id"]]["arms"][a["name"]]["readout"]["next_logit_sha256"]})
    return out


def verify_plan(capture_manifest_sha: str) -> dict:
    plan = json.loads((T2 / "plan.json").read_text())
    rows = load_jsonl(T2 / "run/runner-out.jsonl")
    return {"schema": VERIFY_SCHEMA, "capture_manifest_sha256": capture_manifest_sha,
            "reinject": sorted(plan["runs"]), "arms": sample_arms(plan, rows),
            "sources": {"test2_output_sha256": sha(T2 / "run/runner-out.jsonl")}}


if __name__ == "__main__":
    import sys
    Path(sys.argv[1]).write_text(json.dumps(capture_plan(), indent=1) + "\n")
