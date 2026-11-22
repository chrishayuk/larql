"""Test 3 Stage 4 (holdout): qualification on a development stand-in. No holdout data is read and no model is run.

    gwtemplate_test3_holdout_qualify.py OUT_DIR      (OUT_DIR must not exist)

Every Stage 4 function is applied to development data and compared with an
existing qualified artifact. The 20 development-validation countries stand in
for the holdout: their captures, Stage 2 vectors and Stage 2 outputs already
exist.

| Gate | Check |
|---|---|
| QH1 | eligibility(joint − holdout) = the 59 Test 2 development countries |
| QH2 | donors(dev, Test 2 salt) = the Test 2 donors |
| QH3 | runs(dev) = the Stage 1b capture-plan runs (tokens, window, ids) |
| QH4 | seal_plan(dev) = the Test 2 seal plan |
| QH5 | capture_plan(dev, Test 2 seal) = the Stage 1b capture plan, byte-equal as JSON |
| QH6 | the seal's last step equals the baseline natural readout for every Test 2 run (validates the Stage 4 seal-vs-baseline check) |
| QH7 | stand-in derived plan (targets = 20 validation countries, refusal = fit): ID/S/C1/C2 vector ids equal Stage 2's; runner `--check` passes; a refused target country refuses; admission 0 failures; every vector recomputes |
| QH8 | planted defects are each rejected by admission |
| QH9 | Test 2-format replication plan: arms and runs equal Test 2's plan; the xpos runner's `--check` passes on it, sealed |
| QH10 | analysis: holdout per_arm on Stage 2 rows reproduces Stage 2's per-arm metrics exactly for S/C1/C2; the gate points equal Stage 2's |
| QH11 | the capture view refuses holdout and blocked countries |
| QH12 | tests: the Stage 4 synthetic tests, Stage 2 analysis tests and v1 builder tests |
"""
from __future__ import annotations

import copy
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import gwtemplate_test2_feasibility as feas  # noqa: E402
import gwtemplate_test3_candidates as v1  # noqa: E402
import gwtemplate_test3_holdout as h  # noqa: E402
import gwtemplate_test3_holdout_admit as adm  # noqa: E402
import gwtemplate_test3_holdout_analyze as an  # noqa: E402
import gwtemplate_test3_holdout_run as hr  # noqa: E402

GF = ROOT / "output/gw-template-factorial"
STAGE2 = GF / "test3-stage2-02"
INPUT_KEYS = [k for k in hr.FILES if k not in ("qualification",)]


def load_jsonl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def run(args):
    return subprocess.run([str(a) for a in args], capture_output=True, text=True, cwd=ROOT)


def planted(plan, index, caps, fit, elig, dmap):
    """Each mutation keeps recipe hashes and input hashes valid, so only the Stage 4 rule can reject it."""
    t = next(t for t in plan["targets"] if t["relation"] == h.REL and len(t["donors"]) >= 1)
    Y, ph = t["country"], t["phrasing"]
    other_elig = next(c for c in sorted(elig) if c != Y)

    def inp(role, rel, c, l, js):
        return [role, rel, ph, c, l, js, caps.window_sha(f"{rel}|{ph}|{c}", l, js)]

    def mutate(arm_name, fn):
        p2, i2 = copy.deepcopy(plan), copy.deepcopy(index)
        tt = next(x for x in p2["targets"] if x["id"] == t["id"])
        a = next(x for x in tt["arms"] if x["name"] == arm_name)
        for key, vid in list(a["vectors"].items()):
            e = i2[vid]
            rec = copy.deepcopy(e["recipe"])
            fn(rec)
            e2 = dict(e, recipe=rec, recipe_sha256=adm.rsha(rec))
            nv = vid + "-x"
            i2[nv] = e2
            a["vectors"][key] = nv
        return len(adm.admit(p2, i2, None, caps, fit, elig, set(), dmap)) > 0

    def c1_add_holdout(rec):
        rec["inputs"].append(inp("member", h.RP, other_elig, rec["layer"], rec["source_slot"]))
        rec["members"] = rec["members"] + [other_elig]

    def c2_reads_target_rp(rec):
        rec["inputs"] = [inp("own", h.RP, Y, rec["layer"], rec["source_slot"]) if i[0] == "own" else i for i in rec["inputs"]]

    def c2_plus_target_rp(rec):
        rec["inputs"].append(inp("plus", h.RP, Y, rec["layer"], rec["source_slot"]))

    def c1_drop_member(rec):
        rec["inputs"] = rec["inputs"][1:]
        rec["members"] = rec["members"][1:]

    def c0_wrong_donor(rec):
        rec["donor"] = other_elig if other_elig != t["donors"][0] else sorted(elig)[-1]
        rec["inputs"] = [inp("copy", h.RP, rec["donor"], rec["layer"], rec["source_slot"])]

    def s_points_to_donor(rec):
        rec["inputs"] = [inp("copy", h.RP, t["donors"][0], rec["layer"], rec["source_slot"])]

    res = {
        "c1_member_is_holdout_country": mutate("C1-window-L1", c1_add_holdout),
        "c2_own_term_reads_target_currency": mutate("C2-window-L1", c2_reads_target_rp),
        "c2_plus_includes_target_currency": mutate("C2-window-L1", c2_plus_target_rp),
        "c1_membership_not_full_fit": mutate("C1-window-L1", c1_drop_member),
        "c0_undeclared_donor": mutate("C0-D1-L1", c0_wrong_donor),
        "s_not_target_window": mutate("S-window-L1", s_points_to_donor),
    }
    p2 = copy.deepcopy(plan)
    tt = next(x for x in p2["targets"] if x["id"] == t["id"])
    tt["arms"] = [a for a in tt["arms"] if a["name"] != "C2-window-L1"]
    res["arm_missing"] = len(adm.admit(p2, index, None, caps, fit, elig, set(), dmap)) > 0
    p3 = copy.deepcopy(plan)
    p3["holdout"] = sorted(set(p3["holdout"]) | {Y})
    res["target_on_refusal_list"] = len(adm.admit(p3, index, None, caps, fit, elig, set(), dmap)) > 0
    res["target_not_eligible"] = len(adm.admit(plan, index, None, caps, fit, elig - {Y}, set(), dmap)) > 0
    return res


def main(out: Path):
    if out.exists():
        raise SystemExit(f"FAIL CLOSED: {out} exists")
    out.mkdir(parents=False)
    t0 = time.time()
    q, g = {}, {}
    pop = json.loads(h.POP.read_text())
    bplan = json.loads((h.BASE / "plan.json").read_text())
    cells = {c["id"]: c for c in bplan["cells"]}
    t2 = json.loads((GF / "test2-02/plan.json").read_text())
    dev = sorted({t["country"] for t in t2["targets"]})
    fit, val = v1.split(dev)
    hold = set(pop["proposed_test3_entity_holdout"]["countries"])
    cands = set(next(iter(t2["candidate_sets"].values())))
    canon = h.canonical_window_ids(cells, dev)
    ok, exc = h.eligibility(pop, cells, [c for c in pop["populations"]["joint_countries"] if c not in hold], canon, cands)
    q["QH1"] = {"eligible": len(ok), "excluded": exc}
    g["QH1_eligibility_reproduces_dev"] = ok == dev
    d2 = h.donors(cells, dev, feas.DONOR_SALT)
    g["QH2_donors_reproduce_test2"] = all(d2[t["country"]]["donors"] == t["donors"] for t in t2["targets"])
    rs = h.runs(cells, dev)
    cp = json.loads((GF / "test3-capture-01/capture-plan.json").read_text())
    g["QH3_runs_reproduce_stage1b"] = set(rs) == set(cp["runs"]) and all(
        rs[r][k] == cp["runs"][r][k] for r in rs for k in ("country", "tokens", "window", "window_ids"))
    sp = h.seal_plan(t2["container"], rs)
    t2sp = json.loads((GF / "test2-02/seal-plan.json").read_text())
    g["QH4_seal_plan_reproduces_test2"] = sp["schema"] == t2sp["schema"] and sp["container"] == t2sp["container"] and \
        {r["id"]: (r["tokens"], r["positions"]) for r in sp["runs"]} == {r["id"]: (r["tokens"], r["positions"]) for r in t2sp["runs"]}
    seal = load_jsonl(GF / "test2-02/run/seal.jsonl")
    mycp = h.capture_plan({"container": t2["container"], "tokenizer_sha256": t2["tokenizer_sha256"], "layers": 34},
                          rs, seal, cp["holdout"], cp["sources"])
    g["QH5_capture_plan_byte_equal"] = json.dumps(mycp, indent=1) + "\n" == (GF / "test3-capture-01/capture-plan.json").read_text()
    sd = {r["id"]: r for r in seal}
    eq = sum(sd[t["run"]]["step_logit_sha256"][-1] == t["expect_natural_logit_sha256"] for t in t2["targets"])
    q["QH6"] = {"equal": eq, "runs": len(t2["targets"])}
    g["QH6_seal_last_step_equals_baseline"] = eq == len(t2["targets"]) == 236
    # QH7 stand-in derived plan
    cap = h.capture_view([GF / "test3-capture-01/capture"], allowed=set(fit) | set(val), blocked=hold)
    dmap = h.donors(cells, val, h.HOLDOUT_DONOR_SALT)
    q["QH7_donor_counts"] = {y: len(v["donors"]) for y, v in dmap.items()}
    run_meta = {rid: cp["runs"][rid] for rid in cp["runs"] if rid.split("|")[2] in val}
    built = h.derived(out / "standin", cap, fit, val, dmap, run_meta, {"container": t2["container"],
                      "tokenizer_sha256": t2["tokenizer_sha256"], "layers": 34}, fit, t2["candidate_sets"], id_prefix="qh")
    s2b = json.loads((STAGE2 / "binding.json").read_text())
    s2arms = {}
    for s in s2b["shards"]:
        for t in json.loads((STAGE2 / s["plan"]).read_text())["targets"]:
            for a in t["arms"]:
                s2arms[(t["run"], a["name"])] = a["vectors"]
    same = diff = 0
    checks, refusal_ok, adm_f, recomp_f = {}, True, [], 0
    caps = adm.Captures([GF / "test3-capture-01/capture"])
    import numpy as np
    plans = {}
    for s in built["shards"]:
        plan = json.loads((out / "standin" / s["plan"]).read_text())
        index = json.loads((out / "standin" / s["index"]).read_text())
        plans[s["name"]] = (plan, index)
        for t in plan["targets"]:
            for a in t["arms"]:
                if a["name"].startswith("C0-"):
                    continue
                if a["vectors"] == s2arms[(t["run"], a["name"])]:
                    same += 1
                else:
                    diff += 1
        p = run([hr.ROOT / hr.EXE["derived"], "--check", out / "standin" / s["plan"]])
        checks[s["name"]] = p.stdout.strip() if p.returncode == 0 else f"FAILED {p.stderr[-300:]}"
        bad = dict(plan, holdout=sorted(set(plan["holdout"]) | {plan["targets"][0]["country"]}))
        (out / "standin" / f"refuse-{s['name']}.json").write_text(json.dumps(bad))
        pr = run([hr.ROOT / hr.EXE["derived"], "--check", out / "standin" / f"refuse-{s['name']}.json"])
        refusal_ok &= pr.returncode != 0 and "holdout country" in (pr.stderr + pr.stdout)
        adm_f += adm.admit(plan, index, None, caps, set(fit), set(val), set(), dmap)
        data = np.fromfile(out / "standin" / s["store"], dtype="<f4")
        for vid, e in index.items():
            if adm.vsha(adm.recompute(e["recipe"], caps)) != e["sha256"] or adm.vsha(data[e["offset"]:e["offset"] + built["hidden"]]) != e["sha256"]:
                recomp_f += 1
    q["QH7"] = {"arms_equal_stage2": same, "arms_differing": diff, "runner_check": checks, "admission_failures": adm_f[:20],
                "recompute_failures": recomp_f, "shards": built["shards"]}
    g["QH7_standin_equals_stage2_vectors"] = diff == 0 and same > 0
    g["QH7_runner_check"] = all(not v.startswith("FAILED") for v in checks.values())
    g["QH7_runner_refuses_listed_target"] = refusal_ok
    g["QH7_admission_and_recompute"] = not adm_f and recomp_f == 0
    # QH8 planted defects
    plan, index = plans["capital-P2"]
    q["QH8"] = planted(plan, index, caps, set(fit), set(val), dmap)
    g["QH8_planted_defects_rejected"] = all(q["QH8"].values())
    # QH9 Test 2-format replication plan vs Test 2
    t2d = {t["country"]: {"donors": t["donors"]} for t in t2["targets"]}
    expect = {t["run"]: t["expect_natural_logit_sha256"] for t in t2["targets"]}
    xp = h.xpos_plan({"container": t2["container"], "tokenizer_sha256": t2["tokenizer_sha256"], "layers": 34, "top_k": t2["top_k"]},
                     cells, dev, t2d, expect, t2["candidate_sets"])
    t2t = {t["run"]: t for t in t2["targets"]}
    arms_eq = all(a == next(b for b in t2t[t["run"]]["arms"] if b["name"] == a["name"]) for t in xp["targets"] for a in t["arms"])
    runs_eq = all(xp["runs"][r] == t2["runs"][r] for r in xp["runs"])
    sdx = {r["id"]: r for r in seal}
    for rid in {a["donor_run"] for t in xp["targets"] for a in t["arms"]}:
        xp["runs"][rid]["expected_carrier_sha256"] = dict(sdx[rid]["carriers"])
    (out / "xpos-standin-sealed.json").write_text(json.dumps(xp) + "\n")
    px = run([hr.ROOT / hr.EXE["xpos"], "--check", out / "xpos-standin-sealed.json"])
    q["QH9"] = {"arms_equal_test2": arms_eq, "runs_equal_test2": runs_eq, "targets": len(xp["targets"]),
                "arms": sum(len(t["arms"]) for t in xp["targets"]), "xpos_check": px.stdout.strip()[-200:] or px.stderr[-300:]}
    g["QH9_replication_plan_equals_test2_and_checks"] = arms_eq and runs_eq and px.returncode == 0
    # QH10 analysis reproduces Stage 2 per-arm metrics
    s2rows = {}
    for s in s2b["shards"]:
        for r in load_jsonl(STAGE2 / s["output"]):
            s2rows[r["run"]] = r
    tg, rows = [], []
    keep = ("ID-window-L1", "S-window-L1", "C1-window-L1", "C2-window-L1")
    for s in built["shards"]:
        for t in json.loads((out / "standin" / s["plan"]).read_text())["targets"]:
            t2_ = dict(t, arms=[a for a in t["arms"] if a["name"] in keep])
            tg.append(t2_)
            r = s2rows[t["run"]]
            rows.append({"id": t["id"], "run": t["run"], "natural": r["natural"], "arms": {k: r["arms"][k] for k in keep if k in r["arms"]}})
    cl = an.HoldoutClasses(bplan, dev, [], next(iter(t2["candidate_sets"].values())))
    import gwtemplate_baseline_plan as bp
    tok = bp.tokenizer()
    mine = an.per_arm(tg, rows, cl, {t["run"]: t["donors"] for t in tg}, lambda i: tok.decode([i]))
    rep = json.loads((STAGE2 / "report.json").read_text())
    theirs = {(a["target"], a["phrasing"], a["kind"]): a for a in rep["per_arm"]
              if a["relation"] == "capital" and a["layer"] == 1 and a["population"] == "validation" and a["kind"] in ("S", "C1", "C2")}
    fields = ("outcome", "m_switch", "gain_rY", "gain_rpY", "kl_ref", "b3_mean", "b3_wmean", "b3_excess", "lp_rY", "lp_rpY")
    mism = [k for k in theirs if (k not in {(a["target"], a["phrasing"], a["kind"]) for a in mine})
            or any(next(a for a in mine if (a["target"], a["phrasing"], a["kind"]) == k)[f] != theirs[k][f] for f in fields)]
    groups = an.summarise(mine)
    gt = an.gates(groups)
    s2g = rep["gates"]
    pts = {k: (gt[f"{k}|capital"]["A"]["pooled_share"], gt[f"{k}|capital"]["B"]["b3_mean"], gt[f"{k}|capital"]["B"]["b3_max"],
               gt[f"{k}|capital"]["B"]["other_top1"]) for k in ("C1", "C2")}
    s2pts = {k: (s2g[f"{k}|capital"]["A"]["pooled_share"], s2g[f"{k}|capital"]["B"]["b3_mean"], s2g[f"{k}|capital"]["B"]["b3_max"],
                 s2g[f"{k}|capital"]["B"]["other_top1"]) for k in ("C1", "C2")}
    q["QH10"] = {"per_arm_compared": len(theirs), "mismatches": mism[:20], "gate_points": pts, "stage2_gate_points": s2pts,
                 "standin_verdicts": {k: v["verdict"] for k, v in gt.items()}}
    g["QH10_analysis_reproduces_stage2"] = not mism and len(theirs) == 120 and pts == s2pts
    # QH11 capture view refusals
    hc = sorted(hold)[0]
    ref = {}
    for name, fn in (("holdout_country", lambda: cap.raw("capital", "P2", hc)),
                     ("validation_rprime_for_candidate", lambda: v1.Builder(cap, fit)._cand_read(val[0], "capital", "P2", h.RP, val[0])),
                     ("other_target_for_candidate", lambda: v1.Builder(cap, fit)._cand_read(val[0], "capital", "P2", "capital", val[1]))):
        try:
            fn()
            ref[name] = False
        except v1.InformationRuleError:
            ref[name] = True
    q["QH11"] = ref
    g["QH11_capture_view_refusals"] = all(ref.values())
    # QH12 tests
    tests = {}
    for t in ("scripts/test_gwtemplate_test3_holdout.py", "scripts/test_gwtemplate_test3_stage2_analyze.py",
              "scripts/test_gwtemplate_test3_candidates.py"):
        p = run([sys.executable, t])
        tests[t] = p.returncode
    q["QH12"] = tests
    g["QH12_tests"] = all(v == 0 for v in tests.values())
    inputs = {k: hr.sha(ROOT / hr.FILES[k]) for k in INPUT_KEYS}
    res = {"schema": "larql.gwtemplate.test3-stage4-qualification.v1",
           "status": "qualified" if all(g.values()) else "FAILED",
           "scope": "development stand-in (20 validation countries); no holdout data read; no model run",
           "gates": g, "details": q, "inputs": inputs, "seconds": time.time() - t0}
    (out / "qualification.json").write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"status": res["status"], "gates": g}, indent=1))
    sys.exit(0 if res["status"] == "qualified" else 1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    main(Path(sys.argv[1]).resolve())
