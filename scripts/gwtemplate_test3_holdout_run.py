"""Test 3 Stage 4 (holdout, capital→currency): bind, prepare, execute.

Protocol: `docs/gw-template-factorial-test3-holdout-protocol.md`. Each step
needs the owner's GO. Each writes into a fresh directory, fails closed and is
never retried in place.

    gwtemplate_test3_holdout_run.py bind OUT_DIR
        Checks the qualified identities (including the Stage 4 qualification
        record), runs the tests, and writes binding.json. No holdout data is read.
    gwtemplate_test3_holdout_run.py prepare OUT_DIR
        The first step that reads holdout data (the baseline plan cells and the
        population file only). It writes:
        - eligibility and exclusions;
        - donors;
        - runs;
        - the seal plan;
        - the unsealed Test 2-format replication plan;
        - prepared.json.
        No model is loaded.
    gwtemplate_test3_holdout_run.py execute OUT_DIR
        Runs, in order:
        - seal;
        - seal == baseline;
        - capture;
        - seal agreement and reinject verify;
        - derived build;
        - --check, admission and recompute;
        - derived runner;
        - Test 2-runner replication of S and C0;
        - integrity;
        - analysis.

**Power:** execution refuses to start on battery. Every model process runs
under `PowerGuard`, which polls the power source every 15 s. On battery it
sends SIGSTOP to the process group; on mains power it sends SIGCONT. Every
pause is recorded in integrity.json, so no computation runs on battery.

**Owner waiver:** `bind OUT_DIR --waive-mains` records
`power_policy: "mains_waived_by_owner"` in binding.json. Only that binding
lets execution start on battery, and its guard only records battery
intervals (no SIGSTOP). The waiver is part of the binding and is reported as
a protocol deviation. It cannot be applied at execute time.
"""
from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

GF = ROOT / "output/gw-template-factorial"
EXE = {"seal": "target/release/examples/gwbreadcrumb2_seal",
       "capture": "target/release/examples/gwtemplate_test3_capture",
       "derived": "target/release/examples/gwtemplate_test3_derived_runner",
       "xpos": "target/release/examples/gwtemplate_xpos_runner"}
FILES = {
    "protocol": "docs/gw-template-factorial-test3-holdout-protocol.md",
    "core": "scripts/gwtemplate_test3_holdout.py",
    "admission": "scripts/gwtemplate_test3_holdout_admit.py",
    "analysis": "scripts/gwtemplate_test3_holdout_analyze.py",
    "orchestrator": "scripts/gwtemplate_test3_holdout_run.py",
    "qualify": "scripts/gwtemplate_test3_holdout_qualify.py",
    "tests": "scripts/test_gwtemplate_test3_holdout.py",
    "qualification": "output/gw-template-factorial/test3-holdout-qual-02/qualification.json",
    "builder_v1": "scripts/gwtemplate_test3_candidates.py",
    "admission_v1": "scripts/gwtemplate_test3_derived_admit.py",
    "stage2_analysis": "scripts/gwtemplate_test3_stage2_analyze.py",
    "feasibility": "scripts/gwtemplate_test2_feasibility.py",
    "test2_plan_builder": "scripts/gwtemplate_test2_plan.py",
    "xpos_admission": "scripts/gwtemplate_xpos_admit.py",
    "capture_harness": "scripts/gwtemplate_test3_capture_run.py",
    "tokenizer_helper": "scripts/gwtemplate_baseline_plan.py",
    "seal_exe": EXE["seal"], "capture_exe": EXE["capture"], "derived_exe": EXE["derived"], "xpos_exe": EXE["xpos"],
    "population": "output/gw-template-factorial/test1-population-01.json",
    "baseline_plan": "output/gw-template-factorial/baseline-01/plan.json",
    "baseline_out": "output/gw-template-factorial/baseline-01/run/baseline-out.jsonl",
    "dev_capture_manifest": "output/gw-template-factorial/test3-capture-01/capture/manifest.json",
    "dev_capture_plan": "output/gw-template-factorial/test3-capture-01/capture-plan.json",
    "test2_plan": "output/gw-template-factorial/test2-02/plan.json",
    "derived_qualification": "output/gw-template-factorial/test3-derived-qual-01/qualification.json",
    "test2_qualification": "output/gw-template-factorial/test2-qual-02/qualification.json",
    "stage2_report": "output/gw-template-factorial/test3-stage2-02/report.json",
    "c2ab_report": "output/gw-template-factorial/test3-c2ab-01/report.json",
}
QUALIFIED = {
    "dev_capture_manifest": "sha256:ce46d6923ee3686c970d91cdc9d1fbe88e9fe0a8260f12661a5b1da6b403599b",
    "stage2_report": "sha256:463517e18c240248dd2db14b8e47dfafab931edfe910e0cba555dc0a0c1c0dfd",
    "c2ab_report": "sha256:2bc1eec9f6e38301da705fce22aa64e259a3f0294ae85ea73be8cc7b1b4d369c",
    "test2_plan": "sha256:bdb643033324984ac19a12881f1f9ca14249b5896fd68cf54a8fc2184f98c712",
}
POLL_SECONDS = 15


def sha(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return "sha256:" + h.hexdigest()


def fail(m):
    raise SystemExit(f"FAIL CLOSED: {m}")


def run(args, **kw):
    return subprocess.run([str(a) for a in args], capture_output=True, text=True, cwd=ROOT, **kw)


def on_mains() -> bool:
    return "AC Power" in subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout


class PowerGuard:
    """Runs a command in its own process group; SIGSTOPs the group while on battery and SIGCONTs it on mains power."""

    def __init__(self, probe=on_mains, interval: float = POLL_SECONDS, enforce: bool = True):
        self.probe, self.interval, self.events, self.enforce = probe, interval, [], enforce
        self.on_battery = False

    def run(self, args, stdout=subprocess.DEVNULL, stderr=None, label: str = "") -> int:
        p = subprocess.Popen([str(a) for a in args], stdout=stdout, stderr=stderr, cwd=ROOT, start_new_session=True)
        paused = False
        while True:
            try:
                p.wait(timeout=self.interval)
                break
            except subprocess.TimeoutExpired:
                pass
            ac = self.probe()
            if not self.enforce:
                if not ac and not self.on_battery:
                    self.events.append({"label": label, "event": "on_battery_waived", "t": time.time()})
                elif ac and self.on_battery:
                    self.events.append({"label": label, "event": "mains_returned", "t": time.time()})
                self.on_battery = not ac
                continue
            if not ac and not paused:
                os.killpg(p.pid, signal.SIGSTOP)
                paused = True
                self.events.append({"label": label, "event": "paused_on_battery", "t": time.time()})
            elif ac and paused:
                os.killpg(p.pid, signal.SIGCONT)
                paused = False
                self.events.append({"label": label, "event": "resumed_on_mains", "t": time.time()})
        return p.returncode


def load_jsonl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def check_identities() -> dict:
    hs = {k: sha(ROOT / p) for k, p in FILES.items()}
    for k, want in QUALIFIED.items():
        if hs[k] != want:
            fail(f"{k} differs from its recorded identity ({hs[k]})")
    dq = json.loads((ROOT / FILES["derived_qualification"]).read_text())
    for k, qk in (("derived_exe", "runner_sha256"), ("builder_v1", "builder_sha256"), ("admission_v1", "admission_sha256")):
        if hs[k] != dq["inputs"][qk]:
            fail(f"{k} is not the qualified version")
    q = json.loads((ROOT / FILES["qualification"]).read_text())
    if q.get("status") != "qualified":
        fail("Stage 4 qualification is not qualified")
    for k, v in q["inputs"].items():
        if k in hs and hs[k] != v:
            fail(f"{k} differs from the Stage 4 qualification record")
    return hs


def bind(d: Path, waive_mains: bool = False):
    if d.exists():
        fail(f"{d} exists")
    hs = check_identities()
    p = run([sys.executable, FILES["tests"]])
    if p.returncode:
        fail(f"tests: {p.stderr[-800:]}")
    d.mkdir(parents=False)
    b = {"schema": "larql.gwtemplate.test3-stage4-binding.v1",
         "status": "bound; holdout not yet read; prepare and execute require owner GO",
         "power_policy": "mains_waived_by_owner" if waive_mains else "mains_required",
         "files": {k: {"path": FILES[k], "sha256": v} for k, v in hs.items()}}
    (d / "binding.json").write_text(json.dumps(b, indent=1, sort_keys=True) + "\n")
    print(f"bound {d} {sha(d / 'binding.json')}")


def verify_binding(d: Path) -> dict:
    b = json.loads((d / "binding.json").read_text())
    for k, f in b["files"].items():
        if sha(ROOT / f["path"]) != f["sha256"]:
            fail(f"bound file changed: {k}")
    return b


def prepare(d: Path):
    import gwtemplate_test3_holdout as h
    import gwtemplate_test2_feasibility as feas
    verify_binding(d)
    if (d / "prepared.json").exists():
        fail("already prepared")
    import gwtemplate_test3_candidates as v1
    pop = json.loads(h.POP.read_text())
    bplan = json.loads((h.BASE / "plan.json").read_text())
    cells = {c["id"]: c for c in bplan["cells"]}
    t2 = json.loads((GF / "test2-02/plan.json").read_text())
    dev = sorted({t["country"] for t in t2["targets"]})
    fit, val = v1.split(dev)
    cand_sets = t2["candidate_sets"]
    canon = h.canonical_window_ids(cells, dev)
    holdout = sorted(pop["proposed_test3_entity_holdout"]["countries"])
    if set(holdout) & set(dev):
        fail("holdout and development overlap")
    eligible, excluded = h.eligibility(pop, cells, holdout, canon, set(next(iter(cand_sets.values()))))
    dmap = h.donors(cells, eligible, h.HOLDOUT_DONOR_SALT)
    rs = h.runs(cells, eligible)
    refusal = sorted(set(dev) | (set(holdout) - set(eligible)))
    seal_plan = h.seal_plan(bplan["container"], rs)
    expect = {}
    for r in load_jsonl(h.BASE / "run/baseline-out.jsonl"):
        if r["id"] in rs:
            expect[r["id"]] = r["readout"]["next_logit_sha256"]
    if set(expect) != set(rs):
        fail("baseline natural records missing for some holdout runs")
    meta = {"container": bplan["container"], "tokenizer_sha256": bplan["tokenizer_sha256"], "layers": 34, "top_k": bplan["top_k"]}
    xp = h.xpos_plan(meta, cells, eligible, dmap, expect, cand_sets)
    (d / "seal-plan.json").write_text(json.dumps(seal_plan, indent=1) + "\n")
    (d / "xpos-plan.json").write_text(json.dumps(xp, indent=1) + "\n")
    prep = {"schema": "larql.gwtemplate.test3-stage4-prepared.v1", "development": dev, "fit": fit, "validation": val,
            "holdout": holdout, "eligible": eligible, "excluded": excluded, "donors": dmap, "refusal": refusal,
            "runs": rs, "expect_natural": expect, "meta": meta, "feasibility_donor_rule": feas.__name__,
            "seal_plan_sha256": sha(d / "seal-plan.json"), "xpos_plan_sha256": sha(d / "xpos-plan.json")}
    (d / "prepared.json").write_text(json.dumps(prep, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"eligible": len(eligible), "excluded": excluded,
                      "donors_per_target": {y: len(v["donors"]) for y, v in dmap.items()}}, indent=1))


def execute(d: Path):
    import gwtemplate_test3_holdout as h
    import gwtemplate_test3_holdout_admit as adm
    import gwtemplate_xpos_admit as xadm
    import gwtemplate_test3_capture_run as crun
    verify_binding(d)
    prep = json.loads((d / "prepared.json").read_text())
    for f, k in (("seal-plan.json", "seal_plan_sha256"), ("xpos-plan.json", "xpos_plan_sha256")):
        if sha(d / f) != prep[k]:
            fail(f"{f} changed since prepare")
    if (d / "run").exists():
        fail("an execution exists; never retried in place")
    waived = json.loads((d / "binding.json").read_text()).get("power_policy") == "mains_waived_by_owner"
    if not waived and not on_mains():
        fail("on battery power; the owner requires mains power")
    rd = d / "run"
    rd.mkdir()
    guard = PowerGuard(enforce=not waived)
    integ = {"schema": "larql.gwtemplate.test3-stage4-integrity.v1", "binding_sha256": sha(d / "binding.json"), "steps": {},
             "power_policy": "mains_waived_by_owner" if waived else "mains_required", "on_mains_at_start": on_mains()}

    def save(status=None):
        if status:
            integ["status"] = status
        integ["power_events"] = guard.events
        (d / "integrity.json").write_text(json.dumps(integ, indent=1) + "\n")

    def step(name, args, stdout=subprocess.DEVNULL):
        t0 = time.time()
        with open(rd / f"{name}-stderr.txt", "w") as err:
            rc = guard.run(["caffeinate", "-i", *args], stdout=stdout, stderr=err, label=name)
        integ["steps"][name] = {"returncode": rc, "seconds": time.time() - t0}
        if rc:
            save("FAILED")
            fail(f"{name} exited {rc}")
        print(f"{name} done in {time.time() - t0:.0f}s", flush=True)

    exe = lambda k: ROOT / EXE[k]  # noqa: E731
    # 1. seal; the last step of every run must equal the baseline natural readout hash
    step("seal", [exe("seal"), d / "seal-plan.json", rd / "seal.jsonl"])
    seal_rows = load_jsonl(rd / "seal.jsonl")
    bad = [r["id"] for r in seal_rows if r["step_logit_sha256"][-1] != prep["expect_natural"][r["id"]]]
    integ["seal_equals_baseline"] = {"runs": len(seal_rows), "mismatches": bad}
    if bad or len(seal_rows) != len(prep["runs"]):
        save("FAILED")
        fail("seal differs from the baseline")
    # 2. capture, seal agreement and reinject verify
    cp = h.capture_plan(prep["meta"], prep["runs"], seal_rows, prep["refusal"],
                        {"seal_sha256": sha(rd / "seal.jsonl"), "purpose": "Test 3 Stage 4 holdout capture"})
    (rd / "capture-plan.json").write_text(json.dumps(cp, indent=1) + "\n")
    step("capture", [exe("capture"), "capture", rd / "capture-plan.json", rd / "capture"])
    sa = crun.seal_agreement(rd / "capture", cp)
    integ["capture_seal_agreement"] = {k: v for k, v in sa.items() if k != "failures"} | {"failures": sa["failures"][:50]}
    if sa["failures"]:
        save("FAILED")
        fail("capture differs from the seal")
    (rd / "verify-plan.json").write_text(json.dumps({"schema": "larql.gwtemplate.test3-capture-verify.v1",
                                                     "capture_manifest_sha256": sha(rd / "capture/manifest.json"),
                                                     "reinject": sorted(cp["runs"]), "arms": [], "sources": {}}, indent=1) + "\n")
    step("verify", [exe("capture"), "verify", rd / "capture-plan.json", rd / "capture", rd / "verify-plan.json", rd / "verify.json"])
    # 3. derived build over the restricted capture view
    cap = h.capture_view([GF / "test3-capture-01/capture", rd / "capture"], allowed=set(prep["fit"]) | set(prep["eligible"]),
                         blocked=set(prep["validation"]) | (set(prep["holdout"]) - set(prep["eligible"])))
    built = h.derived(rd / "derived", cap, prep["fit"], prep["eligible"], prep["donors"], cp["runs"], prep["meta"],
                      prep["refusal"], json.loads((GF / "test2-02/plan.json").read_text())["candidate_sets"])
    (rd / "derived" / "shards.json").write_text(json.dumps(built, indent=1) + "\n")
    caps = adm.Captures([GF / "test3-capture-01/capture", rd / "capture"])
    import numpy as np
    pre = []
    for s in built["shards"]:
        p = run([exe("derived"), "--check", rd / "derived" / s["plan"]])
        if p.returncode:
            save("FAILED")
            fail(f"--check {s['name']}: {p.stderr[-400:]}")
        plan = json.loads((rd / "derived" / s["plan"]).read_text())
        index = json.loads((rd / "derived" / s["index"]).read_text())
        pre += adm.admit(plan, index, None, caps, set(prep["fit"]), set(prep["eligible"]), set(prep["validation"]), prep["donors"])
        data = np.fromfile(rd / "derived" / s["store"], dtype="<f4")
        for vid, e in index.items():
            if adm.vsha(adm.recompute(e["recipe"], caps)) != e["sha256"] or \
                    adm.vsha(data[e["offset"]:e["offset"] + built["hidden"]]) != e["sha256"]:
                pre.append(f"{s['name']} {vid}: recompute")
    integ["admission_pre"] = {"failures": pre[:100], "n": len(pre)}
    if pre:
        save("FAILED")
        fail("pre-execution admission or recompute")
    # 4. derived runner
    for s in built["shards"]:
        step(f"derived-{s['name']}", [exe("derived"), rd / "derived" / s["plan"], rd / "derived" / s["output"]])
    # 5. Test 2-runner replication of S and C0 (sealed donor carriers)
    seal = xadm.seal_index(seal_rows)
    xp = json.loads((d / "xpos-plan.json").read_text())
    for rid in {a["donor_run"] for t in xp["targets"] for a in t["arms"]}:
        xp["runs"][rid]["expected_carrier_sha256"] = dict(seal[rid]["carriers"])
    (rd / "xpos-plan-sealed.json").write_text(json.dumps(xp, indent=1) + "\n")
    if run([exe("xpos"), "--check", rd / "xpos-plan-sealed.json"]).returncode:
        save("FAILED")
        fail("replication plan --check")
    step("xpos", [exe("xpos"), rd / "xpos-plan-sealed.json", rd / "xpos-out.jsonl"])
    xrows = load_jsonl(rd / "xpos-out.jsonl")
    xf = xadm.admit(xp, seal, xrows)
    # 6. integrity
    post, ident, nat, rep, mism = [], 0, 0, 0, []
    xr = {r["id"].split("|", 1)[1]: r for r in xrows}
    n_cells = n_rep = 0
    for s in built["shards"]:
        plan = json.loads((rd / "derived" / s["plan"]).read_text())
        index = json.loads((rd / "derived" / s["index"]).read_text())
        rows = load_jsonl(rd / "derived" / s["output"])
        post += adm.admit(plan, index, rows, caps, set(prep["fit"]), set(prep["eligible"]), set(prep["validation"]), prep["donors"])
        for row in rows:
            n_cells += 1
            rid = row["run"]
            ident += bool(row["arms"]["ID-window-L1"]["identity_bit_exact"])
            if row["natural"]["readout"]["next_logit_sha256"] == prep["expect_natural"][rid]:
                nat += 1
            else:
                mism.append(f"{rid} natural")
            if rid.startswith(h.REL + "|"):
                y = rid.split("|")[2]
                pairs = [("S-window-L1", "S-window-L1")] + [(f"C0-D{k}-L1", f"D{k}-window-L1")
                                                             for k in range(1, len(prep["donors"][y]["donors"]) + 1)]
                for ours, theirs in pairs:
                    n_rep += 1
                    if row["arms"][ours]["readout"]["next_logit_sha256"] == xr[rid]["arms"][theirs]["readout"]["next_logit_sha256"]:
                        rep += 1
                    else:
                        mism.append(f"{rid} {ours}")
    integ["checks"] = {"admission_post_failures": post[:100], "xpos_admission_failures": xf[:100], "identity_bit_exact": ident,
                       "natural_equals_baseline": nat, "cells": n_cells, "s_c0_equal_test2_runner": rep, "replication_arms": n_rep,
                       "mismatches": mism[:100]}
    gates = {"admission_post": not post, "xpos_admission": not xf, "identity_all": ident == n_cells,
             "natural_all": nat == n_cells, "replication_all": rep == n_rep and n_rep > 0}
    integ["gates"] = gates
    save("admitted" if all(gates.values()) else "FAILED")
    if not all(gates.values()):
        fail(f"integrity: {gates}")
    # 7. analysis
    import gwtemplate_test3_holdout_analyze as an
    prep2 = dict(prep, shards=built["shards"])
    rep_ = an.analyze_dir(rd / "derived", d / "report.json", prep2)
    integ["report_sha256"] = sha(d / "report.json")
    save()
    print(json.dumps(rep_["gates"], indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["bind"] and len(a) in (2, 3) and a[2:] in ([], ["--waive-mains"]):
        bind(Path(a[1]).resolve(), waive_mains=a[2:] == ["--waive-mains"])
    elif len(a) == 2 and a[0] in ("prepare", "execute"):
        {"prepare": prepare, "execute": execute}[a[0]](Path(a[1]).resolve())
    else:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
