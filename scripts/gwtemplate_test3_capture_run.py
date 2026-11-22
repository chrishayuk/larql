"""GW-TEMPLATE-FACTORIAL test 3, stage 1b: bound capture and reconstruction qualification.

    gwtemplate_test3_capture_run.py bind OUT_DIR
        Writes OUT_DIR/capture-plan.json from frozen test 2 evidence. It
        fails closed unless the executable's static check passes and the tests
        pass. It writes binding.json with the hashes of every executable,
        source, script and input.

    gwtemplate_test3_capture_run.py execute OUT_DIR
        Re-verifies the binding, then:
        1. captures (development runs only; the executable checks replay and
           seal hashes);
        2. independently, in numpy, recomputes every captured vector's hash
           and compares it with the test 2 seal; compares the manifest's K/V
           and step hashes with the seal;
        3. builds the verify plan and runs it: same-run reinjection of all
           236 runs, then the declared 12 D and 4 E arms rebuilt from file
           vectors against their test 2 logit hashes;
        4. runs the independent RoPE check on the dumped K rows;
        5. runs the refusal attacks;
        6. writes qualification.json, with status `qualified` only if every
           check passes.
        It never overwrites. A failure is recorded, not hidden.
"""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import gwtemplate_test3_capture_plan as cplan  # noqa: E402
import gwtemplate_xpos_qualify as xq  # noqa: E402  (independent numpy RoPE; read-only reuse)

EXE = ROOT / "target/release/examples/gwtemplate_test3_capture"
BOUND = {
    "capture_exe": "target/release/examples/gwtemplate_test3_capture",
    "capture_source": "crates/larql-demos/examples/inference/gwtemplate_test3_capture.rs",
    "plan_builder": "scripts/gwtemplate_test3_capture_plan.py",
    "orchestrator": "scripts/gwtemplate_test3_capture_run.py",
    "tests": "scripts/test_gwtemplate_test3_capture.py",
    "rope_reference": "scripts/gwtemplate_xpos_qualify.py",
    "design": "docs/gw-template-factorial-test3-design.md",
    "test2_plan": "output/gw-template-factorial/test2-02/plan.json",
    "test2_seal": "output/gw-template-factorial/test2-02/run/seal.jsonl",
    "test2_output": "output/gw-template-factorial/test2-02/run/runner-out.jsonl",
    "test2_binding": "output/gw-template-factorial/test2-02/binding.json",
    "population": "output/gw-template-factorial/test1-population-01.json",
    "test2_runner_exe": "target/release/examples/gwtemplate_xpos_runner",
}


def sha(p) -> str:
    return "sha256:" + hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(args, **kw):
    return subprocess.run([str(a) for a in args], capture_output=True, text=True, cwd=ROOT, **kw)


def fail(msg):
    raise SystemExit(f"FAIL CLOSED: {msg}")


def vector_sha256(v: np.ndarray) -> str:
    """Independent re-implementation of the runner's vector_sha256 (u64 LE length, then f32 LE)."""
    return hashlib.sha256(struct.pack("<Q", v.size) + v.astype("<f4").tobytes()).hexdigest()


def bind(out: Path) -> int:
    out.mkdir(parents=False)
    plan = cplan.capture_plan()
    (out / "capture-plan.json").write_text(json.dumps(plan, indent=1) + "\n")
    gates = {"static_check": run([EXE, "check", out / "capture-plan.json"]).returncode == 0,
             "tests": run([sys.executable, ROOT / BOUND["tests"]]).returncode == 0,
             "runs": len(plan["runs"]) == 236,
             "no_holdout_run": not ({r["country"] for r in plan["runs"].values()} & set(plan["holdout"]))}
    t2b = json.loads((ROOT / BOUND["test2_binding"]).read_text())
    gates["test2_runner_unchanged"] = sha(ROOT / BOUND["test2_runner_exe"]) == t2b["files"]["runner"]["sha256"]
    if not all(gates.values()):
        (out / "binding-refused.json").write_text(json.dumps(gates, indent=1) + "\n")
        fail(f"gates {[k for k, v in gates.items() if not v]}")
    b = {"schema": "larql.gwtemplate.test3-capture-binding.v1",
         "status": "bound before capture; development runs only; no candidate, fit or holdout",
         "capture_plan_sha256": sha(out / "capture-plan.json"),
         "files": {k: {"path": v, "sha256": sha(ROOT / v)} for k, v in BOUND.items()}, "gates": gates}
    (out / "binding.json").write_text(json.dumps(b, indent=1, sort_keys=True) + "\n")
    print(f"bound: capture plan {b['capture_plan_sha256']}, binding {sha(out / 'binding.json')}")
    return 0


def seal_agreement(cap: Path, plan: dict) -> dict:
    m = json.loads((cap / "manifest.json").read_text())
    res = {"vectors_checked": 0, "kv_checked": 0, "steps_checked": 0, "readout_vectors": 0, "failures": []}
    for rid, r in plan["runs"].items():
        e = m["runs"][rid]
        data = np.fromfile(cap / e["file"], dtype="<f4")
        if sha(cap / e["file"]) != e["file_sha256"]:
            res["failures"].append(f"{rid}: file sha")
        pos, hid, L = e["positions"], e["hidden"], e["layers"]
        data = data.reshape(L, len(pos), hid)
        for l in range(L):
            for j, p in enumerate(pos):
                h = vector_sha256(data[l, j])
                key = f"{l}|{p}"
                if h != e["vector_sha256"][key]:
                    res["failures"].append(f"{rid} {key}: manifest hash")
                if p in r["window"]:
                    res["vectors_checked"] += 1
                    if h != r["expected_carrier_sha256"][key]:
                        res["failures"].append(f"{rid} {key}: differs from the test 2 seal")
                    if e["kv_sha256"][key] != r["expected_kv_sha256"][key]:
                        res["failures"].append(f"{rid} {key}: K/V differs from the seal")
                    res["kv_checked"] += 1
                else:
                    res["readout_vectors"] += 1
        if e["step_logit_sha256"] != r["expected_step_logit_sha256"]:
            res["failures"].append(f"{rid}: step logits differ from the seal")
        res["steps_checked"] += 1
    return res


def attacks(out: Path, plan: dict, cap: Path, vplan: dict) -> dict:
    atk = out / "attacks"
    atk.mkdir()
    res = {}

    def expect_refusal(name, args):
        p = run(args)
        res[name] = p.returncode != 0 and "REFUSED" in (p.stderr + p.stdout)

    def write(name, obj):
        f = atk / name
        f.write_text(json.dumps(obj))
        return f

    first = sorted(plan["runs"])[0]
    m = copy.deepcopy(plan)
    r = copy.deepcopy(m["runs"][first])
    r["country"] = plan["holdout"][0]
    m["runs"][first.rsplit("|", 1)[0] + "|" + plan["holdout"][0]] = r
    expect_refusal("holdout_run", [EXE, "check", write("p-holdout.json", m)])
    m = copy.deepcopy(plan)
    m["runs"][first]["window_ids"][0] += 1
    expect_refusal("wrong_window_ids", [EXE, "check", write("p-wids.json", m)])
    m = copy.deepcopy(plan)
    m["runs"][first]["expected_carrier_sha256"].pop(next(iter(m["runs"][first]["expected_carrier_sha256"])))
    expect_refusal("carrier_coverage", [EXE, "check", write("p-cover.json", m)])
    m = copy.deepcopy(plan)
    k = next(iter(m["runs"][first]["expected_carrier_sha256"]))
    h = m["runs"][first]["expected_carrier_sha256"][k]
    m["runs"][first]["expected_carrier_sha256"][k] = ("0" if h[0] != "0" else "1") + h[1:]
    m["runs"] = {first: m["runs"][first]}
    expect_refusal("seal_mismatch_at_capture", [EXE, "capture", write("p-seal.json", m), atk / "cap-seal"])
    expect_refusal("capture_output_exists", [EXE, "capture", out / "capture-plan.json", cap])
    # verify-side attacks on a one-run tamper capture
    man = json.loads((cap / "manifest.json").read_text())
    e = copy.deepcopy(man["runs"][first])
    for variant in ("file", "vector"):
        d = atk / f"tamper-{variant}"
        (d / "vectors").mkdir(parents=True)
        b = bytearray((cap / e["file"]).read_bytes())
        b[100] ^= 0x01
        (d / e["file"]).write_bytes(b)
        tm = copy.deepcopy(man)
        tm["runs"] = {first: copy.deepcopy(e)}
        if variant == "vector":
            tm["runs"][first]["file_sha256"] = "sha256:" + hashlib.sha256(bytes(b)).hexdigest()
        (d / "manifest.json").write_text(json.dumps(tm))
        vp = {"schema": cplan.VERIFY_SCHEMA, "capture_manifest_sha256": sha(d / "manifest.json"),
              "reinject": [first], "arms": []}
        expect_refusal(f"corrupted_{variant}", [EXE, "verify", out / "capture-plan.json", d,
                                                write(f"v-{variant}.json", vp), atk / f"out-{variant}.json"])
    vp = copy.deepcopy(vplan)
    vp["reinject"] = []
    vp["arms"] = [dict(vp["arms"][0], source_span=[x + 20 for x in vp["arms"][0]["source_span"]])]
    expect_refusal("wrong_slot_map", [EXE, "verify", out / "capture-plan.json", cap, write("v-slot.json", vp),
                                      atk / "out-slot.json"])
    vp = copy.deepcopy(vplan)
    vp["capture_manifest_sha256"] = "sha256:" + "0" * 64
    expect_refusal("wrong_manifest", [EXE, "verify", out / "capture-plan.json", cap, write("v-man.json", vp),
                                      atk / "out-man.json"])
    expect_refusal("verify_output_exists", [EXE, "verify", out / "capture-plan.json", cap, out / "verify-plan.json",
                                            out / "verify.json"])
    for d in atk.glob("tamper-*"):
        shutil.rmtree(d)
    return res


def execute(out: Path) -> int:
    b = json.loads((out / "binding.json").read_text())
    diffs = [k for k, v in b["files"].items() if sha(ROOT / v["path"]) != v["sha256"]]
    if diffs or sha(out / "capture-plan.json") != b["capture_plan_sha256"]:
        fail(f"bound files changed: {diffs}")
    cap = out / "capture"
    if cap.exists() or (out / "qualification.json").exists():
        fail("an execution already exists here; never overwrites")
    plan = json.loads((out / "capture-plan.json").read_text())
    q = {"schema": "larql.gwtemplate.test3-capture-qualification.v1", "binding_sha256": sha(out / "binding.json")}
    print("capturing", flush=True)
    p = run([EXE, "capture", out / "capture-plan.json", cap])
    (out / "capture-stderr.txt").write_text(p.stderr)
    q["capture"] = {"returncode": p.returncode, "stdout": p.stdout.strip()}
    if p.returncode != 0:
        q["status"] = "FAILED"
        (out / "qualification.json").write_text(json.dumps(q, indent=1) + "\n")
        fail("capture failed; see capture-stderr.txt")
    q["manifest_sha256"] = sha(cap / "manifest.json")
    print("seal agreement", flush=True)
    q["seal_agreement"] = seal_agreement(cap, plan)
    vplan = cplan.verify_plan(q["manifest_sha256"])
    (out / "verify-plan.json").write_text(json.dumps(vplan, indent=1) + "\n")
    print("verify", flush=True)
    p = run([EXE, "verify", out / "capture-plan.json", cap, out / "verify-plan.json", out / "verify.json"])
    (out / "verify-stderr.txt").write_text(p.stderr)
    ok_verify = p.returncode == 0
    v = json.loads((out / "verify.json").read_text()) if ok_verify else {"reinject": {}, "arms": {}}
    q["reinject"] = {"runs": len(v["reinject"]), "bit_identical": sum(x["bit_identical"] for x in v["reinject"].values())}
    q["arms"] = {"declared": len(vplan["arms"]), "reproduced": sum(x["reproduced"] for x in v["arms"].values()),
                 "ids": {k: x["reproduced"] for k, x in v["arms"].items()}}
    rc = xq.rope_check([{"id": "verify", "arms": v["arms"]}]) if ok_verify else {"rows": 0, "fail": ["verify failed"]}
    q["rope"] = {k: rc.get(k) for k in ("rows", "max_rel_err", "power_rows", "power_min_rel_err", "same_position_rows")}
    q["rope"]["failures"] = rc["fail"][:20]
    print("attacks", flush=True)
    q["attacks"] = attacks(out, plan, cap, vplan)
    sa = q["seal_agreement"]
    gates = {
        "capture": True,
        "seal_vectors": not sa["failures"] and sa["vectors_checked"] == sum(34 * len(r["window"]) for r in plan["runs"].values()),
        "reinject_all": ok_verify and q["reinject"]["bit_identical"] == 236,
        "arms_reproduced": ok_verify and q["arms"]["reproduced"] == q["arms"]["declared"] == 16,
        "rope": ok_verify and not rc["fail"] and rc["rows"] > 0,
        "attacks": all(q["attacks"].values()),
        "bound_files_unchanged_after": all(sha(ROOT / x["path"]) == x["sha256"] for x in b["files"].values()),
    }
    q["gates"] = gates
    q["status"] = "qualified" if all(gates.values()) else "FAILED"
    (out / "qualification.json").write_text(json.dumps(q, indent=1, sort_keys=True) + "\n")
    print(f"qualification: {q['status']} {gates}", flush=True)
    return 0 if q["status"] == "qualified" else 1


def main(argv):
    if len(argv) != 2 or argv[0] not in ("bind", "execute"):
        print(__doc__, file=sys.stderr)
        return 2
    return (bind if argv[0] == "bind" else execute)(Path(argv[1]))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
