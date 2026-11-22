"""GW-ELEMENT-1: synthetic end-to-end integration qualification of the confirmation orchestrator.

    gwelem_confirm_dryrun.py OUT_DIR

The real orchestrator code (`gwelem_confirm.prepare` / `execute`) runs on **stand-in targets**: two
development elements (3, 8). The real seal and capture binaries capture their natural states only, as
qual-03 already did.

The band runner is replaced by `gwelem_mock_runner.py`, which loads no model and writes format-valid
records whose readouts are copies of the natural readouts. **No constructed state is run through the
model, and no development switching effect is produced.** Each scenario uses its own global marker
inside OUT_DIR, so the real `CONFIRMATION-OPENED.txt` is never touched.

| Scenario | Property | Expected |
|---|---|---|
| happy | 1, 3, 5 | admitted; 16/16 arms ok; release consumed; one analysis written |
| second_execution | 4, 5 | a second prepared directory refuses once the marker exists; nothing written |
| altered_binding | 1, 4 | refused before the marker; no model process |
| partial_capture | 4 | FAILED at capture; no plan, no release, no analysis; marker blocks re-execution |
| leak_view | 2 | FAILED: information rule at build; no runner |
| tampered_store | 2, 4 | FAILED at pre-run audit (recompute); release unused |
| runner_crash | 3, 4 | FAILED; accounting lists the missing arms; release consumed |
| invalid_records | 3 | NOT admitted; accounting lists the invalid identity arms; no analysis |
| hard_kill | 4 | SIGKILL of the orchestrator mid-capture; the fsync'd marker alone refuses re-execution and fresh prepares |
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import gwelem_confirm as G  # noqa: E402

STANDIN = [3, 8]
MOCK = ROOT / "scripts/gwelem_mock_runner.py"


def cfg_for(d: Path, extra=()):
    return {"targets": STANDIN, "global_marker": d / "OPENED.txt", "runner": MOCK,
            "runner_args": ["--caps", f"{d / 'run' / 'capture-number'},{d / 'run' / 'capture-symbol'}", *extra],
            "expect_sealed_targets": False, "label": "integration"}


def go_file(d: Path) -> Path:
    f = d / "GO-dryrun.txt"
    f.write_text("SYNTHETIC INTEGRATION QUALIFICATION ONLY - not an authorisation of any confirmation")
    return f


def scenario(root: Path, name: str, extra=(), hooks=None):
    d = root / name
    d.mkdir(parents=True)
    cfg = cfg_for(d, extra)
    G.prepare(d / "run", cfg)
    t0 = time.time()
    try:
        res = G.execute(d / "run", go_file(d), cfg, hooks or {})
        refused = None
    except SystemExit as e:
        res, refused = None, str(e)
    integ = json.loads((d / "run" / "integrity.json").read_text()) if (d / "run" / "integrity.json").exists() else None
    auth = d / "run" / "authorisation.json"
    return {"dir": d, "cfg": cfg, "res": res, "refused": refused, "integ": integ, "seconds": round(time.time() - t0, 1),
            "marker": cfg["global_marker"].exists(), "release_used": Path(str(auth) + ".used").exists(),
            "report": (d / "run" / "report.json").exists(), "plan": (d / "run" / "plan.json").exists()}


def main(root: Path):
    if root.exists():
        raise SystemExit(f"{root} exists")
    root.mkdir(parents=True)
    checks = {}

    def record(name, passed, **detail):
        checks[name] = {"pass": bool(passed)} | detail
        print(f"{name}: {'PASS' if passed else 'FAIL'} {json.dumps(detail, default=str)[:260]}", flush=True)
        (root / "integration.json").write_text(json.dumps({"schema": "larql.gwelem.confirm-integration.v1",
                                                           "status": "in progress", "checks": checks}, indent=1, default=str))

    # happy path + second execution
    d = root / "happy"
    d.mkdir()
    cfg = cfg_for(d)
    G.prepare(d / "run", cfg)
    d2 = d / "second"
    G.prepare(d2, cfg)  # prepared while the marker is still absent
    rep = G.execute(d / "run", go_file(d), cfg)
    integ = json.loads((d / "run" / "integrity.json").read_text())
    acct = integ["accounting"]
    record("happy", integ["status"] == "admitted; analysed" and acct["ok"] == acct["planned"] == 16 and not acct["missing"]
           and not acct["invalid"] and integ["release_consumed"] and (d / "run" / "report.json").exists()
           and "gates" in rep and integ["audit_c2"]["recompute_vectors"] > 0 and not integ["audit_c2"]["c2_caps_symbol_leaks"],
           status=integ["status"], accounting={k: acct[k] for k in ("planned", "ok", "identity_ok", "identity_planned")},
           verdict_on_mock_readouts=rep["gates"]["primary"]["verdict"], audit=integ["audit_c2"]["dependency"])
    try:
        G.execute(d2, go_file(d), cfg)
        sec = None
    except SystemExit as e:
        sec = str(e)
    record("second_execution", sec is not None and "already been opened" in sec and not (d2 / "integrity.json").exists(),
           refused=sec)
    try:
        G.prepare(d / "third", cfg)
        third = None
    except SystemExit as e:
        third = str(e)
    record("prepare_after_open", third is not None and "already been opened" in third, refused=third)

    def alter(out):
        b = json.loads((out / "binding.json").read_text())
        b["targets"] = [3]
        (out / "binding.json").write_text(json.dumps(b))
    s = scenario(root, "altered_binding", hooks={"alter_binding": alter})
    record("altered_binding", s["refused"] and "changed since prepare" in s["refused"] and not s["marker"] and not s["integ"],
           refused=s["refused"])

    s = scenario(root, "partial_capture", hooks={"partial_capture": "symbol"})
    record("partial_capture", s["integ"] and s["integ"]["status"].startswith("FAILED") and "capture" in s["integ"]["failure"]
           and s["marker"] and not s["plan"] and not s["release_used"] and not s["report"], failure=s["integ"] and s["integ"]["failure"])
    try:
        G.execute(s["dir"] / "run", go_file(s["dir"]), s["cfg"])
        again = None
    except SystemExit as e:
        again = str(e)
    record("no_resume_after_failure", again is not None, refused=again)

    s = scenario(root, "leak_view", hooks={"leak_view": True})
    record("leak_view", s["integ"] and "information rule" in s["integ"].get("failure", "") and not s["release_used"] and not s["report"],
           failure=s["integ"] and s["integ"].get("failure"))

    s = scenario(root, "tampered_store", hooks={"tamper_store": True})
    record("tampered_store", s["integ"] and "pre-run" in s["integ"].get("failure", "") and s["integ"]["audit_c2"]["recompute_failures"]
           and not s["release_used"] and not s["report"], failure=s["integ"] and s["integ"].get("failure"),
           recompute_failures=s["integ"] and len(s["integ"]["audit_c2"]["recompute_failures"]))

    s = scenario(root, "runner_crash", extra=("--crash-after", "1"))
    a = (s["integ"] or {}).get("accounting") or {}
    record("runner_crash", s["integ"] and s["integ"]["status"].startswith("FAILED") and "runner exited" in s["integ"]["failure"]
           and a.get("planned") == 16 and len(a.get("missing", [])) == 16 - a.get("ok", -1) and a.get("ok", 0) < 16
           and s["release_used"] and not s["report"], failure=s["integ"] and s["integ"]["failure"],
           ok=a.get("ok"), missing=len(a.get("missing", [])))

    s = scenario(root, "invalid_records", extra=("--corrupt-identity",))
    a = (s["integ"] or {}).get("accounting") or {}
    record("invalid_records", s["integ"] and s["integ"]["status"].startswith("NOT admitted") and len(a.get("invalid", [])) == 8
           and not s["report"], status=s["integ"] and s["integ"]["status"], invalid=len(a.get("invalid", [])),
           post_failures=s["integ"] and s["integ"]["post_admission"]["n"])

    # hard kill: SIGKILL the orchestrator process mid-capture; refusal must rest on the durable marker alone
    import os
    import signal
    import subprocess
    hk = root / "hard_kill"
    hk.mkdir()
    hcfg = cfg_for(hk)
    G.prepare(hk / "run", hcfg)
    gf = go_file(hk)
    code = ("import sys; sys.path.insert(0, 'scripts'); import gwelem_confirm as G, gwelem_confirm_dryrun as R; "
            f"from pathlib import Path; d = Path({str(hk)!r}); G.execute(d / 'run', Path({str(gf)!r}), R.cfg_for(d))")
    proc = subprocess.Popen([sys.executable, "-c", code], cwd=ROOT, start_new_session=True,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    killed_at = None
    for _ in range(1200):
        time.sleep(0.5)
        ij = hk / "run" / "integrity.json"
        if ij.exists() and any(st["step"] == "seal_equals_baseline" for st in json.loads(ij.read_text())["steps"]):
            os.killpg(proc.pid, signal.SIGKILL)
            killed_at = "after seal_equals_baseline (during capture)"
            break
        if proc.poll() is not None:
            break
    proc.wait()
    integ_k = json.loads((hk / "run" / "integrity.json").read_text()) if (hk / "run" / "integrity.json").exists() else {}
    try:
        G.execute(hk / "run", gf, hcfg)
        again_k = None
    except SystemExit as e:
        again_k = str(e)
    try:
        G.prepare(hk / "fresh", hcfg)
        fresh_k = None
    except SystemExit as e:
        fresh_k = str(e)
    record("hard_kill", killed_at is not None and proc.returncode == -signal.SIGKILL and hcfg["global_marker"].exists()
           and integ_k.get("status") == "in progress" and again_k is not None and "already" in again_k
           and fresh_k is not None and not (hk / "run" / "plan.json").exists(),
           killed_at=killed_at, returncode=proc.returncode, integrity_status_left=integ_k.get("status"),
           re_execute=again_k, fresh_prepare=fresh_k,
           note="a SIGKILL leaves integrity 'in progress' (it cannot be recorded); refusal rests on the durable marker")

    real_marker_untouched = not G.GLOBAL_MARKER.exists()
    record("real_confirmation_untouched", real_marker_untouched, marker=str(G.GLOBAL_MARKER))
    status = "qualified" if all(c["pass"] for c in checks.values()) else "NOT qualified"
    (root / "integration.json").write_text(json.dumps({"schema": "larql.gwelem.confirm-integration.v1", "status": status,
                                                       "checks": checks, "orchestrator_sha256": G.sha(ROOT / "scripts/gwelem_confirm.py"),
                                                       "mock_runner_sha256": G.sha(MOCK)}, indent=1, default=str))
    print("STATUS", status)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
