"""GW-ELEMENT-1: the confirmation orchestrator (protocol r2, sha 74f1fb79…). One sealed execution.

    gwelem_confirm.py prepare OUT_DIR                 # model-free: verify identities, write binding.json
    gwelem_confirm.py execute OUT_DIR GO_FILE         # the single confirmation execution

Owner GO 2026-11-21: "GO for the orchestrator, synthetic integration qualification, and then one sealed
confirmation execution under protocol r2". It composes only qualified modules (confirm-qual-03):
gwelem_design, gwelem_plan, gwelem_candidates, gwelem_admit, gwelem_analyze, and the generic band and
sub-window record admissions.

**The five execution properties:**

1. **Exact binding.** `prepare` verifies that every qualified file is byte-identical to the
   confirm-qual-03 bound hashes:
   - protocol, population, runner and capture binaries, modules and analysis;
   - the fit capture manifest and every fit vector file.

   `execute` re-verifies the binding before it touches anything sealed.
2. **No leakage:**
   - the C2 candidate view is loaded from the fit capture plus the confirmation **number** capture only;
   - the symbol runs are captured into a separate directory, which only the S/reference view opens;
   - `audit_c2` recomputes every C2 through capture sets without any target symbol run, and
     dependency-audits them.
3. **Complete accounting.** Every one of the 200 planned (target, arm) pairs is listed with a status
   (ok / missing / invalid). Refusals, crashes and failed audits are recorded with the step where they
   happened. Nothing is silently excluded.
4. **Fail closed:**
   - a global marker `CONFIRMATION-OPENED.txt` is created exclusively before the first model process
     touches a confirmation element;
   - any later `execute`, in any directory, refuses;
   - the runner release is single-use, bound to this plan's runs digest, and consumed before the model loads;
   - every failure ends the run with `integrity.json` naming the failed step; nothing resumes or retries.
5. **Single evaluation.** The analysis runs once, only if integrity is admitted, with the frozen
   `gwelem_analyze` gates and `expected` = the 25 confirmation elements.

`hooks` (fault injection) exists only for the synthetic integration qualification
(`gwelem_confirm_dryrun.py`). The CLI never passes it.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import gwcompress_band_admit as badm  # noqa: E402
import gwcompress_subspan_admit as sadm  # noqa: E402
import gwelem_admit as A  # noqa: E402
import gwelem_analyze as AN  # noqa: E402
import gwelem_candidates as C  # noqa: E402
import gwelem_design as D  # noqa: E402
import gwelem_plan as P  # noqa: E402
import gwtemplate_baseline_plan as bp  # noqa: E402
import gwtemplate_test3_capture_run as crun  # noqa: E402
import gwtemplate_test3_holdout_run as hr  # noqa: E402
import gwelem_spent as SPENT  # noqa: E402

EL = ROOT / "output/gw-element-1"
QUAL = EL / "confirm-qual-03"
PROTOCOL_SHA = "sha256:74f1fb791c6cd24130b5991bacdb20e665102d6053bfc8dac9de20de69c85a10"
FIT_CAPTURE = QUAL / "capture-fit"
FIT_MANIFEST_SHA = "sha256:9a1797180232052e7cd7fc55d94928c2be5945ded17803c07c6668a55d3d4a4a"
GLOBAL_MARKER = EL / "CONFIRMATION-OPENED.txt"
QUALIFIED_KEYS = ("protocol", "design", "plan_builder", "candidates", "admission", "analysis", "runner_source",
                  "band_admission", "subspan_admission", "capture_harness", "feasibility", "baseline_plan",
                  "baseline_out", "baseline_report", "exe_seal", "exe_capture", "exe_elem", "exe_band",
                  "model_index.json", "model_system_graph.json", "model_tokenizer.json")
AUTH_SCHEMA = "larql.gwelem.confirmation-authorisation.v1"


def sha(p) -> str:
    return "sha256:" + hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_jsonl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def run(args):
    return subprocess.run([str(a) for a in args], capture_output=True, text=True, cwd=ROOT)


def default_cfg() -> dict:
    return {"targets": list(D.CONFIRMATION), "global_marker": GLOBAL_MARKER, "runner": ROOT / "target/release/examples/gwelem_band_runner",
            "runner_args": [], "expect_sealed_targets": True, "label": "confirmation"}


class Failed(Exception):
    pass


def qualified_files() -> dict:
    return json.loads((QUAL / "bound-files.json").read_text())


def verify_identities(cfg) -> dict:
    """Every qualified file byte-identical to confirm-qual-03; the fit capture intact."""
    q = json.loads((QUAL / "qualification.json").read_text())
    if q["status"] != "qualified":
        raise Failed("confirm-qual-03 is not qualified")
    bound = qualified_files()
    out, bad = {}, []
    for k in QUALIFIED_KEYS:
        p = bound[k]["path"]
        path = Path(p) if p.startswith("/") else ROOT / p
        h = sha(path)
        out[k] = {"path": p, "sha256": h}
        if h != bound[k]["sha256"]:
            bad.append(k)
    if out["protocol"]["sha256"] != PROTOCOL_SHA:
        bad.append("protocol is not r2 74f1fb79")
    man = FIT_CAPTURE / "manifest.json"
    if sha(man) != FIT_MANIFEST_SHA:
        bad.append("fit capture manifest")
    else:
        for rid, e in json.loads(man.read_text())["runs"].items():
            if sha(FIT_CAPTURE / e["file"]) != e["file_sha256"]:
                bad.append(f"fit capture file {rid}")
    if bad:
        raise Failed(f"identities differ from confirm-qual-03: {bad[:10]}")
    out["orchestrator"] = {"path": "scripts/gwelem_confirm.py", "sha256": sha(ROOT / "scripts/gwelem_confirm.py")}
    out["fit_capture_manifest"] = {"path": str(man.relative_to(ROOT)), "sha256": FIT_MANIFEST_SHA}
    out["runner_used"] = {"path": str(cfg["runner"]), "sha256": sha(cfg["runner"])}
    return out


def protected():
    import gwelem_qualify as Q
    return {str(p.relative_to(ROOT)): sha(p) for p in Q.PROTECTED + [QUAL / "qualification.json", QUAL / "bound-files.json"]}


def prepare(out: Path, cfg=None):
    cfg = cfg or default_cfg()
    if out.exists():
        raise SystemExit(f"{out} exists")
    if Path(cfg["global_marker"]).exists():
        raise SystemExit("FAIL CLOSED: the confirmation has already been opened")
    if cfg["label"] == "confirmation":
        SPENT.refuse_if_spent(ROOT / "docs/gw-element-1-confirmation-protocol.md", cfg["targets"])
    out.mkdir(parents=True)
    pops = D.verify_populations()
    files = verify_identities(cfg)
    inv = D.inventory(tuple(cfg["targets"]))
    if cfg["label"] == "confirmation" and (inv["arms_total"], inv["identity_arms"], inv["constructed_or_reference_arms"]) != (200, 100, 100):
        raise SystemExit(f"FAIL CLOSED: inventory {inv}")
    b = {"schema": "larql.gwelem.confirm-binding.v1", "status": "bound", "label": cfg["label"],
         "authorisation": "owner GO 2026-11-21: orchestrator, synthetic integration qualification, then ONE sealed "
                          "confirmation execution under protocol r2 (74f1fb79)",
         "files": files, "targets": list(cfg["targets"]), "populations": pops, "inventory": inv,
         "protected": protected(), "power_policy": "log_only", "global_marker": str(cfg["global_marker"])}
    (out / "binding.json").write_text(json.dumps(b, indent=1))
    print("bound", sha(out / "binding.json"), json.dumps({k: inv[k] for k in ("arms_total", "identity_arms")}))
    return b


class Exec:
    def __init__(self, out: Path, cfg: dict, hooks: dict):
        self.out, self.cfg, self.hooks = out, cfg, hooks
        self.integ = {"steps": [], "accounting": None, "power_events": []}
        self.guard = hr.PowerGuard(enforce=False)
        self.t0 = time.time()

    def step(self, name, **detail):
        self.integ["steps"].append({"step": name, "t": round(time.time() - self.t0, 1)} | detail)
        self.save("in progress")
        print(f"[{name}] {json.dumps(detail)[:240]}", flush=True)

    def save(self, status):
        self.integ["status"] = status
        self.integ["power_events"] = self.guard.events
        self.integ["wall_seconds"] = round(time.time() - self.t0, 1)
        (self.out / "integrity.json").write_text(json.dumps(self.integ, indent=1) + "\n")

    def model(self, label, args) -> int:
        with open(self.out / f"{label}-stderr.txt", "w") as err:
            return self.guard.run([str(a) for a in args], stderr=err, label=label)


def execute(out: Path, go_file: Path, cfg=None, hooks=None) -> dict:
    cfg, hooks = cfg or default_cfg(), hooks or {}
    b = json.loads((out / "binding.json").read_text())
    E = Exec(out, cfg, hooks)
    # ── before anything sealed: refuse cleanly, consuming nothing ──
    marker = Path(cfg["global_marker"])
    if marker.exists():
        raise SystemExit("FAIL CLOSED: the confirmation has already been opened; no second execution")
    if cfg["label"] == "confirmation":
        SPENT.refuse_if_spent(ROOT / "docs/gw-element-1-confirmation-protocol.md", cfg["targets"])
    if (out / "integrity.json").exists() or (out / "EXECUTION-STARTED").exists():
        raise SystemExit("FAIL CLOSED: this directory has already executed")
    go = Path(go_file).read_text().strip()
    if not go:
        raise SystemExit("FAIL CLOSED: empty GO record")
    if hooks.get("alter_binding"):
        hooks["alter_binding"](out)
    now = verify_identities(cfg)
    if any(now[k]["sha256"] != b["files"][k]["sha256"] for k in b["files"]) or json.loads((out / "binding.json").read_text()) != b \
            or list(cfg["targets"]) != b["targets"]:
        raise SystemExit("FAIL CLOSED: a bound file changed since prepare")
    # ── open the confirmation: exclusive global marker, then never again ──
    # Durable before the first protected operation: the marker file and its directory entry are fsync'd, so
    # refusal of any later execution depends only on this file existing, not on exception handling. It
    # survives SIGKILL and power loss once this block returns.
    import os
    with open(marker, "x") as f:
        f.write(json.dumps({"opened": time.strftime("%Y-%m-%dT%H:%M:%S"), "out": str(out),
                            "binding_sha256": sha(out / "binding.json"), "go_sha256": sha(go_file)}))
        f.flush()
        os.fsync(f.fileno())
    dfd = os.open(marker.parent, os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)
    (out / "EXECUTION-STARTED").write_text(sha(out / "binding.json"))
    try:
        rep = _execute(E, b, go, go_file)
    except Failed as e:
        E.integ["failure"] = str(e)
        E.save(f"FAILED: {e}"[:200])
        print("FAILED:", e)
        return E.integ
    except BaseException as e:  # any unexpected error is recorded as a failure, never left "in progress"
        import traceback
        last = E.integ["steps"][-1]["step"] if E.integ["steps"] else "start"
        E.integ["failure"] = f"unexpected {type(e).__name__} after step '{last}': {e}"
        E.integ["traceback"] = traceback.format_exc()[-4000:]
        E.save(f"FAILED: unexpected {type(e).__name__} after step '{last}'")
        print("FAILED:", E.integ["failure"])
        return E.integ
    return rep


def _execute(E: Exec, b: dict, go: str, go_file: Path) -> dict:
    out, cfg, hooks = E.out, E.cfg, E.hooks
    tok = bp.tokenizer()
    dec = lambda ids: tok.decode(ids)  # noqa: E731
    targets = list(cfg["targets"])
    meta = P.meta()
    subj = A.sealed_subjects(tok)
    cells = {c["id"]: c["token_ids"] for c in P.baseline()[0]["cells"]}
    rs = P.runs(targets, dec, release=True)
    if len(rs) != 4 * len(targets):
        raise Failed("run count")
    E.step("runs", runs=len(rs))
    # 1. seal
    (out / "seal-plan.json").write_text(json.dumps(P.seal_plan(meta["container"], rs), indent=1))
    if E.model("seal", [ROOT / b["files"]["exe_seal"]["path"], out / "seal-plan.json", out / "seal.jsonl"]):
        raise Failed("seal process")
    seal_rows = load_jsonl(out / "seal.jsonl")
    bad = [s["id"] for s in seal_rows if s["step_logit_sha256"] != rs[s["id"]]["expected_step_logit_sha256"]
           or s["step_logit_sha256"][-1] != rs[s["id"]]["baseline_next_logit_sha256"]]
    if bad or len(seal_rows) != len(rs):
        raise Failed(f"seal differs from the baseline: {bad[:5]}")
    E.step("seal_equals_baseline", runs=len(seal_rows))
    # 2. captures: number and symbol into separate directories
    others = {D.key(z) for z in range(1, 119)} - {D.key(z) for z in targets}
    refusal = sorted(others | set(badm.SPENT_HOLDOUT))
    caps_dirs = {}
    for rel in D.RELATIONS:
        sub = {k: v for k, v in rs.items() if k.startswith(rel + "|")}
        cp = P.capture_plan(meta, sub, [s for s in seal_rows if s["id"] in sub], refusal,
                            {"seal_sha256": sha(out / "seal.jsonl"), "purpose": f"GW-ELEMENT-1 {cfg['label']} {rel}"})
        pf = out / f"capture-plan-{rel}.json"
        pf.write_text(json.dumps(cp, indent=1))
        d = out / f"capture-{rel}"
        if E.model(f"capture-{rel}", [ROOT / b["files"]["exe_capture"]["path"], "capture", pf, d]):
            raise Failed(f"capture {rel} process")
        if hooks.get("partial_capture") == rel:
            e0 = next(iter(json.loads((d / "manifest.json").read_text())["runs"].values()))
            (d / e0["file"]).write_bytes((d / e0["file"]).read_bytes()[:-4])
        sa = crun.seal_agreement(d, cp)
        if sa["failures"]:
            raise Failed(f"capture {rel} differs from the seal: {sa['failures'][:3]}")
        vp = out / f"verify-plan-{rel}.json"
        vp.write_text(json.dumps({"schema": "larql.gwtemplate.test3-capture-verify.v1",
                                  "capture_manifest_sha256": sha(d / "manifest.json"),
                                  "reinject": sorted(cp["runs"]), "arms": [], "sources": {}}, indent=1))
        if E.model(f"verify-{rel}", [ROOT / b["files"]["exe_capture"]["path"], "verify", pf, d, vp, out / f"verify-{rel}.json"]):
            raise Failed(f"verify {rel} process")
        rein = json.loads((out / f"verify-{rel}.json").read_text())["reinject"]
        if sorted(rein) != sorted(sub) or not all(v["bit_identical"] for v in rein.values()):
            raise Failed(f"reinjection {rel}")
        caps_dirs[rel] = d
        E.step(f"capture_{rel}", runs=len(sub), vectors_checked=sa["vectors_checked"], reinject_bit_identical=len(rein))
    # 3. build: candidate view = fit + target NUMBER runs only; references = number + symbol
    fit_ids = {D.run_id(r, p, z) for z in D.FIT for r in D.RELATIONS for p in D.PHRASINGS}
    num_ids = {k for k in rs if k.startswith(D.SOURCE + "|")}
    sym_ids = {k for k in rs if k.startswith(D.INJECTED + "|")}
    cand_dirs = [FIT_CAPTURE, caps_dirs[D.SOURCE]] + ([caps_dirs[D.INJECTED]] if hooks.get("leak_view") else [])
    try:
        cview = C.CandidateView(C.load_windows(cand_dirs, fit_ids | num_ids | (sym_ids if hooks.get("leak_view") else set())),
                                D.FIT, targets)
    except C.InformationRuleError as e:
        raise Failed(f"information rule at build: {e}")
    rview = C.ReferenceView(C.load_windows([caps_dirs[D.SOURCE], caps_dirs[D.INJECTED]], num_ids | sym_ids))
    store = C.Store(P.HIDDEN)
    tgts = C.build_targets(rs, targets, cview, rview, store, D.ARMS)
    if any(r.startswith(D.INJECTED + "|") and int(r.split("|")[2]) in targets for r in cview.reads):
        raise Failed("a candidate read a target symbol run")
    sf = out / "store.f32"
    ssha = store.write(sf)
    if hooks.get("tamper_store"):
        data = bytearray(sf.read_bytes())
        data[16] ^= 0x01
        sf.write_bytes(bytes(data))
        ssha = sha(sf)
    plan = P.band_plan(meta, rs, tgts, sf, ssha, store.index, meta["candidates"], refusal)
    auth = out / "authorisation.json"
    auth.write_text(json.dumps({"schema": AUTH_SCHEMA, "population": D.SEALED_POPULATION, "owner_go": go,
                                "go_record_sha256": sha(go_file), "runs_digest": A.runs_digest(plan["runs"]),
                                "binding_sha256": sha(out / "binding.json")}, indent=1))
    plan["sealed_release"] = {"population": D.SEALED_POPULATION, "authorisation_file": str(auth), "authorisation_sha256": sha(auth)}
    (out / "plan.json").write_text(json.dumps(plan, indent=1))
    (out / "index.json").write_text(json.dumps(store.index))
    E.step("built", targets=len(tgts), arms=sum(len(t["arms"]) for t in tgts), vectors=len(store.index))
    # 4. pre-run admission, audit, inventory, static check
    caps_all = A.Caps([FIT_CAPTURE, caps_dirs[D.SOURCE], caps_dirs[D.INJECTED]])
    caps_c2 = A.Caps([FIT_CAPTURE, caps_dirs[D.SOURCE]])
    caps_ref = A.Caps([caps_dirs[D.SOURCE], caps_dirs[D.INJECTED]])
    pre = A.admit_plan(plan, store.index, caps_all, dec, subj, cells, release=True)
    aud = A.audit_c2(plan, store.index, caps_c2, caps_ref)
    cov = AN.check_coverage(tgts, targets)
    E.integ["pre_admission"] = {"failures": pre[:50], "n": len(pre)}
    E.integ["audit_c2"] = {"recompute_vectors": aud["recompute"]["vectors"], "recompute_failures": aud["recompute"]["failures"][:20],
                           "dependency": aud["dependency"], "c2_caps_symbol_leaks": aud["c2_caps_symbol_leaks"]}
    if pre or not aud["pass"] or cov:
        raise Failed(f"pre-run admission/audit/coverage: {(pre + cov)[:3]} audit_pass={aud['pass']}")
    chk = run([ROOT / b["files"]["exe_elem"]["path"], "--check", out / "plan.json"])
    if chk.returncode:
        raise Failed(f"runner --check: {chk.stderr[-300:]}")
    if cfg["expect_sealed_targets"]:
        noref = dict(plan)
        noref.pop("sealed_release")
        (out / "plan-without-release.json").write_text(json.dumps(noref))
        r2 = run([ROOT / b["files"]["exe_elem"]["path"], "--check", out / "plan-without-release.json"])
        if r2.returncode == 0 or "sealed population" not in r2.stderr:
            raise Failed("the seal did not refuse the plan without its release")
    E.step("pre_run_admitted", vectors=aud["recompute"]["vectors"], check=chk.stdout.strip()[:120])
    # 5. runner (consumes the single-use release before loading the model)
    o = out / "out.jsonl"
    rc = E.model("runner", [cfg["runner"], *cfg["runner_args"], out / "plan.json", o])
    used = Path(str(auth) + ".used").exists()
    rows = load_jsonl(o) if o.exists() else (load_jsonl(str(o) + ".partial") if Path(str(o) + ".partial").exists() else [])
    acct = account(plan, rows)
    E.integ["accounting"] = acct
    E.integ["release_consumed"] = used
    if rc:
        raise Failed(f"runner exited {rc}; {acct['ok']}/{acct['planned']} arms recorded before the failure")
    # 6. post-run integrity
    man = json.loads((Path(str(o) + ".manifest.json")).read_text())
    post = []
    if man.get("status") != "complete" or man.get("executable_sha256") != sha(cfg["runner"]) \
            or man.get("plan_sha256") != sha(out / "plan.json") or man.get("output_sha256") != sha(o):
        post.append("manifest")
    if (man.get("sealed_release") or {}).get("runs_digest") != A.runs_digest(plan["runs"]) or not used:
        post.append("release not recorded or not consumed")
    post += badm.admit_records(plan, store.index, rows, caps_all)
    post += sadm.admit_subspan_records(plan, rows, caps_all)
    recs_b = P.baseline()[1]
    space = tok.encode(" ", add_special_tokens=False).ids[0]
    sym_tok = {c["id"]: tok.encode(" " + c["symbol"], add_special_tokens=False).ids[0] for c in P.baseline()[0]["cells"]
               if c["id"] in rs}
    nat_bad = []
    for t, r in zip(plan["targets"], rows):
        nr = r["natural"]["readout"]
        if nr["next_logit_sha256"] != recs_b[t["run"]]["readout"]["next_logit_sha256"]:
            nat_bad.append(f"{t['run']}: natural readout differs from the baseline")
        want = space if t["run"].startswith(D.SOURCE) else sym_tok[t["run"]]
        if nr["next"] != want:
            nat_bad.append(f"{t['run']}: natural top-1")
    post += nat_bad
    prot = protected() == b["protected"]
    if not prot:
        post.append("protected records changed")
    E.integ["post_admission"] = {"failures": post[:50], "n": len(post)}
    E.integ["identity"] = [acct["identity_ok"], acct["identity_planned"]]
    admitted = not post and acct["ok"] == acct["planned"] and acct["identity_ok"] == acct["identity_planned"]
    E.step("post_run", admitted=admitted, failures=len(post), ok=acct["ok"], planned=acct["planned"])
    if not admitted:
        E.save("NOT admitted: no analysis")
        return E.integ
    E.save("admitted")
    # 7. the single evaluation
    import gwelem_feasibility as fz
    T = AN.Tokens(tok, meta["candidates"], fz.ELEMENTS, list(D.FIT + D.DEVELOPMENT))
    rep = AN.analyse(plan["targets"], rows, T, expected=targets)
    rep["schema"] = "larql.gwelem.confirmation-report.v1"
    rep["label"] = cfg["label"]
    rep["protocol_sha256"] = PROTOCOL_SHA
    rep["disclosure"] = ("Confirmation elements were evaluated naturally for eligibility (baseline-01); never captured, "
                         "fitted on or intervened on before this execution. Only window slots 2-3 are numerically changed "
                         "(slots 0-1 are relation-identical at L4), so a pass supports a portable relation-state "
                         "intervention under these prompts, elements and conditions.")
    (out / "report.json").write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n")
    (out / "CONFIRMATION-SPENT.txt").write_text(f"{cfg['label']} executed once; report sha {sha(out / 'report.json')}\n")
    E.step("analysed", verdict=rep["gates"]["primary"]["verdict"])
    E.save("admitted; analysed")
    return rep


def account(plan: dict, rows: list[dict]) -> dict:
    """Every planned (target, arm) with a status; nothing silently excluded."""
    by = {r["id"]: r for r in rows}
    items, ok, iok, iplan = [], 0, 0, 0
    for t in plan["targets"]:
        r = by.get(t["id"])
        for a in t["arms"]:
            iplan += bool(a.get("identity"))
            if r is None or a["name"] not in r.get("arms", {}):
                items.append({"target": t["id"], "arm": a["name"], "status": "missing"})
                continue
            x = r["arms"][a["name"]]
            valid = (x.get("replacements") == x.get("firings") == len(a["target_span"])
                     and (not a.get("identity") or x.get("identity_bit_exact") is True))
            items.append({"target": t["id"], "arm": a["name"], "status": "ok" if valid else "invalid"})
            ok += valid
            iok += valid and bool(a.get("identity"))
    extra = sorted(set(by) - {t["id"] for t in plan["targets"]})
    return {"planned": len(items), "ok": ok, "missing": [i for i in items if i["status"] == "missing"],
            "invalid": [i for i in items if i["status"] == "invalid"], "extra_records": extra,
            "identity_planned": iplan, "identity_ok": iok}


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) == 2 and a[0] == "prepare":
        prepare(Path(a[1]).resolve())
    elif len(a) == 3 and a[0] == "execute":
        r = execute(Path(a[1]).resolve(), Path(a[2]).resolve())
        print(json.dumps(r.get("gates", r.get("status")), indent=1)[:3000])
    else:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
