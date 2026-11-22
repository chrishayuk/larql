"""GW-ELEMENT-1 confirmation: engineering qualification (QE0–QE12). No constructed-state inference on elements.

    gwelem_qualify.py OUT_DIR

Owner GO 2026-11-21: "confirmation-protocol freezing, engineering and independent qualification only".
OUT_DIR must not exist. Power: log only. The confirmation population is never read: every element
plan, capture and runner invocation here carries the sealed-split refusal.

**Model processes, and what they compute:**
- `gwbreadcrumb2_seal` (unchanged) and `gwtemplate_test3_capture` (unchanged): **natural** carriers of
  the 45 fit and 15 development elements;
- `gwelem_band_runner` (new):
  - **identity** arms on development runs (a target's own natural carriers re-injected);
  - **historical** capital-domain plans from CP3a, CP3b, CP3c and band-qual-02, whose records it must
    reproduce bit-exactly.

**Development C2 vectors** are built, admitted, recomputed and audited, and pass the runner's `--check`.
They are **never executed**. No switching, margin or B3 statistic is computed for any element arm.

| Gate | Check |
|---|---|
| QE0 | identities: seal, capture and historical band-runner binaries equal their qualified hashes; new runner and sources hashed; model files hashed |
| QE1 | populations re-derived (45/15/25, sealed 35); confirmation inventory exactly 200 arms = 100 identity + 100 C2/S (protocol r2) |
| QE2 | unit tests: element confirmation (synthetic), parser, historical band admission |
| QE3 | the runner's compiled seal equals the independent re-derivation (35 keys, 140 subject token sequences in 4 surface forms, 28 spent); QE3b: every plan given to the seal/capture binaries passes the orchestrator seal |
| QE4 | seal of 240 natural fit/dev runs: prompt steps = baseline, last step = baseline readout hash |
| QE5 | captures (fit / dev-number / dev-symbol, separate dirs): capture = seal, bit-exact reinjection of every run |
| QE6 | development C2 plan (the r2 confirmation inventory on dev elements): admission (incl. "differs from natural at a designated slot"), recompute (C2 without any symbol capture of a target), dependency audit, disclosed slot equivalence (slots 0-1 == natural, 2-3 differ), `--check`; NOT executed |
| QE7 | development identity runs (full window, L4): identity bit-exact, coverage, locality (band + sub-window admission), natural readouts = baseline hashes |
| QE8 | historical replication: sampled CP3a/CP3b/CP3c/band-qual-02 targets re-run by the new runner equal the original records byte for byte |
| QE9 | runner refusals: sealed key, sealed content under a clean key, malformed releases, spent holdout with and without a valid release; a valid release lifts only the seal; band hygiene refusals |
| QE10 | admission refusals on the real development plan: own-symbol input, non-fit member, sealed key, inventory, address |
| QE11 | sealed populations never touched: no confirmation-split run id in any capture, plan or record produced here |
| QE12 | protected records unchanged before and after |
"""
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import gwcompress_band_admit as badm  # noqa: E402
import gwcompress_subspan_admit as sadm  # noqa: E402
import gwelem_admit as A  # noqa: E402
import gwelem_candidates as C  # noqa: E402
import gwelem_design as D  # noqa: E402
import gwelem_plan as P  # noqa: E402
import gwtemplate_baseline_plan as bp  # noqa: E402
import gwtemplate_test3_capture_run as crun  # noqa: E402
import gwtemplate_test3_holdout_run as hr  # noqa: E402

GF = ROOT / "output/gw-template-factorial"
EL = ROOT / "output/gw-element-1"
EXE = {"seal": "target/release/examples/gwbreadcrumb2_seal",
       "capture": "target/release/examples/gwtemplate_test3_capture",
       "band": "target/release/examples/gwcompress_band_runner",
       "elem": "target/release/examples/gwelem_band_runner"}
QUALIFIED = {"seal": "sha256:0c99ef50c20948d4fb08a52582edd32e78205b32ce8e1a5d9f0be8e13ab9a9b2",
             "capture": "sha256:84e39352e5ccdc5df949253cfa959c6cc3586fea4c7a975e64f6159e2dd5c035",
             "band": "sha256:3ac83622bb339f471eebea86590de65b8613c01eea2cfcaec34f3213062a0d2c",
             "band_source": "sha256:e8f7de1c9adf10e3429eeb5d0afd398ac48f564912341fc381bef8ec4050af09"}
MODEL = Path("/Users/christopherhay/chris-models/gemma3-4b-it.vindex3")
FILES = {
    "protocol": "docs/gw-element-1-confirmation-protocol.md",
    "design": "scripts/gwelem_design.py", "plan_builder": "scripts/gwelem_plan.py",
    "candidates": "scripts/gwelem_candidates.py", "admission": "scripts/gwelem_admit.py",
    "analysis": "scripts/gwelem_analyze.py", "tests": "scripts/test_gwelem_confirm.py",
    "harness": "scripts/gwelem_qualify.py", "parser": "scripts/gwelem_parse.py",
    "runner_source": "crates/larql-demos/examples/inference/gwelem_band_runner.rs",
    "band_runner_source": "crates/larql-demos/examples/inference/gwcompress_band_runner.rs",
    "band_admission": "scripts/gwcompress_band_admit.py", "subspan_admission": "scripts/gwcompress_subspan_admit.py",
    "capture_harness": "scripts/gwtemplate_test3_capture_run.py",
    "feasibility": "output/gw-element-1/feasibility-01/feasibility.json",
    "baseline_plan": "output/gw-element-1/baseline-01/plan.json",
    "baseline_out": "output/gw-element-1/baseline-01/out.jsonl",
    "baseline_report": "output/gw-element-1/baseline-01/report.json",
}
for _k, _p in EXE.items():
    FILES[f"exe_{_k}"] = _p
PROTECTED = [GF / "test3-stage2-02/binding.json", GF / "test3-stage2-02/report.json",
             GF / "test3-stage4-02/binding.json", GF / "test3-stage4-02/HOLDOUT-SPENT.txt",
             GF / "test3-stage4-02/report.json", GF / "test3-c2ab-01/binding.json",
             GF / "compress1-band-qual-02/qualification.json"]
for _c in ("compress1-cp3a-01", "compress1-cp3b-01", "compress1-cp3c-01"):
    PROTECTED += [GF / _c / f for f in ("binding.json", "integrity.json", "report.json")]
PROTECTED += [EL / "feasibility-01/feasibility.json"] + [EL / "baseline-01" / f for f in
                                                          ("binding.json", "integrity.json", "report.json", "out.jsonl", "plan.json")]
HIST = [("compress1-cp3a-01", s) for s in ("capital-P2", "capital-P3")] + \
       [("compress1-cp3b-01", s) for s in ("capital-P2", "capital-P3")] + \
       [("compress1-cp3c-01", s) for s in ("capital-P2", "capital-P3")] + \
       [("compress1-band-qual-02", f"qb2-{s}") for s in ("capital-P2", "capital-P3")]
HIST_SALT = "gw-element-1/runner-qual-sample/v1|"
ID_ARMS = {"A2": ("ID-full-L4",), "A3": ("ID-full-L4",)}


def sha(p) -> str:
    return "sha256:" + hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_jsonl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def run(args):
    return subprocess.run([str(a) for a in args], capture_output=True, text=True, cwd=ROOT)


class Q:
    def __init__(self, out: Path):
        self.out, self.gates, self.t0 = out, {}, time.time()
        self.guard = hr.PowerGuard(enforce=False)

    def gate(self, name, ok, **detail):
        self.gates[name] = {"pass": bool(ok)} | detail
        print(f"{name}: {'PASS' if ok else 'FAIL'} {json.dumps(detail)[:300]}", flush=True)
        self.save("in progress")

    def model(self, label, args):
        with open(self.out / f"{label}-stderr.txt", "w") as err:
            rc = self.guard.run([str(a) for a in args], stderr=err, label=label)
        return rc

    def save(self, status):
        rec = {"schema": "larql.gwelem.confirm-qualification.v1", "status": status,
               "authorisation": "owner GO 2026-11-21: confirmation-protocol freezing, engineering and independent "
                                "qualification only; no constructed-state inference on elements",
               "gates": self.gates, "power_policy": "log_only", "power_events": self.guard.events,
               "wall_seconds": round(time.time() - self.t0, 1)}
        (self.out / "qualification.json").write_text(json.dumps(rec, indent=1) + "\n")


def main(out: Path):
    if out.exists():
        raise SystemExit(f"{out} exists")
    out.mkdir(parents=True)
    q = Q(out)
    tok = bp.tokenizer()
    dec = lambda ids: tok.decode(ids)  # noqa: E731

    # QE0 identities
    files = {k: {"path": p, "sha256": sha(ROOT / p)} for k, p in FILES.items()}
    files["band_runner_source_frozen"] = files.pop("band_runner_source")
    for f in ("index.json", "system_graph.json", "tokenizer.json"):
        files[f"model_{f}"] = {"path": str(MODEL / f), "sha256": sha(MODEL / f)}
    ok = (files["exe_seal"]["sha256"] == QUALIFIED["seal"] and files["exe_capture"]["sha256"] == QUALIFIED["capture"]
          and files["exe_band"]["sha256"] == QUALIFIED["band"]
          and files["band_runner_source_frozen"]["sha256"] == QUALIFIED["band_source"]
          and files["model_tokenizer.json"]["sha256"] == bp_tok_sha())
    protected_before = {str(p.relative_to(ROOT)): sha(p) for p in PROTECTED}
    q.gate("QE0_identities", ok, files=files, note="seal/capture/band binaries rebuilt from unchanged sources after "
           "the worktree target/ deletion; each reproduced its qualified hash")
    (out / "bound-files.json").write_text(json.dumps(files, indent=1))

    # QE1 populations and inventory
    pops = D.verify_populations()
    inv = D.inventory()
    q.gate("QE1_populations_inventory", inv["arms_total"] == 200 and inv["identity_arms"] == 100
           and inv["constructed_or_reference_arms"] == 100, populations={k: len(v) if isinstance(v, list) else v
                                                                         for k, v in pops.items()}, inventory=inv)

    # QE2 tests
    tests = {}
    for t in ("scripts/test_gwelem_confirm.py", "scripts/test_gwelem_parse.py", "scripts/test_gwcompress_band.py"):
        r = run([sys.executable, "-m", "unittest", t])
        tests[t] = {"rc": r.returncode, "tail": (r.stderr.strip().splitlines() or ["<no output>"])[-1]}
    q.gate("QE2_unit_tests", all(v["rc"] == 0 for v in tests.values()), tests=tests)

    # QE3 compiled seal
    r = run([ROOT / EXE["elem"], "--sealed"])
    seal_c = json.loads(r.stdout)
    subj = A.sealed_subjects(tok)
    ok = (seal_c["population"] == D.SEALED_POPULATION and seal_c["keys"] == [D.key(z) for z in D.CONFIRMATION_SPLIT]
          and {(s["z"], s["form"]): s["tokens"] for s in seal_c["subjects"]} == subj and len(subj) == 140
          and sorted(seal_c["spent_holdout"]) == sorted(badm.SPENT_HOLDOUT))
    q.gate("QE3_compiled_seal", ok, keys=len(seal_c["keys"]), subjects=len(seal_c["subjects"]),
           spent=len(seal_c["spent_holdout"]))

    # model-free preparation
    meta = P.meta()
    rs = P.runs(D.FIT + D.DEVELOPMENT, dec)
    refusal = sorted({D.key(z) for z in D.CONFIRMATION_SPLIT} | set(badm.SPENT_HOLDOUT))
    (out / "seal-plan.json").write_text(json.dumps(P.seal_plan(meta["container"], rs), indent=1))
    cells = {c["id"]: c["token_ids"] for c in P.baseline()[0]["cells"]}
    # QE3b orchestrator seal: the seal and capture binaries carry no element seal, so every plan handed
    # to them is checked here, independently of the plan builder (reviewer finding I2).
    sp = json.loads((out / "seal-plan.json").read_text())
    sp_runs = {r["id"]: {"country": r["id"].split("|")[-1], "tokens": r["tokens"]} for r in sp["runs"]}
    sf = A.seal_check({"runs": sp_runs}, subj)
    q.gate("QE3b_orchestrator_seal", not sf and len(sp_runs) == 240, runs=len(sp_runs), failures=sf[:5])
    if sf:
        return q.save("FAILED")

    # QE4 seal
    rc = q.model("seal", [ROOT / EXE["seal"], out / "seal-plan.json", out / "seal.jsonl"])
    seal_rows = load_jsonl(out / "seal.jsonl") if rc == 0 else []
    bad = [s["id"] for s in seal_rows if s["step_logit_sha256"] != rs[s["id"]]["expected_step_logit_sha256"]
           or s["step_logit_sha256"][-1] != rs[s["id"]]["baseline_next_logit_sha256"]]
    q.gate("QE4_seal_equals_baseline", rc == 0 and len(seal_rows) == 240 and not bad, runs=len(seal_rows), mismatches=bad[:10])
    if rc:
        return q.save("FAILED")

    # QE5 captures
    groups = {"fit": [r for r in rs if int(r.split("|")[2]) in D.FIT],
              "dev-number": [r for r in rs if int(r.split("|")[2]) in D.DEVELOPMENT and r.startswith("number")],
              "dev-symbol": [r for r in rs if int(r.split("|")[2]) in D.DEVELOPMENT and r.startswith("symbol")]}
    cap_res = {}
    for g, ids in groups.items():
        sub = {k: rs[k] for k in ids}
        cp = P.capture_plan(meta, sub, [s for s in seal_rows if s["id"] in sub], refusal,
                            {"seal_sha256": sha(out / "seal.jsonl"), "purpose": f"GW-ELEMENT-1 qualification {g}"})
        if A.seal_check(cp, subj):
            return q.save("FAILED")
        (out / f"capture-plan-{g}.json").write_text(json.dumps(cp, indent=1))
        rc1 = q.model(f"capture-{g}", [ROOT / EXE["capture"], "capture", out / f"capture-plan-{g}.json", out / f"capture-{g}"])
        sa = crun.seal_agreement(out / f"capture-{g}", cp) if rc1 == 0 else {"failures": ["capture rc"]}
        (out / f"verify-plan-{g}.json").write_text(json.dumps(
            {"schema": "larql.gwtemplate.test3-capture-verify.v1", "capture_manifest_sha256": sha(out / f"capture-{g}/manifest.json"),
             "reinject": sorted(cp["runs"]), "arms": [], "sources": {}}, indent=1))
        rc2 = q.model(f"verify-{g}", [ROOT / EXE["capture"], "verify", out / f"capture-plan-{g}.json", out / f"capture-{g}",
                                      out / f"verify-plan-{g}.json", out / f"verify-{g}.json"])
        v = json.loads((out / f"verify-{g}.json").read_text()) if rc2 == 0 else {}
        rein = v.get("reinject", {})
        n_ok = sum(1 for x in (rein.values() if isinstance(rein, dict) else rein)
                   if (x.get("bit_identical") if isinstance(x, dict) else x)) if rein else 0
        cap_res[g] = {"runs": len(ids), "capture_rc": rc1, "seal_failures": sa["failures"][:10],
                      "vectors_checked": sa.get("vectors_checked"), "verify_rc": rc2, "reinject_bit_identical": n_ok}
    ok = all(c["capture_rc"] == 0 and not c["seal_failures"] and c["verify_rc"] == 0 and c["reinject_bit_identical"] == c["runs"]
             for c in cap_res.values())
    q.gate("QE5_capture", ok, groups=cap_res)
    if not ok:
        return q.save("FAILED")
    dirs = {g: out / f"capture-{g}" for g in groups}

    # QE6 development C2 plan (built, admitted, checked; never executed)
    cview = C.CandidateView(C.load_windows([dirs["fit"], dirs["dev-number"]], set(groups["fit"]) | set(groups["dev-number"])),
                            D.FIT, D.DEVELOPMENT)
    rview = C.ReferenceView(C.load_windows([dirs["dev-number"], dirs["dev-symbol"]],
                                           set(groups["dev-number"]) | set(groups["dev-symbol"])))
    store = C.Store(P.HIDDEN)
    targets = C.build_targets(rs, D.DEVELOPMENT, cview, rview, store, D.ARMS)
    ssha = store.write(out / "dev-c2-store.f32")
    plan = P.band_plan(meta, rs, targets, out / "dev-c2-store.f32", ssha, store.index, meta["candidates"], refusal)
    (out / "dev-c2-plan.json").write_text(json.dumps(plan, indent=1))
    (out / "dev-c2-index.json").write_text(json.dumps(store.index))
    caps_all = A.Caps([dirs["fit"], dirs["dev-number"], dirs["dev-symbol"]])
    caps_c2 = A.Caps([dirs["fit"], dirs["dev-number"]])
    pre = A.admit_plan(plan, store.index, caps_all, dec, subj, cells)
    aud = A.audit_c2(plan, store.index, caps_c2, caps_all)
    rec, audit = aud["recompute"], aud["dependency"]
    try:
        cview.read(D.run_id(D.INJECTED, "A2", D.DEVELOPMENT[0]))
        view_refuses = False
    except C.InformationRuleError:
        view_refuses = True
    chk = run([ROOT / EXE["elem"], "--check", out / "dev-c2-plan.json"])
    dinv = D.inventory(D.DEVELOPMENT)
    n_arms = sum(len(t["arms"]) for t in targets)
    # disclosed equivalence (protocol r2): C2 == own natural at slots 0-1, differs at the designated slots 2-3
    data6 = np.fromfile(out / "dev-c2-store.f32", dtype="<f4")
    eq = {}
    for e in store.index.values():
        r = e["recipe"]
        if r["kind"] != "C2":
            continue
        v = data6[e["offset"]:e["offset"] + P.HIDDEN]
        same = A.vsha(v) == caps_all.sha(D.run_id(D.SOURCE, r["phrasing"], int(r["target"])), r["layer"], r["slot"])
        k = f"{r['phrasing']}|slot{r['slot']}"
        eq.setdefault(k, [0, 0])
        eq[k][0] += same
        eq[k][1] += 1
    slot_eq = {"c2_equals_natural": eq,
               "as_disclosed": all(v[0] == v[1] if int(k[-1]) not in D.EFFECTIVE_SLOTS else v[0] == 0 for k, v in eq.items())}
    q.gate("QE6_dev_c2_plan_not_executed", not pre and aud["pass"] and chk.returncode == 0
           and n_arms == dinv["arms_total"] == 120 and view_refuses and slot_eq["as_disclosed"],
           admission_failures=pre[:10], recompute=rec["vectors"], recompute_failures=rec["failures"][:5], audit=audit,
           runner_check=chk.stdout.strip() or chk.stderr[-300:], arms=n_arms, executed=False,
           c2_caps_symbol_leaks=aud["c2_caps_symbol_leaks"], candidate_view_refuses_target_symbol=view_refuses,
           slot_equivalence=slot_eq)

    # QE7 development identity runs
    store_i = C.Store(P.HIDDEN)
    idt = C.build_targets(rs, D.DEVELOPMENT, None, rview, store_i, ID_ARMS, reference_arms=ID_ARMS)
    issha = store_i.write(out / "dev-id-store.f32")
    iplan = P.band_plan(meta, rs, idt, out / "dev-id-store.f32", issha, store_i.index, meta["candidates"], refusal)
    (out / "dev-id-plan.json").write_text(json.dumps(iplan, indent=1))
    ipre = A.admit_plan(iplan, store_i.index, caps_all, dec, subj, cells, arms=ID_ARMS, reference_arms=ID_ARMS)
    chk = run([ROOT / EXE["elem"], "--check", out / "dev-id-plan.json"])
    rc = q.model("dev-id", [ROOT / EXE["elem"], out / "dev-id-plan.json", out / "dev-id-out.jsonl"]) if chk.returncode == 0 else 1
    rows = load_jsonl(out / "dev-id-out.jsonl") if rc == 0 else []
    man = json.loads((out / "dev-id-out.jsonl.manifest.json").read_text()) if rc == 0 else {}
    post = badm.admit_records(iplan, store_i.index, rows, caps_all) + sadm.admit_subspan_records(iplan, rows, caps_all) if rows else ["no records"]
    plan_b, recs_b = P.baseline()
    ident = [0, 0]
    cov = [0, 0]
    nat = {"baseline_hash_equal": 0, "number_space": 0, "symbol_own": 0, "of": 0}
    space = tok.encode(" ", add_special_tokens=False).ids[0]
    for t, r in zip(iplan["targets"], rows):
        rel, ph, z = t["run"].split("|")
        nr = r["natural"]["readout"]
        nat["of"] += 1
        nat["baseline_hash_equal"] += nr["next_logit_sha256"] == recs_b[t["run"]]["readout"]["next_logit_sha256"]
        if rel == "number":
            nat["number_space"] += nr["next"] == space
        else:
            sym = next(c["symbol"] for c in plan_b["cells"] if c["id"] == t["run"])
            nat["symbol_own"] += nr["next"] == tok.encode(" " + sym, add_special_tokens=False).ids[0]
        for a in t["arms"]:
            x = r["arms"][a["name"]]
            ident[1] += 1
            ident[0] += bool(x["identity_bit_exact"])
            cov[1] += 1
            cov[0] += x["replacements"] == x["firings"] == len(a["target_span"])
    n_id = sum(len(t["arms"]) for t in idt)
    ok = (rc == 0 and man.get("executable_sha256") == sha(ROOT / EXE["elem"]) and man.get("sealed_release") is None
          and not post and ident == [n_id, n_id] and cov == [n_id, n_id] and nat["baseline_hash_equal"] == nat["of"] == 60
          and nat["number_space"] == 30 and nat["symbol_own"] == 30 and not ipre)
    q.gate("QE7_dev_identity", ok, targets=len(idt), identity=ident, coverage=cov, natural=nat,
           post_admission_failures=post[:10], pre_admission_failures=ipre[:10], manifest_release=man.get("sealed_release"))

    # QE8 historical replication
    hist = {}
    for exp, shard in HIST:
        src = json.loads((GF / exp / f"plan-{shard}.json").read_text())
        old = {r["id"]: line for line in (GF / exp / f"out-{shard}.jsonl").read_text().splitlines() if line.strip()
               for r in [json.loads(line)]}
        tid = min((t["id"] for t in src["targets"]), key=lambda i: hashlib.sha256((HIST_SALT + exp + shard + i).encode()).hexdigest())
        t = next(t for t in src["targets"] if t["id"] == tid)
        p2 = dict(src, targets=[t], runs={t["run"]: src["runs"][t["run"]]})
        pf = out / f"hist-{exp}-{shard}.json"
        pf.write_text(json.dumps(p2))
        o = out / f"hist-{exp}-{shard}.jsonl"
        rc = q.model(f"hist-{exp}-{shard}", [ROOT / EXE["elem"], pf, o])
        new = o.read_text().splitlines() if rc == 0 else []
        hist[f"{exp}/{shard}"] = {"target": tid, "arms": len(t["arms"]), "rc": rc,
                                  "byte_identical": bool(new) and new[0] == old[tid],
                                  "json_equal": bool(new) and json.loads(new[0]) == json.loads(old[tid])}
    q.gate("QE8_historical_replication", all(h["rc"] == 0 and h["byte_identical"] for h in hist.values()), cases=hist,
           arms=sum(h["arms"] for h in hist.values()))

    # QE9 runner refusals (static --check; no inference)
    att = out / "attacks"
    att.mkdir()
    base = json.loads((out / "dev-id-plan.json").read_text())
    rid0 = base["targets"][0]["run"]

    def variant(f):
        p = copy.deepcopy(base)
        f(p)
        return p

    def set_key(k):
        """Set the run key, and take it OFF the plan's own holdout list, so the attack reaches the compiled
        seal / spent list / release logic rather than the plan-list refusal (qual-02 QE9 finding)."""
        def f(p):
            p["runs"][rid0]["country"] = k
            p["holdout"] = [h for h in p["holdout"] if h != k]
        return f

    def auth_for(p, name, **over):
        """A fixture authorisation bound to p's runs. It is used only by static --check, never by an execution."""
        a = {"schema": "larql.gwelem.confirmation-authorisation.v1", "population": D.SEALED_POPULATION,
             "owner_go": "QUALIFICATION FIXTURE ONLY - NOT AN AUTHORISATION; used for a static --check",
             "runs_digest": A.runs_digest(p["runs"])} | over
        f = att / f"auth-{name}.json"
        f.write_text(a if isinstance(a, str) else json.dumps(a))
        return {"population": D.SEALED_POPULATION, "authorisation_file": str(f), "authorisation_sha256": sha(f)}

    def auth_text(p):
        f = att / "auth-notjson.json"
        f.write_text("not json at all")
        return {"population": D.SEALED_POPULATION, "authorisation_file": str(f), "authorisation_sha256": sha(f)}

    def released(name, mut, **over):
        def f(p):
            mut(p)
            p["sealed_release"] = auth_for(p, name, **over)
        return f

    iron = subj[(26, "mid_lower")]
    subj0 = next(c for c in P.baseline()[0]["cells"] if c["id"] == rid0)["subject"]["tokens"]
    if len(subj0) != 1 or len(iron) != 1:
        raise SystemExit("FAIL CLOSED: content attack fixture expects single-token subjects")

    def content(p):
        r = p["runs"][rid0]
        r["tokens"] = r["tokens"][:subj0[0]] + iron + r["tokens"][subj0[0] + 1:]

    multi = next(((z, f), t) for (z, f), t in sorted(subj.items()) if f == "start_cap" and len(t) > 1)

    def content_multi(p):
        p["runs"][rid0]["tokens"] = [2] + multi[1] + p["runs"][rid0]["tokens"][1:]

    def id_part(p):
        p["holdout"] = [h for h in p["holdout"] if h != "26"]
        new = rid0.rsplit("|", 1)[0] + "|26"
        p["runs"][new] = p["runs"].pop(rid0)
        for t in p["targets"]:
            if t["run"] == rid0:
                t["run"] = new

    def reuse(p):
        set_key("26")(p)
        p["sealed_release"] = auth_for(p, "reuse")
        Path(p["sealed_release"]["authorisation_file"] + ".used").write_text("prior use")

    def other_plan(p):
        q2 = copy.deepcopy(p)
        set_key("26")(p)
        q2["runs"][rid0]["country"] = "27"
        p["sealed_release"] = auth_for(p, "other", runs_digest=A.runs_digest(q2["runs"]))

    cases = {
        "sealed_key": (variant(set_key("26")), False, "sealed entity key"),
        "sealed_id_part_clean_key": (variant(id_part), False, "sealed entity key"),
        "sealed_content_clean_key": (variant(content), False, "tokens contain the subject of sealed element 26"),
        "sealed_content_multi_token_start_cap": (variant(content_multi), False, "tokens contain the subject of sealed element"),
        "release_wrong_population": (variant(lambda p: (set_key("26")(p), p.__setitem__("sealed_release", dict(auth_for(p, "pop"), population="gw-element-1/fit")))), False, "another population"),
        "release_wrong_sha": (variant(lambda p: (set_key("26")(p), p.__setitem__("sealed_release", dict(auth_for(p, "sha"), authorisation_sha256="sha256:" + "0" * 64)))), False, "differs from its declared hash"),
        "release_missing_file": (variant(lambda p: (set_key("26")(p), p.__setitem__("sealed_release", dict(auth_for(p, "missing"), authorisation_file=str(att / "absent.json"))))), False, "unreadable"),
        "release_not_json": (variant(lambda p: (set_key("26")(p), p.__setitem__("sealed_release", auth_text(p)))), False, "not JSON"),
        "release_wrong_schema": (variant(released("schema", set_key("26"), schema="larql.other.v1")), False, "authorisation schema"),
        "release_empty_owner_go": (variant(released("empty", set_key("26"), owner_go=" ")), False, "owner_go"),
        "release_for_another_plan": (variant(other_plan), False, "runs_digest"),
        "release_already_used": (variant(reuse), False, "already used"),
        "spent_holdout": (variant(set_key("TO")), False, "spent Test 3 holdout"),
        "spent_holdout_with_valid_release": (variant(released("spent", set_key("TO"))), False, "spent Test 3 holdout"),
        "plan_holdout": (variant(lambda p: (set_key("ZZ")(p), p.__setitem__("holdout", p["holdout"] + ["ZZ"]))), False, "a holdout country"),
        "valid_release_does_not_lift_plan_list": (variant(released("planlist", lambda p: (set_key("26")(p), p.__setitem__("holdout", p["holdout"] + ["26"])))), False, "a holdout country"),
        "valid_release_lifts_seal_only": (variant(released("valid", set_key("26"))), True, "sealed release"),
        "missing_vector": (variant(lambda p: p["targets"][0]["arms"][0]["vectors"].popitem()), False, "do not cover exactly"),
        "span_outside_window": (variant(lambda p: p["targets"][0]["arms"][0].__setitem__("target_span", [0])), False, "outside the window"),
        "band_outside_model": (variant(lambda p: p["targets"][0]["arms"][0].__setitem__("to_layer", 40)), False, "is not inside"),
        "index_hash": (variant(lambda p: next(iter(p["vector_store"]["index"].values())).__setitem__("sha256", "0" * 64)), False, "differs from its index hash"),
    }
    res = {}
    for name, (p, accept, needle) in cases.items():
        f = att / f"{name}.json"
        f.write_text(json.dumps(p))
        r = run([ROOT / EXE["elem"], "--check", f])
        full = (r.stdout + r.stderr).strip()
        res[name] = {"expected": "accept" if accept else "refuse", "rc": r.returncode, "message": full[-300:],
                     "pass": (r.returncode == 0) == accept and needle in full}
    valid_file = Path(cases["valid_release_lifts_seal_only"][0]["sealed_release"]["authorisation_file"])
    check_left_no_marker = not Path(str(valid_file) + ".used").exists()
    fixtures = sorted(att.glob("auth-*.json*"))
    for f in fixtures:
        f.unlink()
    q.gate("QE9_runner_refusals", all(v["pass"] for v in res.values()) and check_left_no_marker, cases=res,
           check_creates_no_used_marker=check_left_no_marker,
           fixture_authorisations_deleted=[f.name for f in fixtures])

    # QE10 admission refusals on the real development plan
    def admit_variant(mut):
        p, idx = copy.deepcopy(plan), copy.deepcopy(store.index)
        mut(p, idx)
        return A.admit_plan(p, idx, caps_all, dec, subj, cells)

    def c2vid(idx):
        return next(v for v, e in idx.items() if e["recipe"]["kind"] == "C2")

    def own_symbol(p, idx):
        v = c2vid(idx)
        r = idx[v]["recipe"]
        o = next(i for i in r["inputs"] if i[0] == "own")
        o[1] = "symbol"
        o[6] = caps_all.sha(f"symbol|{r['phrasing']}|{r['target']}", r["layer"], r["slot"])
        idx[v]["recipe_sha256"] = A.rsha(r)
        p["vector_store"]["index"][v]["recipe_sha256"] = idx[v]["recipe_sha256"]

    def non_fit(p, idx):
        v = c2vid(idx)
        r = idx[v]["recipe"]
        i = next(i for i in r["inputs"] if i[0] == "plus")
        other = D.run_id("symbol", r["phrasing"], D.DEVELOPMENT[0] if r["target"] != D.key(D.DEVELOPMENT[0]) else D.DEVELOPMENT[1])
        i[3] = other.split("|")[2]
        i[6] = caps_all.sha(other, r["layer"], r["slot"])
        idx[v]["recipe_sha256"] = A.rsha(r)
        p["vector_store"]["index"][v]["recipe_sha256"] = idx[v]["recipe_sha256"]

    def address(p, idx):
        v = c2vid(idx)
        idx[v]["recipe"]["slot"] = (idx[v]["recipe"]["slot"] + 1) % 4
        idx[v]["recipe_sha256"] = A.rsha(idx[v]["recipe"])
        p["vector_store"]["index"][v]["recipe_sha256"] = idx[v]["recipe_sha256"]

    acases = {"own_symbol_input": (own_symbol, "own symbol state"), "non_fit_member": (non_fit, "non-fit"),
              "address": (address, "address"),
              "sealed_key": (lambda p, idx: p["runs"][p["targets"][0]["run"]].__setitem__("country", "26"), "sealed confirmation key"),
              "inventory": (lambda p, idx: p["targets"][0]["arms"].pop(), "arm inventory"),
              "non_identity_arm_equal_to_natural": (lambda p, idx: p["targets"][0]["arms"][1].__setitem__(
                  "vectors", dict(p["targets"][0]["arms"][0]["vectors"])), "identical to the natural carrier")}
    ares = {}
    for name, (mut, needle) in acases.items():
        f = admit_variant(mut)
        ares[name] = {"pass": any(needle in x for x in f), "first": f[:2]}
    # the leaky vector: an honest recipe, but the stored vector computed from the target's symbol state
    leak_dir = out / "attacks" / "leak"
    leak_dir.mkdir()
    lp_, lidx = copy.deepcopy(plan), copy.deepcopy(store.index)
    lv = next(v for v, e in lidx.items() if e["recipe"]["kind"] == "C2" and e["recipe"]["slot"] in D.EFFECTIVE_SLOTS)
    lr = lidx[lv]["recipe"]
    leak = caps_all.vec(f"symbol|{lr['phrasing']}|{lr['target']}", lr["layer"], lr["slot"]).astype("<f4")
    leak_distinct = A.vsha(leak) != store.index[lv]["sha256"]  # precondition: the fixture is a real leak
    data = np.fromfile(out / "dev-c2-store.f32", dtype="<f4")
    data[lidx[lv]["offset"]:lidx[lv]["offset"] + P.HIDDEN] = leak
    data.tofile(leak_dir / "store.f32")
    lidx[lv]["sha256"] = A.vsha(leak)
    lp_["vector_store"]["file"] = str(leak_dir / "store.f32")
    lp_["vector_store"]["sha256"] = sha(leak_dir / "store.f32")
    lp_["vector_store"]["index"][lv]["sha256"] = lidx[lv]["sha256"]
    labels = A.admit_plan(lp_, lidx, caps_all, dec, subj, cells)
    la = A.audit_c2(lp_, lidx, caps_c2, caps_all)
    ares["leaky_vector_honest_recipe"] = {"pass": leak_distinct and not la["pass"] and lv in la["recompute"]["failures"],
                                          "slot": lr["slot"], "leak_distinct_from_honest": leak_distinct,
                                          "label_admission_failures": len(labels),
                                          "recompute_failures": la["recompute"]["failures"][:3]}
    (leak_dir / "store.f32").unlink()
    q.gate("QE10_admission_refusals", all(v["pass"] for v in ares.values()), cases=ares)

    # QE11 sealed population untouched
    produced = set()
    for g in groups:
        produced |= set(json.loads((dirs[g] / "manifest.json").read_text())["runs"])
    for pf in (out / "dev-c2-plan.json", out / "dev-id-plan.json", out / "seal-plan.json"):
        d = json.loads(pf.read_text())
        produced |= set(d["runs"]) if isinstance(d["runs"], dict) else {r["id"] for r in d["runs"]}
    sealed_hits = [r for r in produced if r.split("|")[-1] in {D.key(z) for z in D.CONFIRMATION_SPLIT}]
    q.gate("QE11_sealed_untouched", not sealed_hits, runs_examined=len(produced), sealed_hits=sealed_hits)

    # QE12 protected
    after = {str(p.relative_to(ROOT)): sha(p) for p in PROTECTED}
    q.gate("QE12_protected_unchanged", after == protected_before and sha(ROOT / EXE["band"]) == QUALIFIED["band"],
           protected=after)
    q.save("qualified" if all(g["pass"] for g in q.gates.values()) else "NOT qualified")
    print("STATUS", json.loads((out / "qualification.json").read_text())["status"])


def bp_tok_sha():
    plan = json.loads((D.BASELINE / "plan.json").read_text())
    return plan["tokenizer_sha256"]


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
