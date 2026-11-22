"""GW-TEMPLATE-FACTORIAL stage 3: qualification of the cross-position runner.

    gwtemplate_xpos_qualify.py OUT_DIR

Model inference happens only through three executables:

- the runner;
- the generic seal executable;
- the spot-check reproduction.

No test 1 transplant is run. Qualification uses a small number of
development pairs and the step C targets.

**Phases:**

1. **Plans.**
   - **The template qualification plan** (`kv_dump` on). It uses 3 pairs
     per relation (capital, ISO 4217) from `test1-population-01`: two
     single-token subjects and one two-token subject. Each target cell
     (pair target Y in Pj) gets these arms:
     - identity;
     - donor X from each of P1, P2 and P3 (`replace`, L16);
     - Y's own state from the other two phrasings;
     - norm-matched donor X from the other two phrasings.

     Each target also declares its admitted baseline hash, and donor runs
     declare their sealed carriers.
   - **The step C equivalence plan:** the 40 step C targets with
     W-subject-L1 and W-subject-L16 for both donors, through the generic
     runner.
2. **Seal:** the generic seal executable on every run and span (BEFORE the
   runner).
3. **Runner:** on both plans.
4. **Admission:** the independent admission module, against plans and seal.
5. **RoPE:** an independent numpy implementation of the container's
   declared rotary encoding.
   - Local layers: θ = 10,000. Global layers: θ = 1,000,000, linear
     factor 8, rotate-half pairing.
   - For every dumped K row at L ≥ 16, the un-rotated target K at its target
     position must equal the un-rotated donor K at its source position.
   - A power control un-rotates at the wrong position and must fail when
     the positions differ.
6. **Step C equivalence:** for each step C target and arm, the generic
   runner's readout hash, candidate logits, injected carriers and K/V hashes
   must equal the recorded step C runner output (c01) bit for bit. The
   natural readouts must match too.
7. **Refusals:**
   - plan mutations (`--check`): wrong source positions, wrong target
     positions, span-length mismatch, span reaching the readout, a
     non-contiguous span, corrupted span IDs, an identity misdeclared, and
     more;
   - run-time: a corrupted declared carrier (provenance) and a wrong
     baseline hash;
   - an existing output.
8. **Admission attacks:** mutated records must be rejected.
9. **Spot check:** the reproduction on named targets.

It writes `OUT_DIR/qualification.json`, with status `qualified` only if
every phase passes.
"""
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import gwtemplate_xpos_admit as admission  # noqa: E402

EXE = ROOT / "target/release/examples/gwtemplate_xpos_runner"
EXE_SRC = ROOT / "crates/larql-demos/examples/inference/gwtemplate_xpos_runner.rs"
SPOT = ROOT / "target/release/examples/gwtemplate_xpos_spotcheck"
SPOT_SRC = ROOT / "crates/larql-demos/examples/inference/gwtemplate_xpos_spotcheck.rs"
SEAL = ROOT / "target/release/examples/gwbreadcrumb2_seal"
BASE = ROOT / "output/gw-template-factorial/baseline-01"
POP = ROOT / "output/gw-template-factorial/test1-population-01.json"
STEPC = ROOT / "output/gw-subject-path-explore/c01"
FROM = 16
LAYERS = 34
PH = ("P1", "P2", "P3")
LOCATED_IN = [5628, 528]
ROPE_RTOL = 1e-4


def sha(p) -> str:
    return "sha256:" + hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(args, **kw):
    return subprocess.run([str(a) for a in args], capture_output=True, text=True, cwd=ROOT, **kw)


def load_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


# ── plans ─────────────────────────────────────────────────────────────────

def template_plan():
    bplan = json.loads((BASE / "plan.json").read_text())
    cells = {c["id"]: c for c in bplan["cells"]}
    brec = {r["id"]: r for r in load_jsonl(BASE / "run/baseline-out.jsonl")}
    pop = json.loads(POP.read_text())
    runs, targets = {}, []

    def add_run(cid):
        c = cells[cid]
        runs.setdefault(cid, {"tokens": c["token_ids"], "text": c["text"]})
        return c

    chosen = {}
    for rel in ("capital", "currency_code_iso4217"):
        pairs = pop["proposed_test1_donors"][rel]["pairs"]
        one = [y for y in sorted(pairs) if pairs[y] and len(cells[f"{rel}|P1|{y}"]["subject"]["tokens"]) == 1][:2]
        two = [y for y in sorted(pairs) if pairs[y] and len(cells[f"{rel}|P1|{y}"]["subject"]["tokens"]) == 2][:1]
        chosen[rel] = [(y, pairs[y][0]) for y in one + two]
        for y, x in chosen[rel]:
            for pj in PH:
                tid = f"{rel}|{pj}|{y}"
                tc = add_run(tid)
                ts = tc["subject"]["tokens"]
                tids = [tc["token_ids"][i] for i in ts]

                def arm(name, did, mode="replace", identity=False):
                    dc = add_run(did)
                    ss = dc["subject"]["tokens"]
                    a = {"name": name, "donor_run": did, "source_span": ss, "target_span": ts,
                         "source_span_ids": [dc["token_ids"][i] for i in ss], "target_span_ids": tids,
                         "from_layer": FROM, "mode": mode}
                    if identity:
                        a["identity"] = True
                    return a

                arms = [arm("identity", tid, identity=True)]
                arms += [arm(f"x-{pi}", f"{rel}|{pi}|{x}") for pi in PH]
                arms += [arm(f"self-{pi}", f"{rel}|{pi}|{y}") for pi in PH if pi != pj]
                arms += [arm(f"nm-{pi}", f"{rel}|{pi}|{x}", mode="norm_match") for pi in PH if pi != pj]
                targets.append({"id": f"q|{tid}", "run": tid, "candidate_set": tc["candidate_set"],
                                "expect_natural_logit_sha256": brec[tid]["readout"]["next_logit_sha256"],
                                "arms": arms})
    plan = {"schema": "larql.gwtemplate.xpos-plan.v1", "container": bplan["container"],
            "tokenizer_sha256": bplan["tokenizer_sha256"], "layers": LAYERS, "top_k": bplan["top_k"],
            "kv_dump": True, "candidate_sets": bplan["candidate_sets"], "runs": runs, "targets": targets}
    return plan, chosen


def stepc_plan():
    cp = json.loads((STEPC / "plan.json").read_text())
    runs, targets = {}, []
    for t in cp["targets"]:
        n = len(t["token_ids"])
        forced = t["token_ids"] + t["path"][:-1]
        tr = f"c|{t['code']}|T"
        runs[tr] = {"tokens": forced}
        span = list(range(4, n - 1))
        arms = []
        for role, suffix in (("B_s", "T"), ("B2_s", "T2")):
            d = t["donors"][role]
            dr = f"c|{t['code']}|{role}"
            runs[dr] = {"tokens": d["token_ids"] + LOCATED_IN}
            for a in (1, 16):
                arms.append({"name": f"W-subject-L{a}-{suffix}", "donor_run": dr, "source_span": span,
                             "target_span": span, "source_span_ids": [runs[dr]["tokens"][i] for i in span],
                             "target_span_ids": [forced[i] for i in span], "from_layer": a, "mode": "replace"})
        targets.append({"id": f"c|{t['code']}", "run": tr, "candidate_set": "stepc", "arms": arms})
    bplan = json.loads((BASE / "plan.json").read_text())
    return {"schema": "larql.gwtemplate.xpos-plan.v1", "container": cp["container"],
            "tokenizer_sha256": bplan["tokenizer_sha256"], "layers": LAYERS, "top_k": 64, "kv_dump": False,
            "candidate_sets": {"stepc": cp["candidate_token_ids"]}, "runs": runs, "targets": targets}


def seal_plan(plans):
    spans = {}
    container = None
    for p in plans:
        container = p["container"]
        for t in p["targets"]:
            for a in t["arms"]:
                for rid, s in ((t["run"], a["target_span"]), (a["donor_run"], a["source_span"])):
                    lo, hi = spans.get(rid, (s[0], s[-1]))
                    spans[rid] = (min(lo, s[0]), max(hi, s[-1]))
    allruns = {}
    for p in plans:
        allruns.update(p["runs"])
    return {"schema": "larql.gwbreadcrumb2.seal-plan.v1", "container": container,
            "runs": [{"id": rid, "tokens": allruns[rid]["tokens"], "positions": [lo, hi]}
                     for rid, (lo, hi) in sorted(spans.items())]}


def declare_from_seal(plan, seal, donor_only=True):
    donors = {a["donor_run"] for t in plan["targets"] for a in t["arms"]}
    for rid, r in plan["runs"].items():
        if rid in seal and (rid in donors or not donor_only):
            r["expected_carrier_sha256"] = dict(seal[rid]["carriers"])


# ── RoPE (independent of the runner) ──────────────────────────────────────

def layer_rope():
    g = json.loads((Path(json.loads((BASE / "plan.json").read_text())["container"]) / "system_graph.json").read_text())
    comp = [c for c in g["components"] if c["id"] == "target"][0]
    out = []
    for a in comp["attention"]:
        pos = a["position"]
        out.append((float(pos["theta"]), float(pos.get("factor", 1.0)) if pos["kind"] == "linear" else 1.0,
                    a["geometry"]["head_dim"], a["geometry"]["num_kv_heads"], pos["kind"]))
    return out


def rotate(x, pos, theta, factor, hd):
    half = hd // 2
    inv = 1.0 / (theta ** (np.arange(0, hd, 2, dtype=np.float64) / hd))
    ang = (pos / factor) * inv
    c, s = np.cos(ang), np.sin(ang)
    a, b = x[..., :half], x[..., half:]
    return np.concatenate([a * c - b * s, b * c + a * s], axis=-1)


def f32hex(h):
    return np.frombuffer(bytes.fromhex(h), dtype="<f4").astype(np.float64)


def rope_check(rows):
    geo = layer_rope()
    res = {"rows": 0, "max_rel_err": 0.0, "fail": [], "power_rows": 0, "power_min_rel_err": None,
           "same_position_rows": 0, "kinds": sorted({g[4] for g in geo})}
    for r in rows:
        for name, a in r["arms"].items():
            for key, d in a.get("kv_dump", {}).items():
                l, sp, tp = map(int, key.split("|"))
                theta, factor, hd, nkv, _ = geo[l]
                tk = f32hex(d["target_k"]).reshape(nkv, hd)
                dk = f32hex(d["donor_k"]).reshape(nkv, hd)
                scale = np.abs(dk).max()
                ut, ud = rotate(tk, -tp, theta, factor, hd), rotate(dk, -sp, theta, factor, hd)
                err = float(np.abs(ut - ud).max() / scale)
                res["rows"] += 1
                res["max_rel_err"] = max(res["max_rel_err"], err)
                if err > ROPE_RTOL:
                    res["fail"].append(f"{r['id']} {name} {key}: rel err {err:.3g}")
                if sp == tp:
                    res["same_position_rows"] += 1
                    if not np.array_equal(tk, dk):
                        res["fail"].append(f"{r['id']} {name} {key}: same position but K differs")
                else:
                    wrong = float(np.abs(rotate(tk, -sp, theta, factor, hd) - ud).max() / scale)
                    res["power_rows"] += 1
                    res["power_min_rel_err"] = wrong if res["power_min_rel_err"] is None else min(res["power_min_rel_err"], wrong)
                    if wrong <= ROPE_RTOL:
                        res["fail"].append(f"{r['id']} {name} {key}: power control did not fail")
    return res


# ── step C equivalence ────────────────────────────────────────────────────

def stepc_equivalence(rows):
    c01 = {r["code"]: r for r in load_jsonl(STEPC / "run/runner-out.jsonl")}
    fails, n = [], 0
    for r in rows:
        code = r["id"].split("|", 1)[1]
        ref = c01[code]
        if r["natural"]["readout"]["next_logit_sha256"] != ref["natural"]["readout"]["next_logit_sha256"]:
            fails.append(f"{code}: natural readout")
        for name, a in r["arms"].items():
            b = ref["arms"][name]
            n += 1
            for k in ("next_logit_sha256", "next", "candidate_logits", "log_normalizer"):
                if a["readout"][k] != b["readout"][k]:
                    fails.append(f"{code} {name}: readout {k}")
            if a["carriers"] != b["carriers"]:
                fails.append(f"{code} {name}: carriers")
            if a["kv_sha256"] != b["kv_sha256"]:
                fails.append(f"{code} {name}: kv")
    return {"arms_compared": n, "failures": fails}


# ── refusals and attacks ──────────────────────────────────────────────────

def plan_mutations(plan):
    t0 = plan["targets"][0]
    xi = next(i for i, a in enumerate(t0["arms"]) if a["name"] == "x-P1")

    def m(f):
        p = copy.deepcopy(plan)
        f(p, p["targets"][0], p["targets"][0]["arms"][xi])
        return p

    n = len(plan["runs"][t0["run"]]["tokens"])
    return {
        "wrong source position": (m(lambda p, t, a: a.update(source_span=[s + 1 for s in a["source_span"]])),
                                  "source span token ids differ"),
        "wrong target position": (m(lambda p, t, a: a.update(target_span=[s - 1 for s in a["target_span"]])),
                                  "target span token ids differ"),
        "span length mismatch": (m(lambda p, t, a: a.update(source_span=a["source_span"] + [a["source_span"][-1] + 1])),
                                 "differ in length"),
        "span reaches readout": (m(lambda p, t, a: a.update(target_span=[n - 1], source_span=[a["source_span"][0]])),
                                 "reaches the readout"),
        "non-contiguous span": (m(lambda p, t, a: a.update(target_span=[a["target_span"][0], a["target_span"][0] + 2],
                                                          source_span=[a["source_span"][0], a["source_span"][0] + 1])),
                                "not contiguous"),
        "source span outside donor": (m(lambda p, t, a: a.update(source_span=[999])), "outside the donor run"),
        "corrupted span ids": (m(lambda p, t, a: a.update(source_span_ids=[1] * len(a["source_span_ids"]))),
                               "source span token ids differ"),
        "identity not self": (m(lambda p, t, a: a.update(identity=True)), "identity must be declared"),
        "self not declared identity": (m(lambda p, t, a: t["arms"][0].pop("identity")), "identity must be declared"),
        "unknown donor run": (m(lambda p, t, a: a.update(donor_run="nope")), "unknown donor run"),
        "from layer out of range": (m(lambda p, t, a: a.update(from_layer=LAYERS)), "from_layer"),
        "bad mode": (m(lambda p, t, a: a.update(mode="add")), "mode must be"),
        "declared carrier out of range": (m(lambda p, t, a: p["runs"][a["donor_run"]].update(
            expected_carrier_sha256={"0|999": "0" * 64})), "out of range"),
        "malformed declared carrier": (m(lambda p, t, a: p["runs"][a["donor_run"]].update(
            expected_carrier_sha256={"0|4": "sha256:00"})), "malformed"),
        "text does not tokenise": (m(lambda p, t, a: p["runs"][t["run"]].update(text="something else")),
                                   "re-tokenised"),
        "tokenizer hash": (m(lambda p, t, a: p.update(tokenizer_sha256="sha256:00")), "tokenizer hash"),
        "schema": (m(lambda p, t, a: p.update(schema="x")), "plan schema"),
        "arm repeated": (m(lambda p, t, a: t["arms"].append(copy.deepcopy(a))), "arm repeated"),
    }


def record_attacks(rows):
    t = next(r for r in rows if r["id"].startswith("q|"))
    out = {}

    def m(name, f):
        rr = copy.deepcopy(rows)
        tt = next(r for r in rr if r["id"] == t["id"])
        f(tt)
        out[name] = rr

    k0 = next(iter(t["arms"]["x-P1"]["carriers"]))
    m("injected carrier altered", lambda r: r["arms"]["x-P1"]["carriers"].update({k0: "sha256:" + "0" * 64}))
    kk = next(iter(r for r in t["arms"]["x-P1"]["kv_sha256"]))
    m("natural K/V below start altered", lambda r: r["arms"]["x-P1"]["kv_sha256"].update({kk: "sha256:" + "1" * 64}))
    m("donor K/V altered", lambda r: r["arms"]["x-P1"]["donor_kv_sha256"].update(
        {next(iter(r["arms"]["x-P1"]["donor_kv_sha256"])): "sha256:" + "2" * 64}))
    m("natural step altered", lambda r: r["natural"]["step_logit_sha256"].__setitem__(0, "sha256:" + "3" * 64))
    m("arm dropped", lambda r: r["arms"].pop("nm-P2", None) or r["arms"].pop("nm-P1"))
    m("readout greedy swapped", lambda r: r["arms"]["x-P1"]["readout"].update(next=-1))
    m("replacement count", lambda r: r["arms"]["x-P1"].update(replacements=1))
    m("identity step altered", lambda r: r["arms"]["identity"]["step_logit_sha256"].__setitem__(-1, "sha256:" + "4" * 64))
    nmk = next(a for a in t["arms"] if a.startswith("nm-"))
    m("norm not matched", lambda r: next(iter(r["arms"][nmk]["norms"].values())).update(injected=1e9))
    m("records reordered", lambda r: None)
    out["records reordered"] = list(reversed(out["records reordered"]))
    return out


# ── main ──────────────────────────────────────────────────────────────────

def main(argv):
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    out = Path(argv[0])
    out.mkdir(parents=False)
    rec = {"schema": "larql.gwtemplate.xpos-qualification.v1",
           "executable_sha256": sha(EXE), "executable_source_sha256": sha(EXE_SRC),
           "spotcheck_sha256": sha(SPOT), "spotcheck_source_sha256": sha(SPOT_SRC),
           "seal_executable_sha256": sha(SEAL), "harness_sha256": sha(__file__),
           "admission_sha256": sha(ROOT / "scripts/gwtemplate_xpos_admit.py"),
           "inputs": {"baseline_plan": sha(BASE / "plan.json"), "baseline_out": sha(BASE / "run/baseline-out.jsonl"),
                      "population": sha(POP), "stepc_plan": sha(STEPC / "plan.json"),
                      "stepc_out": sha(STEPC / "run/runner-out.jsonl")},
           "phases": {}}
    ph = rec["phases"]
    tplan, chosen = template_plan()
    cplan = stepc_plan()
    rec["template_pairs"] = chosen

    # Seal first, before any runner execution.
    sp = seal_plan([tplan, cplan])
    (out / "seal-plan.json").write_text(json.dumps(sp, indent=1) + "\n")
    s = run([SEAL, out / "seal-plan.json", out / "seal.jsonl"])
    if s.returncode != 0:
        raise SystemExit(f"seal failed: {s.stderr[-2000:]}")
    seal = admission.seal_index(load_jsonl(out / "seal.jsonl"))
    ph["seal"] = {"runs": len(seal), "output_sha256": sha(out / "seal.jsonl")}
    declare_from_seal(tplan, seal)
    for name, p in (("template", tplan), ("stepc", cplan)):
        (out / f"plan-{name}.json").write_text(json.dumps(p, indent=1) + "\n")

    for name in ("template", "stepc"):
        c = run([EXE, "--check", out / f"plan-{name}.json"])
        ph[f"check {name}"] = {"ok": c.returncode == 0, "stdout": c.stdout.strip(), "stderr": c.stderr[-500:]}
        r = run([EXE, out / f"plan-{name}.json", out / f"out-{name}.jsonl"])
        ph[f"run {name}"] = {"ok": r.returncode == 0, "stderr_tail": r.stderr[-1000:]}
        if r.returncode != 0:
            (out / "qualification.json").write_text(json.dumps(rec, indent=1) + "\n")
            raise SystemExit(f"runner failed on {name}: {r.stderr[-2000:]}")

    rows_t = load_jsonl(out / "out-template.jsonl")
    rows_c = load_jsonl(out / "out-stepc.jsonl")
    ph["admission"] = {"template": admission.admit(tplan, seal, rows_t),
                       "stepc": admission.admit(cplan, seal, rows_c)}
    ph["admission"]["ok"] = not ph["admission"]["template"] and not ph["admission"]["stepc"]

    rc = rope_check(rows_t)
    rc["ok"] = not rc["fail"] and rc["rows"] > 0 and rc["power_rows"] > 0
    ph["rope"] = rc

    eq = stepc_equivalence(rows_c)
    eq["ok"] = not eq["failures"] and eq["arms_compared"] == 160
    ph["stepc_equivalence"] = eq

    # Descriptive: K equal to donor where positions differ (expected false).
    kcross = [v for r in rows_t for a in r["arms"].values() if a["mode"] == "replace"
              for k, v in a["k_equal_donor"].items()
              if a["source_span"][0] != a["target_span"][0]]
    ph["cross_position_k_rows"] = {"rows": len(kcross), "equal_to_donor": sum(kcross)}

    refusals = {}
    for name, (p, want) in plan_mutations(tplan).items():
        f = out / "refusal-plan.json"
        f.write_text(json.dumps(p))
        c = run([EXE, "--check", f])
        refusals[name] = {"refused": c.returncode != 0 and want in c.stderr, "stderr": c.stderr.strip()[-200:]}
        f.unlink()
    # Run-time refusals on a one-target plan.
    small = copy.deepcopy(tplan)
    small["targets"] = small["targets"][:1]
    small["targets"][0]["arms"] = [a for a in small["targets"][0]["arms"] if a["name"] == "x-P1"]
    small["kv_dump"] = False
    drun = small["targets"][0]["arms"][0]["donor_run"]
    k = next(iter(small["runs"][drun]["expected_carrier_sha256"]))
    bad = copy.deepcopy(small)
    bad["runs"][drun]["expected_carrier_sha256"][k] = "0" * 64
    wrongbase = copy.deepcopy(small)
    wrongbase["targets"][0]["expect_natural_logit_sha256"] = "sha256:" + "0" * 64
    for name, p, want in (("corrupted provenance (declared carrier)", bad, "differs from its declared hash"),
                          ("wrong baseline hash", wrongbase, "differs from the declared baseline hash")):
        f = out / "refusal-plan.json"
        f.write_text(json.dumps(p))
        r = run([EXE, f, out / "refusal-out.jsonl"])
        refusals[name] = {"refused": r.returncode != 0 and want in r.stderr, "stderr": r.stderr.strip()[-200:],
                          "no_output": not (out / "refusal-out.jsonl").exists()}
        refusals[name]["refused"] &= refusals[name]["no_output"]
        f.unlink()
        for leftover in out.glob("refusal-out.jsonl*"):
            leftover.unlink()
    r = run([EXE, out / "plan-template.json", out / "out-template.jsonl"])
    refusals["existing output"] = {"refused": r.returncode != 0 and "never overwrites" in r.stderr}
    ph["refusals"] = {"cases": refusals, "ok": all(v["refused"] for v in refusals.values()),
                      "n": len(refusals), "refused": sum(v["refused"] for v in refusals.values())}

    attacks = {}
    for name, rr in record_attacks(rows_t).items():
        attacks[name] = bool(admission.admit(tplan, seal, rr))
    ph["admission_attacks"] = {"cases": attacks, "ok": all(attacks.values()), "n": len(attacks)}

    ids = [rows_t[0]["id"], rows_t[4]["id"], rows_t[-1]["id"], rows_c[0]["id"]]
    spot = []
    for name, plan_name, out_name, tids in (("template", "plan-template.json", "out-template.jsonl", ids[:3]),
                                            ("stepc", "plan-stepc.json", "out-stepc.jsonl", ids[3:])):
        r = run([SPOT, out / plan_name, out / out_name, out / f"spotcheck-{name}.json", *tids])
        rep = json.loads((out / f"spotcheck-{name}.json").read_text()) if r.returncode == 0 else {"error": r.stderr[-500:]}
        spot.append({"plan": name, "targets": tids, "result": rep.get("result"),
                     "differences": rep.get("total_differences")})
    ph["spotcheck"] = {"runs": spot, "ok": all(s["result"] == "REPRODUCED" for s in spot)}

    ok = all(v.get("ok", True) for k, v in ph.items() if isinstance(v, dict))
    rec["status"] = "qualified" if ok else "NOT qualified"
    rec["failed_phases"] = [k for k, v in ph.items() if isinstance(v, dict) and not v.get("ok", True)]
    rec["outputs"] = {p.name: sha(p) for p in sorted(out.glob("*.json*")) if p.name != "qualification.json"}
    (out / "qualification.json").write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"status": rec["status"], "failed": rec["failed_phases"],
                      "rope": {k: rc[k] for k in ("rows", "max_rel_err", "power_rows", "power_min_rel_err",
                                                  "same_position_rows")},
                      "stepc": eq["arms_compared"], "refusals": f"{ph['refusals']['refused']}/{ph['refusals']['n']}",
                      "attacks": ph["admission_attacks"]["cases"], "spot": spot}, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
