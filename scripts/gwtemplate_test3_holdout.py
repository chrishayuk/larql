"""GW-TEMPLATE-FACTORIAL Test 3 Stage 4 (holdout, capital→currency): model-free core.

Protocol: `docs/gw-template-factorial-test3-holdout-protocol.md`. Every function
here is generic over the country set it is given. Qualification
(`gwtemplate_test3_holdout_qualify.py`) applies each one to the development
population and requires exact reproduction of the existing Test 2 and
Stage 1b/Stage 2 artifacts. The holdout itself is read only by the bound
`prepare` step, after owner approval.

Functions:
- **eligibility:** the d3 §2 rule (joint capital × ISO 4217, minus subject
  echo), plus the window-geometry and answer-readability requirements that
  the Test 2 builder and Stage 2 analysis already enforce.
- **donors:** the Test 2 donor rule (same subject token count, four
  pairwise-disjoint first-token sets in P2 and P3), applied within a pool
  with a given salt.
- **runs:** tokens and the right-aligned window, by Test 2's `windows()`.
- **seal_plan / capture_plan:** the BREADCRUMB-2 seal plan and the Stage 1b
  capture plan for those runs.
- **derived:** the C1/C2/S/C0/ID arms built by the qualified v1 builder over a
  restricted capture view.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import gwtemplate_test2_feasibility as feas  # noqa: E402
import gwtemplate_test3_candidates as v1  # noqa: E402

GF = ROOT / "output/gw-template-factorial"
POP = GF / "test1-population-01.json"
BASE = GF / "baseline-01"
REL, RP = "capital", "currency_code_iso4217"
PHRASINGS = ("P2", "P3")
HOLDOUT_DONOR_SALT = "gw-template-factorial/test3-holdout-donors/v1|"
N_DONORS = 2
SEAL_SCHEMA = "larql.gwbreadcrumb2.seal-plan.v1"
CAPTURE_SCHEMA = "larql.gwtemplate.test3-capture-plan.v1"
DERIVED_SCHEMA = "larql.gwtemplate.test3-derived-plan.v1"
TOP_K = 64
UNION = "capital+currency_code_iso4217"
CAPITAL_ARMS = ("ID", "S", "C0-D1", "C0-D2", "C1", "C2")  # C0-Dk present only when donor k exists


def canonical_window_ids(cells: dict, countries: list[str]) -> dict:
    """The window token ids per (phrasing, relation) of a reference population (Test 2 requires one value)."""
    out = {}
    for p in PHRASINGS:
        for r in feas.RELS:
            ids = {tuple(feas.windows(cells, c, p)[r]["ids"][i] for i in feas.windows(cells, c, p)[r]["window"])
                   for c in countries}
            if len(ids) != 1:
                raise ValueError(f"reference window ids vary in {p} {r}")
            out[(p, r)] = next(iter(ids))
    return out


def eligibility(pop: dict, cells: dict, countries: list[str], canon: dict, cands: set[int]) -> tuple[list[str], dict]:
    joint = set(pop["populations"]["joint_countries"])
    echo = set(pop["subject_echo"]["capital"]) | set(pop["subject_echo"]["currency_code_iso4217"])
    ok, excluded = [], {}
    for c in sorted(countries):
        why = []
        if c not in joint:
            why.append("not in the joint capital x ISO 4217 population")
        if c in echo:
            why.append("subject echo")
        if any(f"{r}|{p}|{c}" not in cells for r in feas.RELS for p in PHRASINGS):
            why.append("missing baseline cell")
        if not why:
            for p in PHRASINGS:
                w = feas.windows(cells, c, p)
                for r in feas.RELS:
                    win = w[r]["window"]
                    if len(win) != feas.EXPECT_WINDOW[p] or win[-1] != w[r]["readout"] - 1 or not set(w[r]["word"]) <= set(win):
                        why.append(f"{p} {r}: window geometry")
                    elif tuple(w[r]["ids"][i] for i in win) != canon[(p, r)]:
                        why.append(f"{p} {r}: window token ids")
                acc = {r: cells[f"{r}|{p}|{c}"]["accepted_first_tokens"] for r in feas.RELS}
                if any(len(a) != 1 for a in acc.values()):
                    why.append(f"{p}: accepted first-token set is not a single token")
                elif set(acc[REL]) & set(acc[RP]):
                    why.append(f"{p}: capital and currency share a first token")
                elif any(a[0] not in cands for a in acc.values()):
                    why.append(f"{p}: answer token outside the candidate set")
        if why:
            excluded[c] = why
        else:
            ok.append(c)
    return ok, excluded


def donors(cells: dict, pool: list[str], salt: str, n: int = N_DONORS) -> dict:
    """The Test 2 donor rule within `pool` (feasibility.py §3). Returns {y: {"donors": [...<=n], "eligible_donors": k}}."""
    def acc(r, p, c):
        return set(cells[f"{r}|{p}|{c}"]["accepted_first_tokens"])

    def slen(c):
        return len(cells[f"capital|P2|{c}"]["subject"]["tokens"])

    out = {}
    for y in sorted(pool):
        ok = []
        for x in sorted(pool):
            if x == y or slen(x) != slen(y):
                continue
            if all(not (s[i] & s[j]) for p in PHRASINGS
                   for s in [[acc(r, p, z) for z in (y, x) for r in feas.RELS]] for i in range(4) for j in range(i + 1, 4)):
                ok.append(x)
        ok.sort(key=lambda x: hashlib.sha256((salt + y + "|" + x).encode()).hexdigest())
        out[y] = {"donors": ok[:n], "eligible_donors": len(ok)}
    return out


def runs(cells: dict, countries: list[str]) -> dict:
    out = {}
    for c in sorted(countries):
        for r in feas.RELS:
            for p in PHRASINGS:
                rid = f"{r}|{p}|{c}"
                win = feas.windows(cells, c, p)[r]["window"]
                toks = cells[rid]["token_ids"]
                out[rid] = {"country": c, "tokens": toks, "window": win, "window_ids": [toks[i] for i in win]}
    return out


def seal_plan(container: str, rs: dict) -> dict:
    return {"schema": SEAL_SCHEMA, "container": container,
            "runs": [{"id": rid, "tokens": r["tokens"], "positions": [r["window"][0], r["window"][-1]]}
                     for rid, r in sorted(rs.items())]}


def capture_plan(meta: dict, rs: dict, seal_rows: list[dict], refusal: list[str], sources: dict) -> dict:
    seal = {r["id"]: r for r in seal_rows}
    if set(seal) != set(rs):
        raise ValueError("seal and runs differ")
    out = {}
    for rid in sorted(rs):
        s, r = seal[rid], rs[rid]
        lo, hi = s["positions"]
        if list(range(lo, hi + 1)) != r["window"]:
            raise ValueError(f"{rid}: seal window differs")
        out[rid] = {"country": r["country"], "tokens": r["tokens"], "window": r["window"], "window_ids": r["window_ids"],
                    "expected_carrier_sha256": s["carriers"], "expected_kv_sha256": s["kv_sha256"],
                    "expected_step_logit_sha256": s["step_logit_sha256"]}
    return {"schema": CAPTURE_SCHEMA, "container": meta["container"], "tokenizer_sha256": meta["tokenizer_sha256"],
            "layers": meta["layers"], "holdout": sorted(refusal), "runs": out, "sources": sources}


def capture_view(cap_dirs: list[Path], allowed: set[str], blocked: set[str]) -> "v1.Capture":
    """A v1 Capture over several Stage-1b-format capture dirs, loading only `allowed` countries.

    `blocked` becomes the Capture's refusal set, so reading those countries raises.
    """
    w = {}
    for d in cap_dirs:
        man = json.loads((d / "manifest.json").read_text())
        for rid, e in man["runs"].items():
            rel, ph, c = rid.split("|")
            if c not in allowed or ph not in PHRASINGS:
                continue
            a = np.fromfile(d / e["file"], dtype="<f4").reshape(e["layers"], len(e["positions"]), e["hidden"])
            w[(rel, ph, c)] = a[:, :len(e["window"]), :]
    return v1.Capture(w, set(blocked))


def derived(out: Path, cap: "v1.Capture", fit: list[str], targets: list[str], donor_map: dict, run_meta: dict,
            meta: dict, refusal: list[str], cand_sets: dict, id_prefix: str = "s4") -> dict:
    """Builds the four shards (capital P2/P3 targets with all arms; currency P2/P3 reference cells with ID only).

    run_meta: {rid: {country, tokens, window, window_ids, expected_step_logit_sha256}} for every target run.
    """
    out.mkdir(parents=False)
    b = v1.Builder(cap, fit)
    layers = meta["layers"]
    hidden = next(iter(cap._w.values())).shape[2]
    cset = next(iter(cand_sets))
    shards = []
    for rel in (REL, RP):
        for ph in PHRASINGS:
            st = v1.Store(hidden)
            tl = []
            rids = [f"{rel}|{ph}|{y}" for y in sorted(targets)]
            for rid in rids:
                y = rid.split("|")[2]
                win = run_meta[rid]["window"]

                def add(v, rec):
                    return {f"{l}|{win[j]}": st.add(v[l, j], rec, l, j, win[j], cap, source_slot=j)
                            for l in range(layers) for j in range(len(win))}

                v, rec = b.reference("ID", y, rel, ph)
                arms = [{"name": "ID-window-L1", "from_layer": 1, "target_span": win, "identity": True, "vectors": add(v, rec)}]
                if rel == REL:
                    v, rec = b.reference("S", y, rel, ph)
                    arms.append({"name": "S-window-L1", "from_layer": 1, "target_span": win, "vectors": add(v, rec)})
                    for k, x in enumerate(donor_map[y]["donors"], 1):
                        v, rec = b.reference("C0", y, rel, ph, x)
                        arms.append({"name": f"C0-D{k}-L1", "from_layer": 1, "target_span": win, "donor": x,
                                     "vectors": add(v, rec)})
                    for kind in ("C1", "C2"):
                        v, rec = b.candidate(kind, y, rel, ph)
                        arms.append({"name": f"{kind}-window-L1", "from_layer": 1, "target_span": win, "vectors": add(v, rec)})
                tl.append({"id": f"{id_prefix}|{rid}", "run": rid, "candidate_set": cset, "relation": rel, "phrasing": ph,
                           "country": y, "population": "holdout", "role": "target" if rel == REL else "reference",
                           "donors": donor_map[y]["donors"], "arms": arms})
            name = f"{'capital' if rel == REL else 'currency'}-{ph}"
            store_file = out / f"store-{name}.f32"
            store_sha = st.write(store_file)
            (out / f"index-{name}.json").write_text(json.dumps(st.index, sort_keys=True) + "\n")
            plan = {"schema": DERIVED_SCHEMA, "container": meta["container"], "tokenizer_sha256": meta["tokenizer_sha256"],
                    "layers": layers, "top_k": TOP_K, "kv_dump": False, "holdout": sorted(refusal), "candidate_sets": cand_sets,
                    "vector_store": {"file": str(store_file), "sha256": store_sha, "hidden": hidden,
                                     "index": {k: {kk: e[kk] for kk in ("offset", "sha256", "recipe_sha256")} for k, e in st.index.items()}},
                    "runs": {rid: {k: run_meta[rid][k] for k in ("country", "tokens", "window", "window_ids", "expected_step_logit_sha256")}
                             for rid in rids},
                    "targets": tl}
            (out / f"plan-{name}.json").write_text(json.dumps(plan) + "\n")
            shards.append({"name": name, "relation": rel, "phrasing": ph, "plan": f"plan-{name}.json",
                           "index": f"index-{name}.json", "store": store_file.name, "store_sha256": store_sha,
                           "vectors": len(st.index), "targets": len(tl), "arms": sum(len(t["arms"]) for t in tl),
                           "output": f"out-{name}.jsonl"})
    return {"shards": shards, "hidden": hidden, "layers": layers}


def xpos_plan(meta: dict, cells: dict, targets: list[str], donor_map: dict, expect_natural: dict, cand_sets: dict) -> dict:
    """The Test 2 runner plan for the C0/S replication check: per capital target cell, Test 2's
    identity arm plus S-window-L1 and Dk-window-L1, in Test 2's arm format (`gwtemplate_test2_plan.cell_arms`)."""
    import gwtemplate_test2_plan as t2p
    rs, tl = {}, []
    for y in sorted(targets):
        for p in PHRASINGS:
            tid = f"{REL}|{p}|{y}"
            full = t2p.cell_arms(cells, rs, y, donor_map[y]["donors"], REL, p)
            keep = ["identity", "S-window-L1"] + [f"D{k}-window-L1" for k in range(1, len(donor_map[y]["donors"]) + 1)]
            arms = [a for a in full if a["name"] in keep]
            tl.append({"id": f"s4x|{tid}", "run": tid, "candidate_set": next(iter(cand_sets)), "relation": REL,
                       "other_relation": RP, "phrasing": p, "country": y, "donors": donor_map[y]["donors"],
                       "expect_natural_logit_sha256": expect_natural[tid], "arms": arms})
    used = {t["run"] for t in tl} | {a["donor_run"] for t in tl for a in t["arms"]}
    return {"schema": "larql.gwtemplate.xpos-plan.v1", "container": meta["container"], "tokenizer_sha256": meta["tokenizer_sha256"],
            "layers": meta["layers"], "top_k": meta["top_k"], "kv_dump": False, "candidate_sets": cand_sets,
            "runs": {k: v for k, v in rs.items() if k in used}, "targets": tl}
