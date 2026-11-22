"""GW-ELEMENT-1 confirmation: element-domain plan builder (model-free).

Builds, from the registered natural baseline (`output/gw-element-1/baseline-01`,
plan and records hash-checked against its binding):
- **runs**: `rel|ph|Z` token runs. The window is the tokens after the subject up
  to (not including) the readout `is`. The decoded window must equal the frozen
  slot strings, and the step hashes are the baseline's prompt steps;
- the **seal plan** (`gwbreadcrumb2_seal`, unchanged) and the **capture plan**
  (`gwtemplate_test3_capture`, unchanged);
- **band plans** for the element runner.

Sealing: any run of a confirmation-split element refuses unless the caller
passes `release=True`. Only the confirmation execution does that, after the
owner's GO. Qualification never does.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gwelem_design as D  # noqa: E402

BASE = D.BASELINE
LAYERS = 34
HIDDEN = 2560
SEAL_SCHEMA = "larql.gwbreadcrumb2.seal-plan.v1"
CAPTURE_SCHEMA = "larql.gwtemplate.test3-capture-plan.v1"
BAND_SCHEMA = "larql.gwcompress.band-plan.v1"
CANDIDATE_SET = "element"


class SealedError(Exception):
    pass


def _load_jsonl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def baseline() -> tuple[dict, dict]:
    """(plan, records by id) from baseline-01, verified against its binding."""
    b = json.loads((BASE / "binding.json").read_text())
    plan_p, out_p = BASE / "plan.json", BASE / "out.jsonl"
    man = json.loads((BASE / "out.jsonl.manifest.json").read_text())
    if D.sha(plan_p) != b["files"]["plan"]["sha256"] or D.sha(plan_p) != man["plan_sha256"]:
        raise SystemExit("FAIL CLOSED: baseline-01 plan differs from its binding")
    if D.sha(out_p) != man["output_sha256"] or man["status"] != "complete":
        raise SystemExit("FAIL CLOSED: baseline-01 records differ from their manifest")
    integ = json.loads((BASE / "integrity.json").read_text())
    if integ.get("status") != "admitted":
        raise SystemExit("FAIL CLOSED: baseline-01 is not admitted")
    return json.loads(plan_p.read_text()), {r["id"]: r for r in _load_jsonl(out_p)}


def runs(zs, decode, release: bool = False, plan_recs=None) -> dict:
    """Band/capture-format runs for every relation and phrasing of each element in zs."""
    zs = [int(z) for z in zs]  # normalise: a string key must not slip past the seal (reviewer finding I1)
    sealed = [z for z in zs if z in D.CONFIRMATION_SPLIT]
    if sealed and not release:
        raise SealedError(f"sealed confirmation elements requested without release: {sealed}")
    plan, recs = plan_recs or baseline()
    cells = {c["id"]: c for c in plan["cells"]}
    out = {}
    for z in sorted(zs):
        for rel in D.RELATIONS:
            for ph in D.PHRASINGS:
                rid = D.run_id(rel, ph, z)
                c = cells[rid]
                toks = c["token_ids"]
                lo, hi = c["subject"]["tokens"][-1] + 1, c["readout"] - 1
                win = list(range(lo, hi + 1))
                if c["readout"] != len(toks) - 1 or len(win) != D.WINDOW_LEN:
                    raise ValueError(f"{rid}: window geometry")
                got = [decode([toks[p]]) for p in win]
                if got != D.DECODED_WINDOWS[(rel, ph)]:
                    raise ValueError(f"{rid}: decoded window {got}")
                r = recs[rid]
                if r["prompt_tokens"] != toks:
                    raise ValueError(f"{rid}: baseline record tokens differ from the plan")
                out[rid] = {"country": D.key(z), "tokens": toks, "window": win, "window_ids": [toks[p] for p in win],
                            "expected_step_logit_sha256": r["step_logit_sha256"][:len(toks)],
                            "baseline_next_logit_sha256": r["readout"]["next_logit_sha256"]}
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
        if list(range(lo, hi + 1)) != r["window"] or s["step_logit_sha256"] != r["expected_step_logit_sha256"]:
            raise ValueError(f"{rid}: seal differs from the baseline run")
        out[rid] = {"country": r["country"], "tokens": r["tokens"], "window": r["window"],
                    "window_ids": r["window_ids"], "expected_carrier_sha256": s["carriers"],
                    "expected_kv_sha256": s["kv_sha256"], "expected_step_logit_sha256": s["step_logit_sha256"]}
    return {"schema": CAPTURE_SCHEMA, "container": meta["container"], "tokenizer_sha256": meta["tokenizer_sha256"],
            "layers": LAYERS, "holdout": sorted(refusal), "runs": out, "sources": sources}


def arm(name: str, run: dict, vectors_by_slot: dict, identity: bool = False) -> dict:
    """A single-carrier L4 arm. name = KIND-SPAN-L4; vectors_by_slot: slot -> vector id."""
    span_name = name.split("-")[1]
    slots = D.SPANS[span_name]
    if sorted(vectors_by_slot) != list(slots):
        raise ValueError(f"{name}: vectors for slots {sorted(vectors_by_slot)} != {slots}")
    span = [run["window"][j] for j in slots]
    a = {"name": name, "from_layer": D.LAYER + 1, "to_layer": D.LAYER, "target_span": span,
         "vectors": {f"{D.LAYER}|{run['window'][j]}": vectors_by_slot[j] for j in slots}}
    if identity:
        a["identity"] = True
    return a


def band_plan(meta: dict, rs: dict, targets: list[dict], store_file: Path, store_sha: str, index: dict,
              candidates: list[int], refusal: list[str], hidden: int = HIDDEN) -> dict:
    used = {t["run"] for t in targets}
    return {"schema": BAND_SCHEMA, "container": meta["container"], "tokenizer_sha256": meta["tokenizer_sha256"],
            "layers": LAYERS, "top_k": 64, "kv_dump": False, "propagation_check": True,
            "holdout": sorted(refusal), "candidate_sets": {CANDIDATE_SET: candidates},
            "vector_store": {"file": str(store_file), "sha256": store_sha, "hidden": hidden,
                             "index": {v: {k: e[k] for k in ("offset", "sha256", "recipe_sha256")} for v, e in index.items()}},
            "runs": {rid: {k: rs[rid][k] for k in ("country", "tokens", "window", "window_ids",
                                                   "expected_step_logit_sha256")} for rid in sorted(used)},
            "targets": targets}


def meta() -> dict:
    plan, _ = baseline()
    return {"container": plan["container"], "tokenizer_sha256": plan["tokenizer_sha256"],
            "candidates": plan["candidate_sets"][CANDIDATE_SET]}
