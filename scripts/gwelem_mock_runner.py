#!/usr/bin/env python3
"""GW-ELEMENT-1 synthetic integration qualification: a MOCK band runner (no model is loaded).

It stands in for `gwelem_band_runner` execution only in `gwelem_confirm_dryrun.py`, so that the
orchestrator's composition can be tested end to end (accounting, fail-closed paths, release consumption,
post-run admission and the single analysis) without running any constructed state through the model.

It writes records in the band runner's exact schema. Every value is derived, not computed:
- carrier hashes come from the plan's vector index and the capture manifests;
- natural readouts are the registered baseline readouts;
- every arm's readout is a copy of the natural readout of its run. **No switching effect is computed
  or implied.**

The release handling mirrors the real runner: the declared file and hash, the runs digest, refusal if
the release has already been used, and an exclusive `.used` marker.

    gwelem_mock_runner.py --caps D1,D2,.. [--crash-after N] [--corrupt-identity] PLAN OUT
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gwelem_admit as A  # noqa: E402
import gwelem_plan as P  # noqa: E402

RECORD_SCHEMA = "larql.gwcompress.band-record.v1"


def sha(p):
    return "sha256:" + hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(argv):
    caps, crash, corrupt = None, None, False
    while argv[0].startswith("--"):
        f = argv.pop(0)
        if f == "--caps":
            caps = A.Caps([Path(x) for x in argv.pop(0).split(",")])
        elif f == "--crash-after":
            crash = int(argv.pop(0))
        elif f == "--corrupt-identity":
            corrupt = True
    plan_p, out = Path(argv[0]), Path(argv[1])
    plan = json.loads(plan_p.read_text())
    rel = plan.get("sealed_release")
    if rel:
        f = rel["authorisation_file"]
        a = json.loads(Path(f).read_text())
        if sha(f) != rel["authorisation_sha256"] or a["runs_digest"] != A.runs_digest(plan["runs"]):
            print("REFUSED: release", file=sys.stderr)
            return 1
        try:
            with open(f + ".used", "x") as m:
                m.write(sha(plan_p))
        except FileExistsError:
            print("REFUSED: authorisation already used", file=sys.stderr)
            return 1
    recs = P.baseline()[1]
    layers = plan["layers"]
    idx = plan["vector_store"]["index"]
    partial = Path(str(out) + ".partial")
    n = 0
    with open(partial, "x") as fh:
        for t in plan["targets"]:
            if crash is not None and n >= crash:
                print("mock crash", file=sys.stderr)
                return 3
            run = plan["runs"][t["run"]]
            cap = caps.entries[t["run"]]["vector_sha256"]
            nat_sc = {f"{l}|{p}": cap[f"{l}|{p}"] for l in range(layers) for p in run["window"]}
            ro = recs[t["run"]]["readout"]
            arms = {}
            for a in t["arms"]:
                lo, hi = a["from_layer"] - 1, a["to_layer"]
                sc = dict(nat_sc)
                vec = {k: idx[v]["sha256"] for k, v in a["vectors"].items()}
                for k, v in vec.items():
                    sc[k] = v
                identity = bool(a.get("identity")) and not corrupt
                arms[a["name"]] = {"from_layer": a["from_layer"], "to_layer": hi, "band": [lo, hi],
                                   "target_span": a["target_span"], "replacements": len(vec), "firings": len(vec),
                                   "span_carrier_sha256": sc, "above_band_changed": [],
                                   "propagation_fixpoint": {"bit_exact": True, "replayed_addresses": (layers - hi - 1) * len(a["target_span"])},
                                   "vectors": vec, "recipes": {idx[v]["recipe_sha256"]: 1 for v in a["vectors"].values()},
                                   "identity_bit_exact": identity, "step_logit_sha256": [], "readout": ro}
            fh.write(json.dumps({"schema": RECORD_SCHEMA, "id": t["id"], "run": t["run"],
                                 "natural": {"readout": ro, "replay_bit_exact": True, "span_carrier_sha256": nat_sc},
                                 "arms": arms}) + "\n")
            n += 1
    partial.rename(out)
    man = {"schema": "larql.gwelem.band-manifest.v1", "status": "complete", "executable_sha256": sha(__file__),
           "plan_sha256": sha(plan_p), "output_sha256": sha(out), "vector_store_sha256": plan["vector_store"]["sha256"],
           "targets": len(plan["targets"]), "propagation_check": True,
           "sealed_release": ({"population": rel["population"], "authorisation_sha256": rel["authorisation_sha256"],
                               "runs_digest": A.runs_digest(plan["runs"])} if rel else None), "mock": True}
    Path(str(out) + ".manifest.json").write_text(json.dumps(man, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
