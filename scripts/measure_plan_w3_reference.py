#!/usr/bin/env python3
"""MEASURE-PLAN-1 W3: an independent numpy computation of the procedure's
per-position metrics, over the committed teacher-forced logits.

Shares no code with `represent/measure/plan/metrics.rs`. It follows the same
stated conventions, which are part of the metric's definition:
- full vocabulary, nats, float64 log-softmax;
- ties resolve to the lower token id, so argmax is the first maximum;
- top-5 by a stable descending sort;
- delta NLL = NLL(candidate) - NLL(reference) of the next token.

    python3 scripts/measure_plan_w3_reference.py          # write expected.json
    python3 scripts/measure_plan_w3_reference.py --check  # recompute and compare
"""

import json
import sys
from pathlib import Path

import numpy as np

DIR = Path(__file__).resolve().parents[1] / (
    "crates/larql-vindex/src/format/vindex3/represent/fixtures/measure-plan-w3"
)
TOP_K = 5


def log_softmax(row):
    row = row.astype(np.float64)
    m = row.max()
    return row - (m + np.log(np.exp(row - m).sum()))


def position(ref_row, cand_row, next_token):
    lp, lq = log_softmax(ref_row), log_softmax(cand_row)
    p = np.exp(lp)
    ref_top = np.argsort(-lp, kind="stable")[:TOP_K]
    cand_top = np.argsort(-lq, kind="stable")[:TOP_K]
    ps = np.sort(p)[::-1]
    return {
        "kl": float((p * (lp - lq)).sum()),
        "top1_agree": bool(int(np.argmax(lp)) == int(np.argmax(lq))),
        "top5_overlap": int(len(set(ref_top.tolist()) & set(cand_top.tolist()))),
        "delta_nll": None if next_token is None else float(lp[next_token] - lq[next_token]),
        "reference_margin": float(ps[0] - (ps[1] if len(ps) > 1 else 0.0)),
        "reference_entropy": float(-(p * lp).sum()),
    }


def compute():
    meta = json.loads((DIR / "samples.json").read_text())
    vocab = meta["vocab"]
    ref = np.fromfile(DIR / "reference.f32", dtype="<f4").reshape(-1, vocab)
    cand = np.fromfile(DIR / "candidate.f32", dtype="<f4").reshape(-1, vocab)
    rows, r = [], 0
    for sample in meta["samples"]:
        ids = sample["ids"]
        for i in range(len(ids)):
            nxt = ids[i + 1] if i + 1 < len(ids) else None
            rows.append(position(ref[r], cand[r], nxt))
            r += 1
    assert r == ref.shape[0] == cand.shape[0], (r, ref.shape, cand.shape)
    return {"generator": "scripts/measure_plan_w3_reference.py", "positions": rows}


def main():
    result = compute()
    path = DIR / "expected.json"
    if "--check" in sys.argv:
        committed = json.loads(path.read_text())
        assert committed == result, "expected.json does not match a fresh numpy computation"
        print(f"expected.json matches ({len(result['positions'])} positions)")
        return
    path.write_text(json.dumps(result, indent=1) + "\n")
    print(f"wrote {path} ({len(result['positions'])} positions)")


if __name__ == "__main__":
    main()
