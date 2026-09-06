#!/usr/bin/env python3
"""ADDRESS-SUPERPOSITION-1 Stage A — cached diagnostic, zero model forwards.

Is entity-relation binding hidden underneath a surface-form subspace, or is the
address genuinely different until later computation resolves it?

Estimates a surface-form subspace from within-record form differences on the
discrimination-v1 DEVELOPMENT split, removes it, and re-scores the nearest-
address reader on the EVALUATION split. The two splits share no entity and no
form, so the cross-fit is double: a subspace that only helps when fitted on the
entities it is scored on is memorisation, not a shared wording direction.

Arms (preregistered):
  A0 raw
  A1 form-subspace removed                     hypothesis
  A2 matched-rank random subspace removed      capacity control
  A3 entity-discriminative subspace removed    destructive positive control

Gate: A1 must beat BOTH A0 and A2 by >= 10 points of binding accuracy at >= 2
of ranks {4,8,16,32}, with relation dropping <= 5 points, and A3 must degrade.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DISC = HERE.parent / "results" / "discrimination-v1"
LAYERS = (22, 24, 26, 28, 30)
RANKS = (4, 8, 16, 32)
SEED = 20260906
GAIN_POINTS = 10.0
RELATION_TOLERANCE = 5.0


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def load():
    caps = np.load(DISC / "captures.npz", allow_pickle=True)
    fx = json.loads((DISC / "fixture.json").read_text())
    out = {}
    for split in ("development", "evaluation"):
        rows = fx[split]
        out[split] = {
            "rows": rows,
            "states": {L: np.asarray(caps[f"{split}_L{L}"], np.float64) for L in LAYERS},
        }
    return out


def basis(diffs, rank):
    """Top-`rank` right singular directions of a set of difference vectors."""
    if len(diffs) == 0:
        return np.zeros((diffs.shape[1], 0))
    _, _, Vt = np.linalg.svd(np.asarray(diffs), full_matrices=False)
    return Vt[:rank].T


def form_diffs(rows, S):
    """Within-record differences: same entity AND relation, different wording."""
    by_label = {}
    for i, r in enumerate(rows):
        if r.get("label") is not None:
            by_label.setdefault(r["label"], []).append(i)
    out = []
    for idxs in by_label.values():
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                out.append(S[idxs[a]] - S[idxs[b]])
    return np.asarray(out)


def entity_diffs(rows, S):
    """Between-entity differences at matched relation (destructive control)."""
    by_rel = {}
    for i, r in enumerate(rows):
        if r.get("label") is not None:
            by_rel.setdefault(r["relation"], []).append(i)
    out = []
    for idxs in by_rel.values():
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                if rows[idxs[a]]["label"] != rows[idxs[b]]["label"]:
                    out.append(S[idxs[a]] - S[idxs[b]])
    return np.asarray(out)


def project_out(X, U):
    return X if U.shape[1] == 0 else X - (X @ U) @ U.T


def score(rows, S, U):
    """Nearest-address reader after removing subspace U from queries and addresses."""
    enroll = [i for i, r in enumerate(rows) if r["group"] == "enroll"]
    q = [i for i, r in enumerate(rows) if r["group"] in ("test", "alias")]
    A = unit(project_out(S[enroll], U))
    Q = unit(project_out(S[q], U))
    top = (Q @ A.T).argmax(axis=1)
    rel = sum(rows[enroll[t]]["relation"] == rows[i]["relation"] for i, t in zip(q, top))
    bind = sum(rows[enroll[t]]["label"] == rows[i]["label"] for i, t in zip(q, top))
    ent = sum(
        rows[enroll[t]]["label"] // 3 == rows[i]["label"] // 3 for i, t in zip(q, top)
    )
    return {"n": len(q), "relation": rel, "entity": ent, "binding": bind}


def main() -> None:
    data = load()
    dev, ev = data["development"], data["evaluation"]
    rng = np.random.default_rng(SEED)
    results = []

    for L in LAYERS:
        Sd, Se = dev["states"][L], ev["states"][L]
        fd = form_diffs(dev["rows"], Sd)
        ed = entity_diffs(dev["rows"], Sd)
        base = score(ev["rows"], Se, np.zeros((Se.shape[1], 0)))
        results.append({"layer": L, "arm": "A0_raw", "rank": 0, **base})

        for rank in RANKS:
            for arm, U in (
                ("A1_form_removed", basis(fd, rank)),
                ("A2_random_removed", np.linalg.qr(rng.standard_normal((Se.shape[1], rank)))[0]),
                ("A3_entity_removed", basis(ed, rank)),
            ):
                results.append({"layer": L, "arm": arm, "rank": rank,
                                **score(ev["rows"], Se, U)})

    # gate
    verdict = {}
    for L in LAYERS:
        raw = next(r for r in results if r["layer"] == L and r["arm"] == "A0_raw")
        n = raw["n"]
        wins = []
        for rank in RANKS:
            a1 = next(r for r in results if r["layer"] == L and r["arm"] == "A1_form_removed" and r["rank"] == rank)
            a2 = next(r for r in results if r["layer"] == L and r["arm"] == "A2_random_removed" and r["rank"] == rank)
            gain_raw = 100.0 * (a1["binding"] - raw["binding"]) / n
            gain_rand = 100.0 * (a1["binding"] - a2["binding"]) / n
            rel_drop = 100.0 * (raw["relation"] - a1["relation"]) / n
            if gain_raw >= GAIN_POINTS and gain_rand >= GAIN_POINTS and rel_drop <= RELATION_TOLERANCE:
                wins.append(rank)
        a3 = [next(r for r in results if r["layer"] == L and r["arm"] == "A3_entity_removed" and r["rank"] == k)
              for k in RANKS]
        verdict[f"L{L}"] = {
            "ranks_clearing": wins,
            "rank_stable_pass": len(wins) >= 2,
            "destructive_control_degrades": all(x["binding"] <= raw["binding"] for x in a3),
        }
    passed = any(v["rank_stable_pass"] and v["destructive_control_degrades"] for v in verdict.values())

    body = json.dumps(results, sort_keys=True)
    out = {
        "material": "discrimination-v1 development (fit) -> evaluation (score); disjoint entities AND forms",
        "ranks": list(RANKS), "seed": SEED,
        "gate": {"gain_points": GAIN_POINTS, "relation_tolerance": RELATION_TOLERANCE,
                 "ranks_required": 2},
        "results": results, "verdict": verdict,
        "GATE": "PASS" if passed else "FAIL",
        "results_sha256": hashlib.sha256(body.encode()).hexdigest(),
    }
    (HERE / "superposition_a_result.json").write_text(json.dumps(out, indent=2) + "\n")

    print(f"{'layer':>5} {'arm':<18} {'rank':>4} {'rel':>7} {'ent':>7} {'bind':>7}")
    for r in results:
        print(f"{r['layer']:>5} {r['arm']:<18} {r['rank']:>4} "
              f"{r['relation']:>3}/{r['n']:<3} {r['entity']:>3}/{r['n']:<3} {r['binding']:>3}/{r['n']:<3}")
    print("\nVERDICT:", json.dumps(verdict, indent=2))
    print("GATE:", out["GATE"], " results_sha256", out["results_sha256"])


if __name__ == "__main__":
    main()
