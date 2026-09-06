#!/usr/bin/env python3
"""ADDRESS-SELECT-1 selector development — BURNED DATA ONLY.

Runs the preregistered selector families across the preregistered layer axis on
the already-observed campaign material, and selects exactly one configuration.
The select-v1 roster is NOT read by this script and must not be, until the
chosen configuration is recorded on the registry record.

Material (all burned by discrimination-v1):
  development  72 rows = 12 enroll + 24 train + 24 validation + 12 unknown
  evaluation   96 rows = 12 enroll + 48 test   + 24 alias      + 12 unknown
  states cached at L22 / L24 / L26 / L28 / L30, 2560-d, per row

Protocol, mirroring discrimination-v1 so the numbers are directly comparable to
its 1/72 and 3/72 (L26) and 5/72 and 7/72 (L30) accepted-and-correct figures:

  1. fit on development/train against development/enroll addresses
  2. choose the acceptance threshold on development/validation + /unknown,
     maximising accepted-and-correct subject to false acceptance <= 5%
  3. score on evaluation/test + /alias (72 positives) vs /unknown (12)
  4. select ONE configuration by accepted-and-correct, ties broken by lower
     false acceptance, then by lower rank of layer in the frozen axis

Families (frozen in the registry record):
  S1 nearest-address cosine                      known-weak baseline
  S2 factorised relation -> entity                the hypothesis under test
  S3 low-rank bilinear entity x relation scorer   joint alternative to S2

S0 (the editor's own native detector) is NOT evaluable on this split: the
discrimination rosters were probe-only, with enrollment addresses rather than
installed gate/up slots, so no detector exists for these records. Reported as
not-evaluable here rather than silently dropped; it remains the Stage 2
baseline on select-v1, where slots are installed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
DISC = CAMPAIGN / "results" / "discrimination-v1"

LAYERS = (22, 24, 26, 28, 30)
RELATIONS = ("capital", "currency", "language")
MAX_FALSE_ACCEPT = 0.05
SEED = 20260906


def load():
    caps = np.load(DISC / "captures.npz", allow_pickle=True)
    fixture = json.loads((DISC / "fixture.json").read_text())
    out = {}
    for split in ("development", "evaluation"):
        rows = fixture[split]
        states = {L: np.asarray(caps[f"{split}_L{L}"], dtype=np.float64) for L in LAYERS}
        assert states[LAYERS[0]].shape[0] == len(rows), f"{split} row/state mismatch"
        idx = {g: [i for i, r in enumerate(rows) if r["group"] == g]
               for g in sorted({r["group"] for r in rows})}
        out[split] = {"rows": rows, "states": states, "idx": idx}
    return out


def unit(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def relation_of(rows, i) -> str:
    return rows[i]["relation"]


def label_of(rows, i) -> int:
    return rows[i]["label"]


# ---------------------------------------------------------------- families

def s1_scores(Q, A, **_):
    """Nearest stored address, cosine."""
    return unit(Q) @ unit(A).T


def _ridge_multiclass(X, y, n_class, lam):
    Y = np.zeros((len(y), n_class))
    Y[np.arange(len(y)), y] = 1.0
    G = X.T @ X + lam * np.eye(X.shape[1])
    return np.linalg.solve(G, X.T @ Y)


def s2_scores(Q, A, *, Xtr, rel_tr, lab_tr, addr_rel, lam_rel, lam_ent, **_):
    """Factorised: predict the relation, then discriminate entity WITHIN it.

    Relation is fitted as a ridge classifier on the training queries. Entity is
    scored by cosine in a within-relation whitened space, so the metric spends
    its capacity on the axis the campaign showed to be the bottleneck.
    """
    rel_idx = {r: k for k, r in enumerate(RELATIONS)}
    Wr = _ridge_multiclass(unit(Xtr), np.array([rel_idx[r] for r in rel_tr]), len(RELATIONS), lam_rel)
    rel_pred = (unit(Q) @ Wr).argmax(axis=1)

    # within-relation whitening fitted on training queries grouped by entity
    Qn, An, Xn = unit(Q), unit(A), unit(Xtr)
    centres = {}
    for r in RELATIONS:
        rows = [i for i, rr in enumerate(rel_tr) if rr == r]
        if rows:
            centres[r] = Xn[rows].mean(axis=0)
    within = []
    for i, rr in enumerate(rel_tr):
        if rr in centres:
            within.append(Xn[i] - centres[rr])
    within = np.asarray(within) if within else Xn
    cov = within.T @ within / max(len(within), 1) + lam_ent * np.eye(Xn.shape[1])
    Wt = np.linalg.cholesky(np.linalg.inv(cov))

    Qw, Aw = unit(Qn @ Wt), unit(An @ Wt)
    scores = Qw @ Aw.T
    # hard-restrict the candidate set to the predicted relation
    mask = np.full(scores.shape, -np.inf)
    for i in range(scores.shape[0]):
        want = RELATIONS[rel_pred[i]]
        for j, ar in enumerate(addr_rel):
            if ar == want:
                mask[i, j] = 0.0
    return scores + mask


def s3_scores(Q, A, *, Xtr, Atr, match_tr, rank, lam, **_):
    """Low-rank bilinear match score q^T U V^T a, ridge-fitted on match labels."""
    Xn, An_tr = unit(Xtr), unit(Atr)
    feats = Xn[:, :, None] * An_tr[:, None, :]
    d = Xn.shape[1]
    # project to a low-rank basis via randomised SVD of the training query and
    # address matrices, then ridge-fit the small bilinear form in that basis
    rng = np.random.default_rng(SEED)
    Uq, _, _ = np.linalg.svd(Xn.T @ rng.standard_normal((Xn.shape[0], rank)), full_matrices=False)
    Ua, _, _ = np.linalg.svd(An_tr.T @ rng.standard_normal((An_tr.shape[0], rank)), full_matrices=False)
    Pq, Pa = Xn @ Uq, An_tr @ Ua
    Z = (Pq[:, :, None] * Pa[:, None, :]).reshape(len(Xn), -1)
    G = Z.T @ Z + lam * np.eye(Z.shape[1])
    w = np.linalg.solve(G, Z.T @ match_tr).reshape(rank, rank)
    del feats
    Zq, Za = unit(Q) @ Uq, unit(A) @ Ua
    return Zq @ w @ Za.T


# ---------------------------------------------------------------- scoring

def evaluate(scores, rows, q_idx, a_idx, threshold):
    """Return (accepted_and_correct, n_positive) or false-acceptance count."""
    acc = 0
    for r, qi in enumerate(q_idx):
        top = int(np.argmax(scores[r]))
        if scores[r, top] >= threshold and label_of(rows, a_idx[top]) == label_of(rows, qi):
            acc += 1
    return acc


def false_accepts(scores, threshold):
    return int(sum(1 for r in range(scores.shape[0]) if np.max(scores[r]) >= threshold))


def relation_entity_breakdown(scores, rows, q_idx, a_idx):
    rel_ok = ent_ok = 0
    for r, qi in enumerate(q_idx):
        top = a_idx[int(np.argmax(scores[r]))]
        rel_ok += relation_of(rows, top) == relation_of(rows, qi)
        ent_ok += label_of(rows, top) == label_of(rows, qi)
    return rel_ok, ent_ok


def pick_threshold(pos_scores, neg_scores, n_neg):
    """Highest accepted-and-correct subject to false acceptance <= 5%."""
    budget = int(np.floor(MAX_FALSE_ACCEPT * n_neg))
    cands = sorted({float(v) for v in np.concatenate(
        [pos_scores.max(axis=1), neg_scores.max(axis=1)])}, reverse=True)
    best = None
    for t in cands:
        fa = int(sum(1 for r in range(neg_scores.shape[0]) if np.max(neg_scores[r]) >= t))
        if fa <= budget:
            best = t
    return best if best is not None else float("inf")


def main() -> None:
    data = load()
    dev, ev = data["development"], data["evaluation"]
    results = []

    for L in LAYERS:
        Sd, Se = dev["states"][L], ev["states"][L]
        d_enroll, d_train = dev["idx"]["enroll"], dev["idx"]["train"]
        d_val, d_unk = dev["idx"]["validation"], dev["idx"]["unknown"]
        e_enroll, e_unk = ev["idx"]["enroll"], ev["idx"]["unknown"]
        e_pos = ev["idx"]["test"] + ev["idx"]["alias"]

        Ad, Ae = Sd[d_enroll], Se[e_enroll]
        addr_rel_d = [relation_of(dev["rows"], i) for i in d_enroll]
        addr_rel_e = [relation_of(ev["rows"], i) for i in e_enroll]
        Xtr = Sd[d_train]
        rel_tr = [relation_of(dev["rows"], i) for i in d_train]
        lab_tr = [label_of(dev["rows"], i) for i in d_train]

        # paired (query, address, match) training set for the bilinear arm
        pair_q, pair_a, pair_m = [], [], []
        for i, qi in enumerate(d_train):
            for j, aj in enumerate(d_enroll):
                pair_q.append(Xtr[i]); pair_a.append(Ad[j])
                pair_m.append(1.0 if label_of(dev["rows"], aj) == label_of(dev["rows"], qi) else 0.0)
        pair_q = np.asarray(pair_q); pair_a = np.asarray(pair_a); pair_m = np.asarray(pair_m)

        configs = [("S1", "nearest-address cosine", s1_scores, {})]
        for lam_rel in (1e-2, 1.0):
            for lam_ent in (1e-3, 1e-2, 1e-1):
                configs.append(("S2", f"factorised rel->ent lam_rel={lam_rel} lam_ent={lam_ent}",
                                s2_scores, {"lam_rel": lam_rel, "lam_ent": lam_ent}))
        for rank in (8, 16, 32):
            for lam in (1e-2, 1.0):
                configs.append(("S3", f"bilinear rank={rank} lam={lam}", s3_scores,
                                {"rank": rank, "lam": lam}))

        for family, name, fn, hp in configs:
            common_d = {"Xtr": Xtr, "rel_tr": rel_tr, "lab_tr": lab_tr, "addr_rel": addr_rel_d,
                        "Atr": pair_a, "match_tr": pair_m}
            common_d["Xtr"] = pair_q if family == "S3" else Xtr
            try:
                s_val = fn(Sd[d_val], Ad, **common_d, **hp)
                s_unk_d = fn(Sd[d_unk], Ad, **common_d, **hp)
                thr = pick_threshold(s_val, s_unk_d, len(d_unk))

                common_e = dict(common_d); common_e["addr_rel"] = addr_rel_e
                s_pos = fn(Se[e_pos], Ae, **common_e, **hp)
                s_unk_e = fn(Se[e_unk], Ae, **common_e, **hp)
            except np.linalg.LinAlgError as exc:
                results.append({"layer": L, "family": family, "config": name, "error": str(exc)})
                continue

            acc = evaluate(s_pos, ev["rows"], e_pos, e_enroll, thr)
            fa = false_accepts(s_unk_e, thr)
            rel_ok, ent_ok = relation_entity_breakdown(s_pos, ev["rows"], e_pos, e_enroll)
            results.append({
                "layer": L, "family": family, "config": name,
                "threshold": float(thr),
                "accepted_and_correct": acc, "n_positive": len(e_pos),
                "false_accepts": fa, "n_negative": len(e_unk),
                "relation_correct": rel_ok, "entity_correct": ent_ok,
            })

    ok = [r for r in results if "error" not in r]
    budget = int(np.floor(MAX_FALSE_ACCEPT * ok[0]["n_negative"])) if ok else 0
    eligible = [r for r in ok if r["false_accepts"] <= budget]
    pool = eligible or ok
    chosen = sorted(pool, key=lambda r: (-r["accepted_and_correct"], r["false_accepts"],
                                         LAYERS.index(r["layer"])))[0]

    body = json.dumps(results, sort_keys=True)
    out = {
        "seed": SEED,
        "material": "discrimination-v1 development (fit + threshold) and evaluation (selection); BURNED",
        "layers": list(LAYERS),
        "false_accept_budget_rows": budget,
        "prior_baseline_accepted_and_correct": {
            "L26_nearest": "1/72", "L26_ridge": "3/72",
            "L30_nearest": "5/72", "L30_ridge": "7/72",
        },
        "s0_note": "native detector not evaluable on this split (probe-only rosters, no installed slots)",
        "results": results,
        "chosen": chosen,
        "results_sha256": hashlib.sha256(body.encode()).hexdigest(),
    }
    (HERE / "development_scores.json").write_text(json.dumps(out, indent=2) + "\n")

    print(f"{'layer':>5} {'family':<4} {'acc&corr':>9} {'FA':>4} {'rel':>4} {'ent':>4}  config")
    for r in sorted(ok, key=lambda r: (-r["accepted_and_correct"], r["false_accepts"])):
        print(f"{r['layer']:>5} {r['family']:<4} "
              f"{r['accepted_and_correct']:>4}/{r['n_positive']:<4} "
              f"{r['false_accepts']:>2}/{r['n_negative']:<2} "
              f"{r['relation_correct']:>4} {r['entity_correct']:>4}  {r['config']}")
    print("\nCHOSEN:", json.dumps(chosen, indent=2))


if __name__ == "__main__":
    main()
