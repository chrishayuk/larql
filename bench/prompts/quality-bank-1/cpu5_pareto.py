#!/usr/bin/env python3
"""CPU-5 rescue frontier: what does each restored class BUY, and COST?

    python3 cpu5_pareto.py <screen-outdir> <census-dir> [--baseline R0-all]

Joins the two halves no single number can carry:

    cost      the byte census, priced by the CPU-4Y cost model
    quality   KL, dNLL and margin-conditioned flips on the Bank-1 screen

and reports each rung's MARGINAL effect against blanket Q4, because
absolute numbers do not order operand families — differences do.

The ratios at the end (`KL recovered per added GB`, `per added ms`) are
EXPLANATORY, never the selection metric. AMENDMENT 2 selects on predicted
real ms among plans that clear the gates, precisely because a ratio
rewards a tiny denominator: an operand class worth 1 GB and a whisker of
quality can top a per-GB table while being useless.
"""
import json, os, re, sys, glob
import numpy as np


def parse_census(path):
    """Bytes per plan and the model's own predicted ms, from a run's output."""
    plans, syn, real = {}, None, None
    for line in open(path):
        m = re.match(r"\s+(\w+)\s+([\d.]+) GB over\s+(\d+) calls", line)
        if m:
            plans[m.group(1)] = {"gb": float(m.group(2)), "calls": int(m.group(3))}
        m = re.search(r"predicted from bytes: ([\d.]+) ms synthetic, ([\d.]+) ms real", line)
        if m:
            syn, real = float(m.group(1)), float(m.group(2))
    return {"plans": plans, "synthetic_ms": syn, "real_ms": real,
            "total_gb": sum(p["gb"] for p in plans.values())}


def quality(outdir, label):
    path = os.path.join(outdir, f"compare-{label}.json")
    if not os.path.exists(path):
        return None
    rows = json.load(open(path))["rows"]
    kl = np.array([r["kl"] for r in rows])
    flips = [r for r in rows if r["flip"]]
    dn = np.array([r["dnll"] for r in rows if r["dnll"] is not None])
    return {"positions": len(rows), "kl_mean": float(kl.mean()),
            "kl_p99": float(np.percentile(kl, 99)),
            "top1": 1.0 - len(flips) / len(rows),
            "flips_01": sum(1 for r in flips if r["margin"] >= 0.01),
            "flips_10": sum(1 for r in flips if r["margin"] >= 0.10),
            "dnll": float(dn.mean()) if len(dn) else float("nan")}


def main():
    screen, census_dir = sys.argv[1], sys.argv[2]
    base_label = (sys.argv[sys.argv.index("--baseline") + 1]
                  if "--baseline" in sys.argv else "R0-all")

    rungs = []
    for c in sorted(glob.glob(os.path.join(census_dir, "*.census"))):
        label = os.path.basename(c)[:-7]
        qual = quality(screen, label)
        rungs.append({"label": label, "cost": parse_census(c), "quality": qual})
    if not rungs:
        raise SystemExit(f"no .census files in {census_dir}")

    print(f"{'rung':<16}{'GB/tok':>9}{'pred ms':>9}{'KL mean':>10}{'KL p99':>10}"
          f"{'top-1':>9}{'>=.01':>7}{'>=.10':>7}")
    for r in rungs:
        c, q = r["cost"], r["quality"]
        if q is None:
            print(f"{r['label']:<16}{c['total_gb']:>9.2f}{c['real_ms'] or 0:>9.1f}"
                  f"{'(quality pending)':>43}")
            continue
        print(f"{r['label']:<16}{c['total_gb']:>9.2f}{c['real_ms']:>9.1f}"
              f"{q['kl_mean']:>10.5f}{q['kl_p99']:>10.5f}{q['top1']:>8.2%}"
              f"{q['flips_01']:>7}{q['flips_10']:>7}")

    base = next((r for r in rungs if r["label"].startswith(base_label)), None)
    if not base or base["quality"] is None:
        print(f"\n(no baseline `{base_label}` with quality — marginals need it)")
        return

    bc, bq = base["cost"], base["quality"]
    print(f"\nmarginal against {base['label']} (blanket Q4):")
    print(f"{'rung':<16}{'d GB':>8}{'d ms':>8}{'d KL':>11}{'d dNLL':>10}"
          f"{'d>=.01':>8}{'KL/GB':>11}{'KL/ms':>11}")
    for r in rungs:
        if r is base or r["quality"] is None:
            continue
        c, q = r["cost"], r["quality"]
        dgb = c["total_gb"] - bc["total_gb"]
        dms = c["real_ms"] - bc["real_ms"]
        dkl = q["kl_mean"] - bq["kl_mean"]          # negative = quality RECOVERED
        rec = -dkl
        per_gb = rec / dgb if dgb > 1e-9 else float("nan")
        per_ms = rec / dms if dms > 1e-9 else float("nan")
        print(f"{r['label']:<16}{dgb:>+8.2f}{dms:>+8.1f}{dkl:>+11.5f}"
              f"{q['dnll'] - bq['dnll']:>+10.5f}{q['flips_01'] - bq['flips_01']:>+8}"
              f"{per_gb:>11.5f}{per_ms:>11.5f}")
    print("\nKL/GB and KL/ms are EXPLANATORY. Selection is AMENDMENT 2: lowest predicted")
    print("real ms among plans clearing the gates on the FULL bank, ties to fewer")
    print("restored classes, then fewer bytes. A ratio rewards a small denominator.")


if __name__ == "__main__":
    main()
