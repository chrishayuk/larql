#!/usr/bin/env python3
"""CPU-5 analysis: apply the FROZEN gates, and test for interaction.

    python3 cpu5_analyse.py <bank-outdir> [--arms a,b,c] [--vectors]

Two jobs, deliberately separate.

**1. Gates.** The thresholds come from AMENDMENT 2 of the frozen spec and
are applied mechanically. A2 (`shipped`) supplies the anchor `K2`, and
the bands are multiples of it, so nothing here is a number chosen after
seeing a result.

**2. Interaction.** Absolute KLs cannot say whether the weight and the
activation perturbations are independent. With `--vectors` (requires the
arms to have been compared with `--keep`) this measures the ANGLE between
the two error vectors, which can:

```
cos ~  0   independent — the errors add in quadrature
cos >  0   they REINFORCE, and a mixed-precision plan cannot be reasoned
           about one operand at a time
cos <  0   they partially cancel, and isolated arms OVERSTATE the damage
```

That distinction matters more than either arm's KL: if the perturbations
interact, the per-class marginal analysis the exception search rests on
is invalid rather than merely noisy.
"""
import json, os, sys
import numpy as np

# --- AMENDMENT 2, frozen before any rescue rung ran -----------------
ACCEPT_KL_MULTIPLE = 2.0      # G1: KL mean <= 2 x K2
P99_MULTIPLE = 10.0           # G2: KL p99 <= 10 x A2's p99
MIN_TOP1 = 0.99               # G3
HIGH_MARGIN = 0.10            # G4: zero flips at or above this BF16 margin
ANCHOR = "shipped"


def q(a, p):
    return float(np.percentile(a, p)) if len(a) else float("nan")


def load(outdir, label):
    path = os.path.join(outdir, f"compare-{label}.json")
    if not os.path.exists(path):
        return None
    return json.load(open(path))


def stats(d):
    rows = d["rows"]
    kl = np.array([r["kl"] for r in rows])
    flips = [r for r in rows if r["flip"]]
    dn = np.array([r["dnll"] for r in rows if r["dnll"] is not None])
    return {
        "positions": len(rows),
        "kl_mean": kl.mean(), "kl_p95": q(kl, 95), "kl_p99": q(kl, 99), "kl_max": kl.max(),
        "top1": 1.0 - len(flips) / len(rows),
        "flips": len(flips),
        "flips_hi": sum(1 for r in flips if r["margin"] >= HIGH_MARGIN),
        "flips_01": sum(1 for r in flips if r["margin"] >= 0.01),
        "dnll_mean": dn.mean() if len(dn) else float("nan"),
        "dmax": np.array([r["dmax"] for r in rows]).mean(),
    }


def main():
    outdir = sys.argv[1]
    arms = (sys.argv[sys.argv.index("--arms") + 1].split(",")
            if "--arms" in sys.argv else
            [os.path.basename(p)[8:-5] for p in sorted(
                __import__("glob").glob(os.path.join(outdir, "compare-*.json")))])

    data = {a: load(outdir, a) for a in arms}
    data = {a: d for a, d in data.items() if d}
    if ANCHOR not in data:
        print(f"!! anchor `{ANCHOR}` absent — the bands are multiples of it and "
              f"cannot be evaluated without it")
    S = {a: stats(d) for a, d in data.items()}

    print(f"{'arm':<12}{'pos':>7}{'KL mean':>10}{'KL p95':>10}{'KL p99':>10}"
          f"{'top-1':>9}{'flips':>7}{'>=.01':>7}{'>=.10':>7}{'dNLL':>10}")
    for a in data:
        s = S[a]
        print(f"{a:<12}{s['positions']:>7}{s['kl_mean']:>10.5f}{s['kl_p95']:>10.5f}"
              f"{s['kl_p99']:>10.5f}{s['top1']:>8.2%}{s['flips']:>7}"
              f"{s['flips_01']:>7}{s['flips_hi']:>7}{s['dnll_mean']:>+10.5f}")

    if ANCHOR in S:
        k2, p99 = S[ANCHOR]["kl_mean"], S[ANCHOR]["kl_p99"]
        g1, g2 = ACCEPT_KL_MULTIPLE * k2, P99_MULTIPLE * p99
        print(f"\nfrozen gates from the anchor:  G1 KL mean <= {g1:.5f}   "
              f"G2 KL p99 <= {g2:.5f}   G3 top-1 >= {MIN_TOP1:.0%}   "
              f"G4 flips at margin >= {HIGH_MARGIN} == 0")
        print(f"{'arm':<12}{'G1':>6}{'G2':>6}{'G3':>6}{'G4':>6}   verdict")
        for a in data:
            s = S[a]
            g = [s["kl_mean"] <= g1, s["kl_p99"] <= g2,
                 s["top1"] >= MIN_TOP1, s["flips_hi"] == 0]
            band = ("NEGLIGIBLE" if s["kl_mean"] <= k2 else
                    "ACCEPT" if s["kl_mean"] <= g1 else
                    "MIXED REQUIRED" if s["kl_mean"] <= 5 * k2 else "REJECT")
            mark = lambda b: "  ok" if b else "FAIL"
            print(f"{a:<12}{mark(g[0]):>6}{mark(g[1]):>6}{mark(g[2]):>6}{mark(g[3]):>6}   "
                  f"{'PASS' if all(g) else 'FAIL'} — {band}")

    # --- interaction, at KL level (always available) -----------------
    need = ("bf16xq8b", "q8xq8b")
    if ANCHOR in S and all(n in S for n in need):
        a_cost = S["bf16xq8b"]["kl_mean"]           # activation alone, exact weights
        w_cost = S[ANCHOR]["kl_mean"]               # Q8 weights alone, f32 activation
        both = S["q8xq8b"]["kl_mean"]               # both
        print(f"\ninteraction at KL level (not additive by construction — the question "
              f"is only whether\nthe combined arm is FAR from what its parts predict):")
        print(f"  activation alone (BF16 x Q8[64])   {a_cost:.5f}")
        print(f"  weights alone    (Q8 x F32)        {w_cost:.5f}")
        print(f"  sum of parts                       {a_cost + w_cost:.5f}")
        print(f"  measured together (Q8 x Q8[64])    {both:.5f}"
              f"   ratio {both / (a_cost + w_cost):.2f}x")

    # --- interaction, at VECTOR level (needs --keep dumps) -----------
    if "--vectors" in sys.argv:
        vector_interaction(outdir, data)


def entry_pairs(outdir, meta, label):
    d = os.path.join(outdir, f"_cand-{label}")
    if not os.path.isdir(d):
        return None
    return d


def vector_interaction(outdir, data):
    meta = json.load(open(os.path.join(outdir, "reference.json")))
    dirs = {a: entry_pairs(outdir, meta, a) for a in data}
    dirs = {a: p for a, p in dirs.items() if p}
    if not {"bf16xq8b", "q8xq8b"} <= set(dirs):
        print("\n(vector interaction needs bf16xq8b and q8xq8b compared with --keep)")
        return

    print("\nerror-vector interaction, over the whole bank:")
    print("  e_a  = BF16xQ8[64] - reference        the ACTIVATION perturbation")
    print("  e_w  = Q8xQ8[64]   - BF16xQ8[64]      the WEIGHT perturbation, same activation")
    # Two spaces, and only the second one is about the model's predictions.
    #
    # Logits are SHIFT-INVARIANT under softmax: adding a constant to every
    # logit at a position changes no probability. That all-ones direction
    # is therefore an exactly-identified nuisance, and a correlation
    # measured along it is a correlation between two things the model does
    # not express. Projecting it out is not mean-subtraction of a
    # data-derived statistic — it is removing a known symmetry of the
    # objective, which is the only case where subtracting a mean IS the
    # projection.
    raw = {"num": 0.0, "a": 0.0, "w": 0.0}
    tan = {"num": 0.0, "a": 0.0, "w": 0.0}
    for e in meta["entries"]:
        n = len(e["ids"])
        ref = np.fromfile(os.path.join(outdir, e["dump"]),
                          dtype=np.float32).astype(np.float64).reshape(n, -1)
        a = np.fromfile(os.path.join(dirs["bf16xq8b"], f"{e['id']}.f32"),
                        dtype=np.float32).astype(np.float64).reshape(n, -1)
        w = np.fromfile(os.path.join(dirs["q8xq8b"], f"{e['id']}.f32"),
                        dtype=np.float32).astype(np.float64).reshape(n, -1)
        ea, ew = a - ref, w - a
        raw["num"] += float((ea * ew).sum())
        raw["a"] += float((ea * ea).sum())
        raw["w"] += float((ew * ew).sum())
        # Per POSITION, because the symmetry is per softmax.
        ca = ea - ea.mean(1, keepdims=True)
        cw = ew - ew.mean(1, keepdims=True)
        tan["num"] += float((ca * cw).sum())
        tan["a"] += float((ca * ca).sum())
        tan["w"] += float((cw * cw).sum())

    def cosine(d):
        return d["num"] / (d["a"] ** 0.5 * d["w"] ** 0.5) if d["a"] and d["w"] else float("nan")

    cos_raw, cos = cosine(raw), cosine(tan)
    print(f"  raw logit space      ||e_a|| {raw['a'] ** 0.5:.4e}   "
          f"||e_w|| {raw['w'] ** 0.5:.4e}   cos {cos_raw:+.4f}")
    print(f"  softmax tangent      ||e_a|| {tan['a'] ** 0.5:.4e}   "
          f"||e_w|| {tan['w'] ** 0.5:.4e}   cos {cos:+.4f}   <- the one that matters")
    shift = 1.0 - tan["a"] / raw["a"]
    print(f"  common-mode share of the activation error: {shift:.1%} "
          f"(shift-invariant, changes no probability)")
    verdict = ("INDEPENDENT — errors add in quadrature; per-class marginal analysis is sound"
               if abs(cos) < 0.1 else
               "REINFORCING — the perturbations are not separable, and per-operand marginal "
               "reasoning is invalid" if cos > 0 else
               "CANCELLING — isolated arms OVERSTATE the combined damage")
    print(f"  => {verdict}")


if __name__ == "__main__":
    main()
