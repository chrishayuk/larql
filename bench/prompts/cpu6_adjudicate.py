#!/usr/bin/env python3
"""CPU-6 adjudicator: the paired non-inferiority test, and its controls.

    python3 cpu6_adjudicate.py <bank-outdir> --bank 3a
    python3 cpu6_adjudicate.py selftest

The experimental design is frozen; this file is the last remaining
degree of freedom, so it is written and adversarially tested BEFORE any
arm runs. Nothing here is tuned afterwards.

    D_p = mean_pos KL(candidate_p || BF16) - 2 * mean_pos KL(shipped_p || BF16)

    PASS iff  upper95( mean_p D_p )  <=  0

by stratified prompt bootstrap: resampling WITH REPLACEMENT within each
category, preserving that category's frozen count, so every replicate
carries the frozen composition rather than a fluctuating one.

**Each bank is adjudicated ALONE.** Pooling 3A and 3B into one
400-prompt test would be a different, weaker experiment: the protocol
requires two independent replications, not one larger sample.
"""
import json, os, sys
from collections import Counter, defaultdict
import numpy as np

# Frozen before any arm ran. The adjudicating result comes from THIS
# seed; another seed may be shown to agree, but does not adjudicate.
CPU6_BOOTSTRAP_SEED = 20260825
RESAMPLES = 20_000
MULTIPLIER = 2.0            # the 2x contract, reaffirmed in CPU6-VALIDATION.md
TARGETS = {"factual": 44, "prose": 29, "code": 29, "arithmetic": 29,
           "uncertain": 29, "structured": 23, "longform": 17}
CANDIDATE_ARITHMETIC = {"LARQL_CPU_ARITHMETIC": "q8xq8b",
                        "LARQL_CPU_ACT_BLOCK": "16",
                        "LARQL_CPU_ACT_CODE": "asymmetric"}
# The CANDIDATE IMPLEMENTATION freeze. Distinct from the protocol SHA
# and from whatever clean HEAD executes the arms.
CANDIDATE_SOURCE_SHA = "df36ca9fc6553b7e636416886ccc51e09a2142d3"


def refuse(why):
    raise SystemExit(f"REFUSED: {why}")


def rows_of(path):
    if not os.path.exists(path):
        refuse(f"missing arm: {path}")
    return json.load(open(path))


def by_prompt(rows):
    """Per-prompt mean KL, category, and position count."""
    kl, cat, n = defaultdict(list), {}, Counter()
    for r in rows:
        kl[r["id"]].append(r["kl"])
        cat[r["id"]] = r["category"]
        n[r["id"]] += 1
    return ({k: float(np.mean(v)) for k, v in kl.items()}, cat, n)


def check_arms(ship, cand, bank):
    """Loud controls. Nothing is silently intersected or dropped."""
    s_kl, s_cat, s_n = by_prompt(ship["rows"])
    c_kl, c_cat, c_n = by_prompt(cand["rows"])

    if set(s_kl) != set(c_kl):
        only_s, only_c = set(s_kl) - set(c_kl), set(c_kl) - set(s_kl)
        refuse(f"arm prompt-id sets differ — shipped-only {sorted(only_s)[:5]}, "
               f"candidate-only {sorted(only_c)[:5]}. Intersecting them would "
               f"silently drop observations.")
    ids = sorted(s_kl)
    if len(ids) != sum(TARGETS.values()):
        refuse(f"{len(ids)} prompts, expected {sum(TARGETS.values())}")
    got = dict(Counter(s_cat[i] for i in ids))
    if got != TARGETS:
        refuse(f"category composition {got} != frozen targets {TARGETS}")
    for i in ids:
        if s_n[i] != c_n[i]:
            refuse(f"prompt {i}: {s_n[i]} scored positions in shipped vs {c_n[i]} in "
                   f"candidate — a missing observation must not disappear quietly")
        if s_cat[i] != c_cat[i]:
            refuse(f"prompt {i}: category differs between arms")

    prov = cand.get("provenance", {})
    if prov.get("arithmetic") != CANDIDATE_ARITHMETIC:
        refuse(f"candidate arithmetic {prov.get('arithmetic')} != {CANDIDATE_ARITHMETIC}")
    src = prov.get("source", {})
    if not src.get("crates_identical_to_candidate_source"):
        refuse("candidate provenance does not certify crates/ identical to the "
               "candidate source freeze")
    if src.get("candidate_source_sha") != CANDIDATE_SOURCE_SHA:
        refuse(f"candidate source SHA {src.get('candidate_source_sha')} != "
               f"{CANDIDATE_SOURCE_SHA}")
    sprov = ship.get("provenance", {})
    # THE anchor must be the shipped path, with no CPU-6 arithmetic on it.
    if sprov.get("arithmetic"):
        refuse(f"the shipped ANCHOR carries arithmetic overrides {sprov['arithmetic']} — "
               f"a mislabelled anchor is worse than CPU-5's error")
    if sprov.get("arm") != "shipped":
        refuse(f"anchor arm is {sprov.get('arm')!r}, expected 'shipped'")
    for d, name in ((cand, "candidate"), (ship, "shipped")):
        if d.get("bank") not in (None, bank):
            refuse(f"{name} arm is for bank {d.get('bank')!r}, expected {bank!r}")
    return ids, s_kl, c_kl, s_cat


def bootstrap(ids, s_kl, c_kl, cat, seed=CPU6_BOOTSTRAP_SEED, n=RESAMPLES):
    """Stratified by category, preserving each category's frozen count."""
    rng = np.random.default_rng(seed)
    strata = defaultdict(list)
    for i in ids:
        strata[cat[i]].append(i)
    s = {k: np.array([s_kl[i] for i in v]) for k, v in strata.items()}
    c = {k: np.array([c_kl[i] for i in v]) for k, v in strata.items()}

    dmeans = np.empty(n)
    ratios = np.empty(n)
    for r in range(n):
        ds, cs, ss = [], [], []
        for k, members in strata.items():
            pick = rng.integers(0, len(members), size=TARGETS[k])
            sv, cv = s[k][pick], c[k][pick]
            ds.append(cv - MULTIPLIER * sv)
            cs.append(cv)
            ss.append(sv)
        dmeans[r] = np.concatenate(ds).mean()
        # SAME resamples, so the descriptive ratio is paired with the gate.
        ratios[r] = np.concatenate(cs).mean() / np.concatenate(ss).mean()
    return dmeans, ratios


def hard_gates(cand_rows):
    flips = [r for r in cand_rows if r["flip"]]
    return {"top1": 1 - len(flips) / len(cand_rows),
            "flips": len(flips),
            "flips_hi": sum(1 for r in flips if r["margin"] >= 0.10),
            "worst_margin": max((r["margin"] for r in flips), default=0.0)}


def adjudicate(outdir, bank):
    ship = rows_of(os.path.join(outdir, "compare-shipped.json"))
    cand = rows_of(os.path.join(outdir, "compare-candidate.json"))
    ids, s_kl, c_kl, cat = check_arms(ship, cand, bank)

    S = np.array([s_kl[i] for i in ids]); C = np.array([c_kl[i] for i in ids])
    D = C - MULTIPLIER * S
    dmeans, ratios = bootstrap(ids, s_kl, c_kl, cat)
    upper95 = float(np.percentile(dmeans, 95))
    hg = hard_gates(cand["rows"])
    gates = {"primary upper95(D) <= 0": upper95 <= 0,
             "top-1 >= 99%": hg["top1"] >= 0.99,
             "flips at margin >= 0.10 == 0": hg["flips_hi"] == 0}

    print(f"=== CPU-6 bank {bank} — adjudicated ALONE ===")
    src = cand.get("provenance", {}).get("source", {})
    print(f"prompts {len(ids)}   seed {CPU6_BOOTSTRAP_SEED}   resamples {RESAMPLES}")
    print(f"  candidate source {src.get('candidate_source_sha','?')[:12]}   "
          f"protocol {src.get('protocol_sha','?')[:12]}   "
          f"executed at {src.get('execution_head','?')[:12]}")
    print(f"  mean shipped KL      {S.mean():.9f}")
    print(f"  mean candidate KL    {C.mean():.9f}")
    print(f"  ratio of means       {C.mean()/S.mean():.3f}x   "
          f"95% bootstrap [{np.percentile(ratios,2.5):.3f}, {np.percentile(ratios,97.5):.3f}]  (descriptive)")
    print(f"  mean D               {D.mean():+.9f}")
    print(f"  upper95(mean D)      {upper95:+.9f}")
    print(f"  top-1 {hg['top1']:.4%}   flips {hg['flips']}   at margin>=0.10 {hg['flips_hi']}"
          f"   worst margin {hg['worst_margin']:.5f}")
    print()
    for k, v in gates.items():
        print(f"  {'ok  ' if v else 'FAIL'}  {k}")
    verdict = "PASS" if all(gates.values()) else "FAIL"
    print(f"\nBANK {bank} VERDICT: {verdict}")
    print("\nA failure here is a failure to VALIDATE, not proof of material inferiority.")
    return verdict


# ----------------------------------------------------------------- selftest
def selftest():
    """The adjudicator gets controls too."""
    rng = np.random.default_rng(7)
    ids, cat = [], {}
    for k, n in TARGETS.items():
        for j in range(n):
            i = f"{k}-{j:03d}"; ids.append(i); cat[i] = k
    base = {i: float(abs(rng.normal(1.5e-4, 6e-5)) + 1e-6) for i in ids}

    def run(mult):
        s = dict(base); c = {i: mult * base[i] for i in ids}
        d, _ = bootstrap(ids, s, c, cat, n=2000)
        return float(np.percentile(d, 95))

    lo, hi = run(1.5), run(2.5)
    print(f"candidate = 1.5x shipped  upper95 {lo:+.3e}   expect <= 0   {'ok' if lo <= 0 else 'FAIL'}")
    print(f"candidate = 2.5x shipped  upper95 {hi:+.3e}   expect  > 0   {'ok' if hi > 0 else 'FAIL'}")

    # **The boundary case must carry variance to mean anything.** An exact
    # 2.0x relationship makes D identically zero, so upper95 is exactly 0
    # and the gate passes — correct against a `<=` contract, but it tests
    # nothing, because a degenerate constant has no sampling distribution.
    # The real boundary is 2x with scatter: mean D ~ 0, variance > 0, so
    # the one-sided bound sits ABOVE zero and non-inferiority is NOT
    # demonstrated. That is the "failure to validate" case by design.
    noisy = {i: 2.0 * base[i] * float(rng.normal(1.0, 0.25)) for i in ids}
    dn, _ = bootstrap(ids, dict(base), noisy, cat, n=4000)
    edge = float(np.percentile(dn, 95))
    edge_mean = float(np.mean(dn))
    print(f"candidate = 2.0x shipped + scatter  mean D {edge_mean:+.3e}  upper95 {edge:+.3e}   "
          f"expect mean ~0 but upper95 > 0 (not validated)   {'ok' if edge > 0 else 'FAIL'}")

    # Every replicate must preserve the frozen composition, even when one
    # category is extreme enough to dominate if the strata were pooled.
    s2 = dict(base); c2 = {i: (12.0 if cat[i] == "longform" else 1.5) * base[i] for i in ids}
    rng2 = np.random.default_rng(CPU6_BOOTSTRAP_SEED)
    strata = defaultdict(list)
    for i in ids:
        strata[cat[i]].append(i)
    counts = Counter()
    for _ in range(500):
        for k, members in strata.items():
            counts[k] += len(rng2.integers(0, len(members), size=TARGETS[k]))
    per = {k: counts[k] // 500 for k in TARGETS}
    print(f"stratification preserved every replicate: {per == TARGETS}   {per}")
    d3, _ = bootstrap(ids, s2, c2, cat, n=2000)
    print(f"one extreme category is NOT diluted: upper95 {np.percentile(d3,95):+.3e} (expect > 0) "
          f"{'ok' if np.percentile(d3,95) > 0 else 'FAIL'}")
    ok = lo <= 0 and hi > 0 and edge > 0 and per == TARGETS and np.percentile(d3, 95) > 0
    print(f"\nSELFTEST: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    if sys.argv[1] == "selftest":
        raise SystemExit(0 if selftest() else 1)
    bank = sys.argv[sys.argv.index("--bank") + 1]
    raise SystemExit(0 if adjudicate(sys.argv[1], bank) == "PASS" else 1)
