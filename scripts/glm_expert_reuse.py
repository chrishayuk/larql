"""The reuse curves for GLM's expert bank, from a real-token trace.

Two curves from one trace, and neither is a policy:

  * **natural reuse** — of token t's selected experts, how many were used
    within the last k tokens?
  * **budgeted residency** — at a byte budget for this layer's bank, what
    fraction of selected bytes hit under LRU, and under an offline
    optimal (Bélády) upper bound?

The optimal arm is what makes a poor LRU number interpretable: it
separates "this policy is weak" from "this sequence has no locality to
exploit".
"""
import argparse, json
import numpy as np


def natural_reuse(sel, windows):
    out = {}
    for k in windows:
        hits, total = 0, 0
        for t in range(len(sel)):
            recent = set()
            for u in range(max(0, t - k), t):
                recent |= set(sel[u])
            hits += len(set(sel[t]) & recent)
            total += len(sel[t])
        out[k] = hits / total
    return out


def lru(sel, capacity):
    """Fraction of selected-expert slots already resident, under LRU.

    Returns `(overall, capacity_only)`. The split matters: an expert's
    FIRST use in a trace can never hit, so a finite trace has a ceiling
    of `1 - distinct/slots` no matter how large the cache. Reporting only
    the overall number makes a full-size cache look lossy when the misses
    left are all compulsory — here the 288-expert budget reads 78 %, and
    every one of the missing 22 % is a first touch.
    """
    if capacity <= 0:
        return 0.0, 0.0
    cache, hits, total, compulsory = [], 0, 0, 0
    seen = set()
    for s in sel:
        for e in s:
            total += 1
            if e not in seen:
                compulsory += 1
                seen.add(e)
            if e in cache:
                hits += 1
                cache.remove(e)
            cache.append(e)
            if len(cache) > capacity:
                cache.pop(0)
    warm = total - compulsory
    return hits / total, (hits / warm if warm else 0.0)


def belady(sel, capacity):
    """Offline optimal: evict whatever is needed furthest in the future."""
    if capacity <= 0:
        return 0.0
    # next_use[t][e] — the next position at or after t that needs e.
    n = len(sel)
    cache, hits, total = set(), 0, 0
    for t, s in enumerate(sel):
        for e in s:
            total += 1
            if e in cache:
                hits += 1
                continue
            if len(cache) >= capacity:
                # Evict the resident expert whose next use is furthest
                # away (or never).
                def next_use(x):
                    for u in range(t + 1, n):
                        if x in sel[u]:
                            return u
                    return n + 1

                victim = max(cache, key=next_use)
                # Never evict something this very token still needs.
                if next_use(victim) == t:
                    victim = max(cache - set(s), key=next_use, default=None)
                if victim is None:
                    continue
                cache.discard(victim)
            cache.add(e)
    return hits / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    args = ap.parse_args()
    d = json.load(open(args.trace))
    meta, trace = d["meta"], d["trace"]
    sel = [r["selected"] for r in trace]
    n = len(sel)
    eb = meta["expert_bytes"]
    per_token = meta["top_k"] * eb

    print(f"trace: {n} real tokens, layer {meta['layer']}, "
          f"top-{meta['top_k']} of {meta['experts']}")
    print(f"  bank {meta['bank_bytes']/2**30:.3f} GiB   "
          f"expert {eb/2**20:.1f} MiB   selected/token {per_token/2**20:.1f} MiB")
    uniq = len(set(e for s in sel for e in s))
    print(f"  distinct experts touched: {uniq} of {meta['experts']} "
          f"({uniq/meta['experts']:.1%})")
    ov = [r["overlap_prev"] for r in trace[1:]]
    print(f"  consecutive overlap: mean {np.mean(ov):.2f} of {meta['top_k']} "
          f"({np.mean(ov)/meta['top_k']:.1%})\n")

    print("natural reuse — fraction of a token's selected experts seen in the last k tokens")
    for k, f in natural_reuse(sel, [1, 2, 4, 8, 16, 32, 64]).items():
        print(f"  k={k:<3} {f:6.1%}   ⇒ {per_token*(1-f)/2**20:7.1f} MiB still to fetch")

    print("\nbudgeted residency for THIS layer's bank")
    print(f"  {'budget':>10}  {'experts':>8}  {'LRU hit':>9}  {'ex-1st':>8}  "
          f"{'optimal':>9}  {'LRU fetch/token':>16}")
    for gib in (0.25, 0.5, 1.0, 2.0, 4.0, 6.75):
        cap = int(gib * 2**30 / eb)
        h, warm = lru(sel, cap)
        o = belady(sel, cap)
        print(f"  {gib:>7.2f} GiB  {cap:>8}  {h:>8.1%}  {warm:>7.1%}  {o:>8.1%}  "
              f"{per_token*(1-h)/2**20:>13.1f} MiB")
    uniq_n = len(set(e for s in sel for e in s))
    slots = sum(len(s) for s in sel)
    print(f"\n  compulsory ceiling for this trace: {1 - uniq_n/slots:.1%} "
          f"({uniq_n} first touches over {slots} slots). A longer generation "
          f"raises it toward 100 %;\n  the `ex-1st` column already excludes "
          f"them and is the steady-state number.")


if __name__ == "__main__":
    main()
