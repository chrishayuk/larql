"""ANE-3 analyser — turn three conditions into the four-outcome verdict.

Reports two separate things, because reducing the rung to
`t_both ~ max()` vs `sum()` would throw away the asymmetry that
distinguishes hard contention from bubble recovery:

  1. per-engine wall-clock degradation   tX(GA) / tX(X alone)
  2. delivered useful work               iterations x bytes / window

Only samples that started AND finished inside the interval where both
engines were genuinely running are counted for GA. Individual operations
are ~0.6 ms (GPU) and ~1.6 ms (ANE), so a launch skew of a few hundred
microseconds could otherwise manufacture overlap.

Usage:
    python ane3_analyze.py <run_root> [out.json]
"""

import json
import os
import sys

# ANE-0b's frozen baseline, for the record. ANE-3's own G condition is the
# operative control; this is carried to show the two agree.
ANE0B_FROZEN_GBS = (288.7, 289.1)


def load(path):
    with open(path) as fh:
        return json.load(fh)


def stats(ms):
    s = sorted(ms)
    n = len(s)
    if n == 0:
        return {"min": float("nan"), "p50": float("nan"), "n": 0}
    return {"min": s[0], "p50": s[n // 2], "p90": s[(n * 9) // 10], "n": n}


def in_window(doc, lo, hi):
    """Samples fully inside [lo, hi]."""
    out = []
    for start, ms in zip(doc["sample_start_epoch"], doc["sample_ms"]):
        if start >= lo and (start + ms / 1e3) <= hi:
            out.append(ms)
    return out


def throughput_gbs(count, weight_bytes, seconds):
    return count * weight_bytes / seconds / 1e9 if seconds > 0 else float("nan")


def main():
    root = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else None

    g = load(os.path.join(root, "G", "gpu.json"))
    a = load(os.path.join(root, "A", "ane.json"))
    ga_g = load(os.path.join(root, "GA", "gpu.json"))
    ga_a = load(os.path.join(root, "GA", "ane.json"))

    wb = g["weight_bytes"]

    # Isolated conditions: the whole window is valid.
    g_solo = stats(g["sample_ms"])
    a_solo = stats(a["sample_ms"])
    g_solo_gbs = throughput_gbs(g["iters"], wb, g["window_end"] - g["window_start"])
    a_solo_gbs = throughput_gbs(a["iters"], wb, a["window_end"] - a["window_start"])

    # Concurrent: intersect the two active windows.
    lo = max(ga_g["window_start"], ga_a["window_start"])
    hi = min(ga_g["window_end"], ga_a["window_end"])
    overlap_s = hi - lo
    g_conc_ms = in_window(ga_g, lo, hi)
    a_conc_ms = in_window(ga_a, lo, hi)
    g_conc = stats(g_conc_ms)
    a_conc = stats(a_conc_ms)
    g_conc_gbs = throughput_gbs(len(g_conc_ms), wb, overlap_s)
    a_conc_gbs = throughput_gbs(len(a_conc_ms), wb, overlap_s)

    print(f"ANE-3 — overlap window {overlap_s:.3f} s")
    print(f"  GPU samples inside overlap: {len(g_conc_ms)} of {ga_g['iters']}")
    print(f"  ANE samples inside overlap: {len(a_conc_ms)} of {ga_a['iters']}")
    print(f"  ANE-0b frozen baseline: {ANE0B_FROZEN_GBS[0]}/{ANE0B_FROZEN_GBS[1]} GB/s\n")

    print(f"{'':<8}{'alone min ms':>14}{'conc min ms':>14}{'latency x':>12}"
          f"{'alone GB/s':>13}{'conc GB/s':>12}{'kept':>8}")
    for name, solo, conc, sg, cg in (
        ("GPU", g_solo, g_conc, g_solo_gbs, g_conc_gbs),
        ("ANE", a_solo, a_conc, a_solo_gbs, a_conc_gbs),
    ):
        print(
            f"{name:<8}{solo['min']:>14.3f}{conc['min']:>14.3f}"
            f"{conc['min'] / solo['min']:>12.2f}{sg:>13.1f}{cg:>12.1f}"
            f"{cg / sg:>8.2f}"
        )

    solo_sum = g_solo_gbs + a_solo_gbs
    conc_sum = g_conc_gbs + a_conc_gbs
    print(f"\n{'sum':<8}{'':<28}{'':<12}{solo_sum:>13.1f}{conc_sum:>12.1f}"
          f"{conc_sum / solo_sum:>8.2f}")

    gpu_kept = g_conc_gbs / g_solo_gbs
    ane_kept = a_conc_gbs / a_solo_gbs
    aggregate_gain = conc_sum / g_solo_gbs

    print("\ninterpretation inputs:")
    print(f"  GPU keeps {gpu_kept * 100:.0f}% of its isolated throughput")
    print(f"  ANE keeps {ane_kept * 100:.0f}% of its isolated throughput")
    print(f"  aggregate is {aggregate_gain:.2f}x the GPU alone")
    print(f"  nominal isolated sum {solo_sum:.1f} GB/s vs ~400 GB/s fabric prior")

    # The taxonomy from the roadmap. Thresholds are stated here rather than
    # left to the eye, but the raw rows above are what should be read.
    if conc_sum <= g_solo_gbs * 1.05:
        verdict = "1 HARD CONTENTION — the engines cannibalise each other"
    elif ane_kept >= 0.85 and gpu_kept >= 0.90:
        verdict = "4 GENUINE ADDITIVE — treat as a measurement problem first"
    elif gpu_kept >= 0.90:
        verdict = "3 STRONG COMPLEMENTARITY — GPU largely unharmed"
    else:
        verdict = "2 BUBBLE RECOVERY — aggregate rises, GPU pays some"
    print(f"\nverdict: outcome {verdict}")

    doc = {
        "experiment": "ANE-3",
        "overlap_s": overlap_s,
        "gpu": {
            "solo": g_solo, "conc": g_conc,
            "solo_gbs": g_solo_gbs, "conc_gbs": g_conc_gbs,
            "kept": gpu_kept, "latency_x": g_conc["min"] / g_solo["min"],
        },
        "ane": {
            "solo": a_solo, "conc": a_conc,
            "solo_gbs": a_solo_gbs, "conc_gbs": a_conc_gbs,
            "kept": ane_kept, "latency_x": a_conc["min"] / a_solo["min"],
        },
        "solo_sum_gbs": solo_sum,
        "conc_sum_gbs": conc_sum,
        "aggregate_vs_gpu_alone": aggregate_gain,
        "ane0b_frozen_gbs": ANE0B_FROZEN_GBS,
        "verdict": verdict,
    }
    if out_path:
        with open(out_path, "w") as fh:
            json.dump(doc, fh, indent=2)
        print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
