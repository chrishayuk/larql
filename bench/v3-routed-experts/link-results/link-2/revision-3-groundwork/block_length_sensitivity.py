"""LINK-2 revision 3 groundwork: block-length sensitivity on the existing
diagnostic, and interval coverage on simulated correlated series. Analysis
only; reclassifies nothing."""
import json, math, random, statistics, sys
import link_pair_model as pm
raw_dir = sys.argv[1]
BLOCKS = (1, 5, 10, 20, 25, 50)
def fast_median(v):
    s = sorted(v); n = len(s); return (s[n // 2] + s[(n - 1) // 2]) / 2
def interval(values, block, resamples, rng):
    n = len(values); meds = []
    for _ in range(resamples):
        sample = []
        while len(sample) < n:
            start = rng.randrange(n)
            sample.extend(values[(start + k) % n] for k in range(block))
        meds.append(fast_median(sample[:n]))
    meds.sort()
    return meds[int(0.025 * (resamples - 1))], meds[int(math.ceil(0.975 * (resamples - 1)))]
out = {"diagnostic_half_width_us_by_block": [], "coverage": {}}
print("== existing diagnostic: 95% interval half-width (us) by bootstrap block length; tolerance/2 for comparison")
points = pm.frozen_points()
for i, point in enumerate(points):
    d = pm.pair_differences(json.load(open(f"{raw_dir}/point-{i:02d}.json"))["pairs"])
    rng = random.Random(pm.BOOTSTRAP_SEED)
    hw = {b: (lambda iv: (iv[1] - iv[0]) / 2e3)(interval(d, b, 2000, rng)) for b in BLOCKS}
    acf = {lag: pm.autocorrelation(d, lag) for lag in (1, 2, 5, 10, 20, 30)}
    exp = pm.expected_ns(point, 11576, 46136, 8)
    tol = pm.NULL_BOUND_NS if point["kind"] == "null" else max(0.05 * exp, 50_000)
    out["diagnostic_half_width_us_by_block"].append({"point": point, "half_width_us": hw, "autocorrelation": acf, "half_tolerance_us": tol / 2e3})
    print(f"{point['kind']:>5} {point['rtt_ms']:>5} {str(point['rate_bits_per_s']):>12} " + " ".join(f"L{b}={hw[b]:5.1f}" for b in BLOCKS) + f" | tol/2 {tol/2e3:6.1f} | acf " + " ".join(f"{l}:{a:+.2f}" for l, a in acf.items()))
print("\n== simulated coverage of the nominal 95% interval for the median, N=100, AR(1) series")
TRIALS, RESAMPLES = 300, 300
for phi in (0.0, 0.4, 0.76, 0.9):
    row = {}
    for block in (1, 5, 10, 20, 25):
        rng = random.Random(1000 + int(phi * 100) + block); hit = 0; widths = []
        for _ in range(TRIALS):
            x, series = 0.0, []
            for t in range(100 + 200):
                x = phi * x + rng.gauss(0, 1)
                if t >= 200: series.append(x)
            lo, hi = interval(series, block, RESAMPLES, rng)
            hit += lo <= 0.0 <= hi; widths.append((hi - lo) / 2)
        row[block] = (hit / TRIALS, statistics.fmean(widths))
    out["coverage"][phi] = {b: {"coverage": c, "mean_half_width_sd_units": w} for b, (c, w) in row.items()}
    print(f"phi={phi:4.2f} " + " ".join(f"L{b}: {c:4.0%} (hw {w:.2f})" for b, (c, w) in row.items()))
json.dump(out, open(sys.argv[2], "w"), indent=1, default=str)
