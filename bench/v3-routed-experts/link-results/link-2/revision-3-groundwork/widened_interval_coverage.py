"""Coverage of the block-bootstrap interval for the median when its half-width
is widened by a fixed factor, for 100 and 200 pairs. Simulation only."""
import json, math, random, sys
def med(v):
    s = sorted(v); n = len(s); return (s[n // 2] + s[(n - 1) // 2]) / 2
def interval(values, block, resamples, rng):
    n = len(values); meds = []
    for _ in range(resamples):
        sample = []
        while len(sample) < n:
            start = rng.randrange(n); sample.extend(values[(start + k) % n] for k in range(block))
        meds.append(med(sample[:n]))
    meds.sort(); return meds[int(0.025 * (resamples - 1))], meds[int(math.ceil(0.975 * (resamples - 1)))]
TRIALS, RESAMPLES = 400, 300
out = {}
for n in (100, 200):
    for block in (10, 20):
        for phi in (0.0, 0.4, 0.76, 0.9):
            rng = random.Random(7000 + n + block + int(phi * 100)); hits = {1.0: 0, 1.5: 0, 2.0: 0}
            for _ in range(TRIALS):
                x, series = 0.0, []
                for t in range(n + 200):
                    x = phi * x + rng.gauss(0, 1)
                    if t >= 200: series.append(x)
                lo, hi = interval(series, block, RESAMPLES, rng); centre, half = (lo + hi) / 2, (hi - lo) / 2
                for factor in hits: hits[factor] += abs(centre) <= factor * half
            out[f"n{n}_L{block}_phi{phi}"] = {str(f): h / TRIALS for f, h in hits.items()}
            print(f"N={n} L={block:2d} phi={phi:4.2f}  " + "  ".join(f"x{f}: {h / TRIALS:4.0%}" for f, h in hits.items()), flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
