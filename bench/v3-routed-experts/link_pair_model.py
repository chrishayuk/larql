"""LINK-2 paired calibration: pure analysis of `v3_link_pair` output.

Every constant here is a frozen parameter of LINK-2 revision 3
(LINK-2-revision-3.md, which amends LINK-2.md). Changing one is a new protocol
revision, not an edit. Nothing in this module starts a process.
"""
import math
import random
import statistics

import link_model as lm

REVISION = "LINK-2 revision 3"
# Measurement.
WARMUP = 20
BLOCK = 10
PAIRS = 200
# Uncertainty: circular block bootstrap over the pair sequence.
BOOTSTRAP_BLOCK = 20
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20261114
INTERVAL = 0.95
# Serial correlation is reported at these lags and is not a pass/fail gate.
REPORTED_LAGS = range(1, 31)
# The bootstrap half-width is widened by this factor before it is judged:
# at 200 pairs and block 20 the nominal interval under-covers correlated
# series, and doubling it restored at least nominal coverage in simulation.
HALF_WIDTH_FACTOR = 2.0
# Accuracy: LINK-1's tolerance, unchanged.
TOLERANCE_RELATIVE = 0.05
TOLERANCE_ABSOLUTE_NS = 50_000
# Precision: the widened half-width may be at most this share of the tolerance.
PRECISION_SHARE = 0.5
# The null point is held to half the tolerance floor.
NULL_BOUND_NS = 25_000
# "Pairing removed common-mode variation" may be said only at or below this ratio.
PAIRING_CLAIM_RATIO = 0.5

NULL = {"kind": "null", "rtt_ms": 0.0, "rate_bits_per_s": lm.UNLIMITED}


def frozen_points():
    """LINK-1's nine shaped points, then the null point."""
    return [p for p in lm.frozen_points() if p["kind"] != "direct"] + [dict(NULL)]


def expected_ns(point, request_bytes, response_bytes, header_bytes):
    """Configured cost of one exchange: round trip plus frame bits over the rate."""
    shape = {"crossings": 1,
             "critical_bits": (header_bytes + request_bytes + response_bytes) * lm.BITS_PER_BYTE}
    return sum(lm.regressors(point, shape))


def pair_differences(pairs):
    """Median shaped exchange minus median baseline exchange, per pair, in order."""
    if not pairs:
        raise ValueError("no pairs measured")
    differences = []
    for pair in pairs:
        if len(pair["baseline_ns"]) != 2 * BLOCK or len(pair["shaped_ns"]) != 2 * BLOCK:
            raise ValueError("a pair does not hold two blocks of each path")
        differences.append(statistics.median(pair["shaped_ns"]) - statistics.median(pair["baseline_ns"]))
    return differences


def autocorrelation(values, lag):
    """Sample autocorrelation at `lag`; zero for a constant series."""
    mean = statistics.fmean(values)
    spread = sum((v - mean) ** 2 for v in values)
    if spread == 0:
        return 0.0
    return sum((values[i] - mean) * (values[i + lag] - mean) for i in range(len(values) - lag)) / spread


def percentile_interval(samples, level):
    ordered = sorted(samples)
    tail = (1 - level) / 2
    low = ordered[int(math.floor(tail * (len(ordered) - 1)))]
    high = ordered[int(math.ceil((1 - tail) * (len(ordered) - 1)))]
    return low, high


def block_bootstrap_interval(values, block, resamples, seed, level=INTERVAL):
    """Interval of the median under a circular block bootstrap of the sequence.

    Blocks of consecutive values (wrapping at the end) are drawn until the
    resample is as long as the original, so correlation within a block is
    kept. With `block` = 1 this is the ordinary independent bootstrap.
    """
    if block < 1 or block > len(values):
        raise ValueError("block length must be between 1 and the series length")
    generator = random.Random(seed)
    count = len(values)
    medians = []
    for _ in range(resamples):
        sample = []
        while len(sample) < count:
            start = generator.randrange(count)
            sample.extend(values[(start + offset) % count] for offset in range(block))
        medians.append(statistics.median(sample[:count]))
    return percentile_interval(medians, level)


def repaired_spread(pairs, seed):
    """Spread of the differences when shaped and baseline halves are paired at random."""
    generator = random.Random(seed)
    shaped = [statistics.median(pair["shaped_ns"]) for pair in pairs]
    baseline = [statistics.median(pair["baseline_ns"]) for pair in pairs]
    generator.shuffle(baseline)
    return statistics.pstdev(s - b for s, b in zip(shaped, baseline))


def judge(point, pairs, expected):
    """Apply LINK-2's decision rules to one point. Nothing is excluded."""
    if len(pairs) != PAIRS:
        raise ValueError(f"a point needs exactly {PAIRS} pairs, got {len(pairs)}")
    differences = pair_differences(pairs)
    estimate = statistics.median(differences)
    low, high = block_bootstrap_interval(differences, BOOTSTRAP_BLOCK, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED)
    independent = block_bootstrap_interval(differences, 1, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED)
    half_width = (high - low) / 2
    lags = {lag: autocorrelation(differences, lag) for lag in REPORTED_LAGS}
    judged_half_width = HALF_WIDTH_FACTOR * half_width
    null = point["kind"] == "null"
    tolerance = NULL_BOUND_NS if null else max(TOLERANCE_RELATIVE * expected, TOLERANCE_ABSOLUTE_NS)
    accurate = abs(estimate - expected) <= tolerance
    precise = judged_half_width <= PRECISION_SHARE * tolerance
    paired = statistics.pstdev(differences)
    repaired = repaired_spread(pairs, BOOTSTRAP_SEED)
    return {
        "point": point, "expected_ns": expected, "estimate_ns": estimate,
        "error_ns": estimate - expected, "tolerance_ns": tolerance,
        "interval_ns": [low, high], "interval_half_width_ns": half_width,
        "judged_half_width_ns": judged_half_width,
        "independent_bootstrap_interval_ns": list(independent),
        "autocorrelation": {str(lag): value for lag, value in lags.items()},
        "autocorrelation_note": "reported only; not a pass/fail gate, and no value of it "
                                "establishes that the pairs are independent",
        "paired_spread_ns": paired, "repaired_spread_ns": repaired,
        "pairing_removed_common_mode": repaired > 0 and paired <= PAIRING_CLAIM_RATIO * repaired,
        "accurate": accurate, "precise": precise, "passes": accurate and precise,
    }


def run_verdict(judged):
    """A run passes only if every frozen point, including the null point, passes."""
    wanted = [(p["kind"], p["rtt_ms"], p["rate_bits_per_s"]) for p in frozen_points()]
    got = [(j["point"]["kind"], j["point"]["rtt_ms"], j["point"]["rate_bits_per_s"]) for j in judged]
    if got != wanted:
        raise ValueError("a run must judge exactly the frozen points, in order")
    return all(j["passes"] for j in judged)


SCHEMA = "larql.v3.link-pair-calibration.v1"


def frozen_parameters():
    """The parameters a receipt must have been measured and judged under."""
    return {"warmup": WARMUP, "block": BLOCK, "pairs": PAIRS, "bootstrap_block": BOOTSTRAP_BLOCK,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES, "bootstrap_seed": BOOTSTRAP_SEED,
            "half_width_factor": HALF_WIDTH_FACTOR, "tolerance_relative": TOLERANCE_RELATIVE,
            "tolerance_absolute_ns": TOLERANCE_ABSOLUTE_NS, "precision_share": PRECISION_SHARE,
            "null_bound_ns": NULL_BOUND_NS}


def check_qualifying_receipt(receipt, relay_sha256):
    """Refuse anything but a passing qualifying run of this revision on this relay build.

    A diagnostic, a run under another revision or other parameters, a run on
    another relay binary, and a run that did not pass every frozen point are
    each refused by name. Nothing is inferred from a receipt's file name.
    """
    if receipt.get("schema") != SCHEMA or receipt.get("protocol") != REVISION:
        raise ValueError(f"not a {REVISION} calibration receipt")
    if receipt.get("kind") != "qualifying" or receipt.get("counts_toward_qualification") is not True:
        raise ValueError("a diagnostic run never counts toward qualification")
    frozen = receipt.get("frozen", {})
    for name, value in frozen_parameters().items():
        if frozen.get(name) != value:
            raise ValueError(f"receipt was judged with {name}={frozen.get(name)!r}, not the frozen {value!r}")
    if receipt.get("binary_sha256", {}).get("v3_link_shaper") != relay_sha256:
        raise ValueError("calibration was recorded for a different relay build")
    if receipt.get("run_passes") is not True or not run_verdict(receipt.get("points", [])):
        raise ValueError("the calibration run did not pass every frozen point")
