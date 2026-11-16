#!/usr/bin/env python3
"""LINK-2 paired calibration harness (LINK-2-revision-3.md). Nothing runs on import.

  --diagnostic LABEL   a development run; declared as such before it starts
                       and never counted toward qualification
  --qualifying         a qualification attempt; refuses to start while a
                       compiler or model process is running

Either mode measures the frozen points in order with `v3_link_pair`, applies
the pre-registered rules in `link_pair_model.py` and writes a receipt. No
exchange, pair or point is ever excluded.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import link_model as lm
import link_pair_model as pm
from link_sweep import RELAY, REQUEST_BYTES, RESPONSE_BYTES, ECHO_HEADER
from profile_exact import save, sha

PROBE = "target/release/examples/v3_link_pair"
# The relay build LINK-2 certifies. A different binary is a different
# instrument, certified only by its own receipts (LINK-2 revision 3 and the
# block-frame addendum, two passing qualifying runs each, recorded for its
# hash). Every admissible run requires this build.
#   7b1097f814216208: certified 2026-11-14/15 (link-results/link-2,
#     link-results/link-2-block-frames); used by BLOCK-TIME-1 and -2 and
#     VERIFY-REMOTE-1 diagnostic 01.
FROZEN_RELAY_SHA256_PREFIX = "7b1097f814216208"
# A build under requalification: the qualification harnesses may run it so
# its receipts can bind to it, and VERIFY-REMOTE-1 diagnostics may run it.
# Nothing admissible accepts it. If its receipts pass, a reviewed change
# makes it the certified build above.
#   b4068220a544782c: VERIFY-REMOTE-1 revision 2's relay with the
#     generation-boundary mark (#865), named 2026-11-16.
CANDIDATE_RELAY_SHA256_PREFIX = "b4068220a544782c"


def relay_build(relay_sha256):
    """`certified`, `candidate`, or None for any other build."""
    if relay_sha256.startswith(FROZEN_RELAY_SHA256_PREFIX):
        return "certified"
    if relay_sha256.startswith(CANDIDATE_RELAY_SHA256_PREFIX):
        return "candidate"
    return None


LINK2_RUNS_REQUIRED = 2


def check_link2_qualification(receipts, relay_sha256):
    """Two distinct passing LINK-2 revision 3 qualifying receipts for this relay build."""
    for receipt in receipts:
        pm.check_qualifying_receipt(receipt, relay_sha256)
    started = {receipt.get("host_before", {}).get("utc") for receipt in receipts}
    if len(receipts) < LINK2_RUNS_REQUIRED or len(started) < LINK2_RUNS_REQUIRED:
        raise ValueError(f"the relay needs {LINK2_RUNS_REQUIRED} distinct passing LINK-2 qualifying runs")
# Processes whose presence makes a qualifying attempt meaningless.
BUSY_EXECUTABLES = ("rustc", "cargo", "larql", "larql-server")
POINT_TIMEOUT_SECONDS = 900


def busy_processes():
    """Compiler or model executables currently running, by executable name.

    Matched on the executable, not the command line: a shell whose arguments
    merely mention one of these names is not a competing process.
    """
    listing = subprocess.check_output(["ps", "-Ao", "comm"], text=True).splitlines()[1:]
    return sorted({Path(line.strip()).name for line in listing} & set(BUSY_EXECUTABLES))


def host_state():
    return {"load_average": os.getloadavg(), "busy_processes": busy_processes(),
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def measure(root, out, index, point):
    raw = out / f"point-{index:02d}.json"
    rate = point["rate_bits_per_s"]
    subprocess.run(
        [str(root / PROBE), str(root / RELAY), str(lm.one_way_delay_us(point)),
         lm.UNLIMITED if rate == lm.UNLIMITED else repr(float(rate)),
         str(REQUEST_BYTES), str(RESPONSE_BYTES), str(pm.WARMUP), str(pm.BLOCK), str(pm.PAIRS), str(raw)],
        check=True, timeout=POINT_TIMEOUT_SECONDS)
    return json.loads(raw.read_text())


def relay_lateness(records):
    chunks = sum(r["chunks"] for r in records)
    return {"late_mean_ns": sum(r["late_sum_ns"] for r in records) / chunks,
            "late_max_ns": max(r["late_max_ns"] for r in records), "chunks": chunks}


def run(root, out, kind, label):
    before = host_state()
    judged = []
    for index, point in enumerate(pm.frozen_points()):
        raw = measure(root, out, index, point)
        expected = pm.expected_ns(point, REQUEST_BYTES, RESPONSE_BYTES, ECHO_HEADER.size)
        verdict = pm.judge(point, raw["pairs"], expected)
        verdict["baseline_relay"] = relay_lateness(raw["baseline_relay"])
        verdict["shaped_relay"] = relay_lateness(raw["shaped_relay"])
        verdict["raw"] = f"point-{index:02d}.json"
        judged.append(verdict)
        print(f"{point['kind']:>5} {point['rtt_ms']:>5} ms {str(point['rate_bits_per_s']):>12} b/s  "
              f"expected {expected / 1e3:>9.1f} us  error {verdict['error_ns'] / 1e3:>7.1f}  "
              f"±{verdict['judged_half_width_ns'] / 1e3:>6.1f}  tol {verdict['tolerance_ns'] / 1e3:>7.1f}  "
              f"{'pass' if verdict['passes'] else 'FAIL'}", flush=True)
    receipt = {
        "schema": pm.SCHEMA, "protocol": pm.REVISION,
        "kind": kind, "label": label,
        "counts_toward_qualification": kind == "qualifying",
        "frozen": dict(pm.frozen_parameters(), autocorrelation="reported at lags 1 to 30; not gated"),
        "binary_sha256": {"v3_link_shaper": sha(root / RELAY), "v3_link_pair": sha(root / PROBE)},
        "harness_sha256": {name: sha(Path(__file__).with_name(name))
                           for name in ("link_pair.py", "link_pair_model.py", "link_model.py")},
        "host_before": before, "host_after": host_state(),
        "points": judged, "run_passes": pm.run_verdict(judged),
        "scope": "certifies the cost the relay adds under this paired probe; not the timing of inference endpoints",
    }
    save(out / "calibration.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--diagnostic", metavar="LABEL", help="Development run; never counts")
    mode.add_argument("--qualifying", action="store_true", help="Qualification attempt")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    for binary in (RELAY, PROBE):
        if not (root / binary).exists():
            parser.error(f"build first: {binary}")
    if relay_build(sha(root / RELAY)) is None:
        parser.error("the relay binary is neither the certified build nor the named candidate")
    if args.qualifying and busy_processes():
        parser.error("a compiler or model process is running; a qualifying attempt needs a quiet machine")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    receipt = run(root, out, "qualifying" if args.qualifying else "diagnostic", args.diagnostic)
    print(f"run {'passes' if receipt['run_passes'] else 'FAILS'}"
          f"{'' if args.qualifying else ' (diagnostic: does not count)'}")
    raise SystemExit(0 if receipt["run_passes"] or not args.qualifying else 1)


if __name__ == "__main__":
    main()
