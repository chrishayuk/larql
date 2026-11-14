#!/usr/bin/env python3
"""LINK-2 block-frame addendum harness, revision 2 (LINK-2-block-frames-revision-2.md).

Nothing runs on import.

  --diagnostic LABEL   a development run; declared before it starts, never counted
  --qualifying         a qualification attempt; refuses to start while a
                       compiler or model process is running

Drives the frozen relay with `v3_link_pair` at the multi-row frame sizes of
block widths 1, 2, 4 and 8, at the null point and round trips of 5 and 20 ms,
and judges every point with LINK-2 revision 3's rules, unchanged. No exchange, pair or point is ever excluded.
"""
import argparse
import json
from pathlib import Path
import subprocess

import link_block_frames_model as bm
import link_model as lm
import link_pair as lp
import link_pair_model as pm
from link_sweep import ECHO_HEADER, RELAY
from profile_exact import save, sha


def measure(root, out, index, point):
    raw = out / f"point-{index:02d}.json"
    request, response = bm.geometry()[point["width"]]
    subprocess.run(
        [str(root / lp.PROBE), str(root / RELAY), str(lm.one_way_delay_us(point)), lm.UNLIMITED,
         str(request), str(response), str(pm.WARMUP), str(pm.BLOCK), str(pm.PAIRS), str(raw)],
        check=True, timeout=lp.POINT_TIMEOUT_SECONDS)
    return json.loads(raw.read_text())


def run(root, out, kind, label):
    before = lp.host_state()
    judged = []
    for index, point in enumerate(bm.frozen_points()):
        raw = measure(root, out, index, point)
        expected = bm.expected_ns(point, ECHO_HEADER.size)
        verdict = pm.judge(point, raw["pairs"], expected)
        verdict["baseline_relay"] = lp.relay_lateness(raw["baseline_relay"])
        verdict["shaped_relay"] = lp.relay_lateness(raw["shaped_relay"])
        verdict["raw"] = f"point-{index:02d}.json"
        judged.append(verdict)
        print(f"width {point['width']} {point['kind']:>5} {point['rtt_ms']:>5} ms  "
              f"expected {expected / 1e3:>9.1f} us  error {verdict['error_ns'] / 1e3:>7.1f}  "
              f"±{verdict['judged_half_width_ns'] / 1e3:>6.1f}  tol {verdict['tolerance_ns'] / 1e3:>7.1f}  "
              f"{'pass' if verdict['passes'] else 'FAIL'}", flush=True)
    receipt = {
        "schema": bm.SCHEMA, "protocol": bm.ADDENDUM, "rules": pm.REVISION,
        "kind": kind, "label": label,
        "counts_toward_qualification": kind == "qualifying",
        "frozen": pm.frozen_parameters(),
        "geometry": {str(width): list(sizes) for width, sizes in bm.geometry().items()},
        "binary_sha256": {"v3_link_shaper": sha(root / RELAY), "v3_link_pair": sha(root / lp.PROBE)},
        "harness_sha256": {name: sha(Path(__file__).with_name(name))
                           for name in ("link_block_frames.py", "link_block_frames_model.py",
                                        "link_pair_model.py", "link_model.py", bm.GEOMETRY_FILE)},
        "host_before": before, "host_after": lp.host_state(),
        "points": judged, "run_passes": bm.run_verdict(judged),
        "scope": "the cost the relay adds under this paired probe at these frame sizes and unlimited "
                 "rate; not the timing of inference endpoints",
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
    for binary in (RELAY, lp.PROBE):
        if not (root / binary).exists():
            parser.error(f"build first: {binary}")
    if not sha(root / RELAY).startswith(lp.FROZEN_RELAY_SHA256_PREFIX):
        parser.error("the relay binary is not the frozen build LINK-2 certifies")
    if args.qualifying and lp.busy_processes():
        parser.error("a compiler or model process is running; a qualifying attempt needs a quiet machine")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    receipt = run(root, out, "qualifying" if args.qualifying else "diagnostic", args.diagnostic)
    print(f"run {'passes' if receipt['run_passes'] else 'FAILS'}"
          f"{'' if args.qualifying else ' (diagnostic: does not count)'}")
    raise SystemExit(0 if receipt["run_passes"] or not args.qualifying else 1)


if __name__ == "__main__":
    main()
