"""LINK-2 block-frame addendum: points, geometry and verdicts (LINK-2-block-frames.md).

Pure and frozen with the addendum. Every measurement parameter and decision
rule is LINK-2 revision 3's and is taken from `link_pair_model`, never
restated here. Nothing in this module starts a process.
"""
import json
from pathlib import Path

import link_model as lm
import link_pair_model as pm

ADDENDUM = "LINK-2 block-frame addendum 1"
SCHEMA = "larql.v3.link-block-frames.v1"
GEOMETRY_FILE = "block_frames.json"
# Block widths of BLOCK-TIME-1, in the order they are measured.
WIDTHS = (1, 2, 4, 8)
# Round trips of BLOCK-TIME-1 other than zero, which is the null point.
RTT_MS = (1.0, 5.0, 20.0)
PASSING_RUNS_REQUIRED = 2


def geometry():
    """Request and response bytes per width, as pinned against the wire encoder."""
    recorded = json.loads(Path(__file__).with_name(GEOMETRY_FILE).read_text())["widths"]
    return {width: (recorded[str(width)]["request_bytes"], recorded[str(width)]["response_bytes"])
            for width in WIDTHS}


def frozen_points():
    """For each width: the null point, then each round trip, all at unlimited rate."""
    points = []
    for width in WIDTHS:
        points.append(dict(pm.NULL, width=width))
        points += [{"kind": "axis", "rtt_ms": rtt, "rate_bits_per_s": lm.UNLIMITED, "width": width}
                   for rtt in RTT_MS]
    return points


def identity(point):
    return (point["width"], point["kind"], point["rtt_ms"], point["rate_bits_per_s"])


def expected_ns(point, header_bytes):
    request, response = geometry()[point["width"]]
    return pm.expected_ns(point, request, response, header_bytes)


def run_verdict(judged):
    """A run passes only if every frozen point passes, judged in order."""
    if [identity(j["point"]) for j in judged] != [identity(p) for p in frozen_points()]:
        raise ValueError("a run must judge exactly the addendum's points, in order")
    return all(j["passes"] for j in judged)


def check_qualifying_receipt(receipt, relay_sha256):
    """Refuse anything but a passing qualifying run of this addendum on this relay build."""
    if receipt.get("schema") != SCHEMA or receipt.get("protocol") != ADDENDUM:
        raise ValueError(f"not a {ADDENDUM} receipt")
    if receipt.get("rules") != pm.REVISION:
        raise ValueError(f"receipt was not judged under {pm.REVISION}")
    if receipt.get("kind") != "qualifying" or receipt.get("counts_toward_qualification") is not True:
        raise ValueError("a diagnostic run never counts toward qualification")
    frozen = receipt.get("frozen", {})
    for name, value in pm.frozen_parameters().items():
        if frozen.get(name) != value:
            raise ValueError(f"receipt was judged with {name}={frozen.get(name)!r}, not the frozen {value!r}")
    if receipt.get("geometry") != {str(w): list(sizes) for w, sizes in geometry().items()}:
        raise ValueError("receipt was measured at other frame sizes")
    if receipt.get("binary_sha256", {}).get("v3_link_shaper") != relay_sha256:
        raise ValueError("qualification was recorded for a different relay build")
    if receipt.get("run_passes") is not True or not run_verdict(receipt.get("points", [])):
        raise ValueError("the run did not pass every point")


def check_qualification(receipts, relay_sha256):
    """Two distinct passing qualifying receipts, each checked in full."""
    for receipt in receipts:
        check_qualifying_receipt(receipt, relay_sha256)
    started = {receipt.get("host_before", {}).get("utc") for receipt in receipts}
    if len(receipts) < PASSING_RUNS_REQUIRED or len(started) < PASSING_RUNS_REQUIRED:
        raise ValueError(f"qualification needs {PASSING_RUNS_REQUIRED} distinct passing runs")
