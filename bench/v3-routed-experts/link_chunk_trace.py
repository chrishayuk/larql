#!/usr/bin/env python3
"""Where an exchange's time goes inside the relay (diagnostic; qualifies nothing).

Reads one `v3_link_pair` point measured through `v3_link_shaper_trace`, the
tracing copy of the relay, and splits every exchange on each relay's own
clock into the stages below, for the baseline relay and the shaped relay:

  request in     first request chunk read  -> last request chunk read
  request hold   last request chunk read   -> last request chunk written
  turnaround     last request chunk written -> first response chunk read
  response in    first response chunk read -> last response chunk read
  response hold  last response chunk read  -> last response chunk written

The five stages sum to the relay's view of the exchange. A stage is a
"hold" because it is where the configured one-way delay lives: with no
delay it is only the relay's own forwarding time. The probe's exchange time
less the relay's view is what happened outside the relay, at the endpoints.

Usage: link_chunk_trace.py <point.json> <request_bytes> <response_bytes>
"""
import gzip
import json
from pathlib import Path
import statistics
import sys

from link_sweep import ECHO_HEADER

STAGES = ("request_in", "request_hold", "turnaround", "response_in", "response_hold")


def exchanges(chunks, direction, frame_bytes):
    """Group one direction's chunks into exchanges by the bytes each carries."""
    groups, current, carried = [], [], 0
    for chunk in (c for c in chunks if c["direction"] == direction):
        current.append(chunk)
        carried += chunk["bytes"]
        if carried > frame_bytes:
            raise ValueError(f"a {direction} chunk spans two exchanges; frames cannot be told apart")
        if carried == frame_bytes:
            groups.append(current)
            current, carried = [], 0
    if current:
        raise ValueError(f"{direction} trace ends inside a frame")
    return groups


def read_trace(path):
    """Lines of a trace, stored plain or gzipped."""
    plain, packed = Path(path), Path(str(path) + ".gz")
    if plain.exists():
        return plain.read_text().splitlines()
    with gzip.open(packed, "rt") as handle:
        return handle.read().splitlines()


def stages(trace_path, request_bytes, response_bytes, warmup):
    chunks = [json.loads(line) for line in read_trace(trace_path)]
    requests = exchanges(chunks, "request", ECHO_HEADER.size + request_bytes)
    responses = exchanges(chunks, "response", response_bytes)
    if len(requests) != len(responses):
        raise ValueError("request and response exchange counts differ")
    rows = []
    for request, response in list(zip(requests, responses))[warmup:]:
        rows.append({
            "request_in": request[-1]["arrival_ns"] - request[0]["arrival_ns"],
            "request_hold": request[-1]["done_ns"] - request[-1]["arrival_ns"],
            "turnaround": response[0]["arrival_ns"] - request[-1]["done_ns"],
            "response_in": response[-1]["arrival_ns"] - response[0]["arrival_ns"],
            "response_hold": response[-1]["done_ns"] - response[-1]["arrival_ns"],
            "relay_total": response[-1]["done_ns"] - request[0]["arrival_ns"],
            "request_chunks": len(request), "response_chunks": len(response),
            # How late the last chunk of each direction left, against its schedule.
            "request_last_late": request[-1]["start_ns"] - request[-1]["release_ns"],
            "response_last_late": response[-1]["start_ns"] - response[-1]["release_ns"],
            "response_first_hold": response[0]["done_ns"] - response[0]["arrival_ns"],
        })
    return rows


def median(rows, key):
    return statistics.median(row[key] for row in rows)


def analyse(point_path, request_bytes, response_bytes):
    point = json.loads(Path(point_path).read_text())
    delay_ns = int(point["one_way_delay_us"]) * 1000
    probe = {name: statistics.median(ns for pair in point["pairs"] for ns in pair[f"{name}_ns"])
             for name in ("baseline", "shaped")}
    relay = {name: stages(f"{point_path}.{name}-relay.jsonl.trace.jsonl", request_bytes, response_bytes,
                          point["warmup"]) for name in ("baseline", "shaped")}
    report = {"one_way_delay_ns": delay_ns, "request_bytes": request_bytes, "response_bytes": response_bytes,
              "exchanges": {name: len(rows) for name, rows in relay.items()}, "median_ns": {}}
    keys = STAGES + ("relay_total", "request_chunks", "response_chunks", "request_last_late",
                     "response_last_late", "response_first_hold")
    for key in keys:
        base, shaped = median(relay["baseline"], key), median(relay["shaped"], key)
        configured = delay_ns if key.endswith("_hold") else 2 * delay_ns if key == "relay_total" else 0
        report["median_ns"][key] = {"baseline": base, "shaped": shaped,
                                    "shaped_minus_baseline_minus_configured": shaped - base - configured}
    base_out = probe["baseline"] - median(relay["baseline"], "relay_total")
    shaped_out = probe["shaped"] - median(relay["shaped"], "relay_total")
    report["probe_median_ns"] = probe
    report["probe_error_ns"] = probe["shaped"] - probe["baseline"] - 2 * delay_ns
    report["outside_relay_ns"] = {"baseline": base_out, "shaped": shaped_out, "shaped_minus_baseline": shaped_out - base_out}
    return report


def main():
    point, request_bytes, response_bytes = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    report = analyse(point, request_bytes, response_bytes)
    print(f"one-way delay {report['one_way_delay_ns'] / 1e3:.0f} us; "
          f"frames {request_bytes} / {response_bytes} bytes; exchanges {report['exchanges']}")
    print(f"{'stage (median, us)':<22}{'baseline':>10}{'shaped':>10}{'excess':>10}")
    for key, row in report["median_ns"].items():
        scale = 1 if key.endswith("chunks") else 1e3
        print(f"{key:<22}{row['baseline'] / scale:>10.1f}{row['shaped'] / scale:>10.1f}"
              f"{row['shaped_minus_baseline_minus_configured'] / scale:>10.1f}")
    outside = report["outside_relay_ns"]
    print(f"{'outside the relay':<22}{outside['baseline'] / 1e3:>10.1f}{outside['shaped'] / 1e3:>10.1f}"
          f"{outside['shaped_minus_baseline'] / 1e3:>10.1f}")
    print(f"probe error against configured round trip: {report['probe_error_ns'] / 1e3:+.1f} us")
    Path(point + ".chunk-analysis.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
