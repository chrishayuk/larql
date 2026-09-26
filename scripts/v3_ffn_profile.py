#!/usr/bin/env python3
"""Print per-position decomposition from --v3-profile (or the HTTP smoke fixture).

This is a diagnostic reader, not a benchmark gate. Nested RPC/worker intervals
must not be added to the top-level attention/FFN/re-entry/other partition.
"""
import argparse
import csv
import json
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace")
    parser.add_argument("--smoke-arm", choices=("local", "remote"), default="remote")
    parser.add_argument("--skip", type=int, default=0, help="Omit this many input positions")
    args = parser.parse_args()
    with open(args.trace) as source:
        metadata = json.loads(next(source))
        rows = [json.loads(line) for line in source if line.strip()]
    smoke = metadata.get("schema") == "larql.v3.ffn-smoke.v1"
    if smoke:
        print("Synthetic debug fixture: these are diagnostics, not performance evidence.", file=sys.stderr)
        rows = [row[args.smoke_arm] for row in rows]
    elif metadata.get("schema") != "larql.v3.position-profile.v1":
        parser.error("unknown trace schema")
    elif not metadata.get("complete"):
        parser.error("run failed; inspect incomplete trace directly")
    writer = csv.writer(sys.stdout, delimiter="\t", lineterminator="\n")
    writer.writerow(["position", "token", "attention_ms", "ffn_provider_ms", "reentry_ms", "other_ms", "total_ms",
                     "request_bytes", "encode_us", "roundtrip_ms", "remote_ffn_ms", "response_bytes", "decode_us",
                     "worker_encode_us", "transport_remainder_ms"])
    for row in rows[args.skip:]:
        if not row["complete"]:
            parser.error("incomplete position; inspect trace directly")
        calls = row["provider_calls"]
        if any(not call["complete"] for call in calls):
            parser.error("incomplete RPC; inspect trace directly")

        def total(key, divisor=1):
            return sum(call[key] for call in calls) / divisor

        def worker_total(key, divisor):
            if any("worker" not in call for call in calls):
                return "unknown"
            return sum(call["worker"][key] for call in calls) / divisor

        remainder = (total("transport_remainder_ns", 1e6)
                     if all(call.get("transport_remainder_ns") is not None for call in calls) else "unknown")
        writer.writerow([row["position"], row["token"], *[row[key] / 1e6 for key in
                         ("attention_ns", "ffn_ns", "reentry_ns", "other_ns", "total_ns")],
                         int(total("request_bytes")), total("encode_ns", 1e3), total("roundtrip_ns", 1e6),
                         worker_total("ffn_ns", 1e6), int(total("response_bytes")), total("decode_ns", 1e3),
                         worker_total("encode_ns", 1e3), remainder])


if __name__ == "__main__":
    main()
