#!/usr/bin/env python3
"""Collect real CLI local/remote/local diagnostics with two persistent FFN workers.

This driver does not establish peer exclusivity or claim a performance gate.
It preserves raw records, excludes prompt/early decode positions from summaries,
and reports bracket drift without pooling invalid brackets.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import time
import urllib.request


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def summarize(rows):
    keys = ("attention_ns", "ffn_ns", "reentry_ns", "other_ns", "total_ns")
    result = {key: statistics.mean(row[key] for row in rows) for key in keys}
    result["positions"] = len(rows)
    result["tokens_per_second"] = 1e9 / result["total_ns"]
    result["median_total_ns"] = statistics.median(row["total_ns"] for row in rows)
    calls = [row["provider_calls"] for row in rows]
    for key in ("request_bytes", "response_bytes", "encode_ns", "decode_ns", "roundtrip_ns", "transport_remainder_ns"):
        values = [sum(c[key] for c in cs) for cs in calls if all(c.get(key) is not None for c in cs)]
        result[key] = statistics.mean(values) if len(values) == len(rows) else None
    for key in ("decode_ns", "queue_ns", "execute_ns", "ffn_ns", "encode_ns", "handler_ns"):
        values = [sum(c["worker"][key] for c in cs) for cs in calls if all("worker" in c for c in cs)]
        result["worker_" + key] = statistics.mean(values) if len(values) == len(rows) else None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--port", type=int, default=19181)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--skip-decode", type=int, default=8)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--wire", choices=("json", "binary"), default="json")
    parser.add_argument("--wire-comparison", action="store_true", help="Nest JSON / binary / JSON within local controls")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    model = args.model.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    settings = {"LARQL_CPU_WORKERS": "8", "RAYON_NUM_THREADS": "8", "VECLIB_MAXIMUM_THREADS": "1", "TOKIO_WORKER_THREADS": "2", "LARQL_KV_ENGINE": ""}
    env.update(settings)
    index = json.loads((model / "index.json").read_text())
    layers = index["num_layers"]
    prompt = "Explain step by step why the sky is blue, including how sunlight interacts with the atmosphere."
    cli, server = (root / "target/release" / name for name in ("larql", "larql-server"))
    manifest = {
        "schema": "larql.v3.dense-loopback-diagnostic.v1",
        "benchmark_gate": False,
        "peer_exclusivity": "unconfirmed; no promoted performance claim",
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "model": str(model), "index": index,
        "metadata_sha256": {name: hashlib.sha256((model / name).read_bytes()).hexdigest() for name in ("index.json", "system_graph.json")},
        "binary_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (cli, server)},
        "settings": settings, "prompt": prompt, "max_tokens": args.max_tokens,
        "skip_decode": args.skip_decode, "wire": args.wire, "wire_comparison": args.wire_comparison,
        "clock": "per-position profile; load/tokenization/output excluded",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(out / "manifest.json", manifest)
    processes, logs, bindings = [], [], []
    trials = []
    reference = None
    try:
        urls = []
        for worker, (start, end) in enumerate(((0, layers // 2), (layers // 2, layers))):
            port = args.port + worker
            url = f"http://127.0.0.1:{port}"
            urls.append(url)
            log = (out / f"worker-{worker}.log").open("w")
            logs.append(log)
            command = [str(server), str(model), "--host", "127.0.0.1", "--port", str(port), "--ffn-only", "--layers", f"{start}-{end - 1}"]
            process = subprocess.Popen(command, env=env, cwd=root, stdout=log, stderr=subprocess.STDOUT)
            processes.append(process)
            deadline = time.monotonic() + 180
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"worker exited: see {log.name}")
                try:
                    with urllib.request.urlopen(url + "/v1/vindex3/ffn", timeout=2) as response:
                        binding = json.load(response)
                    if binding["program"]["start"] != start or binding["program"]["end"] != end:
                        raise RuntimeError("worker binding range mismatch")
                    bindings.append(binding)
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"worker not ready: {url}")
                    time.sleep(0.25)
            print(f"ready: worker {worker}, layers {start}..{end - 1}", flush=True)
        write_json(out / "bindings.json", bindings)

        def run(label, remote, wire=None):
            nonlocal reference
            profile = out / (label + ".jsonl")
            command = [str(cli), "run", str(model), prompt, "--max-tokens", str(args.max_tokens), "--emit-ids", "--v3-profile", str(profile)]
            if remote:
                command.extend(["--v3-ffn-shards", ",".join(urls), "--v3-ffn-wire", wire or args.wire])
            print(f"running {label}", flush=True)
            with (out / (label + ".stdout")).open("w") as stdout, (out / (label + ".stderr")).open("w") as stderr:
                subprocess.run(command, env=env, cwd=root, stdout=stdout, stderr=stderr, check=True, timeout=600)
            stderr = (out / (label + ".stderr")).read_text()
            prompt_ids = json.loads(re.search(r"prompt ids: (\[.*\])", stderr)[1])
            generated_ids = json.loads(re.search(r"generated ids: (\[.*\])", stderr)[1])
            rows = [json.loads(line) for line in profile.read_text().splitlines()]
            if not rows[0]["complete"] or any(not row["complete"] for row in rows[1:]):
                raise RuntimeError("incomplete run")
            identity = (prompt_ids, generated_ids, [row["token"] for row in rows[1:]])
            if reference is None:
                reference = identity
            elif identity != reference:
                raise RuntimeError(f"token parity failed: {label}")
            measured = rows[1 + len(prompt_ids) + args.skip_decode:]
            if len(measured) < 8:
                raise RuntimeError(f"too few continuation positions: {len(measured)}")
            result = {"label": label, "remote": remote, "token_parity": True, "command": command, "summary": summarize(measured)}
            trials.append(result)
            write_json(out / "trials.json", trials)
            print(f"{label}: {result['summary']['total_ns'] / 1e6:.3f} ms/position; {len(measured)} measured positions; IDs match", flush=True)
            return result["summary"]

        for warm in range(2):
            run(f"warm-{warm}-local", False)
            run(f"warm-{warm}-remote", True, "json" if args.wire_comparison else args.wire)
            if args.wire_comparison:
                run(f"warm-{warm}-binary", True, "binary")
        brackets = []
        for block in range(args.blocks):
            before = run(f"block-{block}-local-before", False)
            remote = run(f"block-{block}-remote", True, "json" if args.wire_comparison else args.wire)
            binary = run(f"block-{block}-binary", True, "binary") if args.wire_comparison else None
            json_after = run(f"block-{block}-json-after", True, "json") if args.wire_comparison else None
            after = run(f"block-{block}-local-after", False)
            midpoint = (before["total_ns"] + after["total_ns"]) / 2
            drift = abs(before["total_ns"] - after["total_ns"]) / midpoint
            brackets.append({"block": block, "control_drift": drift, "controls_agree_within_one_percent": drift <= 0.01,
                             "diagnostic_only": True, "local_before": before, "remote": remote, "local_after": after,
                             "remote_over_local": remote["total_ns"] / midpoint if drift <= 0.01 else None})
            if args.wire_comparison:
                json_midpoint = (remote["total_ns"] + json_after["total_ns"]) / 2
                json_drift = abs(remote["total_ns"] - json_after["total_ns"]) / json_midpoint
                brackets[-1].update({"binary": binary, "json_after": json_after, "json_control_drift": json_drift,
                    "binary_over_json": binary["total_ns"] / json_midpoint if json_drift <= .01 else None,
                    "binary_over_local": binary["total_ns"] / midpoint if drift <= .01 else None})
                print(f"block {block}: JSON control drift {json_drift:.2%}", flush=True)
            write_json(out / "brackets.json", brackets)
            print(f"block {block}: control drift {drift:.2%}; diagnostic only", flush=True)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for log in logs:
            log.close()


if __name__ == "__main__":
    main()
