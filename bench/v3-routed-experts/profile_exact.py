#!/usr/bin/env python3
"""Exact routed profiling. No execution without an explicit --execute flag.

Controls are local / candidate / local, warmed separately. Managed loopback
workers stop before local controls. External workers are never started/stopped.
This module can be imported by analysis tests without starting any processes.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import time
import urllib.parse
import urllib.request


LAYOUTS = {
    "local": [], "one": [(0, 23, 0, 31)],
    "experts": [(0, 23, 0, 15), (0, 23, 16, 31)],
    "mixed": [(0, 11, 0, 31), (12, 23, 0, 7), (12, 23, 8, 31)],
}


def external_urls(values, topology):
    """Validate receipt-safe endpoints without contacting them."""
    if len(values) != len(LAYOUTS[topology]):
        raise ValueError("worker URL count must match the selected topology")
    urls = []
    for value in values:
        parsed = urllib.parse.urlsplit(value)
        if (parsed.scheme not in ("http", "https") or not parsed.hostname
                or parsed.username is not None or parsed.password is not None
                or parsed.query or parsed.fragment or "," in value
                or any(c.isspace() for c in value)):
            raise ValueError("worker URLs must be HTTP(S), without credentials, queries or fragments")
        parsed.port  # Reject malformed/out-of-range ports before starting any process.
        urls.append(value.rstrip("/"))
    if len(set(urls)) != len(urls):
        raise ValueError("worker URLs must be distinct")
    return urls


def read_external_bindings(urls, topology):
    """Check experimental layout; the Rust client remains execution authority."""
    bindings, identity = [], None
    for url, (start, end, first, last) in zip(urls, LAYOUTS[topology]):
        with urllib.request.urlopen(url + "/v1/vindex3/experts", timeout=10) as response:
            binding = json.load(response)
        program = binding["program"]
        if (program["start"], program["end"], binding["expert_start"], binding["expert_end"]) != (
                start, end + 1, first, last + 1):
            raise ValueError(f"worker layout differs from requested topology: {url}")
        if program["layers"] != 24 or program["hidden"] != 2880 or program["backend"] != "cpu":
            raise ValueError(f"worker is not the GPT-OSS CPU profile shape: {url}")
        current = {k: program[k] for k in ("schema", "artifact", "backend", "lowering", "layers", "hidden")}
        if identity is not None and current != identity:
            raise ValueError("external workers advertise different program identities")
        identity = current
        bindings.append(binding)
    return bindings


def agreement(a, b):
    return abs(a - b) / ((a + b) / 2) if a + b > 0 else float("inf")


def summarize(rows, remote, hidden, layers, top_k):
    """Sum sequential layers; preserve the distinction between work and wall time."""
    if not rows:
        raise ValueError("no measured decode positions")
    totals = []
    for row in rows:
        assert row["complete"], "incomplete position"
        calls = row["provider_calls"]
        operations = [c for c in calls if c.get("kind") == "routed_ffn"]
        assert [c["layer"] for c in operations] == list(range(layers))
        assert all(c["complete"] and c["remote"] == remote for c in operations)
        assert all(c["selected_count"] == top_k for c in operations)
        value = {key: row[key] for key in (
            "total_ns", "attention_ns", "ffn_ns", "reentry_ns", "other_ns"
        )}
        value.update(
            router_ns=sum(c["router_ns"] for c in operations),
            reduction_ns=sum(c["reduction_ns"] for c in operations),
            dispatch_ns=sum(c["dispatch_ns"] for c in operations),
        )
        if not remote:
            value["local_expert_ns"] = sum(c["local_expert_ns"] for c in operations)
            totals.append(value)
            continue
        for key in (
            "requests", "request_bytes", "response_bytes", "client_encode_ns",
            "client_decode_ns", "worker_experts_sum_ns", "worker_experts_max_ns",
            "critical_worker_experts_ns", "critical_transport_remainder_ns",
            "worker_queue_sum_ns", "worker_codec_sum_ns", "fanout_wall_ns",
            "slowest_shard_wait_ns", "fanout_post_completion_ns",
        ):
            value[key] = 0
        fanouts = [c for c in calls if c.get("kind") == "expert_fanout"]
        assert [c["layer"] for c in fanouts] == list(range(layers))
        for layer in range(layers):
            shards = [c for c in calls if c.get("kind") == "expert_shard" and c["layer"] == layer]
            assert shards and all(c["complete"] for c in shards)
            assert sum(c["selected_count"] for c in shards) == top_k
            assert len({c["shard"] for c in shards}) == len(shards)
            critical = max(shards, key=lambda c: c["dispatch_finish_ns"])
            latest = critical["dispatch_finish_ns"]
            fanout = fanouts[layer]
            assert fanout["complete"] and fanout["total_ns"] >= latest
            value["fanout_wall_ns"] += fanout["total_ns"]
            value["slowest_shard_wait_ns"] += latest
            value["fanout_post_completion_ns"] += fanout["total_ns"] - latest
            worker_compute = []
            for shard in shards:
                assert shard["dispatch_finish_ns"] >= shard["dispatch_start_ns"]
                assert len(shard["transport"]) == 1
                transport = shard["transport"][0]
                assert transport["complete"] and transport["worker_profile_complete"]
                count = shard["selected_count"]
                assert transport["request_bytes"] == 40 + count * 4 + hidden * 4
                assert transport["response_bytes"] == 40 + count * (4 + hidden * 4)
                worker = transport["worker"]
                assert 0 <= worker["experts_ns"] <= worker["execute_ns"]
                assert worker["handler_ns"] >= sum(worker[k] for k in (
                    "decode_ns", "queue_ns", "execute_ns", "encode_ns"
                ))
                assert transport["transport_remainder_ns"] is not None
                assert transport["roundtrip_ns"] >= worker["handler_ns"]
                value["requests"] += 1
                value["request_bytes"] += transport["request_bytes"]
                value["response_bytes"] += transport["response_bytes"]
                value["client_encode_ns"] += transport["encode_ns"]
                value["client_decode_ns"] += transport["decode_ns"]
                value["worker_experts_sum_ns"] += worker["experts_ns"]
                value["worker_queue_sum_ns"] += worker["queue_ns"]
                value["worker_codec_sum_ns"] += worker["decode_ns"] + worker["encode_ns"]
                worker_compute.append(worker["experts_ns"])
                if shard is critical:
                    value["critical_worker_experts_ns"] += worker["experts_ns"]
                    value["critical_transport_remainder_ns"] += transport["transport_remainder_ns"]
            value["worker_experts_max_ns"] += max(worker_compute)
        totals.append(value)
    return {key: statistics.mean(row[key] for row in totals) for key in totals[0]}


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@contextmanager
def workers(root, model, out, topology, port, env, remote_urls=None, expected_bindings=None):
    if remote_urls and topology != "local":
        bindings = read_external_bindings(remote_urls, topology)
        if expected_bindings is not None and bindings != expected_bindings:
            raise ValueError("external worker bindings changed since preflight")
        save(out / "workers.json", {"lifecycle": "external", "urls": remote_urls,
                                   "commands": [], "bindings": bindings})
        yield remote_urls
        return
    processes, logs, urls, bindings, commands = [], [], [], [], []
    try:
        for i, (start, end, first, last) in enumerate(LAYOUTS[topology]):
            url = f"http://127.0.0.1:{port + i}"
            command = [str(root / "target/release/larql-server"), str(model),
                       "--host", "127.0.0.1", "--port", str(port + i), "--ffn-only",
                       "--layers", f"{start}-{end}", "--experts", f"{first}-{last}"]
            log = (out / f"worker-{i}.log").open("w")
            logs.append(log)
            process = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
            processes.append(process)
            commands.append(command)
            deadline = time.monotonic() + 600
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"worker {i} exited; see {log.name}")
                try:
                    with urllib.request.urlopen(url + "/v1/vindex3/experts", timeout=2) as response:
                        binding = json.load(response)
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"worker {i} readiness")
                    time.sleep(0.25)
            bindings.append(binding)
            urls.append(url)
        save(out / "workers.json", {"commands": commands, "bindings": bindings})
        yield urls
    finally:
        for process in processes:
            process.terminate()
        for process in processes:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for log in logs:
            log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Explicitly start model runs")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--topology", choices=["one", "experts", "mixed"], default="experts")
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--skip-decode", type=int, default=8)
    parser.add_argument("--max-warmups", type=int, default=4)
    parser.add_argument("--port", type=int, default=19381)
    parser.add_argument("--worker-url", action="append", default=[],
                        help="Existing worker base URL, repeated in topology order; never started/stopped")
    parser.add_argument("--peer-handshake", type=Path, help="Saved peer-session exclusivity acknowledgements")
    parser.add_argument("--reference", type=Path, help="Uninstrumented prompt/generated ID control")
    args = parser.parse_args()
    if not args.execute:
        parser.error("execution is held; pass --execute only after authorization")
    if args.max_warmups < 2 or args.blocks < 1 or not 0 <= args.skip_decode < args.max_tokens - 1:
        parser.error("need two warmups, a block and at least one measured decode position")
    try:
        remote_urls = external_urls(args.worker_url, args.topology) if args.worker_url else []
    except ValueError as error:
        parser.error(str(error))
    root = Path(__file__).resolve().parents[2]
    model, out = args.model.resolve(), args.out.resolve()
    index = json.loads((model / "index.json").read_text())
    if index["num_layers"] != 24:
        parser.error("this topology harness is scoped to GPT-OSS 20B")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    source_dirty = bool(subprocess.check_output(
        ["git", "diff", "HEAD", "--name-only", "--", "crates", "Cargo.toml", "Cargo.lock"],
        cwd=root, text=True))
    out.mkdir(parents=True, exist_ok=False)
    prompt = json.loads(Path(__file__).with_name("prompts.json").read_text())["p3"]
    settings = dict(LARQL_CPU_WORKERS="8", RAYON_NUM_THREADS="8",
                    VECLIB_MAXIMUM_THREADS="1", TOKIO_WORKER_THREADS="2", LARQL_KV_ENGINE="")
    env = dict(os.environ, **settings)
    handshake = args.peer_handshake.read_text() if args.peer_handshake else None
    save(out / "manifest.json", {
        "schema": "larql.v3.routed-profile.v1", "revision": revision,
        "tracked_source_dirty": source_dirty, "model": str(model), "topology": args.topology,
        "worker_lifecycle": "external" if remote_urls else "managed-loopback",
        "worker_urls": remote_urls,
        "external_worker_environment": "not managed or verified by harness" if remote_urls else None,
        "settings": settings, "prompt": prompt, "max_tokens": args.max_tokens,
        "skip_decode": args.skip_decode, "gate": 0.01,
        "peer_handshake": handshake, "peer_handshake_requires_manual_verification": True,
        "reference": str(args.reference) if args.reference else None,
        "reference_sha256": sha(args.reference) if args.reference else None,
        "binary_sha256": {name: sha(root / "target/release" / name)
                          for name in (["larql"] if remote_urls else ["larql", "larql-server"])},
        "harness_sha256": sha(Path(__file__)),
        "metadata_sha256": {name: sha(model / name) for name in ["index.json", "system_graph.json"]},
        "clock": "per-position decode wall time; startup, tokenization, prefill and output excluded",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    reference = json.loads(args.reference.read_text()) if args.reference else None
    expected_bindings = read_external_bindings(remote_urls, args.topology) if remote_urls else None
    if remote_urls:
        save(out / "preflight-workers.json", {"urls": remote_urls, "bindings": expected_bindings})
    trials, blocks = [], []

    def trial(directory, label, urls):
        nonlocal reference
        profile = directory / f"{label}.jsonl"
        command = [str(root / "target/release/larql"), "run", str(model), prompt["text"],
                   "--max-tokens", str(args.max_tokens), "--emit-ids", "--v3-profile", str(profile)]
        if urls:
            command += ["--v3-ffn-shards", ",".join(urls), "--v3-ffn-wire", "binary"]
        print(f"running {directory.name}/{label}", flush=True)
        with (directory / f"{label}.stdout").open("w") as stdout, (directory / f"{label}.stderr").open("w") as stderr:
            subprocess.run(command, cwd=root, env=env, stdout=stdout, stderr=stderr,
                           check=True, timeout=1200)
        stderr = (directory / f"{label}.stderr").read_text()
        ids = {key: json.loads(re.search(rf"{key} ids: (\[.*\])", stderr)[1]) for key in ["prompt", "generated"]}
        assert ids["prompt"] == prompt["ids"]
        assert len(ids["generated"]) == args.max_tokens, "early EOS changes the workload"
        if reference is None:
            reference = ids
        assert ids == reference, "token parity failed"
        records = [json.loads(line) for line in profile.read_text().splitlines()]
        assert records[0]["complete"]
        assert records[0]["placement"] == ("remote-routed-experts" if urls else "local")
        positions = records[1:]
        expected = ids["prompt"] + ids["generated"][:-1]
        assert [r["position"] for r in positions] == list(range(len(expected)))
        assert [r["token"] for r in positions] == expected
        measured = positions[len(ids["prompt"]) + args.skip_decode:]
        metrics = summarize(measured, bool(urls), 2880, 24, 4)
        result = {"arm": directory.name, "label": label, "command": command,
                  "measured_positions": len(measured), "ids": ids, "per_position": metrics}
        trials.append(result)
        save(out / "trials.json", trials)
        return result

    def arm(block, name, topology):
        directory = out / f"block-{block}-{name}"
        directory.mkdir()
        with workers(root, model, directory, topology, args.port, env,
                     remote_urls, expected_bindings) as urls:
            previous, plateau = None, False
            warm = []
            for i in range(args.max_warmups):
                sample = trial(directory, f"warm-{i}", urls)
                value = sample["per_position"]["total_ns"]
                warm.append(value)
                if previous is not None and agreement(previous, value) <= 0.01:
                    plateau = True
                    break
                previous = value
            sample = trial(directory, "measured", urls)
            sample["plateau"] = plateau
            sample["warm_total_ns"] = warm
            save(out / "trials.json", trials)
            return sample

    for block in range(args.blocks):
        before = arm(block, "local-before", "local")
        candidate = arm(block, args.topology, args.topology)
        after = arm(block, "local-after", "local")
        a, c, b = [x["per_position"] for x in [before, candidate, after]]
        drift = agreement(a["total_ns"], b["total_ns"])
        valid = drift <= 0.01 and all(x["plateau"] for x in [before, candidate, after])
        blocks.append({
            "block": block, "controls_agree": drift <= 0.01, "control_drift": drift,
            "all_arms_warmed": all(x["plateau"] for x in [before, candidate, after]),
            "valid_timing_block": valid, "exclusivity_evidence_supplied": handshake is not None,
            "performance_claim_requires_peer_handshake_review": True,
            "local_before": a, "candidate": c, "local_after": b,
            "net_total_delta_ns": c["total_ns"] - (a["total_ns"] + b["total_ns"]) / 2,
            "net_ffn_delta_ns": c["ffn_ns"] - (a["ffn_ns"] + b["ffn_ns"]) / 2,
            "note": "void blocks remain separate diagnostics and are never pooled",
        })
        save(out / "blocks.json", blocks)
    save(out / "verification.json", {"complete": True, "all_ids_equal": True,
        "trials": len(trials), "valid_timing_blocks": sum(b["valid_timing_block"] for b in blocks),
        "sha256": {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*")) if p.is_file()}})


if __name__ == "__main__":
    main()
