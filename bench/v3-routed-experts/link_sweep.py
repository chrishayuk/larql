#!/usr/bin/env python3
"""DEC-V3-1: exact routed experts behind a controlled link (LINK-1.md).

Two modes, neither of which runs on import:

  --calibrate   no model; drives the relay with an echo peer and records how
                closely it delivers each configured delay and rate
  --execute     GPT-OSS 20B sweep: workers stay resident, every trial is a
                fresh coordinator process, and each point is bracketed by
                unshaped relay controls

The numerical path is PROFILE-1's: binary HTTP, exact F32 frames, local
routing and ordered reduction. Only the link between coordinator and worker
changes, and every trial must reproduce the reference token IDs.
"""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import socket
import statistics
import struct
import subprocess
import threading
import time

import link_model as lm
from profile_exact import LAYOUTS, PROMPTS, agreement, save, sha, summarize, workers

# GPT-OSS 20B, the only shape this harness admits (as PROFILE-1).
LAYERS, HIDDEN, TOP_K = 24, 2880, 4
REQUEST_BYTES = 40 + 4 * TOP_K + 4 * HIDDEN
RESPONSE_BYTES = 40 + TOP_K * (4 + 4 * HIDDEN)

RELAY = "target/release/examples/v3_link_shaper"
PROBE = "target/release/examples/v3_link_probe"
CALIBRATIONS_REQUIRED = 2
# Read by the relay and the probe; recorded in every receipt they contribute to.
LINK_QOS_ENV = "LARQL_LINK_QOS"
RELAY_READY = "listening "
RELAY_START_SECONDS = 10
# A relay appends two records per connection as it closes; wait for them to
# land and stop changing before reading, instead of guessing a sleep.
STATS_SETTLE_SECONDS = 0.2
STATS_TIMEOUT_SECONDS = 10

CALIBRATION_EXCHANGES = 200
CALIBRATION_WARMUP = 20
# The echo peer is Python, so its own jitter bounds what calibration can
# resolve: accept the larger of a relative and an absolute error per exchange.
CALIBRATION_RELATIVE = 0.05
CALIBRATION_ABSOLUTE_NS = 50_000
ECHO_HEADER = struct.Struct("<II")
TRIAL_TIMEOUT_SECONDS = 3600


@contextmanager
def relays(root, upstream_urls, point, stats_dir, label):
    """One relay per worker: each shard gets its own independent link."""
    processes, urls, stats = [], [], []
    try:
        for i, upstream in enumerate(upstream_urls):
            path = stats_dir / f"{label}.relay-{i}.jsonl"
            rate = point["rate_bits_per_s"]
            process = subprocess.Popen(
                [str(root / RELAY), "127.0.0.1:0", upstream.split("://", 1)[1],
                 str(lm.one_way_delay_us(point)), lm.UNLIMITED if rate == lm.UNLIMITED else repr(float(rate)),
                 str(path)],
                stdout=subprocess.PIPE, text=True)
            processes.append(process)
            ready = {}
            reader = threading.Thread(target=lambda: ready.update(line=process.stdout.readline()), daemon=True)
            reader.start()
            reader.join(RELAY_START_SECONDS)
            line = ready.get("line", "")
            if not line.startswith(RELAY_READY):
                raise RuntimeError(f"relay {i} did not start: {line!r}")
            urls.append("http://" + line[len(RELAY_READY):].strip())
            stats.append(path)
        yield urls, stats
    finally:
        for process in processes:
            process.terminate()
        for process in processes:
            process.wait(timeout=15)


def read_relay_stats(paths):
    """Records from every relay, once each file has stopped growing."""
    records = []
    for path in paths:
        deadline, previous = time.monotonic() + STATS_TIMEOUT_SECONDS, None
        while True:
            lines = path.read_text().splitlines() if path.exists() else []
            if lines and len(lines) % 2 == 0 and lines == previous:
                break
            if time.monotonic() > deadline:
                raise TimeoutError(f"relay statistics incomplete: {path}")
            previous = lines
            time.sleep(STATS_SETTLE_SECONDS)
        records += [json.loads(line) for line in lines]
    return records


def relay_directions(records, point):
    """What each direction of a relay did, by the relay's own clock."""
    directions = {}
    for name in ("request", "response"):
        part = [r for r in records if r["direction"] == name]
        chunks = sum(r["chunks"] for r in part)
        directions[name] = {
            "requested_delay_us": lm.one_way_delay_us(point), "chunks": chunks,
            "bytes": sum(r["bytes"] for r in part),
            "late_mean_ns": sum(r["late_sum_ns"] for r in part) / chunks,
            "late_max_ns": max(r["late_max_ns"] for r in part),
        }
    return directions


class PythonProbe:
    """Blocking echo peer and client. Diagnostic only: its own wake-ups are in the path."""
    name = "python-blocking"
    authority = False

    def __init__(self, root):
        listener = socket.create_server(("127.0.0.1", 0))

        def receive(connection, count):
            data = bytearray()
            while len(data) < count:
                part = connection.recv(count - len(data))
                if not part:
                    return None
                data += part
            return data

        def serve():
            while True:
                connection, _ = listener.accept()
                connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                with connection:
                    while (header := receive(connection, ECHO_HEADER.size)) is not None:
                        request, response = ECHO_HEADER.unpack(header)
                        if receive(connection, request) is None:
                            break
                        connection.sendall(bytes(response))

        threading.Thread(target=serve, daemon=True).start()
        self.address = "%s:%d" % listener.getsockname()

    def measure(self, address):
        host, port = address.rsplit(":", 1)
        payload = ECHO_HEADER.pack(REQUEST_BYTES, RESPONSE_BYTES) + bytes(REQUEST_BYTES)
        samples = []
        with socket.create_connection((host, int(port))) as connection:
            connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            for i in range(CALIBRATION_EXCHANGES + CALIBRATION_WARMUP):
                started, remaining = time.perf_counter_ns(), RESPONSE_BYTES
                connection.sendall(payload)
                while remaining:
                    part = connection.recv(remaining)
                    if not part:
                        raise ConnectionError("echo peer closed")
                    remaining -= len(part)
                if i >= CALIBRATION_WARMUP:
                    samples.append(time.perf_counter_ns() - started)
        return {"probe": self.name, "exchanges": len(samples), "median_ns": statistics.median(samples),
                "min_ns": min(samples), "p90_ns": statistics.quantiles(samples, n=10)[-1],
                "max_ns": max(samples), "mean_ns": statistics.mean(samples)}

    def close(self):
        pass


class NativeProbe:
    """Polling echo peer and client (`v3_link_probe`): the calibration authority."""
    name = "native-polling"
    authority = True

    def __init__(self, root):
        self.binary = str(root / PROBE)
        self.peer = subprocess.Popen([self.binary, "serve"], stdout=subprocess.PIPE, text=True)
        line = self.peer.stdout.readline()
        if not line.startswith(RELAY_READY):
            raise RuntimeError(f"probe peer did not start: {line!r}")
        self.address = line[len(RELAY_READY):].strip()

    def measure(self, address):
        output = subprocess.check_output(
            [self.binary, "measure", address, str(REQUEST_BYTES), str(RESPONSE_BYTES),
             str(CALIBRATION_EXCHANGES), str(CALIBRATION_WARMUP)], text=True)
        return json.loads(output)

    def close(self):
        self.peer.terminate()
        self.peer.wait(timeout=15)


def calibrate(root, out, probe):
    """Achieved against configured link cost at every frozen point, without a model.

    Each point is measured between two zero-delay relay baselines. Both
    baselines, their drift and the error against each are recorded, and a
    point passes only if it is within tolerance of BOTH: a drifting baseline
    fails the point instead of being averaged into a pass.
    """
    shape = {"crossings": 1,
             "critical_bits": (ECHO_HEADER.size + REQUEST_BYTES + RESPONSE_BYTES) * lm.BITS_PER_BYTE}
    points = [p for p in lm.frozen_points() if p["kind"] != "direct"]

    def through_relay(point, label):
        with relays(root, ["http://" + probe.address], point, out, label) as (urls, stats):
            sample = probe.measure(urls[0].split("://", 1)[1])
            records = read_relay_stats(stats)
        sample["relay"] = relay_directions(records, point)
        sample["host_load_average"] = os.getloadavg()
        return sample

    rows = []
    direct_before = probe.measure(probe.address)
    for i, point in enumerate(points):
        before = through_relay(dict(lm.FLOOR), f"calibrate-{i}-floor-before")
        shaped = through_relay(point, f"calibrate-{i}-point")
        after = through_relay(dict(lm.FLOOR), f"calibrate-{i}-floor-after")
        expected = sum(lm.regressors(point, shape))
        tolerance = max(CALIBRATION_RELATIVE * expected, CALIBRATION_ABSOLUTE_NS)
        errors = {name: shaped["median_ns"] - floor["median_ns"] - expected
                  for name, floor in (("before", before), ("after", after))}
        rows.append({
            "point": point, "expected_extra_ns": expected, "tolerance_ns": tolerance,
            "floor_before": before, "shaped": shaped, "floor_after": after,
            "floor_drift_ns": after["median_ns"] - before["median_ns"],
            "error_against_floor_before_ns": errors["before"],
            "error_against_floor_after_ns": errors["after"],
            "within_tolerance": all(abs(e) <= tolerance for e in errors.values()),
        })
        print(f"{point['rtt_ms']:>6} ms {str(point['rate_bits_per_s']):>12} b/s  expected {expected / 1e3:>9.1f} us"
              f"  error {errors['before'] / 1e3:>8.1f} / {errors['after'] / 1e3:>8.1f} us"
              f"  tolerance {tolerance / 1e3:>7.1f}  {'ok' if rows[-1]['within_tolerance'] else 'MISS'}", flush=True)
    direct_after = probe.measure(probe.address)
    result = {"schema": "larql.v3.link-calibration.v2", "probe": probe.name,
              "calibration_authority": probe.authority,
              "link_qos": os.environ.get(LINK_QOS_ENV, "default"),
              "request_bytes": REQUEST_BYTES, "response_bytes": RESPONSE_BYTES,
              "exchanges_per_measurement": CALIBRATION_EXCHANGES, "statistic": "median",
              "tolerance": {"relative": CALIBRATION_RELATIVE, "absolute_ns": CALIBRATION_ABSOLUTE_NS},
              "binary_sha256": {"v3_link_shaper": sha(root / RELAY), "v3_link_probe": sha(root / PROBE)},
              "direct_before": direct_before, "direct_after": direct_after, "points": rows,
              "all_within_tolerance": all(r["within_tolerance"] for r in rows),
              "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    save(out / "calibration.json", result)
    return result


def qualified_calibrations(root, paths):
    """The timed sweep needs two complete passing native calibrations of this relay build."""
    if len(paths) < CALIBRATIONS_REQUIRED:
        raise ValueError(f"need {CALIBRATIONS_REQUIRED} passing calibration receipts")
    receipts = []
    for path in paths:
        receipt = json.loads(path.read_text())
        if not (receipt.get("calibration_authority") and receipt.get("all_within_tolerance")):
            raise ValueError(f"not a passing authoritative calibration: {path}")
        if receipt["binary_sha256"]["v3_link_shaper"] != sha(root / RELAY):
            raise ValueError(f"calibration was recorded for a different relay build: {path}")
        if receipt["link_qos"] != os.environ.get(LINK_QOS_ENV, "default"):
            raise ValueError(f"calibration used a different relay scheduling class: {path}")
        receipts.append({"path": str(path), "sha256": sha(path), "recorded_utc": receipt["recorded_utc"]})
    return receipts


def sweep(args, root, model, out):
    calibrations = qualified_calibrations(root, args.calibration)
    fixture = json.loads(Path(__file__).with_name(PROMPTS).read_text())
    prompt = fixture["p3"]
    reference = json.loads(args.reference.read_text())
    if reference["prompt"] != prompt["ids"]:
        raise ValueError(f"reference was not recorded on workload {fixture['workload']}")
    settings = dict(LARQL_CPU_WORKERS="8", RAYON_NUM_THREADS="8",
                    VECLIB_MAXIMUM_THREADS="1", TOKIO_WORKER_THREADS="2", LARQL_KV_ENGINE="")
    env = dict(os.environ, **settings)
    order = lm.schedule(lm.frozen_points(), args.repeats)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "crates", "Cargo.toml", "Cargo.lock"], cwd=root, text=True)
    handshake = args.peer_handshake.read_text() if args.peer_handshake else None
    save(out / "manifest.json", {
        "schema": "larql.v3.link-sweep.v1", "revision": revision, "source_dirty": dirty.splitlines(),
        "workload": fixture["workload"], "calibrations": calibrations,
        "link_qos": os.environ.get(LINK_QOS_ENV, "default"),
        "model": str(model), "topology": args.topology, "settings": settings, "prompt": prompt,
        "max_tokens": args.max_tokens, "skip_decode": args.skip_decode, "gate": lm.GATE,
        "schedule": order, "reference": str(args.reference), "reference_sha256": sha(args.reference),
        "peer_handshake": handshake, "peer_handshake_requires_manual_verification": True,
        "binary_sha256": {name: sha(root / "target/release" / name) for name in ("larql", "larql-server")}
                         | {"v3_link_shaper": sha(root / RELAY)},
        "harness_sha256": {name: sha(Path(__file__).with_name(name))
                           for name in ("link_sweep.py", "link_model.py", "profile_exact.py")},
        "metadata_sha256": {name: sha(model / name) for name in ("index.json", "system_graph.json")},
        "link": "userspace relay per worker on loopback: delay and serialization rate only",
        "clock": "per-position decode wall time; startup, tokenization, prefill and output excluded",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    trials = []

    def trial(label, point, worker_urls, floor_ns):
        profile = out / f"{label}.jsonl"

        def run(urls):
            command = [str(root / "target/release/larql"), "run", str(model), prompt["text"],
                       "--max-tokens", str(args.max_tokens), "--emit-ids", "--raw", "--v3-profile", str(profile),
                       "--v3-ffn-shards", ",".join(urls), "--v3-ffn-wire", "binary"]
            with (out / f"{label}.stdout").open("w") as stdout, (out / f"{label}.stderr").open("w") as stderr:
                subprocess.run(command, cwd=root, env=env, stdout=stdout, stderr=stderr,
                               check=True, timeout=TRIAL_TIMEOUT_SECONDS)

        print(f"running {label}: {point['kind']} rtt={point['rtt_ms']}ms rate={point['rate_bits_per_s']}", flush=True)
        relay_records = None
        if point["kind"] == "direct":
            run(worker_urls)
        else:
            with relays(root, worker_urls, point, out, label) as (urls, stats):
                run(urls)
                relay_records = read_relay_stats(stats)
        stderr = (out / f"{label}.stderr").read_text()
        ids = {key: json.loads(re.search(rf"{key} ids: (\[.*\])", stderr)[1]) for key in ("prompt", "generated")}
        if ids != reference:
            raise AssertionError(f"token parity failed in {label}")
        records = [json.loads(line) for line in profile.read_text().splitlines()]
        assert records[0]["complete"] and records[0]["placement"] == "remote-routed-experts"
        measured = records[1:][len(ids["prompt"]) + args.skip_decode:]
        per_position = summarize(measured, True, HIDDEN, LAYERS, TOP_K)
        result = {"label": label, "point": point, "measured_positions": len(measured),
                  "total_ns": per_position["total_ns"], "ffn_ns": per_position["ffn_ns"],
                  "per_position": per_position, "shape": lm.link_shape(measured, LAYERS)}
        if relay_records is not None:
            # Relays also carried prefill and the discarded positions, so
            # normalise their lateness by every position the trial executed.
            result["relay"] = lm.relay_gate(relay_records, len(records) - 1,
                                            floor_ns or per_position["total_ns"])
        trials.append(result)
        save(out / "trials.json", trials)
        return result

    with workers(root, model, out, args.topology, args.port, env) as worker_urls:
        warm, plateau = [], False
        for i in range(args.max_warmups):
            warm.append(trial(f"warm-{i}", dict(lm.FLOOR, repeat=-1), worker_urls, None)["total_ns"])
            if len(warm) > 1 and agreement(warm[-2], warm[-1]) <= lm.GATE:
                plateau = True
                break
        del trials[:]
        floor_ns = warm[-1]
        for index, point in enumerate(order):
            result = trial(f"trial-{index:03d}", point, worker_urls, floor_ns)
            if point["kind"] == "floor":
                floor_ns = result["total_ns"]
    report = lm.analyse(trials)
    report.update(warm_total_ns=warm, warm_plateau=plateau, topology=args.topology,
                  exclusivity_evidence_supplied=handshake is not None,
                  performance_claim_requires_peer_handshake_review=True,
                  note="void points stay in observations and are never fitted or pooled")
    save(out / "analysis.json", report)
    save(out / "verification.json", {"complete": True, "all_ids_equal": True, "trials": len(trials),
         "sha256": {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*")) if p.is_file()}})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--calibrate", action="store_true", help="Relay calibration only; no model")
    mode.add_argument("--execute", action="store_true", help="Explicitly start model runs")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--topology", choices=["one", "experts", "mixed"], default="one")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--skip-decode", type=int, default=8)
    parser.add_argument("--max-warmups", type=int, default=4)
    parser.add_argument("--port", type=int, default=19481)
    parser.add_argument("--reference", type=Path,
                        default=Path(__file__).with_name("link-results") / "control" / "ids.json",
                        help="Uninstrumented prompt/generated ID control")
    parser.add_argument("--probe", choices=["native", "python"], default="native",
                        help="Calibration probe; only native is the authority, python is a diagnostic")
    parser.add_argument("--link-qos", choices=["default", "user-interactive"], default="default",
                        help="Scheduling class requested by relay and probe threads (a request, not core affinity)")
    parser.add_argument("--calibration", type=Path, action="append", default=[],
                        help="Passing native calibration receipt; --execute needs two")
    parser.add_argument("--peer-handshake", type=Path, help="Saved peer-session exclusivity acknowledgements")
    args = parser.parse_args()
    os.environ[LINK_QOS_ENV] = args.link_qos
    root = Path(__file__).resolve().parents[2]
    if not (root / RELAY).exists():
        parser.error(f"build the relay first: {RELAY}")
    out = args.out.resolve()
    if args.calibrate:
        out.mkdir(parents=True, exist_ok=False)
    if args.calibrate:
        probe = (NativeProbe if args.probe == "native" else PythonProbe)(root)
        try:
            result = calibrate(root, out, probe)
        finally:
            probe.close()
        raise SystemExit(0 if result["all_within_tolerance"] or not probe.authority else 1)
    try:
        qualified_calibrations(root, args.calibration)
    except ValueError as error:
        parser.error(str(error))
    if args.model is None:
        parser.error("--execute needs --model")
    if args.max_warmups < 2 or args.repeats < 1 or not 0 <= args.skip_decode < args.max_tokens - 1:
        parser.error("need two warmups, a repeat and at least one measured decode position")
    model = args.model.resolve()
    if json.loads((model / "index.json").read_text())["num_layers"] != LAYERS or args.topology not in LAYOUTS:
        parser.error("this harness is scoped to GPT-OSS 20B")
    out.mkdir(parents=True, exist_ok=False)
    report = sweep(args, root, model, out)
    print(json.dumps({k: report[k] for k in ("valid", "void", "model", "held_out")}, indent=2))


if __name__ == "__main__":
    main()
