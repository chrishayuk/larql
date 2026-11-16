#!/usr/bin/env python3
"""VERIFY-REMOTE-1 revision 2: the relay mark's functional check on the real model (VERIFY-REMOTE-1.md).

  --diagnostic LABEL   declared before it starts; functional, never timed as a result

One trial per arm (`sequential` first) on `json_edit` at 20 ms, each a fresh
trial process through the relay with the generation-boundary mark. Gate 1
applies against the `sequential` trial's ids. The requirements are in
`verify_remote_mark_check_model.py`. A busy host is allowed: whether the
relay kept to schedule is reported, not required.
"""
import argparse
import json
import os
from pathlib import Path
import traceback

import block_time as b1
import host_telemetry as ht
import link_pair as lp
import verify_remote as driver
import verify_remote_mark_check_model as mc
import verify_remote_model as vr
from link_sweep import LINK_QOS_ENV, RELAY
from profile_exact import save, sha, workers

HARNESS = driver.HARNESS + ("verify_remote_mark_check.py", "verify_remote_mark_check_model.py")


def execute(args, root, model, out):
    env = dict(os.environ, **b1.SETTINGS)
    order = mc.entries()
    path = driver.workload_path(mc.WORKLOAD)
    relay_sha = sha(root / RELAY)
    manifest = {
        "schema": mc.SCHEMA, "protocol": vr.PROTOCOL, "kind": "mark-check", "label": args.diagnostic,
        "counts": False, "timed_result": False, "model": str(model), "topology": b1.TOPOLOGY,
        "provenance": b1.provenance(root, driver.TRIAL_SOURCES),
        "settings": b1.SETTINGS, "link_qos": os.environ.get(LINK_QOS_ENV, "default"),
        "workload": {"name": mc.WORKLOAD, "path": str(path), "sha256": sha(path)},
        "stop_ids": list(vr.STOP_IDS), "generate": vr.GENERATE, "rtt_ms": mc.RTT_MS, "schedule": order,
        "binary_sha256": {name: sha(root / p) for name, p in
                          (("v3_verify_remote", driver.TRIAL), ("larql-server", b1.SERVER), ("v3_link_shaper", RELAY))},
        "relay_build": lp.relay_build(relay_sha),
        "harness_sha256": {name: sha(Path(__file__).with_name(name)) for name in HARNESS},
        "metadata_sha256": {name: sha(model / name) for name in ("index.json", "system_graph.json")},
        "host_before": lp.host_state(), "started_utc": ht.utc_now(),
    }
    save(out / "manifest.json", manifest)
    observations, results, stop = [], [], None
    try:
        with workers(root, model, out, b1.TOPOLOGY, args.port, env) as worker_urls:
            references = {}
            for entry in order:
                trial, relay, clock = driver.timed_trial(root, model, out, env, entry, worker_urls)
                observation = driver.observe(entry, trial, relay, references)
                observation.update(clock)
                observations.append(observation)
                results.append((entry["arm"], trial, relay))
                save(out / "trials.json", observations)
                if entry["arm"] == vr.SEQUENTIAL:
                    references[mc.WORKLOAD] = trial["ids"]
    except vr.Stop as error:
        stop = str(error)
        print(f"STOPPED: {stop}", flush=True)
    except (Exception, KeyboardInterrupt) as error:
        stop = f"{type(error).__name__}: {error}"
        (out / "failure.txt").write_text(traceback.format_exc())
        print(f"STOPPED: {stop}", flush=True)
    complete = stop is None and len(results) == len(order)
    report = {"kind": "mark-check", "counts": False, "complete": complete, "stopped": stop, "passes": False}
    if complete:
        try:
            report.update(mc.reading(results))
        except Exception as error:  # the records stay; the reading's failure is recorded
            report.update(passes=False, reading_error=f"{type(error).__name__}: {error}")
            (out / "failure.txt").write_text(traceback.format_exc())
    report.update(host_after=lp.host_state(), ended_utc=ht.utc_now())
    save(out / "analysis.json", report)
    save(out / "verification.json",
         {"complete": complete, "sha256": {str(p.relative_to(out)): sha(p)
                                           for p in sorted(out.rglob("*")) if p.is_file()}})
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--diagnostic", metavar="LABEL", required=True, help="Declared label; never counts")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--port", type=int, default=19581)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    for binary in (driver.TRIAL, b1.SERVER, RELAY):
        if not (root / binary).exists():
            parser.error(f"build first: {binary}")
    if lp.relay_build(sha(root / RELAY)) is None:
        parser.error("the relay binary is neither the certified build nor the named candidate")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = execute(args, root, args.model.resolve(), out)
    print(json.dumps({k: report.get(k) for k in ("complete", "stopped", "passes")}, indent=2))
    raise SystemExit(0 if report.get("passes") else 1)


if __name__ == "__main__":
    main()
