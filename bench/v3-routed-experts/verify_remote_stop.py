#!/usr/bin/env python3
"""VERIFY-REMOTE-1 revision 2: the real-model stop check (VERIFY-REMOTE-1.md).

  --diagnostic LABEL   declared before it starts; functional, never timed as a result

One prompt outside the four performance workloads
(`verify-remote/functional/stop_check.json`). A `sequential` reference at
0 ms, then `verify-2`, `verify-4` and `verify-8`, each a fresh trial
process through the relay with the generation-boundary mark. Gate 1
applies (any id that differs from the reference stops the check). The
requirements are in `verify_remote_stop_model.py`. A busy host is allowed:
nothing here is timed as a result. The relay's hash is recorded, and
whether it is the pinned build.
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
import verify_remote_model as vr
import verify_remote_stop_model as sm
from link_sweep import LINK_QOS_ENV, RELAY
from profile_exact import save, sha, workers

HARNESS = driver.HARNESS + ("verify_remote_stop.py", "verify_remote_stop_model.py")


def execute(args, root, model, out):
    env = dict(os.environ, **b1.SETTINGS)
    reference_entry, arm_entries = sm.entries()
    relay_sha = sha(root / RELAY)
    path = driver.workload_path(sm.WORKLOAD)
    manifest = {
        "schema": sm.SCHEMA, "protocol": vr.PROTOCOL, "kind": "stop-check", "label": args.diagnostic,
        "counts": False, "timed_result": False, "model": str(model), "topology": b1.TOPOLOGY,
        "provenance": b1.provenance(root, driver.TRIAL_SOURCES),
        "settings": b1.SETTINGS, "link_qos": os.environ.get(LINK_QOS_ENV, "default"),
        "workload": {"name": sm.WORKLOAD, "path": str(path), "sha256": sha(path)},
        "stop_ids": list(vr.STOP_IDS), "generate": vr.GENERATE, "rtt_ms": sm.RTT_MS,
        "schedule": [reference_entry] + arm_entries,
        "binary_sha256": {name: sha(root / p) for name, p in
                          (("v3_verify_remote", driver.TRIAL), ("larql-server", b1.SERVER), ("v3_link_shaper", RELAY))},
        "relay_is_pinned_build": relay_sha.startswith(lp.FROZEN_RELAY_SHA256_PREFIX),
        "harness_sha256": {name: sha(Path(__file__).with_name(name)) for name in HARNESS},
        "metadata_sha256": {name: sha(model / name) for name in ("index.json", "system_graph.json")},
        "host_before": lp.host_state(), "started_utc": ht.utc_now(),
    }
    save(out / "manifest.json", manifest)
    observations, trials, stop = [], {}, None
    try:
        with workers(root, model, out, b1.TOPOLOGY, args.port, env) as worker_urls:
            references = {}
            for entry in [reference_entry] + arm_entries:
                trial, relay, clock = driver.timed_trial(root, model, out, env, entry, worker_urls)
                observation = driver.observe(entry, trial, relay, references)
                observation.update(clock)
                observations.append(observation)
                trials[entry["label"]] = trial
                save(out / "trials.json", observations)
                if entry is reference_entry:
                    if sm.reference_stop(trial) != sm.NATURAL:
                        raise vr.Stop(f"the reference did not stop naturally within {vr.GENERATE} ids "
                                      f"(ended by {trial['id_counts']['ended_by']})")
                    references[sm.WORKLOAD] = trial["ids"]
    except vr.Stop as error:
        stop = str(error)
        print(f"STOPPED: {stop}", flush=True)
    except (Exception, KeyboardInterrupt) as error:
        stop = f"{type(error).__name__}: {error}"
        (out / "failure.txt").write_text(traceback.format_exc())
        print(f"STOPPED: {stop}", flush=True)
    complete = stop is None and len(observations) == 1 + len(arm_entries)
    report = {"kind": "stop-check", "counts": False, "complete": complete, "stopped": stop}
    if reference_entry["label"] in trials:
        report["reference_stop"] = sm.reference_stop(trials[reference_entry["label"]])
    if complete:
        report.update(sm.reading(trials[reference_entry["label"]],
                                 [(e["arm"], trials[e["label"]]) for e in arm_entries]))
        report["relay_marks"] = {o["label"]: (o["relay"] or {}).get("mark_defects") for o in observations}
    else:
        report["passes"] = False
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
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = execute(args, root, args.model.resolve(), out)
    print(json.dumps({k: report.get(k) for k in ("complete", "stopped", "reference_stop", "passes")}, indent=2))
    raise SystemExit(0 if report.get("passes") else 1)


if __name__ == "__main__":
    main()
