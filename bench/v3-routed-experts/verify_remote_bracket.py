#!/usr/bin/env python3
"""VERIFY-REMOTE-1 revision 2: the bracket study (VERIFY-REMOTE-1.md, "Revision 2: bracket length").

  --diagnostic LABEL   declared before it starts; it never counts and measures no arm's speed
  --peer-handshake F   required: the study runs only in a quiet window agreed with the
                       priority session, and refuses to start while a compiler or model
                       process is running

Eight stand-in units. Each opens with the 16, 32 and 64-id brackets in a
seeded order, runs the filler (`explain` at 20 ms, the four arms in a fixed
order), and closes with the same brackets in the same order. Warm-up,
references (`prose_free` and `explain`, gate 1 on every trial), relays with
the generation-boundary mark, telemetry, the stall watch and the
`PAUSE`/`PAUSED` hold (before a stand-in unit only) are the main driver's.
The selection rule is in `verify_remote_bracket_model.py`, fixed before the
run.
"""
import argparse
import json
import os
from pathlib import Path
import traceback

import block_time as b1
import block_time_2 as d2
import host_telemetry as ht
import link_pair as lp
import verify_remote as driver
import verify_remote_bracket_model as bs
import verify_remote_model as vr
from link_sweep import LINK_QOS_ENV, RELAY
from profile_exact import save, sha, workers

HARNESS = driver.HARNESS + ("verify_remote_bracket.py", "verify_remote_bracket_model.py")


def take_references(root, model, out, env, worker_urls, references, observations):
    """Sequential references at 0 ms for the bracket and filler workloads, saved as they come."""
    for workload in bs.REFERENCE_WORKLOADS:
        entry = vr.reference_entry(workload)
        trial, relay, clock = driver.timed_trial(root, model, out, env, entry, worker_urls)
        observation = driver.observe(entry, trial, relay, references)
        observation.update(clock)
        observations.append(observation)
        save(out / "references_run.json", observations)
        if observation["ledger_defects"] or observation["frame_defects"]:
            raise vr.Stop(f"the {workload} reference is not usable: "
                          f"{observation['ledger_defects'] + observation['frame_defects']}")
        references[workload] = trial["ids"]
        save(out / "references.json", {"ids": references})


def execute(args, root, model, out):
    env = dict(os.environ, **b1.SETTINGS)
    order = bs.schedule()
    handshake = args.peer_handshake.read_text()
    started = ht.utc_now()
    manifest = {
        "schema": bs.SCHEMA, "protocol": vr.PROTOCOL, "kind": "bracket-study", "label": args.diagnostic,
        "counts": False, "measures_arm_speed": False, "continue_unsettled": args.continue_unsettled,
        "model": str(model), "topology": b1.TOPOLOGY,
        "provenance": b1.provenance(root, driver.TRIAL_SOURCES),
        "trial_environment": {name: value for name, value in sorted(env.items())
                              if name.startswith(b1.ENVIRONMENT_PREFIXES)},
        "settings": b1.SETTINGS, "link_qos": os.environ.get(LINK_QOS_ENV, "default"),
        "candidates": list(bs.CANDIDATES), "stand_ins": bs.STAND_INS, "seed": bs.SEED,
        "bracket_orders": bs.bracket_orders(),
        "bracket": {"workload": vr.BRACKET_WORKLOAD, "rtt_ms": vr.BRACKET_RTT_MS, "arm": vr.SEQUENTIAL},
        "filler": {"workload": bs.FILLER_WORKLOAD, "rtt_ms": bs.FILLER_RTT_MS, "arms": list(bs.FILLER_ARMS),
                   "generate": vr.GENERATE},
        "selection_rule": f"shortest length with at least {bs.QUALIFYING_WITHIN} of {bs.STAND_INS} drifts "
                          f"within {bs.GATE}; none selected if none qualifies",
        "missed_disturbances": "filler spread over the gate while a candidate's brackets agree; reported, "
                               "not part of the rule",
        "workloads": {name: {"path": str(driver.workload_path(name)), "sha256": sha(driver.workload_path(name))}
                      for name in bs.REFERENCE_WORKLOADS},
        "stop_ids": list(vr.STOP_IDS), "schedule": order, "gate": bs.GATE,
        "binary_sha256": {name: sha(root / path) for name, path in
                          (("v3_verify_remote", driver.TRIAL), ("larql-server", b1.SERVER), ("v3_link_shaper", RELAY))},
        "relay_build": lp.relay_build(sha(root / RELAY)),
        "harness_sha256": {name: sha(Path(__file__).with_name(name)) for name in HARNESS},
        "metadata_sha256": {name: sha(model / name) for name in ("index.json", "system_graph.json")},
        "host_before": lp.host_state(), "peer_handshake": handshake,
        "telemetry": {"process_interval_s": ht.PROCESS_INTERVAL_S, "state_interval_s": ht.STATE_INTERVAL_S,
                      "gates_anything": False},
        "pause": {"file": d2.PAUSE_FILE, "where": "before a stand-in unit only", "poll_s": d2.PAUSE_POLL_S},
        "stall_watch": {"tick_s": vr.b2.STALL_TICK_S, "gap_s": vr.b2.STALL_GAP_S,
                        "effect": "a trial overlapping a stall leaves its unit uncounted"},
        "started_utc": started,
    }
    save(out / "manifest.json", manifest)
    observations, references, references_run, stop, pauses = [], {}, [], None, []
    with ht.Sampler(out / "telemetry.jsonl") as sampler, d2.StallWatch() as watch:
        try:
            with workers(root, model, out, b1.TOPOLOGY, args.port, env) as worker_urls:
                d2.wait_while_paused(out, driver.BEFORE_WARM_UP, pauses)
                driver.warm_up(root, model, args, out, env, worker_urls)
                take_references(root, model, out, env, worker_urls, references, references_run)
                for entry in order:
                    if bs.is_pause_point(entry):
                        d2.wait_while_paused(out, entry["unit"], pauses)
                        save(out / "pauses.json", pauses)
                    trial, relay, clock = driver.timed_trial(root, model, out, env, entry, worker_urls)
                    observation = driver.observe(entry, trial, relay, references)
                    observation.update(clock)
                    observations.append(observation)
                    save(out / "trials.json", observations)
        except vr.Stop as error:
            stop = str(error)
            print(f"STOPPED: {stop}", flush=True)
        except (Exception, KeyboardInterrupt) as error:
            stop = f"{type(error).__name__}: {error}"
            (out / "failure.txt").write_text(traceback.format_exc())
            print(f"STOPPED: {stop}", flush=True)
    ended = ht.utc_now()
    vr.attach_stalls(observations + references_run, watch.stalls)
    save(out / "suspensions.json", watch.stalls)
    save(out / "references_run.json", references_run)
    save(out / "trials.json", observations)
    complete = stop is None and len(observations) == len(order)
    readings = bs.unit_readings(observations)
    report = {"units": readings}
    if complete:
        report["selection"] = bs.select(readings)
    else:
        report["note"] = "the study stopped; no length is selected"
    host_after = lp.host_state()
    report.update(kind="bracket-study", counts=False, complete=complete, stopped=stop,
                  scheduled=len(order), observed=len(observations), telemetry_samples=sampler.samples,
                  pauses=pauses, suspensions=watch.stalls, host_after=host_after,
                  quiet_at_end=not host_after["busy_processes"], ended_utc=ended)
    save(out / "analysis.json", report)
    try:
        save(out / "spotlight.json", ht.spotlight_starts(started, ended))
    except (Exception, KeyboardInterrupt) as error:
        save(out / "spotlight.json", {"error": f"{type(error).__name__}: {error}"})
    save(out / "verification.json",
         {"complete": complete, "sha256": {str(p.relative_to(out)): sha(p)
                                           for p in sorted(out.rglob("*")) if p.is_file()}})
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--diagnostic", metavar="LABEL", required=True, help="Declared label; never counts")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--peer-handshake", type=Path, required=True,
                        help="Saved acknowledgements of the agreed quiet window")
    parser.add_argument("--port", type=int, default=19581)
    parser.add_argument("--continue-unsettled", action="store_true",
                        help="Run the schedule even if warm-up never settles (recorded)")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    for binary in (driver.TRIAL, b1.SERVER, RELAY):
        if not (root / binary).exists():
            parser.error(f"build first: {binary}")
    if lp.relay_build(sha(root / RELAY)) is None:
        parser.error("the relay binary is neither the certified build nor the named candidate")
    if os.environ.get(LINK_QOS_ENV, "default") != "default":
        parser.error("the relay was qualified at the default scheduling class only")
    if not args.peer_handshake.exists() or not args.peer_handshake.read_text().strip():
        parser.error("the peer handshake file is missing or empty")
    busy = lp.busy_processes()
    if busy:
        parser.error(f"the study needs a quiet machine; running: {busy}")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = execute(args, root, args.model.resolve(), out)
    print(json.dumps(report.get("selection", report.get("note")), indent=2))
    raise SystemExit(0 if report["complete"] else 1)


if __name__ == "__main__":
    main()
