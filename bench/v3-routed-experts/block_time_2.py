#!/usr/bin/env python3
"""BLOCK-TIME-2 driver (BLOCK-TIME-2.md, revision 1, frozen). Nothing runs on import.

  --diagnostic LABEL   a development run; declared before it starts, never counted
  --admissible         the one run the protocol admits; needs two passing
                       block-frame qualification receipts and a quiet machine

Twenty independently bracketed pairs, each `bracket, arm at one delay, the
same arm at the other, bracket`. Trials are run by BLOCK-TIME-1's runner and
judged by its per-trial gates; validity and the analysis are per pair, in
`block_time_2_model.py`. Host telemetry is sampled throughout and never
gates anything.
"""
import argparse
import json
import os
from pathlib import Path
import threading
import time
import traceback

import block_time as b1
import block_time_2_model as b2
import block_time_model as bt
import host_telemetry as ht
import link_block_frames_model as bm
import link_pair as lp
from link_sweep import LINK_QOS_ENV, RELAY
from profile_exact import agreement, save, sha, workers

HARNESS = ("block_time_2.py", "block_time_2_model.py", "host_telemetry.py", "block_time.py",
           "block_time_model.py", "link_model.py", "link_sweep.py", "link_block_frames_model.py",
           "profile_exact.py")
# Another session that needs the machine creates this file in the run
# directory; the driver finishes the pair it is in, then waits until the file
# is removed. Pausing only between pairs voids nothing, because every pair is
# bracketed on its own.
PAUSE_FILE = "PAUSE"
# Written by the driver while it is holding, and removed when it resumes, so
# the session that asked knows when the in-flight pair has finished.
PAUSED_FILE = "PAUSED"
PAUSE_POLL_S = 5.0
BEFORE_WARM_UP = -1


def is_pause_point(entry):
    """The first trial of a pair: the only place the run may hold."""
    return entry["role"] == bt.ROLE_BRACKET and entry["bracket"] == "before"


def wait_while_paused(out, before_pair, pauses, sleep=time.sleep, now=None):
    """Hold at a pair boundary while the pause file exists; record the pause, if any."""
    now = now or ht.utc_now
    flag = out / PAUSE_FILE
    if not flag.exists():
        return
    started = now()
    held = out / PAUSED_FILE
    held.write_text(f"holding before pair {before_pair} since {started}; remove {PAUSE_FILE} to resume\n")
    print(f"paused before pair {before_pair} at {started}: remove {flag} to resume", flush=True)
    try:
        while flag.exists():
            sleep(PAUSE_POLL_S)
    finally:
        held.unlink(missing_ok=True)
    pauses.append({"before_pair": before_pair, "paused_utc": started, "resumed_utc": now()})
    print(f"resumed before pair {before_pair}", flush=True)


class StallWatch:
    """Ticks in a thread of the driver and records every stall (rule 5, `block_time_2_model`).

    It sees the driver's own process only. Stopping the run's process group,
    the documented way to stop everything, stops the driver too and is seen.
    A stop of the worker, relay or trial process alone is not.
    """

    def __init__(self, tick=b2.STALL_TICK_S, clocks=lambda: (time.monotonic(), time.time())):
        self.tick, self.clocks = tick, clocks
        self.stalls, self.stop_event = [], threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def loop(self):
        previous = self.clocks()
        while not self.stop_event.wait(self.tick):
            current = self.clocks()
            found = b2.stall_between(previous, current)
            if found:
                self.stalls.append(found)
            previous = current

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop_event.set()
        self.thread.join(timeout=5)
        return False


def timed_trial(root, model, ids, out, env, entry, worker_urls):
    started, started_monotonic = ht.utc_now(), time.monotonic()
    trial, relay_records = b1.run_trial(root, model, ids, out, env, entry, worker_urls)
    return trial, relay_records, {"started_utc": started, "ended_utc": ht.utc_now(),
                                  "started_monotonic": started_monotonic, "ended_monotonic": time.monotonic()}


def warm_up(root, model, args, out, env, worker_urls, saved, stream):
    """BLOCK-TIME-1's warm-up rule, unchanged: returns the reference and the last warm time."""
    warm, reference = [], None
    for index in range(bt.MAX_WARMUPS):
        entry = {"label": f"warm-{index}", "role": "warm", "arm": bt.SEQUENTIAL, "rtt_ms": 0.0,
                 "repeat": -1, "profiled": True}
        trial, relay_records, _ = timed_trial(root, model, args.ids, out, env, entry, worker_urls)
        if reference is None:
            defects = bt.reference_defects(len(saved["prompt"]), stream, trial)
            if defects:
                raise b1.Stop(f"the reference trial is not usable: {defects}")
            reference = trial["logits_sha256"]
            save(out / "reference.json", {"logits_sha256": reference, "argmax": trial["argmax"]})
        warm.append(b1.observe(entry, trial, relay_records, reference, None)["summary"]["per_position_ns"])
        if len(warm) > 1 and agreement(warm[-2], warm[-1]) <= bt.GATE:
            break
    else:
        if not args.continue_unsettled:
            raise b1.Stop(f"warm-up never settled within {bt.GATE}: {warm}")
        print(f"warm-up never settled within {bt.GATE}; continuing as declared: {warm}", flush=True)
    save(out / "warm.json", {"per_position_ns": warm, "settled": len(warm) < bt.MAX_WARMUPS
                             or agreement(warm[-2], warm[-1]) <= bt.GATE})
    return reference, warm[-1]


def execute(args, root, model, out, kind):
    receipts = [json.loads(path.read_text()) for path in args.frames_qualification]
    if kind == "admissible":
        bm.check_qualification(receipts, sha(root / RELAY))
    saved = json.loads(args.ids.read_text())
    stream = saved["prompt"] + saved["generated"]
    if len(stream) != bt.POSITIONS:
        raise ValueError(f"the supplied stream has {len(stream)} ids, not {bt.POSITIONS}")
    env = dict(os.environ, **b1.SETTINGS)
    order = b2.schedule()
    handshake = args.peer_handshake.read_text() if args.peer_handshake else None
    started = ht.utc_now()
    manifest = {
        "schema": b2.SCHEMA, "protocol": b2.PROTOCOL, "kind": kind, "label": args.diagnostic,
        "counts": kind == "admissible", "continue_unsettled": args.continue_unsettled,
        "model": str(model), "topology": b1.TOPOLOGY,
        "provenance": b1.provenance(root, b1.TRIAL_SOURCES),
        "trial_environment": {name: value for name, value in sorted(env.items())
                              if name.startswith(b1.ENVIRONMENT_PREFIXES)},
        "settings": b1.SETTINGS, "link_qos": os.environ.get(LINK_QOS_ENV, "default"),
        "ids": str(args.ids), "ids_sha256": sha(args.ids), "prompt_tokens": len(saved["prompt"]),
        "positions": bt.POSITIONS, "measure_from": bt.MEASURE_FROM,
        "shuffle_seed": b2.SHUFFLE_SEED, "pairs": b2.pair_units(), "schedule": order,
        "thresholds": {"gate": b2.GATE, "slope_band": b2.SLOPE_BAND,
                       "minimum_valid_pairs": b2.MINIMUM_VALID_PAIRS,
                       "within_pair_consistency": b2.CONSISTENCY},
        "frames_qualification": [{"path": str(path), "sha256": sha(path)} for path in args.frames_qualification],
        "binary_sha256": {name: sha(root / path) for name, path in
                          (("v3_block_time", b1.TRIAL), ("larql-server", b1.SERVER), ("v3_link_shaper", RELAY))},
        "harness_sha256": {name: sha(Path(__file__).with_name(name)) for name in HARNESS},
        "metadata_sha256": {name: sha(model / name) for name in ("index.json", "system_graph.json")},
        "host_before": lp.host_state(), "peer_handshake": handshake,
        "telemetry": {"process_interval_s": ht.PROCESS_INTERVAL_S, "state_interval_s": ht.STATE_INTERVAL_S,
                      "gates_anything": False},
        "clock": "monotonic wall time around the measured region inside each trial process",
        "pause": {"file": PAUSE_FILE, "where": "between pairs only", "poll_s": PAUSE_POLL_S},
        "stall_watch": {"tick_s": b2.STALL_TICK_S, "gap_s": b2.STALL_GAP_S, "sees": "the driver process only",
                        "effect": "a trial overlapping a stall voids its pair"},
        "started_utc": started,
    }
    save(out / "manifest.json", manifest)
    observations, stop, pauses = [], None, []
    with ht.Sampler(out / "telemetry.jsonl") as sampler, StallWatch() as watch:
        try:
            with workers(root, model, out, b1.TOPOLOGY, args.port, env) as worker_urls:
                wait_while_paused(out, BEFORE_WARM_UP, pauses)
                reference, floor_ns = warm_up(root, model, args, out, env, worker_urls, saved, stream)
                for entry in order:
                    if is_pause_point(entry):
                        wait_while_paused(out, entry["pair"], pauses)
                        save(out / "pauses.json", pauses)
                    trial, relay_records, clock = timed_trial(root, model, args.ids, out, env, entry, worker_urls)
                    observation = b1.observe(entry, trial, relay_records, reference, floor_ns)
                    observation.update(clock, exchange_remainder=b2.largest_exchange_remainder(trial, entry["rtt_ms"]))
                    if entry["role"] == bt.ROLE_BRACKET:
                        floor_ns = observation["summary"]["per_position_ns"]
                    observations.append(observation)
                    save(out / "trials.json", observations)
        except b1.Stop as error:
            stop = str(error)
            print(f"STOPPED: {stop}", flush=True)
        except (Exception, KeyboardInterrupt) as error:
            # Any other failure (a trial that crashed or timed out, a worker
            # that died, an interrupt) also stops the run. The records so far
            # are still judged and written; nothing is analysed.
            stop = f"{type(error).__name__}: {error}"
            (out / "failure.txt").write_text(traceback.format_exc())
            print(f"STOPPED: {stop}", flush=True)
    ended = ht.utc_now()
    b2.attach_stalls(observations, watch.stalls)
    save(out / "suspensions.json", watch.stalls)
    verdicts = b2.judge_pairs(observations)
    save(out / "trials.json", observations)
    complete = stop is None and len(observations) == len(order)
    report = b2.analyse(observations, verdicts) if complete else {"note": "the run stopped; nothing is analysed"}
    host_after = lp.host_state()
    # The protocol requires a quiet machine at the end as well as the start.
    quiet_at_end = not host_after["busy_processes"]
    report.update(kind=kind, counts=kind == "admissible" and complete and quiet_at_end,
                  quiet_at_end=quiet_at_end, complete=complete, stopped=stop,
                  scheduled=len(order), observed=len(observations), telemetry_samples=sampler.samples,
                  pauses=pauses, suspensions=watch.stalls,
                  outcomes={
                      "valid_pairs": [v["pair"] for v in verdicts if v["valid"]],
                      "void_pairs": {str(v["pair"]): v["void_reasons"] for v in verdicts if not v["valid"]},
                      "not_run": [entry["label"] for entry in order[len(observations):]],
                      "worker_timing_incomplete": [o["label"] for o in observations
                                                   if o.get("worker_timing") == b1.WORKER_TIMING_INCOMPLETE],
                  },
                  host_after=host_after, exclusivity_evidence_supplied=handshake is not None,
                  ended_utc=ended)
    save(out / "analysis.json", report)
    # Last, and after every record is saved: the unified-log query can take
    # minutes, and nothing above may depend on it finishing.
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
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--diagnostic", metavar="LABEL", help="Development run; never counts")
    mode.add_argument("--admissible", action="store_true", help="The run the protocol admits")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ids", type=Path,
                        default=Path(__file__).with_name("link-results") / "control" / "ids.json")
    parser.add_argument("--frames-qualification", type=Path, action="append", default=[],
                        help="Passing block-frame qualifying receipt; --admissible needs two distinct ones")
    parser.add_argument("--peer-handshake", type=Path, help="Saved peer-session quiet acknowledgements")
    parser.add_argument("--port", type=int, default=19581)
    parser.add_argument("--continue-unsettled", action="store_true",
                        help="Diagnostic only: run the schedule even if warm-up never settles")
    args = parser.parse_args(argv)
    if args.admissible and args.continue_unsettled:
        parser.error("--continue-unsettled is for diagnostics; the admissible run stops if warm-up never settles")
    if args.admissible and b2.PROTOCOL.endswith("(draft)"):
        parser.error("BLOCK-TIME-2 is a draft; it admits no run until it is frozen")
    root = Path(__file__).resolve().parents[2]
    for binary in (b1.TRIAL, b1.SERVER, RELAY):
        if not (root / binary).exists():
            parser.error(f"build first: {binary}")
    if not sha(root / RELAY).startswith(lp.FROZEN_RELAY_SHA256_PREFIX):
        parser.error("the relay binary is not the frozen build LINK-2 certifies")
    if os.environ.get(LINK_QOS_ENV, "default") != "default":
        parser.error("the relay was qualified at the default scheduling class only")
    if args.admissible:
        if lp.busy_processes():
            parser.error("a compiler or model process is running; the admissible run needs a quiet machine")
        if args.peer_handshake is None:
            parser.error("the admissible run needs the saved peer handshake")
        try:
            bm.check_qualification([json.loads(p.read_text()) for p in args.frames_qualification],
                                   sha(root / RELAY))
        except ValueError as error:
            parser.error(str(error))
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = execute(args, root, args.model.resolve(), out, "admissible" if args.admissible else "diagnostic")
    print(json.dumps({key: report.get(key) for key in ("kind", "complete", "stopped", "valid_pairs", "void_pairs")},
                     indent=2))
    raise SystemExit(0 if report["complete"] else 1)


if __name__ == "__main__":
    main()
