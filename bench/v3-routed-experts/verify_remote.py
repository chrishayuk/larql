#!/usr/bin/env python3
"""VERIFY-REMOTE-1 driver (VERIFY-REMOTE-1.md, revision 2, draft). Nothing runs on import.

  --diagnostic LABEL   a development run; declared before it starts, never counted.
                       Adds one corrupted-proposer control per width.
  --admissible         refused while the protocol is a draft

Twenty-four units, each `bracket, four arms in a seeded order, bracket`, at
one workload and one delay. Every trial is a fresh `v3_verify_remote`
process through one relay per worker. Before the units: BLOCK-TIME's warm-up
rule on the bracket, then one sequential reference per workload, whose ids
every later trial of that workload must equal (gate 1; a mismatch stops the
run). Gates 2 and 3, the relay gate and the stall rule void a trial; unit
validity and the statements are in `verify_remote_model.py`.

Every trial marks its relay's generation boundary (revision 2): the relay
gate counts only lateness after the mark.

Relays, the worker, provenance, admission checks, telemetry, the stall watch
and the `PAUSE`/`PAUSED` hold (between units only) are BLOCK-TIME-2's.
"""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import traceback

import block_time as b1
import block_time_2 as d2
import host_telemetry as ht
import link_block_frames_model as bm
import link_pair as lp
import verify_remote_model as vr
import link_model as lm
from link_sweep import LINK_QOS_ENV, RELAY, RELAY_READY, RELAY_START_SECONDS, read_relay_stats
from profile_exact import agreement, save, sha, workers

TRIAL = "target/release/examples/v3_verify_remote"
TRIAL_SOURCES = ("crates/larql-cli/examples/v3_verify_remote.rs",
                 "crates/larql-cli/examples/support/timing.rs",
                 "crates/larql-cli/examples/support/verify.rs",
                 "crates/larql-cli/examples/support/mark.rs",
                 "crates/larql-inference/src/vindex3/speculate.rs",
                 "crates/larql-inference/src/speculation/prompt_lookup.rs",
                 "crates/larql-inference/src/speculation/greedy.rs",
                 "crates/larql-inference/src/vindex3/routed_block.rs",
                 "crates/larql-server/examples/v3_link_shaper.rs",
                 "crates/larql-server/examples/support/relay.rs",
                 "crates/larql-server/examples/support/qos.rs")
WORKLOADS_DIR = Path(__file__).with_name("verify-remote") / "workloads"
HARNESS = ("verify_remote.py", "verify_remote_model.py", "block_time_2.py", "block_time_2_model.py",
           "host_telemetry.py", "block_time.py", "block_time_model.py", "link_model.py", "link_sweep.py",
           "link_block_frames_model.py", "profile_exact.py")
TRIAL_TIMEOUT_SECONDS = 3600
BEFORE_WARM_UP = -1
# The relay prints its control listener's address after its ready line, and
# writes mark records beside its stats file.
RELAY_CONTROL = "control "
MARKS_SUFFIX = ".marks"


def workload_path(name):
    return WORKLOADS_DIR / f"{name}.json"


@contextmanager
def marked_relay(root, worker_url, point, out, label):
    """One relay for the one worker, with its control listener's address.

    `link_sweep.relays` reads only the ready line; this also reads the control
    line the relay prints next.
    """
    stats = out / f"{label}.relay-0.jsonl"
    rate = point["rate_bits_per_s"]
    process = subprocess.Popen(
        [str(root / RELAY), "127.0.0.1:0", worker_url.split("://", 1)[1], str(lm.one_way_delay_us(point)),
         lm.UNLIMITED if rate == lm.UNLIMITED else repr(float(rate)), str(stats)],
        stdout=subprocess.PIPE, text=True)
    try:
        lines = []
        reader = threading.Thread(target=lambda: lines.extend(
            [process.stdout.readline(), process.stdout.readline()]), daemon=True)
        reader.start()
        reader.join(RELAY_START_SECONDS)
        ready, control = (lines + ["", ""])[:2]
        if not ready.startswith(RELAY_READY) or not control.startswith(RELAY_CONTROL):
            raise RuntimeError(f"relay did not start: {ready!r} {control!r}")
        yield ("http://" + ready[len(RELAY_READY):].strip(), control[len(RELAY_CONTROL):].strip(),
               stats, Path(f"{stats}{MARKS_SUFFIX}"))
    finally:
        process.terminate()
        process.wait(timeout=15)


def read_marks(path):
    """The relay's mark records; none if it was never marked. They are
    written before the relay answers, so they are complete once the trial
    has its answer."""
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def run_trial(root, model, out, env, entry, worker_urls):
    """One fresh coordinator process through the relay at the entry's delay.

    Returns its record and the relay's: closing records and mark records.
    """
    raw = out / f"{entry['label']}.json"
    command = [str(root / TRIAL), "--model", str(model), "--workload", str(workload_path(entry["workload"])),
               "--arm", entry["arm"], "--generate", str(entry["generate"]),
               "--stop", ",".join(str(i) for i in vr.STOP_IDS), "--corrupt", entry["corrupt"],
               "--profile", "yes", "--out", str(raw)]
    print(f"running {entry['label']}: {entry['role']} {entry['arm']} {entry['workload']} "
          f"rtt={entry['rtt_ms']}ms corrupt={entry['corrupt']}", flush=True)
    if len(worker_urls) != 1:
        raise ValueError("VERIFY-REMOTE-1 runs one worker")
    with marked_relay(root, worker_urls[0], b1.point_of(entry), out, entry["label"]) as (url, control, stats,
                                                                                          marks):
        with (out / f"{entry['label']}.stderr").open("w") as stderr:
            subprocess.run(command + ["--shards", url, "--relay-mark", control], cwd=root, env=env,
                           stderr=stderr, check=True, timeout=TRIAL_TIMEOUT_SECONDS)
        relay = {"stats": read_relay_stats([stats]), "marks": read_marks(marks)}
    return json.loads(raw.read_text()), relay


def timed_trial(root, model, out, env, entry, worker_urls):
    started, started_monotonic = ht.utc_now(), time.monotonic()
    trial, relay_records = run_trial(root, model, out, env, entry, worker_urls)
    return trial, relay_records, {"started_utc": started, "ended_utc": ht.utc_now(),
                                  "started_monotonic": started_monotonic, "ended_monotonic": time.monotonic()}


def observe(entry, trial, relay_records, references):
    """Gate 1 (stops the run), then gates 2 and 3 and the relay gate as reasons.

    A trial that stops the run is not observed; its raw record stays in the
    run directory beside the stop message.
    """
    if trial.get("schema") != vr.TRIAL_SCHEMA:
        raise vr.Stop(f"{entry['label']}: not a {vr.TRIAL_SCHEMA} record")
    wrong = vr.record_mismatches(entry, trial, sha(workload_path(entry["workload"])))
    if wrong:
        raise vr.Stop(f"{entry['label']}: the record is not the scheduled trial: {wrong}")
    reference = references.get(entry["workload"])
    if reference is not None:
        expected = reference[:entry["generate"]]
        at = vr.first_difference(expected, trial["ids"])
        if at is not None:
            raise vr.Stop(f"{entry['label']}: ids differ from the sequential reference at position {at}")
    summary = vr.summarize(trial)
    observation = dict(entry, summary=summary, ledger_defects=vr.ledger_defects(trial),
                       frame_defects=vr.frame_defects(trial), host_load_average=os.getloadavg(), relay=None)
    if relay_records is not None:
        observation["relay"] = vr.relay_check(relay_records, trial)
    return observation


def warm_up(root, model, args, out, env, worker_urls):
    """BLOCK-TIME's warm-up rule on the bracket trial."""
    warm = []
    for index in range(vr.MAX_WARMUPS):
        entry = vr.warm_entry(index)
        trial, relay_records, _ = timed_trial(root, model, out, env, entry, worker_urls)
        warm.append(observe(entry, trial, relay_records, {})["summary"]["per_position_ns"])
        if len(warm) > 1 and agreement(warm[-2], warm[-1]) <= vr.GATE:
            break
    else:
        if not args.continue_unsettled:
            raise vr.Stop(f"warm-up never settled within {vr.GATE}: {warm}")
        print(f"warm-up never settled within {vr.GATE}; continuing as declared: {warm}", flush=True)
    save(out / "warm.json", {"per_position_ns": warm, "settled": len(warm) < vr.MAX_WARMUPS
                             or agreement(warm[-2], warm[-1]) <= vr.GATE})


def take_references(root, model, out, env, worker_urls, references, observations):
    """One sequential trial per workload at 0 ms: the ids every later trial must equal.

    Fills `references` and `observations` as it goes and saves after each,
    so a failure part way loses no reference already taken.
    """
    for workload in vr.WORKLOADS:
        entry = vr.reference_entry(workload)
        trial, relay_records, clock = timed_trial(root, model, out, env, entry, worker_urls)
        observation = observe(entry, trial, relay_records, references)
        observation.update(clock)
        observations.append(observation)
        save(out / "references_run.json", observations)
        if observation["ledger_defects"] or observation["frame_defects"]:
            raise vr.Stop(f"the {workload} reference is not usable: "
                          f"{observation['ledger_defects'] + observation['frame_defects']}")
        references[workload] = trial["ids"]
        save(out / "references.json", {"ids": references})


def execute(args, root, model, out, kind):
    receipts = [json.loads(path.read_text()) for path in args.frames_qualification]
    if kind == "admissible":
        bm.check_qualification(receipts, sha(root / RELAY))
    env = dict(os.environ, **b1.SETTINGS)
    order = vr.schedule(controls=kind == "diagnostic")
    handshake = args.peer_handshake.read_text() if args.peer_handshake else None
    started = ht.utc_now()
    manifest = {
        "schema": vr.SCHEMA, "protocol": vr.PROTOCOL, "kind": kind, "label": args.diagnostic,
        "counts": kind == "admissible", "continue_unsettled": args.continue_unsettled,
        "model": str(model), "topology": b1.TOPOLOGY,
        "provenance": b1.provenance(root, TRIAL_SOURCES),
        "trial_environment": {name: value for name, value in sorted(env.items())
                              if name.startswith(b1.ENVIRONMENT_PREFIXES)},
        "settings": b1.SETTINGS, "link_qos": os.environ.get(LINK_QOS_ENV, "default"),
        "workloads": {name: {"kind": kind_, "path": str(workload_path(name)), "sha256": sha(workload_path(name))}
                      for name, kind_ in vr.WORKLOADS.items()},
        "generate": vr.GENERATE, "stop_ids": list(vr.STOP_IDS), "prefill_width": vr.PREFILL_WIDTH,
        "bracket": {"workload": vr.BRACKET_WORKLOAD, "generate": vr.BRACKET_GENERATE,
                    "rtt_ms": vr.BRACKET_RTT_MS},
        "shuffle_seed": vr.SHUFFLE_SEED, "units": vr.unit_order(), "schedule": order,
        "thresholds": {"gate": vr.GATE, "effect": vr.EFFECT},
        "relay_gate": "relay lateness after the generation-boundary mark at most the gate share of the "
                      "timed generation; a mark defect voids the trial; the whole-trial figure is a diagnostic",
        "frames_qualification": [{"path": str(path), "sha256": sha(path)} for path in args.frames_qualification],
        "binary_sha256": {name: sha(root / path) for name, path in
                          (("v3_verify_remote", TRIAL), ("larql-server", b1.SERVER), ("v3_link_shaper", RELAY))},
        "harness_sha256": {name: sha(Path(__file__).with_name(name)) for name in HARNESS},
        "metadata_sha256": {name: sha(model / name) for name in ("index.json", "system_graph.json")},
        "host_before": lp.host_state(), "peer_handshake": handshake,
        "telemetry": {"process_interval_s": ht.PROCESS_INTERVAL_S, "state_interval_s": ht.STATE_INTERVAL_S,
                      "gates_anything": False},
        "clock": "monotonic wall time inside each trial process, from after prefill to the last generated id",
        "pause": {"file": d2.PAUSE_FILE,
                  "where": "between units and before each control only (recorded as before_pair = unit or label)",
                  "poll_s": d2.PAUSE_POLL_S},
        "stall_watch": {"tick_s": vr.b2.STALL_TICK_S, "gap_s": vr.b2.STALL_GAP_S,
                        "sees": "the driver process only", "effect": "a trial overlapping a stall voids its unit"},
        "started_utc": started,
    }
    save(out / "manifest.json", manifest)
    observations, references, references_run, stop, pauses = [], {}, [], None, []
    with ht.Sampler(out / "telemetry.jsonl") as sampler, d2.StallWatch() as watch:
        try:
            with workers(root, model, out, b1.TOPOLOGY, args.port, env) as worker_urls:
                d2.wait_while_paused(out, BEFORE_WARM_UP, pauses)
                warm_up(root, model, args, out, env, worker_urls)
                take_references(root, model, out, env, worker_urls, references, references_run)
                for entry in order:
                    if vr.is_pause_point(entry):
                        where = entry["unit"] if entry["unit"] is not None else entry["label"]
                        d2.wait_while_paused(out, where, pauses)
                        save(out / "pauses.json", pauses)
                    trial, relay_records, clock = timed_trial(root, model, out, env, entry, worker_urls)
                    observation = observe(entry, trial, relay_records, references)
                    observation.update(clock)
                    observations.append(observation)
                    save(out / "trials.json", observations)
        except vr.Stop as error:
            stop = str(error)
            print(f"STOPPED: {stop}", flush=True)
        except (Exception, KeyboardInterrupt) as error:
            # Any other failure (a crashed or timed-out trial, a dead worker,
            # an interrupt) also stops the run. Every record so far is still
            # judged and written; nothing is analysed.
            stop = f"{type(error).__name__}: {error}"
            (out / "failure.txt").write_text(traceback.format_exc())
            print(f"STOPPED: {stop}", flush=True)
    ended = ht.utc_now()
    vr.attach_stalls(observations + references_run, watch.stalls)
    save(out / "suspensions.json", watch.stalls)
    save(out / "references_run.json", references_run)
    verdicts = vr.judge_units(observations)
    save(out / "trials.json", observations)
    complete = stop is None and len(observations) == len(order)
    if complete:
        report = vr.analyse(observations, verdicts)
        controls = [o for o in observations if o["role"] == vr.ROLE_CONTROL]
        if controls:
            report["controls"] = vr.control_reading(controls, observations)
    else:
        report = {"note": "the run stopped; nothing is analysed"}
    host_after = lp.host_state()
    quiet_at_end = not host_after["busy_processes"]
    report.update(kind=kind, counts=kind == "admissible" and complete and quiet_at_end,
                  quiet_at_end=quiet_at_end, complete=complete, stopped=stop,
                  scheduled=len(order), observed=len(observations), telemetry_samples=sampler.samples,
                  pauses=pauses, suspensions=watch.stalls,
                  outcomes={
                      "valid_units": [v["unit"] for v in verdicts if v["valid"]],
                      "void_units": {str(v["unit"]): v["void_reasons"] for v in verdicts if not v["valid"]},
                      "not_run": [entry["label"] for entry in order[len(observations):]],
                  },
                  host_after=host_after, exclusivity_evidence_supplied=handshake is not None, ended_utc=ended)
    save(out / "analysis.json", report)
    # Last, after every record is saved: the unified-log query can take minutes.
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
    parser.add_argument("--frames-qualification", type=Path, action="append", default=[],
                        help="Passing block-frame qualifying receipt; --admissible needs two distinct ones")
    parser.add_argument("--peer-handshake", type=Path, help="Saved peer-session quiet acknowledgements")
    parser.add_argument("--port", type=int, default=19581)
    parser.add_argument("--continue-unsettled", action="store_true",
                        help="Diagnostic only: run the schedule even if warm-up never settles")
    args = parser.parse_args(argv)
    if args.admissible and args.continue_unsettled:
        parser.error("--continue-unsettled is for diagnostics; the admissible run stops if warm-up never settles")
    if args.admissible and vr.PROTOCOL.endswith("(draft)"):
        parser.error("VERIFY-REMOTE-1 is a draft; it admits no run until it is frozen")
    root = Path(__file__).resolve().parents[2]
    for binary in (TRIAL, b1.SERVER, RELAY):
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
    print(json.dumps({key: report.get(key) for key in ("kind", "complete", "stopped", "valid_units", "void_units")},
                     indent=2))
    raise SystemExit(0 if report["complete"] else 1)


if __name__ == "__main__":
    main()
