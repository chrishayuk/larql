#!/usr/bin/env python3
"""BLOCK-TIME-1 driver (BLOCK-TIME-1.md). Nothing runs on import.

  --diagnostic LABEL   a development run; declared before it starts, never counted
  --admissible         the one run the protocol admits; needs two passing
                       block-frame qualification receipts and a quiet machine

The worker starts once and stays resident. Every trial is a fresh
`v3_block_time` process running one arm over the supplied ids, through one
relay per worker. The driver applies the gates and the analysis of
`block_time_model.py`; it excludes nothing and replaces no void trial.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import block_time_model as bt
import link_block_frames_model as bm
import link_model as lm
import link_pair as lp
from link_sweep import LINK_QOS_ENV, RELAY, read_relay_stats, relays
from profile_exact import agreement, save, sha, workers

TRIAL = "target/release/examples/v3_block_time"
SERVER = "target/release/larql-server"
TOPOLOGY = "one"
TRIAL_TIMEOUT_SECONDS = 3600
# LINK-1's settings, recorded in the manifest.
SETTINGS = dict(LARQL_CPU_WORKERS="8", RAYON_NUM_THREADS="8",
                VECLIB_MAXIMUM_THREADS="1", TOKIO_WORKER_THREADS="2", LARQL_KV_ENGINE="")


class Stop(Exception):
    """A gate that ends the run: nothing after it is timed."""


def point_of(entry):
    return {"kind": "floor" if entry["rtt_ms"] == 0.0 else "axis",
            "rtt_ms": entry["rtt_ms"], "rate_bits_per_s": lm.UNLIMITED}


def run_trial(root, model, ids, out, env, entry, worker_urls):
    """One fresh coordinator process; returns its record and the relay's."""
    raw = out / f"{entry['label']}.json"

    def execute(urls):
        command = [str(root / TRIAL), "--model", str(model), "--shards", ",".join(urls), "--ids", str(ids),
                   "--arm", entry["arm"], "--measure-from", str(bt.MEASURE_FROM),
                   "--profile", "yes" if entry["profiled"] else "no", "--out", str(raw)]
        with (out / f"{entry['label']}.stderr").open("w") as stderr:
            subprocess.run(command, cwd=root, env=env, stderr=stderr, check=True, timeout=TRIAL_TIMEOUT_SECONDS)

    print(f"running {entry['label']}: {entry['role']} {entry['arm']} rtt={entry['rtt_ms']}ms "
          f"{'profiled' if entry['profiled'] else 'unprofiled'}", flush=True)
    if entry["role"] == bt.ROLE_DIRECT:
        execute(worker_urls)
        return json.loads(raw.read_text()), None
    with relays(root, worker_urls, point_of(entry), out, entry["label"]) as (urls, stats):
        execute(urls)
        records = read_relay_stats(stats)
    return json.loads(raw.read_text()), records


def observe(entry, trial, relay_records, reference, floor_ns):
    """Apply the per-trial gates. A parity failure stops the run."""
    wrong = bt.parity_mismatches(reference, trial)
    if wrong:
        raise Stop(f"{entry['label']}: logits differ from the reference at positions {wrong[:8]}")
    summary = bt.summarize(trial, entry["rtt_ms"])
    observation = dict(entry, summary=summary, frame_defects=bt.frame_defects(trial),
                       host_load_average=os.getloadavg(), relay=None)
    if relay_records is not None:
        # The relay carried the untimed positions too.
        observation["relay"] = lm.relay_gate(relay_records, bt.POSITIONS, floor_ns or summary["per_position_ns"])
    return observation


def judge(observations):
    """Bracket every arm trial and mark each observation valid or void, with reasons."""
    brackets = [i for i, o in enumerate(observations) if o["role"] == bt.ROLE_BRACKET]
    for index, observation in enumerate(observations):
        reasons = list(observation["frame_defects"])
        if observation["relay"] is not None and not observation["relay"]["relay_on_schedule"]:
            reasons.append("relay behind schedule")
        if observation["role"] in (bt.ROLE_ARM, bt.ROLE_UNPROFILED):
            before = max((b for b in brackets if b < index), default=None)
            after = min((b for b in brackets if b > index), default=None)
            if before is None or after is None:
                reasons.append("not bracketed")
            else:
                drift = bt.bracket_drift(observations[before]["summary"], observations[after]["summary"])
                observation["bracket_drift"] = drift
                if drift > bt.GATE:
                    reasons.append(f"brackets differ by {drift:.4f}")
        observation["void_reasons"] = reasons
        observation["valid"] = not reasons
    return observations


def execute(args, root, model, out, kind):
    relay_sha = sha(root / RELAY)
    receipts = [json.loads(path.read_text()) for path in args.frames_qualification]
    if kind == "admissible":
        bm.check_qualification(receipts, relay_sha)
    saved = json.loads(args.ids.read_text())
    stream = saved["prompt"] + saved["generated"]
    if len(stream) != bt.POSITIONS:
        raise ValueError(f"the supplied stream has {len(stream)} ids, not {bt.POSITIONS}")
    env = dict(os.environ, **SETTINGS)
    order = bt.schedule()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).splitlines()
    handshake = args.peer_handshake.read_text() if args.peer_handshake else None
    manifest = {
        "schema": bt.SCHEMA, "protocol": bt.PROTOCOL, "kind": kind, "label": args.diagnostic,
        "counts": kind == "admissible",
        "continue_unsettled": args.continue_unsettled,
        "revision": revision, "source_dirty": dirty, "model": str(model), "topology": TOPOLOGY,
        "settings": SETTINGS, "link_qos": os.environ.get(LINK_QOS_ENV, "default"),
        "ids": str(args.ids), "ids_sha256": sha(args.ids), "prompt_tokens": len(saved["prompt"]),
        "positions": bt.POSITIONS, "measure_from": bt.MEASURE_FROM,
        "schedule": order, "thresholds": {"gate": bt.GATE, "slope_band": bt.SLOPE_BAND,
                                          "minimum_effect": bt.MINIMUM_EFFECT},
        "frames_qualification": [{"path": str(path), "sha256": sha(path)} for path in args.frames_qualification],
        "binary_sha256": {name: sha(root / path) for name, path in
                          (("v3_block_time", TRIAL), ("larql-server", SERVER), ("v3_link_shaper", RELAY))},
        "harness_sha256": {name: sha(Path(__file__).with_name(name)) for name in
                           ("block_time.py", "block_time_model.py", "link_model.py", "link_sweep.py",
                            "link_block_frames_model.py", "profile_exact.py")},
        "metadata_sha256": {name: sha(model / name) for name in ("index.json", "system_graph.json")},
        "host_before": lp.host_state(), "peer_handshake": handshake,
        "clock": "monotonic wall time around the measured region inside each trial process",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    save(out / "manifest.json", manifest)
    observations, stop = [], None
    try:
        with workers(root, model, out, TOPOLOGY, args.port, env) as worker_urls:
            warm, reference = [], None
            for index in range(bt.MAX_WARMUPS):
                entry = {"label": f"warm-{index}", "role": "warm", "arm": bt.SEQUENTIAL, "rtt_ms": 0.0,
                         "repeat": -1, "profiled": True}
                trial, relay_records = run_trial(root, model, args.ids, out, env, entry, worker_urls)
                if reference is None:
                    defects = bt.reference_defects(len(saved["prompt"]), stream, trial)
                    if defects:
                        raise Stop(f"the reference trial is not usable: {defects}")
                    reference = trial["logits_sha256"]
                    save(out / "reference.json", {"logits_sha256": reference, "argmax": trial["argmax"]})
                warm.append(observe(entry, trial, relay_records, reference, None)["summary"]["per_position_ns"])
                if len(warm) > 1 and agreement(warm[-2], warm[-1]) <= bt.GATE:
                    break
            else:
                # The protocol's rule is to stop. A diagnostic may be told to
                # go on, to exercise the rest of the driver on a busy machine.
                if not args.continue_unsettled:
                    raise Stop(f"warm-up never settled within {bt.GATE}: {warm}")
                print(f"warm-up never settled within {bt.GATE}; continuing as declared: {warm}", flush=True)
            save(out / "warm.json", {"per_position_ns": warm, "settled": len(warm) < bt.MAX_WARMUPS
                                     or agreement(warm[-2], warm[-1]) <= bt.GATE})
            floor_ns = warm[-1]
            for entry in order:
                trial, relay_records = run_trial(root, model, args.ids, out, env, entry, worker_urls)
                observation = observe(entry, trial, relay_records, reference, floor_ns)
                if entry["role"] == bt.ROLE_BRACKET:
                    floor_ns = observation["summary"]["per_position_ns"]
                observations.append(observation)
                save(out / "trials.json", observations)
    except Stop as error:
        stop = str(error)
        print(f"STOPPED: {stop}", flush=True)
    judged = judge(observations)
    save(out / "trials.json", judged)
    complete = stop is None and len(judged) == len(order)
    report = bt.analyse(judged) if complete else {"note": "the run stopped; nothing is analysed"}
    report.update(kind=kind, counts=kind == "admissible" and complete, complete=complete, stopped=stop,
                  host_after=lp.host_state(), exclusivity_evidence_supplied=handshake is not None)
    save(out / "analysis.json", report)
    save(out / "verification.json",
         {"complete": complete, "sha256": {str(p.relative_to(out)): sha(p)
                                           for p in sorted(out.rglob("*")) if p.is_file()}})
    return report


def main():
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
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    for binary in (TRIAL, SERVER, RELAY):
        if not (root / binary).exists():
            parser.error(f"build first: {binary}")
    if not sha(root / RELAY).startswith(lp.FROZEN_RELAY_SHA256_PREFIX):
        parser.error("the relay binary is not the frozen build LINK-2 certifies")
    if os.environ.get(LINK_QOS_ENV, "default") != "default":
        parser.error("the relay was qualified at the default scheduling class only")
    if args.admissible and args.continue_unsettled:
        parser.error("--continue-unsettled is for diagnostics; the admissible run stops if warm-up never settles")
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
    print(json.dumps({key: report.get(key) for key in ("kind", "complete", "stopped", "valid", "void")}, indent=2))
    raise SystemExit(0 if report["complete"] else 1)


if __name__ == "__main__":
    main()
