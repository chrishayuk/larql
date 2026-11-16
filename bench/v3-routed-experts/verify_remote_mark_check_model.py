"""VERIFY-REMOTE-1 revision 2: the relay mark's functional check on the real model (VERIFY-REMOTE-1.md).

Pure. A declared functional check, never timed as a result: one trial per
arm on `json_edit` at 20 ms, through the relay with the generation-boundary
mark. The mechanism qualifies if, in every trial:

- the mark was answered `ok` and the relay gate found no mark defect;
- in both directions, the relay bytes before the mark cover every prefill
  exchange's frame and the bytes after it every generation exchange's
  frame (a generation frame counted before the mark would leave the
  generation phase short);
- the generation lateness plus the snapshot equals the closing total;
- the ids equal the `sequential` trial's (gate 1).

Whether a trial's relay was on schedule is reported, not required: the
host is not quiet, and timing is not what this checks. Transport overhead
per exchange (relay bytes beyond the frames) is reported per phase but is
not required to agree: the bytes before the mark also carry the
coordinator's setup requests, which are not exchanges, and a multi-row
response's headers are longer than a single row's (the stop check's
records show both).
"""
import verify_remote_model as vr

SCHEMA = "larql.v3.verify-remote-mark-check.v1"
WORKLOAD = "json_edit"
RTT_MS = 20.0
DIRECTIONS = {"request": "request_bytes", "response": "response_bytes"}


def entries():
    """`sequential` first (the ids every arm must equal), then each verify arm."""
    return [{"label": f"mark-{arm}", "unit": None, "role": "mark-check", "bracket": None, "arm": arm,
             "workload": WORKLOAD, "generate": vr.GENERATE, "rtt_ms": RTT_MS, "corrupt": "none"}
            for arm in vr.ARMS]


def phase_bytes(trial, relay):
    """Per direction: relay bytes and frame bytes before and after the mark, and exchanges per phase."""
    summary = next(m for m in relay["marks"] if m.get("summary"))
    snapshot = next(m for m in relay["marks"] if not m.get("summary") and m["mark"] == summary["mark"])
    split = trial["generation_from"]["exchange"]
    phases = {"prefill": trial["exchanges"][:split], "generation": trial["exchanges"][split:]}
    out = {}
    for direction, field in DIRECTIONS.items():
        closing = next(r for r in relay["stats"] if r["connection"] == snapshot["connection"]
                       and r["direction"] == direction)
        relay_bytes = {"prefill": snapshot[direction]["bytes"],
                       "generation": closing["bytes"] - snapshot[direction]["bytes"]}
        frames = {phase: sum(e[field] for e in exchanges) for phase, exchanges in phases.items()}
        overhead = {phase: (relay_bytes[phase] - frames[phase]) / len(phases[phase])
                    for phase in phases if phases[phase]}
        out[direction] = {"relay_bytes": relay_bytes, "frame_bytes": frames, "overhead_per_exchange": overhead,
                          "exchanges": {phase: len(e) for phase, e in phases.items()}}
    return out


def failures(trial, relay, reference_ids):
    """Every requirement this trial does not meet."""
    found = []
    gate = vr.relay_check(relay, trial)
    found += [f"mark: {d}" for d in gate["mark_defects"]]
    if trial["ids"] != reference_ids:
        found.append(f"ids differ from sequential at position {vr.first_difference(reference_ids, trial['ids'])}")
    if gate["mark_defects"]:
        return found
    for direction, reading in phase_bytes(trial, relay).items():
        for phase in ("prefill", "generation"):
            if reading["relay_bytes"][phase] < reading["frame_bytes"][phase]:
                found.append(f"{direction}: the relay carried fewer {phase} bytes than its frames")
    summary = next(m for m in relay["marks"] if m.get("summary"))
    snapshot = next(m for m in relay["marks"] if not m.get("summary") and m["mark"] == summary["mark"])
    closing = sum(r["late_sum_ns"] for r in relay["stats"] if r["connection"] == snapshot["connection"])
    before = sum(snapshot[d]["late_sum_ns"] for d in DIRECTIONS)
    if before + gate["generation_late_sum_ns"] != closing:
        found.append("snapshot plus generation lateness is not the closing total")
    return found


def reading(results):
    """`results`: (arm, trial, relay) in schedule order, `sequential` first."""
    reference_ids = results[0][1]["ids"]
    arms = []
    for arm, trial, relay in results:
        gate = vr.relay_check(relay, trial)
        arms.append({"arm": arm, "answer": (trial.get("relay_mark") or {}).get("answer"),
                     "waited_ns": (trial.get("relay_mark") or {}).get("waited_ns"),
                     "failures": failures(trial, relay, reference_ids),
                     "phases": None if gate["mark_defects"] else phase_bytes(trial, relay),
                     "generation_late_share_of_wall": gate["relay_late_share_of_wall"],
                     "whole_trial_late_share_of_wall": gate["whole_trial"]["share_of_wall"],
                     "on_schedule_in_generation": gate["relay_on_schedule"]})
    return {"arms": arms, "passes": len(arms) == len(vr.ARMS) and all(not a["failures"] for a in arms),
            "timed_result": False}
