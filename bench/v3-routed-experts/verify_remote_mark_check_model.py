"""VERIFY-REMOTE-1 revision 2: the relay mark's functional check on the real model (VERIFY-REMOTE-1.md).

Pure. A declared functional check, never timed as a result: one trial per
arm on `json_edit` at 20 ms, through the relay with the generation-boundary
mark. The mechanism qualifies if, in every trial:

- the mark was answered `ok` and the relay gate found no mark defect;
- the relay carried exactly one connection, the one the mark snapshotted;
- **the split falls on the exchange boundary**: in each direction, the
  relay bytes after the mark equal the generation exchanges' frames plus
  their HTTP overhead, predicted per route from constants measured in the
  stop check's records (`20261116-01-stop-check`, fixed before this check
  ran), to within `RESIDUAL_TOLERANCE_BYTES`. That tolerance is under half
  the smallest frame, so a whole exchange on the wrong side of the mark,
  late (a generation frame missing) or early (a prefill frame added),
  always shows. The bytes before the mark also carry the coordinator's
  setup requests, which are not exchanges, so only the generation side is
  predicted;
- the ids equal the `sequential` trial's (gate 1; the driver also stops on
  a mismatch).

Whether a trial's relay was on schedule is reported, not required: the
host is not quiet, and timing is not what this checks.
"""
import verify_remote_model as vr

SCHEMA = "larql.v3.verify-remote-mark-check.v1"
WORKLOAD = "json_edit"
RTT_MS = 20.0
DIRECTIONS = {"request": "request_bytes", "response": "response_bytes"}
# HTTP bytes per exchange beyond its frame, by direction and route, from the
# stop check's generation phases: single-row 182 / 272.5, multi-row 184 /
# 371.7 to 376.3 (the response headers' digits vary).
OVERHEAD_BYTES = {"request": {vr.SINGLE_ROW: 182.0, vr.MULTI_ROW: 184.0},
                  "response": {vr.SINGLE_ROW: 272.5, vr.MULTI_ROW: 374.0}}
# Under half the smallest frame (a single-row request, 11,576 bytes).
RESIDUAL_TOLERANCE_BYTES = 5_000


def entries():
    """`sequential` first (the ids every arm must equal), then each verify arm."""
    return [{"label": f"mark-{arm}", "unit": None, "role": "mark-check", "bracket": None, "arm": arm,
             "workload": WORKLOAD, "generate": vr.GENERATE, "rtt_ms": RTT_MS, "corrupt": "none"}
            for arm in vr.ARMS]


def connection_faults(relay):
    """The check needs exactly one relay connection, the one the mark snapshotted."""
    summary = next(m for m in relay["marks"] if m.get("summary"))
    connections = sorted({r["connection"] for r in relay["stats"]})
    if summary["connections"] != connections or len(connections) != 1:
        return [f"relay connections {connections}, marked {summary['connections']}: one is required"]
    return []


def phase_bytes(trial, relay):
    """Per direction: relay and frame bytes before and after the mark, exchanges per phase, and the
    generation residual against the predicted HTTP overhead. Assumes one connection."""
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
        predicted = frames["generation"] + sum(OVERHEAD_BYTES[direction][e["route"]]
                                               for e in phases["generation"])
        out[direction] = {
            "relay_bytes": relay_bytes, "frame_bytes": frames,
            "exchanges": {phase: len(e) for phase, e in phases.items()},
            "overhead_per_exchange": {phase: (relay_bytes[phase] - frames[phase]) / len(phases[phase])
                                      for phase in phases if phases[phase]},
            "generation_predicted_bytes": predicted,
            "generation_residual_bytes": relay_bytes["generation"] - predicted,
        }
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
    connections = connection_faults(relay)
    found += connections
    if connections:
        return found
    for direction, reading in phase_bytes(trial, relay).items():
        residual = reading["generation_residual_bytes"]
        if abs(residual) > RESIDUAL_TOLERANCE_BYTES:
            found.append(f"{direction}: the relay carried {residual:+.0f} bytes after the mark beyond the "
                         f"generation exchanges' predicted bytes (tolerance {RESIDUAL_TOLERANCE_BYTES})")
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
                     "phases": None if gate["mark_defects"] or connection_faults(relay)
                     else phase_bytes(trial, relay),
                     "generation_late_share_of_wall": gate["relay_late_share_of_wall"],
                     "whole_trial_late_share_of_wall": gate["whole_trial"]["share_of_wall"],
                     "on_schedule_in_generation": gate["relay_on_schedule"]})
    return {"arms": arms, "passes": len(arms) == len(vr.ARMS) and all(not a["failures"] for a in arms),
            "timed_result": False}
