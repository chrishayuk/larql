"""VERIFY-REMOTE-1 revision 2: the bracket study's schedule, measures and selection rule.

Pure. A declared diagnostic that chooses the bracket length before freeze
(VERIFY-REMOTE-1.md, "Revision 2: bracket length"). It measures no arm's
speed. Eight stand-in units each open with every candidate bracket in a
seeded order, run a filler shaped like a real unit, and close with the same
brackets in the same order. Drift is the protocol's bracket drift
(`verify_remote_model.bracket_drift`), unchanged, at the unchanged 1% gate.

**Selection, fixed in advance:** a length qualifies if at least 7 of its 8
drifts are at most 1%; the bracket is the shortest qualifying length; if
none qualifies, none is selected. Missed disturbances (a filler whose four
identical-work trials spread by more than 1% while the brackets agree) are
counted per candidate and reported beside the selection, not added to it.
"""
import random
import statistics

import verify_remote_model as vr

SCHEMA = "larql.v3.verify-remote-bracket-study.v1"
CANDIDATES = (16, 32, 64)
STAND_INS = 8
QUALIFYING_WITHIN = 7
SEED = 20261117
FILLER_WORKLOAD = "explain"
FILLER_RTT_MS = 20.0
# Fixed, not shuffled: the filler's arms do identical work on `explain`
# (prompt lookup proposes nothing there), so their order carries nothing.
FILLER_ARMS = tuple(vr.ARMS)
ROLE_FILLER = "filler"
REFERENCE_WORKLOADS = (vr.BRACKET_WORKLOAD, FILLER_WORKLOAD)
GATE = vr.GATE


def bracket_orders():
    """Per stand-in unit, the order its candidate brackets open and close in."""
    generator = random.Random(SEED)
    orders = []
    for _ in range(STAND_INS):
        order = list(CANDIDATES)
        generator.shuffle(order)
        orders.append(order)
    return orders


def schedule():
    """Every trial after warm-up and the references, in order."""
    trials = []

    def label():
        return f"trial-{len(trials):03d}"

    def bracket(unit, side, length):
        entry = vr.bracket_entry(unit, side, label())
        entry.update(generate=length, candidate=length)
        return entry

    for unit, order in enumerate(bracket_orders()):
        for length in order:
            trials.append(bracket(unit, "before", length))
        for arm in FILLER_ARMS:
            trials.append({"label": label(), "unit": unit, "role": ROLE_FILLER, "bracket": None,
                           "candidate": None, "arm": arm, "workload": FILLER_WORKLOAD, "generate": vr.GENERATE,
                           "rtt_ms": FILLER_RTT_MS, "corrupt": "none"})
        for length in order:
            trials.append(bracket(unit, "after", length))
    return trials


def is_pause_point(entry):
    """Only before a stand-in unit's first trial."""
    return entry["role"] == vr.ROLE_BRACKET and entry["bracket"] == "before" \
        and entry["label"] == first_labels().get(entry["unit"])


def first_labels():
    firsts = {}
    for entry in schedule():
        firsts.setdefault(entry["unit"], entry["label"])
    return firsts


def filler_spread(fillers):
    """Spread of the filler's time per timed id: (max - min) / mean, over its four trials."""
    times = [f["summary"]["per_position_ns"] for f in fillers]
    if len(times) != len(FILLER_ARMS) or any(t is None for t in times):
        return None
    return (max(times) - min(times)) / statistics.fmean(times)


def unit_readings(observations):
    """Per stand-in unit: each candidate's drift, the filler spread, and defects by name.

    A unit with a defective trial (gates 2 and 3, a relay mark defect or a
    late relay, a stall) is reported with its reasons, and its drifts are
    still shown but not counted.
    """
    units = {}
    for observation in observations:
        if observation.get("unit") is not None:
            units.setdefault(observation["unit"], []).append(observation)
    readings = []
    for unit, members in sorted(units.items()):
        reasons = [f"{m['label']}: {r}" for m in members for r in vr.trial_defects(m)]
        drifts = {}
        for length in CANDIDATES:
            sides = {m["bracket"]: m for m in members
                     if m["role"] == vr.ROLE_BRACKET and m.get("candidate") == length}
            if set(sides) == {"before", "after"}:
                drifts[length] = vr.bracket_drift(sides["before"], sides["after"])
            else:
                reasons.append(f"candidate {length}: incomplete")
        fillers = [m for m in members if m["role"] == ROLE_FILLER]
        spread = filler_spread(fillers)
        if spread is None:
            reasons.append("incomplete filler")
        readings.append({"unit": unit, "drifts": {str(k): v for k, v in drifts.items()},
                         "filler_spread": spread, "counted": not reasons, "reasons": reasons})
    return readings


def candidate_reading(readings, length):
    counted = [r for r in readings if r["counted"]]
    drifts = [r["drifts"][str(length)] for r in counted]
    within = sum(d <= GATE for d in drifts)
    missed = [r["unit"] for r in counted
              if r["filler_spread"] > GATE and r["drifts"][str(length)] <= GATE]
    return {
        "candidate": length, "counted_units": len(counted), "drifts": drifts,
        "within_gate": within, "median": statistics.median(drifts) if drifts else None,
        "max": max(drifts) if drifts else None,
        "qualifies": within >= QUALIFYING_WITHIN,
        "missed_disturbances": missed,
    }


def select(readings):
    """The pre-registered rule. Every candidate is reported; none is selected if none qualifies.

    A unit lost to a defect has no drift, and a missing drift is never
    within the gate: a candidate still needs 7 drifts within it.
    """
    candidates = [candidate_reading(readings, length) for length in CANDIDATES]
    qualifying = [c["candidate"] for c in candidates if c["qualifies"]]
    selected = min(qualifying) if qualifying else None
    chosen = next((c for c in candidates if c["candidate"] == selected), None)
    return {
        "rule": f"shortest length with at least {QUALIFYING_WITHIN} of {STAND_INS} drifts within {GATE}",
        "candidates": candidates, "selected": selected,
        "outcome": ("no candidate qualifies; no length is selected and the protocol is not frozen"
                    if selected is None else f"{selected} ids"),
        "selected_missed_disturbances": None if chosen is None else chosen["missed_disturbances"],
        "review_needed": selected is None or bool(chosen["missed_disturbances"]),
    }
