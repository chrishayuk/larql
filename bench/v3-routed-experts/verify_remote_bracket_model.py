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
counted per candidate and reported beside the selection, not added to it;
a stall or late relay during the filler also marks the unit disturbed.

A candidate's drift in a unit counts unless one of its own two brackets is
defective; a defect elsewhere in the unit does not void it, and an
uncounted drift is never within the gate.
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
    """Spread of the filler's time per timed id: (max - min) / mean, over its four trials.

    `None` unless all four ran and did the same sequential work: a verify
    arm that proposed anything on `explain` did different work, and its
    time would mix arm speed into the measure of disturbance.
    """
    times = [f["summary"]["per_position_ns"] for f in fillers]
    if len(times) != len(FILLER_ARMS) or any(t is None for t in times):
        return None
    if any(f["summary"].get("proposed", 0) for f in fillers):
        return None
    return (max(times) - min(times)) / statistics.fmean(times)


# Filler defects that are themselves evidence of a disturbance inside the
# unit, as opposed to a record that cannot be read.
DISTURBANCE_MARKS = ("driver suspended", "relay behind schedule")


def unit_readings(observations):
    """Per stand-in unit: each candidate's drift and whether it counts, and the filler.

    A candidate's drift counts unless one of its own two brackets has a
    defect (gates 2 and 3, a relay mark defect or late relay, a stall); a
    defect elsewhere in the unit does not void it. The filler's readable
    defects make its spread unavailable; a stall or late relay in it marks
    the unit disturbed, as does a spread over the gate.
    """
    units = {}
    for observation in observations:
        if observation.get("unit") is not None:
            units.setdefault(observation["unit"], []).append(observation)
    readings = []
    for unit, members in sorted(units.items()):
        candidates = {}
        for length in CANDIDATES:
            sides = {m["bracket"]: m for m in members
                     if m["role"] == vr.ROLE_BRACKET and m.get("candidate") == length}
            reasons = [f"{m['label']}: {r}" for m in sides.values() for r in vr.trial_defects(m)]
            drift = None
            if set(sides) == {"before", "after"}:
                drift = vr.bracket_drift(sides["before"], sides["after"])
            else:
                reasons.append("incomplete")
            candidates[str(length)] = {"drift": drift, "counted": not reasons, "reasons": reasons}
        fillers = [m for m in members if m["role"] == ROLE_FILLER]
        filler_reasons = [f"{m['label']}: {r}" for m in fillers for r in vr.trial_defects(m)]
        unreadable = any(not any(mark in r for mark in DISTURBANCE_MARKS) for r in filler_reasons)
        spread = None if unreadable else filler_spread(fillers)
        if spread is None:
            filler_reasons.append("filler incomplete, or not identical sequential work")
        disturbed = any(mark in r for r in filler_reasons for mark in DISTURBANCE_MARKS) \
            or (spread is not None and spread > GATE)
        readings.append({"unit": unit, "candidates": candidates, "filler_spread": spread,
                         "filler_reasons": filler_reasons, "disturbed": disturbed})
    return readings


def candidate_reading(readings, length):
    """One candidate across the stand-in units: its counted drifts against the rule."""
    own = [(r, r["candidates"][str(length)]) for r in readings if str(length) in r["candidates"]]
    counted = [(r, c) for r, c in own if c["counted"]]
    drifts = [c["drift"] for _, c in counted]
    within = sum(d <= GATE for d in drifts)
    missed = [r["unit"] for r, c in counted if r["disturbed"] and c["drift"] <= GATE]
    return {
        "candidate": length, "counted_units": len(counted), "drifts": drifts,
        "uncounted": {str(r["unit"]): c["reasons"] for r, c in own if not c["counted"]},
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
