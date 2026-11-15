"""BLOCK-TIME-2: schedule, pair validity and analysis (BLOCK-TIME-2.md).

Pure. Every constant is a parameter of the protocol; once it is frozen,
changing one is a new revision, not an edit. Nothing here starts a process
or reads a clock.

Per-trial gates (parity, frame counts, relay lateness), the decomposition and
the bracket drift are BLOCK-TIME-1's, imported, not restated.
"""
import math
import random
import statistics

import block_time_model as bt

PROTOCOL = "BLOCK-TIME-2 revision 1 (draft)"
SCHEMA = "larql.v3.block-time-2.v1"

ARMS = tuple(bt.ARMS)
LOW_MS, HIGH_MS = bt.DIFFERENTIAL_RTT_MS
DELAY_SPAN_NS = (HIGH_MS - LOW_MS) * bt.NS_PER_MS
# Four rounds; every arm has exactly one pair in each round, so a slow drift
# of the host is spread over every arm alike rather than over whichever arms
# a global shuffle happened to put late.
ROUNDS = 4
PAIRS_PER_ARM = ROUNDS
# An arm's slope is reported only with at least this many valid pairs.
# A reporting minimum, not a confidence level.
MINIMUM_VALID_PAIRS = 3
# Delay order by round, the same for every arm: two pairs in each order per
# arm, and A-B-B-A over the rounds, so a linear drift weighs on both orders
# alike.
LOW_FIRST, HIGH_FIRST = "5->20", "20->5"
ORDER_SEQUENCE = (LOW_FIRST, HIGH_FIRST, HIGH_FIRST, LOW_FIRST)
# Shuffles the arms within each round, one draw per round from one generator.
SHUFFLE_SEED = 20261115

GATE = bt.GATE
SLOPE_BAND = bt.SLOPE_BAND
# Within a pair, the arm's two trials must agree on coordinator time and on
# worker execution per position, each within this share of the smaller one.
# Both costs are expected not to depend on the configured round trip, so a
# larger difference is an execution-cost change inside the pair that the
# brackets either side cannot see. An engineering tolerance, not a measured
# bound on natural variation, and not evidence of external interference.
CONSISTENCY = 0.05
ROLE_BRACKET, ROLE_ARM = bt.ROLE_BRACKET, bt.ROLE_ARM

if len(ORDER_SEQUENCE) != ROUNDS:
    raise AssertionError("every round needs a delay order")


def pair_units():
    """The twenty pairs in run order: four rounds of one pair per arm, arms shuffled within each round."""
    generator = random.Random(SHUFFLE_SEED)
    ordered = []
    for round_index in range(ROUNDS):
        arms = list(ARMS)
        generator.shuffle(arms)
        for arm in arms:
            ordered.append({"arm": arm, "round": round_index, "arm_pair": round_index,
                            "order": ORDER_SEQUENCE[round_index]})
    return ordered


def schedule():
    """Every trial after warm-up, in order: bracket, two delays, bracket, per pair."""
    trials = []
    for pair, unit in enumerate(pair_units()):
        delays = (LOW_MS, HIGH_MS) if unit["order"] == LOW_FIRST else (HIGH_MS, LOW_MS)
        shape = [(ROLE_BRACKET, bt.SEQUENTIAL, 0.0, "before")] + \
                [(ROLE_ARM, unit["arm"], rtt_ms, None) for rtt_ms in delays] + \
                [(ROLE_BRACKET, bt.SEQUENTIAL, 0.0, "after")]
        for role, arm, rtt_ms, side in shape:
            trials.append({"label": f"trial-{len(trials):03d}", "pair": pair, "round": unit["round"],
                           "pair_arm": unit["arm"], "arm_pair": unit["arm_pair"], "order": unit["order"], "role": role,
                           "bracket": side, "arm": arm, "rtt_ms": rtt_ms, "profiled": True})
    return trials


def trial_defects(observation):
    """Per-trial gates other than parity, which stops the run instead."""
    reasons = list(observation["frame_defects"])
    if observation.get("relay") is not None and not observation["relay"]["relay_on_schedule"]:
        reasons.append("relay behind schedule")
    for suspension in observation.get("suspensions", []):
        reasons.append(f"driver suspended {suspension['seconds']:.1f} s during the trial")
    return reasons


def relative_difference(a, b):
    """|a - b| over the smaller; None unless both are positive and finite."""
    if a is None or b is None or not (math.isfinite(a) and math.isfinite(b)) or min(a, b) <= 0:
        return None
    return abs(a - b) / min(a, b)


def coordinator_per_position(observation):
    return observation["summary"]["coordinator_ns"] / bt.MEASURED_POSITIONS


def within_pair(arms):
    """Coordinator and worker-execution differences between a pair's two arm trials, and reasons to void."""
    coordinator = relative_difference(*(coordinator_per_position(m) for m in arms))
    execution = relative_difference(*(execution_per_position(m) for m in arms))
    reasons = []
    for name, value in (("coordinator", coordinator), ("worker execution", execution)):
        if value is None:
            reasons.append(f"{name} time missing, zero or not finite in an arm trial")
        elif value > CONSISTENCY:
            reasons.append(f"{name} time differs by {value:.4f} between the pair's delays")
    return {"coordinator": coordinator, "execution": execution}, reasons


def judge_pairs(observations):
    """Mark every pair valid or void, with reasons; every trial inherits its pair's verdict."""
    pairs = {}
    for observation in observations:
        pairs.setdefault(observation["pair"], []).append(observation)
    verdicts = []
    for pair, members in sorted(pairs.items()):
        reasons = []
        for member in members:
            reasons += [f"{member['label']}: {reason}" for reason in trial_defects(member)]
        brackets = {m["bracket"]: m for m in members if m["role"] == ROLE_BRACKET}
        arms = [m for m in members if m["role"] == ROLE_ARM]
        drift, differences = None, {"coordinator": None, "execution": None}
        if set(brackets) != {"before", "after"} or len(arms) != 2:
            reasons.append("incomplete pair")
        else:
            drift = bt.bracket_drift(brackets["before"]["summary"], brackets["after"]["summary"])
            if drift > GATE:
                reasons.append(f"brackets differ by {drift:.4f}")
            differences, inconsistent = within_pair(arms)
            reasons += inconsistent
        for member in members:
            member["pair_valid"] = not reasons
            member["valid"] = not reasons
            member["void_reasons"] = trial_defects(member)
        verdicts.append({"pair": pair, "round": members[0]["round"], "arm": members[0]["pair_arm"],
                         "arm_pair": members[0]["arm_pair"],
                         "order": members[0]["order"], "bracket_drift": drift,
                         "within_pair_difference": differences,
                         "valid": not reasons, "void_reasons": reasons})
    return verdicts


def pair_slope(members):
    """Round trips per evaluated position between the pair's two delays."""
    by_delay = {m["rtt_ms"]: m["summary"]["per_position_ns"] for m in members if m["role"] == ROLE_ARM}
    if set(by_delay) != {LOW_MS, HIGH_MS}:
        return None, None, None
    low, high = by_delay[LOW_MS], by_delay[HIGH_MS]
    return (high - low) / DELAY_SPAN_NS, low, high


def execution_per_position(observation):
    worker = observation["summary"].get("worker")
    return None if worker is None else worker["execute_ns"] / bt.MEASURED_POSITIONS


def spread(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    return {"min": values[0], "median": statistics.median(values), "max": values[-1], "values": values}


def within_pair_failures(pairs):
    """Pairs whose coordinator or worker-execution difference exceeded the consistency gate."""
    def failed(pair, name):
        value = pair["within_pair_difference"][name]
        return value is not None and value > CONSISTENCY
    return {"coordinator": sum(failed(p, "coordinator") for p in pairs),
            "execution": sum(failed(p, "execution") for p in pairs), "of": len(pairs)}


def analyse(observations, verdicts):
    """The pre-registered per-pair and per-arm statements. Void pairs are never fitted or pooled."""
    members = {}
    for observation in observations:
        members.setdefault(observation["pair"], []).append(observation)
    pairs = []
    for verdict in verdicts:
        slope, low, high = pair_slope(members[verdict["pair"]])
        pairs.append(dict(verdict, slope=slope, low_ns=low, high_ns=high))

    arms = {}
    for arm in ARMS:
        predicted = bt.exchanges_per_position(arm)
        mine = [p for p in pairs if p["arm"] == arm]
        valid = [p for p in mine if p["valid"] and p["slope"] is not None]
        entry = {"predicted": predicted, "pairs_scheduled": len(mine), "pairs_valid": len(valid),
                 "valid_slopes": [p["slope"] for p in valid],
                 # How often the within-pair gate fired for this arm: reported,
                 # so a gate that keeps rejecting one arm is visible as a finding.
                 "within_pair_failures": within_pair_failures(mine)}
        if len(valid) >= MINIMUM_VALID_PAIRS:
            mean = statistics.fmean(entry["valid_slopes"])
            entry.update(slope=mean, ratio=mean / predicted,
                         follows_exchange_model=abs(mean / predicted - 1) <= SLOPE_BAND, note=None)
        else:
            entry.update(slope=None, ratio=None, follows_exchange_model=None,
                         note=f"{len(valid)} of {len(mine)} pairs valid; {MINIMUM_VALID_PAIRS} required")
        by_order = {}
        for order in (LOW_FIRST, HIGH_FIRST):
            slopes = [p["slope"] for p in valid if p["order"] == order]
            by_order[order] = statistics.fmean(slopes) if slopes else None
        entry["order"] = dict(by_order, difference=None if None in by_order.values()
                              else by_order[LOW_FIRST] - by_order[HIGH_FIRST])
        entry["times_ns"] = {str(rtt): statistics.fmean([p["low_ns" if rtt == LOW_MS else "high_ns"]
                                                         for p in valid]) if valid else None
                             for rtt in (LOW_MS, HIGH_MS)}
        trials = [o for o in observations if o["role"] == ROLE_ARM and o["arm"] == arm]
        entry["worker_execution_ns"] = {
            str(rtt): {"valid_pairs": spread(execution_per_position(o) for o in trials
                                             if o["rtt_ms"] == rtt and o["pair_valid"]),
                       "all_pairs": spread(execution_per_position(o) for o in trials if o["rtt_ms"] == rtt)}
            for rtt in (LOW_MS, HIGH_MS)}
        arms[arm] = entry

    return {
        "protocol": PROTOCOL, "pairs": pairs, "arms": arms,
        "within_pair_failures_by_order": {order: within_pair_failures([p for p in pairs if p["order"] == order])
                                          for order in (LOW_FIRST, HIGH_FIRST)},
        "valid_pairs": sum(p["valid"] for p in pairs), "void_pairs": sum(not p["valid"] for p in pairs),
        "thresholds": {"gate": GATE, "slope_band": SLOPE_BAND, "minimum_valid_pairs": MINIMUM_VALID_PAIRS,
                       "within_pair_consistency": CONSISTENCY},
        "reading": "evaluated positions only; the three-of-four rule is a reporting minimum and the band "
                   "a model-fit criterion, neither is statistical significance; no speed comparison is made",
    }
