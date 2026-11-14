"""BLOCK-TIME-1: schedule, gates and analysis (BLOCK-TIME-1.md).

Pure. Every constant is a parameter of the protocol; once it is frozen,
changing one is a new revision, not an edit. Nothing here starts a process
or reads a clock.
"""
import statistics

import link_model as lm
from profile_exact import agreement

PROTOCOL = "BLOCK-TIME-1 draft"
SCHEMA = "larql.v3.block-time.v1"

# GPT-OSS 20B, one worker owning every routed layer and expert.
ROUTED_LAYERS = 24
SEQUENTIAL = "sequential"
# Arm name to block width; `sequential` runs one position per step.
ARMS = {SEQUENTIAL: None, "block-1": 1, "block-2": 2, "block-4": 4, "block-8": 8}
# The arm with sequential's exchange count on the block route: a control.
CONTROL_ARM = "block-1"
RTT_MS = (0.0, 1.0, 5.0, 20.0)
REPEATS = 2
# The supplied stream and the region of it that is timed.
POSITIONS = 216
MEASURE_FROM = 160
MEASURED_POSITIONS = POSITIONS - MEASURE_FROM
MAX_WARMUPS = 4
NS_PER_MS = 1_000_000

# Brackets and relay lateness: LINK-1's gate.
GATE = lm.GATE
# "Follows the exchange model": fitted slope within this share of predicted.
SLOPE_BAND = 0.10
# "Faster": a minimum effect in every repeat. Not a significance level.
MINIMUM_EFFECT = 0.01

ROLE_BRACKET, ROLE_ARM, ROLE_UNPROFILED, ROLE_DIRECT = "bracket", "arm", "unprofiled", "direct"


def exchanges_per_position(arm):
    """Dependent exchanges per evaluated position, with one worker."""
    return ROUTED_LAYERS / (ARMS[arm] or 1)


def schedule():
    """Every trial after warm-up, in order.

    Per repeat: a no-relay control, then each delay bracketed by sequential
    0 ms trials. Delays run in order, reversed in the second repeat. At each
    delay the arms run in an order rotated by one per delay, reversed in the
    second repeat. At 0 ms each arm also runs once unprofiled, so the cost
    of asking the worker for its timing is measured and not assumed.
    """
    trials = []

    def add(role, arm, rtt_ms, repeat, profiled):
        trials.append({"label": f"trial-{len(trials):03d}", "role": role, "arm": arm, "rtt_ms": rtt_ms,
                       "repeat": repeat, "profiled": profiled})

    arms = list(ARMS)
    for repeat in range(REPEATS):
        delays = list(RTT_MS) if repeat % 2 == 0 else list(reversed(RTT_MS))
        add(ROLE_DIRECT, SEQUENTIAL, 0.0, repeat, True)
        add(ROLE_BRACKET, SEQUENTIAL, 0.0, repeat, True)
        for index, rtt_ms in enumerate(delays):
            order = arms[index % len(arms):] + arms[:index % len(arms)]
            if repeat % 2:
                order.reverse()
            for arm in order:
                add(ROLE_ARM, arm, rtt_ms, repeat, True)
            if rtt_ms == 0.0:
                for arm in order:
                    add(ROLE_UNPROFILED, arm, rtt_ms, repeat, False)
            add(ROLE_BRACKET, SEQUENTIAL, 0.0, repeat, True)
    return trials


def control_alignment(prompt_tokens, stream, argmax):
    """Positions where a run's greedy choice is not the saved control's next id.

    The decoder's convention is that the logits at position `i` are the
    distribution over the token at position `i + 1`. The first generated id
    is therefore predicted at the last prompt position, and the last
    position's logits predict a token beyond the stream and are not compared.
    """
    if len(argmax) != len(stream):
        raise ValueError("one argmax per stream position is required")
    if not 0 < prompt_tokens < len(stream):
        raise ValueError("the stream must hold a prompt and at least one generated id")
    return [position for position in range(prompt_tokens - 1, len(stream) - 1)
            if argmax[position] != stream[position + 1]]


def reference_defects(prompt_tokens, stream, trial):
    """Why a trial cannot serve as the logits reference; empty if it can."""
    defects = []
    if trial["positions"] != len(stream) or len(trial["logits_sha256"]) != len(stream):
        defects.append("the reference does not cover the supplied stream")
        return defects
    wrong = control_alignment(prompt_tokens, stream, trial["argmax"])
    if wrong:
        defects.append(f"greedy ids leave the saved control at positions {wrong[:8]}")
    return defects


def parity_mismatches(reference_sha256, trial):
    """Positions whose logits are not bit-identical to the reference."""
    got = trial["logits_sha256"]
    if len(got) != len(reference_sha256):
        return list(range(max(len(got), len(reference_sha256))))
    return [i for i, (a, b) in enumerate(zip(reference_sha256, got)) if a != b]


def frame_defects(trial):
    """The measured region's frames against what the arm must send."""
    width = ARMS[trial["arm"]]
    frames = trial["frames"]["measured"]
    expected_single = ROUTED_LAYERS * MEASURED_POSITIONS if width is None else 0
    expected_multi = 0 if width is None else ROUTED_LAYERS * MEASURED_POSITIONS // width
    defects = []
    if frames["single-row"] != expected_single:
        defects.append(f"{frames['single-row']} single-row frames, expected {expected_single}")
    if frames["multi-row"] != expected_multi:
        defects.append(f"{frames['multi-row']} multi-row frames, expected {expected_multi}")
    if len(trial["exchanges"]) != expected_single + expected_multi:
        defects.append("the exchange records do not match the frame counts")
    return defects


def summarize(trial, rtt_ms):
    """Where one trial's measured wall time went. Sums over the measured region.

    One worker only: its exchanges are sequential, so their round trips sum
    to coordinator waiting time. With several workers a layer's exchanges
    overlap and these sums would count concurrent time more than once; the
    critical path there is the slowest worker per layer, not the total.
    """
    exchanges = trial["exchanges"]
    exchange_ns = sum(e["roundtrip_ns"] for e in exchanges)
    configured_ns = len(exchanges) * rtt_ms * NS_PER_MS
    summary = {
        "per_position_ns": trial["wall_ns"] / MEASURED_POSITIONS,
        "wall_ns": trial["wall_ns"],
        "exchanges": len(exchanges),
        "exchange_ns": exchange_ns,
        # Attention, routing, reduction, the head and session bookkeeping.
        "coordinator_ns": trial["wall_ns"] - exchange_ns,
        "configured_delay_ns": configured_ns,
        "request_bytes": sum(e["request_bytes"] for e in exchanges),
        "response_bytes": sum(e["response_bytes"] for e in exchanges),
        "worker": None,
    }
    records = trial.get("worker_records", [])
    workers = [r.get("worker") for r in records]
    if trial.get("profiled") and len(records) == len(exchanges) and all(workers):
        handler_ns = sum(w["handler_ns"] for w in workers)
        summary["worker"] = {
            "handler_ns": handler_ns,
            "execute_ns": sum(w["execute_ns"] for w in workers),
            "experts_ns": sum(w["experts_ns"] for w in workers),
            "queue_ns": sum(w["queue_ns"] for w in workers),
            # What is left of the exchanges after the configured delay and
            # the worker's handler: serialisation, HTTP, scheduling, loopback
            # and relay processing together. Not network latency alone.
            "unattributed_exchange_ns": exchange_ns - configured_ns - handler_ns,
        }
    return summary


def bracket_drift(before, after):
    return agreement(before["per_position_ns"], after["per_position_ns"])


def fit_slope(points):
    """Least-squares line of per-position time on configured round trip.

    `points` are (round trip ns, per-position ns). The slope is in round
    trips per position, which the exchange model predicts as the number of
    dependent exchanges per position.
    """
    if len(points) < 2:
        raise ValueError("a slope needs at least two delays")
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
    spread = sum((x - mean_x) ** 2 for x in xs)
    if spread == 0:
        raise ValueError("a slope needs distinct delays")
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / spread
    return slope, mean_y - slope * mean_x


def analyse(observations):
    """The pre-registered table, slopes and statements.

    Each observation is a scheduled trial with `valid` and `summary`. Void
    trials are never fitted or pooled; a cell with no valid trial is missing.
    """
    def per_position(role, arm, rtt_ms, repeat):
        found = [o["summary"]["per_position_ns"] for o in observations
                 if o["valid"] and o["role"] == role and o["arm"] == arm
                 and o["rtt_ms"] == rtt_ms and o["repeat"] == repeat]
        return found[0] if found else None

    cells = {}
    for arm in ARMS:
        for rtt_ms in RTT_MS:
            repeats = [per_position(ROLE_ARM, arm, rtt_ms, repeat) for repeat in range(REPEATS)]
            valid = [value for value in repeats if value is not None]
            cells[(arm, rtt_ms)] = {"repeats_ns": repeats,
                                    "mean_ns": statistics.fmean(valid) if valid else None}

    slopes = {}
    for arm in ARMS:
        points = [(rtt_ms * NS_PER_MS, cells[(arm, rtt_ms)]["mean_ns"]) for rtt_ms in RTT_MS]
        predicted = exchanges_per_position(arm)
        if any(value is None for _, value in points):
            slopes[arm] = {"predicted": predicted, "fitted": None, "ratio": None, "intercept_ns": None,
                           "follows_exchange_model": None, "note": "a delay has no valid trial"}
            continue
        slope, intercept = fit_slope(points)
        slopes[arm] = {"predicted": predicted, "fitted": slope, "ratio": slope / predicted,
                       "intercept_ns": intercept,
                       "follows_exchange_model": abs(slope / predicted - 1) <= SLOPE_BAND}

    faster = []
    for arm in ARMS:
        if arm in (SEQUENTIAL, CONTROL_ARM):
            continue
        for rtt_ms in RTT_MS:
            block = cells[(arm, rtt_ms)]["repeats_ns"]
            base = cells[(SEQUENTIAL, rtt_ms)]["repeats_ns"]
            if None in block or None in base:
                reductions, supported = None, None
            else:
                reductions = [(s - b) / s for s, b in zip(base, block)]
                supported = all(reduction > MINIMUM_EFFECT for reduction in reductions)
            faster.append({"arm": arm, "rtt_ms": rtt_ms, "reduction_per_repeat": reductions,
                           "faster_in_every_repeat": supported})

    overhead = []
    for arm in ARMS:
        for repeat in range(REPEATS):
            profiled = per_position(ROLE_ARM, arm, 0.0, repeat)
            plain = per_position(ROLE_UNPROFILED, arm, 0.0, repeat)
            overhead.append({"arm": arm, "repeat": repeat, "profiled_ns": profiled, "unprofiled_ns": plain,
                             "relative": None if None in (profiled, plain) else (profiled - plain) / plain})

    return {
        "cells": [{"arm": arm, "rtt_ms": rtt_ms, **cell} for (arm, rtt_ms), cell in cells.items()],
        "slopes": slopes, "faster": faster, "profiling_overhead": overhead,
        "valid": sum(o["valid"] for o in observations),
        "void": sum(not o["valid"] for o in observations),
        "thresholds": {"slope_band": SLOPE_BAND, "minimum_effect": MINIMUM_EFFECT, "gate": GATE},
        "reading": "evaluated positions only; the thresholds are a model-fit band and a minimum "
                   "effect across two repeats, not statistical significance",
    }
