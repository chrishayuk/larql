"""VERIFY-REMOTE-1: schedule, gates, unit validity and statements (VERIFY-REMOTE-1.md).

Pure. Every constant is a parameter of the protocol; once it is frozen,
changing one is a new revision, not an edit. Nothing here starts a process
or reads a clock.

Reads the records `v3_verify_remote` writes (schema
`larql.v3.verify-remote-trial.v1`). The bracket drift, the relay gate and
the stall rule are BLOCK-TIME's, imported, not restated.
"""
import random
import statistics

import block_time_2_model as b2
import block_time_model as bt
from profile_exact import agreement

PROTOCOL = "VERIFY-REMOTE-1 revision 1 (draft)"
SCHEMA = "larql.v3.verify-remote.v1"
TRIAL_SCHEMA = "larql.v3.verify-remote-trial.v1"

ROUTED_LAYERS = bt.ROUTED_LAYERS
SEQUENTIAL = "sequential"
ARMS = {SEQUENTIAL: None, "verify-2": 2, "verify-4": 4, "verify-8": 8}
RTT_MS = (0.0, 5.0, 20.0)
REPEATS = 2
# Kinds are reported separately and never aggregated.
COPY_HEAVY, LOW_COPY = "copy-heavy", "low-copy"
WORKLOADS = {"json_edit": COPY_HEAVY, "code_docstring": COPY_HEAVY,
             "prose_free": LOW_COPY, "explain": LOW_COPY}
GENERATE = 64
# gpt-oss-20b.vindex3/generation_config.json eos_token_id:
# <|return|>, <|endoftext|>, <|call|>.
STOP_IDS = (200002, 199999, 200012)
# The trial binary's untimed prefill width, the same in every arm.
PREFILL_WIDTH = 8
# The bracket: short, sequential, 0 ms, on the shortest prompt.
BRACKET_WORKLOAD = "prose_free"
BRACKET_GENERATE = 16
BRACKET_RTT_MS = 0.0
# Frame widths the block-frame qualification covers.
QUALIFIED_WIDTHS = (1, 2, 4, 8)
# Shuffles the units within each repeat and the arms within each unit, from one generator.
SHUFFLE_SEED = 20261116
GATE = bt.GATE
# "Faster" or "slower" needs a ratio beyond this in both repeats.
EFFECT = 0.01
MAX_WARMUPS = bt.MAX_WARMUPS
# Diagnostic only: one corrupted-proposer control per width, never timed as a result.
CONTROL_WORKLOAD = "json_edit"
CONTROL_RTT_MS = 0.0
CONTROL_CORRUPTION = "middle"

ROLE_BRACKET, ROLE_ARM, ROLE_CONTROL = "bracket", "arm", "control"
VERIFY, STEP = "verify", "step"
SOURCES = ("prefill", "accepted-proposal", "target-correction", "single-step")
SINGLE_ROW, MULTI_ROW = "single-row", "multi-row"
NS_PER_S = 1_000_000_000


class Stop(Exception):
    """A gate that ends the run: a correctness failure."""


def unit_order():
    """The units in run order: every workload at every delay once per repeat, shuffled within the repeat."""
    generator = random.Random(SHUFFLE_SEED)
    units = []
    for repeat in range(REPEATS):
        cells = [(workload, rtt_ms) for workload in WORKLOADS for rtt_ms in RTT_MS]
        generator.shuffle(cells)
        for workload, rtt_ms in cells:
            arms = list(ARMS)
            generator.shuffle(arms)
            units.append({"repeat": repeat, "workload": workload, "rtt_ms": rtt_ms, "arms": arms})
    return units


def bracket_entry(unit, side, label):
    return {"label": label, "unit": unit, "role": ROLE_BRACKET, "bracket": side, "arm": SEQUENTIAL,
            "workload": BRACKET_WORKLOAD, "generate": BRACKET_GENERATE, "rtt_ms": BRACKET_RTT_MS,
            "corrupt": "none"}


def schedule(controls=False):
    """Every trial after warm-up and the references, in order: bracket, four arms, bracket, per unit.

    With `controls` (a diagnostic), one corrupted-proposer trial per width
    follows the last unit. Controls are never brackets' business and never timed results.
    """
    trials = []

    def label():
        return f"trial-{len(trials):03d}"

    for index, unit in enumerate(unit_order()):
        trials.append(bracket_entry(index, "before", label()))
        for arm in unit["arms"]:
            trials.append({"label": label(), "unit": index, "repeat": unit["repeat"], "role": ROLE_ARM,
                           "bracket": None, "arm": arm, "workload": unit["workload"], "generate": GENERATE,
                           "rtt_ms": unit["rtt_ms"], "corrupt": "none"})
        trials.append(bracket_entry(index, "after", label()))
    if controls:
        for arm, width in ARMS.items():
            if width is not None:
                trials.append({"label": label(), "unit": None, "role": ROLE_CONTROL, "bracket": None,
                               "arm": arm, "workload": CONTROL_WORKLOAD, "generate": GENERATE,
                               "rtt_ms": CONTROL_RTT_MS, "corrupt": CONTROL_CORRUPTION})
    return trials


def reference_entry(workload):
    """The sequential trial whose ids every later trial of the workload must equal."""
    return {"label": f"reference-{workload}", "unit": None, "role": "reference", "bracket": None,
            "arm": SEQUENTIAL, "workload": workload, "generate": GENERATE, "rtt_ms": 0.0, "corrupt": "none"}


def warm_entry(index):
    entry = bracket_entry(None, None, f"warm-{index}")
    entry["role"] = "warm"
    return entry


def is_pause_point(entry):
    """The first trial of a unit: the only place the run may hold."""
    return entry["role"] == ROLE_BRACKET and entry["bracket"] == "before"


# ── per-trial records ────────────────────────────────────────────────────


def generation_units(trial):
    return trial["units"][trial["generation_from"]["unit"]:]


def first_difference(reference, ids):
    """The first position at which `ids` leaves `reference`, or None if they are equal."""
    for position, (a, b) in enumerate(zip(reference, ids)):
        if a != b:
            return position
    return None if len(reference) == len(ids) else min(len(reference), len(ids))


def ledger_defects(trial):
    """Gate 2: the event ledger against the output and against the units that ran."""
    ids, units = trial["ids"], generation_units(trial)
    if trial["timed_ids"] != max(len(ids) - 1, 0):
        return [f"{trial['timed_ids']} timed ids recorded for {len(ids)} ids"]
    steps = [u for u in units if u["kind"] == STEP]
    verifies = [u for u in units if u["kind"] == VERIFY]
    ledger = trial["ledger"]
    if trial["width"] is None:
        defects = [] if ledger is None else ["a sequential trial carries a ledger"]
        if verifies:
            defects.append(f"a sequential trial ran {len(verifies)} verify blocks")
        if len(steps) != max(len(ids) - 1, 0):
            defects.append(f"{len(steps)} single steps for {len(ids)} ids")
        return defects
    if ledger is None:
        return ["a verify trial has no ledger"]
    defects = []
    if ledger["reconcile"] != "ok":
        defects.append(f"the ledger does not reconcile: {ledger['reconcile']}")
    events = ledger["events"]
    emitted = [e["id"] for e in events if e["emitted"]]
    if emitted != ids:
        defects.append("the emitted events are not the generated ids")
    if ids and (not events or events[0]["source"] != "prefill"):
        defects.append("the first event is not the prefill id")
    count = {source: sum(e["source"] == source for e in events) for source in SOURCES}
    if any(e["source"] not in SOURCES for e in events):
        defects.append("an event has an unknown source")
    blocks = ledger["blocks"]
    if count["prefill"] != (1 if ids else 0):
        defects.append(f"{count['prefill']} prefill events")
    if count["accepted-proposal"] != sum(b["accepted"] for b in blocks):
        defects.append("accepted-proposal events do not match the blocks' accepted counts")
    if count["target-correction"] != len(blocks):
        defects.append("target corrections do not match the blocks")
    if count["single-step"] != ledger["single_steps"] or len(steps) != ledger["single_steps"]:
        defects.append("single steps do not match the ledger")
    if len(verifies) != len(blocks):
        defects.append(f"{len(verifies)} verify units for {len(blocks)} blocks")
    else:
        for index, (unit, block) in enumerate(zip(verifies, blocks)):
            if (unit["width"], unit["start"], unit["committed"]) != (
                    block["width"], block["start"], block["accepted"] + 1):
                defects.append(f"verify unit {index} does not match block {index}")
    return defects


def frame_defects(trial):
    """Gate 3: every unit's exchanges are one frame per routed layer, of the unit's own shape, and nothing else."""
    exchanges, defects = trial["exchanges"], []
    if len(trial["shards"]) != 1:
        defects.append(f"{len(trial['shards'])} workers configured, one required")
    covered = 0
    for index, unit in enumerate(trial["units"]):
        start, end = unit["exchanges"]
        if start != covered:
            defects.append(f"unit {index}: exchanges {start}..{end} do not follow the previous unit")
        covered = end
        mine = exchanges[start:end]
        if len(mine) != ROUTED_LAYERS:
            defects.append(f"unit {index}: {len(mine)} exchanges, expected {ROUTED_LAYERS}")
        route, rows = (MULTI_ROW, unit["width"]) if unit["kind"] == VERIFY else (SINGLE_ROW, 1)
        if any(e["route"] != route or e["rows"] != rows for e in mine):
            defects.append(f"unit {index}: an exchange is not a {rows}-row {route} frame")
        if sorted(e["layer"] for e in mine) != sorted(set(e["layer"] for e in mine)):
            defects.append(f"unit {index}: a routed layer was contacted twice")
    if covered != len(exchanges):
        defects.append(f"{len(exchanges) - covered} exchanges belong to no unit")
    whole = trial["frames"]["whole_run"]
    if whole[SINGLE_ROW] + whole[MULTI_ROW] != len(exchanges):
        defects.append("the frame counts do not match the exchange records")
    return defects


def evaluated_positions(trial):
    """Positions the target ran, rejected ones included: what the relay carried."""
    return sum(unit["width"] for unit in trial["units"])


def traffic_by_width(trial):
    """Generation exchanges and bytes by frame width, and the share at widths the qualification does not cover."""
    exchanges = trial["exchanges"][trial["generation_from"]["exchange"]:]
    widths = {}
    for e in exchanges:
        entry = widths.setdefault(str(e["rows"]), {"exchanges": 0, "bytes": 0})
        entry["exchanges"] += 1
        entry["bytes"] += e["request_bytes"] + e["response_bytes"]
    total_exchanges = sum(w["exchanges"] for w in widths.values())
    total_bytes = sum(w["bytes"] for w in widths.values())
    unqualified = [w for k, w in widths.items() if int(k) not in QUALIFIED_WIDTHS]
    return {
        "by_width": widths,
        "unqualified_exchange_share": (sum(w["exchanges"] for w in unqualified) / total_exchanges
                                       if total_exchanges else 0.0),
        "unqualified_byte_share": sum(w["bytes"] for w in unqualified) / total_bytes if total_bytes else 0.0,
    }


def summarize(trial):
    """Where one trial's generation went, and what speculation did in it."""
    ids, units = trial["ids"], generation_units(trial)
    # The first id is prefill's and produced untimed; gate 2 checks this count.
    produced = trial["timed_ids"]
    wall_ns = trial["wall_ns"]
    ledger = trial["ledger"] or {"blocks": [], "single_steps": len(units)}
    blocks = ledger["blocks"]
    committed = sum(u["committed"] or 0 for u in units)
    in_blocks = sum(u["committed"] or 0 for u in units if u["kind"] == VERIFY)
    verified = sum(u["width"] for u in units if u["kind"] == VERIFY)
    exchanges = trial["exchanges"][trial["generation_from"]["exchange"]:]
    records = trial["worker_records"][trial["generation_from"]["worker_record"]:]
    workers = [r.get("worker") for r in records]
    execute_ns = (sum(w["execute_ns"] for w in workers)
                  if trial.get("profiled") and len(records) == len(exchanges) and all(workers) else None)
    return {
        "ids": len(ids),
        # The first id is prefill's and produced untimed; the clock covers the rest.
        "timed_ids": produced,
        "wall_ns": wall_ns,
        "tokens_per_s": produced * NS_PER_S / wall_ns if produced and wall_ns > 0 else None,
        "per_position_ns": wall_ns / produced if produced else None,
        "exchanges": len(exchanges),
        "exchange_ns": sum(e["roundtrip_ns"] for e in exchanges),
        "blocks": len(blocks),
        "single_steps": ledger["single_steps"],
        "proposed": sum(b["proposed"] for b in blocks),
        "accepted": sum(b["accepted"] for b in blocks),
        "mean_accepted_per_block": statistics.fmean([b["accepted"] for b in blocks]) if blocks else None,
        "rejection_offsets": [b["rejected_at"] for b in blocks if b["rejected_at"] is not None],
        "block_widths": [b["width"] for b in blocks],
        "verified_positions": verified,
        "share_committed_in_blocks": in_blocks / committed if committed else 0.0,
        # A diagnostic, never a gate: the arms do different work by design.
        "worker_execute_ns_per_verified_position": (execute_ns / (verified + len(units) - len(blocks))
                                                    if execute_ns is not None and units else None),
        "traffic": traffic_by_width(trial),
        # Controls only: whether every corrupted block rejected at or before its changed offset.
        "rejected_within_change": rejected_within_change(blocks, trial.get("corrupted_offsets")),
    }


def rejected_within_change(blocks, offsets):
    """None for an uncorrupted trial; else whether each changed block rejected at or before the change."""
    if not offsets or all(o is None for o in offsets):
        return None
    if len(offsets) != len(blocks):
        return False
    return all(b["rejected_at"] is not None and b["rejected_at"] <= o
               for b, o in zip(blocks, offsets) if o is not None)


def trial_defects(observation):
    """Reasons that void a trial: gates 2 and 3, the relay and the stall rule."""
    reasons = list(observation["ledger_defects"]) + list(observation["frame_defects"])
    if observation.get("relay") is not None and not observation["relay"]["relay_on_schedule"]:
        reasons.append("relay behind schedule")
    for suspension in observation.get("suspensions", []):
        reasons.append(f"driver suspended {suspension['seconds']:.1f} s during the trial")
    return reasons


def bracket_drift(before, after):
    return agreement(before["summary"]["per_position_ns"], after["summary"]["per_position_ns"])


def judge_units(observations):
    """Mark every unit valid or void, with reasons; every trial inherits its unit's verdict."""
    units = {}
    for observation in observations:
        if observation.get("unit") is not None:
            units.setdefault(observation["unit"], []).append(observation)
    verdicts = []
    for unit, members in sorted(units.items()):
        reasons = []
        for member in members:
            reasons += [f"{member['label']}: {reason}" for reason in trial_defects(member)]
        brackets = {m["bracket"]: m for m in members if m["role"] == ROLE_BRACKET}
        arms = [m for m in members if m["role"] == ROLE_ARM]
        drift = None
        if set(brackets) != {"before", "after"} or sorted(m["arm"] for m in arms) != sorted(ARMS):
            reasons.append("incomplete unit")
        else:
            drift = bracket_drift(brackets["before"], brackets["after"])
            if drift > GATE:
                reasons.append(f"brackets differ by {drift:.4f}")
        for member in members:
            member["valid"] = not reasons
            member["void_reasons"] = list(reasons)
        first = arms[0] if arms else members[0]
        verdicts.append({"unit": unit, "repeat": first.get("repeat"), "workload": first["workload"],
                         "rtt_ms": first["rtt_ms"], "bracket_drift": drift,
                         "valid": not reasons, "void_reasons": reasons})
    return verdicts


def statement(ratios):
    """The pre-registered reading of one cell's per-repeat speed ratios (arm over sequential)."""
    if len(ratios) != REPEATS or any(r is None for r in ratios):
        return "missing"
    if all(r > 1 + EFFECT for r in ratios):
        return "faster than sequential"
    if all(r < 1 - EFFECT for r in ratios):
        return "slower than sequential"
    return "no difference shown"


def analyse(observations, verdicts):
    """Per workload, delay and width: the statement, the ratios and what speculation did. Never pooled."""
    valid_units = {v["unit"] for v in verdicts if v["valid"]}
    by_cell = {}
    for o in observations:
        if o["role"] == ROLE_ARM:
            by_cell.setdefault((o["workload"], o["rtt_ms"], o["repeat"]), {})[o["arm"]] = o
    cells = []
    for workload, kind in WORKLOADS.items():
        for rtt_ms in RTT_MS:
            for arm, width in ARMS.items():
                if width is None:
                    continue
                repeats = []
                for repeat in range(REPEATS):
                    trials = by_cell.get((workload, rtt_ms, repeat), {})
                    mine, base = trials.get(arm), trials.get(SEQUENTIAL)
                    if mine is None or base is None or mine["unit"] not in valid_units:
                        repeats.append({"repeat": repeat, "ratio": None, "valid": False})
                        continue
                    a, s = mine["summary"], base["summary"]
                    ratio = (a["tokens_per_s"] / s["tokens_per_s"]
                             if a["tokens_per_s"] and s["tokens_per_s"] else None)
                    repeats.append({
                        "repeat": repeat, "valid": True, "ratio": ratio,
                        "tokens_per_s": a["tokens_per_s"], "sequential_tokens_per_s": s["tokens_per_s"],
                        "proposed_nothing": a["proposed"] == 0,
                        "mean_accepted_per_block": a["mean_accepted_per_block"],
                        "share_committed_in_blocks": a["share_committed_in_blocks"],
                        "rejection_offsets": a["rejection_offsets"],
                        # Descriptive only: round trips sequential needed per round trip this arm needed.
                        "exchange_ratio": s["exchanges"] / a["exchanges"] if a["exchanges"] else None,
                        "unqualified_exchange_share": a["traffic"]["unqualified_exchange_share"],
                        "unqualified_byte_share": a["traffic"]["unqualified_byte_share"],
                    })
                cells.append({"workload": workload, "kind": kind, "rtt_ms": rtt_ms, "arm": arm,
                              "statement": statement([r["ratio"] for r in repeats]), "repeats": repeats})
    return {
        "protocol": PROTOCOL, "cells": cells,
        "valid_units": sum(v["valid"] for v in verdicts), "void_units": sum(not v["valid"] for v in verdicts),
        "thresholds": {"gate": GATE, "effect": EFFECT},
        "reading": "committed new tokens per second, prefill excluded; copy-heavy and low-copy workloads are "
                   "never aggregated; the 1% rule is a pre-registered reporting rule, not statistical "
                   "significance; the exchange ratio is descriptive",
    }


def control_reading(controls, observations):
    """Diagnostic: each corrupted control's ids equal the reference (checked when run), its
    acceptance fell below every clean trial of the same width, and every changed block rejected
    at or before its change."""
    clean = {}
    for o in observations:
        if (o["role"] == ROLE_ARM and o["workload"] == CONTROL_WORKLOAD and o["rtt_ms"] == CONTROL_RTT_MS):
            clean.setdefault(o["arm"], []).append(o["summary"]["accepted"])
    readings = []
    for control in controls:
        accepted = control["summary"]["accepted"]
        baseline = clean.get(control["arm"])
        readings.append({"arm": control["arm"], "accepted": accepted, "clean_accepted": baseline,
                         "acceptance_fell": None if not baseline else accepted < min(baseline),
                         "rejected_within_change": control["summary"]["rejected_within_change"]})
    return readings


def attach_stalls(observations, stalls):
    """BLOCK-TIME-2's stall rule: each trial gets the stalls that overlap it."""
    b2.attach_stalls(observations, stalls)
