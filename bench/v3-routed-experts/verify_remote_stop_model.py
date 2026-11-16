"""VERIFY-REMOTE-1 revision 2: the real-model stop check's requirements (VERIFY-REMOTE-1.md).

Pure. A declared functional check, never timed as a result, on one prompt
outside the four performance workloads. A `sequential` reference at 0 ms
must end at a configured stop id within the generation limit (a natural
stop). Then each verify arm must reproduce it exactly and end the same way.

Prompt lookup never proposes a stop id (`greedy_speculate` cuts every
proposal before one), so the stop arrives as the target's own id: the
correction after a block, or a single step. The record says which, and
whether the block before it accepted its whole proposal.
"""
import verify_remote_model as vr

SCHEMA = "larql.v3.verify-remote-stop-check.v1"
WORKLOAD = "stop_check"
RTT_MS = 0.0
NATURAL = "natural"
NO_NATURAL_STOP = "no natural stop"


def entries():
    """The reference first, then one trial per verify arm."""
    reference = dict(vr.reference_entry(WORKLOAD), label="stop-reference", role="stop-reference")
    arms = [{"label": f"stop-{arm}", "unit": None, "role": "stop-arm", "bracket": None, "arm": arm,
             "workload": WORKLOAD, "generate": vr.GENERATE, "rtt_ms": RTT_MS, "corrupt": "none"}
            for arm, width in vr.ARMS.items() if width is not None]
    return reference, arms


def reference_stop(trial):
    """`natural` if the reference ended at a configured stop id within the limit."""
    ids = trial["ids"]
    if ids and ids[-1] in vr.STOP_IDS and trial["id_counts"]["ended_by"] == "stop id":
        return NATURAL
    return NO_NATURAL_STOP


def stop_delivery(trial):
    """How the stop id arrived: the last emitted event's source and, for a
    correction, its block's proposal and acceptance."""
    if trial["ledger"] is None:
        return {"source": "single-step"}
    events = [e for e in trial["ledger"]["events"] if e["emitted"]]
    last = events[-1]
    delivery = {"source": last["source"]}
    if last["source"] == "target-correction":
        block = trial["ledger"]["blocks"][last["block"]]
        delivery.update(block=last["block"], proposed=block["proposed"], accepted=block["accepted"],
                        whole_proposal_accepted=block["accepted"] == block["proposed"])
    return delivery


def requirement_failures(trial, reference):
    """Every requirement the trial does not meet; empty if it meets them all.

    Exact ids ending at the stop, nothing after it, the ledger and frames
    (gates 2 and 3, which also show no exchange after the last unit), the
    continuation position, and the stop id's own event emitted last.
    """
    failures = []
    ids = trial["ids"]
    if ids != reference["ids"]:
        at = vr.first_difference(reference["ids"], ids)
        failures.append(f"ids differ from the sequential reference at position {at}")
    if not ids or ids[-1] not in vr.STOP_IDS or trial["id_counts"]["ended_by"] != "stop id":
        failures.append("did not end at a stop id")
    failures += [f"gate 2: {d}" for d in vr.ledger_defects(trial)]
    failures += [f"gate 3: {d}" for d in vr.frame_defects(trial)]
    # The stop id is emitted but never fed back: the session holds the
    # prompt and every emitted id but the last.
    expected = trial["prompt_tokens"] + len(ids) - 1
    if trial["final_position"] != expected:
        failures.append(f"final position {trial['final_position']}, expected {expected}")
    if trial["final_position"] != reference["final_position"]:
        failures.append("final position differs from the reference's")
    if trial["ledger"] is not None:
        events = trial["ledger"]["events"]
        emitted = [e for e in events if e["emitted"]]
        if not emitted or emitted[-1]["id"] not in vr.STOP_IDS:
            failures.append("the last emitted event is not the stop id")
        if any(e["emitted"] for e in events[len(emitted):]):
            failures.append("an event after the stop was emitted")
    return failures


def reading(reference_trial, arm_trials):
    """The check's result: whether the reference stopped naturally, and each arm against it."""
    stop = reference_stop(reference_trial)
    reference_failures = requirement_failures(reference_trial, reference_trial)
    arms = []
    for arm, trial in arm_trials:
        arms.append({"arm": arm, "ids": len(trial["ids"]), "final_position": trial["final_position"],
                     "stop_delivery": stop_delivery(trial),
                     "failures": requirement_failures(trial, reference_trial)})
    passes = stop == NATURAL and not reference_failures and all(not a["failures"] for a in arms) \
        and len(arms) == sum(w is not None for w in vr.ARMS.values())
    return {"reference": {"stop": stop, "ids": reference_trial["ids"],
                          "final_position": reference_trial["final_position"],
                          "failures": reference_failures},
            "arms": arms, "passes": passes,
            "synthetic_stop": False}
