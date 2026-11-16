"""VERIFY-REMOTE-1: schedule, gates, unit validity, statements and the driver loop."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import unittest

import verify_remote_model as vr

WORKLOADS_DIR = Path(__file__).with_name("verify-remote") / "workloads"
PROMPT = 20
BASE_NS = 40_000_000


def reference_ids(n=vr.GENERATE):
    return [1000 + i for i in range(n)]


FAKE_SHA = "0" * 64


def fake_trial(arm, ids=None, accept=2, shards=1, profiled=True, per_id_ns=BASE_NS, corrupt="none",
               generate=None):
    """A record shaped as `v3_verify_remote` writes it, for a consistent run.

    Prefill runs the prompt in blocks of eight. A verify arm proposes as much
    as the loop allows and has `accept` of each proposal accepted.
    """
    ids = reference_ids() if ids is None else ids
    width = vr.ARMS[arm]
    units, exchanges, records = [], [], []
    position = 0

    def unit(kind, w, committed):
        nonlocal position
        start = len(exchanges)
        for layer in range(vr.ROUTED_LAYERS):
            exchanges.append({"route": vr.MULTI_ROW if kind == vr.VERIFY else vr.SINGLE_ROW, "layer": layer,
                              "rows": w, "pairs": 4 * w, "request_bytes": 100 * w, "response_bytes": 100 * w,
                              "roundtrip_ns": 1000})
            records.append({"worker": {"execute_ns": 10 * w, "handler_ns": 20 * w}})
        units.append({"kind": kind, "start": position, "width": w, "committed": committed, "run_ns": 1,
                      "commit_ns": 0, "exchanges": [start, len(exchanges)],
                      "worker_records": [start, len(records)]})
        position += committed

    for chunk in range(0, PROMPT, vr.PREFILL_WIDTH):
        w = min(vr.PREFILL_WIDTH, PROMPT - chunk)
        unit(vr.VERIFY, w, w)
    generation = {"unit": len(units), "exchange": len(exchanges), "worker_record": len(records)}
    ledger, corrupted = None, None
    if width is None:
        for _ in ids[1:]:
            unit(vr.STEP, 1, 1)
    else:
        events = [{"id": ids[0], "source": "prefill", "block": None, "emitted": True}] if ids else []
        blocks, steps, produced = [], 0, 1
        while produced < len(ids):
            budget = min(width - 1, len(ids) - produced - 1)
            if budget == 0:
                unit(vr.STEP, 1, 1)
                steps += 1
                events.append({"id": ids[produced], "source": "single-step", "block": None, "emitted": True})
                produced += 1
                continue
            accepted = min(accept, budget)
            start = position
            unit(vr.VERIFY, budget + 1, accepted + 1)
            index = len(blocks)
            blocks.append({"start": start, "width": budget + 1, "proposed": budget, "accepted": accepted,
                           "rejected_at": accepted if accepted < budget else None})
            for _ in range(accepted):
                events.append({"id": ids[produced], "source": "accepted-proposal", "block": index,
                               "emitted": True})
                produced += 1
            events.append({"id": ids[produced], "source": "target-correction", "block": index, "emitted": True})
            produced += 1
        ledger = {"events": events, "blocks": blocks, "single_steps": steps, "prompt_len": PROMPT,
                  "start_position": PROMPT, "final_position": position, "reconcile": "ok"}
        if corrupt != "none":
            # The changed token is at the accepted count: the first rejection.
            corrupted = [b["accepted"] if b["rejected_at"] is not None else b["proposed"] - 1 for b in blocks]
        else:
            corrupted = [None] * len(blocks)
    singles = sum(e["route"] == vr.SINGLE_ROW for e in exchanges)
    return {
        "schema": vr.TRIAL_SCHEMA, "arm": arm, "width": width, "corrupt": corrupt, "profiled": profiled,
        "generate": len(ids) if generate is None else generate, "stop": list(vr.STOP_IDS),
        "prefill_width": vr.PREFILL_WIDTH, "workload": {"name": "fake", "sha256": FAKE_SHA}, "prompt_tokens": PROMPT,
        "shards": ["http://relay"] * shards, "ids": ids, "timed_ids": max(len(ids) - 1, 0),
        "id_counts": {"requested": len(ids) if generate is None else generate, "emitted": len(ids),
                      "prefill_produced": 1 if ids else 0, "timed": max(len(ids) - 1, 0),
                      "ended_by": ("nothing requested" if not ids else "stop id" if ids[-1] in vr.STOP_IDS
                                   else "generation limit")},
        "corrupted_offsets": corrupted, "wall_ns": per_id_ns * max(len(ids) - 1, 1),
        "final_position": position, "generation_from": generation, "units": units, "ledger": ledger,
        "frames": {"whole_run": {vr.SINGLE_ROW: singles, vr.MULTI_ROW: len(exchanges) - singles}},
        "exchanges": exchanges, "worker_records": records,
        "relay_mark": {"control": "127.0.0.1:1", "answer": "ok 1", "error": None, "waited_ns": 1000,
                       "exchanges_completed": generation["exchange"]},
    }


def fake_relay(prefill_late_ns=0, generation_late_ns=0, answer="ok", reasons=(), chunks=(10, 6)):
    """A relay's closing and mark records for one connection marked once.

    Prefill lateness is all in the snapshot; generation lateness is what the
    closing records add after it, split across the two directions.
    """
    before, after = chunks
    snapshot = {"mark": 1, "connection": 0, "queued": 0, "unread": False, "last_delivered": "response",
                "read_after_snapshot": 0}
    stats = []
    for direction, share in (("request", 0.25), ("response", 0.75)):
        prefill = int(prefill_late_ns * share)
        generation = int(generation_late_ns * share)
        snapshot[direction] = {"bytes": 1000 * before, "chunks": before, "late_sum_ns": prefill,
                               "late_max_ns": prefill}
        stats.append({"connection": 0, "direction": direction, "qos": "default", "delay_us": 0,
                      "rate_bits_per_s": "inf", "bytes": 1000 * (before + after), "chunks": before + after,
                      "late_sum_ns": prefill + generation, "late_max_ns": max(prefill, generation)})
    summary = {"mark": 1, "summary": True, "connections": [0], "accepted_before": 1, "answer": answer,
               "reasons": list(reasons)}
    return {"stats": stats, "marks": [snapshot, summary]}


def observation(entry, trial):
    return dict(entry, summary=vr.summarize(trial), ledger_defects=vr.ledger_defects(trial),
                frame_defects=vr.frame_defects(trial), relay=None, suspensions=[])


def clean_run(per_arm_ns=None, bracket_ns=None):
    """Observations for the whole schedule; `per_arm_ns(entry)` sets each arm's per-id time."""
    out = []
    for entry in vr.schedule():
        if entry["role"] == vr.ROLE_BRACKET:
            ns = bracket_ns(entry) if bracket_ns else BASE_NS
            trial = fake_trial(vr.SEQUENTIAL, reference_ids(vr.BRACKET_GENERATE), per_id_ns=ns)
        else:
            ns = per_arm_ns(entry) if per_arm_ns else BASE_NS
            trial = fake_trial(entry["arm"], per_id_ns=ns)
        out.append(observation(entry, trial))
    return out


class Schedule(unittest.TestCase):
    def test_twenty_four_units_of_six_trials(self):
        order = vr.schedule()
        self.assertEqual(len(vr.unit_order()), 24)
        self.assertEqual(len(order), 24 * 6)
        for unit in range(24):
            members = [e for e in order if e["unit"] == unit]
            self.assertEqual([m["role"] for m in members], ["bracket"] + ["arm"] * 4 + ["bracket"])
            self.assertEqual([m["bracket"] for m in members if m["role"] == "bracket"], ["before", "after"])
            self.assertEqual(sorted(m["arm"] for m in members if m["role"] == "arm"), sorted(vr.ARMS))
            self.assertEqual(len({(m["workload"], m["rtt_ms"]) for m in members if m["role"] == "arm"}), 1)

    def test_every_workload_and_delay_once_per_repeat(self):
        units = vr.unit_order()
        for repeat in range(vr.REPEATS):
            cells = Counter((u["workload"], u["rtt_ms"]) for u in units if u["repeat"] == repeat)
            self.assertEqual(set(cells), {(w, d) for w in vr.WORKLOADS for d in vr.RTT_MS})
            self.assertEqual(set(cells.values()), {1})
        self.assertEqual([u["repeat"] for u in units], [0] * 12 + [1] * 12)

    def test_brackets_are_sequential_zero_delay_and_short(self):
        for entry in vr.schedule():
            if entry["role"] == "bracket":
                self.assertEqual((entry["arm"], entry["rtt_ms"], entry["workload"], entry["generate"]),
                                 (vr.SEQUENTIAL, 0.0, vr.BRACKET_WORKLOAD, vr.BRACKET_GENERATE))
            else:
                self.assertEqual(entry["generate"], vr.GENERATE)

    def test_the_schedule_is_deterministic_and_pinned(self):
        self.assertEqual(vr.schedule(), vr.schedule())
        self.assertEqual(vr.SHUFFLE_SEED, 20261116)
        digest = hashlib.sha256(json.dumps(vr.unit_order(), sort_keys=True).encode()).hexdigest()
        self.assertEqual(digest, PINNED_UNIT_ORDER_SHA256)
        first_arms = [tuple(u["arms"]) for u in vr.unit_order()]
        self.assertGreater(len(set(first_arms)), 1, "arm order is shuffled per unit")

    def test_controls_only_on_request_one_per_width(self):
        self.assertFalse(any(e["role"] == "control" for e in vr.schedule()))
        controls = [e for e in vr.schedule(controls=True) if e["role"] == "control"]
        self.assertEqual(sorted(e["arm"] for e in controls), ["verify-2", "verify-4", "verify-8"])
        for control in controls:
            self.assertEqual((control["workload"], control["rtt_ms"], control["corrupt"]),
                             (vr.CONTROL_WORKLOAD, 0.0, vr.CONTROL_CORRUPTION))
        self.assertEqual(vr.WORKLOADS[vr.CONTROL_WORKLOAD], vr.COPY_HEAVY)

    def test_labels_are_unique_and_ordered(self):
        labels = [e["label"] for e in vr.schedule(controls=True)]
        self.assertEqual(labels, sorted(labels))
        self.assertEqual(len(labels), len(set(labels)))

    def test_only_unit_openings_and_controls_are_pause_points(self):
        points = [e for e in vr.schedule(controls=True) if vr.is_pause_point(e)]
        self.assertEqual([e["unit"] for e in points], list(range(24)) + [None] * 3)
        self.assertEqual({e["role"] for e in points[24:]}, {"control"})


class Workloads(unittest.TestCase):
    def test_the_registered_workloads_and_stop_ids_match_the_committed_files(self):
        index = json.loads((WORKLOADS_DIR / "index.json").read_text())
        self.assertEqual(index["stop_ids"]["ids"], list(vr.STOP_IDS))
        for name, kind in vr.WORKLOADS.items():
            path = WORKLOADS_DIR / f"{name}.json"
            saved = json.loads(path.read_text())
            self.assertEqual(index[name]["kind"], kind)
            self.assertEqual(saved["kind"], kind)
            self.assertEqual(len(saved["prompt"]), index[name]["tokens"])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), index[name]["sha256"])
        self.assertEqual(Counter(vr.WORKLOADS.values()), {vr.COPY_HEAVY: 2, vr.LOW_COPY: 2})

    def test_the_bracket_uses_the_shortest_prompt(self):
        index = json.loads((WORKLOADS_DIR / "index.json").read_text())
        self.assertEqual(vr.BRACKET_WORKLOAD, min(vr.WORKLOADS, key=lambda n: index[n]["tokens"]))


class Gates(unittest.TestCase):
    def test_identity_reports_the_first_differing_position(self):
        ref = reference_ids(8)
        self.assertIsNone(vr.first_difference(ref, list(ref)))
        changed = list(ref)
        changed[5] += 1
        self.assertEqual(vr.first_difference(ref, changed), 5)
        self.assertEqual(vr.first_difference(ref, ref[:6]), 6)
        self.assertEqual(vr.first_difference(ref[:6], ref), 6)

    def test_clean_records_pass_gates_2_and_3_for_every_arm(self):
        for arm in vr.ARMS:
            for accept in (0, 1, 7):
                trial = fake_trial(arm, accept=accept)
                self.assertEqual(vr.ledger_defects(trial), [], f"{arm} {accept}")
                self.assertEqual(vr.frame_defects(trial), [], f"{arm} {accept}")

    def test_a_short_run_ending_at_a_stop_passes(self):
        trial = fake_trial("verify-4", ids=reference_ids(5))
        self.assertEqual(vr.ledger_defects(trial) + vr.frame_defects(trial), [])

    def test_a_ledger_that_does_not_reconcile_or_match_is_void(self):
        trial = fake_trial("verify-4")
        trial["ledger"]["reconcile"] = "the target ended at 3"
        self.assertTrue(any("does not reconcile" in d for d in vr.ledger_defects(trial)))
        trial = fake_trial("verify-4")
        trial["ids"] = trial["ids"][:-1] + [7]
        self.assertIn("the emitted events are not the generated ids", vr.ledger_defects(trial))
        trial = fake_trial("verify-4")
        trial["ledger"]["blocks"][0]["accepted"] -= 1
        defects = vr.ledger_defects(trial)
        self.assertTrue(any("accepted-proposal" in d for d in defects))
        self.assertTrue(any("does not match block 0" in d for d in defects))
        trial = fake_trial("verify-4")
        correction = next(e for e in trial["ledger"]["events"] if e["source"] == "target-correction")
        correction["source"] = "single-step"
        defects = vr.ledger_defects(trial)
        self.assertIn("target corrections do not match the blocks", defects)
        self.assertIn("single steps do not match the ledger", defects)
        trial = fake_trial("verify-4")
        trial["ledger"]["events"][0]["source"] = "accepted-proposal"
        self.assertIn("the first event is not the prefill id", vr.ledger_defects(trial))

    def test_unemitted_trailing_events_are_counted_but_not_output(self):
        produced = reference_ids(9) + [vr.STOP_IDS[0], 1234]
        trial = fake_trial("verify-4", ids=produced, generate=vr.GENERATE)
        # The last id came after the stop: unemitted, and not in the output.
        trial["ledger"]["events"][-1]["emitted"] = False
        trial["ids"] = produced[:-1]
        trial["timed_ids"] -= 1
        trial["id_counts"].update(emitted=10, timed=9, ended_by="stop id")
        self.assertEqual(vr.ledger_defects(trial), [])
        # An unemitted id in the middle breaks the output.
        trial = fake_trial("verify-4", ids=reference_ids(10))
        trial["ledger"]["events"][2]["emitted"] = False
        self.assertIn("the emitted events are not the generated ids", vr.ledger_defects(trial))

    def test_a_record_that_is_not_the_scheduled_trial_is_named(self):
        entry = {"arm": "verify-8", "corrupt": "none", "generate": vr.GENERATE}
        self.assertEqual(vr.record_mismatches(entry, fake_trial("verify-8"), FAKE_SHA), [])
        wrong = vr.record_mismatches(entry, fake_trial("verify-2"), FAKE_SHA)
        self.assertTrue(any(w.startswith("arm") for w in wrong) and any(w.startswith("width") for w in wrong))
        self.assertTrue(vr.record_mismatches(entry, fake_trial("verify-8"), "1" * 64))
        trial = fake_trial("verify-8")
        trial["stop"] = [1]
        self.assertTrue(any(w.startswith("stop ids") for w in vr.record_mismatches(entry, trial, FAKE_SHA)))
        relabelled = fake_trial("verify-8")
        relabelled["units"][-2]["width"] = 9
        self.assertIn("a verify block is wider than 8", vr.record_mismatches(entry, relabelled, FAKE_SHA))

    def test_worker_records_and_generation_start_are_checked(self):
        trial = fake_trial("verify-4")
        trial["units"][3]["worker_records"] = [trial["units"][3]["worker_records"][0] + 1,
                                               trial["units"][3]["worker_records"][1]]
        self.assertTrue(any("worker records" in d for d in vr.frame_defects(trial)))
        trial = fake_trial("verify-4")
        trial["generation_from"]["exchange"] += 1
        self.assertIn("generation_from does not point at the first generation unit's records",
                      vr.frame_defects(trial))
        trial = fake_trial("verify-4")
        trial["exchanges"][0]["layer"] = vr.ROUTED_LAYERS
        self.assertTrue(any("outside" in d for d in vr.frame_defects(trial)))

    def test_sequential_must_be_single_steps_without_a_ledger(self):
        trial = fake_trial(vr.SEQUENTIAL)
        trial["ledger"] = {"reconcile": "ok"}
        self.assertIn("a sequential trial carries a ledger", vr.ledger_defects(trial))
        trial = fake_trial(vr.SEQUENTIAL)
        trial["units"][-1]["kind"] = vr.VERIFY
        self.assertTrue(any("verify blocks" in d for d in vr.ledger_defects(trial)))

    def test_frames_must_be_one_per_layer_of_the_units_shape(self):
        trial = fake_trial("verify-8")
        start = trial["units"][trial["generation_from"]["unit"]]["exchanges"][0]
        trial["exchanges"][start]["route"] = vr.SINGLE_ROW
        self.assertTrue(any("is not a" in d for d in vr.frame_defects(trial)))
        trial = fake_trial(vr.SEQUENTIAL)
        trial["exchanges"][-1]["rows"] = 2
        self.assertTrue(any("is not a 1-row" in d for d in vr.frame_defects(trial)))
        trial = fake_trial("verify-2")
        trial["exchanges"].append(dict(trial["exchanges"][-1]))
        self.assertTrue(any("belong to no unit" in d for d in vr.frame_defects(trial)))
        trial = fake_trial("verify-2")
        trial["units"][-1]["exchanges"][1] -= 1
        defects = vr.frame_defects(trial)
        self.assertTrue(any("23 exchanges" in d for d in defects))
        # A gap between units: exchanges the record attributes to nothing.
        trial = fake_trial("verify-2")
        middle = trial["units"][5]
        middle["exchanges"] = [middle["exchanges"][0] + 1, middle["exchanges"][1]]
        self.assertTrue(any("do not follow the previous unit" in d for d in vr.frame_defects(trial)))
        trial = fake_trial("verify-2")
        trial["exchanges"][1]["layer"] = 0
        self.assertTrue(any("contacted twice" in d for d in vr.frame_defects(trial)))
        self.assertTrue(any("one required" in d for d in vr.frame_defects(fake_trial("verify-2", shards=2))))


class Summaries(unittest.TestCase):
    def test_tokens_per_second_counts_only_ids_produced_inside_the_clock(self):
        summary = vr.summarize(fake_trial(vr.SEQUENTIAL, per_id_ns=50_000_000))
        self.assertEqual(summary["timed_ids"], vr.GENERATE - 1)
        self.assertAlmostEqual(summary["tokens_per_s"], 20.0)
        self.assertEqual(summary["blocks"], 0)
        self.assertEqual(summary["share_committed_in_blocks"], 0.0)

    def test_acceptance_and_traffic_shares(self):
        summary = vr.summarize(fake_trial("verify-8", accept=2))
        self.assertEqual(summary["mean_accepted_per_block"], 2)
        self.assertTrue(all(offset == 2 for offset in summary["rejection_offsets"]))
        # Width-8 blocks are qualified; the closing blocks are narrower, some unqualified.
        traffic = summary["traffic"]
        self.assertIn("8", traffic["by_width"])
        widths = {int(w) for w in traffic["by_width"]}
        unqualified = widths - set(vr.QUALIFIED_WIDTHS)
        self.assertEqual(traffic["unqualified_exchange_share"] > 0, bool(unqualified))
        clean = vr.summarize(fake_trial(vr.SEQUENTIAL))["traffic"]
        self.assertEqual(clean["unqualified_exchange_share"], 0.0)

    def test_worker_execution_per_verified_position_is_a_diagnostic(self):
        summary = vr.summarize(fake_trial("verify-4"))
        self.assertIsNotNone(summary["worker_execute_ns_per_verified_position"])
        summary = vr.summarize(fake_trial("verify-4", profiled=False))
        self.assertIsNone(summary["worker_execute_ns_per_verified_position"])


class Units(unittest.TestCase):
    def test_a_clean_run_has_every_unit_valid(self):
        verdicts = vr.judge_units(clean_run())
        self.assertEqual(len(verdicts), 24)
        self.assertTrue(all(v["valid"] for v in verdicts))

    def test_brackets_beyond_one_percent_void_only_their_unit(self):
        def bracket_ns(entry):
            return BASE_NS * 1.012 if entry["unit"] == 5 and entry["bracket"] == "after" else BASE_NS
        verdicts = vr.judge_units(clean_run(bracket_ns=bracket_ns))
        self.assertEqual([v["unit"] for v in verdicts if not v["valid"]], [5])
        self.assertTrue(any("brackets differ" in r for r in verdicts[5]["void_reasons"]))

    def test_drift_within_the_gate_is_valid(self):
        def bracket_ns(entry):
            return BASE_NS * 1.009 if entry["bracket"] == "after" else BASE_NS
        self.assertTrue(all(v["valid"] for v in vr.judge_units(clean_run(bracket_ns=bracket_ns))))

    def test_a_defect_or_late_relay_or_stall_in_any_member_voids_the_unit_by_name(self):
        for change in ("frame", "relay", "stall"):
            observations = clean_run()
            target = observations[9]
            if change == "frame":
                target["frame_defects"] = ["unit 3: 23 exchanges, expected 24"]
            elif change == "relay":
                target["relay"] = {"relay_on_schedule": False}
            else:
                target["suspensions"] = [{"seconds": 2.0}]
            verdicts = vr.judge_units(observations)
            void = [v for v in verdicts if not v["valid"]]
            self.assertEqual([v["unit"] for v in void], [target["unit"]], change)
            self.assertTrue(all(r.startswith(target["label"]) for r in void[0]["void_reasons"]), change)
            for member in observations:
                if member["unit"] == target["unit"]:
                    self.assertFalse(member["valid"])

    def test_an_incomplete_unit_is_void(self):
        observations = [o for o in clean_run() if not (o["unit"] == 2 and o["role"] == "arm"
                                                        and o["arm"] == "verify-4")]
        verdicts = vr.judge_units(observations)
        self.assertIn("incomplete unit", verdicts[2]["void_reasons"])


class Statements(unittest.TestCase):
    def test_the_one_percent_rule_at_below_and_above(self):
        self.assertEqual(vr.statement([1.02, 1.011]), "faster than sequential")
        self.assertEqual(vr.statement([1.02, 1.01]), "no difference shown")
        self.assertEqual(vr.statement([0.98, 0.989]), "slower than sequential")
        self.assertEqual(vr.statement([0.98, 0.99]), "no difference shown")
        self.assertEqual(vr.statement([1.2, 0.8]), "no difference shown")
        self.assertEqual(vr.statement([1.2, None]), "missing")
        self.assertEqual(vr.statement([1.2]), "missing")

    def test_a_faster_arm_is_reported_per_cell_never_pooled(self):
        def per_arm_ns(entry):
            return BASE_NS // 2 if entry["arm"] == "verify-8" else BASE_NS
        observations = clean_run(per_arm_ns=per_arm_ns)
        report = vr.analyse(observations, vr.judge_units(observations))
        self.assertEqual(len(report["cells"]), 4 * 3 * 3)
        for cell in report["cells"]:
            expected = "faster than sequential" if cell["arm"] == "verify-8" else "no difference shown"
            self.assertEqual(cell["statement"], expected, cell)
            self.assertEqual(cell["kind"], vr.WORKLOADS[cell["workload"]])
            for repeat in cell["repeats"]:
                self.assertAlmostEqual(repeat["ratio"], 2.0 if cell["arm"] == "verify-8" else 1.0)
        self.assertNotIn("pooled", json.dumps(report))

    def test_the_delay_only_prediction_is_descriptive(self):
        # Sequential slow enough that the delay saved is a part of its time.
        observations = clean_run(per_arm_ns=lambda e: 2_000_000_000 if e["arm"] == vr.SEQUENTIAL else BASE_NS)
        report = vr.analyse(observations, vr.judge_units(observations))
        for cell in report["cells"]:
            for repeat in cell["repeats"]:
                if cell["rtt_ms"] == 0.0:
                    self.assertAlmostEqual(repeat["predicted_ratio"], 1.0)
                else:
                    self.assertGreater(repeat["predicted_ratio"], 1.0)
                self.assertEqual(repeat["rejected"], repeat["proposed"] - repeat["accepted"])

    def test_a_prefill_stop_reads_no_timed_generation(self):
        observations = clean_run()
        for o in observations:
            if o["role"] == "arm" and o["workload"] == "explain":
                o["summary"].update(timed_ids=0, tokens_per_s=None)
        report = vr.analyse(observations, vr.judge_units(observations))
        cells = [c for c in report["cells"] if c["workload"] == "explain"]
        self.assertEqual({c["statement"] for c in cells}, {"no timed generation"})

    def test_a_void_unit_makes_its_cells_missing(self):
        def bracket_ns(entry):
            return BASE_NS * 1.05 if entry["unit"] == 0 and entry["bracket"] == "after" else BASE_NS
        observations = clean_run(bracket_ns=bracket_ns)
        report = vr.analyse(observations, vr.judge_units(observations))
        unit = vr.unit_order()[0]
        cells = [c for c in report["cells"] if (c["workload"], c["rtt_ms"]) == (unit["workload"], unit["rtt_ms"])]
        self.assertEqual({c["statement"] for c in cells}, {"missing"})
        self.assertEqual(report["void_units"], 1)

    def test_a_width_that_proposes_nothing_is_reported_not_dropped(self):
        observations = clean_run()
        for o in observations:
            if o["role"] == "arm" and o["arm"] != vr.SEQUENTIAL and o["workload"] == "explain":
                o["summary"]["proposed"] = 0
        report = vr.analyse(observations, vr.judge_units(observations))
        cells = [c for c in report["cells"] if c["workload"] == "explain"]
        self.assertEqual(len(cells), 9)
        self.assertTrue(all(r["proposed_nothing"] for c in cells for r in c["repeats"]))

    def test_controls_report_whether_acceptance_fell(self):
        observations = clean_run()
        control = observation({"label": "c", "unit": None, "role": "control", "arm": "verify-4",
                               "workload": vr.CONTROL_WORKLOAD, "rtt_ms": 0.0},
                              fake_trial("verify-4", accept=0, corrupt="middle"))
        [reading] = vr.control_reading([control], observations)
        self.assertEqual(reading["accepted"], 0)
        self.assertTrue(reading["acceptance_fell"])
        self.assertTrue(reading["rejected_within_change"])

    def test_a_control_reports_its_own_gate_defects(self):
        observations = clean_run()
        trial = fake_trial("verify-4", accept=0, corrupt="middle")
        trial["ledger"]["reconcile"] = "broken"
        control = observation({"label": "c", "unit": None, "role": "control", "arm": "verify-4",
                               "workload": vr.CONTROL_WORKLOAD, "rtt_ms": 0.0}, trial)
        [reading] = vr.control_reading([control], observations)
        self.assertTrue(any("does not reconcile" in d for d in reading["defects"]))

    def test_a_change_the_target_accepted_past_is_flagged(self):
        blocks = [{"rejected_at": 1}, {"rejected_at": None}]
        self.assertTrue(vr.rejected_within_change(blocks, [1, None]))
        self.assertFalse(vr.rejected_within_change(blocks, [0, None]))
        self.assertFalse(vr.rejected_within_change(blocks, [1, 2]))
        self.assertFalse(vr.rejected_within_change(blocks, [1]))
        self.assertIsNone(vr.rejected_within_change(blocks, [None, None]))
        self.assertIsNone(vr.rejected_within_change(blocks, None))

    def test_a_wrong_timed_id_count_voids_the_trial(self):
        trial = fake_trial("verify-4")
        trial["timed_ids"] = len(trial["ids"])
        self.assertTrue(any("timed ids" in d for d in vr.ledger_defects(trial)))

    def test_id_counts_and_how_generation_ended_must_agree(self):
        ids = reference_ids(10) + [vr.STOP_IDS[0]]
        trial = fake_trial(vr.SEQUENTIAL, ids=ids, generate=vr.GENERATE)
        self.assertEqual(trial["id_counts"]["ended_by"], "stop id")
        self.assertEqual(vr.id_count_defects(trial), [])
        self.assertEqual(vr.summarize(trial)["timed_ids"], 10)
        trial["id_counts"]["ended_by"] = "generation limit"
        self.assertTrue(any("ended by" in d for d in vr.id_count_defects(trial)))
        short = fake_trial(vr.SEQUENTIAL, ids=reference_ids(10), generate=vr.GENERATE)
        self.assertIn("generation ended short of the limit without a stop id", vr.id_count_defects(short))
        past = fake_trial(vr.SEQUENTIAL, ids=[1, vr.STOP_IDS[1], 2], generate=vr.GENERATE)
        self.assertIn("an id after a stop id was emitted", vr.id_count_defects(past))
        trial = fake_trial("verify-4")
        trial["id_counts"]["prefill_produced"] = 0
        self.assertTrue(vr.ledger_defects(trial))
        self.assertEqual(vr.id_count_defects(fake_trial(vr.SEQUENTIAL, ids=[], generate=0)), [])
        nothing = fake_trial(vr.SEQUENTIAL, ids=[], generate=5)
        self.assertIn("generation ended short of the limit without a stop id", vr.id_count_defects(nothing))

    def test_bracket_stability_is_reported_against_the_gate(self):
        def bracket_ns(entry):
            return BASE_NS * 1.02 if entry["unit"] in (1, 2) and entry["bracket"] == "after" else BASE_NS
        observations = clean_run(bracket_ns=bracket_ns)
        report = vr.analyse(observations, vr.judge_units(observations))
        stability = report["bracket_stability"]
        self.assertEqual((stability["units_with_both_brackets"], stability["over_gate"]), (24, 2))
        self.assertEqual(stability["gate"], vr.GATE)
        self.assertEqual(stability["median"], 0.0)


class Execute(unittest.TestCase):
    """The driver's run loop end to end, with every process replaced by a fake."""

    def run_execute(self, fail_at=None, wrong_ids_at=None, stall_trial=None, kind="diagnostic",
                    reference_crash_at=None, relay_late_ns=None):
        import argparse
        import contextlib
        import tempfile
        import verify_remote as driver

        folder = Path(tempfile.mkdtemp())
        model = folder / "model"
        model.mkdir()
        for name in ("index.json", "system_graph.json"):
            (model / name).write_text("{}")
        out = folder / "out"
        out.mkdir()
        clock = {"t": 0.0, "count": 0}

        def run_trial(root, model_, out_, env, entry, urls):
            if fail_at is not None and clock["count"] == fail_at:
                raise RuntimeError("trial process exited 101")
            clock["count"] += 1
            ids = reference_ids(entry["generate"])
            if wrong_ids_at is not None and clock["count"] - 1 == wrong_ids_at:
                ids[3] += 1
            if reference_crash_at is not None and entry["role"] == "reference" \
                    and clock["count"] - 1 == reference_crash_at:
                raise RuntimeError("reference process exited 101")
            return fake_trial(entry["arm"], ids=ids, corrupt=entry["corrupt"], generate=entry["generate"],
                              accept=0 if entry["corrupt"] != "none" else 2), relay

        # Prefill lateness far over the gate in every trial: it must not count.
        relay = None if relay_late_ns is None else fake_relay(prefill_late_ns=100 * BASE_NS,
                                                              generation_late_ns=relay_late_ns)

        def timed_trial(root, model_, out_, env, entry, urls):
            trial, relay = run_trial(root, model_, out_, env, entry, urls)
            start = clock["t"]
            clock["t"] += 60.0
            return trial, relay, {"started_utc": "x", "ended_utc": "y",
                                  "started_monotonic": start, "ended_monotonic": start + 59}

        class Watch:
            def __init__(self):
                self.stalls = [] if stall_trial is None else [
                    {"start_monotonic": 60.0 * stall_trial + 5, "end_monotonic": 60.0 * stall_trial + 8,
                     "seconds": 3.0}]

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        class Sampler:
            samples = 0

            def __init__(self, path):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        replaced = {"timed_trial": timed_trial, "workers": lambda *a, **k: contextlib.nullcontext(["http://w"]),
                    "sha": lambda path: "0" * 64}
        saved = {name: getattr(driver, name) for name in replaced}
        saved_other = (driver.d2.StallWatch, driver.ht.Sampler, driver.ht.spotlight_starts,
                       driver.b1.provenance, driver.lp.host_state)
        try:
            for name, value in replaced.items():
                setattr(driver, name, value)
            driver.d2.StallWatch = Watch
            driver.ht.Sampler = Sampler
            driver.ht.spotlight_starts = lambda start, end: {"per_minute": {}}
            driver.b1.provenance = lambda root, sources: {}
            driver.lp.host_state = lambda: {"load_average": [0, 0, 0], "busy_processes": []}
            args = argparse.Namespace(frames_qualification=[], link2_qualification=[], peer_handshake=None,
                                      diagnostic="unit test",
                                      continue_unsettled=False, port=0)
            report = driver.execute(args, Path(driver.__file__).resolve().parents[2], model, out, kind)
        finally:
            for name, value in saved.items():
                setattr(driver, name, value)
            (driver.d2.StallWatch, driver.ht.Sampler, driver.ht.spotlight_starts,
             driver.b1.provenance, driver.lp.host_state) = saved_other
        files = {p.name for p in out.iterdir()}
        trials = json.loads((out / "trials.json").read_text()) if (out / "trials.json").exists() else []
        return report, files, trials

    # Warm-up settles at two trials, then four references: the schedule starts at call 6.
    BEFORE_SCHEDULE = 2 + len(vr.WORKLOADS)

    def test_a_clean_diagnostic_writes_everything_and_reads_the_controls(self):
        report, files, trials = self.run_execute()
        self.assertTrue(report["complete"], report.get("stopped"))
        self.assertEqual((report["valid_units"], report["void_units"]), (24, 0))
        self.assertEqual(len(trials), 24 * 6 + 3)
        self.assertTrue({"manifest.json", "trials.json", "analysis.json", "verification.json", "warm.json",
                         "references.json", "suspensions.json", "spotlight.json"} <= files)
        self.assertEqual(len(report["controls"]), 3)
        self.assertTrue(all(c["acceptance_fell"] for c in report["controls"]))

    def test_ids_that_leave_the_reference_stop_the_run(self):
        report, files, trials = self.run_execute(wrong_ids_at=self.BEFORE_SCHEDULE + 3)
        self.assertFalse(report["complete"])
        self.assertIn("ids differ from the sequential reference at position 3", report["stopped"])
        self.assertEqual(len(trials), 3)
        self.assertIn("nothing is analysed", report["note"])

    def test_any_failure_stops_the_run_and_still_writes_every_record(self):
        report, files, trials = self.run_execute(fail_at=self.BEFORE_SCHEDULE + 30)
        self.assertFalse(report["complete"])
        self.assertIn("RuntimeError: trial process exited 101", report["stopped"])
        self.assertEqual(len(trials), 30)
        self.assertEqual(len(report["outcomes"]["not_run"]), 24 * 6 + 3 - 30)
        self.assertTrue({"analysis.json", "verification.json", "failure.txt"} <= files)

    def test_a_stall_voids_the_unit_it_falls_in(self):
        stalled = self.BEFORE_SCHEDULE + 13
        report, _, trials = self.run_execute(stall_trial=stalled)
        unit = trials[13]["unit"]
        self.assertEqual([int(u) for u in report["outcomes"]["void_units"]], [unit])

    def test_the_admissible_run_is_refused_while_a_draft(self):
        import contextlib
        import io
        import verify_remote as driver
        err = io.StringIO()
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(err):
            driver.main(["--admissible", "--model", "m", "--out", "o"])
        self.assertIn("is a draft; it admits no run until it is frozen", err.getvalue())

    def test_a_crash_part_way_through_the_references_keeps_those_taken(self):
        report, files, _ = self.run_execute(reference_crash_at=2 + 2)
        self.assertIn("reference process exited 101", report["stopped"])
        self.assertTrue({"references.json", "references_run.json", "failure.txt"} <= files)

    def test_a_late_relay_in_generation_voids_its_units(self):
        # Generation lateness above 1% of every trial's own timed generation.
        report, _, trials = self.run_execute(relay_late_ns=10 * BASE_NS)
        self.assertEqual(report["valid_units"], 0)
        self.assertTrue(all(not t["relay"]["relay_on_schedule"] for t in trials))
        # Generation lateness well under 1%; prefill lateness far over it.
        report, _, trials = self.run_execute(relay_late_ns=1000)
        self.assertEqual(report["void_units"], 0)


class RelayGate(unittest.TestCase):
    """Revision 2: only relay lateness after the generation-boundary mark is gated."""

    def setUp(self):
        self.trial = fake_trial("verify-8")
        self.wall = self.trial["wall_ns"]

    def check(self, relay, trial=None):
        return vr.relay_check(relay, self.trial if trial is None else trial)

    def test_any_prefill_lateness_cannot_affect_the_verdict(self):
        for prefill in (0, self.wall, 1000 * self.wall):
            gate = self.check(fake_relay(prefill_late_ns=prefill, generation_late_ns=self.wall // 200))
            self.assertTrue(gate["relay_on_schedule"], prefill)
            self.assertEqual(gate["mark_defects"], [])
        gate = self.check(fake_relay(prefill_late_ns=1000 * self.wall))
        self.assertGreater(gate["whole_trial"]["share_of_wall"], 100)  # kept, as a diagnostic

    def test_generation_lateness_over_one_percent_fails_and_exactly_one_percent_passes(self):
        exactly = self.check(fake_relay(generation_late_ns=self.wall // 100))
        self.assertEqual(exactly["generation_late_sum_ns"], self.wall // 100)
        self.assertTrue(exactly["relay_on_schedule"])
        over = self.check(fake_relay(generation_late_ns=self.wall // 100 + 4))
        self.assertFalse(over["relay_on_schedule"])
        self.assertEqual(over["mark_defects"], [])
        reasons = vr.trial_defects(dict(ledger_defects=[], frame_defects=[], relay=over))
        self.assertEqual(reasons, ["relay behind schedule in generation"])

    def test_the_generation_split_is_closing_minus_snapshot(self):
        gate = self.check(fake_relay(prefill_late_ns=4000, generation_late_ns=800, chunks=(10, 6)))
        self.assertEqual(gate["generation_late_sum_ns"], 800)
        self.assertEqual(gate["generation_chunks"], 12)

    def assert_void(self, relay, phrase, trial=None):
        gate = self.check(relay, trial)
        self.assertFalse(gate["relay_on_schedule"])
        self.assertTrue(any(phrase in d for d in gate["mark_defects"]), gate["mark_defects"])
        reasons = vr.trial_defects(dict(ledger_defects=[], frame_defects=[], relay=gate))
        self.assertTrue(reasons and all(r.startswith("relay mark: ") for r in reasons), reasons)

    def test_a_trial_that_sent_no_mark_is_void(self):
        trial = dict(self.trial, relay_mark=None)
        self.assert_void(fake_relay(), "sent no mark", trial)

    def test_an_unanswered_or_delayed_mark_is_void(self):
        trial = dict(self.trial, relay_mark=dict(self.trial["relay_mark"], answer=None, error="timed out"))
        self.assert_void(fake_relay(), "never answered", trial)

    def test_a_mark_the_relay_did_not_record_is_void(self):
        relay = fake_relay()
        relay["marks"] = []
        self.assert_void(relay, "recorded no mark")

    def test_a_duplicated_mark_is_void(self):
        relay = fake_relay()
        relay["marks"].append(dict(relay["marks"][1], mark=2))
        self.assert_void(relay, "recorded 2 marks")

    def test_a_refused_mark_is_void_with_its_reasons(self):
        relay = fake_relay(answer="refused", reasons=["connection 0: 2 chunks queued on the link"])
        trial = dict(self.trial, relay_mark=dict(self.trial["relay_mark"], answer="refused 1 connection 0: ..."))
        self.assert_void(relay, "chunks queued on the link", trial)

    def test_an_answer_that_disagrees_with_the_relay_is_void(self):
        trial = dict(self.trial, relay_mark=dict(self.trial["relay_mark"], answer="ok 7"))
        self.assert_void(fake_relay(), "is not the relay's", trial)

    def test_a_summary_that_does_not_name_the_boundary_connections_is_void(self):
        relay = fake_relay()
        del relay["marks"][1]["accepted_before"]
        self.assert_void(relay, "which connections came before it")

    def test_a_missing_snapshot_is_void(self):
        relay = fake_relay()
        relay["marks"] = relay["marks"][1:]
        self.assert_void(relay, "snapshots for connections []")

    def test_a_closing_tally_below_its_snapshot_is_void(self):
        relay = fake_relay(prefill_late_ns=4000)
        relay["stats"][1] = dict(relay["stats"][1], late_sum_ns=0)
        self.assert_void(relay, "closing late_sum_ns below its snapshot")

    def test_a_connection_opened_after_the_mark_counts_whole_and_one_closed_before_not_at_all(self):
        relay = fake_relay(generation_late_ns=400)
        relay["marks"][1]["accepted_before"] = 2
        relay["marks"][1]["connections"] = [1]
        relay["marks"][0]["connection"] = 1
        for record in relay["stats"]:
            record["connection"] = 1
        # Connection 0 closed before the mark; connection 2 opened after it.
        for connection, late in ((0, 10 ** 12), (2, 300)):
            for direction in ("request", "response"):
                relay["stats"].append({"connection": connection, "direction": direction, "bytes": 1,
                                       "chunks": 1, "late_sum_ns": late, "late_max_ns": late})
        gate = self.check(relay)
        self.assertEqual(gate["generation_late_sum_ns"], 400 + 600)


class RelayQualification(unittest.TestCase):
    """The admissible run needs two distinct passing LINK-2 qualifying receipts for the relay it runs."""
    RELAY = "b" * 64

    def receipt(self, started, relay=None, **changes):
        import link_pair_model as pm
        points = [{"point": p, "passes": True} for p in pm.frozen_points()]
        receipt = {"schema": pm.SCHEMA, "protocol": pm.REVISION, "kind": "qualifying",
                   "counts_toward_qualification": True, "frozen": pm.frozen_parameters(),
                   "binary_sha256": {"v3_link_shaper": relay or self.RELAY}, "run_passes": True,
                   "points": points, "host_before": {"utc": started}}
        receipt.update(changes)
        return receipt

    def test_two_distinct_passing_receipts_for_this_relay_qualify(self):
        import verify_remote as driver
        driver.check_link2_qualification([self.receipt("t1"), self.receipt("t2")], self.RELAY)

    def test_one_receipt_or_the_same_run_twice_does_not(self):
        import verify_remote as driver
        for receipts in ([self.receipt("t1")], [self.receipt("t1"), self.receipt("t1")]):
            with self.assertRaises(ValueError):
                driver.check_link2_qualification(receipts, self.RELAY)

    def test_receipts_for_another_relay_build_do_not(self):
        import verify_remote as driver
        receipts = [self.receipt("t1", relay="7" * 64), self.receipt("t2", relay="7" * 64)]
        with self.assertRaisesRegex(ValueError, "different relay build"):
            driver.check_link2_qualification(receipts, self.RELAY)

    def test_a_diagnostic_does_not(self):
        import verify_remote as driver
        receipts = [self.receipt("t1"), self.receipt("t2", kind="diagnostic")]
        with self.assertRaises(ValueError):
            driver.check_link2_qualification(receipts, self.RELAY)


PINNED_UNIT_ORDER_SHA256 = "f3fe7765dfc0bc4977d33f95ed41af3ae9a24eab5daaba59790dafb4169fee0f"

if __name__ == "__main__":
    unittest.main()
