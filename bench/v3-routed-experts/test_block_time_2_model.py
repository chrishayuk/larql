"""Tests for BLOCK-TIME-2's schedule, pair validity and analysis; nothing here starts a process."""
from collections import Counter
import unittest

import block_time_2_model as b2
import block_time_model as bt
from test_block_time_model import BASE_NS, HASHES, trial

WORKER_NS = 4_000_000        # per row, inside the worker
COORDINATOR_NS = 20_000_000  # per position, outside the exchanges
RELAY_ON_TIME = [{"late_sum_ns": 0, "late_max_ns": 0, "chunks": 10, "bytes": 1}]


def record(entry, extra_ns=0.0, exchange_extra_ns=0.0):
    """A trial that obeys the exchange model exactly, plus `extra_ns` per position of
    coordinator time and `exchange_extra_ns` per position spread over its exchanges."""
    width = bt.ARMS[entry["arm"]] or 1
    run = trial(entry["arm"], handler_ns=width * WORKER_NS)
    count = len(run["exchanges"])
    roundtrip = width * WORKER_NS + entry["rtt_ms"] * bt.NS_PER_MS \
        + exchange_extra_ns * bt.MEASURED_POSITIONS / count
    for exchange in run["exchanges"]:
        exchange["roundtrip_ns"] = roundtrip
    run["wall_ns"] = count * roundtrip + bt.MEASURED_POSITIONS * (COORDINATOR_NS + extra_ns)
    return run


def run_schedule(extra=lambda entry: 0.0, relay=lambda entry: RELAY_ON_TIME, damage=lambda entry, run: None,
                 exchange_extra=lambda entry: 0.0):
    """Every scheduled trial through BLOCK-TIME-1's per-trial gates, then BLOCK-TIME-2's judging."""
    import block_time as b1
    observations = []
    for entry in b2.schedule():
        run = record(entry, extra(entry), exchange_extra(entry))
        damage(entry, run)
        observations.append(b1.observe(entry, run, relay(entry), HASHES, BASE_NS))
    return observations, b2.judge_pairs(observations)


class Schedule(unittest.TestCase):
    def test_twenty_pairs_of_four_trials(self):
        trials = b2.schedule()
        self.assertEqual(len(trials), len(b2.ARMS) * b2.PAIRS_PER_ARM * 4)
        self.assertEqual(Counter(t["pair"] for t in trials), {p: 4 for p in range(20)})

    def test_every_pair_is_bracket_two_delays_of_one_arm_bracket(self):
        trials = b2.schedule()
        for pair in range(20):
            members = [t for t in trials if t["pair"] == pair]
            self.assertEqual([m["role"] for m in members], ["bracket", "arm", "arm", "bracket"])
            self.assertEqual([m["bracket"] for m in members], ["before", None, None, "after"])
            for bracket in (members[0], members[3]):
                self.assertEqual((bracket["arm"], bracket["rtt_ms"]), (bt.SEQUENTIAL, 0.0))
            self.assertEqual(members[1]["arm"], members[2]["arm"])
            self.assertEqual({members[1]["rtt_ms"], members[2]["rtt_ms"]}, {5.0, 20.0})

    def test_brackets_are_never_shared_between_pairs(self):
        trials = b2.schedule()
        for index, entry in enumerate(trials):
            if entry["role"] == bt.ROLE_BRACKET and entry["bracket"] == "after" and index + 1 < len(trials):
                following = trials[index + 1]
                self.assertEqual(following["bracket"], "before")
                self.assertEqual(following["pair"], entry["pair"] + 1)

    def test_each_arm_has_four_pairs_two_in_each_delay_order(self):
        units = b2.pair_units()
        for arm in b2.ARMS:
            mine = [u for u in units if u["arm"] == arm]
            self.assertEqual(len(mine), b2.PAIRS_PER_ARM, arm)
            # The protocol's pattern, written out: not read back from the constant.
            self.assertEqual([u["order"] for u in mine], ["5->20", "20->5", "20->5", "5->20"], arm)
            self.assertEqual(Counter(u["order"] for u in mine), {b2.LOW_FIRST: 2, b2.HIGH_FIRST: 2}, arm)

    def test_the_order_written_matches_the_delays_run(self):
        trials = b2.schedule()
        for pair in range(20):
            first, second = [t for t in trials if t["pair"] == pair and t["role"] == bt.ROLE_ARM]
            expected = (5.0, 20.0) if first["order"] == b2.LOW_FIRST else (20.0, 5.0)
            self.assertEqual((first["rtt_ms"], second["rtt_ms"]), expected)

    def test_every_round_has_exactly_one_pair_of_each_arm(self):
        units = b2.pair_units()
        self.assertEqual(len(units), b2.ROUNDS * len(b2.ARMS))
        for round_index in range(b2.ROUNDS):
            block = units[round_index * len(b2.ARMS):(round_index + 1) * len(b2.ARMS)]
            self.assertEqual({u["round"] for u in block}, {round_index})
            self.assertEqual(sorted(u["arm"] for u in block), sorted(b2.ARMS))

    def test_delay_order_follows_the_round_a_b_b_a(self):
        for unit in b2.pair_units():
            self.assertEqual(unit["order"], ["5->20", "20->5", "20->5", "5->20"][unit["round"]])

    def test_the_schedule_is_deterministic_and_shuffled_within_rounds(self):
        self.assertEqual(b2.schedule(), b2.schedule())
        rounds = [[u["arm"] for u in b2.pair_units() if u["round"] == r] for r in range(b2.ROUNDS)]
        self.assertTrue(any(order != list(b2.ARMS) for order in rounds))
        self.assertGreater(len({tuple(order) for order in rounds}), 1)

    def test_the_seed_and_the_schedule_it_gives_are_the_registered_ones(self):
        # Changing the seed, the round composition or the delay pattern is a
        # new revision; this pins the schedule reviewed.
        self.assertEqual(b2.SHUFFLE_SEED, 20261115)
        self.assertEqual(b2.ROUNDS, 4)
        rounds = [[u["arm"] for u in b2.pair_units() if u["round"] == r] for r in range(b2.ROUNDS)]
        self.assertEqual(rounds, [
            ["block-8", "block-4", "block-1", "sequential", "block-2"],
            ["sequential", "block-4", "block-8", "block-1", "block-2"],
            ["block-8", "block-2", "block-4", "sequential", "block-1"],
            ["block-4", "block-2", "sequential", "block-1", "block-8"]])

    def test_only_the_two_qualified_delays_and_the_zero_delay_bracket_are_run(self):
        for entry in b2.schedule():
            allowed = {0.0} if entry["role"] == bt.ROLE_BRACKET else {5.0, 20.0}
            self.assertIn(entry["rtt_ms"], allowed)
            self.assertTrue(entry["profiled"])

    def test_labels_are_unique_and_ordered(self):
        labels = [t["label"] for t in b2.schedule()]
        self.assertEqual(labels, [f"trial-{i:03d}" for i in range(len(labels))])


class PairValidity(unittest.TestCase):
    def test_a_clean_run_has_every_pair_valid(self):
        observations, verdicts = run_schedule()
        self.assertTrue(all(v["valid"] for v in verdicts))
        self.assertTrue(all(o["valid"] for o in observations))

    def test_drifting_brackets_void_only_their_own_pair(self):
        def extra(entry):
            # Pair 7's closing bracket is 3% slower.
            return 0.03 * BASE_NS if entry["pair"] == 7 and entry["bracket"] == "after" else 0.0
        observations, verdicts = run_schedule(extra=extra)
        self.assertEqual([v["pair"] for v in verdicts if not v["valid"]], [7])
        self.assertTrue(any("brackets differ" in r for r in verdicts[7]["void_reasons"]))
        for observation in observations:
            self.assertEqual(observation["valid"], observation["pair"] != 7, observation["label"])

    def test_drift_within_the_gate_is_valid(self):
        def extra(entry):
            return 0.002 * BASE_NS if entry["bracket"] == "after" else 0.0
        _, verdicts = run_schedule(extra=extra)
        self.assertTrue(all(v["valid"] for v in verdicts))

    def test_the_gate_is_one_percent_on_either_side(self):
        for share, valid in ((0.8 * b2.GATE, True), (1.25 * b2.GATE, False)):
            def extra(entry, share=share):
                return share * BASE_NS if entry["pair"] == 0 and entry["bracket"] == "after" else 0.0
            _, verdicts = run_schedule(extra=extra)
            self.assertEqual(verdicts[0]["valid"], valid, share)
        self.assertEqual(b2.GATE, 0.01)

    def test_a_frame_defect_or_late_relay_in_any_member_voids_the_pair_by_name(self):
        def damage(entry, run):
            if entry["pair"] == 3 and entry["role"] == bt.ROLE_ARM and entry["rtt_ms"] == 20.0:
                run["frames"]["measured"]["single-row"] += 1

        def relay(entry):
            if entry["pair"] == 11 and entry["bracket"] == "before":
                return [{"late_sum_ns": 10**12, "late_max_ns": 10**9, "chunks": 10, "bytes": 1}]
            return RELAY_ON_TIME
        _, verdicts = run_schedule(damage=damage, relay=relay)
        self.assertEqual([v["pair"] for v in verdicts if not v["valid"]], [3, 11])
        self.assertTrue(any("frames" in r for r in verdicts[3]["void_reasons"]))
        self.assertTrue(any("relay behind schedule" in r for r in verdicts[11]["void_reasons"]))

    def test_an_incomplete_pair_is_void(self):
        observations, _ = run_schedule()
        verdicts = b2.judge_pairs(observations[:-1])
        self.assertIn("incomplete pair", verdicts[-1]["void_reasons"])

    def test_a_parity_failure_stops_the_run(self):
        import block_time as b1
        entry = b2.schedule()[1]
        changed = dict(record(entry), logits_sha256=["f" * 64] + HASHES[1:])
        with self.assertRaises(b1.Stop):
            b1.observe(entry, changed, RELAY_ON_TIME, HASHES, BASE_NS)


class WithinPairConsistency(unittest.TestCase):
    """The arm's two trials in a pair must agree on coordinator and worker execution time."""

    PAIR = 4

    def coordinator_case(self, share):
        def extra(entry):
            if entry["pair"] == self.PAIR and entry["role"] == bt.ROLE_ARM and entry["rtt_ms"] == 20.0:
                return share * COORDINATOR_NS
            return 0.0
        return run_schedule(extra=extra)[1][self.PAIR]

    def execution_case(self, share):
        def damage(entry, run):
            if entry["pair"] == self.PAIR and entry["role"] == bt.ROLE_ARM and entry["rtt_ms"] == 20.0:
                for record_ in run["worker_records"]:
                    record_["worker"]["execute_ns"] = record_["worker"]["execute_ns"] * (1 + share)
        return run_schedule(damage=damage)[1][self.PAIR]

    def test_the_threshold_is_five_percent(self):
        self.assertEqual(b2.CONSISTENCY, 0.05)

    def test_coordinator_at_below_and_above_five_percent(self):
        for share, valid in ((0.05, True), (0.049, True), (0.051, False)):
            verdict = self.coordinator_case(share)
            self.assertAlmostEqual(verdict["within_pair_difference"]["coordinator"], share, places=9)
            self.assertEqual(verdict["valid"], valid, share)
            if not valid:
                self.assertTrue(any(r.startswith("coordinator time differs") for r in verdict["void_reasons"]))

    def test_worker_execution_at_below_and_above_five_percent(self):
        for share, valid in ((0.05, True), (0.049, True), (0.051, False)):
            verdict = self.execution_case(share)
            self.assertAlmostEqual(verdict["within_pair_difference"]["execution"], share, places=9)
            self.assertEqual(verdict["valid"], valid, share)
            if not valid:
                self.assertTrue(any(r.startswith("worker execution time differs") for r in verdict["void_reasons"]))

    def test_the_difference_is_symmetric_over_the_smaller_value(self):
        self.assertEqual(b2.relative_difference(100.0, 105.0), b2.relative_difference(105.0, 100.0))
        self.assertAlmostEqual(b2.relative_difference(100.0, 105.0), 0.05)

    def test_missing_zero_or_non_finite_measurements_fail(self):
        for a, b in ((None, 1.0), (0.0, 1.0), (1.0, float("nan")), (float("inf"), 1.0), (-1.0, 1.0)):
            self.assertIsNone(b2.relative_difference(a, b), (a, b))

        def damage(entry, run):
            if entry["pair"] == self.PAIR and entry["role"] == bt.ROLE_ARM and entry["rtt_ms"] == 5.0:
                run["worker_records"][0] = {"worker_profile_complete": False}
        observations, verdicts = run_schedule(damage=damage)
        verdict = verdicts[self.PAIR]
        self.assertFalse(verdict["valid"])
        self.assertIn("worker execution time missing, zero or not finite in an arm trial", verdict["void_reasons"])
        # A missing value is a rule-4 failure in the counts too, not silently skipped.
        failures = b2.analyse(observations, verdicts)["arms"][verdict["arm"]]["within_pair_failures"]
        self.assertEqual(failures["execution"], 1)

    def test_differences_are_reported_for_every_pair_valid_or_void(self):
        _, verdicts = run_schedule(extra=lambda e: 0.03 * BASE_NS if e["pair"] == 2 and e["bracket"] == "after" else 0.0)
        self.assertFalse(verdicts[2]["valid"])
        for verdict in verdicts:
            for name in ("coordinator", "execution"):
                self.assertIsNotNone(verdict["within_pair_difference"][name], (verdict["pair"], name))

    def test_failures_are_counted_by_arm_and_by_delay_order(self):
        observations, verdicts = run_schedule(extra=lambda e: 0.2 * COORDINATOR_NS
                                              if e["pair"] == self.PAIR and e["role"] == bt.ROLE_ARM
                                              and e["rtt_ms"] == 20.0 else 0.0)
        report = b2.analyse(observations, verdicts)
        arm = verdicts[self.PAIR]["arm"]
        self.assertEqual(report["arms"][arm]["within_pair_failures"], {"coordinator": 1, "execution": 0, "of": 4})
        order = verdicts[self.PAIR]["order"]
        self.assertEqual(report["within_pair_failures_by_order"][order]["coordinator"], 1)

    def test_the_diagnostics_five_second_pause_fails_the_coordinator_gate(self):
        # Pair 0 of the recorded functional diagnostic: a 5 s SIGSTOP fell inside
        # its 20 ms trial and its brackets still agreed. Read from the record,
        # not reconstructed; the diagnostic's own verdict is not changed.
        import json
        from pathlib import Path
        record_path = (Path(__file__).with_name("link-results") / "block-time-2" / "20261115-01-diagnostic"
                       / "trials.json")
        members = [o for o in json.loads(record_path.read_text()) if o["pair"] == 0]
        brackets = [m for m in members if m["role"] == bt.ROLE_BRACKET]
        self.assertLessEqual(bt.bracket_drift(brackets[0]["summary"], brackets[1]["summary"]), bt.GATE)
        differences, reasons = b2.within_pair([m for m in members if m["role"] == bt.ROLE_ARM])
        self.assertGreater(differences["coordinator"], 2.0)
        self.assertLessEqual(differences["execution"], b2.CONSISTENCY)
        self.assertEqual(len(reasons), 1)
        self.assertTrue(reasons[0].startswith("coordinator time differs"))


class Suspension(unittest.TestCase):
    """Rule 5: a stall of the driver inside a trial voids that trial's pair."""

    def test_the_stall_rule_is_in_the_contract_and_pinned(self):
        self.assertEqual((b2.STALL_GAP_S, b2.STALL_TICK_S), (0.5, 0.1))
        self.assertEqual(b2.SLOPE_BAND, 0.10)

    def test_a_jump_in_either_clock_beyond_the_threshold_is_a_stall(self):
        self.assertIsNone(b2.stall_between((0.0, 100.0), (0.1, 100.1)))
        self.assertIsNone(b2.stall_between((0.0, 100.0), (0.5, 100.5)))
        found = b2.stall_between((0.0, 100.0), (5.2, 105.2))
        self.assertEqual((found["start_monotonic"], found["end_monotonic"]), (0.0, 5.2))
        self.assertAlmostEqual(found["seconds"], 5.2)
        # A system sleep can stop the monotonic clock but not the wall clock.
        sleep = b2.stall_between((10.0, 100.0), (10.1, 160.0))
        self.assertIsNotNone(sleep)

    def test_a_backward_wall_clock_step_neither_hides_nor_invents_a_stall(self):
        self.assertIsNone(b2.stall_between((0.0, 100.0), (0.1, 40.0)))
        found = b2.stall_between((0.0, 100.0), (3.0, 40.0))
        self.assertEqual(b2.stalls_within([found], 1.0, 2.0), [found])

    def test_overlap_is_judged_on_the_monotonic_span_including_its_edges(self):
        stall = {"start_monotonic": 10.0, "end_monotonic": 12.0, "seconds": 2.0}
        self.assertEqual(b2.stalls_within([stall], 12.0, 20.0), [stall])  # touches the start
        self.assertEqual(b2.stalls_within([stall], 0.0, 10.0), [stall])   # touches the end
        self.assertEqual(b2.stalls_within([stall], 12.01, 20.0), [])
        self.assertEqual(b2.stalls_within([stall], 0.0, 9.99), [])
        point = {"start_monotonic": 15.0, "end_monotonic": 15.0, "seconds": 60.0}  # a sleep
        self.assertEqual(b2.stalls_within([point], 14.0, 16.0), [point])

    def test_the_watch_records_a_stall_from_its_clocks(self):
        import time
        import block_time_2 as driver
        ticks = iter([(0.0, 100.0), (0.1, 100.1), (6.0, 106.0)] + [(6.1 + i / 10, 106.1 + i / 10)
                                                                    for i in range(10**6)])
        with driver.StallWatch(tick=0.001, clocks=lambda: next(ticks)) as watch:
            time.sleep(0.05)
        self.assertEqual(len(watch.stalls), 1)
        self.assertEqual(watch.stalls[0]["start_monotonic"], 0.1)

    def test_a_trial_overlapping_a_stall_voids_its_pair_by_name(self):
        observations, _ = run_schedule()
        for index, observation in enumerate(observations):
            observation["started_monotonic"], observation["ended_monotonic"] = 60.0 * index, 60.0 * index + 59
        target = observations[9]
        stall = {"start_monotonic": target["started_monotonic"] + 10,
                 "end_monotonic": target["started_monotonic"] + 15, "seconds": 5.0}
        b2.attach_stalls(observations, [stall])
        verdicts = b2.judge_pairs(observations)
        self.assertEqual([v["pair"] for v in verdicts if not v["valid"]], [target["pair"]])
        self.assertIn(f"{target['label']}: driver suspended 5.0 s during the trial",
                      verdicts[target["pair"]]["void_reasons"])
        for member in observations:
            if member["pair"] == target["pair"]:
                # Every member reads void, with the pair's reasons, not only its own.
                self.assertFalse(member["valid"])
                self.assertEqual(member["void_reasons"], verdicts[target["pair"]]["void_reasons"])


class Execute(unittest.TestCase):
    """The driver's run loop end to end, with every process replaced by a fake."""

    def run_execute(self, fail_at=None, stall_trial=None, busy_at_end=(), kind="diagnostic", spotlight=None):
        import argparse
        import contextlib
        import json
        import tempfile
        from pathlib import Path
        import block_time_2 as driver

        folder = Path(tempfile.mkdtemp())
        model = folder / "model"
        model.mkdir()
        for name in ("index.json", "system_graph.json"):
            (model / name).write_text("{}")
        out = folder / "out"
        out.mkdir()
        clock = {"t": 0.0, "count": 0}

        def timed_trial(root, model_, ids, out_, env, entry, urls):
            if fail_at is not None and clock["count"] == fail_at:
                raise RuntimeError("trial process exited 101")
            clock["count"] += 1
            start = clock["t"]
            clock["t"] += 60.0
            return record(entry), RELAY_ON_TIME, {"started_utc": "x", "ended_utc": "y",
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

        replaced = {"timed_trial": timed_trial, "StallWatch": Watch,
                    "workers": lambda *a, **k: contextlib.nullcontext(["http://worker"]),
                    "warm_up": lambda *a, **k: (HASHES, BASE_NS), "sha": lambda path: "0" * 64}
        saved = {name: getattr(driver, name) for name in replaced}
        saved_ht = (driver.ht.Sampler, driver.ht.spotlight_starts)
        saved_b1 = driver.b1.provenance
        saved_host = driver.lp.host_state
        saved_check = driver.bm.check_qualification
        try:
            driver.bm.check_qualification = lambda receipts, relay_sha: None
            for name, value in replaced.items():
                setattr(driver, name, value)
            driver.ht.Sampler = Sampler
            driver.ht.spotlight_starts = spotlight or (lambda start, end: {"per_minute": {}})
            driver.b1.provenance = lambda root, sources: {}
            driver.lp.host_state = lambda: {"load_average": [0, 0, 0], "busy_processes": list(busy_at_end)}
            ids = Path(driver.__file__).with_name("link-results") / "control" / "ids.json"
            args = argparse.Namespace(frames_qualification=[], ids=ids, peer_handshake=None,
                                      diagnostic="unit test", continue_unsettled=False, port=0)
            report = driver.execute(args, Path(driver.__file__).resolve().parents[2], model, out, kind)
        finally:
            for name, value in saved.items():
                setattr(driver, name, value)
            driver.ht.Sampler, driver.ht.spotlight_starts = saved_ht
            driver.b1.provenance = saved_b1
            driver.lp.host_state = saved_host
            driver.bm.check_qualification = saved_check
        files = {p.name for p in out.iterdir()}
        trials = json.loads((out / "trials.json").read_text())
        return report, files, trials, out

    def test_a_clean_run_writes_everything_and_analyses_all_pairs(self):
        report, files, trials, _ = self.run_execute()
        self.assertTrue(report["complete"])
        self.assertEqual((report["valid_pairs"], report["void_pairs"]), (20, 0))
        self.assertTrue({"manifest.json", "trials.json", "analysis.json", "verification.json",
                         "suspensions.json", "spotlight.json"} <= files)
        self.assertEqual(len(trials), 80)

    def test_stalls_are_attached_before_pairs_are_judged(self):
        report, _, trials, _ = self.run_execute(stall_trial=13)
        pair = trials[13]["pair"]
        self.assertEqual([int(p) for p in report["outcomes"]["void_pairs"]], [pair])
        self.assertTrue(any("driver suspended 3.0 s" in r for r in report["outcomes"]["void_pairs"][str(pair)]))

    def test_any_failure_stops_the_run_and_still_writes_every_record(self):
        report, files, trials, out = self.run_execute(fail_at=30)
        self.assertFalse(report["complete"])
        self.assertIn("RuntimeError: trial process exited 101", report["stopped"])
        self.assertEqual(len(trials), 30)
        self.assertEqual(len(report["outcomes"]["not_run"]), 50)
        self.assertTrue({"analysis.json", "verification.json", "suspensions.json", "failure.txt"} <= files)
        self.assertIn("nothing is analysed", report["note"])

    def test_a_failing_spotlight_query_loses_no_record(self):
        def spotlight(start, end):
            raise KeyboardInterrupt
        report, files, trials, out = self.run_execute(spotlight=spotlight)
        self.assertTrue(report["complete"])
        self.assertTrue({"trials.json", "analysis.json", "suspensions.json", "verification.json",
                         "spotlight.json"} <= files)
        import json
        self.assertIn("KeyboardInterrupt", json.loads((out / "spotlight.json").read_text())["error"])

    def test_every_arm_trial_records_its_largest_exchange_remainder(self):
        _, _, trials, _ = self.run_execute()
        for trial_ in trials:
            self.assertEqual(trial_["exchange_remainder"]["largest_ns"], 0, trial_["label"])

    def test_a_busy_machine_at_the_end_means_an_admissible_run_does_not_count(self):
        quiet, _, _, _ = self.run_execute(kind="admissible")
        self.assertTrue(quiet["counts"])
        busy, _, _, _ = self.run_execute(kind="admissible", busy_at_end=["rustc"])
        self.assertTrue(busy["complete"])
        self.assertFalse(busy["quiet_at_end"])
        self.assertFalse(busy["counts"])


class Analysis(unittest.TestCase):
    def test_a_run_on_the_model_recovers_every_predicted_slope(self):
        observations, verdicts = run_schedule()
        report = b2.analyse(observations, verdicts)
        self.assertEqual((report["valid_pairs"], report["void_pairs"]), (20, 0))
        for arm, entry in report["arms"].items():
            predicted = bt.exchanges_per_position(arm)
            self.assertEqual(entry["predicted"], predicted)
            self.assertEqual(entry["pairs_valid"], 4)
            self.assertAlmostEqual(entry["slope"], predicted, places=6, msg=arm)
            for slope in entry["valid_slopes"]:
                self.assertAlmostEqual(slope, predicted, places=6, msg=arm)
            self.assertTrue(entry["follows_exchange_model"], arm)
            self.assertAlmostEqual(entry["order"]["difference"], 0.0, places=6)

    def test_three_valid_pairs_report_their_mean_and_two_report_missing(self):
        block8 = [u["arm"] for u in b2.pair_units()]
        pairs8 = [i for i, arm in enumerate(block8) if arm == "block-8"]
        pairs4 = [i for i, arm in enumerate(block8) if arm == "block-4"]
        spoiled = {pairs8[0], pairs4[0], pairs4[1]}

        def extra(entry):
            return 0.05 * BASE_NS if entry["pair"] in spoiled and entry["bracket"] == "after" else 0.0
        observations, verdicts = run_schedule(extra=extra)
        arms = b2.analyse(observations, verdicts)["arms"]
        self.assertEqual(arms["block-8"]["pairs_valid"], 3)
        self.assertAlmostEqual(arms["block-8"]["slope"], 3.0, places=6)
        self.assertEqual(arms["block-4"]["pairs_valid"], 2)
        self.assertIsNone(arms["block-4"]["slope"])
        self.assertIsNone(arms["block-4"]["follows_exchange_model"])
        self.assertIn("2 of 4 pairs valid", arms["block-4"]["note"])

    def test_void_pairs_are_reported_but_never_pooled(self):
        target = next(i for i, u in enumerate(b2.pair_units()) if u["arm"] == "block-2")

        def extra(entry):
            # The void pair's arm trials carry a huge, wrong slope, and its bracket drifts.
            if entry["pair"] != target:
                return 0.0
            if entry["role"] == bt.ROLE_ARM and entry["rtt_ms"] == 20.0:
                return 50 * bt.NS_PER_MS
            return 0.05 * BASE_NS if entry["bracket"] == "after" else 0.0
        observations, verdicts = run_schedule(extra=extra)
        report = b2.analyse(observations, verdicts)
        void = next(p for p in report["pairs"] if p["pair"] == target)
        self.assertFalse(void["valid"])
        self.assertGreater(void["slope"], 12.0 + 3)  # still computed and shown
        self.assertAlmostEqual(report["arms"]["block-2"]["slope"], 12.0, places=6)
        self.assertEqual(report["arms"]["block-2"]["pairs_valid"], 3)

    def test_a_slope_outside_the_band_is_not_said_to_follow_the_model(self):
        def exchange_extra(entry):
            # block-8 pays 4.5 round trips per position at each delay instead of 3.
            if entry["role"] == bt.ROLE_ARM and entry["arm"] == "block-8":
                return 1.5 * entry["rtt_ms"] * bt.NS_PER_MS
            return 0.0
        observations, verdicts = run_schedule(exchange_extra=exchange_extra)
        entry = b2.analyse(observations, verdicts)["arms"]["block-8"]
        self.assertAlmostEqual(entry["slope"], 4.5, places=6)
        self.assertFalse(entry["follows_exchange_model"])

    def test_a_delay_order_effect_is_reported(self):
        def exchange_extra(entry):
            # Whichever delay runs second in a pair is 1 ms per position slower.
            if entry["role"] != bt.ROLE_ARM:
                return 0.0
            second = (entry["order"] == b2.LOW_FIRST) == (entry["rtt_ms"] == 20.0)
            return bt.NS_PER_MS if second else 0.0
        observations, verdicts = run_schedule(exchange_extra=exchange_extra)
        order = b2.analyse(observations, verdicts)["arms"]["sequential"]["order"]
        span = b2.DELAY_SPAN_NS
        self.assertAlmostEqual(order[b2.LOW_FIRST], 24 + bt.NS_PER_MS / span, places=6)
        self.assertAlmostEqual(order[b2.HIGH_FIRST], 24 - bt.NS_PER_MS / span, places=6)
        self.assertAlmostEqual(order["difference"], 2 * bt.NS_PER_MS / span, places=6)

    def test_worker_execution_is_reported_for_valid_and_all_pairs(self):
        observations, verdicts = run_schedule()
        execution = b2.analyse(observations, verdicts)["arms"]["block-8"]["worker_execution_ns"]
        for rtt in ("5.0", "20.0"):
            self.assertEqual(len(execution[rtt]["all_pairs"]["values"]), 4)
            self.assertEqual(execution[rtt]["valid_pairs"], execution[rtt]["all_pairs"])

    def test_no_speed_comparison_is_made(self):
        observations, verdicts = run_schedule()
        report = b2.analyse(observations, verdicts)
        self.assertNotIn("faster", report)
        self.assertIn("no speed comparison", report["reading"])


class SecondaryReports(unittest.TestCase):
    """What the analysis reports beside the registered slope; none of it is a statement."""

    def test_the_secondary_slope_keeps_pairs_that_only_rule_4_voided(self):
        target = [i for i, u in enumerate(b2.pair_units()) if u["arm"] == "block-4"][:2]

        def exchange_extra(entry):
            # The 20 ms trial of two block-4 pairs pays 0.9 extra round trips per position...
            if entry["pair"] in target and entry["role"] == bt.ROLE_ARM and entry["rtt_ms"] == 20.0:
                return 0.9 * b2.DELAY_SPAN_NS
            return 0.0

        def extra(entry):
            # ...and its coordinator moves 10%, so rule 4 alone voids those pairs.
            if entry["pair"] in target and entry["role"] == bt.ROLE_ARM and entry["rtt_ms"] == 20.0:
                return 0.10 * COORDINATOR_NS
            return 0.0
        observations, verdicts = run_schedule(extra=extra, exchange_extra=exchange_extra)
        for pair in target:
            self.assertFalse(verdicts[pair]["valid"])
            self.assertTrue(verdicts[pair]["valid_without_rule_4"])
        entry = b2.analyse(observations, verdicts)["arms"]["block-4"]
        self.assertEqual(entry["pairs_valid"], 2)
        self.assertIsNone(entry["slope"])
        secondary = entry["secondary_without_rule_4"]
        self.assertEqual(secondary["pairs"], 4)
        shift = 0.9 + 0.10 * COORDINATOR_NS / b2.DELAY_SPAN_NS
        self.assertAlmostEqual(secondary["slope"], 6.0 + shift / 2, places=6)
        self.assertAlmostEqual(secondary["ratio"], (6.0 + shift / 2) / 6.0, places=6)

    def test_a_pair_void_by_another_rule_is_not_in_the_secondary_slope(self):
        target = next(i for i, u in enumerate(b2.pair_units()) if u["arm"] == "block-2")
        _, verdicts = run_schedule(extra=lambda e: 0.03 * BASE_NS
                                   if e["pair"] == target and e["bracket"] == "after" else 0.0)
        self.assertFalse(verdicts[target]["valid_without_rule_4"])

    def test_the_shift_rule_4_admits_is_the_tolerance_on_both_costs_over_the_span(self):
        observations, verdicts = run_schedule()
        for arm, entry in b2.analyse(observations, verdicts)["arms"].items():
            width = bt.ARMS[arm] or 1
            # Per position: coordinator, plus the worker's execution (half its handler) per exchange.
            execution = bt.exchanges_per_position(arm) * width * WORKER_NS / 2
            expected = b2.CONSISTENCY * (COORDINATOR_NS + execution) / b2.DELAY_SPAN_NS
            self.assertAlmostEqual(entry["rule_4_admits_shift"]["round_trips"], expected, places=9, msg=arm)
            self.assertAlmostEqual(entry["rule_4_admits_shift"]["share_of_predicted"],
                                   expected / bt.exchanges_per_position(arm), places=9, msg=arm)

    def test_the_shift_bound_uses_valid_pairs_only(self):
        target = next(i for i, u in enumerate(b2.pair_units()) if u["arm"] == "block-8")
        def extra(entry):
            # A void pair (its brackets drift) whose arm trials carry huge costs.
            if entry["pair"] != target:
                return 0.0
            if entry["role"] == bt.ROLE_ARM:
                return 100 * COORDINATOR_NS
            return 0.03 * BASE_NS if entry["bracket"] == "after" else 0.0
        observations, verdicts = run_schedule(extra=extra)
        self.assertFalse(verdicts[target]["valid"])
        clean_obs, clean_verdicts = run_schedule()
        bound = b2.analyse(observations, verdicts)["arms"]["block-8"]["rule_4_admits_shift"]
        clean = b2.analyse(clean_obs, clean_verdicts)["arms"]["block-8"]["rule_4_admits_shift"]
        self.assertAlmostEqual(bound["round_trips"], clean["round_trips"], places=9)

    def test_spread_counts_missing_values(self):
        self.assertEqual(b2.spread([3.0, None, 1.0, None])["missing"], 2)
        self.assertEqual(b2.spread([3.0, None, 1.0])["values"], [1.0, 3.0])
        self.assertEqual(b2.spread([None, None]), {"missing": 2, "values": []})
        self.assertIsNone(b2.spread([]))

    def test_the_largest_exchange_remainder_shows_a_stall_in_one_exchange(self):
        entry = next(e for e in b2.schedule() if e["role"] == bt.ROLE_ARM and e["arm"] == "block-8")
        run = record(entry)
        run["exchanges"][17]["roundtrip_ns"] += 500 * bt.NS_PER_MS
        found = b2.largest_exchange_remainder(run, entry["rtt_ms"])
        self.assertEqual(found["median_ns"], 0)
        self.assertEqual(found["largest_ns"], 500 * bt.NS_PER_MS)
        run["worker_records"][0] = {}
        self.assertIsNone(b2.largest_exchange_remainder(run, entry["rtt_ms"]))


class Pausing(unittest.TestCase):
    """Another session can hold the run between pairs; nothing inside a pair."""

    def test_the_run_may_hold_only_at_the_start_of_each_pair(self):
        import block_time_2 as driver
        points = [entry for entry in b2.schedule() if driver.is_pause_point(entry)]
        self.assertEqual([entry["pair"] for entry in points], list(range(20)))
        self.assertTrue(all(entry["bracket"] == "before" for entry in points))

    def test_no_pause_file_means_no_wait_and_no_record(self):
        import tempfile
        from pathlib import Path
        import block_time_2 as driver
        pauses = []
        with tempfile.TemporaryDirectory() as folder:
            driver.wait_while_paused(Path(folder), 3, pauses, sleep=self.fail)
        self.assertEqual(pauses, [])

    def test_a_pause_holds_until_the_file_is_removed_and_is_recorded(self):
        import tempfile
        from pathlib import Path
        import block_time_2 as driver
        pauses, polls = [], []
        clock = iter(["2026-11-15T12:00:00Z", "2026-11-15T12:10:00Z"])
        with tempfile.TemporaryDirectory() as folder:
            flag = Path(folder) / driver.PAUSE_FILE
            flag.touch()

            def sleep(seconds):
                polls.append(seconds)
                # The driver says it is holding while the file is there.
                self.assertTrue((Path(folder) / driver.PAUSED_FILE).exists())
                if len(polls) == 3:
                    flag.unlink()
            driver.wait_while_paused(Path(folder), 7, pauses, sleep=sleep, now=lambda: next(clock))
            self.assertFalse((Path(folder) / driver.PAUSED_FILE).exists())
        self.assertEqual(polls, [driver.PAUSE_POLL_S] * 3)
        self.assertEqual(pauses, [{"before_pair": 7, "paused_utc": "2026-11-15T12:00:00Z",
                                   "resumed_utc": "2026-11-15T12:10:00Z"}])


class Admission(unittest.TestCase):
    def test_a_draft_protocol_admits_no_run_and_starts_nothing(self):
        import contextlib
        import io
        import block_time_2 as driver

        def must_not_run(*args, **kwargs):
            raise AssertionError("the driver started work for a run it must refuse")
        saved = {name: getattr(driver, name) for name in ("workers", "execute", "timed_trial")}
        try:
            for name in saved:
                setattr(driver, name, must_not_run)
            message = io.StringIO()
            with self.assertRaises(SystemExit) as refused, contextlib.redirect_stderr(message):
                driver.main(["--admissible", "--model", "unused", "--out", "/nonexistent/never-created"])
        finally:
            for name, original in saved.items():
                setattr(driver, name, original)
        self.assertNotEqual(refused.exception.code, 0)
        self.assertIn("draft", message.getvalue())


if __name__ == "__main__":
    unittest.main()
