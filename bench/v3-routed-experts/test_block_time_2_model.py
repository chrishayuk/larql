"""Tests for BLOCK-TIME-2's schedule, pair validity and analysis; nothing here starts a process."""
from collections import Counter
import unittest

import block_time_2_model as b2
import block_time_model as bt
from test_block_time_model import BASE_NS, HASHES, trial

WORKER_NS = 4_000_000        # per row, inside the worker
COORDINATOR_NS = 20_000_000  # per position, outside the exchanges
RELAY_ON_TIME = [{"late_sum_ns": 0, "late_max_ns": 0, "chunks": 10, "bytes": 1}]


def record(entry, extra_ns=0.0):
    """A trial that obeys the exchange model exactly, plus `extra_ns` per position."""
    width = bt.ARMS[entry["arm"]] or 1
    run = trial(entry["arm"], handler_ns=width * WORKER_NS)
    roundtrip = width * WORKER_NS + entry["rtt_ms"] * bt.NS_PER_MS
    for exchange in run["exchanges"]:
        exchange["roundtrip_ns"] = int(roundtrip)
    run["wall_ns"] = int(len(run["exchanges"]) * roundtrip
                         + bt.MEASURED_POSITIONS * (COORDINATOR_NS + extra_ns))
    return run


def run_schedule(extra=lambda entry: 0.0, relay=lambda entry: RELAY_ON_TIME, damage=lambda entry, run: None):
    """Every scheduled trial through BLOCK-TIME-1's per-trial gates, then BLOCK-TIME-2's judging."""
    import block_time as b1
    observations = []
    for entry in b2.schedule():
        run = record(entry, extra(entry))
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
        def extra(entry):
            # block-8 pays 4.5 round trips per position at each delay instead of 3.
            if entry["role"] == bt.ROLE_ARM and entry["arm"] == "block-8":
                return 1.5 * entry["rtt_ms"] * bt.NS_PER_MS
            return 0.0
        observations, verdicts = run_schedule(extra=extra)
        entry = b2.analyse(observations, verdicts)["arms"]["block-8"]
        self.assertAlmostEqual(entry["slope"], 4.5, places=6)
        self.assertFalse(entry["follows_exchange_model"])

    def test_a_delay_order_effect_is_reported(self):
        def extra(entry):
            # Whichever delay runs second in a pair is 1 ms per position slower.
            if entry["role"] != bt.ROLE_ARM:
                return 0.0
            second = (entry["order"] == b2.LOW_FIRST) == (entry["rtt_ms"] == 20.0)
            return bt.NS_PER_MS if second else 0.0
        observations, verdicts = run_schedule(extra=extra)
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
                if len(polls) == 3:
                    flag.unlink()
            driver.wait_while_paused(Path(folder), 7, pauses, sleep=sleep, now=lambda: next(clock))
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
