"""Tests for BLOCK-TIME-1's schedule, gates and analysis; nothing here starts a process."""
import unittest

import block_time_model as bt

PROMPT_TOKENS = 152
STREAM = list(range(1000, 1000 + bt.POSITIONS))
BASE_NS = 100_000_000.0  # per position at zero delay
HASHES = [f"{position:064x}" for position in range(bt.POSITIONS)]


def greedy(stream):
    """What an exact run chooses: at each position, the next supplied id."""
    return stream[1:] + [0]


def exchange(route, rows, roundtrip_ns=1_000_000):
    return {"route": route, "layer": 0, "rows": rows, "pairs": 4 * rows, "request_bytes": 10 * rows,
            "response_bytes": 40 * rows, "roundtrip_ns": roundtrip_ns}


def trial(arm, wall_ns=None, profiled=True, handler_ns=600_000):
    width = bt.ARMS[arm]
    count = bt.ROUTED_LAYERS * bt.MEASURED_POSITIONS // (width or 1)
    route = "single-row" if width is None else "multi-row"
    exchanges = [exchange(route, width or 1) for _ in range(count)]
    worker = {"handler_ns": handler_ns, "execute_ns": handler_ns // 2, "experts_ns": handler_ns // 4,
              "queue_ns": 10, "decode_ns": 1, "encode_ns": 1}
    return {
        "arm": arm, "width": width, "profiled": profiled, "positions": bt.POSITIONS,
        "wall_ns": wall_ns if wall_ns is not None else bt.MEASURED_POSITIONS * BASE_NS,
        "frames": {"measured": {"single-row": count if width is None else 0,
                                "multi-row": 0 if width is None else count}},
        "exchanges": exchanges,
        "worker_records": [{"worker": dict(worker)} for _ in exchanges] if profiled else [],
        "argmax": greedy(STREAM), "logits_sha256": list(HASHES),
    }


def observations(per_position, valid=lambda entry: True):
    """One observation per scheduled trial, timed by `per_position(arm, rtt_ms, repeat)`."""
    out = []
    for entry in bt.schedule():
        ns = per_position(entry["arm"], entry["rtt_ms"], entry["repeat"])
        out.append(dict(entry, valid=valid(entry), summary={"per_position_ns": ns}))
    return out


def model(arm, rtt_ms, repeat):
    """The exchange model exactly: a base plus one round trip per dependent exchange."""
    return BASE_NS + bt.exchanges_per_position(arm) * rtt_ms * bt.NS_PER_MS


class Schedule(unittest.TestCase):
    def test_every_arm_runs_at_every_delay_in_every_repeat(self):
        arm_trials = [t for t in bt.schedule() if t["role"] == bt.ROLE_ARM]
        self.assertEqual(len(arm_trials), len(bt.ARMS) * len(bt.RTT_MS) * bt.REPEATS)
        seen = {(t["arm"], t["rtt_ms"], t["repeat"]) for t in arm_trials}
        self.assertEqual(len(seen), len(arm_trials))
        self.assertTrue(all(t["profiled"] for t in arm_trials))

    def test_every_delay_group_sits_between_two_sequential_zero_delay_brackets(self):
        trials = bt.schedule()
        for index, entry in enumerate(trials):
            if entry["role"] not in (bt.ROLE_ARM, bt.ROLE_UNPROFILED):
                continue
            before = next(t for t in reversed(trials[:index]) if t["role"] == bt.ROLE_BRACKET)
            after = next(t for t in trials[index:] if t["role"] == bt.ROLE_BRACKET)
            for bracket in (before, after):
                self.assertEqual((bracket["arm"], bracket["rtt_ms"], bracket["repeat"]),
                                 (bt.SEQUENTIAL, 0.0, entry["repeat"]))

    def test_second_repeat_reverses_delays_and_arm_order(self):
        def order(repeat):
            return [(t["rtt_ms"], t["arm"]) for t in bt.schedule()
                    if t["role"] == bt.ROLE_ARM and t["repeat"] == repeat]
        first, second = order(0), order(1)
        self.assertEqual([d for d, _ in first][::len(bt.ARMS)], list(bt.RTT_MS))
        self.assertEqual([d for d, _ in second][::len(bt.ARMS)], list(reversed(bt.RTT_MS)))
        # The same arm does not always lead a delay.
        self.assertGreater(len({first[i][1] for i in range(0, len(first), len(bt.ARMS))}), 1)
        self.assertNotEqual([a for _, a in first[:len(bt.ARMS)]], [a for _, a in second[-len(bt.ARMS):]])

    def test_each_arm_runs_unprofiled_once_per_repeat_at_zero_delay_only(self):
        plain = [t for t in bt.schedule() if t["role"] == bt.ROLE_UNPROFILED]
        self.assertEqual(len(plain), len(bt.ARMS) * bt.REPEATS)
        self.assertTrue(all(t["rtt_ms"] == 0.0 and not t["profiled"] for t in plain))

    def test_the_schedule_has_the_registered_number_of_trials(self):
        roles = [t["role"] for t in bt.schedule()]
        self.assertEqual({role: roles.count(role) for role in set(roles)},
                         {bt.ROLE_ARM: 30, bt.ROLE_UNPROFILED: 10, bt.ROLE_BRACKET: 8, bt.ROLE_DIRECT: 2})
        self.assertEqual(len(roles), 50)

    def test_only_qualified_delays_are_ever_scheduled(self):
        import link_block_frames_model as bm
        self.assertEqual(bt.RTT_MS, (0.0, 5.0, 20.0))
        scheduled = {t["rtt_ms"] for t in bt.schedule()}
        self.assertEqual(scheduled, {0.0, *bm.RTT_MS})
        self.assertNotIn(1.0, scheduled)
        self.assertEqual(bt.DIFFERENTIAL_RTT_MS, bm.RTT_MS)

    def test_measured_region_is_whole_blocks_at_every_width(self):
        for width in filter(None, bt.ARMS.values()):
            self.assertEqual(bt.MEASURE_FROM % width, 0)
            self.assertEqual(bt.MEASURED_POSITIONS % width, 0)

    def test_labels_are_unique_and_ordered(self):
        labels = [t["label"] for t in bt.schedule()]
        self.assertEqual(labels, sorted(set(labels)))


class Alignment(unittest.TestCase):
    def test_an_exact_run_reproduces_every_saved_id(self):
        self.assertEqual(bt.control_alignment(PROMPT_TOKENS, STREAM, greedy(STREAM)), [])

    def test_exactly_the_generated_ids_are_compared(self):
        # Every prediction wrong: one report per generated id, from the last
        # prompt position to the position before the last.
        wrong = bt.control_alignment(PROMPT_TOKENS, STREAM, [0] * len(STREAM))
        self.assertEqual(wrong, list(range(PROMPT_TOKENS - 1, len(STREAM) - 1)))
        self.assertEqual(len(wrong), len(STREAM) - PROMPT_TOKENS)

    def test_an_alignment_shifted_by_one_in_either_direction_is_caught(self):
        exact = greedy(STREAM)
        # Reading position i's logits as predicting token i (one too early).
        early = list(STREAM)
        # Reading position i's logits as predicting token i + 2 (one too late).
        late = exact[1:] + [0]
        for shifted in (early, late):
            self.assertEqual(len(bt.control_alignment(PROMPT_TOKENS, STREAM, shifted)),
                             len(STREAM) - PROMPT_TOKENS)

    def test_a_prompt_length_off_by_one_is_caught_when_the_boundary_id_is_wrong(self):
        argmax = greedy(STREAM)
        argmax[PROMPT_TOKENS - 1] = 0  # the first generated id is wrong
        self.assertEqual(bt.control_alignment(PROMPT_TOKENS, STREAM, argmax), [PROMPT_TOKENS - 1])
        # A gate that started one position late would not have looked there.
        self.assertEqual(bt.control_alignment(PROMPT_TOKENS + 1, STREAM, argmax), [])

    def test_the_last_positions_logits_are_not_compared(self):
        argmax = greedy(STREAM)
        argmax[-1] = 12345
        self.assertEqual(bt.control_alignment(PROMPT_TOKENS, STREAM, argmax), [])

    def test_a_reference_must_cover_the_stream_and_match_the_control(self):
        good = trial(bt.SEQUENTIAL)
        self.assertEqual(bt.reference_defects(PROMPT_TOKENS, STREAM, good), [])
        drifted = dict(good, argmax=[0] * bt.POSITIONS)
        self.assertTrue(bt.reference_defects(PROMPT_TOKENS, STREAM, drifted))
        short = dict(good, logits_sha256=HASHES[:-1])
        self.assertTrue(bt.reference_defects(PROMPT_TOKENS, STREAM, short))


class Gates(unittest.TestCase):
    def test_parity_names_the_differing_positions(self):
        good = trial("block-8")
        self.assertEqual(bt.parity_mismatches(HASHES, good), [])
        changed = dict(good, logits_sha256=HASHES[:7] + ["f" * 64] + HASHES[8:])
        self.assertEqual(bt.parity_mismatches(HASHES, changed), [7])
        self.assertTrue(bt.parity_mismatches(HASHES, dict(good, logits_sha256=HASHES[:-1])))

    def test_each_arm_must_send_exactly_its_frames(self):
        for arm in bt.ARMS:
            self.assertEqual(bt.frame_defects(trial(arm)), [], arm)
        # A block arm that fell back to single-row frames.
        fallback = trial("block-8")
        fallback["frames"]["measured"] = {"single-row": 1, "multi-row": fallback["frames"]["measured"]["multi-row"]}
        self.assertTrue(bt.frame_defects(fallback))
        # A block arm that sent one frame per position.
        ungrouped = trial("block-8")
        ungrouped["frames"]["measured"]["multi-row"] *= 8
        self.assertTrue(bt.frame_defects(ungrouped))
        # A sequential arm that used the multi-row route.
        wrong_route = trial(bt.SEQUENTIAL)
        wrong_route["frames"]["measured"] = {"single-row": 0, "multi-row": bt.ROUTED_LAYERS * bt.MEASURED_POSITIONS}
        self.assertTrue(bt.frame_defects(wrong_route))
        missing = trial("block-2")
        missing["exchanges"].pop()
        self.assertTrue(bt.frame_defects(missing))

    def test_width_eight_sends_an_eighth_of_sequentials_frames(self):
        sequential = trial(bt.SEQUENTIAL)["frames"]["measured"]["single-row"]
        self.assertEqual(trial("block-8")["frames"]["measured"]["multi-row"] * 8, sequential)

    def test_bracket_drift_is_judged_against_the_gate(self):
        steady = bt.bracket_drift({"per_position_ns": 100.0}, {"per_position_ns": 100.5})
        moved = bt.bracket_drift({"per_position_ns": 100.0}, {"per_position_ns": 102.0})
        self.assertLessEqual(steady, bt.GATE)
        self.assertGreater(moved, bt.GATE)


class Decomposition(unittest.TestCase):
    def test_wall_time_splits_into_exchanges_and_coordinator(self):
        run = trial("block-8", wall_ns=400_000_000)
        summary = bt.summarize(run, rtt_ms=1.0)
        count = bt.ROUTED_LAYERS * bt.MEASURED_POSITIONS // 8
        self.assertEqual(summary["exchanges"], count)
        self.assertEqual(summary["exchange_ns"], count * 1_000_000)
        self.assertEqual(summary["coordinator_ns"], 400_000_000 - count * 1_000_000)
        self.assertEqual(summary["configured_delay_ns"], count * 1_000_000)
        self.assertEqual(summary["per_position_ns"], 400_000_000 / bt.MEASURED_POSITIONS)
        self.assertEqual(summary["request_bytes"], count * 80)

    def test_worker_time_is_reported_only_when_every_exchange_has_it(self):
        full = bt.summarize(trial("block-4", handler_ns=600_000), rtt_ms=0.0)
        count = bt.ROUTED_LAYERS * bt.MEASURED_POSITIONS // 4
        self.assertEqual(full["worker"]["handler_ns"], count * 600_000)
        self.assertEqual(full["worker"]["unattributed_exchange_ns"], count * (1_000_000 - 600_000))
        partial = trial("block-4")
        partial["worker_records"].pop()
        self.assertIsNone(bt.summarize(partial, rtt_ms=0.0)["worker"])
        self.assertIsNone(bt.summarize(trial("block-4", profiled=False), rtt_ms=0.0)["worker"])

    def test_configured_delay_is_removed_before_anything_is_called_unattributed(self):
        summary = bt.summarize(trial(bt.SEQUENTIAL, handler_ns=100_000), rtt_ms=0.5)
        count = bt.ROUTED_LAYERS * bt.MEASURED_POSITIONS
        self.assertEqual(summary["worker"]["unattributed_exchange_ns"], count * (1_000_000 - 500_000 - 100_000))


class Analysis(unittest.TestCase):
    def test_a_run_that_obeys_the_exchange_model_is_recognised(self):
        report = bt.analyse(observations(model))
        for arm, slope in report["slopes"].items():
            self.assertAlmostEqual(slope["fitted"], bt.exchanges_per_position(arm), places=6)
            self.assertTrue(slope["follows_exchange_model"])
            self.assertAlmostEqual(slope["intercept_ns"], BASE_NS, places=0)
        self.assertEqual(report["void"], 0)

    def test_the_differential_slope_does_not_use_the_zero_delay_arm(self):
        # The zero-delay relay costs a wide block extra; the delayed arms obey the model.
        offset = 0.5 * bt.exchanges_per_position("block-8") * 20.0 * bt.NS_PER_MS

        def offset_at_zero(arm, rtt_ms, repeat):
            return model(arm, rtt_ms, repeat) + (offset if arm == "block-8" and rtt_ms == 0.0 else 0)
        slope = bt.analyse(observations(offset_at_zero))["slopes"]["block-8"]
        self.assertAlmostEqual(slope["differential"], bt.exchanges_per_position("block-8"), places=6)
        self.assertTrue(slope["differential_within_band"])
        self.assertFalse(slope["fit_within_band"])
        # The statement needs both.
        self.assertFalse(slope["follows_exchange_model"])
        self.assertEqual([round(value, 6) for value in slope["differential_per_repeat"]], [3.0, 3.0])

    def test_a_differential_outside_the_band_fails_even_if_the_fit_passes(self):
        def bent(arm, rtt_ms, repeat):
            # Raise the middle delay: the three-point fit barely moves, the 5-to-20 slope does.
            lift = 0.2 * bt.exchanges_per_position(arm) * 15.0 * bt.NS_PER_MS
            return model(arm, rtt_ms, repeat) + (lift if arm == "block-4" and rtt_ms == 5.0 else 0)
        slope = bt.analyse(observations(bent))["slopes"]["block-4"]
        self.assertTrue(slope["fit_within_band"])
        self.assertFalse(slope["differential_within_band"])
        self.assertFalse(slope["follows_exchange_model"])

    def test_residuals_are_reported_at_every_fitted_delay_and_vanish_on_the_model(self):
        slope = bt.analyse(observations(model))["slopes"]["block-2"]
        self.assertEqual(set(slope["residuals_ns"]), {str(rtt_ms) for rtt_ms in bt.RTT_MS})
        for value in slope["residuals_ns"].values():
            self.assertAlmostEqual(value, 0.0, places=3)

    def test_predicted_slopes_are_exchanges_per_position(self):
        self.assertEqual([bt.exchanges_per_position(arm) for arm in bt.ARMS], [24, 24, 12, 6, 3])

    def test_a_slope_outside_the_band_is_not_said_to_follow_the_model(self):
        def slow(arm, rtt_ms, repeat):
            extra = 1 + 2 * bt.SLOPE_BAND if arm == "block-8" else 1
            return BASE_NS + extra * bt.exchanges_per_position(arm) * rtt_ms * bt.NS_PER_MS
        report = bt.analyse(observations(slow))
        self.assertFalse(report["slopes"]["block-8"]["follows_exchange_model"])
        self.assertTrue(report["slopes"]["block-4"]["follows_exchange_model"])

    def test_faster_needs_the_minimum_effect_in_every_repeat(self):
        def one_repeat_only(arm, rtt_ms, repeat):
            if arm == "block-8" and rtt_ms == 5.0 and repeat == 1:
                return model(bt.SEQUENTIAL, rtt_ms, repeat)
            return model(arm, rtt_ms, repeat)
        report = bt.analyse(observations(one_repeat_only))
        claims = {(c["arm"], c["rtt_ms"]): c for c in report["faster"]}
        self.assertFalse(claims[("block-8", 5.0)]["faster_in_every_repeat"])
        self.assertTrue(claims[("block-8", 20.0)]["faster_in_every_repeat"])
        # Equal at zero delay: no claim of being faster.
        self.assertFalse(claims[("block-8", 0.0)]["faster_in_every_repeat"])
        # The control arm and sequential are never claimed about.
        self.assertFalse(any(arm in (bt.SEQUENTIAL, bt.CONTROL_ARM) for arm, _ in claims))

    def test_an_effect_at_or_below_the_minimum_is_not_a_claim(self):
        def slight(arm, rtt_ms, repeat):
            base = model(bt.SEQUENTIAL, rtt_ms, repeat)
            return base * (1 - bt.MINIMUM_EFFECT) if arm == "block-2" else base
        claims = {(c["arm"], c["rtt_ms"]): c for c in bt.analyse(observations(slight))["faster"]}
        self.assertFalse(claims[("block-2", 20.0)]["faster_in_every_repeat"])

    def test_void_trials_are_neither_fitted_nor_claimed_on(self):
        def void(entry):
            return not (entry["arm"] == "block-4" and entry["rtt_ms"] == 5.0 and entry["repeat"] == 0)
        report = bt.analyse(observations(model, valid=void))
        self.assertEqual(report["void"], 1)
        cell = next(c for c in report["cells"] if c["arm"] == "block-4" and c["rtt_ms"] == 5.0)
        self.assertIsNone(cell["repeats_ns"][0])
        self.assertIsNotNone(cell["mean_ns"])
        claim = next(c for c in report["faster"] if c["arm"] == "block-4" and c["rtt_ms"] == 5.0)
        self.assertIsNone(claim["faster_in_every_repeat"])

        def all_void(entry):
            return not (entry["arm"] == "block-4" and entry["rtt_ms"] == 5.0)
        slope = bt.analyse(observations(model, valid=all_void))["slopes"]["block-4"]
        self.assertIsNone(slope["fitted"])
        self.assertIsNone(slope["follows_exchange_model"])

    def test_profiling_overhead_compares_the_same_arm_at_zero_delay(self):
        def costly(arm, rtt_ms, repeat):
            return model(arm, rtt_ms, repeat)
        observed = observations(costly)
        for entry in observed:
            if entry["role"] == bt.ROLE_UNPROFILED:
                entry["summary"]["per_position_ns"] = BASE_NS / 1.02
        overhead = bt.analyse(observed)["profiling_overhead"]
        self.assertEqual(len(overhead), len(bt.ARMS) * bt.REPEATS)
        for row in overhead:
            self.assertAlmostEqual(row["relative"], 0.02, places=6)

    def test_a_slope_needs_distinct_delays(self):
        with self.assertRaises(ValueError):
            bt.fit_slope([(0.0, 1.0)])
        with self.assertRaises(ValueError):
            bt.fit_slope([(1.0, 1.0), (1.0, 2.0)])


class Judging(unittest.TestCase):
    """The driver's validity marking, on synthetic observations."""

    def observed(self, bracket_ns=lambda index: BASE_NS):
        import block_time as driver
        out, brackets = [], 0
        for entry in bt.schedule():
            ns = model(entry["arm"], entry["rtt_ms"], entry["repeat"])
            if entry["role"] == bt.ROLE_BRACKET:
                ns = bracket_ns(brackets)
                brackets += 1
            relay = None if entry["role"] == bt.ROLE_DIRECT else {"relay_on_schedule": True}
            out.append(dict(entry, summary={"per_position_ns": ns}, frame_defects=[], relay=relay))
        return driver, out

    def test_a_clean_run_is_wholly_valid(self):
        driver, out = self.observed()
        self.assertTrue(all(o["valid"] for o in driver.judge(out)))

    def test_only_the_trials_between_drifting_brackets_are_void(self):
        # The third bracket of the run drifts beyond the gate.
        driver, out = self.observed(lambda index: BASE_NS * (1 + 3 * bt.GATE) if index == 2 else BASE_NS)
        judged = driver.judge(out)
        brackets = [i for i, o in enumerate(judged) if o["role"] == bt.ROLE_BRACKET]
        for index, observation in enumerate(judged):
            if observation["role"] not in (bt.ROLE_ARM, bt.ROLE_UNPROFILED):
                continue
            beside = brackets[1] < index < brackets[3]
            self.assertEqual(observation["valid"], not beside, observation["label"])

    def test_frame_and_relay_defects_void_a_trial_by_name(self):
        driver, out = self.observed()
        out[5]["frame_defects"] = ["1 single-row frames, expected 0"]
        out[7]["relay"] = {"relay_on_schedule": False}
        judged = driver.judge(out)
        self.assertEqual(judged[5]["void_reasons"], ["1 single-row frames, expected 0"])
        self.assertEqual(judged[7]["void_reasons"], ["relay behind schedule"])
        self.assertEqual(sum(not o["valid"] for o in judged), 2)

    def test_a_parity_failure_stops_the_run(self):
        import block_time as driver
        entry = bt.schedule()[2]
        changed = dict(trial(entry["arm"]), logits_sha256=["f" * 64] + HASHES[1:])
        with self.assertRaises(driver.Stop):
            driver.observe(entry, changed, None, HASHES, BASE_NS)

    def test_zero_delay_is_the_floor_point_and_delays_are_unlimited_rate(self):
        import block_time as driver
        import link_model as lm
        self.assertEqual(driver.point_of({"rtt_ms": 0.0}), dict(lm.FLOOR))
        self.assertEqual(driver.point_of({"rtt_ms": 5.0})["rate_bits_per_s"], lm.UNLIMITED)


class Admission(unittest.TestCase):
    """What the driver refuses before it starts anything."""

    def test_an_admissible_run_refuses_to_continue_unsettled_before_starting_anything(self):
        import contextlib
        import io
        import block_time as driver

        def must_not_run(*args, **kwargs):
            raise AssertionError("the driver started work for a run it must refuse")

        started = {name: getattr(driver, name) for name in ("workers", "relays", "run_trial", "execute")}
        popen, run = driver.subprocess.Popen, driver.subprocess.run
        try:
            for name in started:
                setattr(driver, name, must_not_run)
            driver.subprocess.Popen = driver.subprocess.run = must_not_run
            message = io.StringIO()
            with self.assertRaises(SystemExit) as refused, contextlib.redirect_stderr(message):
                driver.main(["--admissible", "--continue-unsettled", "--model", "unused",
                             "--out", "/nonexistent/never-created"])
        finally:
            for name, original in started.items():
                setattr(driver, name, original)
            driver.subprocess.Popen, driver.subprocess.run = popen, run
        self.assertNotEqual(refused.exception.code, 0)
        self.assertIn("--continue-unsettled is for diagnostics", message.getvalue())

    def test_missing_worker_timing_is_named_and_is_not_the_same_as_unprofiled(self):
        import block_time as driver
        arm = next(e for e in bt.schedule() if e["role"] == bt.ROLE_ARM and e["arm"] == "block-4")
        plain = next(e for e in bt.schedule() if e["role"] == bt.ROLE_UNPROFILED and e["arm"] == "block-4")
        full = driver.observe(arm, trial("block-4"), None, HASHES, BASE_NS)
        self.assertEqual(full["worker_timing"], driver.WORKER_TIMING_COMPLETE)
        # One exchange came back without worker fields, as from an older worker.
        partial = trial("block-4")
        partial["worker_records"][3] = {"worker_profile_complete": False}
        lacking = driver.observe(arm, partial, None, HASHES, BASE_NS)
        self.assertEqual(lacking["worker_timing"], driver.WORKER_TIMING_INCOMPLETE)
        self.assertIsNone(lacking["summary"]["worker"])
        # Its wall time is still a measurement and no gate voids it for this.
        self.assertEqual(lacking["frame_defects"], [])
        self.assertEqual(lacking["summary"]["per_position_ns"], full["summary"]["per_position_ns"])
        unasked = driver.observe(plain, trial("block-4", profiled=False), None, HASHES, BASE_NS)
        self.assertEqual(unasked["worker_timing"], driver.WORKER_TIMING_NOT_REQUESTED)

    def test_provenance_binds_a_receipt_to_source_and_environment(self):
        import os
        from pathlib import Path
        import block_time as driver
        root = Path(driver.__file__).resolve().parents[2]
        os.environ["LARQL_PROVENANCE_PROBE"] = "1"
        try:
            record = driver.provenance(root, driver.TRIAL_SOURCES)
        finally:
            del os.environ["LARQL_PROVENANCE_PROBE"]
        self.assertEqual(len(record["revision"]), 40)
        self.assertIsInstance(record["uncommitted"], list)
        self.assertEqual(set(record["source_sha256"]), set(driver.TRIAL_SOURCES))
        self.assertTrue(all(len(digest) == 64 for digest in record["source_sha256"].values()))
        self.assertEqual(record["environment"]["LARQL_PROVENANCE_PROBE"], "1")
        self.assertNotIn("PATH", record["environment"])


if __name__ == "__main__":
    unittest.main()
