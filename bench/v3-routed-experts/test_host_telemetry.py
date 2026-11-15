"""Tests for host telemetry parsing; nothing here runs a command."""
import json
from pathlib import Path
import tempfile
import time
import unittest

import host_telemetry as ht


class Parsing(unittest.TestCase):
    def test_processes_are_ranked_by_cpu_and_totalled(self):
        text = ("  1   0.0  26192 /sbin/launchd\n"
                " 42 480.7 900000 target/release/examples/gwbreadcrumb1_spotcheck\n"
                " 77  12.5   3000 /usr/libexec/mds_stores\n"
                "garbage line\n")
        parsed = ht.parse_ps(text, top=2)
        self.assertEqual([row["pid"] for row in parsed["top"]], [42, 77])
        self.assertAlmostEqual(parsed["total_cpu_percent"], 493.2)
        self.assertEqual(parsed["processes"], 3)
        self.assertEqual(parsed["top"][0]["command"], "target/release/examples/gwbreadcrumb1_spotcheck")

    def test_commands_with_spaces_are_kept_whole(self):
        parsed = ht.parse_ps(" 9 1.0 10 /Applications/Brave Browser.app/Contents/Brave Helper\n")
        self.assertEqual(parsed["top"][0]["command"], "/Applications/Brave Browser.app/Contents/Brave Helper")

    def test_thermal_notes_and_levels_are_both_read(self):
        quiet = ("Note: No thermal warning level has been recorded\n"
                 "Note: No performance warning level has been recorded\n")
        self.assertEqual(len(ht.parse_thermal(quiet)["notes"]), 2)
        warm = "CPU_Scheduler_Limit = 100\nThermal warning level = 2\n"
        self.assertEqual(ht.parse_thermal(warm)["Thermal warning level"], "2")

    def test_sysctl_values_follow_the_requested_keys(self):
        self.assertEqual(ht.parse_sysctl("0\n94\n"), {"vm.memory_pressure": 0, "kern.memorystatus_level": 94})

    def test_log_lines_are_counted_per_minute(self):
        text = ("Timestamp               Ty Process[PID:TID]\n"
                "2026-11-15 10:04:01.100 Df launchd[1:2] spawned mdworker_shared[1]\n"
                "2026-11-15 10:04:59.900 Df launchd[1:2] spawned mdworker_shared[2]\n"
                "2026-11-15 10:05:00.000 Df launchd[1:2] spawned mdworker_shared[3]\n")
        self.assertEqual(ht.count_per_minute(text), {"10:04": 2, "10:05": 1})


class SamplerRecords(unittest.TestCase):
    def test_the_sampler_writes_process_and_state_records_and_stops(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "telemetry.jsonl"
            with ht.Sampler(path, process_interval=0.05, state_interval=0.0) as sampler:
                time.sleep(0.3)
            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertFalse(sampler.thread.is_alive())
            self.assertEqual(len(records), sampler.samples)
            kinds = {record["kind"] for record in records}
            self.assertEqual(kinds, {"processes", "state"})
            for record in records:
                self.assertIn("cost_s", record)
                self.assertIn("utc", record)


if __name__ == "__main__":
    unittest.main()
