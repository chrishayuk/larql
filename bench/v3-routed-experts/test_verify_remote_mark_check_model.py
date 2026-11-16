"""VERIFY-REMOTE-1 revision 2: the relay mark's functional check, against records shaped as the real ones."""
import copy
import json
from pathlib import Path
import unittest

import verify_remote_mark_check_model as mc
import verify_remote_model as vr

# The stop check's real records (relay with the mark, real model, 0 ms):
# the requirements must hold on them exactly as written.
STOP_CHECK = Path(__file__).with_name("link-results") / "verify-remote-1" / "20261116-01-stop-check"


def real(label):
    trial = json.loads((STOP_CHECK / f"{label}.json").read_text())
    relay = {"stats": [json.loads(l) for l in (STOP_CHECK / f"{label}.relay-0.jsonl").read_text().splitlines()],
             "marks": [json.loads(l) for l in
                       (STOP_CHECK / f"{label}.relay-0.jsonl.marks").read_text().splitlines()]}
    return trial, relay


class Requirements(unittest.TestCase):
    def test_the_schedule_is_every_arm_on_json_edit_at_20_ms_sequential_first(self):
        order = mc.entries()
        self.assertEqual([e["arm"] for e in order], list(vr.ARMS))
        self.assertEqual(order[0]["arm"], vr.SEQUENTIAL)
        self.assertTrue(all(e["workload"] == "json_edit" and e["rtt_ms"] == 20.0 and e["corrupt"] == "none"
                            for e in order))

    def test_the_real_stop_check_records_meet_every_requirement(self):
        reference, _ = real("stop-reference")
        for label in ("stop-reference", "stop-verify-2", "stop-verify-4", "stop-verify-8"):
            trial, relay = real(label)
            self.assertEqual(mc.failures(trial, relay, reference["ids"]), [], label)
            phases = mc.phase_bytes(trial, relay)
            for direction, reading in phases.items():
                for phase in ("prefill", "generation"):
                    self.assertGreaterEqual(reading["relay_bytes"][phase], reading["frame_bytes"][phase])

    def test_a_generation_frame_counted_before_the_mark_is_caught(self):
        trial, relay = real("stop-verify-8")
        reference, _ = real("stop-reference")
        moved = copy.deepcopy(relay)
        snapshot = next(m for m in moved["marks"] if not m.get("summary"))
        # Move the first generation exchange's response bytes into the snapshot.
        frame = trial["exchanges"][trial["generation_from"]["exchange"]]["response_bytes"]
        snapshot["response"]["bytes"] += frame
        found = mc.failures(trial, moved, reference["ids"])
        self.assertTrue(any("fewer generation bytes" in f for f in found), found)

    def test_a_mark_defect_fails_and_stops_the_byte_checks(self):
        trial, relay = real("stop-verify-4")
        reference, _ = real("stop-reference")
        refused = copy.deepcopy(relay)
        refused["marks"][-1]["answer"] = "refused"
        found = mc.failures(trial, refused, reference["ids"])
        self.assertTrue(found and all(f.startswith("mark: ") for f in found), found)

    def test_ids_that_differ_fail(self):
        trial, relay = real("stop-verify-2")
        reference, _ = real("stop-reference")
        ids = list(reference["ids"])
        ids[2] += 1
        self.assertTrue(any("ids differ" in f for f in mc.failures(trial, relay, ids)))

    def test_a_reading_passes_only_with_every_arm_clean(self):
        labels = ("stop-reference", "stop-verify-2", "stop-verify-4", "stop-verify-8")
        results = [(arm, *real(label)) for arm, label in zip(vr.ARMS, labels)]
        reading = mc.reading(results)
        self.assertTrue(reading["passes"], reading)
        self.assertFalse(reading["timed_result"])
        self.assertTrue(all(a["answer"] == "ok 1" for a in reading["arms"]))
        self.assertFalse(mc.reading(results[:3])["passes"])


if __name__ == "__main__":
    unittest.main()
