"""Tests for the chunk-trace decomposition; nothing here starts a process."""
import json
import tempfile
import unittest
from pathlib import Path

import link_chunk_trace as ct
from link_sweep import ECHO_HEADER

REQUEST, RESPONSE = 100, 300
DELAY_NS = 500_000


def chunk(direction, size, arrival, hold):
    return {"connection": 0, "direction": direction, "bytes": size, "arrival_ns": arrival,
            "release_ns": arrival + hold, "start_ns": arrival + hold, "done_ns": arrival + hold + 10}


def exchange(start, hold, response_spread):
    """One request chunk, then a response in three chunks arriving `response_spread` apart."""
    request = [chunk("request", ECHO_HEADER.size + REQUEST, start, hold)]
    answered = start + hold + 10 + 1_000
    response = [chunk("response", RESPONSE // 3, answered + i * response_spread, hold) for i in range(3)]
    return request + response


def write(directory, name, chunks):
    path = Path(directory) / name
    path.write_text("".join(json.dumps(c) + "\n" for c in chunks))
    return path


class Stages(unittest.TestCase):
    def test_chunks_are_grouped_into_exchanges_by_frame_bytes(self):
        chunks = exchange(0, 0, 100) + exchange(1_000_000, 0, 100)
        self.assertEqual(len(ct.exchanges(chunks, "request", ECHO_HEADER.size + REQUEST)), 2)
        self.assertEqual([len(g) for g in ct.exchanges(chunks, "response", RESPONSE)], [3, 3])

    def test_a_trace_that_does_not_fit_whole_frames_is_refused(self):
        chunks = exchange(0, 0, 100)
        with self.assertRaises(ValueError):
            ct.exchanges(chunks[:-1], "response", RESPONSE)
        with self.assertRaises(ValueError):
            ct.exchanges(chunks, "response", RESPONSE - 1)

    def test_stages_sum_to_the_relays_view_of_the_exchange(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write(directory, "t.jsonl", exchange(0, DELAY_NS, 40_000) + exchange(5_000_000, DELAY_NS, 40_000))
            rows = ct.stages(path, REQUEST, RESPONSE, warmup=1)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(sum(row[stage] for stage in ct.STAGES), row["relay_total"])
        self.assertEqual(row["response_in"], 80_000)
        self.assertEqual(row["request_hold"], DELAY_NS + 10)
        self.assertEqual(row["turnaround"], 1_000)

    def test_a_faster_ingest_shows_as_a_shortfall_against_baseline_plus_delay(self):
        # The shaped relay holds every chunk the full delay, but takes the
        # response in faster than the baseline relay did.
        with tempfile.TemporaryDirectory() as directory:
            point = Path(directory) / "p.json"
            baseline = [c for i in range(3) for c in exchange(i * 10_000_000, 0, 90_000)]
            shaped = [c for i in range(3) for c in exchange(i * 10_000_000, DELAY_NS, 40_000)]
            write(directory, "p.json.baseline-relay.jsonl.trace.jsonl", baseline)
            write(directory, "p.json.shaped-relay.jsonl.trace.jsonl", shaped)
            point.write_text(json.dumps({"one_way_delay_us": "500", "warmup": 1,
                                         "pairs": [{"baseline_ns": [200_000], "shaped_ns": [1_100_000]}]}))
            report = ct.analyse(str(point), REQUEST, RESPONSE)
        medians = report["median_ns"]
        self.assertEqual(medians["response_in"]["shaped_minus_baseline_minus_configured"], -100_000)
        self.assertEqual(medians["response_hold"]["shaped_minus_baseline_minus_configured"], 0)
        self.assertEqual(medians["relay_total"]["shaped_minus_baseline_minus_configured"], -100_000)
        self.assertEqual(report["probe_error_ns"], -100_000)


if __name__ == "__main__":
    unittest.main()
