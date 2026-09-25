"""Analysis-only tests; importing the harness never starts a model."""
import unittest
from unittest.mock import patch
from pathlib import Path
import io
import json
import tempfile

from profile_exact import agreement, external_urls, read_external_bindings, summarize, workers


def position():
    shards = []
    for shard, finish, compute in [(0, 20, 7), (1, 18, 8)]:
        worker = dict(decode_ns=1, queue_ns=0, execute_ns=compute + 1,
                      experts_ns=compute, encode_ns=1, handler_ns=compute + 3)
        transport = dict(complete=True, worker_profile_complete=True,
                         request_bytes=52, response_bytes=52, encode_ns=1,
                         decode_ns=1, worker=worker, roundtrip_ns=compute + 5,
                         transport_remainder_ns=2)
        shards.append(dict(kind="expert_shard", layer=0, shard=shard,
                           selected_count=1, complete=True, dispatch_start_ns=2,
                           dispatch_finish_ns=finish, transport=[transport]))
    operation = dict(kind="routed_ffn", layer=0, remote=True, complete=True,
                     selected_count=2, router_ns=3, reduction_ns=4, dispatch_ns=26)
    fanout = dict(kind="expert_fanout", layer=0, total_ns=25, complete=True)
    return dict(complete=True, total_ns=49, attention_ns=10, ffn_ns=34,
                reentry_ns=2, other_ns=3, provider_calls=shards + [fanout, operation])


class ProfileAnalysis(unittest.TestCase):
    def test_parallel_work_is_not_added_to_wall_time(self):
        result = summarize([position()], True, 2, 1, 2)
        self.assertEqual(result["worker_experts_sum_ns"], 15)
        self.assertEqual(result["worker_experts_max_ns"], 8)
        self.assertEqual(result["critical_worker_experts_ns"], 7)
        self.assertEqual(result["slowest_shard_wait_ns"], 20)
        self.assertEqual(result["fanout_wall_ns"], 25)
        self.assertEqual(result["fanout_post_completion_ns"], 5)
        self.assertEqual(result["request_bytes"], 104)

    def test_missing_worker_timing_cannot_masquerade_as_free_compute(self):
        row = position()
        row["provider_calls"][0]["transport"][0]["worker_profile_complete"] = False
        with self.assertRaises(AssertionError):
            summarize([row], True, 2, 1, 2)

    def test_control_gate_uses_unrounded_values(self):
        self.assertLess(agreement(100_000, 100_999), 0.01)
        self.assertGreater(agreement(100_000, 101_011), 0.01)


def bindings():
    return [dict(program=dict(schema=1, artifact="same-artifact", backend="cpu",
                              lowering="cpu-production/v1", start=0, end=24,
                              layers=24, hidden=2880),
                 expert_start=start, expert_end=end, operands=[], regions=[])
            for start, end in [(0, 16), (16, 32)]]


class ExternalWorkers(unittest.TestCase):
    urls = ["http://worker-a:9181", "http://worker-b:9181"]

    def test_endpoint_errors_are_rejected_before_execution(self):
        self.assertEqual(external_urls([u + "/" for u in self.urls], "experts"), self.urls)
        for values in ([self.urls[0]], [self.urls[0], self.urls[0] + "/"],
                       ["http://user:secret@worker-a", self.urls[1]],
                       ["file:///model", self.urls[1]],
                       ["http://worker-a:99999", self.urls[1]],
                       ["http://worker-a?token=secret", self.urls[1]]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                external_urls(values, "experts")

    def test_layout_and_identity_are_checked_from_advertisements(self):
        def fetch(values):
            responses = [io.BytesIO(json.dumps(b).encode()) for b in values]
            with patch("profile_exact.urllib.request.urlopen", side_effect=responses):
                return read_external_bindings(self.urls, "experts")
        self.assertEqual(fetch(bindings()), bindings())
        wrong_range = bindings()
        wrong_range[1]["expert_start"] = 15
        wrong_artifact = bindings()
        wrong_artifact[1]["program"]["artifact"] = "another-artifact"
        for values in (list(reversed(bindings())), wrong_range, wrong_artifact):
            with self.assertRaises(ValueError):
                fetch(values)

    def test_external_lifecycle_is_untouched_on_success_and_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for fail in (False, True):
                with patch("profile_exact.read_external_bindings", return_value=bindings()), \
                        patch("profile_exact.subprocess.Popen") as spawn:
                    try:
                        with workers(root, root, root, "experts", 1, {}, self.urls, bindings()) as urls:
                            self.assertEqual(urls, self.urls)
                            if fail:
                                raise RuntimeError("candidate failed")
                    except RuntimeError:
                        self.assertTrue(fail)
                    spawn.assert_not_called()
                receipt = json.loads((root / "workers.json").read_text())
                self.assertEqual(receipt["lifecycle"], "external")
                self.assertEqual(receipt["bindings"], bindings())

    def test_changed_binding_refuses_before_candidate(self):
        changed = bindings()
        changed[0]["operands"] = ["changed"]
        with tempfile.TemporaryDirectory() as temp, \
                patch("profile_exact.read_external_bindings", return_value=changed), \
                patch("profile_exact.subprocess.Popen") as spawn:
            root = Path(temp)
            with self.assertRaises(ValueError):
                with workers(root, root, root, "experts", 1, {}, self.urls, bindings()):
                    self.fail("must refuse before yielding")
            spawn.assert_not_called()

    def test_local_controls_never_contact_external_workers(self):
        with tempfile.TemporaryDirectory() as temp, \
                patch("profile_exact.read_external_bindings") as fetch, \
                patch("profile_exact.subprocess.Popen") as spawn:
            root = Path(temp)
            with workers(root, root, root, "local", 1, {}, self.urls, bindings()) as urls:
                self.assertEqual(urls, [])
            fetch.assert_not_called()
            spawn.assert_not_called()


if __name__ == "__main__":
    unittest.main()
