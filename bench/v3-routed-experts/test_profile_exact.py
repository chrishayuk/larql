"""Analysis-only tests; importing the harness never starts a model."""
import unittest

from profile_exact import agreement, summarize


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


if __name__ == "__main__":
    unittest.main()
