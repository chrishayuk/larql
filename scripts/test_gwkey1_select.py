import unittest

from gwkey1_select import FULL, METRICS, select


class GwKey1SelectTests(unittest.TestCase):
    def test_nonpositive_denominator_refuses_without_selecting(self):
        effects = {(mask, FULL): {metric: 1.0 for metric in METRICS} for mask in range(64)}
        effects[(0, FULL)] = {"candidate_raw": 0.9, "candidate_zscore": 1.1}
        result = select(
            "K",
            list(effects),
            lambda key, value: effects[(key, value)],
            (0, FULL),
            effects[(FULL, FULL)],
            0.8,
            [0.0] * 64,
            [0.0] * 64,
        )
        self.assertFalse(result["estimable"])
        self.assertIsNone(result["selected"])

    def test_joint_rule_minimizes_union_before_retention(self):
        effects = {
            (key, value): {metric: 0.0 for metric in METRICS}
            for key in range(64)
            for value in range(64)
        }
        effects[(FULL, FULL)] = {metric: 1.0 for metric in METRICS}
        effects[(1, 1)] = {metric: 0.81 for metric in METRICS}
        effects[(3, 0)] = {metric: 0.99 for metric in METRICS}
        selected = select(
            "joint",
            sorted(effects),
            lambda key, value: effects[(key, value)],
            (0, 0),
            effects[(FULL, FULL)],
            0.8,
            [0.0] * 64,
            [0.0] * 64,
        )["selected"]
        self.assertEqual((selected["K_mask"], selected["V_mask"]), (1, 1))
        self.assertEqual(selected["union_cardinality"], 1)

    def test_K_rule_uses_replacement_norm_after_retention(self):
        effects = {(mask, FULL): {metric: 0.0 for metric in METRICS} for mask in range(64)}
        effects[(FULL, FULL)] = {metric: 1.0 for metric in METRICS}
        effects[(1, FULL)] = {metric: 0.9 for metric in METRICS}
        effects[(2, FULL)] = {metric: 0.9 for metric in METRICS}
        norms = [0.0] * 64
        norms[1] = 2.0
        norms[2] = 1.0
        selected = select(
            "K",
            list(effects),
            lambda key, value: effects[(key, value)],
            (0, FULL),
            effects[(FULL, FULL)],
            0.8,
            norms,
            [0.0] * 64,
        )["selected"]
        self.assertEqual(selected["K_mask"], 2)
        self.assertEqual(selected["full_minus_selected"]["K_mask"], FULL ^ 2)
        self.assertEqual(selected["full_minus_selected"]["V_mask"], FULL)


if __name__ == "__main__":
    unittest.main()
