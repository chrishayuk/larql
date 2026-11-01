"""Synthetic source/dependency witnesses; no captured or replayed model outcomes."""
import dataclasses
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gwcar1b_sources import (  # noqa: E402
    FEATURES, GRID, SUPPORT, WIDTH, Row, construct, fit, oracle, top_support,
)


def fixture():
    rows, carriers, features = [], [], {name: [] for name in FEATURES}
    for subject in range(4):
        entity = np.zeros(WIDTH, dtype="<f4")
        entity[subject] = 1
        for group in range(2):
            rows.append(Row(f"row-{subject}-{group}", f"S{subject}", "capital", f"t{group}", "train"))
            carrier = np.full(WIDTH, 3 + group * 2, dtype="<f4")
            carrier[:SUPPORT] += (subject - 1.5) * np.linspace(1, 2, SUPPORT)
            carriers.append(carrier)
            features["subject_embedding"].append(entity.copy())
            for name in FEATURES[1:]:
                early = entity.copy()
                early[10] = group
                features[name].append(early)
    return rows, np.asarray(carriers), {name: np.asarray(values) for name, values in features.items()}


class SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.carriers, cls.features = fixture()
        cls.bank = fit(cls.rows, cls.carriers, cls.features)
        cls.query = Row("new-row", "NEW", "capital", "t0", "validation")

    def test_all_sources_are_finite_and_disclose_dependencies(self):
        for candidate in GRID:
            with self.subTest(candidate=candidate.name):
                if candidate.family == "oracle":
                    value = oracle(self.bank, candidate.candidate_id, self.query,
                                   self.carriers[0], self.carriers[2])
                else:
                    feature = self.features[candidate.feature][0] if candidate.feature else None
                    value, receipt = construct(self.bank, candidate.candidate_id, self.query, feature)
                    self.assertFalse(receipt["reads_target_l24_carrier"])
                    self.assertFalse(receipt["efficiency_claim"])
                    self.assertEqual(receipt["dense_output_bytes"], WIDTH * 4)
                    self.assertEqual(receipt["retained_prefix_blocks"],
                                     {None: 0, "subject_embedding": 0, "enter_l12": 12,
                                      "enter_l16": 16}[candidate.feature])
                self.assertEqual(value.shape, (WIDTH,))
                self.assertTrue(np.isfinite(value).all())

    def test_known_additive_structure_and_correspondence_sham(self):
        value, receipt = construct(self.bank, 11, self.query, self.features["subject_embedding"][0])
        template, _ = construct(self.bank, 8, self.query)
        sham, _ = construct(self.bank, 17, self.query, self.features["subject_embedding"][0])
        natural = self.carriers[0]
        self.assertLess(np.linalg.norm(value - natural), 0.02 * np.linalg.norm(template - natural))
        self.assertGreater(np.linalg.norm(sham - natural), 20 * np.linalg.norm(value - natural))
        np.testing.assert_array_equal(value[SUPPORT:], template[SUPPORT:])
        self.assertEqual(receipt["ridge_multiply_add_pairs"], WIDTH * SUPPORT)
        expected_bytes = 2 * WIDTH * 4 + SUPPORT * 2 + WIDTH * 4 + SUPPORT * 4 + WIDTH * SUPPORT * 4
        self.assertEqual(receipt["numeric_shared_bytes"], expected_bytes)

    def test_retrieval_uses_declared_group_and_stable_tie_break(self):
        # Equidistant from every one-hot subject vector; lexical row ID wins.
        value, receipt = construct(self.bank, 9, self.query, np.zeros(WIDTH))
        self.assertEqual(receipt["retrieved_row_ids"], ["row-0-0"])
        np.testing.assert_array_equal(value, self.carriers[0])
        mean, receipt = construct(self.bank, 10, self.query, np.zeros(WIDTH))
        self.assertEqual(receipt["retrieved_row_ids"], [f"row-{s}-0" for s in range(4)])
        np.testing.assert_array_equal(mean, self.bank.templates[self.query.group])
        self.assertEqual(receipt["distance_coordinates"], 4 * WIDTH)

    def test_support_selection_is_distinct_from_value_construction(self):
        natural = np.zeros(WIDTH, dtype="<f4")
        outside = next(i for i in range(WIDTH) if i not in self.bank.carrier_support)
        natural[outside] = 100
        adaptive = oracle(self.bank, 2, self.query, natural)
        fixed = oracle(self.bank, 3, self.query, natural)
        self.assertEqual(adaptive[outside], 100)
        self.assertEqual(fixed[outside], 0)
        template = self.bank.templates[self.query.group]
        natural = template.copy()
        natural[:SUPPORT] += np.arange(SUPPORT)
        np.testing.assert_array_equal(oracle(self.bank, 5, self.query, natural), natural)
        np.testing.assert_array_equal(top_support(np.ones(WIDTH)), np.arange(SUPPORT))

    def test_training_order_does_not_change_fit_or_retrieval(self):
        reverse = np.arange(len(self.rows))[::-1]
        bank = fit([self.rows[i] for i in reverse], self.carriers[reverse],
                   {name: values[reverse] for name, values in self.features.items()})
        for key in bank.ridge:
            np.testing.assert_array_equal(bank.ridge[key].weights.view("<u4"),
                                          self.bank.ridge[key].weights.view("<u4"))

    def test_heldout_fit_and_incomplete_crossing_refuse(self):
        rows = self.rows.copy()
        rows[0] = dataclasses.replace(rows[0], split="test")
        with self.assertRaisesRegex(ValueError, "train rows only"):
            fit(rows, self.carriers, self.features)
        with self.assertRaisesRegex(ValueError, "one row per"):
            fit(self.rows[:-1], self.carriers[:-1], {k: v[:-1] for k, v in self.features.items()})

    def test_confirmation_cannot_retrieve_same_subject_or_accept_oracle_input(self):
        seen = dataclasses.replace(self.query, subject_id=self.rows[0].subject_id)
        with self.assertRaisesRegex(ValueError, "overlaps"):
            construct(self.bank, 9, seen, self.features["subject_embedding"][0])
        with self.assertRaisesRegex(ValueError, "oracle"):
            construct(self.bank, 2, self.query)
        with self.assertRaises(TypeError):
            construct(self.bank, 8, self.query, natural=self.carriers[0])
        with self.assertRaisesRegex(ValueError, "declared source"):
            construct(self.bank, 8, self.query, self.carriers[0])
        with self.assertRaisesRegex(ValueError, "no fallback"):
            construct(self.bank, 8, dataclasses.replace(self.query, template="unseen"))

    def test_wrong_feature_shape_nonfinite_and_template_conditioned_embedding_refuse(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            construct(self.bank, 9, self.query, np.zeros(WIDTH - 1))
        x = np.zeros(WIDTH)
        x[0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            construct(self.bank, 9, self.query, x)
        changed = {k: v.copy() for k, v in self.features.items()}
        changed["subject_embedding"][0, 8] = 1
        with self.assertRaisesRegex(ValueError, "varies with template"):
            fit(self.rows, self.carriers, changed)

    def test_zero_variance_features_have_finite_constant_prediction(self):
        features = {k: np.zeros_like(v) for k, v in self.features.items()}
        bank = fit(self.rows, self.carriers, features)
        value, _ = construct(bank, 11, self.query, features["subject_embedding"][0])
        np.testing.assert_array_equal(value, bank.templates[self.query.group])


if __name__ == "__main__":
    unittest.main()
