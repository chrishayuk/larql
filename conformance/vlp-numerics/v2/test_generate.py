"""Checks on the v2 golden vectors (draft). Run: python3 -I -m unittest test_generate  (from this directory)."""
import hashlib
import json
import os
import subprocess
import sys
import unittest

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(os.path.dirname(HERE), "v1")
sys.path.insert(0, HERE)
import generate as G  # noqa: E402

BASE_REFUSALS = {"columns_invalid", "fixup_no_op", "fixups_invalid", "non_finite_intermediate", "non_finite_operand",
                 "numerics_invalid", "overflow", "rank_invalid", "shape_invalid", "tensor_refused"}
DIFFERING_FROM_V1 = {"eq-lr-negzero-plus-poszero"}
NEG_ZERO = "8000"


def load():
    with open(G.OUT) as f:
        return json.load(f)


class Reproduces(unittest.TestCase):
    def test_vectors_are_what_the_generator_writes(self):
        with open(G.OUT) as f:
            self.assertEqual(f.read(), G.dumps(G.document()))

    def test_v1_is_untouched_and_nothing_is_written_into_it(self):
        frozen = json.load(open(os.path.join(V1, "FREEZE.json")))["sha256"]
        for name in ("vectors.json", "generate.py"):
            with open(os.path.join(V1, name), "rb") as f:
                self.assertEqual(hashlib.sha256(f.read()).hexdigest(), frozen[name], name)
        self.assertEqual(sorted(os.listdir(V1)), ["FREEZE.json", "SPEC.md", "generate.py", "test_generate.py", "vectors.json"])

    def test_the_generator_runs_in_isolated_mode(self):
        r = subprocess.run([sys.executable, "-I", os.path.join(HERE, "generate.py"), "--check"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)


class Equivalence(unittest.TestCase):
    def test_every_v1_add_low_rank_case_is_re_expressed(self):
        v1 = {c["id"] for c in json.load(open(os.path.join(V1, "vectors.json")))["cases"]
              if c["op"]["kind"] == "add_low_rank"}
        got = {c["equivalent_to_v1"] for c in load()["cases"] if "equivalent_to_v1" in c}
        self.assertEqual(got, v1)

    def test_only_the_enumerated_case_differs_and_it_differs_by_a_kept_negative_zero(self):
        differing = {c["id"] for c in load()["cases"] if c.get("same_as_v1") is False}
        self.assertEqual(differing, DIFFERING_FROM_V1)
        for c in load()["cases"]:
            if c.get("same_as_v1") is False:
                self.assertIn(NEG_ZERO, c["expect"]["bf16"])
                self.assertIn("keep a base -0", c["differs_from_v1_because"])


class Behaviour(unittest.TestCase):
    def get(self, cid):
        return next(c for c in load()["cases"] if c["id"] == cid)

    def test_a_zero_update_leaves_every_byte_of_the_tensor_unchanged(self):
        c = self.get("sham-whole-tensor-unchanged")
        self.assertEqual(c["expect"]["bf16"], c["base"]["bf16"])
        self.assertIn(NEG_ZERO, c["base"]["bf16"])

    def test_columns_outside_j_keep_their_bits_including_special_values(self):
        c = self.get("out-keeps-special-values")
        cols = c["op"]["shape"][1]
        for i, (old, new) in enumerate(zip(c["base"]["bf16"], c["expect"]["bf16"])):
            if i % cols not in c["op"]["columns"]:
                self.assertEqual(old, new, i)

    def test_the_variants_that_motivate_the_op_change_the_result(self):
        self.assertIn("zero_padded_full", self.get("out-keeps-special-values")["under_wrong_semantics"])
        self.assertIn("zero_update_not_kept", self.get("sham-whole-tensor-unchanged")["under_wrong_semantics"])
        self.assertIn("low_rank_from_zero", self.get("k-order-start-from-base")["under_wrong_semantics"])

    def test_a_zero_in_a_or_b_leaves_that_row_or_column_untouched(self):
        c = self.get("k1-zero-row-of-a-untouched")
        cols = c["op"]["shape"][1]
        row1 = range(1 * cols, 2 * cols)
        self.assertTrue(all(c["expect"]["bf16"][i] == c["base"]["bf16"][i] for i in row1))
        self.assertEqual({c["base"]["bf16"][4], c["base"]["bf16"][6]}, {NEG_ZERO})
        self.assertNotEqual(c["expect"]["bf16"][0], c["base"]["bf16"][0])  # row 0 is written
        d = self.get("k1-zero-entry-of-b-untouched")
        col1 = [r * d["op"]["shape"][1] + 1 for r in range(3)]
        self.assertTrue(all(d["expect"]["bf16"][i] == d["base"]["bf16"][i] for i in col1))
        self.assertNotEqual(d["expect"]["bf16"][3], d["base"]["bf16"][3])  # column 3 is written

    def test_written_is_decided_on_operands_not_on_rounded_products(self):
        c = self.get("nonzero-operands-underflowing-product-f32")
        self.assertEqual((c["base"]["bf16"], c["expect"]["bf16"]), ([NEG_ZERO], ["0000"]))

    def test_near_cancellation_leaves_a_tiny_nonzero_and_exact_cancellation_gives_positive_zero(self):
        self.assertEqual(self.get("near-cancellation")["expect"]["bf16"][0], "b080")  # -2^-30
        self.assertEqual(self.get("exact-cancellation")["expect"]["bf16"], ["0000"])

    def test_ties_go_to_even_in_both_directions(self):
        self.assertEqual(self.get("tie-to-even-down")["expect"]["bf16"][0], "3f80")
        self.assertEqual(self.get("tie-to-even-up")["expect"]["bf16"][0], "3f82")
        self.assertNotEqual(self.get("just-above-tie")["expect"]["bf16"][0], self.get("tie-to-even-down")["expect"]["bf16"][0])
        self.assertEqual(self.get("just-below-tie")["expect"]["bf16"][0], "3f81")


class Frozen(unittest.TestCase):
    """FREEZE.json holds the frozen files at their recorded bytes; any change is a new version (v3), never an edit to v2."""

    def test_the_frozen_files_match_their_recorded_hashes(self):
        freeze = json.load(open(os.path.join(HERE, "FREEZE.json")))
        for name, want in freeze["sha256"].items():
            with open(os.path.join(HERE, name), "rb") as f:
                self.assertEqual(hashlib.sha256(f.read()).hexdigest(), want, name)
        ind = freeze["independent_reproduction"]
        for key, name in (("evaluator_sha256", "evaluate.py"), ("record_sha256", "RECORD.json"), ("history_sha256", "HISTORY.md")):
            with open(os.path.join(HERE, "independent", name), "rb") as f:
                self.assertEqual(hashlib.sha256(f.read()).hexdigest(), ind[key], name)
        self.assertEqual((ind["cases_reproduced"], ind["cases"]), (91, 91))

    def test_the_v1_files_it_extends_are_unchanged(self):
        recorded = json.load(open(os.path.join(HERE, "FREEZE.json")))["extends_v1_unchanged"]
        for name, want in recorded.items():
            with open(os.path.join(V1, name), "rb") as f:
                self.assertEqual(hashlib.sha256(f.read()).hexdigest(), want, name)


class Independent(unittest.TestCase):
    """The committed independent reproduction (see independent/HISTORY.md)."""

    def test_the_evaluator_reproduces_every_case_blind_and_the_comparison_can_fail(self):
        ind = os.path.join(HERE, "independent")
        r = subprocess.run([sys.executable, "-I", "compare.py", "results.json", "mapping.json", "inputs.json"],
                           cwd=ind, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        rec = json.load(open(os.path.join(ind, "RECORD.json")))
        self.assertEqual((rec["reproduced"], rec["cases"], rec["not_reproduced"]), (91, 91, []))
        self.assertEqual(rec["comparison_self_test"]["undetected"], [])
        self.assertTrue(rec["v1_equivalence"]["differing_is_exactly_the_documented_one"])
        self.assertEqual(rec["documents_sha256"]["v2/SPEC.md"],
                         hashlib.sha256(open(os.path.join(HERE, "SPEC.md"), "rb").read()).hexdigest())

    def test_the_inputs_hold_no_expected_values_or_names(self):
        for case in json.load(open(os.path.join(HERE, "independent", "inputs.json")))["cases"]:
            self.assertEqual(sorted(case), sorted({"id", "base", "op"} | ({"tensor_role"} & set(case))))
            self.assertRegex(case["id"], r"^case-\d{3}$")

    def test_the_evaluator_imports_only_the_standard_library_and_nothing_of_the_generator(self):
        src = open(os.path.join(HERE, "independent", "evaluate.py")).read()
        for line in (l for l in src.splitlines() if l.startswith(("import ", "from "))):
            self.assertIn(line.split()[1], {"json", "sys", "fractions"}, line)


class Coverage(unittest.TestCase):
    def test_every_refusal_the_op_can_give_is_exercised_exactly_by_one_defect(self):
        seen = {c["expect"]["refuse"] for c in load()["cases"] if "refuse" in c["expect"]}
        self.assertEqual(seen, BASE_REFUSALS)

    def test_the_native_route_agrees_on_the_f64_profile_cases(self):
        self.assertGreater(load()["native_cross_checked_cases"], 40)  # the generator asserts agreement while counting


if __name__ == "__main__":
    unittest.main()
