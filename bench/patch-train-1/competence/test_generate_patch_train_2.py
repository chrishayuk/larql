"""Tests for the PATCH-TRAIN-2 generator. DUMMY seeds only: the real seeds in patch_train_2_seeds.json are never used here.
    python3 -I test_generate_patch_train_2.py
Needs only the local tokenizer cache. Nothing is written outside a temporary directory."""
import collections
import json
import os
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import generate_patch_train_2 as g  # noqa: E402

REAL = set(g.load_seeds().values())
DUMMY = {"cal": 11, "conf": 22, "ret": 33}
TOKEN_COUNT, TOK = g.make_token_counter()


def build(seeds=DUMMY, directory=None):
    """cal -> ret -> conf, as the real order of operations, into `directory`."""
    docs, excluded = {}, g.discovery_names()
    for split in ("cal", "ret", "conf"):
        doc = g.generate_split(split, seeds[split], set(excluded), TOKEN_COUNT, {"dummy": True})
        excluded.update(e["name"] for e in doc["entities"])
        docs[split] = doc
    return docs


class GeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docs = build()

    def test_dummy_seeds_are_not_real(self):
        self.assertFalse(set(DUMMY.values()) & REAL)

    def test_deterministic_and_seed_dependent(self):
        again = build()
        self.assertEqual(json.dumps(self.docs, sort_keys=True), json.dumps(again, sort_keys=True))
        other = build({"cal": 12, "conf": 23, "ret": 34})
        self.assertNotEqual([e["name"] for e in self.docs["conf"]["entities"]], [e["name"] for e in other["conf"]["entities"]])

    def test_sizes(self):
        self.assertEqual([len(self.docs[s]["entities"]) for s in ("cal", "conf", "ret")], [8, 24, 24])
        self.assertEqual(len(self.docs["cal"]["pairs"]), 4)
        self.assertEqual(len(self.docs["conf"]["pairs"]), 12)

    def test_entity_disjoint_from_discovery_and_across_splits(self):
        disc = g.discovery_names()
        sets = {s: {e["name"] for e in d["entities"]} for s, d in self.docs.items()}
        for s, names in sets.items():
            self.assertEqual(len(names), len(self.docs[s]["entities"]), f"duplicate names in {s}")
            self.assertFalse(names & disc, f"{s} overlaps discovery")
        self.assertFalse(sets["cal"] & sets["conf"] or sets["cal"] & sets["ret"] or sets["conf"] & sets["ret"])

    def test_answer_balance_cal(self):
        ans = [e["answer"] for e in self.docs["cal"]["entities"]]
        self.assertEqual(max(collections.Counter(ans).values()), 1)          # each answer at most once
        self.assertTrue(set(ans) <= set(g.ANSWERS))
        for p in self.docs["cal"]["pairs"]:
            a, b = p["entities"]
            self.assertNotEqual(ans[a], ans[b])

    def test_answer_balance_conf(self):
        ents = self.docs["conf"]["entities"]
        self.assertEqual(sorted(collections.Counter(e["answer"] for e in ents).values()), [2] * 12)   # each answer exactly twice
        self.assertEqual(set(collections.Counter(e["answer"] for e in ents)), set(g.ANSWERS))
        for p in self.docs["conf"]["pairs"]:
            a, b = p["entities"]
            self.assertNotEqual(ents[a]["answer"], ents[b]["answer"])

    def test_answer_balance_ret(self):
        c = collections.Counter(e["answer"] for e in self.docs["ret"]["entities"])
        self.assertEqual(sorted(c.values()), [2] * 12)

    def test_answer_assignment_never_looks_at_a_model(self):
        # the assignment is a pure function of (rng, split): same stream -> same answers, regardless of names or tokenisation
        import random
        for split in ("cal", "conf", "ret"):
            a = g.assign_answers(random.Random(5), split)
            self.assertEqual(a, g.assign_answers(random.Random(5), split))

    def test_pairs_and_primary_order(self):
        for s in ("cal", "conf"):
            for i, p in enumerate(self.docs[s]["pairs"]):
                self.assertEqual(p["entities"], [2 * i, 2 * i + 1])
                self.assertIn(p["primary_first"], p["entities"])
                self.assertEqual(self.docs[s]["entities"][p["primary_first"]]["pair"], i)

    def test_token_range_and_format(self):
        lo, hi = g.TOKEN_RANGE
        for d in self.docs.values():
            for e in d["entities"]:
                self.assertTrue(lo <= TOKEN_COUNT(e["name"]) <= hi)
                self.assertEqual(e["token_count"], TOKEN_COUNT(e["name"]))
                self.assertTrue(e["name"].endswith("ia"))
        bad = [a for a in g.ANSWERS if len(TOK(a, add_special_tokens=False)["input_ids"]) != 1]
        self.assertEqual(bad, [])

    def test_near_matches_recorded_not_filtered(self):
        for d in self.docs.values():
            for e in d["entities"]:
                self.assertIn("nearest_discovery_name", e)
                self.assertIn("normalised_edit_distance", e["nearest_discovery_name"])
        self.assertIn("nearest_other_split_name", self.docs["conf"]["entities"][0])

    def test_edit_distance(self):
        self.assertEqual(g.edit_distance("abc", "abc"), 0)
        self.assertEqual(g.edit_distance("kitten", "sitting"), 3)

    def test_retention_set_is_24_prompts_with_first_patch_prompt(self):
        ret = self.docs["ret"]
        xa = "The capital of Zzzia is"
        prompts = g.retention_prompts(ret, xa)
        self.assertEqual(len(prompts), 24)
        self.assertEqual(prompts[0], xa)
        names = [e["name"] for e in ret["entities"]]
        self.assertEqual(prompts[1:], [f"The capital of {n} is" for n in names[:23]])
        self.assertNotIn(f"The capital of {names[23]} is", prompts)            # 24th entity held back for leakage
        for t in ret["heldout_templates"]:
            self.assertFalse(any(t.format(e=n) in prompts for n in names))      # no held-out paraphrase in RET
        self.assertEqual(len(set(prompts)), 24)

    def test_retention_prompts_never_touch_cal_or_conf(self):
        prompts = g.retention_prompts(self.docs["ret"], "The capital of Zzzia is")
        others = {e["name"] for s in ("cal", "conf") for e in self.docs[s]["entities"]}
        self.assertFalse(any(n in p for p in prompts for n in others))

    def test_uncovered_unrelated_set(self):
        for s, others in (("conf", 22), ("cal", 6)):
            u = g.uncovered_unrelated(self.docs[s], self.docs["ret"], 0)
            self.assertEqual(len(u), others + 1)
            members = {self.docs[s]["entities"][i]["name"] for i in self.docs[s]["pairs"][0]["entities"]}
            self.assertFalse(set(u) & members)                                    # the pair itself is not in it
            ret_covered = {e["name"] for e in self.docs["ret"]["entities"][:23]}
            self.assertFalse(set(u) & ret_covered)                                # RET-covered entities are not gated on
            self.assertIn(self.docs["ret"]["entities"][23]["name"], u)

    def test_cli_dummy_roundtrip_and_refusals(self):
        with tempfile.TemporaryDirectory() as d:
            for split, seed in (("cal", 41), ("ret", 42), ("conf", 43)):
                g.main(["--split", split, "--dummy-seed", str(seed), "--out-dir", d])
                doc = g.read_split(os.path.join(d, g.OUT_NAME.format(split=split)))
                self.assertTrue(doc["dummy"] and doc["seed"] == seed)
            with self.assertRaises(SystemExit):                                     # never regenerate over a split
                g.main(["--split", "cal", "--dummy-seed", "44", "--out-dir", d])
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit):                                     # conf needs cal and ret first
                g.main(["--split", "conf", "--dummy-seed", "45", "--out-dir", d])
        for real in REAL:
            with tempfile.TemporaryDirectory() as d, self.assertRaises(SystemExit):  # a dummy seed may not be a real seed
                g.main(["--split", "cal", "--dummy-seed", str(real), "--out-dir", d])
        with self.assertRaises(SystemExit):                                         # exactly one of --real / --dummy-seed
            g.main(["--split", "cal"])

    def test_real_generation_is_guarded(self):
        with tempfile.TemporaryDirectory() as d, self.assertRaises(SystemExit):     # real splits only next to the generator
            g.main(["--split", "cal", "--real", "--out-dir", d])
        # conf without a frozen protocol is refused regardless of the git state
        with self.assertRaises(SystemExit):
            g.main(["--split", "conf", "--real"])


if __name__ == "__main__":
    unittest.main()
