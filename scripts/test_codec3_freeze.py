"""CODEC-3's freeze inherits CODEC-2's rules unchanged: checked, not asserted."""
import hashlib
import json
import unittest
from pathlib import Path

FORECASTS = Path(__file__).resolve().parent.parent / "docs/represent/forecasts"
CODEC2 = FORECASTS / "continuation-codec-2.json"
CODEC3 = FORECASTS / "continuation-codec-3.json"
BANK3 = FORECASTS / "continuation-codec-3-token-bank.json"


class InheritanceTests(unittest.TestCase):
    def setUp(self):
        self.raw2 = CODEC2.read_bytes()
        self.c2 = json.loads(self.raw2)
        self.c3 = json.loads(CODEC3.read_text(encoding="utf-8"))

    def test_the_pinned_codec_2_file_is_the_one_on_disk(self):
        pinned = self.c3["acceptance"]["inherited_from"]["sha256"]
        self.assertEqual(hashlib.sha256(self.raw2).hexdigest(), pinned)

    def test_the_rule_metric_and_positions_are_codec_2s_byte_for_byte(self):
        for key in ("rule", "metric", "scored_positions"):
            self.assertEqual(self.c3["acceptance"][key], self.c2["acceptance"][key], key)
        for key in ("rule", "null", "A_yardstick_usable"):
            self.assertEqual(
                self.c3["acceptance"]["guards"][key], self.c2["acceptance"]["guards"][key], key
            )

    def test_the_bank_named_is_the_bank_committed(self):
        bank = json.loads(BANK3.read_text(encoding="utf-8"))
        self.assertEqual(self.c3["bank"]["ids_sha256_u32le"], bank["ids_sha256_u32le"])
        self.assertEqual(len(bank["ids"]), bank["count"])


class RuleBindingTests(unittest.TestCase):
    """The verdict subject is named once, in acceptance.rule_binding, and
    the inherited rule (still CODEC-2's bytes, still saying CH) is read
    through it (maintainer review of #656)."""

    def setUp(self):
        self.c3 = json.loads(CODEC3.read_text(encoding="utf-8"))
        self.acceptance = self.c3["acceptance"]
        self.binding = self.acceptance["rule_binding"]

    def test_the_binding_is_pinned(self):
        self.assertEqual(
            {k: self.binding[k] for k in
             ("inherited_symbol", "primary", "secondary_report_only", "control_report_only")},
            {"inherited_symbol": "CH", "primary": "CH-recent256",
             "secondary_report_only": "CH-recent128", "control_report_only": "CH"},
        )

    def test_the_inherited_symbol_is_the_candidate_the_inherited_rule_names(self):
        symbol = self.binding["inherited_symbol"]
        self.assertTrue(
            self.acceptance["rule"]["subject"].startswith(f"{symbol} is ACCEPTABLE only if"),
            self.acceptance["rule"]["subject"],
        )

    def test_every_bound_role_is_a_distinct_declared_arm(self):
        roles = [self.binding[k] for k in ("primary", "secondary_report_only", "control_report_only")]
        self.assertEqual(len(set(roles)), len(roles))
        for arm in roles:
            self.assertIn(arm, self.acceptance["arms"])

    def test_no_second_authority_names_the_subject(self):
        self.assertNotIn("verdict_subject", self.acceptance)
        self.assertNotIn("substitution", self.acceptance["inherited_from"])
        for arm, text in self.acceptance["arms"].items():
            for role_word in ("PRIMARY", "SECONDARY", "not adjudicated"):
                self.assertNotIn(role_word, text, arm)


if __name__ == "__main__":
    unittest.main()
