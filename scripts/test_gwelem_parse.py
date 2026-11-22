"""Parser qualification for GW-ELEMENT-1: positive, negative and truncated outputs, using real token ids."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gwelem_parse as ps  # noqa: E402
import gwtemplate_baseline_plan as bp  # noqa: E402

TOK = bp.tokenizer()
SPACE = TOK.encode(" ", add_special_tokens=False).ids[0]
SPECIAL = frozenset({TOK.token_to_id("<eos>"), TOK.token_to_id("<end_of_turn>")})


def ids(*parts):
    out = []
    for p in parts:
        if p in ("<eos>", "<end_of_turn>"):
            out.append(TOK.token_to_id(p))
        else:
            e = TOK.encode(p, add_special_tokens=False).ids
            assert len(e) == 1, (p, e)
            out += e
    return out


def dec(i):
    return TOK.decode([i])


def num(cont, z):
    return ps.classify_number(cont, z, SPACE, dec, SPECIAL)


def sym(cont, s):
    return ps.classify_symbol(cont, TOK.encode(" " + s, add_special_tokens=False).ids[0], dec, SPECIAL)


class Number(unittest.TestCase):
    def test_positive(self):
        self.assertTrue(num(ids(" ", "8", ".", "\n"), 8)["correct"])
        self.assertTrue(num(ids(" ", "2", "6", ","), 26)["correct"])
        self.assertTrue(num(ids(" ", "1", "1", "8", "<end_of_turn>"), 118)["correct"])
        self.assertTrue(num(ids(" ", "7", "9", "\n", "\n"), 79)["correct"])

    def test_negative_format(self):
        self.assertEqual(num(ids("**", "8", "**"), 8)["reason"], "format: first_token_not_space")
        self.assertEqual(num(ids(" is", " ", "8"), 8)["reason"], "format: first_token_not_space")
        self.assertEqual(num(ids(" ", "**", "8"), 8)["reason"], "format: no_digits")
        self.assertEqual(num(ids(" ", "8", "th"), 8)["reason"], "format: bad_terminator")

    def test_negative_value(self):
        self.assertEqual(num(ids(" ", "9", "."), 8)["reason"], "wrong_value")
        self.assertEqual(num(ids(" ", "0", "8", "."), 8)["reason"], "wrong_value")
        self.assertEqual(num(ids(" ", "2", "6", "0", "."), 26)["reason"], "wrong_value")
        self.assertEqual(num(ids(" ", "2", "."), 26)["reason"], "wrong_value")

    def test_truncated(self):
        self.assertEqual(num(ids(" ", "1", "1", "8"), 118)["reason"], "truncated")
        self.assertEqual(num(ids(" ", "2", "6"), 26)["reason"], "truncated")
        self.assertEqual(num(ids(" ", "8"), 8)["reason"], "truncated")
        self.assertEqual(num(ids(" ", "1", "1", "1", "1", "1", "1", "1"), 1)["reason"], "truncated")


class Symbol(unittest.TestCase):
    def test_positive(self):
        self.assertTrue(sym(ids(" O", ".", "\n"), "O")["correct"])
        self.assertTrue(sym(ids(" Fe", ","), "Fe")["correct"])
        self.assertTrue(sym(ids(" Cl", "<end_of_turn>"), "Cl")["correct"])
        self.assertTrue(sym(ids(" C", " ("), "C")["correct"])

    def test_negative(self):
        self.assertEqual(sym(ids(" Cl", "."), "C")["reason"], "wrong_first_token")
        self.assertEqual(sym(ids(" C", "l", "."), "C")["reason"], "symbol_continues")
        self.assertEqual(sym(ids("**", " O"), "O")["reason"], "wrong_first_token")

    def test_truncated(self):
        self.assertEqual(sym(ids(" O"), "O")["reason"], "truncated")


if __name__ == "__main__":
    unittest.main()
