"""Synthetic tests for Test 3 Stage 4: the power guard, the gate rule and the admission arm inventory (no model)."""
from __future__ import annotations

import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gwtemplate_test3_holdout_admit as adm  # noqa: E402
import gwtemplate_test3_holdout_analyze as an  # noqa: E402
import gwtemplate_test3_holdout_run as hr  # noqa: E402
import gwtemplate_test3_stage2_analyze as s2  # noqa: E402

TICKER = "import sys,time\nfor i in range(20):\n    open(sys.argv[1],'a').write('x')\n    time.sleep(0.05)\n"


class PowerGuardTests(unittest.TestCase):
    def _run(self, probes):
        seq = iter(probes)
        g = hr.PowerGuard(probe=lambda: next(seq, True), interval=0.2)
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "ticks"
            t0 = time.time()
            rc = g.run([sys.executable, "-c", TICKER, f], label="t")
            return rc, g.events, time.time() - t0, len(f.read_text())

    def test_on_mains_never_pauses(self):
        rc, ev, wall, n = self._run([True] * 50)
        self.assertEqual((rc, ev, n), (0, [], 20))

    def test_battery_pauses_then_resumes(self):
        rc, ev, wall, n = self._run([False, False, False, True])
        self.assertEqual(rc, 0)
        self.assertEqual([e["event"] for e in ev], ["paused_on_battery", "resumed_on_mains"])
        self.assertEqual(n, 20)
        self.assertGreater(ev[1]["t"] - ev[0]["t"], 0.35)
        self.assertGreater(wall, 1.0 + 0.35)

    def test_waived_guard_logs_and_never_pauses(self):
        seq = iter([False, False, True])
        g = hr.PowerGuard(probe=lambda: next(seq, True), interval=0.2, enforce=False)
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "ticks"
            t0 = time.time()
            rc = g.run([sys.executable, "-c", TICKER, f], label="t")
            wall = time.time() - t0
            n = len(f.read_text())
        self.assertEqual((rc, n), (0, 20))
        self.assertEqual([e["event"] for e in g.events], ["on_battery_waived", "mains_returned"])
        self.assertLess(wall, 2.0)

    def test_child_does_not_progress_while_paused(self):
        seq = iter([False] + [False] * 3 + [True] * 50)
        g = hr.PowerGuard(probe=lambda: next(seq, True), interval=0.2)
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "ticks"
            import threading
            sizes = []

            def watch():
                while not g.events:
                    time.sleep(0.01)
                time.sleep(0.1)
                a = len(f.read_text()) if f.exists() else 0
                time.sleep(0.4)
                b = len(f.read_text()) if f.exists() else 0
                sizes.append((a, b, len(g.events)))
            th = threading.Thread(target=watch)
            th.start()
            g.run([sys.executable, "-c", TICKER, f], label="t")
            th.join()
        a, b, nev = sizes[0]
        self.assertEqual(a, b)  # no ticks while stopped


class GateTests(unittest.TestCase):
    def groups(self, share_ci, p2, p3, b3m, b3x, other):
        g = {}
        for kind in ("C1", "C2"):
            for ph, sh in ((s2.POOLED, 0.9), ("P2", p2), ("P3", p3)):
                g[f"{kind}|capital|{ph}|L1|holdout"] = {
                    "shares": {"r'(Y)": {"mean": sh, "ci95": share_ci}, "other_country_r'": {"mean": other}},
                    "b3_mean": {"mean": b3m}, "b3_max": {"max": b3x, "class": 1, "support": 20}}
        return g

    def test_pass(self):
        v = an.gates(self.groups([0.85, 1.0], 0.9, 0.9, 0.1, 0.5, 0.0))
        self.assertEqual(v["C1|capital"]["verdict"], "pass")
        self.assertEqual(v["C2|capital"]["B"]["b2_carried_from_development"]["result"], 6290)

    def test_switch_lower_bound(self):
        self.assertEqual(an.gates(self.groups([0.79, 1.0], 0.9, 0.9, 0, 0, 0))["C1|capital"]["verdict"], "no switch")
        self.assertEqual(an.gates(self.groups([0.85, 1.0], 0.55, 0.9, 0, 0, 0))["C1|capital"]["verdict"], "no switch")

    def test_b3_and_other(self):
        v = an.gates(self.groups([0.85, 1.0], 0.9, 0.9, 0.51, 1.01, 0.06))["C1|capital"]["verdict"]
        self.assertEqual(v, "axis B fail: b3_mean_pass, b3_max_pass, other_top1_pass")

    def test_missing_b3_max_fails(self):
        v = an.gates(self.groups([0.85, 1.0], 0.9, 0.9, 0.1, None, 0.0))["C1|capital"]["verdict"]
        self.assertEqual(v, "axis B fail: b3_max_pass")


class InventoryTests(unittest.TestCase):
    def test_expected_arms(self):
        self.assertEqual(adm.expected_arms("currency_code_iso4217", 2), ["ID-window-L1"])
        self.assertEqual(adm.expected_arms("capital", 1),
                         ["ID-window-L1", "S-window-L1", "C0-D1-L1", "C1-window-L1", "C2-window-L1"])


if __name__ == "__main__":
    unittest.main()
