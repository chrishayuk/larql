"""GW-ELEMENT-1: the committed spent-confirmation declaration and its guard."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gwelem_design as D  # noqa: E402
import gwelem_spent as SP  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PROTOCOL = ROOT / "docs/gw-element-1-confirmation-protocol.md"


class SpentGuard(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.other = self.tmp / "other-protocol.md"
        self.other.write_text("a different protocol\n")

    def test_declaration_matches_frozen_records(self):
        d = SP.declaration()
        self.assertEqual(d["protocol"]["sha256"], SP._sha(PROTOCOL))
        self.assertEqual(d["spent_targets"], list(D.CONFIRMATION))
        self.assertEqual(d["sealed_split"], sorted(D.CONFIRMATION_SPLIT))
        self.assertEqual((d["registry"]["run"], d["registry"]["verdict_result"]), ("RUN-20261122-023716-01262", 6320))

    def test_refuses_the_spent_protocol(self):
        with self.assertRaisesRegex(SystemExit, "is spent"):
            SP.refuse_if_spent(PROTOCOL, [200])

    def test_refuses_any_spent_or_sealed_target(self):
        for z in (5, 110, 10, "47"):
            with self.assertRaisesRegex(SystemExit, "spent confirmation split"):
                SP.refuse_if_spent(self.other, [z])

    def test_allows_new_protocol_on_unspent_targets(self):
        self.assertEqual(SP.refuse_if_spent(self.other, [3, 8])["checked_targets"], 2)

    def test_missing_or_malformed_declaration_fails_closed(self):
        with self.assertRaisesRegex(SystemExit, "unreadable"):
            SP.refuse_if_spent(self.other, [3], path=self.tmp / "absent.json")
        bad = self.tmp / "bad.json"
        bad.write_text(json.dumps({"schema": "larql.gwelem.spent-confirmation.v1", "status": "open"}))
        with self.assertRaisesRegex(SystemExit, "malformed"):
            SP.refuse_if_spent(self.other, [3], path=bad)

    def test_record_hashes_match_local_evidence_when_present(self):
        for rel, h in SP.declaration()["records_sha256"].items():
            p = ROOT / rel
            if p.exists():
                self.assertEqual(SP._sha(p), h, rel)


if __name__ == "__main__":
    unittest.main()
