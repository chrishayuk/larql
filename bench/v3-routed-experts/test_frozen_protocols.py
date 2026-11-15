"""The frozen protocols are what was frozen; nothing here starts a process."""
import unittest

import block_time_model as bt
import frozen_protocols as fp
import link_block_frames_model as bm


class Frozen(unittest.TestCase):
    def test_no_frozen_file_has_changed(self):
        self.assertEqual(fp.changed(), [], "a frozen protocol file changed: that is a new revision")

    def test_both_protocols_are_recorded_under_their_revision_names(self):
        protocols = fp.recorded()["protocols"]
        self.assertEqual(set(protocols), {bm.ADDENDUM, bt.PROTOCOL})
        self.assertEqual(bm.ADDENDUM, "LINK-2 block-frame addendum revision 2")
        self.assertEqual(bt.PROTOCOL, "BLOCK-TIME-1 revision 2")

    def test_each_record_covers_the_document_and_its_rules(self):
        protocols = fp.recorded()["protocols"]
        self.assertIn("LINK-2-block-frames-revision-2.md", protocols[bm.ADDENDUM]["sha256"])
        self.assertIn("link_pair_model.py", protocols[bm.ADDENDUM]["sha256"])
        self.assertIn("BLOCK-TIME-1.md", protocols[bt.PROTOCOL]["sha256"])
        self.assertIn("block_time_model.py", protocols[bt.PROTOCOL]["sha256"])
        for record in protocols.values():
            self.assertIn("block_frames.json", record["sha256"])

    def test_a_changed_hash_is_reported_by_protocol_and_file(self):
        original = fp.recorded

        def tampered():
            record = original()
            record["protocols"][bt.PROTOCOL]["sha256"]["block_time_model.py"] = "0" * 64
            return record
        fp.recorded = tampered
        try:
            self.assertEqual(fp.changed(), [(bt.PROTOCOL, "block_time_model.py")])
        finally:
            fp.recorded = original


if __name__ == "__main__":
    unittest.main()
