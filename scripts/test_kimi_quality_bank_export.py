"""Producer-side authority checks without a model or PyTorch installation."""
import hashlib
from pathlib import Path
import tempfile
import unittest

from kimi_quality_bank_export import seal_sequence


class SequenceAuthorityTests(unittest.TestCase):
    def test_digest_describes_persisted_bytes_and_detects_same_size_changes(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "seq_0.f32"
            payload = b"\0\0\0\0" * (300_000)  # crosses the streaming buffer
            path.write_bytes(payload)
            first = seal_sequence(path)
            self.assertEqual(first, {
                "len": len(payload), "sha256": hashlib.sha256(payload).hexdigest()
            })
            with path.open("r+b") as stream:
                stream.seek(len(payload) - 1)
                stream.write(b"\1")
            second = seal_sequence(path)
            self.assertEqual(first["len"], second["len"])
            self.assertNotEqual(first["sha256"], second["sha256"])

    def test_move_preserves_authority_and_missing_payload_refuses(self):
        with tempfile.TemporaryDirectory() as root:
            original = Path(root) / "original.f32"
            moved = Path(root) / "moved.f32"
            original.write_bytes(b"\0\0\x80\x3f")
            before = seal_sequence(original)
            original.rename(moved)
            self.assertEqual(before, seal_sequence(moved))
            with self.assertRaises(FileNotFoundError):
                seal_sequence(original)


if __name__ == "__main__":
    unittest.main()
