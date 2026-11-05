"""Protocol and artifact integrity tests; no model outcomes."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gwcar1b_runtime as runtime  # noqa: E402
from gwcar1b_sources import FEATURES, WIDTH, fit  # noqa: E402
from test_gwcar1b_sources import fixture  # noqa: E402


class RuntimeTests(unittest.TestCase):
    def test_bound_population_and_control_derangement(self):
        train, rows, proposal = runtime.population()
        self.assertEqual(len(train), 396)
        self.assertEqual(len(rows), 201)
        self.assertEqual(len(proposal["candidate_token_ids"]), 57)
        self.assertFalse({r["subject_id"] for r in train} & {r["subject_id"] for r in rows})
        self.assertEqual(sorted(r["donor_index"] for r in rows), list(range(201)))
        for row in rows:
            donor = rows[row["donor_index"]]
            self.assertNotEqual(row["subject_id"], donor["subject_id"])
            self.assertEqual((row["split"], row["template"]), (donor["split"], donor["template"]))

    def test_artifact_shape_identity_and_finiteness(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            values = np.arange(12).reshape(3, 4)
            entry = runtime.artifact(root / "values.f32", values)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"artifacts": [entry]}))
            np.testing.assert_array_equal(runtime.array(manifest, "values.f32", (3, 4)), values)
            with self.assertRaisesRegex(ValueError, "geometry"):
                runtime.array(manifest, "values.f32", (2, 6))
            (root / "values.f32").write_bytes(b"\x00" * 48)
            with self.assertRaisesRegex(ValueError, "identity"):
                runtime.array(manifest, "values.f32", (3, 4))
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                runtime.artifact(root / "bad.f32", np.array([np.nan]))

    def test_draft_and_mutated_protocol_refuse_execution(self):
        expected = {"status": "draft_before_capture", "executable_sha256": None,
                    "protocol_sha256": None, "model_metadata": {}, "container": "/unused"}
        with tempfile.TemporaryDirectory() as root, patch.object(runtime, "expected", return_value=expected):
            path = Path(root) / "draft.json"
            runtime.write_new(path, expected, "protocol_sha256")
            with self.assertRaisesRegex(ValueError, "frozen"):
                runtime.validate(path)
            runtime.validate(path, frozen=False)
            mutated = json.loads(path.read_text())
            mutated["container"] = "/different"
            path.write_text(json.dumps(mutated))
            with self.assertRaisesRegex(ValueError, "differs"):
                runtime.validate(path, frozen=False)

    def test_rehashed_candidate_tampering_still_fails_source_reproduction(self):
        rows, carriers, features = fixture()
        bank = fit(rows, carriers, features)
        fields = lambda row: {key: getattr(row, key) for key in ("row_id", "subject_id", "relation", "template", "split")}
        confirmation = [{"row_id": f"fresh-{i}", "subject_id": f"new-{i}", "relation": "capital",
                         "template": "t0", "split": "validation", "donor_index": (i + 1) % 201}
                        for i in range(201)]
        protocol = {"protocol_sha256": "synthetic", "train_rows": [fields(r) for r in rows],
                    "confirmation_rows": confirmation}
        capture = {f"{name}.f32": np.zeros((597, WIDTH), dtype="<f4") for name in FEATURES}
        capture["carrier-before.f32"] = np.ones((597, WIDTH), dtype="<f4")
        with tempfile.TemporaryDirectory() as root, patch.object(runtime, "validate", return_value=protocol), \
                patch.object(runtime, "checked_capture", return_value=capture), patch.object(runtime, "fit", return_value=bank):
            root = Path(root)
            capture_path = root / "capture.json"
            capture_path.write_text("{}")
            output = root / "candidates"
            runtime.build_candidates(root / "protocol.json", capture_path, output)
            path = output / "candidates.json"
            runtime.check_candidates(root / "protocol.json", path)
            payload = output / "carriers.f32"
            with payload.open("r+b") as handle:
                handle.seek((6 * WIDTH) * 4)  # first query, global-mean source
                handle.write(np.array([987.0], dtype="<f4").tobytes())
            manifest = json.loads(path.read_text())
            manifest["artifacts"][0]["sha256"] = runtime.sha256(payload)
            manifest["candidates_sha256"] = runtime.canonical_hash(manifest, "candidates_sha256")
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "declared source"):
                runtime.check_candidates(root / "protocol.json", path)


if __name__ == "__main__":
    unittest.main()
