"""The cross-language golden: a container LARQL compiled from a structured patch, judged by this independent verifier.

`conformance/structured_patch/` holds the fixture base, the `.vlp` LQL wrote, and the container LQL compiled (written by
`crates/larql-lql/tests/weight_patch_lql/cross_language.rs`, which pins the Rust half byte for byte). This pins the other
half: the verifier says SATISFIED of the committed trio, with the digests recorded, and a violation or a refusal for each
way of breaking it. The patch carries a near-cancelling element that no f32 delta can write, so a pass here is a pass on
the case the delta route could not express.
"""

import hashlib
import json
import os
import shutil
import struct
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from larql_contract_verifier import cli  # noqa: E402
from larql_contract_verifier.contracts import structured_patch as S  # noqa: E402
from refcheck import container as C  # noqa: E402

GOLDEN = os.path.join(ROOT, "conformance", "structured_patch")
BF16 = "BF16"
F32_PROFILE = "f32-seq-rne1/v1"
STALE_INDEX_SHA = "0" * 64
BF16_BYTES = 2
SIGN_BIT = 0x8000


def read_json(name):
    with open(os.path.join(GOLDEN, name)) as f:
        return json.load(f)


class Golden(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.work)
        self.golden = read_json("golden.json")
        self.verdict = read_json("verdict.json")
        self.base = os.path.join(GOLDEN, "base")
        self.patch = os.path.join(GOLDEN, "patch.vlp")
        self.compiled = os.path.join(GOLDEN, "compiled")

    def copy_compiled(self, name="mutant"):
        root = os.path.join(self.work, name)
        shutil.copytree(self.compiled, root)
        return root

    def tensor_ref(self, root, name):
        return next(t for t in C.open_container(root).tensors(BF16) if t.name == name)

    def reseal(self, root, seg):
        """Rewrite the segment and payload digests of one representation to match its bytes, as a careful tamperer would,
        so the container reads as intact and only the verifier's comparison with the recomputed bits can object."""
        with open(seg.path, "rb") as f:
            data = f.read()
        payload = data[seg.payload_start:seg.payload_start + seg.payload_bytes]
        index_path = os.path.join(root, "index.json")
        with open(index_path) as f:
            index = json.load(f)
        entry = index["representations"][seg.rep_id]
        entry["segment_sha256"] = hashlib.sha256(data).hexdigest()
        entry["payload_sha256"] = hashlib.sha256(payload).hexdigest()
        with open(index_path, "w") as f:
            json.dump(index, f)

    def flip(self, root, tensor, row, col, cols, seal=True):
        """Flip the lowest mantissa bit of one BF16 element in place (and, by default, re-seal the digests)."""
        container = C.open_container(root)  # opened before the flip: it checks the digests it is about to go stale
        ref = next(t for t in container.tensors(BF16) if t.name == tensor)
        seg = container.segments[ref.rep_id]
        at = ref.abs_start + (row * cols + col) * BF16_BYTES
        with open(ref.segment_path, "r+b") as f:
            f.seek(at)
            (bits,) = struct.unpack("<H", f.read(BF16_BYTES))
            f.seek(at)
            f.write(struct.pack("<H", bits ^ 1))
        if seal:
            self.reseal(root, seg)

    def judge(self, container, patch=None, base=None):
        return cli.evaluate_patch(S.CONTRACT, patch or self.patch, base or self.base, container)

    def cols(self):
        return self.tensor_ref(self.base, self.golden["tensor"]).shape[1]


class Verdict(Golden):
    def test_the_committed_trio_is_satisfied_with_the_recorded_digests(self):
        out = self.judge(self.compiled)
        self.assertEqual(out["status"], self.verdict["status"], out)
        self.assertEqual(out["contract"], self.verdict["contract"])
        self.assertEqual(out["violations"], 0)
        self.assertEqual(out["numerical_satisfaction"], "satisfied")
        self.assertEqual((out["ledger_integrity"], out["provenance"]), ("not_checked", "not_checked"))
        self.assertEqual(out["patch"]["sha256"], self.verdict["patch_sha256"])
        self.assertEqual(out["base"]["sha256"], self.verdict["base_sha256"])
        self.assertEqual(out["container"]["sha256"], self.verdict["container_sha256"])
        measured = {m["name"]: m["value"] for m in out["measurements"]}
        self.assertEqual(measured["elements compared"], str(self.verdict["elements_compared"]))
        self.assertEqual(measured["elements expected to change"], str(self.verdict["elements_expected_to_change"]))
        self.assertEqual(measured["elements differing from the recomputed bits"], "0")
        self.assertEqual(len(self.golden["changed_elements"]), self.verdict["elements_expected_to_change"])

    def test_the_verifier_is_repeatable(self):
        self.assertEqual(self.judge(self.compiled)["container"], self.judge(self.compiled)["container"])

    def test_the_ledger_is_not_what_decides(self):
        root = self.copy_compiled("noledger")
        with open(os.path.join(root, "weight_patches.json"), "w") as f:
            f.write("not even json")
        self.assertEqual(self.judge(root)["status"], "satisfied")

    def test_the_uncompiled_base_is_violated_at_every_changed_element(self):
        out = self.judge(self.base)
        self.assertEqual((out["status"], out["violations"]), ("violated", len(self.golden["changed_elements"])))


class Mutations(Golden):
    def test_a_flip_that_leaves_the_container_digests_stale_is_not_judged_at_all(self):
        root = self.copy_compiled()
        row, col = self.golden["near_cancelling_element"]
        self.flip(root, self.golden["tensor"], row, col, self.cols(), seal=False)
        out = self.judge(root)
        self.assertEqual((out["status"], out["violations"]), ("could_not_evaluate", 0), out)
        self.assertTrue(any("sha256 mismatch" in r for r in out["reasons"]), out["reasons"])

    def test_a_changed_element_with_one_bit_flipped_is_found(self):
        root = self.copy_compiled()
        row, col = self.golden["near_cancelling_element"]
        self.flip(root, self.golden["tensor"], row, col, self.cols())
        out = self.judge(root)
        self.assertEqual((out["status"], out["violations"]), ("violated", 1), out)
        self.assertIn(f"({row}, {col})", out["offending"][0]["detail"])

    def test_an_untouched_element_with_one_bit_flipped_is_found(self):
        changed = {tuple(e) for e in self.golden["changed_elements"]}
        row, col = next((r, c) for r in range(2) for c in range(self.cols()) if (r, c) not in changed)
        root = self.copy_compiled()
        self.flip(root, self.golden["tensor"], row, col, self.cols())
        out = self.judge(root)
        self.assertEqual((out["status"], out["violations"]), ("violated", 1), out)
        self.assertIn(f"({row}, {col})", out["offending"][0]["detail"])

    def test_another_tensor_with_one_bit_flipped_is_found(self):
        tensors = [t for t in C.open_container(self.compiled).tensors(BF16) if t.name != self.golden["tensor"]]
        other = tensors[0]
        root = self.copy_compiled()
        self.flip(root, other.name, 0, 0, other.shape[-1] if len(other.shape) > 1 else 1)
        out = self.judge(root)
        self.assertEqual(out["status"], "violated", out)
        self.assertTrue(out["offending"][0]["name"].endswith(other.name), out["offending"])


class Refusals(Golden):
    def patched(self, edit):
        with open(self.patch) as f:
            doc = json.load(f)
        edit(doc["operations"][0])
        path = os.path.join(self.work, "mutant.vlp")
        with open(path, "w") as f:
            json.dump(doc, f)
        return path

    def test_a_stale_source_identity_is_not_judged(self):
        path = self.patched(lambda op: op["source"].update(index_sha256=STALE_INDEX_SHA))
        out = self.judge(self.compiled, patch=path)
        self.assertEqual(out["status"], "could_not_evaluate", out)
        self.assertEqual(out["violations"], 0)

    def test_a_base_that_is_not_the_one_bound_is_not_judged(self):
        base = os.path.join(self.work, "otherbase")
        shutil.copytree(self.base, base)
        with open(os.path.join(base, "index.json"), "a") as f:
            f.write("\n")
        self.assertEqual(self.judge(self.compiled, base=base)["status"], "could_not_evaluate")

    def test_a_rejected_numerical_profile_is_not_judged(self):
        path = self.patched(lambda op: op.update(numerics=F32_PROFILE))
        out = self.judge(self.compiled, patch=path)
        self.assertEqual(out["status"], "could_not_evaluate", out)
        self.assertTrue(any(F32_PROFILE in r or "profile" in r for r in out["reasons"]), out["reasons"])


if __name__ == "__main__":
    unittest.main()
