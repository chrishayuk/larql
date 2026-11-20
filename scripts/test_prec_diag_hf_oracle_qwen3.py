"""Tests for prec_diag/hf_oracle.py on a synthetic Qwen3 checkpoint.

The checkpoint is a tiny random Qwen3 dense model saved as BF16, with
non-trivial norm weights (Qwen3's norm is `w * x`, not Gemma's
`(1 + w) * x`, so a Gemma formula would fail the witness). No real
checkpoint is read, and nothing is downloaded.

Qwen3 is supported for logits only (E1, E2): a capture run is refused.

Run: python3 -I -m unittest discover -s scripts -p 'test_prec_diag_*.py'
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prec_diag import evidence as ev  # noqa: E402
from prec_diag import families as fam  # noqa: E402
from prec_diag import hf_oracle as ho  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import Qwen3Config, Qwen3ForCausalLM  # noqa: E402

LAYERS, HIDDEN, HEADS, KV, HEAD_DIM, VOCAB = 3, 64, 4, 2, 32, 300
IDS = [(37 * k + 11) % (VOCAB - 10) + 5 for k in range(24)]
BOUNDARY = {"allowlist": {}, "in_closure_prefixes": {"model": "target", "lm_head": "target"}}
THETA = 1e6


def make_checkpoint(path: Path, **over) -> Path:
    torch.manual_seed(1)
    kw = dict(vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=96, num_hidden_layers=LAYERS,
              num_attention_heads=HEADS, num_key_value_heads=KV, head_dim=HEAD_DIM, max_position_embeddings=128,
              rope_theta=THETA, tie_word_embeddings=True, use_sliding_window=False, bos_token_id=None,
              eos_token_id=1, pad_token_id=0)
    kw.update(over)
    model = Qwen3ForCausalLM(Qwen3Config(**kw)).to(torch.bfloat16)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name:
                p.normal_(1.0, 0.3)
    model.save_pretrained(path, safe_serialization=True)
    return path


class Fixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        cls.ckpt = make_checkpoint(cls.tmp / "ckpt")
        cls.tables = cls.write_larql_tables(cls.tmp / "tables")
        cls.runs = {arm: cls.arm(arm) for arm in ("A32", "A32s", "A64")}
        cls.runs["A32r"] = cls.arm("A32r", table_dirs=cls.tables)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    @classmethod
    def arm(cls, arm, name=None, ckpt=None, **kw):
        out = cls.tmp / "runs" / (name or arm)
        kw.setdefault("positions", None)
        rec = ho.run(arm=arm, ckpt=ckpt or cls.ckpt, ids=IDS, out_dir=out, boundary=BOUNDARY,
                     require_threads=False, **kw)
        return out, rec

    @classmethod
    def write_larql_tables(cls, root: Path) -> dict:
        i = np.arange(HEAD_DIM // 2, dtype=np.float64)
        ang = np.arange(len(IDS), dtype=np.float64)[:, None] * THETA ** (-2.0 * i / HEAD_DIM)
        d = root / "full_attention"
        d.mkdir(parents=True)
        np.cos(ang).astype("<f4").tofile(d / "larql.cos.f32")
        np.sin(ang).astype("<f4").tofile(d / "larql.sin.f32")
        return {"full_attention": d}

    def logits(self, arm):
        run_dir, _ = self.runs[arm]
        ext, dt = ("f64", "<f8") if arm == "A64" else ("f32", "<f4")
        return np.fromfile(run_dir / f"logits.{ext}", dtype=dt).reshape(len(IDS), VOCAB)

    def audit_with(self, skip=frozenset()):
        model, _ = ho.load(self.ckpt, "A64", BOUNDARY)
        family = ho.Family("qwen3")
        audit = ho.A64Audit(model, family)
        with ho.SiteCapture(model, None, family), ho.a64_patches(model, audit, skip), \
                ho.module_dtype_hooks(model, audit):
            with torch.no_grad():
                logits = model(torch.tensor([IDS]), use_cache=False).logits[0]
        return audit.finish(logits, LAYERS), ho.audit_parameters(model, self.ckpt)


def rel(a, b):
    return float(np.linalg.norm(np.asarray(a, np.float64) - b) / max(np.linalg.norm(b), 1e-30))


class Family(unittest.TestCase):
    def test_the_installed_modeling_file_is_the_frozen_one(self):
        self.assertEqual(ho.modeling_sha256(ho.Family("qwen3")), fam.FAMILIES["qwen3"]["modeling_sha256"])
        self.assertEqual(ho.modeling_sha256(ho.Family("gemma3_text")),
                         fam.FAMILIES["gemma3_text"]["modeling_sha256"])

    def test_qwen3_is_full_attention_throughout(self):
        self.assertEqual(fam.config_layer_types({"model_type": "qwen3", "num_hidden_layers": 3}),
                         ["full_attention"] * 3)
        with self.assertRaises(fam.UnsupportedFamily):
            fam.config_layer_types({"model_type": "qwen3", "num_hidden_layers": 3, "use_sliding_window": True})

    def test_an_unknown_family_is_refused(self):
        with self.assertRaises(fam.UnsupportedFamily):
            fam.family_of({"model_type": "llama"})

    def test_the_call_counts_are_the_architectures(self):
        self.assertEqual(ho.Family("qwen3").expected_calls(28),
                         {"rmsnorm": 4 * 28 + 1, "rotary": 1, "softmax": 28})


class Arms(Fixture):
    def test_every_arm_completes_and_is_admitted(self):
        for arm, (run_dir, rec) in self.runs.items():
            self.assertEqual(rec["status"], ev.COMPLETE, (arm, rec.get("detail")))
            self.assertTrue(ev.admit(run_dir).admitted, arm)

    def test_the_identity_names_the_family_and_its_modeling_file(self):
        _, rec = self.runs["A64"]
        identity = json.loads((self.runs["A64"][0] / ev.RESULT_FILE).read_text())["identity"]
        self.assertEqual(identity["modeling"], {"family": "qwen3",
                                                "sha256": fam.FAMILIES["qwen3"]["modeling_sha256"]})
        self.assertIn("families.py", identity["scripts"])

    def test_a64_is_close_to_a32_and_not_equal(self):
        d = rel(self.logits("A64"), self.logits("A32").astype(np.float64))
        self.assertGreater(d, 0)
        self.assertLess(d, 1e-4)

    def test_a64_audit_passes_with_qwen3_counts_and_witnesses(self):
        audit = json.loads((self.runs["A64"][0] / "audit.json").read_text())
        self.assertTrue(audit["pass"] and audit["parameters"]["pass"], audit["problems"])
        self.assertEqual(audit["call_totals"], {"rmsnorm": 4 * LAYERS + 1, "rotary": 1, "softmax": LAYERS})
        self.assertEqual(set(audit["witness_relative_error"]), {"rmsnorm", "rotary", "softmax"})
        for kind, r in audit["witness_relative_error"].items():
            self.assertLess(r, ho.F64_WITNESS_REL, kind)

    def test_a32r_injects_the_full_attention_table_once(self):
        t = json.loads((self.runs["A32r"][0] / "tables.json").read_text())
        self.assertEqual(t, {"injected_calls": 1, "bit_identical_to_files": True})
        d = rel(self.logits("A32r"), self.logits("A32"))
        self.assertGreater(d, 0)
        self.assertLess(d, 1e-4)


class A64AuditCatchesEachMissingPatch(Fixture):
    def test_the_full_patch_set_passes(self):
        audit, params = self.audit_with()
        self.assertTrue(audit["pass"] and params["pass"], (audit["problems"], params["problems"]))

    def test_removing_any_one_patch_fails(self):
        for patch in ("rmsnorm", "rotary", "softmax"):
            audit, _ = self.audit_with(frozenset({patch}))
            self.assertFalse(audit["pass"], patch)
            self.assertTrue(any(patch in p for p in audit["problems"]), (patch, audit["problems"]))

    def test_a_gemma_norm_formula_is_far_from_a32(self):
        """The witness recomputes the patch's own formula, so it cannot see a
        wrong formula; A64 against A32 can. Gemma's `(1 + w)` norm on a Qwen3
        checkpoint moves the logits by orders of magnitude more than f64 vs f32."""
        family = ho.Family("qwen3")
        family.norm_offset = 1.0
        model, _ = ho.load(self.ckpt, "A64", BOUNDARY)
        audit = ho.A64Audit(model, family)
        with ho.SiteCapture(model, None, family), ho.a64_patches(model, audit):
            with torch.no_grad():
                wrong = model(torch.tensor([IDS]), use_cache=False).logits[0].numpy()
        self.assertGreater(rel(wrong, self.logits("A32").astype(np.float64)), 1e-2)

    def test_patches_are_restored_after_the_run(self):
        q = ho.Q
        before = (q.Qwen3RMSNorm.forward, q.Qwen3RotaryEmbedding.forward, q.eager_attention_forward,
                  q.apply_rotary_pos_emb)
        self.audit_with()
        after = (q.Qwen3RMSNorm.forward, q.Qwen3RotaryEmbedding.forward, q.eager_attention_forward,
                 q.apply_rotary_pos_emb)
        self.assertEqual(before, after)


class Refusals(Fixture):
    def test_a_capture_run_is_refused(self):
        run_dir, rec = self.arm("A32", name="capture", positions=[0, 3])
        self.assertEqual(rec["status"], ev.REFUSED)
        self.assertIn("captures", rec["detail"])
        self.assertFalse(ev.admit(run_dir).admitted)

    def test_a_different_modeling_file_is_refused(self):
        run_dir, rec = self.arm("A64", name="modeling", expected_modeling_sha256="0" * 64)
        self.assertEqual(rec["status"], ev.REFUSED)
        self.assertIn("modeling_qwen3.py", rec["detail"])

    def test_a_sliding_window_checkpoint_is_refused(self):
        sw = make_checkpoint(self.tmp / "ckpt-sw", use_sliding_window=True, sliding_window=8, max_window_layers=1)
        with self.assertRaises(fam.UnsupportedFamily):
            fam.config_layer_types(json.loads((sw / "config.json").read_text()))
        run_dir, rec = self.arm("A32", name="sliding", ckpt=sw)
        self.assertEqual(rec["status"], ev.REFUSED)
        self.assertIn("use_sliding_window", rec["detail"])

if __name__ == "__main__":
    unittest.main()
