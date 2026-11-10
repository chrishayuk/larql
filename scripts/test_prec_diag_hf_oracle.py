"""Tests for prec_diag/hf_oracle.py on a synthetic Gemma 3 checkpoint.

The checkpoint is a tiny random Gemma 3 text model saved as BF16:
- 4 layers, mixing sliding and full attention;
- `query_pre_attn_scalar` != `head_dim`;
- non-trivial norm weights.

No real checkpoint is read, and nothing is downloaded.

The LARQL cross-check (`LarqlCrossCheck`) needs a larql binary. Name it in
`PREC_DIAG_LARQL_BIN`; without one the test is reported as skipped, and
says why.

Run: python3 -I -m unittest discover -s scripts -p 'test_prec_diag_*.py'
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prec_diag import evidence as ev  # noqa: E402
from prec_diag import hf_oracle as ho  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import Gemma3ForCausalLM, Gemma3TextConfig  # noqa: E402

# HIDDEN is not a perfect square, so sqrt(HIDDEN) is inexact in f32 (as sqrt(640) is).
LAYERS, HIDDEN, HEADS, KV, HEAD_DIM, VOCAB = 4, 80, 2, 1, 32, 300
IDS = [2] + [(37 * k + 11) % (VOCAB - 10) + 5 for k in range(23)]
POSITIONS = [0, 3, 9, 10, 17, 23]
BOUNDARY = {"allowlist": {}, "in_closure_prefixes": {"model": "target", "lm_head": "target"}}


def make_checkpoint(path: Path, **over) -> Path:
    torch.manual_seed(0)
    kw = dict(vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=96, num_hidden_layers=LAYERS,
              num_attention_heads=HEADS, num_key_value_heads=KV, head_dim=HEAD_DIM, query_pre_attn_scalar=24,
              sliding_window=6, max_position_embeddings=128, bos_token_id=2, eos_token_id=1, pad_token_id=0,
              tie_word_embeddings=True, sliding_window_pattern=2,
              layer_types=["sliding_attention", "full_attention", "sliding_attention", "full_attention"])
    kw.update(over)
    cfg = Gemma3TextConfig(**kw)
    model = Gemma3ForCausalLM(cfg).to(torch.bfloat16)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name:
                p.normal_(0, 0.3)
    model.save_pretrained(path, safe_serialization=True)
    return path


class Fixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        cls.ckpt = make_checkpoint(cls.tmp / "ckpt")
        cls.runs = {}
        for arm in ("A32", "A32s", "A64"):
            cls.runs[arm] = cls.arm(arm)
        cls.tables = cls.write_larql_tables(cls.tmp / "tables")
        cls.runs["A32r"] = cls.arm("A32r", table_dirs=cls.tables)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    @classmethod
    def arm(cls, arm, name=None, **kw):
        out = cls.tmp / "runs" / (name or arm)
        kw.setdefault("positions", POSITIONS)
        rec = ho.run(arm=arm, ckpt=cls.ckpt, ids=IDS, out_dir=out, boundary=BOUNDARY,
                     expected_modeling_sha256=ho.modeling_sha256(), require_threads=False, **kw)
        return out, rec

    @classmethod
    def write_larql_tables(cls, root: Path) -> dict:
        """Tables built the way LARQL's shipped construction builds them:
        f64 `theta^(-2i/d)`, f64 angle, f64 sin/cos rounded to f32. (In the
        real run they come from the `rope_angle_tables` example instead.)"""
        cfg = json.loads((cls.ckpt / "config.json").read_text())
        dirs = {}
        for layer_type, params in cfg["rope_parameters"].items():
            theta = params["rope_theta"]
            i = np.arange(HEAD_DIM // 2, dtype=np.float64)
            ang = np.arange(len(IDS), dtype=np.float64)[:, None] * theta ** (-2.0 * i / HEAD_DIM)
            d = root / layer_type
            d.mkdir(parents=True)
            np.cos(ang).astype("<f4").tofile(d / "larql.cos.f32")
            np.sin(ang).astype("<f4").tofile(d / "larql.sin.f32")
            dirs[layer_type] = d
        return dirs

    def plane(self, arm, site, layer=None):
        run_dir, _ = self.runs[arm]
        man = json.loads((run_dir / "sites" / "manifest.json").read_text())
        ext = "f64" if arm == "A64" else "f32"
        name = f"{site}.{ext}" if layer is None else f"{site}.L{layer:03}.{ext}"
        e = next(f for f in man["files"] if f["file"] == name)
        return np.fromfile(run_dir / "sites" / name, dtype=e["dtype"]).reshape(e["rows"], e["width"])

    def logits(self, arm):
        run_dir, _ = self.runs[arm]
        ext, dt = ("f64", "<f8") if arm == "A64" else ("f32", "<f4")
        return np.fromfile(run_dir / f"logits.{ext}", dtype=dt).reshape(len(IDS), VOCAB)

    def weights(self):
        from safetensors.torch import load_file
        return {k: v.to(torch.float64).numpy() for k, v in load_file(str(self.ckpt / "model.safetensors")).items()}


def rel(a, b):
    return float(np.linalg.norm(np.asarray(a, np.float64) - b) / max(np.linalg.norm(b), 1e-30))


def rms(x, w, eps=1e-6):
    x = np.asarray(x, np.float64)
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * (1.0 + w)


class Arms(Fixture):
    def test_every_arm_completes_and_is_admitted(self):
        for arm, (run_dir, rec) in self.runs.items():
            self.assertEqual(rec["status"], ev.COMPLETE, (arm, rec.get("detail")))
            adm = ev.admit(run_dir, {"arm": arm, "tool": "hf_oracle"})
            self.assertTrue(adm.admitted, (arm, adm.reasons))

    def test_capture_does_not_perturb_the_logits(self):
        model, _ = ho.load(self.ckpt, "A32", BOUNDARY)
        with torch.no_grad():
            plain = model(torch.tensor([IDS]), use_cache=False).logits[0].numpy()
        self.assertEqual(plain.tobytes(), self.logits("A32").tobytes())

    def test_a32s_is_sdpa_and_differs_from_eager_only_at_f32_noise(self):
        self.assertLess(rel(self.logits("A32s"), self.logits("A32")), 1e-5)

    def test_a64_logits_are_f64_and_close_to_a32(self):
        run_dir, _ = self.runs["A64"]
        self.assertEqual((run_dir / "logits.f64").stat().st_size, len(IDS) * VOCAB * 8)
        self.assertLess(rel(self.logits("A32"), self.logits("A64")), 1e-5)
        self.assertGreater(rel(self.logits("A32"), self.logits("A64")), 0)

    def test_a64_audit_records_every_part(self):
        run_dir, _ = self.runs["A64"]
        audit = json.loads((run_dir / "audit.json").read_text())
        self.assertTrue(audit["pass"], audit["problems"])
        self.assertTrue(audit["parameters"]["pass"])
        self.assertEqual(audit["parameters"]["parameters_checked"], audit["parameters"]["parameters"])
        self.assertEqual(audit["call_totals"], {"rmsnorm": 6 * LAYERS + 1, "rotary": LAYERS,
                                                "softmax": LAYERS, "embedding": 1})
        self.assertEqual(set(audit["calls"]["softmax"]), {str(L) for L in range(LAYERS)})
        for kind, r in audit["witness_relative_error"].items():
            self.assertLess(r, ho.F64_WITNESS_REL, kind)
        self.assertGreater(audit["module_tensors_checked"], 100)

    def test_a32r_injects_larql_tables_bit_for_bit(self):
        run_dir, _ = self.runs["A32r"]
        t = json.loads((run_dir / "tables.json").read_text())
        self.assertEqual(t, {"injected_calls": LAYERS, "bit_identical_to_files": True})
        cfg = json.loads((self.ckpt / "config.json").read_text())
        for L, layer_type in enumerate(cfg["layer_types"]):
            cos = np.fromfile(self.tables[layer_type] / "larql.cos.f32", dtype="<f4").reshape(len(IDS), -1)
            sin = np.fromfile(self.tables[layer_type] / "larql.sin.f32", dtype="<f4").reshape(len(IDS), -1)
            c = torch.from_numpy(np.concatenate([cos, cos], -1)[POSITIONS])
            s = torch.from_numpy(np.concatenate([sin, sin], -1)[POSITIONS])
            for site, heads in (("query", HEADS), ("key", KV)):
                un = torch.from_numpy(self.plane("A32r", f"{site}_unpositioned", L)).reshape(len(POSITIONS), heads, -1)
                rot = un * c[:, None] + torch.cat((-un[..., HEAD_DIM // 2:], un[..., :HEAD_DIM // 2]), -1) * s[:, None]
                got = self.plane("A32r", f"{site}_positioned", L).reshape(len(POSITIONS), heads, -1)
                self.assertEqual(rot.numpy().tobytes(), got.tobytes(), (L, site))
        self.assertGreater(rel(self.logits("A32r"), self.logits("A32")), 0)
        self.assertLess(rel(self.logits("A32r"), self.logits("A32")), 1e-4)


class SiteMapping(Fixture):
    """Each captured site is the quantity its E0a name says, shown from the
    model's own weights: same mathematical value, not just same shape."""

    def test_the_residual_chain_re_adds_in_f32(self):
        r = self.plane("A32", "embedding")
        for L in range(LAYERS):
            mid = r + self.plane("A32", "attention_delta", L)
            r = mid + self.plane("A32", "ffn_delta", L)
            self.assertEqual(r.tobytes(), self.plane("A32", "residual", L).tobytes(), L)
            r = self.plane("A32", "residual", L)

    def test_embedding_is_the_scaled_token_row(self):
        w = self.weights()["model.embed_tokens.weight"]
        want = w[np.asarray(IDS)[POSITIONS]] * np.sqrt(HIDDEN)
        self.assertLess(rel(self.plane("A64", "embedding"), want), 1e-15)

    def test_norm_sites_are_the_named_norms_of_the_named_inputs(self):
        W = self.weights()
        prev = self.plane("A64", "embedding")
        for L in range(LAYERS):
            p = f"model.layers.{L}."
            att_in = self.plane("A64", "attention_input", L)
            self.assertLess(rel(att_in, rms(prev, W[p + "input_layernorm.weight"])), 1e-13, L)
            delta = self.plane("A64", "attention_delta", L)
            self.assertLess(rel(delta, rms(self.plane("A64", "attention_output", L),
                                           W[p + "post_attention_layernorm.weight"])), 1e-13, L)
            self.assertLess(rel(self.plane("A64", "ffn_input", L),
                                rms(prev + delta, W[p + "pre_feedforward_layernorm.weight"])), 1e-13, L)
            self.assertLess(rel(self.plane("A64", "ffn_delta", L),
                                rms(self.plane("A64", "ffn_output", L),
                                    W[p + "post_feedforward_layernorm.weight"])), 1e-13, L)
            prev = self.plane("A64", "residual", L)
        self.assertLess(rel(self.plane("A64", "head_input"), rms(prev, W["model.norm.weight"])), 1e-13)

    def test_q_and_k_are_after_qk_norm_and_unscaled(self):
        # query_pre_attn_scalar (24) != head_dim (32): a scaled q would be off by sqrt(24/32).
        W = self.weights()
        for L in range(LAYERS):
            p = f"model.layers.{L}.self_attn."
            x = self.plane("A64", "attention_input", L)
            for site, proj, norm, heads in (("query", "q_proj", "q_norm", HEADS), ("key", "k_proj", "k_norm", KV)):
                raw = (x @ W[p + proj + ".weight"].T).reshape(len(POSITIONS), heads, HEAD_DIM)
                want = rms(raw, W[p + norm + ".weight"]).reshape(len(POSITIONS), -1)
                self.assertLess(rel(self.plane("A64", f"{site}_unpositioned", L), want), 1e-13, (L, site))

    def test_positioned_q_and_k_are_the_rotation_of_the_unpositioned(self):
        cfg = json.loads((self.ckpt / "config.json").read_text())
        for L, layer_type in enumerate(cfg["layer_types"]):
            theta = cfg["rope_parameters"][layer_type]["rope_theta"]
            f = theta ** (-np.arange(0, HEAD_DIM, 2) / HEAD_DIM)
            ang = np.asarray(POSITIONS, np.float64)[:, None] * f
            c, s = np.cos(np.concatenate([ang, ang], -1)), np.sin(np.concatenate([ang, ang], -1))
            for site, heads in (("query", HEADS), ("key", KV)):
                un = self.plane("A64", f"{site}_unpositioned", L).reshape(len(POSITIONS), heads, HEAD_DIM)
                rh = np.concatenate([-un[..., HEAD_DIM // 2:], un[..., :HEAD_DIM // 2]], -1)
                want = (un * c[:, None] + rh * s[:, None]).reshape(len(POSITIONS), -1)
                self.assertLess(rel(self.plane("A64", f"{site}_positioned", L), want), 1e-13, (L, site))

    def test_the_head_input_produces_the_logits(self):
        W = self.weights()
        want = self.plane("A64", "head_input") @ W["model.embed_tokens.weight"].T
        self.assertLess(rel(self.logits("A64")[POSITIONS], want), 1e-13)

    def test_the_ffn_output_is_the_mlp_of_its_input(self):
        W = self.weights()
        L = 2
        p = f"model.layers.{L}.mlp."
        x = self.plane("A64", "ffn_input", L)
        g = x @ W[p + "gate_proj.weight"].T
        act = 0.5 * g * (1 + np.tanh(np.sqrt(2 / np.pi) * (g + 0.044715 * g ** 3)))
        want = (act * (x @ W[p + "up_proj.weight"].T)) @ W[p + "down_proj.weight"].T
        self.assertLess(rel(self.plane("A64", "ffn_output", L), want), 1e-13)

    def test_every_site_is_far_from_every_other_same_shape_site(self):
        """Uniqueness: no two sites could be swapped and still match."""
        for L in (0, LAYERS - 1):
            planes = {s: self.plane("A32", s, L) for s in ho.LAYER_SITES}
            planes |= {s: self.plane("A32", s) for s in ho.LAYERLESS_SITES}
            for a in planes:
                for b in planes:
                    if a < b and planes[a].shape == planes[b].shape:
                        self.assertGreater(rel(planes[a], planes[b]), 0.05, (L, a, b))

    def test_the_manifest_carries_the_site_map_and_the_query_scale_note(self):
        run_dir, _ = self.runs["A32"]
        man = json.loads((run_dir / "sites" / "manifest.json").read_text())
        self.assertEqual(set(man["site_map"]), set(ho.LAYER_SITES) | set(ho.LAYERLESS_SITES))
        self.assertEqual(len(man["files"]), 2 + len(ho.LAYER_SITES) * LAYERS)
        self.assertIn("not to q", man["query_scale"])
        for f in man["files"]:
            self.assertEqual(ev.sha256_file(run_dir / "sites" / f["file"]), f["sha256"])


class A64AuditCatchesEachMissingPatch(Fixture):
    def audit_with(self, skip=frozenset(), tamper=None):
        model, _ = ho.load(self.ckpt, "A64", BOUNDARY)
        if tamper:
            tamper(model)
        audit = ho.A64Audit(model)
        with ho.SiteCapture(model, None), ho.a64_patches(model, audit, skip), ho.module_dtype_hooks(model, audit):
            with torch.no_grad():
                logits = model(torch.tensor([IDS]), use_cache=False).logits[0]
        return audit.finish(logits, LAYERS), ho.audit_parameters(model, self.ckpt)

    def test_the_full_patch_set_passes(self):
        audit, params = self.audit_with()
        self.assertTrue(audit["pass"] and params["pass"], (audit["problems"], params["problems"]))

    def test_removing_any_one_patch_fails(self):
        for patch in ("rmsnorm", "rotary", "softmax", "embedding"):
            audit, _ = self.audit_with(frozenset({patch}))
            self.assertFalse(audit["pass"], patch)
            self.assertTrue(any(patch in p for p in audit["problems"]), (patch, audit["problems"]))

    def test_a_patch_that_computes_in_f32_fails_its_witness(self):
        real = torch.rsqrt
        with mock.patch.object(torch, "rsqrt", side_effect=lambda t: real(t.float()).double()):
            audit, _ = self.audit_with()
        self.assertTrue(any("rmsnorm: f64 witness" in p for p in audit["problems"]), audit["problems"])

    def test_a_module_running_in_f32_fails_the_module_audit(self):
        class F32Out(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, x):
                return self.inner(x).float()

        def tamper(model):
            model.model.layers[1].mlp = F32Out(model.model.layers[1].mlp)

        audit, _ = self.audit_with(tamper=tamper)
        self.assertTrue(any("module" in p and "float32" in p for p in audit["problems"]), audit["problems"])

    def test_a_norm_that_bypasses_the_patch_fails_the_call_count(self):
        # Its output is still float64 and the witness was taken on an earlier
        # call, so only the per-layer call count can see it.
        class Bypass(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, x):
                return (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
                        * (1.0 + self.inner.weight.float())).double()

        def tamper(model):
            layer = model.model.layers[2]
            layer.post_feedforward_layernorm = Bypass(layer.post_feedforward_layernorm)

        audit, _ = self.audit_with(tamper=tamper)
        self.assertFalse(audit["pass"])
        self.assertTrue(any("call counts" in p for p in audit["problems"]), audit["problems"])
        self.assertEqual(audit["calls"]["rmsnorm"]["2"], 5)

    def test_a_parameter_not_equal_to_the_checkpoint_fails(self):
        def tamper(model):
            with torch.no_grad():
                model.model.layers[0].mlp.up_proj.weight[0, 0] += 1e-9
        _, params = self.audit_with(tamper=tamper)
        self.assertFalse(params["pass"])
        self.assertIn("model.layers.0.mlp.up_proj.weight", params["problems"][0])

    def test_a_parameter_left_in_f32_fails(self):
        def tamper(model):
            model.model.norm.weight.data = model.model.norm.weight.data.float()
        _, params = self.audit_with(tamper=tamper)
        self.assertIn("model.norm.weight: torch.float32", params["problems"])

    def test_a_failed_audit_records_fault_and_is_not_admitted(self):
        with mock.patch.object(ho, "audit_parameters",
                               return_value={"pass": False, "problems": ["x"], "parameters_checked": 0,
                                             "parameters": 1}):
            run_dir, rec = self.arm("A64", name="A64-bad")
        self.assertEqual(rec["status"], ev.FAULT)
        self.assertFalse(ev.admit(run_dir).admitted)

    def test_patches_are_restored_after_the_run(self):
        before = (ho.G.Gemma3RMSNorm.forward, ho.G.Gemma3RotaryEmbedding.forward, ho.G.eager_attention_forward,
                  ho.G.Gemma3TextScaledWordEmbedding.forward, ho.G.apply_rotary_pos_emb)
        self.audit_with()
        after = (ho.G.Gemma3RMSNorm.forward, ho.G.Gemma3RotaryEmbedding.forward, ho.G.eager_attention_forward,
                 ho.G.Gemma3TextScaledWordEmbedding.forward, ho.G.apply_rotary_pos_emb)
        self.assertEqual(before, after)


class Refusals(Fixture):
    def refused(self, needle, name, arm="A32", **kw):
        run_dir, rec = self.arm(arm, name=name, **kw)
        self.assertEqual(rec["status"], ev.REFUSED, rec.get("detail"))
        self.assertIn(needle, rec["detail"])
        self.assertFalse(ev.admit(run_dir).admitted)

    def test_a_different_modeling_file_is_refused(self):
        run_dir = self.tmp / "runs" / "modeling"
        rec = ho.run(arm="A64", ckpt=self.ckpt, ids=IDS, positions=None, out_dir=run_dir, boundary=BOUNDARY,
                     expected_modeling_sha256="0" * 64, require_threads=False)
        self.assertEqual(rec["status"], ev.REFUSED)
        self.assertIn("modeling_gemma3.py", rec["detail"])

    def test_other_thread_settings_are_refused(self):
        run_dir = self.tmp / "runs" / "threads"
        with mock.patch.object(ho, "thread_state", return_value={"torch_num_threads": 3}):
            rec = ho.run(arm="A32", ckpt=self.ckpt, ids=IDS, positions=None, out_dir=run_dir,
                         boundary=BOUNDARY, expected_modeling_sha256=ho.modeling_sha256())
        self.assertEqual(rec["status"], ev.REFUSED)
        self.assertIn("threads", rec["detail"])

    def test_bad_positions_are_refused(self):
        self.refused("positions", "pos-order", positions=[3, 1])
        self.refused("positions", "pos-range", positions=[0, len(IDS)])

    def test_a32r_without_tables_or_with_wrong_tables_is_refused(self):
        self.refused("tables", "no-tables", arm="A32r")
        bad = self.tmp / "badtables" / "full_attention"
        bad.mkdir(parents=True)
        (bad / "larql.cos.f32").write_bytes(b"\0" * 8)
        (bad / "larql.sin.f32").write_bytes(b"\0" * 8)
        self.refused("bytes", "bad-tables", arm="A32r", table_dirs={**self.tables, "full_attention": bad})

    def test_an_unexpected_tensor_fails_the_loading_gate(self):
        from safetensors.torch import load_file, save_file
        other = self.tmp / "ckpt-extra"
        other.mkdir()
        for f in self.ckpt.iterdir():
            (other / f.name).write_bytes(f.read_bytes())
        tensors = load_file(str(other / "model.safetensors"))
        tensors["vision_tower.extra.weight"] = torch.zeros(2, dtype=torch.bfloat16)
        save_file(tensors, str(other / "model.safetensors"), metadata={"format": "pt"})
        run_dir = self.tmp / "runs" / "gate"
        rec = ho.run(arm="A32", ckpt=other, ids=IDS, positions=None, out_dir=run_dir, boundary=BOUNDARY,
                     expected_modeling_sha256=ho.modeling_sha256(), require_threads=False)
        self.assertEqual(rec["status"], ev.REFUSED)
        self.assertIn("loading gate", rec["detail"])

    def test_a_non_default_rope_is_refused(self):
        other = make_checkpoint(self.tmp / "ckpt-linear",
                                rope_parameters={"full_attention": {"rope_type": "linear", "factor": 8.0,
                                                                    "rope_theta": 1e6},
                                                 "sliding_attention": {"rope_type": "default", "rope_theta": 1e4}})
        rec = ho.run(arm="A64", ckpt=other, ids=IDS, positions=None, out_dir=self.tmp / "runs" / "rope",
                     boundary=BOUNDARY, expected_modeling_sha256=ho.modeling_sha256(), require_threads=False)
        self.assertEqual(rec["status"], ev.REFUSED)
        self.assertIn("rope_type", rec["detail"])

    def test_a_run_directory_is_never_reused(self):
        run_dir, _ = self.runs["A32"]
        with self.assertRaises(ev.EvidenceError):
            ho.run(arm="A32", ckpt=self.ckpt, ids=IDS, positions=None, out_dir=run_dir, boundary=BOUNDARY,
                   expected_modeling_sha256=ho.modeling_sha256(), require_threads=False)

    def test_a32r_whose_tables_were_not_injected_is_a_fault(self):
        import contextlib

        @contextlib.contextmanager
        def not_injected(tables, record):
            record.update({"injected_calls": 0, "bit_identical_to_files": True})
            yield record

        with mock.patch.object(ho, "larql_tables", not_injected):
            run_dir, rec = self.arm("A32r", name="A32r-uninjected", table_dirs=self.tables)
        self.assertEqual(rec["status"], ev.FAULT)
        self.assertIn("A32r table injection", rec["detail"])

    def test_an_error_mid_run_is_a_fault_not_a_stuck_run(self):
        with mock.patch.object(ho, "forward", side_effect=RuntimeError("boom")):
            run_dir, rec = self.arm("A32", name="boom")
        self.assertEqual(rec["status"], ev.FAULT)
        self.assertIn("boom", rec["detail"])


@unittest.skipUnless(os.environ.get("PREC_DIAG_LARQL_BIN"),
                     "set PREC_DIAG_LARQL_BIN to a larql binary to cross-check the site map against LARQL")
class LarqlCrossCheck(Fixture):
    """LARQL's E0a capture of the same synthetic checkpoint matches A32 at
    every site and layer, on both backends: the site map holds against the
    real capture, not only against the model's arithmetic."""

    def test_every_site_matches_larql_on_both_backends(self):
        larql = os.environ["PREC_DIAG_LARQL_BIN"]
        container = self.tmp / "c.vindex3"
        done = subprocess.run([larql, "vindex3", "encode", str(self.ckpt), "--output", str(container),
                               "--capability", "text-generation"],
                              capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr[-2000:])
        worst = {}
        for backend, flags in (("reference", []), ("production", ["--representation-source", "stored"])):
            out = self.tmp / f"larql-{backend}"
            out.mkdir()
            done = subprocess.run(
                [larql, "vindex3", "exec", str(container), "--backend", backend, *flags,
                 "--tokens", ",".join(map(str, IDS)), "--logit-dump", str(out / "logits.f32"),
                 "--capture-sites", str(out / "sites"), "--capture-positions", ",".join(map(str, POSITIONS))],
                capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stderr[-2000:])
            man = json.loads((out / "sites" / "manifest.json").read_text())
            for f in man["files"]:
                larql_plane = np.fromfile(out / "sites" / f["file"], dtype="<f4").reshape(f["rows"], f["width"])
                site, layer = f["site"], f["layer"]
                r = rel(larql_plane, self.plane("A32", site, layer).astype(np.float64))
                worst[(backend, site)] = max(worst.get((backend, site), 0.0), r)
                self.assertLess(r, 1e-5, (backend, site, layer))
                # ...and far from every OTHER transformers site of the same shape.
                for other in ho.LAYER_SITES + ho.LAYERLESS_SITES:
                    o_layer = None if other in ho.LAYERLESS_SITES else (layer if layer is not None else 0)
                    if other != site:
                        hf = self.plane("A32", other, o_layer)
                        if hf.shape == larql_plane.shape:
                            self.assertGreater(rel(larql_plane, hf.astype(np.float64)), 0.05,
                                               (backend, site, other, layer))
            lg = np.fromfile(out / "logits.f32", dtype="<f4").reshape(len(IDS), VOCAB)
            self.assertLess(rel(lg, self.logits("A32")), 1e-4, backend)
        print("\nLARQL vs A32 worst relative L2 per (backend, site):")
        for k, v in sorted(worst.items()):
            print(f"  {k[0]:<10} {k[1]:<20} {v:.2e}")


if __name__ == "__main__":
    unittest.main()
