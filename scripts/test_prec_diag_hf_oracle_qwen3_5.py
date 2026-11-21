"""Tests for the oracle's `qwen3_5` family (`prec_diag/hf_qwen3_5.py`) on a
synthetic Qwen3.5 checkpoint.

The checkpoint is a tiny random Qwen3.5 hybrid written in the real
checkpoint's layout: a composite `config.json` (`text_config`,
`vision_config`), the language model under `model.language_model.`, a
vision tower under `model.visual.` and an MTP head under `mtp.`, the head
tied, BF16 tensors except `linear_attn.A_log` and `linear_attn.norm.weight`,
which are F32 (with values BF16 cannot hold) as in Qwen3.5-0.8B. The tensors
are written directly: transformers 5.5.0's `save_pretrained` mangles the
composite key names. It has DeltaNet (`linear_attention`) and full-attention
layers, more key heads per value head than one (`repeat_interleave` runs),
partial rotary and interleaved M-RoPE, and the ids are longer than two
chunks of the chunked delta rule (64), so chunk boundaries and a padded last
chunk are exercised. Nothing is downloaded and no real checkpoint is read.

No LARQL output exists here: nothing is compared between implementations.

Run: python3 -I -m unittest discover -s scripts -p 'test_prec_diag_*.py'
"""

import json
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prec_diag import evidence as ev  # noqa: E402
from prec_diag import families as fam  # noqa: E402
from prec_diag import hf_oracle as ho  # noqa: E402
from prec_diag import hf_qwen3_5 as q35  # noqa: E402
from prec_diag import op_audit as oa  # noqa: E402
from prec_diag import op_probes  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from transformers import Qwen3_5Config, Qwen3_5ForCausalLM  # noqa: E402

TYPES = ["linear_attention", "linear_attention", "full_attention", "linear_attention", "full_attention"]
LAYERS, HIDDEN, HEADS, KV, HEAD_DIM, VOCAB = len(TYPES), 64, 4, 2, 32, 300
LINEAR = TYPES.count("linear_attention")
FULL = TYPES.count("full_attention")
#: 2 full chunks of 64 and a padded third.
SEQ = 2 * q35.CHUNK + 8
IDS = [(37 * k + 11) % (VOCAB - 10) + 5 for k in range(SEQ)]
THETA = 1e7
ROPE = {"rope_type": "default", "rope_theta": THETA, "partial_rotary_factor": 0.25, "mrope_section": [2, 1, 1],
        "mrope_interleaved": True}
ROTARY = int(HEAD_DIM * ROPE["partial_rotary_factor"])
TEXT = dict(vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=96, num_hidden_layers=LAYERS,
            num_attention_heads=HEADS, num_key_value_heads=KV, head_dim=HEAD_DIM, linear_conv_kernel_dim=4,
            linear_key_head_dim=16, linear_value_head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
            layer_types=TYPES, max_position_embeddings=512, tie_word_embeddings=True, rope_parameters=ROPE,
            eos_token_id=1, pad_token_id=0)
VISION = dict(depth=1, hidden_size=32, intermediate_size=32, num_heads=2, out_hidden_size=HIDDEN, patch_size=4,
              num_position_embeddings=16)
#: LARQL's capability boundary for this layout, as `larql vindex3 plan` declares it for Qwen3.5-0.8B: the
#: language model in closure, the vision tower out of it, the MTP head bound to no component.
BOUNDARY = {"allowlist": {"model.visual.blocks": "vision.perception_tower",
                          "model.visual.patch_embed.proj": "vision.perception_tower"},
            "in_closure_prefixes": {"model.language_model.embed_tokens": "target.embedding",
                                    "model.language_model.layers": "target.decoder_stack",
                                    "model.language_model.norm": "target.final_norm"}}
#: Tensors standing in for the vision tower and the MTP head (never loaded).
EXTRA = {"model.visual.blocks.0.attn.qkv.weight": (96, 32), "model.visual.patch_embed.proj.weight": (32, 3, 2, 4, 4),
         "mtp.fc.weight": (HIDDEN, 2 * HIDDEN), "mtp.layers.0.mlp.up_proj.weight": (96, HIDDEN),
         "mtp.norm.weight": (HIDDEN,)}
F32_SOURCES = ("linear_attn.A_log", "linear_attn.norm.weight")


def text_tensors() -> dict:
    """The language model's tensors, as the composite checkpoint names them."""
    torch.manual_seed(1)
    model = Qwen3_5ForCausalLM(Qwen3_5Config(text_config=TEXT, tie_word_embeddings=True).text_config)
    out = {}
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name == "lm_head.weight":
                continue  # tied
            key = "model.language_model." + name[len("model."):]
            if name.endswith(F32_SOURCES):
                # F32 at source, with values bfloat16 cannot hold.
                value = (torch.rand(p.shape, dtype=torch.float64) + 0.5).log() if name.endswith("A_log") \
                    else torch.normal(1.0, 0.3, p.shape, dtype=torch.float64)
                out[key] = value.to(torch.float32)
                continue
            if "norm" in name:
                p.normal_(0.0, 0.3)  # (1 + w): a non-trivial offset
            elif name.endswith("dt_bias"):
                p.normal_(0.0, 1.0)
            out[key] = p.detach().to(torch.bfloat16).clone()
    return out


def make_checkpoint(path: Path, extra: dict | None = None, drop: tuple = (), text_over: dict | None = None) -> Path:
    path.mkdir(parents=True)
    tensors = text_tensors()
    gen = torch.Generator().manual_seed(2)
    for key, shape in (EXTRA if extra is None else extra).items():
        tensors[key] = torch.randn(shape, generator=gen).to(torch.bfloat16)
    for key in drop:
        tensors.pop(key)
    save_file(tensors, str(path / "model.safetensors"), metadata={"format": "pt"})
    config = Qwen3_5Config(text_config={**TEXT, **(text_over or {})}, vision_config=VISION, tie_word_embeddings=True)
    config.architectures = ["Qwen3_5ForConditionalGeneration"]
    config.save_pretrained(path)
    return path


def rel(a, b):
    return float(np.linalg.norm(np.asarray(a, np.float64) - b) / max(np.linalg.norm(b), 1e-30))


def linear_layers(model) -> int:
    return sum(isinstance(m, torch.nn.Linear) for m in model.modules())


class Fixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        cls.ckpt = make_checkpoint(cls.tmp / "ckpt")
        cls.tables = cls.write_larql_tables(cls.tmp / "tables")
        cls.runs = {arm: cls.arm(arm) for arm in ("A32", "A32s", "A32rec")}
        cls.runs["A64"] = cls.arm("A64", op_audit=oa.SCHEMA)
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
        """A stand-in for LARQL's full-attention table at the rotary width."""
        i = np.arange(ROTARY // 2, dtype=np.float64)
        ang = np.arange(len(IDS), dtype=np.float64)[:, None] * THETA ** (-2.0 * i / ROTARY)
        d = root / "full_attention"
        d.mkdir(parents=True)
        np.cos(ang).astype("<f4").tofile(d / "larql.cos.f32")
        np.sin(ang).astype("<f4").tofile(d / "larql.sin.f32")
        return {"full_attention": d}

    def logits(self, arm):
        run_dir, _ = self.runs[arm]
        ext, dt = ("f64", "<f8") if arm == "A64" else ("f32", "<f4")
        return np.fromfile(run_dir / f"logits.{ext}", dtype=dt).reshape(len(IDS), VOCAB)

    def a64(self, skip=frozenset(), ids=IDS):
        """The A64 forward as `hf_oracle.forward` composes it, every part
        collected, with `skip` patches removed."""
        model, _ = ho.load(self.ckpt, "A64", BOUNDARY)
        family = ho.Family("qwen3_5")
        audit, ops = ho.A64Audit(model, family), oa.OpDtypeAudit(linear_layers=linear_layers(model))
        with ho.SiteCapture(model, None, family), ho.a64_patches(model, audit, skip), \
                ho.module_dtype_hooks(model, audit), ops:
            with torch.no_grad():
                logits = model(torch.tensor([ids]), use_cache=False).logits[0]
        return audit.finish(logits, LAYERS), ops.finish(), ho.audit_parameters(model, self.ckpt)


#: The real Qwen3.5-0.8B layout: three DeltaNet layers, then one full-attention layer, six times.
REAL_TYPES = (["linear_attention"] * 3 + ["full_attention"]) * 6
#: transformers' shipped f32 rule misses the f64 witness by at least this (it is ~1e-7).
F32_WITNESS_FLOOR = 1e-9


def deltanet_inputs(seed: int, T: int = SEQ, b: int = 1, H: int = 4, dk: int = 16, dv: int = 16, dtype=torch.float64):
    """Inputs as a DeltaNet layer hands them to the rule: raw q, k (the rule
    L2-normalises them), v, a negative decay g and beta in (0, 1)."""
    gen = torch.Generator().manual_seed(seed)
    q, k = torch.randn(b, T, H, dk, generator=gen), torch.randn(b, T, H, dk, generator=gen)
    v = torch.randn(b, T, H, dv, generator=gen)
    g = -torch.rand(b, T, H, generator=gen) * torch.linspace(0.01, 2.0, H)  # slow to fast decay heads
    beta = torch.rand(b, T, H, generator=gen)
    return [t.to(dtype) for t in (q, k, v, g, beta)]


class FamilyFacts(unittest.TestCase):
    def test_the_installed_modeling_file_is_the_frozen_one(self):
        self.assertEqual(ho.modeling_sha256(ho.Family("qwen3_5")), fam.FAMILIES["qwen3_5"]["modeling_sha256"])

    def test_the_layer_types_come_from_text_config_and_only_two_are_admitted(self):
        config = {"model_type": "qwen3_5", "text_config": {"layer_types": REAL_TYPES}}
        self.assertEqual(fam.config_layer_types(config), REAL_TYPES)
        for bad in ({"model_type": "qwen3_5", "text_config": {"layer_types": ["linear_attention", "mamba"]}},
                    {"model_type": "qwen3_5", "text_config": {}}, {"model_type": "qwen3_5"}):
            with self.subTest(bad=bad), self.assertRaises(fam.UnsupportedFamily):
                fam.config_layer_types(bad)

    def test_the_call_counts_are_derived_from_the_config(self):
        self.assertEqual(ho.Family("qwen3_5").expected_calls(24, REAL_TYPES),
                         {"rmsnorm": 61, "gated_rmsnorm": 18, "rotary": 1, "softmax": 6, "gdn": 18, "chunk_rule": 18})
        self.assertEqual(ho.Family("qwen3_5").expected_calls(LAYERS, TYPES),
                         {"rmsnorm": 2 * LAYERS + 2 * FULL + 1, "gated_rmsnorm": LINEAR, "rotary": 1, "softmax": FULL,
                          "gdn": LINEAR, "chunk_rule": LINEAR})
        with self.assertRaises(ValueError):
            ho.Family("qwen3_5").expected_calls(24)

    def test_rope_tables_are_scoped_to_the_layers_that_apply_rotary(self):
        self.assertEqual(fam.rope_layer_types({"model_type": "qwen3_5", "text_config": {"layer_types": REAL_TYPES}}),
                         ["full_attention"])
        self.assertEqual(fam.rope_layer_types({"model_type": "gemma3_text", "num_hidden_layers": 6,
                                               "sliding_window_pattern": 6}),
                         ["full_attention", "sliding_attention"])
        self.assertEqual(fam.rope_layer_types({"model_type": "qwen3", "num_hidden_layers": 2}), ["full_attention"])

    def test_the_family_module_is_bound_by_its_own_label(self):
        self.assertEqual(fam.oracle_module_label("qwen3_5"), "script.hf_qwen3_5")
        self.assertIsNone(fam.oracle_module_label("qwen3"))
        self.assertIn("hf_qwen3_5.py", ho.script_digests(ho.Family("qwen3_5")))
        self.assertNotIn("hf_qwen3_5.py", ho.script_digests(ho.Family("qwen3")))
        self.assertEqual(set(ho.script_digests()), set(ho.script_digests(ho.Family("gemma3_text"))))


class Arms(Fixture):
    def test_every_arm_completes_and_is_admitted(self):
        for arm, (run_dir, rec) in self.runs.items():
            self.assertEqual(rec["status"], ev.COMPLETE, (arm, rec.get("detail")))
            self.assertTrue(ev.admit(run_dir).admitted, arm)

    def test_the_identity_names_the_family_its_modeling_file_and_its_module(self):
        identity = json.loads((self.runs["A64"][0] / ev.RESULT_FILE).read_text())["identity"]
        self.assertEqual(identity["modeling"], {"family": "qwen3_5",
                                                "sha256": fam.FAMILIES["qwen3_5"]["modeling_sha256"]})
        self.assertEqual(identity["scripts"]["hf_qwen3_5.py"], ev.sha256_file(Path(q35.__file__)))
        self.assertEqual(identity["op_audit"], oa.SCHEMA)

    def test_a64_passes_all_five_audit_parts(self):
        audit = json.loads((self.runs["A64"][0] / "audit.json").read_text())
        self.assertTrue(audit["pass"] and audit["parameters"]["pass"], audit["problems"])
        self.assertEqual(audit["call_totals"], ho.Family("qwen3_5").expected_calls(LAYERS, TYPES))
        self.assertEqual(set(audit["witness_relative_error"]), set(ho.Family("qwen3_5").patch_kinds))
        for kind, r in audit["witness_relative_error"].items():
            self.assertLess(r, ho.F64_WITNESS_REL, kind)
        ops = audit["operations"]
        self.assertTrue(ops["pass"], ops["problems"])
        self.assertEqual((ops["non_f64_ops"], ops["violation_sites"], ops["unclassified"]), (0, 0, {}))
        probed = {p["op"] for p in ops["probes"]}
        self.assertTrue({"aten.convolution.default", "aten.softplus.default", "aten.cumsum.default"} <= probed)
        self.assertTrue(all(p["pass"] for p in ops["probes"]))
        # Every DeltaNet layer's chunked rule was checked against the step-by-step witness, at every chunk end.
        detail = audit["witness_detail"]["chunk_rule"]
        self.assertEqual(sorted(map(int, detail)), [i for i, t in enumerate(TYPES) if t == "linear_attention"])
        for d in detail.values():
            self.assertEqual(d["chunk_ends"], [63, 127, SEQ - 1])

    def test_the_parameter_audit_reads_bf16_and_f32_sources_and_the_tied_head(self):
        params = json.loads((self.runs["A64"][0] / "audit.json").read_text())["parameters"]
        self.assertEqual(params["checkpoint_source_dtypes"]["torch.float32"], 2 * LINEAR)
        self.assertEqual(params["f32_source_parameters"], sorted(F32_SOURCES))
        self.assertEqual(params["parameters_checked"], params["parameters"])
        self.assertIn("lm_head.weight", dict(ho.load(self.ckpt, "A64", BOUNDARY)[0].named_parameters(
            remove_duplicate=False)))

    def test_a64_is_close_to_a32_and_not_equal(self):
        d = rel(self.logits("A64"), self.logits("A32").astype(np.float64))
        self.assertGreater(d, 0)
        self.assertLess(d, 1e-4)

    def test_a32rec_is_the_recurrent_rule_close_to_a32_and_not_equal(self):
        rec = json.loads((self.runs["A32rec"][0] / "recurrent_prefill.json").read_text())
        self.assertEqual((rec["calls"], rec["expected_calls"], rec["diagnostic_only"]), (LINEAR, LINEAR, True))
        d = rel(self.logits("A32rec"), self.logits("A32").astype(np.float64))
        self.assertGreater(d, 0)
        self.assertLess(d, 1e-4)

    def test_a32r_injects_the_full_attention_table_once_at_the_rotary_width(self):
        t = json.loads((self.runs["A32r"][0] / "tables.json").read_text())
        self.assertEqual(t, {"injected_calls": 1, "bit_identical_to_files": True})
        d = rel(self.logits("A32r"), self.logits("A32"))
        self.assertGreater(d, 0)
        self.assertLess(d, 1e-4)


class Loading(Fixture):
    def test_every_checkpoint_tensor_is_loaded_or_excluded(self):
        facts = json.loads((self.runs["A64"][0] / "loading.json").read_text())
        text = len(text_tensors())
        self.assertEqual(facts["gate"], "pass", facts["gate_problems"])
        self.assertEqual(facts["loaded"]["tensors"], text)
        self.assertEqual(facts["checkpoint_tensors"], text + len(EXTRA))
        self.assertEqual(facts["excluded"]["model.visual"]["larql_binding"], {"larql_allowlist": 2})
        self.assertEqual(facts["excluded"]["mtp"]["larql_binding"], {"larql_no_component": 3})
        self.assertEqual(facts["excluded_tensors"], sorted(EXTRA))
        self.assertTrue(facts["tie_word_embeddings"] and facts["tied_head_shares_embedding"])
        # transformers' own report hides the exclusions entirely, which is why it is not relied on.
        self.assertEqual(facts["unexpected"], [])

    def test_every_parameter_and_buffer_has_the_arm_dtype_as_loaded(self):
        for arm, dtype in (("A64", torch.float64), ("A32", torch.float32), ("A32s", torch.float32)):
            with self.subTest(arm):
                model, facts = ho.load(self.ckpt, arm, BOUNDARY)
                self.assertEqual({p.dtype for p in model.parameters()}, {dtype})
                self.assertEqual({b.dtype for b in model.buffers() if b.is_floating_point()}, {dtype})
                self.assertEqual(set(facts["observed_dtypes"]["as_loaded"]["parameters"]), {str(dtype)})

    def test_a64_rebuilds_the_rotary_buffers_from_the_f64_formula(self):
        model, facts = ho.load(self.ckpt, "A64", BOUNDARY)
        exact = 1.0 / (THETA ** (torch.arange(0, ROTARY, 2, dtype=torch.float64) / ROTARY))
        self.assertTrue(torch.equal(model.model.rotary_emb.inv_freq, exact))
        self.assertEqual(facts["rotary_buffers_rebuilt_f64"]["dtype_before"]["inv_freq"], "torch.float32")
        a32, _ = ho.load(self.ckpt, "A32", BOUNDARY)
        self.assertFalse(torch.equal(a32.model.rotary_emb.inv_freq.double(), exact))  # f32-rounded at construction

    def test_a_parameter_loaded_in_the_wrong_dtype_is_refused(self):
        real = q35.AutoModelForCausalLM.from_pretrained

        def bf16_norm(*a, **kw):
            model, info = real(*a, **kw)
            model.model.norm.weight.data = model.model.norm.weight.data.to(torch.bfloat16)
            return model, info

        with unittest.mock.patch.object(q35.AutoModelForCausalLM, "from_pretrained", bf16_norm):
            with self.assertRaises(ho.Refused) as e:
                ho.load(self.ckpt, "A64", BOUNDARY)
        self.assertIn("model.norm.weight: torch.bfloat16", str(e.exception))

    def refusal(self, name, needle, boundary=BOUNDARY, **ckpt_kw):
        ckpt = make_checkpoint(self.tmp / name, **ckpt_kw) if ckpt_kw else self.ckpt
        with self.assertRaises(ho.Refused) as e:
            ho.load(ckpt, "A32", boundary)
        self.assertIn(needle, str(e.exception))
        run_dir = self.tmp / "runs" / f"refused-{name}"
        rec = ho.run(arm="A32", ckpt=ckpt, ids=IDS, positions=None, out_dir=run_dir, boundary=boundary,
                     require_threads=False)
        self.assertEqual(rec["status"], ev.REFUSED)
        self.assertFalse(ev.admit(run_dir).admitted)

    def test_a_tensor_neither_loaded_nor_excluded_is_refused(self):
        self.refusal("unaccounted", "neither loaded nor excluded", extra={**EXTRA, "audio.encoder.weight": (4, 4)})

    def test_a_text_tensor_the_model_does_not_load_is_refused(self):
        self.refusal("skipped", "not loaded", extra={**EXTRA, "model.language_model.layers.0.extra.weight": (4, 4)})

    def test_a_missing_text_tensor_is_refused(self):
        self.refusal("missing", "no checkpoint source", drop=("model.language_model.norm.weight",))

    def test_an_excluded_tensor_inside_larqls_closure_is_refused(self):
        closure = {**BOUNDARY, "in_closure_prefixes": {**BOUNDARY["in_closure_prefixes"], "mtp": "target.mtp"}}
        self.refusal("closure", "inside LARQL's text-generation closure", boundary=closure)

    def test_a_loaded_tensor_outside_larqls_closure_is_refused(self):
        closure = {**BOUNDARY, "in_closure_prefixes": {k: v for k, v in BOUNDARY["in_closure_prefixes"].items()
                                                       if not k.endswith("norm")}}
        self.refusal("outside", "outside LARQL's closure", boundary=closure)


class Witness(unittest.TestCase):
    """The step-by-step numpy witness of the chunked rule: agreement with
    the patched f64 rule, and power against the shipped f32 rule."""

    def rule64(self, *inputs, chunk_size=q35.CHUNK):
        seen = {}
        rule = q35.chunk_rule(lambda i, o, s, c: seen.update(cmp=q35.compare_with_witness(i, o, s, c)))
        rule(*inputs[:3], g=inputs[3], beta=inputs[4], chunk_size=chunk_size, use_qk_l2norm_in_kernel=True)
        return seen["cmp"]

    def test_the_patched_f64_chunked_rule_agrees_with_the_step_by_step_recurrence(self):
        for seed, T in ((0, SEQ), (1, 3 * q35.CHUNK), (2, 5), (3, 1040 // 8)):
            with self.subTest(seed=seed, T=T):
                cmp = self.rule64(*deltanet_inputs(seed, T))
                # Agreement two orders inside the bound (the bound's margin): rounding only.
                self.assertLess(cmp["output"], q35.RECURRENCE_WITNESS_REL / 100)
                self.assertLess(cmp["state"], q35.RECURRENCE_WITNESS_REL / 100)
                self.assertEqual(cmp["chunk_ends"], q35.chunk_ends(T))

    def test_the_witness_checks_the_function_not_the_chunking(self):
        cmp = self.rule64(*deltanet_inputs(4), chunk_size=16)
        self.assertLess(max(cmp["output"], cmp["state"]), q35.RECURRENCE_WITNESS_REL / 100)

    def test_the_witness_rejects_transformers_shipped_f32_rules(self):
        q, k, v, g, beta = deltanet_inputs(5)
        ends = q35.chunk_ends(SEQ)
        w_out, w_states = q35.step_witness(q, k, v, g, beta, True, ends)
        for name, rule in (("chunk", q35.Q35.torch_chunk_gated_delta_rule),
                           ("recurrent", q35.Q35.torch_recurrent_gated_delta_rule)):
            with self.subTest(name):
                out, state = rule(q, k, v, g=g, beta=beta, initial_state=None, output_final_state=True,
                                  use_qk_l2norm_in_kernel=True)
                self.assertEqual(out.dtype, torch.float64)  # cast back: dtype alone would pass it
                self.assertGreater(q35.position_rel(out.numpy(), w_out), F32_WITNESS_FLOOR)
                self.assertGreater(q35.position_rel(state.double().numpy(), w_states[-1], axis=1), F32_WITNESS_FLOOR)

    def test_the_witness_rejects_a_wrong_recurrence(self):
        q, k, v, g, beta = deltanet_inputs(6)
        cmp = self.rule64(q, k, v, g, beta)
        self.assertLess(cmp["output"], q35.RECURRENCE_WITNESS_REL)
        # The same rule with the state not carried across a chunk boundary.
        out_a, _ = q35.chunk_rule()(q[:, :64], k[:, :64], v[:, :64], g=g[:, :64], beta=beta[:, :64],
                                    use_qk_l2norm_in_kernel=True)
        out_b, _ = q35.chunk_rule()(q[:, 64:], k[:, 64:], v[:, 64:], g=g[:, 64:], beta=beta[:, 64:],
                                    use_qk_l2norm_in_kernel=True)
        w_out, _ = q35.step_witness(q, k, v, g, beta, True, [SEQ - 1])
        self.assertGreater(q35.position_rel(torch.cat([out_a, out_b], 1).numpy(), w_out), 1e-3)

    def test_on_f32_inputs_the_patched_rule_is_transformers_rule_bit_for_bit(self):
        inputs = deltanet_inputs(7, dtype=torch.float32)
        mine = q35.chunk_rule()(*inputs[:3], g=inputs[3], beta=inputs[4], output_final_state=True,
                                use_qk_l2norm_in_kernel=True)
        theirs = q35.Q35.torch_chunk_gated_delta_rule(*inputs[:3], g=inputs[3], beta=inputs[4],
                                                      output_final_state=True, use_qk_l2norm_in_kernel=True)
        for a, b in zip(mine, theirs):
            self.assertEqual(a.dtype, b.dtype)
            self.assertTrue(torch.equal(a, b))

    def test_the_mrope_axis_map_is_the_modeling_files_interleave(self):
        section, half = [11, 11, 10], 32
        freqs = torch.arange(3)[:, None, None, None].expand(3, 1, 1, half).double().clone()
        rot = q35.Q35.Qwen3_5TextRotaryEmbedding.apply_interleaved_mrope(None, freqs, section)
        np.testing.assert_array_equal(rot[0, 0].numpy(), q35.mrope_axis(half, section))

    def test_a_vision_position_layout_is_refused(self):
        p = torch.arange(6).view(1, 1, 6).expand(3, 1, 6).clone()
        q35.text_positions(p)
        p[1, 0, 2] = 0
        with self.assertRaises(q35.Unsupported):
            q35.text_positions(p)


class DeltaNetForward(Fixture):
    def test_on_an_f32_model_the_patched_forward_is_transformers_forward_bit_for_bit(self):
        model, _ = ho.load(self.ckpt, "A32", BOUNDARY)
        layer = q35.deltanets(model)[0]
        x = torch.randn(1, SEQ, HIDDEN, generator=torch.Generator().manual_seed(8))
        with torch.no_grad():
            theirs = q35.Q35.Qwen3_5GatedDeltaNet.forward(layer, x)
            mine = q35.gdn_forward()(layer, x)
        self.assertTrue(torch.equal(mine, theirs))

    def test_the_patched_forward_refuses_a_cache_or_a_mask(self):
        model, _ = ho.load(self.ckpt, "A64", BOUNDARY)
        layer = q35.deltanets(model)[0]
        x = torch.zeros(1, 2, HIDDEN, dtype=torch.float64)
        with self.assertRaises(q35.Unsupported):
            q35.gdn_forward()(layer, x, attention_mask=torch.ones(1, 2))


class A64AuditCatchesEachMissingPatch(Fixture):
    #: The modeling-file function each patch replaces, as part 5 attributes it.
    SITES = {"rmsnorm": "Qwen3_5RMSNorm.", "gated_rmsnorm": "Qwen3_5RMSNormGated.forward",
             "rotary": "Qwen3_5TextRotaryEmbedding.", "softmax": "eager_attention_forward",
             "gdn": "Qwen3_5GatedDeltaNet.forward", "chunk_rule": "torch_chunk_gated_delta_rule"}

    def test_the_full_patch_set_passes_every_part(self):
        parts, ops, params = self.a64()
        self.assertTrue(parts["pass"] and params["pass"], (parts["problems"], params["problems"]))
        self.assertTrue(ops["pass"], ops["problems"])

    def test_removing_any_one_patch_fails_parts_3_and_5_attributed_to_its_site(self):
        self.assertEqual(set(self.SITES), set(ho.Family("qwen3_5").patch_kinds))
        for patch, site in self.SITES.items():
            with self.subTest(patch):
                parts, ops, _ = self.a64(frozenset({patch}))
                self.assertFalse(parts["pass"])
                self.assertTrue(any(p.startswith(f"{patch}:") for p in parts["problems"]), parts["problems"])
                self.assertFalse(ops["pass"])
                functions = {v["site"].split(" ")[1] for v in ops["violations"]}
                self.assertTrue(functions and all(f.startswith(site) for f in functions), functions)
                self.assertEqual({v["site"].split(":")[0] for v in ops["violations"]}, {"modeling_qwen3_5.py"})
                # The module-boundary audit (part 2) sees none of it: the casts are undone inside the call.
                self.assertEqual([p for p in parts["problems"] if p.startswith("module ")], [])

    def test_a_qwen3_norm_formula_is_far_from_a32(self):
        """The witness recomputes the patch's own formula, so it cannot see a
        wrong formula; A64 against A32 can. Qwen3's `w * x` norm on a Qwen3.5
        checkpoint moves the logits by far more than f64 against f32."""
        family = ho.Family("qwen3_5")
        family.norm_offset = 0.0
        model, _ = ho.load(self.ckpt, "A64", BOUNDARY)
        audit = ho.A64Audit(model, family)
        with ho.SiteCapture(model, None, family), ho.a64_patches(model, audit):
            with torch.no_grad():
                wrong = model(torch.tensor([IDS]), use_cache=False).logits[0].numpy()
        self.assertGreater(rel(wrong, self.logits("A32").astype(np.float64)), 1e-2)

    def test_patches_are_restored_after_the_run(self):
        m = q35.Q35

        def state(model):
            return ((m.Qwen3_5RMSNorm.forward, m.Qwen3_5RMSNormGated.forward, m.Qwen3_5TextRotaryEmbedding.forward,
                     m.Qwen3_5GatedDeltaNet.forward, m.eager_attention_forward, m.apply_rotary_pos_emb),
                    [(d.chunk_gated_delta_rule, d.recurrent_gated_delta_rule) for d in q35.deltanets(model)])

        model, _ = ho.load(self.ckpt, "A64", BOUNDARY)
        before = state(model)
        family = ho.Family("qwen3_5")
        audit = ho.A64Audit(model, family)
        with ho.SiteCapture(model, None, family), ho.a64_patches(model, audit), torch.no_grad():
            model(torch.tensor([IDS[:4]]), use_cache=False)
        self.assertEqual(state(model), before)
        self.assertTrue(all(r == (m.torch_chunk_gated_delta_rule, m.torch_recurrent_gated_delta_rule)
                            for r in before[1]))
        with ho.recurrent_prefill(model, {}):
            pass
        self.assertEqual(state(model), before)


class Refusals(Fixture):
    def test_a_capture_run_is_refused(self):
        run_dir, rec = self.arm("A32", name="capture", positions=[0, 3])
        self.assertEqual(rec["status"], ev.REFUSED)
        self.assertIn("captures", rec["detail"])

    def test_a_different_modeling_file_is_refused(self):
        _, rec = self.arm("A64", name="modeling", expected_modeling_sha256="0" * 64)
        self.assertEqual(rec["status"], ev.REFUSED)
        self.assertIn("modeling_qwen3_5.py", rec["detail"])

    def test_a32r_with_a_deltanet_table_is_refused(self):
        tables = {**self.tables, "linear_attention": self.tables["full_attention"]}
        run_dir, rec = self.arm("A32r", name="a32r-linear", table_dirs=tables)
        self.assertEqual(rec["status"], ev.REFUSED)
        self.assertIn("DeltaNet layers take no positions", rec["detail"])
        self.assertFalse(ev.admit(run_dir).admitted)

    def test_a_rope_layout_the_patch_does_not_implement_is_refused(self):
        for name, over in (("rope-type", {"rope_type": "linear", "factor": 2.0}),
                           ("mrope-flat", {"mrope_interleaved": False})):
            with self.subTest(name):
                ckpt = make_checkpoint(self.tmp / name, text_over={"rope_parameters": {**ROPE, **over}})
                _, rec = self.arm("A64", name=name, ckpt=ckpt)
                self.assertEqual(rec["status"], ev.REFUSED, rec.get("detail"))

    def test_the_fast_path_is_refused(self):
        with unittest.mock.patch.object(q35.Q35, "is_fast_path_available", True):
            _, rec = self.arm("A32", name="fast-path")
        self.assertEqual(rec["status"], ev.REFUSED)
        self.assertIn("fast path", rec["detail"])


class A32recIsNeverAGateBaseline(Fixture):
    def test_a32rec_under_a_freeze_is_refused(self):
        bind = {"experiment": "E1", "run_id": "E1.A32rec.I_rep", "freeze_sha256": "0" * 64}
        run_dir, rec = self.arm("A32rec", name="a32rec-bound", bind=bind)
        self.assertEqual(rec["status"], ev.REFUSED)
        self.assertIn("diagnostic", rec["detail"])
        self.assertFalse(ev.admit(run_dir).admitted)

    def test_a32rec_cannot_be_admitted_as_a32(self):
        run_dir, _ = self.runs["A32rec"]
        self.assertTrue(ev.admit(run_dir).admitted)
        self.assertFalse(ev.admit(run_dir, {"arm": "A32"}).admitted)

    def test_a32rec_on_a_family_without_a_recurrence_is_refused(self):
        import test_prec_diag_hf_oracle_qwen3 as qwen3
        ckpt = qwen3.make_checkpoint(self.tmp / "qwen3")
        _, rec = self.arm("A32rec", name="a32rec-qwen3", ckpt=ckpt)
        self.assertEqual(rec["status"], ev.REFUSED)
        self.assertIn("no recurrence", rec["detail"])

    def test_the_runner_plans_no_a32rec_and_the_v2_report_reads_only_a32_a64_c_d(self):
        from prec_diag import experiment as ex
        from prec_diag import f64_accuracy_v2 as v2
        for experiment in ex.READING_EXPERIMENTS + ex.CONTROL_EXPERIMENTS:
            self.assertFalse({r["arm"] for r in ex.plan(experiment)} & set(ho.DIAGNOSTIC_ARMS), experiment)
            self.assertFalse({r["base"] for r in ex.plan(experiment)} & set(ho.DIAGNOSTIC_ARMS), experiment)
        self.assertEqual(v2.REPORT_EXPERIMENTS, ("E1", "E2"))
        # With no runs at all, v2 names every run it would read: C, D, A32 and A64 on each input, nothing else.
        problems = v2.evaluate({}, {}, VOCAB)["problems"]
        read = {p.split(":")[0] for p in problems}
        self.assertEqual(read, {str((arm, i)) for arm in ("C", "D", "A32", "A64") for i in ex.INPUTS})


class ConvolutionProbe(unittest.TestCase):
    """Part 5's probe of the DeltaNet's depthwise causal conv (added with this
    family): each of its two sub-probes catches what the other cannot."""

    SIG = {"x": [[1, 12, 70], [840, 1, 12]], "w": [[12, 1, 4], [4, 4, 1]], "bias": None, "stride": [1],
           "padding": [3], "dilation": [1], "transposed": False, "output_padding": [0], "groups": 12}

    def rounding(self, which: int, low=torch.float32):
        real = op_probes.resolve("aten.convolution.default")

        def kernel(*args, **kwargs):
            args = list(args)
            args[which] = args[which].to(low).to(args[which].dtype)  # rounded, in its own dtype
            return real(*args, **kwargs)
        return kernel

    def test_it_passes_bit_exact_at_the_observed_geometry(self):
        r = op_probes.probe("aten.convolution.default", self.SIG)
        self.assertTrue(r["pass"], r)
        self.assertEqual(r["error"], 0.0)
        self.assertTrue(r["layout_reproduced"])
        self.assertEqual(r["reduction_length"], 4)

    def test_rounding_only_the_input_or_only_the_weight_fails(self):
        for which, sub in ((0, "input_operand"), (1, "accumulator_and_weight")):
            for low in (torch.float32, torch.bfloat16):
                with self.subTest(which=which, low=str(low)):
                    r = op_probes.probe("aten.convolution.default", self.SIG, op=self.rounding(which, low))
                    self.assertFalse(r["pass"])
                    self.assertFalse(r["sub_probes"][sub]["pass"])

    def test_a_layout_it_does_not_cover_fails_rather_than_passes(self):
        r = op_probes.probe("aten.convolution.default", {**self.SIG, "bias": [[12], [1]]})
        self.assertFalse(r["pass"])


class ExperimentScoping(Fixture):
    def test_the_runner_asks_for_full_attention_tables_only_and_binds_the_family_module(self):
        from prec_diag import experiment as ex
        manifest = {"checkpoint": {"directory": str(self.ckpt)}}
        self.assertEqual(ex.rope_table_labels(manifest), ["rope_table.full_attention.cos",
                                                          "rope_table.full_attention.sin"])
        self.assertEqual(ex.family_module_label(manifest), "script.hf_qwen3_5")
        self.assertEqual(ex.FAMILY_MODULES, {"script.hf_qwen3_5": "hf_qwen3_5.py"})


if __name__ == "__main__":
    unittest.main()
