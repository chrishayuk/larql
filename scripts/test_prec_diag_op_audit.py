"""Tests for prec_diag/op_audit.py and op_probes.py: A64 audit part 5 (#897).

- `Dispatch`: the trace on plain torch snippets (no model): what is a fault,
  what is recorded, and that a cast undone later is still seen at both ends.
- `Probes`: every kernel probe passes on this build, discriminates float32
  from float64, and fails a deliberately float32 kernel.
- `ForwardAD`: a forward-mode AD JVP through a tiny random Qwen3 passes
  with every op classified (probed, or exact with a stated reason), and a
  custom `autograd.Function` whose `jvp` computes in float32 is a fault.
- `GapTheModuleAuditCannotSee`: on the tiny random Gemma 3 and Qwen3
  checkpoints of the oracle's own tests, one known float32 site left
  unpatched (or one hidden float32 computation with no patch at all) fails
  this part, while the module-boundary audit (part 2) reports nothing.
- `Wiring`: `hf_oracle.run(op_audit=SCHEMA)` records the part, binds it in the
  identity, and makes a failed part a `fault`.
- `Rereading`: the hf-f64-accuracy/2 report reads an audit.json written
  before this part existed exactly as before.
- `Backfill`: the diagnostic backfill (`op_audit_backfill.py`) over the
  synthetic subjects.

No real checkpoint is read, and nothing is downloaded.

Run: python3 -I -m unittest discover -s scripts -p 'test_prec_diag_*.py'
"""

import ast
import json
import math
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prec_diag import evidence as ev  # noqa: E402
from prec_diag import f64_accuracy as fa  # noqa: E402
from prec_diag import f64_accuracy_v2 as f2  # noqa: E402
from prec_diag import families as fam  # noqa: E402
from prec_diag import hf_oracle as ho  # noqa: E402
from prec_diag import op_audit as oa  # noqa: E402
from prec_diag import op_audit_backfill as backfill  # noqa: E402
from prec_diag import op_probes as op  # noqa: E402
import test_prec_diag_hf_oracle as gemma  # noqa: E402
import test_prec_diag_hf_oracle_qwen3 as qwen3  # noqa: E402

import torch  # noqa: E402
import torch.autograd.forward_ad as fwAD  # noqa: E402

HERE = Path(__file__).name
F64 = torch.float64


def traced(fn):
    audit = oa.OpDtypeAudit()
    with torch.no_grad(), audit:
        fn()
    return audit.finish()


def linear_layers(model) -> int:
    return sum(isinstance(m, torch.nn.Linear) for m in model.modules())


def sites(result, key="violations"):
    return {r["site"].split(":")[0] for r in result[key]}


class Dispatch(unittest.TestCase):
    def setUp(self):
        self.x = torch.linspace(-1, 1, 12, dtype=F64).reshape(3, 4)

    def test_an_f64_computation_passes_and_probes_its_arithmetic(self):
        r = traced(lambda: torch.softmax(self.x @ self.x.T + 1.0, dim=-1).sum())
        self.assertTrue(r["pass"], r["problems"])
        probed = {p["op"] for p in r["probes"]}
        self.assertTrue({"aten.mm.default", "aten._softmax.default", "aten.add.Tensor",
                         "aten.sum.default"} <= probed, probed)
        self.assertEqual(r["non_f64_ops"], 0)
        self.assertEqual(r["schema"], oa.SCHEMA)

    def test_a_cast_that_is_undone_is_recorded_at_both_ends(self):
        out = {}
        r = traced(lambda: out.setdefault("y", (self.x.float() * 2).double()))
        self.assertEqual(out["y"].dtype, F64)  # a module boundary would see float64 only
        self.assertFalse(r["pass"])
        casts = {(c["from"], c["to"]) for c in r["casts"] if c["non_f64"]}
        self.assertEqual(casts, {("torch.float64", "torch.float32"), ("torch.float32", "torch.float64")})
        self.assertEqual(sites(r, "casts"), {HERE})
        self.assertIn(("aten.mul.Tensor", "input", "torch.float32"),
                      {(v["op"], v["role"], v["dtype"]) for v in r["violations"]})
        stack = r["casts"][0]["call_stack"][0]
        self.assertIn(".float()", stack["code"] + r["casts"][1]["call_stack"][0]["code"])

    def test_a_float32_dtype_argument_is_a_fault(self):
        r = traced(lambda: torch.softmax(self.x, -1, dtype=torch.float32))
        self.assertIn("dtype_argument", {v["role"] for v in r["violations"]})

    def test_a_default_dtype_factory_is_a_fault(self):
        r = traced(lambda: self.x + torch.zeros(4))
        self.assertIn(("aten.zeros.default", "output"), {(v["op"], v["role"]) for v in r["violations"]})
        self.assertIn(("aten.add.Tensor", "input"), {(v["op"], v["role"]) for v in r["violations"]})

    def test_writing_into_a_float32_state_buffer_is_a_fault(self):
        state = torch.zeros(3, 4, dtype=torch.float32)  # made before the trace, as a cache would be
        r = traced(lambda: state.copy_(self.x))
        self.assertIn(("aten.copy_.default", "write"), {(v["op"], v["role"]) for v in r["violations"]})
        self.assertEqual(r["writes"]["aten.copy_.default"], {"torch.float32": 1})

    def test_an_integer_to_f64_cast_is_recorded_and_is_not_a_fault(self):
        r = traced(lambda: torch.arange(4).to(F64) * self.x[0])
        self.assertTrue(r["pass"], r["problems"])
        self.assertEqual([(c["from"], c["to"], c["non_f64"]) for c in r["casts"]],
                         [("torch.int64", "torch.float64", False)])

    def test_an_op_with_no_precision_evidence_fails(self):
        r = traced(lambda: torch.atan2(self.x, self.x + 2))
        self.assertFalse(r["pass"])
        self.assertEqual(r["unclassified"], {"aten.atan2.default": 1})

    def test_an_empty_trace_fails(self):
        r = traced(lambda: None)
        self.assertFalse(r["pass"])
        self.assertIn("not active", r["problems"][0])

    def test_the_result_says_what_the_dispatch_level_cannot_see(self):
        r = traced(lambda: self.x + 1)
        self.assertEqual(r["visibility"], oa.VISIBILITY)
        self.assertTrue(any("inside one kernel" in v for v in r["visibility"]["does_not_see"]))
        self.assertEqual(r["environment"]["threads"], torch.get_num_threads())
        self.assertNotEqual(r["environment"]["blas"], "unknown")
        self.assertTrue(any("thread-local" in v for v in r["visibility"]["does_not_see"]))

    def test_the_probes_memory_is_recorded(self):
        r = traced(lambda: self.x @ self.x.T)
        self.assertEqual(r["memory"]["largest_probe_bytes"], max(p["probe_bytes"] for p in r["probes"]))
        self.assertGreaterEqual(r["memory"]["peak_rss_bytes_after_probes"],
                                r["memory"]["peak_rss_bytes_before_probes"])

    def test_every_probed_op_is_classified_and_no_op_is_in_two_classes(self):
        self.assertFalse(op.PROBED & oa.EXACT_OPS)
        names = [n for names in op.CLASSES.values() for n in names]
        self.assertEqual(len(names), len(set(names)))
        # The meta-only rule is narrower than a class: off meta these ops are probed.
        self.assertTrue(oa.SHAPE_INFERENCE_OPS <= op.CLASSES[op.BASIC])
        self.assertFalse(oa.SHAPE_INFERENCE_OPS & oa.EXACT_OPS)

    def test_a_prim_on_data_is_probed_and_on_meta_is_shape_inference(self):
        a, b = self.x, self.x + 0.5
        r = traced(lambda: torch.ops.prims.add.default(a, b))
        self.assertTrue(r["pass"], r["problems"])
        self.assertEqual(r["shape_inference"], {})
        self.assertIn("prims.add.default", {p["op"] for p in r["probes"]})
        r = traced(lambda: torch.ops.prims.mul.default(a.to("meta"), b.to("meta")))
        self.assertTrue(r["pass"], r["problems"])
        self.assertEqual(r["shape_inference"], {"prims.mul.default": 1})
        self.assertNotIn("prims.mul.default", {p["op"] for p in r["probes"]})
        self.assertNotIn(op.BASIC, r["ops_by_class"])

    def test_a_prim_mixing_meta_and_data_is_not_shape_inference(self):
        # The meta-only rule needs every tensor on meta: a CPU 0-dim operand keeps the call on data.
        s = torch.tensor(2.0, dtype=torch.float64)
        r = traced(lambda: torch.ops.prims.mul.default(s, self.x.to("meta")))
        self.assertEqual(r["shape_inference"], {})

    def test_a_float32_prim_on_meta_is_still_a_dtype_fault(self):
        m = torch.empty(3, 4, dtype=torch.float32, device="meta")
        r = traced(lambda: torch.ops.prims.add.default(m, m))
        self.assertFalse(r["pass"])
        self.assertIn(("prims.add.default", "input", "torch.float32"),
                      {(v["op"], v["role"], v["dtype"]) for v in r["violations"]})

    def test_an_aten_op_on_meta_is_not_shape_inference(self):
        # Only the named primitives have the meta-only rule; any other op keeps its class.
        r = traced(lambda: torch.atan2(self.x.to("meta"), self.x.to("meta")))
        self.assertEqual(r["unclassified"], {"aten.atan2.default": 1})


class Probes(unittest.TestCase):
    """Each probe on the real ATen kernel passes and discriminates; the same
    probe on a deliberately float32 kernel fails."""

    MATRIX = {"a": [[5, 640], [640, 1]], "b": [[640, 96], [1, 640]]}  # b transposed, as F.linear's weight
    SIGNATURES = {
        "aten.mm.default": MATRIX,
        "aten.bmm.default": {"a": [[2, 5, 33], [165, 33, 1]], "b": [[2, 33, 7], [231, 7, 1]]},
        "aten.addmm.default": {"bias": [[96], [1]], **MATRIX},
        "aten.sum.default": {"x": [[3, 50], [50, 1]]},
        "aten.sum.dim_IntList": {"x": [[3, 50], [50, 1]], "dim": [-1], "keepdim": False},
        "aten.mean.dim": {"x": [[1, 5, 640], [3200, 640, 1]], "dim": [-1], "keepdim": True},
        "aten.cumsum.default": {"x": [[4, 300], [300, 1]], "dim": 1},
        "aten.cumprod.default": {"x": [[4, 300], [300, 1]], "dim": -1},
        "aten._softmax.default": {"x": [[1, 2, 9, 9], [162, 81, 9, 1]], "dim": -1},
        "aten._log_softmax.default": {"x": [[3, 9], [9, 1]], "dim": -1},
        "aten.pow.Tensor_Scalar": {"exponent": 2.0},
        "aten.pow.Scalar": {"base": 10000.0},
        "aten.gelu.default": {"approximate": "tanh"},
        # Qwen3.5's depthwise causal conv: channels-last input (the projection transposed), groups = channels.
        "aten.convolution.default": {"x": [[1, 12, 70], [840, 1, 12]], "w": [[12, 1, 4], [4, 4, 1]], "bias": None,
                                     "stride": [1], "padding": [3], "dilation": [1], "transposed": False,
                                     "output_padding": [0], "groups": 12},
        # rsqrt's tangent in RMSNorm under forward AD: -0.5 * rsqrt^3 * t.
        "aten.mul.Scalar": {"scalar_operand": 1, "scalar": -0.5},
    }
    #: The basic-arithmetic probes forward-mode AD added (#897), at every scalar variant the forward used.
    FORWARD_AD_BASIC = (("aten.mul.Scalar", {"scalar_operand": 1, "scalar": 2.0}),
                        ("aten.mul.Scalar", {"scalar_operand": 1, "scalar": -0.5}),
                        ("aten.div_.Tensor", {}),
                        ("prims.add.default", {}), ("prims.add.default", {"scalar_operand": 1, "scalar": 0.0}),
                        ("prims.mul.default", {}), ("prims.mul.default", {"scalar_operand": 0, "scalar": 1.5}))

    def sig(self, name):
        return self.SIGNATURES.get(name, {})

    def test_every_probe_passes_and_discriminates_on_this_build(self):
        for name in sorted(op.PROBED):
            with self.subTest(name):
                r = op.probe(name, self.sig(name))
                self.assertTrue(r["pass"], r)
                self.assertGreater(r["f32_counterfactual_error"], r["bound"])

    def test_the_exact_probes_are_bit_exact(self):
        for name in op.CLASSES[op.ACCUMULATING] - {"aten._softmax.default", "aten._log_softmax.default"}:
            self.assertEqual(op.probe(name, self.sig(name))["error"], 0.0, name)

    def test_a_float32_accumulator_fails_every_accumulating_probe(self):
        for name in sorted(op.CLASSES[op.ACCUMULATING]):
            real = op.resolve(name)

            def f32_kernel(*args, real=real, **kwargs):
                f32 = [a.float() if isinstance(a, torch.Tensor) else a for a in args]
                return real(*f32, **kwargs).double()

            with self.subTest(name):
                r = op.probe(name, self.sig(name), op=f32_kernel)
                self.assertFalse(r["pass"], r)
                self.assertIn("not float64-accurate", r["detail"])

    def test_a_float32_elementwise_kernel_fails_its_probe(self):
        r = op.probe("aten.exp.default", {}, op=lambda x: torch.exp(x.float()).double())
        self.assertFalse(r["pass"])

    def test_every_forward_ad_basic_probe_passes_and_fails_a_float32_kernel(self):
        for name, sig in self.FORWARD_AD_BASIC:
            real = op.resolve(name)

            def f32_kernel(*args, real=real):
                return real(*[a.float() if isinstance(a, torch.Tensor) else a for a in args]).double()

            with self.subTest(name=name, sig=sig):
                ok = op.probe(name, sig)
                self.assertTrue(ok["pass"], ok)
                self.assertEqual(ok["error"], 0.0)
                bad = op.probe(name, sig, op=f32_kernel)
                self.assertFalse(bad["pass"], bad)
                self.assertIn("not float64-accurate", bad["detail"])

    def test_a_kernel_that_rounds_its_scalar_to_float32_fails(self):
        def rounded_scalar(x, scalar):
            return torch.ops.aten.mul.Scalar(x, float(torch.tensor(scalar, dtype=torch.float32)))

        sig = {"scalar_operand": 1, "scalar": 1.0 / 3.0}
        self.assertTrue(op.probe("aten.mul.Scalar", sig)["pass"])
        r = op.probe("aten.mul.Scalar", sig, op=rounded_scalar)
        self.assertFalse(r["pass"], r)
        self.assertIn("not float64-accurate", r["detail"])

    def test_the_in_place_probe_reads_unmodified_operands(self):
        # div_ writes into its first operand: a probe that let it would feed the f32 counterfactual x / y / y.
        seen = []

        def spy(a, b):
            seen.append(a.double().clone())
            return torch.ops.aten.div_.Tensor(a, b)

        r = op.probe("aten.div_.Tensor", {}, op=spy)
        self.assertTrue(r["pass"], r)
        f64_operand, f32_operand = seen
        self.assertTrue(torch.equal(f64_operand.float().double(), f32_operand))

    def test_a_float32_silu_backward_fails_its_probe(self):
        real = torch.ops.aten.silu_backward.default
        for kernel in (lambda g, x: real(g.float(), x.float()).double(),  # both operands
                       lambda g, x: real(g, x.float().double())):  # only x rounded
            r = op.probe("aten.silu_backward.default", {}, op=kernel)
            self.assertFalse(r["pass"], r)
            self.assertIn("not float64-accurate", r["detail"])

    def test_a_probe_that_cannot_tell_f32_from_f64_fails(self):
        # A "kernel" that returns the expected value whatever its operands makes
        # the float32 counterfactual meet the bound too: the probe then proves nothing.
        def blind(a, b):
            value = op.HEAD_A * op.HEAD_B + (a.shape[-1] - 1) * op.F64_EPS
            return torch.full((*a.shape[:-1], b.shape[-1]), value, dtype=F64)

        r = op.probe("aten.mm.default", self.MATRIX, op=blind)
        self.assertFalse(r["pass"])
        self.assertIn("cannot see the difference", r["detail"])

    def test_the_operand_heads_are_not_representable_in_f32_or_bf16(self):
        for head in (op.HEAD_A, op.HEAD_B, op.CUMPROD_FACTOR):
            for low in (torch.float32, torch.bfloat16):
                self.assertNotEqual(torch.tensor(head, dtype=F64).to(low).double().item(), head, (head, low))
        self.assertEqual(op.HEAD_A * op.HEAD_B, 1 + 2.0 ** -24 + 2.0 ** -26 + 2.0 ** -50)

    def test_rounding_the_operands_fails_every_accumulating_probe(self):
        """A kernel that rounds float64 operands to float32 or bfloat16 and then
        accumulates in float64. The float32 counterfactual rounds operands and
        accumulator together, so only the operand heads can catch this."""
        for low in (torch.float32, torch.bfloat16):
            for name in sorted(op.CLASSES[op.ACCUMULATING]):
                real = op.resolve(name)

                def rounding(*args, real=real, low=low, **kwargs):
                    rounded = [a.to(low).double() if isinstance(a, torch.Tensor) and a.dtype == F64 else a
                               for a in args]
                    return real(*rounded, **kwargs)

                with self.subTest(name=name, operands=str(low)):
                    r = op.probe(name, self.sig(name), op=rounding)
                    self.assertFalse(r["pass"], r)
                    self.assertIn("not float64-accurate", r["detail"])

    UNIFORM = ("aten.mm.default", "aten.addmm.default", "aten.bmm.default", "aten.sum.default",
               "aten.sum.dim_IntList", "aten.mean.dim")

    def test_a_nan_output_fails_every_uniform_probe(self):
        """Review of #899: Python's max drops a NaN, so a chunked scalar
        comparison must treat any NaN as an infinite error."""
        for name in self.UNIFORM:
            real = op.resolve(name)
            for where in ("one", "all"):
                def nan(*args, real=real, where=where, **kwargs):
                    out = real(*args, **kwargs).clone()
                    if where == "one":
                        out.view(-1)[0] = math.nan
                    else:
                        out.fill_(math.nan)
                    return out

                with self.subTest(name=name, nan=where):
                    r = op.probe(name, self.sig(name), op=nan)
                    self.assertFalse(r["pass"], r)
                    self.assertEqual(r["error"], math.inf)

    def test_a_wrong_shaped_output_fails_every_uniform_probe(self):
        """A uniform reference is one scalar, so the output shape is checked
        against the op's own shape inference."""
        for name in self.UNIFORM:
            real = op.resolve(name)

            def first(*args, real=real, **kwargs):
                return real(*args, **kwargs).reshape(-1)[:1].clone()

            with self.subTest(name=name):
                r = op.probe(name, self.sig(name), op=first)
                self.assertFalse(r["pass"], r)
                self.assertEqual(r["error"], math.inf)

    def test_every_probe_records_the_bytes_it_allocated(self):
        r = op.probe("aten.mm.default", self.MATRIX)
        (m, k), (_, n) = self.MATRIX["a"][0], self.MATRIX["b"][0]
        self.assertEqual(r["probe_bytes"], 8 * (m * k + k * n + m * n))

    def test_the_observed_layout_is_reproduced(self):
        self.assertTrue(op.probe("aten.mm.default", self.MATRIX)["layout_reproduced"])
        broadcast = {"a": [[2, 5, 33], [0, 33, 1]], "b": [[2, 33, 7], [231, 7, 1]]}
        r = op.probe("aten.bmm.default", broadcast)
        self.assertTrue(r["pass"], r)
        self.assertFalse(r["layout_reproduced"])

    def test_no_reduction_means_nothing_to_probe(self):
        r = op.probe("aten.mm.default", {"a": [[3, 1], [1, 1]], "b": [[1, 4], [4, 1]]})
        self.assertTrue(r["pass"])
        self.assertIn("nothing accumulates", r["detail"])

    def test_the_tolerance_is_the_patched_function_witness_bound(self):
        self.assertEqual(op.TOLERANCE, ho.F64_WITNESS_REL)


class GapTheModuleAuditCannotSee(unittest.TestCase):
    """The must-fail cases of #897 on the oracle's own tiny checkpoints."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls._tmp.name)
        cls.ckpts = {"gemma3_text": gemma.make_checkpoint(tmp / "gemma"), "qwen3": qwen3.make_checkpoint(tmp / "qwen3")}
        cls.fixtures = {"gemma3_text": gemma, "qwen3": qwen3}

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def forward(self, family_name, skip=frozenset(), tamper=None):
        """The A64 forward as `hf_oracle.forward` composes it, with `skip` and `tamper`."""
        fx = self.fixtures[family_name]
        model, _ = ho.load(self.ckpts[family_name], "A64", fx.BOUNDARY)
        if tamper:
            tamper(model)
        family = ho.Family(family_name)
        audit, ops = ho.A64Audit(model, family), oa.OpDtypeAudit(linear_layers=linear_layers(model))
        with ho.SiteCapture(model, None, family), ho.a64_patches(model, audit, skip), \
                ho.module_dtype_hooks(model, audit), ops:
            with torch.no_grad():
                logits = model(torch.tensor([fx.IDS]), use_cache=False).logits[0]
        parts = audit.finish(logits, fx.LAYERS)
        parts["parameters"] = ho.audit_parameters(model, self.ckpts[family_name])
        return parts, ops.finish()

    @staticmethod
    def module_boundary_problems(parts):
        return [p for p in parts["problems"] if p.startswith("module ")]

    def test_the_fully_patched_forward_passes(self):
        for name in self.ckpts:
            with self.subTest(name):
                parts, ops = self.forward(name)
                self.assertTrue(parts["pass"] and parts["parameters"]["pass"], parts["problems"])
                self.assertTrue(ops["pass"], ops["problems"])
                self.assertEqual(ops["non_f64_ops"], 0)
                self.assertGreater(ops["probes_run"], 0)
                # Without forward AD nothing is shape inference, and none of its op kinds appear.
                self.assertEqual(ops["shape_inference"], {})
                self.assertFalse(set(ops["op_counts"]) & set(ForwardAD.AD_OPS), ops["op_counts"])

    def test_one_known_f32_site_left_unpatched_fails_and_the_module_audit_sees_nothing(self):
        # Gemma 3 rotary and Qwen3's eager softmax each cast to float32 and back inside
        # one call, so every module boundary stays float64.
        for name, skip, modeling, function in (
                ("gemma3_text", "rotary", "modeling_gemma3.py", "Gemma3RotaryEmbedding.forward"),
                ("qwen3", "softmax", "modeling_qwen3.py", "eager_attention_forward"),
                ("qwen3", "rotary", "modeling_qwen3.py", "Qwen3RotaryEmbedding.forward")):
            with self.subTest(name=name, skip=skip):
                parts, ops = self.forward(name, frozenset({skip}))
                self.assertFalse(ops["pass"])
                self.assertEqual({v["site"].split(" ")[1] for v in ops["violations"]}, {function})
                self.assertEqual(sites(ops), {modeling})
                self.assertTrue(any(c["non_f64"] and c["to"] == "torch.float32" for c in ops["casts"]))
                self.assertTrue(any(c["non_f64"] and c["to"] == "torch.float64" for c in ops["casts"]))
                self.assertEqual(self.module_boundary_problems(parts), [])

    def test_a_forward_run_on_another_thread_is_a_partial_trace(self):
        """A dispatch mode is thread-local: the forward on another thread is
        not seen at all, and the matmul count derived from the model says so."""
        model, _ = ho.load(self.ckpts["gemma3_text"], "A64", gemma.BOUNDARY)
        ops = oa.OpDtypeAudit(linear_layers=linear_layers(model))
        ids = torch.tensor([gemma.IDS])

        def elsewhere():
            with torch.no_grad():
                model(ids, use_cache=False)

        with ops:
            model.model.embed_tokens(ids)  # a little work here, the forward on another thread
            worker = threading.Thread(target=elsewhere)
            worker.start()
            worker.join()
        r = ops.finish()
        self.assertFalse(r["pass"])
        self.assertTrue(any(p.startswith("partial trace: 0 mm/addmm") for p in r["problems"]), r["problems"])

    def test_one_mlp_on_another_thread_is_a_partial_trace(self):
        """Review of #899: counting attention's bmm towards the floor left
        slack for a layer's linear work moved off the trace."""
        model, _ = ho.load(self.ckpts["gemma3_text"], "A64", gemma.BOUNDARY)
        mlp = model.model.layers[0].mlp
        inner = mlp.forward

        def elsewhere(x):
            out = {}
            worker = threading.Thread(target=lambda: out.setdefault("y", inner(x)))
            worker.start()
            worker.join()
            return out["y"]

        mlp.forward = elsewhere
        ops = oa.OpDtypeAudit(linear_layers=linear_layers(model))
        with ops, torch.no_grad():
            model(torch.tensor([gemma.IDS]), use_cache=False)
        r = ops.finish()
        self.assertFalse(r["pass"])
        self.assertTrue(any(p.startswith("partial trace:") for p in r["problems"]), r["problems"])

    def test_tracing_part_of_the_forward_fails(self):
        model, _ = ho.load(self.ckpts["gemma3_text"], "A64", gemma.BOUNDARY)
        hidden = torch.zeros(1, 3, gemma.HIDDEN, dtype=F64)
        ops = oa.OpDtypeAudit(linear_layers=linear_layers(model))
        with ops, torch.no_grad():
            model.lm_head(hidden)
        r = ops.finish()
        self.assertFalse(r["pass"])
        self.assertIn(f"partial trace: 1 mm/addmm for {linear_layers(model)} nn.Linear modules",
                      " ".join(r["problems"]))

    def test_a_hidden_f32_computation_with_no_patch_fails_only_this_part(self):
        """No patch covers this site, so part 3 has nothing to count or witness,
        and every module input and output is float64: parts 1-4 pass."""
        def f32_inside(self_mlp, x):
            gate = self_mlp.act_fn(self_mlp.gate_proj(x))
            return self_mlp.down_proj((gate.float() * self_mlp.up_proj(x).float()).double())

        def tamper(model):
            mlp = model.model.layers[1].mlp
            mlp.forward = f32_inside.__get__(mlp)

        parts, ops = self.forward("gemma3_text", tamper=tamper)
        self.assertTrue(parts["pass"] and parts["parameters"]["pass"], parts["problems"])
        self.assertFalse(ops["pass"])
        self.assertEqual(sites(ops), {HERE})
        self.assertEqual({v["site"].split(" ")[1] for v in ops["violations"]},
                         {"GapTheModuleAuditCannotSee.test_a_hidden_f32_computation_with_no_patch_fails_only_this_part"
                          ".<locals>.f32_inside"})


class _F32Jvp(torch.autograd.Function):
    """`x @ w.T`: float64 forward, but a jvp that computes in float32."""

    @staticmethod
    def forward(x, w):
        return x @ w.T

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_forward(*inputs)

    @staticmethod
    def jvp(ctx, gx, gw):
        x, w = ctx.saved_tensors
        return (gx.float() @ w.float().T).double()


class ForwardAD(unittest.TestCase):
    """A forward-mode AD JVP (`torch.autograd.forward_ad`, dual tensors)
    through a tiny random Qwen3 with the oracle's A64 patches. The tangent of
    the logits in the input embeddings is computed by ordinary ATen ops below
    autograd, so the audit sees and classifies every one of them."""

    LAYERS, VOCAB = 2, 512
    #: The op kinds the JVP adds to the plain forward, and the class each call of it is counted in.
    AD_OPS = {"aten._efficientzerotensor.default": oa.EXACT, "aten.is_same_size.default": oa.EXACT,
              "aten._has_same_storage_numel.default": oa.EXACT, "aten.mul.Scalar": op.BASIC,
              "aten.div_.Tensor": op.BASIC, "aten.silu_backward.default": op.ELEMENTWISE,
              "prims.add.default": oa.EXACT, "prims.mul.default": oa.EXACT}  # on meta only: shape inference

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.ckpt = qwen3.make_checkpoint(Path(cls._tmp.name) / "qwen3", num_hidden_layers=cls.LAYERS,
                                         vocab_size=cls.VOCAB)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def jvp(self, tamper=None):
        model, _ = ho.load(self.ckpt, "A64", qwen3.BOUNDARY)
        self.assertEqual(model.config._attn_implementation, "eager")
        self.assertEqual((model.config.hidden_size, model.config.num_hidden_layers, model.config.vocab_size),
                         (qwen3.HIDDEN, self.LAYERS, self.VOCAB))
        if tamper:
            tamper(model)
        family = ho.Family("qwen3")
        audit, ops = ho.A64Audit(model, family), oa.OpDtypeAudit(linear_layers=linear_layers(model))
        with torch.no_grad():
            embeds = model.model.embed_tokens(torch.tensor([qwen3.IDS]))
        tangent = torch.linspace(-1, 1, embeds.numel(), dtype=F64).reshape(embeds.shape)
        with ho.a64_patches(model, audit), ops, torch.no_grad(), fwAD.dual_level():
            out = model(inputs_embeds=fwAD.make_dual(embeds, tangent), use_cache=False).logits
            logits, jvp = fwAD.unpack_dual(out)
        self.assertEqual(jvp.dtype, F64)
        self.assertGreater(float(jvp.abs().max()), 0.0)
        return audit.finish(logits[0], self.LAYERS), ops.finish()

    def test_a_jvp_passes_with_every_op_classified(self):
        parts, ops = self.jvp()
        self.assertTrue(parts["pass"], parts["problems"])
        self.assertTrue(ops["pass"], ops["problems"])
        self.assertEqual(ops["unclassified"], {})
        self.assertEqual(ops["non_f64_ops"], 0)
        self.assertTrue(set(self.AD_OPS) <= set(ops["op_counts"]), set(self.AD_OPS) - set(ops["op_counts"]))
        for name, kind in self.AD_OPS.items():
            if name in oa.SHAPE_INFERENCE_OPS:
                self.assertEqual(op.class_of(name), op.BASIC)  # probed when it runs on data
            else:
                self.assertEqual(op.class_of(name) or (oa.EXACT if name in oa.EXACT_OPS else None), kind, name)
        # Every prim call was the ZeroTensor kernels' shape inference on meta; none computed, none was probed.
        self.assertEqual(ops["shape_inference"], {n: ops["op_counts"][n] for n in oa.SHAPE_INFERENCE_OPS})
        probed = {p["op"] for p in ops["probes"]}
        self.assertFalse(probed & oa.SHAPE_INFERENCE_OPS)
        self.assertTrue({n for n, k in self.AD_OPS.items() if k != oa.EXACT} <= probed)
        self.assertEqual({p["signature"]["scalar"] for p in ops["probes"] if p["op"] == "aten.mul.Scalar"},
                         {2.0, -0.5})  # pow(2)'s and rsqrt's tangents
        self.assertEqual(sum(ops["ops_by_class"].values()), ops["floating_ops"])

    def test_a_float32_jvp_is_a_dtype_fault(self):
        """The negative control: one MLP's down_proj runs a custom Function
        whose forward is float64 and whose jvp computes in float32."""
        def tamper(model):
            proj = model.model.layers[1].mlp.down_proj
            proj.forward = lambda x: _F32Jvp.apply(x, proj.weight)

        parts, ops = self.jvp(tamper)
        self.assertTrue(parts["pass"], parts["problems"])
        self.assertFalse(ops["pass"])
        self.assertEqual(ops["unclassified"], {})
        self.assertEqual(sites(ops), {HERE})
        self.assertEqual({v["site"].split(" ")[1] for v in ops["violations"]}, {"_F32Jvp.jvp"})
        self.assertIn(("aten.mm.default", "input", "torch.float32"),
                      {(v["op"], v["role"], v["dtype"]) for v in ops["violations"]})
        self.assertTrue(any(c["non_f64"] and c["to"] == "torch.float32" for c in ops["casts"]))


class Wiring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        cls.ckpt = gemma.make_checkpoint(cls.tmp / "ckpt")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def arm(self, arm, name, **kw):
        out = self.tmp / "runs" / name
        rec = ho.run(arm=arm, ckpt=self.ckpt, ids=gemma.IDS, positions=None, out_dir=out, boundary=gemma.BOUNDARY,
                     require_threads=False, **kw)
        return out, rec

    def identity(self, run_dir):
        return json.loads((run_dir / ev.RESULT_FILE).read_text())["identity"]

    def test_a64_with_the_audit_records_and_binds_it(self):
        run_dir, rec = self.arm("A64", "with", op_audit=oa.SCHEMA)
        self.assertEqual(rec["status"], ev.COMPLETE, rec.get("detail"))
        audit = json.loads((run_dir / "audit.json").read_text())
        self.assertTrue(audit["operations"]["pass"], audit["operations"]["problems"])
        identity = self.identity(run_dir)
        self.assertEqual(identity["op_audit"], oa.SCHEMA)
        for name in ("op_audit.py", "op_probes.py"):
            self.assertEqual(identity["scripts"][name], ev.sha256_file(Path(oa.__file__).parent / name))
        self.assertEqual(fa.audit_problems(run_dir) + f2.operations_problems(run_dir, identity), [])

    def test_a64_without_the_audit_is_as_before(self):
        run_dir, rec = self.arm("A64", "without")
        self.assertEqual(rec["status"], ev.COMPLETE, rec.get("detail"))
        self.assertNotIn("operations", json.loads((run_dir / "audit.json").read_text()))
        self.assertIsNone(self.identity(run_dir)["op_audit"])
        self.assertEqual(fa.audit_problems(run_dir) + f2.operations_problems(run_dir, self.identity(run_dir)), [])
        self.assertTrue(f2.operations_problems(run_dir, {"op_audit": oa.SCHEMA}))

    def test_the_audit_on_another_arm_is_refused(self):
        _, rec = self.arm("A32", "a32", op_audit=oa.SCHEMA)
        self.assertEqual(rec["status"], ev.REFUSED)
        self.assertIn("A64", rec["detail"])

    def test_a_schema_this_oracle_does_not_implement_is_refused(self):
        _, rec = self.arm("A64", "other-schema", op_audit="larql.prec-diag-1.a64-op-audit/999")
        self.assertEqual(rec["status"], ev.REFUSED)
        self.assertIn("this oracle implements", rec["detail"])

    def test_a_failed_operation_audit_is_a_fault_and_not_admitted(self):
        failed = {"schema": oa.SCHEMA, "pass": False, "problems": ["x"], "problem_count": 1}
        with mock.patch.object(oa.OpDtypeAudit, "finish", return_value=failed):
            run_dir, rec = self.arm("A64", "failed", op_audit=oa.SCHEMA)
        self.assertEqual(rec["status"], ev.FAULT)
        self.assertFalse(ev.admit(run_dir).admitted)


class Rereading(unittest.TestCase):
    """An audit.json from before part 5 is read exactly as before."""

    OLD = {"pass": True, "problem_count": 0, "call_totals": {"rmsnorm": 3}, "expected_call_totals": {"rmsnorm": 3},
           "parameters": {"pass": True, "parameters": 5, "parameters_checked": 5}}
    LEGACY_IDENTITY = {"tool": "hf_oracle", "arm": "A64", "scripts": {"hf_oracle.py": "0" * 64}}

    def problems(self, audit, expected=None):
        """What the hf-f64-accuracy/2 report finds in this audit.json: v1's
        re-read (unchanged) plus v2's part-5 re-read."""
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "audit.json").write_text(json.dumps(audit))
            found = fa.audit_problems(Path(d)) + f2.operations_problems(Path(d), expected)
            return [p.replace(d, "<dir>") for p in found]

    def test_an_old_audit_gives_the_problems_it_always_gave(self):
        failed = dict(self.OLD, **{"pass": False})
        for audit, want in ((self.OLD, []), (failed, ["<dir>/audit.json: audit did not pass"])):
            self.assertEqual(self.problems(audit), want)
            self.assertEqual(self.problems(audit, self.LEGACY_IDENTITY), want)

    def test_the_report_imports_nothing_new(self):
        """A reporting repair binds exactly the modules the report imports
        (`f64_accuracy_v2.TOOLING_LABELS`), and in-flight freezes bind none of
        part 5's modules: the report must not import them."""
        code = ("import sys; sys.modules['torchvision'] = None; sys.path.insert(0, sys.argv[1]); "
                "from prec_diag import f64_accuracy_v2; "
                "print(sorted(m for m in sys.modules if m.startswith('prec_diag.')))")
        done = subprocess.run([sys.executable, "-I", "-c", code, str(Path(__file__).resolve().parent)],
                              capture_output=True, text=True, check=True)
        imported = {m.removeprefix("prec_diag.") for m in ast.literal_eval(done.stdout.strip())}
        bound = {label.removeprefix("script.") for label in f2.TOOLING_LABELS}
        self.assertEqual(imported, bound)

    def test_a_failed_operation_part_is_a_problem(self):
        audit = dict(self.OLD, operations={"schema": oa.SCHEMA, "pass": False, "problem_count": 2})
        self.assertEqual(self.problems(audit), ["<dir>/audit.json: operation-level dtype audit did not pass"])

    def test_a_freeze_that_requires_the_part_needs_it_recorded(self):
        required = dict(self.LEGACY_IDENTITY, op_audit=oa.SCHEMA)
        self.assertEqual(len(self.problems(self.OLD, required)), 1)
        passed = dict(self.OLD, operations={"schema": oa.SCHEMA, "pass": True, "problem_count": 0})
        self.assertEqual(self.problems(passed, required), [])
        other = dict(self.OLD, operations={"schema": "other/1", "pass": True, "problem_count": 0})
        self.assertEqual(len(self.problems(other, required)), 1)



class Backfill(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.report = backfill.report(backfill.synthetic_subjects(Path(cls._tmp.name)))
        cls.md = backfill.markdown(cls.report)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_it_is_labelled_diagnostic_and_leaves_sealed_verdicts_alone(self):
        self.assertTrue(self.report["diagnostic"])
        self.assertIn("Sealed verdicts are unchanged", self.md)
        self.assertIn(backfill.STATEMENT, self.md)

    def test_both_families_pass_and_the_unpatched_control_shows_the_gap(self):
        self.assertEqual({s["family"] for s in self.report["subjects"]}, set(fam.FAMILIES))
        for s in self.report["subjects"]:
            self.assertTrue(s["operations"]["pass"], s["operations"]["problems"])
            self.assertTrue(s["existing_audit"]["parts_2_to_4_pass"])
            self.assertFalse(s["control_unpatched"]["part_5_pass"])
            self.assertEqual(s["control_unpatched"]["module_boundary_problems"], 0)
            self.assertFalse(s["control_unpatched"]["part_3"]["pass"])
            self.assertNotIn("logits", json.dumps(s))
        gemma_scale = next(s for s in self.report["subjects"] if s["family"] == "gemma3_text")[
            "control_unpatched"]["embed_scale"]
        self.assertEqual(gemma_scale["dtype"], "torch.float64")
        # A64's float64 load builds the buffer in float64: sqrt(hidden), not a float32-rounded value.
        self.assertTrue(gemma_scale["equals_sqrt_hidden_float64"])
        self.assertFalse(gemma_scale["equals_sqrt_hidden_rounded_to_float32"])


if __name__ == "__main__":
    unittest.main()
