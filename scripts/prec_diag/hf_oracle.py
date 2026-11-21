"""PREC-DIAG-1 transformers-side oracle: arms A32, A32s, A32r, A64 (B1), and A32rec (diagnostic).

Families (`families.py`): Gemma 3 text, Qwen3 dense (logits only) and
Qwen3.5 hybrid (logits only; its patches, witnesses and loader are in
`hf_qwen3_5.py`). The family is the checkpoint's `model_type`; A64 patches
that family's modeling file, and the audit's call counts are that family's,
derived from the config.

One process runs one arm over one input:
- it writes logits at every position;
- with `--positions`, it also writes captures at E0a's 13 sites;
- everything runs under the lifecycle and admission rules in `evidence.py`.

  A32   f32 weights (the BF16 checkpoint widened), eager attention, no cache
  A32s  A32 with sdpa attention
  A32r  A32 with LARQL's own cos/sin tables injected into the rotary
        embedding (one value per pair, used for both halves); the rotation
        itself stays transformers' f32 rotate-half. Tables are given only
        for the layer types that apply rotary (`families.rope_layer_types`):
        for Qwen3.5 the full-attention layers, at the partial rotary width,
        on text-only positions; a table for its DeltaNet layers is refused
  A32rec  DIAGNOSTIC ONLY (Qwen3.5): A32 with transformers' own float32
        step-by-step `torch_recurrent_gated_delta_rule` used for the prefill
        instead of the chunked rule. It is never a gate baseline: the
        experiment runner plans no A32rec run, the oracle refuses one made
        under a freeze (`bind`), and its identity names its own arm, so it
        can never be admitted as A32
  A64   float64 throughout, with the f32 forcings of transformers 5.5.0's
        modeling file patched to f64 (Gemma 3: RMSNorm, rotary, eager
        softmax, scaled embedding; Qwen3: RMSNorm, rotary, eager softmax;
        Qwen3.5: (1+w) RMSNorm, gated RMSNorm, rotary with partial rotary
        and M-RoPE, eager softmax, the DeltaNet forward's decay and the
        chunked gated delta rule, per layer instance), and the audit (four
        parts, and a fifth when requested):
          1. parameters: every parameter is float64 and equals the
             checkpoint's BF16 value widened exactly (compared against the
             safetensors bytes);
          2. modules: every floating tensor into or out of every module is
             float64;
          3. patched functions: each one asserts its own intermediates' dtype,
             counts its calls per layer (the counts must equal the
             architecture's), and is checked once against an independent f64
             recomputation in numpy (relative error <= 1e-12; the unpatched
             f32 path is ~1e-8, so a silently skipped patch fails);
          4. logits are float64;
          5. operations (`op_audit.py`, #897), only when the run is made with
             `op_audit` (`--op-audit SCHEMA`; the experiment runner passes the
             freeze's `values.a64_op_audit`): every ATen op of the
             forward reads, returns and writes float64 only, every dtype
             conversion is recorded with its call site, and every op with
             arithmetic has kernel-precision evidence (probes). Part 2 sees
             module boundaries only, so it cannot see a cast to float32 that
             is undone before a module returns; this part can. A run made
             without it records `op_audit: null` in its identity and no
             `operations` part, exactly as before.
        A failed audit makes the run `fault`: A64 is never admitted unaudited.

Captures go into `sites/` (Gemma 3 only; a Qwen3 capture is refused). They use E0a's names and layout:
- `[positions, width]` row-major;
- q and k are per head, head-major (`[heads, seq, hd]` -> `[seq, heads*hd]`);
- A32* planes are `<f4`, A64 planes `<f8`.

The site map (`SITE_MAP`) names the transformers tensor for each E0a site.
LARQL keeps q *after* QK-norm and before rotation. Gemma 3 applies
`query_pre_attn_scalar` to the scores, not to q. So, as long as the plan
declares no `query_scale`, no transformation is applied to any site before
comparison (`QUERY_SCALE_NOTE`).

Refused rather than run:
- a composite (multimodal) config, except a family that declares its text
  prefix (Qwen3.5: text-config load, every checkpoint tensor accounted for);
- a modeling file whose sha256 differs from the one the protocol froze;
- a rope type other than `default`;
- thread settings other than the protocol's (when `--require-threads`).
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import platform
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

sys.modules.setdefault("torchvision", None)  # the installed torchvision does not match torch
import numpy as np  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402
import transformers.models.gemma3.modeling_gemma3 as G  # noqa: E402
import transformers.models.qwen3.modeling_qwen3 as Q  # noqa: E402
import transformers.models.qwen3_5.modeling_qwen3_5 as Q35  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402

import oracle_loading  # noqa: E402
from prec_diag import evidence as ev  # noqa: E402
from prec_diag import families as fam  # noqa: E402
from prec_diag import hf_qwen3_5 as q35  # noqa: E402
from prec_diag import op_audit as oa  # noqa: E402

ARMS = ("A32", "A32s", "A32r", "A64", "A32rec")
#: Arms that are diagnostics only: never planned by the experiment runner and
#: refused under a freeze, so never a gate baseline.
DIAGNOSTIC_ARMS = ("A32rec",)

#: The protocol's thread settings (rev 3 [C8]).
THREADS = {"torch_num_threads": 8, "torch_num_interop_threads": 1, "OMP_NUM_THREADS": "8"}
#: An f64 computation agrees with its numpy f64 recomputation far below this;
#: the f32 path it replaces misses it by orders of magnitude.
F64_WITNESS_REL = 1e-12

LAYERLESS_SITES = ("embedding", "head_input")
LAYER_SITES = ("attention_input", "query_unpositioned", "key_unpositioned", "query_positioned",
               "key_positioned", "attention_output", "attention_delta", "ffn_input", "ffn_output",
               "ffn_delta", "residual")
#: E0a site -> the transformers tensor captured (Gemma 3, transformers 5.5.0).
SITE_MAP = {
    "embedding": "model.embed_tokens output (scaled by embed_scale)",
    "attention_input": "layers[L].input_layernorm output",
    "query_unpositioned": "layers[L].self_attn.q_norm output, [b,h,s,hd] -> [s, h*hd]",
    "key_unpositioned": "layers[L].self_attn.k_norm output, [b,kv,s,hd] -> [s, kv*hd]",
    "query_positioned": "apply_rotary_pos_emb q output (layer L's call), [b,h,s,hd] -> [s, h*hd]",
    "key_positioned": "apply_rotary_pos_emb k output (layer L's call), [b,kv,s,hd] -> [s, kv*hd]",
    "attention_output": "layers[L].self_attn.o_proj output",
    "attention_delta": "layers[L].post_attention_layernorm output",
    "ffn_input": "layers[L].pre_feedforward_layernorm output",
    "ffn_output": "layers[L].mlp output",
    "ffn_delta": "layers[L].post_feedforward_layernorm output",
    "residual": "layers[L] (decoder layer) output",
    "head_input": "model.norm output",
}
QUERY_SCALE_NOTE = ("transformers applies query_pre_attn_scalar**-0.5 to the attention scores "
                    "(eager_attention_forward `scaling`), not to q; LARQL's captured q is unscaled "
                    "unless the plan declares a query_scale. No transformation is applied.")


class Refused(Exception):
    """The run cannot be made under the protocol (recorded as `refused`)."""


class AuditFailure(Exception):
    """A64's audit failed (recorded as `fault`)."""


# --------------------------------------------------------------- families

class Family:
    """The torch side of one `families.FAMILIES` entry: which classes A64
    patches, how its norm and rotary compute, and how often each runs."""

    def __init__(self, name: str):
        self.name = name
        self.facts = fam.FAMILIES[name]
        if name == "gemma3_text":
            self.module, self.norm_offset = G, 1.0
            self.rms_cls, self.rot_cls, self.emb_cls = (G.Gemma3RMSNorm, G.Gemma3RotaryEmbedding,
                                                        G.Gemma3TextScaledWordEmbedding)
        elif name == "qwen3":
            self.module, self.norm_offset = Q, 0.0
            self.rms_cls, self.rot_cls, self.emb_cls = Q.Qwen3RMSNorm, Q.Qwen3RotaryEmbedding, None
        elif name == "qwen3_5":
            self.module, self.norm_offset = Q35, 1.0
            self.rms_cls, self.rot_cls, self.emb_cls = Q35.Qwen3_5RMSNorm, Q35.Qwen3_5TextRotaryEmbedding, None
        else:
            raise Refused(f"no oracle patches for family {name!r}")

    @classmethod
    def of(cls, ckpt: Path) -> "Family":
        config = json.loads((Path(ckpt) / "config.json").read_text())
        try:
            fam.config_layer_types(config)  # refuses a layout no patch covers
            return cls(fam.family_of(config))
        except fam.UnsupportedFamily as e:
            raise Refused(str(e)) from None

    def eps(self, norm) -> float:
        return norm.eps if self.name in ("gemma3_text", "qwen3_5") else norm.variance_epsilon

    def theta(self, config, layer_type) -> float:
        params = config.rope_parameters
        return params[layer_type]["rope_theta"] if self.name == "gemma3_text" else params["rope_theta"]

    def rotary_calls(self, num_layers: int) -> int:
        """Gemma 3 builds cos/sin per layer; Qwen3 and Qwen3.5 once per forward."""
        return num_layers if self.name == "gemma3_text" else 1

    def attention_layers(self, config) -> int:
        """Layers that run softmax attention (and so apply rotary)."""
        if self.name == "qwen3_5":
            return q35.layer_types(config).count("full_attention")
        return config.num_hidden_layers

    def rotary_width(self, config) -> int:
        """The rotated width per head (Qwen3.5: partial rotary)."""
        return q35.rotary_dim(config) if self.name == "qwen3_5" else config.head_dim

    @property
    def patch_kinds(self) -> tuple:
        if self.name == "gemma3_text":
            return ("rmsnorm", "rotary", "softmax", "embedding")
        if self.name == "qwen3_5":
            return ("rmsnorm", "gated_rmsnorm", "rotary", "softmax", "gdn", "chunk_rule")
        return ("rmsnorm", "rotary", "softmax")

    def expected_calls(self, num_layers: int, layer_types: list[str] | None = None) -> dict:
        """Gemma 3: input, post-attention, pre- and post-FFN norms plus q/k
        norms per layer, and the final norm. Qwen3: input and post-attention
        norms plus q/k norms per layer, and the final norm. Qwen3.5: from
        the config's layer types (`hf_qwen3_5.expected_calls`)."""
        if self.name == "gemma3_text":
            return {"rmsnorm": 6 * num_layers + 1, "rotary": num_layers, "softmax": num_layers, "embedding": 1}
        if self.name == "qwen3_5":
            if layer_types is None:
                raise ValueError("qwen3_5 call counts need the config's layer types")
            return q35.expected_calls(num_layers, layer_types)
        return {"rmsnorm": 4 * num_layers + 1, "rotary": 1, "softmax": num_layers}


# ------------------------------------------------------------- identity

def modeling_sha256(family: Family | None = None) -> str:
    return ev.sha256_file(Path((family.module if family else G).__file__))


def script_digests(family: Family | None = None) -> dict:
    """The oracle's scripts, plus the family's own module when it has one
    (so the identity of an older family is unchanged)."""
    here = Path(__file__).resolve().parent
    names = ("hf_oracle.py", "evidence.py", "families.py", "op_audit.py", "op_probes.py")
    module = family.facts.get("oracle_module") if family is not None else None
    extra = (module,) if module else ()
    return {name: ev.sha256_file(here / name) for name in names + extra} | {
        "oracle_loading.py": ev.sha256_file(here.parent / "oracle_loading.py")}


def checkpoint_files(ckpt: Path) -> list[Path]:
    files = sorted(Path(ckpt).glob("*.safetensors"))
    if not files:
        raise Refused(f"{ckpt}: no safetensors files")
    return files


def checkpoint_identity(ckpt: Path) -> dict:
    return {"config_sha256": ev.sha256_file(Path(ckpt) / "config.json"),
            "safetensors": {p.name: ev.sha256_file(p) for p in checkpoint_files(ckpt)}}


def thread_state() -> dict:
    return {"torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS")}


def set_threads() -> None:
    torch.set_num_threads(THREADS["torch_num_threads"])
    torch.set_num_interop_threads(THREADS["torch_num_interop_threads"])


def packages() -> dict:
    import safetensors
    return {"transformers": transformers.__version__, "torch": torch.__version__,
            "numpy": np.__version__, "safetensors": safetensors.__version__,
            "python": platform.python_version()}


def ids_digests(ids: list[int]) -> dict:
    return {"ids_json_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            "ids_u32le_sha256": hashlib.sha256(np.asarray(ids, dtype="<u4").tobytes()).hexdigest()}


# --------------------------------------------------------------- loading

def composite_facts(ckpt: Path) -> dict | None:
    """The family facts of a checkpoint whose family loads from a composite
    config (it declares `text_prefix`), else None."""
    model_type = json.loads((Path(ckpt) / "config.json").read_text()).get("model_type")
    facts = fam.FAMILIES.get(model_type)
    return facts if facts and "text_prefix" in facts else None


def load(ckpt: Path, arm: str, boundary: dict) -> tuple[torch.nn.Module, dict]:
    """The model for `arm` and its loading facts (gate as harness 2.1.0).
    A composite family (Qwen3.5) loads its text model from `text_config`,
    with every parameter and buffer dtype asserted as loaded."""
    facts = composite_facts(ckpt)
    if facts is not None:
        dtype = torch.float64 if arm == "A64" else torch.float32
        try:
            model, record = q35.load_text_model(ckpt, dtype, "sdpa" if arm == "A32s" else "eager", boundary, facts)
        except q35.Unsupported as e:
            raise Refused(str(e)) from None
        if record["gate"] != "pass":
            raise Refused(f"loading gate: {record['gate_problems']}")
        return model, record
    config = AutoConfig.from_pretrained(ckpt)
    if getattr(config, "text_config", None) is not None:
        raise Refused("composite config: this oracle loads text-only checkpoints")
    rope = getattr(config, "rope_parameters", None) or {}
    per_type = rope.items() if all(isinstance(v, dict) for v in rope.values()) else [("all", rope)]
    for layer_type, params in per_type:
        if params and params.get("rope_type", "default") != "default":
            raise Refused(f"rope_type {params.get('rope_type')!r} for {layer_type}: only default is patched")
    dtype = torch.float64 if arm == "A64" else torch.float32
    attn = "sdpa" if arm == "A32s" else "eager"
    model, info = AutoModelForCausalLM.from_pretrained(ckpt, dtype=dtype, attn_implementation=attn,
                                                       output_loading_info=True)
    keys = []
    for f in checkpoint_files(ckpt):
        keys += oracle_loading.safetensors_header_keys(f)
    facts = oracle_loading.loading_facts(
        model_class=type(model).__name__, expected_keys=model.state_dict().keys(), checkpoint_keys=keys,
        missing=info.get("missing_keys", []), unexpected=info.get("unexpected_keys", []),
        mismatched=info.get("mismatched_keys", []),
        tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", False)), key_mapping=None,
        boundary=boundary)
    if facts["gate"] != "pass":
        raise Refused(f"loading gate: {facts['gate_problems']}")
    return model.to(dtype).eval(), facts


# ---------------------------------------------------------------- capture

def _heads_to_rows(t: torch.Tensor) -> torch.Tensor:
    """[1, heads, seq, hd] -> [seq, heads*hd] (head-major per position)."""
    return t[0].transpose(0, 1).reshape(t.shape[2], -1)


def _first(o):
    return o[0] if isinstance(o, tuple) else o


class SiteCapture:
    """Hooks for E0a's 13 sites. Values are kept for `positions` only."""

    def __init__(self, model, positions: list[int] | None, family: Family):
        self.model, self.positions, self.family = model, positions, family
        self.planes: dict[tuple[str, int | None], np.ndarray] = {}
        self._handles = []
        self._rotary_calls = 0
        self._orig_rotary = None

    def _put(self, site, layer, rows: torch.Tensor):
        if self.positions is None:
            return
        key = (site, layer)
        if key in self.planes:
            raise RuntimeError(f"site {key} delivered twice in one forward")
        self.planes[key] = rows[self.positions].detach().numpy().copy()

    def __enter__(self):
        m = self.model.model
        h = self._handles
        self._orig_rotary = self.family.module.apply_rotary_pos_emb
        self._install_rotary()
        if self.positions is None:
            return self
        h.append(m.embed_tokens.register_forward_hook(lambda mod, i, o: self._put("embedding", None, o[0])))
        h.append(m.norm.register_forward_hook(lambda mod, i, o: self._put("head_input", None, o[0])))
        for L, layer in enumerate(m.layers):
            def out(site, L=L, rows=lambda o: o[0]):
                return lambda mod, i, o: self._put(site, L, rows(_first(o)))
            heads = lambda o: _heads_to_rows(o)  # noqa: E731
            a = layer.self_attn
            h += [layer.input_layernorm.register_forward_hook(out("attention_input")),
                  a.q_norm.register_forward_hook(out("query_unpositioned", rows=heads)),
                  a.k_norm.register_forward_hook(out("key_unpositioned", rows=heads)),
                  a.o_proj.register_forward_hook(out("attention_output")),
                  layer.post_attention_layernorm.register_forward_hook(out("attention_delta")),
                  layer.pre_feedforward_layernorm.register_forward_hook(out("ffn_input")),
                  layer.mlp.register_forward_hook(out("ffn_output")),
                  layer.post_feedforward_layernorm.register_forward_hook(out("ffn_delta")),
                  layer.register_forward_hook(out("residual"))]
        return self

    def _install_rotary(self):
        def rotary(q, k, cos, sin, *args, **kwargs):
            qo, ko = self._orig_rotary(q, k, cos, sin, *args, **kwargs)
            layer = self._rotary_calls
            self._rotary_calls += 1
            self._put("query_positioned", layer, _heads_to_rows(qo))
            self._put("key_positioned", layer, _heads_to_rows(ko))
            return qo, ko

        self.family.module.apply_rotary_pos_emb = rotary

    def __exit__(self, *exc):
        for handle in self._handles:
            handle.remove()
        self.family.module.apply_rotary_pos_emb = self._orig_rotary
        return False

    def check_complete(self, num_layers: int, attention_layers: int | None = None) -> None:
        """`attention_layers`: how many layers apply rotary (all of them
        unless given; Qwen3.5's full-attention layers only)."""
        want = num_layers if attention_layers is None else attention_layers
        if self._rotary_calls != want:
            raise RuntimeError(f"apply_rotary_pos_emb ran {self._rotary_calls} times, expected {want}")
        if self.positions is None:
            return
        want = {(s, None) for s in LAYERLESS_SITES} | {(s, L) for s in LAYER_SITES for L in range(num_layers)}
        if set(self.planes) != want:
            raise RuntimeError(f"capture incomplete: missing {sorted(want - set(self.planes), key=str)[:5]}")


# ------------------------------------------------------------------- A32r

def load_tables(table_dirs: dict[str, Path], positions: int, head_dim: int) -> dict:
    """LARQL's shipped cos/sin tables per layer type, as the
    `rope_angle_tables` example wrote them: `[positions, head_dim/2]` f32.
    `head_dim` is the rotated width (`Family.rotary_width`: Qwen3.5's
    partial rotary width, not its head size)."""
    out = {}
    for layer_type, d in table_dirs.items():
        entry = {}
        for name in ("cos", "sin"):
            path = Path(d) / f"larql.{name}.f32"
            raw = path.read_bytes()
            if len(raw) != positions * (head_dim // 2) * 4:
                raise Refused(f"{path}: {len(raw)} bytes, expected [{positions}, {head_dim // 2}] f32")
            entry[name] = np.frombuffer(raw, dtype="<f4").reshape(positions, head_dim // 2)
            entry[f"{name}_sha256"] = ev.sha256_bytes(raw)
        out[layer_type] = entry
    return out


@contextlib.contextmanager
def larql_tables(tables: dict, record: dict, family: Family):
    """Replace the rotary embedding's output with LARQL's tables, each value
    used for both halves, exactly as LARQL applies them. A family whose
    rotary takes no layer type (Qwen3) is all full attention."""
    orig = family.rot_cls.forward
    record.update({"injected_calls": 0, "bit_identical_to_files": True})

    def forward(self, x, position_ids, layer_type=None):
        t = tables[layer_type or "full_attention"]
        if family.name == "qwen3_5":
            position_ids = q35.text_positions(position_ids)[0]  # one M-RoPE axis: text-only, all equal
        pos = position_ids[0].numpy()
        cos = torch.from_numpy(np.concatenate([t["cos"][pos]] * 2, axis=-1))[None]
        sin = torch.from_numpy(np.concatenate([t["sin"][pos]] * 2, axis=-1))[None]
        half = t["cos"].shape[1]
        for built, name in ((cos, "cos"), (sin, "sin")):
            for part in (built[0, :, :half], built[0, :, half:]):
                if part.numpy().tobytes() != t[name][pos].tobytes():
                    record["bit_identical_to_files"] = False
        record["injected_calls"] += 1
        return cos.to(x.dtype), sin.to(x.dtype)

    family.rot_cls.forward = forward
    try:
        yield record
    finally:
        family.rot_cls.forward = orig


# -------------------------------------------------------------------- A64

def _rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


class A64Audit:
    """Parts 2 and 3 of the audit, collected during the forward."""

    def __init__(self, model, family: Family):
        self.model, self.family = model, family
        self.layer_types = getattr(model.config, "layer_types", None) if family.name == "qwen3_5" else None
        self.problems: list[str] = []
        self.calls: dict[str, dict] = {k: {} for k in family.patch_kinds}
        self.witness: dict[str, float] = {}
        self.witness_detail: dict[str, dict] = {}
        self.module_tensors = 0
        names = {id(mod): name for name, mod in model.named_modules()}
        self.names = names

    def check(self, where: str, *tensors):
        for t in tensors:
            if isinstance(t, torch.Tensor) and t.is_floating_point() and t.dtype != torch.float64:
                self.problems.append(f"{where}: {t.dtype}")

    def count(self, kind: str, key) -> None:
        self.calls[kind][str(key)] = self.calls[kind].get(str(key), 0) + 1

    def witness_once(self, kind: str, got: torch.Tensor, want: np.ndarray) -> None:
        if kind in self.witness:
            return
        rel = _rel(got.detach().numpy(), want)
        self.witness[kind] = rel
        if not rel <= F64_WITNESS_REL:
            self.problems.append(f"{kind}: f64 witness relative error {rel:.3e} > {F64_WITNESS_REL}")

    def witness_every(self, kind: str, where: str, rel: float, bound: float = F64_WITNESS_REL) -> None:
        """A witness taken on every call: the worst relative error is kept,
        and every call over `bound` is a problem."""
        self.witness[kind] = max(self.witness.get(kind, 0.0), rel)
        if not rel <= bound:
            self.problems.append(f"{kind} {where}: f64 witness relative error {rel:.3e} > {bound}")

    def expected_calls(self, num_layers: int) -> dict:
        return self.family.expected_calls(num_layers, self.layer_types)

    def finish(self, logits: torch.Tensor, num_layers: int) -> dict:
        self.check("logits", logits)
        totals = {k: sum(v.values()) for k, v in self.calls.items()}
        expected = self.expected_calls(num_layers)
        if totals != expected:
            self.problems.append(f"patched-function call counts {totals} != {expected}")
        for kind in expected:
            if kind not in self.witness:
                self.problems.append(f"{kind}: no f64 witness taken")
        return {"pass": not self.problems, "problems": self.problems[:50],
                "problem_count": len(self.problems), "calls": self.calls, "call_totals": totals,
                "expected_call_totals": expected, "witness_relative_error": self.witness,
                "module_tensors_checked": self.module_tensors,
                **({"witness_detail": self.witness_detail} if self.witness_detail else {})}


def audit_parameters(model, ckpt: Path) -> dict:
    """Audit part 1: every parameter float64 and equal to the checkpoint's
    BF16 value widened exactly (the composite-config trap would load BF16).
    A composite family maps names through its text prefix and admits BF16
    and F32 sources (`hf_qwen3_5.audit_parameters`)."""
    facts = composite_facts(ckpt)
    if facts is not None:
        return q35.audit_parameters(model, ckpt, facts)
    problems, checked = [], 0
    stored = {}
    for f in checkpoint_files(ckpt):
        stored.update(load_file(str(f)))
    params = dict(model.named_parameters(remove_duplicate=False))
    for name, p in params.items():
        if p.dtype != torch.float64:
            problems.append(f"{name}: {p.dtype}")
            continue
        key = name if name in stored else ("model.embed_tokens.weight" if name == "lm_head.weight" else None)
        if key not in stored:
            problems.append(f"{name}: not in the checkpoint")
            continue
        src = stored[key]
        if src.dtype != torch.bfloat16:
            problems.append(f"{name}: checkpoint dtype {src.dtype}, expected bfloat16")
            continue
        if not torch.equal(p.detach(), src.to(torch.float64)):
            problems.append(f"{name}: differs from the checkpoint widened exactly")
        checked += 1
    buffers = {}
    for name, b in model.named_buffers():
        if b.is_floating_point() and b.dtype != torch.float64:
            problems.append(f"buffer {name}: {b.dtype}")
        buffers[name] = str(b.dtype)
    return {"pass": not problems, "problems": problems[:50], "parameters_checked": checked,
            "parameters": len(params)}


@contextlib.contextmanager
def a64_patches(model, audit: A64Audit, skip: frozenset = frozenset()):
    """The family's f32 forcings, patched to f64. `skip` exists for the tests
    that prove the audit catches a missing patch; a real run passes none."""
    family = audit.family
    M = family.module
    saved = {"rms": family.rms_cls.forward, "rot": family.rot_cls.forward, "eager": M.eager_attention_forward}
    if family.emb_cls is not None:
        saved["emb"] = family.emb_cls.forward
    names = audit.names
    offset = family.norm_offset
    hidden = model.config.hidden_size

    def layer_of(name: str) -> str:
        parts = name.split(".")
        return parts[parts.index("layers") + 1] if "layers" in parts else "final"

    def rms(self, x):
        audit.check(f"rmsnorm in {names.get(id(self))}", x, self.weight)
        eps = family.eps(self)
        out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        out = out * (offset + self.weight)
        audit.check(f"rmsnorm out {names.get(id(self))}", out)
        audit.count("rmsnorm", layer_of(names.get(id(self), "")))
        xd = x.detach().numpy()
        want = xd / np.sqrt(np.mean(xd * xd, axis=-1, keepdims=True) + eps) * (offset + self.weight.detach().numpy())
        audit.witness_once("rmsnorm", out, want)
        return out

    def rot(self, x, position_ids, layer_type=None):
        base = family.theta(self.config, layer_type)
        dim = getattr(self.config, "head_dim", None) or self.config.hidden_size // self.config.num_attention_heads
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
        angles = position_ids[0].to(torch.float64)[:, None] * inv_freq[None, :]
        emb = torch.cat((angles, angles), dim=-1)[None]
        cos, sin = emb.cos(), emb.sin()
        audit.check(f"rotary {layer_type}", inv_freq, angles, cos, sin)
        audit.count("rotary", layer_type or "full_attention")
        p = position_ids[0].numpy().astype(np.float64)
        f = 1.0 / (float(base) ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
        audit.witness_once("rotary", cos[0, :, : dim // 2], np.cos(p[:, None] * f[None, :]))
        return cos, sin

    def eager(module, query, key, value, attention_mask, dropout=0.0, scaling=None, softcap=None, **kw):
        if scaling is None:
            scaling = module.head_dim ** -0.5
        k = M.repeat_kv(key, module.num_key_value_groups)
        v = M.repeat_kv(value, module.num_key_value_groups)
        scores = torch.matmul(query, k.transpose(2, 3)) * scaling
        if softcap is not None:
            scores = torch.tanh(scores / softcap) * softcap
        if attention_mask is not None:
            scores = scores + attention_mask
        weights = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float64)
        audit.check(f"softmax layer {module.layer_idx}", query, k, v, scores, weights, attention_mask)
        audit.count("softmax", module.layer_idx)
        s = scores.detach().numpy()
        e = np.exp(s - s.max(axis=-1, keepdims=True))
        audit.witness_once("softmax", weights, e / e.sum(axis=-1, keepdims=True))
        out = torch.matmul(weights, v).transpose(1, 2).contiguous()
        return out, weights

    def emb(self, input_ids):
        scale = torch.tensor(math.sqrt(hidden), dtype=torch.float64)
        base = torch.nn.functional.embedding(input_ids, self.weight, self.padding_idx)
        out = base * scale
        audit.check("embedding", self.weight, base, scale, out)
        audit.count("embedding", "embed_tokens")
        audit.witness_once("embedding", out, base.detach().numpy() * math.sqrt(hidden))
        return out

    restore = []
    if family.name == "qwen3_5":
        rot = q35.rotary_forward(audit)
        restore = qwen3_5_patches(model, audit, skip, layer_of)
    if "rmsnorm" not in skip:
        family.rms_cls.forward = rms
    if "rotary" not in skip:
        family.rot_cls.forward = rot
    if "softmax" not in skip:
        M.eager_attention_forward = eager
    if family.emb_cls is not None and "embedding" not in skip:
        family.emb_cls.forward = emb
    try:
        yield audit
    finally:
        family.rms_cls.forward = saved["rms"]
        family.rot_cls.forward = saved["rot"]
        M.eager_attention_forward = saved["eager"]
        if family.emb_cls is not None:
            family.emb_cls.forward = saved["emb"]
        for owner, attr, value in reversed(restore):
            setattr(owner, attr, value)


def qwen3_5_patches(model, audit: A64Audit, skip: frozenset, layer_of) -> list:
    """Qwen3.5's own A64 patches (`hf_qwen3_5.py`): the gated norm, the
    DeltaNet forward (decay in float64) and the chunked rule, installed on
    every DeltaNet instance (the modeling file binds it in `__init__`, so a
    module-level patch would never run). The decode rule is replaced by a
    refusal on every instance: a no-cache prefill must never reach it.
    Returns what to restore, as (owner, attribute, original)."""
    names = audit.names
    restore = [(Q35.Qwen3_5RMSNormGated, "forward", Q35.Qwen3_5RMSNormGated.forward),
               (Q35.Qwen3_5GatedDeltaNet, "forward", Q35.Qwen3_5GatedDeltaNet.forward)]

    def on_gates(module, conv_in, conv_out, a, b, beta, g):
        layer = module.layer_idx
        audit.check(f"gdn layer {layer}", conv_in, conv_out, a, b, beta, g, module.A_log, module.dt_bias,
                    module.conv1d.weight)
        audit.count("gdn", layer)
        want = q35.gates_witness(module, conv_in, a, b)
        rel = {"conv": _rel(conv_out.detach().numpy(), want["conv"]), "beta": _rel(beta.detach().numpy(), want["beta"]),
               "g": _rel(g.detach().numpy(), want["g"])}
        audit.witness_detail.setdefault("gdn", {})[str(layer)] = rel
        audit.witness_every("gdn", f"layer {layer}", max(rel.values()))

    if "gated_rmsnorm" not in skip:
        Q35.Qwen3_5RMSNormGated.forward = q35.gated_norm_forward(audit, names, layer_of, lambda m: m.variance_epsilon)
    if "gdn" not in skip:
        Q35.Qwen3_5GatedDeltaNet.forward = q35.gdn_forward(on_gates)
    for module in q35.deltanets(model):
        restore += [(module, "chunk_gated_delta_rule", module.chunk_gated_delta_rule),
                    (module, "recurrent_gated_delta_rule", module.recurrent_gated_delta_rule)]
        module.recurrent_gated_delta_rule = q35.recurrent_rule_refused
        if "chunk_rule" in skip:
            continue

        def on_result(inputs, out, states, chunk_size, layer=module.layer_idx):
            audit.check(f"chunk_rule layer {layer}", *inputs[:5], out, *states)
            audit.count("chunk_rule", layer)
            cmp = q35.compare_with_witness(inputs, out, states, chunk_size)
            audit.witness_detail.setdefault("chunk_rule", {})[str(layer)] = cmp
            for part in ("output", "state"):
                audit.witness_every("chunk_rule", f"layer {layer} {part}s", cmp[part], q35.RECURRENCE_WITNESS_REL)

        module.chunk_gated_delta_rule = q35.chunk_rule(on_result)
    return restore


@contextlib.contextmanager
def recurrent_prefill(model, record: dict):
    """A32rec (diagnostic): transformers' own float32 step-by-step rule for
    the prefill, on every DeltaNet instance; restored afterwards."""
    modules = q35.deltanets(model)
    saved = [(m, m.chunk_gated_delta_rule) for m in modules]
    counter: dict = {}
    for m in modules:
        m.chunk_gated_delta_rule = q35.recurrent_rule_for_prefill(counter)
    record.update({"diagnostic_only": True, "expected_calls": len(modules),
                   "rule": "transformers torch_recurrent_gated_delta_rule (float32, unmodified) for the prefill",
                   "note": "not a gate baseline: hf-f64-accuracy/2 reads A32, A64, C and D only"})
    try:
        yield record
    finally:
        for m, rule in saved:
            m.chunk_gated_delta_rule = rule
        record["calls"] = counter.get("calls", 0)


@contextlib.contextmanager
def module_dtype_hooks(model, audit: A64Audit):
    """Audit part 2: every floating tensor into or out of every module."""
    def flat(x):
        if isinstance(x, torch.Tensor):
            yield x
        elif isinstance(x, (tuple, list)):
            for y in x:
                yield from flat(y)
        elif isinstance(x, dict):
            for y in x.values():
                yield from flat(y)
        elif hasattr(x, "__dict__") and type(x).__name__.endswith("Output"):
            yield from flat(dict(x))

    def hook(mod, args, kwargs, out):
        tensors = list(flat(args)) + list(flat(kwargs)) + list(flat(out))
        audit.module_tensors += len(tensors)
        audit.check(f"module {audit.names.get(id(mod)) or type(mod).__name__}", *tensors)

    handles = [m.register_forward_hook(hook, with_kwargs=True) for m in model.modules()]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


# -------------------------------------------------------------------- run

def forward(model, ids: list[int], arm: str, positions, tables=None, ckpt=None, family: Family | None = None,
            op_audit: bool = False) -> dict:
    """One forward for `arm`; returns logits, planes and the arm's records.
    With `op_audit` (A64 only), audit part 5 traces the forward's operations."""
    family = family or Family("gemma3_text")
    tracer = None
    num_layers = model.config.num_hidden_layers
    out: dict = {"records": {}}
    with contextlib.ExitStack() as stack:
        cap = stack.enter_context(SiteCapture(model, positions, family))
        if arm == "A32r":
            stack.enter_context(larql_tables(tables, out["records"].setdefault("tables", {}), family))
        if arm == "A32rec":
            stack.enter_context(recurrent_prefill(model, out["records"].setdefault("recurrent_prefill", {})))
        if arm == "A64":
            audit = A64Audit(model, family)
            stack.enter_context(a64_patches(model, audit))
            stack.enter_context(module_dtype_hooks(model, audit))
            if op_audit:
                linear = sum(isinstance(m, torch.nn.Linear) for m in model.modules())
                tracer = stack.enter_context(oa.OpDtypeAudit(linear_layers=linear))
        with torch.no_grad():
            logits = model(torch.tensor([ids]), use_cache=False).logits[0]
    cap.check_complete(num_layers, family.attention_layers(model.config))
    if arm == "A32rec":
        r = out["records"]["recurrent_prefill"]
        if r["calls"] != r["expected_calls"]:
            raise RuntimeError(f"A32rec: the recurrent rule ran {r['calls']} times, expected {r['expected_calls']}")
    if arm == "A64":
        out["records"]["audit"] = {"parameters": audit_parameters(model, ckpt),
                                   **audit.finish(logits, num_layers)}
        if tracer is not None:
            out["records"]["audit"]["operations"] = tracer.finish()
    if arm == "A32r":
        t = out["records"]["tables"]
        if t["injected_calls"] != family.rotary_calls(num_layers) or not t["bit_identical_to_files"]:
            raise RuntimeError(f"A32r table injection: {t}")
    out["logits"], out["planes"] = logits.detach().numpy(), cap.planes
    return out


def write_outputs(run_dir: Path, result: dict, arm: str, positions, model) -> list[str]:
    dt = "<f8" if arm == "A64" else "<f4"
    ext = "f64" if arm == "A64" else "f32"
    outputs = [f"logits.{ext}"]
    (run_dir / f"logits.{ext}").write_bytes(result["logits"].astype(dt).tobytes())
    if positions is not None:
        sites = run_dir / "sites"
        sites.mkdir()
        files = []
        for (site, layer), plane in sorted(result["planes"].items(), key=lambda kv: (kv[0][1] is not None,
                                                                                    kv[0][1] or 0, kv[0][0])):
            name = f"{site}.{ext}" if layer is None else f"{site}.L{layer:03}.{ext}"
            data = plane.astype(dt).tobytes()
            (sites / name).write_bytes(data)
            files.append({"site": site, "layer": layer, "rows": plane.shape[0], "width": plane.shape[1],
                          "dtype": dt, "file": name, "sha256": ev.sha256_bytes(data)})
            outputs.append(f"sites/{name}")
        cfg = model.config
        manifest = {"schema": "larql.prec-diag-1.hf-capture/1", "positions": positions,
                    "hidden": cfg.hidden_size, "site_map": SITE_MAP, "query_scale": QUERY_SCALE_NOTE,
                    "layers": [{"layer": L, "num_q_heads": cfg.num_attention_heads,
                                "num_kv_heads": cfg.num_key_value_heads, "head_dim": cfg.head_dim}
                               for L in range(cfg.num_hidden_layers)],
                    "files": files}
        (sites / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
        outputs.append("sites/manifest.json")
    for name, rec in result["records"].items():
        (run_dir / f"{name}.json").write_text(json.dumps(rec, indent=1, sort_keys=True, default=str))
        outputs.append(f"{name}.json")
    return outputs


def run(*, arm: str, ckpt: Path, ids: list[int], positions: list[int] | None, out_dir: Path,
        boundary: dict, table_dirs: dict | None = None, expected_modeling_sha256: str | None = None,
        require_threads: bool = True, bind: dict | None = None, op_audit: str | None = None) -> dict:
    """One arm over one input under the lifecycle. Returns the final record.

    `bind` is recorded verbatim in the identity: the experiment runner
    passes the freeze digest, experiment and run id, so a run made under one
    freeze is never admitted under another.

    `op_audit` names the schema of audit part 5 to add (A64 only; any
    schema but `op_audit.SCHEMA` is refused) and is recorded in the identity,
    so a freeze that requires the part admits no run made without it."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    try:
        family = Family.of(ckpt)
    except Refused:
        family = None  # refused below, once the record exists
    identity = {"tool": "hf_oracle", "scripts": script_digests(family), "arm": arm,
                "checkpoint": checkpoint_identity(ckpt), "input": ids_digests(ids) | {"length": len(ids)},
                "positions": positions,
                "modeling": ({"family": family.name, "sha256": modeling_sha256(family)} if family else None),
                "threads": thread_state(), "environment": ev.numeric_environment(dict(os.environ)),
                "packages": packages(), "machine": platform.machine(), "os": platform.platform(),
                "bind": bind, "op_audit": op_audit}
    rr = ev.RunRecord.begin(out_dir, (bind or {}).get("run_id", arm), "hf-oracle", identity)
    try:
        if family is None:
            Family.of(ckpt)  # raises the refusal with its reason
        frozen = expected_modeling_sha256 or family.facts["modeling_sha256"]
        if identity["modeling"]["sha256"] != frozen:
            raise Refused(f"modeling_{family.facts['modeling']}.py differs from the frozen file")
        if op_audit is not None and arm != "A64":
            raise Refused(f"the operation-level audit is part of A64's audit, not {arm}'s")
        if op_audit is not None and op_audit != oa.SCHEMA:
            raise Refused(f"operation-level audit {op_audit!r} requested; this oracle implements {oa.SCHEMA!r}")
        if positions is not None and not family.facts["captures"]:
            raise Refused(f"site captures are not mapped for {family.name}")
        if arm in DIAGNOSTIC_ARMS and family.name != "qwen3_5":
            raise Refused(f"{arm} replaces the chunked gated delta rule; {family.name} has no recurrence")
        if arm in DIAGNOSTIC_ARMS and bind is not None:
            raise Refused(f"{arm} is a diagnostic arm, never run under a freeze (it is not a gate baseline)")
        if arm == "A32r" and table_dirs:
            rotary = fam.rope_layer_types(json.loads((Path(ckpt) / "config.json").read_text()))
            if sorted(table_dirs) != rotary:
                raise Refused(f"A32r tables for {sorted(table_dirs)}: {family.name} applies rotary on {rotary} only"
                              + (" (its DeltaNet layers take no positions, so a RoPE table has no meaning there)"
                                 if family.name == "qwen3_5" else ""))
        if require_threads and identity["threads"] != THREADS:
            raise Refused(f"threads {identity['threads']} != {THREADS}")
        if positions is not None and (positions != sorted(set(positions)) or positions[-1] >= len(ids)):
            raise Refused("positions must be strictly increasing and inside the input")
        model, facts = load(ckpt, arm, boundary)
        (out_dir / "loading.json").write_text(json.dumps(facts, indent=1, sort_keys=True))
        tables = None
        if arm == "A32r":
            if not table_dirs:
                raise Refused("A32r needs LARQL's tables (--tables layer_type=DIR)")
            tables = load_tables(table_dirs, len(ids), family.rotary_width(model.config))
            identity["tables"] = {lt: {k: v for k, v in t.items() if k.endswith("sha256")}
                                  for lt, t in tables.items()}
        result = forward(model, ids, arm, positions, tables, ckpt, family, op_audit=op_audit is not None)
        if arm == "A64":
            audit = result["records"]["audit"]
            if not (audit["pass"] and audit["parameters"]["pass"] and audit.get("operations", {}).get("pass", True)):
                write_outputs(out_dir, {"logits": result["logits"], "planes": {}, "records": result["records"]},
                              arm, None, model)
                return rr.fail(ev.FAULT, {"a64_audit": "failed"}, audit=audit)
        if not np.isfinite(result["logits"]).all():
            return rr.fail(ev.FAULT, {"logits": "non-finite values"})
        outputs = write_outputs(out_dir, result, arm, positions, model) + ["loading.json"]
        return rr.complete(outputs, tables=identity.get("tables"))
    except Refused as e:
        return rr.fail(ev.REFUSED, str(e))
    except Exception as e:  # anything else after begin is a fault, never a stuck in_progress
        return rr.fail(ev.FAULT, f"{type(e).__name__}: {e}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--ids", required=True, type=Path)
    ap.add_argument("--positions", help="comma-separated capture positions")
    ap.add_argument("--boundary", required=True, type=Path,
                    help="JSON from oracle_loading.capability_boundary over `larql vindex3 plan`")
    ap.add_argument("--tables", action="append", default=[], help="LAYER_TYPE=DIR (A32r)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--bind-json", type=Path, help="JSON object recorded verbatim in the identity")
    ap.add_argument("--op-audit", metavar="SCHEMA",
                    help="A64 only: add the operation-level dtype audit (audit part 5, op_audit.py) of this schema")
    a = ap.parse_args(argv)
    if os.environ.get("OMP_NUM_THREADS") != THREADS["OMP_NUM_THREADS"]:
        print(f"refused: OMP_NUM_THREADS must be {THREADS['OMP_NUM_THREADS']} in the environment",
              file=sys.stderr)
        return 2
    set_threads()
    ids = json.loads(a.ids.read_text())
    positions = [int(p) for p in a.positions.split(",")] if a.positions else None
    tables = dict(t.split("=", 1) for t in a.tables)
    rec = run(arm=a.arm, ckpt=a.checkpoint, ids=ids, positions=positions, out_dir=a.out,
              boundary=json.loads(a.boundary.read_text()), table_dirs={k: Path(v) for k, v in tables.items()},
              bind=json.loads(a.bind_json.read_text()) if a.bind_json else None, op_audit=a.op_audit)
    print(rec["status"])
    return 0 if rec["status"] == ev.COMPLETE else 1


if __name__ == "__main__":
    sys.exit(main())
