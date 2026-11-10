"""PREC-DIAG-1 transformers-side oracle: arms A32, A32s, A32r and A64 (B1).

One process runs one arm over one input:
- it writes logits at every position;
- with `--positions`, it also writes captures at E0a's 13 sites;
- everything runs under the lifecycle and admission rules in `evidence.py`.

  A32   f32 weights (the BF16 checkpoint widened), eager attention, no cache
  A32s  A32 with sdpa attention
  A32r  A32 with LARQL's own cos/sin tables injected into the rotary
        embedding (one value per pair, used for both halves); the rotation
        itself stays transformers' f32 rotate-half
  A64   float64 throughout, with the five f32 forcings of transformers 5.5.0's
        `modeling_gemma3.py` patched to f64, and the four-part audit:
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
          4. logits are float64.
        A failed audit makes the run `fault`: A64 is never admitted unaudited.

Captures go into `sites/`. They use E0a's names and layout:
- `[positions, width]` row-major;
- q and k are per head, head-major (`[heads, seq, hd]` -> `[seq, heads*hd]`);
- A32* planes are `<f4`, A64 planes `<f8`.

The site map (`SITE_MAP`) names the transformers tensor for each E0a site.
LARQL keeps q *after* QK-norm and before rotation. Gemma 3 applies
`query_pre_attn_scalar` to the scores, not to q. So, as long as the plan
declares no `query_scale`, no transformation is applied to any site before
comparison (`QUERY_SCALE_NOTE`).

Refused rather than run:
- a composite (multimodal) config;
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
from safetensors.torch import load_file  # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402

import oracle_loading  # noqa: E402
from prec_diag import evidence as ev  # noqa: E402

ARMS = ("A32", "A32s", "A32r", "A64")
#: transformers 5.5.0's modeling_gemma3.py, the file whose f32 forcings A64 patches.
FROZEN_MODELING_SHA256 = "221f147ccc0b16d7fa0aa007231ee0a6bd40fb37feb79eb9e537290498468aef"
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


# ------------------------------------------------------------- identity

def modeling_sha256() -> str:
    return ev.sha256_file(Path(G.__file__))


def script_digests() -> dict:
    here = Path(__file__).resolve().parent
    return {name: ev.sha256_file(here / name) for name in ("hf_oracle.py", "evidence.py")} | {
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

def load(ckpt: Path, arm: str, boundary: dict) -> tuple[torch.nn.Module, dict]:
    """The model for `arm` and its loading facts (gate as harness 2.1.0)."""
    config = AutoConfig.from_pretrained(ckpt)
    if getattr(config, "text_config", None) is not None:
        raise Refused("composite config: this oracle loads text-only Gemma 3 checkpoints")
    rope = getattr(config, "rope_parameters", None) or {}
    for layer_type, params in rope.items():
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

    def __init__(self, model, positions: list[int] | None):
        self.model, self.positions = model, positions
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
        self._orig_rotary = G.apply_rotary_pos_emb

        def rotary(q, k, cos, sin, *args, **kwargs):
            qo, ko = self._orig_rotary(q, k, cos, sin, *args, **kwargs)
            layer = self._rotary_calls
            self._rotary_calls += 1
            self._put("query_positioned", layer, _heads_to_rows(qo))
            self._put("key_positioned", layer, _heads_to_rows(ko))
            return qo, ko

        G.apply_rotary_pos_emb = rotary
        return self

    def __exit__(self, *exc):
        for handle in self._handles:
            handle.remove()
        G.apply_rotary_pos_emb = self._orig_rotary
        return False

    def check_complete(self, num_layers: int) -> None:
        if self._rotary_calls != num_layers:
            raise RuntimeError(f"apply_rotary_pos_emb ran {self._rotary_calls} times, expected {num_layers}")
        if self.positions is None:
            return
        want = {(s, None) for s in LAYERLESS_SITES} | {(s, L) for s in LAYER_SITES for L in range(num_layers)}
        if set(self.planes) != want:
            raise RuntimeError(f"capture incomplete: missing {sorted(want - set(self.planes), key=str)[:5]}")


# ------------------------------------------------------------------- A32r

def load_tables(table_dirs: dict[str, Path], positions: int, head_dim: int) -> dict:
    """LARQL's shipped cos/sin tables per layer type, as the
    `rope_angle_tables` example wrote them: `[positions, head_dim/2]` f32."""
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
def larql_tables(tables: dict, record: dict):
    """Replace the rotary embedding's output with LARQL's tables, each value
    used for both halves, exactly as LARQL applies them."""
    orig = G.Gemma3RotaryEmbedding.forward
    record.update({"injected_calls": 0, "bit_identical_to_files": True})

    def forward(self, x, position_ids, layer_type=None):
        t = tables[layer_type]
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

    G.Gemma3RotaryEmbedding.forward = forward
    try:
        yield record
    finally:
        G.Gemma3RotaryEmbedding.forward = orig


# -------------------------------------------------------------------- A64

def _rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


class A64Audit:
    """Parts 2 and 3 of the audit, collected during the forward."""

    def __init__(self, model):
        self.model = model
        self.problems: list[str] = []
        self.calls: dict[str, dict] = {"rmsnorm": {}, "rotary": {}, "softmax": {}, "embedding": {}}
        self.witness: dict[str, float] = {}
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

    def expected_calls(self, num_layers: int) -> dict:
        return {"rmsnorm": 6 * num_layers + 1, "rotary": num_layers, "softmax": num_layers, "embedding": 1}

    def finish(self, logits: torch.Tensor, num_layers: int) -> dict:
        self.check("logits", logits)
        totals = {k: sum(v.values()) for k, v in self.calls.items()}
        expected = self.expected_calls(num_layers)
        if totals != expected:
            self.problems.append(f"patched-function call counts {totals} != {expected}")
        for kind in ("rmsnorm", "rotary", "softmax", "embedding"):
            if kind not in self.witness:
                self.problems.append(f"{kind}: no f64 witness taken")
        return {"pass": not self.problems, "problems": self.problems[:50],
                "problem_count": len(self.problems), "calls": self.calls, "call_totals": totals,
                "expected_call_totals": expected, "witness_relative_error": self.witness,
                "module_tensors_checked": self.module_tensors}


def audit_parameters(model, ckpt: Path) -> dict:
    """Audit part 1: every parameter float64 and equal to the checkpoint's
    BF16 value widened exactly (the composite-config trap would load BF16)."""
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
    """The five f32 forcings, patched to f64. `skip` exists for the tests
    that prove the audit catches a missing patch; a real run passes none."""
    saved = {"rms": G.Gemma3RMSNorm.forward, "rot": G.Gemma3RotaryEmbedding.forward,
             "eager": G.eager_attention_forward, "emb": G.Gemma3TextScaledWordEmbedding.forward}
    names = audit.names
    hidden = model.config.hidden_size

    def layer_of(name: str) -> str:
        parts = name.split(".")
        return parts[parts.index("layers") + 1] if "layers" in parts else "final"

    def rms(self, x):
        audit.check(f"rmsnorm in {names.get(id(self))}", x, self.weight)
        out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        out = out * (1.0 + self.weight)
        audit.check(f"rmsnorm out {names.get(id(self))}", out)
        audit.count("rmsnorm", layer_of(names.get(id(self), "")))
        xd = x.detach().numpy()
        want = xd / np.sqrt(np.mean(xd * xd, axis=-1, keepdims=True) + self.eps) * (1.0 + self.weight.detach().numpy())
        audit.witness_once("rmsnorm", out, want)
        return out

    def rot(self, x, position_ids, layer_type=None):
        base = self.config.rope_parameters[layer_type]["rope_theta"]
        dim = getattr(self.config, "head_dim", None) or self.config.hidden_size // self.config.num_attention_heads
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
        angles = position_ids[0].to(torch.float64)[:, None] * inv_freq[None, :]
        emb = torch.cat((angles, angles), dim=-1)[None]
        cos, sin = emb.cos(), emb.sin()
        audit.check(f"rotary {layer_type}", inv_freq, angles, cos, sin)
        audit.count("rotary", layer_type)
        p = position_ids[0].numpy().astype(np.float64)
        f = 1.0 / (float(base) ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
        audit.witness_once("rotary", cos[0, :, : dim // 2], np.cos(p[:, None] * f[None, :]))
        return cos, sin

    def eager(module, query, key, value, attention_mask, dropout=0.0, scaling=None, softcap=None, **kw):
        if scaling is None:
            scaling = module.head_dim ** -0.5
        k = G.repeat_kv(key, module.num_key_value_groups)
        v = G.repeat_kv(value, module.num_key_value_groups)
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

    if "rmsnorm" not in skip:
        G.Gemma3RMSNorm.forward = rms
    if "rotary" not in skip:
        G.Gemma3RotaryEmbedding.forward = rot
    if "softmax" not in skip:
        G.eager_attention_forward = eager
    if "embedding" not in skip:
        G.Gemma3TextScaledWordEmbedding.forward = emb
    try:
        yield audit
    finally:
        G.Gemma3RMSNorm.forward = saved["rms"]
        G.Gemma3RotaryEmbedding.forward = saved["rot"]
        G.eager_attention_forward = saved["eager"]
        G.Gemma3TextScaledWordEmbedding.forward = saved["emb"]


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

def forward(model, ids: list[int], arm: str, positions, tables=None, ckpt=None) -> dict:
    """One forward for `arm`; returns logits, planes and the arm's records."""
    num_layers = model.config.num_hidden_layers
    out: dict = {"records": {}}
    with contextlib.ExitStack() as stack:
        cap = stack.enter_context(SiteCapture(model, positions))
        if arm == "A32r":
            stack.enter_context(larql_tables(tables, out["records"].setdefault("tables", {})))
        if arm == "A64":
            audit = A64Audit(model)
            stack.enter_context(a64_patches(model, audit))
            stack.enter_context(module_dtype_hooks(model, audit))
        with torch.no_grad():
            logits = model(torch.tensor([ids]), use_cache=False).logits[0]
    cap.check_complete(num_layers)
    if arm == "A64":
        out["records"]["audit"] = {"parameters": audit_parameters(model, ckpt),
                                   **audit.finish(logits, num_layers)}
    if arm == "A32r":
        t = out["records"]["tables"]
        if t["injected_calls"] != num_layers or not t["bit_identical_to_files"]:
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
        boundary: dict, table_dirs: dict | None = None, expected_modeling_sha256: str = FROZEN_MODELING_SHA256,
        require_threads: bool = True, bind: dict | None = None) -> dict:
    """One arm over one input under the lifecycle. Returns the final record.

    `bind` is recorded verbatim in the identity: the experiment runner
    passes the freeze digest, experiment and run id, so a run made under one
    freeze is never admitted under another."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    identity = {"tool": "hf_oracle", "scripts": script_digests(), "arm": arm,
                "checkpoint": checkpoint_identity(ckpt), "input": ids_digests(ids) | {"length": len(ids)},
                "positions": positions, "modeling_gemma3_sha256": modeling_sha256(),
                "threads": thread_state(), "environment": ev.numeric_environment(dict(os.environ)),
                "packages": packages(), "machine": platform.machine(), "os": platform.platform(),
                "bind": bind}
    rr = ev.RunRecord.begin(out_dir, (bind or {}).get("run_id", arm), "hf-oracle", identity)
    try:
        if identity["modeling_gemma3_sha256"] != expected_modeling_sha256:
            raise Refused("modeling_gemma3.py differs from the frozen file")
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
            tables = load_tables(table_dirs, len(ids), model.config.head_dim)
            identity["tables"] = {lt: {k: v for k, v in t.items() if k.endswith("sha256")}
                                  for lt, t in tables.items()}
        result = forward(model, ids, arm, positions, tables, ckpt)
        if arm == "A64":
            audit = result["records"]["audit"]
            if not (audit["pass"] and audit["parameters"]["pass"]):
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
              bind=json.loads(a.bind_json.read_text()) if a.bind_json else None)
    print(rec["status"])
    return 0 if rec["status"] == ev.COMPLETE else 1


if __name__ == "__main__":
    sys.exit(main())
