"""The `qwen3_5` family of the PREC-DIAG transformers oracle (`hf_oracle.py`).

Qwen3.5 is a hybrid: most layers are Gated DeltaNet (`linear_attention`, a
gated delta-rule recurrence), the rest gated full attention. transformers
5.5.0's `modeling_qwen3_5.py` hard-casts to float32 inside the forward
whatever the model dtype (#897, the Qwen3.5-0.8B feasibility study):

- both gated-delta-rule kernels: q, k, v, beta, g and the recurrent state;
- the decay `g = -exp(A_log) * softplus(a + dt_bias)` (`A_log.float()`);
- the (1 + w) RMSNorm and the gated RMSNorm (w * norm(x) * silu(z));
- rotary: an f32 `inv_freq` buffer, partial rotary, interleaved M-RoPE;
- the eager softmax.

This module holds what A64 patches for the family, each with an independent
numpy float64 witness, plus the text-config loader, the loading gate and two
diagnostic pieces:

`chunk_rule`  transformers' `torch_chunk_gated_delta_rule` with its float32
    casts removed and nothing else changed: the same chunked (WY-style)
    algorithm, dtype-preserving. On float32 inputs it is bit-identical to
    transformers' own rule (a test pins this), so on float64 inputs it is
    that algorithm in float64.
`step_witness`  the independent check of it: a numpy float64 step-by-step
    gated delta recurrence written from the published update (Yang, Kautz,
    Hatamizadeh, "Gated Delta Networks", ICLR 2025, eq. 10; the state laid
    out [d_k, d_v] as transformers keeps it):

        S_t = alpha_t (I - beta_t k_t k_t^T) S_{t-1} + beta_t k_t v_t^T
        o_t = S_t^T q_t,   alpha_t = exp(g_t),

    with q and k L2-normalised (`q / sqrt(sum q^2 + eps)`, the FLA l2norm the
    rule applies in-kernel) and q scaled by d_k^-0.5. It is a different
    algorithm (one rank-1 update per token, no chunking, no triangular
    solve), so agreement says the patched chunked rule computes the
    recurrence, not only that it runs in float64. It is checked on every
    DeltaNet layer of every A64 forward: the outputs at every position and
    the state at every chunk end (positions 64k - 1) and after the last token.
`gdn_forward`  `Qwen3_5GatedDeltaNet.forward` for the no-cache prefill A64
    runs, line for line, with the decay computed in the model dtype. On a
    float32 model it is bit-identical to transformers' forward (a test pins
    this). Its witness recomputes the depthwise causal conv + SiLU, beta and
    g in numpy.
`recurrent_rule_for_prefill`  A32rec only (diagnostic): transformers' own,
    unmodified float32 `torch_recurrent_gated_delta_rule` used for the
    prefill instead of the chunked rule. Never a gate baseline.

The witness bound for the recurrence (`RECURRENCE_WITNESS_REL`) is the
oracle's `F64_WITNESS_REL`, 1e-12, applied per position (and per chunk-end
state) rather than over the whole tensor, so a late position cannot hide
behind early ones. Justification: both computations are float64 (unit
roundoff 1.1e-16). The chunked form reorders the arithmetic (a forward
substitution over 64 positions, decays as exp(g_i - g_j) differences), so
the two differ by rounding only; the observed worst case is recorded in
every audit (`witness_relative_error`) and is orders of magnitude below the
bound on the test fixture and on Qwen3.5-0.8B, while transformers' float32
rule misses the same witness by ~1e-7 (the witness has power: a test shows
it rejects the shipped f32 rule).
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import transformers.models.qwen3_5.modeling_qwen3_5 as Q35
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM

import oracle_loading

#: The chunk rule's default chunk length (transformers' `chunk_size=64`).
CHUNK = 64
#: The epsilon the modeling file passes to l2norm (`use_qk_l2norm_in_kernel`).
L2NORM_EPS = 1e-6
#: `F.softplus`'s default threshold: above it softplus(x) returns x.
SOFTPLUS_THRESHOLD = 20.0
#: Per-position relative bound for the recurrence witness (see the module docstring).
RECURRENCE_WITNESS_REL = 1e-12
#: Source dtypes a parameter may have in the checkpoint: both widen exactly to f32 and f64.
SOURCE_DTYPES = (torch.bfloat16, torch.float32)
#: The rope settings the patched rotary implements.
ROPE_TYPE = "default"


class Unsupported(Exception):
    """A checkpoint or input this family's patches do not cover."""


# ------------------------------------------------------------ the family

def layer_types(config) -> list[str]:
    return list(config.layer_types)


def expected_calls(num_layers: int, types: list[str]) -> dict:
    """Patched-function calls per forward, from the config's layer types:
    the (1+w) norm runs twice per layer (input, post-attention), on q and k
    in every full-attention layer, and once at the end; the gated norm,
    the DeltaNet forward and the chunk rule once per DeltaNet layer; the
    softmax once per full-attention layer; the rotary once per forward."""
    if len(types) != num_layers:
        raise Unsupported(f"{len(types)} layer types for {num_layers} layers")
    full, linear = types.count("full_attention"), types.count("linear_attention")
    return {"rmsnorm": 2 * num_layers + 2 * full + 1, "gated_rmsnorm": linear, "rotary": 1,
            "softmax": full, "gdn": linear, "chunk_rule": linear}


def rotary_dim(config) -> int:
    params = config.rope_parameters
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    return int(head_dim * params.get("partial_rotary_factor", 1.0))


def rope_problems(config) -> list[str]:
    params = config.rope_parameters or {}
    problems = []
    if params.get("rope_type", ROPE_TYPE) != ROPE_TYPE:
        problems.append(f"rope_type {params.get('rope_type')!r}: only {ROPE_TYPE} is patched")
    if params.get("mrope_interleaved") is not True:
        problems.append("mrope_interleaved is not true: the modeling file interleaves M-RoPE regardless")
    section = params.get("mrope_section", [11, 11, 10])
    if 2 * sum(section) != rotary_dim(config):
        problems.append(f"mrope_section {section} does not cover the rotary width {rotary_dim(config)}")
    return problems


def deltanets(model) -> list:
    return [m for m in model.modules() if isinstance(m, Q35.Qwen3_5GatedDeltaNet)]


def fast_path_problems(model) -> list[str]:
    """fla / causal_conv1d would swap in other kernels (another algorithm):
    every arm requires the torch fallback the frozen modeling file runs."""
    problems = []
    if Q35.is_fast_path_available or Q35.FusedRMSNormGated is not None:
        problems.append("fla or causal_conv1d is installed: the fast path is another algorithm")
    for m in deltanets(model):
        if m.causal_conv1d_fn is not None or m.chunk_gated_delta_rule is not Q35.torch_chunk_gated_delta_rule \
                or m.recurrent_gated_delta_rule is not Q35.torch_recurrent_gated_delta_rule \
                or type(m.norm) is not Q35.Qwen3_5RMSNormGated:
            problems.append(f"DeltaNet layer {m.layer_idx} is not on the torch fallback path")
    return problems


# --------------------------------------------------------------- loading

def load_text_model(ckpt: Path, dtype: torch.dtype, attn: str, boundary: dict, facts: dict):
    """The language model alone, from `text_config`, with its loading facts.

    `from_pretrained(dtype=...)` on the composite config builds the
    composite class and leaves every parameter bfloat16 (the trap #897
    names), so the text config is passed explicitly. The observed dtype of
    every parameter and buffer is asserted after the load, never coerced:
    there is no `.to(dtype)` that could hide a wrong load. Under A64 the
    rotary buffers, which transformers computes in float32 at construction,
    are rebuilt from the float64 formula (dtype alone would not show a
    float32-rounded value held in float64 storage)."""
    config = AutoConfig.from_pretrained(ckpt)
    tc = getattr(config, "text_config", None)
    if tc is None:
        raise Unsupported("qwen3_5 checkpoint without a text_config")
    problems = rope_problems(tc)
    if problems:
        raise Unsupported("; ".join(problems))
    tc.dtype = dtype
    model, info = AutoModelForCausalLM.from_pretrained(ckpt, config=tc, dtype=dtype, attn_implementation=attn,
                                                       output_loading_info=True)
    loaded_dtypes = observed_dtypes(model)
    rebuilt = rebuild_rotary_buffers(model) if dtype == torch.float64 else None
    after = observed_dtypes(model)
    wrong = sorted(f"{kind} {name}: {d}" for kind in ("parameters", "buffers")
                   for name, d in after[kind].items() if d != str(dtype))
    if wrong:
        raise Unsupported(f"after the text-config load, not {dtype}: {wrong[:10]}")
    problems = fast_path_problems(model)
    if problems:
        raise Unsupported("; ".join(problems))
    keys = []
    for f in sorted(Path(ckpt).glob("*.safetensors")):
        keys += oracle_loading.safetensors_header_keys(f)
    record = loading_facts(model, keys, info, boundary, facts, bool(getattr(config, "tie_word_embeddings", False)))
    record["observed_dtypes"] = {"as_loaded": dtype_counts(loaded_dtypes), "asserted": dtype_counts(after),
                                 "required": str(dtype)}
    record["rotary_buffers_rebuilt_f64"] = rebuilt
    return model.eval(), record


def observed_dtypes(model) -> dict:
    return {"parameters": {n: str(p.dtype) for n, p in model.named_parameters()},
            "buffers": {n: str(b.dtype) for n, b in model.named_buffers() if b.is_floating_point()}}


def dtype_counts(observed: dict) -> dict:
    return {kind: dict(collections.Counter(v.values())) for kind, v in observed.items()}


def f64_inv_freq(config) -> torch.Tensor:
    dim = rotary_dim(config)
    return 1.0 / (config.rope_parameters["rope_theta"] ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))


def rebuild_rotary_buffers(model) -> dict:
    rot = model.model.rotary_emb
    before = {n: str(getattr(rot, n).dtype) for n in ("inv_freq", "original_inv_freq")}
    exact = f64_inv_freq(model.config)
    rot.inv_freq = exact.clone()
    rot.original_inv_freq = exact.clone()
    return {"buffers": sorted(before), "dtype_before": before, "formula": "1 / theta^(arange(0, rotary_dim, 2) / "
            "rotary_dim) in float64"}


def checkpoint_key(name: str, facts: dict) -> str | None:
    """The checkpoint tensor a text-model parameter is loaded from."""
    prefix = facts["text_prefix"]
    if name == "lm_head.weight":
        return prefix + "embed_tokens.weight"  # tied
    return prefix + name[len("model."):] if name.startswith("model.") else None


def loading_facts(model, checkpoint_keys: list[str], info: dict, boundary: dict, facts: dict, tied: bool) -> dict:
    """Every checkpoint tensor is accounted for: loaded (under `text_prefix`,
    onto a model key) or excluded (under one of the family's
    `excluded_prefixes`). Refused (gate `fail`):
    - a checkpoint tensor neither loaded nor under an exclusion;
    - a text-prefixed tensor the model does not load (it would be skipped);
    - an excluded tensor inside LARQL's text-generation closure;
    - a loaded tensor outside LARQL's closure;
    - a model parameter with no checkpoint source but the tied head;
    - transformers reporting a missing key other than the tied head, a
      mismatched shape, or an unexpected key that is not an excluded one.
    transformers' own `unexpected` list is not enough: the modeling file's
    `_keys_to_ignore_on_load_unexpected` drops `mtp.*` and `model.visual.*`
    from it silently."""
    prefix = facts["text_prefix"]
    excluded_prefixes = facts["excluded_prefixes"]
    closure, allow = boundary["in_closure_prefixes"], boundary["allowlist"]
    under = oracle_loading.under
    state_keys = set(model.state_dict().keys())
    loaded, excluded, unaccounted, skipped = {}, {}, [], []
    for key in checkpoint_keys:
        if key.startswith(prefix):
            target = "model." + key[len(prefix):]
            (loaded.__setitem__(key, target) if target in state_keys else skipped.append(key))
            continue
        reason = next((p for p in excluded_prefixes if under(key, [p])), None)
        (excluded.__setitem__(key, reason) if reason else unaccounted.append(key))
    allowed_missing = {oracle_loading.TIED_HEAD_KEY} if tied else set()
    unsourced = sorted(state_keys - set(loaded.values()) - allowed_missing)
    head_shared = model.lm_head.weight is model.model.embed_tokens.weight
    binding = {k: ("larql_in_closure" if under(k, closure) else "larql_allowlist" if under(k, allow)
                   else "larql_no_component") for k in excluded}
    problems = []
    if unaccounted:
        problems.append(f"{len(unaccounted)} checkpoint tensor(s) neither loaded nor excluded: {unaccounted[:5]}")
    if skipped:
        problems.append(f"{len(skipped)} text-model tensor(s) in the checkpoint not loaded: {skipped[:5]}")
    inside = [k for k, b in binding.items() if b == "larql_in_closure"]
    if inside:
        problems.append(f"{len(inside)} excluded tensor(s) inside LARQL's text-generation closure: {inside[:5]}")
    outside = sorted(k for k in loaded if not under(k, closure))
    if outside:
        problems.append(f"{len(outside)} loaded tensor(s) outside LARQL's closure: {outside[:5]}")
    if unsourced:
        problems.append(f"{len(unsourced)} model key(s) with no checkpoint source: {unsourced[:5]}")
    if tied and not head_shared:
        problems.append("tie_word_embeddings, but lm_head does not share the embedding")
    bad_missing = sorted(set(info.get("missing_keys", [])) - allowed_missing)
    if bad_missing:
        problems.append(f"transformers reports {len(bad_missing)} missing key(s): {bad_missing[:5]}")
    if info.get("mismatched_keys"):
        problems.append(f"{len(info['mismatched_keys'])} shape-mismatched key(s)")
    odd = sorted(set(info.get("unexpected_keys", [])) - set(excluded))
    if odd:
        problems.append(f"transformers reports {len(odd)} unexpected key(s) that are not excluded: {odd[:5]}")
    by_reason: dict = {}
    for key, reason in excluded.items():
        entry = by_reason.setdefault(reason, {"reason": excluded_prefixes[reason], "tensors": 0,
                                              "larql_binding": collections.Counter()})
        entry["tensors"] += 1
        entry["larql_binding"][binding[key]] += 1
    return {
        "transformers_class": type(model).__name__,
        "loader": "text_config (composite checkpoint)",
        "key_mapping": {prefix: "model."},
        "checkpoint_tensors": len(checkpoint_keys),
        "expected_parameter_keys": len(state_keys),
        "loaded": {"tensors": len(loaded), "by_prefix": oracle_loading.group_by_prefix(loaded, depth=3)},
        "excluded": {p: {**e, "larql_binding": dict(e["larql_binding"])} for p, e in sorted(by_reason.items())},
        "excluded_tensors": sorted(excluded),
        "tie_word_embeddings": tied, "tied_head_shares_embedding": head_shared,
        "missing": sorted(info.get("missing_keys", [])), "unexpected": sorted(info.get("unexpected_keys", [])),
        "mismatched": sorted(str(m) for m in info.get("mismatched_keys", [])),
        "capability_boundary": boundary,
        "gate": "pass" if not problems else "fail", "gate_problems": problems,
    }


def audit_parameters(model, ckpt: Path, facts: dict) -> dict:
    """A64 audit part 1 for the family: every parameter float64 and equal to
    its checkpoint tensor (BF16 or F32 at source) widened exactly; every
    floating buffer float64; the rotary buffers equal the float64 formula."""
    stored = {}
    for f in sorted(Path(ckpt).glob("*.safetensors")):
        stored.update(load_file(str(f)))
    problems, checked, sources, f32_sources = [], 0, collections.Counter(), []
    params = dict(model.named_parameters(remove_duplicate=False))
    for name, p in params.items():
        if p.dtype != torch.float64:
            problems.append(f"{name}: {p.dtype}")
            continue
        key = checkpoint_key(name, facts)
        if key not in stored:
            problems.append(f"{name}: not in the checkpoint (as {key})")
            continue
        src = stored[key]
        if src.dtype not in SOURCE_DTYPES:
            problems.append(f"{name}: checkpoint dtype {src.dtype}, expected one of {SOURCE_DTYPES}")
            continue
        sources[str(src.dtype)] += 1
        if src.dtype == torch.float32:
            f32_sources.append(name)
        if not torch.equal(p.detach(), src.to(torch.float64)):
            problems.append(f"{name}: differs from the checkpoint widened exactly")
        checked += 1
    for name, b in model.named_buffers():
        if b.is_floating_point() and b.dtype != torch.float64:
            problems.append(f"buffer {name}: {b.dtype}")
    exact = f64_inv_freq(model.config)
    rot = model.model.rotary_emb
    for name in ("inv_freq", "original_inv_freq"):
        if not torch.equal(getattr(rot, name), exact):
            problems.append(f"buffer rotary_emb.{name} is not the float64 formula")
    return {"pass": not problems, "problems": problems[:50], "parameters_checked": checked,
            "parameters": len(params), "checkpoint_source_dtypes": dict(sources),
            "f32_source_parameters": sorted({n.split(".", 3)[-1] for n in f32_sources}),
            "f32_source_count": len(f32_sources)}


# ------------------------------------------------------- the chunk rule

def chunk_rule(on_result=None):
    """transformers' `torch_chunk_gated_delta_rule` with the float32 casts
    removed, dtype-preserving. `on_result(inputs, out, states)` (the audit)
    sees the call's inputs, its output and the state after every chunk.
    The final state is returned only when asked for, as transformers does."""

    def rule(query, key, value, g, beta, chunk_size=CHUNK, initial_state=None, output_final_state=False,
             use_qk_l2norm_in_kernel=False):
        inputs = (query, key, value, g, beta, use_qk_l2norm_in_kernel, initial_state)
        initial_dtype = query.dtype
        if use_qk_l2norm_in_kernel:
            query = Q35.l2norm(query, dim=-1, eps=L2NORM_EPS)
            key = Q35.l2norm(key, dim=-1, eps=L2NORM_EPS)
        query, key, value, beta, g = [x.transpose(1, 2).contiguous() for x in (query, key, value, beta, g)]

        batch_size, num_heads, sequence_length, k_head_dim = key.shape
        v_head_dim = value.shape[-1]
        pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
        query = F.pad(query, (0, 0, 0, pad_size))
        key = F.pad(key, (0, 0, 0, pad_size))
        value = F.pad(value, (0, 0, 0, pad_size))
        beta = F.pad(beta, (0, pad_size))
        g = F.pad(g, (0, pad_size))
        total_sequence_length = sequence_length + pad_size
        scale = 1 / (query.shape[-1] ** 0.5)
        query = query * scale

        v_beta = value * beta.unsqueeze(-1)
        k_beta = key * beta.unsqueeze(-1)
        query, key, value, k_beta, v_beta = [
            x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
        ]
        g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
        mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

        g = g.cumsum(dim=-1)
        decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp()).tril()
        attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
        for i in range(1, chunk_size):
            row = attn[..., i, :i].clone()
            sub = attn[..., :i, :i].clone()
            attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
        attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
        value = attn @ v_beta
        k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
        last_recurrent_state = (
            torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
            if initial_state is None
            else initial_state.to(value)
        )
        core_attn_out = torch.zeros_like(value)
        mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)
        states = []

        for i in range(0, total_sequence_length // chunk_size):
            q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
            attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
            v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
            v_new = v_i - v_prime
            attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
            core_attn_out[:, :, i] = attn_inter + attn @ v_new
            last_recurrent_state = (
                last_recurrent_state * g[:, :, i, -1, None, None].exp()
                + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
            )
            states.append(last_recurrent_state)

        core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
        core_attn_out = core_attn_out[:, :, :sequence_length]
        core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
        if on_result is not None:
            on_result(inputs, core_attn_out, states, chunk_size)
        return core_attn_out, (last_recurrent_state if output_final_state else None)

    return rule


def chunk_ends(sequence_length: int, chunk_size: int = CHUNK) -> list[int]:
    """The positions whose state the chunked rule holds after each chunk:
    64k - 1 for each full chunk, and the last position for a partial one
    (padding has beta = 0 and g = 0, so it leaves the state unchanged)."""
    ends = list(range(chunk_size - 1, sequence_length, chunk_size))
    if not ends or ends[-1] != sequence_length - 1:
        ends.append(sequence_length - 1)
    return ends


def step_witness(query, key, value, g, beta, use_qk_l2norm: bool, record_at: list[int]):
    """The gated delta recurrence, one token at a time, in numpy float64.

    Inputs as the rule receives them: q, k `[b, T, H, d_k]`, v
    `[b, T, H, d_v]`, g, beta `[b, T, H]` (numpy or torch). Returns the
    outputs `[b, T, H, d_v]` and the states `[b, H, d_k, d_v]` after each
    position in `record_at`. Written from the published update (module
    docstring), not from transformers' code:
        S_t = alpha_t (I - beta_t k_t k_t^T) S_{t-1} + beta_t k_t v_t^T,  o_t = S_t^T q_t."""
    q, k, v, g, beta = (np.asarray(t.detach().numpy() if isinstance(t, torch.Tensor) else t, dtype=np.float64)
                        for t in (query, key, value, g, beta))
    if use_qk_l2norm:
        q = q / np.sqrt(np.sum(q * q, axis=-1, keepdims=True) + L2NORM_EPS)
        k = k / np.sqrt(np.sum(k * k, axis=-1, keepdims=True) + L2NORM_EPS)
    q = q * (q.shape[-1] ** -0.5)
    b, T, H, dk = k.shape
    S = np.zeros((b, H, dk, v.shape[-1]))
    out = np.empty(v.shape)
    states, wanted = {}, set(record_at)
    for t in range(T):
        alpha = np.exp(g[:, t])[..., None, None]                  # [b, H, 1, 1]
        bt = beta[:, t][..., None, None]
        kt, vt = k[:, t], v[:, t]                                  # [b, H, d]
        kS = np.einsum("bhk,bhkv->bhv", kt, S)                     # k_t^T S_{t-1}
        S = alpha * (S - bt * kt[..., :, None] * kS[..., None, :]) + bt * kt[..., :, None] * vt[..., None, :]
        out[:, t] = np.einsum("bhk,bhkv->bhv", q[:, t], S)
        if t in wanted:
            states[t] = S.copy()
    return out, [states[t] for t in record_at]


def position_rel(got: np.ndarray, want: np.ndarray, axis: int = 1) -> float:
    """The worst per-position relative error (Frobenius norm over all other
    axes). A position whose witness is exactly zero is compared absolutely."""
    diff = np.moveaxis(got - want, axis, 0).reshape(got.shape[axis], -1)
    ref = np.moveaxis(want, axis, 0).reshape(want.shape[axis], -1)
    num, den = np.linalg.norm(diff, axis=1), np.linalg.norm(ref, axis=1)
    return float(np.max(num / np.where(den > 0, den, 1.0)))


def compare_with_witness(inputs, out, states, chunk_size: int) -> dict:
    """The rule's output and chunk-end states against `step_witness`."""
    query, key, value, g, beta, l2, initial = inputs
    if initial is not None:
        raise Unsupported("the witness covers a prefill from the zero state only")
    T = query.shape[1]
    ends = chunk_ends(T, chunk_size)
    w_out, w_states = step_witness(query, key, value, g, beta, l2, ends)
    got = out.detach().numpy()
    # Per head (axis 1 of [b, H, d_k, d_v]) for the states, per position for the outputs.
    state_rel = max(position_rel(s.detach().numpy(), w, axis=1) for s, w in zip(states, w_states))
    whole = float(np.linalg.norm(got - w_out) / max(np.linalg.norm(w_out), 1e-300))
    return {"output": position_rel(got, w_out), "state": state_rel, "output_whole_tensor": whole,
            "positions": T, "chunk_ends": ends, "states_compared": len(ends)}


# ----------------------------------------------------- the DeltaNet forward

def gdn_forward(on_gates=None):
    """`Qwen3_5GatedDeltaNet.forward` on the no-cache, unmasked prefill path
    (the only one A64 runs), line for line, except the decay
    `g = -exp(A_log) * softplus(a + dt_bias)` is computed in the model dtype
    (transformers: `A_log.float()`, `a.float()`). `on_gates` (the audit)
    sees the conv input and output, a, b, beta and g."""

    def forward(self, hidden_states, cache_params=None, attention_mask=None):
        if cache_params is not None or attention_mask is not None:
            raise Unsupported("the patched DeltaNet forward covers the no-cache, unmasked prefill only")
        if self.causal_conv1d_fn is not None:
            raise Unsupported("causal_conv1d is installed")
        hidden_states = Q35.apply_mask_to_padding_states(hidden_states, attention_mask)
        batch_size, seq_len, _ = hidden_states.shape

        mixed_qkv = self.in_proj_qkv(hidden_states)
        mixed_qkv = mixed_qkv.transpose(1, 2)
        z = self.in_proj_z(hidden_states)
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        conv_in = mixed_qkv
        mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])
        conv_out = mixed_qkv

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.exp() * F.softplus(a + self.dt_bias)
        if on_gates is not None:
            on_gates(self, conv_in, conv_out, a, b, beta, g)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        core_attn_out, _ = self.chunk_gated_delta_rule(query, key, value, g=g, beta=beta, initial_state=None,
                                                       output_final_state=False, use_qk_l2norm_in_kernel=True)
        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        return self.out_proj(core_attn_out)

    return forward


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def _softplus(x: np.ndarray) -> np.ndarray:
    """F.softplus(beta=1, threshold=20): x above the threshold, else log(1 + e^x)."""
    return np.where(x > SOFTPLUS_THRESHOLD, x, np.log1p(np.exp(np.minimum(x, SOFTPLUS_THRESHOLD))))


def gates_witness(module, conv_in, a, b) -> dict:
    """numpy float64: the depthwise causal conv + SiLU over the projected
    q/k/v channels, beta = sigmoid(b), g = -exp(A_log) * softplus(a + dt_bias)."""
    x = conv_in.detach().numpy()                                    # [b, C, T]
    w = module.conv1d.weight.detach().numpy()[:, 0, :]               # [C, K]
    K, T = w.shape[1], x.shape[-1]
    xp = np.concatenate([np.zeros(x.shape[:-1] + (K - 1,)), x], axis=-1)
    conv = sum(w[None, :, j, None] * xp[..., j:j + T] for j in range(K))
    if module.conv1d.bias is not None:
        conv = conv + module.conv1d.bias.detach().numpy()[None, :, None]
    an, bn = a.detach().numpy(), b.detach().numpy()
    return {"conv": _silu(conv), "beta": 1.0 / (1.0 + np.exp(-bn)),
            "g": -np.exp(module.A_log.detach().numpy()) * _softplus(an + module.dt_bias.detach().numpy())}


# --------------------------------------------------------- the norms, rotary

def gated_norm_forward(audit, names, layer_of, eps_of):
    """`Qwen3_5RMSNormGated.forward` without its float32 casts:
    w * x / sqrt(mean(x^2) + eps) * silu(gate), in the order transformers
    multiplies."""

    def forward(self, hidden_states, gate=None):
        name = names.get(id(self), "")
        audit.check(f"gated_rmsnorm in {name}", hidden_states, gate, self.weight)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        normed = hidden_states * torch.rsqrt(variance + eps_of(self))
        out = self.weight * normed
        out = out * F.silu(gate)
        audit.check(f"gated_rmsnorm out {name}", out)
        audit.count("gated_rmsnorm", layer_of(name))
        x, z = hidden_states.detach().numpy(), gate.detach().numpy()
        want = self.weight.detach().numpy() * (x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps_of(self))) \
            * _silu(z)
        audit.witness_once("gated_rmsnorm", out, want)
        return out

    return forward


def text_positions(position_ids: torch.Tensor) -> torch.Tensor:
    """The M-RoPE axes `[3, b, T]` of a text-only input are identical; a
    vision input (distinct axes) is out of scope and refused."""
    if position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
    if position_ids.shape[0] != 3 or not all(torch.equal(position_ids[0], position_ids[i]) for i in (1, 2)):
        raise Unsupported("M-RoPE position axes differ (a vision input): the oracle is text-only")
    return position_ids


def mrope_axis(rotary_half: int, section: list[int]) -> np.ndarray:
    """Which M-RoPE axis (0 = T, 1 = H, 2 = W) feeds each frequency index,
    for the interleaved layout [T H W T H W ... T T]: index j is H when
    j % 3 == 1 and j < 3 * section[1], W when j % 3 == 2 and j < 3 * section[2],
    T otherwise."""
    j = np.arange(rotary_half)
    axis = np.zeros(rotary_half, dtype=int)
    axis[(j % 3 == 1) & (j < 3 * section[1])] = 1
    axis[(j % 3 == 2) & (j < 3 * section[2])] = 2
    return axis


def rotary_forward(audit):
    """`Qwen3_5TextRotaryEmbedding.forward` in float64: inv_freq from the f64
    formula over the partial rotary width, the angles per M-RoPE axis, the
    modeling file's own interleave (an exact copy), then cos and sin."""

    def forward(self, x, position_ids):
        position_ids = text_positions(position_ids)
        config = self.config
        inv_freq = f64_inv_freq(config)
        freqs = position_ids[..., None].to(torch.float64) * inv_freq           # [3, b, T, rotary/2]
        freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos() * self.attention_scaling, emb.sin() * self.attention_scaling
        audit.check("rotary", inv_freq, freqs, cos, sin)
        audit.count("rotary", "full_attention")
        if self.attention_scaling != 1.0:
            audit.problems.append(f"rotary attention_scaling {self.attention_scaling} for a default rope")
        p = position_ids.numpy().astype(np.float64)[:, 0]                      # [3, T]
        dim = rotary_dim(config)
        f = 1.0 / (float(config.rope_parameters["rope_theta"]) ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
        axis = mrope_axis(dim // 2, list(self.mrope_section))
        ang = p[axis, :].T * f[None, :]                                        # [T, rotary/2]
        want = np.concatenate([np.cos(ang), np.cos(ang), np.sin(ang), np.sin(ang)], axis=-1)
        audit.witness_once("rotary", torch.cat((cos[0], sin[0]), dim=-1), want)
        return cos, sin

    return forward


# -------------------------------------------------------------------- A32rec

def recurrent_rule_for_prefill(counter: dict):
    """A32rec: transformers' own float32 `torch_recurrent_gated_delta_rule`,
    unmodified, called for the prefill in place of the chunked rule (the
    same token-recurrent algorithm LARQL runs). `counter` counts the calls."""

    def rule(query, key, value, g, beta, chunk_size=CHUNK, initial_state=None, output_final_state=False,
             use_qk_l2norm_in_kernel=False):
        counter["calls"] = counter.get("calls", 0) + 1
        return Q35.torch_recurrent_gated_delta_rule(query, key, value, g=g, beta=beta, initial_state=initial_state,
                                                    output_final_state=output_final_state,
                                                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel)

    return rule


def recurrent_rule_refused(*args, **kwargs):
    raise Unsupported("the recurrent (decode) rule ran during a no-cache prefill")


def config_json(ckpt: Path) -> dict:
    return json.loads((Path(ckpt) / "config.json").read_text())
