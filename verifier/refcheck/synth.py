"""Synthetic container writer, for testing the reader only.

Written from specs/vindex3-container/SPEC.md revision 2, sections 4 to 10 (framing, index.json,
system_graph.json, tensor layouts, name mapping, role policy). It shares no code with the reader
(`refcheck.container`) or with the hand-built container in tests/_mini.py.

`write_container` takes a dict of Hugging Face tensor keys to float32 arrays, rounds them to BF16
(the stand-in for the upstream checkpoint), encodes the eligible projections as Q8_0, and writes a
container directory. `Hooks` injects faults at each layer of the format so that every reader rule
can be tested in isolation; `FAULTS` is the catalogue used by the tests.

The Q8_0 encoder (`ref_quantize_q8_0`) follows ggml `quantize_row_q8_0_ref`. It exists so that the
synthetic Q8_0 blocks are realistic; the checker's E1 machinery will build on it later.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import numconv, q8_0

SEGMENT_SCHEMA = 1
GRAPH_SCHEMA = 6
INDEX_VERSION = 4

CODEC = {
    "family": "Q8_0",
    "revision": 1,
    "group_elems": 32,
    "element": "ggml-q8_0",
    "group_scale": "in-block",
    "tensor_scale": "in-block",
    "layout": "ggml-block-34",
}

# section 10.2 / 9.1: HF prefix of each logical object
OBJECT_PREFIX = {
    "target.decoder_stack": ("model.layers", "decoder_stack"),
    "target.embedding": ("model.embed_tokens", "embedding"),
    "target.final_norm": ("model.norm", "final_norm"),
    "target.output_head": ("lm_head", "output_head"),
}


# ---------------------------------------------------------------------------------------
# value encoders


def bf16_bytes(a: np.ndarray) -> bytes:
    """float32 -> BF16 bytes, round to nearest even, little-endian. NaN stays NaN."""
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32).reshape(-1)
    nan = (u & 0x7FFFFFFF) > 0x7F800000
    rounded = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    h = np.where(nan, (u >> 16) | 0x40, rounded).astype(np.uint16)
    out = np.empty(h.size * 2, dtype=np.uint8)
    out[0::2] = h & 0xFF
    out[1::2] = h >> 8
    return out.tobytes()


def bf16_widen(data: bytes) -> np.ndarray:
    raw = np.frombuffer(data, dtype=np.uint8)
    u = raw[0::2].astype(np.uint32) | (raw[1::2].astype(np.uint32) << 8)
    return (u << 16).view(np.float32)


def _roundf(x: np.ndarray) -> np.ndarray:
    """C roundf: half away from zero (not numpy's half-to-even)."""
    t = np.trunc(x)
    return t + np.where(np.abs(x - t) >= 0.5, np.sign(x), 0).astype(np.float32)


def ref_quantize_q8_0(x: np.ndarray) -> bytes:
    """ggml quantize_row_q8_0_ref: d = amax/127 (f32), q = roundf(x * (1/d)), d stored as RNE f16."""
    x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1, 32)
    amax = np.abs(x).max(axis=1)
    d = (amax / np.float32(127)).astype(np.float32)
    with np.errstate(divide="ignore"):
        idv = np.where(d != 0, np.float32(1) / d, np.float32(0)).astype(np.float32)
    q = _roundf((x * idv[:, None]).astype(np.float32)).astype(np.int8)
    out = np.empty((x.shape[0], 34), dtype=np.uint8)
    dh = d.astype(np.float16).view(np.uint16)
    out[:, 0] = dh & 0xFF
    out[:, 1] = dh >> 8
    out[:, 2:] = q.view(np.uint8)
    return out.tobytes()


def q8_block(d_bits: int, qs) -> bytes:
    qs = list(qs) + [0] * (32 - len(qs))
    return struct.pack("<H", d_bits) + struct.pack("<32b", *qs)


# ---------------------------------------------------------------------------------------
# model and plan


def tiny_model(rng: np.random.Generator, layers: int = 12, hidden: int = 64, vocab: int = 40) -> dict[str, np.ndarray]:
    """A small Qwen3-shaped set of HF tensors: 11 per layer plus embed, norm and a tied lm_head."""
    q_out, kv, inter, hd = 2 * hidden, hidden, 3 * hidden, 32
    w = lambda *s: (rng.standard_normal(s) * 0.05).astype(np.float32)  # noqa: E731
    t: dict[str, np.ndarray] = {}
    for L in range(layers):
        p = f"model.layers.{L}."
        t[p + "input_layernorm.weight"] = 1 + w(hidden)
        t[p + "post_attention_layernorm.weight"] = 1 + w(hidden)
        t[p + "self_attn.q_proj.weight"] = w(q_out, hidden)
        t[p + "self_attn.k_proj.weight"] = w(kv, hidden)
        t[p + "self_attn.v_proj.weight"] = w(kv, hidden)
        t[p + "self_attn.o_proj.weight"] = w(hidden, q_out)
        t[p + "self_attn.q_norm.weight"] = 1 + w(hd)
        t[p + "self_attn.k_norm.weight"] = 1 + w(hd)
        t[p + "mlp.gate_proj.weight"] = w(inter, hidden)
        t[p + "mlp.up_proj.weight"] = w(inter, hidden)
        t[p + "mlp.down_proj.weight"] = w(hidden, inter)
    emb = w(vocab, hidden)
    t["model.embed_tokens.weight"] = emb
    t["model.norm.weight"] = 1 + w(hidden)
    t["lm_head.weight"] = emb.copy()
    return t


def execution_surface(
    layers: int,
    hidden: int,
    inter: int,
    nq: int,
    nkv: int,
    hd: int,
    vocab: int,
    rms_norm_eps: float = 1e-6,
    rope_theta: float = 1000000.0,
    activation: str = "silu",
    attention_bias: bool = False,
    head_reuses_embedding: bool = True,
    context_length: int | None = None,
) -> tuple[dict, list]:
    """SPEC revision 4 section 17: the `execution` object and the `attention` array of the one component, as the
    section describes the writer's output for this profile. Returns (execution, attention). Numbers are Python
    floats, which `json` writes in shortest round-trip form, as the section says `serde_json` does.

    Encodings follow 17.3 and 17.4: the epsilon is `f64(f32(rms_norm_eps))` (numpy's float32 here, not the checker's
    own converter), the score scale is the exact 1/sqrt(head_dim) rounded once to binary64 (D8). Callers build a fault by editing the result."""
    eps = float(np.float32(rms_norm_eps))
    x = {
        "attention": {
            "num_q_heads": nq, "num_kv_heads": nkv, "head_dim": hd, "score_scale": numconv.inv_sqrt_binary64(hd),
            "attention_bias": attention_bias, "qk_norm_scope": "per_head", "qk_norm_weight_offset": 0.0,
            "parameter_free_qk_norm": {"q": False, "k": False, "v": False},
        },
        "ffn": {"intermediate_size": inter, "ffn_type": "gated", "activation": activation},
        "norm": {
            "pre": {"kind": "rms_norm", "eps": eps, "weight_offset": 0.0},
            "final_norm": {"kind": "rms_norm", "eps": eps, "weight_offset": 0.0},
            "placement": "pre_only",
        },
        "head": {"vocab_size": vocab, "head_reuses_embedding": head_reuses_embedding},
        "residual_topology": {"kind": "single_stream"},
    }
    if context_length is not None:
        x["context_length"] = context_length
    attn = [
        {
            "operator": "softmax", "span": "full", "position": {"kind": "rope", "theta": float(rope_theta)},
            "geometry": {"head_dim": hd, "num_kv_heads": nkv},
        }
        for _ in range(layers)
    ]
    return x, attn


def default_policy(hf_key: str, shape: tuple[int, ...]) -> bool:
    """Section 11 default: decoder-linear projections (2-D, under model.layers) are compiled to Q8_0."""
    return hf_key.startswith("model.layers.") and len(shape) == 2 and hf_key.endswith("_proj.weight")


@dataclass
class T:
    """One tensor of a segment under construction."""

    name: str
    dtype: str
    shape: list[int]
    data: bytes


@dataclass
class Hooks:
    """Fault injection points. All optional. See FAULTS for the catalogue."""

    tensors: Callable[[str, list[T]], list[T]] | None = None  # rep id, table before serialising
    header: Callable[[str, dict], dict] | None = None  # rep id, header dict before serialising
    segment: Callable[[str, bytes], bytes] | None = None  # rep id, final segment bytes
    index: Callable[[dict], None] | None = None
    graph: Callable[[dict], None] | None = None
    files: Callable[[str], None] | None = None  # after everything is written, given the root
    seal: bool = True  # False: digests describe the bytes as they were before `segment` ran
    pad_extra: int = 0  # extra header padding spaces (still valid JSON)


@dataclass
class Written:
    root: str
    source: dict[str, np.ndarray]  # HF key -> BF16-rounded float32 values (the stand-in upstream)
    q8_keys: list[str]
    q8_bytes: dict[str, bytes]  # HF key -> the Q8_0 block bytes written


def _frame(rep_id: str, tensors: list[T], hooks: Hooks) -> tuple[bytes, int]:
    table, off = [], 0
    for t in tensors:
        table.append({"name": t.name, "dtype": t.dtype, "shape": list(t.shape), "offset": off, "len": len(t.data)})
        off += len(t.data)
    header = {"schema": SEGMENT_SCHEMA, "representation": rep_id, "tensors": table}
    if hooks.header:
        header = hooks.header(rep_id, header)
    js = json.dumps(header, separators=(",", ":")).encode()
    pad = (16 - (8 + len(js)) % 16) % 16 + hooks.pad_extra
    hdr = js + b" " * pad
    payload = b"".join(t.data for t in tensors)
    return struct.pack("<Q", len(hdr)) + hdr + payload, 8 + len(hdr)


def write_container(
    root: str,
    source: dict[str, np.ndarray],
    hooks: Hooks | None = None,
    policy: Callable[[str, tuple[int, ...]], bool] = default_policy,
    q8_override: dict[str, bytes] | None = None,
    surface: tuple[dict, list] | None = None,
) -> Written:
    """Write a container for `source` (HF key -> float32 array) under `root` and return what was written.
    `surface` is an optional (execution, attention) pair from `execution_surface`; without it the component carries only
    the small head object of revision 2."""
    hooks = hooks or Hooks()
    q8_override = q8_override or {}
    os.makedirs(os.path.join(root, "segments"), exist_ok=True)

    # section 10.1: split keys into objects by longest covering prefix; relative name = key minus prefix and dot
    by_obj: dict[str, list[tuple[str, str]]] = {o: [] for o in OBJECT_PREFIX}
    for key in source:
        owner = None
        for oid, (pre, _) in OBJECT_PREFIX.items():
            if key == pre or key.startswith(pre + "."):
                if owner is None or len(pre) > len(OBJECT_PREFIX[owner][0]):
                    owner = oid
        if owner is None:
            raise ValueError(f"no object covers {key}")
        pre = OBJECT_PREFIX[owner][0]
        by_obj[owner].append((key, key[len(pre) + 1 :]))

    src_bf16: dict[str, np.ndarray] = {}
    bf: dict[str, bytes] = {}
    for key, a in source.items():
        b = bf16_bytes(a)
        bf[key] = b
        src_bf16[key] = bf16_widen(b).reshape(a.shape)

    q8_keys, q8_bytes = [], {}
    segs: dict[str, tuple[str, list[T]]] = {}  # rep id -> (relative path, tensors)
    for oid, items in by_obj.items():
        if not items:
            continue
        items.sort(key=lambda kn: kn[1].encode())  # section 4.3: byte order of the relative name
        canon = [T(n, "BF16", list(source[k].shape), bf[k]) for k, n in items]
        segs[f"{oid}@BF16"] = (f"segments/{oid}.bin", canon)
        compiled = []
        for k, n in items:
            shp = tuple(source[k].shape)
            if (k in q8_override or policy(k, shp)) and shp and shp[-1] % 32 == 0:
                data = q8_override.get(k) or ref_quantize_q8_0(src_bf16[k])
                compiled.append(T(n, "Q8_0", list(shp), data))
                q8_keys.append(k)
                q8_bytes[k] = data
            else:
                compiled.append(T(n, "BF16", list(shp), bf[k]))
        if any(t.dtype == "Q8_0" for t in compiled):
            segs[f"{oid}@Q8_0"] = (f"segments/{oid}@Q8_0.bin", compiled)

    reps = {}
    for rid, (rel, tensors) in segs.items():
        if hooks.tensors:
            tensors = hooks.tensors(rid, list(tensors))
        built, p = _frame(rid, tensors, hooks)
        final = hooks.segment(rid, built) if hooks.segment else built
        if hooks.seal:  # digests describe the bytes as written, faults included
            sealed, sp = final, p
        else:  # digests describe the unfaulted segment, so the fault is visible to a digest check
            sealed, sp = _frame(rid, tensors, Hooks())
        with open(os.path.join(root, rel), "wb") as f:
            f.write(final)
        oid, enc = rid.split("@")
        e = {
            "object": oid,
            "encoding": enc,
            "segment": rel,
            "tensor_count": len(tensors),
            "payload_bytes": sum(len(t.data) for t in tensors),
            "payload_sha256": hashlib.sha256(sealed[sp:]).hexdigest(),
            "segment_sha256": hashlib.sha256(sealed).hexdigest(),
        }
        if enc == "Q8_0":
            e["compiled_from"] = f"{oid}@BF16"
            e["codec"] = dict(CODEC)
            e["encoder"] = {"algorithm": "kquant-larql-native", "revision": 1}
        reps[rid] = e
    for rid, e in reps.items():
        if e["encoding"] == "Q8_0":
            e["source_representation_digest"] = reps[e["compiled_from"]]["payload_sha256"]

    num_layers = 1 + max((int(k.split(".")[2]) for k in source if k.startswith("model.layers.")), default=-1)
    hidden = int(source["model.norm.weight"].shape[0]) if "model.norm.weight" in source else 0
    index = {
        "version": INDEX_VERSION,
        "model": "synthetic",
        "family": "qwen3",
        "hidden_size": hidden,
        "num_layers": num_layers,
        "system_graph": "system_graph.json",
        "representations": reps,
        "profiles": [{"name": "exact"}],
        "segments": {rel[:-4]: 1 for rel, _ in segs.values()},
        "precision_map": {"name": "r0-uniform", "encoding": "Q8_0", "roles": ["decoder-linear"]},
    }
    objects = []
    for oid in sorted(by_obj):
        if not by_obj[oid]:
            continue
        pre, kind = OBJECT_PREFIX[oid]
        canon_rep = reps[f"{oid}@BF16"]
        r = [{"encoding": "BF16", "fidelity": "canonical"}]
        if f"{oid}@Q8_0" in reps:
            r.append({"encoding": "Q8_0", "fidelity": "approximate"})
        objects.append(
            {
                "id": oid,
                "component": "target",
                "kind": kind,
                "source_bindings": [
                    {"artifact": "synthetic", "tensor_prefix": pre, "tensors": canon_rep["tensor_count"], "bytes": canon_rep["payload_bytes"]}
                ],
                "representations": r,
            }
        )
    graph = {
        "schema": GRAPH_SCHEMA,
        "components": [
            {
                "id": "target",
                "role": "primary_text",
                "num_layers": num_layers,
                "hidden_size": hidden,
                "execution": {"head": {"vocab_size": int(source["lm_head.weight"].shape[0]), "head_reuses_embedding": True}}
                if "lm_head.weight" in source
                else {},
            }
        ],
        "objects": objects,
        "edges": [],
    }
    if surface is not None:
        graph["components"][0]["execution"] = json.loads(json.dumps(surface[0]))
        graph["components"][0]["attention"] = json.loads(json.dumps(surface[1]))
    if hooks.index:
        hooks.index(index)
    if hooks.graph:
        hooks.graph(graph)
    with open(os.path.join(root, "index.json"), "w") as f:
        json.dump(index, f, indent=2)
    with open(os.path.join(root, "system_graph.json"), "w") as f:
        json.dump(graph, f, indent=2)
    if hooks.files:
        hooks.files(root)
    return Written(root, src_bf16, q8_keys, q8_bytes)


# ---------------------------------------------------------------------------------------
# fault catalogue


@dataclass(frozen=True)
class Fault:
    name: str
    hooks: Hooks
    expect: str | None  # substring of the ContainerError, or None when the reader must accept
    note: str = ""


EMB = "target.embedding@BF16"
STACK_Q = "target.decoder_stack@Q8_0"
STACK_B = "target.decoder_stack@BF16"
NORM = "target.final_norm@BF16"


def _hdr(rep: str, fn: Callable[[dict], None]) -> Callable[[str, dict], dict]:
    def h(rid: str, header: dict) -> dict:
        if rid == rep:
            fn(header)
        return header

    return h


def _seg(rep: str, fn: Callable[[bytes], bytes]) -> Callable[[str, bytes], bytes]:
    return lambda rid, b: fn(b) if rid == rep else b


def _drop(path: str):
    return lambda root: os.remove(os.path.join(root, path))


def _overwrite(path: str, text: str):
    def f(root):
        with open(os.path.join(root, path), "w") as fh:
            fh.write(text)

    return f


def _first_q8(tensors: list[T]) -> T:
    return next(t for t in tensors if t.dtype == "Q8_0")


def _make_faults() -> list[Fault]:
    F = []
    add = lambda name, expect, note="", **kw: F.append(Fault(name, Hooks(**kw), expect, note))  # noqa: E731

    # --- controls: the reader must accept ---
    add("control-clean", None)
    add("control-unknown-index-key", None, index=lambda i: i.update(future_key={"a": 1}))
    add("control-unknown-rep-key", None, index=lambda i: i["representations"][EMB].update(note="x"))
    add("control-unknown-header-key", None, header=_hdr(EMB, lambda h: h.update(extra=1)))
    add("control-unknown-tensor-entry-key", None, header=_hdr(EMB, lambda h: h["tensors"][0].update(extra="y")))
    add("control-extra-header-padding", None, "padding is any JSON whitespace; digests sealed over the result", pad_extra=16)
    add("control-payload-change-resealed", None,
        "a payload edit with digests recomputed is invisible to the reader: only E1 against upstream can see it",
        segment=_seg(STACK_Q, lambda b: b[:-1] + bytes([b[-1] ^ 0x01])))
    add("control-unsorted-table", None, "warning only (AMB-004)", tensors=lambda rid, ts: list(reversed(ts)) if rid == NORM or rid == STACK_B else ts)

    # --- index.json ---
    add("index-version-3", "version", index=lambda i: i.update(version=3))
    add("index-version-5", "version", index=lambda i: i.update(version=5))
    add("index-version-string", "version", index=lambda i: i.update(version="4"))
    add("index-version-float", "version", index=lambda i: i.update(version=4.0))
    add("index-version-missing", "no version", index=lambda i: i.pop("version"))
    add("index-no-system-graph", "no system_graph", index=lambda i: i.pop("system_graph"))
    add("index-moe-manifest", "moe_manifest", index=lambda i: i.update(moe_manifest="moe_manifest.json"))
    add("index-auxiliary-references", "auxiliary_references", index=lambda i: i.update(auxiliary_references="a.json"))
    add("index-authority-derived", "authority", index=lambda i: i.update(authority="derived"))
    add("index-missing", "index.json: file is missing", files=_drop("index.json"))
    add("index-not-json", "not valid JSON", files=_overwrite("index.json", "{not json"))
    add("index-not-object", "not a JSON object", files=_overwrite("index.json", "[]"))
    add("index-graph-path-absolute", "absolute or contains", index=lambda i: i.update(system_graph="/etc/hosts"))
    add("index-graph-path-dotdot", "absolute or contains", index=lambda i: i.update(system_graph="../system_graph.json"))
    add("index-segment-path-absolute", "absolute or contains", index=lambda i: i["representations"][EMB].update(segment="/etc/hosts"))
    add("index-segment-path-dotdot", "absolute or contains", index=lambda i: i["representations"][EMB].update(segment="segments/../index.json"))
    add("index-tensor-count-wrong", "tensor_count", index=lambda i: i["representations"][EMB].update(tensor_count=2))
    add("index-payload-bytes-wrong", "payload_bytes", index=lambda i: i["representations"][EMB].update(payload_bytes=8))
    add("index-rep-field-missing", "missing or of the wrong type", index=lambda i: i["representations"][EMB].pop("payload_sha256"))
    add("index-rep-key-mismatch", "not '<object>@<encoding>'", index=lambda i: i["representations"][EMB].update(object="target.output_head"))
    add("index-rep-dropped", "has no target.final_norm@BF16", index=lambda i: i["representations"].pop(NORM))
    add("index-q8-pack-dropped", "has no target.decoder_stack@Q8_0", index=lambda i: i["representations"].pop(STACK_Q))
    add("index-digest-uppercase", "mismatch", index=lambda i: i["representations"][EMB].update(segment_sha256=i["representations"][EMB]["segment_sha256"].upper()))

    # --- codec ---
    for k, v in [("revision", 2), ("group_elems", 64), ("element", "ggml-q4_0"), ("group_scale", "tensor"), ("tensor_scale", "global"), ("layout", "ggml-block-18"), ("family", "Q4_0"), ("group_elems", 32.0), ("revision", True)]:
        add(f"codec-{k}-{v!r}", "codec", index=lambda i, k=k, v=v: i["representations"][STACK_Q]["codec"].update({k: v}))
    add("codec-missing", "codec", index=lambda i: i["representations"][STACK_Q].pop("codec"))
    add("codec-extra-field", "codec", index=lambda i: i["representations"][STACK_Q]["codec"].update(extra=1))

    # --- system_graph.json ---
    add("graph-schema-5", "schema", graph=lambda g: g.update(schema=5))
    add("graph-schema-missing", "schema", graph=lambda g: g.pop("schema"))
    add("graph-missing", "system_graph.json: file is missing", files=_drop("system_graph.json"))
    add("graph-not-object", "not a JSON object", files=_overwrite("system_graph.json", "[1]"))
    add("graph-object-missing", "absent from the graph", graph=lambda g: g["objects"].pop(0))
    add("graph-two-bindings", "exactly one source binding", graph=lambda g: g["objects"][0]["source_bindings"].append(dict(g["objects"][0]["source_bindings"][0], tensor_prefix="x")))
    add("graph-zero-bindings", "exactly one source binding", graph=lambda g: g["objects"][0].update(source_bindings=[]))
    add("graph-duplicate-object", "duplicate graph object", graph=lambda g: g["objects"].append(dict(g["objects"][0])))

    # --- segment framing ---
    add("seg-header-schema-2", "schema", header=_hdr(EMB, lambda h: h.update(schema=2)))
    add("seg-header-schema-missing", "schema", header=_hdr(EMB, lambda h: h.pop("schema")))
    add("seg-representation-mismatch", "header.representation", header=_hdr(EMB, lambda h: h.update(representation="target.embedding@Q8_0")))
    add("seg-tensors-not-array", "not an array", header=_hdr(EMB, lambda h: h.update(tensors={})))
    add("seg-shorter-than-prefix", "shorter than", segment=_seg(NORM, lambda b: b[:5]), seal=False)
    add("seg-header-len-over-256MiB", "exceeds 256 MiB", segment=_seg(NORM, lambda b: struct.pack("<Q", 268435457) + b[8:]), seal=False)
    add("seg-header-len-past-eof", "past the end", segment=_seg(NORM, lambda b: struct.pack("<Q", len(b)) + b[8:]), seal=False)
    add("seg-header-len-too-short", "not valid JSON", segment=_seg(NORM, lambda b: struct.pack("<Q", 10) + b[8:]), seal=False)
    add("seg-truncated-1", "file length", segment=_seg(EMB, lambda b: b[:-1]), seal=False)
    add("seg-trailing-byte", "file length", segment=_seg(EMB, lambda b: b + b"\0"), seal=False)
    add("seg-header-not-json", "not valid JSON", segment=_seg(NORM, lambda b: b[:8] + b"\x00" * 4 + b[12:]), seal=False)
    add("seg-tensor-name-duplicate", "duplicate tensor name", header=_hdr(STACK_B, lambda h: h["tensors"][1].update(name=h["tensors"][0]["name"])))
    add("seg-offset-gap", "contiguity", header=_hdr(STACK_Q, lambda h: h["tensors"][1].update(offset=h["tensors"][1]["offset"] + 2)))
    add("seg-offset-overlap", "contiguity", header=_hdr(STACK_Q, lambda h: h["tensors"][1].update(offset=h["tensors"][1]["offset"] - 2)))
    add("seg-offset-first-nonzero", "contiguity", header=_hdr(NORM, lambda h: h["tensors"][0].update(offset=2)))
    add("seg-len-wrong-bf16", "disagrees", header=_hdr(NORM, lambda h: h["tensors"][0].update(len=h["tensors"][0]["len"] + 2)))
    add("seg-dtype-unknown", "not BF16 or Q8_0", header=_hdr(NORM, lambda h: h["tensors"][0].update(dtype="F32")))
    add("seg-dtype-lowercase", "not BF16 or Q8_0", header=_hdr(NORM, lambda h: h["tensors"][0].update(dtype="bf16")))
    add("seg-dtype-q4k", "not BF16 or Q8_0", header=_hdr(NORM, lambda h: h["tensors"][0].update(dtype="Q4_K")))
    add("seg-shape-negative", "invalid shape", header=_hdr(NORM, lambda h: h["tensors"][0].update(shape=[-1])))
    add("seg-shape-float", "invalid shape", header=_hdr(NORM, lambda h: h["tensors"][0].update(shape=[64.0])))
    add("seg-shape-vs-len", "disagrees", header=_hdr(NORM, lambda h: h["tensors"][0].update(shape=[1, 1])))
    add("seg-q8-row-not-multiple-of-32", "multiple of 32", header=_hdr(STACK_Q, lambda h: next(t for t in h["tensors"] if t["dtype"] == "Q8_0").update(shape=[2, 33], len=2 * 34)))
    add("seg-q8-len-wrong", "disagrees", header=_hdr(STACK_Q, lambda h: next(t for t in h["tensors"] if t["dtype"] == "Q8_0").update(len=next(t for t in h["tensors"] if t["dtype"] == "Q8_0")["len"] - 34)))
    add("control-q8-shape-transposed", None,
        "[128,64] and [64,128] need the same number of bytes, so the framing cannot see a transposed shape: only E1 can",
        header=_hdr(STACK_Q, _transpose_first_nonsquare_q8))
    add("seg-q8-dtype-in-canonical", "non-BF16", tensors=_q8_in_canonical)
    add("seg-pack-tensor-count-vs-table", "tensor_count", index=lambda i: i["representations"][STACK_Q].update(tensor_count=i["representations"][STACK_Q]["tensor_count"] - 1))

    # --- digests ---
    add("digest-payload-bit-flip", "sha256 mismatch", "both digests break; the reader reports segment_sha256 first",
        segment=_seg(STACK_Q, lambda b: b[:-1] + bytes([b[-1] ^ 0x80])), seal=False)
    add("digest-payload-bit-flip-embedding", "sha256 mismatch",
        segment=_seg(EMB, lambda b: b[:-4] + bytes([b[-4] ^ 1]) + b[-3:]), seal=False)
    add("digest-header-relabel", "segment_sha256 mismatch",
        "section 13.3: a relabel with the same byte length leaves payload_sha256 intact but changes segment_sha256",
        header=_hdr(NORM, lambda h: h["tensors"][0].update(shape=[2, h["tensors"][0]["shape"][0] // 2])), seal=False)
    add("digest-extra-padding-unsealed", "segment_sha256 mismatch",
        "padding changes segment_sha256 but not payload_sha256", pad_extra=4, seal=False)
    return F


def _transpose_first_nonsquare_q8(h: dict) -> None:
    t = next(t for t in h["tensors"] if t["dtype"] == "Q8_0" and t["shape"][0] != t["shape"][1])
    t["shape"] = list(reversed(t["shape"]))


def _q8_in_canonical(rid: str, ts: list[T]) -> list[T]:
    if rid != STACK_B:
        return ts
    target = next(t for t in ts if len(t.shape) == 2)
    return [T(t.name, "Q8_0", t.shape, ref_quantize_q8_0(bf16_widen(t.data))) if t is target else t for t in ts]


FAULTS: list[Fault] = _make_faults()
