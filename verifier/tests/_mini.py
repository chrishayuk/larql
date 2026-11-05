"""A deliberately plain, hand-assembled tiny container for the reader tests.

It shares no code with the synthetic writer (refcheck.synth), so the reader is tested against
bytes laid out directly from SPEC.md sections 4, 5, 6 and 8, not against its own counterpart.
"""

import hashlib
import json
import os
import struct
import tempfile

SCRATCH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".tmp")


def tmpdir():
    """Scratch directories live in the repo's gitignored .tmp (the system temp dir is not always readable)."""
    os.makedirs(SCRATCH, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=SCRATCH)

CODEC = {
    "family": "Q8_0",
    "revision": 1,
    "group_elems": 32,
    "element": "ggml-q8_0",
    "group_scale": "in-block",
    "tensor_scale": "in-block",
    "layout": "ggml-block-34",
}


def q8_block(d_bits, qs):
    qs = list(qs) + [0] * (32 - len(qs))
    return struct.pack("<H", d_bits) + struct.pack("<32b", *qs)


def segment(rep_id, tensors):
    """tensors: list of (name, dtype, shape, payload_bytes), already in sorted order."""
    table, payload, off = [], b"", 0
    for name, dtype, shape, data in tensors:
        table.append({"name": name, "dtype": dtype, "shape": shape, "offset": off, "len": len(data)})
        payload += data
        off += len(data)
    js = json.dumps({"schema": 1, "representation": rep_id, "tensors": table}, separators=(",", ":")).encode()
    pad = (16 - (8 + len(js)) % 16) % 16
    hdr = js + b" " * pad
    return struct.pack("<Q", len(hdr)) + hdr + payload, payload


BF16_NORM = struct.pack("<32H", *([0x3F80] * 32))  # 1.0 in bf16
BF16_EMB = struct.pack("<128H", *range(0x3F00, 0x3F00 + 128))  # [4, 32]
Q8_UP = q8_block(0x3C00, range(-16, 16)) + q8_block(0x4000, [1] * 32) + q8_block(0xB800, [127, -128]) + q8_block(0, [5])


def build(root, **mut):
    """Write a valid tiny container into `root`. `mut` hooks allow targeted corruption:
    index(dict), graph(dict), segs(dict rep_id -> bytes transform), skip_index(bool)."""
    os.makedirs(os.path.join(root, "segments"), exist_ok=True)
    blobs = {}
    stack_bf16 = mut.get("stack_bf16") or [
        ("0.input_layernorm.weight", "BF16", [32], BF16_NORM),
        ("0.mlp.up_proj.weight", "BF16", [2, 64], struct.pack("<128H", *([0x3F80] * 128))),
    ]
    stack_q8 = [
        ("0.input_layernorm.weight", "BF16", [32], BF16_NORM),
        ("0.mlp.up_proj.weight", "Q8_0", [2, 64], Q8_UP),
    ]
    blobs["target.decoder_stack@BF16"] = ("segments/target.decoder_stack.bin", segment("target.decoder_stack@BF16", stack_bf16))
    blobs["target.decoder_stack@Q8_0"] = ("segments/target.decoder_stack@Q8_0.bin", segment("target.decoder_stack@Q8_0", stack_q8))
    blobs["target.embedding@BF16"] = ("segments/target.embedding.bin", segment("target.embedding@BF16", [("weight", "BF16", [4, 32], BF16_EMB)]))
    blobs["target.final_norm@BF16"] = ("segments/target.final_norm.bin", segment("target.final_norm@BF16", [("weight", "BF16", [32], BF16_NORM)]))
    blobs["target.output_head@BF16"] = ("segments/target.output_head.bin", segment("target.output_head@BF16", [("weight", "BF16", [4, 32], BF16_EMB)]))

    reps = {}
    for rid, (path, (whole, payload)) in blobs.items():
        obj, enc = rid.split("@")
        if rid in mut.get("segs", {}):
            whole = mut["segs"][rid](whole)
        with open(os.path.join(root, path), "wb") as f:
            f.write(whole)
        e = {
            "object": obj,
            "encoding": enc,
            "segment": path,
            "tensor_count": 2 if obj == "target.decoder_stack" else 1,
            "payload_bytes": len(payload),
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "segment_sha256": hashlib.sha256(blobs[rid][1][0]).hexdigest(),
        }
        if rid.endswith("@Q8_0"):
            e.update(
                compiled_from="target.decoder_stack@BF16",
                codec=dict(CODEC),
                source_representation_digest=hashlib.sha256(blobs["target.decoder_stack@BF16"][1][1]).hexdigest(),
                encoder={"algorithm": "kquant-larql-native", "revision": 1},
            )
        reps[rid] = e

    index = {
        "version": 4,
        "model": "mini",
        "family": "qwen3",
        "hidden_size": 32,
        "num_layers": 1,
        "system_graph": "system_graph.json",
        "representations": reps,
        "profiles": [{"name": "exact"}],
        "segments": {p[:-4]: 1 for p, _ in blobs.values()},
        "precision_map": {"name": "r0-uniform", "encoding": "Q8_0", "roles": ["decoder-linear"]},
    }
    prefixes = {
        "target.decoder_stack": ("model.layers", "decoder_stack"),
        "target.embedding": ("model.embed_tokens", "embedding"),
        "target.final_norm": ("model.norm", "final_norm"),
        "target.output_head": ("lm_head", "output_head"),
    }
    objs = []
    for oid, (pre, kind) in sorted(prefixes.items()):
        r = [{"encoding": "BF16", "fidelity": "canonical"}]
        if oid == "target.decoder_stack":
            r.append({"encoding": "Q8_0", "fidelity": "approximate"})
        objs.append(
            {
                "id": oid,
                "component": "target",
                "kind": kind,
                "source_bindings": [{"artifact": "mini", "tensor_prefix": pre, "tensors": 2 if kind == "decoder_stack" else 1, "bytes": reps[f"{oid}@BF16"]["payload_bytes"]}],
                "representations": r,
            }
        )
    graph = {
        "schema": 6,
        "components": [{"id": "target", "role": "primary_text", "execution": {"head": {"vocab_size": 4, "head_reuses_embedding": True}}}],
        "objects": objs,
        "edges": [],
    }
    if "index" in mut:
        mut["index"](index)
    if "graph" in mut:
        mut["graph"](graph)
    if not mut.get("skip_index"):
        with open(os.path.join(root, "index.json"), "w") as f:
            json.dump(index, f, indent=2)
    with open(os.path.join(root, "system_graph.json"), "w") as f:
        json.dump(graph, f, indent=2)
    return index, graph
