#!/usr/bin/env python3
"""Write a tiny Llama-shaped Hugging Face checkpoint for the release smoke.

The release smoke (scripts/release-smoke.sh) proves a *published* binary can
encode, run and serve a model with no network and no Rust toolchain. No
model ships in the repo, so this writes the smallest checkpoint the released
`larql vindex3 encode` admits: the geometry of the dense fixture in
crates/larql-vindex/src/format/vindex3/fixtures.rs (2 layers, hidden 64,
GQA 8/2), F32 weights from a fixed LCG, a WordLevel tokenizer whose token
`[i]` is id `i`, and a chat template. The encoder carries tokenizer.json and
chat_template.jinja into the container, so `larql run` and `larql serve`
exercise their real template + tokenizer paths.

Standard library only: the smoke runs on bare GitHub runners (macOS, Linux,
Windows) and must not pip-install anything.

    python3 scripts/release_smoke_fixture.py <output-dir>
"""

import json
import struct
import sys
from pathlib import Path

# Geometry of the dense fixture (fixtures.rs DENSE_*).
HIDDEN = 64
INTERMEDIATE = 256
VOCAB = 128
Q_HEADS = 8
KV_HEADS = 2
HEAD_DIM = 8
LAYERS = 2
RMS_NORM_EPS = 1e-5
ROPE_THETA = 10000.0

# The LCG fixtures.rs uses, so weights are small and deterministic.
LCG_MUL = 6364136223846793005
LCG_ADD = 1442695040888963407
U64 = (1 << 64) - 1
WEIGHT_SPAN = 0.1

# A template whose rendering is a run of fixture tokens: an opener, each
# turn's content, then a generation marker (the shape the CLI/server
# chat-template tests use).
CHAT_TEMPLATE = "[9] {% for m in messages %}{{ m.content }} {% endfor %}[8]"
UNK_TOKEN = "[0]"


def lcg_values(n, seed):
    state = (seed * LCG_MUL + LCG_ADD) & U64
    out = []
    for _ in range(n):
        state = (state * LCG_MUL + LCG_ADD) & U64
        unit = (state >> 33) / float(1 << 31)
        out.append((unit - 0.5) * WEIGHT_SPAN)
    return out


def norm_values(n, seed):
    return [1.0 + v for v in lcg_values(n, seed)]


def tensors():
    q_rows = Q_HEADS * HEAD_DIM
    kv_rows = KV_HEADS * HEAD_DIM
    yield "model.embed_tokens.weight", [VOCAB, HIDDEN], lcg_values(VOCAB * HIDDEN, 1)
    yield "model.norm.weight", [HIDDEN], norm_values(HIDDEN, 2)
    yield "lm_head.weight", [VOCAB, HIDDEN], lcg_values(VOCAB * HIDDEN, 3)
    for layer in range(LAYERS):
        seed = 100 + layer * 10
        p = f"model.layers.{layer}"
        yield f"{p}.self_attn.q_proj.weight", [q_rows, HIDDEN], lcg_values(q_rows * HIDDEN, seed)
        yield f"{p}.self_attn.k_proj.weight", [kv_rows, HIDDEN], lcg_values(kv_rows * HIDDEN, seed + 1)
        yield f"{p}.self_attn.v_proj.weight", [kv_rows, HIDDEN], lcg_values(kv_rows * HIDDEN, seed + 2)
        yield f"{p}.self_attn.o_proj.weight", [HIDDEN, q_rows], lcg_values(HIDDEN * q_rows, seed + 3)
        yield f"{p}.input_layernorm.weight", [HIDDEN], norm_values(HIDDEN, seed + 4)
        yield f"{p}.post_attention_layernorm.weight", [HIDDEN], norm_values(HIDDEN, seed + 5)
        yield f"{p}.mlp.gate_proj.weight", [INTERMEDIATE, HIDDEN], lcg_values(INTERMEDIATE * HIDDEN, seed + 6)
        yield f"{p}.mlp.up_proj.weight", [INTERMEDIATE, HIDDEN], lcg_values(INTERMEDIATE * HIDDEN, seed + 7)
        yield f"{p}.mlp.down_proj.weight", [HIDDEN, INTERMEDIATE], lcg_values(HIDDEN * INTERMEDIATE, seed + 8)


def write_safetensors(path):
    header = {}
    payload = bytearray()
    for name, shape, values in tensors():
        start = len(payload)
        payload += struct.pack(f"<{len(values)}f", *values)
        header[name] = {"dtype": "F32", "shape": shape, "data_offsets": [start, len(payload)]}
    header_bytes = json.dumps(header, separators=(",", ":")).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(payload)


def config():
    return {
        "architectures": ["LlamaForCausalLM"],
        "torch_dtype": "float32",
        "model_type": "llama",
        "tie_word_embeddings": False,
        "hidden_size": HIDDEN,
        "num_hidden_layers": LAYERS,
        "intermediate_size": INTERMEDIATE,
        "num_attention_heads": Q_HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "vocab_size": VOCAB,
        "rms_norm_eps": RMS_NORM_EPS,
        "rope_theta": ROPE_THETA,
    }


def tokenizer():
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": {"type": "WhitespaceSplit"},
        "post_processor": None,
        "decoder": None,
        "model": {
            "type": "WordLevel",
            "vocab": {f"[{i}]": i for i in range(VOCAB)},
            "unk_token": UNK_TOKEN,
        },
    }


def main(argv):
    if len(argv) != 2:
        sys.exit(f"usage: {argv[0]} <output-dir>")
    out = Path(argv[1])
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(config(), indent=2))
    (out / "tokenizer.json").write_text(json.dumps(tokenizer()))
    (out / "chat_template.jinja").write_text(CHAT_TEMPLATE)
    write_safetensors(out / "model.safetensors")
    print(f"wrote tiny llama checkpoint to {out}")


if __name__ == "__main__":
    main(sys.argv)
