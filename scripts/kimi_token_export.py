#!/usr/bin/env python3
"""Export a real-weight token-logits parity fixture — the P3d-k gate:

    token IDs -> embedding -> proven 27-layer stack -> final RMSNorm ->
    lm_head -> logits

No sampling, no chat template, no performance work. Reuses `run_layers`
from `kimi_stack_export.py` UNCHANGED (verified byte-identical to the
pre-refactor script before this file was written) for the stack half, so
this script's only NEW work is embedding lookup and the lm_head
projection.

`model.embed_tokens.weight` and `lm_head.weight` are separate matrices
(`tie_word_embeddings=False` in the checkpoint's own `config.json`), each
`[vocab_size, hidden]`. `lm_head` needs its FULL width loaded — the top-k
ranking and argmax are claims about the WHOLE vocabulary distribution,
not a subset — but `embed_tokens` only needs the handful of rows this
fixture's own token ids select: same sparsity discipline as every
expert-loading rung before this one, applied to the embedding table.

    python scripts/kimi_token_export.py ~/chris-models/Kimi-Linear-48B-A3B-Instruct \
        --tokens 1008 10484 318 --out DIR
    LARQL_KIMI_TOKEN_FIXTURE=DIR cargo test -p larql-vindex --lib token_real --release
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import kimi_mla_reference as mla_ref
from kimi_mla_layer_reference import rms_norm
from kimi_moe_export import read_tensors, write, write_bf16
from kimi_stack_export import layer_geometry_from_config, run_layers


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--tokens", type=int, nargs="+", required=True, help="real tokenizer IDs")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    positions = len(args.tokens)
    assert positions >= 2, (
        "MLA's attention math needs >=2 real positions to be non-degenerate "
        "(softmax over one cached score is 1.0 regardless of its value)"
    )
    assert len(set(args.tokens)) == len(args.tokens), "duplicate token ids would collide in the sparse embedding table"

    cfg = json.loads((args.checkpoint / "config.json").read_text())
    hidden = cfg["hidden_size"]
    eps = cfg["rms_norm_eps"]
    top_k = cfg["num_experts_per_token"]
    moe_renormalize = cfg["moe_renormalize"]
    branch_scale = cfg["routed_scaling_factor"]
    num_layers, kda_layers, mla_layers, dense_layers, mla_geometry = layer_geometry_from_config(cfg)

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "checkpoint": str(args.checkpoint),
        "hidden": hidden, "rms_eps": eps,
        "kda_num_heads": None, "kda_head_dim": None, "kda_conv_kernel": cfg["linear_attn_config"]["short_conv_kernel_size"],
        "mla_num_heads": mla_geometry.num_heads, "mla_kv_lora_rank": mla_geometry.kv_lora_rank,
        "mla_qk_nope_head_dim": mla_geometry.qk_nope_head_dim, "mla_qk_rope_head_dim": mla_geometry.qk_rope_head_dim,
        "mla_v_head_dim": mla_geometry.v_head_dim, "mla_kv_a_norm_eps": mla_ref.KV_A_NORM_EPS,
        "experts": cfg["num_experts"], "top_k": top_k, "moe_intermediate_size": cfg["moe_intermediate_size"],
        "num_shared_experts": cfg["num_shared_experts"], "dense_intermediate_size": cfg["intermediate_size"],
        "moe_renormalize": moe_renormalize, "routed_scaling_factor": branch_scale,
        "num_layers": num_layers, "mla_layers": sorted(mla_layers), "dense_layers": sorted(dense_layers),
        "token_ids": args.tokens, "positions": positions, "files": {}, "layers": [],
    }

    def dump(name: str, t: torch.Tensor) -> None:
        manifest["files"][name] = write(args.out / f"{name}.f32", t)

    def dump_bf16(name: str, t: torch.Tensor) -> None:
        manifest["files"][name] = write_bf16(args.out / f"{name}.bf16", t)

    embed_tokens = read_tensors(args.checkpoint, {"model.embed_tokens.weight"})["model.embed_tokens.weight"]
    manifest["vocab_size"] = embed_tokens.shape[0]
    print(f"embed_tokens.weight {list(embed_tokens.shape)} (config vocab_size={cfg['vocab_size']})")

    ids = torch.tensor(args.tokens, dtype=torch.long)
    h = embed_tokens[ids].clone()  # [positions, hidden] — sparse gather, not the full 163840-row table
    for p in range(positions):
        dump(f"embedding_{p}", h[p])
    del embed_tokens

    h, total_expert_loads = run_layers(
        args.checkpoint, num_layers, kda_layers, mla_layers, dense_layers, mla_geometry,
        eps, top_k, moe_renormalize, branch_scale, positions, h, dump, dump_bf16, manifest,
    )
    for p in range(positions):
        dump(f"stack_output_{p}", h[p])

    norm_w = read_tensors(args.checkpoint, {"model.norm.weight"})["model.norm.weight"]
    dump("final_norm_weight", norm_w)
    normed = rms_norm(h, norm_w, eps)
    for p in range(positions):
        dump(f"final_normed_{p}", normed[p])

    lm_head_w = read_tensors(args.checkpoint, {"lm_head.weight"})["lm_head.weight"]
    dump("lm_head_weight", lm_head_w)
    logits = normed.float() @ lm_head_w.float().T  # [positions, vocab]
    for p in range(positions):
        dump(f"logits_{p}", logits[p])

    argmax_ids = logits.argmax(dim=-1).tolist()
    top10 = logits.topk(10, dim=-1).indices.tolist()
    manifest["argmax_ids"] = argmax_ids
    manifest["top10_ids"] = top10

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    total = sum(p.stat().st_size for p in args.out.glob("*.f32")) / 2**20 + sum(p.stat().st_size for p in args.out.glob("*.bf16")) / 2**20
    print(f"\ntoken ids: {args.tokens}")
    print(f"argmax next-token id per position: {argmax_ids}")
    print(f"{total_expert_loads} total (layer, expert) loads across {num_layers} layers")
    print(f"{total:.1f} MiB written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
