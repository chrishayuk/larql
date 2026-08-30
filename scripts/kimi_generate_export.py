#!/usr/bin/env python3
"""16-token deterministic greedy decode export — P3d-m.

Greedy, real weights, no sampling: at each step, run the model on the
GROWING sequence, take argmax at the last position, append it, repeat.
Whole-sequence-per-step is mathematically identical to true incremental
decode-with-cache for any internally-causal stack — the same equivalence
every real-weight gate in this ladder already leans on (`stack_real.rs`'s
own doc comment states it first).

**Real weight traffic is not repeated 16 times.** Attention/norm/router
weights are checkpoint-static — independent of the token sequence — and
routed-expert selections only ever GROW as new positions are considered
(never shrink), so an in-PROCESS tensor cache shared across all 16 steps
means each real tensor is read from the 92 GB container AT MOST ONCE for
this whole run, matching the disk cost of one large whole-sequence
export rather than sixteen separate ones. The cache wraps — never
modifies — `kda_fixture.read_tensors` and `kimi_moe_export.read_tensors`
(the same function `kimi_stack_export.run_layers` calls internally).

Weight/boundary files are only written on the FINAL step, at the full
19-position sequence (3-token prompt + 16 generated) — the earlier 16
steps exist only to DISCOVER which token comes next, exactly as real
greedy decoding does; their own intermediate boundaries are not
independently interesting once the final full-sequence pass covers the
same ground with the same cache warm.

    python scripts/kimi_generate_export.py ~/chris-models/Kimi-Linear-48B-A3B-Instruct \
        --tokens 1008 10484 318 --new 16 --out DIR
    LARQL_KIMI_GENERATE_FIXTURE=DIR cargo test -p larql-vindex --lib generate_real --release
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import kda_fixture
import kimi_mla_reference as mla_ref
import kimi_moe_export
import kimi_stack_export
from kimi_mla_layer_reference import rms_norm
from kimi_moe_export import write, write_bf16
from kimi_stack_export import layer_geometry_from_config, run_layers


def install_tensor_cache() -> dict:
    """Wrap the two REAL tensor readers (`kda_fixture`'s own, and the one
    `kimi_stack_export.run_layers` calls) so a name already read this
    process is never read from disk twice. Returns the cache dict for the
    caller's own inspection (e.g. counting cache hits)."""
    cache: dict[str, torch.Tensor] = {}

    def wrap(original):
        def wrapper(checkpoint: Path, names: set[str]) -> dict[str, torch.Tensor]:
            missing = names - cache.keys()
            if missing:
                cache.update(original(checkpoint, missing))
            return {n: cache[n] for n in names}
        return wrapper

    kda_fixture.read_tensors = wrap(kda_fixture.read_tensors)
    kimi_moe_export.read_tensors = wrap(kimi_moe_export.read_tensors)
    kimi_stack_export.read_tensors = kimi_moe_export.read_tensors
    return cache


def no_dump(_name: str, _t: torch.Tensor) -> None:
    pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--tokens", type=int, nargs="+", required=True, help="real prompt tokenizer IDs")
    ap.add_argument("--new", type=int, default=16, help="how many new tokens to greedily generate")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cfg = json.loads((args.checkpoint / "config.json").read_text())
    hidden = cfg["hidden_size"]
    eps = cfg["rms_norm_eps"]
    top_k = cfg["num_experts_per_token"]
    moe_renormalize = cfg["moe_renormalize"]
    branch_scale = cfg["routed_scaling_factor"]
    num_layers, kda_layers, mla_layers, dense_layers, mla_geometry = layer_geometry_from_config(cfg)

    cache = install_tensor_cache()

    embed_tokens = kimi_moe_export.read_tensors(args.checkpoint, {"model.embed_tokens.weight"})["model.embed_tokens.weight"]
    norm_w = kimi_moe_export.read_tensors(args.checkpoint, {"model.norm.weight"})["model.norm.weight"]
    lm_head_w = kimi_moe_export.read_tensors(args.checkpoint, {"lm_head.weight"})["lm_head.weight"]
    vocab_size = embed_tokens.shape[0]

    tokens = list(args.tokens)
    per_step = []
    t_generate_start = time.monotonic()
    for step in range(args.new):
        t0 = time.monotonic()
        h = embed_tokens[torch.tensor(tokens, dtype=torch.long)]
        manifest_stub = {"kda_num_heads": None, "kda_head_dim": None, "layers": []}
        h, _ = run_layers(
            args.checkpoint, num_layers, kda_layers, mla_layers, dense_layers, mla_geometry,
            eps, top_k, moe_renormalize, branch_scale, len(tokens), h, no_dump, no_dump, manifest_stub,
        )
        normed_last = rms_norm(h[-1:], norm_w, eps)[0]
        logits = normed_last.float() @ lm_head_w.float().T
        top2 = logits.topk(2).values.tolist()
        next_id = int(logits.argmax())
        elapsed = time.monotonic() - t0
        per_step.append({
            "step": step, "sequence_length_before": len(tokens), "next_id": next_id,
            "top1_logit": top2[0], "top2_logit": top2[1], "top1_top2_margin": top2[0] - top2[1],
            "seconds": elapsed, "cache_size_after": len(cache),
        })
        print(f"  step {step:>2}: len {len(tokens):>2} -> token {next_id:>6} "
              f"(margin {top2[0]-top2[1]:.4f}, {elapsed:.2f}s, cache {len(cache)} tensors)")
        tokens.append(next_id)
    t_generate = time.monotonic() - t_generate_start

    print(f"\ngenerated {args.new} tokens in {t_generate:.1f}s: {tokens[len(args.tokens):]}")

    # ── Final pass: full 19-position boundaries, REAL dump, cache already warm ──
    args.out.mkdir(parents=True, exist_ok=True)
    positions = len(tokens)
    manifest = {
        "checkpoint": str(args.checkpoint), "hidden": hidden, "rms_eps": eps,
        "kda_num_heads": None, "kda_head_dim": None, "kda_conv_kernel": cfg["linear_attn_config"]["short_conv_kernel_size"],
        "mla_num_heads": mla_geometry.num_heads, "mla_kv_lora_rank": mla_geometry.kv_lora_rank,
        "mla_qk_nope_head_dim": mla_geometry.qk_nope_head_dim, "mla_qk_rope_head_dim": mla_geometry.qk_rope_head_dim,
        "mla_v_head_dim": mla_geometry.v_head_dim,
        "mla_kv_a_norm_eps": mla_ref.KV_A_NORM_EPS,
        "experts": cfg["num_experts"], "top_k": top_k, "moe_intermediate_size": cfg["moe_intermediate_size"],
        "num_shared_experts": cfg["num_shared_experts"], "dense_intermediate_size": cfg["intermediate_size"],
        "moe_renormalize": moe_renormalize, "routed_scaling_factor": branch_scale,
        "num_layers": num_layers, "mla_layers": sorted(mla_layers), "dense_layers": sorted(dense_layers),
        "prompt_tokens": args.tokens, "generated_tokens": tokens[len(args.tokens):],
        "token_ids": tokens, "positions": positions, "vocab_size": vocab_size,
        "per_step": per_step, "files": {}, "layers": [],
    }

    def dump(name: str, t: torch.Tensor) -> None:
        manifest["files"][name] = write(args.out / f"{name}.f32", t)

    def dump_bf16(name: str, t: torch.Tensor) -> None:
        manifest["files"][name] = write_bf16(args.out / f"{name}.bf16", t)

    ids = torch.tensor(tokens, dtype=torch.long)
    h = embed_tokens[ids].clone()
    for p in range(positions):
        dump(f"embedding_{p}", h[p])

    t_dump_start = time.monotonic()
    h, total_expert_loads = run_layers(
        args.checkpoint, num_layers, kda_layers, mla_layers, dense_layers, mla_geometry,
        eps, top_k, moe_renormalize, branch_scale, positions, h, dump, dump_bf16, manifest,
    )
    for p in range(positions):
        dump(f"stack_output_{p}", h[p])

    dump("final_norm_weight", norm_w)
    normed = rms_norm(h, norm_w, eps)
    for p in range(positions):
        dump(f"final_normed_{p}", normed[p])

    dump("lm_head_weight", lm_head_w)
    logits = normed.float() @ lm_head_w.float().T
    for p in range(positions):
        dump(f"logits_{p}", logits[p])
    t_dump = time.monotonic() - t_dump_start

    argmax_ids = logits.argmax(dim=-1).tolist()
    top10 = logits.topk(10, dim=-1).indices.tolist()
    manifest["argmax_ids"] = argmax_ids
    manifest["top10_ids"] = top10
    manifest["generate_seconds"] = t_generate
    manifest["final_dump_seconds"] = t_dump
    manifest["cache_final_size"] = len(cache)

    # Sanity: the greedy loop's own choice at each step must be exactly
    # what the final full-sequence pass predicts at that same position —
    # both are the SAME deterministic computation, so any disagreement
    # would mean the cache broke determinism, not a real model property.
    prompt_len = len(args.tokens)
    for step, s in enumerate(per_step):
        pos = prompt_len + step - 1
        assert argmax_ids[pos] == s["next_id"], (step, pos, argmax_ids[pos], s["next_id"])

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    total = sum(p.stat().st_size for p in args.out.glob("*.f32")) / 2**20 + sum(p.stat().st_size for p in args.out.glob("*.bf16")) / 2**20
    print(f"\nfull {positions}-position argmax: {argmax_ids}")
    print(f"{total_expert_loads} total (layer, expert) loads across {num_layers} layers, this pass")
    print(f"cache holds {len(cache)} distinct tensors after the whole run")
    print(f"{total:.1f} MiB written to {args.out}")
    print(f"generate: {t_generate:.1f}s, final dump: {t_dump:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
