#!/usr/bin/env python3
"""Export a real-weight Kimi MoE router + block parity fixture.

Router weights are tiny (`[256, 2304]` + `[256]`) and always exported.
Expert weights are ~4.7 MiB EACH (`w1`/`w2`/`w3` at `moe_intermediate_size
1024 x hidden 2304`) — this script routes the fixed seeded input FIRST,
then loads only the SELECTED experts plus the shared one, matching the
sparsity the whole design exists to exploit (top_k=8 of 256 ≈ 38 MiB, not
256 x 14 MiB ≈ 3.6 GiB).

    python scripts/kimi_moe_export.py <checkpoint> --layer 1 --out DIR
    LARQL_KIMI_MOE_FIXTURE=DIR cargo test -p larql-vindex --lib kimi_moe_real
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import torch

import kimi_moe_reference as ref

#: Fixed so the fixture is reproducible.
INPUT_SEED = 20260827

DTYPES = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16}


def read_tensors(checkpoint: Path, names: set[str]) -> dict[str, torch.Tensor]:
    """Read named tensors out of a sharded safetensors checkpoint, as f32."""
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    wanted = {n: index["weight_map"][n] for n in names}
    out: dict[str, torch.Tensor] = {}
    for shard in sorted(set(wanted.values())):
        path = checkpoint / shard
        with path.open("rb") as fh:
            header_len = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(header_len))
            base = 8 + header_len
            for name, target in wanted.items():
                if target != shard:
                    continue
                meta = header[name]
                start, end = meta["data_offsets"]
                fh.seek(base + start)
                raw = fh.read(end - start)
                t = torch.frombuffer(bytearray(raw), dtype=DTYPES[meta["dtype"]])
                out[name] = t.view(*meta["shape"]).to(torch.float32)
    return out


def write(path: Path, t: torch.Tensor) -> int:
    a = t.detach().contiguous().float().numpy()
    path.write_bytes(a.tobytes())
    return a.size


def write_bf16(path: Path, t: torch.Tensor) -> int:
    """P4a: expert weights only — `exec::kimi_moe_block::ExpertWeights`
    is BF16 code units, never F32 (see that module's own doc comment).
    `t` is already f32 (read via `read_tensors`, which upcasts from the
    checkpoint's native BF16); that upcast is EXACT — bf16 IS the top 16
    bits of an f32, so truncating back here recovers the checkpoint's
    OWN bits exactly, not an independent rounding of a lossy value."""
    a = t.detach().contiguous().float().numpy()
    bits = a.view(np.uint32)
    bf16 = (bits >> 16).astype("<u2")
    path.write_bytes(bf16.tobytes())
    return a.size


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--layer", type=int, default=1, help="a routed (non-dense-prefix) layer")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cfg = json.loads((args.checkpoint / "config.json").read_text())
    hidden = cfg["hidden_size"]
    experts_n = cfg["num_experts"]
    top_k = cfg["num_experts_per_token"]
    moe_renormalize = cfg["moe_renormalize"]
    branch_scale = cfg["routed_scaling_factor"]
    assert cfg.get("num_expert_group", 1) == 1 and cfg.get("topk_group", 1) == 1, (
        "this oracle only models the identity expert-group case — see "
        "kimi_moe_reference.py's module doc"
    )
    prefix = f"model.layers.{args.layer}.block_sparse_moe."

    router_tensors = read_tensors(
        args.checkpoint,
        {prefix + "gate.weight", prefix + "gate.e_score_correction_bias"},
    )
    router = ref.RouterWeights(
        weight=router_tensors[prefix + "gate.weight"],
        e_score_correction_bias=router_tensors[prefix + "gate.e_score_correction_bias"],
    )

    torch.manual_seed(INPUT_SEED)
    x = torch.randn(hidden, dtype=torch.float32) * 0.02

    routed = ref.route(x, router, top_k, moe_renormalize, branch_scale)
    ids = routed["ids"].tolist()
    print(f"layer {args.layer}: selected experts {sorted(ids)}")

    expert_names = {
        f"{prefix}experts.{i}.{suf}.weight" for i in ids for suf in ("w1", "w2", "w3")
    }
    shared_names = {f"{prefix}shared_experts.{suf}.weight" for suf in ("gate_proj", "up_proj", "down_proj")}
    tensors = read_tensors(args.checkpoint, expert_names | shared_names)

    experts = {
        i: ref.ExpertWeights(
            w1=tensors[f"{prefix}experts.{i}.w1.weight"],
            w2=tensors[f"{prefix}experts.{i}.w2.weight"],
            w3=tensors[f"{prefix}experts.{i}.w3.weight"],
        )
        for i in ids
    }
    shared = ref.ExpertWeights(
        w1=tensors[f"{prefix}shared_experts.gate_proj.weight"],
        w2=tensors[f"{prefix}shared_experts.down_proj.weight"],
        w3=tensors[f"{prefix}shared_experts.up_proj.weight"],
    )

    block = ref.moe_block_forward(x, router, experts, shared, top_k, moe_renormalize, branch_scale)

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "checkpoint": str(args.checkpoint),
        "layer": args.layer,
        "hidden": hidden,
        "moe_intermediate_size": cfg["moe_intermediate_size"],
        "num_shared_experts": cfg["num_shared_experts"],
        "experts": experts_n,
        "top_k": top_k,
        "moe_renormalize": moe_renormalize,
        "routed_scaling_factor": branch_scale,
        "selected_ids": sorted(ids),  # as chosen, not de-duplicated order
        "selected_ids_order": ids,    # as `route()` actually returned them
        "files": {},
    }

    def dump(name: str, t: torch.Tensor) -> None:
        manifest["files"][name] = write(args.out / f"{name}.f32", t)

    def dump_bf16(name: str, t: torch.Tensor) -> None:
        # P4a: real checkpoint tensors are ALREADY bf16-native — the f32
        # `read_tensors` upcast is lossless — so truncating back here
        # needs no separate quantisation step the way the SYNTHETIC
        # oracle's randomly-generated weights do.
        manifest["files"][name] = write_bf16(args.out / f"{name}.bf16", t)

    dump("input", x)
    dump("router_weight", router.weight)
    dump("router_bias", router.e_score_correction_bias)
    r = block["router"]
    for k in ref.ROUTER_BOUNDARIES:
        dump(f"router_{k}", r[k])
    for i in ids:
        e = experts[i]
        dump_bf16(f"expert{i}_w1", e.w1)
        dump_bf16(f"expert{i}_w2", e.w2)
        dump_bf16(f"expert{i}_w3", e.w3)
    dump_bf16("shared_w1", shared.w1)
    dump_bf16("shared_w2", shared.w2)
    dump_bf16("shared_w3", shared.w3)
    for idx, out in enumerate(block["expert_outputs"]):
        dump(f"expert_output_{idx}", out)  # in `selected_ids_order` position
    dump("routed_sum", block["routed_sum"])
    dump("shared_output", block["shared_output"])
    dump("output", block["output"])

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    total = sum(p.stat().st_size for p in args.out.glob("*.f32")) / 2**20 + sum(p.stat().st_size for p in args.out.glob("*.bf16")) / 2**20
    print(f"output absmax {block['output'].abs().max():.6f}")
    print(f"{total:.1f} MiB written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
