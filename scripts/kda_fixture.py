#!/usr/bin/env python3
"""Freeze an attention-only KDA parity fixture from a real checkpoint.

Deliberately narrow: one KDA layer's fifteen operands, a fixed input, and
no router, no MoE, no residual, no MLP. A mismatch in this fixture can only
be the recurrence or the tensors feeding it.

    python scripts/kda_fixture.py <checkpoint-dir> --layer 0 --out fixture/

The ladder is `N = 1, 2, 8, 32` for correctness and `64, 65` for the seam:
the reference switches from `fused_recurrent_kda` to `chunk_kda` above 64
positions. LARQL need not reproduce the chunked *implementation*, but it
must reproduce the same recurrence across that boundary, and a fixture that
never crosses it cannot say so.

Every boundary is dumped, not just the output, and the recurrent and conv
states are dumped alongside: an implementation can match token 0 exactly and
be wrong from token 1, so **state parity is the gate**, not output parity.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import torch

import kda_reference as ref

#: Positions the correctness ladder runs at, then the reference-path seam.
LADDER = (1, 2, 8, 32)
SEAM = (64, 65)

#: Fixed so a fixture is reproducible; the input is not learned, only stable.
INPUT_SEED = 20260827

#: Suffixes of the fifteen operands, in `KdaWeights` field order.
OPERANDS = (
    "q_proj.weight", "k_proj.weight", "v_proj.weight",
    "q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight",
    "f_a_proj.weight", "f_b_proj.weight", "g_a_proj.weight", "g_b_proj.weight",
    "b_proj.weight", "A_log", "dt_bias", "o_norm.weight", "o_proj.weight",
)

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


def load_layer(checkpoint: Path, layer: int) -> ref.KdaWeights:
    prefix = f"model.layers.{layer}.self_attn."
    tensors = read_tensors(checkpoint, {prefix + s for s in OPERANDS})
    got = [tensors[prefix + s] for s in OPERANDS]
    got[11] = got[11].reshape(-1)          # A_log stored [1,1,H,1]
    return ref.KdaWeights(*got)


def run(weights: ref.KdaWeights, positions: int, hidden: int) -> dict:
    torch.manual_seed(INPUT_SEED)
    x = torch.randn(positions, hidden, dtype=torch.float32) * 0.02
    return x, ref.kda_forward(x, weights)


def summarise(name: str, t: torch.Tensor) -> dict:
    """A shape-and-statistics digest — enough to localise a divergence
    without carrying megabytes of tensor into a report."""
    f = t.float()
    return {
        "shape": list(t.shape),
        "mean": round(f.mean().item(), 8),
        "std": round(f.std().item(), 8) if f.numel() > 1 else 0.0,
        "absmax": round(f.abs().max().item(), 8),
        "sum": round(f.sum().item(), 6),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cfg = json.loads((args.checkpoint / "config.json").read_text())
    hidden = cfg["hidden_size"]
    kda_layers = {i - 1 for i in cfg["linear_attn_config"]["kda_layers"]}
    if args.layer not in kda_layers:
        print(f"layer {args.layer} is not a KDA layer; KDA layers are {sorted(kda_layers)[:8]}…")
        return 2

    w = load_layer(args.checkpoint, args.layer)
    print(f"layer {args.layer}: {w.num_heads} heads x {w.head_dim}, hidden {hidden}")

    args.out.mkdir(parents=True, exist_ok=True)
    fixture = {
        "checkpoint": str(args.checkpoint),
        "layer": args.layer,
        "num_heads": w.num_heads,
        "head_dim": w.head_dim,
        "hidden": hidden,
        "input_seed": INPUT_SEED,
        "runs": {},
    }
    for n in (*LADDER, *SEAM):
        x, b = run(w, n, hidden)
        fixture["runs"][str(n)] = {
            "input": summarise("input", x),
            "boundaries": {k: summarise(k, b[k]) for k in ref.BOUNDARIES},
            "recurrent_state": summarise("state", b["_state"]),
            "conv_state": [summarise("cs", c) for c in b["_conv_state"]],
        }
        print(f"  N={n:>3}  output absmax {b['output'].abs().max():.6f}   "
              f"state absmax {b['_state'].abs().max():.6f}")
        torch.save({"x": x, **{k: b[k] for k in ref.BOUNDARIES},
                    "state": b["_state"], "conv_state": b["_conv_state"]},
                   args.out / f"n{n}.pt")

    (args.out / "fixture.json").write_text(json.dumps(fixture, indent=1))
    print(f"\nfixture written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
