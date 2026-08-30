#!/usr/bin/env python3
"""Export a real-weight KDA parity fixture as raw f32 + a manifest.

The committed fixture is 2 heads x 4: it proves the arithmetic, and the
arithmetic is identical at any width. What it cannot prove is indexing,
stride, state sizing, convolution layout and flatten order under realistic
geometry — a transposed head axis or a wrong `h*D + d` is invisible at
`D = 4` and fatal at `D = 128`.

So this exports the full-width fixture instead of committing it: ~150 MB of
f32, written where a local test can find it and pointed at by an env var,
the same shape as the other real-model gates in this repo.

    python scripts/kda_fixture_export.py <checkpoint> --layer 0 --out DIR
    LARQL_KDA_FIXTURE=DIR cargo test -p larql-vindex --lib kda_parity_real
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import kda_reference as ref
from kda_fixture import INPUT_SEED, OPERANDS, load_layer
from kimi_moe_export import write_bf16

#: Correctness ladder, then the seam where the reference switches from
#: `fused_recurrent_kda` to `chunk_kda`. LARQL need not reproduce the
#: chunked implementation — it must stay equivalent across the point where
#: the reference changes strategy.
POSITIONS = (8, 32, 64, 65)

#: `KdaWeights`'s four widest operands (P4c-4) — dumped bf16 (real
#: checkpoint tensors are already bf16-native, so this is lossless), the
#: other eleven stay f32 like the rest of this file.
BF16_FIELDS = {"q_proj", "k_proj", "v_proj", "o_proj"}


def write(path: Path, t: torch.Tensor) -> int:
    a = t.detach().contiguous().float().numpy()
    path.write_bytes(a.tobytes())
    return a.size


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cfg = json.loads((args.checkpoint / "config.json").read_text())
    hidden = cfg["hidden_size"]
    w = load_layer(args.checkpoint, args.layer)
    args.out.mkdir(parents=True, exist_ok=True)

    manifest = {
        "checkpoint": str(args.checkpoint),
        "layer": args.layer,
        "num_heads": w.num_heads,
        "head_dim": w.head_dim,
        "conv_kernel": cfg["linear_attn_config"]["short_conv_kernel_size"],
        "hidden": hidden,
        "rms_eps": cfg["rms_norm_eps"],
        "weights": {},
        "runs": {},
    }
    field = dict(zip(
        ("q_proj", "k_proj", "v_proj", "q_conv1d", "k_conv1d", "v_conv1d",
         "f_a_proj", "f_b_proj", "g_a_proj", "g_b_proj", "b_proj", "a_log",
         "dt_bias", "o_norm", "o_proj"),
        OPERANDS,
    ))
    total = 0
    for name in field:
        t = getattr(w, name)
        if name in BF16_FIELDS:
            n = write_bf16(args.out / f"w_{name}.bf16", t)
        else:
            n = write(args.out / f"w_{name}.f32", t)
        manifest["weights"][name] = n
        total += n

    for n in POSITIONS:
        torch.manual_seed(INPUT_SEED)
        x = torch.randn(n, hidden, dtype=torch.float32) * 0.02
        b = ref.kda_forward(x, w)
        run = {"input": write(args.out / f"n{n}_input.f32", x), "boundaries": {}}
        for k in ref.BOUNDARIES:
            run["boundaries"][k] = write(args.out / f"n{n}_{k}.f32", b[k])
        run["state"] = write(args.out / f"n{n}_state.f32", b["_state"])
        run["conv_state"] = [
            write(args.out / f"n{n}_conv{i}.f32", c) for i, c in enumerate(b["_conv_state"])
        ]
        manifest["runs"][str(n)] = run
        total += sum(run["boundaries"].values()) + run["state"] + run["input"]
        print(f"  N={n:>3}  output absmax {b['output'].abs().max():.6f}  "
              f"state absmax {b['_state'].abs().max():.6f}")

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    # Actual on-disk size, not `total * 4`: that assumed every dumped file
    # was uniformly f32, wrong now that q/k/v/o_proj are 2-byte bf16 —
    # same cosmetic bug P4a already fixed in the other export scripts.
    written = sum(p.stat().st_size for p in args.out.glob("*.f32"))
    written += sum(p.stat().st_size for p in args.out.glob("*.bf16"))
    print(f"\n{written / 2**20:.1f} MiB written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
