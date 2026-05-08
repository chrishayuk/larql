#!/usr/bin/env python3
"""Run the vendored upstream Triton RotorQuant reference kernels.

Input floats are read from stdin as whitespace-separated f32 values.
Output floats are written to stdout in row-major order.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: triton_reference.py planar3|iso3 N_ROWS HEAD_DIM", file=sys.stderr)
        return 2

    fmt = sys.argv[1]
    n_rows = int(sys.argv[2])
    head_dim = int(sys.argv[3])
    values = [float(x) for x in sys.stdin.read().split()]
    if len(values) != n_rows * head_dim:
        print(
            f"expected {n_rows * head_dim} floats, got {len(values)}",
            file=sys.stderr,
        )
        return 2

    try:
        import torch

        from upstream.triton_isoquant import triton_iso_fast_fused
        from upstream.triton_planarquant import triton_planar2_fused
    except Exception as exc:  # pragma: no cover - exercised by Rust harness
        print(f"missing Python/Triton dependency: {exc}", file=sys.stderr)
        return 77

    if not torch.cuda.is_available():
        print("torch CUDA is not available", file=sys.stderr)
        return 77

    root = Path(__file__).resolve().parent
    if not (root / "upstream/triton_planarquant.py").is_file():
        print("missing vendored upstream reference files", file=sys.stderr)
        return 2

    x = torch.tensor(values, dtype=torch.float32, device="cuda").reshape(n_rows, head_dim)
    centroids = torch.tensor(
        [-0.875, -0.625, -0.375, -0.125, 0.125, 0.375, 0.625, 0.875],
        dtype=torch.float32,
        device="cuda",
    )

    if fmt == "planar3":
        if head_dim % 2 != 0:
            print("planar3 requires even head_dim", file=sys.stderr)
            return 2
        idx = torch.arange(head_dim // 2, dtype=torch.float32, device="cuda")
        theta = idx * (math.pi / 2.0 / 8.0)
        rotations = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
        y = triton_planar2_fused(x, rotations, centroids)
    elif fmt == "iso3":
        if head_dim % 4 != 0:
            print("iso3 requires head_dim divisible by 4", file=sys.stderr)
            return 2
        idx = torch.arange(head_dim // 4, dtype=torch.float32, device="cuda")
        half = idx * (math.pi / 2.0 / 15.0)
        s = torch.sin(half)
        c = torch.cos(half)
        inv_sqrt3 = 1.0 / math.sqrt(3.0)
        q_left = torch.stack([c, s * inv_sqrt3, s * inv_sqrt3, s * inv_sqrt3], dim=-1)
        y = triton_iso_fast_fused(x, q_left, centroids)
    else:
        print(f"unsupported format {fmt}", file=sys.stderr)
        return 2

    y_cpu = y.detach().cpu().reshape(-1).tolist()
    sys.stdout.write(" ".join(f"{v:.9g}" for v in y_cpu))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
