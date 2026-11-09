"""PREC-DIAG-1 E1b: LARQL's RoPE cos/sin tables against transformers', bit for bit.

usage:
  python3 -I scripts/prec_diag_rope_tables.py --repo google/gemma-3-270m-it \
      --revision <sha> --example-bin <path to rope_angle_tables> --positions 1040 \
      --out <report.json> [--linear-factor 8]

For each of the checkpoint's layer types (Gemma 3: `sliding_attention`,
`full_attention`), builds transformers' own `Gemma3RotaryEmbedding` from the
checkpoint's CONFIG (no weights are loaded, no forward pass runs) and takes
the cos/sin it produces for positions 0..N-1 in f32. It runs LARQL's
`rope_angle_tables` example for the same head width and base, then compares:

  * `hf-f32` (E0b's construction) against transformers: inv_freq and cos/sin.
  * `larql` (the shipped construction) against transformers: cos/sin.

It reports, per table, the count and fraction of bit-identical entries and
the maximum distance in f32 ULPs, with where it occurs. Arithmetic only.
`--linear-factor` additionally compares the linear-scaled construction
(Gemma 3 4B's global layers), built by setting that layer type's
`rope_type: linear` on the loaded config.
"""

from __future__ import annotations

import argparse
import copy
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

# The installed torchvision does not match torch; transformers imports it
# for image utilities this script never uses (as in hf_equivalence.py).
sys.modules["torchvision"] = None

HF_HALVES = 2  # transformers' `emb = cat(freqs, freqs)`: the table repeats once.


def ordered(values: np.ndarray) -> np.ndarray:
    """f32 bit patterns mapped to integers whose order is the floats' order
    (so a difference is a distance in ULPs). -0 and +0 map to the same 0."""
    bits = values.astype(np.float32).view(np.uint32).astype(np.int64)
    sign = np.int64(0x80000000)
    return np.where(bits >= sign, sign - bits, bits)


def ulp_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(ordered(a) - ordered(b))


def compare(name: str, larql: np.ndarray, hf: np.ndarray) -> dict:
    if larql.shape != hf.shape:
        raise ValueError(f"{name}: shape {larql.shape} vs {hf.shape}")
    same = larql.view(np.uint32) == hf.view(np.uint32)
    ulps = ulp_distance(larql, hf)
    worst = np.unravel_index(int(np.argmax(ulps)), ulps.shape)
    # ULPs explode near zero (cos crossing zero), so the absolute
    # difference is reported beside them; neither alone is the size.
    absolute = np.abs(larql.astype(np.float64) - hf.astype(np.float64))
    worst_abs = np.unravel_index(int(np.argmax(absolute)), absolute.shape)
    return {
        "table": name,
        "entries": int(same.size),
        "bit_identical": int(same.sum()),
        "bit_identical_fraction": float(same.mean()),
        "max_ulp": int(ulps.max()),
        "max_ulp_at": [int(i) for i in worst],
        "entries_over_1_ulp": int((ulps > 1).sum()),
        "max_abs": float(absolute.max()),
        "max_abs_at": [int(i) for i in worst_abs],
    }


def read(path: Path, rows: int, cols: int) -> np.ndarray:
    return np.fromfile(path, dtype="<f4").reshape(rows, cols)


def hf_tables(config, layer_type: str, positions: int):
    import torch
    from transformers.models.gemma3.modeling_gemma3 import Gemma3RotaryEmbedding

    rotary = Gemma3RotaryEmbedding(config).eval()
    head_dim = config.head_dim
    x = torch.zeros(1, 1, head_dim, dtype=torch.float32)
    position_ids = torch.arange(positions, dtype=torch.int64)[None, :]
    with torch.no_grad():
        cos, sin = rotary(x, position_ids, layer_type=layer_type)
    half = head_dim // HF_HALVES
    cos, sin = cos[0].numpy(), sin[0].numpy()
    inv_freq = getattr(rotary, f"{layer_type}_inv_freq").numpy()
    # `cat(freqs, freqs)` gives both halves the same ANGLES, but torch's
    # vectorised trig need not round both halves alike: HF's rotation then
    # uses the first half's cos/sin for output dims [0, half) and the second
    # half's for [half, head_dim). Both are returned and compared.
    return (inv_freq.astype(np.float32),
            cos[:, :half].copy(), sin[:, :half].copy(),
            cos[:, half:].copy(), sin[:, half:].copy())


def larql_tables(binary: Path, head_dim: int, theta: float, positions: int, factor):
    out = Path(tempfile.mkdtemp(prefix="rope-tables-"))
    cmd = [str(binary), str(head_dim), repr(theta), str(positions), str(out)]
    if factor is not None:
        cmd.append(repr(factor))
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    half = head_dim // 2
    return {
        "larql": (read(out / "larql.cos.f32", positions, half), read(out / "larql.sin.f32", positions, half)),
        "hf-f32": (read(out / "hf-f32.cos.f32", positions, half), read(out / "hf-f32.sin.f32", positions, half)),
        "inv_freq": np.fromfile(out / "hf-f32.inv_freq.f32", dtype="<f4"),
    }


def run_case(config, layer_type: str, binary: Path, positions: int, factor) -> dict:
    params = config.rope_parameters[layer_type]
    theta = float(params["rope_theta"])
    inv_hf, cos_hf, sin_hf, cos_hf2, sin_hf2 = hf_tables(config, layer_type, positions)
    ours = larql_tables(binary, config.head_dim, theta, positions, factor)
    return {
        "layer_type": layer_type,
        "rope_parameters": {k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in params.items()},
        "theta": theta,
        "linear_factor": factor,
        "head_dim": config.head_dim,
        "positions": positions,
        "transformers_first_half_vs_second_half": [
            compare("cos", cos_hf, cos_hf2),
            compare("sin", sin_hf, sin_hf2),
        ],
        "hf-f32_vs_transformers": [
            compare("inv_freq", ours["inv_freq"], inv_hf),
            compare("cos[first half]", ours["hf-f32"][0], cos_hf),
            compare("sin[first half]", ours["hf-f32"][1], sin_hf),
            compare("cos[second half]", ours["hf-f32"][0], cos_hf2),
            compare("sin[second half]", ours["hf-f32"][1], sin_hf2),
        ],
        "larql_vs_transformers": [
            compare("cos[first half]", ours["larql"][0], cos_hf),
            compare("sin[first half]", ours["larql"][1], sin_hf),
            compare("cos[second half]", ours["larql"][0], cos_hf2),
            compare("sin[second half]", ours["larql"][1], sin_hf2),
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--example-bin", required=True, type=Path)
    ap.add_argument("--positions", type=int, required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--linear-factor", type=float, default=None)
    args = ap.parse_args()

    import torch
    import transformers
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(args.repo, revision=args.revision, local_files_only=True)
    config = getattr(config, "text_config", None) or config
    cases = [run_case(config, lt, args.example_bin, args.positions, None)
             for lt in sorted(config.rope_parameters)]
    if args.linear_factor is not None:
        linear = copy.deepcopy(config)
        linear.rope_parameters["full_attention"] = {
            **linear.rope_parameters["full_attention"],
            "rope_type": "linear",
            "factor": args.linear_factor,
        }
        cases.append(run_case(linear, "full_attention", args.example_bin, args.positions,
                              args.linear_factor))
    report = {
        "schema": "larql.prec-diag-1.e1b-rope-tables/1",
        "scope": "arithmetic only: rotary tables from config; no weights, no forward pass",
        "repo": args.repo,
        "revision": args.revision,
        "example_bin": str(args.example_bin),
        "environment": {"transformers": transformers.__version__, "torch": torch.__version__,
                        "numpy": np.__version__, "python": sys.version.split()[0],
                        "platform": platform.platform(), "machine": platform.machine()},
        "cases": cases,
    }
    args.out.write_text(json.dumps(report, indent=1))
    for case in cases:
        label = f"{case['layer_type']} theta={case['theta']:g} factor={case['linear_factor']}"
        for arm in ("transformers_first_half_vs_second_half", "hf-f32_vs_transformers",
                    "larql_vs_transformers"):
            for c in case[arm]:
                print(f"{label:<40} {arm:<40} {c['table']:<17} "
                      f"bit-identical {c['bit_identical']}/{c['entries']} "
                      f"({100 * c['bit_identical_fraction']:.2f}%)  max ULP {c['max_ulp']} "
                      f"at {c['max_ulp_at']}  max |d| {c['max_abs']:.2e} at {c['max_abs_at']}")


if __name__ == "__main__":
    main()
