#!/usr/bin/env python3
"""Generate the tiny, COMMITTED `kimi_mla_oracle.json` fixture for
`exec::mla`'s own parity test.

Synthetic on purpose, same reasoning `kda_oracle.json` (2 heads x 4)
already established for the KDA operator: the arithmetic is identical at
any width, small enough to commit is small enough to actually get run,
and this operator's real-checkpoint width (32 heads, 512-wide latent) is
its own SEPARATE env-gated real-weight gate
(`kimi_mla_layer_export.py`/`kimi_mla_layer_real.rs`) — proving indexing
and stride at scale, not the formula.

    python scripts/kimi_mla_oracle_export.py > \
        crates/larql-vindex/src/format/vindex3/opplan/exec/tests/kimi_mla_oracle.json

K3-REP-GATE-1 adds a SECOND arm, never an overwrite of the first: the same
geometry, seed, every operand and the input, plus Kimi-K3's OUTPUT GATE
(`mla_use_output_gate: true`) — one `g_proj` of `[Hq*v_head_dim, hidden]`
read from the block input, `sigmoid`, multiplied into the aggregated
value before `o_proj` (`modeling_kimi_linear.py` L470-472). Every
boundary up to `attn_value` is IDENTICAL between the arms; the gated arm
adds `output_gate` and `gated_value` and changes `output`. `hidden` (7)
differs from `Hq*v_head_dim` (10) on purpose, so a gate applied after
`o_proj` cannot even be expressed here. The arm ships its named controls
(`kimi_mla_reference.GATE_MUTATIONS`) with measured deltas and asserts
the gate band first.

    python scripts/kimi_mla_oracle_export.py --output-gate > \
        crates/larql-vindex/src/format/vindex3/opplan/exec/tests/kimi_mla_oracle_output_gate.json
"""

from __future__ import annotations

import argparse
import json
import sys

import torch

import kimi_mla_reference as ref

SEED = 20260828
POSITIONS = 3

#: Small enough to commit, large enough that heads, nope/rope split and
#: the asymmetric v_head_dim are all genuinely exercised. Every width
#: below is DIFFERENT from every other (`hidden` included) so a
#: transposed axis or swapped slice is NOT invisible here the way
#: `kda_oracle.json`'s own doc warns a too-small fixture can be — see
#: `mla_parity.rs::geometry_matches_kimis_ratios_not_a_symmetric_placeholder`.
GEOMETRY = ref.MlaGeometry(
    num_heads=2, kv_lora_rank=6, qk_nope_head_dim=3, qk_rope_head_dim=4, v_head_dim=5,
)
HIDDEN = 7

#: Scale of the output gate's rows — chosen so the pre-activations sit
#: inside `GATE_BAND` on this fixture, and checked rather than hoped.
OUTPUT_GATE_SCALE = 1.5

#: The non-saturation band (K3-REP-GATE-1 freeze, D10): |g| <= 4 keeps
#: sigmoid inside [0.018, 0.982]; every head reaches |g| >= 0.5 somewhere.
GATE_BAND = {"max_abs": 4.0, "min_head_max_abs": 0.5}

#: Floor under a control's relative-L2 delta on `output`; below it the
#: control is inert on this fixture and is reported, never listed as caught.
CONTROL_FLOOR = 1e-3


def lst(t: torch.Tensor) -> list:
    return [round(v, 8) for v in t.detach().flatten().tolist()]


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def gate_band(pre: torch.Tensor, heads: int, v_hd: int) -> dict:
    """Band on the `[T, Hq*v_head_dim]` pre-activations, per head."""
    per_head = [float(pre[:, h * v_hd:(h + 1) * v_hd].abs().max()) for h in range(heads)]
    measured = {"max_abs": float(pre.abs().max()), "per_head_max_abs": per_head}
    assert measured["max_abs"] <= GATE_BAND["max_abs"], f"gate saturated: {measured}"
    assert min(per_head) >= GATE_BAND["min_head_max_abs"], f"gate near-constant: {measured}"
    return {**measured, "limits": GATE_BAND}


def per_position(result: dict, name: str, positions: int) -> list:
    if name == "attn_weights":
        return [lst(result[name][:, p, : p + 1]) for p in range(positions)]
    return [lst(result[name][p]) for p in range(positions)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-gate",
        action="store_true",
        help="export the K3 output-gate arm instead of Kimi Linear's ungated one",
    )
    args = parser.parse_args()

    torch.manual_seed(SEED)
    g = GEOMETRY
    w = ref.MlaWeights(
        q_proj=torch.randn(g.num_heads * g.q_head_dim, HIDDEN) * 0.3,
        kv_a_proj=torch.randn(g.kv_lora_rank + g.qk_rope_head_dim, HIDDEN) * 0.3,
        kv_a_norm=torch.rand(g.kv_lora_rank) * 0.5 + 0.75,  # near 1.0, never zero
        kv_b_proj=torch.randn(g.num_heads * (g.qk_nope_head_dim + g.v_head_dim), g.kv_lora_rank) * 0.3,
        o_proj=torch.randn(HIDDEN, g.num_heads * g.v_head_dim) * 0.3,
    )
    x = torch.randn(POSITIONS, HIDDEN) * 0.2

    # The gated arm draws its gate AFTER everything the ungated arm draws,
    # so every shared operand and the input are bit-identical across arms.
    g_proj = None
    if args.output_gate:
        g_proj = torch.randn(g.num_heads * g.v_head_dim, HIDDEN) * OUTPUT_GATE_SCALE
    result = ref.mla_forward(x, w, g, g_proj=g_proj)

    fixture = {
        "num_heads": g.num_heads,
        "kv_lora_rank": g.kv_lora_rank,
        "qk_nope_head_dim": g.qk_nope_head_dim,
        "qk_rope_head_dim": g.qk_rope_head_dim,
        "v_head_dim": g.v_head_dim,
        "hidden": HIDDEN,
        "kv_a_norm_eps": ref.KV_A_NORM_EPS,
        "weights": {
            "q_proj": lst(w.q_proj), "kv_a_proj": lst(w.kv_a_proj),
            "kv_a_norm": lst(w.kv_a_norm), "kv_b_proj": lst(w.kv_b_proj),
            "o_proj": lst(w.o_proj),
        },
        "positions": POSITIONS,
        "input": [lst(x[p]) for p in range(POSITIONS)],
        # Per-position boundaries. `attn_weights`/`attn_value`/`output` at
        # position p depend on positions [0..=p] (causal) — sliced from
        # the full-sequence run, exactly what step-by-step KV-cache
        # decoding must reproduce.
        "boundaries": {
            "q_proj": [lst(result["q_proj"][p]) for p in range(POSITIONS)],
            "compressed_kv": [lst(result["compressed_kv"][p]) for p in range(POSITIONS)],
            "kv_a_normed": [lst(result["kv_a_normed"][p]) for p in range(POSITIONS)],
            "kv_b": [lst(result["kv_b"][p]) for p in range(POSITIONS)],
            # weights[h, p, 0..=p] flattened head-major, causal row only.
            "attn_weights": [
                lst(result["attn_weights"][:, p, : p + 1]) for p in range(POSITIONS)
            ],
            "attn_value": [lst(result["attn_value"][p]) for p in range(POSITIONS)],
            "output": [lst(result["output"][p]) for p in range(POSITIONS)],
        },
    }
    if args.output_gate:
        fixture["weights"]["g_proj"] = lst(g_proj)
        fixture["output_gate"] = True
        fixture["boundaries"] = {
            name: per_position(result, name, POSITIONS) for name in ref.GATED_BOUNDARIES
        }
        # The band is measured and asserted BEFORE any control is scored.
        fixture["gate_band"] = gate_band((x @ g_proj.T).float(), g.num_heads, g.v_head_dim)
        controls = {}
        for mutation in ref.GATE_MUTATIONS:
            if mutation == "none":
                continue
            mutant = ref.mla_forward(x, w, g, g_proj=g_proj, mutation=mutation)
            named = ("output_gate", "gated_value", "output")
            deltas = {name: rel_l2(mutant[name], result[name]) for name in named}
            inert = deltas["output"] < CONTROL_FLOOR
            print(
                f"control {mutation:34s} output_gate {deltas['output_gate']:.3e}"
                f"  gated_value {deltas['gated_value']:.3e}  output {deltas['output']:.3e}"
                f"{'  INERT' if inert else ''}",
                file=sys.stderr,
            )
            controls[mutation] = {
                "delta_rel_l2": deltas,
                "inert_on_this_fixture": inert,
                # The mutant's own boundaries, so the Rust mutant can be
                # required to EQUAL the oracle's wrong answer rather than
                # merely differ from the right one.
                "boundaries": {name: per_position(mutant, name, POSITIONS) for name in named},
            }
        assert not any(c["inert_on_this_fixture"] for c in controls.values()), (
            "an inert control cannot be listed as a control"
        )
        fixture["controls"] = controls
    print(json.dumps(fixture, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
