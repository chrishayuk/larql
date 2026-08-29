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
"""

from __future__ import annotations

import json

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


def lst(t: torch.Tensor) -> list:
    return [round(v, 8) for v in t.detach().flatten().tolist()]


def main() -> int:
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

    result = ref.mla_forward(x, w, g)

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
    print(json.dumps(fixture, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
