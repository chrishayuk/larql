#!/usr/bin/env python3
"""Generate the tiny, COMMITTED `kda_oracle.json` fixture for
`exec::kda`'s own parity test.

P4c-4 regenerates this: `q_proj`/`k_proj`/`v_proj`/`o_proj` are now BF16 in
the executor (`KdaWeights`'s four widest operands, routed through the
row-parallel `FusedBf16` path — see `exec/kda.rs::matvec_bf16`), so the
oracle must compute from the SAME bf16-exact bits the Rust side reads,
not from arbitrary-precision f32 randoms. Everything else — convolution,
gates, recurrence, gated norm — stays plain f32, unchanged from the
fixture this replaces.

Same reasoning `kimi_mla_oracle_export.py` already used for MLA: the
arithmetic is identical at any width, small enough to commit is small
enough to actually get run, and Kimi's real geometry (32 heads x 128) is
its own separate env-gated real-weight gate
(`kimi_kda_layer_export.py`/`kimi_kda_layer_real.rs`,
`kimi_stack_export.py`/`stack_real.rs`) — proving indexing and stride at
scale, not the formula.

    python scripts/kda_oracle_export.py > \
        crates/larql-vindex/src/format/vindex3/opplan/exec/tests/kda_oracle.json

K3-REP-GATE-1 adds a SECOND arm, never an overwrite of the first: the same
geometry, seed and every non-gate operand, with Kimi-K3's FULL-RANK output
gate (`linear_attn_config.use_full_rank_gate: true`) — one `g_proj` of
`[H*D, hidden]` in place of the `g_a_proj`/`g_b_proj` pair. Every boundary
up to `recurrent_out` is therefore IDENTICAL between the two arms, which
the Rust side can use as a cross-arm control; only `o_gate`, `o_norm` and
`output` differ. The arm ships its named controls (`kda_reference.
GATE_MUTATIONS`) with measured deltas and asserts the gate band first.

    python scripts/kda_oracle_export.py --full-rank-gate > \
        crates/larql-vindex/src/format/vindex3/opplan/exec/tests/kda_oracle_full_rank_gate.json
"""

from __future__ import annotations

import argparse
import json
import sys

import torch

import kda_reference as ref

SEED = 20260827

#: Same geometry the fixture this replaces used — 2 heads x 4, hidden 6,
#: conv kernel 4 — kept for continuity; nothing about it is load-bearing,
#: `kda_parity.rs` reads geometry from the JSON rather than assuming it.
NUM_HEADS = 2
HEAD_DIM = 4
HIDDEN = 6
CONV_KERNEL = 4

#: The longest run; N=1 and N=2 take a causal prefix of this same
#: sequence, so a state-carry defect that only shows up past N=1 is not
#: hiding behind a differently-seeded shorter input.
MAX_POSITIONS = 8

#: Scale of the full-rank gate's rows. Chosen so the pre-activations sit
#: inside `GATE_BAND` on this fixture — checked, not hoped: the exporter
#: refuses to write an arm whose gate is saturated or near-constant.
FULL_RANK_GATE_SCALE = 1.5

#: The non-saturation band every gate arm must satisfy on its own values
#: (K3-REP-GATE-1 freeze, D10): |g| <= 4 keeps sigmoid inside
#: [0.018, 0.982]; every head must reach |g| >= 0.5 somewhere so the gate
#: is not a near-constant scale that every control could hide behind.
GATE_BAND = {"max_abs": 4.0, "min_head_max_abs": 0.5}

#: Floor under a control's relative-L2 delta on `output`; a control below
#: it is numerically inert on this fixture and must be reported as such,
#: never listed as caught.
CONTROL_FLOOR = 1e-3


def bf16_exact(t: torch.Tensor) -> torch.Tensor:
    """Round-trip through bf16 so Python computes from the SAME bits the
    Rust side reads back from a bf16 code unit — truncation is lossless
    only when the f32 value already has zero in its low 16 mantissa
    bits, which a real bf16-native checkpoint tensor has by construction
    and an arbitrary f32 random does not."""
    return t.detach().to(torch.bfloat16).to(torch.float32)


def lst(t: torch.Tensor) -> list:
    return [round(v, 8) for v in t.detach().flatten().tolist()]


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative L2 against the reference's own scale — a defect is judged
    against the magnitude of what it perturbs, not element by element."""
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def gate_band(o_gate: torch.Tensor, heads: int) -> dict:
    """Measure the band on `[T, H, D]` gate pre-activations; fail loudly
    outside it rather than write a blind fixture."""
    per_head = [float(o_gate[:, h, :].abs().max()) for h in range(heads)]
    measured = {"max_abs": float(o_gate.abs().max()), "per_head_max_abs": per_head}
    assert measured["max_abs"] <= GATE_BAND["max_abs"], f"gate saturated: {measured}"
    assert min(per_head) >= GATE_BAND["min_head_max_abs"], f"gate near-constant: {measured}"
    return {**measured, "limits": GATE_BAND}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-rank-gate",
        action="store_true",
        help="export the K3 full-rank output-gate arm instead of the low-rank one",
    )
    args = parser.parse_args()

    torch.manual_seed(SEED)
    H, D, hidden, K = NUM_HEADS, HEAD_DIM, HIDDEN, CONV_KERNEL

    w = ref.KdaWeights(
        q_proj=bf16_exact(torch.randn(H * D, hidden) * 0.3),
        k_proj=bf16_exact(torch.randn(H * D, hidden) * 0.3),
        v_proj=bf16_exact(torch.randn(H * D, hidden) * 0.3),
        q_conv1d=torch.randn(H * D, 1, K) * 0.3,
        k_conv1d=torch.randn(H * D, 1, K) * 0.3,
        v_conv1d=torch.randn(H * D, 1, K) * 0.3,
        f_a_proj=torch.randn(D, hidden) * 0.3,
        f_b_proj=torch.randn(H * D, D) * 0.3,
        g_a_proj=torch.randn(D, hidden) * 0.3,
        g_b_proj=torch.randn(H * D, D) * 0.3,
        b_proj=torch.randn(H, hidden) * 0.3,
        a_log=torch.rand(H) * 0.5,  # kept small: exp(A_log) scales the decay
        dt_bias=torch.randn(H * D) * 0.2,
        o_norm=torch.rand(D) * 0.5 + 0.75,  # near 1.0, never zero
        o_proj=bf16_exact(torch.randn(hidden, H * D) * 0.3),
    )
    x_full = torch.randn(MAX_POSITIONS, hidden) * 0.2

    # The full-rank arm draws its gate AFTER everything the low-rank arm
    # draws, so every shared operand and the input are bit-identical
    # across the two arms; the pair is then dropped, as K3's checkpoint
    # drops it.
    g_proj = None
    if args.full_rank_gate:
        g_proj = torch.randn(H * D, hidden) * FULL_RANK_GATE_SCALE
        w.g_a_proj = None
        w.g_b_proj = None

    weights = {
        name: lst(getattr(w, name))
        for name in ref.KdaWeights.__dataclass_fields__
        if getattr(w, name) is not None
    }
    if g_proj is not None:
        weights["g_proj"] = lst(g_proj)

    runs = {}
    for n in (1, 2, 8):
        x_n = x_full[:n]
        result = ref.kda_forward(x_n, w, g_proj=g_proj)
        runs[str(n)] = {
            "input": lst(x_n),
            "boundaries": {name: lst(result[name]) for name in ref.BOUNDARIES},
            "state": lst(result["_state"]),
            "conv_state": [lst(s) for s in result["_conv_state"]],
        }

    band = None
    controls = None
    if args.full_rank_gate:
        full = ref.kda_forward(x_full, w, g_proj=g_proj)
        # The band is measured and asserted BEFORE any control is scored.
        band = gate_band(full["o_gate"], H)
        controls = {}
        for mutation in ref.GATE_MUTATIONS:
            if mutation == "none":
                continue
            mutant = ref.kda_forward(x_full, w, g_proj=g_proj, mutation=mutation)
            deltas = {name: rel_l2(mutant[name], full[name]) for name in ("o_gate", "o_norm", "output")}
            inert = deltas["output"] < CONTROL_FLOOR
            print(
                f"control {mutation:36s} o_gate {deltas['o_gate']:.3e}  o_norm {deltas['o_norm']:.3e}"
                f"  output {deltas['output']:.3e}{'  INERT' if inert else ''}",
                file=sys.stderr,
            )
            controls[mutation] = {
                "delta_rel_l2": deltas,
                "inert_on_this_fixture": inert,
                # The mutant's own boundaries on the full run, so the Rust
                # mutant can be required to EQUAL the oracle's wrong answer
                # rather than merely differ from the right one.
                "boundaries": {name: lst(mutant[name]) for name in ("o_gate", "o_norm", "output")},
            }
        assert not any(c["inert_on_this_fixture"] for c in controls.values()), (
            "an inert control cannot be listed as a control"
        )

    fixture = {
        "_comment": (
            "Tiny KDA layer + oracle outputs. Generated by "
            "scripts/kda_oracle_export.py; q/k/v/o_proj are BF16-exact "
            "(P4c-4), everything else f32. See docs/glm5-flash-funnel.md "
            "4.15 for the operator's own provenance."
        ),
        "num_heads": H,
        "head_dim": D,
        "hidden": hidden,
        "conv_kernel": K,
        "rms_eps": ref.RMS_EPS,
        "seed": SEED,
        "weights": weights,
        "runs": runs,
    }
    if args.full_rank_gate:
        fixture["_comment"] = (
            "Tiny KDA layer with Kimi-K3's FULL-RANK output gate (K3-REP-GATE-1) "
            "+ oracle outputs and named controls. Generated by "
            "scripts/kda_oracle_export.py --full-rank-gate; shares every non-gate "
            "operand and the input with kda_oracle.json bit for bit."
        )
        fixture["output_gate_form"] = "full_rank"
        fixture["gate_band"] = band
        fixture["controls"] = controls
    print(json.dumps(fixture, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
