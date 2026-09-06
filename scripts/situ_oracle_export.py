#!/usr/bin/env python3
"""Generate the committed `situ_oracle.json` fixture — K3-ACT-1's parity
oracle for SiTU-GLU.

    python3 scripts/situ_oracle_export.py > \
        crates/larql-compute/src/ffn/expert_weight/tests/situ_oracle.json

Two ARMS, both of them legitimate configurations of the reference and
neither a control of the other:

    k3            beta = 4.0,  linear_beta = 25.0   (Kimi-K3's config.json)
    no_linear_cap beta = 4.0,  linear_beta = None   (the reference's own
                                                     supported form, L80)

Each arm carries its own controls with MEASURED deltas, so a control that
reads ~0 is visible in the fixture rather than absent from the test. The
`swiglu` control is the defect this rung exists to remove: it is what
`Activation::from_hf_name("situ") -> None -> unwrap_or(Silu)` computes.

The input grid is chosen, not sampled, so that every band of the gate's
pre-activation is populated and the export ASSERTS it — a fixture living
only in the saturated regime would make half the controls read zero and
the test would pass while measuring nothing.
"""

from __future__ import annotations

import json
import sys

import torch

import situ_reference as ref

#: Gate pre-activations, hand-picked to populate all three bands of
#: `ref.GATE_BANDS` on both signs, and to include the two exact points
#: where the function is analytically pinned: `gate = 0` (SiTU is exactly
#: 0 for every up) and `gate = beta` (tanh's argument is exactly 1).
GATE_VALUES = (
    0.0,
    0.25,
    -0.75,
    1.5,
    -2.0,
    4.0,
    -4.0,
    5.5,
    9.0,
    -12.0,
    30.0,
)

#: Up pre-activations. Deliberately NOT symmetric with the gate values —
#: a symmetric grid hides a branch swap, since swapping gate and up on a
#: symmetric grid is invisible (`feedback_fixture_symmetry_hides_
#: representation_bugs`). Includes values well past `linear_beta = 25`
#: so the up cap is actually exercised, and 0.0 so the product's zero is
#: reachable from the other side.
UP_VALUES = (
    0.0,
    1.0,
    -3.5,
    7.25,
    -18.0,
    25.0,
    40.0,
    -60.0,
)

ARMS = (
    ("k3", ref.K3_BETA, ref.K3_LINEAR_BETA),
    ("no_linear_cap", ref.K3_BETA, None),
)


def grid() -> tuple[torch.Tensor, torch.Tensor]:
    """The full cross product of `GATE_VALUES` x `UP_VALUES`, flattened.

    A cross product rather than zipped pairs: the up cap and the gate cap
    are independent, and a zipped diagonal would leave (saturated gate,
    tiny up) and (tiny gate, saturated up) untested — the two corners
    where a mis-placed cap is easiest to miss.
    """
    g, u = torch.meshgrid(
        torch.tensor(GATE_VALUES, dtype=torch.float32),
        torch.tensor(UP_VALUES, dtype=torch.float32),
        indexing="ij",
    )
    return g.reshape(-1), u.reshape(-1)


def assert_bands_populated(gate: torch.Tensor) -> dict[str, int]:
    """Every band of `ref.GATE_BANDS` carries at least one point, on both
    signs where the band admits a sign.

    This is the saturation gate. Without it a grid could drift into the
    regime where `tanh(g/beta) ~ ±1` everywhere, `sigmoid(g)` is 0 or 1,
    and `sigmoid_omitted`, `gate_cap_omitted` and `betas_swapped` all read
    ~0 against the reference — a passing test that measures nothing.
    """
    counts: dict[str, int] = {name: 0 for name, _, _ in ref.GATE_BANDS}
    signs: dict[str, set[int]] = {name: set() for name, _, _ in ref.GATE_BANDS}
    for value in gate.tolist():
        band = ref.band_of(value)
        counts[band] += 1
        if value != 0.0:
            signs[band].add(1 if value > 0 else -1)
    for name, count in counts.items():
        assert count > 0, f"gate band {name!r} is empty — the fixture is blind there"
    for name in ("transition", "saturated"):
        assert signs[name] == {1, -1}, (
            f"gate band {name!r} carries only {sorted(signs[name])} — the gate's "
            "softcap is not symmetric in its effect and one sign proves half of it"
        )
    return counts


def arm(name: str, beta: float, linear_beta: float | None) -> dict:
    """One arm: the reference values, then every control's delta."""
    gate, up = grid()
    out = ref.situ(gate, up, beta, linear_beta)

    # The concat form must agree elementwise, or the combine is not
    # elementwise and every executor in this build is shaped wrongly.
    concat = ref.situ_concat(torch.cat([gate, up]), beta, linear_beta)
    concat_delta = ref.rel_l2(concat, out)
    assert concat_delta == 0.0, (
        f"the concat form disagrees with the elementwise form by {concat_delta} — "
        "SiTU is not elementwise and this fixture's shape is wrong"
    )

    controls = {}
    for mutation in ref.MUTATIONS:
        if mutation == "none":
            continue
        mutated = ref.situ(gate, up, beta, linear_beta, mutation=mutation)
        controls[mutation] = {
            "rel_l2": ref.rel_l2(mutated, out),
            "max_abs": float((mutated - out).abs().max()),
        }

    # The oracle's self-check. Removing the gate's softcap leaves
    # `g*sigmoid(g)` = silu(g); on the arm that has no up cap either,
    # that IS SwiGLU. Two independently-coded mutation paths must
    # therefore land on the same number, and if they do not, one of them
    # is wrong and no Rust has been consulted yet.
    if linear_beta is None:
        agreement = abs(
            controls["gate_cap_omitted"]["rel_l2"] - controls["swiglu"]["rel_l2"]
        )
        assert agreement < 1e-6, (
            "with no up cap, `gate_cap_omitted` IS SwiGLU; they disagree by "
            f"{agreement}, so one of the two mutation paths is wrong"
        )

    # The defect this rung removes must be large at these parameters. If
    # SwiGLU were a good approximation here, the whole rung would be
    # arguing about nothing — so the fixture states the size.
    assert controls["swiglu"]["rel_l2"] > 0.1, (
        "the SwiGLU substitution reads "
        f"{controls['swiglu']['rel_l2']} on this grid — too small to be the "
        "defect this rung claims; the grid is in the wrong regime"
    )

    return {
        "beta": beta,
        "linear_beta": linear_beta,
        "gate": gate.tolist(),
        "up": up.tolist(),
        "out": out.tolist(),
        "controls": controls,
    }


def parameter_resolution() -> list[dict]:
    """`beta or 1.0` (L91) across every input a config can present.

    Exported as data rather than asserted only in Python, so the Rust
    side pins the same four answers against the same four inputs and the
    truthiness rule cannot be transcribed twice.
    """
    cases = []
    for declared_beta, declared_linear in (
        (None, None),
        (0.0, None),
        (4.0, None),
        (4.0, 25.0),
        (None, 25.0),
        (4.0, 0.0),
    ):
        beta, linear = ref.resolve_params(declared_beta, declared_linear)
        cases.append(
            {
                "declared_beta": declared_beta,
                "declared_linear_beta": declared_linear,
                "resolved_beta": beta,
                "resolved_linear_beta": linear,
            }
        )
    return cases


def main() -> int:
    gate, _ = grid()
    bands = assert_bands_populated(gate)

    fixture = {
        "_source": "moonshotai/Kimi-K3 modeling_kimi_linear.py L64-91 (SituAndMul)",
        "_generator": "scripts/situ_oracle_export.py, from scripts/situ_reference.py",
        "_precision": (
            "gate and up are upcast to f32 before the nonlinearity and the product "
            "is rounded once at the end (L77-78, L82). `bf16_throughout` measures "
            "what that upcast is worth at these parameters."
        ),
        "gate_bands": {
            "definition": [
                {"name": n, "min_abs": lo, "max_abs": None if hi == float("inf") else hi}
                for n, lo, hi in ref.GATE_BANDS
            ],
            "populated": bands,
        },
        "parameter_resolution": parameter_resolution(),
        "arms": {name: arm(name, beta, linear) for name, beta, linear in ARMS},
    }
    json.dump(fixture, sys.stdout, indent=1)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
