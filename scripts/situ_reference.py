#!/usr/bin/env python3
"""Torch transcription of SiTU-GLU — Kimi-K3's FFN combine (K3-ACT-1).

Provenance, and it is unusually short because the function is:

    moonshotai/Kimi-K3  modeling_kimi_linear.py  L64-91

        class SituAndMul(nn.Module):
            def forward(self, x):
                d = x.shape[-1] // 2
                gate = x[..., :d].to(torch.float32)
                up   = x[..., d:].to(torch.float32)
                situ_a = self.beta * torch.tanh(gate / self.beta) * torch.sigmoid(gate)
                if self.linear_beta is not None:
                    up = self.linear_beta * torch.tanh(up / self.linear_beta)
                return (situ_a * up).to(x.dtype)

        ACT2FN["situ"] = SituAndMul

        def _get_situ_activation_params(config):
            beta = getattr(config, "activation_situ_beta", None)
            linear_beta = getattr(config, "activation_situ_linear_beta", None)
            return beta or 1.0, linear_beta

Three facts this file exists to carry, each of which a plausible
reimplementation gets wrong:

**The f32 upcast is part of the definition, not an implementation
choice.** L77-78 upcast BOTH branches before the nonlinearity and L82
rounds ONCE, at the end. On a bf16 checkpoint the softcap is therefore
computed in f32. `bf16_throughout` below measures what that is worth.

**`beta or 1.0` is Python truthiness.** A declared `0.0` — or `None`, or
an absent key — resolves to `1.0`, not to `0.0`. A transcription that
reads `0.0` literally divides by zero. `linear_beta` has NO such
fallback: absent means the up branch is untouched, which is a different
function and not `linear_beta = inf`.

**It degenerates to SwiGLU.** As `beta -> inf`, `beta*tanh(g/beta) -> g`
and `situ_a -> g*sigmoid(g) = silu(g)`; as `linear_beta -> inf` the up
cap vanishes. That is exactly why substituting SiLU produces a forward
pass that looks healthy, and why `swiglu` is the first control here.

The combine is ELEMENTWISE — the concat in the reference exists only
because HF's `ACT2FN` takes one tensor — so this file works on separate
`gate`/`up` tensors and `situ_concat` pins that the two agree.
"""

from __future__ import annotations

import torch

#: Kimi-K3's declared parameters, `config.json` `text_config`.
K3_BETA = 4.0
K3_LINEAR_BETA = 25.0


def resolve_params(
    beta: float | None,
    linear_beta: float | None,
) -> tuple[float, float | None]:
    """`_get_situ_activation_params` (L88-91), verbatim semantics.

    `beta or 1.0` — falsy (absent, None, 0.0) becomes 1.0. `linear_beta`
    passes through untouched, including a declared 0.0, which the
    reference does not normalise and which makes the up branch zero.
    """
    return (beta or 1.0), linear_beta


#: Named defects, for the controls. Each perturbs the real forward below
#: at exactly one point — never a hand-rolled copy — so the reference and
#: its controls cannot drift apart.
MUTATIONS = (
    "none",
    # Plain SwiGLU: both softcaps removed. THE defect on main — the
    # substitution `Activation::from_hf_name("situ") -> None ->
    # unwrap_or(Silu)` performs.
    "swiglu",
    # The gate's softcap removed, the up cap kept.
    "gate_cap_omitted",
    # The up cap removed, the gate cap kept. NOTE this is also a
    # LEGITIMATE configuration (`linear_beta` absent), so it is exported
    # as its own arm too; here it is the control for K3's parameters.
    "up_cap_omitted",
    # beta and linear_beta swapped. Catches a parameter-ordering bug that
    # every magnitude check passes.
    "betas_swapped",
    # `situ_a = beta*tanh(g/beta)` — the sigmoid dropped, leaving a pure
    # tanh GLU.
    "sigmoid_omitted",
    # NOTE there is deliberately no `tanh_omitted` here. Dropping the tanh
    # from `beta*tanh(g/beta)` leaves `beta*(g/beta) = g`, which is
    # `gate_cap_omitted` exactly — the same perturbation reached by a
    # second route, not a second control. An earlier draft carried both
    # and they read identical deltas on every arm, which is a redundant
    # control inflating the count. The identity is still USED, as the
    # oracle's own self-check: on the arm with no up cap,
    # `gate_cap_omitted` must equal `swiglu` (see the export).
    #
    # Both branches uncapped, the cap applied to the PRODUCT instead.
    "cap_on_product",
    # The f32 upcast dropped: everything in bf16. Measures whether L77-78
    # is observable at K3's betas. May read small; a small reading is
    # RECORDED as small, never dropped.
    "bf16_throughout",
)

#: Mutations that remove the gate branch's softcap.
_GATE_CAP_OFF = ("swiglu", "gate_cap_omitted", "cap_on_product")
#: Mutations that remove the up branch's softcap.
_UP_CAP_OFF = ("swiglu", "up_cap_omitted", "cap_on_product")


def situ(
    gate: torch.Tensor,
    up: torch.Tensor,
    beta: float,
    linear_beta: float | None,
    *,
    mutation: str = "none",
) -> torch.Tensor:
    """SiTU-GLU on separate gate/up tensors. `mutation` names one of
    `MUTATIONS`; `"none"` is the reference.

    `beta`/`linear_beta` arrive already resolved (`resolve_params`), so
    the truthiness rule lives in exactly one place.
    """
    if mutation not in MUTATIONS:
        raise ValueError(f"unknown mutation {mutation!r}")

    if mutation == "betas_swapped":
        beta, linear_beta = (linear_beta if linear_beta is not None else 1.0), beta

    work = torch.bfloat16 if mutation == "bf16_throughout" else torch.float32
    out_dtype = gate.dtype

    g = gate.to(work)
    u = up.to(work)

    # The gate branch: beta * tanh(g / beta) * sigmoid(g).
    cap = g if mutation in _GATE_CAP_OFF else torch.tanh(g / beta) * beta
    sig = torch.ones_like(g) if mutation == "sigmoid_omitted" else torch.sigmoid(g)
    situ_a = cap * sig

    # The up branch: linear_beta * tanh(u / linear_beta), or untouched.
    if linear_beta is not None and mutation not in _UP_CAP_OFF:
        u = torch.tanh(u / linear_beta) * linear_beta

    out = situ_a * u
    if mutation == "cap_on_product":
        out = torch.tanh(out / beta) * beta
    return out.to(out_dtype)


def situ_concat(
    gate_up: torch.Tensor,
    beta: float,
    linear_beta: float | None,
) -> torch.Tensor:
    """`SituAndMul.forward` on its own input shape — one tensor whose last
    axis is `[gate | up]`, split at the halfway point (L76-78).

    Kept beside the elementwise form so the export can pin that they
    agree: if they ever disagree, the combine is not elementwise and every
    executor in this build is shaped wrongly for it.
    """
    d = gate_up.shape[-1] // 2
    return situ(gate_up[..., :d], gate_up[..., d:], beta, linear_beta)


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative L2 of `a` against `b`, normalised by `b`'s own scale.

    Elementwise relative error reads huge wherever the reference is near
    zero — and SiTU is exactly zero at `gate = 0` for every `up` — so the
    row scale is the denominator (`feedback_dot_product_error_metric_
    normalise_by_row_scale`).
    """
    a32, b32 = a.to(torch.float32), b.to(torch.float32)
    denom = float(torch.linalg.vector_norm(b32))
    if denom == 0.0:
        return float(torch.linalg.vector_norm(a32 - b32))
    return float(torch.linalg.vector_norm(a32 - b32) / denom)


#: The band the fixture's gate values must straddle for the controls to be
#: readable. tanh(g/4) saturates past |g| ~ 12 and sigmoid(g) past |g| ~ 6;
#: in the saturated regime SiTU -> ±beta·step(g)·u and several mutations
#: above go numerically invisible. A fixture that lives only in that
#: regime is a blind instrument, so the export ASSERTS all three zones are
#: populated (`feedback_saturated_softmax_blinds_an_oracle`).
GATE_BANDS = (
    ("near_linear", 0.0, 1.0),
    ("transition", 1.0, 6.0),
    ("saturated", 6.0, float("inf")),
)


def band_of(value: float) -> str:
    """Which of `GATE_BANDS` a gate pre-activation falls in, by |value|."""
    mag = abs(value)
    for name, lo, hi in GATE_BANDS:
        if lo <= mag < hi:
            return name
    raise AssertionError(f"unreachable: {value} in no band")
