#!/usr/bin/env python3
"""Controls that make the KDA parity fixture load-bearing.

A fixture proves nothing until the things it is meant to catch actually
break it. Each control below perturbs one decision the transcription makes
and asserts the fixture *notices*:

1. **the decay clamp is not an input** — applying the declared
   `gate_lower_bound` changes the output materially, so the choice not to
   apply it is a real decision and not a harmless no-op;
2. **q/k L2 normalisation is load-bearing** — it appears nowhere in the
   checkpoint's modeling file (it lives behind
   `use_qk_l2norm_in_kernel=True`), which makes it the easiest operation in
   the whole block to omit by accident;
3. **the recurrence is actually recurrent** — resetting the state mid-sequence
   diverges from the true trajectory, so the fixture is not passing on a
   degenerate path that any stateless implementation would reproduce.

    python scripts/kda_controls.py <checkpoint-dir> --layer 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

import kda_reference as ref
from kda_fixture import INPUT_SEED, load_layer

#: A control has fired when the relative change exceeds this. Far above f32
#: noise: these are meant to be obvious, and a control that only just trips
#: is not evidence the fixture would catch a real defect.
MIN_RELATIVE_CHANGE = 1e-3

#: Positions to run the controls at. Long enough that state has accumulated
#: — at one position several of these are indistinguishable.
CONTROL_POSITIONS = 32


def relative_change(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).norm() / a.norm().clamp_min(1e-12)).item()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--layer", type=int, default=0)
    args = ap.parse_args()

    w = load_layer(args.checkpoint, args.layer)
    H, D = w.num_heads, w.head_dim
    hidden = w.q_proj.shape[1]
    torch.manual_seed(INPUT_SEED)
    x = torch.randn(CONTROL_POSITIONS, hidden, dtype=torch.float32) * 0.02
    base = ref.kda_forward(x, w)

    failures = []

    def control(label, changed_out, changed_state=None):
        d_out = relative_change(base["output"], changed_out)
        d_state = relative_change(base["_state"], changed_state) if changed_state is not None else None
        fired = d_out > MIN_RELATIVE_CHANGE
        state_note = f", state Δ {d_state:.3e}" if d_state is not None else ""
        print(f"  [{'FIRED' if fired else 'SILENT'}] {label}: output Δ {d_out:.3e}{state_note}")
        if not fired:
            failures.append(label)

    print(f"controls on layer {args.layer} ({H}x{D}) at {CONTROL_POSITIONS} positions\n")

    # 1. The declared clamp, applied. The reference does NOT apply it; this
    #    shows what would change if a future reader "fixed" that.
    LOWER_BOUND = -5.0
    g_pre = (x @ w.f_a_proj.T) @ w.f_b_proj.T
    g_clamped = LOWER_BOUND * torch.sigmoid(
        w.a_log.float().view(H, 1).exp()
        * (g_pre.float().view(-1, H, D) + w.dt_bias.view(H, D))
    )
    clamped = dict(base)
    q = F.normalize(base["q_conv"].view(-1, H, D).float(), p=2, dim=-1)
    k = F.normalize(base["k_conv"].view(-1, H, D).float(), p=2, dim=-1)
    v = base["v_conv"].view(-1, H, D)
    out_c, state_c = ref.recurrent_kda(q, k, v, g_clamped, base["beta"])
    normed = out_c * torch.rsqrt(out_c.pow(2).mean(-1, keepdim=True) + ref.RMS_EPS)
    o_c = (normed * w.o_norm * torch.sigmoid(base["o_gate"].float())).reshape(-1, H * D) @ w.o_proj.T
    control("applying the declared gate_lower_bound changes the result", o_c, state_c)

    # 2. q/k L2 normalisation, omitted.
    for which in ("q", "k"):
        qq = base["q_conv"].view(-1, H, D).float() if which == "q" else base["q_norm"]
        kk = base["k_conv"].view(-1, H, D).float() if which == "k" else base["k_norm"]
        if which == "q":
            kk = base["k_norm"]
        else:
            qq = base["q_norm"]
        out_n, state_n = ref.recurrent_kda(qq, kk, v, base["g_decay"], base["beta"])
        normed = out_n * torch.rsqrt(out_n.pow(2).mean(-1, keepdim=True) + ref.RMS_EPS)
        o_n = (normed * w.o_norm * torch.sigmoid(base["o_gate"].float())).reshape(-1, H * D) @ w.o_proj.T
        control(f"omitting the {which} L2 normalisation changes the result", o_n, state_n)

    # 3. The recurrence is recurrent: reset the state halfway and the tail
    #    must diverge from the true trajectory.
    half = CONTROL_POSITIONS // 2
    tail_true = base["recurrent_out"][half:]
    out_reset, state_reset = ref.recurrent_kda(
        base["q_norm"][half:], base["k_norm"][half:], v[half:],
        base["g_decay"][half:], base["beta"][half:], state=None)
    d = relative_change(tail_true, out_reset)
    fired = d > MIN_RELATIVE_CHANGE
    print(f"  [{'FIRED' if fired else 'SILENT'}] resetting the state at t={half} diverges: Δ {d:.3e}")
    if not fired:
        failures.append("state reset control")

    print()
    if failures:
        print(f"{len(failures)} CONTROL(S) DID NOT FIRE: {failures}")
        print("the fixture would not catch the defect they stand for")
        return 1
    print("all controls fired — the fixture is load-bearing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
