#!/usr/bin/env python3
"""Golden values for the hyper-connection residual topology (wave 17).

A faithful transcription of DeepSeek-V4-Flash's OWN reference:

  * `inference/kernel.py`, `hc_split_sinkhorn_kernel` — the split itself.
    The tilelang kernel is transcribed to torch step for step, including
    the two asymmetries a tidier would remove: `pre` carries `+ eps` and
    no factor, `post` carries a factor of 2 and no eps.
  * `inference/model.py`, `Block.hc_pre` / `Block.hc_post` — the
    reduction and the expansion around an ordinary sublayer.
  * `inference/model.py`, `ParallelHead.hc_head` — the head's OWN
    reduction, which runs no Sinkhorn at all.

The reference works in `[b, s, hc, d]`; this collapses `b` and `s` into
one leading axis `n`, so every `dim=2` there is `dim=1` here and nothing
else moves.

The fixture is deliberately awkward (see the module docs on the Rust
side): `hidden = 5` is neither a power of two nor equal to the stream
count, and the four streams differ in scale by 70x so that a wrong `pre`
vector cannot average its way to the right answer.

    scripts/hyper_connection_oracle_export.py > <the oracle json>
"""

from __future__ import annotations

import json
import sys

import torch
import torch.nn.functional as F

# The real checkpoint's own values (config.json, DeepSeek-V4-Flash).
HC_MULT = 4
SINKHORN_ITERS = 20
HC_EPS = 1e-6
NORM_EPS = 1e-6

# Deliberately not 4, not 8, not equal to HC_MULT.
HIDDEN = 5
POSITIONS = 3
SEED = 20260903

# Stream magnitudes spanning 70x, so the reduction cannot be right by
# accident and a transposed combination matrix cannot look plausible.
STREAM_SCALES = [0.1, 1.0, 3.0, 7.0]


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc: int,
    iters: int,
    eps: float,
):
    """`kernel.py::hc_split_sinkhorn_kernel`, step for step.

    `mixes` is `[n, (2 + hc) * hc]`. Returns `pre[n, hc]`,
    `post[n, hc]`, `comb[n, hc, hc]`.
    """
    n = mixes.shape[0]

    # pre[j] = sigmoid(mixes[j] * hc_scale[0] + hc_base[j]) + eps
    pre = torch.sigmoid(mixes[:, :hc] * hc_scale[0] + hc_base[:hc]) + eps
    # post[j] = 2 * sigmoid(mixes[j+hc] * hc_scale[1] + hc_base[j+hc])   -- no eps
    post = 2.0 * torch.sigmoid(mixes[:, hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc])
    # comb[j,k] reads flat index j*hc + k + 2*hc
    comb = mixes[:, 2 * hc :].reshape(n, hc, hc) * hc_scale[2] + hc_base[2 * hc :].reshape(hc, hc)

    # comb = comb.softmax(-1) + eps
    comb = torch.softmax(comb, dim=-1) + eps
    # comb = comb / (comb.sum(-2) + eps)
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)

    # The kernel spells the first row+column pass out before the loop, so
    # `iters` total passes means `iters - 1` here.
    for _ in range(iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)

    return pre, post, comb


def mix_projection(x: torch.Tensor, hc_fn: torch.Tensor, norm_eps: float):
    """Stage 1 — `Block.hc_pre`'s first three lines.

    The RMS epsilon here is the component's `norm_eps`, NOT `hc_eps`.
    """
    x_flat = x.flatten(1).float()
    rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + norm_eps)
    return F.linear(x_flat, hc_fn) * rsqrt


def reduce_streams(pre: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Stage 3 — `y = sum(pre[..., None] * x, dim=streams)`."""
    return torch.sum(pre.unsqueeze(-1) * x, dim=1)


def expand_streams(
    branch: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor
) -> torch.Tensor:
    """Stage 5 — `Block.hc_post`.

    `out[k] = post[k] * branch + sum_j comb[j, k] * residual[j]`. The
    output stream is comb's SECOND index; the sum runs over the first.
    """
    return post.unsqueeze(-1) * branch.unsqueeze(-2) + torch.sum(
        comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=1
    )


def hc_head(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    """`ParallelHead.hc_head` — NO Sinkhorn, scalar scale, `[hc, hc*d]` fn."""
    x_flat = x.flatten(1).float()
    rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + norm_eps)
    mixes = F.linear(x_flat, hc_fn) * rsqrt
    pre = torch.sigmoid(mixes * hc_scale + hc_base) + hc_eps
    return torch.sum(pre.unsqueeze(-1) * x, dim=1)


def branch_function(v: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Stage 4 stand-in: an ordinary sublayer on the REDUCED vector.

    A plain linear map is enough — the point of the stage is that it sees
    `[d]` and not `[hc, d]`, not what it computes.
    """
    return torch.tanh(F.linear(v, weight))


def main() -> None:
    torch.manual_seed(SEED)
    hc, d = HC_MULT, HIDDEN
    mix_hc = (2 + hc) * hc
    hc_dim = hc * d

    # Unequal streams, per-stream scale, plus a per-position tilt so no
    # two positions are the same problem.
    base = torch.randn(POSITIONS, hc, d)
    scales = torch.tensor(STREAM_SCALES).view(1, hc, 1)
    tilt = torch.linspace(0.5, 1.5, POSITIONS).view(POSITIONS, 1, 1)
    x = base * scales * tilt

    hc_fn = torch.randn(mix_hc, hc_dim) * 0.3
    hc_base = torch.randn(mix_hc) * 0.5
    hc_scale = torch.tensor([0.7, 1.3, 0.9])
    branch_weight = torch.randn(d, d) * 0.4

    head_fn = torch.randn(hc, hc_dim) * 0.3
    head_base = torch.randn(hc) * 0.5
    head_scale = torch.tensor(1.1)

    # --- the five stages, each recorded on its own -------------------
    mixes = mix_projection(x, hc_fn, NORM_EPS)
    pre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base, hc, SINKHORN_ITERS, HC_EPS)
    reduced = reduce_streams(pre, x)
    branched = branch_function(reduced, branch_weight)
    expanded = expand_streams(branched, x, post, comb)

    # --- controls: each must DIFFER from the reference ---------------
    _, _, comb_one_iter = hc_split_sinkhorn(mixes, hc_scale, hc_base, hc, 1, HC_EPS)
    # At the checkpoint's twenty iterations the split has converged to a
    # doubly-stochastic matrix, and the row/column order INSIDE the loop
    # stops being observable (measured: 9e-8, below f32 noise). At two it
    # is plainly observable (9e-2), so the low-iteration value is what
    # pins the order the reference actually spells.
    _, _, comb_two_iter = hc_split_sinkhorn(mixes, hc_scale, hc_base, hc, 2, HC_EPS)
    expanded_one_iter = expand_streams(branched, x, post, comb_one_iter)
    expanded_transposed = expand_streams(branched, x, post, comb.transpose(-1, -2))
    mixes_wrong_eps = mix_projection(x, hc_fn, HC_EPS if HC_EPS != NORM_EPS else 1e-3)

    # A single-stream reading of the same layer: h + f(h) on stream 0.
    single_stream = x[:, 0, :] + branch_function(x[:, 0, :], branch_weight)

    head_reduced = hc_head(x, head_fn, head_scale, head_base, NORM_EPS, HC_EPS)

    def flat(t: torch.Tensor) -> list:
        return [round(v, 9) for v in t.double().flatten().tolist()]

    doc = {
        "_comment": (
            "Hyper-connection residual topology oracle. Generated by "
            "scripts/hyper_connection_oracle_export.py, a torch transcription of "
            "DeepSeek-V4-Flash's own inference/kernel.py (hc_split_sinkhorn_kernel) "
            "and inference/model.py (Block.hc_pre, Block.hc_post, "
            "ParallelHead.hc_head). Shapes are [positions, streams, hidden] with the "
            "reference's b and s collapsed into one axis. Regenerate rather than "
            "hand-edit."
        ),
        "streams": hc,
        "hidden": d,
        "positions": POSITIONS,
        "mix_rows": mix_hc,
        "sinkhorn_iters": SINKHORN_ITERS,
        "hc_eps": HC_EPS,
        "norm_eps": NORM_EPS,
        "seed": SEED,
        "stream_scales": STREAM_SCALES,
        "weights": {
            "hc_fn": flat(hc_fn),
            "hc_base": flat(hc_base),
            "hc_scale": flat(hc_scale),
            "branch_weight": flat(branch_weight),
            "head_fn": flat(head_fn),
            "head_base": flat(head_base),
            "head_scale": float(head_scale),
        },
        "input": {"x": flat(x)},
        "stages": {
            "mix_projection": flat(mixes),
            "sinkhorn_pre": flat(pre),
            "sinkhorn_post": flat(post),
            "sinkhorn_comb": flat(comb),
            "reduced": flat(reduced),
            "branched": flat(branched),
            "expanded": flat(expanded),
        },
        "head": {"reduced": flat(head_reduced)},
        "controls": {
            "_comment": (
                "Every one of these MUST differ from its reference counterpart. A "
                "fixture on which they agree cannot see the property it is there to "
                "police."
            ),
            "comb_one_iteration": flat(comb_one_iter),
            "comb_two_iterations": flat(comb_two_iter),
            "expanded_one_iteration": flat(expanded_one_iter),
            "expanded_transposed_comb": flat(expanded_transposed),
            "mix_projection_wrong_eps": flat(mixes_wrong_eps),
            "single_stream_residual": flat(single_stream),
        },
    }
    json.dump(doc, sys.stdout, indent=1)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
