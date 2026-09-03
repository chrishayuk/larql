#!/usr/bin/env python3
"""HF reference oracle for OLMo-2 — the real-checkpoint witness for
post-norm execution (VINDEX3 wave 13).

Three taps rather than one. Final logits alone say only that something
is wrong; the earlier taps say WHERE, and the middle one is the tap that
exercises the post-norm causal position on real weights:

  * embedding      — the residual stream BEFORE any operator runs. Proves
                     the container reproduces the checkpoint's own
                     starting state.
  * layer 0 output — the first layer's residual output. Under post-norm
                     placement this is `h + post_ffn_norm(mlp(h_attn))`
                     where `h_attn = h + post_attn_norm(attn(h))` and the
                     sublayers read the RAW residual. A build that
                     normalised the INPUT instead lands here first.
  * logits         — the end-to-end claim.

Precision is a decision, not a default: fp32 throughout, on a checkpoint
that ships fp32, so nothing in the comparison is a rounding artefact of
the oracle's own making.

Self-checks before anything is emitted, on the principle that a reference
must be shown consistent before it is used as an authority:

  * determinism   — a second forward is bitwise identical;
  * placement     — the captured layer-0 output is reproduced by
                    recomputing the post-norm program from the captured
                    hidden states, and is NOT reproduced by the pre-norm
                    program. If both reproduce it, this checkpoint cannot
                    tell the placements apart and is useless as a witness.

Usage:
  python3 scripts/olmo2_reference_oracle.py ~/chris-models/OLMo-2-0425-1B \
      --out ~/chris-models/oracles/olmo2-1b.json
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

# Fixed, so the oracle and the parity test cannot drift on the input.
TOKENS = [100257, 510, 3823, 1917, 374, 264, 2361]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()
    ids = torch.tensor([TOKENS], dtype=torch.long)

    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
        again = model(ids, output_hidden_states=True)

    assert torch.equal(out.logits, again.logits), "reference is not deterministic"

    hidden = [h[0].to(torch.float32) for h in out.hidden_states]
    embedding, layer0 = hidden[0], hidden[1]

    # Placement self-check, recomputed from the captured embedding using
    # the layer's own weights. `decoder` is the reference module.
    layer = model.model.layers[0]
    with torch.no_grad():
        h = embedding.unsqueeze(0)
        seq = h.shape[1]
        position_ids = torch.arange(seq).unsqueeze(0)
        rotary = model.model.rotary_emb(h, position_ids)
        # A CAUSAL mask, explicitly. Passing `None` to eager attention
        # lets every position attend to every other, which is a different
        # model — and the first version of this script did exactly that,
        # which is what the self-check below caught.
        causal = torch.triu(torch.full((seq, seq), float("-inf")), diagonal=1)[None, None]

        def attend(x):
            return layer.self_attn(
                hidden_states=x, position_embeddings=rotary, attention_mask=causal
            )[0]

        # Post-norm (what OLMo-2 does): each sublayer reads the RAW
        # residual and its output is normalised before the add.
        post = h + layer.post_attention_layernorm(attend(h))
        post = post + layer.post_feedforward_layernorm(layer.mlp(post))
        # Pre-norm: the same two norm weights conditioning the INPUTS
        # instead. Same parameters, different program.
        pre = h + attend(layer.post_attention_layernorm(h))
        pre = pre + layer.mlp(layer.post_feedforward_layernorm(pre))

    post_err = (post[0] - layer0).abs().max().item()
    pre_err = (pre[0] - layer0).abs().max().item()
    assert post_err < 1e-3, f"post-norm recomputation does not reproduce layer 0 ({post_err})"
    assert pre_err > 100 * max(post_err, 1e-6), (
        f"this checkpoint cannot tell the placements apart "
        f"(post {post_err}, pre {pre_err}) — it is not a witness"
    )

    record = {
        "model": args.model,
        "tokens": TOKENS,
        "dtype": "float32",
        "self_checks": {
            "deterministic": True,
            "post_norm_recomputation_max_abs_err": post_err,
            "pre_norm_recomputation_max_abs_err": pre_err,
        },
        "embedding": embedding.tolist(),
        "layer0_output": layer0.tolist(),
        "logits": out.logits[0].to(torch.float32).tolist(),
    }
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(record))
    print(f"post-norm recomputation err {post_err:.3e}; pre-norm {pre_err:.3e}")
    print(f"wrote {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
