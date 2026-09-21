"""How large is the KDA layer's output next to the stream it joins?

KDA-BEHAV-1 prerequisite. `out_traj` is normalised by the KDA layer's own
output norm, so `out_traj = 0.10` means "10 % of what this layer
contributes" — not "10 % of the model's state". If the layer's
contribution is a small fraction of the stream it writes into, then the
output gate frozen at 0.10 is far stricter than any behavioural
requirement, and the sub-4-bpw question changes.

This measures the ratio on REAL tokens through the REAL mHC join, because
the earlier estimate used a unit-RMS synthetic input and a plain-norm
argument, and GLM does not join with a plain residual add.

    $LARQL_GLM_ORACLE/bin/python tools/oracles/glm5/kda_stream_ratio.py
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts"))
CKPT = "/Volumes/model-drive/models/GLM-5.3-Flash"


def main():
    from safetensors import safe_open

    from glm_layer_oracle import build_config, load_layer, remap_checkpoint_to_module, shard_map
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextDecoderLayer

    cfg = build_config(CKPT)
    cfg._attn_implementation = "eager"

    # Real tokens. Layer 0 is the FIRST layer, so the stream entering it is
    # the embedding itself — no other layer has to run for this to be a
    # real hidden state, which is what makes this measurement cheap.
    # NATURAL prompt tokens, not uniform-random vocabulary ids.
    #
    # Random ids sample the vocabulary uniformly; natural text does not.
    # Measured on this checkpoint: natural token ids have median ~1.1k
    # against a random median of ~58k, and their embedding rows carry
    # **0.705x** the RMS with a far wider spread (min 0.0022 vs 0.0065).
    # KDA is scale-sensitive, so that is a different operating point, and
    # the roster's state margins are ~1 %.
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(os.path.join(CKPT, "tokenizer.json"))
    prompts = [
        "The capital of France is Paris, and the Seine runs through it.",
        "def quicksort(a):\n    if len(a) <= 1: return a\n    p = a[0]",
        "In 1947, the partition of British India displaced millions of people.",
        "Photosynthesis converts light energy into chemical energy stored in glucose.",
        "She argued that the contract was void because consideration had failed.",
        "The derivative of sin(x) with respect to x is cos(x).",
        "Patient presents with a three-day history of fever and productive cough.",
        "Es war einmal ein kleines Maedchen, das lebte in einem Dorf am Waldrand.",
        "Interest rates rose sharply, and the yield curve inverted for the first time.",
        "To be, or not to be, that is the question: whether 'tis nobler in the mind",
        "SELECT customer_id, SUM(total) FROM orders GROUP BY customer_id HAVING SUM(total) > 1000;",
        "The mitochondrion is often called the powerhouse of the cell.",
    ]
    rows_n = 154880
    flat = [i for pr in prompts for i in tok.encode(pr).ids if i < rows_n]
    ids = torch.tensor([flat])
    print(f"natural-prompt bank: {len(prompts)} prompts, {len(flat)} tokens, median id {sorted(flat)[len(flat)//2]}")
    name = "model.language_model.embed_tokens.weight"
    shard = os.path.join(CKPT, shard_map(CKPT)[name])
    with safe_open(shard, framework="pt") as f:
        emb = f.get_slice(name)
        rows = torch.stack([torch.tensor(emb[int(i)]).float() for i in ids[0]])
    x = rows.unsqueeze(0)
    print(f"embedded {ids.shape[1]} real tokens: RMS {x.pow(2).mean().sqrt():.6f}")

    layer = Glm5NextTextDecoderLayer(cfg, 0).to(torch.float32)
    sd = remap_checkpoint_to_module(load_layer(CKPT, "model.language_model.layers.0", torch.float32))
    layer.load_state_dict(sd, strict=True)
    layer.eval()

    # mHC carries `hc_mult` streams, so the layer's input is [B, S, H, D].
    streams = x.unsqueeze(2).expand(-1, -1, cfg.hc_mult, -1).contiguous()

    grab = {}
    # The KDA sub-module's ACTUAL input, after input_layernorm and the mHC
    # attn site — this, not the raw embedding, is what the Metal arm's `x`
    # must match.
    def pre(_m, args, kwargs):
        t = args[0] if args else kwargs.get("hidden_states")
        grab["kda_in"] = t.detach().float()

    # `with_kwargs`: the decoder layer calls `self_attn(hidden_states=...)`,
    # so the positional tuple is empty.
    hp = layer.self_attn.register_forward_pre_hook(pre, with_kwargs=True)
    h = layer.self_attn.register_forward_hook(
        lambda _m, _i, o: grab.__setitem__("kda", (o[0] if isinstance(o, tuple) else o).detach().float())
    )
    with torch.no_grad():
        out = layer(hidden_states=streams, attention_mask=None)
    h.remove()
    hp.remove()
    out = out[0] if isinstance(out, tuple) else out

    kda = grab["kda"]
    kin = grab["kda_in"]
    rms = lambda t: t.pow(2).mean().sqrt().item()
    r_in = rms(kda) / rms(streams)
    r_out = rms(kda) / rms(out)
    print(f"  KDA sub-INPUT  RMS {rms(kin):.6f}   <- what the Metal arm's x must match")
    print(f"  KDA sub-output RMS {rms(kda):.6f}")
    print(f"  KDA out/in (its own boundary) {rms(kda) / rms(kin):.5f}")
    print(f"  layer input  streams RMS {rms(streams):.6f}  -> ratio {r_in:.5f}")
    print(f"  layer output streams RMS {rms(out):.6f}  -> ratio {r_out:.5f}")
    print("\n  an out_traj of X is this much of the stream the layer writes into:")
    for v in (0.10, 0.20, 0.35, 1.00):
        print(f"    out_traj {v:4.2f} -> {v * r_out * 100:7.4f}% of the output stream")
    out_dir = os.environ.get("KDA_BANK_OUT")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        # The REAL post-norm KDA inputs, flat little-endian f32 — the
        # on-manifold substrate the Rust arms must use instead of a
        # unit-RMS synthetic vector.
        kin.reshape(-1, kin.shape[-1]).numpy().astype("<f4").tofile(
            os.path.join(out_dir, "kda_input.f32")
        )
        per_tok = kin.reshape(-1, kin.shape[-1]).pow(2).mean(-1).sqrt()
        per_tok.numpy().astype("<f4").tofile(os.path.join(out_dir, "kda_input_rms.f32"))
        with open(os.path.join(out_dir, "meta.txt"), "w") as f:
            f.write(f"tokens {kin.shape[1]}\nhidden {kin.shape[-1]}\nrms_mean {rms(kin)}\n")
        q = torch.quantile(per_tok, torch.tensor([0.0, 0.5, 0.95, 1.0]))
        print(
            f"\n  wrote {kin.shape[1]} real KDA inputs to {out_dir}\n"
            f"  per-token input RMS: min {q[0]:.6f} median {q[1]:.6f} p95 {q[2]:.6f} max {q[3]:.6f}"
        )

    print(
        "\n  NOTE: a ratio, not a behavioural result. It says how much of the stream a\n"
        "  given layer-output error is; it does NOT say what the model tolerates. Only\n"
        "  logits can say that, and that needs the whole 305 GB remainder."
    )


if __name__ == "__main__":
    main()
