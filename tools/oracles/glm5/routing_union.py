"""How much does the frozen candidate roster inflate expert fetches?

KDA-BEHAV-1's cost is dominated by streaming the routed bank. If each arm
is traversed separately the bill is `arms x per-prompt bytes`. If the arms
are fanned through ONE traversal, the bill is the UNION of their selected
experts per layer — and because every arm is a perturbation of the same
hidden state, that union may be far smaller than the sum.

**That is measurable, so it is measured rather than assumed.** This runs
embed -> layers 0-2 -> layer 3's routing vector for every arm, perturbing
ONLY layer 0's four KDA projections, and reports the amplification.

    $LARQL_GLM_ORACLE/bin/python tools/oracles/glm5/routing_union.py
"""
import gc
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts"))
CKPT = "/Volumes/model-drive/models/GLM-5.3-Flash"
SPARSE = 3  # first sparse layer: 0-2 are dense

# The roster, frozen in the funnel doc before any behavioural measurement.
# (q, k, v, o) RTN widths; None = untouched BF16 authority.
ROSTER = [
    ("bf16-authority", None),
    ("uniform-8", (8, 8, 8, 8)),
    ("uniform-6", (6, 6, 6, 6)),
    ("uniform-4", (4, 4, 4, 4)),
    ("strict-4.25", (3, 4, 5, 5)),
    ("slightly-aggr-3.75", (3, 4, 4, 4)),
    ("aggressive-3.50", (3, 4, 4, 3)),
    ("D-probe-3.00", (2, 3, 4, 3)),
]
PROJ = ["q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight"]


def rtn(t: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-256 absmax symmetric round-to-nearest — the same reference
    quantiser the Metal arms use, so the two agree by construction."""
    flat = t.reshape(-1)
    pad = (-flat.numel()) % 256
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, dtype=flat.dtype)])
    blocks = flat.view(-1, 256)
    levels = float((1 << (bits - 1)) - 1)
    scale = blocks.abs().amax(-1, keepdim=True) / levels
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    out = (blocks / scale).round().clamp(-levels, levels) * scale
    return out.reshape(-1)[: t.numel()].view_as(t)


def main():
    from tokenizers import Tokenizer
    from transformers.cache_utils import DynamicCache
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextDecoderLayer

    from glm_expert_trace import embed_rows, load_prefix, rss
    from glm_layer_oracle import build_config, remap_checkpoint_to_module

    cfg = build_config(CKPT)
    cfg._attn_implementation = "eager"
    tok = Tokenizer.from_file(os.path.join(CKPT, "tokenizer.json"))
    prompt = (
        "The capital of France is Paris. Photosynthesis converts light energy into "
        "chemical energy. def quicksort(a): return a if len(a) <= 1 else None. "
        "In 1947 the partition of British India displaced millions of people."
    )
    ids = torch.tensor([i for i in tok.encode(prompt).ids if i < cfg.vocab_size])[:128]
    n = len(ids)
    print(f"prompt: {n} real tokens")
    x = embed_rows(CKPT, ids).unsqueeze(0)
    base_streams = x.unsqueeze(2).expand(-1, -1, cfg.hc_mult, -1).contiguous()
    mask = torch.ones(1, n, dtype=torch.bool)

    print("loading layers 0..3 once (layer 3 without its 288-expert bank) …", flush=True)
    sds, mods = {}, {}
    for li in range(SPARSE + 1):
        skip = (lambda nm: ".mlp.experts." in nm or ".mlp.shared_experts." in nm) if li == SPARSE else None
        sd = remap_checkpoint_to_module(load_prefix(CKPT, f"model.language_model.layers.{li}", skip=skip))
        layer = Glm5NextTextDecoderLayer(cfg, li).to(torch.float32)
        if li == SPARSE:
            layer.load_state_dict(sd, strict=False)
            gate_w, gate_b = sd["mlp.gate.weight"], sd["mlp.gate.e_score_correction_bias"]
        else:
            layer.load_state_dict(sd, strict=True)
        layer.eval()
        mods[li] = layer
        if li == 0:
            sds[0] = {p: sd[f"self_attn.{p}"].clone() for p in PROJ}
        del sd
        gc.collect()
        print(f"  layer {li} ready (RSS {rss():.1f} GiB)", flush=True)

    chosen_by_arm = {}
    for name, widths in ROSTER:
        with torch.no_grad():
            for j, p in enumerate(PROJ):
                src = sds[0][p]
                getattr(mods[0].self_attn, p.split(".")[0]).weight.copy_(
                    src if widths is None else rtn(src, widths[j])
                )
            streams = base_streams
            for li in range(SPARSE):
                streams, _ = mods[li](hidden_states=streams, attention_mask=mask)
            layer = mods[SPARSE]
            residual = streams
            post, comb, h = layer.attn_hc(streams)
            h = layer.input_layernorm(h)
            h, _, _ = layer.self_attn(
                hidden_states=h, attention_mask=mask, position_ids=None,
                past_key_values=DynamicCache(config=cfg), use_cache=True,
                position_embeddings=None, prev_topk_indices=None,
            )
            h = post.unsqueeze(-1) * h.unsqueeze(-2) + torch.matmul(comb.transpose(-1, -2), residual)
            _, _, h = layer.ffn_hc(h)
            moe_in = layer.post_attention_layernorm(h).view(-1, cfg.hidden_size)
            scores = torch.sigmoid(moe_in.float() @ gate_w.float().T)
            chosen_by_arm[name] = torch.topk(scores + gate_b, cfg.num_experts_per_tok, dim=-1).indices
        print(f"  {name:<20} routed (RSS {rss():.1f} GiB)", flush=True)

    k = cfg.num_experts_per_tok
    arms = [a for a, _ in ROSTER]
    print(f"\nlayer {SPARSE} routing, top-{k} of {cfg.n_routed_experts}, {n} tokens\n")
    print(f"{'arm':<20}{'agree w/ bf16':>15}{'mean |sel ∩ bf16|':>20}")
    ref = chosen_by_arm["bf16-authority"]
    for a in arms:
        c = chosen_by_arm[a]
        inter = [len(set(c[t].tolist()) & set(ref[t].tolist())) for t in range(n)]
        exact = sum(1 for t in range(n) if set(c[t].tolist()) == set(ref[t].tolist()))
        print(f"{a:<20}{exact / n * 100:>14.1f}%{sum(inter) / n:>19.2f}/{k}")

    unions = [len({int(i) for a in arms for i in chosen_by_arm[a][t]}) for t in range(n)]
    u = torch.tensor(unions, dtype=torch.float)
    q = torch.quantile(u, torch.tensor([0.0, 0.5, 0.95, 1.0]))
    print(
        f"\nUNION over all {len(arms)} arms, per token:"
        f"\n  min {q[0]:.0f}  median {q[1]:.0f}  p95 {q[2]:.0f}  max {q[3]:.0f}  (of {len(arms) * k} if disjoint)"
        f"\n  FETCH AMPLIFICATION vs one arm: mean {u.mean() / k:.2f}x   worst {q[3] / k:.2f}x"
        f"\n  vs traversing separately ({len(arms)}x): saving {(1 - u.mean() / (k * len(arms))) * 100:.1f}%"
    )


if __name__ == "__main__":
    main()
