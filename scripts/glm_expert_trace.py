"""Real GLM-5.3-Flash layer-3 expert selection over a real token sequence.

**Design frozen before any result was looked at.**

Purpose: the reuse curve for GLM's 288-expert bank. The first residency
result settled that layout is not the problem — predicted selected bytes
and physical page-in agree at ratio 1.000 with zero unselected residency,
and `MADV_WILLNEED` moves 219.8 ms to 214.1 ms. What remains unknown is
hit rate, and hit rate is a property of the *routing sequence*, which
synthetic input cannot supply: 24 random residual-scale vectors select
108 distinct experts of 288 with a mean consecutive overlap of 0.65 of 8.

So this produces the trace from REAL hidden states — embed → layers 0-2 →
layer 3's attention and mHC sites → the vector its MoE actually routes on
— through the pinned upstream reference, and records per position:

    selected expert ids · selected weights · overlap with the previous
    token · reuse distance of each selected expert · cumulative unique

Two curves are derived from that one trace, and NEITHER is a policy:

  * **natural reuse** — of token t's 192 MiB, how much was among the
    experts used in the last k tokens?
  * **budgeted residency** — at 0.25 / 0.5 / 1 / 2 / 4 / 6.75 GiB
    allocated to this layer's bank, what fraction of selected bytes hit
    under LRU, and under an offline optimal (Bélády) upper bound?

The optimal arm is there so a poor LRU number can be told apart from a
sequence with no locality to exploit. No cache policy is tuned here.

**Scope.** One layer's routing, from one prompt bank. Every sparse layer
in GLM has identical geometry (288 experts, top-8, moe_intermediate 2048),
so the per-layer BYTE arithmetic transfers; whether the routing
*statistics* transfer across depth is a separate question this does not
answer, and the trace is emitted per layer so it can be asked later.
"""
import argparse, gc, json, os, resource, sys
import numpy as np
import torch

CKPT = "/Volumes/model-drive/models/GLM-5.3-Flash"

# One expert's three matrices at GLM's geometry, FP8: 3 x 2048 x 4096
# bytes. Derived, not assumed — asserted against the checkpoint below.
EXPERT_BYTES = 3 * 2048 * 4096


def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 3)


def load_prefix(ckpt, prefix, dtype=torch.float32, skip=None):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from glm_layer_oracle import load_layer
    return load_layer(ckpt, prefix, dtype, skip=skip)


def embed_rows(ckpt, ids):
    """Only the rows the prompt needs — the table is 1.27 GiB and this
    experiment needs a few hundred rows of it."""
    from safetensors import safe_open
    idx = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))
    name = "model.language_model.embed_tokens.weight"
    with safe_open(os.path.join(ckpt, idx["weight_map"][name]), framework="pt") as f:
        t = f.get_slice(name)
        return torch.stack([t[int(i) : int(i) + 1][0].float() for i in ids])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--layer", type=int, default=3, help="the sparse layer to trace")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--max-tokens", type=int, default=192)
    ap.add_argument("--out", required=True, help="trace JSON")
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from glm_layer_oracle import build_config, remap_checkpoint_to_module
    from transformers import AutoTokenizer
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextDecoderLayer

    tcfg = build_config(args.ckpt)
    tcfg._attn_implementation = "eager"

    prompt = args.prompt or (
        "The physical cost of running a large mixture-of-experts model on a "
        "single machine depends less on the total size of the checkpoint than "
        "on which experts each token actually selects, and on how much of that "
        "selection is already resident when the token arrives. "
        "In practice the routing is not uniform: some experts are chosen far "
        "more often than others, and consecutive tokens frequently reuse them. "
    ) * 3
    tok = AutoTokenizer.from_pretrained(args.ckpt)
    ids = tok(prompt, return_tensors="pt")["input_ids"][0][: args.max_tokens]
    n = len(ids)
    print(f"prompt: {n} real tokens")

    x = embed_rows(args.ckpt, ids).unsqueeze(0)
    streams = x.unsqueeze(2).expand(-1, -1, tcfg.hc_mult, -1).contiguous()
    print(f"streams: {tuple(streams.shape)}")

    mask = torch.ones(1, n, dtype=torch.bool)
    for li in range(args.layer + 1):
        prefix = f"model.language_model.layers.{li}"
        # The traced layer's 288-expert bank is deliberately NOT loaded:
        # 6.78 GiB of FP8 is 27 GB widened, and the routing needs only
        # `mlp.gate`. Everything before it is a dense layer and loads
        # whole.
        skip = (lambda n: ".mlp.experts." in n or ".mlp.shared_experts." in n) \
            if li == args.layer else None
        print(f"  layer {li}: loading (peak RSS {rss():.1f} GiB) …", flush=True)
        sd = remap_checkpoint_to_module(load_prefix(args.ckpt, prefix, skip=skip))
        print(f"  layer {li}: {len(sd)} tensors, "
              f"{sum(v.numel() * v.element_size() for v in sd.values()) / 2**30:.2f} GiB "
              f"(peak RSS {rss():.1f} GiB)", flush=True)
        layer = Glm5NextTextDecoderLayer(tcfg, li).to(torch.float32)
        if li == args.layer:
            # The traced layer runs only as far as its MoE's INPUT — the
            # routing is what this experiment is about, and running the
            # 288-expert branch would cost 6.8 GiB for a value nothing
            # here reads.
            missing, _ = layer.load_state_dict(sd, strict=False)
            gate_w = sd["mlp.gate.weight"]
            gate_b = sd["mlp.gate.e_score_correction_bias"]
            layer.eval()
            with torch.no_grad():
                residual = streams
                post, comb, h = layer.attn_hc(streams)
                h = layer.input_layernorm(h)
                h, _, _ = layer.self_attn(
                    hidden_states=h,
                    attention_mask=mask,
                    position_ids=None,
                    past_key_values=__import__(
                        "transformers.cache_utils", fromlist=["DynamicCache"]
                    ).DynamicCache(config=tcfg),
                    use_cache=True,
                    position_embeddings=None,
                    prev_topk_indices=None,
                )
                h = post.unsqueeze(-1) * h.unsqueeze(-2) + torch.matmul(
                    comb.transpose(-1, -2), residual
                )
                _, _, h = layer.ffn_hc(h)
                moe_input = layer.post_attention_layernorm(h)
            break
        layer.load_state_dict(sd, strict=True)
        del sd
        layer.eval()
        with torch.no_grad():
            streams, _ = layer(hidden_states=streams, attention_mask=mask)
        del layer
        gc.collect()
        print(f"  layer {li} done   |streams| {streams.norm():.4f}   "
              f"peak RSS {rss():.1f} GiB", flush=True)

    # ── Routing, from the reference's own router math ──
    with torch.no_grad():
        h = moe_input.view(-1, tcfg.hidden_size)
        scores = torch.sigmoid(h.float() @ gate_w.float().T)
        chosen = torch.topk(scores + gate_b, tcfg.num_experts_per_tok, dim=-1).indices
        weights = scores.gather(1, chosen)

    trace = []
    seen_last = {}
    unique = set()
    for t in range(n):
        sel = sorted(int(i) for i in chosen[t])
        prev = set(int(i) for i in chosen[t - 1]) if t else set()
        dists = [t - seen_last[e] if e in seen_last else None for e in sel]
        for e in sel:
            seen_last[e] = t
        unique.update(sel)
        trace.append(
            {
                "position": t,
                "token_id": int(ids[t]),
                "selected": sel,
                "weights": [float(w) for w in weights[t]],
                "overlap_prev": len(set(sel) & prev),
                "reuse_distance": dists,
                "cumulative_unique": len(unique),
            }
        )

    meta = {
        "checkpoint": args.ckpt,
        "layer": args.layer,
        "tokens": n,
        "experts": tcfg.num_local_experts,
        "top_k": tcfg.num_experts_per_tok,
        "expert_bytes": EXPERT_BYTES,
        "bank_bytes": EXPERT_BYTES * tcfg.num_local_experts,
    }
    json.dump({"meta": meta, "trace": trace}, open(args.out, "w"))
    print(f"\nwrote {n} positions to {args.out}")
    print(f"  distinct experts touched: {len(unique)} of {tcfg.num_local_experts}")
    print(f"  mean consecutive overlap: "
          f"{np.mean([r['overlap_prev'] for r in trace[1:]]):.2f} of {tcfg.num_experts_per_tok}")


if __name__ == "__main__":
    main()
