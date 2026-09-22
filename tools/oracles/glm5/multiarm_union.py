"""MULTIARM-1: does union-of-experts stay cheap with DEPTH?

The layer-3 result (funnel 8.2.2l) measured 1.15x fetch amplification for
eight candidate arms at ONE sparse layer on ONE prompt. If that holds
across the stack, REPRESENT can evaluate a population of nearby physical
candidates for roughly the I/O of evaluating one — which is infrastructure,
not a KDA detail. If it grows with depth, the whole fan-through design
needs rethinking before anything is built on it.

**Selective expert fetch is the enabling primitive.** `Glm5NextTextExperts`
stores `gate_up_proj[n_experts, 2*inter, hidden]`, so a COMPACT module
holding only the union, with top-k indices remapped to local positions,
reproduces the full bank **bit-equally** (verified before this was
written). That is what makes a 288-expert layer runnable without
allocating 288 experts.

    $LARQL_GLM_ORACLE/bin/python tools/oracles/glm5/multiarm_union.py --max-layer 12
"""
import argparse
import copy
import gc
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts"))
CKPT = "/Volumes/model-drive/models/GLM-5.3-Flash"

ROSTER = [
    ("bf16", None), ("q8", (8, 8, 8, 8)), ("q6", (6, 6, 6, 6)), ("q4", (4, 4, 4, 4)),
    ("strict-4.25", (3, 4, 5, 5)), ("aggr-3.75", (3, 4, 4, 4)),
    ("aggr-3.50", (3, 4, 4, 3)), ("probe-3.00", (2, 3, 4, 3)),
]
PROJ = ["q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight"]


def rtn(t, bits):
    flat = t.reshape(-1)
    pad = (-flat.numel()) % 256
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, dtype=flat.dtype)])
    b = flat.view(-1, 256)
    lv = float((1 << (bits - 1)) - 1)
    s = b.abs().amax(-1, keepdim=True) / lv
    s = torch.where(s > 0, s, torch.ones_like(s))
    return ((b / s).round().clamp(-lv, lv) * s).reshape(-1)[: t.numel()].view_as(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-layer", type=int, default=12)
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--out", default=None)
    ap.add_argument("--bank", default=None, help="path to a frozen prompt bank JSON")
    ap.add_argument("--prompt-index", type=int, default=None)
    ap.add_argument("--logits", action="store_true",
                    help="run the head and score behaviour (requires --max-layer 44)")
    args = ap.parse_args()

    # FAIL LOUD ON A MISSING CHECKPOINT.
    #
    # `from_pretrained` treats an absent local path as a HuggingFace repo
    # id, so an unmounted model drive surfaces as `HFValidationError` deep
    # in `huggingface_hub` rather than as "the weights are gone". On
    # 2026-09-07 that turned a 20-prompt holdout run into 18 identical
    # tracebacks and zero observations — recoverable only because the log
    # was read. A run that produces no output must be distinguishable from
    # a run that found nothing.
    for required in ("config.json", "model.safetensors.index.json"):
        if not os.path.isfile(os.path.join(CKPT, required)):
            sys.exit(
                f"CHECKPOINT UNAVAILABLE: {os.path.join(CKPT, required)} is not readable.\n"
                f"  The model drive is probably not mounted. Nothing was measured; no bank\n"
                f"  has been spent. Remount and re-run."
            )


    from tokenizers import Tokenizer
    from transformers.cache_utils import DynamicCache
    from transformers.models.glm5_next.modeling_glm5_next import (
        Glm5NextTextDecoderLayer, Glm5NextTextExperts, Glm5NextTextMLP, Glm5NextTextTopkRouter,
    )

    from glm_expert_trace import embed_rows, load_prefix, rss
    from glm_layer_oracle import build_config, dequant_fp8, remap_checkpoint_to_module, shard_map

    cfg = build_config(CKPT)
    cfg._attn_implementation = "eager"
    # A layer built with ONE expert allocates a tiny MoE; its router and
    # bank are constructed separately at full width and the MoE is run by
    # hand, following `Glm5NextTextMoE.forward` exactly.
    cfg1 = copy.deepcopy(cfg)
    cfg1.num_local_experts = cfg1.n_routed_experts = 1

    tok = Tokenizer.from_file(os.path.join(CKPT, "tokenizer.json"))
    if args.bank:
        bank = json.load(open(args.bank))
        text = bank["prompts"][args.prompt_index or 0]["text"]
        cat = bank["prompts"][args.prompt_index or 0]["category"]
        print(f"bank {bank['name']} [{args.prompt_index or 0}] ({cat})", flush=True)
    else:
        text = "The capital of France is Paris and the Seine runs through it."
        cat = "adhoc"
    ids = torch.tensor([i for i in tok.encode(text).ids if i < cfg.vocab_size][: args.tokens])
    n = len(ids)
    print(f"prompt {n} tokens, arms {len(ROSTER)}, layers 0..{args.max_layer}", flush=True)
    x = embed_rows(CKPT, ids).unsqueeze(0)
    base = x.unsqueeze(2).expand(-1, -1, cfg.hc_mult, -1).contiguous()
    mask = torch.ones(1, n, dtype=torch.bool)

    arms = [a for a, _ in ROSTER]
    streams = {a: base.clone() for a in arms}
    caches = {a: DynamicCache(config=cfg) for a in arms}
    prev_topk = {a: None for a in arms}
    smap = shard_map(CKPT)
    census = []

    for li in range(args.max_layer + 1):
        prefix = f"model.language_model.layers.{li}"
        sd = remap_checkpoint_to_module(
            load_prefix(CKPT, prefix, skip=lambda nm: ".mlp.experts." in nm)
        )
        layer = Glm5NextTextDecoderLayer(cfg1, li).to(torch.float32)
        sparse = "mlp.gate.weight" in sd
        # `strict=False` still RAISES on a shape mismatch, and the
        # one-expert layer's router is [1, hidden] against the
        # checkpoint's [288, hidden]. The MoE is run separately at full
        # width, so its keys are withheld from the layer entirely.
        layer.load_state_dict(
            {k: v for k, v in sd.items() if not (sparse and k.startswith("mlp."))},
            strict=not sparse,
        )
        layer.eval()
        if sparse:
            router = Glm5NextTextTopkRouter(cfg).to(torch.float32).eval()
            with torch.no_grad():
                router.weight.copy_(sd["mlp.gate.weight"])
                router.e_score_correction_bias.copy_(sd["mlp.gate.e_score_correction_bias"])
            shared = Glm5NextTextMLP(
                cfg, intermediate_size=cfg.moe_intermediate_size * cfg.n_shared_experts
            ).to(torch.float32).eval()
            shared.load_state_dict(
                {k.split("mlp.shared_experts.")[1]: v for k, v in sd.items()
                 if "mlp.shared_experts." in k}, strict=True
            )
        if li == 0:
            pristine = {p: sd[f"self_attn.{p}"].clone() for p in PROJ}
        del sd
        gc.collect()

        # ---- attention half, per arm ----
        post_attn = {}
        for a, widths in ROSTER:
            if li == 0:
                with torch.no_grad():
                    for j, p in enumerate(PROJ):
                        src = pristine[p]
                        getattr(layer.self_attn, p.split(".")[0]).weight.copy_(
                            src if widths is None else rtn(src, widths[j])
                        )
            with torch.no_grad():
                s = streams[a]
                residual = s
                po, cb, h = layer.attn_hc(s)
                h = layer.input_layernorm(h)
                if layer.block_type == "linear_attention":
                    h = layer.self_attn(hidden_states=h, cache_params=caches[a], attention_mask=mask)
                else:
                    h, _, tk = layer.self_attn(
                        hidden_states=h, attention_mask=mask, position_ids=None,
                        past_key_values=caches[a], use_cache=True,
                        position_embeddings=None, prev_topk_indices=prev_topk[a],
                    )
                    prev_topk[a] = tk
                h = po.unsqueeze(-1) * h.unsqueeze(-2) + torch.matmul(cb.transpose(-1, -2), residual)
                res2 = h
                po2, cb2, h = layer.ffn_hc(h)
                post_attn[a] = (layer.post_attention_layernorm(h), res2, po2, cb2)

        # ---- FFN half ----
        if not sparse:
            for a in arms:
                mi, res2, po2, cb2 = post_attn[a]
                with torch.no_grad():
                    o = layer.mlp(mi)
                    streams[a] = po2.unsqueeze(-1) * o.unsqueeze(-2) + torch.matmul(
                        cb2.transpose(-1, -2), res2)
        else:
            sel, wts = {}, {}
            with torch.no_grad():
                for a in arms:
                    _, w, idx = router(post_attn[a][0])
                    sel[a], wts[a] = idx, w
            per_tok = [len({int(i) for a in arms for i in sel[a].view(n, -1)[t]}) for t in range(n)]
            union = sorted({int(i) for a in arms for i in sel[a].flatten()})
            bf16_u = sorted({int(i) for i in sel["bf16"].flatten()})
            # incremental novelty: what each arm adds beyond the ones before it
            seen, incr = set(bf16_u), {}
            for a in arms:
                s_a = {int(i) for i in sel[a].flatten()}
                incr[a] = len(s_a - seen)
                seen |= s_a

            # ---- selective fetch: only the union ----
            cs = copy.deepcopy(cfg)
            cs.num_local_experts = cs.n_routed_experts = len(union)
            comp = Glm5NextTextExperts(cs).to(torch.float32).eval()
            from safetensors import safe_open
            loc = {g: l for l, g in enumerate(union)}
            with torch.no_grad():
                for g in union:
                    parts = {}
                    for nm in ("gate_proj", "up_proj", "down_proj"):
                        key = f"{prefix}.mlp.experts.{g}.{nm}.weight"
                        with safe_open(os.path.join(CKPT, smap[key]), framework="pt") as f:
                            wq = f.get_tensor(key)
                            sc = f.get_tensor(key + "_scale_inv")
                        parts[nm] = dequant_fp8(wq, sc)
                    comp.gate_up_proj[loc[g]] = torch.cat([parts["gate_proj"], parts["up_proj"]], 0)
                    comp.down_proj[loc[g]] = parts["down_proj"]
                for a in arms:
                    mi, res2, po2, cb2 = post_attn[a]
                    idx = sel[a].clone().apply_(lambda v: loc[int(v)])
                    o = comp(mi.view(-1, cfg.hidden_size), idx, wts[a]).view(mi.shape)
                    o = o + shared(mi)
                    streams[a] = po2.unsqueeze(-1) * o.unsqueeze(-2) + torch.matmul(
                        cb2.transpose(-1, -2), res2)
            k = cfg.num_experts_per_tok
            agree = {a: sum(1 for t in range(n)
                            if set(sel[a].view(n, -1)[t].tolist()) == set(sel["bf16"].view(n, -1)[t].tolist())) / n
                     for a in arms}
            row = {
                "layer": li, "bf16_distinct": len(bf16_u), "union": len(union),
                "amp_prompt": len(union) / max(len(bf16_u), 1),
                "amp_token_mean": sum(per_tok) / n / k, "amp_token_max": max(per_tok) / k,
                "incremental": incr, "agreement": agree,
            }
            census.append(row)
            print(
                f"  L{li:>2} sparse | bf16 {len(bf16_u):>3} distinct | union {len(union):>3}"
                f" | amp {row['amp_prompt']:.2f}x prompt, {row['amp_token_mean']:.2f}x/token"
                f" | agree3.50 {agree['aggr-3.50']*100:>5.1f}% | RSS {rss():.1f} GiB", flush=True)
            del comp
        del layer
        gc.collect()

    # ---- head: hc_head (unweighted mean over streams) -> norm -> lm_head ----
    # Exactly `Glm5NextTextModel.forward`'s `self.norm(self.hc_head(h))`.
    if args.logits:
        from safetensors import safe_open
        from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextRMSNorm

        def tensor(key):
            with safe_open(os.path.join(CKPT, smap[key]), framework="pt") as f:
                t = f.get_tensor(key)
                if key + "_scale_inv" in smap:
                    return dequant_fp8(t, f.get_tensor(key + "_scale_inv"))
                return t.float()

        norm = Glm5NextTextRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).to(torch.float32).eval()
        with torch.no_grad():
            norm.weight.copy_(tensor("model.language_model.norm.weight"))
        head_w = tensor("lm_head.weight")
        logits = {}
        with torch.no_grad():
            for a in arms:
                h = norm(streams[a].mean(dim=2))
                logits[a] = (h @ head_w.T).float().view(-1, head_w.shape[0])
        del head_w
        gc.collect()

        ref = logits["bf16"]
        lp_ref = torch.log_softmax(ref, -1)
        p_ref = lp_ref.exp()
        top1_ref = ref.argmax(-1)
        s_ref = ref.softmax(-1).sort(-1, descending=True).values
        margin_ref = (s_ref[:, 0] - s_ref[:, 1])
        # SATURATION CHECK FIRST. A bank whose BF16 top-1 probability is
        # ~1.0 everywhere cannot discriminate candidates: every arm reads
        # 100 % top-1 agreement whatever it does. Report the band before
        # reporting any agreement number.
        conf = s_ref[:, 0]
        q = conf.quantile(torch.tensor([0.0, 0.5, 0.95, 1.0]))
        sat = (conf > 0.9).float().mean()
        print(f"\nBF16 confidence over {n} positions: min {q[0]:.4f} med {q[1]:.4f} "
              f"p95 {q[2]:.4f} max {q[3]:.4f} | {sat*100:.0f}% above 0.9")
        if sat > 0.5:
            print("  ** SATURATED: most positions are near-certain, so top-1 agreement "
                  "cannot discriminate. Treat agreement as uninformative on this bank. **")
        print(f"\nKDA-BEHAV-1 ({cat}), {n} positions, logits vs BF16\n")
        print(f"{'candidate':<13}{'KL':>10}{'KL p95':>10}{'top-1':>9}{'top-5':>9}"
              f"{'rank>1':>8}{'margin':>9}")
        beh = []
        for a in arms:
            if a == "bf16":
                continue
            lp = torch.log_softmax(logits[a], -1)
            kl = (p_ref * (lp_ref - lp)).sum(-1)
            t1 = (logits[a].argmax(-1) == top1_ref).float().mean()
            k5r = ref.topk(5, -1).indices
            k5c = logits[a].topk(5, -1).indices
            ov = torch.tensor([len(set(k5r[i].tolist()) & set(k5c[i].tolist())) / 5 for i in range(n)])
            rank = (logits[a] > logits[a].gather(1, top1_ref[:, None])).sum(-1)
            sc = logits[a].softmax(-1).sort(-1, descending=True).values
            mg = (sc[:, 0] - sc[:, 1])
            print(f"{a:<13}{kl.mean():>10.5f}{kl.quantile(0.95):>10.5f}{t1*100:>8.1f}%"
                  f"{ov.mean()*100:>8.1f}%{(rank > 0).float().mean()*100:>7.1f}%"
                  f"{(mg.mean()/margin_ref.mean()):>9.3f}")
            beh.append({"arm": a, "kl_mean": float(kl.mean()), "kl_p95": float(kl.quantile(0.95)),
                        "top1": float(t1), "top5_overlap": float(ov.mean()),
                        "rank_moved": float((rank > 0).float().mean()),
                        "margin_ratio": float(mg.mean() / margin_ref.mean())})
        if args.out:
            json.dump({"census": census, "behaviour": beh},
                      open(args.out.replace(".json", "_behav.json"), "w"), indent=1)

    if census and args.out:
        json.dump(census, open(args.out, "w"), indent=1)
    if census:
        a = [r["amp_prompt"] for r in census]
        print(f"\nDEPTH CURVE over {len(census)} sparse layers:")
        print(f"  prompt-union amplification: first {a[0]:.2f}x  last {a[-1]:.2f}x  "
              f"mean {sum(a)/len(a):.2f}x  max {max(a):.2f}x")
        print(f"  trend: {'GROWING' if a[-1] > a[0]*1.25 else 'FLAT/BOUNDED'}")


if __name__ == "__main__":
    main()
