"""Real GLM-5.3-Flash MLA-NoPE attention, q-LoRA path, against the oracle.

Drives the pinned upstream `Glm5NextTextAttention` on layer 3's real
weights and compares it against LARQL's MLA executor over the same
checkpoint bytes (`examples/glm_mla_probe`) — nothing widened: the four
FP8 projections are read natively and `kv_b_proj` stays BF16.

**Scope, stated because it is not obvious.** Layer 3 is a
`deepseek_sparse_attention` layer: the reference gates attention with a
DSA top-k mask. Below `index_topk` positions that mask selects every
causally-visible key, so sparse attention and dense causal attention are
the same function — which is what makes MLA parity establishable before
the indexer exists. This script CHECKS that rather than assuming it: it
reads the mask the reference actually built and refuses to report parity
if it is not full-causal.

A free diagnostic the two controls hand you, recorded so it is not
rediscovered: **position 0 cannot witness a query-path defect.** With one
visible key the softmax is 1.0 whatever the query, so `omit-q-a-norm`
moves `q_latent` by 0.97 and every later position by ~0.5-0.65 while
leaving `output[0]` bit-unchanged. `omit-kv-a-norm` moves every position
INCLUDING the first, and leaves `q_latent` untouched. So a disagreement
that spares position 0 is in the query path; one that moves it is not —
the MLA counterpart of KDA's "q never touches the recurrent state".
"""
import argparse, json, os, subprocess, sys, tempfile
import numpy as np
import torch

CKPT = "/Volumes/model-drive/models/GLM-5.3-Flash"
TOL = 5e-5


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--positions", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--probe", default="target/release/examples/glm_mla_probe")
    ap.add_argument(
        "--control",
        choices=["omit-q-a-norm", "omit-kv-a-norm"],
        help="Perturb the REAL executor and require the comparison to FAIL. "
             "`omit-q-a-norm` is the q-LoRA path's own norm, which no "
             "pre-existing MLA control exercised.",
    )
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from glm_layer_oracle import load_layer, build_config
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextAttention

    tcfg = build_config(args.ckpt)
    tcfg._attn_implementation = "eager"
    prefix = f"model.language_model.layers.{args.layer}"
    sd = load_layer(args.ckpt, f"{prefix}.self_attn", torch.float32)
    attn = Glm5NextTextAttention(tcfg, args.layer).to(torch.float32)
    missing, unexpected = attn.load_state_dict(sd, strict=False)
    print(f"loaded self_attn: missing={list(missing)} unexpected={list(unexpected)}")
    attn.eval()

    g = torch.Generator().manual_seed(args.seed)
    x = torch.randn(1, args.positions, tcfg.hidden_size, generator=g) * 0.02

    # ── Is the DSA mask inert at this length? ──
    #
    # Captured from the REAL forward rather than reconstructed: the mask
    # checked is the one attention actually used, so a reconstruction
    # that differed from it could not make this pass.
    from transformers.cache_utils import DynamicCache

    seen = {}
    original = type(attn).build_attention_mask_from_topk

    def capture(self, topk_indices, query_states, kv_length):
        m = original(self, topk_indices, query_states, kv_length)
        seen["mask"] = m
        return m

    type(attn).build_attention_mask_from_topk = capture
    try:
        with torch.no_grad():
            out, _, _ = attn(
                hidden_states=x,
                # The indexer's own argument is a [B, S] BOOLEAN PADDING
                # mask, not an attention bias — it is a different object
                # from the mask the attention finally uses, which the
                # layer derives from the top-k below.
                attention_mask=torch.ones(1, args.positions, dtype=torch.bool),
                past_key_values=DynamicCache(config=tcfg),
                use_cache=True,
            )
    finally:
        type(attn).build_attention_mask_from_topk = original

    mask = seen["mask"]
    causal = torch.tril(torch.ones(args.positions, args.positions, dtype=torch.bool))
    visible = mask if mask.dtype == torch.bool else (mask == 0)
    visible = visible.reshape(-1, args.positions, args.positions)[0]
    inert = bool((visible == causal).all())
    print(f"\nDSA mask at {args.positions} positions vs index_topk={tcfg.index_topk}: "
          f"{'FULL-CAUSAL (indexer inert)' if inert else 'SPARSE'}")
    if not inert:
        extra = int((visible != causal).sum())
        print(f"  {extra} entries differ from causal — MLA parity is NOT separable "
              f"from the indexer at this length, and this run proves nothing")
        sys.exit(1)

    with torch.no_grad():
        q_resid = attn.q_a_layernorm(attn.q_a_proj(x))
    want = out.view(args.positions, -1).numpy()

    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "x.f32"), "wb") as f:
            f.write(x.view(-1).numpy().astype("<f4").tobytes())
        env = dict(os.environ, GLM_MLA_HEADS=str(tcfg.num_attention_heads),
                   GLM_RMS_EPS=repr(tcfg.rms_norm_eps))
        if args.control:
            env["GLM_MLA_MUTATION"] = args.control
        subprocess.run(
            [args.probe, shard_of(args.ckpt, f"{prefix}.self_attn.q_b_proj.weight"),
             prefix, os.path.join(d, "x.f32"), str(args.positions), d],
            check=True, env=env,
        )
        got = np.fromfile(os.path.join(d, "output.f32"), dtype="<f4").reshape(args.positions, -1)
        q_latent = np.fromfile(os.path.join(d, "q_latent.f32"), dtype="<f4")

    # The q-LoRA latent, the boundary that did not exist before this rung
    # — and the one GLM's DSA indexer will read.
    want_latent = q_resid[0, -1].detach().numpy()
    e_latent = rel(q_latent, want_latent)
    print(f"\n  {'ok  ' if e_latent <= TOL else 'FAIL'} q_latent   "
          f"{q_latent.size:>5} values   rel {e_latent:.3e}")

    worst = e_latent
    for p in range(args.positions):
        e = rel(got[p], want[p])
        worst = max(worst, e)
        print(f"  {'ok  ' if e <= TOL else 'FAIL'} output[{p}]  "
              f"{got[p].size:>5} values   rel {e:.3e}")

    print(f"\nworst {worst:.3e} against a bar of {TOL:.0e}")
    if args.control:
        if worst <= TOL:
            print(f"UNEXPECTED: control `{args.control}` did not move the result — "
                  f"the comparison is not checking what it claims")
            sys.exit(1)
        print(f"control `{args.control}` fires: the gate can return the other answer")
        return
    if worst > TOL:
        print("DISAGREEMENT")
        sys.exit(1)
    print("real GLM MLA-NoPE with q-LoRA agrees with the upstream reference")


def shard_of(ckpt, name):
    idx = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))
    return os.path.join(ckpt, idx["weight_map"][name])


if __name__ == "__main__":
    main()
