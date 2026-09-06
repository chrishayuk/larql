"""Control: does GLM's KDA decay gate actually differ from Kimi's?

LARQL's `KdaOp` documents `gate_lower_bound` as provenance, NOT an
execution input — measured on Kimi, where `modeling_kimi.py` reads it
nowhere. GLM's reference reads it. This perturbs the REAL reference
module (not a transcription) to put a number on the difference.

Arms
  glm     : as shipped — lower_bound * sigmoid(exp(A_log) * g)
  kimi    : the softplus form LARQL's executor implements today
  identity: re-run of `glm` (proves the harness can return 0.0)
"""
import sys, os, argparse
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from glm_layer_oracle import load_layer, build_config, remap_checkpoint_to_module, CKPT


def run(layer_mod, streams):
    with torch.no_grad():
        out, _ = layer_mod(hidden_states=streams, attention_mask=None)
    return out.float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--positions", type=int, default=8)
    ap.add_argument("--ckpt", default=CKPT)
    args = ap.parse_args()

    tcfg = build_config(args.ckpt)
    tcfg._attn_implementation = "eager"
    from transformers.models.glm5_next import modeling_glm5_next as M

    sd = remap_checkpoint_to_module(load_layer(args.ckpt, f"model.language_model.layers.{args.layer}", torch.float32))
    layer = M.Glm5NextTextDecoderLayer(tcfg, args.layer).to(torch.float32)
    layer.load_state_dict(sd, strict=True)
    layer.eval()

    g = torch.Generator().manual_seed(0)
    streams = torch.randn(1, args.positions, tcfg.hc_mult, tcfg.hidden_size, generator=g) * 0.02

    print(f"config linear_lower_bound = {tcfg.linear_lower_bound!r}")
    print(f"module safe_gate_lower_bound = {layer.self_attn.forget_gate.safe_gate_lower_bound!r}\n")

    ref = run(layer, streams)

    original = M.Glm5NextTextForgetGate.forward

    def kimi_forward(self, hidden_states):
        """Kimi's form: -exp(A_log) * softplus(f_b(f_a(x)) + dt_bias)."""
        hidden_shape = (*hidden_states.shape[:2], -1, self.head_dim)
        fg = self.f_b_proj(self.f_a_proj(hidden_states))
        gg = (fg.float() + self.dt_bias.float().view(1, 1, -1)).view(hidden_shape)
        decay = torch.exp(self.A_log.float().view(1, 1, self.num_heads, 1))
        soft = torch.where(gg > 20.0, gg, torch.log(1.0 + torch.exp(gg)))
        return -decay * soft

    def report(name, other):
        num = (other - ref).norm()
        den = ref.norm()
        cos = torch.nn.functional.cosine_similarity(other.flatten(), ref.flatten(), dim=0)
        print(f"  {name:10s} relΔ(fro) = {num/den:.6e}   cos = {cos:.9f}   absmaxΔ = {(other-ref).abs().max():.6e}")

    print("layer-output deltas vs the shipped GLM gate:")
    report("identity", run(layer, streams))
    M.Glm5NextTextForgetGate.forward = kimi_forward
    try:
        report("kimi", run(layer, streams))
    finally:
        M.Glm5NextTextForgetGate.forward = original

    # And the gate tensor itself, so the effect is attributed, not just observed.
    fgm = layer.self_attn.forget_gate
    x = layer.input_layernorm(layer.attn_hc(streams)[2])
    with torch.no_grad():
        g_glm = fgm(x)
        M.Glm5NextTextForgetGate.forward = kimi_forward
        try:
            g_kimi = fgm(x)
        finally:
            M.Glm5NextTextForgetGate.forward = original
    print("\ndecay-gate tensor g (fed to exp(g) in the recurrence):")
    print(f"  glm  : range [{g_glm.min():+.4f}, {g_glm.max():+.4f}]  mean {g_glm.mean():+.4f}")
    print(f"  kimi : range [{g_kimi.min():+.4f}, {g_kimi.max():+.4f}]  mean {g_kimi.mean():+.4f}")
    print(f"  exp(g) glm  in [{g_glm.exp().min():.6f}, {g_glm.exp().max():.6f}]")
    print(f"  exp(g) kimi in [{g_kimi.exp().min():.6f}, {g_kimi.exp().max():.6f}]")


if __name__ == "__main__":
    main()
