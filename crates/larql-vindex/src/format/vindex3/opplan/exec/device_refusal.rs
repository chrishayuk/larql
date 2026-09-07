//! Why a DEVICE path must refuse a layer, decided from DECLARED facts
//! alone and before any tensor is bound.
//!
//! Every function here is pure: it takes what the container says and
//! answers with the sentence a reader should meet, or `None`. That is
//! the point of the module, not an accident of its size. A refusal
//! raised later — while binding — surfaces as a missing tensor name or a
//! byte count that does not add up, and sends a reader to a buffer
//! instead of to the config line that governs it. So the decision is
//! taken here, early, and a test without a GPU can witness every arm.
//!
//! The rule these share: a device path refuses what it cannot EXPRESS,
//! never what it merely has not seen. Each refusal below names a
//! declaration the container makes and a lowering that has no arm for
//! it, and each is paired with a test proving the ordinary form is left
//! open.

use super::super::LatentBranchOp;

/// Why a fully-LOWERED routed FFN must refuse `layer`, if it must —
/// named before any tensor is bound (K3-LATENTMOE-1, freeze D12).
///
/// The descriptor MoE path encodes a whole layer into one command buffer
/// from the plan's `RoutedFfnOp` alone: router, expert bank, combine. It
/// has no arm for a bottleneck around that bank — no down-projection into
/// it, no norm on the aggregate, no up-projection out — and the operands
/// it does bind would all still be the right shape, because the bank is
/// stored at the latent width and the descriptor path would happily run
/// it there. The layer would then consume a `[latent]` vector it was
/// never given and write a `[latent]` result into a `[hidden]` residual.
///
/// So the refusal is here, at weight-build time, and it names the
/// DECLARATION and the width rather than surfacing later as a byte count
/// that does not add up. `None` = nothing stands in the way.
///
/// This is NOT the general device refusal: the interpreter's device
/// backend runs the wrapper through `project`/`norm` like any other
/// projection and needs no exemption. What cannot express it is the
/// single-command-buffer lowering, and only that.
///
/// The program is spelled from the op's OWN operands rather than from
/// tensor names written out here. A message that restated the spellings
/// would be a second place they live, and would go on naming
/// `routed_expert_down_proj` after a family arrived that spells it
/// otherwise — telling a reader to look for a tensor their checkpoint
/// does not contain. The DECLARATION is named as a literal, because a
/// config leaf is what it is called in the config.
///
/// A pure function so a test without a GPU can witness the refusal.
pub fn lowered_latent_branch_refusal(
    layer: usize,
    latent: Option<&LatentBranchOp>,
) -> Option<String> {
    let latent = latent?;
    let program = match &latent.norm {
        Some(norm) => format!(
            "{} -> experts -> {} -> {}",
            latent.down.tensor, norm.weight.tensor, latent.up.tensor
        ),
        None => format!(
            "{} -> experts -> (no norm) -> {}",
            latent.down.tensor, latent.up.tensor
        ),
    };
    Some(format!(
        "layer {layer}: the container declares a latent routed branch \
         (`routed_expert_hidden_size: {}` — {program}), which the descriptor MoE lowering does \
         not carry (it encodes router, bank and combine at one width); refusing rather than \
         running the experts as though the bottleneck were the residual stream",
        latent.width,
    ))
}

/// Why an attention gate whose weights ride in the query projection has
/// no device path.
///
/// A gate sourced from the fused query projection is not a separate
/// operand: the projection's rows carry both, so binding it means
/// splitting rows the device gemv path has no arm for. Stated once and
/// read from both the decode and the batch site — the two used to carry
/// the sentence separately, which is one message with two places to
/// drift.
pub fn fused_query_gate_projection_refusal() -> String {
    "a fused query/gate projection has no device kernel; refusing".to_string()
}

/// Why a DEVICE attention path must refuse `layer`, if it must — named
/// before any tensor is bound, from declared facts alone (K3-REP-GATE-1,
/// freeze D6).
///
/// Neither of Kimi-K3's declared output gates is carried by the device
/// paths yet: the KDA device path binds the low-rank `g_a_proj`/`g_b_proj`
/// pair by NAME, so a full-rank layer would fail on a missing name (or, on
/// a container shipping both, bind the wrong one); the MLA device path
/// has no gate at all, so a gated layer would run ungated with every shape
/// still closing. `None` = nothing stands in the way. A pure function so
/// a test without a GPU can witness both refusals (freeze P8).
pub fn declared_gate_refusal(
    layer: usize,
    mla_layer: bool,
    kda_full_rank_gate: bool,
    mla_output_gate: bool,
    mla_q_lora_rank: Option<usize>,
) -> Option<String> {
    // K3-MLA-Q-LORA-1. `MlaDeviceWeights` has ONE `q_proj` slot, and
    // `q_b_proj` has the same row count as the `q_proj` it replaces — so
    // binding it there would be finite, plausible and wrong, with every
    // shape still closing. Refused BY NAME and, like the two gates,
    // before any tensor is bound: a refusal raised later would surface
    // as a missing-tensor error naming `q_proj`, which the checkpoint
    // never shipped and a reader would go looking for.
    if mla_layer {
        if let Some(rank) = mla_q_lora_rank {
            return Some(format!(
                "layer {layer}: the container declares a factorised MLA query \
                 (`q_lora_rank: {rank}` — q_a_proj -> q_a_layernorm -> q_b_proj), which the \
                 Metal MLA path does not carry (it binds one dense q_proj); refusing rather \
                 than binding q_b_proj into the q_proj slot"
            ));
        }
    }
    if mla_layer && mla_output_gate {
        return Some(format!(
            "layer {layer}: the container declares an MLA output gate (`mla_use_output_gate`), \
             which the Metal MLA path does not carry; refusing rather than running the layer \
             ungated"
        ));
    }
    if !mla_layer && kda_full_rank_gate {
        return Some(format!(
            "layer {layer}: the container declares a full-rank KDA output gate \
             (`use_full_rank_gate`), which the Metal KDA path does not carry (it binds the \
             low-rank g_a_proj/g_b_proj pair); refusing rather than binding the wrong form"
        ));
    }
    None
}
