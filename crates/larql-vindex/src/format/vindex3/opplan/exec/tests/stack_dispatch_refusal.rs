//! `exec::stack::layer_forward`'s two defensive panics — invariants
//! `KimiDecoderLayer.__init__` guarantees for every REAL layer (no layer
//! is ever MLA+dense; attention weights and carried state always name
//! the same operator family), never exercised by the 27-real-layer
//! topology itself, so pinned directly here instead.
//!
//! Dummy geometry and empty weight slices throughout: the panic fires on
//! the enum discriminants alone, before any of this module's real math
//! ever runs, so what the slices CONTAIN is irrelevant — only that a
//! well-typed `LayerSpec`/`LayerState` pair exists to mismatch.

use larql_models::config::{KdaGeometry, MlaGeometry};

use crate::format::vindex3::opplan::exec::kda::{zero_state, KdaOutputGateWeights, KdaWeights};
use crate::format::vindex3::opplan::exec::kimi_moe_block::ExpertWeights;
use crate::format::vindex3::opplan::exec::mla::{MlaState, MlaWeights};
use crate::format::vindex3::opplan::exec::stack::{
    stack_forward, LayerAttention, LayerFfn, LayerSpec, LayerState,
};

fn kda_weights<'a>(empty: &'a [f32], empty_bf16: &'a [u16]) -> KdaWeights<'a> {
    KdaWeights {
        q_proj: WeightRows::Bf16(empty_bf16),
        k_proj: WeightRows::Bf16(empty_bf16),
        v_proj: WeightRows::Bf16(empty_bf16),
        q_conv1d: empty,
        k_conv1d: empty,
        v_conv1d: empty,
        f_a_proj: empty,
        f_b_proj: empty,
        output_gate: KdaOutputGateWeights::LowRank {
            g_a_proj: empty,
            g_b_proj: empty,
        },
        b_proj: empty,
        a_log: empty,
        dt_bias: empty,
        o_norm: empty,
        o_proj: WeightRows::Bf16(empty_bf16),
        norm_eps: 1e-6,
        // The rank the gate factorisations meet at — this fixture's
        // own `f_a_proj`, not the head dim the executor used to assume.
        gate_rank: 1,
    }
}

fn mla_weights(empty: &[f32]) -> MlaWeights<'_> {
    MlaWeights {
        output_gate: None,
        q_proj: WeightRows::F32(empty),
        kv_a_proj: WeightRows::F32(empty),
        kv_a_norm: empty,
        kv_b_proj: WeightRows::F32(empty),
        o_proj: WeightRows::F32(empty),
        kv_a_norm_eps: 1e-6,
    }
}

const DUMMY_KDA_GEOMETRY: KdaGeometry = KdaGeometry {
    num_heads: 1,
    head_dim: 1,
    conv_kernel: 1,
};
const DUMMY_MLA_GEOMETRY: MlaGeometry = MlaGeometry {
    num_heads: 1,
    kv_lora_rank: 1,
    qk_nope_head_dim: 1,
    qk_rope_head_dim: 1,
    v_head_dim: 1,
};
use crate::format::vindex3::opplan::exec::cpu::projector::WeightRows;

#[test]
#[should_panic(expected = "no Kimi layer is MLA+dense")]
fn mla_attention_with_a_dense_ffn_panics() {
    let empty: Vec<f32> = Vec::new();
    let empty_bf16: Vec<u16> = Vec::new();
    let spec = LayerSpec {
        attention: LayerAttention::Mla(mla_weights(&empty), DUMMY_MLA_GEOMETRY),
        ffn: LayerFfn::Dense {
            weights: ExpertWeights {
                gate: &empty_bf16,
                up: &empty_bf16,
                down: &empty_bf16,
            },
            inter: 1,
        },
        input_norm_weight: &empty,
        post_attention_norm_weight: &empty,
        norm_eps: 1e-5,
    };
    // The real topology never builds this combination — `KimiDecoderLayer.
    // __init__` only ever wires `first_k_dense_replace`'s one dense layer
    // to KDA (layer 0) — so this state is otherwise unreachable and must
    // be constructed by hand to exercise the guard at all.
    let mut states = [LayerState::Mla(MlaState::default())];
    let _ = stack_forward(&[0.0], 1, &[spec], &mut states);
}

#[test]
#[should_panic(expected = "disagree on operator family")]
fn kda_attention_paired_with_mla_state_panics() {
    let empty: Vec<f32> = Vec::new();
    let empty_bf16: Vec<u16> = Vec::new();
    let spec = LayerSpec {
        attention: LayerAttention::Kda(kda_weights(&empty, &empty_bf16), DUMMY_KDA_GEOMETRY),
        ffn: LayerFfn::Dense {
            weights: ExpertWeights {
                gate: &empty_bf16,
                up: &empty_bf16,
                down: &empty_bf16,
            },
            inter: 1,
        },
        input_norm_weight: &empty,
        post_attention_norm_weight: &empty,
        norm_eps: 1e-5,
    };
    // A caller that built the wrong state array for this layer's own
    // declared attention family — `stack_forward`'s own contract
    // ("one state per layer, in layer order") does not by itself rule
    // this out at the type level, so it is a runtime guard, not dead code.
    let mut states = [LayerState::Mla(MlaState::default())];
    let _ = stack_forward(&[0.0], 1, &[spec], &mut states);
}

#[test]
#[should_panic(expected = "disagree on operator family")]
fn mla_attention_paired_with_kda_state_panics() {
    let empty: Vec<f32> = Vec::new();
    let spec = LayerSpec {
        attention: LayerAttention::Mla(mla_weights(&empty), DUMMY_MLA_GEOMETRY),
        ffn: LayerFfn::Moe {
            router_weight: &empty,
            router_bias: &empty,
            experts: 0,
            top_k: 0,
            renormalize: false,
            branch_scale: 1.0,
            loaded: &[],
            shared: None,
            inter: 1,
        },
        input_norm_weight: &empty,
        post_attention_norm_weight: &empty,
        norm_eps: 1e-5,
    };
    let mut states = [LayerState::Kda(zero_state(DUMMY_KDA_GEOMETRY))];
    let _ = stack_forward(&[0.0], 1, &[spec], &mut states);
}
