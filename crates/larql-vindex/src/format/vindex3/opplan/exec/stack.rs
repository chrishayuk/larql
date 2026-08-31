//! The mixed 27-layer Kimi Linear stack, executed — the first place
//! every already-proven piece meets ACROSS layers, not just within one.
//!
//! Nothing new is transcribed here either, the same posture
//! `kimi_kda_layer.rs`/`kimi_mla_layer.rs` already hold for their own
//! compositions: this module dispatches to whichever of
//! [`super::kimi_kda_layer::kda_decoder_layer_forward`],
//! [`super::kimi_kda_layer::kda_dense_decoder_layer_forward`] (layer 0
//! only — the one layer `first_k_dense_replace=1` excludes from MoE
//! routing) or [`super::kimi_mla_layer::mla_decoder_layer_forward`] a
//! layer's own declared kind calls for, and threads the residual stream
//! and each layer's OWN state object between calls. `KimiDecoderLayer`
//! itself declares which: `config.is_kda_layer(layer_idx)` picks
//! `KimiDeltaAttention` or `KimiMLAAttention`, and
//! `layer_idx >= first_k_dense_replace` picks `block_sparse_moe` or
//! `mlp` — read from the checkpoint's own `modeling_kimi.py`
//! `KimiDecoderLayer.__init__`, not guessed.
//!
//! **Layer state is per LAYER, carried across POSITIONS — never across
//! layers.** A KDA layer's recurrent state and an MLA layer's KV cache
//! both belong to that one layer's own attention operator; the residual
//! stream is the only thing that crosses a layer boundary. Call
//! [`stack_forward`] once per position, in order, threading the SAME
//! `states` slice across calls — exactly the shape autoregressive decode
//! needs: run the whole stack for token `t`, keep every layer's state,
//! run the whole stack again for token `t+1`.

use larql_models::config::{KdaGeometry, MlaGeometry};

use super::continuation::RecurrentState;
use super::kda::KdaWeights;
use super::kimi_kda_layer::{kda_decoder_layer_forward, kda_dense_decoder_layer_forward};
use super::kimi_mla_layer::mla_decoder_layer_forward;
use super::kimi_moe_block::ExpertWeights;
use super::mla::{MlaState, MlaWeights};

/// Which attention operator a layer runs — the boundary the user asked
/// this trace to name explicitly, distinct from whether its FFN branch
/// is dense or routed (every MLA layer is routed; only layer 0, a KDA
/// layer, is dense).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionKind {
    Kda,
    Mla,
}

/// One layer's declared weights — attention family and FFN family are
/// independent axes, so this is two enums, not one four-way sum: Kimi's
/// real 27 layers only ever combine them as (Kda, Dense) once, (Kda,
/// Moe) nineteen times and (Mla, Moe) seven times, but nothing here
/// FORBIDS the fourth combination — that is `KimiDecoderLayer.
/// __init__`'s job to have decided, not this module's to assume.
pub enum LayerAttention<'a> {
    Kda(KdaWeights<'a>, KdaGeometry),
    Mla(MlaWeights<'a>, MlaGeometry),
}

/// One already-selected expert's weights, tagged by id — a slice of
/// these stands in for the closures the single-layer executors take,
/// avoiding a `dyn Fn` in a struct that already has enough lifetime
/// parameters. The caller loads exactly the ids it knows will be
/// selected (same sparsity discipline every prior MoE-adjacent rung
/// holds) and this module looks them up by linear scan — small counts
/// per layer, correctness over cleverness.
pub struct LoadedExpert<'a> {
    pub id: usize,
    pub weights: ExpertWeights<'a>,
}

fn find_expert<'a>(loaded: &[LoadedExpert<'a>], id: usize) -> ExpertWeights<'a> {
    loaded
        .iter()
        .find(|e| e.id == id)
        .unwrap_or_else(|| panic!("layer forward asked for un-loaded expert {id}"))
        .weights
}

pub enum LayerFfn<'a> {
    /// `KimiMLP` — layer 0 only.
    Dense {
        weights: ExpertWeights<'a>,
        inter: usize,
    },
    /// `KimiSparseMoeBlock` — every other layer.
    Moe {
        router_weight: &'a [f32],
        router_bias: &'a [f32],
        experts: usize,
        top_k: usize,
        renormalize: bool,
        branch_scale: f64,
        loaded: &'a [LoadedExpert<'a>],
        shared: Option<(ExpertWeights<'a>, usize)>,
        inter: usize,
    },
}

/// One layer's complete declaration: which attention operator, which FFN
/// branch, and the two plain `KimiRMSNorm` weights `KimiDecoderLayer`
/// applies around both, at whatever `norm_eps` the checkpoint declares
/// (the SAME value for both norms — `kv_a_layernorm`'s own DIFFERENT eps
/// lives inside `MlaWeights` instead, per `exec::mla`'s own doc comment).
pub struct LayerSpec<'a> {
    pub attention: LayerAttention<'a>,
    pub ffn: LayerFfn<'a>,
    pub input_norm_weight: &'a [f32],
    pub post_attention_norm_weight: &'a [f32],
    pub norm_eps: f64,
}

/// One layer's carried state — a KDA layer's recurrent+conv state or an
/// MLA layer's per-position KV cache, never both, never neither.
pub enum LayerState {
    Kda(RecurrentState),
    Mla(MlaState),
}

/// Every boundary the user asked to see per layer, so the FIRST
/// mismatching layer says immediately whether the defect is operator
/// dispatch, state carriage, residual sequencing, layer ordering, or an
/// expert selection that changed because the upstream hidden state had
/// already drifted.
#[derive(Debug, Clone, PartialEq)]
pub struct StackLayerTrace {
    pub layer: usize,
    pub kind: AttentionKind,
    /// The residual stream ENTERING this layer — before `input_layernorm`.
    pub input_residual: Vec<f32>,
    /// The attention operator's own output, pre-residual-add.
    pub attention_output: Vec<f32>,
    /// `input_residual + attention_output`.
    pub post_attention_residual: Vec<f32>,
    /// The FFN branch's output, pre-residual-add — `KimiMLP`'s for layer
    /// 0, `KimiSparseMoeBlock`'s for every other layer.
    pub ffn_output: Vec<f32>,
    /// `post_attention_residual + ffn_output` — this layer's contribution
    /// to the residual stream the NEXT layer reads as `input_residual`.
    pub layer_output: Vec<f32>,
    /// KDA: the recurrent state's element count — constant, `O(1)` in
    /// position count. MLA: the KV cache's cached-position count —
    /// grows by exactly one per call. The two state MODELS this
    /// programme's whole point is to keep distinct, read back out where
    /// a stack trace can show it moving (or not) layer by layer.
    pub state_size: usize,
}

/// One token through the ENTIRE mixed stack — call once per position, in
/// order, threading the SAME `states` slice (one entry per layer, in
/// layer order) across calls.
pub fn stack_forward(
    x: &[f32],
    hidden: usize,
    layers: &[LayerSpec<'_>],
    states: &mut [LayerState],
) -> Vec<StackLayerTrace> {
    assert_eq!(
        layers.len(),
        states.len(),
        "one state per layer, in layer order"
    );
    let mut h = x.to_vec();
    let mut traces = Vec::with_capacity(layers.len());
    for (i, (spec, state)) in layers.iter().zip(states.iter_mut()).enumerate() {
        let trace = layer_forward(i, &h, hidden, spec, state);
        h = trace.layer_output.clone();
        traces.push(trace);
    }
    traces
}

/// [`layer_forward`] for callers outside this module — the mixed
/// Metal/CPU stack runs host layers through exactly this path, so the
/// two cannot drift.
pub fn layer_forward_public(
    layer: usize,
    x: &[f32],
    hidden: usize,
    spec: &LayerSpec<'_>,
    state: &mut LayerState,
) -> StackLayerTrace {
    layer_forward(layer, x, hidden, spec, state)
}

fn layer_forward(
    layer: usize,
    x: &[f32],
    hidden: usize,
    spec: &LayerSpec<'_>,
    state: &mut LayerState,
) -> StackLayerTrace {
    match (&spec.attention, &spec.ffn, &mut *state) {
        (LayerAttention::Kda(w, g), LayerFfn::Dense { weights, inter }, LayerState::Kda(s)) => {
            let t = kda_dense_decoder_layer_forward(
                x,
                hidden,
                spec.input_norm_weight,
                spec.post_attention_norm_weight,
                spec.norm_eps,
                *w,
                *g,
                s,
                *weights,
                *inter,
            );
            StackLayerTrace {
                layer,
                kind: AttentionKind::Kda,
                input_residual: x.to_vec(),
                attention_output: t.attention.output,
                post_attention_residual: t.after_attention,
                ffn_output: t.ffn_output,
                layer_output: t.output,
                state_size: s.buffer(super::kda::RECURRENT).cells().len(),
            }
        }
        (
            LayerAttention::Kda(w, g),
            LayerFfn::Moe {
                router_weight,
                router_bias,
                experts,
                top_k,
                renormalize,
                branch_scale,
                loaded,
                shared,
                inter,
            },
            LayerState::Kda(s),
        ) => {
            let t = kda_decoder_layer_forward(
                x,
                hidden,
                spec.input_norm_weight,
                spec.post_attention_norm_weight,
                spec.norm_eps,
                *w,
                *g,
                s,
                *inter,
                router_weight,
                router_bias,
                *experts,
                *top_k,
                *renormalize,
                *branch_scale,
                |id| find_expert(loaded, id),
                *shared,
            );
            StackLayerTrace {
                layer,
                kind: AttentionKind::Kda,
                input_residual: x.to_vec(),
                attention_output: t.attention.output,
                post_attention_residual: t.after_attention,
                ffn_output: t.moe.output,
                layer_output: t.output,
                state_size: s.buffer(super::kda::RECURRENT).cells().len(),
            }
        }
        (
            LayerAttention::Mla(w, g),
            LayerFfn::Moe {
                router_weight,
                router_bias,
                experts,
                top_k,
                renormalize,
                branch_scale,
                loaded,
                shared,
                inter,
            },
            LayerState::Mla(s),
        ) => {
            let t = mla_decoder_layer_forward(
                x,
                hidden,
                spec.input_norm_weight,
                spec.post_attention_norm_weight,
                spec.norm_eps,
                *w,
                *g,
                s,
                *inter,
                router_weight,
                router_bias,
                *experts,
                *top_k,
                *renormalize,
                *branch_scale,
                |id| find_expert(loaded, id),
                *shared,
            );
            StackLayerTrace {
                layer,
                kind: AttentionKind::Mla,
                input_residual: x.to_vec(),
                attention_output: t.attention.output,
                post_attention_residual: t.after_attention,
                ffn_output: t.moe.output,
                layer_output: t.output,
                state_size: s.len(),
            }
        }
        (LayerAttention::Mla(..), LayerFfn::Dense { .. }, _) => {
            panic!("layer {layer}: no Kimi layer is MLA+dense — first_k_dense_replace=1 excludes only layer 0, which is always KDA")
        }
        _ => {
            panic!("layer {layer}: attention weights and carried state disagree on operator family")
        }
    }
}
