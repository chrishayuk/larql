//! Structured explanation of a bound V3 program (LQL-2 `EXPLAIN INFER`).
//!
//! Static and deterministic: built from the executable
//! [`ComponentOpPlan`] alone — the same object the executor runs — so
//! the explanation IS the authority that will execute, not a
//! reconstruction of one. No tokens run to produce it.
//!
//! **The structured value is primary; renderings are derived.** LQL
//! prints it, and a JSON/server surface serialises the same value
//! later — nothing should ever parse pretty text to learn what a
//! program does. `PartialEq` is deliberate: the explain-stability gate
//! compares whole values across repeated opens.
//!
//! Operand provenance quotes the plan's own [`OperandRef`] bindings —
//! object, segment-relative tensor, dtype — exactly the coordinates
//! `OperandStore` resolves at execution time, so the explain chain and
//! the execution chain cannot name different bytes.

use larql_vindex::format::vindex3::opplan::exec::backend::PlanBackend;
use larql_vindex::format::vindex3::opplan::{
    ComponentOpPlan, LayerAttention, LayerFfn, OperandRef,
};
use serde::Serialize;

use super::runtime::Vindex3Runtime;

/// The whole program, explained. Field order mirrors execution order.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ExplainPlan {
    /// The container's self-declared model name.
    pub model: String,
    pub component: String,
    pub generation: u32,
    /// True by construction: a runtime only opens a closed plan.
    pub execution_closed: bool,
    pub embedding: ExplainEmbedding,
    pub layers: Vec<ExplainLayer>,
    /// Per-layer continuation geometry, as a provider will be
    /// `prepare`d with it.
    pub continuation: Vec<ExplainKvGeometry>,
    pub final_norm: bool,
    pub output: Option<ExplainOutput>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ExplainEmbedding {
    pub vocab_size: usize,
    pub scaled: bool,
    pub normed: bool,
    pub table: ExplainOperand,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ExplainLayer {
    pub layer: usize,
    /// The layer's operations in execution order — the step's own
    /// sequence, including the optional ops only when the plan
    /// declares them (absence is never an identity op).
    pub ops: Vec<String>,
    pub attention: ExplainAttention,
    pub ffn: ExplainFfn,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ExplainAttention {
    /// The layer's `layer_types` spelling: `"sliding"`, `"full"`, or
    /// `"linear_attention"`.
    pub mode: String,
    pub window: Option<usize>,
    /// Softmax head geometry. Absent on a linear-attention layer — which
    /// is a statement, not a gap: reporting a DeltaNet layer's 48 value
    /// heads as `kv_heads` would tell a reader it retains 48 heads of KV
    /// when it retains none.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub q_heads: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_heads: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub head_dim: Option<usize>,
    /// Elements in one linear-attention layer's recurrent state, constant
    /// in sequence length. Absent on a softmax layer, whose continuation
    /// state grows per position instead.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub state_elements: Option<usize>,
    pub gated: bool,
    pub qk_norm: bool,
    /// Per-head sink logits participate in the softmax.
    pub sinks: bool,
    /// Q/K/V/O projection biases are applied.
    pub biased: bool,
    /// Q/K/V/O (and gate, when present) bindings.
    pub operands: Vec<ExplainOperand>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ExplainFfn {
    /// `"dense"`, `"routed"`, or `"hybrid"`.
    pub kind: String,
    /// Routed layers: `(experts, top_k)`.
    pub experts: Option<(usize, usize)>,
    pub operands: Vec<ExplainOperand>,
}

/// One executable operand binding: the exact coordinates execution
/// resolves — `object → tensor → dtype` (the representation encoding).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ExplainOperand {
    pub role: String,
    pub object: String,
    pub tensor: String,
    pub dtype: String,
    pub shape: Vec<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct ExplainKvGeometry {
    pub kv_dim: usize,
    pub window: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ExplainOutput {
    pub vocab: usize,
    pub multiplied: bool,
    pub softcapped: bool,
    pub projection: ExplainOperand,
}

impl ExplainPlan {
    /// Explain a bound runtime's program.
    pub fn from_runtime<B: PlanBackend>(runtime: &Vindex3Runtime<B>) -> Self {
        Self::from_plan(runtime.plan(), runtime.model_name())
    }

    /// Explain a plan directly — the negative-control seam: tests
    /// mutate a plan and prove the explanation changes with it.
    pub fn from_plan(plan: &ComponentOpPlan, model: &str) -> Self {
        let embedding = plan
            .embedding
            .as_ref()
            .expect("a decode-servable component carries an embedding op");
        // Per layer, from the op itself — `plan_kv_geometry` refuses
        // (formerly: panicked, drill F8) on recurrent continuation, and a
        // recurrence genuinely has no KV row to explain: the per-layer
        // sections below carry its state story instead.
        let continuation = plan
            .layers
            .iter()
            .filter_map(|layer| layer.attention.softmax())
            .map(|op| ExplainKvGeometry {
                kv_dim: op.num_kv_heads * op.head_dim,
                window: op.window,
            })
            .collect();
        Self {
            model: model.to_string(),
            component: plan.component.clone(),
            generation: 3,
            execution_closed: true,
            embedding: ExplainEmbedding {
                vocab_size: embedding.vocab_size,
                scaled: embedding.scale.is_some(),
                normed: embedding.norm.is_some(),
                table: operand("table", &embedding.table),
            },
            layers: plan
                .layers
                .iter()
                .enumerate()
                .map(|(index, layer)| explain_layer(index, layer))
                .collect(),
            continuation,
            final_norm: plan.final_norm.is_some(),
            output: plan.output.as_ref().map(|op| ExplainOutput {
                vocab: op.projection.shape.first().copied().unwrap_or(0),
                multiplied: op.multiplier.is_some(),
                softcapped: op.softcapping.is_some(),
                projection: operand("projection", &op.projection),
            }),
        }
    }
}

fn explain_layer(
    index: usize,
    layer: &larql_vindex::format::vindex3::opplan::LayerPlan,
) -> ExplainLayer {
    // The pre-block norm is `input_layernorm` on an attention-class
    // layer and the single pre-mixer norm on a mixer-only one — one
    // program position, named for what the layer runs.
    let mixer_only = layer.ffn.is_none();
    let mut ops = vec![
        if mixer_only {
            "pre_mixer_norm".to_string()
        } else {
            "pre_attention_norm".to_string()
        },
        if mixer_only {
            "mamba2_mixer".to_string()
        } else {
            "attention".to_string()
        },
    ];
    if layer.post_attention_norm.is_some() {
        ops.push("post_attention_norm".into());
    }
    ops.push("residual_add".into());
    // A mixer-only layer's program ends at its one residual add: no FFN
    // exists, and listing absent ops would be the presentation-side twin
    // of the fabrication schema 6 removed.
    if !mixer_only {
        ops.push("pre_ffn_norm".into());
        ops.push("ffn".into());
        if layer.post_ffn_norm.is_some() {
            ops.push("post_ffn_norm".into());
        }
        ops.push("residual_add".into());
    }
    if layer.layer_scale.is_some() {
        ops.push("layer_scale".into());
    }

    let attention = match &layer.attention {
        LayerAttention::Softmax(op) => {
            let mut operands = vec![
                operand("q", &op.q),
                operand("k", &op.k),
                operand("v", &op.v),
                operand("o", &op.o),
            ];
            if let Some(gate) = &op.output_gate {
                operands.push(operand("output_gate", &gate.projection));
            }
            ExplainAttention {
                mode: if op.window.is_some() {
                    "sliding".into()
                } else {
                    "full".into()
                },
                window: op.window,
                q_heads: Some(op.num_q_heads),
                kv_heads: Some(op.num_kv_heads),
                head_dim: Some(op.head_dim),
                state_elements: None,
                gated: op.output_gate.is_some(),
                qk_norm: op.qk_norm.is_some(),
                sinks: op.sinks.is_some(),
                biased: op.q_bias.is_some(),
                operands,
            }
        }
        LayerAttention::GatedDelta(op) => ExplainAttention {
            mode: layer.attention.declared_name().into(),
            window: None,
            q_heads: None,
            kv_heads: None,
            head_dim: None,
            state_elements: Some(op.state_elements()),
            // The z projection gates this operator's output the way an
            // attention output gate does; the rest are softmax-only
            // features a recurrence has no analogue for.
            gated: true,
            qk_norm: false,
            sinks: false,
            biased: false,
            operands: vec![
                operand("in_proj_qkv", &op.in_proj_qkv),
                operand("in_proj_a", &op.in_proj_a),
                operand("in_proj_b", &op.in_proj_b),
                operand("in_proj_z", &op.in_proj_z),
                operand("conv1d", &op.conv1d),
                operand("a_log", &op.a_log),
                operand("dt_bias", &op.dt_bias),
                operand("norm", &op.norm),
                operand("out_proj", &op.out_proj),
            ],
        },
        LayerAttention::Kda(op) => ExplainAttention {
            mode: layer.attention.declared_name().into(),
            window: None,
            // One head count, unlike Gated DeltaNet's two — reported on
            // both sides because KDA's key and value geometries coincide,
            // rather than left absent as if unknown.
            q_heads: Some(op.num_heads),
            kv_heads: Some(op.num_heads),
            head_dim: Some(op.head_dim),
            state_elements: Some(op.state_elements()),
            // The g gate is this operator's output gate.
            gated: true,
            qk_norm: false,
            sinks: false,
            biased: false,
            operands: {
                let mut operands = vec![
                    operand("q_proj", &op.q_proj),
                    operand("k_proj", &op.k_proj),
                    operand("v_proj", &op.v_proj),
                    operand("q_conv1d", &op.q_conv1d),
                    operand("k_conv1d", &op.k_conv1d),
                    operand("v_conv1d", &op.v_conv1d),
                    operand("f_a_proj", &op.f_a_proj),
                    operand("f_b_proj", &op.f_b_proj),
                ];
                // The output gate's operands follow its DECLARED form —
                // the low-rank pair, or Kimi-K3's one full-rank `g_proj`.
                operands.extend(
                    op.output_gate
                        .operands()
                        .into_iter()
                        .map(|(name, r)| operand(name, r)),
                );
                operands.extend([
                    operand("b_proj", &op.b_proj),
                    operand("a_log", &op.a_log),
                    operand("dt_bias", &op.dt_bias),
                    operand("o_norm", &op.o_norm),
                    operand("out_proj", &op.out_proj),
                ]);
                operands
            },
        },
        // Named specifically: this layer attends, but NOT by plain
        // softmax — the conv over the fused QKV and the partial rotary
        // are what a reader asking "what does this layer run" must see.
        LayerAttention::ConvQkv(op) => ExplainAttention {
            mode: "conv-qkv-attention".into(),
            window: None,
            q_heads: Some(op.geometry.num_heads),
            kv_heads: Some(op.geometry.num_kv_heads),
            head_dim: Some(op.geometry.head_dim),
            state_elements: None,
            gated: false,
            qk_norm: false,
            sinks: false,
            biased: false,
            operands: {
                let mut operands = vec![
                    operand("in_proj", &op.in_proj),
                    operand("conv1d", &op.conv1d),
                ];
                if let Some(bias) = &op.conv1d_bias {
                    operands.push(operand("conv1d_bias", bias));
                }
                operands.push(operand("out_proj", &op.out_proj));
                operands
            },
        },
        LayerAttention::Mamba2(op) => ExplainAttention {
            // Named specifically, not the canonical `linear_attention`
            // spelling: EXPLAIN is where a reader asks what a layer
            // actually runs, and this one runs the SSD mixer.
            mode: "mamba2".into(),
            window: None,
            q_heads: None,
            kv_heads: None,
            head_dim: Some(op.geometry.head_dim),
            state_elements: Some(op.state_elements()),
            // The z half of the fused projection gates the output.
            gated: true,
            qk_norm: false,
            sinks: false,
            biased: false,
            operands: {
                let mut operands = vec![
                    operand("in_proj", &op.in_proj),
                    operand("conv1d", &op.conv1d),
                ];
                if let Some(bias) = &op.conv1d_bias {
                    operands.push(operand("conv1d_bias", bias));
                }
                operands.push(operand("a_log", &op.a_log));
                operands.push(operand("d", &op.d));
                operands.push(operand("dt_bias", &op.dt_bias));
                if let Some(norm) = &op.gated_norm {
                    operands.push(operand("gated_norm", &norm.weight));
                }
                operands.push(operand("out_proj", &op.out_proj));
                operands
            },
        },
        LayerAttention::Mla(op) => ExplainAttention {
            mode: layer.attention.declared_name().into(),
            window: None,
            q_heads: Some(op.num_heads),
            kv_heads: Some(op.num_heads),
            // No single width honestly describes MLA: the query/key side
            // is `q_head_dim()` (nope+rope), the output side is
            // `v_head_dim` — reporting either alone as "head_dim" would
            // tell a reader the wrong one is uniform. The operand list
            // below carries both, with their real shapes.
            head_dim: None,
            // Retains a per-position cache, not a fixed recurrent state —
            // the same `None` a softmax layer reports, for the same
            // reason.
            state_elements: None,
            gated: false,
            qk_norm: false,
            sinks: false,
            biased: false,
            operands: {
                // The query's operands under whichever form the layer
                // declared, at their own spellings — a factorised query
                // is three operands, not one under a borrowed name.
                let mut operands: Vec<_> = op
                    .query
                    .operands()
                    .into_iter()
                    .map(|(name, reference)| operand(name, reference))
                    .collect();
                operands.extend([
                    operand("kv_a_proj", &op.kv_a_proj),
                    operand("kv_a_norm", &op.kv_a_norm),
                    operand("kv_b_proj", &op.kv_b_proj),
                    operand("out_proj", &op.out_proj),
                ]);
                operands
            },
        },
    };

    let (kind, experts, ffn_operands) = match layer.ffn.as_ref() {
        // A mixer-only (Mamba2) layer has no FFN — the mixer is the whole
        // block, and reporting an absent op as anything else would be the
        // presentation-side twin of the fabrication schema 6 removed.
        None => ("absent", None, Vec::new()),
        Some(LayerFfn::Dense(op)) => {
            let mut operands = Vec::new();
            if let Some(gate) = &op.gate {
                operands.push(operand("gate", gate));
            }
            operands.push(operand("up", &op.up));
            operands.push(operand("down", &op.down));
            ("dense", None, operands)
        }
        Some(LayerFfn::Routed(op)) => (
            "routed",
            Some((op.experts, op.top_k)),
            vec![operand("router", &op.router)],
        ),
        Some(LayerFfn::Hybrid(_)) => ("hybrid", None, Vec::new()),
    };

    ExplainLayer {
        layer: index,
        ops,
        attention,
        ffn: ExplainFfn {
            kind: kind.into(),
            experts,
            operands: ffn_operands,
        },
    }
}

fn operand(role: &str, op: &OperandRef) -> ExplainOperand {
    ExplainOperand {
        role: role.to_string(),
        object: op.object.clone(),
        tensor: op.tensor.clone(),
        dtype: op.dtype.clone(),
        shape: op.shape.clone(),
    }
}
