//! Layer attention, layer FFN and the per-layer plan.

use super::super::graph::policy::AttentionSpan;
use larql_models::config::ResidualTopology;
use serde::Serialize;

#[allow(unused_imports)]
use super::*;

impl LayerAttention {
    /// The softmax op, when this layer attends by softmax. `None` on a
    /// DeltaNet or MLA layer — which is the point: a consumer that needs
    /// a span, a KV shape or a head geometry must handle the absence
    /// rather than receive a fabricated one.
    pub fn softmax(&self) -> Option<&AttentionOp> {
        match self {
            Self::Softmax(op) => Some(op.as_ref()),
            Self::GatedDelta(_)
            | Self::Kda(_)
            | Self::Mla(_)
            | Self::Mamba2(_)
            | Self::ConvQkv(_) => None,
        }
    }

    /// [`Self::softmax`], mutably.
    pub fn softmax_mut(&mut self) -> Option<&mut AttentionOp> {
        match self {
            Self::Softmax(op) => Some(op.as_mut()),
            Self::GatedDelta(_)
            | Self::Kda(_)
            | Self::Mla(_)
            | Self::Mamba2(_)
            | Self::ConvQkv(_) => None,
        }
    }

    /// The Gated DeltaNet op, when this layer runs that recurrence.
    ///
    /// `None` on a KDA layer — the two are different operators, and a
    /// consumer that wants "whichever recurrence this is" must ask for
    /// each rather than receive one standing in for the other.
    pub fn gated_delta(&self) -> Option<&GatedDeltaOp> {
        match self {
            Self::GatedDelta(op) => Some(op.as_ref()),
            Self::Softmax(_) | Self::Kda(_) | Self::Mla(_) | Self::Mamba2(_) | Self::ConvQkv(_) => {
                None
            }
        }
    }

    /// The KDA op, when this layer runs Kimi Delta Attention.
    pub fn kda(&self) -> Option<&KdaOp> {
        match self {
            Self::Kda(op) => Some(op.as_ref()),
            Self::Softmax(_)
            | Self::GatedDelta(_)
            | Self::Mla(_)
            | Self::Mamba2(_)
            | Self::ConvQkv(_) => None,
        }
    }

    /// The MLA op, when this layer runs Multi-Latent Attention.
    pub fn mla(&self) -> Option<&MlaOp> {
        match self {
            Self::Mla(op) => Some(op.as_ref()),
            Self::Softmax(_)
            | Self::GatedDelta(_)
            | Self::Kda(_)
            | Self::Mamba2(_)
            | Self::ConvQkv(_) => None,
        }
    }

    /// The Mamba2 op, when this layer runs the SSD mixer.
    pub fn mamba2(&self) -> Option<&Mamba2Op> {
        match self {
            Self::Mamba2(op) => Some(op.as_ref()),
            Self::Softmax(_)
            | Self::GatedDelta(_)
            | Self::Kda(_)
            | Self::Mla(_)
            | Self::ConvQkv(_) => None,
        }
    }

    /// Elements of recurrent state this layer retains, or `None` when it
    /// keeps a per-position cache instead — a softmax OR an MLA layer:
    /// MLA compresses the cache, it does not remove it, so it answers
    /// `None` here exactly as plain softmax does, not a state-elements
    /// count.
    ///
    /// The question a KV planner asks once a stack may hold either
    /// recurrence: both answer in state elements, and neither answers in
    /// positions.
    pub fn recurrent_state_elements(&self) -> Option<usize> {
        match self {
            // Conv-QKV attention keeps a per-position cache (its conv
            // history is a fixed-size extra region, not a recurrence),
            // so it answers as the cache-keeping operators do.
            Self::Softmax(_) | Self::Mla(_) | Self::ConvQkv(_) => None,
            Self::GatedDelta(op) => Some(op.state_elements()),
            Self::Kda(op) => Some(op.state_elements()),
            Self::Mamba2(op) => Some(op.state_elements()),
        }
    }

    /// The `layer_types` spelling this operator corresponds to, for
    /// comparing what the plan carries against what the checkpoint
    /// declared.
    pub fn declared_name(&self) -> &'static str {
        match self {
            Self::Softmax(op) => op.span.declared_name(),
            Self::GatedDelta(_) | Self::Kda(_) | Self::Mamba2(_) => {
                larql_models::config::LAYER_TYPE_LINEAR_ATTENTION
            }
            // The hybrid's index-set spelling names WHICH layers attend;
            // the span vocabulary is its round-trip, as in the graph's
            // identical judgment.
            Self::ConvQkv(_) => AttentionSpan::Full.declared_name(),
            // MLA has no `layer_types` spelling of its own — see
            // `graph::policy::AttentionLayerPolicy::declared_name`'s
            // identical judgment.
            Self::Mla(_) => AttentionSpan::Full.declared_name(),
        }
    }
}

/// One layer's FFN: dense, routed, or both. Untagged, so a dense layer
/// serialises exactly as its [`FfnOp`] always has — a dense plan is
/// byte-identical before and after routed and hybrid FFNs existed.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum LayerFfn {
    /// All boxed: the ops differ several-fold in size and a plan holds one
    /// per layer; the untagged serialisation is unaffected.
    Dense(Box<FfnOp>),
    Routed(Box<RoutedFfnOp>),
    Hybrid(Box<HybridFfnOp>),
}

impl LayerFfn {
    /// The dense op, when this layer's FFN is dense ONLY. A hybrid layer's
    /// dense half is reached through [`Self::hybrid`] — an executor that
    /// ran only the dense half of a hybrid layer would run a different
    /// model, so this does not hand it out.
    pub fn dense(&self) -> Option<&FfnOp> {
        match self {
            Self::Dense(op) => Some(op.as_ref()),
            Self::Routed(_) | Self::Hybrid(_) => None,
        }
    }

    /// The routed op, when this layer's FFN is a mixture of experts ONLY
    /// (same reasoning as [`Self::dense`]).
    pub fn routed(&self) -> Option<&RoutedFfnOp> {
        match self {
            Self::Routed(op) => Some(op.as_ref()),
            Self::Dense(_) | Self::Hybrid(_) => None,
        }
    }

    /// The hybrid op, when this layer runs both branches.
    pub fn hybrid(&self) -> Option<&HybridFfnOp> {
        match self {
            Self::Hybrid(op) => Some(op.as_ref()),
            Self::Dense(_) | Self::Routed(_) => None,
        }
    }
}

/// The generic program of one decoder layer. Norm placement is explicit
/// op positions, not a count: under two-norm placement the
/// `post_attention_layernorm` operand *is* [`Self::pre_ffn_norm`] and the
/// post-positions are absent; under four-norm placement attention and FFN
/// are each wrapped pre + post.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LayerPlan {
    pub layer: usize,
    /// The pre-block norm. On an attention-class layer this is
    /// `input_layernorm`; on a mixer-only layer it is the layer's single
    /// pre-mixer norm — same position in the program, one field.
    ///
    /// `None` under [`NormPlacement::PostOnly`](crate::format::vindex3::graph::NormPlacement):
    /// the sublayer reads the RAW residual there and the norm applies to
    /// its output instead. Absence is the program, not a missing operand
    /// — closure requires this norm's operand exactly when the placement
    /// says the site exists.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pre_attention_norm: Option<NormOp>,
    /// The component's DECLARED norm epsilon (`rms_norm_eps`).
    ///
    /// Carried on the layer in its own right rather than read back off
    /// [`Self::pre_attention_norm`], because the operators that need it
    /// are not all norms at that site: QK norm, DeltaNet's gated RMSNorm
    /// and KDA's all run at this epsilon. Reading it from one norm SITE
    /// was a coupling between an epsilon and a placement — two unrelated
    /// facts — and it is what made a post-norm stack, which has no
    /// pre-attention norm to borrow from, unrepresentable in the op plan.
    ///
    /// The same reasoning `MlaOp::kv_a_norm_eps` already applies in the
    /// other direction: a norm whose epsilon differs from the layer's
    /// carries its own, and one that shares it reads this.
    pub declared_norm_eps: f64,
    /// This layer's attention-class operator. Not every layer attends by
    /// softmax: a hybrid checkpoint interleaves DeltaNet recurrences with
    /// full-attention layers, so the op is a choice, not a shape.
    pub attention: LayerAttention,
    /// Normalises attention output before its residual add (four-norm
    /// placement only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub post_attention_norm: Option<NormOp>,
    /// Absent on a mixer-only (Mamba2) layer, which has no FFN to
    /// normalise into — schema 6's presence-means-presence, in the plan.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pre_ffn_norm: Option<NormOp>,
    /// This layer's FFN, absent on a mixer-only layer (the mixer is the
    /// whole block; no `intermediate_size` exists in the checkpoint).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ffn: Option<LayerFfn>,
    /// Normalises FFN output before its residual add (four-norm only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub post_ffn_norm: Option<NormOp>,
    /// A learned scalar `[1]` the whole layer output is multiplied by,
    /// after the FFN residual add (Gemma 4 `layer_scalar`). Present iff
    /// the layer ships the operand — an absent scalar is no multiply, not
    /// a multiply by one that a reader may assume.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layer_scale: Option<OperandRef>,
    /// The two Sinkhorn hyper-connection sites this layer wraps its
    /// sublayers in — present iff the component's
    /// [`ComponentOpPlan::residual_topology`] is a hyper-connection, and
    /// then on every layer (closure requires all six operands). `None` on
    /// a single-stream component, whose serialisation is unchanged.
    ///
    /// Carried so the wave-17 executor can be instantiated from a
    /// checkpoint's own addresses; NOT a claim that this build traverses
    /// the bundle. That refusal lives in
    /// [`exec::prepared`](crate::format::vindex3::opplan::exec::prepared)
    /// and reads the same authority the plan report does.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hyper_connection: Option<HyperConnectionLayerOp>,
    /// The two attention-residual sites this layer wraps its sublayers
    /// in — present iff the component declares
    /// [`ResidualTopology::AttentionResidual`], and then on every layer
    /// (closure requires all four operands). `None` otherwise, so a
    /// plan without the topology serialises exactly as it did.
    ///
    /// The pair is bound at BOTH sites even though the reference's
    /// attention site does not reduce at layer 0. Absence of a
    /// REDUCTION is a schedule fact, decided per layer by the snapshot
    /// set; absence of an OPERAND would be an estate fact, and the
    /// checkpoint ships all four on every layer.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attention_residual: Option<AttentionResidualLayerOp>,
    /// Residual-stream scaling: the attention/FFN sublayer's own output
    /// (after any post-norm above) is multiplied by this immediately
    /// before its residual add, at both sites with the same value.
    /// `None` = the op is absent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub residual_scale: Option<f32>,
    /// Operand accounting: consumed by the ops above / present in the
    /// segment for this layer. Closure requires equality.
    pub operands_accounted: usize,
    pub operands_present: usize,
}

/// Embedding lookup.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EmbeddingOp {
    pub table: OperandRef,
    /// Weightless normalisation of the looked-up row. `None` = absent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub norm: Option<larql_models::config::EmbeddingNorm>,
    /// The embedding-scale operation. `None` = the op is absent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scale: Option<f32>,
    pub vocab_size: usize,
    /// The learned input position table added to every looked-up row,
    /// AFTER its scale and norm: `h0[p] = embed(token[p]) + table[p]`.
    /// Present iff the head surface declares an input position encoding.
    /// Absent from the serialised op otherwise.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub position: Option<PositionTableOp>,
}

/// A learned absolute position table and the encoding it implements.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PositionTableOp {
    pub encoding: larql_models::config::InputPositionEncoding,
    /// `[max_positions, hidden]`.
    pub table: OperandRef,
}

/// The output head.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct OutputOp {
    pub projection: OperandRef,
    /// The output-multiplier operation. `None` = the op is absent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub multiplier: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub softcapping: Option<f32>,
}

/// One Sinkhorn hyper-connection SITE's operands (wave 18) — what the
/// five wave-17 stages read for one sublayer.
///
/// ```text
/// mix_fn   [(2 + hc)·hc, hc·hidden]   stage 1's dynamic mix projection
/// base     [(2 + hc)·hc]              stage 2's additive logit offset
/// scale    [3]                        stage 2's pre / post / comb scalars
/// ```
///
/// Geometry is checked at closure against the component's declared
/// stream count, so a bound site is one the executor's `SiteWeights` can
/// be built from without a further length check. Dtype is whatever the
/// checkpoint stored (F32 on DeepSeek-V4, BF16 on GLM-5.3-Flash): the
/// operand reference carries it, the role does not.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct HcSiteOp {
    pub mix_fn: OperandRef,
    pub base: OperandRef,
    pub scale: OperandRef,
}

/// The two sites one hyper-connected layer wraps its sublayers in.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct HyperConnectionLayerOp {
    /// Reduces the bundle before attention and expands its output after.
    pub attention: HcSiteOp,
    /// The same around the FFN.
    pub ffn: HcSiteOp,
}

/// The hyper-connection HEAD's own reduction operands — a different
/// operation from a site's, with different geometry (one row per stream,
/// a single scalar, no Sinkhorn), bound from the component's
/// `hyper_connection_head` object.
///
/// ```text
/// reduce_fn   [hc, hc·hidden]
/// base        [hc]
/// scale       [1]
/// ```
///
/// Optional even on a hyper-connected component: GLM-5.3-Flash declares
/// the topology and ships no head operands at all, and what reduces its
/// bundle before the final norm is not something this build has read
/// (its `mhc` flag is recorded as unknown, not guessed). A plan with the
/// topology and no head is therefore representable and, for a second
/// reason, not executable.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct HyperConnectionHeadOp {
    pub reduce_fn: OperandRef,
    pub base: OperandRef,
    pub scale: OperandRef,
}

/// One attention-residual site's operand pair: a `[hidden]` norm weight
/// and a `[1, hidden]` projection, multiplied elementwise into ONE
/// learned score vector. There is no query and no per-token projection
/// of the state, which is why the site stores two vectors rather than a
/// mix projection.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AttnResSiteOp {
    pub norm: OperandRef,
    pub proj: OperandRef,
}

/// The two attention-residual sites of one layer.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AttentionResidualLayerOp {
    /// The site before attention. The reference guards its reduction on
    /// a non-empty snapshot set, so layer 0 binds this pair and reduces
    /// with it on no layer where the set is empty.
    pub attention: AttnResSiteOp,
    /// The site before the FFN. UNCONDITIONAL in the reference.
    pub ffn: AttnResSiteOp,
}

/// The attention-residual EXIT's pair, bound from the component's
/// `attention_residual_exit` object.
///
/// A site's geometry and a site's arithmetic, and still not a site: it
/// runs once, at the stack's end, over the whole snapshot history, and
/// the declaration REQUIRES it — unlike the hyper-connection head, which
/// GLM-5.3-Flash declines to ship. A component that declares the period
/// and owns no exit object is refused before this field could be filled.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AttentionResidualExitOp {
    pub norm: OperandRef,
    pub proj: OperandRef,
}

/// The complete generic program of one component.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ComponentOpPlan {
    pub component: String,
    /// How the residual stream is shaped across the whole component — a
    /// COMPONENT fact, carried once so the executor, the embedding and
    /// the head cannot disagree about it. Serialised only when it is not
    /// the single stream every plan before wave 18 implicitly carried, so
    /// those plans' JSON is byte-identical.
    #[serde(skip_serializing_if = "ResidualTopology::is_single_stream")]
    pub residual_topology: ResidualTopology,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding: Option<EmbeddingOp>,
    pub layers: Vec<LayerPlan>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub final_norm: Option<NormOp>,
    /// The hyper-connection head's reduction, when the component owns
    /// one. See [`HyperConnectionHeadOp`] for why this is optional even
    /// under the topology.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hyper_connection_head: Option<HyperConnectionHeadOp>,
    /// The attention-residual exit reduction, present iff the component
    /// declares the topology — where the hyper-connection head is
    /// optional under its own topology, this is not.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attention_residual_exit: Option<AttentionResidualExitOp>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<OutputOp>,
}

/// Which FFN a layer runs, as a declaration or as operand evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum FfnIdentity {
    /// A dense MLP.
    Dense,
    /// A routed expert block (routed alone, or routed-plus-dense hybrid).
    Routed,
}

impl FfnIdentity {
    /// Lower-case name for messages.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Dense => "dense",
            Self::Routed => "routed",
        }
    }
}
