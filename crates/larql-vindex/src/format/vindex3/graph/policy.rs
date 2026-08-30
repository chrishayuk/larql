//! Per-layer attention policy as the graph records it.

use larql_models::config::{
    PositionPolicy, LAYER_TYPE_FULL_ATTENTION, LAYER_TYPE_LINEAR_ATTENTION,
    LAYER_TYPE_SLIDING_ATTENTION, LAYER_TYPE_WINDOW_ATTENTION,
};
use serde::{Deserialize, Serialize};

/// Which recurrence a checkpoint's declared geometry identifies.
///
/// The input `resolve_layer_kind` needs and `layer_types` cannot supply:
/// every linear-attention family spells itself `linear_attention`, so the
/// operator is named by the geometry the config declares, never by the
/// label. `None` means a recurrence was declared and no geometry
/// identified it — [`LayerOperator::Recurrent`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecurrenceKind {
    /// Qwen3.8's `linear_*` geometry.
    GatedDelta,
    /// `linear_attn_config`'s KDA geometry.
    Kda,
    /// The Mamba2/SSD geometry (`state_size`, `num_heads`, `expand`, …).
    Mamba2,
}

/// Which attention-class operator a layer runs.
///
/// Separate from [`AttentionSpan`] on purpose, and this is the whole point
/// of the type: a span answers *how far back this layer's softmax
/// attends*, and consumers read it as **KV liveness**. A Gated DeltaNet
/// layer has no answer — nothing it retains is indexed by position, so
/// there is no prefix to bound. Spelling `linear_attention` as a span
/// would hand a KV planner a number that looks like liveness and is not,
/// which is exactly the defect this rung exists to remove: before it,
/// every one of Qwen3.8's 48 recurrent layers resolved to
/// [`AttentionSpan::Full`] and the graph reported a 64-layer full-attention
/// tower.
///
/// The op plan's `LayerAttention` is the executable form of this same
/// distinction; this is the graph's, carrying the kind without the
/// operands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerOperator {
    /// Scaled dot-product attention over a per-position key/value cache.
    /// The only operator containers written before this field existed
    /// could describe, which is why it is the deserialisation default.
    #[default]
    Softmax,
    /// Gated DeltaNet recurrence: one dense `Dk × Dv` state per value
    /// head, no per-position key or value, no span, no softmax.
    GatedDelta,
    /// Kimi Delta Attention: split q/k/v projections each through their
    /// own depthwise causal conv, low-rank f (decay) and g (output) gates,
    /// a per-**channel** timestep bias `[Hv·Dv]`, and one `Dk × Dv` state
    /// per head.
    ///
    /// A sibling of [`Self::GatedDelta`], never a mode of it. The operand
    /// contracts differ structurally — fused vs split projections, one
    /// conv vs three, full-rank vs low-rank gates, `[Hv]` vs `[Hv·Dv]`
    /// `dt_bias` — so a checkpoint of either kind read as the other would
    /// bind the wrong tensors to the wrong roles with plausible shapes.
    /// Observed on Kimi Linear (20 layers) and GLM-5.3-Flash (34).
    Kda,
    /// Mamba2 / SSD: one fused `in_proj` emitting z|x|B|C|dt, one
    /// depthwise causal conv over the x|B|C channels only, per-**head**
    /// scalar decay (`A_log`), skip (`D`) and timestep bias, a gated
    /// RMSNorm between state read-out and `out_proj`, and one
    /// `head_dim × state_size` state per head.
    ///
    /// A third sibling, never a mode of [`Self::GatedDelta`] or
    /// [`Self::Kda`]: the operand contracts differ structurally — one
    /// fused five-way projection vs DeltaNet's qkv|a|b|z split or KDA's
    /// three separate convs, a per-head scalar decay vs full/low-rank
    /// gates, and a conv that deliberately excludes the gate channels.
    /// Observed on mamba2-780m (48 layers, zero attention anywhere).
    Mamba2,
    /// A declared recurrence whose operator family this build could not
    /// identify — the checkpoint says `linear_attention`, and nothing
    /// resolved the geometry that would say *which* recurrence.
    ///
    /// This variant exists because the spelling and the operator are two
    /// different facts, and only one of them is evidence. Before it,
    /// every `linear_attention` layer resolved to [`Self::GatedDelta`]
    /// from the string alone, so a stack whose recurrence this build has
    /// never implemented was reported as executable Gated DeltaNet:
    /// GLM-5.3-Flash's 34 KDA layers and Kimi Linear's 20 both graded
    /// that way while every `linear_attn_config.*` key that would supply
    /// the operator's geometry graded `unrepresented`
    /// (`docs/glm5-flash-funnel.md` §4.2).
    ///
    /// [`AttentionLayerPolicy::declared_name`] answers `None` here, so
    /// such a layer counts as *unexpressed* rather than as a recurrence
    /// this build can run — the fail-closed direction.
    Recurrent,
    /// Multi-Latent Attention: query is one dense projection (or, on a
    /// family that compresses it too, a low-rank pair this build does not
    /// model yet), key/value are a SHARED low-rank latent
    /// (`kv_a_proj_with_mqa`) decompressed per head
    /// (`kv_b_proj`/`kv_a_layernorm`) into an asymmetric nope+rope query
    /// width and a DIFFERENT value width. Retains a per-position KV cache
    /// (compressed, not absent) — unlike [`Self::Kda`]/[`Self::
    /// GatedDelta`] it is not a recurrence, and unlike [`Self::Softmax`]
    /// its Q/K/V/O operands do not share that operator's shape contract:
    /// `self_attn.q_proj.weight`/`o_proj.weight` are byte-identical
    /// SUFFIXES to the softmax set at a DIFFERENT width (Kimi Linear:
    /// `q_proj` `[32·192, hidden]` against a softmax layer's `[32·head_dim,
    /// hidden]`), the same collision [`Self::Kda`]'s role table exists to
    /// resolve, one operator over.
    Mla,
}

impl LayerOperator {
    /// Whether this layer runs the one recurrence this build implements.
    ///
    /// Deliberately **not** "is it recurrent": [`Self::Recurrent`] is also
    /// a recurrence, and the whole point of that variant is that no
    /// executable operator has been identified for it. Every consumer
    /// that asks this question wants the executable answer, so the name
    /// says which question it is.
    pub fn is_gated_delta(&self) -> bool {
        matches!(self, Self::GatedDelta)
    }

    /// Whether this build has an executor for this operator.
    ///
    /// Deliberately separate from whether the operator is *represented*.
    /// A container can describe a KDA layer completely — every operand
    /// bound, every dimension stated — and still have nothing able to run
    /// it, and those two facts must not be read off one another. Collapsing
    /// them is how a stack whose operator was merely *named* came to be
    /// reported as executable Gated DeltaNet.
    ///
    /// `declared → represented → executable`: this predicate is the last
    /// arrow, and it is the one a plan cannot infer from the checkpoint.
    pub fn has_executor(&self) -> bool {
        match self {
            Self::Softmax | Self::GatedDelta | Self::Mamba2 => true,
            // Represented, not executable — the operand contract is
            // complete and no executor consumes it yet.
            Self::Kda | Self::Mla => false,
            Self::Recurrent => false,
        }
    }

    /// Whether this layer runs the Mamba2/SSD mixer.
    pub fn is_mamba2(&self) -> bool {
        matches!(self, Self::Mamba2)
    }

    /// Whether this layer runs Kimi Delta Attention.
    pub fn is_kda(&self) -> bool {
        matches!(self, Self::Kda)
    }

    /// Whether this layer runs Multi-Latent Attention.
    pub fn is_mla(&self) -> bool {
        matches!(self, Self::Mla)
    }

    /// Whether this layer is a recurrence at all, identified or not — the
    /// question a KV planner asks, because none of these retain a
    /// per-position prefix.
    pub fn is_recurrent(&self) -> bool {
        matches!(
            self,
            Self::GatedDelta | Self::Kda | Self::Mamba2 | Self::Recurrent
        )
    }

    /// Whether this layer is a recurrence whose operator family this build
    /// could not identify — see [`Self::Recurrent`].
    ///
    /// Its own predicate rather than `!is_gated_delta()`, which would also
    /// be true of every softmax layer.
    pub fn is_unidentified_recurrence(&self) -> bool {
        matches!(self, Self::Recurrent)
    }
}

/// Attention span kind of one layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttentionSpan {
    /// Attends to the last `window` positions only.
    Sliding,
    /// Attends to the whole prefix.
    Full,
    /// Attends within a bounded region the component's own geometry
    /// defines — a perception tower's spatial window — rather than a
    /// trailing sequence window. No `window` count applies, because the
    /// extent is not a position count and the config does not declare
    /// one.
    ///
    /// Distinct from [`Self::Sliding`] on purpose. Aliasing the two would
    /// let a KV planner infer that positions beyond a window are dead,
    /// which is true of a sequence window and not of a spatial one;
    /// aliasing to [`Self::Full`] would erase the distinction the
    /// checkpoint actually declares (Muse-Glimmer's vision tower splits
    /// 37/13).
    Windowed,
}

impl AttentionSpan {
    /// The span a declared `layer_types` entry names, or `None` when the
    /// vocabulary does not contain it.
    ///
    /// Fail-closed by construction: an unrecognised spelling answers
    /// `None` so the caller refuses, rather than resolving to a
    /// behavioural default. That is the [§4.7.8] shape — `layer_types`
    /// was once parsed and validated but never consulted, so every model
    /// ran full attention on every layer — and the same shape one level
    /// up is what a "not sliding, therefore full" rule would reintroduce
    /// for any new spelling.
    ///
    /// [§4.7.8]: ../../../../../docs/k3-funnel.md
    pub fn from_declared(entry: &str) -> Option<Self> {
        if entry.eq_ignore_ascii_case(LAYER_TYPE_SLIDING_ATTENTION) {
            Some(Self::Sliding)
        } else if entry.eq_ignore_ascii_case(LAYER_TYPE_FULL_ATTENTION) {
            Some(Self::Full)
        } else if entry.eq_ignore_ascii_case(LAYER_TYPE_WINDOW_ATTENTION) {
            Some(Self::Windowed)
        } else {
            None
        }
    }

    /// The `layer_types` spelling this span corresponds to — the inverse
    /// of [`Self::from_declared`], used to compare what the graph carries
    /// against what the checkpoint declared.
    pub fn declared_name(self) -> &'static str {
        match self {
            Self::Sliding => LAYER_TYPE_SLIDING_ATTENTION,
            Self::Full => LAYER_TYPE_FULL_ATTENTION,
            Self::Windowed => LAYER_TYPE_WINDOW_ATTENTION,
        }
    }
}

/// One layer's attention policy: span, window, and positional encoding.
/// This is architectural liveness information — a KV planner reading it
/// knows that positions beyond `window` on a sliding layer are
/// *architecturally* dead, before any semantic analysis runs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AttentionLayerPolicy {
    /// Which attention-class operator this layer runs.
    ///
    /// Explicit since GRAPH_SCHEMA 6 (drill F7): the field is always
    /// written and required on read. The previous serde default let an
    /// absent field silently mean softmax — a reinterpretation, not a
    /// fact — and schema 6's rule is that presence means semantic
    /// presence. Pre-v6 graphs are refused by the schema check before
    /// deserialisation, so no compatibility default is needed here.
    pub operator: LayerOperator,
    /// How far back this layer's softmax attends.
    ///
    /// `None` exactly when no span exists to state — a
    /// [`LayerOperator::GatedDelta`] layer, or a declared spelling outside
    /// this schema's executable vocabulary. Deliberately an absence rather
    /// than a stand-in value: a consumer planning KV must handle "this
    /// layer has no prefix" instead of receiving a fabricated `Full`.
    ///
    /// `Some(x)` serialises exactly as the bare `x` did before this field
    /// became optional, so existing containers are unaffected.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub span: Option<AttentionSpan>,
    /// Window size when [`AttentionSpan::Sliding`]; `None` on full and
    /// windowed layers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub window: Option<usize>,
    /// How the layer encodes position — including intentional absence.
    pub position: PositionPolicy,
    /// This layer's head geometry when the family varies it by layer
    /// (Gemma 4: `head_dim` 256 / 8 KV heads on sliding layers,
    /// `global_head_dim` 512 / 2 KV heads on full layers). `None` = the
    /// container predates per-layer geometry and every layer has the
    /// component surface's geometry — an absence with one meaning, not
    /// a default: the graph builder always records `Some` today, so a
    /// `None` on a fresh encode is a bug, not a fallback.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub geometry: Option<HeadGeometry>,
    /// The value projection IS the key projection on this layer (Gemma 4
    /// `attention_k_eq_v`, full layers only): no V operand exists and V is
    /// the raw K projection, before the key's norm and rotation. Closure
    /// pairs it both ways — a V operand on such a layer is a stray, a
    /// missing V on any other layer is missing. Defaults for containers
    /// written before it was recorded.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub v_from_k: bool,
    /// The checkpoint's own `layer_types` entry for this layer, verbatim,
    /// when it declares one. `None` when the config states no per-layer
    /// array.
    ///
    /// Carried alongside [`Self::span`], never folded into it: `span` is
    /// this schema's *executable* three-way vocabulary, and a checkpoint
    /// declaring a spelling outside it (a hybrid linear-attention layer)
    /// still needs its raw declaration recorded rather than silently
    /// collapsed to whatever `span` defaulted to. Consumers that need to
    /// know whether `span` is a genuine resolution or a fallback default
    /// compare this field against `span` via [`AttentionSpan::from_declared`]
    /// — see `plan::carriage::probe_layer_types`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub declared_span: Option<String>,
}

impl AttentionLayerPolicy {
    /// The `layer_types` spelling this layer's resolved policy corresponds
    /// to, or `None` when the schema holds no vocabulary for what it
    /// resolved.
    ///
    /// Operator first: a recurrence is `linear_attention` whatever a span
    /// would have said, because it has no span. Only a softmax layer
    /// defers to [`AttentionSpan::declared_name`]. `None` is the
    /// fail-closed answer — a caller comparing against the checkpoint's
    /// own array must refuse rather than invent a spelling.
    pub fn declared_name(&self) -> Option<&'static str> {
        match self.operator {
            // Mamba2 rides the same canonical spelling: the resolved
            // inventory canonicalises every identified recurrence —
            // whatever key declared it — to `linear_attention`
            // (`layer_kind_spelling`), and this is its inverse.
            LayerOperator::GatedDelta | LayerOperator::Kda | LayerOperator::Mamba2 => {
                Some(LAYER_TYPE_LINEAR_ATTENTION)
            }
            // An unidentified recurrence has no executable spelling to
            // round-trip to. Answering `linear_attention` here would make
            // `matches_declaration` true and hide the layer inside the
            // faithful bucket — the exact collapse this variant exists to
            // prevent.
            LayerOperator::Recurrent => None,
            // MLA has no `layer_types` spelling of its own — the
            // checkpoint states its geometry through the MLA config keys,
            // not through a hybrid interleave entry — so it round-trips
            // through the same span vocabulary a softmax layer does.
            LayerOperator::Softmax | LayerOperator::Mla => {
                self.span.map(AttentionSpan::declared_name)
            }
        }
    }

    /// Whether this layer's carried policy round-trips to the spelling the
    /// checkpoint declared for it.
    ///
    /// A layer that declares nothing is vacuously faithful — there is no
    /// claim to contradict. A layer whose declaration this schema cannot
    /// express answers `false`, never "close enough".
    pub fn matches_declaration(&self) -> bool {
        match self.declared_span.as_deref() {
            None => true,
            Some(raw) => self
                .declared_name()
                .is_some_and(|name| raw.eq_ignore_ascii_case(name)),
        }
    }
}

/// Decide one layer's operator and span from the checkpoint's own
/// `layer_types` entry plus the resolved sliding/full boolean.
///
/// The single place this decision is made. `build.rs` records it into the
/// graph and `plan::compare` grades the checkpoint against it; two
/// implementations of the same rule would be free to drift, and the fact
/// they must agree on is exactly the one this rung is repairing.
///
/// Note what does **not** happen here: the span of a softmax layer is
/// taken from `resolved_sliding` — the boolean the parser derived — and
/// never from `declared`. Sourcing both sides of the comparison from the
/// declared array would make `plan::compare::layer_types_finding`
/// tautological, and a gate that cannot fail is not a gate.
pub fn resolve_layer_kind(
    declared: Option<&str>,
    resolved_sliding: bool,
    recurrence: Option<RecurrenceKind>,
    mla: bool,
) -> (LayerOperator, Option<AttentionSpan>) {
    match declared {
        // A declared recurrence. No span exists, and saying so is the
        // repair.
        //
        // *Which* recurrence is a second question, and `declared` cannot
        // answer it — the spelling is the same for every linear-attention
        // family. `recurrence_identified` carries whether the geometry
        // that names the operator actually resolved; without it this
        // returned `GatedDelta` from the string, which is a claim about
        // an operator made from evidence about a label.
        Some(raw) if raw.eq_ignore_ascii_case(LAYER_TYPE_LINEAR_ATTENTION) => (
            match recurrence {
                Some(RecurrenceKind::GatedDelta) => LayerOperator::GatedDelta,
                Some(RecurrenceKind::Kda) => LayerOperator::Kda,
                Some(RecurrenceKind::Mamba2) => LayerOperator::Mamba2,
                None => LayerOperator::Recurrent,
            },
            None,
        ),
        // A declared softmax spelling this vocabulary knows, or one it
        // does not. Either way the span comes from the resolved boolean,
        // so the comparison downstream stays meaningful; an unrecognised
        // spelling is caught there rather than silently absorbed here.
        //
        // A full (non-sliding) layer on an MLA family is Multi-Latent
        // Attention, not plain softmax — see `graph::build::
        // operator_and_span`'s identical judgment for the declared-kind
        // path; this is its layer-blind twin. Sliding stays softmax: no
        // evidence in this build associates MLA with a sliding span.
        Some(_) | None if mla && !resolved_sliding => {
            (LayerOperator::Mla, Some(AttentionSpan::Full))
        }
        Some(_) | None => (
            LayerOperator::Softmax,
            Some(if resolved_sliding {
                AttentionSpan::Sliding
            } else {
                AttentionSpan::Full
            }),
        ),
    }
}

/// One layer's attention head geometry. Query-head count is a component
/// fact (no judged family varies it by layer); the KV side and the head
/// width are what Gemma 4 varies, so those are the per-layer facts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct HeadGeometry {
    pub head_dim: usize,
    pub num_kv_heads: usize,
}
