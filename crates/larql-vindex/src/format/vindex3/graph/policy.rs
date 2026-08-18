//! Per-layer attention policy as the graph records it.

use larql_models::config::{
    PositionPolicy, LAYER_TYPE_FULL_ATTENTION, LAYER_TYPE_SLIDING_ATTENTION,
    LAYER_TYPE_WINDOW_ATTENTION,
};
use serde::{Deserialize, Serialize};

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
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AttentionLayerPolicy {
    pub span: AttentionSpan,
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
}

/// One layer's attention head geometry. Query-head count is a component
/// fact (no judged family varies it by layer); the KV side and the head
/// width are what Gemma 4 varies, so those are the per-layer facts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct HeadGeometry {
    pub head_dim: usize,
    pub num_kv_heads: usize,
}
