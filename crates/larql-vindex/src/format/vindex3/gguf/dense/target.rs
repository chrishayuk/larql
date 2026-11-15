//! What a dense-softmax GGUF target is, as data.
//!
//! A target is llama.cpp's view of one architecture: the tensor names its
//! loader binds, the arch string its metadata keys hang off, and the
//! behaviour its graph builds in without reading a key (Gemma 3 scales the
//! embedding by `sqrt(hidden)` because the code says so, not because the
//! file does). Those built-in behaviours are facts about the *runtime*, so
//! they live here, and the preflight compares them with what the
//! container's graph declares. When the two disagree the export refuses:
//! llama.cpp would load the file and run a different model.
//!
//! Adding a family is adding a [`DenseTarget`] value and registering it in
//! [`TARGETS`]. Nothing in the pipeline below names a family.

use larql_models::config::Activation;

use super::gemma3::GEMMA3;

/// A semantic operand role the dense lowering binds. Assigned from the
/// operation plan, never from a tensor name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DenseRole {
    Embedding,
    FinalNorm,
    /// An untied output head. A tied head reuses the embedding operand and
    /// is never emitted (llama.cpp ties when `output.weight` is absent).
    OutputHead,
    PreAttentionNorm,
    PostAttentionNorm,
    PreFfnNorm,
    PostFfnNorm,
    Query,
    Key,
    Value,
    AttentionOutput,
    QueryNorm,
    KeyNorm,
    FfnGate,
    FfnUp,
    FfnDown,
}

impl DenseRole {
    /// Model-scope roles carry no layer index.
    pub fn is_global(self) -> bool {
        matches!(self, Self::Embedding | Self::FinalNorm | Self::OutputHead)
    }

    /// Roles whose operand is a norm weight.
    pub fn is_norm(self) -> bool {
        matches!(
            self,
            Self::FinalNorm
                | Self::PreAttentionNorm
                | Self::PostAttentionNorm
                | Self::PreFfnNorm
                | Self::PostFfnNorm
                | Self::QueryNorm
                | Self::KeyNorm
        )
    }
}

/// How the runtime scales attention scores, built into its graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionScaleRule {
    /// `1 / sqrt(head_dim)`.
    InverseSqrtHeadDim,
}

impl AttentionScaleRule {
    pub fn value(self, head_dim: usize) -> f64 {
        match self {
            Self::InverseSqrtHeadDim => 1.0 / (head_dim as f64).sqrt(),
        }
    }
}

/// How the runtime scales the looked-up embedding row, built into its
/// graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingScaleRule {
    /// `sqrt(hidden_size)`.
    SqrtHidden,
}

impl EmbeddingScaleRule {
    pub fn value(self, hidden_size: usize) -> f64 {
        match self {
            Self::SqrtHidden => (hidden_size as f64).sqrt(),
        }
    }
}

/// One llama.cpp architecture, as the exporter needs to know it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DenseTarget {
    /// The container `family` values this target lowers (`index.json`).
    pub families: &'static [&'static str],
    /// `general.architecture`, and the prefix of every arch key.
    pub architecture: &'static str,
    /// Prefix of per-layer tensor names (`blk` → `blk.3.attn_q.weight`).
    pub layer_prefix: &'static str,
    /// Role → tensor name (per-layer names without the `blk.N.` prefix).
    pub tensor_names: &'static [(DenseRole, &'static str)],
    /// Roles every layer must bind — the runtime's loader requires them.
    pub required_layer_roles: &'static [DenseRole],
    /// The FFN activation the runtime's graph applies.
    pub activation: Activation,
    pub attention_scale: AttentionScaleRule,
    pub embedding_scale: EmbeddingScaleRule,
    /// The sliding-window period the runtime assumes when the file does
    /// not state one. A model at a different period gets the key written.
    pub default_swa_period: usize,
    /// The context length to state when the container predates
    /// `execution.context_length`: the value the family's reference
    /// configuration class defaults to, which is also what llama.cpp's
    /// converter writes for a config that omits it. Reported whenever used.
    pub default_context_length: u64,
    /// `tokenizer.ggml.add_space_prefix`, which the reference converter
    /// writes as a constant for this architecture.
    pub add_space_prefix: bool,
}

impl DenseTarget {
    /// The GGUF name for `role`, or `None` when this target has no name
    /// for it.
    pub fn tensor_name(&self, role: DenseRole, layer: Option<usize>) -> Option<String> {
        let suffix = self
            .tensor_names
            .iter()
            .find(|(r, _)| *r == role)
            .map(|(_, n)| *n)?;
        match (role.is_global(), layer) {
            (true, None) => Some(suffix.to_string()),
            (false, Some(l)) => Some(format!("{}.{l}.{suffix}", self.layer_prefix)),
            _ => None,
        }
    }

    /// `<architecture>.<suffix>`.
    pub fn key(&self, suffix: &str) -> String {
        format!("{}.{suffix}", self.architecture)
    }
}

/// Every registered dense target.
pub const TARGETS: &[&DenseTarget] = &[&GEMMA3];

/// The target for a container's declared family.
pub fn target_for_family(family: &str) -> Option<&'static DenseTarget> {
    TARGETS
        .iter()
        .copied()
        .find(|t| t.families.contains(&family))
}

/// The declared families every registered target accepts, for refusals.
pub fn supported_families() -> Vec<&'static str> {
    TARGETS
        .iter()
        .flat_map(|t| t.families.iter().copied())
        .collect()
}
