//! Normalisation kinds and the scope their statistic reduces over.

use serde::{Deserialize, Serialize};

/// Normalization type used by the model.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NormType {
    /// RMSNorm (Gemma, Llama)
    RmsNorm,
    /// Standard LayerNorm (GPT-2, BERT)
    LayerNorm,
}

/// One normalisation operation, complete.
///
/// Every fact the arithmetic needs, at the site that consumes it. There
/// is deliberately no "the model's norm kind/epsilon/offset" anywhere
/// above this: Muse-Glimmer has proven twice that norm facts are
/// *per-site*, not model-scope — its post-norms use a different epsilon
/// from its pre-norms (1e-8 vs 1e-5), and its final norm uses a
/// different weight offset from its layer norms (0.0 vs 1.0). A field
/// meaning "the model's norm X" is a latent bug wherever more than one
/// norm site exists.
///
/// `weight_offset` carries the affine convention rather than a separate
/// norm kind: upstream's centred variant is
/// `RMSNorm(x, eps) * (weight + 1.0)`, which is this type with
/// `weight_offset: 1.0`. Keeping it a number rather than an enum variant
/// means a future architecture with some other offset needs no new
/// kind, and the executor's arithmetic never branches on provenance.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct NormSpec {
    pub kind: NormType,
    pub eps: f64,
    pub weight_offset: f32,
}

/// A normalisation applied to the embedding table's output, before the
/// residual stream begins.
///
/// **Weightless by construction.** A normalisation with learned weights
/// ships a tensor, and that tensor would arrive at operand closure as an
/// unclassified operand — which refuses, and forces the judgment. This
/// type exists for the case no tensor can evidence: Muse-Glimmer's
/// `MuseGlimmerTextNormedEmbedding` RMS-normalises every looked-up row
/// with `with_scale=False`, so nothing in the checkpoint records that it
/// happens at all.
///
/// It is the same blind spot as the attention gate and the parameter-free
/// QK norm: no operand to classify, and all four G4 authorities agree on
/// a model that simply lacks the operation. It was found by the upstream
/// oracle, not by any gate — plane 000 disagreed by a *pure per-row
/// rescale*, which is exactly what a missing weightless norm looks like.
///
/// `eps` is a concrete value, not a reference to another norm's epsilon:
/// the family resolves it from its own config when it declares the
/// judgment, so the operation site carries the number it will execute
/// with and execution inherits nothing.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingNorm {
    pub kind: NormType,
    pub eps: f64,
}

/// Which epsilon the post-norms of a four-norm stack use.
///
/// A four-norm decoder normalises attention and FFN *output* as well as
/// their inputs, and the two groups need not share an epsilon: Muse-Glimmer
/// declares `rms_norm_eps` 1e-5 for its pre-norms and `post_norm_eps` 1e-8
/// for its post-norms — three orders of magnitude apart.
///
/// The variants exist so that "the source establishes that they share" is a
/// different statement from "nobody has established what the post-norms
/// use". Wrapped in `Option`, the three representable states are:
///
/// - `None` — unjudged. A four-norm stack in this state is not executable,
///   and closure refuses it rather than inheriting a plausible value.
/// - `Some(Shared)` — the source semantics establish that post-norms use
///   the pre-norm epsilon. A judgment, not a fallback.
/// - `Some(Value(e))` — the source declares a distinct epsilon.
///
/// Carries `f64` rather than `f32` so a declared `1e-8` survives exactly;
/// `norm_eps` reaches the surface through an `f32` round-trip (recording
/// `1e-5` as `9.999999747e-06`) and this must not inherit that artefact.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PostNormEps {
    /// Post-norms use the pre-norm epsilon, established as a judgment.
    Shared,
    /// Post-norms use this distinct, explicitly declared epsilon.
    Value(f64),
}

impl PostNormEps {
    /// The epsilon these post-norms actually apply, given the pre-norm one.
    ///
    /// The single place a `Shared` judgment is turned into a number. It
    /// takes the pre-norm epsilon as an argument rather than reading it
    /// from anywhere, so sharing can never be applied to a value the
    /// caller did not intend.
    pub fn resolve(self, pre_norm_eps: f64) -> f64 {
        match self {
            PostNormEps::Shared => pre_norm_eps,
            PostNormEps::Value(eps) => eps,
        }
    }
}

/// Parameter-free projection normalisation: RMS-normalise Q, K and/or V
/// per head with no learned weight tensors. Distinct from weighted QK-norm
/// (whose weights exist in the stack and carry [`QkNormScope`]) — a judged
/// semantic fact, evidenced by an implementation that normalises while the
/// operand estate ships no weights for it. `v` is Gemma 4's `v_norm`
/// (`Gemma4RMSNorm(with_scale=False)` on the value states, every layer,
/// alongside WEIGHTED q/k norms) — a family can mix the two, so V is its
/// own flag rather than a "qk" pair. Defaults for inventories written
/// before V was recorded.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ParameterFreeQkNorm {
    pub q: bool,
    pub k: bool,
    #[serde(default)]
    pub v: bool,
}

/// The vector over which a QK-norm's RMS statistic is reduced.
///
/// Both variants apply the *same* weight vector elementwise across the whole
/// projection. They differ only in the denominator, and that difference is not
/// visible in the tensor names — OLMoE and Qwen3 both store
/// `self_attn.{q,k}_norm.weight`, and `transformers`' own Qwen3 module marks
/// the distinction in a comment (`# unlike olmo, only on the head dim!`)
/// rather than in a type. It is not always visible in the shapes either:
/// OLMoE-1B-7B is MHA with `num_heads * head_dim == hidden_size`, so its
/// `[2048]` weight is exactly as wide as a per-head convention would imply if
/// you only counted elements.
///
/// Getting it wrong rescales every head to a common norm, which discards the
/// relative magnitude *between* heads — a structural change to attention, not
/// a rounding one. See `docs/k3-funnel.md` §4.7.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QkNormScope {
    /// RMS over each head's `head_dim` slice independently.
    /// `Qwen3RMSNorm(head_dim)` — Qwen3, Gemma 3/4.
    PerHead,
    /// RMS over the entire `num_heads * head_dim` projection, before any
    /// reshape into heads. `OlmoeRMSNorm(hidden_size)` — OLMoE.
    FullProjection,
}
