use super::enums::{
    Activation, MoeDownPaddingPolicy, MoeExpertScalePolicy, MoeInputSource,
    MoePostExpertNormPolicy, MoeRouterNormPolicy, MoeTopKWeightPolicy,
};
use super::quant_format::QuantFormat;

/// How one expert's gate and up projections combine into the down
/// projection's input — the compute-side form of
/// [`larql_models::ExpertGatePolicy`] with the activation folded in, so a
/// layer carries **one** combine rule rather than a policy and an
/// activation that can disagree.
///
/// The scalar core here is the single authority for the combine math on
/// slice paths; the `Array2` reference tier (`ffn::expert_weight::gate`)
/// delegates to it, so the PyTorch-pinned tests there cover this too.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MoeGateRule {
    /// Conventional gated MLP: `act(gate) * up`.
    Gated(Activation),
    /// GPT-OSS's clamped GLU, transcribed from `GptOssExperts._apply_gate`:
    /// one-sided clamp on gate, symmetric clamp on up, `alpha` scaling the
    /// sigmoid's argument, and the `(up + 1)` offset. None of it is SwiGLU;
    /// see `ffn::expert_weight::gate` for the full derivation.
    ClampedGlu { limit: f32, alpha: f32 },
    /// Kimi-K3's SiTU-GLU, transcribed from `SituAndMul.forward`
    /// (`modeling_kimi_linear.py` L75-82):
    ///
    /// ```text
    /// situ_a = beta * tanh(g / beta) * sigmoid(g)
    /// u      = linear_beta * tanh(u / linear_beta)   // only when declared
    /// out    = situ_a * u
    /// ```
    ///
    /// A softcapped SwiGLU, and it degenerates to SwiGLU as the bounds
    /// grow — which is why substituting SiLU looks healthy and is not.
    /// See `ffn::expert_weight::gate` for the oracle-pinned derivation.
    SituGlu { beta: f32, linear_beta: Option<f32> },
    /// GLM-5.3-Flash: GPT-OSS's clamp, then ORDINARY SwiGLU —
    /// `act(g) * u`, with neither the `alpha` scaling nor the `(u + 1)`
    /// offset. Its reference carries the comment "Simple swiglu instead
    /// of alpha" over exactly this line.
    ///
    /// A separate rule and not a flag, because the two differ by the
    /// arithmetic and not by the clamp. Serving one for the other on
    /// GLM's real expert bank measures relative 31.7.
    ClampedGated { limit: f32, activation: Activation },
}

impl MoeGateRule {
    /// The ONE translation from an architecture's declared policy +
    /// activation to a compute rule. Exhaustive on both enums so a new
    /// policy variant fails to compile here instead of silently taking
    /// the gated path.
    pub fn from_arch(
        policy: larql_models::ExpertGatePolicy,
        activation: larql_models::Activation,
    ) -> Self {
        match policy {
            larql_models::ExpertGatePolicy::Gated => Self::Gated(Activation::from(activation)),
            larql_models::ExpertGatePolicy::ClampedGlu { limit, alpha } => {
                Self::ClampedGlu { limit, alpha }
            }
            larql_models::ExpertGatePolicy::SituGlu { beta, linear_beta } => {
                Self::SituGlu { beta, linear_beta }
            }
            larql_models::ExpertGatePolicy::ClampedGated { limit } => Self::ClampedGated {
                limit,
                activation: Activation::from(activation),
            },
        }
    }

    /// Combine one `(gate, up)` pair. Inputs already carry their biases.
    #[inline]
    pub fn combine(self, g: f32, u: f32) -> f32 {
        #[inline]
        fn sigmoid(x: f32) -> f32 {
            1.0 / (1.0 + (-x).exp())
        }
        match self {
            Self::Gated(activation) => {
                if activation.gate_up_is_gelu_tanh() {
                    crate::cpu::ops::moe::math::gelu_tanh(g) * u
                } else {
                    crate::cpu::ops::moe::math::silu(g) * u
                }
            }
            Self::ClampedGlu { limit, alpha } => {
                let g = g.min(limit);
                let u = u.clamp(-limit, limit);
                (u + 1.0) * (g * sigmoid(g * alpha))
            }
            // The reference upcasts both branches to f32 before the
            // nonlinearity and rounds once at the end (L77-78, L82).
            // These inputs are already f32, so that contract is met by
            // construction here; the bf16 arm of `situ_oracle.json`
            // measures what the upcast is worth (rel-L2 3.2e-3 at K3's
            // parameters) so the statement is priced rather than assumed.
            Self::SituGlu { beta, linear_beta } => {
                let situ_a = beta * (g / beta).tanh() * sigmoid(g);
                let u = match linear_beta {
                    Some(cap) => cap * (u / cap).tanh(),
                    None => u,
                };
                situ_a * u
            }
            // The SAME clamp, and then the ordinary gated product — no
            // `alpha`, no `(u + 1)`.
            Self::ClampedGated { limit, activation } => {
                let g = g.min(limit);
                let u = u.clamp(-limit, limit);
                if activation.gate_up_is_gelu_tanh() {
                    crate::cpu::ops::moe::math::gelu_tanh(g) * u
                } else {
                    crate::cpu::ops::moe::math::silu(g) * u
                }
            }
        }
    }
}

/// One expert's MLP parameters beyond its weight bytes: the combine rule
/// and this expert's bias rows, if the checkpoint has them.
///
/// `gate_up_bias` is the expert's fused bias row `[2 * inter]` exactly as
/// the checkpoint stores it — **interleaved**, even entries gate and odd
/// entries up, the same convention as the weight rows
/// ([`larql_models::quant::mxfp4::FusedHalf`] owns it). Empty = no bias.
#[derive(Clone, Copy)]
pub struct ExpertMlp<'a> {
    pub rule: MoeGateRule,
    /// Fused interleaved gate/up bias row for this expert, `[2 * inter]`.
    pub gate_up_bias: &'a [f32],
    /// Down-projection bias for this expert, `[hidden]`.
    pub down_bias: &'a [f32],
}

impl ExpertMlp<'_> {
    /// Bias-free gated MLP — the shape every pre-bias call site had.
    pub const fn gated(activation: Activation) -> Self {
        Self {
            rule: MoeGateRule::Gated(activation),
            gate_up_bias: &[],
            down_bias: &[],
        }
    }

    /// This expert's gate bias for intermediate row `j` (0.0 when absent).
    #[inline]
    pub fn gate_bias(&self, j: usize) -> f32 {
        if self.gate_up_bias.is_empty() {
            0.0
        } else {
            self.gate_up_bias[larql_models::quant::mxfp4::FusedHalf::Gate.fused_row(j)]
        }
    }

    /// This expert's up bias for intermediate row `j` (0.0 when absent).
    #[inline]
    pub fn up_bias(&self, j: usize) -> f32 {
        if self.gate_up_bias.is_empty() {
            0.0
        } else {
            self.gate_up_bias[larql_models::quant::mxfp4::FusedHalf::Up.fused_row(j)]
        }
    }

    /// Add this expert's down bias into `out` (no-op when absent).
    #[inline]
    pub fn add_down_bias(&self, out: &mut [f32]) {
        if !self.down_bias.is_empty() {
            for (o, b) in out.iter_mut().zip(self.down_bias) {
                *o += b;
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MoeWeightLayout {
    pub down_padding: MoeDownPaddingPolicy,
}

impl MoeWeightLayout {
    pub const fn unpadded() -> Self {
        Self {
            down_padding: MoeDownPaddingPolicy::None,
        }
    }

    pub const fn quant_block_padded_down() -> Self {
        Self {
            down_padding: MoeDownPaddingPolicy::QuantBlock,
        }
    }

    pub fn down_cols(self, intermediate_size: usize, format: QuantFormat) -> usize {
        match self.down_padding {
            MoeDownPaddingPolicy::None => intermediate_size,
            MoeDownPaddingPolicy::QuantBlock => format
                .packed_block_layout()
                .map(|(block_elems, _)| intermediate_size.div_ceil(block_elems) * block_elems)
                .unwrap_or(intermediate_size),
        }
    }
}

impl Default for MoeWeightLayout {
    fn default() -> Self {
        Self::quant_block_padded_down()
    }
}

/// Where an expert bank's dequantisation scales physically live.
///
/// A format class does **not** answer this on its own: the same codec can
/// keep its scales inside the payload blocks or split them into a partner
/// stream, and MXFP4 checkpoints ship the split form while the k-quants
/// ship the inline one. The container-side statement of the same fact is
/// `larql_vindex::format::vindex3::ExpertScaleStreams`; this is its
/// consumer-side twin, resolved once at the boundary that can see both.
///
/// An enum rather than two optional vectors, so that "split-scale format,
/// scales missing" has no spelling.
#[derive(Clone, Debug)]
pub enum MoeExpertScales<'a> {
    /// Scales ride inside the payload blocks. No partner stream exists —
    /// not "an empty one".
    Inline,
    /// Per-expert e8m0 exponent streams, one per payload stream, index-aligned
    /// with `experts_gate_up` / `experts_down`.
    Paired {
        /// Exponents for the fused gate+up payload, `[num_experts]`.
        gate_up: Vec<&'a [u8]>,
        /// Exponents for the down payload, `[num_experts]`.
        down: Vec<&'a [u8]>,
    },
}

impl<'a> MoeExpertScales<'a> {
    /// Whether this bank carries a separate exponent stream.
    pub fn is_paired(&self) -> bool {
        matches!(self, Self::Paired { .. })
    }

    /// Expert `e`'s gate+up exponents, or `None` under [`Self::Inline`].
    ///
    /// # Panics
    /// If a paired table is too short for expert `e`. Falling back to
    /// `None` would put an inline-scale binding on a split-scale bank,
    /// which decodes every group against the wrong bytes rather than
    /// failing — the same reason [`MoeLayerWeights::expert_mlp`] panics.
    pub fn gate_up(&self, e: usize) -> Option<&'a [u8]> {
        match self {
            Self::Inline => None,
            Self::Paired { gate_up, .. } => Some(pick_stream(gate_up, e, "gate_up")),
        }
    }

    /// Expert `e`'s down exponents, or `None` under [`Self::Inline`].
    ///
    /// # Panics
    /// As [`Self::gate_up`].
    pub fn down(&self, e: usize) -> Option<&'a [u8]> {
        match self {
            Self::Inline => None,
            Self::Paired { down, .. } => Some(pick_stream(down, e, "down")),
        }
    }
}

/// Expert `e`'s stream out of a paired table, loudly.
fn pick_stream<'a>(table: &[&'a [u8]], e: usize, what: &str) -> &'a [u8] {
    table.get(e).copied().unwrap_or_else(|| {
        panic!(
            "{what} scale table has {} streams — too short for expert {e}",
            table.len()
        )
    })
}

/// How a fused `gate_up` expert region arranges its two projections' rows.
///
/// A third axis, independent of [`MoeWeightLayout`] (down-projection
/// padding) and [`MoeExpertScales`] (where scales live). Three axes, one
/// field each: describing a bank with a value borrowed from a neighbouring
/// axis is how a store ends up read under a convention it never claimed.
///
/// There is deliberately no `Unspecified` variant, though the container's
/// `RegionLayout` has one. A container written before that field existed
/// genuinely does not say; a consumer holding *this* type has already
/// resolved that question or refused to serve.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoeFusedRowLayout {
    /// `[all gate rows | all up rows]` — up starts one half-region in.
    /// Every larql-written k-quant expert store, because the extraction
    /// path de-interleaves before writing.
    ContiguousHalves,
    /// `gate = rows 0, 2, 4, …`, `up = rows 1, 3, 5, …`. What a GPT-OSS
    /// checkpoint ships, and what a verbatim MXFP4 passthrough preserves.
    ///
    /// Read as [`Self::ContiguousHalves`] it yields two 50/50 mixtures of
    /// the real gate and up rows, with matching summary statistics and
    /// coherent-looking output — `docs/k3-funnel.md` §4.7, which cost a
    /// served model once already.
    Interleaved,
}

impl MoeFusedRowLayout {
    /// `(first row, row stride)` walking one half's rows in fused-row
    /// space, where the fused region holds `2 * inter` rows.
    ///
    /// [`larql_models::quant::mxfp4::FusedHalf`] owns the interleaved
    /// convention; this reads the base off it rather than restating that
    /// gate is even and up is odd.
    pub const fn row_walk(
        self,
        half: larql_models::quant::mxfp4::FusedHalf,
        inter: usize,
    ) -> (usize, usize) {
        use larql_models::quant::mxfp4::{FusedHalf, FUSED_HALVES};
        match self {
            // Rows within a half are adjacent; the base is however many
            // rows precede this half.
            Self::ContiguousHalves => match half {
                FusedHalf::Gate => (0, 1),
                FusedHalf::Up => (inter, 1),
            },
            Self::Interleaved => (half.fused_row(0), FUSED_HALVES),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MoeRoutingPolicy {
    pub expert_input: MoeInputSource,
    pub router_input: MoeInputSource,
    pub router_norm: MoeRouterNormPolicy,
    pub selected_weight: MoeTopKWeightPolicy,
    pub expert_scale: MoeExpertScalePolicy,
    pub post_expert_norm: MoePostExpertNormPolicy,
}

impl MoeRoutingPolicy {
    /// Gemma 4 A4B hybrid-MoE behavior validated by local CPU/Metal parity:
    /// route and run experts from the pre-experts-normalized residual, apply
    /// router RMSNorm/scale, renormalize selected top-k probabilities, apply
    /// learned per-expert scales, then post-normalize the expert branch.
    pub const fn gemma4_hybrid() -> Self {
        Self {
            expert_input: MoeInputSource::PreExpertsNorm,
            router_input: MoeInputSource::PreExpertsNorm,
            router_norm: MoeRouterNormPolicy::LearnedOrParameterFree,
            selected_weight: MoeTopKWeightPolicy::RenormalizedSoftmax,
            expert_scale: MoeExpertScalePolicy::PerExpert,
            post_expert_norm: MoePostExpertNormPolicy::RmsNorm,
        }
    }

    /// Conventional sparse-MoE router behavior: route on the provided input,
    /// keep top-k probabilities as softmax weights, and do not apply Gemma 4
    /// branch-specific scales or post norms.
    pub const fn top_k_softmax() -> Self {
        Self {
            expert_input: MoeInputSource::Residual,
            router_input: MoeInputSource::Residual,
            router_norm: MoeRouterNormPolicy::None,
            selected_weight: MoeTopKWeightPolicy::RawSoftmax,
            expert_scale: MoeExpertScalePolicy::None,
            post_expert_norm: MoePostExpertNormPolicy::None,
        }
    }

    /// Select the top-k logits *first*, then softmax over just those, so the
    /// selected weights sum to 1. GPT-OSS.
    ///
    /// Distinct from [`Self::top_k_softmax`], whose weights sum to *less*
    /// than 1 by whatever mass the unselected experts hold. Getting the two
    /// confused rescales the entire expert branch — a large error that still
    /// produces coherent-looking output, which is why this rule now arrives
    /// as a typed [`larql_models::MoeRouterKind`] rather than a string that
    /// could miss its `match` arm (`docs/k3-funnel.md` §4.7.10).
    ///
    /// **Both inputs are the pre-experts-normed hidden.** In the reference
    /// (`modeling_gpt_oss.py`) the router lives *inside* the MLP block:
    /// `post_attention_layernorm` runs first and its output feeds the
    /// router and the experts alike. On the f32 tier that norm is applied
    /// by `run_ffn` before the backend is called, so `Residual` here
    /// looked correct there — but this policy's consumer is the quantised
    /// block path, which hands `cpu_moe_forward` the RAW residual and
    /// relies on the policy for the norm. Serving with `Residual` ran
    /// every router and every expert on the un-normed stream: structurally
    /// intact routing over garbage magnitudes, incoherent output, no
    /// crash. The norm weights resolve through
    /// `moe_pre_experts_norm_key`, which GPT-OSS maps to its
    /// `post_attention_layernorm`.
    pub const fn top_k_then_softmax() -> Self {
        Self {
            expert_input: MoeInputSource::PreExpertsNorm,
            router_input: MoeInputSource::PreExpertsNorm,
            selected_weight: MoeTopKWeightPolicy::RenormalizedSoftmax,
            router_norm: MoeRouterNormPolicy::None,
            expert_scale: MoeExpertScalePolicy::None,
            post_expert_norm: MoePostExpertNormPolicy::None,
        }
    }
}

// NOTE: deliberately no `Default` impl. A default here is a silent
// family choice made in the substrate crate — the failure class this
// project has hit repeatedly (`norm_topk_prob`, `rope_type: yarn`,
// `layer_types`). The policy must arrive explicitly, derived from the
// model's typed `larql_models::MoeRouterKind`.

pub struct MoeLayerWeights<'a> {
    /// Per-expert gate+up weight bytes (`experts_gate_up[e]` is expert `e`'s
    /// gate+up slice). Bytes are interpreted under `expert_data_format`.
    /// Built from `layers/{L}/{e}/gate_up` mmap ranges (per-layer Q4_K) or
    /// from `[num_experts, 2*inter, hidden]` strides (legacy BF16 monolith).
    pub experts_gate_up: Vec<&'a [u8]>,
    /// Per-expert down weight bytes (`experts_down[e]` is expert `e`'s down).
    pub experts_down: Vec<&'a [u8]>,
    /// Explicit routing behavior for this layer/model family.
    pub routing_policy: MoeRoutingPolicy,
    /// Explicit byte layout for expert tensors.
    pub weight_layout: MoeWeightLayout,
    /// Where this bank's dequantisation scales live. `Inline` for every
    /// k-quant store; `Paired` for a natively-stored MXFP4 bank, whose
    /// e8m0 exponents are a stream of their own.
    pub expert_scales: MoeExpertScales<'a>,
    /// How the fused gate+up rows are arranged **as stored**, which is a
    /// property of the store and not of the architecture: the same
    /// checkpoint yields `Interleaved` under a verbatim passthrough and
    /// `ContiguousHalves` under a canonicalising extraction.
    ///
    /// Governs the *weight* rows only. The bias table is interleaved
    /// regardless — it is carried exactly as the checkpoint ships it, and
    /// `expert_mlp` reads it through `FusedHalf`.
    pub fused_row_layout: MoeFusedRowLayout,
    /// Format of the per-expert byte slices. `Q4_K` = per-layer Q4_K files;
    /// `BF16` = legacy monolith. Both flow through the same per-expert tables.
    pub expert_data_format: QuantFormat,
    /// Router linear projection weight [num_experts, hidden_size].
    pub router_proj: &'a [f32],
    /// Router bias [num_experts], added to the logits **before** softmax /
    /// selection (it changes which experts win, not just their weights).
    /// Empty = no bias.
    pub router_bias: &'a [f32],
    /// Per-expert fused gate/up bias rows, flat `[num_experts, 2*inter]`,
    /// stored exactly as the checkpoint ships them — interleaved along the
    /// fused axis. Empty = the architecture has none (or a pre-bias vindex).
    pub experts_gate_up_bias: &'a [f32],
    /// Per-expert down bias rows, flat `[num_experts, hidden]`. Empty = none.
    pub experts_down_bias: &'a [f32],
    /// Router learned input-scale [hidden_size].
    pub router_scale: &'a [f32],
    /// Router per-expert output-scale [num_experts].
    pub router_per_expert_scale: &'a [f32],
    /// Router's own RMS-norm weight applied to the router input before projection.
    /// Empty slice → fall back to parameter-free RMSNorm (if the flag below
    /// is set) or to `pre_experts_norm`.
    pub router_norm: &'a [f32],
    /// Parameter-free router RMSNorm: apply `x / sqrt(mean(x²) + eps)` on
    /// the router input when `router_norm` is empty. HF Gemma 4 sets this
    /// true (`Gemma4RMSNorm(with_scale=False)` — no learned weight on disk).
    pub router_norm_parameter_free: bool,
    /// Scalar multiplier on the router input after the norm and `router_scale`.
    /// HF Gemma 4: `hidden_size^-0.5`. Use `1.0` to disable.
    pub router_input_scalar: f32,
    /// Pre-norm applied to the expert matmuls' input (not the router's). [hidden_size].
    pub pre_experts_norm: &'a [f32],
    /// Post-norm for dense FFN output (replaces plain post_ffn_norm). [hidden_size].
    pub post_ffn1_norm: &'a [f32],
    /// Post-norm for expert block output. [hidden_size].
    pub post_experts_norm: &'a [f32],
    /// Total number of routed experts.
    pub num_experts: usize,
    /// Experts activated per token (top-K).
    pub top_k: usize,
    /// Per-expert intermediate (hidden) dimension.
    pub intermediate_size: usize,
    /// How each expert's gate and up combine. Gemma 4 is `Gated(GeluTanh)`,
    /// Mixtral-likes `Gated(Silu)`, GPT-OSS `ClampedGlu`. Replaces the old
    /// bare `activation` field so the policy and the activation cannot be
    /// set inconsistently.
    pub gate_rule: MoeGateRule,
}

/// The expert-bank facts a routed container substitutes into a layer —
/// and nothing else. Attention, norms, router state and semantic
/// topology stay spine-owned; this struct owns only the bank's bytes
/// and the representation facts that make them readable.
///
/// Generic by design: "expert bank authority = routed container if
/// supplied, otherwise spine." No representation is named here — a
/// native MXFP4 bank is merely the first through the seam.
///
/// Everything is borrowed: applying an override moves references, never
/// bank bytes, so the container backing them must outlive the layer
/// views (the composing caller owns both).
pub struct ExpertBankOverride<'a> {
    pub experts_gate_up: Vec<&'a [u8]>,
    pub experts_down: Vec<&'a [u8]>,
    pub expert_scales: MoeExpertScales<'a>,
    pub fused_row_layout: MoeFusedRowLayout,
    pub expert_data_format: QuantFormat,
}

impl<'a> MoeLayerWeights<'a> {
    /// Replace this layer's expert bank with a routed container's —
    /// the representation authority swap, in one place. Refuses an
    /// override whose expert count disagrees with the layer topology
    /// (that is a semantic fact, and the spine owns it).
    pub fn apply_expert_bank_override(&mut self, ovr: ExpertBankOverride<'a>) {
        assert_eq!(
            ovr.experts_gate_up.len(),
            self.num_experts,
            "expert-bank override carries {} gate/up banks for a layer of \
             {} experts — the container cannot re-decide topology",
            ovr.experts_gate_up.len(),
            self.num_experts,
        );
        assert_eq!(ovr.experts_down.len(), self.num_experts);
        self.experts_gate_up = ovr.experts_gate_up;
        self.experts_down = ovr.experts_down;
        self.expert_scales = ovr.expert_scales;
        self.fused_row_layout = ovr.fused_row_layout;
        self.expert_data_format = ovr.expert_data_format;
    }

    pub fn inter_padded(&self) -> usize {
        self.weight_layout
            .down_cols(self.intermediate_size, self.expert_data_format)
    }

    /// The [`ExpertMlp`] view for expert `e`: the layer's combine rule plus
    /// this expert's slice of the flat bias tables.
    ///
    /// Panics if a non-empty bias table is too short for expert `e` — a
    /// table that does not describe this expert bank would otherwise bias
    /// every expert with expert 0's rows, a plausible-looking wrong forward.
    pub fn expert_mlp(&self, e: usize) -> ExpertMlp<'a> {
        let slice = |table: &'a [f32], stride: usize, what: &str| -> &'a [f32] {
            if table.is_empty() {
                return &[];
            }
            table.get(e * stride..(e + 1) * stride).unwrap_or_else(|| {
                panic!(
                    "{what} bias table has {} elements — too short for \
                         expert {e} at stride {stride}",
                    table.len()
                )
            })
        };
        ExpertMlp {
            rule: self.gate_rule,
            gate_up_bias: slice(
                self.experts_gate_up_bias,
                2 * self.intermediate_size,
                "gate_up",
            ),
            down_bias: slice(
                self.experts_down_bias,
                self.hidden_size_from_router(),
                "down",
            ),
        }
    }

    /// Hidden size, derived from the router projection — the struct does
    /// not carry it explicitly and every caller already has a router.
    fn hidden_size_from_router(&self) -> usize {
        self.router_proj
            .len()
            .checked_div(self.num_experts)
            .unwrap_or(0)
    }

    /// The gate/up matrices' STORED row width in elements, derived from
    /// expert 0's byte count.
    ///
    /// The writer pads each gate/up row to a super-block boundary so that
    /// per-row integer kernels can index the store (GPT-OSS's hidden 2880
    /// stores as 3072); block-multiple hidden sizes store unpadded and
    /// this returns `hidden` exactly. The byte count is the authority —
    /// deriving from `hidden` re-creates the assumption the padding
    /// removes. Falls back to `hidden` when the bytes don't divide
    /// (legacy monolith strides, synthetic fixtures).
    pub fn gate_up_cols(&self, hidden: usize) -> usize {
        let Some(&bytes) = self.experts_gate_up.first() else {
            return hidden;
        };
        stored_gate_up_cols(
            bytes.len(),
            self.intermediate_size,
            self.expert_data_format,
            hidden,
        )
    }
}

/// Free-function form of [`MoeLayerWeights::gate_up_cols`] for callers that
/// hold one expert's bytes without the struct (the single-expert entry
/// points the HTTP expert server drives). One derivation, two shapes.
pub fn stored_gate_up_cols(
    gate_up_bytes_len: usize,
    inter: usize,
    format: QuantFormat,
    hidden: usize,
) -> usize {
    if inter == 0 || !gate_up_bytes_len.is_multiple_of(2 * inter) {
        return hidden;
    }
    let row_bytes = gate_up_bytes_len / (2 * inter);
    let cols = match format.packed_block_layout() {
        Some((block_elems, block_bytes)) => {
            if !row_bytes.is_multiple_of(block_bytes) {
                return hidden;
            }
            row_bytes / block_bytes * block_elems
        }
        None => match format {
            QuantFormat::BF16 | QuantFormat::F16 => row_bytes / 2,
            QuantFormat::F32 => row_bytes / 4,
            _ => return hidden,
        },
    };
    if cols >= hidden {
        cols
    } else {
        hidden
    }
}

/// Hybrid MoE behavior for one layer. The expert tensors remain in
/// [`MoeLayerWeights`]; this view captures how the dense and expert branches
/// are combined.
#[derive(Clone, Copy)]
pub struct MoeSpec<'layer, 'data> {
    pub weights: Option<&'layer MoeLayerWeights<'data>>,
    pub combined_output_norm: bool,
    pub outer_post_norm: Option<&'data [f32]>,
}
