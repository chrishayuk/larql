//! Pipeline layer types — per-layer architecture parameters for the compute pipeline.
//!
//! These types carry all model-specific behavior per-layer:
//! norm type, activation, attention geometry, RoPE, FFN type, etc.
//! The compute backends read these fields per-layer — no hardcoded
//! model assumptions in the execution path.

/// Bytes per Q4_KF pre-baked super-block. Q4_KF keeps the 256-element
/// Q4_K block shape but expands packed scale/min metadata for faster decode.
pub const Q4_KF_BLOCK_BYTES: usize = 160;

/// Default RoPE base frequency (Llama, Gemma sliding-window layers).
pub const ROPE_BASE_DEFAULT: f32 = 10_000.0;

/// Long-context RoPE base frequency (Gemma global-attention layers).
pub const ROPE_BASE_GLOBAL: f32 = 1_000_000.0;

/// Default RMSNorm epsilon. Prevents division by zero in normalization.
pub const RMSNORM_EPSILON_DEFAULT: f32 = 1e-6;

/// Quantization format for a weight tensor.
/// Names match GGUF conventions (Q4_K, Q6_K, etc.).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
pub enum QuantFormat {
    Q4_0,  // 18 bytes per 32 values (one f16 scale)
    Q4_K,  // 144 bytes per 256 values (GGUF-canonical, Ollama-compatible)
    Q4_KF, // 160 bytes per 256 values (pre-baked half scales — fast decode)
    Q6_K,  // 210 bytes per 256 values (6-bit with sub-block scales)
    Q8_0,  // int8 values + separate f32 scales
    BF16,  // raw bfloat16 (2 bytes per value, no quantization scales)
    F16,   // raw float16  (2 bytes per value)
    F32,   // raw float32  (4 bytes per value)
}

impl QuantFormat {
    /// Packed block geometry as `(elements_per_block, bytes_per_block)`.
    ///
    /// This is the compute-side mirror of the GGML layout constants used by
    /// the quantizers. Callers that need byte offsets should ask the format
    /// instead of spelling `256 * 144` or `32 * 18` locally.
    pub fn packed_block_layout(self) -> Option<(usize, usize)> {
        use larql_models::quant::ggml;

        match self {
            Self::Q4_0 => Some((ggml::Q4_0_BLOCK_ELEMS, ggml::Q4_0_BLOCK_BYTES)),
            Self::Q4_K => Some((ggml::Q4_K_BLOCK_ELEMS, ggml::Q4_K_BLOCK_BYTES)),
            Self::Q4_KF => Some((ggml::Q4_K_BLOCK_ELEMS, Q4_KF_BLOCK_BYTES)),
            Self::Q6_K => Some((ggml::Q6_K_BLOCK_ELEMS, ggml::Q6_K_BLOCK_BYTES)),
            _ => None,
        }
    }

    /// Byte length for a packed row-major matrix with `rows * cols` values.
    ///
    /// Current interleaved FFN fallback stores each matrix contiguously, so
    /// this intentionally preserves the historical flat packing calculation.
    /// Manifest-aware paths should prefer recorded offsets and lengths.
    pub fn packed_matrix_bytes(self, rows: usize, cols: usize) -> Option<usize> {
        let elems = rows.checked_mul(cols)?;
        let (block_elems, block_bytes) = self.packed_block_layout()?;
        Some(elems.div_ceil(block_elems) * block_bytes)
    }

    /// Whether this format uses the GGUF k-quant 256-element super-block
    /// layout that flows through the dedicated Q4_K / Q4_KF / Q6_K matvec
    /// dispatchers (vs the legacy block-32 Q4_0 / Q8_0 path). Used to gate
    /// the "skip Q8 quantize" fast path in `residual_norm` and FFN routing.
    ///
    /// Adding a future k-quant format (e.g. Q5_K) extends this one method,
    /// not the ~10 OR-chains it currently replaces. Roadmap #7
    /// (`FormatRoute` enum) is the fuller version of this idea; this helper
    /// is the contained step that addresses the user-visible code-duplication
    /// cost without rippling through 49 files.
    pub fn is_kquant_family(self) -> bool {
        matches!(self, Self::Q4_K | Self::Q4_KF | Self::Q6_K)
    }

    /// Whether this format uses the llama.cpp-exact "Q4_KF" pre-baked
    /// half-scale fast path (`q4kf_proj` shader). Distinct from the
    /// canonical `Q4_K` GGUF layout used by Ollama extracts.
    pub fn is_q4kf(self) -> bool {
        matches!(self, Self::Q4_KF)
    }

    /// Whether this format uses the legacy block-32 Q8 dispatch path
    /// (`q4_matvec` / `q8_matvec` against pre-quantised Q8 input). The
    /// inverse of [`Self::is_kquant_family`] for the dense matvec dispatch
    /// (the float-input `BF16` / `F16` / `F32` branches don't run on
    /// these dispatchers, so `is_legacy_q8` covers exactly the rest).
    pub fn is_legacy_q8(self) -> bool {
        matches!(self, Self::Q4_0 | Self::Q8_0)
    }
}

/// A quantized weight matrix — raw bytes with format tag.
#[derive(Clone, Copy)]
pub struct QuantWeight<'a> {
    pub data: &'a [u8],
    pub scales: Option<&'a [f32]>, // only for Q8_0 (separate scale array)
    pub format: QuantFormat,
}

/// Norm type for layer normalization.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NormType {
    /// RMSNorm — Llama, Gemma, Qwen, most modern models.
    RmsNorm,
    /// Standard LayerNorm (mean-subtraction + variance normalization) — StarCoder2, GPT-2.
    LayerNorm,
}

/// FFN type: gated (gate+up→GEGLU→down) vs standard (up→activation→down).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FfnType {
    /// Gated: SiLU(x @ gate.T) * (x @ up.T) @ down.T — Llama, Gemma, Mistral.
    Gated,
    /// Standard: activation(x @ up.T) @ down.T — StarCoder2, GPT-2.
    Standard,
}

/// Activation function for FFN.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Activation {
    /// SiLU / Swish — Llama, Mistral, Qwen.
    Silu,
    /// GELU with tanh approximation — Gemma, StarCoder2.
    GeluTanh,
    /// Exact GELU (erf-based) — used in some GPT-2 variants.
    GeluExact,
    /// ReLU — legacy models (GPT-J, etc.).
    ReLU,
}

/// Positional encoding strategy for attention.
///
/// Most transformer models use RoPE. Non-RoPE variants (ALiBi, absolute,
/// none) are tracked here so future backends can guard on the type rather
/// than assuming RoPE is always present.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PositionEncodingType {
    /// Rotary position embedding. `base` and `rotary_dim` live in
    /// [`FullPipelineLayer`] so the encoding is still fully per-layer.
    RoPE,
    /// Attention with Linear Biases (no learned embeddings).
    ALiBi,
    /// Fixed absolute sinusoidal or learned embeddings (injected at
    /// the embedding layer, not per-attention-head).
    Absolute,
    /// No position encoding (e.g. some cross-attention blocks).
    None,
}

/// Hybrid MoE (Mixture-of-Experts) weights for one layer.
///
/// Gemma 4 26B A4B runs a dense MLP and an expert block in parallel per layer,
/// summing their outputs. This struct carries the expert-block tensors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoeInputSource {
    /// Use the residual stream exactly as passed into the MoE block.
    Residual,
    /// Use `rms_norm(residual, pre_experts_norm)` as the stage input.
    PreExpertsNorm,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoeRouterNormPolicy {
    /// Do not apply a router-specific norm.
    None,
    /// Apply `router_norm` when present; otherwise leave the router input unchanged.
    Learned,
    /// Apply parameter-free RMSNorm regardless of learned router weights.
    ParameterFree,
    /// Prefer learned `router_norm`; otherwise use parameter-free RMSNorm when enabled.
    LearnedOrParameterFree,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoeTopKWeightPolicy {
    /// Keep selected weights as the original softmax probabilities.
    RawSoftmax,
    /// Renormalize selected top-k weights so they sum to 1 before scaling.
    RenormalizedSoftmax,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoeExpertScalePolicy {
    /// Ignore `router_per_expert_scale`.
    None,
    /// Multiply selected weights by `router_per_expert_scale[expert_id]` when present.
    PerExpert,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoePostExpertNormPolicy {
    /// Return the weighted expert sum directly.
    None,
    /// Apply `post_experts_norm` via RMSNorm when the tensor is present.
    RmsNorm,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoeDownPaddingPolicy {
    /// Expert down matrices use `intermediate_size` columns.
    None,
    /// Expert down matrices are padded to the quant format's block width.
    QuantBlock,
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
}

impl Default for MoeRoutingPolicy {
    fn default() -> Self {
        Self::gemma4_hybrid()
    }
}

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
    /// Format of the per-expert byte slices. `Q4_K` = per-layer Q4_K files;
    /// `BF16` = legacy monolith. Both flow through the same per-expert tables.
    pub expert_data_format: QuantFormat,
    /// Router linear projection weight [num_experts, hidden_size].
    pub router_proj: &'a [f32],
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
    /// Activation function for expert MLPs. Gemma 4 uses GeluTanh; Mixtral/others use Silu.
    pub activation: Activation,
}

impl MoeLayerWeights<'_> {
    pub fn inter_padded(&self) -> usize {
        self.weight_layout
            .down_cols(self.intermediate_size, self.expert_data_format)
    }
}

/// Attention projection weights for one layer.
#[derive(Clone, Copy)]
pub struct AttentionWeights<'a> {
    pub wq: QuantWeight<'a>,
    pub wk: QuantWeight<'a>,
    pub wv: QuantWeight<'a>,
    pub wo: QuantWeight<'a>,
}

/// Dense FFN projection weights for one layer.
#[derive(Clone, Copy)]
pub struct FfnWeights<'a> {
    /// Gate projection. Used only when [`FfnSpec::ffn_type`] is [`FfnType::Gated`].
    pub gate: QuantWeight<'a>,
    pub up: QuantWeight<'a>,
    pub down: QuantWeight<'a>,
}

/// Grouped weight view for one layer.
#[derive(Clone, Copy)]
pub struct LayerWeights<'a> {
    pub attention: AttentionWeights<'a>,
    pub ffn: FfnWeights<'a>,
}

/// Norm weights, biases, and scalar norm behavior for one layer.
#[derive(Clone, Copy)]
pub struct LayerNorms<'a> {
    pub input_norm: &'a [f32],
    pub post_attn_norm: &'a [f32],
    pub pre_ffn_norm: Option<&'a [f32]>,
    pub post_ffn_norm: Option<&'a [f32]>,
    pub input_norm_bias: Option<&'a [f32]>,
    pub post_attn_norm_bias: Option<&'a [f32]>,
    pub norm_offset: f32,
    pub qk_norm_offset: f32,
    pub eps: f32,
    pub has_post_norms: bool,
    pub norm_type: NormType,
}

/// Per-layer attention geometry and position-encoding behavior.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AttentionSpec {
    pub attn_scale: f32,
    pub head_dim: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub rope_base: f32,
    pub rotary_dim: usize,
    pub sliding_window: usize,
    pub has_v_norm: bool,
    pub q_norm_enabled: bool,
    pub k_norm_enabled: bool,
    pub position_encoding: PositionEncodingType,
}

/// Dense FFN architecture behavior for one layer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FfnSpec {
    pub ffn_type: FfnType,
    pub activation: Activation,
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

/// Remote FFN dispatch behavior for one layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RemoteFfnSpec {
    pub is_remote: bool,
}

/// Per-layer quantized weights for the full pipeline.
///
/// Carries all architecture-specific behavior per-layer — no model
/// type strings or hardcoded constants in the compute path.
/// Supports Q4_K/Q6_K (Ollama strategy) or Q8_0 (higher precision fallback).
pub struct FullPipelineLayer<'a> {
    // ── Attention weights ──
    pub wq: QuantWeight<'a>,
    pub wk: QuantWeight<'a>,
    pub wv: QuantWeight<'a>,
    pub wo: QuantWeight<'a>,

    // ── FFN weights ──
    /// Gate projection (only used when ffn_type == Gated).
    pub gate: QuantWeight<'a>,
    pub up: QuantWeight<'a>,
    pub down: QuantWeight<'a>,

    // ── Norm weights (f32 vectors, hidden_size elements) ──
    pub input_norm: &'a [f32],
    pub post_attn_norm: &'a [f32],
    pub pre_ffn_norm: Option<&'a [f32]>,
    pub post_ffn_norm: Option<&'a [f32]>,
    /// Norm bias (only for LayerNorm). None for RMSNorm.
    pub input_norm_bias: Option<&'a [f32]>,
    pub post_attn_norm_bias: Option<&'a [f32]>,

    // ── Per-layer architecture parameters ──
    /// Norm weight offset: 0.0 (Llama, Gemma 4), 1.0 (Gemma 2/3).
    pub norm_offset: f32,
    /// QK norm weight offset: 0.0 (Llama, Gemma 4), 1.0 (Gemma 2/3).
    pub qk_norm_offset: f32,
    /// RMSNorm epsilon. Default: 1e-6.
    pub eps: f32,
    /// Whether this model uses post-norms (4 norms per layer: Gemma 2/3/4).
    pub has_post_norms: bool,
    /// Norm type: RMSNorm (default) or LayerNorm (StarCoder2).
    pub norm_type: NormType,
    /// FFN type: Gated (default) or Standard (StarCoder2).
    pub ffn_type: FfnType,
    /// Activation function for the FFN.
    pub activation: Activation,
    /// Attention scale for this layer. Default: 1/sqrt(head_dim).
    /// Gemma 4 (with QK-norm): 1.0.
    pub attn_scale: f32,
    /// Head dimension for this layer. Gemma 4: 256 (sliding) or 512 (global).
    pub head_dim: usize,
    /// Number of Q heads for this layer.
    pub num_q_heads: usize,
    /// Number of KV heads for this layer.
    pub num_kv_heads: usize,
    /// RoPE base frequency for this layer. Gemma 3/4: 10k (sliding) or 1M (global).
    pub rope_base: f32,
    /// Dimensions to apply RoPE to. 0 = full head_dim. Gemma 4 global: head_dim * 0.25.
    pub rotary_dim: usize,
    /// Sliding window size. 0 = full attention (no window).
    pub sliding_window: usize,
    /// Whether to apply parameter-free V-norm (Gemma 4).
    pub has_v_norm: bool,
    /// Per-layer scalar multiplier. 0.0 = disabled (no scaling). Gemma 4: learned scalar.
    pub layer_scalar: f32,
    /// QK-norm weight for Q heads (Gemma 3 / Gemma 4). Length = head_dim.
    /// Applied per-head as RMS-norm before RoPE. `None` means skip QK-norm.
    pub q_norm_weight: Option<&'a [f32]>,
    /// QK-norm weight for K heads. Same shape as `q_norm_weight`.
    pub k_norm_weight: Option<&'a [f32]>,
    /// FFN bias on up projection (StarCoder2). None = no bias.
    pub ffn_up_bias: Option<&'a [f32]>,
    /// FFN bias on down projection (StarCoder2). None = no bias.
    pub ffn_down_bias: Option<&'a [f32]>,

    /// Hybrid MoE block (Gemma 4 26B A4B: dense MLP + expert block, outputs summed).
    /// None for all dense models.
    pub moe: Option<MoeLayerWeights<'a>>,

    /// When true, the local FFN (gate/up/down) is skipped and the FFN
    /// contribution is provided externally via `moe_fn`. Used by
    /// `generate_with_remote_ffn` where ALL FFN goes to a remote server.
    /// Default: false.
    pub ffn_is_remote: bool,

    /// When true, a final RMS norm is applied to the combined (dense + expert)
    /// output before the residual add. Gemma 4 26B A4B: true. Other models:
    /// false (use `layer_scalar` instead).
    pub moe_combined_output_norm: bool,

    /// Outer post-FFN norm weight applied to `(h1 + h2)` before the residual
    /// add. When present and `moe_combined_output_norm` is true, this weight
    /// is used instead of `post_ffn_norm` for the combined norm.
    /// HF Gemma 4: `layers.N.post_feedforward_layernorm.weight` (un-suffixed,
    /// distinct from the `_1` dense-branch norm stored in `post_ffn_norm`).
    /// `None` → fall back to `post_ffn_norm` (legacy behavior).
    pub moe_outer_post_norm: Option<&'a [f32]>,

    // ── Per-Layer Embeddings (Gemma 4 E2B) ──
    /// Per-layer input gate matrix `[ple_dim, hidden]` row-major, f32.
    /// `None` for non-PLE archs.
    pub ple_input_gate: Option<&'a [f32]>,
    /// Per-layer output projection matrix `[hidden, ple_dim]` row-major, f32.
    /// `None` for non-PLE archs.
    pub ple_projection: Option<&'a [f32]>,
    /// Post-PLE RMSNorm weight `[hidden]`, f32. `None` for non-PLE archs.
    pub ple_post_norm: Option<&'a [f32]>,

    /// KV-cache sharing source: when `Some(src)`, this layer reuses K/V from
    /// layer `src`'s cache instead of computing its own. Gemma 4 E2B's last
    /// 20 of 35 layers point to the last non-shared sliding (or global) layer
    /// of the same attention type. `None` for non-shared layers — the
    /// production case for every model except E2B and similar future
    /// KV-shared archs.
    pub kv_shared_source: Option<usize>,

    /// Granite-style residual-stream multiplier. The residual add inside
    /// each transformer block becomes `h += residual_multiplier * x` where
    /// `x` is the attention output (post-W_O) or FFN output (post-down).
    /// Granite 4.1: 0.22 on 3B/8B, 0.175 on 30B (μP scaling). Every other
    /// model: 1.0 (identity — keep the residual add unchanged).
    ///
    /// Plumbed through `encode_post_attn` / `encode_post_ffn` →
    /// `encode_residual_add` → the `residual_add` Metal shader's `b_scale`
    /// binding. The shader treats 1.0 as a no-op so non-Granite paths are
    /// bit-identical to the pre-Granite implementation.
    pub residual_multiplier: f32,
}

impl<'a> FullPipelineLayer<'a> {
    /// Group the layer's quantized attention and FFN weights.
    pub fn weights(&self) -> LayerWeights<'a> {
        LayerWeights {
            attention: AttentionWeights {
                wq: self.wq,
                wk: self.wk,
                wv: self.wv,
                wo: self.wo,
            },
            ffn: FfnWeights {
                gate: self.gate,
                up: self.up,
                down: self.down,
            },
        }
    }

    /// Group the layer's norm weights, biases, and norm scalar behavior.
    pub fn norms(&self) -> LayerNorms<'a> {
        LayerNorms {
            input_norm: self.input_norm,
            post_attn_norm: self.post_attn_norm,
            pre_ffn_norm: self.pre_ffn_norm,
            post_ffn_norm: self.post_ffn_norm,
            input_norm_bias: self.input_norm_bias,
            post_attn_norm_bias: self.post_attn_norm_bias,
            norm_offset: self.norm_offset,
            qk_norm_offset: self.qk_norm_offset,
            eps: self.eps,
            has_post_norms: self.has_post_norms,
            norm_type: self.norm_type,
        }
    }

    /// Return the layer's attention shape, RoPE, and attention-normalization behavior.
    pub fn attention_spec(&self) -> AttentionSpec {
        AttentionSpec {
            attn_scale: self.attn_scale,
            head_dim: self.head_dim,
            num_q_heads: self.num_q_heads,
            num_kv_heads: self.num_kv_heads,
            rope_base: self.rope_base,
            rotary_dim: self.rotary_dim,
            sliding_window: self.sliding_window,
            has_v_norm: self.has_v_norm,
            q_norm_enabled: self.q_norm_weight.is_some(),
            k_norm_enabled: self.k_norm_weight.is_some(),
            position_encoding: PositionEncodingType::RoPE,
        }
    }

    /// Return the layer's dense FFN architecture behavior.
    pub fn ffn_spec(&self) -> FfnSpec {
        FfnSpec {
            ffn_type: self.ffn_type,
            activation: self.activation,
        }
    }

    /// Return the layer's hybrid-MoE behavior.
    pub fn moe_spec(&self) -> MoeSpec<'_, 'a> {
        MoeSpec {
            weights: self.moe.as_ref(),
            combined_output_norm: self.moe_combined_output_norm,
            outer_post_norm: self.moe_outer_post_norm,
        }
    }

    /// Return the layer's remote-FFN dispatch behavior.
    pub fn remote_ffn_spec(&self) -> RemoteFfnSpec {
        RemoteFfnSpec {
            is_remote: self.ffn_is_remote,
        }
    }

    /// Whether this layer uses gated FFN (gate + up → GEGLU → down).
    pub fn is_gated(&self) -> bool {
        self.ffn_type == FfnType::Gated
    }

    /// Whether this layer has a hybrid MoE block alongside the dense FFN.
    /// When true, the forward pass runs both branches and sums their outputs.
    pub fn is_hybrid_moe(&self) -> bool {
        self.moe.is_some()
    }

    /// Per-Layer Embeddings spec for this layer, if active. Returns `None`
    /// unless all three required weights are present (gate, projection, post-norm).
    pub fn ple_spec(&self) -> Option<PleSpec<'a>> {
        Some(PleSpec {
            input_gate: self.ple_input_gate?,
            projection: self.ple_projection?,
            post_norm: self.ple_post_norm?,
        })
    }
}

/// Per-Layer Embeddings (Gemma 4 E2B) per-layer weights view.
///
/// Returned by [`FullPipelineLayer::ple_spec`] only when all three required
/// weights are present.  All slices are row-major f32.
#[derive(Clone, Copy)]
pub struct PleSpec<'a> {
    /// `[ple_dim, hidden]` — projects hidden state down to `ple_dim` for gating.
    pub input_gate: &'a [f32],
    /// `[hidden, ple_dim]` — projects gated PLE signal back up to hidden state.
    pub projection: &'a [f32],
    /// `[hidden]` — RMSNorm applied to the projection output before residual add.
    pub post_norm: &'a [f32],
}

// ── Defaults ──
//
// `Default` for the leaf types (`QuantWeight`, `FullPipelineLayer`, …) lets
// tests construct minimal instances with `..Default::default()` instead of
// spelling out all 30+ fields. The roadmap's "FullPipelineLayer 63 pub
// fields" cleanup tracks a fuller restructure into LayerWeights /
// LayerNorms / LayerArchParams sub-structs; that's deferred until the
// MoE refactor settles. In the meantime `Default` collapses the test
// boilerplate without rippling through 30 caller files.

impl Default for QuantWeight<'_> {
    fn default() -> Self {
        Self {
            data: &[],
            scales: None,
            format: QuantFormat::Q4_0,
        }
    }
}

impl Default for FullPipelineLayer<'_> {
    fn default() -> Self {
        let qw = QuantWeight::default();
        Self {
            wq: qw,
            wk: qw,
            wv: qw,
            wo: qw,
            gate: qw,
            up: qw,
            down: qw,
            input_norm: &[],
            post_attn_norm: &[],
            pre_ffn_norm: None,
            post_ffn_norm: None,
            input_norm_bias: None,
            post_attn_norm_bias: None,
            norm_offset: 0.0,
            qk_norm_offset: 0.0,
            eps: RMSNORM_EPSILON_DEFAULT,
            has_post_norms: false,
            norm_type: NormType::RmsNorm,
            ffn_type: FfnType::Gated,
            activation: Activation::Silu,
            attn_scale: 1.0,
            head_dim: 0,
            num_q_heads: 0,
            num_kv_heads: 0,
            rope_base: ROPE_BASE_DEFAULT,
            rotary_dim: 0,
            sliding_window: 0,
            has_v_norm: false,
            layer_scalar: 0.0,
            q_norm_weight: None,
            k_norm_weight: None,
            ffn_up_bias: None,
            ffn_down_bias: None,
            moe: None,
            moe_combined_output_norm: false,
            moe_outer_post_norm: None,
            ffn_is_remote: false,
            ple_input_gate: None,
            ple_projection: None,
            ple_post_norm: None,
            kv_shared_source: None,
            residual_multiplier: 1.0,
        }
    }
}

// ── Backward compatibility: convert old-style bool to new enums ──

impl From<bool> for Activation {
    /// `true` = GeluTanh (Gemma), `false` = Silu (Llama).
    fn from(use_gelu_tanh: bool) -> Self {
        if use_gelu_tanh {
            Activation::GeluTanh
        } else {
            Activation::Silu
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_qw(data: &[u8]) -> QuantWeight<'_> {
        QuantWeight {
            data,
            scales: None,
            format: QuantFormat::Q4_0,
        }
    }

    fn minimal_layer<'a>(
        data: &'a [u8],
        norms: &'a [f32],
        ffn_type: FfnType,
        moe: Option<MoeLayerWeights<'a>>,
    ) -> FullPipelineLayer<'a> {
        let qw = minimal_qw(data);
        // Spread `..Default::default()` collapses the 30-field boilerplate
        // to just the fields this test actually exercises. Pin this pattern
        // so any future test that wants a minimal layer copies it.
        FullPipelineLayer {
            wq: qw,
            wk: qw,
            wv: qw,
            wo: qw,
            gate: qw,
            up: qw,
            down: qw,
            input_norm: norms,
            post_attn_norm: norms,
            ffn_type,
            attn_scale: 0.5,
            head_dim: 4,
            num_q_heads: 1,
            num_kv_heads: 1,
            moe,
            ..FullPipelineLayer::default()
        }
    }

    #[test]
    fn activation_from_bool() {
        assert_eq!(Activation::from(true), Activation::GeluTanh);
        assert_eq!(Activation::from(false), Activation::Silu);
    }

    #[test]
    fn is_gated_matches_ffn_type() {
        let norms = [1.0f32; 4];
        let gated = minimal_layer(&[], &norms, FfnType::Gated, None);
        let standard = minimal_layer(&[], &norms, FfnType::Standard, None);
        assert!(gated.is_gated());
        assert!(!standard.is_gated());
    }

    #[test]
    fn is_hybrid_moe_reflects_option() {
        let norms = [1.0f32; 4];
        let no_moe = minimal_layer(&[], &norms, FfnType::Gated, None);
        assert!(!no_moe.is_hybrid_moe());

        let moe = MoeLayerWeights {
            experts_gate_up: Vec::new(),
            experts_down: Vec::new(),
            routing_policy: MoeRoutingPolicy::default(),
            weight_layout: MoeWeightLayout::default(),
            router_proj: &[],
            router_scale: &[],
            router_per_expert_scale: &[],
            router_norm: &[],
            router_norm_parameter_free: false,
            router_input_scalar: 1.0,
            pre_experts_norm: &[],
            post_ffn1_norm: &[],
            post_experts_norm: &[],
            num_experts: 2,
            top_k: 1,
            intermediate_size: 4,
            activation: Activation::Silu,
            expert_data_format: QuantFormat::BF16,
        };
        let with_moe = minimal_layer(&[], &norms, FfnType::Gated, Some(moe));
        assert!(with_moe.is_hybrid_moe());
    }

    #[test]
    fn quant_format_equality() {
        assert_eq!(QuantFormat::Q4_K, QuantFormat::Q4_K);
        assert_ne!(QuantFormat::Q4_K, QuantFormat::Q6_K);
        assert_ne!(QuantFormat::Q4_0, QuantFormat::Q4_KF);
    }

    /// Pin the k-quant family taxonomy. Adding a new format requires
    /// updating exactly one of these classifiers.
    #[test]
    fn quant_format_classifiers() {
        // k-quant family (256-element super-blocks)
        assert!(QuantFormat::Q4_K.is_kquant_family());
        assert!(QuantFormat::Q4_KF.is_kquant_family());
        assert!(QuantFormat::Q6_K.is_kquant_family());
        // Legacy block-32 Q8 path
        assert!(QuantFormat::Q4_0.is_legacy_q8());
        assert!(QuantFormat::Q8_0.is_legacy_q8());
        // Float-input formats are neither
        for fmt in [QuantFormat::BF16, QuantFormat::F16, QuantFormat::F32] {
            assert!(!fmt.is_kquant_family());
            assert!(!fmt.is_legacy_q8());
        }
        // Q4_KF is a subset of the k-quant family
        assert!(QuantFormat::Q4_KF.is_q4kf());
        assert!(!QuantFormat::Q4_K.is_q4kf());
        assert!(!QuantFormat::Q6_K.is_q4kf());
    }

    #[test]
    fn quant_format_reports_packed_matrix_bytes() {
        assert_eq!(QuantFormat::Q4_0.packed_matrix_bytes(2, 32), Some(36));
        assert_eq!(QuantFormat::Q4_K.packed_matrix_bytes(2, 256), Some(288));
        assert_eq!(QuantFormat::Q4_KF.packed_matrix_bytes(2, 256), Some(320));
        assert_eq!(QuantFormat::Q6_K.packed_matrix_bytes(2, 256), Some(420));
        assert_eq!(QuantFormat::F16.packed_matrix_bytes(2, 256), None);
    }

    /// `..Default::default()` must work with stack-local borrowed data —
    /// the compiler reborrows the `'static` defaults at the caller's
    /// shorter lifetime. Pin the pattern.
    #[test]
    fn default_layer_accepts_local_borrows_via_spread() {
        let data: Vec<u8> = vec![0, 1, 2];
        let norms: Vec<f32> = vec![1.0; 4];

        let layer = FullPipelineLayer {
            input_norm: &norms,
            post_attn_norm: &norms,
            wq: QuantWeight {
                data: &data,
                ..Default::default()
            },
            head_dim: 4,
            num_q_heads: 1,
            num_kv_heads: 1,
            ..Default::default()
        };

        // Defaulted fields carry through.
        assert_eq!(layer.eps, RMSNORM_EPSILON_DEFAULT);
        assert_eq!(layer.norm_type, NormType::RmsNorm);
        assert_eq!(layer.ffn_type, FfnType::Gated);
        assert_eq!(layer.activation, Activation::Silu);
        assert!(!layer.has_v_norm);
        assert!(layer.moe.is_none());

        // Explicit fields are honoured.
        assert_eq!(layer.input_norm.len(), 4);
        assert_eq!(layer.wq.data.len(), 3);
        assert_eq!(layer.head_dim, 4);
    }

    #[test]
    fn layer_spec_views_preserve_flat_field_values() {
        let data: Vec<u8> = vec![0, 1, 2, 3];
        let norms: Vec<f32> = vec![1.0; 8];
        let q_norm: Vec<f32> = vec![0.5; 4];

        let qw = QuantWeight {
            data: &data,
            scales: None,
            format: QuantFormat::Q4_K,
        };
        let layer = FullPipelineLayer {
            wq: qw,
            wk: qw,
            wv: qw,
            wo: qw,
            gate: qw,
            up: qw,
            down: qw,
            input_norm: &norms,
            post_attn_norm: &norms,
            norm_offset: 1.0,
            qk_norm_offset: 0.25,
            eps: 1e-5,
            has_post_norms: true,
            activation: Activation::GeluTanh,
            attn_scale: 0.125,
            head_dim: 4,
            num_q_heads: 2,
            num_kv_heads: 1,
            rope_base: ROPE_BASE_GLOBAL,
            rotary_dim: 2,
            sliding_window: 32,
            has_v_norm: true,
            q_norm_weight: Some(&q_norm),
            ffn_is_remote: true,
            moe_combined_output_norm: true,
            moe_outer_post_norm: Some(&norms),
            ..FullPipelineLayer::default()
        };

        let weights = layer.weights();
        assert_eq!(weights.attention.wq.format, QuantFormat::Q4_K);
        assert_eq!(weights.ffn.down.data.len(), data.len());

        let norms_view = layer.norms();
        assert_eq!(norms_view.input_norm.len(), norms.len());
        assert_eq!(norms_view.norm_offset, 1.0);
        assert_eq!(norms_view.qk_norm_offset, 0.25);
        assert_eq!(norms_view.eps, 1e-5);
        assert!(norms_view.has_post_norms);

        let attn = layer.attention_spec();
        assert_eq!(attn.head_dim, 4);
        assert_eq!(attn.num_q_heads, 2);
        assert_eq!(attn.num_kv_heads, 1);
        assert_eq!(attn.rope_base, ROPE_BASE_GLOBAL);
        assert_eq!(attn.rotary_dim, 2);
        assert_eq!(attn.sliding_window, 32);
        assert!(attn.has_v_norm);
        assert!(attn.q_norm_enabled);
        assert!(!attn.k_norm_enabled);

        assert_eq!(layer.ffn_spec().activation, Activation::GeluTanh);
        assert!(layer.remote_ffn_spec().is_remote);
        assert!(layer.moe_spec().combined_output_norm);
        assert_eq!(layer.moe_spec().outer_post_norm.unwrap().len(), norms.len());
    }
}
