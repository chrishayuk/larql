//! Model architecture trait and shared types.
//!
//! Every model architecture implements `ModelArchitecture`. This trait
//! describes *what the model is* — tensor key patterns, norm behavior,
//! activation functions, scaling — without any compute dependencies.

use crate::validation::ConfigValidationResult;

/// Normalization type used by the model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NormType {
    /// RMSNorm (Gemma, Llama)
    RmsNorm,
    /// Standard LayerNorm (GPT-2, BERT)
    LayerNorm,
}

/// Activation function used in the FFN.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Activation {
    /// SiLU / Swish (Gemma, Llama)
    Silu,
    /// GELU (GPT-2, BERT)
    Gelu,
    /// GELU with tanh approximation
    GeluTanh,
    /// ReLU
    Relu,
}

/// Whether the FFN uses a gated architecture.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FfnType {
    /// Gated: SiLU(x @ gate.T) * (x @ up.T) @ down.T (Gemma, Llama)
    Gated,
    /// Standard: activation(x @ up.T) @ down.T (GPT-2)
    Standard,
}

/// How expert weights are stored in a MoE model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExpertFormat {
    /// Per-expert separate tensors (Mixtral, DeepSeek).
    /// Keys: `experts.{id}.w1.weight`, `experts.{id}.w2.weight`, etc.
    PerExpert,
    /// Packed MXFP4 (GPT-OSS/OpenAI).
    /// All experts fused into one tensor with block quantization.
    /// Keys: `experts.gate_up_proj_blocks`, `experts.gate_up_proj_scales`, etc.
    PackedMxfp4,
    /// Packed BF16/F16 stacked tensors (Gemma 4 26B A4B).
    /// All experts fused into one tensor per projection, no quantization scales.
    /// Keys: `experts.gate_up_proj` [num_experts, 2*moe_intermediate, hidden],
    ///        `experts.down_proj`   [num_experts, hidden, moe_intermediate].
    PackedBF16,
}

/// RoPE scaling configuration (YaRN, linear, dynamic).
#[derive(Debug, Clone)]
pub struct RopeScaling {
    pub scaling_type: String,
    pub factor: f64,
}

/// Model dimensions and architecture parameters, parsed from config.json.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub model_type: String,
    pub num_layers: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub head_dim: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub vocab_size: Option<usize>,
    pub rope_base: f64,
    /// RoPE base for local/sliding window layers (Gemma3: 10,000).
    pub rope_local_base: Option<f64>,
    pub sliding_window: Option<usize>,
    // MoE fields
    pub num_experts: Option<usize>,
    pub num_experts_per_token: Option<usize>,
    pub num_shared_experts: Option<usize>,
    /// Gemma 4 A4B: enables hybrid dense-MLP + MoE-experts block per layer.
    pub enable_moe_block: bool,
    /// Gemma 4 A4B: experts activated per token (stored as `top_k_experts` in config.json).
    pub top_k_experts: Option<usize>,
    /// Gemma 4 A4B: intermediate (hidden) dimension of each expert's FFN.
    pub moe_intermediate_size: Option<usize>,
    // MLA fields
    pub kv_lora_rank: Option<usize>,
    pub q_lora_rank: Option<usize>,
    // RoPE scaling
    pub rope_scaling: Option<RopeScaling>,
    // Softcapping (Gemma2)
    pub attn_logit_softcapping: Option<f64>,
    pub final_logit_softcapping: Option<f64>,
    /// Override attention scale denominator (Gemma: query_pre_attn_scalar).
    pub query_pre_attn_scalar: Option<f64>,
    // Granite-style scaling multipliers
    pub embedding_multiplier: Option<f64>,
    pub residual_multiplier: Option<f64>,
    pub attention_multiplier: Option<f64>,
    pub logits_scaling: Option<f64>,
    // Per-layer attention geometry (Gemma 4 style: different head_dim / KV heads
    // for sliding vs global attention layers).
    /// Head dimension for global (full) attention layers. If None, all layers use head_dim.
    pub global_head_dim: Option<usize>,
    /// Number of KV heads for global attention layers. If None, all layers use num_kv_heads.
    pub num_global_kv_heads: Option<usize>,
    /// Fraction of head_dim dimensions to apply RoPE to (0.0–1.0). If None, full rotation.
    pub partial_rotary_factor: Option<f64>,
    /// Sliding window pattern: every Nth layer is full attention.
    /// E.g., 6 means layers 5, 11, 17, ... are full attention.
    pub sliding_window_pattern: Option<usize>,
    /// Explicit per-layer type array (e.g., ["sliding_attention", "full_attention", ...]).
    /// When present, overrides sliding_window_pattern.
    pub layer_types: Option<Vec<String>>,
    /// Whether value projection shares key projection (K=V) on some layers.
    pub attention_k_eq_v: bool,
    /// Per-layer embedding dimension (PLE). If > 0, each layer adds a gated
    /// per-layer embedding lookup to the hidden state before attention.
    pub per_layer_embed_dim: Option<usize>,
    /// Number of layers at the end of the model that share KV from earlier layers.
    /// E.g., 20 means the last 20 layers reuse KV cache from earlier source layers.
    pub num_kv_shared_layers: Option<usize>,

    // ── Hybrid Gated DeltaNet + attention metadata (Qwen 3.6 family) ──
    // All optional; `None` on non-hybrid architectures.
    /// Stride at which a full softmax-attention layer appears.
    /// Qwen 3.6: 4 (i.e. every 4th layer is full attention).
    pub full_attention_interval: Option<usize>,
    /// DeltaNet per-head state width (head_k = head_v dim).
    /// Qwen 3.6: 128.
    pub ssm_state_size: Option<usize>,
    /// DeltaNet value-stream width (`head_v * n_v_heads`).
    /// Qwen 3.6: 6144 dense / 4096 MoE.
    pub ssm_inner_size: Option<usize>,
    /// DeltaNet number of V heads — GGUF metadata calls this
    /// `time_step_rank` despite the name. Qwen 3.6: 48 / 32.
    pub ssm_dt_rank: Option<usize>,
    /// DeltaNet number of K heads. Qwen 3.6: 16.
    pub ssm_group_count: Option<usize>,
    /// DeltaNet causal Conv1D window. Qwen 3.6: 4.
    pub ssm_conv_kernel: Option<usize>,
    /// Multi-section RoPE dimension partition. None = vanilla RoPE.
    pub rope_dimension_sections: Option<Vec<usize>>,
}

/// Architecture-specific behavior. Describes how a model is structured
/// without performing any computation.
pub trait ModelArchitecture: Send + Sync {
    /// Model family name (e.g., "gemma3", "llama").
    fn family(&self) -> &str;

    /// Parsed model configuration.
    fn config(&self) -> &ModelConfig;

    /// Validate parsed architecture dimensions and cross-field invariants.
    ///
    /// Detection is intentionally permissive so callers can inspect partially
    /// specified configs. Call this before inference or extraction to fail
    /// early on inconsistent head geometry, RoPE settings, per-layer metadata,
    /// MoE routing, or other values that would otherwise surface later.
    fn validate(&self) -> ConfigValidationResult {
        crate::validation::validate_architecture(self)
    }

    // ── Tensor key patterns ──

    /// Key prefix for a layer's tensors (e.g., "layers.5.").
    fn layer_prefix(&self, layer: usize) -> String {
        format!("layers.{layer}.")
    }

    /// Prefixes to strip from raw safetensors keys.
    /// Tried in order; first match wins.
    fn key_prefixes_to_strip(&self) -> &[&str] {
        &["language_model.model.", "model."]
    }

    /// Embedding tensor key (after prefix stripping).
    fn embed_key(&self) -> &str {
        "embed_tokens.weight"
    }

    /// Final norm weight key.
    fn final_norm_key(&self) -> &str {
        "norm.weight"
    }

    /// Attention weight keys for a layer.
    fn attn_q_key(&self, layer: usize) -> String {
        format!("{}self_attn.q_proj.weight", self.layer_prefix(layer))
    }
    fn attn_k_key(&self, layer: usize) -> String {
        format!("{}self_attn.k_proj.weight", self.layer_prefix(layer))
    }
    fn attn_v_key(&self, layer: usize) -> String {
        format!("{}self_attn.v_proj.weight", self.layer_prefix(layer))
    }
    fn attn_o_key(&self, layer: usize) -> String {
        format!("{}self_attn.o_proj.weight", self.layer_prefix(layer))
    }

    /// Attention bias keys (None if model doesn't use attention bias).
    fn attn_o_bias_key(&self, _layer: usize) -> Option<String> {
        None
    }
    fn attn_q_bias_key(&self, layer: usize) -> Option<String> {
        let _ = layer;
        None
    }
    fn attn_k_bias_key(&self, layer: usize) -> Option<String> {
        let _ = layer;
        None
    }
    fn attn_v_bias_key(&self, layer: usize) -> Option<String> {
        let _ = layer;
        None
    }

    /// QK norm weight keys (None if model doesn't use QK norm).
    fn attn_q_norm_key(&self, layer: usize) -> Option<String> {
        let _ = layer;
        None
    }
    fn attn_k_norm_key(&self, layer: usize) -> Option<String> {
        let _ = layer;
        None
    }

    /// FFN bias keys (None if model doesn't use FFN bias).
    fn ffn_up_bias_key(&self, _layer: usize) -> Option<String> {
        None
    }
    fn ffn_down_bias_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// FFN weight keys for a layer.
    fn ffn_gate_key(&self, layer: usize) -> String {
        format!("{}mlp.gate_proj.weight", self.layer_prefix(layer))
    }
    fn ffn_up_key(&self, layer: usize) -> String {
        format!("{}mlp.up_proj.weight", self.layer_prefix(layer))
    }
    fn ffn_down_key(&self, layer: usize) -> String {
        format!("{}mlp.down_proj.weight", self.layer_prefix(layer))
    }

    /// Layer norm weight keys.
    fn input_layernorm_key(&self, layer: usize) -> String {
        format!("{}input_layernorm.weight", self.layer_prefix(layer))
    }
    fn post_attention_layernorm_key(&self, layer: usize) -> String {
        format!(
            "{}post_attention_layernorm.weight",
            self.layer_prefix(layer)
        )
    }
    fn pre_feedforward_layernorm_key(&self, layer: usize) -> Option<String> {
        Some(format!(
            "{}pre_feedforward_layernorm.weight",
            self.layer_prefix(layer)
        ))
    }
    fn post_feedforward_layernorm_key(&self, layer: usize) -> Option<String> {
        Some(format!(
            "{}post_feedforward_layernorm.weight",
            self.layer_prefix(layer)
        ))
    }

    // ── Behavior ──

    /// Norm type (RMSNorm vs LayerNorm).
    fn norm_type(&self) -> NormType {
        NormType::RmsNorm
    }

    /// Weight offset added during layer normalization.
    /// Default 0.0 — saved weights are the final multiplier.
    fn norm_weight_offset(&self) -> f32 {
        0.0
    }

    /// Weight offset added during QK normalization (per-head Q/K norms).
    /// Gemma 2/3: 1.0 (weight = 1 + learned_weight at runtime), Gemma 4 and others: 0.0.
    fn qk_norm_weight_offset(&self) -> f32 {
        0.0
    }

    /// Embedding scaling factor applied after lookup.
    /// Gemma: sqrt(hidden_size), Granite: embedding_multiplier, Llama: 1.0.
    fn embed_scale(&self) -> f32 {
        self.config()
            .embedding_multiplier
            .map(|v| v as f32)
            .unwrap_or(1.0)
    }

    /// BOS token to prepend before inference when the tokenizer's
    /// `post_processor` doesn't already add one.
    ///
    /// Gemma 4's shipped `tokenizer.json` leaves BOS out of the
    /// `TemplateProcessing.single` template (unlike Gemma 2/3), so
    /// `tokenizer.encode(prompt, true)` returns tokens without BOS and
    /// the model sees a broken sequence. Architectures that need BOS
    /// return `Some(id)` here and callers prepend it if the encoding
    /// doesn't already start with it.
    fn bos_token_id(&self) -> Option<u32> {
        None
    }

    /// Activation function for the FFN.
    fn activation(&self) -> Activation {
        Activation::Silu
    }

    /// FFN type (gated vs standard).
    fn ffn_type(&self) -> FfnType {
        FfnType::Gated
    }

    /// Whether this model has separate pre/post norms around attention and FFN
    /// (Gemma 2/3 style with 4 norms per layer) vs standard pre-norm only.
    fn has_post_norms(&self) -> bool {
        false
    }

    /// Whether this layer uses sliding window attention.
    fn is_sliding_window_layer(&self, _layer: usize) -> bool {
        false
    }

    /// Sliding window size (None = full attention).
    fn sliding_window_size(&self) -> Option<usize> {
        self.config().sliding_window
    }

    /// RoPE base frequency for a given layer.
    /// Gemma3 uses different bases for sliding vs global attention layers.
    fn rope_base_for_layer(&self, layer: usize) -> f64 {
        let _ = layer;
        self.config().rope_base
    }

    /// Head dimension for a given layer. Models with different head dims for
    /// sliding vs global attention (e.g., Gemma 4) override this.
    /// Default: config.head_dim for all layers.
    fn head_dim_for_layer(&self, _layer: usize) -> usize {
        self.config().head_dim
    }

    /// Number of KV heads for a given layer. Models with different KV head counts
    /// for sliding vs global attention override this.
    /// Default: config.num_kv_heads for all layers.
    fn num_kv_heads_for_layer(&self, _layer: usize) -> usize {
        self.config().num_kv_heads
    }

    /// Number of Q heads for a given layer. Usually constant, but can vary
    /// when head_dim changes across layers (Q proj dim stays constant,
    /// but num_q_heads = q_proj_dim / head_dim).
    /// Default: config.num_q_heads for all layers.
    fn num_q_heads_for_layer(&self, _layer: usize) -> usize {
        self.config().num_q_heads
    }

    /// Fraction of head_dim to apply RoPE to (0.0–1.0).
    /// Models with partial rotary embedding (e.g., 0.25) override per layer.
    /// Default: 1.0 (full rotation).
    fn rotary_fraction_for_layer(&self, _layer: usize) -> f64 {
        1.0
    }

    /// Whether value shares key projections at this layer (V = K).
    /// When true, the forward pass uses K in place of V.
    /// Default: false.
    fn v_shares_k(&self, _layer: usize) -> bool {
        false
    }

    /// Whether to apply parameter-free RMSNorm to V states before attention.
    /// Gemma 4 applies V-norm (normalization only, no learned scale).
    /// Default: false.
    fn has_v_norm(&self) -> bool {
        false
    }

    /// Per-layer scalar multiplier key. When present, the layer output is
    /// multiplied by this learned scalar after the residual add.
    /// Default: None (no per-layer scalar).
    fn layer_scalar_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// Attention scale: 1/sqrt(query_pre_attn_scalar) or 1/sqrt(head_dim).
    fn attention_scale(&self) -> f64 {
        let scalar = self
            .config()
            .query_pre_attn_scalar
            .unwrap_or(self.config().head_dim as f64);
        scalar.powf(-0.5)
    }

    /// Attention scale for a specific layer. Accounts for per-layer head_dim
    /// when query_pre_attn_scalar is not set.
    fn attention_scale_for_layer(&self, layer: usize) -> f64 {
        if let Some(scalar) = self.config().query_pre_attn_scalar {
            scalar.powf(-0.5)
        } else {
            (self.head_dim_for_layer(layer) as f64).powf(-0.5)
        }
    }

    /// Source layer for KV sharing. Returns Some(source_layer) if this layer
    /// should reuse K/V from an earlier layer instead of computing its own.
    /// Default: None (every layer computes its own K/V).
    fn kv_shared_source_layer(&self, _layer: usize) -> Option<usize> {
        None
    }

    // ── Per-Layer Embeddings (PLE) ──

    /// Whether this model uses per-layer embeddings (PLE).
    /// When true, each layer adds a gated embedding lookup to the hidden state.
    fn has_per_layer_embeddings(&self) -> bool {
        self.config().per_layer_embed_dim.unwrap_or(0) > 0
    }

    /// Per-layer embedding dimension. 0 if PLE is not used.
    fn per_layer_embed_dim(&self) -> usize {
        self.config().per_layer_embed_dim.unwrap_or(0)
    }

    /// Key for the shared per-layer embedding matrix [vocab, num_layers * ple_dim].
    fn per_layer_embed_key(&self) -> Option<String> {
        if self.has_per_layer_embeddings() {
            Some("embed_tokens_per_layer.weight".to_string())
        } else {
            None
        }
    }

    /// Key for the per-layer input gate projection [ple_dim, hidden].
    fn per_layer_input_gate_key(&self, layer: usize) -> Option<String> {
        if self.has_per_layer_embeddings() {
            Some(format!(
                "{}per_layer_input_gate.weight",
                self.layer_prefix(layer)
            ))
        } else {
            None
        }
    }

    /// Key for the per-layer output projection [hidden, ple_dim].
    fn per_layer_projection_key(&self, layer: usize) -> Option<String> {
        if self.has_per_layer_embeddings() {
            Some(format!(
                "{}per_layer_projection.weight",
                self.layer_prefix(layer)
            ))
        } else {
            None
        }
    }

    /// Key for the post-PLE norm weight.
    fn post_per_layer_input_norm_key(&self, layer: usize) -> Option<String> {
        if self.has_per_layer_embeddings() {
            Some(format!(
                "{}post_per_layer_input_norm.weight",
                self.layer_prefix(layer)
            ))
        } else {
            None
        }
    }

    // ── Softcapping (Gemma2) ──

    /// Attention logit softcapping value (None = disabled).
    /// Applied before softmax: scores = tanh(scores / cap) * cap
    fn attn_logit_softcapping(&self) -> Option<f32> {
        self.config().attn_logit_softcapping.map(|v| v as f32)
    }

    /// Final logit softcapping value (None = disabled).
    /// Applied to output logits: logits = tanh(logits / cap) * cap
    fn final_logit_softcapping(&self) -> Option<f32> {
        self.config().final_logit_softcapping.map(|v| v as f32)
    }

    // ── Scaling multipliers (Granite-style) ──

    /// Residual stream scaling factor applied after attention and FFN additions.
    fn residual_multiplier(&self) -> f32 {
        self.config()
            .residual_multiplier
            .map(|v| v as f32)
            .unwrap_or(1.0)
    }

    /// Attention score scaling factor (applied on top of 1/sqrt(head_dim)).
    fn attention_multiplier(&self) -> f32 {
        self.config()
            .attention_multiplier
            .map(|v| v as f32)
            .unwrap_or(1.0)
    }

    /// Logits scaling factor applied to final logits before softmax.
    fn logits_scaling(&self) -> f32 {
        self.config()
            .logits_scaling
            .map(|v| v as f32)
            .unwrap_or(1.0)
    }

    // ── MoE (Mixture of Experts) ──

    /// How expert weights are stored in this model.
    fn expert_format(&self) -> ExpertFormat {
        ExpertFormat::PerExpert
    }

    /// Whether this model uses Mixture of Experts.
    fn is_moe(&self) -> bool {
        false
    }

    /// Number of routed experts per layer.
    fn num_experts(&self) -> usize {
        0
    }

    /// Number of experts activated per token.
    fn num_experts_per_token(&self) -> usize {
        0
    }

    /// Number of shared (always-active) experts.
    fn num_shared_experts(&self) -> usize {
        0
    }

    /// Router weight key for expert selection.
    fn moe_router_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// Router algorithm identifier (written into MoeConfig.router_type in vindex).
    /// Override in architectures with non-standard routing (e.g., Gemma 4's normalised softmax + per-expert scale).
    fn moe_router_type(&self) -> &str {
        "top_k_softmax"
    }

    /// Expert FFN gate weight key.
    fn expert_ffn_gate_key(&self, _layer: usize, _expert_id: usize) -> Option<String> {
        None
    }

    /// Expert FFN up-projection weight key.
    fn expert_ffn_up_key(&self, _layer: usize, _expert_id: usize) -> Option<String> {
        None
    }

    /// Expert FFN down-projection weight key.
    fn expert_ffn_down_key(&self, _layer: usize, _expert_id: usize) -> Option<String> {
        None
    }

    // ── Packed expert keys (MXFP4 models) ──

    /// Packed gate+up projection blocks key (all experts fused, MXFP4).
    fn packed_gate_up_blocks_key(&self, _layer: usize) -> Option<String> {
        None
    }
    /// Packed gate+up projection scales key.
    fn packed_gate_up_scales_key(&self, _layer: usize) -> Option<String> {
        None
    }
    /// Packed down projection blocks key.
    fn packed_down_blocks_key(&self, _layer: usize) -> Option<String> {
        None
    }
    /// Packed down projection scales key.
    fn packed_down_scales_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// Shared expert FFN gate weight key.
    fn shared_expert_gate_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// Shared expert FFN up-projection weight key.
    fn shared_expert_up_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// Shared expert FFN down-projection weight key.
    fn shared_expert_down_key(&self, _layer: usize) -> Option<String> {
        None
    }

    // ── Hybrid MoE (Gemma 4 A4B: dense MLP + expert block summed per layer) ──

    /// Whether this model has a hybrid dense-MLP + expert block per layer.
    /// Unlike pure MoE (Mixtral/DeepSeek), both branches run and their outputs are summed.
    fn is_hybrid_moe(&self) -> bool {
        false
    }

    /// Per-expert intermediate (hidden) dimension. 0 for non-MoE models.
    fn moe_intermediate_size(&self) -> usize {
        0
    }

    /// Packed stacked gate+up projection key (Gemma 4 PackedBF16 format).
    /// Tensor shape: [num_experts, 2 * moe_intermediate_size, hidden_size].
    fn packed_experts_gate_up_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// Packed stacked down projection key (Gemma 4 PackedBF16 format).
    /// Tensor shape: [num_experts, hidden_size, moe_intermediate_size].
    fn packed_experts_down_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// Gemma 4 router learned input-scale key (`router.scale`).
    fn moe_router_scale_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// Gemma 4 router per-expert output-scale key (`router.per_expert_scale`).
    fn moe_router_per_expert_scale_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// Router's own RMS-norm weight, applied to the router's input *before*
    /// the router projection. HF Gemma 4's `Gemma4TextRouter.norm` is
    /// **parameter-free** (`with_scale=False`) so no tensor exists on disk —
    /// see [`moe_router_norm_parameter_free`](Self::moe_router_norm_parameter_free).
    /// This key is provided for architectures that DO ship a learned router
    /// norm weight. Return `None` to fall back to either parameter-free
    /// RMSNorm (when the flag is set) or the experts' pre-norm output.
    fn moe_router_norm_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// Whether the router applies a parameter-free RMSNorm to its input
    /// before `router_scale`/projection. Gemma 4 sets this true. When true
    /// AND `moe_router_norm_key` returns `None`, the forward pass runs
    /// `x / sqrt(mean(x²) + eps)` on the raw residual instead of reusing
    /// the experts' pre-norm (which would apply the wrong learned weight).
    fn moe_router_norm_parameter_free(&self) -> bool {
        false
    }

    /// Scalar multiplier applied to the router input after `router.norm` and
    /// after the learned `router.scale` vector. Gemma 4 uses `hidden_size^-0.5`
    /// (called `scalar_root_size` in HF). Return `None` for no scaling.
    fn moe_router_input_scalar(&self) -> Option<f32> {
        None
    }

    /// Outer post-FFN norm for hybrid MoE layers — applied to `(h1 + h2)`
    /// before the residual add, where `h1 = post_ffn_norm_1(dense)` and
    /// `h2 = post_ffn_norm_2(moe)`. HF Gemma 4 stores this at the un-suffixed
    /// key `post_feedforward_layernorm.weight`, while the dense-branch norm
    /// uses the suffixed `_1` variant (see `post_feedforward_layernorm_key`).
    /// Return `None` for architectures that either don't combine via an outer
    /// norm or reuse the dense-branch norm as the outer norm.
    fn moe_post_outer_norm_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// Post-FFN norm for dense MLP output in hybrid MoE layers.
    /// Gemma 4 A4B: `post_feedforward_layernorm_1.weight` (replaces the plain variant).
    fn moe_post_ffn1_norm_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// Pre-norm applied to the residual before feeding into the expert block.
    /// Gemma 4 A4B: `pre_feedforward_layernorm_2.weight`.
    fn moe_pre_experts_norm_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// Post-norm applied to the expert block output.
    /// Gemma 4 A4B: `post_feedforward_layernorm_2.weight`.
    fn moe_post_experts_norm_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// Whether the hybrid MoE forward applies a final RMS norm to the
    /// combined (dense + expert) output before adding to the residual.
    ///
    /// Gemma 4 26B A4B: true — matches HF `post_feedforward_layernorm(combined)`.
    /// All other models: false — use `layer_scalar * combined` instead.
    fn moe_has_combined_output_norm(&self) -> bool {
        false
    }

    // ── MLA (Multi-head Latent Attention) ──

    /// Whether this model uses MLA instead of standard GQA.
    fn uses_mla(&self) -> bool {
        false
    }

    /// MLA compressed KV dimension.
    fn kv_lora_rank(&self) -> usize {
        0
    }

    /// MLA Q compression rank.
    fn q_lora_rank(&self) -> usize {
        0
    }

    /// MLA KV down-projection key (compress).
    fn mla_kv_a_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// MLA KV up-projection key (decompress).
    fn mla_kv_b_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// MLA Q down-projection key (compress).
    fn mla_q_a_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// MLA Q up-projection key (decompress).
    fn mla_q_b_key(&self, _layer: usize) -> Option<String> {
        None
    }

    // ── RoPE scaling ──

    /// RoPE scaling type (None, "linear", "yarn", "dynamic", "llama3").
    fn rope_scaling_type(&self) -> Option<&str> {
        self.config()
            .rope_scaling
            .as_ref()
            .map(|s| s.scaling_type.as_str())
    }

    /// RoPE scaling factor.
    fn rope_scaling_factor(&self) -> f64 {
        self.config()
            .rope_scaling
            .as_ref()
            .map_or(1.0, |s| s.factor)
    }

    /// Norm epsilon for RMSNorm / LayerNorm. Default: 1e-6.
    fn norm_eps(&self) -> f32 {
        1e-6
    }

    // ── Hybrid Gated DeltaNet + full-attention (Qwen 3.6 family) ──
    //
    // Models in this family (qwen35, qwen35moe) interleave Gated
    // DeltaNet linear-attention layers with periodic full softmax-
    // attention layers. The dispatcher picks the layer kind via
    // `is_linear_attention_layer(layer)`. For linear layers, the
    // `*_linear_*_key` methods supply the per-layer tensor names
    // (fused QKV, Z gate, SSM block weights, etc.). For full-attn
    // layers the existing `attn_q_key`/`attn_k_key`/... apply, though
    // qwen35's full-attn layers also have a fused Q+per-head-gate
    // projection (the `attn_q_key` returns the fused tensor; the
    // caller splits the output along the head dim).
    //
    // Default impls return `false` / `None` so non-hybrid
    // architectures are unaffected.

    /// True iff layer `layer` is a Gated DeltaNet linear-attention
    /// layer (as opposed to a full softmax-attention layer).
    fn is_linear_attention_layer(&self, _layer: usize) -> bool {
        false
    }

    /// Linear-attention fused QKV projection
    /// (`attn_qkv.weight` → `[n_embd, 2*key_dim + value_dim]`).
    fn attn_qkv_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// Z-gate projection on linear-attention layers (DeltaNet's
    /// post-mixer gate). For qwen35 full-attn layers this is
    /// `None`; the fused Q+gate is part of `attn_q_key`'s tensor
    /// instead.
    fn attn_gate_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// Causal depthwise Conv1D weight (`[d_conv, conv_dim]`).
    fn ssm_conv1d_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// Bias added to `alpha` before the softplus gate.
    fn ssm_dt_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// Per-head log-decay scalar (`[n_v_heads]`).
    fn ssm_a_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// Beta projection (delta-rule learning rate).
    fn ssm_beta_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// Alpha projection (pre-softplus gate input).
    fn ssm_alpha_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// Post-mixer RMSNorm weight (`[head_v_dim]`).
    fn ssm_norm_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// Linear-attention output projection (`[value_dim, n_embd]`).
    fn ssm_out_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// Per-head RMSNorm weight on Q in full-attention layers
    /// (qwen35: `[head_dim]`, applied after Q split, before RoPE).
    /// Distinct from `attn_q_norm_key` which on Qwen3 has shape
    /// `[n_embd]`. None for non-qwen35 archs.
    fn attn_q_per_head_norm_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// Per-head RMSNorm weight on K in full-attention layers.
    fn attn_k_per_head_norm_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// Multi-section RoPE: per-section dimension counts. None =
    /// vanilla single-section RoPE.
    fn rope_dimension_sections(&self) -> Option<&[usize]> {
        None
    }

    /// DeltaNet per-head state width (head_k = head_v dim).
    /// 0 if this isn't a hybrid model.
    fn ssm_state_size(&self) -> usize {
        0
    }

    /// DeltaNet value-stream width (`head_v * n_v_heads`).
    fn ssm_inner_size(&self) -> usize {
        0
    }

    /// DeltaNet K head count.
    fn ssm_group_count(&self) -> usize {
        0
    }

    /// DeltaNet V head count.
    fn ssm_dt_rank(&self) -> usize {
        0
    }

    /// DeltaNet causal Conv1D window size.
    fn ssm_conv_kernel(&self) -> usize {
        0
    }

    /// Stride at which a full softmax-attention layer appears in a
    /// hybrid architecture. `(layer + 1) % interval == 0` →
    /// full-attention; else linear. 0 = not hybrid.
    fn full_attention_interval(&self) -> usize {
        0
    }
}
