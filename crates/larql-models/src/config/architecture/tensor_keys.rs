//! The [`TensorKeys`] slice of [`super::ModelArchitecture`].

use super::ArchitectureCore;
use crate::config::SourceLayout;

/// Checkpoint tensor names: where each weight of a layer lives.
pub trait TensorKeys: ArchitectureCore {
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

    /// Whether the model has a text output projection at all — a
    /// standalone `lm_head.weight` or one tied to the embedding matrix.
    /// `true` for every text LM. `false` for models whose backbone's only
    /// product is its final hidden state (MOSS-TTS-Realtime: the output
    /// heads live in a side-loaded audio depth transformer, and
    /// `tie_word_embeddings` semantics do not apply). When `false`, the
    /// safetensors loader stores the embedding as a placeholder head that
    /// must never be sampled from; it becomes properly optional when
    /// generation output moves behind the output-adapter seam
    /// (`docs/tts-funnel.md` §3 step 4).
    fn has_lm_head(&self) -> bool {
        true
    }

    /// Whether a *missing* standalone output-head tensor should be read as
    /// tied to the embedding matrix, rather than a lost one.
    ///
    /// `true` when this architecture has no independent head at all
    /// ([`Self::has_lm_head`] false — the embedding is a
    /// never-sampled placeholder) or when the checkpoint does not declare
    /// `tie_word_embeddings: false`. `false` only when a checkpoint
    /// explicitly declares `tie_word_embeddings: false` and still has no
    /// separate head tensor — the one case that must surface as a loud
    /// missing-tensor error rather than silently serving the wrong
    /// projection (GPT-OSS, OLMoE both declare `false`).
    ///
    /// The one statement of this rule: both loaders'
    /// `resolve_lm_head` (`loading/lm_head.rs`) checks the
    /// identical `tie_word_embeddings == Some(false)` fact to decide the
    /// same question for the VINDEX2 load path.
    fn output_head_reuses_embedding(&self) -> bool {
        !self.has_lm_head() || self.config().tie_word_embeddings != Some(false)
    }

    /// Learned positional-embedding tensor key, when the architecture uses
    /// absolute learned position embeddings added to the token embedding at
    /// the input (GPT-2). Architectures that use rotary or no positional
    /// signal return `None` (the default). When `Some`, the loader populates
    /// `ModelWeights::position_embed`.
    fn position_embed_key(&self) -> Option<&str> {
        None
    }

    /// How this family STORES its projections, when that is not the
    /// canonical `[out, in]` layout: input-major (`Conv1D`) modules, fused
    /// projections to split, and registered buffers that are not operands.
    /// Normalised once at the inventory boundary (see
    /// [`SourceLayout`]); the default derives nothing.
    fn source_layout(&self) -> SourceLayout {
        SourceLayout::default()
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

    /// Tensor key for a fused Q/K/V projection, when the architecture packs
    /// all three into a single matrix (Conv1D-style: GPT-2). The loader splits
    /// the fused tensor into the per-projection keys returned by
    /// `attn_q_key` / `attn_k_key` / `attn_v_key` after loading. Default: None.
    fn fused_qkv_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// Tensor key for a fused Q/K/V bias vector, paired with `fused_qkv_key`.
    /// Default: None.
    fn fused_qkv_bias_key(&self, _layer: usize) -> Option<String> {
        None
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

    /// Attention-sink key (None if the architecture has no sinks).
    ///
    /// A sink is a learned per-head logit concatenated to the attention
    /// logits, included in the softmax, then dropped — so the weights
    /// over real keys deliberately sum to less than one. Architectures
    /// that use them (GPT-OSS) must both return this key *and* have the
    /// attention kernel apply it; returning the key alone changes
    /// nothing but the extracted bytes.
    fn attn_sinks_key(&self, _layer: usize) -> Option<String> {
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

    /// Per-layer scalar multiplier key. When present, the layer output is
    /// multiplied by this learned scalar after the residual add.
    /// Default: None (no per-layer scalar).
    fn layer_scalar_key(&self, _layer: usize) -> Option<String> {
        None
    }
}
