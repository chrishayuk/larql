//! Shared numerical defaults for model architectures.
//!
//! These are the values HuggingFace transformers uses when an HF config omits
//! the corresponding field. They live in one module so the parser
//! (`detect/parser.rs`), the trait defaults (`config.rs`), and per-architecture
//! fallbacks (`architectures/gemma{3,4}.rs`, etc.) all reference the same
//! number — preventing the kind of drift that, before 2026-05-16, had
//! `rms_norm_eps` defaulting to 1e-6 in inference even when the model's
//! config.json explicitly specified 1e-5.

/// Default RoPE theta for Gemma-family models when `rope_theta` is absent.
/// Matches HF `Gemma3TextConfig.rope_theta` class default.
pub const ROPE_BASE_GEMMA: f64 = 1_000_000.0;

/// Default RoPE theta for non-Gemma families and the per-layer
/// `rope_local_base` fallback (Gemma 3 sliding layers use this).
/// Matches HF `LlamaConfig.rope_theta` class default.
pub const ROPE_BASE_DEFAULT: f64 = 10_000.0;

/// Default RMS-norm / LayerNorm epsilon when the model's config omits any
/// of `rms_norm_eps` / `layer_norm_eps` / `layer_norm_epsilon` /
/// `norm_epsilon`. Older models (BERT, Llama 1, Gemma 1) used 1e-6;
/// most modern architectures (Llama 3.x, Mistral, Gemma 3) ship 1e-5
/// explicitly so this fallback rarely fires.
pub const DEFAULT_NORM_EPS: f32 = 1e-6;

// ── Llama-3 `rope_scaling` class defaults ───────────────────────────────
// Llama-3.x configs ship the full four-field set, but synthetic / partial
// configs (test fixtures, custom checkpoints) may omit individual fields.
// These match HF's class defaults in `Llama3TextConfig` so a partial
// llama3 rope_scaling block resolves identically on both sides.

/// Default `low_freq_factor` for HF `rope_scaling = {rope_type: llama3}`.
pub const LLAMA3_LOW_FREQ_FACTOR_DEFAULT: f64 = 1.0;

/// Default `high_freq_factor` for HF `rope_scaling = {rope_type: llama3}`.
pub const LLAMA3_HIGH_FREQ_FACTOR_DEFAULT: f64 = 4.0;

/// Default `original_max_position_embeddings` for HF `rope_scaling =
/// {rope_type: llama3}`. Llama-3 was pre-trained at 8K context before
/// long-context fine-tuning.
pub const LLAMA3_ORIGINAL_MAX_POSITION_EMBEDDINGS_DEFAULT: f64 = 8192.0;
