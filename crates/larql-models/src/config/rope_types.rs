//! HF `rope_scaling.rope_type` selector values.
//!
//! These are protocol strings — they name what a checkpoint's `config.json`
//! says, not an internal concept — so they are matched case-insensitively and
//! declared once here rather than spelled inline at each comparison.

/// No scaling — plain RoPE at the layer's base frequency. HF writes this
/// explicitly in the `sliding_attention` slot of Gemma 3's structured
/// per-layer-type form, which is the one place larql has to *emit* it
/// rather than only recognise it (see `RopeScaling::to_config_json`).
pub const ROPE_TYPE_DEFAULT: &str = "default";

/// Divide the position by `factor`. Gemma 3 applies this to global layers only,
/// via the structured per-layer-type form.
pub const ROPE_TYPE_LINEAR: &str = "linear";

/// Wavelength-dependent per-channel frequency scaling. Needs all four
/// `llama3_*` fields populated.
pub const ROPE_TYPE_LLAMA3: &str = "llama3";

/// Ramped interpolation/extrapolation frequency blend **plus** an amplitude on
/// `cos`/`sin`. GPT-OSS and DeepSeek.
pub const ROPE_TYPE_YARN: &str = "yarn";

/// Partial rotary whose inverse frequencies are taken over the FULL head
/// width (`base^(2i/head_dim)`, first `partial_rotary_factor · head_dim`
/// dims rotated, the rest at zero frequency) — HF
/// `_compute_proportional_rope_parameters`. Gemma 4 declares it on its
/// full-attention layers, over `global_head_dim`. Distinct from the plain
/// partial rotary (`default` + `partial_rotary_factor`), whose frequencies
/// are taken over the rotary width: same rotated dims, different angles.
pub const ROPE_TYPE_PROPORTIONAL: &str = "proportional";
