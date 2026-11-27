//! What a family's config class means when `config.json` omits a field.
//!
//! These are per-family facts (they mirror each family's `transformers`
//! config class), so they live on the family's registry row. The generic
//! parser reads them through [`super::find_architecture`] and never names a
//! family itself.

use crate::defaults::ROPE_BASE_DEFAULT;

/// How `intermediate_size` is found when the config does not declare it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntermediateSize {
    /// It must be declared; its absence is refused on the disk path.
    Required,
    /// Derived as this multiple of `hidden_size` (GPT-2's `4 * n_embd`).
    HiddenMultiple(usize),
    /// The family has no FFN, so there is nothing to declare (Mamba2).
    NotApplicable,
}

/// A family's defaults for fields its config class lets `config.json` omit.
#[derive(Clone, Copy, Debug)]
pub struct ConfigDefaults {
    /// `rope_theta` when no RoPE base is declared.
    pub rope_theta: f64,
    /// `head_dim` when undeclared. `None` derives
    /// `hidden_size / num_attention_heads`.
    pub head_dim: Option<usize>,
    /// `num_attention_heads` when undeclared. `None` leaves it absent (0),
    /// for the architecture's validation to refuse.
    pub num_attention_heads: Option<usize>,
    /// `num_key_value_heads` when undeclared. `None` is multi-head
    /// attention: one KV head per query head.
    pub num_key_value_heads: Option<usize>,
    /// How `intermediate_size` is found when undeclared.
    pub intermediate_size: IntermediateSize,
    /// The family's own spelling of the layer count, consulted when no
    /// generic alias (`num_hidden_layers`, `n_layer`) is declared —
    /// GPT-Neo's `num_layers`. Family-scoped because the same word means
    /// something else in other config classes.
    pub layer_count_key: Option<&'static str>,
    /// The family's own spelling of the attention-head count — GPT-Neo's
    /// `num_heads`, which on a Mamba2 config is the MIXER's head count and
    /// must never be read as attention heads there.
    pub head_count_key: Option<&'static str>,
    /// What the family's TOKENIZER class does about a BOS when
    /// `tokenizer_config.json` does not declare `add_bos_token`:
    /// `Some(false)` = it adds none (`GPT2Tokenizer`'s class default, so
    /// GPT-2 and GPT-Neo prompts start with the first text token). `None`
    /// = no family judgment, and a declared BOS id is prepended — the rule
    /// every other family is served by.
    pub tokenizer_adds_bos: Option<bool>,
}

impl ConfigDefaults {
    /// The conventional decoder: RoPE base 10 000, derived head width,
    /// multi-head attention, and a declared FFN width.
    pub const STANDARD: Self = Self {
        rope_theta: ROPE_BASE_DEFAULT,
        head_dim: None,
        num_attention_heads: None,
        num_key_value_heads: None,
        intermediate_size: IntermediateSize::Required,
        layer_count_key: None,
        head_count_key: None,
        tokenizer_adds_bos: None,
    };
}
