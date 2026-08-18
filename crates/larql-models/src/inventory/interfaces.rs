//! The multimodal interface reader: the root-level facts that bind a text
//! decoder to its perception towers — which special-token ids delimit an
//! image / audio / video span, how many soft tokens an image yields, and
//! whether the text model attends bidirectionally across a span — plus
//! the declaration that a component is ABSENT (`audio_config: null`).
//!
//! The third recorded reader (after the nested-component topology and the
//! stored-representation readers): it records the full path of every key
//! it reads into [`InterfaceReading::consumed_paths`], and key
//! classification credits exactly that set. Interface facts are neither
//! tensor semantics (nothing is stored) nor forward-pass execution
//! semantics of one component — they are the contract BETWEEN components,
//! which is why they have their own home rather than a slot on
//! `ModelConfig`.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Root-level special-token roles, in the spelling HF multimodal configs
/// use. Each names the token that opens, closes or stands in for one
/// perception span. `image_token_id` and `video_token_id` are also read by
/// the text parser; recording them here too keeps the interface complete
/// in one place.
pub const TOKEN_ROLE_KEYS: &[&str] = &[
    "image_token_id",
    "video_token_id",
    "audio_token_id",
    "boi_token_id",
    "eoi_token_id",
    "boa_token_id",
    "eoa_token_id",
    // A second spelling HF Gemma 4 ships beside `eoa_token_id`, same value.
    "eoa_token_index",
];

/// Root-level count of soft tokens one image expands to.
pub const SOFT_TOKENS_PER_IMAGE_KEY: &str = "vision_soft_tokens_per_image";

/// Component keys whose declared ABSENCE (`null`) is itself an interface
/// fact: the checkpoint says it has no such tower.
pub const OPTIONAL_COMPONENT_KEYS: &[&str] = &["audio_config"];

/// The text-side masking policy over perception spans
/// (`text_config.use_bidirectional_attention`, Gemma 4: `"vision"`).
pub const BIDIRECTIONAL_ATTENTION_PATH: (&str, &str) =
    ("text_config", "use_bidirectional_attention");

/// What the checkpoint declares about the joins between its components.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct MultimodalInterface {
    /// `(key, token id)` for every declared special-token role, in
    /// [`TOKEN_ROLE_KEYS`] order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub token_roles: Vec<(String, u64)>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub soft_tokens_per_image: Option<u64>,
    /// Components the checkpoint declares it does NOT have.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub absent_components: Vec<String>,
    /// The span kind the text model attends bidirectionally over, verbatim.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bidirectional_attention: Option<String>,
}

impl MultimodalInterface {
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }
}

/// One reading: the interface plus the paths it consumed.
#[derive(Debug, Clone)]
pub struct InterfaceReading {
    pub interface: MultimodalInterface,
    pub consumed_paths: BTreeSet<String>,
}

/// Read the interface facts a config declares. `None` when it declares
/// none — a text-only checkpoint has no interface to record.
pub fn read_interface(config: &Value) -> Option<InterfaceReading> {
    let mut consumed_paths = BTreeSet::new();
    let mut interface = MultimodalInterface::default();
    for key in TOKEN_ROLE_KEYS {
        if let Some(id) = config.get(key).and_then(Value::as_u64) {
            consumed_paths.insert((*key).to_string());
            interface.token_roles.push(((*key).to_string(), id));
        }
    }
    if let Some(n) = config
        .get(SOFT_TOKENS_PER_IMAGE_KEY)
        .and_then(Value::as_u64)
    {
        consumed_paths.insert(SOFT_TOKENS_PER_IMAGE_KEY.to_string());
        interface.soft_tokens_per_image = Some(n);
    }
    for key in OPTIONAL_COMPONENT_KEYS {
        if config.get(key).is_some_and(Value::is_null) {
            consumed_paths.insert((*key).to_string());
            interface.absent_components.push((*key).to_string());
        }
    }
    let (container, leaf) = BIDIRECTIONAL_ATTENTION_PATH;
    if let Some(kind) = config
        .get(container)
        .and_then(|c| c.get(leaf))
        .and_then(Value::as_str)
    {
        consumed_paths.insert(format!("{container}.{leaf}"));
        interface.bidirectional_attention = Some(kind.to_string());
    }
    (!interface.is_empty()).then_some(InterfaceReading {
        interface,
        consumed_paths,
    })
}

#[cfg(test)]
mod tests;
