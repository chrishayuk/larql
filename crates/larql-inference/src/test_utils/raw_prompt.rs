//! The three BOS shapes a raw (non-templated) VINDEX3 prompt must encode
//! identically on every surface — `larql run --raw`, `vindex3 observe`,
//! the server's plain completions and LQL. Each surface's parity test
//! writes these declarations beside its container and asserts the ids it
//! feeds equal [`BosShape::expected_ids`], so the surfaces agree with
//! each other through one table.

use std::path::Path;

use larql_vindex::format::filenames::{
    GENERATION_CONFIG_JSON, TOKENIZER_CONFIG_JSON, TOKENIZER_JSON,
};
use serde_json::json;

use super::synthetic_tokenizer_json;

/// The raw prompt every case encodes: two word-level tokens.
pub const RAW_PROMPT: &str = "[3] [4]";
/// The prompt's own ids, before any BOS.
pub const RAW_PROMPT_IDS: [u32; 2] = [3, 4];
/// The BOS id every case declares.
pub const RAW_PROMPT_BOS: u32 = 2;

/// How a checkpoint declares BOS.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BosShape {
    /// Granite / Qwen3: a BOS id declared, `add_bos_token: false`, a
    /// tokenizer that adds none. No BOS.
    DeclaredNotAdded,
    /// Gemma 4: a BOS id declared, no `add_bos_token`, a tokenizer that
    /// adds none. BOS prepended once.
    DeclaredRequired,
    /// A tokenizer whose post-processor already adds the declared BOS.
    /// Not doubled.
    TokenizerAdds,
}

impl BosShape {
    pub const ALL: [BosShape; 3] = [
        BosShape::DeclaredNotAdded,
        BosShape::DeclaredRequired,
        BosShape::TokenizerAdds,
    ];

    /// The ids every surface must feed for [`RAW_PROMPT`].
    pub fn expected_ids(self) -> Vec<u32> {
        let mut ids = Vec::new();
        if self != BosShape::DeclaredNotAdded {
            ids.push(RAW_PROMPT_BOS);
        }
        ids.extend(RAW_PROMPT_IDS);
        ids
    }

    /// Write this shape's `tokenizer.json`, `generation_config.json` and
    /// `tokenizer_config.json` into `dir` (a container root).
    pub fn write_into(self, dir: &Path, vocab: usize) {
        let mut tokenizer: serde_json::Value =
            serde_json::from_str(&synthetic_tokenizer_json(vocab)).expect("synthetic tokenizer");
        tokenizer["pre_tokenizer"] = json!({"type": "WhitespaceSplit"});
        if self == BosShape::TokenizerAdds {
            let bos = format!("[{RAW_PROMPT_BOS}]");
            tokenizer["post_processor"] = json!({
                "type": "TemplateProcessing",
                "single": [
                    {"SpecialToken": {"id": bos, "type_id": 0}},
                    {"Sequence": {"id": "A", "type_id": 0}}
                ],
                "pair": [
                    {"SpecialToken": {"id": bos, "type_id": 0}},
                    {"Sequence": {"id": "A", "type_id": 0}},
                    {"Sequence": {"id": "B", "type_id": 1}}
                ],
                "special_tokens": {}
            });
            tokenizer["post_processor"]["special_tokens"][bos.as_str()] =
                json!({"id": bos, "ids": [RAW_PROMPT_BOS], "tokens": [bos]});
        }
        let tokenizer_config = match self {
            BosShape::DeclaredNotAdded => json!({"add_bos_token": false}),
            BosShape::DeclaredRequired | BosShape::TokenizerAdds => json!({}),
        };
        let write = |name: &str, body: String| {
            std::fs::write(dir.join(name), body).expect("write fixture declaration")
        };
        write(TOKENIZER_JSON, tokenizer.to_string());
        write(
            GENERATION_CONFIG_JSON,
            json!({"bos_token_id": RAW_PROMPT_BOS}).to_string(),
        );
        write(TOKENIZER_CONFIG_JSON, tokenizer_config.to_string());
    }
}
