//! Every `tokenizer.ggml.*` key, and the container file that produces it.
//!
//! The same rule as the metadata table: **no literal unless it is a
//! target constant.** The tokens, merges, types and special ids all
//! come from the capability snapshot the container carries
//! (`tokenizer.json` and friends); the two constants are llama.cpp's
//! vocabulary-model name (`gpt2` — the byte-level BPE loader) and its
//! pre-tokenizer id (`qwen2` — the regex family this tokenizer's own
//! declared pre-tokenizer matches). Both are facts about llama.cpp.
//!
//! Two decisions worth stating rather than burying:
//!
//! - **The token table is padded to the model's vocabulary.** The
//!   embedding carries `vocab_size` rows (a graph fact); the tokenizer
//!   defines fewer ids. llama.cpp sizes the model from the token list,
//!   so the gap is filled with explicit `[PAD{id}]` entries marked
//!   UNUSED — the same spelling its own converter writes. An id the
//!   tokenizer defines *beyond* the model's vocabulary is refused: that
//!   is a tokenizer for a different model.
//! - **Special ids resolve through the files, in order.** The eos/pad
//!   ids come from `tokenizer_config.json`'s named tokens when present,
//!   else `generation_config.json`'s ids (first of a list). A container
//!   with neither refuses — an unterminated chat model is not a
//!   convention this table is willing to guess.

use std::path::Path;

use larql_models::loading::gguf::GgufValue;

use crate::VindexError;

/// llama.cpp's byte-level BPE vocabulary model.
pub const VOCAB_MODEL: &str = "gpt2";
/// llama.cpp's pre-tokenizer id whose regex family matches this
/// tokenizer's declared `pre_tokenizer`.
pub const VOCAB_PRE: &str = "qwen2";

/// gguf token-type ids (llama.cpp's `TokenType` enum).
const TYPE_NORMAL: i32 = 1;
const TYPE_CONTROL: i32 = 3;
const TYPE_USER_DEFINED: i32 = 4;
const TYPE_UNUSED: i32 = 5;

#[derive(Debug)]
pub struct VocabTable {
    pub entries: Vec<(String, GgufValue)>,
    pub tokens: usize,
    pub padded: usize,
    pub merges: usize,
    pub control: usize,
    pub user_defined: usize,
}

fn read_json(path: &Path) -> Result<Option<serde_json::Value>, VindexError> {
    if !path.exists() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(path)?;
    serde_json::from_str(&text)
        .map(Some)
        .map_err(|e| VindexError::Parse(format!("{}: {e}", path.display())))
}

/// Build the complete tokenizer table from the container's capability
/// snapshot, sized to the model's declared vocabulary.
pub fn qwen35_vocab(container: &Path, vocab_size: usize) -> Result<VocabTable, VindexError> {
    let tokenizer = read_json(&container.join("tokenizer.json"))?.ok_or_else(|| {
        VindexError::Parse(format!(
            "{}: no tokenizer.json — the container carries no text capability to export",
            container.display()
        ))
    })?;

    let model = &tokenizer["model"];
    let kind = model["type"].as_str().unwrap_or("");
    if kind != "BPE" {
        return Err(VindexError::Parse(format!(
            "tokenizer model type `{kind}` — this table serialises byte-level BPE only, and \
             pretending otherwise would load and mis-tokenise"
        )));
    }

    // Tokens by id: the base vocabulary, then the added tokens.
    let mut tokens: Vec<Option<String>> = vec![None; vocab_size];
    let mut types: Vec<i32> = vec![TYPE_NORMAL; vocab_size];
    let mut place = |content: &str, id: u64, ty: i32| -> Result<(), VindexError> {
        let Some(slot) = tokens.get_mut(id as usize) else {
            return Err(VindexError::Parse(format!(
                "token id {id} (`{content}`) is outside the model's {vocab_size}-token \
                 vocabulary — this tokenizer belongs to a different model"
            )));
        };
        if let Some(existing) = slot {
            return Err(VindexError::Parse(format!(
                "token id {id} is claimed twice: `{existing}` and `{content}`"
            )));
        }
        *slot = Some(content.to_string());
        types[id as usize] = ty;
        Ok(())
    };

    let vocab = model["vocab"]
        .as_object()
        .ok_or_else(|| VindexError::Parse("tokenizer.json: model.vocab is not an object".into()))?;
    for (content, id) in vocab {
        let id = id.as_u64().ok_or_else(|| {
            VindexError::Parse(format!(
                "tokenizer.json: vocab id for `{content}` is not a u64"
            ))
        })?;
        place(content, id, TYPE_NORMAL)?;
    }
    let mut control = 0usize;
    let mut user_defined = 0usize;
    let mut by_content: std::collections::BTreeMap<String, u64> = vocab
        .iter()
        .filter_map(|(c, id)| id.as_u64().map(|i| (c.clone(), i)))
        .collect();
    if let Some(added) = tokenizer["added_tokens"].as_array() {
        for t in added {
            let content = t["content"].as_str().unwrap_or_default();
            let id = t["id"]
                .as_u64()
                .ok_or_else(|| VindexError::Parse(format!("added token `{content}` has no id")))?;
            let special = t["special"].as_bool().unwrap_or(false);
            let ty = if special {
                TYPE_CONTROL
            } else {
                TYPE_USER_DEFINED
            };
            if special {
                control += 1;
            } else {
                user_defined += 1;
            }
            place(content, id, ty)?;
            by_content.insert(content.to_string(), id);
        }
    }
    let defined = tokens.iter().filter(|t| t.is_some()).count();
    let mut padded = 0usize;
    let token_list: Vec<GgufValue> = tokens
        .into_iter()
        .enumerate()
        .map(|(id, t)| match t {
            Some(t) => GgufValue::String(t),
            None => {
                // The embedding has this row; the tokenizer defines no
                // id for it. The gap is stated, not hidden.
                padded += 1;
                types[id] = TYPE_UNUSED;
                GgufValue::String(format!("[PAD{id}]"))
            }
        })
        .collect();

    // Merges, verbatim. Newer HF spells a merge as a two-element array;
    // llama.cpp wants the joined "left right" form either way.
    let merges: Vec<GgufValue> = tokenizer["model"]["merges"]
        .as_array()
        .ok_or_else(|| VindexError::Parse("tokenizer.json: model.merges missing".into()))?
        .iter()
        .map(|m| {
            if let Some(s) = m.as_str() {
                Ok(GgufValue::String(s.to_string()))
            } else if let Some(pair) = m.as_array() {
                match (pair[0].as_str(), pair[1].as_str()) {
                    (Some(a), Some(b)) => Ok(GgufValue::String(format!("{a} {b}"))),
                    _ => Err(VindexError::Parse(
                        "tokenizer.json: malformed merge pair".into(),
                    )),
                }
            } else {
                Err(VindexError::Parse(
                    "tokenizer.json: malformed merge entry".into(),
                ))
            }
        })
        .collect::<Result<_, _>>()?;
    let merge_count = merges.len();

    // Special ids: the config files speak; this table does not guess.
    let tokenizer_config = read_json(&container.join("tokenizer_config.json"))?;
    let generation_config = read_json(&container.join("generation_config.json"))?;
    let id_of_named = |key: &str| -> Option<u64> {
        let name = tokenizer_config.as_ref()?.get(key)?;
        let content = name.as_str().or_else(|| name.get("content")?.as_str())?;
        by_content.get(content).copied()
    };
    let id_of_generation = |key: &str| -> Option<u64> {
        let v = generation_config.as_ref()?.get(key)?;
        v.as_u64().or_else(|| v.as_array()?.first()?.as_u64())
    };
    let eos = id_of_named("eos_token")
        .or_else(|| id_of_generation("eos_token_id"))
        .ok_or_else(|| {
            VindexError::Parse(
                "no eos token: neither tokenizer_config.json nor generation_config.json \
                 names one, and an unterminated chat model is not a guessable convention"
                    .into(),
            )
        })?;
    let pad = id_of_named("pad_token").or_else(|| id_of_generation("pad_token_id"));
    let bos = id_of_named("bos_token").or_else(|| id_of_generation("bos_token_id"));

    let mut entries = vec![
        (
            "tokenizer.ggml.model".to_string(),
            GgufValue::String(VOCAB_MODEL.into()),
        ),
        (
            "tokenizer.ggml.pre".to_string(),
            GgufValue::String(VOCAB_PRE.into()),
        ),
        (
            "tokenizer.ggml.tokens".to_string(),
            GgufValue::Array(token_list),
        ),
        (
            "tokenizer.ggml.token_type".to_string(),
            GgufValue::Array(types.iter().map(|t| GgufValue::I32(*t)).collect()),
        ),
        (
            "tokenizer.ggml.merges".to_string(),
            GgufValue::Array(merges),
        ),
        (
            "tokenizer.ggml.eos_token_id".to_string(),
            GgufValue::U32(eos as u32),
        ),
    ];
    if let Some(pad) = pad {
        entries.push((
            "tokenizer.ggml.padding_token_id".to_string(),
            GgufValue::U32(pad as u32),
        ));
    }
    if let Some(bos) = bos {
        entries.push((
            "tokenizer.ggml.bos_token_id".to_string(),
            GgufValue::U32(bos as u32),
        ));
    }
    if let Some(add_bos) = tokenizer_config
        .as_ref()
        .and_then(|c| c.get("add_bos_token"))
        .and_then(|v| v.as_bool())
    {
        entries.push((
            "tokenizer.ggml.add_bos_token".to_string(),
            GgufValue::Bool(add_bos),
        ));
    }
    let template_path = container.join("chat_template.jinja");
    if template_path.exists() {
        entries.push((
            "tokenizer.chat_template".to_string(),
            GgufValue::String(std::fs::read_to_string(&template_path)?),
        ));
    } else if let Some(t) = tokenizer_config
        .as_ref()
        .and_then(|c| c.get("chat_template"))
        .and_then(|v| v.as_str())
    {
        entries.push((
            "tokenizer.chat_template".to_string(),
            GgufValue::String(t.into()),
        ));
    }

    Ok(VocabTable {
        entries,
        tokens: defined,
        padded,
        merges: merge_count,
        control,
        user_defined,
    })
}

#[cfg(test)]
mod vocab_tests {
    use super::*;

    fn write_tokenizer(dir: &Path, vocab: &[(&str, u64)], added: &[(&str, u64, bool)]) {
        let vocab_map: serde_json::Map<String, serde_json::Value> = vocab
            .iter()
            .map(|(c, id)| (c.to_string(), serde_json::json!(id)))
            .collect();
        let added_list: Vec<serde_json::Value> = added
            .iter()
            .map(|(c, id, s)| serde_json::json!({"content": c, "id": id, "special": s}))
            .collect();
        std::fs::write(
            dir.join("tokenizer.json"),
            serde_json::json!({
                "model": {"type": "BPE", "vocab": vocab_map, "merges": ["a b", ["b", "c"]]},
                "added_tokens": added_list,
            })
            .to_string(),
        )
        .unwrap();
    }

    /// The table is padded to the MODEL's vocabulary, and says so.
    #[test]
    fn tokens_pad_to_the_declared_vocabulary_and_types_follow_the_files() {
        let dir = tempfile::tempdir().unwrap();
        write_tokenizer(
            dir.path(),
            &[("a", 0), ("b", 1), ("c", 2)],
            &[("<eos>", 3, true), ("<fim>", 4, false)],
        );
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            serde_json::json!({"eos_token": "<eos>", "add_bos_token": false}).to_string(),
        )
        .unwrap();

        let table = qwen35_vocab(dir.path(), 8).unwrap();
        assert_eq!((table.tokens, table.padded), (5, 3));
        assert_eq!((table.control, table.user_defined), (1, 1));
        let get = |k: &str| {
            table
                .entries
                .iter()
                .find(|(key, _)| key == k)
                .map(|(_, v)| v.clone())
        };
        let GgufValue::Array(tokens) = get("tokenizer.ggml.tokens").unwrap() else {
            panic!()
        };
        assert_eq!(tokens.len(), 8);
        assert_eq!(tokens[4], GgufValue::String("<fim>".into()));
        assert_eq!(
            tokens[7],
            GgufValue::String("[PAD7]".into()),
            "the gap is explicit"
        );
        let GgufValue::Array(types) = get("tokenizer.ggml.token_type").unwrap() else {
            panic!()
        };
        assert_eq!(types[0], GgufValue::I32(1), "vocab tokens are NORMAL");
        assert_eq!(
            types[3],
            GgufValue::I32(3),
            "special added tokens are CONTROL"
        );
        assert_eq!(
            types[4],
            GgufValue::I32(4),
            "non-special added are USER_DEFINED"
        );
        assert_eq!(types[7], GgufValue::I32(5), "padding is UNUSED");
        // Merges: verbatim string, and the two-element spelling joined.
        let GgufValue::Array(merges) = get("tokenizer.ggml.merges").unwrap() else {
            panic!()
        };
        assert_eq!(merges[0], GgufValue::String("a b".into()));
        assert_eq!(merges[1], GgufValue::String("b c".into()));
        // The eos resolved through the named token, not a guess.
        assert_eq!(get("tokenizer.ggml.eos_token_id"), Some(GgufValue::U32(3)));
        assert_eq!(
            get("tokenizer.ggml.add_bos_token"),
            Some(GgufValue::Bool(false))
        );
    }

    /// A tokenizer that defines ids beyond the model's vocabulary is a
    /// tokenizer for a different model.
    #[test]
    fn an_id_beyond_the_vocabulary_refuses() {
        let dir = tempfile::tempdir().unwrap();
        write_tokenizer(dir.path(), &[("a", 0)], &[("<x>", 9, true)]);
        let err = qwen35_vocab(dir.path(), 4).unwrap_err().to_string();
        assert!(err.contains("different model"), "{err}");
    }

    /// No eos anywhere is a refusal, not a default.
    #[test]
    fn a_missing_eos_refuses_rather_than_guessing() {
        let dir = tempfile::tempdir().unwrap();
        write_tokenizer(dir.path(), &[("a", 0)], &[]);
        let err = qwen35_vocab(dir.path(), 2).unwrap_err().to_string();
        assert!(err.contains("eos"), "{err}");
        // And the generation config alone is enough to resolve it.
        std::fs::write(
            dir.path().join("generation_config.json"),
            serde_json::json!({"eos_token_id": [1, 0]}).to_string(),
        )
        .unwrap();
        let table = qwen35_vocab(dir.path(), 2).unwrap();
        let eos = table
            .entries
            .iter()
            .find(|(k, _)| k == "tokenizer.ggml.eos_token_id");
        assert_eq!(eos.map(|(_, v)| v.clone()), Some(GgufValue::U32(1)));
    }
}
