//! Every `tokenizer.ggml.*` key, and the container file that produces it.
//!
//! The same rule as the metadata table: **no literal unless it is a
//! target constant.** The tokens, merges, types and special ids all
//! come from the capability snapshot the container carries
//! (`tokenizer.json` and friends). The one constant is llama.cpp's
//! vocabulary-model name (`gpt2` — the byte-level BPE loader). The
//! pre-tokenizer id is looked up from the split regex the tokenizer
//! declares, in a table of llama.cpp's ids and the regex each one runs;
//! a regex the table does not hold is refused, because llama.cpp would
//! load the file and silently split text with a different regex.
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

/// llama.cpp pre-tokenizer ids, each with the split regex it runs,
/// spelled as HF declares it in `tokenizer.json` (llama.cpp's
/// `llama-vocab.cpp` quotes each one as its "original regex"). Qwen3.5
/// adds `\p{M}` (combining marks) to the letter classes, and llama.cpp
/// gives that its own id.
pub(crate) const VOCAB_PRE_BY_REGEX: &[(&str, &str)] = &[
    (
        "qwen2",
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    ),
    (
        "qwen35",
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    ),
];

/// The split regex a `tokenizer.json` `pre_tokenizer` declares: a bare
/// `Split`, or the `Split` inside a `Sequence`.
fn declared_split_regex(pre_tokenizer: &serde_json::Value) -> Option<&str> {
    fn split_regex(p: &serde_json::Value) -> Option<&str> {
        (p["type"] == "Split").then(|| p["pattern"]["Regex"].as_str())?
    }
    match pre_tokenizer["type"].as_str()? {
        "Split" => split_regex(pre_tokenizer),
        "Sequence" => pre_tokenizer["pretokenizers"]
            .as_array()?
            .iter()
            .find_map(split_regex),
        _ => None,
    }
}

/// llama.cpp's pre-tokenizer id for a `tokenizer.json` `pre_tokenizer`,
/// or a refusal when its split regex is not one llama.cpp names.
pub fn vocab_pre(pre_tokenizer: &serde_json::Value) -> Result<&'static str, VindexError> {
    let regex = declared_split_regex(pre_tokenizer).ok_or_else(|| {
        VindexError::Parse(
            "tokenizer.json: pre_tokenizer declares no split regex — llama.cpp's \
             pre-tokenizer id cannot be derived from it"
                .into(),
        )
    })?;
    VOCAB_PRE_BY_REGEX
        .iter()
        .find(|(_, known)| *known == regex)
        .map(|(id, _)| *id)
        .ok_or_else(|| {
            VindexError::Parse(format!(
                "tokenizer.json: split regex `{regex}` matches no llama.cpp pre-tokenizer \
                 this exporter knows — refusing rather than labelling it with another \
                 regex's id"
            ))
        })
}

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

    let pre = vocab_pre(&tokenizer["pre_tokenizer"])?;
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
            GgufValue::String(pre.into()),
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
mod tests;
