//! Special-token ids, `add_*_token` flags and the chat template.
//!
//! A port of `gguf.SpecialVocab` as the reference converter runs it for a
//! SentencePiece vocabulary (no merges): the post-processor's template
//! decides which specials a sequence is wrapped in, `tokenizer_config.json`
//! names the special tokens and overrides the flags, and the chat template
//! comes from `tokenizer_config.json`, else `chat_template.jinja`. Keys are
//! emitted in the order the reference discovers them, because GGUF keeps
//! insertion order and the file should read the same.

use std::path::Path;

use larql_models::loading::gguf::GgufValue;

use crate::VindexError;

/// `SpecialVocab.special_token_types`, in the order the reference visits
/// them.
const SPECIAL_TYPES: &[&str] = &["bos", "eos", "unk", "sep", "pad", "cls", "mask"];
/// llama.cpp's key spelling for each type's id.
fn id_key(typ: &str) -> String {
    let stem = match typ {
        "pad" => "padding",
        "unk" => "unknown",
        "sep" => "seperator",
        other => other,
    };
    format!("tokenizer.ggml.{stem}_token_id")
}
const CHAT_TEMPLATE_KEY: &str = "tokenizer.chat_template";
const CHAT_TEMPLATE_JINJA: &str = "chat_template.jinja";
const TEMPLATE_PROCESSING: &str = "TemplateProcessing";
const ROBERTA_PROCESSING: &str = "RobertaProcessing";
const SEQUENCE_A: &str = "A";
const SEQUENCE_B: &str = "B";

/// An insertion-ordered map, as Python's dict is.
#[derive(Debug, Default)]
struct Ordered<V>(Vec<(String, V)>);

impl<V> Ordered<V> {
    fn set(&mut self, key: &str, value: V) {
        match self.0.iter_mut().find(|(k, _)| k == key) {
            Some(slot) => slot.1 = value,
            None => self.0.push((key.to_string(), value)),
        }
    }
    fn contains(&self, key: &str) -> bool {
        self.0.iter().any(|(k, _)| k == key)
    }
}

fn special_id(item: &serde_json::Value) -> Option<&str> {
    item.get("SpecialToken")?.get("id")?.as_str()
}

fn sequence_id(item: &serde_json::Value) -> Option<&str> {
    item.get("Sequence")?.get("id")?.as_str()
}

/// Whether `id` equals one of the configured values, compared as JSON the
/// way the reference compares a string with whatever the config holds.
fn is_one_of(id: &str, values: &[&Option<serde_json::Value>]) -> bool {
    values
        .iter()
        .any(|v| v.as_ref().and_then(|v| v.as_str()) == Some(id))
}

/// The post-processor's effect on the `add_*` flags and on the config's
/// named tokens — `_try_load_from_tokenizer_json`'s processor branch.
fn apply_post_processor(
    post: &serde_json::Value,
    config: &mut serde_json::Map<String, serde_json::Value>,
    add: &mut Ordered<bool>,
) -> Result<(), VindexError> {
    let processors: Vec<&serde_json::Value> =
        match post.get("processors").and_then(|p| p.as_array()) {
            Some(list) => list.iter().collect(),
            None => vec![post],
        };
    // With no config at all, the template itself names bos and eos.
    let configured = !config.is_empty();
    for processor in processors {
        let get = |c: &serde_json::Map<String, serde_json::Value>, k: &str| c.get(k).cloned();
        let mut bos = get(config, "bos_token");
        let cls = get(config, "cls_token");
        let mut eos = get(config, "eos_token");
        let sep = get(config, "sep_token");
        match processor.get("type").and_then(|t| t.as_str()) {
            Some(ROBERTA_PROCESSING) => {
                for typ in ["bos", "eos", "sep"] {
                    add.set(typ, true);
                }
                if configured {
                    let first_of = |key: &str, fallback: Option<serde_json::Value>| {
                        processor
                            .get(key)
                            .and_then(|v| v.get(0))
                            .cloned()
                            .or(fallback)
                    };
                    if cls.is_none() {
                        if let Some(v) = first_of("cls", bos.clone()) {
                            config.insert("cls_token".into(), v);
                        }
                    }
                    if sep.is_none() {
                        if let Some(v) = first_of("sep", eos.clone()) {
                            config.insert("sep_token".into(), v);
                        }
                    }
                }
            }
            Some(TEMPLATE_PROCESSING) => {
                let empty = Vec::new();
                let single = processor["single"].as_array().unwrap_or(&empty);
                let pair = processor["pair"].as_array().unwrap_or(&empty);
                let mut first = None;
                let mut last = None;
                if single.len() > 1 {
                    first = special_id(&single[0]);
                    if let Some(f) = first {
                        if !configured {
                            bos = Some(serde_json::Value::String(f.into()));
                        }
                        add.set("bos", is_one_of(f, &[&bos, &cls]));
                    }
                    last = special_id(&single[single.len() - 1]);
                    if let Some(l) = last {
                        if !configured {
                            eos = Some(serde_json::Value::String(l.into()));
                        } else if !is_one_of(l, &[&eos]) {
                            // The reference re-homes the configured eos as
                            // an eot/eom token here; that fallback is not
                            // ported, so the export refuses rather than
                            // writing different special ids.
                            return Err(VindexError::Parse(format!(
                                "tokenizer.json's post-processor ends sequences with `{l}`, \
                                 which is not the configured eos — the reference's eot/eom \
                                 re-homing for this case is not implemented"
                            )));
                        }
                        add.set("eos", true);
                    }
                }
                if pair.is_empty() {
                    continue;
                }
                let start = usize::from(first.is_some() && special_id(&pair[0]) == first);
                let stop = if last.is_some() && special_id(&pair[pair.len() - 1]) == last {
                    pair.len() - 1
                } else {
                    pair.len()
                };
                let inner = &pair[start.min(stop)..stop];
                let (Some(a), Some(b)) = (inner.first(), inner.last()) else {
                    continue;
                };
                if inner.len() < 2
                    || sequence_id(a) != Some(SEQUENCE_A)
                    || sequence_id(b) != Some(SEQUENCE_B)
                {
                    continue;
                }
                let middle = &inner[1..inner.len() - 1];
                if middle.is_empty() {
                    continue;
                }
                let separates = |item: &serde_json::Value| {
                    special_id(item).is_some_and(|e| is_one_of(e, &[&sep, &eos]))
                };
                let add_sep = (separates(&middle[0]) && last.is_none())
                    || (middle.len() == 2 && separates(&middle[1]));
                add.set("sep", add_sep);
                if add_sep && sep.is_none() && configured {
                    if let Some(e) = eos {
                        config.insert("sep_token".into(), e);
                    }
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// The special-token entries for a vocabulary of `n_vocab` ids.
pub fn special_entries(
    container: &Path,
    tokenizer: &serde_json::Value,
    tokenizer_config: Option<&serde_json::Value>,
    n_vocab: usize,
) -> Result<Vec<(String, GgufValue)>, VindexError> {
    let mut add: Ordered<bool> = Ordered::default();
    let mut ids: Ordered<u32> = Ordered::default();
    let mut config = match tokenizer_config.cloned() {
        Some(serde_json::Value::Object(c)) => c,
        _ => serde_json::Map::new(),
    };
    // The reference stops after the post-processor when there is no
    // tokenizer config: no named ids, no template.
    let configured = !config.is_empty();
    if let Some(post) = tokenizer.get("post_processor").filter(|p| !p.is_null()) {
        apply_post_processor(post, &mut config, &mut add)?;
    }
    let empty = Vec::new();
    let added = tokenizer["added_tokens"].as_array().unwrap_or(&empty);
    for typ in SPECIAL_TYPES {
        if let Some(flag) = config
            .get(&format!("add_{typ}_token"))
            .and_then(|v| v.as_bool())
        {
            add.set(typ, flag);
        }
        let content = match config.get(&format!("{typ}_token")) {
            Some(serde_json::Value::String(s)) => Some(s.as_str()),
            Some(serde_json::Value::Object(o)) => o.get("content").and_then(|c| c.as_str()),
            _ => None,
        };
        let Some(content) = content else { continue };
        let id = added
            .iter()
            .find(|t| t["content"].as_str() == Some(content))
            .and_then(|t| t["id"].as_u64());
        if let Some(id) = id {
            if (id as usize) < n_vocab && !ids.contains(typ) {
                ids.set(typ, id as u32);
            }
        }
    }

    let mut out: Vec<(String, GgufValue)> = ids
        .0
        .into_iter()
        .map(|(typ, id)| (id_key(&typ), GgufValue::U32(id)))
        .collect();
    out.extend(add.0.into_iter().map(|(typ, v)| {
        (
            format!("tokenizer.ggml.add_{typ}_token"),
            GgufValue::Bool(v),
        )
    }));
    let template = match config.get("chat_template").filter(|_| configured) {
        Some(serde_json::Value::String(t)) => Some(t.clone()),
        Some(_) => None,
        None if configured => {
            let jinja = container.join(CHAT_TEMPLATE_JINJA);
            if jinja.exists() {
                Some(std::fs::read_to_string(jinja)?)
            } else {
                None
            }
        }
        None => None,
    };
    if let Some(t) = template {
        out.push((CHAT_TEMPLATE_KEY.into(), GgufValue::String(t)));
    }
    Ok(out)
}
