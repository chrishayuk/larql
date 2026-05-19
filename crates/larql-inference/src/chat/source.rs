//! Resolve a chat template from on-disk sources snapshotted into the
//! vindex by the extractor.
//!
//! HF has two conventions for where the chat template lives, and we
//! handle both:
//!
//! 1. **Standalone `.jinja` file** — `chat_template.jinja` next to
//!    `tokenizer.json`. Used by Gemma 4, Qwen3, and most 2025-era
//!    releases where the template is complex (macros, tool-call
//!    formatting) and doesn't round-trip cleanly through JSON escaping.
//! 2. **Embedded JSON string** — `tokenizer_config.json::chat_template`.
//!    The older convention used by Gemma 2/3, Llama-2-chat, Llama-3,
//!    Mistral-Instruct, etc. May be either a single string or an array
//!    of `{name, template}` entries when a model ships multiple
//!    templates (e.g. default vs. tool-use).
//!
//! The template *consumer* also needs the `tokenizer_config.json` for
//! `bos_token` / `eos_token` context values that templates reference, so
//! we always load it when present — even when the template itself comes
//! from the standalone `.jinja` file.

use std::path::Path;

use larql_vindex::format::filenames::TOKENIZER_CONFIG_JSON;
use serde_json::Value;

use super::render::render_chat_template;
use super::ChatWrap;

/// Resolve and render the HF-published template from the vindex.
///
/// Returns:
/// - `Ok(ChatWrap { applied: true, .. })` — template found and rendered.
/// - `Ok(ChatWrap { applied: false, .. })` — no template source in the
///   vindex; caller may try a hardcoded fallback.
/// - `Err(ChatWrap { applied: false, .. })` — template was found but
///   reading / parsing / rendering failed. Caller should still try
///   fallbacks; the note explains what broke.
pub(super) fn try_hf_template(vindex_dir: &Path, user_prompt: &str) -> Result<ChatWrap, ChatWrap> {
    let cfg = load_tokenizer_config(vindex_dir);

    // Source 1: standalone chat_template.jinja.
    let jinja_path = vindex_dir.join("chat_template.jinja");
    if jinja_path.is_file() {
        return match std::fs::read_to_string(&jinja_path) {
            Ok(template_str) => {
                finish_render(&template_str, &cfg, user_prompt, "chat_template.jinja")
            }
            Err(e) => Err(ChatWrap {
                prompt: user_prompt.to_string(),
                applied: false,
                note: format!("read chat_template.jinja failed: {e}"),
            }),
        };
    }

    // Source 2: chat_template field embedded in tokenizer_config.json.
    if let Some(template_str) = extract_chat_template_field(&cfg) {
        return finish_render(&template_str, &cfg, user_prompt, "tokenizer_config.json");
    }

    Ok(ChatWrap {
        prompt: user_prompt.to_string(),
        applied: false,
        note: "no chat_template.jinja and no chat_template in tokenizer_config.json".to_string(),
    })
}

/// Shared tail of both template-source branches: render the Jinja, tag the
/// `ChatWrap` with which source was used, upgrade render errors to `Err` so
/// the caller can still try hardcoded fallbacks.
fn finish_render(
    template_str: &str,
    cfg: &Value,
    user_prompt: &str,
    source_label: &str,
) -> Result<ChatWrap, ChatWrap> {
    match render_chat_template(template_str, cfg, user_prompt) {
        Ok(s) => Ok(ChatWrap {
            prompt: s,
            applied: true,
            note: format!("rendered from {source_label}"),
        }),
        Err(e) => {
            eprintln!("[chat] {source_label} render failed: {e}; trying fallbacks");
            Err(ChatWrap {
                prompt: user_prompt.to_string(),
                applied: false,
                note: format!("{source_label} render failed: {e}"),
            })
        }
    }
}

/// Read `tokenizer_config.json` into a `serde_json::Value`. Returns an
/// empty object on any failure (missing file, parse error) so downstream
/// rendering can continue without special-token context. Errors here are
/// non-fatal — many models ship without a config, and the template itself
/// might be purely self-contained.
pub(super) fn load_tokenizer_config(vindex_dir: &Path) -> Value {
    let path = vindex_dir.join(TOKENIZER_CONFIG_JSON);
    if !path.is_file() {
        return Value::Object(Default::default());
    }
    std::fs::read(&path)
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap_or_else(|| Value::Object(Default::default()))
}

/// Pull a `chat_template` value out of a parsed `tokenizer_config.json`.
/// HF ships it either as a single string, or (for models with multiple
/// templates like Llama-3) an array of `{name, template}` entries. We
/// prefer the `default`-named entry, falling back to the first entry's
/// template as a last resort.
pub(super) fn extract_chat_template_field(cfg: &Value) -> Option<String> {
    let v = cfg.get("chat_template")?;
    if let Some(s) = v.as_str() {
        return Some(s.to_string());
    }
    if let Some(arr) = v.as_array() {
        for entry in arr {
            if entry.get("name").and_then(|n| n.as_str()) == Some("default") {
                if let Some(s) = entry.get("template").and_then(|t| t.as_str()) {
                    return Some(s.to_string());
                }
            }
        }
        if let Some(first) = arr.first() {
            if let Some(s) = first.get("template").and_then(|t| t.as_str()) {
                return Some(s.to_string());
            }
        }
    }
    None
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    #[test]
    fn extract_prefers_default_in_array_form() {
        let cfg: Value = serde_json::from_str(
            r#"{"chat_template": [
                {"name": "tool_use", "template": "TOOL"},
                {"name": "default", "template": "DEFAULT"}
            ]}"#,
        )
        .unwrap();
        assert_eq!(
            extract_chat_template_field(&cfg).as_deref(),
            Some("DEFAULT")
        );
    }

    #[test]
    fn extract_falls_back_to_first_entry_when_no_default() {
        let cfg: Value =
            serde_json::from_str(r#"{"chat_template": [{"name": "rag", "template": "FIRST"}]}"#)
                .unwrap();
        assert_eq!(extract_chat_template_field(&cfg).as_deref(), Some("FIRST"));
    }

    #[test]
    fn extract_accepts_bare_string_form() {
        let cfg: Value = serde_json::from_str(r#"{"chat_template": "STR"}"#).unwrap();
        assert_eq!(extract_chat_template_field(&cfg).as_deref(), Some("STR"));
    }

    #[test]
    fn extract_none_when_missing() {
        let cfg: Value = serde_json::from_str(r#"{"bos_token": "<s>"}"#).unwrap();
        assert!(extract_chat_template_field(&cfg).is_none());
    }

    #[test]
    fn try_hf_template_passes_through_when_neither_source_exists() {
        let tmp = tempfile::tempdir().unwrap();
        let w = try_hf_template(tmp.path(), "hi").unwrap();
        assert!(!w.applied);
        assert!(w.note.contains("no chat_template.jinja"));
    }

    #[test]
    fn try_hf_template_reads_standalone_jinja_file() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            tmp.path().join("chat_template.jinja"),
            "{{ messages[0].content }}!",
        )
        .unwrap();
        let w = try_hf_template(tmp.path(), "hi").unwrap();
        assert!(w.applied);
        assert_eq!(w.prompt, "hi!");
        assert!(w.note.contains("chat_template.jinja"));
    }

    #[test]
    fn try_hf_template_reads_tokenizer_config_fallback() {
        // No standalone .jinja → should read from tokenizer_config.json.
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            tmp.path().join("tokenizer_config.json"),
            r#"{"chat_template": "tc:{{ messages[0].content }}"}"#,
        )
        .unwrap();
        let w = try_hf_template(tmp.path(), "hi").unwrap();
        assert!(w.applied);
        assert_eq!(w.prompt, "tc:hi");
        assert!(w.note.contains("tokenizer_config.json"));
    }

    #[test]
    fn render_error_produces_err_wrap() {
        let tmp = tempfile::tempdir().unwrap();
        // Intentionally invalid Jinja — bare `{%` with no closing tag.
        std::fs::write(tmp.path().join("chat_template.jinja"), "{% bogus").unwrap();
        let w = try_hf_template(tmp.path(), "hi").unwrap_err();
        assert!(!w.applied);
        assert!(
            w.note.contains("chat_template.jinja render failed"),
            "note={}",
            w.note
        );
    }
}
