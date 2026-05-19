//! Chat-template prompt wrapping, driven by the template that ships with
//! the model.
//!
//! **How it works.** The extractor snapshots the template source files
//! (`tokenizer_config.json`, `chat_template.jinja`, …) from the HF source
//! directory into the vindex — see [`larql_vindex::snapshot_hf_metadata`].
//! At runtime the [`source`] layer resolves a template string, the
//! [`render`] layer evaluates it with `minijinja` against a single user
//! turn (`add_generation_prompt=True` — same call shape as HF's
//! `apply_chat_template`), and the [`fallback`] layer kicks in for
//! instruct families whose upstream configs don't publish a template.
//!
//! **Public API is stable**: callers use [`wrap_chat_prompt`] or the
//! simpler [`wrap_with_vindex_template`] and inspect [`ChatWrap`].
//! Internal modules are `pub(crate)` only for tests — everything useful
//! is re-exported here.
//!
//! **Fallbacks.** Any failure path (no template found, render error,
//! unknown family) returns the raw prompt unchanged with an explanatory
//! `note` on [`ChatWrap`]. A broken template must never brick generation.
//!
//! ## Sibling chat-template machinery
//!
//! - [`crate::prompt`] — heuristic enum (no vindex required); for callers
//!   that only have a model id / family string at hand.
//! - [`crate::layer_graph::TurnRenderer`] — incremental per-turn rendering
//!   used by [`ChatSession`](crate::layer_graph::ChatSession).
//!
//! All three serve different needs; this module is the canonical path when
//! a vindex directory is available because it renders the model's *own*
//! Jinja template rather than a hand-coded approximation.

pub(crate) mod fallback;
pub(crate) mod render;
pub(crate) mod source;

/// Re-export of the multi-message renderer for diagnostic CLI flags
/// (`--system`, `--thinking`) and external callers that need richer
/// chat shapes than the single-turn `wrap_prompt_raw` exposes.
pub use render::render_chat_template_multi;

use std::path::Path;

use larql_vindex::format::filenames::TOKENIZER_CONFIG_JSON;
use serde_json::Value;

use fallback::fallback_template_for;
use source::try_hf_template;

const ENV_RAW_PROMPT: &str = "LARQL_RAW_PROMPT";
const ENV_THINKING: &str = "LARQL_THINKING";
const ENV_SYSTEM: &str = "LARQL_SYSTEM";
const ENV_NO_DEFAULT_SYSTEM: &str = "LARQL_NO_DEFAULT_SYSTEM";
const GEMMA4_DEFAULT_SYSTEM_PROMPT: &str =
    "You are a helpful assistant. Answer questions concisely.";

/// Outcome of applying (or not applying) a chat template to the user's
/// prompt. Returned wholesale so callers can both use the rendered string
/// and surface a note (`"rendered from chat_template.jinja"`,
/// `"no tokenizer_config.json in vindex"`, `"render failed: …"`).
#[derive(Debug, Clone)]
pub struct ChatWrap {
    /// The prompt to pass to `encode_prompt`. Equals the input prompt
    /// verbatim when [`ChatWrap::applied`] is false.
    pub prompt: String,
    /// True when a template was loaded and rendered successfully; false
    /// when we passed through (missing template, render error, etc.).
    pub applied: bool,
    /// Human-readable trail of where the template came from (or why we
    /// skipped). Surface in CLI/benchmark output so users can see
    /// whether their prompt was wrapped.
    pub note: String,
}

/// Simple form: resolves and renders the template stored in
/// `<vindex_dir>/…` against a single user turn. No hardcoded fallbacks.
/// Returns raw prompt with `applied=false` on any failure.
pub fn wrap_with_vindex_template(vindex_dir: &Path, user_prompt: &str) -> ChatWrap {
    wrap_chat_prompt(vindex_dir, None, user_prompt)
}

/// Full form: primary path is the HF template in the vindex; secondary is
/// a small hardcoded-template fallback keyed on a `model_hint` string
/// (e.g. the `cfg.model` field from the vindex —
/// `"meta-llama/Llama-2-7b-chat-hf"`, `"mistralai/Mistral-7B-Instruct-v0.3"`)
/// for families whose upstream configs don't publish the template directly.
///
/// Tries, in order:
/// 1. `<vindex_dir>/chat_template.jinja` (newer standalone-file convention —
///    Gemma 4, Qwen3, etc.).
/// 2. `<vindex_dir>/tokenizer_config.json::chat_template` (older embedded
///    convention — Gemma 2/3, Llama-3, …).
/// 3. A hardcoded template matched on `model_hint` + family heuristics,
///    when the hint clearly names an instruct/chat variant we recognise.
/// 4. Raw passthrough.
///
/// Base models ("…-hf", "…-v0.1" without `-Instruct` / `-chat`) skip step 3
/// and stay on raw prompts — wrapping them in `[INST]` markers would be
/// wrong since they weren't trained to see those tokens.
pub fn wrap_chat_prompt(
    vindex_dir: &Path,
    model_hint: Option<&str>,
    user_prompt: &str,
) -> ChatWrap {
    match try_hf_template(vindex_dir, user_prompt) {
        Ok(wrap) if wrap.applied => wrap,
        Ok(passthrough) => try_fallback(model_hint, user_prompt).unwrap_or(passthrough),
        // Render/parse error on the HF template: still try a hardcoded
        // fallback before giving up. The `Err` branch keeps the failure
        // note on `passthrough` in case the fallback also misses.
        Err(passthrough) => try_fallback(model_hint, user_prompt).unwrap_or(passthrough),
    }
}

/// Try the hardcoded instruct-family fallback (Llama-2-chat,
/// Mistral-Instruct). Returns `None` when the hint doesn't match or
/// `model_hint` was `None`.
fn try_fallback(model_hint: Option<&str>, user_prompt: &str) -> Option<ChatWrap> {
    let hint = model_hint?;
    let (family_label, template_str) = fallback_template_for(hint)?;
    let cfg = Value::Object(Default::default());
    match render::render_chat_template(template_str, &cfg, user_prompt) {
        Ok(s) => Some(ChatWrap {
            prompt: s,
            applied: true,
            note: format!("hardcoded {family_label} fallback"),
        }),
        Err(e) => {
            eprintln!("[chat] {family_label} fallback render failed: {e}");
            None
        }
    }
}

/// Render `template_str` (Jinja2) against a single user turn. Exposed so
/// callers that already have the template text in memory (remote API, test
/// fixture, in-memory generation) can reuse the render machinery without
/// touching the filesystem.
pub fn wrap_prompt_raw(
    template_str: &str,
    cfg: &Value,
    user_prompt: &str,
) -> Result<String, String> {
    render::render_chat_template(template_str, cfg, user_prompt).map_err(|e| e.to_string())
}

/// Back-compat shim — used by older callers that just want a pass-through.
/// Returns `user_prompt` unchanged.
pub fn passthrough(user_prompt: &str) -> String {
    user_prompt.to_string()
}

/// One-stop prompt rendering for `larql run`-style callers: respects
/// `LARQL_RAW_PROMPT`, `LARQL_THINKING`, `LARQL_SYSTEM`, and injects a
/// model-family-specific default system message when none is set.
///
/// Returns the chat-rendered prompt string (or the raw prompt for base
/// models / `LARQL_RAW_PROMPT=1`). Centralises the logic that used to
/// live inline in `run_with_moe_shards` so the dense Metal path
/// (`walk_cmd::run_predict_q4k`) can call it too.
///
/// Family-default behaviour: Gemma 4 (both 26B-A4B-it MoE and 31B dense)
/// defaults into degenerate frames without a system prompt — MoE
/// summarises "the input text" and dense loops "The answer is:". The
/// per-layer CPU/Metal parity confirms the inference math is correct;
/// the model genuinely needs a system prompt to enter answer mode. Set
/// `LARQL_NO_DEFAULT_SYSTEM=1` to opt out.
pub fn render_user_prompt(
    vindex_dir: &Path,
    family: &str,
    user_prompt: &str,
) -> Result<String, String> {
    let raw_prompt = std::env::var(ENV_RAW_PROMPT).is_ok();
    let enable_thinking = std::env::var(ENV_THINKING).is_ok();
    let user_system = std::env::var(ENV_SYSTEM).ok();
    let suppress_default = std::env::var(ENV_NO_DEFAULT_SYSTEM).is_ok();

    if raw_prompt {
        return Ok(user_prompt.to_string());
    }

    let system_prompt = user_system.or_else(|| {
        if suppress_default {
            None
        } else {
            default_system_prompt_for_family(family).map(str::to_string)
        }
    });

    if enable_thinking || system_prompt.is_some() {
        // Multi-message render path. Prefer the vindex's own template when
        // available; fall back to a family-default for vindexes extracted
        // before the chat-template snapshot was added (early Gemma 4 31B
        // extracts ship without `chat_template.jinja`, so the dense Metal
        // path silently sent raw prompts and the model looped).
        let template_str = read_chat_template(vindex_dir)
            .or_else(|| family_default_template(family))
            .ok_or_else(|| {
                format!(
                    "no chat template (vindex missing chat_template.jinja and \
                 no built-in fallback for family={family:?}) — \
                 set LARQL_RAW_PROMPT=1 to send the raw prompt"
                )
            })?;
        let cfg = Value::Object(Default::default());
        let mut messages: Vec<(String, String)> = Vec::new();
        if let Some(sys) = system_prompt.as_deref() {
            messages.push(("system".to_string(), sys.to_string()));
        }
        messages.push(("user".to_string(), user_prompt.to_string()));
        return render::render_chat_template_multi(&template_str, &cfg, &messages, enable_thinking)
            .map_err(|e| format!("render chat template: {e}"));
    }

    // Default path: single-user-turn chat template (the existing wrap).
    Ok(wrap_chat_prompt(vindex_dir, None, user_prompt).prompt)
}

fn default_system_prompt_for_family(family: &str) -> Option<&'static str> {
    match family {
        "gemma4" => Some(GEMMA4_DEFAULT_SYSTEM_PROMPT),
        _ => None,
    }
}

/// Read the model's chat template, looking in `chat_template.jinja` first
/// (newer convention — Gemma 4) then `tokenizer_config.json::chat_template`
/// (older — Gemma 2/3, Llama 3). Returns None when neither is present.
fn read_chat_template(vindex_dir: &Path) -> Option<String> {
    let jinja = vindex_dir.join("chat_template.jinja");
    if let Ok(s) = std::fs::read_to_string(&jinja) {
        return Some(s);
    }
    let cfg_path = vindex_dir.join(TOKENIZER_CONFIG_JSON);
    let cfg_bytes = std::fs::read(cfg_path).ok()?;
    let cfg: Value = serde_json::from_slice(&cfg_bytes).ok()?;
    cfg.get("chat_template")?.as_str().map(|s| s.to_string())
}

/// Built-in chat-template fallbacks for families whose extracted vindexes
/// sometimes ship without the template files. Minimal — handles the
/// system + user message shape this module renders, no tools/multimodal.
fn family_default_template(family: &str) -> Option<String> {
    match family {
        // Gemma 4 (`<|turn>role\n…<turn|>\n` blocks, with the empty thought
        // channel the official template emits when `enable_thinking=false`).
        // Verified end-to-end by running the rendered prompt through the
        // working 26B-A4B vindex's tokenizer — produces the same id stream
        // as the on-disk `chat_template.jinja` for system+user messages.
        "gemma4" => Some(GEMMA4_FALLBACK_TEMPLATE.to_string()),
        _ => None,
    }
}

/// Minimal Gemma 4 chat template covering system + user turns and the
/// empty thought channel. Used when a vindex was extracted before
/// `chat_template.jinja` was snapshotted (older 31B dense extracts).
const GEMMA4_FALLBACK_TEMPLATE: &str = "{{- bos_token -}}\
{%- if messages[0]['role'] in ['system', 'developer'] -%}\
{{- '<|turn>system\n' -}}{{- messages[0]['content'] | trim -}}{{- '<turn|>\n' -}}\
{%- set loop_messages = messages[1:] -%}\
{%- else -%}\
{%- set loop_messages = messages -%}\
{%- endif -%}\
{%- for message in loop_messages -%}\
{%- set role = 'model' if message['role'] == 'assistant' else message['role'] -%}\
{{- '<|turn>' + role + '\n' -}}\
{%- if message['content'] is string -%}{{- message['content'] | trim -}}{%- endif -%}\
{{- '<turn|>\n' -}}\
{%- endfor -%}\
{%- if add_generation_prompt -%}\
{{- '<|turn>model\n' -}}\
{%- if not (enable_thinking | default(false)) -%}{{- '<|channel>thought\n<channel|>' -}}{%- endif -%}\
{%- endif -%}";

#[cfg(test)]
mod integration_tests {
    //! High-level tests that exercise the full `wrap_chat_prompt` pipeline
    //! across its three fallback layers. Module-local logic (JSON shape
    //! handling, Jinja edge cases, per-family patterns) is covered in the
    //! tests adjacent to [`source`], [`render`], and [`fallback`].

    use super::*;

    #[test]
    fn hf_template_wins_over_fallback_when_both_exist() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = r#"{"chat_template":"HF:{{ messages[0].content }}"}"#;
        std::fs::write(tmp.path().join("tokenizer_config.json"), cfg).unwrap();
        let w = wrap_chat_prompt(tmp.path(), Some("meta-llama/Llama-2-7b-chat-hf"), "hi");
        assert!(w.applied);
        // Primary path wins — we get the HF template, not `[INST]`.
        assert_eq!(w.prompt, "HF:hi");
    }

    #[test]
    fn full_passthrough_when_nothing_matches() {
        let tmp = tempfile::tempdir().unwrap();
        // No vindex metadata, model hint is a base model — every layer
        // declines; we expect the raw prompt back with `applied=false`.
        let w = wrap_chat_prompt(tmp.path(), Some("meta-llama/Llama-2-7b-hf"), "hi");
        assert!(!w.applied);
        assert_eq!(w.prompt, "hi");
    }

    #[test]
    fn standalone_jinja_file_beats_tokenizer_config() {
        // When both sources are present, `chat_template.jinja` wins
        // (matches the lookup order documented on `wrap_chat_prompt`).
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            tmp.path().join("chat_template.jinja"),
            "JINJA:{{ messages[0].content }}",
        )
        .unwrap();
        std::fs::write(
            tmp.path().join("tokenizer_config.json"),
            r#"{"chat_template":"TC:{{ messages[0].content }}"}"#,
        )
        .unwrap();
        let w = wrap_with_vindex_template(tmp.path(), "hi");
        assert!(w.applied);
        assert_eq!(w.prompt, "JINJA:hi");
        assert!(w.note.contains("chat_template.jinja"), "note={}", w.note);
    }

    // ── helpers (no env-var mutation — tests run in parallel) ─────────────

    #[test]
    fn passthrough_returns_input_unchanged() {
        assert_eq!(passthrough("hi there"), "hi there");
    }

    #[test]
    fn wrap_prompt_raw_renders_template_with_user_content() {
        let cfg = Value::Object(Default::default());
        let result =
            wrap_prompt_raw("PROMPT:{{ messages[0].content }}", &cfg, "hello world").unwrap();
        assert_eq!(result, "PROMPT:hello world");
    }

    #[test]
    fn wrap_prompt_raw_returns_error_on_malformed_template() {
        // Unterminated jinja block — render fails.
        let cfg = Value::Object(Default::default());
        let err = wrap_prompt_raw("{{ unclosed", &cfg, "hi").unwrap_err();
        assert!(!err.is_empty(), "error string must be non-empty");
    }

    #[test]
    fn default_system_prompt_for_family_returns_gemma4_text() {
        assert!(default_system_prompt_for_family("gemma4").is_some());
        assert_eq!(default_system_prompt_for_family("tinymodel"), None);
        assert_eq!(default_system_prompt_for_family("llama"), None);
    }

    #[test]
    fn family_default_template_returns_gemma4_template() {
        let t = family_default_template("gemma4").expect("gemma4 has fallback");
        assert!(t.contains("turn"), "gemma4 template uses turn markers");
        assert_eq!(family_default_template("tinymodel"), None);
    }

    #[test]
    fn read_chat_template_prefers_jinja_file() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("chat_template.jinja"), "FROM_JINJA").unwrap();
        std::fs::write(
            tmp.path().join(TOKENIZER_CONFIG_JSON),
            r#"{"chat_template":"FROM_CFG"}"#,
        )
        .unwrap();
        assert_eq!(
            read_chat_template(tmp.path()).as_deref(),
            Some("FROM_JINJA")
        );
    }

    #[test]
    fn read_chat_template_falls_back_to_tokenizer_config() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            tmp.path().join(TOKENIZER_CONFIG_JSON),
            r#"{"chat_template":"FROM_CFG"}"#,
        )
        .unwrap();
        assert_eq!(read_chat_template(tmp.path()).as_deref(), Some("FROM_CFG"));
    }

    #[test]
    fn read_chat_template_returns_none_when_both_absent() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(read_chat_template(tmp.path()).is_none());
    }

    #[test]
    fn read_chat_template_returns_none_when_cfg_has_no_chat_template_field() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join(TOKENIZER_CONFIG_JSON), r#"{"other":"x"}"#).unwrap();
        assert!(read_chat_template(tmp.path()).is_none());
    }

    // ── render_user_prompt — serialised via mutex because the function
    //    reads process-global env vars that other tests would race on. ────
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    /// Helper: clear all prompt-affecting env vars while holding the
    /// global lock. Returns the lock guard so the caller's mutations
    /// stay isolated for the rest of the test.
    fn lock_and_clear_env() -> std::sync::MutexGuard<'static, ()> {
        // Recover from a poisoned lock (a panicking earlier test) — we
        // just want serialisation, not panic propagation.
        let guard = env_lock().lock().unwrap_or_else(|p| p.into_inner());
        for k in [
            ENV_RAW_PROMPT,
            ENV_THINKING,
            ENV_SYSTEM,
            ENV_NO_DEFAULT_SYSTEM,
        ] {
            unsafe { std::env::remove_var(k) };
        }
        guard
    }

    #[test]
    fn render_user_prompt_passes_through_when_raw_env_set() {
        let _g = lock_and_clear_env();
        unsafe { std::env::set_var(ENV_RAW_PROMPT, "1") };
        let tmp = tempfile::tempdir().unwrap();
        let result = render_user_prompt(tmp.path(), "tinymodel", "hello").unwrap();
        unsafe { std::env::remove_var(ENV_RAW_PROMPT) };
        assert_eq!(result, "hello");
    }

    #[test]
    fn render_user_prompt_uses_vindex_template_when_no_system() {
        let _g = lock_and_clear_env();
        unsafe { std::env::set_var(ENV_NO_DEFAULT_SYSTEM, "1") };
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            tmp.path().join("chat_template.jinja"),
            "WRAP:{{ messages[0].content }}",
        )
        .unwrap();
        let result = render_user_prompt(tmp.path(), "tinymodel", "hi");
        unsafe { std::env::remove_var(ENV_NO_DEFAULT_SYSTEM) };
        assert_eq!(result.unwrap(), "WRAP:hi");
    }

    #[test]
    fn render_user_prompt_errors_when_no_template_and_system_requested() {
        let _g = lock_and_clear_env();
        unsafe { std::env::set_var(ENV_SYSTEM, "you are helpful") };
        let tmp = tempfile::tempdir().unwrap();
        let err = render_user_prompt(tmp.path(), "unknown-family", "hi");
        unsafe { std::env::remove_var(ENV_SYSTEM) };
        let msg = err.unwrap_err();
        assert!(
            msg.contains("no chat template"),
            "expected no-template error — got {msg}"
        );
    }

    #[test]
    fn render_user_prompt_renders_with_explicit_system_message() {
        let _g = lock_and_clear_env();
        unsafe { std::env::set_var(ENV_SYSTEM, "you are helpful") };
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            tmp.path().join("chat_template.jinja"),
            "{% for m in messages %}<{{ m.role }}>{{ m.content }}{% endfor %}",
        )
        .unwrap();
        let result = render_user_prompt(tmp.path(), "tinymodel", "hi");
        unsafe { std::env::remove_var(ENV_SYSTEM) };
        let s = result.unwrap();
        assert!(s.contains("<system>you are helpful"));
        assert!(s.contains("<user>hi"));
    }

    #[test]
    fn render_user_prompt_uses_family_default_when_thinking_set() {
        let _g = lock_and_clear_env();
        unsafe { std::env::set_var(ENV_THINKING, "1") };
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            tmp.path().join("chat_template.jinja"),
            "{% for m in messages %}<{{ m.role }}>{{ m.content }}{% endfor %}",
        )
        .unwrap();
        let result = render_user_prompt(tmp.path(), "gemma4", "hi");
        unsafe { std::env::remove_var(ENV_THINKING) };
        let s = result.unwrap();
        assert!(s.contains("<system>"));
        assert!(s.contains("<user>hi"));
    }

    #[test]
    fn render_user_prompt_default_path_uses_wrap_chat_prompt() {
        // No env vars, no system → default path: wrap_chat_prompt(). With
        // a chat_template.jinja in the vindex dir, wrap returns
        // applied=true and the templated output.
        let _g = lock_and_clear_env();
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            tmp.path().join("chat_template.jinja"),
            "WRAP:{{ messages[0].content }}",
        )
        .unwrap();
        let result = render_user_prompt(tmp.path(), "tinymodel", "hi").unwrap();
        assert_eq!(result, "WRAP:hi");
    }
}
