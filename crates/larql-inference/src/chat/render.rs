//! Jinja2 template rendering for chat prompts.
//!
//! HF chat templates are standard Jinja2 with a couple of Python-flavoured
//! conveniences: `.get(k)`/`.items()`/`.startswith(s)` on maps and strings,
//! and host-provided functions like `raise_exception(msg)` and
//! `strftime_now("%Y-%m-%d")`. This module sets up a `minijinja::Environment`
//! with the same surface so templates written against HF Python render
//! unchanged — no per-template patching.
//!
//! Input shape mirrors HF's `tokenizer.apply_chat_template(..., add_generation_prompt=True)`:
//! `messages=[{role, content}]`, `add_generation_prompt=true`, plus
//! `bos_token` / `eos_token` from the tokenizer config. One user turn
//! only — multi-turn rendering can be built on top but isn't needed for
//! the one-shot prompt path.

use minijinja::{context, Environment};
use serde_json::Value;

/// Render `template_str` (Jinja2) against a single-turn conversation.
/// Returns the rendered string or a `minijinja::Error` with full diagnostic
/// info (line/column, template frame).
pub(crate) fn render_chat_template(
    template_str: &str,
    cfg: &Value,
    user_prompt: &str,
) -> Result<String, minijinja::Error> {
    let env = build_env(template_str)?;
    let tmpl = env.get_template("chat")?;
    let ctx = build_context(cfg, user_prompt);
    tmpl.render(ctx)
}

/// Render `template_str` against an arbitrary multi-message conversation
/// plus optional `enable_thinking` flag.  Used by the CLI's diagnostic
/// `--system` / thinking flags so callers can inject a system prompt or
/// flip the thinking-channel default without forking the env setup
/// (which the bare `wrap_prompt_raw` API doesn't expose — it hard-codes
/// a single user turn).
///
/// `messages` is a list of `(role, content)` pairs; roles are passed
/// through to the template verbatim ("system", "user", "assistant",
/// "model" — pick what your model's template recognises).
pub fn render_chat_template_multi(
    template_str: &str,
    cfg: &Value,
    messages: &[(String, String)],
    enable_thinking: bool,
) -> Result<String, String> {
    let env = build_env(template_str).map_err(|e| e.to_string())?;
    let tmpl = env.get_template("chat").map_err(|e| e.to_string())?;
    let bos_token = cfg_string_field(cfg, "bos_token").unwrap_or_default();
    let eos_token = cfg_string_field(cfg, "eos_token").unwrap_or_default();
    let msgs: Vec<minijinja::Value> = messages
        .iter()
        .map(|(role, content)| context! { role => role.clone(), content => content.clone() })
        .collect();
    let ctx = context! {
        messages => msgs,
        add_generation_prompt => true,
        enable_thinking => enable_thinking,
        bos_token => bos_token,
        eos_token => eos_token,
    };
    tmpl.render(ctx).map_err(|e| e.to_string())
}

/// Assemble the minijinja environment with all HF-compat shims attached.
/// Factored out so tests can poke at individual shims in isolation.
fn build_env(template_str: &str) -> Result<Environment<'static>, minijinja::Error> {
    let mut env = Environment::new();

    // Python-style method compat: HF templates frequently call
    // `.get(key)`, `.items()`, `.startswith(s)` etc. on dict / string
    // values. minijinja treats those as unknown methods by default; the
    // contrib crate's `pycompat::unknown_method_callback` implements them
    // against minijinja's native filter/value machinery. Gemma 4's
    // 347-line template needs this for `tool_body.get('type')` and
    // friends; Qwen3 and Llama-3 also use `.startswith(...)`.
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);

    // `raise_exception(msg)` — HF templates use this to reject malformed
    // conversations (e.g. tool-use template when `tools` arg is missing).
    // Map it to a rendering-time error so the template fails cleanly.
    env.add_function(
        "raise_exception",
        |msg: String| -> Result<String, minijinja::Error> {
            Err(minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                msg,
            ))
        },
    );

    // `strftime_now(fmt)` — Llama-3, Qwen, some DeepSeek variants inline
    // the current date in a system message. We return an empty string to
    // keep rendering deterministic; a richer runtime can override this.
    env.add_function("strftime_now", |_fmt: String| -> String { String::new() });

    // Compile the template. Wrap syntax errors so the outer `get_template`
    // call surfaces a useful diagnostic instead of a bare `TemplateNotFound`.
    let template_owned = template_str.to_string();
    env.add_template_owned("chat", template_owned)
        .map_err(|e| minijinja::Error::new(minijinja::ErrorKind::SyntaxError, e.to_string()))?;
    Ok(env)
}

/// Build the minijinja context for a single-turn user→model conversation.
/// Mirrors HF's `apply_chat_template(messages, add_generation_prompt=True)`.
fn build_context(cfg: &Value, user_prompt: &str) -> minijinja::Value {
    let bos_token = cfg_string_field(cfg, "bos_token").unwrap_or_default();
    let eos_token = cfg_string_field(cfg, "eos_token").unwrap_or_default();

    context! {
        messages => vec![
            context! { role => "user", content => user_prompt },
        ],
        add_generation_prompt => true,
        bos_token => bos_token,
        eos_token => eos_token,
    }
}

/// Read a tokenizer_config field that may be either a plain string or a
/// `{content: "…"}` object — HF wraps some special-token metadata this way.
fn cfg_string_field(cfg: &Value, key: &str) -> Option<String> {
    let v = cfg.get(key)?;
    if let Some(s) = v.as_str() {
        return Some(s.to_string());
    }
    v.as_object()?
        .get("content")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    fn empty_cfg() -> Value {
        Value::Object(Default::default())
    }

    #[test]
    fn renders_basic_single_turn_template() {
        let tmpl = "{{ messages[0].content }}!";
        let out = render_chat_template(tmpl, &empty_cfg(), "hi").unwrap();
        assert_eq!(out, "hi!");
    }

    #[test]
    fn passes_bos_and_eos_from_config() {
        let cfg: Value =
            serde_json::from_str(r#"{"bos_token": "<s>", "eos_token": "</s>"}"#).unwrap();
        let tmpl = "{{ bos_token }}/{{ eos_token }}/{{ messages[0].content }}";
        let out = render_chat_template(tmpl, &cfg, "x").unwrap();
        assert_eq!(out, "<s>/</s>/x");
    }

    #[test]
    fn unwraps_object_form_special_tokens() {
        // HF sometimes serializes bos_token as {"content": "<bos>", ...}.
        let cfg: Value =
            serde_json::from_str(r#"{"bos_token": {"content": "<bos>", "lstrip": false}}"#)
                .unwrap();
        let tmpl = "{{ bos_token }}|{{ messages[0].content }}";
        let out = render_chat_template(tmpl, &cfg, "hi").unwrap();
        assert_eq!(out, "<bos>|hi");
    }

    #[test]
    fn pycompat_dot_get_on_map_works() {
        // Gemma 4's template calls `.get('type')` on tool-body maps.
        // Without the pycompat shim this raises `UnknownMethod`.
        let tmpl = "{{ messages[0].get('content') }}!";
        let out = render_chat_template(tmpl, &empty_cfg(), "via-get").unwrap();
        assert_eq!(out, "via-get!");
    }

    #[test]
    fn pycompat_startswith_on_string_works() {
        let tmpl = "{% if messages[0]['content'].startswith('hi') %}yes{% else %}no{% endif %}";
        assert_eq!(
            render_chat_template(tmpl, &empty_cfg(), "hi there").unwrap(),
            "yes"
        );
        assert_eq!(
            render_chat_template(tmpl, &empty_cfg(), "bye").unwrap(),
            "no"
        );
    }

    #[test]
    fn raise_exception_propagates_as_error() {
        let tmpl = "{{ raise_exception('nope') }}";
        let err = render_chat_template(tmpl, &empty_cfg(), "x").unwrap_err();
        assert!(err.to_string().contains("nope"), "err={err}");
    }

    #[test]
    fn strftime_now_stub_returns_empty() {
        let tmpl = "[{{ strftime_now('%Y-%m-%d') }}]:{{ messages[0]['content'] }}";
        let out = render_chat_template(tmpl, &empty_cfg(), "x").unwrap();
        assert_eq!(out, "[]:x");
    }

    #[test]
    fn add_generation_prompt_is_true() {
        let tmpl = "{% if add_generation_prompt %}ON{% else %}OFF{% endif %}";
        assert_eq!(render_chat_template(tmpl, &empty_cfg(), "x").unwrap(), "ON");
    }

    #[test]
    fn syntax_error_surfaces_at_compile_time() {
        // Open `{%` with no closing tag — minijinja should flag this at
        // `add_template_owned` time, surfaced as a SyntaxError by
        // `build_env`.
        let err = render_chat_template("{% for x in", &empty_cfg(), "x").unwrap_err();
        assert!(err.to_string().contains("syntax"), "err={err}");
    }

    // ── render_chat_template_multi ────────────────────────────────────────────

    fn msgs(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs
            .iter()
            .map(|(r, c)| ((*r).to_owned(), (*c).to_owned()))
            .collect()
    }

    #[test]
    fn multi_renders_all_messages_in_order() {
        let tmpl = "{% for m in messages %}[{{ m.role }}:{{ m.content }}]{% endfor %}\
             {% if add_generation_prompt %}{{ '<go>' }}{% endif %}";
        let out = render_chat_template_multi(
            tmpl,
            &empty_cfg(),
            &msgs(&[("system", "S"), ("user", "U"), ("assistant", "A")]),
            false,
        )
        .unwrap();
        assert_eq!(out, "[system:S][user:U][assistant:A]<go>");
    }

    #[test]
    fn multi_threads_enable_thinking_flag() {
        let tmpl = "{% if enable_thinking %}think{% else %}plain{% endif %}";
        let on =
            render_chat_template_multi(tmpl, &empty_cfg(), &msgs(&[("user", "x")]), true).unwrap();
        let off =
            render_chat_template_multi(tmpl, &empty_cfg(), &msgs(&[("user", "x")]), false).unwrap();
        assert_eq!(on, "think");
        assert_eq!(off, "plain");
    }

    #[test]
    fn multi_passes_bos_eos_from_cfg() {
        let cfg: Value =
            serde_json::from_str(r#"{"bos_token": "<s>", "eos_token": "</s>"}"#).unwrap();
        let tmpl = "{{ bos_token }}/{{ messages[0].content }}/{{ eos_token }}";
        let out = render_chat_template_multi(tmpl, &cfg, &msgs(&[("user", "U")]), false).unwrap();
        assert_eq!(out, "<s>/U/</s>");
    }

    #[test]
    fn multi_surfaces_syntax_errors_as_string() {
        let err =
            render_chat_template_multi("{% for", &empty_cfg(), &msgs(&[("user", "x")]), false)
                .unwrap_err();
        assert!(err.contains("syntax"), "err={err}");
    }

    #[test]
    fn multi_surfaces_render_error_as_string() {
        // raise_exception → ErrorKind::InvalidOperation, surfaces via
        // `tmpl.render(..).map_err(|e| e.to_string())`.
        let err = render_chat_template_multi(
            "{{ raise_exception('boom') }}",
            &empty_cfg(),
            &msgs(&[("user", "x")]),
            false,
        )
        .unwrap_err();
        assert!(err.contains("boom"), "err={err}");
    }

    // ── cfg_string_field ──────────────────────────────────────────────────────

    #[test]
    fn cfg_string_field_returns_none_for_array() {
        // Neither string nor object: as_str() fails, then as_object() fails.
        let cfg: Value = serde_json::from_str(r#"{"bos_token": ["a", "b"]}"#).unwrap();
        let tmpl = "[{{ bos_token }}]";
        let out = render_chat_template(tmpl, &cfg, "x").unwrap();
        // Empty fallback (`unwrap_or_default()`).
        assert_eq!(out, "[]");
    }

    #[test]
    fn cfg_string_field_returns_none_when_object_missing_content() {
        let cfg: Value = serde_json::from_str(r#"{"bos_token": {"lstrip": false}}"#).unwrap();
        let tmpl = "[{{ bos_token }}]";
        let out = render_chat_template(tmpl, &cfg, "x").unwrap();
        assert_eq!(out, "[]");
    }
}
