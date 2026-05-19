//! Model-agnostic chat-template helpers (heuristic, no vindex required).
//!
//! Wraps a raw user prompt in the instruction-tuning format the target model
//! was trained on. Selecting the wrong template produces garbage output, so
//! detection is conservative: known families map to their canonical template,
//! everything else falls back to [`ChatTemplate::Plain`] (pass-through).
//!
//! Resolution precedence used by [`ChatTemplate::for_model_id`]:
//!
//!   1. Substring match on common family tokens (`gemma`, `mistral`, …).
//!   2. Fallback to `Plain` (no wrapping).
//!
//! When you have a loaded model, prefer [`ChatTemplate::for_family`] with the
//! string returned by [`larql_models::ModelArchitecture::family`] — that's the
//! authoritative signal.
//!
//! ## Sibling chat-template machinery
//!
//! Three modules share this space; they are complementary, not redundant:
//!
//! - **This module ([`crate::prompt`])** — heuristic enum + hardcoded format
//!   strings. Use when you only have a model id / family string and no vindex
//!   bytes available (e.g. an OpenAI-shaped HTTP request whose model id picks
//!   the renderer before any weights are loaded).
//! - **[`crate::chat`]** — Jinja-driven rendering against a vindex's
//!   `chat_template.jinja` / `tokenizer_config.json::chat_template`. The
//!   canonical, model-faithful path when the vindex directory is on disk.
//! - **[`crate::layer_graph::TurnRenderer`]** — incremental per-turn rendering
//!   for the multi-turn [`ChatSession`](crate::layer_graph::ChatSession) token
//!   buffer. [`ChatTemplate::render_messages`] delegates to these renderers
//!   for Gemma / Llama / ChatML rather than re-implementing the format
//!   strings, so the per-turn shape stays consistent across the two stacks.
//!
//! [`ChatTemplate::wrap`] still owns its own single-prompt format strings
//! (slightly different shape — adds the assistant-open marker). Keep them
//! aligned with the [`TurnRenderer`](crate::layer_graph::TurnRenderer) impls
//! when adjusting either side.

/// Chat-template format for instruction-tuned models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatTemplate {
    /// Gemma 2/3/4 turn format:
    /// `<start_of_turn>user\n{}\n<end_of_turn>\n<start_of_turn>model\n`
    Gemma,
    /// Mistral / Mixtral instruction format: `[INST] {} [/INST]`.
    ///
    /// The official HF template prefixes `<s>`, but `encode_prompt` calls the
    /// tokenizer with `add_special_tokens=true`, which already prepends BOS.
    /// Including a literal `<s>` here risks double-BOS depending on tokenizer
    /// post-processor behavior, so we omit it.
    Mistral,
    /// Llama 3 chat format using `<|begin_of_text|>` and header tags.
    Llama,
    /// ChatML used by Qwen, DeepSeek, and others:
    /// `<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n`.
    ChatML,
    /// Pass-through — returns the prompt verbatim. Default for unknown
    /// families and base (non-instruct) models.
    Plain,
}

impl ChatTemplate {
    /// Heuristic resolution from a model id like `"google/gemma-3-4b-it"`.
    ///
    /// Use this when you only have a string identifier (CLI flag, HF id).
    /// When a `ModelArchitecture` is in scope, prefer [`Self::for_family`].
    pub fn for_model_id(model_id: &str) -> Self {
        let id = model_id.to_ascii_lowercase();
        // Order matters: more-specific patterns first.
        if id.contains("gemma") {
            Self::Gemma
        } else if id.contains("mixtral") || id.contains("mistral") {
            Self::Mistral
        } else if id.contains("llama") {
            Self::Llama
        } else if id.contains("qwen") || id.contains("deepseek") || id.contains("chatml") {
            Self::ChatML
        } else {
            Self::Plain
        }
    }

    /// Resolution from a model-architecture family string (the value returned
    /// by `ModelArchitecture::family()`).
    ///
    /// Recognised values: `gemma2`, `gemma3`, `gemma4`, `mistral`, `mixtral`,
    /// `llama`, `qwen`, `qwen2`, `qwen3`, `deepseek`, `gpt_oss`. Anything else
    /// (`generic`, `tinymodel`, `starcoder2`, …) falls back to `Plain`.
    pub fn for_family(family: &str) -> Self {
        match family {
            "gemma2" | "gemma3" | "gemma4" => Self::Gemma,
            "mistral" | "mixtral" => Self::Mistral,
            "llama" => Self::Llama,
            "qwen" | "qwen2" | "qwen3" | "deepseek" | "gpt_oss" => Self::ChatML,
            _ => Self::Plain,
        }
    }

    /// Wrap `user_prompt` in the template. The output is ready to feed to the
    /// tokenizer — no extra BOS handling needed (templates that need a leading
    /// `<s>` include it).
    pub fn wrap(&self, user_prompt: &str) -> String {
        match self {
            Self::Gemma => {
                format!("<start_of_turn>user\n{user_prompt}\n<end_of_turn>\n<start_of_turn>model\n")
            }
            Self::Mistral => format!("[INST] {user_prompt} [/INST]"),
            Self::Llama => format!(
                "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n\
                 {user_prompt}<|eot_id|>\
                 <|start_header_id|>assistant<|end_header_id|>\n\n"
            ),
            Self::ChatML => {
                format!("<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n")
            }
            Self::Plain => user_prompt.to_string(),
        }
    }

    /// Short identifier suitable for logging.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Gemma => "gemma",
            Self::Mistral => "mistral",
            Self::Llama => "llama",
            Self::ChatML => "chatml",
            Self::Plain => "plain",
        }
    }

    /// Render a multi-turn message list (OpenAI chat-completions shape)
    /// into a single prompt ready for the tokenizer. Always appends the
    /// assistant-open marker so the model continues the conversation.
    ///
    /// Roles: `"system"`, `"user"`, `"assistant"`. Unknown roles are
    /// emitted verbatim by the renderer (Gemma/ChatML/Llama) or surfaced
    /// as a generic label by `Plain`/`Mistral`.
    pub fn render_messages<I, R, C>(&self, messages: I) -> String
    where
        I: IntoIterator<Item = (R, C)>,
        R: AsRef<str>,
        C: AsRef<str>,
    {
        let messages: Vec<(String, String)> = messages
            .into_iter()
            .map(|(r, c)| (r.as_ref().to_owned(), c.as_ref().to_owned()))
            .collect();
        match self {
            Self::Gemma => render_via_renderer(&crate::layer_graph::GemmaRenderer, &messages),
            Self::Llama => render_via_renderer(&crate::layer_graph::Llama3Renderer, &messages),
            Self::ChatML => render_via_renderer(&crate::layer_graph::ChatMLRenderer, &messages),
            Self::Mistral => render_mistral(&messages),
            Self::Plain => render_plain(&messages),
        }
    }
}

/// Generic multi-turn rendering for any `TurnRenderer`.
fn render_via_renderer<R: crate::layer_graph::TurnRenderer>(
    renderer: &R,
    messages: &[(String, String)],
) -> String {
    let mut out = String::new();
    for (role, content) in messages {
        let role = role.as_str();
        let role = if role == "assistant" {
            // Some renderers (Gemma) use "model" instead of "assistant".
            // The renderer trait's `render` already handles this case.
            "assistant"
        } else {
            role
        };
        out.push_str(&renderer.render(role, content));
    }
    out.push_str(&renderer.assistant_open());
    out
}

/// Mistral / Mixtral: `[INST] {user} [/INST] {assistant}` with system
/// prompts prepended to the first user turn.
fn render_mistral(messages: &[(String, String)]) -> String {
    let mut out = String::new();
    let mut pending_system: Vec<String> = Vec::new();
    let mut i = 0;
    while i < messages.len() {
        let (role, content) = &messages[i];
        match role.as_str() {
            "system" => {
                pending_system.push(content.clone());
                i += 1;
            }
            "user" => {
                let prefix = if pending_system.is_empty() {
                    String::new()
                } else {
                    let p = pending_system.join("\n") + "\n\n";
                    pending_system.clear();
                    p
                };
                out.push_str(&format!("[INST] {prefix}{content} [/INST]"));
                i += 1;
                if let Some((next_role, next_content)) = messages.get(i) {
                    if next_role == "assistant" {
                        out.push_str(&format!(" {next_content} "));
                        i += 1;
                    }
                }
            }
            "assistant" => {
                out.push_str(&format!(" {content} "));
                i += 1;
            }
            _ => i += 1,
        }
    }
    if !pending_system.is_empty() {
        out.push_str(&format!("[INST] {} [/INST]", pending_system.join("\n")));
    }
    out
}

/// Plain template — base / non-instruct models. Concatenates messages
/// with `User:` / `Assistant:` / `System:` markers, ending with an
/// `Assistant:` open so the model continues. Not great for instruct
/// behaviour, but better than dropping system prompts.
fn render_plain(messages: &[(String, String)]) -> String {
    let mut out = String::new();
    for (role, content) in messages {
        let label = match role.as_str() {
            "user" => "User",
            "assistant" => "Assistant",
            "system" => "System",
            other => other,
        };
        out.push_str(&format!("{label}: {content}\n"));
    }
    out.push_str("Assistant: ");
    out
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod render_messages_tests {
    use super::*;

    fn msgs(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs
            .iter()
            .map(|(r, c)| ((*r).to_owned(), (*c).to_owned()))
            .collect()
    }

    #[test]
    fn gemma_multi_turn_includes_model_open() {
        let out = ChatTemplate::Gemma.render_messages(msgs(&[
            ("user", "hi"),
            ("assistant", "hello"),
            ("user", "more"),
        ]));
        assert!(out.contains("<start_of_turn>user\nhi<end_of_turn>"));
        assert!(out.contains("<start_of_turn>model\nhello<end_of_turn>"));
        assert!(out.ends_with("<start_of_turn>model\n"));
    }

    #[test]
    fn chatml_multi_turn() {
        let out = ChatTemplate::ChatML
            .render_messages(msgs(&[("system", "Be concise."), ("user", "hi")]));
        assert!(out.contains("<|im_start|>system\nBe concise.<|im_end|>"));
        assert!(out.contains("<|im_start|>user\nhi<|im_end|>"));
        assert!(out.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn llama_multi_turn() {
        let out = ChatTemplate::Llama.render_messages(msgs(&[("user", "hi")]));
        assert!(out.contains("<|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|>"));
        assert!(out.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
    }

    #[test]
    fn mistral_prepends_system_to_first_user() {
        let out =
            ChatTemplate::Mistral.render_messages(msgs(&[("system", "Be brief."), ("user", "hi")]));
        assert_eq!(out, "[INST] Be brief.\n\nhi [/INST]");
    }

    #[test]
    fn mistral_multi_turn() {
        let out = ChatTemplate::Mistral.render_messages(msgs(&[
            ("user", "hi"),
            ("assistant", "hello"),
            ("user", "more"),
        ]));
        assert_eq!(out, "[INST] hi [/INST] hello [INST] more [/INST]");
    }

    #[test]
    fn plain_uses_role_labels() {
        let out =
            ChatTemplate::Plain.render_messages(msgs(&[("system", "Concise."), ("user", "hi")]));
        assert_eq!(out, "System: Concise.\nUser: hi\nAssistant: ");
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    #[test]
    fn for_model_id_gemma() {
        assert_eq!(
            ChatTemplate::for_model_id("google/gemma-3-4b-it"),
            ChatTemplate::Gemma
        );
        assert_eq!(
            ChatTemplate::for_model_id("Gemma-2-2B"),
            ChatTemplate::Gemma
        );
    }

    #[test]
    fn for_model_id_mistral_family() {
        assert_eq!(
            ChatTemplate::for_model_id("mistralai/Mistral-7B-Instruct-v0.3"),
            ChatTemplate::Mistral
        );
        assert_eq!(
            ChatTemplate::for_model_id("mistralai/Mixtral-8x7B"),
            ChatTemplate::Mistral
        );
    }

    #[test]
    fn for_model_id_llama() {
        assert_eq!(
            ChatTemplate::for_model_id("meta-llama/Llama-3.2-3B-Instruct"),
            ChatTemplate::Llama
        );
        assert_eq!(
            ChatTemplate::for_model_id("TinyLlama/TinyLlama-1.1B"),
            ChatTemplate::Llama
        );
    }

    #[test]
    fn for_model_id_chatml_family() {
        assert_eq!(
            ChatTemplate::for_model_id("Qwen/Qwen2.5-7B-Instruct"),
            ChatTemplate::ChatML
        );
        assert_eq!(
            ChatTemplate::for_model_id("deepseek-ai/DeepSeek-V2"),
            ChatTemplate::ChatML
        );
    }

    #[test]
    fn for_model_id_unknown_falls_back_to_plain() {
        assert_eq!(
            ChatTemplate::for_model_id("some-random-model"),
            ChatTemplate::Plain
        );
        assert_eq!(ChatTemplate::for_model_id(""), ChatTemplate::Plain);
    }

    #[test]
    fn for_family_recognises_all_canonical_strings() {
        assert_eq!(ChatTemplate::for_family("gemma2"), ChatTemplate::Gemma);
        assert_eq!(ChatTemplate::for_family("gemma3"), ChatTemplate::Gemma);
        assert_eq!(ChatTemplate::for_family("gemma4"), ChatTemplate::Gemma);
        assert_eq!(ChatTemplate::for_family("mistral"), ChatTemplate::Mistral);
        assert_eq!(ChatTemplate::for_family("mixtral"), ChatTemplate::Mistral);
        assert_eq!(ChatTemplate::for_family("llama"), ChatTemplate::Llama);
        assert_eq!(ChatTemplate::for_family("qwen"), ChatTemplate::ChatML);
        assert_eq!(ChatTemplate::for_family("qwen2"), ChatTemplate::ChatML);
        assert_eq!(ChatTemplate::for_family("qwen3"), ChatTemplate::ChatML);
        assert_eq!(ChatTemplate::for_family("deepseek"), ChatTemplate::ChatML);
        assert_eq!(ChatTemplate::for_family("gpt_oss"), ChatTemplate::ChatML);
    }

    #[test]
    fn for_family_unknown_falls_back_to_plain() {
        assert_eq!(ChatTemplate::for_family("generic"), ChatTemplate::Plain);
        assert_eq!(ChatTemplate::for_family("tinymodel"), ChatTemplate::Plain);
        assert_eq!(ChatTemplate::for_family("starcoder2"), ChatTemplate::Plain);
        assert_eq!(ChatTemplate::for_family(""), ChatTemplate::Plain);
    }

    #[test]
    fn gemma_wrap_includes_turn_markers() {
        let w = ChatTemplate::Gemma.wrap("hello");
        assert!(w.starts_with("<start_of_turn>user\n"));
        assert!(w.contains("hello"));
        assert!(w.ends_with("<start_of_turn>model\n"));
    }

    #[test]
    fn mistral_wrap_includes_inst_markers() {
        assert_eq!(ChatTemplate::Mistral.wrap("hello"), "[INST] hello [/INST]");
    }

    #[test]
    fn mistral_wrap_omits_literal_bos() {
        // BOS is added by the tokenizer via add_special_tokens=true;
        // including a literal `<s>` risks double-BOS.
        assert!(!ChatTemplate::Mistral.wrap("hello").contains("<s>"));
    }

    #[test]
    fn llama_wrap_includes_header_tags() {
        let w = ChatTemplate::Llama.wrap("hello");
        assert!(w.starts_with("<|begin_of_text|>"));
        assert!(w.contains("<|eot_id|>"));
        assert!(w.ends_with("assistant<|end_header_id|>\n\n"));
    }

    #[test]
    fn chatml_wrap_includes_im_tags() {
        let w = ChatTemplate::ChatML.wrap("hello");
        assert!(w.starts_with("<|im_start|>user\nhello<|im_end|>\n"));
        assert!(w.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn plain_wrap_is_passthrough() {
        assert_eq!(ChatTemplate::Plain.wrap("hello"), "hello");
        assert_eq!(ChatTemplate::Plain.wrap(""), "");
    }

    #[test]
    fn name_returns_lowercase_label() {
        assert_eq!(ChatTemplate::Gemma.name(), "gemma");
        assert_eq!(ChatTemplate::Mistral.name(), "mistral");
        assert_eq!(ChatTemplate::Llama.name(), "llama");
        assert_eq!(ChatTemplate::ChatML.name(), "chatml");
        assert_eq!(ChatTemplate::Plain.name(), "plain");
    }
}
