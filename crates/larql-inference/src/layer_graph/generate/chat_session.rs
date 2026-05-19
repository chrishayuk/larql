//! Multi-turn chat session — running token buffer with max-context eviction.
//!
//! [`ChatSession`] is the caller-side companion to [`generate_with_sampling`]
//! / [`generate_streaming`]. It owns the running token buffer, lets the
//! caller append user / assistant turns one at a time, and evicts the
//! oldest *whole turns* (not individual tokens) when the buffer exceeds
//! `max_context`.
//!
//! Whole-turn eviction (rather than sliding-window over individual tokens)
//! keeps the conversation coherent: the model never sees a half-rendered
//! turn fragment. If the very first turn alone exceeds `max_context`, the
//! session keeps it — eviction is a no-op when only one turn remains, so
//! the caller's prompt is never silently truncated.
//!
//! Templating is pluggable via [`TurnRenderer`]. Built-in renderers cover
//! Gemma, ChatML, and Llama-3. Pass any `Box<dyn TurnRenderer>` for other
//! families or a Jinja-rendered fragment.
//!
//! Note on KV state: this is a *token-buffer* multi-turn implementation —
//! every `generate_with_sampling` call still does a full prefill against
//! the buffer. KV carryover across turns is its own follow-up; this module
//! is the API surface that carryover would later plug into without
//! changing the caller's code.
//!
//! ## Sibling chat-template machinery
//!
//! - [`crate::chat`] — Jinja-driven, vindex-faithful rendering (preferred
//!   when a vindex directory is on disk).
//! - [`crate::prompt::ChatTemplate`] — heuristic enum + single-prompt
//!   wrapping for callers without vindex bytes; its `render_messages`
//!   delegates to the renderers below for Gemma / Llama / ChatML.
//!
//! [`generate_with_sampling`]: super::gpu::generate_with_sampling
//! [`generate_streaming`]: super::gpu::generate_streaming

use tokenizers::Tokenizer;

/// Context window default. Real models report this in their config; the
/// caller can override with [`ChatSession::with_max_context`].
pub const DEFAULT_MAX_CONTEXT: usize = 8192;

/// Role identifiers passed into [`TurnRenderer::render`]. Renderers may
/// choose to ignore unknown roles or emit them verbatim.
pub mod roles {
    pub const USER: &str = "user";
    pub const ASSISTANT: &str = "assistant";
    pub const SYSTEM: &str = "system";
}

/// Render a conversation turn into the model's text format.
///
/// Implementations must be deterministic — same `(role, text)` always
/// produces the same bytes — so the tokeniser produces stable IDs and
/// eviction is reproducible.
pub trait TurnRenderer {
    /// Render a single turn. Examples:
    /// - Gemma: `("user", "hi")` → `"<start_of_turn>user\nhi<end_of_turn>\n"`
    /// - ChatML: `("user", "hi")` → `"<|im_start|>user\nhi<|im_end|>\n"`
    fn render(&self, role: &str, text: &str) -> String;

    /// Marker that opens the assistant's response — appended after the
    /// user turn before generation starts. Lets the model "speak" by
    /// continuing the assistant's open turn.
    /// - Gemma: `"<start_of_turn>model\n"`
    /// - ChatML: `"<|im_start|>assistant\n"`
    fn assistant_open(&self) -> String;
}

/// Gemma 1/2/3/4 chat template.
pub struct GemmaRenderer;

impl TurnRenderer for GemmaRenderer {
    fn render(&self, role: &str, text: &str) -> String {
        // Gemma uses "model" rather than "assistant" inside the tag.
        let role = if role == roles::ASSISTANT {
            "model"
        } else {
            role
        };
        format!("<start_of_turn>{role}\n{text}<end_of_turn>\n")
    }
    fn assistant_open(&self) -> String {
        "<start_of_turn>model\n".to_string()
    }
}

/// ChatML — used by Qwen, OpenAI base, and a few finetunes.
pub struct ChatMLRenderer;

impl TurnRenderer for ChatMLRenderer {
    fn render(&self, role: &str, text: &str) -> String {
        format!("<|im_start|>{role}\n{text}<|im_end|>\n")
    }
    fn assistant_open(&self) -> String {
        "<|im_start|>assistant\n".to_string()
    }
}

/// Llama 3 chat template.
pub struct Llama3Renderer;

impl TurnRenderer for Llama3Renderer {
    fn render(&self, role: &str, text: &str) -> String {
        format!("<|start_header_id|>{role}<|end_header_id|>\n\n{text}<|eot_id|>")
    }
    fn assistant_open(&self) -> String {
        "<|start_header_id|>assistant<|end_header_id|>\n\n".to_string()
    }
}

/// Multi-turn chat session — owns the running token buffer and per-turn
/// lengths so eviction can drop *whole oldest turns* when the buffer
/// exceeds `max_context`.
pub struct ChatSession {
    tokenizer: Tokenizer,
    renderer: Box<dyn TurnRenderer>,
    max_context: usize,
    token_ids: Vec<u32>,
    turn_lengths: Vec<usize>,
    /// True if an assistant-open marker has been pushed and the next
    /// `extend_with_generated` will close out that turn.
    pending_assistant_turn: bool,
}

impl ChatSession {
    pub fn new(tokenizer: Tokenizer, renderer: Box<dyn TurnRenderer>) -> Self {
        Self {
            tokenizer,
            renderer,
            max_context: DEFAULT_MAX_CONTEXT,
            token_ids: Vec::new(),
            turn_lengths: Vec::new(),
            pending_assistant_turn: false,
        }
    }

    /// Convenience: Gemma-templated session.
    pub fn gemma(tokenizer: Tokenizer) -> Self {
        Self::new(tokenizer, Box::new(GemmaRenderer))
    }

    /// Convenience: ChatML-templated session.
    pub fn chatml(tokenizer: Tokenizer) -> Self {
        Self::new(tokenizer, Box::new(ChatMLRenderer))
    }

    /// Convenience: Llama-3-templated session.
    pub fn llama3(tokenizer: Tokenizer) -> Self {
        Self::new(tokenizer, Box::new(Llama3Renderer))
    }

    pub fn with_max_context(mut self, max: usize) -> Self {
        self.max_context = max;
        self
    }

    /// Append a system prompt as the very first turn. Optional — many
    /// templates handle the absence of a system turn fine.
    pub fn append_system(&mut self, text: &str) {
        self.append_role(roles::SYSTEM, text);
    }

    /// Append a fully-formed user turn. Eviction runs after.
    pub fn append_user(&mut self, text: &str) {
        self.append_role(roles::USER, text);
    }

    /// Append a fully-formed assistant turn. Eviction runs after. Useful
    /// when seeding the conversation with a few-shot example.
    pub fn append_assistant(&mut self, text: &str) {
        self.append_role(roles::ASSISTANT, text);
    }

    fn append_role(&mut self, role: &str, text: &str) {
        let rendered = self.renderer.render(role, text);
        let ids = self
            .tokenizer
            .encode(rendered, false)
            .map(|e| e.get_ids().to_vec())
            .unwrap_or_default();
        self.turn_lengths.push(ids.len());
        self.token_ids.extend(ids);
        self.evict_to_max_context();
    }

    /// Append the assistant-open marker so the model can continue with its
    /// response. The next [`Self::extend_with_generated`] / [`Self::extend_with_generated_text`]
    /// call closes this turn.
    pub fn open_assistant_turn(&mut self) {
        if self.pending_assistant_turn {
            return;
        }
        let marker = self.renderer.assistant_open();
        let ids = self
            .tokenizer
            .encode(marker, false)
            .map(|e| e.get_ids().to_vec())
            .unwrap_or_default();
        self.turn_lengths.push(ids.len());
        self.token_ids.extend(ids);
        self.pending_assistant_turn = true;
    }

    /// Append the assistant's generated token IDs to the running buffer.
    /// Closes the open assistant turn (must have called
    /// [`Self::open_assistant_turn`] first). Eviction runs after.
    pub fn extend_with_generated(&mut self, ids: &[u32]) {
        if !self.pending_assistant_turn {
            self.open_assistant_turn();
        }
        // Extend the open turn's length rather than starting a new one.
        if let Some(last) = self.turn_lengths.last_mut() {
            *last += ids.len();
        }
        self.token_ids.extend(ids);
        self.pending_assistant_turn = false;
        self.evict_to_max_context();
    }

    /// Tokenise the assistant's response text and append. Equivalent to
    /// `extend_with_generated(&tokenizer.encode(text)…)` but keeps the
    /// session as the single owner of the tokenizer.
    pub fn extend_with_generated_text(&mut self, text: &str) {
        let ids = self
            .tokenizer
            .encode(text, false)
            .map(|e| e.get_ids().to_vec())
            .unwrap_or_default();
        self.extend_with_generated(&ids);
    }

    /// Full token buffer to pass into generate_with_sampling.
    pub fn token_ids(&self) -> &[u32] {
        &self.token_ids
    }

    pub fn token_count(&self) -> usize {
        self.token_ids.len()
    }

    pub fn turn_count(&self) -> usize {
        self.turn_lengths.len()
    }

    pub fn max_context(&self) -> usize {
        self.max_context
    }

    /// Drop the oldest whole turns until `token_ids.len() <= max_context`,
    /// or until only one turn remains (whichever happens first). Never
    /// drops the only remaining turn — the caller's most recent prompt is
    /// always preserved even if it alone exceeds `max_context`.
    fn evict_to_max_context(&mut self) {
        while self.token_ids.len() > self.max_context && self.turn_lengths.len() > 1 {
            let drop_n = self.turn_lengths.remove(0);
            self.token_ids.drain(0..drop_n);
        }
    }

    /// Reset the session to empty. Tokenizer and renderer are kept.
    pub fn reset(&mut self) {
        self.token_ids.clear();
        self.turn_lengths.clear();
        self.pending_assistant_turn = false;
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    fn tiny_tokenizer() -> Tokenizer {
        // Whitespace-split word-level — every distinct word becomes one token.
        // Tokens used by the tests are: hi, bye, good, morning, the, capital,
        // of, france, model, user, assistant, system, plus role markers.
        let words = [
            "[UNK]",
            "<start_of_turn>",
            "<end_of_turn>",
            "<|im_start|>",
            "<|im_end|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|eot_id|>",
            "user",
            "assistant",
            "system",
            "model",
            "hi",
            "bye",
            "good",
            "morning",
            "the",
            "capital",
            "of",
            "france",
        ];
        let mut vocab = serde_json::Map::new();
        for (i, w) in words.iter().enumerate() {
            vocab.insert(w.to_string(), serde_json::Value::Number((i as u64).into()));
        }
        let json = serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": { "type": "Whitespace" },
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": vocab,
                "unk_token": "[UNK]",
            },
        });
        let bytes = serde_json::to_vec(&json).unwrap();
        Tokenizer::from_bytes(&bytes).unwrap()
    }

    #[test]
    fn gemma_renderer_uses_model_role_for_assistant() {
        let r = GemmaRenderer;
        assert!(r.render("assistant", "hi").contains("model"));
        assert!(!r.render("assistant", "hi").contains("assistant"));
    }

    #[test]
    fn chatml_renderer_uses_role_verbatim() {
        let r = ChatMLRenderer;
        assert!(r
            .render("assistant", "hi")
            .contains("<|im_start|>assistant"));
        assert!(r.render("user", "hi").contains("<|im_end|>"));
    }

    #[test]
    fn llama3_renderer_includes_eot() {
        let r = Llama3Renderer;
        assert!(r.render("user", "hi").contains("<|eot_id|>"));
        assert!(r.assistant_open().contains("assistant"));
    }

    #[test]
    fn empty_session_is_empty() {
        let s = ChatSession::gemma(tiny_tokenizer());
        assert_eq!(s.token_count(), 0);
        assert_eq!(s.turn_count(), 0);
        assert!(s.token_ids().is_empty());
    }

    #[test]
    fn append_user_records_one_turn() {
        let mut s = ChatSession::gemma(tiny_tokenizer());
        s.append_user("hi");
        assert_eq!(s.turn_count(), 1);
        assert!(s.token_count() > 0);
    }

    #[test]
    fn open_and_close_assistant_turn() {
        let mut s = ChatSession::gemma(tiny_tokenizer());
        s.append_user("hi");
        s.open_assistant_turn();
        assert_eq!(s.turn_count(), 2);
        let after_open = s.token_count();
        s.extend_with_generated(&[12u32, 13]);
        // Generated tokens extend the open turn, not a new one.
        assert_eq!(s.turn_count(), 2);
        assert_eq!(s.token_count(), after_open + 2);
    }

    #[test]
    fn extend_without_open_auto_opens() {
        let mut s = ChatSession::gemma(tiny_tokenizer());
        s.append_user("hi");
        let before = s.turn_count();
        s.extend_with_generated(&[12]);
        // extend_with_generated must implicitly open the assistant turn.
        assert_eq!(s.turn_count(), before + 1);
    }

    #[test]
    fn eviction_drops_oldest_whole_turns() {
        let mut s = ChatSession::gemma(tiny_tokenizer()).with_max_context(20);
        for _ in 0..5 {
            s.append_user("hi bye good morning"); // multi-token turn
        }
        // Buffer must fit max_context after eviction (or have only 1 turn left).
        assert!(s.token_count() <= s.max_context() || s.turn_count() == 1);
    }

    #[test]
    fn eviction_never_drops_last_turn() {
        // A single turn far larger than max_context must be preserved —
        // truncating the caller's prompt would silently corrupt the
        // request.
        let mut s = ChatSession::gemma(tiny_tokenizer()).with_max_context(2);
        s.append_user("hi bye good morning the capital of france");
        assert_eq!(s.turn_count(), 1);
        assert!(s.token_count() > s.max_context());
    }

    #[test]
    fn reset_clears_state() {
        let mut s = ChatSession::gemma(tiny_tokenizer());
        s.append_user("hi");
        s.open_assistant_turn();
        s.extend_with_generated(&[12]);
        s.reset();
        assert_eq!(s.token_count(), 0);
        assert_eq!(s.turn_count(), 0);
    }

    #[test]
    fn token_ids_grows_monotonically_within_a_turn() {
        let mut s = ChatSession::gemma(tiny_tokenizer());
        let n0 = s.token_count();
        s.append_user("hi");
        let n1 = s.token_count();
        s.append_assistant("bye");
        let n2 = s.token_count();
        assert!(n1 > n0);
        assert!(n2 > n1);
    }

    #[test]
    fn extend_with_generated_text_tokenises_through_session_tokenizer() {
        let mut s = ChatSession::gemma(tiny_tokenizer());
        s.append_user("hi");
        let before = s.token_count();
        s.extend_with_generated_text("bye");
        assert!(s.token_count() > before);
    }

    #[test]
    fn chatml_session_round_trips_tokens() {
        let mut s = ChatSession::chatml(tiny_tokenizer());
        s.append_user("hi");
        s.open_assistant_turn();
        // Buffer should contain ChatML markers tokenisable by the test vocab.
        let ids = s.token_ids().to_vec();
        assert!(!ids.is_empty());
    }
}
