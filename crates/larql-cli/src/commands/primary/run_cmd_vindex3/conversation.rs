//! What a VINDEX3 run feeds the model: the user's text through the chat
//! template the container carries, or the text verbatim.
//!
//! Ollama's semantics: `run <model> "<prompt>"` and the chat loop both
//! talk to the model as a chat model, and `--raw` sends the prompt as a
//! plain completion. The rendering itself is
//! [`larql_inference::chat::render_conversation`] — the same template
//! resolution and the same Jinja environment the VINDEX2 `run` path uses
//! through `render_user_prompt`; this module only owns the turn history
//! and the choice of encoding.
//!
//! **History.** Every turn re-renders the whole conversation and runs it
//! through a fresh continuation state. Re-rendering (rather than
//! appending to the previous turn's tokens and keeping its KV) is what
//! the template requires: a template may rewrite earlier turns (Qwen3
//! strips an earlier assistant turn's `<think>` block), so the previous
//! prompt is not, in general, a token prefix of the next one.
//!
//! **Encoding.** A templated prompt already carries its special tokens,
//! so it is encoded without the tokenizer's post-processor — HF's
//! `apply_chat_template(tokenize=True)` contract, and the one
//! `larql bench` uses for VINDEX3. The rule is
//! [`larql_inference::chat::RenderedConversation::encode`], which the
//! server's VINDEX3 chat routes call too, so `larql run` and `larql serve`
//! give the same conversation the same ids: a rendered template carries its
//! own special tokens, and a raw prompt goes through the one raw-prompt
//! encoder every V3 surface shares (post-processor, then the declared BOS).

use std::io::Write;
use std::path::Path;

use larql_inference::chat::{render_conversation, ChatTurn, PromptShape, RenderedConversation};
use larql_inference::vindex3::declared_prompt_bos;
use larql_vindex::tokenizers::Tokenizer;

use super::BoxErr;

/// Why a prompt is sent without a template although the run did not ask.
const NO_TEMPLATE_NOTE: &str = "carries no chat template; sending prompts raw (plain completion)";

/// The refusal for a prompt that encodes to nothing.
const EMPTY_PROMPT: &str = "prompt encodes to no tokens — nothing to condition on";

/// The token ids for the next generation, and whether a template made them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct PreparedPrompt {
    pub(super) ids: Vec<u32>,
    pub(super) templated: bool,
}

/// One run's conversation with the model.
pub(super) struct Conversation<'a> {
    container: &'a Path,
    family: &'a str,
    /// Template rendering is off for this run (`--raw`, or an input the
    /// template path does not cover).
    raw: bool,
    /// Completed `user` / `assistant` turns, oldest first.
    history: Vec<ChatTurn>,
    /// The user turn whose reply is being generated.
    pending: Option<ChatTurn>,
    /// The no-template note has been written once already.
    noted_raw: bool,
    /// The BOS a raw prompt starts with, as the container declares it.
    bos: Option<u32>,
}

impl<'a> Conversation<'a> {
    pub(super) fn new(container: &'a Path, family: &'a str, raw: bool) -> Self {
        Self {
            container,
            family,
            raw,
            history: Vec::new(),
            pending: None,
            noted_raw: false,
            bos: declared_prompt_bos(container),
        }
    }

    /// The ids for the model's reply to `user`, given every earlier turn.
    /// Says once on `status` when the container has no template.
    pub(super) fn prompt(
        &mut self,
        tokenizer: &Tokenizer,
        user: &str,
        status: &mut dyn Write,
    ) -> Result<PreparedPrompt, BoxErr> {
        let turn = ChatTurn::user(user);
        let rendered = if self.raw {
            RenderedConversation {
                prompt: user.to_string(),
                shape: PromptShape::RawRequested,
            }
        } else {
            let mut turns = self.history.clone();
            turns.push(turn.clone());
            let rendered = render_conversation(self.container, self.family, &turns)
                .map_err(|e| format!("{e} (pass --raw to send the prompt verbatim)"))?;
            if rendered.shape == PromptShape::NoTemplate && !self.noted_raw {
                writeln!(status, "{} {NO_TEMPLATE_NOTE}", self.container.display())?;
                self.noted_raw = true;
            }
            rendered
        };
        // The encoding rule is the shared one: a rendered template carries
        // its own special tokens; a raw prompt gets the shared raw encoder.
        let templated = rendered.templated();
        let ids = rendered.encode(tokenizer, self.bos)?;
        if ids.is_empty() {
            return Err(EMPTY_PROMPT.into());
        }
        self.pending = templated.then_some(turn);
        Ok(PreparedPrompt { ids, templated })
    }

    /// Close the pending turn with the model's `reply`. Only a templated
    /// exchange joins the history: a raw prompt is a plain completion,
    /// with no turn structure to carry forward.
    pub(super) fn record_reply(&mut self, reply: &str) {
        if let Some(user) = self.pending.take() {
            self.history.push(user);
            self.history.push(ChatTurn::assistant(reply));
        }
    }

    /// Completed turns so far.
    #[cfg(test)]
    pub(super) fn history(&self) -> &[ChatTurn] {
        &self.history
    }
}
