//! A whole conversation rendered through the template the model ships.
//!
//! [`super::render_user_prompt`] answers for one user turn; a chat loop
//! needs the same answer for a growing history. Re-rendering the full
//! history each turn — rather than appending to a previous rendering — is
//! deliberate: templates rewrite past turns (Qwen3 drops an earlier
//! assistant turn's `<think>` block), so the only prompt that matches what
//! HF's `apply_chat_template` would build is a fresh render of every turn.
//!
//! The rules are the single-prompt path's, so a one-turn conversation
//! renders the identical string `render_user_prompt` does on its default
//! path:
//!
//! - `LARQL_RAW_PROMPT` → the latest user turn, verbatim.
//! - The container's own template (`chat_template.jinja`, then
//!   `tokenizer_config.json::chat_template`).
//! - When a system message or thinking is requested (`LARQL_SYSTEM`,
//!   `LARQL_THINKING`, or the family's registry default system prompt),
//!   the registry's fallback template for a container that ships none,
//!   and `enable_thinking` is stated; otherwise it is left undefined.
//! - A conversation that opens with its own `system` turn (an HTTP
//!   client's) keeps it: the environment's system message and the
//!   family default are not added in front of it.
//! - No template at all → the latest user turn, verbatim, marked
//!   [`PromptShape::NoTemplate`] so the caller can say so.
//!
//! One deliberate difference: the special-token context (`bos_token`,
//! `eos_token`) always comes from the container's `tokenizer_config.json`,
//! as `apply_chat_template` supplies it. The result is the full text the
//! model should see, special tokens included, so a caller encodes it
//! *without* the tokenizer's post-processor —
//! [`RenderedConversation::encode`] is that rule, shared by every entry
//! point (`larql run`/`chat` and the server's VINDEX3 chat routes) so
//! they cannot tokenise the same conversation differently.

use std::path::Path;

use larql_vindex::tokenizers::Tokenizer;

use super::family_default_template;
use super::options::PromptOptions;
use super::render::render_messages;
use super::source::{container_template, load_tokenizer_config};

/// The role a chat template expects for the person's turns.
pub const USER_ROLE: &str = "user";
/// The role a chat template expects for the model's turns.
pub const ASSISTANT_ROLE: &str = "assistant";
/// The role a chat template expects for a system message.
pub const SYSTEM_ROLE: &str = "system";

/// One turn of a conversation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatTurn {
    pub role: String,
    pub content: String,
}

impl ChatTurn {
    /// A turn the person wrote.
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: USER_ROLE.to_string(),
            content: content.into(),
        }
    }

    /// A turn the model produced.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: ASSISTANT_ROLE.to_string(),
            content: content.into(),
        }
    }
}

/// How the prompt in a [`RenderedConversation`] was produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptShape {
    /// Rendered through a chat template; carries its own special tokens.
    Templated,
    /// `LARQL_RAW_PROMPT` asked for the latest user turn verbatim.
    RawRequested,
    /// The container carries no chat template; the latest user turn is
    /// sent verbatim, a plain completion.
    NoTemplate,
}

/// The text to tokenise for the next assistant turn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderedConversation {
    pub prompt: String,
    pub shape: PromptShape,
}

impl RenderedConversation {
    /// Whether a template rendered the prompt.
    pub fn templated(&self) -> bool {
        self.shape == PromptShape::Templated
    }

    /// Whether encoding adds the tokenizer's special tokens. A rendered
    /// template already carries its own (HF `apply_chat_template`
    /// semantics); only a raw prompt gets the post-processor.
    pub fn adds_special_tokens(&self) -> bool {
        !self.templated()
    }

    /// The prompt's token ids. A rendered template carries its own special
    /// tokens and is encoded without the post-processor; a raw prompt goes
    /// through the one V3 raw-prompt encoder
    /// ([`crate::vindex3::encode_raw_prompt`]) with `raw_bos`, the BOS the
    /// container declares ([`crate::vindex3::declared_prompt_bos`]). An
    /// empty encoding is returned empty; callers judge it.
    pub fn encode(&self, tokenizer: &Tokenizer, raw_bos: Option<u32>) -> Result<Vec<u32>, String> {
        if self.templated() {
            return tokenizer
                .encode(self.prompt.as_str(), false)
                .map(|encoding| encoding.get_ids().to_vec())
                .map_err(|e| format!("encode prompt: {e}"));
        }
        match crate::vindex3::encode_raw_prompt(tokenizer, self.prompt.as_str(), raw_bos) {
            Ok(ids) => Ok(ids),
            Err(crate::vindex3::PromptEncodeError::Empty) => Ok(Vec::new()),
            Err(e) => Err(format!("encode prompt: {e}")),
        }
    }
}

/// Whether the container at `vindex_dir` carries its own chat template
/// (`chat_template.jinja`, else `tokenizer_config.json::chat_template`).
pub fn container_has_chat_template(vindex_dir: &Path) -> bool {
    container_template(vindex_dir, &load_tokenizer_config(vindex_dir)).is_some()
}

/// Render `turns` (which must end with a user turn) for the next
/// assistant reply, through the template carried in `vindex_dir`.
/// `turns` may open with a `system` turn.
///
/// `family` is the model's declared family; it is consulted only for the
/// architecture registry's default system message and fallback template,
/// never matched by name here. A template that is present but fails to
/// render is an error, not a silent raw fallback.
pub fn render_conversation(
    vindex_dir: &Path,
    family: &str,
    turns: &[ChatTurn],
) -> Result<RenderedConversation, String> {
    let latest = match turns.last() {
        Some(turn) if turn.role == USER_ROLE => turn.content.clone(),
        Some(turn) => {
            return Err(format!(
                "a conversation renders for the next assistant turn, so it must end with a \
                 `{USER_ROLE}` turn; it ends with `{}`",
                turn.role
            ))
        }
        None => return Err("a conversation needs at least one turn".to_string()),
    };
    let options = PromptOptions::from_env();
    if options.raw {
        return Ok(RenderedConversation {
            prompt: latest,
            shape: PromptShape::RawRequested,
        });
    }
    // A conversation's own system turn is the caller's authority; the
    // environment / family default only fills an absent one.
    let has_own_system = turns.first().is_some_and(|t| t.role == SYSTEM_ROLE);
    let system = (!has_own_system)
        .then(|| options.system_prompt(family))
        .flatten();
    let shaped = options.thinking || system.is_some();
    let cfg = load_tokenizer_config(vindex_dir);
    let template = container_template(vindex_dir, &cfg)
        .or_else(|| shaped.then(|| family_default_template(family)).flatten());
    let Some(template) = template else {
        return Ok(RenderedConversation {
            prompt: latest,
            shape: PromptShape::NoTemplate,
        });
    };
    let messages: Vec<(String, String)> = system
        .map(|s| (SYSTEM_ROLE.to_string(), s))
        .into_iter()
        .chain(turns.iter().map(|t| (t.role.clone(), t.content.clone())))
        .collect();
    let enable_thinking = shaped.then_some(options.thinking);
    let prompt = render_messages(&template, &cfg, &messages, enable_thinking)
        .map_err(|e| format!("render chat template: {e}"))?;
    Ok(RenderedConversation {
        prompt,
        shape: PromptShape::Templated,
    })
}
