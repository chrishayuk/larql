//! A conversation's prompt ids on a VINDEX3 container — the one
//! derivation `/v1/chat/completions` and `/v1/responses` share, and the
//! same one `larql run` / `larql chat` use on the same container.
//!
//! **Template.** The container's own chat template
//! (`chat_template.jinja`, else `tokenizer_config.json::chat_template`),
//! rendered by [`larql_inference::chat::render_conversation`] — never a
//! template guessed from the model's family or id. The wire messages
//! are [`flatten`]ed first (tool turns become text in the
//! `user`/`assistant` slots, as on VINDEX2), then rendered in full:
//! system, user and assistant turns all reach the template.
//!
//! **Encoding.** [`RenderedConversation::encode`]: a rendered template
//! carries its own special tokens, so it is encoded without the
//! tokenizer's post-processor (HF `apply_chat_template` semantics).
//!
//! **No template.** A container without one (and with no registry
//! fallback for its family) has no defined chat format. As in
//! `larql run`, a lone user message is then sent verbatim — a plain
//! completion, with the post-processor — and the server says so once,
//! at bind ([`crate::vindex3::load_v3_model_experts`]). A conversation
//! of more than that is refused: there is nowhere honest to put its
//! other turns, and dropping them silently would answer a different
//! question. `LARQL_RAW_PROMPT` on the server takes the same branch.
//!
//! [`RenderedConversation::encode`]: larql_inference::chat::RenderedConversation::encode

use larql_inference::chat::{render_conversation, ChatTurn, PromptShape, RenderedConversation};

use crate::error::ServerError;
use crate::vindex3::V3Model;

use super::chat::ChatMessage;
use super::prompt::{flatten, USER_ROLE};

/// Messages an untemplated prompt can carry: the one user turn.
const UNTEMPLATED_MESSAGE_LIMIT: usize = 1;

/// Render `messages` through `model`'s container template and encode
/// them for generation.
pub fn encode_v3_conversation(
    model: &V3Model,
    messages: &[ChatMessage],
) -> Result<Vec<u32>, ServerError> {
    let rendered = render_v3_conversation(model, messages)?;
    let ids = rendered
        .encode(&model.tokenizer)
        .map_err(ServerError::Internal)?;
    if ids.is_empty() {
        return Err(ServerError::BadRequest(
            "rendered prompt tokenises to empty".into(),
        ));
    }
    tracing::debug!(
        model = %model.id,
        templated = rendered.templated(),
        prompt_ids = ?ids,
        "V3 chat prompt"
    );
    Ok(ids)
}

/// The rendered prompt for `messages`, or the refusal for a
/// conversation an untemplated container cannot express.
fn render_v3_conversation(
    model: &V3Model,
    messages: &[ChatMessage],
) -> Result<RenderedConversation, ServerError> {
    let turns: Vec<ChatTurn> = flatten(messages)
        .into_iter()
        .map(|(role, content)| ChatTurn { role, content })
        .collect();
    let rendered =
        render_conversation(&model.path, &model.family, &turns).map_err(ServerError::BadRequest)?;
    let why_raw = match rendered.shape {
        PromptShape::Templated => return Ok(rendered),
        PromptShape::NoTemplate => "carries no chat template",
        PromptShape::RawRequested => "is served raw (LARQL_RAW_PROMPT is set)",
    };
    if !is_lone_user_turn(&turns) {
        return Err(ServerError::BadRequest(format!(
            "model {:?} {why_raw}, so only a single `{USER_ROLE}` message can be served (sent \
             verbatim as a plain completion); this conversation has {} messages. Send one user \
             message, or use /v1/completions",
            model.id,
            turns.len()
        )));
    }
    Ok(rendered)
}

fn is_lone_user_turn(turns: &[ChatTurn]) -> bool {
    turns.len() == UNTEMPLATED_MESSAGE_LIMIT && turns.iter().all(|t| t.role == USER_ROLE)
}
