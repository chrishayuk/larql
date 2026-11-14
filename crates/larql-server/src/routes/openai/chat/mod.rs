//! `POST /v1/chat/completions` — OpenAI-compatible chat completions (N0.1, slice 2).
//!
//! Implements the [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat/create)
//! shape so existing `openai` SDKs work unmodified:
//!
//! ```python
//! from openai import OpenAI
//! client = OpenAI(base_url="http://larql:8080/v1", api_key="sk-...")
//! resp = client.chat.completions.create(
//!     model="gemma-3-4b",
//!     messages=[
//!         {"role": "system", "content": "You are a helpful assistant."},
//!         {"role": "user",   "content": "What is the capital of France?"},
//!     ],
//!     max_tokens=20,
//! )
//! ```
//!
//! ## Chat template handling
//!
//! `messages` is rendered to a single prompt via the model's chat
//! template. A VINDEX3 container renders through its own template
//! (`super::v3_prompt`, shared with `larql run`). A VINDEX2 model uses
//! the heuristic template (Gemma / Llama / ChatML / Mistral / plain)
//! detected from the model's `family` and `id`. The rendered prompt then
//! runs through the same generation loop as `/v1/completions`.
//!
//! VINDEX2 template detection precedence:
//! 1. `arch.family()` (authoritative when available)
//! 2. Substring match on `model.id` ("gemma", "llama", "qwen", …)
//! 3. Plain (fallback for unknown families and base models)
//!
//! ## Generation path
//!
//! Buffered + SSE streaming both call
//! `larql_inference::layer_graph::generate{,_streaming}` which is KV-
//! cached on f16 vindexes (and falls back to a per-step Q4_K decode
//! when the backend is CPU + Q4K). Generation acquires an exclusive
//! write guard on `LoadedModel.weights` for the duration; concurrent
//! reads block but other endpoints are unaffected in steady state.
//!
//! ## Slice 2-3 limitations
//!
//! - `tools` / `tool_choice` returns 400 (slice 4 = N0.6 constrained decoding)
//! - `response_format: json_object | json_schema` returns 400 (slice 4)
//! - `n>1` returns 400
//! - `logprobs` request field accepted, response field always `null` (F18)
//!
//! Module layout:
//!
//! ```text
//! routes/openai/chat/
//! ├── mod.rs     — module declarations + re-exports
//! ├── types.rs   — request/response wire types
//! ├── handler.rs — buffered handler + generation loop
//! ├── stream.rs  — SSE streaming + chunk builders
//! ├── tools.rs   — tool/response_format resolution + constrained mask
//! └── tests/     — unit tests (module tests folder)
//! ```

mod handler;
mod stream;
mod tools;
mod types;
mod v3;

#[cfg(test)]
mod tests;

pub use handler::handle_chat_completions;
pub use types::{
    ChatChoice, ChatChoiceMessage, ChatCompletionsRequest, ChatCompletionsResponse, ChatLogprobs,
    ChatMessage, ChatUsage, TokenLogprob, ToolCall, ToolCallFunction,
};

// `openapi.rs` names this handler as
// `crate::routes::openai::chat::handle_chat_completions`; utoipa's
// `paths(...)` macro resolves that to the macro-generated
// `__path_handle_chat_completions` struct, so re-export it alongside
// the handler to keep the OpenAPI aggregator compiling unchanged.
#[doc(hidden)]
pub use handler::__path_handle_chat_completions;

// Internal surface shared with the sibling `responses/` module (and
// documented from `prompt.rs`): generation entry point, its output
// carrier, and the tool/schema helpers. `run_chat_completion` /
// `ChatGenerationOutput` have no external caller today but were
// `pub(super)` before the module split — keep the path stable for the
// sibling modules that document/depend on it.
#[allow(unused_imports)]
pub(super) use handler::{run_chat_completion, ChatGenerationOutput};
pub(super) use tools::{
    build_constrained_mask, build_tool_call_message, schema_for_response_format,
};

/// `object` field on buffered chat completion responses.
const CHAT_COMPLETION_OBJECT: &str = "chat.completion";

/// `object` field on SSE chat completion chunks.
const CHAT_COMPLETION_CHUNK_OBJECT: &str = "chat.completion.chunk";

/// Completion-token budget when the request sets neither `max_tokens`
/// nor `max_completion_tokens`.
const DEFAULT_MAX_TOKENS: usize = 256;
