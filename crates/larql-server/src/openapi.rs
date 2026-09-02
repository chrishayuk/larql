//! OpenAPI / Swagger UI aggregation.
//!
//! Spec JSON is served at `/v1/openapi.json` and the browse-friendly
//! Swagger UI at `/swagger-ui`. Both can be disabled with `--no-docs`.
//!
//! Handlers are annotated in place with `#[utoipa::path]`. This module
//! owns:
//! - `ApiDoc` — the aggregator `#[derive(OpenApi)]` struct.
//! - `schemas` — synthetic response structs for handlers that return
//!   `Json<serde_json::Value>` (most of the browse/inference surface).
//! - `params` — shared request parameters (e.g. `model_id`).
//! - `swagger_router()` — helper that returns a ready-to-merge router
//!   hosting both the UI and the spec JSON.

use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

use crate::error::ErrorBody;

pub mod params {
    use utoipa::IntoParams;

    /// Path parameter selecting which vindex to target in multi-model mode.
    #[derive(IntoParams)]
    #[into_params(parameter_in = Path)]
    #[allow(dead_code)]
    pub struct ModelIdParam {
        /// The id of a loaded vindex, e.g. `gemma-3-1b-it`.
        pub model_id: String,
    }
}

pub mod schemas {
    //! Synthetic response schemas.
    //!
    //! Populated as each handler group is annotated. Structs here are
    //! `Serialize + ToSchema` mirrors of the actual JSON the handlers
    //! emit via `Json<serde_json::Value>`. They are never constructed at
    //! runtime — they exist purely for spec generation.

    use serde::Serialize;
    use utoipa::ToSchema;

    // ---- browse ------------------------------------------------------

    /// One knowledge edge returned from `/v1/describe`.
    #[derive(Serialize, ToSchema)]
    pub struct DescribeEdge {
        /// Top token at this feature (trimmed).
        pub target: String,
        /// Gate activation score (rounded to 0.1).
        pub gate_score: f32,
        /// Layer the feature lives on.
        pub layer: usize,
        /// Feature index within the layer.
        pub feature: usize,
        /// Relation label (present when a probe-confirmed label exists).
        #[serde(skip_serializing_if = "Option::is_none")]
        pub relation: Option<String>,
    }

    #[derive(Serialize, ToSchema)]
    pub struct DescribeResponse {
        pub entity: String,
        pub model: String,
        pub edges: Vec<DescribeEdge>,
        pub latency_ms: f64,
    }

    /// One walk hit returned from `/v1/walk`.
    #[derive(Serialize, ToSchema)]
    pub struct WalkHit {
        pub layer: usize,
        pub feature: usize,
        pub gate_score: f32,
        pub target: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub relation: Option<String>,
    }

    #[derive(Serialize, ToSchema)]
    pub struct WalkResponse {
        pub prompt: String,
        pub hits: Vec<WalkHit>,
        pub latency_ms: f64,
    }

    #[derive(Serialize, ToSchema)]
    pub struct RelationEntry {
        pub name: String,
        pub count: usize,
        pub max_score: f32,
        pub min_layer: usize,
        pub max_layer: usize,
        pub examples: Vec<String>,
    }

    #[derive(Serialize, ToSchema)]
    pub struct RelationsResponse {
        pub relations: Vec<RelationEntry>,
        pub total: usize,
        pub latency_ms: f64,
    }

    #[derive(Serialize, ToSchema)]
    pub struct LayerBands {
        pub syntax: [usize; 2],
        pub knowledge: [usize; 2],
        pub output: [usize; 2],
    }

    #[derive(Serialize, ToSchema)]
    pub struct LoadedCapabilities {
        pub browse: bool,
        pub inference: bool,
        pub ffn_service: bool,
        pub embed_service: bool,
    }

    #[derive(Serialize, ToSchema)]
    pub struct StatsResponse {
        pub model: String,
        pub family: String,
        pub layers: usize,
        pub features: usize,
        pub features_per_layer: usize,
        pub hidden_size: usize,
        pub vocab_size: usize,
        pub extract_level: String,
        pub dtype: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub mode: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub layer_bands: Option<LayerBands>,
        pub loaded: LoadedCapabilities,
        /// Server-level counters: uptime, request count, and the
        /// bounded per-client stores (patch sessions, stored
        /// responses, V3 KV continuation cache with hit/miss counts).
        #[serde(skip_serializing_if = "Option::is_none")]
        pub server: Option<serde_json::Value>,
    }

    /// One entry in the OpenAI-compatible `/v1/models` list.
    #[derive(Serialize, ToSchema)]
    pub struct ModelEntry {
        pub id: String,
        pub object: String,
        pub created: u64,
        pub owned_by: String,
        /// Route prefix for this model. `/v1/{id}` in multi-model mode, `/v1` otherwise.
        pub path: String,
        /// Total features across all layers.
        pub features: usize,
        pub loaded: bool,
    }

    #[derive(Serialize, ToSchema)]
    pub struct ModelsListResponse {
        pub object: String,
        pub data: Vec<ModelEntry>,
    }

    // ---- inference ---------------------------------------------------

    #[derive(Serialize, ToSchema)]
    pub struct SelectRow {
        pub layer: usize,
        pub feature: usize,
        pub target: String,
        pub confidence: f32,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub relation: Option<String>,
    }

    #[derive(Serialize, ToSchema)]
    pub struct SelectResponse {
        pub rows: Vec<SelectRow>,
        pub total: usize,
        pub latency_ms: f64,
    }

    #[derive(Serialize, ToSchema)]
    pub struct Prediction {
        pub token: String,
        pub probability: f64,
    }

    #[derive(Serialize, ToSchema)]
    pub struct InferResponse {
        pub prompt: String,
        pub mode: String,
        /// Single-mode (`walk` or `dense`).
        #[serde(skip_serializing_if = "Option::is_none")]
        pub predictions: Option<Vec<Prediction>>,
        /// Populated in `compare` mode.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub walk: Option<Vec<Prediction>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub dense: Option<Vec<Prediction>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub walk_ms: Option<f64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub dense_ms: Option<f64>,
        pub latency_ms: f64,
    }

    #[derive(Serialize, ToSchema)]
    pub struct ExplainLayerEntry {
        pub layer: usize,
        pub top_features: Vec<serde_json::Value>,
        pub top_tokens: Vec<(String, f64)>,
    }

    #[derive(Serialize, ToSchema)]
    pub struct ExplainResponse {
        pub prompt: String,
        pub predictions: Vec<Prediction>,
        pub layers: Vec<ExplainLayerEntry>,
        pub latency_ms: f64,
    }

    #[derive(Serialize, ToSchema)]
    pub struct InsertResponse {
        pub success: bool,
        pub entity: String,
        pub relation: String,
        pub target: String,
        pub layers_written: Vec<usize>,
        pub latency_ms: f64,
    }

    // ---- patches -----------------------------------------------------

    /// Request body for `POST /v1/patches/apply`. Provide either a `url`
    /// pointing at a `.vlp` file (local path or `hf://` URL) or an
    /// inline `patch` object. One of the two is required.
    #[derive(Serialize, ToSchema)]
    pub struct ApplyPatchBody {
        /// Local path, `http(s)://`, or `hf://` URL to a `.vlp` patch file.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub url: Option<String>,
        /// Inline patch payload. See VindexPatch docs for schema; includes
        /// `description`, `base_model`, and `operations` (INSERT / DELETE).
        #[serde(skip_serializing_if = "Option::is_none")]
        pub patch: Option<serde_json::Value>,
    }

    #[derive(Serialize, ToSchema)]
    pub struct ApplyPatchResponse {
        pub applied: String,
        pub operations: usize,
        pub active_patches: usize,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub session: Option<String>,
    }

    #[derive(Serialize, ToSchema)]
    pub struct PatchEntry {
        pub name: String,
        pub operations: usize,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub base_model: Option<String>,
    }

    #[derive(Serialize, ToSchema)]
    pub struct ListPatchesResponse {
        pub patches: Vec<PatchEntry>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub session: Option<String>,
    }

    #[derive(Serialize, ToSchema)]
    pub struct RemovePatchResponse {
        pub removed: String,
        pub active_patches: usize,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub session: Option<String>,
    }

    // ---- sessions ----------------------------------------------------

    #[derive(Serialize, ToSchema)]
    pub struct SessionPatches {
        /// How many patches are applied to this session.
        pub active: usize,
        /// Patch identities, in application order. Contents are not
        /// exposed here — `/v1/patches` is the authority for those.
        pub ids: Vec<String>,
    }

    #[derive(Serialize, ToSchema)]
    pub struct SessionContinuation {
        /// Whether a resident KV state owned by this session can be
        /// resumed from.
        pub available: bool,
        /// Prompt tokens already absorbed into the resident state(s).
        pub input_tokens: u64,
        /// Generations on this session where resumption engaged.
        pub resumptions: u64,
        /// Prompt tokens served from resumed KV, cumulative.
        pub reused_tokens_total: u64,
    }

    #[derive(Serialize, ToSchema)]
    pub struct SessionResponse {
        pub object: String,
        /// The `X-Session-Id` value the client uses.
        pub id: String,
        /// Runtime binding the session was created against.
        pub model: String,
        pub created_at: u64,
        pub last_used_at: u64,
        /// When the session expires if left idle from `last_used_at`.
        pub expires_at: u64,
        /// Lifecycle state; `active` is the only observable value —
        /// expired sessions are absent, not listed.
        pub state: String,
        pub patches: SessionPatches,
        pub continuation: SessionContinuation,
    }

    #[derive(Serialize, ToSchema)]
    pub struct SessionListResponse {
        pub object: String,
        /// Live sessions, most recently used first.
        pub data: Vec<SessionResponse>,
    }

    #[derive(Serialize, ToSchema)]
    pub struct SessionDeletedResponse {
        pub object: String,
        pub id: String,
        /// False when the session was already gone — deletion is
        /// idempotent, not an error.
        pub deleted: bool,
        pub patches_freed: usize,
        pub continuations_freed: usize,
    }

    // ---- admin -------------------------------------------------------

    /// `GET /v1/capabilities`. The `sources` / `explorer` /
    /// `runtime` blocks are generated from
    /// `crate::capabilities::ROUTE_CAPABILITIES` and the mounted-route
    /// ledger, so this schema documents the *shape*; the authority for
    /// which keys appear is that table, not this struct.
    #[derive(Serialize, ToSchema)]
    pub struct CapabilitiesResponse {
        /// Always `"capabilities"`.
        pub object: String,
        /// Report schema version. A client that does not recognise it
        /// must refuse the document rather than read the keys it knows.
        pub schema: u32,
        /// `"public_explorer"` | `"single_model"` | `"multi_model"`.
        pub profile: String,
        /// Which source reference forms each source-taking verb accepts.
        pub sources: serde_json::Value,
        /// The read surface over a bound container.
        pub explorer: serde_json::Value,
        /// What this process will do with the bound model, and on what.
        pub runtime: serde_json::Value,
        /// Every path this server mounted, sorted.
        pub routes: Vec<String>,
    }

    #[derive(Serialize, ToSchema)]
    pub struct HealthResponse {
        pub status: String,
        pub uptime_seconds: u64,
        pub requests_served: u64,
    }

    #[derive(Serialize, ToSchema)]
    pub struct RuntimeModel {
        pub id: String,
        pub architecture: String,
        /// `"vindex2"` | `"vindex3"`.
        pub format: String,
        /// `None` for a VINDEX3 container — it carries no single
        /// top-level quant tag the way a VINDEX2 `index.json` does.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub quantization: Option<String>,
    }

    #[derive(Serialize, ToSchema)]
    pub struct RuntimeBackend {
        /// Whether this binary was compiled with Metal-accelerated MoE
        /// expert dispatch — a compile-time fact, not a claim that
        /// Metal is driving the current request.
        pub metal_compiled: bool,
    }

    #[derive(Serialize, ToSchema)]
    pub struct RuntimeMemory {
        /// Peak resident-set size of this process (`getrusage`), in
        /// bytes. `None` on a non-Unix target or a syscall failure.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub resident_bytes: Option<u64>,
        /// Estimated resident footprint of the bound model's weights.
        /// `None` when no model is bound, or on a VINDEX3 container
        /// (no estimator exists yet).
        #[serde(skip_serializing_if = "Option::is_none")]
        pub model_bytes: Option<u64>,
    }

    #[derive(Serialize, ToSchema)]
    pub struct RuntimePerformance {
        /// `None` until at least one generation has completed.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub prefill_tokens_per_second: Option<f64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub decode_tokens_per_second: Option<f64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub last_request_latency_ms: Option<f64>,
    }

    #[derive(Serialize, ToSchema)]
    pub struct RuntimeGeneration {
        pub active: bool,
        pub active_requests: u32,
    }

    /// `GET /v1/runtime` — server + model + backend + memory +
    /// performance snapshot. Never constructed at runtime (the handler
    /// builds `serde_json::Value` directly); this exists purely so the
    /// OpenAPI spec documents the shape.
    #[derive(Serialize, ToSchema)]
    pub struct RuntimeResponse {
        pub status: String,
        pub version: String,
        pub uptime_ms: u64,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub model: Option<RuntimeModel>,
        pub backend: RuntimeBackend,
        pub memory: RuntimeMemory,
        pub performance: RuntimePerformance,
        pub generation: RuntimeGeneration,
    }

    #[derive(Serialize, ToSchema)]
    pub struct TokenEncodeResponse {
        pub token_ids: Vec<u32>,
        pub text: String,
    }

    #[derive(Serialize, ToSchema)]
    pub struct TokenDecodeResponse {
        pub text: String,
        pub token_ids: Vec<u32>,
    }

    #[derive(Serialize, ToSchema)]
    pub struct EmbedSingleJsonResponse {
        pub token_id: u32,
        pub embedding: Vec<f32>,
        pub hidden_size: usize,
    }

    // ---- openai ------------------------------------------------------
    //
    // These mirror the OpenAI wire contract at a high level.
    // Full nested types (tools, tool_calls, logprobs, usage) are documented
    // inline as open JSON objects to avoid a deep ToSchema tree.

    /// Subset of the OpenAI `POST /v1/embeddings` request body.
    #[derive(Serialize, ToSchema)]
    pub struct OpenAiEmbeddingsRequest {
        /// Model id. Required in multi-model mode; ignored otherwise.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub model: Option<String>,
        /// String, string[], int[] (single sequence), or int[][] (batch of sequences).
        pub input: serde_json::Value,
        /// `"float"` (default) or `"base64"`.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub encoding_format: Option<String>,
        /// Requested output dimensionality (ignored; returns native hidden size).
        #[serde(skip_serializing_if = "Option::is_none")]
        pub dimensions: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub user: Option<String>,
    }

    #[derive(Serialize, ToSchema)]
    pub struct OpenAiEmbeddingObject {
        pub object: String,
        pub index: usize,
        /// `[f32]` when `encoding_format = "float"`, or a base64 string otherwise.
        pub embedding: serde_json::Value,
    }

    #[derive(Serialize, ToSchema)]
    pub struct OpenAiEmbeddingsResponse {
        pub object: String,
        pub data: Vec<OpenAiEmbeddingObject>,
        pub model: String,
        pub usage: serde_json::Value,
    }

    /// OpenAI `POST /v1/completions` request.
    #[derive(Serialize, ToSchema)]
    pub struct OpenAiCompletionsRequest {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub model: Option<String>,
        /// Prompt — string or string[].
        pub prompt: serde_json::Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub max_tokens: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub temperature: Option<f32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub top_p: Option<f32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub stream: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub n: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub stop: Option<serde_json::Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub echo: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub logprobs: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub seed: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub user: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub frequency_penalty: Option<f32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub presence_penalty: Option<f32>,
    }

    #[derive(Serialize, ToSchema)]
    pub struct OpenAiCompletionsResponse {
        pub id: String,
        pub object: String,
        pub created: u64,
        pub model: String,
        pub choices: Vec<serde_json::Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub usage: Option<serde_json::Value>,
    }

    /// OpenAI `POST /v1/chat/completions` request. `messages` is an array
    /// of `{role: "system"|"user"|"assistant"|"tool", content, ...}`; tools
    /// and structured output are open JSON (see OpenAI docs).
    #[derive(Serialize, ToSchema)]
    pub struct OpenAiChatRequest {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub model: Option<String>,
        pub messages: Vec<serde_json::Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub max_tokens: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub temperature: Option<f32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub top_p: Option<f32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub stream: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub n: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub stop: Option<serde_json::Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub logprobs: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub top_logprobs: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub seed: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub user: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub frequency_penalty: Option<f32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub presence_penalty: Option<f32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub response_format: Option<serde_json::Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub tools: Option<serde_json::Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub tool_choice: Option<serde_json::Value>,
    }

    #[derive(Serialize, ToSchema)]
    pub struct OpenAiChatResponse {
        pub id: String,
        pub object: String,
        pub created: u64,
        pub model: String,
        pub choices: Vec<serde_json::Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub usage: Option<serde_json::Value>,
    }

    /// OpenAI `POST /v1/responses` request. `input` is a string or an
    /// array of input items (`message`, `function_call`,
    /// `function_call_output`); tools use the flattened Responses shape.
    #[derive(Serialize, ToSchema)]
    pub struct OpenAiResponsesRequest {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub model: Option<String>,
        /// String, or array of input items.
        pub input: serde_json::Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub instructions: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub max_output_tokens: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub temperature: Option<f32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub top_p: Option<f32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub stream: Option<bool>,
        /// Persist for `previous_response_id` chaining (default true;
        /// in-memory and bounded).
        #[serde(skip_serializing_if = "Option::is_none")]
        pub store: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub previous_response_id: Option<String>,
        /// Function tools, Responses shape: `[{type, name, description, parameters}]`.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub tools: Option<serde_json::Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub tool_choice: Option<serde_json::Value>,
        /// `{format: {type: "text" | "json_object" | "json_schema", ...}}`.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub text: Option<serde_json::Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub metadata: Option<serde_json::Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub user: Option<String>,
    }

    /// OpenAI Responses envelope (`object: "response"`).
    #[derive(Serialize, ToSchema)]
    pub struct OpenAiResponsesResponse {
        pub id: String,
        pub object: String,
        pub created_at: u64,
        /// `completed` | `incomplete` | `failed`.
        pub status: String,
        pub model: String,
        /// Output items: `message` (with `output_text` content parts)
        /// and `function_call`.
        pub output: Vec<serde_json::Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub previous_response_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub usage: Option<serde_json::Value>,
    }
}

#[derive(OpenApi)]
#[openapi(
    info(
        title = "larql-server",
        version = env!("CARGO_PKG_VERSION"),
        description = "HTTP API for vindex knowledge queries, inference, and remote MoE expert shards.",
    ),
    tags(
        (name = "browse",    description = "Knowledge graph browse (no weights required)"),
        (name = "inference", description = "Forward passes, explain, insert, warmup"),
        (name = "openai",    description = "OpenAI-compatible endpoints"),
        (name = "expert",    description = "Remote MoE shard endpoints (binary wire)"),
        (name = "patches",   description = "Runtime patch overlay"),
        (name = "sessions",  description = "Session observability and eviction"),
        (name = "admin",     description = "Health, models, embed, tokens, WebSocket"),
    ),
    paths(
        // browse
        crate::routes::describe::handle_describe,
        crate::routes::walk::handle_walk,
        crate::routes::relations::handle_relations,
        crate::routes::stats::handle_stats,
        crate::routes::topology::handle_topology,
        crate::routes::models::handle_models,
        // inference
        crate::routes::select::handle_select,
        crate::routes::infer::handle_infer,
        crate::routes::explain::handle_explain,
        crate::routes::insert::handle_insert,
        crate::routes::warmup::handle_warmup,
        // sessions
        crate::routes::sessions::handle_list_sessions,
        crate::routes::sessions::handle_get_session,
        crate::routes::sessions::handle_delete_session,
        // patches
        crate::routes::patches::handle_apply_patch,
        crate::routes::patches::handle_list_patches,
        crate::routes::patches::handle_remove_patch,
        // admin
        crate::routes::health::handle_health,
        crate::routes::capabilities::handle_capabilities,
        crate::routes::runtime::handle_runtime,
        crate::routes::runtime_lifecycle::handle_load_model,
        crate::routes::runtime_lifecycle::handle_unload_model,
        crate::routes::embed::handle_embed,
        crate::routes::embed::handle_embed_single,
        crate::routes::embed::handle_logits,
        crate::routes::embed::handle_token_encode,
        crate::routes::embed::handle_token_decode,
        crate::routes::stream::handle_stream,
        // openai
        crate::routes::openai::embeddings::handle_embeddings,
        crate::routes::openai::completions::handle_completions,
        crate::routes::openai::chat::handle_chat_completions,
        crate::routes::openai::responses::handler::handle_responses,
        crate::routes::openai::responses::retrieve::handle_get_response,
        crate::routes::openai::responses::retrieve::handle_delete_response,
        crate::routes::models::handle_model_retrieve,
        // expert
        crate::routes::walk_ffn::handle_walk_ffn,
        crate::routes::walk_ffn::handle_walk_ffn_q8k,
        crate::routes::expert::single::handle_expert,
        crate::routes::expert::batch_legacy::handle_expert_batch,
        crate::routes::expert::layer_batch::handle_experts_layer_batch,
        crate::routes::expert::layer_batch::handle_experts_layer_batch_f16,
        crate::routes::expert::multi_layer_batch::handle_experts_multi_layer_batch,
        crate::routes::expert::multi_layer_batch::handle_experts_multi_layer_batch_q8k,
        // multi-model variants — same handlers with a `{model_id}` path prefix
        crate::routes::describe::handle_describe_multi,
        crate::routes::walk::handle_walk_multi,
        crate::routes::relations::handle_relations_multi,
        crate::routes::stats::handle_stats_multi,
        crate::routes::select::handle_select_multi,
        crate::routes::infer::handle_infer_multi,
        crate::routes::explain::handle_explain_multi,
        crate::routes::insert::handle_insert_multi,
        crate::routes::patches::handle_apply_patch_multi,
        crate::routes::patches::handle_list_patches_multi,
        crate::routes::patches::handle_remove_patch_multi,
        crate::routes::embed::handle_embed_multi,
        crate::routes::embed::handle_embed_single_multi,
        crate::routes::embed::handle_logits_multi,
        crate::routes::embed::handle_token_encode_multi,
        crate::routes::embed::handle_token_decode_multi,
    ),
    components(schemas(
        ErrorBody,
        crate::routes::openai::error::OpenAIErrorBody,
        crate::routes::openai::error::OpenAIErrorPayload,
        // browse
        schemas::DescribeEdge,
        schemas::DescribeResponse,
        schemas::WalkHit,
        schemas::WalkResponse,
        schemas::RelationEntry,
        schemas::RelationsResponse,
        schemas::LayerBands,
        schemas::LoadedCapabilities,
        schemas::CapabilitiesResponse,
        schemas::StatsResponse,
        schemas::ModelEntry,
        schemas::ModelsListResponse,
        crate::routes::topology::TopologyResponse,
        // inference
        crate::routes::select::SelectRequest,
        schemas::SelectRow,
        schemas::SelectResponse,
        crate::routes::infer::InferRequest,
        schemas::Prediction,
        schemas::InferResponse,
        crate::routes::explain::ExplainRequest,
        schemas::ExplainLayerEntry,
        schemas::ExplainResponse,
        crate::routes::insert::InsertRequest,
        schemas::InsertResponse,
        crate::routes::warmup::WarmupRequest,
        crate::routes::warmup::WarmupResponse,
        // patches
        schemas::ApplyPatchBody,
        schemas::ApplyPatchResponse,
        schemas::PatchEntry,
        schemas::ListPatchesResponse,
        schemas::RemovePatchResponse,
        // sessions
        schemas::SessionPatches,
        schemas::SessionContinuation,
        schemas::SessionResponse,
        schemas::SessionListResponse,
        schemas::SessionDeletedResponse,
        // admin
        schemas::HealthResponse,
        schemas::RuntimeModel,
        schemas::RuntimeBackend,
        schemas::RuntimeMemory,
        schemas::RuntimePerformance,
        schemas::RuntimeGeneration,
        schemas::RuntimeResponse,
        crate::routes::runtime_lifecycle::LoadModelRequest,
        schemas::TokenEncodeResponse,
        schemas::TokenDecodeResponse,
        schemas::EmbedSingleJsonResponse,
        crate::routes::embed::EmbedRequest,
        crate::routes::embed::EmbedResponse,
        crate::routes::embed::LogitsRequest,
        crate::routes::embed::LogitsResponse,
        crate::routes::embed::TokenProb,
        // openai
        schemas::OpenAiEmbeddingsRequest,
        schemas::OpenAiEmbeddingObject,
        schemas::OpenAiEmbeddingsResponse,
        schemas::OpenAiCompletionsRequest,
        schemas::OpenAiCompletionsResponse,
        schemas::OpenAiChatRequest,
        schemas::OpenAiChatResponse,
        schemas::OpenAiResponsesRequest,
        schemas::OpenAiResponsesResponse,
        // expert
        crate::routes::expert::SingleExpertRequest,
        crate::routes::expert::SingleExpertResponse,
        crate::routes::expert::BatchExpertItem,
        crate::routes::expert::BatchExpertRequest,
        crate::routes::expert::BatchExpertResult,
        crate::routes::expert::BatchExpertResponse,
    )),
)]
pub struct ApiDoc;

/// Build a router hosting Swagger UI at `/swagger-ui` and the spec at
/// `/v1/openapi.json`. Merge into the main app router.
pub fn swagger_router() -> axum::Router {
    SwaggerUi::new("/swagger-ui")
        .url("/v1/openapi.json", ApiDoc::openapi())
        .into()
}
