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

    // ---- admin -------------------------------------------------------

    #[derive(Serialize, ToSchema)]
    pub struct HealthResponse {
        pub status: String,
        pub uptime_seconds: u64,
        pub requests_served: u64,
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
        // patches
        crate::routes::patches::handle_apply_patch,
        crate::routes::patches::handle_list_patches,
        crate::routes::patches::handle_remove_patch,
        // admin
        crate::routes::health::handle_health,
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
        // browse
        schemas::DescribeEdge,
        schemas::DescribeResponse,
        schemas::WalkHit,
        schemas::WalkResponse,
        schemas::RelationEntry,
        schemas::RelationsResponse,
        schemas::LayerBands,
        schemas::LoadedCapabilities,
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
        // admin
        schemas::HealthResponse,
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
