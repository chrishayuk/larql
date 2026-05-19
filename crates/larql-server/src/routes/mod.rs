//! Router setup — maps URL paths to handlers.

pub mod describe;
pub mod embed;
pub mod expert;
pub mod explain;
pub mod health;
pub mod infer;
pub mod insert;
pub mod models;
pub mod openai;
pub mod patches;
pub mod relations;
pub mod select;
pub mod shard;
pub mod stats;
pub mod stream;
pub mod topology;
pub mod walk;
pub mod walk_ffn;
pub mod warmup;

use std::sync::Arc;

use axum::extract::DefaultBodyLimit;
use axum::routing::{delete, get, post};
use axum::Router;

// Expert batch payloads can be large when the client batches all sequence
// positions into one call per layer (N_positions × top_K × hidden floats as
// JSON). 64 MB covers: 512 positions × 8 experts × 2816 floats × ~7 bytes/float.
const EXPERT_BATCH_BODY_LIMIT: usize = crate::http::REQUEST_BODY_LIMIT_BYTES;

use crate::state::AppState;

const HEALTH: &str = "/v1/health";
const MODELS: &str = "/v1/models";
const DESCRIBE: &str = "/v1/describe";
const WALK: &str = "/v1/walk";
const SELECT: &str = "/v1/select";
const RELATIONS: &str = "/v1/relations";
const STATS: &str = "/v1/stats";
const INFER: &str = "/v1/infer";
const PATCHES_APPLY: &str = "/v1/patches/apply";
const PATCHES: &str = "/v1/patches";
const PATCH_BY_NAME: &str = "/v1/patches/{name}";
const WALK_FFN: &str = "/v1/walk-ffn";
const WALK_FFN_Q8K: &str = "/v1/walk-ffn-q8k";
const EXPERT_TOPOLOGY: &str = "/v1/expert/topology";
const EXPERT_BATCH: &str = "/v1/expert/batch";
const EXPERTS_LAYER_BATCH: &str = "/v1/experts/layer-batch";
const EXPERTS_LAYER_BATCH_F16: &str = "/v1/experts/layer-batch-f16";
const EXPERTS_MULTI_LAYER_BATCH: &str = "/v1/experts/multi-layer-batch";
const EXPERTS_MULTI_LAYER_BATCH_Q8K: &str = "/v1/experts/multi-layer-batch-q8k";
const EXPERT: &str = "/v1/expert/{layer}/{expert_id}";
const EXPLAIN_INFER: &str = "/v1/explain-infer";
const INSERT: &str = "/v1/insert";
const STREAM: &str = "/v1/stream";
const WARMUP: &str = "/v1/warmup";
const EMBED: &str = "/v1/embed";
const EMBED_TOKEN: &str = "/v1/embed/{token_id}";
const LOGITS: &str = "/v1/logits";
const TOKEN_ENCODE: &str = "/v1/token/encode";
const TOKEN_DECODE: &str = "/v1/token/decode";
// Mode B shard handoff: donor streams its on-disk vindex as a tar so a
// freshly-assigned spare server can mirror the shard locally.
const SHARD: &str = "/v1/shard/{model_id}/{range}";

const OPENAI_EMBEDDINGS: &str = "/v1/embeddings";
const OPENAI_COMPLETIONS: &str = "/v1/completions";
const OPENAI_CHAT_COMPLETIONS: &str = "/v1/chat/completions";

const M_DESCRIBE: &str = "/v1/{model_id}/describe";
const M_WALK: &str = "/v1/{model_id}/walk";
const M_SELECT: &str = "/v1/{model_id}/select";
const M_RELATIONS: &str = "/v1/{model_id}/relations";
const M_STATS: &str = "/v1/{model_id}/stats";
const M_INFER: &str = "/v1/{model_id}/infer";
const M_PATCHES_APPLY: &str = "/v1/{model_id}/patches/apply";
const M_PATCHES: &str = "/v1/{model_id}/patches";
const M_PATCH_BY_NAME: &str = "/v1/{model_id}/patches/{name}";
const M_EXPLAIN_INFER: &str = "/v1/{model_id}/explain-infer";
const M_INSERT: &str = "/v1/{model_id}/insert";
const M_EMBED: &str = "/v1/{model_id}/embed";
const M_EMBED_TOKEN: &str = "/v1/{model_id}/embed/{token_id}";
const M_LOGITS: &str = "/v1/{model_id}/logits";
const M_TOKEN_ENCODE: &str = "/v1/{model_id}/token/encode";
const M_TOKEN_DECODE: &str = "/v1/{model_id}/token/decode";

/// Build the router for single-model serving.
pub fn single_model_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route(DESCRIBE, get(describe::handle_describe))
        .route(WALK, get(walk::handle_walk))
        .route(SELECT, post(select::handle_select))
        .route(RELATIONS, get(relations::handle_relations))
        .route(STATS, get(stats::handle_stats))
        .route(INFER, post(infer::handle_infer))
        .route(PATCHES_APPLY, post(patches::handle_apply_patch))
        .route(PATCHES, get(patches::handle_list_patches))
        .route(PATCH_BY_NAME, delete(patches::handle_remove_patch))
        .route(WALK_FFN, post(walk_ffn::handle_walk_ffn))
        .route(WALK_FFN_Q8K, post(walk_ffn::handle_walk_ffn_q8k))
        .route(EXPERT_TOPOLOGY, get(topology::handle_topology))
        .route(
            EXPERT_BATCH,
            post(expert::handle_expert_batch).layer(DefaultBodyLimit::max(EXPERT_BATCH_BODY_LIMIT)),
        )
        .route(
            EXPERTS_LAYER_BATCH,
            post(expert::handle_experts_layer_batch)
                .layer(DefaultBodyLimit::max(EXPERT_BATCH_BODY_LIMIT)),
        )
        .route(
            EXPERTS_LAYER_BATCH_F16,
            post(expert::handle_experts_layer_batch_f16)
                .layer(DefaultBodyLimit::max(EXPERT_BATCH_BODY_LIMIT)),
        )
        .route(
            EXPERTS_MULTI_LAYER_BATCH,
            post(expert::handle_experts_multi_layer_batch)
                .layer(DefaultBodyLimit::max(EXPERT_BATCH_BODY_LIMIT)),
        )
        .route(
            EXPERTS_MULTI_LAYER_BATCH_Q8K,
            post(expert::handle_experts_multi_layer_batch_q8k)
                .layer(DefaultBodyLimit::max(EXPERT_BATCH_BODY_LIMIT)),
        )
        .route(EXPERT, post(expert::handle_expert))
        .route(EXPLAIN_INFER, post(explain::handle_explain))
        .route(INSERT, post(insert::handle_insert))
        .route(STREAM, get(stream::handle_stream))
        .route(HEALTH, get(health::handle_health))
        .route(MODELS, get(models::handle_models))
        .route(WARMUP, post(warmup::handle_warmup))
        // Embed server endpoints (always available, required for --embed-only mode)
        .route(EMBED, post(embed::handle_embed))
        .route(EMBED_TOKEN, get(embed::handle_embed_single))
        .route(LOGITS, post(embed::handle_logits))
        .route(TOKEN_ENCODE, get(embed::handle_token_encode))
        .route(TOKEN_DECODE, get(embed::handle_token_decode))
        .route(SHARD, get(shard::handle_shard))
        .route(OPENAI_EMBEDDINGS, post(openai::handle_embeddings))
        .route(OPENAI_COMPLETIONS, post(openai::handle_completions))
        .route(
            OPENAI_CHAT_COMPLETIONS,
            post(openai::handle_chat_completions),
        )
        .with_state(state)
}

/// Build the router for multi-model serving.
pub fn multi_model_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route(HEALTH, get(health::handle_health))
        .route(MODELS, get(models::handle_models))
        .route(M_DESCRIBE, get(describe::handle_describe_multi))
        .route(M_WALK, get(walk::handle_walk_multi))
        .route(M_SELECT, post(select::handle_select_multi))
        .route(M_RELATIONS, get(relations::handle_relations_multi))
        .route(M_STATS, get(stats::handle_stats_multi))
        .route(M_INFER, post(infer::handle_infer_multi))
        .route(M_PATCHES_APPLY, post(patches::handle_apply_patch_multi))
        .route(M_PATCHES, get(patches::handle_list_patches_multi))
        .route(M_PATCH_BY_NAME, delete(patches::handle_remove_patch_multi))
        .route(M_EXPLAIN_INFER, post(explain::handle_explain_multi))
        .route(M_INSERT, post(insert::handle_insert_multi))
        // Embed server endpoints for multi-model mode
        .route(M_EMBED, post(embed::handle_embed_multi))
        .route(M_EMBED_TOKEN, get(embed::handle_embed_single_multi))
        .route(M_LOGITS, post(embed::handle_logits_multi))
        .route(M_TOKEN_ENCODE, get(embed::handle_token_encode_multi))
        .route(M_TOKEN_DECODE, get(embed::handle_token_decode_multi))
        .route(SHARD, get(shard::handle_shard))
        // OpenAI-compat endpoints (multi-model: client passes `model` in body).
        .route(OPENAI_EMBEDDINGS, post(openai::handle_embeddings))
        .route(OPENAI_COMPLETIONS, post(openai::handle_completions))
        .route(
            OPENAI_CHAT_COMPLETIONS,
            post(openai::handle_chat_completions),
        )
        .with_state(state)
}
