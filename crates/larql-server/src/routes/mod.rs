//! Router setup — maps URL paths to handlers.

pub mod capabilities;
pub mod container_facts;
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
pub(crate) mod paths;
pub mod plan;
pub mod query;
pub mod relations;
pub mod runtime;
pub mod runtime_lifecycle;
pub mod select;
pub mod sessions;
pub mod shard;
pub mod stats;
pub mod stream;
pub mod topology;
pub mod walk;
pub mod walk_ffn;
pub mod warmup;

use std::sync::Arc;

use axum::extract::DefaultBodyLimit;
use axum::routing::{delete, get, post, MethodRouter};
use axum::Router;

// Expert batch payloads can be large when the client batches all sequence
// positions into one call per layer (N_positions × top_K × hidden floats as
// JSON). 64 MB covers: 512 positions × 8 experts × 2816 floats × ~7 bytes/float.
const EXPERT_BATCH_BODY_LIMIT: usize = crate::http::REQUEST_BODY_LIMIT_BYTES;

use crate::capabilities::{Capabilities, MountedRoutes, ServerProfile};
use crate::state::AppState;
use paths::*;

/// Builds a router while recording every path it mounts.
///
/// `GET /v1/capabilities` answers from this ledger, so the surface a
/// server advertises and the surface axum actually serves are two
/// readings of one act — not two lists that agree by discipline. A
/// route added below is advertised automatically; a route deleted
/// stops being advertised automatically. See [`crate::capabilities`].
struct Mount {
    router: Router<Arc<AppState>>,
    /// Sub-routers with their own state (the LQL bridge), merged in
    /// `finish`. Their paths go in the same ledger — where the state
    /// comes from is not the client's question.
    grafts: Vec<Router>,
    mounted: MountedRoutes,
}

impl Mount {
    fn new() -> Self {
        Self {
            router: Router::new(),
            grafts: Vec::new(),
            mounted: MountedRoutes::default(),
        }
    }

    fn at(mut self, path: &'static str, method: MethodRouter<Arc<AppState>>) -> Self {
        // axum keeps the last registration for a duplicated path, so a
        // double-mount silently drops handlers. The ledger would still
        // be right; the router would not.
        assert!(self.mounted.record(path), "route mounted twice: {path}");
        self.router = self.router.route(path, method);
        self
    }

    /// Record and merge a sub-router that carries its own state.
    fn graft(mut self, path: &'static str, sub: Router) -> Self {
        assert!(self.mounted.record(path), "route mounted twice: {path}");
        self.grafts.push(sub);
        self
    }

    /// Close the ledger and serve it. `/v1/capabilities` is recorded
    /// like any other route — a server that reports its surface should
    /// appear in the surface it reports.
    fn finish(self, profile: ServerProfile, state: Arc<AppState>) -> Router {
        let Mount {
            router,
            grafts,
            mut mounted,
        } = self;
        assert!(
            mounted.record(CAPABILITIES),
            "capabilities mounted twice: {CAPABILITIES}"
        );
        // Planning is served on every profile — it reads a source's
        // headers and builds nothing, so it is safe to offer publicly.
        // WHICH sources a profile will plan is the policy question, and
        // it lives in one place that both the handler and the
        // capability report ask (`capabilities::plans_source`).
        assert!(mounted.record(PLAN), "plan mounted twice: {PLAN}");

        let caps = Arc::new(Capabilities::derive(profile, mounted));
        let planning = Arc::new(crate::plan_service::PlanService::new(profile));
        let mut app = router
            .with_state(state)
            .merge(
                Router::new()
                    .route(CAPABILITIES, get(capabilities::handle_capabilities))
                    .with_state(caps),
            )
            .merge(
                Router::new()
                    .route(PLAN, post(plan::handle_plan))
                    .with_state(planning),
            );
        for graft in grafts {
            app = app.merge(graft);
        }
        app
    }
}

/// Build the router for the PUBLIC_EXPLORER profile: the read routes
/// plus `POST /v1/query`. Mutating and lifecycle routes are not
/// mounted — absent, not gated: a route that does not exist cannot be
/// mis-gated, and the statement surface's own gate lives inside the
/// LQL session (`CapabilityProfile::PublicExplorer`), not here.
pub fn public_explorer_router(
    state: Arc<AppState>,
    bridge: Arc<crate::lql_bridge::LqlBridge>,
) -> Router {
    Mount::new()
        .at(HEALTH, get(health::handle_health))
        .at(MODELS, get(models::handle_models))
        .at(MODEL_BY_ID, get(models::handle_model_retrieve))
        .at(DESCRIBE, get(describe::handle_describe))
        .at(WALK, get(walk::handle_walk))
        .at(RELATIONS, get(relations::handle_relations))
        .at(STATS, get(stats::handle_stats))
        .at(COMPONENTS, get(container_facts::handle_components))
        .at(
            REPRESENTATIONS,
            get(container_facts::handle_representations),
        )
        .at(PROVENANCE, get(container_facts::handle_provenance))
        .at(AUTHORITY, get(container_facts::handle_authority))
        .graft(
            QUERY,
            Router::new()
                .route(QUERY, post(query::handle_query))
                .with_state(bridge),
        )
        .finish(ServerProfile::PublicExplorer, state)
}

/// Build the router for single-model serving.
pub fn single_model_router(state: Arc<AppState>) -> Router {
    Mount::new()
        .at(DESCRIBE, get(describe::handle_describe))
        .at(WALK, get(walk::handle_walk))
        .at(SELECT, post(select::handle_select))
        .at(RELATIONS, get(relations::handle_relations))
        .at(STATS, get(stats::handle_stats))
        .at(INFER, post(infer::handle_infer))
        .at(SESSIONS, get(sessions::handle_list_sessions))
        .at(
            SESSION_BY_ID,
            get(sessions::handle_get_session).delete(sessions::handle_delete_session),
        )
        .at(PATCHES_APPLY, post(patches::handle_apply_patch))
        .at(PATCHES, get(patches::handle_list_patches))
        .at(PATCH_BY_NAME, delete(patches::handle_remove_patch))
        .at(WALK_FFN, post(walk_ffn::handle_walk_ffn))
        .at(WALK_FFN_Q8K, post(walk_ffn::handle_walk_ffn_q8k))
        .at(EXPERT_TOPOLOGY, get(topology::handle_topology))
        .at(
            EXPERT_BATCH,
            post(expert::handle_expert_batch).layer(DefaultBodyLimit::max(EXPERT_BATCH_BODY_LIMIT)),
        )
        .at(
            EXPERTS_LAYER_BATCH,
            post(expert::handle_experts_layer_batch)
                .layer(DefaultBodyLimit::max(EXPERT_BATCH_BODY_LIMIT)),
        )
        .at(
            EXPERTS_LAYER_BATCH_F16,
            post(expert::handle_experts_layer_batch_f16)
                .layer(DefaultBodyLimit::max(EXPERT_BATCH_BODY_LIMIT)),
        )
        .at(
            EXPERTS_MULTI_LAYER_BATCH,
            post(expert::handle_experts_multi_layer_batch)
                .layer(DefaultBodyLimit::max(EXPERT_BATCH_BODY_LIMIT)),
        )
        .at(
            EXPERTS_MULTI_LAYER_BATCH_Q8K,
            post(expert::handle_experts_multi_layer_batch_q8k)
                .layer(DefaultBodyLimit::max(EXPERT_BATCH_BODY_LIMIT)),
        )
        .at(EXPERT, post(expert::handle_expert))
        .at(EXPLAIN_INFER, post(explain::handle_explain))
        .at(INSERT, post(insert::handle_insert))
        .at(STREAM, get(stream::handle_stream))
        .at(HEALTH, get(health::handle_health))
        .at(RUNTIME, get(runtime::handle_runtime))
        // Dynamic model lifecycle — single-model topology only (0↔1
        // invariant, `docs/runtime-lifecycle-design.md` §7). Not present
        // on `multi_model_router`: that route table is sized for a
        // fixed boot-time model count with no slot for this to mutate.
        .at(
            RUNTIME_MODEL,
            post(runtime_lifecycle::handle_load_model)
                .delete(runtime_lifecycle::handle_unload_model),
        )
        .at(MODELS, get(models::handle_models))
        .at(WARMUP, post(warmup::handle_warmup))
        // Embed server endpoints (always available, required for --embed-only mode)
        .at(EMBED, post(embed::handle_embed))
        .at(EMBED_TOKEN, get(embed::handle_embed_single))
        .at(LOGITS, post(embed::handle_logits))
        .at(TOKEN_ENCODE, get(embed::handle_token_encode))
        .at(TOKEN_DECODE, get(embed::handle_token_decode))
        .at(SHARD, get(shard::handle_shard))
        .at(OPENAI_EMBEDDINGS, post(openai::handle_embeddings))
        .at(OPENAI_COMPLETIONS, post(openai::handle_completions))
        .at(
            OPENAI_CHAT_COMPLETIONS,
            post(openai::handle_chat_completions),
        )
        .at(OPENAI_RESPONSES, post(openai::handle_responses))
        .at(
            OPENAI_RESPONSE_BY_ID,
            get(openai::handle_get_response).delete(openai::handle_delete_response),
        )
        .at(MODEL_BY_ID, get(models::handle_model_retrieve))
        .finish(ServerProfile::SingleModel, state)
}

/// Build the router for multi-model serving.
pub fn multi_model_router(state: Arc<AppState>) -> Router {
    Mount::new()
        .at(HEALTH, get(health::handle_health))
        .at(RUNTIME, get(runtime::handle_runtime))
        .at(MODELS, get(models::handle_models))
        .at(M_DESCRIBE, get(describe::handle_describe_multi))
        .at(M_WALK, get(walk::handle_walk_multi))
        .at(M_SELECT, post(select::handle_select_multi))
        .at(M_RELATIONS, get(relations::handle_relations_multi))
        .at(M_STATS, get(stats::handle_stats_multi))
        .at(M_INFER, post(infer::handle_infer_multi))
        .at(SESSIONS, get(sessions::handle_list_sessions))
        .at(
            SESSION_BY_ID,
            get(sessions::handle_get_session).delete(sessions::handle_delete_session),
        )
        .at(M_PATCHES_APPLY, post(patches::handle_apply_patch_multi))
        .at(M_PATCHES, get(patches::handle_list_patches_multi))
        .at(M_PATCH_BY_NAME, delete(patches::handle_remove_patch_multi))
        .at(M_EXPLAIN_INFER, post(explain::handle_explain_multi))
        .at(M_INSERT, post(insert::handle_insert_multi))
        // Embed server endpoints for multi-model mode
        .at(M_EMBED, post(embed::handle_embed_multi))
        .at(M_EMBED_TOKEN, get(embed::handle_embed_single_multi))
        .at(M_LOGITS, post(embed::handle_logits_multi))
        .at(M_TOKEN_ENCODE, get(embed::handle_token_encode_multi))
        .at(M_TOKEN_DECODE, get(embed::handle_token_decode_multi))
        .at(SHARD, get(shard::handle_shard))
        // OpenAI-compat endpoints (multi-model: client passes `model` in body).
        .at(OPENAI_EMBEDDINGS, post(openai::handle_embeddings))
        .at(OPENAI_COMPLETIONS, post(openai::handle_completions))
        .at(
            OPENAI_CHAT_COMPLETIONS,
            post(openai::handle_chat_completions),
        )
        .at(OPENAI_RESPONSES, post(openai::handle_responses))
        .at(
            OPENAI_RESPONSE_BY_ID,
            get(openai::handle_get_response).delete(openai::handle_delete_response),
        )
        .at(MODEL_BY_ID, get(models::handle_model_retrieve))
        .finish(ServerProfile::MultiModel, state)
}
