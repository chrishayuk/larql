//! Every URL path this server can mount, in one place.
//!
//! Extracted from `routes/mod.rs` so that
//! [`crate::capabilities`] and the router build from the *same*
//! strings. A capability is advertised by naming the route that
//! serves it, and a route is mounted by naming the same constant —
//! two readings of one fact rather than two facts that agree by
//! discipline.
//!
//! Some paths here are declared but mounted by nobody yet
//! ([`PLAN`], [`ENCODE`], [`RESIDENCY`]). That is deliberate and is
//! not a stub: `/v1/capabilities` reports a capability as present iff
//! its route was actually mounted, so an unmounted path reports
//! `false` on every profile today, and the rung that mounts it flips
//! the advertised flag without editing the capability table at all.

pub(crate) const HEALTH: &str = "/v1/health";
pub(crate) const RUNTIME: &str = "/v1/runtime";
pub(crate) const RUNTIME_MODEL: &str = "/v1/runtime/model";
pub(crate) const MODELS: &str = "/v1/models";
pub(crate) const DESCRIBE: &str = "/v1/describe";
pub(crate) const WALK: &str = "/v1/walk";
pub(crate) const SELECT: &str = "/v1/select";
pub(crate) const RELATIONS: &str = "/v1/relations";
pub(crate) const STATS: &str = "/v1/stats";
pub(crate) const INFER: &str = "/v1/infer";
pub(crate) const SESSIONS: &str = "/v1/sessions";
pub(crate) const SESSION_BY_ID: &str = "/v1/sessions/{session_id}";
pub(crate) const PATCHES_APPLY: &str = "/v1/patches/apply";
pub(crate) const PATCHES: &str = "/v1/patches";
pub(crate) const PATCH_BY_NAME: &str = "/v1/patches/{name}";
pub(crate) const WALK_FFN: &str = "/v1/walk-ffn";
pub(crate) const WALK_FFN_Q8K: &str = "/v1/walk-ffn-q8k";
pub(crate) const EXPERT_TOPOLOGY: &str = "/v1/expert/topology";
pub(crate) const EXPERT_BATCH: &str = "/v1/expert/batch";
pub(crate) const EXPERTS_LAYER_BATCH: &str = "/v1/experts/layer-batch";
pub(crate) const EXPERTS_LAYER_BATCH_F16: &str = "/v1/experts/layer-batch-f16";
pub(crate) const EXPERTS_MULTI_LAYER_BATCH: &str = "/v1/experts/multi-layer-batch";
pub(crate) const EXPERTS_MULTI_LAYER_BATCH_Q8K: &str = "/v1/experts/multi-layer-batch-q8k";
pub(crate) const EXPERT: &str = "/v1/expert/{layer}/{expert_id}";
pub(crate) const EXPLAIN_INFER: &str = "/v1/explain-infer";
pub(crate) const INSERT: &str = "/v1/insert";
pub(crate) const STREAM: &str = "/v1/stream";
pub(crate) const WARMUP: &str = "/v1/warmup";
pub(crate) const EMBED: &str = "/v1/embed";
pub(crate) const EMBED_TOKEN: &str = "/v1/embed/{token_id}";
pub(crate) const LOGITS: &str = "/v1/logits";
pub(crate) const TOKEN_ENCODE: &str = "/v1/token/encode";
pub(crate) const TOKEN_DECODE: &str = "/v1/token/decode";
// Mode B shard handoff: donor streams its on-disk vindex as a tar so a
// freshly-assigned spare server can mirror the shard locally.
pub(crate) const SHARD: &str = "/v1/shard/{model_id}/{range}";
pub(crate) const QUERY: &str = "/v1/query";
pub(crate) const COMPONENTS: &str = "/v1/components";
pub(crate) const REPRESENTATIONS: &str = "/v1/representations";
pub(crate) const PROVENANCE: &str = "/v1/provenance";
pub(crate) const AUTHORITY: &str = "/v1/authority";

pub(crate) const OPENAI_EMBEDDINGS: &str = "/v1/embeddings";
pub(crate) const OPENAI_COMPLETIONS: &str = "/v1/completions";
pub(crate) const OPENAI_CHAT_COMPLETIONS: &str = "/v1/chat/completions";
pub(crate) const OPENAI_RESPONSES: &str = "/v1/responses";
pub(crate) const OPENAI_RESPONSE_BY_ID: &str = "/v1/responses/{response_id}";
pub(crate) const MODEL_BY_ID: &str = "/v1/models/{model}";

pub(crate) const M_DESCRIBE: &str = "/v1/{model_id}/describe";
pub(crate) const M_WALK: &str = "/v1/{model_id}/walk";
pub(crate) const M_SELECT: &str = "/v1/{model_id}/select";
pub(crate) const M_RELATIONS: &str = "/v1/{model_id}/relations";
pub(crate) const M_STATS: &str = "/v1/{model_id}/stats";
pub(crate) const M_INFER: &str = "/v1/{model_id}/infer";
pub(crate) const M_PATCHES_APPLY: &str = "/v1/{model_id}/patches/apply";
pub(crate) const M_PATCHES: &str = "/v1/{model_id}/patches";
pub(crate) const M_PATCH_BY_NAME: &str = "/v1/{model_id}/patches/{name}";
pub(crate) const M_EXPLAIN_INFER: &str = "/v1/{model_id}/explain-infer";
pub(crate) const M_INSERT: &str = "/v1/{model_id}/insert";
pub(crate) const M_EMBED: &str = "/v1/{model_id}/embed";
pub(crate) const M_EMBED_TOKEN: &str = "/v1/{model_id}/embed/{token_id}";
pub(crate) const M_LOGITS: &str = "/v1/{model_id}/logits";
pub(crate) const M_TOKEN_ENCODE: &str = "/v1/{model_id}/token/encode";
pub(crate) const M_TOKEN_DECODE: &str = "/v1/{model_id}/token/decode";

// ── Declared, not yet mounted ────────────────────────────────────────
// The Explorer contract's remaining verbs. See the module doc: these
// exist so the capability table can name a route for a capability the
// server does not have yet, and answer `false` from the ledger rather
// than from a hand-maintained "not supported" list.

/// `POST /v1/plan` — judge a *source* (a raw checkpoint) for
/// architecture support from its headers, without encoding it.
/// Explorer contract step 4.
pub(crate) const PLAN: &str = "/v1/plan";

/// `POST /v1/encode` — compile a source checkpoint into a VINDEX3
/// container. No profile mounts this; on a public deployment it never
/// should.
pub(crate) const ENCODE: &str = "/v1/encode";

/// `GET /v1/residency` — the described/resident/remote ledger per
/// object, read from the same ledger execution uses. Explorer
/// contract step 7.
pub(crate) const RESIDENCY: &str = "/v1/residency";

/// `GET /v1/capabilities` — this contract's own front door: what this
/// server will and will not do, enforced here rather than guessed by
/// the client from a hostname.
pub(crate) const CAPABILITIES: &str = "/v1/capabilities";
