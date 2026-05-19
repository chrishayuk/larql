//! `RemoteMoeBackend` — Mixture-of-Experts weight-shard dispatch over HTTP.
//!
//! Not to be confused with [`crate::experts`] — that module hosts deterministic
//! WASM compute experts (gcd, base64, …). This module dispatches *MoE expert
//! weights* (the FFN sub-blocks of an MoE transformer) to remote shard servers.
//!
//! For hybrid MoE models (e.g. Gemma 4 26B A4B), the client holds attention
//! weights + router weights (~5.5 GB). Expert weights live on remote shard
//! servers. For each layer:
//!
//!   1. Client runs the router locally: norm → scale → proj → softmax → top-K.
//!   2. Client groups selected experts by shard.
//!   3. One `POST /v1/expert/batch` per shard (parallel via rayon).
//!   4. Client assembles weighted sum from responses.
//!
//! Wire format: JSON — `{"requests": [{layer, expert_id, residual}]}`
//!              → `{"results": [{layer, expert_id, output}], "latency_ms": f64}`
//!
//! This mirrors [`crate::ffn::RemoteWalkBackend`] at the MoE level, replacing
//! `POST /v1/walk-ffn` with `POST /v1/expert/batch`.
//!
//! # Shard map
//!
//! Expert IDs are contiguous ranges owned by each shard:
//!
//! ```text
//! "0-31"  → https://shard-a.local:8081
//! "32-63" → https://shard-b.local:8082
//! ```
//!
//! A single-shard setup (`"0-63"`) routes all experts to one server.
//! `reshard()` swaps the map live without reloading the model.
//!
//! # Module layout (post-2026-05-02 split from a 2,691-line single file):
//!
//! - [`error`][] — `RemoteMoeError`.
//! - [`config`][] — `ShardConfig`, `UnitManifest`, `UnitShard`,
//!   `parse_unit_manifest`.
//! - [`wire`][] — binary encode/decode helpers + `ExpertCallItem` /
//!   `ExpertResultItem` payload types.
//! - [`router`][] — client-side routing math (`MoeRouterWeights`, `rms_norm`).
//! - [`shard`][] — internal `Shard` struct + per-transport (HTTP / UDS /
//!   gRPC) dispatch logic.
//! - [`stream`][] — `ShardStream` (gRPC bi-di) + `InflightMoe` (the fire /
//!   collect handle).
//! - [`backend`][] — the public `RemoteMoeBackend`.

mod backend;
mod config;
mod error;
pub(crate) mod metrics;
pub mod multi_layer_wire;
mod router;
pub(crate) mod runtime;
mod shard;
mod stream;
mod wire;

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;

// ── Public re-exports (preserve the pre-split crate-public API) ──────────────

pub use backend::RemoteMoeBackend;
pub use config::{parse_unit_manifest, ShardConfig, UnitManifest, UnitShard};
pub use error::RemoteMoeError;
pub use multi_layer_wire::{
    decode_multi_layer_request, decode_multi_layer_request_q8k, decode_multi_layer_response,
    encode_multi_layer_request, encode_multi_layer_request_q8k, encode_multi_layer_response,
    MultiLayerResult, MultiLayerTask, MultiLayerTaskQ8K, MULTI_LAYER_BATCH_CONTENT_TYPE,
    MULTI_LAYER_BATCH_Q8K_CONTENT_TYPE,
};
pub use router::MoeRouterWeights;
pub use stream::{InflightMoe, ShardStream};
pub use wire::{
    decode_expert_request, decode_expert_response, decode_layer_batch_request,
    decode_layer_batch_request_f16, decode_layer_batch_response, decode_layer_batch_response_f16,
    encode_expert_request, encode_expert_response, encode_layer_batch_request,
    encode_layer_batch_request_f16, encode_layer_batch_response, encode_layer_batch_response_f16,
    ExpertCallItem, ExpertResultItem, EXPERT_BINARY_CONTENT_TYPE, LAYER_BATCH_CONTENT_TYPE,
    LAYER_BATCH_F16_CONTENT_TYPE,
};
