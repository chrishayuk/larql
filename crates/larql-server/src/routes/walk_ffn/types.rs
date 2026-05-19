//! Public types for the walk-ffn endpoint and the in-process RAII guard
//! that tracks `requests_in_flight` for GT6 drain.
//!
//! Split out of the previous monolithic `walk_ffn.rs` so the request
//! shape + binary constants are reachable from the codec, validators,
//! and handler without circular imports.

use serde::Deserialize;

/// RAII guard that decrements the `requests_in_flight` counter on drop.
/// Used by [`super::handler::handle_walk_ffn`] so the GT6 drain protocol
/// (ADR-0011 §Phase B2) sees an accurate in-flight count even when the
/// handler errors out before sending a response.
pub(crate) struct RifGuard(pub(crate) std::sync::Arc<std::sync::atomic::AtomicU32>);

impl Drop for RifGuard {
    fn drop(&mut self) {
        use std::sync::atomic::Ordering;
        // Saturating sub to avoid wrapping if something incremented 0 and dropped twice.
        let prev = self
            .0
            .fetch_update(Ordering::Release, Ordering::Relaxed, |v| {
                Some(v.saturating_sub(1))
            });
        let _ = prev;
    }
}

pub(crate) const BINARY_CT: &str = "application/x-larql-ffn";
pub(crate) const BATCH_MARKER: u32 = 0xFFFF_FFFF;

#[derive(Deserialize)]
pub struct WalkFfnRequest {
    /// Single layer mode.
    #[serde(default)]
    pub layer: Option<usize>,
    /// Batched mode — multiple layers in one request.
    #[serde(default)]
    pub layers: Option<Vec<usize>>,
    /// Residual vector(s), row-major flat. Length must be `seq_len *
    /// hidden_size`. Features-only mode requires `seq_len == 1` (only the
    /// first `hidden_size` elements are consulted).
    pub residual: Vec<f32>,
    /// Sequence length — number of residual rows in the flat `residual`
    /// array. Defaults to 1. Ignored in features-only mode.
    #[serde(default = "default_seq_len")]
    pub seq_len: usize,
    /// Top-K features to select. Ignored in `full_output` mode (WalkFfn uses
    /// its own unlimited-K default there).
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    /// When true, return the computed FFN output vector per layer instead of
    /// feature indices + scores. Requires loadable model weights.
    #[serde(default)]
    pub full_output: bool,
    /// When true, `residual` is `h_post_attn` (post-attention, pre-norm). The
    /// server runs the full hybrid MoE layer: dense-FFN + remote expert dispatch
    /// + combine + outer norm. Requires `full_output: true` and the server to
    ///   have `--moe-shards` configured.
    #[serde(default)]
    pub moe_layer: bool,
}

fn default_seq_len() -> usize {
    1
}
fn default_top_k() -> usize {
    8092
}

// ── Typed output structs (shared by JSON + binary encoders) ──────────────────

pub(crate) struct FfnEntry {
    pub(crate) layer: usize,
    pub(crate) output: Vec<f32>,
}

pub(crate) struct FfnOutput {
    pub(crate) entries: Vec<FfnEntry>,
    pub(crate) seq_len: usize,
    pub(crate) latency_ms: f64,
}
