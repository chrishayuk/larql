//! Stage and per-layer labels passed to `IndexBuildCallbacks`.
//!
//! Same pattern as `format::filenames`: every label that's emitted to
//! progress callbacks lives here as a `pub const`. Use these instead
//! of bare string literals.
//!
//! Why: a typo in `callbacks.on_stage(STAGE_GATE_VECTORS)` and the matching
//! `on_stage_done("gate_vectro")` causes silent event mismatch — tools
//! consuming the callbacks (progress bars, profilers, the bench rig)
//! never see the close event. Centralising means a typo is a compile
//! error.
//!
//! Two flavours:
//! - **Stage labels** (`STAGE_*`) — passed to `on_stage` /
//!   `on_stage_done`. One per major pipeline phase.
//! - **Component labels** (`COMP_*`) — passed to `on_layer_start` /
//!   `on_layer_done` / `on_feature_progress`. One per per-layer
//!   component the writers track.

// ── Stage labels (`on_stage` / `on_stage_done`) ───────────────────────

/// `loading` — opening + mmap'ing safetensors shards.
pub const STAGE_LOADING: &str = "loading";
/// `gate_vectors` — write `gate_vectors.bin`.
pub const STAGE_GATE_VECTORS: &str = "gate_vectors";
/// `router_weights` — MoE router weights write.
pub const STAGE_ROUTER_WEIGHTS: &str = "router_weights";
/// `embeddings` — write `embeddings.bin`.
pub const STAGE_EMBEDDINGS: &str = "embeddings";
/// `down_meta` — extract per-feature top-K and write `down_meta.bin`.
pub const STAGE_DOWN_META: &str = "down_meta";
/// `tokenizer` — write `tokenizer.json`.
pub const STAGE_TOKENIZER: &str = "tokenizer";
/// `model_weights` — f32 / Q4_0 model weight serialisation.
pub const STAGE_MODEL_WEIGHTS: &str = "model_weights";
/// `model_weights_kquant` — streaming Q4_K/Q6_K weight serialisation.
pub const STAGE_MODEL_WEIGHTS_KQUANT: &str = "model_weights_kquant";
/// `relation_clusters` — cluster discovery + `relation_clusters.json` write.
pub const STAGE_RELATION_CLUSTERS: &str = "relation_clusters";

// ── Component labels (`on_layer_start` / `on_layer_done`) ─────────────

/// `gate` — per-layer gate vector extraction.
pub const COMP_GATE: &str = "gate";
/// `down` — per-layer down-meta extraction.
pub const COMP_DOWN: &str = "down";
/// `attn_weights` — f32 attention weight write per layer.
pub const COMP_ATTN_WEIGHTS: &str = "attn_weights";
/// `up/down_weights` — f32 FFN up/down weight write per layer.
pub const COMP_UP_DOWN_WEIGHTS: &str = "up/down_weights";
/// `attn_kquant` — Q4_K/Q6_K attention weight write per layer.
pub const COMP_ATTN_KQUANT: &str = "attn_kquant";
/// `ffn_kquant` — Q4_K/Q6_K FFN weight write per layer.
pub const COMP_FFN_KQUANT: &str = "ffn_kquant";

#[cfg(test)]
mod tests {
    use super::*;

    /// Labels must be unique — a duplicate would silently route two
    /// progress streams under the same name.
    #[test]
    fn all_labels_unique() {
        let labels = [
            STAGE_LOADING,
            STAGE_GATE_VECTORS,
            STAGE_ROUTER_WEIGHTS,
            STAGE_EMBEDDINGS,
            STAGE_DOWN_META,
            STAGE_TOKENIZER,
            STAGE_MODEL_WEIGHTS,
            STAGE_MODEL_WEIGHTS_KQUANT,
            STAGE_RELATION_CLUSTERS,
            COMP_GATE,
            COMP_DOWN,
            COMP_ATTN_WEIGHTS,
            COMP_UP_DOWN_WEIGHTS,
            COMP_ATTN_KQUANT,
            COMP_FFN_KQUANT,
        ];
        let unique: std::collections::HashSet<_> = labels.iter().collect();
        assert_eq!(unique.len(), labels.len(), "duplicate stage label");
    }
}
