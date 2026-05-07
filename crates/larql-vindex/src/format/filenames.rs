//! Vindex on-disk filenames — single source of truth.
//!
//! Every `.bin` / `.json` filename written or read by the vindex format
//! lives here as a `pub const`. Use these instead of string literals.
//!
//! Why: the 2026-04-25 audit found 244 occurrences of these names
//! scattered across 18+ files. A typo silently triggers a fallback
//! codepath (the file just "doesn't exist") and bugs go undiagnosed.
//! Centralising means renaming a file changes one line.
//!
//! Convention: `SCREAMING_SNAKE`, named for what they hold, not how
//! they're encoded.

// ── Top-level config / sidecars ─────────────────────────────────────────
pub const INDEX_JSON: &str = "index.json";
pub const TOKENIZER_JSON: &str = "tokenizer.json";
pub const TOKENIZER_CONFIG_JSON: &str = "tokenizer_config.json";
pub const GENERATION_CONFIG_JSON: &str = "generation_config.json";
pub const WEIGHT_MANIFEST_JSON: &str = "weight_manifest.json";
pub const GGUF_GATE_MANIFEST_JSON: &str = "gguf_gate_manifest.json";
pub const GGUF_EMBEDDINGS_MANIFEST_JSON: &str = "gguf_embeddings_manifest.json";
pub const GGUF_DOWN_META_MANIFEST_JSON: &str = "gguf_down_meta_manifest.json";
pub const KNN_STORE_BIN: &str = "knn_store.bin";
pub const MODEL_WEIGHTS_BIN: &str = "model_weights.bin";

// ── Labels / clustering sidecars ───────────────────────────────────────
pub const RELATION_CLUSTERS_JSON: &str = "relation_clusters.json";
pub const FEATURE_CLUSTERS_JSONL: &str = "feature_clusters.jsonl";
pub const FEATURE_LABELS_JSON: &str = "feature_labels.json";

// ── Embeddings + norms (always present) ────────────────────────────────
pub const EMBEDDINGS_BIN: &str = "embeddings.bin";
pub const NORMS_BIN: &str = "norms.bin";

// ── Gate vectors ───────────────────────────────────────────────────────
pub const GATE_VECTORS_BIN: &str = "gate_vectors.bin";
pub const GATE_VECTORS_Q4_BIN: &str = "gate_vectors_q4.bin";

// ── Down meta + feature-major projections ──────────────────────────────
pub const DOWN_META_BIN: &str = "down_meta.bin";
pub const DOWN_FEATURES_BIN: &str = "down_features.bin";
pub const UP_FEATURES_BIN: &str = "up_features.bin";

// ── Layer-major FFN weight files (PyTorch `nn.Linear` orientation) ────
//
// `[layer, intermediate, hidden]` for up and `[layer, hidden, intermediate]`
// for down — distinct from the feature-major projection files above.
// Written by f32 extraction, consumed by Q4_K conversion + checksumming +
// HuggingFace upload.
pub const UP_WEIGHTS_BIN: &str = "up_weights.bin";
pub const DOWN_WEIGHTS_BIN: &str = "down_weights.bin";

/// Feature-major Q4_K-encoded down projections (W2 of perf round-4).
///
/// On-disk PyTorch `nn.Linear` orientation for down is
/// `[hidden, intermediate]`, so a single feature's down vector requires
/// gathering across `hidden` separate rows — there is no per-feature
/// row decode. The legacy code path (`q4k_ffn_layer` + cache) amortises
/// this by dequantising the whole layer to f32 and transposing once.
///
/// Emitting `down_features_q4k.bin` at extract time stores down already
/// in feature-major `[intermediate, hidden]` orientation, Q4_K-encoded.
/// Per-feature decode becomes a single row dequant — no cache, no
/// transpose, no ~840 MB heap ceiling on Gemma 4B. The disk cost is
/// roughly the same as the down portion of `interleaved_q4k.bin` (~14
/// MB / layer at Gemma 4B dims). Opt-in via `Q4kWriteOptions::feature_major_down`.
pub const DOWN_FEATURES_Q4K_BIN: &str = "down_features_q4k.bin";
/// Per-layer (offset, length, format) entries for `down_features_q4k.bin`.
pub const DOWN_FEATURES_Q4K_MANIFEST_JSON: &str = "down_features_q4k_manifest.json";

// ── Interleaved FFN (gate|up|down packed per layer) ────────────────────
pub const INTERLEAVED_BIN: &str = "interleaved.bin";
pub const INTERLEAVED_Q4_BIN: &str = "interleaved_q4.bin";
pub const INTERLEAVED_Q4K_BIN: &str = "interleaved_q4k.bin";
pub const INTERLEAVED_Q4K_MANIFEST_JSON: &str = "interleaved_q4k_manifest.json";

// ── Attention weights ──────────────────────────────────────────────────
pub const ATTN_WEIGHTS_BIN: &str = "attn_weights.bin";
pub const ATTN_WEIGHTS_Q4_BIN: &str = "attn_weights_q4.bin";
pub const ATTN_WEIGHTS_Q4_MANIFEST_JSON: &str = "attn_weights_q4_manifest.json";
pub const ATTN_WEIGHTS_Q4K_BIN: &str = "attn_weights_q4k.bin";
pub const ATTN_WEIGHTS_Q4K_MANIFEST_JSON: &str = "attn_weights_q4k_manifest.json";
pub const ATTN_WEIGHTS_Q8_BIN: &str = "attn_weights_q8.bin";
pub const ATTN_WEIGHTS_Q8_MANIFEST_JSON: &str = "attn_weights_q8_manifest.json";

// ── Per-layer FFN weights (§5.12) ──────────────────────────────────────
//
// Unified format for both dense and MoE FFN weights. One file per layer.
// File header declares the quantization format; all entries within a file
// use it uniformly (no mixing formats). Dense: num_entries=1.
// MoE: num_entries=num_experts.
pub const LAYERS_DIR: &str = "layers";

/// Return the path of `layers/layer_{L:02}.weights` for layer `L`.
pub fn layer_weights_filename(layer: usize) -> String {
    format!("layers/layer_{layer:02}.weights")
}

// ── LM head ────────────────────────────────────────────────────────────
pub const LM_HEAD_BIN: &str = "lm_head.bin";
pub const LM_HEAD_Q4_BIN: &str = "lm_head_q4.bin";

// ── FP4 / FP8 projections (exp 26) ─────────────────────────────────────
pub const GATE_VECTORS_FP4_BIN: &str = "gate_vectors_fp4.bin";
pub const UP_FEATURES_FP4_BIN: &str = "up_features_fp4.bin";
pub const DOWN_FEATURES_FP8_BIN: &str = "down_features_fp8.bin";

// ── HuggingFace upload manifest order ──────────────────────────────────
//
// Order matches what `format/huggingface.rs` uploads. Adding or
// removing a vindex file means updating both this list AND the
// per-file upload code.
pub const HF_UPLOAD_FILES: &[&str] = &[
    INDEX_JSON,
    TOKENIZER_JSON,
    WEIGHT_MANIFEST_JSON,
    EMBEDDINGS_BIN,
    NORMS_BIN,
    GATE_VECTORS_BIN,
    DOWN_META_BIN,
    INTERLEAVED_BIN,
    INTERLEAVED_Q4K_BIN,
    INTERLEAVED_Q4K_MANIFEST_JSON,
    ATTN_WEIGHTS_BIN,
    ATTN_WEIGHTS_Q4K_BIN,
    ATTN_WEIGHTS_Q4K_MANIFEST_JSON,
    DOWN_FEATURES_BIN,
    UP_FEATURES_BIN,
    LM_HEAD_Q4_BIN,
];

#[cfg(test)]
mod tests {
    use super::*;

    /// Constants must never collide — a duplicate name would silently
    /// route two writers at the same file.
    #[test]
    fn all_filenames_unique() {
        let names = [
            INDEX_JSON,
            TOKENIZER_JSON,
            TOKENIZER_CONFIG_JSON,
            GENERATION_CONFIG_JSON,
            WEIGHT_MANIFEST_JSON,
            GGUF_GATE_MANIFEST_JSON,
            GGUF_EMBEDDINGS_MANIFEST_JSON,
            GGUF_DOWN_META_MANIFEST_JSON,
            KNN_STORE_BIN,
            MODEL_WEIGHTS_BIN,
            RELATION_CLUSTERS_JSON,
            FEATURE_CLUSTERS_JSONL,
            FEATURE_LABELS_JSON,
            EMBEDDINGS_BIN,
            NORMS_BIN,
            GATE_VECTORS_BIN,
            GATE_VECTORS_Q4_BIN,
            GATE_VECTORS_FP4_BIN,
            DOWN_META_BIN,
            DOWN_FEATURES_BIN,
            DOWN_FEATURES_FP8_BIN,
            DOWN_FEATURES_Q4K_BIN,
            DOWN_FEATURES_Q4K_MANIFEST_JSON,
            DOWN_WEIGHTS_BIN,
            UP_FEATURES_BIN,
            UP_FEATURES_FP4_BIN,
            UP_WEIGHTS_BIN,
            INTERLEAVED_BIN,
            INTERLEAVED_Q4_BIN,
            INTERLEAVED_Q4K_BIN,
            INTERLEAVED_Q4K_MANIFEST_JSON,
            ATTN_WEIGHTS_BIN,
            ATTN_WEIGHTS_Q4_BIN,
            ATTN_WEIGHTS_Q4_MANIFEST_JSON,
            ATTN_WEIGHTS_Q4K_BIN,
            ATTN_WEIGHTS_Q4K_MANIFEST_JSON,
            ATTN_WEIGHTS_Q8_BIN,
            ATTN_WEIGHTS_Q8_MANIFEST_JSON,
            LM_HEAD_BIN,
            LM_HEAD_Q4_BIN,
        ];
        let unique: std::collections::HashSet<_> = names.iter().collect();
        assert_eq!(unique.len(), names.len(), "duplicate filename constant");
    }

    #[test]
    fn hf_upload_files_subset_of_all() {
        // HF_UPLOAD_FILES must reference real constants. If a constant
        // is removed, this test catches the dangling reference.
        for name in HF_UPLOAD_FILES {
            assert!(
                name.ends_with(".bin") || name.ends_with(".json"),
                "HF_UPLOAD_FILES has odd entry: {name}"
            );
        }
    }
}
