//! Read model weights back from a `.vindex` directory.
//!
//! Mirror of `super::write` — reconstructs `ModelWeights` from the
//! split `attn_weights.bin` / `up_weights.bin` / `down_weights.bin` /
//! `norms.bin` / `lm_head.bin` files using the architecture metadata
//! recorded in `index.json`.
//!
//! Structure (round-5, 2026-05-09): `mod.rs` holds the public API
//! (`LoadWeightsOptions`, `load_model_weights*`, `load_model_weights_kquant*`,
//! `find_tokenizer_path`, `expert_in_shard`) plus tests. The two heavy
//! loader bodies live in:
//! - [`f32`] — `load_model_weights_with_opts`
//! - [`q4k`] — `load_model_weights_kquant_shard`

mod arch;
mod embeddings;
mod f32;
mod q4k;

use std::path::Path;

use larql_models::ModelWeights;

use crate::error::VindexError;
use crate::format::filenames::*;
use crate::index::core::IndexLoadCallbacks;

/// Whether expert `e` is owned by a shard with `expert_filter`. `None`
/// means "no shard, keep all experts". `Some((start, end_excl))` is a
/// half-open range — `e == start` is kept, `e == end_excl` is skipped.
///
/// Pulled out as a free function so the boundary behaviour can be unit
/// tested directly without standing up a Q4_K MoE fixture.
pub(crate) fn expert_in_shard(e: usize, expert_filter: Option<(usize, usize)>) -> bool {
    match expert_filter {
        None => true,
        Some((start, end_excl)) => e >= start && e < end_excl,
    }
}

/// Options for [`load_model_weights_with_opts`]. Filter which
/// component tensors are actually mmap'd + decoded at load time —
/// unlike the post-load `drop_*` helpers on `ModelWeights`, these
/// options mean we never allocate the f32 heap in the first place, so
/// the process RSS genuinely drops.
#[derive(Default, Clone, Copy, Debug)]
pub struct LoadWeightsOptions {
    /// Skip attention weight tensors (Q / K / V / O projections +
    /// q_norm / k_norm). Used by `larql serve --ffn-only` — the
    /// client holds attention locally, the server doesn't need it.
    pub skip_attn: bool,
    /// Skip FFN weight tensors (gate / up / down projections).
    /// Used by clients running `--ffn URL` — the remote server holds
    /// those, the local heap shouldn't carry them.
    pub skip_ffn: bool,
    /// Skip `lm_head` (and any `lm_head_q4.bin` rebuild). Used by
    /// servers that don't compute logits.
    pub skip_lm_head: bool,
    /// Skip the input embedding matrix. Used by servers that only
    /// receive residual vectors, not token IDs.
    pub skip_embed: bool,
}

impl LoadWeightsOptions {
    /// Pattern match for FFN weight keys (matches
    /// [`ModelWeights::drop_ffn_weights`] so the two strategies stay
    /// in sync).
    pub(super) fn is_ffn_key(key: &str) -> bool {
        const FFN_PATTERNS: &[&str] = &[
            "gate_proj",
            "up_proj",
            "down_proj",
            "ffn_gate",
            "ffn_up",
            "ffn_down",
            "mlp.experts",
            "block_sparse_moe.experts",
            "packed_gate_up_blocks",
            "packed_down_blocks",
        ];
        FFN_PATTERNS.iter().any(|p| key.contains(p))
    }

    /// Pattern match for attention weight keys (matches
    /// [`ModelWeights::drop_attn_weights`]).
    pub(super) fn is_attn_key(key: &str) -> bool {
        const ATTN_PATTERNS: &[&str] = &[
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "attn_q",
            "attn_k",
            "attn_v",
            "attn_o",
            "q_norm",
            "k_norm",
        ];
        ATTN_PATTERNS.iter().any(|p| key.contains(p))
    }

    pub(super) fn should_skip(&self, key: &str) -> bool {
        if self.skip_ffn && Self::is_ffn_key(key) {
            return true;
        }
        if self.skip_attn && Self::is_attn_key(key) {
            return true;
        }
        if self.skip_lm_head && key == "lm_head.weight" {
            return true;
        }
        false
    }
}

/// Load a full `ModelWeights` from a vindex directory (no filtering).
pub fn load_model_weights(
    dir: &Path,
    callbacks: &mut dyn IndexLoadCallbacks,
) -> Result<ModelWeights, VindexError> {
    load_model_weights_with_opts(dir, callbacks, LoadWeightsOptions::default())
}

/// Load `ModelWeights` from a vindex directory, skipping component
/// tensors per [`LoadWeightsOptions`]. Body in [`f32::load_model_weights_with_opts`].
pub fn load_model_weights_with_opts(
    dir: &Path,
    callbacks: &mut dyn IndexLoadCallbacks,
    opts: LoadWeightsOptions,
) -> Result<ModelWeights, VindexError> {
    f32::load_model_weights_with_opts(dir, callbacks, opts)
}

/// Load the minimum ModelWeights needed to drive a Q4_K vindex forward pass.
///
/// Q4 vindexes store attn / FFN weights as packed blocks in
/// `attn_weights_q4k.bin` and `interleaved_kquant.bin`; the forward pass reads
/// those through [`crate::index::VectorIndex::attn_kquant_layer_data`] /
/// [`crate::index::VectorIndex::interleaved_kquant_layer_data`] and
/// dequantises on demand, so the `ModelWeights.tensors` map stays empty.
pub fn load_model_weights_kquant(
    dir: &Path,
    callbacks: &mut dyn IndexLoadCallbacks,
) -> Result<ModelWeights, VindexError> {
    load_model_weights_kquant_shard(dir, callbacks, None)
}

/// Expert-shard variant of [`load_model_weights_kquant`].
///
/// Identical to the full loader except that when `expert_filter` is `Some((start,
/// end_excl))`, per-layer expert entries outside `[start, end_excl)` are not
/// inserted into `packed_byte_ranges`. Body in
/// [`q4k::load_model_weights_kquant_shard`].
pub fn load_model_weights_kquant_shard(
    dir: &Path,
    callbacks: &mut dyn IndexLoadCallbacks,
    expert_filter: Option<(usize, usize)>,
) -> Result<ModelWeights, VindexError> {
    q4k::load_model_weights_kquant_shard(dir, callbacks, expert_filter)
}

/// Find the tokenizer path near a model or vindex directory.
pub fn find_tokenizer_path(dir: &Path) -> Option<std::path::PathBuf> {
    let p = dir.join(TOKENIZER_JSON);
    if p.exists() {
        return Some(p);
    }
    if let Some(parent) = dir.parent() {
        let p = parent.join(TOKENIZER_JSON);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expert_in_shard_none_keeps_everything() {
        // No filter ⇒ every expert is owned. Pin this so a future
        // refactor doesn't accidentally invert the meaning of `None`.
        for e in [0, 1, 7, 127, usize::MAX] {
            assert!(expert_in_shard(e, None), "e={e} should be kept");
        }
    }

    #[test]
    fn expert_in_shard_half_open_boundaries() {
        // [4, 8): 4..=7 kept, 3 and 8 dropped.
        let f = Some((4, 8));
        assert!(!expert_in_shard(3, f));
        assert!(expert_in_shard(4, f), "start is inclusive");
        assert!(expert_in_shard(7, f));
        assert!(!expert_in_shard(8, f), "end_excl is exclusive");
        assert!(!expert_in_shard(9, f));
    }

    #[test]
    fn expert_in_shard_empty_range_keeps_nothing() {
        // start == end_excl is a valid (empty) shard, e.g. a worker
        // configured with `--experts 4-4`. Nothing should slip through.
        for e in [0, 4, 5, 100] {
            assert!(!expert_in_shard(e, Some((4, 4))));
        }
    }

    #[test]
    fn expert_in_shard_zero_based_first_shard() {
        // The Gemma-4-26B "first 16 experts" case: shard (0, 16).
        let f = Some((0, 16));
        assert!(expert_in_shard(0, f));
        assert!(expert_in_shard(15, f));
        assert!(!expert_in_shard(16, f));
        assert!(!expert_in_shard(127, f));
    }

    // ─── LoadWeightsOptions pattern matchers ───────────────────────

    #[test]
    fn is_ffn_key_matches_dense_and_moe_patterns() {
        for key in [
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_down.weight",
            "model.layers.0.mlp.experts.0.gate_proj.weight",
            "model.layers.0.block_sparse_moe.experts.0.w1",
            "model.layers.0.mlp.experts.packed_gate_up_blocks",
            "model.layers.0.mlp.experts.packed_down_blocks",
        ] {
            assert!(LoadWeightsOptions::is_ffn_key(key), "missed: {key}");
        }
    }

    #[test]
    fn is_ffn_key_rejects_attn_and_norms() {
        for key in [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.input_layernorm.weight",
            "model.embed_tokens.weight",
            "lm_head.weight",
        ] {
            assert!(
                !LoadWeightsOptions::is_ffn_key(key),
                "false positive: {key}"
            );
        }
    }

    #[test]
    fn is_attn_key_matches_qkvo_and_norms() {
        for key in [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_v.weight",
            "blk.0.attn_o.weight",
            "model.layers.0.self_attn.q_norm.weight",
            "model.layers.0.self_attn.k_norm.weight",
        ] {
            assert!(LoadWeightsOptions::is_attn_key(key), "missed: {key}");
        }
    }

    #[test]
    fn is_attn_key_rejects_ffn_and_embed() {
        for key in [
            "model.layers.0.mlp.gate_proj.weight",
            "model.embed_tokens.weight",
            "model.layers.0.input_layernorm.weight",
        ] {
            assert!(
                !LoadWeightsOptions::is_attn_key(key),
                "false positive: {key}"
            );
        }
    }

    #[test]
    fn should_skip_routes_through_per_component_predicates() {
        let opts = LoadWeightsOptions {
            skip_attn: true,
            skip_ffn: false,
            skip_lm_head: false,
            skip_embed: false,
        };
        assert!(opts.should_skip("self_attn.q_proj.weight"));
        assert!(!opts.should_skip("mlp.gate_proj.weight"));
        assert!(!opts.should_skip("lm_head.weight"));
    }

    #[test]
    fn should_skip_lm_head_is_exact_match() {
        // The lm_head check is `key == "lm_head.weight"` — must not
        // accidentally drop attn/ffn keys that happen to contain
        // "lm_head" as a substring (none in practice, but pin the
        // contract).
        let opts = LoadWeightsOptions {
            skip_attn: false,
            skip_ffn: false,
            skip_lm_head: true,
            skip_embed: false,
        };
        assert!(opts.should_skip("lm_head.weight"));
        assert!(!opts.should_skip("not_lm_head.weight"));
        assert!(!opts.should_skip("lm_head.bias"));
    }

    #[test]
    fn should_skip_default_options_skips_nothing() {
        let opts = LoadWeightsOptions::default();
        for key in [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "lm_head.weight",
            "model.embed_tokens.weight",
        ] {
            assert!(!opts.should_skip(key), "should keep {key}");
        }
    }

    // ─── find_tokenizer_path ───────────────────────────────────────

    #[test]
    fn find_tokenizer_path_in_directory() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokenizer.json");
        std::fs::write(&path, "{}").unwrap();
        assert_eq!(find_tokenizer_path(dir.path()), Some(path));
    }

    #[test]
    fn find_tokenizer_path_falls_back_to_parent() {
        // Some vindex layouts nest a subdir but the tokenizer lives
        // alongside. Pin the parent-fallback so a refactor doesn't
        // silently drop it.
        let parent = tempfile::tempdir().unwrap();
        let path = parent.path().join("tokenizer.json");
        std::fs::write(&path, "{}").unwrap();
        let nested = parent.path().join("vindex");
        std::fs::create_dir_all(&nested).unwrap();
        assert_eq!(find_tokenizer_path(&nested), Some(path));
    }

    #[test]
    fn find_tokenizer_path_returns_none_when_missing() {
        let dir = tempfile::tempdir().unwrap();
        assert!(find_tokenizer_path(dir.path()).is_none());
    }
}
