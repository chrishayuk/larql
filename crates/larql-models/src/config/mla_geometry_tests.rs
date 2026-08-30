//! `MlaGeometry`'s own arithmetic, pinned at two checkpoints' real
//! numbers — the same posture `kda_geometry_tests.rs` holds for KDA:
//! prove the geometry generic before any execution code leans on it.
//!
//! Kimi Linear and (public) DeepSeek-V3 share every MLA width EXCEPT
//! head count (32 vs 128) — a real-world pair where the two derived
//! quantities either track the shared widths (`q_head_dim`, unaffected
//! by head count) or the differing one (`compressed_kv_width` does NOT
//! depend on `num_heads` either, by construction: the compressed latent
//! is per-POSITION, shared across every head, which is the entire point
//! of the compression). Both facts are worth pinning, not just one.

use super::mla::MlaGeometry;

/// Kimi Linear 48B-A3B: 32 heads.
fn kimi() -> MlaGeometry {
    MlaGeometry {
        num_heads: 32,
        kv_lora_rank: 512,
        qk_nope_head_dim: 128,
        qk_rope_head_dim: 64,
        v_head_dim: 128,
    }
}

/// DeepSeek-V3 (public config): 128 heads, otherwise identical widths —
/// the lineage `MlaOp`'s own doc comment names.
fn deepseek_v3() -> MlaGeometry {
    MlaGeometry {
        num_heads: 128,
        ..kimi()
    }
}

#[test]
fn q_head_dim_is_the_nope_rope_sum_independent_of_head_count() {
    let (kimi, ds) = (kimi(), deepseek_v3());
    assert_eq!(kimi.q_head_dim(), 128 + 64);
    assert_eq!(ds.q_head_dim(), 128 + 64);
    assert_eq!(
        kimi.q_head_dim(),
        ds.q_head_dim(),
        "q_head_dim is a per-head width; head COUNT must not change it"
    );
}

#[test]
fn compressed_kv_width_is_per_position_not_per_head() {
    let (kimi, ds) = (kimi(), deepseek_v3());
    assert_eq!(kimi.compressed_kv_width(), 512 + 64);
    // The whole point of the compression: identical per-position cache
    // cost whether the model has 32 heads or 128.
    assert_eq!(
        kimi.compressed_kv_width(),
        ds.compressed_kv_width(),
        "the compressed latent is shared across every head — head count must not change its width"
    );
}

/// The one field that DOES separate these two real checkpoints, stated
/// so a future reader cannot assume MLA geometry is head-count-portable
/// in general.
#[test]
fn head_count_is_the_one_real_difference_between_the_two_checkpoints() {
    assert_ne!(kimi().num_heads, deepseek_v3().num_heads);
}

#[test]
fn geometry_is_copy_and_plain_data_no_hidden_state() {
    let g = kimi();
    let copy = g; // Copy, not a move — every call site constructs cheaply.
    assert_eq!(g, copy);
}
