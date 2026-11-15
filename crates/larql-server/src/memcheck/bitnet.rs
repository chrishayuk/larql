//! Resident-size estimate for a BitNet 1.58 `--keep-quant` container.
//!
//! `VindexConfig::estimate_resident_bytes` models the dense path — every
//! projection at the storage dtype width — and would over-count a BitNet
//! container many times over, because its BitLinear projections stay
//! I2_S-packed (2 bits per weight) in memory. This estimate sizes what
//! `larql_inference::ternary::load_bitnet_model` actually allocates:
//!
//! - every BitLinear's packed I2_S bytes, from `bitnet_layout` (exact);
//! - the per-row scales, `total_scale_count` f32s (exact);
//! - the embedding table and the LM head, each `vocab × hidden` f32 —
//!   the ternary model owns both as f32 `Array2`s, tied or not;
//! - the dense loader's own embed + LM head, alive alongside the owned
//!   copies until the load returns, so they count toward the peak;
//! - the RMSNorm vectors, bounded by one f32 vector per BitLinear input
//!   width plus the output norm. Every norm normalises the input of a
//!   BitLinear, so this is an upper bound read from the layout rather
//!   than a per-family norm count.
//!
//! The same allocator / padding slack the dense estimator applies is
//! added on top.

use larql_models::quant::ggml::{I2_S_BLOCK_BYTES, I2_S_BLOCK_ELEMS};
use larql_vindex::VindexConfig;

/// Bytes per element of every f32 buffer the ternary loader materialises.
const F32_BYTES: u64 = std::mem::size_of::<f32>() as u64;
/// `vocab × hidden` tables the ternary model owns: embed + LM head.
const OWNED_EMBED_TABLES: u64 = 2;
/// `vocab × hidden` tables the dense loader holds while the ternary model
/// is assembled from it (embed + LM head, worst case untied).
const TRANSIENT_EMBED_TABLES: u64 = 2;
/// Allocator / padding slack as a fraction `1 / N` of the estimate —
/// the 12.5 % the dense estimator applies.
const OVERHEAD_DIVISOR: u64 = 8;

/// Bytes an I2_S-packed `rows × cols` BitLinear occupies.
pub fn i2s_packed_bytes(rows: usize, cols: usize) -> u64 {
    let elems = (rows as u64).saturating_mul(cols as u64);
    elems
        .div_ceil(I2_S_BLOCK_ELEMS as u64)
        .saturating_mul(I2_S_BLOCK_BYTES as u64)
}

/// Peak resident bytes loading this BitNet container takes, or `None`
/// when the config carries no `bitnet_layout` (not a `--keep-quant`
/// build).
pub fn estimate_bitnet_resident_bytes(config: &VindexConfig) -> Option<u64> {
    let layout = config.bitnet_layout.as_ref()?;
    let ternary = layout
        .tensors
        .iter()
        .map(|t| i2s_packed_bytes(t.rows, t.cols))
        .fold(0u64, u64::saturating_add);
    let scales = (layout.total_scale_count as u64).saturating_mul(F32_BYTES);
    let norms = layout
        .tensors
        .iter()
        .map(|t| t.cols as u64)
        .fold(config.hidden_size as u64, u64::saturating_add)
        .saturating_mul(F32_BYTES);
    let embed_table = (config.vocab_size as u64)
        .saturating_mul(config.hidden_size as u64)
        .saturating_mul(F32_BYTES);
    let embed_tables =
        embed_table.saturating_mul(OWNED_EMBED_TABLES.saturating_add(TRANSIENT_EMBED_TABLES));

    let total = ternary
        .saturating_add(scales)
        .saturating_add(norms)
        .saturating_add(embed_tables);
    Some(total.saturating_add(total / OVERHEAD_DIVISOR))
}
