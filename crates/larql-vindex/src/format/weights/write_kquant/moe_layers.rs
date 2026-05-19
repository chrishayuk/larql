//! Stage 3 — per-layer FFN weights for hybrid MoE models (§5.12).
//!
//! For hybrid MoE models with `ExpertFormat::PackedBF16` (e.g. Gemma 4
//! 26B A4B): the source BF16 expert tensors are quantised to Q4_K per
//! expert and written to `layers/layer_{L:02}.weights` with
//! `num_entries=num_experts`. Replaces the legacy ~43 GB BF16
//! `experts_packed.bin` monolithic blob.
//!
//! For dense models: this is a no-op — `interleaved_kquant.bin` (stage 2)
//! remains the primary FFN store. Per-layer format for dense is a
//! future migration (`--ffn-layout` flag).

use std::path::Path;

use crate::error::VindexError;

use super::super::write_f32::WeightSource;
use super::super::write_layers::{quantize_moe_entries, write_layer_weights, LayerWeightFormat};

pub(super) fn write_per_layer_moe_kquant(
    source: &dyn WeightSource,
    dir: &Path,
    num_layers: usize,
) -> Result<(), VindexError> {
    let arch = source.arch();
    if !(arch.is_hybrid_moe() && arch.expert_format() == larql_models::ExpertFormat::PackedBF16) {
        return Ok(());
    }

    let num_experts = arch.num_experts();
    let moe_inter = arch.moe_intermediate_size();
    let hidden = arch.config().hidden_size;

    for layer in 0..num_layers {
        let gu_key = arch.packed_experts_gate_up_key(layer);
        let dn_key = arch.packed_experts_down_key(layer);
        let gu_bytes = gu_key.as_ref().and_then(|k| source.get_packed_bf16(k));
        let dn_bytes = dn_key.as_ref().and_then(|k| source.get_packed_bf16(k));

        if let (Some(gu), Some(dn)) = (gu_bytes, dn_bytes) {
            // Default: Q4_K for the whole file. Format is uniform — no mixing.
            let fmt = LayerWeightFormat::Q4_K;
            let entries = quantize_moe_entries(&gu, &dn, num_experts, moe_inter, hidden, fmt)?;
            write_layer_weights(dir, layer, fmt, &entries, moe_inter, hidden)?;
        }
    }
    Ok(())
}
