//! **Addressing an arbitrarily-ordered source expert bank.**
//!
//! A source container stores what the checkpoint gave it. Kimi's expert
//! tensors are NOT in expert order — expert 7's `w1` sits at byte
//! 3,156,738,048, not at `7 x stride` — and the layer stride is not
//! regular either, so nothing about the physical layout can be inferred
//! from an expert's identity.
//!
//! What IS true, and measured across all 1024 experts of layers 1, 5, 13
//! and 26 with zero exceptions, is that ONE expert's three projections
//! are contiguous:
//!
//! ```text
//! base_e        w1  (gate)
//! base_e + per  w2  (down)
//! base_e + 2per w3  (up)
//! ```
//!
//! So the whole bank is addressable as three SHIFTED VIEWS of one
//! mapping plus a single per-expert base table — no rewrite, and no
//! 94 GB execution-shaped duplicate.
//!
//! That the source needs a `Table` while a compiled bank needs none is
//! the distinction `ExpertLayout` exists to express. **Semantic identity
//! is stable; physical layout is replaceable.**

use std::collections::BTreeMap;
use std::sync::Arc;

use super::physical::{
    EncodedRegion, ExpertBankBinding, ExpertEncoding, ExtentPolicy, PhysicalStore,
    ProjectionAddressing, RoutedProjection,
};
use crate::error::VindexError;

/// A layer's experts, addressed inside a source segment.
pub struct SourceExpertBank {
    pub binding: ExpertBankBinding,
    /// Byte offset of each expert's block, relative to each view's own
    /// start. One table serves all three projections because the three
    /// are contiguous within an expert.
    pub bases: Vec<u32>,
    /// Payload offset where this layer's expert block begins — the base
    /// every table entry is relative to. Exposed so a caller can
    /// register the layer's span with a compute backend.
    pub layer_base: u64,
    /// Bytes from `layer_base` to the end of the layer's last expert.
    pub layer_len: u64,
}

/// Build the addressing for one layer from the segment's own tensor
/// table.
///
/// `offsets` maps tensor name → payload-relative byte offset, exactly as
/// the segment header records it. Nothing is computed from an expert id.
pub fn source_expert_bank(
    store: &Arc<PhysicalStore>,
    offsets: &BTreeMap<String, u64>,
    layer: u32,
    experts: u32,
    per_projection_bytes: u64,
) -> Result<SourceExpertBank, VindexError> {
    if experts == 0 {
        return Err(VindexError::Parse(format!(
            "layer {layer}: a source expert bank over zero experts addresses nothing"
        )));
    }
    let name = |e: u32, proj: &str| format!("{layer}.block_sparse_moe.experts.{e}.{proj}.weight");
    // Absolute payload offset of each expert's block. Collected first,
    // because the table entries are RELATIVE to the layer's own base:
    // the kernel's offset table is 32-bit, and while one layer's experts
    // span ~3.6 GB, the segment holding all 26 layers is ~94 GB — an
    // absolute offset stops fitting after the second layer the encoder
    // happened to place.
    let mut absolute = Vec::with_capacity(experts as usize);
    for e in 0..experts {
        let w1 = *offsets.get(&name(e, "w1")).ok_or_else(|| {
            VindexError::Parse(format!("source segment has no `{}`", name(e, "w1")))
        })?;
        // The contiguity the whole scheme rests on, checked per expert
        // rather than assumed from a sample.
        for (proj, expect) in [
            ("w2", per_projection_bytes),
            ("w3", 2 * per_projection_bytes),
        ] {
            let got = *offsets.get(&name(e, proj)).ok_or_else(|| {
                VindexError::Parse(format!("source segment has no `{}`", name(e, proj)))
            })?;
            if got != w1 + expect {
                return Err(VindexError::Parse(format!(
                    "layer {layer} expert {e}: `{proj}` is at {got}, not {} — this source's \
                     projections are not contiguous within an expert, so one base table \
                     cannot address all three",
                    w1 + expect
                )));
            }
        }
        absolute.push(w1);
    }
    let layer_base = *absolute.iter().min().expect("experts > 0 was checked");
    let layer_end = absolute
        .iter()
        .map(|w1| w1 + 3 * per_projection_bytes)
        .max()
        .expect("experts > 0 was checked");
    let layer_len = layer_end - layer_base;
    let mut bases = Vec::with_capacity(experts as usize);
    for (e, w1) in absolute.iter().enumerate() {
        bases.push(u32::try_from(w1 - layer_base).map_err(|_| {
            VindexError::Parse(format!(
                "layer {layer} expert {e} sits {} bytes past the layer's own base — beyond \
                 what a 32-bit offset table can address, so this layer's experts are not \
                 the contiguous block the rebasing assumes",
                w1 - layer_base
            ))
        })?);
    }

    // Three views of ONE mapping, each starting at its own projection
    // WITHIN THIS LAYER's block. A source container's experts are stored
    // as the checkpoint had them, which for Kimi is BF16.
    //
    // Each view carries its own addressing and extent rather than
    // inheriting a bank-wide pair. The VALUES coincide here — all three
    // are source windows over the same layer block, addressed by the
    // same rebased table — and that is the point: a source bank is the
    // symmetric case, so it should be expressible without the type
    // forcing symmetry on the asymmetric ones.
    //
    // `bases[e]` is the expert's offset from the layer base, and each
    // view is itself rebased to `layer_base + shift`, so one table
    // addresses all three.
    let projection = |shift: u64| -> Result<RoutedProjection, VindexError> {
        Ok(RoutedProjection {
            region: EncodedRegion {
                region: store
                    .span(layer_base + shift, layer_len - shift)
                    .ok_or_else(|| {
                        VindexError::Parse(format!(
                            "segment is too short for layer {layer}'s +{shift} view"
                        ))
                    })?,
                encoding: ExpertEncoding::Bf16,
            },
            addressing: ProjectionAddressing::Table(bases.clone()),
            // A window onto the layer's block of a larger segment:
            // surplus bytes are other experts of the same layer, not a
            // mislabelled encoding.
            extent: ExtentPolicy::ContainingView,
        })
    };
    Ok(SourceExpertBank {
        binding: ExpertBankBinding {
            gate: projection(0)?,
            down: projection(per_projection_bytes)?,
            up: projection(2 * per_projection_bytes)?,
            // This function addresses the ROUTED bank inside the
            // expert-bank segment. Kimi's shared expert lives in the
            // decoder stack — a different store — so its binding is
            // attached by the caller that holds that store; a layer
            // executed with it still `None` is refused downstream.
            shared: None,
        },
        bases,
        layer_base,
        layer_len,
    })
}

#[cfg(test)]
#[path = "source_bank_tests.rs"]
mod tests;
