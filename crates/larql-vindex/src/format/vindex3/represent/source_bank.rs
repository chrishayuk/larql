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
    EncodedRegion, ExpertBankBinding, ExpertEncoding, ExpertLayout, PhysicalStore,
};
use crate::error::VindexError;

/// A layer's experts, addressed inside a source segment.
pub struct SourceExpertBank {
    pub binding: ExpertBankBinding,
    /// Byte offset of each expert's block, relative to each view's own
    /// start. One table serves all three projections because the three
    /// are contiguous within an expert.
    pub bases: Vec<u32>,
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
    let name = |e: u32, proj: &str| format!("{layer}.block_sparse_moe.experts.{e}.{proj}.weight");
    let mut bases = Vec::with_capacity(experts as usize);
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
        bases.push(u32::try_from(w1).map_err(|_| {
            VindexError::Parse(format!(
                "layer {layer} expert {e} sits at byte {w1}, past what a 32-bit offset table \
                 can address"
            ))
        })?);
    }

    // Three views of ONE mapping, each starting at its own projection.
    // A source container's experts are stored as the checkpoint had
    // them, which for Kimi is BF16.
    let view = |shift: u64| -> Result<EncodedRegion, VindexError> {
        Ok(EncodedRegion {
            region: store
                .span(shift, store.payload_len() - shift)
                .ok_or_else(|| {
                    VindexError::Parse(format!("segment is too short for a +{shift} view"))
                })?,
            encoding: ExpertEncoding::Bf16,
        })
    };
    Ok(SourceExpertBank {
        binding: ExpertBankBinding {
            gate: view(0)?,
            down: view(per_projection_bytes)?,
            up: view(2 * per_projection_bytes)?,
            // Arbitrary order: the table is the only thing that knows
            // where an expert lives.
            layout: ExpertLayout::Mapped {
                ids: (0..experts).collect(),
            },
        },
        bases,
    })
}

#[cfg(test)]
#[path = "source_bank_tests.rs"]
mod tests;
