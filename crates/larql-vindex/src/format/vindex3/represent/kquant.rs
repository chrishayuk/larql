//! The K-quant vocabulary REPRESENT can compile, in one place.
//!
//! Before this module the `Q8_0` / `Q6_K` / `Q4_K` encoders existed but
//! only Kimi's routed-expert and KDA bank compilers could reach them.
//! `compile_representation` — the general, model-agnostic path — refused
//! everything but NVFP4, so a dense model could not be compiled to a
//! K-quant at all. This is the vocabulary that refusal was hiding.
//!
//! ## Why a geometry table, and why it is not trusted
//!
//! Planning a segment needs each tensor's encoded length *before* any
//! bytes are written, so the compiler cannot simply encode and measure.
//! That forces a table — and a table is a second statement of a fact the
//! codecs already state, which is exactly how the two drift apart.
//!
//! So the table is **checked against the codecs, in both directions**,
//! rather than believed: [`tests`] encodes a real buffer and asserts the
//! byte count matches [`KQuant::encoded_len`], then decodes it and
//! asserts the element count comes back. A wrong entry here fails the
//! test rather than silently mis-sizing a 20 GB segment.
//!
//! ```text
//! encoding   elements/block   bytes/block   bits/weight
//! Q8_0             32              34          8.5
//! Q6_K            256             210          6.5625
//! Q4_K            256             144          4.5
//! ```
//!
//! ## Not the whole family
//!
//! The *decoders* in `larql_models::quant::ggml` already cover Q2_K,
//! Q3_K and Q5_K. Only these three have encoders in the workspace, and
//! an encoding whose bytes we cannot write is not a representation this
//! compiler can offer. Adding one here means adding its encoder, not
//! adding a row.

use super::nvfp4_pack::CodecIdentity;
use crate::error::VindexError;

/// One compilable K-quant encoding: its name, its ggml type, and the
/// block geometry the planner sizes segments with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KQuant {
    /// The encoding's name, as it appears in a `PrecisionMap` and as the
    /// representation-id suffix (`object@Q6_K`).
    pub name: &'static str,
    /// The ggml tensor type these bytes are, which is what makes a
    /// compiled pack a candidate for byte pass-through on export.
    pub ggml_type: u32,
    /// Elements one block covers. A tensor whose element count is not a
    /// whole number of these is refused, never padded — padding would
    /// make rows share a scale with values that are not theirs.
    pub elements_per_block: usize,
    /// Bytes one block occupies.
    pub bytes_per_block: usize,
}

/// 32 elements, f16 scale + 32 int8.
pub const Q8_0: KQuant = KQuant {
    name: "Q8_0",
    ggml_type: larql_models::quant::ggml::TYPE_Q8_0,
    elements_per_block: 32,
    bytes_per_block: 34,
};

/// 256-element super-block: 4-bit lows, 2-bit highs, 16 int8 scales, f16 d.
pub const Q6_K: KQuant = KQuant {
    name: "Q6_K",
    ggml_type: larql_models::quant::ggml::TYPE_Q6_K,
    elements_per_block: 256,
    bytes_per_block: 210,
};

/// 256-element super-block, 4-bit quants with 6-bit scales/mins.
pub const Q4_K: KQuant = KQuant {
    name: "Q4_K",
    ggml_type: larql_models::quant::ggml::TYPE_Q4_K,
    elements_per_block: 256,
    bytes_per_block: 144,
};

/// Every encoding this compiler can write, in ascending bit order.
pub const COMPILABLE: [KQuant; 3] = [Q4_K, Q6_K, Q8_0];

/// The encoding named, or `None` if this compiler cannot write it.
///
/// Deliberately exact-match and case-sensitive: `PrecisionMap` names are
/// compared as-is everywhere else, and accepting `q6_k` here while the
/// representation id says `Q6_K` would produce a pack nothing selects.
pub fn lookup(name: &str) -> Option<KQuant> {
    COMPILABLE.into_iter().find(|k| k.name == name)
}

/// The names this compiler accepts, for an error message that stays
/// correct when the list changes.
pub fn compilable_names() -> String {
    COMPILABLE
        .iter()
        .map(|k| k.name)
        .collect::<Vec<_>>()
        .join(", ")
}

impl KQuant {
    /// Encoded length of `n_elements`, or a refusal naming the geometry.
    ///
    /// The refusal is the point: a tensor whose element count is not a
    /// whole number of blocks cannot be encoded without either padding
    /// (values sharing a scale with invented zeros) or a ragged final
    /// block (a layout no decoder in the ecosystem reads).
    pub fn encoded_len(&self, n_elements: usize, tensor: &str) -> Result<usize, VindexError> {
        if !n_elements.is_multiple_of(self.elements_per_block) {
            return Err(VindexError::Parse(format!(
                "tensor `{tensor}`: {n_elements} values is not a whole number of \
                 {}-element blocks, so {} rows would share a scale",
                self.elements_per_block, self.name
            )));
        }
        Ok(n_elements / self.elements_per_block * self.bytes_per_block)
    }

    /// Encoded length of a tensor of `shape`, or a refusal.
    ///
    /// **This is not [`Self::encoded_len`] on the element product, and
    /// the difference is a correctness bug waiting to happen.** ggml
    /// blocks run along the innermost (row) dimension, so it is the ROW
    /// LENGTH that must be a whole number of blocks. A `[2, 128]` tensor
    /// has 256 elements — one whole Q6_K super-block — and quantising it
    /// flat would put the end of row 0 and the start of row 1 under a
    /// single shared scale. The totals agree; the meaning does not.
    ///
    /// The role policy protects 1-D operands anyway, but a shape with no
    /// dimensions has no row to block along and is refused rather than
    /// treated as a degenerate success.
    pub fn plan(&self, shape: &[usize], tensor: &str) -> Result<usize, VindexError> {
        let Some(&row) = shape.last() else {
            return Err(VindexError::Parse(format!(
                "tensor `{tensor}`: a scalar has no row to block along, so {} does not apply",
                self.name
            )));
        };
        if !row.is_multiple_of(self.elements_per_block) {
            return Err(VindexError::Parse(format!(
                "tensor `{tensor}`: row length {row} of shape {shape:?} is not a whole number \
                 of {}-element blocks, so {} rows would share a scale",
                self.elements_per_block, self.name
            )));
        }
        let elements = shape.iter().try_fold(1usize, |a, d| a.checked_mul(*d));
        let elements = elements.ok_or_else(|| {
            VindexError::Parse(format!(
                "tensor `{tensor}`: shape {shape:?} overflows an element count"
            ))
        })?;
        self.encoded_len(elements, tensor)
    }

    /// Effective bits per weight — the ratio the byte ledger prices.
    pub fn bits_per_weight(&self) -> f64 {
        self.bytes_per_block as f64 * 8.0 / self.elements_per_block as f64
    }

    /// Encode f32 values to this representation's bytes.
    ///
    /// The length is checked against [`Self::encoded_len`] before
    /// returning, so a codec that ever disagrees with the table is
    /// caught at the point of writing rather than by a segment whose
    /// tensor table no longer describes its payload.
    pub fn encode(&self, values: &[f32], tensor: &str) -> Result<Vec<u8>, VindexError> {
        use larql_compute::cpu::ops::q4_common::{quantize_q4_k, quantize_q6_k, quantize_q8_0};
        let expect = self.encoded_len(values.len(), tensor)?;
        let bytes = match self.name {
            "Q8_0" => quantize_q8_0(values),
            "Q6_K" => quantize_q6_k(values),
            "Q4_K" => quantize_q4_k(values),
            other => {
                return Err(VindexError::Parse(format!(
                    "tensor `{tensor}`: `{other}` is in the geometry table but has no encoder \
                     — refusing rather than binding source bytes under a name that claims \
                     otherwise"
                )))
            }
        };
        if bytes.len() != expect {
            return Err(VindexError::Parse(format!(
                "tensor `{tensor}`: {} encoder produced {} bytes, geometry implies {expect} \
                 — the table and the codec disagree",
                self.name,
                bytes.len()
            )));
        }
        Ok(bytes)
    }

    /// Decode this representation's bytes back to f32.
    ///
    /// Routed through the workspace's single ggml decode dispatch rather
    /// than a second opinion about the arithmetic — the same rule the
    /// NVFP4 pack follows.
    pub fn decode(
        &self,
        bytes: &[u8],
        n_elements: usize,
        tensor: &str,
    ) -> Result<Vec<f32>, VindexError> {
        // Validates the geometry before the decoder sees the bytes, so a
        // truncated segment is named here rather than as a short vector
        // somewhere downstream.
        self.encoded_len(n_elements, tensor)?;
        larql_models::quant::ggml::dequantize(bytes, self.ggml_type, n_elements).map_err(|e| {
            VindexError::Parse(format!("tensor `{tensor}`: decoding {}: {e}", self.name))
        })
    }
}

impl KQuant {
    /// The decode contract these bytes are written under.
    ///
    /// A K-quant is its own **family**, not a revision of one: `Q4_K` and
    /// `Q6_K` are different formats with different block layouts, and
    /// filing them under a shared family would let a reader that
    /// implements one accept the other's bytes on a revision match.
    ///
    /// The `CodecIdentity` fields are reused rather than extended, each
    /// carrying the K-quant fact that corresponds to it, so one ABI gate
    /// serves both representation families.
    pub fn codec_identity(&self) -> CodecIdentity {
        CodecIdentity {
            family: self.name.into(),
            revision: KQUANT_REVISION,
            group_elems: self.elements_per_block,
            element: format!("ggml-{}", self.name.to_lowercase()),
            group_scale: "in-block".into(),
            tensor_scale: "in-block".into(),
            layout: format!("ggml-block-{}", self.bytes_per_block),
        }
    }
}

/// ABI revision of the K-quant packs this build writes and reads.
///
/// These are the published ggml block layouts, so a bump here means this
/// workspace's encoder or decoder stopped agreeing with them — which is
/// the [`project_ggml_nibble_layout_nonconformance`] hazard, and exactly
/// what a stored pack must be refused over rather than decoded through.
pub const KQUANT_REVISION: u32 = 1;

#[cfg(test)]
#[path = "kquant_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "kquant_conformance_tests.rs"]
mod conformance_tests;
