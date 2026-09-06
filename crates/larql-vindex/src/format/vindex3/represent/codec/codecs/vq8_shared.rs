//! `VQ8_SHARED`: one byte per four weights, and a codebook that is
//! another represented object.
//!
//! The first codec whose bytes do not mean anything on their own. Its
//! stream is a list of indices; what they index is a SEPARATE operand,
//! addressed by the container's reference table, possibly shared with
//! other tensors, and decoded through its own codec at its own extent
//! before this one is called. Take the codebook away and the codes are
//! not a degraded tensor — they are not a tensor at all.
//!
//! ```text
//! codes      [e/4] u8      one index per vector of four weights
//! codebook   [256, 4] f32  ANOTHER OPERAND, named `codebook`, shared
//! decode     w[i] = codebook[codes[i/4]][i % 4]
//! ```
//!
//! Deliberately the simplest scheme in which the grouping is real: a code
//! is not an element, so a row's codes and its values have different
//! lengths and nothing can quietly treat the codebook as a per-group
//! scale. Equally deliberately, it declares no direct kernel and no
//! progressive extent — the dependency is the subject, and one hostile
//! thing at a time.
//!
//! What it does NOT declare is a reconstruction radius. The error of a
//! vector quantiser is a property of the encoder and the data it was fit
//! to, not of the format; declaring a number here would promote one
//! measurement to a property of every artifact ever stored this way.

use std::ops::Range;

use super::super::auxiliary::{AuxiliaryMetadata, AuxiliarySpec};
use super::super::capability::{AccessGranularity, CodecCapabilities};
use super::super::error::CodecError;
use super::super::extent::{ExtentCertificate, RepresentationExtent, BITS_PER_BYTE};
use super::super::geometry::RowGeometry;
use super::super::residency::ResidencyProfile;
use super::super::streams::{CodecOperands, StreamSpec, VALUES};
use super::super::RepresentationCodec;
use super::vocabulary::{BYTE_ALIGN, SCALE_NONE};
use crate::format::vindex3::represent::nvfp4_pack::CodecIdentity;

/// The segment / graph label for codes indexing a shared codebook.
pub const DTYPE_VQ8_SHARED: &str = "VQ8_SHARED";
/// ABI revision. Revision 1 is: one `u8` code per four row-major weights,
/// indexing a `[256, 4]` f32 codebook named `codebook`. The REQUIREMENT's
/// name is part of this revision, because a stored container's references
/// are keyed by it.
pub const VQ8_SHARED_REVISION: u32 = 1;

/// Weights one code stands for.
pub const VQ_VECTOR_ELEMS: usize = 4;
/// Codebook entries a `u8` code can address.
pub const VQ_CODEBOOK_ENTRIES: usize = 256;
/// The name this codec gives its dependency.
pub const CODEBOOK: &str = "codebook";

const ELEMENT_U8_INDEX: &str = "u8-index";
const LAYOUT_ROW_MAJOR_CODES: &str = "row-major/vq8-codes";

const VQ8_STREAMS: [StreamSpec; 1] = [VALUES];
const VQ8_AUXILIARIES: [AuxiliarySpec; 1] = [AuxiliarySpec::new(CODEBOOK)];

/// Codes indexing a shared codebook.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Vq8SharedCodec;

pub const VQ8_SHARED: Vq8SharedCodec = Vq8SharedCodec;

impl Vq8SharedCodec {
    /// The shape a codebook must have, whatever the tensor using it.
    pub const CODEBOOK_SHAPE: [usize; 2] = [VQ_CODEBOOK_ENTRIES, VQ_VECTOR_ELEMS];

    /// Bits per weight the codes cost. The codebook's own bytes are the
    /// CODEBOOK's footprint — it is another operand, and one shared by
    /// many owners cannot be amortised into any single owner's rate.
    pub const fn bits_per_weight() -> f64 {
        BITS_PER_BYTE / VQ_VECTOR_ELEMS as f64
    }

    /// `[rows, k]` with `k` a whole number of vectors, and the element
    /// count that follows.
    fn geometry(shape: &[usize], tensor: &str) -> Result<(RowGeometry, usize), CodecError> {
        let geometry = RowGeometry::of(shape, DTYPE_VQ8_SHARED, tensor)?;
        geometry.check_group(VQ_VECTOR_ELEMS, DTYPE_VQ8_SHARED, tensor)?;
        let elements = geometry.elements(DTYPE_VQ8_SHARED, tensor)?;
        Ok((geometry, elements))
    }

    /// Encode `values` as codes against `codebook`, by nearest entry —
    /// the fixture side of the format, kept beside the decoder it must
    /// match. Not a quantiser anyone should use: it is exhaustive, and it
    /// exists so a test can produce bytes this codec reads.
    pub fn encode_codes(values: &[f32], codebook: &[f32]) -> Vec<u8> {
        values
            .chunks(VQ_VECTOR_ELEMS)
            .map(|vector| {
                let mut best = (0usize, f64::INFINITY);
                for entry in 0..VQ_CODEBOOK_ENTRIES {
                    let base = entry * VQ_VECTOR_ELEMS;
                    let distance: f64 = vector
                        .iter()
                        .enumerate()
                        .map(|(i, v)| {
                            let d = f64::from(*v) - f64::from(codebook[base + i]);
                            d * d
                        })
                        .sum();
                    if distance < best.1 {
                        best = (entry, distance);
                    }
                }
                best.0 as u8
            })
            .collect()
    }
}

impl RepresentationCodec for Vq8SharedCodec {
    fn encoding_label(&self) -> &'static str {
        DTYPE_VQ8_SHARED
    }

    fn identity(&self) -> CodecIdentity {
        CodecIdentity {
            family: DTYPE_VQ8_SHARED.into(),
            revision: VQ8_SHARED_REVISION,
            group_elems: VQ_VECTOR_ELEMS,
            element: ELEMENT_U8_INDEX.into(),
            group_scale: SCALE_NONE.into(),
            tensor_scale: SCALE_NONE.into(),
            layout: LAYOUT_ROW_MAJOR_CODES.into(),
        }
    }

    fn streams(&self) -> &'static [StreamSpec] {
        &VQ8_STREAMS
    }

    fn required_auxiliaries(&self, _: RepresentationExtent) -> &'static [AuxiliarySpec] {
        // One extent, one requirement: the codes mean nothing without it,
        // at any depth this codec will ever have.
        &VQ8_AUXILIARIES
    }

    fn validate_auxiliary(
        &self,
        name: &str,
        target: &AuxiliaryMetadata,
        _: &[usize],
        _: RepresentationExtent,
        tensor: &str,
    ) -> Result<(), CodecError> {
        // The codebook's shape is this codec's business and nothing
        // else's: 256 entries because a code is a byte, four wide because
        // a code stands for four weights.
        target.require_shape(&Self::CODEBOOK_SHAPE, tensor, DTYPE_VQ8_SHARED, name)
    }

    fn capabilities(&self) -> CodecCapabilities {
        CodecCapabilities {
            // A row is a whole number of vectors by construction (see
            // `row_align_elems`), so any row is addressable.
            access: AccessGranularity::RowRandom,
            group_elems: VQ_VECTOR_ELEMS,
            row_align_elems: VQ_VECTOR_ELEMS,
            physical_align_bytes: BYTE_ALIGN,
        }
    }

    fn extents(&self) -> Vec<ExtentCertificate> {
        vec![ExtentCertificate::terminal(Self::bits_per_weight())]
    }

    fn stored_bytes(
        &self,
        shape: &[usize],
        extent: RepresentationExtent,
        tensor: &str,
    ) -> Result<u64, CodecError> {
        self.certificate_at(extent, tensor)?;
        let (_, elements) = Self::geometry(shape, tensor)?;
        Ok((elements / VQ_VECTOR_ELEMS) as u64)
    }

    fn validate(
        &self,
        operands: &CodecOperands<'_>,
        shape: &[usize],
        extent: RepresentationExtent,
        tensor: &str,
    ) -> Result<(), CodecError> {
        self.certificate_at(extent, tensor)?;
        let (_, elements) = Self::geometry(shape, tensor)?;
        let codes = operands.stream(VALUES, DTYPE_VQ8_SHARED, tensor)?;
        let need = elements / VQ_VECTOR_ELEMS;
        if codes.len() != need {
            return Err(CodecError::StreamLength {
                tensor: tensor.into(),
                label: DTYPE_VQ8_SHARED.into(),
                stream: VALUES.name.into(),
                need,
                have: codes.len(),
            });
        }
        self.codebook(operands, tensor).map(|_| ())
    }

    fn decode_rows(
        &self,
        operands: &CodecOperands<'_>,
        shape: &[usize],
        rows: Range<usize>,
        extent: RepresentationExtent,
        dst: &mut [f32],
        tensor: &str,
    ) -> Result<(), CodecError> {
        self.certificate_at(extent, tensor)?;
        let (geometry, elements) = Self::geometry(shape, tensor)?;
        geometry.check_rows(&rows, DTYPE_VQ8_SHARED, tensor)?;
        geometry.check_destination(&rows, dst.len(), tensor)?;
        let codes =
            operands.stream_of_len(VALUES, elements / VQ_VECTOR_ELEMS, DTYPE_VQ8_SHARED, tensor)?;
        let codebook = self.codebook(operands, tensor)?;
        for (out, element) in dst.iter_mut().zip(rows.start * geometry.k..) {
            let entry = usize::from(codes[element / VQ_VECTOR_ELEMS]);
            *out = codebook[entry * VQ_VECTOR_ELEMS + element % VQ_VECTOR_ELEMS];
        }
        Ok(())
    }

    fn decode_residency(&self) -> ResidencyProfile {
        // Canonical decode: an f32 image, and the codebook's own lifetime
        // is the codebook's realization to declare, not this one's.
        ResidencyProfile::DECODED_F32
    }
}

impl Vq8SharedCodec {
    /// The resolved codebook's values, refusing one the wrong size.
    ///
    /// The SHAPE was judged from metadata before any byte was read
    /// ([`RepresentationCodec::validate_auxiliary`]); this is the same
    /// rule met again on the values, because a decode that indexed past
    /// the end of a short codebook would read whatever followed it.
    fn codebook<'a>(
        &self,
        operands: &CodecOperands<'a>,
        tensor: &str,
    ) -> Result<&'a [f32], CodecError> {
        let resolved = operands
            .auxiliaries
            .require(CODEBOOK, DTYPE_VQ8_SHARED, tensor)?;
        let need = VQ_CODEBOOK_ENTRIES * VQ_VECTOR_ELEMS;
        if resolved.values.len() != need || resolved.shape != Self::CODEBOOK_SHAPE {
            return Err(CodecError::AuxiliaryGeometry {
                tensor: tensor.into(),
                label: DTYPE_VQ8_SHARED.into(),
                name: CODEBOOK.into(),
                why: format!(
                    "the resolved codebook is {:?} with {} values; {:?} with {need} is required",
                    resolved.shape,
                    resolved.values.len(),
                    Self::CODEBOOK_SHAPE,
                ),
            });
        }
        Ok(resolved.values)
    }
}
