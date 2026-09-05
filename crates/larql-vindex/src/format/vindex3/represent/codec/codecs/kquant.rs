//! The ggml K-quants and Q8_0: blocked codes with their scales inline.
//!
//! One stream, because the scales ride inside the blocks; row-random,
//! because the container refuses a row that is not a whole number of
//! blocks, so every row starts on one. No direct realization on this
//! executor: the V3 CPU plan set has no K-quant kernel, and the trait
//! says so instead of routing through a decode nobody declared. The
//! grouped Metal kernels that serve K-quant expert banks are a device
//! realization, declared by the crate that owns them.

use std::ops::Range;

use super::super::capability::{AccessGranularity, CodecCapabilities};
use super::super::error::CodecError;
use super::super::extent::{ExtentCertificate, RepresentationExtent};
use super::super::geometry::RowGeometry;
use super::super::residency::ResidencyProfile;
use super::super::streams::{CodecOperands, StreamSpec, VALUES};
use super::super::RepresentationCodec;
use crate::format::vindex3::represent::kquant::{self, KQuant};
use crate::format::vindex3::represent::nvfp4_pack::CodecIdentity;

/// The widest field in a ggml block is an f16 scale.
const BLOCK_FIELD_ALIGN_BYTES: usize = std::mem::size_of::<u16>();

const KQUANT_STREAMS: [StreamSpec; 1] = [VALUES];

/// One compilable K-quant, as a codec.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KQuantCodec {
    quant: KQuant,
}

pub const Q4_K: KQuantCodec = KQuantCodec {
    quant: kquant::Q4_K,
};
pub const Q6_K: KQuantCodec = KQuantCodec {
    quant: kquant::Q6_K,
};
pub const Q8_0: KQuantCodec = KQuantCodec {
    quant: kquant::Q8_0,
};

impl KQuantCodec {
    pub const fn quant(&self) -> KQuant {
        self.quant
    }

    fn row_bytes(&self, k: usize) -> usize {
        k / self.quant.elements_per_block * self.quant.bytes_per_block
    }
}

impl RepresentationCodec for KQuantCodec {
    fn encoding_label(&self) -> &'static str {
        self.quant.name
    }

    fn identity(&self) -> CodecIdentity {
        self.quant.codec_identity()
    }

    fn streams(&self) -> &'static [StreamSpec] {
        &KQUANT_STREAMS
    }

    fn capabilities(&self) -> CodecCapabilities {
        CodecCapabilities {
            access: AccessGranularity::RowRandom,
            group_elems: self.quant.elements_per_block,
            row_align_elems: self.quant.elements_per_block,
            physical_align_bytes: BLOCK_FIELD_ALIGN_BYTES,
        }
    }

    fn extents(&self) -> Vec<ExtentCertificate> {
        vec![ExtentCertificate::terminal(self.quant.bits_per_weight())]
    }

    fn stored_bytes(
        &self,
        shape: &[usize],
        extent: RepresentationExtent,
        tensor: &str,
    ) -> Result<u64, CodecError> {
        self.certificate_at(extent, tensor)?;
        let label = self.encoding_label();
        let geometry = RowGeometry::of(shape, label, tensor)?;
        geometry.check_group(self.quant.elements_per_block, label, tensor)?;
        Ok((geometry.rows * self.row_bytes(geometry.k)) as u64)
    }

    fn validate(
        &self,
        operands: &CodecOperands<'_>,
        shape: &[usize],
        extent: RepresentationExtent,
        tensor: &str,
    ) -> Result<(), CodecError> {
        let need = self.stored_bytes(shape, extent, tensor)? as usize;
        operands.stream_of_len(VALUES, need, self.encoding_label(), tensor)?;
        Ok(())
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
        let label = self.encoding_label();
        let geometry = RowGeometry::of(shape, label, tensor)?;
        geometry.check_group(self.quant.elements_per_block, label, tensor)?;
        geometry.check_rows(&rows, label, tensor)?;
        geometry.check_destination(&rows, dst.len(), tensor)?;
        let row_bytes = self.row_bytes(geometry.k);
        let bytes = operands.stream_of_len(VALUES, rows.end * row_bytes, label, tensor)?;
        let span = &bytes[rows.start * row_bytes..rows.end * row_bytes];
        let values = self
            .quant
            .decode(span, rows.len() * geometry.k, tensor)
            .map_err(|e| CodecError::Decode {
                tensor: tensor.into(),
                label: label.into(),
                detail: e.to_string(),
            })?;
        dst.copy_from_slice(&values);
        Ok(())
    }

    fn decode_residency(&self) -> ResidencyProfile {
        ResidencyProfile::DECODED_F32
    }
}
