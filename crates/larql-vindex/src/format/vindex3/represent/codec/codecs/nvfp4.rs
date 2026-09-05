//! NVFP4: three streams packed into one row with a split the shape
//! derives — the codec that overrides [`RepresentationCodec::bind_packed`]
//! because its streams are stored together, not apart.

use std::ops::Range;

use larql_models::quant::nvfp4::{
    dequantize_into, Nvfp4Matrix, NVFP4_GROUP_BYTES, NVFP4_GROUP_ELEMS,
};

use super::super::capability::{AccessGranularity, CodecCapabilities};
use super::super::error::CodecError;
use super::super::extent::{ExtentCertificate, RepresentationExtent, BITS_PER_BYTE};
use super::super::geometry::RowGeometry;
use super::super::residency::{Acceleration, ResidencyProfile};
use super::super::streams::{
    CodecOperands, NamedStreams, StreamSpec, GROUP_SCALES, TENSOR_SCALE, VALUES,
};
use super::super::RepresentationCodec;
use crate::format::vindex3::opplan::exec::cpu::physical::PhysicalProjectionPlan;
use crate::format::vindex3::represent::nvfp4_pack::CodecIdentity;
use crate::format::vindex3::represent::nvfp4_pack::{PackLayout, DTYPE_NVFP4};

/// One E4M3 scale per group.
const E4M3_SCALE_BYTES: usize = 1;
/// The f32 tensor scale, little-endian.
const TENSOR_SCALE_BYTES: usize = std::mem::size_of::<f32>();

const NVFP4_STREAMS: [StreamSpec; 3] = [VALUES, GROUP_SCALES, TENSOR_SCALE];

/// The compiled NVFP4 pack.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Nvfp4Codec;

pub const NVFP4: Nvfp4Codec = Nvfp4Codec;

impl Nvfp4Codec {
    /// 4.5: eight code bytes and one scale byte per sixteen elements. The
    /// tensor scale amortises to nothing.
    pub const fn bits_per_weight() -> f64 {
        (NVFP4_GROUP_BYTES + E4M3_SCALE_BYTES) as f64 * BITS_PER_BYTE / NVFP4_GROUP_ELEMS as f64
    }

    /// The pack layout is the one derivation of NVFP4 geometry — its
    /// refusal of a row the group cannot tile is the codec's refusal.
    fn layout(shape: &[usize], tensor: &str) -> Result<PackLayout, CodecError> {
        let geometry = RowGeometry::of(shape, DTYPE_NVFP4, tensor)?;
        PackLayout::derive(&[geometry.rows, geometry.k], tensor).map_err(|e| CodecError::Geometry {
            tensor: tensor.into(),
            label: DTYPE_NVFP4.into(),
            shape: shape.to_vec(),
            why: e.to_string(),
        })
    }

    fn map_err(tensor: &str, detail: impl ToString) -> CodecError {
        CodecError::Decode {
            tensor: tensor.into(),
            label: DTYPE_NVFP4.into(),
            detail: detail.to_string(),
        }
    }
}

impl RepresentationCodec for Nvfp4Codec {
    fn encoding_label(&self) -> &'static str {
        DTYPE_NVFP4
    }

    fn identity(&self) -> CodecIdentity {
        CodecIdentity::nvfp4_v1()
    }

    fn streams(&self) -> &'static [StreamSpec] {
        &NVFP4_STREAMS
    }

    fn capabilities(&self) -> CodecCapabilities {
        CodecCapabilities {
            access: AccessGranularity::RowRandom,
            group_elems: NVFP4_GROUP_ELEMS,
            row_align_elems: NVFP4_GROUP_ELEMS,
            physical_align_bytes: TENSOR_SCALE_BYTES,
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
        Ok(Self::layout(shape, tensor)?.total_len as u64)
    }

    /// Codes, group scales and the tensor scale are one row; the split is
    /// arithmetic from the shape, and this is where it happens.
    fn bind_packed<'a>(
        &self,
        payload: &'a [u8],
        shape: &[usize],
        tensor: &str,
    ) -> Result<NamedStreams<'a>, CodecError> {
        let layout = Self::layout(shape, tensor)?;
        if payload.len() != layout.total_len {
            return Err(CodecError::StreamLength {
                tensor: tensor.into(),
                label: DTYPE_NVFP4.into(),
                stream: VALUES.name.into(),
                need: layout.total_len,
                have: payload.len(),
            });
        }
        Ok(NamedStreams::new()
            .with(VALUES, &payload[..layout.packed_len])
            .with(
                GROUP_SCALES,
                &payload[layout.scales_offset()..layout.tensor_scale_offset()],
            )
            .with(TENSOR_SCALE, &payload[layout.tensor_scale_offset()..]))
    }

    fn validate(
        &self,
        operands: &CodecOperands<'_>,
        shape: &[usize],
        extent: RepresentationExtent,
        tensor: &str,
    ) -> Result<(), CodecError> {
        self.certificate_at(extent, tensor)?;
        let layout = Self::layout(shape, tensor)?;
        operands.stream_of_len(VALUES, layout.packed_len, DTYPE_NVFP4, tensor)?;
        operands.stream_of_len(GROUP_SCALES, layout.scales_len, DTYPE_NVFP4, tensor)?;
        operands.stream_of_len(TENSOR_SCALE, TENSOR_SCALE_BYTES, DTYPE_NVFP4, tensor)?;
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
        let layout = Self::layout(shape, tensor)?;
        let geometry = RowGeometry {
            rows: layout.rows,
            k: layout.k,
        };
        geometry.check_rows(&rows, DTYPE_NVFP4, tensor)?;
        geometry.check_destination(&rows, dst.len(), tensor)?;
        let code_row = layout.groups * NVFP4_GROUP_BYTES;
        let scale_row = layout.groups * E4M3_SCALE_BYTES;
        let codes = operands.stream_of_len(VALUES, rows.end * code_row, DTYPE_NVFP4, tensor)?;
        let scales =
            operands.stream_of_len(GROUP_SCALES, rows.end * scale_row, DTYPE_NVFP4, tensor)?;
        let tail = operands.stream_of_len(TENSOR_SCALE, TENSOR_SCALE_BYTES, DTYPE_NVFP4, tensor)?;
        let matrix = Nvfp4Matrix {
            packed: codes[rows.start * code_row..rows.end * code_row].to_vec(),
            scales: scales[rows.start * scale_row..rows.end * scale_row].to_vec(),
            tensor_scale: f32::from_le_bytes([tail[0], tail[1], tail[2], tail[3]]),
        };
        dequantize_into(&matrix, rows.len(), layout.k, dst).map_err(|e| Self::map_err(tensor, e))
    }

    fn decode_residency(&self) -> ResidencyProfile {
        ResidencyProfile::DECODED_F32
    }

    fn accelerations(&self) -> Vec<Acceleration> {
        // The loader copies a pack's regions into page-aligned buffers
        // and changes no value: rebound, not stored in place.
        vec![Acceleration::cpu(
            PhysicalProjectionPlan::FusedNvfp4,
            ResidencyProfile::rebound(Self::bits_per_weight()),
        )]
    }
}
