//! MXFP4: codes and e8m0 group scales stored APART — the multi-stream
//! witness. It does not override [`RepresentationCodec::bind_packed`], so
//! handing it one slice is refused by name, which is the answer
//! `QuantMatVec` could only give as `None`.

use std::ops::Range;

use larql_models::quant::mxfp4::{dequantize_expert, MXFP4_GROUP_BYTES, MXFP4_GROUP_ELEMS};

use super::super::capability::{AccessGranularity, CodecCapabilities};
use super::super::error::CodecError;
use super::super::extent::{ExtentCertificate, RepresentationExtent, BITS_PER_BYTE};
use super::super::geometry::RowGeometry;
use super::super::residency::ResidencyProfile;
use super::super::streams::{CodecOperands, StreamSpec, GROUP_SCALES, VALUES};
use super::super::RepresentationCodec;
use super::vocabulary::{BYTE_ALIGN, SCALE_NONE};
use crate::format::vindex3::represent::nvfp4_pack::CodecIdentity;

/// The segment / graph label for natively stored MXFP4 bytes.
pub const DTYPE_MXFP4: &str = "MXFP4";
/// The ABI family, spelled the way `nvfp4` is.
pub const MXFP4_FAMILY: &str = "mxfp4";
/// ABI revision: OCP microscaling e2m1 codes, lo nibble first, one e8m0
/// scale per 32 — unchanged since the format was admitted.
pub const MXFP4_REVISION: u32 = 1;

/// One e8m0 scale per group.
const E8M0_SCALE_BYTES: usize = 1;
const ELEMENT_E2M1: &str = "e2m1";
const GROUP_SCALE_E8M0: &str = "e8m0";
/// Two streams, each its own region.
const LAYOUT_APART: &str = "codes;group_scales";

const MXFP4_STREAMS: [StreamSpec; 2] = [VALUES, GROUP_SCALES];

/// Natively stored MXFP4.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mxfp4Codec;

pub const MXFP4: Mxfp4Codec = Mxfp4Codec;

impl Mxfp4Codec {
    /// 4.25: sixteen code bytes and one scale byte per thirty-two.
    pub const fn bits_per_weight() -> f64 {
        (MXFP4_GROUP_BYTES + E8M0_SCALE_BYTES) as f64 * BITS_PER_BYTE / MXFP4_GROUP_ELEMS as f64
    }

    fn geometry(shape: &[usize], tensor: &str) -> Result<(RowGeometry, usize), CodecError> {
        let geometry = RowGeometry::of(shape, DTYPE_MXFP4, tensor)?;
        let groups = geometry.check_group(MXFP4_GROUP_ELEMS, DTYPE_MXFP4, tensor)?;
        Ok((geometry, groups))
    }
}

impl RepresentationCodec for Mxfp4Codec {
    fn encoding_label(&self) -> &'static str {
        DTYPE_MXFP4
    }

    fn identity(&self) -> CodecIdentity {
        CodecIdentity {
            family: MXFP4_FAMILY.into(),
            revision: MXFP4_REVISION,
            group_elems: MXFP4_GROUP_ELEMS,
            element: ELEMENT_E2M1.into(),
            group_scale: GROUP_SCALE_E8M0.into(),
            tensor_scale: SCALE_NONE.into(),
            layout: LAYOUT_APART.into(),
        }
    }

    fn streams(&self) -> &'static [StreamSpec] {
        &MXFP4_STREAMS
    }

    fn capabilities(&self) -> CodecCapabilities {
        CodecCapabilities {
            access: AccessGranularity::RowRandom,
            group_elems: MXFP4_GROUP_ELEMS,
            row_align_elems: MXFP4_GROUP_ELEMS,
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
        let (geometry, groups) = Self::geometry(shape, tensor)?;
        let per_row = groups * (MXFP4_GROUP_BYTES + E8M0_SCALE_BYTES);
        Ok((geometry.rows * per_row) as u64)
    }

    fn validate(
        &self,
        operands: &CodecOperands<'_>,
        shape: &[usize],
        extent: RepresentationExtent,
        tensor: &str,
    ) -> Result<(), CodecError> {
        self.certificate_at(extent, tensor)?;
        let (geometry, groups) = Self::geometry(shape, tensor)?;
        let cells = geometry.rows * groups;
        operands.stream_of_len(VALUES, cells * MXFP4_GROUP_BYTES, DTYPE_MXFP4, tensor)?;
        operands.stream_of_len(GROUP_SCALES, cells * E8M0_SCALE_BYTES, DTYPE_MXFP4, tensor)?;
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
        let (geometry, groups) = Self::geometry(shape, tensor)?;
        geometry.check_rows(&rows, DTYPE_MXFP4, tensor)?;
        geometry.check_destination(&rows, dst.len(), tensor)?;
        let code_row = groups * MXFP4_GROUP_BYTES;
        let scale_row = groups * E8M0_SCALE_BYTES;
        let codes = operands.stream_of_len(VALUES, rows.end * code_row, DTYPE_MXFP4, tensor)?;
        let scales =
            operands.stream_of_len(GROUP_SCALES, rows.end * scale_row, DTYPE_MXFP4, tensor)?;
        let values = dequantize_expert(
            &codes[rows.start * code_row..rows.end * code_row],
            &scales[rows.start * scale_row..rows.end * scale_row],
            rows.len(),
            groups,
        )
        .map_err(|e| CodecError::Decode {
            tensor: tensor.into(),
            label: DTYPE_MXFP4.into(),
            detail: e.to_string(),
        })?;
        dst.copy_from_slice(&values);
        Ok(())
    }

    fn decode_residency(&self) -> ResidencyProfile {
        ResidencyProfile::DECODED_F32
    }
}
