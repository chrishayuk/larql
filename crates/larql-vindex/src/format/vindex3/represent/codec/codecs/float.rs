//! Float tensor tables — the boring baseline, and the proof that the
//! contract costs a flat codec nothing.

use std::ops::Range;

use super::super::capability::{AccessGranularity, CodecCapabilities};
use super::super::error::CodecError;
use super::super::extent::{ExtentCertificate, RepresentationExtent, BITS_PER_BYTE};
use super::super::geometry::RowGeometry;
use super::super::residency::{Acceleration, ResidencyProfile};
use super::super::streams::{CodecOperands, StreamSpec, VALUES};
use super::super::RepresentationCodec;
use super::vocabulary::{SCALE_NONE, UNGROUPED};
use crate::format::vindex3::opplan::exec::cpu::physical::PhysicalProjectionPlan;
use crate::format::vindex3::opplan::exec::operands::widen;
use crate::format::vindex3::represent::nvfp4_pack::CodecIdentity;

/// ABI revision of the float row layouts. Little-endian rows of one
/// fixed-width element have not changed and will not.
pub const FLOAT_REVISION: u32 = 1;

/// Region order of a float tensor: one row-major, little-endian stream.
const LAYOUT_ROW_MAJOR_LE: &str = "row-major-le";

const FLOAT_STREAMS: [StreamSpec; 1] = [VALUES];

/// The three float element types a segment stores.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatDtype {
    Bf16,
    F16,
    F32,
}

impl FloatDtype {
    /// The segment `dtype` label.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Bf16 => "BF16",
            Self::F16 => "F16",
            Self::F32 => "F32",
        }
    }

    pub const fn width_bytes(self) -> usize {
        match self {
            Self::Bf16 | Self::F16 => std::mem::size_of::<u16>(),
            Self::F32 => std::mem::size_of::<f32>(),
        }
    }

    /// The element grid, as the ABI identity names it.
    const fn element(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::F16 => "ieee-f16",
            Self::F32 => "ieee-f32",
        }
    }
}

/// A float tensor table: rows of one fixed-width element.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FloatCodec {
    dtype: FloatDtype,
}

pub const BF16: FloatCodec = FloatCodec {
    dtype: FloatDtype::Bf16,
};
pub const F16: FloatCodec = FloatCodec {
    dtype: FloatDtype::F16,
};
pub const F32: FloatCodec = FloatCodec {
    dtype: FloatDtype::F32,
};

impl FloatCodec {
    pub const fn dtype(&self) -> FloatDtype {
        self.dtype
    }

    fn bits_per_weight(&self) -> f64 {
        self.dtype.width_bytes() as f64 * BITS_PER_BYTE
    }

    fn row_bytes(&self, k: usize) -> usize {
        k * self.dtype.width_bytes()
    }
}

impl RepresentationCodec for FloatCodec {
    fn encoding_label(&self) -> &'static str {
        self.dtype.label()
    }

    fn identity(&self) -> CodecIdentity {
        CodecIdentity {
            family: self.dtype.label().into(),
            revision: FLOAT_REVISION,
            group_elems: UNGROUPED,
            element: self.dtype.element().into(),
            group_scale: SCALE_NONE.into(),
            tensor_scale: SCALE_NONE.into(),
            layout: LAYOUT_ROW_MAJOR_LE.into(),
        }
    }

    fn streams(&self) -> &'static [StreamSpec] {
        &FLOAT_STREAMS
    }

    fn capabilities(&self) -> CodecCapabilities {
        CodecCapabilities {
            access: AccessGranularity::ElementRandom,
            group_elems: UNGROUPED,
            row_align_elems: UNGROUPED,
            physical_align_bytes: self.dtype.width_bytes(),
        }
    }

    fn extents(&self) -> Vec<ExtentCertificate> {
        vec![ExtentCertificate::terminal(self.bits_per_weight())]
    }

    fn stored_bytes(
        &self,
        shape: &[usize],
        extent: RepresentationExtent,
        tensor: &str,
    ) -> Result<u64, CodecError> {
        self.certificate_at(extent, tensor)?;
        let label = self.encoding_label();
        let elements = RowGeometry::of(shape, label, tensor)?.elements(label, tensor)?;
        Ok((elements * self.dtype.width_bytes()) as u64)
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
        geometry.check_rows(&rows, label, tensor)?;
        geometry.check_destination(&rows, dst.len(), tensor)?;
        let row_bytes = self.row_bytes(geometry.k);
        let bytes = operands.stream_of_len(VALUES, rows.end * row_bytes, label, tensor)?;
        let span = &bytes[rows.start * row_bytes..rows.end * row_bytes];
        let values = widen(label, span, tensor).map_err(|e| CodecError::Decode {
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

    fn accelerations(&self) -> Vec<Acceleration> {
        let stored = ResidencyProfile::stored(self.bits_per_weight());
        match self.dtype {
            FloatDtype::Bf16 => vec![
                Acceleration::cpu(PhysicalProjectionPlan::FusedBf16, stored),
                Acceleration::cpu(PhysicalProjectionPlan::Bf16xQ8, stored),
            ],
            FloatDtype::F32 => vec![
                Acceleration::cpu(PhysicalProjectionPlan::BlasF32, stored),
                Acceleration::cpu(PhysicalProjectionPlan::ScalarF32, stored),
            ],
            // A device residency format: the CPU executor has no f16
            // kernel, and says so rather than widening in silence.
            FloatDtype::F16 => Vec::new(),
        }
    }
}
