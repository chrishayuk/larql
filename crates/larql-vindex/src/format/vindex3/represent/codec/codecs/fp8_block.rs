//! Fine-grained (block-wise) FP8 as frontier checkpoints ship it: E4M3
//! codes with a two-dimensional grid of f32 scales — GLM-5.3-Flash's own
//! bytes (95.8 % of a 306 GiB checkpoint) and the DeepSeek-V3 lineage's.
//!
//! The first PRODUCTION codec whose bytes do not mean anything on their
//! own. The codes are one stream; the scales are ANOTHER REPRESENTED
//! OBJECT — the `*.weight_scale_inv` tensor the checkpoint shipped beside
//! them, stored under its own label, addressed by the container's
//! reference table under the name [`SCALES`], and decoded through its own
//! codec before this one is called. The tile is DERIVED from the two
//! shapes, per tensor, never read from a config: one checkpoint legally
//! ships `[128, 128]` grids beside `[1, 32]` ones, so no identity could
//! state it and no stream could carry it — a stream is bytes, and the
//! grid's SHAPE is what the tile needs. That is what makes the scales a
//! dependency rather than a second stream, and it is the same rule the
//! `VQ8_SHARED` codebook follows.
//!
//! ```text
//! values     [rows, cols] u8 E4M3        one code per weight
//! scales     [rows/br, cols/bc] f32      ANOTHER OPERAND, named `scales`
//! decode     w[r, c] = e4m3(code[r, c]) * scales[r / br, c / bc]
//! ```
//!
//! Before this codec the format was an execution-side special case: a
//! loader that spelled the sibling's name, a `companion` accessor that
//! existed for one format, and a kernel reachable only by a caller that
//! already knew the answer. Selection could not offer it, the reference
//! oracle could not decode it, and an external provider could not have
//! done what it did. Now it is one registration, and the direct kernel
//! it declares is the same `FusedFp8Block` — reached by selection.
//!
//! What it does NOT declare is a reconstruction radius. Fine-grained FP8
//! is lossy against its bf16 source, and the error is the encoder's and
//! the data's rather than the format's — the same reason `VQ8_SHARED`
//! declares none. An attestation supplies a measured radius per artifact.

use std::ops::Range;

use larql_models::quant::fp8::e4m3_to_f32;
use larql_models::quant::fp8_finegrained::Fp8Grid;

use super::super::auxiliary::{AuxiliaryMetadata, AuxiliarySpec};
use super::super::capability::{AccessGranularity, CodecCapabilities};
use super::super::error::CodecError;
use super::super::extent::{ExtentCertificate, RepresentationExtent, BITS_PER_BYTE};
use super::super::geometry::RowGeometry;
use super::super::residency::{Acceleration, ResidencyProfile};
use super::super::streams::{CodecOperands, StreamSpec, VALUES};
use super::super::RepresentationCodec;
use super::vocabulary::{BYTE_ALIGN, SCALE_NONE, UNGROUPED};
use crate::format::vindex3::opplan::exec::cpu::physical::PhysicalProjectionPlan;
use crate::format::vindex3::represent::nvfp4_pack::CodecIdentity;

/// The segment label: the safetensors dtype the encoder carries through
/// unchanged, because the bytes ARE the checkpoint's. A container tensor
/// under this label with no `scales` reference is refused by name — a
/// bare E4M3 tensor is not this representation, and this build ships no
/// other reading of one.
pub const DTYPE_FP8_BLOCK: &str = "F8_E4M3";
/// The ABI family.
pub const FP8_BLOCK_FAMILY: &str = "fp8-block";
/// ABI revision. Revision 1 is: row-major E4M3 codes, one per weight, and
/// a row-major f32 scale grid named `scales` whose shape tiles the codes'
/// evenly; the value is code times scale, multiplied, in f32. The
/// REQUIREMENT's name is part of this revision, because a stored
/// container's references are keyed by it.
pub const FP8_BLOCK_REVISION: u32 = 1;
/// The name this codec gives its dependency.
pub const SCALES: &str = "scales";

/// E4M3 is one byte, exactly. The scale grid is the SCALES operand's own
/// footprint — another object, priced as one — so the rate here is the
/// codes' alone, as `VQ8_SHARED` prices its codes without its codebook.
const FP8_BITS_PER_WEIGHT: f64 = BITS_PER_BYTE;

const ELEMENT_E4M3: &str = "e4m3";
const GROUP_SCALE_TILE_GRID: &str = "f32-le/tile-derived";
const LAYOUT_ROW_MAJOR_CODES: &str = "row-major-codes;scales-by-reference";

const FP8_STREAMS: [StreamSpec; 1] = [VALUES];
const FP8_AUXILIARIES: [AuxiliarySpec; 1] = [AuxiliarySpec::new(SCALES)];

/// Fine-grained FP8: E4M3 codes scaled by a referenced f32 tile grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fp8BlockCodec;

/// The scale grid as a decode reads it: its geometry, the tile it implies,
/// and its values.
struct ResolvedGrid<'a> {
    grid: Fp8Grid,
    block_rows: usize,
    block_cols: usize,
    values: &'a [f32],
}

pub const FP8_BLOCK: Fp8BlockCodec = Fp8BlockCodec;

impl Fp8BlockCodec {
    pub const fn bits_per_weight() -> f64 {
        FP8_BITS_PER_WEIGHT
    }

    /// `[rows, cols]` — the tile is two-dimensional, so the shape must be.
    fn matrix(shape: &[usize], tensor: &str) -> Result<(usize, usize), CodecError> {
        match shape {
            [rows, cols] => Ok((*rows, *cols)),
            other => Err(CodecError::Geometry {
                tensor: tensor.into(),
                label: DTYPE_FP8_BLOCK.into(),
                shape: other.to_vec(),
                why: "fine-grained FP8 tiles a `[out, in]` matrix and has no reading of any \
                      other rank"
                    .into(),
            }),
        }
    }

    /// The grid the shapes describe, or why they describe none.
    fn grid(
        shape: &[usize],
        scale_shape: &[usize],
        tensor: &str,
    ) -> Result<(Fp8Grid, (usize, usize)), CodecError> {
        let (rows, cols) = Self::matrix(shape, tensor)?;
        let unusable = |why: String| CodecError::AuxiliaryGeometry {
            tensor: tensor.into(),
            label: DTYPE_FP8_BLOCK.into(),
            name: SCALES.into(),
            why,
        };
        let [scale_rows, scale_cols] = scale_shape else {
            return Err(unusable(format!(
                "the scale grid has shape {scale_shape:?}, which is not two-dimensional"
            )));
        };
        let grid = Fp8Grid {
            rows,
            cols,
            scale_rows: *scale_rows,
            scale_cols: *scale_cols,
        };
        let tile = grid.tile().map_err(|e| unusable(e.to_string()))?;
        Ok((grid, tile))
    }

    /// The resolved scales, judged again on the values: the shape was
    /// admitted from metadata before any byte was read
    /// ([`RepresentationCodec::validate_auxiliary`]), and a grid whose
    /// values fall short of its shape would index past its end.
    fn scales<'a>(
        &self,
        operands: &CodecOperands<'a>,
        shape: &[usize],
        tensor: &str,
    ) -> Result<ResolvedGrid<'a>, CodecError> {
        let resolved = operands
            .auxiliaries
            .require(SCALES, DTYPE_FP8_BLOCK, tensor)?;
        let (grid, (block_rows, block_cols)) = Self::grid(shape, resolved.shape, tensor)?;
        if resolved.values.len() != grid.scales() {
            return Err(CodecError::AuxiliaryGeometry {
                tensor: tensor.into(),
                label: DTYPE_FP8_BLOCK.into(),
                name: SCALES.into(),
                why: format!(
                    "the resolved grid is {:?} with {} values; {} are required",
                    resolved.shape,
                    resolved.values.len(),
                    grid.scales()
                ),
            });
        }
        Ok(ResolvedGrid {
            grid,
            block_rows,
            block_cols,
            values: resolved.values,
        })
    }
}

impl RepresentationCodec for Fp8BlockCodec {
    fn encoding_label(&self) -> &'static str {
        DTYPE_FP8_BLOCK
    }

    fn identity(&self) -> CodecIdentity {
        CodecIdentity {
            family: FP8_BLOCK_FAMILY.into(),
            revision: FP8_BLOCK_REVISION,
            // The group is the tile, and the tile is a per-tensor fact of
            // the artifact; the identity states that it is not fixed.
            group_elems: UNGROUPED,
            element: ELEMENT_E4M3.into(),
            group_scale: GROUP_SCALE_TILE_GRID.into(),
            tensor_scale: SCALE_NONE.into(),
            layout: LAYOUT_ROW_MAJOR_CODES.into(),
        }
    }

    fn streams(&self) -> &'static [StreamSpec] {
        &FP8_STREAMS
    }

    fn required_auxiliaries(&self, _: RepresentationExtent) -> &'static [AuxiliarySpec] {
        // One extent, one requirement: the codes mean nothing without it.
        &FP8_AUXILIARIES
    }

    fn validate_auxiliary(
        &self,
        _: &str,
        target: &AuxiliaryMetadata,
        shape: &[usize],
        _: RepresentationExtent,
        tensor: &str,
    ) -> Result<(), CodecError> {
        // The grid's shape is this codec's business: it must tile the
        // matrix evenly, and the tile it implies is the only tile there
        // is. Its LABEL is not judged — the grid arrives decoded through
        // its own codec, and what this one needs is values.
        Self::grid(shape, &target.shape, tensor).map(|_| ())
    }

    fn capabilities(&self) -> CodecCapabilities {
        CodecCapabilities {
            // One byte per weight and a scale looked up per element: any
            // row is addressable, and a row needs only its own tile row.
            access: AccessGranularity::RowRandom,
            group_elems: UNGROUPED,
            row_align_elems: UNGROUPED,
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
        let (rows, cols) = Self::matrix(shape, tensor)?;
        Ok((rows * cols) as u64)
    }

    fn validate(
        &self,
        operands: &CodecOperands<'_>,
        shape: &[usize],
        extent: RepresentationExtent,
        tensor: &str,
    ) -> Result<(), CodecError> {
        self.certificate_at(extent, tensor)?;
        // The codes first, from the shape alone: a short stream is refused
        // by its name before any dependency is consulted, the same order
        // every other codec judges a header in.
        let (rows, cols) = Self::matrix(shape, tensor)?;
        let codes = operands.stream(VALUES, DTYPE_FP8_BLOCK, tensor)?;
        if codes.len() != rows * cols {
            return Err(CodecError::StreamLength {
                tensor: tensor.into(),
                label: DTYPE_FP8_BLOCK.into(),
                stream: VALUES.name.into(),
                need: rows * cols,
                have: codes.len(),
            });
        }
        self.scales(operands, shape, tensor).map(|_| ())
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
        let ResolvedGrid {
            grid,
            block_rows,
            block_cols,
            values: scales,
        } = self.scales(operands, shape, tensor)?;
        let geometry = RowGeometry {
            rows: grid.rows,
            k: grid.cols,
        };
        geometry.check_rows(&rows, DTYPE_FP8_BLOCK, tensor)?;
        geometry.check_destination(&rows, dst.len(), tensor)?;
        let codes = operands.stream_of_len(VALUES, grid.elements(), DTYPE_FP8_BLOCK, tensor)?;
        // The reference's arithmetic, row by row: one f32 multiply per
        // element, the scale row hoisted per matrix row — bit-exact
        // against `fp8_finegrained::dequantize_into`, which is the
        // transcription of upstream's `Fp8Dequantize`.
        for (out, r) in dst.chunks_exact_mut(grid.cols).zip(rows) {
            let scale_row = &scales[(r / block_rows) * grid.scale_cols..][..grid.scale_cols];
            let src = &codes[r * grid.cols..][..grid.cols];
            for (c, (d, &code)) in out.iter_mut().zip(src).enumerate() {
                *d = e4m3_to_f32(code) * scale_row[c / block_cols];
            }
        }
        Ok(())
    }

    fn decode_residency(&self) -> ResidencyProfile {
        ResidencyProfile::DECODED_F32
    }

    fn accelerations(&self) -> Vec<Acceleration> {
        // The checkpoint's bytes executed in place — decoded and
        // tile-scaled in registers — with the scale grid RETAINED beside
        // them, which is the dependency lifetime the ledger prices for a
        // direct kernel over codes. `stored` at the codes' width; the
        // retained grid is accounted on the pin's dependency, not here.
        vec![Acceleration::cpu(
            PhysicalProjectionPlan::FusedFp8Block,
            ResidencyProfile::stored(Self::bits_per_weight()),
        )]
    }
}
