//! Entropy-coded bf16 — the codec that is hostile to every assumption
//! rung 1 might have kept from mmap'd tensor tables.
//!
//! One zlib stream (RFC 1950 framing over RFC 1951 DEFLATE, Adler-32
//! trailer) whose inflated bytes are the row-major little-endian bf16
//! image of the tensor — exactly `2 * elements` of them. That single
//! design decision buys three properties no other registered codec has:
//!
//! * **sequential access** — no row can be reached without inflating
//!   every row before it, so [`AccessGranularity::Sequential`] is
//!   literally true rather than a conservative declaration;
//! * **instance-sized storage** — the STORED byte count depends on the
//!   values, not the shape, and falls on either side of the raw image (a
//!   repetitive tensor stores fewer bytes than raw bf16, a noise tensor
//!   more); the decoded length stays shape-derived;
//! * **no direct realization is registered** — a fused streaming
//!   inflate-and-multiply could exist one day, but none does, so this
//!   codec executes through the reference decode path and says so.
//!
//! What this file deliberately does NOT contain: an encoder (the contract
//! is decode-only; the compiler's encoder table in `represent::arena` is
//! untouched), an acceleration (the trait default answers none), or any
//! new arm in the operand widener — the inflated bytes are bf16 and go
//! through the one bf16 widening the executor already judges.
//!
//! The identity names the wire format and the element grid. It must never
//! name the library that inflates it: a stream written by any conforming
//! DEFLATE implementation decodes here, and the fixture that proves it
//! was written by a different one.

use std::ops::Range;

use flate2::{Decompress, FlushDecompress, Status};

use super::super::capability::{AccessGranularity, CodecCapabilities};
use super::super::error::CodecError;
use super::super::extent::{ExtentCertificate, RepresentationExtent, BITS_PER_BYTE};
use super::super::geometry::RowGeometry;
use super::super::residency::ResidencyProfile;
use super::super::streams::{CodecOperands, StreamSpec, VALUES};
use super::super::RepresentationCodec;
use super::vocabulary::{BYTE_ALIGN, SCALE_NONE, UNGROUPED};
use crate::format::vindex3::opplan::exec::operands::widen;
use crate::format::vindex3::represent::nvfp4_pack::CodecIdentity;

/// Segment `dtype` label of an entropy-coded bf16 tensor.
pub const DTYPE_BF16_ZLIB: &str = "BF16_ZLIB";

/// ABI revision. Revision 1 is: one RFC 1950 stream, inflating to the
/// row-major little-endian bf16 image, no preset dictionary, nothing
/// after the Adler-32 trailer.
pub const BF16_ZLIB_REVISION: u32 = 1;

/// The element grid the stream inflates to, as the identity names it.
const ELEMENT_BF16: &str = "bf16";
/// The wire format, as the identity names it: the RFC, never the library.
const LAYOUT_ROW_MAJOR_LE_ZLIB: &str = "row-major-le/zlib-rfc1950";
/// The label the inflated bytes are widened under — the executor's own
/// bf16 arm, reused rather than duplicated.
const INFLATED_DTYPE: &str = "BF16";
/// Width of one inflated element.
const BF16_WIDTH_BYTES: usize = std::mem::size_of::<u16>();
/// The rate this codec certifies: the SUPREMUM. A zlib stream over bf16
/// never asymptotically exceeds the raw rate (a stored block costs five
/// bytes per 65 535), and the achieved rate is a property of the
/// instance, carried by the container's recorded length — the same
/// division the contract already makes for fidelity.
const BITS_PER_WEIGHT_SUPREMUM: f64 = BF16_WIDTH_BYTES as f64 * BITS_PER_BYTE;

/// RFC 1950 header: `CMF` then `FLG`.
const ZLIB_HEADER_BYTES: usize = 2;
/// `CMF` low nibble: compression method. 8 is DEFLATE, the only one.
const ZLIB_CM_MASK: u8 = 0x0F;
const ZLIB_CM_DEFLATE: u8 = 8;
/// `CMF` high nibble: window size exponent minus eight; 7 is the widest.
const ZLIB_CINFO_SHIFT: u8 = 4;
const ZLIB_CINFO_MAX: u8 = 7;
/// `FLG` bit 5: a preset dictionary follows the header. Refused — a
/// dictionary is a represented object the stream would depend on, and
/// this revision declares no such dependency.
const ZLIB_FDICT: u8 = 0x20;
/// `CMF * 256 + FLG` is a multiple of this by construction.
const ZLIB_FCHECK_MODULUS: u16 = 31;

const BF16_ZLIB_STREAMS: [StreamSpec; 1] = [VALUES];

/// The entropy-coded bf16 codec.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bf16ZlibCodec;

pub const BF16_ZLIB: Bf16ZlibCodec = Bf16ZlibCodec;

/// Why an inflate stopped short of what was asked. Told apart because
/// the remedies differ: a short stream is a truncated operand, a corrupt
/// one is damage, a missing trailer is a stream that cannot vouch for
/// itself, and the last two are a container whose length disagrees with
/// its shape.
enum Inflate {
    Short,
    Corrupt(flate2::DecompressError),
    NoTrailer,
    PastTheEnd(usize),
    Trailing(usize),
}

impl Bf16ZlibCodec {
    /// Inflate the first `need` bytes of `bytes`'s image. A `whole` decode
    /// is then held to the whole contract: the stream must end exactly
    /// there, its Adler-32 must verify, and nothing may follow it.
    ///
    /// The low-level inflater rather than the reader adapter, because the
    /// adapter reports "no more input" and "the stream ended" the same
    /// way — and a stream missing its trailer must not pass for whole.
    fn inflate(bytes: &[u8], need: usize, whole: bool) -> Result<Vec<u8>, Inflate> {
        let mut inflater = Decompress::new(true);
        let mut out = vec![0u8; need];
        let mut ended = false;
        while (inflater.total_out() as usize) < need {
            let consumed = inflater.total_in() as usize;
            let produced = inflater.total_out() as usize;
            let status = inflater
                .decompress(
                    &bytes[consumed..],
                    &mut out[produced..],
                    FlushDecompress::None,
                )
                .map_err(Inflate::Corrupt)?;
            let progressed =
                inflater.total_in() as usize > consumed || inflater.total_out() as usize > produced;
            match status {
                Status::StreamEnd => {
                    ended = true;
                    break;
                }
                _ if !progressed => break,
                _ => {}
            }
        }
        if (inflater.total_out() as usize) < need {
            return Err(Inflate::Short);
        }
        if !whole {
            return Ok(out);
        }
        // One spare byte of output: a stream with more image to give
        // proves it here, and a stream with only its trailer left needs
        // no room at all.
        let mut spare = [0u8; 1];
        while !ended {
            let consumed = inflater.total_in() as usize;
            let status = inflater
                .decompress(&bytes[consumed..], &mut spare, FlushDecompress::Finish)
                .map_err(Inflate::Corrupt)?;
            if inflater.total_out() as usize > need {
                return Err(Inflate::PastTheEnd(need));
            }
            match status {
                Status::StreamEnd => ended = true,
                _ if inflater.total_in() as usize == consumed => return Err(Inflate::NoTrailer),
                _ => {}
            }
        }
        let consumed = inflater.total_in() as usize;
        if consumed != bytes.len() {
            return Err(Inflate::Trailing(bytes.len() - consumed));
        }
        Ok(out)
    }

    fn decode_error(tensor: &str, detail: impl std::fmt::Display) -> CodecError {
        CodecError::Decode {
            tensor: tensor.into(),
            label: DTYPE_BF16_ZLIB.into(),
            detail: detail.to_string(),
        }
    }

    /// Judge the two-byte RFC 1950 header — everything about the stream
    /// that CAN be judged without inflating it.
    fn check_header(bytes: &[u8], tensor: &str) -> Result<(), CodecError> {
        let (cmf, flg) = (bytes[0], bytes[1]);
        if cmf & ZLIB_CM_MASK != ZLIB_CM_DEFLATE {
            return Err(Self::decode_error(
                tensor,
                format!(
                    "zlib header names compression method {}, not DEFLATE ({ZLIB_CM_DEFLATE})",
                    cmf & ZLIB_CM_MASK
                ),
            ));
        }
        if cmf >> ZLIB_CINFO_SHIFT > ZLIB_CINFO_MAX {
            return Err(Self::decode_error(
                tensor,
                format!(
                    "zlib header declares window exponent {}, past the {ZLIB_CINFO_MAX} the \
                     format allows",
                    cmf >> ZLIB_CINFO_SHIFT
                ),
            ));
        }
        if (u16::from(cmf) << u8::BITS | u16::from(flg)) % ZLIB_FCHECK_MODULUS != 0 {
            return Err(Self::decode_error(
                tensor,
                "zlib header check bits do not verify; this is not the start of a stream",
            ));
        }
        if flg & ZLIB_FDICT != 0 {
            return Err(Self::decode_error(
                tensor,
                "zlib stream demands a preset dictionary; revision 1 declares no such \
                 dependency",
            ));
        }
        Ok(())
    }
}

impl RepresentationCodec for Bf16ZlibCodec {
    fn encoding_label(&self) -> &'static str {
        DTYPE_BF16_ZLIB
    }

    fn identity(&self) -> CodecIdentity {
        CodecIdentity {
            family: DTYPE_BF16_ZLIB.into(),
            revision: BF16_ZLIB_REVISION,
            group_elems: UNGROUPED,
            element: ELEMENT_BF16.into(),
            group_scale: SCALE_NONE.into(),
            tensor_scale: SCALE_NONE.into(),
            layout: LAYOUT_ROW_MAJOR_LE_ZLIB.into(),
        }
    }

    fn streams(&self) -> &'static [StreamSpec] {
        &BF16_ZLIB_STREAMS
    }

    fn capabilities(&self) -> CodecCapabilities {
        CodecCapabilities {
            access: AccessGranularity::Sequential,
            group_elems: UNGROUPED,
            row_align_elems: UNGROUPED,
            physical_align_bytes: BYTE_ALIGN,
        }
    }

    fn extents(&self) -> Vec<ExtentCertificate> {
        vec![ExtentCertificate::terminal(BITS_PER_WEIGHT_SUPREMUM)]
    }

    /// Refused: the stored size of an entropy-coded tensor is a property
    /// of the instance, not of its shape. The extent is judged first so
    /// an absent depth is refused as every terminal codec refuses it.
    fn stored_bytes(
        &self,
        shape: &[usize],
        extent: RepresentationExtent,
        tensor: &str,
    ) -> Result<u64, CodecError> {
        self.certificate_at(extent, tensor)?;
        RowGeometry::of(shape, DTYPE_BF16_ZLIB, tensor)?;
        Err(CodecError::InstanceSized {
            tensor: tensor.into(),
            label: DTYPE_BF16_ZLIB.into(),
        })
    }

    /// Everything judgeable without inflating: the stream is bound and
    /// its header is a DEFLATE stream with no dictionary. Length against
    /// shape is judged at decode, because only inflating can know it.
    fn validate(
        &self,
        operands: &CodecOperands<'_>,
        shape: &[usize],
        extent: RepresentationExtent,
        tensor: &str,
    ) -> Result<(), CodecError> {
        self.certificate_at(extent, tensor)?;
        RowGeometry::of(shape, DTYPE_BF16_ZLIB, tensor)?;
        let bytes = operands.stream_of_len(VALUES, ZLIB_HEADER_BYTES, DTYPE_BF16_ZLIB, tensor)?;
        Self::check_header(bytes, tensor)
    }

    /// Inflate from the start of the stream through the end of `rows`,
    /// then widen the requested rows. A range that does not start at row
    /// 0 pays for the prefix — that is what sequential means, and the
    /// planner is told so by [`AccessGranularity::Sequential`].
    ///
    /// A whole decode is held to the whole contract: the stream must
    /// inflate to exactly the bytes the shape implies, its Adler-32 must
    /// verify, and nothing may follow it. A prefix decode can judge only
    /// that the prefix exists.
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
        let geometry = RowGeometry::of(shape, DTYPE_BF16_ZLIB, tensor)?;
        geometry.check_rows(&rows, DTYPE_BF16_ZLIB, tensor)?;
        geometry.check_destination(&rows, dst.len(), tensor)?;
        let bytes = operands.stream_of_len(VALUES, ZLIB_HEADER_BYTES, DTYPE_BF16_ZLIB, tensor)?;
        Self::check_header(bytes, tensor)?;

        let row_bytes = geometry.k * BF16_WIDTH_BYTES;
        let whole = rows.end == geometry.rows;
        let staged = Self::inflate(bytes, rows.end * row_bytes, whole).map_err(|why| {
            Self::decode_error(
                tensor,
                match why {
                    Inflate::Short => format!(
                        "stream ends before row {} of {} could be inflated",
                        rows.end, geometry.rows
                    ),
                    Inflate::Corrupt(e) => format!("corrupt stream: {e}"),
                    Inflate::NoTrailer => "stream ends before its Adler-32 trailer; the image \
                                          cannot be trusted whole"
                        .to_string(),
                    Inflate::PastTheEnd(need) => {
                        format!("stream inflates past the {need} bytes shape {shape:?} implies")
                    }
                    Inflate::Trailing(extra) => format!(
                        "{extra} bytes follow the end of the zlib stream; a stream is the whole \
                         operand"
                    ),
                },
            )
        })?;

        let span = &staged[rows.start * row_bytes..rows.end * row_bytes];
        let values =
            widen(INFLATED_DTYPE, span, tensor).map_err(|e| Self::decode_error(tensor, e))?;
        dst.copy_from_slice(&values);
        Ok(())
    }

    fn decode_residency(&self) -> ResidencyProfile {
        ResidencyProfile::DECODED_F32
    }
}
