//! `F32_PLANES`: an f32 image cut into byte planes, so one stored
//! representation offers three fidelities and a caller chooses how much of
//! it to read.
//!
//! The first codec in this build with more than one extent, and it exists
//! to test the contract rather than to ship a compression scheme. The
//! scheme is therefore the simplest one that is EXACT at the end: the
//! planes PARTITION the source bytes, so nothing is stored twice and the
//! deepest extent reconstructs the original bit pattern by construction
//! rather than by a numerical argument.
//!
//! ```text
//! f32 bit pattern   s eeeeeeee mmmmmmm mmmmmmmm mmmmmmmm
//!                   ├── base_hi16 ───┤├refine_8a┤├refine_8b┤
//! depth 0   16 bits  sign, exponent and 7 mantissa bits
//! depth 1   24 bits  + the next 8
//! depth 2   32 bits  + the last 8 — the source, exactly
//! ```
//!
//! **What is certified.** Truncation is one-sided, so a shallow extent
//! reads low. For a FINITE NORMAL value the significand is `1.m` in
//! `[1, 2)`, the discarded residue is under one ulp of the bits kept, and
//! the relative error is under `2^-7` at depth 0 and `2^-15` at depth 1.
//! That is the domain the declared radius covers, and
//! [`conformance`](super::super::conformance) checks it rather than
//! trusting it.
//!
//! **What is defined but not certified.** Signed zeroes and infinities
//! decode exactly at every depth, because the sign and exponent live
//! entirely in the base plane. A subnormal truncates toward zero and may
//! flush to `±0`, which no relative bound can describe. And a NaN whose
//! payload exists only in an omitted plane decodes as an INFINITY of the
//! same sign — the one case where the pattern's CLASS changes, named here
//! rather than left to be discovered. [`Domain`] reads all three cases off
//! the base plane alone, without opening a deeper one, so a caller can
//! know when a shallow decode has left the certified domain.
//!
//! The codec declares no direct realization: it executes through the
//! mandatory canonical decode, which is the point — an extent must be
//! selectable without any kernel knowing extents exist.

use std::ops::Range;

use super::super::capability::{AccessGranularity, CodecCapabilities};
use super::super::error::CodecError;
use super::super::extent::{ExtentCertificate, RepresentationExtent, BITS_PER_BYTE};
use super::super::fidelity::FidelityCertificate;
use super::super::geometry::RowGeometry;
use super::super::residency::ResidencyProfile;
use super::super::streams::{CodecOperands, StreamRole, StreamSpec};
use super::super::RepresentationCodec;
use super::vocabulary::{BYTE_ALIGN, SCALE_NONE, UNGROUPED};
use crate::format::vindex3::represent::nvfp4_pack::CodecIdentity;

/// The segment / graph label for an f32 image stored as byte planes.
pub const DTYPE_F32_PLANES: &str = "F32_PLANES";
/// ABI revision. Revision 1 is: three planes of the IEEE-754 binary32 bit
/// pattern, the high half first as a little-endian `u16` and then two
/// single bytes, each plane row-major over the whole tensor.
pub const F32_PLANES_REVISION: u32 = 1;

/// The element grid, as the identity names it.
const ELEMENT_F32: &str = "f32";
/// The arrangement, as the identity names it: planes, not interleaving.
const LAYOUT_BYTE_PLANES: &str = "row-major/byte-planes-le";

/// Bytes of the base plane per element: the high half of the pattern.
const BASE_PLANE_BYTES: usize = 2;
/// Bytes of each refinement plane per element.
const REFINEMENT_PLANE_BYTES: usize = 1;
/// Mantissa bits the base plane keeps, implicit leading bit excluded.
const BASE_MANTISSA_BITS: u32 = 7;
/// Mantissa bits one refinement plane adds.
const REFINEMENT_MANTISSA_BITS: u32 = 8;
/// Bits the base plane's `u16` is shifted up by to sit in the pattern.
const BASE_SHIFT: u32 = 16;
/// Bits the first refinement plane is shifted up by.
const REFINE_A_SHIFT: u32 = 8;
/// The square root of three, the divisor turning a uniform residue's
/// width into its RMS.
const SQRT_3: f64 = 1.732_050_807_568_877_2;

/// The high half of the pattern: sign, exponent and the top mantissa bits.
pub const BASE_HI16: StreamSpec = StreamSpec {
    name: "base_hi16",
    role: StreamRole::Values,
};
/// The third byte of the pattern, read at depth 1 and above.
pub const REFINE_8A: StreamSpec = StreamSpec {
    name: "refine_8a",
    role: StreamRole::Refinement { depth: 1 },
};
/// The fourth byte of the pattern, read only at the terminal extent.
pub const REFINE_8B: StreamSpec = StreamSpec {
    name: "refine_8b",
    role: StreamRole::Refinement { depth: 2 },
};

/// Every plane, in the order a decode reads them.
const F32_PLANES_STREAMS: [StreamSpec; 3] = [BASE_HI16, REFINE_8A, REFINE_8B];

/// The deepest extent: every plane, the source exactly.
pub const TERMINAL_DEPTH: u32 = (F32_PLANES_STREAMS.len() - 1) as u32;

/// An f32 image stored as byte planes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct F32PlanesCodec;

pub const F32_PLANES: F32PlanesCodec = F32PlanesCodec;

/// What a shallow decode would find outside the certified domain, read
/// from the base plane alone.
///
/// The exponent field lives entirely in `base_hi16`, so every one of these
/// counts is available at depth 0 without opening a refinement plane —
/// which is what lets a caller know it has left the numeric domain
/// instead of discovering it in a result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Domain {
    /// Elements whose exponent is all ones: infinities, and NaNs.
    pub non_finite: usize,
    /// Elements whose exponent is all zeros: `±0` and subnormals, which a
    /// shallow extent cannot bound relatively and may flush to zero.
    pub subnormal_or_zero: usize,
}

impl Domain {
    /// Whether every element is a finite normal — the domain the declared
    /// radius certifies.
    pub const fn is_certified(self) -> bool {
        self.non_finite == 0 && self.subnormal_or_zero == 0
    }
}

impl F32PlanesCodec {
    /// Stored bytes per element at `depth`: the base plane plus one byte
    /// for each refinement the extent reads.
    const fn bytes_per_element(depth: u32) -> usize {
        BASE_PLANE_BYTES + (depth as usize) * REFINEMENT_PLANE_BYTES
    }

    /// Mantissa bits kept at `depth`, the implicit leading bit excluded.
    const fn mantissa_bits(depth: u32) -> u32 {
        BASE_MANTISSA_BITS + depth * REFINEMENT_MANTISSA_BITS
    }

    /// The RMS of a truncation residue uniform over one ulp of the bits
    /// kept — the bound this codec certifies for finite normal values.
    /// Zero at the terminal extent, which is exact.
    fn relative_rms_bound(depth: u32) -> f64 {
        if depth >= TERMINAL_DEPTH {
            return 0.0;
        }
        // The significand is at least 1.0, so one ulp of `m` kept mantissa
        // bits is at most 2^-m relative.
        2f64.powi(-(Self::mantissa_bits(depth) as i32)) / SQRT_3
    }

    /// The planes an extent at `depth` reads, in declaration order.
    fn planes_at(depth: u32) -> &'static [StreamSpec] {
        let read = (depth as usize + 1).min(F32_PLANES_STREAMS.len());
        &F32_PLANES_STREAMS[..read]
    }

    /// Bytes one plane holds for `elements` elements.
    const fn plane_bytes(plane: usize, elements: usize) -> usize {
        if plane == 0 {
            elements * BASE_PLANE_BYTES
        } else {
            elements * REFINEMENT_PLANE_BYTES
        }
    }

    /// Cut an f32 image into its planes: `(base_hi16, refine_8a,
    /// refine_8b)`, each row-major over the whole image.
    ///
    /// The encoder side of the format lives with the decoder it must
    /// match, and it works on BIT PATTERNS with explicit little-endian
    /// halves, so the bytes a big-endian host writes are the same bytes.
    pub fn encode_planes(values: &[f32]) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let mut base = Vec::with_capacity(values.len() * BASE_PLANE_BYTES);
        let mut refine_a = Vec::with_capacity(values.len());
        let mut refine_b = Vec::with_capacity(values.len());
        for value in values {
            let bits = value.to_bits();
            base.extend_from_slice(&((bits >> BASE_SHIFT) as u16).to_le_bytes());
            refine_a.push((bits >> REFINE_A_SHIFT) as u8);
            refine_b.push(bits as u8);
        }
        (base, refine_a, refine_b)
    }

    /// What a shallow decode of `base` would leave uncertified: the
    /// non-finite and subnormal-or-zero counts, read from the base plane
    /// alone.
    pub fn domain_of_base_plane(base: &[u8]) -> Domain {
        const EXPONENT_MASK: u16 = 0x7f80;
        let mut domain = Domain::default();
        for pair in base.chunks_exact(BASE_PLANE_BYTES) {
            let hi = u16::from_le_bytes([pair[0], pair[1]]);
            match hi & EXPONENT_MASK {
                EXPONENT_MASK => domain.non_finite += 1,
                0 => domain.subnormal_or_zero += 1,
                _ => {}
            }
        }
        domain
    }

    /// Refuse an extent this codec does not declare, and answer the depth
    /// it names.
    fn depth_of(&self, extent: RepresentationExtent, tensor: &str) -> Result<u32, CodecError> {
        self.certificate_at(extent, tensor)?;
        Ok(extent.depth)
    }

    /// The planes an extent needs, bound and EXACTLY the right length.
    ///
    /// Exactly, per stream, per extent: a total is not enough, because a
    /// base plane one byte short and a refinement one byte long total
    /// correctly and mean nothing.
    fn planes_of<'a>(
        &self,
        operands: &CodecOperands<'a>,
        depth: u32,
        elements: usize,
        tensor: &str,
    ) -> Result<Vec<&'a [u8]>, CodecError> {
        Self::planes_at(depth)
            .iter()
            .enumerate()
            .map(|(plane, spec)| {
                let bytes = operands.stream(*spec, DTYPE_F32_PLANES, tensor)?;
                let need = Self::plane_bytes(plane, elements);
                if bytes.len() != need {
                    return Err(CodecError::StreamLength {
                        tensor: tensor.into(),
                        label: DTYPE_F32_PLANES.into(),
                        stream: spec.name.into(),
                        need,
                        have: bytes.len(),
                    });
                }
                Ok(bytes)
            })
            .collect()
    }
}

impl RepresentationCodec for F32PlanesCodec {
    fn encoding_label(&self) -> &'static str {
        DTYPE_F32_PLANES
    }

    fn identity(&self) -> CodecIdentity {
        CodecIdentity {
            family: DTYPE_F32_PLANES.into(),
            revision: F32_PLANES_REVISION,
            group_elems: UNGROUPED,
            element: ELEMENT_F32.into(),
            group_scale: SCALE_NONE.into(),
            tensor_scale: SCALE_NONE.into(),
            layout: LAYOUT_BYTE_PLANES.into(),
        }
    }

    fn streams(&self) -> &'static [StreamSpec] {
        &F32_PLANES_STREAMS
    }

    fn capabilities(&self) -> CodecCapabilities {
        CodecCapabilities {
            // Every plane is a fixed width per element, so any element is
            // addressable in each of them without decoding what precedes.
            access: AccessGranularity::ElementRandom,
            group_elems: UNGROUPED,
            row_align_elems: UNGROUPED,
            physical_align_bytes: BYTE_ALIGN,
        }
    }

    fn extents(&self) -> Vec<ExtentCertificate> {
        (0..=TERMINAL_DEPTH)
            .map(|depth| {
                ExtentCertificate::certified(
                    depth,
                    Self::bytes_per_element(depth) as f64 * BITS_PER_BYTE,
                    // The metric and domain this codec always meant, now
                    // said rather than implied: relative RMS, over finite
                    // normal values. Its edges (subnormals, infinities,
                    // NaN payloads) are outside the domain, and named in
                    // this module's own documentation.
                    FidelityCertificate::relative_rms(Self::relative_rms_bound(depth))
                        .expect("a bound derived from a shift is finite and not negative"),
                )
            })
            .collect()
    }

    fn stored_bytes(
        &self,
        shape: &[usize],
        extent: RepresentationExtent,
        tensor: &str,
    ) -> Result<u64, CodecError> {
        let depth = self.depth_of(extent, tensor)?;
        let geometry = RowGeometry::of(shape, DTYPE_F32_PLANES, tensor)?;
        let elements = geometry.elements(DTYPE_F32_PLANES, tensor)?;
        Ok((elements * Self::bytes_per_element(depth)) as u64)
    }

    fn validate(
        &self,
        operands: &CodecOperands<'_>,
        shape: &[usize],
        extent: RepresentationExtent,
        tensor: &str,
    ) -> Result<(), CodecError> {
        let depth = self.depth_of(extent, tensor)?;
        let geometry = RowGeometry::of(shape, DTYPE_F32_PLANES, tensor)?;
        let elements = geometry.elements(DTYPE_F32_PLANES, tensor)?;
        self.planes_of(operands, depth, elements, tensor)?;
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
        let depth = self.depth_of(extent, tensor)?;
        let geometry = RowGeometry::of(shape, DTYPE_F32_PLANES, tensor)?;
        let elements = geometry.elements(DTYPE_F32_PLANES, tensor)?;
        geometry.check_rows(&rows, DTYPE_F32_PLANES, tensor)?;
        geometry.check_destination(&rows, dst.len(), tensor)?;
        let planes = self.planes_of(operands, depth, elements, tensor)?;
        let base = planes[0];
        for (out, element) in dst.iter_mut().zip(rows.start * geometry.k..) {
            let hi = u16::from_le_bytes([
                base[element * BASE_PLANE_BYTES],
                base[element * BASE_PLANE_BYTES + 1],
            ]);
            let mut bits = u32::from(hi) << BASE_SHIFT;
            if let Some(refine_a) = planes.get(1) {
                bits |= u32::from(refine_a[element]) << REFINE_A_SHIFT;
            }
            if let Some(refine_b) = planes.get(2) {
                bits |= u32::from(refine_b[element]);
            }
            *out = f32::from_bits(bits);
        }
        Ok(())
    }

    fn decode_residency(&self) -> ResidencyProfile {
        // Canonical decode widens to f32 at EVERY depth: what a shallower
        // extent saves is what is read, never what is held.
        ResidencyProfile::DECODED_F32
    }
}
