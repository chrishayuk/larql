//! Two codecs that exist because no SHIPPED codec has the combination
//! this arm needs: graded extents AND a required auxiliary.
//!
//! `VQ8_SHARED` has one terminal extent and no radius; `F32_PLANES`
//! grades radii but needs no auxiliary. Without a codec that does both,
//! nothing can tell the DECLARED reading of a floor from the COMPOSED
//! one — which is exactly why the over-promise survived until now.

use std::ops::Range;

use super::vocabulary::*;
use crate::format::vindex3::represent::codec::auxiliary::AuxiliaryMetadata;
use crate::format::vindex3::represent::codec::codecs::{float, kquant, mxfp4, nvfp4};
use crate::format::vindex3::represent::codec::{
    AccessGranularity, AuxiliarySpec, CodecCapabilities, CodecError, CodecOperands, CodecRegistry,
    ExtentCertificate, FidelityCertificate, RepresentationCodec, RepresentationExtent,
    ResidencyProfile, StreamSpec,
};
use crate::format::vindex3::represent::nvfp4_pack::CodecIdentity;

/// A palette codec with two extents and a required codebook — the
/// combination no shipped codec has, and the only combination under which
/// the declared and composed readings of a floor differ.
struct GradedPalette;

impl RepresentationCodec for GradedPalette {
    fn encoding_label(&self) -> &'static str {
        OWNER_LABEL
    }
    fn identity(&self) -> CodecIdentity {
        CodecIdentity {
            family: OWNER_LABEL.into(),
            revision: 1,
            group_elems: 1,
            element: "u8-index".into(),
            group_scale: "none".into(),
            tensor_scale: "none".into(),
            layout: "row-major".into(),
        }
    }
    fn streams(&self) -> &'static [StreamSpec] {
        &OWNER_STREAMS
    }
    fn required_auxiliaries(&self, _: RepresentationExtent) -> &'static [AuxiliarySpec] {
        // At every depth: a code means nothing without the palette.
        &OWNER_AUXILIARIES
    }
    fn validate_auxiliary(
        &self,
        name: &str,
        target: &AuxiliaryMetadata,
        _: &[usize],
        _: RepresentationExtent,
        tensor: &str,
    ) -> Result<(), CodecError> {
        // The palette's shape is this codec's business: one entry per
        // code, and a code is a byte.
        target.require_shape(&[ENTRIES], tensor, OWNER_LABEL, name)
    }
    fn capabilities(&self) -> CodecCapabilities {
        CodecCapabilities {
            access: AccessGranularity::ElementRandom,
            group_elems: 1,
            row_align_elems: 1,
            physical_align_bytes: 1,
        }
    }
    fn extents(&self) -> Vec<ExtentCertificate> {
        vec![
            ExtentCertificate::certified(
                0,
                8.0,
                FidelityCertificate::relative_rms(PARENT).expect("a well-formed radius"),
            ),
            // Terminal: the residual plane restores the values exactly,
            // so it SAYS 0.0. Declining to state would have been the
            // dishonest option — under the settled referent an absent
            // radius means "no claim about the source", which is not what
            // this extent does.
            ExtentCertificate::certified(
                1,
                16.0,
                FidelityCertificate::relative_rms(0.0).expect("a well-formed radius"),
            ),
        ]
    }
    fn stored_bytes(
        &self,
        shape: &[usize],
        extent: RepresentationExtent,
        tensor: &str,
    ) -> Result<u64, CodecError> {
        self.certificate_at(extent, tensor)?;
        let per = if extent.depth == 0 { 1 } else { 2 };
        Ok((elements(shape) * per) as u64)
    }
    fn validate(
        &self,
        operands: &CodecOperands<'_>,
        shape: &[usize],
        extent: RepresentationExtent,
        tensor: &str,
    ) -> Result<(), CodecError> {
        operands.stream_of_len(VALUES, elements(shape), OWNER_LABEL, tensor)?;
        if extent.depth >= 1 {
            operands.stream_of_len(REFINE, elements(shape), OWNER_LABEL, tensor)?;
        }
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
        let k = shape.last().copied().unwrap_or(1);
        let codes = operands.stream(VALUES, OWNER_LABEL, tensor)?;
        let palette = operands
            .auxiliaries
            .require(CODEBOOK, OWNER_LABEL, tensor)?
            .values;
        let residual = if extent.depth >= 1 {
            Some(operands.stream(REFINE, OWNER_LABEL, tensor)?)
        } else {
            None
        };
        for (out, element) in dst.iter_mut().zip(rows.start * k..) {
            let base = palette[usize::from(codes[element])];
            *out = match residual {
                // The residual plane is a signed byte of the palette step,
                // which is what makes depth 1 terminal rather than merely
                // better.
                Some(bytes) => base + f32::from(bytes[element] as i8) * PALETTE_STEP,
                None => base,
            };
        }
        Ok(())
    }
    fn decode_residency(&self) -> ResidencyProfile {
        ResidencyProfile::DECODED_F32
    }
}

// ── The dependency, at two fidelities ────────────────────────────────

/// A codebook codec that CERTIFIES its own reconstruction — the term the
/// owner's bound has to be widened by.
struct Codebook {
    label: &'static str,
    radius: f64,
}

const COARSEBOOK: Codebook = Codebook {
    label: COARSE_LABEL,
    radius: COARSE,
};
const FINEBOOK: Codebook = Codebook {
    label: FINE_LABEL,
    radius: FINE,
};

impl RepresentationCodec for Codebook {
    fn encoding_label(&self) -> &'static str {
        self.label
    }
    fn identity(&self) -> CodecIdentity {
        CodecIdentity {
            family: self.label.into(),
            revision: 1,
            group_elems: 1,
            element: "f32".into(),
            group_scale: "none".into(),
            tensor_scale: "none".into(),
            layout: "row-major".into(),
        }
    }
    fn streams(&self) -> &'static [StreamSpec] {
        &BOOK_STREAMS
    }
    fn capabilities(&self) -> CodecCapabilities {
        CodecCapabilities {
            access: AccessGranularity::ElementRandom,
            group_elems: 1,
            row_align_elems: 1,
            physical_align_bytes: 1,
        }
    }
    fn extents(&self) -> Vec<ExtentCertificate> {
        // One extent, which is therefore its terminal one — and unlike
        // `ExtentCertificate::terminal` it CERTIFIES that extent, because
        // a lossy codebook's error is what the owner must inherit.
        vec![ExtentCertificate::certified(
            0,
            32.0,
            FidelityCertificate::relative_rms(self.radius).expect("a well-formed radius"),
        )]
    }
    fn stored_bytes(
        &self,
        shape: &[usize],
        extent: RepresentationExtent,
        tensor: &str,
    ) -> Result<u64, CodecError> {
        self.certificate_at(extent, tensor)?;
        Ok((elements(shape) * 4) as u64)
    }
    fn validate(
        &self,
        operands: &CodecOperands<'_>,
        shape: &[usize],
        _: RepresentationExtent,
        tensor: &str,
    ) -> Result<(), CodecError> {
        operands
            .stream_of_len(VALUES, elements(shape) * 4, self.label, tensor)
            .map(|_| ())
    }
    fn decode_rows(
        &self,
        operands: &CodecOperands<'_>,
        shape: &[usize],
        rows: Range<usize>,
        _: RepresentationExtent,
        dst: &mut [f32],
        tensor: &str,
    ) -> Result<(), CodecError> {
        let k = shape.last().copied().unwrap_or(1);
        let bytes = operands.stream(VALUES, self.label, tensor)?;
        for (out, element) in dst.iter_mut().zip(rows.start * k..) {
            let at = element * 4;
            *out = f32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]);
        }
        Ok(())
    }
    fn decode_residency(&self) -> ResidencyProfile {
        ResidencyProfile::DECODED_F32
    }
}

/// Enough shipped codecs for the fixture's other tensors, plus the
/// owner and BOTH codebooks — one registry, so the arms differ only in
/// what the container stores.
pub(super) fn registry() -> &'static CodecRegistry {
    use std::sync::OnceLock;
    static ONCE: OnceLock<CodecRegistry> = OnceLock::new();
    ONCE.get_or_init(|| {
        CodecRegistry::new()
            .register(Box::new(float::BF16))
            .and_then(|r| r.register(Box::new(float::F16)))
            .and_then(|r| r.register(Box::new(float::F32)))
            .and_then(|r| r.register(Box::new(kquant::Q4_K)))
            .and_then(|r| r.register(Box::new(kquant::Q6_K)))
            .and_then(|r| r.register(Box::new(kquant::Q8_0)))
            .and_then(|r| r.register(Box::new(nvfp4::NVFP4)))
            .and_then(|r| r.register(Box::new(mxfp4::MXFP4)))
            .and_then(|r| r.register(Box::new(GradedPalette)))
            .and_then(|r| r.register(Box::new(COARSEBOOK)))
            .and_then(|r| r.register(Box::new(FINEBOOK)))
            .expect("distinct labels")
    })
}
