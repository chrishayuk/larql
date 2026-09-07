//! Everything this acceptance test contributes that `larql-vindex` does
//! not ship: a parent representation, the dependency it decodes through,
//! the role name binding them, and the authority and method that attest
//! an instance.
//!
//! Every identifier below is spelled HERE and nowhere in the crate under
//! test. `genericity` asserts that, because a plane that only works for
//! the identities its own build knows is not a plugin plane.

use std::ops::Range;

use larql_vindex::format::vindex3::represent::codec::auxiliary::AuxiliaryMetadata;
use larql_vindex::format::vindex3::represent::codec::codecs::{float, kquant, mxfp4, nvfp4};
use larql_vindex::format::vindex3::represent::codec::{
    AccessGranularity, AuxiliarySpec, CodecCapabilities, CodecError, CodecOperands, CodecRegistry,
    ExtentCertificate, FidelityCertificate, RepresentationCodec, RepresentationExtent,
    ResidencyProfile, StreamRole, StreamSpec,
};
use larql_vindex::format::vindex3::represent::nvfp4_pack::CodecIdentity;

// ── Identities this build does not know ──────────────────────────────

/// The parent representation's label and family.
pub const LATTICE: &str = "ACME_LATTICE12";
/// The dependency's.
pub const ANCHORS: &str = "ACME_ANCHORBOOK";
/// The ROLE the parent declares its dependency under — a name, not a
/// codec, and one no shipped codec uses.
pub const ANCHOR_ROLE: &str = "acme_anchor_table";
/// Who attests, and by what method.
pub const AUTHORITY: &str = "acme-metrology";
pub const METHOD: &str = "acme-montecarlo-rms";
pub const METHOD_VERSION: u32 = 1;

/// Entries in the anchor table. A code is one byte.
pub const ENTRIES: usize = 256;
/// The dtype the refinement plane is stored under.
pub const REFINE_DTYPE: &str = "U8";
/// The quantum a residual byte corrects by.
pub const LATTICE_STEP: f32 = 1.0 / 512.0;

// ── The premises ─────────────────────────────────────────────────────

/// What the anchor table certifies — a real term in every composition,
/// which is what stops the parent's own bound from being the whole story.
pub const ANCHOR_RADIUS: f64 = 0.002;
/// What the parent's SCHEME declares at its shallow extent. Conservative,
/// because it must hold for every tensor this codec will ever encode.
pub const DECLARED: f64 = 0.004;
/// What `AUTHORITY` measured on THIS instance at that extent.
pub const ATTESTED: f64 = 0.001;
/// What execution requires.
pub const FLOOR: f64 = 0.005;

/// The arithmetic that makes the arms mean what they say, checked at
/// compile time so no edit can quietly turn a witness into a tautology.
const _: () = {
    // Unrecognised evidence leaves the DECLARED bound, which misses.
    assert!(DECLARED + ANCHOR_RADIUS > FLOOR);
    // Recognised and verified evidence replaces it, and fits.
    assert!(ATTESTED + ANCHOR_RADIUS < FLOOR);
    // Neither half decides it alone.
    assert!(DECLARED < FLOOR);
    assert!(ANCHOR_RADIUS < FLOOR);
    // The parent is EXACT at its terminal extent, so any non-zero in the
    // composition comes from the dependency — which is what
    // `CertifiedExact` has to notice.
    assert!(ANCHOR_RADIUS > 0.0);
};

const VALUES: StreamSpec = StreamSpec {
    name: "acme_codes",
    role: StreamRole::Values,
};
const REFINE: StreamSpec = StreamSpec {
    name: "acme_residual",
    role: StreamRole::Refinement { depth: 1 },
};
const LATTICE_STREAMS: [StreamSpec; 2] = [VALUES, REFINE];
const ANCHOR_STREAMS: [StreamSpec; 1] = [VALUES];
const LATTICE_AUXILIARIES: [AuxiliarySpec; 1] = [AuxiliarySpec::new(ANCHOR_ROLE)];

pub fn elements(shape: &[usize]) -> usize {
    shape.iter().product::<usize>().max(1)
}

// ── The parent ───────────────────────────────────────────────────────

/// Graded extents AND a required dependency, with a revision the
/// substitution arm can change.
pub struct Lattice12 {
    pub revision: u32,
}

pub const LATTICE12: Lattice12 = Lattice12 { revision: 1 };
/// The same label at a revision this container was not written by.
pub const LATTICE12_NEXT: Lattice12 = Lattice12 { revision: 2 };

impl RepresentationCodec for Lattice12 {
    fn encoding_label(&self) -> &'static str {
        LATTICE
    }
    fn identity(&self) -> CodecIdentity {
        CodecIdentity {
            family: LATTICE.into(),
            revision: self.revision,
            group_elems: 1,
            element: "u8-lattice-index".into(),
            group_scale: "none".into(),
            tensor_scale: "none".into(),
            layout: "row-major".into(),
        }
    }
    fn streams(&self) -> &'static [StreamSpec] {
        &LATTICE_STREAMS
    }
    fn required_auxiliaries(&self, _: RepresentationExtent) -> &'static [AuxiliarySpec] {
        &LATTICE_AUXILIARIES
    }
    fn validate_auxiliary(
        &self,
        name: &str,
        target: &AuxiliaryMetadata,
        _: &[usize],
        _: RepresentationExtent,
        tensor: &str,
    ) -> Result<(), CodecError> {
        target.require_shape(&[ENTRIES], tensor, LATTICE, name)
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
                FidelityCertificate::relative_rms(DECLARED).expect("a well-formed radius"),
            ),
            // Exact at terminal, against the typed logical source it was
            // handed: the residual plane restores every value.
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
        operands.stream_of_len(VALUES, elements(shape), LATTICE, tensor)?;
        if extent.depth >= 1 {
            operands.stream_of_len(REFINE, elements(shape), LATTICE, tensor)?;
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
        let codes = operands.stream(VALUES, LATTICE, tensor)?;
        let anchors = operands.auxiliaries.require(ANCHOR_ROLE, LATTICE, tensor)?;
        let residual = if extent.depth >= 1 {
            Some(operands.stream(REFINE, LATTICE, tensor)?)
        } else {
            None
        };
        for (out, element) in dst.iter_mut().zip(rows.start * k..) {
            let base = anchors.values[usize::from(codes[element])];
            *out = match residual {
                Some(bytes) => base + f32::from(bytes[element] as i8) * LATTICE_STEP,
                None => base,
            };
        }
        Ok(())
    }
    fn decode_residency(&self) -> ResidencyProfile {
        ResidencyProfile::DECODED_F32
    }
}

// ── The dependency ───────────────────────────────────────────────────

/// The anchor table's own representation, which CERTIFIES itself — the
/// term the parent's bound must be widened by.
pub struct AnchorBook;

impl RepresentationCodec for AnchorBook {
    fn encoding_label(&self) -> &'static str {
        ANCHORS
    }
    fn identity(&self) -> CodecIdentity {
        CodecIdentity {
            family: ANCHORS.into(),
            revision: 1,
            group_elems: 1,
            element: "f32".into(),
            group_scale: "none".into(),
            tensor_scale: "none".into(),
            layout: "row-major".into(),
        }
    }
    fn streams(&self) -> &'static [StreamSpec] {
        &ANCHOR_STREAMS
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
        vec![ExtentCertificate::certified(
            0,
            32.0,
            FidelityCertificate::relative_rms(ANCHOR_RADIUS).expect("a well-formed radius"),
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
            .stream_of_len(VALUES, elements(shape) * 4, ANCHORS, tensor)
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
        let bytes = operands.stream(VALUES, ANCHORS, tensor)?;
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

// ── Registries: with both providers, and each way of losing one ──────

fn shipped() -> CodecRegistry {
    CodecRegistry::new()
        .register(Box::new(float::BF16))
        .and_then(|r| r.register(Box::new(float::F16)))
        .and_then(|r| r.register(Box::new(float::F32)))
        .and_then(|r| r.register(Box::new(kquant::Q4_K)))
        .and_then(|r| r.register(Box::new(kquant::Q6_K)))
        .and_then(|r| r.register(Box::new(kquant::Q8_0)))
        .and_then(|r| r.register(Box::new(nvfp4::NVFP4)))
        .and_then(|r| r.register(Box::new(mxfp4::MXFP4)))
        .expect("the shipped labels are distinct")
}

fn leak(r: CodecRegistry) -> &'static CodecRegistry {
    Box::leak(Box::new(r))
}

pub fn with_providers() -> &'static CodecRegistry {
    leak(
        shipped()
            .register(Box::new(LATTICE12))
            .and_then(|r| r.register(Box::new(AnchorBook)))
            .expect("two new labels"),
    )
}

/// The parent alone — the dependency's provider is gone.
pub fn without_anchor_provider() -> &'static CodecRegistry {
    leak(
        shipped()
            .register(Box::new(LATTICE12))
            .expect("one new label"),
    )
}

/// The dependency alone — the parent's provider is gone.
pub fn without_parent_provider() -> &'static CodecRegistry {
    leak(shipped().register(Box::new(AnchorBook)).expect("one label"))
}

/// Both present, but the parent SUBSTITUTED: the same label at a
/// revision that means different bytes.
pub fn with_substituted_parent() -> &'static CodecRegistry {
    leak(
        shipped()
            .register(Box::new(LATTICE12_NEXT))
            .and_then(|r| r.register(Box::new(AnchorBook)))
            .expect("two new labels"),
    )
}
