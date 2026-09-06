//! What a codec may require of another represented object, and what the
//! contract refuses before a payload is opened.
//!
//! The subject here is the DECLARATION, not any one codec: test-only
//! codecs exercise the rules — a requirement that differs by extent, and
//! one whose owner judges nothing — while the shipped registry is held to
//! what is true of all of it. `VQ8_SHARED` is the only shipped codec that
//! depends on another object; its own decode is tested beside it.

use super::super::auxiliary::{admit_auxiliary_names, AuxiliaryMetadata, AuxiliarySpec};
use super::super::codecs::vq8_shared::DTYPE_VQ8_SHARED;
use super::*;

const CODEBOOK: &str = "codebook";
const PALETTE: &str = "palette";
/// The width a `Needy` codebook must have — a rule the CODEC states, not
/// the contract.
const CODEBOOK_WIDTH: usize = 4;

/// A codec that requires one auxiliary at its base extent and two at its
/// deeper one — the extent-dependence the contract promises, in the
/// smallest form that can show it.
struct Needy;

const BASE_REQUIREMENTS: [AuxiliarySpec; 1] = [AuxiliarySpec::new(CODEBOOK)];
const DEEP_REQUIREMENTS: [AuxiliarySpec; 2] =
    [AuxiliarySpec::new(CODEBOOK), AuxiliarySpec::new(PALETTE)];

impl RepresentationCodec for Needy {
    fn encoding_label(&self) -> &'static str {
        "NEEDY"
    }
    fn identity(&self) -> super::super::super::nvfp4_pack::CodecIdentity {
        F32.identity()
    }
    fn streams(&self) -> &'static [StreamSpec] {
        F32.streams()
    }
    fn capabilities(&self) -> CodecCapabilities {
        F32.capabilities()
    }
    fn extents(&self) -> Vec<ExtentCertificate> {
        vec![
            ExtentCertificate::terminal(16.0),
            ExtentCertificate {
                extent: RepresentationExtent::at_depth(1),
                bits_per_weight: 32.0,
                radius: None,
            },
        ]
    }
    fn required_auxiliaries(&self, extent: RepresentationExtent) -> &'static [AuxiliarySpec] {
        if extent.depth == 0 {
            &BASE_REQUIREMENTS
        } else {
            &DEEP_REQUIREMENTS
        }
    }
    fn validate_auxiliary(
        &self,
        name: &str,
        target: &AuxiliaryMetadata,
        shape: &[usize],
        _: RepresentationExtent,
        tensor: &str,
    ) -> Result<(), CodecError> {
        // The codec's own rule, judged from metadata alone: a codebook is
        // [entries, width], and its width is the codec's business.
        let entries = shape.last().copied().unwrap_or(0);
        target.require_shape(
            &[entries, CODEBOOK_WIDTH],
            tensor,
            self.encoding_label(),
            name,
        )
    }
    fn stored_bytes(
        &self,
        shape: &[usize],
        _: RepresentationExtent,
        tensor: &str,
    ) -> Result<u64, CodecError> {
        // The bytes are not this stub's subject; its declarations are.
        F32.stored_bytes(shape, RepresentationExtent::BASE, tensor)
    }
    fn validate(
        &self,
        operands: &CodecOperands<'_>,
        shape: &[usize],
        _: RepresentationExtent,
        tensor: &str,
    ) -> Result<(), CodecError> {
        F32.validate(operands, shape, RepresentationExtent::BASE, tensor)
    }
    fn decode_rows(
        &self,
        operands: &CodecOperands<'_>,
        shape: &[usize],
        rows: std::ops::Range<usize>,
        _: RepresentationExtent,
        dst: &mut [f32],
        tensor: &str,
    ) -> Result<(), CodecError> {
        F32.decode_rows(
            operands,
            shape,
            rows,
            RepresentationExtent::BASE,
            dst,
            tensor,
        )
    }
    fn decode_residency(&self) -> ResidencyProfile {
        ResidencyProfile::DECODED_F32
    }
}

/// A codec that requires an object and judges nothing about it — the
/// default `validate_auxiliary` must refuse rather than accept.
struct Careless;

impl RepresentationCodec for Careless {
    fn encoding_label(&self) -> &'static str {
        "CARELESS"
    }
    fn identity(&self) -> super::super::super::nvfp4_pack::CodecIdentity {
        F32.identity()
    }
    fn streams(&self) -> &'static [StreamSpec] {
        F32.streams()
    }
    fn capabilities(&self) -> CodecCapabilities {
        F32.capabilities()
    }
    fn extents(&self) -> Vec<ExtentCertificate> {
        vec![ExtentCertificate::terminal(32.0)]
    }
    fn required_auxiliaries(&self, _: RepresentationExtent) -> &'static [AuxiliarySpec] {
        &BASE_REQUIREMENTS
    }
    fn stored_bytes(
        &self,
        shape: &[usize],
        _: RepresentationExtent,
        tensor: &str,
    ) -> Result<u64, CodecError> {
        F32.stored_bytes(shape, RepresentationExtent::BASE, tensor)
    }
    fn validate(
        &self,
        _: &CodecOperands<'_>,
        _: &[usize],
        _: RepresentationExtent,
        _: &str,
    ) -> Result<(), CodecError> {
        Ok(())
    }
    fn decode_rows(
        &self,
        _: &CodecOperands<'_>,
        _: &[usize],
        _: std::ops::Range<usize>,
        _: RepresentationExtent,
        dst: &mut [f32],
        _: &str,
    ) -> Result<(), CodecError> {
        dst.fill(0.0);
        Ok(())
    }
    fn decode_residency(&self) -> ResidencyProfile {
        ResidencyProfile::DECODED_F32
    }
}

fn metadata(shape: &[usize]) -> AuxiliaryMetadata {
    AuxiliaryMetadata {
        object: "target.codebooks".into(),
        tensor: "shared.codebook".into(),
        label: "F32".into(),
        shape: shape.to_vec(),
        identity: Some(F32.identity()),
    }
}

/// Exactly one shipped codec depends on another object, and it says which
/// object by name at every extent it declares. Every other reads only its
/// own bytes.
///
/// This asserted that NO shipped codec required anything, which was true
/// of what had been implemented rather than of the contract. What survives
/// that becoming false is stated here: the set of dependants is named, and
/// a codec that requires something requires it at every extent it offers —
/// silently dropping a requirement at one depth would make a stored
/// container's meaning depend on how much of it someone chose to read.
#[test]
fn exactly_one_shipped_codec_depends_on_another_object() {
    let dependants: Vec<&str> = builtin()
        .into_iter()
        .filter(|codec| {
            codec
                .extents()
                .iter()
                .any(|c| !codec.required_auxiliaries(c.extent).is_empty())
        })
        .map(|codec| codec.encoding_label())
        .collect();
    assert_eq!(dependants, [DTYPE_VQ8_SHARED]);

    for codec in builtin() {
        let names_at = |extent| -> Vec<&str> {
            codec
                .required_auxiliaries(extent)
                .iter()
                .map(|a| a.name)
                .collect()
        };
        let base = names_at(RepresentationExtent::BASE);
        for certificate in codec.extents() {
            assert_eq!(
                names_at(certificate.extent),
                base,
                "{} changes what it requires at depth {}",
                codec.encoding_label(),
                certificate.extent.depth
            );
        }
        if codec.encoding_label() == DTYPE_VQ8_SHARED {
            assert_eq!(base, vec![CODEBOOK]);
        } else {
            assert!(base.is_empty(), "{}", codec.encoding_label());
        }
    }
}

/// A requirement's NAME is part of the codec revision's semantics, so the
/// declaration has to be well formed: unique, non-empty, and the same
/// answer every time it is asked.
#[test]
fn requirement_names_are_unique_and_stable_at_every_extent() {
    for codec in [&Needy as &dyn RepresentationCodec, &Careless] {
        for certificate in codec.extents() {
            let required = codec.required_auxiliaries(certificate.extent);
            let mut names: Vec<&str> = required.iter().map(|a| a.name).collect();
            let asked_again: Vec<&str> = codec
                .required_auxiliaries(certificate.extent)
                .iter()
                .map(|a| a.name)
                .collect();
            assert_eq!(names, asked_again, "a declaration is not a computation");
            assert!(
                names.iter().all(|n| !n.trim().is_empty()),
                "{}",
                codec.encoding_label()
            );
            names.sort_unstable();
            let unique = names.len();
            names.dedup();
            assert_eq!(unique, names.len(), "{}", codec.encoding_label());
        }
    }
}

/// The extent decides what is required, which is what lets a codec need a
/// dependency only where it reads one.
#[test]
fn a_deeper_extent_may_require_more_than_the_base() {
    let base: Vec<&str> = Needy
        .required_auxiliaries(RepresentationExtent::BASE)
        .iter()
        .map(|a| a.name)
        .collect();
    let deep: Vec<&str> = Needy
        .required_auxiliaries(RepresentationExtent::at_depth(1))
        .iter()
        .map(|a| a.name)
        .collect();
    assert_eq!(base, vec![CODEBOOK]);
    assert_eq!(deep, vec![CODEBOOK, PALETTE]);
}

/// Exactly the declared names: a missing one and an undeclared one are
/// both refused, and each refusal says at which depth and what was
/// required.
#[test]
fn a_container_provides_exactly_what_the_extent_requires() {
    let deep = RepresentationExtent::at_depth(1);
    let required = Needy.required_auxiliaries(deep);
    admit_auxiliary_names(required, &[CODEBOOK, PALETTE], "NEEDY", "w", deep)
        .expect("exactly the declared names");
    // Order is not identity: a table is a map, not a list.
    admit_auxiliary_names(required, &[PALETTE, CODEBOOK], "NEEDY", "w", deep).unwrap();

    let err = admit_auxiliary_names(required, &[CODEBOOK], "NEEDY", "w", deep).unwrap_err();
    assert!(
        matches!(&err, CodecError::MissingAuxiliary { name, depth, required, .. }
            if name == PALETTE && *depth == 1 && required.len() == 2),
        "{err}"
    );

    let err = admit_auxiliary_names(required, &[CODEBOOK, PALETTE, "atlas"], "NEEDY", "w", deep)
        .unwrap_err();
    assert!(
        matches!(&err, CodecError::UnexpectedAuxiliary { name, depth, .. }
            if name == "atlas" && *depth == 1),
        "{err}"
    );

    // And the base extent refuses the palette the DEEPER extent needs:
    // requirements are per extent, so what is provided is judged per
    // extent too.
    let base = RepresentationExtent::BASE;
    let err = admit_auxiliary_names(
        Needy.required_auxiliaries(base),
        &[CODEBOOK, PALETTE],
        "NEEDY",
        "w",
        base,
    )
    .unwrap_err();
    assert!(
        matches!(&err, CodecError::UnexpectedAuxiliary { name, depth, .. }
            if name == PALETTE && *depth == 0),
        "{err}"
    );
}

/// The shape rule is the CODEC's, judged from the container's metadata
/// with no payload opened.
#[test]
fn the_owning_codec_judges_its_dependency_from_metadata_alone() {
    let owner_shape = [8usize, 256];
    Needy
        .validate_auxiliary(
            CODEBOOK,
            &metadata(&[256, CODEBOOK_WIDTH]),
            &owner_shape,
            RepresentationExtent::BASE,
            "w",
        )
        .expect("the codebook is the shape the codec requires");
    let err = Needy
        .validate_auxiliary(
            CODEBOOK,
            &metadata(&[256, CODEBOOK_WIDTH + 1]),
            &owner_shape,
            RepresentationExtent::BASE,
            "w",
        )
        .unwrap_err();
    assert!(
        matches!(&err, CodecError::AuxiliaryGeometry { name, tensor, why, .. }
            if name == CODEBOOK && tensor == "w" && why.contains("shared.codebook")),
        "{err}"
    );
}

/// A codec that requires an object and judges nothing about it is refused
/// by the default, rather than accepting whatever the table pointed at.
#[test]
fn requiring_an_object_without_judging_it_is_refused_by_the_default() {
    let err = Careless
        .validate_auxiliary(
            CODEBOOK,
            &metadata(&[256, CODEBOOK_WIDTH]),
            &[8, 256],
            RepresentationExtent::BASE,
            "w",
        )
        .unwrap_err();
    assert!(
        matches!(&err, CodecError::AuxiliaryUnjudged { name, label, .. }
            if name == CODEBOOK && label == "CARELESS"),
        "{err}"
    );
}
