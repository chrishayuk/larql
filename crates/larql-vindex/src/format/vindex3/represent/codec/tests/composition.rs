//! Composed fidelity: what a caller may rely on when one representation
//! decodes through another.
//!
//! Two controls, because one of them cannot succeed and that is the
//! result rather than a gap:
//!
//! * an UNCERTIFIED parent — `VQ8_SHARED`, whose assignment error is a
//!   fact about the artifact and the encoder that made it, not about the
//!   format — composes to nothing, however well its codebook is
//!   certified, and a quality floor therefore cannot be met by it;
//! * a CERTIFIED parent composes generically: its own radius widened by
//!   its dependency's, and a floor that the composition misses forces a
//!   deeper dependency extent rather than being quietly satisfied by the
//!   parent's half of the story.

use std::collections::BTreeMap;

use super::super::auxiliary::{AuxiliaryMetadata, AuxiliarySpec};
use super::super::codecs::f32_planes::F32_PLANES;
use super::super::codecs::vq8_shared::{CODEBOOK, DTYPE_VQ8_SHARED, VQ8_SHARED};
use super::super::fidelity::{DomainId, FidelityCertificate, MetricId};
use super::*;

/// The parent's OWN error, if something attested it: a certified twin of
/// `VQ8_SHARED` that declares an assignment radius. Test-only on purpose
/// — a codec cannot honestly declare this, and the wave's finding is that
/// it would have to come from the artifact instead.
struct CertifiedVq {
    radius: f64,
    metric: MetricId,
}

const VQ_REQUIREMENTS: [AuxiliarySpec; 1] = [AuxiliarySpec::new(CODEBOOK)];

impl CertifiedVq {
    fn own(radius: f64) -> Self {
        Self {
            radius,
            metric: MetricId::relative_rms(),
        }
    }

    /// The same, certifying in terms this build does not use — for the
    /// arm where composition must refuse rather than convert.
    fn in_another_metric(radius: f64) -> Self {
        Self {
            radius,
            metric: MetricId::new("max-absolute", 1).unwrap(),
        }
    }
}

impl RepresentationCodec for CertifiedVq {
    fn encoding_label(&self) -> &'static str {
        "VQ8_ATTESTED"
    }
    fn identity(&self) -> super::super::super::nvfp4_pack::CodecIdentity {
        VQ8_SHARED.identity()
    }
    fn streams(&self) -> &'static [StreamSpec] {
        VQ8_SHARED.streams()
    }
    fn capabilities(&self) -> CodecCapabilities {
        VQ8_SHARED.capabilities()
    }
    fn required_auxiliaries(&self, _: RepresentationExtent) -> &'static [AuxiliarySpec] {
        &VQ_REQUIREMENTS
    }
    fn validate_auxiliary(
        &self,
        name: &str,
        target: &AuxiliaryMetadata,
        shape: &[usize],
        extent: RepresentationExtent,
        tensor: &str,
    ) -> Result<(), CodecError> {
        VQ8_SHARED.validate_auxiliary(name, target, shape, extent, tensor)
    }
    fn extents(&self) -> Vec<ExtentCertificate> {
        vec![ExtentCertificate::certified(
            0,
            2.0,
            FidelityCertificate::new(self.metric.clone(), DomainId::finite_normals(), self.radius)
                .unwrap(),
        )]
    }
    fn stored_bytes(
        &self,
        shape: &[usize],
        extent: RepresentationExtent,
        tensor: &str,
    ) -> Result<u64, CodecError> {
        VQ8_SHARED.stored_bytes(shape, extent, tensor)
    }
    fn validate(
        &self,
        operands: &CodecOperands<'_>,
        shape: &[usize],
        extent: RepresentationExtent,
        tensor: &str,
    ) -> Result<(), CodecError> {
        VQ8_SHARED.validate(operands, shape, extent, tensor)
    }
    fn decode_rows(
        &self,
        operands: &CodecOperands<'_>,
        shape: &[usize],
        rows: std::ops::Range<usize>,
        extent: RepresentationExtent,
        dst: &mut [f32],
        tensor: &str,
    ) -> Result<(), CodecError> {
        VQ8_SHARED.decode_rows(operands, shape, rows, extent, dst, tensor)
    }
    fn decode_residency(&self) -> ResidencyProfile {
        ResidencyProfile::DECODED_F32
    }
}

/// What a codebook stored as planes certifies at each of its extents.
fn codebook_certificates() -> Vec<FidelityCertificate> {
    F32_PLANES
        .extents()
        .into_iter()
        .map(|c| c.radius.expect("every plane extent certifies"))
        .collect()
}

fn with_codebook(certificate: &FidelityCertificate) -> BTreeMap<String, FidelityCertificate> {
    BTreeMap::from([(CODEBOOK.to_string(), certificate.clone())])
}

/// The first control: however well the codebook is certified, an
/// uncertified parent composes to nothing — and the refusal says why.
#[test]
fn an_uncertified_parent_composes_to_nothing_however_good_its_dependency_is() {
    for certificate in codebook_certificates() {
        let err = VQ8_SHARED
            .composed_certificate(
                RepresentationExtent::BASE,
                &with_codebook(&certificate),
                TENSOR,
            )
            .unwrap_err();
        assert!(
            matches!(&err, CodecError::CertificateUnavailable { label, why, .. }
                if label == DTYPE_VQ8_SHARED && why.contains("no radius of its own")),
            "codebook at {}: {err}",
            certificate.describe()
        );
    }
    // Including the exact codebook: a perfect dependency does not supply
    // the parent's own assignment error.
    let exact = codebook_certificates().pop().expect("a terminal extent");
    assert_eq!(exact.radius(), 0.0);
    assert!(VQ8_SHARED
        .composed_certificate(RepresentationExtent::BASE, &with_codebook(&exact), TENSOR)
        .is_err());
}

/// The second control: a parent whose own error IS attested composes
/// generically — the sum, in the shared terms, and the dependency's
/// extent moves it.
#[test]
fn a_certified_parent_composes_its_own_radius_with_its_dependencys() {
    let parent = CertifiedVq::own(0.01);
    let books = codebook_certificates();
    let (shallow, exact) = (books.first().unwrap(), books.last().unwrap());

    let with_exact = parent
        .composed_certificate(RepresentationExtent::BASE, &with_codebook(exact), TENSOR)
        .unwrap();
    assert_eq!(
        with_exact.radius(),
        0.01,
        "an exact dependency adds nothing"
    );

    let with_shallow = parent
        .composed_certificate(RepresentationExtent::BASE, &with_codebook(shallow), TENSOR)
        .unwrap();
    assert_eq!(with_shallow.radius(), 0.01 + shallow.radius());
    assert!(
        with_shallow.radius() > with_exact.radius(),
        "a shallower codebook widens what its owner can promise"
    );
    // And the terms survive composition: a composed bound is still a
    // relative RMS over finite normals, or it would not be comparable
    // with anything.
    assert_eq!(*with_shallow.metric(), MetricId::relative_rms());
    assert_eq!(*with_shallow.domain(), DomainId::finite_normals());
}

/// A floor is evaluated against the COMPOSED bound, so a dependency's
/// extent can decide whether its owner meets it. Reading the two
/// certificates side by side would pass a plan the composition refuses.
#[test]
fn a_floor_is_met_or_missed_by_the_composition_not_by_either_half() {
    let parent = CertifiedVq::own(0.004);
    let books = codebook_certificates();
    let (shallow, exact) = (books.first().unwrap(), books.last().unwrap());
    // A floor the parent alone clears with room to spare.
    let floor = 0.005;
    assert!(parent.extents()[0].radius.as_ref().unwrap().radius() <= floor);

    let composed_shallow = parent
        .composed_certificate(RepresentationExtent::BASE, &with_codebook(shallow), TENSOR)
        .unwrap();
    assert!(
        composed_shallow.radius() > floor,
        "the shallow codebook pushes the pair over the floor: {}",
        composed_shallow.describe()
    );
    let composed_exact = parent
        .composed_certificate(RepresentationExtent::BASE, &with_codebook(exact), TENSOR)
        .unwrap();
    assert!(
        composed_exact.radius() <= floor,
        "and a deeper codebook brings it back under: {}",
        composed_exact.describe()
    );
}

/// Composition refuses terms it cannot add, rather than converting them.
#[test]
fn a_parent_and_a_dependency_in_different_terms_do_not_compose() {
    let parent = CertifiedVq::in_another_metric(0.004);
    let books = codebook_certificates();
    let err = parent
        .composed_certificate(
            RepresentationExtent::BASE,
            &with_codebook(books.first().unwrap()),
            TENSOR,
        )
        .unwrap_err();
    assert!(
        matches!(&err, CodecError::IncomparableCertificates { own, other, .. }
            if own.contains("max-absolute@1") && other.contains("relative-rms@1")),
        "{err}"
    );
}

/// A dependency that certifies nothing leaves the composition unavailable
/// too — the parent's own bound is not the answer on its own.
#[test]
fn a_dependency_that_certifies_nothing_leaves_nothing_to_compose() {
    let parent = CertifiedVq::own(0.004);
    let err = parent
        .composed_certificate(RepresentationExtent::BASE, &BTreeMap::new(), TENSOR)
        .unwrap_err();
    assert!(
        matches!(&err, CodecError::CertificateUnavailable { why, .. }
            if why.contains(CODEBOOK) && why.contains("certifies nothing")),
        "{err}"
    );
}

/// A codec that depends on nothing composes to its own certificate — the
/// default is the same rule with an empty sum, not a special case.
#[test]
fn a_codec_without_dependencies_composes_to_what_it_declares() {
    for certificate in F32_PLANES.extents() {
        let composed = F32_PLANES
            .composed_certificate(certificate.extent, &BTreeMap::new(), TENSOR)
            .unwrap();
        assert_eq!(composed, certificate.radius.unwrap());
    }
    // And a codec that declares no radius has nothing to compose, whether
    // or not it depends on anything.
    let err = super::super::codecs::float::F32
        .composed_certificate(RepresentationExtent::BASE, &BTreeMap::new(), TENSOR)
        .unwrap_err();
    assert!(
        matches!(&err, CodecError::CertificateUnavailable { why, .. }
            if why.contains("no radius of its own")),
        "{err}"
    );
}
