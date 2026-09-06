//! A certificate is a bound plus the terms it is a bound in, and the
//! terms are what composition checks.

use super::super::fidelity::{DomainId, FidelityCertificate, MetricId, SemanticId};
use super::*;

/// A metric this build has never heard of — what a provider certifying in
/// its own terms would mint. The point is that it is SPELLABLE.
fn foreign_metric() -> MetricId {
    MetricId::new("max-absolute", 1).unwrap()
}

#[test]
fn a_semantic_id_is_a_name_and_a_version_and_says_so_both_ways() {
    let metric = MetricId::relative_rms();
    assert_eq!(metric.to_string(), "relative-rms@1");
    assert_eq!(metric.id().name(), "relative-rms");
    assert_eq!(metric.id().version(), 1);
    assert_eq!(MetricId::parse("relative-rms@1").unwrap(), metric);
    assert_eq!(
        DomainId::finite_normals().to_string(),
        "finite-normals@1",
        "the domain a relative bound can describe"
    );
    // A version is part of the identity: the same name at another version
    // is another metric, and composition will say so.
    assert_ne!(MetricId::new("relative-rms", 2).unwrap(), metric);
}

#[test]
fn a_malformed_id_is_refused_rather_than_normalised() {
    for spelling in [
        "",
        "relative rms@1",
        "relative-rms",
        "relative-rms@x",
        "a@b@1",
    ] {
        let err = MetricId::parse(spelling).unwrap_err();
        // The refusal names the KIND and quotes the offending text — the
        // whole spelling when the shape is wrong, the name alone when the
        // name is (`a@b@1` parses a version and refuses the name `a@b`).
        assert!(
            matches!(&err, CodecError::MalformedSemanticId { kind, given }
                if kind == MetricId::KIND && spelling.contains(given.as_str())),
            "{spelling}: {err}"
        );
    }
    // A name with whitespace or an `@` cannot be constructed either — a
    // container has to be able to write one as a single token.
    assert!(SemanticId::new("metric", "two words", 1).is_err());
    assert!(SemanticId::new("metric", "at@sign", 1).is_err());
    assert!(SemanticId::new("metric", "", 1).is_err());
    assert!(SemanticId::new("metric", "naïve", 1).is_err(), "ASCII only");
}

#[test]
fn a_radius_is_finite_and_not_negative() {
    assert_eq!(
        FidelityCertificate::relative_rms(0.0).unwrap().radius(),
        0.0,
        "exact reconstruction states zero rather than declining to state"
    );
    for bad in [f64::NAN, f64::INFINITY, -1e-9] {
        let err = FidelityCertificate::relative_rms(bad).unwrap_err();
        assert!(
            matches!(&err, CodecError::MalformedRadius { radius, .. } if radius.to_bits() == bad.to_bits()),
            "{bad}: {err}"
        );
    }
}

/// Composition adds two bounds stated in the same terms, and refuses two
/// that are not — naming both, so a reader sees what was compared.
#[test]
fn composition_adds_like_terms_and_refuses_unlike_ones() {
    let own = FidelityCertificate::relative_rms(0.004).unwrap();
    let dependency = FidelityCertificate::relative_rms(0.001).unwrap();
    let composed = own.widened_by(&dependency, TENSOR, "X").unwrap();
    assert_eq!(composed.radius(), 0.005);
    assert!(composed.comparable_with(&own));

    // Another metric: the number is not converted, the composition is
    // refused, and the refusal spells both sides.
    let foreign =
        FidelityCertificate::new(foreign_metric(), DomainId::finite_normals(), 0.001).unwrap();
    assert!(!own.comparable_with(&foreign));
    let err = own.widened_by(&foreign, TENSOR, "X").unwrap_err();
    assert!(
        matches!(&err, CodecError::IncomparableCertificates { own, other, .. }
            if own.contains("relative-rms@1") && other.contains("max-absolute@1")),
        "{err}"
    );

    // Another DOMAIN, same metric: also refused. A bound over all values
    // may well hold over finite normals — v1 simply does not INFER that,
    // because it holds no description of which domain contains which, and
    // a wrong containment would widen a guarantee silently.
    let everywhere = FidelityCertificate::new(
        MetricId::relative_rms(),
        DomainId::new("all-values", 1).unwrap(),
        0.001,
    )
    .unwrap();
    let err = own.widened_by(&everywhere, TENSOR, "X").unwrap_err();
    assert!(
        matches!(&err, CodecError::IncomparableCertificates { other, .. }
            if other.contains("all-values@1")),
        "{err}"
    );
}

/// The plugin-plane property: a provider may certify in terms this build
/// does not know, and is refused for INCOMPATIBILITY rather than for
/// being unable to say what it means.
#[test]
fn a_metric_this_build_never_heard_of_is_expressible_and_refused_by_name() {
    let foreign = FidelityCertificate::new(
        MetricId::parse("chebyshev@3").unwrap(),
        DomainId::parse("nonzero-weights@2").unwrap(),
        0.01,
    )
    .unwrap();
    assert_eq!(foreign.metric().to_string(), "chebyshev@3");
    assert_eq!(
        foreign.describe(),
        "1.000e-2 chebyshev@3 over nonzero-weights@2"
    );
    let mine = FidelityCertificate::relative_rms(0.01).unwrap();
    let err = mine.widened_by(&foreign, TENSOR, "X").unwrap_err();
    assert!(err.to_string().contains("chebyshev@3"), "{err}");
}

/// Every certificate this build ships is stated in the one metric and
/// domain its codecs mean — so a mismatch inside the shipped set would be
/// a bug rather than a plugin's prerogative.
#[test]
fn every_shipped_certificate_is_stated_in_the_builds_own_terms() {
    let mut certified = 0;
    for codec in builtin() {
        for certificate in codec.extents() {
            let Some(radius) = &certificate.radius else {
                continue;
            };
            assert_eq!(
                *radius.metric(),
                MetricId::relative_rms(),
                "{}",
                codec.encoding_label()
            );
            assert_eq!(
                *radius.domain(),
                DomainId::finite_normals(),
                "{}",
                codec.encoding_label()
            );
            certified += 1;
        }
    }
    assert_eq!(
        certified, 3,
        "the progressive codec's three extents, and no other"
    );
}
