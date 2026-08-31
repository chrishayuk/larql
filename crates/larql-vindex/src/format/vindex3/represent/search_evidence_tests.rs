//! `tests` for [`super`].

use super::super::measurement::{MeasurementStatus, TailSupport, TailSupportPolicy};
use super::*;

fn thin() -> MeasurementStatus {
    TailSupportPolicy::route_cal_1().status(Some(TailSupport {
        quantile: 0.99,
        observations: 46,
    }))
}

fn supported() -> MeasurementStatus {
    TailSupportPolicy::route_cal_1().status(Some(TailSupport {
        quantile: 0.99,
        observations: 1303,
    }))
}

#[test]
fn measurement_and_usefulness_are_orthogonal() {
    // The point of the whole layer. Diagnostic kl is thin-tailed AND
    // useful; diagnostic route mass is thin-tailed AND useless. The
    // measurement status is identical; only the calibration separates
    // them.
    let r = SearchCalibrationRegistry::route_cal_1();
    assert_eq!(thin(), thin(), "identical measurement status");
    let kl = r.evidence_for("kl p99", EvidenceScale::Diagnostic, &thin());
    let route = r.evidence_for(
        "routed mixture moved at p99",
        EvidenceScale::Diagnostic,
        &thin(),
    );
    assert!(matches!(kl, SearchEvidence::OrderingProxy { .. }));
    assert_eq!(route, SearchEvidence::Unusable);
}

#[test]
fn an_ordering_proxy_may_not_be_priced_against_the_contract() {
    // The trap this exists to close: correlation is not magnitude.
    let r = SearchCalibrationRegistry::route_cal_1();
    let kl = r.evidence_for("kl p99", EvidenceScale::Diagnostic, &thin());
    assert!(kl.orders(), "it does order candidates — rho +0.857");
    assert!(
        !kl.is_priceable(),
        "but a diagnostic 2.4e-3 may NOT be turned into a fraction of a 3.5e-3 budget"
    );
}

#[test]
fn a_well_measured_statistic_can_still_be_only_a_proxy() {
    // Route flip rate is a sound count statistic — no tail problem at
    // all — and is STILL a proxy, because the contract judges mixture
    // mass. A registration must be able to lower, not only raise.
    let r = SearchCalibrationRegistry::route_cal_1();
    let e = r.evidence_for(
        "route flip rate",
        EvidenceScale::Diagnostic,
        &MeasurementStatus::Measured,
    );
    assert!(matches!(e, SearchEvidence::OrderingProxy { .. }));
    assert!(!e.is_priceable(), "flips are not the criterion");
    assert!(e.orders());
}

#[test]
fn an_unregistered_thin_percentile_is_unusable_not_usable() {
    // The default must fail safe. A search that treated an unmeasured
    // dimension as cheap would prefer exactly the candidates whose
    // expensive dimension it could not see.
    let r = SearchCalibrationRegistry::route_cal_1();
    assert_eq!(
        r.evidence_for("some future p99", EvidenceScale::Diagnostic, &thin()),
        SearchEvidence::Unusable
    );
    // And a directly supported one needs no registration.
    assert_eq!(
        r.evidence_for("some future p99", EvidenceScale::Authority, &supported()),
        SearchEvidence::Direct
    );
}

#[test]
fn top10_is_unusable_by_absence_of_evidence_and_says_so() {
    let r = SearchCalibrationRegistry::route_cal_1();
    let e = r
        .lookup("top-10 mass displaced at p99", EvidenceScale::Diagnostic)
        .expect("registered");
    assert_eq!(e.verdict, SearchEvidence::Unusable);
    assert_eq!(e.pairs, 0, "no paired calibration has been run");
    assert!(
        e.finding.contains("way to change that is to measure it"),
        "an absence of evidence must read as a TODO, not as a property of the statistic"
    );
}

#[test]
fn the_ladder_is_ordered_and_stated_once() {
    let proxy = SearchEvidence::OrderingProxy {
        calibration: "x".into(),
    };
    let est = SearchEvidence::CalibratedEstimate {
        calibration: "x".into(),
    };
    let ranks: Vec<u8> = [
        SearchEvidence::Direct,
        est.clone(),
        proxy.clone(),
        SearchEvidence::Unusable,
    ]
    .iter()
    .map(SearchEvidence::confidence_rank)
    .collect();
    assert_eq!(ranks, [3, 2, 1, 0], "strictly descending");
    // Only the top two may produce a number a contract is priced on.
    assert!(SearchEvidence::Direct.is_priceable());
    assert!(est.is_priceable());
    assert!(!proxy.is_priceable());
    assert!(!SearchEvidence::Unusable.is_priceable());
    // Everything above Unusable can at least order.
    assert!(proxy.orders());
    assert!(!SearchEvidence::Unusable.orders());
}

#[test]
fn every_registration_names_its_evidence() {
    // A registry entry without evidence is an assumption with a
    // citation field.
    for e in SearchCalibrationRegistry::route_cal_1().entries {
        assert!(!e.id.is_empty());
        assert!(!e.finding.is_empty(), "{}", e.statistic);
        match e.verdict {
            SearchEvidence::Unusable => {}
            _ => assert!(
                e.rank_correlation.is_some() && e.pairs > 0,
                "{} claims search value and must show the measurement",
                e.statistic
            ),
        }
    }
}

#[test]
fn authority_scale_route_mass_is_priced_directly() {
    // The contract criterion at the scale it was written for: no
    // calibration needed, the measurement carries it.
    let r = SearchCalibrationRegistry::route_cal_1();
    let e = r.evidence_for(
        "routed mixture moved at p99",
        EvidenceScale::Authority,
        &supported(),
    );
    assert_eq!(e, SearchEvidence::Direct);
    assert!(e.is_priceable());
}
