//! The contract of the three floors, stated against hand-built options.
//!
//! The carriage tests beside this one prove the planner hands the floor a
//! COMPOSED bound. This file proves the floors themselves mean three
//! different things — which is the whole reason the vocabulary was split.
//!
//! `TerminalExtent` was called `Exact`, and `admits` has always answered
//! it by comparing depths and never by reading a radius. That was
//! harmless while the only progressive codec was lossless. It stops being
//! harmless the moment a lossy codec has a terminal extent: `VQ8_SHARED`
//! at full depth is every byte the artifact holds AND no bound at all
//! against the tensor it was fitted to. The old name would have had this
//! build report "exact reconstruction" for it.

use crate::format::vindex3::opplan::exec::accounting::RepresentationFloor;
use crate::format::vindex3::opplan::exec::realization::ExtentOption;
use crate::format::vindex3::represent::codec::{
    ExtentCertificate, FidelityCertificate, MetricId, RepresentationExtent,
};

/// The deepest extent in these fixtures.
const TERMINAL: u32 = 1;
/// A radius comfortably inside any bound asserted here.
const SMALL: f64 = 0.001;
/// The bound the `Within` cases are judged against.
const BOUND: f64 = 0.005;

fn at(depth: u32, radius: Option<f64>) -> ExtentOption {
    ExtentOption {
        certificate: ExtentCertificate {
            extent: RepresentationExtent { depth },
            bits_per_weight: 8.0,
            radius: radius.map(|r| FidelityCertificate::relative_rms(r).expect("well formed")),
        },
        stored_bytes: None,
    }
}

fn terminal() -> RepresentationExtent {
    RepresentationExtent::at_depth(TERMINAL)
}

/// A lossy codec's terminal extent: every stored byte, and no claim.
/// This is `VQ8_SHARED`, and it is the case the split exists for.
fn lossy_terminal() -> ExtentOption {
    at(TERMINAL, None)
}

/// A lossless codec's terminal extent, which CAN say so. This is
/// `F32_PLANES`, whose terminal depth certifies `0.0`.
fn certified_terminal() -> ExtentOption {
    at(TERMINAL, Some(0.0))
}

#[test]
fn terminal_extent_is_structural_and_reads_no_radius() {
    let floor = RepresentationFloor::TerminalExtent;
    assert!(
        floor.admits(&lossy_terminal(), terminal()),
        "every stored byte is exactly what this floor asks for"
    );
    assert!(
        floor.admits(&certified_terminal(), terminal()),
        "and a certificate neither helps nor is needed"
    );
    assert!(
        !floor.admits(&at(0, Some(SMALL)), terminal()),
        "a shallower extent is refused however good its bound"
    );
}

/// **The decisive one for the rename.** The same option that satisfies
/// `TerminalExtent` does NOT satisfy `CertifiedExact` — so the two
/// questions can no longer be answered by one word.
#[test]
fn certified_exact_refuses_a_terminal_extent_that_certifies_nothing() {
    let lossy = lossy_terminal();
    assert!(
        RepresentationFloor::TerminalExtent.admits(&lossy, terminal()),
        "it IS the complete stored representation"
    );
    assert!(
        !RepresentationFloor::CertifiedExact.admits(&lossy, terminal()),
        "and it is NOT a certified exact reconstruction — the old `Exact` \
         conflated these and would have admitted it"
    );
    assert!(
        RepresentationFloor::CertifiedExact.admits(&certified_terminal(), terminal()),
        "a codec that can say 0.0 satisfies it"
    );
}

#[test]
fn certified_exact_admits_only_a_zero_radius() {
    let floor = RepresentationFloor::CertifiedExact;
    assert!(floor.admits(&at(TERMINAL, Some(0.0)), terminal()));
    assert!(
        !floor.admits(&at(TERMINAL, Some(f64::MIN_POSITIVE)), terminal()),
        "any positive radius is not exact, however small"
    );
}

/// `Within` is an evidence requirement too: being terminal buys nothing.
#[test]
fn within_requires_a_certificate_at_every_depth() {
    let floor = RepresentationFloor::Within(BOUND);
    assert!(
        !floor.admits(&lossy_terminal(), terminal()),
        "an undeclared error is not a small one, terminal or not"
    );
    assert!(floor.admits(&at(TERMINAL, Some(SMALL)), terminal()));
    assert!(
        floor.admits(&at(0, Some(BOUND)), terminal()),
        "at the bound"
    );
    assert!(!floor.admits(&at(0, Some(BOUND * 2.0)), terminal()));
}

/// v1 compares like with like: a bound in another metric is not converted
/// into one that would satisfy this floor.
#[test]
fn a_foreign_metric_satisfies_neither_evidence_floor() {
    let foreign = ExtentOption {
        certificate: ExtentCertificate {
            extent: RepresentationExtent { depth: TERMINAL },
            bits_per_weight: 8.0,
            radius: Some(
                FidelityCertificate::new(
                    MetricId::new("max-abs", 1).expect("well formed"),
                    crate::format::vindex3::represent::codec::DomainId::finite_normals(),
                    0.0,
                )
                .expect("well formed"),
            ),
        },
        stored_bytes: None,
    };
    assert!(!RepresentationFloor::CertifiedExact.admits(&foreign, terminal()));
    assert!(!RepresentationFloor::Within(BOUND).admits(&foreign, terminal()));
    assert!(
        RepresentationFloor::TerminalExtent.admits(&foreign, terminal()),
        "the structural floor is indifferent to the metric"
    );
}

/// The default is the structural one, so a plan that asks for nothing
/// gets no silent quality change — and is not told it got an exact
/// reconstruction either.
#[test]
fn the_default_floor_is_the_structural_one() {
    assert_eq!(
        RepresentationFloor::default(),
        RepresentationFloor::TerminalExtent
    );
    assert_eq!(
        RepresentationFloor::TerminalExtent.describe(),
        "the complete stored representation",
        "the refusal a caller reads must not say 'exact'"
    );
}
