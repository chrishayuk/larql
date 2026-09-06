//! Residency profiles, extents and the declared-cost arithmetic.

use super::super::fidelity::FidelityCertificate;
use super::*;
use crate::format::vindex3::opplan::exec::cpu::physical::PhysicalProjectionPlan;

#[test]
fn the_decode_profile_is_an_f32_image_and_direct_profiles_are_the_stored_bits() {
    let decoded = ResidencyProfile::DECODED_F32;
    assert_eq!(decoded.class, ResidencyClass::TransientDecoded);
    assert_eq!(decoded.bytes_per_weight, 4.0);

    let stored = ResidencyProfile::stored(4.5);
    assert_eq!(stored.class, ResidencyClass::Stored);
    assert_eq!(stored.bytes_per_weight, 0.5625);

    let rebound = ResidencyProfile::rebound(16.0);
    assert_eq!(rebound.class, ResidencyClass::Rebound);
    assert_eq!(rebound.bytes_per_weight, 2.0);
}

#[test]
fn only_requantisation_changes_the_values_and_every_class_has_a_name() {
    use ResidencyClass as C;
    for (class, name, preserves) in [
        (C::Stored, "stored", true),
        (C::Rebound, "rebound", true),
        (C::TransientDecoded, "transient-decoded", true),
        (C::TransientRequantised, "transient-requantised", false),
    ] {
        assert_eq!(class.name(), name);
        assert_eq!(class.preserves_values(), preserves, "{name}");
    }
}

#[test]
fn an_acceleration_names_its_plan_its_cost_and_row_access() {
    let accel = Acceleration::cpu(
        PhysicalProjectionPlan::FusedBf16,
        ResidencyProfile::stored(16.0),
    );
    assert_eq!(accel.backend, AccelerationBackend::Cpu);
    assert_eq!(accel.plan, PhysicalProjectionPlan::FusedBf16);
    assert_eq!(accel.residency.bytes_per_weight, 2.0);
    assert_eq!(accel.requires, RequiredAccess::RowRandom);
}

#[test]
fn extents_are_ordered_by_depth_and_a_terminal_certificate_carries_no_radius() {
    assert_eq!(RepresentationExtent::BASE.depth, 0);
    assert_eq!(RepresentationExtent::at_depth(2).depth, 2);
    assert!(RepresentationExtent::BASE < RepresentationExtent::at_depth(1));
    let cert = ExtentCertificate::terminal(4.5);
    assert_eq!(cert.extent, RepresentationExtent::BASE);
    assert_eq!(cert.bits_per_weight, 4.5);
    assert_eq!(cert.radius, None);
    let bounded = ExtentCertificate {
        extent: RepresentationExtent::at_depth(1),
        bits_per_weight: 3.0,
        radius: Some(FidelityCertificate::relative_rms(0.05).unwrap()),
    };
    assert_eq!(bounded.radius.as_ref().unwrap().radius(), 0.05);
}

#[test]
fn a_codec_error_becomes_a_parse_error_with_its_text_intact() {
    let err = CodecError::Destination {
        tensor: TENSOR.into(),
        need: 1,
        have: 2,
    };
    let text = err.to_string();
    let vindex: crate::error::VindexError = err.into();
    assert!(vindex.to_string().contains(&text));
}
