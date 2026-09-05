//! What every registered codec must declare — checked over all of them.

use super::*;
use crate::format::vindex3::opplan::exec::backend::WeightFormat;

#[test]
fn every_codec_names_itself_and_its_abi_is_admitted_back_to_it() {
    let registry = CodecRegistry::builtin();
    for codec in builtin() {
        let label = codec.encoding_label();
        assert!(!label.is_empty());
        let id = codec.identity();
        assert!(!id.family.is_empty(), "{label} names no family");
        let admitted = registry
            .admit(&id)
            .unwrap_or_else(|e| panic!("{label}: {e}"));
        assert_eq!(admitted.encoding_label(), label);
        assert_eq!(registry.by_label(label).unwrap().identity(), id);
    }
}

#[test]
fn every_codec_declares_its_values_stream_first_and_only_once() {
    for codec in builtin() {
        let streams = codec.streams();
        assert_eq!(streams[0], VALUES, "{}", codec.encoding_label());
        let values = streams
            .iter()
            .filter(|s| s.role == StreamRole::Values)
            .count();
        assert_eq!(values, 1, "{}", codec.encoding_label());
    }
}

#[test]
fn every_codec_is_terminal_today_and_says_so_at_any_other_depth() {
    for codec in builtin() {
        let label = codec.encoding_label();
        let extents = codec.extents();
        assert_eq!(extents.len(), 1, "{label}");
        assert_eq!(extents[0].extent, RepresentationExtent::TERMINAL, "{label}");
        assert!(extents[0].bits_per_weight > 0.0 && extents[0].bits_per_weight <= 32.0);
        assert!(
            extents[0].radius.is_none(),
            "{label}: no radius is declared, only measured"
        );
        assert_eq!(
            codec
                .certificate_at(RepresentationExtent::TERMINAL, TENSOR)
                .unwrap(),
            extents[0]
        );
        let deeper = RepresentationExtent::at_depth(1);
        let err = codec.certificate_at(deeper, TENSOR).unwrap_err();
        assert!(
            matches!(
                &err,
                CodecError::ExtentUnavailable {
                    depth: 1,
                    available: 1,
                    ..
                }
            ),
            "{label}: {err}"
        );
        assert!(matches!(
            codec.stored_bytes(&[ROWS, K], deeper, TENSOR).unwrap_err(),
            CodecError::ExtentUnavailable { .. }
        ));
    }
}

#[test]
fn every_codec_admits_its_own_alignment_and_refuses_access_it_lacks() {
    for codec in builtin() {
        let label = codec.encoding_label();
        let caps = codec.capabilities();
        assert!(caps.admits_k(caps.row_align_elems), "{label}");
        assert!(caps.admits_k(K), "{label}: the fixture width is admissible");
        if caps.row_align_elems > 1 {
            assert!(!caps.admits_k(caps.row_align_elems + 1), "{label}");
        }
        assert!(
            caps.group_elems >= 1 && caps.row_align_elems >= caps.group_elems,
            "{label}"
        );
        caps.require(RequiredAccess::Sequential, label).unwrap();
        caps.require(RequiredAccess::RowRandom, label).unwrap();
        let element = caps.require(RequiredAccess::ElementRandom, label);
        assert_eq!(
            element.is_ok(),
            caps.access == AccessGranularity::ElementRandom,
            "{label}"
        );
    }
}

#[test]
fn the_mandatory_realization_widens_to_f32_for_every_codec() {
    for codec in builtin() {
        assert_eq!(
            codec.decode_residency(),
            ResidencyProfile::DECODED_F32,
            "{}",
            codec.encoding_label()
        );
    }
}

/// The label a CPU plan's resident format corresponds to.
fn label_of(format: WeightFormat) -> Option<&'static str> {
    match format {
        WeightFormat::F32 => Some("F32"),
        WeightFormat::Bf16 => Some("BF16"),
        WeightFormat::F16 => Some("F16"),
        WeightFormat::Nvfp4 => Some("NVFP4"),
        WeightFormat::Mxfp4 => Some("MXFP4"),
        // Runtime re-quantisations of a float source: no stored codec.
        WeightFormat::Q8 | WeightFormat::Q4 => None,
    }
}

#[test]
fn every_acceleration_runs_over_the_stored_bytes_and_names_a_plan_for_them() {
    for codec in builtin() {
        let label = codec.encoding_label();
        let bpw = codec.extents()[0].bits_per_weight;
        for accel in codec.accelerations() {
            assert_eq!(accel.backend, AccelerationBackend::Cpu, "{label}");
            assert!(accel.residency.class.preserves_values(), "{label}");
            assert!(
                matches!(
                    accel.residency.class,
                    ResidencyClass::Stored | ResidencyClass::Rebound
                ),
                "{label}: a direct realization touches stored bytes, got {:?}",
                accel.residency.class
            );
            let touched = accel.residency.bytes_per_weight * extent::BITS_PER_BYTE;
            assert!((touched - bpw).abs() < 1e-9, "{label}: {touched} vs {bpw}");
            assert_eq!(accel.requires, RequiredAccess::RowRandom, "{label}");
            assert_eq!(
                label_of(accel.plan.format()),
                Some(label),
                "{label}: {:?}",
                accel.plan
            );
        }
    }
}

#[test]
fn codecs_with_no_direct_cpu_realization_say_so_rather_than_claim_one() {
    let without: Vec<&str> = builtin()
        .into_iter()
        .filter(|c| c.accelerations().is_empty())
        .map(|c| c.encoding_label())
        .collect();
    assert_eq!(without, ["F16", "Q4_K", "Q6_K", "Q8_0", "MXFP4"]);
    let with: Vec<&str> = builtin()
        .into_iter()
        .filter(|c| !c.accelerations().is_empty())
        .map(|c| c.encoding_label())
        .collect();
    assert_eq!(with, ["BF16", "F32", "NVFP4"]);
}

#[test]
fn stored_bytes_are_the_certificate_s_bits_at_scale() {
    // The certificate is asymptotic; on a large matrix the exact byte
    // count must agree with it to the per-tensor overhead.
    let shape = [4096usize, 4096];
    for codec in builtin() {
        let label = codec.encoding_label();
        let bytes = codec
            .stored_bytes(&shape, RepresentationExtent::TERMINAL, TENSOR)
            .unwrap() as f64;
        let bpw = bytes * extent::BITS_PER_BYTE / (shape[0] * shape[1]) as f64;
        let declared = codec.extents()[0].bits_per_weight;
        assert!(
            (bpw - declared).abs() < 1e-3,
            "{label}: {bpw} vs {declared}"
        );
    }
}
