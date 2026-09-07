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

/// The values stream is declared first and only once. It was asserted as
/// `streams[0] == VALUES` — the shared spec, by name — which was true of
/// every codec that had one stream to name and generalised wrongly: a
/// progressive codec's base plane is a values stream called `base_hi16`.
/// The ROLE is the invariant; the name is the codec's own.
#[test]
fn every_codec_declares_one_values_stream_and_declares_it_first() {
    for codec in builtin() {
        let label = codec.encoding_label();
        let streams = codec.streams();
        assert_eq!(streams[0].role, StreamRole::Values, "{label}");
        let values = streams
            .iter()
            .filter(|s| s.role == StreamRole::Values)
            .count();
        assert_eq!(values, 1, "{label}");
        // Refinements come after the base, at strictly increasing depths:
        // a stream that refines nothing before it is not a refinement.
        let mut deepest = 0;
        for spec in streams {
            if let StreamRole::Refinement { depth } = spec.role {
                assert!(depth > deepest, "{label}: {} refines at {depth}", spec.name);
                deepest = depth;
            }
        }
    }
}

/// Every codec declares its extents from the base up, prices each one, and
/// refuses the depth past its last.
///
/// This test asserted `extents.len() == 1` for every codec, which was a
/// statement about what had been implemented rather than about the
/// contract. What survives that becoming false is stated here: the base
/// first, contiguous depths, more of the representation as depth grows, a
/// certificate that certifies at least as much as the shallower one, and
/// `terminal_extent` naming the deepest — which for a terminal codec is
/// still the base.
#[test]
fn every_codec_declares_its_extents_from_the_base_up_and_refuses_past_them() {
    for codec in builtin() {
        let label = codec.encoding_label();
        let extents = codec.extents();
        assert!(!extents.is_empty(), "{label} declares no extent");
        assert_eq!(extents[0].extent, RepresentationExtent::BASE, "{label}");
        for (depth, certificate) in extents.iter().enumerate() {
            assert_eq!(certificate.extent.depth as usize, depth, "{label}");
            assert!(
                certificate.bits_per_weight > 0.0 && certificate.bits_per_weight <= 32.0,
                "{label} at depth {depth}"
            );
            assert_eq!(
                &codec.certificate_at(certificate.extent, TENSOR).unwrap(),
                certificate,
                "{label} at depth {depth}"
            );
            let Some(shallower) = depth.checked_sub(1).map(|d| extents[d].clone()) else {
                continue;
            };
            assert!(
                certificate.bits_per_weight > shallower.bits_per_weight,
                "{label}: depth {depth} costs no more than the one before it"
            );
            if let (Some(before), Some(after)) = (&shallower.radius, &certificate.radius) {
                assert!(
                    after.radius() <= before.radius(),
                    "{label}: depth {depth} certifies less than the one before it"
                );
            }
        }
        assert_eq!(
            codec.terminal_extent(),
            extents.last().unwrap().extent,
            "{label}: the terminal extent is the deepest declared"
        );
        let past = RepresentationExtent::at_depth(extents.len() as u32);
        let err = codec.certificate_at(past, TENSOR).unwrap_err();
        assert!(
            matches!(
                &err,
                CodecError::ExtentUnavailable { depth, available, .. }
                    if *depth == extents.len() as u32 && *available == extents.len() as u32
            ),
            "{label}: {err}"
        );
        assert!(matches!(
            codec.stored_bytes(&[ROWS, K], past, TENSOR).unwrap_err(),
            CodecError::ExtentUnavailable { .. }
        ));
    }
}

/// Which codecs are progressive, and what the terminal ones still promise.
///
/// What a codec may declare follows from the REFERENT, not from how many
/// extents it has.
///
/// A certificate bounds decoded values against the typed logical source
/// tensor presented at the representation boundary. So a codec that
/// carries its source losslessly can honestly say `0.0` — the identity,
/// or a widening that loses nothing, or entropy coding that is
/// byte-identical once inflated. A codec that FITS its source can say
/// nothing: the error belongs to the instance that was encoded, and a
/// number in the certificate would promote one measurement to a property
/// of the format. Only an attestation can supply that.
///
/// The old rule here was "a single-extent codec declares no radius",
/// which conflated the two questions: it is not the extent count that
/// decides, it is whether anything was thrown away.
#[test]
fn only_a_lossless_carrier_declares_a_radius_and_a_progressive_one_declares_one_per_extent() {
    // Lossless carriers of their own typed logical source.
    const LOSSLESS: [&str; 4] = ["BF16", "F16", "F32", "BF16_ZLIB"];
    // Codecs that fit their source, and so certify nothing without an
    // attestation for the instance.
    const FITTED: [&str; 6] = ["Q4_K", "Q6_K", "Q8_0", "NVFP4", "MXFP4", "VQ8_SHARED"];

    let progressive: Vec<&str> = builtin()
        .into_iter()
        .filter(|c| c.extents().len() > 1)
        .map(|c| c.encoding_label())
        .collect();
    assert_eq!(progressive, ["F32_PLANES"]);

    for codec in builtin() {
        let label = codec.encoding_label();
        let extents = codec.extents();
        if extents.len() > 1 {
            assert!(
                extents.iter().all(|c| c.radius.is_some()),
                "{label}: an extent without a radius cannot be chosen by fidelity"
            );
            assert_eq!(
                extents.last().unwrap().radius.as_ref().unwrap().radius(),
                0.0,
                "{label}: the terminal extent reconstructs its source exactly"
            );
            continue;
        }
        let radius = extents[0].radius.as_ref().map(|r| r.radius());
        if LOSSLESS.contains(&label) {
            assert_eq!(
                radius,
                Some(0.0),
                "{label}: a lossless carrier states 0.0 rather than declining to state"
            );
        } else {
            assert!(
                FITTED.contains(&label),
                "{label}: every shipped codec is a lossless carrier or a fitted one;                  a new one must say which"
            );
            assert_eq!(
                radius, None,
                "{label}: a fitted codec's error is measured per instance, not declared"
            );
        }
    }
}

#[test]
fn every_codec_admits_its_own_alignment_and_refuses_access_it_lacks() {
    let mut sequential = 0;
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
        // Every codec serves a front-to-back read. Finer access is
        // provided exactly when declared and refused by name otherwise —
        // rung 1 asserted row access for every codec, which was true of
        // every codec it had and not a property of the contract.
        caps.require(RequiredAccess::Sequential, label).unwrap();
        for required in [RequiredAccess::RowRandom, RequiredAccess::ElementRandom] {
            let result = caps.require(required, label);
            assert_eq!(
                result.is_ok(),
                caps.access.provides(required),
                "{label}: {}",
                required.name()
            );
            if let Err(err) = result {
                assert!(
                    matches!(
                        &err,
                        CodecError::AccessRefused { provided, required: r, .. }
                            if *provided == caps.access.name() && r == required.name()
                    ),
                    "{label}: {err}"
                );
            }
        }
        // The requirement that IS genuine: a codec offering a direct
        // realization has promised the kernel behind it arbitrary rows.
        if !codec.accelerations().is_empty() {
            caps.require(RequiredAccess::RowRandom, label).unwrap();
        }
        if caps.access == AccessGranularity::Sequential {
            sequential += 1;
        }
    }
    assert_eq!(
        sequential, 1,
        "the sequential witness is registered, so the refusal arm ran"
    );
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
        // The three K-quants share one resident format — the codec rides
        // in the bound operand, not the format — so this cannot name one.
        // `every_acceleration_...` checks the family membership directly.
        WeightFormat::KQuant => None,
        // A SOURCE format, not a REPRESENT target: fine-grained FP8
        // arrives in the checkpoint and this build never compiles a
        // tensor into it, so no `represent` codec answers to it. `None`
        // here is that fact, not an omission — if a codec ever emits
        // FP8, this arm names it and the assertion above starts applying.
        WeightFormat::Fp8Block => None,
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
            // A codec's direct plan names it. The K-quants are the one
            // family that shares a resident format (`WeightFormat::KQuant`,
            // the codec carried by the operand): there the plan names the
            // FAMILY, and the label is Q4_K/Q6_K/Q8_0.
            if matches!(label, "Q4_K" | "Q6_K" | "Q8_0") {
                assert_eq!(
                    accel.plan.format(),
                    WeightFormat::KQuant,
                    "{label}: {:?}",
                    accel.plan
                );
            } else {
                assert_eq!(
                    label_of(accel.plan.format()),
                    Some(label),
                    "{label}: {:?}",
                    accel.plan
                );
            }
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
    // K-quants gained a direct CPU realization (FusedKQuant); the
    // entropy-coded codec registers none, and neither do the progressive
    // or the codebook-dependent ones — deliberately, so an extent and a
    // dependency can each be shown to work without any kernel knowing
    // they exist.
    assert_eq!(
        without,
        ["F16", "MXFP4", "BF16_ZLIB", "F32_PLANES", "VQ8_SHARED"]
    );
    let with: Vec<&str> = builtin()
        .into_iter()
        .filter(|c| !c.accelerations().is_empty())
        .map(|c| c.encoding_label())
        .collect();
    assert_eq!(with, ["BF16", "F32", "Q4_K", "Q6_K", "Q8_0", "NVFP4"]);
}

#[test]
fn stored_bytes_are_the_certificate_s_bits_at_scale() {
    // The certificate is asymptotic; on a large matrix the exact byte
    // count must agree with it to the per-tensor overhead.
    let shape = [4096usize, 4096];
    let (mut priced, mut instance_sized) = (0, 0);
    for codec in builtin() {
        let label = codec.encoding_label();
        // Every declared extent, not only the base: a progressive codec
        // prices each depth, and a certificate nothing prices is a claim.
        match codec.stored_bytes(&shape, RepresentationExtent::BASE, TENSOR) {
            Ok(_) => {
                for certificate in codec.extents() {
                    let bytes = codec
                        .stored_bytes(&shape, certificate.extent, TENSOR)
                        .unwrap_or_else(|e| panic!("{label} at {:?}: {e}", certificate.extent));
                    let bpw = bytes as f64 * extent::BITS_PER_BYTE / (shape[0] * shape[1]) as f64;
                    let declared = certificate.bits_per_weight;
                    assert!(
                        (bpw - declared).abs() < 1e-3,
                        "{label} at depth {}: {bpw} vs {declared}",
                        certificate.extent.depth
                    );
                }
                priced += 1;
            }
            // A variable-rate code cannot price a shape: its certificate
            // is a bound, and the refusal names the container's recorded
            // length as the authority. Rung 1 asserted pricing for every
            // codec; that was true of fixed-rate codes, not of the trait.
            Err(CodecError::InstanceSized { label: refused, .. }) => {
                assert_eq!(refused, label);
                instance_sized += 1;
            }
            Err(other) => panic!("{label}: {other}"),
        }
    }
    assert_eq!(
        (priced, instance_sized),
        (builtin().len() - 1, 1),
        "both arms ran: every codec but the instance-sized one prices a shape"
    );
}
