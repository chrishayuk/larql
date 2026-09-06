//! The progressive codec on its own terms: three widths of the same
//! bytes, a terminal extent that is the source bit for bit, a certificate
//! that is checked rather than believed, and edges named rather than
//! averaged away.

use super::super::codecs::f32_planes::{
    Domain, F32PlanesCodec, BASE_HI16, F32_PLANES, REFINE_8A, REFINE_8B, TERMINAL_DEPTH,
};
use super::super::conformance;
use super::super::fidelity::FidelityCertificate;
use super::*;

/// Bit patterns a ramp never produces and a truncating codec must survive:
/// both zeroes, the smallest and largest subnormals, the smallest and
/// largest normals, both infinities, a NaN whose payload lives ONLY in the
/// omitted planes, and one whose payload survives in the base plane.
const ADVERSARIAL_BITS: [u32; 14] = [
    0x0000_0000, // +0
    0x8000_0000, // -0
    0x0000_0001, // the smallest subnormal — every significant bit omitted at depth 0
    0x007f_ffff, // the largest subnormal
    0x0080_0000, // the smallest normal
    0x7f7f_ffff, // the largest normal
    0x3f80_0000, // 1.0
    0xbf80_0000, // -1.0
    0x7f80_0000, // +inf
    0xff80_0000, // -inf
    0x7f80_0001, // a NaN whose payload is only in the last plane
    0x7fc0_0000, // a quiet NaN, payload in the base plane
    0xffc0_dead, // a negative NaN with payload in every plane
    0x1234_5678, // an ordinary normal with four distinct bytes
];

fn planes_of(values: &[f32]) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    F32PlanesCodec::encode_planes(values)
}

/// Bind whichever planes the caller wants bound — the loader's job, done
/// by hand here so a test can bind FEWER planes than the container holds.
fn operands<'a>(
    base: &'a [u8],
    refine_a: Option<&'a [u8]>,
    refine_b: Option<&'a [u8]>,
) -> CodecOperands<'a> {
    let mut streams = NamedStreams::new().with(BASE_HI16, base);
    if let Some(bytes) = refine_a {
        streams = streams.with(REFINE_8A, bytes);
    }
    if let Some(bytes) = refine_b {
        streams = streams.with(REFINE_8B, bytes);
    }
    CodecOperands::from_streams(streams)
}

fn all_planes<'a>(planes: &'a (Vec<u8>, Vec<u8>, Vec<u8>)) -> CodecOperands<'a> {
    operands(&planes.0, Some(&planes.1), Some(&planes.2))
}

#[test]
fn the_planes_are_the_bit_pattern_cut_in_three_little_endian_halves_first() {
    // Chosen so all four bytes differ: nothing about the order can be read
    // two ways.
    let value = f32::from_bits(0x1234_5678);
    let (base, refine_a, refine_b) = planes_of(&[value]);
    assert_eq!(base, [0x34, 0x12], "the high half, little-endian");
    assert_eq!(refine_a, [0x56]);
    assert_eq!(refine_b, [0x78]);
    // And the planes partition the pattern: nothing is stored twice, and
    // together they are exactly four bytes.
    assert_eq!(base.len() + refine_a.len() + refine_b.len(), 4);
}

#[test]
fn every_extent_is_stored_at_its_declared_width() {
    let shape = [ROWS, K];
    let elements = ROWS * K;
    for certificate in F32_PLANES.extents() {
        let bytes = F32_PLANES
            .stored_bytes(&shape, certificate.extent, TENSOR)
            .unwrap();
        assert_eq!(
            bytes as f64 * extent::BITS_PER_BYTE / elements as f64,
            certificate.bits_per_weight,
            "depth {}",
            certificate.extent.depth
        );
    }
    assert_eq!(
        F32_PLANES.terminal_extent(),
        RepresentationExtent::at_depth(TERMINAL_DEPTH)
    );
}

#[test]
fn the_terminal_extent_reproduces_every_bit_pattern_exactly() {
    let values: Vec<f32> = ADVERSARIAL_BITS
        .iter()
        .map(|b| f32::from_bits(*b))
        .collect();
    let shape = [values.len()];
    let planes = planes_of(&values);
    let decoded = F32_PLANES
        .decode_all(
            &all_planes(&planes),
            &shape,
            F32_PLANES.terminal_extent(),
            TENSOR,
        )
        .unwrap();
    // Compared as BIT PATTERNS: a NaN that changed payload is not equal to
    // itself as a float, and would pass a value comparison by vanishing.
    for (source, out) in values.iter().zip(&decoded) {
        assert_eq!(
            source.to_bits(),
            out.to_bits(),
            "0x{:08x} came back as 0x{:08x}",
            source.to_bits(),
            out.to_bits()
        );
    }
}

#[test]
fn a_shallower_extent_truncates_toward_zero_and_stays_inside_its_radius() {
    let values = ramp(ROWS * K);
    let shape = [ROWS, K];
    let planes = planes_of(&values);
    let bound = all_planes(&planes);
    let mut worst_so_far = f64::INFINITY;
    for certificate in F32_PLANES.extents() {
        let decoded = F32_PLANES
            .decode_all(&bound, &shape, certificate.extent, TENSOR)
            .unwrap();
        let mut worst = 0.0f64;
        for (source, out) in values.iter().zip(&decoded) {
            assert!(
                out.abs() <= source.abs(),
                "depth {}: truncation cannot grow a magnitude ({source} -> {out})",
                certificate.extent.depth
            );
            assert_eq!(
                out.is_sign_negative(),
                source.is_sign_negative(),
                "depth {}: the sign lives in the base plane",
                certificate.extent.depth
            );
            let relative = ((f64::from(*out) - f64::from(*source)) / f64::from(*source)).abs();
            worst = worst.max(relative);
        }
        assert!(
            worst < worst_so_far || worst == 0.0,
            "depth {} is no better than the extent before it",
            certificate.extent.depth
        );
        worst_so_far = worst;
        let declared = certificate.radius.as_ref().unwrap().radius();
        if certificate.extent.depth == TERMINAL_DEPTH {
            assert_eq!(
                (worst, declared),
                (0.0, 0.0),
                "the terminal extent is exact"
            );
        } else {
            assert!(
                worst > 0.0,
                "a shallow extent that loses nothing is not one"
            );
            assert!(declared > 0.0);
        }
    }
}

#[test]
fn the_certificate_is_checked_against_the_source_not_believed() {
    let values = ramp(ROWS * K);
    let shape = [ROWS, K];
    let planes = planes_of(&values);
    let report = conformance::certify(&F32_PLANES, &all_planes(&planes), &shape, &values, TENSOR)
        .expect("the codec meets what it declares");
    assert_eq!(report.readings.len(), (TERMINAL_DEPTH + 1) as usize);
    assert_eq!(
        report.uncertified_elements, 0,
        "the ramp is finite and normal throughout"
    );
    let base = report.at_depth(0).unwrap();
    assert!(
        base.measured_relative_rms > 0.0
            && base.measured_relative_rms < base.declared_relative_rms.unwrap(),
        "{base:?}"
    );
    assert!(!base.bit_exact);
    let terminal = report.at_depth(TERMINAL_DEPTH).unwrap();
    assert!(terminal.bit_exact && terminal.measured_relative_rms == 0.0);
}

/// The same bytes, the same decode, a better radius declared for depth 0
/// than truncation can deliver: the harness must catch the DECLARATION,
/// since nothing else about the codec is wrong.
struct OverclaimingPlanes;

impl RepresentationCodec for OverclaimingPlanes {
    fn encoding_label(&self) -> &'static str {
        "F32_PLANES_OVERCLAIMED"
    }
    fn identity(&self) -> super::super::super::nvfp4_pack::CodecIdentity {
        F32_PLANES.identity()
    }
    fn streams(&self) -> &'static [StreamSpec] {
        F32_PLANES.streams()
    }
    fn capabilities(&self) -> CodecCapabilities {
        F32_PLANES.capabilities()
    }
    fn extents(&self) -> Vec<ExtentCertificate> {
        let mut extents = F32_PLANES.extents();
        // Depth 0 keeps seven mantissa bits; this claims the fifteen of
        // depth 1 — the plausible lie, one extent out.
        extents[0].radius = extents[1].radius.clone();
        extents
    }
    fn stored_bytes(
        &self,
        shape: &[usize],
        extent: RepresentationExtent,
        tensor: &str,
    ) -> Result<u64, CodecError> {
        F32_PLANES.stored_bytes(shape, extent, tensor)
    }
    fn validate(
        &self,
        operands: &CodecOperands<'_>,
        shape: &[usize],
        extent: RepresentationExtent,
        tensor: &str,
    ) -> Result<(), CodecError> {
        F32_PLANES.validate(operands, shape, extent, tensor)
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
        F32_PLANES.decode_rows(operands, shape, rows, extent, dst, tensor)
    }
    fn decode_residency(&self) -> ResidencyProfile {
        F32_PLANES.decode_residency()
    }
}

#[test]
fn a_radius_the_decode_cannot_meet_is_refused_by_depth_and_number() {
    let values = ramp(ROWS * K);
    let shape = [ROWS, K];
    let planes = planes_of(&values);
    let err = conformance::certify(
        &OverclaimingPlanes,
        &all_planes(&planes),
        &shape,
        &values,
        TENSOR,
    )
    .unwrap_err();
    let CodecError::CertificateViolated {
        depth,
        declared,
        measured,
        ..
    } = &err
    else {
        panic!("the overclaimed radius must be refused as one: {err}");
    };
    assert_eq!(*depth, 0);
    assert!(measured > declared, "{err}");
}

#[test]
fn a_shallow_extent_names_the_edges_it_cannot_certify() {
    let values: Vec<f32> = ADVERSARIAL_BITS
        .iter()
        .map(|b| f32::from_bits(*b))
        .collect();
    let shape = [values.len()];
    let planes = planes_of(&values);
    let decoded = F32_PLANES
        .decode_all(
            &all_planes(&planes),
            &shape,
            RepresentationExtent::BASE,
            TENSOR,
        )
        .unwrap();
    let at = |bits: u32| {
        let index = ADVERSARIAL_BITS.iter().position(|b| *b == bits).unwrap();
        decoded[index].to_bits()
    };
    // Exact at every depth: sign and exponent live in the base plane.
    assert_eq!(at(0x0000_0000), 0x0000_0000, "+0 keeps its sign");
    assert_eq!(at(0x8000_0000), 0x8000_0000, "-0 keeps its sign");
    assert_eq!(at(0x7f80_0000), 0x7f80_0000, "+inf");
    assert_eq!(at(0xff80_0000), 0xff80_0000, "-inf");
    // Defined, and outside any relative bound: a subnormal whose bits are
    // all in omitted planes flushes to zero.
    assert_eq!(
        at(0x0000_0001),
        0x0000_0000,
        "the smallest subnormal flushes"
    );
    // The one case where the CLASS changes, named rather than discovered.
    assert_eq!(
        at(0x7f80_0001),
        0x7f80_0000,
        "a NaN whose payload is only in an omitted plane decodes as an infinity"
    );
    // A NaN whose payload reaches the base plane stays a NaN.
    assert!(f32::from_bits(at(0x7fc0_0000)).is_nan());
    // And all of it is visible from the base plane alone, without opening
    // a refinement: the two infinities and three NaNs, and the two zeroes
    // and two subnormals.
    assert_eq!(
        F32PlanesCodec::domain_of_base_plane(&planes.0),
        Domain {
            non_finite: 5,
            subnormal_or_zero: 4,
        }
    );
    assert!(!F32PlanesCodec::domain_of_base_plane(&planes.0).is_certified());
    let ramp_planes = planes_of(&ramp(K));
    assert!(F32PlanesCodec::domain_of_base_plane(&ramp_planes.0).is_certified());
}

#[test]
fn a_depth_reads_its_own_planes_and_refuses_a_missing_one() {
    let values = ramp(K);
    let shape = [K];
    let planes = planes_of(&values);
    // Depth 0 needs only the base plane, and gets the same answer whether
    // or not the deeper ones are bound.
    let base_only = operands(&planes.0, None, None);
    let shallow = F32_PLANES
        .decode_all(&base_only, &shape, RepresentationExtent::BASE, TENSOR)
        .unwrap();
    let with_everything = F32_PLANES
        .decode_all(
            &all_planes(&planes),
            &shape,
            RepresentationExtent::BASE,
            TENSOR,
        )
        .unwrap();
    assert_eq!(shallow, with_everything, "a bound plane it must not read");
    // A depth whose plane is absent refuses by name — never zero-filled,
    // which would read as a value the source never held.
    let err = F32_PLANES
        .decode_all(
            &base_only,
            &shape,
            RepresentationExtent::at_depth(1),
            TENSOR,
        )
        .unwrap_err();
    assert!(
        matches!(&err, CodecError::MissingStream { stream, bound, .. }
            if stream == REFINE_8A.name && bound == &vec![BASE_HI16.name.to_string()]),
        "{err}"
    );
}

#[test]
fn each_plane_is_exactly_its_own_length_and_a_compensating_pair_is_refused() {
    let values = ramp(K);
    let shape = [K];
    let (base, refine_a, refine_b) = planes_of(&values);
    // The mutation a total cannot catch: one byte moved from the base
    // plane to a refinement — one stream short and another long by the
    // SAME amount, so every byte is still there and none of them is where
    // its extent says it is.
    let short_base = &base[..base.len() - 1];
    let mut long_refine = refine_a.clone();
    long_refine.push(0);
    // Stated rather than implied: the mutated streams total exactly what
    // the codec declares, so a rule that checked the total would admit
    // them and a rule that checks each stream at its extent must not.
    let declared_total = F32_PLANES
        .stored_bytes(&shape, F32_PLANES.terminal_extent(), TENSOR)
        .unwrap();
    let mutated_total = (short_base.len() + long_refine.len() + refine_b.len()) as u64;
    let honest_total = (base.len() + refine_a.len() + refine_b.len()) as u64;
    assert_eq!(
        (mutated_total, honest_total),
        (declared_total, declared_total),
        "the mutation must be byte-neutral, or it is not the mutation under test"
    );
    let compensated = operands(short_base, Some(&long_refine), Some(&refine_b));
    let err = F32_PLANES
        .validate(&compensated, &shape, F32_PLANES.terminal_extent(), TENSOR)
        .unwrap_err();
    assert!(
        matches!(&err, CodecError::StreamLength { stream, need, have, .. }
            if stream == BASE_HI16.name && *need == base.len() && *have == base.len() - 1),
        "{err}"
    );
    // Overlong is refused as firmly as short: unclaimed bytes in a plane
    // mean the container and the declared geometry disagree.
    let mut long_base = base.clone();
    long_base.push(0);
    let overlong = operands(&long_base, Some(&refine_a), Some(&refine_b));
    let err = F32_PLANES
        .validate(&overlong, &shape, F32_PLANES.terminal_extent(), TENSOR)
        .unwrap_err();
    assert!(
        matches!(&err, CodecError::StreamLength { stream, need, have, .. }
            if stream == BASE_HI16.name && *need == base.len() && *have == base.len() + 1),
        "{err}"
    );
    // A refinement bound at a depth that does not read it is not judged:
    // the length rule is per extent, not per tensor.
    let wrong_refinement = operands(&base, Some(&long_refine), Some(&refine_b));
    F32_PLANES
        .validate(
            &wrong_refinement,
            &shape,
            RepresentationExtent::BASE,
            TENSOR,
        )
        .expect("depth 0 judges the base plane, which is correct");
    let err = F32_PLANES
        .validate(
            &wrong_refinement,
            &shape,
            RepresentationExtent::at_depth(1),
            TENSOR,
        )
        .unwrap_err();
    assert!(
        matches!(&err, CodecError::StreamLength { stream, .. } if stream == REFINE_8A.name),
        "{err}"
    );
}

#[test]
fn a_row_range_at_a_shallow_extent_is_that_slice_of_the_whole() {
    let values = ramp(ROWS * K);
    let shape = [ROWS, K];
    let planes = planes_of(&values);
    let bound = all_planes(&planes);
    for extent in F32_PLANES.extents().into_iter().map(|c| c.extent) {
        let whole = F32_PLANES
            .decode_all(&bound, &shape, extent, TENSOR)
            .unwrap();
        let mut middle = vec![0.0f32; K];
        F32_PLANES
            .decode_rows(&bound, &shape, 1..2, extent, &mut middle, TENSOR)
            .unwrap();
        assert_eq!(middle, whole[K..2 * K], "depth {}", extent.depth);
    }
}

// ── The harness's own refusals ───────────────────────────────────────
//
// A checker whose refusal paths are never taken is a checker nobody has
// checked. Each stub below is correct in every way but one, so exactly
// one refusal can fire and the test says which.

/// A codec whose deeper extent reconstructs WORSE than its shallower one,
/// inside both declared radii: no certificate is violated, the ORDERING
/// is. Two extents, so its terminal one is depth 1.
struct RegressingPlanes;

/// A codec that improves with depth and never reaches the source: every
/// radius is met, the ordering holds, and the deepest extent is still not
/// the bytes it came from.
struct InexactTerminalPlanes;

/// The radius the stubs declare where the test is not about the radius —
/// loose enough that only the property under test can fire.
const LOOSE_RADIUS: f64 = 0.05;
/// What the stubs multiply a shallow decode by: enough error to be
/// measurable, well inside `LOOSE_RADIUS`.
const SHALLOW_SPOIL: f32 = 1.01;
/// What `InexactTerminalPlanes` leaves at its deepest extent.
const RESIDUAL_SPOIL: f32 = 1.000_000_1;

fn two_extents(deepest_radius: f64) -> Vec<ExtentCertificate> {
    vec![
        ExtentCertificate::certified(
            0,
            16.0,
            FidelityCertificate::relative_rms(LOOSE_RADIUS).unwrap(),
        ),
        ExtentCertificate::certified(
            1,
            24.0,
            FidelityCertificate::relative_rms(deepest_radius).unwrap(),
        ),
    ]
}

/// Both stubs are `F32_PLANES` with two declared extents and a decode
/// that spoils the terminal image in their own way, so the BYTES are
/// never what is under test.
macro_rules! spoiling_codec {
    ($name:ident, $label:literal, $extents:expr, $spoil:expr) => {
        impl RepresentationCodec for $name {
            fn encoding_label(&self) -> &'static str {
                $label
            }
            fn identity(&self) -> super::super::super::nvfp4_pack::CodecIdentity {
                F32_PLANES.identity()
            }
            fn streams(&self) -> &'static [StreamSpec] {
                F32_PLANES.streams()
            }
            fn capabilities(&self) -> CodecCapabilities {
                F32_PLANES.capabilities()
            }
            fn extents(&self) -> Vec<ExtentCertificate> {
                $extents
            }
            fn stored_bytes(
                &self,
                shape: &[usize],
                extent: RepresentationExtent,
                tensor: &str,
            ) -> Result<u64, CodecError> {
                self.certificate_at(extent, tensor)?;
                F32_PLANES.stored_bytes(shape, RepresentationExtent::BASE, tensor)
            }
            fn validate(
                &self,
                operands: &CodecOperands<'_>,
                shape: &[usize],
                extent: RepresentationExtent,
                tensor: &str,
            ) -> Result<(), CodecError> {
                self.certificate_at(extent, tensor)?;
                F32_PLANES.validate(operands, shape, RepresentationExtent::BASE, tensor)
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
                self.certificate_at(extent, tensor)?;
                F32_PLANES.decode_rows(
                    operands,
                    shape,
                    rows,
                    F32_PLANES.terminal_extent(),
                    dst,
                    tensor,
                )?;
                let spoil: fn(&mut [f32], u32) = $spoil;
                spoil(dst, extent.depth);
                Ok(())
            }
            fn decode_residency(&self) -> ResidencyProfile {
                ResidencyProfile::DECODED_F32
            }
        }
    };
}

spoiling_codec!(
    RegressingPlanes,
    "F32_PLANES_REGRESSING",
    two_extents(LOOSE_RADIUS),
    |dst, depth| {
        // Depth 0 is the source; the "refinement" moves away from it.
        if depth >= 1 {
            for value in dst.iter_mut() {
                *value *= SHALLOW_SPOIL;
            }
        }
    }
);

spoiling_codec!(
    InexactTerminalPlanes,
    "F32_PLANES_INEXACT",
    two_extents(1e-6),
    |dst, depth| {
        let spoil = if depth >= 1 {
            RESIDUAL_SPOIL
        } else {
            SHALLOW_SPOIL
        };
        for value in dst.iter_mut() {
            *value *= spoil;
        }
    }
);

#[test]
fn an_extent_that_reconstructs_worse_than_the_one_before_it_is_refused() {
    let values = ramp(K);
    let shape = [K];
    let planes = planes_of(&values);
    let err = conformance::certify(
        &RegressingPlanes,
        &all_planes(&planes),
        &shape,
        &values,
        TENSOR,
    )
    .unwrap_err();
    let CodecError::CertificateNotMonotone {
        depth,
        shallower,
        measured,
        before,
        ..
    } = &err
    else {
        panic!("a deeper extent that lost ground must be refused as one: {err}");
    };
    assert_eq!((*depth, *shallower), (1, 0));
    assert!(measured > before, "{err}");
}

#[test]
fn a_deepest_extent_that_is_not_the_source_is_refused() {
    let values = ramp(K);
    let shape = [K];
    let planes = planes_of(&values);
    let err = conformance::certify(
        &InexactTerminalPlanes,
        &all_planes(&planes),
        &shape,
        &values,
        TENSOR,
    )
    .unwrap_err();
    let CodecError::TerminalNotExact {
        depth,
        differing,
        elements,
        ..
    } = &err
    else {
        panic!("an inexact terminal extent must be refused as one: {err}");
    };
    assert_eq!((*depth, *elements), (1, K));
    assert!(*differing > 0 && differing <= elements, "{err}");
}

#[test]
fn a_source_the_decode_cannot_match_in_length_is_refused() {
    let values = ramp(K);
    let shape = [K];
    let planes = planes_of(&values);
    // A source one element short of what the shape declares: the harness
    // is being asked to compare two different tensors.
    let err = conformance::certify(
        &F32_PLANES,
        &all_planes(&planes),
        &shape,
        &values[..K - 1],
        TENSOR,
    )
    .unwrap_err();
    assert!(
        matches!(&err, CodecError::Destination { need, have, .. } if *need == K - 1 && *have == K),
        "{err}"
    );
}

#[test]
fn a_report_answers_only_for_the_depths_the_codec_declared() {
    let values = ramp(K);
    let shape = [K];
    let planes = planes_of(&values);
    let report =
        conformance::certify(&F32_PLANES, &all_planes(&planes), &shape, &values, TENSOR).unwrap();
    assert!(report.at_depth(TERMINAL_DEPTH).is_some());
    assert!(
        report.at_depth(TERMINAL_DEPTH + 1).is_none(),
        "an extent nobody declared has no reading"
    );
    assert_eq!(report.elements, K);
}
