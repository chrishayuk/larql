//! The entropy-coded bf16 codec against its preregistration
//! (`docs/represent/forecasts/rung2-entropy-coded-bf16.json`): lossless
//! at the bit level against a FOREIGN encoder's stream, sequential by
//! construction, instance-sized, and refusing by name at every edge.

use super::fixtures::bf16_zlib_foreign as foreign;
use super::*;
use crate::format::vindex3::represent::codec::codecs::bf16_zlib::{
    BF16_ZLIB_REVISION, DTYPE_BF16_ZLIB,
};

/// Row width of the awkward image, deliberately not a multiple of anything.
const AWKWARD_K: usize = foreign::AWKWARD_SHAPE[1];
const AWKWARD_ROWS: usize = foreign::AWKWARD_SHAPE[0];
const NOISE_K: usize = foreign::NOISE_SHAPE[1];
const NOISE_ROWS: usize = foreign::NOISE_SHAPE[0];
/// Where the truncation tests cut a stream: past the first row's worth of
/// input on the noise image, well short of the whole.
const TRUNCATE_NUMERATOR: usize = 3;
const TRUNCATE_DENOMINATOR: usize = 5;
/// Bytes appended after a complete stream to witness the trailing refusal.
const TRAILING_GARBAGE: [u8; 3] = [0xDE, 0xAD, 0x00];
/// Names the identity must never carry: implementations, not formats.
const FORBIDDEN_IN_IDENTITY: [&str; 4] = ["flate2", "miniz", "zlib-ng", "libz"];

fn bits_of(units: &[u16]) -> Vec<u32> {
    units.iter().map(|u| u32::from(*u) << u16::BITS).collect()
}

fn decoded_bits(stream: &[u8], shape: &[usize]) -> Vec<u32> {
    BF16_ZLIB
        .decode_packed(stream, shape, RepresentationExtent::TERMINAL, TENSOR)
        .unwrap_or_else(|e| panic!("{e}"))
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn operands(stream: &[u8]) -> CodecOperands<'_> {
    CodecOperands::from_streams(NamedStreams::single(VALUES, stream))
}

fn decode_rows(
    stream: &[u8],
    shape: &[usize],
    rows: std::ops::Range<usize>,
) -> Result<Vec<f32>, CodecError> {
    let k = shape[1];
    let mut dst = vec![f32::NAN; rows.len() * k];
    BF16_ZLIB.decode_rows(
        &operands(stream),
        shape,
        rows,
        RepresentationExtent::TERMINAL,
        &mut dst,
        TENSOR,
    )?;
    Ok(dst)
}

fn detail_of(err: CodecError) -> String {
    match err {
        CodecError::Decode { label, detail, .. } => {
            assert_eq!(label, DTYPE_BF16_ZLIB);
            detail
        }
        other => panic!("expected a decode refusal, got {other}"),
    }
}

fn truncated(stream: &[u8]) -> &[u8] {
    &stream[..stream.len() * TRUNCATE_NUMERATOR / TRUNCATE_DENOMINATOR]
}

// ── The fixture is what it claims ────────────────────────────────────

#[test]
fn the_foreign_fixture_carries_the_rows_its_generator_describes() {
    let k = AWKWARD_K;
    let row = |r: usize| &foreign::AWKWARD_UNITS[r * k..(r + 1) * k];
    let (a, b) = foreign::AWKWARD_REPEATED_ROWS;
    assert_eq!(row(a), row(b), "the repeated row");
    assert!(row(foreign::AWKWARD_ZERO_ROW).iter().all(|u| *u == 0));
    let specials = &row(foreign::AWKWARD_SPECIAL_ROW)[..foreign::SPECIALS.len()];
    assert_eq!(specials, foreign::SPECIALS);
    // The specials are the ONLY non-finite units, so surviving the round
    // trip is a statement about them and not about luck elsewhere.
    let non_finite = foreign::AWKWARD_UNITS
        .iter()
        .filter(|u| **u & 0x7F80 == 0x7F80)
        .count();
    assert_eq!(non_finite, 4, "+inf, -inf and the two NaNs");
}

// ── P1: lossless, against a foreign encoder, at the bit level ────────

#[test]
fn a_foreign_stream_decodes_bit_for_bit_including_every_special_value() {
    assert_eq!(
        decoded_bits(&foreign::AWKWARD_STREAM, &foreign::AWKWARD_SHAPE),
        bits_of(&foreign::AWKWARD_UNITS)
    );
    assert_eq!(
        decoded_bits(&foreign::NOISE_STREAM, &foreign::NOISE_SHAPE),
        bits_of(&foreign::NOISE_UNITS)
    );
}

#[test]
fn the_crate_s_own_encoder_agrees_with_the_foreign_one_on_meaning_not_bytes() {
    let values: Vec<f32> = bits_of(&foreign::AWKWARD_UNITS)
        .into_iter()
        .map(f32::from_bits)
        .collect();
    let ours = encode_bf16_zlib(&values);
    // Two conforming encoders may emit different streams for one image;
    // what they may not do is disagree about the image.
    assert_eq!(
        decoded_bits(&ours, &foreign::AWKWARD_SHAPE),
        bits_of(&foreign::AWKWARD_UNITS)
    );
}

// ── P5 / P6: stored size is a property of the instance ───────────────

#[test]
fn stored_size_falls_on_both_sides_of_the_raw_image_across_instances() {
    let awkward_raw = AWKWARD_ROWS * AWKWARD_K * 2;
    let noise_raw = NOISE_ROWS * NOISE_K * 2;
    assert!(foreign::AWKWARD_STREAM.len() < awkward_raw);
    assert!(foreign::NOISE_STREAM.len() > noise_raw);
}

#[test]
fn stored_bytes_refuses_to_price_from_shape_and_names_the_container_as_the_authority() {
    let err = BF16_ZLIB
        .stored_bytes(
            &foreign::AWKWARD_SHAPE,
            RepresentationExtent::TERMINAL,
            TENSOR,
        )
        .unwrap_err();
    assert_eq!(
        err,
        CodecError::InstanceSized {
            tensor: TENSOR.into(),
            label: DTYPE_BF16_ZLIB.into(),
        }
    );
    assert!(err.to_string().contains("the container records"), "{err}");
    // The extent is judged first, as every terminal codec judges it.
    let deeper = BF16_ZLIB
        .stored_bytes(
            &foreign::AWKWARD_SHAPE,
            RepresentationExtent::at_depth(1),
            TENSOR,
        )
        .unwrap_err();
    assert!(
        matches!(deeper, CodecError::ExtentUnavailable { depth: 1, .. }),
        "{deeper}"
    );
}

// ── P2: sequential, and what that costs ──────────────────────────────

#[test]
fn a_row_range_of_the_foreign_image_is_that_slice_of_the_whole() {
    let all = decoded_bits(&foreign::AWKWARD_STREAM, &foreign::AWKWARD_SHAPE);
    for (start, end) in [(0, 1), (2, 5), (6, 7), (3, 3), (0, AWKWARD_ROWS)] {
        let got: Vec<u32> = decode_rows(
            &foreign::AWKWARD_STREAM,
            &foreign::AWKWARD_SHAPE,
            start..end,
        )
        .unwrap_or_else(|e| panic!("rows {start}..{end}: {e}"))
        .iter()
        .map(|v| v.to_bits())
        .collect();
        assert_eq!(
            got,
            all[start * AWKWARD_K..end * AWKWARD_K],
            "rows {start}..{end}"
        );
    }
}

#[test]
fn a_prefix_is_readable_from_a_truncated_stream_and_the_whole_is_not() {
    // The noise image codes at roughly one byte per byte, so three fifths
    // of its stream carry the first row and not the third.
    let cut = truncated(&foreign::NOISE_STREAM);
    let whole = decoded_bits(&foreign::NOISE_STREAM, &foreign::NOISE_SHAPE);
    let first: Vec<u32> = decode_rows(cut, &foreign::NOISE_SHAPE, 0..1)
        .expect("row 0 lies within the surviving prefix")
        .iter()
        .map(|v| v.to_bits())
        .collect();
    assert_eq!(first, whole[..NOISE_K]);
    let detail = detail_of(decode_rows(cut, &foreign::NOISE_SHAPE, 0..NOISE_ROWS).unwrap_err());
    assert!(detail.contains("ends before row 3 of 3"), "{detail}");
}

#[test]
fn capabilities_are_sequential_and_row_access_is_refused_by_class() {
    let caps = BF16_ZLIB.capabilities();
    assert_eq!(caps.access, AccessGranularity::Sequential);
    assert_eq!(
        (
            caps.group_elems,
            caps.row_align_elems,
            caps.physical_align_bytes
        ),
        (1, 1, 1)
    );
    caps.require(RequiredAccess::Sequential, DTYPE_BF16_ZLIB)
        .unwrap();
    let err = caps
        .require(RequiredAccess::RowRandom, DTYPE_BF16_ZLIB)
        .unwrap_err();
    assert_eq!(
        err,
        CodecError::AccessRefused {
            label: DTYPE_BF16_ZLIB.into(),
            provided: "sequential".into(),
            required: "row-random".into(),
        }
    );
}

// ── P3 / P4: no direct realization; decode is the realization ────────

#[test]
fn it_declares_no_acceleration_and_an_f32_decode_residency() {
    assert!(BF16_ZLIB.accelerations().is_empty());
    assert_eq!(BF16_ZLIB.decode_residency(), ResidencyProfile::DECODED_F32);
    let extents = BF16_ZLIB.extents();
    assert_eq!(extents.len(), 1);
    assert_eq!(
        extents[0].bits_per_weight, 16.0,
        "the supremum, not a measurement"
    );
    assert!(extents[0].radius.is_none());
}

// ── Refusals at the stream's edges ───────────────────────────────────

#[test]
fn a_whole_decode_refuses_a_truncated_stream_naming_the_row_it_reached() {
    let detail = detail_of(
        decode_rows(
            truncated(&foreign::AWKWARD_STREAM),
            &foreign::AWKWARD_SHAPE,
            0..AWKWARD_ROWS,
        )
        .unwrap_err(),
    );
    assert!(detail.contains("ends before row 7 of 7"), "{detail}");
}

#[test]
fn a_whole_decode_refuses_a_corrupted_checksum() {
    let mut stream = foreign::AWKWARD_STREAM.to_vec();
    let last = stream.len() - 1;
    stream[last] ^= 0xFF;
    let detail =
        detail_of(decode_rows(&stream, &foreign::AWKWARD_SHAPE, 0..AWKWARD_ROWS).unwrap_err());
    // The inflater reaches the trailer inside the read that fills the last
    // row, so the refusal arrives as a corrupt stream rather than as a
    // short one — and says so, because the remedies differ.
    assert!(detail.contains("corrupt stream"), "{detail}");
}

#[test]
fn a_whole_decode_refuses_an_image_longer_or_shorter_than_the_shape_implies() {
    let narrower = [AWKWARD_ROWS, AWKWARD_K - 1];
    let detail =
        detail_of(decode_rows(&foreign::AWKWARD_STREAM, &narrower, 0..AWKWARD_ROWS).unwrap_err());
    assert!(detail.contains("inflates past the 504 bytes"), "{detail}");
    let wider = [AWKWARD_ROWS, AWKWARD_K + 1];
    let detail =
        detail_of(decode_rows(&foreign::AWKWARD_STREAM, &wider, 0..AWKWARD_ROWS).unwrap_err());
    assert!(detail.contains("ends before row 7 of 7"), "{detail}");
}

#[test]
fn bytes_after_the_stream_are_refused_because_a_stream_is_the_whole_operand() {
    let mut stream = foreign::AWKWARD_STREAM.to_vec();
    stream.extend_from_slice(&TRAILING_GARBAGE);
    let detail =
        detail_of(decode_rows(&stream, &foreign::AWKWARD_SHAPE, 0..AWKWARD_ROWS).unwrap_err());
    assert!(detail.contains("3 bytes follow the end"), "{detail}");
}

#[test]
fn the_header_is_judged_before_any_byte_is_inflated() {
    let valid = foreign::AWKWARD_STREAM;
    let validate = |stream: &[u8]| {
        BF16_ZLIB.validate(
            &operands(stream),
            &foreign::AWKWARD_SHAPE,
            RepresentationExtent::TERMINAL,
            TENSOR,
        )
    };
    validate(&valid).unwrap();

    let mut method = valid;
    method[0] = (method[0] & 0xF0) | 9;
    assert!(detail_of(validate(&method).unwrap_err()).contains("method 9"));

    let mut window = valid;
    window[0] = (window[0] & 0x0F) | (8 << 4);
    assert!(detail_of(validate(&window).unwrap_err()).contains("window exponent 8"));

    let mut check = valid;
    check[1] ^= 0x01;
    assert!(detail_of(validate(&check).unwrap_err()).contains("check bits"));

    // A dictionary flag with VALID check bits, so the refusal is about
    // the dictionary and not about the arithmetic.
    let mut dictionary = valid;
    dictionary[1] = (0..=u8::MAX)
        .find(|flg| flg & 0x20 != 0 && (u16::from(dictionary[0]) << 8 | u16::from(*flg)) % 31 == 0)
        .expect("some FLG carries FDICT and verifies");
    assert!(detail_of(validate(&dictionary).unwrap_err()).contains("preset dictionary"));

    let short = validate(&valid[..1]).unwrap_err();
    assert!(
        matches!(&short, CodecError::StreamLength { stream, need: 2, have: 1, .. } if stream == "values"),
        "{short}"
    );
}

#[test]
fn a_scalar_and_an_empty_tensor_are_streams_like_any_other() {
    let scalar = encode_bf16_zlib(&[-1.5]);
    let got = BF16_ZLIB
        .decode_packed(&scalar, &[], RepresentationExtent::TERMINAL, TENSOR)
        .unwrap();
    assert_eq!(got, [-1.5]);
    let empty = encode_bf16_zlib(&[]);
    let got = BF16_ZLIB
        .decode_packed(&empty, &[0, 5], RepresentationExtent::TERMINAL, TENSOR)
        .unwrap();
    assert!(got.is_empty());
}

// ── P7: identity names the format, never the provider ────────────────

#[test]
fn the_identity_names_the_wire_format_and_no_implementation() {
    let id = BF16_ZLIB.identity();
    assert_eq!(id.family, DTYPE_BF16_ZLIB);
    assert_eq!(id.revision, BF16_ZLIB_REVISION);
    assert_eq!(id.element, "bf16");
    assert!(id.layout.contains("rfc1950"), "{}", id.layout);
    let rendered = format!("{id:?}").to_lowercase();
    for forbidden in FORBIDDEN_IN_IDENTITY {
        assert!(
            !rendered.contains(forbidden),
            "identity names {forbidden}: {rendered}"
        );
    }
    let registry = CodecRegistry::builtin();
    assert_eq!(
        registry.admit(&id).unwrap().encoding_label(),
        DTYPE_BF16_ZLIB
    );
    let mut future = id.clone();
    future.revision += 1;
    assert!(matches!(
        registry.admit(&future).unwrap_err(),
        CodecError::AbiRevision {
            found: 2,
            implemented: 1,
            ..
        }
    ));
}
