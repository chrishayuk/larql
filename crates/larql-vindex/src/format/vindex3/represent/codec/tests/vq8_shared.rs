//! The codebook-dependent codec on its own terms: codes that mean
//! nothing alone, a dependency judged twice — once from metadata and once
//! from the values — and a decode that reads a row without reading the
//! rows before it.

use super::super::codecs::vq8_shared::{
    Vq8SharedCodec, CODEBOOK, DTYPE_VQ8_SHARED, VQ8_SHARED, VQ_CODEBOOK_ENTRIES, VQ_VECTOR_ELEMS,
};
use super::super::streams::ResolvedAuxiliary;
use super::*;

/// The codebook shape every `VQ8_SHARED` operand requires.
const BOOK_SHAPE: [usize; 2] = Vq8SharedCodec::CODEBOOK_SHAPE;

fn bound<'a>(codes: &'a [u8], book: &'a (Vec<usize>, Vec<f32>)) -> CodecOperands<'a> {
    CodecOperands {
        streams: NamedStreams::single(VALUES, codes),
        auxiliaries: AuxiliaryOperands::new().with(
            CODEBOOK,
            ResolvedAuxiliary {
                shape: &book.0,
                values: &book.1,
            },
        ),
    }
}

fn codebook() -> (Vec<usize>, Vec<f32>) {
    (BOOK_SHAPE.to_vec(), vq_codebook())
}

/// A code is an index, not a value: decoding is a gather from the
/// dependency, one entry per four weights.
#[test]
fn a_code_stands_for_four_weights_taken_from_the_codebook() {
    let book = codebook();
    // Two vectors, chosen so the codes are far apart in the book.
    let codes = [7u8, 200];
    let shape = [1usize, 2 * VQ_VECTOR_ELEMS];
    let decoded = VQ8_SHARED
        .decode_all(
            &bound(&codes, &book),
            &shape,
            RepresentationExtent::BASE,
            TENSOR,
        )
        .unwrap();
    let expected: Vec<f32> = codes
        .iter()
        .flat_map(|code| {
            let base = usize::from(*code) * VQ_VECTOR_ELEMS;
            book.1[base..base + VQ_VECTOR_ELEMS].to_vec()
        })
        .collect();
    assert_eq!(decoded, expected);
    // The codes alone say nothing about magnitude: the same codes against
    // another codebook decode to other values entirely.
    let shifted = (
        BOOK_SHAPE.to_vec(),
        book.1.iter().map(|v| v + 100.0).collect(),
    );
    let elsewhere = VQ8_SHARED
        .decode_all(
            &bound(&codes, &shifted),
            &shape,
            RepresentationExtent::BASE,
            TENSOR,
        )
        .unwrap();
    assert!(elsewhere.iter().zip(&decoded).all(|(a, b)| a > b));
}

/// Encode and decode meet: nearest-entry coding round-trips to the entry
/// it chose, and the error is the assignment's, not the format's.
#[test]
fn encoding_chooses_an_entry_and_decoding_returns_exactly_that_entry() {
    let book = codebook();
    let values = ramp(ROWS * K);
    let codes = Vq8SharedCodec::encode_codes(&values, &book.1);
    assert_eq!(codes.len(), values.len() / VQ_VECTOR_ELEMS);
    let shape = [ROWS, K];
    let decoded = VQ8_SHARED
        .decode_all(
            &bound(&codes, &book),
            &shape,
            RepresentationExtent::BASE,
            TENSOR,
        )
        .unwrap();
    for (vector, code) in decoded.chunks(VQ_VECTOR_ELEMS).zip(&codes) {
        let base = usize::from(*code) * VQ_VECTOR_ELEMS;
        assert_eq!(vector, &book.1[base..base + VQ_VECTOR_ELEMS]);
    }
    // And the assignment is a real one: the reconstruction is close but
    // never exact, which is why the codec declares no radius — the error
    // belongs to the encoder and the data it was fit to, not to the
    // format. The bound is the fixture codebook's, not the format's.
    assert!(decoded.iter().zip(&values).any(|(d, v)| d != v));
    let worst = decoded
        .iter()
        .zip(&values)
        .map(|(d, v)| (d - v).abs())
        .fold(0.0f32, f32::max);
    assert!(
        worst > 0.0 && worst <= VQ_WORST_COMPONENT_ERROR,
        "worst per-component error {worst}"
    );
}

/// A row is addressable: decoding one row reads that row's codes and
/// gives the same values as decoding everything.
#[test]
fn a_row_decodes_to_that_slice_of_the_whole() {
    let book = codebook();
    let values = ramp(ROWS * K);
    let codes = Vq8SharedCodec::encode_codes(&values, &book.1);
    let shape = [ROWS, K];
    let operands = bound(&codes, &book);
    let whole = VQ8_SHARED
        .decode_all(&operands, &shape, RepresentationExtent::BASE, TENSOR)
        .unwrap();
    let mut middle = vec![0.0f32; K];
    VQ8_SHARED
        .decode_rows(
            &operands,
            &shape,
            1..2,
            RepresentationExtent::BASE,
            &mut middle,
            TENSOR,
        )
        .unwrap();
    assert_eq!(middle, whole[K..2 * K]);
}

/// The codes cost two bits a weight; the codebook's bytes are the
/// CODEBOOK's footprint, not this operand's.
#[test]
fn the_rate_is_the_codes_and_the_codebook_is_another_operands_cost() {
    let shape = [ROWS, K];
    let bytes = VQ8_SHARED
        .stored_bytes(&shape, RepresentationExtent::BASE, TENSOR)
        .unwrap();
    assert_eq!(bytes, (ROWS * K / VQ_VECTOR_ELEMS) as u64);
    let certificate = VQ8_SHARED.extents()[0].clone();
    assert_eq!(
        bytes as f64 * extent::BITS_PER_BYTE / (ROWS * K) as f64,
        certificate.bits_per_weight
    );
    assert_eq!(certificate.bits_per_weight, 2.0);
    // No radius: a vector quantiser's error is its encoder's and its
    // data's, and a number here would make it the format's.
    assert!(certificate.radius.is_none());
}

/// A width that is not a whole number of vectors is refused, because a
/// code cannot stand for part of a vector.
#[test]
fn a_row_that_is_not_whole_vectors_is_refused_by_group() {
    let err = VQ8_SHARED
        .stored_bytes(&[2, 6], RepresentationExtent::BASE, TENSOR)
        .unwrap_err();
    assert!(
        matches!(&err, CodecError::Geometry { why, .. } if why.contains("4-element groups")),
        "{err}"
    );
}

/// The dependency is judged twice, and the second time is not redundant:
/// metadata is what admission has, values are what the decode indexes.
#[test]
fn the_codebook_is_judged_from_metadata_and_again_from_its_values() {
    // From metadata, before any byte: the shape rule.
    let good = AuxiliaryMetadata {
        object: "target.codebooks".into(),
        tensor: "shared".into(),
        label: "F32".into(),
        shape: BOOK_SHAPE.to_vec(),
        identity: None,
    };
    VQ8_SHARED
        .validate_auxiliary(
            CODEBOOK,
            &good,
            &[ROWS, K],
            RepresentationExtent::BASE,
            TENSOR,
        )
        .unwrap();
    let narrow = AuxiliaryMetadata {
        shape: vec![VQ_CODEBOOK_ENTRIES, VQ_VECTOR_ELEMS - 1],
        ..good.clone()
    };
    let err = VQ8_SHARED
        .validate_auxiliary(
            CODEBOOK,
            &narrow,
            &[ROWS, K],
            RepresentationExtent::BASE,
            TENSOR,
        )
        .unwrap_err();
    assert!(
        matches!(&err, CodecError::AuxiliaryGeometry { name, label, .. }
            if name == CODEBOOK && label == DTYPE_VQ8_SHARED),
        "{err}"
    );

    // From the values, at decode: a codebook that arrived short would be
    // indexed past its end, so the same rule is met again.
    let short = (BOOK_SHAPE.to_vec(), vec![0.0f32; 16]);
    let codes = [0u8, 1];
    let err = VQ8_SHARED
        .validate(
            &bound(&codes, &short),
            &[1, 2 * VQ_VECTOR_ELEMS],
            RepresentationExtent::BASE,
            TENSOR,
        )
        .unwrap_err();
    assert!(
        matches!(&err, CodecError::AuxiliaryGeometry { why, .. } if why.contains("16 values")),
        "{err}"
    );
}

/// Without its dependency the codes are not a degraded tensor — they are
/// not a tensor at all, and the refusal says so by name.
#[test]
fn codes_without_a_codebook_are_refused_by_name() {
    let codes = [0u8, 1];
    let operands = CodecOperands::from_streams(NamedStreams::single(VALUES, &codes));
    let err = VQ8_SHARED
        .validate(
            &operands,
            &[1, 2 * VQ_VECTOR_ELEMS],
            RepresentationExtent::BASE,
            TENSOR,
        )
        .unwrap_err();
    assert!(
        matches!(&err, CodecError::MissingAuxiliary { name, label, .. }
            if name == CODEBOOK && label == DTYPE_VQ8_SHARED),
        "{err}"
    );
}

/// The stream is judged on its own terms too: one code per vector,
/// exactly.
#[test]
fn a_codes_stream_of_the_wrong_length_is_refused_before_the_codebook() {
    let book = codebook();
    let shape = [1usize, 4 * VQ_VECTOR_ELEMS];
    for codes in [vec![0u8; 3], vec![0u8; 5]] {
        let err = VQ8_SHARED
            .validate(
                &bound(&codes, &book),
                &shape,
                RepresentationExtent::BASE,
                TENSOR,
            )
            .unwrap_err();
        assert!(
            matches!(&err, CodecError::StreamLength { need, have, .. }
                if *need == 4 && *have == codes.len()),
            "{err}"
        );
    }
}
