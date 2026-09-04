//! The geometry table is checked against the codecs, never believed.

use super::*;

/// Values with enough structure that a decoder reading them under the
/// wrong layout cannot accidentally agree — a constant or a symmetric
/// ramp would let a wrong reading look right.
fn sample(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = i as f32;
            (x * 0.017).sin() * (1.0 + (x % 7.0)) - 0.31 * (x % 13.0)
        })
        .collect()
}

#[test]
fn the_table_agrees_with_every_encoder() {
    for k in COMPILABLE {
        // Four blocks, so a per-block error shows up as a multiple.
        let n = k.elements_per_block * 4;
        let bytes = k.encode(&sample(n), "t").expect("encodes");
        assert_eq!(
            bytes.len(),
            k.encoded_len(n, "t").unwrap(),
            "{}: encoder and geometry table disagree",
            k.name
        );
        assert_eq!(
            bytes.len(),
            4 * k.bytes_per_block,
            "{}: four blocks should be four block-lengths",
            k.name
        );
    }
}

#[test]
fn every_encoding_round_trips_to_its_element_count() {
    for k in COMPILABLE {
        let n = k.elements_per_block * 3;
        let bytes = k.encode(&sample(n), "t").expect("encodes");
        let back = k.decode(&bytes, n, "t").expect("decodes");
        assert_eq!(back.len(), n, "{}: element count lost", k.name);
    }
}

/// The round trip is lossy, and it has to be *visibly* lossy: a decode
/// that returned the input exactly would mean the encoder never ran.
#[test]
fn the_round_trip_is_lossy_but_close() {
    for k in COMPILABLE {
        let n = k.elements_per_block * 4;
        let src = sample(n);
        let back = k
            .decode(&k.encode(&src, "t").unwrap(), n, "t")
            .expect("decodes");
        let err: f32 = src
            .iter()
            .zip(&back)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(
            err > 0.0,
            "{}: a lossless K-quant would mean no encode",
            k.name
        );
        let scale = src.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(
            err < 0.25 * scale,
            "{}: max error {err} is too large against scale {scale} to be quantisation",
            k.name
        );
    }
}

/// Fewer bits must cost more error. This is the ordering the whole
/// Pareto anchor curve assumes, pinned at the codec level so a swapped
/// table entry cannot invert it silently.
#[test]
fn error_increases_as_bits_decrease() {
    let n = 256 * 4;
    let src = sample(n);
    let rms = |k: KQuant| -> f32 {
        let back = k.decode(&k.encode(&src, "t").unwrap(), n, "t").unwrap();
        (src.iter()
            .zip(&back)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / n as f32)
            .sqrt()
    };
    let (q8, q6, q4) = (rms(Q8_0), rms(Q6_K), rms(Q4_K));
    assert!(q8 < q6, "Q8_0 rms {q8} should beat Q6_K {q6}");
    assert!(q6 < q4, "Q6_K rms {q6} should beat Q4_K {q4}");
}

/// The control the discipline demands: prove the decode could have
/// returned the other answer. If reading Q6_K bytes as Q4_K produced
/// the same values, `decode` would not actually depend on the encoding
/// and every agreement it reports would be worthless.
#[test]
fn reading_one_encoding_as_another_gives_a_different_answer() {
    let n = 256;
    let src = sample(n);
    let q6_bytes = Q6_K.encode(&src, "t").unwrap();
    let as_q6 = Q6_K.decode(&q6_bytes, n, "t").unwrap();
    // Q6_K's 210 bytes are more than Q4_K's 144, so the wrong reading
    // has enough bytes to succeed — which is what makes it a real test.
    let as_q4 = Q4_K
        .decode(&q6_bytes, n, "t")
        .expect("enough bytes for the wrong reading to complete");
    assert_ne!(as_q6, as_q4, "the decode does not depend on the encoding");
}

#[test]
fn a_ragged_element_count_is_refused_naming_the_geometry() {
    for k in COMPILABLE {
        let n = k.elements_per_block + 1;
        let err = k.encoded_len(n, "some.tensor").unwrap_err().to_string();
        assert!(err.contains("some.tensor"), "{}: {err}", k.name);
        assert!(
            err.contains(&k.elements_per_block.to_string()),
            "{}: the refusal must name the block size: {err}",
            k.name
        );
        assert!(err.contains("share a scale"), "{}: {err}", k.name);
    }
}

/// A short segment is named here, not downstream as a mysteriously
/// empty vector.
#[test]
fn decoding_refuses_a_ragged_count_before_reading_bytes() {
    let err = Q6_K.decode(&[0u8; 210], 257, "t").unwrap_err().to_string();
    assert!(err.contains("256"), "{err}");
}

#[test]
fn bits_per_weight_matches_the_published_ratios() {
    assert_eq!(Q8_0.bits_per_weight(), 8.5);
    assert_eq!(Q6_K.bits_per_weight(), 6.5625);
    assert_eq!(Q4_K.bits_per_weight(), 4.5);
}

#[test]
fn lookup_accepts_exactly_the_compilable_names() {
    for k in COMPILABLE {
        assert_eq!(lookup(k.name), Some(k));
    }
    // Case matters: a `PrecisionMap` naming `q6_k` would compile a pack
    // that the `object@Q6_K` selection never finds.
    assert_eq!(lookup("q6_k"), None);
    // Decoders for these exist in the workspace; encoders do not, and an
    // encoding whose bytes cannot be written is not on offer.
    assert_eq!(lookup("Q3_K"), None);
    assert_eq!(lookup("Q2_K"), None);
    assert_eq!(lookup("Q5_K"), None);
    assert_eq!(lookup("NVFP4"), None);
}

#[test]
fn the_error_message_lists_what_is_actually_compilable() {
    let names = compilable_names();
    for k in COMPILABLE {
        assert!(names.contains(k.name), "{names} omits {}", k.name);
    }
}

/// The ggml type is what makes a compiled pack a byte pass-through
/// candidate on export, so a wrong constant here would be invisible
/// until a GGUF was unreadable.
#[test]
fn ggml_types_are_the_published_constants() {
    use larql_models::quant::ggml::{TYPE_Q4_K, TYPE_Q6_K, TYPE_Q8_0};
    assert_eq!(Q8_0.ggml_type, TYPE_Q8_0);
    assert_eq!(Q6_K.ggml_type, TYPE_Q6_K);
    assert_eq!(Q4_K.ggml_type, TYPE_Q4_K);
}

/// The fixture that separates the right check from the plausible one.
///
/// `[2, 128]` is 256 elements — exactly one Q6_K super-block — so a
/// planner that checked the element PRODUCT would accept it and emit a
/// block spanning the end of row 0 and the start of row 1 under one
/// shared scale. ggml blocks along the row, so the row length is what
/// must divide. Both facts are asserted here so the test fails if the
/// check is ever "simplified" back to the total.
#[test]
fn a_shape_whose_total_divides_but_whose_row_does_not_is_refused() {
    let shape = [2usize, 128];
    let elements: usize = shape.iter().product();
    assert_eq!(
        elements % Q6_K.elements_per_block,
        0,
        "fixture is pointless unless the element total DOES divide"
    );
    assert!(
        Q6_K.encoded_len(elements, "t").is_ok(),
        "the flat check must accept it — that is what makes it a trap"
    );

    let err = Q6_K.plan(&shape, "some.tensor").unwrap_err().to_string();
    assert!(err.contains("row length 128"), "{err}");
    assert!(err.contains("share a scale"), "{err}");
}

#[test]
fn plan_sizes_a_real_matrix_from_its_row_length() {
    // Qwen3.8's hidden size, which is a whole number of super-blocks.
    let shape = [1024usize, 5120];
    for k in COMPILABLE {
        let want = 1024 * (5120 / k.elements_per_block) * k.bytes_per_block;
        assert_eq!(k.plan(&shape, "t").unwrap(), want, "{}", k.name);
    }
}

#[test]
fn plan_refuses_a_scalar_rather_than_succeeding_degenerately() {
    let err = Q6_K.plan(&[], "t").unwrap_err().to_string();
    assert!(err.contains("no row to block along"), "{err}");
}

/// A 1-D operand with a conforming length is plannable — the role
/// policy is what keeps norms and biases at source precision, not an
/// accident of the geometry check.
#[test]
fn plan_accepts_a_conforming_one_dimensional_operand() {
    assert_eq!(Q6_K.plan(&[512], "t").unwrap(), 2 * Q6_K.bytes_per_block);
}

/// A shape whose element count overflows `usize` is refused rather than
/// wrapping into a plausible-looking small length.
#[test]
fn a_shape_that_overflows_an_element_count_is_refused() {
    let err = Q6_K
        .plan(&[usize::MAX, 256], "some.tensor")
        .unwrap_err()
        .to_string();
    assert!(err.contains("overflows an element count"), "{err}");
    assert!(err.contains("some.tensor"), "{err}");
}

/// A row in the geometry table with no encoder behind it must refuse,
/// not fall through to binding source bytes under a name that claims
/// they were encoded. Constructed directly because the shipped table
/// deliberately contains no such row — the guard exists for the moment
/// someone adds one.
#[test]
fn a_table_row_without_an_encoder_refuses_rather_than_passing_bytes_through() {
    let phantom = KQuant {
        name: "Q9_9",
        ggml_type: 0,
        elements_per_block: 32,
        bytes_per_block: 34,
    };
    let err = phantom
        .encode(&sample(64), "some.tensor")
        .unwrap_err()
        .to_string();
    assert!(err.contains("has no encoder"), "{err}");
    assert!(err.contains("claims otherwise"), "{err}");
}

/// If the table and the codec ever disagree about a length, the write is
/// refused at that point rather than producing a segment whose tensor
/// table does not describe its payload.
#[test]
fn a_table_that_disagrees_with_its_codec_is_caught_at_encode_time() {
    let wrong = KQuant {
        bytes_per_block: 99, // Q8_0 really writes 34
        ..Q8_0
    };
    let err = wrong
        .encode(&sample(64), "some.tensor")
        .unwrap_err()
        .to_string();
    assert!(err.contains("geometry implies"), "{err}");
    assert!(
        err.contains("table and the codec disagree"),
        "the message must say which two things disagree: {err}"
    );
}

/// A truncated segment is named as a decode failure for this tensor, not
/// surfaced as a short vector somewhere downstream.
#[test]
fn a_truncated_pack_fails_the_decode_by_name() {
    let full = Q6_K.encode(&sample(512), "t").unwrap();
    let err = Q6_K
        .decode(&full[..full.len() - 1], 512, "some.tensor")
        .unwrap_err()
        .to_string();
    assert!(err.contains("some.tensor"), "{err}");
    assert!(err.contains("Q6_K"), "{err}");
}

/// Each K-quant's codec identity is its OWN family, and the ABI gate
/// admits it. Two different K-quants must not admit each other's
/// contract — that is the whole reason they are separate families.
#[test]
fn each_kquant_codec_identity_is_admitted_and_is_not_another_s() {
    for k in COMPILABLE {
        let id = k.codec_identity();
        assert_eq!(id.family, k.name);
        id.admit()
            .unwrap_or_else(|e| panic!("{}: this build wrote it and must admit it: {e}", k.name));
        for other in COMPILABLE.into_iter().filter(|o| o.name != k.name) {
            assert_ne!(
                id,
                other.codec_identity(),
                "{} and {} share a codec identity",
                k.name,
                other.name
            );
        }
    }
}
