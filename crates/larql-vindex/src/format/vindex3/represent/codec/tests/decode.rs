//! Each codec decodes to exactly what the path it replaced decoded to.
//!
//! The reference on every arm is the decoder the executor already used
//! — the operand widener, `KQuant::decode`, the NVFP4 pack split and the
//! models-crate MXFP4 decoder — so the trait is shown to be an
//! extraction and not a second opinion.

use super::*;
use crate::format::vindex3::opplan::exec::operands::widen;
use crate::format::vindex3::represent::nvfp4_pack::split;
use larql_models::quant::mxfp4::dequantize_expert;
use larql_models::quant::nvfp4::{dequantize_into, Nvfp4Matrix};

/// One unit in the last place of a bf16 mantissa (7 bits), relative.
const BF16_RELATIVE_STEP: f32 = 1.0 / 128.0;

fn decoded(fixture: &Fixture) -> Vec<f32> {
    fixture
        .codec
        .decode_all(
            &fixture.operands(),
            &fixture.shape,
            RepresentationExtent::TERMINAL,
            TENSOR,
        )
        .unwrap_or_else(|e| panic!("{}: {e}", fixture.label()))
}

#[test]
fn floats_decode_through_the_operand_widener() {
    for fixture in fixtures().iter().filter(|f| f.codec.streams().len() == 1) {
        let label = fixture.label();
        if !["BF16", "F16", "F32"].contains(&label) {
            continue;
        }
        let expected = widen(label, &fixture.buffers[0], TENSOR).unwrap();
        assert_eq!(decoded(fixture), expected, "{label}");
        // And a float is a float: the ramp survives to the narrowest
        // mantissa here (bf16's 7 bits), which is a plumbing check — a
        // row read from the wrong place would miss by the ramp's span.
        let values = ramp(ROWS * K);
        let span = values.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let worst = decoded(fixture)
            .iter()
            .zip(&values)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            worst <= span * BF16_RELATIVE_STEP,
            "{label}: worst {worst} of span {span}"
        );
    }
}

#[test]
fn kquants_decode_through_the_workspace_decoder() {
    for codec in [Q4_K, Q6_K, Q8_0] {
        let fixture = fixtures()
            .into_iter()
            .find(|f| f.label() == codec.encoding_label())
            .unwrap();
        let expected = codec
            .quant()
            .decode(&fixture.buffers[0], ROWS * K, TENSOR)
            .unwrap();
        assert_eq!(decoded(&fixture), expected, "{}", codec.encoding_label());
    }
}

#[test]
fn nvfp4_decodes_through_the_pack_split_and_the_reference_decoder() {
    let fixture = fixtures()
        .into_iter()
        .find(|f| f.label() == "NVFP4")
        .unwrap();
    let layout = PackLayout::derive(&fixture.shape, TENSOR).unwrap();
    let (packed, scales, tensor_scale) = split(&fixture.buffers[0], &layout, TENSOR).unwrap();
    let matrix = Nvfp4Matrix {
        packed: packed.to_vec(),
        scales: scales.to_vec(),
        tensor_scale,
    };
    let mut expected = vec![0.0f32; ROWS * K];
    dequantize_into(&matrix, ROWS, K, &mut expected).unwrap();
    assert_eq!(decoded(&fixture), expected);
    // Bound through the packed row, the three streams are the pack's
    // three regions, in order.
    let operands = fixture.operands();
    assert_eq!(
        operands.streams.names(),
        ["values", "group_scales", "tensor_scale"]
    );
    let tail = operands.stream(TENSOR_SCALE, "NVFP4", TENSOR).unwrap();
    assert_eq!(tail.len(), std::mem::size_of::<f32>());
}

#[test]
fn mxfp4_decodes_its_two_streams_through_the_models_crate() {
    let fixture = fixtures()
        .into_iter()
        .find(|f| f.label() == DTYPE_MXFP4)
        .unwrap();
    let groups = K / MXFP4_GROUP_ELEMS;
    let expected =
        dequantize_expert(&fixture.buffers[0], &fixture.buffers[1], ROWS, groups).unwrap();
    assert_eq!(decoded(&fixture), expected);
}

#[test]
fn mxfp4_refuses_one_payload_by_naming_the_streams_it_keeps_apart() {
    let err = MXFP4.bind_packed(&[0u8; 64], &[1, 32], TENSOR).unwrap_err();
    assert_eq!(
        err,
        CodecError::StreamsStoredApart {
            tensor: TENSOR.into(),
            label: DTYPE_MXFP4.into(),
            streams: vec!["values".into(), "group_scales".into()],
        }
    );
    assert!(
        err.to_string().contains("bind each stream by name"),
        "{err}"
    );
    // And therefore the packed path refuses too, before any decode.
    assert!(MXFP4
        .decode_packed(&[0u8; 64], &[1, 32], RepresentationExtent::TERMINAL, TENSOR)
        .is_err());
}

#[test]
fn decode_packed_is_bind_then_validate_then_decode_for_every_packed_codec() {
    for fixture in fixtures().iter().filter(|f| f.packed) {
        let via_packed = fixture
            .codec
            .decode_packed(
                &fixture.buffers[0],
                &fixture.shape,
                RepresentationExtent::TERMINAL,
                TENSOR,
            )
            .unwrap_or_else(|e| panic!("{}: {e}", fixture.label()));
        assert_eq!(via_packed, decoded(fixture), "{}", fixture.label());
    }
}

#[test]
fn every_fixture_validates_and_is_the_size_the_codec_declares() {
    for fixture in fixtures() {
        let operands = fixture.operands();
        fixture
            .codec
            .validate(
                &operands,
                &fixture.shape,
                RepresentationExtent::TERMINAL,
                TENSOR,
            )
            .unwrap_or_else(|e| panic!("{}: {e}", fixture.label()));
        let declared = fixture
            .codec
            .stored_bytes(&fixture.shape, RepresentationExtent::TERMINAL, TENSOR)
            .unwrap();
        let held: usize = fixture.buffers.iter().map(Vec::len).sum();
        assert_eq!(held as u64, declared, "{}", fixture.label());
    }
}
