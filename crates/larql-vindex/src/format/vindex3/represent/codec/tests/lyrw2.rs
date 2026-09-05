//! A LYRW v2 region schema reaches the contract, or is refused by tag.

use super::*;
use crate::format::lyrw2::region_format::{Packing, RegionFormat};
use crate::format::vindex3::represent::codec::codecs::lyrw2::{
    bind_region, codec_label, region_binding, RegionBinding,
};

#[test]
fn every_registered_region_format_names_a_built_in_codec() {
    let registry = CodecRegistry::builtin();
    for (format, label) in [
        (RegionFormat::F32, "F32"),
        (RegionFormat::F16, "F16"),
        (RegionFormat::BF16, "BF16"),
        (RegionFormat::Q4K, "Q4_K"),
        (RegionFormat::Q6K, "Q6_K"),
        (RegionFormat::Q8_0, "Q8_0"),
        (RegionFormat::Mxfp4, "MXFP4"),
        (RegionFormat::Nvfp4, "NVFP4"),
    ] {
        assert_eq!(codec_label(format), Some(label), "{format:?}");
        assert!(registry.by_label(label).is_some(), "{label}");
    }
}

#[test]
fn a_recognised_tag_with_no_codec_is_none_not_a_guess() {
    for format in [
        RegionFormat::Q4_0,
        RegionFormat::Fp4Larql,
        RegionFormat::Mxfp8,
        RegionFormat::Unknown(99),
    ] {
        assert_eq!(codec_label(format), None, "{format:?}");
    }
}

#[test]
fn a_packing_says_how_the_streams_arrive() {
    assert_eq!(
        region_binding(Packing::RowMajor),
        Some(RegionBinding::Single)
    );
    assert_eq!(
        region_binding(Packing::BlocksWithScalesInline),
        Some(RegionBinding::Single)
    );
    assert_eq!(
        region_binding(Packing::BlocksValues),
        Some(RegionBinding::PairedValues)
    );
    assert_eq!(
        region_binding(Packing::BlocksScales),
        Some(RegionBinding::PairedScales)
    );
    assert_eq!(region_binding(Packing::Unknown(7)), None);
}

#[test]
fn a_paired_mxfp4_region_binds_both_streams_and_validates() {
    let fixture = fixtures()
        .into_iter()
        .find(|f| f.label() == DTYPE_MXFP4)
        .unwrap();
    let operands = bind_region(
        &MXFP4,
        &fixture.shape,
        &fixture.buffers[0],
        Some(&fixture.buffers[1]),
        TENSOR,
    )
    .unwrap();
    assert_eq!(operands.streams.names(), ["values", "group_scales"]);
    assert!(operands.auxiliaries.is_empty());
}

#[test]
fn an_mxfp4_region_without_its_partner_is_refused_by_name() {
    let fixture = fixtures()
        .into_iter()
        .find(|f| f.label() == DTYPE_MXFP4)
        .unwrap();
    let err = bind_region(&MXFP4, &fixture.shape, &fixture.buffers[0], None, TENSOR).unwrap_err();
    assert!(
        matches!(err, CodecError::StreamsStoredApart { .. }),
        "{err}"
    );
}

#[test]
fn a_partner_the_codec_never_declared_is_refused_by_name() {
    let fixture = fixtures()
        .into_iter()
        .find(|f| f.label() == "Q6_K")
        .unwrap();
    let scales = [0u8; 8];
    let err = bind_region(
        &Q6_K,
        &fixture.shape,
        &fixture.buffers[0],
        Some(&scales),
        TENSOR,
    )
    .unwrap_err();
    assert_eq!(
        err,
        CodecError::UnexpectedStream {
            tensor: TENSOR.into(),
            label: "Q6_K".into(),
            stream: "group_scales".into(),
            declared: vec!["values".into()],
        }
    );
    assert!(err.to_string().contains("declares no stream"), "{err}");
}

#[test]
fn an_inline_region_binds_as_one_payload_and_a_short_one_is_refused() {
    let fixture = fixtures()
        .into_iter()
        .find(|f| f.label() == "Q6_K")
        .unwrap();
    let operands = bind_region(&Q6_K, &fixture.shape, &fixture.buffers[0], None, TENSOR).unwrap();
    assert_eq!(operands.streams.names(), ["values"]);
    let short = &fixture.buffers[0][..fixture.buffers[0].len() / 2];
    let err = bind_region(&Q6_K, &fixture.shape, short, None, TENSOR).unwrap_err();
    assert!(matches!(err, CodecError::StreamLength { .. }), "{err}");
}
