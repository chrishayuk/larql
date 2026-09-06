//! The tables that stated these facts before the contract existed agree
//! with it — the witness that the trait was extracted, not invented.
//!
//! Each of these facts had at least two homes: block bytes in the K-quant
//! table and in the expert-encoding match, group sizes in the models crate
//! and the loader, alignment in the segment writer and the binder. The
//! codec is now the derivation; these tests pin the older statements to
//! it so that a drift fails here and not in a 20 GB segment.

use super::*;
use crate::format::vindex3::represent::physical::{ExpertEncoding, WEIGHT_BINDING_ALIGN};
use larql_models::quant::nvfp4::NVFP4_GROUP_ELEMS;

#[test]
fn the_expert_encoding_table_prices_a_matrix_as_the_codec_does() {
    let registry = CodecRegistry::builtin();
    for (encoding, label) in [
        (ExpertEncoding::Bf16, "BF16"),
        (ExpertEncoding::Q80, "Q8_0"),
        (ExpertEncoding::Q6K, "Q6_K"),
        (ExpertEncoding::Q4K, "Q4_K"),
    ] {
        assert_eq!(encoding.name(), label);
        let codec = registry.resolve(label, TENSOR).unwrap();
        for (n, k) in [(1, 256), (7, 512), (64, 4096)] {
            assert_eq!(
                encoding.matrix_bytes(n, k).unwrap(),
                codec
                    .stored_bytes(&[n, k], RepresentationExtent::BASE, TENSOR)
                    .unwrap(),
                "{label} [{n}, {k}]"
            );
        }
        // And both refuse a row that is not a whole number of blocks.
        assert_eq!(
            encoding.matrix_bytes(2, 100).is_err(),
            codec
                .stored_bytes(&[2, 100], RepresentationExtent::BASE, TENSOR)
                .is_err(),
            "{label}"
        );
    }
}

#[test]
fn the_kquant_table_s_bits_are_the_certificate_s() {
    for codec in [Q4_K, Q6_K, Q8_0] {
        let quant = codec.quant();
        let cert = codec.extents()[0].clone();
        assert_eq!(cert.bits_per_weight, quant.bits_per_weight());
        assert_eq!(
            cert.bits_per_weight,
            quant.bytes_per_block as f64 * extent::BITS_PER_BYTE / quant.elements_per_block as f64
        );
        assert_eq!(codec.capabilities().group_elems, quant.elements_per_block);
        assert_eq!(codec.identity().group_elems, quant.elements_per_block);
    }
}

#[test]
fn the_nvfp4_pack_layout_is_the_codec_s_geometry() {
    for (rows, k) in [(1usize, 16usize), (3, 256), (1024, 4096)] {
        let layout = PackLayout::derive(&[rows, k], TENSOR).unwrap();
        assert_eq!(
            layout.total_len as u64,
            NVFP4
                .stored_bytes(&[rows, k], RepresentationExtent::BASE, TENSOR)
                .unwrap()
        );
    }
    // The layout's per-tensor figure converges on the certificate's
    // asymptotic one as the tensor scale amortises.
    let large = PackLayout::derive(&[1024, 4096], TENSOR).unwrap();
    assert!((large.bits_per_weight() - NVFP4.extents()[0].bits_per_weight).abs() < 1e-3);
    assert_eq!(NVFP4.capabilities().group_elems, NVFP4_GROUP_ELEMS);
    assert_eq!(NVFP4.identity().group_elems, NVFP4_GROUP_ELEMS);
}

#[test]
fn mxfp4_geometry_is_the_models_crate_s_and_nothing_else() {
    assert_eq!(MXFP4.capabilities().group_elems, MXFP4_GROUP_ELEMS);
    assert_eq!(MXFP4.identity().group_elems, MXFP4_GROUP_ELEMS);
    let (rows, k) = (5usize, 96usize);
    let groups = k / MXFP4_GROUP_ELEMS;
    assert_eq!(
        MXFP4
            .stored_bytes(&[rows, k], RepresentationExtent::BASE, TENSOR)
            .unwrap(),
        (rows * groups * (MXFP4_GROUP_BYTES + 1)) as u64
    );
    assert_eq!(MXFP4.extents()[0].bits_per_weight, 4.25);
}

#[test]
fn a_conforming_container_s_alignment_satisfies_every_codec_s_widest_field() {
    // The execution binding requirement (4 bytes, measured) is the
    // conformance bar; no codec may need more than a conforming segment
    // provides, or the Kimi container's 4-aligned expert segment would be
    // condemned for no measurable reason.
    for codec in builtin() {
        assert!(
            codec.capabilities().physical_align_bytes as u64 <= WEIGHT_BINDING_ALIGN,
            "{}",
            codec.encoding_label()
        );
    }
}

#[test]
fn the_float_widths_are_the_element_widths() {
    for (codec, width) in [(BF16, 2usize), (F16, 2), (F32, 4)] {
        assert_eq!(codec.dtype().width_bytes(), width);
        assert_eq!(codec.capabilities().physical_align_bytes, width);
        assert_eq!(
            codec.extents()[0].bits_per_weight,
            width as f64 * extent::BITS_PER_BYTE
        );
        assert_eq!(
            codec
                .stored_bytes(&[ROWS, K], RepresentationExtent::BASE, TENSOR)
                .unwrap(),
            (ROWS * K * width) as u64
        );
        assert_eq!(codec.identity().family, codec.encoding_label());
    }
}
