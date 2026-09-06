//! The contract's tests, and the fixtures they share.
//!
//! Every codec the registry ships is exercised through the SAME table, so
//! a test that passes for the floats and fails for MXFP4 is a statement
//! about MXFP4's declaration and not about the test.

mod auxiliary;
mod baseline;
mod bf16_zlib;
mod capability;
mod composition;
mod contract;
mod decode;
mod f32_planes;
mod fidelity;
mod fixtures;
mod geometry;
mod lyrw2;
mod ranges;
mod registry;
mod residency;
mod streams;
mod vq8_shared;

use super::codecs::bf16_zlib::BF16_ZLIB;
use super::codecs::f32_planes::{F32PlanesCodec, F32_PLANES};
use super::codecs::float::{BF16, F16, F32};
use super::codecs::kquant::{Q4_K, Q6_K, Q8_0};
use super::codecs::mxfp4::{DTYPE_MXFP4, MXFP4};
use super::codecs::nvfp4::NVFP4;
use super::codecs::vq8_shared::{
    Vq8SharedCodec, CODEBOOK as VQ_CODEBOOK, VQ8_SHARED, VQ_CODEBOOK_ENTRIES, VQ_VECTOR_ELEMS,
};
// Named explicitly: the `streams` TEST module below shadows the crate's
// `streams` module for every file under it.
use super::streams::{ResolvedAuxiliary, GROUP_SCALES, TENSOR_SCALE, VALUES};
use super::*;
use crate::format::vindex3::fixtures::encode_bf16_zlib;
use crate::format::vindex3::opplan::exec::weights::{quantize_mxfp4, LoadedWeight};
use crate::format::vindex3::represent::nvfp4_pack::{encode as encode_nvfp4, PackLayout};
use larql_models::quant::half::{encode_bf16, encode_f16};
use larql_models::quant::mxfp4::{MXFP4_GROUP_BYTES, MXFP4_GROUP_ELEMS};

pub(super) const TENSOR: &str = "layer.0.w";
/// Every fixture is this shape: three rows of one K-quant super-block,
/// which is also a whole number of every smaller group.
pub(super) const ROWS: usize = 3;
pub(super) const K: usize = 256;

/// Values that are not symmetric about zero and not periodic in any
/// block size, so a row or block read from the wrong place is visible.
pub(super) fn ramp(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (i as f32 * 0.37 - 7.0) * 0.03125 + (i % 13) as f32 * 0.01)
        .collect()
}

/// A codebook whose entries span the RAMP's range — `[-0.22, 8.76]` — so
/// nearest-entry coding is a real assignment rather than everything
/// clamping to the end of a book that does not reach the data.
///
/// Its entries are constant vectors, evenly spaced, which leaves two
/// sources of error a test can reason about: half an entry spacing
/// (`10 / 255 / 2`), and half the spread WITHIN a four-weight vector,
/// which the ramp's every-thirteenth-element step makes as large as
/// ~0.08. Nothing here is a good quantiser; it is a fixture whose error
/// is predictable.
pub(super) const VQ_CODEBOOK_LOW: f32 = -0.5;
pub(super) const VQ_CODEBOOK_HIGH: f32 = 9.5;
/// The worst per-component error the fixture codebook can leave on the
/// ramp — measured, and asserted against so a regression in either the
/// codebook or the coder shows up as a number rather than a vibe.
pub(super) const VQ_WORST_COMPONENT_ERROR: f32 = 0.09;

pub(super) fn vq_codebook() -> Vec<f32> {
    let span = VQ_CODEBOOK_HIGH - VQ_CODEBOOK_LOW;
    (0..VQ_CODEBOOK_ENTRIES * VQ_VECTOR_ELEMS)
        .map(|i| {
            let entry = (i / VQ_VECTOR_ELEMS) as f32;
            VQ_CODEBOOK_LOW + entry * span / (VQ_CODEBOOK_ENTRIES - 1) as f32
        })
        .collect()
}

pub(super) fn builtin() -> Vec<&'static dyn RepresentationCodec> {
    CodecRegistry::builtin().codecs().collect()
}

/// One encoded fixture: the bytes a container would hold, how they bind
/// onto the codec's streams, and — for a codec that depends on another
/// represented object — that dependency already resolved.
pub(super) struct Fixture {
    pub codec: &'static dyn RepresentationCodec,
    pub shape: Vec<usize>,
    /// Owned stream bytes in declaration order.
    pub buffers: Vec<Vec<u8>>,
    /// Whether the buffers are one packed payload (bound through
    /// `bind_packed`) or one buffer per declared stream.
    pub packed: bool,
    /// Dependencies the codec requires, RESOLVED: name, shape, values.
    /// Empty for every codec that reads only its own bytes — which is
    /// every codec but one.
    pub auxiliaries: Vec<(&'static str, Vec<usize>, Vec<f32>)>,
}

impl Fixture {
    pub fn operands(&self) -> CodecOperands<'_> {
        let streams = if self.packed {
            self.codec
                .bind_packed(&self.buffers[0], &self.shape, TENSOR)
                .expect("a packed fixture binds")
        } else {
            let mut streams = NamedStreams::new();
            for (spec, bytes) in self.codec.streams().iter().zip(&self.buffers) {
                streams = streams.with(*spec, bytes);
            }
            streams
        };
        let mut operands = CodecOperands::from_streams(streams);
        for (name, shape, values) in &self.auxiliaries {
            operands.auxiliaries = std::mem::take(&mut operands.auxiliaries)
                .with(*name, ResolvedAuxiliary { shape, values });
        }
        operands
    }

    pub fn label(&self) -> &'static str {
        self.codec.encoding_label()
    }
}

/// A fixture per built-in codec, all of `[ROWS, K]`.
pub(super) fn fixtures() -> Vec<Fixture> {
    let shape = vec![ROWS, K];
    let values = ramp(ROWS * K);
    let f32_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let packed = |codec: &'static dyn RepresentationCodec, bytes: Vec<u8>| Fixture {
        codec,
        shape: shape.clone(),
        buffers: vec![bytes],
        packed: true,
        auxiliaries: Vec::new(),
    };
    let mut out = vec![
        packed(&BF16, encode_bf16(&values)),
        packed(&F16, encode_f16(&values)),
        packed(&F32, f32_bytes),
    ];
    for k in [Q4_K, Q6_K, Q8_0] {
        let bytes = k.quant().encode(&values, TENSOR).expect("encodes");
        out.push(Fixture {
            codec: match k.quant().name {
                "Q4_K" => &Q4_K,
                "Q6_K" => &Q6_K,
                _ => &Q8_0,
            },
            shape: shape.clone(),
            buffers: vec![bytes],
            packed: true,
            auxiliaries: Vec::new(),
        });
    }
    let layout = PackLayout::derive(&shape, TENSOR).expect("layout");
    let matrix = larql_models::quant::nvfp4::quantize(&values, ROWS, K).expect("nvfp4");
    out.push(packed(
        &NVFP4,
        encode_nvfp4(&matrix, &layout, TENSOR).expect("packs"),
    ));
    // The sequential codec: the same ramp, one zlib stream. Its stored
    // size is whatever the stream came to, which is the point. Pushed
    // here, before MXFP4's destructuring shadows `packed`.
    out.push(packed(&BF16_ZLIB, encode_bf16_zlib(&values)));
    // The progressive codec: three planes of the same ramp, bound as three
    // streams. Its fixture is the whole of it — the terminal extent —
    // because a fixture is what a container holds, and a container holds
    // every plane whatever extent is later selected.
    let (base, refine_a, refine_b) = F32PlanesCodec::encode_planes(&values);
    out.push(Fixture {
        codec: &F32_PLANES,
        shape: shape.clone(),
        buffers: vec![base, refine_a, refine_b],
        packed: false,
        auxiliaries: Vec::new(),
    });
    // The dependency-bearing codec: its codes mean nothing without the
    // codebook beside them, so its fixture carries one — RESOLVED, the
    // way the loader will hand it over once a container declares it.
    let codebook = vq_codebook();
    out.push(Fixture {
        codec: &VQ8_SHARED,
        shape: shape.clone(),
        buffers: vec![Vq8SharedCodec::encode_codes(&values, &codebook)],
        packed: false,
        auxiliaries: vec![(
            VQ_CODEBOOK,
            Vq8SharedCodec::CODEBOOK_SHAPE.to_vec(),
            codebook,
        )],
    });
    let LoadedWeight::Mxfp4 { packed, scales } =
        quantize_mxfp4(&values, ROWS, K, TENSOR).expect("mxfp4")
    else {
        panic!("quantize_mxfp4 yields the MXFP4 residency");
    };
    let groups = K / MXFP4_GROUP_ELEMS;
    out.push(Fixture {
        codec: &MXFP4,
        shape,
        buffers: vec![
            packed.as_slice()[..ROWS * groups * MXFP4_GROUP_BYTES].to_vec(),
            scales.as_slice()[..ROWS * groups].to_vec(),
        ],
        packed: false,
        auxiliaries: Vec::new(),
    });
    assert_eq!(out.len(), builtin().len(), "one fixture per built-in codec");
    assert!(out.iter().any(|f| f.label() == DTYPE_MXFP4));
    out
}

/// A foreign implementor that overrides nothing optional, so the trait's
/// defaults are exercised by a codec this crate does not otherwise ship.
pub(super) struct Stub {
    pub label: &'static str,
    pub identity: super::super::nvfp4_pack::CodecIdentity,
    pub streams: &'static [StreamSpec],
}

impl RepresentationCodec for Stub {
    fn encoding_label(&self) -> &'static str {
        self.label
    }
    fn identity(&self) -> super::super::nvfp4_pack::CodecIdentity {
        self.identity.clone()
    }
    fn streams(&self) -> &'static [StreamSpec] {
        self.streams
    }
    fn capabilities(&self) -> CodecCapabilities {
        CodecCapabilities {
            access: AccessGranularity::Sequential,
            group_elems: 1,
            row_align_elems: 1,
            physical_align_bytes: 1,
        }
    }
    fn extents(&self) -> Vec<ExtentCertificate> {
        vec![ExtentCertificate::terminal(1.0)]
    }
    fn stored_bytes(
        &self,
        _: &[usize],
        _: RepresentationExtent,
        _: &str,
    ) -> Result<u64, CodecError> {
        Ok(0)
    }
    fn validate(
        &self,
        _: &CodecOperands<'_>,
        _: &[usize],
        _: RepresentationExtent,
        _: &str,
    ) -> Result<(), CodecError> {
        Ok(())
    }
    fn decode_rows(
        &self,
        _: &CodecOperands<'_>,
        _: &[usize],
        _: std::ops::Range<usize>,
        _: RepresentationExtent,
        dst: &mut [f32],
        _: &str,
    ) -> Result<(), CodecError> {
        dst.fill(0.0);
        Ok(())
    }
    fn decode_residency(&self) -> ResidencyProfile {
        ResidencyProfile::DECODED_F32
    }
}
