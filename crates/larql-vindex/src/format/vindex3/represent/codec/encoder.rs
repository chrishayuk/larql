//! **The write-side counterpart to [`RepresentationCodec`]** — additive,
//! optional, and deliberately kept out of the decode contract it sits
//! beside.
//!
//! `RepresentationCodec` is closed on purpose: rung 3's own record calls
//! it a contract meant to be frozen once the forcing proofs (progressive,
//! codebook-dependent, entropy-coded) land without changing it, and its
//! `decode_rows` is "the universal correctness surface" every codec must
//! answer. Nothing about that should change here. What that trait does
//! NOT provide is any way for LARQL itself to WRITE bytes a codec would
//! recognize — today every registered codec is what
//! `docs/represent-codec-contract.md` calls DECODABLE, and becoming
//! COMPILABLE (what `vindex represent`-style tooling can produce) is a
//! separate, unrelated mechanism for LARQL's own native formats.
//!
//! [`RepresentationEncoder`] is a second, SEPARATE trait an external
//! codec MAY additionally implement to close that gap for itself,
//! without LARQL's core decode contract ever knowing the difference: a
//! codec implementing only `RepresentationCodec` is unaffected by this
//! module's existence, in every way — it does not implement this trait,
//! nothing in `RepresentationCodec` references it, and the built-in
//! registry (`CodecRegistry::builtin()`) neither registers anything
//! against it nor is capable of being asked to.
//!
//! Shape mirrors `decode_packed` (bind + validate + decode one payload)
//! rather than the lower-level `decode_rows` (range-aware, into a
//! caller-owned buffer): every concrete need so far encodes a whole
//! tensor at once, and a row-range-aware encode API would be new
//! surface with no proven caller — smallest surface that answers the
//! actual question, not the largest one that could conceivably be asked.

use super::error::CodecError;
use super::extent::RepresentationExtent;
use super::RepresentationCodec;

/// A codec that can also WRITE its own bytes, not just read them.
///
/// Implementing this trait changes nothing about `RepresentationCodec`
/// conformance — it is additive surface for a codec that chooses to
/// offer it, checked independently (see
/// [`EncoderRegistry`]) from the decode-side `CodecRegistry`.
pub trait RepresentationEncoder: RepresentationCodec {
    /// Encode canonical f32 `values` (row-major, `shape[0]` rows of
    /// `shape[1..].product()` elements each) into this codec's on-disk
    /// payload bytes at `extent`.
    ///
    /// The counterpart to `RepresentationCodec::decode_packed`: bytes
    /// this method returns, handed to `decode_packed` unmodified
    /// (through this SAME codec, or any codec admitting the same
    /// identity), must decode back to a value this codec's own
    /// `decode_rows` would produce — not necessarily `values` itself,
    /// since a lossy codec's encode is not required to be invertible to
    /// its input, only STABLE under its own round trip. What "stable"
    /// means for a given codec is that codec's own claim to prove, the
    /// same way `decode_rows`'s bit-exactness is each codec's own claim
    /// today.
    fn encode_packed(
        &self,
        values: &[f32],
        shape: &[usize],
        extent: RepresentationExtent,
        tensor: &str,
    ) -> Result<Vec<u8>, CodecError>;
}

/// Named the same way `dyn RepresentationCodec`'s own `Debug` impl is
/// (`mod.rs`): label + ABI, since a codec is what it decodes bytes as,
/// not its address.
impl std::fmt::Debug for dyn RepresentationEncoder + '_ {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let id = self.identity();
        write!(
            f,
            "encoder {} ({} r{})",
            self.encoding_label(),
            id.family,
            id.revision
        )
    }
}

/// The codecs this build can WRITE, addressed by label — the encode-side
/// analogue of [`super::CodecRegistry`], kept as its own type rather than
/// folded into `CodecRegistry` because a `Box<dyn RepresentationCodec>`
/// cannot be asked whether its concrete type also implements
/// `RepresentationEncoder` without either downcasting (which would need
/// every codec to carry `Any`, decode-side surface this trait has no
/// business adding) or duplicating storage. A codec that implements both
/// traits is registered in both registries by its own caller; nothing
/// here keeps them in sync automatically, the same way nothing keeps two
/// independently-constructed `CodecRegistry`s in sync today.
#[derive(Default)]
pub struct EncoderRegistry {
    encoders: Vec<Box<dyn RepresentationEncoder>>,
}

impl EncoderRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register an encoder, refusing a label already taken — same rule
    /// `CodecRegistry::register` applies to its own label axis (the
    /// family axis does not apply here: two encoder implementations of
    /// the same ABI family would still disagree on-disk, but that is a
    /// decode-side admission question `CodecRegistry::admit` already
    /// owns, not something this registry duplicates).
    pub fn register(mut self, encoder: Box<dyn RepresentationEncoder>) -> Result<Self, CodecError> {
        let label = encoder.encoding_label();
        if self.by_label(label).is_some() {
            return Err(CodecError::DuplicateLabel {
                label: label.into(),
            });
        }
        self.encoders.push(encoder);
        Ok(self)
    }

    pub fn encoders(&self) -> impl Iterator<Item = &dyn RepresentationEncoder> {
        self.encoders.iter().map(|c| c.as_ref())
    }

    pub fn by_label(&self, label: &str) -> Option<&dyn RepresentationEncoder> {
        self.encoders().find(|c| c.encoding_label() == label)
    }

    /// Registered labels, in registration order — what a refusal lists.
    pub fn labels(&self) -> Vec<String> {
        self.encoders()
            .map(|c| c.encoding_label().to_string())
            .collect()
    }

    /// The encoder `label` names, or a refusal naming the registered
    /// ones — mirrors `CodecRegistry::resolve`, with its own error
    /// variant so a caller can tell "nothing decodes this label" apart
    /// from "something decodes it, but nothing can WRITE it" (the
    /// common case: every built-in codec today is in the second
    /// category, since none of them implement `RepresentationEncoder`).
    pub fn resolve(
        &self,
        label: &str,
        tensor: &str,
    ) -> Result<&dyn RepresentationEncoder, CodecError> {
        self.by_label(label)
            .ok_or_else(|| CodecError::NoEncoderRegistered {
                tensor: tensor.into(),
                label: label.into(),
                registered: self.labels(),
            })
    }
}

#[cfg(test)]
mod tests {
    //! Two falsifiers this trait's existence must not fail:
    //!
    //! 1. A codec implementing only `RepresentationCodec` — every
    //!    built-in codec, today — is unaffected: `CodecRegistry::builtin()`
    //!    still admits and decodes exactly as it did before this module
    //!    existed. Checked directly, not inferred from "the diff didn't
    //!    touch `RepresentationCodec`" — a compiler fact, not a behavior
    //!    one, and this crate's own culture (rung 3's F8 falsifier) is to
    //!    check the seam, not trust that it held.
    //! 2. A codec implementing BOTH traits can write bytes through
    //!    `RepresentationEncoder::encode_packed` and read them back
    //!    through the EXISTING, unmodified `RepresentationCodec::
    //!    decode_packed` — bit-exact, entirely through this crate's own
    //!    code, no external tooling.

    use std::ops::Range;

    use super::*;
    use crate::format::vindex3::represent::codec::{
        AccessGranularity, CodecCapabilities, CodecOperands, ExtentCertificate, ResidencyProfile,
        RowGeometry, StreamRole, StreamSpec,
    };
    use crate::format::vindex3::represent::nvfp4_pack::CodecIdentity;

    #[test]
    fn adding_this_trait_leaves_the_builtin_decode_only_registry_unaffected() {
        let registry = super::super::registry::CodecRegistry::builtin();
        // Same labels, same order, as of this module landing — a
        // regression here means something about admission or
        // registration order moved, which this trait must not cause.
        assert_eq!(
            registry.labels(),
            vec![
                "BF16",
                "F16",
                "F32",
                "Q4_K",
                "Q6_K",
                "Q8_0",
                "NVFP4",
                "MXFP4",
                "BF16_ZLIB",
                "F32_PLANES",
                "VQ8_SHARED",
                "F8_E4M3",
                "Q5_K",
                "Q3_K",
            ]
        );
        // And it still decodes: one real value through the untouched F32
        // codec, proving `RepresentationCodec`'s own machinery (not just
        // its label list) is unaffected by this trait's existence.
        let f32_codec = registry.by_label("F32").unwrap();
        let bytes = 1.5f32.to_le_bytes();
        let out = f32_codec
            .decode_packed(&bytes, &[1], f32_codec.terminal_extent(), "t")
            .unwrap();
        assert_eq!(out, vec![1.5f32]);
    }

    /// A trivial double: one stream, raw little-endian f32, no scale, no
    /// grouping — implements BOTH traits, purely to exercise the seam
    /// between them without depending on any real external codec's own
    /// logic (an external crate's own test suite covers that codec
    /// itself).
    const VALUES: StreamSpec = StreamSpec {
        name: "values",
        role: StreamRole::Values,
    };
    const STREAMS: [StreamSpec; 1] = [VALUES];
    const LABEL: &str = "TEST_RAW_F32";
    const WIDTH: usize = std::mem::size_of::<f32>();

    struct RawF32Codec;

    impl RepresentationCodec for RawF32Codec {
        fn encoding_label(&self) -> &'static str {
            LABEL
        }
        fn identity(&self) -> CodecIdentity {
            CodecIdentity {
                family: "test-raw-f32".into(),
                revision: 1,
                group_elems: 1,
                element: "f32".into(),
                group_scale: "none".into(),
                tensor_scale: "none".into(),
                layout: "row-major-le".into(),
            }
        }
        fn streams(&self) -> &'static [StreamSpec] {
            &STREAMS
        }
        fn capabilities(&self) -> CodecCapabilities {
            CodecCapabilities {
                access: AccessGranularity::ElementRandom,
                group_elems: 1,
                row_align_elems: 1,
                physical_align_bytes: WIDTH,
            }
        }
        fn extents(&self) -> Vec<ExtentCertificate> {
            vec![ExtentCertificate::terminal(32.0)]
        }
        fn stored_bytes(
            &self,
            shape: &[usize],
            _: RepresentationExtent,
            tensor: &str,
        ) -> Result<u64, CodecError> {
            let g = RowGeometry::of(shape, LABEL, tensor)?;
            Ok((g.elements(LABEL, tensor)? * WIDTH) as u64)
        }
        fn validate(
            &self,
            operands: &CodecOperands<'_>,
            shape: &[usize],
            extent: RepresentationExtent,
            tensor: &str,
        ) -> Result<(), CodecError> {
            let need = self.stored_bytes(shape, extent, tensor)? as usize;
            operands.stream_of_len(VALUES, need, LABEL, tensor)?;
            Ok(())
        }
        fn decode_rows(
            &self,
            operands: &CodecOperands<'_>,
            shape: &[usize],
            rows: Range<usize>,
            _: RepresentationExtent,
            dst: &mut [f32],
            tensor: &str,
        ) -> Result<(), CodecError> {
            let g = RowGeometry::of(shape, LABEL, tensor)?;
            let need = g.elements(LABEL, tensor)? * WIDTH;
            let bytes = operands.stream_of_len(VALUES, need, LABEL, tensor)?;
            let region = &bytes[rows.start * g.k * WIDTH..rows.end * g.k * WIDTH];
            for (out, chunk) in dst.iter_mut().zip(region.chunks_exact(WIDTH)) {
                *out = f32::from_le_bytes(chunk.try_into().unwrap());
            }
            Ok(())
        }
        fn decode_residency(&self) -> ResidencyProfile {
            ResidencyProfile::DECODED_F32
        }
    }

    impl RepresentationEncoder for RawF32Codec {
        fn encode_packed(
            &self,
            values: &[f32],
            shape: &[usize],
            extent: RepresentationExtent,
            tensor: &str,
        ) -> Result<Vec<u8>, CodecError> {
            self.certificate_at(extent, tensor)?;
            let g = RowGeometry::of(shape, LABEL, tensor)?;
            let want = g.elements(LABEL, tensor)?;
            if values.len() != want {
                return Err(CodecError::Destination {
                    tensor: tensor.into(),
                    need: want,
                    have: values.len(),
                });
            }
            let mut out = Vec::with_capacity(values.len() * WIDTH);
            for v in values {
                out.extend_from_slice(&v.to_le_bytes());
            }
            Ok(out)
        }
    }

    #[test]
    fn a_codec_implementing_both_traits_writes_and_reads_back_bit_exact_through_larql_alone() {
        let codec_registry = super::super::registry::CodecRegistry::new()
            .register(Box::new(RawF32Codec))
            .unwrap();
        let encoder_registry = EncoderRegistry::new()
            .register(Box::new(RawF32Codec))
            .unwrap();

        let shape = [3usize, 4usize];
        let values: Vec<f32> = vec![
            1.0,
            -2.5,
            0.0,
            3.25,
            f32::MIN_POSITIVE,
            1e10,
            -1e-10,
            42.0,
            7.0,
            8.0,
            9.0,
            10.0,
        ];
        assert_eq!(values.len(), 12);

        let encoder = encoder_registry.resolve(LABEL, "t").unwrap();
        let extent = encoder.terminal_extent();
        let bytes = encoder.encode_packed(&values, &shape, extent, "t").unwrap();
        assert_eq!(bytes.len(), values.len() * WIDTH);

        let decoder = codec_registry.resolve(LABEL, "t").unwrap();
        let decoded = decoder.decode_packed(&bytes, &shape, extent, "t").unwrap();
        assert_eq!(
            decoded, values,
            "encode_packed then decode_packed must be bit-exact, entirely through LARQL's own \
             RepresentationCodec + RepresentationEncoder machinery"
        );
    }

    #[test]
    fn resolving_an_unregistered_label_names_what_is_registered() {
        let empty = EncoderRegistry::new();
        let err = empty.resolve("TEST_RAW_F32", "t").unwrap_err().to_string();
        assert!(err.contains("TEST_RAW_F32"), "{err}");

        let with_one = EncoderRegistry::new()
            .register(Box::new(RawF32Codec))
            .unwrap();
        let err = with_one
            .resolve("NOT_A_LABEL", "t")
            .unwrap_err()
            .to_string();
        assert!(err.contains("NOT_A_LABEL"), "{err}");
        assert!(err.contains("TEST_RAW_F32"), "{err}");
    }

    #[test]
    fn registering_a_second_encoder_under_a_taken_label_is_refused() {
        // CodecRegistry does not derive Debug either (registry.rs), so
        // `unwrap_err()` isn't available here — match instead.
        let with_one = EncoderRegistry::new()
            .register(Box::new(RawF32Codec))
            .unwrap();
        let Err(err) = with_one.register(Box::new(RawF32Codec)) else {
            panic!("a duplicate label must be refused");
        };
        assert!(err.to_string().contains("TEST_RAW_F32"), "{err}");
    }

    #[test]
    fn debug_names_the_label_and_the_identity_the_same_way_the_decode_side_does() {
        let registry = EncoderRegistry::new()
            .register(Box::new(RawF32Codec))
            .unwrap();
        let encoder = registry.by_label(LABEL).unwrap();
        let rendered = format!("{encoder:?}");
        assert!(rendered.contains(LABEL), "{rendered}");
        assert!(rendered.contains("test-raw-f32"), "{rendered}");
        assert!(rendered.contains('1'), "{rendered}");
    }

    #[test]
    fn encode_packed_refuses_a_values_slice_that_does_not_match_the_shape() {
        let registry = EncoderRegistry::new()
            .register(Box::new(RawF32Codec))
            .unwrap();
        let encoder = registry.by_label(LABEL).unwrap();
        let shape = [3usize, 4usize];
        let too_few = vec![1.0f32; 11];
        let err = encoder
            .encode_packed(&too_few, &shape, encoder.terminal_extent(), "t")
            .unwrap_err()
            .to_string();
        assert!(err.contains("11"), "{err}");
        assert!(err.contains("12"), "{err}");
    }

    #[test]
    fn the_double_answers_capabilities_and_decode_residency_the_same_as_any_real_codec() {
        // Exercised directly rather than only through the round-trip test:
        // these are plain accessors, but a codec that answered them
        // incorrectly would still pass every OTHER test here, so they earn
        // their own check.
        let codec = RawF32Codec;
        let caps = codec.capabilities();
        assert_eq!(caps.access, AccessGranularity::ElementRandom);
        assert_eq!(caps.group_elems, 1);
        assert_eq!(caps.row_align_elems, 1);
        assert_eq!(caps.physical_align_bytes, WIDTH);
        assert_eq!(codec.decode_residency(), ResidencyProfile::DECODED_F32);
    }
}
