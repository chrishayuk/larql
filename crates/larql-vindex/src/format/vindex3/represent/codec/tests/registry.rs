//! The registry: two keys, three refusals, no duplicates.

use super::*;
use crate::format::vindex3::represent::nvfp4_pack::CodecIdentity;

#[test]
fn the_built_in_registry_carries_the_eleven_encodings_in_declaration_order() {
    assert_eq!(
        CodecRegistry::builtin().labels(),
        [
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
            "VQ8_SHARED"
        ]
    );
    assert_eq!(
        CodecRegistry::builtin().families(),
        [
            "BF16",
            "F16",
            "F32",
            "Q4_K",
            "Q6_K",
            "Q8_0",
            "nvfp4",
            "mxfp4",
            "BF16_ZLIB",
            "F32_PLANES",
            "VQ8_SHARED"
        ]
    );
}

#[test]
fn a_label_and_a_family_resolve_to_the_same_codec() {
    let registry = CodecRegistry::builtin();
    for codec in builtin() {
        let id = codec.identity();
        let by_family = registry.by_family(&id.family).unwrap();
        assert_eq!(by_family.encoding_label(), codec.encoding_label());
        // A diagnostic names the codec by label and ABI.
        assert_eq!(
            format!("{by_family:?}"),
            format!(
                "codec {} ({} r{})",
                codec.encoding_label(),
                id.family,
                id.revision
            )
        );
        assert_eq!(
            registry
                .resolve(codec.encoding_label(), TENSOR)
                .unwrap()
                .encoding_label(),
            codec.encoding_label()
        );
    }
}

#[test]
fn an_unregistered_label_is_refused_naming_every_registered_one() {
    let err = CodecRegistry::builtin()
        .resolve("Q5_K", TENSOR)
        .unwrap_err();
    assert_eq!(
        err,
        CodecError::UnknownEncoding {
            tensor: TENSOR.into(),
            label: "Q5_K".into(),
            registered: CodecRegistry::builtin().labels(),
        }
    );
    let text = err.to_string();
    assert!(
        text.contains("is not registered") && text.contains("Q6_K"),
        "{text}"
    );
}

#[test]
fn admit_refuses_a_future_revision_an_alien_family_and_disagreeing_geometry() {
    let registry = CodecRegistry::builtin();
    registry.admit(&CodecIdentity::nvfp4_v1()).unwrap();

    let mut future = CodecIdentity::nvfp4_v1();
    future.revision += 1;
    let err = registry.admit(&future).unwrap_err();
    assert!(
        matches!(err, CodecError::AbiRevision { found, implemented, .. } if found == implemented + 1)
    );
    let text = err.to_string();
    assert!(
        text.contains("another build") && text.contains("Recompile"),
        "{text}"
    );

    let mut alien = CodecIdentity::nvfp4_v1();
    alien.family = "vq-codebook".into();
    let err = registry.admit(&alien).unwrap_err();
    assert!(
        matches!(&err, CodecError::UnknownFamily { registered, .. } if registered == &registry.families())
    );
    assert!(err.to_string().contains("is not registered"));

    let mut bad = CodecIdentity::nvfp4_v1();
    bad.group_elems = 32;
    let err = registry.admit(&bad).unwrap_err();
    assert!(matches!(err, CodecError::AbiGeometry { revision: 1, .. }));
    assert!(err.to_string().contains("disagrees with its own revision"));
}

#[test]
fn the_identity_gate_the_index_calls_is_the_registry_s() {
    // `CodecIdentity::admit` is the entry the operand store uses; it must
    // be the same decision, worded the same way.
    let mut future = CodecIdentity::nvfp4_v1();
    future.revision += 1;
    let via_identity = future.admit().unwrap_err().to_string();
    let via_registry = CodecRegistry::builtin()
        .admit(&future)
        .unwrap_err()
        .to_string();
    assert!(
        via_identity.ends_with(&via_registry),
        "{via_identity:?} should carry {via_registry:?} intact"
    );
    for k in [Q4_K, Q6_K, Q8_0] {
        k.identity().admit().unwrap();
        let mut other = k.identity();
        other.family = "mxfp4".into();
        assert!(
            other.admit().is_err(),
            "{}: mxfp4's family does not admit a K-quant's geometry",
            k.encoding_label()
        );
    }
}

#[test]
fn registering_a_duplicate_label_or_family_is_refused() {
    let err = CodecRegistry::new()
        .register(Box::new(BF16))
        .and_then(|r| r.register(Box::new(BF16)))
        .err()
        .unwrap();
    assert_eq!(
        err,
        CodecError::DuplicateLabel {
            label: "BF16".into()
        }
    );

    let alias = Stub {
        label: "BF16-ALIAS",
        identity: BF16.identity(),
        streams: &[VALUES],
    };
    let err = CodecRegistry::new()
        .register(Box::new(BF16))
        .and_then(|r| r.register(Box::new(alias)))
        .err()
        .unwrap();
    assert_eq!(
        err,
        CodecError::DuplicateLabel {
            label: "BF16".into()
        }
    );
}

#[test]
fn a_foreign_codec_registers_and_gets_the_trait_defaults() {
    let mut identity = NVFP4.identity();
    identity.family = "stub".into();
    let stub = Stub {
        label: "STUB",
        identity,
        streams: &[VALUES],
    };
    let registry = CodecRegistry::new().register(Box::new(stub)).unwrap();
    let codec = registry.resolve("STUB", TENSOR).unwrap();
    assert!(
        codec.accelerations().is_empty(),
        "the default offers no acceleration"
    );
    let bound = codec.bind_packed(&[1, 2, 3], &[3], TENSOR).unwrap();
    assert_eq!(bound.names(), ["values"]);
    assert_eq!(
        codec
            .decode_packed(&[1, 2, 3], &[3], RepresentationExtent::BASE, TENSOR)
            .unwrap(),
        [0.0; 3]
    );
    assert_eq!(registry.codecs().count(), 1);
}
