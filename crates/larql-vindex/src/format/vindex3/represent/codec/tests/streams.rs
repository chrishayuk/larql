//! Named streams and auxiliary operands: binding, lookup, refusal.

use super::*;
use crate::format::vindex3::represent::codec::streams::ResolvedAuxiliary;

#[test]
fn a_stream_bound_twice_is_replaced_and_order_is_binding_order() {
    let a = [1u8, 2];
    let b = [3u8];
    let c = [4u8, 5, 6];
    let streams = NamedStreams::new()
        .with(VALUES, &a)
        .with(GROUP_SCALES, &b)
        .with(VALUES, &c);
    assert_eq!(streams.names(), ["group_scales", "values"]);
    assert_eq!(streams.get(VALUES), Some(&c[..]));
    assert_eq!(streams.get(GROUP_SCALES), Some(&b[..]));
    assert_eq!(streams.get(TENSOR_SCALE), None);
    assert_eq!(streams.len(), 2);
    assert!(!streams.is_empty());
    assert!(NamedStreams::new().is_empty());
}

#[test]
fn a_missing_stream_lists_what_was_bound_and_a_short_one_says_how_short() {
    let bytes = [0u8; 4];
    let operands = CodecOperands::from_streams(NamedStreams::single(VALUES, &bytes));
    let err = operands.stream(GROUP_SCALES, "X", TENSOR).unwrap_err();
    assert_eq!(
        err.to_string(),
        "tensor `layer.0.w`: `X` needs stream `group_scales`, which was not bound; bound: [values]"
    );
    assert_eq!(
        operands.stream_of_len(VALUES, 4, "X", TENSOR).unwrap(),
        &bytes[..]
    );
    let err = operands.stream_of_len(VALUES, 5, "X", TENSOR).unwrap_err();
    assert_eq!(
        err.to_string(),
        "tensor `layer.0.w`: `X` stream `values` is 4 bytes; 5 needed"
    );
}

/// A dependency reaches a decode as VALUES, by the name its owner
/// declared — never as a reference, which a codec could not resolve.
#[test]
fn auxiliary_operands_are_named_dependencies_and_empty_by_default() {
    let operands = CodecOperands::default();
    assert!(operands.auxiliaries.is_empty());
    assert_eq!(operands.auxiliaries.len(), 0);
    let shape = [4usize, 2];
    let values = [0.5f32, -1.0, 2.0, 3.5, 0.0, 1.5, -2.5, 7.0];
    let codebook = ResolvedAuxiliary {
        shape: &shape,
        values: &values,
    };
    let aux = AuxiliaryOperands::new().with("codebook", codebook);
    assert_eq!(aux.get("codebook"), Some(codebook));
    assert_eq!(aux.get("scales"), None);
    assert_eq!(aux.names(), vec!["codebook".to_string()]);
    assert_eq!(aux.len(), 1);
    assert!(!aux.is_empty());
    // A name nobody resolved is refused listing what was, rather than
    // answering with something plausible.
    let err = aux.require("palette", "X", TENSOR).unwrap_err();
    assert!(
        matches!(&err, CodecError::MissingAuxiliary { name, required, .. }
            if name == "palette" && required == &vec!["codebook".to_string()]),
        "{err}"
    );
    assert_eq!(aux.require("codebook", "X", TENSOR).unwrap(), codebook);
    let with_aux = CodecOperands {
        streams: NamedStreams::new(),
        auxiliaries: aux,
    };
    assert_eq!(with_aux.auxiliaries.len(), 1);
}

#[test]
fn stream_roles_are_what_the_declared_specs_say() {
    assert_eq!(VALUES.role, StreamRole::Values);
    assert_eq!(GROUP_SCALES.role, StreamRole::GroupScales);
    assert_eq!(TENSOR_SCALE.role, StreamRole::TensorScale);
}
