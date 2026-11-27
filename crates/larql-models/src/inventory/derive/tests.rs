//! The storage-layout normalisation: fused split, input-major transpose,
//! buffer exclusion, and the byte transform the encoder runs.

use super::*;
use crate::config::{BufferKind, FusedProjection, NonParameterBuffer};

const FUSED: &[FusedProjection] = &[FusedProjection {
    module: "attn.c_attn",
    parts: &["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
}];
const LAYOUT: SourceLayout = SourceLayout {
    input_major: &["attn.c_attn", "mlp.c_fc"],
    fused: FUSED,
    buffers: &[NonParameterBuffer {
        suffix: "attn.bias",
        kind: BufferKind::CausalMask,
        reason: "causal mask",
    }],
};

fn fact(name: &str, shape: &[usize]) -> TensorFact {
    TensorFact {
        name: name.to_string(),
        dtype: "F32".to_string(),
        shape: shape.to_vec(),
        bytes: (shape.iter().product::<usize>() * 4) as u64,
        file: "model.safetensors".to_string(),
        derivation: None,
    }
}

fn inventory(tensors: Vec<TensorFact>) -> TensorInventory {
    TensorInventory {
        files: vec![],
        total_tensors: tensors.len(),
        total_bytes: 0,
        groups: vec![],
        tensors,
        excluded: vec![],
    }
}

#[test]
fn fused_input_major_weight_splits_into_transposed_parts() {
    let mut inv = inventory(vec![
        fact("h.0.attn.c_attn.weight", &[4, 6]),
        fact("h.0.attn.c_attn.bias", &[6]),
        fact("h.0.mlp.c_fc.weight", &[4, 8]),
        fact("h.0.mlp.c_fc.bias", &[8]),
        fact("h.0.attn.bias", &[1, 1, 3, 3]),
        fact("wte.weight", &[5, 4]),
    ]);
    apply_source_layout(&LAYOUT, &mut inv);
    let by = |n: &str| inv.tensors.iter().find(|t| t.name == n).unwrap();
    let q = by("h.0.self_attn.q_proj.weight");
    assert_eq!(q.shape, vec![2, 4]);
    let d = q.derivation.as_ref().unwrap();
    assert_eq!(
        d.slice,
        Some(AxisSlice {
            axis: 1,
            start: 0,
            len: 2
        })
    );
    assert!(d.transpose);
    let vb = by("h.0.self_attn.v_proj.bias");
    assert_eq!(vb.shape, vec![2]);
    assert_eq!(vb.derivation.as_ref().unwrap().slice.unwrap().start, 4);
    assert_eq!(by("h.0.mlp.c_fc.weight").shape, vec![8, 4]);
    // A bias of an input-major module is not transposed: it is 1-D.
    assert!(by("h.0.mlp.c_fc.bias").derivation.is_none());
    assert!(by("wte.weight").derivation.is_none());
    assert!(inv
        .tensors
        .iter()
        .all(|t| t.name != "h.0.attn.c_attn.weight"));
    assert_eq!(inv.excluded.len(), 1);
    assert_eq!(inv.excluded[0].name, "h.0.attn.bias");
    assert_eq!(inv.total_tensors, inv.tensors.len());
}

#[test]
fn a_shape_that_does_not_fit_the_declaration_is_left_as_stored() {
    let mut inv = inventory(vec![
        fact("h.0.attn.c_attn.weight", &[4, 7]),
        // Not a mask shape: stays an operand, and closure refuses it.
        fact("h.1.attn.bias", &[3]),
    ]);
    apply_source_layout(&LAYOUT, &mut inv);
    assert!(inv
        .tensors
        .iter()
        .any(|t| t.name == "h.0.attn.c_attn.weight"));
    assert!(inv.tensors.iter().any(|t| t.name == "h.1.attn.bias"));
    assert!(inv.excluded.is_empty());
}

#[test]
fn derive_bytes_slices_then_transposes() {
    // A [2, 6] input-major weight, values 0..12, row-major.
    let source: Vec<u8> = (0..12u32).flat_map(|v| (v as f32).to_le_bytes()).collect();
    let derivation = TensorDerivation {
        source: "w".into(),
        source_shape: vec![2, 6],
        slice: Some(AxisSlice {
            axis: 1,
            start: 2,
            len: 2,
        }),
        transpose: true,
    };
    let out = derive_bytes(&derivation, "F32", &source).unwrap();
    let values: Vec<f32> = out
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    // Columns 2 and 3 of [[0..6], [6..12]], transposed: [[2, 8], [3, 9]].
    assert_eq!(values, vec![2.0, 8.0, 3.0, 9.0]);

    let bias = TensorDerivation {
        source: "b".into(),
        source_shape: vec![6],
        slice: Some(AxisSlice {
            axis: 0,
            start: 4,
            len: 2,
        }),
        transpose: false,
    };
    let out = derive_bytes(&bias, "F32", &source[..24]).unwrap();
    assert_eq!(out, source[16..24].to_vec());
}

#[test]
fn derive_bytes_refuses_a_payload_of_the_wrong_length() {
    let derivation = TensorDerivation {
        source: "w".into(),
        source_shape: vec![2, 2],
        slice: None,
        transpose: true,
    };
    assert!(derive_bytes(&derivation, "F32", &[0u8; 12]).is_err());
    assert!(derive_bytes(&derivation, "Q4_K", &[0u8; 16]).is_err());
}

#[test]
fn the_canonical_layout_derives_nothing() {
    let mut inv = inventory(vec![fact("model.layers.0.mlp.up_proj.weight", &[8, 4])]);
    apply_source_layout(&SourceLayout::default(), &mut inv);
    assert!(inv.tensors[0].derivation.is_none());
}
