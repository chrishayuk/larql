//! The emitter executes; these prove it executes exactly.

use super::*;
use crate::format::vindex3::represent::nvfp4_pack::{encode, PackLayout};
use larql_models::quant::half::f32_to_bf16;
use larql_models::quant::nvfp4::{dequantize_into, quantize};

/// The reference the permutation is measured against: the converter's
/// reshape([K, r, D]) → transpose → flatten, written as the naive
/// triple loop. Independent of `head_perm` on purpose.
fn reference_tiled<T: Clone>(rows: &[T], key_heads: usize, v_per_k: usize, d: usize) -> Vec<T> {
    let mut out = Vec::with_capacity(rows.len());
    for a in 0..v_per_k {
        for b in 0..key_heads {
            for e in 0..d {
                out.push(rows[(b * v_per_k + a) * d + e].clone());
            }
        }
    }
    out
}

fn plan(
    source_shape: Vec<u64>,
    representation: RepresentationKind,
    target_type: u32,
    layout: Vec<LayoutTransform>,
    value: Vec<ValueTransform>,
    scale: Option<&str>,
) -> LoweredTensorPlan {
    LoweredTensorPlan::new(
        "src",
        "t.weight",
        representation,
        target_type,
        source_shape,
        layout,
        value,
        scale.map(str::to_string),
    )
    .unwrap()
}

/// **Row blocks move exactly as the converter's reshape/transpose.**
/// Marker rows, an offset region that must not move, BF16 lattice.
#[test]
fn v_head_rows_move_as_the_reference_reshape_transpose() {
    const K: usize = 2;
    const R: usize = 3;
    const D: usize = 2;
    const OFFSET: usize = 4;
    const COLS: usize = 3;
    let rows = OFFSET + K * R * D;
    // Row i holds the marker value i in every column.
    let mut bytes = Vec::new();
    for i in 0..rows {
        for _ in 0..COLS {
            bytes.extend_from_slice(&f32_to_bf16(i as f32).to_le_bytes());
        }
    }
    let p = plan(
        vec![rows as u64, COLS as u64],
        RepresentationKind::Bf16,
        TYPE_BF16,
        vec![LayoutTransform::ReorderVRows {
            key_heads: K,
            v_per_k: R,
            head_dim: D,
            v_offset_rows: OFFSET,
        }],
        vec![],
        None,
    );
    let out = lower_unquantised(&p, bytes).unwrap();
    let marker = |row: usize| {
        let at = row * COLS * 2;
        bf16_to_f32(u16::from_le_bytes([out[at], out[at + 1]])) as usize
    };
    // The offset region did not move.
    for i in 0..OFFSET {
        assert_eq!(marker(i), i, "row {i} is outside the V region");
    }
    // The V region is in tiled order, exactly as the reference says.
    let grouped: Vec<usize> = (0..K * R * D).collect();
    let tiled = reference_tiled(&grouped, K, R, D);
    for (j, want) in tiled.iter().enumerate() {
        assert_eq!(marker(OFFSET + j), OFFSET + want, "V row {j}");
    }
}

/// **Column blocks move per row, whole heads at a time.** F32 lattice,
/// marker = row·1000 + column.
#[test]
fn v_head_columns_move_as_whole_heads_within_every_row() {
    const K: usize = 2;
    const R: usize = 2;
    const ROWS: usize = 3;
    const HEAD_COLS: usize = 4;
    const COLS: usize = K * R * HEAD_COLS;
    let mut bytes = Vec::new();
    for r in 0..ROWS {
        for c in 0..COLS {
            bytes.extend_from_slice(&((r * 1000 + c) as f32).to_le_bytes());
        }
    }
    let p = plan(
        vec![ROWS as u64, COLS as u64],
        RepresentationKind::F32,
        TYPE_F32,
        vec![LayoutTransform::ReorderVColumnsByGroups {
            key_heads: K,
            v_per_k: R,
            groups_per_head: HEAD_COLS / 16, // not used on a raw lattice
        }],
        vec![],
        None,
    );
    let out = lower_unquantised(&p, bytes).unwrap();
    let read = |r: usize, c: usize| {
        let at = (r * COLS + c) * 4;
        f32::from_le_bytes([out[at], out[at + 1], out[at + 2], out[at + 3]]) as usize
    };
    let cols: Vec<usize> = (0..COLS).collect();
    let want = reference_tiled(&cols, K, R, HEAD_COLS);
    for r in 0..ROWS {
        for c in 0..COLS {
            assert_eq!(read(r, c), r * 1000 + want[c], "row {r} col {c}");
        }
    }
}

/// Decode a GGML NVFP4 stream by its published layout — written here
/// from the spec, sharing nothing with `repack_nvfp4`.
fn decode_ggml(blocks: &[u8], tensor_scale: f32, n: usize) -> Vec<f32> {
    let e2m1 = |c: u8| -> f32 {
        let mag = match c & 0x7 {
            0 => 0.0,
            1 => 0.5,
            2 => 1.0,
            3 => 1.5,
            4 => 2.0,
            5 => 3.0,
            6 => 4.0,
            _ => 6.0,
        };
        if c & 0x8 != 0 {
            -mag
        } else {
            mag
        }
    };
    let ue4m3 = |b: u8| -> f32 {
        let e = (b >> 3) & 0xF;
        let m = (b & 0x7) as f32;
        if e == 0 {
            m / 8.0 * 2f32.powi(-6)
        } else {
            (1.0 + m / 8.0) * 2f32.powi(e as i32 - 7)
        }
    };
    let mut out = vec![0.0f32; n];
    for (block_i, block) in blocks.chunks_exact(36).enumerate() {
        for g in 0..4 {
            let s = ue4m3(block[g]) * tensor_scale;
            for j in 0..8 {
                let byte = block[4 + g * 8 + j];
                out[block_i * 64 + g * 16 + j] = e2m1(byte & 0x0F) * s;
                out[block_i * 64 + g * 16 + j + 8] = e2m1(byte >> 4) * s;
            }
        }
    }
    out
}

/// **The quantised reorder is the same permutation the floats
/// undergo** — decoded element by element on both sides, exactly. The
/// strongest form: if the emitter moved a code without its scale, or a
/// scale without its codes, no byte comparison would need to notice,
/// but the decoded values could not agree.
#[test]
fn nvfp4_row_reorder_decodes_to_the_permuted_reference_exactly() {
    const K: usize = 2;
    const R: usize = 3;
    const D: usize = 2;
    const OFFSET: usize = 4;
    const COLS: usize = 64;
    let rows = OFFSET + K * R * D;
    // Distinct values per row and per group so a moved half is visible.
    let values: Vec<f32> = (0..rows * COLS)
        .map(|i| ((i % 97) as f32 - 48.0) / 7.0)
        .collect();
    let matrix = quantize(&values, rows, COLS).unwrap();
    let layout = PackLayout::derive(&[rows, COLS], "t").unwrap();
    let payload = encode(&matrix, &layout, "t").unwrap();

    let p = plan(
        vec![rows as u64, COLS as u64],
        RepresentationKind::Nvfp4,
        TYPE_NVFP4,
        vec![LayoutTransform::ReorderVRows {
            key_heads: K,
            v_per_k: R,
            head_dim: D,
            v_offset_rows: OFFSET,
        }],
        vec![],
        Some("t.scale"),
    );
    let (blocks, scale) = lower_quantised(&p, &payload).unwrap();
    assert_eq!(
        scale, matrix.tensor_scale,
        "the tensor scale is carried, not recomputed"
    );

    // Reference: decode the source pack, permute the FLOAT rows.
    let mut reference = vec![0.0f32; rows * COLS];
    dequantize_into(&matrix, rows, COLS, &mut reference).unwrap();
    let head_rows: Vec<Vec<f32>> = reference[OFFSET * COLS..]
        .chunks_exact(COLS)
        .map(|r| r.to_vec())
        .collect();
    let tiled = reference_tiled(&head_rows, K, R, D);
    let mut want = reference[..OFFSET * COLS].to_vec();
    for row in tiled {
        want.extend_from_slice(&row);
    }

    let got = decode_ggml(&blocks, scale, rows * COLS);
    assert_eq!(
        got, want,
        "same codes, same scales, permuted rows — exactly"
    );
}

/// The column variant: whole heads of intact 16-element groups.
#[test]
fn nvfp4_column_reorder_decodes_to_the_permuted_reference_exactly() {
    const K: usize = 2;
    const R: usize = 2;
    const ROWS: usize = 3;
    const HEAD_COLS: usize = 32;
    const COLS: usize = K * R * HEAD_COLS; // 128
    let values: Vec<f32> = (0..ROWS * COLS)
        .map(|i| ((i % 89) as f32 - 44.0) / 5.0)
        .collect();
    let matrix = quantize(&values, ROWS, COLS).unwrap();
    let layout = PackLayout::derive(&[ROWS, COLS], "t").unwrap();
    let payload = encode(&matrix, &layout, "t").unwrap();

    let p = plan(
        vec![ROWS as u64, COLS as u64],
        RepresentationKind::Nvfp4,
        TYPE_NVFP4,
        vec![LayoutTransform::ReorderVColumnsByGroups {
            key_heads: K,
            v_per_k: R,
            groups_per_head: HEAD_COLS / 16,
        }],
        vec![],
        Some("t.scale"),
    );
    let (blocks, scale) = lower_quantised(&p, &payload).unwrap();

    let mut reference = vec![0.0f32; ROWS * COLS];
    dequantize_into(&matrix, ROWS, COLS, &mut reference).unwrap();
    let cols: Vec<usize> = (0..COLS).collect();
    let perm = reference_tiled(&cols, K, R, HEAD_COLS);
    let mut want = vec![0.0f32; ROWS * COLS];
    for r in 0..ROWS {
        for c in 0..COLS {
            want[r * COLS + c] = reference[r * COLS + perm[c]];
        }
    }
    assert_eq!(decode_ggml(&blocks, scale, ROWS * COLS), want);

    // And a group accounting that disagrees with the tensor refuses.
    let bad = plan(
        vec![ROWS as u64, COLS as u64],
        RepresentationKind::Nvfp4,
        TYPE_NVFP4,
        vec![LayoutTransform::ReorderVColumnsByGroups {
            key_heads: K,
            v_per_k: R,
            groups_per_head: 1,
        }],
        vec![],
        Some("t.scale"),
    );
    assert!(lower_quantised(&bad, &payload)
        .unwrap_err()
        .to_string()
        .contains("group accounting"));
}

/// **Value arithmetic computes in f32 and stores its exact result.**
#[test]
fn value_transforms_store_the_exact_f32_result() {
    let a_log: Vec<f32> = vec![0.5, -1.25, 2.0, -0.0078125];
    let bytes: Vec<u8> = a_log
        .iter()
        .flat_map(|v| f32_to_bf16(*v).to_le_bytes())
        .collect();
    let p = plan(
        vec![4],
        RepresentationKind::Bf16,
        TYPE_F32,
        vec![],
        vec![ValueTransform::MaterializeLogDecay],
        None,
    );
    let out = lower_unquantised(&p, bytes).unwrap();
    for (i, v) in a_log.iter().enumerate() {
        let want = -(bf16_to_f32(f32_to_bf16(*v))).exp();
        let got = f32::from_le_bytes(out[i * 4..i * 4 + 4].try_into().unwrap());
        assert_eq!(got.to_bits(), want.to_bits(), "element {i} is bit-exact");
    }

    // The norm offset folds the declared number.
    let w = vec![0.25f32, -1.0, 0.0];
    let bytes: Vec<u8> = w
        .iter()
        .flat_map(|v| f32_to_bf16(*v).to_le_bytes())
        .collect();
    let p = plan(
        vec![3],
        RepresentationKind::Bf16,
        TYPE_F32,
        vec![],
        vec![ValueTransform::ApplyWeightOffset(1.0)],
        None,
    );
    let out = lower_unquantised(&p, bytes).unwrap();
    for (i, v) in w.iter().enumerate() {
        let got = f32::from_le_bytes(out[i * 4..i * 4 + 4].try_into().unwrap());
        assert_eq!(got, v + 1.0);
    }

    // A lowering nobody defined refuses by name rather than inventing.
    let p = plan(
        vec![2, 2],
        RepresentationKind::Bf16,
        TYPE_BF16,
        vec![],
        vec![ValueTransform::ApplyWeightOffset(1.0)],
        None,
    );
    let err = lower_unquantised(&p, vec![0u8; 8]).unwrap_err().to_string();
    assert!(err.contains("no such lowering"), "{err}");
}

/// **The round trip, through the independent reader.** Emit a file
/// from plans and resolved metadata, parse it back with `GgufFile` —
/// the reader written for foreign GGUFs — and verify. Then hand the
/// verifier wrong expectations and require it to name every defect.
#[test]
fn an_emitted_file_verifies_exactly_and_wrong_expectations_are_named() {
    use super::super::metadata::{MetaKey, MetaValue};

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("out.gguf");

    let metadata = metadata_to_gguf(&[
        MetaKey {
            key: "general.architecture".into(),
            value: MetaValue::Str("qwen35".into()),
            derived_from: "target constant",
        },
        MetaKey {
            key: "qwen35.block_count".into(),
            value: MetaValue::U32(2),
            derived_from: "component.num_layers",
        },
        MetaKey {
            key: "qwen35.rope.dimension_sections".into(),
            value: MetaValue::ArrU32(vec![11, 11, 10, 0]),
            derived_from: "layer position.section",
        },
    ]);

    // Three plans, three encodings: F32 passthrough, BF16 matrix, NVFP4.
    let f32_data: Vec<u8> = [1.0f32, 2.0, 3.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let bf16_data: Vec<u8> = (0..6)
        .flat_map(|i| f32_to_bf16(i as f32).to_le_bytes())
        .collect();
    let nv_values: Vec<f32> = (0..2 * 64).map(|i| (i as f32 - 64.0) / 9.0).collect();
    let nv = quantize(&nv_values, 2, 64).unwrap();
    let nv_payload = encode(&nv, &PackLayout::derive(&[2, 64], "nv").unwrap(), "nv").unwrap();

    let plans = vec![
        LoweredTensorPlan::new(
            "a",
            "output.weight",
            RepresentationKind::F32,
            TYPE_F32,
            vec![3],
            vec![],
            vec![],
            None,
        )
        .unwrap(),
        LoweredTensorPlan::new(
            "b",
            "blk.0.ffn_up.weight",
            RepresentationKind::Bf16,
            TYPE_BF16,
            vec![2, 3],
            vec![],
            vec![],
            None,
        )
        .unwrap(),
        LoweredTensorPlan::new(
            "c",
            "blk.0.ffn_down.weight",
            RepresentationKind::Nvfp4,
            TYPE_NVFP4,
            vec![2, 64],
            vec![],
            vec![],
            Some("blk.0.ffn_down.scale".into()),
        )
        .unwrap(),
    ];

    let mut open = |source: &str| -> std::io::Result<Box<dyn std::io::Read>> {
        let bytes = match source {
            "a" => f32_data.clone(),
            "b" => bf16_data.clone(),
            "c" => nv_payload.clone(),
            other => panic!("no source `{other}`"),
        };
        Ok(Box::new(std::io::Cursor::new(bytes)))
    };
    let report = emit_gguf(&metadata, &plans, &mut open, &path).unwrap();
    assert_eq!(report.tensors, 3);
    assert_eq!(report.scale_siblings, 1);

    // The independent reader agrees with everything.
    let ok = verify_emitted(&path, &metadata, &plans, &["output.weight"]).unwrap();
    assert_eq!(ok.tensors, 4, "three planned tensors and one scale sibling");
    assert_eq!(ok.nvfp4_tensors, 1);
    assert_eq!(ok.metadata_keys, 3);

    // The scale sibling holds the pack's own tensor scale, exactly.
    let gguf = larql_models::loading::gguf::GgufFile::open(&path).unwrap();
    let scale_info = gguf
        .tensor_infos
        .iter()
        .find(|t| t.name() == "blk.0.ffn_down.scale")
        .unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let at = (gguf.data_offset + scale_info.offset()) as usize;
    let scale = f32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
    assert_eq!(scale, nv.tensor_scale);

    // Wrong expectations are named, all of them.
    let mut foreign = plans.clone();
    foreign[1].target_shape = vec![3, 3];
    foreign.pop();
    let wrong = verify_emitted(&path, &metadata, &foreign, &["missing.weight"]).unwrap_err();
    let text = wrong.join("\n");
    assert!(
        text.contains("blk.0.ffn_up.weight") && text.contains("[3, 3]"),
        "{text}"
    );
    assert!(
        text.contains("blk.0.ffn_down.weight"),
        "unplanned tensor named: {text}"
    );
    assert!(text.contains("missing.weight"), "{text}");

    let mut altered = metadata.clone();
    altered[1].1 = GgufValue::U32(64);
    let wrong = verify_emitted(&path, &altered, &plans, &[]).unwrap_err();
    assert!(
        wrong.iter().any(|w| w.contains("qwen35.block_count")),
        "{wrong:?}"
    );
}

/// A plan that declares one length and produces another is refused at
/// the byte it happens — the offsets are already committed.
#[test]
fn a_short_write_is_refused_loudly() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("short.gguf");
    let plans = vec![LoweredTensorPlan::new(
        "a",
        "t.weight",
        RepresentationKind::F32,
        TYPE_F32,
        vec![4],
        vec![],
        vec![],
        None,
    )
    .unwrap()];
    let mut open = |_: &str| -> std::io::Result<Box<dyn std::io::Read>> {
        Ok(Box::new(std::io::Cursor::new(vec![0u8; 8]))) // 8 bytes, plan says 16
    };
    let err = emit_gguf(&[], &plans, &mut open, &path).unwrap_err();
    assert!(err.to_string().contains("declared"), "{err}");
}
