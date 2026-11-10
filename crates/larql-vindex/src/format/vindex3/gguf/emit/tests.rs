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
        for (c, w) in want.iter().enumerate() {
            assert_eq!(read(r, c), r * 1000 + w, "row {r} col {c}");
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
    let a_log: [f32; 4] = [0.5, -1.25, 2.0, -0.0078125];
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
    let w = [0.25f32, -1.0, 0.0];
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

fn f32_bytes(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
    values.into_iter().flat_map(f32::to_le_bytes).collect()
}

/// A V-head reorder whose region runs past the payload is the plan and
/// the tensor disagreeing about the tensor's extent. Refused before a
/// byte moves, on both lattices.
#[test]
fn a_reorder_reaching_past_the_payload_is_refused_on_both_lattices() {
    const K: usize = 2;
    const R: usize = 2;
    const D: usize = 2;
    const COLS: usize = 64;
    let rows = K * R * D;
    let reorder = |v_offset_rows| LayoutTransform::ReorderVRows {
        key_heads: K,
        v_per_k: R,
        head_dim: D,
        v_offset_rows,
    };

    // Unquantised: the plan's shape is right, the payload is one row short.
    let p = plan(
        vec![rows as u64, COLS as u64],
        RepresentationKind::F32,
        TYPE_F32,
        vec![reorder(0)],
        vec![],
        None,
    );
    let short = vec![0u8; (rows - 1) * COLS * 4];
    let err = lower_unquantised(&p, short).unwrap_err().to_string();
    assert!(err.contains("disagree about its extent"), "{err}");

    // Quantised: the pack is whole, the plan's offset pushes the V
    // region past its last row.
    let values: Vec<f32> = (0..rows * COLS).map(|i| i as f32 / 64.0).collect();
    let matrix = quantize(&values, rows, COLS).unwrap();
    let payload = encode(
        &matrix,
        &PackLayout::derive(&[rows, COLS], "t").unwrap(),
        "t",
    )
    .unwrap();
    let p = plan(
        vec![rows as u64, COLS as u64],
        RepresentationKind::Nvfp4,
        TYPE_NVFP4,
        vec![reorder(1)],
        vec![],
        Some("t.scale"),
    );
    let err = lower_quantised(&p, &payload).unwrap_err().to_string();
    assert!(err.contains("disagree about its extent"), "{err}");
}

/// A column reorder needs whole V heads in every row. A width that
/// does not divide is refused with the arithmetic, not truncated.
#[test]
fn a_column_reorder_over_a_width_that_is_not_whole_heads_is_refused() {
    const K: usize = 2;
    const R: usize = 2;
    const COLS: usize = K * R + 1;
    let p = plan(
        vec![1, COLS as u64],
        RepresentationKind::F32,
        TYPE_F32,
        vec![LayoutTransform::ReorderVColumnsByGroups {
            key_heads: K,
            v_per_k: R,
            groups_per_head: 1,
        }],
        vec![],
        None,
    );
    let err = lower_unquantised(&p, f32_bytes((0..COLS).map(|c| c as f32)))
        .unwrap_err()
        .to_string();
    assert!(err.contains(&format!("{COLS} columns")), "{err}");
    assert!(err.contains(&format!("{} V heads", K * R)), "{err}");
}

/// **A squeeze moves no byte.** The conv kernel's singleton channel
/// axis disappears from the shape; the payload is the source payload,
/// re-encoded F32 → F32 exactly.
#[test]
fn a_singleton_squeeze_changes_the_shape_and_not_one_byte() {
    const CHANNELS: usize = 3;
    const KERNEL: usize = 4;
    let source = f32_bytes((0..CHANNELS * KERNEL).map(|i| i as f32 - 5.5));
    let p = plan(
        vec![CHANNELS as u64, 1, KERNEL as u64],
        RepresentationKind::F32,
        TYPE_F32,
        vec![LayoutTransform::SqueezeSingletonAxis { axis: 1 }],
        vec![],
        None,
    );
    assert_eq!(p.target_shape, vec![CHANNELS as u64, KERNEL as u64]);
    assert_eq!(lower_unquantised(&p, source.clone()).unwrap(), source);
}

/// Each path refuses what it was not built to carry: an NVFP4 source on
/// the unquantised path, a non-matrix pack, and a squeeze on a pack.
#[test]
fn each_lowering_path_refuses_a_representation_it_does_not_define() {
    let p = plan(
        vec![2, 64],
        RepresentationKind::Nvfp4,
        TYPE_NVFP4,
        vec![],
        vec![],
        Some("t.scale"),
    );
    let err = lower_unquantised(&p, vec![]).unwrap_err().to_string();
    assert!(
        err.contains("unquantised path with an NVFP4 source"),
        "{err}"
    );

    let p = plan(
        vec![64],
        RepresentationKind::Nvfp4,
        TYPE_NVFP4,
        vec![],
        vec![],
        Some("t.scale"),
    );
    let err = lower_quantised(&p, &[]).unwrap_err().to_string();
    assert!(err.contains("defined for matrices"), "{err}");

    const COLS: usize = 64;
    let values: Vec<f32> = (0..COLS).map(|i| i as f32 / 8.0).collect();
    let matrix = quantize(&values, 1, COLS).unwrap();
    let payload = encode(&matrix, &PackLayout::derive(&[1, COLS], "t").unwrap(), "t").unwrap();
    let p = plan(
        vec![1, COLS as u64],
        RepresentationKind::Nvfp4,
        TYPE_NVFP4,
        vec![LayoutTransform::SqueezeSingletonAxis { axis: 0 }],
        vec![],
        Some("t.scale"),
    );
    let err = lower_quantised(&p, &payload).unwrap_err().to_string();
    assert!(err.contains("squeeze on a quantised pack"), "{err}");
}

/// The metadata re-spelling is one-to-one, every variant: nothing
/// added, renamed, or defaulted.
#[test]
fn metadata_is_respelled_variant_for_variant() {
    use super::super::metadata::{MetaKey, MetaValue};
    let table = [
        ("s", MetaValue::Str("qwen35".into())),
        ("u", MetaValue::U32(7)),
        ("f", MetaValue::F32(1e-6)),
        ("a", MetaValue::ArrU32(vec![1, 2])),
    ];
    let keys: Vec<MetaKey> = table
        .iter()
        .map(|(k, v)| MetaKey {
            key: k.to_string(),
            value: v.clone(),
            derived_from: "test",
        })
        .collect();
    assert_eq!(
        metadata_to_gguf(&keys),
        vec![
            ("s".to_string(), GgufValue::String("qwen35".into())),
            ("u".to_string(), GgufValue::U32(7)),
            ("f".to_string(), GgufValue::F32(1e-6)),
            (
                "a".to_string(),
                GgufValue::Array(vec![GgufValue::U32(1), GgufValue::U32(2)])
            ),
        ]
    );
}

/// A ggml type the emitter has no length rule for is refused before
/// the file's offsets are committed — and nothing is written.
#[test]
fn a_target_type_the_emitter_cannot_express_is_refused_before_writing() {
    const UNDEFINED_TYPE: u32 = u32::MAX;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("x.gguf");
    let plans = vec![plan(
        vec![4],
        RepresentationKind::F32,
        UNDEFINED_TYPE,
        vec![],
        vec![],
        None,
    )];
    let mut open = |_: &str| -> std::io::Result<Box<dyn std::io::Read>> {
        panic!("no payload is read for a plan that cannot be laid out")
    };
    let err = emit_gguf(&[], &plans, &mut open, &path)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains(&format!("ggml type {UNDEFINED_TYPE}")),
        "{err}"
    );
    assert!(!path.exists());
}

/// **Arithmetic goes through the emitter, not around it.** A plan with
/// a value transform is buffered and lowered — the file holds the
/// transformed numbers, re-read from disk. And a plan that names a
/// scale sibling its source cannot supply (only NVFP4 packs carry a
/// tensor scale) is refused, naming the plan, rather than writing a
/// zero scale.
#[test]
fn transformed_plans_are_lowered_in_the_file_and_an_unsuppliable_sibling_is_refused() {
    const OFFSET: f32 = 1.0;
    let source = [0.25f32, -1.0, 0.5];
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("norm.gguf");
    let norm = plan(
        vec![source.len() as u64],
        RepresentationKind::F32,
        TYPE_F32,
        vec![],
        vec![ValueTransform::ApplyWeightOffset(OFFSET)],
        None,
    );
    let mut open = |_: &str| -> std::io::Result<Box<dyn std::io::Read>> {
        Ok(Box::new(std::io::Cursor::new(f32_bytes(source))))
    };
    emit_gguf(&[], std::slice::from_ref(&norm), &mut open, &path).unwrap();
    let gguf = GgufFile::open(&path).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let at = (gguf.data_offset + gguf.tensor_infos[0].offset()) as usize;
    let got: Vec<f32> = bytes[at..at + source.len() * 4]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(got, source.map(|v| v + OFFSET));

    let mut sibling = norm;
    sibling.scale_tensor = Some("t.scale".into());
    let err = emit_gguf(&[], &[sibling], &mut open, &dir.path().join("s.gguf"))
        .unwrap_err()
        .to_string();
    assert!(err.contains("names a scale sibling"), "{err}");
    assert!(
        err.contains("t.weight"),
        "the refusal names the plan: {err}"
    );
}

/// The verifier's remaining defect classes, each named: an unreadable
/// file, metadata missing from or extra to the file, a type the file
/// disagrees on, a planned tensor the file lacks, and an NVFP4 plan
/// whose scale sibling is absent or was never planned.
#[test]
fn every_class_of_verify_defect_is_named() {
    let dir = tempfile::tempdir().unwrap();

    let not_gguf = dir.path().join("not.gguf");
    std::fs::write(&not_gguf, b"not a gguf").unwrap();
    let wrong = verify_emitted(&not_gguf, &[], &[], &[]).unwrap_err();
    assert!(wrong[0].starts_with("parse:"), "{wrong:?}");

    let path = dir.path().join("one.gguf");
    let metadata = vec![("k".to_string(), GgufValue::U32(1))];
    let f32_plan = plan(
        vec![2],
        RepresentationKind::F32,
        TYPE_F32,
        vec![],
        vec![],
        None,
    );
    let mut open = |_: &str| -> std::io::Result<Box<dyn std::io::Read>> {
        Ok(Box::new(std::io::Cursor::new(f32_bytes([1.0, 2.0]))))
    };
    emit_gguf(&metadata, std::slice::from_ref(&f32_plan), &mut open, &path).unwrap();

    // Metadata both directions: one resolved key absent, one file key
    // never resolved.
    let other = vec![("other".to_string(), GgufValue::U32(1))];
    let text = verify_emitted(&path, &other, std::slice::from_ref(&f32_plan), &[])
        .unwrap_err()
        .join("\n");
    assert!(text.contains("`other` is missing from the file"), "{text}");
    assert!(
        text.contains("`k` appears in the file but was never resolved"),
        "{text}"
    );

    // A type disagreement, and a planned tensor the file never got.
    let mut retyped = f32_plan.clone();
    retyped.target_type = TYPE_BF16;
    let mut absent = f32_plan.clone();
    absent.target_name = "absent.weight".into();
    let text = verify_emitted(&path, &metadata, &[retyped, absent], &[])
        .unwrap_err()
        .join("\n");
    assert!(
        text.contains(&format!(
            "ggml type {TYPE_F32} in the file but the plan says {TYPE_BF16}"
        )),
        "{text}"
    );
    assert!(
        text.contains("planned tensor `absent.weight` is missing"),
        "{text}"
    );

    // NVFP4 travels with its sibling: one whose sibling the file lacks,
    // one planned without any.
    let mut lost = f32_plan.clone();
    lost.target_name = "lost.weight".into();
    lost.target_type = TYPE_NVFP4;
    lost.scale_tensor = Some("lost.scale".into());
    let mut unscaled = f32_plan;
    unscaled.target_type = TYPE_NVFP4;
    let text = verify_emitted(&path, &metadata, &[unscaled, lost], &[])
        .unwrap_err()
        .join("\n");
    assert!(
        text.contains("NVFP4 `lost.weight` lost its sibling `lost.scale`"),
        "{text}"
    );
    assert!(
        text.contains("NVFP4 `t.weight` was planned without a scale sibling"),
        "{text}"
    );
}
