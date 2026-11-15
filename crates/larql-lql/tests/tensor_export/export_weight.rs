//! `EXPORT WEIGHT`: the exported values are the stored ones, exactly, and the sidecar describes the file.

use crate::npy;
use crate::support::*;

#[test]
fn a_bf16_matrix_exports_as_its_exact_f32_values() {
    let fx = Fixture::bf16();
    let mut s = fx.session();
    let out = fx.path("down.npy");
    let reply = run(&mut s, &export(MATRIX, "", &out, ""));
    assert!(reply[0].contains(CANONICAL), "{reply:?}");
    let file = npy::read(&out);
    assert_eq!(file.descr, "<f4");
    assert!(!file.fortran_order);
    assert_eq!(file.shape, vec![MATRIX_ROWS, MATRIX_COLS]);
    let stored = bf16_bits(&segment_bytes(&fx.root, CANONICAL, MATRIX));
    let exported = f32_values(&file.body);
    assert_eq!(exported.len(), stored.len());
    for (i, (v, bits)) in exported.iter().zip(&stored).enumerate() {
        assert_eq!(v.to_bits(), widen(*bits).to_bits(), "element {i}");
    }
}

#[test]
fn as_bits_writes_the_stored_bits() {
    let fx = Fixture::bf16();
    let mut s = fx.session();
    let out = fx.path("down_bits.npy");
    run(&mut s, &export(MATRIX, "", &out, " AS BITS"));
    let file = npy::read(&out);
    assert_eq!(file.descr, "<u2");
    assert_eq!(file.shape, vec![MATRIX_ROWS, MATRIX_COLS]);
    assert_eq!(file.body, segment_bytes(&fx.root, CANONICAL, MATRIX));
    assert_eq!(sidecar(&out)["exported_dtype"], "<u2");
}

#[test]
fn bits_and_values_agree() {
    let fx = Fixture::bf16();
    let mut s = fx.session();
    let (values, bits) = (fx.path("v.npy"), fx.path("b.npy"));
    run(&mut s, &export(GAIN, "", &values, ""));
    run(&mut s, &export(GAIN, "", &bits, " AS BITS"));
    let widened: Vec<u32> = bf16_bits(&npy::read(&bits).body)
        .into_iter()
        .map(|b| widen(b).to_bits())
        .collect();
    let exported: Vec<u32> = f32_values(&npy::read(&values).body)
        .into_iter()
        .map(f32::to_bits)
        .collect();
    assert_eq!(widened, exported);
}

#[test]
fn rows_and_columns_select_in_the_order_given() {
    let fx = Fixture::bf16();
    let mut s = fx.session();
    let out = fx.path("part.npy");
    let (rows, cols) = ([9usize, 0, 63], [17usize, 255, 3, 4]);
    run(
        &mut s,
        &export(MATRIX, " ROWS (9, 0, 63) COLUMNS (17, 255, 3, 4)", &out, ""),
    );
    let file = npy::read(&out);
    assert_eq!(file.shape, vec![rows.len(), cols.len()]);
    let stored = bf16_bits(&segment_bytes(&fx.root, CANONICAL, MATRIX));
    let expected: Vec<u32> = rows
        .iter()
        .flat_map(|r| cols.iter().map(move |c| r * MATRIX_COLS + c))
        .map(|i| widen(stored[i]).to_bits())
        .collect();
    let exported: Vec<u32> = f32_values(&file.body)
        .into_iter()
        .map(f32::to_bits)
        .collect();
    assert_eq!(exported, expected);
    let meta = sidecar(&out);
    assert_eq!(meta["rows"], serde_json::json!(rows));
    assert_eq!(meta["columns"], serde_json::json!(cols));
    assert_eq!(meta["exported_shape"], serde_json::json!([3, 4]));
    assert_eq!(meta["shape"], serde_json::json!([MATRIX_ROWS, MATRIX_COLS]));
}

#[test]
fn the_qualified_embedding_exports_where_the_bare_name_is_ambiguous() {
    let fx = Fixture::bf16();
    let mut s = fx.session();
    let bare = fx.path("bare.npy");
    let err = try_run(&mut s, &export(AMBIGUOUS, "", &bare, "")).unwrap_err();
    assert!(err.contains("<representation>::weight"), "{err}");
    assert!(err.contains(EMBEDDING), "{err}");
    assert!(!bare.exists(), "a refused export writes nothing");

    let out = fx.path("token_rows.npy");
    let name = format!("{EMBEDDING}::{AMBIGUOUS}");
    run(&mut s, &export(&name, " ROWS (100, 3)", &out, " AS BITS"));
    let file = npy::read(&out);
    assert_eq!(file.shape, vec![2, EMBEDDING_SHAPE[1]]);
    let stored = segment_bytes(&fx.root, EMBEDDING, AMBIGUOUS);
    let row_bytes = EMBEDDING_SHAPE[1] * BF16_BYTES;
    let expected = [
        &stored[100 * row_bytes..101 * row_bytes],
        &stored[3 * row_bytes..4 * row_bytes],
    ]
    .concat();
    assert_eq!(file.body, expected);
    let meta = sidecar(&out);
    assert_eq!(meta["representation"], EMBEDDING);
    assert_eq!(meta["tensor"], AMBIGUOUS);
}

#[test]
fn a_one_dimensional_gain_exports() {
    let fx = Fixture::bf16();
    let mut s = fx.session();
    let out = fx.path("gain.npy");
    run(&mut s, &export(GAIN, "", &out, ""));
    let file = npy::read(&out);
    assert_eq!(file.shape, vec![GAIN_LEN]);
    let stored = bf16_bits(&segment_bytes(&fx.root, CANONICAL, GAIN));
    let exported: Vec<u32> = f32_values(&file.body)
        .into_iter()
        .map(f32::to_bits)
        .collect();
    let expected: Vec<u32> = stored.iter().map(|b| widen(*b).to_bits()).collect();
    assert_eq!(exported, expected);
}

#[test]
fn a_derived_float_tensor_exports_by_its_qualified_name() {
    let fx = Fixture::bf16();
    let mut s = fx.session();
    let out = fx.path("derived_gain.npy");
    run(
        &mut s,
        &export(&format!("{DERIVED}::{GAIN}"), "", &out, " AS BITS"),
    );
    assert_eq!(npy::read(&out).body, segment_bytes(&fx.root, DERIVED, GAIN));
    let meta = sidecar(&out);
    assert_eq!(meta["representation"], DERIVED);
    assert_eq!(meta["canonical"], false);
}

#[test]
fn an_f32_tensor_exports_as_stored() {
    let fx = Fixture::f32();
    let mut s = fx.session();
    let (values, bits) = (fx.path("f32.npy"), fx.path("f32_bits.npy"));
    run(&mut s, &export(F32_MATRIX, "", &values, ""));
    run(&mut s, &export(F32_MATRIX, "", &bits, " AS BITS"));
    let stored = segment_bytes(&fx.root, F32_CANONICAL, F32_MATRIX);
    let v = npy::read(&values);
    assert_eq!(
        (v.descr.as_str(), v.shape.clone()),
        ("<f4", F32_SHAPE.to_vec())
    );
    assert_eq!(v.body, stored);
    let b = npy::read(&bits);
    assert_eq!(b.descr, "<u4");
    assert_eq!(b.body, stored);
    assert_eq!(sidecar(&values)["stored_dtype"], "F32");
}

#[test]
fn the_sidecar_says_where_the_numbers_came_from() {
    let fx = Fixture::bf16();
    let mut s = fx.session();
    let out = fx.path("meta.npy");
    let stmt = export(MATRIX, " COLUMNS (2)", &out, "");
    run(&mut s, &stmt);
    let meta = sidecar(&out);
    let index = std::fs::read(fx.root.join("index.json")).unwrap();
    let model: serde_json::Value = serde_json::from_slice(&index).unwrap();
    assert_eq!(meta["statement"], stmt);
    assert_eq!(
        meta["container"],
        std::fs::canonicalize(&fx.root)
            .unwrap()
            .display()
            .to_string()
    );
    assert_eq!(meta["index_sha256"], sha256_hex(&index));
    assert_eq!(meta["model"], model["model"]);
    assert_eq!(meta["reads"], "stored");
    assert_eq!(meta["representation"], CANONICAL);
    assert_eq!(meta["canonical"], true);
    assert_eq!(meta["tensor"], MATRIX);
    assert_eq!(meta["stored_dtype"], "BF16");
    assert_eq!(meta["exported_dtype"], "<f4");
    assert_eq!(meta["rows"], serde_json::Value::Null);
    assert_eq!(meta["columns"], serde_json::json!([2]));
    assert_eq!(meta["exported_shape"], serde_json::json!([MATRIX_ROWS, 1]));
    assert_eq!(
        meta["npy_sha256"],
        sha256_hex(&std::fs::read(&out).unwrap())
    );
    assert_eq!(meta["larql_version"], env!("CARGO_PKG_VERSION"));
    assert_eq!(meta["schema"], "larql-export-weight/v1");
}

#[test]
fn a_staged_edit_is_not_in_the_export() {
    let fx = Fixture::bf16();
    let mut s = fx.session();
    run(
        &mut s,
        &format!(
            "UPDATE WEIGHTS SET DELTA = 0.5 WHERE TENSOR = \"{MATRIX}\" AND ROW = 1 AND COL = 2;"
        ),
    );
    let out = fx.path("staged.npy");
    let reply = run(&mut s, &export(MATRIX, "", &out, " AS BITS"));
    assert!(
        reply.iter().any(|l| l.contains("stored weights")),
        "{reply:?}"
    );
    assert_eq!(
        npy::read(&out).body,
        segment_bytes(&fx.root, CANONICAL, MATRIX)
    );
    assert_eq!(sidecar(&out)["reads"], "stored");
}
