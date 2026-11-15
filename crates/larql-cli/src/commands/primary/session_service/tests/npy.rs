//! Operand arrays are read strictly: 2-D, C order, little-endian float64 or float32, exactly the body the
//! shape needs.

use base64::Engine;

use super::super::npy::{decode_operand, Operand};
use super::npy_v1;

fn b64(bytes: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

fn header(descr: &str, fortran: &str, shape: &str) -> String {
    format!("{{'descr': '{descr}', 'fortran_order': {fortran}, 'shape': {shape}, }}")
}

fn refused(file: &[u8], needle: &str) {
    let f = decode_operand("a", &b64(file)).unwrap_err();
    assert_eq!(f.kind, super::super::protocol::ErrorKind::InvalidRequest);
    assert!(f.reason.contains(needle), "{}", f.reason);
}

#[test]
fn float64_and_float32_operands_decode() {
    let body: Vec<u8> = [1.5f64, -2.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let d = decode_operand("a", &b64(&npy_v1(&header("<f8", "False", "(2, 1)"), &body))).unwrap();
    assert_eq!(d.values, Operand::F64(vec![1.5, -2.0]));
    assert_eq!(d.shape, [2, 1]);
    let body: Vec<u8> = [0.25f32, 4.0, 8.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let d = decode_operand("b", &b64(&npy_v1(&header("<f4", "False", "(1, 3)"), &body))).unwrap();
    assert_eq!(d.values, Operand::F32(vec![0.25, 4.0, 8.0]));
    assert_eq!(d.sha256.len(), 64);
}

#[test]
fn a_version_2_header_decodes() {
    let dict = header("<f8", "False", "(1, 1)");
    let mut file = b"\x93NUMPY\x02\x00".to_vec();
    file.extend_from_slice(&(dict.len() as u32 + 1).to_le_bytes());
    file.extend_from_slice(dict.as_bytes());
    file.push(b'\n');
    file.extend_from_slice(&3.0f64.to_le_bytes());
    assert_eq!(
        decode_operand("a", &b64(&file)).unwrap().values,
        Operand::F64(vec![3.0])
    );
}

#[test]
fn everything_else_is_refused() {
    let body = 1.0f64.to_le_bytes();
    let f = decode_operand("a", "%%%").unwrap_err();
    assert!(f.reason.contains("not base64"));
    refused(b"short", "not a .npy file");
    refused(b"NOTNUMPY\x01\x00\x00\x00", "not a .npy file");
    refused(b"\x93NUMPY\x09\x00\x00\x00", "unsupported .npy version 9");
    refused(b"\x93NUMPY\x01\x00", "truncated");
    refused(b"\x93NUMPY\x01\x00\xff\x00{}", "truncated");
    refused(
        &npy_v1(&header(">f8", "False", "(1, 1)"), &body),
        "dtype `>f8`",
    );
    refused(
        &npy_v1(&header("<i8", "False", "(1, 1)"), &body),
        "dtype `<i8`",
    );
    refused(
        &npy_v1(&header("<f8", "True", "(1, 1)"), &body),
        "Fortran-order",
    );
    refused(
        &npy_v1(&header("<f8", "Maybe", "(1, 1)"), &body),
        "fortran_order",
    );
    refused(
        &npy_v1(&header("<f8", "False", "(1,)"), &body),
        "must be 2-D",
    );
    refused(
        &npy_v1(&header("<f8", "False", "(1, x)"), &body),
        "`shape` has `x`",
    );
    refused(
        &npy_v1(&header("<f8", "False", "[1, 1]"), &body),
        "not a tuple",
    );
    refused(
        &npy_v1(&header("<f8", "False", "(2, 1)"), &body),
        "needs 16",
    );
    refused(
        &npy_v1("{'fortran_order': False, 'shape': (1, 1), }", &body),
        "no `descr`",
    );
    refused(
        &npy_v1(
            "{'descr': 3, 'fortran_order': False, 'shape': (1, 1), }",
            &body,
        ),
        "not a string",
    );
    refused(&npy_v1("not a dict", &body), "not a dict");
    let huge = format!("({}, {})", usize::MAX, 2);
    refused(&npy_v1(&header("<f8", "False", &huge), &body), "overflows");
    let mut bad_utf8 = npy_v1(&header("<f8", "False", "(1, 1)"), &body);
    bad_utf8[12] = 0xff;
    refused(&bad_utf8, "not text");
}
