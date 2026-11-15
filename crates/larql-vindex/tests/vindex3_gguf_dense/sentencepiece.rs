//! The `tokenizer.model` piece reader, against hand-encoded protobuf.

use larql_vindex::format::vindex3::gguf::dense::sentencepiece::{parse_pieces, Piece, PieceKind};

fn varint(mut v: u64, out: &mut Vec<u8>) {
    loop {
        let b = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(b);
            return;
        }
        out.push(b | 0x80);
    }
}

fn len_field(field: u64, body: &[u8], out: &mut Vec<u8>) {
    varint(field << 3 | 2, out);
    varint(body.len() as u64, out);
    out.extend_from_slice(body);
}

/// A `ModelProto` holding `(text, score, type)` pieces, plus a trainer
/// spec and a stray varint field the reader must skip.
fn encode_model(pieces: &[(&str, f32, u64)]) -> Vec<u8> {
    let mut out = Vec::new();
    for (text, score, kind) in pieces {
        let mut p = Vec::new();
        len_field(1, text.as_bytes(), &mut p);
        varint(2 << 3 | 5, &mut p);
        p.extend_from_slice(&score.to_le_bytes());
        varint(3 << 3, &mut p);
        varint(*kind, &mut p);
        // An unknown fixed64 field inside the piece.
        varint(9 << 3 | 1, &mut p);
        p.extend_from_slice(&[0u8; 8]);
        len_field(1, &p, &mut out);
    }
    len_field(2, b"trainer spec", &mut out);
    varint(8 << 3 | 5, &mut out);
    out.extend_from_slice(&[0u8; 4]);
    varint(7 << 3, &mut out);
    varint(300, &mut out);
    out
}

#[test]
fn pieces_round_trip_and_other_fields_are_skipped() {
    let bytes = encode_model(&[
        ("<unk>", 0.0, 2),
        ("<s>", 0.0, 3),
        ("\u{2581}a", -1.5, 1),
        ("x", -2.0, 4),
        ("u", 0.0, 5),
        ("<0x00>", 0.0, 6),
    ]);
    let pieces = parse_pieces(&bytes).unwrap();
    let kinds: Vec<PieceKind> = pieces.iter().map(|p| p.kind).collect();
    assert_eq!(
        kinds,
        vec![
            PieceKind::Unknown,
            PieceKind::Control,
            PieceKind::Normal,
            PieceKind::UserDefined,
            PieceKind::Unused,
            PieceKind::Byte
        ]
    );
    assert_eq!(
        pieces[2],
        Piece {
            text: "\u{2581}a".into(),
            score: -1.5,
            kind: PieceKind::Normal
        }
    );
}

#[test]
fn a_piece_without_a_type_is_normal() {
    let mut piece = Vec::new();
    len_field(1, b"z", &mut piece);
    let mut model = Vec::new();
    len_field(1, &piece, &mut model);
    let pieces = parse_pieces(&model).unwrap();
    assert_eq!(pieces[0].kind, PieceKind::Normal);
    assert_eq!(pieces[0].score, 0.0);
}

#[test]
fn malformed_models_refuse() {
    let good = encode_model(&[("a", 0.0, 1)]);
    let cut = &good[..3];
    assert!(parse_pieces(cut)
        .unwrap_err()
        .to_string()
        .contains("truncated"));
    assert!(parse_pieces(&[])
        .unwrap_err()
        .to_string()
        .contains("no pieces"));
    // Wire type 3 (a group) is not something a ModelProto carries.
    assert!(parse_pieces(&[(5 << 3) | 3])
        .unwrap_err()
        .to_string()
        .contains("wire type"));
    // An eleven-byte varint overflows.
    assert!(parse_pieces(&[0xff; 11])
        .unwrap_err()
        .to_string()
        .contains("overflow"));
    let bad_type = encode_model(&[("a", 0.0, 9)]);
    assert!(parse_pieces(&bad_type)
        .unwrap_err()
        .to_string()
        .contains("piece type 9"));
    let mut not_utf8 = Vec::new();
    len_field(1, &[0xff, 0xfe], &mut not_utf8);
    let mut model = Vec::new();
    len_field(1, &not_utf8, &mut model);
    assert!(parse_pieces(&model)
        .unwrap_err()
        .to_string()
        .contains("UTF-8"));
}
