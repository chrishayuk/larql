//! The piece table of a SentencePiece `tokenizer.model`.
//!
//! A `ModelProto` is protobuf; the pieces are its field 1, each a
//! `SentencePiece { piece = 1 (string), score = 2 (float), type = 3
//! (enum) }`. That is all llama.cpp's converter reads from the file
//! (`IdToPiece`, `GetScore`, `IsUnknown`/`IsControl`/`IsUnused`/`IsByte`),
//! so that is all this reads. Every other field is skipped by wire type,
//! never interpreted.

use crate::VindexError;

/// `ModelProto.pieces`.
const MODEL_PIECES_FIELD: u64 = 1;
const PIECE_TEXT_FIELD: u64 = 1;
const PIECE_SCORE_FIELD: u64 = 2;
const PIECE_TYPE_FIELD: u64 = 3;

const WIRE_VARINT: u64 = 0;
const WIRE_FIXED64: u64 = 1;
const WIRE_LEN: u64 = 2;
const WIRE_FIXED32: u64 = 5;
const WIRE_TYPE_BITS: u32 = 3;
const WIRE_TYPE_MASK: u64 = 0b111;
const VARINT_PAYLOAD_BITS: u32 = 7;
const VARINT_CONTINUE: u8 = 0x80;
const VARINT_MAX_SHIFT: u32 = 63;
const FIXED64_BYTES: usize = 8;
const FIXED32_BYTES: usize = 4;

/// `SentencePiece.Type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PieceKind {
    Normal,
    Unknown,
    Control,
    UserDefined,
    Unused,
    Byte,
}

impl PieceKind {
    fn from_proto(v: u64) -> Result<Self, VindexError> {
        Ok(match v {
            1 => Self::Normal,
            2 => Self::Unknown,
            3 => Self::Control,
            4 => Self::UserDefined,
            5 => Self::Unused,
            6 => Self::Byte,
            other => {
                return Err(VindexError::Parse(format!(
                    "tokenizer.model: piece type {other} is not a SentencePiece type"
                )))
            }
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Piece {
    pub text: String,
    pub score: f32,
    pub kind: PieceKind,
}

struct Cursor<'a> {
    bytes: &'a [u8],
    at: usize,
}

fn truncated() -> VindexError {
    VindexError::Parse("tokenizer.model: truncated protobuf".into())
}

impl<'a> Cursor<'a> {
    fn done(&self) -> bool {
        self.at >= self.bytes.len()
    }

    fn varint(&mut self) -> Result<u64, VindexError> {
        let mut value = 0u64;
        let mut shift = 0u32;
        loop {
            let b = *self.bytes.get(self.at).ok_or_else(truncated)?;
            self.at += 1;
            value |= u64::from(b & !VARINT_CONTINUE) << shift;
            if b & VARINT_CONTINUE == 0 {
                return Ok(value);
            }
            shift += VARINT_PAYLOAD_BITS;
            if shift > VARINT_MAX_SHIFT {
                return Err(VindexError::Parse(
                    "tokenizer.model: varint overflow".into(),
                ));
            }
        }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], VindexError> {
        let end = self.at.checked_add(n).ok_or_else(truncated)?;
        let out = self.bytes.get(self.at..end).ok_or_else(truncated)?;
        self.at = end;
        Ok(out)
    }

    /// `(field, wire type)`.
    fn key(&mut self) -> Result<(u64, u64), VindexError> {
        let k = self.varint()?;
        Ok((k >> WIRE_TYPE_BITS, k & WIRE_TYPE_MASK))
    }

    fn skip(&mut self, wire: u64) -> Result<(), VindexError> {
        match wire {
            WIRE_VARINT => self.varint().map(drop),
            WIRE_FIXED64 => self.take(FIXED64_BYTES).map(drop),
            WIRE_LEN => {
                let n = self.varint()? as usize;
                self.take(n).map(drop)
            }
            WIRE_FIXED32 => self.take(FIXED32_BYTES).map(drop),
            other => Err(VindexError::Parse(format!(
                "tokenizer.model: unsupported protobuf wire type {other}"
            ))),
        }
    }
}

fn parse_piece(bytes: &[u8]) -> Result<Piece, VindexError> {
    let mut c = Cursor { bytes, at: 0 };
    let mut text = String::new();
    let mut score = 0.0f32;
    let mut kind = PieceKind::Normal;
    while !c.done() {
        match c.key()? {
            (PIECE_TEXT_FIELD, WIRE_LEN) => {
                let n = c.varint()? as usize;
                text = String::from_utf8(c.take(n)?.to_vec()).map_err(|e| {
                    VindexError::Parse(format!("tokenizer.model: piece is not UTF-8: {e}"))
                })?;
            }
            (PIECE_SCORE_FIELD, WIRE_FIXED32) => {
                let b = c.take(FIXED32_BYTES)?;
                score = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            }
            (PIECE_TYPE_FIELD, WIRE_VARINT) => kind = PieceKind::from_proto(c.varint()?)?,
            (_, wire) => c.skip(wire)?,
        }
    }
    Ok(Piece { text, score, kind })
}

/// Every piece of a serialised `ModelProto`, in id order.
pub fn parse_pieces(bytes: &[u8]) -> Result<Vec<Piece>, VindexError> {
    let mut c = Cursor { bytes, at: 0 };
    let mut pieces = Vec::new();
    while !c.done() {
        match c.key()? {
            (MODEL_PIECES_FIELD, WIRE_LEN) => {
                let n = c.varint()? as usize;
                pieces.push(parse_piece(c.take(n)?)?);
            }
            (_, wire) => c.skip(wire)?,
        }
    }
    if pieces.is_empty() {
        return Err(VindexError::Parse(
            "tokenizer.model: no pieces — not a SentencePiece model".into(),
        ));
    }
    Ok(pieces)
}
