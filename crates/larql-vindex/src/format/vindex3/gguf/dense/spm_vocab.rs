//! llama.cpp's SentencePiece vocabulary (`tokenizer.ggml.model = llama`).
//!
//! A port of the reference converter's `_create_vocab_sentencepiece`:
//! every id up to the model's vocabulary starts as an UNUSED `[PAD{id}]`,
//! the SentencePiece pieces fill their ids with their own scores and
//! types, then `tokenizer_config.json`'s `added_tokens_decoder` overrides
//! — CONTROL when special (or when the text looks special), USER_DEFINED
//! otherwise, both at score -1000.
//!
//! The scores live only in `tokenizer.model`. A container that does not
//! carry one still has every piece and type (from `tokenizer.json`), but
//! not the scores; for that case the scores are **derived** from the
//! vocabulary order — `-(rank among normal pieces)`, the convention
//! SentencePiece's BPE trainer writes. Derived scores preserve every
//! pairwise order llama.cpp's merge loop compares, but they are not the
//! trainer's numbers, and the result says how many it derived.

use larql_models::loading::gguf::GgufValue;

use super::sentencepiece::{Piece, PieceKind};
use crate::VindexError;

pub const VOCAB_MODEL: &str = "llama";
pub const VOCAB_PRE: &str = "default";

/// gguf `TokenType`.
const TOKEN_NORMAL: i32 = 1;
const TOKEN_UNKNOWN: i32 = 2;
const TOKEN_CONTROL: i32 = 3;
const TOKEN_USER_DEFINED: i32 = 4;
const TOKEN_UNUSED: i32 = 5;
const TOKEN_BYTE: i32 = 6;

/// The score an id no piece defines starts with.
const PAD_SCORE: f32 = -10000.0;
/// The score every `added_tokens_decoder` entry receives.
const ADDED_SCORE: f32 = -1000.0;
/// SentencePiece's whitespace marker, pre-normalised to a space in
/// user-defined tokens.
const SPM_SPACE: &str = "\u{2581}";
const SPACE: &str = " ";
/// `TextModel.does_token_look_special`.
const LOOKS_SPECIAL_EXACT: &[&str] = &["<pad>", "<mask>", "<2mass>", "[@BOS@]"];
const LOOKS_SPECIAL_WRAPPED: &[(&str, &str)] =
    &[("<|", "|>"), ("<\u{ff5c}", "\u{ff5c}>"), ("<unused", ">")];
/// `<0xHH>`, a byte-fallback piece.
const BYTE_PIECE_PREFIX: &str = "<0x";
const BYTE_PIECE_SUFFIX: &str = ">";
const BYTE_PIECE_HEX_DIGITS: usize = 2;
const BPE_MODEL: &str = "BPE";

pub fn looks_special(text: &str) -> bool {
    LOOKS_SPECIAL_EXACT.contains(&text)
        || LOOKS_SPECIAL_WRAPPED
            .iter()
            .any(|(open, close)| text.starts_with(open) && text.ends_with(close))
}

fn is_byte_piece(text: &str) -> bool {
    text.strip_prefix(BYTE_PIECE_PREFIX)
        .and_then(|t| t.strip_suffix(BYTE_PIECE_SUFFIX))
        .is_some_and(|hex| {
            hex.len() == BYTE_PIECE_HEX_DIGITS && hex.chars().all(|c| c.is_ascii_hexdigit())
        })
}

/// Where the scores came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoreSource {
    SentencePieceModel,
    DerivedFromVocabularyOrder,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpmVocab {
    pub tokens: Vec<String>,
    pub scores: Vec<f32>,
    pub types: Vec<i32>,
    pub source: ScoreSource,
    /// Ids no piece or added token defines.
    pub padded: usize,
}

impl SpmVocab {
    /// The `tokenizer.ggml.*` entries, in the reference's order.
    pub fn entries(&self) -> Vec<(String, GgufValue)> {
        vec![
            (
                "tokenizer.ggml.model".into(),
                GgufValue::String(VOCAB_MODEL.into()),
            ),
            (
                "tokenizer.ggml.pre".into(),
                GgufValue::String(VOCAB_PRE.into()),
            ),
            (
                "tokenizer.ggml.tokens".into(),
                GgufValue::Array(self.tokens.iter().cloned().map(GgufValue::String).collect()),
            ),
            (
                "tokenizer.ggml.scores".into(),
                GgufValue::Array(self.scores.iter().copied().map(GgufValue::F32).collect()),
            ),
            (
                "tokenizer.ggml.token_type".into(),
                GgufValue::Array(self.types.iter().copied().map(GgufValue::I32).collect()),
            ),
        ]
    }
}

/// Pieces from `tokenizer.json` when no `tokenizer.model` is carried:
/// every text and type is there, the scores are not (left at zero, and
/// derived after the overrides).
fn pieces_from_tokenizer_json(tokenizer: &serde_json::Value) -> Result<Vec<Piece>, VindexError> {
    let model = &tokenizer["model"];
    let kind = model["type"].as_str().unwrap_or_default();
    if kind != BPE_MODEL {
        return Err(VindexError::Parse(format!(
            "no tokenizer.model, and tokenizer.json's model is `{kind}` — piece scores can be \
             derived from a SentencePiece BPE vocabulary only"
        )));
    }
    let vocab = model["vocab"]
        .as_object()
        .ok_or_else(|| VindexError::Parse("tokenizer.json: model.vocab is not an object".into()))?;
    let byte_fallback = model["byte_fallback"].as_bool().unwrap_or(false);
    let unk = model["unk_token"].as_str();
    let mut by_id: Vec<Option<Piece>> = vec![None; vocab.len()];
    for (text, id) in vocab {
        let slot = id
            .as_u64()
            .and_then(|i| by_id.get_mut(i as usize))
            .ok_or_else(|| {
                VindexError::Parse(format!(
                    "tokenizer.json: `{text}` has id {id}, outside a contiguous vocabulary"
                ))
            })?;
        let kind = if Some(text.as_str()) == unk {
            PieceKind::Unknown
        } else if byte_fallback && is_byte_piece(text) {
            PieceKind::Byte
        } else {
            PieceKind::Normal
        };
        *slot = Some(Piece {
            text: text.clone(),
            score: 0.0,
            kind,
        });
    }
    by_id
        .into_iter()
        .enumerate()
        .map(|(i, p)| {
            p.ok_or_else(|| {
                VindexError::Parse(format!(
                    "tokenizer.json: id {i} has no piece — the vocabulary is not contiguous"
                ))
            })
        })
        .collect()
}

/// Build the vocabulary for a model of `vocab_size` rows.
pub fn spm_vocab(
    vocab_size: usize,
    sentencepiece: Option<&[Piece]>,
    tokenizer: &serde_json::Value,
    tokenizer_config: Option<&serde_json::Value>,
) -> Result<SpmVocab, VindexError> {
    let (pieces, source) = match sentencepiece {
        Some(p) => (p.to_vec(), ScoreSource::SentencePieceModel),
        None => (
            pieces_from_tokenizer_json(tokenizer)?,
            ScoreSource::DerivedFromVocabularyOrder,
        ),
    };
    if pieces.len() > vocab_size {
        return Err(VindexError::Parse(format!(
            "the tokenizer defines {} pieces and the model has {vocab_size} rows — this \
             tokenizer belongs to a different model",
            pieces.len()
        )));
    }
    let mut tokens: Vec<String> = (0..vocab_size).map(|i| format!("[PAD{i}]")).collect();
    let mut scores = vec![PAD_SCORE; vocab_size];
    let mut types = vec![TOKEN_UNUSED; vocab_size];
    let mut defined = vec![false; vocab_size];
    for (id, p) in pieces.iter().enumerate() {
        tokens[id] = p.text.clone();
        scores[id] = p.score;
        types[id] = match p.kind {
            PieceKind::Unknown => TOKEN_UNKNOWN,
            PieceKind::Control => TOKEN_CONTROL,
            PieceKind::Unused => TOKEN_UNUSED,
            PieceKind::Byte => TOKEN_BYTE,
            PieceKind::Normal | PieceKind::UserDefined => TOKEN_NORMAL,
        };
        defined[id] = true;
    }

    let decoder = tokenizer_config
        .and_then(|c| c.get("added_tokens_decoder"))
        .and_then(|d| d.as_object());
    for (id, data) in decoder.into_iter().flatten() {
        let id: usize = id.parse().map_err(|_| {
            VindexError::Parse(format!(
                "tokenizer_config.json: added token id `{id}` is not a number"
            ))
        })?;
        if id >= vocab_size {
            continue;
        }
        let content = data["content"].as_str().ok_or_else(|| {
            VindexError::Parse(format!(
                "tokenizer_config.json: added token {id} has no content"
            ))
        })?;
        let special = data["special"].as_bool().unwrap_or(false);
        let (text, ty) = if special || looks_special(content) {
            (content.to_string(), TOKEN_CONTROL)
        } else {
            (content.replace(SPM_SPACE, SPACE), TOKEN_USER_DEFINED)
        };
        tokens[id] = text;
        types[id] = ty;
        scores[id] = ADDED_SCORE;
        defined[id] = true;
    }

    if source == ScoreSource::DerivedFromVocabularyOrder {
        let mut rank = 0f32;
        for (id, ty) in types.iter().enumerate() {
            if *ty == TOKEN_NORMAL && defined[id] && id < pieces.len() {
                scores[id] = -rank;
                rank += 1.0;
            }
        }
    }
    let padded = defined.iter().filter(|d| !**d).count();
    Ok(SpmVocab {
        tokens,
        scores,
        types,
        source,
        padded,
    })
}
