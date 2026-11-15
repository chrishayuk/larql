//! Every `tokenizer.*` key, from the container's capability snapshot.

use std::path::{Path, PathBuf};

use larql_models::loading::gguf::GgufValue;

use super::sentencepiece::parse_pieces;
use super::special::special_entries;
use super::spm_vocab::{spm_vocab, ScoreSource};
use super::target::DenseTarget;
use crate::VindexError;

pub const TOKENIZER_JSON: &str = "tokenizer.json";
pub const TOKENIZER_CONFIG_JSON: &str = "tokenizer_config.json";
pub const SENTENCEPIECE_MODEL: &str = "tokenizer.model";
const ADD_SPACE_PREFIX_KEY: &str = "tokenizer.ggml.add_space_prefix";
/// The key a non-exact export carries: llama.cpp ignores keys it does not
/// know, so the file itself says its token scores are not the trainer's.
pub const DERIVED_SCORES_KEY: &str = "larql.export.tokenizer_scores";
pub const DERIVED_SCORES_VALUE: &str = "derived_from_vocabulary_order";

/// The tokenizer keys and where their scores came from.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenizerTable {
    pub entries: Vec<(String, GgufValue)>,
    pub score_source: ScoreSource,
    /// The `tokenizer.model` the scores were read from, if any.
    pub sentencepiece_model: Option<PathBuf>,
    pub tokens: usize,
    pub padded: usize,
}

pub(crate) fn read_json(path: &Path) -> Result<Option<serde_json::Value>, VindexError> {
    if !path.exists() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(path)?;
    serde_json::from_str(&text)
        .map(Some)
        .map_err(|e| VindexError::Parse(format!("{}: {e}", path.display())))
}

/// Build the table for `container`. `sentencepiece` names a
/// `tokenizer.model` to read scores from when the container carries none.
/// With neither, the export refuses unless `allow_derived` opts into
/// scores derived from vocabulary order — which are not the trainer's
/// (on Gemma 3 4B, 19,225 of 262,208 differ), so the table then carries
/// [`DERIVED_SCORES_KEY`] marking the file non-exact.
pub fn tokenizer_table(
    container: &Path,
    target: &DenseTarget,
    vocab_size: usize,
    sentencepiece: Option<&Path>,
    allow_derived: bool,
) -> Result<TokenizerTable, VindexError> {
    let tokenizer = read_json(&container.join(TOKENIZER_JSON))?.ok_or_else(|| {
        VindexError::Parse(format!(
            "{}: no {TOKENIZER_JSON} — the container carries no text capability to export",
            container.display()
        ))
    })?;
    let config = read_json(&container.join(TOKENIZER_CONFIG_JSON))?;
    let carried = container.join(SENTENCEPIECE_MODEL);
    let model_path = if carried.exists() {
        Some(carried)
    } else {
        sentencepiece.map(Path::to_path_buf)
    };
    if model_path.is_none() && !allow_derived {
        return Err(VindexError::Parse(format!(
            "vindex-to-gguf: {} carries no {SENTENCEPIECE_MODEL} and none was given — the \
             piece scores live only there. Scores derived from vocabulary order are not the \
             trainer's (on Gemma 3 4B, 19,225 of 262,208 differ), so the file would not be an \
             exact export. Pass --tokenizer-model <path>, or --derived-tokenizer-scores to \
             accept a non-exact file marked `{DERIVED_SCORES_KEY}`",
            container.display()
        )));
    }
    let pieces = match &model_path {
        Some(p) => Some(parse_pieces(&std::fs::read(p)?)?),
        None => None,
    };
    let vocab = spm_vocab(vocab_size, pieces.as_deref(), &tokenizer, config.as_ref())?;
    let mut entries = vocab.entries();
    entries.extend(special_entries(
        container,
        &tokenizer,
        config.as_ref(),
        vocab.tokens.len(),
    )?);
    entries.push((
        ADD_SPACE_PREFIX_KEY.into(),
        GgufValue::Bool(target.add_space_prefix),
    ));
    if vocab.source == ScoreSource::DerivedFromVocabularyOrder {
        entries.push((
            DERIVED_SCORES_KEY.into(),
            GgufValue::String(DERIVED_SCORES_VALUE.into()),
        ));
    }
    Ok(TokenizerTable {
        entries,
        score_source: vocab.source,
        sentencepiece_model: model_path,
        tokens: vocab.tokens.len(),
        padded: vocab.padded,
    })
}
