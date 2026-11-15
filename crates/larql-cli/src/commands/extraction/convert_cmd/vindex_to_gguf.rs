//! `larql convert vindex-to-gguf` — a VINDEX3 container's primary text
//! model as a llama.cpp GGUF.

use std::path::{Path, PathBuf};

use larql_vindex::format::vindex3::gguf::dense::spm_vocab::ScoreSource;
use larql_vindex::format::vindex3::gguf::dense::tokenizer::{
    DERIVED_SCORES_KEY, DERIVED_SCORES_VALUE,
};
use larql_vindex::format::vindex3::gguf::dense::{
    export_dense, ContextSource, DenseExport, DenseExportOptions,
};

/// What the command was asked to do.
pub(super) struct VindexToGgufArgs<'a> {
    pub input: &'a Path,
    pub output: &'a Path,
    pub text_only: bool,
    pub tokenizer_model: Option<PathBuf>,
    pub derived_tokenizer_scores: bool,
}

/// The lines printed after a successful export — every count observed
/// from the finished file.
pub(super) fn summary(report: &DenseExport) -> Vec<String> {
    let mut lines = vec![
        format!(
            "wrote {} ({} bytes): {} {} tensors ({} BF16), {} metadata keys",
            report.out.display(),
            report.bytes,
            report.architecture,
            report.tensors,
            report.bf16_tensors,
            report.metadata_keys
        ),
        format!(
            "  verified: {} tensors, {} metadata keys re-read",
            report.verify.tensors, report.verify.metadata_keys
        ),
        format!(
            "  vocabulary: {} ids ({} padding)",
            report.vocab_tokens, report.vocab_padded
        ),
    ];
    if !report.excluded_components.is_empty() {
        lines.push(format!(
            "  text only: left out {}",
            report.excluded_components.join(", ")
        ));
    }
    if report.context_source == ContextSource::TargetDefault {
        lines.push(format!(
            "  note: the container declares no context length; stated the {} default {}",
            report.architecture, report.context_length
        ));
    }
    match (&report.sentencepiece_model, report.score_source) {
        (Some(p), ScoreSource::SentencePieceModel) => {
            lines.push(format!("  token scores: from {}", p.display()))
        }
        _ => lines.push(format!(
            "  NOT EXACT: no tokenizer.model — token scores derived from vocabulary order \
             (order-preserving, not the trainer's values); the file carries \
             `{DERIVED_SCORES_KEY} = {DERIVED_SCORES_VALUE}`"
        )),
    }
    lines
}

/// The last line of a run: whether the file is an exact export.
pub(super) fn exit_note(report: &DenseExport) -> String {
    if report.exact() {
        "exact export".into()
    } else {
        format!(
            "non-exact export: token scores derived (marked in the file as \
             `{DERIVED_SCORES_KEY}`); pass --tokenizer-model for an exact one"
        )
    }
}

pub(super) fn run(args: VindexToGgufArgs<'_>) -> Result<(), Box<dyn std::error::Error>> {
    let options = DenseExportOptions {
        text_only: args.text_only,
        sentencepiece_model: args.tokenizer_model,
        derived_tokenizer_scores: args.derived_tokenizer_scores,
    };
    let report = export_dense(args.input, args.output, &options)?;
    for line in summary(&report) {
        eprintln!("{line}");
    }
    eprintln!("{}", exit_note(&report));
    Ok(())
}
