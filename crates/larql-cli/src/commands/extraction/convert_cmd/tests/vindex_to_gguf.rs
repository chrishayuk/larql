//! `larql convert vindex-to-gguf`: the run and the lines it prints.

use std::path::PathBuf;

use larql_vindex::format::vindex3::fixtures::gemma3_text_container;

use super::super::vindex_to_gguf::{exit_note, run, summary, VindexToGgufArgs};

#[test]
fn a_run_writes_the_file_and_the_summary_names_its_sources() {
    let work = tempfile::tempdir().unwrap();
    let container = gemma3_text_container(work.path());
    let out = work.path().join("m.gguf");
    run(VindexToGgufArgs {
        input: &container,
        output: &out,
        text_only: false,
        tokenizer_model: None,
        derived_tokenizer_scores: false,
    })
    .unwrap();
    assert!(out.is_file());

    let report = larql_vindex::format::vindex3::gguf::dense::export_dense(
        &container,
        &work.path().join("again.gguf"),
        &Default::default(),
    )
    .unwrap();
    let lines = summary(&report);
    assert!(lines[0].contains("gemma3 80 tensors"), "{lines:?}");
    assert!(lines.iter().any(|l| l.contains("token scores: from")));
    assert_eq!(exit_note(&report), "exact export");
    assert!(!lines
        .iter()
        .any(|l| l.contains("text only") || l.contains("note:")));

    let mut noted = report.clone();
    noted.excluded_components = vec!["vision".into()];
    noted.context_length = 131_072;
    noted.context_source = larql_vindex::format::vindex3::gguf::dense::ContextSource::TargetDefault;
    noted.sentencepiece_model = Some(PathBuf::from("/m/tokenizer.model"));
    let lines = summary(&noted);
    assert!(lines.iter().any(|l| l.contains("left out vision")));
    assert!(lines.iter().any(|l| l.contains("default 131072")));
    assert!(lines.iter().any(|l| l.contains("from /m/tokenizer.model")));

    noted.sentencepiece_model = None;
    noted.score_source =
        larql_vindex::format::vindex3::gguf::dense::spm_vocab::ScoreSource::DerivedFromVocabularyOrder;
    let lines = summary(&noted);
    assert!(lines
        .iter()
        .any(|l| l.contains("NOT EXACT") && l.contains("larql.export.tokenizer_scores")));
    assert!(exit_note(&noted).starts_with("non-exact export"));
}

#[test]
fn a_run_without_a_tokenizer_model_needs_the_opt_in() {
    let work = tempfile::tempdir().unwrap();
    let container = gemma3_text_container(work.path());
    std::fs::remove_file(container.join("tokenizer.model")).unwrap();
    fn args<'a>(
        container: &'a std::path::Path,
        out: &'a std::path::Path,
        derived: bool,
    ) -> VindexToGgufArgs<'a> {
        VindexToGgufArgs {
            input: container,
            output: out,
            text_only: false,
            tokenizer_model: None,
            derived_tokenizer_scores: derived,
        }
    }
    let refused = work.path().join("refused.gguf");
    let err = run(args(&container, &refused, false))
        .unwrap_err()
        .to_string();
    assert!(err.contains("--derived-tokenizer-scores"), "{err}");
    assert!(!refused.exists());
    let marked = work.path().join("marked.gguf");
    run(args(&container, &marked, true)).unwrap();
    assert!(marked.is_file());
}

#[test]
fn a_refused_run_is_an_error() {
    let work = tempfile::tempdir().unwrap();
    let container = gemma3_text_container(work.path());
    let out = work.path().join("m.gguf");
    std::fs::write(&out, b"x").unwrap();
    let err = run(VindexToGgufArgs {
        input: &container,
        output: &out,
        text_only: true,
        tokenizer_model: None,
        derived_tokenizer_scores: false,
    })
    .unwrap_err();
    assert!(err.to_string().contains("already exists"));
}
