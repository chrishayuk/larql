//! `larql convert vindex-to-gguf` through the real binary: the argument
//! surface, the file it writes, and its refusals' exit status.

use std::process::Command;

use larql_vindex::format::vindex3::fixtures::gemma3_text_container;

fn larql(args: &[&std::ffi::OsStr]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_larql"))
        .arg("convert")
        .arg("vindex-to-gguf")
        .args(args)
        .output()
        .expect("runs larql")
}

#[test]
fn the_binary_exports_and_refuses_an_existing_output() {
    let work = tempfile::tempdir().unwrap();
    let container = gemma3_text_container(work.path());
    let out = work.path().join("m.gguf");
    let o = larql(&[
        container.as_os_str(),
        out.as_os_str(),
        "--text-only".as_ref(),
    ]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let stderr = String::from_utf8_lossy(&o.stderr);
    assert!(stderr.contains("verified: 80 tensors"), "{stderr}");
    let gguf = larql_models::loading::gguf::GgufFile::open(&out).unwrap();
    assert_eq!(gguf.tensor_infos.len(), 80);

    assert!(stderr.trim_end().ends_with("exact export"), "{stderr}");

    let again = larql(&[container.as_os_str(), out.as_os_str()]);
    assert!(!again.status.success());
    assert!(String::from_utf8_lossy(&again.stderr).contains("already exists"));
}

#[test]
fn derived_scores_refuse_without_the_flag_and_are_marked_with_it() {
    let work = tempfile::tempdir().unwrap();
    let container = gemma3_text_container(work.path());
    std::fs::remove_file(container.join("tokenizer.model")).unwrap();
    let out = work.path().join("m.gguf");
    let refused = larql(&[container.as_os_str(), out.as_os_str()]);
    assert!(!refused.status.success());
    let stderr = String::from_utf8_lossy(&refused.stderr);
    assert!(stderr.contains("19,225 of 262,208"), "{stderr}");
    assert!(!out.exists());

    let marked = larql(&[
        container.as_os_str(),
        out.as_os_str(),
        "--derived-tokenizer-scores".as_ref(),
    ]);
    assert!(marked.status.success());
    let stderr = String::from_utf8_lossy(&marked.stderr);
    assert!(
        stderr.contains("NOT EXACT") && stderr.contains("non-exact export"),
        "{stderr}"
    );
    let gguf = larql_models::loading::gguf::GgufFile::open(&out).unwrap();
    assert_eq!(
        gguf.metadata
            .get("larql.export.tokenizer_scores")
            .and_then(|v| v.as_str()),
        Some("derived_from_vocabulary_order")
    );
}
