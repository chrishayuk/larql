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

    let again = larql(&[container.as_os_str(), out.as_os_str()]);
    assert!(!again.status.success());
    assert!(String::from_utf8_lossy(&again.stderr).contains("already exists"));
}
