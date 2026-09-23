//! Link the ecosystem's ggml ONLY when the reference encoder is asked for.
//!
//! This dependency belongs to artifact PRODUCTION, not to artifact
//! consumption: the VINDEX3 reader and runtime keep their own decoders
//! and must never require llama.cpp. Off by default, so no ordinary
//! build or CI job is affected.
fn main() {
    plugin_abi();
    println!("cargo:rerun-if-env-changed=LARQL_GGML_LIB_DIR");
    println!("cargo:rerun-if-env-changed=LARQL_GGML_REVISION");
    if std::env::var_os("CARGO_FEATURE_REFERENCE_ENCODER").is_none() {
        return;
    }
    let dir = std::env::var("LARQL_GGML_LIB_DIR").expect(
        "feature `reference-encoder` needs LARQL_GGML_LIB_DIR pointing at a built \
         llama.cpp library directory — the reference encoder is linked, not vendored, \
         so the artifact's provenance can name the exact upstream it used",
    );
    println!("cargo:rustc-link-search=native={dir}");
    for lib in ["ggml", "ggml-base", "ggml-cpu"] {
        println!("cargo:rustc-link-lib=dylib={lib}");
    }
    println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
}

/// Stamp the facts a dynamically loaded plugin must share with its host
/// (`format::vindex3::plugin::ABI`): the compiler, and the source commit
/// this crate was built from. Rust has no stable ABI, so a plugin whose
/// `larql-vindex` differs from the host's in either is refused before
/// any Rust-typed symbol is called. Best effort on the commit: a build
/// outside a git checkout stamps `unknown`, and two `unknown`s are
/// refused rather than trusted.
fn plugin_abi() {
    use std::process::Command;
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let version = Command::new(&rustc)
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".into());
    println!("cargo:rustc-env=LARQL_PLUGIN_RUSTC={version}");

    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".into());
    let git = |args: &[&str]| {
        Command::new("git")
            .arg("-C")
            .arg(&manifest)
            .args(args)
            .output()
            .ok()
            .filter(|o| o.status.success())
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
    };
    let commit = git(&["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".into());
    println!("cargo:rustc-env=LARQL_PLUGIN_COMMIT={commit}");
    // Re-stamp when HEAD moves: the HEAD file itself (a branch switch) and
    // the ref it points at (a commit on the branch).
    for path in [
        git(&["rev-parse", "--git-path", "HEAD"]),
        git(&["symbolic-ref", "-q", "HEAD"]).and_then(|r| git(&["rev-parse", "--git-path", &r])),
    ]
    .into_iter()
    .flatten()
    {
        let path = std::path::Path::new(&manifest).join(path);
        println!("cargo:rerun-if-changed={}", path.display());
    }
}
