//! Link the ecosystem's ggml ONLY when the reference encoder is asked for.
//!
//! This dependency belongs to artifact PRODUCTION, not to artifact
//! consumption: the VINDEX3 reader and runtime keep their own decoders
//! and must never require llama.cpp. Off by default, so no ordinary
//! build or CI job is affected.
fn main() {
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
