fn main() {
    // Cargo's TARGET describes the binary being tested even when its build
    // script runs on a different host. Runtime env vars and directory names
    // cannot reliably recover it (custom target directories are allowed).
    let target = std::env::var("TARGET").expect("Cargo supplies TARGET");
    println!("cargo:rustc-env=LARQL_TEST_TARGET={target}");
    println!("cargo:rerun-if-changed=build.rs");
}
