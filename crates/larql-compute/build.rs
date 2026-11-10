fn main() {
    // Rebuild if anything under csrc/ changes (new .c, new .h, modified source).
    // The cc crate only auto-tracks files passed to .file(); this widens the net so
    // a new or modified C source always triggers recompilation of q4_dot.
    println!("cargo:rerun-if-changed=csrc");
    println!("cargo:rerun-if-changed=build.rs");

    // A build script's cfg describes the HOST, not the consuming target.
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").expect("Cargo supplies target arch");
    let mut baseline = cc::Build::new();
    baseline.file("csrc/q4_dot.c").opt_level(3);
    if arch == "aarch64" {
        baseline.flag("-march=armv8-a");
    }
    baseline.compile("q4_dot");

    if arch == "aarch64" {
        // Separate symbols: only the runtime-checked Rust dispatch may enter
        // this object. The baseline object remains usable without dotprod.
        cc::Build::new()
            .file("csrc/q4_dot.c")
            .opt_level(3)
            .define("LARQL_DOTPROD_VARIANT", None)
            .flag("-march=armv8.2-a+dotprod")
            .compile("q4_dot_dotprod");
    }
}
