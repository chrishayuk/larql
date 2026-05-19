//! Full Metal shader bench and inventory.
//!
//! Usage:
//!   cargo run --release --features metal -p larql-compute --example diag_shader_bench
//!   cargo run --release --features metal -p larql-compute --example diag_shader_bench -- --profile gemma3 --json /tmp/shaders.json

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("This example requires macOS and --features metal");
}

#[cfg(target_os = "macos")]
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cfg = match larql_compute_metal::diag::shader_bench::Config::from_args(&args) {
        Ok(cfg) => cfg,
        Err(e) => {
            eprintln!("{e}");
            eprintln!();
            eprintln!("{}", larql_compute_metal::diag::shader_bench::usage());
            std::process::exit(2);
        }
    };

    if let Err(e) = larql_compute_metal::diag::shader_bench::run(&cfg) {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
