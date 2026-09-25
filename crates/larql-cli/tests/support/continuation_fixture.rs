//! The C6 plugin fixture, built on demand, and the container it runs on.
//!
//! Shared by the in-process gate (`vindex3_cmd/tests/continuation_plugin.rs`)
//! and the binary gate (`tests/test_continuation_plugin_cli.rs`), each of
//! which includes this file by path. Nothing here links the fixture: it
//! is BUILT by cargo into its own target directory and handed back as a
//! path for `--plugin` to `dlopen`.
//!
//! Its own target directory because cargo holds its build lock on a
//! target directory for the whole `cargo test` run, so building into the
//! running test's directory would wait on itself. A build failure panics
//! with cargo's output — the gate is never skipped.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use larql_vindex::format::filenames::TOKENIZER_JSON;
use larql_vindex::format::vindex3::fixtures::{
    encode_fixture_container, miniature_glimmer, G_VOCAB,
};

/// The fixture crate, as cargo names it.
pub const FIXTURE_PACKAGE: &str = "larql-continuation-fixture";
/// The library stem cargo gives it.
const FIXTURE_LIB: &str = "larql_continuation_fixture";
/// The identity the fixture registers — the C5 provider's.
pub const HOSTILE: &str = "hostile-test-provider/v77";
/// The fixture's target directory, beside the running test's.
const FIXTURE_TARGET_DIR: &str = "c6-plugin-fixture";

/// The workspace root: two levels above this crate's manifest.
pub fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crates/larql-cli sits two levels below the workspace")
        .to_path_buf()
}

/// The running test binary's target directory (`<target>/<profile>/deps/<bin>`).
fn target_dir() -> PathBuf {
    let exe = std::env::current_exe().expect("the test binary has a path");
    exe.ancestors()
        .nth(3)
        .expect("a test binary lives in <target>/<profile>/deps")
        .to_path_buf()
}

/// Build the fixture once per test binary and return the shared
/// library's path.
pub fn fixture_dylib() -> PathBuf {
    static BUILT: OnceLock<PathBuf> = OnceLock::new();
    BUILT
        .get_or_init(|| {
            let dir = target_dir().join(FIXTURE_TARGET_DIR);
            let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".into());
            let started = std::time::Instant::now();
            let mut build = Command::new(cargo);
            build
                .current_dir(workspace_root())
                .args(["build", "--locked", "-p", FIXTURE_PACKAGE, "--target-dir"])
                .arg(&dir);
            // The same larql-vindex features as the CLI under test.
            if cfg!(feature = "gpu") {
                build.args(["--features", "gpu"]);
            }
            let output = build
                .output()
                .expect("run cargo to build the plugin fixture");
            assert!(
                output.status.success(),
                "building {FIXTURE_PACKAGE} failed:\n{}",
                String::from_utf8_lossy(&output.stderr)
            );
            eprintln!(
                "c6: built {FIXTURE_PACKAGE} in {:.1} s",
                started.elapsed().as_secs_f64()
            );
            let lib = dir.join("debug").join(format!(
                "{}{FIXTURE_LIB}{}",
                std::env::consts::DLL_PREFIX,
                std::env::consts::DLL_SUFFIX
            ));
            assert!(
                lib.is_file(),
                "cargo reported success but {lib:?} is absent"
            );
            lib
        })
        .clone()
}

/// A WordLevel tokenizer whose token `[i]` is id `i`, split on
/// whitespace only, so a prompt is its own id list.
fn word_level_tokenizer(vocab: usize) -> String {
    let entries: Vec<String> = (0..vocab).map(|i| format!("\"[{i}]\":{i}")).collect();
    format!(
        "{{\"version\":\"1.0\",\"truncation\":null,\"padding\":null,\"added_tokens\":[],\
         \"normalizer\":null,\"pre_tokenizer\":{{\"type\":\"WhitespaceSplit\"}},\
         \"post_processor\":null,\"decoder\":null,\
         \"model\":{{\"type\":\"WordLevel\",\"vocab\":{{{}}},\"unk_token\":\"[0]\"}}}}",
        entries.join(",")
    )
}

/// The sliding-window fixture as a container under `root`, with a
/// tokenizer, so `larql run` can take text.
pub fn glimmer_container(root: &Path) -> PathBuf {
    let checkpoint = root.join("checkpoint");
    let container = root.join("container");
    std::fs::create_dir_all(&checkpoint).unwrap();
    std::fs::create_dir_all(&container).unwrap();
    encode_fixture_container(miniature_glimmer, &checkpoint, &container, "c6-glimmer");
    std::fs::write(
        container.join(TOKENIZER_JSON),
        word_level_tokenizer(G_VOCAB),
    )
    .unwrap();
    container
}
