//! Detect newly introduced crates in a PR that lack a wasm-cert manifest.
//!
//! When running in a GitHub Actions PR context, queries the PR file list via
//! `gh api` and posts a `neutral` Check Annotation for any new `Cargo.toml`
//! that is missing `[package.metadata.wasm-cert]`.

use anyhow::Result;

pub fn run() -> Result<()> {
    let Some(pr) = crate::github::pr_number() else {
        return Ok(());
    };

    let files = crate::github::pr_files(pr);
    let new_cargo_tomls: Vec<_> = files
        .iter()
        .filter(|f| {
            // Added Cargo.toml files matching crates/*/Cargo.toml
            let p = std::path::Path::new(f.as_str());
            p.file_name().and_then(|n| n.to_str()) == Some("Cargo.toml")
                && p.components().count() == 3  // crates / <name> / Cargo.toml
        })
        .collect();

    for toml_path in new_cargo_tomls {
        let content = std::fs::read_to_string(toml_path).unwrap_or_default();
        if !content.contains("[package.metadata.wasm-cert]") {
            // Extract crate name from path: crates/<name>/Cargo.toml
            let crate_name = std::path::Path::new(toml_path)
                .parent()
                .and_then(|p| p.file_name())
                .and_then(|n| n.to_str())
                .unwrap_or("unknown");

            let annotation = (
                toml_path.clone(),
                1u32,
                format!(
                    "New crate `{crate_name}` has no `[package.metadata.wasm-cert]` manifest. \
                     Add `[package.metadata.wasm-cert]\\nclaimed-level = 0` to register its \
                     wasm32 boundary status."
                ),
            );

            crate::github::post_check(crate_name, "neutral", &[annotation])?;

            println!(
                "WARNING: new crate `{crate_name}` at `{toml_path}` \
                 has no wasm-cert manifest"
            );
        }
    }

    Ok(())
}
