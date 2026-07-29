//! FETCH — download the pinned upstream revision via `larql model
//! pull`, isolated to a per-build HF cache directory.
//!
//! The isolation matters: `larql_models::resolve_model_path` (what
//! `larql extract` uses to find a cached model) doesn't disambiguate
//! its cache lookup by revision — it returns the first cached snapshot
//! directory that has safetensors, whichever revision that happens to
//! be. If the shared HF cache already holds a *different* revision of
//! the same repo (e.g. from a previous manual `larql pull`), EXTRACT
//! could silently build from the wrong commit despite FETCH correctly
//! pulling the pinned one. A build-scoped `HF_HUB_CACHE` makes that
//! ambiguity impossible: there's only ever one revision in it.

use crate::build::runner::{CommandOutput, CommandRunner, Invocation};
use crate::Recipe;

const HF_HUB_CACHE_ENV: &str = "HF_HUB_CACHE";

/// Build the `larql model pull <repo> --revision <sha>` invocation,
/// scoped to `hf_cache_dir` via `HF_HUB_CACHE`.
pub fn invocation(recipe: &Recipe, hf_cache_dir: &std::path::Path) -> Invocation {
    Invocation::new(
        &["model", "pull"],
        vec![
            recipe.spec.source.hf_repo.clone(),
            "--revision".to_string(),
            recipe.spec.source.revision.clone(),
        ],
    )
    .with_env(HF_HUB_CACHE_ENV, &hf_cache_dir.to_string_lossy())
}

/// Run FETCH via `runner`.
pub fn run(
    runner: &dyn CommandRunner,
    recipe: &Recipe,
    hf_cache_dir: &std::path::Path,
) -> Result<CommandOutput, String> {
    let inv = invocation(recipe, hf_cache_dir);
    let out = runner.run(&inv)?;
    if !out.success() {
        return Err(format!("larql model pull failed: {}", out.stderr));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::build::runner::MockRunner;

    fn sample_recipe() -> Recipe {
        Recipe::from_yaml(include_str!("../../../testdata/gemma-3-4b-it.yaml")).unwrap()
    }

    #[test]
    fn invocation_pins_repo_and_revision() {
        let recipe = sample_recipe();
        let inv = invocation(&recipe, Path::new("/scratch/hf-cache"));
        assert_eq!(inv.subcommand, vec!["model", "pull"]);
        assert_eq!(
            inv.args,
            vec![
                "google/gemma-3-4b-it".to_string(),
                "--revision".to_string(),
                "9b5b1a2f7c3e0d1a4b8f2c6e9d0a3b5c7e1f4a82".to_string(),
            ]
        );
    }

    #[test]
    fn invocation_isolates_the_hf_cache() {
        let recipe = sample_recipe();
        let inv = invocation(&recipe, Path::new("/scratch/hf-cache"));
        assert_eq!(
            inv.env,
            vec![("HF_HUB_CACHE".to_string(), "/scratch/hf-cache".to_string())]
        );
    }

    #[test]
    fn run_succeeds_on_zero_exit() {
        let recipe = sample_recipe();
        let cache_dir = Path::new("/scratch/hf-cache");
        let runner = MockRunner::new().expect(
            invocation(&recipe, cache_dir),
            CommandOutput {
                status: 0,
                ..Default::default()
            },
        );
        assert!(run(&runner, &recipe, cache_dir).is_ok());
    }

    #[test]
    fn run_errors_with_stderr_on_nonzero_exit() {
        let recipe = sample_recipe();
        let cache_dir = Path::new("/scratch/hf-cache");
        let runner = MockRunner::new().expect(
            invocation(&recipe, cache_dir),
            CommandOutput {
                status: 1,
                stderr: "404 not found".into(),
                ..Default::default()
            },
        );
        let err = run(&runner, &recipe, cache_dir).unwrap_err();
        assert!(err.contains("404"), "{err}");
    }
}
