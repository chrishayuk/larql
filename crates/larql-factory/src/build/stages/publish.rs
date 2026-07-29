//! PUBLISH — `larql hf publish` per output, always `--private` first
//! (docs/vindex-factory.md §8: "nothing goes public unverified" — a
//! build only flips visibility at RELEASE, after VERIFY-B passes).
//!
//! One output at a time via the low-level `larql hf publish` (not the
//! higher-level `larql publish --slices ...`, which re-slices from the
//! full vindex internally) — SLICE already produced each output
//! directory, and re-slicing would duplicate that work for no reason.
//! Repo names are computed with the same [`crate::card::naming`] logic
//! the card generator's `USE` snippet uses, so a build and its own
//! card never disagree about where something was published.

use std::path::Path;

use crate::build::runner::{CommandRunner, Invocation};
use crate::card::naming::{hub_repo_name, slice_suffix};
use crate::{Metadata, OutputSpec, Publish};

/// The exact Hub repo name `output` publishes to.
pub fn repo_name(metadata: &Metadata, publish: &Publish, output: &OutputSpec) -> String {
    hub_repo_name(&publish.hub, &metadata.name, &slice_suffix(&output.preset))
}

/// Build the `larql hf publish <dir> --repo <name> --private [--repo-type T]` invocation.
pub fn invocation(dir: &Path, repo: &str, repo_type: &str) -> Invocation {
    Invocation::new(
        &["hf", "publish"],
        vec![
            dir.to_string_lossy().into_owned(),
            "--repo".to_string(),
            repo.to_string(),
            "--private".to_string(),
            "--repo-type".to_string(),
            repo_type.to_string(),
        ],
    )
}

/// Run PUBLISH for one output directory via `runner`. Returns the repo
/// name it published to.
pub fn run(
    runner: &dyn CommandRunner,
    dir: &Path,
    metadata: &Metadata,
    publish: &Publish,
    output: &OutputSpec,
) -> Result<String, String> {
    let repo = repo_name(metadata, publish, output);
    let inv = invocation(dir, &repo, &publish.hub.repo_type);
    let out = runner.run(&inv)?;
    if !out.success() {
        return Err(format!(
            "larql hf publish failed for {}: {}",
            repo, out.stderr
        ));
    }
    Ok(repo)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build::runner::{CommandOutput, MockRunner};
    use crate::Recipe;

    fn sample_recipe() -> Recipe {
        Recipe::from_yaml(include_str!("../../../testdata/gemma-3-4b-it.yaml")).unwrap()
    }

    #[test]
    fn repo_name_matches_the_card_generators_naming() {
        let recipe = sample_recipe();
        let full = OutputSpec {
            preset: "full".into(),
            shard_cap_gib: None,
        };
        let client = OutputSpec {
            preset: "client".into(),
            shard_cap_gib: None,
        };
        assert_eq!(
            repo_name(&recipe.metadata, &recipe.spec.publish, &full),
            "chrishayuk/gemma-3-4b-it-vindex"
        );
        assert_eq!(
            repo_name(&recipe.metadata, &recipe.spec.publish, &client),
            "chrishayuk/gemma-3-4b-it-vindex-client"
        );
    }

    #[test]
    fn invocation_is_always_private() {
        let dir = Path::new("/scratch/full.vindex");
        let inv = invocation(dir, "owner/repo", "dataset");
        assert!(inv.args.contains(&"--private".to_string()));
        assert_eq!(inv.subcommand, vec!["hf", "publish"]);
        assert_eq!(
            inv.args,
            vec![
                "/scratch/full.vindex".to_string(),
                "--repo".to_string(),
                "owner/repo".to_string(),
                "--private".to_string(),
                "--repo-type".to_string(),
                "dataset".to_string(),
            ]
        );
    }

    #[test]
    fn run_returns_the_repo_name_on_success() {
        let recipe = sample_recipe();
        let output = OutputSpec {
            preset: "full".into(),
            shard_cap_gib: None,
        };
        let dir = Path::new("/scratch/full.vindex");
        let repo = repo_name(&recipe.metadata, &recipe.spec.publish, &output);
        let runner = MockRunner::new().expect(
            invocation(dir, &repo, &recipe.spec.publish.hub.repo_type),
            CommandOutput {
                status: 0,
                ..Default::default()
            },
        );
        let result = run(
            &runner,
            dir,
            &recipe.metadata,
            &recipe.spec.publish,
            &output,
        )
        .unwrap();
        assert_eq!(result, "chrishayuk/gemma-3-4b-it-vindex");
    }

    #[test]
    fn run_errors_with_repo_name_and_stderr_on_failure() {
        let recipe = sample_recipe();
        let output = OutputSpec {
            preset: "full".into(),
            shard_cap_gib: None,
        };
        let dir = Path::new("/scratch/full.vindex");
        let repo = repo_name(&recipe.metadata, &recipe.spec.publish, &output);
        let runner = MockRunner::new().expect(
            invocation(dir, &repo, &recipe.spec.publish.hub.repo_type),
            CommandOutput {
                status: 1,
                stderr: "401 unauthorized".into(),
                ..Default::default()
            },
        );
        let err = run(
            &runner,
            dir,
            &recipe.metadata,
            &recipe.spec.publish,
            &output,
        )
        .unwrap_err();
        assert!(err.contains("gemma-3-4b-it-vindex"), "{err}");
        assert!(err.contains("401"), "{err}");
    }
}
