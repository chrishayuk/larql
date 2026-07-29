//! RELEASE — flip a published repo's visibility public via `larql hf
//! visibility`, once PUBLISH and (checksum) VERIFY have both passed.
//!
//! Tagging an immutable Hub revision (§9's `v{spec}-{extractor}{ver}-
//! {build_id[:8]}` — already computable via [`crate::revision_tag`])
//! and adding the repo to a collection aren't implemented here: this
//! codebase has no CLI surface for either (no `larql hf tag`, no
//! collection-membership call outside the higher-level `larql publish`
//! orchestrator, which this driver deliberately bypasses per
//! `publish.rs`'s doc comment). Visibility is the one RELEASE action
//! that's both load-bearing (§8's "nothing goes public unverified"
//! contract) and has a real CLI command behind it.

use crate::build::runner::{CommandOutput, CommandRunner, Invocation};

/// Build the `larql hf visibility <repo> --public [--repo-type T]` invocation.
pub fn invocation(repo: &str, repo_type: &str) -> Invocation {
    Invocation::new(
        &["hf", "visibility"],
        vec![
            repo.to_string(),
            "--public".to_string(),
            "--repo-type".to_string(),
            repo_type.to_string(),
        ],
    )
}

/// Run RELEASE for one published repo via `runner`.
pub fn run(
    runner: &dyn CommandRunner,
    repo: &str,
    repo_type: &str,
) -> Result<CommandOutput, String> {
    let inv = invocation(repo, repo_type);
    let out = runner.run(&inv)?;
    if !out.success() {
        return Err(format!(
            "larql hf visibility --public failed for {repo}: {}",
            out.stderr
        ));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build::runner::MockRunner;

    #[test]
    fn invocation_flips_to_public() {
        let inv = invocation("owner/repo", "dataset");
        assert_eq!(inv.subcommand, vec!["hf", "visibility"]);
        assert_eq!(
            inv.args,
            vec![
                "owner/repo".to_string(),
                "--public".to_string(),
                "--repo-type".to_string(),
                "dataset".to_string(),
            ]
        );
    }

    #[test]
    fn run_succeeds_on_zero_exit() {
        let runner = MockRunner::new().expect(
            invocation("owner/repo", "dataset"),
            CommandOutput {
                status: 0,
                ..Default::default()
            },
        );
        assert!(run(&runner, "owner/repo", "dataset").is_ok());
    }

    #[test]
    fn run_errors_with_repo_and_stderr_on_failure() {
        let runner = MockRunner::new().expect(
            invocation("owner/repo", "dataset"),
            CommandOutput {
                status: 1,
                stderr: "403 forbidden".into(),
                ..Default::default()
            },
        );
        let err = run(&runner, "owner/repo", "dataset").unwrap_err();
        assert!(err.contains("owner/repo"), "{err}");
        assert!(err.contains("403"), "{err}");
    }
}
