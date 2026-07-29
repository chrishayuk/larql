//! SLICE — `larql slice` for every declared output whose preset isn't
//! `"full"` (the full vindex is EXTRACT's own output, not a slice of
//! it).

use std::path::Path;

use crate::build::runner::{CommandOutput, CommandRunner, Invocation};
use crate::OutputSpec;

const FULL_PRESET: &str = "full";

/// True if `output` needs a SLICE step at all.
pub fn needs_slicing(output: &OutputSpec) -> bool {
    output.preset != FULL_PRESET
}

/// Build the `larql slice <full_dir> -o <dst> --preset <name>` invocation.
pub fn invocation(full_dir: &Path, dst_dir: &Path, output: &OutputSpec) -> Invocation {
    Invocation::new(
        &["slice"],
        vec![
            full_dir.to_string_lossy().into_owned(),
            "-o".to_string(),
            dst_dir.to_string_lossy().into_owned(),
            "--preset".to_string(),
            output.preset.clone(),
            "--force".to_string(),
        ],
    )
}

/// Run SLICE for one output via `runner`.
pub fn run(
    runner: &dyn CommandRunner,
    full_dir: &Path,
    dst_dir: &Path,
    output: &OutputSpec,
) -> Result<CommandOutput, String> {
    let inv = invocation(full_dir, dst_dir, output);
    let out = runner.run(&inv)?;
    if !out.success() {
        return Err(format!(
            "larql slice --preset {} failed: {}",
            output.preset, out.stderr
        ));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build::runner::MockRunner;

    fn full_dir() -> std::path::PathBuf {
        std::path::PathBuf::from("/scratch/full.vindex")
    }

    #[test]
    fn full_preset_does_not_need_slicing() {
        let output = OutputSpec {
            preset: "full".into(),
            shard_cap_gib: None,
        };
        assert!(!needs_slicing(&output));
    }

    #[test]
    fn non_full_preset_needs_slicing() {
        let output = OutputSpec {
            preset: "client".into(),
            shard_cap_gib: None,
        };
        assert!(needs_slicing(&output));
    }

    #[test]
    fn invocation_carries_src_dst_and_preset() {
        let output = OutputSpec {
            preset: "client".into(),
            shard_cap_gib: None,
        };
        let dst = std::path::PathBuf::from("/scratch/client.vindex");
        let inv = invocation(&full_dir(), &dst, &output);
        assert_eq!(inv.subcommand, vec!["slice"]);
        assert_eq!(
            inv.args,
            vec![
                "/scratch/full.vindex".to_string(),
                "-o".to_string(),
                "/scratch/client.vindex".to_string(),
                "--preset".to_string(),
                "client".to_string(),
                "--force".to_string(),
            ]
        );
    }

    #[test]
    fn run_errors_with_preset_name_and_stderr_on_failure() {
        let output = OutputSpec {
            preset: "client".into(),
            shard_cap_gib: None,
        };
        let dst = std::path::PathBuf::from("/scratch/client.vindex");
        let runner = MockRunner::new().expect(
            invocation(&full_dir(), &dst, &output),
            CommandOutput {
                status: 1,
                stderr: "unknown part".into(),
                ..Default::default()
            },
        );
        let err = run(&runner, &full_dir(), &dst, &output).unwrap_err();
        assert!(err.contains("client"), "{err}");
        assert!(err.contains("unknown part"), "{err}");
    }
}
