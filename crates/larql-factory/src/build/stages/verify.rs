//! VERIFY — checksum integrity via the existing `larql verify` command.
//!
//! This is **not** §8.1's full verification contract. Reconstruction
//! fidelity (cosine/max-diff vs the upstream tensor) and logit-match
//! (teacher-forced bits/char drift vs a reference forward pass) are
//! not implemented here — both need per-architecture tensor-naming
//! knowledge (mapping "layer N's down_proj" to the right safetensors
//! key per model family) this session has no way to validate against
//! real model weights, and neither `larql shannon verify` nor `larql
//! dec-bench drift` do what the spec assumed (see `build/mod.rs`'s
//! module doc). What runs here — every shard's SHA256 against
//! `index.json` — is real and correct; it's the manifest-integrity
//! third of §8.1, not the whole thing.

use std::path::Path;

use crate::build::runner::{CommandOutput, CommandRunner, Invocation};

/// Build the `larql verify <dir>` invocation.
pub fn invocation(dir: &Path) -> Invocation {
    Invocation::new(&["verify"], vec![dir.to_string_lossy().into_owned()])
}

/// Run checksum VERIFY for one output directory via `runner`.
pub fn run(runner: &dyn CommandRunner, dir: &Path) -> Result<CommandOutput, String> {
    let inv = invocation(dir);
    let out = runner.run(&inv)?;
    if !out.success() {
        return Err(format!(
            "larql verify failed for {}: {}",
            dir.display(),
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
    fn invocation_carries_the_directory() {
        let dir = Path::new("/scratch/full.vindex");
        let inv = invocation(dir);
        assert_eq!(inv.subcommand, vec!["verify"]);
        assert_eq!(inv.args, vec!["/scratch/full.vindex".to_string()]);
    }

    #[test]
    fn run_succeeds_on_zero_exit() {
        let dir = Path::new("/scratch/full.vindex");
        let runner = MockRunner::new().expect(
            invocation(dir),
            CommandOutput {
                status: 0,
                ..Default::default()
            },
        );
        assert!(run(&runner, dir).is_ok());
    }

    #[test]
    fn run_errors_with_directory_and_stderr_on_failure() {
        let dir = Path::new("/scratch/full.vindex");
        let runner = MockRunner::new().expect(
            invocation(dir),
            CommandOutput {
                status: 1,
                stderr: "checksum mismatch".into(),
                ..Default::default()
            },
        );
        let err = run(&runner, dir).unwrap_err();
        assert!(err.contains("full.vindex"), "{err}");
        assert!(err.contains("checksum mismatch"), "{err}");
    }
}
