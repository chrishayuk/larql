//! [`CommandRunner`] — the seam between stage orchestration and actually
//! executing anything. Every stage in this module asks a `CommandRunner`
//! to run a `larql` subcommand rather than calling `std::process::Command`
//! directly, so the orchestration logic (which command, which args, in
//! which order, how errors propagate) is fully unit-testable against a
//! [`MockRunner`] without spawning a process, touching the network, or
//! needing any credentials.

use std::path::PathBuf;
use std::process::Command;

/// One subprocess invocation: the `larql` subcommand path (e.g.
/// `["recipe", "validate"]` or `["extract"]`), its arguments, and any
/// extra environment variables to set (e.g. an isolated `HF_HUB_CACHE`).
#[derive(Clone, Debug, PartialEq)]
pub struct Invocation {
    /// Subcommand path, e.g. `vec!["model".into(), "pull".into()]`.
    pub subcommand: Vec<String>,
    /// Positional and flag arguments, in order.
    pub args: Vec<String>,
    /// Extra environment variables for this call only.
    pub env: Vec<(String, String)>,
}

impl Invocation {
    /// Build an invocation for `larql <subcommand...> <args...>`, no
    /// extra environment overrides.
    pub fn new(subcommand: &[&str], args: Vec<String>) -> Self {
        Self {
            subcommand: subcommand.iter().map(|s| s.to_string()).collect(),
            args,
            env: Vec::new(),
        }
    }

    /// Add an environment variable override.
    pub fn with_env(mut self, key: &str, value: &str) -> Self {
        self.env.push((key.to_string(), value.to_string()));
        self
    }
}

/// Result of running an [`Invocation`].
#[derive(Clone, Debug, Default)]
pub struct CommandOutput {
    /// Process exit code. `0` is success by convention (matches every
    /// `larql` subcommand's own exit-code contract).
    pub status: i32,
    /// Captured stdout.
    pub stdout: String,
    /// Captured stderr.
    pub stderr: String,
}

impl CommandOutput {
    /// True if the process exited successfully.
    pub fn success(&self) -> bool {
        self.status == 0
    }
}

/// Something that can execute a `larql` [`Invocation`] and report what
/// happened. Implemented by [`SubprocessRunner`] for real builds and
/// [`MockRunner`] for tests.
pub trait CommandRunner {
    /// Run `invocation`, returning its captured output. Errors here mean
    /// the process itself couldn't be spawned (missing binary, OS
    /// error) — a non-zero *exit code* is a normal [`CommandOutput`],
    /// not an `Err`, since callers need to inspect stdout/stderr either
    /// way.
    fn run(&self, invocation: &Invocation) -> Result<CommandOutput, String>;
}

/// Runs invocations against a real `larql` binary via
/// `std::process::Command`. Resolves the binary path from
/// `std::env::current_exe()` — the build driver always runs *inside*
/// the same `larql` binary it's orchestrating, so self-invocation needs
/// no separate binary discovery.
pub struct SubprocessRunner {
    binary: PathBuf,
}

impl SubprocessRunner {
    /// Use the currently-running executable as the `larql` binary.
    pub fn current_exe() -> Result<Self, String> {
        let binary =
            std::env::current_exe().map_err(|e| format!("resolving current executable: {e}"))?;
        Ok(Self { binary })
    }

    /// Use an explicit binary path — mainly for tests that want a real
    /// subprocess without depending on `current_exe()`.
    pub fn at(binary: PathBuf) -> Self {
        Self { binary }
    }
}

impl CommandRunner for SubprocessRunner {
    fn run(&self, invocation: &Invocation) -> Result<CommandOutput, String> {
        let mut cmd = Command::new(&self.binary);
        cmd.args(&invocation.subcommand).args(&invocation.args);
        for (key, value) in &invocation.env {
            cmd.env(key, value);
        }
        let output = cmd
            .output()
            .map_err(|e| format!("spawning {}: {e}", self.binary.display()))?;
        Ok(CommandOutput {
            status: output.status.code().unwrap_or(-1),
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        })
    }
}

/// A scripted [`CommandRunner`] for tests: each call to [`run`] consumes
/// the next `(expected_invocation, response)` pair in order and asserts
/// the actual invocation matches what was expected.
///
/// [`run`]: CommandRunner::run
#[cfg(test)]
pub struct MockRunner {
    expected:
        std::cell::RefCell<std::collections::VecDeque<(Invocation, Result<CommandOutput, String>)>>,
}

#[cfg(test)]
impl MockRunner {
    /// Build a mock that expects no calls at all.
    pub fn new() -> Self {
        Self {
            expected: std::cell::RefCell::new(std::collections::VecDeque::new()),
        }
    }

    /// Queue an expected invocation and the output to return for it.
    pub fn expect(mut self, invocation: Invocation, response: CommandOutput) -> Self {
        self.expected
            .get_mut()
            .push_back((invocation, Ok(response)));
        self
    }

    /// Queue an expected invocation that fails to even spawn.
    pub fn expect_spawn_error(mut self, invocation: Invocation, error: &str) -> Self {
        self.expected
            .get_mut()
            .push_back((invocation, Err(error.to_string())));
        self
    }

    /// True if every queued expectation was consumed. Call at the end
    /// of a test to catch a stage that stopped short.
    pub fn all_consumed(&self) -> bool {
        self.expected.borrow().is_empty()
    }
}

#[cfg(test)]
impl Default for MockRunner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
impl CommandRunner for MockRunner {
    fn run(&self, invocation: &Invocation) -> Result<CommandOutput, String> {
        let (expected, response) = self
            .expected
            .borrow_mut()
            .pop_front()
            .unwrap_or_else(|| panic!("unexpected invocation, none queued: {invocation:?}"));
        assert_eq!(
            &expected, invocation,
            "invocation mismatch — expected {expected:?}, got {invocation:?}"
        );
        response
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invocation_builder_shape() {
        let inv = Invocation::new(&["model", "pull"], vec!["owner/repo".into()])
            .with_env("HF_HUB_CACHE", "/tmp/scratch");
        assert_eq!(inv.subcommand, vec!["model", "pull"]);
        assert_eq!(inv.args, vec!["owner/repo"]);
        assert_eq!(
            inv.env,
            vec![("HF_HUB_CACHE".to_string(), "/tmp/scratch".to_string())]
        );
    }

    #[test]
    fn command_output_success_checks_zero_status() {
        assert!(CommandOutput {
            status: 0,
            ..Default::default()
        }
        .success());
        assert!(!CommandOutput {
            status: 1,
            ..Default::default()
        }
        .success());
    }

    #[test]
    fn mock_runner_returns_queued_response_for_matching_invocation() {
        let inv = Invocation::new(&["recipe", "validate"], vec!["r.yaml".into()]);
        let runner = MockRunner::new().expect(
            inv.clone(),
            CommandOutput {
                status: 0,
                stdout: "ok".into(),
                stderr: String::new(),
            },
        );
        let out = runner.run(&inv).unwrap();
        assert_eq!(out.stdout, "ok");
        assert!(runner.all_consumed());
    }

    #[test]
    fn mock_runner_returns_queued_spawn_error() {
        let inv = Invocation::new(&["extract"], vec![]);
        let runner = MockRunner::new().expect_spawn_error(inv.clone(), "boom");
        let err = runner.run(&inv).unwrap_err();
        assert_eq!(err, "boom");
    }

    #[test]
    #[should_panic(expected = "unexpected invocation")]
    fn mock_runner_panics_on_unqueued_call() {
        let runner = MockRunner::new();
        let _ = runner.run(&Invocation::new(&["extract"], vec![]));
    }

    #[test]
    #[should_panic(expected = "invocation mismatch")]
    fn mock_runner_panics_on_mismatched_call() {
        let runner = MockRunner::new().expect(
            Invocation::new(&["extract"], vec!["a".into()]),
            CommandOutput::default(),
        );
        let _ = runner.run(&Invocation::new(&["extract"], vec!["b".into()]));
    }

    #[test]
    fn all_consumed_false_when_expectations_remain() {
        let runner = MockRunner::new().expect(
            Invocation::new(&["extract"], vec![]),
            CommandOutput::default(),
        );
        assert!(!runner.all_consumed());
    }

    #[test]
    fn subprocess_runner_current_exe_resolves_a_binary() {
        // Doesn't run anything — just confirms current_exe() succeeds
        // in a test binary (it always does; std::env::current_exe()
        // only fails on exotic platforms/sandboxes).
        assert!(SubprocessRunner::current_exe().is_ok());
    }

    #[test]
    fn subprocess_runner_runs_a_real_process() {
        // `true`/`cmd /c exit 0` — no larql binary needed, just proves
        // the Command-building plumbing (args, env, output capture)
        // actually works end to end.
        let bin = if cfg!(windows) {
            PathBuf::from("cmd")
        } else {
            PathBuf::from("/usr/bin/env")
        };
        let runner = SubprocessRunner::at(bin);
        let inv = if cfg!(windows) {
            Invocation::new(&["/c", "exit"], vec!["0".into()])
        } else {
            Invocation::new(&["true"], vec![])
        };
        let out = runner.run(&inv).unwrap();
        assert!(out.success());
    }
}
