//! `VERIFY REPRESENTATION`: ask an external verifier whether one container
//! meets a numerical contract, and report what it measured.
//!
//! **LQL does not evaluate the contract.** A check written here would be
//! LARQL marking its own work, so the measurement belongs to a separate
//! program with no LARQL code in it. This module runs that program, checks
//! that its answer is well formed and about the question asked, and renders
//! it. No verifier means a refusal, never a pass.
//!
//! The statement is read-only. It writes nothing and binds nothing. A result
//! is a numerical measurement of the bytes of one container, not a proof about
//! the encoder that produced them, and the output says so.
//!
//! ## Protocol `larql-verify-contract/1`
//!
//! Specified in `docs/verify-contract-protocol.md`. In short:
//!
//! ```text
//! <verifier> contract --contract <id> --container <dir> --source <dir>
//! ```
//!
//! One JSON object on stdout. Exit 0 when the contract was judged or could not
//! be evaluated; any other exit means the verifier itself failed. `status` is
//! `satisfied`, `violated` or `could_not_evaluate`. A violation is an answer,
//! so it returns normally; only `could_not_evaluate` is an error.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use serde::Deserialize;

use super::Session;
use crate::error::LqlError;

const SCHEMA: &str = "larql-verify-contract/1";
const EVIDENCE_KIND: &str = "numerical_measurement";
const VERIFIER_ENV: &str = "LARQL_VERIFIER";
const TIMEOUT: Duration = Duration::from_secs(60 * 60);
const OFFENDING_SHOWN: usize = 10;
const STDERR_SHOWN: usize = 2000;

#[derive(Debug, Deserialize)]
struct ContractResult {
    schema: String,
    contract: String,
    #[serde(default)]
    contract_statement: String,
    evidence_kind: String,
    status: String,
    violations: u64,
    container: Identified,
    source: Source,
    #[serde(default)]
    measurements: Vec<Measurement>,
    #[serde(default)]
    offending: Vec<Offending>,
    #[serde(default)]
    reasons: Vec<String>,
    verifier: serde_json::Map<String, serde_json::Value>,
    #[serde(default)]
    report_sha256: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Identified {
    path: String,
    sha256: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Source {
    path: String,
    identity: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Measurement {
    name: String,
    value: String,
}

#[derive(Debug, Deserialize)]
struct Offending {
    name: String,
    violations: u64,
    #[serde(default)]
    detail: String,
}

impl Session {
    pub(crate) fn exec_verify_representation(
        &mut self,
        container: &str,
        contract: &str,
        source: &str,
    ) -> Result<Vec<String>, LqlError> {
        let verifier = self
            .verifier
            .clone()
            .or_else(|| std::env::var_os(VERIFIER_ENV).map(PathBuf::from))
            .ok_or_else(|| {
                LqlError::Execution(format!(
                    "VERIFY REPRESENTATION needs an independent verifier: set {VERIFIER_ENV}. \
                     LARQL does not evaluate contracts itself."
                ))
            })?;
        let container = existing_dir(container, "container")?;
        let source = existing_dir(source, "source")?;
        let stdout = run_verifier(&verifier, contract, &container, &source, TIMEOUT)?;
        let result = parse_result(&stdout, contract, &container, &source)?;
        if result.status == "could_not_evaluate" {
            return Err(LqlError::Execution(format!(
                "contract {contract} could not be evaluated on {container}: {}",
                result.reasons.join("; ")
            )));
        }
        if result.status == "violated" {
            self.contract_violations += 1;
        }
        Ok(render(&result))
    }
}

fn existing_dir(path: &str, what: &str) -> Result<String, LqlError> {
    let canonical = std::fs::canonicalize(path)
        .map_err(|e| LqlError::exec(format!("VERIFY REPRESENTATION {what} {path}"), e))?;
    if !canonical.is_dir() {
        return Err(LqlError::Execution(format!(
            "VERIFY REPRESENTATION {what} {path} is not a directory"
        )));
    }
    Ok(canonical.to_string_lossy().into_owned())
}

/// Run the verifier, without a shell, and return its stdout. Both pipes are
/// drained on their own threads, so a large answer cannot block the child.
fn run_verifier(
    verifier: &Path,
    contract: &str,
    container: &str,
    source: &str,
    timeout: Duration,
) -> Result<String, LqlError> {
    let shown = verifier.display();
    let mut child = Command::new(verifier)
        .args([
            "contract",
            "--contract",
            contract,
            "--container",
            container,
            "--source",
            source,
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| LqlError::exec(format!("starting verifier {shown}"), e))?;
    let drain = |mut pipe: Box<dyn Read + Send>| {
        std::thread::spawn(move || {
            let mut bytes = Vec::new();
            let _ = pipe.read_to_end(&mut bytes);
            String::from_utf8_lossy(&bytes).into_owned()
        })
    };
    let out = drain(Box::new(child.stdout.take().expect("stdout is piped")));
    let err = drain(Box::new(child.stderr.take().expect("stderr is piped")));

    let started = Instant::now();
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if started.elapsed() >= timeout => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(LqlError::Execution(format!(
                    "verifier {shown} did not finish within {}s",
                    timeout.as_secs()
                )));
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(20)),
            Err(e) => return Err(LqlError::exec(format!("waiting for verifier {shown}"), e)),
        }
    };
    let stdout = out.join().unwrap_or_default();
    let stderr = err.join().unwrap_or_default();
    if !status.success() {
        let tail: String = {
            let chars: Vec<char> = stderr.trim().chars().collect();
            chars[chars.len().saturating_sub(STDERR_SHOWN)..]
                .iter()
                .collect()
        };
        return Err(LqlError::Execution(format!(
            "verifier {shown} failed ({status}): {tail}"
        )));
    }
    Ok(stdout)
}

/// The answer must be well formed, about the question asked, and consistent
/// with itself. Anything else is refused rather than displayed as a result.
fn parse_result(
    stdout: &str,
    contract: &str,
    container: &str,
    source: &str,
) -> Result<ContractResult, LqlError> {
    let bad =
        |why: String| LqlError::Execution(format!("the verifier's answer was refused: {why}"));
    let r: ContractResult =
        serde_json::from_str(stdout).map_err(|e| bad(format!("not a valid result: {e}")))?;
    if r.schema != SCHEMA {
        return Err(bad(format!("schema {:?}, expected {SCHEMA:?}", r.schema)));
    }
    if r.contract != contract {
        return Err(bad(format!(
            "it answers contract {:?}, not {contract:?}",
            r.contract
        )));
    }
    if r.container.path != container {
        return Err(bad(format!(
            "it measured {:?}, not {container:?}",
            r.container.path
        )));
    }
    if r.source.path != source {
        return Err(bad(format!(
            "it used source {:?}, not {source:?}",
            r.source.path
        )));
    }
    if r.evidence_kind != EVIDENCE_KIND {
        return Err(bad(format!(
            "evidence kind {:?}; this statement reports only {EVIDENCE_KIND:?}",
            r.evidence_kind
        )));
    }
    let listed: u64 = r.offending.iter().map(|o| o.violations).sum();
    match r.status.as_str() {
        "satisfied" if r.violations != 0 || !r.offending.is_empty() => {
            Err(bad(format!("satisfied, with {} violations", r.violations)))
        }
        "violated" if r.violations == 0 || listed != r.violations => Err(bad(format!(
            "violated, with {} violations but {listed} listed",
            r.violations
        ))),
        "could_not_evaluate" if r.reasons.is_empty() => {
            Err(bad("could_not_evaluate, with no reason".into()))
        }
        "satisfied" | "violated" | "could_not_evaluate" => Ok(r),
        other => Err(bad(format!("unknown status {other:?}"))),
    }
}

fn render(r: &ContractResult) -> Vec<String> {
    let verdict = if r.status == "satisfied" {
        "SATISFIED"
    } else {
        "VIOLATED"
    };
    let mut out = vec![format!("Contract {}: {verdict}", r.contract)];
    if !r.contract_statement.is_empty() {
        out.push(format!("  Requires: {}", r.contract_statement));
    }
    out.push(format!(
        "  Container: {} (sha256 {})",
        r.container.path,
        r.container.sha256.as_deref().unwrap_or("not reported")
    ));
    out.push(format!(
        "  Source: {} ({})",
        r.source.path,
        r.source
            .identity
            .as_deref()
            .unwrap_or("identity not reported")
    ));
    out.extend(
        r.measurements
            .iter()
            .map(|m| format!("  {}: {}", m.name, m.value)),
    );
    if !r.offending.is_empty() {
        out.push(format!("  Offending tensors: {}", r.offending.len()));
        for o in r.offending.iter().take(OFFENDING_SHOWN) {
            let detail = if o.detail.is_empty() {
                String::new()
            } else {
                format!(" ({})", o.detail)
            };
            out.push(format!("    {}: {}{detail}", o.name, o.violations));
        }
        if r.offending.len() > OFFENDING_SHOWN {
            out.push(format!(
                "    ... and {} more",
                r.offending.len() - OFFENDING_SHOWN
            ));
        }
    }
    let by = r
        .verifier
        .iter()
        .map(|(k, v)| {
            format!(
                "{k}={}",
                v.as_str().map_or_else(|| v.to_string(), str::to_owned)
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    out.push(format!("  Measured by: {by}"));
    if let Some(digest) = &r.report_sha256 {
        out.push(format!("  Report sha256: {digest}"));
    }
    out.push("  Evidence: a numerical measurement of this container, not a formal proof.".into());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    #[test]
    fn a_verifier_that_hangs_is_killed_at_the_timeout() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let script = dir.path().join("hang");
        std::fs::write(&script, "#!/bin/sh\nexec sleep 30\n").unwrap();
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
        let started = Instant::now();
        let err = run_verifier(&script, "c", "a", "b", Duration::from_millis(100)).unwrap_err();
        assert!(err.to_string().contains("did not finish"), "{err}");
        assert!(started.elapsed() < Duration::from_secs(10));
    }
}
