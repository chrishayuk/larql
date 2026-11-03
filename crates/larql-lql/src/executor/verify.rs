//! `VERIFY REPRESENTATION`: ask a separate verifier whether one container
//! meets a numerical contract, and report what it measured.
//!
//! **LQL does not evaluate the contract.** A check written here would be
//! LARQL marking its own work, so the measurement belongs to a separate
//! program with no LARQL code in it: by default the repository's own
//! `verifier/`, but any program speaking the protocol will do. This module runs that program, checks
//! that its answer is well formed and about the question asked, and renders
//! it. No verifier means a refusal, never a pass.
//!
//! The statement is read-only. It writes nothing and binds nothing. A result
//! is a numerical measurement of the bytes of one container, not a proof about
//! the encoder that produced them, and the output says so.
//!
//! ## Protocol `larql-verify-contract/1`
//!
//! Specified in `docs/verification/contract-protocol.md`. In short:
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
const VERIFIER_NAME: &str = "larql-verifier";
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
        let verifier = find_verifier(
            self.verifier.clone(),
            std::env::var_os(VERIFIER_ENV).map(PathBuf::from),
            std::env::var_os("PATH"),
            std::env::current_exe().ok(),
        )
        .ok_or_else(|| {
            LqlError::Execution(format!(
                "VERIFY REPRESENTATION needs an independent verifier and found none: install \
                 verifier/ from this repository, or set {VERIFIER_ENV}. LARQL does not evaluate \
                 contracts itself."
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

/// Which verifier to run: the embedding program's choice, then the environment's, then
/// `larql-verifier` on `PATH`, then a checkout's own, in `verifier/.venv` in a directory above the
/// running binary. An empty choice turns discovery off. The current directory is never
/// searched, so a statement can't run a program from wherever it happens to be run.
fn find_verifier(
    explicit: Option<PathBuf>,
    env: Option<PathBuf>,
    path: Option<std::ffi::OsString>,
    exe: Option<PathBuf>,
) -> Option<PathBuf> {
    if let Some(chosen) = explicit.or(env) {
        return (!chosen.as_os_str().is_empty()).then_some(chosen);
    }
    let executable = format!("{VERIFIER_NAME}{}", std::env::consts::EXE_SUFFIX);
    let on_path = path
        .iter()
        .flat_map(std::env::split_paths)
        .filter(|d| d.is_absolute());
    let in_checkout = exe
        .iter()
        .flat_map(|e| {
            e.ancestors()
                .skip(1)
                .map(Path::to_path_buf)
                .collect::<Vec<_>>()
        })
        .map(|d| d.join("verifier").join(".venv").join("bin"));
    on_path
        .chain(in_checkout)
        .map(|d| d.join(&executable))
        .find(|candidate| candidate.is_file())
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
    out.push(format!(
        "  Measured by: {by} (as reported by the verifier, not authenticated)"
    ));
    if let Some(digest) = &r.report_sha256 {
        out.push(format!("  Report sha256: {digest}"));
    }
    out.push("  Evidence: a numerical measurement of this container, not a formal proof.".into());
    out
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    struct Tree {
        _dir: tempfile::TempDir,
        root: PathBuf,
    }

    /// `root/bin/larql-verifier` for `PATH`, `root/checkout/verifier/.venv/bin/larql-verifier`
    /// for a checkout, and `root/checkout/target/debug/larql` as the running binary.
    fn tree() -> Tree {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        for d in [
            "bin",
            "checkout/verifier/.venv/bin",
            "checkout/target/debug",
        ] {
            std::fs::create_dir_all(root.join(d)).unwrap();
        }
        std::fs::write(root.join("bin/larql-verifier"), "").unwrap();
        std::fs::write(root.join("checkout/verifier/.venv/bin/larql-verifier"), "").unwrap();
        Tree { _dir: dir, root }
    }

    #[test]
    fn verifier_discovery_follows_its_order() {
        let t = tree();
        let exe = Some(t.root.join("checkout/target/debug/larql"));
        let path = Some(t.root.join("bin").into_os_string());
        let on_path = t.root.join("bin/larql-verifier");
        let in_checkout = t.root.join("checkout/verifier/.venv/bin/larql-verifier");
        let chosen = || Some(PathBuf::from("/chosen"));
        let find = find_verifier;

        assert_eq!(
            find(chosen(), Some("/env".into()), path.clone(), exe.clone()),
            chosen()
        );
        assert_eq!(find(None, chosen(), path.clone(), exe.clone()), chosen());
        assert_eq!(find(None, None, path.clone(), exe.clone()), Some(on_path));
        assert_eq!(find(None, None, None, exe.clone()), Some(in_checkout));
        assert_eq!(find(None, None, None, None), None);
    }

    #[test]
    fn an_empty_choice_turns_discovery_off() {
        let t = tree();
        let exe = Some(t.root.join("checkout/target/debug/larql"));
        let path = Some(t.root.join("bin").into_os_string());
        assert_eq!(
            find_verifier(Some("".into()), None, path.clone(), exe.clone()),
            None
        );
        assert_eq!(find_verifier(None, Some("".into()), path, exe), None);
    }

    /// A relative `PATH` entry such as `.` would run a program from the current directory.
    #[test]
    fn a_relative_path_entry_is_never_searched() {
        let dir = tempfile::tempdir_in(".").unwrap();
        std::fs::write(dir.path().join("larql-verifier"), "").unwrap();
        let name = dir.path().file_name().unwrap().to_owned();
        assert!(
            Path::new(&name).join("larql-verifier").is_file(),
            "the control"
        );
        assert_eq!(find_verifier(None, None, Some(name), None), None);
    }

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
