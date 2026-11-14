//! `VERIFY REPRESENTATION` and `VERIFY OPERATOR`: ask a separate verifier whether
//! one container, or one recorded kernel call, meets a numerical contract, and
//! report what it measured.
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
//!
//! ## Protocol `larql-verify-operator/1`
//!
//! ```text
//! <verifier> operator --contract <id> --operator <tensor> --witness <file> --container <dir>
//! ```
//!
//! `VERIFY OPERATOR` never runs the engine. It reads a witness an earlier run
//! wrote, so that LARQL isn't asked whether its own computation was correct.
//! A witness is still the engine's report, and the output says so.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use serde::Deserialize;

use super::Session;
use crate::error::LqlError;

const SCHEMA: &str = "larql-verify-contract/1";
const OPERATOR_SCHEMA: &str = "larql-verify-operator/1";
const PATCH_SCHEMA: &str = "larql-verify-patch/1";
const WITNESS_BASIS: &str = "engine_reported";
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
struct OperatorResult {
    schema: String,
    contract: String,
    #[serde(default)]
    contract_statement: String,
    evidence_kind: String,
    status: String,
    violations: u64,
    operator: String,
    witness: Witness,
    container: Identified,
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
struct PatchResult {
    schema: String,
    contract: String,
    #[serde(default)]
    contract_statement: String,
    evidence_kind: String,
    status: String,
    violations: u64,
    patch: Identified,
    base: Identified,
    container: Identified,
    #[serde(default)]
    recomputed_from: String,
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
struct Witness {
    path: String,
    sha256: Option<String>,
    basis: String,
    #[serde(default)]
    reported: serde_json::Map<String, serde_json::Value>,
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
        let verifier = self.discover_verifier("VERIFY REPRESENTATION")?;
        let container = existing_dir(container, "container")?;
        let source = existing_dir(source, "source")?;
        let stdout = run_verifier(
            &verifier,
            &[
                "contract",
                "--contract",
                contract,
                "--container",
                &container,
                "--source",
                &source,
            ],
            TIMEOUT,
        )?;
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

    pub(crate) fn exec_verify_operator(
        &mut self,
        operator: &str,
        contract: &str,
        witness: &str,
        container: &str,
    ) -> Result<Vec<String>, LqlError> {
        let verifier = self.discover_verifier("VERIFY OPERATOR")?;
        let witness = existing_file(witness, "witness")?;
        let container = existing_dir(container, "container")?;
        let stdout = run_verifier(
            &verifier,
            &[
                "operator",
                "--contract",
                contract,
                "--operator",
                operator,
                "--witness",
                &witness,
                "--container",
                &container,
            ],
            TIMEOUT,
        )?;
        let result = parse_operator_result(&stdout, operator, contract, &witness, &container)?;
        if result.status == "could_not_evaluate" {
            return Err(LqlError::Execution(format!(
                "contract {contract} could not be evaluated for {operator}: {}",
                result.reasons.join("; ")
            )));
        }
        if result.status == "violated" {
            self.contract_violations += 1;
        }
        Ok(render_operator(&result))
    }

    pub(crate) fn exec_verify_patch(
        &mut self,
        patch: &str,
        contract: &str,
        base: &str,
        container: &str,
    ) -> Result<Vec<String>, LqlError> {
        let verifier = self.discover_verifier("VERIFY PATCH")?;
        let patch = existing_file(patch, "patch")?;
        let base = existing_dir(base, "base")?;
        let container = existing_dir(container, "container")?;
        let stdout = run_verifier(
            &verifier,
            &[
                "patch",
                "--contract",
                contract,
                "--patch",
                &patch,
                "--base",
                &base,
                "--container",
                &container,
            ],
            TIMEOUT,
        )?;
        let result = parse_patch_result(&stdout, contract, &patch, &base, &container)?;
        if result.status == "could_not_evaluate" {
            return Err(LqlError::Execution(format!(
                "contract {contract} could not be evaluated for {patch}: {}",
                result.reasons.join("; ")
            )));
        }
        if result.status == "violated" {
            self.contract_violations += 1;
        }
        Ok(render_patch(&result))
    }

    fn discover_verifier(&self, statement: &str) -> Result<PathBuf, LqlError> {
        find_verifier(
            self.verifier.clone(),
            std::env::var_os(VERIFIER_ENV).map(PathBuf::from),
            std::env::var_os("PATH"),
            std::env::current_exe().ok(),
        )
        .ok_or_else(|| {
            LqlError::Execution(format!(
                "{statement} needs an independent verifier and found none: install \
                 verifier/ from this repository, or set {VERIFIER_ENV}. LARQL does not evaluate \
                 contracts itself."
            ))
        })
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
        .map_err(|e| LqlError::exec(format!("VERIFY {what} {path}"), e))?;
    if !canonical.is_dir() {
        return Err(LqlError::Execution(format!(
            "VERIFY {what} {path} is not a directory"
        )));
    }
    Ok(canonical.to_string_lossy().into_owned())
}

fn existing_file(path: &str, what: &str) -> Result<String, LqlError> {
    let canonical = std::fs::canonicalize(path)
        .map_err(|e| LqlError::exec(format!("VERIFY {what} {path}"), e))?;
    if !canonical.is_file() {
        return Err(LqlError::Execution(format!(
            "VERIFY {what} {path} is not a file"
        )));
    }
    Ok(canonical.to_string_lossy().into_owned())
}

/// How many times a spawn that found the executable busy is tried in all.
const BUSY_SPAWN_ATTEMPTS: u32 = 40;
/// The wait before the first retry; each later wait is one step longer.
const BUSY_SPAWN_BACKOFF: Duration = Duration::from_millis(5);

/// Spawn, retrying while the executable is "text file busy".
///
/// An executable cannot be run while any process holds it open for writing.
/// That is true for an instant after a verifier is written, even by this
/// process: a `fork` on another thread copies the still-open descriptor
/// into a child, which holds it until its own `exec`. The condition clears
/// by itself, so the spawn is retried over about four seconds before the
/// error is reported. Every other error is returned at once.
fn spawn_when_not_busy(
    mut spawn: impl FnMut() -> std::io::Result<std::process::Child>,
) -> std::io::Result<std::process::Child> {
    let mut attempt = 1;
    loop {
        match spawn() {
            Err(e)
                if e.kind() == std::io::ErrorKind::ExecutableFileBusy
                    && attempt < BUSY_SPAWN_ATTEMPTS =>
            {
                std::thread::sleep(BUSY_SPAWN_BACKOFF * attempt);
                attempt += 1;
            }
            result => return result,
        }
    }
}

/// Run the verifier, without a shell, and return its stdout. Both pipes are
/// drained on their own threads, so a large answer cannot block the child.
fn run_verifier(verifier: &Path, args: &[&str], timeout: Duration) -> Result<String, LqlError> {
    let shown = verifier.display();
    let mut child = spawn_when_not_busy(|| {
        Command::new(verifier)
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
    })
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
    check_consistent(&r.status, r.violations, &r.offending, &r.reasons).map_err(bad)?;
    Ok(r)
}

/// A verdict that contradicts itself is refused, whichever statement it answers.
fn check_consistent(
    status: &str,
    violations: u64,
    offending: &[Offending],
    reasons: &[String],
) -> Result<(), String> {
    let listed: u64 = offending.iter().map(|o| o.violations).sum();
    match status {
        "satisfied" if violations != 0 || !offending.is_empty() => {
            Err(format!("satisfied, with {violations} violations"))
        }
        "violated" if violations == 0 || listed != violations => Err(format!(
            "violated, with {violations} violations but {listed} listed"
        )),
        "could_not_evaluate" if reasons.is_empty() => {
            Err("could_not_evaluate, with no reason".into())
        }
        "satisfied" | "violated" | "could_not_evaluate" => Ok(()),
        other => Err(format!("unknown status {other:?}")),
    }
}

/// The same refusals as [`parse_result`], for a question about one recorded call.
fn parse_operator_result(
    stdout: &str,
    operator: &str,
    contract: &str,
    witness: &str,
    container: &str,
) -> Result<OperatorResult, LqlError> {
    let bad =
        |why: String| LqlError::Execution(format!("the verifier's answer was refused: {why}"));
    let r: OperatorResult =
        serde_json::from_str(stdout).map_err(|e| bad(format!("not a valid result: {e}")))?;
    let mismatch = |field: &str, got: &str, want: &str| {
        (got != want).then(|| bad(format!("its {field} is {got:?}, not {want:?}")))
    };
    let first = mismatch("schema", &r.schema, OPERATOR_SCHEMA)
        .or_else(|| mismatch("contract", &r.contract, contract))
        .or_else(|| mismatch("operator", &r.operator, operator))
        .or_else(|| mismatch("witness", &r.witness.path, witness))
        .or_else(|| mismatch("container", &r.container.path, container))
        .or_else(|| mismatch("evidence kind", &r.evidence_kind, EVIDENCE_KIND))
        .or_else(|| mismatch("witness basis", &r.witness.basis, WITNESS_BASIS));
    if let Some(e) = first {
        return Err(e);
    }
    check_consistent(&r.status, r.violations, &r.offending, &r.reasons).map_err(bad)?;
    Ok(r)
}

/// The same refusals as [`parse_result`], for a question about what a patch declares of a base.
fn parse_patch_result(
    stdout: &str,
    contract: &str,
    patch: &str,
    base: &str,
    container: &str,
) -> Result<PatchResult, LqlError> {
    let bad =
        |why: String| LqlError::Execution(format!("the verifier's answer was refused: {why}"));
    let r: PatchResult =
        serde_json::from_str(stdout).map_err(|e| bad(format!("not a valid result: {e}")))?;
    let mismatch = |field: &str, got: &str, want: &str| {
        (got != want).then(|| bad(format!("its {field} is {got:?}, not {want:?}")))
    };
    let first = mismatch("schema", &r.schema, PATCH_SCHEMA)
        .or_else(|| mismatch("contract", &r.contract, contract))
        .or_else(|| mismatch("patch", &r.patch.path, patch))
        .or_else(|| mismatch("base", &r.base.path, base))
        .or_else(|| mismatch("container", &r.container.path, container))
        .or_else(|| mismatch("evidence kind", &r.evidence_kind, EVIDENCE_KIND));
    if let Some(e) = first {
        return Err(e);
    }
    check_consistent(&r.status, r.violations, &r.offending, &r.reasons).map_err(bad)?;
    Ok(r)
}

fn render_patch(r: &PatchResult) -> Vec<String> {
    let verdict = if r.status == "satisfied" {
        "SATISFIED"
    } else {
        "VIOLATED"
    };
    let mut out = vec![format!("Patch contract {}: {verdict}", r.contract)];
    if !r.contract_statement.is_empty() {
        out.push(format!("  Requires: {}", r.contract_statement));
    }
    out.push(format!(
        "  Patch: {} (sha256 {})",
        r.patch.path,
        r.patch.sha256.as_deref().unwrap_or("not reported")
    ));
    out.push(format!(
        "  Base: {} (sha256 {})",
        r.base.path,
        r.base.sha256.as_deref().unwrap_or("not reported")
    ));
    out.push(format!(
        "  Container: {} (sha256 {})",
        r.container.path,
        r.container.sha256.as_deref().unwrap_or("not reported")
    ));
    if !r.recomputed_from.is_empty() {
        out.push(format!("  Recomputed from: {}", r.recomputed_from));
    }
    render_findings(
        &mut out,
        &r.measurements,
        &r.offending,
        &r.verifier,
        &r.report_sha256,
    );
    out.push(
        "  Evidence: a numerical measurement of these two containers, not a formal proof.".into(),
    );
    out
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
    render_findings(
        &mut out,
        &r.measurements,
        &r.offending,
        &r.verifier,
        &r.report_sha256,
    );
    out.push("  Evidence: a numerical measurement of this container, not a formal proof.".into());
    out
}

fn render_operator(r: &OperatorResult) -> Vec<String> {
    let verdict = if r.status == "satisfied" {
        "SATISFIED"
    } else {
        "VIOLATED"
    };
    let mut out = vec![format!(
        "Operator {}, contract {}: {verdict}",
        r.operator, r.contract
    )];
    if !r.contract_statement.is_empty() {
        out.push(format!("  Requires: {}", r.contract_statement));
    }
    out.push(format!(
        "  Witness: {} (sha256 {}), reported by the engine, not authenticated",
        r.witness.path,
        r.witness.sha256.as_deref().unwrap_or("not reported")
    ));
    if !r.witness.reported.is_empty() {
        out.push(format!(
            "  The witness reports: {}",
            key_values(&r.witness.reported)
        ));
    }
    out.push(format!(
        "  Container: {} (sha256 {})",
        r.container.path,
        r.container.sha256.as_deref().unwrap_or("not reported")
    ));
    render_findings(
        &mut out,
        &r.measurements,
        &r.offending,
        &r.verifier,
        &r.report_sha256,
    );
    out.push(
        "  Evidence: a numerical measurement of this one recorded call, not a formal proof.".into(),
    );
    out
}

fn key_values(map: &serde_json::Map<String, serde_json::Value>) -> String {
    map.iter()
        .map(|(k, v)| {
            format!(
                "{k}={}",
                v.as_str().map_or_else(|| v.to_string(), str::to_owned)
            )
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// What both statements report the same way.
fn render_findings(
    out: &mut Vec<String>,
    measurements: &[Measurement],
    offending: &[Offending],
    verifier: &serde_json::Map<String, serde_json::Value>,
    report_sha256: &Option<String>,
) {
    out.extend(
        measurements
            .iter()
            .map(|m| format!("  {}: {}", m.name, m.value)),
    );
    if !offending.is_empty() {
        out.push(format!("  Offending tensors: {}", offending.len()));
        for o in offending.iter().take(OFFENDING_SHOWN) {
            let detail = if o.detail.is_empty() {
                String::new()
            } else {
                format!(" ({})", o.detail)
            };
            out.push(format!("    {}: {}{detail}", o.name, o.violations));
        }
        if offending.len() > OFFENDING_SHOWN {
            out.push(format!(
                "    ... and {} more",
                offending.len() - OFFENDING_SHOWN
            ));
        }
    }
    out.push(format!(
        "  Measured by: {} (as reported by the verifier, not authenticated)",
        key_values(verifier)
    ));
    if let Some(digest) = report_sha256 {
        out.push(format!("  Report sha256: {digest}"));
    }
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
        let err = run_verifier(&script, &["c"], Duration::from_millis(100)).unwrap_err();
        assert!(err.to_string().contains("did not finish"), "{err}");
        assert!(started.elapsed() < Duration::from_secs(10));
    }
}

#[cfg(test)]
mod busy_spawn_tests {
    //! `spawn_when_not_busy` against a scripted spawn: the retry is a
    //! property of the helper, and the condition it waits out (another
    //! process holding the executable open for writing) cannot be staged
    //! the same way on every platform.

    use super::*;
    use std::io::{Error, ErrorKind};

    /// A spawn that reports "text file busy" `busy_first` times and then
    /// starts a real, trivial child. Returns the outcome and the number of
    /// attempts made.
    fn scripted(busy_first: u32, then: ErrorKind) -> (std::io::Result<()>, u32) {
        let mut attempts = 0;
        let result = spawn_when_not_busy(|| {
            attempts += 1;
            if attempts <= busy_first {
                return Err(Error::from(ErrorKind::ExecutableFileBusy));
            }
            if then != ErrorKind::Other {
                return Err(Error::from(then));
            }
            Command::new(std::env::current_exe()?)
                .arg("--list")
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
        })
        .map(|mut child| {
            let _ = child.wait();
        });
        (result, attempts)
    }

    #[test]
    fn a_spawn_that_is_busy_twice_succeeds_on_the_third_attempt() {
        let (result, attempts) = scripted(2, ErrorKind::Other);

        result.expect("the spawn succeeds once the file is free");
        assert_eq!(attempts, 3);
    }

    #[test]
    fn any_other_error_is_returned_at_once() {
        let (result, attempts) = scripted(0, ErrorKind::NotFound);

        assert_eq!(result.unwrap_err().kind(), ErrorKind::NotFound);
        assert_eq!(attempts, 1);
    }

    #[test]
    fn an_error_after_a_busy_attempt_is_returned_without_more_retries() {
        let (result, attempts) = scripted(1, ErrorKind::PermissionDenied);

        assert_eq!(result.unwrap_err().kind(), ErrorKind::PermissionDenied);
        assert_eq!(attempts, 2);
    }

    #[test]
    fn a_file_that_stays_busy_is_reported_after_the_last_attempt() {
        let (result, attempts) = scripted(u32::MAX, ErrorKind::Other);

        assert_eq!(result.unwrap_err().kind(), ErrorKind::ExecutableFileBusy);
        assert_eq!(attempts, BUSY_SPAWN_ATTEMPTS);
    }
}
