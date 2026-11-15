//! The service's state, its start-up checks, and one request in, one response out.
//!
//! [`Service::start`] binds the policy's container and opens the audit log; [`Service::handle_line`] answers
//! one request line and appends its audit record. The operations themselves are in [`super::ops`].

use std::path::{Path, PathBuf};

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use larql_lql::Session;

use super::audit::{AuditLog, EVENT_ADMITTED, EVENT_REQUEST, EVENT_START, EVENT_STOP};
use super::paths::Workdir;
use super::policy::{CheckedPolicy, Limits};
use super::protocol::{
    budget_value, parse_request, response, ErrorKind, Failure, Meta, Op, BUDGET, HELLO, PROTOCOL,
};
use crate::commands::primary::lql_session::parse_representation_source;

/// What a response says in place of the container's path. The container is identified by its index digest.
pub const CONTAINER_LABEL: &str = "<bound container>";
/// Argument strings longer than this are logged by digest.
pub const MAX_LOGGED_STRING_BYTES: usize = 1024;
/// The prefix of the service's private staging directory.
const STAGING_PREFIX: &str = "larql-session-";
/// The container file whose digest identifies the container in the audit log.
const INDEX_JSON: &str = "index.json";
/// Environment variables recorded in the start record: the ones that change what LARQL computes.
const RECORDED_ENV_PREFIX: &str = "LARQL_";

/// What one request did, for its response and its audit record.
#[derive(Debug, Default)]
pub(super) struct Turn {
    /// An execution ran (or was refused for) an edited model.
    pub edited: bool,
    /// The execution was charged to the budget.
    pub counted: bool,
    /// The arrays returned: their digests, dtypes and shapes.
    pub arrays: Vec<Value>,
    /// The request's id and op, for the `admitted` record.
    pub id: Value,
    pub op: Value,
    /// The sequence number of the `admitted` record, when the request was charged.
    pub admitted: Option<u64>,
}

pub struct Service {
    pub(super) session: Session,
    pub(super) allowed: Vec<Op>,
    pub(super) limits: Limits,
    pub(super) budget_limit: u64,
    pub(super) budget_used: u64,
    pub(super) workdir: Workdir,
    /// The private staging directory; dropped (and removed) with the service.
    pub(super) staging: tempfile::TempDir,
    /// The same directory, for confined opens.
    pub(super) staged: Workdir,
    /// The staging directory's path, with a trailing separator: removed from everything sent back.
    pub(super) staging_prefix: String,
    /// The container's paths as the service knows them, longest first: replaced in everything sent back.
    container_texts: Vec<String>,
    /// The workdir name of the open recording, if one was begun.
    pub(super) recording: Option<String>,
    pub(super) turn: Turn,
    audit: AuditLog,
    container: PathBuf,
    container_index_sha256: String,
    /// Set when the audit log could not be written: the service stops answering.
    fatal: Option<String>,
    /// The policy's `require_capture_bound`.
    pub(super) require_capture_bound: bool,
    /// Test hook: treat every site as one the plan gives no width.
    #[cfg(test)]
    pub(super) forget_capture_width: bool,
}

fn inside(path: &Path, dir: &Path) -> bool {
    path.starts_with(dir)
}

fn canonical_dir(what: &str, path: &Path) -> Result<PathBuf, String> {
    let c = path
        .canonicalize()
        .map_err(|e| format!("the {what} {}: {e}", path.display()))?;
    if !c.is_dir() {
        return Err(format!("the {what} {} is not a directory", path.display()));
    }
    Ok(c)
}

/// The audit log's path with its directory resolved; the file itself need not exist.
fn resolved_log_path(path: &Path) -> Result<PathBuf, String> {
    let file = path
        .file_name()
        .ok_or_else(|| format!("the audit log {} names no file", path.display()))?;
    let parent = match path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        _ => Path::new("."),
    };
    Ok(canonical_dir("audit log's directory", parent)?.join(file))
}

impl Service {
    /// Check the policy, bind its container, open the audit log and record the start.
    pub fn start(checked: CheckedPolicy) -> Result<Self, String> {
        let CheckedPolicy {
            policy,
            operations,
            sha256: policy_sha256,
        } = checked;
        let container = canonical_dir("container", &policy.container)?;
        let workdir = Workdir::open(&policy.workdir)?;
        let work = workdir.canonical().to_path_buf();
        if inside(&container, &work) || inside(&work, &container) {
            return Err(format!(
                "the container {} and the workdir {} must not contain one another",
                container.display(),
                work.display()
            ));
        }
        let log_path = resolved_log_path(&policy.audit_log)?;
        if inside(&log_path, &work) || inside(&log_path, &container) {
            return Err(format!(
                "the audit log {} must be outside the workdir and the container",
                log_path.display()
            ));
        }
        let staging_base = canonical_dir("staging directory", &policy.staging_dir)?;
        if inside(&staging_base, &work)
            || inside(&work, &staging_base)
            || inside(&staging_base, &container)
        {
            return Err(format!(
                "the staging directory {} must be outside the workdir and the container, and not contain the \
                 workdir",
                staging_base.display()
            ));
        }
        let staging = tempfile::Builder::new()
            .prefix(STAGING_PREFIX)
            .tempdir_in(&staging_base)
            .map_err(|e| format!("create the staging directory: {e}"))?;
        let staging_path = canonical_dir("staging directory", staging.path())?;
        let staging_prefix = format!("{}{}", staging_path.display(), std::path::MAIN_SEPARATOR);
        let staged = Workdir::open(&staging_path)?;
        let mut container_texts: Vec<String> = [container.clone(), policy.container.clone()]
            .iter()
            .map(|p| p.display().to_string())
            .filter(|t| !t.is_empty())
            .collect();
        container_texts.sort_by_key(|t| std::cmp::Reverse(t.len()));
        container_texts.dedup();

        let index = std::fs::read(container.join(INDEX_JSON)).map_err(|e| {
            format!(
                "read {} of the container {}: {e}",
                INDEX_JSON,
                container.display()
            )
        })?;
        let container_index_sha256 = format!("{:x}", Sha256::digest(&index));

        let session = bind(&container, policy.representation_source.as_deref())?;
        let (audit, resumed) = AuditLog::open(&log_path, policy.resume, &container_index_sha256)?;

        let mut service = Self {
            session,
            allowed: operations,
            limits: policy.limits.clone(),
            budget_limit: policy.budget.edited_executions,
            budget_used: resumed.budget_used,
            workdir,
            staging,
            staged,
            staging_prefix,
            container_texts,
            recording: None,
            turn: Turn::default(),
            audit,
            container,
            container_index_sha256,
            fatal: None,
            require_capture_bound: policy.require_capture_bound,
            #[cfg(test)]
            forget_capture_width: false,
        };
        let env: serde_json::Map<String, Value> = std::env::vars()
            .filter(|(k, _)| k.starts_with(RECORDED_ENV_PREFIX))
            .map(|(k, v)| (k, Value::String(v)))
            .collect();
        let start = json!({
            "protocol": PROTOCOL,
            "larql_version": env!("CARGO_PKG_VERSION"),
            "pid": std::process::id(),
            "policy_sha256": policy_sha256,
            "container": service.container.display().to_string(),
            "container_index_sha256": service.container_index_sha256,
            "workdir": work.display().to_string(),
            "staging": staging_path.display().to_string(),
            "operations": service.allowed_names(),
            "budget_limit": service.budget_limit,
            "limits": service.limits,
            "require_capture_bound": service.require_capture_bound,
            "resumed": resumed.next_seq > 1,
            "env": env,
        });
        service
            .audit
            .append(EVENT_START, service.budget_used, start)
            .map_err(|e| format!("write the audit log: {e}"))?;
        Ok(service)
    }

    fn allowed_names(&self) -> Vec<&'static str> {
        self.allowed.iter().map(|op| op.name()).collect()
    }

    pub(super) fn remaining(&self) -> u64 {
        self.budget_limit.saturating_sub(self.budget_used)
    }

    /// The reason the service stopped answering, if the audit log failed.
    pub fn fatal(&self) -> Option<&str> {
        self.fatal.as_deref()
    }

    /// Put the service in the state an unwritable audit log leaves it in.
    #[cfg(test)]
    pub(super) fn mark_fatal_for_test(&mut self, why: &str) {
        self.fatal = Some(why.to_string());
    }

    /// Append a record that is not a request (a connection opened or closed, a stop).
    pub fn record(&mut self, event: &str, fields: Value) {
        if let Err(e) = self.audit.append(event, self.budget_used, fields) {
            self.fatal = Some(format!("the audit log could not be written: {e}"));
        }
    }

    /// Record the stop.
    pub fn stop(&mut self, reason: &str) {
        self.record(EVENT_STOP, json!({"reason": reason}));
    }

    /// Answer one request line (without its newline) and append its audit record.
    pub fn handle_line(&mut self, line: &[u8]) -> Vec<u8> {
        self.turn = Turn::default();
        let (id, op, args, outcome) = match parse_request(line) {
            Err((id, failure)) => {
                // What could not be read as a request is logged by digest, never verbatim.
                let digest = json!({
                    "unparsed_line_sha256": format!("{:x}", Sha256::digest(line)),
                    "bytes": line.len(),
                });
                (id, Value::Null, digest, Err(failure))
            }
            Ok(request) => {
                self.turn.id = request.id.clone();
                self.turn.op = Value::String(request.op.clone());
                let outcome = match &self.fatal {
                    Some(why) => Err(Failure::failed(why.clone())),
                    None => self.dispatch(&request.op, &request.args),
                };
                (
                    request.id,
                    Value::String(request.op),
                    summarise_args(&request.args),
                    outcome,
                )
            }
        };
        self.finish(id, op, args, outcome)
    }

    /// Answer a line that could not be read as a request at all (one longer than the policy allows).
    pub fn refuse_line(&mut self, failure: Failure) -> Vec<u8> {
        self.turn = Turn::default();
        self.finish(Value::Null, Value::Null, Value::Null, Err(failure))
    }

    /// The response to a request, its audit record appended.
    fn finish(
        &mut self,
        id: Value,
        op: Value,
        args: Value,
        outcome: Result<Value, Failure>,
    ) -> Vec<u8> {
        let outcome = outcome.map(|v| self.unstage(v)).map_err(|f| Failure {
            reason: self.unstage_text(&f.reason),
            ..f
        });
        let meta = Meta {
            seq: self.audit.next_seq(),
            budget: budget_value(self.budget_limit, self.budget_used),
            edited: self.turn.edited,
            counted: self.turn.counted,
        };
        let line = response(&id, &meta, &outcome);
        let record = json!({
            "id": id,
            "op": op,
            "args": args,
            "ok": outcome.is_ok(),
            "error": outcome.as_ref().err().map(|f| json!({"kind": f.kind.label(), "reason": f.reason})),
            "edited": self.turn.edited,
            "counted": self.turn.counted,
            "admitted_seq": self.turn.admitted,
            "outcome": outcome_label(self.turn.admitted.is_some(), outcome.is_ok()),
            "budget_remaining": self.remaining(),
            "arrays": self.turn.arrays,
            "response_sha256": format!("{:x}", Sha256::digest(&line)),
        });
        match self.audit.append(EVENT_REQUEST, self.budget_used, record) {
            Ok(_) => line,
            Err(e) => {
                // Nothing is answered that the log does not hold.
                let why = format!("the audit log could not be written: {e}");
                self.fatal = Some(why.clone());
                response(&id, &meta, &Err(Failure::failed(why)))
            }
        }
    }

    fn dispatch(&mut self, name: &str, args: &Value) -> Result<Value, Failure> {
        match name {
            HELLO => return Ok(self.hello()),
            BUDGET => return Ok(budget_value(self.budget_limit, self.budget_used)),
            _ => {}
        }
        let op = Op::from_name(name).ok_or_else(|| {
            Failure::refused(format!(
                "`{name}` is not an operation of this service; it answers {HELLO}, {BUDGET} and {}",
                self.allowed_names().join(", ")
            ))
        })?;
        if !self.allowed.contains(&op) {
            return Err(Failure::refused(format!(
                "the policy does not allow `{name}`; it allows {}",
                self.allowed_names().join(", ")
            )));
        }
        self.run(op, name, args)
    }

    fn hello(&self) -> Value {
        json!({
            "protocol": PROTOCOL,
            "service": "larql serve-session",
            "larql_version": env!("CARGO_PKG_VERSION"),
            "container_index_sha256": self.container_index_sha256,
            "operations": self.allowed_names(),
            "limits": self.limits,
            "require_capture_bound": self.require_capture_bound,
            "budget": budget_value(self.budget_limit, self.budget_used),
        })
    }

    /// Whether what the session executes now is not the stored model: a patch applied, a weight op staged, or
    /// an override in the session overlay.
    pub(super) fn model_is_edited(&self) -> Result<bool, Failure> {
        let pending = self.session.has_pending_edits().map_err(Failure::failed)?;
        let report = self.session.patches().map_err(Failure::failed)?;
        Ok(pending || !report.applied.is_empty() || report.weight_ops > 0)
    }

    /// Gate an execution: when the model is edited, refuse it once the budget is spent, and otherwise charge
    /// it now, before it runs, so nothing that happens during or after the run leaves it uncharged. Returns
    /// whether the model is edited.
    pub(super) fn gate(&mut self) -> Result<bool, Failure> {
        let edited = self.model_is_edited()?;
        self.turn.edited = edited;
        if edited {
            self.charge()?;
        }
        Ok(edited)
    }

    pub(super) fn exhausted(&self) -> Failure {
        Failure::new(
            ErrorKind::BudgetExhausted,
            format!(
                "the budget of {} edited-model execution(s) is spent; executions of the base model \
                 (no patch applied, no weight op staged) are still answered",
                self.budget_limit
            ),
        )
    }

    /// Charge one execution of an edited model, or refuse it when the budget is spent. The charge is made
    /// durable first: an `admitted` record carrying the budget after it is appended and synced, and only then
    /// may the model run. A resumed log recounts the budget from these records.
    pub(super) fn charge(&mut self) -> Result<(), Failure> {
        self.turn.edited = true;
        if self.turn.counted {
            return Ok(());
        }
        if self.remaining() == 0 {
            return Err(self.exhausted());
        }
        let used = self.budget_used + 1;
        let record = json!({
            "id": self.turn.id,
            "op": self.turn.op,
            "budget_remaining": self.budget_limit.saturating_sub(used),
        });
        match self.audit.append(EVENT_ADMITTED, used, record) {
            Ok(seq) => self.turn.admitted = Some(seq),
            Err(e) => {
                let why = format!("the audit log could not be written: {e}");
                self.fatal = Some(why.clone());
                return Err(Failure::failed(why));
            }
        }
        self.budget_used = used;
        self.turn.counted = true;
        Ok(())
    }

    /// The staging path of a workdir name.
    pub(super) fn staged_path(&self, name: &str) -> PathBuf {
        self.staging.path().join(name)
    }

    /// `text` with every mention of the staging directory removed, so a staged copy reads as its workdir
    /// name, and the container's path replaced by [`CONTAINER_LABEL`].
    pub(super) fn unstage_text(&self, text: &str) -> String {
        let raw = format!(
            "{}{}",
            self.staging.path().display(),
            std::path::MAIN_SEPARATOR
        );
        let mut out = text.replace(&self.staging_prefix, "").replace(&raw, "");
        for path in &self.container_texts {
            out = out.replace(path.as_str(), CONTAINER_LABEL);
        }
        out
    }

    fn unstage(&self, value: Value) -> Value {
        match value {
            Value::String(s) => Value::String(self.unstage_text(&s)),
            Value::Array(items) => {
                Value::Array(items.into_iter().map(|v| self.unstage(v)).collect())
            }
            Value::Object(map) => {
                Value::Object(map.into_iter().map(|(k, v)| (k, self.unstage(v))).collect())
            }
            other => other,
        }
    }
}

/// What a request record says happened: an edited execution that was admitted (charged) and then
/// `completed` or `failed`, or a request that was `answered` or `rejected` without one.
fn outcome_label(admitted: bool, ok: bool) -> &'static str {
    match (admitted, ok) {
        (true, true) => "completed",
        (true, false) => "failed",
        (false, true) => "answered",
        (false, false) => "rejected",
    }
}

/// Bind `container` in a fresh session, refusing anything but VINDEX3.
fn bind(container: &Path, representation_source: Option<&str>) -> Result<Session, String> {
    let mut session = Session::new();
    if let Some(spec) = representation_source {
        session.set_representation_source(parse_representation_source(spec)?);
    }
    let stmt = larql_lql::Statement::Use {
        target: larql_lql::ast::UseTarget::Vindex(container.display().to_string()),
    };
    session
        .execute(&stmt)
        .map_err(|e| format!("bind the container {}: {e}", container.display()))?;
    match session.local_artifact() {
        Some((_, larql_vindex::format::generation::ContainerGeneration::V3)) => Ok(session),
        _ => Err(format!(
            "the container {} is not a VINDEX3 container",
            container.display()
        )),
    }
}

/// The request's arguments as the audit log records them: arrays replaced by their digest and size.
fn summarise_args(args: &Value) -> Value {
    match args {
        Value::Object(map) => {
            if let Some(Value::String(text)) = map.get("npy_base64") {
                use base64::Engine;
                let digest = match base64::engine::general_purpose::STANDARD.decode(text) {
                    Ok(bytes) => {
                        json!({"npy_sha256": format!("{:x}", Sha256::digest(&bytes)), "bytes": bytes.len()})
                    }
                    Err(_) => json!({"undecodable_base64_chars": text.len()}),
                };
                return digest;
            }
            Value::Object(
                map.iter()
                    .map(|(k, v)| (k.clone(), summarise_args(v)))
                    .collect(),
            )
        }
        Value::Array(items) => Value::Array(items.iter().map(summarise_args).collect()),
        Value::String(text) if text.len() > MAX_LOGGED_STRING_BYTES => json!({
            "sha256": format!("{:x}", Sha256::digest(text.as_bytes())),
            "bytes": text.len(),
        }),
        other => other.clone(),
    }
}
