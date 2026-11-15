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
/// A request's arguments, summarised, longer than this as JSON are logged by digest and length alone.
pub const MAX_LOGGED_ARGS_BYTES: usize = 4096;
/// What the client is told in place of a LARQL reason that names a path the client must not learn about.
pub const WITHHELD_PATH_REASON: &str =
    "LARQL refused the request; its reason names a path and is withheld (it is in the audit log)";
/// The bytes of the instance identifier the service draws at start.
const INSTANCE_ID_BYTES: usize = 8;
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
    /// The request's id (the client's, echoed) and op, for the `admitted` record.
    pub id: Value,
    pub op: Value,
    /// The service's own identifier of the request: unique for the service's lifetime and across resumes.
    pub request_uid: Value,
    /// The arguments as the log records them, and the SHA-256 of the arguments as sent.
    pub args: Value,
    pub args_sha256: Value,
    /// The client's strings that look like paths: a LARQL reason that echoes one is withheld.
    pub path_like: Vec<String>,
    /// The sequence number of the `admitted` record, when the request was charged.
    pub admitted: Option<u64>,
}

/// A copy of a workdir file in the private staging directory, under a name unique to the service's lifetime.
#[derive(Debug, Clone)]
pub(super) struct StagedCopy {
    /// The workdir name the client used.
    pub name: String,
    /// The staged file's name in the staging directory, and its path as LARQL was given it.
    pub file: String,
    pub path: String,
    pub bytes: u64,
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
    /// The staging directory's path, with a trailing separator.
    pub(super) staging_prefix: String,
    /// The container's paths as the service knows them: a LARQL reason naming one is withheld.
    container_texts: Vec<String>,
    /// The open recording: the workdir name and the staged file LARQL saves into.
    pub(super) recording: Option<StagedCopy>,
    /// The staged copies of the patches applied now, in the order applied.
    pub(super) applied: Vec<StagedCopy>,
    /// Staged file names are `<counter>.vlp`: never reused in the service's lifetime.
    pub(super) staged_counter: u64,
    /// Drawn at start: with a counter, the request identifiers.
    instance_id: String,
    requests: u64,
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
    /// Test hook: the edited predicate always answers "not edited".
    #[cfg(test)]
    pub(super) blind_predicate: bool,
    /// Test hook: a capture fails, after admission, with this LARQL reason instead of running.
    #[cfg(test)]
    pub(super) fail_capture_with: Option<String>,
}

/// Eight random bytes from the OS, as hex.
fn instance_id() -> Result<String, String> {
    use std::io::Read;
    let mut bytes = [0u8; INSTANCE_ID_BYTES];
    std::fs::File::open("/dev/urandom")
        .and_then(|mut f| f.read_exact(&mut bytes))
        .map_err(|e| format!("draw the instance identifier: {e}"))?;
    Ok(bytes.iter().map(|b| format!("{b:02x}")).collect())
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
        // LARQL is given the canonical path, so that is the one its text can name.
        let container_texts = vec![container.display().to_string()];
        let instance_id = instance_id()?;

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
            applied: Vec::new(),
            staged_counter: 0,
            instance_id,
            requests: 0,
            turn: Turn::default(),
            audit,
            container,
            container_index_sha256,
            fatal: None,
            require_capture_bound: policy.require_capture_bound,
            #[cfg(test)]
            forget_capture_width: false,
            #[cfg(test)]
            blind_predicate: false,
            #[cfg(test)]
            fail_capture_with: None,
        };
        let env: serde_json::Map<String, Value> = std::env::vars()
            .filter(|(k, _)| k.starts_with(RECORDED_ENV_PREFIX))
            .map(|(k, v)| (k, Value::String(v)))
            .collect();
        let start = json!({
            "protocol": PROTOCOL,
            "larql_version": env!("CARGO_PKG_VERSION"),
            "pid": std::process::id(),
            "instance_id": service.instance_id,
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
        self.begin_turn();
        let (id, op, outcome) = match parse_request(line) {
            Err((id, failure)) => {
                // What could not be read as a request is logged by digest, never verbatim.
                self.turn.args = json!({
                    "unparsed_line_sha256": format!("{:x}", Sha256::digest(line)),
                    "bytes": line.len(),
                });
                (id, Value::Null, Err(failure))
            }
            Ok(request) => {
                self.turn.id = request.id.clone();
                self.turn.op = Value::String(request.op.clone());
                let raw = serde_json::to_vec(&request.args).unwrap_or_default();
                self.turn.args_sha256 = json!(format!("{:x}", Sha256::digest(&raw)));
                self.turn.args = logged_args(&request.args, raw.len());
                self.turn.path_like = path_like_strings(&request.args);
                let outcome = if let Some(why) = &self.fatal {
                    Err(Failure::failed(why.clone()))
                } else if self.audit.next_seq() > self.limits.max_audit_records {
                    Err(Failure::refused(format!(
                        "the audit log holds the policy's max_audit_records ({}); no more requests are answered",
                        self.limits.max_audit_records
                    )))
                } else {
                    self.dispatch(&request.op, &request.args)
                };
                (request.id, Value::String(request.op), outcome)
            }
        };
        self.finish(id, op, outcome)
    }

    fn begin_turn(&mut self) {
        self.requests += 1;
        self.turn = Turn {
            request_uid: json!(format!("{}-{}", self.instance_id, self.requests)),
            ..Turn::default()
        };
    }

    /// Answer a line that could not be read as a request at all (one longer than the policy allows).
    pub fn refuse_line(&mut self, failure: Failure) -> Vec<u8> {
        self.begin_turn();
        self.finish(Value::Null, Value::Null, Err(failure))
    }

    /// What the client is told of a failure. A LARQL reason (`failed`) that names the container, the
    /// staging directory, or echoes a path-like string the client sent is withheld whole: replacing only the
    /// path would let the client confirm a guess. A staged copy's path is told as its workdir name.
    fn outward(&self, failure: Failure) -> Failure {
        if failure.kind != ErrorKind::Failed {
            return failure;
        }
        let echoes = self
            .container_texts
            .iter()
            .chain(self.turn.path_like.iter())
            .any(|p| failure.reason.contains(p.as_str()));
        if echoes {
            return failure.withheld(WITHHELD_PATH_REASON);
        }
        // The service's own staged paths (random, never sent) are told as the workdir names they stand for.
        let mut told = failure.reason.clone();
        for copy in self.applied.iter().chain(self.recording.iter()) {
            told = told.replace(&copy.path, &copy.name);
        }
        let raw_staging = self.staging.path().display().to_string();
        if told.contains(&self.staging_prefix) || told.contains(&raw_staging) {
            return failure.withheld(WITHHELD_PATH_REASON);
        }
        if told == failure.reason {
            failure
        } else {
            failure.withheld(told)
        }
    }

    /// The response to a request, its audit record appended.
    fn finish(&mut self, id: Value, op: Value, outcome: Result<Value, Failure>) -> Vec<u8> {
        let outcome = outcome.map_err(|f| self.outward(f));
        let meta = Meta {
            seq: self.audit.next_seq(),
            request_uid: self.turn.request_uid.clone(),
            budget: budget_value(self.budget_limit, self.budget_used),
            edited: self.turn.edited,
            counted: self.turn.counted,
        };
        let line = response(&id, &meta, &outcome);
        let record = json!({
            "id": id,
            "request_uid": self.turn.request_uid,
            "op": op,
            "args": self.turn.args,
            "args_sha256": self.turn.args_sha256,
            "ok": outcome.is_ok(),
            "error": outcome.as_ref().err().map(|f| json!({
                "kind": f.kind.label(), "reason": f.reason, "internal": f.internal,
            })),
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
        #[cfg(test)]
        if self.blind_predicate {
            return Ok(false);
        }
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
    /// durable first: an `admitted` record (the request, its arguments' digest and summary, and the edits in
    /// effect with their digest) carrying the budget after it is appended and synced, and only then may the
    /// model run. A resumed log recounts the budget from these records.
    pub(super) fn charge(&mut self) -> Result<(), Failure> {
        self.turn.edited = true;
        if self.turn.counted {
            return Ok(());
        }
        if self.remaining() == 0 {
            return Err(self.exhausted());
        }
        self.admit(false)
    }

    /// An execution ran edited weights that the predicate called stored: it is charged whatever the budget
    /// (which may go past its limit, and is recorded so), and the service stops answering, since its gate
    /// cannot be trusted.
    pub(super) fn charge_unpredicted(&mut self) -> Failure {
        self.turn.edited = true;
        let why = format!(
            "request {} executed edited weights the service's gate called stored; it is charged, and the \
             service stops answering",
            self.turn.request_uid
        );
        if !self.turn.counted {
            if let Err(f) = self.admit(true) {
                return f;
            }
        }
        self.fatal = Some(why.clone());
        Failure::failed(why)
    }

    fn admit(&mut self, unpredicted: bool) -> Result<(), Failure> {
        let used = self.budget_used + 1;
        let edits = self.session.edits_in_effect().map_err(Failure::failed)?;
        let edits_bytes = serde_json::to_vec(&edits).unwrap_or_default();
        let applied_vlp_sha256: Vec<Value> = edits
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|e| e.get("sha256").filter(|s| s.is_string()).cloned())
            .collect();
        let weight_op_digests: Vec<Value> = edits
            .as_array()
            .into_iter()
            .flatten()
            .flat_map(|e| e["weight_ops"].as_array().cloned().unwrap_or_default())
            .map(|op| op["digest"].clone())
            .collect();
        let record = json!({
            "request_uid": self.turn.request_uid,
            "id": self.turn.id,
            "op": self.turn.op,
            "args_sha256": self.turn.args_sha256,
            "args": self.turn.args,
            "edits_sha256": format!("{:x}", Sha256::digest(&edits_bytes)),
            "applied_vlp_sha256": applied_vlp_sha256,
            "weight_op_digests": weight_op_digests,
            "edits": edits,
            "budget_limit": self.budget_limit,
            "budget_remaining": self.budget_limit.saturating_sub(used),
            "overdrawn": used > self.budget_limit,
            "unpredicted": unpredicted,
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

    /// A fresh staged file for workdir name `name`: its name in staging and its path as LARQL is given it.
    pub(super) fn next_staged(&mut self, name: &str) -> Result<StagedCopy, Failure> {
        self.staged_counter += 1;
        let file = format!("{}.vlp", self.staged_counter);
        let path = self
            .staging
            .path()
            .join(&file)
            .to_str()
            .map(str::to_string)
            .ok_or_else(|| Failure::refused("the staging path is not valid UTF-8"))?;
        Ok(StagedCopy {
            name: name.to_string(),
            file,
            path,
            bytes: 0,
        })
    }

    /// The staged file name the next copy will get (for tests that plant something there).
    #[cfg(test)]
    pub(super) fn upcoming_staged_path(&self) -> PathBuf {
        self.staging
            .path()
            .join(format!("{}.vlp", self.staged_counter + 1))
    }

    /// The bytes the staged copies of applied patches hold.
    pub(super) fn staged_bytes(&self) -> u64 {
        self.applied.iter().map(|c| c.bytes).sum()
    }

    /// Delete every staged copy whose patch is no longer applied.
    pub(super) fn reconcile_staged(&mut self) {
        let live: Vec<String> = self
            .session
            .patches()
            .map(|r| r.applied.into_iter().filter_map(|p| p.path).collect())
            .unwrap_or_default();
        let (keep, gone): (Vec<StagedCopy>, Vec<StagedCopy>) = std::mem::take(&mut self.applied)
            .into_iter()
            .partition(|c| live.contains(&c.path));
        for copy in gone {
            let _ = self.staged.clear(&copy.file);
        }
        self.applied = keep;
    }

    /// The workdir name a staged path stands for, if it is one.
    fn name_of(&self, text: &str) -> Option<String> {
        self.applied
            .iter()
            .chain(self.recording.iter())
            .find(|c| c.path == text)
            .map(|c| c.name.clone())
    }

    /// A patch entry's `path` and `description` (and `name`), told as the workdir name when they are a
    /// staged path.
    pub(super) fn redact_patch_entry(&self, entry: &mut Value) {
        for key in ["path", "description", "name"] {
            if let Some(text) = entry.get(key).and_then(Value::as_str) {
                if let Some(name) = self.name_of(text) {
                    entry[key] = json!(name);
                }
            }
        }
    }

    /// A provenance record as the client is told it: the container's path replaced by [`CONTAINER_LABEL`],
    /// each patch's staged path by its workdir name. Nothing else is rewritten: the arrays and the strings
    /// the client sent are not searched.
    pub(super) fn redact_provenance(&self, mut provenance: Value) -> Value {
        if provenance.get("container").is_some() {
            provenance["container"] = json!(CONTAINER_LABEL);
        }
        if let Some(patches) = provenance.get_mut("patches").and_then(Value::as_array_mut) {
            for entry in patches {
                self.redact_patch_entry(entry);
            }
        }
        provenance
    }
}

/// The client's strings that could name an absolute path (the only form a container's path takes in LARQL's
/// text): a LARQL reason that repeats one is withheld, whether or not it names the container.
fn path_like_strings(args: &Value) -> Vec<String> {
    match args {
        Value::String(s) if s.starts_with('/') || s.starts_with('~') || s.contains("/..") => {
            vec![s.clone()]
        }
        Value::Array(items) => items.iter().flat_map(path_like_strings).collect(),
        Value::Object(map) => map
            .iter()
            .filter(|(k, _)| k.as_str() != "npy_base64")
            .flat_map(|(_, v)| path_like_strings(v))
            .collect(),
        _ => Vec::new(),
    }
}

/// The arguments as the log records them: summarised, or by digest and length alone when even the summary is
/// long.
fn logged_args(args: &Value, raw_bytes: usize) -> Value {
    let summary = summarise_args(args);
    let summary_bytes = serde_json::to_vec(&summary).map(|v| v.len()).unwrap_or(0);
    if summary_bytes > MAX_LOGGED_ARGS_BYTES {
        json!({"summary_bytes": summary_bytes, "bytes": raw_bytes, "logged": "by digest (args_sha256)"})
    } else {
        summary
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
