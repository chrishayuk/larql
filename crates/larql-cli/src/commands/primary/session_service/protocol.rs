//! The wire protocol, `larql-session/v1`: one JSON object per line each way, one response per request.
//!
//! A request is `{"protocol": "larql-session/v1", "id": <any JSON, echoed>, "op": "<name>", "args": {…}}`.
//! A response is `{"protocol", "id", "seq", "ok", "result" | "error", "budget", "edited", "counted"}`:
//! - `seq` is the request's sequence number in the audit log;
//! - `budget` is `{"limit", "used", "remaining"}` after the request;
//! - `edited` says whether an execution ran an edited model, `counted` whether it was charged to the budget;
//! - an error is `{"kind", "reason"}`, `kind` one of the [`ErrorKind`] labels.
//!
//! Arrays travel inside the JSON as a whole `.npy` file, base64: `{"npy_base64", "sha256", "dtype",
//! "shape"}`, `sha256` the digest of the decoded `.npy` bytes.

use serde::Deserialize;
use serde_json::{json, Value};

/// The protocol every request and response names.
pub const PROTOCOL: &str = "larql-session/v1";
/// The longest `id` echoed, as JSON.
pub const MAX_ID_BYTES: usize = 256;

/// An operation the policy can allow. `hello` and `budget` are not here: they are always answered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Tensors,
    Tensor,
    Tokenize,
    Decode,
    Capture,
    Infer,
    Generate,
    BeginPatch,
    AddLowRank,
    UpdateWeights,
    SavePatch,
    ApplyPatch,
    RemovePatch,
    Patches,
    DiscardWeightEdits,
}

impl Op {
    pub const ALL: [Op; 15] = [
        Op::Tensors,
        Op::Tensor,
        Op::Tokenize,
        Op::Decode,
        Op::Capture,
        Op::Infer,
        Op::Generate,
        Op::BeginPatch,
        Op::AddLowRank,
        Op::UpdateWeights,
        Op::SavePatch,
        Op::ApplyPatch,
        Op::RemovePatch,
        Op::Patches,
        Op::DiscardWeightEdits,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Op::Tensors => "tensors",
            Op::Tensor => "tensor",
            Op::Tokenize => "tokenize",
            Op::Decode => "decode",
            Op::Capture => "capture",
            Op::Infer => "infer",
            Op::Generate => "generate",
            Op::BeginPatch => "begin_patch",
            Op::AddLowRank => "add_low_rank",
            Op::UpdateWeights => "update_weights",
            Op::SavePatch => "save_patch",
            Op::ApplyPatch => "apply_patch",
            Op::RemovePatch => "remove_patch",
            Op::Patches => "patches",
            Op::DiscardWeightEdits => "discard_weight_edits",
        }
    }

    pub fn from_name(name: &str) -> Option<Op> {
        Op::ALL.into_iter().find(|op| op.name() == name)
    }

    pub fn names() -> Vec<&'static str> {
        Op::ALL.iter().map(|op| op.name()).collect()
    }
}

/// The always-answered operations.
pub const HELLO: &str = "hello";
pub const BUDGET: &str = "budget";

/// Why a request got no result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    /// The line is not a request: not JSON, not an object, a missing or unknown field, a bad argument.
    InvalidRequest,
    /// The request names a protocol this service does not speak.
    UnsupportedProtocol,
    /// The policy does not allow it: an operation not on the list, a path outside the workdir, a limit.
    Refused,
    /// The evaluation budget is spent and the request would execute an edited model.
    BudgetExhausted,
    /// LARQL ran the operation and it failed; the reason is LARQL's.
    Failed,
    /// Another client is connected.
    Busy,
}

impl ErrorKind {
    pub fn label(self) -> &'static str {
        match self {
            ErrorKind::InvalidRequest => "invalid_request",
            ErrorKind::UnsupportedProtocol => "unsupported_protocol",
            ErrorKind::Refused => "refused",
            ErrorKind::BudgetExhausted => "budget_exhausted",
            ErrorKind::Failed => "failed",
            ErrorKind::Busy => "busy",
        }
    }
}

/// A request that got no result, and why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Failure {
    pub kind: ErrorKind,
    pub reason: String,
}

impl Failure {
    pub fn new(kind: ErrorKind, reason: impl Into<String>) -> Self {
        Self {
            kind,
            reason: reason.into(),
        }
    }
    pub fn invalid(reason: impl Into<String>) -> Self {
        Self::new(ErrorKind::InvalidRequest, reason)
    }
    pub fn refused(reason: impl Into<String>) -> Self {
        Self::new(ErrorKind::Refused, reason)
    }
    pub fn failed(reason: impl std::fmt::Display) -> Self {
        Self::new(ErrorKind::Failed, reason.to_string())
    }
}

/// A request line, parsed as far as the envelope.
#[derive(Debug, Clone)]
pub struct Request {
    pub id: Value,
    pub op: String,
    pub args: Value,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Envelope {
    protocol: Option<String>,
    #[serde(default)]
    id: Value,
    op: Option<String>,
    #[serde(default)]
    args: Option<Value>,
}

/// Parse one request line. The `id` is recovered whenever the line is a JSON object, so even a refused
/// request's response carries it.
pub fn parse_request(line: &[u8]) -> Result<Request, (Value, Failure)> {
    let value: Value = serde_json::from_slice(line)
        .map_err(|e| (Value::Null, Failure::invalid(format!("not JSON: {e}"))))?;
    let id = value.get("id").cloned().unwrap_or(Value::Null);
    if id.to_string().len() > MAX_ID_BYTES {
        return Err((
            Value::Null,
            Failure::invalid(format!("`id` is longer than {MAX_ID_BYTES} bytes as JSON")),
        ));
    }
    let envelope: Envelope = serde_json::from_value(value)
        .map_err(|e| (id.clone(), Failure::invalid(format!("not a request: {e}"))))?;
    match envelope.protocol.as_deref() {
        Some(PROTOCOL) => {}
        Some(other) => {
            return Err((
                id,
                Failure::new(
                    ErrorKind::UnsupportedProtocol,
                    format!("the request speaks `{other}`; this service speaks `{PROTOCOL}`"),
                ),
            ))
        }
        None => {
            return Err((
                id,
                Failure::new(
                    ErrorKind::UnsupportedProtocol,
                    format!("the request names no protocol; this service speaks `{PROTOCOL}`"),
                ),
            ))
        }
    }
    let op = envelope
        .op
        .ok_or_else(|| (id.clone(), Failure::invalid("the request names no `op`")))?;
    let args = match envelope.args {
        None | Some(Value::Null) => json!({}),
        Some(args @ Value::Object(_)) => args,
        Some(_) => return Err((id, Failure::invalid("`args` must be an object"))),
    };
    Ok(Request {
        id: envelope.id,
        op,
        args,
    })
}

/// Arguments of type `T`, or an invalid-request failure naming the operation.
pub fn args<T: serde::de::DeserializeOwned>(op: &str, args: &Value) -> Result<T, Failure> {
    serde_json::from_value(args.clone())
        .map_err(|e| Failure::invalid(format!("`{op}` arguments: {e}")))
}

/// The budget as a response reports it.
pub fn budget_value(limit: u64, used: u64) -> Value {
    json!({"limit": limit, "used": used, "remaining": limit.saturating_sub(used)})
}

/// What a response says besides its result or error.
pub struct Meta {
    pub seq: u64,
    pub budget: Value,
    pub edited: bool,
    pub counted: bool,
}

/// The response line for `id`, without the trailing newline.
pub fn response(id: &Value, meta: &Meta, outcome: &Result<Value, Failure>) -> Vec<u8> {
    let mut out = json!({
        "protocol": PROTOCOL,
        "id": id,
        "seq": meta.seq,
        "ok": outcome.is_ok(),
        "budget": meta.budget,
        "edited": meta.edited,
        "counted": meta.counted,
    });
    match outcome {
        Ok(result) => out["result"] = result.clone(),
        Err(f) => out["error"] = json!({"kind": f.kind.label(), "reason": f.reason}),
    }
    serde_json::to_vec(&out).expect("a JSON value serialises")
}
