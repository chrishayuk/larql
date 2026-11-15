//! The policy a session service runs under: the one container it binds, the directory every path a request
//! names must stay inside, where the audit log goes, which operations are allowed, the evaluation budget and
//! the resource limits.
//!
//! The policy is a JSON file read once at start. Unknown fields are refused, so a misspelt limit fails the
//! start instead of silently taking its default.

use std::path::{Path, PathBuf};

use serde::Deserialize;

use super::protocol::Op;

/// The `policy` field every policy file carries.
pub const POLICY_VERSION: &str = "larql-session-policy/v1";

const MIB: u64 = 1 << 20;
/// Default: the largest total of array bytes one response may carry.
pub const DEFAULT_MAX_RESPONSE_BYTES: u64 = 256 * MIB;
/// Default: the longest request line accepted, in bytes. The largest request is an `add_low_rank` operand:
/// a float64 `rows x rank` array, base64 (4/3 larger). Over Gemma 3 4B's largest tensor (262,208 rows, the
/// embedding and head) one rank is 2.8 MB of base64, so 16 MiB carries a rank-5 operand there and a rank-200
/// one over a 10,240-row FFN tensor.
pub const DEFAULT_MAX_REQUEST_BYTES: u64 = 16 * MIB;
/// Default: the longest text a request may carry (a prompt, a continuation, text to tokenize), and the
/// longest text `decode` returns, in bytes.
pub const DEFAULT_MAX_TEXT_BYTES: u64 = 64 * 1024;
/// Default: the bytes the service's staged patch copies may hold at once.
pub const DEFAULT_MAX_STAGING_BYTES: u64 = 256 * MIB;
/// Default: the most records the audit log holds; past it, every request is refused.
pub const DEFAULT_MAX_AUDIT_RECORDS: u64 = 10_000_000;
/// Default: the most tokens a `generate`, or a capture's `generate`, may produce.
pub const DEFAULT_MAX_GENERATE_TOKENS: u64 = 256;
/// Default: the most positions one capture may return.
pub const DEFAULT_MAX_CAPTURE_POSITIONS: u64 = 4096;
/// Default: the most layers one capture may return.
pub const DEFAULT_MAX_CAPTURE_LAYERS: u64 = 64;
/// Default: the longest sequence (prompt plus continuation) one execution may run, in tokens.
pub const DEFAULT_MAX_SEQUENCE_TOKENS: u64 = 4096;
/// Default: the largest `.vlp` file `apply_patch` reads.
pub const DEFAULT_MAX_PATCH_BYTES: u64 = 64 * MIB;
/// Default: the largest `top` an `infer` may ask for.
pub const DEFAULT_MAX_INFER_TOP: u64 = 1000;
/// Default: the most patches applied at once.
pub const DEFAULT_MAX_APPLIED_PATCHES: u64 = 16;

/// The policy file, as written.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Policy {
    /// Must be [`POLICY_VERSION`].
    pub policy: String,
    /// The VINDEX3 container the service binds, once, at start.
    pub container: PathBuf,
    /// The only directory request paths may name. Patch files are read from and written into it.
    pub workdir: PathBuf,
    /// The JSONL audit log. Refused if it exists, unless `resume` is true.
    pub audit_log: PathBuf,
    /// Continue an existing audit log: its sequence numbers and its budget use carry on.
    #[serde(default)]
    pub resume: bool,
    /// The operations a request may name. `hello` and `budget` are always answered.
    pub operations: Vec<String>,
    pub budget: Budget,
    #[serde(default)]
    pub limits: Limits,
    /// `stored` (the default), `auto` or `transient`, as `larql lql --representation-source` spells it.
    #[serde(default)]
    pub representation_source: Option<String>,
    /// Where the service keeps its private copies of patch files: a fresh directory (mode 0700) is created
    /// inside it. Required, and must be outside the workdir and the container; it should be a directory the
    /// client cannot write.
    pub staging_dir: PathBuf,
    /// Refuse, before running it, any capture whose response size the service cannot bound from the plan
    /// first (a site the plan gives no width, such as a hyper-connection residual or a routed FFN's down
    /// projection input). Default true: then no capture is ever executed and afterwards refused on size.
    #[serde(default = "default_true")]
    pub require_capture_bound: bool,
}

fn default_true() -> bool {
    true
}

/// The evaluation budget.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Budget {
    /// How many executions of an edited model (an infer, generate or capture while a patch is applied or a
    /// weight op is staged) the service answers. Executions of the base model are not counted.
    pub edited_executions: u64,
}

/// Resource limits. Every field has a default.
#[derive(Debug, Clone, Deserialize, serde::Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields, default)]
pub struct Limits {
    pub max_response_bytes: u64,
    pub max_request_bytes: u64,
    pub max_generate_tokens: u64,
    pub max_capture_positions: u64,
    pub max_capture_layers: u64,
    pub max_sequence_tokens: u64,
    pub max_patch_bytes: u64,
    pub max_infer_top: u64,
    pub max_applied_patches: u64,
    pub max_text_bytes: u64,
    pub max_staging_bytes: u64,
    pub max_audit_records: u64,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_response_bytes: DEFAULT_MAX_RESPONSE_BYTES,
            max_request_bytes: DEFAULT_MAX_REQUEST_BYTES,
            max_generate_tokens: DEFAULT_MAX_GENERATE_TOKENS,
            max_capture_positions: DEFAULT_MAX_CAPTURE_POSITIONS,
            max_capture_layers: DEFAULT_MAX_CAPTURE_LAYERS,
            max_sequence_tokens: DEFAULT_MAX_SEQUENCE_TOKENS,
            max_patch_bytes: DEFAULT_MAX_PATCH_BYTES,
            max_infer_top: DEFAULT_MAX_INFER_TOP,
            max_applied_patches: DEFAULT_MAX_APPLIED_PATCHES,
            max_text_bytes: DEFAULT_MAX_TEXT_BYTES,
            max_staging_bytes: DEFAULT_MAX_STAGING_BYTES,
            max_audit_records: DEFAULT_MAX_AUDIT_RECORDS,
        }
    }
}

/// A policy that was read and checked: its operations parsed, and the SHA-256 of the file's bytes.
#[derive(Debug, Clone)]
pub struct CheckedPolicy {
    pub policy: Policy,
    pub operations: Vec<Op>,
    pub sha256: String,
}

impl Policy {
    /// Parse and check a policy from the file's bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<CheckedPolicy, String> {
        use sha2::{Digest, Sha256};
        let policy: Policy =
            serde_json::from_slice(bytes).map_err(|e| format!("the policy is not valid: {e}"))?;
        if policy.policy != POLICY_VERSION {
            return Err(format!(
                "the policy declares `{}`; this service reads `{POLICY_VERSION}`",
                policy.policy
            ));
        }
        let mut operations = Vec::new();
        for name in &policy.operations {
            let op = Op::from_name(name).ok_or_else(|| {
                format!(
                    "the policy allows `{name}`, which is not an operation; operations are {}",
                    Op::names().join(", ")
                )
            })?;
            if !operations.contains(&op) {
                operations.push(op);
            }
        }
        Ok(CheckedPolicy {
            policy,
            operations,
            sha256: format!("{:x}", Sha256::digest(bytes)),
        })
    }

    /// Read and check the policy file at `path`.
    pub fn read(path: &Path) -> Result<CheckedPolicy, String> {
        let bytes =
            std::fs::read(path).map_err(|e| format!("read the policy {}: {e}", path.display()))?;
        Self::from_bytes(&bytes)
    }
}
