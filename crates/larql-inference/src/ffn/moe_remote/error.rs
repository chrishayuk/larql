// ── Public error type ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum RemoteMoeError {
    /// Could not reach the shard server (connection refused, DNS failure, etc.).
    Unreachable { url: String, cause: String },
    /// The server responded with a non-2xx status.
    ServerError { status: u16, body: String },
    /// Response body could not be parsed.
    BadResponse(String),
    /// No shard owns a required expert ID.
    NoShard { expert_id: usize },
    /// HTTP client construction failed.
    Client(String),
    /// One shard failed the experts it was asked for. Wraps the transport
    /// cause with where it happened, so the error that reaches the caller of
    /// generation says which shard, which layer and which experts went
    /// missing — the cause alone does not, since a status and a body are the
    /// same on every shard.
    ShardFailed {
        url: String,
        /// Every layer the failed request carried — one for a per-layer
        /// call, several for a multi-layer pre-dispatch.
        layers: Vec<usize>,
        expert_ids: Vec<u32>,
        cause: Box<RemoteMoeError>,
    },
}

impl RemoteMoeError {
    /// This error, attributed to the shard at `url` serving `expert_ids` of
    /// `layers`. Idempotent: an error already attributed keeps its original
    /// attribution rather than nesting a second one.
    pub fn at_shard(self, url: &str, layers: &[usize], expert_ids: &[u32]) -> Self {
        match self {
            attributed @ Self::ShardFailed { .. } => attributed,
            cause => Self::ShardFailed {
                url: url.to_string(),
                layers: layers.to_vec(),
                expert_ids: expert_ids.to_vec(),
                cause: Box::new(cause),
            },
        }
    }
}

impl std::fmt::Display for RemoteMoeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unreachable { url, cause } => {
                write!(f, "expert shard unreachable: {url} ({cause})")
            }
            Self::ServerError { status, body } => {
                write!(f, "expert shard returned {status}: {body}")
            }
            Self::BadResponse(msg) => write!(f, "bad expert response: {msg}"),
            Self::NoShard { expert_id } => write!(f, "no shard owns expert {expert_id}"),
            Self::Client(msg) => write!(f, "HTTP client error: {msg}"),
            Self::ShardFailed {
                url,
                layers,
                expert_ids,
                cause,
            } => write!(
                f,
                "expert shard {url} failed layers {layers:?} experts {expert_ids:?}: {cause}"
            ),
        }
    }
}

impl std::error::Error for RemoteMoeError {}
