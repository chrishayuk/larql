//! Remote executor — forwards LQL queries to a `larql-server` over HTTP.
//!
//! Layout:
//!
//!   - `mod.rs` (this file): connection plumbing (`exec_use_remote`,
//!     `is_remote`, `require_remote`), generic HTTP helpers, and
//!     centralised endpoint / header constants.
//!   - `query.rs`: read-side verbs (DESCRIBE, WALK, INFER, EXPLAIN
//!     INFER, STATS, SHOW RELATIONS, SELECT) plus shared `knn_override`
//!     formatters.
//!   - `mutation.rs`: write-side verbs (INSERT, DELETE, UPDATE) plus
//!     local-patch management (APPLY/SHOW/REMOVE PATCH).

use super::Backend;
use super::Session;
use crate::ast::Condition;
use crate::error::LqlError;
use crate::executor::predicate::{EdgePredicate, PredicateScope};

mod mutation;
mod query;

// ── Endpoint constants ───────────────────────────────────────────
//
// Every remote forwarder used to embed its `/v1/foo` path inline. A
// stale or moved endpoint would silently 404 across multiple call
// sites; centralising means a server-side rename is a one-line patch.

pub(super) const ENDPOINT_STATS: &str = "/v1/stats";
pub(super) const ENDPOINT_DESCRIBE: &str = "/v1/describe";
pub(super) const ENDPOINT_WALK: &str = "/v1/walk";
pub(super) const ENDPOINT_INFER: &str = "/v1/infer";
pub(super) const ENDPOINT_EXPLAIN_INFER: &str = "/v1/explain-infer";
pub(super) const ENDPOINT_RELATIONS: &str = "/v1/relations";
pub(super) const ENDPOINT_SELECT: &str = "/v1/select";
pub(super) const ENDPOINT_INSERT: &str = "/v1/insert";
pub(super) const ENDPOINT_PATCHES_APPLY: &str = "/v1/patches/apply";

/// HTTP header used to identify the per-session patch queue on the
/// server. Sent on POSTs that mutate session state.
pub(super) const SESSION_HEADER: &str = "x-session-id";

/// Default request timeout. Long enough to absorb a deep-walk INFER
/// on a 4B-class model; short enough that a hung server doesn't
/// freeze the REPL indefinitely.
pub(super) const DEFAULT_REMOTE_TIMEOUT_SECS: u64 = 30;

// ── WHERE-clause helpers ─────────────────────────────────────────
//
// Used by remote DELETE / UPDATE: both verbs require an explicit
// `layer = N AND feature = M` predicate; missing fields used to
// silently default to `0` and would clobber `L0 F0` on the server.

/// Require an explicit `layer = N AND feature = M` from a remote
/// DELETE / UPDATE WHERE clause. Errors with a verb-tagged message
/// when either is missing or non-positive-integer.
pub(super) fn require_layer_feature(
    conditions: &[Condition],
    verb: &str,
) -> Result<(usize, usize), LqlError> {
    // The same grammar as a local mutation: a clause with a condition
    // that cannot be evaluated is refused before anything is sent.
    let predicate = EdgePredicate::parse(conditions, PredicateScope::EDGES)?;
    if predicate.entity.is_some() || predicate.relation.is_some() || predicate.score.is_some() {
        return Err(LqlError::Execution(format!(
            "remote {verb} addresses one slot by `layer` and `feature`; entity, relation \
             and score conditions are not evaluated remotely and would be dropped"
        )));
    }
    let layer = predicate.layer.ok_or_else(|| {
        LqlError::Execution(format!(
            "remote {verb} requires `layer = <int>` in WHERE clause"
        ))
    })?;
    let feature = predicate.feature.ok_or_else(|| {
        LqlError::Execution(format!(
            "remote {verb} requires `feature = <int>` in WHERE clause"
        ))
    })?;
    Ok((layer, feature))
}

impl Session {
    /// Connect to a remote larql-server.
    pub(crate) fn exec_use_remote(&mut self, url: &str) -> Result<Vec<String>, LqlError> {
        let url = url.trim_end_matches('/').to_string();

        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(DEFAULT_REMOTE_TIMEOUT_SECS))
            .build()
            .map_err(|e| LqlError::exec("failed to create HTTP client", e))?;

        // Verify the server is reachable by hitting /v1/stats.
        let stats_url = format!("{url}{ENDPOINT_STATS}");
        let resp = client
            .get(&stats_url)
            .send()
            .map_err(|e| LqlError::exec(format!("failed to connect to {url}"), e))?;

        if !resp.status().is_success() {
            return Err(LqlError::Execution(format!(
                "server returned {}: {}",
                resp.status(),
                resp.text().unwrap_or_default()
            )));
        }

        let stats: serde_json::Value = resp
            .json()
            .map_err(|e| LqlError::exec("invalid response from server", e))?;

        let model = stats["model"].as_str().unwrap_or("unknown");
        let layers = stats["layers"].as_u64().unwrap_or(0);
        let features = stats["features"].as_u64().unwrap_or(0);

        // Per-session id for server-side patch queue. PID + millis is
        // weak entropy but acceptable for the localhost dev model;
        // production deployments should authenticate before this is
        // load-bearing.
        let session_id = format!(
            "larql-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis()
        );

        self.backend = Backend::Remote {
            url: url.clone(),
            client,
            local_patches: Vec::new(),
            session_id,
        };
        self.patch_recording = None;
        self.auto_patch = false;

        Ok(vec![format!(
            "Connected: {model} ({layers} layers, {features} features)\n  Remote: {url}"
        )])
    }

    /// True iff the active backend is the Remote variant.
    pub(crate) fn is_remote(&self) -> bool {
        matches!(&self.backend, Backend::Remote { .. })
    }

    /// Get the remote URL, client, and session ID, or error.
    pub(super) fn require_remote(
        &self,
    ) -> Result<(&str, &reqwest::blocking::Client, &str), LqlError> {
        match &self.backend {
            Backend::Remote {
                url,
                client,
                session_id,
                ..
            } => Ok((url, client, session_id)),
            _ => Err(LqlError::Execution(
                "not connected to a remote server".into(),
            )),
        }
    }

    // ── Generic HTTP forwarding helpers ──

    /// GET `{remote_url}{endpoint}` with optional query parameters,
    /// check the response status, and parse the body as JSON.
    pub(super) fn remote_get_json(
        &self,
        endpoint: &str,
        query: &[(&str, &str)],
    ) -> Result<serde_json::Value, LqlError> {
        let (url, client, _sid) = self.require_remote()?;
        let resp = client
            .get(format!("{url}{endpoint}"))
            .query(query)
            .send()
            .map_err(|e| LqlError::exec("request failed", e))?;
        Self::check_and_parse(endpoint, resp)
    }

    /// POST `{remote_url}{endpoint}` with a JSON body. When
    /// `with_session` is true, attaches the `x-session-id` header so
    /// the server can route this request to the correct session-side
    /// patch queue.
    pub(super) fn remote_post_json(
        &self,
        endpoint: &str,
        body: &serde_json::Value,
        with_session: bool,
    ) -> Result<serde_json::Value, LqlError> {
        let (url, client, sid) = self.require_remote()?;
        let mut req = client.post(format!("{url}{endpoint}")).json(body);
        if with_session {
            req = req.header(SESSION_HEADER, sid);
        }
        let resp = req
            .send()
            .map_err(|e| LqlError::exec("request failed", e))?;
        Self::check_and_parse(endpoint, resp)
    }

    fn check_and_parse(
        endpoint: &str,
        resp: reqwest::blocking::Response,
    ) -> Result<serde_json::Value, LqlError> {
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().unwrap_or_default();
            return Err(LqlError::Execution(format!(
                "{endpoint} failed ({status}): {text}"
            )));
        }
        resp.json::<serde_json::Value>()
            .map_err(|e| LqlError::exec("invalid response", e))
    }
}

#[cfg(test)]
mod tests;
