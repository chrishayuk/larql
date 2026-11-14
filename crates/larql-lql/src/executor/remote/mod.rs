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
//!     session patch management (APPLY/SHOW/REMOVE PATCH).
//!
//! Every request carries the session header, so every mutation lands in
//! this session's patch overlay on the server (never the shared model)
//! and every read is answered against that same overlay.

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
/// `GET` lists the session's patches; `DELETE {ENDPOINT_PATCHES}/{name}`
/// removes one.
pub(super) const ENDPOINT_PATCHES: &str = "/v1/patches";

/// HTTP header naming this client's session on the server. Sent on every
/// request: mutations are scoped to the session's patch overlay, and
/// reads are answered against it.
pub(super) const SESSION_HEADER: &str = "x-session-id";

/// Field of the `/v1/stats` response naming the served model — the
/// identity a patch must declare as its `base_model`.
const STATS_MODEL_FIELD: &str = "model";

/// Banner text when `/v1/stats` names no model.
const UNNAMED_MODEL_BANNER: &str = "unknown";

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

        // The served model's identity, kept so every patch this session
        // sends can declare it as its `base_model`. Absent or empty means
        // the server did not say, and mutations are refused rather than
        // sent with an empty one.
        let model_name = stats[STATS_MODEL_FIELD]
            .as_str()
            .filter(|name| !name.is_empty())
            .map(str::to_string);
        let model = model_name.as_deref().unwrap_or(UNNAMED_MODEL_BANNER);
        let layers = stats["layers"].as_u64().unwrap_or(0);
        let features = stats["features"].as_u64().unwrap_or(0);

        // Per-session id for the server-side patch overlay. PID + millis is
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

        let banner =
            format!("Connected: {model} ({layers} layers, {features} features)\n  Remote: {url}");
        self.backend = Backend::Remote {
            url,
            client,
            session_id,
            model_name,
        };
        self.patch_recording = None;
        self.auto_patch = false;

        Ok(vec![banner])
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

    /// The served model's identity, as `/v1/stats` named it at `USE
    /// REMOTE` — the `base_model` every patch this session sends must
    /// declare. A server that named no model gets no patch: `verb` is
    /// refused by name instead of sending an empty `base_model`.
    pub(super) fn remote_model_name(&self, verb: &str) -> Result<&str, LqlError> {
        match &self.backend {
            Backend::Remote { model_name, .. } => model_name.as_deref().ok_or_else(|| {
                LqlError::Execution(format!(
                    "remote {verb} refused: the server's {ENDPOINT_STATS} named no model, \
                     so the patch cannot declare which model it belongs to. Nothing was sent."
                ))
            }),
            _ => Err(LqlError::Execution(
                "not connected to a remote server".into(),
            )),
        }
    }

    // ── Generic HTTP forwarding helpers ──
    //
    // Each one attaches the session header: a mutation lands in this
    // session's overlay, and a read is answered against it.

    /// GET `{remote_url}{endpoint}` with optional query parameters,
    /// check the response status, and parse the body as JSON.
    pub(super) fn remote_get_json(
        &self,
        endpoint: &str,
        query: &[(&str, &str)],
    ) -> Result<serde_json::Value, LqlError> {
        let (url, client, sid) = self.require_remote()?;
        let resp = client
            .get(format!("{url}{endpoint}"))
            .query(query)
            .header(SESSION_HEADER, sid)
            .send()
            .map_err(|e| LqlError::exec("request failed", e))?;
        Self::check_and_parse(endpoint, resp)
    }

    /// POST `{remote_url}{endpoint}` with a JSON body.
    pub(super) fn remote_post_json(
        &self,
        endpoint: &str,
        body: &serde_json::Value,
    ) -> Result<serde_json::Value, LqlError> {
        let (url, client, sid) = self.require_remote()?;
        let resp = client
            .post(format!("{url}{endpoint}"))
            .json(body)
            .header(SESSION_HEADER, sid)
            .send()
            .map_err(|e| LqlError::exec("request failed", e))?;
        Self::check_and_parse(endpoint, resp)
    }

    /// DELETE `{remote_url}{endpoint}/{segment}` and hand back the raw
    /// response, so the caller can name a refusal. `segment` is
    /// percent-encoded as one path segment: a name with spaces or slashes
    /// addresses exactly that name.
    pub(super) fn remote_delete_segment(
        &self,
        endpoint: &str,
        segment: &str,
    ) -> Result<reqwest::blocking::Response, LqlError> {
        let (url, client, sid) = self.require_remote()?;
        let mut target = reqwest::Url::parse(&format!("{url}{endpoint}"))
            .map_err(|e| LqlError::exec("invalid remote URL", e))?;
        target
            .path_segments_mut()
            .map_err(|()| LqlError::Execution(format!("remote URL {url} cannot take a path")))?
            .push(segment);
        client
            .delete(target)
            .header(SESSION_HEADER, sid)
            .send()
            .map_err(|e| LqlError::exec("request failed", e))
    }

    pub(super) fn check_and_parse(
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
