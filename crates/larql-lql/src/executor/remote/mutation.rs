//! Write-side remote forwarders (INSERT, DELETE, UPDATE) plus the
//! session's patch management on the server (APPLY / SHOW / REMOVE PATCH).
//!
//! Every one of them is per-session: the request carries the session
//! header, so the server applies it to this session's overlay and never to
//! the model it shares with other clients. A patch this session sends
//! declares the served model as its `base_model`.

use crate::ast::{Assignment, Condition};
use crate::error::LqlError;
use crate::executor::mutation::update_set::UpdateSet;
use crate::executor::Session;

use super::{require_layer_feature, ENDPOINT_INSERT, ENDPOINT_PATCHES, ENDPOINT_PATCHES_APPLY};

/// Default `confidence` for INSERT/UPDATE when the user doesn't
/// pass one. 0.9 lands well above the retrieval floor without
/// dominating template-matched siblings.
const REMOTE_DEFAULT_CONFIDENCE: f32 = 0.9;

/// Format version of the patches remote DELETE / UPDATE build.
const REMOTE_PATCH_VERSION: u32 = 1;

/// Field of an apply-patch request body carrying the inline patch.
const APPLY_BODY_PATCH_FIELD: &str = "patch";

/// Name shown for a listed patch the server reports without one.
const UNNAMED_PATCH_LABEL: &str = "(unnamed)";

impl Session {
    pub(crate) fn remote_insert(
        &self,
        entity: &str,
        relation: &str,
        target: &str,
        layer: Option<u32>,
        confidence: Option<f32>,
    ) -> Result<Vec<String>, LqlError> {
        let request = serde_json::json!({
            "entity": entity,
            "relation": relation,
            "target": target,
            "layer": layer,
            "confidence": confidence.unwrap_or(REMOTE_DEFAULT_CONFIDENCE),
        });

        let result = self.remote_post_json(ENDPOINT_INSERT, &request)?;

        let inserted = result["inserted"].as_u64().unwrap_or(0);
        let mode = result["mode"].as_str().unwrap_or("unknown");
        let ms = result["latency_ms"].as_f64().unwrap_or(0.0);

        Ok(vec![
            format!("Inserted: {entity} —[{relation}]→ {target} ({inserted} layers, mode: {mode})"),
            format!("{ms:.0}ms (remote)"),
        ])
    }

    /// Build a one-operation patch against the served model and apply it
    /// to this session's overlay on the server. Refused, before anything
    /// is sent, when the server named no model at `USE REMOTE`.
    fn remote_apply_op(
        &self,
        verb: &str,
        description: String,
        op: larql_vindex::PatchOp,
    ) -> Result<(), LqlError> {
        let base_model = self.remote_model_name(verb)?.to_string();
        let patch = larql_vindex::VindexPatch {
            version: REMOTE_PATCH_VERSION,
            base_model,
            base_checksum: None,
            created_at: String::new(),
            description: Some(description),
            author: None,
            tags: vec![],
            operations: vec![op],
        };
        self.remote_post_json(
            ENDPOINT_PATCHES_APPLY,
            &serde_json::json!({ APPLY_BODY_PATCH_FIELD: patch }),
        )?;
        Ok(())
    }

    pub(crate) fn remote_delete(&self, conditions: &[Condition]) -> Result<Vec<String>, LqlError> {
        let (layer, feature) = require_layer_feature(conditions, "DELETE")?;

        let op = larql_vindex::PatchOp::Delete {
            layer,
            feature,
            reason: Some("remote DELETE".into()),
        };
        self.remote_apply_op("DELETE", format!("DELETE L{layer} F{feature}"), op)?;

        Ok(vec![format!(
            "Deleted: L{layer} F{feature} → remote session"
        )])
    }

    pub(crate) fn remote_update(
        &self,
        set: &[Assignment],
        conditions: &[Condition],
    ) -> Result<Vec<String>, LqlError> {
        // The SET side is checked by the same rule as a local UPDATE, and
        // first, as there: before the WHERE side and before anything is sent.
        let UpdateSet { target, confidence } = UpdateSet::parse(set)?;
        let (layer, feature) = require_layer_feature(conditions, "UPDATE")?;

        let down_meta = target
            .as_ref()
            .map(|t| larql_vindex::patch::core::PatchDownMeta {
                top_token: t.clone(),
                top_token_id: 0,
                c_score: confidence.unwrap_or(REMOTE_DEFAULT_CONFIDENCE),
            });

        let op = larql_vindex::PatchOp::Update {
            layer,
            feature,
            gate_vector_b64: None,
            up_vector_b64: None,
            down_vector_b64: None,
            down_meta,
        };
        self.remote_apply_op("UPDATE", format!("UPDATE L{layer} F{feature}"), op)?;

        let desc = target
            .as_deref()
            .map(|t| format!(" target={t}"))
            .unwrap_or_default();
        Ok(vec![format!(
            "Updated: L{layer} F{feature}{desc} → remote session"
        )])
    }

    // ── Session patch management (on the server) ─────────────────

    /// APPLY PATCH: load the `.vlp`, check it was written against the
    /// served model (as a local APPLY PATCH does), and apply it to this
    /// session's overlay on the server. A patch the server refuses is an
    /// error, never reported as applied.
    pub(crate) fn remote_apply_patch(&self, path: &str) -> Result<Vec<String>, LqlError> {
        let model = self.remote_model_name("APPLY PATCH")?;

        let patch_path = std::path::PathBuf::from(path);
        if !patch_path.exists() {
            return Err(LqlError::Execution(format!("patch not found: {path}")));
        }
        let patch = larql_vindex::VindexPatch::load(&patch_path)
            .map_err(|e| LqlError::exec("failed to load patch", e))?;
        patch
            .check_base_model(model)
            .map_err(|e| LqlError::exec("failed to apply patch", e))?;

        let (ins, upd, del) = patch.counts();
        let total = patch.len();

        let result = self.remote_post_json(
            ENDPOINT_PATCHES_APPLY,
            &serde_json::json!({ APPLY_BODY_PATCH_FIELD: patch }),
        )?;
        let active = result["active_patches"].as_u64().unwrap_or(0);

        Ok(vec![format!(
            "Applied to remote session: {path} ({total} ops: {ins} ins, {upd} upd, {del} del)\n\
             {active} patch(es) active in this session."
        )])
    }

    /// SHOW PATCHES: the patches active in this session on the server.
    pub(crate) fn remote_show_patches(&self) -> Result<Vec<String>, LqlError> {
        let body = self.remote_get_json(ENDPOINT_PATCHES, &[])?;
        let patches = body["patches"].as_array().cloned().unwrap_or_default();

        let mut out = Vec::new();
        if patches.is_empty() {
            out.push("  (no patches in this remote session)".into());
            return Ok(out);
        }
        out.push("Remote session patches:".into());
        for (i, patch) in patches.iter().enumerate() {
            let name = patch["name"].as_str().unwrap_or(UNNAMED_PATCH_LABEL);
            let ops = patch["operations"].as_u64().unwrap_or(0);
            out.push(format!("  {}. {:<40} {} ops", i + 1, name, ops));
        }
        Ok(out)
    }

    /// REMOVE PATCH: drop the named patch from this session's overlay on
    /// the server. A name the session does not hold is an error by name.
    pub(crate) fn remote_remove_patch(&self, name: &str) -> Result<Vec<String>, LqlError> {
        let resp = self.remote_delete_segment(ENDPOINT_PATCHES, name)?;
        if resp.status() == reqwest::StatusCode::NOT_FOUND {
            return Err(LqlError::Execution(format!(
                "remote patch not found: no patch named `{name}` in this session"
            )));
        }
        let result = Self::check_and_parse(ENDPOINT_PATCHES, resp)?;
        let active = result["active_patches"].as_u64().unwrap_or(0);
        Ok(vec![format!(
            "Removed remote patch: {name} ({active} patch(es) active in this session)"
        )])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remote_default_confidence_is_in_band() {
        // Pinned: must land between PROB_FLOOR and PROB_CEILING from
        // `executor::tuning` so newly-inserted facts come out of the
        // gate already inside the per-INSERT balance band.
        use crate::executor::tuning::{PROB_CEILING, PROB_FLOOR};
        assert!((REMOTE_DEFAULT_CONFIDENCE as f64) >= PROB_FLOOR);
        assert!((REMOTE_DEFAULT_CONFIDENCE as f64) <= PROB_CEILING);
    }

    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn remote_default_confidence_is_finite() {
        assert!(REMOTE_DEFAULT_CONFIDENCE.is_finite());
        assert!(REMOTE_DEFAULT_CONFIDENCE > 0.0);
    }

    // ── Mockito end-to-end tests for the mutation forwarders ────────

    use super::super::ENDPOINT_STATS;
    use crate::ast::{Assignment, CompareOp, Condition, Value};

    fn stats_body() -> String {
        serde_json::json!({
            "model": "test-model",
            "family": "llama",
            "layers": 32,
            "features": 4096,
            "hidden_size": 1024,
            "dtype": "f32",
            "extract_level": "all",
            "loaded": {"browse": true, "inference": false},
        })
        .to_string()
    }

    fn connect(server_url: &str) -> Session {
        let mut session = Session::new();
        session
            .exec_use_remote(server_url)
            .expect("exec_use_remote with mocked /v1/stats");
        session
    }

    #[test]
    fn remote_insert_renders_summary_from_response() {
        let mut server = mockito::Server::new();
        let _stats = server
            .mock("GET", ENDPOINT_STATS)
            .with_status(200)
            .with_body(stats_body())
            .create();
        let _insert = server
            .mock("POST", ENDPOINT_INSERT)
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                serde_json::json!({
                    "inserted": 3,
                    "mode": "compose",
                    "latency_ms": 17.0,
                })
                .to_string(),
            )
            .create();

        let session = connect(&server.url());
        let out = session
            .remote_insert("France", "capital", "Paris", Some(26), Some(0.9))
            .expect("remote_insert");
        let joined = out.join("\n");
        assert!(joined.contains("France"));
        assert!(joined.contains("compose"));
        assert!(joined.contains("3 layers"));
        assert!(joined.contains("17ms"));
    }

    #[test]
    fn remote_insert_uses_default_confidence_when_none() {
        // confidence=None should fall through to REMOTE_DEFAULT_CONFIDENCE.
        let mut server = mockito::Server::new();
        let _stats = server
            .mock("GET", ENDPOINT_STATS)
            .with_status(200)
            .with_body(stats_body())
            .create();
        let _insert = server
            .mock("POST", ENDPOINT_INSERT)
            .with_status(200)
            .with_body(
                serde_json::json!({
                    "inserted": 1,
                    "mode": "knn",
                    "latency_ms": 5.0,
                })
                .to_string(),
            )
            .create();

        let session = connect(&server.url());
        let _ = session
            .remote_insert("X", "r", "Y", None, None)
            .expect("remote_insert with default confidence");
    }

    #[test]
    fn remote_delete_posts_patch_and_renders_summary() {
        let mut server = mockito::Server::new();
        let _stats = server
            .mock("GET", ENDPOINT_STATS)
            .with_status(200)
            .with_body(stats_body())
            .create();
        let _apply = server
            .mock("POST", ENDPOINT_PATCHES_APPLY)
            .with_status(200)
            .with_body(serde_json::json!({"applied": 1}).to_string())
            .create();

        let session = connect(&server.url());
        let conds = vec![
            Condition {
                field: "layer".into(),
                op: CompareOp::Eq,
                value: Value::Integer(26),
            },
            Condition {
                field: "feature".into(),
                op: CompareOp::Eq,
                value: Value::Integer(7),
            },
        ];
        let out = session.remote_delete(&conds).expect("remote_delete");
        let joined = out.join("\n");
        assert!(joined.contains("L26"));
        assert!(joined.contains("F7"));
        assert!(joined.contains("remote session"));
    }

    #[test]
    fn remote_delete_errors_without_layer_feature_filter() {
        // require_layer_feature should reject filters that omit either
        // layer or feature.
        let mut server = mockito::Server::new();
        let _stats = server
            .mock("GET", ENDPOINT_STATS)
            .with_status(200)
            .with_body(stats_body())
            .create();
        let session = connect(&server.url());
        let conds = vec![Condition {
            field: "layer".into(),
            op: CompareOp::Eq,
            value: Value::Integer(26),
        }];
        let err = session.remote_delete(&conds).unwrap_err();
        assert!(err.to_string().to_lowercase().contains("feature"));
    }

    #[test]
    fn remote_update_with_target_renders_target_in_summary() {
        let mut server = mockito::Server::new();
        let _stats = server
            .mock("GET", ENDPOINT_STATS)
            .with_status(200)
            .with_body(stats_body())
            .create();
        let _apply = server
            .mock("POST", ENDPOINT_PATCHES_APPLY)
            .with_status(200)
            .with_body(serde_json::json!({"applied": 1}).to_string())
            .create();

        let session = connect(&server.url());
        let set = vec![
            Assignment {
                field: "target".into(),
                value: Value::String("Madrid".into()),
            },
            Assignment {
                field: "confidence".into(),
                value: Value::Number(0.85),
            },
        ];
        let conds = vec![
            Condition {
                field: "layer".into(),
                op: CompareOp::Eq,
                value: Value::Integer(26),
            },
            Condition {
                field: "feature".into(),
                op: CompareOp::Eq,
                value: Value::Integer(0),
            },
        ];
        let out = session.remote_update(&set, &conds).expect("remote_update");
        let joined = out.join("\n");
        assert!(joined.contains("target=Madrid"));
        assert!(joined.contains("L26"));
    }

    #[test]
    fn remote_update_without_target_omits_target_clause() {
        let mut server = mockito::Server::new();
        let _stats = server
            .mock("GET", ENDPOINT_STATS)
            .with_status(200)
            .with_body(stats_body())
            .create();
        let _apply = server
            .mock("POST", ENDPOINT_PATCHES_APPLY)
            .with_status(200)
            .with_body(serde_json::json!({"applied": 1}).to_string())
            .create();

        let session = connect(&server.url());
        let set = vec![Assignment {
            field: "confidence".into(),
            value: Value::Integer(1),
        }];
        let conds = vec![
            Condition {
                field: "layer".into(),
                op: CompareOp::Eq,
                value: Value::Integer(0),
            },
            Condition {
                field: "feature".into(),
                op: CompareOp::Eq,
                value: Value::Integer(0),
            },
        ];
        let out = session
            .remote_update(&set, &conds)
            .expect("remote_update no target");
        let joined = out.join("\n");
        assert!(!joined.contains("target="));
    }

    // ── Session patch verbs refuse without a remote backend ────────

    #[test]
    fn session_patch_verbs_error_when_not_remote() {
        let session = Session::new();
        for err in [
            session.remote_apply_patch("any.vlp").unwrap_err(),
            session.remote_show_patches().unwrap_err(),
            session.remote_remove_patch("anything").unwrap_err(),
        ] {
            assert!(err.to_string().contains("not connected"), "got: {err}");
        }
    }

    #[test]
    fn delete_carries_this_sessions_id() {
        // The header value is exactly the id minted at USE REMOTE, so the
        // server files the patch under this session and no other.
        let mut server = mockito::Server::new();
        let _stats = server
            .mock("GET", ENDPOINT_STATS)
            .with_status(200)
            .with_body(stats_body())
            .create();
        let session = connect(&server.url());
        let sid = match &session.backend {
            crate::executor::Backend::Remote { session_id, .. } => session_id.clone(),
            _ => panic!("expected Backend::Remote"),
        };
        let apply = server
            .mock("POST", ENDPOINT_PATCHES_APPLY)
            .match_header(super::super::SESSION_HEADER, sid.as_str())
            .with_status(200)
            .with_body(serde_json::json!({"active_patches": 1}).to_string())
            .create();
        let conds = vec![
            Condition {
                field: "layer".into(),
                op: CompareOp::Eq,
                value: Value::Integer(1),
            },
            Condition {
                field: "feature".into(),
                op: CompareOp::Eq,
                value: Value::Integer(2),
            },
        ];
        session.remote_delete(&conds).expect("remote_delete");
        apply.assert();
    }
}
