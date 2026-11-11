//! Write-side remote forwarders (INSERT, DELETE, UPDATE) plus
//! client-side local patch management (APPLY / SHOW / REMOVE PATCH).

use crate::ast::{Assignment, Condition};
use crate::error::LqlError;
use crate::executor::mutation::update_set::UpdateSet;
use crate::executor::{Backend, Session};

use super::{require_layer_feature, ENDPOINT_INSERT, ENDPOINT_PATCHES_APPLY};

/// Default `confidence` for INSERT/UPDATE when the user doesn't
/// pass one. 0.9 lands well above the retrieval floor without
/// dominating template-matched siblings.
const REMOTE_DEFAULT_CONFIDENCE: f32 = 0.9;

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

        let result = self.remote_post_json(ENDPOINT_INSERT, &request, true)?;

        let inserted = result["inserted"].as_u64().unwrap_or(0);
        let mode = result["mode"].as_str().unwrap_or("unknown");
        let ms = result["latency_ms"].as_f64().unwrap_or(0.0);

        Ok(vec![
            format!("Inserted: {entity} —[{relation}]→ {target} ({inserted} layers, mode: {mode})"),
            format!("{ms:.0}ms (remote)"),
        ])
    }

    pub(crate) fn remote_delete(&self, conditions: &[Condition]) -> Result<Vec<String>, LqlError> {
        let (layer, feature) = require_layer_feature(conditions, "DELETE")?;

        let ops = vec![larql_vindex::PatchOp::Delete {
            layer,
            feature,
            reason: Some("remote DELETE".into()),
        }];

        let patch = larql_vindex::VindexPatch {
            version: 1,
            base_model: String::new(),
            base_checksum: None,
            created_at: String::new(),
            description: Some(format!("DELETE L{layer} F{feature}")),
            author: None,
            tags: vec![],
            operations: ops,
        };

        let _ = self.remote_post_json(
            ENDPOINT_PATCHES_APPLY,
            &serde_json::json!({"patch": patch}),
            false,
        )?;

        Ok(vec![format!(
            "Deleted: L{layer} F{feature} → remote server"
        )])
    }

    pub(crate) fn remote_update(
        &self,
        set: &[Assignment],
        conditions: &[Condition],
    ) -> Result<Vec<String>, LqlError> {
        let (layer, feature) = require_layer_feature(conditions, "UPDATE")?;
        // The SET side is checked by the same rule as a local UPDATE, before
        // anything is sent.
        let UpdateSet { target, confidence } = UpdateSet::parse(set)?;

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

        let patch = larql_vindex::VindexPatch {
            version: 1,
            base_model: String::new(),
            base_checksum: None,
            created_at: String::new(),
            description: Some(format!("UPDATE L{layer} F{feature}")),
            author: None,
            tags: vec![],
            operations: vec![op],
        };

        let _ = self.remote_post_json(
            ENDPOINT_PATCHES_APPLY,
            &serde_json::json!({"patch": patch}),
            false,
        )?;

        let desc = target
            .as_deref()
            .map(|t| format!(" target={t}"))
            .unwrap_or_default();
        Ok(vec![format!(
            "Updated: L{layer} F{feature}{desc} → remote server"
        )])
    }

    // ── Local patch management (client-side overlay) ─────────────

    pub(crate) fn remote_apply_local_patch(&mut self, path: &str) -> Result<Vec<String>, LqlError> {
        let patch_path = std::path::PathBuf::from(path);
        if !patch_path.exists() {
            return Err(LqlError::Execution(format!("patch not found: {path}")));
        }

        let patch = larql_vindex::VindexPatch::load(&patch_path)
            .map_err(|e| LqlError::exec("failed to load patch", e))?;

        let (ins, upd, del) = patch.counts();
        let total = patch.len();

        match &mut self.backend {
            Backend::Remote { local_patches, .. } => {
                local_patches.push(patch);
                Ok(vec![format!(
                    "Applied locally: {path} ({total} ops: {ins} ins, {upd} upd, {del} del)\n\
                     Patch stays client-side — server never sees it."
                )])
            }
            _ => Err(LqlError::Execution(
                "not connected to a remote server".into(),
            )),
        }
    }

    pub(crate) fn remote_show_patches(&self) -> Result<Vec<String>, LqlError> {
        let local_patches = match &self.backend {
            Backend::Remote { local_patches, .. } => local_patches,
            _ => {
                return Err(LqlError::Execution(
                    "not connected to a remote server".into(),
                ))
            }
        };

        let mut out = Vec::new();
        if local_patches.is_empty() {
            out.push("  (no local patches)".into());
        } else {
            out.push("Local patches (client-side only):".into());
            for (i, patch) in local_patches.iter().enumerate() {
                let (ins, upd, del) = patch.counts();
                let name = patch.description.as_deref().unwrap_or("(unnamed)");
                out.push(format!(
                    "  {}. {:<40} {} ops ({} ins, {} upd, {} del)",
                    i + 1,
                    name,
                    patch.len(),
                    ins,
                    upd,
                    del,
                ));
            }
        }
        Ok(out)
    }

    pub(crate) fn remote_remove_local_patch(
        &mut self,
        name: &str,
    ) -> Result<Vec<String>, LqlError> {
        let local_patches = match &mut self.backend {
            Backend::Remote { local_patches, .. } => local_patches,
            _ => {
                return Err(LqlError::Execution(
                    "not connected to a remote server".into(),
                ))
            }
        };

        let pos = local_patches
            .iter()
            .position(|p| p.description.as_deref().unwrap_or("unnamed") == name);

        match pos {
            Some(i) => {
                local_patches.remove(i);
                Ok(vec![format!("Removed local patch: {name}")])
            }
            None => Err(LqlError::Execution(format!(
                "local patch not found: {name}"
            ))),
        }
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
        assert!(joined.contains("remote server"));
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

    // ── Local-patch management (no HTTP needed) ─────────────────────

    fn write_temp_patch(name: &str, description: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "larql_remote_local_patch_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join(format!("{name}.vlp"));
        let patch = larql_vindex::VindexPatch {
            version: 1,
            base_model: String::new(),
            base_checksum: None,
            created_at: String::new(),
            description: Some(description.into()),
            author: None,
            tags: vec![],
            operations: vec![larql_vindex::PatchOp::Delete {
                layer: 0,
                feature: 0,
                reason: Some("test".into()),
            }],
        };
        patch.save(&path).expect("save .vlp");
        path
    }

    #[test]
    fn remote_apply_local_patch_records_patch_in_session() {
        let mut server = mockito::Server::new();
        let _stats = server
            .mock("GET", ENDPOINT_STATS)
            .with_status(200)
            .with_body(stats_body())
            .create();
        let mut session = connect(&server.url());
        let patch_path = write_temp_patch("local_apply", "demo-patch");
        let out = session
            .remote_apply_local_patch(patch_path.to_str().unwrap())
            .expect("apply local patch");
        let joined = out.join("\n");
        assert!(joined.contains("Applied locally"));
        assert!(joined.contains("client-side"));

        // session.local_patches should now have one entry.
        if let Backend::Remote { local_patches, .. } = &session.backend {
            assert_eq!(local_patches.len(), 1);
        } else {
            panic!("expected Backend::Remote");
        }
        let _ = std::fs::remove_file(patch_path);
    }

    #[test]
    fn remote_apply_local_patch_errors_on_missing_file() {
        let mut server = mockito::Server::new();
        let _stats = server
            .mock("GET", ENDPOINT_STATS)
            .with_status(200)
            .with_body(stats_body())
            .create();
        let mut session = connect(&server.url());
        let err = session
            .remote_apply_local_patch("/tmp/no_such_patch_xyz.vlp")
            .unwrap_err();
        assert!(err.to_string().contains("patch not found"));
    }

    #[test]
    fn remote_apply_local_patch_errors_when_not_remote() {
        // Save a valid patch but use a non-Remote session.
        let patch_path = write_temp_patch("local_apply_nonremote", "x");
        let mut session = Session::new();
        let err = session
            .remote_apply_local_patch(patch_path.to_str().unwrap())
            .unwrap_err();
        assert!(err.to_string().contains("not connected to a remote"));
        let _ = std::fs::remove_file(patch_path);
    }

    #[test]
    fn remote_show_patches_empty_list() {
        let mut server = mockito::Server::new();
        let _stats = server
            .mock("GET", ENDPOINT_STATS)
            .with_status(200)
            .with_body(stats_body())
            .create();
        let session = connect(&server.url());
        let out = session.remote_show_patches().expect("show patches");
        let joined = out.join("\n");
        assert!(joined.contains("(no local patches)"));
    }

    #[test]
    fn remote_show_patches_lists_each_entry() {
        let mut server = mockito::Server::new();
        let _stats = server
            .mock("GET", ENDPOINT_STATS)
            .with_status(200)
            .with_body(stats_body())
            .create();
        let mut session = connect(&server.url());
        let p1 = write_temp_patch("show_a", "patch-A");
        let p2 = write_temp_patch("show_b", "patch-B");
        let _ = session
            .remote_apply_local_patch(p1.to_str().unwrap())
            .unwrap();
        let _ = session
            .remote_apply_local_patch(p2.to_str().unwrap())
            .unwrap();

        let out = session.remote_show_patches().expect("show patches");
        let joined = out.join("\n");
        assert!(joined.contains("patch-A"));
        assert!(joined.contains("patch-B"));
        let _ = std::fs::remove_file(p1);
        let _ = std::fs::remove_file(p2);
    }

    #[test]
    fn remote_show_patches_errors_when_not_remote() {
        let session = Session::new();
        let err = session.remote_show_patches().unwrap_err();
        assert!(err.to_string().contains("not connected"));
    }

    #[test]
    fn remote_remove_local_patch_by_name() {
        let mut server = mockito::Server::new();
        let _stats = server
            .mock("GET", ENDPOINT_STATS)
            .with_status(200)
            .with_body(stats_body())
            .create();
        let mut session = connect(&server.url());
        let p = write_temp_patch("remove_target", "deletable");
        let _ = session
            .remote_apply_local_patch(p.to_str().unwrap())
            .unwrap();

        let out = session
            .remote_remove_local_patch("deletable")
            .expect("remove patch");
        let joined = out.join("\n");
        assert!(joined.contains("Removed local patch"));
        if let Backend::Remote { local_patches, .. } = &session.backend {
            assert!(local_patches.is_empty());
        }
        let _ = std::fs::remove_file(p);
    }

    #[test]
    fn remote_remove_local_patch_not_found_errors() {
        let mut server = mockito::Server::new();
        let _stats = server
            .mock("GET", ENDPOINT_STATS)
            .with_status(200)
            .with_body(stats_body())
            .create();
        let mut session = connect(&server.url());
        let err = session
            .remote_remove_local_patch("does-not-exist")
            .unwrap_err();
        assert!(err.to_string().contains("not found"));
    }

    #[test]
    fn remote_remove_local_patch_errors_when_not_remote() {
        let mut session = Session::new();
        let err = session.remote_remove_local_patch("anything").unwrap_err();
        assert!(err.to_string().contains("not connected"));
    }
}
