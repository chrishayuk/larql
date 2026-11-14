//! `USE` — point the session at a vindex, model weights, or remote server.

use crate::ast::UseTarget;
use crate::error::LqlError;
use crate::executor::helpers::{dir_size, format_number};
use crate::executor::memit_persist::load_memit_store;
use crate::executor::{Backend, Session};
use crate::relations::RelationClassifier;
use larql_vindex::format::filenames::KNN_STORE_BIN;

impl Session {
    pub(crate) fn exec_use(&mut self, target: &UseTarget) -> Result<Vec<String>, LqlError> {
        match target {
            UseTarget::Vindex(path_str) => {
                // Resolve hf:// paths to local cache
                let path = if larql_vindex::is_hf_path(path_str) {
                    larql_vindex::resolve_hf_vindex(path_str)
                        .map_err(|e| LqlError::exec("HuggingFace download failed", e))?
                } else {
                    let p = super::super::model_resolver::resolve_use_literal(
                        path_str,
                        self.model_resolver.as_ref(),
                    )
                    .map_err(LqlError::Execution)?;
                    if !p.exists() {
                        return Err(LqlError::Execution(format!(
                            "vindex not found: {}",
                            p.display()
                        )));
                    }
                    p
                };

                // The one V2/V3 decision in LQL: bind by the container's
                // own generation marker. Everything after binding
                // consumes the bound runtime's facts and capabilities —
                // no statement executor asks "is this VINDEX3?" again.
                if matches!(
                    larql_vindex::format::generation::detect_generation(&path),
                    Ok(larql_vindex::format::generation::ContainerGeneration::V3)
                ) {
                    return self.bind_v3_session(path);
                }

                let config = larql_vindex::load_vindex_config(&path)
                    .map_err(|e| LqlError::exec("failed to load vindex config", e))?;

                let mut cb = larql_vindex::SilentLoadCallbacks;
                let index = larql_vindex::VectorIndex::load_vindex(&path, &mut cb)
                    .map_err(|e| LqlError::exec("failed to load vindex", e))?;

                let total_features: usize = config.layers.iter().map(|l| l.num_features).sum();

                let relation_classifier = RelationClassifier::from_vindex(&path);

                let rc_status = match &relation_classifier {
                    Some(rc) if rc.has_clusters() => {
                        let probe_info = if rc.num_probe_labels() > 0 {
                            format!(", {} probe-confirmed", rc.num_probe_labels())
                        } else {
                            String::new()
                        };
                        format!(", relations: {} types{}", rc.num_clusters(), probe_info)
                    }
                    _ => String::new(),
                };

                let out = vec![format!(
                    "Using: {} ({} layers, {} features, model: {}{})",
                    path.display(),
                    config.num_layers,
                    format_number(total_features),
                    config.model,
                    rc_status,
                )];

                let router = larql_vindex::RouterIndex::load(&path, &config);
                let mut patched = larql_vindex::PatchedVindex::new(index);

                // Load KNN store if present (Architecture B)
                let knn_path = path.join(KNN_STORE_BIN);
                if knn_path.exists() {
                    match larql_vindex::KnnStore::load(&knn_path) {
                        Ok(store) => {
                            patched.knn_store = store;
                        }
                        Err(e) => {
                            eprintln!("warning: failed to load knn_store.bin: {e}");
                        }
                    }
                }

                // Rehydrate the L2 MEMIT store from disk if a previous
                // session left a snapshot. A corrupt or unreadable
                // snapshot is logged but does NOT block the USE — the
                // session falls back to an empty store.
                let memit_store = match load_memit_store(&path) {
                    Ok(Some(store)) => store,
                    Ok(None) => larql_vindex::MemitStore::new(),
                    Err(e) => {
                        eprintln!("warning: failed to load memit_store.json: {e}");
                        larql_vindex::MemitStore::new()
                    }
                };

                self.backend = Backend::Vindex {
                    path,
                    config,
                    patched,
                    relation_classifier,
                    router,
                    memit_store,
                };
                // Reset any previous patch session
                self.patch_recording = None;
                self.auto_patch = false;
                Ok(out)
            }
            UseTarget::Model {
                id,
                auto_extract: _,
            } => {
                let mut out = Vec::new();
                out.push(format!("Loading model: {id}..."));

                let model_path = larql_inference::resolve_model_path(id)
                    .map_err(|e| LqlError::exec("failed to resolve model", e))?;
                let weights = larql_inference::load_model_dir(&model_path)
                    .map_err(|e| LqlError::exec("failed to load model", e))?;
                let tokenizer = larql_inference::load_tokenizer(&model_path)
                    .map_err(|e| LqlError::exec("failed to load tokenizer", e))?;

                let size_gb = dir_size(&model_path) as f64 / (1024.0 * 1024.0 * 1024.0);
                out.push(format!(
                    "Using model: {} ({} layers, hidden={}, {:.1} GB, live weights)",
                    id, weights.num_layers, weights.hidden_size, size_gb,
                ));
                out.push("Supported: INFER, EXPLAIN INFER, STATS. For WALK/DESCRIBE/SELECT, use EXTRACT first.".into());

                self.backend = Backend::Weight {
                    model_id: id.clone(),
                    weights,
                    tokenizer,
                };
                self.patch_recording = None;
                self.auto_patch = false;
                Ok(out)
            }
            UseTarget::Remote(url) => self.exec_use_remote(url),
        }
    }

    /// Bind a VINDEX3 container as the session backend — the single V3
    /// binding block, shared by `USE` and by `EXTRACT ... FORMAT VINDEX3`'s
    /// auto-bind so the two can never drift.
    pub(crate) fn bind_v3_session(
        &mut self,
        path: std::path::PathBuf,
    ) -> Result<Vec<String>, LqlError> {
        let source = self.representation_source;
        let (runtime, tokenizer, knowledge) = crate::executor::vindex3::bind(&path, source)?;
        let plan_layers = runtime.plan().layers.len();
        let out = vec![
            format!(
                "Using: {} (VINDEX3, model: {}, component {}, {} layers, \
                 execution closed)",
                path.display(),
                runtime.model_name(),
                crate::executor::vindex3::V3_COMPONENT,
                plan_layers,
            ),
            format!(
                "Supported: {}. Tokenizer: {}.",
                crate::executor::vindex3::SUPPORTED,
                if tokenizer.is_some() {
                    "present"
                } else {
                    "absent (token-id capability only)"
                },
            ),
            format!(
                "Execution: representation source {}.",
                crate::executor::vindex3::describe_source(source)
            ),
        ];
        // A compiled container carries its L0 knowledge as
        // `knn_store.bin` (the same file V2 binds) — load it into the
        // overlay's store, exactly as the V2 arm does.
        let mut overlay = larql_vindex::format::vindex3::knowledge::KnowledgeOverlay::default();
        let knn_path = path.join(KNN_STORE_BIN);
        if knn_path.exists() {
            match larql_vindex::KnnStore::load(&knn_path) {
                Ok(store) => overlay.knn_store = store,
                Err(e) => {
                    eprintln!("warning: failed to load knn_store.bin: {e}");
                }
            }
        }
        let bos_token = larql_inference::vindex3::declared_prompt_bos(&path);
        self.backend = Backend::Vindex3 {
            path,
            runtime,
            tokenizer,
            knowledge,
            overlay,
            bos_token,
        };
        self.patch_recording = None;
        self.auto_patch = false;
        Ok(out)
    }
}
