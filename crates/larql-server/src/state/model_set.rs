//! `ModelSet` — the coherent V2+V3 model snapshot `AppState` locks —
//! and `AppState`'s resolution methods (`model()`, `served()`,
//! `is_multi_model()`, and friends). Split out of the top-level
//! `state` module (see `mod.rs`) purely for file size; nothing here
//! changed behavior.

use std::sync::Arc;

use super::{AppState, LoadedModel};

/// One coherent snapshot of every model this process has bound — V2
/// and V3 together, so a reader can never observe one registry
/// updated without the other. `AppState` holds exactly one of these
/// behind a single `RwLock`, deliberately not two independent locks:
/// once load/unload exists, a transition like "V2 model A → idle →
/// V3 model B" must never let a concurrent reader see a torn
/// combination that was never true at any single instant. See
/// `docs/runtime-lifecycle-design.md` §1.
#[derive(Default, Clone)]
pub struct ModelSet {
    /// Loaded VINDEX2 models, keyed by model ID.
    pub models: Vec<Arc<LoadedModel>>,
    /// Loaded VINDEX3 runtimes (VI3-SERVE-1), keyed by model ID. A
    /// separate list, deliberately: a V3 container binds as an
    /// executable program ([`crate::vindex3::V3Model`]), never as a
    /// reconstituted `ModelWeights`, and the two shapes share nothing
    /// below the serving surface.
    pub v3_models: Vec<Arc<crate::vindex3::V3Model>>,
}

impl ModelSet {
    /// Bind a freshly loaded V2 model. Callers are responsible for the
    /// 0↔1 topology invariant
    /// ([`AppState::validate_lifecycle_mutation`]) *before* calling
    /// this — a `ModelSet` has no opinion of its own about how many
    /// entries it should hold, only about keeping the two lists
    /// coherent with each other.
    pub fn insert_v2(&mut self, model: Arc<LoadedModel>) {
        self.models.push(model);
    }

    /// [`Self::insert_v2`]'s V3 counterpart.
    pub fn insert_v3(&mut self, model: Arc<crate::vindex3::V3Model>) {
        self.v3_models.push(model);
    }

    /// Remove the bound model with `id`, from whichever registry it's
    /// actually in, and hand back what was removed so the caller can
    /// finish tearing it down (drain, cache invalidation) or put it
    /// straight back on a failed drain. `None` means nothing matched —
    /// the idempotent-unload case, not an error.
    pub fn remove(&mut self, id: &str) -> Option<ServedModel> {
        if let Some(pos) = self.models.iter().position(|m| m.id == id) {
            return Some(ServedModel::V2(self.models.remove(pos)));
        }
        if let Some(pos) = self.v3_models.iter().position(|m| m.id == id) {
            return Some(ServedModel::V3(self.v3_models.remove(pos)));
        }
        None
    }

    /// Put a previously-[`remove`](Self::remove)d model back — the
    /// fail-closed path when a drain times out. Nothing about the
    /// binding changed, so it goes back exactly where it came from.
    pub fn reinsert(&mut self, model: ServedModel) {
        match model {
            ServedModel::V2(m) => self.models.push(m),
            ServedModel::V3(m) => self.v3_models.push(m),
        }
    }
}

/// One request's resolved model binding: which runtime serves it.
/// Produced only by [`AppState::served`]; the enum exists so the
/// version distinction lives at model resolution, not inside
/// generation code. Owns its `Arc` rather than borrowing from
/// `AppState`: resolution happens under `model_set`'s read guard,
/// which is released before this value is ever used, so it cannot
/// borrow from it.
#[derive(Clone)]
pub enum ServedModel {
    V2(Arc<LoadedModel>),
    V3(Arc<crate::vindex3::V3Model>),
}

impl AppState {
    /// Take the read guard, run `f` against the current snapshot, and
    /// release the guard before returning. Every resolution method
    /// below is one call to this — the one place that actually
    /// touches the lock, so "hold it only long enough to clone an
    /// `Arc`" is a property of this function, not a convention every
    /// call site has to remember.
    fn with_models<T>(&self, f: impl FnOnce(&ModelSet) -> T) -> T {
        let guard = self
            .model_set
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        f(&guard)
    }

    /// A snapshot clone of the current model set — for callers that
    /// need to enumerate everything bound (`GET /v1/models`, boot-time
    /// warmup loops), not resolve one request's model. Cloning is
    /// cheap (each entry is one `Arc` refcount bump); the result is
    /// independent of any later mutation.
    pub fn models_snapshot(&self) -> ModelSet {
        self.with_models(ModelSet::clone)
    }

    /// The first loaded VINDEX2 model, regardless of how many are
    /// loaded. For the handful of grid/shard call sites
    /// (`bootstrap::serve`'s ShardService registration,
    /// `walk_ffn::types::track_model_request`) that always target
    /// "whichever V2 model this shard hosts" — a different question
    /// from the request-scoped resolution `model()`/`served()` answer.
    pub fn first_model(&self) -> Option<Arc<LoadedModel>> {
        self.with_models(|set| set.models.first().cloned())
    }

    /// Get model by ID, or the only model if single-model serving.
    pub fn model(&self, id: Option<&str>) -> Option<Arc<LoadedModel>> {
        self.with_models(|set| match id {
            Some(id) => set.models.iter().find(|m| m.id == id).cloned(),
            None if set.models.len() == 1 => set.models.first().cloned(),
            None => None,
        })
    }

    /// Whether this is multi-model serving.
    pub fn is_multi_model(&self) -> bool {
        self.with_models(|set| set.models.len() + set.v3_models.len() > 1)
    }

    /// Resolve a request's model across BOTH registries in one lock
    /// acquisition. This is the single place the V2/V3 distinction is
    /// decided — routes match the returned binding once, at the top,
    /// and no version check leaks below it.
    pub fn served(&self, id: Option<&str>) -> Option<ServedModel> {
        self.with_models(|set| match id {
            Some(id) => set
                .models
                .iter()
                .find(|m| m.id == id)
                .cloned()
                .map(ServedModel::V2)
                .or_else(|| {
                    set.v3_models
                        .iter()
                        .find(|m| m.id == id)
                        .cloned()
                        .map(ServedModel::V3)
                }),
            None if set.models.len() + set.v3_models.len() == 1 => set
                .models
                .first()
                .cloned()
                .map(ServedModel::V2)
                .or_else(|| set.v3_models.first().cloned().map(ServedModel::V3)),
            None => None,
        })
    }

    /// [`served`](Self::served), or a `NotFound` error.
    pub fn served_or_err(
        &self,
        id: Option<&str>,
    ) -> Result<ServedModel, crate::error::ServerError> {
        self.served(id).ok_or_else(|| {
            let msg = match id {
                Some(mid) => format!("model '{}' not found", mid),
                None => "no model loaded".into(),
            };
            crate::error::ServerError::NotFound(msg)
        })
    }

    pub fn bump_requests(&self) {
        self.requests_served
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Get a model by ID, or return a `NotFound` error.
    ///
    /// Consolidates the 23+ identical `state.model(...).ok_or_else(|| ...)` call
    /// sites scattered across the route handlers.
    pub fn model_or_err(
        &self,
        id: Option<&str>,
    ) -> Result<Arc<LoadedModel>, crate::error::ServerError> {
        self.v2_or_unsupported(id)
    }

    /// Resolve a request's model for a route that serves VINDEX2 only.
    ///
    /// Three answers, and the distinction is the point: nothing bound
    /// is `NotFound` (404); a VINDEX2 model is the model; a VINDEX3
    /// container is `Unsupported` (501) naming the generation. The
    /// third used to fall into the first — `model()` looked only at the
    /// V2 registry — so a server with one V3 container bound answered
    /// "no model loaded" on every V2-only route while `GET /v1/models`
    /// listed it. A loaded-but-unsupported model never masquerades as
    /// absent. This refuses; it does not make the route serve V3.
    pub fn v2_or_unsupported(
        &self,
        id: Option<&str>,
    ) -> Result<Arc<LoadedModel>, crate::error::ServerError> {
        match self.served_or_err(id)? {
            ServedModel::V2(model) => Ok(model),
            ServedModel::V3(model) => Err(crate::error::ServerError::Unsupported(format!(
                "model '{}' is a VINDEX3 container; this operation is served for VINDEX2 \
                 models only (VINDEX3 is served on /v1/models, /v1/runtime, /v1/stats, \
                 /v1/completions, /v1/chat/completions, /v1/responses and the \
                 container-facts routes)",
                model.id
            ))),
        }
    }
}

#[cfg(test)]
mod model_set_tests {
    //! `ModelSet` / `AppState` resolution mechanics — `model()`,
    //! `served()`, `is_multi_model()`, `first_model()`,
    //! `models_snapshot()`. Only V2 fixtures here: a real `V3Model`
    //! needs an opened `Vindex3Runtime` over an on-disk container
    //! (see `tests/test_vindex3_serve.rs`), too heavy for pure
    //! resolution-logic tests — the V2/V3 fallback branch of
    //! `served()` is unchanged logic under a new lock and is already
    //! exercised end-to-end by the V3 integration suites.
    use super::*;
    use std::collections::HashMap;
    use std::path::PathBuf;

    use larql_vindex::ndarray::Array2;
    use larql_vindex::{ExtractLevel, LayerBands, QuantFormat, VectorIndex, VindexConfig};

    use crate::cache::DescribeCache;
    use crate::session::SessionManager;

    fn stub_model(id: &str) -> Arc<LoadedModel> {
        let hidden = 4;
        let index = VectorIndex::new(
            vec![Some(Array2::<f32>::zeros((1, hidden)))],
            vec![None],
            1,
            hidden,
        );
        let patched = larql_vindex::PatchedVindex::new(index);
        let tok_json =
            r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
        let tokenizer = larql_vindex::tokenizers::Tokenizer::from_bytes(tok_json).unwrap();
        Arc::new(LoadedModel {
            id: id.to_string(),
            path: PathBuf::from("/nonexistent"),
            config: VindexConfig {
                version: 2,
                model: id.to_string(),
                family: "test".to_string(),
                source: None,
                checksums: None,
                num_layers: 1,
                hidden_size: hidden,
                intermediate_size: hidden,
                vocab_size: 4,
                embed_scale: 1.0,
                extract_level: ExtractLevel::Browse,
                dtype: larql_vindex::StorageDtype::default(),
                quant: QuantFormat::None,
                layer_bands: Some(LayerBands {
                    syntax: (0, 0),
                    knowledge: (0, 0),
                    output: (0, 0),
                }),
                layers: vec![],
                down_top_k: 1,
                has_model_weights: false,
                model_config: None,
                fp4: None,
                ffn_layout: None,
                bitnet_layout: None,
            },
            patched: Arc::new(tokio::sync::RwLock::new(patched)),
            embeddings: Array2::<f32>::zeros((4, hidden)),
            embed_scale: 1.0,
            tokenizer,
            infer_disabled: true,
            ffn_only: false,
            embed_only: false,
            embed_store: None,
            release_mmap_after_request: false,
            weights: std::sync::OnceLock::new(),
            weights_init: std::sync::Mutex::new(()),
            probe_labels: HashMap::new(),
            ffn_l2_cache: crate::ffn_l2_cache::FfnL2Cache::new(1),
            layer_latency_tracker: Arc::new(crate::metrics::LayerLatencyTracker::new()),
            requests_in_flight: Arc::new(std::sync::atomic::AtomicU32::new(0)),
            requests_total: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            expert_filter: None,
            unit_filter: None,
            moe_remote: None,
            #[cfg(all(feature = "metal-experts", target_os = "macos"))]
            metal_backend: std::sync::OnceLock::new(),
            #[cfg(all(feature = "metal-experts", target_os = "macos"))]
            moe_scratches: std::sync::Mutex::new(HashMap::new()),
            #[cfg(all(feature = "metal-experts", target_os = "macos"))]
            metal_ffn_layer_bufs: std::sync::OnceLock::new(),
        })
    }

    fn state_with(models: Vec<Arc<LoadedModel>>) -> AppState {
        let router_topology = crate::state::RouterTopology::for_boot_count(models.len());
        AppState {
            model_set: std::sync::RwLock::new(ModelSet {
                models,
                v3_models: Vec::new(),
            }),
            router_topology,
            lifecycle: std::sync::Mutex::new(crate::state::LifecycleState::Idle),
            started_at: std::time::Instant::now(),
            requests_served: std::sync::atomic::AtomicU64::new(0),
            api_key: None,
            sessions: SessionManager::new(3600),
            responses: crate::response_store::ResponseStore::new(),
            v3_kv: crate::response_kv::ResponseKvCache::new(
                crate::response_kv::DEFAULT_MAX_ENTRIES,
                crate::response_kv::DEFAULT_TTL_SECS,
            ),
            describe_cache: DescribeCache::new(0),
            infer_timeout: std::time::Duration::from_secs(60),
            runtime: Arc::new(crate::runtime_stats::RuntimeRecorder::new()),
        }
    }

    #[test]
    fn model_set_default_is_empty() {
        let set = ModelSet::default();
        assert!(set.models.is_empty());
        assert!(set.v3_models.is_empty());
    }

    #[test]
    fn insert_v2_and_remove_round_trip() {
        let mut set = ModelSet::default();
        set.insert_v2(stub_model("a"));
        assert_eq!(set.models.len(), 1);

        let removed = set.remove("a").expect("just inserted");
        assert!(matches!(removed, ServedModel::V2(m) if m.id == "a"));
        assert!(set.models.is_empty());
    }

    #[test]
    fn remove_an_unknown_id_is_none_not_a_panic() {
        let mut set = ModelSet::default();
        set.insert_v2(stub_model("a"));
        assert!(set.remove("missing").is_none());
        assert_eq!(set.models.len(), 1, "the real entry must be untouched");
    }

    #[test]
    fn remove_only_takes_the_matching_id_out_of_several() {
        let mut set = ModelSet::default();
        set.insert_v2(stub_model("a"));
        set.insert_v2(stub_model("b"));
        set.remove("a");
        assert_eq!(set.models.len(), 1);
        assert_eq!(set.models[0].id, "b");
    }

    #[test]
    fn reinsert_puts_a_removed_v2_model_back() {
        let mut set = ModelSet::default();
        set.insert_v2(stub_model("a"));
        let removed = set.remove("a").unwrap();
        assert!(set.models.is_empty());

        set.reinsert(removed);
        assert_eq!(set.models.len(), 1);
        assert_eq!(set.models[0].id, "a");
    }

    #[test]
    fn model_none_id_resolves_the_only_model() {
        let state = state_with(vec![stub_model("solo")]);
        assert_eq!(state.model(None).unwrap().id, "solo");
    }

    #[test]
    fn model_none_id_is_ambiguous_with_two_models() {
        let state = state_with(vec![stub_model("a"), stub_model("b")]);
        assert!(state.model(None).is_none());
    }

    #[test]
    fn model_by_id_finds_the_matching_entry_regardless_of_count() {
        let state = state_with(vec![stub_model("a"), stub_model("b")]);
        assert_eq!(state.model(Some("b")).unwrap().id, "b");
        assert!(state.model(Some("missing")).is_none());
    }

    #[test]
    fn model_or_err_reports_the_requested_id_on_miss() {
        // `Arc<LoadedModel>` doesn't implement `Debug` (its
        // `ModelWeights`/`Tokenizer` fields don't), so `Result::unwrap_err`
        // isn't available here — match explicitly instead.
        let state = state_with(vec![stub_model("a")]);
        match state.model_or_err(Some("missing")) {
            Ok(_) => panic!("expected a NotFound error for a missing id"),
            Err(e) => assert!(format!("{e}").contains("missing")),
        }
    }

    #[test]
    fn model_or_err_reports_no_model_loaded_when_empty() {
        let state = state_with(vec![]);
        match state.model_or_err(None) {
            Ok(_) => panic!("expected a NotFound error with no models loaded"),
            Err(e) => assert!(format!("{e}").contains("no model loaded")),
        }
    }

    #[test]
    fn is_multi_model_reflects_total_count_across_v2_only() {
        assert!(!state_with(vec![]).is_multi_model());
        assert!(!state_with(vec![stub_model("a")]).is_multi_model());
        assert!(state_with(vec![stub_model("a"), stub_model("b")]).is_multi_model());
    }

    #[test]
    fn served_resolves_v2_by_id_and_by_sole_binding() {
        let state = state_with(vec![stub_model("solo")]);
        assert!(matches!(state.served(None), Some(ServedModel::V2(m)) if m.id == "solo"));
        assert!(matches!(
            state.served(Some("solo")),
            Some(ServedModel::V2(m)) if m.id == "solo"
        ));
        assert!(state.served(Some("missing")).is_none());
    }

    #[test]
    fn served_or_err_surfaces_not_found() {
        // Same `Debug`-bound reason as `model_or_err`'s tests above —
        // `ServedModel` wraps the same non-`Debug` types.
        let state = state_with(vec![]);
        match state.served_or_err(None) {
            Ok(_) => panic!("expected a NotFound error with no models loaded"),
            Err(e) => assert!(format!("{e}").contains("no model loaded")),
        }
    }

    #[test]
    fn first_model_ignores_id_and_count() {
        let state = state_with(vec![stub_model("a"), stub_model("b")]);
        assert_eq!(state.first_model().unwrap().id, "a");
        assert!(state_with(vec![]).first_model().is_none());
    }

    #[test]
    fn models_snapshot_is_independent_of_further_reads() {
        let state = state_with(vec![stub_model("a"), stub_model("b")]);
        let snapshot = state.models_snapshot();
        assert_eq!(snapshot.models.len(), 2);
        assert!(snapshot.v3_models.is_empty());
        // The snapshot is a plain owned value now — no lock is held
        // by holding onto it, and it doesn't change if the state
        // could later be mutated (nothing to mutate yet in this rung,
        // but the type itself must not borrow from `state`).
        drop(state);
        assert_eq!(snapshot.models[0].id, "a");
    }
}
