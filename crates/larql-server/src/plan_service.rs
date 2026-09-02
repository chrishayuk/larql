//! `POST /v1/plan` — what VINDEX understands about a source, before
//! anything has been encoded.
//!
//! The Explorer's "enter a model" path: a visitor types
//! `hf://Qwen/Qwen3-4B` and gets an architecture-support verdict —
//! identity, resolved commit, operators, findings, admissibility —
//! without a checkpoint being downloaded or a container being built.
//! Answering costs headers, not weights.
//!
//! Three things this module owns, and one it deliberately does not:
//!
//! - **Policy.** Which source forms this profile will plan, asked
//!   through [`crate::capabilities::plans_source`] — the same function
//!   `GET /v1/capabilities` advertises through, so a client is never
//!   refused for something the report said was available.
//! - **Cost.** A plan is a network fetch (~11 MB of headers for a
//!   0.6B model, ~39 MB for a 328 GB one) and a public endpoint that
//!   performs one per request on demand is trivially abusable. Plans
//!   are bounded in flight and refused, not queued, past the bound.
//! - **Cache.** Keyed on [`VerdictCacheKey`] — every artifact's
//!   immutable commit plus the planner's semantics version — and
//!   therefore only for plans where every artifact is pinned. An
//!   unpinned verdict is served and never stored: caching it would
//!   answer tomorrow's question with today's facts.
//!
//! What it does not own is the verdict. That is
//! `larql_vindex::format::vindex3::plan::plan_resolved`, the same
//! function behind `vindex plan` and `larql vindex3 plan` — so the
//! answer cannot depend on which front door asked.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;

use larql_vindex::format::vindex3::artifact;
use larql_vindex::format::vindex3::plan::{plan_resolved, VerdictCacheKey};
use serde_json::Value;
use tokio::sync::Semaphore;
use tracing::info;

use crate::capabilities::{plans_source, ServerProfile, SourceScheme};
use crate::error::ServerError;

/// A system plan spans artifacts, but a request that names dozens is
/// asking this server to make dozens of round trips on its behalf.
pub const MAX_SOURCES_PER_PLAN: usize = 8;

/// Distinct pinned verdicts retained. Small on purpose: this is a
/// courtesy for repeated views of the same model, not a store.
const CACHE_CAPACITY: usize = 256;

/// Plans in flight at once. Two, because each one is bounded by a
/// remote's response time rather than by this machine, and a queue of
/// them is the cheapest denial of service against a public endpoint.
pub const MAX_CONCURRENT_PLANS: usize = 2;

/// The planning surface for one server.
pub struct PlanService {
    profile: ServerProfile,
    cache: Mutex<HashMap<VerdictCacheKey, Value>>,
    cache_capacity: usize,
    permits: Semaphore,
}

impl PlanService {
    pub fn new(profile: ServerProfile) -> Self {
        Self::with_limits(profile, MAX_CONCURRENT_PLANS, CACHE_CAPACITY)
    }

    /// [`Self::new`] with the bounds stated, so a test can drive the
    /// refusal path deterministically (`max_concurrent = 0` refuses
    /// every plan) instead of racing two real fetches and hoping.
    pub fn with_limits(
        profile: ServerProfile,
        max_concurrent: usize,
        cache_capacity: usize,
    ) -> Self {
        Self {
            profile,
            cache: Mutex::new(HashMap::new()),
            cache_capacity,
            permits: Semaphore::new(max_concurrent),
        }
    }

    /// How this server would classify `spec` — the *plan* resolver's
    /// own branch ([`artifact::is_remote_spec`]), not the load
    /// resolver's. The two verbs take different objects (a checkpoint
    /// versus an encoded container), so they must not share a
    /// classifier just because both understand the string `hf://`.
    pub fn scheme_of(spec: &std::path::Path) -> SourceScheme {
        if artifact::is_remote_spec(spec) {
            SourceScheme::Hf
        } else {
            SourceScheme::Local
        }
    }

    /// Refuse the source forms this profile will not plan, naming the
    /// profile — a client that reads `/v1/capabilities` first should
    /// never see this, and one that does see it should know why.
    fn admit(&self, specs: &[PathBuf]) -> Result<(), ServerError> {
        for spec in specs {
            let scheme = Self::scheme_of(spec);
            if !plans_source(self.profile, scheme) {
                return Err(ServerError::Refused(format!(
                    "this server serves the {} profile, which plans hf:// sources only — \
                     a local path is not planned here. GET /v1/capabilities reports this \
                     as sources.plan.local = false.",
                    self.profile.as_str()
                )));
            }
        }
        Ok(())
    }

    fn validate(&self, sources: &[String]) -> Result<Vec<PathBuf>, ServerError> {
        if sources.is_empty() {
            return Err(ServerError::BadRequest(
                "sources must name at least one artifact".into(),
            ));
        }
        if sources.len() > MAX_SOURCES_PER_PLAN {
            return Err(ServerError::BadRequest(format!(
                "{} sources requested; this server plans at most {MAX_SOURCES_PER_PLAN} \
                 artifacts in one system plan",
                sources.len()
            )));
        }
        let specs: Vec<PathBuf> = sources.iter().map(PathBuf::from).collect();
        self.admit(&specs)?;
        Ok(specs)
    }

    fn cached(&self, key: &VerdictCacheKey) -> Option<Value> {
        self.cache
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .get(key)
            .cloned()
    }

    fn store(&self, key: VerdictCacheKey, value: Value) {
        let mut cache = self.cache.lock().unwrap_or_else(|p| p.into_inner());
        // At capacity, keep what is there rather than evicting: this
        // is a courtesy cache, and a wrong eviction policy is worse
        // than a cold one.
        if cache.len() < self.cache_capacity {
            cache.insert(key, value);
        }
    }

    /// Serve a verdict: a held pinned one from cache, otherwise store
    /// this one and serve it.
    ///
    /// Split out of [`Self::plan`] because the branch that matters —
    /// a pinned verdict coming back from cache instead of being
    /// recomputed — can only be reached with an immutable commit, and
    /// every source that has one is a network fetch. As its own
    /// method it is reachable from a unit test, so the cache rule is
    /// checked rather than assumed.
    fn serve(&self, document: Value, cache_key: Option<&VerdictCacheKey>) -> (Value, bool) {
        let Some(key) = cache_key else {
            // Unpinned: served, never stored. A cache that held this
            // would answer tomorrow's question with today's facts.
            return (document, false);
        };
        match self.cached(key) {
            Some(hit) => (hit, true),
            None => {
                self.store(key.clone(), document.clone());
                (document, false)
            }
        }
    }

    /// Plan `sources`, serving a pinned verdict from cache when one is
    /// held.
    ///
    /// The returned document is the plan exactly as `vindex plan
    /// --json` writes it, plus `staging` and a `serving` block — so a
    /// client can read a server's answer and a CLI's answer with the
    /// same parser.
    pub async fn plan(&self, sources: Vec<String>) -> Result<Value, ServerError> {
        let specs = self.validate(&sources)?;

        // Bound the work before doing any of it. `try_acquire` rather
        // than `acquire`: a caller waiting behind a slow remote fetch
        // learns nothing useful, and a queue is the thing being
        // defended against.
        let _permit = self.permits.try_acquire().map_err(|_| {
            ServerError::Conflict(format!(
                "{MAX_CONCURRENT_PLANS} plans already in flight — planning reads a remote \
                 checkpoint's headers, so this server runs a bounded number at once. Retry."
            ))
        })?;

        let planned = tokio::task::spawn_blocking(move || plan_specs(&specs))
            .await
            .map_err(|e| ServerError::Internal(format!("plan task failed: {e}")))??;

        let Planned {
            document,
            cache_key,
        } = planned;

        let (mut document, served_from_cache) = self.serve(document, cache_key.as_ref());

        document["serving"] = serde_json::json!({
            "cached": served_from_cache,
            // Says why a verdict was not stored without making the
            // client infer it from the absence of a commit.
            "cacheable": cache_key.is_some(),
            "profile": self.profile.as_str(),
        });
        Ok(document)
    }
}

struct Planned {
    document: Value,
    cache_key: Option<VerdictCacheKey>,
}

/// Resolve and plan, on a blocking thread. Everything that touches the
/// network lives here.
fn plan_specs(specs: &[PathBuf]) -> Result<Planned, ServerError> {
    let resolved = artifact::resolve_all(specs)
        .map_err(|e| ServerError::BadRequest(format!("cannot read this source: {e}")))?;
    let staging: Vec<Value> = resolved.iter().filter_map(artifact::staging_json).collect();

    let plan = plan_resolved(specs, resolved)
        .map_err(|e| ServerError::BadRequest(format!("cannot plan this source: {e}")))?;
    let cache_key = plan.cache_key();
    info!(
        artifacts = plan.artifacts.len(),
        blocking = plan.summary.blocking,
        admissible = plan.admissible,
        cacheable = cache_key.is_some(),
        "planned",
    );

    let mut document = serde_json::to_value(&plan)
        .map_err(|e| ServerError::Internal(format!("plan is not serialisable: {e}")))?;
    if !staging.is_empty() {
        document["staging"] = Value::Array(staging);
    }
    Ok(Planned {
        document,
        cache_key,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(revisions: &[&str]) -> VerdictCacheKey {
        VerdictCacheKey {
            revisions: revisions.iter().map(|r| r.to_string()).collect(),
            semantics_version: 1,
        }
    }

    #[test]
    fn a_stored_verdict_comes_back_under_its_own_key() {
        let s = PlanService::new(ServerProfile::SingleModel);
        s.store(key(&["abc"]), serde_json::json!({"schema": 4}));
        assert_eq!(
            s.cached(&key(&["abc"])),
            Some(serde_json::json!({"schema": 4}))
        );
        assert_eq!(s.cached(&key(&["def"])), None);
    }

    /// Two artifacts at the same commits but a different semantics
    /// version are different verdicts — the planner changed its mind,
    /// not the model.
    #[test]
    fn semantics_version_is_part_of_the_identity() {
        let s = PlanService::new(ServerProfile::SingleModel);
        s.store(key(&["abc"]), serde_json::json!({"v": 1}));
        let newer = VerdictCacheKey {
            revisions: vec!["abc".into()],
            semantics_version: 2,
        };
        assert_eq!(s.cached(&newer), None);
    }

    #[test]
    fn a_full_cache_keeps_what_it_has() {
        let s = PlanService::with_limits(ServerProfile::SingleModel, 1, 1);
        s.store(key(&["first"]), serde_json::json!(1));
        s.store(key(&["second"]), serde_json::json!(2));
        assert_eq!(s.cached(&key(&["first"])), Some(serde_json::json!(1)));
        assert_eq!(s.cached(&key(&["second"])), None);
    }

    #[test]
    fn the_plan_resolvers_own_classifier_decides_the_scheme() {
        assert_eq!(
            PlanService::scheme_of(std::path::Path::new("hf://owner/repo")),
            SourceScheme::Hf
        );
        assert_eq!(
            PlanService::scheme_of(std::path::Path::new("/var/lib/checkpoint")),
            SourceScheme::Local
        );
    }

    #[test]
    fn an_unpinned_verdict_is_served_and_never_stored() {
        let s = PlanService::new(ServerProfile::SingleModel);
        let (doc, cached) = s.serve(serde_json::json!({"v": 1}), None);
        assert_eq!(doc, serde_json::json!({"v": 1}));
        assert!(!cached);
        assert!(s.cache.lock().unwrap().is_empty(), "nothing may be stored");
    }

    #[test]
    fn a_pinned_verdict_is_stored_on_the_first_ask_and_served_on_the_second() {
        let s = PlanService::new(ServerProfile::SingleModel);
        let k = key(&["abc"]);

        let (first, cached) = s.serve(serde_json::json!({"v": 1}), Some(&k));
        assert_eq!(first, serde_json::json!({"v": 1}));
        assert!(!cached, "the first ask computed it");

        // A second, DIFFERENT document under the same key must lose to
        // the stored one — that is what "cached" has to mean.
        let (second, cached) = s.serve(serde_json::json!({"v": 2}), Some(&k));
        assert_eq!(second, serde_json::json!({"v": 1}));
        assert!(cached);
    }

    #[tokio::test]
    async fn planning_is_refused_when_every_permit_is_taken() {
        let s = PlanService::with_limits(ServerProfile::SingleModel, 0, 8);
        let err = s.plan(vec!["/nonexistent".into()]).await.unwrap_err();
        assert!(
            matches!(err, ServerError::Conflict(_)),
            "a bounded service must refuse rather than queue: {err:?}"
        );
    }

    #[tokio::test]
    async fn an_empty_or_oversized_request_is_refused_before_any_fetch() {
        let s = PlanService::new(ServerProfile::SingleModel);
        assert!(matches!(
            s.plan(Vec::new()).await.unwrap_err(),
            ServerError::BadRequest(_)
        ));
        let many: Vec<String> = (0..MAX_SOURCES_PER_PLAN + 1)
            .map(|i| format!("hf://owner/repo{i}"))
            .collect();
        assert!(matches!(
            s.plan(many).await.unwrap_err(),
            ServerError::BadRequest(_)
        ));
    }
}
