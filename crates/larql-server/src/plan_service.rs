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
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use larql_vindex::format::vindex3::artifact;
use larql_vindex::format::vindex3::plan::{
    plan_resolved, VerdictCacheKey, PLANNER_SEMANTICS_VERSION,
};
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
/// How this service learns an artifact's immutable commit.
///
/// Injectable because the real one is a network call, and the
/// invariant that matters here — a cache hit performs no staging and
/// no planning — is a statement about work not happening, which can
/// only be asserted by driving both paths deterministically.
pub type CommitResolver = Arc<
    dyn Fn(&[PathBuf]) -> Result<Vec<Option<String>>, larql_vindex::error::VindexError>
        + Send
        + Sync,
>;

pub struct PlanService {
    profile: ServerProfile,
    resolver: CommitResolver,
    cache: Mutex<HashMap<VerdictCacheKey, Value>>,
    cache_capacity: usize,
    permits: Semaphore,
    work: Arc<PlanWork>,
}

/// What this service has actually done, as counts.
///
/// Exists because "the cache is consulted before the work" is a claim
/// about work NOT happening, and the only honest way to assert that is
/// to count the work. A structural check — the lookup appears earlier
/// in the function — would still pass if some staging leaked in ahead
/// of it. Production measurement made the point: before this, a hit
/// answered in 0.9 s and still moved the process high-water mark,
/// because it had already staged and planned.
#[derive(Debug, Default)]
pub struct PlanWork {
    /// Commit probes: one ranged request per artifact, no staging.
    pub commits_resolved: AtomicU64,
    /// Header staging passes — the expensive half.
    pub staged: AtomicU64,
    /// Planner invocations.
    pub planned: AtomicU64,
}

impl PlanWork {
    /// `(commit probes, staging passes, planner invocations)`, for a
    /// caller that wants to assert what this service did — or, more
    /// usefully, what it did not do.
    pub fn snapshot(&self) -> (u64, u64, u64) {
        (
            self.commits_resolved.load(Ordering::Relaxed),
            self.staged.load(Ordering::Relaxed),
            self.planned.load(Ordering::Relaxed),
        )
    }
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
        Self::with_resolver(
            profile,
            max_concurrent,
            cache_capacity,
            Arc::new(artifact::resolve_pinned_commits),
        )
    }

    /// [`Self::with_limits`] with the commit probe stated, so a test
    /// can drive cache hit and cache miss without a hub.
    pub fn with_resolver(
        profile: ServerProfile,
        max_concurrent: usize,
        cache_capacity: usize,
        resolver: CommitResolver,
    ) -> Self {
        Self {
            profile,
            resolver,
            cache: Mutex::new(HashMap::new()),
            cache_capacity,
            permits: Semaphore::new(max_concurrent),
            work: Arc::new(PlanWork::default()),
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

    /// Counts of what this service has actually done. Test surface for
    /// the invariant that a cache hit performs no staging and no
    /// planning.
    pub fn work(&self) -> &PlanWork {
        &self.work
    }

    /// The identity a cached verdict would be filed under, learned
    /// **without staging anything**.
    ///
    /// `None` means "do not consult the cache": a local path, an
    /// unpinned revision name among the artifacts, or a probe that
    /// failed. The last is deliberately not an error — the probe is an
    /// optimisation, and a repo that cannot be reached will fail again
    /// in the planning path below with a better message than a bare
    /// commit lookup can give.
    async fn lookup_key(&self, specs: &[PathBuf]) -> Option<VerdictCacheKey> {
        let owned = specs.to_vec();
        let work = Arc::clone(&self.work);
        let resolver = Arc::clone(&self.resolver);
        let commits = tokio::task::spawn_blocking(move || {
            work.commits_resolved
                .fetch_add(owned.len() as u64, Ordering::Relaxed);
            resolver(&owned)
        })
        .await
        .ok()?
        .ok()?;
        // Every artifact pinned, or no key at all. One unpinned source
        // poisons the whole verdict — a partially immutable verdict is
        // not an immutable verdict.
        let revisions: Vec<String> = commits.into_iter().collect::<Option<Vec<_>>>()?;
        Some(VerdictCacheKey {
            revisions,
            semantics_version: PLANNER_SEMANTICS_VERSION,
        })
    }

    /// Plan `sources`, answering from cache before doing the work that
    /// would produce the answer.
    ///
    /// The order is the point. Keying on the resolved commit means the
    /// identity is only knowable after asking the hub — but asking for
    /// the commit is one ranged request, while producing the verdict is
    /// tens of megabytes of headers and a planner pass. Consulting the
    /// cache after the second step, as this did originally, makes a hit
    /// fast (the hub's bytes are locally cached) while still costing
    /// the full parse and allocation. Measured on the public box: a hit
    /// answered in 0.9 s and still pushed peak RSS up 23 MB.
    ///
    /// The returned document is the plan exactly as `vindex plan
    /// --json` writes it, plus `staging` and a `serving` block.
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

        // ── the cheap half ──
        let probe_key = self.lookup_key(&specs).await;
        if let Some(key) = &probe_key {
            if let Some(hit) = self.cached(key) {
                return Ok(with_serving(hit, true, true, self.profile));
            }
        }

        // ── the expensive half, only on a miss ──
        let work = Arc::clone(&self.work);
        let owned = specs.clone();
        let planned = tokio::task::spawn_blocking(move || plan_specs(&owned, &work))
            .await
            .map_err(|e| ServerError::Internal(format!("plan task failed: {e}")))??;
        let Planned {
            document,
            cache_key,
        } = planned;

        // Filed under the PLAN's key, never the probe's. If the repo
        // moved between the probe and the staging, this verdict belongs
        // to the commit that was actually read.
        if let Some(key) = &cache_key {
            self.store(key.clone(), document.clone());
        }
        Ok(with_serving(
            document,
            false,
            cache_key.is_some(),
            self.profile,
        ))
    }
}

/// Attach the serving block. One place, so a cache hit and a fresh
/// verdict cannot describe themselves differently.
fn with_serving(
    mut document: Value,
    cached: bool,
    cacheable: bool,
    profile: ServerProfile,
) -> Value {
    document["serving"] = serde_json::json!({
        "cached": cached,
        // Says why a verdict was not stored without making the client
        // infer it from the absence of a commit.
        "cacheable": cacheable,
        "profile": profile.as_str(),
    });
    document
}

struct Planned {
    document: Value,
    cache_key: Option<VerdictCacheKey>,
}

/// Resolve and plan, on a blocking thread. Everything that touches the
/// network lives here.
fn plan_specs(specs: &[PathBuf], work: &PlanWork) -> Result<Planned, ServerError> {
    work.staged.fetch_add(1, Ordering::Relaxed);
    let resolved = artifact::resolve_all(specs)
        .map_err(|e| ServerError::BadRequest(format!("cannot read this source: {e}")))?;
    let staging: Vec<Value> = resolved
        .iter()
        .filter_map(artifact::ResolvedArtifact::staging_json)
        .collect();

    work.planned.fetch_add(1, Ordering::Relaxed);
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

    /// A key as the SERVICE would build it — reading the planner's own
    /// constant, not a copy of today's value. Hardcoding `1` here made
    /// every one of these tests fail the moment the planner's semantics
    /// version moved to 5, for a reason that had nothing to do with the
    /// cache.
    fn key(revisions: &[&str]) -> VerdictCacheKey {
        VerdictCacheKey {
            revisions: revisions.iter().map(|r| r.to_string()).collect(),
            semantics_version: PLANNER_SEMANTICS_VERSION,
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
        // Whatever the planner says today, plus one: the point is that a
        // DIFFERENT semantics version is a different verdict, not that
        // any particular number is.
        let newer = VerdictCacheKey {
            revisions: vec!["abc".into()],
            semantics_version: PLANNER_SEMANTICS_VERSION + 1,
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

    // ── cache before work: the invariant, asserted as work NOT done ──

    /// A real checkpoint on disk, so the miss path actually plans.
    fn checkpoint() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        larql_vindex::format::vindex3::fixtures::miniature_glimmer(dir.path());
        dir
    }

    /// A commit probe that answers whatever the test says, without a hub.
    fn resolver(answers: Vec<Option<&'static str>>) -> CommitResolver {
        Arc::new(move |_: &[PathBuf]| {
            Ok(answers
                .iter()
                .map(|a| a.map(str::to_string))
                .collect::<Vec<_>>())
        })
    }

    fn service(answers: Vec<Option<&'static str>>) -> PlanService {
        PlanService::with_resolver(ServerProfile::SingleModel, 2, 8, resolver(answers))
    }

    /// The whole point of the rung. A hit must cost one commit probe
    /// and nothing else — no header staging, no planner pass. Asserted
    /// by counting, because "the lookup happens earlier in the
    /// function" is a structural claim that would still hold if some
    /// staging leaked in ahead of it.
    #[tokio::test]
    async fn a_cache_hit_stages_nothing_and_plans_nothing() {
        let s = service(vec![Some("abc")]);
        s.store(
            key(&["abc"]),
            serde_json::json!({"schema": 4, "stored": true}),
        );
        let dir = checkpoint();

        let before = s.work().snapshot();
        let out = s
            .plan(vec![dir.path().to_string_lossy().into_owned()])
            .await
            .unwrap();
        let after = s.work().snapshot();

        assert_eq!(
            out["stored"],
            serde_json::json!(true),
            "the STORED verdict is served"
        );
        assert_eq!(out["serving"]["cached"], serde_json::json!(true));
        assert_eq!(after.0, before.0 + 1, "exactly one commit probe");
        assert_eq!(after.1, before.1, "no header staging on a hit");
        assert_eq!(after.2, before.2, "no planner invocation on a hit");
    }

    /// The invalidation control. Same request shape, a different
    /// resolved commit: the stored verdict must not answer for it.
    #[tokio::test]
    async fn a_different_resolved_commit_is_a_miss() {
        let s = service(vec![Some("def")]);
        s.store(key(&["abc"]), serde_json::json!({"stored": true}));
        let dir = checkpoint();

        let before = s.work().snapshot();
        let out = s
            .plan(vec![dir.path().to_string_lossy().into_owned()])
            .await
            .unwrap();
        let after = s.work().snapshot();

        assert!(
            out.get("stored").is_none(),
            "a different commit must not reuse the verdict"
        );
        assert_eq!(out["serving"]["cached"], serde_json::json!(false));
        assert_eq!(after.1, before.1 + 1, "a miss stages");
        assert_eq!(after.2, before.2 + 1, "a miss plans");
    }

    /// A local path has no persistent identity, so it is planned
    /// normally and its verdict is never filed.
    #[tokio::test]
    async fn a_local_source_is_planned_and_never_cached() {
        let s = service(vec![None]);
        let dir = checkpoint();
        let out = s
            .plan(vec![dir.path().to_string_lossy().into_owned()])
            .await
            .unwrap();
        assert_eq!(out["serving"]["cached"], serde_json::json!(false));
        assert_eq!(out["serving"]["cacheable"], serde_json::json!(false));
        assert_eq!(s.work().snapshot().2, 1, "it really planned");
        assert!(
            s.cache.lock().unwrap().is_empty(),
            "an unpinned verdict must never be stored"
        );
    }

    /// One unpinned artifact poisons the whole key: a partially
    /// immutable verdict is not an immutable verdict.
    #[tokio::test]
    async fn one_unpinned_artifact_makes_the_whole_plan_uncacheable() {
        let s = service(vec![Some("abc"), None]);
        assert!(
            s.lookup_key(&[PathBuf::from("a"), PathBuf::from("b")])
                .await
                .is_none(),
            "every artifact must be pinned, or there is no key at all"
        );
    }

    /// A probe that fails is not an error — it means "do not consult
    /// the cache", and the planning path below reports the real
    /// problem with a better message.
    #[tokio::test]
    async fn a_failed_commit_probe_falls_through_to_planning() {
        let failing: CommitResolver = Arc::new(|_: &[PathBuf]| {
            Err(larql_vindex::error::VindexError::Parse(
                "hub unreachable".into(),
            ))
        });
        let s = PlanService::with_resolver(ServerProfile::SingleModel, 2, 8, failing);
        assert!(s.lookup_key(&[PathBuf::from("hf://o/r")]).await.is_none());

        let dir = checkpoint();
        let out = s
            .plan(vec![dir.path().to_string_lossy().into_owned()])
            .await
            .unwrap();
        assert_eq!(out["serving"]["cached"], serde_json::json!(false));
        assert_eq!(s.work().snapshot().2, 1, "it planned anyway");
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
