//! **The CLI's continuation selection** (CONTINUATION-PLUGIN-1, C3).
//!
//! The one place a CLI name becomes a continuation identity. `--engine`
//! values are ALIASES for identities; the identity is the authority and is
//! what a run reports. Every CLI path that holds continuation state —
//! generation, replay, benches, probes, measurement arms — selects here,
//! against the plan it will run, so an unsuitable provider is refused
//! before anything executes. Nothing else in the CLI names a built-in
//! provider.
//!
//! [`select_in`] is the resolver; [`select_for`] is it over a fresh
//! [`shipped_continuations`] registry. A run that loaded `--plugin`s
//! passes the registry the loader composed — the shipped providers plus
//! the loaded ones (C6) — and the same resolver answers `--continuation
//! family/vN` from it. There is no plugin-specific selection path.

use larql_kv::shipped_continuations;
use larql_vindex::format::vindex3::opplan::exec::continuation::plan_continuation_geometry;
use larql_vindex::format::vindex3::opplan::exec::continuation_authority::ContinuationConfig;
use larql_vindex::format::vindex3::opplan::exec::continuation_identity::ContinuationIdentity;
use larql_vindex::format::vindex3::opplan::exec::continuation_registry::{
    BoxedContinuation, ContinuationRegistry, SelectedContinuation,
};
use larql_vindex::format::vindex3::opplan::exec::requires_continuation;
use larql_vindex::format::vindex3::opplan::ComponentOpPlan;

/// `--engine` aliases and the identities they name. `no-cache` is not
/// here: it is a replay mode over the default provider, not a provider.
pub(crate) const ENGINE_ALIASES: &[(&str, &str, u32)] =
    &[("row", "row", 1), ("standard", "canonical", 1)];

/// The alias used when no `--engine` is given — the CLI's DECLARED
/// default, stated here and reported in what a run prints; the executor
/// has none.
pub(crate) const DEFAULT_ENGINE: &str = "row";

/// The replay mode's name: accepted by `--engine`, never a provider.
pub(crate) const REPLAY_ENGINE: &str = "no-cache";

/// Whether `engine` is an `--engine` value a VINDEX3 run accepts.
pub(crate) fn is_known_engine(engine: &str) -> bool {
    engine == REPLAY_ENGINE || ENGINE_ALIASES.iter().any(|(alias, ..)| *alias == engine)
}

/// The identity an `--engine` value names; absent or `no-cache` resolve to
/// [`DEFAULT_ENGINE`]'s identity (replay drives the default provider).
pub(crate) fn resolve_engine(engine: Option<&str>) -> Result<ContinuationIdentity, String> {
    let alias = match engine {
        None => DEFAULT_ENGINE,
        Some(name) if name == REPLAY_ENGINE => DEFAULT_ENGINE,
        Some(name) => name,
    };
    ENGINE_ALIASES
        .iter()
        .find(|(name, ..)| *name == alias)
        .map(|(_, family, revision)| ContinuationIdentity::new(*family, *revision))
        .ok_or_else(|| {
            let known: Vec<&str> = ENGINE_ALIASES.iter().map(|(name, ..)| *name).collect();
            format!(
                "--engine `{alias}` names no continuation provider on VINDEX3; known: {}, {REPLAY_ENGINE}",
                known.join(", ")
            )
        })
}

/// What a command line asked of continuation: an `--engine` alias, or an
/// explicit `--continuation` identity with its `--continuation-option`s.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct ContinuationChoice<'a> {
    pub(crate) engine: Option<&'a str>,
    pub(crate) identity: Option<&'a str>,
    pub(crate) options: &'a [String],
}

impl<'a> ContinuationChoice<'a> {
    /// Only an `--engine` alias (or nothing: the declared default).
    pub(crate) fn engine(engine: Option<&'a str>) -> Self {
        Self {
            engine,
            ..Self::default()
        }
    }
}

/// Parse `family/vN` — the form a [`ContinuationIdentity`] displays as.
pub(crate) fn parse_continuation_identity(spec: &str) -> Result<ContinuationIdentity, String> {
    let malformed = || format!("--continuation `{spec}` is not `family/vN`");
    let (family, revision) = spec.rsplit_once('/').ok_or_else(malformed)?;
    let revision: u32 = revision
        .strip_prefix('v')
        .and_then(|n| n.parse().ok())
        .ok_or_else(malformed)?;
    let identity = ContinuationIdentity::new(family, revision);
    identity.validate().map_err(|e| e.to_string())?;
    Ok(identity)
}

/// Resolve `choice` for `plan` in `registry`, refusing before anything
/// runs: `--continuation` together with `--engine`, a malformed identity
/// or option, an identity the registry does not hold, a configuration its
/// provider refuses, or a plan layer it cannot hold.
pub(crate) fn select_in(
    registry: &ContinuationRegistry,
    plan: &ComponentOpPlan,
    choice: &ContinuationChoice<'_>,
) -> Result<SelectedContinuation, String> {
    let identity = match (choice.identity, choice.engine) {
        (Some(spec), Some(engine)) => {
            return Err(format!(
                "--continuation `{spec}` and --engine `{engine}` both name a continuation                  provider; give one"
            ))
        }
        (Some(spec), None) => parse_continuation_identity(spec)?,
        (None, engine) => resolve_engine(engine)?,
    };
    let config = ContinuationConfig::parse(choice.options)?;
    let geometry = plan_continuation_geometry(plan)?;
    registry
        .select(&identity, &config, &geometry)
        .map_err(|e| e.to_string())
}

/// Select the provider `engine` names for `plan`, from a fresh shipped
/// registry, refusing before anything runs.
pub(crate) fn select_for(
    plan: &ComponentOpPlan,
    engine: Option<&str>,
) -> Result<SelectedContinuation, String> {
    select_in(
        &shipped_continuations(),
        plan,
        &ContinuationChoice::engine(engine),
    )
}

/// State for a one-shot forward over `plan`: a fresh provider from the
/// default selection when the executor says the plan needs one
/// ([`requires_continuation`]), none for a wholly-softmax plan — which then
/// materialises no rows, exactly as before.
pub(crate) fn one_shot_state(plan: &ComponentOpPlan) -> Result<Option<BoxedContinuation>, String> {
    if !requires_continuation(plan) {
        return Ok(None);
    }
    Ok(Some(select_for(plan, None)?.build()))
}

#[cfg(test)]
#[path = "continuation_tests.rs"]
mod tests;
