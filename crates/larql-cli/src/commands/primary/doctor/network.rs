//! Network: whether the hub serves the quick start model, and whether a
//! newer LARQL release exists. Both are skipped under `--offline`.

use std::time::Duration;

use super::report::{Check, Group};
use super::requirements::QUICKSTART_REPO;

/// The hub when `HF_ENDPOINT` does not name another, as hf-hub defaults.
pub(crate) const DEFAULT_HF_ENDPOINT: &str = "https://huggingface.co";
/// The variable naming a hub mirror, honoured as hf-hub honours it.
pub(crate) const HF_ENDPOINT_ENV: &str = "HF_ENDPOINT";
/// GitHub's answer for the newest LARQL release.
pub(crate) const LATEST_RELEASE_API: &str =
    "https://api.github.com/repos/chrishayuk/larql/releases/latest";
/// Where a newer release is downloaded from.
pub(crate) const LATEST_RELEASE_PAGE: &str = "https://github.com/chrishayuk/larql/releases/latest";
/// How long either probe may take: a doctor must not hang on a bad network.
pub(crate) const NETWORK_TIMEOUT: Duration = Duration::from_secs(5);
/// The prefix on LARQL release tags (`v0.2.1`).
const RELEASE_TAG_PREFIX: &str = "v";
/// Shown for a probe `--offline` skipped.
const SKIPPED_OFFLINE: &str = "skipped (--offline)";

/// One network probe's outcome.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Probe<T> {
    Skipped,
    Done(T),
    Failed(String),
}

/// The hub's answer for the quick start model's metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct HubReply {
    pub status: u16,
    pub millis: u128,
}

/// The hub base URL: `HF_ENDPOINT` when set and not blank, else
/// huggingface.co.
pub(crate) fn hub_endpoint(env_value: Option<String>) -> String {
    env_value
        .map(|v| v.trim().trim_end_matches('/').to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| DEFAULT_HF_ENDPOINT.to_string())
}

/// The URL the hub probe reads: the quick start model's metadata.
pub(crate) fn hub_probe_url(endpoint: &str) -> String {
    format!("{endpoint}/api/models/{QUICKSTART_REPO}")
}

/// `major.minor.patch` from `0.2.1` or `v0.2.1`; `None` for anything else
/// (a `vindex-v…` tag, a pre-release).
pub(crate) fn parse_semver(v: &str) -> Option<(u64, u64, u64)> {
    let v = v.trim();
    let v = v.strip_prefix(RELEASE_TAG_PREFIX).unwrap_or(v);
    let mut parts = v.split('.');
    let parsed = (
        parts.next()?.parse().ok()?,
        parts.next()?.parse().ok()?,
        parts.next()?.parse().ok()?,
    );
    parts.next().is_none().then_some(parsed)
}

pub(crate) fn hub_checks(endpoint: &str, probe: &Probe<HubReply>) -> Vec<Check> {
    let name = endpoint.split("://").nth(1).unwrap_or(endpoint).to_string();
    vec![match probe {
        Probe::Skipped => Check::skip(Group::HuggingFace, &name, SKIPPED_OFFLINE),
        Probe::Failed(why) => Check::fail(
            Group::HuggingFace,
            &name,
            format!("unreachable: {why}"),
            "the first `larql run hf://…` downloads from the hub: check the connection or \
             proxy (HTTPS_PROXY). Models already cached run offline (`larql doctor --offline`)",
        ),
        Probe::Done(reply) if (200..300).contains(&reply.status) => Check::ok(
            Group::HuggingFace,
            &name,
            format!("reachable, serves {QUICKSTART_REPO} ({} ms)", reply.millis),
        ),
        Probe::Done(reply) => Check::warn(
            Group::HuggingFace,
            &name,
            format!("answered HTTP {} for {QUICKSTART_REPO}", reply.status),
            "the hub answered but not with the model; check HF_ENDPOINT and any proxy",
        ),
    }]
}

pub(crate) fn release_checks(current: &str, probe: &Probe<String>) -> Vec<Check> {
    const NAME: &str = "Latest release";
    vec![match probe {
        Probe::Skipped => Check::skip(Group::Updates, NAME, SKIPPED_OFFLINE),
        Probe::Failed(why) => Check::skip(Group::Updates, NAME, format!("could not check: {why}")),
        Probe::Done(tag) => match (parse_semver(tag), parse_semver(current)) {
            (Some(latest), Some(ours)) if latest > ours => Check::warn(
                Group::Updates,
                NAME,
                format!("{tag} is available (this is {current})"),
                format!("download it from {LATEST_RELEASE_PAGE}"),
            ),
            (Some(_), Some(_)) => Check::ok(
                Group::Updates,
                NAME,
                format!("up to date (latest is {tag})"),
            ),
            _ => Check::skip(
                Group::Updates,
                NAME,
                format!("cannot compare `{tag}` with {current}"),
            ),
        },
    }]
}
