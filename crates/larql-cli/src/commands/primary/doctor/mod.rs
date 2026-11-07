//! `larql doctor` — a first-run self-check.
//!
//! Prints a checklist over this build, the machine, the `larql-server`
//! executable, the caches, Hugging Face access and the newest release:
//! `✓` fine, `!` worth fixing, `✗` stops the quick start, `-` not
//! checked, with a one-line fix under every line that is not fine. Exits
//! non-zero only for `✗`: what will stop the README quick start working.
//!
//! Every check is a pure function of facts gathered once by
//! [`live_runtime`], so each decision is tested without the real
//! machine, caches or network. The facts come from the code the other
//! commands use — `serve`'s server lookup, `list`'s cache paths, the
//! backend registry `run --backend` selects from, `larql_vindex`'s token
//! search — so the doctor cannot report a world those commands do not
//! see.

mod auth;
mod build_info;
mod caches;
mod disk;
pub(crate) mod live_runtime;
mod network;
mod platform;
mod report;
mod requirements;
mod server;
mod units;

use clap::Args;

pub use live_runtime::run;

#[derive(Args, Debug)]
pub struct DoctorArgs {
    /// Skip the network checks (Hugging Face reachability, newest release).
    #[arg(long)]
    pub offline: bool,

    /// Print the checklist as JSON.
    #[arg(long)]
    pub json: bool,
}

/// Everything the checks read, gathered once.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Facts {
    pub build: build_info::BuildFacts,
    pub platform: platform::PlatformFacts,
    pub server: server::ServerFacts,
    pub caches: caches::CacheFacts,
    /// Where the HF token comes from; never the token.
    pub token_source: Option<String>,
    pub hub_endpoint: String,
    pub hub: network::Probe<network::HubReply>,
    pub release: network::Probe<String>,
}

/// The checklist for `facts`, in display order.
pub(crate) fn evaluate(facts: &Facts) -> report::Report {
    let mut checks = build_info::checks(&facts.build);
    checks.extend(platform::checks(&facts.platform));
    checks.extend(server::checks(&facts.server));
    checks.extend(caches::checks(&facts.caches));
    checks.extend(auth::checks(facts.token_source.as_deref()));
    checks.extend(network::hub_checks(&facts.hub_endpoint, &facts.hub));
    checks.extend(network::release_checks(
        &facts.build.version,
        &facts.release,
    ));
    report::Report { checks }
}

#[cfg(test)]
mod tests;
